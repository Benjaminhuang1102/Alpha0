"""Soft Actor-Critic (SAC) training for Alpha0.

Why SAC:
  - Off-policy + entropy regularisation → sample efficient + exploration
  - Continuous action space (portfolio weights are continuous)
  - More stable than PPO for continuous control tasks
  - Auto-tunes the entropy coefficient (alpha), avoiding a sensitive
    hyperparameter

Algorithm overview (Haarnoja et al., 2018, SAC v2 with auto-alpha):

  Replay buffer R stores transitions (s, a, r, s', done).

  Each update step:
    1. Sample a mini-batch from R.
    2. Critic update (TD backup):
         y = r + gamma * (min(Q1', Q2')(s', a') - alpha * log_pi(a'|s'))
         where a' ~ pi(s') (next action sampled from current policy)
         L_critic = MSE(Q(s,a), y)  for both critics
    3. Actor update (maximise expected Q - entropy):
         L_actor = E[alpha * log_pi(a|s) - min(Q1, Q2)(s, a)]
         where a ~ pi(s) (action sampled, not from buffer)
    4. Auto-alpha update:
         L_alpha = E[-alpha * (log_pi(a|s) + target_entropy)]
         target_entropy = -n_actions  (heuristic)
    5. Soft update target critics: theta' = tau * theta + (1-tau) * theta'
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from alpha0.policy.allocation_head import AlphaPolicy
from alpha0.policy.transformer import MarketEncoder
from alpha0.training.augmentation import ObservationAugmenter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────

@dataclass
class Transition:
    obs:       np.ndarray   # (T, N, F)
    portfolio: np.ndarray   # (N+1,)
    action:    np.ndarray   # (N+1,)
    reward:    float
    next_obs:       np.ndarray
    next_portfolio: np.ndarray
    done:      bool


class ReplayBuffer:
    """Fixed-capacity circular replay buffer.

    Parameters
    ----------
    capacity:
        Maximum number of transitions stored.
    """

    def __init__(self, capacity: int) -> None:
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self._buf.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)

    def is_ready(self, min_size: int) -> bool:
        return len(self._buf) >= min_size


def _batch_to_tensors(
    batch: list[Transition],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Convert a list of Transition objects to batched tensors."""
    obs       = torch.tensor(np.stack([t.obs       for t in batch]), dtype=torch.float32, device=device)
    portfolio = torch.tensor(np.stack([t.portfolio for t in batch]), dtype=torch.float32, device=device)
    action    = torch.tensor(np.stack([t.action    for t in batch]), dtype=torch.float32, device=device)
    reward    = torch.tensor([t.reward              for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
    next_obs  = torch.tensor(np.stack([t.next_obs  for t in batch]), dtype=torch.float32, device=device)
    next_port = torch.tensor(np.stack([t.next_portfolio for t in batch]), dtype=torch.float32, device=device)
    done      = torch.tensor([float(t.done)         for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
    return obs, portfolio, action, reward, next_obs, next_port, done


# ─────────────────────────────────────────────────────────────
# SAC Trainer
# ─────────────────────────────────────────────────────────────

@dataclass
class SACConfig:
    """Hyperparameters for SAC training.  All sourced from config/default.yaml."""
    # Architecture
    d_model:          int   = 128
    n_temporal_layers:int   = 3
    n_cross_layers:   int   = 1
    n_heads:          int   = 4
    d_ff:             int   = 256
    actor_hidden:     int   = 256
    critic_hidden:    int   = 256
    dropout:          float = 0.2

    # SAC algorithm
    batch_size:       int   = 256
    replay_capacity:  int   = 100_000
    learning_rate:    float = 3e-4
    gamma:            float = 0.99
    tau:              float = 0.005
    alpha_init:       float = 0.2      # entropy coefficient (auto-tuned)
    auto_alpha:       bool  = True
    weight_decay:     float = 1e-4
    max_grad_norm:    float = 1.0
    warmup_steps:     int   = 1_000    # fill buffer before training starts
    n_updates_per_step: int = 1

    # Environment
    n_assets:         int   = 100
    n_features:       int   = 14
    lookback:         int   = 60

    # Augmentation
    obs_noise_std:    float = 0.01

    # Misc
    device:           str   = "cpu"
    seed:             int   = 0

    @classmethod
    def from_cfg(cls, cfg: dict) -> "SACConfig":
        """Build a SACConfig from the full config dict."""
        t = cfg["training"]
        e = cfg["env"]
        d = cfg["data"]
        return cls(
            d_model           = t.get("d_model",           128),
            n_temporal_layers = t.get("n_temporal_layers", 3),
            n_cross_layers    = t.get("n_cross_layers",    1),
            n_heads           = t.get("n_heads",           4),
            d_ff              = t.get("d_ff",              256),
            actor_hidden      = t.get("actor_hidden",      256),
            critic_hidden     = t.get("critic_hidden",     256),
            dropout           = t.get("dropout",           0.2),
            batch_size        = t.get("batch_size",        256),
            replay_capacity   = t.get("replay_buffer_size",100_000),
            learning_rate     = t.get("learning_rate",     3e-4),
            gamma             = t.get("gamma",             0.99),
            tau               = t.get("tau",               0.005),
            alpha_init        = t.get("alpha",             0.2),
            auto_alpha        = True,
            weight_decay      = t.get("weight_decay",      1e-4),
            max_grad_norm     = t.get("max_grad_norm",     1.0),
            warmup_steps      = t.get("warmup_steps",      1_000),
            n_updates_per_step= t.get("n_steps_per_update",1),
            n_assets          = e["n_assets"],
            n_features        = len(d["asset_features"]) + len(d["macro_features"]),
            lookback          = d["lookback_window"],
            obs_noise_std     = t.get("obs_noise_std",     0.01),
            seed              = t.get("seed",              0),
        )


class SACTrainer:
    """Soft Actor-Critic trainer for the Alpha0 portfolio environment.

    Parameters
    ----------
    cfg:
        Full config dict.
    device:
        PyTorch device string (e.g. ``"cpu"``, ``"cuda"``).
    """

    def __init__(self, cfg: dict, device: str = "cpu", tb_log_dir: str | Path | None = None) -> None:
        """
        Parameters
        ----------
        cfg:
            Full config dict.
        device:
            PyTorch device string.
        tb_log_dir:
            TensorBoard log directory.  If None, defaults to
            ``cfg["logging"]["tensorboard_dir"]`` from config.
            Pass ``False`` to disable TensorBoard entirely.
        """
        self._cfg  = cfg
        self._sc   = SACConfig.from_cfg(cfg)
        self._sc.device = device
        self.device = torch.device(device)

        # TensorBoard writer (optional)
        self._writer = None
        _tb_dir = tb_log_dir
        if _tb_dir is None:
            _tb_dir = cfg.get("logging", {}).get("tensorboard_dir", None)
        if _tb_dir is not False and _tb_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=str(_tb_dir))
                logger.info("TensorBoard logging to %s", _tb_dir)
            except ImportError:
                logger.warning(
                    "tensorboard not installed — TensorBoard logging disabled. "
                    "Install with: pip install tensorboard"
                )

        torch.manual_seed(self._sc.seed)
        np.random.seed(self._sc.seed)
        random.seed(self._sc.seed)

        # Build policy
        encoder_kwargs = dict(
            n_features        = self._sc.n_features,
            n_assets          = self._sc.n_assets,
            lookback          = self._sc.lookback,
            d_model           = self._sc.d_model,
            n_temporal_layers = self._sc.n_temporal_layers,
            n_cross_layers    = self._sc.n_cross_layers,
            n_heads           = self._sc.n_heads,
            d_ff              = self._sc.d_ff,
            dropout           = self._sc.dropout,
        )
        self.policy = AlphaPolicy(
            encoder_cls    = MarketEncoder,
            encoder_kwargs = encoder_kwargs,
            actor_hidden   = self._sc.actor_hidden,
            critic_hidden  = self._sc.critic_hidden,
            dropout        = self._sc.dropout,
        ).to(self.device)

        param_counts = self.policy.count_parameters()
        logger.info("Policy parameter counts: %s", param_counts)

        # Optimisers (separate for actor and critics)
        actor_params  = list(self.policy.actor_encoder.parameters()) + \
                        list(self.policy.actor.parameters())
        critic_params = list(self.policy.critic1_encoder.parameters()) + \
                        list(self.policy.critic1.parameters()) + \
                        list(self.policy.critic2_encoder.parameters()) + \
                        list(self.policy.critic2.parameters())

        self.actor_opt  = optim.Adam(actor_params,  lr=self._sc.learning_rate,
                                      weight_decay=self._sc.weight_decay)
        self.critic_opt = optim.Adam(critic_params, lr=self._sc.learning_rate,
                                      weight_decay=self._sc.weight_decay)

        # Automatic entropy coefficient
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha      = self._sc.alpha_init
        self.alpha_opt  = optim.Adam([self.log_alpha], lr=self._sc.learning_rate)
        n_actions       = self._sc.n_assets + 1
        self.target_entropy = float(-n_actions)   # heuristic: -|A|

        # Replay buffer and augmenter
        self.buffer    = ReplayBuffer(self._sc.replay_capacity)
        self.augmenter = ObservationAugmenter(obs_noise_std=self._sc.obs_noise_std)

        # Mixed precision (uses Tensor Cores on T4/A100/V100)
        self._use_amp = (self.device.type == "cuda")
        self._scaler  = torch.amp.GradScaler("cuda", enabled=self._use_amp)

        # Training state
        self.total_steps   = 0
        self.total_updates = 0
        self._metrics: dict[str, list[float]] = {
            "critic_loss": [], "actor_loss": [], "alpha": [],
            "episode_reward": [], "episode_length": [],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        env,
        n_steps: int,
        val_env=None,
        eval_every: int = 5_000,
        save_dir: str | Path | None = None,
        early_stop_patience: int = 10,
    ) -> dict[str, list[float]]:
        """Run the SAC training loop.

        Parameters
        ----------
        env:
            Training :class:`~alpha0.env.market_env.MarketEnv`
            (split="train").
        n_steps:
            Total environment steps to train for.
        val_env:
            Optional validation environment.  If provided, used for
            early stopping on validation Sharpe ratio.
        eval_every:
            Evaluate on val_env every this many steps.
        save_dir:
            If provided, save model checkpoints here.
        early_stop_patience:
            Stop if validation Sharpe does not improve for this many
            evaluation intervals.

        Returns
        -------
        dict[str, list[float]]
            Training metrics history.
        """
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        obs_dict, _ = env.reset()
        episode_reward = 0.0
        episode_len    = 0
        best_val_sharpe = -np.inf
        patience_count  = 0

        logger.info(
            "Starting SAC training for %d steps (warmup=%d, batch=%d).",
            n_steps, self._sc.warmup_steps, self._sc.batch_size,
        )

        for step in range(n_steps):
            obs       = obs_dict["obs"]
            portfolio = obs_dict["portfolio"]

            # Select action
            if self.total_steps < self._sc.warmup_steps:
                # Random action during warmup
                action = env.action_space.sample()
            else:
                action = self._select_action(obs, portfolio)

            # Augment observation noise (training only)
            obs_aug = self.augmenter.augment_obs(obs)

            # Environment step
            next_obs_dict, reward, terminated, truncated, info = env.step(action)
            reward = self.augmenter.augment_reward(reward)
            done   = terminated or truncated

            # Store transition
            self.buffer.push(Transition(
                obs            = obs_aug,
                portfolio      = portfolio,
                action         = action,
                reward         = float(reward),
                next_obs       = next_obs_dict["obs"],
                next_portfolio = next_obs_dict["portfolio"],
                done           = done,
            ))

            episode_reward += reward
            episode_len    += 1
            self.total_steps += 1

            if done:
                self._metrics["episode_reward"].append(episode_reward)
                self._metrics["episode_length"].append(episode_len)
                if step % 500 == 0:
                    logger.info(
                        "Step %6d | EpReward: %+.4f | EpLen: %3d | Alpha: %.4f | "
                        "BufSize: %d",
                        step, episode_reward, episode_len,
                        self.alpha, len(self.buffer),
                    )
                if self._writer is not None:
                    self._writer.add_scalar("train/episode_reward", episode_reward, self.total_steps)
                    self._writer.add_scalar("train/episode_length", episode_len,    self.total_steps)
                obs_dict, _ = env.reset()
                episode_reward = 0.0
                episode_len    = 0
            else:
                obs_dict = next_obs_dict

            # Update networks
            if self.buffer.is_ready(self._sc.batch_size) and \
               self.total_steps >= self._sc.warmup_steps:
                for _ in range(self._sc.n_updates_per_step):
                    losses = self._update()
                    for k, v in losses.items():
                        self._metrics[k].append(v)
                    if self._writer is not None:
                        self._writer.add_scalar("train/critic_loss", losses["critic_loss"], self.total_updates)
                        self._writer.add_scalar("train/actor_loss",  losses["actor_loss"],  self.total_updates)
                        self._writer.add_scalar("train/alpha",       losses["alpha"],        self.total_updates)
                self.total_updates += 1

            # Periodic evaluation
            if val_env is not None and self.total_steps % eval_every == 0:
                val_sharpe = self.evaluate(val_env, n_episodes=5)
                logger.info(
                    "Step %6d | Val Sharpe: %.3f (best: %.3f)",
                    self.total_steps, val_sharpe, best_val_sharpe,
                )
                if self._writer is not None:
                    self._writer.add_scalar("eval/val_sharpe", val_sharpe, self.total_steps)
                if val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    patience_count  = 0
                    if save_dir:
                        self.save(save_dir / "best.pt")
                else:
                    patience_count += 1
                    if patience_count >= early_stop_patience:
                        logger.info(
                            "Early stopping: no val improvement for %d evals.",
                            patience_count,
                        )
                        break

        if save_dir:
            self.save(save_dir / "final.pt")

        if self._writer is not None:
            self._writer.flush()

        return self._metrics

    def close(self) -> None:
        """Close the TensorBoard writer (call when training is complete)."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def evaluate(
        self,
        env,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> float:
        """Run evaluation episodes and return mean Sharpe ratio.

        Parameters
        ----------
        env:
            Evaluation environment.
        n_episodes:
            Number of episodes to average over.
        deterministic:
            If True, use the deterministic (mean) policy.

        Returns
        -------
        float
            Mean annualised Sharpe ratio across episodes.
        """
        from alpha0.eval.metrics import sharpe_ratio
        import pandas as pd

        self.policy.eval()
        all_sharpes: list[float] = []

        with torch.no_grad():
            for _ in range(n_episodes):
                obs_dict, _ = env.reset()
                episode_returns: list[float] = []
                done = False
                while not done:
                    action = self._select_action(
                        obs_dict["obs"], obs_dict["portfolio"], deterministic
                    )
                    obs_dict, reward, term, trunc, info = env.step(action)
                    episode_returns.append(info.get("portfolio_return", 0.0))
                    done = term or trunc
                ret_series = pd.Series(episode_returns)
                all_sharpes.append(sharpe_ratio(ret_series,
                                                rf_rate=self._cfg["eval"]["risk_free_rate"]))

        self.policy.train()
        return float(np.mean(all_sharpes)) if all_sharpes else 0.0

    def save(self, path: str | Path) -> None:
        """Save policy weights and optimizer states."""
        torch.save({
            "policy_state_dict":      self.policy.state_dict(),
            "actor_opt_state_dict":   self.actor_opt.state_dict(),
            "critic_opt_state_dict":  self.critic_opt.state_dict(),
            "log_alpha":              self.log_alpha.item(),
            "total_steps":            self.total_steps,
            "total_updates":          self.total_updates,
        }, path)
        logger.info("Saved checkpoint to %s", path)

    def load(self, path: str | Path) -> None:
        """Load policy weights and optimizer states."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.actor_opt.load_state_dict(ckpt["actor_opt_state_dict"])
        self.critic_opt.load_state_dict(ckpt["critic_opt_state_dict"])
        self.log_alpha = torch.tensor(
            [ckpt["log_alpha"]], requires_grad=True, device=self.device
        )
        self.total_steps   = ckpt.get("total_steps",   0)
        self.total_updates = ckpt.get("total_updates", 0)
        logger.info("Loaded checkpoint from %s (step %d)", path, self.total_steps)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _select_action(
        self,
        obs: np.ndarray,
        portfolio: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Convert numpy obs to tensor, sample action, return numpy."""
        obs_t  = torch.tensor(obs[None],       dtype=torch.float32, device=self.device)
        port_t = torch.tensor(portfolio[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_t, _ = self.policy.act(obs_t, port_t, deterministic=deterministic)
        return action_t.squeeze(0).cpu().numpy()

    def _update(self) -> dict[str, float]:
        """One SAC gradient step. Returns loss values for logging."""
        batch = self.buffer.sample(self._sc.batch_size)
        obs, portfolio, action, reward, next_obs, next_port, done = \
            _batch_to_tensors(batch, self.device)

        autocast_ctx = torch.amp.autocast("cuda", enabled=self._use_amp)

        # ── Critic update ──────────────────────────────────────────────
        with torch.no_grad(), autocast_ctx:
            next_action, next_log_prob = self.policy.act(next_obs, next_port)
            q1_tgt, q2_tgt = self.policy.q_values(
                next_obs, next_port, next_action, use_targets=True
            )
            min_q_tgt = torch.min(q1_tgt, q2_tgt)
            y = reward + self._sc.gamma * (1.0 - done) * \
                (min_q_tgt - self.alpha * next_log_prob.unsqueeze(1))

        with autocast_ctx:
            q1, q2 = self.policy.q_values(obs, portfolio, action)
            critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad()
        self._scaler.scale(critic_loss).backward()
        self._scaler.unscale_(self.critic_opt)
        nn.utils.clip_grad_norm_(
            list(self.policy.critic1_encoder.parameters()) +
            list(self.policy.critic1.parameters()) +
            list(self.policy.critic2_encoder.parameters()) +
            list(self.policy.critic2.parameters()),
            self._sc.max_grad_norm,
        )
        self._scaler.step(self.critic_opt)
        self._scaler.update()

        # ── Actor update ───────────────────────────────────────────────
        with autocast_ctx:
            action_new, log_prob = self.policy.act(obs, portfolio)
            q1_new, q2_new = self.policy.q_values(obs, portfolio, action_new)
            min_q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha * log_prob.unsqueeze(1) - min_q_new).mean()

        self.actor_opt.zero_grad()
        self._scaler.scale(actor_loss).backward()
        self._scaler.unscale_(self.actor_opt)
        nn.utils.clip_grad_norm_(
            list(self.policy.actor_encoder.parameters()) +
            list(self.policy.actor.parameters()),
            self._sc.max_grad_norm,
        )
        self._scaler.step(self.actor_opt)
        self._scaler.update()

        # ── Auto-alpha update ─────────────────────────────────────────
        if self._sc.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_prob.detach() + self.target_entropy)
            ).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        # ── Soft target update ────────────────────────────────────────
        self.policy.soft_update_targets(self._sc.tau)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss":  float(actor_loss.item()),
            "alpha":       self.alpha,
        }
