"""Ensemble training for Alpha0 — trains N agents with different seeds.

Why ensembles?
--------------
A single SAC agent can overfit to a particular random seed (weight
initialisation + replay buffer sampling order).  Training five agents with
seeds 0–4 and averaging their deterministic actions at inference time:

1. Reduces variance — the average of N uncorrelated forecasts has 1/N the
   variance of a single forecast.
2. Provides calibrated uncertainty — disagreement between ensemble members
   is a signal of prediction uncertainty (high disagreement → lower position
   confidence).
3. Improves robustness — no single model's idiosyncrasies dominate.

Usage::

    trainer = EnsembleTrainer(cfg, data_loader, n_agents=5, device="cuda")
    ensemble = trainer.train_all(n_steps=500_000)

    # Or load a previously trained ensemble
    ensemble = trainer.load_ensemble("artifacts/ensemble/run_01")
    sharpe = trainer.evaluate_ensemble(ensemble, test_env, n_episodes=20)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from alpha0.policy.allocation_head import AlphaPolicy
from alpha0.policy.ensemble import EnsemblePolicy
from alpha0.policy.transformer import MarketEncoder
from alpha0.training.sac import SACTrainer

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Training outcome for one ensemble member."""
    seed: int
    val_sharpe: float
    checkpoint_path: str
    train_history: dict[str, list[float]] = field(default_factory=dict)
    n_steps_trained: int = 0


@dataclass
class EnsembleTrainingResult:
    """Aggregated result of all ensemble members."""
    agent_results: list[AgentResult]
    ensemble_val_sharpe: float = 0.0
    ensemble_test_sharpe: float = 0.0
    artifacts_dir: str = ""

    def summary(self) -> str:
        lines = [
            f"Ensemble training summary ({len(self.agent_results)} agents):",
            f"  Ensemble val  Sharpe: {self.ensemble_val_sharpe:+.3f}",
            f"  Ensemble test Sharpe: {self.ensemble_test_sharpe:+.3f}",
            "  Per-agent val Sharpe:",
        ]
        for r in sorted(self.agent_results, key=lambda x: x.val_sharpe, reverse=True):
            lines.append(f"    seed={r.seed}: {r.val_sharpe:+.3f}  ({r.n_steps_trained:,} steps)")
        return "\n".join(lines)


class EnsembleTrainer:
    """Trains N independent SAC agents and combines them into an ensemble.

    Each agent:
    - Uses the same architecture and hyperparameters
    - Differs only in random seed (weight init + replay buffer sampling)
    - Is saved to ``{artifacts_dir}/agent_{seed:02d}/best.pt``

    Parameters
    ----------
    cfg:
        Full config dict.
    data_loader:
        Pre-built :class:`~alpha0.data.loader.DataLoader`.
    n_agents:
        Number of ensemble members.  Default 5.
    device:
        PyTorch device string.
    artifacts_dir:
        Root directory for checkpoints.
    """

    def __init__(
        self,
        cfg: dict,
        data_loader,
        n_agents: int = 5,
        device: str = "cpu",
        artifacts_dir: str | Path = "artifacts/ensemble",
    ) -> None:
        self._cfg       = cfg
        self._loader    = data_loader
        self._n_agents  = n_agents
        self._device    = device
        self._art_dir   = Path(artifacts_dir)
        self._art_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_all(
        self,
        n_steps: int = 200_000,
        eval_every: int | None = None,
        early_stop_patience: int = 10,
        seeds: list[int] | None = None,
        resume: bool = False,
    ) -> tuple[EnsemblePolicy, EnsembleTrainingResult]:
        """Train all N agents sequentially and return the ensemble.

        Parameters
        ----------
        n_steps:
            Total environment steps per agent.
        eval_every:
            Evaluate on val env every N steps.  Defaults to n_steps // 50.
        early_stop_patience:
            Stop early if val Sharpe doesn't improve for this many evals.
        seeds:
            List of random seeds.  Defaults to ``range(n_agents)``.
        resume:
            If True, skip agents whose ``best.pt`` checkpoint already exists.

        Returns
        -------
        tuple[EnsemblePolicy, EnsembleTrainingResult]
        """
        from alpha0.env.market_env import MarketEnv

        if seeds is None:
            seeds = list(range(self._n_agents))

        _eval_every = eval_every or max(1_000, n_steps // 50)
        agent_results: list[AgentResult] = []
        loaded_policies: list[AlphaPolicy] = []

        for seed in seeds:
            ckpt_dir  = self._art_dir / f"agent_{seed:02d}"
            best_ckpt = ckpt_dir / "best.pt"

            if resume and best_ckpt.exists():
                logger.info("Resuming: loading existing checkpoint for seed=%d", seed)
                policy = self._load_policy(best_ckpt)
                loaded_policies.append(policy)
                agent_results.append(AgentResult(
                    seed=seed,
                    val_sharpe=0.0,  # not re-evaluated
                    checkpoint_path=str(best_ckpt),
                ))
                continue

            logger.info("=" * 60)
            logger.info("Training ensemble member %d/%d  (seed=%d)", seeds.index(seed) + 1, len(seeds), seed)
            logger.info("=" * 60)

            # Override seed in config
            cfg_seed = dict(self._cfg)
            cfg_seed = {**self._cfg, "training": {**self._cfg.get("training", {}), "seed": seed}}

            t0 = time.perf_counter()
            trainer = SACTrainer(cfg_seed, device=self._device)

            train_env = MarketEnv(cfg_seed, self._loader, split="train")
            val_env   = MarketEnv(cfg_seed, self._loader, split="val")

            history = trainer.train(
                env                 = train_env,
                n_steps             = n_steps,
                val_env             = val_env,
                eval_every          = _eval_every,
                save_dir            = ckpt_dir,
                early_stop_patience = early_stop_patience,
            )
            elapsed = time.perf_counter() - t0

            # Load best checkpoint for ensemble
            if best_ckpt.exists():
                trainer.load(best_ckpt)
            val_sharpe = trainer.evaluate(val_env, n_episodes=10, deterministic=True)

            logger.info(
                "Agent seed=%d complete in %.1f min | Val Sharpe=%.3f",
                seed, elapsed / 60, val_sharpe,
            )

            agent_results.append(AgentResult(
                seed             = seed,
                val_sharpe       = val_sharpe,
                checkpoint_path  = str(best_ckpt if best_ckpt.exists() else ckpt_dir / "final.pt"),
                train_history    = history,
                n_steps_trained  = trainer.total_steps,
            ))
            loaded_policies.append(trainer.policy)

        # Build ensemble
        ensemble = EnsemblePolicy(loaded_policies)

        # Evaluate ensemble on val
        val_env = MarketEnv(self._cfg, self._loader, split="val")
        ens_val_sharpe = self._eval_ensemble_sharpe(ensemble, val_env, n_episodes=10)

        # Evaluate ensemble on test
        test_env = MarketEnv(self._cfg, self._loader, split="test")
        ens_test_sharpe = self._eval_ensemble_sharpe(ensemble, test_env, n_episodes=10)

        result = EnsembleTrainingResult(
            agent_results      = agent_results,
            ensemble_val_sharpe= ens_val_sharpe,
            ensemble_test_sharpe=ens_test_sharpe,
            artifacts_dir      = str(self._art_dir),
        )
        logger.info("\n%s", result.summary())

        # Save ensemble manifest
        self._save_manifest(result)

        return ensemble, result

    def load_ensemble(
        self, ensemble_dir: str | Path | None = None
    ) -> EnsemblePolicy:
        """Load a previously trained ensemble from disk.

        Loads ``best.pt`` from each ``agent_XX/`` subdirectory.

        Parameters
        ----------
        ensemble_dir:
            Root directory.  Defaults to ``self._art_dir``.

        Returns
        -------
        EnsemblePolicy
        """
        root = Path(ensemble_dir) if ensemble_dir else self._art_dir
        policies: list[AlphaPolicy] = []

        agent_dirs = sorted(root.glob("agent_*"))
        if not agent_dirs:
            raise FileNotFoundError(f"No agent directories found in {root}")

        for agent_dir in agent_dirs:
            ckpt = agent_dir / "best.pt"
            if not ckpt.exists():
                ckpt = agent_dir / "final.pt"
            if not ckpt.exists():
                logger.warning("No checkpoint found in %s, skipping.", agent_dir)
                continue
            policies.append(self._load_policy(ckpt))
            logger.info("Loaded %s", ckpt)

        if not policies:
            raise FileNotFoundError(f"No valid checkpoints found in {root}")

        logger.info("Loaded ensemble with %d members.", len(policies))
        return EnsemblePolicy(policies)

    def evaluate_ensemble(
        self,
        ensemble: EnsemblePolicy,
        env,
        n_episodes: int = 20,
        deterministic: bool = True,
    ) -> dict[str, float]:
        """Evaluate an ensemble on an environment and return metrics.

        Parameters
        ----------
        ensemble:
            Trained :class:`~alpha0.policy.ensemble.EnsemblePolicy`.
        env:
            Any Gymnasium-compatible market environment.
        n_episodes:
            Number of evaluation episodes.
        deterministic:
            If True, use deterministic (mean) actions.

        Returns
        -------
        dict[str, float]
            ``{sharpe, sortino, max_drawdown, total_return, hit_rate}``
        """
        from alpha0.eval.metrics import compute_metrics

        all_returns: list[float] = []

        ensemble.eval()
        with torch.no_grad():
            for _ in range(n_episodes):
                obs_dict, _ = env.reset()
                done = False
                while not done:
                    obs_t  = torch.tensor(obs_dict["obs"][None],       dtype=torch.float32)
                    port_t = torch.tensor(obs_dict["portfolio"][None], dtype=torch.float32)
                    action = ensemble.act(obs_t, port_t).squeeze(0).cpu().numpy()
                    obs_dict, _, term, trunc, info = env.step(action)
                    all_returns.append(info.get("portfolio_return", 0.0))
                    done = term or trunc

        if not all_returns:
            return {}

        ret_series = pd.Series(all_returns)
        rf = self._cfg["eval"].get("risk_free_rate", 0.04)
        return compute_metrics(ret_series, rf_rate=rf)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_policy(self, checkpoint_path: Path) -> AlphaPolicy:
        """Load an AlphaPolicy from a checkpoint file."""
        tc  = self._cfg.get("training", {})
        dc  = self._cfg["data"]
        ec  = self._cfg["env"]

        encoder_kwargs = dict(
            n_features        = len(dc.get("asset_features", [])) + len(dc.get("macro_features", [])),
            n_assets          = ec["n_assets"],
            lookback          = dc["lookback_window"],
            d_model           = tc.get("d_model", 128),
            n_temporal_layers = tc.get("n_temporal_layers", 3),
            n_cross_layers    = tc.get("n_cross_layers", 1),
            n_heads           = tc.get("n_heads", 4),
            d_ff              = tc.get("d_ff", 256),
            dropout           = 0.0,  # eval mode
        )
        policy = AlphaPolicy(
            encoder_cls    = MarketEncoder,
            encoder_kwargs = encoder_kwargs,
            actor_hidden   = tc.get("actor_hidden", 256),
            critic_hidden  = tc.get("critic_hidden", 256),
            dropout        = 0.0,
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        policy.load_state_dict(ckpt["policy_state_dict"])
        policy.eval()
        return policy

    def _eval_ensemble_sharpe(
        self, ensemble: EnsemblePolicy, env, n_episodes: int = 10
    ) -> float:
        """Quick Sharpe estimate for an ensemble on env."""
        from alpha0.eval.metrics import sharpe_ratio

        all_rets: list[float] = []
        ensemble.eval()
        with torch.no_grad():
            for _ in range(n_episodes):
                obs, _ = env.reset()
                done   = False
                while not done:
                    obs_t  = torch.tensor(obs["obs"][None],       dtype=torch.float32)
                    port_t = torch.tensor(obs["portfolio"][None], dtype=torch.float32)
                    action = ensemble.act(obs_t, port_t).squeeze(0).cpu().numpy()
                    obs, _, term, trunc, info = env.step(action)
                    all_rets.append(info.get("portfolio_return", 0.0))
                    done = term or trunc

        if not all_rets:
            return 0.0
        return float(sharpe_ratio(
            pd.Series(all_rets),
            rf_rate=self._cfg["eval"].get("risk_free_rate", 0.04),
        ))

    def _save_manifest(self, result: EnsembleTrainingResult) -> None:
        """Save a JSON manifest of the ensemble training run."""
        import json
        manifest = {
            "n_agents":           len(result.agent_results),
            "ensemble_val_sharpe": result.ensemble_val_sharpe,
            "ensemble_test_sharpe":result.ensemble_test_sharpe,
            "agents": [
                {
                    "seed":            r.seed,
                    "val_sharpe":      r.val_sharpe,
                    "checkpoint_path": r.checkpoint_path,
                    "n_steps_trained": r.n_steps_trained,
                }
                for r in result.agent_results
            ],
        }
        path = self._art_dir / "ensemble_manifest.json"
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Saved ensemble manifest to %s", path)
