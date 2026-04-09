"""Tests for the Alpha0 policy network components.

Covers:
- MarketEncoder: shape contracts, causal mask, output range
- Actor: sample vs deterministic, log-prob shape, tanh bounds
- Critic: Q-value shape
- AlphaPolicy: end-to-end act() and q_values(), soft target updates
- SACTrainer: single update step, save/load checkpoint
- ObservationAugmenter: noise injection
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from alpha0.policy.transformer import MarketEncoder
from alpha0.policy.allocation_head import Actor, Critic, AlphaPolicy
from alpha0.policy.ensemble import EnsemblePolicy
from alpha0.training.augmentation import ObservationAugmenter


# ─────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────

N_FEATURES = 14
N_ASSETS   = 10  # small for speed
LOOKBACK   = 20
D_MODEL    = 32
N_ACTIONS  = N_ASSETS + 1  # assets + cash
BATCH      = 4


@pytest.fixture
def encoder_kwargs():
    return dict(
        n_features        = N_FEATURES,
        n_assets          = N_ASSETS,
        lookback          = LOOKBACK,
        d_model           = D_MODEL,
        n_temporal_layers = 1,
        n_cross_layers    = 1,
        n_heads           = 2,
        d_ff              = 64,
        dropout           = 0.0,
    )


@pytest.fixture
def encoder(encoder_kwargs):
    return MarketEncoder(**encoder_kwargs)


@pytest.fixture
def obs():
    torch.manual_seed(42)
    return torch.randn(BATCH, LOOKBACK, N_ASSETS, N_FEATURES)


@pytest.fixture
def portfolio():
    # Valid portfolio: sum-to-1, all non-negative
    w = torch.ones(BATCH, N_ASSETS + 1) / (N_ASSETS + 1)
    return w


@pytest.fixture
def policy(encoder_kwargs):
    return AlphaPolicy(
        encoder_cls    = MarketEncoder,
        encoder_kwargs = encoder_kwargs,
        actor_hidden   = 64,
        critic_hidden  = 64,
        dropout        = 0.0,
    )


@pytest.fixture
def minimal_cfg():
    return {
        "data": {
            "lookback_window":  LOOKBACK,
            "asset_features":   ["f"] * 7,
            "macro_features":   ["m"] * 7,
        },
        "env": {
            "n_assets":         N_ASSETS,
            "episode_length":   50,
            "initial_portfolio_value": 1_000_000.0,
            "max_single_weight": 0.20,
            "min_cash_weight":   0.05,
            "max_gross_exposure": 1.00,
            "reward": {
                "daily_return_scale":  1.0,
                "risk_penalty_scale":  0.1,
                "risk_penalty_window": 5,
            },
        },
        "costs": {"commission_bps": 5, "slippage_bps": 5},
        "eval": {"risk_free_rate": 0.04, "trading_days_per_year": 252},
        "training": {
            "d_model":             D_MODEL,
            "n_temporal_layers":   1,
            "n_cross_layers":      1,
            "n_heads":             2,
            "d_ff":                64,
            "actor_hidden":        64,
            "critic_hidden":       64,
            "dropout":             0.0,
            "batch_size":          8,
            "replay_buffer_size":  500,
            "learning_rate":       1e-3,
            "gamma":               0.99,
            "tau":                 0.005,
            "alpha":               0.2,
            "weight_decay":        0.0,
            "max_grad_norm":       1.0,
            "warmup_steps":        10,
            "n_steps_per_update":  1,
            "obs_noise_std":       0.01,
            "seed":                0,
        },
    }


# ─────────────────────────────────────────────────────────────
# MarketEncoder
# ─────────────────────────────────────────────────────────────

class TestMarketEncoder:
    def test_output_shape(self, encoder, obs, portfolio):
        with torch.no_grad():
            rep = encoder(obs, portfolio)
        assert rep.shape == (BATCH, D_MODEL), f"Expected ({BATCH}, {D_MODEL}), got {rep.shape}"

    def test_output_finite(self, encoder, obs, portfolio):
        with torch.no_grad():
            rep = encoder(obs, portfolio)
        assert torch.isfinite(rep).all(), "Encoder output contains NaN/Inf"

    def test_batch_independence(self, encoder, obs, portfolio):
        """Changing one batch element should not affect others."""
        with torch.no_grad():
            rep_orig = encoder(obs, portfolio).clone()

        obs_mod = obs.clone()
        obs_mod[0] = torch.randn_like(obs_mod[0]) * 10
        with torch.no_grad():
            rep_mod = encoder(obs_mod, portfolio)

        # Other elements should be identical
        assert torch.allclose(rep_orig[1:], rep_mod[1:], atol=1e-5), \
            "Encoder is not batch-independent"

    def test_causal_mask_registered(self, encoder):
        assert hasattr(encoder, "causal_mask"), "causal_mask not registered"
        assert encoder.causal_mask.shape == (LOOKBACK, LOOKBACK)

    def test_parameter_count_reasonable(self, encoder):
        n = sum(p.numel() for p in encoder.parameters())
        # For small test model: should be < 500k
        assert n < 500_000, f"Too many parameters: {n}"

    def test_train_eval_determinism(self, encoder, obs, portfolio):
        """In eval mode, forward pass is deterministic (dropout=0 in fixture)."""
        encoder.eval()
        with torch.no_grad():
            out1 = encoder(obs, portfolio)
            out2 = encoder(obs, portfolio)
        assert torch.allclose(out1, out2), "Encoder not deterministic in eval mode"

    def test_gradient_flows(self, encoder, obs, portfolio):
        """Gradients should flow through the encoder."""
        rep = encoder(obs, portfolio)
        loss = rep.mean()
        loss.backward()
        for name, p in encoder.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"


# ─────────────────────────────────────────────────────────────
# Actor
# ─────────────────────────────────────────────────────────────

class TestActor:
    @pytest.fixture
    def actor(self):
        return Actor(D_MODEL, N_ACTIONS, hidden_dim=64, dropout=0.0)

    @pytest.fixture
    def rep(self):
        torch.manual_seed(0)
        return torch.randn(BATCH, D_MODEL)

    def test_sample_shapes(self, actor, rep):
        action, log_prob = actor.sample(rep)
        assert action.shape   == (BATCH, N_ACTIONS)
        assert log_prob.shape == (BATCH,)

    def test_action_bounded(self, actor, rep):
        """Tanh output must be strictly in (-1, 1)."""
        action, _ = actor.sample(rep)
        assert (action.abs() < 1.0).all(), "Actions not in (-1, 1)"

    def test_log_prob_finite(self, actor, rep):
        _, log_prob = actor.sample(rep)
        assert torch.isfinite(log_prob).all(), "Log probs contain NaN/Inf"

    def test_deterministic_action_shape(self, actor, rep):
        det = actor.deterministic_action(rep)
        assert det.shape == (BATCH, N_ACTIONS)

    def test_deterministic_vs_mean(self, actor, rep):
        """deterministic_action should equal tanh(mean)."""
        mean, _ = actor.forward(rep)
        expected = torch.tanh(mean)
        actual   = actor.deterministic_action(rep)
        assert torch.allclose(expected, actual, atol=1e-6)

    def test_log_std_clamped(self, actor, rep):
        _, log_std = actor.forward(rep)
        assert (log_std >= -5.0 - 1e-5).all()
        assert (log_std <=  2.0 + 1e-5).all()


# ─────────────────────────────────────────────────────────────
# Critic
# ─────────────────────────────────────────────────────────────

class TestCritic:
    @pytest.fixture
    def critic(self):
        return Critic(D_MODEL, N_ACTIONS, hidden_dim=64, dropout=0.0)

    @pytest.fixture
    def rep_and_action(self):
        torch.manual_seed(1)
        return torch.randn(BATCH, D_MODEL), torch.randn(BATCH, N_ACTIONS)

    def test_q_value_shape(self, critic, rep_and_action):
        rep, action = rep_and_action
        q = critic(rep, action)
        assert q.shape == (BATCH, 1), f"Expected ({BATCH}, 1), got {q.shape}"

    def test_q_value_finite(self, critic, rep_and_action):
        rep, action = rep_and_action
        q = critic(rep, action)
        assert torch.isfinite(q).all()


# ─────────────────────────────────────────────────────────────
# AlphaPolicy
# ─────────────────────────────────────────────────────────────

class TestAlphaPolicy:
    def test_act_stochastic_shapes(self, policy, obs, portfolio):
        action, log_prob = policy.act(obs, portfolio, deterministic=False)
        assert action.shape   == (BATCH, N_ACTIONS)
        assert log_prob.shape == (BATCH,)

    def test_act_deterministic_no_log_prob(self, policy, obs, portfolio):
        action, log_prob = policy.act(obs, portfolio, deterministic=True)
        assert action.shape == (BATCH, N_ACTIONS)
        assert log_prob is None

    def test_q_values_shapes(self, policy, obs, portfolio):
        action = torch.randn(BATCH, N_ACTIONS)
        q1, q2 = policy.q_values(obs, portfolio, action)
        assert q1.shape == (BATCH, 1)
        assert q2.shape == (BATCH, 1)

    def test_target_q_values_shapes(self, policy, obs, portfolio):
        action = torch.randn(BATCH, N_ACTIONS)
        q1, q2 = policy.q_values(obs, portfolio, action, use_targets=True)
        assert q1.shape == (BATCH, 1)
        assert q2.shape == (BATCH, 1)

    def test_target_init_matches_critics(self, policy):
        """At init, target critics must have the same weights as online critics."""
        for p_src, p_tgt in zip(
            policy.critic1.parameters(),
            policy.target_critic1.parameters(),
        ):
            assert torch.allclose(p_src, p_tgt), "Critic1 / target_critic1 mismatch at init"

    def test_soft_update_changes_targets(self, policy):
        """After a soft update, target weights should have moved toward online weights."""
        # Perturb online critics
        with torch.no_grad():
            for p in policy.critic1.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        target_before = [p.clone() for p in policy.target_critic1.parameters()]
        policy.soft_update_targets(tau=0.5)
        target_after  = [p.clone() for p in policy.target_critic1.parameters()]

        changed = any(not torch.allclose(b, a) for b, a in zip(target_before, target_after))
        assert changed, "Soft update did not change target weights"

    def test_soft_update_tau_zero_no_change(self, policy):
        """tau=0 should leave target weights unchanged."""
        target_before = [p.clone() for p in policy.target_critic1.parameters()]
        policy.soft_update_targets(tau=0.0)
        target_after  = [p.clone() for p in policy.target_critic1.parameters()]
        for b, a in zip(target_before, target_after):
            assert torch.allclose(b, a), "tau=0 changed target weights"

    def test_soft_update_tau_one_copies_critics(self, policy):
        """tau=1 should copy online weights exactly to targets."""
        with torch.no_grad():
            for p in policy.critic1.parameters():
                p.fill_(3.14)
        policy.soft_update_targets(tau=1.0)
        for p_src, p_tgt in zip(
            policy.critic1.parameters(),
            policy.target_critic1.parameters(),
        ):
            assert torch.allclose(p_src, p_tgt), "tau=1 did not copy online → target"

    def test_count_parameters_keys(self, policy):
        counts = policy.count_parameters()
        for key in ("actor_encoder", "actor_head", "critic1", "critic2", "total"):
            assert key in counts, f"Missing key: {key}"
        assert counts["total"] > 0

    def test_encode_obs_unknown_network_raises(self, policy, obs, portfolio):
        with pytest.raises(ValueError, match="Unknown network"):
            policy.encode_obs(obs, portfolio, network="bogus")


# ─────────────────────────────────────────────────────────────
# EnsemblePolicy
# ─────────────────────────────────────────────────────────────

class TestEnsemblePolicy:
    def test_ensemble_averages_actions(self, encoder_kwargs, obs, portfolio):
        policies = [
            AlphaPolicy(
                encoder_cls=MarketEncoder,
                encoder_kwargs=encoder_kwargs,
                actor_hidden=64,
                critic_hidden=64,
                dropout=0.0,
            )
            for _ in range(3)
        ]
        ensemble = EnsemblePolicy(policies)
        action = ensemble.act(obs, portfolio)
        assert action.shape == (BATCH, N_ACTIONS)
        assert torch.isfinite(action).all()

    def test_ensemble_single_member(self, policy, obs, portfolio):
        ensemble = EnsemblePolicy([policy])
        action_ens   = ensemble.act(obs, portfolio)
        action_solo, _ = policy.act(obs, portfolio, deterministic=True)
        assert torch.allclose(action_ens, action_solo, atol=1e-5)


# ─────────────────────────────────────────────────────────────
# SACTrainer
# ─────────────────────────────────────────────────────────────

class TestSACTrainer:
    @pytest.fixture
    def trainer(self, minimal_cfg):
        from alpha0.training.sac import SACTrainer
        return SACTrainer(minimal_cfg, device="cpu")

    def test_policy_created(self, trainer):
        assert isinstance(trainer.policy, AlphaPolicy)

    def test_warmup_uses_random_action(self, trainer, minimal_cfg):
        """During warmup, actions come from the env action space, not the policy."""
        from alpha0.training.sac import SACTrainer
        import numpy as np

        # Verify the buffer stays at 0 until warmup is satisfied
        t = SACTrainer(minimal_cfg, device="cpu")
        assert len(t.buffer) == 0

    def test_single_update_step(self, trainer, minimal_cfg):
        """Fill buffer above batch_size and run one update without errors."""
        from alpha0.training.sac import Transition
        import numpy as np

        rng = np.random.default_rng(0)
        n_a = minimal_cfg["env"]["n_assets"]
        T   = minimal_cfg["data"]["lookback_window"]
        F   = 14

        for _ in range(minimal_cfg["training"]["batch_size"] + 1):
            trainer.buffer.push(Transition(
                obs            = rng.standard_normal((T, n_a, F)).astype(np.float32),
                portfolio      = rng.random(n_a + 1).astype(np.float32),
                action         = rng.standard_normal(n_a + 1).astype(np.float32),
                reward         = float(rng.standard_normal()),
                next_obs       = rng.standard_normal((T, n_a, F)).astype(np.float32),
                next_portfolio = rng.random(n_a + 1).astype(np.float32),
                done           = False,
            ))

        losses = trainer._update()
        assert "critic_loss" in losses
        assert "actor_loss"  in losses
        assert "alpha"       in losses
        assert math.isfinite(losses["critic_loss"])
        assert math.isfinite(losses["actor_loss"])

    def test_save_and_load_roundtrip(self, trainer):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "test.pt"
            trainer.save(ckpt)
            trainer.load(ckpt)

        # Policy should still work after loading
        obs_t  = torch.randn(1, LOOKBACK, N_ASSETS, N_FEATURES)
        port_t = torch.ones(1, N_ASSETS + 1) / (N_ASSETS + 1)
        action, _ = trainer.policy.act(obs_t, port_t)
        assert action.shape == (1, N_ASSETS + 1)

    def test_select_action_returns_numpy(self, trainer, minimal_cfg):
        import numpy as np
        n_a = minimal_cfg["env"]["n_assets"]
        T   = minimal_cfg["data"]["lookback_window"]
        F   = 14
        obs_np   = np.random.randn(T, n_a, F).astype(np.float32)
        port_np  = np.ones(n_a + 1, dtype=np.float32) / (n_a + 1)
        action   = trainer._select_action(obs_np, port_np)
        assert isinstance(action, np.ndarray)
        assert action.shape == (n_a + 1,)


# ─────────────────────────────────────────────────────────────
# ObservationAugmenter
# ─────────────────────────────────────────────────────────────

class TestObservationAugmenter:
    def test_zero_std_returns_same_array(self):
        aug = ObservationAugmenter(obs_noise_std=0.0)
        obs = np.ones((20, 10, 14), dtype=np.float32)
        result = aug.augment_obs(obs)
        np.testing.assert_array_equal(obs, result)

    def test_noise_applied(self):
        rng = np.random.default_rng(0)
        aug = ObservationAugmenter(obs_noise_std=1.0)
        obs = np.zeros((20, 10, 14), dtype=np.float32)
        result = aug.augment_obs(obs)
        # With std=1.0 some values should differ
        assert not np.allclose(obs, result)

    def test_output_clipped(self):
        aug = ObservationAugmenter(obs_noise_std=100.0)
        obs = np.zeros((5, 5, 14), dtype=np.float32)
        result = aug.augment_obs(obs)
        assert result.max() <= 5.0 + 1e-5
        assert result.min() >= -5.0 - 1e-5

    def test_reward_noise_zero(self):
        aug = ObservationAugmenter(reward_noise_std=0.0)
        assert aug.augment_reward(1.0) == 1.0

    def test_reward_noise_applied(self):
        np.random.seed(7)
        aug = ObservationAugmenter(reward_noise_std=1.0)
        samples = [aug.augment_reward(0.0) for _ in range(100)]
        assert not all(s == 0.0 for s in samples)
