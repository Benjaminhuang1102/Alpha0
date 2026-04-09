"""Tests for the Alpha0 environment components: costs, reward, and MarketEnv."""

from __future__ import annotations

import numpy as np
import pytest

from alpha0.env.costs import TransactionCostModel
from alpha0.env.reward import RewardFunction


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _cost_cfg(commission: int = 5, slippage: int = 5) -> dict:
    return {"costs": {"commission_bps": commission, "slippage_bps": slippage}}


def _reward_cfg(scale: float = 0.1, window: int = 5) -> dict:
    return {
        "env": {
            "reward": {
                "daily_return_scale": 1.0,
                "risk_penalty_scale": scale,
                "risk_penalty_window": window,
            }
        }
    }


# ─────────────────────────────────────────────────────────────
# TransactionCostModel
# ─────────────────────────────────────────────────────────────

class TestTransactionCostModel:
    def test_zero_turnover_zero_cost(self):
        model = TransactionCostModel(_cost_cfg())
        w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        cost, turnover = model.compute_cost(w, w)
        assert cost == pytest.approx(0.0)
        assert turnover == pytest.approx(0.0)

    def test_full_turnover_max_cost(self):
        """Moving 100% of portfolio from asset 0 to asset 1 = 100% one-way turnover."""
        model = TransactionCostModel(_cost_cfg(5, 5))
        w_before = np.array([1.0, 0.0])
        w_after  = np.array([0.0, 1.0])
        cost, turnover = model.compute_cost(w_before, w_after)
        # turnover = sum(|1.0, 1.0|) / 2 = 1.0
        assert turnover == pytest.approx(1.0)
        # cost = 10bps * 1.0 = 0.001
        assert cost == pytest.approx(0.001)

    def test_partial_rebalance(self):
        """Partial rebalance produces proportional cost."""
        model = TransactionCostModel(_cost_cfg(5, 5))
        w_before = np.array([0.5, 0.5])
        w_after  = np.array([0.6, 0.4])
        cost, turnover = model.compute_cost(w_before, w_after)
        # |0.1| + |-0.1| = 0.2  → turnover = 0.1
        assert turnover == pytest.approx(0.1)
        assert cost == pytest.approx(0.001 * 0.1 / 0.1 * turnover)

    def test_cash_included_in_turnover(self):
        """Moving from equity to cash counts as turnover."""
        model = TransactionCostModel(_cost_cfg())
        # 3 assets + 1 cash slot
        w_before = np.array([0.5, 0.5, 0.0, 0.0])  # last = cash
        w_after  = np.array([0.0, 0.0, 0.0, 1.0])  # all cash
        cost, turnover = model.compute_cost(w_before, w_after)
        assert turnover == pytest.approx(1.0)
        assert cost > 0.0

    def test_total_bps_property(self):
        model = TransactionCostModel(_cost_cfg(3, 7))
        assert model.total_bps == pytest.approx(10.0)

    def test_annual_drag_full_daily_rebalance(self):
        """Full daily rebalance at 10bps ≈ 25.2% annual cost (10bps × 252)."""
        model = TransactionCostModel(_cost_cfg(5, 5))
        w_a = np.array([1.0, 0.0])
        w_b = np.array([0.0, 1.0])
        daily_cost, _ = model.compute_cost(w_a, w_b)
        annual_drag = daily_cost * 252
        assert annual_drag == pytest.approx(0.252, abs=0.001)


# ─────────────────────────────────────────────────────────────
# RewardFunction
# ─────────────────────────────────────────────────────────────

class TestRewardFunction:
    def test_reset_clears_history(self):
        rf = RewardFunction(_reward_cfg())
        rf.compute(0.01)
        rf.compute(0.02)
        rf.reset()
        # After reset, vol = 0 → risk_penalty = 0
        reward, info = rf.compute(0.01)
        assert info["realised_vol"] == pytest.approx(0.0)

    def test_positive_return_positive_reward(self):
        """With scale=0 risk penalty, reward equals log_return."""
        cfg = {
            "env": {
                "reward": {
                    "daily_return_scale": 1.0,
                    "risk_penalty_scale": 0.0,  # no risk penalty
                    "risk_penalty_window": 5,
                }
            }
        }
        rf = RewardFunction(cfg)
        reward, info = rf.compute(0.01)
        assert reward == pytest.approx(0.01)

    def test_risk_penalty_grows_with_volatility(self):
        """Higher volatility returns produce a larger risk penalty."""
        rf_low  = RewardFunction(_reward_cfg(scale=0.1))
        rf_high = RewardFunction(_reward_cfg(scale=0.1))

        # Low-vol returns
        for r in [0.001, -0.001, 0.001, -0.001]:
            rf_low.compute(r)

        # High-vol returns
        for r in [0.05, -0.05, 0.05, -0.05]:
            rf_high.compute(r)

        _, info_low  = rf_low.compute(0.0)
        _, info_high = rf_high.compute(0.0)
        assert info_high["risk_penalty"] > info_low["risk_penalty"]

    def test_info_dict_keys(self):
        rf = RewardFunction(_reward_cfg())
        _, info = rf.compute(0.005)
        assert "log_return"     in info
        assert "risk_penalty"   in info
        assert "realised_vol"   in info
        assert "portfolio_value" in info

    def test_reward_decreases_with_risk_scale(self):
        """Higher risk_penalty_scale reduces reward for volatile episodes."""
        # Build up some volatility
        def _run(scale):
            cfg = {
                "env": {
                    "reward": {
                        "daily_return_scale": 1.0,
                        "risk_penalty_scale": scale,
                        "risk_penalty_window": 10,
                    }
                }
            }
            rf = RewardFunction(cfg)
            returns = [0.03, -0.03, 0.03, -0.03, 0.03, -0.03, 0.03, -0.03, 0.03]
            for r in returns:
                rf.compute(r)
            reward, _ = rf.compute(0.01)
            return reward

        assert _run(0.0) > _run(0.5), "Higher risk scale should reduce reward"


# ─────────────────────────────────────────────────────────────
# MarketEnv tests
# ─────────────────────────────────────────────────────────────

import pandas as pd
from gymnasium.utils.env_checker import check_env

from alpha0.data.loader import DataLoader, N_FEATURES
from alpha0.data.storage import ParquetStore
from alpha0.env.market_env import MarketEnv


def _make_ohlcv_for_env(tickers: list[str], n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    fields = ["Open", "High", "Low", "Close", "Volume"]
    data = {}
    for t in tickers:
        base_price = rng.uniform(20, 200)
        prices = np.cumprod(1 + rng.normal(0, 0.015, n)) * base_price
        prices = np.abs(prices) + 1.0
        data[("Close", t)] = prices
        data[("Open",  t)] = prices * rng.uniform(0.99, 1.01, n)
        data[("High",  t)] = prices * rng.uniform(1.00, 1.03, n)
        data[("Low",   t)] = prices * rng.uniform(0.97, 1.00, n)
        data[("Volume",t)] = rng.integers(100_000, 5_000_000, n).astype(float)
    cols = pd.MultiIndex.from_tuples(data.keys(), names=["field", "ticker"])
    return pd.DataFrame(np.column_stack(list(data.values())), index=idx, columns=cols)


def _make_macro_for_env(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "vix":         rng.uniform(10, 40, n),
            "dgs10":       rng.uniform(1, 5,  n),
            "dgs2":        rng.uniform(0.5, 3, n),
            "dxy":         rng.uniform(90, 110, n),
            "gold_return": rng.normal(0, 0.01, n),
            "oil_return":  rng.normal(0, 0.02, n),
        },
        index=idx,
    )


def _env_cfg(n_assets: int = 5) -> dict:
    return {
        "data": {
            "lookback_window": 10,
            "train_end": "2012-12-31",
            "val_end":   "2013-06-30",
            "normalization": {"window": 60, "clip": 5.0},
        },
        "env": {
            "n_assets":               n_assets,
            "episode_length":         30,
            "initial_portfolio_value": 100_000.0,
            "max_single_weight":      0.30,
            "min_cash_weight":        0.05,
            "max_gross_exposure":     1.00,
            "reward": {
                "daily_return_scale":  1.0,
                "risk_penalty_scale":  0.1,
                "risk_penalty_window": 10,
            },
        },
        "costs": {"commission_bps": 5, "slippage_bps": 5},
    }


@pytest.fixture(scope="module")
def built_loader(tmp_path_factory):
    n_assets = 5
    tickers = [f"T{i}" for i in range(n_assets)]
    ohlcv = _make_ohlcv_for_env(tickers, n=800)
    macro = _make_macro_for_env(n=800)
    proc = tmp_path_factory.mktemp("proc")
    raw  = tmp_path_factory.mktemp("raw")
    cfg = _env_cfg(n_assets)
    loader = DataLoader(cfg, ParquetStore(raw), ParquetStore(proc))
    loader.build(ohlcv, macro, tickers)
    return loader, cfg


class TestMarketEnv:
    def _make_env(self, built_loader, split="train") -> MarketEnv:
        loader, cfg = built_loader
        return MarketEnv(cfg, loader, split=split)

    def test_check_env_passes(self, built_loader):
        """gymnasium check_env should not raise."""
        env = self._make_env(built_loader)
        check_env(env, warn=True, skip_render_check=True)

    def test_reset_returns_valid_obs(self, built_loader):
        env = self._make_env(built_loader)
        obs, info = env.reset(seed=0)
        assert "obs" in obs and "portfolio" in obs
        assert obs["obs"].shape == (10, 5, N_FEATURES)
        assert obs["portfolio"].shape == (6,)  # 5 assets + cash
        assert env.observation_space.contains(obs)

    def test_step_returns_valid_obs(self, built_loader):
        env = self._make_env(built_loader)
        env.reset(seed=1)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_weights_sum_to_one_after_step(self, built_loader):
        env = self._make_env(built_loader)
        env.reset(seed=2)
        for _ in range(5):
            action = env.action_space.sample()
            env.step(action)
        w = env._current_weights
        assert abs(w.sum() - 1.0) < 1e-4, f"Weights sum to {w.sum()}"

    def test_cash_floor_enforced(self, built_loader):
        """Target allocation (at rebalance time) respects min_cash_weight.

        Post-step drifted weights may fall below the floor because asset
        returns shift weights relatively (cash earns 0 while equities move).
        The constraint is enforced on the target allocation.
        """
        env = self._make_env(built_loader)
        env.reset(seed=3)
        cfg = _env_cfg()
        min_cash = cfg["env"]["min_cash_weight"]
        for _ in range(10):
            action = np.zeros(6, dtype=np.float32)
            action[0] = 100.0
            target = env._action_to_weights(action)
            assert target[-1] >= min_cash - 1e-5, (
                f"Target cash weight {target[-1]} < min_cash {min_cash}"
            )
            env.step(action)

    def test_max_weight_enforced(self, built_loader):
        """Target allocation (at rebalance time) respects max_single_weight.

        Note: post-step drifted weights may exceed the cap naturally as
        asset returns shift the weights intra-day.  The constraint is on
        the *target allocation* chosen by the agent.
        """
        env = self._make_env(built_loader)
        env.reset(seed=4)
        cfg = _env_cfg()
        max_w = cfg["env"]["max_single_weight"]
        for _ in range(10):
            action = np.zeros(6, dtype=np.float32)
            action[0] = 100.0  # extreme concentration
            target = env._action_to_weights(action)
            assert target[:-1].max() <= max_w + 1e-4, (
                f"Target weight {target[:-1].max()} > max_single_weight {max_w}"
            )
            env.step(action)

    def test_portfolio_value_positive(self, built_loader):
        env = self._make_env(built_loader)
        env.reset(seed=5)
        for _ in range(30):
            env.step(env.action_space.sample())
        assert env._portfolio_value > 0

    def test_truncated_at_episode_length(self, built_loader):
        env = self._make_env(built_loader)
        env.reset(seed=6)
        done = False
        steps = 0
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            steps += 1
        assert steps == 30  # episode_length from cfg

    def test_1000_random_episodes(self, built_loader):
        """Smoke test: 1000 episodes with random policy, invariants must hold."""
        env = self._make_env(built_loader)
        for ep in range(1000):
            obs, _ = env.reset(seed=ep)
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                # Invariants
                assert env._portfolio_value > 0, f"Episode {ep}: negative value"
                w = env._current_weights
                assert abs(w.sum() - 1.0) < 1e-3, f"Episode {ep}: weights sum {w.sum()}"
                assert env.observation_space.contains(obs)
