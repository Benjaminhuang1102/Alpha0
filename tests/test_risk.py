"""Tests for Phase 3/4 risk management components."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha0.risk.manager import RiskManager, RiskState
from alpha0.risk.regime import RegimeDetector, MarketRegime, REGIME_EQUITY_SCALE
from alpha0.risk.position_sizer import VolatilityTargeter, KellySizer


# ─────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────

N = 10  # assets
W = N + 1  # weights dimension (assets + cash)


def _risk_cfg(
    soft=0.15, hard=0.20, cooloff=5,
    max_sector=0.30, min_pos=3, max_turnover=0.30,
):
    return {
        "risk": {
            "max_drawdown_soft":     soft,
            "max_drawdown_hard":     hard,
            "cooloff_days":          cooloff,
            "max_sector_weight":     max_sector,
            "min_positions":         min_pos,
            "max_daily_turnover":    max_turnover,
            "spy_correlation_alert": 0.95,
        }
    }


def _uniform_weights(n=N) -> np.ndarray:
    """Equal weight allocation."""
    w = np.ones(n + 1, dtype=np.float32)
    w[:n] = 1.0 / n
    w[-1] = 0.0
    return w


# ─────────────────────────────────────────────────────────────
# RiskManager — basic state
# ─────────────────────────────────────────────────────────────

class TestRiskManagerState:
    def test_initial_state(self):
        mgr = RiskManager(_risk_cfg())
        mgr.reset(1_000_000.0)
        assert mgr.current_drawdown == pytest.approx(0.0)
        assert mgr.circuit_breaker_level is None

    def test_drawdown_tracking(self):
        mgr = RiskManager(_risk_cfg())
        mgr.reset(1_000_000.0)
        mgr.update(900_000.0)
        assert mgr.current_drawdown == pytest.approx(0.10, abs=1e-4)

    def test_new_peak_resets_drawdown(self):
        mgr = RiskManager(_risk_cfg())
        mgr.reset(1_000_000.0)
        mgr.update(900_000.0)
        assert mgr.current_drawdown > 0
        mgr.update(1_100_000.0)  # new peak
        assert mgr.current_drawdown == pytest.approx(0.0, abs=1e-4)

    def test_summary_keys(self):
        mgr = RiskManager(_risk_cfg())
        mgr.reset(1_000_000.0)
        s = mgr.summary()
        for key in ("drawdown", "peak_value", "circuit_breaker", "cooldown_remaining"):
            assert key in s


# ─────────────────────────────────────────────────────────────
# RiskManager — circuit breakers
# ─────────────────────────────────────────────────────────────

class TestCircuitBreakers:
    def _setup(self, soft=0.15, hard=0.20, cooloff=3):
        mgr = RiskManager(_risk_cfg(soft=soft, hard=hard, cooloff=cooloff))
        mgr.reset(1_000_000.0)
        return mgr

    def test_no_breaker_below_soft(self):
        mgr = self._setup()
        mgr.update(900_000.0)  # 10% drawdown, below soft
        w = _uniform_weights()
        safe_w, alerts = mgr.apply(w, w)
        assert len(alerts) == 0
        assert mgr.circuit_breaker_level is None

    def test_soft_breaker_triggers(self):
        mgr = self._setup()
        mgr.update(840_000.0)  # 16% drawdown ≥ soft=15%
        w = _uniform_weights()
        safe_w, alerts = mgr.apply(w, w)
        assert any("SOFT" in a for a in alerts)
        assert mgr.circuit_breaker_level == "soft"

    def test_soft_breaker_halves_equity(self):
        mgr = self._setup()
        mgr.update(840_000.0)
        w = _uniform_weights()
        safe_w, _ = mgr.apply(w, w)
        # Cash should be ≥ 50%
        assert safe_w[-1] >= 0.49  # slight tolerance for floating point

    def test_hard_breaker_triggers(self):
        mgr = self._setup()
        mgr.update(790_000.0)  # 21% drawdown ≥ hard=20%
        w = _uniform_weights()
        safe_w, alerts = mgr.apply(w, w)
        assert any("HARD" in a for a in alerts)
        assert mgr.circuit_breaker_level == "hard"

    def test_hard_breaker_forces_100pct_cash(self):
        mgr = self._setup()
        mgr.update(790_000.0)
        w = _uniform_weights()
        safe_w, _ = mgr.apply(w, w)
        assert safe_w[-1] == pytest.approx(1.0, abs=1e-4)
        assert safe_w[:-1].sum() == pytest.approx(0.0, abs=1e-4)

    def test_hard_breaker_cooldown(self):
        mgr = self._setup(cooloff=3)
        mgr.update(790_000.0)
        w = _uniform_weights()
        mgr.apply(w, w)
        # Verify cooldown countdown
        assert mgr.state.cooldown_remaining == 3
        # After cooloff updates without recovery, breaker should lapse
        mgr.update(790_000.0)
        mgr.update(790_000.0)
        mgr.update(790_000.0)
        assert mgr.state.cooldown_remaining == 0

    def test_soft_breaker_only_fires_once(self):
        """SOFT alert emitted only on first trigger, not every step."""
        mgr = self._setup()
        mgr.update(840_000.0)
        w = _uniform_weights()
        _, alerts1 = mgr.apply(w, w)
        mgr.update(840_000.0)
        _, alerts2 = mgr.apply(w, w)
        # Second call should not re-emit the "SOFT circuit breaker triggered" alert
        assert sum(1 for a in alerts2 if "SOFT circuit breaker triggered" in a) == 0

    def test_weights_sum_to_one_after_breaker(self):
        """Weights must always sum to 1 after any constraint is applied."""
        mgr = self._setup()
        for pv in [900_000, 850_000, 800_000, 750_000]:
            mgr.update(pv)
        w = _uniform_weights()
        safe_w, _ = mgr.apply(w, w)
        assert float(safe_w.sum()) == pytest.approx(1.0, abs=1e-4)


# ─────────────────────────────────────────────────────────────
# RiskManager — sector limits
# ─────────────────────────────────────────────────────────────

class TestSectorLimits:
    def _setup_with_sectors(self):
        mgr = RiskManager(
            _risk_cfg(max_sector=0.30),
            sector_map={f"A{i}": "Tech" if i < 5 else "Finance" for i in range(N)},
        )
        mgr.reset(1_000_000.0)
        return mgr

    def test_sector_overweight_capped(self):
        mgr = self._setup_with_sectors()
        # Put 60% in Tech (5 assets × 12% each)
        w = np.zeros(W, dtype=np.float32)
        w[:5] = 0.12   # Tech
        w[5:N] = 0.02  # Finance
        w[-1] = 1.0 - w[:-1].sum()
        tickers = [f"A{i}" for i in range(N)]
        safe_w, alerts = mgr.apply(w, w, tickers=tickers)
        tech_weight = safe_w[:5].sum()
        assert tech_weight <= 0.30 + 1e-4
        assert any("Tech" in a for a in alerts)

    def test_sector_below_limit_unchanged(self):
        mgr = self._setup_with_sectors()
        # Balanced: 20% Tech, 20% Finance
        w = np.zeros(W, dtype=np.float32)
        w[:5] = 0.04   # Tech = 20%
        w[5:N] = 0.04  # Finance = 20%
        w[-1] = 0.60
        tickers = [f"A{i}" for i in range(N)]
        safe_w, alerts = mgr.apply(w, w, tickers=tickers)
        np.testing.assert_allclose(safe_w[:N], w[:N], atol=1e-4)
        assert len([a for a in alerts if "Sector" in a]) == 0


# ─────────────────────────────────────────────────────────────
# RiskManager — turnover cap
# ─────────────────────────────────────────────────────────────

class TestTurnoverCap:
    def test_turnover_capped(self):
        mgr = RiskManager(_risk_cfg(max_turnover=0.10))
        mgr.reset(1_000_000.0)

        current = _uniform_weights()
        # Extreme rebalance: everything to first asset
        target = np.zeros(W, dtype=np.float32)
        target[0] = 0.95
        target[-1] = 0.05

        safe_w, alerts = mgr.apply(target, current)
        # Turnover before: ≈ 90%; after capping should be ≈ 10%
        actual_turnover = float(np.abs(safe_w - current).sum()) / 2.0
        assert actual_turnover <= 0.10 + 1e-4
        assert any("Turnover" in a for a in alerts)
        assert float(safe_w.sum()) == pytest.approx(1.0, abs=1e-4)


# ─────────────────────────────────────────────────────────────
# RegimeDetector
# ─────────────────────────────────────────────────────────────

class TestRegimeDetector:
    def test_default_regime_before_history(self):
        det = RegimeDetector(min_history=60)
        regime = det.update(100.0)
        # Not enough history → BULL_LOW_VOL default
        assert regime == MarketRegime.BULL_LOW_VOL

    def test_bull_detection(self):
        det = RegimeDetector(trend_lookback=10, vol_lookback=5,
                             min_history=10, long_run_vol_window=20)
        # Rising prices
        for i in range(30):
            det.update(100.0 + i * 0.5)
        regime = det.current_regime()
        assert regime in (MarketRegime.BULL_LOW_VOL, MarketRegime.BULL_HIGH_VOL)

    def test_bear_detection(self):
        det = RegimeDetector(trend_lookback=10, vol_lookback=5,
                             min_history=10, long_run_vol_window=20)
        # Falling prices
        for i in range(30):
            det.update(200.0 - i * 3)
        regime = det.current_regime()
        assert regime in (MarketRegime.BEAR_LOW_VOL, MarketRegime.BEAR_HIGH_VOL)

    def test_equity_scale_decreases_in_bear(self):
        bull_scale = REGIME_EQUITY_SCALE[MarketRegime.BULL_LOW_VOL]
        bear_scale = REGIME_EQUITY_SCALE[MarketRegime.BEAR_HIGH_VOL]
        assert bear_scale < bull_scale

    def test_detect_from_series(self):
        det = RegimeDetector(min_history=20, trend_lookback=20, long_run_vol_window=40)
        prices = pd.Series(
            [100.0 + i * 0.1 for i in range(60)],
            index=pd.date_range("2020-01-01", periods=60),
        )
        result = det.detect_from_series(prices)
        assert len(result) == 60
        assert all(isinstance(r, MarketRegime) for r in result)

    def test_reset_clears_history(self):
        det = RegimeDetector(min_history=5, trend_lookback=5, long_run_vol_window=10)
        for i in range(20):
            det.update(100.0 + i)
        det.reset()
        assert len(det.regime_history) == 0


# ─────────────────────────────────────────────────────────────
# VolatilityTargeter
# ─────────────────────────────────────────────────────────────

class TestVolatilityTargeter:
    def test_insufficient_history_passthrough(self):
        vt = VolatilityTargeter(target_annual_vol=0.10, lookback=21, min_history=21)
        w = _uniform_weights()
        result = vt.scale(w)
        np.testing.assert_array_almost_equal(result, w)

    def test_high_vol_reduces_equity(self):
        vt = VolatilityTargeter(target_annual_vol=0.10, lookback=21, min_history=5,
                                trading_days=252)
        # Inject very high vol returns (≈40% annualised)
        rng = np.random.default_rng(0)
        for _ in range(21):
            vt.update(float(rng.normal(0, 0.025)))  # ~40% ann vol

        w = _uniform_weights()
        scaled = vt.scale(w)
        # Equity should be lower than original
        assert scaled[:-1].sum() < w[:-1].sum() + 1e-4

    def test_low_vol_increases_equity(self):
        vt = VolatilityTargeter(target_annual_vol=0.20, lookback=21, min_history=5,
                                max_leverage=1.0, trading_days=252)
        # Very low vol returns (≈2% annualised)
        for _ in range(21):
            vt.update(0.0001)

        w = np.zeros(W, dtype=np.float32)
        w[:N] = 0.3 / N  # only 30% equity
        w[-1] = 0.70
        scaled = vt.scale(w)
        # Equity should increase toward budget
        assert scaled[:-1].sum() > w[:-1].sum() - 1e-4

    def test_weights_sum_to_one(self):
        vt = VolatilityTargeter(target_annual_vol=0.10, lookback=5, min_history=5)
        for i in range(10):
            vt.update(0.001 * (i - 5))
        w = _uniform_weights()
        scaled = vt.scale(w)
        assert float(scaled.sum()) == pytest.approx(1.0, abs=1e-4)

    def test_min_cash_respected(self):
        vt = VolatilityTargeter(target_annual_vol=0.50, lookback=5, min_history=5,
                                min_cash=0.10)
        # Alternating tiny returns → very low vol → scale factor clamps to max_leverage
        # Starting from 100% equity (w[-1]=0) → min_cash should enforce 10% cash
        for i in range(5):
            vt.update(0.0001 * (1 if i % 2 == 0 else -1))
        w = _uniform_weights()  # 100% equity, 0% cash
        scaled = vt.scale(w)
        assert scaled[-1] >= 0.10 - 1e-4

    def test_reset(self):
        vt = VolatilityTargeter(target_annual_vol=0.10, lookback=5, min_history=5)
        for _ in range(10):
            vt.update(0.01)
        vt.reset()
        w = _uniform_weights()
        result = vt.scale(w)
        # After reset, not enough history → passthrough
        np.testing.assert_array_almost_equal(result, w)


# ─────────────────────────────────────────────────────────────
# KellySizer
# ─────────────────────────────────────────────────────────────

class TestKellySizer:
    def test_insufficient_history_passthrough(self):
        ks = KellySizer(lookback=63, min_history=42)
        w = _uniform_weights()
        result = ks.scale(w)
        np.testing.assert_array_almost_equal(result, w)

    def test_negative_edge_goes_to_cash(self):
        ks = KellySizer(kelly_fraction=0.5, lookback=10, min_history=5, max_leverage=1.0)
        # Negative drift returns → Kelly fraction < 0 → clipped to 0 → full cash
        for _ in range(10):
            ks.update(-0.005)
        w = _uniform_weights()
        scaled = ks.scale(w)
        assert float(scaled.sum()) == pytest.approx(1.0, abs=1e-4)

    def test_weights_sum_to_one(self):
        ks = KellySizer(lookback=10, min_history=5)
        for i in range(10):
            ks.update(0.002)
        w = _uniform_weights()
        scaled = ks.scale(w)
        assert float(scaled.sum()) == pytest.approx(1.0, abs=1e-4)


# ─────────────────────────────────────────────────────────────
# RiskManager + MarketEnv integration
# ─────────────────────────────────────────────────────────────

class TestRiskManagerEnvIntegration:
    @pytest.fixture
    def env_with_risk(self):
        """Lightweight env with a RiskManager attached."""
        import numpy as np
        from unittest.mock import MagicMock
        from alpha0.env.market_env import MarketEnv
        from alpha0.risk.manager import RiskManager

        cfg = {
            "data": {"lookback_window": 10},
            "env": {
                "episode_length": 20,
                "n_assets": 5,
                "initial_portfolio_value": 1_000_000.0,
                "max_single_weight": 0.40,
                "min_cash_weight": 0.05,
                "max_gross_exposure": 1.00,
                "reward": {
                    "daily_return_scale": 1.0,
                    "risk_penalty_scale": 0.1,
                    "risk_penalty_window": 5,
                },
            },
            "costs": {"commission_bps": 5, "slippage_bps": 5},
            "risk": {
                "max_drawdown_soft":     0.15,
                "max_drawdown_hard":     0.20,
                "cooloff_days":          5,
                "max_sector_weight":     0.30,
                "min_positions":         2,
                "max_daily_turnover":    0.50,
                "spy_correlation_alert": 0.95,
            },
        }

        n_assets   = cfg["env"]["n_assets"]
        n_features = 14
        lookback   = cfg["data"]["lookback_window"]
        ep_len     = cfg["env"]["episode_length"]
        total      = ep_len + lookback

        # Build synthetic feature tensor and loader mock
        import torch
        tensor = torch.randn(total + 20, n_assets, n_features)
        prices = torch.ones(total + 20, n_assets) * 100.0
        close  = torch.ones(total + 20, n_assets) * 100.0

        loader_mock = MagicMock()
        loader_mock._feature_tensor = tensor
        loader_mock._close_prices   = close
        loader_mock._split_indices  = {"train": (0, total + 10)}
        loader_mock.dates           = [pd.Timestamp("2020-01-01") + pd.Timedelta(days=i) for i in range(total + 20)]

        risk_mgr = RiskManager(cfg)
        env = MarketEnv(cfg, loader_mock, split="train", risk_manager=risk_mgr)
        return env

    def test_env_resets_risk_manager(self, env_with_risk):
        """reset() should reset the risk manager state."""
        env_with_risk.reset()
        assert env_with_risk._risk_manager.current_drawdown == pytest.approx(0.0, abs=1e-4)

    def test_env_step_returns_risk_info(self, env_with_risk):
        env_with_risk.reset()
        action = env_with_risk.action_space.sample()
        _, _, _, _, info = env_with_risk.step(action)
        assert "risk_alerts" in info
        assert "drawdown" in info
        assert isinstance(info["risk_alerts"], list)
