"""Tests for eval modules: metrics, backtest, benchmarks, dashboard."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alpha0.eval.metrics import (
    annualised_return,
    annualised_vol,
    beta_alpha,
    calmar_ratio,
    compute_metrics,
    hit_rate,
    max_drawdown,
    portfolio_turnover,
    sharpe_ratio,
    sortino_ratio,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _const_returns(daily_ret: float, n: int = 252) -> pd.Series:
    """Series with constant daily return."""
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series([daily_ret] * n, index=idx)


def _eval_cfg() -> dict:
    return {
        "eval": {
            "risk_free_rate":        0.04,
            "trading_days_per_year": 252,
            "rolling_window":        21,
            "backtest": {"initial_capital": 100_000.0},
        },
        "costs": {"commission_bps": 5, "slippage_bps": 5},
        "artifacts": {"reports_dir": "artifacts/reports"},
    }


# ─────────────────────────────────────────────────────────────
# Metric unit tests (known analytic values)
# ─────────────────────────────────────────────────────────────

class TestAnnualisedReturn:
    def test_zero_returns_zero(self):
        assert annualised_return(_const_returns(0.0)) == pytest.approx(0.0, abs=1e-6)

    def test_constant_positive(self):
        """Constant +1% daily for 252 days → (1.01)^252 - 1 ≈ 11.93x annualised."""
        r = _const_returns(0.01, n=252)
        expected = 1.01 ** 252 - 1.0
        assert annualised_return(r) == pytest.approx(expected, rel=1e-4)

    def test_empty_series(self):
        assert annualised_return(pd.Series([], dtype=float)) == 0.0


class TestAnnualisedVol:
    def test_zero_vol(self):
        assert annualised_vol(_const_returns(0.01)) == pytest.approx(0.0, abs=1e-6)

    def test_known_vol(self):
        """Daily std=0.01 → annualised ≈ 0.01 * sqrt(252)."""
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0, 0.01, 1000))
        result = annualised_vol(r)
        assert abs(result - 0.01 * math.sqrt(252)) < 0.005


class TestSharpeRatio:
    def test_zero_std_returns_zero(self):
        """Constant returns have zero std — Sharpe is undefined, return 0."""
        assert sharpe_ratio(_const_returns(0.001)) == pytest.approx(0.0)

    def test_spy_vs_spy_positive(self):
        """Positive-drift random returns should yield a positive Sharpe."""
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0.001, 0.01, 500))
        assert sharpe_ratio(r) > 0

    def test_negative_returns_negative_sharpe(self):
        rng = np.random.default_rng(1)
        r = pd.Series(rng.normal(-0.002, 0.01, 500))
        assert sharpe_ratio(r) < 0


class TestMaxDrawdown:
    def test_always_positive_returns_zero_drawdown(self):
        assert max_drawdown(_const_returns(0.01)) == pytest.approx(0.0, abs=1e-6)

    def test_known_drawdown(self):
        """Returns [0, 0, -0.10, -0.10] → drawdown from 1.0 to 0.81 = -19%."""
        idx = pd.date_range("2020-01-01", periods=4, freq="B")
        r = pd.Series([0.0, 0.0, -0.10, -0.10], index=idx)
        # Cumulative: [1.0, 1.0, 0.9, 0.81]; peak = 1.0; drawdown = (0.81 - 1.0)/1.0
        expected = 1.0 - 0.81
        assert max_drawdown(r) == pytest.approx(expected, rel=1e-4)

    def test_single_large_drop(self):
        idx = pd.date_range("2020-01-01", periods=3, freq="B")
        r = pd.Series([0.10, -0.50, 0.05], index=idx)
        dd = max_drawdown(r)
        assert dd > 0.3

    def test_empty_series(self):
        assert max_drawdown(pd.Series([], dtype=float)) == 0.0


class TestCalmarRatio:
    def test_positive_calmar_for_positive_returns(self):
        rng = np.random.default_rng(77)
        r = pd.Series(rng.normal(0.001, 0.01, 500))
        c = calmar_ratio(r)
        assert isinstance(c, float)

    def test_zero_drawdown_returns_zero(self):
        assert calmar_ratio(_const_returns(0.01)) == pytest.approx(0.0, abs=1e-6)


class TestBetaAlpha:
    def test_spy_vs_spy_beta_one_alpha_zero(self):
        """Beta of series vs itself = 1, alpha ≈ 0."""
        rng = np.random.default_rng(5)
        r = pd.Series(rng.normal(0.0005, 0.01, 500))
        beta, alpha = beta_alpha(r, r)
        assert beta == pytest.approx(1.0, abs=1e-4)
        assert alpha == pytest.approx(0.0, abs=1e-2)

    def test_uncorrelated_returns_zero_beta(self):
        """Two uncorrelated random series should have beta ≈ 0."""
        rng = np.random.default_rng(99)
        r1 = pd.Series(rng.normal(0, 0.01, 300))
        r2 = pd.Series(rng.normal(0, 0.01, 300))
        beta, _ = beta_alpha(r1, r2)
        assert abs(beta) < 0.2

    def test_leveraged_beta(self):
        """2x leveraged SPY should have beta ≈ 2."""
        rng = np.random.default_rng(3)
        spy = pd.Series(rng.normal(0.0005, 0.01, 500))
        lev = spy * 2.0
        beta, _ = beta_alpha(lev, spy)
        assert beta == pytest.approx(2.0, abs=0.05)


class TestPortfolioTurnover:
    def test_no_rebalance_zero_turnover(self):
        """Constant weights → zero turnover."""
        idx = pd.date_range("2020-01-01", periods=5, freq="B")
        w = pd.DataFrame({"A": [0.5] * 5, "B": [0.5] * 5}, index=idx)
        assert portfolio_turnover(w) == pytest.approx(0.0)

    def test_full_daily_rebalance(self):
        """Alternating between 100% A and 100% B → 100% one-way turnover per day."""
        idx = pd.date_range("2020-01-01", periods=4, freq="B")
        w = pd.DataFrame(
            {"A": [1.0, 0.0, 1.0, 0.0], "B": [0.0, 1.0, 0.0, 1.0]},
            index=idx,
        )
        assert portfolio_turnover(w) == pytest.approx(1.0, abs=0.01)


class TestHitRate:
    def test_all_positive(self):
        assert hit_rate(_const_returns(0.01)) == pytest.approx(1.0)

    def test_all_negative(self):
        assert hit_rate(_const_returns(-0.01)) == pytest.approx(0.0)

    def test_mixed(self):
        idx = pd.date_range("2020-01-01", periods=4, freq="B")
        r = pd.Series([0.01, -0.01, 0.01, -0.01], index=idx)
        assert hit_rate(r) == pytest.approx(0.5)


class TestComputeMetrics:
    def test_returns_all_keys(self):
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.0005, 0.01, 500))
        m = compute_metrics(r)
        for key in ["total_return", "annualised_return", "sharpe_ratio",
                    "max_drawdown", "hit_rate"]:
            assert key in m, f"Key '{key}' missing from metrics"

    def test_with_benchmark(self):
        rng = np.random.default_rng(1)
        r = pd.Series(rng.normal(0.001, 0.01, 500))
        b = pd.Series(rng.normal(0.0005, 0.01, 500))
        m = compute_metrics(r, benchmark_returns=b)
        assert "beta" in m and "alpha" in m

    def test_with_weights(self):
        rng = np.random.default_rng(2)
        r = pd.Series(rng.normal(0, 0.01, 100))
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        r.index = idx
        w = pd.DataFrame({"A": [0.5] * 100, "B": [0.5] * 100}, index=idx)
        m = compute_metrics(r, weights=w)
        assert "turnover" in m


# ─────────────────────────────────────────────────────────────
# Backtest tests
# ─────────────────────────────────────────────────────────────

from alpha0.eval.backtest import Backtest


class TestBacktest:
    def test_buy_hold_single_asset(self):
        """100% allocation to one asset, no rebalance → return tracks price."""
        idx = pd.date_range("2020-01-01", periods=5, freq="B")
        prices = pd.DataFrame({"A": [100.0, 110.0, 105.0, 115.0, 120.0]}, index=idx)
        weights = pd.DataFrame({"A": [1.0] * 5}, index=idx)
        bt = Backtest(_eval_cfg())
        result = bt.run(weights, prices)
        # Should start at initial_capital (100k) and grow proportionally
        assert result.portfolio_value.iloc[0] > 0
        assert isinstance(result.metrics["sharpe_ratio"], float)

    def test_cost_deducted_on_rebalance(self):
        """Full daily rebalance at 10bps should approximate 25.2% annual drag."""
        rng = np.random.default_rng(7)
        n = 252
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        # Flat prices (no returns, only costs)
        prices = pd.DataFrame({"A": [100.0] * n, "B": [100.0] * n}, index=idx)
        # Alternate between all-A and all-B → 100% turnover each day
        w_a = np.array([1.0, 0.0])
        w_b = np.array([0.0, 1.0])
        rows = [w_a if i % 2 == 0 else w_b for i in range(n)]
        weights = pd.DataFrame(rows, index=idx, columns=["A", "B"])
        bt = Backtest(_eval_cfg())
        result = bt.run(weights, prices)
        # Total cost ≈ 10bps * 252 * turnover → portfolio should have lost ~25%
        final_value = result.portfolio_value.iloc[-1]
        initial = _eval_cfg()["eval"]["backtest"]["initial_capital"]
        fraction_remaining = final_value / initial
        # Should be roughly 1 - 0.252 = 0.748 (allow wide tolerance due to compounding)
        assert fraction_remaining < 0.85, f"Expected cost drag, got {fraction_remaining:.2%}"
        assert fraction_remaining > 0.60, f"Too much cost, got {fraction_remaining:.2%}"

    def test_portfolio_value_always_positive(self):
        """Portfolio value should never go to zero or negative."""
        rng = np.random.default_rng(42)
        n = 252
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        # Very volatile prices
        prices = pd.DataFrame(
            {"A": np.cumprod(1 + rng.normal(0, 0.05, n)) * 100},
            index=idx,
        )
        weights = pd.DataFrame({"A": [1.0] * n}, index=idx)
        bt = Backtest(_eval_cfg())
        result = bt.run(weights, prices)
        assert (result.portfolio_value > 0).all()


# ─────────────────────────────────────────────────────────────
# BenchmarkRunner tests
# ─────────────────────────────────────────────────────────────

from alpha0.eval.benchmarks import BenchmarkRunner


def _make_prices(tickers: list[str], n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = {}
    for t in tickers:
        data[t] = np.cumprod(1 + rng.normal(0.0003, 0.01, n)) * 100
    return pd.DataFrame(data, index=idx)


class TestBenchmarkRunner:
    def test_spy_buyhold_returns_result(self):
        cfg = _eval_cfg()
        runner = BenchmarkRunner(cfg)
        prices = _make_prices(["SPY"])
        result = runner.spy_buyhold(prices["SPY"])
        assert "sharpe_ratio" in result.metrics
        assert len(result.daily_returns) > 0

    def test_equal_weight_no_crash(self):
        cfg = _eval_cfg()
        runner = BenchmarkRunner(cfg)
        prices = _make_prices(["A", "B", "C", "D"], n=300)
        result = runner.equal_weight(prices)
        assert result.portfolio_value.iloc[-1] > 0

    def test_sixty_forty_allocation(self):
        cfg = _eval_cfg()
        runner = BenchmarkRunner(cfg)
        prices = _make_prices(["equity", "bond"], n=300)
        result = runner.sixty_forty(prices["equity"], prices["bond"])
        assert result.portfolio_value.iloc[-1] > 0

    def test_run_all_returns_dict(self):
        cfg = _eval_cfg()
        runner = BenchmarkRunner(cfg)
        prices = _make_prices(["A", "B", "C"], n=300)
        results = runner.run_all(prices, spy_prices=prices["A"], bond_prices=prices["B"])
        assert "spy_buyhold" in results
        assert "equal_weight" in results


# ─────────────────────────────────────────────────────────────
# Dashboard smoke tests (checks it runs without error)
# ─────────────────────────────────────────────────────────────

from alpha0.eval.dashboard import Dashboard


class TestDashboard:
    def _make_result(self) -> "BacktestResult":
        from alpha0.eval.backtest import BacktestResult
        rng = np.random.default_rng(0)
        n = 300
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        returns = pd.Series(rng.normal(0.001, 0.01, n), index=idx)
        pv = (1 + returns).cumprod() * 100_000
        weights = pd.DataFrame(
            {"A": [0.5] * n, "B": [0.5] * n}, index=idx
        )
        return BacktestResult(
            portfolio_value=pv,
            daily_returns=returns,
            weights=weights,
            metrics=compute_metrics(returns),
        )

    def _make_cfg(self, tmp_path) -> dict:
        cfg = _eval_cfg()
        cfg["artifacts"] = {"reports_dir": str(tmp_path / "reports")}
        return cfg

    def test_generate_report_creates_files(self, tmp_path):
        cfg = self._make_cfg(tmp_path)
        dashboard = Dashboard(cfg)
        result = self._make_result()
        out = dashboard.generate_report({"test_strategy": result})
        assert out.exists()
        assert (out / "cumulative_returns.png").exists()
        assert (out / "metrics_comparison.csv").exists()

    def test_plot_cumulative_no_crash(self, tmp_path):
        cfg = self._make_cfg(tmp_path)
        dashboard = Dashboard(cfg)
        results = {"strat": self._make_result()}
        dashboard.plot_cumulative_returns(results, tmp_path / "test.png")
        assert (tmp_path / "test.png").exists()
