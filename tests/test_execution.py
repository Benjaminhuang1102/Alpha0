"""Tests for Phase 3/4 execution layer: OrderManager and PaperTrader."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha0.execution.order_manager import OrderManager, Order, OrderSide, Fill


# ─────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────

N = 5


def _exec_cfg(commission_bps=5, slippage_bps=5):
    return {"costs": {"commission_bps": commission_bps, "slippage_bps": slippage_bps}}


def _prices(n=N, price=100.0) -> dict[str, float]:
    return {f"A{i}": price for i in range(n)}


def _tickers(n=N) -> list[str]:
    return [f"A{i}" for i in range(n)]


# ─────────────────────────────────────────────────────────────
# OrderManager — initialisation
# ─────────────────────────────────────────────────────────────

class TestOrderManagerInit:
    def test_reset_sets_cash(self):
        om = OrderManager(_exec_cfg())
        om.reset(initial_cash=1_000_000.0)
        assert om.cash == pytest.approx(1_000_000.0)

    def test_reset_clears_positions(self):
        om = OrderManager(_exec_cfg())
        om.reset(1_000_000.0, {"A0": 100.0})
        om.reset(500_000.0)
        assert len(om.positions) == 0

    def test_initial_portfolio_value(self):
        om = OrderManager(_exec_cfg())
        om.reset(1_000_000.0, {"A0": 100.0})
        pv = om.mark_to_market(_prices())
        # 100 shares × $100 + $1M cash = $1,010,000
        assert pv == pytest.approx(1_010_000.0)


# ─────────────────────────────────────────────────────────────
# OrderManager — order generation
# ─────────────────────────────────────────────────────────────

class TestOrderGeneration:
    def test_no_orders_at_target(self):
        """If already at target weights, no orders generated."""
        om = OrderManager(_exec_cfg())
        om.reset(1_000_000.0)

        # Give equal positions (20% each, 0% cash)
        for i in range(N):
            om._positions[f"A{i}"] = 2_000.0  # 2000 shares × $100 = $200k
        om._cash = 0.0

        target = np.array([0.20] * N + [0.0], dtype=np.float32)
        prices = _prices()
        pv = om.mark_to_market(prices)
        orders = om.compute_orders(target, _tickers(), prices, pv)
        # All orders should be near-zero and filtered out
        assert len(orders) == 0

    def test_buy_order_on_increase(self):
        om = OrderManager(_exec_cfg())
        om.reset(1_000_000.0)
        # Start with 0 positions → want 20% in A0
        target = np.zeros(N + 1, dtype=np.float32)
        target[0] = 0.20
        target[-1] = 0.80
        orders = om.compute_orders(target, _tickers(), _prices(), 1_000_000.0)
        buy_orders = [o for o in orders if o.side == OrderSide.BUY]
        assert any(o.ticker == "A0" for o in buy_orders)

    def test_sell_order_on_decrease(self):
        om = OrderManager(_exec_cfg())
        om.reset(1_000_000.0)
        om._positions["A0"] = 2_000.0  # hold $200k in A0
        om._cash = 800_000.0

        # Reduce A0 to 5%
        target = np.zeros(N + 1, dtype=np.float32)
        target[0] = 0.05
        target[-1] = 0.95
        orders = om.compute_orders(target, _tickers(), _prices(), 1_000_000.0)
        sell_orders = [o for o in orders if o.side == OrderSide.SELL and o.ticker == "A0"]
        assert len(sell_orders) == 1


# ─────────────────────────────────────────────────────────────
# OrderManager — order execution
# ─────────────────────────────────────────────────────────────

class TestOrderExecution:
    def test_buy_reduces_cash(self):
        om = OrderManager(_exec_cfg(commission_bps=0, slippage_bps=0))
        om.reset(1_000_000.0)
        order = Order("A0", OrderSide.BUY, 100_000.0, pd.Timestamp("2020-01-02"))
        fills = om.fill_orders([order], {"A0": 100.0})
        assert len(fills) == 1
        assert om.cash < 1_000_000.0  # cash decreased

    def test_sell_increases_cash(self):
        om = OrderManager(_exec_cfg(commission_bps=0, slippage_bps=0))
        om.reset(1_000_000.0, {"A0": 100.0})
        order = Order("A0", OrderSide.SELL, 10_000.0, pd.Timestamp("2020-01-02"))
        fills = om.fill_orders([order], {"A0": 100.0})
        assert om.cash > 1_000_000.0

    def test_costs_charged_on_trade(self):
        """With non-zero costs, net value < gross for buys."""
        om = OrderManager(_exec_cfg(commission_bps=10, slippage_bps=10))
        om.reset(1_000_000.0)
        order = Order("A0", OrderSide.BUY, 100_000.0, pd.Timestamp("2020-01-02"))
        fills = om.fill_orders([order], {"A0": 100.0})
        f = fills[0]
        assert f.commission > 0
        assert f.slippage > 0
        assert f.net_value > f.gross_value  # buys: paid more than gross

    def test_zero_cost_fill_exact(self):
        """With no costs, cash change = gross trade value."""
        om = OrderManager(_exec_cfg(commission_bps=0, slippage_bps=0),
                          commission_bps=0, slippage_bps=0, market_impact_eta=0.0)
        om.reset(1_000_000.0)
        order = Order("A0", OrderSide.BUY, 100_000.0, pd.Timestamp("2020-01-02"))
        om.fill_orders([order], {"A0": 100.0})
        # Cash should equal initial - 100k exactly
        assert om.cash == pytest.approx(900_000.0, abs=1e-2)

    def test_fills_dataframe_structure(self):
        om = OrderManager(_exec_cfg())
        om.reset(1_000_000.0)
        order = Order("A0", OrderSide.BUY, 50_000.0, pd.Timestamp("2020-01-02"))
        om.fill_orders([order], {"A0": 100.0})
        df = om.fills_dataframe()
        assert len(df) == 1
        for col in ("date", "ticker", "side", "shares", "fill_price", "gross_value"):
            assert col in df.columns

    def test_unknown_ticker_skipped(self):
        om = OrderManager(_exec_cfg())
        om.reset(1_000_000.0)
        order = Order("UNKNOWN", OrderSide.BUY, 10_000.0, pd.Timestamp("2020-01-02"))
        fills = om.fill_orders([order], {"A0": 100.0})  # UNKNOWN not in prices
        assert len(fills) == 0


# ─────────────────────────────────────────────────────────────
# OrderManager — mark-to-market and weights
# ─────────────────────────────────────────────────────────────

class TestMarkToMarket:
    def test_mtm_cash_only(self):
        om = OrderManager(_exec_cfg())
        om.reset(500_000.0)
        pv = om.mark_to_market(_prices())
        assert pv == pytest.approx(500_000.0)

    def test_mtm_with_positions(self):
        om = OrderManager(_exec_cfg())
        om.reset(500_000.0, {"A0": 1_000.0})  # 1000 shares × $100 = $100k
        pv = om.mark_to_market(_prices(price=100.0))
        assert pv == pytest.approx(600_000.0)

    def test_weights_sum_to_one(self):
        om = OrderManager(_exec_cfg())
        om.reset(800_000.0, {"A0": 500.0, "A1": 300.0})
        w = om.get_weights(_prices(), _tickers())
        assert float(w.sum()) == pytest.approx(1.0, abs=1e-4)

    def test_weights_all_nonnegative(self):
        om = OrderManager(_exec_cfg())
        om.reset(800_000.0, {"A0": 500.0})
        w = om.get_weights(_prices(), _tickers())
        assert (w >= 0).all()


# ─────────────────────────────────────────────────────────────
# OrderManager — execution summary
# ─────────────────────────────────────────────────────────────

class TestExecutionSummary:
    def test_summary_after_trades(self):
        om = OrderManager(_exec_cfg())
        om.reset(1_000_000.0)
        orders = [
            Order("A0", OrderSide.BUY,  50_000.0, pd.Timestamp("2020-01-02")),
            Order("A1", OrderSide.BUY,  30_000.0, pd.Timestamp("2020-01-02")),
        ]
        om.fill_orders(orders, _prices())
        summary = om.execution_summary()
        assert summary["total_trades"] == 2
        assert summary["buy_count"] == 2
        assert summary["total_value_traded"] > 0
        assert summary["cost_bps"] > 0

    def test_empty_summary(self):
        om = OrderManager(_exec_cfg())
        om.reset(1_000_000.0)
        summary = om.execution_summary()
        assert summary == {}


# ─────────────────────────────────────────────────────────────
# Market impact model
# ─────────────────────────────────────────────────────────────

class TestMarketImpact:
    def test_large_order_has_more_impact(self):
        """Larger order fraction of ADV → higher slippage."""
        om_low  = OrderManager(_exec_cfg(), market_impact_eta=0.1)
        om_high = OrderManager(_exec_cfg(), market_impact_eta=0.1)

        om_low.reset(1_000_000.0)
        om_high.reset(1_000_000.0)

        adv = {"A0": 10_000_000.0}  # $10M ADV
        vol = {"A0": 0.02}          # 2% daily vol

        small_order = Order("A0", OrderSide.BUY, 10_000.0, pd.Timestamp("2020-01-02"))
        large_order = Order("A0", OrderSide.BUY, 500_000.0, pd.Timestamp("2020-01-02"))

        fills_small = om_low.fill_orders([small_order],  {"A0": 100.0}, adv=adv, daily_vols=vol)
        fills_large = om_high.fill_orders([large_order], {"A0": 100.0}, adv=adv, daily_vols=vol)

        small_slip_bps = fills_small[0].slippage / fills_small[0].gross_value * 10_000
        large_slip_bps = fills_large[0].slippage / fills_large[0].gross_value * 10_000

        assert large_slip_bps > small_slip_bps


# ─────────────────────────────────────────────────────────────
# Enhanced metrics
# ─────────────────────────────────────────────────────────────

class TestEnhancedMetrics:
    @pytest.fixture
    def daily_returns(self):
        np.random.seed(42)
        return pd.Series(np.random.normal(0.0004, 0.01, 252))

    def test_var_positive(self, daily_returns):
        from alpha0.eval.metrics import value_at_risk
        var = value_at_risk(daily_returns, 0.95)
        assert var > 0

    def test_cvar_ge_var(self, daily_returns):
        from alpha0.eval.metrics import value_at_risk, conditional_var
        var  = value_at_risk(daily_returns, 0.95)
        cvar = conditional_var(daily_returns, 0.95)
        # CVaR is always ≥ VaR
        assert cvar >= var - 1e-8

    def test_omega_ratio_gt_1_for_positive_return_series(self, daily_returns):
        from alpha0.eval.metrics import omega_ratio
        # Positive-drift series → Omega > 1
        pos_returns = pd.Series([0.001] * 100)
        assert omega_ratio(pos_returns) > 1.0

    def test_omega_ratio_lt_1_for_negative_return_series(self):
        from alpha0.eval.metrics import omega_ratio
        neg_returns = pd.Series([-0.001] * 100)
        assert omega_ratio(neg_returns) < 1.0

    def test_ulcer_index_zero_for_monotone_up(self):
        from alpha0.eval.metrics import ulcer_index
        up_returns = pd.Series([0.001] * 50)
        ui = ulcer_index(up_returns)
        assert ui == pytest.approx(0.0, abs=1e-6)

    def test_ulcer_index_positive_for_drawdown(self):
        from alpha0.eval.metrics import ulcer_index
        rets = pd.Series([0.01] * 10 + [-0.05] * 5 + [0.01] * 10)
        ui = ulcer_index(rets)
        assert ui > 0

    def test_recovery_factor(self, daily_returns):
        from alpha0.eval.metrics import recovery_factor
        rf = recovery_factor(daily_returns)
        assert isinstance(rf, float)

    def test_profit_factor_positive_series(self):
        from alpha0.eval.metrics import profit_factor
        pos = pd.Series([0.01] * 60 + [-0.005] * 40)
        pf = profit_factor(pos)
        assert pf > 1.0

    def test_information_ratio(self, daily_returns):
        from alpha0.eval.metrics import information_ratio
        benchmark = pd.Series(np.random.normal(0.0003, 0.01, 252))
        ir = information_ratio(daily_returns, benchmark)
        assert isinstance(ir, float)

    def test_compute_metrics_includes_var(self, daily_returns):
        from alpha0.eval.metrics import compute_metrics
        m = compute_metrics(daily_returns)
        assert "var_95" in m
        assert "cvar_95" in m
        assert "omega_ratio" in m
        assert "ulcer_index" in m
        assert "profit_factor" in m
        assert "skewness" in m
        assert "kurtosis" in m

    def test_max_drawdown_duration(self):
        from alpha0.eval.metrics import max_drawdown_duration
        # 10 up, 5 down, 10 up → max underwater = 5
        rets = pd.Series([0.01] * 10 + [-0.02] * 5 + [0.01] * 10)
        dur = max_drawdown_duration(rets)
        assert dur >= 5
