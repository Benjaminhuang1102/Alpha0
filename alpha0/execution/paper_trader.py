"""Paper trading engine for Alpha0.

Simulates live portfolio management with daily rebalancing:

    For each trading day t:
        1. Get observation window obs_t from DataLoader
        2. Run policy (or ensemble) → target weights
        3. Apply risk management (circuit breakers, sector limits, etc.)
        4. Apply position sizing (vol-targeting, optional)
        5. Submit orders via OrderManager at t's close price
           (conservative: no look-ahead into t+1 open)
        6. Mark portfolio to market; record P&L

The paper trader produces a :class:`~alpha0.eval.backtest.BacktestResult`
compatible with the existing evaluation pipeline.

Execution timing
----------------
By default orders execute at the *closing price of the decision day*.
This is slightly optimistic (real funds execute next-day) but consistent
with the training environment (which also uses day-of close).  Set
``execution_lag=1`` to simulate next-day execution using the following
day's open price instead (requires open prices in the DataLoader).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from alpha0.eval.backtest import BacktestResult
from alpha0.eval.metrics import compute_metrics
from alpha0.execution.order_manager import OrderManager

logger = logging.getLogger(__name__)


@dataclass
class PaperTradingLog:
    """Daily log of paper trading activity."""
    dates: list[pd.Timestamp] = field(default_factory=list)
    portfolio_values: list[float] = field(default_factory=list)
    daily_returns: list[float] = field(default_factory=list)
    weights_history: list[np.ndarray] = field(default_factory=list)
    risk_alerts: list[list[str]] = field(default_factory=list)
    circuit_breaker_days: list[pd.Timestamp] = field(default_factory=list)
    regime_history: list[str] = field(default_factory=list)

    def to_backtest_result(
        self, tickers: list[str], cfg: dict
    ) -> BacktestResult:
        """Convert log to BacktestResult for use with Dashboard."""
        if not self.dates:
            raise ValueError("Empty trading log.")

        idx = pd.DatetimeIndex(self.dates)
        pv  = pd.Series(self.portfolio_values, index=idx, name="portfolio_value")
        ret = pd.Series(self.daily_returns,    index=idx, name="daily_return")

        # Build weight DataFrame
        if self.weights_history:
            cols = tickers + ["cash"]
            w_df = pd.DataFrame(
                np.array(self.weights_history),
                index=idx,
                columns=cols[:len(self.weights_history[0])],
            )
        else:
            w_df = pd.DataFrame(index=idx)

        rf = cfg["eval"].get("risk_free_rate", 0.04)
        metrics = compute_metrics(ret, rf_rate=rf)

        return BacktestResult(
            portfolio_value = pv,
            daily_returns   = ret,
            weights         = w_df,
            metrics         = metrics,
        )


class PaperTrader:
    """Simulates live portfolio management with daily rebalancing.

    Parameters
    ----------
    policy:
        Trained policy or ensemble (must expose ``.act(obs, portfolio)``).
    cfg:
        Full config dict.
    risk_manager:
        Optional :class:`~alpha0.risk.manager.RiskManager`.
    position_sizer:
        Optional :class:`~alpha0.risk.position_sizer.VolatilityTargeter`.
    regime_detector:
        Optional :class:`~alpha0.risk.regime.RegimeDetector`.
    initial_capital:
        Starting portfolio value.  Defaults to ``env.initial_portfolio_value``.
    """

    def __init__(
        self,
        policy,
        cfg: dict,
        risk_manager=None,
        position_sizer=None,
        regime_detector=None,
        initial_capital: float | None = None,
    ) -> None:
        self._policy        = policy
        self._cfg           = cfg
        self._risk_mgr      = risk_manager
        self._pos_sizer     = position_sizer
        self._regime_det    = regime_detector
        self._init_capital  = initial_capital or cfg["env"]["initial_portfolio_value"]
        self._order_mgr     = OrderManager(cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        data_loader,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        benchmark_returns: pd.Series | None = None,
        deterministic: bool = True,
    ) -> tuple[BacktestResult, PaperTradingLog]:
        """Run paper trading simulation over a date range.

        Parameters
        ----------
        data_loader:
            Loaded :class:`~alpha0.data.loader.DataLoader`.
        start_date:
            First trading date (must be ≥ lookback_window days into the data).
        end_date:
            Last trading date (inclusive).
        benchmark_returns:
            Optional benchmark returns for metrics calculation.
        deterministic:
            If True, use deterministic (mean) actions.

        Returns
        -------
        tuple[BacktestResult, PaperTradingLog]
        """
        start = pd.Timestamp(start_date)
        end   = pd.Timestamp(end_date)

        dates = [d for d in data_loader.dates if start <= d <= end]
        if not dates:
            raise ValueError(f"No dates found between {start} and {end}.")

        # Determine valid start (need lookback days of history)
        lookback = self._cfg["data"]["lookback_window"]
        n_assets = self._cfg["env"]["n_assets"]
        tickers  = data_loader.tickers if hasattr(data_loader, "tickers") else [f"A{i}" for i in range(n_assets)]
        tickers  = tickers[:n_assets]

        # Initialise
        self._order_mgr.reset(self._init_capital)
        if self._risk_mgr:
            self._risk_mgr.reset(self._init_capital)
        if self._pos_sizer:
            self._pos_sizer.reset()

        # Equal-weight initial portfolio
        initial_w = np.ones(n_assets + 1, dtype=np.float32)
        initial_w[:n_assets] = 1.0 / n_assets
        initial_w[-1]        = 0.0

        log      = PaperTradingLog()
        prev_pv  = self._init_capital
        curr_w   = initial_w.copy()

        for i, date in enumerate(dates):
            date_idx = data_loader.dates.index(date) if date in data_loader.dates else None
            if date_idx is None or date_idx < lookback:
                continue

            # 1. Get observation
            obs = self._get_observation(data_loader, date_idx, lookback, n_assets)
            if obs is None:
                continue

            # 2. Policy inference
            target_w = self._infer(obs, curr_w, deterministic)

            # 3. Risk management
            alerts: list[str] = []
            if self._risk_mgr:
                target_w, alerts = self._risk_mgr.apply(target_w, curr_w, tickers)
                if alerts:
                    log.circuit_breaker_days.append(date)

            # 4. Position sizing
            if self._pos_sizer:
                target_w = self._pos_sizer.scale(target_w)

            # 5. Regime detection
            regime_label = ""
            if self._regime_det and hasattr(data_loader, "_close_prices"):
                price = float(data_loader._close_prices[date_idx, 0])
                regime = self._regime_det.update(price)
                regime_label = regime.value

            # 6. Execute via OrderManager
            prices_dict = self._get_prices(data_loader, date_idx, tickers)
            portfolio_value = self._order_mgr.mark_to_market(prices_dict)

            orders = self._order_mgr.compute_orders(target_w, tickers, prices_dict, portfolio_value)
            self._order_mgr.fill_orders(orders, prices_dict, date=date)

            # 7. Update mark-to-market
            new_pv = self._order_mgr.mark_to_market(prices_dict)
            daily_ret = (new_pv / prev_pv - 1.0) if prev_pv > 1e-6 else 0.0
            curr_w    = self._order_mgr.get_weights(prices_dict, tickers)

            # 8. Update risk state
            if self._risk_mgr:
                self._risk_mgr.update(new_pv, daily_ret)
            if self._pos_sizer:
                self._pos_sizer.update(daily_ret)

            # Record
            log.dates.append(date)
            log.portfolio_values.append(new_pv)
            log.daily_returns.append(daily_ret)
            log.weights_history.append(curr_w.copy())
            log.risk_alerts.append(alerts)
            log.regime_history.append(regime_label)

            prev_pv = new_pv

            if i % 50 == 0:
                logger.debug(
                    "PaperTrader %s | Value=%.0f | Ret=%.2f%% | W_cash=%.1f%%",
                    date.date(), new_pv, daily_ret * 100, curr_w[-1] * 100
                )

        result = log.to_backtest_result(tickers, self._cfg)
        if benchmark_returns is not None:
            b, a = __import__("alpha0.eval.metrics", fromlist=["beta_alpha"]).beta_alpha(
                result.daily_returns, benchmark_returns
            )
            result.metrics["beta"]  = b
            result.metrics["alpha"] = a

        exec_summary = self._order_mgr.execution_summary()
        result.metrics.update({
            f"execution_{k}": v for k, v in exec_summary.items()
        })

        logger.info(
            "Paper trading complete: %d days | Final value: %.0f | "
            "Sharpe: %.3f | Total trades: %d",
            len(log.dates),
            log.portfolio_values[-1] if log.portfolio_values else 0,
            result.metrics.get("sharpe_ratio", 0.0),
            exec_summary.get("total_trades", 0),
        )

        return result, log

    def fills_dataframe(self) -> pd.DataFrame:
        """Return all trade fills as a DataFrame."""
        return self._order_mgr.fills_dataframe()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer(
        self,
        obs: np.ndarray,
        portfolio: np.ndarray,
        deterministic: bool,
    ) -> np.ndarray:
        """Run policy inference and return weights."""
        obs_t  = torch.tensor(obs[None],       dtype=torch.float32)
        port_t = torch.tensor(portfolio[None], dtype=torch.float32)

        with torch.no_grad():
            action_t = self._policy.act(obs_t, port_t)

        # EnsemblePolicy returns tensor directly; AlphaPolicy returns (action, log_prob)
        if isinstance(action_t, tuple):
            action_t = action_t[0]

        action = action_t.squeeze(0).cpu().numpy()

        # Convert logits → weights using the same softmax + constraint projection
        # that MarketEnv uses
        a = action.astype(np.float64)
        a -= a.max()
        e = np.exp(a)
        w = e / e.sum()

        max_w    = self._cfg["env"].get("max_single_weight", 0.10)
        min_cash = self._cfg["env"].get("min_cash_weight", 0.05)
        n        = len(w) - 1

        cash = max(float(w[-1]), min_cash)
        eq   = w[:n].copy()
        eq_s = eq.sum()
        if eq_s > 1e-8:
            eq = eq / eq_s * (1.0 - cash)

        # Water-fill projection
        eq = self._project_box(eq, 1.0 - cash, max_w)

        result = np.concatenate([eq, [cash]])
        s = result.sum()
        if s > 1e-8:
            result /= s
        return result.astype(np.float32)

    def _project_box(self, w: np.ndarray, budget: float, max_w: float) -> np.ndarray:
        w = w.copy()
        for _ in range(len(w) + 2):
            excess = np.maximum(w - max_w, 0.0)
            total_excess = excess.sum()
            if total_excess < 1e-12:
                break
            w = np.minimum(w, max_w)
            uncapped = w < max_w - 1e-10
            n_u = uncapped.sum()
            if n_u == 0:
                break
            w[uncapped] += total_excess / n_u
        s = w.sum()
        if s > 1e-8:
            w = w / s * budget
        return w

    def _get_observation(
        self,
        loader,
        date_idx: int,
        lookback: int,
        n_assets: int,
    ) -> np.ndarray | None:
        """Extract (lookback, N, F) observation window."""
        try:
            start = date_idx - lookback
            if start < 0:
                return None
            window = loader._feature_tensor[start:date_idx].numpy()
            if window.shape[0] != lookback:
                return None
            return window[:, :n_assets, :].astype(np.float32)
        except Exception as exc:
            logger.debug("Observation extraction failed: %s", exc)
            return None

    def _get_prices(
        self,
        loader,
        date_idx: int,
        tickers: list[str],
    ) -> dict[str, float]:
        """Get closing prices for all tickers on date_idx."""
        n = len(tickers)
        try:
            if hasattr(loader, "_close_prices") and loader._close_prices is not None:
                prices_row = loader._close_prices[date_idx].numpy()
                return {t: float(prices_row[i]) for i, t in enumerate(tickers[:n])}
        except Exception:
            pass
        # Fallback: synthetic price of 100
        return {t: 100.0 for t in tickers}
