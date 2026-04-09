"""Vectorised backtest engine for Alpha0 strategy evaluation.

This module is for *offline* backtest evaluation of trained policies.
It is separate from :class:`~alpha0.env.market_env.MarketEnv`, which is
used during online RL training.

Usage::

    bt = Backtest(cfg)
    result = bt.run(weights, prices)
    print(result.metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha0.eval.metrics import compute_metrics


@dataclass
class BacktestResult:
    """Container for backtest outputs.

    Attributes
    ----------
    portfolio_value:
        Daily portfolio value series (DatetimeIndex).
    daily_returns:
        Daily arithmetic return series.
    weights:
        Weight matrix ``(T, N)`` used during the backtest.
    metrics:
        Flat dict of performance metrics for the strategy.
    benchmark_metrics:
        Dict of ``{benchmark_name: metrics_dict}`` for comparisons.
    """

    portfolio_value: pd.Series
    daily_returns: pd.Series
    weights: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)
    benchmark_metrics: dict[str, dict[str, float]] = field(default_factory=dict)


class Backtest:
    """Vectorised backtest engine.

    Applies a weight matrix to a price matrix, deducts transaction costs
    on rebalance days, and computes performance metrics.

    Parameters
    ----------
    cfg:
        Full config dict.  Reads ``costs.*`` and ``eval.*``.
    """

    def __init__(self, cfg: dict) -> None:
        cc = cfg["costs"]
        self._total_rate: float = (cc["commission_bps"] + cc["slippage_bps"]) / 10_000.0
        ec = cfg["eval"]
        self._rf_rate: float = ec["risk_free_rate"]
        self._trading_days: int = ec["trading_days_per_year"]
        self._init_capital: float = ec["backtest"]["initial_capital"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        weights: pd.DataFrame,
        prices: pd.DataFrame,
        benchmark_returns: pd.Series | None = None,
        initial_capital: float | None = None,
    ) -> BacktestResult:
        """Run a vectorised backtest.

        Parameters
        ----------
        weights:
            Target portfolio weights, shape ``(T, N)``.  Each row must
            sum to 1.0.  The last column may optionally represent cash
            (zero return); if so it should be named ``"cash"``.  If the
            number of columns in ``weights`` exceeds ``prices``, the
            extra columns (e.g. cash) are handled correctly.
        prices:
            Adjusted close prices, shape ``(T, N)``.  Columns must
            match the equity columns of ``weights``.
        benchmark_returns:
            Optional benchmark return series for beta/alpha metrics.
        initial_capital:
            Starting portfolio value.  Defaults to
            ``eval.backtest.initial_capital`` from config.

        Returns
        -------
        BacktestResult
        """
        capital = initial_capital if initial_capital is not None else self._init_capital

        # Align weights and prices to common dates/tickers
        equity_cols = [c for c in weights.columns if c != "cash"]
        prices_aligned = prices.reindex(
            columns=equity_cols, index=weights.index
        ).ffill()

        # Compute daily asset returns
        asset_returns = prices_aligned.pct_change().fillna(0.0)

        n_days = len(weights)
        portfolio_values = np.zeros(n_days + 1)
        portfolio_values[0] = capital

        prev_weights = pd.Series(0.0, index=weights.columns)

        for t in range(n_days):
            pv = portfolio_values[t]
            w_today = weights.iloc[t]

            # Transaction cost on turnover vs previous weights
            turnover = float((w_today - prev_weights).abs().sum()) / 2.0
            cost = self._total_rate * turnover
            pv *= (1.0 - cost)

            # Apply asset returns
            eq_w = w_today[equity_cols].values
            ret = asset_returns.iloc[t].values
            port_return = float(np.dot(eq_w, ret))
            pv *= (1.0 + port_return)
            pv = max(pv, 1e-6)

            portfolio_values[t + 1] = pv

            # Drifted weights (simplified — renormalise after returns)
            eq_drifted = eq_w * (1.0 + ret)
            if "cash" in w_today.index:
                cash_w = float(w_today["cash"])
                all_drifted = np.append(eq_drifted, cash_w)
                col_order = equity_cols + ["cash"]
            else:
                all_drifted = eq_drifted
                col_order = equity_cols
            s = all_drifted.sum()
            prev_weights = pd.Series(
                all_drifted / s if s > 1e-8 else w_today.values,
                index=col_order,
            )

        # Build output series
        idx = weights.index
        port_series = pd.Series(portfolio_values[1:], index=idx, name="portfolio_value")
        port_prev   = pd.Series(portfolio_values[:-1], index=idx)
        daily_ret   = (port_series / port_prev - 1.0).rename("daily_return")

        metrics = compute_metrics(
            daily_ret,
            benchmark_returns=benchmark_returns,
            weights=weights[equity_cols],
            rf_rate=self._rf_rate,
            trading_days=self._trading_days,
        )

        return BacktestResult(
            portfolio_value=port_series,
            daily_returns=daily_ret,
            weights=weights,
            metrics=metrics,
        )
