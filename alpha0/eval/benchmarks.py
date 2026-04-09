"""Benchmark strategies for Alpha0 performance comparison.

Provides three reference strategies:
- SPY buy-and-hold
- Equal-weight rebalanced portfolio
- 60/40 equity/bond portfolio (SPY + TLT)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from alpha0.eval.backtest import Backtest, BacktestResult
from alpha0.eval.metrics import compute_metrics

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs baseline benchmark strategies for comparison with Alpha0.

    Parameters
    ----------
    cfg:
        Full config dict.
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._backtest = Backtest(cfg)
        self._trading_days: int = cfg["eval"]["trading_days_per_year"]
        self._rf_rate: float = cfg["eval"]["risk_free_rate"]

    # ------------------------------------------------------------------
    # Individual benchmarks
    # ------------------------------------------------------------------

    def spy_buyhold(
        self,
        prices: pd.Series,
    ) -> BacktestResult:
        """SPY buy-and-hold benchmark.

        Parameters
        ----------
        prices:
            SPY adjusted close price series (DatetimeIndex).

        Returns
        -------
        BacktestResult
        """
        returns = prices.pct_change().dropna().rename("SPY")
        # Weight is always 100% SPY
        weights = pd.DataFrame(
            {"SPY": np.ones(len(returns))},
            index=returns.index,
        )
        prices_df = prices.to_frame("SPY").reindex(returns.index)
        return self._backtest.run(weights, prices_df)

    def equal_weight(
        self,
        prices: pd.DataFrame,
        rebalance_freq: str = "monthly",
    ) -> BacktestResult:
        """Equal-weight portfolio rebalanced at the specified frequency.

        Parameters
        ----------
        prices:
            Adjusted close prices, shape ``(T, N)``, columns = tickers.
        rebalance_freq:
            One of ``"daily"``, ``"weekly"``, ``"monthly"``.

        Returns
        -------
        BacktestResult
        """
        # Drop tickers with all-NaN prices
        prices = prices.dropna(axis=1, how="all").ffill()
        n = prices.shape[1]
        if n == 0:
            raise ValueError("No valid tickers in prices DataFrame.")

        returns = prices.pct_change().dropna()

        # Build weight matrix: equal weight on rebalance days, drifted otherwise
        rebalance_mask = self._rebalance_mask(returns.index, rebalance_freq)
        weights = self._build_ew_weights(n, returns, rebalance_mask)
        weights.columns = prices.columns

        return self._backtest.run(weights, prices.reindex(returns.index))

    def sixty_forty(
        self,
        equity_prices: pd.Series,
        bond_prices: pd.Series,
        rebalance_freq: str = "monthly",
    ) -> BacktestResult:
        """60/40 equity/bond portfolio (e.g. SPY/TLT), monthly rebalance.

        Parameters
        ----------
        equity_prices:
            Equity price series (e.g. SPY).
        bond_prices:
            Bond price series (e.g. TLT).
        rebalance_freq:
            Rebalance frequency.

        Returns
        -------
        BacktestResult
        """
        prices = pd.DataFrame(
            {"equity": equity_prices, "bond": bond_prices}
        ).dropna().ffill()
        returns = prices.pct_change().dropna()

        rebalance_mask = self._rebalance_mask(returns.index, rebalance_freq)
        target = np.array([0.60, 0.40])

        rows: list[np.ndarray] = []
        prev = target.copy()
        for i, is_rebalance in enumerate(rebalance_mask):
            if is_rebalance:
                w = target.copy()
            else:
                # Drift: apply today's returns to prev weights
                ret = returns.iloc[i].values
                w = prev * (1.0 + ret)
                s = w.sum()
                w = w / s if s > 1e-8 else target.copy()
            rows.append(w)
            prev = w

        weights = pd.DataFrame(rows, index=returns.index, columns=["equity", "bond"])
        return self._backtest.run(weights, prices.reindex(returns.index))

    def run_all(
        self,
        prices: pd.DataFrame,
        spy_prices: pd.Series | None = None,
        bond_prices: pd.Series | None = None,
    ) -> dict[str, BacktestResult]:
        """Run all available benchmarks and return a comparison dict.

        Parameters
        ----------
        prices:
            Full price DataFrame for the universe.
        spy_prices:
            SPY close prices.  If None, uses the first column of ``prices``
            as a proxy.
        bond_prices:
            Bond proxy prices (e.g. TLT).  If None, 60/40 is skipped.

        Returns
        -------
        dict[str, BacktestResult]
            Keys: ``"spy_buyhold"``, ``"equal_weight"``,
            ``"sixty_forty"`` (if bond_prices provided).
        """
        results: dict[str, BacktestResult] = {}

        spy = spy_prices if spy_prices is not None else prices.iloc[:, 0]
        spy = spy.dropna()

        try:
            results["spy_buyhold"] = self.spy_buyhold(spy)
        except Exception as exc:
            logger.warning("spy_buyhold failed: %s", exc)

        try:
            results["equal_weight"] = self.equal_weight(prices)
        except Exception as exc:
            logger.warning("equal_weight failed: %s", exc)

        if bond_prices is not None:
            try:
                results["sixty_forty"] = self.sixty_forty(spy, bond_prices)
            except Exception as exc:
                logger.warning("sixty_forty failed: %s", exc)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebalance_mask(
        self, idx: pd.DatetimeIndex, freq: str
    ) -> np.ndarray:
        """Boolean array: True on rebalance days."""
        if freq == "daily":
            return np.ones(len(idx), dtype=bool)
        elif freq == "weekly":
            return np.array([d.weekday() == 0 for d in idx])
        elif freq == "monthly":
            # First trading day of each month
            mask = np.zeros(len(idx), dtype=bool)
            prev_month = -1
            for i, d in enumerate(idx):
                if d.month != prev_month:
                    mask[i] = True
                    prev_month = d.month
            return mask
        else:
            raise ValueError(f"Unknown rebalance_freq: {freq!r}")

    def _build_ew_weights(
        self,
        n_assets: int,
        returns: pd.DataFrame,
        rebalance_mask: np.ndarray,
    ) -> pd.DataFrame:
        """Build equal-weight weight matrix with drift between rebalances."""
        target = np.ones(n_assets) / n_assets
        rows: list[np.ndarray] = []
        prev = target.copy()
        for i, is_rebalance in enumerate(rebalance_mask):
            if is_rebalance:
                w = target.copy()
            else:
                ret = returns.iloc[i].values
                w = prev * (1.0 + ret)
                s = w.sum()
                w = w / s if s > 1e-8 else target.copy()
            rows.append(w)
            prev = w
        return pd.DataFrame(rows, index=returns.index)
