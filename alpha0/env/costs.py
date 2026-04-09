"""Transaction cost model for the Alpha0 portfolio environment.

Phase 1 uses a simple linear proportional model:

    cost = (commission_bps + slippage_bps) / 10_000 * turnover

where one-way turnover is:

    turnover = sum(|w_new - w_old|) / 2

The cash position (last element of the weight vector) is included in
turnover because moving capital from equities to cash (or vice versa) is
a real transaction that incurs costs.

A linear cost model is appropriate for liquid large-caps at the $10k–$100k
account size targeted by Alpha0.  Phase 2 can extend this to a market-impact
model proportional to sqrt(trade_size / avg_daily_volume).
"""

from __future__ import annotations

import numpy as np


class TransactionCostModel:
    """Linear proportional transaction cost model.

    Parameters
    ----------
    cfg:
        Full config dict (from ``config/default.yaml``).  Reads
        ``costs.commission_bps`` and ``costs.slippage_bps``.
    """

    def __init__(self, cfg: dict) -> None:
        cc = cfg["costs"]
        self._total_bps: float = cc["commission_bps"] + cc["slippage_bps"]
        self._total_rate: float = self._total_bps / 10_000.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_cost(
        self,
        weights_before: np.ndarray,
        weights_after: np.ndarray,
        portfolio_value: float = 1.0,
    ) -> tuple[float, float]:
        """Compute transaction cost for a rebalancing trade.

        Parameters
        ----------
        weights_before:
            Portfolio weights immediately before rebalancing,
            shape ``(n_assets + 1,)`` (last element = cash).
            Must sum to 1.0.
        weights_after:
            Target portfolio weights after rebalancing,
            same shape.  Must sum to 1.0.
        portfolio_value:
            Current portfolio value in dollars.  Included for API
            compatibility with future market-impact models; not used in
            the linear model.

        Returns
        -------
        tuple[float, float]
            ``(cost_fraction, turnover_fraction)``

            * ``cost_fraction`` — cost as a fraction of portfolio value
              (e.g. 0.001 = 10bps).
            * ``turnover_fraction`` — one-way turnover fraction
              (e.g. 1.0 = full portfolio traded once).
        """
        turnover = float(np.sum(np.abs(weights_after - weights_before))) / 2.0
        cost_fraction = self._total_rate * turnover
        return cost_fraction, turnover

    @property
    def total_bps(self) -> float:
        """Total round-trip cost in basis points (commission + slippage)."""
        return self._total_bps
