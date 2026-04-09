"""Position sizing utilities for Alpha0.

Two complementary approaches:

1. **VolatilityTargeter** — scales the entire portfolio so that realised
   volatility matches a target level (e.g. 10 % annualised).  Robust,
   widely used in systematic funds.  Cash acts as the buffer: when vol is
   high the equity fraction shrinks and excess goes to cash.

2. **KellySizer** — applies fractional Kelly criterion to scale position
   sizes according to estimated edge.  Uses the Sharpe of recent returns
   as a proxy for edge.  More aggressive than vol-targeting but can be
   unstable; 50 % Kelly (half-Kelly) is a common institutional choice.

Both return a *modified* weight array of the same shape as the input
with the equity fraction rescaled and cash adjusted to maintain sum=1.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class VolatilityTargeter:
    """Scale portfolio weights to hit a target annualised volatility.

    The scaler is computed from recent portfolio returns.  When the realised
    vol is above target, equity is trimmed.  When below, equity is scaled
    up, subject to ``max_leverage`` (default 1.0 = no leverage).

    If fewer than ``min_history`` returns are available, the weights are
    returned unchanged.

    Parameters
    ----------
    target_annual_vol:
        Desired annualised portfolio volatility (e.g. 0.10 = 10%).
    lookback:
        Rolling window for realised vol estimation (trading days).
    max_leverage:
        Maximum allowed equity fraction (1.0 = long-only, no leverage).
    min_cash:
        Minimum cash fraction to maintain after scaling.
    trading_days:
        Trading days per year for annualisation.
    min_history:
        Minimum observations before vol-targeting kicks in.
    """

    def __init__(
        self,
        target_annual_vol: float = 0.10,
        lookback: int = 21,
        max_leverage: float = 1.0,
        min_cash: float = 0.05,
        trading_days: int = 252,
        min_history: int = 21,
    ) -> None:
        self._target_vol  = target_annual_vol
        self._lookback    = lookback
        self._max_lev     = max_leverage
        self._min_cash    = min_cash
        self._tdays       = trading_days
        self._min_history = min_history

        self._returns: deque[float] = deque(maxlen=lookback + 5)

    def update(self, portfolio_return: float) -> None:
        """Feed in today's portfolio return."""
        self._returns.append(portfolio_return)

    def scale(self, weights: np.ndarray) -> np.ndarray:
        """Rescale ``weights`` to target volatility.

        Parameters
        ----------
        weights:
            Current weight vector ``(N+1,)``.  Last element is cash.

        Returns
        -------
        np.ndarray
            Rescaled weights, shape ``(N+1,)``, sums to 1.0.
        """
        if len(self._returns) < self._min_history:
            return weights.copy()

        rets = np.array(self._returns)
        realised_vol = float(np.std(rets, ddof=1)) * np.sqrt(self._tdays)

        if realised_vol < 1e-8:
            return weights.copy()

        raw_scale = self._target_vol / realised_vol

        # Clamp to [0, max_leverage]
        scale = float(np.clip(raw_scale, 0.0, self._max_lev))

        w = weights.copy().astype(np.float64)
        n = len(w) - 1  # number of equity positions

        equity = w[:n] * scale
        equity_sum = equity.sum()

        # If scaling would violate min_cash, trim equity
        if 1.0 - equity_sum < self._min_cash:
            equity_sum_max = 1.0 - self._min_cash
            if equity_sum > 1e-8:
                equity = equity * (equity_sum_max / equity_sum)
            equity_sum = equity.sum()

        cash = 1.0 - equity_sum
        w_new = np.append(equity, cash)

        total = w_new.sum()
        if total > 1e-8:
            w_new /= total

        return w_new.astype(np.float32)

    def reset(self) -> None:
        self._returns.clear()

    @property
    def current_scale(self) -> float | None:
        """Return the current scale factor (or None if not enough history)."""
        if len(self._returns) < self._min_history:
            return None
        rets = np.array(self._returns)
        rv = float(np.std(rets, ddof=1)) * np.sqrt(self._tdays)
        if rv < 1e-8:
            return None
        return float(np.clip(self._target_vol / rv, 0.0, self._max_lev))


class KellySizer:
    """Fractional Kelly position sizer.

    Uses the rolling Sharpe ratio of recent portfolio returns as a proxy
    for the Kelly fraction, then multiplies by ``kelly_fraction`` (0.5 =
    half-Kelly, the standard institutional choice).

    Kelly fraction = Sharpe² / (annualised vol)  (simplified continuous-time)

    In practice this is equivalent to a more aggressive vol-targeting scheme
    capped at ``max_leverage``.

    Parameters
    ----------
    kelly_fraction:
        Fraction of the full Kelly bet to use.  0.5 = half-Kelly.
    lookback:
        Rolling window for Sharpe estimation.
    max_leverage:
        Maximum equity fraction.
    trading_days:
        Trading days per year.
    min_history:
        Minimum observations before Kelly sizing kicks in.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.50,
        lookback: int = 63,
        max_leverage: float = 1.0,
        trading_days: int = 252,
        min_history: int = 42,
    ) -> None:
        self._kelly_frac  = kelly_fraction
        self._lookback    = lookback
        self._max_lev     = max_leverage
        self._tdays       = trading_days
        self._min_history = min_history

        self._returns: deque[float] = deque(maxlen=lookback + 5)

    def update(self, portfolio_return: float) -> None:
        self._returns.append(portfolio_return)

    def scale(self, weights: np.ndarray) -> np.ndarray:
        """Scale weights using fractional Kelly."""
        if len(self._returns) < self._min_history:
            return weights.copy()

        rets = np.array(self._returns)
        mu   = float(np.mean(rets)) * self._tdays
        vol  = float(np.std(rets, ddof=1)) * np.sqrt(self._tdays)

        if vol < 1e-8:
            return weights.copy()

        # Simplified Kelly: f* = mu / vol²  (continuous-time approximation)
        kelly_full = mu / (vol ** 2)
        kelly_frac = self._kelly_frac * kelly_full
        scale = float(np.clip(kelly_frac, 0.0, self._max_lev))

        w = weights.copy().astype(np.float64)
        n = len(w) - 1
        equity = w[:n] * scale
        cash   = 1.0 - equity.sum()
        w_new  = np.append(equity, max(cash, 0.0))

        total = w_new.sum()
        if total > 1e-8:
            w_new /= total

        return w_new.astype(np.float32)

    def reset(self) -> None:
        self._returns.clear()
