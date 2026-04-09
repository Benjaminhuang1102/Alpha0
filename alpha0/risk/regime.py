"""Market regime detection for Alpha0.

Classifies the current market environment into one of four regimes based on:

1. **Trend direction** — whether the broad market is in a bull or bear phase,
   measured by the index level relative to its N-day moving average.

2. **Volatility level** — whether realised volatility is elevated or benign,
   measured by the rolling N-day annualised vol compared to a long-run median.

Regimes are used by :class:`~alpha0.risk.position_sizer.VolatilityTargeter`
and can be used to adjust position sizing and risk limits:

+-------------------+-----------+------------------+
| Regime            | Equity    | Target Vol Scale |
+===================+===========+==================+
| BULL_LOW_VOL      | +100 %    | 1.00             |
| BULL_HIGH_VOL     | +75 %     | 0.75             |
| BEAR_LOW_VOL      | +60 %     | 0.60             |
| BEAR_HIGH_VOL     | +40 %     | 0.40             |
+-------------------+-----------+------------------+

The regime history is stored so callers can inspect regime transitions.
"""

from __future__ import annotations

import logging
from collections import deque
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Four-state market regime."""
    BULL_LOW_VOL  = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL  = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"


# Position scale factors per regime
REGIME_EQUITY_SCALE: dict[MarketRegime, float] = {
    MarketRegime.BULL_LOW_VOL:  1.00,
    MarketRegime.BULL_HIGH_VOL: 0.75,
    MarketRegime.BEAR_LOW_VOL:  0.60,
    MarketRegime.BEAR_HIGH_VOL: 0.40,
}


class RegimeDetector:
    """Classifies the current market regime from a stream of daily observations.

    Parameters
    ----------
    trend_lookback:
        Days for the trend moving average.  Default 200 (classic 200-day MA).
    vol_lookback:
        Days for rolling realised volatility.  Default 21.
    vol_high_multiplier:
        If realised vol > ``vol_high_multiplier × long_run_median_vol``,
        the regime is "high vol".  Default 1.5 (50 % above long-run median).
    long_run_vol_window:
        Window for estimating the long-run median vol baseline.  Default 252.
    min_history:
        Minimum number of data points before a regime can be declared.
        Before that, returns ``BULL_LOW_VOL`` as a neutral default.
    """

    def __init__(
        self,
        trend_lookback: int = 200,
        vol_lookback: int = 21,
        vol_high_multiplier: float = 1.5,
        long_run_vol_window: int = 252,
        min_history: int = 60,
    ) -> None:
        self._trend_lookback    = trend_lookback
        self._vol_lookback      = vol_lookback
        self._vol_high_mult     = vol_high_multiplier
        self._lr_vol_window     = long_run_vol_window
        self._min_history       = min_history

        self._prices: deque[float] = deque(maxlen=max(trend_lookback, long_run_vol_window) + 10)
        self._returns: deque[float] = deque(maxlen=max(vol_lookback, long_run_vol_window) + 10)
        self._regime_history: list[MarketRegime] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, price: float) -> MarketRegime:
        """Ingest a new daily price and return the current regime.

        Parameters
        ----------
        price:
            Today's close price (or a broad market proxy, e.g. SPY).

        Returns
        -------
        MarketRegime
            Current regime classification.
        """
        if len(self._prices) > 0:
            prev = self._prices[-1]
            ret  = (price / prev - 1.0) if prev > 0 else 0.0
            self._returns.append(ret)

        self._prices.append(price)

        regime = self._classify()
        self._regime_history.append(regime)
        return regime

    def current_regime(self) -> MarketRegime:
        """Return the most recent regime without updating."""
        if self._regime_history:
            return self._regime_history[-1]
        return MarketRegime.BULL_LOW_VOL

    def equity_scale(self, regime: MarketRegime | None = None) -> float:
        """Return the equity scale factor for the given (or current) regime."""
        r = regime if regime is not None else self.current_regime()
        return REGIME_EQUITY_SCALE[r]

    def detect_from_series(
        self,
        prices: pd.Series,
    ) -> pd.Series:
        """Vectorised regime detection over a full price series.

        Parameters
        ----------
        prices:
            DatetimeIndex price series.

        Returns
        -------
        pd.Series
            Regime enum values indexed like ``prices``.
        """
        regimes: list[MarketRegime] = []
        self.reset()
        for p in prices:
            regimes.append(self.update(float(p)))
        return pd.Series(regimes, index=prices.index, name="regime")

    def reset(self) -> None:
        """Clear all history."""
        self._prices.clear()
        self._returns.clear()
        self._regime_history.clear()

    @property
    def regime_history(self) -> list[MarketRegime]:
        return list(self._regime_history)

    # ------------------------------------------------------------------
    # Internal classification
    # ------------------------------------------------------------------

    def _classify(self) -> MarketRegime:
        """Determine regime from current price/return history."""
        if len(self._prices) < self._min_history:
            return MarketRegime.BULL_LOW_VOL  # neutral default

        # --- Trend direction ---
        prices = np.array(self._prices)
        if len(prices) >= self._trend_lookback:
            ma = float(np.mean(prices[-self._trend_lookback:]))
        else:
            ma = float(np.mean(prices))

        current_price = prices[-1]
        is_bull = current_price > ma

        # --- Volatility regime ---
        rets = np.array(self._returns)
        if len(rets) < self._vol_lookback:
            return MarketRegime.BULL_LOW_VOL if is_bull else MarketRegime.BEAR_LOW_VOL

        recent_vol = float(np.std(rets[-self._vol_lookback:], ddof=1)) * np.sqrt(252)

        # Long-run median vol for comparison
        if len(rets) >= self._lr_vol_window:
            # Compute rolling 21-day vols over the long-run window
            lr_rets = rets[-self._lr_vol_window:]
            step    = self._vol_lookback
            rolling_vols = [
                float(np.std(lr_rets[i:i + step], ddof=1)) * np.sqrt(252)
                for i in range(0, len(lr_rets) - step, max(1, step // 2))
            ]
            lr_median_vol = float(np.median(rolling_vols)) if rolling_vols else recent_vol
        else:
            lr_median_vol = recent_vol  # no baseline yet

        threshold = lr_median_vol * self._vol_high_mult
        is_high_vol = recent_vol > threshold

        if is_bull and not is_high_vol:
            return MarketRegime.BULL_LOW_VOL
        elif is_bull and is_high_vol:
            return MarketRegime.BULL_HIGH_VOL
        elif not is_bull and not is_high_vol:
            return MarketRegime.BEAR_LOW_VOL
        else:
            return MarketRegime.BEAR_HIGH_VOL
