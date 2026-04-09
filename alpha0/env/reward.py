"""Reward function for the Alpha0 market environment.

The reward at each step is:

    R_t = log_return_t - risk_penalty_t

where:

    log_return_t    = log(portfolio_value_t / portfolio_value_{t-1})
                      (transaction costs are already reflected in
                       portfolio_value_t via the env's step logic)

    risk_penalty_t  = risk_penalty_scale
                      * annualised_realised_vol(last risk_penalty_window steps)

The risk penalty encourages the agent to maintain low-volatility portfolios
without requiring a separate explicit volatility constraint.  It uses only
backward-looking realised volatility (no lookahead).

Note on cost accounting: transaction costs are deducted from portfolio
value inside :class:`~alpha0.env.market_env.MarketEnv` *before* the
log-return is computed.  Therefore, costs are automatically embedded in
``log_return_t`` and do not need a separate penalty term.  The reward
function receives the post-cost return.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


class RewardFunction:
    """Computes the scalar reward for each environment step.

    Parameters
    ----------
    cfg:
        Full config dict (from ``config/default.yaml``).  Reads
        ``env.reward.*``.
    trading_days_per_year:
        Used to annualise the realised volatility penalty.
        Defaults to 252.
    """

    def __init__(
        self,
        cfg: dict,
        trading_days_per_year: int = 252,
    ) -> None:
        rc = cfg["env"]["reward"]
        self._scale: float = rc.get("daily_return_scale", 1.0)
        self._risk_scale: float = rc["risk_penalty_scale"]
        self._risk_window: int = rc["risk_penalty_window"]
        self._ann_factor: float = math.sqrt(trading_days_per_year)
        self._return_history: deque[float] = deque(maxlen=self._risk_window)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear return history at the start of a new episode."""
        self._return_history.clear()

    def compute(
        self,
        log_return: float,
        portfolio_value: float = 1.0,
    ) -> tuple[float, dict[str, float]]:
        """Compute the reward for one environment step.

        Parameters
        ----------
        log_return:
            Log return of the portfolio for this step
            (post transaction cost).
        portfolio_value:
            Current portfolio value (for logging only; not used in
            the reward calculation).

        Returns
        -------
        tuple[float, dict[str, float]]
            ``(scalar_reward, info_dict)``

            ``info_dict`` keys: ``log_return``, ``risk_penalty``,
            ``realised_vol``, ``portfolio_value``.
        """
        self._return_history.append(log_return)

        realised_vol = self._realised_vol()
        risk_penalty = self._risk_scale * realised_vol

        reward = self._scale * log_return - risk_penalty

        info = {
            "log_return": log_return,
            "risk_penalty": risk_penalty,
            "realised_vol": realised_vol,
            "portfolio_value": portfolio_value,
        }
        return reward, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _realised_vol(self) -> float:
        """Annualised realised volatility from the return history buffer."""
        if len(self._return_history) < 2:
            return 0.0
        arr = np.array(self._return_history)
        return float(np.std(arr, ddof=1)) * self._ann_factor
