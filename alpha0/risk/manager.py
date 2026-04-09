"""Portfolio risk manager for Alpha0.

Enforces risk constraints on portfolio allocations at every rebalance:

1. **Drawdown circuit breakers**
   - Soft (≥ ``max_drawdown_soft``): reduce equity to 50 %, rest to cash.
   - Hard (≥ ``max_drawdown_hard``): go to 100 % cash.
   - Cooldown period after hard trigger prevents re-entry for N days.

2. **Sector concentration limits**
   Requires an optional ``sector_map: dict[ticker, sector]``.  If present,
   no single GICS sector may exceed ``max_sector_weight``.

3. **Minimum active positions**
   At least ``min_positions`` stocks must have weight > 0.1 %.

4. **Daily turnover cap**
   Blend toward current weights if one-way turnover exceeds
   ``max_daily_turnover``.

5. **Correlation / beta alert**
   Emits a non-blocking alert if rolling portfolio beta to SPY exceeds
   ``spy_correlation_alert``.  Does NOT modify weights — purely informational.

Integration
-----------
Pass a ``RiskManager`` to ``MarketEnv``::

    risk_mgr = RiskManager(cfg)
    env = MarketEnv(cfg, loader, split="train", risk_manager=risk_mgr)

Or use it as a post-processing wrapper on any policy output::

    raw_weights = policy_output
    safe_weights, alerts = risk_mgr.apply(raw_weights, prev_weights)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    """All mutable state tracked by the risk manager.

    Reset this via :meth:`RiskManager.reset` at the start of each episode.
    """
    peak_value: float = 1.0
    current_drawdown: float = 0.0
    circuit_breaker_level: Optional[str] = None   # None | "soft" | "hard"
    cooldown_remaining: int = 0
    portfolio_value_history: list[float] = field(default_factory=list)
    recent_returns: deque = field(default_factory=lambda: deque(maxlen=63))
    alerts_history: list[list[str]] = field(default_factory=list)

    def is_in_circuit_breaker(self) -> bool:
        return self.circuit_breaker_level is not None or self.cooldown_remaining > 0


class RiskManager:
    """Stateful risk manager applied at every portfolio rebalance.

    Parameters
    ----------
    cfg:
        Full config dict.  Reads from ``cfg["risk"]``.
    sector_map:
        Optional mapping ``{ticker: sector_name}``.  Required for sector
        concentration limits; constraint is skipped if not provided.
    spy_returns:
        Optional pandas Series of SPY daily returns for beta computation.
        If None, the correlation alert is disabled.
    """

    def __init__(
        self,
        cfg: dict,
        sector_map: dict[str, str] | None = None,
        spy_returns: pd.Series | None = None,
    ) -> None:
        rc = cfg.get("risk", {})
        self._soft_dd      = rc.get("max_drawdown_soft",     0.15)
        self._hard_dd      = rc.get("max_drawdown_hard",     0.20)
        self._cooloff      = rc.get("cooloff_days",          5)
        self._max_sector   = rc.get("max_sector_weight",     0.30)
        self._min_pos      = rc.get("min_positions",         10)
        self._max_turnover = rc.get("max_daily_turnover",    0.30)
        self._spy_alert    = rc.get("spy_correlation_alert", 0.95)

        self._sector_map   = sector_map
        self._spy_returns  = spy_returns

        self._state = RiskState()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, initial_value: float = 1.0) -> None:
        """Reset all state for a new episode."""
        self._state = RiskState(peak_value=initial_value)

    def update(self, portfolio_value: float, portfolio_return: float | None = None) -> None:
        """Update state with the current portfolio value.

        Call this *after* each trading day, before the next ``apply()``.

        Parameters
        ----------
        portfolio_value:
            Current total portfolio value.
        portfolio_return:
            Daily return (used for correlation tracking).
        """
        st = self._state
        st.portfolio_value_history.append(portfolio_value)

        # Peak and drawdown
        if portfolio_value > st.peak_value:
            st.peak_value = portfolio_value
            # Coming back to a new high — reset soft breaker if in cooldown
            if st.circuit_breaker_level == "soft" and st.cooldown_remaining == 0:
                st.circuit_breaker_level = None

        st.current_drawdown = 1.0 - portfolio_value / max(st.peak_value, 1e-8)

        # Cooldown countdown
        if st.cooldown_remaining > 0:
            st.cooldown_remaining -= 1
            if st.cooldown_remaining == 0 and st.circuit_breaker_level == "hard":
                logger.info("RiskManager: hard circuit breaker cooldown ended.")
                st.circuit_breaker_level = None

        # Return history for beta calc
        if portfolio_return is not None:
            st.recent_returns.append(portfolio_return)

    def apply(
        self,
        target_weights: np.ndarray,
        current_weights: np.ndarray,
        tickers: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Apply all risk constraints to ``target_weights``.

        Parameters
        ----------
        target_weights:
            Desired weights from the policy, shape ``(N+1,)``.
            Last element is cash.
        current_weights:
            Current (drifted) weights before rebalance, same shape.
        tickers:
            Optional list of ticker names (length N, no cash).  Required
            for sector limit enforcement.

        Returns
        -------
        tuple[np.ndarray, list[str]]
            ``(safe_weights, alerts)`` — modified weights and list of
            human-readable alert strings generated this step.
        """
        alerts: list[str] = []
        w = target_weights.copy().astype(np.float64)
        st = self._state

        # 1. Circuit breaker -------------------------------------------------
        w, cb_alerts = self._apply_circuit_breaker(w, current_weights)
        alerts.extend(cb_alerts)

        # 2. Sector limits ---------------------------------------------------
        if self._sector_map and tickers:
            w, sec_alerts = self._apply_sector_limits(w, tickers)
            alerts.extend(sec_alerts)

        # 3. Minimum positions -----------------------------------------------
        equity = w[:-1]
        n_active = int((equity > 1e-3).sum())
        if n_active < self._min_pos and n_active > 0:
            # Redistribute smallest weights upward to reach min_positions
            w, mp_alerts = self._enforce_min_positions(w, n_active)
            alerts.extend(mp_alerts)

        # 4. Turnover cap (skipped when circuit breaker is forcing a direction) --
        # Circuit breakers must be able to act immediately, regardless of the
        # daily turnover limit.  The turnover cap only applies to normal trading.
        _breaker_active = (st.circuit_breaker_level is not None
                           or st.cooldown_remaining > 0)
        turnover = float(np.abs(w - current_weights).sum()) / 2.0
        if not _breaker_active and turnover > self._max_turnover + 1e-6:
            blend = self._max_turnover / turnover
            w = blend * w + (1.0 - blend) * current_weights.astype(np.float64)
            s = w.sum()
            if s > 1e-8:
                w /= s
            alerts.append(
                f"Turnover capped: {turnover:.1%} → {self._max_turnover:.1%}"
            )

        # 5. Beta alert (non-blocking) ----------------------------------------
        if len(st.recent_returns) >= 21 and self._spy_returns is not None:
            beta_alert = self._check_beta_alert()
            if beta_alert:
                alerts.append(beta_alert)

        # Persist alerts
        st.alerts_history.append(alerts)

        if alerts:
            logger.debug("RiskManager alerts: %s", alerts)

        return w.astype(np.float32), alerts

    @property
    def current_drawdown(self) -> float:
        return self._state.current_drawdown

    @property
    def circuit_breaker_level(self) -> str | None:
        return self._state.circuit_breaker_level

    @property
    def state(self) -> RiskState:
        return self._state

    def summary(self) -> dict[str, object]:
        """Return a snapshot of current risk state for logging."""
        st = self._state
        return {
            "drawdown":             round(st.current_drawdown, 4),
            "peak_value":           round(st.peak_value, 2),
            "circuit_breaker":      st.circuit_breaker_level,
            "cooldown_remaining":   st.cooldown_remaining,
            "n_alerts_total":       sum(len(a) for a in st.alerts_history),
        }

    # ------------------------------------------------------------------
    # Internal constraint methods
    # ------------------------------------------------------------------

    def _apply_circuit_breaker(
        self, w: np.ndarray, current: np.ndarray
    ) -> tuple[np.ndarray, list[str]]:
        """Enforce drawdown circuit breakers."""
        alerts: list[str] = []
        st = self._state
        dd = st.current_drawdown

        if dd >= self._hard_dd:
            if st.circuit_breaker_level != "hard":
                alerts.append(
                    f"HARD circuit breaker triggered: drawdown={dd:.1%} "
                    f"≥ {self._hard_dd:.1%}. Going to 100% cash."
                )
                logger.warning("HARD circuit breaker: DD=%.1f%%", dd * 100)
                st.circuit_breaker_level = "hard"
                st.cooldown_remaining    = self._cooloff
            # Force 100% cash
            w = np.zeros_like(w)
            w[-1] = 1.0

        elif dd >= self._soft_dd or st.circuit_breaker_level == "soft":
            if st.circuit_breaker_level != "soft" and st.cooldown_remaining == 0:
                alerts.append(
                    f"SOFT circuit breaker triggered: drawdown={dd:.1%} "
                    f"≥ {self._soft_dd:.1%}. Reducing equity by 50%."
                )
                logger.warning("SOFT circuit breaker: DD=%.1f%%", dd * 100)
                st.circuit_breaker_level = "soft"
            # Halve equity, double cash
            if st.circuit_breaker_level == "soft":
                n = len(w) - 1
                w[:n] *= 0.5
                w[-1]  = 1.0 - w[:n].sum()
                w[-1]  = max(w[-1], 0.0)
                s = w.sum()
                if s > 1e-8:
                    w /= s

        elif st.cooldown_remaining > 0:
            # In cooldown after hard breaker: maintain half-equity posture
            n = len(w) - 1
            w[:n] *= 0.5
            w[-1]  = 1.0 - w[:n].sum()
            s = w.sum()
            if s > 1e-8:
                w /= s

        return w, alerts

    def _apply_sector_limits(
        self, w: np.ndarray, tickers: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Cap any single sector at ``max_sector_weight``."""
        alerts: list[str] = []
        equity = w[:-1].copy()
        n = len(equity)

        # Build sector → [indices] mapping
        sector_indices: dict[str, list[int]] = {}
        for i, ticker in enumerate(tickers[:n]):
            sector = self._sector_map.get(ticker, "Unknown")
            sector_indices.setdefault(sector, []).append(i)

        modified = False
        for sector, idxs in sector_indices.items():
            sector_weight = equity[idxs].sum()
            if sector_weight > self._max_sector + 1e-6:
                # Scale down this sector proportionally
                scale = self._max_sector / sector_weight
                equity[idxs] *= scale
                alerts.append(
                    f"Sector '{sector}' capped: {sector_weight:.1%} → {self._max_sector:.1%}"
                )
                modified = True

        if modified:
            # Rebuild w with new equity, keep cash the same, renormalize
            total_equity = equity.sum()
            cash = max(1.0 - total_equity, w[-1])
            w_new = np.append(equity, cash)
            s = w_new.sum()
            if s > 1e-8:
                w = w_new / s
            else:
                w = w_new

        return w, alerts

    def _enforce_min_positions(
        self, w: np.ndarray, n_active: int
    ) -> tuple[np.ndarray, list[str]]:
        """Ensure at least ``min_positions`` assets have meaningful weight."""
        alerts: list[str] = []
        needed = self._min_pos - n_active
        if needed <= 0:
            return w, alerts

        equity = w[:-1].copy()
        cash   = float(w[-1])

        # Find currently zero (or tiny) positions to activate
        inactive = np.where(equity <= 1e-3)[0]
        if len(inactive) == 0:
            return w, alerts

        to_activate = inactive[:needed]
        # Give each new position a minimal weight drawn from cash
        min_w = 1e-3
        total_needed = min_w * len(to_activate)
        if total_needed > cash * 0.5:
            # Not enough cash; draw from largest positions
            top = np.argsort(equity)[::-1][:len(to_activate)]
            for i, src in enumerate(top):
                transfer = min(equity[src] * 0.05, min_w)
                equity[to_activate[i]] += transfer
                equity[src] -= transfer
        else:
            cash -= total_needed
            equity[to_activate] = min_w

        w_new = np.append(equity, max(cash, 0.0))
        s = w_new.sum()
        if s > 1e-8:
            w = w_new / s

        alerts.append(
            f"Min positions enforced: activated {len(to_activate)} positions "
            f"({n_active} → {n_active + len(to_activate)})"
        )
        return w, alerts

    def _check_beta_alert(self) -> str | None:
        """Return an alert string if rolling beta to SPY is too high."""
        if self._spy_returns is None or len(self._state.recent_returns) < 21:
            return None

        port_rets = np.array(list(self._state.recent_returns)[-21:])
        spy_window = self._spy_returns.iloc[-21:].values if len(self._spy_returns) >= 21 else None

        if spy_window is None or len(spy_window) != len(port_rets):
            return None

        cov = float(np.cov(port_rets, spy_window)[0, 1])
        spy_var = float(np.var(spy_window, ddof=1))
        if spy_var < 1e-10:
            return None

        beta = cov / spy_var
        if beta > self._spy_alert:
            return (
                f"High SPY correlation alert: rolling beta={beta:.2f} "
                f"> {self._spy_alert}"
            )
        return None
