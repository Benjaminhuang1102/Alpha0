"""Order management system for Alpha0 paper/live trading.

Models the order life-cycle from submission through execution:

    Policy decides target weights
        → Translate to share counts (OrderManager.compute_orders)
        → Execute orders at simulated fill prices (OrderManager.fill_orders)
        → Update positions and cash balance

Market impact model
-------------------
For small orders (< 1% of ADV), basic slippage (bid-ask + commission) dominates.
For larger orders, the square-root market impact model adds additional slippage:

    impact_bps = η * σ_daily * sqrt(order_value / ADV)

where η = 0.1 (empirically calibrated), σ_daily = daily stock vol,
ADV = average daily volume in dollars.

References: Almgren et al. (2005), "Direct Estimation of Equity Market Impact".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY  = "buy"
    SELL = "sell"


@dataclass
class Order:
    """A single order submitted to the order manager.

    Attributes
    ----------
    ticker:       Asset identifier.
    side:         BUY or SELL.
    target_value: Dollar amount to trade (positive).
    date:         Trading date.
    """
    ticker: str
    side: OrderSide
    target_value: float
    date: pd.Timestamp


@dataclass
class Fill:
    """Execution record for one order.

    Attributes
    ----------
    ticker:       Asset identifier.
    side:         BUY or SELL.
    shares:       Number of shares transacted.
    fill_price:   Average execution price per share.
    gross_value:  shares × fill_price (before commission).
    commission:   Commission paid.
    slippage:     Total slippage cost (bid-ask + market impact).
    net_value:    gross_value ± commission ± slippage (cash flow).
    date:         Execution date.
    """
    ticker: str
    side: OrderSide
    shares: float
    fill_price: float
    gross_value: float
    commission: float
    slippage: float
    net_value: float
    date: pd.Timestamp


class OrderManager:
    """Manages order generation, execution simulation, and position tracking.

    Parameters
    ----------
    cfg:
        Full config dict.  Reads ``costs.*``.
    commission_bps:
        Override commission in basis points.
    slippage_bps:
        Override base slippage (bid-ask spread proxy) in basis points.
    market_impact_eta:
        Market impact coefficient η for square-root model.  0 disables it.
    """

    def __init__(
        self,
        cfg: dict,
        commission_bps: float | None = None,
        slippage_bps: float | None = None,
        market_impact_eta: float = 0.1,
    ) -> None:
        cc = cfg.get("costs", {})
        self._commission    = (commission_bps if commission_bps is not None
                               else cc.get("commission_bps", 5)) / 10_000
        self._slippage_base = (slippage_bps if slippage_bps is not None
                               else cc.get("slippage_bps", 5)) / 10_000
        self._mi_eta = market_impact_eta

        # Persistent state
        self._positions: dict[str, float] = {}   # ticker → shares
        self._cash: float = 0.0
        self._fills: list[Fill] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset(self, initial_cash: float, initial_positions: dict[str, float] | None = None) -> None:
        """Reset to a fresh state.

        Parameters
        ----------
        initial_cash:
            Starting cash balance.
        initial_positions:
            Optional initial share holdings ``{ticker: shares}``.
        """
        self._cash      = initial_cash
        self._positions = dict(initial_positions) if initial_positions else {}
        self._fills     = []

    # ------------------------------------------------------------------
    # Order generation
    # ------------------------------------------------------------------

    def compute_orders(
        self,
        target_weights: np.ndarray,
        tickers: list[str],
        prices: dict[str, float],
        portfolio_value: float,
    ) -> list[Order]:
        """Convert target portfolio weights to a list of orders.

        Parameters
        ----------
        target_weights:
            Desired weights ``(N+1,)``; last element is cash weight.
        tickers:
            List of asset names (length N, no cash).
        prices:
            Current prices ``{ticker: price}``.
        portfolio_value:
            Current total portfolio value.

        Returns
        -------
        list[Order]
            Orders to execute (zero-value orders omitted).
        """
        n = len(tickers)
        orders: list[Order] = []
        date = pd.Timestamp.today().normalize()

        for i, ticker in enumerate(tickers):
            price = prices.get(ticker, 0.0)
            if price <= 0:
                continue

            target_value = float(target_weights[i]) * portfolio_value
            current_shares = self._positions.get(ticker, 0.0)
            current_value  = current_shares * price

            delta = target_value - current_value
            if abs(delta) < 0.01 * portfolio_value * 0.001:
                # Skip trades smaller than 0.1bps of portfolio (rounding noise)
                continue

            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(
                ticker       = ticker,
                side         = side,
                target_value = abs(delta),
                date         = date,
            ))

        return orders

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def fill_orders(
        self,
        orders: list[Order],
        prices: dict[str, float],
        adv: dict[str, float] | None = None,
        daily_vols: dict[str, float] | None = None,
        date: pd.Timestamp | None = None,
    ) -> list[Fill]:
        """Simulate order execution with realistic cost model.

        Parameters
        ----------
        orders:
            Orders from :meth:`compute_orders`.
        prices:
            Execution price per ticker (typically next-day open or same-day close).
        adv:
            Average daily volume in dollars per ticker.  Used for market
            impact calculation.  If None, market impact is skipped.
        daily_vols:
            Daily return volatility per ticker.  Used for market impact.
        date:
            Execution date (defaults to today).

        Returns
        -------
        list[Fill]
            Execution records.  Positions and cash updated in-place.
        """
        exec_date = date or pd.Timestamp.today().normalize()
        fills: list[Fill] = []

        for order in orders:
            price = prices.get(order.ticker, 0.0)
            if price <= 0:
                logger.warning("No price for %s; skipping order.", order.ticker)
                continue

            shares = order.target_value / price
            gross  = shares * price

            # Cost components
            commission  = gross * self._commission
            slippage_bps = self._slippage_base

            # Market impact (square-root model)
            if adv and daily_vols and order.ticker in adv and order.ticker in daily_vols:
                adv_val  = adv[order.ticker]
                sig      = daily_vols[order.ticker]
                if adv_val > 0 and sig > 0:
                    mi_bps  = self._mi_eta * sig * np.sqrt(gross / adv_val)
                    slippage_bps += mi_bps

            slippage = gross * slippage_bps

            # Direction of slippage: buys get worse fills, sells get better
            if order.side == OrderSide.BUY:
                net_value = gross + commission + slippage
                fill_price = (gross + commission + slippage) / shares
                self._cash -= net_value
                self._positions[order.ticker] = (
                    self._positions.get(order.ticker, 0.0) + shares
                )
            else:
                net_value = gross - commission - slippage
                fill_price = net_value / shares
                self._cash += net_value
                current = self._positions.get(order.ticker, 0.0)
                self._positions[order.ticker] = max(current - shares, 0.0)

            fill = Fill(
                ticker      = order.ticker,
                side        = order.side,
                shares      = shares,
                fill_price  = fill_price,
                gross_value = gross,
                commission  = commission,
                slippage    = slippage,
                net_value   = net_value,
                date        = exec_date,
            )
            fills.append(fill)

        self._fills.extend(fills)
        return fills

    # ------------------------------------------------------------------
    # Mark-to-market
    # ------------------------------------------------------------------

    def mark_to_market(self, prices: dict[str, float]) -> float:
        """Compute current portfolio value at given prices.

        Returns
        -------
        float
            Total portfolio value = cash + sum(shares × price).
        """
        equity_value = sum(
            self._positions.get(t, 0.0) * prices.get(t, 0.0)
            for t in self._positions
        )
        return self._cash + equity_value

    def get_weights(self, prices: dict[str, float], tickers: list[str]) -> np.ndarray:
        """Return current portfolio weights as ``(N+1,)`` array.

        Last element is cash weight.
        """
        total = self.mark_to_market(prices)
        if total < 1e-6:
            n = len(tickers)
            return np.append(np.zeros(n), 1.0)

        equity = np.array([
            self._positions.get(t, 0.0) * prices.get(t, 0.0)
            for t in tickers
        ])
        cash = self._cash
        w    = np.append(equity, cash) / total
        return w.astype(np.float32)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def positions(self) -> dict[str, float]:
        return dict(self._positions)

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def fills(self) -> list[Fill]:
        return list(self._fills)

    def fills_dataframe(self) -> pd.DataFrame:
        """Return all fills as a tidy DataFrame."""
        if not self._fills:
            return pd.DataFrame(columns=[
                "date", "ticker", "side", "shares",
                "fill_price", "gross_value", "commission", "slippage", "net_value",
            ])
        rows = [
            {
                "date":        f.date,
                "ticker":      f.ticker,
                "side":        f.side.value,
                "shares":      f.shares,
                "fill_price":  f.fill_price,
                "gross_value": f.gross_value,
                "commission":  f.commission,
                "slippage":    f.slippage,
                "net_value":   f.net_value,
            }
            for f in self._fills
        ]
        return pd.DataFrame(rows)

    def execution_summary(self) -> dict[str, float]:
        """Summary statistics of all fills."""
        df = self.fills_dataframe()
        if df.empty:
            return {}
        total_traded = float(df["gross_value"].sum())
        total_costs  = float((df["commission"] + df["slippage"]).sum())
        return {
            "total_trades":       len(df),
            "total_value_traded": total_traded,
            "total_commission":   float(df["commission"].sum()),
            "total_slippage":     float(df["slippage"].sum()),
            "total_cost":         total_costs,
            "cost_bps":           10_000 * total_costs / total_traded if total_traded > 0 else 0.0,
            "buy_count":          int((df["side"] == "buy").sum()),
            "sell_count":         int((df["side"] == "sell").sum()),
        }
