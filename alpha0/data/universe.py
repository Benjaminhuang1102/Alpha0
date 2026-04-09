"""S&P 500 universe management for Alpha0.

Fetches and filters the asset universe used for training and live trading.

SURVIVORSHIP BIAS NOTE
----------------------
Phase 1 uses the *current* S&P 500 constituent list.  This introduces
survivorship bias because stocks that were removed from the index
(due to bankruptcy, acquisition, delisting, etc.) are excluded from
the backtest.  As a result, historical performance metrics will be
optimistically biased.  A historical point-in-time constituent database
(e.g. from Tiingo or a curated CSV) is required to fully correct for this
and is deferred to Phase 2.

Mitigation applied in Phase 1:
  - Only stocks with at least ``min_history_days`` of clean data are included.
  - Universe is filtered by average dollar volume, favouring liquid large-caps
    where survivorship bias is less severe (large-caps rarely go bankrupt).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from alpha0.data.storage import ParquetStore

logger = logging.getLogger(__name__)

# Bundled fallback list shipped with the package (populated at build time).
_FALLBACK_CSV = Path(__file__).parent / "_sp500_fallback.csv"

_WIKIPEDIA_URL = (
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
)


class UniverseManager:
    """Manages the tradeable asset universe derived from the S&P 500.

    Parameters
    ----------
    cfg:
        Full config dict (loaded from ``config/default.yaml``).
    store:
        :class:`~alpha0.data.storage.ParquetStore` pointing at the
        universe directory (e.g. ``data/universe``).
    """

    CONSTITUENT_DATASET = "constituents"

    def __init__(self, cfg: dict, store: ParquetStore) -> None:
        self._cfg = cfg
        self._u = cfg["universe"]
        self._store = store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_current_constituents(self) -> list[str]:
        """Return current S&P 500 ticker symbols scraped from Wikipedia.

        Falls back to a bundled CSV if the Wikipedia request fails.

        Returns
        -------
        list[str]
            Raw ticker symbols as they appear in the index table.
            These are *not* yet validated against yfinance.
        """
        try:
            tickers = self._scrape_wikipedia()
            logger.info("Fetched %d S&P 500 constituents from Wikipedia.", len(tickers))
            return tickers
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Wikipedia scrape failed (%s). Falling back to bundled CSV.", exc
            )
            return self._load_fallback()

    def build_universe(
        self,
        tickers: list[str],
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
    ) -> list[str]:
        """Filter and rank tickers to produce the final tradeable universe.

        Selection criteria (applied in order):
        1. At least ``min_history_days`` non-NaN close prices.
        2. Average daily dollar volume >= ``min_avg_dollar_volume``.
        3. Return top ``max_assets`` by average dollar volume.

        Parameters
        ----------
        tickers:
            Candidate tickers (e.g. from :meth:`fetch_current_constituents`).
        prices:
            DataFrame of adjusted close prices, shape ``(T, N)``, columns = tickers.
        volumes:
            DataFrame of daily volumes, shape ``(T, N)``, columns = tickers.

        Returns
        -------
        list[str]
            Filtered and ranked ticker list (at most ``max_assets`` entries).
        """
        min_hist = self._u["min_history_days"]
        min_dv = self._u["min_avg_dollar_volume"]
        max_n = self._u["max_assets"]

        # Restrict to tickers present in price data
        available = [t for t in tickers if t in prices.columns]
        if len(available) < len(tickers):
            logger.info(
                "%d tickers not found in price data (dropped).",
                len(tickers) - len(available),
            )

        # 1. History filter
        history_ok = [
            t for t in available
            if prices[t].notna().sum() >= min_hist
        ]
        logger.info(
            "%d / %d tickers pass min_history_days=%d filter.",
            len(history_ok), len(available), min_hist,
        )

        # 2. Dollar volume filter
        dv_records: list[tuple[str, float]] = []
        for t in history_ok:
            if t not in volumes.columns:
                continue
            avg_dv = (prices[t] * volumes[t]).mean()
            if avg_dv >= min_dv:
                dv_records.append((t, avg_dv))

        logger.info(
            "%d / %d tickers pass min_avg_dollar_volume=%.0f filter.",
            len(dv_records), len(history_ok), min_dv,
        )

        # 3. Rank and cap
        dv_records.sort(key=lambda x: x[1], reverse=True)
        final = [t for t, _ in dv_records[:max_n]]
        logger.info("Final universe: %d tickers (max_assets=%d).", len(final), max_n)
        return final

    def save_universe(self, tickers: list[str]) -> None:
        """Persist the universe list to ``{universe_dir}/constituents.parquet``.

        Parameters
        ----------
        tickers:
            Ordered list of ticker symbols.
        """
        df = pd.DataFrame(
            {"ticker": tickers},
            index=pd.RangeIndex(len(tickers), name="rank"),
        )
        # ParquetStore expects a DatetimeIndex; use a simple wrapper
        self._store.write(df, self.CONSTITUENT_DATASET)
        logger.info("Saved universe (%d tickers).", len(tickers))

    def load_universe(self) -> list[str]:
        """Load the previously saved universe list.

        Returns
        -------
        list[str]
            Ordered list of tickers (same order as saved).

        Raises
        ------
        FileNotFoundError
            If the universe has not been saved yet.
        """
        df = self._store.read(self.CONSTITUENT_DATASET)
        tickers = df["ticker"].tolist()
        logger.info("Loaded universe (%d tickers).", len(tickers))
        return tickers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scrape_wikipedia(self) -> list[str]:
        """Scrape the S&P 500 constituent table from Wikipedia."""
        tables = pd.read_html(_WIKIPEDIA_URL, attrs={"id": "constituents"})
        df = tables[0]
        # Wikipedia column is "Symbol" but has been "Ticker symbol" historically
        col = next(
            (c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()),
            df.columns[0],
        )
        tickers: list[str] = df[col].str.strip().tolist()
        # yfinance uses '-' instead of '.' for some tickers (e.g. BRK.B → BRK-B)
        tickers = [t.replace(".", "-") for t in tickers]
        return tickers

    def _load_fallback(self) -> list[str]:
        """Load the bundled fallback CSV of S&P 500 tickers."""
        if not _FALLBACK_CSV.exists():
            raise FileNotFoundError(
                f"Fallback CSV not found at {_FALLBACK_CSV}. "
                "Please either restore internet connectivity or provide "
                "the bundled _sp500_fallback.csv file."
            )
        df = pd.read_csv(_FALLBACK_CSV)
        col = next(
            (c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()),
            df.columns[0],
        )
        tickers = df[col].str.strip().tolist()
        logger.info("Loaded %d tickers from fallback CSV.", len(tickers))
        return tickers
