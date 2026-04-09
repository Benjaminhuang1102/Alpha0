"""Point-in-time S&P 500 universe manager for Alpha0.

Why this matters
----------------
Using *current* S&P 500 constituents to backtest from 2005 is survivorship-
biased: every company in today's index survived — bankrupt, acquired, or
delisted firms are excluded.  This inflates backtested returns by 1–3 % per
year depending on the period.

This module reconstructs the constituent list as-of any historical date by
applying a log of additions and removals to a base list.

Data sources
------------
1. **Bundled CSV** (``data/universe/sp500_changes.csv``) — a crowd-sourced
   log of S&P 500 additions/removals.  The file ships with the repo and
   covers changes from 2000 onward.  Format::

       date,action,ticker,company
       2023-10-02,added,BX,Blackstone Inc
       2023-10-02,removed,DISH,DISH Network

2. **Wikipedia live fetch** — ``pd.read_html()`` on the S&P 500 changes
   page for any gaps.  Falls back silently on network failure.

3. **Current-only fallback** — if no change log exists, uses the current
   ``UniverseManager`` constituent list with a clear survivorship-bias
   warning.

Usage::

    mgr = HistoricalUniverseManager(cfg, cache_store)
    tickers_2015 = mgr.get_constituents("2015-01-02")
    tickers_2020 = mgr.get_constituents("2020-03-15")
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from alpha0.data.universe import UniverseManager
from alpha0.data.storage import ParquetStore

logger = logging.getLogger(__name__)

_CHANGES_CSV_URL = (
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
)
_CHANGES_STORE_KEY = "sp500_changes"


class HistoricalUniverseManager:
    """Builds point-in-time S&P 500 constituent lists.

    Parameters
    ----------
    cfg:
        Full config dict.
    universe_store:
        :class:`~alpha0.data.storage.ParquetStore` for universe data.
    changes_csv:
        Path to a CSV with columns ``[date, action, ticker, company]``.
        If not provided, looks for ``data/universe/sp500_changes.csv``
        inside the project root.
    """

    def __init__(
        self,
        cfg: dict,
        universe_store: ParquetStore,
        changes_csv: str | Path | None = None,
    ) -> None:
        self._cfg   = cfg
        self._store = universe_store
        self._base_mgr = UniverseManager(cfg, universe_store)

        # Locate the changes CSV
        if changes_csv is not None:
            self._changes_path = Path(changes_csv)
        else:
            self._changes_path = (
                Path(__file__).parent.parent.parent
                / "data" / "universe" / "sp500_changes.csv"
            )

        self._changes: pd.DataFrame | None = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_constituents(self, as_of_date: str | pd.Timestamp) -> list[str]:
        """Return the S&P 500 constituent list as of ``as_of_date``.

        Applies the change log (additions and removals) forward from the
        earliest available base list.

        Parameters
        ----------
        as_of_date:
            Date for which to reconstruct the index.

        Returns
        -------
        list[str]
            Sorted list of ticker symbols.
        """
        date = pd.Timestamp(as_of_date)
        changes = self._load_changes()

        if changes is None or changes.empty:
            logger.warning(
                "No historical change log available. Using current constituents. "
                "SURVIVORSHIP BIAS WARNING: backtest results will be inflated."
            )
            return self._base_mgr.fetch_current_constituents()

        return self._reconstruct_as_of(changes, date)

    def build_date_universe_map(
        self,
        dates: list[str | pd.Timestamp],
    ) -> dict[str, list[str]]:
        """Pre-compute constituent lists for each date in ``dates``.

        Efficient: processes changes once and snapshots at each date.

        Returns
        -------
        dict[str, list[str]]
            ``{date_str: [tickers]}``
        """
        dates_ts = sorted(pd.Timestamp(d) for d in dates)
        changes  = self._load_changes()

        if changes is None or changes.empty:
            logger.warning("No change log; returning current constituents for all dates.")
            current = self._base_mgr.fetch_current_constituents()
            return {str(d.date()): list(current) for d in dates_ts}

        result: dict[str, list[str]] = {}
        for d in dates_ts:
            result[str(d.date())] = self._reconstruct_as_of(changes, d)
        return result

    def save_changes_cache(self) -> None:
        """Persist the loaded change log to the universe store."""
        df = self._load_changes()
        if df is not None and not df.empty:
            self._store.write(df, _CHANGES_STORE_KEY)
            logger.info("Saved S&P 500 change log (%d records).", len(df))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_changes(self) -> pd.DataFrame | None:
        """Load and cache the constituent change log."""
        if self._changes is not None:
            return self._changes

        # Try bundled CSV first
        if self._changes_path.exists():
            try:
                df = pd.read_csv(self._changes_path, parse_dates=["date"])
                df.columns = df.columns.str.lower().str.strip()
                df = df[["date", "action", "ticker"]].dropna()
                df["ticker"] = df["ticker"].str.upper().str.strip()
                df["action"] = df["action"].str.lower().str.strip()
                df = df.sort_values("date")
                self._changes = df
                logger.info("Loaded S&P 500 change log from %s (%d rows).", self._changes_path, len(df))
                return df
            except Exception as exc:
                logger.warning("Failed to load changes CSV: %s", exc)

        # Try stored cache
        if self._store.exists(_CHANGES_STORE_KEY):
            try:
                df = self._store.read(_CHANGES_STORE_KEY)
                self._changes = df
                return df
            except Exception as exc:
                logger.warning("Failed to load changes cache: %s", exc)

        # Try Wikipedia live fetch
        df = self._fetch_wikipedia_changes()
        if df is not None:
            self._changes = df
            return df

        return None

    def _fetch_wikipedia_changes(self) -> pd.DataFrame | None:
        """Attempt to scrape S&P 500 change history from Wikipedia."""
        try:
            tables = pd.read_html(_CHANGES_CSV_URL, header=0)
            # Wikipedia typically has the changes table as the 2nd table (index 1)
            for tbl in tables[1:3]:
                cols = [str(c).lower() for c in tbl.columns]
                if any("date" in c for c in cols) and any("tick" in c for c in cols):
                    tbl.columns = cols
                    break
            else:
                logger.debug("Could not find changes table on Wikipedia.")
                return None

            # Parse date column
            date_col = next((c for c in tbl.columns if "date" in c), None)
            if date_col is None:
                return None

            rows = []
            for _, row in tbl.iterrows():
                d = pd.to_datetime(row[date_col], errors="coerce")
                if pd.isna(d):
                    continue

                added_col   = next((c for c in tbl.columns if "add" in c and "tick" in c), None)
                removed_col = next((c for c in tbl.columns if "remov" in c and "tick" in c), None)

                if added_col and pd.notna(row[added_col]) and str(row[added_col]).strip():
                    rows.append({"date": d, "action": "added", "ticker": str(row[added_col]).strip()})
                if removed_col and pd.notna(row[removed_col]) and str(row[removed_col]).strip():
                    rows.append({"date": d, "action": "removed", "ticker": str(row[removed_col]).strip()})

            if not rows:
                return None

            df = pd.DataFrame(rows).sort_values("date")
            logger.info("Fetched %d S&P 500 changes from Wikipedia.", len(df))

            # Persist to CSV for future use
            self._changes_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self._changes_path, index=False)

            return df

        except Exception as exc:
            logger.warning("Wikipedia fetch failed: %s", exc)
            return None

    def _reconstruct_as_of(
        self, changes: pd.DataFrame, date: pd.Timestamp
    ) -> list[str]:
        """Apply the change log up to ``date`` to reconstruct the index.

        Strategy:
        - Start from the current constituent list as a base.
        - *Reverse* all changes that occurred *after* ``date``:
          additions after ``date`` → remove them
          removals after ``date`` → add them back
        This effectively walks the index backwards in time.
        """
        # Get current constituents as base
        current = set(self._base_mgr.fetch_current_constituents())

        # Reverse future changes
        future_changes = changes[changes["date"] > date].sort_values("date", ascending=False)

        constituents = set(current)
        for _, row in future_changes.iterrows():
            ticker = str(row["ticker"]).upper()
            action = str(row["action"]).lower()
            if action == "added":
                # This addition happened after our date → reverse it (remove)
                constituents.discard(ticker)
            elif action == "removed":
                # This removal happened after our date → reverse it (add back)
                constituents.add(ticker)

        return sorted(constituents)
