"""Market data fetching for Alpha0.

Downloads OHLCV from Yahoo Finance (via yfinance) and macro series from
the Federal Reserve Economic Data API (via fredapi).  All downloads are
cached to local parquet files so subsequent runs avoid network calls.

FRED API KEY
------------
``fetch_fred_series`` requires a free FRED API key.  Register at:
    https://fred.stlouisfed.org/docs/api/api_key.html

Set the key via the environment variable ``FRED_API_KEY`` before running
any data pipeline that fetches FRED series.

Fallback: VIX and the 10-year yield are also available via yfinance
(``^VIX`` and ``^TNX``).  Set ``use_yfinance_fallback=True`` on
:class:`DataFetcher` to use those instead of FRED for testing without
an API key.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from alpha0.data.storage import ParquetStore

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches and caches OHLCV + macro data for the Alpha0 pipeline.

    Parameters
    ----------
    cfg:
        Full config dict (from ``config/default.yaml``).
    cache_store:
        :class:`~alpha0.data.storage.ParquetStore` pointing at the cache
        directory (e.g. ``data/cache``).
    """

    def __init__(self, cfg: dict, cache_store: ParquetStore) -> None:
        self._cfg = cfg
        self._dc = cfg["data"]
        self._store = cache_store
        self._chunk_size: int = self._dc.get("fetch_chunk_size", 50)
        self._chunk_delay: float = self._dc.get("fetch_chunk_delay", 2.0)
        self._max_retries: int = self._dc.get("fetch_max_retries", 3)

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        tickers: list[str],
        start: str,
        end: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Download adjusted OHLCV for all tickers.

        Returns a wide DataFrame with a two-level column MultiIndex
        ``(field, ticker)`` where ``field`` ∈ {Open, High, Low, Close, Volume}.

        Results are cached per-ticker in ``cache/ohlcv/{TICKER}.parquet``.
        On subsequent calls, only tickers missing from cache are fetched.

        Parameters
        ----------
        tickers:
            List of yfinance-compatible ticker symbols.
        start:
            Start date string (inclusive), e.g. ``"2005-01-01"``.
        end:
            End date string (inclusive), e.g. ``"2024-12-31"``.
        force_refresh:
            If True, ignore cache and re-fetch everything.

        Returns
        -------
        pd.DataFrame
            MultiIndex columns ``(field, ticker)``, DatetimeIndex.
        """
        import yfinance as yf

        to_fetch: list[str] = []
        cached_frames: dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            cache_key = f"ohlcv/{ticker}"
            if not force_refresh and self._store.exists(cache_key):
                cached_frames[ticker] = self._store.read(cache_key)
            else:
                to_fetch.append(ticker)

        if to_fetch:
            logger.info("Fetching OHLCV for %d tickers via yfinance…", len(to_fetch))
            chunks = [
                to_fetch[i : i + self._chunk_size]
                for i in range(0, len(to_fetch), self._chunk_size)
            ]
            for idx, chunk in enumerate(chunks):
                logger.debug(
                    "  Chunk %d/%d: %d tickers", idx + 1, len(chunks), len(chunk)
                )
                raw = self._download_with_retry(
                    lambda c=chunk: yf.download(
                        c,
                        start=start,
                        end=end,
                        auto_adjust=True,
                        progress=False,
                        threads=True,
                    )
                )
                if raw is None or raw.empty:
                    logger.warning("Chunk %d returned empty data.", idx + 1)
                    continue

                # Normalise column levels to ("field", "ticker").
                # Modern yfinance (≥0.2.50) always returns MultiIndex
                # with levels named ["Price", "Ticker"].
                if isinstance(raw.columns, pd.MultiIndex):
                    names = raw.columns.names
                    # Detect level order: either (Price, Ticker) or (Ticker, Price)
                    price_level = next(
                        (i for i, n in enumerate(names)
                         if str(n).lower() in ("price", "field")), 0
                    )
                    ticker_level = 1 - price_level
                    raw.columns = pd.MultiIndex.from_tuples(
                        [
                            (raw.columns[i][price_level], raw.columns[i][ticker_level])
                            for i in range(len(raw.columns))
                        ],
                        names=["field", "ticker"],
                    )
                else:
                    # Flat columns — single ticker fallback
                    if len(chunk) == 1:
                        raw.columns = pd.MultiIndex.from_tuples(
                            [(col, chunk[0]) for col in raw.columns],
                            names=["field", "ticker"],
                        )

                # Cache per-ticker
                for ticker in chunk:
                    ticker_vals = raw.columns.get_level_values("ticker")
                    if ticker not in ticker_vals:
                        logger.warning("Ticker %s not in downloaded data.", ticker)
                        continue
                    df_t = raw.xs(ticker, axis=1, level="ticker")
                    df_t = df_t.dropna(how="all")
                    if df_t.empty:
                        logger.warning("Ticker %s: all NaN — skipping cache.", ticker)
                        continue
                    self._store.write(df_t, f"ohlcv/{ticker}")
                    cached_frames[ticker] = df_t

                if idx < len(chunks) - 1:
                    time.sleep(self._chunk_delay)

        # Assemble wide MultiIndex DataFrame
        frames = {}
        for ticker, df_t in cached_frames.items():
            frames[ticker] = df_t

        if not frames:
            logger.warning("No OHLCV data assembled — all fetches failed.")
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1)
        combined.columns.names = ["ticker", "field"]
        combined = combined.swaplevel(axis=1).sort_index(axis=1)
        combined.index = pd.to_datetime(combined.index)
        combined.sort_index(inplace=True)
        return combined

    # ------------------------------------------------------------------
    # FRED macro series
    # ------------------------------------------------------------------

    def fetch_fred_series(
        self,
        series: list[dict],
        start: str,
        end: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch macro time series from the FRED API.

        Parameters
        ----------
        series:
            List of dicts with keys ``id`` (FRED series ID) and ``alias``
            (column name in output).  Mirrors ``data.fred_series`` in config.
        start:
            Start date string, e.g. ``"2005-01-01"``.
        end:
            End date string, e.g. ``"2024-12-31"``.
        force_refresh:
            If True, ignore cache.

        Returns
        -------
        pd.DataFrame
            One column per alias, DatetimeIndex (trading calendar aligned
            via forward-fill).

        Raises
        ------
        EnvironmentError
            If ``FRED_API_KEY`` is not set in the environment.
        """
        api_key = os.environ.get("FRED_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "FRED_API_KEY environment variable is not set.\n"
                "Register for a free key at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html\n"
                "Then run: export FRED_API_KEY=your_key_here"
            )

        from fredapi import Fred

        fred = Fred(api_key=api_key)
        dfs: list[pd.Series] = []

        for entry in series:
            fred_id: str = entry["id"]
            alias: str = entry["alias"]
            cache_key = f"fred/{alias}"

            if not force_refresh and self._store.exists(cache_key):
                df_cached = self._store.read(cache_key)
                dfs.append(df_cached[alias])
                continue

            logger.info("Fetching FRED series %s (%s)…", fred_id, alias)
            raw = self._download_with_retry(
                lambda fid=fred_id: fred.get_series(
                    fid, observation_start=start, observation_end=end
                )
            )
            if raw is None:
                logger.warning("FRED series %s fetch failed.", fred_id)
                continue

            s = raw.rename(alias)
            s.index = pd.to_datetime(s.index)
            s = s.loc[start:end]
            # Drop missing values (FRED sometimes includes placeholder '.'
            # which fredapi already converts to NaN)
            s = s.dropna()

            df_s = s.to_frame()
            self._store.write(df_s, cache_key)
            dfs.append(s)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, axis=1)
        result.sort_index(inplace=True)
        return result

    # ------------------------------------------------------------------
    # yfinance macro (commodities / FX proxies)
    # ------------------------------------------------------------------

    def fetch_macro_yfinance(
        self,
        tickers: list[dict],
        start: str,
        end: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch commodity/FX proxies via yfinance (e.g. GC=F, CL=F).

        Returns a DataFrame of log-returns with alias column names.

        Parameters
        ----------
        tickers:
            List of dicts with keys ``ticker`` and ``alias``.
            Mirrors ``data.yfinance_macro`` in config.
        start, end:
            Date range strings.
        force_refresh:
            If True, ignore cache.

        Returns
        -------
        pd.DataFrame
            Log-return columns (one per alias), DatetimeIndex.
        """
        import yfinance as yf

        dfs: list[pd.Series] = []
        for entry in tickers:
            yf_ticker: str = entry["ticker"]
            alias: str = entry["alias"]
            cache_key = f"macro/{alias}"

            if not force_refresh and self._store.exists(cache_key):
                df_cached = self._store.read(cache_key)
                dfs.append(df_cached[alias])
                continue

            logger.info("Fetching macro ticker %s (%s) via yfinance…", yf_ticker, alias)
            raw = self._download_with_retry(
                lambda t=yf_ticker: yf.download(
                    t,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                )
            )
            if raw is None or raw.empty:
                logger.warning("yfinance ticker %s returned empty data.", yf_ticker)
                continue

            close = raw["Close"].squeeze()
            log_ret = np.log(close / close.shift(1)).rename(alias)
            log_ret = log_ret.dropna()
            log_ret.index = pd.to_datetime(log_ret.index)

            df_s = log_ret.to_frame()
            self._store.write(df_s, cache_key)
            dfs.append(log_ret)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, axis=1)
        result.sort_index(inplace=True)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_with_retry(self, fetch_fn: Callable) -> pd.DataFrame | pd.Series | None:
        """Call ``fetch_fn()`` with exponential back-off retry on failure."""
        for attempt in range(self._max_retries):
            try:
                return fetch_fn()
            except Exception as exc:  # noqa: BLE001
                wait = 2 ** attempt
                logger.warning(
                    "Fetch attempt %d/%d failed: %s. Retrying in %ds…",
                    attempt + 1, self._max_retries, exc, wait,
                )
                time.sleep(wait)
        logger.error("All %d fetch attempts failed.", self._max_retries)
        return None
