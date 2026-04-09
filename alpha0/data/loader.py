"""Feature engineering and DataLoader for the Alpha0 training pipeline.

This module transforms raw OHLCV + macro data into the normalised
``(T, N, F)`` tensors consumed by :class:`~alpha0.env.market_env.MarketEnv`
and the policy network.

NO-LOOKAHEAD CONTRACT
---------------------
Every feature computed here must use only information available *before*
the decision timestep.  Concretely:

* Rolling statistics (mean, std) are shifted by 1 period before being
  applied: the normalization parameters at time ``t`` are derived from
  data up to ``t-1`` only.
* log_return at time ``t`` uses close prices at ``t`` and ``t-1``.
  This is valid — the agent sees the return *after* the price has been
  published, not before.

The no-lookahead property is enforced by the ``test_no_lookahead_*``
tests in ``tests/test_data.py``.

TENSOR LAYOUT  ``(T, N, F)``
-----------------------------
T = lookback_window (default 60 trading days)
N = n_assets (default 100)
F = 14 features

Asset features (F[0..6], differ per asset):
  F[0]  log_return       log(close_t / close_{t-1})
  F[1]  vol_21d          21-day rolling std of log_return
  F[2]  dollar_volume    log(close * volume + 1)
  F[3]  high_low_range   log(high / low)
  F[4]  close_to_open    log(close / open)
  F[5]  vol_5d           5-day rolling std of log_return
  F[6]  vol_63d          63-day rolling std of log_return

Macro features (F[7..13], broadcast identically to all N assets):
  F[7]  vix
  F[8]  dgs10
  F[9]  dgs2
  F[10] yield_spread     dgs10 - dgs2
  F[11] dxy
  F[12] gold_return
  F[13] oil_return

All features are rolling z-score normalised (window=252, clipped ±5).
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import torch

from alpha0.data.storage import ParquetStore

logger = logging.getLogger(__name__)

# Canonical feature ordering — must match the docstring above.
ASSET_FEATURE_NAMES: list[str] = [
    "log_return",
    "vol_21d",
    "dollar_volume",
    "high_low_range",
    "close_to_open",
    "vol_5d",
    "vol_63d",
]
MACRO_FEATURE_NAMES: list[str] = [
    "vix",
    "dgs10",
    "dgs2",
    "yield_spread",
    "dxy",
    "gold_return",
    "oil_return",
]
N_ASSET_FEATURES = len(ASSET_FEATURE_NAMES)   # 7
N_MACRO_FEATURES = len(MACRO_FEATURE_NAMES)   # 7
N_FEATURES = N_ASSET_FEATURES + N_MACRO_FEATURES  # 14


class FeatureEngine:
    """Computes the raw feature matrix from OHLCV and macro DataFrames.

    All computations are strictly no-lookahead: rolling statistics use
    ``.shift(1)`` so that the value at time ``t`` uses only data from
    times ``< t``.

    Parameters
    ----------
    cfg:
        Full config dict.
    """

    def __init__(self, cfg: dict) -> None:
        self._norm_window: int = cfg["data"]["normalization"]["window"]
        self._norm_clip: float = cfg["data"]["normalization"]["clip"]

    # ------------------------------------------------------------------
    # Per-asset features
    # ------------------------------------------------------------------

    def compute_asset_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute per-asset feature matrix from OHLCV data.

        Parameters
        ----------
        ohlcv:
            Wide DataFrame with two-level column MultiIndex
            ``(field, ticker)``.  Fields must include at least
            ``Open``, ``High``, ``Low``, ``Close``, ``Volume``.

        Returns
        -------
        pd.DataFrame
            MultiIndex columns ``(feature, ticker)``, same DatetimeIndex.
            Values are **not yet normalised** — call
            :meth:`apply_rolling_zscore` afterward.
        """
        fields = ohlcv.columns.get_level_values("field").unique()
        tickers = ohlcv.columns.get_level_values("ticker").unique().tolist()

        def _field(f: str) -> pd.DataFrame:
            return ohlcv[f] if f in fields else pd.DataFrame(index=ohlcv.index)

        close = _field("Close")
        open_ = _field("Open")
        high = _field("High")
        low = _field("Low")
        volume = _field("Volume")

        log_ret = np.log(close / close.shift(1))

        result_frames: dict[str, pd.DataFrame] = {}

        result_frames["log_return"] = log_ret

        result_frames["vol_5d"] = (
            log_ret.rolling(5, min_periods=3).std()
        )
        result_frames["vol_21d"] = (
            log_ret.rolling(21, min_periods=10).std()
        )
        result_frames["vol_63d"] = (
            log_ret.rolling(63, min_periods=30).std()
        )

        # log(close * volume + 1) — log-scale dampens outliers
        result_frames["dollar_volume"] = np.log(close * volume + 1)

        result_frames["high_low_range"] = np.log(
            (high + 1e-8) / (low + 1e-8)
        )
        result_frames["close_to_open"] = np.log(
            (close + 1e-8) / (open_ + 1e-8)
        )

        # Assemble MultiIndex DataFrame in canonical order
        arrays = [
            pd.MultiIndex.from_product([[feat], tickers], names=["feature", "ticker"])
            for feat in ASSET_FEATURE_NAMES
        ]
        multi_idx = arrays[0]
        for a in arrays[1:]:
            multi_idx = multi_idx.append(a)

        dfs = [result_frames[feat] for feat in ASSET_FEATURE_NAMES]
        combined = pd.concat(dfs, axis=1, keys=ASSET_FEATURE_NAMES)
        combined.columns.names = ["feature", "ticker"]
        return combined

    # ------------------------------------------------------------------
    # Macro features
    # ------------------------------------------------------------------

    def compute_macro_features(
        self,
        macro_df: pd.DataFrame,
        _yf_macro_df: pd.DataFrame | None = None,  # kept for API compat
    ) -> pd.DataFrame:
        """Compute cross-asset macro feature vector from a combined DataFrame.

        Parameters
        ----------
        macro_df:
            Combined macro DataFrame containing any subset of the expected
            columns: ``vix``, ``dgs10``, ``dgs2``, ``dxy``,
            ``gold_return``, ``oil_return``.  Missing columns are
            zero-filled.  Must have a DatetimeIndex.
        _yf_macro_df:
            Deprecated; ignored.  Kept for backward compatibility.

        Returns
        -------
        pd.DataFrame
            Columns matching ``MACRO_FEATURE_NAMES``, same DatetimeIndex.
            Values are **not yet normalised**.
        """
        frames: dict[str, pd.Series] = {}
        ref_index: pd.Index = macro_df.index if len(macro_df) > 0 else pd.DatetimeIndex([])

        for col in ("vix", "dgs10", "dgs2", "dxy", "gold_return", "oil_return"):
            if col in macro_df.columns:
                frames[col] = macro_df[col]
            else:
                logger.warning("Macro feature '%s' missing — zero-filled.", col)

        if "dgs10" in frames and "dgs2" in frames:
            frames["yield_spread"] = frames["dgs10"] - frames["dgs2"]
        else:
            frames["yield_spread"] = pd.Series(0.0, index=ref_index)

        # Assemble in canonical order; missing features filled with 0
        cols: dict[str, pd.Series] = {}
        for feat in MACRO_FEATURE_NAMES:
            cols[feat] = frames.get(feat, pd.Series(0.0, index=ref_index))

        result = pd.DataFrame(cols)
        result.index = pd.to_datetime(result.index)
        result.sort_index(inplace=True)
        return result

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def apply_rolling_zscore(
        self,
        df: pd.DataFrame,
        window: int | None = None,
        clip: float | None = None,
    ) -> pd.DataFrame:
        """Normalise each column with a rolling z-score.

        The mean and std used at time ``t`` are computed from data up to
        ``t-1`` only (achieved via ``.shift(1)``), guaranteeing no
        lookahead bias.

        For the first ``window`` observations where insufficient history
        is available, an expanding window is used (rather than leaving
        NaN) to prevent the leading segment of every episode from being
        unusable.

        Parameters
        ----------
        df:
            DataFrame to normalise (any column structure).
        window:
            Rolling window size. Defaults to ``cfg.data.normalization.window``.
        clip:
            Clip normalised values to ``[-clip, clip]``.
            Defaults to ``cfg.data.normalization.clip``.

        Returns
        -------
        pd.DataFrame
            Same shape as ``df``, values in ``[-clip, clip]``.
        """
        w = window if window is not None else self._norm_window
        c = clip if clip is not None else self._norm_clip

        result = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float32)

        for col in df.columns:
            s = df[col].astype(float)

            # Compute rolling stats then shift — stats at t use data up to t-1
            roll_mean = s.rolling(w, min_periods=1).mean().shift(1)
            roll_std = s.rolling(w, min_periods=2).std().shift(1)

            # Expanding fallback for the first few rows where rolling std is NaN
            exp_mean = s.expanding(min_periods=1).mean().shift(1)
            exp_std = s.expanding(min_periods=2).std().shift(1)

            mean = roll_mean.where(roll_mean.notna(), exp_mean)
            std = roll_std.where(roll_std.notna(), exp_std)

            z = (s - mean) / (std.clip(lower=1e-8))
            z = z.clip(-c, c).fillna(0.0)
            result[col] = z.values

        return result


class DataLoader:
    """Assembles and serves ``(T, N, F)`` observation tensors.

    Workflow:
    1. Call :meth:`build` once to compute features and save processed
       tensors to ``data/processed/``.
    2. Use :meth:`get_window` inside the environment for per-step obs.
    3. Use :meth:`iter_episodes` during training to stream episode blocks.

    Parameters
    ----------
    cfg:
        Full config dict.
    raw_store:
        :class:`~alpha0.data.storage.ParquetStore` for raw/cache data.
    processed_store:
        :class:`~alpha0.data.storage.ParquetStore` for processed features.
    """

    _METADATA_FILE = "metadata.json"

    def __init__(
        self,
        cfg: dict,
        raw_store: ParquetStore,
        processed_store: ParquetStore,
    ) -> None:
        self._cfg = cfg
        self._dc = cfg["data"]
        self._lookback: int = self._dc["lookback_window"]
        self._raw_store = raw_store
        self._proc_store = processed_store
        self._engine = FeatureEngine(cfg)

        # Populated after build() or _load()
        self._feature_tensor: torch.Tensor | None = None   # (T_total, N, F)
        self._close_prices: torch.Tensor | None = None     # (T_total, N) raw close prices
        self._dates: pd.DatetimeIndex | None = None
        self._tickers: list[str] | None = None
        self._split_indices: dict[str, tuple[int, int]] | None = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        ohlcv: pd.DataFrame,
        macro_df: pd.DataFrame,
        tickers: list[str],
        force_rebuild: bool = False,
    ) -> None:
        """Compute features, normalise, and cache tensors to disk.

        Parameters
        ----------
        ohlcv:
            Wide MultiIndex ``(field, ticker)`` OHLCV DataFrame.
        macro_df:
            DataFrame with macro feature columns (already aligned to
            trading calendar; returned by
            :meth:`~alpha0.data.fetcher.DataFetcher.fetch_fred_series`
            after merging with yfinance macro returns).
        tickers:
            Ordered ticker list (the universe).
        force_rebuild:
            If True, recompute even if cached tensors exist.
        """
        meta_path = self._proc_store._resolve(self._METADATA_FILE.replace(".json", ""))
        meta_path = meta_path.parent / self._METADATA_FILE

        if not force_rebuild and meta_path.exists():
            logger.info("Processed tensors found — loading from cache.")
            self._load()
            return

        logger.info("Building feature tensors for %d assets…", len(tickers))

        # 1. Restrict OHLCV to universe tickers
        available = [t for t in tickers if t in ohlcv.columns.get_level_values("ticker")]
        if len(available) < len(tickers):
            logger.warning(
                "%d tickers in universe not found in OHLCV data.",
                len(tickers) - len(available),
            )
        tickers = available

        # 2. Compute asset features
        ohlcv_uni = ohlcv.loc[:, ohlcv.columns.get_level_values("ticker").isin(tickers)]
        asset_feat = self._engine.compute_asset_features(ohlcv_uni)  # (T, N*7) MultiIndex

        # 3. Compute macro features from the combined macro DataFrame.
        #    If macro_df is empty, synthesise a zero-filled frame on the asset dates.
        if macro_df.empty:
            macro_df_use = pd.DataFrame(index=asset_feat.index)
        else:
            macro_df_use = macro_df
        macro_feat = self._engine.compute_macro_features(macro_df_use)

        # 4. Align to common trading days
        asset_feat, macro_feat = self._align_dates(asset_feat, macro_feat)
        common_dates = asset_feat.index

        # 5. Normalise
        # Asset features: normalise each (feature, ticker) column independently
        asset_norm = self._engine.apply_rolling_zscore(asset_feat)

        # Macro features: normalise each column independently
        macro_norm = self._engine.apply_rolling_zscore(macro_feat)

        # 6. Assemble (T_total, N, F) tensor
        T = len(common_dates)
        N = len(tickers)

        tensor = np.zeros((T, N, N_FEATURES), dtype=np.float32)

        # Asset features — shape (T, N, 7)
        for fi, feat_name in enumerate(ASSET_FEATURE_NAMES):
            if feat_name not in asset_norm.columns.get_level_values("feature"):
                logger.warning("Asset feature '%s' missing — leaving zeros.", feat_name)
                continue
            col_data = asset_norm[feat_name]  # (T, N) — one col per ticker
            for ni, ticker in enumerate(tickers):
                if ticker in col_data.columns:
                    vals = col_data[ticker].values.astype(np.float32)
                    tensor[:, ni, fi] = vals

        # Macro features — shape (T, 7) → broadcast to (T, N, 7)
        for fi_rel, feat_name in enumerate(MACRO_FEATURE_NAMES):
            fi = N_ASSET_FEATURES + fi_rel
            if feat_name not in macro_norm.columns:
                continue
            vals = macro_norm[feat_name].values.astype(np.float32)
            tensor[:, :, fi] = vals[:, np.newaxis]  # broadcast over N

        # Store raw close prices (aligned to common_dates) for env return calculation
        close_wide = ohlcv_uni["Close"].reindex(common_dates)
        # Ensure column order matches tickers
        close_wide = close_wide[[t for t in tickers if t in close_wide.columns]]
        self._close_prices = torch.from_numpy(close_wide.ffill().fillna(1.0).values.astype(np.float32))

        self._feature_tensor = torch.from_numpy(tensor)
        self._dates = common_dates
        self._tickers = tickers
        self._split_indices = self._compute_split_indices(common_dates)

        # 7. Save
        self._save(meta_path.parent)
        logger.info(
            "Feature tensor built: shape %s, dates %s – %s",
            tuple(self._feature_tensor.shape),
            common_dates[0].date(),
            common_dates[-1].date(),
        )

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_window(
        self,
        end_date: str | pd.Timestamp,
        lookback: int | None = None,
    ) -> tuple[torch.Tensor, list[str]]:
        """Return a single ``(T, N, F)`` observation window.

        Parameters
        ----------
        end_date:
            The *last* date in the window (inclusive).  The window
            spans ``[end_date - lookback + 1, end_date]``.
        lookback:
            Window length.  Defaults to ``cfg.data.lookback_window``.

        Returns
        -------
        tuple[torch.Tensor, list[str]]
            - Tensor of shape ``(lookback, N, F)``
            - List of tickers (axis-1 labels)
        """
        self._ensure_loaded()
        L = lookback if lookback is not None else self._lookback
        end_dt = pd.Timestamp(end_date)

        idx_arr = self._dates.searchsorted(end_dt, side="right") - 1
        if idx_arr < 0 or idx_arr >= len(self._dates):
            raise ValueError(f"end_date {end_date} not in data range.")

        start_idx = max(0, idx_arr - L + 1)
        window = self._feature_tensor[start_idx : idx_arr + 1]

        # Pad with zeros if insufficient history
        if window.shape[0] < L:
            pad = torch.zeros(L - window.shape[0], window.shape[1], window.shape[2])
            window = torch.cat([pad, window], dim=0)

        return window, self._tickers

    def iter_episodes(
        self,
        episode_length: int,
        split: str = "train",
    ) -> Generator[tuple[torch.Tensor, pd.DatetimeIndex], None, None]:
        """Yield observation blocks for each possible episode start.

        Each block includes a ``lookback`` prefix so that step 0 of the
        episode has full context.

        Parameters
        ----------
        episode_length:
            Number of steps per episode.
        split:
            One of ``"train"``, ``"val"``, ``"test"``.

        Yields
        ------
        tuple[torch.Tensor, pd.DatetimeIndex]
            - Tensor of shape ``(episode_length + lookback, N, F)``
            - DatetimeIndex of length ``episode_length + lookback``
        """
        self._ensure_loaded()
        start_idx, end_idx = self._split_indices[split]
        L = self._lookback
        block_len = episode_length + L

        for i in range(start_idx, end_idx - block_len + 1):
            yield (
                self._feature_tensor[i : i + block_len],
                self._dates[i : i + block_len],
            )

    def load(self) -> None:
        """Public method to load pre-built tensors from disk.

        Equivalent to calling ``_ensure_loaded()``.  Raises
        :exc:`FileNotFoundError` if tensors have not been built yet.

        Use this in scripts that require data to already exist::

            loader = DataLoader(cfg, raw_store, processed_store)
            loader.load()   # raises FileNotFoundError if not built
        """
        if self._feature_tensor is None:
            self._load()

    @property
    def tickers(self) -> list[str]:
        """Ordered list of tickers in the universe."""
        self._ensure_loaded()
        return self._tickers

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Full DatetimeIndex of the processed feature tensor."""
        self._ensure_loaded()
        return self._dates

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _align_dates(
        self,
        asset_df: pd.DataFrame,
        macro_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Align asset and macro DataFrames to a common set of trading days.

        Macro data is forward-filled (max 5 business days) to fill gaps
        on days where FRED published no new observation.  Back-filling is
        never applied (would introduce lookahead).
        """
        # Use asset dates as the calendar anchor
        common_idx = asset_df.index
        macro_aligned = (
            macro_df
            .reindex(common_idx, method="ffill", limit=5)
            .fillna(0.0)
        )
        return asset_df, macro_aligned

    def _compute_split_indices(
        self, dates: pd.DatetimeIndex
    ) -> dict[str, tuple[int, int]]:
        """Map train/val/test labels to integer index ranges."""
        train_end = pd.Timestamp(self._dc["train_end"])
        val_end = pd.Timestamp(self._dc["val_end"])

        train_end_idx = int(dates.searchsorted(train_end, side="right"))
        val_end_idx = int(dates.searchsorted(val_end, side="right"))

        return {
            "train": (0, train_end_idx),
            "val": (train_end_idx, val_end_idx),
            "test": (val_end_idx, len(dates)),
        }

    def _save(self, dest_dir: Path) -> None:
        """Persist tensor and metadata to disk."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._feature_tensor, dest_dir / "features.pt")
        if self._close_prices is not None:
            torch.save(self._close_prices, dest_dir / "close_prices.pt")

        metadata = {
            "tickers": self._tickers,
            "dates": [str(d.date()) for d in self._dates],
            "split_indices": {k: list(v) for k, v in self._split_indices.items()},
            "shape": list(self._feature_tensor.shape),
        }
        with open(dest_dir / self._METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved feature tensor to %s", dest_dir)

    def _load(self) -> None:
        """Load tensor and metadata from disk."""
        proc_dir = self._proc_store.base_dir
        tensor_path = proc_dir / "features.pt"
        meta_path = proc_dir / self._METADATA_FILE

        if not tensor_path.exists():
            raise FileNotFoundError(f"Feature tensor not found: {tensor_path}")

        self._feature_tensor = torch.load(tensor_path, weights_only=True)

        close_path = proc_dir / "close_prices.pt"
        if close_path.exists():
            self._close_prices = torch.load(close_path, weights_only=True)

        with open(meta_path) as f:
            meta = json.load(f)

        self._tickers = meta["tickers"]
        self._dates = pd.DatetimeIndex(meta["dates"])
        self._split_indices = {k: tuple(v) for k, v in meta["split_indices"].items()}
        logger.info(
            "Loaded feature tensor: shape %s", tuple(self._feature_tensor.shape)
        )

    def _ensure_loaded(self) -> None:
        """Raise if build() has not been called and no cached data exists."""
        if self._feature_tensor is None:
            try:
                self._load()
            except FileNotFoundError:
                raise RuntimeError(
                    "DataLoader has no data. Call build() first or ensure "
                    "processed tensors exist in the processed_store directory."
                )
