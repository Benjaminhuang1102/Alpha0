"""Tests for the data engine modules: storage, universe, fetcher, loader."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha0.data.storage import ParquetStore


# ─────────────────────────────────────────────────────────────
# ParquetStore tests
# ─────────────────────────────────────────────────────────────


class TestParquetStore:
    def test_roundtrip_simple(self, tmp_path):
        """Write and read preserves values, dtypes, and DatetimeIndex."""
        store = ParquetStore(tmp_path)
        idx = pd.date_range("2020-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {"a": np.random.randn(10), "b": np.arange(10, dtype=np.int64)},
            index=idx,
        )
        store.write(df, "test/simple")
        result = store.read("test/simple")

        # Parquet does not preserve DatetimeIndex.freq — drop check_freq
        pd.testing.assert_frame_equal(df, result, check_freq=False)

    def test_roundtrip_multiindex_columns(self, tmp_path):
        """MultiIndex column DataFrames round-trip correctly."""
        store = ParquetStore(tmp_path)
        idx = pd.date_range("2021-01-01", periods=5, freq="B")
        arrays = [["close", "close", "volume"], ["AAPL", "MSFT", "AAPL"]]
        cols = pd.MultiIndex.from_arrays(arrays, names=["field", "ticker"])
        df = pd.DataFrame(np.random.randn(5, 3), index=idx, columns=cols)
        store.write(df, "ohlcv/multi")
        result = store.read("ohlcv/multi")
        pd.testing.assert_frame_equal(df, result, check_freq=False)

    def test_exists_true_after_write(self, tmp_path):
        store = ParquetStore(tmp_path)
        idx = pd.date_range("2020-01-01", periods=3, freq="B")
        df = pd.DataFrame({"x": [1, 2, 3]}, index=idx)
        assert not store.exists("foo")
        store.write(df, "foo")
        assert store.exists("foo")

    def test_exists_false_for_unknown(self, tmp_path):
        store = ParquetStore(tmp_path)
        assert not store.exists("nonexistent/path")

    def test_read_missing_raises(self, tmp_path):
        store = ParquetStore(tmp_path)
        with pytest.raises(FileNotFoundError):
            store.read("does/not/exist")

    def test_list_datasets_empty(self, tmp_path):
        store = ParquetStore(tmp_path)
        assert store.list_datasets() == []

    def test_list_datasets_returns_names(self, tmp_path):
        store = ParquetStore(tmp_path)
        idx = pd.date_range("2020-01-01", periods=3, freq="B")
        df = pd.DataFrame({"v": [1, 2, 3]}, index=idx)
        store.write(df, "a/x")
        store.write(df, "a/y")
        store.write(df, "b/z")
        names = store.list_datasets()
        assert set(names) == {"a/x", "a/y", "b/z"}

    def test_list_datasets_prefix_filter(self, tmp_path):
        store = ParquetStore(tmp_path)
        idx = pd.date_range("2020-01-01", periods=3, freq="B")
        df = pd.DataFrame({"v": [1, 2, 3]}, index=idx)
        store.write(df, "prices/AAPL")
        store.write(df, "prices/MSFT")
        store.write(df, "macro/vix")
        names = store.list_datasets(prefix="prices")
        assert set(names) == {"prices/AAPL", "prices/MSFT"}

    def test_column_pruning(self, tmp_path):
        """Reading with columns= returns only requested columns."""
        store = ParquetStore(tmp_path)
        idx = pd.date_range("2020-01-01", periods=5, freq="B")
        df = pd.DataFrame({"a": range(5), "b": range(5), "c": range(5)}, index=idx)
        store.write(df, "wide")
        result = store.read("wide", columns=["a", "c"])
        assert list(result.columns) == ["a", "c"]

    def test_nested_path_created(self, tmp_path):
        """Write creates intermediate directories."""
        store = ParquetStore(tmp_path)
        idx = pd.date_range("2020-01-01", periods=2, freq="B")
        df = pd.DataFrame({"x": [1, 2]}, index=idx)
        store.write(df, "deep/nested/path/data")
        assert (tmp_path / "deep" / "nested" / "path" / "data.parquet").exists()

    def test_datetime_index_preserved(self, tmp_path):
        """DatetimeIndex timezone and frequency survive roundtrip."""
        store = ParquetStore(tmp_path)
        idx = pd.date_range("2022-01-01", periods=4, freq="B")
        df = pd.DataFrame({"v": range(4)}, index=idx)
        store.write(df, "ts")
        result = store.read("ts")
        assert result.index.dtype == df.index.dtype
        pd.testing.assert_index_equal(df.index, result.index)


# ─────────────────────────────────────────────────────────────
# UniverseManager tests (no network — uses synthetic price data)
# ─────────────────────────────────────────────────────────────

from alpha0.data.universe import UniverseManager


def _make_universe_cfg(max_assets: int = 10) -> dict:
    return {
        "universe": {
            "min_history_days": 10,
            "min_avg_dollar_volume": 100.0,
            "max_assets": max_assets,
        }
    }


def _make_price_volume(tickers: list[str], n_days: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic price/volume DataFrames for universe filtering tests."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    prices = pd.DataFrame(
        rng.uniform(10, 200, (n_days, len(tickers))),
        index=idx,
        columns=tickers,
    )
    volumes = pd.DataFrame(
        rng.integers(1_000, 10_000_000, (n_days, len(tickers))),
        index=idx,
        columns=tickers,
    )
    return prices, volumes


class TestUniverseManager:
    def test_build_universe_respects_max_assets(self, tmp_path):
        tickers = [f"T{i:03d}" for i in range(50)]
        prices, volumes = _make_price_volume(tickers, n_days=50)
        store = ParquetStore(tmp_path)
        mgr = UniverseManager(_make_universe_cfg(max_assets=10), store)
        result = mgr.build_universe(tickers, prices, volumes)
        assert len(result) <= 10

    def test_build_universe_history_filter(self, tmp_path):
        """Tickers with insufficient history are excluded."""
        tickers = ["A", "B", "C"]
        rng = np.random.default_rng(0)
        idx = pd.date_range("2020-01-01", periods=30, freq="B")
        prices = pd.DataFrame(rng.uniform(10, 100, (30, 3)), index=idx, columns=tickers)
        volumes = pd.DataFrame(np.ones((30, 3)) * 1_000_000, index=idx, columns=tickers)
        # Make "C" have too many NaNs
        prices["C"] = np.nan
        cfg = {"universe": {"min_history_days": 20, "min_avg_dollar_volume": 1.0, "max_assets": 10}}
        store = ParquetStore(tmp_path)
        mgr = UniverseManager(cfg, store)
        result = mgr.build_universe(tickers, prices, volumes)
        assert "C" not in result
        assert "A" in result and "B" in result

    def test_build_universe_dollar_volume_filter(self, tmp_path):
        """Tickers with low dollar volume are excluded."""
        tickers = ["HIGH", "LOW"]
        idx = pd.date_range("2020-01-01", periods=30, freq="B")
        prices = pd.DataFrame({"HIGH": [100.0] * 30, "LOW": [100.0] * 30}, index=idx)
        volumes = pd.DataFrame({"HIGH": [1_000_000] * 30, "LOW": [1] * 30}, index=idx)
        cfg = {"universe": {"min_history_days": 5, "min_avg_dollar_volume": 10_000.0, "max_assets": 10}}
        store = ParquetStore(tmp_path)
        mgr = UniverseManager(cfg, store)
        result = mgr.build_universe(tickers, prices, volumes)
        assert "HIGH" in result
        assert "LOW" not in result

    def test_save_and_load_universe(self, tmp_path):
        """Save/load roundtrip preserves order."""
        tickers = ["AAPL", "MSFT", "GOOG"]
        store = ParquetStore(tmp_path)
        mgr = UniverseManager(_make_universe_cfg(), store)
        mgr.save_universe(tickers)
        loaded = mgr.load_universe()
        assert loaded == tickers

    def test_load_fallback_csv_exists(self):
        """The bundled fallback CSV ships with the package and has expected format."""
        from alpha0.data.universe import _FALLBACK_CSV
        assert _FALLBACK_CSV.exists(), "Fallback CSV missing from package"
        df = pd.read_csv(_FALLBACK_CSV)
        assert len(df) > 400, "Fallback CSV should have ~500 tickers"


# ─────────────────────────────────────────────────────────────
# DataFetcher tests (mock network calls)
# ─────────────────────────────────────────────────────────────

from alpha0.data.fetcher import DataFetcher


def _fetcher_cfg() -> dict:
    return {
        "data": {
            "fetch_chunk_size": 50,
            "fetch_chunk_delay": 0.0,
            "fetch_max_retries": 2,
        }
    }


def _make_fake_ohlcv(tickers: list[str], n: int = 30) -> pd.DataFrame:
    """Fake MultiIndex OHLCV DataFrame as modern yfinance returns (Price, Ticker) levels."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    fields = ["Open", "High", "Low", "Close", "Volume"]
    # New yfinance uses ("Price", "Ticker") level names
    arrays = pd.MultiIndex.from_product([fields, tickers], names=["Price", "Ticker"])
    data = rng.uniform(10, 200, (n, len(fields) * len(tickers)))
    return pd.DataFrame(data, index=idx, columns=arrays)


class TestDataFetcher:
    def test_fetch_ohlcv_uses_cache(self, tmp_path, mocker):
        """Second call reads from cache without hitting yfinance."""
        store = ParquetStore(tmp_path)
        fetcher = DataFetcher(_fetcher_cfg(), store)

        tickers = ["AAPL", "MSFT"]
        fake_raw = _make_fake_ohlcv(tickers)
        mock_dl = mocker.patch("yfinance.download", return_value=fake_raw)

        fetcher.fetch_ohlcv(tickers, "2020-01-01", "2020-02-28")
        assert mock_dl.call_count == 1

        # Second call — should hit cache, not yfinance
        fetcher.fetch_ohlcv(tickers, "2020-01-01", "2020-02-28")
        assert mock_dl.call_count == 1  # unchanged

    def test_fetch_ohlcv_force_refresh(self, tmp_path, mocker):
        """force_refresh=True bypasses cache."""
        store = ParquetStore(tmp_path)
        fetcher = DataFetcher(_fetcher_cfg(), store)

        tickers = ["AAPL", "MSFT"]
        fake_raw = _make_fake_ohlcv(tickers)
        mock_dl = mocker.patch("yfinance.download", return_value=fake_raw)

        fetcher.fetch_ohlcv(tickers, "2020-01-01", "2020-02-28")
        fetcher.fetch_ohlcv(tickers, "2020-01-01", "2020-02-28", force_refresh=True)
        assert mock_dl.call_count == 2

    def test_fetch_ohlcv_returns_multiindex(self, tmp_path, mocker):
        """Result has (field, ticker) MultiIndex columns."""
        store = ParquetStore(tmp_path)
        fetcher = DataFetcher(_fetcher_cfg(), store)

        tickers = ["AAPL", "MSFT"]
        mocker.patch("yfinance.download", return_value=_make_fake_ohlcv(tickers))

        result = fetcher.fetch_ohlcv(tickers, "2020-01-01", "2020-02-28")
        assert isinstance(result.columns, pd.MultiIndex)
        assert "Close" in result.columns.get_level_values("field")
        assert "AAPL" in result.columns.get_level_values("ticker")

    def test_fetch_fred_raises_without_api_key(self, tmp_path, monkeypatch):
        """Raises EnvironmentError when FRED_API_KEY is not set."""
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        store = ParquetStore(tmp_path)
        fetcher = DataFetcher(_fetcher_cfg(), store)
        with pytest.raises(EnvironmentError, match="FRED_API_KEY"):
            fetcher.fetch_fred_series(
                [{"id": "VIXCLS", "alias": "vix"}], "2020-01-01", "2020-12-31"
            )

    def test_fetch_macro_yfinance_returns_log_returns(self, tmp_path, mocker):
        """fetch_macro_yfinance returns log-return DataFrame."""
        store = ParquetStore(tmp_path)
        fetcher = DataFetcher(_fetcher_cfg(), store)

        idx = pd.date_range("2020-01-01", periods=20, freq="B")
        fake_close = pd.DataFrame({"Close": np.cumprod(1 + np.random.randn(20) * 0.01) * 100}, index=idx)
        mocker.patch("yfinance.download", return_value=fake_close)

        result = fetcher.fetch_macro_yfinance(
            [{"ticker": "GC=F", "alias": "gold"}], "2020-01-01", "2020-12-31"
        )
        assert "gold" in result.columns
        # Log returns should be small numbers (not prices)
        assert result["gold"].abs().max() < 1.0

    def test_fetch_ohlcv_single_ticker(self, tmp_path, mocker):
        """Single-ticker yfinance response returns (Price, Ticker) MultiIndex."""
        store = ParquetStore(tmp_path)
        fetcher = DataFetcher(_fetcher_cfg(), store)

        idx = pd.date_range("2020-01-01", periods=10, freq="B")
        # New yfinance always returns MultiIndex even for single ticker
        fields = ["Open", "High", "Low", "Close", "Volume"]
        cols = pd.MultiIndex.from_product([fields, ["AAPL"]], names=["Price", "Ticker"])
        mi = pd.DataFrame(np.random.rand(10, 5), index=idx, columns=cols)
        mocker.patch("yfinance.download", return_value=mi)

        result = fetcher.fetch_ohlcv(["AAPL"], "2020-01-01", "2020-02-01")
        assert "AAPL" in result.columns.get_level_values("ticker")
        assert "Close" in result.columns.get_level_values("field")


# ─────────────────────────────────────────────────────────────
# FeatureEngine + DataLoader tests
# ─────────────────────────────────────────────────────────────

from alpha0.data.loader import (
    ASSET_FEATURE_NAMES,
    MACRO_FEATURE_NAMES,
    N_FEATURES,
    DataLoader,
    FeatureEngine,
)


def _loader_cfg() -> dict:
    return {
        "data": {
            "lookback_window": 10,
            "train_end": "2021-06-30",
            "val_end":   "2021-09-30",
            "normalization": {"window": 20, "clip": 5.0},
        }
    }


def _make_ohlcv(tickers: list[str], n: int = 400) -> pd.DataFrame:
    """Synthetic MultiIndex OHLCV DataFrame."""
    rng = np.random.default_rng(123)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    fields = ["Open", "High", "Low", "Close", "Volume"]
    data = {}
    for t in tickers:
        for f in fields:
            base = rng.uniform(50, 200) if f != "Volume" else rng.uniform(1e5, 1e7)
            vals = np.cumprod(1 + rng.normal(0, 0.01, n)) * base
            vals = np.abs(vals)
            data[(f, t)] = vals
    cols = pd.MultiIndex.from_tuples(data.keys(), names=["field", "ticker"])
    df = pd.DataFrame(np.column_stack(list(data.values())), index=idx, columns=cols)
    # Ensure High >= Low
    for t in tickers:
        hi = np.maximum(df[("High", t)].values, df[("Low", t)].values)
        lo = np.minimum(df[("High", t)].values, df[("Low", t)].values)
        df[("High", t)] = hi
        df[("Low", t)] = lo
    return df


def _make_macro(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "vix":        rng.uniform(10, 40, n),
            "dgs10":      rng.uniform(1, 5, n),
            "dgs2":       rng.uniform(0.5, 3, n),
            "dxy":        rng.uniform(90, 110, n),
            "gold_return": rng.normal(0, 0.01, n),
            "oil_return":  rng.normal(0, 0.02, n),
        },
        index=idx,
    )


class TestFeatureEngine:
    def test_compute_asset_features_shape(self):
        tickers = ["A", "B", "C"]
        ohlcv = _make_ohlcv(tickers, n=100)
        engine = FeatureEngine(_loader_cfg())
        result = engine.compute_asset_features(ohlcv)
        # Columns: MultiIndex (feature, ticker)
        assert result.columns.names == ["feature", "ticker"]
        feats = result.columns.get_level_values("feature").unique()
        for f in ASSET_FEATURE_NAMES:
            assert f in feats, f"Feature '{f}' missing"

    def test_compute_asset_features_no_nan_after_warmup(self):
        """After the rolling warmup period no NaN should remain."""
        tickers = ["X"]
        ohlcv = _make_ohlcv(tickers, n=200)
        engine = FeatureEngine(_loader_cfg())
        result = engine.compute_asset_features(ohlcv)
        # Skip first 63 rows (longest rolling window)
        tail = result.iloc[63:]
        assert not tail.isnull().values.any(), "NaN found after warmup period"

    def test_rolling_zscore_values_clipped(self):
        """Normalised values must stay within [-clip, clip]."""
        engine = FeatureEngine(_loader_cfg())
        rng = np.random.default_rng(0)
        idx = pd.date_range("2020-01-01", periods=300, freq="B")
        # Inject extreme outlier
        vals = rng.normal(0, 1, 300)
        vals[150] = 1000.0
        df = pd.DataFrame({"col": vals}, index=idx)
        result = engine.apply_rolling_zscore(df, window=20, clip=5.0)
        assert result.abs().max().max() <= 5.0 + 1e-6

    def test_rolling_zscore_no_lookahead(self):
        """Adding one row to the series must NOT change any earlier normalised value.

        This is the critical no-lookahead test.  If rolling stats are computed
        without shift(1), appending a row would update the stats used for
        earlier rows in the rolling window.
        """
        engine = FeatureEngine(_loader_cfg())
        rng = np.random.default_rng(99)
        idx_short = pd.date_range("2020-01-01", periods=50, freq="B")
        idx_long  = pd.date_range("2020-01-01", periods=51, freq="B")
        vals = rng.normal(0, 1, 51)

        df_short = pd.DataFrame({"col": vals[:50]}, index=idx_short)
        df_long  = pd.DataFrame({"col": vals},       index=idx_long)

        norm_short = engine.apply_rolling_zscore(df_short, window=20, clip=5.0)
        norm_long  = engine.apply_rolling_zscore(df_long,  window=20, clip=5.0)

        # Values at overlapping rows must be identical
        np.testing.assert_array_almost_equal(
            norm_short["col"].values,
            norm_long["col"].values[:50],
            decimal=6,
            err_msg="No-lookahead violated: appending a row changed earlier normalised values.",
        )

    def test_compute_macro_features_yield_spread(self):
        """yield_spread = dgs10 - dgs2."""
        engine = FeatureEngine(_loader_cfg())
        idx = pd.date_range("2020-01-01", periods=10, freq="B")
        macro = pd.DataFrame({
            "dgs10": [3.0] * 10, "dgs2": [1.0] * 10,
            "gold_return": [0.0] * 10, "oil_return": [0.0] * 10,
        }, index=idx)
        result = engine.compute_macro_features(macro)
        assert "yield_spread" in result.columns
        np.testing.assert_allclose(result["yield_spread"].values, 2.0)


class TestDataLoader:
    def _build_loader(self, tmp_path) -> DataLoader:
        tickers = ["A", "B", "C", "D", "E"]
        ohlcv = _make_ohlcv(tickers, n=500)
        macro = _make_macro(n=500)
        raw_store = ParquetStore(tmp_path / "raw")
        proc_store = ParquetStore(tmp_path / "proc")
        cfg = {
            "data": {
                "lookback_window": 10,
                "train_end": "2021-06-30",
                "val_end":   "2021-09-30",
                "normalization": {"window": 20, "clip": 5.0},
            }
        }
        loader = DataLoader(cfg, raw_store, proc_store)
        loader.build(ohlcv, macro, tickers)
        return loader

    def test_get_window_shape(self, tmp_path):
        loader = self._build_loader(tmp_path)
        date = loader.dates[20]
        window, tickers = loader.get_window(date)
        assert window.shape == (10, 5, N_FEATURES)
        assert len(tickers) == 5

    def test_get_window_values_in_range(self, tmp_path):
        loader = self._build_loader(tmp_path)
        window, _ = loader.get_window(loader.dates[50])
        assert float(window.abs().max()) <= 5.0 + 1e-5

    def test_split_indices_disjoint(self, tmp_path):
        loader = self._build_loader(tmp_path)
        train_end = loader._split_indices["train"][1]
        val_start = loader._split_indices["val"][0]
        val_end   = loader._split_indices["val"][1]
        test_start = loader._split_indices["test"][0]
        assert train_end == val_start, "Train/val split not contiguous"
        assert val_end   == test_start, "Val/test split not contiguous"

    def test_iter_episodes_yields_correct_shape(self, tmp_path):
        loader = self._build_loader(tmp_path)
        ep_len = 5
        blocks = list(loader.iter_episodes(ep_len, split="train"))
        assert len(blocks) > 0
        for tensor, dates in blocks:
            assert tensor.shape[0] == ep_len + 10  # ep_len + lookback
            assert tensor.shape[2] == N_FEATURES
            assert len(dates) == ep_len + 10

    def test_build_idempotent_via_cache(self, tmp_path):
        """Second build() call loads from cache without recomputing."""
        tickers = ["A", "B"]
        ohlcv = _make_ohlcv(tickers, n=300)
        macro = _make_macro(n=300)
        cfg = {
            "data": {
                "lookback_window": 5,
                "train_end": "2021-01-01",
                "val_end":   "2021-06-01",
                "normalization": {"window": 20, "clip": 5.0},
            }
        }
        proc_store = ParquetStore(tmp_path / "proc")
        loader1 = DataLoader(cfg, ParquetStore(tmp_path / "raw"), proc_store)
        loader1.build(ohlcv, macro, tickers)
        shape1 = tuple(loader1._feature_tensor.shape)

        loader2 = DataLoader(cfg, ParquetStore(tmp_path / "raw"), proc_store)
        loader2.build(ohlcv, macro, tickers)  # should load from cache
        assert tuple(loader2._feature_tensor.shape) == shape1
