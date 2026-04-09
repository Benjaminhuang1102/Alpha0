"""CLI script: fetch all data and build processed feature tensors.

Usage::

    # Normal run (uses cached data where available)
    python scripts/fetch_data.py

    # Force re-fetch everything from source
    python scripts/fetch_data.py --force-refresh

    # Use a custom config file
    python scripts/fetch_data.py --config path/to/config.yaml

    # Skip FRED (useful if FRED_API_KEY is not set; macro will be zero-filled)
    python scripts/fetch_data.py --skip-fred

Prerequisites:
    1. Install the package: pip install -e ".[dev]"
    2. Set your FRED API key (free):
         export FRED_API_KEY=your_key_here
       Registration: https://fred.stlouisfed.org/docs/api/api_key.html

After this script completes successfully:
    - Raw OHLCV cache:   data/cache/ohlcv/{TICKER}.parquet
    - FRED macro cache:  data/cache/fred/{alias}.parquet
    - YF macro cache:    data/cache/macro/{alias}.parquet
    - Universe list:     data/universe/constituents.parquet
    - Feature tensor:    data/processed/features.pt (+ metadata.json)
    - Close prices:      data/processed/close_prices.pt

The processed tensors are ready for use by MarketEnv and the training loop.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make sure the project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from alpha0 import load_config
from alpha0.data.fetcher import DataFetcher
from alpha0.data.loader import DataLoader
from alpha0.data.storage import ParquetStore
from alpha0.data.universe import UniverseManager
from alpha0.eval.backtest import Backtest
from alpha0.eval.benchmarks import BenchmarkRunner
from alpha0.eval.dashboard import Dashboard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("alpha0.fetch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alpha0 data fetch + feature build pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore all cached data and re-fetch from source.",
    )
    parser.add_argument(
        "--skip-fred",
        action="store_true",
        help="Skip FRED fetching (macro will be zero-filled). "
             "Use when FRED_API_KEY is not set.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="After building tensors, run the equal-weight baseline "
             "backtest and save a dashboard report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Config ──────────────────────────────────────────────────────
    cfg = load_config(args.config)
    dc = cfg["data"]
    start = dc["start_date"]
    end   = dc["end_date"]

    # ── 2. Storage layers ──────────────────────────────────────────────
    cache_store     = ParquetStore(dc["cache_dir"])
    universe_store  = ParquetStore(dc["universe_dir"])
    processed_store = ParquetStore(dc["processed_dir"])
    raw_store       = ParquetStore(dc["raw_dir"])

    # ── 3. Universe ────────────────────────────────────────────────────
    logger.info("=== Step 1: Universe ===")
    universe_mgr = UniverseManager(cfg, universe_store)
    tickers = universe_mgr.fetch_current_constituents()
    logger.info("Candidate universe: %d tickers", len(tickers))

    # ── 4. Fetch OHLCV ─────────────────────────────────────────────────
    logger.info("=== Step 2: OHLCV (yfinance) ===")
    fetcher = DataFetcher(cfg, cache_store)
    ohlcv = fetcher.fetch_ohlcv(
        tickers, start, end, force_refresh=args.force_refresh
    )

    if ohlcv.empty:
        logger.error("OHLCV download failed or returned empty data. Aborting.")
        sys.exit(1)

    logger.info(
        "OHLCV shape: %s | date range: %s – %s",
        ohlcv.shape,
        ohlcv.index.min().date(),
        ohlcv.index.max().date(),
    )

    # ── 5. Filter universe ─────────────────────────────────────────────
    logger.info("=== Step 3: Build filtered universe ===")
    close = ohlcv["Close"] if "Close" in ohlcv.columns.get_level_values("field") else ohlcv.iloc[:, 0]
    volume = ohlcv["Volume"] if "Volume" in ohlcv.columns.get_level_values("field") else pd.DataFrame()

    universe = universe_mgr.build_universe(tickers, close, volume)
    universe_mgr.save_universe(universe)
    logger.info("Final universe: %d tickers", len(universe))

    # ── 6. Fetch FRED macro ────────────────────────────────────────────
    macro_df = pd.DataFrame()

    macro_parts: list[pd.DataFrame] = []

    if not args.skip_fred:
        logger.info("=== Step 4a: FRED macro series ===")
        if not os.environ.get("FRED_API_KEY"):
            logger.warning(
                "FRED_API_KEY not set. Skipping FRED fetch.\n"
                "  Register for a free key at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html\n"
                "  Then set: export FRED_API_KEY=your_key_here\n"
                "  Or use --skip-fred to suppress this warning."
            )
        else:
            try:
                fred_df = fetcher.fetch_fred_series(
                    dc["fred_series"], start, end, force_refresh=args.force_refresh
                )
                if not fred_df.empty:
                    macro_parts.append(fred_df)
                    logger.info("FRED series fetched: %s", list(fred_df.columns))
            except Exception as exc:
                logger.warning("FRED fetch failed: %s. Continuing without FRED data.", exc)
    else:
        logger.info("Skipping FRED (--skip-fred).")

    # ── 7. Fetch yfinance macro (GC=F, CL=F) ──────────────────────────
    logger.info("=== Step 4b: yfinance macro (commodities) ===")
    try:
        yf_macro = fetcher.fetch_macro_yfinance(
            dc["yfinance_macro"], start, end, force_refresh=args.force_refresh
        )
        if not yf_macro.empty:
            macro_parts.append(yf_macro)
            logger.info("yfinance macro fetched: %s", list(yf_macro.columns))
    except Exception as exc:
        logger.warning("yfinance macro fetch failed: %s.", exc)

    # Combine all macro sources into one DataFrame
    if macro_parts:
        macro_df = pd.concat(macro_parts, axis=1).sort_index()
    else:
        macro_df = pd.DataFrame()

    if macro_df.empty:
        logger.warning("No macro data available — macro features will be zero-filled.")
        macro_df = pd.DataFrame(index=ohlcv.index)

    # ── 8. Build feature tensors ───────────────────────────────────────
    logger.info("=== Step 5: Build feature tensors ===")
    loader = DataLoader(cfg, raw_store, processed_store)

    # Filter OHLCV to universe tickers
    uni_tickers = [t for t in universe if t in ohlcv.columns.get_level_values("ticker")]
    ohlcv_uni = ohlcv.loc[:, ohlcv.columns.get_level_values("ticker").isin(uni_tickers)]

    loader.build(
        ohlcv_uni,
        macro_df,
        uni_tickers,
        force_rebuild=args.force_refresh,
    )

    tensor_shape = tuple(loader._feature_tensor.shape)
    split_info = loader._split_indices
    logger.info("Feature tensor shape: %s", tensor_shape)
    logger.info(
        "Split indices — train: [0, %d), val: [%d, %d), test: [%d, %d)",
        split_info["train"][1],
        split_info["val"][0], split_info["val"][1],
        split_info["test"][0], split_info["test"][1],
    )
    logger.info(
        "Date range: %s – %s",
        loader.dates[0].date(),
        loader.dates[-1].date(),
    )

    # ── 9. Sanity checks ───────────────────────────────────────────────
    logger.info("=== Step 6: Sanity checks ===")
    _run_sanity_checks(loader, cfg)

    # ── 10. Optional baseline backtest ────────────────────────────────
    if args.baseline:
        logger.info("=== Step 7: Equal-weight baseline backtest ===")
        _run_baseline(loader, cfg, close, uni_tickers)

    logger.info("=== Data pipeline complete ===")


def _run_sanity_checks(loader: DataLoader, cfg: dict) -> None:
    """Quick checks on the feature tensor."""
    import torch
    t = loader._feature_tensor

    nan_count = torch.isnan(t).sum().item()
    inf_count = torch.isinf(t).sum().item()
    max_val   = t.abs().max().item()

    logger.info("NaN count: %d | Inf count: %d | Max abs value: %.4f", nan_count, inf_count, max_val)

    if nan_count > 0:
        logger.warning("WARNING: %d NaN values in feature tensor!", nan_count)
    if inf_count > 0:
        logger.warning("WARNING: %d Inf values in feature tensor!", inf_count)
    if max_val > cfg["data"]["normalization"]["clip"] + 0.1:
        logger.warning("WARNING: Max value %.4f exceeds clip threshold!", max_val)

    splits = loader._split_indices
    for name, (s, e) in splits.items():
        logger.info("  Split %-6s: %d trading days (%s – %s)",
                    name, e - s,
                    loader.dates[s].date() if s < len(loader.dates) else "N/A",
                    loader.dates[e - 1].date() if e > 0 and e <= len(loader.dates) else "N/A")


def _run_baseline(
    loader: DataLoader, cfg: dict, close: pd.DataFrame, universe: list[str]
) -> None:
    """Run equal-weight monthly rebalance baseline and print metrics."""
    dc = cfg["data"]
    val_end   = pd.Timestamp(dc["val_end"])
    test_start_idx = loader._split_indices["test"][0]

    if test_start_idx >= len(loader.dates):
        logger.warning("No test period data available for baseline.")
        return

    test_start = loader.dates[test_start_idx]
    test_prices = close.loc[test_start:].reindex(columns=universe).dropna(axis=1, how="all")

    if test_prices.empty:
        logger.warning("No price data for test period baseline.")
        return

    runner = BenchmarkRunner(cfg)
    results = runner.run_all(test_prices, spy_prices=test_prices.iloc[:, 0])

    logger.info("\n%s", "=" * 60)
    logger.info("BASELINE BACKTEST RESULTS (test period)")
    logger.info("%s", "=" * 60)
    for name, result in results.items():
        m = result.metrics
        logger.info(
            "%-20s | Ret: %+.1f%% | Vol: %.1f%% | Sharpe: %.2f | MaxDD: %.1f%%",
            name,
            m.get("annualised_return", 0) * 100,
            m.get("annualised_vol",    0) * 100,
            m.get("sharpe_ratio",      0),
            m.get("max_drawdown",      0) * 100,
        )

    # Save dashboard
    dashboard = Dashboard(cfg)
    out = dashboard.generate_report(
        {name: result for name, result in results.items()}
    )
    logger.info("Baseline dashboard saved to: %s", out)


if __name__ == "__main__":
    main()
