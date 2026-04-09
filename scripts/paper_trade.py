"""Paper trading CLI for Alpha0.

Runs a trained policy (single agent or ensemble) in simulation over a
historical date range, applying full risk management and execution cost
modelling.  Produces a BacktestResult-compatible report identical to the
offline backtest pipeline.

Examples::

    # Paper trade with the best single-agent checkpoint
    python scripts/paper_trade.py \\
        --checkpoint artifacts/models/latest/best.pt \\
        --start 2022-01-01 --end 2024-12-31

    # Paper trade with an ensemble
    python scripts/paper_trade.py \\
        --ensemble-dir artifacts/ensemble \\
        --start 2022-01-01 --end 2024-12-31 \\
        --vol-target 0.10 \\
        --risk

    # Compare vs benchmarks and generate full dashboard
    python scripts/paper_trade.py \\
        --checkpoint artifacts/models/latest/best.pt \\
        --start 2022-01-01 --end 2024-12-31 \\
        --benchmark --report
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch

from alpha0 import load_config
from alpha0.data.loader import DataLoader
from alpha0.data.storage import ParquetStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("alpha0.paper_trade")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Alpha0 paper trading simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",       default="config/default.yaml")
    p.add_argument("--checkpoint",   default=None,
                   help="Path to a single-agent best.pt checkpoint.")
    p.add_argument("--ensemble-dir", default=None,
                   help="Directory with agent_00/, agent_01/, … subdirs.")
    p.add_argument("--start",        required=True,
                   help="Start date  YYYY-MM-DD  (inclusive).")
    p.add_argument("--end",          required=True,
                   help="End date    YYYY-MM-DD  (inclusive).")
    p.add_argument("--risk",         action="store_true",
                   help="Enable RiskManager (circuit breakers, sector limits).")
    p.add_argument("--vol-target",   type=float, default=None,
                   help="Annual vol target for VolatilityTargeter (e.g. 0.10). "
                        "Disabled if not set.")
    p.add_argument("--regime",       action="store_true",
                   help="Enable RegimeDetector and shade equity curve.")
    p.add_argument("--benchmark",    action="store_true",
                   help="Run SPY/EW benchmarks for comparison.")
    p.add_argument("--report",       action="store_true",
                   help="Generate full Dashboard report.")
    p.add_argument("--output-dir",   default="artifacts/paper_trading",
                   help="Directory to save results.")
    p.add_argument("--device",       default="cpu")
    return p.parse_args()


def _load_policy(args, cfg):
    """Load a single-agent policy or an ensemble."""
    if args.ensemble_dir:
        from alpha0.training.ensemble_trainer import EnsembleTrainer
        trainer = EnsembleTrainer(cfg, data_loader=None, device=args.device)
        policy = trainer.load_ensemble(args.ensemble_dir)
        logger.info("Loaded ensemble from %s", args.ensemble_dir)
    elif args.checkpoint:
        from alpha0.training.ensemble_trainer import EnsembleTrainer
        trainer = EnsembleTrainer(cfg, data_loader=None, device=args.device)
        policy = trainer._load_policy(Path(args.checkpoint))
        logger.info("Loaded single checkpoint from %s", args.checkpoint)
    else:
        logger.error("Provide --checkpoint or --ensemble-dir.")
        sys.exit(1)
    return policy


def _build_risk_manager(args, cfg):
    if not args.risk:
        return None
    from alpha0.risk.manager import RiskManager
    mgr = RiskManager(cfg)
    logger.info("RiskManager enabled (soft=%.0f%%, hard=%.0f%%)",
                cfg["risk"]["max_drawdown_soft"] * 100,
                cfg["risk"]["max_drawdown_hard"] * 100)
    return mgr


def _build_position_sizer(args, cfg):
    if args.vol_target is None:
        return None
    from alpha0.risk.position_sizer import VolatilityTargeter
    sizer = VolatilityTargeter(
        target_annual_vol = args.vol_target,
        lookback          = cfg.get("position_sizing", {}).get("vol_lookback", 21),
        max_leverage      = cfg.get("position_sizing", {}).get("max_leverage", 1.0),
        min_cash          = cfg["env"]["min_cash_weight"],
    )
    logger.info("VolatilityTargeter enabled (target=%.0f%% annual vol)", args.vol_target * 100)
    return sizer


def _build_regime_detector(args, cfg):
    if not args.regime:
        return None
    from alpha0.risk.regime import RegimeDetector
    rc = cfg.get("regime", {})
    det = RegimeDetector(
        trend_lookback      = rc.get("trend_lookback",      200),
        vol_lookback        = rc.get("vol_lookback",        21),
        vol_high_multiplier = rc.get("vol_high_multiplier", 1.5),
        long_run_vol_window = rc.get("long_run_vol_window", 252),
        min_history         = rc.get("min_history",         60),
    )
    logger.info("RegimeDetector enabled.")
    return det


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    # ── 1. Load data ──────────────────────────────────────────────────
    dc = cfg["data"]
    loader = DataLoader(cfg, ParquetStore(dc["raw_dir"]), ParquetStore(dc["processed_dir"]))
    try:
        loader.load()
    except FileNotFoundError:
        logger.error("Feature tensors not found. Run:  python scripts/fetch_data.py")
        sys.exit(1)

    # ── 2. Load policy ────────────────────────────────────────────────
    policy = _load_policy(args, cfg)

    # ── 3. Build risk / sizing / regime objects ───────────────────────
    risk_mgr    = _build_risk_manager(args, cfg)
    pos_sizer   = _build_position_sizer(args, cfg)
    regime_det  = _build_regime_detector(args, cfg)

    # ── 4. Paper trade ────────────────────────────────────────────────
    from alpha0.execution.paper_trader import PaperTrader

    trader = PaperTrader(
        policy          = policy,
        cfg             = cfg,
        risk_manager    = risk_mgr,
        position_sizer  = pos_sizer,
        regime_detector = regime_det,
    )

    result, log = trader.run(
        data_loader = loader,
        start_date  = args.start,
        end_date    = args.end,
    )

    # ── 5. Print metrics ──────────────────────────────────────────────
    m = result.metrics
    logger.info("\n%s", "=" * 60)
    logger.info("PAPER TRADING RESULTS  %s – %s", args.start, args.end)
    logger.info("=" * 60)
    for k in ("annualised_return", "annualised_vol", "sharpe_ratio",
              "sortino_ratio", "max_drawdown", "calmar_ratio",
              "hit_rate", "var_95", "cvar_95"):
        if k in m:
            logger.info("  %-30s %+.4f", k, m[k])
    if log.circuit_breaker_days:
        logger.info("  Circuit breaker days:    %d", len(log.circuit_breaker_days))
    logger.info("=" * 60)

    # ── 6. Save trades log ────────────────────────────────────────────
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fills_df = trader.fills_dataframe()
    if not fills_df.empty:
        fills_path = out / "fills.csv"
        fills_df.to_csv(fills_path, index=False)
        logger.info("Fills saved to %s", fills_path)

    # ── 7. Benchmark comparison ───────────────────────────────────────
    results_for_report = {"Alpha0": result}

    if args.benchmark:
        from alpha0.eval.benchmarks import BenchmarkRunner
        try:
            tickers   = loader.tickers
            n_assets  = cfg["env"]["n_assets"]
            all_tickers = tickers[:n_assets]

            # Reconstruct price DataFrame from close_prices tensor
            dates_range = [d for d in loader.dates
                           if pd.Timestamp(args.start) <= d <= pd.Timestamp(args.end)]
            if dates_range and loader._close_prices is not None:
                start_i = list(loader.dates).index(dates_range[0])
                end_i   = list(loader.dates).index(dates_range[-1]) + 1
                prices_np = loader._close_prices[start_i:end_i].numpy()
                prices_df = pd.DataFrame(
                    prices_np,
                    index=pd.DatetimeIndex(dates_range),
                    columns=all_tickers,
                )
                runner = BenchmarkRunner(cfg)
                bm_results = runner.run_all(prices_df)
                results_for_report.update(bm_results)
                for name, bm in bm_results.items():
                    bm_m = bm.metrics
                    logger.info("%-20s | Ann.Ret=%+.1f%% | Sharpe=%.2f | MaxDD=%.1f%%",
                                name, bm_m.get("annualised_return", 0) * 100,
                                bm_m.get("sharpe_ratio", 0),
                                bm_m.get("max_drawdown", 0) * 100)
        except Exception as exc:
            logger.warning("Benchmark comparison failed: %s", exc)

    # ── 8. Dashboard report ───────────────────────────────────────────
    if args.report:
        from alpha0.eval.dashboard import Dashboard
        regime_series = None
        if log.regime_history and log.dates:
            from alpha0.risk.regime import MarketRegime
            regime_vals = []
            for r in log.regime_history:
                try:
                    regime_vals.append(MarketRegime(r) if r else MarketRegime.BULL_LOW_VOL)
                except ValueError:
                    regime_vals.append(MarketRegime.BULL_LOW_VOL)
            regime_series = pd.Series(regime_vals, index=pd.DatetimeIndex(log.dates))

        dashboard = Dashboard(cfg)
        report_dir = dashboard.generate_report(
            results_for_report,
            output_dir    = out / "report",
            regime_series = regime_series,
        )
        logger.info("Dashboard report saved to %s", report_dir)


if __name__ == "__main__":
    main()
