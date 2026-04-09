"""Offline evaluation CLI for Alpha0.

Loads a trained checkpoint (single agent or ensemble), runs a full vectorised
backtest over the test split, compares against benchmarks, and generates a
complete Dashboard report with all charts and a metrics CSV.

Examples::

    # Evaluate best single-agent checkpoint
    python scripts/evaluate.py \\
        --checkpoint artifacts/models/latest/best.pt

    # Evaluate an ensemble
    python scripts/evaluate.py \\
        --ensemble-dir artifacts/ensemble/latest

    # Walk-forward results report
    python scripts/evaluate.py \\
        --checkpoint artifacts/models/latest/best.pt \\
        --walk-forward-csv artifacts/walk_forward/summary.csv

    # Full report with all options
    python scripts/evaluate.py \\
        --checkpoint artifacts/models/latest/best.pt \\
        --split test \\
        --n-episodes 20 \\
        --risk \\
        --vol-target 0.10 \\
        --report-dir artifacts/reports/final_eval
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
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
logger = logging.getLogger("alpha0.evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Alpha0 offline backtest evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",           default="config/default.yaml")
    p.add_argument("--checkpoint",       default=None,
                   help="Path to a single-agent best.pt / final.pt.")
    p.add_argument("--ensemble-dir",     default=None,
                   help="Ensemble root directory (contains agent_00/ …).")
    p.add_argument("--split",            default="test",
                   choices=["train", "val", "test"],
                   help="Which data split to evaluate on.")
    p.add_argument("--n-episodes",       type=int, default=20,
                   help="Number of episodes to average over.")
    p.add_argument("--deterministic",    action="store_true", default=True,
                   help="Use deterministic (mean) policy actions.")
    p.add_argument("--risk",             action="store_true",
                   help="Apply RiskManager during evaluation.")
    p.add_argument("--vol-target",       type=float, default=None,
                   help="Annualised vol target for VolatilityTargeter (e.g. 0.10).")
    p.add_argument("--benchmarks",       action="store_true", default=True,
                   help="Include SPY/EW benchmark comparison.")
    p.add_argument("--report-dir",       default=None,
                   help="Output directory for dashboard report. "
                        "Defaults to artifacts/reports/{timestamp}.")
    p.add_argument("--walk-forward-csv", default=None,
                   help="Path to walk-forward summary CSV to include in report.")
    p.add_argument("--device",           default="cpu")
    return p.parse_args()


def _load_data(cfg: dict) -> DataLoader:
    dc = cfg["data"]
    loader = DataLoader(cfg, ParquetStore(dc["raw_dir"]), ParquetStore(dc["processed_dir"]))
    try:
        loader.load()
    except FileNotFoundError:
        logger.error("Feature tensors not found. Run:  python scripts/fetch_data.py")
        sys.exit(1)
    logger.info("Data loaded: shape=%s | %s – %s",
                tuple(loader._feature_tensor.shape),
                loader.dates[0].date(), loader.dates[-1].date())
    return loader


def _load_policy(args, cfg):
    from alpha0.training.ensemble_trainer import EnsembleTrainer
    trainer = EnsembleTrainer(cfg, data_loader=None, device=args.device)
    if args.ensemble_dir:
        policy = trainer.load_ensemble(args.ensemble_dir)
        logger.info("Loaded ensemble from %s", args.ensemble_dir)
    elif args.checkpoint:
        policy = trainer._load_policy(Path(args.checkpoint))
        policy.eval()
        logger.info("Loaded checkpoint from %s", args.checkpoint)
    else:
        logger.error("Provide --checkpoint or --ensemble-dir.")
        sys.exit(1)
    return policy


def _run_episodes(policy, env, cfg, n_episodes: int, risk_mgr=None, pos_sizer=None) -> pd.Series:
    """Run N episodes and return a concatenated daily-return Series."""
    all_returns: list[float] = []

    if risk_mgr:
        risk_mgr.reset(cfg["env"]["initial_portfolio_value"])
    if pos_sizer:
        pos_sizer.reset()

    with torch.no_grad():
        for ep in range(n_episodes):
            obs_dict, _ = env.reset()
            ep_returns: list[float] = []
            done = False
            while not done:
                obs_t  = torch.tensor(obs_dict["obs"][None],       dtype=torch.float32)
                port_t = torch.tensor(obs_dict["portfolio"][None], dtype=torch.float32)

                # Handle both AlphaPolicy and EnsemblePolicy
                out = policy.act(obs_t, port_t)
                if isinstance(out, tuple):
                    action_t = out[0]
                else:
                    action_t = out
                action = action_t.squeeze(0).cpu().numpy()

                obs_dict, _, term, trunc, info = env.step(action)
                ep_returns.append(info.get("portfolio_return", 0.0))
                done = term or trunc

            all_returns.extend(ep_returns)

    return pd.Series(all_returns, name="daily_return")


def _print_metrics_table(metrics: dict[str, float], title: str = "METRICS") -> None:
    """Pretty-print a metrics dict."""
    divider = "=" * 55
    print(f"\n{divider}")
    print(f"  {title}")
    print(divider)
    display_order = [
        ("total_return",        "Total Return"),
        ("annualised_return",   "Ann. Return"),
        ("annualised_vol",      "Ann. Volatility"),
        ("sharpe_ratio",        "Sharpe Ratio"),
        ("sortino_ratio",       "Sortino Ratio"),
        ("calmar_ratio",        "Calmar Ratio"),
        ("omega_ratio",         "Omega Ratio"),
        ("max_drawdown",        "Max Drawdown"),
        ("max_drawdown_days",   "Max DD Duration (days)"),
        ("ulcer_index",         "Ulcer Index"),
        ("recovery_factor",     "Recovery Factor"),
        ("var_95",              "VaR 95%"),
        ("cvar_95",             "CVaR 95%"),
        ("hit_rate",            "Hit Rate"),
        ("profit_factor",       "Profit Factor"),
        ("skewness",            "Skewness"),
        ("kurtosis",            "Excess Kurtosis"),
        ("beta",                "Beta (vs benchmark)"),
        ("alpha",               "Jensen's Alpha (ann.)"),
        ("information_ratio",   "Information Ratio"),
        ("turnover",            "Mean Daily Turnover"),
    ]
    for key, label in display_order:
        if key in metrics:
            val = metrics[key]
            # Format percentages
            if key in ("total_return", "annualised_return", "annualised_vol",
                       "max_drawdown", "var_95", "cvar_95", "hit_rate", "turnover"):
                print(f"  {label:<30} {val:>+.2%}")
            elif key == "max_drawdown_days":
                print(f"  {label:<30} {int(val):>5d}")
            else:
                print(f"  {label:<30} {val:>+.4f}")
    print(divider)


def main() -> None:
    args   = parse_args()
    cfg    = load_config(args.config)
    loader = _load_data(cfg)
    policy = _load_policy(args, cfg)

    # ── Build optional risk/sizing objects ────────────────────────────
    risk_mgr  = None
    pos_sizer = None

    if args.risk:
        from alpha0.risk.manager import RiskManager
        risk_mgr = RiskManager(cfg)
        logger.info("RiskManager enabled.")

    if args.vol_target is not None:
        from alpha0.risk.position_sizer import VolatilityTargeter
        pos_sizer = VolatilityTargeter(
            target_annual_vol = args.vol_target,
            lookback          = cfg.get("position_sizing", {}).get("vol_lookback", 21),
            max_leverage      = cfg.get("position_sizing", {}).get("max_leverage", 1.0),
            min_cash          = cfg["env"]["min_cash_weight"],
        )
        logger.info("VolatilityTargeter enabled (target=%.0f%%)", args.vol_target * 100)

    # ── Evaluate policy ───────────────────────────────────────────────
    from alpha0.env.market_env import MarketEnv
    from alpha0.eval.metrics import compute_metrics

    env = MarketEnv(cfg, loader, split=args.split, risk_manager=risk_mgr)
    logger.info("Evaluating on split=%s  (%d episodes, deterministic=%s)",
                args.split, args.n_episodes, args.deterministic)

    returns = _run_episodes(policy, env, cfg, args.n_episodes, risk_mgr, pos_sizer)
    rf_rate = cfg["eval"].get("risk_free_rate", 0.04)
    metrics = compute_metrics(returns, rf_rate=rf_rate)
    _print_metrics_table(metrics, f"ALPHA0 — {args.split.upper()} SPLIT ({args.n_episodes} eps)")

    # ── Benchmarks ────────────────────────────────────────────────────
    from alpha0.eval.backtest import BacktestResult, Backtest
    from alpha0.eval.benchmarks import BenchmarkRunner

    # Build a synthetic BacktestResult for the policy (no weights available
    # from episode runs, so we create a minimal one for the dashboard)
    split_start, split_end = loader._split_indices[args.split]
    split_dates = loader.dates[split_start:split_end]

    # Align returns to dates (pad/truncate as needed)
    idx_len = min(len(returns), len(split_dates))
    ret_series = pd.Series(
        returns.values[:idx_len],
        index=pd.DatetimeIndex(split_dates[:idx_len]),
        name="daily_return",
    )
    pv = (1.0 + ret_series).cumprod() * cfg["eval"]["backtest"]["initial_capital"]

    alpha0_result = BacktestResult(
        portfolio_value = pv,
        daily_returns   = ret_series,
        weights         = pd.DataFrame(index=ret_series.index),
        metrics         = metrics,
    )

    results_for_report: dict[str, BacktestResult] = {"Alpha0": alpha0_result}
    benchmark_returns: pd.Series | None = None

    if args.benchmarks and loader._close_prices is not None:
        n_assets = cfg["env"]["n_assets"]
        tickers  = loader.tickers[:n_assets]
        prices_np = loader._close_prices[split_start:split_end, :n_assets].numpy()
        prices_df = pd.DataFrame(prices_np, index=pd.DatetimeIndex(split_dates), columns=tickers)

        runner = BenchmarkRunner(cfg)
        spy_prices = prices_df.iloc[:, 0]

        bm_results = runner.run_all(prices_df, spy_prices=spy_prices)
        benchmark_returns = spy_prices.pct_change().dropna()

        for bm_name, bm_result in bm_results.items():
            bm_result.metrics = compute_metrics(
                bm_result.daily_returns,
                benchmark_returns=benchmark_returns,
                rf_rate=rf_rate,
            )
            results_for_report[bm_name] = bm_result
            _print_metrics_table(bm_result.metrics, bm_name.upper())

        # Re-compute alpha0 metrics with benchmark
        alpha0_result.metrics = compute_metrics(
            ret_series,
            benchmark_returns=benchmark_returns,
            rf_rate=rf_rate,
        )
        results_for_report["Alpha0"] = alpha0_result
        _print_metrics_table(alpha0_result.metrics, "ALPHA0 (with benchmark metrics)")

    # ── Dashboard report ──────────────────────────────────────────────
    from alpha0.eval.dashboard import Dashboard

    wf_df = None
    if args.walk_forward_csv and Path(args.walk_forward_csv).exists():
        wf_df = pd.read_csv(args.walk_forward_csv)
        logger.info("Loaded walk-forward summary from %s", args.walk_forward_csv)

    dashboard = Dashboard(cfg)
    out_dir = dashboard.generate_report(
        results_for_report,
        output_dir      = args.report_dir,
        walk_forward_df = wf_df,
    )
    logger.info("Full report saved to:  %s", out_dir)
    print(f"\n  Report saved to: {out_dir}\n")


if __name__ == "__main__":
    main()
