"""Ensemble training CLI for Alpha0.

Trains N independent SAC agents (default 5) with different random seeds
and combines them into an :class:`~alpha0.policy.ensemble.EnsemblePolicy`.

Usage::

    # Train 5 agents sequentially
    python scripts/train_ensemble.py

    # Resume (skip agents whose best.pt already exists)
    python scripts/train_ensemble.py --resume

    # Custom settings
    python scripts/train_ensemble.py \\
        --n-agents 5 \\
        --n-steps 500000 \\
        --device cuda \\
        --output-dir artifacts/ensemble/run_01

    # After training, evaluate the ensemble on the test set
    python scripts/train_ensemble.py --eval-only \\
        --ensemble-dir artifacts/ensemble/run_01

Prerequisites:
    1. ``pip install -e ".[dev]"``
    2. Build feature tensors:  ``python scripts/fetch_data.py``

Outputs:
    - Per-agent checkpoints:      ``{output_dir}/agent_XX/best.pt``
    - Ensemble manifest (JSON):   ``{output_dir}/ensemble_manifest.json``
    - Walk-forward summary (CSV): ``{output_dir}/summary.csv``  (if --walk-forward)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from alpha0 import load_config
from alpha0.data.loader import DataLoader
from alpha0.data.storage import ParquetStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("alpha0.train_ensemble")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Alpha0 ensemble training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",       default="config/default.yaml")
    p.add_argument("--n-agents",     type=int, default=None,
                   help="Number of ensemble members (default: config ensemble.n_agents).")
    p.add_argument("--n-steps",      type=int, default=None,
                   help="Steps per agent (default: config training.n_steps).")
    p.add_argument("--device",       default="cpu",
                   choices=["cpu", "cuda", "mps"])
    p.add_argument("--output-dir",   default="artifacts/ensemble/latest",
                   help="Root directory for checkpoints.")
    p.add_argument("--resume",       action="store_true",
                   help="Skip agents whose best.pt already exists.")
    p.add_argument("--eval-only",    action="store_true",
                   help="Skip training; just evaluate an existing ensemble.")
    p.add_argument("--ensemble-dir", default=None,
                   help="Existing ensemble dir for --eval-only.")
    p.add_argument("--walk-forward", action="store_true",
                   help="Run walk-forward validation with the ensemble (slow).")
    return p.parse_args()


def _resolve_device(device_str: str) -> str:
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — using CPU.")
        return "cpu"
    if device_str == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        logger.warning("MPS not available — using CPU.")
        return "cpu"
    return device_str


def _load_data(cfg: dict) -> DataLoader:
    dc = cfg["data"]
    loader = DataLoader(cfg, ParquetStore(dc["raw_dir"]), ParquetStore(dc["processed_dir"]))
    try:
        loader.load()
    except FileNotFoundError:
        logger.error("Feature tensors not found.  Run:  python scripts/fetch_data.py")
        sys.exit(1)
    logger.info("Data loaded: shape=%s | %s – %s",
                tuple(loader._feature_tensor.shape),
                loader.dates[0].date(), loader.dates[-1].date())
    return loader


def main() -> None:
    args   = parse_args()
    cfg    = load_config(args.config)
    device = _resolve_device(args.device)

    loader = _load_data(cfg)

    from alpha0.training.ensemble_trainer import EnsembleTrainer

    n_agents = args.n_agents or cfg.get("ensemble", {}).get("n_agents", 5)
    n_steps  = args.n_steps  or cfg.get("training", {}).get("n_steps", 200_000)
    out_dir  = Path(args.output_dir)

    trainer = EnsembleTrainer(
        cfg           = cfg,
        data_loader   = loader,
        n_agents      = n_agents,
        device        = device,
        artifacts_dir = out_dir,
    )

    # ── Eval-only mode ────────────────────────────────────────────────
    if args.eval_only:
        ens_dir = Path(args.ensemble_dir) if args.ensemble_dir else out_dir
        ensemble = trainer.load_ensemble(ens_dir)

        from alpha0.env.market_env import MarketEnv
        test_env = MarketEnv(cfg, loader, split="test")
        metrics  = trainer.evaluate_ensemble(ensemble, test_env, n_episodes=20)

        logger.info("\n%s", "=" * 60)
        logger.info("ENSEMBLE EVALUATION  (test split, 20 episodes)")
        logger.info("=" * 60)
        for k, v in sorted(metrics.items()):
            logger.info("  %-30s %+.4f", k, v)
        logger.info("=" * 60)
        return

    # ── Training ──────────────────────────────────────────────────────
    seeds = cfg.get("ensemble", {}).get("seeds", list(range(n_agents)))

    logger.info("Starting ensemble training: %d agents × %d steps on %s",
                n_agents, n_steps, device)

    ensemble, result = trainer.train_all(
        n_steps             = n_steps,
        seeds               = seeds,
        resume              = args.resume,
        early_stop_patience = 10,
    )

    # ── Evaluation summary ────────────────────────────────────────────
    logger.info("\n%s", result.summary())

    # ── Optional walk-forward ─────────────────────────────────────────
    if args.walk_forward:
        from alpha0.training.walk_forward import WalkForwardTrainer
        logger.info("Running walk-forward validation with best single agent…")
        wf = WalkForwardTrainer(cfg, loader, device=device, n_steps=n_steps)
        wf_result = wf.run()
        summary_path = out_dir / "walk_forward_summary.csv"
        wf_result.summary_table().to_csv(summary_path, index=False)
        logger.info("Walk-forward summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
