"""Training CLI for Alpha0 SAC agent.

Examples::

    # Single training run (train split)
    python scripts/train.py

    # Walk-forward cross-validation across all configured folds
    python scripts/train.py --walk-forward

    # Single fold only (0-indexed)
    python scripts/train.py --walk-forward --fold 2

    # Custom settings
    python scripts/train.py \\
        --config config/default.yaml \\
        --device cuda \\
        --n-steps 500000 \\
        --save-dir artifacts/models/run_01

Prerequisites:
    1. ``pip install -e ".[dev]"``
    2. Build the feature tensors first:
         python scripts/fetch_data.py
    3. (Optional) CUDA device for faster training.

Outputs:
    - Checkpoints:  ``{save_dir}/best.pt`` and ``{save_dir}/final.pt``
    - Walk-forward: ``artifacts/walk_forward/fold_XX/best.pt``
    - Training metrics logged to stdout (and optionally TensorBoard).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
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
logger = logging.getLogger("alpha0.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alpha0 SAC training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="PyTorch device.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Total environment steps. Overrides config if set.",
    )
    parser.add_argument(
        "--save-dir",
        default="artifacts/models/latest",
        help="Directory for model checkpoints.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override.",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward cross-validation across all configured folds.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Run a single fold (0-indexed). Only used with --walk-forward.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Evaluate on val env every N steps.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume from.",
    )
    return parser.parse_args()


def _resolve_device(device_str: str) -> str:
    """Validate the device; fall back to CPU if CUDA/MPS is unavailable."""
    if device_str == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
    elif device_str == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS requested but not available. Falling back to CPU.")
            return "cpu"
    return device_str


def _load_data(cfg: dict) -> DataLoader:
    """Load pre-built feature tensors from disk.

    Raises a clear error if the tensors haven't been built yet.
    """
    dc = cfg["data"]
    raw_store       = ParquetStore(dc["raw_dir"])
    processed_store = ParquetStore(dc["processed_dir"])

    loader = DataLoader(cfg, raw_store, processed_store)

    try:
        loader.load()
    except FileNotFoundError as exc:
        logger.error(
            "Feature tensors not found. Run the data pipeline first:\n"
            "  python scripts/fetch_data.py\n"
            "Error: %s", exc,
        )
        sys.exit(1)

    logger.info(
        "Loaded feature tensor: shape=%s | dates: %s – %s",
        tuple(loader._feature_tensor.shape),
        loader.dates[0].date(),
        loader.dates[-1].date(),
    )
    return loader


def _single_run(args: argparse.Namespace, cfg: dict, loader: DataLoader) -> None:
    """Train a single SAC agent on the train split and evaluate on test."""
    from alpha0.env.market_env import MarketEnv
    from alpha0.training.sac import SACTrainer

    device = _resolve_device(args.device)

    if args.seed is not None:
        cfg.setdefault("training", {})["seed"] = args.seed

    n_steps = args.n_steps or cfg.get("training", {}).get("n_steps", 200_000)
    eval_every = args.eval_every or max(1_000, n_steps // 50)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_env = MarketEnv(cfg, loader, split="train")
    val_env   = MarketEnv(cfg, loader, split="val")
    test_env  = MarketEnv(cfg, loader, split="test")

    trainer = SACTrainer(cfg, device=device)

    if args.resume:
        trainer.load(args.resume)
        logger.info("Resumed from checkpoint: %s", args.resume)

    param_info = trainer.policy.count_parameters()
    logger.info("Policy parameters: %s", param_info)
    logger.info(
        "Training for %d steps on device=%s | "
        "batch=%d | lr=%.0e | gamma=%.3f",
        n_steps, device,
        cfg["training"].get("batch_size", 256),
        cfg["training"].get("learning_rate", 3e-4),
        cfg["training"].get("gamma", 0.99),
    )

    t0 = time.perf_counter()
    metrics = trainer.train(
        env                 = train_env,
        n_steps             = n_steps,
        val_env             = val_env,
        eval_every          = eval_every,
        save_dir            = save_dir,
        early_stop_patience = 10,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Training complete in %.1f minutes.", elapsed / 60)

    # Final test evaluation
    test_sharpe = trainer.evaluate(test_env, n_episodes=10, deterministic=True)
    logger.info("Test Sharpe (10 episodes, deterministic): %.3f", test_sharpe)

    _print_training_summary(metrics, test_sharpe)


def _walk_forward_run(args: argparse.Namespace, cfg: dict, loader: DataLoader) -> None:
    """Run walk-forward validation across all (or one) fold(s)."""
    from alpha0.training.walk_forward import WalkForwardTrainer

    device = _resolve_device(args.device)
    n_steps = args.n_steps  # None → WalkForwardTrainer reads from config

    wf = WalkForwardTrainer(
        cfg           = cfg,
        data_loader   = loader,
        device        = device,
        n_steps       = n_steps,
        artifacts_dir = "artifacts/walk_forward",
    )

    folds = [args.fold] if args.fold is not None else None
    t0 = time.perf_counter()
    result = wf.run(folds=folds)
    elapsed = time.perf_counter() - t0
    logger.info("Walk-forward complete in %.1f minutes.", elapsed / 60)

    # Save summary CSV
    summary_path = Path("artifacts/walk_forward/summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    result.summary_table().to_csv(summary_path, index=False)
    logger.info("Summary saved to %s", summary_path)


def _print_training_summary(
    metrics: dict[str, list[float]], test_sharpe: float
) -> None:
    """Print a brief training summary to stdout."""
    n_updates = len(metrics.get("critic_loss", []))
    if n_updates == 0:
        return

    ep_rewards = metrics.get("episode_reward", [])
    recent_rewards = ep_rewards[-50:] if ep_rewards else []

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Total gradient updates:    {n_updates:,}")
    print(f"  Final alpha (entropy coef): {metrics['alpha'][-1]:.4f}" if metrics.get("alpha") else "")
    print(f"  Mean critic loss (last 100): {_tail_mean(metrics['critic_loss'], 100):.4f}")
    print(f"  Mean actor  loss (last 100): {_tail_mean(metrics['actor_loss'],  100):.4f}")
    if recent_rewards:
        print(f"  Mean episode reward (last 50 eps): {float(sum(recent_rewards)/len(recent_rewards)):.4f}")
    print(f"  Test Sharpe (10 episodes): {test_sharpe:.3f}")
    print("=" * 60)


def _tail_mean(lst: list[float], n: int) -> float:
    tail = lst[-n:] if lst else []
    return float(sum(tail) / len(tail)) if tail else 0.0


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    logger.info("Loading feature tensors...")
    loader = _load_data(cfg)

    if args.walk_forward:
        _walk_forward_run(args, cfg, loader)
    else:
        _single_run(args, cfg, loader)


if __name__ == "__main__":
    main()
