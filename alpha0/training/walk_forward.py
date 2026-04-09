"""Walk-forward validation for Alpha0 training.

Walk-forward cross-validation mimics live deployment: the model is trained on
historical data up to a cutoff, validated on the immediately following period,
then evaluated on the out-of-sample test period.  No future data leaks into
any previous fold.

Fold structure (from config/default.yaml):
  Each fold has: train_end, val_end, test_end
  Train data:   [train_start, train_end]
  Val data:     (train_end,   val_end]
  Test data:    (val_end,     test_end]

Usage::

    from alpha0 import load_config
    from alpha0.data.loader import DataLoader
    from alpha0.training.walk_forward import WalkForwardTrainer

    cfg = load_config()
    loader = DataLoader(cfg, raw_store, processed_store)
    wf = WalkForwardTrainer(cfg, loader, device="cuda")
    results = wf.run()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results for one walk-forward fold.

    Parameters
    ----------
    fold_idx:
        Zero-based fold index.
    train_end:
        Last training date (inclusive).
    val_end:
        Last validation date (inclusive).
    test_end:
        Last test date (inclusive).
    val_sharpe:
        Best validation Sharpe during training (used for early stopping).
    test_metrics:
        Metric dict from the test-period backtest.
    train_metrics_history:
        Training loss/alpha history from :class:`~alpha0.training.sac.SACTrainer`.
    checkpoint_path:
        Path to the saved model checkpoint for this fold.
    """

    fold_idx: int
    train_end: str
    val_end: str
    test_end: str
    val_sharpe: float = 0.0
    test_metrics: dict[str, float] = field(default_factory=dict)
    train_metrics_history: dict[str, list[float]] = field(default_factory=dict)
    checkpoint_path: str | None = None


@dataclass
class WalkForwardResult:
    """Aggregated results across all walk-forward folds.

    Parameters
    ----------
    folds:
        Per-fold results.
    aggregate:
        Mean and std of test metrics across folds.
    """

    folds: list[FoldResult]
    aggregate: dict[str, dict[str, float]] = field(default_factory=dict)

    def summary_table(self) -> pd.DataFrame:
        """Return a DataFrame with one row per fold, key test metrics as columns."""
        rows = []
        for r in self.folds:
            row = {
                "fold":        r.fold_idx,
                "train_end":   r.train_end,
                "val_end":     r.val_end,
                "test_end":    r.test_end,
                "val_sharpe":  round(r.val_sharpe, 3),
            }
            row.update({k: round(v, 4) for k, v in r.test_metrics.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def print_summary(self) -> None:
        """Print the summary table and aggregate statistics to stdout."""
        df = self.summary_table()
        print("\n" + "=" * 80)
        print("WALK-FORWARD VALIDATION RESULTS")
        print("=" * 80)
        print(df.to_string(index=False))
        print()
        print("Aggregate (mean ± std across folds):")
        for k, v in self.aggregate.items():
            print(f"  {k:30s}: {v['mean']:+.4f} ± {v['std']:.4f}")
        print("=" * 80)


class WalkForwardTrainer:
    """Runs walk-forward training and evaluation across all configured folds.

    For each fold:
      1. Slice the DataLoader to the fold's date ranges.
      2. Build train/val/test :class:`~alpha0.env.market_env.MarketEnv` instances.
      3. Train a fresh :class:`~alpha0.training.sac.SACTrainer`.
      4. Evaluate on the test split and record metrics.

    Parameters
    ----------
    cfg:
        Full config dict (``config/default.yaml``).
    data_loader:
        Fully-built :class:`~alpha0.data.loader.DataLoader`.
    device:
        PyTorch device string.
    n_steps:
        Total environment steps per fold.  Overrides config if provided.
    artifacts_dir:
        Root directory for saving per-fold checkpoints and logs.
    """

    def __init__(
        self,
        cfg: dict,
        data_loader,
        device: str = "cpu",
        n_steps: int | None = None,
        artifacts_dir: str | Path = "artifacts/walk_forward",
    ) -> None:
        self._cfg      = cfg
        self._loader   = data_loader
        self._device   = device
        self._n_steps  = n_steps or cfg.get("training", {}).get("n_steps", 200_000)
        self._art_dir  = Path(artifacts_dir)
        self._art_dir.mkdir(parents=True, exist_ok=True)

        self._folds: list[dict] = cfg.get("walk_forward", {}).get("folds", [])
        self._train_start: str = cfg.get("walk_forward", {}).get(
            "train_start", cfg["data"]["start_date"]
        )

        if not self._folds:
            raise ValueError(
                "No walk-forward folds defined in config. "
                "Add a 'walk_forward.folds' section to default.yaml."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, folds: list[int] | None = None) -> WalkForwardResult:
        """Execute walk-forward training across all (or selected) folds.

        Parameters
        ----------
        folds:
            Indices of folds to run.  If None, run all folds.

        Returns
        -------
        WalkForwardResult
            Aggregated results across all executed folds.
        """
        fold_indices = folds if folds is not None else list(range(len(self._folds)))
        results: list[FoldResult] = []

        for fi in fold_indices:
            if fi >= len(self._folds):
                logger.warning("Fold index %d out of range (%d folds total).", fi, len(self._folds))
                continue

            fold_cfg = self._folds[fi]
            logger.info(
                "=" * 70 + "\nFold %d/%d: train_end=%s | val_end=%s | test_end=%s\n" + "=" * 70,
                fi + 1, len(fold_indices),
                fold_cfg["train_end"], fold_cfg["val_end"], fold_cfg["test_end"],
            )

            result = self._run_fold(fi, fold_cfg)
            results.append(result)

            logger.info(
                "Fold %d complete. Val Sharpe=%.3f | Test metrics: %s",
                fi + 1, result.val_sharpe,
                {k: f"{v:.3f}" for k, v in result.test_metrics.items()},
            )

        wf_result = WalkForwardResult(
            folds     = results,
            aggregate = self._aggregate_metrics(results),
        )
        wf_result.print_summary()
        return wf_result

    # ------------------------------------------------------------------
    # Internal per-fold logic
    # ------------------------------------------------------------------

    def _run_fold(self, fold_idx: int, fold_cfg: dict) -> FoldResult:
        """Train, early-stop on val, then evaluate on test for one fold."""
        from alpha0.env.market_env import MarketEnv
        from alpha0.training.sac import SACTrainer

        train_end = fold_cfg["train_end"]
        val_end   = fold_cfg["val_end"]
        test_end  = fold_cfg["test_end"]

        # Slice loader split indices to fold dates
        train_loader = self._make_fold_loader(self._train_start, train_end)
        val_loader   = self._make_fold_loader(train_end,         val_end)
        test_loader  = self._make_fold_loader(val_end,           test_end)

        # Build environments
        train_env = MarketEnv(self._cfg, train_loader, split="train")
        val_env   = MarketEnv(self._cfg, val_loader,   split="val")
        test_env  = MarketEnv(self._cfg, test_loader,  split="test")

        # Per-fold checkpoint directory
        fold_dir = self._art_dir / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Train
        trainer = SACTrainer(self._cfg, device=self._device)
        train_history = trainer.train(
            env                 = train_env,
            n_steps             = self._n_steps,
            val_env             = val_env,
            eval_every          = max(1_000, self._n_steps // 50),
            save_dir            = fold_dir,
            early_stop_patience = 10,
        )

        # Best val Sharpe
        best_val_sharpe = float(np.max(
            [trainer.evaluate(val_env, n_episodes=10)]
        ))

        # Load best checkpoint if available
        best_ckpt = fold_dir / "best.pt"
        if best_ckpt.exists():
            trainer.load(best_ckpt)

        # Test evaluation
        test_metrics = self._evaluate_test(trainer, test_env)

        return FoldResult(
            fold_idx             = fold_idx,
            train_end            = train_end,
            val_end              = val_end,
            test_end             = test_end,
            val_sharpe           = best_val_sharpe,
            test_metrics         = test_metrics,
            train_metrics_history= train_history,
            checkpoint_path      = str(best_ckpt) if best_ckpt.exists() else str(fold_dir / "final.pt"),
        )

    def _make_fold_loader(self, start_date: str, end_date: str):
        """Return a DataLoader view sliced to [start_date, end_date].

        Creates a lightweight proxy around the shared loader that overrides
        ``_split_indices`` so the environment samples only within the fold
        window.  The underlying feature tensor is shared (not copied).
        """
        return _FoldLoaderProxy(self._loader, start_date, end_date)

    def _evaluate_test(self, trainer, test_env) -> dict[str, float]:
        """Evaluate the trained policy on the test environment.

        Returns a metrics dict with Sharpe, return, vol, and drawdown.
        """
        from alpha0.eval.metrics import compute_metrics
        import pandas as pd

        n_episodes = 5
        all_returns: list[list[float]] = []

        for _ in range(n_episodes):
            obs_dict, _ = test_env.reset()
            ep_returns: list[float] = []
            done = False
            with torch.no_grad():
                while not done:
                    action = trainer._select_action(
                        obs_dict["obs"], obs_dict["portfolio"], deterministic=True
                    )
                    obs_dict, _, term, trunc, info = test_env.step(action)
                    ep_returns.append(info.get("portfolio_return", 0.0))
                    done = term or trunc
            all_returns.append(ep_returns)

        # Flatten all returns across episodes into one series
        flat = [r for ep in all_returns for r in ep]
        if not flat:
            return {}

        ret_series = pd.Series(flat)
        rf = self._cfg["eval"].get("risk_free_rate", 0.04)
        return compute_metrics(ret_series, rf_rate=rf)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_metrics(results: list[FoldResult]) -> dict[str, dict[str, float]]:
        """Compute mean and std of test metrics across folds."""
        if not results:
            return {}

        all_keys = set()
        for r in results:
            all_keys.update(r.test_metrics.keys())

        agg: dict[str, dict[str, float]] = {}
        for key in sorted(all_keys):
            vals = [r.test_metrics[key] for r in results if key in r.test_metrics]
            if vals:
                agg[key] = {
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals)),
                    "min":  float(np.min(vals)),
                    "max":  float(np.max(vals)),
                }
        return agg


# ─────────────────────────────────────────────────────────────
# Fold Loader Proxy
# ─────────────────────────────────────────────────────────────

class _FoldLoaderProxy:
    """Lightweight proxy that wraps a DataLoader and exposes a date-sliced view.

    This avoids copying the potentially large feature tensor; only the
    ``_split_indices`` attribute is overridden.

    Parameters
    ----------
    loader:
        The underlying :class:`~alpha0.data.loader.DataLoader`.
    start_date:
        First date of this fold (inclusive), as ``"YYYY-MM-DD"`` string.
    end_date:
        Last date of this fold (inclusive), as ``"YYYY-MM-DD"`` string.
    """

    def __init__(self, loader, start_date: str, end_date: str) -> None:
        self._loader    = loader
        self._start     = pd.Timestamp(start_date)
        self._end       = pd.Timestamp(end_date)

        # Compute integer indices into loader.dates
        dates = pd.DatetimeIndex(loader.dates)
        start_idx = int(np.searchsorted(dates, self._start, side="left"))
        end_idx   = int(np.searchsorted(dates, self._end,   side="right"))

        # Expose a single "train" split mapping to this fold's range.
        # MarketEnv looks up split_indices[split] to find valid start positions.
        self._split_indices: dict[str, tuple[int, int]] = {
            "train": (start_idx, end_idx),
            "val":   (start_idx, end_idx),
            "test":  (start_idx, end_idx),
        }

    # Delegate everything else to the wrapped loader
    def __getattr__(self, name: str):
        return getattr(self._loader, name)
