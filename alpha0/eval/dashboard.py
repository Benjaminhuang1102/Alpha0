"""Visualisation dashboard for Alpha0 backtest results.

Generates static matplotlib/seaborn plots and a metrics comparison table,
saved to the configured artifacts directory.

Phase 3/4 additions:
- Monthly returns heatmap (calendar view)
- Return distribution with normal overlay
- Rolling VaR / CVaR chart
- Regime overlay on equity curve
- Execution cost analysis
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from alpha0.eval.backtest import BacktestResult
from alpha0.eval.metrics import (
    sharpe_ratio, max_drawdown, drawdown_series,
    value_at_risk, conditional_var,
)

logger = logging.getLogger(__name__)

sns.set_theme(style="darkgrid", palette="muted")

_REGIME_COLORS = {
    "bull_low_vol":  "#2ecc71",
    "bull_high_vol": "#f39c12",
    "bear_low_vol":  "#e67e22",
    "bear_high_vol": "#e74c3c",
}


class Dashboard:
    """Generates a static PNG performance report.

    Parameters
    ----------
    cfg:
        Full config dict.
    """

    def __init__(self, cfg: dict) -> None:
        self._reports_dir  = Path(cfg["artifacts"]["reports_dir"])
        self._rolling_window: int = cfg["eval"].get("rolling_window", 63)
        self._trading_days: int   = cfg["eval"]["trading_days_per_year"]
        self._rf_rate: float      = cfg["eval"]["risk_free_rate"]
        self._var_conf: float     = cfg["eval"].get("var_confidence", 0.95)

    # ------------------------------------------------------------------
    # Individual plots
    # ------------------------------------------------------------------

    def plot_cumulative_returns(
        self,
        results: dict[str, BacktestResult],
        save_path: str | Path,
        log_scale: bool = True,
        regime_series: pd.Series | None = None,
    ) -> None:
        """Equity curves, optionally with regime background shading."""
        fig, ax = plt.subplots(figsize=(13, 6))

        # Regime background shading
        if regime_series is not None:
            _shade_regimes(ax, regime_series)

        for name, result in results.items():
            cum = (1.0 + result.daily_returns).cumprod()
            ax.plot(cum.index, cum.values, label=name, linewidth=1.5, zorder=3)

        ax.set_title("Cumulative Returns", fontsize=14)
        ax.set_ylabel("Portfolio Value (normalised to 1.0)")
        ax.legend(fontsize=9, loc="upper left")
        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}x"))
        ax.grid(True, alpha=0.3, zorder=1)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Saved cumulative returns chart to %s", save_path)

    def plot_drawdown(
        self,
        result: BacktestResult,
        save_path: str | Path,
        label: str = "Strategy",
    ) -> None:
        """Underwater (drawdown) chart with duration annotation."""
        dd = drawdown_series(result.daily_returns)

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.fill_between(dd.index, dd.values, 0, alpha=0.7, color="firebrick", label=label)

        # Mark max drawdown point
        min_idx = dd.idxmin()
        ax.annotate(
            f"Max DD: {dd.min():.1%}",
            xy=(min_idx, float(dd.min())),
            xytext=(min_idx, float(dd.min()) * 0.5),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=8,
        )

        ax.set_title("Drawdown", fontsize=14)
        ax.set_ylabel("Drawdown (%)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Saved drawdown chart to %s", save_path)

    def plot_weight_heatmap(
        self,
        result: BacktestResult,
        save_path: str | Path,
        top_n: int = 20,
    ) -> None:
        """Heatmap of top-N asset weights over time."""
        weights = result.weights
        eq_cols = [c for c in weights.columns if c != "cash"]
        w = weights[eq_cols]
        top_assets = w.mean().nlargest(top_n).index.tolist()
        w_top = w[top_assets]

        fig, ax = plt.subplots(figsize=(14, max(4, top_n // 2)))
        step   = max(1, len(w_top) // 100)
        w_plot = w_top.iloc[::step].T

        sns.heatmap(
            w_plot, ax=ax, cmap="YlOrRd", vmin=0,
            cbar_kws={"label": "Weight"}, linewidths=0.0, xticklabels=False,
        )
        ax.set_title(f"Portfolio Weights — Top {top_n} Assets", fontsize=14)
        ax.set_ylabel("Asset")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Saved weight heatmap to %s", save_path)

    def plot_rolling_metrics(
        self,
        result: BacktestResult,
        save_path: str | Path,
        window: int | None = None,
    ) -> None:
        """Rolling Sharpe ratio and annualised volatility."""
        w = window or self._rolling_window
        returns = result.daily_returns
        rf_daily = (1.0 + self._rf_rate) ** (1.0 / self._trading_days) - 1.0
        excess = returns - rf_daily

        rolling_sharpe = (
            excess.rolling(w).mean()
            / excess.rolling(w).std(ddof=1).clip(lower=1e-8)
            * np.sqrt(self._trading_days)
        )
        rolling_vol = returns.rolling(w).std(ddof=1) * np.sqrt(self._trading_days)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        ax1.plot(rolling_sharpe.index, rolling_sharpe.values, color="steelblue", lw=1)
        ax1.axhline(0, color="black", lw=0.8, ls="--")
        ax1.axhline(1, color="green",  lw=0.8, ls=":")
        ax1.set_ylabel(f"Rolling {w}d Sharpe")
        ax1.set_title("Rolling Performance Metrics", fontsize=14)

        ax2.plot(rolling_vol.index, rolling_vol.values, color="orange", lw=1)
        ax2.set_ylabel(f"Rolling {w}d Ann. Vol")
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

        for ax in (ax1, ax2):
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Saved rolling metrics chart to %s", save_path)

    def plot_monthly_returns_heatmap(
        self,
        result: BacktestResult,
        save_path: str | Path,
    ) -> None:
        """Calendar heatmap of monthly returns (year × month grid)."""
        returns = result.daily_returns
        if len(returns) < 20:
            return

        monthly = (1.0 + returns).resample("ME").prod() - 1.0
        monthly_df = monthly.to_frame("ret")
        monthly_df["year"]  = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month

        pivot = monthly_df.pivot(index="year", columns="month", values="ret")
        pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]

        # Add annual total
        annual = pivot.apply(lambda row: float((1.0 + row.dropna()).prod() - 1.0), axis=1)
        pivot["Annual"] = annual

        fig, ax = plt.subplots(figsize=(16, max(4, len(pivot) * 0.5 + 1)))

        # Build annotation matrix (formatted percentages)
        annot = pivot.map(lambda x: f"{x:.1%}" if pd.notna(x) else "")

        sns.heatmap(
            pivot,
            annot=annot,
            fmt="",
            cmap="RdYlGn",
            center=0,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "Monthly Return"},
            vmin=-0.15, vmax=0.15,
        )
        ax.set_title("Monthly Returns Calendar", fontsize=14)
        ax.set_ylabel("Year")
        ax.set_xlabel("")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Saved monthly returns heatmap to %s", save_path)

    def plot_return_distribution(
        self,
        result: BacktestResult,
        save_path: str | Path,
    ) -> None:
        """Return distribution histogram with normal and VaR overlays."""
        returns = result.daily_returns.dropna()
        if len(returns) < 20:
            return

        fig, ax = plt.subplots(figsize=(10, 5))

        # Histogram
        ax.hist(returns, bins=60, density=True, alpha=0.6, color="steelblue", label="Daily returns")

        # Normal distribution overlay
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 200)
        from scipy.stats import norm
        ax.plot(x, norm.pdf(x, mu, sigma), "r-", lw=2, label="Normal fit")

        # VaR and CVaR lines
        var_95  = value_at_risk(returns, 0.95)
        cvar_95 = conditional_var(returns, 0.95)
        ax.axvline(-var_95,  color="orange", lw=2, ls="--", label=f"95% VaR  = {var_95:.2%}")
        ax.axvline(-cvar_95, color="red",    lw=2, ls=":",  label=f"95% CVaR = {cvar_95:.2%}")

        ax.set_title("Return Distribution", fontsize=14)
        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Saved return distribution to %s", save_path)

    def plot_rolling_var(
        self,
        result: BacktestResult,
        save_path: str | Path,
        window: int | None = None,
        confidence: float | None = None,
    ) -> None:
        """Rolling VaR and CVaR over time."""
        w    = window or self._rolling_window
        conf = confidence or self._var_conf
        returns = result.daily_returns.dropna()

        if len(returns) < w + 5:
            return

        rolling_var  = returns.rolling(w).quantile(1.0 - conf).abs()
        rolling_cvar = returns.rolling(w).apply(
            lambda r: float(-r[r <= np.quantile(r, 1.0 - conf)].mean())
            if len(r[r <= np.quantile(r, 1.0 - conf)]) > 0 else 0.0,
            raw=True,
        )

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.fill_between(rolling_var.index,  rolling_var.values,  alpha=0.4, color="orange", label=f"Rolling {w}d VaR ({int(conf*100)}%)")
        ax.fill_between(rolling_cvar.index, rolling_cvar.values, alpha=0.3, color="red",    label=f"Rolling {w}d CVaR ({int(conf*100)}%)")
        ax.set_title(f"Rolling {w}-Day VaR / CVaR at {int(conf*100)}% Confidence", fontsize=14)
        ax.set_ylabel("Daily Loss (positive = bad)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Saved rolling VaR chart to %s", save_path)

    def plot_execution_costs(
        self,
        result: BacktestResult,
        save_path: str | Path,
    ) -> None:
        """Cumulative execution cost drag over time (if metrics available)."""
        total_cost_bps = result.metrics.get("execution_cost_bps", None)
        turnover       = result.metrics.get("turnover", None)

        if total_cost_bps is None and turnover is None:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax1 = axes[0]
        cost_items = {k: v for k, v in result.metrics.items() if k.startswith("execution_")}
        if cost_items:
            labels = [k.replace("execution_", "").replace("_", " ").title() for k in cost_items]
            values = list(cost_items.values())
            ax1.barh(labels, values, color="firebrick", alpha=0.8)
            ax1.set_title("Execution Cost Summary", fontsize=12)
            ax1.set_xlabel("Value")
            ax1.grid(True, alpha=0.3, axis="x")

        ax2 = axes[1]
        if turnover is not None:
            years_in_sample = len(result.daily_returns) / 252
            ann_tc_drag_bps = turnover * 252 * 10.0  # 10bps round-trip
            ax2.bar(["Annual TC Drag"], [ann_tc_drag_bps], color="darkorange", alpha=0.8)
            ax2.set_title(f"Estimated Annual TC Drag\n(turnover={turnover:.1%}/day × 10bps)", fontsize=11)
            ax2.set_ylabel("Basis Points")
            ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Saved execution cost chart to %s", save_path)

    def plot_walk_forward_results(
        self,
        summary_df: pd.DataFrame,
        save_path: str | Path,
    ) -> None:
        """Bar chart of per-fold Sharpe and drawdown for walk-forward results."""
        if summary_df.empty or "sharpe_ratio" not in summary_df.columns:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        folds = [f"Fold {i}" for i in summary_df.get("fold", range(len(summary_df)))]

        ax1.bar(folds, summary_df["sharpe_ratio"], color="steelblue", alpha=0.8)
        ax1.axhline(0, color="black", lw=0.8)
        ax1.axhline(1, color="green", lw=0.8, ls="--", label="Sharpe=1")
        ax1.set_ylabel("Out-of-Sample Sharpe")
        ax1.set_title("Walk-Forward Out-of-Sample Performance", fontsize=14)
        ax1.legend(fontsize=9)

        if "max_drawdown" in summary_df.columns:
            ax2.bar(folds, summary_df["max_drawdown"] * 100, color="firebrick", alpha=0.8)
            ax2.set_ylabel("Max Drawdown (%)")
            ax2.invert_yaxis()

        for ax in (ax1, ax2):
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Saved walk-forward chart to %s", save_path)

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        results: dict[str, BacktestResult],
        output_dir: str | Path | None = None,
        regime_series: pd.Series | None = None,
        walk_forward_df: pd.DataFrame | None = None,
    ) -> Path:
        """Generate all plots and a metrics comparison CSV.

        Parameters
        ----------
        results:
            ``{strategy_name: BacktestResult}``
        output_dir:
            Override default report dir.  If None, uses
            ``artifacts/reports/{timestamp}``.
        regime_series:
            Optional pd.Series of MarketRegime values for equity curve shading.
        walk_forward_df:
            Optional DataFrame from WalkForwardResult.summary_table().

        Returns
        -------
        Path
            Report directory.
        """
        if output_dir is None:
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = self._reports_dir / ts
        else:
            out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        primary_name = next(iter(results))
        primary = results[primary_name]

        # 1. Cumulative returns (with optional regime shading)
        self.plot_cumulative_returns(
            results, out / "cumulative_returns.png",
            regime_series=regime_series,
        )

        # 2. Drawdown
        self.plot_drawdown(primary, out / "drawdown.png", label=primary_name)

        # 3. Rolling Sharpe + vol
        if len(primary.daily_returns) >= self._rolling_window:
            self.plot_rolling_metrics(primary, out / "rolling_metrics.png")

        # 4. Weight heatmap
        if primary.weights is not None and not primary.weights.empty:
            self.plot_weight_heatmap(primary, out / "weights_heatmap.png")

        # 5. Monthly returns calendar
        self.plot_monthly_returns_heatmap(primary, out / "monthly_returns.png")

        # 6. Return distribution
        self.plot_return_distribution(primary, out / "return_distribution.png")

        # 7. Rolling VaR
        if len(primary.daily_returns) >= self._rolling_window + 5:
            self.plot_rolling_var(primary, out / "rolling_var.png")

        # 8. Execution costs
        self.plot_execution_costs(primary, out / "execution_costs.png")

        # 9. Walk-forward results
        if walk_forward_df is not None and not walk_forward_df.empty:
            self.plot_walk_forward_results(walk_forward_df, out / "walk_forward.png")

        # 10. Metrics comparison CSV
        rows = []
        for name, result in results.items():
            row = {"strategy": name}
            row.update(result.metrics)
            rows.append(row)
        metrics_df = pd.DataFrame(rows).set_index("strategy")
        metrics_df.to_csv(out / "metrics_comparison.csv")
        logger.info("Report generated at %s", out)
        return out


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _shade_regimes(ax: plt.Axes, regime_series: pd.Series) -> None:
    """Add coloured background shading for market regimes."""
    if regime_series is None or regime_series.empty:
        return

    dates = regime_series.index
    regimes = regime_series.values

    start = dates[0]
    prev  = str(regimes[0].value) if hasattr(regimes[0], "value") else str(regimes[0])

    for i in range(1, len(dates)):
        curr = str(regimes[i].value) if hasattr(regimes[i], "value") else str(regimes[i])
        if curr != prev:
            color = _REGIME_COLORS.get(prev, "#cccccc")
            ax.axvspan(start, dates[i], alpha=0.08, color=color, zorder=0)
            start = dates[i]
            prev  = curr

    # Last segment
    color = _REGIME_COLORS.get(prev, "#cccccc")
    ax.axvspan(start, dates[-1], alpha=0.08, color=color, zorder=0)
