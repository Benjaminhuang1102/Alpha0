"""Performance metrics for Alpha0 strategy evaluation.

All functions are pure (no side effects) and stateless.  They operate on
``pd.Series`` of daily returns with a DatetimeIndex.

All ratio metrics are annualised to a 252-trading-day year.  Risk-free
rate is an annualised value; functions convert it to a daily rate
internally using the exact formula:

    rf_daily = (1 + rf_annual) ** (1 / trading_days) - 1
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────
# Return and volatility
# ─────────────────────────────────────────────────────────────

def annualised_return(
    returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """Compound annualised return."""
    n = len(returns)
    if n == 0:
        return 0.0
    total = float((1.0 + returns).prod())
    if total <= 0:
        return -1.0
    return float(total ** (trading_days / n) - 1.0)


def annualised_vol(
    returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """Annualised standard deviation of daily returns."""
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1)) * math.sqrt(trading_days)


# ─────────────────────────────────────────────────────────────
# Risk-adjusted ratios
# ─────────────────────────────────────────────────────────────

def sharpe_ratio(
    returns: pd.Series,
    rf_rate: float = 0.04,
    trading_days: int = 252,
) -> float:
    """Annualised Sharpe ratio."""
    rf_daily = (1.0 + rf_rate) ** (1.0 / trading_days) - 1.0
    excess = returns - rf_daily
    vol = float(excess.std(ddof=1))
    if vol < 1e-10:
        return 0.0
    return float(excess.mean() / vol * math.sqrt(trading_days))


def sortino_ratio(
    returns: pd.Series,
    rf_rate: float = 0.04,
    trading_days: int = 252,
    mar: float = 0.0,
) -> float:
    """Annualised Sortino ratio (uses downside deviation)."""
    rf_daily = (1.0 + rf_rate) ** (1.0 / trading_days) - 1.0
    downside = returns[returns < mar] - mar
    if len(downside) < 2:
        return 0.0
    downside_dev = float(np.sqrt((downside ** 2).mean())) * math.sqrt(trading_days)
    if downside_dev < 1e-10:
        return 0.0
    ann_ret = annualised_return(returns, trading_days) - rf_rate
    return ann_ret / downside_dev


def calmar_ratio(
    returns: pd.Series,
    rf_rate: float = 0.04,
    trading_days: int = 252,
) -> float:
    """Calmar ratio: annualised return / max drawdown."""
    dd = max_drawdown(returns)
    if dd < 1e-10:
        return 0.0
    return annualised_return(returns, trading_days) / dd


def omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0,
    trading_days: int = 252,
) -> float:
    """Omega ratio: probability-weighted gain vs loss above a threshold.

    Omega = sum(max(r - T, 0)) / sum(max(T - r, 0))

    Returns 0 if there are no returns below the threshold.
    """
    # Convert annualised threshold to daily
    daily_threshold = (1.0 + threshold) ** (1.0 / trading_days) - 1.0
    gains  = (returns - daily_threshold).clip(lower=0.0).sum()
    losses = (daily_threshold - returns).clip(lower=0.0).sum()
    if losses < 1e-10:
        return np.inf if gains > 0 else 1.0
    return float(gains / losses)


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """Information ratio: annualised active return / tracking error."""
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return 0.0
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = float(active.std(ddof=1))
    if te < 1e-10:
        return 0.0
    return float(active.mean() / te * math.sqrt(trading_days))


# ─────────────────────────────────────────────────────────────
# Drawdown metrics
# ─────────────────────────────────────────────────────────────

def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (positive fraction)."""
    if len(returns) == 0:
        return 0.0
    cum = (1.0 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    return float(-drawdown.min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Return the full drawdown (underwater) series."""
    cum = (1.0 + returns).cumprod()
    running_max = cum.cummax()
    return (cum - running_max) / running_max


def recovery_factor(returns: pd.Series, trading_days: int = 252) -> float:
    """Total return divided by max drawdown.

    Measures how much return was earned per unit of worst-case pain.
    """
    dd = max_drawdown(returns)
    if dd < 1e-10:
        return 0.0
    total = float((1.0 + returns).prod() - 1.0)
    return total / dd


def ulcer_index(returns: pd.Series) -> float:
    """Ulcer Index: root mean square of drawdowns.

    Unlike max_drawdown, this captures the *duration* and *depth* of
    underwater periods.  Lower is better.
    """
    if len(returns) < 2:
        return 0.0
    cum = (1.0 + returns).cumprod()
    running_max = cum.cummax()
    dd_pct = 100.0 * (cum - running_max) / running_max  # in percent
    return float(np.sqrt((dd_pct ** 2).mean()))


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest period (in days) spent below a previous peak."""
    if len(returns) < 2:
        return 0
    cum = (1.0 + returns).cumprod()
    running_max = cum.cummax()
    underwater = (cum < running_max)

    max_dur  = 0
    curr_dur = 0
    for uw in underwater:
        if uw:
            curr_dur += 1
            max_dur = max(max_dur, curr_dur)
        else:
            curr_dur = 0
    return max_dur


# ─────────────────────────────────────────────────────────────
# Risk / tail metrics
# ─────────────────────────────────────────────────────────────

def value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Historical (non-parametric) Value at Risk.

    Parameters
    ----------
    returns:
        Daily returns.
    confidence:
        Confidence level (e.g. 0.95 = 95% VaR).

    Returns
    -------
    float
        VaR as a positive fraction (e.g. 0.02 = 2% daily loss at 95% confidence).
    """
    if len(returns) < 5:
        return 0.0
    return float(-np.quantile(returns, 1.0 - confidence))


def conditional_var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Conditional Value at Risk (CVaR / Expected Shortfall).

    Average loss in the tail beyond the VaR threshold.

    Returns
    -------
    float
        CVaR as a positive fraction.
    """
    if len(returns) < 5:
        return 0.0
    cutoff = np.quantile(returns, 1.0 - confidence)
    tail   = returns[returns <= cutoff]
    if len(tail) == 0:
        return 0.0
    return float(-tail.mean())


def return_skewness(returns: pd.Series) -> float:
    """Return distribution skewness (positive = right-skewed / fat right tail)."""
    if len(returns) < 4:
        return 0.0
    return float(returns.skew())


def return_kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis (0 = normal; > 0 = fat tails)."""
    if len(returns) < 4:
        return 0.0
    return float(returns.kurtosis())


# ─────────────────────────────────────────────────────────────
# Trade / execution quality
# ─────────────────────────────────────────────────────────────

def hit_rate(returns: pd.Series) -> float:
    """Fraction of days with a positive return."""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    """Gross profit divided by gross loss.

    > 1 means the strategy earns more on winning days than it loses on
    losing days, weighted by frequency.
    """
    gains  = returns[returns > 0].sum()
    losses = (-returns[returns < 0]).sum()
    if losses < 1e-10:
        return np.inf if gains > 0 else 1.0
    return float(gains / losses)


def avg_win_loss_ratio(returns: pd.Series) -> float:
    """Average winning daily return divided by average absolute losing return."""
    wins   = returns[returns > 0]
    losses = returns[returns < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    return float(wins.mean() / abs(losses.mean()))


def tail_ratio(returns: pd.Series, quantile: float = 0.05) -> float:
    """Ratio of 95th-percentile gain to absolute 5th-percentile loss."""
    upper = float(returns.quantile(1.0 - quantile))
    lower = float(abs(returns.quantile(quantile)))
    if lower < 1e-10:
        return 0.0
    return upper / lower


# ─────────────────────────────────────────────────────────────
# Benchmark-relative
# ─────────────────────────────────────────────────────────────

def beta_alpha(
    returns: pd.Series,
    benchmark: pd.Series,
    rf_rate: float = 0.04,
    trading_days: int = 252,
) -> tuple[float, float]:
    """OLS beta and Jensen's annualised alpha vs a benchmark."""
    rf_daily = (1.0 + rf_rate) ** (1.0 / trading_days) - 1.0
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    aligned.columns = ["strategy", "benchmark"]

    if len(aligned) < 5:
        return 0.0, 0.0

    y = (aligned["strategy"] - rf_daily).values
    x = (aligned["benchmark"] - rf_daily).values
    slope, intercept, _, _, _ = stats.linregress(x, y)
    return float(slope), float(intercept) * trading_days


def portfolio_turnover(weights: pd.DataFrame) -> float:
    """Mean daily one-way portfolio turnover."""
    if len(weights) < 2:
        return 0.0
    daily_turn = weights.diff().abs().sum(axis=1).iloc[1:] / 2.0
    return float(daily_turn.mean())


# ─────────────────────────────────────────────────────────────
# Aggregate
# ─────────────────────────────────────────────────────────────

def compute_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    weights: pd.DataFrame | None = None,
    rf_rate: float = 0.04,
    trading_days: int = 252,
    var_confidence: float = 0.95,
) -> dict[str, float]:
    """Compute the full metrics suite for a return series.

    Parameters
    ----------
    returns:
        Daily arithmetic returns of the strategy.
    benchmark_returns:
        Optional benchmark for beta, alpha, information ratio.
    weights:
        Optional ``(T, N)`` weight matrix for turnover.
    rf_rate:
        Annualised risk-free rate.
    trading_days:
        Trading days per year.
    var_confidence:
        Confidence level for VaR / CVaR.

    Returns
    -------
    dict[str, float]
        Flat dictionary with all computed metrics.
    """
    result: dict[str, float] = {
        # Return
        "total_return":           float((1.0 + returns).prod() - 1.0),
        "annualised_return":      annualised_return(returns, trading_days),
        "annualised_vol":         annualised_vol(returns, trading_days),

        # Risk-adjusted
        "sharpe_ratio":           sharpe_ratio(returns, rf_rate, trading_days),
        "sortino_ratio":          sortino_ratio(returns, rf_rate, trading_days),
        "calmar_ratio":           calmar_ratio(returns, rf_rate, trading_days),
        "omega_ratio":            omega_ratio(returns, 0.0, trading_days),

        # Drawdown
        "max_drawdown":           max_drawdown(returns),
        "ulcer_index":            ulcer_index(returns),
        "recovery_factor":        recovery_factor(returns, trading_days),
        "max_drawdown_days":      float(max_drawdown_duration(returns)),

        # Tail / risk
        f"var_{int(var_confidence*100)}":   value_at_risk(returns, var_confidence),
        f"cvar_{int(var_confidence*100)}":  conditional_var(returns, var_confidence),
        "skewness":               return_skewness(returns),
        "kurtosis":               return_kurtosis(returns),
        "tail_ratio":             tail_ratio(returns),

        # Trade quality
        "hit_rate":               hit_rate(returns),
        "profit_factor":          profit_factor(returns),
        "avg_win_loss_ratio":     avg_win_loss_ratio(returns),
    }

    if benchmark_returns is not None:
        b, a = beta_alpha(returns, benchmark_returns, rf_rate, trading_days)
        ir   = information_ratio(returns, benchmark_returns, trading_days)
        result["beta"]             = b
        result["alpha"]            = a
        result["information_ratio"]= ir

    if weights is not None:
        eq_cols = [c for c in weights.columns if c != "cash"]
        result["turnover"] = portfolio_turnover(weights[eq_cols])

    return result
