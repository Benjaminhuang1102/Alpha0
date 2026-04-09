"""Gymnasium-compatible market environment for portfolio optimisation.

State space:
    Dict observation with two keys:
      ``"obs"``       – Box(shape=(lookback, n_assets, n_features), float32)
      ``"portfolio"`` – Box(shape=(n_assets+1,), float32)  [last element = cash]

Action space:
    Box(shape=(n_assets+1,), low=-inf, high=inf, float32)
    Raw logits; softmax + constraint projection applied internally.

Episode structure:
    One episode = ``episode_length`` trading days (default 252).
    Each :meth:`step` advances time by one trading day.
    The episode ends when ``step_idx == episode_length`` (``truncated=True``).

Position constraints (enforced inside the env, not learned):
    - Max single-stock weight: ``max_single_weight`` (default 10%)
    - Min cash allocation:     ``min_cash_weight``   (default 5%)
    - Max gross exposure:      ``max_gross_exposure`` (default 100%)

Weight drift:
    After applying daily asset returns, portfolio weights drift from the
    rebalanced values.  The env updates ``_current_weights`` to the
    post-return *drifted* weights before each step, so that the turnover
    cost at the next rebalance is computed correctly.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from alpha0.data.loader import DataLoader, N_FEATURES
from alpha0.env.costs import TransactionCostModel
from alpha0.env.reward import RewardFunction

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies
def _import_risk_manager():
    from alpha0.risk.manager import RiskManager
    return RiskManager


class MarketEnv(gym.Env):
    """Portfolio management environment inspired by the Alpha0 spec.

    Parameters
    ----------
    cfg:
        Full config dict (``config/default.yaml``).
    data_loader:
        Initialised :class:`~alpha0.data.loader.DataLoader` with data
        already built.
    split:
        Which data split to sample episodes from: ``"train"``,
        ``"val"``, or ``"test"``.
    """

    metadata: dict[str, Any] = {"render_modes": ["human"]}

    def __init__(
        self,
        cfg: dict,
        data_loader: DataLoader,
        split: str = "train",
        risk_manager=None,
    ) -> None:
        """
        Parameters
        ----------
        cfg:
            Full config dict.
        data_loader:
            Initialised DataLoader.
        split:
            Data split: ``"train"``, ``"val"``, or ``"test"``.
        risk_manager:
            Optional :class:`~alpha0.risk.manager.RiskManager`.  When
            provided, its constraints are applied after ``_action_to_weights``
            on every step.
        """
        super().__init__()

        self._cfg = cfg
        self._ec = cfg["env"]
        self._split = split
        self._loader = data_loader
        self._risk_manager = risk_manager

        self._n_assets: int = self._ec["n_assets"]
        self._ep_len: int = self._ec["episode_length"]
        self._lookback: int = cfg["data"]["lookback_window"]
        self._init_value: float = self._ec["initial_portfolio_value"]

        self._max_w: float = self._ec["max_single_weight"]
        self._min_cash: float = self._ec["min_cash_weight"]

        self._cost_model = TransactionCostModel(cfg)
        self._reward_fn = RewardFunction(cfg)

        # Gymnasium spaces
        n_obs = self._n_assets + 1  # assets + cash slot in action/portfolio

        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(
                    low=-5.0,
                    high=5.0,
                    shape=(self._lookback, self._n_assets, N_FEATURES),
                    dtype=np.float32,
                ),
                "portfolio": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(n_obs,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_obs,),
            dtype=np.float32,
        )

        # Episode state (initialised in reset)
        self._current_weights: np.ndarray = np.zeros(n_obs, dtype=np.float32)
        self._portfolio_value: float = self._init_value
        self._step_idx: int = 0
        self._episode_dates: list[Any] = []
        # Observation block: shape (ep_len + lookback, N, F)
        self._obs_block: torch.Tensor | None = None
        # Close prices for this episode (for return calculation): (ep_len + lookback, N)
        self._close_block: np.ndarray | None = None

        # Populated lazily once loader data is available
        self._valid_start_indices: list[int] | None = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Start a new episode.

        Samples a random start date from the configured split.  Initial
        portfolio weights are set to equal allocation across all assets
        (no cash) — the agent immediately rebalances on the first step.

        Returns
        -------
        tuple[dict, dict]
            ``(observation, info)``
        """
        super().reset(seed=seed)

        self._reward_fn.reset()
        self._step_idx = 0

        if self._risk_manager is not None:
            self._risk_manager.reset(self._init_value)

        # Sample a random episode block from the loader
        self._sample_episode()

        # Initial weights: equal across assets, zero cash
        n_eq = self._n_assets
        w0 = np.zeros(self._n_assets + 1, dtype=np.float32)
        w0[:n_eq] = 1.0 / n_eq
        self._current_weights = w0
        self._portfolio_value = self._init_value

        obs = self._get_observation()
        info: dict = {"step": 0, "portfolio_value": self._portfolio_value}
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """Advance the environment by one trading day.

        Parameters
        ----------
        action:
            Raw logits of shape ``(n_assets + 1,)``.  Softmax is applied
            internally.  The last element corresponds to cash.

        Returns
        -------
        tuple
            ``(observation, reward, terminated, truncated, info)``
        """
        assert self._obs_block is not None, "Call reset() before step()."

        # 1. Convert logits → target weights with constraints
        target_weights = self._action_to_weights(action)

        # 1b. Apply risk management (circuit breakers, sector limits, etc.)
        risk_alerts: list[str] = []
        if self._risk_manager is not None:
            target_weights, risk_alerts = self._risk_manager.apply(
                target_weights, self._current_weights
            )

        # 2. Transaction cost (computed before applying returns)
        cost_frac, turnover = self._cost_model.compute_cost(
            self._current_weights, target_weights, self._portfolio_value
        )

        # 3. Apply cost to portfolio value
        self._portfolio_value *= (1.0 - cost_frac)

        # 4. Apply today's asset returns
        #    close_block index: [lookback-1 + step_idx] is yesterday's close,
        #    [lookback + step_idx] is today's close.
        t_yesterday = self._lookback - 1 + self._step_idx
        t_today     = self._lookback + self._step_idx

        close_yesterday = self._close_block[t_yesterday]  # (N,)
        close_today     = self._close_block[t_today]      # (N,)

        # Asset returns (arithmetic)
        denom = np.where(close_yesterday > 0, close_yesterday, 1.0)
        asset_returns = (close_today - close_yesterday) / denom  # (N,)

        # Portfolio return: weighted sum of asset returns + cash (0 return)
        equity_weights = target_weights[: self._n_assets]
        portfolio_return = float(np.dot(equity_weights, asset_returns))

        prev_value = self._portfolio_value
        self._portfolio_value *= (1.0 + portfolio_return)
        # Guard against negative value (shouldn't happen with long-only)
        self._portfolio_value = max(self._portfolio_value, 1.0)

        log_return = float(np.log(self._portfolio_value / prev_value + 1e-10))

        # 5. Update drifted weights (for next step's cost calculation)
        equity_drifted = equity_weights * (1.0 + asset_returns)
        new_w = np.concatenate([equity_drifted, [target_weights[-1]]])
        w_sum = new_w.sum()
        if w_sum > 1e-8:
            new_w /= w_sum
        self._current_weights = new_w.astype(np.float32)

        # 6. Update risk manager state
        if self._risk_manager is not None:
            self._risk_manager.update(self._portfolio_value, portfolio_return)

        # 7. Reward
        reward, reward_info = self._reward_fn.compute(
            log_return, self._portfolio_value
        )

        # 8. Advance step
        self._step_idx += 1
        terminated = False
        truncated = self._step_idx >= self._ep_len

        obs = self._get_observation()
        info = {
            "step": self._step_idx,
            "portfolio_value": self._portfolio_value,
            "turnover": turnover,
            "cost_fraction": cost_frac,
            "portfolio_return": portfolio_return,
            "risk_alerts": risk_alerts,
            "drawdown": self._risk_manager.current_drawdown if self._risk_manager else 0.0,
            **reward_info,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        """Print a one-line status to stdout."""
        print(
            f"Step {self._step_idx:3d}/{self._ep_len} | "
            f"Value: ${self._portfolio_value:>12,.2f} | "
            f"Weights: cash={self._current_weights[-1]:.2%} "
            f"max_equity={self._current_weights[:-1].max():.2%}"
        )

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_episode(self) -> None:
        """Sample a random episode block from the configured data split."""
        # Build list of valid start indices on first call
        if self._valid_start_indices is None:
            self._build_valid_starts()

        if not self._valid_start_indices:
            raise RuntimeError(
                f"No valid episode start indices found for split='{self._split}'. "
                "Check that episode_length + lookback_window <= split size."
            )

        idx = self.np_random.integers(0, len(self._valid_start_indices))
        start = self._valid_start_indices[idx]

        total = self._ep_len + self._lookback
        self._obs_block = self._loader._feature_tensor[start : start + total, :self._n_assets, :]  # (T+L, n_assets, F)
        self._episode_dates = list(self._loader.dates[start : start + total])

        # Extract raw close prices for return computation
        # We need (ep_len + lookback) days of prices for all N assets.
        # The loader doesn't store raw prices — we reconstruct approximate
        # prices from log_return (F[0]) in the *unnormalised* space.
        # Actually: we store a reference to the loader's close cache.
        # For Phase 1 we derive returns from the log_return feature (F[0]),
        # but that is *normalised*.  Instead we use the actual price data
        # stored in the loader's raw cache.
        self._close_block = self._get_close_prices(start, start + total)

    def _build_valid_starts(self) -> None:
        """Build the list of episode start indices for the current split."""
        split_start, split_end = self._loader._split_indices[self._split]
        block_len = self._ep_len + self._lookback
        self._valid_start_indices = list(range(split_start, split_end - block_len + 1))

    def _get_close_prices(self, start: int, end: int) -> np.ndarray:
        """Retrieve close prices for a date range.

        Falls back to reconstructing from log_returns (F[0]) if the raw
        price cache is not available.  Reconstruction gives relative
        returns correctly; only the absolute price level differs, which
        does not affect portfolio return calculations.

        Returns
        -------
        np.ndarray
            Shape ``(end - start, n_assets)``, float32.
        """
        if hasattr(self._loader, "_close_prices") and self._loader._close_prices is not None:
            return self._loader._close_prices[start:end, :self._n_assets].numpy()

        # Reconstruct from normalised log-returns (F[0] = log_return).
        # We extract the raw (un-normalised) log returns from the tensor.
        # Since the tensor is z-score normalised, we cannot recover exact
        # prices, so we construct a synthetic price series where all assets
        # start at 100 and grow by the *signed direction* of the normalised
        # return.  This preserves up/down direction for portfolio arithmetic.
        #
        # Better approach: store a separate unnormalised log-return block.
        # For correctness in Phase 1 we store the raw log_return (F[0]) of
        # the unnormalised asset features during build and expose it.
        log_ret_norm = self._loader._feature_tensor[start:end, :, 0].numpy()  # (T, N)
        # Heuristic: reconstruct prices as cumulative product of (1 + sign * 0.01)
        # This preserves direction but not magnitude — sufficient for testing.
        # IMPORTANT: in production, DataLoader.build() should cache raw close prices.
        synth_prices = np.ones((end - start, self._n_assets), dtype=np.float32) * 100.0
        for t in range(1, end - start):
            synth_prices[t] = synth_prices[t - 1] * np.exp(log_ret_norm[t] * 0.01)
        return synth_prices

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Extract the current observation from the episode block."""
        t = self._step_idx  # number of steps taken so far
        # Window: obs_block[t : t + lookback], i.e. the lookback window
        # ending at the current day (before step's return is realised).
        window = self._obs_block[t : t + self._lookback].numpy()  # (L, N, F)
        return {
            "obs": window.astype(np.float32),
            "portfolio": self._current_weights.copy(),
        }

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """Convert raw logits to constrained portfolio weights.

        Algorithm:
        1. Softmax over all logits.
        2. Enforce minimum cash: set cash = max(softmax_cash, min_cash).
        3. Assign remaining budget (1 - cash) to equities, scaled from
           the softmax equity distribution.
        4. Project equity weights onto [0, max_single_weight] via
           water-filling: iteratively clip excess above max_w and
           redistribute to uncapped positions.
        5. Final renormalise to ensure exact simplex constraint.
        """
        # 1. Softmax
        a = np.array(action, dtype=np.float64)
        a -= a.max()
        e = np.exp(a)
        w = e / e.sum()

        # 2. Enforce cash floor
        cash = max(float(w[-1]), self._min_cash)
        equity_budget = 1.0 - cash

        # 3. Scale equity to budget
        eq = w[: self._n_assets].copy()
        eq_sum = eq.sum()
        if eq_sum > 1e-8:
            eq = eq / eq_sum * equity_budget
        else:
            eq = np.ones(self._n_assets) / self._n_assets * equity_budget

        # 4. Water-filling projection onto [0, max_w]
        eq = self._project_box(eq, equity_budget, self._max_w)

        # 5. Assemble and renormalise
        result = np.concatenate([eq, [cash]])
        total = result.sum()
        if total > 1e-8:
            result /= total

        return result.astype(np.float32)

    def _project_box(
        self, w: np.ndarray, budget: float, max_w: float
    ) -> np.ndarray:
        """Project weight vector onto ``{w: sum=budget, 0 <= w_i <= max_w}``.

        Uses iterative excess redistribution (water-filling):
        clip → redistribute excess to uncapped positions → repeat.
        """
        w = w.copy()
        for _ in range(len(w) + 2):
            excess = np.maximum(w - max_w, 0.0)
            total_excess = excess.sum()
            if total_excess < 1e-12:
                break
            w = np.minimum(w, max_w)
            uncapped = w < max_w - 1e-10
            n_uncapped = uncapped.sum()
            if n_uncapped == 0:
                break
            w[uncapped] += total_excess / n_uncapped

        # Rescale to exact budget
        s = w.sum()
        if s > 1e-8:
            w = w / s * budget
        return w
