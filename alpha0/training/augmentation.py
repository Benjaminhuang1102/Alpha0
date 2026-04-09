"""Data augmentation for Alpha0 training.

Anti-overfitting measures applied during environment interaction:

1. **Observation noise injection** — small Gaussian noise added to the
   normalised feature tensor.  Prevents the policy from memorising exact
   feature values and improves generalisation across regimes.

2. **Reward noise** — tiny noise on rewards prevents the critic from
   over-fitting to spurious reward patterns.

3. **Random episode start** — already handled by MarketEnv; documented here
   for completeness.

These are applied only during *training*, never during evaluation.
"""

from __future__ import annotations

import numpy as np
import torch


class ObservationAugmenter:
    """Applies stochastic perturbations to observations during training.

    Parameters
    ----------
    obs_noise_std:
        Standard deviation of Gaussian noise added to the ``"obs"``
        tensor.  Features are already z-score normalised (≈ unit variance),
        so 0.01 is a light perturbation (~1% of a typical signal range).
    reward_noise_std:
        Std of noise added to rewards in the replay buffer (set to 0 to
        disable).
    """

    def __init__(
        self,
        obs_noise_std: float = 0.01,
        reward_noise_std: float = 0.0,
    ) -> None:
        self.obs_noise_std    = obs_noise_std
        self.reward_noise_std = reward_noise_std

    def augment_obs(self, obs: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to the observation array in-place (copy returned).

        Parameters
        ----------
        obs:
            Float32 array of shape ``(T, N, F)``.

        Returns
        -------
        np.ndarray
            Noisy copy, clipped to ``[-5, 5]`` (matches normalisation clip).
        """
        if self.obs_noise_std <= 0.0:
            return obs
        noisy = obs + np.random.normal(0.0, self.obs_noise_std, obs.shape).astype(np.float32)
        return np.clip(noisy, -5.0, 5.0)

    def augment_reward(self, reward: float) -> float:
        """Add noise to a scalar reward."""
        if self.reward_noise_std <= 0.0:
            return reward
        return reward + float(np.random.normal(0.0, self.reward_noise_std))
