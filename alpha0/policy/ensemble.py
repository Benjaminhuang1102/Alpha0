"""Ensemble wrapper for multiple Alpha0 agents (Phase 4).

During Phase 2/3, this module is a stub.  In Phase 4, five agents trained
with different random seeds are loaded here, and the final portfolio
allocation is the mean of their individual softmax outputs.

Using an ensemble reduces variance in the live allocation and makes the
strategy more robust to individual model overfitting.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EnsemblePolicy(nn.Module):
    """Average the deterministic actions of multiple :class:`AlphaPolicy` agents.

    Parameters
    ----------
    policies:
        List of trained :class:`~alpha0.policy.allocation_head.AlphaPolicy`
        instances.  All must share the same action dimensionality.
    """

    def __init__(self, policies: list[nn.Module]) -> None:
        super().__init__()
        self.policies = nn.ModuleList(policies)

    def act(
        self,
        obs: torch.Tensor,
        portfolio: torch.Tensor,
    ) -> torch.Tensor:
        """Return the mean deterministic action across all ensemble members.

        Parameters
        ----------
        obs:
            Shape ``(B, T, N, F)``.
        portfolio:
            Shape ``(B, N+1)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, N+1)`` — averaged tanh logits.
        """
        actions = []
        with torch.no_grad():
            for policy in self.policies:
                action, _ = policy.act(obs, portfolio, deterministic=True)
                actions.append(action)
        return torch.stack(actions, dim=0).mean(dim=0)
