"""SAC actor and critic heads for the Alpha0 policy network.

Actor (stochastic policy)
--------------------------
The actor maps an encoded representation → (mean, log_std) for each of
the N+1 output logits (assets + cash).  Actions are sampled via the
reparameterization trick with tanh squashing:

    z      = mean + std * eps,    eps ~ N(0, I)
    action = tanh(z)              (bounded in (-1, 1)^{N+1})

The log-probability is corrected for the tanh Jacobian:

    log_pi(a|s) = sum_i [ log N(z_i; mu_i, sigma_i)
                          - log(1 - tanh²(z_i) + eps) ]

The MarketEnv receives the tanh-squashed logits and internally applies
softmax + constraint projection.  Using bounded logits (tanh output)
keeps the softmax numerically stable.

Critic (Q-function)
--------------------
Twin Q-networks take (encoded_representation, action) → scalar Q-value.
Using two independent critics and taking the minimum (Clipped Double-Q)
prevents over-estimation bias.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MAX =  2.0
LOG_STD_MIN = -5.0


class Actor(nn.Module):
    """SAC stochastic actor.

    Maps an encoded state representation to a distribution over actions.

    Parameters
    ----------
    d_model:
        Encoder output dimension.
    n_actions:
        Action dimensionality (n_assets + 1).
    hidden_dim:
        Hidden layer size of the MLP head.
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_actions: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_head    = nn.Linear(hidden_dim, n_actions)
        self.log_std_head = nn.Linear(hidden_dim, n_actions)

    def forward(
        self,
        representation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute action distribution parameters.

        Parameters
        ----------
        representation:
            Shape ``(B, d_model)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(mean, log_std)`` each of shape ``(B, n_actions)``.
        """
        h = self.net(representation)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self,
        representation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action with reparameterization + tanh squashing.

        Parameters
        ----------
        representation:
            Shape ``(B, d_model)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - ``action``: shape ``(B, n_actions)``, values in ``(-1, 1)``.
            - ``log_prob``: shape ``(B,)``, log probability of the action.
        """
        mean, log_std = self.forward(representation)
        std = log_std.exp()

        # Reparameterization
        eps = torch.randn_like(mean)
        z   = mean + std * eps

        # Tanh squashing
        action   = torch.tanh(z)
        log_prob = self._log_prob(z, mean, std)

        return action, log_prob

    def deterministic_action(self, representation: torch.Tensor) -> torch.Tensor:
        """Return tanh(mean) — deterministic policy for evaluation."""
        mean, _ = self.forward(representation)
        return torch.tanh(mean)

    def _log_prob(
        self,
        z: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """Log prob of the squashed Gaussian distribution."""
        # Gaussian log prob
        log_prob = -0.5 * (
            ((z - mean) / (std + 1e-8)) ** 2
            + 2.0 * torch.log(std + 1e-8)
            + math.log(2.0 * math.pi)
        )
        # Tanh Jacobian correction: log(1 - tanh²(z))
        log_prob = log_prob - torch.log(1.0 - torch.tanh(z).pow(2) + 1e-6)
        return log_prob.sum(dim=-1)  # sum over action dims → (B,)


class Critic(nn.Module):
    """A single Q-network.

    Encodes (representation, action) → scalar Q-value.

    Parameters
    ----------
    d_model:
        Encoder output dimension.
    n_actions:
        Action dimensionality.
    hidden_dim:
        Hidden layer size.
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_actions: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model + n_actions, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        representation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Q-value.

        Parameters
        ----------
        representation:
            Shape ``(B, d_model)``.
        action:
            Shape ``(B, n_actions)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 1)``.
        """
        x = torch.cat([representation, action], dim=-1)
        return self.net(x)


class AlphaPolicy(nn.Module):
    """Full actor + twin critics for SAC.

    This module bundles:
    - One actor encoder + actor head
    - Two critic encoders + critic heads (for Clipped Double-Q)

    The actor and critics have **separate encoders** to prevent the
    critic gradients from directly updating the actor's representation.

    Parameters
    ----------
    encoder_cls:
        Constructor for :class:`~alpha0.policy.transformer.MarketEncoder`.
        Called three times (actor, critic1, critic2) with ``**encoder_kwargs``.
    encoder_kwargs:
        Keyword arguments forwarded to ``encoder_cls``.
    actor_hidden:
        Hidden dim for the actor MLP head.
    critic_hidden:
        Hidden dim for each critic MLP head.
    dropout:
        Dropout probability applied uniformly.
    """

    def __init__(
        self,
        encoder_cls,
        encoder_kwargs: dict,
        actor_hidden: int = 256,
        critic_hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        d_model   = encoder_kwargs.get("d_model", 128)
        n_assets  = encoder_kwargs.get("n_assets", 100)
        n_actions = n_assets + 1  # assets + cash

        # Actor encoder + head
        self.actor_encoder = encoder_cls(**encoder_kwargs)
        self.actor         = Actor(d_model, n_actions, actor_hidden, dropout)

        # Twin critics (separate encoders)
        self.critic1_encoder = encoder_cls(**encoder_kwargs)
        self.critic1         = Critic(d_model, n_actions, critic_hidden, dropout)

        self.critic2_encoder = encoder_cls(**encoder_kwargs)
        self.critic2         = Critic(d_model, n_actions, critic_hidden, dropout)

        # Target critics (soft-updated copies)
        self.target_critic1_encoder = encoder_cls(**encoder_kwargs)
        self.target_critic2_encoder = encoder_cls(**encoder_kwargs)
        self.target_critic1 = Critic(d_model, n_actions, critic_hidden, dropout)
        self.target_critic2 = Critic(d_model, n_actions, critic_hidden, dropout)

        # Initialise targets to match critics
        self._hard_update_targets()

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def encode_obs(
        self,
        obs: torch.Tensor,
        portfolio: torch.Tensor,
        network: str = "actor",
    ) -> torch.Tensor:
        """Encode observations using the specified sub-network's encoder."""
        if network == "actor":
            return self.actor_encoder(obs, portfolio)
        elif network == "critic1":
            return self.critic1_encoder(obs, portfolio)
        elif network == "critic2":
            return self.critic2_encoder(obs, portfolio)
        elif network == "target1":
            return self.target_critic1_encoder(obs, portfolio)
        elif network == "target2":
            return self.target_critic2_encoder(obs, portfolio)
        else:
            raise ValueError(f"Unknown network: {network!r}")

    def act(
        self,
        obs: torch.Tensor,
        portfolio: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample (or deterministically select) an action.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            ``(action, log_prob)`` — log_prob is None if deterministic.
        """
        rep = self.actor_encoder(obs, portfolio)
        if deterministic:
            return self.actor.deterministic_action(rep), None
        return self.actor.sample(rep)

    def q_values(
        self,
        obs: torch.Tensor,
        portfolio: torch.Tensor,
        action: torch.Tensor,
        use_targets: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute twin Q-values.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(q1, q2)`` each of shape ``(B, 1)``.
        """
        if use_targets:
            rep1 = self.target_critic1_encoder(obs, portfolio)
            rep2 = self.target_critic2_encoder(obs, portfolio)
            return self.target_critic1(rep1, action), self.target_critic2(rep2, action)
        rep1 = self.critic1_encoder(obs, portfolio)
        rep2 = self.critic2_encoder(obs, portfolio)
        return self.critic1(rep1, action), self.critic2(rep2, action)

    # ------------------------------------------------------------------
    # Target network maintenance
    # ------------------------------------------------------------------

    def soft_update_targets(self, tau: float) -> None:
        """Exponential moving average update of target network weights.

        Parameters
        ----------
        tau:
            Soft update coefficient (e.g. 0.005).  Target weights are
            updated as: ``theta_target = tau * theta + (1 - tau) * theta_target``.
        """
        pairs = [
            (self.critic1_encoder, self.target_critic1_encoder),
            (self.critic2_encoder, self.target_critic2_encoder),
            (self.critic1,         self.target_critic1),
            (self.critic2,         self.target_critic2),
        ]
        for src, tgt in pairs:
            for p_src, p_tgt in zip(src.parameters(), tgt.parameters()):
                p_tgt.data.copy_(tau * p_src.data + (1.0 - tau) * p_tgt.data)

    def _hard_update_targets(self) -> None:
        """Copy critic weights to target critics (used at initialisation)."""
        pairs = [
            (self.critic1_encoder, self.target_critic1_encoder),
            (self.critic2_encoder, self.target_critic2_encoder),
            (self.critic1,         self.target_critic1),
            (self.critic2,         self.target_critic2),
        ]
        for src, tgt in pairs:
            tgt.load_state_dict(src.state_dict())

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts by sub-module."""
        def _count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        return {
            "actor_encoder": _count(self.actor_encoder),
            "actor_head":    _count(self.actor),
            "critic1":       _count(self.critic1_encoder) + _count(self.critic1),
            "critic2":       _count(self.critic2_encoder) + _count(self.critic2),
            "total":         _count(self),
        }
