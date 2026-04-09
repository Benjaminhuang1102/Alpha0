"""Market observation encoder for Alpha0.

Architecture (per spec):

  Input:  (B, T, N, F)  — batch × lookback × n_assets × n_features

  1. Per-asset embedding (shared MLP, applied identically to every asset
     and every timestep):
        (B, T, N, F) → (B, T, N, d_model)

  2. Temporal Transformer (causal self-attention over T, applied per-asset):
        Reshape  → (B*N, T, d_model)
        Causal transformer (n_temporal_layers layers)
        Take last token → (B*N, d_model)
        Reshape  → (B, N, d_model)

  3. Cross-asset Transformer (self-attention over N):
        (B, N, d_model) → (B, N, d_model)
        Mean-pool        → (B, d_model)

  4. Portfolio state injection (current weights):
        Concat portfolio  → (B, d_model + n_assets + 1)
        Linear projection → (B, d_model)

Output: (B, d_model) representation vector, passed to AllocationHead.

Design decisions
----------------
* Causal masking on the temporal transformer prevents any future leakage
  during the lookback window (the agent should not attend to future days).
* Parameter sharing on the per-asset embedding (same weights for every
  asset) encourages the model to learn transferable features and keeps
  the parameter count low — there are 100 assets, a separate embedding per
  asset would be 100× larger.
* Cross-asset attention has no causal mask (all assets at the same
  timestep are simultaneous observations).
* Total parameter count at default settings ≈ 1.5-2M.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (fixed, not learned)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MarketEncoder(nn.Module):
    """Encodes the (T, N, F) market observation into a fixed-size representation.

    Parameters
    ----------
    n_features:
        Number of input features per asset (F). Default 14.
    n_assets:
        Number of assets in the universe (N). Default 100.
    lookback:
        Temporal lookback window length (T). Default 60.
    d_model:
        Internal embedding dimension. Default 128.
    n_temporal_layers:
        Number of Transformer layers for temporal attention. Default 3.
    n_cross_layers:
        Number of Transformer layers for cross-asset attention. Default 1.
    n_heads:
        Number of attention heads (used for both temporal and cross). Default 4.
    d_ff:
        Feedforward dimension inside each Transformer layer. Default 256.
    dropout:
        Dropout probability. Default 0.1.
    """

    def __init__(
        self,
        n_features: int = 14,
        n_assets: int = 100,
        lookback: int = 60,
        d_model: int = 128,
        n_temporal_layers: int = 3,
        n_cross_layers: int = 1,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_assets = n_assets
        self.lookback = lookback
        self.d_model = d_model

        # 1. Shared per-asset feature embedding (MLP applied identically to all assets)
        self.asset_embed = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # 2. Temporal encoder (causal attention over T)
        self.temporal_pos_enc = PositionalEncoding(d_model, max_len=lookback + 1, dropout=dropout)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=False,  # post-LN enables PyTorch fast path (NestedTensor)
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer, num_layers=n_temporal_layers
        )
        # Causal mask: each position attends only to itself and previous positions
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(lookback),
        )

        # 3. Cross-asset encoder (full attention over N)
        cross_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=False,  # post-LN enables PyTorch fast path
        )
        self.cross_transformer = nn.TransformerEncoder(
            cross_layer, num_layers=n_cross_layers
        )

        # 4. Portfolio state projection
        # Concatenate d_model + (n_assets + 1) → d_model
        self.portfolio_proj = nn.Linear(d_model + n_assets + 1, d_model)

        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        obs: torch.Tensor,
        portfolio: torch.Tensor,
    ) -> torch.Tensor:
        """Encode market observations into a representation vector.

        Parameters
        ----------
        obs:
            Shape ``(B, T, N, F)`` — normalised feature tensor.
        portfolio:
            Shape ``(B, N+1)`` — current portfolio weights (incl. cash).

        Returns
        -------
        torch.Tensor
            Shape ``(B, d_model)`` — encoded representation.
        """
        B, T, N, F = obs.shape

        # 1. Per-asset embedding: treat each (t, n) as independent
        #    Reshape to (B*T*N, F), embed, reshape back
        x = obs.reshape(B * T * N, F)
        x = self.asset_embed(x)             # (B*T*N, d_model)
        x = x.reshape(B, T, N, self.d_model)

        # 2. Temporal attention: process T dimension per asset
        #    Reshape to (B*N, T, d_model)
        x = x.permute(0, 2, 1, 3)          # (B, N, T, d_model)
        x = x.reshape(B * N, T, self.d_model)
        x = self.temporal_pos_enc(x)
        x = self.temporal_transformer(x, is_causal=True)
        # Take the final timestep (most recent)
        x = x[:, -1, :]                     # (B*N, d_model)
        x = x.reshape(B, N, self.d_model)  # (B, N, d_model)

        # 3. Cross-asset attention: process N dimension
        x = self.cross_transformer(x)       # (B, N, d_model)
        x = x.mean(dim=1)                   # (B, d_model) — mean pool over assets

        # 4. Inject portfolio state
        x = torch.cat([x, portfolio], dim=-1)      # (B, d_model + N+1)
        x = self.portfolio_proj(x)                  # (B, d_model)
        x = self.out_norm(x)

        return x
