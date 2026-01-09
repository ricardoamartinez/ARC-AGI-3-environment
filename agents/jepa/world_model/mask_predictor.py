"""
Masked token predictor for V-JEPA-style denoising in representation space.

This is a lightweight (token-wise) predictor used to map encoder token embeddings
to target encoder token embeddings, with the loss applied only on masked patches.

We keep this intentionally small so the predictive burden stays in the encoder representation,
mirroring the JEPA philosophy.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedTokenPredictor(nn.Module):
    """
    Token-wise MLP predictor with a residual connection.

    Input/Output: (B, N, D) -> (B, N, D)
    """

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        d = int(dim)
        h = int(hidden_dim) if hidden_dim is not None else int(d * 4)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, h),
            nn.GELU(),
            nn.Linear(h, d),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # Pre-norm + residual
        x = self.norm(tokens)
        return tokens + self.mlp(x)

