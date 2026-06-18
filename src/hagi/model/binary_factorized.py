"""Binary Factorized Linear layer.

Drop-in replacement for nn.Linear using 1-bit binary low-rank factors
with Straight-Through Estimator (STE) and per-output scale.

G002 of binary-factorized-moe ultragoal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFactorizedLinear(nn.Module):
    """1-bit binary low-rank linear layer.

    Weight matrix is approximated as W = (B1 @ B2).T * scale,
    where B1 and B2 are binarized to ±1 in forward pass.
    Backward flows through the continuous parameters via STE.
    """

    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Continuous parameters (binarized in forward)
        self.B1 = nn.Parameter(torch.randn(in_features, rank))
        self.B2 = nn.Parameter(torch.randn(rank, out_features))
        # Per-output scale
        self.scale = nn.Parameter(torch.ones(out_features))

    def _binarize(self, x: torch.Tensor) -> torch.Tensor:
        """Binarize to ±1 with Straight-Through Estimator.

        Forward uses sign(x). Backward flows as if x itself.
        """
        # torch.where with Python float scalars promotes to fp32 (torch default
        # dtype), which breaks under manual_bf16 precision (no autocast): the
        # model is cast to bf16 (loop.py:431) but the ±1 result is fp32, so the
        # reconstructed weight w = (b1 @ b2).t() is fp32 and F.linear(x_bf16,
        # w_fp32) raises a dtype-mismatch RuntimeError. Cast b back to x's dtype
        # so the binarized matrix matches the (possibly low-precision) input.
        b = torch.where(x >= 0, 1.0, -1.0).to(x.dtype)
        # STE: forward = sign, backward = identity
        return x + (b - x).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute y = x @ W.T where W = (B1 @ B2).T * scale.

        Args:
            x: [..., in_features]

        Returns:
            [..., out_features]
        """
        b1 = self._binarize(self.B1)
        b2 = self._binarize(self.B2)
        # Low-rank weight matrix [out_features, in_features]
        w = (b1 @ b2).t()
        w = w * self.scale.unsqueeze(1)
        return F.linear(x, w)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}"
