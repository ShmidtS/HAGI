"""Binary Factorized Linear prototype.

First slice (G002) of Binary Factorized MoE ultragoal.

W_effective = B1 @ B2, with B1, B2 entries in {-1, +1} projected via STE.
Per-block scale along the output dimension.
Forward:  Y = scale_broadcast * (X @ B1) @ B2
Backward: straight-through estimator on B1, B2; standard on scale.

Drop-in for nn.Linear (no bias). No default model wiring yet.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class _SignSTE(torch.autograd.Function):
    """Sign with straight-through gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def _sign_ste(x: torch.Tensor) -> torch.Tensor:
    return _SignSTE.apply(x)


class BinaryFactorizedLinear(nn.Module):
    """nn.Linear drop-in with binary low-rank factorization.

    Args:
        in_features: input dimension.
        out_features: output dimension.
        rank: low-rank factor inner dimension.
        block_size: must divide both in_features and out_features; scale
            is shared across each block of ``block_size`` output channels.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, block_size: int):
        super().__init__()
        if in_features % block_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by block_size ({block_size})"
            )
        if out_features % block_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by block_size ({block_size})"
            )
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.block_size = block_size
        self.num_out_blocks = out_features // block_size

        self.b1 = nn.Parameter(torch.empty(in_features, rank).bernoulli_(0.5) * 2.0 - 1.0)
        self.b2 = nn.Parameter(torch.empty(rank, out_features).bernoulli_(0.5) * 2.0 - 1.0)
        self.scale = nn.Parameter(torch.full((self.num_out_blocks,), 1.0 / math.sqrt(rank)))

    def effective_weight(self) -> torch.Tensor:
        """Returns the (out_features, in_features) effective dense weight."""
        w = _sign_ste(self.b1) @ _sign_ste(self.b2)  # (in_features, out_features)
        scale_full = self.scale.repeat_interleave(self.block_size)
        return (w * scale_full.unsqueeze(0)).T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1_s = _sign_ste(self.b1)
        b2_s = _sign_ste(self.b2)
        h = torch.matmul(x, b1_s)
        y = torch.matmul(h, b2_s)
        scale_full = self.scale.repeat_interleave(self.block_size)
        return y * scale_full


class BinarySwiGLU(nn.Module):
    """SwiGLU FFN with all three projections replaced by BinaryFactorizedLinear.

    Same forward semantics as SwiGLU:
        y = down(silu(gate(x)) * up(x))
    where gate, up, down are BinaryFactorizedLinear.
    """

    def __init__(self, cfg, rank: int, block_size: int):
        super().__init__()
        self.gate = BinaryFactorizedLinear(cfg.hidden_size, cfg.intermediate_size, rank, block_size)
        self.up = BinaryFactorizedLinear(cfg.hidden_size, cfg.intermediate_size, rank, block_size)
        self.down = BinaryFactorizedLinear(cfg.intermediate_size, cfg.hidden_size, rank, block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        return self.down(F.silu(self.gate(x)) * self.up(x))
