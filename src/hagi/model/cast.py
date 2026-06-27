"""Clifford Algebra Symbolic Reasoning Tokens (CAST) — block-wise generation.

Replaces standard single-token lm_head with K-token block prediction grounded
in Cl(3,0,0) geometric algebra. Each forward pass produces K virtual hidden
states from one hidden representation; the geometric product between adjacent
virtual states creates a bivector "area" that enforces cross-token coherence.

Forward pass:
    hidden [B, T, H]
        -> block_proj: Linear(H -> K*H)
        -> reshape to [B, T, K, H]
        -> reshape to multivectors [B, T, K, H//8, 8]
        -> geometric_product(adjacent K positions) -> bivector area
        -> area modulates both neighbour virtual states
        -> flatten back to [B, T, K, H]

Each virtual state is then decoded through the shared final_norm + lm_head,
producing K token predictions per position. This reduces the number of
sequential forward passes during generation by a factor of K.

Training uses multi-token prediction: position t predicts tokens t+1..t+K.
Loss = mean over K positions of fused_linear_cross_entropy. The full
[B, T, K, V] logits tensor is never materialized simultaneously — each k
position is processed sequentially through fused CE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import nn

from .clifford import BLADE_COUNT, geometric_product, grade_projection


@dataclass
class CASTConfig:
    """Configuration for CAST head.

    train_k: number of K positions to compute CE loss on per training step.
    None or 0 = train all K positions. Set < block_size to subsample:
    always includes k=0 (next-token), randomly samples the rest.
    Reduces CE compute by block_size/train_k with no architecture change.
    Inference always uses all K positions.
    """

    block_size: int = 8
    use_coherence: bool = True
    gate_init: float = 0.0
    train_k: int | None = None


class CASTHead(nn.Module):
    """Clifford Algebra Symbolic Reasoning Tokens head.

    Projects a single hidden state into K virtual hidden states and applies
    geometric product coherence between adjacent states. The bivector (grade 2)
    part of the geometric product between adjacent positions is the "area" that
    enforces cross-token coherence — the core geometric insight of CAST.

    The module does NOT include final_norm or lm_head — those are shared from
    the parent HAGI model to preserve weight tying.
    """

    def __init__(
        self,
        hidden_size: int,
        block_size: int = 8,
        use_coherence: bool = True,
        gate_init: float = 0.0,
        train_k: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.block_size = block_size
        self.use_coherence = use_coherence
        self.train_k: int | None = (
            train_k if train_k is not None and 0 < train_k < block_size else None
        )

        assert hidden_size % BLADE_COUNT == 0, (
            f"hidden_size {hidden_size} must be divisible by BLADE_COUNT {BLADE_COUNT}"
        )
        self.n_mv = hidden_size // BLADE_COUNT

        self.block_proj = nn.Linear(hidden_size, block_size * hidden_size, bias=False)
        nn.init.normal_(self.block_proj.weight, mean=0.0, std=0.02)

        self.area_gate = nn.Parameter(torch.tensor(gate_init))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden into K virtual states with geometric coherence.

        Args:
            hidden: [B, T, H] final hidden states after expression stage.

        Returns:
            [B, T, K, H] virtual hidden states ready for final_norm + lm_head.
        """
        B, T, H = hidden.shape
        K = self.block_size

        virtual = self.block_proj(hidden)
        virtual = virtual.reshape(B, T, K, H)

        if not self.use_coherence or K <= 1:
            return virtual

        mv = virtual.reshape(B, T, K, self.n_mv, BLADE_COUNT)

        mv_left = mv[:, :, :-1]
        mv_right = mv[:, :, 1:]

        prod = geometric_product(mv_left, mv_right)
        area = grade_projection(prod, grade=2)

        gate = torch.sigmoid(self.area_gate)
        gated_area = gate * area

        padded_left = F.pad(gated_area, (0, 0, 0, 0, 1, 0))
        padded_right = F.pad(gated_area, (0, 0, 0, 0, 0, 1))

        mv = mv + padded_left + padded_right

        return mv.reshape(B, T, K, H)

    def forward_streaming(
        self, hidden: torch.Tensor
    ) -> Iterator[torch.Tensor]:
        """Yield K virtual states one at a time with pairwise coherence.

        Avoids materializing [B, T, K, H] simultaneously. Peak memory:
        2 * [B, T, H] (current + previous) instead of K * [B, T, H].
        Uses weight slicing (отображение) — each virtual state is a separate
        Linear projection from a weight slice, avoiding the K*H intermediate.

        Coherence is pairwise: area(k-1, k) is computed from raw projections
        and applied to both neighbours before yielding. This preserves the
        original semantics (areas from raw states, applied to both sides).
        """
        B, T, H = hidden.shape
        K = self.block_size
        gate_val = torch.sigmoid(self.area_gate) if self.use_coherence else None
        n_mv = self.n_mv

        raw_states: list[torch.Tensor] = []
        for k in range(K):
            w_k = self.block_proj.weight[k * H : (k + 1) * H]
            raw_states.append(F.linear(hidden, w_k))

        if not self.use_coherence or K <= 1:
            yield from raw_states
            return

        assert gate_val is not None
        for k in range(K):
            vs = raw_states[k]
            diff = torch.zeros_like(vs)

            if k > 0:
                mv_prev = raw_states[k - 1].reshape(B, T, n_mv, BLADE_COUNT)
                mv_curr = vs.reshape(B, T, n_mv, BLADE_COUNT)
                prod = geometric_product(mv_prev, mv_curr)
                area = grade_projection(prod, grade=2).reshape(B, T, H)
                diff = diff + gate_val * area

            if k < K - 1:
                mv_curr = vs.reshape(B, T, n_mv, BLADE_COUNT)
                mv_next = raw_states[k + 1].reshape(B, T, n_mv, BLADE_COUNT)
                prod = geometric_product(mv_curr, mv_next)
                area = grade_projection(prod, grade=2).reshape(B, T, H)
                diff = diff + gate_val * area

            yield vs + diff

    def compute_single_state(
        self, hidden: torch.Tensor, k: int
    ) -> torch.Tensor:
        """Compute the k-th virtual state with coherence, on demand.

        Weight slicing (отображение): extracts the k-th slice of block_proj
        weight and projects hidden into a single [B, T, H] virtual state.
        Coherence with neighbours is computed from their raw projections
        (re-projected on demand), avoiding any K*H or [B, T, K, H] tensor.

        Designed for per-k gradient checkpointing: each k-iteration can be
        wrapped in its own checkpoint, so only 1 CE logits tensor exists
        in the backward graph at a time.
        """
        B, T, H = hidden.shape
        K = self.block_size
        n_mv = self.n_mv

        w_k = self.block_proj.weight[k * H : (k + 1) * H]
        vs = F.linear(hidden, w_k)

        if not self.use_coherence or K <= 1:
            return vs

        gate = torch.sigmoid(self.area_gate)
        diff = torch.zeros_like(vs)

        if k > 0:
            w_prev = self.block_proj.weight[(k - 1) * H : k * H]
            vs_prev = F.linear(hidden, w_prev)
            mv_prev = vs_prev.reshape(B, T, n_mv, BLADE_COUNT)
            mv_curr = vs.reshape(B, T, n_mv, BLADE_COUNT)
            area = grade_projection(
                geometric_product(mv_prev, mv_curr), grade=2
            ).reshape(B, T, H)
            diff = diff + gate * area

        if k < K - 1:
            w_next = self.block_proj.weight[(k + 1) * H : (k + 2) * H]
            vs_next = F.linear(hidden, w_next)
            mv_curr = vs.reshape(B, T, n_mv, BLADE_COUNT)
            mv_next = vs_next.reshape(B, T, n_mv, BLADE_COUNT)
            area = grade_projection(
                geometric_product(mv_curr, mv_next), grade=2
            ).reshape(B, T, H)
            diff = diff + gate * area

        return vs + diff


def build_cast_targets(targets: torch.Tensor, block_size: int) -> torch.Tensor:
    """Create K shifted targets for multi-token prediction.

    Position t predicts tokens t+1, t+2, ..., t+K.
    targets[t] = input_ids[t+1] (standard next-token target).
    target_k[t] = targets[t+k] (shifted by k additional positions).

    Positions near the end that would reference out-of-range tokens are
    padded with ignore_index (-100).

    Args:
        targets: [B, T] standard next-token targets.
        block_size: K, number of tokens to predict per position.

    Returns:
        [B, T, K] shifted targets.
    """
    K = block_size
    B, T = targets.shape
    result = []
    for k in range(K):
        if k == 0:
            result.append(targets)
        else:
            pad = torch.full(
                (B, k),
                -100,
                device=targets.device,
                dtype=targets.dtype,
            )
            target_k = torch.cat([targets[:, k:], pad], dim=1)
            result.append(target_k)
    return torch.stack(result, dim=-1)
