"""Clifford-space expert router (G005 first slice).

Drop-in for the dense ``nn.Linear(hidden, num_experts)`` used as the MoE
router. Scores per expert are produced by:
  1. projecting hidden states to Cl(3,0,0) multivectors (HiddenToMultivector)
  2. applying a learnable per-expert DomainRotor sandwich
  3. reading off the scalar (grade-0) component

Output shape matches the dense router: (B, T, num_experts).
"""

from __future__ import annotations

import torch
from torch import nn

from .clifford import BLADE_COUNT
from .hdim_full import DomainRotor, HiddenToMultivector


class CliffordExpertRouter(nn.Module):
    """Geometric MoE router using Cl(3,0,0) rotors."""

    def __init__(self, hidden_size: int, num_experts: int, heads: int = 1, blade_count: int = BLADE_COUNT):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.heads = heads
        self.blade_count = blade_count
        self.to_mv = HiddenToMultivector(hidden_size, num_experts, blade_count)
        self.rotor = DomainRotor(num_rotors=num_experts, heads=heads, blade_count=blade_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        mv = self.to_mv(x)  # (B, T, num_experts, blade_count)
        # Per-expert rotor sandwich, take scalar (blade 0) component.
        out = torch.empty(B, T, self.num_experts, self.blade_count, device=x.device, dtype=x.dtype)
        for i in range(self.num_experts):
            out[:, :, i, :] = self.rotor.sandwich(mv[:, :, i, :], i)
        return out[..., 0]
