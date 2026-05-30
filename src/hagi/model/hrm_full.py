"""Hierarchical Recurrent Model reasoning controller."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import nn

from .transformer import TransformerBlock


@dataclass
class HState:
    z_H: torch.Tensor


@dataclass
class LState:
    z_L: torch.Tensor


class HTransition(nn.Module):
    def __init__(self, h_dim: int, l_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim + l_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
        )

    def forward(self, z_H_prev: torch.Tensor, z_L_last: torch.Tensor) -> torch.Tensor:
        return z_H_prev + self.net(torch.cat([z_H_prev, z_L_last], dim=-1))


class LTransition(nn.Module):
    def __init__(self, l_dim: int, hidden_size: int, h_dim: int | None = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(l_dim + hidden_size, l_dim),
            nn.LayerNorm(l_dim),
            nn.GELU(),
            nn.Linear(l_dim, l_dim),
        )
        self.reset_proj = nn.Linear(h_dim if h_dim is not None else hidden_size, l_dim)

    def forward(self, z_L_prev: torch.Tensor, transformer_output: torch.Tensor) -> torch.Tensor:
        pooled = transformer_output.mean(dim=1)
        return z_L_prev + self.net(torch.cat([z_L_prev, pooled], dim=-1))

    def reset(self, z_H: torch.Tensor) -> torch.Tensor:
        return self.reset_proj(z_H)


class HRMCore(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        h_dim: int = 256,
        l_dim: int = 256,
        h_cycles: int = 2,
        l_cycles: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.h_cycles = h_cycles
        self.l_cycles = l_cycles

        self.h_init = nn.Linear(hidden_size, h_dim)
        self.l_init = nn.Linear(hidden_size, l_dim)
        self.h_transition = HTransition(h_dim, l_dim)
        self.l_transition = LTransition(l_dim, hidden_size, h_dim)
        self.z_l_to_hidden = nn.Linear(l_dim, hidden_size)
        self.z_h_to_hidden = nn.Linear(h_dim, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        reasoning_blocks: Sequence[TransformerBlock],
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask=None,
        z_H: torch.Tensor | HState | None = None,
        z_L: torch.Tensor | LState | None = None,
    ):
        h = hidden_states
        B, T, H = h.shape
        pooled = h.mean(dim=1)

        if isinstance(z_H, HState):
            z_H = z_H.z_H
        if isinstance(z_L, LState):
            z_L = z_L.z_L
        if z_H is None:
            z_H = self.h_init(pooled)
        if z_L is None:
            z_L = self.l_init(pooled)

        for _ in range(self.h_cycles):
            for _ in range(self.l_cycles):
                z_l_hidden = self.z_l_to_hidden(z_L).unsqueeze(1).expand(B, T, H)
                z_h_hidden = self.z_h_to_hidden(z_H).unsqueeze(1).expand(B, T, H)
                h_in = h + z_l_hidden + z_h_hidden
                for block in reasoning_blocks:
                    h = block(h_in, cos, sin, attn_mask=attn_mask)
                    h_in = h
                z_L = self.l_transition(z_L, h)
            z_H = self.h_transition(z_H, z_L)
            z_L = self.l_transition.reset(z_H)

        return h, HState(z_H), LState(z_L)
