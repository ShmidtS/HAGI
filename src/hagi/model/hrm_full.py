"""Full Hierarchical Recurrent Model core with strategic and tactical states."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .transformer import TransformerBlock, TransformerConfig, build_rope_cache


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
        vocab_size: int,
        hidden_size: int,
        h_dim: int = 256,
        l_dim: int = 256,
        h_cycles: int = 2,
        l_cycles: int = 3,
        transformer: TransformerConfig | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.h_dim = h_dim
        self.l_dim = l_dim
        self.h_cycles = h_cycles
        self.l_cycles = l_cycles
        self.transformer = transformer if transformer is not None else TransformerConfig(hidden_size=hidden_size)

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.block = TransformerBlock(self.transformer)
        self.z_l_to_hidden = nn.Linear(l_dim, hidden_size)
        self.h_init = nn.Linear(hidden_size, h_dim)
        self.l_init = nn.Linear(hidden_size, l_dim)
        self.h_transition = HTransition(h_dim, l_dim)
        self.l_transition = LTransition(l_dim, hidden_size, h_dim)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self._rope = {}

    def _rope_cache(self, T: int, device, dtype):
        key = (T, device, dtype)
        if key not in self._rope:
            head_dim = self.transformer.hidden_size // self.transformer.num_query_heads
            self._rope[key] = build_rope_cache(T, head_dim, self.transformer.rope_theta, device, dtype)
        return self._rope[key]

    def _attention_mask(self, prefix_mask, partition, device):
        mask = None
        if prefix_mask is not None:
            mask = prefix_mask.to(device=device, dtype=torch.bool)
        if partition is not None:
            part = torch.as_tensor(partition, device=device)
            same_partition = part[:, :, None] == part[:, None, :]
            causal = torch.ones(part.shape[-1], part.shape[-1], device=device, dtype=torch.bool).tril()
            partition_mask = same_partition & causal
            mask = partition_mask if mask is None else mask & partition_mask
        if mask is not None and mask.ndim == 3:
            mask = mask.unsqueeze(1)
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        z_H: torch.Tensor | HState | None = None,
        z_L: torch.Tensor | LState | None = None,
        prefix_mask=None,
        partition=None,
    ):
        h = self.embed(input_ids)
        B, T, _ = h.shape
        pooled = h.mean(dim=1)

        if isinstance(z_H, HState):
            z_H = z_H.z_H
        if isinstance(z_L, LState):
            z_L = z_L.z_L
        if z_H is None:
            z_H = self.h_init(pooled)
        if z_L is None:
            z_L = self.l_init(pooled)

        cos, sin = self._rope_cache(T, h.device, h.dtype)
        attn_mask = self._attention_mask(prefix_mask, partition, h.device)

        for _ in range(self.h_cycles):
            for _ in range(self.l_cycles):
                z_l_hidden = self.z_l_to_hidden(z_L).unsqueeze(1).expand(B, T, self.hidden_size)
                h = self.block(h + z_l_hidden, cos, sin, attn_mask=attn_mask)
                z_L = self.l_transition(z_L, h)
            z_H = self.h_transition(z_H, z_L)
            z_L = self.l_transition.reset(z_H)

        logits = self.lm_head(self.final_norm(h))
        return logits, HState(z_H), LState(z_L)
