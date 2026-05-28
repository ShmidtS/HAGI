"""Standard transformer building blocks: RMSNorm, RoPE, GQA attention, SwiGLU.

Shared by all four ablation models. Nothing novel here — this is the proven
substrate. The novelty lives in `gdr.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TransformerConfig:
    hidden_size: int = 768
    num_query_heads: int = 12
    num_kv_heads: int = 4
    intermediate_size: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    max_seq_len: int = 4096


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]. cos/sin: [T, D/2].
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rx1
    out[..., 1::2] = rx2
    return out


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.nq = cfg.num_query_heads
        self.nkv = cfg.num_kv_heads
        self.head_dim = cfg.hidden_size // cfg.num_query_heads
        assert self.nq % self.nkv == 0, "query heads must be divisible by kv heads"
        self.q_proj = nn.Linear(cfg.hidden_size, self.nq * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.nq * self.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos, sin) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nq, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nkv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # Expand KV heads to match query heads (GQA).
        rep = self.nq // self.nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos, sin) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x
