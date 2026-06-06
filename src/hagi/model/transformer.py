"""Standard transformer building blocks: RMSNorm, RoPE, GQA attention, SwiGLU.

Shared by all four ablation models. Nothing novel here — this is the proven
substrate. The novelty lives in `gdr.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .binary_factorized import BinaryFactorizedLinear
from .triton_kernels import TRITON_AVAILABLE, rmsnorm_triton


@dataclass
class TransformerConfig:
    hidden_size: int = 768
    num_query_heads: int = 12
    num_kv_heads: int = 4
    intermediate_size: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    max_seq_len: int = 4096
    norm: str = "rmsnorm"
    qk_norm: bool = False
    use_binary_factorized: bool = False
    binary_factorized_rank: int = 8
    use_moe: bool = False
    num_experts: int = 8
    moe_top_k: int = 2
    moe_intermediate_size: int | None = None
    moe_alpha: float = 0.01

    def __post_init__(self):
        assert self.hidden_size % self.num_query_heads == 0, (
            f"hidden_size {self.hidden_size} not divisible by num_query_heads {self.num_query_heads}"
        )
        assert self.num_query_heads % self.num_kv_heads == 0, (
            f"num_query_heads {self.num_query_heads} not divisible by num_kv_heads {self.num_kv_heads}"
        )
        head_dim = self.hidden_size // self.num_query_heads
        assert head_dim % 2 == 0, f"head_dim {head_dim} must be even for RoPE"
        if self.use_moe and self.moe_intermediate_size is None:
            self.moe_intermediate_size = self.intermediate_size // self.num_experts


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if TRITON_AVAILABLE and x.is_cuda:
            return rmsnorm_triton(x, self.weight, self.eps)
        w = self.weight.to(x.dtype)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
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


def _make_linear(in_features: int, out_features: int, cfg: TransformerConfig) -> nn.Module:
    if cfg.use_binary_factorized:
        return BinaryFactorizedLinear(in_features, out_features, cfg.binary_factorized_rank)
    return nn.Linear(in_features, out_features, bias=False)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.nq = cfg.num_query_heads
        self.nkv = cfg.num_kv_heads
        self.head_dim = cfg.hidden_size // cfg.num_query_heads
        assert self.nq % self.nkv == 0, "query heads must be divisible by kv heads"
        self.q_proj = _make_linear(cfg.hidden_size, self.nq * self.head_dim, cfg)
        self.k_proj = _make_linear(cfg.hidden_size, self.nkv * self.head_dim, cfg)
        self.v_proj = _make_linear(cfg.hidden_size, self.nkv * self.head_dim, cfg)
        self.o_proj = _make_linear(self.nq * self.head_dim, cfg.hidden_size, cfg)
        self.qk_norm_enabled = bool(getattr(cfg, "qk_norm", False))
        if self.qk_norm_enabled:
            self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps)

    def forward(self, x: torch.Tensor, cos, sin, past_key_value=None, use_cache: bool = False, attn_mask=None) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nq, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nkv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.qk_norm_enabled:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)
        next_key_value = (k, v) if use_cache else None
        is_causal = attn_mask is None and past_key_value is None
        # PyTorch 2.0+ automatically selects flash-attention backend on Ampere+ when
        # head_dim <= 128 and dtype is FP16/BF16; no explicit context manager needed.
        # Use zero-copy expand+view instead of repeat_interleave to avoid materialising copies.
        rep = self.nq // self.nkv
        B, _, T, D = k.shape
        attn_k = k.unsqueeze(2).expand(B, self.nkv, rep, T, D).reshape(B, self.nq, T, D)
        attn_v = v.unsqueeze(2).expand(B, self.nkv, rep, T, D).reshape(B, self.nq, T, D)
        out = F.scaled_dot_product_attention(q, attn_k, attn_v, attn_mask=attn_mask, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, out.shape[2], -1)
        out = self.o_proj(out)
        return (out, next_key_value) if use_cache else out

    def forward_repacked(self, x: torch.Tensor, cos, sin, past_key_value=None, use_cache: bool = False, attn_mask=None):
        """Fused QKV projection for contiguous memory access during inference."""
        B, T, _ = x.shape
        assert isinstance(self.qkv_weight, torch.Tensor)
        qkv = F.linear(x, self.qkv_weight)
        q, k, v = qkv.split(self._qkv_splits, dim=-1)
        q = q.view(B, T, self.nq, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nkv, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nkv, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)
        next_key_value = (k, v) if use_cache else None
        is_causal = attn_mask is None and past_key_value is None
        rep = self.nq // self.nkv
        B, _, T, D = k.shape
        attn_k = k.unsqueeze(2).expand(B, self.nkv, rep, T, D).reshape(B, self.nq, T, D)
        attn_v = v.unsqueeze(2).expand(B, self.nkv, rep, T, D).reshape(B, self.nq, T, D)
        out = F.scaled_dot_product_attention(q, attn_k, attn_v, attn_mask=attn_mask, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, out.shape[2], -1)
        out = self.o_proj(out)
        return (out, next_key_value) if use_cache else out


class SwiGLU(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.gate = _make_linear(cfg.hidden_size, cfg.intermediate_size, cfg)
        self.up = _make_linear(cfg.hidden_size, cfg.intermediate_size, cfg)
        self.down = _make_linear(cfg.intermediate_size, cfg.hidden_size, cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))

    def forward_repacked(self, x: torch.Tensor) -> torch.Tensor:
        """Fused gate-up projection for contiguous memory access during inference."""
        assert isinstance(self.gate_up_weight, torch.Tensor)
        gate_up = F.linear(x, self.gate_up_weight)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        if cfg.use_moe:
            from .moe import MoESwiGLU
            self.mlp = MoESwiGLU(cfg)
        else:
            self.mlp = SwiGLU(cfg)

    def _apply_folded_norm(self, x: torch.Tensor, which: str) -> torch.Tensor:
        """Inline rsqrt-only normalization when the gamma weight has been folded."""
        eps = getattr(self, f"_{which}_norm_eps", None)
        if eps is not None:
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        if which == "attn":
            return self.attn_norm(x)
        return self.mlp_norm(x)

    def forward(
        self,
        x: torch.Tensor,
        cos,
        sin,
        past_key_value=None,
        use_cache: bool = False,
        gradient_checkpointing: bool = False,
        attn_mask=None,
    ):
        use_checkpoint = gradient_checkpointing and self.training and not use_cache
        use_repacked = not self.training and hasattr(self.attn, "qkv_weight")
        if use_checkpoint:
            h = self._apply_folded_norm(x, "attn")
            attn_out = checkpoint(
                lambda h, c, s: self.attn(h, c, s, attn_mask=attn_mask),
                h,
                cos,
                sin,
                use_reentrant=False,
            )
            next_key_value = None
        else:
            h = self._apply_folded_norm(x, "attn")
            if use_repacked:
                attn_out = self.attn.forward_repacked(h, cos, sin, past_key_value, use_cache, attn_mask)
            else:
                attn_out = self.attn(h, cos, sin, past_key_value, use_cache, attn_mask)
            if use_cache:
                attn_out, next_key_value = attn_out
            else:
                next_key_value = None
        assert isinstance(attn_out, torch.Tensor)
        x = x + attn_out
        if use_checkpoint:
            h = self._apply_folded_norm(x, "mlp")
            mlp_out = checkpoint(lambda h: self.mlp(h), h, use_reentrant=False)
        else:
            h = self._apply_folded_norm(x, "mlp")
            if use_repacked and hasattr(self.mlp, "gate_up_weight"):
                mlp_out = self.mlp.forward_repacked(h)
            else:
                mlp_out = self.mlp(h)
        if isinstance(mlp_out, tuple):
            mlp_out, aux_loss = mlp_out
            x = x + mlp_out
            if use_cache:
                return (x, next_key_value)
            return (x, aux_loss)
        assert isinstance(mlp_out, torch.Tensor)
        x = x + mlp_out
        return (x, next_key_value) if use_cache else x
