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
    fuse_qkv: bool = True
    fuse_gate_up: bool = True
    use_moe: bool = False
    num_experts: int = 8
    moe_top_k: int = 2
    moe_intermediate_size: int | None = None
    moe_alpha: float = 0.01

    def __post_init__(self):
        assert (
            self.hidden_size % self.num_query_heads == 0
        ), f"hidden_size {self.hidden_size} not divisible by num_query_heads {self.num_query_heads}"
        assert (
            self.num_query_heads % self.num_kv_heads == 0
        ), f"num_query_heads {self.num_query_heads} not divisible by num_kv_heads {self.num_kv_heads}"
        head_dim = self.hidden_size // self.num_query_heads
        assert head_dim % 2 == 0, f"head_dim {head_dim} must be even for RoPE"
        if self.use_moe and self.moe_intermediate_size is None:
            self.moe_intermediate_size = self.intermediate_size // self.num_experts


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.rms_norm(x, x.shape[-1:], self.weight, self.eps)


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(end=seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def _apply_rope_impl(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.cat([rx1, rx2], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return _apply_rope_impl(x, cos, sin)


_TORCH_MAJOR_MINOR = tuple(
    int(p) for p in torch.__version__.split("+")[0].split(".")[:2]
)
_SDPA_SUPPORTS_GQA = _TORCH_MAJOR_MINOR >= (2, 5)
# Eagerly probe flash-attention availability at import so torch.compile does
# not recompile when the lazy `_flash_available is None` branch fires on the
# first forward pass (a global-mutation guard).
def _probe_flash() -> bool:
    if not _SDPA_SUPPORTS_GQA or not torch.cuda.is_available():
        return False
    try:
        return bool(torch.backends.cuda.is_flash_attention_available())
    except (AttributeError, RuntimeError):
        return False


_flash_available: bool = _probe_flash()


def _use_enable_gqa(q: torch.Tensor) -> bool:
    """Use enable_gqa only when the flash SDPA backend can actually run.

    enable_gqa restricts SDPA backend selection: on builds without flash
    attention (e.g. Windows) SDPA silently falls back to the slow math kernel,
    which can halve training throughput. The expand fallback keeps backend
    choice free (mem-efficient kernel) and is faster in that case.
    """
    global _flash_available
    if not _SDPA_SUPPORTS_GQA or not q.is_cuda:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    return _flash_available


def _update_kv(k, v, past_key_value, use_cache: bool, attn_mask):
    """Apply the KV cache (in-place static cache or tuple concat).

    Returns (k, v, next_key_value, is_causal). A static cache (anything exposing
    ``update``) is written by index — no per-step torch.cat. A first call into an
    empty static cache is a causal prefill.
    """
    if past_key_value is None:
        return k, v, ((k, v) if use_cache else None), attn_mask is None
    if hasattr(past_key_value, "update"):
        prior_len = past_key_value.seq_len
        k, v = past_key_value.update(k, v)
        return (
            k,
            v,
            (past_key_value if use_cache else None),
            (attn_mask is None and prior_len == 0),
        )
    past_key, past_value = past_key_value
    k = torch.cat([past_key, k], dim=2)
    v = torch.cat([past_value, v], dim=2)
    return k, v, ((k, v) if use_cache else None), False


def _sdpa_gqa(q, k, v, attn_mask, is_causal: bool, nq: int, nkv: int):
    """SDPA with native GQA on PyTorch >= 2.5; zero-copy expand fallback otherwise.

    PyTorch 2.0+ automatically selects the flash-attention backend on Ampere+
    when head_dim <= 128 and dtype is FP16/BF16; no explicit context manager
    needed. The fallback's reshape after expand materializes K/V copies, which
    enable_gqa avoids entirely.
    """
    if nq == nkv:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal
        )
    if _use_enable_gqa(q):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, enable_gqa=True
        )
    rep = nq // nkv
    B, _, T, D = k.shape
    attn_k = k.unsqueeze(2).expand(B, nkv, rep, T, D).reshape(B, nq, T, D)
    attn_v = v.unsqueeze(2).expand(B, nkv, rep, T, D).reshape(B, nq, T, D)
    return F.scaled_dot_product_attention(
        q, attn_k, attn_v, attn_mask=attn_mask, is_causal=is_causal
    )


def _make_linear(
    in_features: int, out_features: int, cfg: TransformerConfig
) -> nn.Module:
    if cfg.use_binary_factorized:
        return BinaryFactorizedLinear(
            in_features, out_features, cfg.binary_factorized_rank
        )
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
        if getattr(cfg, "fuse_qkv", False) and not cfg.use_binary_factorized:
            if (
                isinstance(self.q_proj, nn.Linear)
                and isinstance(self.k_proj, nn.Linear)
                and isinstance(self.v_proj, nn.Linear)
            ):
                self._fuse_qkv()

    def _fuse_qkv(self):
        wq = self.q_proj.weight.data
        wk = self.k_proj.weight.data
        wv = self.v_proj.weight.data
        assert (
            isinstance(wq, torch.Tensor)
            and isinstance(wk, torch.Tensor)
            and isinstance(wv, torch.Tensor)
        )
        self.qkv_weight = nn.Parameter(torch.cat([wq, wk, wv], dim=0).contiguous())
        self._qkv_splits = [wq.size(0), wk.size(0), wv.size(0)]
        del self.q_proj, self.k_proj, self.v_proj

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        prefix = kwargs.get("prefix", "")
        if hasattr(self, "qkv_weight"):
            q, k, v = self.qkv_weight.split(self._qkv_splits, dim=0)
            state[f"{prefix}q_proj.weight"] = q
            state[f"{prefix}k_proj.weight"] = k
            state[f"{prefix}v_proj.weight"] = v
            state.pop(f"{prefix}qkv_weight", None)
            state.pop(f"{prefix}_qkv_splits", None)
        return state

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        qkv_key = prefix + "qkv_weight"
        q_key = prefix + "q_proj.weight"
        k_key = prefix + "k_proj.weight"
        v_key = prefix + "v_proj.weight"
        if qkv_key not in state_dict and all(
            k in state_dict for k in (q_key, k_key, v_key)
        ):
            q = state_dict[q_key]
            k = state_dict[k_key]
            v = state_dict[v_key]
            state_dict[qkv_key] = torch.cat([q, k, v], dim=0)
            del state_dict[q_key], state_dict[k_key], state_dict[v_key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos,
        sin,
        past_key_value=None,
        use_cache: bool = False,
        attn_mask=None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, T, _ = x.shape
        if hasattr(self, "qkv_weight"):
            qkv = F.linear(x, self.qkv_weight)
            q, k, v = qkv.split(self._qkv_splits, dim=-1)
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        q = q.view(B, T, self.nq, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nkv, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nkv, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.qk_norm_enabled:
            q = self.q_norm(q)
            k = self.k_norm(k)
        k, v, next_key_value, is_causal = _update_kv(
            k, v, past_key_value, use_cache, attn_mask
        )
        out = _sdpa_gqa(q, k, v, attn_mask, is_causal, self.nq, self.nkv)
        out = out.transpose(1, 2).reshape(B, out.shape[2], -1)
        out = self.o_proj(out)
        return (out, next_key_value) if use_cache else out

    def forward_repacked(
        self,
        x: torch.Tensor,
        cos,
        sin,
        past_key_value=None,
        use_cache: bool = False,
        attn_mask=None,
    ):
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
        k, v, next_key_value, is_causal = _update_kv(
            k, v, past_key_value, use_cache, attn_mask
        )
        out = _sdpa_gqa(q, k, v, attn_mask, is_causal, self.nq, self.nkv)
        out = out.transpose(1, 2).reshape(B, out.shape[2], -1)
        out = self.o_proj(out)
        return (out, next_key_value) if use_cache else out


class SwiGLU(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.gate = _make_linear(cfg.hidden_size, cfg.intermediate_size, cfg)
        self.up = _make_linear(cfg.hidden_size, cfg.intermediate_size, cfg)
        self.down = _make_linear(cfg.intermediate_size, cfg.hidden_size, cfg)
        if getattr(cfg, "fuse_gate_up", False) and not cfg.use_binary_factorized:
            if isinstance(self.gate, nn.Linear) and isinstance(self.up, nn.Linear):
                self._fuse_gate_up()

    def _fuse_gate_up(self):
        w1 = self.gate.weight.data
        w3 = self.up.weight.data
        assert isinstance(w1, torch.Tensor) and isinstance(w3, torch.Tensor)
        self.gate_up_weight = nn.Parameter(torch.cat([w1, w3], dim=0).contiguous())
        del self.gate, self.up

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        prefix = kwargs.get("prefix", "")
        if hasattr(self, "gate_up_weight"):
            gate, up = self.gate_up_weight.chunk(2, dim=0)
            state[f"{prefix}gate.weight"] = gate
            state[f"{prefix}up.weight"] = up
            state.pop(f"{prefix}gate_up_weight", None)
        return state

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        gate_up_key = prefix + "gate_up_weight"
        gate_key = prefix + "gate.weight"
        up_key = prefix + "up.weight"
        if gate_up_key in state_dict:
            state_dict.pop(gate_key, None)
            state_dict.pop(up_key, None)
        elif all(k in state_dict for k in (gate_key, up_key)):
            gate = state_dict[gate_key]
            up = state_dict[up_key]
            state_dict[gate_up_key] = torch.cat([gate, up], dim=0)
            del state_dict[gate_key], state_dict[up_key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "gate_up_weight"):
            gate_up = F.linear(x, self.gate_up_weight)
            gate, up = gate_up.chunk(2, dim=-1)
        else:
            gate = self.gate(x)
            up = self.up(x)
        return self.down(F.silu(gate) * up)

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
        self._use_repacked = hasattr(self.attn, "qkv_weight")
        self._mlp_repacked = hasattr(self.mlp, "gate_up_weight")
        # torch.compile intentionally disabled for this model

    def _apply_folded_norm(self, x: torch.Tensor, which: str) -> torch.Tensor:
        """Inline rsqrt-only normalization when the gamma weight has been folded."""
        eps = getattr(self, f"_{which}_norm_eps", None)
        if eps is not None:
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        if which == "attn":
            return self.attn_norm(x)
        return self.mlp_norm(x)

    def _attn_checkpoint(self, h, cos, sin, attn_mask):
        return self.attn(h, cos, sin, attn_mask=attn_mask)

    def _mlp_checkpoint(self, h):
        return self.mlp(h)

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
        use_repacked = getattr(self, "_use_repacked", False)
        if use_checkpoint:
            h = self._apply_folded_norm(x, "attn")
            attn_out = checkpoint(
                self._attn_checkpoint,
                h,
                cos,
                sin,
                attn_mask,
                use_reentrant=False,
            )
            next_key_value = None
        else:
            h = self._apply_folded_norm(x, "attn")
            if use_repacked:
                attn_out = self.attn.forward_repacked(
                    h, cos, sin, past_key_value, use_cache, attn_mask
                )
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
            mlp_out = checkpoint(self._mlp_checkpoint, h, use_reentrant=False)
        else:
            h = self._apply_folded_norm(x, "mlp")
            if use_repacked and getattr(self, "_mlp_repacked", False):
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
