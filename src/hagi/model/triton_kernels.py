"""Triton-optimized kernels for HAGI with PyTorch fallback paths.

Supports Windows (triton-windows 3.0+), CUDA-only execution, and
fp16 / bf16 / fp32. All kernels fall back to pure PyTorch when Triton
is unavailable or the device is not CUDA.
"""
# pyright: reportInvalidTypeForm=false, reportOptionalMemberAccess=false

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Triton availability probe (Windows-safe)
# ---------------------------------------------------------------------------
_triton_available = False

# Use __getattr__ so basedpyright sees these as Optional
_tl: Any = None
_triton: Any = None

try:
    import triton
    import triton.language as tl

    _triton_available = torch.cuda.is_available() and triton is not None
    _tl = tl
    _triton = triton
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_cuda(t: torch.Tensor) -> bool:
    return t.is_cuda


def _upcast_table(table: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    if table.dtype != target_dtype or table.device.type != "cuda":
        return table.to(device="cuda", dtype=target_dtype)
    return table


def _contiguous(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_contiguous() else t.contiguous()


# ---------------------------------------------------------------------------
# 1. Geometric Product kernel (Clifford algebra)
# ---------------------------------------------------------------------------
if _triton_available:
    assert _triton is not None
    assert _tl is not None

    @_triton.jit
    def _geometric_product_kernel(  # type: ignore
        x_ptr,
        y_ptr,
        table_ptr,
        out_ptr,
        stride_batch,
        BLOCK: "tl.constexpr",
        BLADE_COUNT: "tl.constexpr",
    ):
        pid = tl.program_id(0)
        x_ptr += pid * stride_batch
        y_ptr += pid * stride_batch
        out_ptr += pid * stride_batch

        offs = tl.arange(0, BLOCK)
        mask = offs < BLADE_COUNT

        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x_mat = x[None, :]  # [1, BLOCK]
        out_dtype = x_ptr.dtype.element_ty

        for c in range(BLADE_COUNT):
            table_c = tl.load(
                table_ptr
                + c * BLADE_COUNT * BLADE_COUNT
                + offs[:, None] * BLADE_COUNT
                + offs[None, :],
                mask=mask[:, None] & mask[None, :],
                other=0.0,
            ).to(tl.float32)
            # core computation uses tl.dot as requested
            temp = tl.dot(x_mat, table_c)  # [1, BLOCK]
            out_c = tl.sum(temp * y[None, :])
            tl.store(out_ptr + c, out_c.to(out_dtype))


def _geometric_product_torch(x: torch.Tensor, y: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Reference PyTorch implementation."""
    table = table if table.device == x.device and table.dtype == x.dtype else table.to(x.device, x.dtype)
    return torch.einsum("cab,...a,...b->...c", table, x, y)


def geometric_product_triton(x: torch.Tensor, y: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Geometric product via Triton (Clifford algebra).

    Args:
        x: [..., BLADE_COUNT] multivector coefficients.
        y: [..., BLADE_COUNT] multivector coefficients.
        table: [BLADE_COUNT, BLADE_COUNT, BLADE_COUNT] product table.

    Returns:
        [..., BLADE_COUNT] product coefficients.
    """
    if not _triton_available or not _ensure_cuda(x):
        return _geometric_product_torch(x, y, table)

    assert x.shape[-1] == y.shape[-1] == table.shape[0]
    blade_count = x.shape[-1]
    orig_shape = x.shape
    # table must be on CUDA and contiguous
    table = _contiguous(_upcast_table(table, x.dtype))

    batch = math.prod(x.shape[:-1])
    x = _contiguous(x).view(batch, blade_count)
    y = _contiguous(y).view(batch, blade_count)
    out = torch.empty_like(x)

    block = max(_triton.next_power_of_2(blade_count), 16)
    grid = (batch,)
    _geometric_product_kernel[grid](
        x, y, table, out,
        x.stride(0),
        BLOCK=block,
        BLADE_COUNT=blade_count,
    )
    return out.view(*orig_shape)


# ---------------------------------------------------------------------------
# 2. Sparse Attention kernel (flash-attention-like)
# ---------------------------------------------------------------------------
if _triton_available:
    @_triton.jit
    def _sparse_attention_fwd_kernel(  # type: ignore
        q_ptr,
        k_ptr,
        v_ptr,
        mask_ptr,
        out_ptr,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_mb,
        stride_mh,
        stride_mm,
        stride_mn,
        stride_ob,
        stride_oh,
        stride_om,
        stride_ok,
        batch,
        n_heads,
        seq_len,
        kv_len,
        head_dim,
        scale,
        BLOCK_M: "tl.constexpr",
        BLOCK_N: "tl.constexpr",
        BLOCK_D: "tl.constexpr",
        IS_CAUSAL: "tl.constexpr",
        USE_MASK: "tl.constexpr",
    ):
        # Each block handles one tile of queries
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        batch_id = pid_bh // n_heads
        head_id = pid_bh % n_heads

        start_m = pid_m * BLOCK_M

        # Pointers for this batch/head
        q_ptr += batch_id * stride_qb + head_id * stride_qh
        k_ptr += batch_id * stride_kb + head_id * stride_kh
        v_ptr += batch_id * stride_vb + head_id * stride_vh
        out_ptr += batch_id * stride_ob + head_id * stride_oh
        if USE_MASK:
            mask_ptr += batch_id * stride_mb + head_id * stride_mh

        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        m_mask = offs_m < seq_len
        n_mask = offs_n < kv_len
        d_mask = offs_d < head_dim

        # Load Q tile
        q = tl.load(
            q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
            mask=m_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Online softmax accumulators
        m = tl.full((BLOCK_M,), value=float("-inf"), dtype=tl.float32)
        l = tl.full((BLOCK_M,), value=0.0, dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        # Dynamic loop end
        if IS_CAUSAL:
            loop_end = tl.minimum(start_m + BLOCK_M, kv_len)
        else:
            loop_end = kv_len

        # Loop over key/value blocks
        for start_n in range(0, loop_end, BLOCK_N):
            offs_n_actual = start_n + offs_n

            # Load K tile
            k = tl.load(
                k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk,
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            # Load V tile
            v = tl.load(
                v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            # Compute QK^T
            qk = tl.dot(q, tl.trans(k))  # [BLOCK_M, BLOCK_N]
            qk = qk * scale

            # Apply causal mask
            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= offs_n_actual[None, :]
                qk = tl.where(causal_mask, qk, float("-inf"))

            # Apply sparse mask
            if USE_MASK:
                mask_block = tl.load(
                    mask_ptr
                    + offs_m[:, None] * stride_mm
                    + offs_n_actual[None, :] * stride_mn,
                    mask=m_mask[:, None] & (offs_n_actual[None, :] < kv_len),
                    other=0,
                )
                qk = tl.where(mask_block, qk, float("-inf"))

            # Online softmax
            m_new = tl.maximum(m, tl.max(qk, axis=1))
            p = tl.exp(qk - m_new[:, None])
            l = l * tl.exp(m - m_new) + tl.sum(p, axis=1)
            acc = acc * tl.exp(m - m_new)[:, None] + tl.dot(p, v)
            m = m_new

        # Normalize — avoid NaN when a query has no valid keys
        acc = acc / tl.where(l[:, None] > 0, l[:, None], 1.0)
        acc = tl.where(l[:, None] > 0, acc, 0.0)

        # Store output
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok,
            acc.to(q_ptr.dtype.element_ty),
            mask=m_mask[:, None] & d_mask[None, :],
        )


def _sparse_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    is_causal: bool,
) -> torch.Tensor:
    """Reference PyTorch implementation using scaled_dot_product_attention."""
    # q, k, v: [B, H, T, D]
    if mask is not None:
        # mask: True means attend
        attn_mask = mask
        if is_causal:
            T = q.size(2)
            K = k.size(2)
            causal = torch.tril(torch.ones(T, K, dtype=torch.bool, device=q.device))
            if mask.dim() == 2:
                attn_mask = causal & mask
            elif mask.dim() == 3:
                attn_mask = causal[None, :, :] & mask[:, None, :, :]
            elif mask.dim() == 4:
                attn_mask = causal[None, None, :, :] & mask
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
    else:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=is_causal)
    return out


@lru_cache(maxsize=32)
def _get_sparse_attn_config(seq_len: int, head_dim: int):
    """Cache kernel configurations per (seq_len, head_dim)."""
    block_m = 64 if seq_len >= 64 else 32
    block_n = 64 if seq_len >= 64 else 32
    block_d = head_dim
    return block_m, block_n, block_d


def sparse_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    is_causal: bool = True,
) -> torch.Tensor:
    """Block-sparse flash-attention-like forward pass.

    Args:
        q: [batch, n_heads, seq_len, head_dim].
        k: [batch, n_heads, kv_len, head_dim].
        v: [batch, n_heads, kv_len, head_dim].
        mask: Optional dense boolean mask [batch, n_heads, seq_len, kv_len]
            or [seq_len, kv_len] or [batch, seq_len, kv_len]. True = attend.
            Float masks are converted to bool via ``mask >= 0``.
        is_causal: Apply causal masking in addition to the sparse mask.

    Returns:
        [batch, n_heads, seq_len, head_dim] attention output.
    """
    if not _triton_available or not _ensure_cuda(q):
        return _sparse_attention_torch(q, k, v, mask, is_causal)

    batch, n_heads, seq_len, head_dim = q.shape
    kv_len = k.shape[2]
    assert k.shape == (batch, n_heads, kv_len, head_dim)
    assert v.shape == (batch, n_heads, kv_len, head_dim)

    if head_dim > 128:
        # Fallback for large head_dim to avoid kernel complexity
        return _sparse_attention_torch(q, k, v, mask, is_causal)

    q, k, v = _contiguous(q), _contiguous(k), _contiguous(v)
    out = torch.empty_like(q)

    block_m, block_n, block_d = _get_sparse_attn_config(seq_len, head_dim)
    scale = float(1.0 / math.sqrt(head_dim))

    use_mask = mask is not None
    if use_mask:
        if mask.dtype != torch.bool:
            mask = mask >= 0
        if mask.dim() == 2:
            mask = mask[None, None, :, :].expand(batch, n_heads, seq_len, kv_len)
        elif mask.dim() == 3:
            mask = mask[:, None, :, :].expand(batch, n_heads, seq_len, kv_len)
        elif mask.dim() == 4:
            if mask.shape != (batch, n_heads, seq_len, kv_len):
                mask = mask.expand(batch, n_heads, seq_len, kv_len)
        mask = _contiguous(mask)

    grid = (_triton.cdiv(seq_len, block_m), batch * n_heads)
    if use_mask:
        assert mask is not None
        _mask_ptr: torch.Tensor = mask
    else:
        _mask_ptr: torch.Tensor = q
    _sparse_attention_fwd_kernel[grid](
        q,
        k,
        v,
        _mask_ptr,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        _mask_ptr.stride(0) if use_mask else 0,
        _mask_ptr.stride(1) if use_mask else 0,
        _mask_ptr.stride(2) if use_mask else 0,
        _mask_ptr.stride(3) if use_mask else 0,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        batch,
        n_heads,
        seq_len,
        kv_len,
        head_dim,
        scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        IS_CAUSAL=is_causal,
        USE_MASK=use_mask,
    )
    return out


# ---------------------------------------------------------------------------
# 3. RMSNorm kernel
# ---------------------------------------------------------------------------
if _triton_available:
    @_triton.jit
    def _rmsnorm_kernel(  # type: ignore
        x_ptr,
        w_ptr,
        out_ptr,
        stride_row,
        n_cols,
        eps,
        BLOCK_SIZE: "tl.constexpr",
    ):
        row = tl.program_id(0)
        x_ptr += row * stride_row
        out_ptr += row * stride_row

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        x = tl.load(x_ptr + cols, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        x_sq = x_f32 * x_f32
        var = tl.sum(x_sq, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = x_f32 * rstd * w
        tl.store(out_ptr + cols, out.to(x_ptr.dtype.element_ty), mask=mask)


def _rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference PyTorch fused RMSNorm."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm via Triton.

    Args:
        x: [..., n_cols] input.
        weight: [n_cols] learnable weight.
        eps: numerical stability constant.

    Returns:
        [..., n_cols] normalized output.
    """
    if not _triton_available or not _ensure_cuda(x):
        return _rmsnorm_torch(x, weight, eps)

    n_cols = x.shape[-1]
    x = _contiguous(x)
    out = torch.empty_like(x)
    x_view = x.view(-1, n_cols)
    out_view = out.view(-1, n_cols)
    rows = x_view.shape[0]
    weight = _contiguous(weight)

    block = max(triton.next_power_of_2(n_cols), 128)
    grid = (rows,)
    _rmsnorm_kernel[grid](
        x_view,
        weight,
        out_view,
        x_view.stride(0),
        n_cols,
        eps=eps,
        BLOCK_SIZE=block,
    )
    return out


# ---------------------------------------------------------------------------
# Autograd wrappers (Triton forward, PyTorch backward)
# ---------------------------------------------------------------------------
class RMSNormTriton(torch.autograd.Function):
    """Triton RMSNorm forward with manual backward for gradient correctness."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return rmsnorm_triton(x, weight, eps)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        # Manual RMSNorm backward in float32 for numerical stability
        grad_f = grad_output.float()
        x_f = x.float()
        w_f = weight.float()
        var = x_f.pow(2).mean(-1, keepdim=True) + eps
        rstd = torch.rsqrt(var)
        grad_w = (grad_f * x_f * rstd).sum(dim=list(range(grad_f.ndim - 1)))
        grad_x = grad_f * w_f * rstd - x_f * rstd * (grad_f * x_f * w_f).mean(-1, keepdim=True) / var
        return grad_x.to(x.dtype), grad_w.to(weight.dtype), None


class GeometricProductTriton(torch.autograd.Function):
    """Triton geometric product forward with manual backward for gradient correctness."""

    @staticmethod
    def forward(ctx, x, y, table):
        ctx.save_for_backward(x, y, table)
        return geometric_product_triton(x, y, table)

    @staticmethod
    def backward(ctx, grad_output):
        x, y, table = ctx.saved_tensors
        # Manual backward in float32 for numerical stability
        grad_f = grad_output.float()
        x_f = x.float()
        y_f = y.float()
        table_f = table.float()
        grad_x = torch.einsum("cab,...c,...b->...a", table_f, grad_f, y_f)
        grad_y = torch.einsum("cab,...c,...a->...b", table_f, grad_f, x_f)
        return grad_x.to(x.dtype), grad_y.to(y.dtype), None


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------
__all__ = [
    "geometric_product_triton",
    "sparse_attention_triton",
    "rmsnorm_triton",
    "RMSNormTriton",
    "GeometricProductTriton",
    "TRITON_AVAILABLE",
]

TRITON_AVAILABLE = _triton_available
