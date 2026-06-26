"""Preallocated static KV cache for autoregressive decoding.

Replaces the list-of-tuples + torch.cat-per-step cache (O(T^2) copying during
generation) with per-layer preallocated buffers written by index.

Protocol compatibility: a ``StaticLayerCache`` supports ``cache[0]`` /
``cache[1]`` (returning the currently valid K/V views), so existing code that
reads ``past_key_value[0].shape[2]`` keeps working. Attention layers detect the
static cache via the ``update`` method and write in place instead of
concatenating.

``Int8StaticLayerCache`` extends this with per-head INT8 quantization: K/V are
stored as int8 with per-head fp16 scales, cutting cache memory by 2x. This
allows longer generation sequences within the same VRAM budget.
"""

from __future__ import annotations

import torch


class StaticLayerCache:
    """Per-layer preallocated K/V buffer with in-place index writes."""

    def __init__(
        self,
        batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        device=None,
        dtype=None,
    ):
        self.k_buf = torch.zeros(  # type: ignore[reportCallIssue]
            batch_size, num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype
        )
        self.v_buf = torch.zeros_like(self.k_buf)
        self.seq_len = 0

    def update(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new K/V at the current position; return views over valid range."""
        t = k.size(2)
        end = self.seq_len + t
        if end > self.k_buf.size(2):
            raise RuntimeError(
                f"static KV cache overflow: {end} > {self.k_buf.size(2)}"
            )
        if k.dtype != self.k_buf.dtype or k.device != self.k_buf.device:
            self.k_buf = self.k_buf.to(device=k.device, dtype=k.dtype)
            self.v_buf = self.v_buf.to(device=v.device, dtype=v.dtype)
        self.k_buf[:, :, self.seq_len : end] = k
        self.v_buf[:, :, self.seq_len : end] = v
        self.seq_len = end
        return self.k_buf[:, :, :end], self.v_buf[:, :, :end]

    def __getitem__(self, idx: int) -> torch.Tensor:
        buf = self.k_buf if idx == 0 else self.v_buf
        return buf[:, :, : self.seq_len]

    def reset(self) -> None:
        self.seq_len = 0


class Int8StaticLayerCache:
    """INT8-quantized static KV cache — 2x memory reduction vs bf16.

    K/V are stored as int8 with per-head fp16 scales. Each update quantizes
    the incoming K/V to int8 and stores the scale. Reads dequantize on-the-fly
    back to the original dtype for SDPA. The quantization error is bounded
    (±1/127 of the per-head max) and negligible for attention scores.
    """

    def __init__(
        self,
        batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        device=None,
        dtype=None,
    ):
        self.k_buf = torch.zeros(  # type: ignore[reportCallIssue]
            batch_size, num_kv_heads, max_seq_len, head_dim,
            device=device, dtype=torch.int8,
        )
        self.v_buf = torch.zeros_like(self.k_buf)
        self.k_scale = torch.zeros(  # type: ignore[reportCallIssue]
            batch_size, num_kv_heads, max_seq_len, 1,
            device=device, dtype=dtype or torch.float16,
        )
        self.v_scale = torch.zeros_like(self.k_scale)
        self._read_dtype = dtype or torch.float16
        self.seq_len = 0

    def update(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = k.size(2)
        end = self.seq_len + t
        if end > self.k_buf.size(2):
            raise RuntimeError(
                f"int8 KV cache overflow: {end} > {self.k_buf.size(2)}"
            )
        orig_dtype = k.dtype
        k_f32 = k.float()
        v_f32 = v.float()
        # Per-head abs max for symmetric quantization. Adding 1e-12 is
        # tautological for all practical values (abs_max >> 1e-12) and
        # avoids division by zero when an entire head is zero (0/1e-12 →
        # 0, which int8-casts to 0 — the correct result).
        k_abs_max = k_f32.abs().amax(dim=-1, keepdim=True) + 1e-12
        v_abs_max = v_f32.abs().amax(dim=-1, keepdim=True) + 1e-12
        self.k_buf[:, :, self.seq_len : end] = (k_f32 / k_abs_max * 127).to(torch.int8)
        self.v_buf[:, :, self.seq_len : end] = (v_f32 / v_abs_max * 127).to(torch.int8)
        self.k_scale[:, :, self.seq_len : end] = (k_abs_max / 127).to(self._read_dtype)
        self.v_scale[:, :, self.seq_len : end] = (v_abs_max / 127).to(self._read_dtype)
        self.seq_len = end
        k_out = self.k_buf[:, :, :end].to(orig_dtype) * self.k_scale[:, :, :end].to(orig_dtype)
        v_out = self.v_buf[:, :, :end].to(orig_dtype) * self.v_scale[:, :, :end].to(orig_dtype)
        return k_out, v_out

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx == 0:
            return self.k_buf[:, :, : self.seq_len].to(self._read_dtype) * self.k_scale[:, :, : self.seq_len]
        return self.v_buf[:, :, : self.seq_len].to(self._read_dtype) * self.v_scale[:, :, : self.seq_len]

    def reset(self) -> None:
        self.seq_len = 0


def make_static_cache(
    model, batch_size: int, max_seq_len: int
) -> list[StaticLayerCache] | None:
    """Build per-executed-block static caches from a HAGI model config.

    Returns None when the model config is unavailable or uses the HRM path
    (which does not thread per-block KV caches).
    """
    cfg = getattr(model, "cfg", None)
    tcfg = getattr(cfg, "transformer", None)
    if cfg is None or tcfg is None:
        return None
    if getattr(cfg, "hrm", False):
        return None
    try:
        param = next(model.parameters())
        device, dtype = param.device, param.dtype
    except (AttributeError, StopIteration):
        device, dtype = None, None
    loops = cfg.loop_count if cfg.use_loop else 1
    n_blocks = (
        cfg.perception_layers + loops * cfg.reasoning_layers + cfg.expression_layers
    )
    head_dim = tcfg.hidden_size // tcfg.num_query_heads
    max_seq_len = min(max_seq_len, tcfg.max_seq_len)
    return [
        StaticLayerCache(
            batch_size,
            tcfg.num_kv_heads,
            max_seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
        for _ in range(n_blocks)
    ]


def make_int8_static_cache(
    model, batch_size: int, max_seq_len: int
) -> list[Int8StaticLayerCache] | None:
    """Build INT8-quantized static KV caches — 2x memory vs bf16."""
    cfg = getattr(model, "cfg", None)
    tcfg = getattr(cfg, "transformer", None)
    if cfg is None or tcfg is None:
        return None
    if getattr(cfg, "hrm", False):
        return None
    try:
        param = next(model.parameters())
        device, dtype = param.device, param.dtype
    except (AttributeError, StopIteration):
        device, dtype = None, None
    loops = cfg.loop_count if cfg.use_loop else 1
    n_blocks = (
        cfg.perception_layers + loops * cfg.reasoning_layers + cfg.expression_layers
    )
    head_dim = tcfg.hidden_size // tcfg.num_query_heads
    max_seq_len = min(max_seq_len, tcfg.max_seq_len)
    return [
        Int8StaticLayerCache(
            batch_size,
            tcfg.num_kv_heads,
            max_seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
        for _ in range(n_blocks)
    ]
