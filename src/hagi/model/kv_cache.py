"""Preallocated static KV cache for autoregressive decoding.

Replaces the list-of-tuples + torch.cat-per-step cache (O(T^2) copying during
generation) with per-layer preallocated buffers written by index.

Protocol compatibility: a ``StaticLayerCache`` supports ``cache[0]`` /
``cache[1]`` (returning the currently valid K/V views), so existing code that
reads ``past_key_value[0].shape[2]`` keeps working. Attention layers detect the
static cache via the ``update`` method and write in place instead of
concatenating.
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
        self.k_buf = torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        self.v_buf = torch.zeros_like(self.k_buf)
        self.seq_len = 0

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new K/V at the current position; return views over valid range."""
        t = k.size(2)
        end = self.seq_len + t
        if end > self.k_buf.size(2):
            raise RuntimeError(f"static KV cache overflow: {end} > {self.k_buf.size(2)}")
        if k.dtype != self.k_buf.dtype or k.device != self.k_buf.device:
            self.k_buf = self.k_buf.to(device=k.device, dtype=k.dtype)
            self.v_buf = self.v_buf.to(device=v.device, dtype=v.dtype)
        self.k_buf[:, :, self.seq_len:end] = k
        self.v_buf[:, :, self.seq_len:end] = v
        self.seq_len = end
        return self.k_buf[:, :, :end], self.v_buf[:, :, :end]

    def __getitem__(self, idx: int) -> torch.Tensor:
        buf = self.k_buf if idx == 0 else self.v_buf
        return buf[:, :, : self.seq_len]

    def reset(self) -> None:
        self.seq_len = 0


def make_static_cache(model, batch_size: int, max_seq_len: int) -> list[StaticLayerCache] | None:
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
    n_blocks = cfg.perception_layers + loops * cfg.reasoning_layers + cfg.expression_layers
    head_dim = tcfg.hidden_size // tcfg.num_query_heads
    max_seq_len = min(max_seq_len, tcfg.max_seq_len)
    return [
        StaticLayerCache(batch_size, tcfg.num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        for _ in range(n_blocks)
    ]
