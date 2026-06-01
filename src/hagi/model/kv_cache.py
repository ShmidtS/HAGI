"""KV cache quantization utilities (G006 first slice).

Per-tensor symmetric int8 quantization for K and V tensors.
Round-trip error is bounded by ~1/127 of the tensor dynamic range.
"""

from __future__ import annotations

import torch


def quantize_kv(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize K and V to int8 with a per-tensor scale.

    Returns:
        qk: int8 tensor, same shape as k.
        qv: int8 tensor, same shape as v.
        sk: scalar float tensor (per-tensor scale for k).
        sv: scalar float tensor (per-tensor scale for v).
    """
    sk = k.abs().max() / 127.0
    sv = v.abs().max() / 127.0
    sk = torch.where(sk > 0, sk, torch.ones_like(sk))
    sv = torch.where(sv > 0, sv, torch.ones_like(sv))
    qk = torch.round(k / sk).clamp(-127, 127).to(torch.int8)
    qv = torch.round(v / sv).clamp(-127, 127).to(torch.int8)
    return qk, qv, sk, sv


def dequantize_kv(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize int8 tensor back to float using a per-tensor scale."""
    return q.to(torch.float32) * scale.to(torch.float32)
