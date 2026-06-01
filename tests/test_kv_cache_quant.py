"""Tests for KV cache quantization utilities (G006 first slice).

Per-tensor symmetric int8 quantization for K and V.
Round-trip error is bounded by 1/127 of the tensor dynamic range.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hagi.model.kv_cache import dequantize_kv, quantize_kv  # noqa: E402


def test_quantize_returns_int8_and_per_tensor_scale():
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    qk, qv, sk, sv = quantize_kv(k, v)
    assert qk.dtype == torch.int8
    assert qv.dtype == torch.int8
    assert sk.ndim == 0
    assert sv.ndim == 0
    assert torch.isfinite(sk).item()
    assert torch.isfinite(sv).item()


def test_roundtrip_error_below_five_percent():
    torch.manual_seed(0)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    qk, qv, sk, sv = quantize_kv(k, v)
    k_hat = dequantize_kv(qk, sk)
    v_hat = dequantize_kv(qv, sv)
    k_err = (k - k_hat).abs().mean() / k.abs().mean()
    v_err = (v - v_hat).abs().mean() / v.abs().mean()
    assert k_err < 0.05
    assert v_err < 0.05


def test_storage_size_reduction():
    k = torch.randn(2, 4, 128, 32, dtype=torch.float32)
    v = torch.randn(2, 4, 128, 32, dtype=torch.float32)
    qk, qv, sk, sv = quantize_kv(k, v)
    float_bytes = k.numel() * k.element_size() + v.numel() * v.element_size()
    quant_bytes = qk.numel() * qk.element_size() + qv.numel() * qv.element_size() + 2 * 4
    assert float_bytes > quant_bytes * 3.5


def test_zero_input_handled():
    k = torch.zeros(2, 4, 8, 8)
    v = torch.zeros(2, 4, 8, 8)
    qk, qv, sk, sv = quantize_kv(k, v)
    k_hat = dequantize_kv(qk, sk)
    v_hat = dequantize_kv(qv, sv)
    assert torch.isfinite(k_hat).all()
    assert torch.isfinite(v_hat).all()
