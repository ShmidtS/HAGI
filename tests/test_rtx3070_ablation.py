"""RTX 3070 ablation tests (G007 first slice).

Param-count and memory estimates for binary primitives at HAGI
RTX 3070 canonical hidden/intermediate sizes. No GPU benchmark
(unavailable in CI); the figures are derived from param counts.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hagi.model.binary import BinaryFactorizedLinear, BinarySwiGLU  # noqa: E402
from hagi.model.moe import MoEBinarySwiGLU  # noqa: E402
from hagi.model.transformer import SwiGLU, TransformerConfig  # noqa: E402


RTX3070_HIDDEN = 512
RTX3070_INTERMEDIATE = 2048


def _bytes_fp32(numel: int) -> int:
    return numel * 4


def test_binary_linear_param_reduction_at_rtx3070_dims():
    dense = torch.nn.Linear(RTX3070_HIDDEN, RTX3070_INTERMEDIATE, bias=False)
    layer = BinaryFactorizedLinear(
        RTX3070_HIDDEN, RTX3070_INTERMEDIATE, rank=64, block_size=128
    )
    dense_params = sum(p.numel() for p in dense.parameters())
    bin_params = sum(p.numel() for p in layer.parameters())
    assert bin_params < dense_params
    assert bin_params < dense_params * 0.2  # at least 5x reduction


def test_binary_swiglu_param_reduction_at_rtx3070_dims():
    cfg = TransformerConfig(
        hidden_size=RTX3070_HIDDEN,
        num_query_heads=8,
        num_kv_heads=4,
        intermediate_size=RTX3070_INTERMEDIATE,
        max_seq_len=512,
    )
    dense = SwiGLU(cfg)
    layer = BinarySwiGLU(cfg, rank=64, block_size=128)
    dense_params = sum(p.numel() for p in dense.parameters())
    bin_params = sum(p.numel() for p in layer.parameters())
    assert bin_params < dense_params
    assert bin_params < dense_params * 0.3


def test_moe_param_count_scales_linearly_with_experts():
    cfg = TransformerConfig(
        hidden_size=RTX3070_HIDDEN,
        num_query_heads=8,
        num_kv_heads=4,
        intermediate_size=RTX3070_INTERMEDIATE,
        max_seq_len=512,
    )
    moe4 = MoEBinarySwiGLU(cfg, num_experts=4, top_k=2, rank=64, block_size=128)
    moe8 = MoEBinarySwiGLU(cfg, num_experts=8, top_k=2, rank=64, block_size=128)
    p4 = sum(p.numel() for p in moe4.parameters())
    p8 = sum(p.numel() for p in moe8.parameters())
    # 8 experts ≈ 2x params of 4 experts (router is small).
    assert p8 > p4 * 1.7


def test_estimated_inference_memory_smaller_than_dense():
    """At 4 experts, 2 active per token, binary MoE needs ~half the
    weight memory of the equivalent dense SwiGLU stack."""
    cfg = TransformerConfig(
        hidden_size=RTX3070_HIDDEN,
        num_query_heads=8,
        num_kv_heads=4,
        intermediate_size=RTX3070_INTERMEDIATE,
        max_seq_len=512,
    )
    dense = SwiGLU(cfg)
    moe = MoEBinarySwiGLU(cfg, num_experts=4, top_k=2, rank=64, block_size=128)
    dense_bytes = sum(_bytes_fp32(p.numel()) for p in dense.parameters())
    moe_bytes = sum(_bytes_fp32(p.numel()) for p in moe.parameters())
    # 4 experts, each ~5x smaller than dense SwiGLU → moe_bytes < dense_bytes
    assert moe_bytes < dense_bytes
