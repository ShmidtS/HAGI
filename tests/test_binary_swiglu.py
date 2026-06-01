"""Tests for BinarySwiGLU (G003).

Replaces SwiGLU's gate/up/down nn.Linear with BinaryFactorizedLinear.
Same forward semantics: y = down(silu(gate(x)) * up(x)).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from hagi.model.binary import BinarySwiGLU  # noqa: E402
from hagi.model.transformer import SwiGLU, TransformerConfig  # noqa: E402


@pytest.fixture
def cfg():
    return TransformerConfig(
        hidden_size=64,
        num_query_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        max_seq_len=16,
    )


def test_constructor_creates_three_factorized_layers(cfg):
    layer = BinarySwiGLU(cfg, rank=8, block_size=32)
    assert layer.gate.in_features == 64
    assert layer.gate.out_features == 128
    assert layer.up.in_features == 64
    assert layer.up.out_features == 128
    assert layer.down.in_features == 128
    assert layer.down.out_features == 64


def test_forward_shape_matches_swiglu(cfg):
    dense = SwiGLU(cfg)
    layer = BinarySwiGLU(cfg, rank=16, block_size=32)
    x = torch.randn(2, 8, cfg.hidden_size)
    y_dense = dense(x)
    y_bin = layer(x)
    assert y_bin.shape == y_dense.shape == (2, 8, cfg.hidden_size)


def test_forward_finite_values(cfg):
    layer = BinarySwiGLU(cfg, rank=8, block_size=32)
    x = torch.randn(4, 16, cfg.hidden_size)
    y = layer(x)
    assert torch.isfinite(y).all()


def test_gradients_flow_to_all_three_factorized_layers(cfg):
    layer = BinarySwiGLU(cfg, rank=4, block_size=32)
    x = torch.randn(2, 4, cfg.hidden_size, requires_grad=True)
    target = torch.randn(2, 4, cfg.hidden_size)
    y = layer(x)
    loss = F.mse_loss(y, target)
    loss.backward()
    for sub_name, sub in [("gate", layer.gate), ("up", layer.up), ("down", layer.down)]:
        assert sub.b1.grad is not None, f"{sub_name}.b1 grad missing"
        assert sub.b2.grad is not None, f"{sub_name}.b2 grad missing"
        assert sub.scale.grad is not None, f"{sub_name}.scale grad missing"
        assert torch.isfinite(sub.b1.grad).all()
        assert torch.isfinite(sub.b2.grad).all()
        assert torch.isfinite(sub.scale.grad).all()


def test_param_count_reduction_vs_dense_swiglu(cfg):
    dense = SwiGLU(cfg)
    layer = BinarySwiGLU(cfg, rank=8, block_size=32)
    dense_params = sum(p.numel() for p in dense.parameters())
    bin_params = sum(p.numel() for p in layer.parameters())
    assert bin_params < dense_params
