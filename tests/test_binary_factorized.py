"""Tests for BinaryFactorizedLinear prototype.

First slice of Binary Factorized MoE ultragoal (G002).
Drop-in replacement for nn.Linear with:
  Y = scale * (X @ B1) @ B2
where B1, B2 are {+1, -1} stored as int8, scale is a per-block float.
STE backward on B1, B2; standard backward on scale.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hagi.model.binary import BinaryFactorizedLinear  # noqa: E402


def test_constructor_matches_linear_signature():
    layer = BinaryFactorizedLinear(in_features=128, out_features=64, rank=16, block_size=32)
    assert layer.in_features == 128
    assert layer.out_features == 64
    assert layer.rank == 16
    assert layer.block_size == 32


def test_forward_shape_matches_dense_linear():
    dense = torch.nn.Linear(128, 64, bias=False)
    layer = BinaryFactorizedLinear(in_features=128, out_features=64, rank=64, block_size=64)
    x = torch.randn(2, 8, 128)
    y_dense = dense(x)
    y_bin = layer(x)
    assert y_bin.shape == y_dense.shape == (2, 8, 64)


def test_stored_params_are_binary_and_packed():
    layer = BinaryFactorizedLinear(in_features=64, out_features=32, rank=8, block_size=16)
    b1 = layer.b1.detach()
    b2 = layer.b2.detach()
    assert b1.shape == (64, 8)
    assert b2.shape == (8, 32)
    unique1 = torch.unique(b1)
    unique2 = torch.unique(b2)
    assert set(unique1.tolist()).issubset({-1.0, 1.0})
    assert set(unique2.tolist()).issubset({-1.0, 1.0})


def test_effective_weight_product_shape():
    layer = BinaryFactorizedLinear(in_features=64, out_features=32, rank=8, block_size=16)
    w = layer.effective_weight()
    assert w.shape == (32, 64)


def test_gradients_flow_to_b1_b2_and_scale():
    layer = BinaryFactorizedLinear(in_features=32, out_features=16, rank=4, block_size=8)
    x = torch.randn(2, 4, 32, requires_grad=True)
    target = torch.randn(2, 4, 16)
    y = layer(x)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    assert layer.b1.grad is not None
    assert layer.b2.grad is not None
    assert layer.scale.grad is not None
    assert torch.isfinite(layer.b1.grad).all()
    assert torch.isfinite(layer.b2.grad).all()
    assert torch.isfinite(layer.scale.grad).all()


def test_param_count_reduction_vs_dense():
    dense = torch.nn.Linear(1024, 1024, bias=False)
    layer = BinaryFactorizedLinear(in_features=1024, out_features=1024, rank=64, block_size=128)
    dense_params = sum(p.numel() for p in dense.parameters())
    bin_params = sum(p.numel() for p in layer.parameters())
    assert bin_params < dense_params


def test_input_block_size_must_divide_in_features():
    with pytest.raises(ValueError):
        BinaryFactorizedLinear(in_features=64, out_features=32, rank=8, block_size=10)


def test_forward_finite_values():
    layer = BinaryFactorizedLinear(in_features=32, out_features=16, rank=4, block_size=8)
    x = torch.randn(4, 8, 32)
    y = layer(x)
    assert torch.isfinite(y).all()
