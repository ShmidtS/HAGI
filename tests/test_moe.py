import torch
import pytest
from hagi.model.moe import MoESwiGLU
from hagi.model.transformer import TransformerConfig


def test_moe_shape():
    cfg = TransformerConfig(
        hidden_size=64,
        num_query_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        use_moe=True,
        num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=32,
    )
    moe = MoESwiGLU(cfg)
    x = torch.randn(2, 8, 64)
    out = moe(x)
    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == (2, 8, 64)


def test_moe_top_k():
    cfg = TransformerConfig(
        hidden_size=64,
        num_query_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        use_moe=True,
        num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=32,
    )
    moe = MoESwiGLU(cfg)
    x = torch.randn(2, 8, 64)
    moe.train()
    out, aux_loss = moe(x)
    assert out.shape == (2, 8, 64)
    assert aux_loss.ndim == 0
    assert aux_loss.item() >= 0


def test_moe_load_balancing_loss_nonzero():
    cfg = TransformerConfig(
        hidden_size=64,
        num_query_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        use_moe=True,
        num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=32,
    )
    moe = MoESwiGLU(cfg)
    x = torch.randn(2, 8, 64)
    moe.train()
    _, aux_loss = moe(x)
    assert aux_loss.item() > 0


def test_moe_backward():
    cfg = TransformerConfig(
        hidden_size=64,
        num_query_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        use_moe=True,
        num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=32,
    )
    moe = MoESwiGLU(cfg)
    x = torch.randn(2, 8, 64, requires_grad=True)
    moe.train()
    out, aux_loss = moe(x)
    loss = out.sum() + aux_loss
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    for param in moe.parameters():
        assert param.grad is not None


def test_moe_inference():
    cfg = TransformerConfig(
        hidden_size=64,
        num_query_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        use_moe=True,
        num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=32,
    )
    moe = MoESwiGLU(cfg)
    x = torch.randn(2, 8, 64)
    moe.eval()
    out = moe(x)
    assert not isinstance(out, tuple)
    assert out.shape == (2, 8, 64)


def test_moe_with_binary_factorized():
    cfg = TransformerConfig(
        hidden_size=64,
        num_query_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        use_moe=True,
        num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=32,
        use_binary_factorized=True,
        binary_factorized_rank=4,
    )
    moe = MoESwiGLU(cfg)
    x = torch.randn(2, 8, 64)
    moe.train()
    out, aux_loss = moe(x)
    assert out.shape == (2, 8, 64)
    assert aux_loss.ndim == 0


def test_moe_auto_intermediate_size():
    cfg = TransformerConfig(
        hidden_size=64,
        num_query_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        use_moe=True,
        num_experts=8,
        moe_top_k=2,
        moe_intermediate_size=None,
    )
    moe = MoESwiGLU(cfg)
    assert moe.intermediate_size == 256 // 8
    x = torch.randn(2, 8, 64)
    moe.train()
    out, _ = moe(x)
    assert out.shape == (2, 8, 64)
