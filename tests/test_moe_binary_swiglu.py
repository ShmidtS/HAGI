"""Tests for MoE wrapper around BinarySwiGLU experts (G004).

Top-k routing with softmax gating. Load-balancing aux loss
(switch-transformer style) reported via MoEOutput.aux_loss.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hagi.model.binary import BinarySwiGLU  # noqa: E402
from hagi.model.moe import MoEBinarySwiGLU, MoEOutput  # noqa: E402
from hagi.model.transformer import TransformerConfig  # noqa: E402


@pytest.fixture
def cfg():
    return TransformerConfig(
        hidden_size=64,
        num_query_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        max_seq_len=16,
    )


def test_constructor_creates_router_and_experts(cfg):
    moe = MoEBinarySwiGLU(cfg, num_experts=4, top_k=2, rank=4, block_size=32)
    assert moe.num_experts == 4
    assert moe.top_k == 2
    assert len(moe.experts) == 4
    assert all(isinstance(e, BinarySwiGLU) for e in moe.experts)
    assert moe.router.weight.shape == (4, 64)


def test_forward_returns_moe_output_with_shape(cfg):
    moe = MoEBinarySwiGLU(cfg, num_experts=4, top_k=2, rank=4, block_size=32)
    x = torch.randn(2, 8, cfg.hidden_size)
    out = moe(x)
    assert isinstance(out, MoEOutput)
    assert out.y.shape == (2, 8, cfg.hidden_size)
    assert torch.isfinite(out.y).all()
    assert torch.isfinite(out.aux_loss)


def test_top_k_routing_uses_at_most_k_experts_per_token(cfg):
    moe = MoEBinarySwiGLU(cfg, num_experts=4, top_k=1, rank=4, block_size=32)
    x = torch.randn(1, 4, cfg.hidden_size)
    router_logits = moe.router(x)  # (1, 4, num_experts)
    top1 = router_logits.argmax(dim=-1)
    assert top1.shape == (1, 4)
    assert int(top1.max()) < 4
    assert int(top1.min()) >= 0


def test_aux_loss_decreases_with_balanced_routing(cfg, monkeypatch):
    moe = MoEBinarySwiGLU(cfg, num_experts=4, top_k=1, rank=4, block_size=32)
    moe.eval()
    x = torch.randn(8, 16, cfg.hidden_size)
    with torch.no_grad():
        out = moe(x)
    balanced_loss = out.aux_loss
    # Force a degenerate routing and observe aux loss rises.
    with torch.no_grad():
        degenerate_w = torch.zeros(4, cfg.hidden_size)
        degenerate_w[0] = 1.0
        moe.router.weight.copy_(degenerate_w)
        out2 = moe(x)
    degenerate_loss = out2.aux_loss
    assert degenerate_loss > balanced_loss


def test_gradients_flow_to_router_and_experts(cfg):
    moe = MoEBinarySwiGLU(cfg, num_experts=4, top_k=2, rank=4, block_size=32)
    x = torch.randn(2, 4, cfg.hidden_size, requires_grad=True)
    target = torch.randn(2, 4, cfg.hidden_size)
    out = moe(x)
    loss = torch.nn.functional.mse_loss(out.y, target) + out.aux_loss
    loss.backward()
    assert moe.router.weight.grad is not None
    assert torch.isfinite(moe.router.weight.grad).all()
    for i, expert in enumerate(moe.experts):
        for sub in (expert.gate, expert.up, expert.down):
            assert sub.b1.grad is not None, f"expert{i} b1 grad missing"
            assert sub.b2.grad is not None
            assert sub.scale.grad is not None
            assert torch.isfinite(sub.b1.grad).all()
