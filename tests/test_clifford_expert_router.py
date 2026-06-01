"""Tests for CliffordExpertRouter (G005).

Replaces dense nn.Linear router in MoE with a rotor-based geometric
scoring head. Output shape is identical to nn.Linear(hidden, num_experts).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hagi.model.clifford_router import CliffordExpertRouter  # noqa: E402


def test_constructor_stores_expected_dims():
    router = CliffordExpertRouter(hidden_size=64, num_experts=4, heads=1)
    assert router.hidden_size == 64
    assert router.num_experts == 4
    assert router.blade_count == 8


def test_forward_shape_matches_linear_router():
    router = CliffordExpertRouter(hidden_size=64, num_experts=4, heads=1)
    x = torch.randn(2, 8, 64)
    scores = router(x)
    assert scores.shape == (2, 8, 4)


def test_different_inputs_yield_different_scores():
    router = CliffordExpertRouter(hidden_size=64, num_experts=4, heads=1)
    x1 = torch.randn(2, 8, 64)
    x2 = torch.randn(2, 8, 64) + 5.0
    s1 = router(x1)
    s2 = router(x2)
    assert not torch.allclose(s1, s2)


def test_gradients_flow_to_projection_and_rotor_params():
    router = CliffordExpertRouter(hidden_size=32, num_experts=3, heads=1)
    x = torch.randn(2, 4, 32, requires_grad=True)
    target = torch.randn(2, 4, 3)
    scores = router(x)
    loss = torch.nn.functional.mse_loss(scores, target)
    loss.backward()
    assert router.to_mv.proj.weight.grad is not None
    assert router.rotor.rotor_params.grad is not None
    assert torch.isfinite(router.to_mv.proj.weight.grad).all()
    assert torch.isfinite(router.rotor.rotor_params.grad).all()


def test_forward_finite_values():
    router = CliffordExpertRouter(hidden_size=64, num_experts=4, heads=1)
    x = torch.randn(4, 16, 64)
    scores = router(x)
    assert torch.isfinite(scores).all()
