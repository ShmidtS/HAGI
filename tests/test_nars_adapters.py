"""Tests for NARS adapters bridging HAGI components to OpenNARS-style reasoning.

Covers:
- NarsHrmController observe_train_step + resolve_policy
- NarsHdimReasoner recommend_transfer + observe_transfer_feedback
- NarsMsaReasoner route_top_k_with_nars + observe_route_feedback
- HrmControlPolicy dataclass
- Budget decay and truth revision integration
- Recency weights update correctness
"""

import pytest

torch = pytest.importorskip("torch")

from hagi.nars.adapters import (
    HrmControlPolicy,
    NarsHdimReasoner,
    NarsHrmController,
    NarsMsaReasoner,
)
from hagi.nars.budget import BudgetValue
from hagi.nars.truth import TruthValue


# ---------------------------------------------------------------------------
# HrmControlPolicy
# ---------------------------------------------------------------------------

def test_hrm_control_policy_dataclass():
    policy = HrmControlPolicy(h_cycles=3, l_cycles=2, convergence_eps=1e-6, bp_steps=4)
    assert policy.h_cycles == 3
    assert policy.l_cycles == 2
    assert policy.convergence_eps == 1e-6
    assert policy.bp_steps == 4
    assert hash(policy) is not None  # frozen=True, slots=True


# ---------------------------------------------------------------------------
# NarsHrmController
# ---------------------------------------------------------------------------

def test_nars_hrm_controller_observe_train_step_revises_truths():
    ctrl = NarsHrmController()
    ctrl.observe_train_step(loss=0.5, grad_norm=1.0)
    assert "loss_low" in ctrl.control_truths
    assert "grad_stable" in ctrl.control_truths
    assert ctrl.control_truths["loss_low"].frequency > 0.0
    assert ctrl.control_truths["grad_stable"].confidence > 0.0


def test_nars_hrm_controller_budget_decay_on_observation():
    ctrl = NarsHrmController()
    ctrl.observe_train_step(loss=0.1, grad_norm=0.1)
    first_budget = ctrl.control_budgets["loss_low"]
    ctrl.observe_train_step(loss=0.2, grad_norm=0.2)
    second_budget = ctrl.control_budgets["loss_low"]
    # Budget should have been decayed then merged, so values may differ.
    # We just assert that the budget is a valid BudgetValue.
    assert isinstance(second_budget, BudgetValue)
    assert 0.0 <= second_budget.priority <= 1.0


def test_nars_hrm_controller_resolve_policy_fallback_when_empty():
    ctrl = NarsHrmController()
    ctrl.observe_train_step(loss=1.0, grad_norm=1.0)
    policy = ctrl.resolve_policy()
    assert isinstance(policy, HrmControlPolicy)
    assert policy.h_cycles >= 1
    assert policy.l_cycles >= 1


def test_nars_hrm_controller_resolve_policy_selects_from_bag():
    ctrl = NarsHrmController(policy_capacity=5)
    for _ in range(10):
        ctrl.observe_train_step(loss=0.1, grad_norm=0.1)
    policy = ctrl.resolve_policy()
    assert isinstance(policy, HrmControlPolicy)
    # Bag should not exceed capacity
    assert len(ctrl._policy_bag) <= 5


def test_nars_hrm_controller_apply_policy_to_dict():
    ctrl = NarsHrmController()
    ctrl.observe_train_step(loss=0.5, grad_norm=0.5)
    policy = ctrl.resolve_policy()
    config = {}
    ctrl.apply_policy(policy, config)
    assert config["h_cycles"] == policy.h_cycles
    assert config["l_cycles"] == policy.l_cycles
    assert config["convergence_eps"] == policy.convergence_eps
    assert config["bp_steps"] == policy.bp_steps


def test_nars_hrm_controller_apply_policy_to_object():
    class DummyConfig:
        h_cycles = 0
        l_cycles = 0
        convergence_eps = 0.0
        bp_steps = 0

    ctrl = NarsHrmController()
    ctrl.observe_train_step(loss=0.5, grad_norm=0.5)
    policy = ctrl.resolve_policy()
    cfg = DummyConfig()
    ctrl.apply_policy(policy, cfg)
    assert cfg.h_cycles == policy.h_cycles
    assert cfg.l_cycles == policy.l_cycles
    assert cfg.convergence_eps == policy.convergence_eps
    assert cfg.bp_steps == policy.bp_steps


def test_nars_hrm_controller_policy_bag_capacity_respected():
    ctrl = NarsHrmController(policy_capacity=2)
    for i in range(5):
        ctrl.observe_train_step(loss=float(i), grad_norm=float(i))
    assert len(ctrl._policy_bag) <= 2


# ---------------------------------------------------------------------------
# NarsHdimReasoner
# ---------------------------------------------------------------------------

def test_nars_hdim_reasoner_recommend_transfer_selects_best():
    reasoner = NarsHdimReasoner()
    reasoner.transfer_beliefs[(0, 1)] = TruthValue(0.9, 0.9)
    reasoner.transfer_beliefs[(0, 2)] = TruthValue(0.5, 0.5)
    target = reasoner.recommend_transfer(source=0, known_targets=[1, 2])
    assert target == 1


def test_nars_hdim_reasoner_recommend_transfer_empty_raises():
    reasoner = NarsHdimReasoner()
    with pytest.raises(ValueError, match="known_targets must not be empty"):
        reasoner.recommend_transfer(0, [])


def test_nars_hdim_reasoner_observe_transfer_feedback_revises_belief():
    reasoner = NarsHdimReasoner()
    reasoner.observe_transfer_feedback(0, 1, fidelity=0.8)
    belief = reasoner.transfer_beliefs[(0, 1)]
    assert belief.frequency > 0.5
    assert belief.confidence > 0.0


def test_nars_hdim_reasoner_feedback_revises_existing():
    reasoner = NarsHdimReasoner()
    reasoner.observe_transfer_feedback(0, 1, fidelity=0.9)
    old = reasoner.transfer_beliefs[(0, 1)]
    reasoner.observe_transfer_feedback(0, 1, fidelity=0.1)
    new = reasoner.transfer_beliefs[(0, 1)]
    # After revision with low fidelity, frequency should drop
    assert new.frequency < old.frequency


def test_nars_hdim_reasoner_transfer_domain_fallback():
    class FakeRegistry:
        num_rotors = 3

        def value(self, idx):
            return f"rotor_{idx}"

    reasoner = NarsHdimReasoner()
    reasoner.observe_transfer_feedback(0, 1, fidelity=0.9)
    target, rotor = reasoner.transfer_domain_reasoned_or_fallback(FakeRegistry(), 0, target_hint=2)
    assert target == 1
    assert rotor == "rotor_1"


def test_nars_hdim_reasoner_transfer_domain_fallback_when_weak():
    class FakeRegistry:
        num_rotors = 3

    reasoner = NarsHdimReasoner()
    # No feedback means weak belief; should fallback to target_hint
    target, rotor = reasoner.transfer_domain_reasoned_or_fallback(FakeRegistry(), 0, target_hint=2)
    assert target == 2
    assert rotor is None


# ---------------------------------------------------------------------------
# NarsMsaReasoner
# ---------------------------------------------------------------------------

def test_nars_msa_reasoner_route_top_k_with_nars_returns_tensors():
    class MockRegistry:
        def keys_tensor(self, device=None):
            return torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float32)

        def slot_ids(self):
            return [10, 20, 30]

    reasoner = NarsMsaReasoner()
    reasoner.slot_beliefs[10] = TruthValue(1.0, 1.0)
    reasoner.slot_beliefs[20] = TruthValue(0.0, 0.0)
    reasoner.slot_beliefs[30] = TruthValue(0.5, 0.5)

    query = torch.tensor([1.0, 0.0], dtype=torch.float32)
    top_ids, top_vals = reasoner.route_top_k_with_nars(MockRegistry(), query, top_k=2)
    assert top_ids.shape == (2,)
    assert top_vals.shape == (2,)
    assert top_ids.dtype == torch.long


def test_nars_msa_reasoner_route_top_k_requires_keys_tensor():
    class BadRegistry:
        def slot_ids(self):
            return [1]

    reasoner = NarsMsaReasoner()
    with pytest.raises(TypeError, match="keys_tensor"):
        reasoner.route_top_k_with_nars(BadRegistry(), torch.tensor([1.0]), top_k=1)


def test_nars_msa_reasoner_route_top_k_requires_slot_ids():
    class BadRegistry:
        def keys_tensor(self, device=None):
            return torch.tensor([[1.0]])

    reasoner = NarsMsaReasoner()
    with pytest.raises(TypeError, match="slot_ids"):
        reasoner.route_top_k_with_nars(BadRegistry(), torch.tensor([1.0]), top_k=1)


def test_nars_msa_reasoner_route_top_k_dimension_mismatch_raises():
    class MockRegistry:
        def keys_tensor(self, device=None):
            return torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        def slot_ids(self):
            return [0]

    reasoner = NarsMsaReasoner()
    query = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)  # wrong dim
    with pytest.raises(ValueError, match="Query dim"):
        reasoner.route_top_k_with_nars(MockRegistry(), query, top_k=1)


def test_nars_msa_reasoner_observe_route_feedback_updates_truth_and_budget():
    reasoner = NarsMsaReasoner()
    reasoner.observe_route_feedback(slot_id=1, usefulness=0.8)
    assert 1 in reasoner.slot_beliefs
    assert reasoner.slot_beliefs[1].frequency > 0.5
    assert 1 in reasoner.slot_budgets
    assert reasoner.slot_budgets[1].priority > 0.5


def test_nars_msa_reasoner_recency_weights_decay_and_bump():
    reasoner = NarsMsaReasoner()
    reasoner.observe_route_feedback(1, 0.5)
    reasoner.observe_route_feedback(2, 0.6)
    # After second feedback, slot 1 should be decayed, slot 2 bumped to 1.0
    assert reasoner.recency_weights[1] == pytest.approx(0.99, abs=1e-6)
    assert reasoner.recency_weights[2] == pytest.approx(1.0, abs=1e-6)


def test_nars_msa_reasoner_route_top_k_uses_recency_weights():
    class MockRegistry:
        def keys_tensor(self, device=None):
            return torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        def slot_ids(self):
            return [0, 1]

    reasoner = NarsMsaReasoner()
    reasoner.recency_weights[0] = 0.0
    reasoner.recency_weights[1] = 1.0
    query = torch.tensor([0.5, 0.5], dtype=torch.float32)
    top_ids, _ = reasoner.route_top_k_with_nars(MockRegistry(), query, top_k=1, recency_weight=1.0, nars_weight=0.0, dot_weight=0.0)
    assert int(top_ids[0].item()) == 1


def test_nars_msa_reasoner_route_top_k_clamps_to_registry_size():
    class MockRegistry:
        def keys_tensor(self, device=None):
            return torch.tensor([[1.0]], dtype=torch.float32)

        def slot_ids(self):
            return [42]

    reasoner = NarsMsaReasoner()
    query = torch.tensor([1.0], dtype=torch.float32)
    top_ids, top_vals = reasoner.route_top_k_with_nars(MockRegistry(), query, top_k=10)
    assert top_ids.shape == (1,)
    assert int(top_ids[0].item()) == 42


# ---------------------------------------------------------------------------
# Budget decay / truth revision integration
# ---------------------------------------------------------------------------

def test_nars_hrm_controller_truth_revision_on_repeated_observations():
    ctrl = NarsHrmController()
    ctrl.observe_train_step(loss=0.1, grad_norm=0.1)
    first = ctrl.control_truths["loss_low"].frequency
    ctrl.observe_train_step(loss=0.1, grad_norm=0.1)
    second = ctrl.control_truths["loss_low"].frequency
    # Repeated same observation should converge frequency upward
    assert second > first


def test_nars_hrm_controller_truth_revision_dampened_by_high_loss():
    ctrl = NarsHrmController()
    ctrl.observe_train_step(loss=0.1, grad_norm=0.1)
    high_loss_freq = ctrl.control_truths["loss_low"].frequency
    ctrl.observe_train_step(loss=10.0, grad_norm=10.0)
    low_loss_freq = ctrl.control_truths["loss_low"].frequency
    # High loss should drag frequency down
    assert low_loss_freq < high_loss_freq


# ---------------------------------------------------------------------------
# Recency weights update correctness
# ---------------------------------------------------------------------------

def test_nars_msa_reasoner_recency_decay_all_slots():
    reasoner = NarsMsaReasoner()
    for sid in range(5):
        reasoner.observe_route_feedback(sid, 0.5)
    # After 5th feedback, all earlier slots should be decayed
    assert reasoner.recency_weights[0] == pytest.approx(0.99 ** 4, abs=1e-6)
    assert reasoner.recency_weights[4] == pytest.approx(1.0, abs=1e-6)
