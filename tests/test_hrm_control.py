import numpy as np
import pytest

from hagi.hrm import HRMController
from hagi.nars import BudgetValue, Concept, Term


def test_budget_routing_is_deterministic_with_numpy():
    concepts = [
        Concept(Term.Atom("low"), budget=BudgetValue(0.1, 0.5, 0.5)),
        Concept(Term.Atom("high"), budget=BudgetValue(0.9, 0.5, 0.5)),
    ]
    hidden_states = np.ones((2, 3), dtype=np.float32)
    controller = HRMController()

    routed_once = controller.route(concepts, hidden_states)
    routed_twice = controller.route(concepts, hidden_states)
    priorities = np.array([0.1, 0.9], dtype=np.float64)
    expected_weights = np.exp(priorities - priorities.max())
    expected_weights = expected_weights / expected_weights.sum()

    assert np.allclose(routed_once, routed_twice)
    assert np.allclose(routed_once, hidden_states * expected_weights[:, None])
    assert routed_once[1, 0] > routed_once[0, 0]


def test_budget_routing_is_deterministic_with_torch_if_available():
    torch = pytest.importorskip("torch")
    concepts = [
        Concept(Term.Atom("low"), budget=BudgetValue(0.1, 0.5, 0.5)),
        Concept(Term.Atom("high"), budget=BudgetValue(0.9, 0.5, 0.5)),
    ]
    hidden_states = torch.ones((2, 3), dtype=torch.float32)
    controller = HRMController()

    routed_once = controller.route(concepts, hidden_states)
    routed_twice = controller.route(concepts, hidden_states)
    expected_weights = torch.softmax(torch.tensor([0.1, 0.9]), dim=0)

    assert torch.allclose(routed_once, routed_twice)
    assert torch.allclose(routed_once, hidden_states * expected_weights[:, None])
