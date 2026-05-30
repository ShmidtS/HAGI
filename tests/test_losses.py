import pytest


torch = pytest.importorskip("torch")

from hagi.losses import (
    auxiliary_gdr_loss,
    cross_entropy_loss,
    isomorphic_consistency_loss,
    total_loss,
)


def test_cross_entropy_loss_matches_torch_functional():
    logits = torch.tensor([
        [[2.0, 0.5, -1.0], [0.1, 0.2, 0.3]],
        [[-0.5, 1.0, 0.0], [1.5, -0.5, 0.2]],
    ])
    targets = torch.tensor([[0, 2], [1, -100]])

    loss = cross_entropy_loss(logits, targets)
    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )

    assert torch.allclose(loss, expected)


def test_auxiliary_gdr_loss_returns_zero_without_targets():
    gdr_output = torch.randn(2, 3, 4)

    loss = auxiliary_gdr_loss(gdr_output)

    assert loss.ndim == 0
    assert loss.device == gdr_output.device
    assert torch.equal(loss, torch.tensor(0.0, device=gdr_output.device))


def test_auxiliary_gdr_loss_matches_mse_with_targets():
    gdr_output = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    grade_targets = torch.tensor([[1.5, 1.0], [2.0, 5.0]])

    loss = auxiliary_gdr_loss(gdr_output, grade_targets)
    expected = torch.nn.functional.mse_loss(gdr_output, grade_targets)

    assert torch.allclose(loss, expected)


def test_isomorphic_consistency_loss_matches_mse():
    model_output = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target_output = torch.tensor([[1.0, 1.0], [5.0, 4.0]])

    loss = isomorphic_consistency_loss(model_output, target_output)
    expected = torch.nn.functional.mse_loss(model_output, target_output)

    assert torch.allclose(loss, expected)


def test_total_loss_combines_weighted_components():
    losses = {
        "ce": torch.tensor(2.0),
        "gdr": torch.tensor(3.0),
        "iso": torch.tensor(5.0),
    }
    weights = {"ce": 1.0, "gdr": 0.5}

    loss = total_loss(losses, weights)

    assert torch.equal(loss, torch.tensor(8.5))


def test_total_loss_defaults_to_unit_weights():
    losses = {"ce": torch.tensor(2.0), "gdr": torch.tensor(3.0)}

    loss = total_loss(losses)

    assert torch.equal(loss, torch.tensor(5.0))
