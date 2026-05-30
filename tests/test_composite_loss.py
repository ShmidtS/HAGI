import pytest


torch = pytest.importorskip("torch")

from hagi.losses import composite_loss, compute_auxiliary_loss, compute_isomorphic_loss


def test_composite_loss_returns_all_components():
    logits = torch.randn(2, 3, 5)
    targets = torch.randint(0, 5, (2, 3))
    model_output = logits + 0.1
    auxiliary_output = torch.randn(4, 3)

    losses = composite_loss(logits, targets, model_output, auxiliary_output)

    assert set(losses) == {"L_CE", "L_aux", "L_iso", "L_total"}
    assert all(loss.ndim == 0 for loss in losses.values())


def test_weights_affect_total_loss_correctly():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    targets = torch.tensor([[0, 1]])
    model_output = logits + 1.0
    auxiliary_output = torch.eye(2)
    weights = {"w_ce": 2.0, "w_aux": 3.0, "w_iso": 4.0}

    losses = composite_loss(logits, targets, model_output, auxiliary_output, weights)
    expected = 2.0 * losses["L_CE"] + 3.0 * losses["L_aux"] + 4.0 * losses["L_iso"]

    assert torch.allclose(losses["L_total"], expected)


def test_auxiliary_loss_encourages_grade_separation():
    separated = torch.eye(3)
    similar = torch.ones(3, 3)

    separated_loss = compute_auxiliary_loss(separated)
    similar_loss = compute_auxiliary_loss(similar)

    assert separated_loss < similar_loss


def test_isomorphic_loss_is_non_negative():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, input_ids):
            self.calls += 1
            return input_ids.float().unsqueeze(-1) + self.calls

    model = ToyModel()
    input_ids = torch.tensor([[1, 2, 3]])
    targets = torch.tensor([[2, 3, 4]])

    loss = compute_isomorphic_loss(model, input_ids, targets, device=torch.device("cpu"))

    assert loss.ndim == 0
    assert loss >= 0
