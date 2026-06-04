import pytest


torch = pytest.importorskip("torch")

from hagi.losses import compute_isomorphic_loss
from hagi.model.hdim_full import DelayedHDIM


def _make_model(delay_steps: int = 2) -> DelayedHDIM:
    return DelayedHDIM(hidden_size=16, heads=2, num_rotors=2, delay_steps=delay_steps)


def test_delayed_hdim_identity_for_first_delta_minus_one_steps():
    model = _make_model(delay_steps=2)
    model.train()
    hidden = torch.randn(2, 5, 16)

    out0 = model(hidden, delay_step=0)
    out1 = model(hidden, delay_step=1)

    assert torch.allclose(out0, hidden)
    assert not torch.allclose(out1, hidden)


def test_delayed_hdim_aggregates_on_step_delta():
    model = _make_model(delay_steps=3)
    model.train()
    hidden = torch.randn(2, 5, 16)

    out0 = model(hidden, delay_step=0, return_state=True)
    out1 = model(hidden, delay_step=1, return_state=True)
    out2 = model(hidden, delay_step=2, return_state=True)

    assert torch.allclose(out0["fused"], hidden)
    assert torch.allclose(out1["fused"], hidden)
    assert not torch.allclose(out2["fused"], hidden)
    assert out2["invariant"] is not None
    assert out2["invariant"].abs().max() > 0


def test_delayed_hdim_cycles_correctly():
    model = _make_model(delay_steps=2)
    model.train()
    hidden = torch.randn(2, 5, 16)

    out0 = model(hidden, delay_step=0)
    out1 = model(hidden, delay_step=1)
    out2 = model(hidden, delay_step=2)
    out3 = model(hidden, delay_step=3)

    assert torch.allclose(out0, hidden)
    assert not torch.allclose(out1, hidden)
    assert torch.allclose(out2, hidden)
    assert not torch.allclose(out3, hidden)


def test_delayed_hdim_falls_back_to_normal_when_eval():
    model = _make_model(delay_steps=2)
    model.eval()
    hidden = torch.randn(2, 5, 16)

    out = model(hidden, delay_step=0)

    assert not torch.allclose(out, hidden)
    assert out.shape == hidden.shape


def test_delayed_hdim_falls_back_when_delay_steps_is_one():
    model = DelayedHDIM(hidden_size=16, heads=2, num_rotors=2, delay_steps=1)
    model.train()
    hidden = torch.randn(2, 5, 16)

    out = model(hidden, delay_step=0)

    assert not torch.allclose(out, hidden)
    assert out.shape == hidden.shape


def test_delayed_hdim_loss_components_nonzero_on_aggregation():
    model = _make_model(delay_steps=2)
    model.train()
    hidden = torch.randn(2, 5, 16)

    # First step (identity): no aggregation.
    state0 = model(hidden, delay_step=0, return_state=True)
    l_iso0 = compute_isomorphic_loss(hidden, state0["fused"])
    assert l_iso0.item() == 0.0

    # Second step (aggregation): fused should differ.
    state1 = model(hidden, delay_step=1, return_state=True)
    l_iso1 = compute_isomorphic_loss(hidden, state1["fused"])
    assert l_iso1.item() > 0.0

    # L_aux without labels is now 0 (no L2 regularization to avoid killing gradients)
    from hagi.losses import compute_auxiliary_loss
    l_aux0 = compute_auxiliary_loss(state0["fused"])
    l_aux1 = compute_auxiliary_loss(state1["fused"])
    assert l_aux0.item() == 0.0
    assert l_aux1.item() == 0.0


def test_delayed_hdim_last_step_aggregates_remaining_buffer():
    """If total_steps is not divisible by delay_steps, the last step should flush."""
    model = _make_model(delay_steps=3)
    model.train()
    hidden = torch.randn(2, 5, 16)

    out0 = model(hidden, delay_step=0, total_steps=5)
    out1 = model(hidden, delay_step=1, total_steps=5)
    out2 = model(hidden, delay_step=2, total_steps=5)
    out3 = model(hidden, delay_step=3, total_steps=5)
    out4 = model(hidden, delay_step=4, total_steps=5)

    assert torch.allclose(out0, hidden)
    assert torch.allclose(out1, hidden)
    assert not torch.allclose(out2, hidden)   # buffer full (3 items)
    assert torch.allclose(out3, hidden)       # new buffer starts
    assert not torch.allclose(out4, hidden)  # flush remaining 2 items on last step
