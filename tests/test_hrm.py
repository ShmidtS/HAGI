import pytest


torch = pytest.importorskip("torch")

from hagi.model import HRMCore, HState, HTransition, LState, LTransition, TransformerConfig


def test_hstate_creation_and_shape():
    state = HState(torch.zeros(2, 16))

    assert state.z_H.shape == (2, 16)


def test_lstate_creation_and_shape():
    state = LState(torch.zeros(2, 12))

    assert state.z_L.shape == (2, 12)


def test_htransition_update_preserves_shape():
    transition = HTransition(h_dim=16, l_dim=12)
    z_H = torch.randn(2, 16)
    z_L = torch.randn(2, 12)

    updated = transition(z_H, z_L)

    assert updated.shape == z_H.shape


def test_ltransition_update_preserves_shape():
    transition = LTransition(l_dim=12, hidden_size=32, h_dim=16)
    z_L = torch.randn(2, 12)
    transformer_output = torch.randn(2, 5, 32)

    updated = transition(z_L, transformer_output)

    assert updated.shape == z_L.shape


# Removed legacy tests that instantiated HRMCore as a standalone model
# (vocab_size, transformer, logits).  Current HRMCore is a recurrence
# controller consumed by HAGI; full forward+state behaviour is covered by
# test_model_variants.py.
