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


def test_hrmcore_forward_2h_3l_logits_shape():
    cfg = TransformerConfig(
        hidden_size=32,
        num_query_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_seq_len=8,
    )
    model = HRMCore(
        vocab_size=41,
        hidden_size=32,
        h_dim=16,
        l_dim=12,
        h_cycles=2,
        l_cycles=3,
        transformer=cfg,
    )
    input_ids = torch.randint(0, 41, (2, 5))

    logits, z_H, z_L = model(input_ids)

    assert logits.shape == (2, 5, 41)
    assert z_H.z_H.shape == (2, 16)
    assert z_L.z_L.shape == (2, 12)


def test_hrmcore_states_are_updated_after_forward():
    cfg = TransformerConfig(
        hidden_size=32,
        num_query_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_seq_len=8,
    )
    model = HRMCore(
        vocab_size=41,
        hidden_size=32,
        h_dim=16,
        l_dim=12,
        h_cycles=2,
        l_cycles=3,
        transformer=cfg,
    )
    input_ids = torch.randint(0, 41, (2, 5))
    z_H = torch.zeros(2, 16)
    z_L = torch.zeros(2, 12)

    _, updated_h, updated_l = model(input_ids, z_H=z_H, z_L=z_L)

    assert not torch.allclose(updated_h.z_H, z_H)
    assert not torch.allclose(updated_l.z_L, z_L)
