import pytest


torch = pytest.importorskip("torch")

from hagi.model.clifford import BLADE_COUNT, grade_projection
from hagi.model.hdim_full import (
    DomainRotor,
    DomainTransfer,
    GatedFusion,
    HDIMFull,
    HiddenToMultivector,
    InvariantExtractor,
)


def test_hidden_to_multivector_produces_correct_shape():
    module = HiddenToMultivector(hidden_size=16, heads=3)
    hidden = torch.randn(2, 5, 16)

    out = module(hidden)

    assert out.shape == (2, 5, 3, BLADE_COUNT)


def test_domain_rotor_preserves_multivector_grade_structure_for_identity_rotor():
    rotor = DomainRotor(num_rotors=1, heads=2)
    multivector = torch.randn(2, 4, 2, BLADE_COUNT)

    out = rotor(multivector, 0)

    for grade in range(4):
        assert torch.allclose(grade_projection(out, grade), grade_projection(multivector, grade), atol=1e-6)


def test_invariant_extractor_produces_invariant_under_source_rotation():
    rotor = DomainRotor(num_rotors=1, heads=2)
    extractor = InvariantExtractor()
    invariant = torch.randn(2, 4, 2, BLADE_COUNT)
    rotated = rotor(invariant, 0)

    out = extractor(rotated, rotor, 0)

    assert torch.allclose(out, invariant, atol=1e-5)


def test_domain_transfer_transfers_to_target_domain_correctly():
    rotor = DomainRotor(num_rotors=1, heads=2)
    transfer = DomainTransfer()
    invariant = torch.randn(2, 4, 2, BLADE_COUNT)

    out = transfer(invariant, rotor, 0)
    expected = rotor(invariant, 0)

    assert torch.allclose(out, expected, atol=1e-6)


def test_gated_fusion_output_shape_matches_hidden_shape():
    fusion = GatedFusion(hidden_size=16, heads=2)
    transformed = torch.randn(2, 5, 2, BLADE_COUNT)
    hidden = torch.randn(2, 5, 16)

    out = fusion(transformed, hidden)

    assert out.shape == hidden.shape


def test_hdim_full_forward_pass_end_to_end():
    model = HDIMFull(hidden_size=16, heads=2, num_rotors=2)
    hidden = torch.randn(2, 5, 16)

    out = model(hidden, src_rotor_idx=0, tgt_rotor_idx=1)

    assert out.shape == hidden.shape
