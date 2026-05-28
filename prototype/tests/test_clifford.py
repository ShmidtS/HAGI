"""Tests for Cl(3,0,0) geometric product. Mirrors Lean4 invariants in HDIM.lean."""

import torch

from prototype.model.clifford import (
    BLADE_COUNT,
    GRADE,
    geometric_product,
    grade_projection,
    reverse,
)


def test_grade_lookup():
    # Matches GRADE_LOOKUP_CL3 from the implementation plan.
    assert GRADE == [0, 1, 1, 2, 1, 2, 2, 3]


def test_scalar_identity():
    # Scalar 1 (blade 0) is the multiplicative identity.
    one = torch.zeros(BLADE_COUNT)
    one[0] = 1.0
    x = torch.randn(BLADE_COUNT)
    assert torch.allclose(geometric_product(one, x), x, atol=1e-6)
    assert torch.allclose(geometric_product(x, one), x, atol=1e-6)


def test_basis_vector_squares_to_one():
    # e1*e1 = e2*e2 = e3*e3 = +1 (Euclidean signature).
    for blade in (0b001, 0b010, 0b100):
        e = torch.zeros(BLADE_COUNT)
        e[blade] = 1.0
        prod = geometric_product(e, e)
        expected = torch.zeros(BLADE_COUNT)
        expected[0] = 1.0
        assert torch.allclose(prod, expected, atol=1e-6), f"blade {blade}"


def test_anticommutation():
    # e1*e2 = -e2*e1.
    e1 = torch.zeros(BLADE_COUNT)
    e1[0b001] = 1.0
    e2 = torch.zeros(BLADE_COUNT)
    e2[0b010] = 1.0
    assert torch.allclose(
        geometric_product(e1, e2), -geometric_product(e2, e1), atol=1e-6
    )


def test_pseudoscalar():
    # e1*e2*e3 = e123 (blade 0b111).
    e1 = torch.zeros(BLADE_COUNT)
    e1[0b001] = 1.0
    e2 = torch.zeros(BLADE_COUNT)
    e2[0b010] = 1.0
    e3 = torch.zeros(BLADE_COUNT)
    e3[0b100] = 1.0
    prod = geometric_product(geometric_product(e1, e2), e3)
    expected = torch.zeros(BLADE_COUNT)
    expected[0b111] = 1.0
    assert torch.allclose(prod, expected, atol=1e-6)


def test_batched():
    x = torch.randn(4, 16, BLADE_COUNT)
    y = torch.randn(4, 16, BLADE_COUNT)
    out = geometric_product(x, y)
    assert out.shape == (4, 16, BLADE_COUNT)


def test_grade_projection():
    mv = torch.arange(1.0, BLADE_COUNT + 1.0)
    g2 = grade_projection(mv, 2)
    # Grade-2 blades are indices 3, 5, 6.
    assert g2[3] == 4.0 and g2[5] == 6.0 and g2[6] == 7.0
    assert g2[0] == 0.0 and g2[1] == 0.0


def test_reverse_involution():
    mv = torch.randn(BLADE_COUNT)
    assert torch.allclose(reverse(reverse(mv)), mv, atol=1e-6)
