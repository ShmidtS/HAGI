"""Tests for extended Clifford Cl(3,0,0) operations.

Covers:
- vectorised geometric_product (matches loop version, vectorised path)
- wedge_product (exterior)
- inner_product (grade-reducing)
- commutator / anticommutator
- bivector exponential (rotor)
"""

import pytest

torch = pytest.importorskip("torch")

from hagi.model.clifford import (
    BLADE_COUNT,
    anticommutator,
    bivector_exp,
    commutator,
    geometric_product,
    grade_projection,
    inner_product,
    wedge_product,
)


E1 = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
E2 = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
E3 = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
E12 = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
IDENTITY = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_geometric_product_matches_loop_reference():
    """Vectorised product must match the existing loop-based implementation."""
    from hagi.model.clifford import _OUT_INDEX, _SIGN

    x = torch.randn(4, 5, BLADE_COUNT)
    y = torch.randn(4, 5, BLADE_COUNT)

    out = geometric_product(x, y)

    expected = torch.zeros_like(x)
    for a in range(BLADE_COUNT):
        for b in range(BLADE_COUNT):
            c = int(_OUT_INDEX[a, b])
            expected[..., c] = expected[..., c] + _SIGN[a, b] * x[..., a] * y[..., b]
    assert torch.allclose(out, expected, atol=1e-6)


def test_wedge_product_zero_when_first_arg_zero():
    zero = torch.zeros(BLADE_COUNT)
    out = wedge_product(zero, E2)
    assert torch.allclose(out, torch.zeros(BLADE_COUNT))


def test_wedge_product_antisymmetric_for_vectors():
    a = torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 4.0, 5.0, 0.0, 6.0, 0.0, 0.0, 0.0])
    assert torch.allclose(wedge_product(a, b), -wedge_product(b, a), atol=1e-6)


def test_wedge_product_e1_e2_equals_e12():
    out = wedge_product(E1, E2)
    assert torch.allclose(out, E12, atol=1e-6)


def test_wedge_product_lands_in_grade_2():
    a = torch.randn(BLADE_COUNT)
    b = torch.randn(BLADE_COUNT)
    out = wedge_product(a, b)
    g2 = grade_projection(out, 2)
    assert torch.allclose(out, g2, atol=1e-6)


def test_inner_product_euclidean_basis():
    assert torch.isclose(inner_product(E1, E1), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(inner_product(E1, E2), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(inner_product(E2, E3), torch.tensor(0.0), atol=1e-6)


def test_inner_product_symmetric():
    a = torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 4.0, 5.0, 0.0, 6.0, 0.0, 0.0, 0.0])
    assert torch.isclose(inner_product(a, b), inner_product(b, a), atol=1e-6)


def test_commutator_antisymmetric():
    a = torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 4.0, 5.0, 0.0, 6.0, 0.0, 0.0, 0.0])
    assert torch.allclose(commutator(a, b), -commutator(b, a), atol=1e-6)


def test_commutator_e1_e2_equals_e12():
    out = commutator(E1, E2)
    assert torch.allclose(out, E12, atol=1e-6)


def test_anticommutator_symmetric():
    a = torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 4.0, 5.0, 0.0, 6.0, 0.0, 0.0, 0.0])
    assert torch.allclose(anticommutator(a, b), anticommutator(b, a), atol=1e-6)


def test_anticommutator_e1_e2_zero():
    out = anticommutator(E1, E2)
    assert torch.isclose(out.abs().max(), torch.tensor(0.0), atol=1e-6)


def test_bivector_exp_zero_input_gives_identity():
    zero = torch.zeros(BLADE_COUNT)
    out = bivector_exp(zero)
    assert torch.allclose(out, IDENTITY, atol=1e-6)


def test_bivector_exp_produces_unit_rotor():
    """R = exp(-B/2) must satisfy R * reverse(R) = 1."""
    bv = torch.tensor([0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0])
    from hagi.model.clifford import reverse
    r = bivector_exp(bv)
    rev = reverse(r)
    prod = geometric_product(r, rev)
    assert torch.isclose(prod[0], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(prod[1:].abs().max(), torch.tensor(0.0), atol=1e-5)
