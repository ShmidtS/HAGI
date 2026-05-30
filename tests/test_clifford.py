import pytest

torch = pytest.importorskip("torch")

from hagi.model.clifford import build_product_table, geometric_product, grade_projection, reverse


def test_build_product_table_returns_correct_shapes():
    out_index, sign = build_product_table()

    assert out_index.shape == torch.Size([8, 8])
    assert sign.shape == torch.Size([8, 8])
    assert out_index.dtype == torch.long
    assert sign.dtype == torch.float32


def test_geometric_product_with_identity_multivector():
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x = torch.tensor([2.0, -1.0, 3.0, 4.0, 5.0, -6.0, 7.0, 8.0])

    assert torch.allclose(geometric_product(identity, x), x)
    assert torch.allclose(geometric_product(x, identity), x)


def test_grade_projection_preserves_only_requested_grade():
    mv = torch.arange(8, dtype=torch.float32)

    grade_2 = grade_projection(mv, 2)

    assert torch.equal(grade_2, torch.tensor([0.0, 0.0, 0.0, 3.0, 0.0, 5.0, 6.0, 0.0]))


def test_reverse_sign_pattern():
    mv = torch.arange(8, dtype=torch.float32)

    assert torch.equal(reverse(mv), torch.tensor([0.0, 1.0, 2.0, -3.0, 4.0, -5.0, -6.0, -7.0]))
