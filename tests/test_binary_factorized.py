import torch

from hagi.model.binary_factorized import BinaryFactorizedLinear


def test_shape():
    layer = BinaryFactorizedLinear(64, 32, rank=8)
    x = torch.randn(2, 10, 64)
    y = layer(x)
    assert y.shape == (2, 10, 32)


def test_binarized_values():
    layer = BinaryFactorizedLinear(8, 4, rank=2)
    layer.B1.data = torch.randn(8, 2)
    layer.B2.data = torch.randn(2, 4)
    b1 = torch.where(layer.B1.data >= 0, 1.0, -1.0)
    b2 = torch.where(layer.B2.data >= 0, 1.0, -1.0)
    W = (b1 @ b2).t() * layer.scale.unsqueeze(1)
    x = torch.randn(1, 8)
    y = layer(x)
    expected = torch.nn.functional.linear(x, W)
    assert torch.allclose(y, expected, atol=1e-6)


def test_backward():
    layer = BinaryFactorizedLinear(16, 8, rank=4)
    x = torch.randn(2, 16, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    assert layer.B1.grad is not None
    assert layer.B2.grad is not None
    assert layer.scale.grad is not None
    assert x.grad is not None


def test_ste_approximates_identity():
    """STE should make gradient flow as if binarization is identity."""
    layer = BinaryFactorizedLinear(8, 4, rank=2)
    layer.B1.data = torch.randn(8, 2)
    layer.B2.data = torch.randn(2, 4)
    x = torch.randn(1, 8)
    y = layer(x)
    y.sum().backward()
    # Gradient should not be zero (it would be if sign had zero gradient)
    assert layer.B1.grad is not None
    assert layer.B2.grad is not None
    assert layer.B1.grad.abs().sum().item() > 0
    assert layer.B2.grad.abs().sum().item() > 0
