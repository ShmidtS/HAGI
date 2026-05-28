"""Smoke tests: every ablation model instantiates and runs a forward pass."""

import torch

from prototype.model.hagi import HAGI, HAGIConfig


def _forward(use_loop: bool, use_gdr: bool):
    cfg = HAGIConfig(vocab_size=256, loop_count=2, use_loop=use_loop, use_gdr=use_gdr)
    model = HAGI(cfg)
    ids = torch.randint(0, 256, (2, 16))
    logits = model(ids)
    assert logits.shape == (2, 16, 256)
    assert torch.isfinite(logits).all()
    return model


def test_model_a_baseline():
    _forward(use_loop=False, use_gdr=False)


def test_model_b_loop():
    _forward(use_loop=True, use_gdr=False)


def test_model_c_hdim():
    _forward(use_loop=False, use_gdr=True)


def test_model_d_gdr():
    _forward(use_loop=True, use_gdr=True)


def test_backward():
    model = _forward(use_loop=True, use_gdr=True)
    ids = torch.randint(0, 256, (2, 16))
    logits = model(ids)
    loss = logits.float().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
