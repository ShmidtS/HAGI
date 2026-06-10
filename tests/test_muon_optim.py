"""Tests for the Muon+AdamW hybrid optimizer (G005 §B)."""

from __future__ import annotations

import torch
from torch import nn

from hagi.train.optim import Muon, _is_muon_param, build_optimizer


class TinyModel(nn.Module):
    def __init__(self, vocab: int = 16, hidden: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.fc1 = nn.Linear(hidden, hidden * 2, bias=False)
        self.fc2 = nn.Linear(hidden * 2, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)


def test_is_muon_param_excludes_embed_and_norm():
    m = TinyModel()
    named = dict(m.named_parameters())
    assert _is_muon_param("embed.weight", named["embed.weight"]) is False
    assert _is_muon_param("lm_head.weight", named["lm_head.weight"]) is False
    assert _is_muon_param("norm.weight", named["norm.weight"]) is False
    assert _is_muon_param("fc1.weight", named["fc1.weight"]) is True
    assert _is_muon_param("fc2.weight", named["fc2.weight"]) is True


def test_build_muon_adamw_splits_two_param_groups():
    m = TinyModel()
    opt = build_optimizer(m, {"optimizer": "muon_adamw", "learning_rate": 3e-4, "muon_lr": 0.02})

    # Three sub-optimizer groups: Muon + AdamW(decay) + AdamW(no_decay)
    groups = opt.param_groups
    assert len(groups) == 3
    muon_group = groups[0]
    adam_decay_group = groups[1]
    adam_no_decay_group = groups[2]
    muon_ids = {id(p) for p in muon_group["params"]}
    adam_decay_ids = {id(p) for p in adam_decay_group["params"]}
    adam_no_decay_ids = {id(p) for p in adam_no_decay_group["params"]}
    # No parameter is in both groups.
    assert muon_ids & adam_decay_ids == set()
    assert muon_ids & adam_no_decay_ids == set()
    assert adam_decay_ids & adam_no_decay_ids == set()

    named = dict(m.named_parameters())
    # Embedding + LM head are 2D, not muon -> AdamW decay
    assert id(named["embed.weight"]) in adam_decay_ids
    assert id(named["lm_head.weight"]) in adam_decay_ids
    # LayerNorm -> AdamW no_decay
    assert id(named["norm.weight"]) in adam_no_decay_ids
    # fc1, fc2 are 2D, not embed/head/norm -> Muon
    assert id(named["fc1.weight"]) in muon_ids
    assert id(named["fc2.weight"]) in muon_ids


def test_build_muon_adamw_uses_config_lrs():
    m = TinyModel()
    opt = build_optimizer(
        m,
        {
            "optimizer": "muon_adamw",
            "learning_rate": 3e-4,
            "adamw_lr": 1e-4,
            "muon_lr": 0.05,
        },
    )
    groups = opt.param_groups
    named = dict(m.named_parameters())
    muon_lr = next(
        g["lr"] for g in groups
        if any(id(p) == id(named["fc1.weight"]) for p in g["params"])
    )
    adam_lr = next(
        g["lr"] for g in groups
        if any(id(p) == id(named["norm.weight"]) for p in g["params"])
    )
    assert muon_lr == 0.05
    assert adam_lr == 1e-4


def test_muon_step_decreases_loss_on_tiny_inputs():
    torch.manual_seed(0)
    m = TinyModel(vocab=8, hidden=4)
    opt = build_optimizer(m, {"optimizer": "muon_adamw", "learning_rate": 1e-3, "muon_lr": 0.05})
    x = torch.randint(0, 8, (2, 5))
    y = torch.randint(0, 8, (2, 5))
    losses = []
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        h = m.embed(x)
        h = torch.relu(m.fc1(h))
        h = m.fc2(h)
        h = m.norm(h)
        logits = m.lm_head(h)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 8), y.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    # Final loss <= initial loss.
    assert losses[-1] <= losses[0] + 1e-6


def test_muon_momentum_state_persists():
    p = torch.zeros(4, 4, requires_grad=True)
    opt = Muon([p], lr=0.1, momentum=0.5, ns_steps=2)
    p.grad = torch.ones(4, 4)
    opt.step()
    # momentum_buffer stored.
    state = opt.state[p]
    assert "momentum_buffer" in state
    # Second step should accumulate momentum.
    p.grad = torch.ones(4, 4)
    opt.step()
    assert torch.isfinite(state["momentum_buffer"]).all()
