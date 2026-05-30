"""Optimizers: AdamW baseline + Muon (orthogonalized momentum) hybrid.

Muon orthogonalizes each 2D weight-matrix update via a Newton-Schulz iteration
before the descent step. It powers the nanoGPT speedrun records and converges in
fewer steps than AdamW on small-model FineWeb pretraining. Embeddings, the LM
head, norms, biases, gates, and iteration embeddings are NOT matrices in the
Muon sense — they use AdamW.

Reference: Keller Jordan, modded-nanogpt (https://github.com/KellerJordan/modded-nanogpt).
Hyperparameters (lr, momentum, ns_steps) may need tuning per the Smol Training
Playbook and Muon follow-up papers. Use AdamW for the clean baseline control;
ablate Muon as a separate variable (do not conflate optimizer with architecture).
"""

from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate orthogonalization of a 2D matrix via quintic Newton-Schulz.

    Returns a matrix with (approximately) the same singular vectors as G but all
    singular values pushed toward 1. Runs in bf16 for speed; the quintic
    coefficients are the standard modded-nanogpt values.
    """
    assert G.ndim == 2, "Muon orthogonalization expects a 2D matrix"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Momentum SGD with per-step orthogonalization of the update matrix.

    Only for 2D parameters. Update is scaled by sqrt(max(1, rows/cols)) to match
    the effective step size across non-square matrices.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                update = zeropower_via_newtonschulz5(update, ns_steps)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(update.reshape(p.shape).type_as(p), alpha=-lr * scale)


def _is_muon_param(name: str, p: nn.Parameter) -> bool:
    """2D weight matrices in the transformer/GDR core. Excludes embeddings,
    LM head, norms, biases, gates, and iteration embeddings."""
    if p.ndim != 2:
        return False
    lowered = name.lower()
    excluded = ("embed", "lm_head", "norm", "iter_embed", "gate")
    return not any(tok in lowered for tok in excluded)


class CombinedOptimizer:
    """Steps several optimizers together; exposes a unified zero_grad/step.

    Used for the Muon+AdamW hybrid: Muon for core matrices, AdamW for the rest.
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        for opt in self.optimizers:
            opt.step()

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups


def build_optimizer(model: nn.Module, cfg: dict):
    """Build the optimizer from a training-config dict.

    cfg keys:
        optimizer: "adamw" (default) | "muon"
        learning_rate, weight_decay, betas, eps  (AdamW)
        muon_lr, muon_momentum, muon_ns_steps     (Muon, when optimizer=muon)
    """
    kind = cfg.get("optimizer", "adamw").lower()
    lr = cfg.get("learning_rate", 3e-4)
    wd = cfg.get("weight_decay", 0.1)
    betas = tuple(cfg.get("betas", (0.9, 0.95)))
    eps = cfg.get("eps", 1e-8)

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    if kind == "adamw":
        decay = [p for n, p in named if p.ndim >= 2 and "norm" not in n.lower()]
        no_decay = [p for n, p in named if not (p.ndim >= 2 and "norm" not in n.lower())]
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": wd},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr, betas=betas, eps=eps,
        )

    if kind == "muon":
        muon_params = [p for n, p in named if _is_muon_param(n, p)]
        adam_params = [p for n, p in named if not _is_muon_param(n, p)]
        muon = Muon(
            muon_params,
            lr=cfg.get("muon_lr", 0.02),
            momentum=cfg.get("muon_momentum", 0.95),
            ns_steps=cfg.get("muon_ns_steps", 5),
        )
        adam = torch.optim.AdamW(adam_params, lr=lr, betas=betas, eps=eps, weight_decay=0.0)
        return CombinedOptimizer([muon, adam])

    raise ValueError(f"unknown optimizer: {kind!r} (expected 'adamw' or 'muon')")
