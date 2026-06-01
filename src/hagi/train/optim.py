"""Optimizers: AdamW baseline + Muon hybrid + Schedule-Free AdamW."""

from __future__ import annotations

import math

import torch
from torch import nn


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate orthogonalization of a 2D matrix via quintic Newton-Schulz."""
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
    """Momentum SGD with per-step orthogonalization of 2D updates."""

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
    if p.ndim != 2:
        return False
    lowered = name.lower()
    excluded = ("embed", "lm_head", "norm", "iter_embed", "gate")
    if any(tok in lowered for tok in excluded):
        return False
    return True


class ScheduleFreeAdamW(torch.optim.Optimizer):
    """AdamW with fixed LR and built-in Polyak-Ruppert parameter averaging.

    Defazio et al., NeurIPS 2024 — simplified schedule-free variant.
    The optimizer maintains a running EMA of parameters in ``state[p]["z"]``.
    Evaluation should use the averaged copy (exposed via ``get_avg_params``).
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        avg_decay: float = 0.999,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, avg_decay=avg_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            avg_decay = group["avg_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["z"] = p.detach().clone()

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                z = state["z"]

                # AdamW weight decay (decoupled)
                if wd != 0:
                    p.mul_(1 - lr * wd)

                # Adam momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Schedule-free averaging (Polyak-Ruppert EMA)
                z.mul_(avg_decay).add_(p, alpha=1 - avg_decay)

    def get_avg_params(self) -> list[torch.Tensor]:
        """Return a list of averaged parameter tensors (for evaluation)."""
        avg = []
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "z" in state:
                    avg.append(state["z"])
                else:
                    avg.append(p)
        return avg


class AdamMini(torch.optim.Optimizer):
    """Memory-efficient AdamW variant (ICLR 2025) with block-wise optimizer states.

    For 2D+ weight matrices the first dimension is split into blocks; momentum
    and variance are stored per block rather than per element, cutting state
    memory by ~50% for typical transformer shapes.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        block_size: int = 128,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, block_size=block_size)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            block_size = group["block_size"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if p.ndim <= 1:
                    # Standard AdamW for 1-D / scalar parameters
                    if len(state) == 0:
                        state["step"] = 0
                        state["m"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)
                    m = state["m"]
                    v = state["v"]
                    state["step"] += 1
                    step = state["step"]

                    if wd != 0:
                        p.mul_(1 - lr * wd)

                    m.mul_(beta1).add_(g, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                    bias_corr1 = 1 - beta1 ** step
                    bias_corr2 = 1 - beta2 ** step
                    m_hat = m / bias_corr1
                    v_hat = v / bias_corr2
                    p.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)
                else:
                    # Block-wise Adam-mini for 2-D+ parameters
                    orig_shape = p.shape
                    g2d = g.view(g.shape[0], -1)
                    out_dim, in_dim = g2d.shape
                    num_blocks = max(1, (out_dim + block_size - 1) // block_size)

                    if len(state) == 0:
                        state["step"] = 0
                        state["m"] = torch.zeros(num_blocks, in_dim, dtype=p.dtype, device=p.device)
                        state["v"] = torch.zeros(num_blocks, in_dim, dtype=p.dtype, device=p.device)

                    m = state["m"]
                    v = state["v"]
                    state["step"] += 1
                    step = state["step"]

                    # Per-block mean gradient
                    g_mean = torch.zeros(num_blocks, in_dim, dtype=g2d.dtype, device=g2d.device)
                    for b in range(num_blocks):
                        start = b * block_size
                        end = min((b + 1) * block_size, out_dim)
                        g_mean[b] = g2d[start:end].mean(dim=0)

                    if wd != 0:
                        p.mul_(1 - lr * wd)

                    m.mul_(beta1).add_(g_mean, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(g_mean, g_mean, value=1 - beta2)

                    bias_corr1 = 1 - beta1 ** step
                    bias_corr2 = 1 - beta2 ** step
                    m_hat = m / bias_corr1
                    v_hat = v / bias_corr2

                    # Broadcast block update back to parameter shape
                    update_blocks = []
                    for b in range(num_blocks):
                        start = b * block_size
                        end = min((b + 1) * block_size, out_dim)
                        block_update = m_hat[b] / (v_hat[b].sqrt() + eps)
                        block_update = block_update.unsqueeze(0).expand(end - start, -1)
                        update_blocks.append(block_update)
                    update = torch.cat(update_blocks, dim=0).view(orig_shape)

                    p.add_(update, alpha=-lr)


class CombinedOptimizer:
    """Steps several optimizers together; exposes a unified zero_grad/step/state_dict."""

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

    def state_dict(self):
        return {f"opt_{i}": opt.state_dict() for i, opt in enumerate(self.optimizers)}

    def load_state_dict(self, state_dict):
        for i, opt in enumerate(self.optimizers):
            key = f"opt_{i}"
            if key not in state_dict:
                raise KeyError(f"CombinedOptimizer.load_state_dict: missing key {key!r}")
            opt.load_state_dict(state_dict[key])


def _build_muon_adamw(named: list[tuple[str, nn.Parameter]], cfg: dict) -> "CombinedOptimizer":
    """Muon on 2D hidden weights + AdamW on 1D/embed/head. arch_decision §Optimizer."""
    adamw_lr = float(cfg.get("adamw_lr", cfg.get("learning_rate", 3e-4)))
    wd = float(cfg.get("weight_decay", 0.1))
    betas = tuple(cfg.get("betas", (0.9, 0.95)))
    eps = float(cfg.get("eps", 1e-8))
    muon_params = [p for n, p in named if _is_muon_param(n, p)]
    adam_params = [p for n, p in named if not _is_muon_param(n, p)]
    muon = Muon(
        muon_params,
        lr=float(cfg.get("muon_lr", 0.02)),
        momentum=float(cfg.get("muon_momentum", 0.95)),
        ns_steps=int(cfg.get("muon_ns_steps", 5)),
    )
    adam = torch.optim.AdamW(
        adam_params,
        lr=adamw_lr,
        betas=betas,
        eps=eps,
        weight_decay=0.0,
    )
    return CombinedOptimizer([muon, adam])


def build_optimizer(model: nn.Module, cfg: dict):
    """Build AdamW or Muon+AdamW from a training-config dict."""
    kind = cfg.get("optimizer", "adamw").lower()
    lr = cfg.get("learning_rate", 3e-4)
    wd = cfg.get("weight_decay", 0.1)
    betas = tuple(cfg.get("betas", (0.9, 0.95)))
    eps = cfg.get("eps", 1e-8)

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    if kind in ("muon", "muon_adamw"):
        return _build_muon_adamw(named, cfg)

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

    if kind == "schedule-free-adamw":
        decay = [p for n, p in named if p.ndim >= 2 and "norm" not in n.lower()]
        no_decay = [p for n, p in named if not (p.ndim >= 2 and "norm" not in n.lower())]
        return ScheduleFreeAdamW(
            [
                {"params": decay, "weight_decay": wd},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr, betas=betas, eps=eps,
            avg_decay=cfg.get("schedule_free_avg_decay", 0.999),
        )

    if kind == "adam-mini":
        decay = [p for n, p in named if p.ndim >= 2 and "norm" not in n.lower()]
        no_decay = [p for n, p in named if not (p.ndim >= 2 and "norm" not in n.lower())]
        return AdamMini(
            [
                {"params": decay, "weight_decay": wd},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr, betas=betas, eps=eps,
            block_size=cfg.get("adam_mini_block_size", 128),
        )

    raise ValueError(
        f"unknown optimizer: {kind!r} (expected 'adamw', 'muon_adamw', 'schedule-free-adamw' or 'adam-mini')"
    )
