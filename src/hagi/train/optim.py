"""Optimizers: AdamW baseline + Muon hybrid + Schedule-Free AdamW."""

from __future__ import annotations

import math
from typing import Any, cast

import torch
from torch import nn


@torch.no_grad()
def _zeropower_impl(G: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = G if G.dtype == torch.bfloat16 else G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        A = x @ x.T
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        x = torch.addmm(x, B, x, beta=a, alpha=1.0)
    if transposed:
        x = x.T
    return x if x.dtype == G.dtype else x.to(G.dtype)


try:
    _zeropower_jit = torch.jit.script(_zeropower_impl)
except Exception:
    _zeropower_jit = None


def zeropower_via_newtonschulz5(
    G: torch.Tensor, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """Approximate orthogonalization of a 2D matrix via quintic Newton-Schulz."""
    assert G.ndim == 2, "Muon orthogonalization expects a 2D matrix"
    fn = _zeropower_jit if _zeropower_jit is not None else _zeropower_impl
    return fn(G, steps, eps)


@torch.no_grad()
def _zeropower_batched_impl(G: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = G if G.dtype == torch.bfloat16 else G.bfloat16()
    transposed = G.size(1) > G.size(2)
    if transposed:
        x = x.transpose(1, 2)
    norms = x.flatten(1).norm(dim=1).view(-1, 1, 1) + eps
    x = x / norms
    for _ in range(steps):
        A = x @ x.transpose(1, 2)
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        x = torch.baddbmm(x, B, x, beta=a, alpha=1.0)
    if transposed:
        x = x.transpose(1, 2)
    return x if x.dtype == G.dtype else x.to(G.dtype)


try:
    _zeropower_batched_jit = torch.jit.script(_zeropower_batched_impl)
except Exception:
    _zeropower_batched_jit = None


def zeropower_via_newtonschulz5_batched(
    G: torch.Tensor, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """Batched quintic Newton-Schulz over a [B, M, N] stack of matrices.

    Numerically equivalent to running zeropower_via_newtonschulz5 per matrix,
    but executes one bmm chain instead of B sequential matmul chains — much
    less kernel-launch overhead for many small same-shape parameters.
    """
    assert G.ndim == 3, "batched Muon orthogonalization expects [B, M, N]"
    fn = (
        _zeropower_batched_jit
        if _zeropower_batched_jit is not None
        else _zeropower_batched_impl
    )
    return fn(G, steps, eps)


class Muon(torch.optim.Optimizer):
    """Momentum SGD with per-step orthogonalization of 2D updates."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> float | None:  # type: ignore
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            # Group by shape for fused foreach ops
            shape_groups: dict[tuple[int, ...], list[torch.Tensor]] = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                shape_groups.setdefault(p.shape, []).append(p)
            for params in shape_groups.values():
                grads: list[torch.Tensor] = []
                bufs: list[torch.Tensor] = []
                for p in params:
                    g = p.grad
                    assert g is not None
                    grads.append(g)
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    bufs.append(state["momentum_buffer"])
                torch._foreach_mul_(bufs, momentum)
                torch._foreach_add_(bufs, grads)
                if nesterov:
                    updates = [g.add(b, alpha=momentum) for g, b in zip(grads, bufs)]
                else:
                    updates = list(bufs)
                if len(updates) > 1:
                    # One batched bmm chain for the whole same-shape group.
                    stacked = torch.stack(updates)
                    ortho = zeropower_via_newtonschulz5_batched(stacked, ns_steps)
                    p0 = params[0]
                    scale = min(max(1.0, p0.size(0) / p0.size(1)) ** 0.5, 2.0)
                    torch._foreach_add_(
                        params, list(ortho.type_as(p0).unbind(0)), alpha=-lr * scale
                    )
                else:
                    for p, update in zip(params, updates):
                        update = zeropower_via_newtonschulz5(update, ns_steps)
                        scale = min(max(1.0, p.size(0) / p.size(1)) ** 0.5, 2.0)
                        p.add_(update.reshape(p.shape).type_as(p), alpha=-lr * scale)


def _is_muon_param(name: str, p: nn.Parameter) -> bool:
    if p.ndim != 2:
        return False
    lowered = name.lower()
    excluded = ("embed", "lm_head", "norm", "iter_embed", "gate", "router")
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
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, avg_decay=avg_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> float | None:  # type: ignore
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

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
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
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            block_size=block_size,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> float | None:  # type: ignore
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

                    bias_corr1 = 1 - beta1**step
                    bias_corr2 = 1 - beta2**step
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
                        state["m"] = torch.zeros(
                            num_blocks, in_dim, dtype=p.dtype, device=p.device
                        )
                        state["v"] = torch.zeros(
                            num_blocks, in_dim, dtype=p.dtype, device=p.device
                        )

                    m = state["m"]
                    v = state["v"]
                    state["step"] += 1
                    step = state["step"]

                    # Per-block mean gradient
                    g_mean = torch.zeros(
                        num_blocks, in_dim, dtype=g2d.dtype, device=g2d.device
                    )
                    for b in range(num_blocks):
                        start = b * block_size
                        end = min((b + 1) * block_size, out_dim)
                        g_mean[b] = g2d[start:end].mean(dim=0)

                    if wd != 0:
                        p.mul_(1 - lr * wd)

                    m.mul_(beta1).add_(g_mean, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(g_mean, g_mean, value=1 - beta2)

                    bias_corr1 = 1 - beta1**step
                    bias_corr2 = 1 - beta2**step
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


class AdEMAMix(torch.optim.Optimizer):
    """AdamW with dual EMA: fast (beta1) and slow (beta3) momentum.

    Mixes the two momenta with alpha and applies standard AdamW weight decay.
    Replaces separate EMA copy because the slow momentum is already an EMA of
    parameters.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 0.5,
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        defaults = dict(
            lr=lr, betas=betas, alpha=alpha, eps=eps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> float | None:  # type: ignore
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2, beta3 = group["betas"]
            alpha = group["alpha"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m_fast"] = torch.zeros_like(p)
                    state["m_slow"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]
                m_fast = state["m_fast"]
                m_slow = state["m_slow"]
                v = state["v"]

                if wd != 0:
                    p.mul_(1 - lr * wd)

                m_fast.mul_(beta1).add_(g, alpha=1 - beta1)
                m_slow.mul_(beta3).add_(g, alpha=1 - beta3)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                m_mix = alpha * m_fast + (1 - alpha) * m_slow
                bias_corr1 = 1 - beta1**step
                bias_corr2 = 1 - beta2**step
                m_hat = m_mix / bias_corr1
                v_hat = v / bias_corr2
                p.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)


class CombinedOptimizer(torch.optim.Optimizer):
    """Steps several optimizers together; exposes a unified zero_grad/step/state_dict."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        # pass a dummy param so the base class doesn't error on empty params
        super().__init__([torch.zeros(1)], {})
        self.optimizers = optimizers
        self.param_groups = []
        for opt in optimizers:
            self.param_groups.extend(opt.param_groups)

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None) -> float | None:  # type: ignore
        for opt in self.optimizers:
            opt.step()
        return None

    def state_dict(self):
        return {f"opt_{i}": opt.state_dict() for i, opt in enumerate(self.optimizers)}

    def load_state_dict(self, state_dict):
        for i, opt in enumerate(self.optimizers):
            key = f"opt_{i}"
            if key not in state_dict:
                raise KeyError(
                    f"CombinedOptimizer.load_state_dict: missing key {key!r}"
                )
            opt.load_state_dict(state_dict[key])


def _build_muon_ademamix(
    named: list[tuple[str, nn.Parameter]], cfg: dict[str, Any]
) -> "CombinedOptimizer":
    """Muon on 2D hidden weights + AdEMAMix on 1D/embed/head (replaces AdamW + separate EMA)."""
    adamw_lr = float(cfg.get("adamw_lr", cfg.get("learning_rate", 3e-4)))
    wd = float(cfg.get("weight_decay", 0.1))
    betas_cfg = cast(
        tuple[float, float, float], tuple(cfg.get("betas", (0.9, 0.999, 0.9999)))
    )
    eps = float(cfg.get("eps", 1e-8))
    alpha = float(cfg.get("ademamix_alpha", 0.5))
    muon_params = [p for n, p in named if _is_muon_param(n, p)]
    adam_decay = [
        p
        for n, p in named
        if not _is_muon_param(n, p) and p.ndim >= 2 and "norm" not in n.lower()
    ]
    adam_no_decay = [
        p
        for n, p in named
        if not _is_muon_param(n, p) and not (p.ndim >= 2 and "norm" not in n.lower())
    ]
    muon = Muon(
        muon_params,
        lr=float(cfg.get("muon_lr", 0.02)),
        momentum=float(cfg.get("muon_momentum", 0.95)),
        ns_steps=int(cfg.get("muon_ns_steps", 5)),
    )
    ademamix = AdEMAMix(
        [
            {"params": adam_decay, "weight_decay": wd},
            {"params": adam_no_decay, "weight_decay": 0.0},
        ],
        lr=adamw_lr,
        betas=betas_cfg,
        alpha=alpha,
        eps=eps,
    )
    return CombinedOptimizer([muon, ademamix])


def _build_muon_adamw(
    named: list[tuple[str, nn.Parameter]], cfg: dict[str, Any]
) -> "CombinedOptimizer":
    """Muon on 2D hidden weights + AdamW on 1D/embed/head. arch_decision §Optimizer."""
    adamw_lr = float(cfg.get("adamw_lr", cfg.get("learning_rate", 3e-4)))
    wd = float(cfg.get("weight_decay", 0.1))
    betas_cfg = cast(tuple[float, float], tuple(cfg.get("betas", (0.9, 0.95))))
    eps = float(cfg.get("eps", 1e-8))
    muon_params = [p for n, p in named if _is_muon_param(n, p)]
    adam_decay = [
        p
        for n, p in named
        if not _is_muon_param(n, p) and p.ndim >= 2 and "norm" not in n.lower()
    ]
    adam_no_decay = [
        p
        for n, p in named
        if not _is_muon_param(n, p) and not (p.ndim >= 2 and "norm" not in n.lower())
    ]
    muon = Muon(
        muon_params,
        lr=float(cfg.get("muon_lr", 0.02)),
        momentum=float(cfg.get("muon_momentum", 0.95)),
        ns_steps=int(cfg.get("muon_ns_steps", 5)),
    )
    adam = torch.optim.AdamW(
        [
            {"params": adam_decay, "weight_decay": wd},
            {"params": adam_no_decay, "weight_decay": 0.0},
        ],
        lr=adamw_lr,
        betas=betas_cfg,
        eps=eps,
        fused=True,
        capturable=True,
    )
    return CombinedOptimizer([muon, adam])


def build_optimizer(model: nn.Module, cfg: dict[str, Any]):
    """Build AdamW or Muon+AdamW from a training-config dict."""
    kind = cfg.get("optimizer", "adamw").lower()
    lr = cfg.get("learning_rate", 3e-4)
    wd = cfg.get("weight_decay", 0.1)
    betas_cfg = cast(tuple[float, float], tuple(cfg.get("betas", (0.9, 0.95))))
    eps = cfg.get("eps", 1e-8)

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    if kind in ("muon", "muon_adamw"):
        return _build_muon_adamw(named, cfg)

    if kind == "muon_ademamix":
        return _build_muon_ademamix(named, cfg)

    if kind == "adamw":
        decay = [p for n, p in named if p.ndim >= 2 and "norm" not in n.lower()]
        no_decay = [
            p for n, p in named if not (p.ndim >= 2 and "norm" not in n.lower())
        ]
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": wd},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=betas_cfg,
            eps=eps,
            fused=True,
            capturable=True,
        )

    if kind == "schedule-free-adamw":
        decay = [p for n, p in named if p.ndim >= 2 and "norm" not in n.lower()]
        no_decay = [
            p for n, p in named if not (p.ndim >= 2 and "norm" not in n.lower())
        ]
        return ScheduleFreeAdamW(
            [
                {"params": decay, "weight_decay": wd},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=betas_cfg,
            eps=eps,
            avg_decay=cfg.get("schedule_free_avg_decay", 0.999),
        )

    if kind == "adam-mini":
        decay = [p for n, p in named if p.ndim >= 2 and "norm" not in n.lower()]
        no_decay = [
            p for n, p in named if not (p.ndim >= 2 and "norm" not in n.lower())
        ]
        return AdamMini(
            [
                {"params": decay, "weight_decay": wd},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=betas_cfg,
            eps=eps,
            block_size=cfg.get("adam_mini_block_size", 128),
        )

    if kind in ("adamw8bit", "paged_adamw8bit"):
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("bitsandbytes not installed. `pip install bitsandbytes`.")
        decay = [p for n, p in named if p.ndim >= 2 and "norm" not in n.lower()]
        no_decay = [
            p for n, p in named if not (p.ndim >= 2 and "norm" not in n.lower())
        ]
        cls = (
            bnb.optim.PagedAdamW8bit
            if kind.startswith("paged")
            else bnb.optim.AdamW8bit
        )
        return cls(
            [
                {"params": decay, "weight_decay": wd},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=betas_cfg,
            eps=eps,
        )

    raise ValueError(
        f"unknown optimizer: {kind!r} (expected 'adamw', 'muon_adamw', 'schedule-free-adamw', 'adam-mini', 'adamw8bit' or 'paged_adamw8bit')"
    )
