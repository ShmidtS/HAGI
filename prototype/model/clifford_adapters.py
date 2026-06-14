"""Parameter-efficient adapters for the T1 bake-off: does Clifford structure beat
a low-rank adapter at equal budget on a *frozen* pretrained base?

Three residual/multiplicative adapters, all injected at the same `nn.Linear`
targets so the only thing that differs is the adapter's internal math:

  - LoRAAdapter        : y = base(x) + scale * B(A(x))           (low-rank delta)
  - GeoProductAdapter  : y = base(x) + alpha * proj_out(mv ⊗ w)  (ReZero geo residual)
  - RotorAdapter       : y = R · base(x) · R̃                     (orthogonal rotor sandwich)

GeoProductAdapter is the direct test of the one GDR ember that survived the gated
re-test (the geometric-product cross-term — the only gate that moved off zero).
Both Clifford adapters start as EXACT identities (alpha=0 / bivector=0), the same
ReZero discipline that made the gated GDR test fair: the model must opt INTO the
geometric machinery.

Fairness is by reported trainable-param count (see `count_trainable`) + identical
data/schedule/targets, not by forcing identical algebra.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .clifford import BLADE_COUNT, geometric_product, reverse

# Even-grade (rotor) blade indices in the Cl(3,0,0) layout: scalar + the 3 bivectors.
_BIVECTOR_BLADES = (3, 5, 6)  # e1e2, e1e3, e2e3


def _broadcast_mv(weight: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Broadcast a per-channel multivector [..., n_mv, 8] against batched `like`."""
    return torch.broadcast_to(weight, like.shape)


class LoRAAdapter(nn.Module):
    """Standard low-rank delta on the input: delta = scale * B(A(x)), B init 0."""

    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)  # delta == 0 at init
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        # Adapter params stay fp32 for stable training; base may be fp16/bf16.
        dt = self.A.weight.dtype
        delta = self.B(self.A(x.to(dt))) * self.scale
        return base_out + delta.to(base_out.dtype)


class GeoProductAdapter(nn.Module):
    """ReZero geometric-product residual. Projects x into n_mv multivectors, takes the
    geometric product with a learned per-channel multivector, projects back. alpha init
    0 -> exact identity at init (the gated-GDR discipline, now as an adapter)."""

    def __init__(self, in_features: int, out_features: int, n_mv: int = 2):
        super().__init__()
        self.n_mv = n_mv
        self.proj_in = nn.Linear(in_features, n_mv * BLADE_COUNT, bias=False)
        self.proj_out = nn.Linear(n_mv * BLADE_COUNT, out_features, bias=False)
        # Learned multivector the input is geometrically multiplied by (small init).
        self.weight_mv = nn.Parameter(torch.randn(n_mv, BLADE_COUNT) * 0.02)
        self.alpha = nn.Parameter(torch.zeros(1))  # ReZero gate
        nn.init.kaiming_uniform_(self.proj_in.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.proj_out.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        dt = self.proj_in.weight.dtype  # adapter math in fp32, base may be fp16/bf16
        x = x.to(dt)
        *lead, _ = x.shape
        mv = self.proj_in(x).reshape(*lead, self.n_mv, BLADE_COUNT)
        w = _broadcast_mv(self.weight_mv, mv)
        prod = geometric_product(mv, w)  # [..., n_mv, 8]
        delta = self.alpha * self.proj_out(prod.reshape(*lead, self.n_mv * BLADE_COUNT))
        return base_out + delta.to(base_out.dtype)


class RotorAdapter(nn.Module):
    """Orthogonal fine-tuning via Clifford rotors. Reshapes base_out into n_mv 8-blade
    multivectors and applies the sandwich R v R̃ with a learned rotor R = exp(-B/2) per
    channel (B a learned bivector). bivector init 0 -> R = 1 -> identity at init.
    Param-frugal by design (3 angles/channel); raise coverage to match LoRA's budget."""

    def __init__(self, out_features: int, **_ignored):
        super().__init__()
        assert out_features % BLADE_COUNT == 0, "rotor target out_features must be divisible by 8"
        self.n_mv = out_features // BLADE_COUNT
        # One bivector (3 angles) per multivector channel; init 0 == identity rotor.
        self.bivector = nn.Parameter(torch.zeros(self.n_mv, len(_BIVECTOR_BLADES)))

    def _rotor(self) -> torch.Tensor:
        """Build R = exp(-B/2) as a multivector [n_mv, 8] from the learned bivectors."""
        biv = self.bivector
        phi = biv.norm(dim=-1, keepdim=True)            # [n_mv, 1]
        half = phi * 0.5
        # sin(half)/phi, with the well-defined limit 0.5 as phi -> 0.
        sinc = torch.where(phi > 1e-8, torch.sin(half) / phi.clamp_min(1e-8),
                           torch.full_like(phi, 0.5))
        R = torch.zeros(self.n_mv, BLADE_COUNT, device=biv.device, dtype=biv.dtype)
        R[:, 0] = torch.cos(half).squeeze(-1)
        for k, blade in enumerate(_BIVECTOR_BLADES):   # R = cos(half) - sinc * B
            R[:, blade] = -(sinc.squeeze(-1) * biv[:, k])
        return R

    def forward(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        dt = self.bivector.dtype  # rotor math in fp32, base may be fp16/bf16
        *lead, last = base_out.shape
        v = base_out.to(dt).reshape(*lead, self.n_mv, BLADE_COUNT)
        R = self._rotor()                               # [n_mv, 8]
        Rb = _broadcast_mv(R, v)
        Rt = _broadcast_mv(reverse(R), v)
        rotated = geometric_product(geometric_product(Rb, v), Rt)
        return rotated.reshape(*lead, last).to(base_out.dtype)


class WrappedLinear(nn.Module):
    """Frozen base Linear with an adapter applied to its output."""

    def __init__(self, base: nn.Linear, adapter: nn.Module):
        super().__init__()
        self.base = base
        self.adapter = adapter
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x, self.base(x))


_ADAPTERS = {"lora": LoRAAdapter, "geo": GeoProductAdapter, "rotor": RotorAdapter}


def _make_adapter(kind: str, lin: nn.Linear, **knobs) -> nn.Module:
    in_f, out_f = lin.in_features, lin.out_features
    if kind == "lora":
        return LoRAAdapter(in_f, out_f, rank=knobs.get("rank", 16), alpha=knobs.get("alpha", 16.0))
    if kind == "geo":
        return GeoProductAdapter(in_f, out_f, n_mv=knobs.get("n_mv", 2))
    if kind == "rotor":
        return RotorAdapter(out_f)
    raise ValueError(f"unknown adapter kind: {kind}")


def _set_submodule(root: nn.Module, dotted: str, value: nn.Module) -> None:
    parent = root
    *path, last = dotted.split(".")
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, last, value)


def inject_adapters(model: nn.Module, kind: str, targets=("q_proj", "v_proj"),
                    freeze_base: bool = True, **knobs) -> dict:
    """Wrap every `nn.Linear` whose name ends in a target suffix with `kind` adapter.

    Returns a summary dict (n_wrapped, trainable params, skipped). Base is frozen.
    """
    if freeze_base:
        for p in model.parameters():
            p.requires_grad_(False)

    to_wrap, skipped = [], []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(name.endswith(t) for t in targets):
            if kind == "rotor" and mod.out_features % BLADE_COUNT != 0:
                skipped.append(name)
                continue
            to_wrap.append((name, mod))

    for name, lin in to_wrap:
        _set_submodule(model, name, WrappedLinear(lin, _make_adapter(kind, lin, **knobs)))

    return {
        "kind": kind,
        "n_wrapped": len(to_wrap),
        "skipped": skipped,
        "trainable": count_trainable(model),
    }


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
