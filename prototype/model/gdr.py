"""Grade-Decomposed Recurrence (GDR) — HAGI's core novel mechanism.

The hidden state is split into Clifford grades with distinct update dynamics:

    scalar    (64)  : confidence/resolution  — slow   (momentum 0.9)
    vector    (192) : entities/concepts      — medium (momentum 0.5)
    bivector  (192) : relations              — fast   (full update)
    trivector (64)  : higher-order structure — fast   (full update)
    residual  (256) : unconstrained channel  — standard

The Cl(3,0,0) geometric product provides cross-grade interaction:
    vector x vector -> scalar + bivector

This module is applied once per recurrence iteration inside the reasoning core.
See docs/ARCHITECTURE.md for the full specification and the hypothesis under test.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .clifford import BLADE_COUNT, geometric_product, grade_projection


@dataclass
class GradeConfig:
    scalar: int = 64
    vector: int = 192
    bivector: int = 192
    trivector: int = 64
    residual: int = 256
    scalar_momentum: float = 0.9
    vector_momentum: float = 0.5
    geo_interaction: bool = True  # if False, skip the geometric-product cross-grade signal
    # ReZero-gated variant. The legacy (ungated) update has three optimization faults
    # the held-out ablation could not separate from the algebra itself:
    #   1. sigmoid(0)=0.5 - the geo gates start HALF-open, injecting noise at step 0;
    #   2. bivector/trivector are wholesale REPLACED each iteration (no residual path);
    #   3. default-init MLPs put D's initial function far from B's.
    # gated=True fixes all three: every grade update becomes h + alpha * f(h) with
    # alpha initialized to 0, so the module starts as an EXACT identity (D_gated == B
    # functionally at init) and gradient descent opts INTO grades only if they help.
    # Momentum is dropped in gated mode (the gate subsumes the update-rate role).
    gated: bool = False

    @property
    def hidden_size(self) -> int:
        return self.scalar + self.vector + self.bivector + self.trivector + self.residual

    @property
    def bounds(self) -> list[int]:
        s, v, b, t, r = (
            self.scalar,
            self.vector,
            self.bivector,
            self.trivector,
            self.residual,
        )
        return [0, s, s + v, s + v + b, s + v + b + t, s + v + b + t + r]


class GradeDecomposedRecurrence(nn.Module):
    """One iteration of grade-decomposed update + geometric interaction."""

    def __init__(self, cfg: GradeConfig):
        super().__init__()
        self.cfg = cfg
        ctx = cfg.scalar + cfg.vector + cfg.bivector + cfg.trivector

        # Per-grade update MLPs (each reads the full graded context).
        self.mlp_scalar = nn.Sequential(nn.Linear(ctx, ctx), nn.SiLU(), nn.Linear(ctx, cfg.scalar))
        self.mlp_vector = nn.Sequential(nn.Linear(ctx, ctx), nn.SiLU(), nn.Linear(ctx, cfg.vector))
        self.mlp_bivector = nn.Sequential(nn.Linear(ctx, ctx), nn.SiLU(), nn.Linear(ctx, cfg.bivector))
        self.mlp_trivector = nn.Sequential(nn.Linear(ctx, ctx), nn.SiLU(), nn.Linear(ctx, cfg.trivector))

        # Vector grade reshaped into multivectors for the geometric product.
        assert cfg.vector % BLADE_COUNT == 0, "vector grade must be divisible by 8"
        self.n_mv = cfg.vector // BLADE_COUNT  # structural heads

        # Geometric-product result projected back into scalar and bivector grades.
        self.geo_to_scalar = nn.Linear(cfg.vector, cfg.scalar, bias=False)
        self.geo_to_bivector = nn.Linear(cfg.vector, cfg.bivector, bias=False)
        self.gate_scalar = nn.Parameter(torch.zeros(1))
        self.gate_bivector = nn.Parameter(torch.zeros(1))

        if cfg.gated:
            # ReZero gates: one multiplicative scalar per grade update, init 0 ->
            # forward(h) == h exactly at init. (In gated mode gate_scalar/gate_bivector
            # are also used as PLAIN multipliers - no sigmoid - so geo starts closed.)
            self.alpha_scalar = nn.Parameter(torch.zeros(1))
            self.alpha_vector = nn.Parameter(torch.zeros(1))
            self.alpha_bivector = nn.Parameter(torch.zeros(1))
            self.alpha_trivector = nn.Parameter(torch.zeros(1))

    def split(self, h: torch.Tensor):
        b = self.cfg.bounds
        return (
            h[..., b[0]:b[1]],  # scalar
            h[..., b[1]:b[2]],  # vector
            h[..., b[2]:b[3]],  # bivector
            h[..., b[3]:b[4]],  # trivector
            h[..., b[4]:b[5]],  # residual
        )

    def geometric_interaction(self, vector: torch.Tensor):
        """Self geometric product of the vector grade, projected to scalar+bivector.

        Gate semantics differ by mode. Legacy: sigmoid(gate) - NOTE sigmoid(0)=0.5,
        so the gates start HALF-open (kept for reproducibility of past runs).
        Gated: plain multiplier, init 0 -> geo starts fully closed.
        """
        *lead, _ = vector.shape
        mv = vector.reshape(*lead, self.n_mv, BLADE_COUNT)
        prod = geometric_product(mv, mv)  # [..., n_mv, 8]
        # Keep grade-0 and grade-2 parts, flatten back to [..., vector_dim].
        g0 = grade_projection(prod, 0).reshape(*lead, self.cfg.vector)
        g2 = grade_projection(prod, 2).reshape(*lead, self.cfg.vector)
        gs = self.gate_scalar if self.cfg.gated else torch.sigmoid(self.gate_scalar)
        gb = self.gate_bivector if self.cfg.gated else torch.sigmoid(self.gate_bivector)
        return gs * self.geo_to_scalar(g0), gb * self.geo_to_bivector(g2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scalar, vector, bivector, trivector, residual = self.split(h)
        ctx = torch.cat([scalar, vector, bivector, trivector], dim=-1)

        if self.cfg.gated:
            # ReZero update: h + alpha * f(h) per grade, alpha init 0 -> exact identity
            # at init. The model must EARN the grade dynamics through the gates.
            scalar_new = scalar + self.alpha_scalar * self.mlp_scalar(ctx)
            vector_new = vector + self.alpha_vector * self.mlp_vector(ctx)
            bivector_new = bivector + self.alpha_bivector * self.mlp_bivector(ctx)
            trivector_new = trivector + self.alpha_trivector * self.mlp_trivector(ctx)
        else:
            # Legacy update (faults 1-3 documented on GradeConfig.gated; kept verbatim
            # so prior ablation results remain reproducible).
            sm, vm = self.cfg.scalar_momentum, self.cfg.vector_momentum
            scalar_new = sm * scalar + (1 - sm) * self.mlp_scalar(ctx)
            vector_new = vm * vector + (1 - vm) * self.mlp_vector(ctx)
            bivector_new = self.mlp_bivector(ctx)
            trivector_new = self.mlp_trivector(ctx)

        # The geometric-product cross-grade signal — the part the held-out ablation
        # implicates. `geo_interaction=False` keeps grades + momentum but drops it,
        # isolating whether the geometric product specifically is what hurts.
        if self.cfg.geo_interaction:
            geo_scalar, geo_bivector = self.geometric_interaction(vector_new)
            scalar_new = scalar_new + geo_scalar
            bivector_new = bivector_new + geo_bivector

        return torch.cat([scalar_new, vector_new, bivector_new, trivector_new, residual], dim=-1)
