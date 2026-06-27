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

Implementation note: the per-grade updates use a shared trunk
(Linear(ctx, ctx) + SiLU) followed by a single fused head
(Linear(ctx, scalar+vector+bivector+trivector)) instead of four separate
two-layer MLPs. Each grade still reads the full graded context; this cuts
eight matmuls down to two. Fresh training only (no pre-fusion ckpt compat).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .clifford import BLADE_COUNT, geometric_product_self_g02


@dataclass
class GradeConfig:
    scalar: int = 64
    vector: int = 192
    bivector: int = 192
    trivector: int = 64
    residual: int = 256
    scalar_momentum: float = 0.9
    vector_momentum: float = 0.5
    # Learnable capacity router (MoE-style): a gate over the 4 geometric grades
    # (scalar/vector/bivector/trivector) lets the model self-allocate how much
    # of each forward's update energy flows into entities vs relations vs
    # higher-order structure, instead of the fixed 64/96/96/64 split. The
    # structural grade dims stay (Clifford needs vector % 8 == 0); the router
    # SCALES the per-grade update magnitude, not the dimensions, so the
    # geometric_product math is unchanged. gdr_router=true enables it.
    gdr_router: bool = False
    gdr_router_alpha: float = 0.01
    # Router temperature: divides router logits before softmax. <1 sharpens
    # (stickier capacity allocation), >1 flattens (more uniform/exploration).
    gdr_router_temperature: float = 1.0

    @property
    def hidden_size(self) -> int:
        return (
            self.scalar + self.vector + self.bivector + self.trivector + self.residual
        )

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


class GradeRouter(nn.Module):
    """Learnable capacity gate over the 4 Clifford grades (MoE-style).

    Projects the graded context to 4 logits (one per geometric grade), softmaxes
    them, and returns a per-token gate [B, T, 4] that scales each grade's
    update. This makes the per-grade capacity *trainable*: the model decides
    how much update energy flows into scalar (confidence) vs vector (entities)
    vs bivector (relations) vs trivector (higher-order structure), rather than
    the fixed 64/96/96/64 split.

    The gate scales update MAGNITUDE, not the grade dimensions, so the
    geometric_product (vector x vector -> scalar + bivector) stays structurally
    intact. A Shazeer/Switch load-balance aux loss keeps the gate from
    collapsing onto a single grade.
    """

    def __init__(
        self,
        ctx_size: int,
        num_grades: int = 4,
        alpha: float = 0.01,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_grades = num_grades
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.gate_proj = nn.Linear(ctx_size, num_grades, bias=False)
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.01)

    def forward(
        self, graded_ctx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (gate [.., 4], aux_loss | None).

        aux_loss is the load-balance term (Shazeer/Switch): alpha * N *
        sum_g(fraction_g * mean_prob_g). Computed only in training.
        """
        logits = self.gate_proj(graded_ctx)
        if self.temperature != 1.0:
            logits = logits / self.temperature
        if self.training:
            noise = torch.randn_like(logits) * 0.01
            logits = logits + noise.detach()
        probs = torch.softmax(logits, dim=-1)  # [.., 4]
        aux = None
        if self.training:
            flat = probs.reshape(-1, self.num_grades)
            # fraction per grade: argmax (detached) one-hot mean; mean_prob:
            # mean full-softmax probability (differentiable through gate_proj).
            top_idx = flat.argmax(dim=-1)
            one_hot = torch.zeros_like(flat)
            one_hot.scatter_(1, top_idx.unsqueeze(-1), 1.0)
            fraction = one_hot.mean(dim=0).detach()
            mean_prob = flat.mean(dim=0)
            aux = self.alpha * float(self.num_grades) * (fraction * mean_prob).sum()
        return probs, aux


class GradeDecomposedRecurrence(nn.Module):
    """One iteration of grade-decomposed update + geometric interaction."""

    def __init__(self, cfg: GradeConfig):
        super().__init__()
        self.cfg = cfg
        ctx = cfg.scalar + cfg.vector + cfg.bivector + cfg.trivector
        self.ctx_size = ctx
        self._bounds = cfg.bounds
        self._split_sizes = [cfg.scalar, cfg.vector, cfg.bivector, cfg.trivector]

        # Shared trunk + single fused head replaces four per-grade MLPs:
        # two matmuls instead of eight, identical receptive field (full ctx).
        # Fresh training only — no per-grade-MLP checkpoint compatibility.
        self.grade_trunk = nn.Sequential(nn.Linear(ctx, ctx), nn.SiLU())
        self.grade_head = nn.Linear(ctx, ctx)

        # Learnable grade momentum (plan 1.2, Muon-style): the scalar/vector
        # update rates are learned per-grade instead of fixed config constants.
        # Stored as pre-sigmoid logits, initialized so sigmoid(logit) == the
        # config momentum (logit = log(m/(1-m))); zero-momentum (bivector/
        # trivector) stays a hard full-update (no param). The model can drift
        # each grade's speed as training progresses — the core GDR hypothesis
        # that distinct grades converge on different timescales.
        def _mom_logit(m: float) -> float:
            m = min(max(m, 1e-4), 1 - 1e-4)
            return math.log(m / (1 - m))

        self.scalar_mom_logit = nn.Parameter(torch.tensor(_mom_logit(cfg.scalar_momentum)))
        self.vector_mom_logit = nn.Parameter(torch.tensor(_mom_logit(cfg.vector_momentum)))

        # Vector grade reshaped into multivectors for the geometric product.
        assert cfg.vector % BLADE_COUNT == 0, "vector grade must be divisible by 8"
        self.n_mv = cfg.vector // BLADE_COUNT  # structural heads

        # Geometric-product result projected back into scalar and bivector grades.
        self.geo_to_scalar = nn.Linear(cfg.vector, cfg.scalar, bias=False)
        self.geo_to_bivector = nn.Linear(cfg.vector, cfg.bivector, bias=False)
        self.gate_scalar = nn.Parameter(torch.zeros(1))  # type: ignore[reportCallIssue]
        self.gate_bivector = nn.Parameter(torch.zeros(1))  # type: ignore[reportCallIssue]

        # Learnable capacity router over the 4 geometric grades. Scales the
        # per-grade update magnitude; None when gdr_router=false (legacy fixed
        # capacity). Built only when enabled so checkpoints without it load.
        self.grade_router: GradeRouter | None = (
            GradeRouter(
                ctx,
                num_grades=4,
                alpha=cfg.gdr_router_alpha,
                temperature=cfg.gdr_router_temperature,
            )
            if getattr(cfg, "gdr_router", False)
            else None
        )
        # Last forward's router load-balance aux (None when no router / eval).
        # Read by the training loop to fold into the composite loss.
        self.last_router_aux: torch.Tensor | None = None

    def split(self, h: torch.Tensor):
        b = self._bounds
        return (
            h[..., b[0] : b[1]],  # scalar
            h[..., b[1] : b[2]],  # vector
            h[..., b[2] : b[3]],  # bivector
            h[..., b[3] : b[4]],  # trivector
            h[..., b[4] : b[5]],  # residual
        )

    def geometric_interaction(self, vector: torch.Tensor):
        """Self geometric product of the vector grade, projected to scalar+bivector.

        Uses geometric_product_self_g02: computes only grade-0 and grade-2
        output blades (4 of 8) in a single fused einsum, ~50% less compute
        than the full geometric_product + grade_projection path. Mathematically
        identical."""
        *lead, _ = vector.shape
        mv = vector.reshape(*lead, self.n_mv, BLADE_COUNT)
        g0_raw, g2_raw = geometric_product_self_g02(mv)
        g0 = g0_raw.reshape(*lead, self.cfg.vector)
        g2 = g2_raw.reshape(*lead, self.cfg.vector)
        scalar_signal = torch.sigmoid(self.gate_scalar) * self.geo_to_scalar(g0)
        bivector_signal = torch.sigmoid(self.gate_bivector) * self.geo_to_bivector(g2)
        return scalar_signal, bivector_signal

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scalar, vector, bivector, trivector, residual = self.split(h)
        graded_ctx = h[..., : self._bounds[4]]

        graded = self.grade_head(self.grade_trunk(graded_ctx))
        s_upd, v_upd, b_upd, t_upd = torch.split(
            graded,
            self._split_sizes,
            dim=-1,
        )

        # Learnable capacity gate: scale each grade's UPDATE by a per-token
        # softmax weight so the model self-allocates update energy across
        # scalar/vector/bivector/trivector. The grade dimensions are untouched
        # (Clifford needs vector % 8 == 0), only the update magnitude is gated,
        # so geometric_interaction stays valid. Gate is per-token [B, T, 4];
        # broadcast over the grade's last dim. Skip when no router (legacy).
        if self.grade_router is not None:
            gate, gdr_router_aux = self.grade_router(graded_ctx)
            # gate[..., g] is [.., 1] after unsqueeze; multiply each grade upd.
            s_upd = s_upd * gate[..., 0:1]
            v_upd = v_upd * gate[..., 1:2]
            b_upd = b_upd * gate[..., 2:3]
            t_upd = t_upd * gate[..., 3:4]
            # Expose the load-balance aux so the outer loop can add it to the
            # composite loss. Read via .last_router_aux after forward; cleared
            # each call so stale values never leak across forwards.
            self.last_router_aux = gdr_router_aux
        else:
            self.last_router_aux = None

        # Learnable momentum via sigmoid(logits); bivector/trivector stay full-update.
        sm = torch.sigmoid(self.scalar_mom_logit)
        vm = torch.sigmoid(self.vector_mom_logit)
        scalar_new = sm * scalar + (1 - sm) * s_upd
        vector_new = vm * vector + (1 - vm) * v_upd
        bivector_new = b_upd
        trivector_new = t_upd

        geo_scalar, geo_bivector = self.geometric_interaction(vector_new)
        scalar_new = scalar_new + geo_scalar
        bivector_new = bivector_new + geo_bivector

        return torch.cat(
            [scalar_new, vector_new, bivector_new, trivector_new, residual], dim=-1
        )
