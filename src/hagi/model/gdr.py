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
eight matmuls down to two. State-dict keys changed accordingly
(mlp_scalar/... -> grade_trunk/grade_head): checkpoints produced before this
change cannot be loaded into the new layout.
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


class GradeDecomposedRecurrence(nn.Module):
    """One iteration of grade-decomposed update + geometric interaction."""

    def __init__(self, cfg: GradeConfig):
        super().__init__()
        self.cfg = cfg
        ctx = cfg.scalar + cfg.vector + cfg.bivector + cfg.trivector
        self.ctx_size = ctx

        # Shared trunk + single fused head replaces four per-grade MLPs:
        # two matmuls instead of eight, identical receptive field (full ctx).
        # Checkpoints saved with the old per-grade layout are detected in
        # _load_from_state_dict and the legacy modules are rebuilt on the fly.
        self.grade_trunk = nn.Sequential(nn.Linear(ctx, ctx), nn.SiLU())
        self.grade_head = nn.Linear(ctx, ctx)

        # Vector grade reshaped into multivectors for the geometric product.
        assert cfg.vector % BLADE_COUNT == 0, "vector grade must be divisible by 8"
        self.n_mv = cfg.vector // BLADE_COUNT  # structural heads

        # Geometric-product result projected back into scalar and bivector grades.
        self.geo_to_scalar = nn.Linear(cfg.vector, cfg.scalar, bias=False)
        self.geo_to_bivector = nn.Linear(cfg.vector, cfg.bivector, bias=False)
        self.gate_scalar = nn.Parameter(torch.zeros(1))  # type: ignore[reportCallIssue]
        self.gate_bivector = nn.Parameter(torch.zeros(1))  # type: ignore[reportCallIssue]

    def _build_legacy_mlps(self) -> None:
        """Recreate the pre-fusion per-grade MLP layout (for old checkpoints).

        Modules are re-registered in the ORIGINAL order (mlp_* first, then
        geo_to_* and gates) so that parameters() ordering matches the old
        model exactly and optimizer state resumes correctly.
        """
        cfg = self.cfg
        ctx = self.ctx_size
        ref = self.geo_to_scalar.weight
        device, dtype = ref.device, ref.dtype

        geo_s, geo_b = self.geo_to_scalar, self.geo_to_bivector
        gate_s, gate_b = self.gate_scalar, self.gate_bivector
        del self.grade_trunk, self.grade_head
        del self.geo_to_scalar, self.geo_to_bivector
        del self.gate_scalar, self.gate_bivector

        def _mk(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(ctx, ctx), nn.SiLU(), nn.Linear(ctx, out_dim)
            ).to(device=device, dtype=dtype)

        self.mlp_scalar = _mk(cfg.scalar)
        self.mlp_vector = _mk(cfg.vector)
        self.mlp_bivector = _mk(cfg.bivector)
        self.mlp_trivector = _mk(cfg.trivector)
        self.geo_to_scalar = geo_s
        self.geo_to_bivector = geo_b
        self.gate_scalar = gate_s
        self.gate_bivector = gate_b

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        legacy = any(key.startswith(prefix + "mlp_") for key in state_dict)
        if legacy and not hasattr(self, "mlp_scalar"):
            self._build_legacy_mlps()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def split(self, h: torch.Tensor):
        b = self.cfg.bounds
        return (
            h[..., b[0] : b[1]],  # scalar
            h[..., b[1] : b[2]],  # vector
            h[..., b[2] : b[3]],  # bivector
            h[..., b[3] : b[4]],  # trivector
            h[..., b[4] : b[5]],  # residual
        )

    def geometric_interaction(self, vector: torch.Tensor):
        """Self geometric product of the vector grade, projected to scalar+bivector."""
        *lead, _ = vector.shape
        mv = vector.reshape(*lead, self.n_mv, BLADE_COUNT)
        prod = geometric_product(mv, mv)  # [..., n_mv, 8]
        # Keep grade-0 and grade-2 parts, flatten back to [..., vector_dim].
        g0 = grade_projection(prod, 0).reshape(*lead, self.cfg.vector)
        g2 = grade_projection(prod, 2).reshape(*lead, self.cfg.vector)
        scalar_signal = torch.sigmoid(self.gate_scalar) * self.geo_to_scalar(g0)
        bivector_signal = torch.sigmoid(self.gate_bivector) * self.geo_to_bivector(g2)
        return scalar_signal, bivector_signal

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scalar, vector, bivector, trivector, residual = self.split(h)
        graded_ctx = h[..., : self.cfg.bounds[4]]

        if hasattr(self, "mlp_scalar"):
            # Legacy layout (resumed from a pre-fusion checkpoint).
            s_upd = self.mlp_scalar(graded_ctx)
            v_upd = self.mlp_vector(graded_ctx)
            b_upd = self.mlp_bivector(graded_ctx)
            t_upd = self.mlp_trivector(graded_ctx)
        else:
            graded = self.grade_head(self.grade_trunk(graded_ctx))
            s_upd, v_upd, b_upd, t_upd = torch.split(
                graded,
                [
                    self.cfg.scalar,
                    self.cfg.vector,
                    self.cfg.bivector,
                    self.cfg.trivector,
                ],
                dim=-1,
            )

        sm, vm = self.cfg.scalar_momentum, self.cfg.vector_momentum
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
