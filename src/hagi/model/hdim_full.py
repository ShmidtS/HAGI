"""Full HDIM — Hypercomplex Domain-Invariant Mapping."""

from __future__ import annotations

import torch
from torch import nn

from .clifford import BLADE_COUNT, GRADE, geometric_product, reverse
from .gdr import GradeConfig, GradeDecomposedRecurrence


class HiddenToMultivector(nn.Module):
    """Project hidden states into Clifford multivector heads."""

    def __init__(self, hidden_size: int, heads: int, blade_count: int = BLADE_COUNT):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.blade_count = blade_count
        self.proj = nn.Linear(hidden_size, heads * blade_count, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        return self.proj(hidden_states).reshape(B, T, self.heads, self.blade_count)


class DomainRotor(nn.Module):
    """Learnable unit even multivectors used for domain rotor sandwiches."""

    def __init__(
        self, num_rotors: int = 4, heads: int = 4, blade_count: int = BLADE_COUNT
    ):
        super().__init__()
        self.num_rotors = num_rotors
        self.heads = heads
        self.blade_count = blade_count

        rotor_params = torch.zeros(num_rotors, heads, blade_count)
        rotor_params[..., 0] = 1.0
        if num_rotors > 1:
            rotor_params[1:] = torch.randn(num_rotors - 1, heads, blade_count) * 0.01
            rotor_params[1:, ..., 0] += 1.0
        self.rotor_params = nn.Parameter(rotor_params)

        even_mask = torch.tensor(
            [1.0 if grade % 2 == 0 else 0.0 for grade in GRADE],
            dtype=torch.float32,
        )
        self.register_buffer("even_mask", even_mask)

    def _normalized_rotors(self) -> torch.Tensor:
        assert isinstance(self.even_mask, torch.Tensor)
        rotors = self.rotor_params * self.even_mask
        norm_sq = (
            geometric_product(rotors, reverse(rotors))[..., :1].abs().clamp_min(1e-8)
        )
        return rotors / torch.sqrt(norm_sq)

    def _select(
        self, rotors: torch.Tensor, rotor_idx: int | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(rotor_idx, torch.Tensor):
            idx = rotor_idx.to(device=rotors.device, dtype=torch.long)
            selected = rotors.index_select(0, idx.reshape(-1))
            return selected.reshape(*idx.shape, self.heads, self.blade_count)
        return rotors[int(rotor_idx)]

    def value(
        self, rotor_idx: int | torch.Tensor = 0, rotors: torch.Tensor | None = None
    ) -> torch.Tensor:
        if rotors is None:
            rotors = self._normalized_rotors()
        return self._select(rotors, rotor_idx)

    def inverse(
        self, rotor_idx: int | torch.Tensor = 0, rotors: torch.Tensor | None = None
    ) -> torch.Tensor:
        if rotors is None:
            rotors = self._normalized_rotors()
        return reverse(self._select(rotors, rotor_idx))

    def _expand_like(
        self, rotor: torch.Tensor, multivector: torch.Tensor
    ) -> torch.Tensor:
        while rotor.dim() < multivector.dim():
            rotor = rotor.unsqueeze(-3)
        return rotor.expand_as(multivector)

    def sandwich(
        self,
        multivector: torch.Tensor,
        rotor_idx: int | torch.Tensor = 0,
        rotors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if rotors is None:
            rotors = self._normalized_rotors()
        rotor = self._expand_like(self._select(rotors, rotor_idx), multivector)
        rotor_inv = self._expand_like(
            reverse(self._select(rotors, rotor_idx)), multivector
        )
        return geometric_product(geometric_product(rotor, multivector), rotor_inv)

    def inverse_sandwich(
        self,
        multivector: torch.Tensor,
        rotor_idx: int | torch.Tensor = 0,
        rotors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if rotors is None:
            rotors = self._normalized_rotors()
        rotor = self._expand_like(self._select(rotors, rotor_idx), multivector)
        rotor_inv = self._expand_like(
            reverse(self._select(rotors, rotor_idx)), multivector
        )
        return geometric_product(geometric_product(rotor_inv, multivector), rotor)

    def forward(
        self, multivector: torch.Tensor, rotor_idx: int | torch.Tensor = 0
    ) -> torch.Tensor:
        return self.sandwich(multivector, rotor_idx)


class InvariantExtractor(nn.Module):
    """Extract source-domain invariant U = R_src^-1 * G * R_src."""

    def forward(
        self,
        multivector: torch.Tensor,
        source_rotor: DomainRotor,
        rotor_idx: int | torch.Tensor = 0,
        rotors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return source_rotor.inverse_sandwich(multivector, rotor_idx, rotors)


class DomainTransfer(nn.Module):
    """Transfer invariant U into target domain: R_tgt * U * R_tgt^-1."""

    def forward(
        self,
        invariant: torch.Tensor,
        target_rotor: DomainRotor,
        rotor_idx: int | torch.Tensor = 0,
        rotors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return target_rotor.sandwich(invariant, rotor_idx, rotors)


class GatedFusion(nn.Module):
    """Fuse transformed multivectors back into the hidden stream."""

    def __init__(self, hidden_size: int, heads: int, blade_count: int = BLADE_COUNT):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.blade_count = blade_count
        mv_size = heads * blade_count
        self.mv_to_hidden = nn.Linear(mv_size, hidden_size)
        self.gate_hidden = nn.Linear(hidden_size, hidden_size)
        self.gate_mv = nn.Linear(hidden_size, hidden_size)
        nn.init.constant_(self.gate_hidden.bias, 0.0)

    def forward(
        self,
        transformed: torch.Tensor,
        hidden_states: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, _, _ = transformed.shape
        mv_hidden = self.mv_to_hidden(
            transformed.reshape(B, T, self.heads * self.blade_count)
        )
        gate = torch.sigmoid(self.gate_hidden(hidden_states) + self.gate_mv(mv_hidden))
        fused = hidden_states + mv_hidden
        output = gate * fused + (1.0 - gate) * hidden_states
        if return_gate:
            return output, gate
        return output


class HDIMFull(nn.Module):
    """Full HDIM pipeline: project -> invariant -> transfer -> gated fusion."""

    def __init__(
        self,
        hidden_size: int,
        heads: int = 4,
        num_rotors: int = 4,
        blade_count: int = BLADE_COUNT,
        use_hdim_cross_domain: bool = True,
        grades: GradeConfig | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.blade_count = blade_count
        self.use_hdim_cross_domain = bool(use_hdim_cross_domain)
        self.project = HiddenToMultivector(hidden_size, heads, blade_count)
        self.rotors = DomainRotor(num_rotors, heads, blade_count)
        self.extract = InvariantExtractor()
        self.transfer = DomainTransfer()
        self.fuse = GatedFusion(hidden_size, heads, blade_count)
        self.gdr = GradeDecomposedRecurrence(grades) if grades is not None else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        src_rotor_idx: int | torch.Tensor = 0,
        tgt_rotor_idx: int | torch.Tensor = 0,
        return_state: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        if not self.use_hdim_cross_domain:
            if return_state:
                return {
                    "fused": hidden_states,
                    "multivector": None,
                    "invariant": None,
                    "target_multivector": None,
                    "target_invariant": None,
                    "gate": None,
                }
            return hidden_states
        multivector = self.project(hidden_states)
        # Normalize all rotors ONCE per forward (was 6× via value/inverse in
        # extract/transfer/extract). Each normalization calls geometric_product
        # on a [num_rotors, heads, 8] stack; reusing one result removes 5 of
        # them — the dominant cost in the GDR/HDIM block.
        rotors = self.rotors._normalized_rotors()
        invariant = self.extract(multivector, self.rotors, src_rotor_idx, rotors)
        target = self.transfer(invariant, self.rotors, tgt_rotor_idx, rotors)
        target_invariant = self.extract(multivector, self.rotors, tgt_rotor_idx, rotors)
        fused, gate = self.fuse(target, hidden_states, return_gate=True)
        if self.gdr is not None:
            fused = self.gdr(fused)
        if return_state:
            return {
                "fused": fused,
                "multivector": multivector,
                "invariant": invariant,
                "target_multivector": target,
                "target_invariant": target_invariant,
                "gate": gate,
            }
        return fused


class DelayedHDIM(HDIMFull):
    """HDIM with Delayed Tensor Parallelism (DTP).

    Accumulates projected multivectors for ``delay_steps`` forward calls
    during training, then extracts the invariant once from the mean of the
    accumulated buffer and performs transfer + fusion.  Between aggregations
    the module returns the hidden state unchanged (identity).

    During evaluation (``self.training == False``) or when ``delay_steps <= 1``
    the module behaves exactly like :class:`HDIMFull`.
    """

    def __init__(
        self,
        hidden_size: int,
        heads: int = 4,
        num_rotors: int = 4,
        blade_count: int = BLADE_COUNT,
        delay_steps: int = 2,
        use_hdim_cross_domain: bool = True,
        grades: GradeConfig | None = None,
    ):
        super().__init__(
            hidden_size,
            heads,
            num_rotors,
            blade_count,
            use_hdim_cross_domain=use_hdim_cross_domain,
            grades=grades,
        )
        self.delay_steps = delay_steps
        self._buffer_sum: torch.Tensor | None = None
        self._buffer_count: int = 0
        self._total_steps: int | None = None
        self.reset_buffer()

    def reset_buffer(self, total_steps: int | None = None) -> None:
        """Clear the circular buffer and optional step budget."""
        self._buffer_sum = None
        self._buffer_count = 0
        self._total_steps = total_steps

    def forward(
        self,
        hidden_states: torch.Tensor,
        src_rotor_idx: int | torch.Tensor = 0,
        tgt_rotor_idx: int | torch.Tensor = 0,
        return_state: bool = False,
        delay_step: int | None = None,
        total_steps: int | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        if not self.training or self.delay_steps <= 1 or delay_step is None:
            return super().forward(
                hidden_states, src_rotor_idx, tgt_rotor_idx, return_state
            )

        if delay_step == 0:
            self.reset_buffer(total_steps)
        if total_steps is not None:
            self._total_steps = total_steps

        multivector = self.project(hidden_states)
        if self._buffer_sum is None:
            self._buffer_sum = multivector.detach().clone()
        else:
            self._buffer_sum += multivector.detach()
        self._buffer_count += 1

        # Determine whether we should aggregate now.
        is_last_step = (
            self._total_steps is not None
            and delay_step == self._total_steps - 1
            and self._buffer_count > 0
        )
        is_full = (delay_step + 1) % self.delay_steps == 0
        should_aggregate = is_last_step or is_full

        if should_aggregate and self._buffer_count > 0:
            assert self._buffer_sum is not None
            agg_mv = self._buffer_sum / self._buffer_count
            self._buffer_sum = None
            self._buffer_count = 0

            rotors = self.rotors._normalized_rotors()
            invariant = self.extract(agg_mv, self.rotors, src_rotor_idx, rotors)
            target = self.transfer(invariant, self.rotors, tgt_rotor_idx, rotors)
            target_invariant = self.extract(agg_mv, self.rotors, tgt_rotor_idx, rotors)
            fused, gate = self.fuse(target, hidden_states, return_gate=True)

            if return_state:
                return {
                    "fused": fused,
                    "multivector": agg_mv,
                    "invariant": invariant,
                    "target_multivector": target,
                    "target_invariant": target_invariant,
                    "gate": gate,
                }
            return fused

        # Delayed phase: return identity (hidden state unchanged).
        if return_state:
            return {
                "fused": hidden_states,
                "multivector": multivector,
                "invariant": None,
                "target_multivector": None,
                "target_invariant": None,
                "gate": None,
            }
        return hidden_states
