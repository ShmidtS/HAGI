"""Full HDIM — Hypercomplex Domain-Invariant Mapping."""

from __future__ import annotations

import torch
from torch import nn

from .clifford import BLADE_COUNT, GRADE, geometric_product, reverse


class HiddenToMultivector(nn.Module):
    """Project hidden states into Clifford multivector heads."""

    def __init__(self, hidden_size: int, heads: int, blade_count: int = BLADE_COUNT):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.blade_count = blade_count
        self.proj = nn.Linear(hidden_size, heads * blade_count)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        return self.proj(hidden_states).reshape(B, T, self.heads, self.blade_count)


class DomainRotor(nn.Module):
    """Learnable unit even multivectors used for domain rotor sandwiches."""

    def __init__(self, num_rotors: int = 4, heads: int = 4, blade_count: int = BLADE_COUNT):
        super().__init__()
        self.num_rotors = num_rotors
        self.heads = heads
        self.blade_count = blade_count

        rotor_params = torch.zeros(num_rotors, heads, blade_count)
        rotor_params[..., 0] = 1.0
        self.rotor_params = nn.Parameter(rotor_params)

        even_mask = torch.tensor(
            [1.0 if grade % 2 == 0 else 0.0 for grade in GRADE],
            dtype=torch.float32,
        )
        self.register_buffer("even_mask", even_mask)

    def _normalized_rotors(self) -> torch.Tensor:
        rotors = self.rotor_params * self.even_mask
        norm_sq = geometric_product(rotors, reverse(rotors))[..., :1].abs().clamp_min(1e-8)
        return rotors / torch.sqrt(norm_sq)

    def value(self, rotor_idx: int | torch.Tensor = 0) -> torch.Tensor:
        rotors = self._normalized_rotors()
        if isinstance(rotor_idx, torch.Tensor):
            idx = rotor_idx.to(device=rotors.device, dtype=torch.long)
            selected = rotors.index_select(0, idx.reshape(-1))
            return selected.reshape(*idx.shape, self.heads, self.blade_count)
        return rotors[int(rotor_idx)]

    def inverse(self, rotor_idx: int | torch.Tensor = 0) -> torch.Tensor:
        return reverse(self.value(rotor_idx))

    def _expand_like(self, rotor: torch.Tensor, multivector: torch.Tensor) -> torch.Tensor:
        while rotor.dim() < multivector.dim():
            rotor = rotor.unsqueeze(-3)
        return rotor.expand_as(multivector)

    def sandwich(self, multivector: torch.Tensor, rotor_idx: int | torch.Tensor = 0) -> torch.Tensor:
        rotor = self._expand_like(self.value(rotor_idx), multivector)
        rotor_inv = self._expand_like(self.inverse(rotor_idx), multivector)
        return geometric_product(geometric_product(rotor, multivector), rotor_inv)

    def inverse_sandwich(self, multivector: torch.Tensor, rotor_idx: int | torch.Tensor = 0) -> torch.Tensor:
        rotor = self._expand_like(self.value(rotor_idx), multivector)
        rotor_inv = self._expand_like(self.inverse(rotor_idx), multivector)
        return geometric_product(geometric_product(rotor_inv, multivector), rotor)

    def forward(self, multivector: torch.Tensor, rotor_idx: int | torch.Tensor = 0) -> torch.Tensor:
        return self.sandwich(multivector, rotor_idx)


class InvariantExtractor(nn.Module):
    """Extract source-domain invariant U = R_src^-1 * G * R_src."""

    def forward(
        self,
        multivector: torch.Tensor,
        source_rotor: DomainRotor,
        rotor_idx: int | torch.Tensor = 0,
    ) -> torch.Tensor:
        return source_rotor.inverse_sandwich(multivector, rotor_idx)


class DomainTransfer(nn.Module):
    """Transfer invariant U into target domain: R_tgt * U * R_tgt^-1."""

    def forward(
        self,
        invariant: torch.Tensor,
        target_rotor: DomainRotor,
        rotor_idx: int | torch.Tensor = 0,
    ) -> torch.Tensor:
        return target_rotor.sandwich(invariant, rotor_idx)


class GatedFusion(nn.Module):
    """Fuse transformed multivectors back into the hidden stream."""

    def __init__(self, hidden_size: int, heads: int, blade_count: int = BLADE_COUNT):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.blade_count = blade_count
        mv_size = heads * blade_count
        self.mv_to_hidden = nn.Linear(mv_size, hidden_size)
        self.gate = nn.Linear(hidden_size + hidden_size, hidden_size)

    def forward(
        self,
        transformed: torch.Tensor,
        hidden_states: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, _, _ = transformed.shape
        mv_hidden = self.mv_to_hidden(transformed.reshape(B, T, self.heads * self.blade_count))
        gate = torch.sigmoid(self.gate(torch.cat([hidden_states, mv_hidden], dim=-1)))
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
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.blade_count = blade_count
        self.project = HiddenToMultivector(hidden_size, heads, blade_count)
        self.rotors = DomainRotor(num_rotors, heads, blade_count)
        self.extract = InvariantExtractor()
        self.transfer = DomainTransfer()
        self.fuse = GatedFusion(hidden_size, heads, blade_count)

    def forward(
        self,
        hidden_states: torch.Tensor,
        src_rotor_idx: int | torch.Tensor = 0,
        tgt_rotor_idx: int | torch.Tensor = 0,
        return_state: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        multivector = self.project(hidden_states)
        invariant = self.extract(multivector, self.rotors, src_rotor_idx)
        target = self.transfer(invariant, self.rotors, tgt_rotor_idx)
        target_invariant = self.extract(target, self.rotors, tgt_rotor_idx)
        fused, gate = self.fuse(target, hidden_states, return_gate=True)
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
