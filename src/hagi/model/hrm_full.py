"""Hierarchical Recurrent Model reasoning controller."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .transformer import RMSNorm, TransformerBlock

if TYPE_CHECKING:
    from .msa import MSAMemory, SlotRegistry


@dataclass
class HState:
    z_H: torch.Tensor


@dataclass
class LState:
    z_L: torch.Tensor


class _FusedGatedTransition(nn.Module):
    """Fused up+gate projection for gated transitions.

    Replaces separate nn.Linear(in, mult*out) + nn.Linear(in, out) with a
    single nn.Linear(in, mult*out + out), halving kernel launches. The output
    is split into [up_part, gate_part] after a single matmul. State_dict
    splits the fused weight back to up.weight/up.bias + gate.weight/gate.bias
    for checkpoint portability.

    The parameter is named ``proj`` (not ``up_gate``) so HAGI's residual-scale
    loop does NOT exclude it via the "gate" token — the up portion needs
    1/sqrt(2L) scaling. The gate portion also gets scaled (minor: makes the
    sigmoid slightly smoother at init, model adapts during training).
    """

    def __init__(self, in_dim: int, out_dim: int, mult: int = 2):
        super().__init__()
        self._up_size = mult * out_dim
        self._gate_size = out_dim
        self._splits = [self._up_size, self._gate_size]
        self.proj = nn.Linear(in_dim, self._up_size + self._gate_size)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        prefix = kwargs.get("prefix", "")
        w = self.proj.weight.data
        b = self.proj.bias.data if self.proj.bias is not None else None
        up_w, gate_w = w.split(self._splits, dim=0)
        state[f"{prefix}up.weight"] = up_w
        state[f"{prefix}gate.weight"] = gate_w
        if b is not None:
            up_b, gate_b = b.split(self._splits, dim=0)
            state[f"{prefix}up.bias"] = up_b
            state[f"{prefix}gate.bias"] = gate_b
        state.pop(f"{prefix}proj.weight", None)
        state.pop(f"{prefix}proj.bias", None)
        return state

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ug = self.proj(x)
        up_part, gate_part = ug.split(self._splits, dim=-1)
        return up_part, gate_part


class HTransition(nn.Module):
    def __init__(self, h_dim: int, l_dim: int, mult: int = 2, dropout: float = 0.0):
        super().__init__()
        in_dim = h_dim + l_dim
        self.norm = RMSNorm(in_dim, eps=1e-6)
        self.fused_proj = _FusedGatedTransition(in_dim, h_dim, mult)
        self.act = nn.SiLU()
        self.down = nn.Linear(mult * h_dim, h_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, z_H_prev: torch.Tensor, z_L_last: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_H_prev, z_L_last], dim=-1)
        x = self.norm(x)
        up_part, gate_part = self.fused_proj(x)
        h = self.dropout(self.down(self.act(up_part)))
        g = torch.sigmoid(gate_part)
        return z_H_prev + g * h


class ResetL(nn.Module):
    def __init__(self, h_dim: int, l_dim: int, mult: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(h_dim, eps=1e-6)
        self.fused_proj = _FusedGatedTransition(h_dim, l_dim, mult)
        self.act = nn.SiLU()
        self.down = nn.Linear(mult * l_dim, l_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, z_H: torch.Tensor) -> torch.Tensor:
        x = self.norm(z_H)
        up_part, gate_part = self.fused_proj(x)
        h = self.dropout(self.down(self.act(up_part)))
        g = torch.sigmoid(gate_part)
        return g * h + (1 - g) * x


class AttentionPool(nn.Module):
    """Attention-based pooling of a token sequence into a single vector.

    Replaces ``transformer_output.mean(dim=1)`` in LTransition: mean pooling
    loses positional information and weights every token equally. A learned
    query attends over the sequence so the pooled z_L keeps the tokens most
    relevant to the recurrent state (better long-range information retention
    in z_L). ~50K extra params (hidden_size query + scalar score proj).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.scale = hidden_size ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H] -> [B, H]
        # Single-query attention: q [B, 1, H], k=v=x [B, T, H]
        # SDPA with 1 query head, q_len=1, kv_len=T — flash kernel handles
        # the softmax+matmul fusion efficiently for this pattern.
        B, T, H = x.shape
        q = self.query.expand(B, 1, H)  # [B, 1, H]
        # SDPA expects [B, n_heads, q_len, head_dim] — single head, so
        # reshape to [B, 1, 1, H].
        q_4d = q.unsqueeze(1)  # [B, 1, 1, H]
        k_4d = x.unsqueeze(1)  # [B, 1, T, H]
        v_4d = x.unsqueeze(1)  # [B, 1, T, H]
        out = F.scaled_dot_product_attention(
            q_4d, k_4d, v_4d, attn_mask=None, is_causal=False,
            scale=self.scale,
        )  # [B, 1, 1, H]
        return out.squeeze(1).squeeze(1)  # [B, H]


class LTransition(nn.Module):
    def __init__(
        self,
        l_dim: int,
        hidden_size: int,
        h_dim: int | None = None,
        mult: int = 2,
        dropout: float = 0.0,
        h_cycles: int = 2,
    ):
        super().__init__()
        in_dim = l_dim + hidden_size
        self.norm = RMSNorm(in_dim, eps=1e-6)
        self.fused_proj = _FusedGatedTransition(in_dim, l_dim, mult)
        self.act = nn.SiLU()
        self.down = nn.Linear(mult * l_dim, l_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Attention pooling preserves positional/long-range info that mean
        # pooling discards. Fresh training only — no legacy mean-pool fallback.
        self.pool = AttentionPool(hidden_size)
        if h_cycles > 1:
            self.reset_l = ResetL(
                h_dim if h_dim is not None else hidden_size, l_dim, mult, dropout
            )
        else:
            self.reset_l = None

    def forward(
        self, z_L_prev: torch.Tensor, transformer_output: torch.Tensor
    ) -> torch.Tensor:
        pooled = self.pool(transformer_output)
        x = torch.cat([z_L_prev, pooled], dim=-1)
        x = self.norm(x)
        up_part, gate_part = self.fused_proj(x)
        h = self.dropout(self.down(self.act(up_part)))
        g = torch.sigmoid(gate_part)
        return z_L_prev + g * h

    def reset(self, z_H: torch.Tensor) -> torch.Tensor:
        if self.reset_l is None:
            return z_H
        return self.reset_l(z_H)


class HRMCore(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        h_dim: int = 256,
        l_dim: int = 256,
        h_cycles: int = 2,
        l_cycles: int = 3,
        transition_mult: int = 2,
        transition_dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.h_cycles = h_cycles
        self.l_cycles = l_cycles

        self.h_init = nn.Linear(hidden_size, h_dim)
        self.l_init = nn.Linear(hidden_size, l_dim)
        # Learnable initial state baseline (plan 1.5): a data-independent anchor
        # added to the pooled init so the recurrent state has a stable starting
        # point the model can learn, instead of being purely a function of the
        # first forward's pooled hidden. Zero-init so fresh training starts at
        # the pooled-init behavior and the baseline grows in as it learns.
        self.z_h_init = nn.Parameter(torch.zeros(1, 1, h_dim))
        self.z_l_init = nn.Parameter(torch.zeros(1, 1, l_dim))
        self.h_transition = (
            HTransition(h_dim, l_dim, transition_mult, transition_dropout)
            if h_cycles > 1
            else None
        )
        self.l_transition = LTransition(
            l_dim,
            hidden_size,
            h_dim,
            transition_mult,
            transition_dropout,
            h_cycles=h_cycles,
        )
        self.z_l_to_hidden = nn.Linear(l_dim, hidden_size)
        self.z_h_to_hidden = nn.Linear(h_dim, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        reasoning_blocks: Sequence[TransformerBlock],
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask=None,
        z_H: torch.Tensor | HState | None = None,
        z_L: torch.Tensor | LState | None = None,
        gdr=None,
        training_mode: bool = False,
        gradient_checkpointing: bool = False,
        tgt_rotor_idx: int | torch.Tensor = 0,
        moe_aux_losses: list[torch.Tensor] | None = None,
        nars_controller=None,
        noise_sigma: float = 0.0,
        msa_memory: Optional["MSAMemory"] = None,
        msa_registry: Optional["SlotRegistry"] = None,
        hrm_memory_aware: bool = False,
        gc_group_size: int = 1,
        stochastic_depth: float = 0.0,
    ):
        h = hidden_states
        B, T, H = h.shape
        pooled = h.mean(dim=1)

        if isinstance(z_H, HState):
            z_H = z_H.z_H
        if isinstance(z_L, LState):
            z_L = z_L.z_L
        if z_H is None:
            z_H = self.h_init(pooled) + self.z_h_init.expand(B, -1, -1).squeeze(1)
        if z_L is None:
            z_L = self.l_init(pooled) + self.z_l_init.expand(B, -1, -1).squeeze(1)
        # Past the branches above z_H/z_L are always tensors, but the
        # parameter is typed Tensor|HState|None, so narrow with cast for the
        # dtype/device access and the gate scaling below.
        z_H = cast(torch.Tensor, z_H)
        z_L = cast(torch.Tensor, z_L)
        assert isinstance(z_H, torch.Tensor)
        assert isinstance(z_L, torch.Tensor)

        # NARS truth-weighted gating — active controller
        if nars_controller is not None:
            h_gate, l_gate = nars_controller.compute_gating(z_H, z_L)
            # Cast scalar gates to the state dtype so z_H/z_L stay bf16 (a
            # plain `tensor * python_float` would promote to float32 and force
            # the non-fused rms_norm fallback downstream).
            z_H = z_H * torch.as_tensor(h_gate, dtype=z_H.dtype, device=z_H.device)
            z_L = z_L * torch.as_tensor(l_gate, dtype=z_L.dtype, device=z_L.device)

        # Memory-aware HRM: bind the NARS MSA reasoner onto the bridge (if any)
        # so the intra-cycle read path can blend NARS beliefs when use_nars=true.
        msa_lb_loss: torch.Tensor | None = None
        # Track the slot_id base across writes so successive cycles register
        # disjoint ids (registry dedups by id, but disjoint ids keep the
        # eviction order meaningful and avoid the in-place overwrite path).
        # Defined unconditionally (0 when memory-aware is off) so the write
        # guard never reads an unbound name.
        slot_id_base = 0
        mem_aware = (
            hrm_memory_aware and msa_memory is not None and msa_registry is not None
        )

        gdr_state = None
        pre_gdr_h = None
        block_list = list(reasoning_blocks)
        active_l_cycles = self.l_cycles
        if (
            training_mode
            and stochastic_depth > 0.0
            and self.l_cycles > 1
            and torch.rand(1).item() < stochastic_depth
        ):
            active_l_cycles = 1
        for h_cycle in range(self.h_cycles):
            if training_mode and noise_sigma > 0.0:
                z_H = z_H + torch.randn_like(z_H) * noise_sigma
            # z_H is invariant across all l_cycles within an h_cycle (it only
            # updates after the l-cycle loop via h_transition). Lift its
            # projection out so it runs once per h_cycle instead of l_cycles.
            h_term = self.z_h_to_hidden(z_H).unsqueeze(1)
            for l_cycle in range(active_l_cycles):
                # Loop reassignments can widen the inferred type back to the
                # parameter union; narrow before each use.
                z_L = cast(torch.Tensor, z_L)
                if training_mode and noise_sigma > 0.0:
                    z_L = z_L + torch.randn_like(z_L) * noise_sigma

                # --- Memory-aware READ: attend over what prior cycles wrote.
                # First cycle (or empty registry) -> no-op, pure reasoning.
                # Gradient-checkpoint the read when enabled: the MSA sparse
                # attention activation (qk scores, attn weights) is the main
                # intra-cycle VRAM cost; recomputing it in backward trades one
                # extra fwd for a large activation-memory saving. The load-
                # balance loss is detached-counter-based (no grad through it),
                # so recomputation reproduces it exactly.
                if mem_aware:
                    assert msa_memory is not None and msa_registry is not None
                    # NOTE: do NOT gradient-checkpoint msa_memory.read here. The
                    # registry is STATEFUL (the write below appends slots every
                    # cycle), so between the forward call and the backward
                    # recompute the registry size differs -> read fetches K/V of
                    # a different shape -> checkpoint's saved-vs-recomputed
                    # metadata mismatch aborts backward. The read is kept eager;
                    # VRAM is bounded instead by the per-cycle activation being
                    # small (chunked MSA) and by whole-model grad checkpointing
                    # on the reasoning blocks around it.
                    msa_out, lb, _r_ids, _r_scores = msa_memory.read(
                        h, msa_registry, training_mode=training_mode
                    )
                    h = h + msa_out
                    if lb is not None:
                        msa_lb_loss = lb

                bias = h_term + self.z_l_to_hidden(z_L).unsqueeze(1)
                if gradient_checkpointing and gc_group_size > 1 and len(block_list) > 1:
                    # Group checkpointing: wrap consecutive reasoning blocks in one
                    # checkpoint call -> halves recompute passes in backward
                    # (14 -> 7 for 7 blocks x 2 l_cycles), the HRM hot path.
                    for gi in range(0, len(block_list), gc_group_size):
                        group = block_list[gi : gi + gc_group_size]

                        def run_group(h_in, _blocks=group, _bias=bias):
                            hh = h_in
                            _aux_local: list[torch.Tensor] = []
                            for _b in _blocks:
                                res = _b(hh + _bias, cos, sin, attn_mask=attn_mask)
                                if (
                                    isinstance(res, tuple)
                                    and len(res) == 2
                                    and isinstance(res[1], torch.Tensor)
                                    and res[1].ndim == 0
                                ):
                                    hh = res[0]
                                    _aux_local.append(res[1])
                                else:
                                    hh = res
                            if moe_aux_losses is not None:
                                for _a in _aux_local:
                                    moe_aux_losses.append(_a.detach())
                            return hh

                        h = checkpoint(run_group, h, use_reentrant=False)
                else:
                    for block in block_list:
                        h_in = h + bias
                        if gradient_checkpointing:
                            result = checkpoint(
                                block,
                                h_in,
                                cos,
                                sin,
                                attn_mask=attn_mask,
                                use_reentrant=False,
                            )
                        else:
                            result = block(h_in, cos, sin, attn_mask=attn_mask)
                        if (
                            isinstance(result, tuple)
                            and len(result) == 2
                            and isinstance(result[1], torch.Tensor)
                            and result[1].ndim == 0
                        ):
                            h = result[0]
                            if moe_aux_losses is not None:
                                moe_aux_losses.append(result[1])
                        else:
                            h = result
                assert isinstance(h, torch.Tensor)
                if gdr is not None:
                    current_step = h_cycle * active_l_cycles + l_cycle
                    total_steps = self.h_cycles * active_l_cycles
                    if (
                        training_mode
                        and hasattr(gdr, "delay_steps")
                        and gdr.delay_steps > 1
                    ):
                        pre_gdr_h = h
                        gdr_state = gdr(
                            h,
                            src_rotor_idx=0,
                            tgt_rotor_idx=tgt_rotor_idx,
                            return_state=True,
                            delay_step=current_step,
                            total_steps=total_steps,
                        )
                        h = gdr_state["fused"]
                    else:
                        h = gdr(h)

                # --- Memory-aware WRITE: register the refined hidden as slots
                # so the next cycle (and the post-HRM read) can retrieve it.
                if mem_aware:
                    assert msa_memory is not None and msa_registry is not None
                    ids, _keys = msa_memory.write(h, msa_registry, slot_id_base)
                    slot_id_base = slot_id_base + int(ids.shape[0])

                z_L = self.l_transition(z_L, h)
            if self.h_cycles > 1:
                assert self.h_transition is not None
                z_H = self.h_transition(z_H, z_L)
                assert isinstance(z_H, torch.Tensor)
                z_L = self.l_transition.reset(z_H)

        assert isinstance(z_L, torch.Tensor)
        assert isinstance(z_H, torch.Tensor)
        return h, HState(z_H), LState(z_L), gdr_state, pre_gdr_h, msa_lb_loss
