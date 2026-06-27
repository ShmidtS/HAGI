"""Mixture of Experts (MoE) SwiGLU module."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .binary_factorized import BinaryFactorizedLinear


def _make_linear(in_features: int, out_features: int, cfg: Any) -> nn.Module:
    if getattr(cfg, "use_binary_factorized", False):
        return BinaryFactorizedLinear(
            in_features, out_features, getattr(cfg, "binary_factorized_rank", 8)
        )
    return nn.Linear(in_features, out_features, bias=False)


class _SwiGLUExpert(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        intermediate_size = cfg.moe_intermediate_size or (
            cfg.intermediate_size // cfg.num_experts
        )
        self.hidden_size = cfg.hidden_size
        self.intermediate_size = intermediate_size
        self._use_bf = getattr(cfg, "use_binary_factorized", False)
        if self._use_bf:
            self.gate = _make_linear(cfg.hidden_size, intermediate_size, cfg)
            self.up = _make_linear(cfg.hidden_size, intermediate_size, cfg)
        else:
            gate_w = _make_linear(cfg.hidden_size, intermediate_size, cfg)
            up_w = _make_linear(cfg.hidden_size, intermediate_size, cfg)
            assert isinstance(gate_w, nn.Linear)
            assert isinstance(up_w, nn.Linear)
            # Named gu_weight (not gate_up_weight) to avoid the "gate" token
            # in HAGI's residual-scale exclude list and optim._is_muon_param.
            # Without this, the fused weight would be excluded from Muon
            # orthogonalization and residual scaling — changing training dynamics.
            self.gu_weight = nn.Parameter(
                torch.cat([gate_w.weight, up_w.weight], dim=0).contiguous()
            )
        self.down = _make_linear(intermediate_size, cfg.hidden_size, cfg)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        prefix = kwargs.get("prefix", "")
        if not self._use_bf and hasattr(self, "gu_weight"):
            gate, up = self.gu_weight.chunk(2, dim=0)
            state[f"{prefix}gate.weight"] = gate
            state[f"{prefix}up.weight"] = up
            state.pop(f"{prefix}gu_weight", None)
        return state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_bf:
            return self.down(F.silu(self.gate(x)) * self.up(x))
        gu = F.linear(x, self.gu_weight)
        gate, up = gu.chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class MoESwiGLU(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.moe_top_k
        self.hidden_size = cfg.hidden_size
        self.intermediate_size = cfg.moe_intermediate_size or (
            cfg.intermediate_size // cfg.num_experts
        )
        self.router_temperature = cfg.moe_router_temperature
        self.alpha = getattr(cfg, "moe_alpha", 0.01)
        # Mixture-of-Depths (plan 4.2): an extra "skip" router slot whose
        # selected token bypasses the experts (output 0 -> residual identity).
        # Trivial tokens skip the MLP compute. The skip slot is the LAST router
        # logit; when it wins the topk the token gets zero expert output.
        self.use_mod_skip = bool(getattr(cfg, "moe_mod_skip", False))
        router_out = cfg.num_experts + (1 if self.use_mod_skip else 0)
        self.skip_idx = cfg.num_experts if self.use_mod_skip else -1

        self.router = nn.Linear(cfg.hidden_size, router_out, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
        self.experts = nn.ModuleList(_SwiGLUExpert(cfg) for _ in range(cfg.num_experts))

    def _forward_impl(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        assert self.hidden_size == D

        flat = x.view(B * T, D)
        router_logits = self.router(flat)
        if self.training:
            noise = torch.randn_like(router_logits) * 0.01
            router_logits = router_logits + noise.detach()
        if self.router_temperature != 1.0:
            router_logits = router_logits / self.router_temperature
        # router_logits = router_logits.clamp(-10, 10)  # removed for speed
        router_probs = F.softmax(router_logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.top_k > 1:
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Sparse dispatch: only compute tokens that actually route to each expert
        output = torch.zeros_like(flat)
        for k_idx in range(self.top_k):
            expert_idx = top_k_indices[:, k_idx]  # [B*T]
            probs = top_k_probs[:, k_idx]  # [B*T]
            if self.top_k == 1:
                # Fast path: sort by expert index for contiguous dispatch and fewer
                # kernel launches (one sort instead of num_experts where+index_select).
                # MoD: tokens routed to the skip slot keep output 0 (residual
                # identity) and are excluded from expert dispatch entirely.
                if self.use_mod_skip:
                    nonskip = expert_idx != self.skip_idx
                    if not nonskip.any():
                        # all tokens skip -> no expert compute, output stays 0
                        pass
                    else:
                        keep = torch.where(nonskip)[0]
                        expert_idx = expert_idx[keep]
                        probs = probs[keep]
                        sorted_experts, sort_order = torch.sort(expert_idx)
                        sorted_tokens = flat[keep][sort_order]
                        unique_experts, counts = torch.unique_consecutive(
                            sorted_experts, return_counts=True
                        )
                        offset = 0
                        for e_idx, count in zip(unique_experts.tolist(), counts.tolist()):
                            expert = self.experts[e_idx]
                            slice_tokens = sorted_tokens[offset : offset + count]
                            expert_out = expert(slice_tokens)
                            # map back: sort_order -> keep index -> flat index
                            flat_indices = keep[sort_order[offset : offset + count]]
                            if expert_out.dtype != output.dtype:
                                expert_out = expert_out.to(output.dtype)
                            output.index_copy_(0, flat_indices, expert_out)
                            offset += count
                else:
                    sorted_experts, sort_order = torch.sort(expert_idx)
                    sorted_tokens = flat[sort_order]
                    unique_experts, counts = torch.unique_consecutive(
                        sorted_experts, return_counts=True
                    )
                    offset = 0
                    for e_idx, count in zip(unique_experts.tolist(), counts.tolist()):
                        expert = self.experts[e_idx]
                        slice_tokens = sorted_tokens[offset : offset + count]
                        expert_out = expert(slice_tokens)
                        out_indices = sort_order[offset : offset + count]
                        if expert_out.dtype != output.dtype:
                            expert_out = expert_out.to(output.dtype)
                        output.index_copy_(0, out_indices, expert_out)
                        offset += count
            else:
                for e_idx, expert in enumerate(self.experts):
                    mask = expert_idx == e_idx
                    indices = torch.where(mask)[0]
                    if indices.numel() == 0:
                        continue
                    tokens = flat.index_select(0, indices)
                    expert_out = expert(tokens)
                    idx = indices.unsqueeze(-1).expand(-1, expert_out.size(-1))
                    if expert_out.dtype != output.dtype:
                        expert_out = expert_out.to(output.dtype)
                    output.scatter_add_(
                        0,
                        idx,
                        expert_out * probs.index_select(0, indices).unsqueeze(-1),
                    )

        output = output.view(B, T, D)

        if self.training:
            # Aux load-balance over REAL experts only (skip slot excluded).
            # router_probs has num_experts+1 columns when MoD is on; slice off
            # the skip column. top_k_indices may contain skip_idx — clamp to a
            # valid expert and zero those rows' contribution so skip-routed
            # tokens don't bias the balance loss.
            real_probs = router_probs[:, : self.num_experts]
            router_prob_per_expert = real_probs.mean(dim=0)
            top_k_mask = torch.zeros(
                B * T, self.num_experts, device=x.device, dtype=router_probs.dtype
            )
            if self.use_mod_skip:
                nonskip_sel = (top_k_indices != self.skip_idx)
                safe_idx = top_k_indices.clamp(max=self.num_experts - 1)
                top_k_mask.scatter_(1, safe_idx, 1.0)
                top_k_mask = top_k_mask * nonskip_sel.float()
            else:
                top_k_mask.scatter_(1, top_k_indices, 1.0)
            fraction_per_expert = top_k_mask.mean(dim=0)
            aux_loss = (
                self.alpha
                * self.num_experts
                * (fraction_per_expert * router_prob_per_expert).sum()
            )
            return output, aux_loss

        return output

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._forward_impl(x)

    def forward_repacked(self, x: torch.Tensor) -> torch.Tensor:
        result = self.forward(x)
        if isinstance(result, tuple):
            return result[0]
        return result
