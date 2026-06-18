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
        self.gate = _make_linear(cfg.hidden_size, intermediate_size, cfg)
        self.up = _make_linear(cfg.hidden_size, intermediate_size, cfg)
        self.down = _make_linear(intermediate_size, cfg.hidden_size, cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


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

        self.router = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
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
            router_prob_per_expert = router_probs.mean(dim=0)
            top_k_mask = torch.zeros(
                B * T, self.num_experts, device=x.device, dtype=router_probs.dtype
            )
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
