"""Mixture of Experts (MoE) SwiGLU module."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .binary_factorized import BinaryFactorizedLinear


def _make_linear(in_features: int, out_features: int, cfg: Any) -> nn.Module:
    if getattr(cfg, "use_binary_factorized", False):
        return BinaryFactorizedLinear(in_features, out_features, getattr(cfg, "binary_factorized_rank", 8))
    return nn.Linear(in_features, out_features, bias=False)


class _SwiGLUExpert(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        intermediate_size = cfg.moe_intermediate_size or (cfg.intermediate_size // cfg.num_experts)
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
        self.intermediate_size = cfg.moe_intermediate_size or (cfg.intermediate_size // cfg.num_experts)
        self.router_temperature = getattr(cfg, "moe_router_temperature", 1.0)
        self.alpha = getattr(cfg, "moe_alpha", 0.01)

        self.router = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
        self.experts = nn.ModuleList(_SwiGLUExpert(cfg) for _ in range(cfg.num_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        assert D == self.hidden_size

        flat = x.view(B * T, D)
        router_logits = self.router(flat)
        if self.training:
            noise = torch.randn_like(router_logits) * 0.01
            router_logits = router_logits + noise.detach()
        router_logits = router_logits / self.router_temperature
        router_logits = router_logits.clamp(-10, 10)
        router_probs = F.softmax(router_logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        weight_matrix = torch.zeros(B * T, self.num_experts, device=x.device, dtype=top_k_probs.dtype)
        weight_matrix.scatter_(1, top_k_indices, top_k_probs)

        output = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            w = weight_matrix[:, i]  # [B*T]
            expert_out = expert(flat)
            output += expert_out * w.unsqueeze(-1)

        output = output.view(B, T, D)

        if self.training:
            router_prob_per_expert = router_probs.mean(dim=0)
            top_k_mask = torch.zeros(B * T, self.num_experts, device=x.device, dtype=router_probs.dtype)
            top_k_mask.scatter_(1, top_k_indices, 1.0)
            fraction_per_expert = top_k_mask.mean(dim=0)
            aux_loss = self.alpha * self.num_experts * (fraction_per_expert * router_prob_per_expert).sum()
            aux_loss = aux_loss.clamp_max(10.0)
            return output, aux_loss

        return output

    def forward_repacked(self, x: torch.Tensor) -> torch.Tensor:
        result = self.forward(x)
        if isinstance(result, tuple):
            return result[0]
        return result
