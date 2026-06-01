"""Mixture-of-Experts wrapper using BinarySwiGLU experts.

G004 first slice: top-k softmax routing over dense router weights.
Switch-transformer style load-balancing aux loss.
Naive per-expert loop (small num_experts).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .binary import BinarySwiGLU


@dataclass
class MoEOutput:
    y: torch.Tensor
    aux_loss: torch.Tensor


class MoEBinarySwiGLU(nn.Module):
    """Top-k MoE over BinarySwiGLU experts.

    Args:
        cfg: TransformerConfig (for hidden_size, intermediate_size).
        num_experts: number of expert FFNs.
        top_k: experts activated per token.
        rank: BinaryFactorizedLinear rank for each expert.
        block_size: BinaryFactorizedLinear block size for each expert.
        aux_loss_coef: weight on switch-transformer load-balancing aux loss.
    """

    def __init__(
        self,
        cfg,
        num_experts: int,
        top_k: int,
        rank: int,
        block_size: int,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        self.experts = nn.ModuleList(
            [BinarySwiGLU(cfg, rank, block_size) for _ in range(num_experts)]
        )
        self.router = nn.Linear(cfg.hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> MoEOutput:
        B, T, H = x.shape
        flat = x.reshape(-1, H)

        router_logits = self.router(flat)
        top_logits, top_idx = router_logits.topk(self.top_k, dim=-1)
        top_weights = F.softmax(top_logits, dim=-1)

        out = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            mask = top_idx == i
            if not mask.any():
                continue
            token_idx, slot_idx = mask.nonzero(as_tuple=True)
            expert_in = flat[token_idx]
            expert_out = expert(expert_in)
            weights = top_weights[token_idx, slot_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, expert_out * weights)

        # Switch-transformer aux loss: N * sum_i (f_i * p_i)
        # f_i = fraction of (token, slot) pairs routed to expert i.
        # p_i = mean router probability for expert i.
        with torch.no_grad():
            f = torch.zeros(self.num_experts, device=flat.device, dtype=flat.dtype)
            for i in range(self.num_experts):
                f[i] = (top_idx == i).float().mean()
        probs = F.softmax(router_logits, dim=-1)
        p = probs.mean(dim=0)
        aux_loss = self.num_experts * (f * p).sum() * self.aux_loss_coef

        return MoEOutput(out.reshape(B, T, H), aux_loss)
