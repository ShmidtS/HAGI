"""HAGI model — Perception / Reasoning / Expression with optional GDR.

A single class covers all four ablation models via config flags:

    Model A (baseline): use_loop=False, use_gdr=False
    Model B (loop):     use_loop=True,  use_gdr=False
    Model C (HDIM):     use_loop=False, use_gdr=True   (Clifford bolted on, no loop)
    Model D (GDR):      use_loop=True,  use_gdr=True   (full HAGI)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .gdr import GradeConfig, GradeDecomposedRecurrence
from .transformer import RMSNorm, TransformerBlock, TransformerConfig, build_rope_cache


@dataclass
class HAGIConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    perception_layers: int = 4
    reasoning_layers: int = 4
    expression_layers: int = 4
    loop_count: int = 3
    use_loop: bool = True
    use_gdr: bool = True
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    grades: GradeConfig = field(default_factory=GradeConfig)

    def __post_init__(self):
        assert self.hidden_size == self.transformer.hidden_size
        if self.use_gdr:
            assert self.hidden_size == self.grades.hidden_size, (
                f"grade dims sum to {self.grades.hidden_size}, hidden is {self.hidden_size}"
            )


class HAGI(nn.Module):
    def __init__(self, cfg: HAGIConfig):
        super().__init__()
        self.cfg = cfg
        tcfg = cfg.transformer

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.perception = nn.ModuleList(TransformerBlock(tcfg) for _ in range(cfg.perception_layers))
        self.reasoning = nn.ModuleList(TransformerBlock(tcfg) for _ in range(cfg.reasoning_layers))
        self.expression = nn.ModuleList(TransformerBlock(tcfg) for _ in range(cfg.expression_layers))

        self.gdr = GradeDecomposedRecurrence(cfg.grades) if cfg.use_gdr else None

        loops = cfg.loop_count if cfg.use_loop else 1
        self.iter_embed = nn.Parameter(torch.zeros(loops, cfg.hidden_size))

        self.final_norm = RMSNorm(cfg.hidden_size, tcfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

        self._rope = {}

    def _rope_cache(self, T: int, device, dtype):
        key = (T, device, dtype)
        if key not in self._rope:
            head_dim = self.cfg.transformer.hidden_size // self.cfg.transformer.num_query_heads
            self._rope[key] = build_rope_cache(T, head_dim, self.cfg.transformer.rope_theta, device, dtype)
        return self._rope[key]

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        ignore_index: int = -100,
    ):
        """Returns logits, or (logits, loss) when targets are provided.

        nanoGPT-compatible. Targets are next-token labels aligned to input_ids
        (caller does the shift, or passes -100 for masked positions).
        """
        B, T = input_ids.shape
        h = self.embed(input_ids)
        cos, sin = self._rope_cache(T, h.device, h.dtype)

        for block in self.perception:
            h = block(h, cos, sin)

        loops = self.cfg.loop_count if self.cfg.use_loop else 1
        for i in range(loops):
            if self.gdr is not None:
                h = self.gdr(h)
            for block in self.reasoning:
                h = block(h, cos, sin)
            h = h + self.iter_embed[i]

        for block in self.expression:
            h = block(h, cos, sin)

        h = self.final_norm(h)
        logits = self.lm_head(h)

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            ignore_index=ignore_index,
        )
        return logits, loss

    def num_parameters(self, unique: bool = True) -> int:
        # Reasoning core params count once (shared) regardless of loop_count.
        return sum(p.numel() for p in self.parameters())
