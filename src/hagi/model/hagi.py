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
from torch.utils.checkpoint import checkpoint

from .gdr import GradeConfig, GradeDecomposedRecurrence
from .hdim_full import HDIMFull
from .hrm_full import HRMCore
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
    hdim_full: bool = False
    hdim_heads: int = 4
    hrm: bool = False
    h_dim: int = 256
    l_dim: int = 256
    gradient_checkpointing: bool = False
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    grades: GradeConfig = field(default_factory=GradeConfig)

    def __post_init__(self):
        assert self.hidden_size == self.transformer.hidden_size
        if self.use_gdr and not self.hdim_full and not self.hrm:
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

        self.gdr = None
        if cfg.use_gdr:
            if cfg.hdim_full:
                self.gdr = HDIMFull(hidden_size=cfg.hidden_size, heads=cfg.hdim_heads)
            else:
                self.gdr = GradeDecomposedRecurrence(cfg.grades)
        self.hrm = (
            HRMCore(
                hidden_size=cfg.hidden_size,
                h_dim=cfg.h_dim,
                l_dim=cfg.l_dim,
                h_cycles=cfg.loop_count,
                l_cycles=cfg.reasoning_layers,
            )
            if cfg.hrm
            else None
        )

        loops = cfg.loop_count if cfg.use_loop else 1
        self.iter_embed = nn.Parameter(torch.zeros(loops, cfg.hidden_size))

        self.final_norm = RMSNorm(cfg.hidden_size, tcfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

        self._rope = {}

    def _rope_cache(self, T: int, device, dtype, offset: int = 0):
        key = (T + offset, device, dtype)
        if key not in self._rope:
            head_dim = self.cfg.transformer.hidden_size // self.cfg.transformer.num_query_heads
            self._rope[key] = build_rope_cache(T + offset, head_dim, self.cfg.transformer.rope_theta, device, dtype)
        cos, sin = self._rope[key]
        return cos[offset : offset + T], sin[offset : offset + T]

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        ignore_index: int = -100,
        past_key_values=None,
        use_cache: bool = False,
        training_mode: bool = False,
    ):
        """Returns logits, or (logits, loss) when targets are provided.

        nanoGPT-compatible. Targets are next-token labels aligned to input_ids
        (caller does the shift, or passes -100 for masked positions).
        """
        B, T = input_ids.shape
        cache_pos = 0
        if past_key_values is not None and len(past_key_values) > 0:
            first = past_key_values[0]
            if first is not None:
                cache_pos = int(first[0].shape[2])
        h = self.embed(input_ids)
        cos, sin = self._rope_cache(T, h.device, h.dtype, cache_pos)
        next_key_values = [] if use_cache else None
        layer_idx = 0
        gdr_output = None
        use_gradient_checkpointing = self.cfg.gradient_checkpointing and self.training and not use_cache

        def run_block(block, hidden, past=None):
            if use_gradient_checkpointing:
                return checkpoint(
                    lambda h, c, s: block(h, c, s, gradient_checkpointing=True),
                    hidden,
                    cos,
                    sin,
                    use_reentrant=False,
                )
            if use_cache:
                return block(hidden, cos, sin, past, use_cache=True)
            return block(hidden, cos, sin, gradient_checkpointing=self.cfg.gradient_checkpointing)

        for block in self.perception:
            past = past_key_values[layer_idx] if past_key_values is not None else None
            if use_cache:
                h, next_kv = run_block(block, h, past)
                next_key_values.append(next_kv)
            else:
                h = run_block(block, h)
            layer_idx += 1

        if self.hrm is not None:
            h, _, _ = self.hrm(h, self.reasoning, cos, sin)
            layer_idx += len(self.reasoning)
        else:
            loops = self.cfg.loop_count if self.cfg.use_loop else 1
            for i in range(loops):
                if self.gdr is not None:
                    h = self.gdr(h)
                    gdr_output = h
                for block in self.reasoning:
                    past = past_key_values[layer_idx] if past_key_values is not None else None
                    if use_cache:
                        h, next_kv = run_block(block, h, past)
                        next_key_values.append(next_kv)
                    else:
                        h = run_block(block, h)
                    layer_idx += 1
                h = h + self.iter_embed[i]

        for block in self.expression:
            past = past_key_values[layer_idx] if past_key_values is not None else None
            if use_cache:
                h, next_kv = run_block(block, h, past)
                next_key_values.append(next_kv)
            else:
                h = run_block(block, h)
            layer_idx += 1

        pre_logits_hidden = h.clone()
        h = self.final_norm(h)
        logits = self.lm_head(h)

        if training_mode:
            result = {"logits": logits}
            if gdr_output is not None:
                result["auxiliary_output"] = gdr_output
            if pre_logits_hidden is not None:
                result["model_output"] = pre_logits_hidden
            return result

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )
            return logits, loss
        if use_cache:
            return logits, next_key_values
        return logits

    def num_parameters(self, unique: bool = True) -> int:
        # Reasoning core params count once (shared) regardless of loop_count.
        return sum(p.numel() for p in self.parameters())
