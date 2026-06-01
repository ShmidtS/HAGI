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
from .transformer import RMSNorm, TransformerBlock, TransformerConfig, build_rope_cache


def cross_entropy_loss(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100, chunk_size: int = 0
) -> torch.Tensor:
    """Next-token cross-entropy with fp32 accumulation.

    logits: [N, V] (already flattened). The fp32 upcast of the full [N, V] tensor
    is the dominant activation-memory spike at large N·V (e.g. 16·4096·49152·4B ≈
    13 GB). When chunk_size > 0, the upcast happens per row-chunk so the fp32 copy
    never fully materializes. The result is numerically identical to the unchunked
    path (sum over chunks / valid-token count == mean).
    """
    if chunk_size <= 0 or logits.size(0) <= chunk_size:
        return F.cross_entropy(logits.float(), targets, ignore_index=ignore_index)
    valid = (targets != ignore_index).sum().clamp(min=1)
    total = torch.zeros((), dtype=torch.float32, device=logits.device)
    for i in range(0, logits.size(0), chunk_size):
        lg = logits[i : i + chunk_size].float()
        tg = targets[i : i + chunk_size]
        total = total + F.cross_entropy(lg, tg, ignore_index=ignore_index, reduction="sum")
    return total / valid


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
    gradient_checkpointing: bool = False  # trade ~30% recompute for activation memory
    ce_chunk_size: int = 0                # >0 chunks the fp32 CE upcast (avoids logit spike)
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
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        """GPT-2-style init. Default nn.Embedding is N(0, 1); tied to the LM head
        that gives logits with std ~hidden·1 at init (initial loss ~10× ln(vocab),
        large early gradients). std=0.02 keeps initial logits near-uniform. RMSNorm
        weights (ones) and iteration embeddings (zeros) are left at their init."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

        # Gradient checkpointing only helps in the backward pass; skip in eval.
        use_ckpt = self.cfg.gradient_checkpointing and self.training

        def run_block(block, x):
            if use_ckpt:
                return checkpoint(block, x, cos, sin, use_reentrant=False)
            return block(x, cos, sin)

        for block in self.perception:
            h = run_block(block, h)

        loops = self.cfg.loop_count if self.cfg.use_loop else 1
        for i in range(loops):
            if self.gdr is not None:
                h = checkpoint(self.gdr, h, use_reentrant=False) if use_ckpt else self.gdr(h)
            for block in self.reasoning:
                h = run_block(block, h)
            h = h + self.iter_embed[i]

        for block in self.expression:
            h = run_block(block, h)

        h = self.final_norm(h)
        logits = self.lm_head(h)

        if targets is None:
            return logits

        loss = cross_entropy_loss(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=ignore_index,
            chunk_size=self.cfg.ce_chunk_size,
        )
        return logits, loss

    def num_parameters(self, unique: bool = True) -> int:
        # Reasoning core params count once (shared) regardless of loop_count.
        return sum(p.numel() for p in self.parameters())
