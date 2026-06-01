"""Lean4 bridge: build checker + config validator.

The Lean4 specification at formalization/HAGI/ defines proof obligations for
shape safety, routing safety, and domain transfer identity.
This bridge verifies that Python runtime configs satisfy the same contracts.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class LeanBridge:
    lake_command: str = "lake"

    def verify(self, root_path: str | Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.lake_command, "build"],
            cwd=Path(root_path),
            text=True,
            capture_output=True,
            check=False,
        )

    def check(self, root_path: str | Path) -> bool:
        return self.verify(root_path).returncode == 0

    # ------------------------------------------------------------------
    # Config validation: mirror Lean proof obligations at runtime
    # ------------------------------------------------------------------

    @staticmethod
    def validate_transformer_config(config: dict | object) -> None:
        """Validate config matches Lean `Transformer.TransformerConfig` proof fields.

        Raises ValueError with the same message contract as Lean structure fields.
        """
        if isinstance(config, dict):
            hidden_size = int(config["hidden_size"])
            num_query_heads = int(config["num_query_heads"])
            num_kv_heads = int(config["num_kv_heads"])
            intermediate_size = int(config.get("intermediate_size", 2048))
            max_seq_len = int(config.get("max_seq_len", 4096))
        else:
            hidden_size = getattr(config, "hidden_size")
            num_query_heads = getattr(config, "num_query_heads")
            num_kv_heads = getattr(config, "num_kv_heads")
            intermediate_size = getattr(config, "intermediate_size", 2048)
            max_seq_len = getattr(config, "max_seq_len", 4096)

        if num_query_heads <= 0:
            raise ValueError(
                f"num_query_heads must be positive (Lean: hiddenDivisible.left)"
            )
        if hidden_size % num_query_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} not divisible by num_query_heads {num_query_heads} "
                f"(Lean: Transformer.HeadDivisible)"
            )
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads {num_query_heads} not divisible by num_kv_heads {num_kv_heads} "
                f"(Lean: Transformer.queryDivisible)"
            )
        head_dim = hidden_size // num_query_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim {head_dim} must be even for RoPE (Lean: Transformer.headDimEven)"
            )
        if max_seq_len <= 0:
            raise ValueError(
                f"max_seq_len {max_seq_len} must be positive (Lean: Transformer.maxSeqLen_pos)"
            )
        if intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size {intermediate_size} must be positive (Lean: Transformer.intermediateSize_pos)"
            )

    @staticmethod
    def validate_hrm_config(config: dict | object) -> None:
        """Validate config matches Lean `HRM.HRMConfig` proof fields."""
        if isinstance(config, dict):
            h_cycles = int(config.get("h_cycles", 1))
            l_cycles = int(config.get("l_cycles", 1))
            hidden_size = int(config["hidden_size"])
            num_heads = int(config.get("num_heads", config.get("num_query_heads", 12)))
            max_seq_len = int(config.get("max_seq_len", 4096))
        else:
            h_cycles = getattr(config, "h_cycles", 1)
            l_cycles = getattr(config, "l_cycles", 1)
            hidden_size = getattr(config, "hidden_size")
            num_heads = getattr(
                config, "num_heads", getattr(config, "num_query_heads", 12)
            )
            max_seq_len = getattr(config, "max_seq_len", 4096)

        if h_cycles <= 0:
            raise ValueError(
                f"h_cycles {h_cycles} must be positive (Lean: HRM.positiveHCycles)"
            )
        if l_cycles <= 0:
            raise ValueError(
                f"l_cycles {l_cycles} must be positive (Lean: HRM.positiveLCycles)"
            )
        if num_heads <= 0:
            raise ValueError(
                f"num_heads {num_heads} must be positive (Lean: HRM.validHeads.left)"
            )
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} not divisible by num_heads {num_heads} "
                f"(Lean: HRM.validHeads)"
            )
        if max_seq_len <= 0:
            raise ValueError(
                f"max_seq_len {max_seq_len} must be positive (Lean: HRM.maxSequenceLength_pos)"
            )

    @staticmethod
    def validate_loss_input(
        logits: "Tensor",
        targets: "Tensor",
        vocab_size: int | None = None,
    ) -> None:
        """Validate cross-entropy input matches Lean `Losses.CrossEntropyInput`.

        - logits last dimension = vocab_size
        - targets shape matches logits with last dimension removed
        """
        if logits.dim() < 2:
            raise ValueError(
                f"logits must be at least 2D (Lean: Losses.logits_is_2d)"
            )
        if vocab_size is not None and logits.size(-1) != vocab_size:
            raise ValueError(
                f"logits last dim {logits.size(-1)} != vocab_size {vocab_size} "
                f"(Lean: Losses.CrossEntropyInput.logitsLastDim)"
            )
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        if logits_flat.size(0) != targets_flat.size(0):
            raise ValueError(
                f"logits batch*seq {logits_flat.size(0)} != targets batch*seq {targets_flat.size(0)} "
                f"(Lean: Losses.targets_flattened_shape)"
            )

    @staticmethod
    def validate_prefix_lm_config(
        prefix_lengths: list[int],
        total_len: int,
    ) -> None:
        """Validate prefix-LM construction matches Lean `Data.PrefixLMConfig`.

        - all prefix lengths non-negative
        - sum(prefix_lengths) <= total_len
        - total_len > 0
        """
        if total_len <= 0:
            raise ValueError(
                f"total_len {total_len} must be positive (Lean: Data.PrefixLMConfig.totalLen_pos)"
            )
        if any(p < 0 for p in prefix_lengths):
            raise ValueError(
                "prefix lengths must be non-negative (Lean: Data.PrefixLMConfig.all_prefix_nonneg)"
            )
        if sum(prefix_lengths) > total_len:
            raise ValueError(
                f"sum(prefix_lengths)={sum(prefix_lengths)} > total_len={total_len} "
                f"(Lean: Data.PrefixLMConfig.sum_prefix_le_total)"
            )

    # ------------------------------------------------------------------
    # Full HAGI config validator (combines all sub-validators)
    # ------------------------------------------------------------------

    def validate_hagi_config(self, config: dict | object) -> None:
        """Run all Lean-mirrored validators on a full HAGI training config."""
        self.validate_transformer_config(config)
        self.validate_hrm_config(config)
        logger.info("Lean-mirrored config validation passed")


def verify(root_path: str | Path) -> bool:
    return LeanBridge().check(root_path)


def validate_config(config: dict | object) -> None:
    LeanBridge().validate_hagi_config(config)
