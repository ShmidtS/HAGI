"""Online knowledge distillation from SmolLM2-135M for HAGI training.

Teacher (SmolLM2-135M) provides:
  1. Pretrained token embeddings (exact copy at init)
  2. Soft logit targets for KL distillation during training

Student (HAGI) learns from: alpha * CE_hard + (1-alpha) * T^2 * KL(soft_student || soft_teacher)

VRAM strategy: teacher forward returns hidden states (14MB), NOT logits (1.2GB).
Both student and teacher hidden are projected to logits per-chunk inside KL,
so peak logits memory = 2 * chunk_size * V * dtype_bytes, never [B, T, V].
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

# Load HF_TOKEN from project root .env file (needed for SmolLM2-135M download)
_env_path = Path(__file__).resolve().parents[3] / ".env"
if _env_path.exists():
    with _env_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip("\"'")
                break


def transfer_embeddings(
    model: nn.Module,
    teacher_model_name: str = "HuggingFaceTB/SmolLM2-135M",
) -> int:
    """Copy pretrained SmolLM2 embeddings into HAGI's embedding layer.

    Requires model.cfg.vocab_size == teacher vocab_size and
    model.cfg.hidden_size == teacher hidden_size (576).
    Weight tying: lm_head shares embed.weight, so lm_head is updated automatically.

    Returns number of tokens transferred.
    """
    from transformers import AutoModelForCausalLM

    teacher: Any = AutoModelForCausalLM.from_pretrained(
        teacher_model_name, dtype=torch.bfloat16, local_files_only=True
    )
    teacher_emb = teacher.model.embed_tokens.weight.data  # [V, H]

    embed_weight = model.embed.weight  # type: ignore[attr-defined]
    assert isinstance(embed_weight, torch.Tensor)
    assert embed_weight.shape == teacher_emb.shape, (
        f"embedding shape mismatch: {embed_weight.shape} vs {teacher_emb.shape}. "
        f"hidden_size must match teacher ({teacher_emb.shape[1]})"
    )

    embed_weight.data.copy_(teacher_emb)
    n = int(embed_weight.shape[0])
    del teacher
    return n


class DistillationTeacher:
    """Frozen SmolLM2 teacher for online distillation.

    Loads the teacher model in bf16, freezes all params, and provides
    a forward method returning hidden states (NOT logits) to save VRAM.
    The lm_head weight is exposed for per-chunk projection in KL loss.

    Teacher forward runs in micro-batches to bound peak activation VRAM:
    instead of one [B, T, H] forward (which allocates attention scores for
    the full batch), runs N forwards of micro_batch tokens each and concats.
    Peak teacher activation VRAM = micro_batch/batch of the full forward.

    Resides in VRAM during the distill phase; call .free() to release.
    """

    model: Any
    _base_model: Any

    def __init__(
        self,
        teacher_model_name: str = "HuggingFaceTB/SmolLM2-135M",
        device: str = "cuda",
        micro_batch: int = 0,
    ):
        from transformers import AutoModelForCausalLM

        self.model: Any = AutoModelForCausalLM.from_pretrained(
            teacher_model_name, dtype=torch.bfloat16, local_files_only=True
        )
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self._base_model: Any = self.model.model
        self.device = device
        self._micro_batch = micro_batch
        self._compiled = False

    def _ensure_compiled(self) -> None:
        """Lazily compile the teacher's base transformer for faster inference."""
        if self._compiled or not hasattr(torch, "compile"):
            return
        try:
            self._base_model = torch.compile(
                self._base_model, dynamic=False
            )
            self._compiled = True
        except Exception:
            pass

    @property
    def lm_head_weight(self) -> torch.Tensor:
        """Teacher lm_head weight [V, H] for per-chunk projection."""
        w = self.model.lm_head.weight
        assert isinstance(w, torch.Tensor)
        return w

    @torch.inference_mode()
    def forward_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run teacher forward and return last hidden state [B, T, H].

        Uses the base transformer model (not the CausalLM head) to avoid
        materializing [B, T, V] logits. Saves ~1.2GB VRAM at batch=12.

        When micro_batch > 0, processes the batch in chunks of micro_batch
        to bound peak activation memory (teacher attention scores scale
        linearly with batch). Trades a few extra kernel launches for
        significantly lower VRAM peak.
        """
        self._ensure_compiled()
        if self._micro_batch > 0 and tokens.size(0) > self._micro_batch:
            hiddens: list[torch.Tensor] = []
            for i in range(0, tokens.size(0), self._micro_batch):
                chunk = tokens[i : i + self._micro_batch]
                out = self._base_model(chunk)
                hiddens.append(out.last_hidden_state)
            return torch.cat(hiddens, dim=0)
        out = self._base_model(tokens)
        return out.last_hidden_state  # [B, T, H]

    @torch.inference_mode()
    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_hidden(tokens)

    def free(self) -> None:
        """Release teacher model from VRAM."""
        self.model = None
        self._base_model = None
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def distillation_loss_chunked(
    student_hidden: torch.Tensor,
    student_lm_head_weight: torch.Tensor,
    teacher_hidden: torch.Tensor,
    teacher_lm_head_weight: torch.Tensor,
    targets: torch.Tensor,
    ce_loss: torch.Tensor,
    T: float = 2.0,
    alpha: float = 0.5,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Chunked KL distillation loss — never materializes full [B, T, V] logits.

    CE is already computed via the fused CE path in the student forward.
    This adds the KL term by projecting BOTH student and teacher hidden
    to logits per-chunk, computing KL(softmax(student/T) || softmax(teacher/T)).

    Padded positions (targets == ignore_index/-100) are masked out of the KL
    sum so they don't inflate the loss with meaningless teacher logits for
    padding tokens. With fixed-length training (no padding) this is a no-op.

    Peak VRAM per chunk: 2 * chunk_size * V * (2B logits + 4B softmax) .
    At chunk_size=4096, V=49152: 2 * 4096 * 49152 * 6B ~ 2.4GB.

    Returns: alpha * ce_loss + (1 - alpha) * T^2 * kl_loss
    """
    flat_sh = student_hidden.reshape(-1, student_hidden.size(-1))
    flat_th = teacher_hidden.reshape(-1, teacher_hidden.size(-1))
    flat_t = targets.reshape(-1)
    valid_mask = flat_t != -100
    valid = valid_mask.sum().clamp(min=1)
    total_kl = flat_sh.new_zeros((), dtype=torch.float32)

    for i in range(0, flat_sh.size(0), chunk_size):
        sh_c = flat_sh[i : i + chunk_size]
        th_c = flat_th[i : i + chunk_size]
        t_mask_c = valid_mask[i : i + chunk_size]
        if not t_mask_c.any():
            continue
        s_logits = F.linear(sh_c, student_lm_head_weight)  # [chunk, V]
        t_logits = F.linear(th_c, teacher_lm_head_weight)  # [chunk, V]
        s_log_soft = F.log_softmax(s_logits / T, dim=-1)
        t_soft = F.softmax(t_logits / T, dim=-1)
        kl = F.kl_div(s_log_soft, t_soft, reduction="none").sum(dim=-1)  # [chunk]
        kl = kl * t_mask_c.float()  # zero out padded positions
        total_kl = total_kl + kl.sum()

    kl_loss = total_kl / valid
    return alpha * ce_loss + (1.0 - alpha) * (T * T) * kl_loss


def alpha_at(
    step: int,
    alpha_start: float,
    alpha_end: float,
    max_steps: int,
    distill_end_step: int,
) -> float:
    """Linear alpha schedule: alpha_start -> alpha_end over distill phase, then 1.0."""
    if step > distill_end_step:
        return 1.0
    progress = min(1.0, step / max(1, distill_end_step))
    return alpha_start + (alpha_end - alpha_start) * progress
