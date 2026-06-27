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

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from hagi.utils.env import load_env
load_env()


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


# =============================================================================
# Offline Self-Distillation (VibeThinker-inspired)
# =============================================================================


@torch.inference_mode()
def learning_potential_score(
    student_model: nn.Module,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    device: str = "cuda",
) -> float:
    """Compute the learning-potential score of a trajectory under the student.

    S_LP = -(1/|y|) * sum_t log pi_stu(y_t | q, y<t)

    A higher score means the trace is NOT yet well modeled by the student
    and therefore carries higher distillation value.

    Args:
        student_model: the current student model (eval mode).
        prompt_ids: [prompt_len] prompt token ids.
        response_ids: [response_len] response token ids.
        device: target device.

    Returns:
        Scalar learning-potential score.
    """
    full = torch.cat([prompt_ids, response_ids]).unsqueeze(0).to(device)
    resp_start = prompt_ids.size(0)

    output = student_model(full, training_mode=False)
    logits = output["logits"] if isinstance(output, dict) else output[0]
    if logits is None:
        raise ValueError(
            "logits is None — disable use_fused_ce for self-distillation"
        )

    resp_logits = logits[0, resp_start - 1 : -1, :]
    resp_tokens = full[0, resp_start:]

    log_probs = torch.nn.functional.log_softmax(resp_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, resp_tokens.unsqueeze(-1)).squeeze(-1)
    nll = -token_log_probs.mean().item()
    return float(nll)


class OfflineSelfDistillation:
    """Offline self-distillation from RL-enhanced checkpoints.

    VibeThinker-inspired pipeline:
    1. Generate trajectories with the RL-enhanced model
    2. Verify correctness with reward functions
    3. Compute learning-potential score (NLL under student)
    4. Filter by length buckets + score range
    5. Return selected (prompt, response) pairs for SFT

    The student is the current SFT model; the teacher is the RL checkpoint
    (same architecture, different weights — self-distillation).
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        tokenizer: Any,
        device: str = "cuda",
        min_response_len: int = 32,
        max_response_len: int = 2048,
        score_percentile_low: float = 0.3,
        score_percentile_high: float = 0.95,
        num_length_buckets: int = 5,
    ):
        self.student = student_model
        self.teacher = teacher_model
        self.tokenizer = tokenizer
        self.device = device
        self.min_response_len = min_response_len
        self.max_response_len = max_response_len
        self.score_percentile_low = score_percentile_low
        self.score_percentile_high = score_percentile_high
        self.num_length_buckets = num_length_buckets

    @torch.inference_mode()
    def collect_trajectories(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int | None = 50,
        top_p: float | None = 0.9,
        eos_token_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Generate and verify trajectories from the teacher (RL checkpoint).

        Returns list of dicts with keys:
            prompt, prompt_ids, response, response_ids, length, score, correct
        """
        from hagi.inference.generate import generate

        was_training = self.teacher.training
        self.teacher.eval()
        self.student.eval()

        results: list[dict[str, Any]] = []
        for prompt_text in prompts:
            prompt_ids = self.tokenizer.encode(prompt_text)
            if isinstance(prompt_ids, list):
                prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)

            full = generate(
                self.teacher,
                prompt_ids.unsqueeze(0).to(self.device),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_token_id,
                use_cache=True,
            )
            resp_start = prompt_ids.size(0)
            response_ids = full[0, resp_start:]
            response_text = self.tokenizer.decode(response_ids.tolist())
            resp_len = response_ids.size(0)

            if resp_len < self.min_response_len:
                continue
            if resp_len > self.max_response_len:
                response_ids = response_ids[: self.max_response_len]
                resp_len = self.max_response_len

            score = learning_potential_score(
                self.student,
                prompt_ids,
                response_ids,
                device=self.device,
            )

            results.append(
                {
                    "prompt": prompt_text,
                    "prompt_ids": prompt_ids.cpu(),
                    "response": response_text,
                    "response_ids": response_ids.cpu(),
                    "length": resp_len,
                    "score": score,
                    "correct": None,
                }
            )

        if was_training:
            self.teacher.train()

        return results

    def filter_by_learning_potential(
        self,
        trajectories: list[dict[str, Any]],
        references: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Filter trajectories by correctness and learning-potential score.

        If references are provided, only correct trajectories are kept
        (verified by math_verify). Then, within length buckets, trajectories
        are ranked by learning-potential score and the middle-to-high range
        is selected.

        Args:
            trajectories: output from collect_trajectories.
            references: optional list of reference answers for verification.

        Returns:
            Filtered list of trajectory dicts.
        """
        from hagi.train.rewards import math_verify

        filtered = trajectories

        if references is not None:
            verified = []
            for traj, ref in zip(trajectories, references, strict=True):
                is_correct = math_verify(traj["response"], ref)
                if is_correct:
                    traj["correct"] = True
                    verified.append(traj)
            filtered = verified

        if not filtered:
            return []

        lengths = torch.tensor([t["length"] for t in filtered], dtype=torch.float32)

        length_min = float(lengths.min())
        length_max = float(lengths.max())
        if length_max <= length_min:
            bucket_edges = [length_min]
        else:
            bucket_edges = torch.linspace(length_min, length_max, self.num_length_buckets + 1)

        selected: list[dict[str, Any]] = []
        for i in range(self.num_length_buckets):
            lo = float(bucket_edges[i])
            hi = float(bucket_edges[i + 1]) if i + 1 < len(bucket_edges) else float(length_max + 1)

            bucket = [
                (idx, t)
                for idx, t in enumerate(filtered)
                if lo <= t["length"] < hi
            ]
            if not bucket:
                continue

            bucket_scores = torch.tensor(
                [t["score"] for _, t in bucket], dtype=torch.float32
            )
            lo_pct = torch.quantile(
                bucket_scores, self.score_percentile_low
            ).item()
            hi_pct = torch.quantile(
                bucket_scores, self.score_percentile_high
            ).item()

            for _, t in bucket:
                if lo_pct <= t["score"] <= hi_pct:
                    selected.append(t)

        return selected

    def build_sft_dataset(
        self,
        selected: list[dict[str, Any]],
    ) -> list[dict[str, torch.Tensor]]:
        """Convert selected trajectories into an SFT-ready dataset.

        Returns list of dicts with keys:
            input_ids: [prompt_len + response_len] tensor
            labels: [prompt_len + response_len] tensor (prompt = -100, response = token)
        """
        dataset: list[dict[str, torch.Tensor]] = []
        for traj in selected:
            prompt_ids = traj["prompt_ids"]
            response_ids = traj["response_ids"]
            full = torch.cat([prompt_ids, response_ids])
            labels = full.clone()
            labels[: prompt_ids.size(0)] = -100
            dataset.append({"input_ids": full, "labels": labels})
        return dataset
