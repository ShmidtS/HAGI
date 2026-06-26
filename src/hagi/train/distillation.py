"""Online knowledge distillation for HAGI training.

Supports multiple teacher architectures:
  - Decoder-only CausalLM (SmolLM2, etc.): AutoModelForCausalLM
  - Encoder-decoder Seq2SeqLM (T5Gemma, etc.): AutoModelForSeq2SeqLM
  - Multimodal LM (Gemma 4 Unified, etc.): AutoModelForMultimodalLM

Teacher provides:
  1. Pretrained token embeddings (exact copy at init, if hidden_size matches)
  2. Soft logit targets for KL distillation during training

Student (HAGI) learns from: alpha * CE_hard + (1-alpha) * T^2 * KL(soft_student || soft_teacher)

VRAM strategy: teacher forward returns hidden states, NOT logits.
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


def _detect_model_type(model_name: str) -> str:
    """Detect model architecture type from model name.

    Returns one of: "causal_lm", "seq2seq_lm", "multimodal_lm".
    Gemma 4 models are all multimodal (Unified architecture with
    vision/audio encoders, even when the name doesn't say "unified").
    """
    name_lower = model_name.lower()
    if "t5gemma" in name_lower or "t5-" in name_lower or "ul2" in name_lower:
        return "seq2seq_lm"
    # All Gemma 4 models are multimodal (text + vision, some + audio).
    # The 12B model is "Unified" (encoder-free), E2B/E4B have PLE.
    # 26B A4B and 31B also have vision encoders.
    if "gemma-4" in name_lower or "gemma4" in name_lower:
        return "multimodal_lm"
    return "causal_lm"


def _get_embeddings(model: Any, model_type: str) -> torch.Tensor:
    """Extract token embedding weights from various model architectures.

    Returns [V, H] embedding tensor.
    """
    if model_type == "seq2seq_lm":
        # T5Gemma: encoder and decoder share tied embeddings.
        # Access via encoder.embed_tokens (or decoder.embed_tokens — same weight).
        if hasattr(model, "encoder") and hasattr(model.encoder, "embed_tokens"):
            return model.encoder.embed_tokens.weight.data
        if hasattr(model, "model") and hasattr(model.model, "encoder"):
            return model.model.encoder.embed_tokens.weight.data
        raise AttributeError(f"cannot find encoder embeddings in {type(model).__name__}")

    if model_type == "multimodal_lm":
        # Gemma 4 Unified: text model has embed_tokens.
        # Path: model.model.text_model.embed_tokens or model.language_model.embed_tokens
        for path in [
            ("model", "text_model", "embed_tokens"),
            ("model", "language_model", "embed_tokens"),
            ("language_model", "embed_tokens"),
            ("model", "embed_tokens"),
        ]:
            obj = model
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                return obj.weight.data
            except AttributeError:
                continue
        raise AttributeError(f"cannot find text embeddings in {type(model).__name__}")

    # Default: CausalLM (SmolLM2, Llama, etc.)
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.weight.data
    if hasattr(model, "embed_tokens"):
        return model.embed_tokens.weight.data
    raise AttributeError(f"cannot find embeddings in {type(model).__name__}")


def transfer_embeddings(
    model: nn.Module,
    teacher_model_name: str = "HuggingFaceTB/SmolLM2-135M",
) -> int:
    """Copy pretrained embeddings into HAGI's embedding layer.

    Supports decoder-only (CausalLM), encoder-decoder (Seq2SeqLM / T5Gemma),
    and multimodal (Gemma 4 Unified) architectures.

    Requires model.cfg.vocab_size == teacher vocab_size and
    model.cfg.hidden_size == teacher hidden_size.
    Weight tying: lm_head shares embed.weight, so lm_head is updated automatically.

    Returns number of tokens transferred.
    """
    model_type = _detect_model_type(teacher_model_name)

    if model_type == "seq2seq_lm":
        from transformers import AutoModelForSeq2SeqLM
        teacher: Any = AutoModelForSeq2SeqLM.from_pretrained(
            teacher_model_name, dtype=torch.bfloat16
        )
    elif model_type == "multimodal_lm":
        from transformers import AutoModelForMultimodalLM
        teacher = AutoModelForMultimodalLM.from_pretrained(
            teacher_model_name, dtype=torch.bfloat16
        )
    else:
        from transformers import AutoModelForCausalLM
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_name, dtype=torch.bfloat16
        )

    teacher_emb = _get_embeddings(teacher, model_type)  # [V, H]

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
    """Frozen teacher for online distillation.

    Supports multiple architectures:
      - CausalLM (SmolLM2, etc.): AutoModelForCausalLM, base model = model.model
      - Seq2SeqLM (T5Gemma, etc.): AutoModelForSeq2SeqLM, uses decoder
      - MultimodalLM (Gemma 4 Unified, etc.): AutoModelForMultimodalLM,
        full model loaded with ALL components (vision, audio, text).
        Text-only forward is used for KL distillation, but vision/audio
        encoders remain in VRAM for future multimodal training.

    Loads the teacher model, freezes all params, and provides
    a forward method returning hidden states (NOT logits) to save VRAM.
    The lm_head weight is exposed for per-chunk projection in KL loss.

    Teacher forward runs in micro-batches to bound peak activation VRAM.

    Resides in VRAM during the distill phase; call .free() to release.
    """

    model: Any
    _base_model: Any
    _model_type: str

    def __init__(
        self,
        teacher_model_name: str = "HuggingFaceTB/SmolLM2-135M",
        device: str = "cuda",
        micro_batch: int = 0,
    ):
        self._model_type = _detect_model_type(teacher_model_name)

        if self._model_type == "seq2seq_lm":
            from transformers import AutoModelForSeq2SeqLM
            self.model: Any = AutoModelForSeq2SeqLM.from_pretrained(
                teacher_model_name, dtype=torch.bfloat16
            )
            self._base_model = self.model.model  # encoder-decoder base
        elif self._model_type == "multimodal_lm":
            from transformers import AutoModelForMultimodalLM
            self.model = AutoModelForMultimodalLM.from_pretrained(
                teacher_model_name, dtype=torch.bfloat16
            )
            # Keep the FULL multimodal model (vision + audio + text).
            # Text-only forward is used for KL distillation; vision/audio
            # encoders stay loaded for future multimodal training.
            self._base_model = self.model
        else:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                teacher_model_name, dtype=torch.bfloat16
            )
            self._base_model = self.model.model

        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.device = device
        self._micro_batch = micro_batch
        self._compiled = False

    def _ensure_compiled(self) -> None:
        """Lazily compile the teacher's base model for faster inference.

        For multimodal models, compiles the full model (all components).
        Compilation is best-effort: silently skips on failure.
        """
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
        """Teacher lm_head weight [V, H] for per-chunk projection.

        Handles tied embeddings (common in Gemma/T5Gemma) where lm_head
        shares the embedding weight.
        """
        # Try direct lm_head access first
        if hasattr(self.model, "lm_head") and hasattr(self.model.lm_head, "weight"):
            w = self.model.lm_head.weight
            assert isinstance(w, torch.Tensor)
            return w

        # Tied embeddings: lm_head = embed_tokens
        if self._model_type == "multimodal_lm":
            try:
                w = _get_embeddings(self.model, "multimodal_lm")
                assert isinstance(w, torch.Tensor)
                return w
            except (AttributeError, AssertionError):
                pass
        elif self._model_type == "seq2seq_lm":
            try:
                w = _get_embeddings(self.model, "seq2seq_lm")
                assert isinstance(w, torch.Tensor)
                return w
            except (AttributeError, AssertionError):
                pass

        # Fallback: try model.model.embed_tokens (CausalLM path)
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            w = self.model.model.embed_tokens.weight
            assert isinstance(w, torch.Tensor)
            return w

        raise AttributeError(f"cannot find lm_head weight in {type(self.model).__name__}")

    @torch.inference_mode()
    def forward_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run teacher forward and return last hidden state [B, T, H].

        For CausalLM: uses the base transformer model (not the CausalLM head).
        For Seq2SeqLM (T5Gemma): uses the decoder in decoder-only mode
            (feeds tokens as decoder input, no encoder context).
        For MultimodalLM (Gemma 4): calls the FULL model with input_ids only.
            Vision/audio encoders are loaded but not activated. The text
            decoder produces hidden states for KL distillation. This preserves
            the full multimodal model for future multimodal training.

        When micro_batch > 0, processes the batch in chunks of micro_batch
        to bound peak activation memory.
        """
        self._ensure_compiled()
        if self._micro_batch > 0 and tokens.size(0) > self._micro_batch:
            hiddens: list[torch.Tensor] = []
            for i in range(0, tokens.size(0), self._micro_batch):
                chunk = tokens[i : i + self._micro_batch]
                out = self._forward_single(chunk)
                hiddens.append(out)
            return torch.cat(hiddens, dim=0)
        return self._forward_single(tokens)

    def _forward_single(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single micro-batch, returning [B, T, H].

        For MultimodalLM: calls the FULL model with input_ids only.
        Vision/audio encoders are loaded but not activated (no image/audio
        input). The text decoder produces hidden states for KL distillation.
        This preserves the full multimodal model for future multimodal
        training while using text-only forward for current distillation.
        """
        if self._model_type == "multimodal_lm":
            out = self.model(input_ids=tokens)
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state  # [B, T, H]
            # Some multimodal models return hidden states under different keys
            if hasattr(out, "text_hidden_states") and out.text_hidden_states is not None:
                return out.text_hidden_states
            # Fallback: access the text sub-model directly
            for path in [("model", "text_model"), ("model", "language_model")]:
                obj = self.model
                try:
                    for attr in path:
                        obj = getattr(obj, attr)
                    sub_out = obj(tokens)
                    if hasattr(sub_out, "last_hidden_state"):
                        return sub_out.last_hidden_state
                except (AttributeError, TypeError):
                    continue
            raise RuntimeError(
                f"cannot extract hidden states from {type(self.model).__name__}"
            )

        if self._model_type == "seq2seq_lm":
            # T5Gemma: run decoder-only forward by feeding tokens as
            # decoder_input_ids with empty encoder output.
            decoder = self._base_model.decoder if hasattr(self._base_model, "decoder") else self._base_model
            if hasattr(decoder, "embed_tokens"):
                emb = decoder.embed_tokens(tokens)
                B, T = tokens.shape
                H = emb.size(-1)
                enc_out = torch.zeros(1, 0, H, dtype=emb.dtype, device=emb.device)
                out = self._base_model(
                    input_ids=torch.zeros(1, 0, dtype=tokens.dtype, device=tokens.device),
                    decoder_input_ids=tokens,
                    encoder_outputs=(enc_out, None),
                )
                return out.last_hidden_state  # [B, T, H]
            out = self._base_model(tokens)
            return out.last_hidden_state

        # Default: CausalLM base model
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
