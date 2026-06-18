from __future__ import annotations

from typing import Any
from collections.abc import Callable, Iterator

import torch
import torch.nn.functional as F

from hagi.inference.generate import generate, generate_with_rollouts, stream_generate
from hagi.inference.online import FeedbackBuffer, OnlineLearner
from hagi.inference.lora import apply_lora_to_model
from hagi.model.msa import SlotRegistry


class ChatSession:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = 50,
        top_p: float | None = 0.9,
        eos_token_id: int | None = None,
        system_prompt: str | None = None,
        max_context_length: int | None = None,
        clear_cuda_cache: bool = True,
        compile_model: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        auto_learn_after: int = 3,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.history: list[tuple[str, str]] = []
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.eos_token_id = eos_token_id
        self.system_prompt = system_prompt
        self.max_context_length = max_context_length
        self.clear_cuda_cache = clear_cuda_cache
        self.compile_model = compile_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.rollouts = 1
        self.noise_sigma = 0.0
        self._feedback_max_size = 256
        self.feedback_buffer = FeedbackBuffer(max_size=self._feedback_max_size)
        self.lora_adapter = None
        self.online_learner = None
        self.auto_learn_after = auto_learn_after
        self._positive_feedback_count = 0
        self._last_prompt_ids: list[int] = []
        self._last_response_ids: list[int] = []
        self._msa_session_registry: Any | None = None
        if getattr(getattr(model, "cfg", None), "use_msa", False):
            self._msa_session_registry = SlotRegistry(
                max_slots=model.cfg.msa_slot_count
            )

    def add_user_message(self, text: str) -> None:
        self.history.append(("user", text))

    def add_assistant_message(self, text: str) -> None:
        self.history.append(("assistant", text))

    def set_system_prompt(self, text: str | None) -> None:
        self.system_prompt = text or None

    def clear(self) -> None:
        self.history.clear()
        self._maybe_clear_cuda_cache()

    def _maybe_clear_cuda_cache(self) -> None:
        if self.clear_cuda_cache and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _render_prompt(self) -> str:
        parts: list[str] = []
        if self.system_prompt:
            parts.append(f"<System>\n{self.system_prompt}\n</System>")
        for role, text in self.history:
            marker = "User" if role == "user" else "Assistant"
            parts.append(f"<{marker}>\n{text}\n</{marker}>")
        parts.append("<Assistant>\n")
        return "\n".join(parts)

    def _prompt_ids(self) -> list[int]:
        prompt_ids = self.tokenizer.encode(self._render_prompt())
        if (
            self.max_context_length is not None
            and len(prompt_ids) > self.max_context_length
        ):
            prompt_ids = prompt_ids[-self.max_context_length :]
        return prompt_ids

    def generate_response(self) -> str:
        prompt_ids = self._prompt_ids()
        self._last_prompt_ids = list(prompt_ids)
        self._last_response_ids = []
        if self.rollouts > 1 and self.noise_sigma > 0.0:
            generated_ids = generate_with_rollouts(
                self.model,
                prompt_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                eos_token_id=self.eos_token_id,
                rollouts=self.rollouts,
                noise_sigma=self.noise_sigma,
                use_cache=True,
                compile_model=self.compile_model,
                external_msa_registry=self._msa_session_registry,
            )
        else:
            generated_ids = generate(
                self.model,
                prompt_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                eos_token_id=self.eos_token_id,
                use_cache=True,
                compile_model=self.compile_model,
                external_msa_registry=self._msa_session_registry,
                use_static_cache=True,
            )
        new_ids = generated_ids[0, len(prompt_ids) :].tolist()
        text = self.tokenizer.decode(new_ids)
        self.add_assistant_message(text)
        self._capture_last_ids(prompt_ids, generated_ids)
        full_ids = torch.cat(
            [
                torch.tensor(
                    [self._last_prompt_ids],
                    dtype=torch.long,
                    device=generated_ids.device,
                ),
                generated_ids[:, len(self._last_prompt_ids) :],
            ],
            dim=-1,
        )
        self._observe_nars(full_ids)
        self._maybe_clear_cuda_cache()
        return text

    def stream_response(self) -> Iterator[str]:
        prompt_ids = self._prompt_ids()
        self._last_prompt_ids = list(prompt_ids)
        self._last_response_ids = []
        pieces: list[str] = []
        generated_token_ids: list[int] = []
        for token in stream_generate(
            self.model,
            prompt_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            eos_token_id=self.eos_token_id,
            use_cache=True,
            compile_model=self.compile_model,
            external_msa_registry=self._msa_session_registry,
            use_static_cache=True,
        ):
            token_ids = token.tolist() if hasattr(token, "tolist") else token
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            elif token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            generated_token_ids.extend(token_ids)
            piece = self.tokenizer.decode(token_ids)
            pieces.append(piece)
            yield piece
        self.add_assistant_message("".join(pieces))
        self._last_response_ids = list(generated_token_ids)
        if torch is not None and self._last_prompt_ids:
            device = next(self.model.parameters()).device
            full_ids = torch.cat(
                [
                    torch.tensor(
                        [self._last_prompt_ids], dtype=torch.long, device=device
                    ),
                    torch.tensor(
                        [generated_token_ids], dtype=torch.long, device=device
                    ),
                ],
                dim=-1,
            )
            self._observe_nars(full_ids)
        self._maybe_clear_cuda_cache()

    def mark_response(self, reward: float) -> bool:
        if not self._last_response_ids:
            return False
        self.feedback_buffer.add(self._last_prompt_ids, self._last_response_ids, reward)
        if reward > 0:
            self._positive_feedback_count += 1
            if self._positive_feedback_count >= self.auto_learn_after:
                self.learn_step()
                self._positive_feedback_count = 0
        return True

    def learn_step(self) -> float | None:
        if self.online_learner is None:
            self._init_lora()
        assert self.online_learner is not None
        return self.online_learner.learn_step(
            self.feedback_buffer,
            forward_fn=self._make_forward_fn(),
            batch_size=4,
        )

    def _init_lora(self) -> None:
        self.lora_adapter, _ = apply_lora_to_model(
            self.model, rank=self.lora_rank, alpha=self.lora_alpha
        )
        self.online_learner = OnlineLearner(self.lora_adapter)
        self._maybe_clear_cuda_cache()

    def _make_forward_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        def forward_fn(ids: torch.Tensor) -> torch.Tensor:
            with torch.enable_grad():
                output = self.model(
                    ids,
                    external_msa_registry=self._msa_session_registry,
                    use_cache=True,
                    training_mode=True,
                )
            if isinstance(output, dict):
                return output["logits"]
            if isinstance(output, tuple):
                return output[0]
            return output

        return forward_fn

    def _capture_last_ids(
        self, prompt_ids: list[int], generated_ids: torch.Tensor
    ) -> None:
        self._last_prompt_ids = list(prompt_ids)
        self._last_response_ids = generated_ids[0, len(prompt_ids) :].tolist()

    @torch.no_grad() if torch is not None else (lambda fn: fn)
    def _observe_nars(self, sequence_ids: torch.Tensor) -> None:
        if not hasattr(self.model, "nars_hrm") or self.model.nars_hrm is None:
            return
        if torch is None:
            return
        try:
            output = self.model(sequence_ids)
            logits = output[0] if isinstance(output, tuple) else output
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                sequence_ids[:, 1:].reshape(-1),
            ).item()
            # grad_norm=0.0: no gradient under no_grad; loss-only signal.
            self.model.nars_hrm.observe_train_step(loss, grad_norm=0.0)
        except Exception:
            # Best-effort: NARS observation must never crash chat.
            pass

    def save_adapter(self, path: str) -> None:
        if self.online_learner is None:
            raise RuntimeError(
                "No adapter to save. Provide positive feedback first to trigger learning."
            )
        self.online_learner.save(path)

    def load_adapter(self, path: str) -> None:
        if self.online_learner is None:
            self._init_lora()
        assert self.online_learner is not None
        self.online_learner.load(path)

    def _zero_lora_adapters(self) -> None:
        if self.lora_adapter is not None:
            for adapter in self.lora_adapter:
                b = getattr(adapter, "B", None)
                if isinstance(b, torch.Tensor):
                    with torch.no_grad():
                        b.zero_()

    def reset_adapter(self) -> None:
        self._zero_lora_adapters()
        self.lora_adapter = None
        self.online_learner = None
        self.feedback_buffer = FeedbackBuffer(max_size=self._feedback_max_size)
        self._positive_feedback_count = 0
        self._maybe_clear_cuda_cache()
