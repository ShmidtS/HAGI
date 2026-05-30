from __future__ import annotations

from typing import Any, Iterator

try:
    import torch
except ImportError:  # pragma: no cover - torch is an optional runtime fallback
    torch = None  # type: ignore[assignment]

from hagi.inference.generate import generate, stream_generate


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
        if self.max_context_length is not None and len(prompt_ids) > self.max_context_length:
            prompt_ids = prompt_ids[-self.max_context_length :]
        return prompt_ids

    def generate_response(self) -> str:
        prompt_ids = self._prompt_ids()
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
        )
        new_ids = generated_ids[0, len(prompt_ids):].tolist()
        text = self.tokenizer.decode(new_ids)
        self.add_assistant_message(text)
        self._maybe_clear_cuda_cache()
        return text

    def stream_response(self) -> Iterator[str]:
        prompt_ids = self._prompt_ids()
        pieces: list[str] = []
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
        ):
            token_ids = token.tolist() if hasattr(token, "tolist") else token
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            elif token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            piece = self.tokenizer.decode(token_ids)
            pieces.append(piece)
            yield piece
        self.add_assistant_message("".join(pieces))
        self._maybe_clear_cuda_cache()
