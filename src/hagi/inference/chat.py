from __future__ import annotations

from typing import Any

from hagi.inference.generate import generate


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
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.history: list[tuple[str, str]] = []
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.eos_token_id = eos_token_id

    def add_user_message(self, text: str) -> None:
        self.history.append(("user", text))

    def add_assistant_message(self, text: str) -> None:
        self.history.append(("assistant", text))

    def _render_prompt(self) -> str:
        parts: list[str] = []
        for role, text in self.history:
            marker = "User" if role == "user" else "Assistant"
            parts.append(f"<{marker}>\n{text}\n</{marker}>")
        parts.append("<Assistant>\n")
        return "\n".join(parts)

    def generate_response(self) -> str:
        prompt = self._render_prompt()
        prompt_ids = self.tokenizer.encode(prompt)
        generated_ids = generate(
            self.model,
            prompt_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            eos_token_id=self.eos_token_id,
        )
        new_ids = generated_ids[0, len(prompt_ids):].tolist()
        text = self.tokenizer.decode(new_ids)
        self.add_assistant_message(text)
        return text
