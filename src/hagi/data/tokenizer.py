from __future__ import annotations

from typing import Any


class _DummyTokenizer:
    def encode(self, text: str, **_: Any) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, tokens: list[int], **_: Any) -> str:
        return "".join(chr(int(token)) for token in tokens)


class TokenizerWrapper:
    def __init__(self, tokenizer: Any | None = None, model_name: str | None = None, **kwargs: Any) -> None:
        if tokenizer is not None:
            self.tokenizer = tokenizer
            return

        if model_name is not None:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
                return
            except ImportError:
                pass

        self.tokenizer = _DummyTokenizer()

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return list(self.tokenizer.encode(text, **kwargs))

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        return str(self.tokenizer.decode(tokens, **kwargs))
