from __future__ import annotations

from typing import Any


SMOLLM2_TOKENIZER = "HuggingFaceTB/SmolLM2-135M"


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text: str, **_: Any) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, tokens: list[int], **_: Any) -> str:
        return "".join(chr(int(token)) for token in tokens)

    def batch_decode(self, batch_tokens: list[list[int]], **kwargs: Any) -> list[str]:
        return [self.decode(tokens, **kwargs) for tokens in batch_tokens]


class TokenizerWrapper:
    def __init__(self, tokenizer: Any | None = None, model_name: str | None = None, **kwargs: Any) -> None:
        if tokenizer is not None:
            self.tokenizer = tokenizer
            self._ensure_padding_token()
            return

        if model_name is not None:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
                self._ensure_padding_token()
                return
            except ImportError:
                pass

        self.tokenizer = _DummyTokenizer()

    @classmethod
    def smollm2(cls, model_name: str = SMOLLM2_TOKENIZER, **kwargs: Any) -> "TokenizerWrapper":
        return cls(model_name=model_name, **kwargs)

    @property
    def pad_token_id(self) -> int | None:
        return getattr(self.tokenizer, "pad_token_id", None)

    @property
    def eos_token_id(self) -> int | None:
        return getattr(self.tokenizer, "eos_token_id", None)

    def _ensure_padding_token(self) -> None:
        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return list(self.tokenizer.encode(text, **kwargs))

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        return str(self.tokenizer.decode(tokens, **kwargs))

    def batch_encode(
        self,
        texts: list[str],
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if hasattr(self.tokenizer, "__call__"):
            return self.tokenizer(
                texts,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                **kwargs,
            )
        encoded = [self.encode(text, **kwargs) for text in texts]
        if truncation and max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]
        if padding:
            pad_to = max_length if max_length is not None else max((len(ids) for ids in encoded), default=0)
            pad_id = self.pad_token_id or 0
            encoded = [ids + [pad_id] * max(0, pad_to - len(ids)) for ids in encoded]
        return encoded

    def batch_decode(self, batch_tokens: list[list[int]], **kwargs: Any) -> list[str]:
        if hasattr(self.tokenizer, "batch_decode"):
            return list(self.tokenizer.batch_decode(batch_tokens, **kwargs))
        return [self.decode(tokens, **kwargs) for tokens in batch_tokens]
