from __future__ import annotations

from typing import Any

from hagi.utils.env import load_env
load_env()

SMOLLM2_TOKENIZER = "HuggingFaceTB/SmolLM2-135M"


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    pad_token = None
    eos_token = None

    def __call__(self, texts, **kwargs: Any) -> Any:
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [self.encode(t) for t in texts]}

    def encode(self, text: str, **_: Any) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, tokens: list[int], **_: Any) -> str:
        return "".join(chr(int(token)) for token in tokens)

    def batch_decode(self, batch_tokens: list[list[int]], **kwargs: Any) -> list[str]:
        return [self.decode(tokens, **kwargs) for tokens in batch_tokens]


class TokenizerWrapper:
    def __init__(
        self, model_name: str | None = None, tokenizer: Any | None = None, **kwargs: Any
    ) -> None:
        if model_name is not None:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
                self._ensure_padding_token()
                return
            except ImportError:
                pass

        if tokenizer is not None:
            self.tokenizer = tokenizer
            self._ensure_padding_token()
            return

        self.tokenizer = _DummyTokenizer()

    @classmethod
    def smollm2(
        cls, model_name: str = SMOLLM2_TOKENIZER, **kwargs: Any
    ) -> TokenizerWrapper:
        return cls(model_name=model_name, **kwargs)

    @property
    def pad_token_id(self) -> int | None:
        return getattr(self.tokenizer, "pad_token_id", None)

    @property
    def eos_token_id(self) -> int | None:
        return getattr(self.tokenizer, "eos_token_id", None)

    def _ensure_padding_token(self) -> None:
        assert self.tokenizer is not None
        if (
            getattr(self.tokenizer, "pad_token", None) is None
            and getattr(self.tokenizer, "eos_token", None) is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, texts, **kwargs: Any) -> Any:
        assert self.tokenizer is not None
        return self.tokenizer.__call__(texts, **kwargs)

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        assert self.tokenizer is not None
        return list(self.tokenizer.encode(text, **kwargs))

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        assert self.tokenizer is not None
        return str(self.tokenizer.decode(tokens, **kwargs))

    def batch_decode(self, batch_tokens: list[list[int]], **kwargs: Any) -> list[str]:
        if hasattr(self.tokenizer, "batch_decode"):
            return list(self.tokenizer.batch_decode(batch_tokens, **kwargs))
        return [self.decode(tokens, **kwargs) for tokens in batch_tokens]

    def fast_batch_encode(self, texts: list[str], **kwargs: Any) -> list[list[int]]:
        """Use underlying Rust tokenizer directly — no BatchEncoding overhead."""
        assert self.tokenizer is not None
        # FastTokenizer backend (PreTrainedTokenizerFast) exposes ._tokenizer
        fast = getattr(self.tokenizer, "_tokenizer", None)
        if fast is not None:
            return [
                enc.ids for enc in fast.encode_batch(texts, add_special_tokens=False)
            ]
        # Fallback to slow path
        out = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=8192,
            padding=False,
        )
        return out["input_ids"]
