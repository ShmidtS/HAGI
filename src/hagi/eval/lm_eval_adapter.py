from __future__ import annotations

from typing import Any


class LMEvalAdapter:
    def __init__(self, model: Any | None = None, tokenizer: Any | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def loglikelihood(self, requests: Any) -> list[Any]:
        raise NotImplementedError("lm-eval loglikelihood integration is not implemented yet")

    def generate_until(self, requests: Any) -> list[str]:
        raise NotImplementedError("lm-eval generate_until integration is not implemented yet")
