from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch is an optional runtime fallback
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


@dataclass
class CacheKeyValues:
    layers: list[tuple[Any, Any]]

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        return self.layers[index]

    @classmethod
    def from_model_cache(cls, cache: Any) -> "CacheKeyValues":
        if isinstance(cache, cls):
            return cache
        return cls(list(cache or []))

    def to_model_cache(self) -> list[tuple[Any, Any]]:
        return self.layers


def _filter_top_k(logits: Any, top_k: int | None) -> Any:
    if top_k is None or top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    if torch is not None and torch.is_tensor(logits):
        values, _ = torch.topk(logits, top_k)
        threshold = values[..., -1, None]
        return logits.masked_fill(logits < threshold, float("-inf"))
    indices = np.argpartition(logits, -top_k, axis=-1)[..., :-top_k]
    filtered = np.array(logits, copy=True)
    np.put_along_axis(filtered, indices, -np.inf, axis=-1)
    return filtered


def _filter_top_p(logits: Any, top_p: float | None) -> Any:
    if top_p is None or top_p <= 0.0 or top_p >= 1.0:
        return logits
    if torch is not None and torch.is_tensor(logits):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        return logits.masked_fill(indices_to_remove, float("-inf"))

    sorted_indices = np.argsort(-logits, axis=-1)
    sorted_logits = np.take_along_axis(logits, sorted_indices, axis=-1)
    sorted_probs = _softmax_np(sorted_logits)
    sorted_indices_to_remove = np.cumsum(sorted_probs, axis=-1) > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
    sorted_indices_to_remove[..., 0] = False
    filtered = np.array(logits, copy=True)
    np.put_along_axis(filtered, sorted_indices, np.where(sorted_indices_to_remove, -np.inf, sorted_logits), axis=-1)
    return filtered


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def sample_next_token(
    logits: Any,
    temperature: float = 1.0,
    top_k: int | None = 50,
    top_p: float | None = 0.9,
) -> Any:
    """Sample the next token id from final-position logits."""
    if torch is not None and torch.is_tensor(logits):
        if logits.dim() > 1:
            logits = logits[..., -1, :] if logits.dim() == 3 else logits
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)
        logits = _filter_top_p(_filter_top_k(logits / temperature, top_k), top_p)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    logits = np.asarray(logits)
    if logits.ndim > 1:
        logits = logits[..., -1, :] if logits.ndim == 3 else logits
    if temperature <= 0:
        return np.argmax(logits, axis=-1)
    probs = _softmax_np(_filter_top_p(_filter_top_k(logits / temperature, top_k), top_p))
    if probs.ndim == 1:
        return np.array(np.random.choice(probs.shape[-1], p=probs))
    return np.array([np.random.choice(probs.shape[-1], p=row) for row in probs])


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return None


def _maybe_compile(model: Any, compile_model: bool) -> Any:
    if not compile_model or torch is None or not hasattr(torch, "compile"):
        return model
    device = _model_device(model)
    if device is not None and device.type == "cuda":
        return torch.compile(model)
    return model


def _split_output(output: Any) -> tuple[Any, CacheKeyValues | None]:
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], list):
        return output[0], CacheKeyValues.from_model_cache(output[1])
    return (output[0] if isinstance(output, tuple) else output), None


def _forward(model: Any, input_ids: Any, cache: CacheKeyValues | None, use_cache: bool) -> tuple[Any, CacheKeyValues | None]:
    if use_cache:
        try:
            output = model(input_ids, past_key_values=cache.to_model_cache() if cache is not None else None, use_cache=True)
            return _split_output(output)
        except TypeError:
            pass
    return _split_output(model(input_ids))


@torch.no_grad() if torch is not None else (lambda fn: fn)
def generate(
    model: Any,
    prompt_ids: Any,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int | None = 50,
    top_p: float | None = 0.9,
    eos_token_id: int | None = None,
    cache: CacheKeyValues | None = None,
    use_cache: bool = True,
    compile_model: bool = False,
) -> Any:
    """Generate token ids with optional KV-cache acceleration."""
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    model = _maybe_compile(model, compile_model)

    if torch is not None:
        generated = prompt_ids
        if not torch.is_tensor(generated):
            generated = torch.tensor(generated, dtype=torch.long)
        if generated.dim() == 1:
            generated = generated.unsqueeze(0)
        device = _model_device(model)
        if device is not None:
            generated = generated.to(device)

        next_input = generated if cache is None else generated[:, -1:]
        active_cache = cache
        for _ in range(max_new_tokens):
            logits, active_cache = _forward(model, next_input, active_cache, use_cache)
            next_token = sample_next_token(logits, temperature, top_k, top_p)
            if next_token.dim() == 0:
                next_token = next_token.unsqueeze(0)
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            next_input = next_token.unsqueeze(-1)
    else:
        generated = np.asarray(prompt_ids, dtype=np.int64)
        if generated.ndim == 1:
            generated = generated[None, :]

        for _ in range(max_new_tokens):
            output = model(generated)
            logits = output[0] if isinstance(output, tuple) else output
            next_token = np.asarray(sample_next_token(logits, temperature, top_k, top_p), dtype=np.int64)
            if next_token.ndim == 0:
                next_token = next_token[None]
            generated = np.concatenate([generated, next_token[:, None]], axis=-1)
            if eos_token_id is not None and np.all(next_token == eos_token_id):
                break

    if was_training and hasattr(model, "train"):
        model.train()
    return generated


@torch.no_grad() if torch is not None else (lambda fn: fn)
def stream_generate(
    model: Any,
    prompt_ids: Any,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int | None = 50,
    top_p: float | None = 0.9,
    eos_token_id: int | None = None,
    cache: CacheKeyValues | None = None,
    use_cache: bool = True,
    compile_model: bool = False,
) -> Iterator[Any]:
    """Yield next token ids as they are generated."""
    if torch is None:
        generated = np.asarray(prompt_ids, dtype=np.int64)
        if generated.ndim == 1:
            generated = generated[None, :]
        for _ in range(max_new_tokens):
            output = model(generated)
            logits = output[0] if isinstance(output, tuple) else output
            next_token = np.asarray(sample_next_token(logits, temperature, top_k, top_p), dtype=np.int64)
            if next_token.ndim == 0:
                next_token = next_token[None]
            yield next_token
            generated = np.concatenate([generated, next_token[:, None]], axis=-1)
            if eos_token_id is not None and np.all(next_token == eos_token_id):
                break
        return

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    model = _maybe_compile(model, compile_model)
    generated = prompt_ids if torch.is_tensor(prompt_ids) else torch.tensor(prompt_ids, dtype=torch.long)
    if generated.dim() == 1:
        generated = generated.unsqueeze(0)
    device = _model_device(model)
    if device is not None:
        generated = generated.to(device)
    next_input = generated if cache is None else generated[:, -1:]
    active_cache = cache
    for _ in range(max_new_tokens):
        logits, active_cache = _forward(model, next_input, active_cache, use_cache)
        next_token = sample_next_token(logits, temperature, top_k, top_p)
        if next_token.dim() == 0:
            next_token = next_token.unsqueeze(0)
        yield next_token
        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break
        next_input = next_token.unsqueeze(-1)
    if was_training and hasattr(model, "train"):
        model.train()
