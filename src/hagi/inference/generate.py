from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch is an optional runtime fallback
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


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


@torch.no_grad() if torch is not None else (lambda fn: fn)
def generate(
    model: Any,
    prompt_ids: Any,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int | None = 50,
    top_p: float | None = 0.9,
    eos_token_id: int | None = None,
) -> Any:
    """Generate token ids from a model in eager mode."""
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()

    if torch is not None:
        generated = prompt_ids
        if not torch.is_tensor(generated):
            generated = torch.tensor(generated, dtype=torch.long)
        if generated.dim() == 1:
            generated = generated.unsqueeze(0)

        for _ in range(max_new_tokens):
            output = model(generated)
            logits = output[0] if isinstance(output, tuple) else output
            next_token = sample_next_token(logits, temperature, top_k, top_p)
            if next_token.dim() == 0:
                next_token = next_token.unsqueeze(0)
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
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
