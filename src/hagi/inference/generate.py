from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Iterator

import numpy as np

try:
    import torch
    import torch.nn.functional as _f
except ImportError:  # pragma: no cover - torch is an optional runtime fallback
    torch = None  # type: ignore[assignment]
    _f = None  # type: ignore[assignment]


@dataclass
class CacheKeyValues:
    layers: list[Any]

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> Any:
        return self.layers[index]

    @classmethod
    def from_model_cache(cls, cache: Any) -> CacheKeyValues:
        if isinstance(cache, cls):
            return cache
        return cls(list(cache or []))

    def to_model_cache(self) -> list[Any]:
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
        assert _f is not None
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = _f.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        return logits.masked_fill(indices_to_remove, float("-inf"))

    sorted_indices = np.argsort(-logits, axis=-1)
    sorted_logits = np.take_along_axis(logits, sorted_indices, axis=-1)
    sorted_probs = _softmax_np(sorted_logits)
    sorted_indices_to_remove = np.cumsum(sorted_probs, axis=-1) > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
    sorted_indices_to_remove[..., 0] = False
    filtered = np.array(logits, copy=True)
    np.put_along_axis(
        filtered,
        sorted_indices,
        np.where(sorted_indices_to_remove, -np.inf, sorted_logits),
        axis=-1,
    )
    return filtered


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    # Guard against all -inf (from aggressive top-k/top-p filtering)
    logits = np.where(np.isneginf(logits), -1e9, logits)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def confidence_score(logits: Any) -> float:
    """Scalar confidence from top-2 logit gap (PTRM idea)."""
    if torch is not None and torch.is_tensor(logits):
        vals, _ = torch.topk(logits, k=2, dim=-1)
        gap = vals[..., 0] - vals[..., 1]
        return float(gap.clamp(-10.0, 10.0).mean().item() * 0.1)
    vals = np.partition(-logits, 2, axis=-1)[..., :2]
    gap = -vals[..., 0] + vals[..., 1]
    return float(np.clip(gap, -10.0, 10.0) * 0.1)


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
        assert _f is not None
        probs = _f.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    logits = np.asarray(logits)
    if logits.ndim > 1:
        logits = logits[..., -1, :] if logits.ndim == 3 else logits
    if temperature <= 0:
        return np.argmax(logits, axis=-1)
    probs = _softmax_np(
        _filter_top_p(_filter_top_k(logits / temperature, top_k), top_p)
    )
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
    import sys

    if sys.platform == "win32":
        return model
    device = _model_device(model)
    if device is not None and device.type == "cuda":
        return torch.compile(model)
    return model


def _split_output(output: Any) -> tuple[Any, CacheKeyValues | None]:
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], list):
        return output[0], CacheKeyValues.from_model_cache(output[1])
    return (output[0] if isinstance(output, tuple) else output), None


def _cache_is_empty(cache: Any) -> bool:
    """True when there is no cache or it is a fresh (unwritten) static cache."""
    if cache is None:
        return True
    layers = getattr(cache, "layers", cache)
    if not layers:
        return True
    return getattr(layers[0], "seq_len", None) == 0


def _maybe_static_cache(
    model: Any,
    generated: Any,
    max_new_tokens: int,
    cache: Any,
    use_cache: bool,
    use_static_cache: bool,
) -> Any:
    """Preallocate a static KV cache (write-by-index, no per-step torch.cat)."""
    if cache is not None or not use_static_cache or not use_cache or torch is None:
        return cache
    try:
        from hagi.model.kv_cache import make_static_cache
    except ImportError:
        return cache
    layers = make_static_cache(
        model, generated.size(0), generated.size(1) + max_new_tokens
    )
    if layers is None:
        return cache
    return CacheKeyValues(layers)


def _forward(
    model: Any, input_ids: Any, cache: CacheKeyValues | None, use_cache: bool
) -> tuple[Any, CacheKeyValues | None]:
    if use_cache:
        try:
            output = model(
                input_ids,
                past_key_values=cache.to_model_cache() if cache is not None else None,
                use_cache=True,
            )
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
    pin_memory: bool = False,
    training_mode: bool = False,
    use_static_cache: bool = False,
) -> Any:
    """Generate token ids with optional KV-cache acceleration.

    use_static_cache=True preallocates per-block buffers written by index
    (no per-step torch.cat); requires a HAGI-style model with .cfg.
    """
    was_training = bool(getattr(model, "training", False))
    if not training_mode and hasattr(model, "eval"):
        model.eval()

    if hasattr(model, "clear_rope_cache"):
        model.clear_rope_cache()

    if pin_memory and torch is not None:
        from hagi.model.inference_opt import pin_model_weights

        pin_model_weights(model)

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

        cache = _maybe_static_cache(
            model, generated, max_new_tokens, cache, use_cache, use_static_cache
        )
        # An empty (fresh static) cache still needs the full prompt for prefill.
        next_input = generated if _cache_is_empty(cache) else generated[:, -1:]
        active_cache = cache
        generated_tokens: list[Any] = []
        for _ in range(max_new_tokens):
            logits, active_cache = _forward(model, next_input, active_cache, use_cache)
            next_token = sample_next_token(logits, temperature, top_k, top_p)
            if next_token.dim() == 0:
                next_token = next_token.unsqueeze(0)
            generated_tokens.append(next_token.unsqueeze(-1))
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            next_input = next_token.unsqueeze(-1)
        if generated_tokens:
            generated = torch.cat([generated, *generated_tokens], dim=-1)
    else:
        generated = np.asarray(prompt_ids, dtype=np.int64)
        if generated.ndim == 1:
            generated = generated[None, :]

        for _ in range(max_new_tokens):
            output = model(generated)
            logits = output[0] if isinstance(output, tuple) else output
            next_token = np.asarray(
                sample_next_token(logits, temperature, top_k, top_p), dtype=np.int64
            )
            if next_token.ndim == 0:
                next_token = next_token[None]
            generated = np.concatenate([generated, next_token[:, None]], axis=-1)
            if eos_token_id is not None and np.all(next_token == eos_token_id):
                break

    if not training_mode and was_training and hasattr(model, "train"):
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
    pin_memory: bool = False,
    use_static_cache: bool = False,
) -> Iterator[Any]:
    """Yield next token ids as they are generated."""
    if torch is None:
        generated = np.asarray(prompt_ids, dtype=np.int64)
        if generated.ndim == 1:
            generated = generated[None, :]
        for _ in range(max_new_tokens):
            output = model(generated)
            logits = output[0] if isinstance(output, tuple) else output
            next_token = np.asarray(
                sample_next_token(logits, temperature, top_k, top_p), dtype=np.int64
            )
            if next_token.ndim == 0:
                next_token = next_token[None]
            yield next_token
            generated = np.concatenate([generated, next_token[:, None]], axis=-1)
            if eos_token_id is not None and np.all(next_token == eos_token_id):
                break
        return

    if hasattr(model, "clear_rope_cache"):
        model.clear_rope_cache()

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()

    if pin_memory and torch is not None:
        from hagi.model.inference_opt import pin_model_weights

        pin_model_weights(model)

    model = _maybe_compile(model, compile_model)
    generated = (
        prompt_ids
        if torch.is_tensor(prompt_ids)
        else torch.tensor(prompt_ids, dtype=torch.long)
    )
    if generated.dim() == 1:
        generated = generated.unsqueeze(0)
    device = _model_device(model)
    if device is not None:
        generated = generated.to(device)
    cache = _maybe_static_cache(
        model, generated, max_new_tokens, cache, use_cache, use_static_cache
    )
    # An empty (fresh static) cache still needs the full prompt for prefill.
    next_input = generated if _cache_is_empty(cache) else generated[:, -1:]
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


@torch.no_grad() if torch is not None else (lambda fn: fn)
def generate_with_rollouts(
    model: Any,
    prompt_ids: Any,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int | None = 50,
    top_p: float | None = 0.9,
    eos_token_id: int | None = None,
    rollouts: int = 1,
    noise_sigma: float = 0.0,
    use_cache: bool = True,
    compile_model: bool = False,
) -> Any:
    """Generate with multiple noisy rollouts, select best by confidence (PTRM idea)."""
    if rollouts <= 1 or noise_sigma <= 0.0:
        return generate(
            model,
            prompt_ids,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            eos_token_id,
            use_cache=use_cache,
            compile_model=compile_model,
        )

    best_generated = None
    best_score = float("-inf")
    result = None
    compiled_model = _maybe_compile(model, compile_model)
    for k in range(rollouts):
        old_noise = 0.0
        if hasattr(model, "cfg") and hasattr(model.cfg, "thinking_noise"):
            old_noise = model.cfg.thinking_noise
            model.cfg.thinking_noise = noise_sigma
        try:
            result = generate(
                compiled_model,
                prompt_ids,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                eos_token_id,
                use_cache=use_cache,
                compile_model=False,
                training_mode=True,
            )
        finally:
            if hasattr(model, "cfg") and hasattr(model.cfg, "thinking_noise"):
                model.cfg.thinking_noise = old_noise
        if torch is not None and torch.is_tensor(result):
            with torch.no_grad():
                score_output = compiled_model(result[:, -1:], training_mode=True)
                if isinstance(score_output, dict) and "quality_score" in score_output:
                    score = score_output["quality_score"].mean().item()
                else:
                    last_logits = (
                        score_output[0]
                        if isinstance(score_output, tuple)
                        else score_output["logits"]
                        if isinstance(score_output, dict)
                        else score_output
                    )
                    score = confidence_score(last_logits[..., -1, :])
                if score > best_score:
                    best_score = score
                    best_generated = result
    return best_generated if best_generated is not None else result
