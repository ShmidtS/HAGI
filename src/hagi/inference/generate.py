from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from collections.abc import Callable, Iterator

import numpy as np

if TYPE_CHECKING:
    import torch
    import torch.nn.functional as _f
else:
    try:
        import torch
        import torch.nn.functional as _f
    except ImportError:  # pragma: no cover - torch is an optional runtime fallback
        torch = None
        _f = None


TokenSink = Callable[[Any], None]


@dataclass(frozen=True)
class GenerationConfig:
    """Immutable generation parameters (VibeHarness Config pattern).

    Consolidates all generate() kwargs into one value object so callers
    can pass a single config instead of 15 positional arguments.
    """

    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int | None = 50
    top_p: float | None = 0.9
    eos_token_id: int | None = None
    stop_sequences: tuple[tuple[int, ...], ...] = ()
    use_cache: bool = True
    use_static_cache: bool = False
    compile_model: bool = False
    pin_memory: bool = False
    training_mode: bool = False
    early_exit_confidence: float = 0.0


@dataclass
class GenerationResult:
    """Structured generation output (VibeHarness Decision pattern).

    Separates the raw token ids from metadata like finish reason
    and per-step confidence scores.
    """

    ids: Any
    finish_reason: str = "length"
    tokens_generated: int = 0
    confidences: list[float] = field(default_factory=list)


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
    vals = np.partition(-logits, 1, axis=-1)[..., :2]
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
    if torch is None:
        return None
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        return None
    return device if isinstance(device, torch.device) else None


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
    if cache is None:
        return True
    layers = getattr(cache, "layers", cache)
    if not layers:
        return True
    return getattr(layers[0], "seq_len", None) == 0


def _get_cast_k(model: Any) -> int:
    """Return CAST block size K, or 0 if CAST is not enabled."""
    cast_cfg = getattr(getattr(model, "cfg", None), "cast_config", None)
    return cast_cfg.block_size if cast_cfg is not None else 0


def _maybe_static_cache(
    model: Any,
    generated: Any,
    max_new_tokens: int,
    cache: Any,
    use_cache: bool,
    use_static_cache: bool,
) -> Any:
    """Preallocate a static KV cache (write-by-index, no per-step torch.cat).

    When ``model.cfg.int8_kv_cache`` is True, uses INT8-quantized cache
    (2x memory reduction). Falls back to bf16 cache otherwise.

    With CAST block-wise generation, each forward pass adds K tokens to
    the cache. The total positions needed is prompt_len + ceil(N/K)*K,
    which can exceed prompt_len + N by up to K-1. We round up to avoid
    cache overflow on the last block.
    """
    if cache is not None or not use_static_cache or not use_cache or torch is None:
        return cache
    try:
        from hagi.model.kv_cache import make_int8_static_cache, make_static_cache
    except ImportError:
        return cache
    use_int8 = bool(getattr(getattr(model, "cfg", None), "int8_kv_cache", False))
    factory = make_int8_static_cache if use_int8 else make_static_cache
    cast_k = _get_cast_k(model)
    total_new = max_new_tokens
    if cast_k > 1:
        total_new = ((max_new_tokens + cast_k - 1) // cast_k) * cast_k
    layers = factory(model, generated.size(0), generated.size(1) + total_new)
    if layers is None:
        return cache
    return CacheKeyValues(layers)


def _prepare_torch_inputs(
    model: Any,
    prompt_ids: Any,
    max_new_tokens: int,
    cache: CacheKeyValues | None,
    use_cache: bool,
    use_static_cache: bool,
) -> tuple[Any, Any, CacheKeyValues | None]:
    """Coerce the prompt to a batched long tensor on the model device, then build
    the static KV cache (if requested) and pick the first forward input.

    Returns (generated, next_input, active_cache). Pure — no model side-effects.
    """
    assert torch is not None
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
    next_input = generated if _cache_is_empty(cache) else generated[:, -1:]
    return generated, next_input, cache


def _forward(
    model: Any,
    input_ids: Any,
    cache: CacheKeyValues | None,
    use_cache: bool,
    external_msa_registry: Any | None,
) -> tuple[Any, CacheKeyValues | None]:
    if use_cache:
        try:
            output = model(
                input_ids,
                past_key_values=cache.to_model_cache() if cache is not None else None,
                use_cache=True,
                external_msa_registry=external_msa_registry,
            )
            return _split_output(output)
        except TypeError:
            pass
    try:
        return _split_output(
            model(input_ids, external_msa_registry=external_msa_registry)
        )
    except TypeError:
        return _split_output(model(input_ids))


def _check_stop_sequences(
    generated_tokens: list[Any],
    stop_sequences: tuple[tuple[int, ...], ...],
) -> bool:
    """Check if the tail of generated_tokens matches any stop sequence."""
    if not stop_sequences or not generated_tokens:
        return False
    for seq in stop_sequences:
        if len(generated_tokens) < len(seq):
            continue
        tail = generated_tokens[-len(seq) :]
        if all(
            t.item() if hasattr(t, "item") else t == s
            for t, s in zip(tail, seq)
        ):
            return True
    return False


def _extract_block_logits(logits: Any) -> Any:
    """Extract [B, K, V] from CAST 4D logits [B, T, K, V], else [B, V]."""
    if logits is None:
        return None
    if logits.dim() == 4:
        return logits[:, -1]
    if logits.dim() == 3:
        return logits
    return logits


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
    external_msa_registry: Any | None = None,
    early_exit_confidence: float = 0.0,
    stop_sequences: tuple[tuple[int, ...], ...] | list[list[int]] | None = None,
    on_token: TokenSink | None = None,
    config: GenerationConfig | None = None,
) -> Any:
    """Generate token ids with optional KV-cache acceleration.

    When the model has CAST (Clifford Algebra Symbolic Reasoning Tokens) enabled,
    generation runs in block-wise mode: each forward pass produces K tokens
    instead of 1, reducing the number of sequential forward passes by K.

    VibeHarness patterns applied:
    - ``on_token`` callback (TokenSink) for live streaming without generator overhead
    - ``stop_sequences`` for early termination on arbitrary token sequences
    - ``config`` parameter accepts a frozen GenerationConfig value object
    - ``early_exit_confidence`` for confidence-based stopping

    Args:
        stop_sequences: List of token-id sequences that trigger early stop.
            e.g. [[1, 2, 3]] stops when tokens 1,2,3 appear at the tail.
        on_token: Callback invoked with each generated token id tensor.
        config: Optional GenerationConfig; when provided, individual kwargs
            are ignored in favor of config fields.
    """
    if config is not None:
        max_new_tokens = config.max_new_tokens
        temperature = config.temperature
        top_k = config.top_k
        top_p = config.top_p
        eos_token_id = config.eos_token_id
        use_cache = config.use_cache
        use_static_cache = config.use_static_cache
        compile_model = config.compile_model
        pin_memory = config.pin_memory
        training_mode = config.training_mode
        early_exit_confidence = config.early_exit_confidence
        stop_sequences = config.stop_sequences or None

    if stop_sequences is not None:
        stop_sequences = tuple(tuple(s) for s in stop_sequences)
    else:
        stop_sequences = ()

    was_training = bool(getattr(model, "training", False))
    if not training_mode and hasattr(model, "eval"):
        model.eval()

    if hasattr(model, "clear_rope_cache"):
        model.clear_rope_cache()

    if pin_memory and torch is not None:
        from hagi.model.inference_opt import pin_model_weights

        pin_model_weights(model)

    model = _maybe_compile(model, compile_model)

    cast_k = _get_cast_k(model)

    if torch is not None:
        generated, next_input, active_cache = _prepare_torch_inputs(
            model, prompt_ids, max_new_tokens, cache, use_cache, use_static_cache
        )
        generated_tokens: list[Any] = []
        _ee_confidences: list[float] = []

        if cast_k > 0:
            remaining = max_new_tokens
            while remaining > 0:
                logits, active_cache = _forward(
                    model, next_input, active_cache, use_cache, external_msa_registry
                )
                block_logits = _extract_block_logits(logits)

                block_tokens: list[Any] = []
                stop = False
                for k in range(min(cast_k, remaining)):
                    if block_logits.dim() == 3:
                        tok = sample_next_token(
                            block_logits[:, k], temperature, top_k, top_p
                        )
                    else:
                        tok = sample_next_token(
                            block_logits, temperature, top_k, top_p
                        )
                    if tok.dim() == 0:
                        tok = tok.unsqueeze(0)
                    block_tokens.append(tok)
                    if on_token is not None:
                        on_token(tok)
                    if eos_token_id is not None and torch.all(tok == eos_token_id):
                        stop = True
                        break

                generated_tokens.extend(tok.unsqueeze(-1) for tok in block_tokens)

                if not stop and _check_stop_sequences(
                    [t.squeeze() for t in block_tokens], stop_sequences
                ):
                    stop = True

                remaining -= len(block_tokens)
                if stop or remaining <= 0:
                    break

                next_input = torch.stack(block_tokens, dim=-1)
        else:
            for _ in range(max_new_tokens):
                logits, active_cache = _forward(
                    model, next_input, active_cache, use_cache, external_msa_registry
                )
                next_token = sample_next_token(logits, temperature, top_k, top_p)
                if next_token.dim() == 0:
                    next_token = next_token.unsqueeze(0)
                generated_tokens.append(next_token.unsqueeze(-1))
                if on_token is not None:
                    on_token(next_token)
                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break
                if early_exit_confidence > 0.0:
                    _ee_confidences.append(confidence_score(logits))
                    if (
                        len(_ee_confidences) >= 16
                        and sum(_ee_confidences[-16:]) / 16 >= early_exit_confidence
                    ):
                        break
                if _check_stop_sequences(
                    [t.squeeze() for t in generated_tokens], stop_sequences
                ):
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
    external_msa_registry: Any | None = None,
    stop_sequences: tuple[tuple[int, ...], ...] | list[list[int]] | None = None,
    early_exit_confidence: float = 0.0,
    on_token: TokenSink | None = None,
    config: GenerationConfig | None = None,
) -> Iterator[Any]:
    """Yield next token ids as they are generated.

    Supports CAST block-wise generation (K tokens per forward pass),
    stop sequences, early exit confidence, and on_token callback.
    """
    if config is not None:
        max_new_tokens = config.max_new_tokens
        temperature = config.temperature
        top_k = config.top_k
        top_p = config.top_p
        eos_token_id = config.eos_token_id
        use_cache = config.use_cache
        use_static_cache = config.use_static_cache
        compile_model = config.compile_model
        pin_memory = config.pin_memory
        early_exit_confidence = config.early_exit_confidence
        stop_sequences = config.stop_sequences or None

    if stop_sequences is not None:
        stop_sequences = tuple(tuple(s) for s in stop_sequences)
    else:
        stop_sequences = ()

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
            if on_token is not None:
                on_token(next_token)
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
    _generated, next_input, active_cache = _prepare_torch_inputs(
        model, prompt_ids, max_new_tokens, cache, use_cache, use_static_cache
    )

    cast_k = _get_cast_k(model)
    _ee_confidences: list[float] = []
    generated_tokens: list[Any] = []

    if cast_k > 0:
        remaining = max_new_tokens
        while remaining > 0:
            logits, active_cache = _forward(
                model, next_input, active_cache, use_cache, external_msa_registry
            )
            block_logits = _extract_block_logits(logits)

            block_tokens: list[Any] = []
            stop = False
            for k in range(min(cast_k, remaining)):
                if block_logits.dim() == 3:
                    tok = sample_next_token(
                        block_logits[:, k], temperature, top_k, top_p
                    )
                else:
                    tok = sample_next_token(
                        block_logits, temperature, top_k, top_p
                    )
                if tok.dim() == 0:
                    tok = tok.unsqueeze(0)
                block_tokens.append(tok)
                generated_tokens.append(tok)
                yield tok
                if on_token is not None:
                    on_token(tok)
                remaining -= 1
                if eos_token_id is not None and torch.all(tok == eos_token_id):
                    stop = True
                    break
                if _check_stop_sequences(generated_tokens, stop_sequences):
                    stop = True
                    break

            if stop or remaining <= 0:
                break

            next_input = torch.stack(block_tokens, dim=-1)
    else:
        for _ in range(max_new_tokens):
            logits, active_cache = _forward(
                model, next_input, active_cache, use_cache, external_msa_registry
            )
            next_token = sample_next_token(logits, temperature, top_k, top_p)
            if next_token.dim() == 0:
                next_token = next_token.unsqueeze(0)
            generated_tokens.append(next_token)
            yield next_token
            if on_token is not None:
                on_token(next_token)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            if early_exit_confidence > 0.0:
                _ee_confidences.append(confidence_score(logits))
                if (
                    len(_ee_confidences) >= 16
                    and sum(_ee_confidences[-16:]) / 16 >= early_exit_confidence
                ):
                    break
            if _check_stop_sequences(generated_tokens, stop_sequences):
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
    external_msa_registry: Any | None = None,
    stop_sequences: tuple[tuple[int, ...], ...] | list[list[int]] | None = None,
) -> Any:
    """Generate with multiple noisy rollouts, select best by confidence (PTRM idea).

    CAST-aware: when the model has CAST enabled, scoring uses the first
    block position (k=0) logits for confidence comparison.
    """
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
            external_msa_registry=external_msa_registry,
            stop_sequences=stop_sequences,
        )

    best_generated = None
    best_score = float("-inf")
    result = None
    compiled_model = _maybe_compile(model, compile_model)
    for _ in range(rollouts):
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
                external_msa_registry=external_msa_registry,
                stop_sequences=stop_sequences,
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
                    last_logits: Any = (
                        score_output[0]
                        if isinstance(score_output, tuple)
                        else score_output.get("logits")
                        if isinstance(score_output, dict)
                        else score_output
                    )
                    if last_logits is None:
                        score = 0.0
                    else:
                        if last_logits.dim() == 4:
                            last_logits = last_logits[:, -1, 0]
                        score = confidence_score(last_logits[..., -1, :])
                if score > best_score:
                    best_score = score
                    best_generated = result
    return best_generated if best_generated is not None else result
