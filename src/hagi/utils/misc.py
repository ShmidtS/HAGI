"""Shared utility helpers across the HAGI codebase."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def _clamp01(value: float) -> float:
    """Clamp to [0,1]. Non-finite (NaN/inf) maps to 0.0 — a NaN would poison
    every downstream consumer (e.g. NARS truth revision, which feeds int())."""
    v = float(value)
    if v != v or v in (float("inf"), float("-inf")):  # NaN or inf
        return 0.0
    return max(0.0, min(1.0, v))


def _as_long_tensor(values: Any) -> Any:
    if torch is None:
        return np.asarray(values, dtype=np.int64)
    return torch.as_tensor(values, dtype=torch.long)


def _pad_shift_collate(
    samples: list[Any], ignore_index: int = -100, pad_token: int = 0
) -> Any:
    """Right-pad a variable-length batch and split into (x, y) shift pair.

    Each sample is a 1-D token array whose length may vary (variable-length
    training windows). Input positions x = tokens[:-1] are padded with
    ``pad_token`` (a valid token id the embedding can look up); target
    positions y = tokens[1:] are padded with ``ignore_index`` so CE masks
    padded positions loss-free. Verified leak-free under causal attention: a
    token at position i never attends to positions > i, so right padding does
    not alter the hidden states of real tokens (A/B: real-L loss == padded
    loss, diff 0.0).
    """
    max_len = max(len(s) for s in samples)
    width = max_len - 1
    x = np.full((len(samples), width), pad_token, dtype=np.int64)
    y = np.full((len(samples), width), ignore_index, dtype=np.int64)
    for i, s in enumerate(samples):
        s = np.asarray(s, dtype=np.int64)
        if s.shape[0] < 2:
            continue
        x[i, : s.shape[0] - 1] = s[:-1]
        y[i, : s.shape[0] - 1] = s[1:]
    return _as_long_tensor(x), _as_long_tensor(y)


def _pad_batch(arrays: list[Any], pad_value: int = -100) -> Any:
    """Right-pad a list of variable-length 1-D arrays to a dense 2-D tensor.

    Used by collate fns that receive already-split arrays (no shift here):
    pass ``pad_token`` (e.g. 0) for input-ids arrays and ``ignore_index``
    (-100) for target arrays. Leak-free under causal right-padding.
    """
    max_len = max(len(a) for a in arrays) if arrays else 0
    out = np.full((len(arrays), max_len), pad_value, dtype=np.int64)
    for i, a in enumerate(arrays):
        a = np.asarray(a, dtype=np.int64)
        if a.shape[0]:
            out[i, : a.shape[0]] = a
    return _as_long_tensor(out)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data
