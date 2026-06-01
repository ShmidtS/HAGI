"""Shared utility helpers across the HAGI codebase."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - numpy fallback for non-torch environments
    torch = None  # type: ignore[assignment]

import yaml


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _reordering_sign(a: int, b: int) -> int:
    """Sign from reordering the product of two basis blades into canonical order.

    Counts transpositions needed to sort the concatenated basis vectors.
    Metric is Euclidean (+1) so shared indices contribute no extra sign.
    """
    a >>= 1
    swaps = 0
    while a:
        swaps += bin(a & b).count("1")
        a >>= 1
    return -1 if (swaps & 1) else 1


def _as_long_tensor(values: Any) -> Any:
    if torch is None:
        return np.asarray(values, dtype=np.int64)
    return torch.as_tensor(values, dtype=torch.long)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data
