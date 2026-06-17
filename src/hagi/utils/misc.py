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


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data
