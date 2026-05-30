from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without torch installed
    torch = None  # type: ignore[assignment]

BLADE_COUNT = 8


def _numpy_geometric_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    for a in range(BLADE_COUNT):
        for b in range(BLADE_COUNT):
            c = a ^ b
            sign = _reordering_sign(a, b)
            out[..., c] = out[..., c] + sign * x[..., a] * y[..., b]
    return out


def _reordering_sign(a: int, b: int) -> int:
    a >>= 1
    swaps = 0
    while a:
        swaps += int((a & b).bit_count())
        a >>= 1
    return -1 if swaps & 1 else 1


class HDIMReasoner:
    """Encode NARS terms into Cl(3,0,0) multivectors and compose them."""

    def __init__(self, *, dtype: Any | None = None, device: Any | None = None) -> None:
        self.dtype = dtype
        self.device = device

    def encode(self, term: Any) -> Any:
        values = self._hash_vector(term)
        if torch is not None:
            dtype = self._torch_dtype()
            return torch.tensor(values, dtype=dtype, device=self.device)
        dtype = self.dtype if self.dtype is not None else np.float32
        return values.astype(dtype)

    def reason(self, t1: Any, t2: Any) -> Any:
        left = self.encode(t1)
        right = self.encode(t2)
        if torch is not None and isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            from hagi.model.clifford import geometric_product

            return geometric_product(left, right)
        return _numpy_geometric_product(np.asarray(left), np.asarray(right))

    def _torch_dtype(self) -> Any:
        if self.dtype is None:
            return torch.float32
        if isinstance(self.dtype, torch.dtype):
            return self.dtype
        if self.dtype in {np.float64, np.dtype("float64"), float}:
            return torch.float64
        return torch.float32

    def _hash_vector(self, term: Any) -> np.ndarray:
        digest = hashlib.sha256(repr(term).encode("utf-8")).digest()
        raw = np.frombuffer(digest[:BLADE_COUNT], dtype=np.uint8).astype(np.float32)
        return (raw / 127.5) - 1.0
