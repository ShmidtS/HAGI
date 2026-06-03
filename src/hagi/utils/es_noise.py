"""Deterministic Evolution Strategies noise.

Python port of Rust `es_noise.rs` from PTRM.  Uses SplitMix64 so noise is
fully reproducible from a seed and parameter index.  No GPU needed — the
kernel is just a hash chain.
"""

from __future__ import annotations

import numpy as np


_MIX_A: int = 0x9E37_79B9_7F4A_7C15
_MIX_B: int = 0xBF58_476D_1CE4_E5B9
_MIX_C: int = 0x94D0_49BB_1331_11EB


def _splitmix64(x: int) -> int:
    x = (x + _MIX_A) & 0xFFFF_FFFF_FFFF_FFFF
    x = (x ^ (x >> 30)) * _MIX_B & 0xFFFF_FFFF_FFFF_FFFF
    x = (x ^ (x >> 27)) * _MIX_C & 0xFFFF_FFFF_FFFF_FFFF
    return (x ^ (x >> 31)) & 0xFFFF_FFFF_FFFF_FFFF


def _uniform_24(seed: int, param_index: int, lane: int) -> float:
    x = (
        seed
        ^ (param_index * 0xD1B5_4A32_D192_ED03)
        ^ (lane * 0xABC9_83A5_8B8C_2D4D)
    ) & 0xFFFF_FFFF_FFFF_FFFF
    return (_splitmix64(x) >> 40) / 16_777_216.0


def standard_noise(seed: int, param_index: int) -> float:
    """Standard-normal noise from a 64-bit seed and parameter index.

    Approximates N(0, 1) by summing 12 uniform[0, 1] draws and subtracting 6.
    """
    return sum(_uniform_24(seed, param_index, lane) for lane in range(12)) - 6.0


def noise_vector(seed: int, length: int) -> np.ndarray:
    """Return a NumPy vector of standard-normal noise."""
    return np.array([standard_noise(seed, i) for i in range(length)], dtype=np.float32)


def noise_tensor(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    """Return a NumPy tensor of standard-normal noise."""
    total = int(np.prod(shape))
    return np.array([standard_noise(seed, i) for i in range(total)], dtype=np.float32).reshape(shape)
