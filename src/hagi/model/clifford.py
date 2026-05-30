"""Clifford algebra Cl(3,0,0) geometric product.

Cl(3,0,0): three orthonormal basis vectors e1, e2, e3, each squaring to +1.
8 basis blades indexed by 3-bit bitmask (bit i set => e_{i+1} present):

    0b000 = 1            (grade 0, scalar)
    0b001 = e1           (grade 1)
    0b010 = e2           (grade 1)
    0b100 = e3           (grade 1)
    0b011 = e1 e2        (grade 2, bivector)
    0b101 = e1 e3        (grade 2, bivector)
    0b110 = e2 e3        (grade 2, bivector)
    0b111 = e1 e2 e3     (grade 3, trivector / pseudoscalar)

The geometric product of two basis blades a, b (bitmasks):
    result_blade = a XOR b
    sign         = (-1)^(reordering transpositions)   [metric is all +1]

This module is the foundation of Grade-Decomposed Recurrence. It is pure,
deterministic, and verifiable — the Cayley table is checked against the Lean4
spec (`formalization/HAGI/HDIM.lean`).
"""

from __future__ import annotations

import torch

BLADE_COUNT = 8
DIM = 3

# Grade (popcount) of each blade index.
GRADE = [bin(i).count("1") for i in range(BLADE_COUNT)]  # [0,1,1,2,1,2,2,3]


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


def build_product_table() -> tuple[torch.Tensor, torch.Tensor]:
    """Build the Cl(3,0,0) Cayley table.

    Returns:
        out_index: [8, 8] long tensor, out_index[a, b] = resulting blade index.
        sign:      [8, 8] float tensor, sign[a, b] = +1 or -1.
    """
    out_index = torch.zeros(BLADE_COUNT, BLADE_COUNT, dtype=torch.long)
    sign = torch.zeros(BLADE_COUNT, BLADE_COUNT, dtype=torch.float32)
    for a in range(BLADE_COUNT):
        for b in range(BLADE_COUNT):
            out_index[a, b] = a ^ b
            sign[a, b] = float(_reordering_sign(a, b))
    return out_index, sign


# Precomputed tables (module-level constants).
_OUT_INDEX, _SIGN = build_product_table()


def geometric_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Geometric product of two batched multivectors.

    Args:
        x: [..., 8] multivector coefficients.
        y: [..., 8] multivector coefficients.

    Returns:
        [..., 8] product coefficients.
    """
    assert x.shape[-1] == BLADE_COUNT, f"expected last dim {BLADE_COUNT}, got {x.shape[-1]}"
    assert y.shape[-1] == BLADE_COUNT, f"expected last dim {BLADE_COUNT}, got {y.shape[-1]}"

    out = torch.zeros_like(x)
    # Accumulate every (a, b) blade pair contribution.
    for a in range(BLADE_COUNT):
        for b in range(BLADE_COUNT):
            c = int(_OUT_INDEX[a, b])
            s = _SIGN[a, b]
            out[..., c] = out[..., c] + s * x[..., a] * y[..., b]
    return out


def grade_projection(mv: torch.Tensor, grade: int) -> torch.Tensor:
    """Zero out all blades not of the given grade. Returns [..., 8]."""
    mask = torch.tensor(
        [1.0 if GRADE[i] == grade else 0.0 for i in range(BLADE_COUNT)],
        dtype=mv.dtype,
        device=mv.device,
    )
    return mv * mask


def reverse(mv: torch.Tensor) -> torch.Tensor:
    """Clifford reverse: sign (-1)^(k(k-1)/2) per grade k. Returns [..., 8]."""
    signs = torch.tensor(
        [(-1.0) ** (GRADE[i] * (GRADE[i] - 1) // 2) for i in range(BLADE_COUNT)],
        dtype=mv.dtype,
        device=mv.device,
    )
    return mv * signs
