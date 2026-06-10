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


# [8, 8, 8] tensor: STRUCT[a, b, c] = sign[a, b] if a^b == c else 0.
# Then the geometric product is one contraction: out[..., c] = sum_{a,b} x[..., a] * C[a, b, c] * y[..., b].
_STRUCT = torch.zeros(BLADE_COUNT, BLADE_COUNT, BLADE_COUNT, dtype=torch.float32)
for _a in range(BLADE_COUNT):
    for _b in range(BLADE_COUNT):
        _STRUCT[_a, _b, int(_OUT_INDEX[_a, _b])] = _SIGN[_a, _b]

# Triton kernel expects [c, a, b] layout (cab einsum).
_STRUCT_TRITON = _STRUCT.permute(2, 0, 1).contiguous()


# Cache per (device, dtype) to avoid repeated .to() sync
_STRUCT_CACHE: dict[tuple, torch.Tensor] = {}
_GRADE_MASK_CACHE: dict[tuple, torch.Tensor] = {}
_REVERSE_SIGNS_CACHE: dict[tuple, torch.Tensor] = {}
_BV_MASK_CACHE: dict[tuple, torch.Tensor] = {}
_OTHER_MASK_CACHE: dict[tuple, torch.Tensor] = {}


def _get_struct(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    if key not in _STRUCT_CACHE:
        _STRUCT_CACHE[key] = _STRUCT_TRITON.to(device=device, dtype=dtype)
    return _STRUCT_CACHE[key]


def _get_grade_mask(device: torch.device, dtype: torch.dtype, grade: int) -> torch.Tensor:
    key = (device, dtype, grade)
    if key not in _GRADE_MASK_CACHE:
        _GRADE_MASK_CACHE[key] = torch.tensor(
            [1.0 if GRADE[i] == grade else 0.0 for i in range(BLADE_COUNT)],
            dtype=dtype,
            device=device,
        )
    return _GRADE_MASK_CACHE[key]


def _get_reverse_signs(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    if key not in _REVERSE_SIGNS_CACHE:
        _REVERSE_SIGNS_CACHE[key] = torch.tensor(
            [(-1.0) ** (GRADE[i] * (GRADE[i] - 1) // 2) for i in range(BLADE_COUNT)],
            dtype=dtype,
            device=device,
        )
    return _REVERSE_SIGNS_CACHE[key]


def _get_bv_mask(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    if key not in _BV_MASK_CACHE:
        _BV_MASK_CACHE[key] = torch.tensor(
            [1.0 if GRADE[i] == 2 else 0.0 for i in range(BLADE_COUNT)],
            dtype=dtype,
            device=device,
        )
    return _BV_MASK_CACHE[key]


def _get_other_mask(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    if key not in _OTHER_MASK_CACHE:
        _OTHER_MASK_CACHE[key] = torch.tensor(
            [1.0 if GRADE[i] not in (0, 2) else 0.0 for i in range(BLADE_COUNT)],
            dtype=dtype,
            device=device,
        )
    return _OTHER_MASK_CACHE[key]


def geometric_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Geometric product of two batched multivectors.

    Args:
        x: [..., 8] multivector coefficients.
        y: [..., 8] multivector coefficients.

    Returns:
        [..., 8] product coefficients.

    Triton kernel when CUDA is available; fused einsum fallback otherwise.
    """
    assert x.shape[-1] == BLADE_COUNT, f"expected last dim {BLADE_COUNT}, got {x.shape[-1]}"
    assert y.shape[-1] == BLADE_COUNT, f"expected last dim {BLADE_COUNT}, got {y.shape[-1]}"

    from .triton_kernels import geometric_product_triton
    c = _get_struct(x.device, x.dtype)
    return geometric_product_triton(x, y, c)


def grade_projection(mv: torch.Tensor, grade: int) -> torch.Tensor:
    """Zero out all blades not of the given grade. Returns [..., 8]."""
    mask = _get_grade_mask(mv.device, mv.dtype, grade)
    return mv * mask


def reverse(mv: torch.Tensor) -> torch.Tensor:
    """Clifford reverse: sign (-1)^(k(k-1)/2) per grade k. Returns [..., 8]."""
    signs = _get_reverse_signs(mv.device, mv.dtype)
    return mv * signs


def wedge_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Exterior (antisymmetric) product: (xy - yx) / 2 restricted to grade 2.

    For pure vector inputs returns the bivector ab = (xy - yx) / 2 (e.g.
    e1 ∧ e2 = e12). The result is always grade-2 regardless of input
    grades (the grade-rising part for scalars or higher-grade inputs is
    degenerate in Cl(3,0,0), so we project to bivector consistently).
    """
    assert x.shape[-1] == BLADE_COUNT
    assert y.shape[-1] == BLADE_COUNT
    diff = geometric_product(x, y) - geometric_product(y, x)
    return 0.5 * grade_projection(diff, 2)


def inner_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Inner (symmetric) product: scalar part of (xy + yx) / 2.

    Returns a scalar (the grade-0 component). For pure vector inputs this
    is the Euclidean inner product. Higher-grade inputs return the scalar
    part of the symmetric product.
    """
    assert x.shape[-1] == BLADE_COUNT
    assert y.shape[-1] == BLADE_COUNT
    sym = geometric_product(x, y) + geometric_product(y, x)
    return 0.5 * grade_projection(sym, 0)[..., 0]


def commutator(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Lie commutator [x, y] = (xy - yx) / 2. Bivector-valued for vectors."""
    return 0.5 * (geometric_product(x, y) - geometric_product(y, x))


def anticommutator(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Jordan anticommutator {x, y} = (xy + yx) / 2."""
    return 0.5 * (geometric_product(x, y) + geometric_product(y, x))


def bivector_exp(mv: torch.Tensor) -> torch.Tensor:
    """Exponentiate a bivector: R = exp(-B/2) where B is the grade-2 part of mv.

    Closed form: in Cl(3,0,0) a bivector B satisfies B^2 = -|B|^2 (negative
    scalar), so exp(-B/2) = cos(theta/2) - (B / theta) * sin(theta/2)
    with theta = |B|. Grade-0 and grade-2 components of the result
    contribute; the input's other grades pass through unchanged.

    For |B| -> 0, falls back to first-order Taylor: 1 - B/2.
    """
    assert mv.shape[-1] == BLADE_COUNT
    bv_mask = _get_bv_mask(mv.device, mv.dtype)
    bivector = mv * bv_mask
    b_sq = -geometric_product(bivector, bivector)[..., :1]
    b_sq = b_sq.clamp_min(0.0)
    theta = torch.sqrt(b_sq.squeeze(-1))
    half = 0.5 * theta
    cos_half = torch.cos(half)
    sin_half = torch.sin(half)
    safe_theta = torch.sqrt(b_sq.squeeze(-1) + 1e-6)
    coeff = -(sin_half / safe_theta)
    rotor_bv = coeff.unsqueeze(-1) * bivector
    out = torch.zeros_like(mv)
    out[..., :1] = cos_half.unsqueeze(-1)
    out = out + rotor_bv
    other_mask = _get_other_mask(mv.device, mv.dtype)
    out = out + mv * other_mask
    return out
