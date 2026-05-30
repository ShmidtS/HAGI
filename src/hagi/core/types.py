from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import reduce
from operator import mul
from typing import Sequence


class DType(StrEnum):
    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"
    I32 = "i32"
    I64 = "i64"
    U8 = "u8"

    @property
    def itemsize(self) -> int:
        return {
            DType.F32: 4,
            DType.F16: 2,
            DType.BF16: 2,
            DType.I32: 4,
            DType.I64: 8,
            DType.U8: 1,
        }[self]


class CycleType(StrEnum):
    Forward = "forward"
    Backward = "backward"
    Update = "update"


@dataclass(frozen=True, slots=True)
class Shape:
    dims: tuple[int, ...]

    def __init__(self, dims: Sequence[int]):
        object.__setattr__(self, "dims", tuple(int(dim) for dim in dims))
        if any(dim < 0 for dim in self.dims):
            raise ValueError("shape dimensions must be non-negative")

    @property
    def rank(self) -> int:
        return len(self.dims)

    @property
    def numel(self) -> int:
        return reduce(mul, self.dims, 1)


@dataclass(frozen=True, slots=True)
class Layout:
    strides: tuple[int, ...]
    offset: int = 0

    def __init__(self, strides: Sequence[int], offset: int = 0):
        object.__setattr__(self, "strides", tuple(int(stride) for stride in strides))
        object.__setattr__(self, "offset", int(offset))
        if any(stride < 0 for stride in self.strides):
            raise ValueError("layout strides must be non-negative")
        if self.offset < 0:
            raise ValueError("layout offset must be non-negative")


@dataclass(frozen=True, slots=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str
    device: str

    def __init__(self, shape: Sequence[int] | Shape, dtype: str | DType, device: str = "cpu"):
        shape_tuple = shape.dims if isinstance(shape, Shape) else tuple(int(dim) for dim in shape)
        if any(dim < 0 for dim in shape_tuple):
            raise ValueError("tensor shape dimensions must be non-negative")
        dtype_value = dtype.value if isinstance(dtype, DType) else str(dtype)
        object.__setattr__(self, "shape", shape_tuple)
        object.__setattr__(self, "dtype", dtype_value)
        object.__setattr__(self, "device", str(device))

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return reduce(mul, self.shape, 1)


def row_major_strides(shape: Sequence[int] | Shape) -> tuple[int, ...]:
    dims = shape.dims if isinstance(shape, Shape) else tuple(int(dim) for dim in shape)
    if not dims:
        return ()
    strides = [1] * len(dims)
    for index in range(len(dims) - 2, -1, -1):
        strides[index] = strides[index + 1] * dims[index + 1]
    return tuple(strides)


def contiguous_layout(shape: Sequence[int] | Shape, offset: int = 0) -> Layout:
    return Layout(row_major_strides(shape), offset)


def index_to_offset(index: Sequence[int], shape: Sequence[int] | Shape, layout: Layout | None = None) -> int:
    dims = shape.dims if isinstance(shape, Shape) else tuple(int(dim) for dim in shape)
    idx = tuple(int(value) for value in index)
    active_layout = layout if layout is not None else contiguous_layout(dims)
    if len(idx) != len(dims):
        raise ValueError(f"rank mismatch: expected {len(dims)}, actual {len(idx)}")
    if len(active_layout.strides) != len(dims):
        raise ValueError("layout rank must match shape rank")
    offset = active_layout.offset
    for dim, (value, dim_size) in enumerate(zip(idx, dims, strict=True)):
        if dim_size == 0:
            raise ValueError(f"zero dimension at dim {dim}")
        if value < 0 or value >= dim_size:
            raise IndexError(f"index out of bounds at dim {dim}: index {value}, dim size {dim_size}")
        offset += value * active_layout.strides[dim]
    return offset


def offset_to_index(offset: int, shape: Sequence[int] | Shape, layout: Layout | None = None) -> tuple[int, ...]:
    dims = shape.dims if isinstance(shape, Shape) else tuple(int(dim) for dim in shape)
    active_layout = layout if layout is not None else contiguous_layout(dims)
    expected = row_major_strides(dims)
    if active_layout.strides != expected:
        raise ValueError("offset_to_index requires row-major contiguous strides")
    logical_offset = int(offset) - active_layout.offset
    numel = reduce(mul, dims, 1)
    if logical_offset < 0 or logical_offset >= numel:
        raise IndexError(f"offset out of bounds: offset {offset}, numel {numel}")
    if any(dim == 0 for dim in dims):
        raise ValueError("shape with zero dimension has no valid indices")
    index: list[int] = []
    remaining = logical_offset
    for stride in active_layout.strides:
        quotient, remaining = divmod(remaining, stride)
        index.append(quotient)
    return tuple(index)
