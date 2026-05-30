import pytest

from hagi.core import (
    CycleType,
    DType,
    Layout,
    Shape,
    TensorSpec,
    contiguous_layout,
    index_to_offset,
    offset_to_index,
    row_major_strides,
)


def test_tensor_spec_creation_and_shape_validation():
    spec = TensorSpec([2, 3, 4], DType.F32, device="cpu")

    assert spec.shape == (2, 3, 4)
    assert spec.dtype == "f32"
    assert spec.device == "cpu"
    assert spec.rank == 3
    assert spec.numel == 24

    shaped = TensorSpec(Shape([5, 0]), "i64")
    assert shaped.shape == (5, 0)
    assert shaped.numel == 0

    with pytest.raises(ValueError, match="non-negative"):
        TensorSpec([2, -1], DType.F16)


def test_dtype_enum_like_behavior():
    assert DType.F32.value == "f32"
    assert str(DType.F16) == "f16"
    assert DType("bf16") is DType.BF16
    assert DType.I64.itemsize == 8
    assert [dtype.value for dtype in DType] == ["f32", "f16", "bf16", "i32", "i64", "u8"]


def test_layout_stride_calculation():
    shape = Shape([2, 3, 4])
    layout = contiguous_layout(shape, offset=7)

    assert row_major_strides(shape) == (12, 4, 1)
    assert layout == Layout((12, 4, 1), offset=7)
    assert index_to_offset((1, 2, 3), shape, layout) == 30
    assert offset_to_index(30, shape, layout) == (1, 2, 3)

    with pytest.raises(IndexError):
        index_to_offset((2, 0, 0), shape, layout)


def test_cycle_type_values():
    assert CycleType.Forward.value == "forward"
    assert CycleType.Backward.value == "backward"
    assert CycleType.Update.value == "update"
    assert CycleType("forward") is CycleType.Forward
