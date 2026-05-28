use core_types::dtype::DType;
use core_types::shape::Shape;
use core_types::tensor_layout::{
    canonical_blocked_layout, Strides, TensorLayout, TensorLayoutError,
};
use std::time::Instant;

#[test]
fn strides_row_major_1d() {
    let shape = Shape::new(vec![10]);
    let strides = Strides::row_major(&shape);
    assert_eq!(strides.values.as_slice(), &[1]);
}

#[test]
fn strides_row_major_2d() {
    let shape = Shape::new(vec![3, 4]);
    let strides = Strides::row_major(&shape);
    assert_eq!(strides.values.as_slice(), &[4, 1]);
}

#[test]
fn strides_row_major_3d() {
    let shape = Shape::new(vec![2, 3, 4]);
    let strides = Strides::row_major(&shape);
    assert_eq!(strides.values.as_slice(), &[12, 4, 1]);
}

#[test]
fn strides_row_major_0d() {
    let shape = Shape::new(vec![]);
    let strides = Strides::row_major(&shape);
    assert_eq!(strides.values.as_slice(), &[]);
}

#[test]
fn tensor_layout_contiguous() {
    let shape = Shape::new(vec![2, 3, 4]);
    let layout = TensorLayout::contiguous(shape.clone(), 4);
    assert_eq!(layout.shape, shape);
    assert_eq!(layout.strides.values.as_slice(), &[12, 4, 1]);
    assert_eq!(layout.offset, 0);
    assert_eq!(layout.alignment, 4);
    assert_eq!(layout.alignment_bytes, 4);
    assert_eq!(layout.block_elems_fast_dim, 1);
    assert!(layout.is_contiguous());
    assert!(layout.is_aligned());
    assert_eq!(layout.rank(), 3);
}

#[test]
fn tensor_layout_alignment_zero_elem_size() {
    let shape = Shape::new(vec![2, 3]);
    let layout = TensorLayout::contiguous(shape, 0);
    assert_eq!(layout.alignment, 1);
    assert_eq!(layout.alignment_bytes, 1);
    assert!(layout.is_aligned());
}

#[test]
fn tensor_layout_not_contiguous_with_offset() {
    let shape = Shape::new(vec![2, 3]);
    let mut layout = TensorLayout::contiguous(shape, 4);
    layout.offset = 1;
    assert!(!layout.is_contiguous());
}

#[test]
fn contiguous_layout_uses_smallvec_strides() {
    let layout = TensorLayout::contiguous(Shape::new(vec![2, 3, 4]), 4);
    assert!(!layout.strides.values.spilled());
    assert_eq!(layout.strides.values.as_slice(), &[12, 4, 1]);
}

#[test]
fn contiguous_layout_alignment_bytes_matches_dtype_size() {
    let layout = TensorLayout::contiguous(Shape::new(vec![2, 3]), DType::F64.size_of());
    assert_eq!(layout.alignment_bytes, 8);
    assert_eq!(layout.alignment, 8);
}

#[test]
fn index_to_offset_row_major_rank_3() {
    let layout = TensorLayout::contiguous(Shape::new(vec![2, 3, 4]), 4);
    assert_eq!(layout.index_to_offset(&[1, 2, 3]).unwrap(), 23);
}

#[test]
fn offset_to_index_row_major_rank_3() {
    let layout = TensorLayout::contiguous(Shape::new(vec![2, 3, 4]), 4);
    assert_eq!(layout.offset_to_index(23).unwrap().as_slice(), &[1, 2, 3]);
}

#[test]
fn offset_roundtrip_nonzero_offset() {
    let mut layout = TensorLayout::contiguous(Shape::new(vec![2, 3, 4]), 4);
    layout.offset = 7;
    let index = [1, 2, 3];
    let offset = layout.index_to_offset(&index).unwrap();
    assert_eq!(layout.offset_to_index(offset).unwrap().as_slice(), index);
}

#[test]
fn index_offset_roundtrip_ranks_1_to_6() {
    for rank in 1..=6 {
        let dims = vec![2; rank];
        let layout = TensorLayout::contiguous(Shape::new(dims), 4);
        for offset in 0..layout.shape.numel() {
            let index = layout.offset_to_index(offset).unwrap();
            assert_eq!(layout.index_to_offset(&index).unwrap(), offset);
        }
    }
}

#[test]
fn index_to_offset_rejects_rank_mismatch() {
    let layout = TensorLayout::contiguous(Shape::new(vec![2, 3]), 4);
    let err = layout.index_to_offset(&[1]).unwrap_err();
    assert_eq!(
        err,
        TensorLayoutError::RankMismatch {
            expected: 2,
            actual: 1
        }
    );
}

#[test]
fn index_to_offset_rejects_out_of_bounds() {
    let layout = TensorLayout::contiguous(Shape::new(vec![2, 3]), 4);
    let err = layout.index_to_offset(&[1, 3]).unwrap_err();
    assert_eq!(
        err,
        TensorLayoutError::IndexOutOfBounds {
            dim: 1,
            index: 3,
            dim_size: 3
        }
    );
}

#[test]
fn offset_to_index_rejects_offset_equal_numel() {
    let layout = TensorLayout::contiguous(Shape::new(vec![2, 3]), 4);
    let err = layout.offset_to_index(6).unwrap_err();
    assert_eq!(
        err,
        TensorLayoutError::OffsetOutOfBounds {
            offset: 6,
            numel: 6
        }
    );
}

#[test]
fn canonical_blocked_layout_sets_fast_dim_block() {
    let layout = canonical_blocked_layout(&Shape::new(vec![2, 16]), DType::F32, 64);
    assert_eq!(layout.alignment_bytes, 64);
    assert_eq!(layout.block_elems_fast_dim, 16);
}

#[test]
fn canonical_blocked_layout_keeps_row_major_offsets() {
    let layout = canonical_blocked_layout(&Shape::new(vec![2, 3, 4]), DType::F32, 64);
    assert_eq!(layout.index_to_offset(&[1, 2, 3]).unwrap(), 23);
    assert!(layout.is_contiguous());
}

#[test]
fn layout_alignment_property_for_dtype_sizes() {
    for dtype in [
        DType::F32,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U32,
        DType::Bool,
    ] {
        let layout = TensorLayout::contiguous(Shape::new(vec![4]), dtype.size_of());
        assert_eq!(layout.alignment_bytes, dtype.size_of().max(1));
        assert!(layout.is_aligned());
    }
}

#[test]
fn contiguous_sum_matches_slice_sum() {
    let shape = Shape::new(vec![32, 16]);
    let layout = TensorLayout::contiguous(shape.clone(), 4);
    let data: Vec<usize> = (0..shape.numel()).collect();
    let indexed_sum: usize = (0..shape.dims[0])
        .flat_map(|i| (0..shape.dims[1]).map(move |j| [i, j]))
        .map(|index| data[layout.index_to_offset(&index).unwrap()])
        .sum();
    let slice_sum: usize = data.iter().sum();
    assert_eq!(indexed_sum, slice_sum);
}

#[test]
fn contiguous_sum_indexed_path_not_more_than_1_25x_smoke() {
    let shape = Shape::new(vec![128, 128]);
    let layout = TensorLayout::contiguous(shape.clone(), 4);
    let data: Vec<usize> = (0..shape.numel()).collect();

    let slice_started = Instant::now();
    let slice_sum: usize = data.iter().sum();
    let slice_elapsed = slice_started.elapsed();

    let indexed_started = Instant::now();
    let mut indexed_sum = 0usize;
    for i in 0..shape.dims[0] {
        for j in 0..shape.dims[1] {
            indexed_sum += data[layout.index_to_offset(&[i, j]).unwrap()];
        }
    }
    let indexed_elapsed = indexed_started.elapsed();

    assert_eq!(indexed_sum, slice_sum);
    if !cfg!(debug_assertions) {
        assert!(indexed_elapsed.as_nanos() <= slice_elapsed.as_nanos() * 5 / 4);
    }
}
