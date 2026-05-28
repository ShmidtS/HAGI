use smallvec::SmallVec;
use thiserror::Error;

use crate::dtype::DType;
use crate::shape::Shape;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Strides {
    pub values: SmallVec<[usize; 8]>,
}

impl Strides {
    pub fn row_major(shape: &Shape) -> Self {
        let rank = shape.rank();
        if rank == 0 {
            return Self {
                values: SmallVec::new(),
            };
        }

        let mut values = SmallVec::<[usize; 8]>::from_elem(0, rank);
        values[rank - 1] = 1;
        for i in (0..rank - 1).rev() {
            values[i] = values[i + 1] * shape.dims[i + 1];
        }
        Self { values }
    }
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum TensorLayoutError {
    #[error("rank mismatch: expected {expected}, actual {actual}")]
    RankMismatch { expected: usize, actual: usize },
    #[error("index out of bounds at dim {dim}: index {index}, dim size {dim_size}")]
    IndexOutOfBounds {
        dim: usize,
        index: usize,
        dim_size: usize,
    },
    #[error("offset out of bounds: offset {offset}, numel {numel}")]
    OffsetOutOfBounds { offset: usize, numel: usize },
    #[error("layout rank mismatch: shape rank {shape_rank}, strides rank {strides_rank}")]
    LayoutRankMismatch {
        shape_rank: usize,
        strides_rank: usize,
    },
    #[error("zero dimension at dim {dim}")]
    ZeroDimension { dim: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorLayout {
    pub shape: Shape,
    pub strides: Strides,
    pub offset: usize,
    pub alignment: usize,
    pub alignment_bytes: usize,
    pub block_elems_fast_dim: usize,
}

impl TensorLayout {
    pub fn contiguous(shape: Shape, elem_size: usize) -> Self {
        let strides = Strides::row_major(&shape);
        let alignment_bytes = elem_size.max(1);
        Self {
            shape,
            strides,
            offset: 0,
            alignment: alignment_bytes,
            alignment_bytes,
            block_elems_fast_dim: 1,
        }
    }

    pub fn is_contiguous(&self) -> bool {
        let expected = Strides::row_major(&self.shape);
        self.strides == expected && self.offset == 0
    }

    pub fn is_aligned(&self) -> bool {
        self.alignment_bytes > 0 && self.offset.is_multiple_of(self.alignment_bytes)
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn index_to_offset(&self, index: &[usize]) -> Result<usize, TensorLayoutError> {
        validate_layout_rank(self)?;
        if index.len() != self.shape.rank() {
            return Err(TensorLayoutError::RankMismatch {
                expected: self.shape.rank(),
                actual: index.len(),
            });
        }

        let mut offset = self.offset;
        for (dim, (&idx, &dim_size)) in index.iter().zip(self.shape.dims.iter()).enumerate() {
            if dim_size == 0 {
                return Err(TensorLayoutError::ZeroDimension { dim });
            }
            if idx >= dim_size {
                return Err(TensorLayoutError::IndexOutOfBounds {
                    dim,
                    index: idx,
                    dim_size,
                });
            }
            offset += idx * self.strides.values[dim];
        }
        Ok(offset)
    }

    pub fn offset_to_index(
        &self,
        offset: usize,
    ) -> Result<SmallVec<[usize; 8]>, TensorLayoutError> {
        validate_layout_rank(self)?;
        let numel = self.shape.numel();
        if offset < self.offset {
            return Err(TensorLayoutError::OffsetOutOfBounds { offset, numel });
        }
        let logical_offset = offset - self.offset;
        if logical_offset >= numel {
            return Err(TensorLayoutError::OffsetOutOfBounds { offset, numel });
        }

        for (dim, &dim_size) in self.shape.dims.iter().enumerate() {
            if dim_size == 0 {
                return Err(TensorLayoutError::ZeroDimension { dim });
            }
        }

        let expected = Strides::row_major(&self.shape);
        if self.strides != expected {
            return Err(TensorLayoutError::LayoutRankMismatch {
                shape_rank: self.shape.rank(),
                strides_rank: self.strides.values.len(),
            });
        }

        let mut remaining = logical_offset;
        let mut index = SmallVec::<[usize; 8]>::with_capacity(self.shape.rank());
        for &stride in self.strides.values.iter() {
            if let Some(quotient) = remaining.checked_div(stride) {
                index.push(quotient);
                remaining %= stride;
            } else {
                index.push(0);
            }
        }
        Ok(index)
    }
}

pub fn canonical_blocked_layout(
    shape: &Shape,
    dtype: DType,
    l1_cache_line_bytes: usize,
) -> TensorLayout {
    let dtype_size = dtype.size_of();
    let alignment_bytes = l1_cache_line_bytes.max(dtype_size).max(1);
    TensorLayout {
        shape: shape.clone(),
        strides: Strides::row_major(shape),
        offset: 0,
        alignment: alignment_bytes,
        alignment_bytes,
        block_elems_fast_dim: (l1_cache_line_bytes / dtype_size.max(1)).max(1),
    }
}

fn validate_layout_rank(layout: &TensorLayout) -> Result<(), TensorLayoutError> {
    let shape_rank = layout.shape.rank();
    let strides_rank = layout.strides.values.len();
    if shape_rank != strides_rank {
        return Err(TensorLayoutError::LayoutRankMismatch {
            shape_rank,
            strides_rank,
        });
    }
    Ok(())
}
