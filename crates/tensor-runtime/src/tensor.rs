use std::sync::Arc;

use core_types::dtype::{DType, DTypeTag};
use core_types::shape::Shape;
use core_types::tensor_layout::{TensorLayout, TensorLayoutError};
use core_types::tensor_spec::TensorSpec;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Backend {
    #[default]
    Cpu,
    Cuda,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Cuda => "cuda",
        }
    }
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    #[error("data length mismatch: expected {expected}, actual {actual}")]
    DataLengthMismatch { expected: usize, actual: usize },
    #[error("shape numel overflow: dims {dims:?}")]
    ShapeNumelOverflow { dims: Vec<usize> },
    #[error("dtype mismatch: expected {expected:?}, actual {actual:?}")]
    DTypeMismatch { expected: DType, actual: DType },
    #[error("shape mismatch: expected {expected:?}, actual {actual:?}")]
    ShapeMismatch { expected: Shape, actual: Shape },
    #[error("layout rank mismatch: shape rank {shape_rank}, strides rank {strides_rank}")]
    LayoutRankMismatch {
        shape_rank: usize,
        strides_rank: usize,
    },
    #[error("unaligned allocation: ptr {ptr}, alignment bytes {alignment_bytes}")]
    UnalignedAllocation { ptr: usize, alignment_bytes: usize },
    #[error(transparent)]
    Index(#[from] TensorLayoutError),
    #[error("unsupported backend: {backend}")]
    UnsupportedBackend { backend: &'static str },
    #[error("backend error: {message}")]
    Backend { message: String },
}

pub trait TensorElement: DTypeTag {}

impl<T: DTypeTag> TensorElement for T {}

#[derive(Debug, Clone)]
pub struct Tensor<T: TensorElement> {
    spec: TensorSpec,
    data: Arc<Vec<T>>,
    backend: Backend,
}

impl<T: TensorElement> Tensor<T> {
    pub fn zeros(shape: Shape) -> Self {
        let spec = TensorSpec::new(shape, T::DTYPE);
        Self::zeros_with_spec(spec, Backend::Cpu)
            .expect("zeros constructor must build a CPU tensor")
    }

    pub fn zeros_with_spec(spec: TensorSpec, backend: Backend) -> Result<Self, TensorError> {
        let numel = spec
            .shape
            .checked_numel()
            .ok_or_else(|| TensorError::ShapeNumelOverflow {
                dims: spec.shape.dims.clone(),
            })?;
        Self::from_vec_with_spec(spec, vec![T::default(); numel], backend)
    }

    pub fn from_vec(data: Vec<T>, shape: Shape) -> Self {
        Self::try_from_vec(data, shape).unwrap_or_else(|err| match err {
            TensorError::DataLengthMismatch { .. } => {
                panic!("data length must match shape numel")
            }
            TensorError::ShapeNumelOverflow { .. } => panic!("shape numel must not overflow"),
            err => panic!("from_vec constructor must build a CPU tensor: {err}"),
        })
    }

    pub fn try_from_vec(data: Vec<T>, shape: Shape) -> Result<Self, TensorError> {
        let spec = TensorSpec::new(shape, T::DTYPE);
        Self::from_vec_with_spec(spec, data, Backend::Cpu)
    }

    pub fn from_vec_with_spec(
        spec: TensorSpec,
        data: Vec<T>,
        backend: Backend,
    ) -> Result<Self, TensorError> {
        if spec.dtype != T::DTYPE {
            return Err(TensorError::DTypeMismatch {
                expected: spec.dtype,
                actual: T::DTYPE,
            });
        }

        let expected =
            spec.shape
                .checked_numel()
                .ok_or_else(|| TensorError::ShapeNumelOverflow {
                    dims: spec.shape.dims.clone(),
                })?;
        if data.len() != expected {
            return Err(TensorError::DataLengthMismatch {
                expected,
                actual: data.len(),
            });
        }

        if spec.layout.shape != spec.shape {
            return Err(TensorError::ShapeMismatch {
                expected: spec.shape.clone(),
                actual: spec.layout.shape.clone(),
            });
        }

        let shape_rank = spec.shape.rank();
        let strides_rank = spec.layout.strides.values.len();
        if shape_rank != strides_rank {
            return Err(TensorError::LayoutRankMismatch {
                shape_rank,
                strides_rank,
            });
        }

        if backend == Backend::Cpu {
            let alignment_bytes = spec.layout.alignment_bytes.max(1);
            let ptr = data.as_ptr() as usize;
            if !ptr.is_multiple_of(alignment_bytes) {
                return Err(TensorError::UnalignedAllocation {
                    ptr,
                    alignment_bytes,
                });
            }
        } else {
            return Err(TensorError::UnsupportedBackend {
                backend: backend.name(),
            });
        }

        Ok(Self {
            spec,
            data: Arc::new(data),
            backend,
        })
    }

    pub fn spec(&self) -> &TensorSpec {
        &self.spec
    }

    pub fn shape(&self) -> &Shape {
        &self.spec.shape
    }

    pub fn dtype(&self) -> DType {
        self.spec.dtype
    }

    pub fn layout(&self) -> &TensorLayout {
        &self.spec.layout
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn numel(&self) -> usize {
        self.spec.numel()
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    pub fn as_view(&self) -> TensorView<'_, T> {
        TensorView {
            spec: &self.spec,
            data: &self.data,
            backend: self.backend,
        }
    }

    pub fn as_view_mut(&mut self) -> TensorViewMut<'_, T> {
        let data = Arc::make_mut(&mut self.data);
        TensorViewMut {
            spec: &self.spec,
            data,
            backend: self.backend,
        }
    }

    pub fn as_mut(&mut self) -> TensorMut<'_, T> {
        let data = Arc::make_mut(&mut self.data);
        TensorMut {
            spec: &mut self.spec,
            data,
            backend: self.backend,
        }
    }
}

impl<T: TensorElement + PartialEq> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.spec == other.spec && *self.data == *other.data && self.backend == other.backend
    }
}

#[derive(Debug)]
pub struct TensorView<'a, T: TensorElement> {
    pub spec: &'a TensorSpec,
    pub data: &'a [T],
    pub backend: Backend,
}

impl<'a, T: TensorElement> TensorView<'a, T> {
    pub fn shape(&self) -> &Shape {
        &self.spec.shape
    }

    pub fn dtype(&self) -> DType {
        self.spec.dtype
    }

    pub fn data(&self) -> &[T] {
        self.data
    }

    pub fn numel(&self) -> usize {
        self.spec.numel()
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }
}

#[derive(Debug)]
pub struct TensorViewMut<'a, T: TensorElement> {
    pub spec: &'a TensorSpec,
    pub data: &'a mut [T],
    pub backend: Backend,
}

impl<'a, T: TensorElement> TensorViewMut<'a, T> {
    pub fn shape(&self) -> &Shape {
        &self.spec.shape
    }

    pub fn dtype(&self) -> DType {
        self.spec.dtype
    }

    pub fn data(&self) -> &[T] {
        self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        self.data
    }

    pub fn numel(&self) -> usize {
        self.spec.numel()
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }
}

#[derive(Debug)]
pub struct TensorMut<'a, T: TensorElement> {
    pub spec: &'a mut TensorSpec,
    pub data: &'a mut Vec<T>,
    pub backend: Backend,
}

impl<'a, T: TensorElement> TensorMut<'a, T> {
    pub fn shape(&self) -> &Shape {
        &self.spec.shape
    }

    pub fn dtype(&self) -> DType {
        self.spec.dtype
    }

    pub fn data(&self) -> &[T] {
        self.data
    }

    pub fn data_mut(&mut self) -> &mut Vec<T> {
        self.data
    }

    pub fn numel(&self) -> usize {
        self.spec.numel()
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }
}

pub trait BackendOp<T: TensorElement> {
    fn dispatch(
        &self,
        inputs: &[TensorView<'_, T>],
        output: TensorViewMut<'_, T>,
    ) -> Result<(), TensorError>;
}
