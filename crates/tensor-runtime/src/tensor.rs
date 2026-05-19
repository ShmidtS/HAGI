use core_types::dtype::DType;
use core_types::shape::Shape;

/// Minimal CPU tensor handle for forward-only reference path.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T: Copy> {
    pub data: Vec<T>,
    pub shape: Shape,
    pub dtype: DType,
}

impl<T: Copy> Tensor<T> {
    pub fn zeros(shape: Shape, dtype: DType) -> Self
    where
        T: Default,
    {
        let numel = shape.numel();
        Self {
            data: vec![T::default(); numel],
            shape,
            dtype,
        }
    }

    pub fn from_vec(data: Vec<T>, shape: Shape, dtype: DType) -> Self {
        assert_eq!(data.len(), shape.numel(), "data length must match shape numel");
        Self { data, shape, dtype }
    }
}
