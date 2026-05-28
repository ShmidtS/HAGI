use core_types::dtype::DTypeTag;
use core_types::shape::Shape;
use thiserror::Error;

use crate::tensor::Tensor;

#[derive(Error, Debug)]
pub enum BackendError {
    #[error("backend not available")]
    NotAvailable,
    #[error("shape mismatch")]
    ShapeMismatch,
    #[error("allocation failed")]
    AllocationFailed,
    #[error("unsupported operation")]
    Unsupported,
}

pub enum BackendRequest<'a, T: DTypeTag> {
    Add(&'a Tensor<T>, &'a Tensor<T>),
    Mul(&'a Tensor<T>, &'a Tensor<T>),
    Zeros(Shape),
    Clone(&'a Tensor<T>),
}

pub trait TensorBackend {
    type Elem: DTypeTag;

    fn name(&self) -> &'static str;

    fn zeros(&self, shape: Shape) -> Result<Tensor<Self::Elem>, BackendError>;

    fn execute(
        &self,
        op: BackendRequest<'_, Self::Elem>,
    ) -> Result<Tensor<Self::Elem>, BackendError>;

    fn transfer_from_host(
        &self,
        data: Vec<Self::Elem>,
        shape: Shape,
    ) -> Result<Tensor<Self::Elem>, BackendError>;
}

pub struct CpuBackend;

impl TensorBackend for CpuBackend {
    type Elem = f32;

    fn name(&self) -> &'static str {
        "cpu"
    }

    fn zeros(&self, shape: Shape) -> Result<Tensor<f32>, BackendError> {
        Ok(Tensor::zeros(shape))
    }

    fn execute(&self, op: BackendRequest<'_, f32>) -> Result<Tensor<f32>, BackendError> {
        match op {
            BackendRequest::Add(a, b) => elementwise(a, b, |x, y| x + y),
            BackendRequest::Mul(a, b) => elementwise(a, b, |x, y| x * y),
            BackendRequest::Zeros(shape) => self.zeros(shape),
            BackendRequest::Clone(t) => Ok(t.clone()),
        }
    }

    fn transfer_from_host(
        &self,
        data: Vec<f32>,
        shape: Shape,
    ) -> Result<Tensor<f32>, BackendError> {
        Ok(Tensor::from_vec(data, shape))
    }
}

fn elementwise(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    f: impl Fn(f32, f32) -> f32,
) -> Result<Tensor<f32>, BackendError> {
    if a.shape() != b.shape() {
        return Err(BackendError::ShapeMismatch);
    }
    let data: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(&x, &y)| f(x, y))
        .collect();
    Ok(Tensor::from_vec(data, a.shape().clone()))
}
