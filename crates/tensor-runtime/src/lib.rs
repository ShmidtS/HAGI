pub mod backend;
pub mod tensor;

pub use backend::{BackendError, BackendRequest, CpuBackend, TensorBackend};
pub use tensor::{
    Backend, BackendOp, Tensor, TensorElement, TensorError, TensorMut, TensorView, TensorViewMut,
};

pub use core_types::{Strides, TensorLayout, TensorSpec};

pub fn from_vec<T: TensorElement>(
    spec: TensorSpec,
    data: Vec<T>,
    backend: Backend,
) -> Result<Tensor<T>, TensorError> {
    Tensor::from_vec_with_spec(spec, data, backend)
}
