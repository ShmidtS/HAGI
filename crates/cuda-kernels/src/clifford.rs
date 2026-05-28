//! Clifford algebra GPU kernel wrappers.

use clifford_core::ProductTable;
use tensor_runtime::Tensor;

use crate::dispatch::{AutoDispatch, KernelDispatch};
use crate::CudaKernelError;

/// Clifford algebra kernels dispatched through [`KernelDispatch`].
pub struct CliffordKernels {
    dispatch: AutoDispatch,
}

impl CliffordKernels {
    pub fn new() -> Self {
        Self {
            dispatch: AutoDispatch::new(),
        }
    }

    pub fn with_dispatch(dispatch: AutoDispatch) -> Self {
        Self { dispatch }
    }

    /// Geometric product of two multivector tensors.
    ///
    /// Both inputs must have the same shape: either `[blade_count]` for a
    /// single multivector or `[batch, blade_count]` for a batch.
    pub fn geometric_product_kernel(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        table: &ProductTable,
    ) -> Result<Tensor<f32>, CudaKernelError> {
        self.dispatch.geometric_product(a, b, table)
    }

    /// Rotor sandwich: `R * mv * reverse(R)`.
    ///
    /// `rotor` may be rank-1 (shared across batch) or rank-2 (per-element).
    /// `mv` must be rank-1 or rank-2 with matching batch dimension.
    pub fn rotor_sandwich_kernel(
        &self,
        rotor: &Tensor<f32>,
        mv: &Tensor<f32>,
        table: &ProductTable,
    ) -> Result<Tensor<f32>, CudaKernelError> {
        self.dispatch.rotor_sandwich(rotor, mv, table)
    }
}

impl Default for CliffordKernels {
    fn default() -> Self {
        Self::new()
    }
}
