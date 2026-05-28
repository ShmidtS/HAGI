//! Sparse attention GPU kernel wrappers.

use tensor_runtime::Tensor;

use crate::dispatch::{AutoDispatch, KernelDispatch};
use crate::CudaKernelError;

/// Attention kernels dispatched through [`KernelDispatch`].
pub struct AttentionKernels {
    dispatch: AutoDispatch,
}

impl AttentionKernels {
    pub fn new() -> Self {
        Self {
            dispatch: AutoDispatch::new(),
        }
    }

    pub fn with_dispatch(dispatch: AutoDispatch) -> Self {
        Self { dispatch }
    }

    /// Sparse attention over selected memory slots.
    ///
    /// - `query`: `[B, T, hidden]`
    /// - `keys`: slice of `[hidden]` tensors (selected keys)
    /// - `values`: slice of `[hidden]` tensors (selected values)
    /// - `weights`: router weights (length = num selected)
    ///
    /// Returns `[B, T, hidden]`.
    pub fn sparse_attention_kernel(
        &self,
        query: &Tensor<f32>,
        keys: &[Tensor<f32>],
        values: &[Tensor<f32>],
        weights: &[f32],
    ) -> Result<Tensor<f32>, CudaKernelError> {
        self.dispatch.sparse_attention(query, keys, values, weights)
    }
}

impl Default for AttentionKernels {
    fn default() -> Self {
        Self::new()
    }
}
