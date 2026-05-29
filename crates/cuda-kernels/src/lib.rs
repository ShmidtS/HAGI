//! cuda-oxide GPU kernels — dispatch layer for Clifford and attention kernels.
//!
//! Provides a [`Backend`] enum and [`KernelDispatch`] trait that route
//! computations to CPU or GPU implementations. When CUDA is unavailable the
//! GPU backend transparently falls back to the CPU path, so the crate always
//! compiles and runs on CPU-only systems.

pub mod attention;
pub mod clifford;
#[cfg(feature = "cuda")]
mod cuda_impl;
pub mod dispatch;
pub mod fused;

pub use attention::AttentionKernels;
pub use clifford::CliffordKernels;
pub use dispatch::{AutoDispatch, Backend, CpuBackend, GpuBackend, KernelDispatch};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CudaKernelError {
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    #[error("CUDA unavailable: {0}")]
    Unavailable(String),
    #[error("device error")]
    DeviceError,
    #[error("parity mismatch between CPU and CUDA outputs")]
    ParityMismatch,
    #[error("invalid shape: {0}")]
    InvalidShape(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct KernelReport {
    pub backend: Backend,
    pub used_cuda: bool,
    pub fallback_used: bool,
    pub fallback_error: Option<String>,
    pub launched_fused: bool,
    pub registers_per_thread: u16,
    pub occupancy_percent: f32,
    pub used_tma: bool,
    pub operation: &'static str,
}

/// Returns whether a CUDA runtime is detected in this build.
#[cfg(feature = "cuda")]
pub fn cuda_kernels_available() -> bool {
    cuda_core::CudaContext::new(0).is_ok()
}

/// Returns whether a CUDA runtime is detected in this build.
#[cfg(not(feature = "cuda"))]
pub fn cuda_kernels_available() -> bool {
    false
}
