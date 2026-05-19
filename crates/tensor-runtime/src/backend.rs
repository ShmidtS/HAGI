use thiserror::Error;

#[derive(Error, Debug)]
pub enum BackendError {
    #[error("backend not available")]
    NotAvailable,
}

/// Backend abstraction for future cuda-oxide integration.
pub trait TensorBackend {
    type Elem: Copy;
    fn name(&self) -> &'static str;
}

/// CPU reference backend.
pub struct CpuBackend;

impl TensorBackend for CpuBackend {
    type Elem = f32;
    fn name(&self) -> &'static str {
        "cpu"
    }
}
