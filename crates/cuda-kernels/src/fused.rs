//! Fused HAGI CUDA kernel launch scaffolding.

use tensor_runtime::Tensor;

use crate::{CudaKernelError, KernelReport};

#[cfg(feature = "cuda")]
pub type CudaStream = cuda_core::CudaStream;

#[cfg(not(feature = "cuda"))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaStream;

/// Launches the fused rotor, HRM, and MSA kernel.
pub fn launch_fused_rotor_hrm_msa(
    stream: Option<CudaStream>,
    input: &Tensor<f32>,
    rotor_lut: &Tensor<f32>,
    hrm_weights: &Tensor<f32>,
    routing_keys: &Tensor<f32>,
    output: &mut [f32],
) -> Result<KernelReport, CudaKernelError> {
    validate_fused_inputs(input, rotor_lut, hrm_weights, routing_keys, output.len())?;

    if !crate::cuda_kernels_available() {
        return Err(CudaKernelError::Unavailable(
            "cuda feature/runtime is not available".to_string(),
        ));
    }

    #[cfg(feature = "cuda")]
    {
        launch_fused_rotor_hrm_msa_cuda(stream, input, rotor_lut, hrm_weights, routing_keys, output)
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (stream, input, rotor_lut, hrm_weights, routing_keys, output);
        Err(CudaKernelError::Unavailable(
            "cuda feature/runtime is not available".to_string(),
        ))
    }
}

fn validate_fused_inputs(
    input: &Tensor<f32>,
    rotor_lut: &Tensor<f32>,
    hrm_weights: &Tensor<f32>,
    routing_keys: &Tensor<f32>,
    output_len: usize,
) -> Result<(), CudaKernelError> {
    if input.shape().rank() != 3 {
        return Err(CudaKernelError::InvalidShape(
            "fused_rotor_hrm_msa input must be [B, T, hidden]".to_string(),
        ));
    }
    if output_len != input.numel() {
        return Err(CudaKernelError::InvalidShape(format!(
            "fused_rotor_hrm_msa output length must match input: {} != {}",
            output_len,
            input.numel()
        )));
    }
    if rotor_lut.shape().rank() == 0 || rotor_lut.numel() == 0 {
        return Err(CudaKernelError::InvalidShape(
            "fused_rotor_hrm_msa rotor_lut must be non-empty".to_string(),
        ));
    }
    if hrm_weights.shape().rank() == 0 || hrm_weights.numel() == 0 {
        return Err(CudaKernelError::InvalidShape(
            "fused_rotor_hrm_msa hrm_weights must be non-empty".to_string(),
        ));
    }
    if routing_keys.shape().rank() != 2 {
        return Err(CudaKernelError::InvalidShape(
            "fused_rotor_hrm_msa routing_keys must be [slot_count, routing_key_dim]".to_string(),
        ));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn launch_fused_rotor_hrm_msa_cuda(
    stream: Option<CudaStream>,
    input: &Tensor<f32>,
    rotor_lut: &Tensor<f32>,
    hrm_weights: &Tensor<f32>,
    routing_keys: &Tensor<f32>,
    output: &mut [f32],
) -> Result<KernelReport, CudaKernelError> {
    crate::cuda_impl::launch_fused_rotor_hrm_msa(
        stream,
        input,
        rotor_lut,
        hrm_weights,
        routing_keys,
        output,
    )
}
