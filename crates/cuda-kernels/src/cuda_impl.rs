#[cfg(feature = "cuda")]
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use clifford_core::ProductTable;
#[cfg(feature = "cuda")]
use core_types::shape::Shape;
#[cfg(feature = "cuda")]
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use hrm_model::HrmBackbone;
#[cfg(feature = "cuda")]
use msa_adapter::{MemorySlot, SlotRegistry};
#[cfg(feature = "cuda")]
use tensor_runtime::Tensor;

#[cfg(feature = "cuda")]
use crate::dispatch::{CpuBackend, KernelDispatch};
#[cfg(feature = "cuda")]
use crate::{Backend, CudaKernelError, KernelReport};

#[cfg(feature = "cuda")]
const GEOMETRIC_PRODUCT_PTX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/geometric_product.ptx"));
#[cfg(feature = "cuda")]
const ROTOR_SANDWICH_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/rotor_sandwich.ptx"));
#[cfg(feature = "cuda")]
const SPARSE_ATTENTION_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/sparse_attention.ptx"));
#[cfg(feature = "cuda")]
const MSA_ROUTE_SCORE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/msa_route_score.ptx"));
#[cfg(feature = "cuda")]
const FUSED_ROTOR_HRM_MSA_PTX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/fused_rotor_hrm_msa.ptx"));

#[cfg(feature = "cuda")]
pub(crate) fn driver_error(err: cuda_core::DriverError) -> CudaKernelError {
    CudaKernelError::Unavailable(err.to_string())
}

#[cfg(feature = "cuda")]
pub(crate) fn default_context() -> Result<Arc<CudaContext>, CudaKernelError> {
    CudaContext::new(0).map_err(driver_error)
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_geometric_product(
    ctx: &Arc<CudaContext>,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    table: &ProductTable,
) -> Result<Tensor<f32>, CudaKernelError> {
    if table.blade_count != 8 || table.dim != 3 {
        return Err(CudaKernelError::Unsupported(
            "CUDA geometric product supports Cl<3,0,0> only".to_string(),
        ));
    }
    let (batch, out_shape) = match (a.shape().rank(), b.shape().rank()) {
        (1, 1) if a.numel() == 8 && b.numel() == 8 => (1usize, Shape::new(vec![8])),
        (2, 2)
            if a.shape().dims[0] == b.shape().dims[0]
                && a.shape().dims[1] == 8
                && b.shape().dims[1] == 8 =>
        {
            (a.shape().dims[0], Shape::new(vec![a.shape().dims[0], 8]))
        }
        _ => {
            return Err(CudaKernelError::InvalidShape(
                "geometric_product CUDA expects [8] or [batch, 8] inputs".to_string(),
            ))
        }
    };

    let stream = ctx.new_stream().map_err(driver_error)?;
    let module = ctx
        .load_module_from_ptx_src(GEOMETRIC_PRODUCT_PTX)
        .map_err(driver_error)?;
    let function = module
        .load_function("geometric_product_cl3_kernel")
        .map_err(driver_error)?;
    let a_dev = DeviceBuffer::from_host(&stream, a.data()).map_err(driver_error)?;
    let b_dev = DeviceBuffer::from_host(&stream, b.data()).map_err(driver_error)?;
    let out_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * 8).map_err(driver_error)?;
    let mut a_ptr = a_dev.cu_deviceptr();
    let mut b_ptr = b_dev.cu_deviceptr();
    let mut out_ptr = out_dev.cu_deviceptr();
    let mut batch_arg = batch as u32;
    let mut params = [
        &mut a_ptr as *mut _ as *mut c_void,
        &mut b_ptr as *mut _ as *mut c_void,
        &mut out_ptr as *mut _ as *mut c_void,
        &mut batch_arg as *mut _ as *mut c_void,
    ];
    let config = LaunchConfig::for_num_elems((batch * 8) as u32);
    unsafe {
        cuda_core::launch_kernel_on_stream(
            &function,
            config.grid_dim,
            config.block_dim,
            config.shared_mem_bytes,
            &stream,
            &mut params,
        )
        .map_err(driver_error)?;
    }
    let out = out_dev.to_host_vec(&stream).map_err(driver_error)?;
    Ok(Tensor::from_vec(out, out_shape))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_rotor_sandwich(
    ctx: &Arc<CudaContext>,
    rotor: &Tensor<f32>,
    mv: &Tensor<f32>,
    table: &ProductTable,
) -> Result<Tensor<f32>, CudaKernelError> {
    if table.blade_count != 8 || table.dim != 3 {
        return Err(CudaKernelError::Unsupported(
            "CUDA rotor sandwich supports Cl<3,0,0> only".to_string(),
        ));
    }
    let (batch, rotor_is_batched, out_shape) = match (rotor.shape().rank(), mv.shape().rank()) {
        (1, 1) if rotor.numel() == 8 && mv.numel() == 8 => (1usize, false, Shape::new(vec![8])),
        (2, 2) if rotor.shape().dims[0] == mv.shape().dims[0] && rotor.shape().dims[1] == 8 && mv.shape().dims[1] == 8 => {
            (mv.shape().dims[0], true, Shape::new(vec![mv.shape().dims[0], 8]))
        }
        (1, 2) if rotor.numel() == 8 && mv.shape().dims[1] == 8 => {
            (mv.shape().dims[0], false, Shape::new(vec![mv.shape().dims[0], 8]))
        }
        _ => return Err(CudaKernelError::InvalidShape("rotor_sandwich CUDA expects [8] rotor with [8]/[batch,8] mv or batched [batch,8] inputs".to_string())),
    };

    let stream = ctx.new_stream().map_err(driver_error)?;
    let module = ctx
        .load_module_from_ptx_src(ROTOR_SANDWICH_PTX)
        .map_err(driver_error)?;
    let function = module
        .load_function("rotor_sandwich_cl3_kernel")
        .map_err(driver_error)?;
    let rotor_dev = DeviceBuffer::from_host(&stream, rotor.data()).map_err(driver_error)?;
    let mv_dev = DeviceBuffer::from_host(&stream, mv.data()).map_err(driver_error)?;
    let out_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * 8).map_err(driver_error)?;
    let mut rotor_ptr = rotor_dev.cu_deviceptr();
    let mut mv_ptr = mv_dev.cu_deviceptr();
    let mut out_ptr = out_dev.cu_deviceptr();
    let mut batch_arg = batch as u32;
    let mut rotor_batched_arg = u32::from(rotor_is_batched);
    let mut params = [
        &mut rotor_ptr as *mut _ as *mut c_void,
        &mut mv_ptr as *mut _ as *mut c_void,
        &mut out_ptr as *mut _ as *mut c_void,
        &mut batch_arg as *mut _ as *mut c_void,
        &mut rotor_batched_arg as *mut _ as *mut c_void,
    ];
    let config = LaunchConfig::for_num_elems(batch as u32);
    unsafe {
        cuda_core::launch_kernel_on_stream(
            &function,
            config.grid_dim,
            config.block_dim,
            config.shared_mem_bytes,
            &stream,
            &mut params,
        )
        .map_err(driver_error)?;
    }
    let out = out_dev.to_host_vec(&stream).map_err(driver_error)?;
    Ok(Tensor::from_vec(out, out_shape))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_sparse_attention(
    ctx: &Arc<CudaContext>,
    query: &Tensor<f32>,
    keys: &[Tensor<f32>],
    values: &[Tensor<f32>],
    weights: &[f32],
) -> Result<Tensor<f32>, CudaKernelError> {
    let shape = query.shape();
    if shape.rank() != 3 || keys.is_empty() || keys.len() != values.len() {
        return Err(CudaKernelError::InvalidShape(
            "sparse_attention CUDA expects query [B,T,H] and matching non-empty keys/values"
                .to_string(),
        ));
    }
    let batch = shape.dims[0];
    let tokens = shape.dims[1];
    let hidden = shape.dims[2];
    for item in keys.iter().chain(values.iter()) {
        if item.shape().rank() != 1 || item.numel() != hidden {
            return Err(CudaKernelError::InvalidShape(
                "sparse_attention CUDA keys/values must be [hidden]".to_string(),
            ));
        }
    }

    let mut key_data = Vec::with_capacity(keys.len() * hidden);
    let mut value_data = Vec::with_capacity(values.len() * hidden);
    for key in keys {
        key_data.extend_from_slice(key.data());
    }
    for value in values {
        value_data.extend_from_slice(value.data());
    }
    let weight_data: Vec<f32> = (0..keys.len())
        .map(|index| weights.get(index).copied().unwrap_or(1.0))
        .collect();

    let stream = ctx.new_stream().map_err(driver_error)?;
    let module = ctx
        .load_module_from_ptx_src(SPARSE_ATTENTION_PTX)
        .map_err(driver_error)?;
    let function = module
        .load_function("sparse_attention_kernel")
        .map_err(driver_error)?;
    let query_dev = DeviceBuffer::from_host(&stream, query.data()).map_err(driver_error)?;
    let keys_dev = DeviceBuffer::from_host(&stream, &key_data).map_err(driver_error)?;
    let values_dev = DeviceBuffer::from_host(&stream, &value_data).map_err(driver_error)?;
    let weights_dev = DeviceBuffer::from_host(&stream, &weight_data).map_err(driver_error)?;
    let out_dev = DeviceBuffer::<f32>::zeroed(&stream, query.numel()).map_err(driver_error)?;
    let mut query_ptr = query_dev.cu_deviceptr();
    let mut keys_ptr = keys_dev.cu_deviceptr();
    let mut values_ptr = values_dev.cu_deviceptr();
    let mut weights_ptr = weights_dev.cu_deviceptr();
    let mut out_ptr = out_dev.cu_deviceptr();
    let mut bt_count_arg = (batch * tokens) as u32;
    let mut hidden_arg = hidden as u32;
    let mut selected_arg = keys.len() as u32;
    let mut params = [
        &mut query_ptr as *mut _ as *mut c_void,
        &mut keys_ptr as *mut _ as *mut c_void,
        &mut values_ptr as *mut _ as *mut c_void,
        &mut weights_ptr as *mut _ as *mut c_void,
        &mut out_ptr as *mut _ as *mut c_void,
        &mut bt_count_arg as *mut _ as *mut c_void,
        &mut hidden_arg as *mut _ as *mut c_void,
        &mut selected_arg as *mut _ as *mut c_void,
    ];
    let config = LaunchConfig::for_num_elems(query.numel() as u32);
    unsafe {
        cuda_core::launch_kernel_on_stream(
            &function,
            config.grid_dim,
            config.block_dim,
            config.shared_mem_bytes,
            &stream,
            &mut params,
        )
        .map_err(driver_error)?;
    }
    let out = out_dev.to_host_vec(&stream).map_err(driver_error)?;
    Ok(Tensor::from_vec(
        out,
        Shape::new(vec![batch, tokens, hidden]),
    ))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_msa_route_score(
    ctx: &Arc<CudaContext>,
    query: &[f32],
    routing_keys: &[f32],
    slot_count: usize,
    key_dim: usize,
) -> Result<Vec<f32>, CudaKernelError> {
    let stream = ctx.new_stream().map_err(driver_error)?;
    let module = ctx
        .load_module_from_ptx_src(MSA_ROUTE_SCORE_PTX)
        .map_err(driver_error)?;
    let function = module
        .load_function("msa_route_score_kernel")
        .map_err(driver_error)?;
    let query_dev = DeviceBuffer::from_host(&stream, query).map_err(driver_error)?;
    let keys_dev = DeviceBuffer::from_host(&stream, routing_keys).map_err(driver_error)?;
    let out_dev = DeviceBuffer::<f32>::zeroed(&stream, slot_count).map_err(driver_error)?;
    let mut query_ptr = query_dev.cu_deviceptr();
    let mut keys_ptr = keys_dev.cu_deviceptr();
    let mut out_ptr = out_dev.cu_deviceptr();
    let mut slot_count_arg = slot_count as u32;
    let mut key_dim_arg = key_dim as u32;
    let mut params = [
        &mut query_ptr as *mut _ as *mut c_void,
        &mut keys_ptr as *mut _ as *mut c_void,
        &mut out_ptr as *mut _ as *mut c_void,
        &mut slot_count_arg as *mut _ as *mut c_void,
        &mut key_dim_arg as *mut _ as *mut c_void,
    ];
    let config = LaunchConfig::for_num_elems(slot_count as u32);
    unsafe {
        cuda_core::launch_kernel_on_stream(
            &function,
            config.grid_dim,
            config.block_dim,
            config.shared_mem_bytes,
            &stream,
            &mut params,
        )
        .map_err(driver_error)?;
    }
    out_dev.to_host_vec(&stream).map_err(driver_error)
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_fused_rotor_hrm_msa(
    stream: Option<CudaStream>,
    input: &Tensor<f32>,
    rotor_lut: &Tensor<f32>,
    hrm_weights: &Tensor<f32>,
    routing_keys: &Tensor<f32>,
    output: &mut [f32],
) -> Result<KernelReport, CudaKernelError> {
    let ctx = match stream.as_ref() {
        Some(stream) => stream.context().clone(),
        None => default_context()?,
    };
    let stream = match stream {
        Some(stream) => Arc::new(stream),
        None => ctx.new_stream().map_err(driver_error)?,
    };

    let shape = input.shape();
    let hidden = shape.dims[2];
    let table = ProductTable::generate(3, 0, 0);
    let blade_count = table.blade_count;
    let blade_input = Tensor::from_vec(
        (0..blade_count)
            .map(|index| input.data()[index % input.numel()])
            .collect(),
        Shape::new(vec![blade_count]),
    );
    let rotor = Tensor::from_vec(
        (0..blade_count)
            .map(|index| rotor_lut.data()[index % rotor_lut.numel()])
            .collect(),
        Shape::new(vec![blade_count]),
    );
    let geometric = launch_geometric_product(&ctx, &blade_input, &rotor, &table)?;

    let hrm_config = config::HrmConfig {
        hidden_size: hidden,
        num_heads: 1,
        vocab_size: hidden.max(1),
        max_seq_len: shape.dims[1].max(1),
        ..Default::default()
    };
    let hrm = HrmBackbone::from_config(&hrm_config);
    let hrm_out = CpuBackend.hrm_update(&hrm, input, &vec![shape.dims[1]; shape.dims[0]], 0)?;

    let key_dim = routing_keys.shape().dims[1];
    let mut registry = SlotRegistry::new();
    for slot_id in 0..routing_keys.shape().dims[0] {
        let start = slot_id * key_dim;
        let key = Tensor::from_vec(
            routing_keys.data()[start..start + key_dim].to_vec(),
            Shape::new(vec![key_dim]),
        );
        let value = Tensor::from_vec(vec![0.0; key_dim], Shape::new(vec![key_dim]));
        registry.register(MemorySlot::new(
            slot_id,
            key,
            value,
            0,
            "fallback".to_string(),
        ));
    }
    let query = Tensor::from_vec(
        (0..key_dim)
            .map(|index| hrm_out.data()[index % hrm_out.numel()])
            .collect(),
        Shape::new(vec![key_dim]),
    );
    let route_scores = CpuBackend.msa_route_score(&query, &registry, 1)?;
    let route_scale = route_scores.data().first().copied().unwrap_or(0.0);

    let module = ctx
        .load_module_from_ptx_src(FUSED_ROTOR_HRM_MSA_PTX)
        .map_err(driver_error)?;
    let function = module
        .load_function("fused_rotor_hrm_msa_kernel")
        .map_err(driver_error)?;
    let input_dev = DeviceBuffer::from_host(&stream, input.data()).map_err(driver_error)?;
    let geometric_dev = DeviceBuffer::from_host(&stream, geometric.data()).map_err(driver_error)?;
    let weights_dev = DeviceBuffer::from_host(&stream, hrm_weights.data()).map_err(driver_error)?;
    let hrm_dev = DeviceBuffer::from_host(&stream, hrm_out.data()).map_err(driver_error)?;
    let out_dev = DeviceBuffer::<f32>::zeroed(&stream, input.numel()).map_err(driver_error)?;

    let mut input_ptr = input_dev.cu_deviceptr();
    let mut geometric_ptr = geometric_dev.cu_deviceptr();
    let mut weights_ptr = weights_dev.cu_deviceptr();
    let mut hrm_ptr = hrm_dev.cu_deviceptr();
    let mut out_ptr = out_dev.cu_deviceptr();
    let mut total_arg = input.numel() as u32;
    let mut geometric_len_arg = geometric.numel() as u32;
    let mut weights_len_arg = hrm_weights.numel() as u32;
    let mut route_scale_arg = route_scale;
    let mut params = [
        &mut input_ptr as *mut _ as *mut c_void,
        &mut geometric_ptr as *mut _ as *mut c_void,
        &mut weights_ptr as *mut _ as *mut c_void,
        &mut hrm_ptr as *mut _ as *mut c_void,
        &mut out_ptr as *mut _ as *mut c_void,
        &mut total_arg as *mut _ as *mut c_void,
        &mut geometric_len_arg as *mut _ as *mut c_void,
        &mut weights_len_arg as *mut _ as *mut c_void,
        &mut route_scale_arg as *mut _ as *mut c_void,
    ];
    let config = LaunchConfig::for_num_elems(input.numel() as u32);
    unsafe {
        cuda_core::launch_kernel_on_stream(
            &function,
            config.grid_dim,
            config.block_dim,
            config.shared_mem_bytes,
            &stream,
            &mut params,
        )
        .map_err(driver_error)?;
    }
    out_dev
        .copy_to_host(&stream, output)
        .map_err(driver_error)?;

    let _ = shape;
    Ok(KernelReport {
        backend: Backend::Gpu,
        used_cuda: true,
        fallback_used: false,
        fallback_error: None,
        launched_fused: true,
        registers_per_thread: 0,
        occupancy_percent: 0.0,
        used_tma: false,
        operation: "fused_rotor_hrm_msa",
    })
}
