//! Kernel dispatch: Backend enum, KernelDispatch trait, CPU/GPU implementations.

use clifford_core::ProductTable;
use core_types::shape::Shape;
use hrm_model::HrmBackbone;
use msa_adapter::{route_top_k, MemorySlot, RoutingQueryView, SlotRegistry};
use tensor_runtime::{Tensor, TensorViewMut};

#[cfg(feature = "cuda")]
use cuda_core::CudaContext;
#[cfg(feature = "cuda")]
use std::sync::Arc;

use crate::{CudaKernelError, KernelReport};

/// Which backend is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Gpu,
}

pub enum FusedHagiOp<'a> {
    RotorHrmMsa {
        stream: Option<crate::fused::CudaStream>,
        input: &'a Tensor<f32>,
        rotor_lut: &'a Tensor<f32>,
        hrm_weights: &'a Tensor<f32>,
        routing_keys: &'a Tensor<f32>,
        output: TensorViewMut<'a, f32>,
    },
}

pub fn dispatch_or_fallback(
    op: FusedHagiOp<'_>,
    backend: Backend,
) -> Result<KernelReport, CudaKernelError> {
    match op {
        FusedHagiOp::RotorHrmMsa {
            stream,
            input,
            rotor_lut,
            hrm_weights,
            routing_keys,
            output,
        } => dispatch_fused_rotor_hrm_msa(
            stream,
            input,
            rotor_lut,
            hrm_weights,
            routing_keys,
            output,
            backend,
        ),
    }
}

fn dispatch_fused_rotor_hrm_msa(
    stream: Option<crate::fused::CudaStream>,
    input: &Tensor<f32>,
    rotor_lut: &Tensor<f32>,
    hrm_weights: &Tensor<f32>,
    routing_keys: &Tensor<f32>,
    mut output: TensorViewMut<'_, f32>,
    backend: Backend,
) -> Result<KernelReport, CudaKernelError> {
    if backend == Backend::Gpu && crate::cuda_kernels_available() {
        match crate::fused::launch_fused_rotor_hrm_msa(
            stream,
            input,
            rotor_lut,
            hrm_weights,
            routing_keys,
            output.data_mut(),
        ) {
            Ok(report) => Ok(report),
            Err(err) => {
                cpu_fused_fallback(input, rotor_lut, hrm_weights, routing_keys, &mut output)?;
                Ok(fused_cpu_fallback_report(Some(err.to_string())))
            }
        }
    } else {
        let fallback_error = if backend == Backend::Gpu {
            Some(
                CudaKernelError::Unavailable("cuda feature/runtime is not available".to_string())
                    .to_string(),
            )
        } else {
            None
        };
        cpu_fused_fallback(input, rotor_lut, hrm_weights, routing_keys, &mut output)?;
        Ok(fused_cpu_fallback_report(fallback_error))
    }
}

fn cpu_fused_fallback(
    input: &Tensor<f32>,
    rotor_lut: &Tensor<f32>,
    hrm_weights: &Tensor<f32>,
    routing_keys: &Tensor<f32>,
    output: &mut TensorViewMut<'_, f32>,
) -> Result<(), CudaKernelError> {
    if output.shape() != input.shape() {
        return Err(CudaKernelError::InvalidShape(format!(
            "fused_rotor_hrm_msa output shape must match input: {:?} != {:?}",
            output.shape().dims,
            input.shape().dims
        )));
    }

    let shape = input.shape();
    if shape.rank() != 3 {
        return Err(CudaKernelError::InvalidShape(
            "fused_rotor_hrm_msa input must be [B, T, hidden]".to_string(),
        ));
    }
    if rotor_lut.numel() == 0 || hrm_weights.numel() == 0 {
        return Err(CudaKernelError::InvalidShape(
            "fused_rotor_hrm_msa rotor_lut and hrm_weights must be non-empty".to_string(),
        ));
    }
    if routing_keys.shape().rank() != 2 || routing_keys.shape().dims[0] == 0 {
        return Err(CudaKernelError::InvalidShape(
            "fused_rotor_hrm_msa routing_keys must be non-empty [slot_count, routing_key_dim]"
                .to_string(),
        ));
    }

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
    let geometric = CpuBackend.geometric_product(&blade_input, &rotor, &table)?;

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

    let out = output.data_mut();
    for (index, value) in out.iter_mut().enumerate() {
        let gp = geometric.data()[index % geometric.numel()];
        let hrm = hrm_out.data()[index];
        let weight = hrm_weights.data()[index % hrm_weights.numel()];
        *value = hrm + gp * weight + route_scale;
    }
    Ok(())
}

pub(crate) fn fused_cpu_fallback_report(fallback_error: Option<String>) -> KernelReport {
    KernelReport {
        backend: Backend::Cpu,
        used_cuda: false,
        fallback_used: true,
        fallback_error,
        launched_fused: false,
        registers_per_thread: 0,
        occupancy_percent: 0.0,
        used_tma: false,
        operation: "fused_rotor_hrm_msa",
    }
}

/// Trait defining the kernel surface. Each method is one kernel family.
pub trait KernelDispatch: Send + Sync {
    fn geometric_product(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        table: &ProductTable,
    ) -> Result<Tensor<f32>, CudaKernelError>;

    fn rotor_sandwich(
        &self,
        rotor: &Tensor<f32>,
        mv: &Tensor<f32>,
        table: &ProductTable,
    ) -> Result<Tensor<f32>, CudaKernelError>;

    fn sparse_attention(
        &self,
        query: &Tensor<f32>,
        keys: &[Tensor<f32>],
        values: &[Tensor<f32>],
        weights: &[f32],
    ) -> Result<Tensor<f32>, CudaKernelError>;

    fn hrm_update(
        &self,
        model: &HrmBackbone,
        input: &Tensor<f32>,
        prefix_lens: &[usize],
        step: usize,
    ) -> Result<Tensor<f32>, CudaKernelError>;

    fn msa_route_score(
        &self,
        query: &Tensor<f32>,
        registry: &SlotRegistry,
        top_k: usize,
    ) -> Result<Tensor<f32>, CudaKernelError>;
}

/// Pure CPU backend. Implements all kernels via slice iteration.
pub struct CpuBackend;

impl KernelDispatch for CpuBackend {
    fn geometric_product(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        table: &ProductTable,
    ) -> Result<Tensor<f32>, CudaKernelError> {
        cpu_geometric_product(a.data(), b.data(), a.shape(), b.shape(), table)
    }

    fn rotor_sandwich(
        &self,
        rotor: &Tensor<f32>,
        mv: &Tensor<f32>,
        table: &ProductTable,
    ) -> Result<Tensor<f32>, CudaKernelError> {
        cpu_rotor_sandwich(rotor.data(), mv.data(), rotor.shape(), mv.shape(), table)
    }

    fn sparse_attention(
        &self,
        query: &Tensor<f32>,
        keys: &[Tensor<f32>],
        values: &[Tensor<f32>],
        weights: &[f32],
    ) -> Result<Tensor<f32>, CudaKernelError> {
        cpu_sparse_attention(query, keys, values, weights)
    }

    fn hrm_update(
        &self,
        model: &HrmBackbone,
        input: &Tensor<f32>,
        prefix_lens: &[usize],
        step: usize,
    ) -> Result<Tensor<f32>, CudaKernelError> {
        cpu_hrm_update(model, input, prefix_lens, step)
    }

    fn msa_route_score(
        &self,
        query: &Tensor<f32>,
        registry: &SlotRegistry,
        top_k: usize,
    ) -> Result<Tensor<f32>, CudaKernelError> {
        cpu_msa_route_score(query, registry, top_k)
    }
}

/// GPU backend. Falls back to CpuBackend when CUDA is unavailable.
pub struct GpuBackend {
    fallback: CpuBackend,
    #[cfg(feature = "cuda")]
    context: Option<Arc<CudaContext>>,
}

impl GpuBackend {
    pub fn new() -> Self {
        Self {
            fallback: CpuBackend,
            #[cfg(feature = "cuda")]
            context: crate::cuda_impl::default_context().ok(),
        }
    }

    #[cfg(feature = "cuda")]
    fn context(&self) -> Result<&Arc<CudaContext>, CudaKernelError> {
        self.context.as_ref().ok_or_else(|| {
            CudaKernelError::Unavailable("cuda feature/runtime is not available".to_string())
        })
    }
}

impl Default for GpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelDispatch for GpuBackend {
    fn geometric_product(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        table: &ProductTable,
    ) -> Result<Tensor<f32>, CudaKernelError> {
        #[cfg(feature = "cuda")]
        {
            if let Ok(context) = self.context() {
                if let Ok(output) = crate::cuda_impl::launch_geometric_product(context, a, b, table)
                {
                    return Ok(output);
                }
            }
        }
        self.fallback.geometric_product(a, b, table)
    }

    fn rotor_sandwich(
        &self,
        rotor: &Tensor<f32>,
        mv: &Tensor<f32>,
        table: &ProductTable,
    ) -> Result<Tensor<f32>, CudaKernelError> {
        #[cfg(feature = "cuda")]
        {
            if let Ok(context) = self.context() {
                if let Ok(output) =
                    crate::cuda_impl::launch_rotor_sandwich(context, rotor, mv, table)
                {
                    return Ok(output);
                }
            }
        }
        self.fallback.rotor_sandwich(rotor, mv, table)
    }

    fn sparse_attention(
        &self,
        query: &Tensor<f32>,
        keys: &[Tensor<f32>],
        values: &[Tensor<f32>],
        weights: &[f32],
    ) -> Result<Tensor<f32>, CudaKernelError> {
        #[cfg(feature = "cuda")]
        {
            if let Ok(context) = self.context() {
                if let Ok(output) =
                    crate::cuda_impl::launch_sparse_attention(context, query, keys, values, weights)
                {
                    return Ok(output);
                }
            }
        }
        self.fallback.sparse_attention(query, keys, values, weights)
    }

    fn hrm_update(
        &self,
        model: &HrmBackbone,
        input: &Tensor<f32>,
        prefix_lens: &[usize],
        step: usize,
    ) -> Result<Tensor<f32>, CudaKernelError> {
        if !super::cuda_kernels_available() {
            return self.fallback.hrm_update(model, input, prefix_lens, step);
        }
        self.fallback.hrm_update(model, input, prefix_lens, step)
    }

    fn msa_route_score(
        &self,
        query: &Tensor<f32>,
        registry: &SlotRegistry,
        top_k: usize,
    ) -> Result<Tensor<f32>, CudaKernelError> {
        #[cfg(feature = "cuda")]
        {
            if let Ok(context) = self.context() {
                if !registry.is_empty() {
                    let all_keys = registry.all_keys();
                    let dim = query.shape().dims.last().copied().ok_or_else(|| {
                        CudaKernelError::InvalidShape("msa_route_score query is empty".to_string())
                    })?;
                    if all_keys.shape().rank() == 2 && all_keys.shape().dims[1] == dim {
                        let query_data = if query.data().len() == dim {
                            query.data().to_vec()
                        } else {
                            mean_query_rows(query.data(), dim)
                        };
                        if let Ok(scores) = crate::cuda_impl::launch_msa_route_score(
                            context,
                            &query_data,
                            all_keys.data(),
                            all_keys.shape().dims[0],
                            dim,
                        ) {
                            let mut scored: Vec<(usize, f32)> =
                                registry.slot_ids().into_iter().zip(scores).collect();
                            scored.sort_by(|a, b| {
                                b.1.partial_cmp(&a.1)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                                    .then_with(|| a.0.cmp(&b.0))
                            });
                            scored.truncate(top_k.min(scored.len()));
                            return Ok(Tensor::from_vec(
                                scored.into_iter().map(|(_, score)| score).collect(),
                                Shape::new(vec![top_k.min(registry.len())]),
                            ));
                        }
                    }
                }
            }
        }
        self.fallback.msa_route_score(query, registry, top_k)
    }
}

/// Automatically selects GPU if available, otherwise CPU.
pub struct AutoDispatch {
    backend: Backend,
    gpu: GpuBackend,
    cpu: CpuBackend,
}

impl AutoDispatch {
    pub fn new() -> Self {
        let backend = if super::cuda_kernels_available() {
            Backend::Gpu
        } else {
            Backend::Cpu
        };
        Self {
            backend,
            gpu: GpuBackend::new(),
            cpu: CpuBackend,
        }
    }

    pub fn active_backend(&self) -> Backend {
        self.backend
    }
}

impl Default for AutoDispatch {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelDispatch for AutoDispatch {
    fn geometric_product(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        table: &ProductTable,
    ) -> Result<Tensor<f32>, CudaKernelError> {
        match self.backend {
            Backend::Gpu => self.gpu.geometric_product(a, b, table),
            Backend::Cpu => self.cpu.geometric_product(a, b, table),
        }
    }

    fn rotor_sandwich(
        &self,
        rotor: &Tensor<f32>,
        mv: &Tensor<f32>,
        table: &ProductTable,
    ) -> Result<Tensor<f32>, CudaKernelError> {
        match self.backend {
            Backend::Gpu => self.gpu.rotor_sandwich(rotor, mv, table),
            Backend::Cpu => self.cpu.rotor_sandwich(rotor, mv, table),
        }
    }

    fn sparse_attention(
        &self,
        query: &Tensor<f32>,
        keys: &[Tensor<f32>],
        values: &[Tensor<f32>],
        weights: &[f32],
    ) -> Result<Tensor<f32>, CudaKernelError> {
        match self.backend {
            Backend::Gpu => self.gpu.sparse_attention(query, keys, values, weights),
            Backend::Cpu => self.cpu.sparse_attention(query, keys, values, weights),
        }
    }

    fn hrm_update(
        &self,
        model: &HrmBackbone,
        input: &Tensor<f32>,
        prefix_lens: &[usize],
        step: usize,
    ) -> Result<Tensor<f32>, CudaKernelError> {
        match self.backend {
            Backend::Gpu => self.gpu.hrm_update(model, input, prefix_lens, step),
            Backend::Cpu => self.cpu.hrm_update(model, input, prefix_lens, step),
        }
    }

    fn msa_route_score(
        &self,
        query: &Tensor<f32>,
        registry: &SlotRegistry,
        top_k: usize,
    ) -> Result<Tensor<f32>, CudaKernelError> {
        match self.backend {
            Backend::Gpu => self.gpu.msa_route_score(query, registry, top_k),
            Backend::Cpu => self.cpu.msa_route_score(query, registry, top_k),
        }
    }
}

// ---------------------------------------------------------------------------
// CPU kernel implementations (pure slice ops)
// ---------------------------------------------------------------------------

/// Geometric product on raw slices. Supports rank-1 ([blade_count]) and
/// rank-2 ([batch, blade_count]) tensors.
fn cpu_geometric_product(
    a: &[f32],
    b: &[f32],
    a_shape: &Shape,
    b_shape: &Shape,
    table: &ProductTable,
) -> Result<Tensor<f32>, CudaKernelError> {
    let n = table.blade_count;

    let (batch, a_off, b_off, out_shape) = match (a_shape.rank(), b_shape.rank()) {
        (1, 1) => {
            if a.len() != n {
                return Err(CudaKernelError::InvalidShape(format!(
                    "geometric_product expected left length {}, got {}",
                    n,
                    a.len()
                )));
            }
            if b.len() != n {
                return Err(CudaKernelError::InvalidShape(format!(
                    "geometric_product expected right length {}, got {}",
                    n,
                    b.len()
                )));
            }
            (1usize, a, b, Shape::new(vec![n]))
        }
        (2, 2) => {
            let batch_a = a_shape.dims[0];
            let batch_b = b_shape.dims[0];
            if batch_a != batch_b {
                return Err(CudaKernelError::InvalidShape(format!(
                    "geometric_product batch dimensions must match: {} != {}",
                    batch_a, batch_b
                )));
            }
            if a_shape.dims[1] != n {
                return Err(CudaKernelError::InvalidShape(format!(
                    "geometric_product expected left blade dimension {}, got {}",
                    n, a_shape.dims[1]
                )));
            }
            if b_shape.dims[1] != n {
                return Err(CudaKernelError::InvalidShape(format!(
                    "geometric_product expected right blade dimension {}, got {}",
                    n, b_shape.dims[1]
                )));
            }
            (batch_a, a, b, Shape::new(vec![batch_a, n]))
        }
        _ => {
            return Err(CudaKernelError::InvalidShape(
                "geometric_product expects rank-1 or rank-2 tensors".to_string(),
            ));
        }
    };

    let total = batch * n;
    let mut out = vec![0.0f32; total];

    for i in 0..batch {
        let base = i * n;
        gp_slice(a_off, b_off, &mut out, base, n, table);
    }

    Ok(Tensor::from_vec(out, out_shape))
}

/// Single geometric product on slices starting at `base`.
fn gp_slice(a: &[f32], b: &[f32], out: &mut [f32], base: usize, n: usize, table: &ProductTable) {
    for ai in 0..n {
        let ca = a[base + ai];
        if ca == 0.0 {
            continue;
        }
        for bi in 0..n {
            let cb = b[base + bi];
            if cb == 0.0 {
                continue;
            }
            let entry = &table.entries[ai * n + bi];
            if entry.metric == 0.0 {
                continue;
            }
            let idx = base + entry.result_blade as usize;
            out[idx] += entry.sign as f32 * entry.metric * ca * cb;
        }
    }
}

fn reverse_in_place(rev: &mut [f32], table: &ProductTable) {
    for (i, coeff) in rev.iter_mut().enumerate().take(table.blade_count) {
        let g = table.grade[i] as usize;
        if g >= 2 && !(g * (g - 1) / 2).is_multiple_of(2) {
            *coeff = -*coeff;
        }
    }
}

/// Rotor sandwich: R * mv * reverse(R). Supports rank-1 and rank-2.
fn cpu_rotor_sandwich(
    rotor: &[f32],
    mv: &[f32],
    rotor_shape: &Shape,
    mv_shape: &Shape,
    table: &ProductTable,
) -> Result<Tensor<f32>, CudaKernelError> {
    let n = table.blade_count;

    let (batch, out_shape) = match (rotor_shape.rank(), mv_shape.rank()) {
        (1, 1) => {
            if rotor.len() != n {
                return Err(CudaKernelError::InvalidShape(format!(
                    "rotor_sandwich expected rotor length {}, got {}",
                    n,
                    rotor.len()
                )));
            }
            if mv.len() != n {
                return Err(CudaKernelError::InvalidShape(format!(
                    "rotor_sandwich expected multivector length {}, got {}",
                    n,
                    mv.len()
                )));
            }
            (1usize, Shape::new(vec![n]))
        }
        (2, 2) => {
            let batch_r = rotor_shape.dims[0];
            let batch_m = mv_shape.dims[0];
            if batch_r != batch_m {
                return Err(CudaKernelError::InvalidShape(format!(
                    "rotor_sandwich batch dimensions must match: {} != {}",
                    batch_r, batch_m
                )));
            }
            if rotor_shape.dims[1] != n {
                return Err(CudaKernelError::InvalidShape(format!(
                    "rotor_sandwich expected rotor blade dimension {}, got {}",
                    n, rotor_shape.dims[1]
                )));
            }
            if mv_shape.dims[1] != n {
                return Err(CudaKernelError::InvalidShape(format!(
                    "rotor_sandwich expected multivector blade dimension {}, got {}",
                    n, mv_shape.dims[1]
                )));
            }
            (batch_r, Shape::new(vec![batch_r, n]))
        }
        (1, 2) => {
            let batch_m = mv_shape.dims[0];
            if rotor.len() != n {
                return Err(CudaKernelError::InvalidShape(format!(
                    "rotor_sandwich expected rotor length {}, got {}",
                    n,
                    rotor.len()
                )));
            }
            if mv_shape.dims[1] != n {
                return Err(CudaKernelError::InvalidShape(format!(
                    "rotor_sandwich expected multivector blade dimension {}, got {}",
                    n, mv_shape.dims[1]
                )));
            }
            (batch_m, Shape::new(vec![batch_m, n]))
        }
        _ => {
            return Err(CudaKernelError::InvalidShape(
                "rotor_sandwich expects rank-1 or rank-2 tensors".to_string(),
            ));
        }
    };

    let total = batch * n;
    let mut out = vec![0.0f32; total];

    let mut rev_r = vec![0.0f32; n];
    let mut temp = vec![0.0f32; n];

    for i in 0..batch {
        let base = i * n;
        let r_slice = if rotor_shape.rank() == 1 {
            rotor
        } else {
            &rotor[base..base + n]
        };
        let mv_slice = &mv[base..base + n];

        rev_r.copy_from_slice(r_slice);
        reverse_in_place(&mut rev_r, table);

        // temp = mv * reverse(rotor)
        temp.fill(0.0);
        gp_pair(mv_slice, &rev_r, &mut temp, n, table);

        // result = rotor * temp
        gp_pair(r_slice, &temp, &mut out[base..base + n], n, table);
    }

    Ok(Tensor::from_vec(out, out_shape))
}

/// Compute geometric product of two n-length slices into `out`.
fn gp_pair(a: &[f32], b: &[f32], out: &mut [f32], n: usize, table: &ProductTable) {
    for (ai, &ca) in a.iter().enumerate().take(n) {
        if ca == 0.0 {
            continue;
        }
        for (bi, &cb) in b.iter().enumerate().take(n) {
            if cb == 0.0 {
                continue;
            }
            let entry = &table.entries[ai * n + bi];
            if entry.metric == 0.0 {
                continue;
            }
            let idx = entry.result_blade as usize;
            out[idx] += entry.sign as f32 * entry.metric * ca * cb;
        }
    }
}

/// Sparse attention over selected slots (CPU). Replicates msa-adapter logic.
///
/// Each selected slot contribution is multiplied by `weights[s]`; missing or empty
/// weights default to `1.0` for backward-compatible unweighted attention.
fn cpu_sparse_attention(
    query: &Tensor<f32>,
    selected_keys: &[Tensor<f32>],
    selected_values: &[Tensor<f32>],
    weights: &[f32],
) -> Result<Tensor<f32>, CudaKernelError> {
    let shape = query.shape();
    if shape.rank() != 3 {
        return Err(CudaKernelError::InvalidShape(
            "query must be [B, T, hidden]".to_string(),
        ));
    }
    let batch = shape.dims[0];
    let tokens = shape.dims[1];
    let hidden = shape.dims[2];
    let num_selected = selected_keys.len();
    if selected_values.len() != num_selected {
        return Err(CudaKernelError::InvalidShape(format!(
            "keys and values must have same length: {} != {}",
            num_selected,
            selected_values.len()
        )));
    }
    if num_selected == 0 {
        return Err(CudaKernelError::InvalidShape(
            "must have at least one selected slot".to_string(),
        ));
    }

    let q_data = query.data();
    let scale = 1.0 / (hidden as f32).sqrt();
    let mut out = vec![0.0f32; batch * tokens * hidden];

    for bt in 0..(batch * tokens) {
        let q_off = bt * hidden;

        let mut scores = Vec::with_capacity(num_selected);
        for key in selected_keys.iter() {
            let k_data = key.data();
            let mut dot = 0.0f32;
            for d in 0..hidden {
                dot += q_data[q_off + d] * k_data[d];
            }
            scores.push(dot * scale);
        }

        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        let exp_scores: Vec<f32> = scores
            .iter()
            .map(|&s| {
                let e = (s - max_score).exp();
                exp_sum += e;
                e
            })
            .collect();

        let out_off = bt * hidden;
        for s in 0..num_selected {
            let attn_w = if exp_sum > 0.0 {
                exp_scores[s] / exp_sum
            } else {
                1.0 / num_selected as f32
            };
            let slot_weight = weights.get(s).copied().unwrap_or(1.0);
            let v_data = selected_values[s].data();
            for d in 0..hidden {
                out[out_off + d] += slot_weight * attn_w * v_data[d];
            }
        }
    }

    Ok(Tensor::from_vec(
        out,
        Shape::new(vec![batch, tokens, hidden]),
    ))
}

fn cpu_hrm_update(
    model: &HrmBackbone,
    input: &Tensor<f32>,
    prefix_lens: &[usize],
    step: usize,
) -> Result<Tensor<f32>, CudaKernelError> {
    Ok(model.forward(input, prefix_lens, step).hidden)
}

#[cfg(feature = "cuda")]
fn mean_query_rows(data: &[f32], dim: usize) -> Vec<f32> {
    let rows = data.len() / dim;
    let mut mean = vec![0.0f32; dim];
    for row in 0..rows {
        let offset = row * dim;
        for d in 0..dim {
            mean[d] += data[offset + d];
        }
    }
    for value in &mut mean {
        *value /= rows as f32;
    }
    mean
}

fn cpu_msa_route_score(
    query: &Tensor<f32>,
    registry: &SlotRegistry,
    top_k: usize,
) -> Result<Tensor<f32>, CudaKernelError> {
    let dim = query.shape().dims.last().copied().ok_or_else(|| {
        CudaKernelError::InvalidShape("msa_route_score query is empty".to_string())
    })?;
    let selection = route_top_k(
        registry,
        RoutingQueryView {
            data: query.data(),
            dim,
        },
        top_k,
    )
    .map_err(|err| CudaKernelError::InvalidShape(err.to_string()))?;
    Ok(Tensor::from_vec(
        selection.raw_scores.iter().copied().collect(),
        Shape::new(vec![selection.raw_scores.len()]),
    ))
}
