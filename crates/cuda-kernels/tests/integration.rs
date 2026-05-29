//! Integration tests for cuda-kernels.

use clifford_core::{Multivector, ProductTable};
use core_types::algebra::Cl;
use core_types::shape::Shape;
use cuda_kernels::attention::AttentionKernels;
use cuda_kernels::clifford::CliffordKernels;
use cuda_kernels::cuda_kernels_available;
use cuda_kernels::dispatch::{
    dispatch_or_fallback, AutoDispatch, Backend, CpuBackend, FusedHagiOp, GpuBackend,
    KernelDispatch,
};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor_runtime::Tensor;

fn assert_near(a: f32, b: f32, eps: f32, ctx: &str) {
    assert!(
        (a - b).abs() < eps,
        "{}: |{} - {}| = {} >= {}",
        ctx,
        a,
        b,
        (a - b).abs(),
        eps
    );
}

// ---------------------------------------------------------------------------
// cuda_kernels_available
// ---------------------------------------------------------------------------

#[test]
fn cuda_kernels_available_query_is_callable() {
    let _available = cuda_kernels_available();
}

// ---------------------------------------------------------------------------
// AutoDispatch picks CPU when CUDA is absent
// ---------------------------------------------------------------------------

#[test]
fn auto_dispatch_tracks_cuda_availability() {
    let d = AutoDispatch::new();
    let expected = if cuda_kernels_available() {
        Backend::Gpu
    } else {
        Backend::Cpu
    };
    assert_eq!(d.active_backend(), expected);
}

// ---------------------------------------------------------------------------
// Geometric product: CPU backend vs clifford-core reference
// ---------------------------------------------------------------------------

#[test]
fn cpu_geometric_product_matches_reference_cl300() {
    let table = ProductTable::generate(3, 0, 0);
    let n = table.blade_count;
    assert_eq!(n, 8);

    // a = 1.0 + 0.5*e1 + 0.3*e23
    let mut a_coeffs = vec![0.0f32; n];
    a_coeffs[0] = 1.0; // scalar
    a_coeffs[1] = 0.5; // e1
    a_coeffs[6] = 0.3; // e23

    // b = 0.7 + 0.2*e2 + 0.4*e13
    let mut b_coeffs = vec![0.0f32; n];
    b_coeffs[0] = 0.7; // scalar
    b_coeffs[2] = 0.2; // e2
    b_coeffs[5] = 0.4; // e13

    let a_mv = Multivector::<Cl<3, 0, 0>>::from_coeffs(a_coeffs.clone());
    let b_mv = Multivector::<Cl<3, 0, 0>>::from_coeffs(b_coeffs.clone());
    let ref_result = a_mv.geometric_product(&b_mv, &table);

    let a_t = Tensor::from_vec(a_coeffs, Shape::new(vec![n]));
    let b_t = Tensor::from_vec(b_coeffs, Shape::new(vec![n]));

    let cpu = CpuBackend;
    let result = cpu.geometric_product(&a_t, &b_t, &table).unwrap();

    assert_eq!(result.shape().dims, vec![n]);
    let data = result.data();
    for i in 0..n {
        assert_near(data[i], ref_result.coeffs[i], 1e-4, &format!("blade {}", i));
    }
}

// ---------------------------------------------------------------------------
// Batched geometric product
// ---------------------------------------------------------------------------

#[test]
fn cpu_geometric_product_batched() {
    let table = ProductTable::generate(3, 0, 0);
    let n = table.blade_count;
    let batch = 4;

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let a_data: Vec<f32> = (0..batch * n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let b_data: Vec<f32> = (0..batch * n).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let a_t = Tensor::from_vec(a_data.clone(), Shape::new(vec![batch, n]));
    let b_t = Tensor::from_vec(b_data.clone(), Shape::new(vec![batch, n]));

    let cpu = CpuBackend;
    let result = cpu.geometric_product(&a_t, &b_t, &table).unwrap();
    assert_eq!(result.shape().dims, vec![batch, n]);

    // Verify each batch element against scalar multivector reference
    for i in 0..batch {
        let base = i * n;
        let a_mv = Multivector::<Cl<3, 0, 0>>::from_coeffs(a_data[base..base + n].to_vec());
        let b_mv = Multivector::<Cl<3, 0, 0>>::from_coeffs(b_data[base..base + n].to_vec());
        let ref_result = a_mv.geometric_product(&b_mv, &table);

        let data = result.data();
        for j in 0..n {
            assert_near(
                data[base + j],
                ref_result.coeffs[j],
                1e-4,
                &format!("batch {} blade {}", i, j),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Rotor sandwich: CPU backend vs clifford-core reference
// ---------------------------------------------------------------------------

#[test]
fn cpu_rotor_sandwich_matches_reference_cl300() {
    let table = ProductTable::generate(3, 0, 0);
    let n = table.blade_count;

    // Unit rotor: rotation by pi/4 in e1-e2 plane
    // R = cos(pi/8) + sin(pi/8) * e12
    let theta = std::f32::consts::FRAC_PI_4;
    let mut rotor_coeffs = vec![0.0f32; n];
    rotor_coeffs[0] = (theta / 2.0).cos(); // scalar
    rotor_coeffs[4] = (theta / 2.0).sin(); // e12

    // Multivector to rotate: mv = e1 (blade index 1)
    let mut mv_coeffs = vec![0.0f32; n];
    mv_coeffs[1] = 1.0; // e1

    // Reference: R * mv * reverse(R)
    let r_mv = Multivector::<Cl<3, 0, 0>>::from_coeffs(rotor_coeffs.clone());
    let mv_ref = Multivector::<Cl<3, 0, 0>>::from_coeffs(mv_coeffs.clone());
    let r_rev = r_mv.reverse(&table);
    let expected = r_mv
        .geometric_product(&mv_ref, &table)
        .geometric_product(&r_rev, &table);

    let rotor_t = Tensor::from_vec(rotor_coeffs, Shape::new(vec![n]));
    let mv_t = Tensor::from_vec(mv_coeffs, Shape::new(vec![n]));

    let cpu = CpuBackend;
    let result = cpu.rotor_sandwich(&rotor_t, &mv_t, &table).unwrap();

    assert_eq!(result.shape().dims, vec![n]);
    let data = result.data();
    for i in 0..n {
        assert_near(data[i], expected.coeffs[i], 1e-4, &format!("blade {}", i));
    }
}

// ---------------------------------------------------------------------------
// Rotor sandwich via CliffordKernels API
// ---------------------------------------------------------------------------

#[test]
fn clifford_kernels_rotor_sandwich_via_api() {
    let table = ProductTable::generate(3, 0, 0);
    let n = table.blade_count;

    let theta = std::f32::consts::FRAC_PI_4;
    let mut rotor_coeffs = vec![0.0f32; n];
    rotor_coeffs[0] = (theta / 2.0).cos();
    rotor_coeffs[4] = (theta / 2.0).sin();

    let mut mv_coeffs = vec![0.0f32; n];
    mv_coeffs[2] = 1.0; // e2

    let r_mv = Multivector::<Cl<3, 0, 0>>::from_coeffs(rotor_coeffs.clone());
    let mv_ref = Multivector::<Cl<3, 0, 0>>::from_coeffs(mv_coeffs.clone());
    let r_rev = r_mv.reverse(&table);
    let expected = r_mv
        .geometric_product(&mv_ref, &table)
        .geometric_product(&r_rev, &table);

    let rotor_t = Tensor::from_vec(rotor_coeffs, Shape::new(vec![n]));
    let mv_t = Tensor::from_vec(mv_coeffs, Shape::new(vec![n]));

    let ck = CliffordKernels::new();
    let result = ck.rotor_sandwich_kernel(&rotor_t, &mv_t, &table).unwrap();

    let data = result.data();
    for i in 0..n {
        assert_near(data[i], expected.coeffs[i], 1e-4, &format!("blade {}", i));
    }
}

// ---------------------------------------------------------------------------
// Sparse attention: CPU backend vs dense attention reference
// ---------------------------------------------------------------------------

/// Reference dense attention implementation (independent of msa-adapter).
fn dense_attention_reference(
    query: &[f32],
    keys: &[&[f32]],
    values: &[&[f32]],
    weights: &[f32],
    batch: usize,
    tokens: usize,
    hidden: usize,
) -> Vec<f32> {
    let num_kv = keys.len();
    let scale = 1.0 / (hidden as f32).sqrt();
    let mut out = vec![0.0f32; batch * tokens * hidden];

    for bt in 0..(batch * tokens) {
        let q_off = bt * hidden;
        let mut scores = Vec::with_capacity(num_kv);
        for s in 0..num_kv {
            let mut dot = 0.0f32;
            for d in 0..hidden {
                dot += query[q_off + d] * keys[s][d];
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
        for s in 0..num_kv {
            let attn_w = if exp_sum > 0.0 {
                exp_scores[s] / exp_sum
            } else {
                1.0 / num_kv as f32
            };
            let slot_weight = weights.get(s).copied().unwrap_or(1.0);
            for d in 0..hidden {
                out[out_off + d] += slot_weight * attn_w * values[s][d];
            }
        }
    }
    out
}

#[test]
fn cpu_sparse_attention_matches_dense_on_5_slots() {
    let mut rng = ChaCha8Rng::seed_from_u64(789);
    let hidden = 16;
    let batch = 2;
    let tokens = 3;
    let num_selected = 5;

    let query_data: Vec<f32> = (0..batch * tokens * hidden)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let query = Tensor::from_vec(query_data.clone(), Shape::new(vec![batch, tokens, hidden]));

    let keys: Vec<Tensor<f32>> = (0..num_selected)
        .map(|_| {
            let d: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-1.0..1.0)).collect();
            Tensor::from_vec(d, Shape::new(vec![hidden]))
        })
        .collect();
    let values: Vec<Tensor<f32>> = (0..num_selected)
        .map(|_| {
            let d: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-1.0..1.0)).collect();
            Tensor::from_vec(d, Shape::new(vec![hidden]))
        })
        .collect();
    let weights = vec![0.5, 1.0, 1.5, 0.25, 2.0];

    let cpu = CpuBackend;
    let result = cpu
        .sparse_attention(&query, &keys, &values, &weights)
        .unwrap();
    assert_eq!(result.shape().dims, vec![batch, tokens, hidden]);

    // Build reference
    let key_slices: Vec<&[f32]> = keys.iter().map(|k| k.data()).collect();
    let val_slices: Vec<&[f32]> = values.iter().map(|v| v.data()).collect();
    let expected = dense_attention_reference(
        &query_data,
        &key_slices,
        &val_slices,
        &weights,
        batch,
        tokens,
        hidden,
    );

    let data = result.data();
    for i in 0..data.len() {
        assert_near(data[i], expected[i], 1e-4, &format!("index {}", i));
    }
}

// ---------------------------------------------------------------------------
// Sparse attention via AttentionKernels API
// ---------------------------------------------------------------------------

#[test]
fn attention_kernels_api_matches_cpu() {
    let mut rng = ChaCha8Rng::seed_from_u64(101);
    let hidden = 8;
    let batch = 1;
    let tokens = 2;
    let num_selected = 5;

    let query_data: Vec<f32> = (0..batch * tokens * hidden)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let query = Tensor::from_vec(query_data.clone(), Shape::new(vec![batch, tokens, hidden]));

    let keys: Vec<Tensor<f32>> = (0..num_selected)
        .map(|_| {
            let d: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-1.0..1.0)).collect();
            Tensor::from_vec(d, Shape::new(vec![hidden]))
        })
        .collect();
    let values: Vec<Tensor<f32>> = (0..num_selected)
        .map(|_| {
            let d: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-1.0..1.0)).collect();
            Tensor::from_vec(d, Shape::new(vec![hidden]))
        })
        .collect();
    let weights = vec![0.5, 1.25, 2.0, 0.75, 1.5];

    let ak = AttentionKernels::new();
    let result = ak
        .sparse_attention_kernel(&query, &keys, &values, &weights)
        .unwrap();

    let key_slices: Vec<&[f32]> = keys.iter().map(|k| k.data()).collect();
    let val_slices: Vec<&[f32]> = values.iter().map(|v| v.data()).collect();
    let expected = dense_attention_reference(
        &query_data,
        &key_slices,
        &val_slices,
        &weights,
        batch,
        tokens,
        hidden,
    );

    let data = result.data();
    for i in 0..data.len() {
        assert_near(data[i], expected[i], 1e-4, &format!("index {}", i));
    }
}

// ---------------------------------------------------------------------------
// GPU fallback produces identical results to CPU
// ---------------------------------------------------------------------------

#[test]
fn gpu_fallback_matches_cpu_geometric_product() {
    let table = ProductTable::generate(3, 0, 0);
    let n = table.blade_count;

    let mut rng = ChaCha8Rng::seed_from_u64(555);
    let a_data: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let b_data: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let a_t = Tensor::from_vec(a_data, Shape::new(vec![n]));
    let b_t = Tensor::from_vec(b_data, Shape::new(vec![n]));

    let cpu = CpuBackend;
    let gpu = GpuBackend::new();

    let cpu_result = cpu.geometric_product(&a_t, &b_t, &table).unwrap();
    let gpu_result = gpu.geometric_product(&a_t, &b_t, &table).unwrap();

    for (index, (cpu, gpu)) in cpu_result.data().iter().zip(gpu_result.data()).enumerate() {
        assert_near(*gpu, *cpu, 1e-4, &format!("index {}", index));
    }
}

#[test]
fn gpu_fallback_matches_cpu_rotor_sandwich() {
    let table = ProductTable::generate(3, 0, 0);
    let n = table.blade_count;

    let theta = std::f32::consts::FRAC_PI_3;
    let mut rotor_coeffs = vec![0.0f32; n];
    rotor_coeffs[0] = (theta / 2.0).cos();
    rotor_coeffs[4] = (theta / 2.0).sin();

    let mut mv_coeffs = vec![0.0f32; n];
    mv_coeffs[1] = 1.0;

    let rotor_t = Tensor::from_vec(rotor_coeffs, Shape::new(vec![n]));
    let mv_t = Tensor::from_vec(mv_coeffs, Shape::new(vec![n]));

    let cpu = CpuBackend;
    let gpu = GpuBackend::new();

    let cpu_result = cpu.rotor_sandwich(&rotor_t, &mv_t, &table).unwrap();
    let gpu_result = gpu.rotor_sandwich(&rotor_t, &mv_t, &table).unwrap();

    for (index, (cpu, gpu)) in cpu_result.data().iter().zip(gpu_result.data()).enumerate() {
        assert_near(*gpu, *cpu, 1e-4, &format!("index {}", index));
    }
}

#[test]
fn gpu_fallback_matches_cpu_sparse_attention() {
    let mut rng = ChaCha8Rng::seed_from_u64(777);
    let hidden = 8;
    let batch = 1;
    let tokens = 2;
    let num_selected = 3;

    let query_data: Vec<f32> = (0..batch * tokens * hidden)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let query = Tensor::from_vec(query_data, Shape::new(vec![batch, tokens, hidden]));

    let keys: Vec<Tensor<f32>> = (0..num_selected)
        .map(|_| {
            let d: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-1.0..1.0)).collect();
            Tensor::from_vec(d, Shape::new(vec![hidden]))
        })
        .collect();
    let values: Vec<Tensor<f32>> = (0..num_selected)
        .map(|_| {
            let d: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-1.0..1.0)).collect();
            Tensor::from_vec(d, Shape::new(vec![hidden]))
        })
        .collect();
    let weights = vec![1.0 / num_selected as f32; num_selected];

    let cpu = CpuBackend;
    let gpu = GpuBackend::new();

    let cpu_result = cpu
        .sparse_attention(&query, &keys, &values, &weights)
        .unwrap();
    let gpu_result = gpu
        .sparse_attention(&query, &keys, &values, &weights)
        .unwrap();

    for (index, (cpu, gpu)) in cpu_result.data().iter().zip(gpu_result.data()).enumerate() {
        assert_near(*gpu, *cpu, 1e-4, &format!("index {}", index));
    }
}

#[test]
fn fused_dispatch_gpu_matches_cpu_when_cuda_available() {
    let input = Tensor::from_vec(
        vec![0.25, -0.5, 0.75, 1.0, -0.25, 0.5, -0.75, 0.125],
        Shape::new(vec![1, 2, 4]),
    );
    let rotor_lut = Tensor::from_vec(
        vec![1.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0],
        Shape::new(vec![8]),
    );
    let hrm_weights = Tensor::from_vec(vec![0.5, 0.25, 0.75, 1.0], Shape::new(vec![4]));
    let routing_keys = Tensor::from_vec(
        vec![0.1, -0.2, 0.3, -0.4, -0.3, 0.2, -0.1, 0.4],
        Shape::new(vec![2, 4]),
    );
    let mut cpu_output = Tensor::zeros(input.shape().clone());
    let mut gpu_output = Tensor::zeros(input.shape().clone());

    let cpu_report = dispatch_or_fallback(
        FusedHagiOp::RotorHrmMsa {
            stream: None,
            input: &input,
            rotor_lut: &rotor_lut,
            hrm_weights: &hrm_weights,
            routing_keys: &routing_keys,
            output: cpu_output.as_view_mut(),
        },
        Backend::Cpu,
    )
    .unwrap();
    let gpu_report = dispatch_or_fallback(
        FusedHagiOp::RotorHrmMsa {
            stream: None,
            input: &input,
            rotor_lut: &rotor_lut,
            hrm_weights: &hrm_weights,
            routing_keys: &routing_keys,
            output: gpu_output.as_view_mut(),
        },
        Backend::Gpu,
    )
    .unwrap();

    for (index, (cpu, gpu)) in cpu_output.data().iter().zip(gpu_output.data()).enumerate() {
        assert_near(*gpu, *cpu, 1e-4, &format!("index {}", index));
    }
    assert!(!cpu_report.used_cuda);
    if cuda_kernels_available() {
        assert_eq!(gpu_report.backend, Backend::Gpu);
        assert!(gpu_report.used_cuda);
        assert!(gpu_report.launched_fused);
        assert!(!gpu_report.fallback_used);
    } else {
        assert_eq!(gpu_report.backend, Backend::Cpu);
        assert!(!gpu_report.used_cuda);
        assert!(!gpu_report.launched_fused);
        assert!(gpu_report.fallback_used);
    }
}
