use approx::assert_relative_eq;
use clifford_core::{rotor_sandwich_cl3, Cl3, Multivector, ProductTable, Rotor};
use core_types::shape::Shape;
use cuda_kernels::cuda_kernels_available;
#[cfg(feature = "cuda")]
use cuda_kernels::dispatch::GpuBackend;
use cuda_kernels::dispatch::{
    dispatch_or_fallback, Backend, CpuBackend, FusedHagiOp, KernelDispatch,
};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor_runtime::Tensor;

fn fused_inputs() -> (
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
) {
    let input = Tensor::from_vec(vec![0.25, -0.5, 0.75, 1.0], Shape::new(vec![1, 1, 4]));
    let rotor_lut = Tensor::from_vec(vec![1.0; 8], Shape::new(vec![8]));
    let hrm_weights = Tensor::from_vec(vec![0.5; 4], Shape::new(vec![4]));
    let routing_keys = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], Shape::new(vec![2, 2]));
    let output = Tensor::zeros(input.shape().clone());
    (input, rotor_lut, hrm_weights, routing_keys, output)
}

#[test]
fn cpu_reference_produces_output() {
    let table = ProductTable::generate(3, 0, 0);
    let n = table.blade_count;
    let input = Tensor::from_vec(
        vec![1.0, 0.5, -0.25, 0.75, 0.0, 0.1, -0.2, 0.3],
        Shape::new(vec![n]),
    );
    let rotor = Tensor::from_vec(
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        Shape::new(vec![n]),
    );

    let cpu = CpuBackend;
    let output = cpu.rotor_sandwich(&rotor, &input, &table).unwrap();

    assert_eq!(output.shape(), input.shape());
    assert_eq!(output.dtype(), input.dtype());
}

#[test]
fn dispatch_records_fallback_when_cuda_unavailable() {
    let (input, rotor_lut, hrm_weights, routing_keys, mut output) = fused_inputs();
    let before = output.data().to_vec();

    let report = dispatch_or_fallback(
        FusedHagiOp::RotorHrmMsa {
            stream: None,
            input: &input,
            rotor_lut: &rotor_lut,
            hrm_weights: &hrm_weights,
            routing_keys: &routing_keys,
            output: output.as_view_mut(),
        },
        Backend::Gpu,
    )
    .unwrap();

    if !cuda_kernels_available() {
        assert_eq!(report.backend, Backend::Cpu);
        assert!(!report.used_cuda);
        assert!(report.fallback_used);
        assert!(!report.launched_fused);
        assert_eq!(report.registers_per_thread, 0);
        assert_eq!(report.occupancy_percent, 0.0);
        assert!(!report.used_tma);
        assert!(report.fallback_error.is_some());
        assert_ne!(output.data(), before.as_slice());
        assert!(output.data().iter().any(|value| *value != 0.0));
    }
}

#[test]
fn parity_rotor_sandwich_matches_clifford_core() {
    let table = ProductTable::generate(3, 0, 0);
    let n = table.blade_count;
    let mut rng = ChaCha8Rng::seed_from_u64(606);

    let angle: f32 = rng.gen_range(-1.0..1.0);
    let mut rotor_coeffs = vec![0.0f32; n];
    rotor_coeffs[0] = (angle / 2.0).cos();
    rotor_coeffs[4] = (angle / 2.0).sin();
    let mv_coeffs: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let rotor = Rotor::<Cl3>::unit(Multivector::<Cl3>::from_coeffs(rotor_coeffs.clone()));
    let input = Multivector::<Cl3>::from_coeffs(mv_coeffs.clone());
    let expected = rotor_sandwich_cl3(&rotor, &input);

    let cpu = CpuBackend;
    let actual = cpu
        .rotor_sandwich(
            &Tensor::from_vec(rotor.mv.coeffs.clone(), Shape::new(vec![n])),
            &Tensor::from_vec(mv_coeffs, Shape::new(vec![n])),
            &table,
        )
        .unwrap();

    for (actual, expected) in actual.data().iter().zip(expected.coeffs.iter()) {
        assert_relative_eq!(*actual, *expected, epsilon = 1e-4);
    }
}

#[test]
fn parity_geometric_product_matches_clifford_core() {
    let table = ProductTable::generate(3, 0, 0);
    let n = table.blade_count;
    let mut rng = ChaCha8Rng::seed_from_u64(707);
    let a_coeffs: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let b_coeffs: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let expected = Multivector::<Cl3>::from_coeffs(a_coeffs.clone())
        .geometric_product(&Multivector::<Cl3>::from_coeffs(b_coeffs.clone()), &table);

    let cpu = CpuBackend;
    let actual = cpu
        .geometric_product(
            &Tensor::from_vec(a_coeffs, Shape::new(vec![n])),
            &Tensor::from_vec(b_coeffs, Shape::new(vec![n])),
            &table,
        )
        .unwrap();

    for (actual, expected) in actual.data().iter().zip(expected.coeffs.iter()) {
        assert_relative_eq!(*actual, *expected, epsilon = 1e-4);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_geometric_product_matches_cpu() {
    if !cuda_kernels_available() {
        return;
    }

    let table = ProductTable::generate(3, 0, 0);
    let n = table.blade_count;
    let a = Tensor::from_vec(
        vec![1.0, 0.5, -0.25, 0.75, 0.0, 0.1, -0.2, 0.3],
        Shape::new(vec![n]),
    );
    let b = Tensor::from_vec(
        vec![0.7, -0.1, 0.2, 0.4, -0.3, 0.9, 0.05, -0.6],
        Shape::new(vec![n]),
    );

    let cpu = CpuBackend.geometric_product(&a, &b, &table).unwrap();
    let gpu = GpuBackend::new().geometric_product(&a, &b, &table).unwrap();

    for (actual, expected) in gpu.data().iter().zip(cpu.data()) {
        assert_relative_eq!(*actual, *expected, epsilon = 1e-4);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rotor_sandwich_matches_cpu() {
    if !cuda_kernels_available() {
        return;
    }

    let table = ProductTable::generate(3, 0, 0);
    let n = table.blade_count;
    let rotor = Tensor::from_vec(
        vec![0.9238795, 0.0, 0.0, 0.0, 0.38268343, 0.0, 0.0, 0.0],
        Shape::new(vec![n]),
    );
    let mv = Tensor::from_vec(
        vec![0.0, 1.0, 0.5, -0.25, 0.0, 0.1, -0.2, 0.3],
        Shape::new(vec![n]),
    );

    let cpu = CpuBackend.rotor_sandwich(&rotor, &mv, &table).unwrap();
    let gpu = GpuBackend::new()
        .rotor_sandwich(&rotor, &mv, &table)
        .unwrap();

    for (actual, expected) in gpu.data().iter().zip(cpu.data()) {
        assert_relative_eq!(*actual, *expected, epsilon = 1e-4);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fused_dispatch_reports_gpu_and_writes_output() {
    if !cuda_kernels_available() {
        return;
    }

    let (input, rotor_lut, hrm_weights, routing_keys, mut output) = fused_inputs();
    let report = dispatch_or_fallback(
        FusedHagiOp::RotorHrmMsa {
            stream: None,
            input: &input,
            rotor_lut: &rotor_lut,
            hrm_weights: &hrm_weights,
            routing_keys: &routing_keys,
            output: output.as_view_mut(),
        },
        Backend::Gpu,
    )
    .unwrap();

    assert_eq!(report.backend, Backend::Gpu);
    assert!(report.used_cuda);
    assert!(!report.fallback_used);
    assert!(report.launched_fused);
    assert!(output.data().iter().any(|value| *value != 0.0));
}
