use clifford_core::ProductTable;
use core_types::shape::Shape;
use cuda_kernels::{
    cuda_kernels_available, AutoDispatch, Backend, CliffordKernels, CpuBackend, KernelDispatch,
};
use tensor_runtime::Tensor;

#[test]
fn cuda_availability_query_is_callable() {
    let _available: bool = cuda_kernels_available();
}

#[test]
fn cpu_geometric_product_uses_fallback_kernel_surface() {
    let table = ProductTable::generate(3, 0, 0);
    let mut scalar = vec![0.0f32; table.blade_count];
    scalar[0] = 1.0;
    let mv = Tensor::from_vec(
        (1..=8).map(|value| value as f32).collect(),
        Shape::new(vec![8]),
    );
    let one = Tensor::from_vec(scalar, Shape::new(vec![8]));

    let output = CpuBackend.geometric_product(&one, &mv, &table).unwrap();

    assert_eq!(output.shape(), &Shape::new(vec![8]));
    assert_eq!(output.data(), mv.data());
}

#[test]
fn auto_dispatch_and_clifford_kernel_are_constructible_on_cpu_only_systems() {
    let dispatch = AutoDispatch::new();
    let backend = dispatch.active_backend();
    let kernels = CliffordKernels::with_dispatch(dispatch);
    let table = ProductTable::generate(3, 0, 0);
    let one = Tensor::from_vec(
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        Shape::new(vec![8]),
    );

    let output = kernels
        .geometric_product_kernel(&one, &one, &table)
        .unwrap();

    assert!(matches!(backend, Backend::Cpu | Backend::Gpu));
    assert_eq!(output.data(), one.data());
}
