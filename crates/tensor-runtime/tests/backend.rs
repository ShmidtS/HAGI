use core_types::shape::Shape;
use tensor_runtime::backend::{BackendRequest, CpuBackend, TensorBackend};
use tensor_runtime::Tensor;

#[test]
fn cpu_backend_name() {
    let backend = CpuBackend;
    assert_eq!(backend.name(), "cpu");
}

#[test]
fn cpu_backend_zeros() {
    let backend = CpuBackend;
    let t = backend.zeros(Shape::new(vec![2, 3])).unwrap();
    assert_eq!(t.shape().dims, vec![2, 3]);
    assert_eq!(t.numel(), 6);
    assert_eq!(t.data(), &[0.0f32; 6]);
}

#[test]
fn cpu_backend_add() {
    let backend = CpuBackend;
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], Shape::new(vec![3]));
    let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], Shape::new(vec![3]));
    let result = backend.execute(BackendRequest::Add(&a, &b)).unwrap();
    assert_eq!(result.data(), &[5.0, 7.0, 9.0]);
}

#[test]
fn cpu_backend_mul() {
    let backend = CpuBackend;
    let a = Tensor::from_vec(vec![2.0f32, 3.0, 4.0], Shape::new(vec![3]));
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0], Shape::new(vec![3]));
    let result = backend.execute(BackendRequest::Mul(&a, &b)).unwrap();
    assert_eq!(result.data(), &[10.0, 18.0, 28.0]);
}

#[test]
fn cpu_backend_add_shape_mismatch() {
    let backend = CpuBackend;
    let a = Tensor::from_vec(vec![1.0f32, 2.0], Shape::new(vec![2]));
    let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], Shape::new(vec![3]));
    let result = backend.execute(BackendRequest::Add(&a, &b));
    assert!(result.is_err());
}

#[test]
fn cpu_backend_clone_op() {
    let backend = CpuBackend;
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], Shape::new(vec![3]));
    let result = backend.execute(BackendRequest::Clone(&a)).unwrap();
    assert_eq!(result.data(), a.data());
}

#[test]
fn cpu_backend_transfer_from_host() {
    let backend = CpuBackend;
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let t = backend
        .transfer_from_host(data.clone(), Shape::new(vec![2, 2]))
        .unwrap();
    assert_eq!(t.data(), &data[..]);
    assert_eq!(t.shape().dims, vec![2, 2]);
}
