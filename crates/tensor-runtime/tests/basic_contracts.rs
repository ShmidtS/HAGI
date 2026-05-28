use core_types::{dtype::DType, shape::Shape, TensorSpec};
use tensor_runtime::{from_vec, Backend, Tensor, TensorError};

#[test]
fn tensor_from_vec_preserves_shape_data_and_backend() {
    let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));

    assert_eq!(tensor.shape(), &Shape::new(vec![2, 2]));
    assert_eq!(tensor.numel(), 4);
    assert_eq!(tensor.data(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(tensor.backend(), Backend::Cpu);
}

#[test]
fn zeros_with_spec_uses_dtype_and_zero_values() {
    let spec = TensorSpec::new(Shape::new(vec![2, 3]), DType::F32);
    let tensor = Tensor::<f32>::zeros_with_spec(spec, Backend::Cpu).unwrap();

    assert_eq!(tensor.shape(), &Shape::new(vec![2, 3]));
    assert_eq!(tensor.dtype(), DType::F32);
    assert_eq!(tensor.data(), &[0.0; 6]);
}

#[test]
fn from_vec_reports_length_mismatch() {
    let spec = TensorSpec::new(Shape::new(vec![2, 2]), DType::F32);
    let err = from_vec::<f32>(spec, vec![1.0, 2.0, 3.0], Backend::Cpu).unwrap_err();

    assert_eq!(
        err,
        TensorError::DataLengthMismatch {
            expected: 4,
            actual: 3
        }
    );
}
