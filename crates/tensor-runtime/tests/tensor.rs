use core_types::dtype::DType;
use core_types::shape::Shape;
use core_types::tensor_layout::TensorLayout;
use core_types::tensor_spec::TensorSpec;
use tensor_runtime::{Backend, BackendOp, Tensor, TensorError, TensorView, TensorViewMut};

#[test]
fn tensor_zeros() {
    let t: Tensor<f32> = Tensor::zeros(Shape::new(vec![2, 3]));
    assert_eq!(t.shape().dims, vec![2, 3]);
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.numel(), 6);
    assert_eq!(t.data(), &[0.0f32; 6]);
}

#[test]
fn tensor_from_vec() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::from_vec(data.clone(), Shape::new(vec![2, 3]));
    assert_eq!(t.data(), &data[..]);
    assert_eq!(t.numel(), 6);
}

#[test]
fn tensor_error_data_length_mismatch_is_returned() {
    let spec = TensorSpec::new(Shape::new(vec![2, 3]), DType::F32);
    let err = Tensor::<f32>::from_vec_with_spec(spec, vec![1.0, 2.0], Backend::Cpu).unwrap_err();
    assert_eq!(
        err,
        TensorError::DataLengthMismatch {
            expected: 6,
            actual: 2
        }
    );
}

#[test]
fn from_vec_with_spec_rejects_shape_numel_overflow() {
    let dims = vec![usize::MAX, 2];
    let spec = TensorSpec::new(Shape::new(dims.clone()), DType::F32);
    let err = Tensor::<f32>::from_vec_with_spec(spec, Vec::new(), Backend::Cpu).unwrap_err();
    assert_eq!(err, TensorError::ShapeNumelOverflow { dims });
}

#[test]
fn zeros_with_spec_rejects_shape_numel_overflow() {
    let dims = vec![usize::MAX, 2];
    let spec = TensorSpec::new(Shape::new(dims.clone()), DType::F32);
    let err = Tensor::<f32>::zeros_with_spec(spec, Backend::Cpu).unwrap_err();
    assert_eq!(err, TensorError::ShapeNumelOverflow { dims });
}

#[test]
fn backend_enum_defaults_to_cpu_for_new_constructors() {
    let t: Tensor<f32> = Tensor::zeros(Shape::new(vec![2]));
    assert_eq!(t.backend(), Backend::Cpu);
}

#[test]
#[should_panic(expected = "data length must match shape numel")]
fn tensor_from_vec_mismatch() {
    let _ = Tensor::from_vec(vec![1.0f32, 2.0], Shape::new(vec![3, 3]));
}

#[test]
fn tensor_view() {
    let t: Tensor<f32> = Tensor::zeros(Shape::new(vec![2, 3]));
    let view = t.as_view();
    assert_eq!(view.shape().dims, vec![2, 3]);
    assert_eq!(view.dtype(), DType::F32);
    assert_eq!(view.numel(), 6);
    assert_eq!(view.data().len(), 6);
}

#[test]
fn tensor_mut() {
    let mut t: Tensor<f32> = Tensor::zeros(Shape::new(vec![2, 3]));
    {
        let mut m = t.as_mut();
        m.data_mut()[0] = 42.0;
    }
    assert_eq!(t.data()[0], 42.0);
}

#[test]
fn tensor_clone_arc() {
    let t: Tensor<f32> = Tensor::zeros(Shape::new(vec![2, 3]));
    let t2 = t.clone();
    assert_eq!(t, t2);
    assert_eq!(t.data().as_ptr(), t2.data().as_ptr());
}

#[test]
fn tensor_partial_eq() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0], Shape::new(vec![2]));
    let b = Tensor::from_vec(vec![1.0f32, 2.0], Shape::new(vec![2]));
    let c = Tensor::from_vec(vec![1.0f32, 3.0], Shape::new(vec![2]));
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn from_vec_with_spec_accepts_matching_dtype_shape_and_cpu_backend() {
    let spec = TensorSpec::new(Shape::new(vec![2, 2]), DType::F32);
    let t = Tensor::from_vec_with_spec(spec.clone(), vec![1.0f32, 2.0, 3.0, 4.0], Backend::Cpu)
        .unwrap();
    assert_eq!(t.spec(), &spec);
    assert_eq!(t.backend(), Backend::Cpu);
}

#[test]
fn from_vec_with_spec_rejects_dtype_mismatch() {
    let spec = TensorSpec::new(Shape::new(vec![2]), DType::F64);
    let err = Tensor::<f32>::from_vec_with_spec(spec, vec![1.0, 2.0], Backend::Cpu).unwrap_err();
    assert_eq!(
        err,
        TensorError::DTypeMismatch {
            expected: DType::F64,
            actual: DType::F32
        }
    );
}

#[test]
fn zeros_with_spec_uses_spec_numel_and_backend() {
    let spec = TensorSpec::new(Shape::new(vec![2, 3]), DType::F32);
    let t = Tensor::<f32>::zeros_with_spec(spec, Backend::Cpu).unwrap();
    assert_eq!(t.numel(), 6);
    assert_eq!(t.backend(), Backend::Cpu);
    assert_eq!(t.data(), &[0.0; 6]);
}

#[test]
fn as_view_reports_backend_and_spec() {
    let t: Tensor<f32> = Tensor::zeros(Shape::new(vec![2, 3]));
    let view = t.as_view();
    assert_eq!(view.backend(), Backend::Cpu);
    assert_eq!(view.spec, t.spec());
}

#[test]
fn as_view_mut_allows_mutating_data_slice() {
    let mut t: Tensor<f32> = Tensor::zeros(Shape::new(vec![2, 3]));
    {
        let mut view = t.as_view_mut();
        view.data_mut()[1] = 7.0;
    }
    assert_eq!(t.data()[1], 7.0);
}

struct Fill(f32);

impl BackendOp<f32> for Fill {
    fn dispatch(
        &self,
        _inputs: &[TensorView<'_, f32>],
        mut output: TensorViewMut<'_, f32>,
    ) -> Result<(), TensorError> {
        for value in output.data_mut() {
            *value = self.0;
        }
        Ok(())
    }
}

#[test]
fn tensor_backend_op_dispatch_writes_output() {
    let op = Fill(5.0);
    let mut output: Tensor<f32> = Tensor::zeros(Shape::new(vec![3]));
    op.dispatch(&[], output.as_view_mut()).unwrap();
    assert_eq!(output.data(), &[5.0, 5.0, 5.0]);
}

#[test]
fn tensor_from_vec_with_spec_alignment_check() {
    let shape = Shape::new(vec![4]);
    let mut spec = TensorSpec::new(shape.clone(), DType::F32);
    spec.layout = TensorLayout::contiguous(shape, 4);
    spec.layout.alignment_bytes = usize::MAX;
    let err = Tensor::<f32>::from_vec_with_spec(spec, vec![1.0, 2.0, 3.0, 4.0], Backend::Cpu)
        .unwrap_err();
    assert!(matches!(err, TensorError::UnalignedAllocation { .. }));
}
