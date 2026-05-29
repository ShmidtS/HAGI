use clifford_core::Cl3;
use core_types::shape::Shape;
use hdim_model::{project_hidden_to_multivector, HiddenToMultivector, MultivectorBatch};
use hrm_model::HiddenState;
use tensor_runtime::Tensor;

#[test]
fn projector_forward_returns_multivector_shape() {
    let weights = Tensor::from_vec(vec![1.0f32; 2 * 16], Shape::new(vec![2, 16]));
    let projector = HiddenToMultivector::with_weights(2, 2, 8, weights);
    let hidden = HiddenState::new(Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0],
        Shape::new(vec![1, 2, 2]),
    ));

    let output = projector.forward(&hidden);

    assert_eq!(output.shape(), &Shape::new(vec![1, 2, 2, 8]));
    assert_eq!(output.numel(), 32);
}

#[test]
fn multivector_batch_extracts_expected_head_coefficients() {
    let coeffs = Tensor::from_vec(
        (0..16).map(|value| value as f32).collect(),
        Shape::new(vec![1, 1, 2, 8]),
    );
    let batch = MultivectorBatch::<Cl3>::new(coeffs, 2);

    let mv = batch.multivector_at(0, 0, 1);

    assert_eq!(batch.shape(), &Shape::new(vec![1, 1, 2, 8]));
    assert_eq!(
        mv.coeffs,
        vec![8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    );
}

#[test]
fn project_hidden_to_multivector_wraps_projected_tensor() {
    let weights = Tensor::from_vec(vec![1.0f32; 2 * 8], Shape::new(vec![2, 8]));
    let projector = HiddenToMultivector::with_weights(2, 1, 8, weights);
    let hidden = HiddenState::new(Tensor::from_vec(
        vec![1.0f32, 2.0],
        Shape::new(vec![1, 1, 2]),
    ));

    let batch = project_hidden_to_multivector::<Cl3>(&projector, &hidden).unwrap();

    assert_eq!(batch.shape(), &Shape::new(vec![1, 1, 1, 8]));
    assert_eq!(batch.multivector_at(0, 0, 0).coeffs, vec![3.0; 8]);
}
