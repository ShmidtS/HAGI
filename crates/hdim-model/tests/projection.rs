use core_types::shape::Shape;
use config::HdimConfig;
use hdim_model::{HdimError, HiddenToMultivector};
use hrm_model::HiddenState;
use tensor_runtime::Tensor;

#[test]
fn projection_output_shape() {
    let hidden_size = 16;
    let structural_heads = 4;
    let blade_count = 8;
    let batch = 2;
    let tokens = 4;

    let proj = HiddenToMultivector::new(hidden_size, structural_heads, blade_count);

    let input_data = vec![0.1f32; batch * tokens * hidden_size];
    let input_tensor = Tensor::from_vec(input_data, Shape::new(vec![batch, tokens, hidden_size]));
    let hidden = HiddenState::new(input_tensor);

    let output = proj.forward(&hidden);
    assert_eq!(
        output.shape().dims,
        vec![batch, tokens, structural_heads, blade_count],
        "projection output shape mismatch"
    );
}

#[test]
fn projection_with_zero_weights_gives_zero() {
    let hidden_size = 8;
    let structural_heads = 2;
    let blade_count = 8;
    let batch = 1;
    let tokens = 3;

    let w_proj = Tensor::from_vec(
        vec![0.0f32; hidden_size * structural_heads * blade_count],
        Shape::new(vec![hidden_size, structural_heads * blade_count]),
    );
    let proj =
        HiddenToMultivector::with_weights(hidden_size, structural_heads, blade_count, w_proj);

    let input_data = vec![1.0f32; batch * tokens * hidden_size];
    let input_tensor = Tensor::from_vec(input_data, Shape::new(vec![batch, tokens, hidden_size]));
    let hidden = HiddenState::new(input_tensor);

    let output = proj.forward(&hidden);
    assert!(
        output.data().iter().all(|&v| v.abs() < 1e-9),
        "zero weights should produce zero output"
    );
}

#[test]
fn projection_identity_like_weights() {
    // With hidden_size = structural_dim and identity-like weights,
    // projection should copy hidden state values through.
    let dim = 8;
    let structural_heads = 1;
    let blade_count = dim;
    let batch = 1;
    let tokens = 1;

    let mut w_data = vec![0.0f32; dim * dim];
    for i in 0..dim {
        w_data[i * dim + i] = 1.0;
    }
    let w_proj = Tensor::from_vec(w_data, Shape::new(vec![dim, dim]));
    let proj = HiddenToMultivector::with_weights(dim, structural_heads, blade_count, w_proj);

    let input_data: Vec<f32> = (0..dim).map(|i| (i + 1) as f32).collect();
    let input_tensor = Tensor::from_vec(input_data.clone(), Shape::new(vec![batch, tokens, dim]));
    let hidden = HiddenState::new(input_tensor);

    let output = proj.forward(&hidden);
    let out_data = output.data();
    for i in 0..dim {
        assert!(
            (out_data[i] - input_data[i]).abs() < 1e-6,
            "identity projection mismatch at index {}: expected {}, got {}",
            i,
            input_data[i],
            out_data[i]
        );
    }
}

#[test]
fn projection_different_batch_sizes() {
    let hidden_size = 4;
    let structural_heads = 2;
    let blade_count = 8;

    let proj = HiddenToMultivector::new(hidden_size, structural_heads, blade_count);

    for (batch, tokens) in [(1, 1), (1, 8), (4, 2), (3, 5)] {
        let input_data = vec![0.5f32; batch * tokens * hidden_size];
        let input_tensor =
            Tensor::from_vec(input_data, Shape::new(vec![batch, tokens, hidden_size]));
        let hidden = HiddenState::new(input_tensor);
        let output = proj.forward(&hidden);
        assert_eq!(
            output.shape().dims,
            vec![batch, tokens, structural_heads, blade_count],
            "shape mismatch for batch={}, tokens={}",
            batch,
            tokens
        );
    }
}

#[test]
fn try_projection_with_wrong_shape_returns_error() {
    let config = HdimConfig::default();
    let hidden_size = 8;
    let wrong = Tensor::from_vec(vec![0.0f32; hidden_size * 7], Shape::new(vec![hidden_size, 7]));

    let result = HiddenToMultivector::try_with_weights(&wrong, &config);

    assert!(matches!(result, Err(HdimError::InvalidConfig(_))));
}
