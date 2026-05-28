use config::HdimConfig;
use core_types::shape::Shape;
use hdim_model::{HdimError, StructuralFusion};
use tensor_runtime::Tensor;

#[test]
fn fusion_output_shape_matches_h_state() {
    let hidden_size = 16;
    let structural_dim = 32; // e.g. 4 heads * 8 blades
    let batch = 2;
    let tokens = 4;

    let fusion = StructuralFusion::new(hidden_size, structural_dim);

    let h_data = vec![0.5f32; batch * tokens * hidden_size];
    let h_state = Tensor::from_vec(h_data, Shape::new(vec![batch, tokens, hidden_size]));

    let s_data = vec![0.1f32; batch * tokens * structural_dim];
    let structural = Tensor::from_vec(s_data, Shape::new(vec![batch, tokens, structural_dim]));

    let output = fusion.forward(&h_state, &structural);
    assert_eq!(
        output.shape().dims,
        vec![batch, tokens, hidden_size],
        "fusion output shape must match h_state shape"
    );
}

#[test]
fn fusion_zero_structural_zero_gate_returns_h_state() {
    let hidden_size = 8;
    let structural_dim = 16;
    let batch = 1;
    let tokens = 2;

    // Zero W_gate and zero W_fuse -> gate = sigmoid(0) = 0.5, fuse_proj = 0
    // so output = h + 0.5 * 0 = h
    let w_gate = Tensor::from_vec(
        vec![0.0f32; (hidden_size + structural_dim) * hidden_size],
        Shape::new(vec![hidden_size + structural_dim, hidden_size]),
    );
    let w_fuse = Tensor::from_vec(
        vec![0.0f32; structural_dim * hidden_size],
        Shape::new(vec![structural_dim, hidden_size]),
    );
    let fusion = StructuralFusion::with_weights(hidden_size, structural_dim, w_gate, w_fuse);

    let h_data: Vec<f32> = (0..batch * tokens * hidden_size)
        .map(|i| (i as f32) * 0.1 + 0.5)
        .collect();
    let h_state = Tensor::from_vec(h_data.clone(), Shape::new(vec![batch, tokens, hidden_size]));

    let s_data = vec![0.0f32; batch * tokens * structural_dim];
    let structural = Tensor::from_vec(s_data, Shape::new(vec![batch, tokens, structural_dim]));

    let output = fusion.forward(&h_state, &structural);
    let out_data = output.data();

    for i in 0..h_data.len() {
        assert!(
            (out_data[i] - h_data[i]).abs() < 1e-6,
            "zero weights + zero structural should return h_state at index {}: expected {}, got {}",
            i,
            h_data[i],
            out_data[i]
        );
    }
}

#[test]
fn fusion_with_rank4_structural() {
    // Structural input can be [B, T, heads, blades] (rank 4)
    let hidden_size = 8;
    let heads = 2;
    let blades = 8;
    let structural_dim = heads * blades;
    let batch = 1;
    let tokens = 3;

    let fusion = StructuralFusion::new(hidden_size, structural_dim);

    let h_data = vec![1.0f32; batch * tokens * hidden_size];
    let h_state = Tensor::from_vec(h_data, Shape::new(vec![batch, tokens, hidden_size]));

    let s_data = vec![0.5f32; batch * tokens * heads * blades];
    let structural = Tensor::from_vec(s_data, Shape::new(vec![batch, tokens, heads, blades]));

    let output = fusion.forward(&h_state, &structural);
    assert_eq!(
        output.shape().dims,
        vec![batch, tokens, hidden_size],
        "rank-4 structural input should produce [B, T, hidden] output"
    );
}

#[test]
fn fusion_gate_modulates_output() {
    // With non-zero W_fuse and non-zero structural, output should differ from h_state.
    let hidden_size = 4;
    let structural_dim = 8;
    let batch = 1;
    let tokens = 1;

    // Use ones for W_fuse so structural projects to non-zero.
    let w_gate = Tensor::from_vec(
        vec![0.0f32; (hidden_size + structural_dim) * hidden_size],
        Shape::new(vec![hidden_size + structural_dim, hidden_size]),
    );
    // W_fuse: all ones -> fuse_proj[j] = sum of structural values
    let w_fuse = Tensor::from_vec(
        vec![1.0f32; structural_dim * hidden_size],
        Shape::new(vec![structural_dim, hidden_size]),
    );
    let fusion = StructuralFusion::with_weights(hidden_size, structural_dim, w_gate, w_fuse);

    let h_data = vec![0.0f32; batch * tokens * hidden_size];
    let h_state = Tensor::from_vec(h_data, Shape::new(vec![batch, tokens, hidden_size]));

    let s_data = vec![1.0f32; batch * tokens * structural_dim];
    let structural = Tensor::from_vec(s_data, Shape::new(vec![batch, tokens, structural_dim]));

    let output = fusion.forward(&h_state, &structural);
    let out_data = output.data();

    // gate = sigmoid(0) = 0.5, fuse_proj[j] = sum(1.0 * 8) = 8.0
    // out[j] = 0.0 + 0.5 * 8.0 = 4.0
    let expected = 0.5 * structural_dim as f32;
    for j in 0..hidden_size {
        assert!(
            (out_data[j] - expected).abs() < 1e-5,
            "fusion gate modulation mismatch at index {}: expected {}, got {}",
            j,
            expected,
            out_data[j]
        );
    }
}

#[test]
fn try_fusion_with_wrong_shape_returns_error() {
    let config = HdimConfig::default();
    let hidden_size = 8;
    let structural_dim = config.structural_heads * config.blade_count_per_head;
    let w_gate = Tensor::from_vec(
        vec![0.0f32; (hidden_size + structural_dim) * hidden_size],
        Shape::new(vec![hidden_size + structural_dim, hidden_size]),
    );
    let wrong_fuse = Tensor::from_vec(
        vec![0.0f32; (structural_dim - 1) * hidden_size],
        Shape::new(vec![structural_dim - 1, hidden_size]),
    );

    let result = StructuralFusion::try_with_weights(&w_gate, &wrong_fuse, &config);

    assert!(matches!(result, Err(HdimError::InvalidConfig(_))));
}
