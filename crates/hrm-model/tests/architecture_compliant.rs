use config::hrm::HrmConfig;
use core_types::shape::Shape;
use hrm_model::{HState, HrmBackbone, LState, Linear};
use tensor_runtime::Tensor;

fn test_config() -> HrmConfig {
    HrmConfig {
        total_layers: 2,
        h_layers: 1,
        l_layers: 1,
        hidden_size: 8,
        num_heads: 2,
        expansion: 2,
        h_cycles: 2,
        l_cycles: 2,
        vocab_size: 16,
        max_seq_len: 8,
        convergence_eps: 1e-5,
        bp_warmup_ratio: 0.2,
        bp_max_steps: 5,
        warmup_steps: 1000,
    }
}

#[test]
fn compact_states_keep_creation_shapes() {
    let h_state = HState {
        data: Tensor::zeros(Shape::new(vec![2, 5])),
    };
    let l_state = LState {
        data: Tensor::zeros(Shape::new(vec![2, 3])),
    };

    assert_eq!(h_state.data.shape().dims, vec![2, 5]);
    assert_eq!(l_state.data.shape().dims, vec![2, 3]);
}

#[test]
fn architecture_forward_preserves_compact_state_shapes() {
    let config = test_config();
    let mut model = HrmBackbone::from_config(&config);
    let batch = 2;
    let seq_len = 4;
    let h_dim = 5;
    let l_dim = 3;
    let tokens = Tensor::from_vec(
        vec![0i64, 1, 2, 3, 4, 5, 6, 7],
        Shape::new(vec![batch, seq_len]),
    );
    let prefix_mask = Tensor::from_vec(
        vec![1.0f32, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        Shape::new(vec![batch, seq_len]),
    );
    let mut z_h = HState {
        data: Tensor::zeros(Shape::new(vec![batch, h_dim])),
    };
    let mut z_l = LState {
        data: Tensor::zeros(Shape::new(vec![batch, l_dim])),
    };
    let h_proj = Linear::new_sequence(h_dim, seq_len, config.hidden_size);
    let l_proj = Linear::new_sequence(l_dim, seq_len, config.hidden_size);
    let h_pool = Linear::new(config.hidden_size, h_dim);
    let l_pool = Linear::new(config.hidden_size, l_dim);

    model
        .forward_architecture_compliant(
            &tokens,
            &mut z_h,
            &mut z_l,
            &prefix_mask,
            &h_proj,
            &l_proj,
            &h_pool,
            &l_pool,
            2,
            2,
        )
        .unwrap();

    assert_eq!(z_h.data.shape().dims, vec![batch, h_dim]);
    assert_eq!(z_l.data.shape().dims, vec![batch, l_dim]);
}

#[test]
fn architecture_forward_returns_logits_shape() {
    let config = test_config();
    let mut model = HrmBackbone::from_config(&config);
    let batch = 2;
    let seq_len = 4;
    let h_dim = 5;
    let l_dim = 3;
    let tokens = Tensor::from_vec(
        vec![0i64, 1, 2, 3, 4, 5, 6, 7],
        Shape::new(vec![batch, seq_len]),
    );
    let prefix_mask = Tensor::from_vec(
        vec![1.0f32; batch * seq_len],
        Shape::new(vec![batch, seq_len]),
    );
    let mut z_h = HState {
        data: Tensor::zeros(Shape::new(vec![batch, h_dim])),
    };
    let mut z_l = LState {
        data: Tensor::zeros(Shape::new(vec![batch, l_dim])),
    };
    let h_proj = Linear::new_sequence(h_dim, seq_len, config.hidden_size);
    let l_proj = Linear::new_sequence(l_dim, seq_len, config.hidden_size);
    let h_pool = Linear::new(config.hidden_size, h_dim);
    let l_pool = Linear::new(config.hidden_size, l_dim);

    let logits = model
        .forward_architecture_compliant(
            &tokens,
            &mut z_h,
            &mut z_l,
            &prefix_mask,
            &h_proj,
            &l_proj,
            &h_pool,
            &l_pool,
            2,
            2,
        )
        .unwrap();

    assert_eq!(logits.shape().dims, vec![batch, seq_len, config.vocab_size]);
}
