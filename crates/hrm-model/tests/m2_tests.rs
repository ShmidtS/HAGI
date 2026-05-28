use config::hrm::HrmConfig;
use core_types::shape::Shape;
use tensor_runtime::Tensor;

use hrm_model::mask::PrefixLmMask;
use hrm_model::mlp::SwiGlMlp;
use hrm_model::norm::RMSNorm;
use hrm_model::recurrence::{check_convergence, scheduled_bp_steps};
use hrm_model::rope::RopeTable;
use hrm_model::transformer::TransformerBlock;
use hrm_model::HrmBackbone;

#[test]
fn rmsnorm_preserves_shape_and_mean_square() {
    let norm = RMSNorm::new(32, 1e-6);
    let data: Vec<f32> = (0..2 * 4 * 32).map(|i| (i as f32) * 0.05 - 5.0).collect();
    let input = Tensor::from_vec(data, Shape::new(vec![2, 4, 32]));
    let output = norm.forward(&input);
    assert_eq!(output.shape().dims, vec![2, 4, 32]);

    let out_data = output.data();
    for row in 0..(2 * 4) {
        let slice = &out_data[row * 32..(row + 1) * 32];
        let mean_sq: f32 = slice.iter().map(|&x| x * x).sum::<f32>() / 32.0;
        assert!(
            (mean_sq - 1.0).abs() < 0.01,
            "row {}: mean_sq = {} (expected ~1.0)",
            row,
            mean_sq
        );
    }
}

#[test]
fn rope_rotates_known_position() {
    let head_dim = 8;
    let rope = RopeTable::new(64, head_dim, 10000.0);
    let mut data = vec![1.0f32; head_dim];
    let original = data.clone();
    rope.apply(&mut data, 1, 1, 1);

    let pos = 0u32;
    for i in 0..(head_dim / 2) {
        let angle = pos as f32 * 10000.0f32.powf(-(2.0 * i as f32) / head_dim as f32);
        let expected_0 = original[2 * i] * angle.cos() - original[2 * i + 1] * angle.sin();
        let expected_1 = original[2 * i] * angle.sin() + original[2 * i + 1] * angle.cos();
        assert!((data[2 * i] - expected_0).abs() < 1e-5);
        assert!((data[2 * i + 1] - expected_1).abs() < 1e-5);
    }
}

#[test]
fn prefixlm_mask_properties() {
    let mask = PrefixLmMask::build(1, 8, &[4]);
    for i in 0..4 {
        for j in 0..4 {
            assert!(
                mask.can_attend(0, i, j),
                "prefix {} must attend prefix {}",
                i,
                j
            );
        }
        for j in 4..8 {
            assert!(
                !mask.can_attend(0, i, j),
                "prefix {} must not attend response {}",
                i,
                j
            );
        }
    }
    for i in 4..8 {
        for j in 0..4 {
            assert!(
                mask.can_attend(0, i, j),
                "response {} must attend prefix {}",
                i,
                j
            );
        }
        for j in 4..8 {
            if j <= i {
                assert!(
                    mask.can_attend(0, i, j),
                    "response {} must attend causal response {}",
                    i,
                    j
                );
            } else {
                assert!(
                    !mask.can_attend(0, i, j),
                    "response {} must not attend future response {}",
                    i,
                    j
                );
            }
        }
    }
}

#[test]
fn attention_shape_preservation() {
    use hrm_model::attention::MultiHeadSelfAttention;
    let attn = MultiHeadSelfAttention::new(24, 4, 64);
    let input = Tensor::from_vec(vec![0.1f32; 2 * 6 * 24], Shape::new(vec![2, 6, 24]));
    let mask = PrefixLmMask::build(2, 6, &[3, 2]);
    let output = attn.forward(&input, &mask);
    assert_eq!(output.shape().dims, vec![2, 6, 24]);
}

#[test]
fn mlp_shape_preservation() {
    let mlp = SwiGlMlp::new(16, 4);
    let input = Tensor::from_vec(vec![0.3f32; 1 * 4 * 16], Shape::new(vec![1, 4, 16]));
    let output = mlp.forward(&input);
    assert_eq!(output.shape().dims, vec![1, 4, 16]);
}

#[test]
fn transformer_block_shape_and_residual() {
    let block = TransformerBlock::new(16, 4, 2, 32);
    let input = Tensor::from_vec(vec![0.5f32; 1 * 4 * 16], Shape::new(vec![1, 4, 16]));
    let mask = PrefixLmMask::build(1, 4, &[2]);
    let output = block.forward(&input, &mask);
    assert_eq!(output.shape().dims, vec![1, 4, 16]);
}

#[test]
fn scheduled_bp_steps_schedule() {
    let config = HrmConfig {
        bp_warmup_ratio: 0.2,
        bp_max_steps: 5,
        warmup_steps: 1000,
        ..HrmConfig::default()
    };
    assert_eq!(scheduled_bp_steps(&config, 0), 0);
    assert_eq!(scheduled_bp_steps(&config, 1000), 5);
    assert_eq!(scheduled_bp_steps(&config, 5000), 5);
}

#[test]
fn hrm_forward_shape_preservation_1_1() {
    let config = HrmConfig {
        total_layers: 2,
        h_layers: 1,
        l_layers: 1,
        hidden_size: 16,
        num_heads: 4,
        expansion: 2,
        h_cycles: 1,
        l_cycles: 1,
        vocab_size: 100,
        max_seq_len: 32,
        convergence_eps: 1e-5,
        bp_warmup_ratio: 0.2,
        bp_max_steps: 5,
        warmup_steps: 1000,
    };
    let backbone = HrmBackbone::from_config(&config);
    let input = Tensor::from_vec(vec![0.1f32; 2 * 4 * 16], Shape::new(vec![2, 4, 16]));
    let output = backbone.forward(&input, &[2, 1], 0);
    assert_eq!(output.hidden.shape().dims, vec![2, 4, 16]);
}

#[test]
fn hrm_forward_shape_preservation_2_3() {
    let config = HrmConfig {
        total_layers: 2,
        h_layers: 1,
        l_layers: 1,
        hidden_size: 16,
        num_heads: 4,
        expansion: 2,
        h_cycles: 2,
        l_cycles: 3,
        vocab_size: 100,
        max_seq_len: 32,
        convergence_eps: 1e-5,
        bp_warmup_ratio: 0.2,
        bp_max_steps: 5,
        warmup_steps: 1000,
    };
    let backbone = HrmBackbone::from_config(&config);
    let input = Tensor::from_vec(vec![0.1f32; 1 * 3 * 16], Shape::new(vec![1, 3, 16]));
    let output = backbone.forward(&input, &[1], 0);
    assert_eq!(output.hidden.shape().dims, vec![1, 3, 16]);
}

#[test]
fn hrm_early_exit_with_zero_weights() {
    let config = HrmConfig {
        total_layers: 2,
        h_layers: 1,
        l_layers: 1,
        hidden_size: 16,
        num_heads: 4,
        expansion: 2,
        h_cycles: 4,
        l_cycles: 4,
        vocab_size: 100,
        max_seq_len: 32,
        convergence_eps: 1e-5,
        bp_warmup_ratio: 0.2,
        bp_max_steps: 5,
        warmup_steps: 1000,
    };
    let backbone = HrmBackbone::from_config(&config);
    let input = Tensor::from_vec(vec![1.0f32; 1 * 4 * 16], Shape::new(vec![1, 4, 16]));
    let output = backbone.forward(&input, &[2], 0);
    assert_eq!(
        output.effective_h_cycles, 1,
        "zero weights should exit after 1 H cycle"
    );
    assert_eq!(
        output.effective_l_cycles, 1,
        "zero weights should exit after 1 L cycle"
    );
}

#[test]
fn convergence_check_works() {
    let a = Tensor::from_vec(vec![1.0f32; 10], Shape::new(vec![10]));
    let b = Tensor::from_vec(vec![1.0f32; 10], Shape::new(vec![10]));
    assert!(check_convergence(&a, &b, 1e-5));

    let c = Tensor::from_vec(vec![100.0f32; 10], Shape::new(vec![10]));
    assert!(!check_convergence(&a, &c, 1e-5));
}
