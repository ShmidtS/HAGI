use config::hrm::HrmConfig;
use core_types::shape::Shape;
use data::{PackedBatch, PackedPartition, PackedSpan};
use hrm_model::{
    forward_hrm, forward_hrm_with_control, scheduled_bp_steps, HRMState, HrmBackbone, HrmError,
    HrmRuntimeControl, LmHead,
};
use tensor_runtime::Tensor;

fn test_config(h_cycles: usize, l_cycles: usize) -> HrmConfig {
    HrmConfig {
        total_layers: 2,
        h_layers: 1,
        l_layers: 1,
        hidden_size: 8,
        num_heads: 2,
        expansion: 2,
        h_cycles,
        l_cycles,
        vocab_size: 16,
        max_seq_len: 16,
        convergence_eps: 1e-5,
        bp_warmup_ratio: 0.2,
        bp_max_steps: 5,
        warmup_steps: 1000,
    }
}

fn packed_batch(batch_size: usize, seq_len: usize, prefix_len: usize) -> PackedBatch {
    let tokens: Vec<u32> = (0..batch_size * seq_len).map(|i| (i % 16) as u32).collect();
    let targets = tokens.clone();
    let mut prefix_mask = vec![0u8; batch_size * seq_len];
    let mut spans = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        for i in 0..prefix_len {
            prefix_mask[b * seq_len + i] = 1;
        }
        spans.push(PackedSpan {
            sequence_id: b,
            batch_index: b,
            start: 0,
            len: seq_len,
            prefix_len,
        });
    }

    let shape = Shape::new(vec![batch_size, seq_len]);
    PackedBatch {
        tokens: Tensor::from_vec(tokens, shape.clone()),
        targets: Tensor::from_vec(targets, shape.clone()),
        prefix_mask: Tensor::from_vec(prefix_mask, shape),
        partition: PackedPartition {
            spans,
            batch_size,
            seq_len,
        },
    }
}

fn state(batch_size: usize, seq_len: usize, hidden_size: usize, z_h_value: f32) -> HRMState {
    HRMState {
        z_h: Tensor::from_vec(
            vec![z_h_value; batch_size * seq_len * hidden_size],
            Shape::new(vec![batch_size, seq_len, hidden_size]),
        ),
        z_l: Tensor::zeros(Shape::new(vec![batch_size, seq_len, hidden_size])),
    }
}

fn embedded(batch: &PackedBatch, config: &HrmConfig) -> Tensor<f32> {
    LmHead::new(config.vocab_size, config.hidden_size)
        .embed_tokens(&batch.tokens)
        .unwrap()
}

#[test]
fn hrm_error_invalid_token_rank_formats() {
    let err = HrmError::InvalidTokenRank(3);
    assert_eq!(err.to_string(), "invalid token rank: expected 2, got 3");
}

#[test]
fn hrm_output_contains_compat_and_contract_cycle_fields() {
    let config = test_config(2, 3);
    let model = HrmBackbone::from_config(&config);
    let input = Tensor::zeros(Shape::new(vec![1, 2, config.hidden_size]));

    let output = model.forward(&input, &[1], 0);

    assert_eq!(output.detach_trace.bp_steps, output.bp_steps);
    assert_eq!(
        output.hidden.shape().dims,
        output.final_state.z_h.shape().dims
    );
    assert_eq!(
        output.final_state.z_h.shape().dims,
        output.final_state.z_l.shape().dims
    );
    let logits = LmHead::new(config.vocab_size, config.hidden_size).project(&output.hidden);
    assert_eq!(logits.shape().dims, vec![1, 2, config.vocab_size]);
}

#[test]
fn embed_tokens_maps_rank_2_tokens_to_hidden() {
    let head = LmHead::new(16, 8);
    let tokens = Tensor::from_vec(vec![1u32, 2, 17, 18, 3, 4], Shape::new(vec![2, 3]));

    let embedded = head.embed_tokens(&tokens).unwrap();

    assert_eq!(embedded.shape().dims, vec![2, 3, 8]);
    assert_eq!(embedded.data()[0], head.w_proj.data()[1]);
    assert_eq!(embedded.data()[8], head.w_proj.data()[2]);
    assert_eq!(embedded.data()[16], head.w_proj.data()[1]);
}

#[test]
fn embed_tokens_rejects_rank_3_tokens() {
    let head = LmHead::new(16, 8);
    let tokens = Tensor::from_vec(vec![0u32; 2 * 3 * 1], Shape::new(vec![2, 3, 1]));

    let err = head.embed_tokens(&tokens).unwrap_err();

    assert!(matches!(err, HrmError::InvalidTokenRank(3)));
}

fn assert_forward_shape(batch_size: usize, seq_len: usize) {
    let config = test_config(2, 3);
    let model = HrmBackbone::from_config(&config);
    let batch = packed_batch(batch_size, seq_len, 1);
    let state = state(batch_size, seq_len, config.hidden_size, 0.0);

    let output = forward_hrm(&model, &batch, embedded(&batch, &config), state, 0).unwrap();

    assert_eq!(
        output.hidden.shape().dims,
        vec![batch_size, seq_len, config.hidden_size]
    );
    assert_eq!(
        output.final_state.z_h.shape().dims,
        vec![batch_size, seq_len, config.hidden_size]
    );
    assert_eq!(
        output.final_state.z_l.shape().dims,
        vec![batch_size, seq_len, config.hidden_size]
    );
    assert_eq!(output.final_state.z_h.data(), output.final_state.z_l.data());
}

#[test]
fn forward_hrm_shape_preservation_1x1() {
    assert_forward_shape(1, 1);
}

#[test]
fn forward_hrm_shape_preservation_2x3() {
    assert_forward_shape(2, 3);
}

#[test]
fn forward_hrm_shape_preservation_4x4() {
    assert_forward_shape(4, 4);
}

#[test]
fn caller_projects_forward_hrm_hidden_to_logits() {
    let config = test_config(2, 3);
    let model = HrmBackbone::from_config(&config);
    let head = LmHead::new(config.vocab_size, config.hidden_size);
    let batch = packed_batch(2, 3, 1);
    let state = state(2, 3, config.hidden_size, 0.0);
    let input = head.embed_tokens(&batch.tokens).unwrap();

    let output = forward_hrm(&model, &batch, input, state, 0).unwrap();
    let logits = head.project(&output.hidden);

    assert_eq!(logits.shape().dims, vec![2, 3, config.vocab_size]);
}

#[test]
fn forward_hrm_mask_checksum_unchanged_for_same_partition() {
    let config = test_config(2, 3);
    let model = HrmBackbone::from_config(&config);
    let batch_a = packed_batch(2, 3, 1);
    let original_prefix_mask = batch_a.prefix_mask.data().to_vec();
    let expected_checksum = hrm_model::PrefixLmMask::build(2, 3, &[1, 1]).checksum();
    let mut batch_b = packed_batch(2, 3, 1);
    batch_b.tokens = Tensor::from_vec(vec![9u32; 2 * 3], Shape::new(vec![2, 3]));

    let out_a = forward_hrm(
        &model,
        &batch_a,
        embedded(&batch_a, &config),
        state(2, 3, config.hidden_size, 0.0),
        0,
    )
    .unwrap();
    let out_b = forward_hrm(
        &model,
        &batch_b,
        embedded(&batch_b, &config),
        state(2, 3, config.hidden_size, 0.0),
        0,
    )
    .unwrap();

    assert_eq!(batch_a.prefix_mask.data(), original_prefix_mask.as_slice());
    assert_eq!(out_a.mask_checksum, expected_checksum);
    assert_eq!(out_a.mask_checksum, out_b.mask_checksum);
}

#[test]
fn forward_hrm_mask_checksum_changes_when_prefix_changes() {
    let config = test_config(2, 3);
    let model = HrmBackbone::from_config(&config);
    let batch_a = packed_batch(1, 4, 1);
    let batch_b = packed_batch(1, 4, 2);

    let out_a = forward_hrm(
        &model,
        &batch_a,
        embedded(&batch_a, &config),
        state(1, 4, config.hidden_size, 0.0),
        0,
    )
    .unwrap();
    let out_b = forward_hrm(
        &model,
        &batch_b,
        embedded(&batch_b, &config),
        state(1, 4, config.hidden_size, 0.0),
        0,
    )
    .unwrap();

    assert_ne!(out_a.mask_checksum, out_b.mask_checksum);
}

#[test]
fn forward_hrm_uses_input_state_z_h() {
    let config = test_config(2, 3);
    let model = HrmBackbone::from_config(&config);
    let batch = packed_batch(1, 3, 1);

    let out_a = forward_hrm(
        &model,
        &batch,
        embedded(&batch, &config),
        state(1, 3, config.hidden_size, 0.0),
        0,
    )
    .unwrap();
    let out_b = forward_hrm(
        &model,
        &batch,
        embedded(&batch, &config),
        state(1, 3, config.hidden_size, 1.0),
        0,
    )
    .unwrap();

    assert_ne!(out_a.final_state.z_h.data(), out_b.final_state.z_h.data());
}

#[test]
fn forward_hrm_zero_weight_early_exit_limits_cycles() {
    let config = test_config(4, 4);
    let model = HrmBackbone::from_config(&config);
    let batch = packed_batch(1, 4, 2);

    let output = forward_hrm(
        &model,
        &batch,
        embedded(&batch, &config),
        state(1, 4, config.hidden_size, 0.0),
        0,
    )
    .unwrap();

    assert_eq!(output.effective_h_cycles, 1);
    assert_eq!(output.effective_l_cycles, 1);
}

#[test]
fn forward_hrm_detach_trace_counts_recurrence_steps() {
    let config = test_config(4, 4);
    let model = HrmBackbone::from_config(&config);
    let batch = packed_batch(1, 4, 2);

    let output = forward_hrm(
        &model,
        &batch,
        embedded(&batch, &config),
        state(1, 4, config.hidden_size, 0.0),
        0,
    )
    .unwrap();

    assert_eq!(
        output.detach_trace.total_recurrence_steps,
        output.effective_h_cycles + output.effective_l_cycles
    );
    assert_eq!(
        output.detach_trace.detached_steps,
        output
            .detach_trace
            .total_recurrence_steps
            .saturating_sub(output.detach_trace.bp_steps)
    );
    assert_eq!(
        output.detach_trace.traced_steps,
        output
            .detach_trace
            .total_recurrence_steps
            .min(output.detach_trace.bp_steps)
    );
}

#[test]
fn forward_hrm_scheduled_bp_steps_reflected_in_detach_trace() {
    let config = test_config(4, 4);
    let model = HrmBackbone::from_config(&config);
    let batch = packed_batch(1, 4, 2);

    let output = forward_hrm(
        &model,
        &batch,
        embedded(&batch, &config),
        state(1, 4, config.hidden_size, 0.0),
        1000,
    )
    .unwrap();

    assert_eq!(output.detach_trace.bp_steps, config.bp_max_steps);
    assert_eq!(output.bp_steps, config.bp_max_steps);
}

#[test]
fn forward_hrm_rejects_rank_3_tokens() {
    let config = test_config(2, 3);
    let model = HrmBackbone::from_config(&config);
    let mut batch = packed_batch(1, 3, 1);
    batch.tokens = Tensor::from_vec(vec![0u32; 3], Shape::new(vec![1, 3, 1]));
    let input = Tensor::zeros(Shape::new(vec![1, 3, config.hidden_size]));

    let err = forward_hrm(
        &model,
        &batch,
        input,
        state(1, 3, config.hidden_size, 0.0),
        0,
    )
    .unwrap_err();

    assert!(matches!(err, HrmError::InvalidTokenRank(3)));
}

#[test]
fn forward_hrm_rejects_state_shape_mismatch() {
    let config = test_config(2, 3);
    let model = HrmBackbone::from_config(&config);
    let batch = packed_batch(1, 3, 1);
    let bad_state = state(1, 2, config.hidden_size, 0.0);

    let err = forward_hrm(&model, &batch, embedded(&batch, &config), bad_state, 0).unwrap_err();

    assert!(matches!(err, HrmError::InvalidStateShape { .. }));
}

#[test]
fn forward_hrm_rejects_partition_shape_mismatch() {
    let config = test_config(2, 3);
    let model = HrmBackbone::from_config(&config);
    let mut batch = packed_batch(1, 3, 1);
    batch.partition.seq_len = 4;

    let err = forward_hrm(
        &model,
        &batch,
        embedded(&batch, &config),
        state(1, 3, config.hidden_size, 0.0),
        0,
    )
    .unwrap_err();

    assert!(matches!(err, HrmError::PartitionShapeMismatch));
}

#[test]
fn forward_hrm_with_control_uses_explicit_h_cycles() {
    let mut config = test_config(4, 2);
    config.convergence_eps = 0.0;
    let model = HrmBackbone::from_config(&config);
    let batch = packed_batch(1, 3, 1);
    let control = HrmRuntimeControl {
        h_cycles: 2,
        l_cycles: config.l_cycles,
        convergence_eps: config.convergence_eps,
        bp_steps: 0,
    };

    let output = forward_hrm_with_control(
        &model,
        &batch,
        embedded(&batch, &config),
        state(1, 3, config.hidden_size, 0.0),
        control,
    )
    .unwrap();

    assert_eq!(output.effective_h_cycles, 2);
}

#[test]
fn forward_hrm_with_control_uses_explicit_l_cycles() {
    let mut config = test_config(2, 4);
    config.convergence_eps = 0.0;
    let model = HrmBackbone::from_config(&config);
    let batch = packed_batch(1, 3, 1);
    let control = HrmRuntimeControl {
        h_cycles: 1,
        l_cycles: 2,
        convergence_eps: config.convergence_eps,
        bp_steps: 0,
    };

    let output = forward_hrm_with_control(
        &model,
        &batch,
        embedded(&batch, &config),
        state(1, 3, config.hidden_size, 0.0),
        control,
    )
    .unwrap();

    assert_eq!(output.effective_l_cycles, 2);
}

#[test]
fn forward_hrm_with_control_uses_explicit_convergence_eps() {
    let mut config = test_config(4, 4);
    config.convergence_eps = 0.0;
    let model = HrmBackbone::from_config(&config);
    let batch = packed_batch(1, 3, 1);
    let control = HrmRuntimeControl {
        h_cycles: config.h_cycles,
        l_cycles: config.l_cycles,
        convergence_eps: 1.0,
        bp_steps: 0,
    };

    let output = forward_hrm_with_control(
        &model,
        &batch,
        embedded(&batch, &config),
        state(1, 3, config.hidden_size, 0.0),
        control,
    )
    .unwrap();

    assert_eq!(output.effective_h_cycles, 1);
    assert_eq!(output.effective_l_cycles, 1);
}

#[test]
fn forward_hrm_with_control_uses_explicit_bp_steps() {
    let config = test_config(4, 4);
    let model = HrmBackbone::from_config(&config);
    let batch = packed_batch(1, 3, 1);
    let control = HrmRuntimeControl {
        h_cycles: config.h_cycles,
        l_cycles: config.l_cycles,
        convergence_eps: config.convergence_eps,
        bp_steps: 7,
    };

    let output = forward_hrm_with_control(
        &model,
        &batch,
        embedded(&batch, &config),
        state(1, 3, config.hidden_size, 0.0),
        control,
    )
    .unwrap();

    assert_eq!(output.bp_steps, 7);
    assert_eq!(output.detach_trace.bp_steps, 7);
}

#[test]
fn forward_hrm_delegates_to_control_version() {
    let config = test_config(2, 3);
    let model = HrmBackbone::from_config(&config);
    let batch = packed_batch(1, 3, 1);
    let step = 1000;
    let control = HrmRuntimeControl {
        h_cycles: config.h_cycles,
        l_cycles: config.l_cycles,
        convergence_eps: config.convergence_eps,
        bp_steps: scheduled_bp_steps(&config, step),
    };

    let delegated = forward_hrm(
        &model,
        &batch,
        embedded(&batch, &config),
        state(1, 3, config.hidden_size, 0.0),
        step,
    )
    .unwrap();
    let controlled = forward_hrm_with_control(
        &model,
        &batch,
        embedded(&batch, &config),
        state(1, 3, config.hidden_size, 0.0),
        control,
    )
    .unwrap();

    assert_eq!(delegated.hidden.data(), controlled.hidden.data());
    assert_eq!(
        delegated.final_state.z_h.data(),
        controlled.final_state.z_h.data()
    );
    assert_eq!(
        delegated.final_state.z_l.data(),
        controlled.final_state.z_l.data()
    );
    assert_eq!(delegated.effective_h_cycles, controlled.effective_h_cycles);
    assert_eq!(delegated.effective_l_cycles, controlled.effective_l_cycles);
    assert_eq!(delegated.mask_checksum, controlled.mask_checksum);
    assert_eq!(delegated.detach_trace, controlled.detach_trace);
    assert_eq!(delegated.bp_steps, controlled.bp_steps);
}

#[test]
fn recurrent_depth_monotonicity() {
    let mut config = test_config(3, 3);
    config.convergence_eps = 0.0;
    let model = HrmBackbone::from_config(&config);
    let batch = packed_batch(1, 3, 1);
    let mut last_depth = 0usize;
    let mut last_norm = 0.0f32;

    for (h_cycles, l_cycles) in [(1, 1), (2, 2), (3, 3)] {
        let control = HrmRuntimeControl {
            h_cycles,
            l_cycles,
            convergence_eps: 0.0,
            bp_steps: 0,
        };
        let output = forward_hrm_with_control(
            &model,
            &batch,
            embedded(&batch, &config),
            state(1, 3, config.hidden_size, 0.25),
            control,
        )
        .unwrap();
        let depth = h_cycles * l_cycles;
        let norm = output
            .hidden
            .data()
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();

        assert!(depth > last_depth);
        assert!(norm + 1e-6 >= last_norm);
        last_depth = depth;
        last_norm = norm;
    }
}
