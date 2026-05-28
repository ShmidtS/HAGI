use config::HrmConfig;
use core_types::shape::Shape;
use data::{PackedBatch, PackedPartition, PackedSpan};
use hagi_train::{
    adamw_step, load_checkpoint, train_step, AdamW, AdamWConfig, AdamWState, AsyncCheckpointWriter,
    NarsHrmConfig, NarsTrainingConfig, OptimizerError, TrainingLoop,
};
use losses::LossWeights;
use msa_adapter::MemorySlot;
use tensor_runtime::Tensor;

fn packed_batch() -> PackedBatch {
    PackedBatch {
        tokens: Tensor::from_vec(vec![0, 1, 2, 3], Shape::new(vec![1, 4])),
        targets: Tensor::from_vec(vec![0, 1, 2, 3], Shape::new(vec![1, 4])),
        prefix_mask: Tensor::from_vec(vec![1, 1, 0, 0], Shape::new(vec![1, 4])),
        partition: PackedPartition {
            spans: vec![PackedSpan {
                sequence_id: 0,
                batch_index: 0,
                start: 0,
                len: 4,
                prefix_len: 2,
            }],
            batch_size: 1,
            seq_len: 4,
        },
    }
}

fn trainer() -> TrainingLoop {
    let hrm = HrmConfig {
        total_layers: 2,
        h_layers: 1,
        l_layers: 1,
        hidden_size: 8,
        num_heads: 2,
        expansion: 2,
        h_cycles: 1,
        l_cycles: 1,
        vocab_size: 8,
        max_seq_len: 8,
        convergence_eps: 1e-5,
        bp_warmup_ratio: 0.2,
        bp_max_steps: 2,
        warmup_steps: 10,
    };
    TrainingLoop::new(
        hrm_model::HrmBackbone::from_config(&hrm),
        AdamW::new(0.01, 0.9, 0.95, 1e-8, 0.0),
        LossWeights {
            lambda_aux: 0.0,
            lambda_iso_target: 0.0,
            iso_warmup_steps: 10,
        },
    )
}

fn memory_slot(id: usize, values: Vec<f32>) -> MemorySlot {
    let dim = values.len();
    MemorySlot::new(
        id,
        Tensor::from_vec(values.clone(), Shape::new(vec![dim])),
        Tensor::from_vec(values, Shape::new(vec![dim])),
        0,
        "test".into(),
    )
}

#[test]
fn adamw_hand_computed_step_against_reference() {
    let mut params = vec![Tensor::from_vec(vec![1.0f32, 2.0], Shape::new(vec![2]))];
    let grads = vec![Tensor::from_vec(vec![0.5f32, -0.5], Shape::new(vec![2]))];
    let mut opt = AdamW::new(0.1, 0.9, 0.95, 1e-8, 0.0);

    opt.step(&mut params, &grads);

    assert!((params[0].data()[0] - 0.9).abs() < 1e-5);
    assert!((params[0].data()[1] - 2.1).abs() < 1e-5);
}

#[test]
fn adamw_free_function_matches_hand_computed_step() {
    let mut params = vec![Tensor::from_vec(vec![1.0f32, 2.0], Shape::new(vec![2]))];
    let grads = vec![Tensor::from_vec(vec![0.5f32, -0.5], Shape::new(vec![2]))];
    let mut state = AdamWState::default();
    let config = AdamWConfig {
        lr: 0.1,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
        weight_decay: 0.0,
        max_norm: 1.0,
    };

    adamw_step(&mut params, &grads, &mut state, config).unwrap();

    assert!((params[0].data()[0] - 0.9).abs() < 1e-5);
    assert!((params[0].data()[1] - 2.1).abs() < 1e-5);
    assert_eq!(state.current_step(), 1);
}

#[test]
fn adamw_free_function_clips_by_configured_max_norm() {
    let mut params = vec![Tensor::from_vec(vec![1.0f32], Shape::new(vec![1]))];
    let grads = vec![Tensor::from_vec(vec![10.0f32], Shape::new(vec![1]))];
    let mut state = AdamWState::default();
    let config = AdamWConfig {
        lr: 0.1,
        weight_decay: 0.0,
        max_norm: 0.5,
        ..AdamWConfig::default()
    };

    adamw_step(&mut params, &grads, &mut state, config).unwrap();

    assert!((params[0].data()[0] - 0.9).abs() < 1e-5);
}

#[test]
fn adamw_free_function_rejects_invalid_config() {
    let mut params = vec![Tensor::from_vec(vec![1.0f32], Shape::new(vec![1]))];
    let original = params[0].data()[0];
    let grads = vec![Tensor::from_vec(vec![1.0f32], Shape::new(vec![1]))];
    let mut state = AdamWState::default();
    let config = AdamWConfig {
        lr: f32::NAN,
        ..AdamWConfig::default()
    };

    let err = adamw_step(&mut params, &grads, &mut state, config).unwrap_err();

    assert!(matches!(err, OptimizerError::InvalidConfig(_)));
    assert_eq!(params[0].data()[0], original);
    assert_eq!(state.current_step(), 0);
}

#[test]
fn clipped_global_norm_is_at_most_one() {
    let mut grads = vec![
        Tensor::from_vec(vec![3.0f32], Shape::new(vec![1])),
        Tensor::from_vec(vec![4.0f32], Shape::new(vec![1])),
    ];
    let opt = AdamW::default();

    opt.clip_gradients(&mut grads);

    let norm = grads
        .iter()
        .flat_map(|grad| grad.data().iter())
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!(norm <= 1.0 + 1e-6, "norm={norm}");
}

#[test]
fn hundred_step_loss_decreases_on_synthetic_batch() {
    let mut trainer = trainer();
    let batch = packed_batch();
    let first = train_step(&mut trainer, &batch).unwrap().loss.l_total;
    let mut last = first;

    for _ in 1..100 {
        last = train_step(&mut trainer, &batch).unwrap().loss.l_total;
    }

    assert!(last < first, "first={first}, last={last}");
}

#[test]
fn nars_enabled_training_loop_runs_one_step() {
    let mut config = NarsTrainingConfig::default();
    config.enabled = true;
    let mut trainer = trainer().with_nars(config);
    let batch = packed_batch();

    let report = train_step(&mut trainer, &batch).unwrap();

    assert_eq!(report.step, 0);
    assert!(trainer.nars_hdim_reasoner.is_some());
    assert!(trainer.nars_msa_reasoner.is_some());
}

#[test]
fn with_nars_enabled_hrm_controller_instantiates_controller() {
    let config = NarsTrainingConfig {
        enabled: true,
        hrm_controller: NarsHrmConfig {
            enabled: true,
            ..NarsHrmConfig::default()
        },
        ..NarsTrainingConfig::default()
    };
    let trainer = trainer().with_nars(config);

    assert!(trainer.nars_hrm_controller.is_some());
}

#[test]
fn forward_msa_returns_attention_matching_hidden_shape() {
    let mut trainer = trainer();
    trainer.register_msa_slot(memory_slot(0, vec![1.0; 8]));
    let hidden = Tensor::from_vec(vec![1.0; 32], Shape::new(vec![1, 4, 8]));

    let attention = trainer.forward_msa(&hidden).unwrap();

    assert_eq!(attention.shape(), hidden.shape());
}

#[test]
fn nars_config_default_build_inspect_fields() {
    let mut config = NarsTrainingConfig::default();
    assert!(!config.enabled);
    assert!(config.hrm_controller.enabled);
    assert_eq!(config.hdim_reasoner.recommendation_threshold, 0.5);
    assert_eq!(config.msa_reasoner.default_feedback_confidence, 0.9);

    config.enabled = true;
    let trainer = trainer().with_nars(config.clone());

    assert!(trainer.nars_config.enabled);
    assert_eq!(trainer.nars_config, config);
}

#[test]
fn async_checkpoint_roundtrip() {
    let trainer = trainer();
    let path = std::path::PathBuf::from("tests/hagi_async_checkpoint_roundtrip.bin");
    let _ = std::fs::remove_file(std::path::Path::new("checkpoints").join(&path));

    let writer = AsyncCheckpointWriter::new();
    writer
        .save_snapshot(&path, 7, &trainer.named_tensors())
        .unwrap();
    writer.finish().unwrap();

    let (meta, tensors) = load_checkpoint(&path).unwrap();
    assert_eq!(meta.step, 7);
    assert_eq!(tensors.len(), 4);
    assert_eq!(tensors[3].data(), trainer.lm_head.w_proj.data());

    let _ = std::fs::remove_file(std::path::Path::new("checkpoints").join(&path));
}
