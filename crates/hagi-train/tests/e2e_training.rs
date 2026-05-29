use std::path::{Path, PathBuf};

use config::{HdimConfig, HrmConfig, MsaConfig};
use core_types::shape::Shape;
use data::{PackedBatch, PackedPartition, PackedSpan};
use hagi_train::{AdamW, TrainingLoop};
use losses::LossWeights;
use tensor_runtime::Tensor;

fn tiny_hrm_config() -> HrmConfig {
    HrmConfig {
        total_layers: 1,
        h_layers: 0,
        l_layers: 1,
        hidden_size: 16,
        num_heads: 2,
        expansion: 2,
        h_cycles: 1,
        l_cycles: 1,
        vocab_size: 64,
        max_seq_len: 32,
        convergence_eps: 1e-5,
        bp_warmup_ratio: 0.2,
        bp_max_steps: 1,
        warmup_steps: 1,
    }
}

fn tiny_hdim_config() -> HdimConfig {
    HdimConfig {
        structural_heads: 2,
        blade_count_per_head: 8,
        ..HdimConfig::default()
    }
}

fn make_batches() -> Vec<PackedBatch> {
    (0..3)
        .map(|sequence_id| PackedBatch {
            tokens: Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], Shape::new(vec![1, 6])),
            targets: Tensor::from_vec(vec![2, 3, 4, 5, 6, 6], Shape::new(vec![1, 6])),
            prefix_mask: Tensor::from_vec(vec![1, 1, 1, 0, 0, 0], Shape::new(vec![1, 6])),
            partition: PackedPartition {
                spans: vec![PackedSpan {
                    sequence_id,
                    batch_index: 0,
                    start: 0,
                    len: 6,
                    prefix_len: 3,
                }],
                batch_size: 1,
                seq_len: 6,
            },
        })
        .collect()
}

#[test]
fn e2e_training_run_saves_checkpoint() {
    let hrm = tiny_hrm_config();
    let hdim = tiny_hdim_config();
    let msa = MsaConfig::try_new(1).unwrap();
    hrm.validate().unwrap();
    hdim.validate_hidden_size(hrm.hidden_size).unwrap();
    msa.validate().unwrap();

    let mut trainer = TrainingLoop::try_new_with_msa_config(
        hrm_model::HrmBackbone::from_config(&hrm),
        AdamW::new(0.001, 0.9, 0.95, 1e-8, 0.0),
        LossWeights {
            lambda_aux: 0.0,
            lambda_iso_target: 0.0,
            iso_warmup_steps: 1,
        },
        msa,
    )
    .unwrap();

    for batch in make_batches() {
        let report = trainer.train_step(&batch).unwrap();
        assert!(report.loss.l_total.is_finite());
        assert!(report.loss.l_ce.is_finite());
        assert!(report.loss.l_ce > 0.0);
    }

    let checkpoint_path = PathBuf::from(format!(
        "tests/e2e_training_checkpoint_{}.bin",
        std::process::id()
    ));
    let checkpoint_file = Path::new("checkpoints").join(&checkpoint_path);
    let _ = std::fs::remove_file(&checkpoint_file);

    trainer
        .save_checkpoint(&checkpoint_path, trainer.step as u64)
        .unwrap();

    assert!(checkpoint_file.is_file());

    let _ = std::fs::remove_file(checkpoint_file);
}
