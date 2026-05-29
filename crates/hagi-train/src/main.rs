use std::path::PathBuf;

use config::MsaConfig;
use data::{MultipackScheduler, PrefixLmPacker, ToyDataset};
use hagi_train::{AdamW, AsyncCheckpointWriter, TrainingLoop};
use hrm_model::HrmBackbone;
use losses::LossWeights;

const TRAIN_STEPS: usize = 10;
const MAX_SEQ_LEN: usize = 32;

fn main() {
    if let Err(err) = run() {
        eprintln!("training failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let model_config = config::demo_model_config();
    let mut hrm_config = model_config.hrm;
    hrm_config.expansion = 2;
    hrm_config.h_cycles = 1;
    hrm_config.l_cycles = 1;
    hrm_config.max_seq_len = MAX_SEQ_LEN;
    hrm_config.bp_max_steps = 2;
    hrm_config.warmup_steps = TRAIN_STEPS;
    hrm_config.validate()?;

    let msa_config = MsaConfig::try_new(model_config.msa.top_k)?;
    let mut trainer = TrainingLoop::try_new_with_msa_config(
        HrmBackbone::from_config(&hrm_config),
        AdamW::new(0.001, 0.9, 0.95, 1e-8, 0.01),
        LossWeights {
            lambda_aux: 0.0,
            lambda_iso_target: 0.0,
            iso_warmup_steps: TRAIN_STEPS,
        },
        msa_config,
    )?;

    let mut dataset = ToyDataset::new(hrm_config.vocab_size, TRAIN_STEPS, MAX_SEQ_LEN, 42);
    let packer = PrefixLmPacker::new(0.5)?;
    let scheduler = MultipackScheduler::new(MAX_SEQ_LEN, 0)?;

    for step in 0..TRAIN_STEPS {
        let mut tokens = dataset.next_example();
        tokens.resize(MAX_SEQ_LEN, 0);
        let example = packer.pack(step, &tokens)?;
        let batch = scheduler.schedule(&[example])?;
        let report = trainer.train_step(&batch)?;
        println!("Step {}: loss={:.4}", report.step, report.loss.l_total);
    }

    let checkpoint_path = PathBuf::from("hagi-train-demo.bin");
    let writer = AsyncCheckpointWriter::new();
    writer.save_snapshot(
        &checkpoint_path,
        trainer.step as u64,
        &trainer.named_tensors(),
    )?;
    writer.finish()?;
    println!(
        "Saved checkpoint to ./checkpoints/{}",
        checkpoint_path.display()
    );

    Ok(())
}
