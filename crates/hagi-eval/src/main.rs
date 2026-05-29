use std::path::{Path, PathBuf};

use hagi_eval::{run_eval_subset, EvalBackend, EvalConfig, EvalError, EvalSubset};
use losses::LossWeights;

fn main() {
    if let Err(err) = run() {
        eprintln!("hagi-eval failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint_arg = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./checkpoints/".to_string());
    let checkpoint = resolve_checkpoint_path(Path::new(&checkpoint_arg))?;
    let model_config = config::demo_model_config();
    model_config.validate()?;

    let eval_config = EvalConfig {
        hrm_config: model_config.hrm.clone(),
        loss_weights: LossWeights {
            lambda_aux: 0.0,
            lambda_iso_target: 0.0,
            iso_warmup_steps: 0,
        },
        backend: EvalBackend::Cpu,
        route_top_k: model_config.msa.top_k,
    };

    let subset = EvalSubset::synthetic(&eval_config.hrm_config, 4, 16);
    let report = run_eval_subset(&checkpoint, &subset, &eval_config)?;

    println!("checkpoint: {}", checkpoint.display());
    println!("backend: {:?}", report.backend);
    println!("loss_total: {:.6}", report.loss_total);
    println!("loss_ce: {:.6}", report.loss_ce);
    println!("route_top_k_hit_rate: {:.6}", report.route_top_k_hit_rate);
    println!(
        "effective_h_cycles_mean: {:.6}",
        report.effective_h_cycles_mean
    );
    println!(
        "effective_l_cycles_mean: {:.6}",
        report.effective_l_cycles_mean
    );
    println!("composite_score: {:.6}", report.composite_score());

    Ok(())
}

fn resolve_checkpoint_path(path: &Path) -> Result<PathBuf, EvalError> {
    let candidate = if path.is_dir() {
        newest_file_in_dir(path)?
    } else {
        path.to_path_buf()
    };

    if let Some(relative) = checkpoint_relative_path(&candidate)? {
        Ok(relative)
    } else {
        Ok(candidate)
    }
}

fn checkpoint_relative_path(path: &Path) -> Result<Option<PathBuf>, EvalError> {
    let checkpoints = Path::new("checkpoints");
    if let Ok(relative) = path.strip_prefix(checkpoints) {
        return non_empty_checkpoint_relative(relative).map(Some);
    }
    if let Ok(relative) = path.strip_prefix(Path::new(".").join(checkpoints)) {
        return non_empty_checkpoint_relative(relative).map(Some);
    }

    let cwd_checkpoints = std::env::current_dir()?.join(checkpoints);
    if let Ok(relative) = path.strip_prefix(cwd_checkpoints) {
        return non_empty_checkpoint_relative(relative).map(Some);
    }

    Ok(None)
}

fn non_empty_checkpoint_relative(relative: &Path) -> Result<PathBuf, EvalError> {
    if relative.as_os_str().is_empty() {
        Err(EvalError::DatasetUnavailable(
            "checkpoint directory contains no checkpoint files".to_string(),
        ))
    } else {
        Ok(relative.to_path_buf())
    }
}

fn newest_file_in_dir(dir: &Path) -> Result<PathBuf, EvalError> {
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if !file_type.is_file() {
            continue;
        }
        let modified = entry.metadata()?.modified()?;
        if newest
            .as_ref()
            .map_or(true, |(current, _)| modified > *current)
        {
            newest = Some((modified, entry.path()));
        }
    }

    newest.map(|(_, path)| path).ok_or_else(|| {
        EvalError::DatasetUnavailable(format!(
            "checkpoint directory contains no checkpoint files: {}",
            dir.display()
        ))
    })
}
