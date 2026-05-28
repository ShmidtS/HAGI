use std::path::Path;

use config::HrmConfig;
use core_types::shape::Shape;
use hagi_train::load_checkpoint;
use hrm_model::{HrmBackbone, LmHead};
use losses::{total_loss, AuxTargets, IsoPairBatch, LossError, LossWeights};
use serde::{Deserialize, Serialize};
use tensor_runtime::Tensor;

use crate::bench::{BenchmarkResult, ComponentLatencyResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalBackend {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    pub loss_total: f32,
    pub loss_ce: f32,
    pub route_top_k_hit_rate: f32,
    pub effective_h_cycles_mean: f32,
    pub effective_l_cycles_mean: f32,
    pub backend: EvalBackend,
    pub dataset_breakdowns: Vec<BenchmarkResult>,
    pub component_latencies: Vec<ComponentLatencyResult>,
}

impl EvalReport {
    pub fn add_component_latency(&mut self, result: ComponentLatencyResult) {
        self.component_latencies.push(result);
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("EvalReport serialization should not fail")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenDiff {
    pub compared_elements: usize,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
    pub tolerance: f32,
    pub within_tolerance: bool,
}

#[derive(Debug, Clone)]
pub struct EvalConfig {
    pub hrm_config: HrmConfig,
    pub loss_weights: LossWeights,
    pub backend: EvalBackend,
    pub route_top_k: usize,
}

#[derive(Debug, Clone)]
pub struct EvalSubset {
    pub examples: Vec<EvalExample>,
}

#[derive(Debug, Clone)]
pub struct EvalExample {
    pub input: Tensor<f32>,
    pub targets: Tensor<u32>,
    pub prefix_mask: Tensor<u8>,
    pub prefix_lens: Vec<usize>,
    pub metadata: Option<ExampleMetadata>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExampleMetadata {
    pub dataset: String,
    pub version: String,
    pub split: String,
}

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("checkpoint failed: {0}")]
    Checkpoint(#[from] std::io::Error),
    #[error("loss failed: {0}")]
    Loss(#[from] LossError),
    #[error("empty eval subset")]
    EmptySubset,
    #[error("empty response-token set")]
    EmptyResponse,
    #[error("invalid route_top_k: expected at least 1")]
    InvalidTopK,
    #[error("golden output shape mismatch: cpu={cpu:?}, cuda={cuda:?}")]
    GoldenShapeMismatch { cpu: Vec<usize>, cuda: Vec<usize> },
    #[error(
        "golden output diff exceeded tolerance: max_abs_diff={max_abs_diff}, tolerance={tolerance}"
    )]
    GoldenToleranceExceeded { max_abs_diff: f32, tolerance: f32 },
    #[error("dataset unavailable: {0}")]
    DatasetUnavailable(String),
    #[error("checkpoint tensor mismatch: {0}")]
    CheckpointMismatch(String),
    #[error("checkpoint format unsupported for eval: {0}")]
    CheckpointUnsupported(String),
}

pub fn run_eval_subset(
    checkpoint: impl AsRef<Path>,
    subset: &EvalSubset,
    config: &EvalConfig,
) -> Result<EvalReport, EvalError> {
    let (meta, _tensors) = load_checkpoint(checkpoint.as_ref())?;
    if subset.examples.is_empty() {
        return Err(EvalError::EmptySubset);
    }
    if config.route_top_k == 0 {
        return Err(EvalError::InvalidTopK);
    }

    let backbone = HrmBackbone::from_config(&config.hrm_config);
    let lm_head = LmHead::new(config.hrm_config.vocab_size, config.hrm_config.hidden_size);
    let mut loss_total = 0.0f32;
    let mut loss_ce = 0.0f32;
    let mut h_cycles = 0usize;
    let mut l_cycles = 0.0f32;
    let mut top_k_hits = 0usize;
    let mut response_tokens = 0usize;

    for (index, example) in subset.examples.iter().enumerate() {
        let hrm_out = backbone.forward(
            &example.input,
            &example.prefix_lens,
            meta.step as usize + index,
        );
        let logits = lm_head.project(&hrm_out.hidden);
        let aux_targets = AuxTargets {
            positive_pairs: &[],
            negative_pairs: &[],
            margin: 0.5,
        };
        let iso_pair = IsoPairBatch {
            u_src: logits.as_view(),
            u_tgt: logits.as_view(),
        };
        let loss = total_loss(
            logits.as_view(),
            example.targets.as_view(),
            &aux_targets,
            &iso_pair,
            &config.loss_weights,
            meta.step as usize + index,
            Some(example.prefix_mask.as_view()),
        )?;
        loss_total += loss.l_total;
        loss_ce += loss.l_ce;
        h_cycles += hrm_out.effective_h_cycles;
        l_cycles += hrm_out.effective_l_cycles as f32 / hrm_out.effective_h_cycles.max(1) as f32;

        let (hits, count) = top_k_hit_count(
            &logits,
            &example.targets,
            &example.prefix_mask,
            config.route_top_k,
        );
        top_k_hits += hits;
        response_tokens += count;
    }

    if response_tokens == 0 {
        return Err(EvalError::EmptyResponse);
    }

    let examples = subset.examples.len() as f32;
    Ok(EvalReport {
        loss_total: loss_total / examples,
        loss_ce: loss_ce / examples,
        route_top_k_hit_rate: top_k_hits as f32 / response_tokens as f32,
        effective_h_cycles_mean: h_cycles as f32 / examples,
        effective_l_cycles_mean: l_cycles / examples,
        backend: config.backend,
        dataset_breakdowns: Vec::new(),
        component_latencies: Vec::new(),
    })
}

pub fn compare_golden_outputs(
    cpu: &Tensor<f32>,
    cuda: &Tensor<f32>,
    tolerance: f32,
) -> Result<GoldenDiff, EvalError> {
    if cpu.shape() != cuda.shape() {
        return Err(EvalError::GoldenShapeMismatch {
            cpu: cpu.shape().dims.clone(),
            cuda: cuda.shape().dims.clone(),
        });
    }

    let mut max_abs_diff = 0.0f32;
    let mut sum_abs_diff = 0.0f32;
    for (&a, &b) in cpu.data().iter().zip(cuda.data().iter()) {
        let diff = (a - b).abs();
        max_abs_diff = max_abs_diff.max(diff);
        sum_abs_diff += diff;
    }

    let compared_elements = cpu.numel();
    let mean_abs_diff = if compared_elements == 0 {
        0.0
    } else {
        sum_abs_diff / compared_elements as f32
    };
    let diff = GoldenDiff {
        compared_elements,
        max_abs_diff,
        mean_abs_diff,
        tolerance,
        within_tolerance: max_abs_diff <= tolerance,
    };

    if diff.within_tolerance {
        Ok(diff)
    } else {
        Err(EvalError::GoldenToleranceExceeded {
            max_abs_diff,
            tolerance,
        })
    }
}

fn top_k_hit_count(
    logits: &Tensor<f32>,
    targets: &Tensor<u32>,
    prefix_mask: &Tensor<u8>,
    top_k: usize,
) -> (usize, usize) {
    let shape = logits.shape();
    let batch = shape.dims[0];
    let tokens = shape.dims[1];
    let vocab = shape.dims[2];
    let k = top_k.min(vocab);
    let mut hits = 0usize;
    let mut count = 0usize;

    for bt in 0..(batch * tokens) {
        if prefix_mask.data()[bt] != 0 {
            continue;
        }
        count += 1;
        let target = targets.data()[bt] as usize;
        let offset = bt * vocab;
        let mut better = 0usize;
        let target_score = logits.data()[offset + target];
        for v in 0..vocab {
            if v != target && logits.data()[offset + v] > target_score {
                better += 1;
            }
        }
        if better < k {
            hits += 1;
        }
    }

    (hits, count)
}

impl EvalSubset {
    pub fn synthetic(config: &HrmConfig, examples: usize, tokens: usize) -> Self {
        let tokens = tokens.min(config.max_seq_len).max(2);
        let mut subset = Vec::with_capacity(examples);
        for example_idx in 0..examples {
            let input = Tensor::from_vec(
                vec![0.01f32 * (example_idx as f32 + 1.0); tokens * config.hidden_size],
                Shape::new(vec![1, tokens, config.hidden_size]),
            );
            let targets = Tensor::from_vec(
                (0..tokens)
                    .map(|i| ((i + example_idx) % config.vocab_size) as u32)
                    .collect(),
                Shape::new(vec![1, tokens]),
            );
            let prefix_len = tokens / 2;
            let prefix_mask = Tensor::from_vec(
                (0..tokens).map(|i| u8::from(i < prefix_len)).collect(),
                Shape::new(vec![1, tokens]),
            );
            subset.push(EvalExample {
                input,
                targets,
                prefix_mask,
                prefix_lens: vec![prefix_len],
                metadata: None,
            });
        }
        Self { examples: subset }
    }
}
