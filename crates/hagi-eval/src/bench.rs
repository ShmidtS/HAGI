use std::path::Path;
use std::time::{Duration, Instant};

use clifford_core::ProductTable;
use core_types::shape::Shape;
use cuda_kernels::{attention::AttentionKernels, clifford::CliffordKernels};
use serde::{Deserialize, Serialize};
use tensor_runtime::Tensor;

use config::{HdimConfig, HrmConfig};
use hdim_model::{HiddenToMultivector, StructuralFusion};
use hrm_model::{HiddenState, HrmBackbone, LmHead};

use crate::report::{
    self, EvalBackend, EvalConfig, EvalError, EvalExample, EvalReport, EvalSubset, ExampleMetadata,
};

/// Benchmark dataset selector used by synthetic and real-data evaluation loaders.
///
/// Synthetic examples use tensors shaped `[1, tokens, hidden_size]`, `[1, tokens]` targets, and
/// `[1, tokens]` prefix masks with dataset-specific token counts and target patterns. Real data
/// loading returns an error unless an external dataset source is configured; evaluation currently
/// runs through CPU reference paths, while CUDA benchmark callers report unsupported kernels instead
/// of silently falling back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkDataset {
    Math,
    Drop,
    Arc,
    Mmlu,
}

impl BenchmarkDataset {
    pub const ALL: [Self; 4] = [Self::Math, Self::Drop, Self::Arc, Self::Mmlu];

    pub fn name(self) -> &'static str {
        match self {
            Self::Math => "MATH",
            Self::Drop => "DROP",
            Self::Arc => "ARC",
            Self::Mmlu => "MMLU",
        }
    }

    pub fn version(self) -> &'static str {
        match self {
            Self::Math => "synthetic-math-v1",
            Self::Drop => "synthetic-drop-v1",
            Self::Arc => "synthetic-arc-v1",
            Self::Mmlu => "synthetic-mmlu-v1",
        }
    }
}

/// Loads benchmark examples for a named split.
///
/// Implementations return examples with input shape `[1, tokens, hidden_size]`, target shape
/// `[1, tokens]`, prefix mask shape `[1, tokens]`, and dataset metadata. Loaders should return an
/// [`EvalError`] for unavailable real data instead of panicking; synthetic CPU data has no CUDA
/// dependency or fallback behavior.
pub trait BenchmarkLoader {
    fn load(&self, split: &str) -> Result<Vec<EvalExample>, EvalError>;
}

#[derive(Debug, Clone)]
pub struct SyntheticBenchmarkLoader {
    dataset: BenchmarkDataset,
    hrm_config: HrmConfig,
    real_data: bool,
}

impl SyntheticBenchmarkLoader {
    pub fn new(dataset: BenchmarkDataset, hrm_config: HrmConfig) -> Self {
        Self {
            dataset,
            hrm_config,
            real_data: false,
        }
    }

    pub fn real_data(dataset: BenchmarkDataset, hrm_config: HrmConfig) -> Self {
        Self {
            dataset,
            hrm_config,
            real_data: true,
        }
    }
}

impl BenchmarkLoader for SyntheticBenchmarkLoader {
    fn load(&self, split: &str) -> Result<Vec<EvalExample>, EvalError> {
        if self.real_data {
            return Err(EvalError::DatasetUnavailable(
                "Real dataset requires download from HuggingFace/eval-harness".to_string(),
            ));
        }

        let (tokens, prefix_len, input_base, target_stride) = match self.dataset {
            BenchmarkDataset::Math => (8, 3, 0.11f32, 2usize),
            BenchmarkDataset::Drop => (10, 5, 0.23f32, 3usize),
            BenchmarkDataset::Arc => (12, 4, 0.37f32, 5usize),
            BenchmarkDataset::Mmlu => (14, 6, 0.53f32, 7usize),
        };
        let tokens = tokens.min(self.hrm_config.max_seq_len).max(2);
        let prefix_len = prefix_len.min(tokens - 1).max(1);
        let mut examples = Vec::with_capacity(10);

        for example_idx in 0..10 {
            let input = Tensor::from_vec(
                (0..(tokens * self.hrm_config.hidden_size))
                    .map(|i| {
                        let position = (i / self.hrm_config.hidden_size) as f32;
                        let channel = (i % self.hrm_config.hidden_size) as f32;
                        input_base + example_idx as f32 * 0.01 + position * 0.001 + channel * 0.0001
                    })
                    .collect(),
                Shape::new(vec![1, tokens, self.hrm_config.hidden_size]),
            );
            let targets = Tensor::from_vec(
                (0..tokens)
                    .map(|i| {
                        ((example_idx * target_stride + i * target_stride + prefix_len)
                            % self.hrm_config.vocab_size) as u32
                    })
                    .collect(),
                Shape::new(vec![1, tokens]),
            );
            let prefix_mask = Tensor::from_vec(
                (0..tokens).map(|i| u8::from(i < prefix_len)).collect(),
                Shape::new(vec![1, tokens]),
            );
            examples.push(EvalExample {
                input,
                targets,
                prefix_mask,
                prefix_lens: vec![prefix_len],
                metadata: Some(ExampleMetadata {
                    dataset: self.dataset.name().to_string(),
                    version: self.dataset.version().to_string(),
                    split: split.to_string(),
                }),
            });
        }
        Ok(examples)
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkSubset {
    pub dataset: BenchmarkDataset,
    pub version: String,
    pub split: String,
    pub examples: Vec<EvalExample>,
}

impl BenchmarkSubset {
    pub fn synthetic(
        dataset: BenchmarkDataset,
        split: impl Into<String>,
        config: &HrmConfig,
    ) -> Result<Self, EvalError> {
        let split = split.into();
        let loader = SyntheticBenchmarkLoader::new(dataset, config.clone());
        let examples = loader.load(&split)?;
        Ok(Self {
            dataset,
            version: dataset.version().to_string(),
            split,
            examples,
        })
    }

    pub fn from_dataset_path(
        dataset: BenchmarkDataset,
        split: impl Into<String>,
        config: &HrmConfig,
        dataset_path: impl AsRef<Path>,
    ) -> Result<Self, EvalError> {
        if !dataset_path.as_ref().exists() {
            return Err(EvalError::DatasetUnavailable(format!(
                "dataset path does not exist: {}",
                dataset_path.as_ref().display()
            )));
        }
        let split = split.into();
        let loader = SyntheticBenchmarkLoader::real_data(dataset, config.clone());
        let examples = loader.load(&split)?;
        Ok(Self {
            dataset,
            version: dataset.version().to_string(),
            split,
            examples,
        })
    }
}

/// Results of a benchmark run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub elapsed: Duration,
    pub tokens_per_sec: f64,
    pub total_elements: usize,
    pub accuracy: f32,
    pub steps: usize,
    pub dataset: Option<BenchmarkDataset>,
    pub version: Option<String>,
    pub split: Option<String>,
    pub examples: usize,
    pub loss_total: Option<f32>,
    pub loss_ce: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentLatencyResult {
    pub component: BenchmarkComponent,
    pub backend: EvalBackend,
    pub elapsed: Duration,
    pub iterations: usize,
    pub elements: usize,
    pub tokens_per_sec: Option<f64>,
    pub used_cuda: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkComponent {
    HrmBackbone,
    HdimProjector,
    HdimFusion,
    LmHead,
    CliffordGeometricProduct,
    CliffordRotorSandwich,
    SparseAttention,
    FusedRotorHrmMsa,
    TrainingStep,
}

pub struct ComponentBenchmark {
    pub config: EvalConfig,
    pub backend: EvalBackend,
}

/// Runs an evaluation subset for one benchmark dataset and records per-dataset report fields.
pub fn run_eval_subset(
    checkpoint: impl AsRef<Path>,
    subset: &BenchmarkSubset,
    config: &EvalConfig,
) -> Result<EvalReport, EvalError> {
    let eval_subset = EvalSubset {
        examples: subset.examples.clone(),
    };
    let mut report = report::run_eval_subset(checkpoint, &eval_subset, config)?;
    report.dataset_breakdowns.push(BenchmarkResult {
        elapsed: Duration::default(),
        tokens_per_sec: 0.0,
        total_elements: subset
            .examples
            .iter()
            .map(|example| example.input.numel())
            .sum(),
        accuracy: report.route_top_k_hit_rate,
        steps: subset.examples.len(),
        dataset: Some(subset.dataset),
        version: Some(subset.version.clone()),
        split: Some(subset.split.clone()),
        examples: subset.examples.len(),
        loss_total: Some(report.loss_total),
        loss_ce: Some(report.loss_ce),
    });
    Ok(report)
}

pub fn run_documented_eval_subsets(
    checkpoint: impl AsRef<Path>,
    split: &str,
    config: &EvalConfig,
) -> Result<Vec<EvalReport>, EvalError> {
    let checkpoint = checkpoint.as_ref();
    BenchmarkDataset::ALL
        .iter()
        .map(|&dataset| {
            let subset = BenchmarkSubset::synthetic(dataset, split, &config.hrm_config)?;
            run_eval_subset(checkpoint, &subset, config)
        })
        .collect()
}

impl EvalReport {
    pub fn composite_score(&self) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for row in &self.dataset_breakdowns {
            if let Some(loss_total) = row.loss_total {
                if loss_total.is_finite() && loss_total >= 0.0 {
                    sum += 1.0 / (1.0 + loss_total);
                    count += 1;
                }
            }
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f32
        }
    }
}

/// Benchmark that runs forward passes over synthetic data.
pub struct Benchmark {
    pub hrm_config: HrmConfig,
    pub hdim_config: HdimConfig,
}

impl Benchmark {
    pub fn new(hrm_config: HrmConfig, hdim_config: HdimConfig) -> Self {
        Self {
            hrm_config,
            hdim_config,
        }
    }

    /// Runs `steps` forward passes with synthetic data and reports metrics.
    pub fn run(&self, steps: usize) -> BenchmarkResult {
        let backbone = HrmBackbone::from_config(&self.hrm_config);
        let hidden_size = self.hrm_config.hidden_size;
        let structural_heads = self.hdim_config.structural_heads;
        let blade_count = self.hdim_config.blade_count_per_head;
        let structural_dim = structural_heads * blade_count;

        let projector = HiddenToMultivector::new(hidden_size, structural_heads, blade_count);
        let fusion = StructuralFusion::new(hidden_size, structural_dim);
        let lm_head = LmHead::new(self.hrm_config.vocab_size, hidden_size);

        let batch = 1;
        let tokens = self.hrm_config.max_seq_len.min(32);
        let prefix_lens = vec![tokens / 2];

        // Synthetic input: small constant values
        let input = Tensor::from_vec(
            vec![0.05f32; batch * tokens * hidden_size],
            Shape::new(vec![batch, tokens, hidden_size]),
        );

        // Synthetic targets
        let targets_data: Vec<u32> = (0..(batch * tokens))
            .map(|i| (i as u32) % self.hrm_config.vocab_size as u32)
            .collect();
        let _targets = Tensor::from_vec(targets_data.clone(), Shape::new(vec![batch, tokens]));

        let mut total_elements = 0usize;
        let mut correct = 0u64;
        let mut total_predicted = 0u64;

        let start = Instant::now();

        for step in 0..steps {
            // HRM forward
            let hrm_out = backbone.forward(&input, &prefix_lens, step);
            let hidden = hrm_out.hidden;

            // HDIM projection
            let hidden_state = HiddenState::new(hidden.clone());
            let mv_projected = projector.forward(&hidden_state);

            // Flatten for fusion
            let mv_flat = Tensor::from_vec(
                mv_projected.data().to_vec(),
                Shape::new(vec![batch, tokens, structural_dim]),
            );

            // Fusion
            let fused = fusion.forward(&hidden, &mv_flat);

            // LM head
            let logits = lm_head.project(&fused);

            // Count elements for memory footprint
            total_elements +=
                hidden.numel() + mv_projected.numel() + fused.numel() + logits.numel();

            // Compute argmax accuracy
            let logits_data = logits.data();
            let vocab = logits.shape().dims[2];
            for (bt, &target) in targets_data.iter().enumerate().take(batch * tokens) {
                let offset = bt * vocab;
                let mut best_idx = 0usize;
                let mut best_val = f32::NEG_INFINITY;
                for v in 0..vocab {
                    if logits_data[offset + v] > best_val {
                        best_val = logits_data[offset + v];
                        best_idx = v;
                    }
                }
                if best_idx as u32 == target {
                    correct += 1;
                }
                total_predicted += 1;
            }
        }

        let elapsed = start.elapsed();
        let total_tokens = (batch * tokens * steps) as f64;
        let tokens_per_sec = total_tokens / elapsed.as_secs_f64();
        let accuracy = if total_predicted > 0 {
            correct as f32 / total_predicted as f32
        } else {
            0.0
        };

        BenchmarkResult {
            elapsed,
            tokens_per_sec,
            total_elements,
            accuracy,
            steps,
            dataset: None,
            version: None,
            split: None,
            examples: 0,
            loss_total: None,
            loss_ce: None,
        }
    }
}

impl ComponentBenchmark {
    pub fn new(config: EvalConfig, backend: EvalBackend) -> Self {
        Self { config, backend }
    }

    pub fn run_hrm_backbone(&self, iterations: usize) -> Result<ComponentLatencyResult, EvalError> {
        let backbone = HrmBackbone::from_config(&self.config.hrm_config);
        let hidden_size = self.config.hrm_config.hidden_size;
        let batch = 2;
        let tokens = self.config.hrm_config.max_seq_len.clamp(2, 4);
        let prefix_lens = vec![tokens / 2; batch];
        let input = Tensor::from_vec(
            vec![0.05f32; batch * tokens * hidden_size],
            Shape::new(vec![batch, tokens, hidden_size]),
        );

        let start = Instant::now();
        for step in 0..iterations {
            let _ = backbone.forward(&input, &prefix_lens, step);
        }
        let elapsed = start.elapsed();

        Ok(self.component_result(
            BenchmarkComponent::HrmBackbone,
            elapsed,
            iterations,
            input.numel(),
            batch * tokens,
            None,
        ))
    }

    pub fn run_hdim_projector(
        &self,
        iterations: usize,
    ) -> Result<ComponentLatencyResult, EvalError> {
        let hidden_size = self.config.hrm_config.hidden_size;
        let heads = 2;
        let blade_count = 4;
        let batch = 2;
        let tokens = self.config.hrm_config.max_seq_len.clamp(2, 4);
        let projector = HiddenToMultivector::new(hidden_size, heads, blade_count);
        let hidden = Tensor::from_vec(
            vec![0.05f32; batch * tokens * hidden_size],
            Shape::new(vec![batch, tokens, hidden_size]),
        );
        let hidden_state = HiddenState::new(hidden.clone());

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = projector.forward(&hidden_state);
        }
        let elapsed = start.elapsed();

        Ok(self.component_result(
            BenchmarkComponent::HdimProjector,
            elapsed,
            iterations,
            hidden.numel(),
            batch * tokens,
            None,
        ))
    }

    pub fn run_hdim_fusion(&self, iterations: usize) -> Result<ComponentLatencyResult, EvalError> {
        let hidden_size = self.config.hrm_config.hidden_size;
        let structural_dim = 8;
        let batch = 2;
        let tokens = self.config.hrm_config.max_seq_len.clamp(2, 4);
        let fusion = StructuralFusion::new(hidden_size, structural_dim);
        let hidden = Tensor::from_vec(
            vec![0.05f32; batch * tokens * hidden_size],
            Shape::new(vec![batch, tokens, hidden_size]),
        );
        let structural = Tensor::from_vec(
            vec![0.02f32; batch * tokens * structural_dim],
            Shape::new(vec![batch, tokens, structural_dim]),
        );

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = fusion.forward(&hidden, &structural);
        }
        let elapsed = start.elapsed();

        Ok(self.component_result(
            BenchmarkComponent::HdimFusion,
            elapsed,
            iterations,
            hidden.numel() + structural.numel(),
            batch * tokens,
            None,
        ))
    }

    pub fn run_lm_head(&self, iterations: usize) -> Result<ComponentLatencyResult, EvalError> {
        let hidden_size = self.config.hrm_config.hidden_size;
        let vocab_size = self.config.hrm_config.vocab_size;
        let batch = 2;
        let tokens = self.config.hrm_config.max_seq_len.clamp(2, 4);
        let lm_head = LmHead::new(vocab_size, hidden_size);
        let hidden = Tensor::from_vec(
            vec![0.05f32; batch * tokens * hidden_size],
            Shape::new(vec![batch, tokens, hidden_size]),
        );

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = lm_head.project(&hidden);
        }
        let elapsed = start.elapsed();

        Ok(self.component_result(
            BenchmarkComponent::LmHead,
            elapsed,
            iterations,
            hidden.numel(),
            batch * tokens,
            None,
        ))
    }

    pub fn run_clifford_geometric_product(
        &self,
        iterations: usize,
    ) -> Result<ComponentLatencyResult, EvalError> {
        let kernels = CliffordKernels::new();
        let table = ProductTable::generate(3, 0, 0);
        let a = Tensor::from_vec(vec![0.25f32; 8], Shape::new(vec![8]));
        let b = Tensor::from_vec(vec![0.5f32; 8], Shape::new(vec![8]));

        let start = Instant::now();
        for _ in 0..iterations {
            kernels
                .geometric_product_kernel(&a, &b, &table)
                .map_err(|err| EvalError::DatasetUnavailable(err.to_string()))?;
        }
        let elapsed = start.elapsed();

        Ok(self.component_result(
            BenchmarkComponent::CliffordGeometricProduct,
            elapsed,
            iterations,
            a.numel() + b.numel(),
            1,
            Some(cuda_kernels::cuda_kernels_available()),
        ))
    }

    pub fn run_clifford_rotor_sandwich(
        &self,
        iterations: usize,
    ) -> Result<ComponentLatencyResult, EvalError> {
        let kernels = CliffordKernels::new();
        let table = ProductTable::generate(3, 0, 0);
        let rotor = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            Shape::new(vec![8]),
        );
        let mv = Tensor::from_vec(vec![0.25f32; 8], Shape::new(vec![8]));

        let start = Instant::now();
        for _ in 0..iterations {
            kernels
                .rotor_sandwich_kernel(&rotor, &mv, &table)
                .map_err(|err| EvalError::DatasetUnavailable(err.to_string()))?;
        }
        let elapsed = start.elapsed();

        Ok(self.component_result(
            BenchmarkComponent::CliffordRotorSandwich,
            elapsed,
            iterations,
            rotor.numel() + mv.numel(),
            1,
            Some(cuda_kernels::cuda_kernels_available()),
        ))
    }

    pub fn run_sparse_attention(
        &self,
        iterations: usize,
    ) -> Result<ComponentLatencyResult, EvalError> {
        let kernels = AttentionKernels::new();
        let hidden_size = self.config.hrm_config.hidden_size;
        let batch = 2;
        let tokens = self.config.hrm_config.max_seq_len.clamp(2, 4);
        let query = Tensor::from_vec(
            vec![0.05f32; batch * tokens * hidden_size],
            Shape::new(vec![batch, tokens, hidden_size]),
        );
        let keys = vec![
            Tensor::from_vec(vec![0.1f32; hidden_size], Shape::new(vec![hidden_size])),
            Tensor::from_vec(vec![0.2f32; hidden_size], Shape::new(vec![hidden_size])),
        ];
        let values = vec![
            Tensor::from_vec(vec![0.3f32; hidden_size], Shape::new(vec![hidden_size])),
            Tensor::from_vec(vec![0.4f32; hidden_size], Shape::new(vec![hidden_size])),
        ];
        let weights = vec![0.6f32, 0.4];

        let start = Instant::now();
        for _ in 0..iterations {
            kernels
                .sparse_attention_kernel(&query, &keys, &values, &weights)
                .map_err(|err| EvalError::DatasetUnavailable(err.to_string()))?;
        }
        let elapsed = start.elapsed();

        Ok(self.component_result(
            BenchmarkComponent::SparseAttention,
            elapsed,
            iterations,
            query.numel()
                + keys.iter().map(Tensor::numel).sum::<usize>()
                + values.iter().map(Tensor::numel).sum::<usize>(),
            batch * tokens,
            Some(cuda_kernels::cuda_kernels_available()),
        ))
    }

    pub fn run_full_pipeline(
        &self,
        iterations: usize,
    ) -> Result<ComponentLatencyResult, EvalError> {
        let hdim_config = HdimConfig {
            structural_heads: 2,
            blade_count_per_head: 4,
            ..Default::default()
        };
        let benchmark = Benchmark::new(self.config.hrm_config.clone(), hdim_config);
        let result = benchmark.run(iterations);

        Ok(ComponentLatencyResult {
            component: BenchmarkComponent::TrainingStep,
            backend: self.backend,
            elapsed: result.elapsed,
            iterations,
            elements: result.total_elements / iterations.max(1),
            tokens_per_sec: Some(result.tokens_per_sec),
            used_cuda: None,
        })
    }

    fn component_result(
        &self,
        component: BenchmarkComponent,
        elapsed: Duration,
        iterations: usize,
        elements: usize,
        token_count: usize,
        used_cuda: Option<bool>,
    ) -> ComponentLatencyResult {
        ComponentLatencyResult {
            component,
            backend: self.backend,
            elapsed,
            iterations,
            elements,
            tokens_per_sec: Some(tokens_per_sec(token_count, iterations, elapsed)),
            used_cuda,
        }
    }
}

fn tokens_per_sec(token_count: usize, iterations: usize, elapsed: Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds == 0.0 {
        0.0
    } else {
        (token_count * iterations) as f64 / seconds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> (HrmConfig, HdimConfig) {
        let hrm = HrmConfig {
            total_layers: 2,
            h_layers: 1,
            l_layers: 1,
            hidden_size: 8,
            num_heads: 2,
            expansion: 2,
            h_cycles: 1,
            l_cycles: 1,
            vocab_size: 16,
            max_seq_len: 16,
            convergence_eps: 1e-5,
            bp_warmup_ratio: 0.2,
            bp_max_steps: 2,
            warmup_steps: 10,
        };
        let hdim = HdimConfig {
            algebra_p: 1,
            algebra_q: 0,
            algebra_r: 0,
            structural_heads: 2,
            blade_count_per_head: 2,
            ..Default::default()
        };
        (hrm, hdim)
    }

    #[test]
    fn synthetic_loader_generates_dataset_specific_examples() {
        let (hrm, _) = tiny_config();
        let math = SyntheticBenchmarkLoader::new(BenchmarkDataset::Math, hrm.clone())
            .load("validation")
            .unwrap();
        let drop = SyntheticBenchmarkLoader::new(BenchmarkDataset::Drop, hrm.clone())
            .load("validation")
            .unwrap();
        let arc = SyntheticBenchmarkLoader::new(BenchmarkDataset::Arc, hrm.clone())
            .load("validation")
            .unwrap();
        let mmlu = SyntheticBenchmarkLoader::new(BenchmarkDataset::Mmlu, hrm)
            .load("validation")
            .unwrap();

        assert_ne!(math[0].input.shape().dims, drop[0].input.shape().dims);
        assert_ne!(drop[0].targets.data(), arc[0].targets.data());
        assert_ne!(arc[0].prefix_lens, mmlu[0].prefix_lens);
        assert_eq!(math[0].metadata.as_ref().unwrap().dataset, "MATH");
        assert_eq!(mmlu[0].metadata.as_ref().unwrap().dataset, "MMLU");
    }

    #[test]
    fn benchmark_runs_without_panic() {
        let (hrm, hdim) = tiny_config();
        let bench = Benchmark::new(hrm, hdim);
        let result = bench.run(2);
        assert_eq!(result.steps, 2);
        assert!(result.tokens_per_sec > 0.0);
        assert!(result.total_elements > 0);
        assert!(result.accuracy >= 0.0 && result.accuracy <= 1.0);
    }

    #[test]
    fn benchmark_accuracy_in_range() {
        let (hrm, hdim) = tiny_config();
        let bench = Benchmark::new(hrm, hdim);
        let result = bench.run(5);
        assert!(result.accuracy >= 0.0);
        assert!(result.accuracy <= 1.0);
    }

    #[test]
    fn benchmark_elements_scale_with_steps() {
        let (hrm, hdim) = tiny_config();
        let bench = Benchmark::new(hrm.clone(), hdim.clone());

        let r1 = bench.run(1);
        let r3 = bench.run(3);

        // Elements should scale linearly with steps
        let ratio = r3.total_elements as f64 / r1.total_elements as f64;
        assert!(
            (ratio - 3.0).abs() < 0.01,
            "expected 3x elements ratio, got {}",
            ratio
        );
    }
}
