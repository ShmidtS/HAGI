use config::HrmConfig;
use core_types::shape::Shape;
use hagi_eval::{
    compare_golden_outputs, load_checkpoint_for_eval, run_benchmark_subset,
    run_documented_eval_subsets, run_eval_subset, BenchmarkDataset, BenchmarkSubset, EvalBackend,
    EvalConfig, EvalError, EvalReport, EvalSubset,
};
use hagi_train::save_checkpoint;
use hrm_model::{HrmBackbone, LmHead};
use losses::LossWeights;
use msa_adapter::{MemorySlot, MsaConfig, SlotRegistry, SparseRouter};
use tensor_runtime::Tensor;

fn tiny_hrm_config() -> HrmConfig {
    HrmConfig {
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
    }
}

fn eval_config() -> EvalConfig {
    EvalConfig {
        hrm_config: tiny_hrm_config(),
        loss_weights: LossWeights {
            lambda_aux: 0.0,
            lambda_iso_target: 0.0,
            iso_warmup_steps: 0,
        },
        backend: EvalBackend::Cpu,
        route_top_k: 3,
    }
}

fn checkpoint_path(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from("tests").join(name)
}

fn write_checkpoint(name: &str) -> std::path::PathBuf {
    let config = eval_config();
    write_eval_checkpoint(name, &config)
}

fn write_eval_checkpoint(name: &str, config: &EvalConfig) -> std::path::PathBuf {
    let path = checkpoint_path(name);
    let _ = std::fs::remove_file(&path);
    let hidden_size = config.hrm_config.hidden_size;
    let structural_dim = 8;
    let vocab_size = config.hrm_config.vocab_size;
    let projector = Tensor::from_vec(
        sequence(0.01, hidden_size * structural_dim),
        Shape::new(vec![hidden_size, structural_dim]),
    );
    let fusion_gate = Tensor::from_vec(
        sequence(0.02, (hidden_size + structural_dim) * hidden_size),
        Shape::new(vec![hidden_size + structural_dim, hidden_size]),
    );
    let fusion_fuse = Tensor::from_vec(
        sequence(0.03, structural_dim * hidden_size),
        Shape::new(vec![structural_dim, hidden_size]),
    );
    let lm_head = Tensor::from_vec(
        sequence(0.04, hidden_size * vocab_size),
        Shape::new(vec![hidden_size, vocab_size]),
    );
    save_checkpoint(
        &path,
        11,
        &[
            ("projector.w_proj", &projector),
            ("fusion.w_gate", &fusion_gate),
            ("fusion.w_fuse", &fusion_fuse),
            ("lm_head.w_proj", &lm_head),
        ],
    )
    .unwrap();
    path
}

fn sequence(scale: f32, len: usize) -> Vec<f32> {
    (0..len).map(|index| scale * (index as f32 + 1.0)).collect()
}

fn assert_synthetic_benchmark(dataset: BenchmarkDataset, name: &str) {
    let config = eval_config();
    let subset = BenchmarkSubset::synthetic(dataset, "validation", &config.hrm_config).unwrap();
    let path = write_checkpoint(name);

    let report = run_benchmark_subset(&path, &subset, &config).unwrap();

    assert_eq!(subset.examples.len(), 10);
    assert!(report.loss_total.is_finite());
    assert!(report.loss_ce.is_finite());
    assert_eq!(report.dataset_breakdowns.len(), 1);
    assert_eq!(report.dataset_breakdowns[0].dataset, Some(dataset));
    assert_eq!(
        report.dataset_breakdowns[0].split.as_deref(),
        Some("validation")
    );
    assert_eq!(report.dataset_breakdowns[0].examples, 10);
    assert!(report.dataset_breakdowns[0].loss_total.is_some());

    let _ = std::fs::remove_file(path);
}

#[test]
fn eval_report_load_checkpoint() {
    let config = eval_config();
    let subset = EvalSubset::synthetic(&config.hrm_config, 2, 8);
    let path = write_checkpoint("hagi_eval_report_load_checkpoint.bin");

    let report = run_eval_subset(&path, &subset, &config).unwrap();

    assert!(report.loss_total.is_finite());
    assert!(report.loss_ce.is_finite());
    assert!((0.0..=1.0).contains(&report.route_top_k_hit_rate));
    assert!(report.effective_h_cycles_mean > 0.0);
    assert!(report.effective_l_cycles_mean > 0.0);
    assert_eq!(report.backend, EvalBackend::Cpu);

    let _ = std::fs::remove_file(path);
}

#[test]
fn deterministic_eval_order() {
    let config = eval_config();
    let subset = EvalSubset::synthetic(&config.hrm_config, 3, 8);
    let path = write_checkpoint("hagi_deterministic_eval_order.bin");

    let first = run_eval_subset(&path, &subset, &config).unwrap();
    let second = run_eval_subset(&path, &subset, &config).unwrap();

    assert_eq!(first, second);

    let _ = std::fs::remove_file(path);
}

#[test]
fn eval_report_is_deterministic_for_fixed_seed() {
    let config = eval_config();
    let subset = EvalSubset::synthetic(&config.hrm_config, 3, 8);
    let path = write_checkpoint("hagi_eval_report_is_deterministic_for_fixed_seed.bin");

    let first = run_eval_subset(&path, &subset, &config).unwrap();
    let second = run_eval_subset(&path, &subset, &config).unwrap();

    assert_eq!(first, second);

    let _ = std::fs::remove_file(path);
}

#[test]
fn cpu_baseline_produces_finite_loss() {
    let config = eval_config();
    let subset = EvalSubset::synthetic(&config.hrm_config, 2, 8);
    let path = write_checkpoint("hagi_cpu_baseline_produces_finite_loss.bin");

    let report = run_eval_subset(&path, &subset, &config).unwrap();

    assert!(report.loss_total.is_finite());
    assert!(report.loss_ce.is_finite());

    let _ = std::fs::remove_file(path);
}

#[test]
fn golden_logits_match_within_tolerance() {
    let config = eval_config();
    let backbone = HrmBackbone::from_config(&config.hrm_config);
    let lm_head = LmHead::new(config.hrm_config.vocab_size, config.hrm_config.hidden_size);
    let subset = EvalSubset::synthetic(&config.hrm_config, 1, 8);
    let example = &subset.examples[0];

    let first = backbone.forward(&example.input, &example.prefix_lens, 11);
    let second = backbone.forward(&example.input, &example.prefix_lens, 11);
    let first_logits = lm_head.project(&first.hidden);
    let second_logits = lm_head.project(&second.hidden);
    let diff = compare_golden_outputs(&first_logits, &second_logits, 1e-4).unwrap();

    assert!(diff.within_tolerance);
    assert_eq!(diff.max_abs_diff, 0.0);
}

#[test]
fn route_top_k_hit_rate_is_reasonable() {
    let config = eval_config();
    let mut registry = SlotRegistry::new();
    for id in 0..3 {
        registry.register(MemorySlot::new(
            id,
            Tensor::from_vec(vec![1.0 + id as f32; 8], Shape::new(vec![8])),
            Tensor::from_vec(vec![0.0; 8], Shape::new(vec![8])),
            0,
            "eval".into(),
        ));
    }
    let query = Tensor::from_vec(
        vec![0.5; config.hrm_config.hidden_size],
        Shape::new(vec![1, 1, config.hrm_config.hidden_size]),
    );
    let router = SparseRouter::try_from_config(
        MsaConfig::try_new(config.route_top_k).expect("eval test top_k must be valid"),
    )
    .expect("SparseRouter eval test config must be valid");
    let (slot_ids, _) = router.route(&query, &registry);
    let route_top_k_hit_rate = if slot_ids.is_empty() { 0.0 } else { 1.0 };

    assert!(route_top_k_hit_rate > 0.0);
}

#[test]
fn effective_cycles_within_config_bounds() {
    let config = eval_config();
    let subset = EvalSubset::synthetic(&config.hrm_config, 2, 8);
    let path = write_checkpoint("hagi_effective_cycles_within_config_bounds.bin");

    let report = run_eval_subset(&path, &subset, &config).unwrap();

    assert!(report.effective_h_cycles_mean <= config.hrm_config.h_cycles as f32);
    assert!(report.effective_l_cycles_mean <= config.hrm_config.l_cycles as f32);

    let _ = std::fs::remove_file(path);
}

#[test]
fn checkpoint_loads_eval_model_components() {
    let config = eval_config();
    let path = write_eval_checkpoint("hagi_checkpoint_loads_eval_model_components.bin", &config);

    let model = load_checkpoint_for_eval(&path, &config).unwrap();

    assert_eq!(
        model.backbone.config.hidden_size,
        config.hrm_config.hidden_size
    );
    assert_eq!(model.lm_head.vocab_size, config.hrm_config.vocab_size);
    assert_eq!(
        model.hdim_forward.projector.w_proj.data(),
        sequence(0.01, config.hrm_config.hidden_size * 8).as_slice()
    );
    assert_eq!(
        model.hdim_forward.fusion.w_gate.data(),
        sequence(
            0.02,
            (config.hrm_config.hidden_size + 8) * config.hrm_config.hidden_size
        )
        .as_slice()
    );
    assert_eq!(
        model.hdim_forward.fusion.w_fuse.data(),
        sequence(0.03, 8 * config.hrm_config.hidden_size).as_slice()
    );
    assert_eq!(
        model.lm_head.w_proj.data(),
        sequence(
            0.04,
            config.hrm_config.hidden_size * config.hrm_config.vocab_size
        )
        .as_slice()
    );
    assert!(model.slot_registry.is_none());

    let _ = std::fs::remove_file(path);
}

#[test]
fn documented_subsets_have_individual_rows_and_composite_score() {
    let config = eval_config();
    let path = write_checkpoint("hagi_documented_subsets_have_individual_rows.bin");

    let reports = run_documented_eval_subsets(&path, "validation", &config).unwrap();

    assert_eq!(reports.len(), BenchmarkDataset::ALL.len());
    for report in &reports {
        assert_eq!(report.dataset_breakdowns.len(), 1);
        assert!(report.composite_score() > 0.0);
    }

    let _ = std::fs::remove_file(path);
}

#[test]
fn missing_real_dataset_path_returns_error() {
    let config = eval_config();
    let err = BenchmarkSubset::from_dataset_path(
        BenchmarkDataset::Math,
        "validation",
        &config.hrm_config,
        "tests/definitely_missing_hagi_eval_dataset",
    )
    .unwrap_err();

    assert!(matches!(err, EvalError::DatasetUnavailable(_)));
}

#[test]
fn byte_stable_cpu_report() {
    let report = EvalReport {
        loss_total: 1.25,
        loss_ce: 1.0,
        route_top_k_hit_rate: 0.5,
        effective_h_cycles_mean: 1.0,
        effective_l_cycles_mean: 2.0,
        backend: EvalBackend::Cpu,
        dataset_breakdowns: Vec::new(),
        component_latencies: Vec::new(),
    };

    let bytes = serde_json::to_vec(&report).unwrap();
    let restored: EvalReport = serde_json::from_slice(&bytes).unwrap();
    let bytes_again = serde_json::to_vec(&restored).unwrap();

    assert_eq!(bytes, bytes_again);
}

#[test]
fn golden_diff_cpu_cuda_tolerance() {
    let cpu = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], Shape::new(vec![3]));
    let cuda = Tensor::from_vec(vec![1.00001f32, 1.99999, 3.00005], Shape::new(vec![3]));

    let diff = compare_golden_outputs(&cpu, &cuda, 1e-4).unwrap();

    assert_eq!(diff.compared_elements, 3);
    assert!(diff.within_tolerance);
    assert!(diff.max_abs_diff <= 1e-4);
}

#[test]
fn test_math_synthetic_eval() {
    assert_synthetic_benchmark(BenchmarkDataset::Math, "hagi_math_synthetic_eval.bin");
}

#[test]
fn test_arc_synthetic_eval() {
    assert_synthetic_benchmark(BenchmarkDataset::Arc, "hagi_arc_synthetic_eval.bin");
}

#[test]
fn test_mmlu_synthetic_eval() {
    assert_synthetic_benchmark(BenchmarkDataset::Mmlu, "hagi_mmlu_synthetic_eval.bin");
}

#[test]
fn test_drop_synthetic_eval() {
    assert_synthetic_benchmark(BenchmarkDataset::Drop, "hagi_drop_synthetic_eval.bin");
}
