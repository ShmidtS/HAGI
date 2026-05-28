use std::time::Duration;

use config::HrmConfig;
use hagi_eval::{ComponentBenchmark, EvalBackend, EvalConfig};
use losses::LossWeights;

fn tiny_config() -> EvalConfig {
    EvalConfig {
        hrm_config: HrmConfig {
            total_layers: 2,
            h_layers: 1,
            l_layers: 1,
            hidden_size: 8,
            num_heads: 2,
            expansion: 2,
            h_cycles: 1,
            l_cycles: 1,
            vocab_size: 16,
            max_seq_len: 4,
            convergence_eps: 1e-5,
            bp_warmup_ratio: 0.2,
            bp_max_steps: 2,
            warmup_steps: 10,
        },
        loss_weights: LossWeights {
            lambda_aux: 0.0,
            lambda_iso_target: 0.0,
            iso_warmup_steps: 0,
        },
        backend: EvalBackend::Cpu,
        route_top_k: 3,
    }
}

fn assert_positive_latency(result: hagi_eval::ComponentLatencyResult) {
    assert!(result.elapsed > Duration::ZERO);
    assert!(result.iterations > 0);
    assert!(result.elements > 0);
}

#[test]
fn component_benchmark_hrm_backbone_runs_and_reports_positive_latency() {
    let bench = ComponentBenchmark::new(tiny_config(), EvalBackend::Cpu);
    assert_positive_latency(bench.run_hrm_backbone(3).unwrap());
}

#[test]
fn component_benchmark_hdim_projector_runs_and_reports_positive_latency() {
    let bench = ComponentBenchmark::new(tiny_config(), EvalBackend::Cpu);
    assert_positive_latency(bench.run_hdim_projector(3).unwrap());
}

#[test]
fn component_benchmark_hdim_fusion_runs_and_reports_positive_latency() {
    let bench = ComponentBenchmark::new(tiny_config(), EvalBackend::Cpu);
    assert_positive_latency(bench.run_hdim_fusion(3).unwrap());
}

#[test]
fn component_benchmark_lm_head_runs_and_reports_positive_latency() {
    let bench = ComponentBenchmark::new(tiny_config(), EvalBackend::Cpu);
    assert_positive_latency(bench.run_lm_head(3).unwrap());
}

#[test]
fn component_benchmark_clifford_gp_runs_and_reports_positive_latency() {
    let bench = ComponentBenchmark::new(tiny_config(), EvalBackend::Cpu);
    assert_positive_latency(bench.run_clifford_geometric_product(3).unwrap());
}

#[test]
fn component_benchmark_clifford_rotor_runs_and_reports_positive_latency() {
    let bench = ComponentBenchmark::new(tiny_config(), EvalBackend::Cpu);
    assert_positive_latency(bench.run_clifford_rotor_sandwich(3).unwrap());
}

#[test]
fn component_benchmark_sparse_attention_runs_and_reports_positive_latency() {
    let bench = ComponentBenchmark::new(tiny_config(), EvalBackend::Cpu);
    assert_positive_latency(bench.run_sparse_attention(3).unwrap());
}

#[test]
fn component_benchmark_full_pipeline_runs_and_reports_positive_latency() {
    let bench = ComponentBenchmark::new(tiny_config(), EvalBackend::Cpu);
    assert_positive_latency(bench.run_full_pipeline(3).unwrap());
}
