//! HAGI evaluation and benchmarking infrastructure.

pub mod bench;
pub mod golden;
pub mod report;

pub use bench::{
    run_documented_eval_subsets, run_eval_subset as run_benchmark_subset, Benchmark,
    BenchmarkComponent, BenchmarkDataset, BenchmarkLoader, BenchmarkResult, BenchmarkSubset,
    ComponentBenchmark, ComponentLatencyResult, SyntheticBenchmarkLoader,
};
pub use golden::{load_checkpoint_for_eval, EvalModel, HdimForward};
pub use report::{
    compare_golden_outputs, run_eval_subset, EvalBackend, EvalConfig, EvalError, EvalExample,
    EvalReport, EvalSubset, ExampleMetadata, GoldenDiff,
};
