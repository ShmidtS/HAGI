use core_types::shape::Shape;
use msa_adapter::{
    run_memory_interleave, MemoryInterleaveConfig, MemoryStopReason, RouteSelection,
};
use smallvec::smallvec;
use tensor_runtime::Tensor;

fn tensor(values: Vec<f32>) -> Tensor<f32> {
    Tensor::from_vec(values, Shape::new(vec![2]))
}

#[test]
fn memory_interleave_stops_on_empty_selection() {
    let selection = RouteSelection {
        slot_ids: smallvec![],
        raw_scores: smallvec![],
        normalized_weights: smallvec![],
    };

    let report = run_memory_interleave(
        &tensor(vec![1.0, 1.0]),
        &selection,
        MemoryInterleaveConfig {
            max_steps: 10,
            min_delta: 0.1,
        },
    );

    assert_eq!(report.steps, 0);
    assert_eq!(report.stop_reason, MemoryStopReason::EmptySelection);
}

#[test]
fn memory_interleave_stops_on_max_steps() {
    let selection = RouteSelection {
        slot_ids: smallvec![1],
        raw_scores: smallvec![1.0],
        normalized_weights: smallvec![1.0],
    };

    let report = run_memory_interleave(
        &tensor(vec![10.0, 10.0]),
        &selection,
        MemoryInterleaveConfig {
            max_steps: 3,
            min_delta: 0.1,
        },
    );

    assert_eq!(report.steps, 3);
    assert_eq!(report.stop_reason, MemoryStopReason::MaxSteps);
}

#[test]
fn memory_interleave_stops_on_converged_delta() {
    let selection = RouteSelection {
        slot_ids: smallvec![1],
        raw_scores: smallvec![1.0],
        normalized_weights: smallvec![1.0],
    };

    let report = run_memory_interleave(
        &tensor(vec![0.01, 0.01]),
        &selection,
        MemoryInterleaveConfig {
            max_steps: 10,
            min_delta: 0.1,
        },
    );

    assert_eq!(report.steps, 1);
    assert_eq!(report.stop_reason, MemoryStopReason::Converged);
    assert!(report.last_delta < 0.1);
}
