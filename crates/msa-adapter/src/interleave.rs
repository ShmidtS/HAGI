use tensor_runtime::Tensor;

use crate::route::RouteSelection;

#[derive(Debug, Clone, Copy)]
pub struct MemoryInterleaveConfig {
    pub max_steps: usize,
    pub min_delta: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryStopReason {
    MaxSteps,
    Converged,
    EmptySelection,
}

#[derive(Debug)]
pub struct MemoryInterleaveReport {
    pub steps: usize,
    pub stop_reason: MemoryStopReason,
    pub last_delta: f32,
}

pub fn run_memory_interleave(
    initial: &Tensor<f32>,
    selection: &RouteSelection,
    config: MemoryInterleaveConfig,
) -> MemoryInterleaveReport {
    if selection.slot_ids.is_empty() {
        return MemoryInterleaveReport {
            steps: 0,
            stop_reason: MemoryStopReason::EmptySelection,
            last_delta: 0.0,
        };
    }

    let base_delta =
        initial.data().iter().map(|v| v.abs()).sum::<f32>() / initial.numel().max(1) as f32;
    let mut last_delta = base_delta;

    for step in 0..config.max_steps {
        last_delta = base_delta / (step + 1) as f32;
        if last_delta < config.min_delta {
            return MemoryInterleaveReport {
                steps: step + 1,
                stop_reason: MemoryStopReason::Converged,
                last_delta,
            };
        }
    }

    MemoryInterleaveReport {
        steps: config.max_steps,
        stop_reason: MemoryStopReason::MaxSteps,
        last_delta,
    }
}
