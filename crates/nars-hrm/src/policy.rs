use config::HrmConfig;
use nars_core::{BudgetValue, TruthValue};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HrmControlPolicy {
    pub h_cycle_budget: BudgetValue,
    pub l_cycle_budget: BudgetValue,
    pub convergence_threshold: TruthValue,
    pub bp_depth_policy: TruthValue,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedHrmControl {
    pub h_cycles: usize,
    pub l_cycles: usize,
    pub convergence_eps: f32,
    pub bp_steps: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HrmPolicyLimits {
    pub min_h_cycles: usize,
    pub max_h_cycles: usize,
    pub min_l_cycles: usize,
    pub max_l_cycles: usize,
    pub min_convergence_eps: f32,
    pub max_convergence_eps: f32,
    pub min_bp_steps: usize,
    pub max_bp_steps: usize,
}

impl Default for HrmPolicyLimits {
    fn default() -> Self {
        Self {
            min_h_cycles: 1,
            max_h_cycles: 8,
            min_l_cycles: 1,
            max_l_cycles: 16,
            min_convergence_eps: 1e-6,
            max_convergence_eps: 1e-2,
            min_bp_steps: 1,
            max_bp_steps: 16,
        }
    }
}

impl Default for HrmControlPolicy {
    fn default() -> Self {
        Self {
            h_cycle_budget: BudgetValue::new(0.5, 0.5, 0.5),
            l_cycle_budget: BudgetValue::new(0.5, 0.5, 0.5),
            convergence_threshold: TruthValue::new(0.5, 0.9),
            bp_depth_policy: TruthValue::new(0.5, 0.9),
        }
    }
}

impl HrmControlPolicy {
    pub fn resolve(&self, base: &HrmConfig, limits: &HrmPolicyLimits) -> ResolvedHrmControl {
        ResolvedHrmControl {
            h_cycles: scale_budget_to_usize(
                self.h_cycle_budget,
                limits.min_h_cycles,
                limits.max_h_cycles,
                base.h_cycles,
            ),
            l_cycles: scale_budget_to_usize(
                self.l_cycle_budget,
                limits.min_l_cycles,
                limits.max_l_cycles,
                base.l_cycles,
            ),
            convergence_eps: scale_truth_to_f32(
                self.convergence_threshold,
                limits.min_convergence_eps,
                limits.max_convergence_eps,
                base.convergence_eps,
            ),
            bp_steps: scale_truth_to_usize(
                self.bp_depth_policy,
                limits.min_bp_steps,
                limits.max_bp_steps,
                base.bp_max_steps,
            ),
        }
    }
}

pub fn scale_budget_to_usize(
    budget: BudgetValue,
    min: usize,
    max: usize,
    fallback: usize,
) -> usize {
    if min > max {
        return fallback;
    }
    let score =
        ((budget.priority() + budget.durability() + budget.quality()) / 3.0).clamp(0.0, 1.0);
    let span = max - min;
    (min as f64 + score * span as f64).round() as usize
}

pub fn scale_truth_to_f32(truth: TruthValue, min: f32, max: f32, fallback: f32) -> f32 {
    if min > max || !min.is_finite() || !max.is_finite() {
        return fallback;
    }
    let score = (truth.frequency() * truth.confidence()).clamp(0.0, 1.0) as f32;
    min + score * (max - min)
}

fn scale_truth_to_usize(truth: TruthValue, min: usize, max: usize, fallback: usize) -> usize {
    if min > max {
        return fallback;
    }
    let score = (truth.frequency() * truth.confidence()).clamp(0.0, 1.0);
    let span = max - min;
    (min as f64 + score * span as f64).round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_resolution_clamps_h_cycles() {
        let base = HrmConfig::default();
        let limits = HrmPolicyLimits {
            min_h_cycles: 2,
            max_h_cycles: 3,
            ..HrmPolicyLimits::default()
        };
        let policy = HrmControlPolicy {
            h_cycle_budget: BudgetValue::new(1.0, 1.0, 1.0),
            ..HrmControlPolicy::default()
        };

        let resolved = policy.resolve(&base, &limits);

        assert_eq!(resolved.h_cycles, 3);
    }

    #[test]
    fn policy_resolution_clamps_l_cycles() {
        let base = HrmConfig::default();
        let limits = HrmPolicyLimits {
            min_l_cycles: 4,
            max_l_cycles: 5,
            ..HrmPolicyLimits::default()
        };
        let policy = HrmControlPolicy {
            l_cycle_budget: BudgetValue::new(1.0, 1.0, 1.0),
            ..HrmControlPolicy::default()
        };

        let resolved = policy.resolve(&base, &limits);

        assert_eq!(resolved.l_cycles, 5);
    }

    #[test]
    fn scale_truth_maps_confident_truth_into_range() {
        let value = scale_truth_to_f32(TruthValue::new(0.5, 0.5), 0.0, 1.0, 0.0);

        assert!((value - 0.25).abs() < f32::EPSILON);
    }
}
