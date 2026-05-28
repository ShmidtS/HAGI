pub mod controller;
pub mod feedback;
pub mod policy;

pub use controller::{
    eval_loss_to_frequency, grad_norm_to_frequency, loss_to_frequency, HrmGoalSet,
    NarsHrmCheckpointError, NarsHrmConfig, NarsHrmController,
};
pub use feedback::{
    execution_observation_judgments, generalizes_term, train_step_judgments,
    HrmExecutionObservation, HrmTrainStepFeedback,
};
pub use policy::{
    scale_budget_to_usize, scale_truth_to_f32, HrmControlPolicy, HrmPolicyLimits,
    ResolvedHrmControl,
};
#[cfg(test)]
mod tests {
    use super::*;
    use config::HrmConfig;
    use nars_core::{BudgetValue, Term};

    struct TestTrainStepReport {
        loss: f32,
    }

    impl HrmTrainStepFeedback for TestTrainStepReport {
        fn total_loss(&self) -> f32 {
            self.loss
        }

        fn grad_norm(&self) -> f32 {
            1.0
        }

        fn eval_loss(&self) -> Option<f32> {
            None
        }

        fn should_stop(&self) -> bool {
            false
        }
    }

    fn report(loss: f32) -> TestTrainStepReport {
        TestTrainStepReport { loss }
    }

    #[test]
    fn policy_resolution_clamps_h_cycles() {
        let policy = HrmControlPolicy {
            h_cycle_budget: BudgetValue::new(1.0, 1.0, 1.0),
            ..HrmControlPolicy::default()
        };
        let limits = HrmPolicyLimits {
            min_h_cycles: 2,
            max_h_cycles: 4,
            ..HrmPolicyLimits::default()
        };

        let resolved = policy.resolve(&HrmConfig::default(), &limits);

        assert_eq!(resolved.h_cycles, 4);
    }

    #[test]
    fn policy_resolution_clamps_l_cycles() {
        let policy = HrmControlPolicy {
            l_cycle_budget: BudgetValue::new(1.0, 1.0, 1.0),
            ..HrmControlPolicy::default()
        };
        let limits = HrmPolicyLimits {
            min_l_cycles: 3,
            max_l_cycles: 6,
            ..HrmPolicyLimits::default()
        };

        let resolved = policy.resolve(&HrmConfig::default(), &limits);

        assert_eq!(resolved.l_cycles, 6);
    }

    #[test]
    fn loss_decrease_creates_low_prediction_error_judgment() {
        let high_loss = train_step_judgments(&report(2.0));
        let low_loss = train_step_judgments(&report(0.1));
        let term = Term::atom("hrm_low_prediction_error");
        let high_truth = high_loss
            .iter()
            .find(|(found, _)| *found == term)
            .unwrap()
            .1;
        let low_truth = low_loss.iter().find(|(found, _)| *found == term).unwrap().1;

        assert!(low_truth.frequency() > high_truth.frequency());
    }

    #[test]
    fn early_convergence_creates_stable_state_judgment() {
        let observation = HrmExecutionObservation {
            step: 1,
            h_cycles_used: 1,
            l_cycles_used: 1,
            converged: true,
            convergence_delta: 0.01,
            efficiency: 0.9,
            stable_state: true,
            bp_steps: 1,
        };

        let judgments = execution_observation_judgments(&observation);
        let term = Term::atom("hrm_stable_state");
        let truth = judgments
            .iter()
            .find(|(found, _)| *found == term)
            .unwrap()
            .1;

        assert!(truth.frequency() > 0.9);
    }
}
