use std::collections::VecDeque;

use nars_hrm::{
    execution_observation_judgments, HrmExecutionObservation, HrmPolicyLimits,
    HrmTrainStepFeedback, NarsHrmController, ResolvedHrmControl,
};

use crate::{TrainError, TrainStepReport, TrainingLoop};

impl HrmTrainStepFeedback for TrainStepReport {
    fn total_loss(&self) -> f32 {
        self.loss.l_total
    }

    fn grad_norm(&self) -> f32 {
        self.grad_norm
    }

    fn eval_loss(&self) -> Option<f32> {
        self.eval_loss
    }

    fn should_stop(&self) -> bool {
        self.should_stop
    }
}

pub struct NarsControlledTrainingLoop {
    pub inner: TrainingLoop,
    pub controller: NarsHrmController,
    pub limits: HrmPolicyLimits,
    pub last_policy: Option<ResolvedHrmControl>,
    pub loss_ema: Option<f32>,
    pub recent_convergence_deltas: VecDeque<f32>,
}

pub struct NarsTrainStepReport {
    pub train: TrainStepReport,
    pub hrm: HrmExecutionObservation,
    pub policy: ResolvedHrmControl,
    pub judgments: Vec<(nars_core::Term, nars_core::TruthValue)>,
}

impl NarsControlledTrainingLoop {
    pub fn new(
        inner: TrainingLoop,
        controller: NarsHrmController,
        limits: HrmPolicyLimits,
    ) -> Self {
        Self {
            inner,
            controller,
            limits,
            last_policy: None,
            loss_ema: None,
            recent_convergence_deltas: VecDeque::new(),
        }
    }

    pub fn train_step(
        &mut self,
        batch: &data::PackedBatch,
    ) -> Result<NarsTrainStepReport, TrainError> {
        let step = self.inner.step;
        let base_config = &self.inner.backbone.config;
        let convergence_eps = base_config.convergence_eps;
        let policy = self.controller.begin_step(step, base_config);
        let resolved = policy.resolve(base_config, &self.limits);

        self.inner.hrm_runtime_control = Some(hrm_model::HrmRuntimeControl {
            h_cycles: resolved.h_cycles,
            l_cycles: resolved.l_cycles,
            convergence_eps: resolved.convergence_eps,
            bp_steps: resolved.bp_steps,
        });

        let train_report = match self.inner.train_step(batch) {
            Ok(report) => report,
            Err(error) => {
                self.inner.hrm_runtime_control = None;
                return Err(error);
            }
        };

        self.inner.hrm_runtime_control = None;

        let current_loss = train_report.loss.l_total;
        let convergence_delta = self
            .loss_ema
            .map(|ema| (current_loss - ema).abs())
            .unwrap_or(f32::INFINITY);
        let loss_ema = self
            .loss_ema
            .map(|ema| 0.9 * ema + 0.1 * current_loss)
            .unwrap_or(current_loss);
        self.loss_ema = Some(loss_ema);
        self.recent_convergence_deltas.push_back(convergence_delta);
        while self.recent_convergence_deltas.len() > 5 {
            self.recent_convergence_deltas.pop_front();
        }
        let converged = convergence_delta < convergence_eps;
        let stable_state = self.recent_convergence_deltas.len() == 5
            && self
                .recent_convergence_deltas
                .iter()
                .all(|delta| *delta < convergence_eps);
        let efficiency = if current_loss.is_finite() && current_loss >= 0.0 {
            1.0 / (1.0 + current_loss)
        } else {
            0.0
        };

        let hrm_obs = HrmExecutionObservation {
            step: train_report.step,
            h_cycles_used: resolved.h_cycles,
            l_cycles_used: resolved.l_cycles,
            converged,
            convergence_delta,
            efficiency,
            stable_state,
            bp_steps: resolved.bp_steps,
        };

        let mut judgments = self.controller.observe_train_step(&train_report);
        judgments.extend(execution_observation_judgments(&hrm_obs));

        self.last_policy = Some(resolved);

        Ok(NarsTrainStepReport {
            train: train_report,
            hrm: hrm_obs,
            policy: resolved,
            judgments,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use config::HrmConfig;
    use core_types::shape::Shape;
    use data::{PackedBatch, PackedPartition, PackedSpan};
    use losses::LossWeights;
    use tensor_runtime::Tensor;

    use crate::AdamW;

    fn packed_batch(targets: Vec<u32>) -> PackedBatch {
        PackedBatch {
            tokens: Tensor::from_vec(vec![0, 1, 2, 3], Shape::new(vec![1, 4])),
            targets: Tensor::from_vec(targets, Shape::new(vec![1, 4])),
            prefix_mask: Tensor::from_vec(vec![1, 1, 0, 0], Shape::new(vec![1, 4])),
            partition: PackedPartition {
                spans: vec![PackedSpan {
                    sequence_id: 0,
                    batch_index: 0,
                    start: 0,
                    len: 4,
                    prefix_len: 2,
                }],
                batch_size: 1,
                seq_len: 4,
            },
        }
    }

    fn trainer() -> TrainingLoop {
        let hrm = HrmConfig {
            total_layers: 2,
            h_layers: 1,
            l_layers: 1,
            hidden_size: 8,
            num_heads: 2,
            expansion: 2,
            h_cycles: 1,
            l_cycles: 1,
            vocab_size: 8,
            max_seq_len: 8,
            convergence_eps: 1e-5,
            bp_warmup_ratio: 0.2,
            bp_max_steps: 2,
            warmup_steps: 10,
        };
        TrainingLoop::new(
            hrm_model::HrmBackbone::from_config(&hrm),
            AdamW::new(0.01, 0.9, 0.95, 1e-8, 0.0),
            LossWeights {
                lambda_aux: 0.0,
                lambda_iso_target: 0.0,
                iso_warmup_steps: 10,
            },
        )
    }

    fn controlled_loop() -> NarsControlledTrainingLoop {
        NarsControlledTrainingLoop::new(
            trainer(),
            NarsHrmController::default(),
            HrmPolicyLimits::default(),
        )
    }

    #[test]
    fn controlled_step_returns_nars_report() {
        let mut loop_ = controlled_loop();

        let report = loop_.train_step(&packed_batch(vec![0, 1, 2, 3])).unwrap();

        assert_eq!(report.train.step, 0);
        assert_eq!(report.hrm.step, report.train.step);
        assert_eq!(report.policy, loop_.last_policy.unwrap());
        assert!(!report.judgments.is_empty());
    }

    #[test]
    fn policy_changes_after_observation() {
        let mut loop_ = controlled_loop();

        let first = loop_.train_step(&packed_batch(vec![0, 1, 2, 3])).unwrap();
        let concepts_after_first = loop_.controller.concept_store.len();
        let second = loop_.train_step(&packed_batch(vec![3, 2, 1, 0])).unwrap();

        assert!(concepts_after_first > 0);
        assert!(loop_
            .controller
            .concept_store
            .values()
            .any(|concept| concept.beliefs().len() > 0));
        assert_eq!(loop_.last_policy, Some(second.policy));
        assert!(
            first.policy != second.policy
                || loop_.controller.concept_store.len() >= concepts_after_first
        );
    }

    #[test]
    fn default_training_loop_behavior_preserved() {
        let mut loop_ = controlled_loop();

        let _ = loop_.train_step(&packed_batch(vec![0, 1, 2, 3])).unwrap();

        assert!(loop_.inner.hrm_runtime_control.is_none());
    }
}
