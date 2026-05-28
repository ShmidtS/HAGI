use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use config::HrmConfig;
use nars_core::{BudgetValue, Term, TruthValue};
use serde::{Deserialize, Serialize};

use crate::feedback::{
    efficient_term, eval_loss_to_frequency as feedback_eval_loss_to_frequency,
    grad_norm_to_frequency as feedback_grad_norm_to_frequency,
    loss_to_frequency as feedback_loss_to_frequency, low_prediction_error_term, stable_state_term,
    train_step_judgments, trainable_term, HrmTrainStepFeedback,
};
use crate::policy::{HrmControlPolicy, HrmPolicyLimits};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HrmGoalSet {
    pub prediction_error: Term,
    pub state_stability: Term,
    pub efficiency: Term,
    pub trainability: Term,
}

impl Default for HrmGoalSet {
    fn default() -> Self {
        Self {
            prediction_error: low_prediction_error_term(),
            state_stability: stable_state_term(),
            efficiency: efficient_term(),
            trainability: trainable_term(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NarsHrmConfig {
    pub enabled: bool,
}

impl Default for NarsHrmConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Debug, Clone)]
pub struct NarsHrmController {
    pub concept_store: HashMap<String, Vec<(Term, TruthValue)>>,
    pub goals: HrmGoalSet,
    pub limits: HrmPolicyLimits,
}

#[derive(Debug, thiserror::Error)]
pub enum NarsHrmCheckpointError {
    #[error("checkpoint I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("checkpoint JSON failed: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NarsHrmControllerCheckpoint {
    concept_store: HashMap<String, Vec<(SerializableTerm, SerializableTruthValue)>>,
    goals: SerializableHrmGoalSet,
    limits: HrmPolicyLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableHrmGoalSet {
    prediction_error: SerializableTerm,
    state_stability: SerializableTerm,
    efficiency: SerializableTerm,
    trainability: SerializableTerm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum SerializableTerm {
    Atom(String),
    Compound(String, Vec<SerializableTerm>),
    Variable(String),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct SerializableTruthValue {
    frequency: f64,
    confidence: f64,
}

impl NarsHrmController {
    pub fn new(goals: HrmGoalSet, limits: HrmPolicyLimits) -> Self {
        Self {
            concept_store: HashMap::new(),
            goals,
            limits,
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), NarsHrmCheckpointError> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, &NarsHrmControllerCheckpoint::from(self))?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, NarsHrmCheckpointError> {
        let file = File::open(path)?;
        let checkpoint: NarsHrmControllerCheckpoint = serde_json::from_reader(file)?;
        Ok(checkpoint.into())
    }

    pub fn begin_step(&mut self, step: usize, base_config: &HrmConfig) -> HrmControlPolicy {
        let low_error = self
            .latest_frequency(&self.goals.prediction_error)
            .unwrap_or(0.5);
        let stable = self
            .latest_frequency(&self.goals.state_stability)
            .unwrap_or(0.5);
        let efficient = self.latest_frequency(&self.goals.efficiency).unwrap_or(0.5);
        let trainable = self
            .latest_frequency(&self.goals.trainability)
            .unwrap_or(0.5);
        let warmup = if step < base_config.warmup_steps {
            0.1
        } else {
            0.0
        };

        HrmControlPolicy {
            h_cycle_budget: BudgetValue::new(1.0 - stable + warmup, 0.7, 1.0 - low_error),
            l_cycle_budget: BudgetValue::new(1.0 - low_error + warmup, 0.7, 1.0 - efficient),
            convergence_threshold: TruthValue::new(stable, 0.8),
            bp_depth_policy: TruthValue::new(1.0 - trainable + warmup, 0.8),
        }
    }

    pub fn observe_train_step(
        &mut self,
        report: &impl HrmTrainStepFeedback,
    ) -> Vec<(Term, TruthValue)> {
        self.end_step(report)
    }

    pub fn end_step(&mut self, report: &impl HrmTrainStepFeedback) -> Vec<(Term, TruthValue)> {
        let judgments = train_step_judgments(report);
        for (term, truth) in &judgments {
            self.concept_store
                .entry(concept_key(term))
                .or_default()
                .push((term.clone(), *truth));
        }
        judgments
    }

    fn latest_frequency(&self, term: &Term) -> Option<f64> {
        self.concept_store
            .get(&concept_key(term))
            .and_then(|judgments| judgments.last())
            .map(|(_, truth)| truth.frequency())
    }
}

impl From<&NarsHrmController> for NarsHrmControllerCheckpoint {
    fn from(controller: &NarsHrmController) -> Self {
        Self {
            concept_store: controller
                .concept_store
                .iter()
                .map(|(key, judgments)| {
                    (
                        key.clone(),
                        judgments
                            .iter()
                            .map(|(term, truth)| {
                                (
                                    SerializableTerm::from(term),
                                    SerializableTruthValue::from(*truth),
                                )
                            })
                            .collect(),
                    )
                })
                .collect(),
            goals: SerializableHrmGoalSet::from(&controller.goals),
            limits: controller.limits,
        }
    }
}

impl From<NarsHrmControllerCheckpoint> for NarsHrmController {
    fn from(checkpoint: NarsHrmControllerCheckpoint) -> Self {
        Self {
            concept_store: checkpoint
                .concept_store
                .into_iter()
                .map(|(key, judgments)| {
                    (
                        key,
                        judgments
                            .into_iter()
                            .map(|(term, truth)| (Term::from(term), TruthValue::from(truth)))
                            .collect(),
                    )
                })
                .collect(),
            goals: HrmGoalSet::from(checkpoint.goals),
            limits: checkpoint.limits,
        }
    }
}

impl From<&HrmGoalSet> for SerializableHrmGoalSet {
    fn from(goals: &HrmGoalSet) -> Self {
        Self {
            prediction_error: SerializableTerm::from(&goals.prediction_error),
            state_stability: SerializableTerm::from(&goals.state_stability),
            efficiency: SerializableTerm::from(&goals.efficiency),
            trainability: SerializableTerm::from(&goals.trainability),
        }
    }
}

impl From<SerializableHrmGoalSet> for HrmGoalSet {
    fn from(goals: SerializableHrmGoalSet) -> Self {
        Self {
            prediction_error: Term::from(goals.prediction_error),
            state_stability: Term::from(goals.state_stability),
            efficiency: Term::from(goals.efficiency),
            trainability: Term::from(goals.trainability),
        }
    }
}

impl From<&Term> for SerializableTerm {
    fn from(term: &Term) -> Self {
        match term {
            Term::Atom(name) => Self::Atom(name.clone()),
            Term::Compound(operator, terms) => Self::Compound(
                operator.clone(),
                terms.iter().map(SerializableTerm::from).collect(),
            ),
            Term::Variable(name) => Self::Variable(name.clone()),
        }
    }
}

impl From<SerializableTerm> for Term {
    fn from(term: SerializableTerm) -> Self {
        match term {
            SerializableTerm::Atom(name) => Self::Atom(name),
            SerializableTerm::Compound(operator, terms) => {
                Self::Compound(operator, terms.into_iter().map(Term::from).collect())
            }
            SerializableTerm::Variable(name) => Self::Variable(name),
        }
    }
}

impl From<TruthValue> for SerializableTruthValue {
    fn from(truth: TruthValue) -> Self {
        Self {
            frequency: truth.frequency(),
            confidence: truth.confidence(),
        }
    }
}

impl From<SerializableTruthValue> for TruthValue {
    fn from(truth: SerializableTruthValue) -> Self {
        Self::new(truth.frequency, truth.confidence)
    }
}

impl Default for NarsHrmController {
    fn default() -> Self {
        Self::new(HrmGoalSet::default(), HrmPolicyLimits::default())
    }
}

pub fn loss_to_frequency(loss: f32) -> f64 {
    feedback_loss_to_frequency(loss)
}

pub fn grad_norm_to_frequency(grad_norm: f32) -> f64 {
    feedback_grad_norm_to_frequency(grad_norm)
}

pub fn eval_loss_to_frequency(eval_loss: f32) -> f64 {
    feedback_eval_loss_to_frequency(eval_loss)
}

fn concept_key(term: &Term) -> String {
    format!("{term:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn begin_step_uses_default_policy_without_observations() {
        let mut controller = NarsHrmController::default();
        let policy = controller.begin_step(0, &HrmConfig::default());

        assert!(policy.h_cycle_budget.priority() > 0.0);
        assert!(policy.l_cycle_budget.priority() > 0.0);
    }

    #[test]
    fn helper_frequencies_are_bounded() {
        assert!((0.0..=1.0).contains(&loss_to_frequency(0.5)));
        assert!((0.0..=1.0).contains(&grad_norm_to_frequency(2.0)));
        assert!((0.0..=1.0).contains(&eval_loss_to_frequency(0.5)));
    }
}
