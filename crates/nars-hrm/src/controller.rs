use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use config::HrmConfig;
use nars_core::{BudgetValue, Concept, Sentence, Task, Term, TruthValue};
use serde::{Deserialize, Serialize};

use crate::feedback::{
    efficient_term, eval_loss_to_frequency as feedback_eval_loss_to_frequency,
    grad_norm_to_frequency as feedback_grad_norm_to_frequency,
    loss_to_frequency as feedback_loss_to_frequency, low_prediction_error_term, stable_state_term,
    task_judgments, train_step_tasks, trainable_term, HrmTrainStepFeedback,
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
    pub decay_factor: f64,
}

impl Default for NarsHrmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            decay_factor: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NarsHrmController {
    pub concept_store: HashMap<String, Concept>,
    pub goals: HrmGoalSet,
    pub limits: HrmPolicyLimits,
    pub decay_factor: f64,
    pub default_durability: f64,
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
    concept_store: HashMap<String, Vec<SerializableJudgment>>,
    goals: SerializableHrmGoalSet,
    limits: HrmPolicyLimits,
    #[serde(default = "default_decay_factor")]
    decay_factor: f64,
    #[serde(default = "default_durability")]
    default_durability: f64,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum SerializableJudgment {
    Task {
        term: SerializableTerm,
        truth: SerializableTruthValue,
        budget: SerializableBudgetValue,
    },
    Legacy(SerializableTerm, SerializableTruthValue),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct SerializableTruthValue {
    frequency: f64,
    confidence: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct SerializableBudgetValue {
    priority: f64,
    durability: f64,
    quality: f64,
}

impl NarsHrmController {
    pub fn new(goals: HrmGoalSet, limits: HrmPolicyLimits) -> Self {
        Self {
            concept_store: HashMap::new(),
            goals,
            limits,
            decay_factor: default_decay_factor(),
            default_durability: default_durability(),
        }
    }

    pub fn with_config(goals: HrmGoalSet, limits: HrmPolicyLimits, config: NarsHrmConfig) -> Self {
        Self {
            concept_store: HashMap::new(),
            goals,
            limits,
            decay_factor: config.decay_factor,
            default_durability: default_durability(),
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
        self.decay_concept_budgets();
        let low_error = self
            .latest_truth(&self.goals.prediction_error)
            .map(|truth| truth.frequency())
            .unwrap_or(0.5);
        let stable = self
            .latest_truth(&self.goals.state_stability)
            .map(|truth| truth.frequency())
            .unwrap_or(0.5);
        let efficient = self
            .latest_truth(&self.goals.efficiency)
            .map(|truth| truth.frequency())
            .unwrap_or(0.5);
        let trainable = self
            .latest_truth(&self.goals.trainability)
            .map(|truth| truth.frequency())
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
        let tasks = train_step_tasks(report, self.default_durability);
        let judgments = task_judgments(tasks.clone());
        for task in tasks {
            self.accept_task(task);
        }
        judgments
    }

    fn accept_task(&mut self, task: Task) {
        let term = task.sentence().term().clone();
        let key = concept_key(&term);
        if let Some(existing) = self.concept_store.get(&key) {
            let task = revised_task(existing, task);
            let mut concept = Concept::new(term);
            concept.accept(task);
            self.concept_store.insert(key, concept);
        } else {
            self.concept_store
                .entry(key)
                .or_insert_with(|| Concept::new(term))
                .accept(task);
        }
    }

    fn latest_truth(&self, term: &Term) -> Option<TruthValue> {
        self.concept_store
            .get(&concept_key(term))
            .and_then(Concept::latest_belief_truth)
            .copied()
    }

    fn decay_concept_budgets(&mut self) {
        if self.decay_factor >= 1.0 {
            return;
        }
        for concept in self.concept_store.values_mut() {
            decay_concept(concept, self.decay_factor);
        }
    }
}

impl From<&NarsHrmController> for NarsHrmControllerCheckpoint {
    fn from(controller: &NarsHrmController) -> Self {
        Self {
            concept_store: controller
                .concept_store
                .iter()
                .map(|(key, concept)| {
                    (
                        key.clone(),
                        concept
                            .beliefs()
                            .iter()
                            .filter_map(|task| match task.sentence() {
                                Sentence::Judgment { term, truth, .. } => {
                                    Some(SerializableJudgment::Task {
                                        term: SerializableTerm::from(term),
                                        truth: SerializableTruthValue::from(*truth),
                                        budget: SerializableBudgetValue::from(task.budget()),
                                    })
                                }
                                _ => None,
                            })
                            .collect(),
                    )
                })
                .collect(),
            goals: SerializableHrmGoalSet::from(&controller.goals),
            limits: controller.limits,
            decay_factor: controller.decay_factor,
            default_durability: controller.default_durability,
        }
    }
}

impl From<NarsHrmControllerCheckpoint> for NarsHrmController {
    fn from(checkpoint: NarsHrmControllerCheckpoint) -> Self {
        Self {
            concept_store: checkpoint
                .concept_store
                .into_iter()
                .filter_map(|(key, judgments)| {
                    let concept = judgments_to_concept(judgments, checkpoint.default_durability);
                    (!concept.beliefs().is_empty()).then_some((key, concept))
                })
                .collect(),
            goals: HrmGoalSet::from(checkpoint.goals),
            limits: checkpoint.limits,
            decay_factor: checkpoint.decay_factor,
            default_durability: checkpoint.default_durability,
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

impl SerializableJudgment {
    fn term(&self) -> SerializableTerm {
        match self {
            Self::Task { term, .. } | Self::Legacy(term, _) => term.clone(),
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

impl From<BudgetValue> for SerializableBudgetValue {
    fn from(budget: BudgetValue) -> Self {
        Self {
            priority: budget.priority(),
            durability: budget.durability(),
            quality: budget.quality(),
        }
    }
}

impl From<SerializableBudgetValue> for BudgetValue {
    fn from(budget: SerializableBudgetValue) -> Self {
        Self::new(budget.priority, budget.durability, budget.quality)
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

fn revised_task(concept: &Concept, mut task: Task) -> Task {
    if let (Some(existing_truth), Sentence::Judgment { term, truth, stamp }) = (
        concept.latest_belief_truth().copied(),
        task.sentence().clone(),
    ) {
        task = Task::new(
            Sentence::Judgment {
                term,
                truth: existing_truth.revision(truth),
                stamp,
            },
            task.budget(),
        );
    }
    if let Some(existing) = concept.beliefs().iter().next() {
        task.merge_budget(existing.budget());
    }
    task
}

fn decay_concept(concept: &mut Concept, factor: f64) {
    if factor >= 1.0 {
        return;
    }
    let term = concept.term().clone();
    let belief_cap = concept.beliefs().capacity_limit().unwrap_or(usize::MAX);
    let desire_cap = concept.desires().capacity_limit().unwrap_or(usize::MAX);
    let question_cap = concept.questions().capacity_limit().unwrap_or(usize::MAX);
    let mut decayed =
        if belief_cap == usize::MAX && desire_cap == usize::MAX && question_cap == usize::MAX {
            Concept::new(term)
        } else {
            Concept::with_capacity(term, belief_cap, desire_cap, question_cap)
        };

    for task in concept
        .beliefs()
        .iter()
        .chain(concept.desires().iter())
        .chain(concept.questions().iter())
    {
        let mut task = task.clone();
        task.decay_budget(factor);
        decayed.accept(task);
    }
    *concept = decayed;
}

fn judgments_to_concept(judgments: Vec<SerializableJudgment>, default_durability: f64) -> Concept {
    let mut judgments = judgments.into_iter();
    let Some(first) = judgments.next() else {
        return Concept::new(Term::atom("empty"));
    };
    let first_term = Term::from(first.term());
    let mut concept = Concept::new(first_term.clone());
    accept_checkpoint_judgment(&mut concept, first, default_durability);
    for judgment in judgments {
        accept_checkpoint_judgment(&mut concept, judgment, default_durability);
    }
    concept
}

fn accept_checkpoint_judgment(
    concept: &mut Concept,
    judgment: SerializableJudgment,
    default_durability: f64,
) {
    let (term, truth, budget) = match judgment {
        SerializableJudgment::Task {
            term,
            truth,
            budget,
        } => (
            Term::from(term),
            TruthValue::from(truth),
            BudgetValue::from(budget),
        ),
        SerializableJudgment::Legacy(term, truth) => {
            let truth = TruthValue::from(truth);
            (
                Term::from(term),
                truth,
                BudgetValue::new(truth.confidence(), default_durability, truth.confidence()),
            )
        }
    };
    let task = Task::new(
        Sentence::Judgment {
            term,
            truth,
            stamp: 0,
        },
        budget,
    );
    concept.accept(task);
}

fn default_decay_factor() -> f64 {
    1.0
}

fn default_durability() -> f64 {
    0.7
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

    #[test]
    fn end_step_inserts_tasks_into_concepts() {
        struct Report;
        impl HrmTrainStepFeedback for Report {
            fn total_loss(&self) -> f32 {
                0.25
            }
            fn grad_norm(&self) -> f32 {
                0.5
            }
            fn eval_loss(&self) -> Option<f32> {
                None
            }
            fn should_stop(&self) -> bool {
                false
            }
        }
        let mut controller = NarsHrmController::default();

        controller.end_step(&Report);

        let concept = controller
            .concept_store
            .get(&concept_key(&low_prediction_error_term()))
            .unwrap();
        assert_eq!(concept.beliefs().len(), 1);
        assert!(concept.latest_belief_truth().is_some());
    }

    #[test]
    fn begin_step_reads_from_concept_truth() {
        let mut controller = NarsHrmController::default();
        let term = controller.goals.state_stability.clone();
        let mut concept = Concept::new(term.clone());
        concept.accept(Task::new(
            Sentence::Judgment {
                term: term.clone(),
                truth: TruthValue::new(1.0, 0.9),
                stamp: 0,
            },
            BudgetValue::new(1.0, 0.7, 0.9),
        ));
        controller.concept_store.insert(concept_key(&term), concept);

        let policy = controller.begin_step(0, &HrmConfig::default());

        assert_eq!(policy.h_cycle_budget.priority(), 0.1);
    }

    #[test]
    fn budget_decay_never_increases_priority_or_durability() {
        let mut controller = NarsHrmController {
            decay_factor: 0.5,
            ..NarsHrmController::default()
        };
        let term = controller.goals.prediction_error.clone();
        let mut concept = Concept::new(term.clone());
        concept.accept(Task::new(
            Sentence::Judgment {
                term: term.clone(),
                truth: TruthValue::new(0.7, 0.8),
                stamp: 0,
            },
            BudgetValue::new(0.8, 0.6, 0.8),
        ));
        controller.concept_store.insert(concept_key(&term), concept);
        let before = controller
            .concept_store
            .get(&concept_key(&term))
            .unwrap()
            .beliefs()
            .iter()
            .next()
            .unwrap()
            .budget();

        controller.begin_step(0, &HrmConfig::default());
        let after = controller
            .concept_store
            .get(&concept_key(&term))
            .unwrap()
            .beliefs()
            .iter()
            .next()
            .unwrap()
            .budget();

        assert!(after.priority() <= before.priority());
        assert!(after.durability() <= before.durability());
    }
}
