use std::collections::HashMap;

use core_types::ids::DomainId;
use nars_core::{BudgetValue, Concept, Sentence, Task, Term, TruthValue};

pub const CONCEPT_BELIEF_CAPACITY: usize = 16;
const CONCEPT_DESIRE_CAPACITY: usize = 4;
const CONCEPT_QUESTION_CAPACITY: usize = 4;

#[derive(Debug, Clone, PartialEq)]
pub struct NarsHdimConfig {
    pub recommendation_threshold: f32,
}

impl Default for NarsHdimConfig {
    fn default() -> Self {
        Self {
            recommendation_threshold: 0.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransferRecommendation {
    pub source: DomainId,
    pub target: DomainId,
    pub confidence: f32,
    pub rotor_hint: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NarsHdimReasoner {
    pub domain_concepts: HashMap<DomainId, Concept>,
    pub transfer_beliefs: HashMap<(DomainId, DomainId), TruthValue>,
    pub config: NarsHdimConfig,
}

impl Default for NarsHdimReasoner {
    fn default() -> Self {
        Self::new(NarsHdimConfig::default())
    }
}

impl NarsHdimReasoner {
    pub fn new(config: NarsHdimConfig) -> Self {
        Self {
            domain_concepts: HashMap::new(),
            transfer_beliefs: HashMap::new(),
            config,
        }
    }

    pub fn update_domain_concept(&mut self, domain: DomainId, term: Term, tv: TruthValue) {
        self.accept_domain_judgment(domain, term, tv);
    }

    pub(crate) fn accept_domain_judgment(
        &mut self,
        domain: DomainId,
        term: Term,
        truth: TruthValue,
    ) {
        let task = transfer_task(term, truth);
        self.concept_for_domain(domain).accept(task);
    }

    pub(crate) fn replace_domain_judgment(
        &mut self,
        domain: DomainId,
        term: Term,
        truth: TruthValue,
    ) {
        let mut tasks: Vec<Task> = self
            .domain_concepts
            .get(&domain)
            .into_iter()
            .flat_map(|concept| {
                concept
                    .beliefs()
                    .iter()
                    .chain(concept.desires().iter())
                    .chain(concept.questions().iter())
            })
            .filter(|task| task.sentence().term() != &term)
            .cloned()
            .collect();
        tasks.push(transfer_task(term, truth));

        let mut concept = Concept::with_capacity(
            domain_term(domain),
            CONCEPT_BELIEF_CAPACITY,
            CONCEPT_DESIRE_CAPACITY,
            CONCEPT_QUESTION_CAPACITY,
        );
        for task in tasks {
            concept.accept(task);
        }
        self.domain_concepts.insert(domain, concept);
    }

    pub(crate) fn concept_for_domain(&mut self, domain: DomainId) -> &mut Concept {
        self.domain_concepts.entry(domain).or_insert_with(|| {
            Concept::with_capacity(
                domain_term(domain),
                CONCEPT_BELIEF_CAPACITY,
                CONCEPT_DESIRE_CAPACITY,
                CONCEPT_QUESTION_CAPACITY,
            )
        })
    }

    pub fn recommend_transfer(
        &mut self,
        source_candidates: &[DomainId],
        target_candidates: &[DomainId],
        goal: &str,
    ) -> Option<TransferRecommendation> {
        let mut best: Option<(TransferRecommendation, f32)> = None;

        for &source in source_candidates {
            for &target in target_candidates {
                let Some(truth) = self.transfer_beliefs.get(&(source, target)) else {
                    continue;
                };
                let confidence = (truth.frequency() * truth.confidence()) as f32;
                let concept_priority = self
                    .domain_concepts
                    .get(&target)
                    .and_then(|concept| concept.beliefs().peek_priority(0))
                    .unwrap_or(0.0) as f32;
                let concept_bonus = self
                    .domain_concepts
                    .get(&target)
                    .and_then(|concept| concept.latest_belief_truth())
                    .map(|truth| (truth.frequency() * truth.confidence()) as f32)
                    .unwrap_or(0.0)
                    * 0.05;
                let score = confidence + concept_bonus + concept_priority * 0.01;
                if best
                    .as_ref()
                    .map_or(true, |(_, best_score)| score > *best_score)
                {
                    best = Some((
                        TransferRecommendation {
                            source,
                            target,
                            confidence,
                            rotor_hint: Some(format!("nars:{goal}:{source:?}->{target:?}")),
                        },
                        score,
                    ));
                }
            }
        }

        best.map(|(recommendation, _)| recommendation)
    }
}

pub(crate) fn domain_term(domain: DomainId) -> Term {
    Term::atom(format!("domain:{domain:?}"))
}

pub(crate) fn transfer_term(source: DomainId, target: DomainId) -> Term {
    Term::compound("transfer", vec![domain_term(source), domain_term(target)])
}

pub(crate) fn transfer_task(term: Term, truth: TruthValue) -> Task {
    Task::new(
        Sentence::judgment(term, truth, 0),
        BudgetValue::new(
            truth.frequency() * truth.confidence(),
            truth.confidence(),
            truth.frequency(),
        ),
    )
}
