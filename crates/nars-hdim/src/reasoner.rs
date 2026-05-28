use std::collections::HashMap;

use core_types::ids::DomainId;
use nars_core::{Term, TruthValue};

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
    pub domain_concepts: HashMap<DomainId, Vec<(Term, TruthValue)>>,
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
        self.domain_concepts
            .entry(domain)
            .or_default()
            .push((term, tv));
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
                let concept_bonus = self
                    .domain_concepts
                    .get(&target)
                    .into_iter()
                    .flat_map(|concepts| concepts.iter())
                    .map(|(_, truth)| (truth.frequency() * truth.confidence()) as f32)
                    .fold(0.0, f32::max)
                    * 0.05;
                let score = confidence + concept_bonus;
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
