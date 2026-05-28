use clifford_core::ProductTable;
use core_types::{algebra::AlgebraSignature, ids::DomainId};
use hdim_model::{transfer_domain, MultivectorBatch, TransferError, TransferRegistry};
use nars_core::TruthValue;

use crate::reasoner::transfer_term;
use crate::{NarsHdimConfig, NarsHdimReasoner, TransferRecommendation};

pub fn transfer_domain_reasoned<A: AlgebraSignature>(
    registry: &mut TransferRegistry<A>,
    reasoner: &mut NarsHdimReasoner,
    source: DomainId,
    target: DomainId,
    batch: &MultivectorBatch<A>,
    table: &ProductTable,
) -> Result<MultivectorBatch<A>, TransferError> {
    let _ = table;
    transfer_domain_reasoned_or_fallback(
        registry,
        Some(source),
        Some(target),
        reasoner,
        batch,
        &reasoner.config.clone(),
    )
    .map(|(batch, _)| batch)
}

pub fn transfer_domain_reasoned_or_fallback<A: AlgebraSignature>(
    registry: &mut TransferRegistry<A>,
    source_hint: Option<DomainId>,
    target_hint: Option<DomainId>,
    reasoner: &mut NarsHdimReasoner,
    g_source: &MultivectorBatch<A>,
    config: &NarsHdimConfig,
) -> Result<(MultivectorBatch<A>, TransferRecommendation), TransferError> {
    let table = ProductTable::generate(A::P, A::Q, A::R);
    let source_candidates: Vec<DomainId> =
        source_hint.map(|source| vec![source]).unwrap_or_else(|| {
            registry
                .domains
                .iter()
                .map(|domain| domain.domain_id)
                .collect()
        });
    let target_candidates: Vec<DomainId> =
        target_hint.map(|target| vec![target]).unwrap_or_else(|| {
            registry
                .domains
                .iter()
                .map(|domain| domain.domain_id)
                .collect()
        });

    if source_candidates.is_empty() {
        return Err(TransferError::MissingDomain(
            source_hint.unwrap_or(DomainId(0)),
        ));
    }
    if target_candidates.is_empty() {
        return Err(TransferError::MissingDomain(
            target_hint.unwrap_or(DomainId(0)),
        ));
    }

    let recommendation = reasoner
        .recommend_transfer(&source_candidates, &target_candidates, "transfer_domain")
        .unwrap_or_else(|| TransferRecommendation {
            source: source_hint.unwrap_or_else(|| source_candidates[0]),
            target: target_hint.unwrap_or_else(|| target_candidates[0]),
            confidence: 0.0,
            rotor_hint: None,
        });
    let use_recommendation = recommendation.confidence >= config.recommendation_threshold;
    let source = source_hint
        .or_else(|| use_recommendation.then_some(recommendation.source))
        .ok_or(TransferError::MissingDomain(DomainId(0)))?;
    let target = target_hint
        .or_else(|| use_recommendation.then_some(recommendation.target))
        .ok_or(TransferError::MissingDomain(DomainId(0)))?;

    if use_recommendation {
        if let Some(hint) = recommendation.rotor_hint.as_deref() {
            eprintln!("nars-hdim transfer hint: {hint}");
        }
    }

    transfer_domain(registry, source, target, g_source, &table).map(|batch| (batch, recommendation))
}

impl NarsHdimReasoner {
    pub fn observe_transfer_feedback(&mut self, source: DomainId, target: DomainId, success: bool) {
        let observed = if success {
            TruthValue::new(1.0, 0.9)
        } else {
            TruthValue::new(0.0, 0.9)
        };

        let revised = self
            .transfer_beliefs
            .get(&(source, target))
            .copied()
            .map(|truth| truth.revision(observed))
            .unwrap_or(observed);
        self.transfer_beliefs.insert((source, target), revised);
        let transfer = transfer_term(source, target);
        self.replace_domain_judgment(source, transfer.clone(), revised);
        self.replace_domain_judgment(target, transfer, revised);
    }
}
