use clifford_core::{Multivector, Rotor};
use core_types::{algebra::Cl, ids::DomainId, shape::Shape};
use hdim_model::{MultivectorBatch, TransferRegistry};
use nars_core::{Sentence, Term, TruthValue};
use nars_hdim::{transfer_domain_reasoned_or_fallback, NarsHdimConfig, NarsHdimReasoner};
use tensor_runtime::Tensor;

type Cl3 = Cl<3, 0, 0>;

fn batch() -> MultivectorBatch<Cl3> {
    MultivectorBatch::new(
        Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Shape::new(vec![1, 1, 1, 8]),
        ),
        1,
    )
}

fn registry() -> TransferRegistry<Cl3> {
    let mut registry = TransferRegistry::<Cl3>::new();
    registry.register_domain(DomainId(0), Rotor::unit(Multivector::<Cl3>::scalar_one()));
    registry.register_domain(DomainId(1), Rotor::unit(Multivector::<Cl3>::scalar_one()));
    registry.register_domain(DomainId(2), Rotor::unit(Multivector::<Cl3>::scalar_one()));
    registry
}

#[test]
fn fallback_with_none_hints_uses_reasoner_recommendation() {
    let mut registry = registry();
    let mut reasoner = NarsHdimReasoner::default();
    reasoner
        .transfer_beliefs
        .insert((DomainId(1), DomainId(2)), TruthValue::new(1.0, 0.9));

    let (_, recommendation) = transfer_domain_reasoned_or_fallback(
        &mut registry,
        None,
        None,
        &mut reasoner,
        &batch(),
        &NarsHdimConfig::default(),
    )
    .expect("reasoned transfer should use available recommendation");

    assert_eq!(recommendation.source, DomainId(1));
    assert_eq!(recommendation.target, DomainId(2));
    assert!(recommendation.confidence >= NarsHdimConfig::default().recommendation_threshold);
}

#[test]
fn fallback_with_explicit_hints_ignores_low_confidence_recommendation() {
    let mut registry = registry();
    let mut reasoner = NarsHdimReasoner::default();
    reasoner
        .transfer_beliefs
        .insert((DomainId(0), DomainId(1)), TruthValue::new(0.2, 0.2));

    let (_, recommendation) = transfer_domain_reasoned_or_fallback(
        &mut registry,
        Some(DomainId(0)),
        Some(DomainId(1)),
        &mut reasoner,
        &batch(),
        &NarsHdimConfig::default(),
    )
    .expect("explicit low-confidence transfer should still fall back to hinted pair");

    assert_eq!(recommendation.source, DomainId(0));
    assert_eq!(recommendation.target, DomainId(1));
    assert!(recommendation.confidence < NarsHdimConfig::default().recommendation_threshold);
}

#[test]
fn empty_registry_returns_error_without_panic() {
    let mut registry = TransferRegistry::<Cl3>::new();
    let mut reasoner = NarsHdimReasoner::default();

    let result = transfer_domain_reasoned_or_fallback(
        &mut registry,
        None,
        None,
        &mut reasoner,
        &batch(),
        &NarsHdimConfig::default(),
    );

    assert!(result.is_err());
}

#[test]
fn low_confidence_recommendation_without_hints_returns_error() {
    let mut registry = registry();
    let mut reasoner = NarsHdimReasoner::default();
    reasoner
        .transfer_beliefs
        .insert((DomainId(1), DomainId(2)), TruthValue::new(0.1, 0.1));

    let result = transfer_domain_reasoned_or_fallback(
        &mut registry,
        None,
        None,
        &mut reasoner,
        &batch(),
        &NarsHdimConfig::default(),
    );

    assert!(result.is_err());
}

#[test]
fn update_domain_concept_affects_target_ranking() {
    let source = DomainId(0);
    let target_a = DomainId(1);
    let target_b = DomainId(2);
    let mut reasoner = NarsHdimReasoner::default();
    reasoner
        .transfer_beliefs
        .insert((source, target_a), TruthValue::new(0.8, 0.8));
    reasoner
        .transfer_beliefs
        .insert((source, target_b), TruthValue::new(0.82, 0.75));
    reasoner.update_domain_concept(target_a, Term::atom("target-a"), TruthValue::new(1.0, 1.0));

    let recommendation = reasoner
        .recommend_transfer(&[source], &[target_a, target_b], "test")
        .expect("concept bonus should preserve a transfer recommendation");

    assert_eq!(recommendation.source, source);
    assert_eq!(recommendation.target, target_a);
}

#[test]
fn transfer_feedback_creates_concept_tasks_for_both_domains() {
    let source = DomainId(0);
    let target = DomainId(1);
    let mut reasoner = NarsHdimReasoner::default();

    reasoner.observe_transfer_feedback(source, target, true);

    let source_concept = reasoner
        .domain_concepts
        .get(&source)
        .expect("source feedback should create a source concept");
    let target_concept = reasoner
        .domain_concepts
        .get(&target)
        .expect("target feedback should create a target concept");
    let source_task = source_concept
        .beliefs()
        .iter()
        .next()
        .expect("source concept should receive transfer task");
    let target_task = target_concept
        .beliefs()
        .iter()
        .next()
        .expect("target concept should receive transfer task");

    assert_eq!(source_task.sentence(), target_task.sentence());
    assert!(
        matches!(source_task.sentence(), Sentence::Judgment { term, .. } if matches!(term, Term::Compound(operator, terms) if operator == "transfer" && terms.len() == 2))
    );
}

#[test]
fn transfer_feedback_respects_concept_bag_capacity() {
    let mut reasoner = NarsHdimReasoner::default();
    let source = DomainId(0);

    for target in 1..20 {
        reasoner.observe_transfer_feedback(source, DomainId(target), target % 2 == 0);
    }

    let source_concept = reasoner
        .domain_concepts
        .get(&source)
        .expect("feedback should create bounded source concept");
    assert_eq!(source_concept.beliefs().capacity_limit(), Some(16));
    assert_eq!(source_concept.beliefs().len(), 16);
}

#[test]
fn repeated_transfer_feedback_uses_revision_and_syncs_cache() {
    let source = DomainId(0);
    let target = DomainId(1);
    let mut reasoner = NarsHdimReasoner::default();

    reasoner.observe_transfer_feedback(source, target, true);
    reasoner.observe_transfer_feedback(source, target, false);

    let cached = reasoner
        .transfer_beliefs
        .get(&(source, target))
        .copied()
        .expect("transfer cache should stay populated");
    let concept_truth = reasoner
        .domain_concepts
        .get(&source)
        .and_then(|concept| concept.latest_belief_truth())
        .copied()
        .expect("source concept should keep latest revised belief");

    assert_eq!(cached, concept_truth);
    assert!((cached.frequency() - 0.5).abs() < 1e-12);
}
