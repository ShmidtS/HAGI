use clifford_core::{Multivector, Rotor};
use core_types::{algebra::Cl, ids::DomainId, shape::Shape};
use hdim_model::{MultivectorBatch, TransferRegistry};
use nars_core::{Term, TruthValue};
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
