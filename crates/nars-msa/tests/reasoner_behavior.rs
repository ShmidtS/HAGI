use core_types::shape::Shape;
use msa_adapter::{MemorySlot, RoutingQueryView, SlotRegistry};
use nars_core::{BudgetValue, Concept, Sentence, Task, Term, TruthValue};
use nars_msa::{
    compute_reward_from_retrieval_outcome, route_top_k_with_nars, NarsMsaReasoner, NarsRoutePolicy,
    ScoreBlend,
};
use tensor_runtime::Tensor;

fn make_slot(id: usize, key: Vec<f32>, timestamp: usize) -> MemorySlot {
    let dim = key.len();
    MemorySlot::new(
        id,
        Tensor::from_vec(key, Shape::new(vec![dim])),
        Tensor::from_vec(vec![0.0; dim], Shape::new(vec![dim])),
        timestamp,
        format!("slot_{id}"),
    )
}

fn make_registry() -> SlotRegistry {
    let mut registry = SlotRegistry::new();
    registry.register(make_slot(0, vec![1.0, 0.0], 0));
    registry.register(make_slot(1, vec![0.9, 0.0], 0));
    registry
}

fn slot_feedback_term(slot_id: u16) -> Term {
    Term::atom(format!("slot_{slot_id}_route_feedback"))
}

fn concept_with_truth(slot_id: u16, truth: TruthValue) -> Concept {
    let term = slot_feedback_term(slot_id);
    let mut concept = Concept::with_capacity(term.clone(), 1, 0, 0);
    concept.accept(Task::new(
        Sentence::judgment(term, truth, 0),
        BudgetValue::new(1.0, truth.confidence(), truth.confidence()),
    ));
    concept
}

#[test]
fn default_policy_weights_are_blended_and_change_dot_ranking() {
    let registry = make_registry();
    let mut reasoner = NarsMsaReasoner::default();
    reasoner
        .slot_concepts
        .insert(1, concept_with_truth(1, TruthValue::new(1.0, 1.0)));

    let selection = route_top_k_with_nars(
        &registry,
        RoutingQueryView {
            data: &[1.0, 0.0],
            dim: 2,
        },
        &mut reasoner,
        &NarsRoutePolicy::default(),
        0,
    )
    .unwrap();

    let blend = ScoreBlend::default();
    assert!(blend.truth_weight > 0.0);
    assert!(blend.recency_weight > 0.0);
    assert_eq!(selection.slot_ids.as_slice(), &[1]);
    assert_ne!(selection.raw_scores.as_slice(), &[1.0]);
}

#[test]
fn blended_negative_scores_normalize_to_probability_weights() {
    let mut registry = SlotRegistry::new();
    registry.register(make_slot(0, vec![-3.0], 0));
    registry.register(make_slot(1, vec![-1.0], 0));
    registry.register(make_slot(2, vec![2.0], 0));

    let mut reasoner = NarsMsaReasoner::default();
    let selection = route_top_k_with_nars(
        &registry,
        RoutingQueryView {
            data: &[1.0],
            dim: 1,
        },
        &mut reasoner,
        &NarsRoutePolicy::FixedTopK {
            top_k: 3,
            blend: ScoreBlend {
                dot_weight: 1.0,
                truth_weight: 0.0,
                recency_weight: 1.0,
            },
        },
        0,
    )
    .unwrap();

    assert_eq!(selection.raw_scores.as_slice(), &[3.0, 0.0, -2.0]);
    assert!(selection
        .normalized_weights
        .iter()
        .all(|weight| (0.0..=1.0).contains(weight)));
    let sum: f32 = selection.normalized_weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn repeated_positive_feedback_revises_core_belief_task() {
    let mut reasoner = NarsMsaReasoner::default();
    reasoner.observe_route_feedback(3, 0.5, 1);
    let first = *reasoner
        .slot_concepts
        .get(&3)
        .unwrap()
        .latest_belief_truth()
        .unwrap();

    reasoner.observe_route_feedback(3, 1.0, 2);
    let concept = reasoner.slot_concepts.get(&3).unwrap();
    let second = *concept.latest_belief_truth().unwrap();

    assert_eq!(concept.beliefs().len(), 1);
    assert_eq!(concept.beliefs().capacity_limit(), Some(1));
    assert_eq!(second, first.revision(TruthValue::new(1.0, 0.9)));
    assert!(second.frequency() > first.frequency());
}

#[test]
fn retrieval_outcome_reward_handles_edges() {
    assert_eq!(compute_reward_from_retrieval_outcome(0, 0), -0.3);
    assert_eq!(compute_reward_from_retrieval_outcome(0, 3), 0.0);
    assert_eq!(compute_reward_from_retrieval_outcome(3, 3), 0.5);
    assert_eq!(compute_reward_from_retrieval_outcome(2, 3), 0.2);
    assert_eq!(compute_reward_from_retrieval_outcome(1, 3), -0.3);
}

#[test]
fn query_quality_increases_with_strong_beliefs() {
    let policy = NarsRoutePolicy::FixedTopK {
        top_k: 1,
        blend: ScoreBlend::default(),
    };
    let query = RoutingQueryView {
        data: &[0.2, 0.2],
        dim: 2,
    };
    let mut weak = NarsMsaReasoner::default();
    weak.slot_concepts
        .insert(0, concept_with_truth(0, TruthValue::new(0.2, 0.2)));
    let mut strong = NarsMsaReasoner::default();
    strong
        .slot_concepts
        .insert(0, concept_with_truth(0, TruthValue::new(1.0, 1.0)));

    assert!(
        strong.score_query_quality(&query, &policy) > weak.score_query_quality(&query, &policy)
    );
}
