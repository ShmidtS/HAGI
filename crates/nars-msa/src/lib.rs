pub mod policy;
pub mod reasoner;

pub use policy::{NarsRoutePolicy, ScoreBlend};
pub use reasoner::{
    compute_reward_from_retrieval_outcome, route_top_k_with_nars, score_slot, NarsMsaConfig,
    NarsMsaReasoner, SlotBelief,
};

#[cfg(test)]
mod tests {
    use core_types::shape::Shape;
    use msa_adapter::{route_top_k, MemorySlot, RoutingQueryView, SlotRegistry};
    use nars_core::{BudgetValue, Concept, Sentence, Task, Term, TruthValue};
    use tensor_runtime::Tensor;

    use super::*;

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
        registry.register(make_slot(0, vec![0.25, 0.0], 0));
        registry.register(make_slot(1, vec![1.0, 0.0], 0));
        registry.register(make_slot(2, vec![0.5, 0.0], 0));
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
    fn route_with_zero_truth_weight_falls_back_to_dot_scores() {
        let registry = make_registry();
        let mut reasoner = NarsMsaReasoner::default();
        reasoner
            .slot_concepts
            .insert(0, concept_with_truth(0, TruthValue::new(1.0, 1.0)));
        let policy = NarsRoutePolicy::FixedTopK {
            top_k: 2,
            blend: ScoreBlend {
                dot_weight: 1.0,
                truth_weight: 0.0,
                recency_weight: 0.0,
            },
        };

        let query = [1.0, 0.0];
        let expected = route_top_k(
            &registry,
            RoutingQueryView {
                data: &query,
                dim: 2,
            },
            2,
        )
        .unwrap();
        let actual = route_top_k_with_nars(
            &registry,
            RoutingQueryView {
                data: &query,
                dim: 2,
            },
            &mut reasoner,
            &policy,
            0,
        )
        .unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn confidence_threshold_selects_until_cumulative_confidence() {
        let registry = make_registry();
        let mut reasoner = NarsMsaReasoner::default();
        let policy = NarsRoutePolicy::ConfidenceThreshold {
            min_k: 1,
            max_k: 3,
            cumulative_confidence: 0.7,
            blend: ScoreBlend {
                dot_weight: 1.0,
                truth_weight: 0.0,
                recency_weight: 0.0,
            },
        };

        let query = [1.0, 0.0];
        let selection = route_top_k_with_nars(
            &registry,
            RoutingQueryView {
                data: &query,
                dim: 2,
            },
            &mut reasoner,
            &policy,
            0,
        )
        .unwrap();

        assert_eq!(selection.slot_ids.as_slice(), &[1, 2]);
        assert_eq!(selection.raw_scores.as_slice(), &[1.0, 0.5]);
        assert!((selection.normalized_weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn slot_belief_updates_with_positive_reward() {
        let mut reasoner = NarsMsaReasoner::default();
        reasoner.observe_route_feedback(3, 1.0, 7);

        let concept = reasoner.slot_concepts.get(&3).unwrap();
        let truth = concept.latest_belief_truth().unwrap();
        assert_eq!(concept.beliefs().len(), 1);
        assert!(truth.frequency() > 0.0);
        assert!(truth.confidence() > 0.0);
    }
}
