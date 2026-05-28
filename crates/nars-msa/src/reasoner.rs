use std::cmp::Ordering;
use std::collections::HashMap;

use msa_adapter::{route_top_k, MsaError, RouteSelection, RoutingQueryView, SlotRegistry};
use nars_core::{Term, TruthValue};

use crate::policy::{NarsRoutePolicy, ScoreBlend};

#[derive(Debug, Clone, PartialEq)]
pub struct NarsMsaConfig {
    pub default_feedback_confidence: f64,
}

impl Default for NarsMsaConfig {
    fn default() -> Self {
        Self {
            default_feedback_confidence: 0.9,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NarsMsaReasoner {
    pub slot_concepts: HashMap<u16, Vec<(Term, TruthValue)>>,
    pub config: NarsMsaConfig,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlotBelief {
    pub slot_id: u16,
    pub truth: TruthValue,
    pub last_updated: usize,
}

impl NarsMsaReasoner {
    pub fn new(config: NarsMsaConfig) -> Self {
        Self {
            slot_concepts: HashMap::new(),
            config,
        }
    }

    pub fn observe_route_feedback(&mut self, slot_id: u16, reward: f32, _step: usize) {
        let clamped_reward = reward.clamp(-1.0, 1.0);
        let reward_truth = TruthValue::new(
            ((clamped_reward + 1.0) * 0.5) as f64,
            self.config.default_feedback_confidence,
        );
        let concepts = self.slot_concepts.entry(slot_id).or_insert_with(|| {
            vec![(
                Term::atom(format!("slot_{slot_id}_route_feedback")),
                TruthValue::new(0.5, 0.0),
            )]
        });

        if let Some((_, truth)) = concepts.first_mut() {
            *truth = revise_truth(*truth, reward_truth);
        }
    }

    pub fn score_query_quality(
        &self,
        query: &RoutingQueryView<'_>,
        policy: &NarsRoutePolicy,
    ) -> f32 {
        if query.data.is_empty() || query.dim == 0 {
            return 0.0;
        }

        self.slot_concepts
            .iter()
            .filter_map(|(slot_id, concepts)| {
                let dot_score = query.data.get(*slot_id as usize).copied()?;
                let truth = concepts.first().map(|(_, truth)| truth);
                Some(score_slot(dot_score, truth, 0, 0, policy.blend()))
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(0.0)
    }

    fn slot_truth(&self, slot_id: u16) -> Option<&TruthValue> {
        self.slot_concepts
            .get(&slot_id)
            .and_then(|concepts| concepts.first().map(|(_, truth)| truth))
    }
}

impl Default for NarsMsaReasoner {
    fn default() -> Self {
        Self::new(NarsMsaConfig::default())
    }
}

pub fn compute_reward_from_retrieval_outcome(retrieved_count: usize, total_selected: usize) -> f32 {
    if total_selected == 0 {
        return -0.3;
    }
    if retrieved_count == 0 {
        return 0.0;
    }
    if retrieved_count >= total_selected {
        return 0.5;
    }
    if retrieved_count * 2 >= total_selected {
        return 0.2;
    }
    -0.3
}

fn revise_truth(existing: TruthValue, observed: TruthValue) -> TruthValue {
    let existing_confidence = existing.confidence();
    let observed_confidence = observed.confidence();
    let total_confidence = existing_confidence + observed_confidence;
    if total_confidence == 0.0 {
        return TruthValue::new(0.5, 0.0);
    }

    TruthValue::new(
        (existing.frequency() * existing_confidence + observed.frequency() * observed_confidence)
            / total_confidence,
        (total_confidence / (1.0 + total_confidence)).clamp(0.0, 1.0),
    )
}

pub fn score_slot(
    dot_score: f32,
    belief_opt: Option<&TruthValue>,
    timestamp: usize,
    current_step: usize,
    blend: &ScoreBlend,
) -> f32 {
    let truth_score = belief_opt
        .map(|truth| (truth.frequency() * truth.confidence()) as f32)
        .unwrap_or(0.0);
    let age = current_step.saturating_sub(timestamp) as f32;
    let recency_score = 1.0 / (1.0 + age);

    blend.dot_weight * dot_score
        + blend.truth_weight * truth_score
        + blend.recency_weight * recency_score
}

pub fn route_top_k_with_nars(
    registry: &SlotRegistry,
    query: RoutingQueryView<'_>,
    reasoner: &mut NarsMsaReasoner,
    policy: &NarsRoutePolicy,
    step: usize,
) -> Result<RouteSelection, MsaError> {
    let blend = policy.blend();
    let max_candidates = policy.max_candidates();

    if blend.truth_weight == 0.0 && blend.recency_weight == 0.0 {
        return match policy {
            NarsRoutePolicy::FixedTopK { top_k, .. } => route_top_k(registry, query, *top_k),
            NarsRoutePolicy::ConfidenceThreshold { max_k, .. } => {
                let mut selection = route_top_k(registry, query, *max_k)?;
                apply_confidence_threshold(&mut selection, policy);
                normalize_like_route_top_k(&mut selection);
                Ok(selection)
            }
        };
    }

    if max_candidates == 0 {
        return Err(MsaError::InvalidTopK {
            top_k: max_candidates,
        });
    }
    if registry.is_empty() {
        return route_top_k(registry, query, max_candidates.max(1));
    }

    let mut selection = route_top_k(registry, query, registry.len())?;
    let mut scored: Vec<(u16, f32)> = selection
        .slot_ids
        .iter()
        .copied()
        .zip(selection.raw_scores.iter().copied())
        .map(|(slot_id, dot_score)| {
            let timestamp = registry
                .get(slot_id as usize)
                .map(|slot| slot.timestamp)
                .unwrap_or(step);
            let blended = score_slot(
                dot_score,
                reasoner.slot_truth(slot_id),
                timestamp,
                step,
                blend,
            );
            (slot_id, blended)
        })
        .collect();

    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.truncate(max_candidates.min(scored.len()));

    selection.slot_ids.clear();
    selection.raw_scores.clear();
    selection.normalized_weights.clear();
    for (slot_id, score) in scored {
        selection.slot_ids.push(slot_id);
        selection.raw_scores.push(score);
    }

    apply_confidence_threshold(&mut selection, policy);
    normalize_like_route_top_k(&mut selection);
    Ok(selection)
}

fn apply_confidence_threshold(selection: &mut RouteSelection, policy: &NarsRoutePolicy) {
    let NarsRoutePolicy::ConfidenceThreshold {
        min_k,
        max_k,
        cumulative_confidence,
        ..
    } = policy
    else {
        return;
    };

    let max_k = (*max_k).min(selection.slot_ids.len());
    let min_k = (*min_k).min(max_k);
    if max_k == 0 {
        selection.slot_ids.clear();
        selection.raw_scores.clear();
        selection.normalized_weights.clear();
        return;
    }

    let positive_sum: f32 = selection
        .raw_scores
        .iter()
        .take(max_k)
        .copied()
        .filter(|score| score.is_finite() && *score > 0.0)
        .sum();

    let mut selected = min_k;
    if positive_sum > 0.0 {
        let mut cumulative = 0.0f32;
        for (idx, score) in selection.raw_scores.iter().take(max_k).copied().enumerate() {
            if score.is_finite() && score > 0.0 {
                cumulative += score / positive_sum;
            }
            selected = idx + 1;
            if selected >= min_k && cumulative >= *cumulative_confidence {
                break;
            }
        }
    } else {
        selected = max_k;
    }

    selection.slot_ids.truncate(selected);
    selection.raw_scores.truncate(selected);
    selection.normalized_weights.clear();
}

fn normalize_like_route_top_k(selection: &mut RouteSelection) {
    selection.normalized_weights.clear();
    let k = selection.raw_scores.len();
    if k == 0 {
        return;
    }

    let min_score = selection
        .raw_scores
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let shifted_scores: Vec<f32> = selection
        .raw_scores
        .iter()
        .map(|score| (*score - min_score).max(0.0))
        .collect();
    let score_sum: f32 = shifted_scores.iter().sum();
    if score_sum == 0.0 || !score_sum.is_finite() {
        let uniform = 1.0 / k as f32;
        selection.normalized_weights.extend((0..k).map(|_| uniform));
    } else {
        selection
            .normalized_weights
            .extend(shifted_scores.iter().map(|score| *score / score_sum));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::shape::Shape;
    use msa_adapter::MemorySlot;
    use tensor_runtime::Tensor;

    fn make_registry() -> SlotRegistry {
        let mut registry = SlotRegistry::new();
        registry.register(MemorySlot::new(
            0,
            Tensor::from_vec(vec![1.0, 0.0], Shape::new(vec![2])),
            Tensor::from_vec(vec![0.0, 0.0], Shape::new(vec![2])),
            0,
            "a".into(),
        ));
        registry.register(MemorySlot::new(
            1,
            Tensor::from_vec(vec![0.5, 0.0], Shape::new(vec![2])),
            Tensor::from_vec(vec![0.0, 0.0], Shape::new(vec![2])),
            9,
            "b".into(),
        ));
        registry.register(MemorySlot::new(
            2,
            Tensor::from_vec(vec![0.25, 0.0], Shape::new(vec![2])),
            Tensor::from_vec(vec![0.0, 0.0], Shape::new(vec![2])),
            10,
            "c".into(),
        ));
        registry
    }

    #[test]
    fn score_slot_blends_dot_truth_and_recency() {
        let truth = TruthValue::new(0.8, 0.5);
        let blend = ScoreBlend {
            dot_weight: 1.0,
            truth_weight: 2.0,
            recency_weight: 3.0,
        };
        let score = score_slot(0.25, Some(&truth), 8, 10, &blend);
        assert!((score - 2.05).abs() < 1e-6);
    }

    #[test]
    fn fixed_top_k_prefers_strong_truth_when_weighted() {
        let registry = make_registry();
        let mut reasoner = NarsMsaReasoner::default();
        reasoner
            .slot_concepts
            .insert(2, vec![(Term::atom("slot_2"), TruthValue::new(1.0, 1.0))]);
        let policy = NarsRoutePolicy::FixedTopK {
            top_k: 1,
            blend: ScoreBlend {
                dot_weight: 1.0,
                truth_weight: 1.0,
                recency_weight: 0.0,
            },
        };

        let selection = route_top_k_with_nars(
            &registry,
            RoutingQueryView {
                data: &[1.0, 0.0],
                dim: 2,
            },
            &mut reasoner,
            &policy,
            10,
        )
        .unwrap();

        assert_eq!(selection.slot_ids.as_slice(), &[2]);
    }

    #[test]
    fn positive_feedback_increases_confidence() {
        let mut reasoner = NarsMsaReasoner::default();
        reasoner.observe_route_feedback(7, 1.0, 1);
        let truth = &reasoner.slot_concepts.get(&7).unwrap()[0].1;
        assert!(truth.frequency() > 0.0);
        assert!(truth.confidence() > 0.0);
    }
}
