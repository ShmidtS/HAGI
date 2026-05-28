use std::cmp::Ordering;

use core_types::algebra::{AlgebraSignature, Cl};
use hdim_model::MultivectorBatch;
use smallvec::SmallVec;
use tensor_runtime::Tensor;

use crate::registry::SlotRegistry;

pub type Cl3 = Cl<3, 0, 0>;

#[derive(Debug, Clone, PartialEq)]
pub struct RouteSelection {
    pub slot_ids: SmallVec<[u16; 16]>,
    pub raw_scores: SmallVec<[f32; 16]>,
    pub normalized_weights: SmallVec<[f32; 16]>,
}

#[derive(Debug)]
pub struct RoutingQueryView<'a> {
    pub data: &'a [f32],
    pub dim: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum MsaError {
    #[error("query dim {query_dim} != slot dim {slot_dim}")]
    QueryDimMismatch { query_dim: usize, slot_dim: usize },
    #[error("invalid top_k: {top_k}")]
    InvalidTopK { top_k: usize },
    #[error("slot id {0} out of u16 range")]
    SlotIdOutOfRange(usize),
    #[error("tensor shape: {0}")]
    TensorShape(String),
    #[error("zero dimension: {dim}")]
    ZeroDimension { dim: String },
}

pub fn routing_query_from_invariant(invariant: &MultivectorBatch<Cl3>) -> RoutingQueryView<'_> {
    RoutingQueryView {
        data: invariant.coeffs.data(),
        dim: invariant.structural_heads * <Cl3 as AlgebraSignature>::BLADE_COUNT,
    }
}

pub fn route_from_hdim_invariant(
    registry: &SlotRegistry,
    invariant: &MultivectorBatch<Cl3>,
    top_k: usize,
) -> Result<RouteSelection, MsaError> {
    let query = routing_query_from_invariant(invariant);
    route_top_k(registry, query, top_k)
}

pub fn route_top_k(
    registry: &SlotRegistry,
    query: RoutingQueryView<'_>,
    top_k: usize,
) -> Result<RouteSelection, MsaError> {
    if top_k == 0 {
        return Err(MsaError::InvalidTopK { top_k });
    }

    if registry.is_empty() {
        return Ok(RouteSelection {
            slot_ids: SmallVec::new(),
            raw_scores: SmallVec::new(),
            normalized_weights: SmallVec::new(),
        });
    }

    if query.dim == 0 {
        return Err(MsaError::ZeroDimension {
            dim: "query".to_string(),
        });
    }
    if !query.data.len().is_multiple_of(query.dim) {
        return Err(MsaError::TensorShape(format!(
            "query data length {} is not divisible by dim {}",
            query.data.len(),
            query.dim
        )));
    }

    let query_rows = query.data.len() / query.dim;
    if query_rows == 0 {
        return Err(MsaError::TensorShape("query data is empty".to_string()));
    }
    let mean_query;
    let query_data = if query_rows == 1 {
        query.data
    } else {
        mean_query = mean_rows(query.data, query.dim);
        &mean_query
    };

    let all_keys = registry.all_keys();
    let slot_dim = all_keys.shape().dims[1];
    if query.dim != slot_dim {
        return Err(MsaError::QueryDimMismatch {
            query_dim: query.dim,
            slot_dim,
        });
    }

    let key_data = all_keys.data();
    let slot_ids = registry.slot_ids();
    let mut scored = Vec::with_capacity(registry.len());
    for (idx, slot_id) in slot_ids.into_iter().enumerate() {
        let key_offset = idx * slot_dim;
        let mut score = 0.0f32;
        for d in 0..slot_dim {
            score += query_data[d] * key_data[key_offset + d];
        }
        scored.push((slot_id, score));
    }

    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let k = top_k.min(scored.len());
    scored.truncate(k);

    let mut slot_ids = SmallVec::<[u16; 16]>::new();
    let mut raw_scores = SmallVec::<[f32; 16]>::new();
    for (slot_id, score) in &scored {
        let slot_id = u16::try_from(*slot_id).map_err(|_| MsaError::SlotIdOutOfRange(*slot_id))?;
        slot_ids.push(slot_id);
        raw_scores.push(*score);
    }

    let min_score = raw_scores.iter().copied().fold(f32::INFINITY, f32::min);
    let shifted_scores: Vec<f32> = raw_scores
        .iter()
        .map(|score| (*score - min_score).max(0.0))
        .collect();
    let score_sum: f32 = shifted_scores.iter().sum();
    let mut normalized_weights = SmallVec::<[f32; 16]>::new();
    if score_sum == 0.0 || !score_sum.is_finite() {
        let uniform = 1.0 / k as f32;
        normalized_weights.extend((0..k).map(|_| uniform));
    } else {
        normalized_weights.extend(shifted_scores.iter().map(|score| *score / score_sum));
    }

    Ok(RouteSelection {
        slot_ids,
        raw_scores,
        normalized_weights,
    })
}

fn mean_rows(data: &[f32], dim: usize) -> Vec<f32> {
    let rows = data.len() / dim;
    let mut mean = vec![0.0f32; dim];
    for row in 0..rows {
        let offset = row * dim;
        for d in 0..dim {
            mean[d] += data[offset + d];
        }
    }
    for value in &mut mean {
        *value /= rows as f32;
    }
    mean
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MsaConfig {
    pub top_k: usize,
}

impl MsaConfig {
    pub fn new(top_k: usize) -> Self {
        assert!(top_k > 0, "top_k must be positive");
        Self { top_k }
    }
}

/// Sparse top-k router that selects memory slots by dot-product similarity.
pub struct SparseRouter {
    pub top_k: usize,
}

impl SparseRouter {
    pub fn new(top_k: usize) -> Self {
        Self::from_config(MsaConfig::new(top_k))
    }

    pub fn from_config(config: MsaConfig) -> Self {
        Self {
            top_k: config.top_k,
        }
    }

    /// Route a query tensor [B, T, hidden] to the top-k most relevant slots.
    ///
    /// Returns (sorted_slot_ids, normalized_weights) where weights sum to ~1.0.
    pub fn route(&self, query: &Tensor<f32>, registry: &SlotRegistry) -> (Vec<usize>, Vec<f32>) {
        let shape = query.shape();
        assert_eq!(shape.rank(), 3, "query must be [B, T, hidden]");
        let batch = shape.dims[0];
        let tokens = shape.dims[1];
        let hidden = shape.dims[2];
        let count = batch * tokens;

        let mut mean_query = vec![0.0f32; hidden];
        if count > 0 {
            let q_data = query.data();
            for bt in 0..count {
                let offset = bt * hidden;
                for d in 0..hidden {
                    mean_query[d] += q_data[offset + d];
                }
            }
            for value in &mut mean_query {
                *value /= count as f32;
            }
        }

        let selection = route_top_k(
            registry,
            RoutingQueryView {
                data: &mean_query,
                dim: hidden,
            },
            self.top_k,
        )
        .expect("SparseRouter query must match registry slot dimensions");

        let ids = selection.slot_ids.iter().map(|id| *id as usize).collect();
        let weights = selection.normalized_weights.iter().copied().collect();
        (ids, weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slot::MemorySlot;
    use core_types::shape::Shape;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn build_registry(num_slots: usize, key_dim: usize, rng: &mut ChaCha8Rng) -> SlotRegistry {
        let mut reg = SlotRegistry::new();
        for i in 0..num_slots {
            let key_data: Vec<f32> = (0..key_dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let val_data: Vec<f32> = (0..key_dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let key = Tensor::from_vec(key_data, Shape::new(vec![key_dim]));
            let val = Tensor::from_vec(val_data, Shape::new(vec![key_dim]));
            reg.register(MemorySlot::new(i, key, val, 0, "slot".into()));
        }
        reg
    }

    #[test]
    fn routing_returns_exactly_top_k_sorted() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let reg = build_registry(100, 16, &mut rng);
        let query_data: Vec<f32> = (0..2 * 3 * 16).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let query = Tensor::from_vec(query_data, Shape::new(vec![2, 3, 16]));

        let router = SparseRouter::new(5);
        let (ids, weights) = router.route(&query, &reg);

        assert_eq!(ids.len(), 5);
        assert_eq!(weights.len(), 5);
        for &id in &ids {
            assert!(id < 100);
        }
    }

    #[test]
    fn weights_sum_to_one() {
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let reg = build_registry(100, 16, &mut rng);
        let query_data: Vec<f32> = (0..2 * 3 * 16).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let query = Tensor::from_vec(query_data, Shape::new(vec![2, 3, 16]));

        let router = SparseRouter::new(5);
        let (_, weights) = router.route(&query, &reg);

        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "weight sum {} deviates from 1.0",
            sum
        );
    }

    #[test]
    fn routing_with_fewer_slots_than_k() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let reg = build_registry(3, 8, &mut rng);
        let query_data: Vec<f32> = (0..1 * 2 * 8).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let query = Tensor::from_vec(query_data, Shape::new(vec![1, 2, 8]));

        let router = SparseRouter::new(10);
        let (ids, weights) = router.route(&query, &reg);

        assert_eq!(ids.len(), 3);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn routes_from_hdim_invariant_batch() {
        let mut reg = SlotRegistry::new();
        reg.register(MemorySlot::new(
            0,
            Tensor::from_vec(vec![1.0; 8], Shape::new(vec![8])),
            Tensor::from_vec(vec![0.0; 8], Shape::new(vec![8])),
            0,
            "slot".into(),
        ));
        let invariant = MultivectorBatch::<Cl3>::new(
            Tensor::from_vec(vec![1.0; 2 * 3 * 1 * 8], Shape::new(vec![2, 3, 1, 8])),
            1,
        );

        let selection = route_from_hdim_invariant(&reg, &invariant, 1).unwrap();

        assert_eq!(routing_query_from_invariant(&invariant).dim, 8);
        assert_eq!(selection.slot_ids.as_slice(), &[0]);
    }
}
