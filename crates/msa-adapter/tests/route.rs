use core_types::shape::Shape;
use msa_adapter::{
    route_top_k, MemorySlot, MsaConfig, MsaError, RouteSelection, RoutingQueryView, SlotRegistry,
    SparseRouter,
};
use tensor_runtime::Tensor;

fn slot(id: usize, key: Vec<f32>) -> MemorySlot {
    let dim = key.len();
    MemorySlot::new(
        id,
        Tensor::from_vec(key, Shape::new(vec![dim])),
        Tensor::from_vec(vec![0.0; dim], Shape::new(vec![dim])),
        0,
        "test".into(),
    )
}

#[test]
fn route_selection_empty_default_vectors() {
    let selection = RouteSelection {
        slot_ids: smallvec::SmallVec::new(),
        raw_scores: smallvec::SmallVec::new(),
        normalized_weights: smallvec::SmallVec::new(),
    };
    assert!(selection.slot_ids.is_empty());
    assert!(selection.raw_scores.is_empty());
    assert!(selection.normalized_weights.is_empty());
}

#[test]
fn route_top_k_100_slots_returns_top_5_descending_score() {
    let mut registry = SlotRegistry::new();
    for id in 0..100 {
        registry.register(slot(id, vec![id as f32, 0.0]));
    }

    let query = [1.0, 0.0];
    let selection = route_top_k(
        &registry,
        RoutingQueryView {
            data: &query,
            dim: 2,
        },
        5,
    )
    .unwrap();

    assert_eq!(selection.slot_ids.as_slice(), &[99, 98, 97, 96, 95]);
    assert_eq!(
        selection.raw_scores.as_slice(),
        &[99.0, 98.0, 97.0, 96.0, 95.0]
    );
}

#[test]
fn route_top_k_tie_breaks_by_slot_id_ascending() {
    let mut registry = SlotRegistry::new();
    registry.register(slot(3, vec![1.0, 0.0]));
    registry.register(slot(1, vec![1.0, 0.0]));
    registry.register(slot(2, vec![1.0, 0.0]));

    let query = [1.0, 0.0];
    let selection = route_top_k(
        &registry,
        RoutingQueryView {
            data: &query,
            dim: 2,
        },
        3,
    )
    .unwrap();

    assert_eq!(selection.slot_ids.as_slice(), &[1, 2, 3]);
}

#[test]
fn route_top_k_empty_registry_returns_empty_selection() {
    let registry = SlotRegistry::new();
    let query = [1.0, 0.0];
    let selection = route_top_k(
        &registry,
        RoutingQueryView {
            data: &query,
            dim: 2,
        },
        5,
    )
    .unwrap();

    assert!(selection.slot_ids.is_empty());
    assert!(selection.raw_scores.is_empty());
    assert!(selection.normalized_weights.is_empty());
}

#[test]
fn route_top_k_normalized_weights_sum_to_one() {
    let mut registry = SlotRegistry::new();
    registry.register(slot(0, vec![1.0, 0.0]));
    registry.register(slot(1, vec![2.0, 0.0]));
    registry.register(slot(2, vec![3.0, 0.0]));

    let query = [1.0, 0.0];
    let selection = route_top_k(
        &registry,
        RoutingQueryView {
            data: &query,
            dim: 2,
        },
        3,
    )
    .unwrap();
    let sum: f32 = selection.normalized_weights.iter().sum();

    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn route_top_k_zero_sum_scores_use_uniform_weights() {
    let mut registry = SlotRegistry::new();
    registry.register(slot(0, vec![0.0, 0.0]));
    registry.register(slot(1, vec![0.0, 0.0]));

    let query = [1.0, 0.0];
    let selection = route_top_k(
        &registry,
        RoutingQueryView {
            data: &query,
            dim: 2,
        },
        2,
    )
    .unwrap();

    assert_eq!(selection.normalized_weights.as_slice(), &[0.5, 0.5]);
}

#[test]
fn route_top_k_negative_scores_returns_probability_weights() {
    let mut registry = SlotRegistry::new();
    registry.register(slot(0, vec![-1.0, 0.0]));
    registry.register(slot(1, vec![-2.0, 0.0]));
    registry.register(slot(2, vec![-3.0, 0.0]));

    let query = [1.0, 0.0];
    let selection = route_top_k(
        &registry,
        RoutingQueryView {
            data: &query,
            dim: 2,
        },
        3,
    )
    .unwrap();
    let sum: f32 = selection.normalized_weights.iter().sum();

    assert!(selection
        .normalized_weights
        .iter()
        .all(|weight| (0.0..=1.0).contains(weight)));
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn sparse_router_empty_registry_returns_empty() {
    let registry = SlotRegistry::new();
    let query = Tensor::from_vec(vec![1.0, 2.0], Shape::new(vec![1, 1, 2]));
    let router =
        SparseRouter::try_from_config(MsaConfig::try_new(2).expect("test top_k must be valid"))
            .expect("SparseRouter test config must be valid");

    let (ids, weights) = router.route(&query, &registry);

    assert!(ids.is_empty());
    assert!(weights.is_empty());
}

#[test]
fn sparse_router_compat_returns_weights_sum_one() {
    let mut registry = SlotRegistry::new();
    registry.register(slot(0, vec![1.0, 0.0]));
    registry.register(slot(1, vec![2.0, 0.0]));
    let query = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], Shape::new(vec![1, 2, 2]));
    let router =
        SparseRouter::try_from_config(MsaConfig::try_new(2).expect("test top_k must be valid"))
            .expect("SparseRouter test config must be valid");

    let (_, weights) = router.route(&query, &registry);
    let sum: f32 = weights.iter().sum();

    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn route_top_k_zero_top_k_is_error() {
    let registry = SlotRegistry::new();
    let query = [1.0, 0.0];
    let err = route_top_k(
        &registry,
        RoutingQueryView {
            data: &query,
            dim: 2,
        },
        0,
    )
    .unwrap_err();
    assert!(matches!(err, MsaError::InvalidTopK { top_k: 0 }));
}

#[test]
fn sparse_router_try_from_config_rejects_zero_top_k() {
    match SparseRouter::try_from_config(MsaConfig { top_k: 0 }) {
        Err(MsaError::InvalidTopK { top_k: 0 }) => {}
        Err(err) => panic!("unexpected error: {err}"),
        Ok(_) => panic!("zero top_k must be rejected"),
    }
}

#[test]
fn route_within_slots_contract() {
    let n = 8usize;
    let mut registry = SlotRegistry::new();
    for id in 0..n {
        registry.register(slot(id, vec![id as f32 + 1.0, 1.0]));
    }

    let query = [1.0, 0.0];
    let selection = route_top_k(
        &registry,
        RoutingQueryView {
            data: &query,
            dim: 2,
        },
        4,
    )
    .unwrap();
    let mut seen = std::collections::HashSet::new();

    for &slot_id in &selection.slot_ids {
        let slot_id = slot_id as usize;
        assert!(slot_id < n);
        assert!(registry.get(slot_id).is_some());
        assert!(seen.insert(slot_id));
    }
}
