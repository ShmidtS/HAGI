use core_types::shape::Shape;
use msa_adapter::{
    fetch_pages, route_top_k, sparse_attention_over_pages, HostKvCache, MemorySlot,
    RoutingQueryView, SlotRegistry,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor_runtime::Tensor;

#[test]
fn msa_route_fetch_sparse_attention_end_to_end_cpu() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let hidden = 8;
    let mut registry = SlotRegistry::new();
    for id in 0..10 {
        let key_data: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let value_data: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-1.0..1.0)).collect();
        registry.register(MemorySlot::new(
            id,
            Tensor::from_vec(key_data, Shape::new(vec![hidden])),
            Tensor::from_vec(value_data, Shape::new(vec![hidden])),
            0,
            "integration".into(),
        ));
    }

    let query_data: Vec<f32> = (0..2 * 3 * hidden)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let query = Tensor::from_vec(query_data, Shape::new(vec![2, 3, hidden]));
    let routing_query: Vec<f32> = query.data()[0..hidden].to_vec();
    let selection = route_top_k(
        &registry,
        RoutingQueryView {
            data: &routing_query,
            dim: hidden,
        },
        3,
    )
    .unwrap();

    let mut cache = HostKvCache::new();
    for &slot_id in &selection.slot_ids {
        let key_data: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let value_data: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-1.0..1.0)).collect();
        cache.append_page(
            slot_id,
            Tensor::from_vec(key_data, Shape::new(vec![hidden])),
            Tensor::from_vec(value_data, Shape::new(vec![hidden])),
        );
    }

    let pages = fetch_pages(&cache, selection.slot_ids.as_slice()).wait();
    let output = sparse_attention_over_pages(&query, &pages).unwrap();

    assert_eq!(selection.slot_ids.len(), 3);
    assert!(pages
        .iter()
        .all(|page| selection.slot_ids.contains(&page.slot_id)));
    assert_eq!(output.shape().dims, query.shape().dims);
    assert!(output.data().iter().all(|value| value.is_finite()));
}
