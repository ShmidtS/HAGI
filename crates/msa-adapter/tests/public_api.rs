use core_types::shape::Shape;
use msa_adapter::{
    fetch_pages, route_top_k, run_memory_interleave, sparse_attention_over_pages, FetchEvent,
    HostKvCache, HostKvPage, KVCache, MemoryInterleaveConfig, MemoryInterleaveReport, MemorySlot,
    MemoryStopReason, MsaError, RouteSelection, RoutingQueryView, SlotRegistry, SparseAttention,
    SparseRouter,
};
use tensor_runtime::Tensor;

#[test]
fn msa_public_api_exports_route_cache_fetch_interleave_types() {
    let mut registry = SlotRegistry::new();
    let key = Tensor::from_vec(vec![1.0, 0.0], Shape::new(vec![2]));
    let value = Tensor::from_vec(vec![0.0, 1.0], Shape::new(vec![2]));
    registry.register(MemorySlot::new(0, key, value, 0, "api".into()));

    let query = [1.0, 0.0];
    let selection: RouteSelection = route_top_k(
        &registry,
        RoutingQueryView {
            data: &query,
            dim: 2,
        },
        1,
    )
    .unwrap();

    let mut cache = HostKvCache::new();
    cache.append_page(
        selection.slot_ids[0],
        Tensor::from_vec(vec![1.0, 0.0], Shape::new(vec![2])),
        Tensor::from_vec(vec![0.0, 1.0], Shape::new(vec![2])),
    );
    let event: FetchEvent = fetch_pages(&cache, selection.slot_ids.as_slice());
    let pages: Vec<HostKvPage> = event.wait();

    let tensor_query = Tensor::from_vec(vec![1.0, 0.0], Shape::new(vec![1, 1, 2]));
    let output = sparse_attention_over_pages(&tensor_query, &pages).unwrap();
    assert_eq!(output.shape().dims, vec![1, 1, 2]);

    let report: MemoryInterleaveReport = run_memory_interleave(
        &tensor_query,
        &selection,
        MemoryInterleaveConfig {
            max_steps: 1,
            min_delta: 0.0,
        },
    );
    assert_eq!(report.stop_reason, MemoryStopReason::MaxSteps);

    let _router = SparseRouter::new(1);
    let _attention = SparseAttention::new();
    let _kv_cache = KVCache::new();
    let _error = MsaError::InvalidTopK { top_k: 0 };
}
