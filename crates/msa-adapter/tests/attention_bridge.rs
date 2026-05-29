use core_types::shape::Shape;
use msa_adapter::{
    sparse_attention_with_memory, HostKvCache, MemorySlot, MsaConfig, SlotRegistry, SparseRouter,
};
use tensor_runtime::Tensor;

#[test]
fn sparse_attention_with_memory_preserves_local_hidden_shape() {
    let hidden = 4;
    let local_hidden = Tensor::from_vec(
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2],
        Shape::new(vec![1, 2, hidden]),
    );
    let local_k = Tensor::from_vec(vec![0.1, 0.0, 0.0, 0.0], Shape::new(vec![1, hidden]));
    let local_v = Tensor::from_vec(vec![0.0, 0.1, 0.0, 0.0], Shape::new(vec![1, hidden]));

    let mut registry = SlotRegistry::new();
    registry.register(MemorySlot::new(
        0,
        Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0], Shape::new(vec![hidden])),
        Tensor::from_vec(vec![0.0; hidden], Shape::new(vec![hidden])),
        0,
        "slot0".into(),
    ));
    registry.register(MemorySlot::new(
        1,
        Tensor::from_vec(vec![0.0, 1.0, 0.0, 0.0], Shape::new(vec![hidden])),
        Tensor::from_vec(vec![0.0; hidden], Shape::new(vec![hidden])),
        0,
        "slot1".into(),
    ));

    let mut cache = HostKvCache::new();
    cache.append_page(
        0,
        Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0], Shape::new(vec![hidden])),
        Tensor::from_vec(vec![0.0, 0.0, 1.0, 0.0], Shape::new(vec![hidden])),
    );
    cache.append_page(
        1,
        Tensor::from_vec(vec![0.0, 1.0, 0.0, 0.0], Shape::new(vec![hidden])),
        Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0], Shape::new(vec![hidden])),
    );

    let router =
        SparseRouter::try_from_config(MsaConfig::try_new(1).expect("test top_k must be valid"))
            .expect("SparseRouter test config must be valid");
    let output = sparse_attention_with_memory(
        &local_hidden,
        (&local_k, &local_v),
        &router,
        &registry,
        &cache,
        1,
    )
    .unwrap();

    assert_eq!(output.shape().dims, local_hidden.shape().dims);
}
