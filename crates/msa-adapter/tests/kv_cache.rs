use core_types::shape::Shape;
use msa_adapter::HostKvCache;
use tensor_runtime::Tensor;

fn tensor(value: f32) -> Tensor<f32> {
    Tensor::from_vec(vec![value, value], Shape::new(vec![2]))
}

#[test]
fn host_kv_cache_append_only_keeps_old_pages() {
    let mut cache = HostKvCache::new();
    cache.append_page(1, tensor(1.0), tensor(10.0));
    cache.append_page(1, tensor(2.0), tensor(20.0));

    let pages = cache.pages_for_slot(1);
    assert_eq!(pages.len(), 2);
    assert_eq!(pages[0].keys.data(), &[1.0, 1.0]);
    assert_eq!(pages[1].keys.data(), &[2.0, 2.0]);
}

#[test]
fn host_kv_cache_pages_for_slot_returns_in_append_order() {
    let mut cache = HostKvCache::new();
    cache.append_page(2, tensor(1.0), tensor(10.0));
    cache.append_page(1, tensor(2.0), tensor(20.0));
    cache.append_page(2, tensor(3.0), tensor(30.0));

    let pages = cache.pages_for_slot(2);
    assert_eq!(pages.len(), 2);
    assert_eq!(pages[0].keys.data(), &[1.0, 1.0]);
    assert_eq!(pages[1].keys.data(), &[3.0, 3.0]);
}

#[test]
fn host_kv_cache_page_indices_increment_per_slot() {
    let mut cache = HostKvCache::new();
    assert_eq!(cache.append_page(1, tensor(1.0), tensor(10.0)), 0);
    assert_eq!(cache.append_page(2, tensor(2.0), tensor(20.0)), 0);
    assert_eq!(cache.append_page(1, tensor(3.0), tensor(30.0)), 1);
    assert_eq!(cache.page_count(), 3);
}
