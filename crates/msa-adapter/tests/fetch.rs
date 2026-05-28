use core_types::shape::Shape;
use msa_adapter::{fetch_pages, HostKvCache};
use tensor_runtime::Tensor;

fn tensor(value: f32) -> Tensor<f32> {
    Tensor::from_vec(vec![value, value], Shape::new(vec![2]))
}

#[test]
fn fetch_event_wait_returns_selected_pages() {
    let mut cache = HostKvCache::new();
    cache.append_page(1, tensor(1.0), tensor(10.0));
    cache.append_page(2, tensor(2.0), tensor(20.0));

    let pages = fetch_pages(&cache, &[1, 2]).wait();

    assert_eq!(pages.len(), 2);
    assert_eq!(pages[0].slot_id, 1);
    assert_eq!(pages[1].slot_id, 2);
}

#[test]
fn fetch_pages_does_not_return_unselected_slot_pages() {
    let mut cache = HostKvCache::new();
    cache.append_page(1, tensor(1.0), tensor(10.0));
    cache.append_page(2, tensor(2.0), tensor(20.0));

    let pages = fetch_pages(&cache, &[2]).wait();

    assert_eq!(pages.len(), 1);
    assert_eq!(pages[0].slot_id, 2);
}
