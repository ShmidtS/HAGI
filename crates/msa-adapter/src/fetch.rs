use crate::kv_cache::{HostKvCache, HostKvPage};

#[derive(Debug)]
pub struct FetchEvent {
    pages: Vec<HostKvPage>,
}

impl FetchEvent {
    pub fn wait(self) -> Vec<HostKvPage> {
        self.pages
    }
}

pub fn fetch_pages(cache: &HostKvCache, selected_slot_ids: &[u16]) -> FetchEvent {
    let mut pages = Vec::new();
    for &slot_id in selected_slot_ids {
        pages.extend(cache.pages_for_slot(slot_id).into_iter().cloned());
    }
    FetchEvent { pages }
}
