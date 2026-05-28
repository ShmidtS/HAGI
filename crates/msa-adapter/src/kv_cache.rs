use std::collections::HashMap;

use core_types::shape::Shape;
use tensor_runtime::Tensor;

#[derive(Debug, Clone)]
pub struct HostKvPage {
    pub slot_id: u16,
    pub page_index: usize,
    pub keys: Tensor<f32>,
    pub values: Tensor<f32>,
}

#[derive(Debug, Default)]
pub struct HostKvCache {
    pages: Vec<HostKvPage>,
}

impl HostKvCache {
    pub fn new() -> Self {
        Self { pages: Vec::new() }
    }

    pub fn append_page(&mut self, slot_id: u16, keys: Tensor<f32>, values: Tensor<f32>) -> usize {
        let page_index = self
            .pages
            .iter()
            .filter(|page| page.slot_id == slot_id)
            .count();
        self.pages.push(HostKvPage {
            slot_id,
            page_index,
            keys,
            values,
        });
        page_index
    }

    pub fn pages_for_slot(&self, slot_id: u16) -> Vec<&HostKvPage> {
        self.pages
            .iter()
            .filter(|page| page.slot_id == slot_id)
            .collect()
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }
}

/// Per-layer key/value cache for autoregressive decoding.
pub struct KVCache {
    cache: HashMap<usize, (Tensor<f32>, Tensor<f32>)>,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Store or replace the key/value pair for a given layer.
    pub fn append(&mut self, layer: usize, k: Tensor<f32>, v: Tensor<f32>) {
        self.cache.insert(layer, (k, v));
    }

    /// Retrieve the cached key/value pair for a given layer.
    pub fn get(&self, layer: usize) -> Option<&(Tensor<f32>, Tensor<f32>)> {
        self.cache.get(&layer)
    }

    /// Trim all cached tensors to at most `max_len` along the sequence dimension (dim 1).
    /// Assumes tensors are shaped [B, T, ...].
    pub fn trim(&mut self, max_len: usize) {
        for (_, (k, v)) in self.cache.iter_mut() {
            *k = trim_seq_dim(k, max_len);
            *v = trim_seq_dim(v, max_len);
        }
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Trim a tensor along dimension 1 (sequence) to at most `max_len`.
fn trim_seq_dim(t: &Tensor<f32>, max_len: usize) -> Tensor<f32> {
    let shape = t.shape();
    if shape.rank() < 2 || shape.dims[1] <= max_len {
        return t.clone();
    }
    let b = shape.dims[0];
    let t_dim = shape.dims[1];
    let trailing: usize = shape.dims[2..].iter().product();
    let data = t.data();
    let mut trimmed = Vec::with_capacity(b * max_len * trailing);
    for bi in 0..b {
        let src_off = bi * t_dim * trailing;
        let copy_len = max_len * trailing;
        trimmed.extend_from_slice(&data[src_off..src_off + copy_len]);
    }
    let mut new_dims = shape.dims.clone();
    new_dims[1] = max_len;
    Tensor::from_vec(trimmed, Shape::new(new_dims))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_retrieve_roundtrip_preserves_shape() {
        let mut cache = KVCache::new();
        let k = Tensor::from_vec(vec![0.0; 2 * 10 * 8], Shape::new(vec![2, 10, 8]));
        let v = Tensor::from_vec(vec![0.0; 2 * 10 * 8], Shape::new(vec![2, 10, 8]));
        cache.append(0, k, v);

        let (k_out, v_out) = cache.get(0).unwrap();
        assert_eq!(k_out.shape().dims, vec![2, 10, 8]);
        assert_eq!(v_out.shape().dims, vec![2, 10, 8]);
    }

    #[test]
    fn get_missing_layer_returns_none() {
        let cache = KVCache::new();
        assert!(cache.get(99).is_none());
    }

    #[test]
    fn trim_reduces_sequence_dim() {
        let mut cache = KVCache::new();
        let k = Tensor::from_vec(vec![1.0; 2 * 20 * 4], Shape::new(vec![2, 20, 4]));
        let v = Tensor::from_vec(vec![2.0; 2 * 20 * 4], Shape::new(vec![2, 20, 4]));
        cache.append(0, k, v);
        cache.trim(10);

        let (k_out, v_out) = cache.get(0).unwrap();
        assert_eq!(k_out.shape().dims, vec![2, 10, 4]);
        assert_eq!(v_out.shape().dims, vec![2, 10, 4]);
    }

    #[test]
    fn trim_noop_when_already_short() {
        let mut cache = KVCache::new();
        let k = Tensor::from_vec(vec![0.0; 1 * 5 * 4], Shape::new(vec![1, 5, 4]));
        let v = Tensor::from_vec(vec![0.0; 1 * 5 * 4], Shape::new(vec![1, 5, 4]));
        cache.append(0, k, v);
        cache.trim(100);

        let (k_out, _) = cache.get(0).unwrap();
        assert_eq!(k_out.shape().dims, vec![1, 5, 4]);
    }
}
