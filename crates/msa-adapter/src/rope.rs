use tensor_runtime::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoPEPosition {
    pub doc_id: u64,
    pub position: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DocumentWiseRoPE {
    pub base: f32,
    pub max_len: usize,
}

impl DocumentWiseRoPE {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn position(&self, doc_id: u64, position: usize) -> RoPEPosition {
        RoPEPosition { doc_id, position }
    }

    pub fn same_document(&self, lhs: RoPEPosition, rhs: RoPEPosition) -> bool {
        lhs.doc_id == rhs.doc_id
    }
}

impl Default for DocumentWiseRoPE {
    fn default() -> Self {
        Self {
            base: 10_000.0,
            max_len: 0,
        }
    }
}

pub fn apply_document_wise_rope(q: &mut Tensor<f32>, k: &mut Tensor<f32>, doc_ids: &[u16]) {
    crate::attention_bridge::validate_rope_inputs(q, k, doc_ids)
        .expect("doc_ids length must match batch size");
}
