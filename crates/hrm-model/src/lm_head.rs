use tensor_runtime::Tensor;

/// Token embeddings + vocabulary projection.
pub struct LmHead {
    pub vocab_size: usize,
    pub hidden_size: usize,
}

impl LmHead {
    pub fn new(vocab_size: usize, hidden_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_size,
        }
    }

    pub fn embed(&self, _input_ids: &Tensor<u32>) -> Tensor<f32> {
        // Placeholder: lookup embedding table.
        Tensor::zeros(
            core_types::shape::Shape::new(vec![0, self.hidden_size]),
            core_types::dtype::DType::F32,
        )
    }

    pub fn project(&self, hidden: &Tensor<f32>) -> Tensor<f32> {
        // Placeholder: linear projection to vocab logits.
        hidden.clone()
    }
}
