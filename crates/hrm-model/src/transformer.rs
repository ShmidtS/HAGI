use tensor_runtime::Tensor;

/// Single Transformer block with self-attention + MLP.
pub struct TransformerBlock {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl TransformerBlock {
    pub fn new(hidden_size: usize, num_heads: usize) -> Self {
        assert_eq!(hidden_size % num_heads, 0);
        Self {
            hidden_size,
            num_heads,
            head_dim: hidden_size / num_heads,
        }
    }

    pub fn forward(&self, _hidden_states: &Tensor<f32>) -> Tensor<f32> {
        // Placeholder: naive self-attention + MLP will be implemented here.
        _hidden_states.clone()
    }
}

/// Sequential Transformer stack.
pub struct TransformerStack {
    pub blocks: Vec<TransformerBlock>,
}

impl TransformerStack {
    pub fn new(blocks: Vec<TransformerBlock>) -> Self {
        Self { blocks }
    }

    pub fn forward(&self, mut hidden_states: Tensor<f32>) -> Tensor<f32> {
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states);
        }
        hidden_states
    }
}
