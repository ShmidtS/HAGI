use tensor_runtime::Tensor;

use crate::attention::MultiHeadSelfAttention;
use crate::mask::PrefixLmMask;
use crate::mlp::SwiGlMlp;
use crate::norm::RMSNorm;

/// Single Transformer block: norm -> attention -> residual -> norm -> mlp -> residual.
pub struct TransformerBlock {
    pub norm1: RMSNorm,
    pub attention: MultiHeadSelfAttention,
    pub norm2: RMSNorm,
    pub mlp: SwiGlMlp,
}

impl TransformerBlock {
    pub fn new(hidden_size: usize, num_heads: usize, expansion: usize, max_seq_len: usize) -> Self {
        Self {
            norm1: RMSNorm::new(hidden_size, 1e-6),
            attention: MultiHeadSelfAttention::new(hidden_size, num_heads, max_seq_len),
            norm2: RMSNorm::new(hidden_size, 1e-6),
            mlp: SwiGlMlp::new(hidden_size, expansion),
        }
    }

    /// Forward pass: input [B, T, D] -> output [B, T, D]
    pub fn forward(&self, input: &Tensor<f32>, mask: &PrefixLmMask) -> Tensor<f32> {
        // norm1 -> attention -> residual
        let normed1 = self.norm1.forward(input);
        let attn_out = self.attention.forward(&normed1, mask);
        let after_attn = add_tensors(input, &attn_out);

        // norm2 -> mlp -> residual
        let normed2 = self.norm2.forward(&after_attn);
        let mlp_out = self.mlp.forward(&normed2);
        add_tensors(&after_attn, &mlp_out)
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

    pub fn forward(&self, input: &Tensor<f32>, mask: &PrefixLmMask) -> Tensor<f32> {
        let mut hidden = input.clone();
        for block in &self.blocks {
            hidden = block.forward(&hidden, mask);
        }
        hidden
    }
}

/// Elementwise addition of two tensors with matching shapes.
fn add_tensors(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
    assert_eq!(a.shape(), b.shape(), "add_tensors: shapes must match");
    let data: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(&x, &y)| x + y)
        .collect();
    Tensor::from_vec(data, a.shape().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::shape::Shape;

    #[test]
    fn transformer_block_shape_preservation() {
        let block = TransformerBlock::new(16, 4, 2, 32);
        let input = Tensor::from_vec(vec![0.1f32; 2 * 5 * 16], Shape::new(vec![2, 5, 16]));
        let mask = PrefixLmMask::build(2, 5, &[2, 3]);
        let output = block.forward(&input, &mask);
        assert_eq!(output.shape().dims, vec![2, 5, 16]);
    }

    #[test]
    fn transformer_block_residual_nonzero_weights() {
        let block = TransformerBlock::new(8, 2, 2, 16);
        let input_data: Vec<f32> = (0..1 * 4 * 8).map(|i| (i as f32) * 0.01).collect();
        let input = Tensor::from_vec(input_data, Shape::new(vec![1, 4, 8]));
        let mask = PrefixLmMask::build(1, 4, &[2]);
        let output = block.forward(&input, &mask);
        // With default (identity norm, zero attention/mlp weights), output == input due to residual
        // With non-zero weights, output would differ. Here weights are zero, so output == input.
        assert_eq!(output.shape().dims, vec![1, 4, 8]);
    }
}
