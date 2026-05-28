use core_types::shape::Shape;
use tensor_runtime::Tensor;

use crate::hrm::HrmError;

/// Token embeddings + vocabulary projection.
pub struct LmHead {
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// Projection weight: [hidden_size, vocab_size]
    pub w_proj: Tensor<f32>,
}

impl LmHead {
    pub fn new(vocab_size: usize, hidden_size: usize) -> Self {
        // Initialize with small random-ish values using a simple deterministic pattern
        let numel = hidden_size * vocab_size;
        let scale = (2.0 / hidden_size as f64).sqrt() as f32;
        let data: Vec<f32> = (0..numel)
            .map(|i| {
                // Simple deterministic initialization
                let x = ((i as f64 * 0.618033988).fract() - 0.5) as f32 * 2.0;
                x * scale
            })
            .collect();
        let w_proj = Tensor::from_vec(data, Shape::new(vec![hidden_size, vocab_size]));
        Self {
            vocab_size,
            hidden_size,
            w_proj,
        }
    }

    pub fn embed(&self, input_ids: &Tensor<u32>) -> Tensor<f32> {
        self.embed_tokens(input_ids)
            .expect("embed requires rank-2 token tensor")
    }

    pub fn embed_tokens(&self, input_ids: &Tensor<u32>) -> Result<Tensor<f32>, HrmError> {
        let shape = input_ids.shape();
        if shape.rank() != 2 {
            return Err(HrmError::InvalidTokenRank(shape.rank()));
        }

        let b = shape.dims[0];
        let t = shape.dims[1];
        let mut out = vec![0.0f32; b * t * self.hidden_size];
        let token_data = input_ids.data();
        let weight_data = self.w_proj.data();

        for i in 0..(b * t) {
            let token = token_data[i] as usize % self.vocab_size;
            for h in 0..self.hidden_size {
                out[i * self.hidden_size + h] = weight_data[h * self.vocab_size + token];
            }
        }

        Ok(Tensor::from_vec(
            out,
            Shape::new(vec![b, t, self.hidden_size]),
        ))
    }

    /// Projects hidden states to vocabulary logits: [B, T, hidden] -> [B, T, vocab]
    pub fn project(&self, hidden: &Tensor<f32>) -> Tensor<f32> {
        let shape = hidden.shape();
        assert_eq!(shape.rank(), 3, "hidden must be [B, T, hidden]");
        let batch = shape.dims[0];
        let tokens = shape.dims[1];
        let hidden_dim = shape.dims[2];
        assert_eq!(hidden_dim, self.hidden_size, "hidden dim mismatch");

        let h_data = hidden.data();
        let w_data = self.w_proj.data();
        let bt = batch * tokens;

        let mut out = vec![0.0f32; bt * self.vocab_size];

        for i in 0..bt {
            let h_off = i * hidden_dim;
            let o_off = i * self.vocab_size;
            for v in 0..self.vocab_size {
                let mut acc = 0.0f32;
                for h in 0..hidden_dim {
                    acc += h_data[h_off + h] * w_data[h * self.vocab_size + v];
                }
                out[o_off + v] = acc;
            }
        }

        Tensor::from_vec(out, Shape::new(vec![batch, tokens, self.vocab_size]))
    }
}
