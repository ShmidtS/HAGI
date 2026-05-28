use crate::MultivectorBatch;
use clifford_core::Cl3;
use config::HdimConfig;
use core_types::{algebra::AlgebraSignature, shape::Shape};
use hrm_model::HiddenState;
use rand::Rng;
use tensor_runtime::Tensor;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HdimError {
    #[error("shape mismatch")]
    ShapeMismatch,
    #[error("invalid HDIM configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("domain not found")]
    DomainNotFound,
    #[error("transfer failed")]
    TransferFailed,
}

/// Projects hidden states into Clifford multivector space via learned linear projection.
pub struct HiddenToMultivector {
    pub hidden_size: usize,
    pub structural_heads: usize,
    pub blade_count_per_head: usize,
    /// Weight matrix: [hidden_size, structural_heads * blade_count_per_head]
    pub w_proj: Tensor<f32>,
}

impl HiddenToMultivector {
    pub fn new(hidden_size: usize, structural_heads: usize, blade_count_per_head: usize) -> Self {
        let out_dim = structural_heads * blade_count_per_head;
        let scale = (6.0 / (hidden_size + out_dim) as f64).sqrt() as f32;
        let mut rng = rand::thread_rng();
        let numel = hidden_size * out_dim;
        let data: Vec<f32> = (0..numel).map(|_| rng.gen_range(-scale..scale)).collect();
        let w_proj = Tensor::from_vec(data, Shape::new(vec![hidden_size, out_dim]));
        Self {
            hidden_size,
            structural_heads,
            blade_count_per_head,
            w_proj,
        }
    }

    pub fn try_with_weights(w_proj: &Tensor<f32>, config: &HdimConfig) -> Result<Self, HdimError> {
        let shape = w_proj.shape();
        if shape.rank() != 2 {
            return Err(HdimError::InvalidConfig("projection weight shape mismatch"));
        }
        let hidden_size = shape.dims[0];
        let structural_heads = config.structural_heads;
        let blade_count_per_head = Cl3::BLADE_COUNT;
        let expected_shape = Shape::new(vec![hidden_size, structural_heads * blade_count_per_head]);
        if w_proj.shape() != &expected_shape {
            return Err(HdimError::InvalidConfig("projection weight shape mismatch"));
        }
        Ok(Self {
            hidden_size,
            structural_heads,
            blade_count_per_head,
            w_proj: w_proj.clone(),
        })
    }

    /// Constructs with a provided weight tensor (for testing with known weights).
    #[doc(hidden)]
    pub fn with_weights(
        hidden_size: usize,
        structural_heads: usize,
        blade_count_per_head: usize,
        w_proj: Tensor<f32>,
    ) -> Self {
        assert_eq!(
            w_proj.shape().dims[0],
            hidden_size,
            "weight hidden size mismatch"
        );
        let config = HdimConfig {
            structural_heads,
            blade_count_per_head,
            ..HdimConfig::default()
        };
        Self::try_with_weights(&w_proj, &config)
            .expect("weight shape mismatch for HiddenToMultivector")
    }

    /// Projects [B, T, hidden_size] -> [B, T, structural_heads, blade_count_per_head].
    ///
    /// CPU reference: for each (batch, token), compute
    ///   hidden[hidden_size] @ W_proj[hidden_size, heads*blades] -> [heads*blades]
    /// then reshape to [heads, blades].
    pub fn forward_result(&self, hidden: &HiddenState<f32>) -> Result<Tensor<f32>, HdimError> {
        let tensor = hidden.as_tensor();
        let shape = tensor.shape();
        if shape.rank() != 3 {
            return Err(HdimError::ShapeMismatch);
        }
        let batch = shape.dims[0];
        let tokens = shape.dims[1];
        let hidden_dim = shape.dims[2];
        if hidden_dim != self.hidden_size {
            return Err(HdimError::ShapeMismatch);
        }

        let out_dim = self.structural_heads * self.blade_count_per_head;
        let hidden_data = tensor.data();
        let w_data = self.w_proj.data();

        let mut out = vec![0.0f32; batch * tokens * out_dim];

        for bt in 0..(batch * tokens) {
            let h_offset = bt * hidden_dim;
            let o_offset = bt * out_dim;
            // matmul: out[j] = sum_i hidden[i] * W_proj[i, j]
            for j in 0..out_dim {
                let mut acc = 0.0f32;
                for i in 0..hidden_dim {
                    acc += hidden_data[h_offset + i] * w_data[i * out_dim + j];
                }
                out[o_offset + j] = acc;
            }
        }

        Ok(Tensor::from_vec(
            out,
            Shape::new(vec![
                batch,
                tokens,
                self.structural_heads,
                self.blade_count_per_head,
            ]),
        ))
    }

    pub fn forward(&self, hidden: &HiddenState<f32>) -> Tensor<f32> {
        self.forward_result(hidden)
            .expect("hidden projection shape mismatch")
    }
}

/// Projects HRM hidden states into a typed multivector batch.
///
/// Input `hidden` must be shaped `[batch, tokens, projector.hidden_size]`; the output shape is
/// `[batch, tokens, projector.structural_heads, A::BLADE_COUNT]`. This path is CPU-only and does
/// not dispatch to CUDA.
pub fn project_hidden_to_multivector<A: AlgebraSignature>(
    projector: &HiddenToMultivector,
    hidden: &HiddenState<f32>,
) -> Result<MultivectorBatch<A>, HdimError> {
    if projector.blade_count_per_head != A::BLADE_COUNT {
        return Err(HdimError::InvalidConfig(
            "projector blade count does not match algebra",
        ));
    }
    Ok(MultivectorBatch::new(
        projector.forward_result(hidden)?,
        projector.structural_heads,
    ))
}
