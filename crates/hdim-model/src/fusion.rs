use crate::{HdimError, MultivectorBatch};
use clifford_core::Cl3;
use config::HdimConfig;
use core_types::{algebra::AlgebraSignature, shape::Shape};
use rand::Rng;
use tensor_runtime::Tensor;

/// Fuses structural (HDIM) features back into HRM hidden state via gated residual.
///
/// fused_h = h + sigmoid(gate(h, structural)) * proj(structural)
pub struct StructuralFusion {
    pub hidden_size: usize,
    pub structural_dim: usize,
    /// Gate weight: [hidden_size + structural_dim, hidden_size]
    pub w_gate: Tensor<f32>,
    /// Fuse weight: [structural_dim, hidden_size]
    pub w_fuse: Tensor<f32>,
}

impl StructuralFusion {
    pub fn new(hidden_size: usize, structural_dim: usize) -> Self {
        let gate_in = hidden_size + structural_dim;
        let gate_numel = gate_in * hidden_size;
        let fuse_numel = structural_dim * hidden_size;

        let gate_scale = (6.0 / (gate_in + hidden_size) as f64).sqrt() as f32;
        let fuse_scale = (6.0 / (structural_dim + hidden_size) as f64).sqrt() as f32;

        let mut rng = rand::thread_rng();
        let w_gate = Tensor::from_vec(
            (0..gate_numel)
                .map(|_| rng.gen_range(-gate_scale..gate_scale))
                .collect(),
            Shape::new(vec![gate_in, hidden_size]),
        );
        let w_fuse = Tensor::from_vec(
            (0..fuse_numel)
                .map(|_| rng.gen_range(-fuse_scale..fuse_scale))
                .collect(),
            Shape::new(vec![structural_dim, hidden_size]),
        );

        Self {
            hidden_size,
            structural_dim,
            w_gate,
            w_fuse,
        }
    }

    pub fn try_with_weights(
        w_gate: &Tensor<f32>,
        w_fuse: &Tensor<f32>,
        config: &HdimConfig,
    ) -> Result<Self, HdimError> {
        if w_gate.shape().rank() != 2 || w_fuse.shape().rank() != 2 {
            return Err(HdimError::InvalidConfig("fusion weight shape mismatch"));
        }
        let hidden_size = w_fuse.shape().dims[1];
        let structural_dim = config.structural_heads * Cl3::BLADE_COUNT;
        let gate_in = hidden_size + structural_dim;
        if w_gate.shape().dims != vec![gate_in, hidden_size]
            || w_fuse.shape().dims != vec![structural_dim, hidden_size]
        {
            return Err(HdimError::InvalidConfig("fusion weight shape mismatch"));
        }
        Ok(Self {
            hidden_size,
            structural_dim,
            w_gate: w_gate.clone(),
            w_fuse: w_fuse.clone(),
        })
    }

    /// Constructs with provided weight tensors (for testing with known weights).
    #[doc(hidden)]
    pub fn with_weights(
        hidden_size: usize,
        structural_dim: usize,
        w_gate: Tensor<f32>,
        w_fuse: Tensor<f32>,
    ) -> Self {
        assert_eq!(
            w_gate.shape().dims[1],
            hidden_size,
            "w_gate hidden size mismatch"
        );
        assert_eq!(
            w_fuse.shape().dims[1],
            hidden_size,
            "w_fuse hidden size mismatch"
        );
        let config = HdimConfig {
            structural_heads: structural_dim / Cl3::BLADE_COUNT,
            blade_count_per_head: Cl3::BLADE_COUNT,
            ..HdimConfig::default()
        };
        Self::try_with_weights(&w_gate, &w_fuse, &config)
            .expect("weight shape mismatch for StructuralFusion")
    }

    /// Fuses structural features into hidden state via gated residual.
    ///
    /// - h_state: [B, T, hidden_size]
    /// - structural: [B, T, structural_dim] (flattened from [B, T, heads, blades])
    /// - Returns: [B, T, hidden_size]
    pub fn forward_result(
        &self,
        h_state: &Tensor<f32>,
        structural: &Tensor<f32>,
    ) -> Result<Tensor<f32>, HdimError> {
        let h_shape = h_state.shape();
        let s_shape = structural.shape();

        if h_shape.rank() != 3 || s_shape.rank() < 3 {
            return Err(HdimError::ShapeMismatch);
        }
        let batch = h_shape.dims[0];
        let tokens = h_shape.dims[1];
        let hidden = h_shape.dims[2];
        if hidden != self.hidden_size {
            return Err(HdimError::ShapeMismatch);
        }

        // structural may be [B, T, heads, blades] (rank 4) or [B, T, structural_dim] (rank 3)
        let s_total: usize = s_shape.dims[2..].iter().product();
        if s_total != self.structural_dim || s_shape.dims[0] != batch || s_shape.dims[1] != tokens {
            return Err(HdimError::ShapeMismatch);
        }

        let h_data = h_state.data();
        let s_data = structural.data();
        let gate_w = self.w_gate.data();
        let fuse_w = self.w_fuse.data();

        let bt = batch * tokens;

        let mut out = vec![0.0f32; bt * hidden];

        for i in 0..bt {
            let h_off = i * hidden;
            let s_off = i * self.structural_dim;

            // Compute gate = sigmoid(concat @ W_gate) -> [hidden]
            // and fuse_proj = structural_flat @ W_fuse -> [hidden]
            let mut gate = vec![0.0f32; hidden];
            let mut fuse_proj = vec![0.0f32; hidden];

            // gate[j] = sum_k concat[k] * W_gate[k, j]
            for j in 0..hidden {
                let mut acc = 0.0f32;
                // h_state part
                for k in 0..hidden {
                    acc += h_data[h_off + k] * gate_w[k * hidden + j];
                }
                // structural part
                for k in 0..self.structural_dim {
                    acc += s_data[s_off + k] * gate_w[(hidden + k) * hidden + j];
                }
                gate[j] = sigmoid(acc);
            }

            // fuse_proj[j] = sum_k structural[k] * W_fuse[k, j]
            for j in 0..hidden {
                let mut acc = 0.0f32;
                for k in 0..self.structural_dim {
                    acc += s_data[s_off + k] * fuse_w[k * hidden + j];
                }
                fuse_proj[j] = acc;
            }

            // out[i, j] = h_state[i, j] + gate[j] * fuse_proj[j]
            for j in 0..hidden {
                out[h_off + j] = h_data[h_off + j] + gate[j] * fuse_proj[j];
            }
        }

        Ok(Tensor::from_vec(
            out,
            Shape::new(vec![batch, tokens, hidden]),
        ))
    }

    pub fn forward(&self, h_state: &Tensor<f32>, structural: &Tensor<f32>) -> Tensor<f32> {
        self.forward_result(h_state, structural)
            .expect("structural fusion shape mismatch")
    }
}

/// Injects HDIM structural signal into HRM hidden states with gated residual fusion.
///
/// `hrm_hidden` must be shaped `[batch, tokens, fusion.hidden_size]`; `hdim_signal.coeffs` must be
/// `[batch, tokens, structural_heads, A::BLADE_COUNT]` and flatten to `fusion.structural_dim`.
/// Shape mismatches return [`HdimError::ShapeMismatch`]. This implementation runs on CPU tensors
/// and does not dispatch to CUDA.
pub fn fused_hrm_hdim_inject<A: AlgebraSignature>(
    fusion: &StructuralFusion,
    hrm_hidden: &Tensor<f32>,
    hdim_signal: &MultivectorBatch<A>,
) -> Result<Tensor<f32>, HdimError> {
    fusion.forward_result(hrm_hidden, &hdim_signal.coeffs)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
