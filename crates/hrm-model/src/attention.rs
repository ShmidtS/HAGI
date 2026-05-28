use core_types::shape::Shape;
use tensor_runtime::Tensor;

use crate::mask::PrefixLmMask;
use crate::rope::RopeTable;

/// Multi-head self-attention with RoPE and PrefixLM mask.
pub struct MultiHeadSelfAttention {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub w_q: Tensor<f32>,
    pub w_k: Tensor<f32>,
    pub w_v: Tensor<f32>,
    pub w_o: Tensor<f32>,
    pub rope: RopeTable,
}

impl MultiHeadSelfAttention {
    pub fn new(hidden_size: usize, num_heads: usize, max_seq_len: usize) -> Self {
        assert_eq!(
            hidden_size % num_heads,
            0,
            "hidden_size must be divisible by num_heads"
        );
        let head_dim = hidden_size / num_heads;
        let w_q = Tensor::zeros(Shape::new(vec![hidden_size, hidden_size]));
        let w_k = Tensor::zeros(Shape::new(vec![hidden_size, hidden_size]));
        let w_v = Tensor::zeros(Shape::new(vec![hidden_size, hidden_size]));
        let w_o = Tensor::zeros(Shape::new(vec![hidden_size, hidden_size]));
        let rope = RopeTable::new(max_seq_len, head_dim, 10000.0);
        Self {
            hidden_size,
            num_heads,
            head_dim,
            w_q,
            w_k,
            w_v,
            w_o,
            rope,
        }
    }

    /// Forward pass: input [B, T, hidden_size] -> output [B, T, hidden_size]
    pub fn forward(&self, input: &Tensor<f32>, mask: &PrefixLmMask) -> Tensor<f32> {
        let shape = input.shape();
        assert_eq!(shape.rank(), 3, "attention input must be [B, T, D]");
        let b = shape.dims[0];
        let t = shape.dims[1];
        let d = shape.dims[2];
        assert_eq!(d, self.hidden_size, "input last dim must match hidden_size");

        let h = self.num_heads;
        let hd = self.head_dim;

        // QKV projections: [B, T, D] @ [D, D] -> [B, T, D]
        let q = linear_3d(input.data(), self.w_q.data(), b, t, d, d);
        let k = linear_3d(input.data(), self.w_k.data(), b, t, d, d);
        let v = linear_3d(input.data(), self.w_v.data(), b, t, d, d);

        // Reshape [B, T, H*D] -> [B, H, T, D]
        let mut q_bhtd = reshape_bt_hd_to_bhtd(&q, b, t, h, hd);
        let mut k_bhtd = reshape_bt_hd_to_bhtd(&k, b, t, h, hd);
        let v_bhtd = reshape_bt_hd_to_bhtd(&v, b, t, h, hd);

        // Apply RoPE to Q and K
        self.rope.apply(&mut q_bhtd, b, h, t);
        self.rope.apply(&mut k_bhtd, b, h, t);

        // Attention scores: [B, H, T, T] = Q @ K^T / sqrt(head_dim)
        let scale = 1.0 / (hd as f32).sqrt();
        let scores = batched_qk_scores(&q_bhtd, &k_bhtd, b, h, t, hd, scale);

        // Apply mask and softmax
        let probs = masked_softmax(&scores, mask, b, h, t);

        // Attn output: [B, H, T, D] = probs @ V
        let attn_out = batched_matmul_av(&probs, &v_bhtd, b, h, t, hd);

        // Reshape [B, H, T, D] -> [B, T, H*D]
        let attn_bt_hd = reshape_bhtd_to_bt_hd(&attn_out, b, h, t, hd);

        // Output projection: [B, T, D] @ [D, D] -> [B, T, D]
        let output = linear_3d(&attn_bt_hd, self.w_o.data(), b, t, d, d);

        Tensor::from_vec(output, shape.clone())
    }
}

/// Linear projection for [B, T, d_in] @ [d_in, d_out] -> [B, T, d_out]
fn linear_3d(
    input: &[f32],
    weight: &[f32],
    b: usize,
    t: usize,
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    assert_eq!(weight.len(), d_in * d_out);
    let bt = b * t;
    let mut output = vec![0.0f32; bt * d_out];
    for i in 0..bt {
        let in_offset = i * d_in;
        let out_offset = i * d_out;
        for j in 0..d_out {
            let mut sum = 0.0f32;
            for k in 0..d_in {
                sum += input[in_offset + k] * weight[k * d_out + j];
            }
            output[out_offset + j] = sum;
        }
    }
    output
}

/// Reshape [B, T, H*D] (row-major) to [B, H, T, D] (row-major).
fn reshape_bt_hd_to_bhtd(data: &[f32], b: usize, t: usize, h: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; b * h * t * d];
    for bi in 0..b {
        for ti in 0..t {
            for hi in 0..h {
                for di in 0..d {
                    let src = bi * t * h * d + ti * h * d + hi * d + di;
                    let dst = bi * h * t * d + hi * t * d + ti * d + di;
                    out[dst] = data[src];
                }
            }
        }
    }
    out
}

/// Reshape [B, H, T, D] (row-major) to [B, T, H*D] (row-major).
fn reshape_bhtd_to_bt_hd(data: &[f32], b: usize, h: usize, t: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; b * t * h * d];
    for bi in 0..b {
        for hi in 0..h {
            for ti in 0..t {
                for di in 0..d {
                    let src = bi * h * t * d + hi * t * d + ti * d + di;
                    let dst = bi * t * h * d + ti * h * d + hi * d + di;
                    out[dst] = data[src];
                }
            }
        }
    }
    out
}

/// Compute Q @ K^T / scale for [B, H, T, D] -> [B, H, T, T]
fn batched_qk_scores(
    q: &[f32],
    k: &[f32],
    b: usize,
    h: usize,
    t: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; b * h * t * t];
    for bh in 0..(b * h) {
        let bh_off = bh * t * d;
        let score_off = bh * t * t;
        for i in 0..t {
            for j in 0..t {
                let mut sum = 0.0f32;
                for dd in 0..d {
                    sum += q[bh_off + i * d + dd] * k[bh_off + j * d + dd];
                }
                scores[score_off + i * t + j] = sum * scale;
            }
        }
    }
    scores
}

/// Apply mask and softmax over last dimension.
/// scores: [B, H, T, T], mask: [B, T, T] (broadcast over H)
fn masked_softmax(scores: &[f32], mask: &PrefixLmMask, b: usize, h: usize, t: usize) -> Vec<f32> {
    let neg_inf = f32::NEG_INFINITY;
    let mut probs = vec![0.0f32; b * h * t * t];

    for bi in 0..b {
        for hi in 0..h {
            let bh_off = (bi * h + hi) * t * t;
            for i in 0..t {
                let row_off = bh_off + i * t;
                // Find max for numerical stability
                let mut max_val = f32::NEG_INFINITY;
                for j in 0..t {
                    let val = if mask.can_attend(bi, i, j) {
                        scores[row_off + j]
                    } else {
                        neg_inf
                    };
                    if val > max_val {
                        max_val = val;
                    }
                }
                // Exp and sum
                let mut sum = 0.0f32;
                for j in 0..t {
                    let val = if mask.can_attend(bi, i, j) {
                        scores[row_off + j]
                    } else {
                        neg_inf
                    };
                    let e = if val == neg_inf {
                        0.0
                    } else {
                        (val - max_val).exp()
                    };
                    probs[row_off + j] = e;
                    sum += e;
                }
                // Normalize
                if sum > 0.0 {
                    for j in 0..t {
                        probs[row_off + j] /= sum;
                    }
                }
            }
        }
    }
    probs
}

/// Batched matmul: [B, H, T, T] @ [B, H, T, D] -> [B, H, T, D]
fn batched_matmul_av(probs: &[f32], v: &[f32], b: usize, h: usize, t: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; b * h * t * d];
    for bh in 0..(b * h) {
        let bh_off = bh * t;
        for i in 0..t {
            let prob_row = bh_off * t + i * t;
            let out_off = bh_off * d + i * d;
            for j in 0..t {
                let p = probs[prob_row + j];
                if p == 0.0 {
                    continue;
                }
                let v_off = bh_off * d + j * d;
                for dd in 0..d {
                    out[out_off + dd] += p * v[v_off + dd];
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_output_shape_matches_input() {
        let hidden = 16;
        let heads = 4;
        let max_seq = 32;
        let attn = MultiHeadSelfAttention::new(hidden, heads, max_seq);
        let input = Tensor::from_vec(vec![0.1f32; 2 * 5 * hidden], Shape::new(vec![2, 5, hidden]));
        let mask = PrefixLmMask::build(2, 5, &[2, 3]);
        let output = attn.forward(&input, &mask);
        assert_eq!(output.shape().dims, vec![2, 5, hidden]);
    }
}
