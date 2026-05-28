use core_types::shape::Shape;
use tensor_runtime::Tensor;

/// SwiGLU MLP: gate = x @ W_gate, up = x @ W_up, hidden = swish(gate) * up, out = hidden @ W_down
pub struct SwiGlMlp {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub w_gate: Tensor<f32>,
    pub w_up: Tensor<f32>,
    pub w_down: Tensor<f32>,
}

impl SwiGlMlp {
    pub fn new(hidden_size: usize, expansion: usize) -> Self {
        let intermediate_size = hidden_size * expansion;
        let w_gate = Tensor::zeros(Shape::new(vec![hidden_size, intermediate_size]));
        let w_up = Tensor::zeros(Shape::new(vec![hidden_size, intermediate_size]));
        let w_down = Tensor::zeros(Shape::new(vec![intermediate_size, hidden_size]));
        Self {
            hidden_size,
            intermediate_size,
            w_gate,
            w_up,
            w_down,
        }
    }

    /// Forward pass: input [B, T, hidden_size] -> output [B, T, hidden_size]
    pub fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let shape = input.shape();
        assert_eq!(shape.rank(), 3, "MLP input must be [B, T, D]");
        let b = shape.dims[0];
        let t = shape.dims[1];
        let d = shape.dims[2];
        assert_eq!(d, self.hidden_size, "input last dim must match hidden_size");

        let bt = b * t;
        let d_in = self.hidden_size;
        let d_mid = self.intermediate_size;
        let input_data = input.data();

        // gate = x @ W_gate -> [B*T, intermediate]
        let gate = matmul_2d_batched(input_data, self.w_gate.data(), bt, d_in, d_mid);
        // up = x @ W_up -> [B*T, intermediate]
        let up = matmul_2d_batched(input_data, self.w_up.data(), bt, d_in, d_mid);

        // hidden = swish(gate) * up (elementwise)
        let mut hidden = vec![0.0f32; bt * d_mid];
        for i in 0..(bt * d_mid) {
            let g = gate[i];
            let swish_g = g * sigmoid(g);
            hidden[i] = swish_g * up[i];
        }

        // out = hidden @ W_down -> [B*T, hidden_size]
        let output = matmul_2d_batched(&hidden, self.w_down.data(), bt, d_mid, d_in);

        Tensor::from_vec(output, shape.clone())
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Batched matmul: [batch, d_in] @ [d_in, d_out] -> [batch, d_out]
fn matmul_2d_batched(
    input: &[f32],
    weight: &[f32],
    batch: usize,
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    assert_eq!(weight.len(), d_in * d_out);
    let mut output = vec![0.0f32; batch * d_out];
    for i in 0..batch {
        let in_off = i * d_in;
        let out_off = i * d_out;
        for j in 0..d_out {
            let mut sum = 0.0f32;
            for k in 0..d_in {
                sum += input[in_off + k] * weight[k * d_out + j];
            }
            output[out_off + j] = sum;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlp_output_shape_matches_input() {
        let mlp = SwiGlMlp::new(16, 4);
        let input = Tensor::from_vec(vec![0.5f32; 2 * 3 * 16], Shape::new(vec![2, 3, 16]));
        let output = mlp.forward(&input);
        assert_eq!(output.shape().dims, vec![2, 3, 16]);
    }
}
