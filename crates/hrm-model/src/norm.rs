use core_types::shape::Shape;
use tensor_runtime::Tensor;

/// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
pub struct RMSNorm {
    pub weight: Tensor<f32>,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        let weight = Tensor::from_vec(vec![1.0f32; hidden_size], Shape::new(vec![hidden_size]));
        Self { weight, eps }
    }

    pub fn with_weight(weight: Tensor<f32>, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Forward pass: input shape [B, T, D] -> output shape [B, T, D]
    pub fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let shape = input.shape();
        assert!(
            shape.rank() >= 1,
            "RMSNorm input must have at least 1 dimension"
        );
        let d = shape.dims[shape.rank() - 1];
        let weight_data = self.weight.data();
        assert_eq!(
            weight_data.len(),
            d,
            "RMSNorm weight size must match last dim"
        );

        let input_data = input.data();
        let outer = input_data.len() / d;
        let mut output = vec![0.0f32; input_data.len()];

        for i in 0..outer {
            let offset = i * d;
            let slice = &input_data[offset..offset + d];

            let mean_sq = slice.iter().map(|&x| x * x).sum::<f32>() / d as f32;
            let inv_rms = 1.0 / (mean_sq + self.eps).sqrt();

            for j in 0..d {
                output[offset + j] = slice[j] * inv_rms * weight_data[j];
            }
        }

        Tensor::from_vec(output, shape.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_preserves_shape() {
        let norm = RMSNorm::new(8, 1e-6);
        let input = Tensor::from_vec(vec![1.0f32; 2 * 3 * 8], Shape::new(vec![2, 3, 8]));
        let output = norm.forward(&input);
        assert_eq!(output.shape().dims, vec![2, 3, 8]);
    }

    #[test]
    fn rmsnorm_mean_square_approximately_one() {
        let norm = RMSNorm::new(64, 1e-6);
        let data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.0).collect();
        let input = Tensor::from_vec(data, Shape::new(vec![1, 1, 64]));
        let output = norm.forward(&input);
        let out_data = output.data();
        let mean_sq = out_data.iter().map(|&x| x * x).sum::<f32>() / 64.0;
        assert!(
            (mean_sq - 1.0).abs() < 0.01,
            "mean square should be ~1.0, got {}",
            mean_sq
        );
    }
}
