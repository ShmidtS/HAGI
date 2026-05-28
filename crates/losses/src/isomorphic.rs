use tensor_runtime::Tensor;

/// Isomorphic transfer fidelity loss.
///
/// Measures MSE between original projected multivectors and those that have
/// undergone a round-trip domain transfer (source -> target -> source).
/// A perfect transfer yields zero loss.
pub struct IsomorphicLoss;

impl Default for IsomorphicLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl IsomorphicLoss {
    pub fn new() -> Self {
        Self
    }

    /// Computes mean squared error between original and transferred multivectors.
    ///
    /// - `original`: [B, T, heads, blades] projected multivectors before transfer
    /// - `transferred`: [B, T, heads, blades] multivectors after round-trip transfer
    ///
    /// Returns MSE averaged over all elements.
    pub fn forward(&self, original: &Tensor<f32>, transferred: &Tensor<f32>) -> f32 {
        let o_shape = original.shape();
        let t_shape = transferred.shape();

        assert_eq!(o_shape.rank(), 4, "original must be [B, T, heads, blades]");
        assert_eq!(
            t_shape.rank(),
            4,
            "transferred must be [B, T, heads, blades]"
        );
        assert_eq!(
            o_shape.dims, t_shape.dims,
            "shape mismatch: original {:?} vs transferred {:?}",
            o_shape.dims, t_shape.dims
        );

        let n = original.numel();
        if n == 0 {
            return 0.0;
        }

        let o_data = original.data();
        let t_data = transferred.data();

        let mut sum_sq = 0.0f64;
        for i in 0..n {
            let diff = o_data[i] as f64 - t_data[i] as f64;
            sum_sq += diff * diff;
        }

        (sum_sq / n as f64) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::shape::Shape;

    #[test]
    fn identical_inputs_zero_loss() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = Tensor::from_vec(data.clone(), Shape::new(vec![1, 2, 2, 2]));
        let b = Tensor::from_vec(data, Shape::new(vec![1, 2, 2, 2]));

        let loss = IsomorphicLoss::new();
        let result = loss.forward(&a, &b);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn known_mse() {
        // Two tensors differing by constant 1.0 -> MSE = 1.0
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![1, 1, 2, 2]));
        let b = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], Shape::new(vec![1, 1, 2, 2]));

        let loss = IsomorphicLoss::new();
        let result = loss.forward(&a, &b);
        assert!((result - 1.0).abs() < 1e-6, "expected 1.0, got {}", result);
    }

    #[test]
    fn empty_tensor_zero_loss() {
        let a = Tensor::<f32>::zeros(Shape::new(vec![0, 0, 0, 0]));
        let b = Tensor::<f32>::zeros(Shape::new(vec![0, 0, 0, 0]));

        let loss = IsomorphicLoss::new();
        let result = loss.forward(&a, &b);
        assert_eq!(result, 0.0);
    }
}
