use tensor_runtime::Tensor;

/// Contrastive auxiliary loss that encourages hidden states with the same
/// target label to be close and those with different labels to be separated.
pub struct AuxiliaryLoss {
    pub margin: f32,
}

impl Default for AuxiliaryLoss {
    fn default() -> Self {
        Self { margin: 1.0 }
    }
}

impl AuxiliaryLoss {
    pub fn new(margin: f32) -> Self {
        Self { margin }
    }

    /// Computes contrastive loss over consecutive pairs in the flattened [B, T] layout.
    ///
    /// - `hidden`: [B, T, hidden_size]
    /// - `targets`: [B, T] (u32 class labels)
    ///
    /// For each consecutive pair (i, i+1):
    /// - Same target (positive): L = ||h_i - h_{i+1}||^2
    /// - Different target (negative): L = max(0, margin - ||h_i - h_{i+1}||^2)
    pub fn forward(&self, hidden: &Tensor<f32>, targets: &Tensor<u32>) -> f32 {
        let h_shape = hidden.shape();
        assert_eq!(h_shape.rank(), 3, "hidden must be [B, T, hidden]");
        let batch = h_shape.dims[0];
        let tokens = h_shape.dims[1];
        let hidden_dim = h_shape.dims[2];

        let t_shape = targets.shape();
        assert_eq!(t_shape.rank(), 2, "targets must be [B, T]");
        assert_eq!(t_shape.dims[0], batch);
        assert_eq!(t_shape.dims[1], tokens);

        let n = batch * tokens;
        if n < 2 {
            return 0.0;
        }

        let h_data = hidden.data();
        let t_data = targets.data();

        let mut total_loss = 0.0f64;
        let mut pair_count = 0u64;

        for i in 0..(n - 1) {
            let j = i + 1;
            let off_i = i * hidden_dim;
            let off_j = j * hidden_dim;

            // Squared L2 distance
            let mut dist_sq = 0.0f64;
            for d in 0..hidden_dim {
                let diff = h_data[off_i + d] as f64 - h_data[off_j + d] as f64;
                dist_sq += diff * diff;
            }

            let same_class = t_data[i] == t_data[j];
            let pair_loss = if same_class {
                // Positive pair: pull together
                dist_sq
            } else {
                // Negative pair: push apart up to margin
                let m = self.margin as f64;
                (m - dist_sq).max(0.0)
            };

            total_loss += pair_loss;
            pair_count += 1;
        }

        if pair_count == 0 {
            return 0.0;
        }

        (total_loss / pair_count as f64) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::shape::Shape;

    #[test]
    fn identical_states_same_class_zero_loss() {
        let hidden = Tensor::from_vec(vec![1.0f32; 2 * 1 * 4], Shape::new(vec![2, 1, 4]));
        let targets = Tensor::from_vec(vec![0u32, 0], Shape::new(vec![2, 1]));

        let loss = AuxiliaryLoss::new(1.0);
        let result = loss.forward(&hidden, &targets);
        assert!((result - 0.0).abs() < 1e-6, "expected 0.0, got {}", result);
    }

    #[test]
    fn far_apart_different_class_zero_loss() {
        // Two vectors far apart with different classes: margin - dist_sq < 0 -> 0
        let mut data = vec![0.0f32; 2 * 1 * 4];
        data[0] = 10.0;
        data[1] = 10.0;
        data[2] = 10.0;
        data[3] = 10.0;
        // second vector all zeros
        let hidden = Tensor::from_vec(data, Shape::new(vec![2, 1, 4]));
        let targets = Tensor::from_vec(vec![0u32, 1], Shape::new(vec![2, 1]));

        let loss = AuxiliaryLoss::new(1.0);
        let result = loss.forward(&hidden, &targets);
        assert!((result - 0.0).abs() < 1e-6, "expected 0.0, got {}", result);
    }

    #[test]
    fn close_different_class_positive_loss() {
        // Two identical vectors with different classes: margin - 0 = margin
        let hidden = Tensor::from_vec(vec![1.0f32; 2 * 1 * 4], Shape::new(vec![2, 1, 4]));
        let targets = Tensor::from_vec(vec![0u32, 1], Shape::new(vec![2, 1]));

        let loss = AuxiliaryLoss::new(1.0);
        let result = loss.forward(&hidden, &targets);
        assert!((result - 1.0).abs() < 1e-6, "expected 1.0, got {}", result);
    }

    #[test]
    fn single_element_zero_loss() {
        let hidden = Tensor::from_vec(vec![1.0f32; 1 * 1 * 4], Shape::new(vec![1, 1, 4]));
        let targets = Tensor::from_vec(vec![0u32], Shape::new(vec![1, 1]));

        let loss = AuxiliaryLoss::new(1.0);
        let result = loss.forward(&hidden, &targets);
        assert_eq!(result, 0.0);
    }
}
