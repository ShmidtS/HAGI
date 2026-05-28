use tensor_runtime::Tensor;

/// Cross-entropy loss for next-token prediction with log-sum-exp stability.
pub struct CrossEntropyLoss;

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self
    }

    /// Computes masked cross-entropy loss.
    ///
    /// - `logits`: [B, T, V] raw logits
    /// - `targets`: [B, T] target token ids
    /// - `loss_mask`: [B, T] mask (1.0 = count, 0.0 = ignore)
    ///
    /// Returns mean negative log-likelihood over masked positions.
    pub fn forward(
        &self,
        logits: &Tensor<f32>,
        targets: &Tensor<u32>,
        loss_mask: &Tensor<f32>,
    ) -> f32 {
        let shape = logits.shape();
        assert_eq!(shape.rank(), 3, "logits must be [B, T, V]");
        let batch = shape.dims[0];
        let tokens = shape.dims[1];
        let vocab = shape.dims[2];

        let t_shape = targets.shape();
        assert_eq!(t_shape.rank(), 2, "targets must be [B, T]");
        assert_eq!(t_shape.dims[0], batch);
        assert_eq!(t_shape.dims[1], tokens);

        let m_shape = loss_mask.shape();
        assert_eq!(m_shape.rank(), 2, "loss_mask must be [B, T]");
        assert_eq!(m_shape.dims[0], batch);
        assert_eq!(m_shape.dims[1], tokens);

        let logits_data = logits.data();
        let targets_data = targets.data();
        let mask_data = loss_mask.data();

        let mut total_loss = 0.0f64;
        let mut count = 0u64;

        for bt in 0..(batch * tokens) {
            let mask_val = mask_data[bt];
            if mask_val <= 0.0 {
                continue;
            }

            let target_id = targets_data[bt] as usize;
            assert!(
                target_id < vocab,
                "target id {} out of range (vocab={})",
                target_id,
                vocab
            );

            let offset = bt * vocab;

            // Numerically stable log-sum-exp
            let mut max_val = f32::NEG_INFINITY;
            for v in 0..vocab {
                let val = logits_data[offset + v];
                if val > max_val {
                    max_val = val;
                }
            }

            let mut sum_exp = 0.0f64;
            for v in 0..vocab {
                sum_exp += ((logits_data[offset + v] - max_val) as f64).exp();
            }
            let log_sum_exp = max_val as f64 + sum_exp.ln();

            let target_logit = logits_data[offset + target_id] as f64;
            let nll = log_sum_exp - target_logit;

            total_loss += nll * mask_val as f64;
            count += 1;
        }

        if count == 0 {
            return 0.0;
        }

        (total_loss / count as f64) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::shape::Shape;

    #[test]
    fn uniform_logits_give_log_vocab() {
        let vocab = 4usize;
        let logits = Tensor::from_vec(vec![0.0f32; 1 * 1 * vocab], Shape::new(vec![1, 1, vocab]));
        let targets = Tensor::from_vec(vec![0u32], Shape::new(vec![1, 1]));
        let mask = Tensor::from_vec(vec![1.0f32], Shape::new(vec![1, 1]));

        let loss = CrossEntropyLoss::new();
        let result = loss.forward(&logits, &targets, &mask);
        let expected = (vocab as f32).ln();
        assert!(
            (result - expected).abs() < 1e-4,
            "expected ~{}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn perfect_prediction_low_loss() {
        let vocab = 4usize;
        // Logit for target=2 is very high, others are low
        let logits = Tensor::from_vec(
            vec![-10.0, -10.0, 10.0, -10.0],
            Shape::new(vec![1, 1, vocab]),
        );
        let targets = Tensor::from_vec(vec![2u32], Shape::new(vec![1, 1]));
        let mask = Tensor::from_vec(vec![1.0f32], Shape::new(vec![1, 1]));

        let loss = CrossEntropyLoss::new();
        let result = loss.forward(&logits, &targets, &mask);
        assert!(result < 0.01, "expected near-zero loss, got {}", result);
    }

    #[test]
    fn mask_zero_gives_zero_loss() {
        let vocab = 4usize;
        let logits = Tensor::from_vec(vec![1.0; 1 * 1 * vocab], Shape::new(vec![1, 1, vocab]));
        let targets = Tensor::from_vec(vec![0u32], Shape::new(vec![1, 1]));
        let mask = Tensor::from_vec(vec![0.0f32], Shape::new(vec![1, 1]));

        let loss = CrossEntropyLoss::new();
        let result = loss.forward(&logits, &targets, &mask);
        assert_eq!(result, 0.0);
    }
}
