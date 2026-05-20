use tensor_runtime::Tensor;

/// Cross-entropy loss for next-token prediction.
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

    pub fn forward(
        &self,
        _logits: &Tensor<f32>,
        _targets: &Tensor<u32>,
        _loss_mask: &Tensor<f32>,
    ) -> f32 {
        // Returns zero loss.
        0.0
    }
}
