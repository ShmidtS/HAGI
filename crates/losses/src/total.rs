use clifford_core::GRADE_LOOKUP_CL3;
use core_types::shape::Shape;
use tensor_runtime::{Tensor, TensorView};

use crate::auxiliary::AuxiliaryLoss;
use crate::cross_entropy::CrossEntropyLoss;
use crate::isomorphic::IsomorphicLoss;

#[derive(thiserror::Error, Debug, Clone, PartialEq)]
pub enum LossError {
    #[error("invalid logits rank: expected 3, got {0}")]
    InvalidLogitsRank(usize),
    #[error("invalid targets rank: expected 2, got {0}")]
    InvalidTargetsRank(usize),
    #[error("invalid prefix mask rank: expected 2, got {0}")]
    InvalidPrefixMaskRank(usize),
    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("invalid targets: {0}")]
    InvalidTargets(String),
    #[error("invalid weights: {0}")]
    InvalidWeights(String),
    #[error("target id {target} out of range for vocab {vocab}")]
    TargetOutOfRange { target: u32, vocab: usize },
}

pub struct AuxTargets<'a> {
    pub positive_pairs: &'a [(TensorView<'a, f32>, TensorView<'a, f32>)],
    pub negative_pairs: &'a [(TensorView<'a, f32>, TensorView<'a, f32>)],
    pub margin: f32,
}

pub struct IsoPairBatch<'a> {
    pub u_src: TensorView<'a, f32>,
    pub u_tgt: TensorView<'a, f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossWeights {
    pub lambda_aux: f32,
    pub lambda_iso_target: f32,
    pub iso_warmup_steps: usize,
}

impl Default for LossWeights {
    fn default() -> Self {
        LossWeights {
            lambda_aux: 0.01,
            lambda_iso_target: 0.01,
            iso_warmup_steps: 1000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossBreakdown {
    pub l_ce: f32,
    pub l_aux: f32,
    pub l_iso: f32,
    pub l_total: f32,
    pub lambda_iso: f32,
    pub response_token_count: usize,
}

pub fn lambda_iso(step: usize, target: f32, warmup: usize) -> f32 {
    if step == 0 || target == 0.0 {
        0.0
    } else if warmup == 0 || step >= warmup {
        target
    } else {
        target * (step as f32 / warmup as f32)
    }
}

fn validate_weights(weights: &LossWeights) -> Result<(), LossError> {
    if !weights.lambda_aux.is_finite() || !weights.lambda_iso_target.is_finite() {
        return Err(LossError::InvalidWeights(
            "loss weights must be finite".to_string(),
        ));
    }
    Ok(())
}

fn tensor_from_view<T: core_types::dtype::DTypeTag + Copy>(view: TensorView<'_, T>) -> Tensor<T> {
    Tensor::from_vec(view.data().to_vec(), view.shape().clone())
}

fn cosine_similarity(
    lhs: &TensorView<'_, f32>,
    rhs: &TensorView<'_, f32>,
) -> Result<f32, LossError> {
    if lhs.shape().dims != rhs.shape().dims {
        return Err(LossError::ShapeMismatch {
            expected: lhs.shape().dims.clone(),
            actual: rhs.shape().dims.clone(),
        });
    }

    let mut dot = 0.0f64;
    let mut lhs_norm_sq = 0.0f64;
    let mut rhs_norm_sq = 0.0f64;
    for (&a, &b) in lhs.data().iter().zip(rhs.data().iter()) {
        let a = a as f64;
        let b = b as f64;
        dot += a * b;
        lhs_norm_sq += a * a;
        rhs_norm_sq += b * b;
    }

    if lhs_norm_sq <= f64::EPSILON || rhs_norm_sq <= f64::EPSILON {
        return Ok(0.0);
    }

    Ok((dot / (lhs_norm_sq.sqrt() * rhs_norm_sq.sqrt())) as f32)
}

fn auxiliary_margin_loss(aux: &AuxTargets<'_>) -> Result<f32, LossError> {
    if aux.positive_pairs.len() != aux.negative_pairs.len() {
        return Err(LossError::InvalidTargets(
            "positive and negative pair counts must match".to_string(),
        ));
    }
    if aux.positive_pairs.is_empty() {
        return Ok(0.0);
    }

    let mut sum = 0.0f64;
    for (pos, neg) in aux.positive_pairs.iter().zip(aux.negative_pairs.iter()) {
        let pos_sim = cosine_similarity(&pos.0, &pos.1)?;
        let neg_sim = cosine_similarity(&neg.0, &neg.1)?;
        sum += (aux.margin - pos_sim + neg_sim).max(0.0) as f64;
    }

    Ok((sum / aux.positive_pairs.len() as f64) as f32)
}

fn isomorphic_mse(iso: &IsoPairBatch<'_>) -> Result<f32, LossError> {
    if iso.u_src.shape().dims != iso.u_tgt.shape().dims {
        return Err(LossError::ShapeMismatch {
            expected: iso.u_src.shape().dims.clone(),
            actual: iso.u_tgt.shape().dims.clone(),
        });
    }
    if iso.u_src.numel() == 0 {
        return Ok(0.0);
    }

    let mut sum_sq = 0.0f64;
    for (&src, &tgt) in iso.u_src.data().iter().zip(iso.u_tgt.data().iter()) {
        let diff = src as f64 - tgt as f64;
        sum_sq += diff * diff;
    }

    Ok((sum_sq / iso.u_src.numel() as f64) as f32)
}

pub fn total_loss(
    logits: TensorView<'_, f32>,
    targets: TensorView<'_, u32>,
    aux: &AuxTargets<'_>,
    iso: &IsoPairBatch<'_>,
    weights: &LossWeights,
    step: usize,
    prefix_mask: Option<TensorView<'_, u8>>,
) -> Result<LossBreakdown, LossError> {
    validate_weights(weights)?;
    if !aux.margin.is_finite() {
        return Err(LossError::InvalidTargets("aux margin must be finite".to_string()));
    }

    let logits_shape = logits.shape();
    if logits_shape.rank() != 3 {
        return Err(LossError::InvalidLogitsRank(logits_shape.rank()));
    }
    let batch = logits_shape.dims[0];
    let tokens = logits_shape.dims[1];
    let vocab = logits_shape.dims[2];
    let expected = vec![batch, tokens];

    let targets_shape = targets.shape();
    if targets_shape.rank() != 2 {
        return Err(LossError::InvalidTargetsRank(targets_shape.rank()));
    }
    if targets_shape.dims != expected {
        return Err(LossError::ShapeMismatch {
            expected: expected.clone(),
            actual: targets_shape.dims.clone(),
        });
    }

    if let Some(mask) = &prefix_mask {
        let mask_shape = mask.shape();
        if mask_shape.rank() != 2 {
            return Err(LossError::InvalidPrefixMaskRank(mask_shape.rank()));
        }
        if mask_shape.dims != expected {
            return Err(LossError::ShapeMismatch {
                expected: expected.clone(),
                actual: mask_shape.dims.clone(),
            });
        }
    }

    let mut response_token_count = 0usize;
    let mut loss_mask = Vec::with_capacity(batch * tokens);
    for i in 0..targets.data().len() {
        let target = targets.data()[i];
        if target as usize >= vocab {
            return Err(LossError::TargetOutOfRange { target, vocab });
        }
        let is_response = prefix_mask
            .as_ref()
            .is_none_or(|mask| mask.data()[i] == 0);
        if is_response {
            response_token_count += 1;
            loss_mask.push(1.0);
        } else {
            loss_mask.push(0.0);
        }
    }

    let logits_tensor = tensor_from_view(logits);
    let targets_tensor = tensor_from_view(targets);
    let loss_mask = Tensor::from_vec(loss_mask, Shape::new(vec![batch, tokens]));
    let l_ce = CrossEntropyLoss::new().forward(&logits_tensor, &targets_tensor, &loss_mask);
    let l_aux = auxiliary_margin_loss(aux)?;
    let l_iso = isomorphic_mse(iso)?;
    let lambda_iso = lambda_iso(step, weights.lambda_iso_target, weights.iso_warmup_steps);
    let l_total = l_ce + weights.lambda_aux * l_aux + lambda_iso * l_iso;

    Ok(LossBreakdown {
        l_ce,
        l_aux,
        l_iso,
        l_total,
        lambda_iso,
        response_token_count,
    })
}

/// Composite loss combining cross-entropy, auxiliary contrastive, and isomorphic transfer.
///
/// L_total = L_ce + lambda_aux * L_aux + lambda_iso * L_iso
pub struct CompositeLoss {
    pub ce: CrossEntropyLoss,
    pub aux: AuxiliaryLoss,
    pub iso: IsomorphicLoss,
    pub lambda_aux: f32,
    pub lambda_iso: f32,
    pub initial_lambda_iso: f32,
}

impl Default for CompositeLoss {
    fn default() -> Self {
        Self {
            ce: CrossEntropyLoss::new(),
            aux: AuxiliaryLoss::default(),
            iso: IsomorphicLoss::new(),
            lambda_aux: 0.1,
            lambda_iso: 0.01,
            initial_lambda_iso: 0.01,
        }
    }
}

impl CompositeLoss {
    pub fn new(lambda_aux: f32, lambda_iso: f32) -> Self {
        Self {
            ce: CrossEntropyLoss::new(),
            aux: AuxiliaryLoss::default(),
            iso: IsomorphicLoss::new(),
            lambda_aux,
            lambda_iso,
            initial_lambda_iso: lambda_iso,
        }
    }

    /// Computes composite loss.
    ///
    /// - `logits`: [B, T, V]
    /// - `hidden`: [B, T, hidden_size]
    /// - `mv_original`: [B, T, heads, blades] before transfer
    /// - `mv_transferred`: [B, T, heads, blades] after round-trip transfer
    /// - `targets`: [B, T]
    /// - `loss_mask`: [B, T]
    ///
    /// Returns `(total, ce_component, aux_component, iso_component)`.
    pub fn forward(
        &self,
        logits: &Tensor<f32>,
        hidden: &Tensor<f32>,
        mv_original: &Tensor<f32>,
        mv_transferred: &Tensor<f32>,
        targets: &Tensor<u32>,
        loss_mask: &Tensor<f32>,
    ) -> (f32, f32, f32, f32) {
        let ce_val = self.ce.forward(logits, targets, loss_mask);
        let aux_val = self.aux.forward(hidden, targets);
        let iso_val = self.iso.forward(mv_original, mv_transferred);

        let total = ce_val + self.lambda_aux * aux_val + self.lambda_iso * iso_val;

        (total, ce_val, aux_val, iso_val)
    }

    /// Linearly anneals `lambda_iso` from `initial_lambda_iso` to 0.0 over `max_steps`.
    ///
    /// - At `step = 0`: lambda_iso = initial_lambda_iso
    /// - At `step >= max_steps`: lambda_iso = 0.0
    pub fn anneal_iso(&mut self, step: usize, max_steps: usize) {
        if max_steps == 0 || step >= max_steps {
            self.lambda_iso = 0.0;
        } else {
            let ratio = 1.0 - (step as f32 / max_steps as f32);
            self.lambda_iso = self.initial_lambda_iso * ratio;
        }
    }
}

pub fn clifford_grade_norm(grad: &Tensor<f32>, grade_lookup: &[u8]) -> Vec<f32> {
    if grade_lookup.is_empty() {
        return Vec::new();
    }

    let max_grade = grade_lookup.iter().copied().max().unwrap_or(0) as usize;
    let mut norm_sq = vec![0.0f64; max_grade + 1];
    for (i, &val) in grad.data().iter().enumerate() {
        let grade = grade_lookup[i % grade_lookup.len()] as usize;
        norm_sq[grade] += val as f64 * val as f64;
    }

    norm_sq.into_iter().map(|sum| sum.sqrt() as f32).collect()
}

/// MagicNorm-Clifford gradient clipping.
///
/// Clips CL3 coefficients independently by grade using the static grade lookup
/// `[scalar, vector, bivector, trivector]` repeated across the last tensor axis.
pub fn magic_norm_clip(gradients: &mut [Tensor<f32>], max_norm: f32) {
    if !max_norm.is_finite() {
        return;
    }

    for grad in gradients.iter_mut() {
        let grade_norms = clifford_grade_norm(grad, &GRADE_LOOKUP_CL3);
        let mut scales = vec![1.0f32; grade_norms.len()];
        for (grade, norm) in grade_norms.iter().copied().enumerate() {
            if norm > max_norm && norm > 1e-12 {
                scales[grade] = max_norm / norm;
            }
        }

        let mut view = grad.as_mut();
        let data = view.data_mut();
        for (i, val) in data.iter_mut().enumerate() {
            let grade = GRADE_LOOKUP_CL3[i % GRADE_LOOKUP_CL3.len()] as usize;
            if grade < scales.len() {
                *val *= scales[grade];
            }
        }
    }
}

/// Creates a tensor filled with a given value (utility for gradient initialization).
pub fn tensor_full(val: f32, shape: Shape) -> Tensor<f32> {
    let numel = shape.numel();
    Tensor::from_vec(vec![val; numel], shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_inputs() -> (
        Tensor<f32>,
        Tensor<f32>,
        Tensor<f32>,
        Tensor<f32>,
        Tensor<u32>,
        Tensor<f32>,
    ) {
        let logits = Tensor::from_vec(vec![0.0f32; 1 * 2 * 4], Shape::new(vec![1, 2, 4]));
        let hidden = Tensor::from_vec(vec![0.5f32; 1 * 2 * 8], Shape::new(vec![1, 2, 8]));
        let mv_orig = Tensor::from_vec(vec![1.0f32; 1 * 2 * 2 * 4], Shape::new(vec![1, 2, 2, 4]));
        let mv_trans = Tensor::from_vec(vec![1.0f32; 1 * 2 * 2 * 4], Shape::new(vec![1, 2, 2, 4]));
        let targets = Tensor::from_vec(vec![0u32, 1], Shape::new(vec![1, 2]));
        let mask = Tensor::from_vec(vec![1.0f32, 1.0], Shape::new(vec![1, 2]));
        (logits, hidden, mv_orig, mv_trans, targets, mask)
    }

    #[test]
    fn composite_returns_all_components() {
        let loss = CompositeLoss::new(0.1, 0.01);
        let (logits, hidden, mv_orig, mv_trans, targets, mask) = small_inputs();

        let (total, ce, aux, iso) =
            loss.forward(&logits, &hidden, &mv_orig, &mv_trans, &targets, &mask);

        // mv_orig == mv_trans => iso = 0
        assert_eq!(iso, 0.0);
        // total = ce + 0.1 * aux + 0.01 * 0.0
        let expected = ce + 0.1 * aux;
        assert!(
            (total - expected).abs() < 1e-5,
            "expected {}, got {}",
            expected,
            total
        );
    }

    #[test]
    fn anneal_iso_at_zero_returns_initial() {
        let mut loss = CompositeLoss::new(0.1, 0.5);
        loss.anneal_iso(0, 100);
        assert!((loss.lambda_iso - 0.5).abs() < 1e-6);
    }

    #[test]
    fn anneal_iso_at_max_returns_zero() {
        let mut loss = CompositeLoss::new(0.1, 0.5);
        loss.anneal_iso(100, 100);
        assert_eq!(loss.lambda_iso, 0.0);
    }

    #[test]
    fn anneal_iso_at_half_returns_half() {
        let mut loss = CompositeLoss::new(0.1, 0.5);
        loss.anneal_iso(50, 100);
        assert!((loss.lambda_iso - 0.25).abs() < 1e-6);
    }

    #[test]
    fn anneal_iso_beyond_max_returns_zero() {
        let mut loss = CompositeLoss::new(0.1, 0.5);
        loss.anneal_iso(200, 100);
        assert_eq!(loss.lambda_iso, 0.0);
    }

    #[test]
    fn magic_norm_clip_no_scale_when_under() {
        let mut grads = vec![Tensor::from_vec(vec![0.1f32, 0.2], Shape::new(vec![2]))];
        let original = grads[0].data().to_vec();
        magic_norm_clip(&mut grads, 10.0);
        assert_eq!(grads[0].data(), &original);
    }

    #[test]
    fn magic_norm_clip_scales_grade_wise_when_over() {
        let mut grads = vec![Tensor::from_vec(vec![3.0f32, 4.0], Shape::new(vec![2]))];
        magic_norm_clip(&mut grads, 1.0);
        let data = grads[0].data();
        assert!((data[0] - 1.0).abs() < 1e-5, "expected 1.0, got {}", data[0]);
        assert!((data[1] - 1.0).abs() < 1e-5, "expected 1.0, got {}", data[1]);
    }

    #[test]
    fn magic_norm_clip_multiple_tensors() {
        let mut grads = vec![
            Tensor::from_vec(vec![3.0f32], Shape::new(vec![1])),
            Tensor::from_vec(vec![4.0f32], Shape::new(vec![1])),
        ];
        magic_norm_clip(&mut grads, 2.5);
        assert!((grads[0].data()[0] - 2.5).abs() < 1e-5);
        assert!((grads[1].data()[0] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn magic_norm_clip_zero_max_norm() {
        let mut grads = vec![Tensor::from_vec(vec![1.0f32, 2.0], Shape::new(vec![2]))];
        magic_norm_clip(&mut grads, 0.0);
        // max_norm=0, norm>0, scale = 0/norm = 0
        assert_eq!(grads[0].data(), &[0.0, 0.0]);
    }
}
