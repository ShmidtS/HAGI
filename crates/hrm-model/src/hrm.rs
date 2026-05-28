use config::hrm::HrmConfig;
use core_types::dtype::DTypeTag;
use core_types::shape::Shape;
use data::PackedBatch;
use tensor_runtime::Tensor;

use crate::lm_head::LmHead;
use crate::mask::PrefixLmMask;
use crate::recurrence::{check_convergence, scheduled_bp_steps, HRMState};
use crate::transformer::{TransformerBlock, TransformerStack};

/// Typed output of HRM backbone: hidden states with guaranteed shape [B, T, hidden_size].
#[derive(Debug, Clone)]
pub struct HiddenState<T: DTypeTag>(Tensor<T>);

#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Tensor<f32>,
    pub sequence_output: Option<(usize, usize)>,
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self::new_with_sequence_output(input_dim, output_dim, None)
    }

    pub fn new_sequence(input_dim: usize, seq_len: usize, hidden_size: usize) -> Self {
        Self::new_with_sequence_output(input_dim, seq_len * hidden_size, Some((seq_len, hidden_size)))
    }

    fn new_with_sequence_output(
        input_dim: usize,
        output_dim: usize,
        sequence_output: Option<(usize, usize)>,
    ) -> Self {
        let scale = (2.0 / input_dim as f64).sqrt() as f32;
        let data = (0..input_dim * output_dim)
            .map(|i| (((i as f64 * 0.618033988).fract() - 0.5) as f32 * 2.0) * scale)
            .collect();
        Self {
            weight: Tensor::from_vec(data, Shape::new(vec![input_dim, output_dim])),
            sequence_output,
        }
    }

    pub fn project(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let input_shape = input.shape();
        assert_eq!(input_shape.rank(), 2, "linear input must be [B, input_dim]");
        let batch = input_shape.dims[0];
        let input_dim = input_shape.dims[1];
        let weight_shape = self.weight.shape();
        assert_eq!(weight_shape.rank(), 2, "linear weight must be [input_dim, output_dim]");
        assert_eq!(input_dim, weight_shape.dims[0], "linear input dim mismatch");
        let output_dim = weight_shape.dims[1];
        let input_data = input.data();
        let weight_data = self.weight.data();
        let mut out = vec![0.0f32; batch * output_dim];

        for b in 0..batch {
            for o in 0..output_dim {
                let mut acc = 0.0f32;
                for i in 0..input_dim {
                    acc += input_data[b * input_dim + i] * weight_data[i * output_dim + o];
                }
                out[b * output_dim + o] = acc;
            }
        }

        Tensor::from_vec(out, Shape::new(vec![batch, output_dim]))
    }
}

#[derive(Debug, Clone)]
pub struct HState {
    pub data: Tensor<f32>,
}

#[derive(Debug, Clone)]
pub struct LState {
    pub data: Tensor<f32>,
}

impl HState {
    pub fn project_to_hidden(&self, proj: &Linear) -> Tensor<f32> {
        project_state_to_hidden(&self.data, proj)
    }

    pub fn from_hidden_pool(hidden: &Tensor<f32>, pool: &Linear) -> Self {
        Self {
            data: pool.project(&mean_over_tokens(hidden)),
        }
    }
}

impl LState {
    pub fn project_to_hidden(&self, proj: &Linear) -> Tensor<f32> {
        project_state_to_hidden(&self.data, proj)
    }

    pub fn from_hidden_pool(hidden: &Tensor<f32>, pool: &Linear) -> Self {
        Self {
            data: pool.project(&mean_over_tokens(hidden)),
        }
    }
}

impl<T: DTypeTag> HiddenState<T> {
    pub fn new(tensor: Tensor<T>) -> Self {
        Self(tensor)
    }
    pub fn into_tensor(self) -> Tensor<T> {
        self.0
    }
    pub fn as_tensor(&self) -> &Tensor<T> {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetachTrace {
    pub bp_steps: usize,
    pub total_recurrence_steps: usize,
    pub detached_steps: usize,
    pub traced_steps: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HrmRuntimeControl {
    pub h_cycles: usize,
    pub l_cycles: usize,
    pub convergence_eps: f32,
    pub bp_steps: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum HrmError {
    #[error("invalid token rank: expected 2, got {0}")]
    InvalidTokenRank(usize),
    #[error("invalid state shape: expected {expected:?}, got h={actual_h:?} l={actual_l:?}")]
    InvalidStateShape {
        expected: Shape,
        actual_h: Shape,
        actual_l: Shape,
    },
    #[error("prefix mask shape mismatch")]
    PrefixMaskShapeMismatch,
    #[error("partition shape mismatch")]
    PartitionShapeMismatch,
    #[error("prefix mask does not match partition prefix metadata")]
    PrefixMaskPartitionMismatch,
    #[error("tensor error: {0}")]
    Tensor(#[from] tensor_runtime::TensorError),
}

/// Output of HrmBackbone forward pass.
#[derive(Debug, Clone)]
pub struct HrmOutput {
    pub final_state: HRMState,
    pub effective_h_cycles: usize,
    pub effective_l_cycles: usize,
    pub mask_checksum: u64,
    pub detach_trace: DetachTrace,
    pub hidden: Tensor<f32>,
    pub bp_steps: usize,
}

/// HRM recurrence scheduler and backbone.
pub struct HrmBackbone {
    pub config: HrmConfig,
    pub l_stack: TransformerStack,
    pub h_stack: TransformerStack,
}

impl HrmBackbone {
    pub fn from_config(config: &HrmConfig) -> Self {
        Self {
            l_stack: Self::build_stack(
                config.l_layers,
                config.hidden_size,
                config.num_heads,
                config.expansion,
                config.max_seq_len,
            ),
            h_stack: Self::build_stack(
                config.h_layers,
                config.hidden_size,
                config.num_heads,
                config.expansion,
                config.max_seq_len,
            ),
            config: config.clone(),
        }
    }

    fn build_stack(
        layers: usize,
        hidden_size: usize,
        num_heads: usize,
        expansion: usize,
        max_seq_len: usize,
    ) -> TransformerStack {
        let blocks = (0..layers)
            .map(|_| TransformerBlock::new(hidden_size, num_heads, expansion, max_seq_len))
            .collect();
        TransformerStack::new(blocks)
    }

    /// Forward pass with nested H/L cycles, early exit, and scheduled BP depth.
    ///
    /// - `input`: [B, T, hidden_size]
    /// - `prefix_lens`: per-batch prefix lengths for PrefixLM mask
    /// - `step`: training step for BP scheduling
    #[allow(clippy::too_many_arguments)]
    pub fn forward_architecture_compliant(
        &mut self,
        tokens: &Tensor<i64>,
        z_h: &mut HState,
        z_l: &mut LState,
        prefix_mask: &Tensor<f32>,
        h_proj: &Linear,
        l_proj: &Linear,
        h_pool: &Linear,
        l_pool: &Linear,
        h_cycles: usize,
        l_cycles: usize,
    ) -> Result<Tensor<f32>, HrmError> {
        let token_shape = tokens.shape();
        if token_shape.rank() != 2 {
            return Err(HrmError::InvalidTokenRank(token_shape.rank()));
        }
        if prefix_mask.shape().dims != token_shape.dims {
            return Err(HrmError::PrefixMaskShapeMismatch);
        }

        let batch = token_shape.dims[0];
        let seq_len = token_shape.dims[1];
        let mut prefix_lens = vec![0usize; batch];
        for (b, prefix_len) in prefix_lens.iter_mut().enumerate() {
            for t in 0..seq_len {
                if prefix_mask.data()[b * seq_len + t] != 0.0 {
                    *prefix_len = t + 1;
                }
            }
        }
        let mask = PrefixLmMask::build(batch, seq_len, &prefix_lens);
        let lm_head = LmHead::new(self.config.vocab_size, self.config.hidden_size);
        let token_ids = tensor_i64_to_u32(tokens);
        let mut final_hidden = Tensor::zeros(Shape::new(vec![batch, seq_len, self.config.hidden_size]));

        for _ in 0..h_cycles {
            for _ in 0..l_cycles {
                let embedded = lm_head.embed_tokens(&token_ids)?;
                let x = add_tensors(&embedded, &z_l.project_to_hidden(l_proj));
                final_hidden = self.l_stack.forward(&x, &mask);
                z_l.data = LState::from_hidden_pool(&final_hidden, l_pool).data;
            }

            let pooled_h = HState::from_hidden_pool(&final_hidden, h_pool).data;
            z_h.data = scale_tensor(&add_tensors(&z_h.data, &pooled_h), 0.5);
            let h_hidden = z_h.project_to_hidden(h_proj);
            z_l.data = LState::from_hidden_pool(&h_hidden, l_pool).data;
        }

        Ok(lm_head.project(&final_hidden))
    }

    pub fn forward(&self, input: &Tensor<f32>, prefix_lens: &[usize], step: usize) -> HrmOutput {
        let shape = input.shape();
        assert_eq!(shape.rank(), 3, "input must be [B, T, D]");
        let b = shape.dims[0];
        assert_eq!(
            prefix_lens.len(),
            b,
            "prefix_lens length must match batch size"
        );

        let mask = PrefixLmMask::build(b, shape.dims[1], prefix_lens);
        let bp_steps = scheduled_bp_steps(&self.config, step);

        let mut z_h = input.clone();
        let mut z_l = input.clone();
        let mut h_eff = 0usize;
        let mut l_eff = 0usize;

        for _h in 0..self.config.h_cycles {
            let prev_z_h = z_h.clone();
            let mut this_l_eff = 0usize;

            for _l in 0..self.config.l_cycles {
                let prev_z_l = z_l.clone();
                let x = add_tensors(input, &project_z_l(&z_l));
                let l_out = self.l_stack.forward(&x, &mask);
                z_l = update_l(&l_out, input);
                this_l_eff += 1;
                if check_convergence(&z_l, &prev_z_l, self.config.convergence_eps) {
                    break;
                }
            }

            let h_out = self.h_stack.forward(&z_l, &mask);
            z_h = update_h(&prev_z_h, &h_out);
            h_eff += 1;
            l_eff += this_l_eff;

            let h_converged = check_convergence(&z_h, &prev_z_h, self.config.convergence_eps);
            z_l = reset_l(&z_h);
            if h_converged {
                break;
            }
        }

        let hidden = z_h.clone();
        let total_recurrence_steps = h_eff + l_eff;
        let traced_steps = total_recurrence_steps.min(bp_steps);
        let detach_trace = DetachTrace {
            bp_steps,
            total_recurrence_steps,
            detached_steps: total_recurrence_steps.saturating_sub(bp_steps),
            traced_steps,
        };

        HrmOutput {
            final_state: HRMState { z_h, z_l },
            effective_h_cycles: h_eff,
            effective_l_cycles: l_eff,
            mask_checksum: mask_checksum(&mask),
            detach_trace,
            hidden,
            bp_steps,
        }
    }
}

pub fn forward_hrm(
    model: &HrmBackbone,
    batch: &PackedBatch,
    initial_hidden: Tensor<f32>,
    state: HRMState,
    step: usize,
) -> Result<HrmOutput, HrmError> {
    let control = HrmRuntimeControl {
        h_cycles: model.config.h_cycles,
        l_cycles: model.config.l_cycles,
        convergence_eps: model.config.convergence_eps,
        bp_steps: scheduled_bp_steps(&model.config, step),
    };
    forward_hrm_with_control(model, batch, initial_hidden, state, control)
}

pub fn forward_hrm_with_control(
    model: &HrmBackbone,
    batch: &PackedBatch,
    initial_hidden: Tensor<f32>,
    state: HRMState,
    control: HrmRuntimeControl,
) -> Result<HrmOutput, HrmError> {
    let token_shape = batch.tokens.shape();
    if token_shape.rank() != 2 {
        return Err(HrmError::InvalidTokenRank(token_shape.rank()));
    }

    let b = token_shape.dims[0];
    let t = token_shape.dims[1];
    if batch.partition.batch_size != b || batch.partition.seq_len != t {
        return Err(HrmError::PartitionShapeMismatch);
    }
    if batch.prefix_mask.shape().dims != token_shape.dims {
        return Err(HrmError::PrefixMaskShapeMismatch);
    }

    let expected = Shape::new(vec![b, t, model.config.hidden_size]);
    if initial_hidden.shape() != &expected || state.z_h.shape() != &expected || state.z_l.shape() != &expected {
        return Err(HrmError::InvalidStateShape {
            expected,
            actual_h: state.z_h.shape().clone(),
            actual_l: state.z_l.shape().clone(),
        });
    }

    let prefix_lens = prefix_lens_from_partition(batch)?;
    validate_prefix_mask_matches_partition(batch, &prefix_lens)?;
    let mask = prefix_mask_from_tensor(batch, b, t);
    let mask_checksum = mask.checksum();
    let bp_steps = control.bp_steps;

    let mut z_h = state.z_h;
    let mut z_l = state.z_l;
    let mut h_eff = 0usize;
    let mut l_eff = 0usize;

    for _h in 0..control.h_cycles {
        let prev_z_h = z_h.clone();
        let mut this_l_eff = 0usize;

        for _l in 0..control.l_cycles {
            let prev_z_l = z_l.clone();
            let x = add_tensors(&initial_hidden, &project_z_l(&z_l));
            let l_out = model.l_stack.forward(&x, &mask);
            z_l = update_l(&l_out, &initial_hidden);
            this_l_eff += 1;
            if check_convergence(&z_l, &prev_z_l, control.convergence_eps) {
                break;
            }
        }

        let h_out = model.h_stack.forward(&z_l, &mask);
        z_h = update_h(&prev_z_h, &h_out);
        h_eff += 1;
        l_eff += this_l_eff;

        let h_converged = check_convergence(&z_h, &prev_z_h, control.convergence_eps);
        z_l = reset_l(&z_h);
        if h_converged {
            break;
        }
    }

    let hidden = z_h.clone();
    let total_recurrence_steps = h_eff + l_eff;
    let traced_steps = total_recurrence_steps.min(bp_steps);
    let detach_trace = DetachTrace {
        bp_steps,
        total_recurrence_steps,
        detached_steps: total_recurrence_steps.saturating_sub(bp_steps),
        traced_steps,
    };

    Ok(HrmOutput {
        final_state: HRMState { z_h, z_l },
        effective_h_cycles: h_eff,
        effective_l_cycles: l_eff,
        mask_checksum,
        detach_trace,
        hidden,
        bp_steps,
    })
}

fn prefix_lens_from_partition(batch: &PackedBatch) -> Result<Vec<usize>, HrmError> {
    let mut prefix_lens = vec![0usize; batch.partition.batch_size];
    for span in &batch.partition.spans {
        if span.batch_index >= batch.partition.batch_size
            || span.start + span.len > batch.partition.seq_len
        {
            return Err(HrmError::PartitionShapeMismatch);
        }
        let prefix_end = span.start + span.prefix_len;
        if prefix_end > batch.partition.seq_len || span.prefix_len > span.len {
            return Err(HrmError::PartitionShapeMismatch);
        }
        prefix_lens[span.batch_index] = prefix_lens[span.batch_index].max(prefix_end);
    }
    Ok(prefix_lens)
}

fn validate_prefix_mask_matches_partition(
    batch: &PackedBatch,
    prefix_lens: &[usize],
) -> Result<(), HrmError> {
    let seq_len = batch.partition.seq_len;
    for (b, &prefix_len) in prefix_lens.iter().enumerate() {
        for t in 0..seq_len {
            let expected = u8::from(t < prefix_len);
            if batch.prefix_mask.data()[b * seq_len + t] != expected {
                return Err(HrmError::PrefixMaskPartitionMismatch);
            }
        }
    }
    Ok(())
}

fn prefix_mask_from_tensor(batch: &PackedBatch, batch_size: usize, seq_len: usize) -> PrefixLmMask {
    let total = batch_size * seq_len * seq_len;
    let mut bits = vec![false; total];
    let prefix_data = batch.prefix_mask.data();

    for b in 0..batch_size {
        let base = b * seq_len * seq_len;
        for i in 0..seq_len {
            for j in 0..seq_len {
                let query_is_prefix = prefix_data[b * seq_len + i] == 1;
                let key_is_prefix = prefix_data[b * seq_len + j] == 1;
                let attend = if query_is_prefix {
                    key_is_prefix
                } else {
                    key_is_prefix || j <= i
                };
                bits[base + i * seq_len + j] = attend;
            }
        }
    }

    PrefixLmMask {
        shape: Shape::new(vec![batch_size, seq_len, seq_len]),
        bits,
    }
}

fn mask_checksum(mask: &PrefixLmMask) -> u64 {
    mask.checksum()
}

fn project_state_to_hidden(state: &Tensor<f32>, proj: &Linear) -> Tensor<f32> {
    let projected = proj.project(state);
    let shape = projected.shape();
    let batch = shape.dims[0];
    let (seq_len, hidden_size) = proj.sequence_output.unwrap_or((1, shape.dims[1]));
    assert_eq!(shape.dims[1], seq_len * hidden_size, "linear sequence output dim mismatch");
    Tensor::from_vec(projected.data().to_vec(), Shape::new(vec![batch, seq_len, hidden_size]))
}

fn mean_over_tokens(hidden: &Tensor<f32>) -> Tensor<f32> {
    let shape = hidden.shape();
    assert_eq!(shape.rank(), 3, "hidden must be [B, T, D]");
    let batch = shape.dims[0];
    let seq_len = shape.dims[1];
    let hidden_size = shape.dims[2];
    let mut out = vec![0.0f32; batch * hidden_size];

    for b in 0..batch {
        for t in 0..seq_len {
            let src_base = (b * seq_len + t) * hidden_size;
            let dst_base = b * hidden_size;
            for h in 0..hidden_size {
                out[dst_base + h] += hidden.data()[src_base + h] / seq_len as f32;
            }
        }
    }

    Tensor::from_vec(out, Shape::new(vec![batch, hidden_size]))
}

fn tensor_i64_to_u32(tokens: &Tensor<i64>) -> Tensor<u32> {
    let data = tokens.data().iter().map(|&token| token.max(0) as u32).collect();
    Tensor::from_vec(data, tokens.shape().clone())
}

fn project_z_l(z_l: &Tensor<f32>) -> Tensor<f32> {
    z_l.clone()
}

fn update_l(x: &Tensor<f32>, base: &Tensor<f32>) -> Tensor<f32> {
    sub_tensors(x, base)
}

fn update_h(prev_z_h: &Tensor<f32>, z_l: &Tensor<f32>) -> Tensor<f32> {
    scale_tensor(&add_tensors(prev_z_h, z_l), 0.5)
}

fn reset_l(z_h: &Tensor<f32>) -> Tensor<f32> {
    z_h.clone()
}

fn add_tensors(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
    assert_eq!(a.shape(), b.shape(), "add_tensors: shapes must match");
    let data = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(&x, &y)| x + y)
        .collect();
    Tensor::from_vec(data, a.shape().clone())
}

fn sub_tensors(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
    assert_eq!(a.shape(), b.shape(), "sub_tensors: shapes must match");
    let data = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(&x, &y)| x - y)
        .collect();
    Tensor::from_vec(data, a.shape().clone())
}

fn scale_tensor(tensor: &Tensor<f32>, scale: f32) -> Tensor<f32> {
    let data = tensor.data().iter().map(|&x| x * scale).collect();
    Tensor::from_vec(data, tensor.shape().clone())
}
