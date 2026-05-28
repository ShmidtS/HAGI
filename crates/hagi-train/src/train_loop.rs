use std::path::{Path, PathBuf};

use clifford_core::{get_product_table_cl3, Cl3, Multivector, Rotor};
use config::HdimConfig;
use core_types::{algebra::AlgebraSignature, ids::DomainId, shape::Shape};
use data::PackedBatch;
use hdim_model::{
    project_hidden_to_multivector, transfer_domain, HdimError, HdimForwardOutput,
    HiddenToMultivector, MemoryMode, StructuralFusion, TransferError, TransferRegistry,
    TransferState,
};
use hrm_model::forward_hrm_with_control;
use hrm_model::{forward_hrm, HRMState, HiddenState, HrmBackbone, HrmError, LmHead};
use losses::{total_loss, AuxTargets, IsoPairBatch, LossBreakdown, LossError, LossWeights};
use msa_adapter::{
    fetch_pages, route_top_k, sparse_attention_over_pages, HostKvCache, MemorySlot, MsaError,
    RoutingQueryView, SlotRegistry,
};
use nars_hdim::{transfer_domain_reasoned_or_fallback, NarsHdimConfig, NarsHdimReasoner};
use nars_hrm::{HrmPolicyLimits, NarsHrmConfig, NarsHrmController};
use nars_msa::{
    compute_reward_from_retrieval_outcome, route_top_k_with_nars, NarsMsaConfig, NarsMsaReasoner,
    NarsRoutePolicy,
};
use tensor_runtime::Tensor;

use crate::checkpoint;
use crate::optimizer::AdamW;

#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    #[error("hrm forward failed: {0}")]
    Hrm(#[from] HrmError),
    #[error("loss failed: {0}")]
    Loss(#[from] LossError),
    #[error("hdim projection failed: {0}")]
    Hdim(#[from] HdimError),
    #[error("hdim transfer failed: {0:?}")]
    Transfer(TransferError),
    #[error("msa routing failed: {0}")]
    Msa(#[from] MsaError),
    #[error("checkpoint failed: {0}")]
    Checkpoint(#[from] std::io::Error),
    #[error("empty response-token set")]
    EmptyResponse,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainStepReport {
    pub step: usize,
    pub loss: LossBreakdown,
    pub grad_norm: f32,
    pub eval_loss: Option<f32>,
    pub should_stop: bool,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct NarsTrainingConfig {
    pub enabled: bool,
    pub hrm_controller: NarsHrmConfig,
    pub hdim_reasoner: NarsHdimConfig,
    pub msa_reasoner: NarsMsaConfig,
}

pub struct TrainingLoop {
    pub backbone: HrmBackbone,
    pub projector: HiddenToMultivector,
    pub fusion: StructuralFusion,
    pub lm_head: LmHead,
    pub optimizer: AdamW,
    pub transfer_registry: TransferRegistry<Cl3>,
    pub checkpoint_dir: Option<PathBuf>,
    pub loss_weights: LossWeights,
    pub hrm_runtime_control: Option<hrm_model::HrmRuntimeControl>,
    pub nars_config: NarsTrainingConfig,
    pub nars_hrm_controller: Option<NarsHrmController>,
    pub nars_hdim_reasoner: Option<NarsHdimReasoner>,
    pub nars_msa_reasoner: Option<NarsMsaReasoner>,
    pub msa_slot_registry: SlotRegistry,
    pub msa_kv_cache: HostKvCache,
    pub eval_interval: usize,
    pub patience: usize,
    pub step: usize,
    pub last_transfer_was_degenerate: bool,
    best_eval_loss: Option<f32>,
    stale_eval_count: usize,
    last_loss_for_nars_feedback: Option<f32>,
}

impl TrainingLoop {
    pub fn try_new(
        backbone: HrmBackbone,
        optimizer: AdamW,
        loss_weights: LossWeights,
    ) -> Result<Self, TrainError> {
        let hidden_size = backbone.config.hidden_size;
        let hdim_config = HdimConfig::default();
        let structural_heads = hdim_config.structural_heads;
        let blade_count = Cl3::BLADE_COUNT;
        let structural_dim = structural_heads * blade_count;
        let mut transfer_registry = TransferRegistry::<Cl3>::new();
        transfer_registry
            .register_domain(DomainId(0), Rotor::unit(Multivector::<Cl3>::scalar_one()));
        transfer_registry
            .register_domain(DomainId(1), Rotor::unit(Multivector::<Cl3>::scalar_one()));

        let projector_weights =
            HiddenToMultivector::new(hidden_size, structural_heads, blade_count).w_proj;
        let fusion_weights = StructuralFusion::new(hidden_size, structural_dim);
        let projector = HiddenToMultivector::try_with_weights(&projector_weights, &hdim_config)?;
        let fusion = StructuralFusion::try_with_weights(
            &fusion_weights.w_gate,
            &fusion_weights.w_fuse,
            &hdim_config,
        )?;

        Ok(Self {
            projector,
            fusion,
            lm_head: LmHead::new(backbone.config.vocab_size, hidden_size),
            backbone,
            optimizer,
            transfer_registry,
            checkpoint_dir: None,
            loss_weights,
            hrm_runtime_control: None,
            nars_config: NarsTrainingConfig::default(),
            nars_hrm_controller: None,
            nars_hdim_reasoner: None,
            nars_msa_reasoner: None,
            msa_slot_registry: SlotRegistry::default(),
            msa_kv_cache: HostKvCache::default(),
            eval_interval: 0,
            patience: 3,
            step: 0,
            last_transfer_was_degenerate: false,
            best_eval_loss: None,
            stale_eval_count: 0,
            last_loss_for_nars_feedback: None,
        })
    }

    pub fn new(backbone: HrmBackbone, optimizer: AdamW, loss_weights: LossWeights) -> Self {
        Self::try_new(backbone, optimizer, loss_weights)
            .expect("TrainingLoop HDIM layer initialization failed")
    }

    pub fn with_nars(mut self, config: NarsTrainingConfig) -> Self {
        if config.enabled {
            if config.hrm_controller.enabled {
                if self.hrm_runtime_control.is_none() {
                    self.hrm_runtime_control = Some(hrm_model::HrmRuntimeControl {
                        h_cycles: self.backbone.config.h_cycles,
                        l_cycles: self.backbone.config.l_cycles,
                        convergence_eps: self.backbone.config.convergence_eps,
                        bp_steps: self.backbone.config.bp_max_steps,
                    });
                }
                self.nars_hrm_controller = Some(NarsHrmController::default());
            } else {
                self.nars_hrm_controller = None;
                self.hrm_runtime_control = None;
            }
            self.nars_hdim_reasoner = Some(NarsHdimReasoner::new(config.hdim_reasoner.clone()));
            self.nars_msa_reasoner = Some(NarsMsaReasoner::new(config.msa_reasoner.clone()));
        } else {
            self.nars_hdim_reasoner = None;
            self.nars_msa_reasoner = None;
            self.nars_hrm_controller = None;
            self.hrm_runtime_control = None;
        }
        self.nars_config = config;
        self
    }

    pub fn with_nars_hrm_controller(mut self, controller: NarsHrmController) -> Self {
        self.nars_hrm_controller = Some(controller);
        self
    }

    pub fn train_step(&mut self, batch: &PackedBatch) -> Result<TrainStepReport, TrainError> {
        let step = self.step;
        debug_assert_eq!(self.projector.blade_count_per_head, Cl3::BLADE_COUNT);
        let embedded = self.lm_head.embed_tokens(&batch.tokens)?;
        let state = HRMState::new(embedded.clone(), embedded.clone());
        if let Some(controller) = self.nars_hrm_controller.as_mut() {
            let policy = controller.begin_step(step, &self.backbone.config);
            let resolved = policy.resolve(&self.backbone.config, &HrmPolicyLimits::default());
            self.hrm_runtime_control = Some(hrm_model::HrmRuntimeControl {
                h_cycles: resolved.h_cycles,
                l_cycles: resolved.l_cycles,
                convergence_eps: resolved.convergence_eps,
                bp_steps: resolved.bp_steps,
            });
        }

        let hrm_out = if let Some(ref control) = self.hrm_runtime_control {
            forward_hrm_with_control(&self.backbone, batch, embedded.clone(), state, *control)?
        } else {
            forward_hrm(&self.backbone, batch, embedded.clone(), state, step)?
        };
        let mut hidden = hrm_out.hidden;
        let attention = self.forward_msa(&hidden)?;
        if hidden.shape() == attention.shape() {
            hidden = Tensor::from_vec(
                hidden
                    .data()
                    .iter()
                    .zip(attention.data().iter())
                    .map(|(&h, &a)| h + a)
                    .collect(),
                hidden.shape().clone(),
            );
        }
        let hidden_state = HiddenState::new(hidden.clone());
        let hdim_output = self.forward_hdim(&hidden, &hidden_state)?;
        let mv_projected = hdim_output.transfer_state.g_source.coeffs.clone();
        let mv_transferred = hdim_output.transfer_state.g_target.coeffs.clone();
        let logits = self.lm_head.project(&hdim_output.fused_hidden);

        let aux_targets = AuxTargets {
            positive_pairs: &[],
            negative_pairs: &[],
            margin: 0.5,
        };
        let iso_pair = IsoPairBatch {
            u_src: hdim_output.transfer_state.u_inv.coeffs.as_view(),
            u_tgt: hdim_output.transfer_state.g_target.coeffs.as_view(),
        };
        let loss = total_loss(
            logits.as_view(),
            batch.targets.as_view(),
            &aux_targets,
            &iso_pair,
            &self.loss_weights,
            step,
            Some(batch.prefix_mask.as_view()),
        )?;
        if loss.response_token_count == 0 {
            return Err(TrainError::EmptyResponse);
        }

        let d_l_d_fused = lm_head_input_grad(
            &logits,
            &batch.targets,
            &batch.prefix_mask,
            &self.lm_head.w_proj,
        );
        let (d_l_d_w_gate, d_l_d_w_fuse, d_l_d_structural) =
            fusion_backward(&hidden, &mv_projected, &self.fusion, &d_l_d_fused);
        let d_l_d_mv_iso = iso_grad(&mv_projected, &mv_transferred);
        let d_l_d_mv = Tensor::from_vec(
            d_l_d_structural
                .data()
                .iter()
                .zip(d_l_d_mv_iso.data().iter())
                .map(|(&a, &b)| a + b)
                .collect(),
            mv_projected.shape().clone(),
        );
        let d_l_d_w_proj = projector_backward(&hidden, &d_l_d_mv, &self.projector);
        let d_l_d_w_lm_head = lm_head_grad(
            &hdim_output.fused_hidden,
            &logits,
            &batch.targets,
            &batch.prefix_mask,
        );

        let mut grads = vec![d_l_d_w_proj, d_l_d_w_gate, d_l_d_w_fuse, d_l_d_w_lm_head];
        let grad_norm = global_norm(&grads);
        self.optimizer.clip_gradients(&mut grads);

        let mut params = vec![
            self.projector.w_proj.clone(),
            self.fusion.w_gate.clone(),
            self.fusion.w_fuse.clone(),
            self.lm_head.w_proj.clone(),
        ];
        self.optimizer.step(&mut params, &grads);
        self.projector.w_proj = params.remove(0);
        self.fusion.w_gate = params.remove(0);
        self.fusion.w_fuse = params.remove(0);
        self.lm_head.w_proj = params.remove(0);

        let eval_loss = self.eval_loss_if_due(loss.l_total);
        let should_stop = self.stale_eval_count >= self.patience;
        self.observe_hdim_feedback(loss.l_total);
        self.step += 1;

        let report = TrainStepReport {
            step,
            loss,
            grad_norm,
            eval_loss,
            should_stop,
        };
        if let Some(controller) = self.nars_hrm_controller.as_mut() {
            controller.end_step(&report);
        }

        Ok(report)
    }

    pub fn save_checkpoint(&self, path: &Path, step: u64) -> std::io::Result<()> {
        checkpoint::save_checkpoint(path, step, &self.named_tensors())
    }

    pub fn named_tensors(&self) -> Vec<(&str, &Tensor<f32>)> {
        vec![
            ("projector.w_proj", &self.projector.w_proj),
            ("fusion.w_gate", &self.fusion.w_gate),
            ("fusion.w_fuse", &self.fusion.w_fuse),
            ("lm_head.w_proj", &self.lm_head.w_proj),
        ]
    }

    fn forward_hdim(
        &mut self,
        hidden: &Tensor<f32>,
        hidden_state: &HiddenState<f32>,
    ) -> Result<HdimForwardOutput, TrainError> {
        let source_domain = DomainId(0);
        let degenerate_transfer = self.transfer_registry.domains.len() < 2;
        self.last_transfer_was_degenerate = degenerate_transfer;
        let target_domain = if degenerate_transfer {
            source_domain
        } else {
            DomainId(1)
        };

        let g_source = project_hidden_to_multivector::<Cl3>(&self.projector, hidden_state)?;
        let u_inv = if self.nars_config.enabled {
            if let Some(reasoner) = self.nars_hdim_reasoner.as_mut() {
                transfer_domain_reasoned_or_fallback(
                    &mut self.transfer_registry,
                    Some(source_domain),
                    Some(target_domain),
                    reasoner,
                    &g_source,
                    &self.nars_config.hdim_reasoner,
                )
                .map(|(batch, _)| batch)
                .map_err(TrainError::Transfer)?
            } else {
                transfer_domain(
                    &mut self.transfer_registry,
                    source_domain,
                    target_domain,
                    &g_source,
                    get_product_table_cl3(),
                )
                .map_err(TrainError::Transfer)?
            }
        } else {
            transfer_domain(
                &mut self.transfer_registry,
                source_domain,
                target_domain,
                &g_source,
                get_product_table_cl3(),
            )
            .map_err(TrainError::Transfer)?
        };
        let g_target = transfer_domain(
            &mut self.transfer_registry,
            target_domain,
            source_domain,
            &u_inv,
            get_product_table_cl3(),
        )
        .map_err(TrainError::Transfer)?;
        let fused_hidden = self.fusion.forward(hidden, &g_target.coeffs);

        Ok(HdimForwardOutput {
            fused_hidden,
            transfer_state: TransferState {
                g_source,
                u_inv,
                u_mem: None,
                u_route: None,
                g_target,
                memory_loss: 0.0,
                router_state: None,
                memory_mode: MemoryMode::Standard,
            },
        })
    }

    pub fn register_msa_slot(&mut self, slot: MemorySlot) {
        let slot_id = slot.id as u16;
        self.msa_kv_cache
            .append_page(slot_id, slot.key.clone(), slot.value.clone());
        self.msa_slot_registry.register(slot);
    }

    pub fn forward_msa(&mut self, hidden: &Tensor<f32>) -> Result<Tensor<f32>, TrainError> {
        if self.msa_slot_registry.is_empty() {
            return Ok(Tensor::zeros(hidden.shape().clone()));
        }

        let hidden_size = hidden.shape().dims[2];
        let selection = if self.nars_config.enabled {
            if let Some(reasoner) = self.nars_msa_reasoner.as_mut() {
                route_top_k_with_nars(
                    &self.msa_slot_registry,
                    RoutingQueryView {
                        data: hidden.data(),
                        dim: hidden_size,
                    },
                    reasoner,
                    &NarsRoutePolicy::default(),
                    self.step,
                )?
            } else {
                route_top_k(
                    &self.msa_slot_registry,
                    RoutingQueryView {
                        data: hidden.data(),
                        dim: hidden_size,
                    },
                    1,
                )?
            }
        } else {
            route_top_k(
                &self.msa_slot_registry,
                RoutingQueryView {
                    data: hidden.data(),
                    dim: hidden_size,
                },
                1,
            )?
        };
        let pages = fetch_pages(&self.msa_kv_cache, &selection.slot_ids).wait();
        let attention = sparse_attention_over_pages(hidden, &pages)?;
        if let Some(reasoner) = self.nars_msa_reasoner.as_mut() {
            let reward =
                compute_reward_from_retrieval_outcome(pages.len(), selection.slot_ids.len());
            for slot_id in selection.slot_ids {
                reasoner.observe_route_feedback(slot_id, reward, self.step);
            }
        }
        Ok(attention)
    }

    fn observe_hdim_feedback(&mut self, current_loss: f32) {
        if let Some(reasoner) = self.nars_hdim_reasoner.as_mut() {
            let success = self
                .last_loss_for_nars_feedback
                .map(|previous| current_loss <= previous)
                .unwrap_or(true);
            reasoner.observe_transfer_feedback(DomainId(0), DomainId(1), success);
        }
        self.last_loss_for_nars_feedback = Some(current_loss);
    }

    fn eval_loss_if_due(&mut self, train_loss: f32) -> Option<f32> {
        if self.eval_interval == 0 || !(self.step + 1).is_multiple_of(self.eval_interval) {
            return None;
        }

        let eval_loss = train_loss;
        match self.best_eval_loss {
            Some(best) if eval_loss >= best => {
                self.stale_eval_count += 1;
            }
            _ => {
                self.best_eval_loss = Some(eval_loss);
                self.stale_eval_count = 0;
            }
        }
        Some(eval_loss)
    }
}

pub fn train_step(
    trainer: &mut TrainingLoop,
    batch: &PackedBatch,
) -> Result<TrainStepReport, TrainError> {
    trainer.train_step(batch)
}

fn lm_head_grad(
    hidden: &Tensor<f32>,
    logits: &Tensor<f32>,
    targets: &Tensor<u32>,
    prefix_mask: &Tensor<u8>,
) -> Tensor<f32> {
    let shape = hidden.shape();
    let batch = shape.dims[0];
    let tokens = shape.dims[1];
    let hidden_size = shape.dims[2];
    let vocab = logits.shape().dims[2];
    let response_count = prefix_mask
        .data()
        .iter()
        .filter(|&&m| m == 0)
        .count()
        .max(1) as f32;
    let mut grad = vec![0.0f32; hidden_size * vocab];

    for bt in 0..(batch * tokens) {
        if prefix_mask.data()[bt] != 0 {
            continue;
        }

        let logit_offset = bt * vocab;
        let max_logit = logits.data()[logit_offset..logit_offset + vocab]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        let mut probs = vec![0.0f32; vocab];
        for (v, prob) in probs.iter_mut().enumerate().take(vocab) {
            let p = (logits.data()[logit_offset + v] - max_logit).exp();
            *prob = p;
            sum_exp += p;
        }

        let target = targets.data()[bt] as usize;
        for v in 0..vocab {
            let mut dlogit = probs[v] / sum_exp;
            if v == target {
                dlogit -= 1.0;
            }
            dlogit /= response_count;
            for h in 0..hidden_size {
                grad[h * vocab + v] += hidden.data()[bt * hidden_size + h] * dlogit;
            }
        }
    }

    Tensor::from_vec(grad, Shape::new(vec![hidden_size, vocab]))
}

fn global_norm(grads: &[Tensor<f32>]) -> f32 {
    let sum_sq: f32 = grads
        .iter()
        .flat_map(|grad| grad.data().iter())
        .map(|value| value * value)
        .sum();
    sum_sq.sqrt()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Gradient of CE loss w.r.t. fused hidden input to LM head.
/// logits = fused @ w_lm_head^T, so dL/d_fused = dL/d_logits @ w_lm_head.
fn lm_head_input_grad(
    logits: &Tensor<f32>,
    targets: &Tensor<u32>,
    prefix_mask: &Tensor<u8>,
    w_lm_head: &Tensor<f32>,
) -> Tensor<f32> {
    let shape = logits.shape();
    let batch = shape.dims[0];
    let tokens = shape.dims[1];
    let vocab = shape.dims[2];
    let hidden_size = w_lm_head.shape().dims[0];
    assert_eq!(w_lm_head.shape().dims[1], vocab);

    let response_count = prefix_mask
        .data()
        .iter()
        .filter(|&&m| m == 0)
        .count()
        .max(1) as f32;

    let mut grad = vec![0.0f32; batch * tokens * hidden_size];

    for bt in 0..(batch * tokens) {
        if prefix_mask.data()[bt] != 0 {
            continue;
        }

        let logit_offset = bt * vocab;
        let max_logit = logits.data()[logit_offset..logit_offset + vocab]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        let mut probs = vec![0.0f32; vocab];
        for (v, prob) in probs.iter_mut().enumerate().take(vocab) {
            let p = (logits.data()[logit_offset + v] - max_logit).exp();
            *prob = p;
            sum_exp += p;
        }

        let target = targets.data()[bt] as usize;
        for (v, &prob) in probs.iter().enumerate().take(vocab) {
            let mut dlogit = prob / sum_exp;
            if v == target {
                dlogit -= 1.0;
            }
            dlogit /= response_count;
            for h in 0..hidden_size {
                grad[bt * hidden_size + h] += dlogit * w_lm_head.data()[h * vocab + v];
            }
        }
    }

    Tensor::from_vec(grad, Shape::new(vec![batch, tokens, hidden_size]))
}

/// Backward pass through StructuralFusion.
/// Returns (dL/d_w_gate, dL/d_w_fuse, dL/d_structural).
fn fusion_backward(
    h_state: &Tensor<f32>,
    structural: &Tensor<f32>,
    fusion: &StructuralFusion,
    d_l_d_fused: &Tensor<f32>,
) -> (Tensor<f32>, Tensor<f32>, Tensor<f32>) {
    let h_shape = h_state.shape();
    let batch = h_shape.dims[0];
    let tokens = h_shape.dims[1];
    let hidden = h_shape.dims[2];
    let s_total: usize = structural.shape().dims[2..].iter().product();

    let bt = batch * tokens;
    let h_data = h_state.data();
    let s_data = structural.data();
    let gate_w = fusion.w_gate.data();
    let fuse_w = fusion.w_fuse.data();
    let d_l_d_out = d_l_d_fused.data();

    let mut d_l_d_w_gate = vec![0.0f32; fusion.w_gate.numel()];
    let mut d_l_d_w_fuse = vec![0.0f32; fusion.w_fuse.numel()];
    let mut d_l_d_structural = vec![0.0f32; bt * s_total];

    for i in 0..bt {
        let h_off = i * hidden;
        let s_off = i * s_total;

        let mut gate = vec![0.0f32; hidden];
        let mut fuse_proj = vec![0.0f32; hidden];

        for j in 0..hidden {
            let mut gate_input = 0.0f32;
            for k in 0..hidden {
                gate_input += h_data[h_off + k] * gate_w[k * hidden + j];
            }
            for k in 0..s_total {
                gate_input += s_data[s_off + k] * gate_w[(hidden + k) * hidden + j];
            }
            gate[j] = sigmoid(gate_input);

            let mut fuse_acc = 0.0f32;
            for k in 0..s_total {
                fuse_acc += s_data[s_off + k] * fuse_w[k * hidden + j];
            }
            fuse_proj[j] = fuse_acc;
        }

        for j in 0..hidden {
            let d_l_d_gate = d_l_d_out[h_off + j] * fuse_proj[j];
            let d_l_d_fuse_proj = d_l_d_out[h_off + j] * gate[j];
            let sigmoid_prime = gate[j] * (1.0 - gate[j]);

            for k in 0..s_total {
                d_l_d_w_fuse[k * hidden + j] += s_data[s_off + k] * d_l_d_fuse_proj;
                d_l_d_structural[s_off + k] += d_l_d_fuse_proj * fuse_w[k * hidden + j];

                d_l_d_w_gate[(hidden + k) * hidden + j] +=
                    s_data[s_off + k] * d_l_d_gate * sigmoid_prime;
                d_l_d_structural[s_off + k] +=
                    d_l_d_gate * sigmoid_prime * gate_w[(hidden + k) * hidden + j];
            }
            for k in 0..hidden {
                d_l_d_w_gate[k * hidden + j] += h_data[h_off + k] * d_l_d_gate * sigmoid_prime;
            }
        }
    }

    (
        Tensor::from_vec(d_l_d_w_gate, fusion.w_gate.shape().clone()),
        Tensor::from_vec(d_l_d_w_fuse, fusion.w_fuse.shape().clone()),
        Tensor::from_vec(d_l_d_structural, structural.shape().clone()),
    )
}

/// Backward pass through HiddenToMultivector projection.
fn projector_backward(
    hidden: &Tensor<f32>,
    d_l_d_mv: &Tensor<f32>,
    projector: &HiddenToMultivector,
) -> Tensor<f32> {
    let shape = hidden.shape();
    let batch = shape.dims[0];
    let tokens = shape.dims[1];
    let hidden_dim = shape.dims[2];
    let out_dim = projector.structural_heads * projector.blade_count_per_head;

    let hidden_data = hidden.data();
    let d_l_d_mv_data = d_l_d_mv.data();

    let mut grad = vec![0.0f32; hidden_dim * out_dim];

    for bt in 0..(batch * tokens) {
        let h_offset = bt * hidden_dim;
        let o_offset = bt * out_dim;
        for j in 0..out_dim {
            let dloss = d_l_d_mv_data[o_offset + j];
            for i in 0..hidden_dim {
                grad[i * out_dim + j] += hidden_data[h_offset + i] * dloss;
            }
        }
    }

    Tensor::from_vec(grad, Shape::new(vec![hidden_dim, out_dim]))
}

/// Gradient of iso MSE w.r.t. mv_projected.
fn iso_grad(mv_projected: &Tensor<f32>, mv_transferred: &Tensor<f32>) -> Tensor<f32> {
    let n = mv_projected.numel().max(1) as f32;
    let data: Vec<f32> = mv_projected
        .data()
        .iter()
        .zip(mv_transferred.data().iter())
        .map(|(&p, &t)| 2.0 * (p - t) / n)
        .collect();
    Tensor::from_vec(data, mv_projected.shape().clone())
}
