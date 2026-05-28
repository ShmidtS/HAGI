use std::collections::HashMap;
use std::path::Path;

use hagi_train::load_checkpoint;
use hdim_model::{HiddenToMultivector, StructuralFusion};
use hrm_model::{HrmBackbone, LmHead};
use msa_adapter::SlotRegistry;
use tensor_runtime::Tensor;

use crate::report::{EvalConfig, EvalError};

pub struct HdimForward {
    pub projector: HiddenToMultivector,
    pub fusion: StructuralFusion,
}

pub struct EvalModel {
    pub backbone: HrmBackbone,
    pub hdim_forward: HdimForward,
    pub lm_head: LmHead,
    pub slot_registry: Option<SlotRegistry>,
}

pub fn load_checkpoint_for_eval(
    checkpoint_path: impl AsRef<Path>,
    config: &EvalConfig,
) -> Result<EvalModel, EvalError> {
    let (meta, tensors) = load_checkpoint(checkpoint_path.as_ref())?;
    let named_tensors: HashMap<&str, &Tensor<f32>> = meta
        .tensors
        .iter()
        .zip(tensors.iter())
        .map(|(tensor_meta, tensor)| (tensor_meta.name.as_str(), tensor))
        .collect();

    let hidden_size = config.hrm_config.hidden_size;
    let structural_heads = 1;
    let blade_count_per_head = 8;
    let structural_dim = structural_heads * blade_count_per_head;

    let projector_w = required_tensor(
        &named_tensors,
        "projector.w_proj",
        &[hidden_size, structural_dim],
    )?;
    let fusion_gate_w = required_tensor(
        &named_tensors,
        "fusion.w_gate",
        &[hidden_size + structural_dim, hidden_size],
    )?;
    let fusion_fuse_w = required_tensor(
        &named_tensors,
        "fusion.w_fuse",
        &[structural_dim, hidden_size],
    )?;
    let lm_head_w = required_tensor(
        &named_tensors,
        "lm_head.w_proj",
        &[hidden_size, config.hrm_config.vocab_size],
    )?;

    let projector = HiddenToMultivector::with_weights(
        hidden_size,
        structural_heads,
        blade_count_per_head,
        projector_w.clone(),
    );
    let fusion = StructuralFusion::with_weights(
        hidden_size,
        structural_dim,
        fusion_gate_w.clone(),
        fusion_fuse_w.clone(),
    );
    let lm_head = LmHead {
        vocab_size: config.hrm_config.vocab_size,
        hidden_size,
        w_proj: lm_head_w.clone(),
    };
    let backbone = build_backbone_from_checkpoint(&named_tensors, config)?;

    Ok(EvalModel {
        backbone,
        hdim_forward: HdimForward { projector, fusion },
        lm_head,
        slot_registry: None,
    })
}

fn required_tensor<'a>(
    named_tensors: &HashMap<&str, &'a Tensor<f32>>,
    name: &str,
    expected_shape: &[usize],
) -> Result<&'a Tensor<f32>, EvalError> {
    let tensor = named_tensors.get(name).copied().ok_or_else(|| {
        if named_tensors.keys().any(|key| is_eval_model_tensor(key)) {
            EvalError::CheckpointMismatch(format!("missing required tensor '{name}'"))
        } else {
            EvalError::CheckpointUnsupported(
                "checkpoint does not contain named eval model tensors".to_string(),
            )
        }
    })?;

    if tensor.shape().dims != expected_shape {
        return Err(EvalError::CheckpointMismatch(format!(
            "tensor '{name}' shape mismatch: expected {:?}, got {:?}",
            expected_shape,
            tensor.shape().dims
        )));
    }

    Ok(tensor)
}

fn is_eval_model_tensor(name: &str) -> bool {
    name == "projector.w_proj"
        || name == "fusion.w_gate"
        || name == "fusion.w_fuse"
        || name == "lm_head.w_proj"
        || name.starts_with("backbone.")
        || name.starts_with("l_stack.")
        || name.starts_with("h_stack.")
}

fn build_backbone_from_checkpoint(
    named_tensors: &HashMap<&str, &Tensor<f32>>,
    config: &EvalConfig,
) -> Result<HrmBackbone, EvalError> {
    let mut backbone = HrmBackbone::from_config(&config.hrm_config);
    apply_stack_tensors("l_stack", &mut backbone.l_stack.blocks, named_tensors)?;
    apply_stack_tensors("h_stack", &mut backbone.h_stack.blocks, named_tensors)?;
    Ok(backbone)
}

fn apply_stack_tensors(
    stack_name: &str,
    blocks: &mut [hrm_model::transformer::TransformerBlock],
    named_tensors: &HashMap<&str, &Tensor<f32>>,
) -> Result<(), EvalError> {
    for (index, block) in blocks.iter_mut().enumerate() {
        assign_optional_tensor(
            named_tensors,
            &format!("{stack_name}.{index}.norm1.weight"),
            &mut block.norm1.weight,
        )?;
        assign_optional_tensor(
            named_tensors,
            &format!("{stack_name}.{index}.attention.w_q"),
            &mut block.attention.w_q,
        )?;
        assign_optional_tensor(
            named_tensors,
            &format!("{stack_name}.{index}.attention.w_k"),
            &mut block.attention.w_k,
        )?;
        assign_optional_tensor(
            named_tensors,
            &format!("{stack_name}.{index}.attention.w_v"),
            &mut block.attention.w_v,
        )?;
        assign_optional_tensor(
            named_tensors,
            &format!("{stack_name}.{index}.attention.w_o"),
            &mut block.attention.w_o,
        )?;
        assign_optional_tensor(
            named_tensors,
            &format!("{stack_name}.{index}.norm2.weight"),
            &mut block.norm2.weight,
        )?;
        assign_optional_tensor(
            named_tensors,
            &format!("{stack_name}.{index}.mlp.w_gate"),
            &mut block.mlp.w_gate,
        )?;
        assign_optional_tensor(
            named_tensors,
            &format!("{stack_name}.{index}.mlp.w_up"),
            &mut block.mlp.w_up,
        )?;
        assign_optional_tensor(
            named_tensors,
            &format!("{stack_name}.{index}.mlp.w_down"),
            &mut block.mlp.w_down,
        )?;
    }
    Ok(())
}

fn assign_optional_tensor(
    named_tensors: &HashMap<&str, &Tensor<f32>>,
    name: &str,
    target: &mut Tensor<f32>,
) -> Result<(), EvalError> {
    if let Some(tensor) = named_tensors.get(name).copied() {
        if tensor.shape() != target.shape() {
            return Err(EvalError::CheckpointMismatch(format!(
                "tensor '{name}' shape mismatch: expected {:?}, got {:?}",
                target.shape().dims,
                tensor.shape().dims
            )));
        }
        *target = tensor.clone();
    }
    Ok(())
}
