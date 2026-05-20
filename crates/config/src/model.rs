use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    pub hrm: HrmConfig,
    pub hdim: HdimConfig,
    pub moe: MoeConfig,
    pub memory: MemoryConfig,
    pub losses: LossConfig,
    pub training: TrainingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HrmConfig {
    pub total_layers: usize,
    pub h_layers: usize,
    pub l_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub h_cycles: usize,
    pub l_cycles: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub bp_warmup_ratio: f32,
    pub bp_max_steps: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HdimConfig {
    pub algebra_p: usize,
    pub algebra_q: usize,
    pub algebra_r: usize,
    pub structural_heads: usize,
    pub blade_count_per_head: usize,
    pub insertion_policy: String,
    pub fusion_mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MoeConfig {
    pub enabled: bool,
    pub num_experts: usize,
    pub top_k: usize,
    pub capacity_factor: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryConfig {
    pub enabled: bool,
    pub bank_size: usize,
    pub update_enabled_inference: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LossConfig {
    pub lambda_ce: f32,
    pub lambda_reconstruction: f32,
    pub lambda_isomorphism: f32,
    pub lambda_contrastive: f32,
    pub lambda_routing: f32,
    pub lambda_z: f32,
    pub lambda_expert_ortho: f32,
    pub lambda_memory: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub learning_rate: f32,
    pub epochs: usize,
    pub grad_accum_steps: usize,
    pub seed: u64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hrm: HrmConfig {
                total_layers: 8,
                h_layers: 4,
                l_layers: 4,
                hidden_size: 256,
                num_heads: 4,
                h_cycles: 1,
                l_cycles: 2,
                vocab_size: 32000,
                max_seq_len: 2048,
                bp_warmup_ratio: 0.2,
                bp_max_steps: 5,
            },
            hdim: HdimConfig {
                algebra_p: 8,
                algebra_q: 0,
                algebra_r: 0,
                structural_heads: 4,
                blade_count_per_head: 256,
                insertion_policy: "h_only".to_string(),
                fusion_mode: "gated_residual".to_string(),
            },
            moe: MoeConfig {
                enabled: false,
                num_experts: 4,
                top_k: 2,
                capacity_factor: 1.25,
            },
            memory: MemoryConfig {
                enabled: false,
                bank_size: 256,
                update_enabled_inference: true,
            },
            losses: LossConfig {
                lambda_ce: 1.0,
                lambda_reconstruction: 0.0,
                lambda_isomorphism: 0.0,
                lambda_contrastive: 0.0,
                lambda_routing: 0.0,
                lambda_z: 0.0,
                lambda_expert_ortho: 0.0,
                lambda_memory: 0.0,
            },
            training: TrainingConfig {
                batch_size: 8,
                learning_rate: 1e-4,
                epochs: 4,
                grad_accum_steps: 1,
                seed: 42,
            },
        }
    }
}

impl ModelConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.hrm.h_layers + self.hrm.l_layers != self.hrm.total_layers {
            return Err(format!(
                "h_layers ({}) + l_layers ({}) must equal total_layers ({})",
                self.hrm.h_layers, self.hrm.l_layers, self.hrm.total_layers
            ));
        }
        if self.hrm.num_heads == 0 {
            return Err("num_heads must be positive".to_string());
        }
        if !self.hrm.hidden_size.is_multiple_of(self.hrm.num_heads) {
            return Err(format!(
                "hidden_size ({}) must be divisible by num_heads ({})",
                self.hrm.hidden_size, self.hrm.num_heads
            ));
        }
        if self.hdim.algebra_p + self.hdim.algebra_q + self.hdim.algebra_r > 16 {
            return Err("algebra dimension (p+q+r) must not exceed 16".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_valid() {
        let cfg = ModelConfig::default();
        cfg.validate().expect("default config must be valid");
    }

    #[test]
    fn invalid_layer_split() {
        let mut cfg = ModelConfig::default();
        cfg.hrm.l_layers = 5;
        assert!(cfg.validate().is_err());
    }
}
