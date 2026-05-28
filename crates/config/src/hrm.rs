use crate::ConfigError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HrmConfig {
    pub total_layers: usize,
    pub h_layers: usize,
    pub l_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub expansion: usize,
    pub h_cycles: usize,
    pub l_cycles: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub convergence_eps: f32,
    pub bp_warmup_ratio: f32,
    pub bp_max_steps: usize,
    pub warmup_steps: usize,
}

impl Default for HrmConfig {
    fn default() -> Self {
        Self {
            total_layers: 24,
            h_layers: 8,
            l_layers: 16,
            hidden_size: 1280,
            num_heads: 10,
            expansion: 4,
            h_cycles: 2,
            l_cycles: 3,
            vocab_size: 50000,
            max_seq_len: 2048,
            convergence_eps: 1e-5,
            bp_warmup_ratio: 0.2,
            bp_max_steps: 5,
            warmup_steps: 1000,
        }
    }
}

impl HrmConfig {
    pub fn n_layers(&self) -> usize {
        self.total_layers
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.h_layers + self.l_layers != self.total_layers {
            return Err(ConfigError::new(format!(
                "h_layers ({}) + l_layers ({}) must equal total_layers ({})",
                self.h_layers, self.l_layers, self.total_layers
            )));
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(ConfigError::new(format!(
                "hidden_size ({}) must be divisible by num_heads ({})",
                self.hidden_size, self.num_heads
            )));
        }
        Ok(())
    }
}
