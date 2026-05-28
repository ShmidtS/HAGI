use serde::{Deserialize, Serialize};

use crate::enums::{FusionMode, InsertionPolicy};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HdimConfig {
    pub algebra_p: usize,
    pub algebra_q: usize,
    pub algebra_r: usize,
    pub structural_heads: usize,
    pub blade_count_per_head: usize,
    pub insertion_policy: InsertionPolicy,
    pub fusion_mode: FusionMode,
}

impl Default for HdimConfig {
    fn default() -> Self {
        Self {
            algebra_p: 3,
            algebra_q: 0,
            algebra_r: 0,
            structural_heads: 8,
            blade_count_per_head: 8,
            insertion_policy: InsertionPolicy::default(),
            fusion_mode: FusionMode::default(),
        }
    }
}

impl HdimConfig {
    pub fn new(hidden_size: usize) -> Self {
        let config = Self::default();
        let structural_dim = config.structural_heads * config.blade_count_per_head;
        assert!(
            hidden_size.is_multiple_of(structural_dim),
            "hidden_size ({}) must be divisible by structural_heads ({}) * blade_count ({}) = {}",
            hidden_size,
            config.structural_heads,
            config.blade_count_per_head,
            structural_dim
        );
        config
    }

    pub fn validate(&self) -> Result<(), String> {
        let dim = self.algebra_p + self.algebra_q + self.algebra_r;
        if dim > 16 {
            return Err(format!("algebra dimension {} exceeds 16", dim));
        }
        let expected_blades = 1usize << dim;
        if self.blade_count_per_head != expected_blades {
            return Err(format!(
                "blade_count_per_head ({}) must equal 2^(p+q+r) = {}",
                self.blade_count_per_head, expected_blades
            ));
        }
        Ok(())
    }

    pub fn validate_hidden_size(&self, hidden_size: usize) -> Result<(), String> {
        self.validate()?;
        let structural_dim = self.structural_heads * self.blade_count_per_head;
        if !hidden_size.is_multiple_of(structural_dim) {
            return Err(format!(
                "hidden_size ({}) must be divisible by structural_heads ({}) * blade_count ({}) = {}",
                hidden_size, self.structural_heads, self.blade_count_per_head, structural_dim
            ));
        }
        Ok(())
    }
}
