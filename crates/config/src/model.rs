use crate::hdim::HdimConfig;
use crate::hrm::HrmConfig;
use crate::ConfigError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ModelConfig {
    pub hrm: HrmConfig,
    pub hdim: HdimConfig,
}

impl ModelConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.hrm.validate()?;
        self.hdim.validate_hidden_size(self.hrm.hidden_size)?;
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

    #[test]
    fn invalid_blade_count() {
        let mut cfg = ModelConfig::default();
        cfg.hdim.blade_count_per_head = 16;
        cfg.hdim.algebra_p = 3;
        assert!(
            cfg.validate().is_err(),
            "blade_count_per_head=16 with p=3 (expected 8) must fail"
        );
    }
}
