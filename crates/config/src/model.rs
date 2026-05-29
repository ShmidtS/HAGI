use crate::hdim::HdimConfig;
use crate::hrm::HrmConfig;
use crate::ConfigError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MsaConfig {
    #[serde(default = "default_msa_top_k")]
    pub top_k: usize,
}

impl Default for MsaConfig {
    fn default() -> Self {
        Self {
            top_k: default_msa_top_k(),
        }
    }
}

fn default_msa_top_k() -> usize {
    1
}

impl MsaConfig {
    pub fn try_new(top_k: usize) -> Result<Self, ConfigError> {
        if top_k == 0 {
            return Err(ConfigError::new("msa.top_k must be positive"));
        }
        Ok(Self { top_k })
    }

    // Panics if top_k == 0; prefer try_new.
    pub fn new(top_k: usize) -> Self {
        Self::try_new(top_k).expect("MsaConfig::new requires top_k > 0")
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.top_k == 0 {
            return Err(ConfigError::new("msa.top_k must be positive"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ModelConfig {
    pub hrm: HrmConfig,
    pub hdim: HdimConfig,
    #[serde(default)]
    pub msa: MsaConfig,
}

impl ModelConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.hrm.validate()?;
        self.hdim.validate_hidden_size(self.hrm.hidden_size)?;
        self.msa.validate()?;
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
