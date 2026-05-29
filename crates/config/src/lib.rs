use std::fmt;

pub mod demo;
pub mod enums;
pub mod hdim;
pub mod hrm;
pub mod model;

pub use demo::demo_model_config;
pub use enums::{FusionMode, InsertionPolicy};
pub use hdim::HdimConfig;
pub use hrm::HrmConfig;
pub use model::{ModelConfig, MsaConfig};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError {
    message: String,
}

impl ConfigError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ConfigError {}

impl From<String> for ConfigError {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}
