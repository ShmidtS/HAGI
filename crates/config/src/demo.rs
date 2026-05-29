use crate::{HdimConfig, HrmConfig, ModelConfig, MsaConfig};

pub fn demo_model_config() -> ModelConfig {
    ModelConfig {
        hrm: HrmConfig {
            total_layers: 2,
            h_layers: 1,
            l_layers: 1,
            hidden_size: 64,
            num_heads: 4,
            expansion: 4,
            h_cycles: 2,
            l_cycles: 3,
            vocab_size: 256,
            max_seq_len: 64,
            convergence_eps: 1e-5,
            bp_warmup_ratio: 0.2,
            bp_max_steps: 5,
            warmup_steps: 1000,
        },
        hdim: HdimConfig::new(64),
        msa: MsaConfig::new(1),
    }
}
