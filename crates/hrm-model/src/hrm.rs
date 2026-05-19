use config::model::HrmConfig;
use tensor_runtime::Tensor;
use crate::transformer::TransformerStack;

/// HRM recurrence scheduler and backbone.
pub struct HrmBackbone {
    pub l_stack: TransformerStack,
    pub h_stack: TransformerStack,
    pub h_cycles: usize,
    pub l_cycles: usize,
}

impl HrmBackbone {
    pub fn from_config(config: &HrmConfig) -> Self {
        let l_blocks: Vec<_> = (0..config.l_layers)
            .map(|_| crate::transformer::TransformerBlock::new(config.hidden_size, config.num_heads))
            .collect();
        let h_blocks: Vec<_> = (0..config.h_layers)
            .map(|_| crate::transformer::TransformerBlock::new(config.hidden_size, config.num_heads))
            .collect();
        Self {
            l_stack: TransformerStack::new(l_blocks),
            h_stack: TransformerStack::new(h_blocks),
            h_cycles: config.h_cycles,
            l_cycles: config.l_cycles,
        }
    }

    /// Forward recurrence: nested H/L cycles.
    pub fn forward(&self, mut h_state: Tensor<f32>, mut l_state: Tensor<f32>) -> Tensor<f32> {
        for _h in 0..self.h_cycles {
            for _l in 0..self.l_cycles {
                // L-module: fuse h_state into l_state via addition (placeholder)
                l_state = self.l_stack.forward(l_state);
            }
            // H-module: fuse l_state into h_state via addition (placeholder)
            h_state = self.h_stack.forward(h_state);
        }
        h_state
    }
}
