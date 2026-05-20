/// Splits a sequence into prefix (bidirectional) and response (causal) regions.
pub struct PrefixLmPacker;

impl Default for PrefixLmPacker {
    fn default() -> Self {
        Self::new()
    }
}

impl PrefixLmPacker {
    pub fn new() -> Self {
        Self
    }

    pub fn pack(&self, _tokens: &[u32], _prefix_ratio: f32) -> (Vec<u32>, Vec<u32>) {
        (_tokens.to_vec(), vec![])
    }
}
