use crate::DataError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedExample {
    pub sequence_id: usize,
    pub tokens: Vec<u32>,
    pub targets: Vec<u32>,
    pub prefix_mask: Vec<u8>,
    pub prefix_len: usize,
}

/// Splits a sequence into prefix (bidirectional) and response (causal) regions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrefixLmPacker {
    pub prefix_ratio: f32,
}

impl PrefixLmPacker {
    pub fn new(prefix_ratio: f32) -> Result<Self, DataError> {
        if prefix_ratio <= 0.0 || prefix_ratio >= 1.0 {
            return Err(DataError::InvalidPrefixRatio { prefix_ratio });
        }

        Ok(Self { prefix_ratio })
    }

    pub fn pack(&self, sequence_id: usize, tokens: &[u32]) -> Result<PackedExample, DataError> {
        let len = tokens.len();
        if len < 2 {
            return Err(DataError::EmptySequence);
        }

        let prefix_len = ((len as f32) * self.prefix_ratio).floor() as usize;
        let prefix_len = prefix_len.clamp(1, len - 1);
        let mut targets = Vec::with_capacity(len);
        for i in 0..(len - 1) {
            targets.push(tokens[i + 1]);
        }
        targets.push(tokens[len - 1]);

        let prefix_mask = (0..len)
            .map(|i| if i < prefix_len { 1 } else { 0 })
            .collect();

        Ok(PackedExample {
            sequence_id,
            tokens: tokens.to_vec(),
            targets,
            prefix_mask,
            prefix_len,
        })
    }
}
