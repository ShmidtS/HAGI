use tensor_runtime::Tensor;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedSpan {
    pub sequence_id: usize,
    pub batch_index: usize,
    pub start: usize,
    pub len: usize,
    pub prefix_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedPartition {
    pub spans: Vec<PackedSpan>,
    pub batch_size: usize,
    pub seq_len: usize,
}

#[derive(Debug, Clone)]
pub struct PackedBatch {
    pub tokens: Tensor<u32>,
    pub targets: Tensor<u32>,
    pub prefix_mask: Tensor<u8>,
    pub partition: PackedPartition,
}

#[derive(Error, Debug, Clone, PartialEq)]
pub enum DataError {
    #[error("empty sequence")]
    EmptySequence,
    #[error("invalid prefix ratio: {prefix_ratio}")]
    InvalidPrefixRatio { prefix_ratio: f32 },
    #[error("invalid max tokens: {max_tokens}")]
    InvalidMaxTokens { max_tokens: usize },
    #[error("sequence length {len} exceeds max tokens {max_tokens}")]
    SequenceTooLong { len: usize, max_tokens: usize },
    #[error("partition overlap at batch {batch_index}, start {start}, len {len}")]
    PartitionOverlap {
        batch_index: usize,
        start: usize,
        len: usize,
    },
}
