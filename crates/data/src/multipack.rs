use core_types::shape::Shape;
use tensor_runtime::Tensor;

use crate::{DataError, PackedBatch, PackedExample, PackedPartition, PackedSpan};

/// Greedy bin-packing scheduler for toy scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultipackScheduler {
    pub max_tokens: usize,
    pub pad_token_id: u32,
}

impl MultipackScheduler {
    pub fn new(max_tokens: usize, pad_token_id: u32) -> Result<Self, DataError> {
        if max_tokens == 0 {
            return Err(DataError::InvalidMaxTokens { max_tokens });
        }

        Ok(Self {
            max_tokens,
            pad_token_id,
        })
    }

    pub fn schedule(&self, examples: &[PackedExample]) -> Result<PackedBatch, DataError> {
        for example in examples {
            let len = example.tokens.len();
            if len > self.max_tokens {
                return Err(DataError::SequenceTooLong {
                    len,
                    max_tokens: self.max_tokens,
                });
            }
        }

        let mut row_used = Vec::<usize>::new();
        let mut spans = Vec::<PackedSpan>::new();
        for example in examples {
            let len = example.tokens.len();
            let batch_index = match row_used
                .iter()
                .position(|used| used + len <= self.max_tokens)
            {
                Some(index) => index,
                None => {
                    row_used.push(0);
                    row_used.len() - 1
                }
            };
            let start = row_used[batch_index];
            row_used[batch_index] += len;
            spans.push(PackedSpan {
                sequence_id: example.sequence_id,
                batch_index,
                start,
                len,
                prefix_len: example.prefix_len,
            });
        }

        let batch_size = row_used.len();
        let seq_len = self.max_tokens;
        let mut tokens = vec![self.pad_token_id; batch_size * seq_len];
        let mut targets = vec![self.pad_token_id; batch_size * seq_len];
        let mut prefix_mask = vec![1u8; batch_size * seq_len];

        for (example, span) in examples.iter().zip(spans.iter()) {
            let row_start = span.batch_index * seq_len + span.start;
            let row_end = row_start + span.len;
            tokens[row_start..row_end].copy_from_slice(&example.tokens);
            targets[row_start..row_end].copy_from_slice(&example.targets);
            prefix_mask[row_start..row_end].copy_from_slice(&example.prefix_mask);
        }

        let partition = PackedPartition {
            spans,
            batch_size,
            seq_len,
        };
        validate_no_overlap(&partition)?;

        let shape = Shape::new(vec![batch_size, seq_len]);
        Ok(PackedBatch {
            tokens: Tensor::from_vec(tokens, shape.clone()),
            targets: Tensor::from_vec(targets, shape.clone()),
            prefix_mask: Tensor::from_vec(prefix_mask, shape),
            partition,
        })
    }
}

fn validate_no_overlap(partition: &PackedPartition) -> Result<(), DataError> {
    for (i, span) in partition.spans.iter().enumerate() {
        for other in partition.spans.iter().skip(i + 1) {
            if span.batch_index != other.batch_index {
                continue;
            }
            let span_end = span.start + span.len;
            let other_end = other.start + other.len;
            if span.start < other_end && other.start < span_end {
                return Err(DataError::PartitionOverlap {
                    batch_index: other.batch_index,
                    start: other.start,
                    len: other.len,
                });
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_no_overlap;
    use crate::{DataError, PackedPartition, PackedSpan};

    #[test]
    fn rejects_overlapping_partition() {
        let partition = PackedPartition {
            spans: vec![
                PackedSpan {
                    sequence_id: 0,
                    batch_index: 0,
                    start: 0,
                    len: 3,
                    prefix_len: 1,
                },
                PackedSpan {
                    sequence_id: 1,
                    batch_index: 0,
                    start: 2,
                    len: 2,
                    prefix_len: 1,
                },
            ],
            batch_size: 1,
            seq_len: 4,
        };

        assert_eq!(
            validate_no_overlap(&partition),
            Err(DataError::PartitionOverlap {
                batch_index: 0,
                start: 2,
                len: 2,
            })
        );
    }
}
