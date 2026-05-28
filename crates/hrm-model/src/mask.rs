use core_types::shape::Shape;

/// PrefixLM attention mask.
///
/// For each batch item with prefix_len p:
/// - Prefix positions (i < p) attend to all prefix positions (bidirectional).
/// - Response positions (i >= p) attend to all prefix + causal within response.
/// - Prefix positions cannot attend to response positions.
///
/// Stored as flat Vec<bool> with shape [B, T, T].
pub struct PrefixLmMask {
    pub shape: Shape,
    pub bits: Vec<bool>,
}

impl PrefixLmMask {
    /// Build mask for batch with per-item prefix lengths.
    /// `seq_len` is the total token count per batch item.
    pub fn build(batch_size: usize, seq_len: usize, prefix_lens: &[usize]) -> Self {
        assert_eq!(prefix_lens.len(), batch_size);
        let total = batch_size * seq_len * seq_len;
        let mut bits = vec![false; total];

        for (b, &p) in prefix_lens.iter().enumerate().take(batch_size) {
            assert!(p <= seq_len, "prefix_len cannot exceed seq_len");
            let base = b * seq_len * seq_len;
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let attend = if i < p { j < p } else { j < p || j <= i };
                    bits[base + i * seq_len + j] = attend;
                }
            }
        }

        Self {
            shape: Shape::new(vec![batch_size, seq_len, seq_len]),
            bits,
        }
    }

    /// Returns true if position `query` in batch `b` can attend to position `key`.
    pub fn can_attend(&self, b: usize, query: usize, key: usize) -> bool {
        let t = self.shape.dims[1];
        self.bits[b * t * t + query * t + key]
    }

    pub fn checksum(&self) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        for bit in &self.bits {
            hash ^= u64::from(*bit);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_attends_prefix_bidirectionally() {
        let mask = PrefixLmMask::build(1, 6, &[3]);
        // Prefix positions 0,1,2 can attend each other
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    mask.can_attend(0, i, j),
                    "prefix {} should attend prefix {}",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn prefix_cannot_attend_response() {
        let mask = PrefixLmMask::build(1, 6, &[3]);
        for i in 0..3 {
            for j in 3..6 {
                assert!(
                    !mask.can_attend(0, i, j),
                    "prefix {} should not attend response {}",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn response_attends_prefix_and_causal() {
        let mask = PrefixLmMask::build(1, 6, &[3]);
        // Response position 4 can attend prefix 0,1,2 and response 3,4
        for j in 0..3 {
            assert!(
                mask.can_attend(0, 4, j),
                "response 4 should attend prefix {}",
                j
            );
        }
        assert!(
            mask.can_attend(0, 4, 3),
            "response 4 should attend response 3"
        );
        assert!(mask.can_attend(0, 4, 4), "response 4 should attend self");
        assert!(
            !mask.can_attend(0, 4, 5),
            "response 4 should not attend future response 5"
        );
    }

    #[test]
    fn response_cannot_attend_future() {
        let mask = PrefixLmMask::build(1, 6, &[2]);
        // Response position 3 cannot attend response 4, 5
        assert!(!mask.can_attend(0, 3, 4));
        assert!(!mask.can_attend(0, 3, 5));
    }
}
