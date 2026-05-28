/// Rotary Position Embedding (RoPE).
///
/// Precomputes sin/cos tables for positions 0..max_seq_len.
/// For each head dimension pair (d/2 pairs), applies rotation by angle = pos * base^(-2i/dim).
pub struct RopeTable {
    pub max_seq_len: usize,
    pub head_dim: usize,
    pub cos_table: Vec<f32>,
    pub sin_table: Vec<f32>,
}

impl RopeTable {
    pub fn new(max_seq_len: usize, head_dim: usize, base: f32) -> Self {
        assert_eq!(head_dim % 2, 0, "head_dim must be even for RoPE");
        let half = head_dim / 2;
        let mut cos_table = vec![0.0f32; max_seq_len * half];
        let mut sin_table = vec![0.0f32; max_seq_len * half];

        for pos in 0..max_seq_len {
            for i in 0..half {
                let angle = pos as f32 * base.powf(-(2.0 * i as f32) / head_dim as f32);
                let idx = pos * half + i;
                cos_table[idx] = angle.cos();
                sin_table[idx] = angle.sin();
            }
        }

        Self {
            max_seq_len,
            head_dim,
            cos_table,
            sin_table,
        }
    }

    /// Apply RoPE in-place to Q or K tensor with shape [B, num_heads, T, head_dim].
    /// `data` is modified in place.
    pub fn apply(&self, data: &mut [f32], batch: usize, num_heads: usize, seq_len: usize) {
        let d = self.head_dim;
        let half = d / 2;
        assert_eq!(data.len(), batch * num_heads * seq_len * d);
        assert!(
            seq_len <= self.max_seq_len,
            "seq_len exceeds precomputed table"
        );

        let bh_stride = seq_len * d;

        for bh in 0..(batch * num_heads) {
            let bh_offset = bh * bh_stride;
            for t in 0..seq_len {
                let t_offset = bh_offset + t * d;
                for i in 0..half {
                    let table_idx = t * half + i;
                    let cos_val = self.cos_table[table_idx];
                    let sin_val = self.sin_table[table_idx];
                    let x0 = data[t_offset + 2 * i];
                    let x1 = data[t_offset + 2 * i + 1];
                    data[t_offset + 2 * i] = x0 * cos_val - x1 * sin_val;
                    data[t_offset + 2 * i + 1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_identity_at_position_zero() {
        let rope = RopeTable::new(16, 4, 10000.0);
        // At position 0, angle = 0 for all pairs, so cos=1, sin=0 => identity
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
        let expected = data.clone();
        rope.apply(&mut data, 1, 1, 1);
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "position 0 should be identity");
        }
    }

    #[test]
    fn rope_rotates_known_position() {
        let head_dim = 4;
        let rope = RopeTable::new(16, head_dim, 10000.0);
        // Use seq_len=2 so position 1 is processed.
        // Position 1, pair 0: angle = 1.0 * 10000^0 = 1.0
        // Position 1, pair 1: angle = 1.0 * 10000^(-2/4) = 10000^(-0.5) = 0.01
        let mut data = vec![
            0.0f32, 0.0, 0.0, 0.0, // position 0 (unused)
            1.0, 0.0, 1.0, 0.0, // position 1
        ];
        rope.apply(&mut data, 1, 1, 2);
        let angle0 = 1.0f32;
        let angle1 = 10000.0f32.powf(-0.5);
        // Check position 1 data (offset 4)
        assert!((data[4] - angle0.cos()).abs() < 1e-5);
        assert!((data[5] - angle0.sin()).abs() < 1e-5);
        assert!((data[6] - angle1.cos()).abs() < 1e-5);
        assert!((data[7] - angle1.sin()).abs() < 1e-5);
    }
}
