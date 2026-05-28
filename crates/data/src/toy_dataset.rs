use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Deterministic toy text dataset for initial testing.
pub struct ToyDataset {
    pub vocab_size: usize,
    pub num_examples: usize,
    pub max_seq_len: usize,
    rng: ChaCha8Rng,
}

impl ToyDataset {
    pub fn new(vocab_size: usize, num_examples: usize, max_seq_len: usize, seed: u64) -> Self {
        Self {
            vocab_size,
            num_examples,
            max_seq_len,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    pub fn next_example(&mut self) -> Vec<u32> {
        let len = self.rng.gen_range(16..self.max_seq_len);
        (0..len)
            .map(|_| self.rng.gen_range(0..self.vocab_size as u32))
            .collect()
    }
}
