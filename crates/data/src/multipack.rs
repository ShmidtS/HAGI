/// Greedy bin-packing scheduler for toy scale.
pub struct MultipackScheduler {
    pub max_tokens: usize,
}

impl MultipackScheduler {
    pub fn new(max_tokens: usize) -> Self {
        Self { max_tokens }
    }

    pub fn schedule(&self, _examples: &[Vec<u32>]) -> Vec<Vec<usize>> {
        // Placeholder: return one pack per example.
        _examples.iter().enumerate().map(|(i, _)| vec![i]).collect()
    }
}
