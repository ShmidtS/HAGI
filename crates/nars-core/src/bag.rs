use rand::Rng;

#[derive(Debug, Clone, PartialEq)]
pub struct Bag<T> {
    entries: Vec<(T, f64)>,
}

impl<T> Bag<T> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn put(&mut self, item: T, priority: f64) {
        self.entries.push((item, priority.max(0.0)));
        self.sort_by_priority();
    }

    pub fn take(&mut self) -> Option<T> {
        if self.entries.is_empty() {
            return None;
        }
        Some(self.entries.remove(0).0)
    }

    pub fn take_with_rng<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Option<T> {
        let total: f64 = self.entries.iter().map(|(_, priority)| *priority).sum();
        if total <= 0.0 {
            return self.take();
        }

        let mut threshold = rng.gen_range(0.0..total);
        let index = self
            .entries
            .iter()
            .position(|(_, priority)| {
                threshold -= *priority;
                threshold <= 0.0
            })
            .unwrap_or(self.entries.len() - 1);
        Some(self.entries.remove(index).0)
    }

    pub fn forget(&mut self, durability_decay: f64) {
        let durability_decay = durability_decay.clamp(0.0, 1.0);
        for (_, priority) in &mut self.entries {
            *priority *= durability_decay;
        }
        self.sort_by_priority();
    }

    pub fn peek_priority(&self, index: usize) -> Option<f64> {
        self.entries.get(index).map(|(_, priority)| *priority)
    }
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.entries.iter().map(|(item, _)| item)
    }

    pub fn retain<F>(&mut self, mut predicate: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.entries.retain(|(item, _)| predicate(item));
    }

    fn sort_by_priority(&mut self) {
        self.entries
            .sort_by(|(_, left), (_, right)| right.total_cmp(left));
    }
}

impl<T> Default for Bag<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn put_orders_entries_by_descending_priority() {
        let mut bag = Bag::new();
        bag.put("low", 0.1);
        bag.put("high", 0.9);
        bag.put("mid", 0.5);
        assert_eq!(bag.peek_priority(0), Some(0.9));
        assert_eq!(bag.peek_priority(1), Some(0.5));
        assert_eq!(bag.peek_priority(2), Some(0.1));
    }

    #[test]
    fn take_returns_none_when_bag_is_empty() {
        let mut bag: Bag<i32> = Bag::new();
        assert_eq!(bag.take(), None);
    }

    #[test]
    fn put_take_forget_cycle_decays_priorities_and_removes_selected_item() {
        let mut bag = Bag::new();
        bag.put("a", 0.8);
        bag.put("b", 0.4);
        bag.forget(0.5);
        assert_eq!(bag.peek_priority(0), Some(0.4));
        assert_eq!(bag.take(), Some("a"));
        assert_eq!(bag.len(), 1);
    }

    #[test]
    fn take_with_rng_selects_items_weighted_by_priority() {
        let mut rng = StdRng::seed_from_u64(11);
        let mut high_count = 0;
        let trials = 10_000;
        for _ in 0..trials {
            let mut bag = Bag::new();
            bag.put("low", 1.0);
            bag.put("high", 3.0);
            if bag.take_with_rng(&mut rng) == Some("high") {
                high_count += 1;
            }
        }
        let observed = high_count as f64 / trials as f64;
        assert!((observed - 0.75).abs() < 0.03, "observed={observed}");
    }
}
