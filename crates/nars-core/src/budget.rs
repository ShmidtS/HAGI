#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetValue {
    priority: f64,
    durability: f64,
    quality: f64,
}

impl BudgetValue {
    pub fn new(priority: f64, durability: f64, quality: f64) -> Self {
        Self {
            priority: clamp01(priority),
            durability: clamp01(durability),
            quality: clamp01(quality),
        }
    }

    pub fn priority(&self) -> f64 {
        self.priority
    }
    pub fn durability(&self) -> f64 {
        self.durability
    }
    pub fn quality(&self) -> f64 {
        self.quality
    }

    pub fn merge(self, other: Self) -> Self {
        Self::new(
            self.priority.max(other.priority),
            self.durability.max(other.durability),
            self.quality.max(other.quality),
        )
    }

    pub fn decay(self, factor: f64) -> Self {
        let factor = clamp01(factor);
        Self::new(
            self.priority * factor,
            self.durability * factor,
            self.quality,
        )
    }

    pub fn above_threshold(&self, threshold: f64) -> bool {
        let threshold = clamp01(threshold);
        self.priority >= threshold && self.quality >= threshold
    }

    pub fn is_valid(&self) -> bool {
        (0.0..=1.0).contains(&self.priority)
            && (0.0..=1.0).contains(&self.durability)
            && (0.0..=1.0).contains(&self.quality)
    }
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn new_clamps_all_components_to_unit_interval() {
        let budget = BudgetValue::new(1.2, -0.3, 0.5);
        assert_close(budget.priority(), 1.0);
        assert_close(budget.durability(), 0.0);
        assert_close(budget.quality(), 0.5);
    }

    #[test]
    fn merge_takes_componentwise_maximum() {
        let budget = BudgetValue::new(0.2, 0.8, 0.4).merge(BudgetValue::new(0.7, 0.3, 0.9));
        assert_close(budget.priority(), 0.7);
        assert_close(budget.durability(), 0.8);
        assert_close(budget.quality(), 0.9);
    }

    #[test]
    fn decay_scales_priority_and_durability_but_keeps_quality() {
        let budget = BudgetValue::new(0.8, 0.6, 0.4).decay(0.5);
        assert_close(budget.priority(), 0.4);
        assert_close(budget.durability(), 0.3);
        assert_close(budget.quality(), 0.4);
    }

    #[test]
    fn above_threshold_requires_priority_and_quality_to_meet_threshold() {
        assert!(BudgetValue::new(0.6, 0.1, 0.7).above_threshold(0.5));
        assert!(!BudgetValue::new(0.6, 0.1, 0.4).above_threshold(0.5));
    }

    #[test]
    fn is_valid_reports_all_components_in_unit_interval() {
        assert!(BudgetValue::new(2.0, -1.0, 0.5).is_valid());
    }

    #[test]
    fn decay_with_unit_factor_or_less_never_increases_priority_or_durability() {
        let budget = BudgetValue::new(0.4, 0.7, 0.2);
        let decayed = budget.decay(0.8);
        assert!(decayed.priority() <= budget.priority());
        assert!(decayed.durability() <= budget.durability());
    }

    #[test]
    fn merge_output_is_componentwise_at_least_each_input() {
        let left = BudgetValue::new(0.2, 0.8, 0.4);
        let right = BudgetValue::new(0.7, 0.3, 0.9);
        let merged = left.merge(right);
        assert!(merged.priority() >= left.priority() && merged.priority() >= right.priority());
        assert!(
            merged.durability() >= left.durability() && merged.durability() >= right.durability()
        );
        assert!(merged.quality() >= left.quality() && merged.quality() >= right.quality());
    }
}
