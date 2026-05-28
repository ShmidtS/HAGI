#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TruthValue {
    frequency: f64,
    confidence: f64,
}

impl TruthValue {
    pub fn new(frequency: f64, confidence: f64) -> Self {
        Self {
            frequency: clamp01(frequency),
            confidence: clamp01(confidence),
        }
    }

    pub fn frequency(&self) -> f64 {
        self.frequency
    }
    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    pub fn induction(self, other: Self) -> Self {
        Self::new(
            self.frequency,
            self.confidence * other.confidence * other.frequency,
        )
    }

    pub fn deduction(self, other: Self) -> Self {
        Self::new(
            self.frequency * other.frequency,
            self.confidence * other.confidence,
        )
    }

    pub fn abduction(self, other: Self) -> Self {
        Self::new(
            other.frequency,
            self.confidence * other.confidence * self.frequency,
        )
    }

    pub fn intersection(self, other: Self) -> Self {
        Self::new(
            self.frequency * other.frequency,
            self.confidence + other.confidence - self.confidence * other.confidence,
        )
    }

    pub fn negation(self) -> Self {
        Self::new(1.0 - self.frequency, self.confidence)
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
    fn new_clamps_frequency_and_confidence_to_unit_interval() {
        let truth = TruthValue::new(1.25, -0.5);
        assert_close(truth.frequency(), 1.0);
        assert_close(truth.confidence(), 0.0);
    }

    #[test]
    fn deduction_multiplies_frequencies_and_confidences() {
        let truth = TruthValue::new(0.8, 0.9).deduction(TruthValue::new(0.5, 0.7));
        assert_close(truth.frequency(), 0.4);
        assert_close(truth.confidence(), 0.63);
    }

    #[test]
    fn induction_preserves_subject_frequency_with_evidence_product_confidence() {
        let truth = TruthValue::new(0.8, 0.9).induction(TruthValue::new(0.5, 0.7));
        assert_close(truth.frequency(), 0.8);
        assert_close(truth.confidence(), 0.315);
    }

    #[test]
    fn abduction_preserves_predicate_frequency_with_evidence_product_confidence() {
        let truth = TruthValue::new(0.8, 0.9).abduction(TruthValue::new(0.5, 0.7));
        assert_close(truth.frequency(), 0.5);
        assert_close(truth.confidence(), 0.504);
    }

    #[test]
    fn intersection_combines_positive_evidence() {
        let truth = TruthValue::new(0.8, 0.9).intersection(TruthValue::new(0.5, 0.7));
        assert_close(truth.frequency(), 0.4);
        assert_close(truth.confidence(), 0.97);
    }

    #[test]
    fn negation_inverts_frequency_and_preserves_confidence() {
        let truth = TruthValue::new(0.8, 0.9).negation();
        assert_close(truth.frequency(), 0.2);
        assert_close(truth.confidence(), 0.9);
    }
}
