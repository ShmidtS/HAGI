use core_types::ids::DomainId;
use tensor_runtime::Tensor;

/// A single memory slot with a key vector, value vector, and metadata.
#[derive(Debug, Clone)]
pub struct MemorySlot {
    pub id: usize,
    pub key: Tensor<f32>,
    pub value: Tensor<f32>,
    pub timestamp: usize,
    pub domain_tag: String,
    pub domain_id: DomainId,
}

impl MemorySlot {
    pub fn new(
        id: usize,
        key: Tensor<f32>,
        value: Tensor<f32>,
        timestamp: usize,
        domain_tag: String,
    ) -> Self {
        Self::with_domain_id(id, key, value, timestamp, domain_tag, DomainId(0))
    }

    pub fn with_domain_id(
        id: usize,
        key: Tensor<f32>,
        value: Tensor<f32>,
        timestamp: usize,
        domain_tag: String,
        domain_id: DomainId,
    ) -> Self {
        Self {
            id,
            key,
            value,
            timestamp,
            domain_tag,
            domain_id,
        }
    }

    /// Returns how many ticks have elapsed since this slot was created.
    pub fn age(&self, current_timestamp: usize) -> usize {
        current_timestamp.saturating_sub(self.timestamp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::shape::Shape;

    #[test]
    fn slot_age_saturates() {
        let key = Tensor::from_vec(vec![0.0; 4], Shape::new(vec![4]));
        let val = Tensor::from_vec(vec![0.0; 4], Shape::new(vec![4]));
        let slot = MemorySlot::new(0, key, val, 10, "test".into());
        assert_eq!(slot.age(15), 5);
        assert_eq!(slot.age(10), 0);
        assert_eq!(slot.age(5), 0);
    }
}
