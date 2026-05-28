use core_types::{ids::DomainId, shape::Shape};
use tensor_runtime::Tensor;

use crate::slot::MemorySlot;

/// Registry holding all active memory slots.
#[derive(Debug, Clone)]
pub struct SlotRegistry {
    slots: Vec<MemorySlot>,
}

impl SlotRegistry {
    pub fn new() -> Self {
        Self { slots: Vec::new() }
    }

    pub fn register(&mut self, slot: MemorySlot) {
        self.slots.push(slot);
    }

    /// Remove slot by ID. Returns true if found and removed.
    pub fn deregister(&mut self, id: usize) -> bool {
        if let Some(idx) = self.slots.iter().position(|s| s.id == id) {
            self.slots.remove(idx);
            true
        } else {
            false
        }
    }

    pub fn get(&self, id: usize) -> Option<&MemorySlot> {
        self.slots.iter().find(|s| s.id == id)
    }

    pub fn slots_for_domain(&self, domain_id: DomainId) -> Vec<&MemorySlot> {
        self.slots
            .iter()
            .filter(|slot| slot.domain_id == domain_id)
            .collect()
    }

    pub fn slot_ids_for_domain(&self, domain_id: DomainId) -> Vec<usize> {
        self.slots
            .iter()
            .filter(|slot| slot.domain_id == domain_id)
            .map(|slot| slot.id)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Concatenates all slot keys into a single tensor of shape [num_slots, key_dim].
    pub fn all_keys(&self) -> Tensor<f32> {
        assert!(
            !self.slots.is_empty(),
            "cannot get all_keys from empty registry"
        );
        let num_slots = self.slots.len();
        let key_dim = self.slots[0].key.numel();
        let mut data = Vec::with_capacity(num_slots * key_dim);
        for slot in &self.slots {
            assert_eq!(
                slot.key.numel(),
                key_dim,
                "all slot keys must have the same dimension"
            );
            data.extend_from_slice(slot.key.data());
        }
        Tensor::from_vec(data, Shape::new(vec![num_slots, key_dim]))
    }

    /// Returns slot IDs in registration order.
    pub fn slot_ids(&self) -> Vec<usize> {
        self.slots.iter().map(|s| s.id).collect()
    }
}

impl Default for SlotRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_slot(id: usize, dim: usize) -> MemorySlot {
        let key = Tensor::from_vec(vec![id as f32; dim], Shape::new(vec![dim]));
        let val = Tensor::from_vec(vec![0.0; dim], Shape::new(vec![dim]));
        MemorySlot::new(id, key, val, 0, "test".into())
    }

    #[test]
    fn register_and_get() {
        let mut reg = SlotRegistry::new();
        reg.register(make_slot(0, 4));
        reg.register(make_slot(1, 4));
        assert_eq!(reg.len(), 2);
        assert!(reg.get(0).is_some());
        assert!(reg.get(1).is_some());
        assert!(reg.get(2).is_none());
    }

    #[test]
    fn deregister_removes_slot() {
        let mut reg = SlotRegistry::new();
        reg.register(make_slot(10, 4));
        reg.register(make_slot(20, 4));
        assert!(reg.deregister(10));
        assert_eq!(reg.len(), 1);
        assert!(reg.get(10).is_none());
        assert!(!reg.deregister(10));
    }

    #[test]
    fn all_keys_shape() {
        let mut reg = SlotRegistry::new();
        for i in 0..5 {
            reg.register(make_slot(i, 8));
        }
        let keys = reg.all_keys();
        assert_eq!(keys.shape().dims, vec![5, 8]);
    }

    #[test]
    fn filters_slots_by_domain_id() {
        let mut reg = SlotRegistry::new();
        let key = Tensor::from_vec(vec![1.0; 4], Shape::new(vec![4]));
        let val = Tensor::from_vec(vec![0.0; 4], Shape::new(vec![4]));
        reg.register(MemorySlot::with_domain_id(
            1,
            key.clone(),
            val.clone(),
            0,
            "a".into(),
            DomainId(10),
        ));
        reg.register(MemorySlot::with_domain_id(
            2,
            key,
            val,
            0,
            "b".into(),
            DomainId(20),
        ));

        assert_eq!(reg.slot_ids_for_domain(DomainId(10)), vec![1]);
        assert_eq!(reg.slots_for_domain(DomainId(20))[0].id, 2);
    }
}
