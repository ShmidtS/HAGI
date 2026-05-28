pub mod reasoner;
pub mod registry_ext;

pub use reasoner::{NarsHdimConfig, NarsHdimReasoner, TransferRecommendation};
pub use registry_ext::{transfer_domain_reasoned, transfer_domain_reasoned_or_fallback};

#[cfg(test)]
mod tests {
    use clifford_core::{Multivector, ProductTable, Rotor};
    use core_types::{algebra::Cl, ids::DomainId, shape::Shape};
    use hdim_model::{MultivectorBatch, TransferRegistry};
    use nars_core::TruthValue;
    use tensor_runtime::Tensor;

    use super::*;

    type Cl3 = Cl<3, 0, 0>;

    fn cl3_table() -> ProductTable {
        ProductTable::generate(3, 0, 0)
    }

    fn batch() -> MultivectorBatch<Cl3> {
        MultivectorBatch::new(
            Tensor::from_vec(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                Shape::new(vec![1, 1, 1, 8]),
            ),
            1,
        )
    }

    fn registry() -> TransferRegistry<Cl3> {
        let mut registry = TransferRegistry::<Cl3>::new();
        registry.register_domain(DomainId(0), Rotor::unit(Multivector::<Cl3>::scalar_one()));
        registry.register_domain(DomainId(1), Rotor::unit(Multivector::<Cl3>::scalar_one()));
        registry
    }

    #[test]
    fn fallback_to_standard_transfer_when_no_recommendation() {
        let table = cl3_table();
        let mut standard_registry = registry();
        let mut reasoned_registry = registry();
        let mut reasoner = NarsHdimReasoner::default();
        let batch = batch();

        let standard = hdim_model::transfer_domain(
            &mut standard_registry,
            DomainId(0),
            DomainId(1),
            &batch,
            &table,
        )
        .expect("standard transfer should succeed");
        let reasoned = transfer_domain_reasoned(
            &mut reasoned_registry,
            &mut reasoner,
            DomainId(0),
            DomainId(1),
            &batch,
            &table,
        )
        .expect("reasoned transfer should fall back to standard transfer");

        assert_eq!(standard.coeffs.data(), reasoned.coeffs.data());
    }

    #[test]
    fn transfer_belief_updates_with_successful_transfer() {
        let source = DomainId(0);
        let target = DomainId(1);
        let mut reasoner = NarsHdimReasoner::default();

        reasoner.observe_transfer_feedback(source, target, true);

        let truth = reasoner
            .transfer_beliefs
            .get(&(source, target))
            .expect("successful feedback should create transfer belief");
        assert_eq!(truth.frequency(), 1.0);
        assert_eq!(truth.confidence(), 0.9);
    }

    #[test]
    fn reasoner_selects_highest_confidence_pair() {
        let source_a = DomainId(0);
        let source_b = DomainId(1);
        let target_a = DomainId(2);
        let target_b = DomainId(3);
        let mut reasoner = NarsHdimReasoner::default();
        reasoner
            .transfer_beliefs
            .insert((source_a, target_a), TruthValue::new(0.8, 0.5));
        reasoner
            .transfer_beliefs
            .insert((source_b, target_b), TruthValue::new(0.9, 0.9));

        let recommendation = reasoner
            .recommend_transfer(&[source_a, source_b], &[target_a, target_b], "test")
            .expect("reasoner should select a believed transfer pair");

        assert_eq!(recommendation.source, source_b);
        assert_eq!(recommendation.target, target_b);
        assert!((recommendation.confidence - 0.81).abs() < 1e-6);
    }
}
