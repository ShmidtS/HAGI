use core_types::algebra::AlgebraSignature;
use crate::multivector::Multivector;
use crate::rotor::Rotor;

/// Extracted domain-invariant multivector.
pub struct Invariant<A: AlgebraSignature> {
    pub mv: Multivector<A>,
}

/// Rotor sandwich operations: invariant extraction and domain transfer.
pub trait RotorSandwich<A: AlgebraSignature> {
    fn extract_invariant(
        &self,
        g: &Multivector<A>,
        source: &Rotor<A>,
    ) -> Invariant<A>;

    fn transfer_to_domain(
        &self,
        invariant: &Invariant<A>,
        target: &Rotor<A>,
    ) -> Multivector<A>;
}
