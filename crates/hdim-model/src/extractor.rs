use core_types::algebra::AlgebraSignature;
use clifford_core::{Multivector, Rotor, Invariant, RotorSandwich};

/// Extracts domain-invariant encoding via rotor sandwich.
pub struct InvariantExtractor;

impl Default for InvariantExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl InvariantExtractor {
    pub fn new() -> Self {
        Self
    }
}

impl<A: AlgebraSignature> RotorSandwich<A> for InvariantExtractor {
    fn extract_invariant(
        &self,
        _g: &Multivector<A>,
        _source: &Rotor<A>,
    ) -> Invariant<A> {
        // Returns zero invariant.
        Invariant { mv: Multivector::zeros() }
    }

    fn transfer_to_domain(
        &self,
        _invariant: &Invariant<A>,
        _target: &Rotor<A>,
    ) -> Multivector<A> {
        // Returns zero target-domain multivector.
        Multivector::zeros()
    }
}
