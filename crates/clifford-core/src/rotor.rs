use core_types::algebra::AlgebraSignature;
use crate::multivector::Multivector;

/// Even-grade multivector representing a rotor.
pub struct Rotor<A: AlgebraSignature> {
    pub mv: Multivector<A>,
}

impl<A: AlgebraSignature> Rotor<A> {
    pub fn from_multivector(mv: Multivector<A>) -> Self {
        Self { mv }
    }
}
