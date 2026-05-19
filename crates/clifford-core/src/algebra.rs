use core_types::algebra::AlgebraSignature;

/// CPU reference Clifford algebra for a fixed signature.
pub struct CliffordAlgebra<A: AlgebraSignature> {
    _algebra: std::marker::PhantomData<A>,
}

impl<A: AlgebraSignature> CliffordAlgebra<A> {
    pub fn new() -> Self {
        Self {
            _algebra: std::marker::PhantomData,
        }
    }

    pub fn blade_count(&self) -> usize {
        A::BLADE_COUNT
    }

    pub fn dim(&self) -> usize {
        A::DIM
    }
}
