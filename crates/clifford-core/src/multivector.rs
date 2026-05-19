use core_types::algebra::AlgebraSignature;

/// Dense multivector representation for a given algebra signature.
#[derive(Debug, Clone, PartialEq)]
pub struct Multivector<A: AlgebraSignature> {
    pub coeffs: Vec<f32>,
    _algebra: std::marker::PhantomData<A>,
}

impl<A: AlgebraSignature> Multivector<A> {
    pub fn zeros() -> Self {
        Self {
            coeffs: vec![0.0; A::BLADE_COUNT],
            _algebra: std::marker::PhantomData,
        }
    }

    pub fn from_coeffs(coeffs: Vec<f32>) -> Self {
        assert_eq!(coeffs.len(), A::BLADE_COUNT);
        Self {
            coeffs,
            _algebra: std::marker::PhantomData,
        }
    }
}
