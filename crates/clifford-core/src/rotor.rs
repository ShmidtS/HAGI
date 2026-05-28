use crate::multivector::Multivector;
use crate::product_table::{get_product_table_cl3, ProductTable};
use core_types::algebra::AlgebraSignature;

/// Even-grade multivector representing a rotor.
pub struct Rotor<A: AlgebraSignature> {
    pub mv: Multivector<A>,
}

pub type UnitRotor<A> = Rotor<A>;

impl<A: AlgebraSignature> Rotor<A> {
    pub fn new(mv: Multivector<A>) -> Self {
        Self { mv }
    }

    pub fn unit(mv: Multivector<A>) -> Self {
        Self::new(mv).normalized()
    }

    pub fn normalized(&self) -> Self {
        let norm = self
            .mv
            .coeffs
            .iter()
            .map(|coeff| coeff * coeff)
            .sum::<f32>()
            .sqrt();
        if norm < 1e-8 {
            return Self::new(Multivector::<A>::scalar_one());
        }

        let coeffs = self.mv.coeffs.iter().map(|coeff| coeff / norm).collect();
        Self::new(Multivector::<A>::from_coeffs(coeffs))
    }

    pub fn is_unit(&self, epsilon: f32) -> bool {
        let norm_sq = self
            .mv
            .coeffs
            .iter()
            .map(|coeff| coeff * coeff)
            .sum::<f32>();
        (norm_sq - 1.0).abs() <= epsilon
    }

    pub fn reverse(&self, table: &ProductTable) -> Multivector<A> {
        self.mv.reverse(table)
    }

    pub fn extract_sandwich(
        &self,
        g: &Multivector<A>,
        inverse: &Multivector<A>,
        table: &ProductTable,
    ) -> Multivector<A> {
        let ig = inverse.geometric_product(g, table);
        ig.geometric_product(&self.mv, table)
    }

    pub fn transfer_sandwich(
        &self,
        u: &Multivector<A>,
        inverse: &Multivector<A>,
        table: &ProductTable,
    ) -> Multivector<A> {
        let vu = self.mv.geometric_product(u, table);
        vu.geometric_product(inverse, table)
    }
}

/// Applies a CL(3, 0, 0) unit-rotor sandwich product to one multivector.
///
/// `rotor` and `input` must each contain eight CL3 coefficients. Shape validation is performed by
/// the multivector constructors, and this CPU reference path does not call CUDA kernels or need a
/// fallback. Panics only if the static CL3 product table fails to initialize.
pub fn rotor_sandwich_cl3(
    rotor: &UnitRotor<crate::Cl3>,
    input: &Multivector<crate::Cl3>,
) -> Multivector<crate::Cl3> {
    debug_assert!(rotor.is_unit(1e-5));
    let table = get_product_table_cl3();
    let inverse = rotor.reverse(table);
    rotor.transfer_sandwich(input, &inverse, table)
}
