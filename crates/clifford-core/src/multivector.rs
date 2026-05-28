use crate::product_table::ProductTable;
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

    pub fn geometric_product(&self, rhs: &Self, table: &ProductTable) -> Self {
        let n = table.blade_count;
        assert_eq!(self.coeffs.len(), n);
        assert_eq!(rhs.coeffs.len(), n);
        let mut out = vec![0.0f32; n];
        for a in 0..n {
            if self.coeffs[a] == 0.0 {
                continue;
            }
            for b in 0..n {
                if rhs.coeffs[b] == 0.0 {
                    continue;
                }
                let entry = &table.entries[a * n + b];
                if entry.metric == 0.0 {
                    continue;
                }
                let idx = entry.result_blade as usize;
                out[idx] += entry.sign as f32 * entry.metric * self.coeffs[a] * rhs.coeffs[b];
            }
        }
        Self::from_coeffs(out)
    }

    pub fn reverse(&self, table: &ProductTable) -> Self {
        let mut coeffs = self.coeffs.clone();
        for (i, coeff) in coeffs.iter_mut().enumerate().take(table.blade_count) {
            let g = table.grade[i] as usize;
            if g >= 2 && !(g * (g - 1) / 2).is_multiple_of(2) {
                *coeff = -*coeff;
            }
        }
        Self::from_coeffs(coeffs)
    }

    pub fn scalar_one() -> Self {
        let mut coeffs = vec![0.0; A::BLADE_COUNT];
        coeffs[0] = 1.0;
        Self::from_coeffs(coeffs)
    }

    pub fn grade_involution(&self, table: &ProductTable) -> Self {
        let mut coeffs = self.coeffs.clone();
        for (i, coeff) in coeffs.iter_mut().enumerate().take(table.blade_count) {
            if !table.grade[i].is_multiple_of(2) {
                *coeff = -*coeff;
            }
        }
        Self::from_coeffs(coeffs)
    }
}
