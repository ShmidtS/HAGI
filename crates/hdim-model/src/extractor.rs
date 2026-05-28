use crate::{HdimError, MultivectorBatch};
use clifford_core::{Invariant, Multivector, ProductTable, Rotor, RotorSandwich};
use core_types::{algebra::AlgebraSignature, shape::Shape};
use tensor_runtime::Tensor;

/// Extracts domain-invariant encoding via rotor sandwich.
pub struct InvariantExtractor {
    table: ProductTable,
}

impl InvariantExtractor {
    pub fn new(table: ProductTable) -> Self {
        Self { table }
    }

    pub fn extract_batch<A: AlgebraSignature>(
        &self,
        g: &MultivectorBatch<A>,
        source: &Rotor<A>,
    ) -> Result<MultivectorBatch<A>, HdimError> {
        let shape = g.coeffs.shape();
        if shape.rank() != 4 || shape.dims[2] != g.structural_heads || shape.dims[3] != A::BLADE_COUNT {
            return Err(HdimError::ShapeMismatch);
        }

        let batch_size = shape.dims[0];
        let tokens = shape.dims[1];
        let structural_dim = g.structural_heads * A::BLADE_COUNT;
        let mut out = vec![0.0f32; batch_size * tokens * structural_dim];

        for bt in 0..(batch_size * tokens) {
            for head in 0..g.structural_heads {
                let offset = bt * structural_dim + head * A::BLADE_COUNT;
                let mv = Multivector::<A>::from_coeffs(
                    g.coeffs.data()[offset..offset + A::BLADE_COUNT].to_vec(),
                );
                let invariant = self.extract_invariant(&mv, source);
                out[offset..offset + A::BLADE_COUNT].copy_from_slice(&invariant.mv.coeffs);
            }
        }

        Ok(MultivectorBatch::new(
            Tensor::from_vec(
                out,
                Shape::new(vec![batch_size, tokens, g.structural_heads, A::BLADE_COUNT]),
            ),
            g.structural_heads,
        ))
    }
}

impl<A: AlgebraSignature> RotorSandwich<A> for InvariantExtractor {
    fn extract_invariant(&self, g: &Multivector<A>, source: &Rotor<A>) -> Invariant<A> {
        let inverse = source.reverse(&self.table);
        let mv = source.extract_sandwich(g, &inverse, &self.table);
        Invariant { mv }
    }

    fn transfer_to_domain(&self, invariant: &Invariant<A>, target: &Rotor<A>) -> Multivector<A> {
        let inverse = target.reverse(&self.table);
        target.transfer_sandwich(&invariant.mv, &inverse, &self.table)
    }
}
