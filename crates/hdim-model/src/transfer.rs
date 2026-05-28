use crate::MultivectorBatch;
use clifford_core::{Cl3, Multivector, ProductTable, Rotor};
use core_types::algebra::AlgebraSignature;
use core_types::ids::DomainId;
use core_types::shape::Shape;
use std::collections::HashMap;
use tensor_runtime::Tensor;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferError {
    MissingDomain(DomainId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryMode {
    #[default]
    Standard,
    Cache,
    Route,
}

pub struct TransferState {
    pub g_source: MultivectorBatch<Cl3>,
    pub u_inv: MultivectorBatch<Cl3>,
    pub u_mem: Option<MultivectorBatch<Cl3>>,
    pub u_route: Option<MultivectorBatch<Cl3>>,
    pub g_target: MultivectorBatch<Cl3>,
    pub memory_loss: f32,
    pub router_state: Option<Tensor<f32>>,
    pub memory_mode: MemoryMode,
}

/// A domain represented by its canonical rotor (frame of reference).
pub struct DomainRotor<A: AlgebraSignature> {
    pub domain_id: DomainId,
    pub rotor: Rotor<A>,
}

/// Registry of domain rotors and cached pair-rotor LUT for fast domain transfer.
pub struct TransferRegistry<A: AlgebraSignature> {
    pub domains: Vec<DomainRotor<A>>,
    pub rotor_lut: HashMap<(DomainId, DomainId), Rotor<A>>,
}

impl<A: AlgebraSignature> Default for TransferRegistry<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: AlgebraSignature> TransferRegistry<A> {
    pub fn new() -> Self {
        Self {
            domains: Vec::new(),
            rotor_lut: HashMap::new(),
        }
    }

    pub fn transfer_domain_from_invariant(
        &self,
        invariant: &MultivectorBatch<A>,
        target: &DomainRotor<A>,
    ) -> Result<MultivectorBatch<A>, crate::HdimError> {
        let shape = invariant.coeffs.shape();
        if shape.rank() != 4
            || shape.dims[2] != invariant.structural_heads
            || shape.dims[3] != A::BLADE_COUNT
        {
            return Err(crate::HdimError::ShapeMismatch);
        }

        let table = ProductTable::generate(A::P, A::Q, A::R);
        let inverse = target.rotor.reverse(&table);
        let batch_size = shape.dims[0];
        let tokens = shape.dims[1];
        let structural_heads = invariant.structural_heads;
        let blade_count = A::BLADE_COUNT;
        let structural_dim = structural_heads * blade_count;
        let mut out = vec![0.0f32; batch_size * tokens * structural_dim];

        for bt in 0..(batch_size * tokens) {
            for head in 0..structural_heads {
                let offset = bt * structural_dim + head * blade_count;
                let mv = Multivector::<A>::from_coeffs(
                    invariant.coeffs.data()[offset..offset + blade_count].to_vec(),
                );
                let transferred = target.rotor.transfer_sandwich(&mv, &inverse, &table);
                out[offset..offset + blade_count].copy_from_slice(&transferred.coeffs);
            }
        }

        Ok(MultivectorBatch::new(
            Tensor::from_vec(
                out,
                Shape::new(vec![batch_size, tokens, structural_heads, blade_count]),
            ),
            structural_heads,
        ))
    }

    /// Register a domain with its canonical rotor.
    pub fn register_domain(&mut self, domain_id: DomainId, rotor: Rotor<A>) {
        // Replace if already registered, otherwise push.
        if let Some(pos) = self.domains.iter().position(|d| d.domain_id == domain_id) {
            self.domains[pos] = DomainRotor { domain_id, rotor };
        } else {
            self.domains.push(DomainRotor { domain_id, rotor });
        }
        // Invalidate any cached LUT entries involving this domain.
        self.rotor_lut
            .retain(|&(s, t), _| s != domain_id && t != domain_id);
    }

    fn find_rotor(&self, domain_id: DomainId) -> Option<&Rotor<A>> {
        self.domains
            .iter()
            .find(|d| d.domain_id == domain_id)
            .map(|d| &d.rotor)
    }

    /// Transfer multivector `g` from `source` domain to `target` domain via rotor sandwich.
    ///
    /// R_pair = normalize(R_target * inverse(R_source))
    /// G_target = R_pair * G * inverse(R_pair)
    ///
    /// For unit rotors, inverse = reverse.
    pub fn transfer(
        &mut self,
        source: DomainId,
        target: DomainId,
        g: &Multivector<A>,
        table: &ProductTable,
    ) -> Result<Multivector<A>, TransferError> {
        // Same-domain fast path: identity transfer.
        if source == target {
            return Ok(Multivector::<A>::from_coeffs(g.coeffs.clone()));
        }

        let key = (source, target);
        if !self.rotor_lut.contains_key(&key) {
            let r_source = self
                .find_rotor(source)
                .ok_or(TransferError::MissingDomain(source))?;
            let r_target = self
                .find_rotor(target)
                .ok_or(TransferError::MissingDomain(target))?;
            // R_source inverse = reverse (for unit rotors)
            let r_source_inv = r_source.reverse(table);
            // R_pair_unnorm = R_target * R_source^{-1}
            let r_pair_unnorm = r_target.mv.geometric_product(&r_source_inv, table);
            let r_pair_mv = normalize_multivector(&r_pair_unnorm);
            self.rotor_lut.insert(key, Rotor::unit(r_pair_mv));
        }

        let r_pair = self.rotor_lut.get(&key).unwrap();
        let r_pair_inv = r_pair.reverse(table);
        // Sandwich: R_pair * G * R_pair^{-1}
        Ok(r_pair.transfer_sandwich(g, &r_pair_inv, table))
    }
}

/// Transfers every multivector in a batch between registered domains.
///
/// `batch.coeffs` must be shaped `[batch, tokens, structural_heads, A::BLADE_COUNT]`. Returns
/// [`TransferError`] when source or target domains are missing; shape/index mismatches panic through
/// [`MultivectorBatch`] invariants. This CPU reference path uses
/// the supplied product table and has no CUDA fallback.
pub fn transfer_domain<A: AlgebraSignature>(
    registry: &mut TransferRegistry<A>,
    source: DomainId,
    target: DomainId,
    batch: &MultivectorBatch<A>,
    table: &ProductTable,
) -> Result<MultivectorBatch<A>, TransferError> {
    let shape = batch.coeffs.shape();
    let batch_size = shape.dims[0];
    let tokens = shape.dims[1];
    let structural_heads = batch.structural_heads;
    let blade_count = A::BLADE_COUNT;
    let structural_dim = structural_heads * blade_count;
    let mut out = vec![0.0f32; batch_size * tokens * structural_dim];

    for bt in 0..(batch_size * tokens) {
        for head in 0..structural_heads {
            let offset = bt * structural_dim + head * blade_count;
            let mv = Multivector::<A>::from_coeffs(
                batch.coeffs.data()[offset..offset + blade_count].to_vec(),
            );
            let transferred = registry.transfer(source, target, &mv, table)?;
            out[offset..offset + blade_count].copy_from_slice(&transferred.coeffs);
        }
    }

    Ok(MultivectorBatch::new(
        Tensor::from_vec(
            out,
            Shape::new(vec![batch_size, tokens, structural_heads, blade_count]),
        ),
        structural_heads,
    ))
}

/// Normalize a multivector by dividing all coefficients by its norm (L2 of coefficients).
fn normalize_multivector<A: AlgebraSignature>(mv: &Multivector<A>) -> Multivector<A> {
    let norm_sq: f32 = mv.coeffs.iter().map(|&c| c * c).sum();
    let norm = norm_sq.sqrt();
    if norm < 1e-12 {
        // Degenerate: return scalar-one rotor as fallback
        return Multivector::<A>::scalar_one();
    }
    let inv_norm = 1.0 / norm;
    let coeffs: Vec<f32> = mv.coeffs.iter().map(|&c| c * inv_norm).collect();
    Multivector::<A>::from_coeffs(coeffs)
}
