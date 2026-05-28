//! HDIM structural layer — forward-only CPU reference.

use clifford_core::Multivector;
use core_types::{algebra::AlgebraSignature, shape::Shape};
use hrm_model::HiddenState;
use std::marker::PhantomData;
use tensor_runtime::Tensor;

pub mod extractor;
pub mod fusion;
pub mod projection;
pub mod transfer;

pub use extractor::InvariantExtractor;
pub use fusion::{fused_hrm_hdim_inject, StructuralFusion};
pub use projection::{project_hidden_to_multivector, HdimError, HiddenToMultivector};
pub use transfer::{transfer_domain, MemoryMode, TransferError, TransferState};

pub type DomainRotor<A = clifford_core::Cl3> = transfer::DomainRotor<A>;
pub type TransferRegistry<A = clifford_core::Cl3> = transfer::TransferRegistry<A>;
pub type HDIMError = HdimError;

pub fn hdim_forward(
    hidden: &Tensor<f32>,
    source_rotor: &DomainRotor,
    target_rotor: &DomainRotor,
    projector: &HiddenToMultivector,
    extractor: &InvariantExtractor,
    transfer: &TransferRegistry,
    fusion: &StructuralFusion,
) -> Result<Tensor<f32>, HDIMError> {
    let hidden_state = HiddenState::new(hidden.clone());
    let g = MultivectorBatch::new(
        projector.forward_result(&hidden_state)?,
        projector.structural_heads,
    );
    let u = extractor.extract_batch(&g, &source_rotor.rotor)?;
    let g_target = transfer.transfer_domain_from_invariant(&u, target_rotor)?;
    fusion.forward_result(hidden, &g_target.coeffs)
}

pub struct HdimForwardOutput {
    pub fused_hidden: Tensor<f32>,
    pub transfer_state: TransferState,
}

/// Batch of Clifford multivectors laid out as `[batch, tokens, structural_heads, blade_count]`.
///
/// `blade_count` must equal `A::BLADE_COUNT`. Constructors and accessors panic on rank, shape, or
/// index mismatches. This container stores CPU tensor data; CUDA callers must dispatch outside this
/// type and provide their own fallback behavior.
#[derive(Debug, Clone)]
pub struct MultivectorBatch<A: AlgebraSignature> {
    pub coeffs: Tensor<f32>,
    pub structural_heads: usize,
    pub algebra: PhantomData<A>,
}

impl<A: AlgebraSignature> MultivectorBatch<A> {
    pub fn new(coeffs: Tensor<f32>, structural_heads: usize) -> Self {
        let shape = coeffs.shape();
        assert_eq!(
            shape.rank(),
            4,
            "multivector batch must be rank 4 [B, T, structural_heads, blade_count]"
        );
        assert_eq!(
            shape.dims[2], structural_heads,
            "structural_heads mismatch: expected {}, got {}",
            structural_heads, shape.dims[2]
        );
        assert_eq!(
            shape.dims[3],
            A::BLADE_COUNT,
            "blade_count mismatch: expected {}, got {}",
            A::BLADE_COUNT,
            shape.dims[3]
        );
        Self {
            coeffs,
            structural_heads,
            algebra: PhantomData,
        }
    }

    pub fn shape(&self) -> &Shape {
        self.coeffs.shape()
    }

    pub fn multivector_at(&self, batch: usize, token: usize, head: usize) -> Multivector<A> {
        let shape = self.coeffs.shape();
        assert!(batch < shape.dims[0], "batch index out of bounds");
        assert!(token < shape.dims[1], "token index out of bounds");
        assert!(head < self.structural_heads, "head index out of bounds");
        let structural_dim = self.structural_heads * A::BLADE_COUNT;
        let offset =
            batch * shape.dims[1] * structural_dim + token * structural_dim + head * A::BLADE_COUNT;
        Multivector::<A>::from_coeffs(self.coeffs.data()[offset..offset + A::BLADE_COUNT].to_vec())
    }
}
