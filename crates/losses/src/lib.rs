//! Composite loss components for HAGI training.

pub mod auxiliary;
pub mod cross_entropy;
pub mod isomorphic;
pub mod total;

pub use auxiliary::AuxiliaryLoss;
pub use cross_entropy::CrossEntropyLoss;
pub use isomorphic::IsomorphicLoss;
pub use total::{
    clifford_grade_norm, lambda_iso, magic_norm_clip, tensor_full, total_loss, AuxTargets,
    CompositeLoss, IsoPairBatch, LossBreakdown, LossError, LossWeights,
};
