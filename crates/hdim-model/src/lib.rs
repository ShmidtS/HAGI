//! HDIM structural layer — forward-only CPU reference.

pub mod projection;
pub mod extractor;
pub mod fusion;

pub use projection::HiddenToMultivector;
pub use extractor::InvariantExtractor;
pub use fusion::StructuralFusion;
