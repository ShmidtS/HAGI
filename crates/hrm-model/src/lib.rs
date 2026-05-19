//! Hierarchical Recurrent Model (HRM) — forward-only CPU reference.

pub mod transformer;
pub mod hrm;
pub mod lm_head;

pub use hrm::HrmBackbone;
pub use lm_head::LmHead;
