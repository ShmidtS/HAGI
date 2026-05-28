//! Hierarchical Recurrent Model (HRM) -- M2 CPU reference.

pub mod attention;
pub mod hrm;
pub mod lm_head;
pub mod mask;
pub mod mlp;
pub mod norm;
pub mod recurrence;
pub mod rope;
pub mod transformer;

pub use hrm::forward_hrm;
pub use hrm::forward_hrm_with_control;
pub use hrm::DetachTrace;
pub use hrm::HiddenState;
pub use hrm::HState;
pub use hrm::HrmBackbone;
pub use hrm::HrmError;
pub use hrm::HrmOutput;
pub use hrm::HrmRuntimeControl;
pub use hrm::LState;
pub use hrm::Linear;
pub use lm_head::LmHead;
pub use mask::PrefixLmMask;
pub use recurrence::scheduled_bp_steps;
pub use recurrence::HRMState;
