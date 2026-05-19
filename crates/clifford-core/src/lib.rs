//! Clifford algebra primitives: CPU reference implementation.

pub mod algebra;
pub mod multivector;
pub mod rotor;
pub mod invariants;
pub mod product_table;

pub use algebra::CliffordAlgebra;
pub use multivector::Multivector;
pub use rotor::Rotor;
pub use invariants::{Invariant, RotorSandwich};
