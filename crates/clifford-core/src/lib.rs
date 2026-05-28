//! Clifford algebra primitives: CPU reference implementation.

pub mod invariants;
pub mod multivector;
pub mod product_table;
pub mod rotor;

pub type Cl3 = core_types::algebra::Cl<3, 0, 0>;

pub use invariants::{Invariant, RotorSandwich};
pub use multivector::Multivector;
/// Lazily generated CL(3, 0, 0) product table for CPU reference Clifford products.
///
/// Contains `8 * 8` entries for CL3 blades. Initialization panics only if the fixed CL3 dimensions
/// fail checked generation; CUDA kernels use their own dispatch paths and fall back separately.
pub use product_table::get_product_table_cl3;
/// Precomputed CL(3, 0, 0) blade grades for the eight basis blades.
///
/// Indexed by blade bitmask in `[0, 8)`. This CPU lookup table never panics when indexed with a
/// valid CL3 blade and has no CUDA dependency or fallback behavior.
pub use product_table::GRADE_LOOKUP_CL3;
pub use product_table::{ProductEntry, ProductTable, ProductTableError};
pub use rotor::{rotor_sandwich_cl3, Rotor, UnitRotor};
