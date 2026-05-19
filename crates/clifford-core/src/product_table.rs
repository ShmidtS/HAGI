/// Term in a geometric product table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProductTerm {
    pub lhs_blade: u16,
    pub rhs_blade: u16,
    pub out_blade: u16,
    pub sign: i8,
    pub metric_scale: f32,
}

/// Static geometric product table for a fixed algebra signature.
/// Generated at compile time for hot kernels.
pub struct ProductTable {
    pub terms: &'static [ProductTerm],
}
