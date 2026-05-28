use std::sync::OnceLock;

/// Entry describing the geometric product of two basis blades.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProductEntry {
    /// Index of the resulting basis blade.
    pub result_blade: u16,
    /// Multiplication sign: -1, 0, or 1.
    pub sign: i8,
    /// Metric coefficient from the algebra signature.
    pub metric: f32,
}

/// Errors returned when generating a Clifford product table with checked bounds.
#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum ProductTableError {
    /// Dimension or table-size arithmetic overflowed before allocation.
    #[error("product table dimension arithmetic overflowed")]
    Overflow,
    /// Algebra dimension exceeds the checked generator limit.
    #[error("product table dimension too large: {dim}")]
    DimensionTooLarge { dim: usize },
    /// Product table allocation request exceeds available memory.
    #[error("product table allocation too large")]
    AllocationTooLarge,
}

pub const GRADE_LOOKUP_CL3: [u8; 8] = [0, 1, 1, 2, 1, 2, 2, 3];

static PRODUCT_TABLE_CL3: OnceLock<ProductTable> = OnceLock::new();

pub fn get_product_table_cl3() -> &'static ProductTable {
    PRODUCT_TABLE_CL3.get_or_init(|| ProductTable::generate(3, 0, 0))
}

#[derive(Debug)]
pub struct ProductTable {
    pub dim: usize,
    pub blade_count: usize,
    pub entries: Vec<ProductEntry>,
    pub grade: Vec<u8>,
}

impl ProductTable {
    /// Generates a Clifford product table and panics if checked generation fails.
    ///
    /// Use [`ProductTable::generate_checked`] when caller-controlled dimensions should return an
    /// error instead of panicking.
    pub fn generate(p: usize, q: usize, r: usize) -> Self {
        Self::generate_checked(p, q, r).expect("invalid Clifford product table dimensions")
    }

    /// Generates a Clifford product table with bounded dimensions and checked allocation sizes.
    ///
    /// Returns [`ProductTableError::DimensionTooLarge`] when `p + q + r > 10`,
    /// [`ProductTableError::Overflow`] if dimension or table-size arithmetic overflows, or
    /// [`ProductTableError::AllocationTooLarge`] if vector allocation fails. For valid inputs, the
    /// table contains `(2^dim)^2` entries and runs entirely on the CPU.
    pub fn generate_checked(p: usize, q: usize, r: usize) -> Result<Self, ProductTableError> {
        let dim = p
            .checked_add(q)
            .and_then(|dim| dim.checked_add(r))
            .ok_or(ProductTableError::Overflow)?;
        if dim > 10 {
            return Err(ProductTableError::DimensionTooLarge { dim });
        }
        let blade_count = 1usize
            .checked_shl(dim as u32)
            .ok_or(ProductTableError::Overflow)?;
        let entry_count = blade_count
            .checked_mul(blade_count)
            .ok_or(ProductTableError::Overflow)?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(entry_count)
            .map_err(|_| ProductTableError::AllocationTooLarge)?;
        let mut grade = Vec::new();
        grade
            .try_reserve_exact(blade_count)
            .map_err(|_| ProductTableError::AllocationTooLarge)?;

        for i in 0..blade_count {
            grade.push(i.count_ones() as u8);
        }

        for a in 0..blade_count {
            for b in 0..blade_count {
                let result_blade = (a ^ b) as u16;
                let mut inversion_count = 0usize;
                for i in 0..dim {
                    if (a & (1 << i)) != 0 {
                        inversion_count += (b & ((1 << i) - 1)).count_ones() as usize;
                    }
                }
                let sign = if inversion_count.is_multiple_of(2) { 1i8 } else { -1i8 };

                let mut metric = 1.0f32;
                for i in 0..dim {
                    if (a & b & (1 << i)) != 0 {
                        if i < p {
                            continue;
                        } else if i < p + q {
                            metric *= -1.0;
                        } else {
                            metric = 0.0;
                            break;
                        }
                    }
                }

                entries.push(ProductEntry {
                    result_blade,
                    sign,
                    metric,
                });
            }
        }

        Ok(Self {
            dim,
            blade_count,
            entries,
            grade,
        })
    }
}
