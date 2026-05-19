/// Compile-time Clifford algebra signature.
pub trait AlgebraSignature {
    const P: usize;
    const Q: usize;
    const R: usize = 0;
    const DIM: usize = Self::P + Self::Q + Self::R;
    const BLADE_COUNT: usize = 1usize << Self::DIM;
}

/// Concrete algebra signatures used by hot kernels.
pub struct Cl<const P: usize, const Q: usize, const R: usize = 0>;

impl<const P: usize, const Q: usize, const R: usize> AlgebraSignature for Cl<P, Q, R> {
    const P: usize = P;
    const Q: usize = Q;
    const R: usize = R;
    const DIM: usize = P + Q + R;
    const BLADE_COUNT: usize = 1usize << (P + Q + R);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blade_counts() {
        assert_eq!(<Cl<3, 0, 0> as AlgebraSignature>::BLADE_COUNT, 8);
        assert_eq!(<Cl<8, 0, 0> as AlgebraSignature>::BLADE_COUNT, 256);
        assert_eq!(<Cl<4, 0, 0> as AlgebraSignature>::BLADE_COUNT, 16);
    }
}
