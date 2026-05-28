use clifford_core::{Multivector, ProductTable, Rotor, RotorSandwich};
use core_types::algebra::Cl;
use hdim_model::InvariantExtractor;

type Cl3 = Cl<3, 0, 0>;

fn cl3_extractor() -> InvariantExtractor {
    InvariantExtractor::new(ProductTable::generate(3, 0, 0))
}

fn assert_mv_eq(actual: &Multivector<Cl3>, expected: &Multivector<Cl3>, msg: &str) {
    for i in 0..8 {
        assert!(
            (actual.coeffs[i] - expected.coeffs[i]).abs() < 1e-6,
            "{msg} at index {i}: expected {}, got {}",
            expected.coeffs[i],
            actual.coeffs[i]
        );
    }
}

#[test]
fn identity_rotor_roundtrip() {
    let extractor = cl3_extractor();
    let rotor = Rotor::unit(Multivector::<Cl3>::scalar_one());

    let g = Multivector::<Cl3>::from_coeffs(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let invariant = extractor.extract_invariant(&g, &rotor);
    let result = extractor.transfer_to_domain(&invariant, &rotor);
    assert_mv_eq(
        &result,
        &g,
        "roundtrip with identity rotor should preserve multivector",
    );
}

#[test]
fn identity_rotor_extract_preserves() {
    let extractor = cl3_extractor();
    let rotor = Rotor::unit(Multivector::<Cl3>::scalar_one());

    let g = Multivector::<Cl3>::from_coeffs(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let invariant = extractor.extract_invariant(&g, &rotor);
    assert_mv_eq(
        &invariant.mv,
        &g,
        "extract with identity rotor should preserve multivector",
    );
}
