use clifford_core::{Multivector, ProductTable, Rotor};
use core_types::algebra::Cl;
use core_types::ids::DomainId;
use hdim_model::{TransferError, TransferRegistry};

type Cl3 = Cl<3, 0, 0>;

fn cl3_table() -> ProductTable {
    ProductTable::generate(3, 0, 0)
}

fn make_rotor_from_coeffs(coeffs: Vec<f32>) -> Rotor<Cl3> {
    Rotor::unit(Multivector::<Cl3>::from_coeffs(coeffs))
}

/// Normalize a coefficient vector to unit L2 norm.
fn normalize_coeffs(coeffs: &mut Vec<f32>) {
    let norm: f32 = coeffs.iter().map(|&c| c * c).sum::<f32>().sqrt();
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        for c in coeffs.iter_mut() {
            *c *= inv;
        }
    }
}

#[test]
fn register_and_transfer_between_domains() {
    let table = cl3_table();
    let mut registry = TransferRegistry::<Cl3>::new();

    let domain_a = DomainId(0);
    let domain_b = DomainId(1);

    // Create two distinct unit rotors (normalized).
    let mut coeffs_a = vec![1.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0];
    normalize_coeffs(&mut coeffs_a);
    let rotor_a = make_rotor_from_coeffs(coeffs_a);

    let mut coeffs_b = vec![0.9, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0];
    normalize_coeffs(&mut coeffs_b);
    let rotor_b = make_rotor_from_coeffs(coeffs_b);

    registry.register_domain(domain_a, rotor_a);
    registry.register_domain(domain_b, rotor_b);

    // Create a test multivector.
    let g = Multivector::<Cl3>::from_coeffs(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let result = registry
        .transfer(domain_a, domain_b, &g, &table)
        .expect("transfer should succeed");
    assert_eq!(
        result.coeffs.len(),
        8,
        "result must have blade_count coefficients"
    );

    // Verify LUT was populated.
    assert!(
        registry.rotor_lut.contains_key(&(domain_a, domain_b)),
        "LUT should contain (A, B) pair after transfer"
    );
}

#[test]
fn same_domain_transfer_preserves_multivector() {
    let table = cl3_table();
    let mut registry = TransferRegistry::<Cl3>::new();

    let domain_a = DomainId(0);

    // Any rotor for domain A.
    let mut coeffs_a = vec![1.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0];
    normalize_coeffs(&mut coeffs_a);
    let rotor_a = make_rotor_from_coeffs(coeffs_a.clone());

    registry.register_domain(domain_a, rotor_a);

    let g = Multivector::<Cl3>::from_coeffs(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let result = registry
        .transfer(domain_a, domain_a, &g, &table)
        .expect("same-domain transfer should succeed");

    for i in 0..8 {
        assert!(
            (result.coeffs[i] - g.coeffs[i]).abs() < 1e-5,
            "same-domain transfer should preserve multivector at index {}: expected {}, got {}",
            i,
            g.coeffs[i],
            result.coeffs[i]
        );
    }
}

#[test]
fn lut_hit_on_repeated_transfer() {
    let table = cl3_table();
    let mut registry = TransferRegistry::<Cl3>::new();

    let domain_a = DomainId(0);
    let domain_b = DomainId(1);

    let mut coeffs_a = vec![1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    normalize_coeffs(&mut coeffs_a);
    registry.register_domain(domain_a, make_rotor_from_coeffs(coeffs_a));

    let mut coeffs_b = vec![0.9, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0];
    normalize_coeffs(&mut coeffs_b);
    registry.register_domain(domain_b, make_rotor_from_coeffs(coeffs_b));

    let g = Multivector::<Cl3>::from_coeffs(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    // First call: LUT miss, computes and inserts.
    let _r1 = registry
        .transfer(domain_a, domain_b, &g, &table)
        .expect("transfer should succeed");
    let lut_size_after_first = registry.rotor_lut.len();

    // Second call: LUT hit.
    let _r2 = registry
        .transfer(domain_a, domain_b, &g, &table)
        .expect("transfer should succeed");
    let lut_size_after_second = registry.rotor_lut.len();

    assert_eq!(
        lut_size_after_first, lut_size_after_second,
        "LUT should not grow on repeated transfer (cache hit)"
    );
}

#[test]
fn identity_rotor_transfer_preserves() {
    let table = cl3_table();
    let mut registry = TransferRegistry::<Cl3>::new();

    let domain_a = DomainId(0);
    let domain_b = DomainId(1);

    // Both domains with identity rotor.
    let identity = Rotor::unit(Multivector::<Cl3>::scalar_one());
    registry.register_domain(domain_a, Rotor::unit(Multivector::<Cl3>::scalar_one()));
    registry.register_domain(domain_b, identity);

    let g = Multivector::<Cl3>::from_coeffs(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let result = registry
        .transfer(domain_a, domain_b, &g, &table)
        .expect("transfer should succeed");

    for i in 0..8 {
        assert!(
            (result.coeffs[i] - g.coeffs[i]).abs() < 1e-5,
            "identity-to-identity transfer should preserve multivector at index {}: expected {}, got {}",
            i, g.coeffs[i], result.coeffs[i]
        );
    }
}

#[test]
fn transfer_output_has_correct_blade_count() {
    let table = cl3_table();
    let mut registry = TransferRegistry::<Cl3>::new();

    let d0 = DomainId(10);
    let d1 = DomainId(20);

    registry.register_domain(d0, Rotor::unit(Multivector::<Cl3>::scalar_one()));
    let mut coeffs = vec![1.0, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0];
    normalize_coeffs(&mut coeffs);
    registry.register_domain(d1, Rotor::unit(Multivector::<Cl3>::from_coeffs(coeffs)));

    let g = Multivector::<Cl3>::from_coeffs(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let result = registry
        .transfer(d0, d1, &g, &table)
        .expect("transfer should succeed");
    assert_eq!(
        result.coeffs.len(),
        <Cl3 as core_types::algebra::AlgebraSignature>::BLADE_COUNT,
        "transferred multivector must have BLADE_COUNT coefficients"
    );
}

#[test]
fn transfer_returns_error_for_missing_domain() {
    let table = cl3_table();
    let mut registry = TransferRegistry::<Cl3>::new();
    let source = DomainId(10);
    let target = DomainId(20);
    registry.register_domain(source, Rotor::unit(Multivector::<Cl3>::scalar_one()));

    let g = Multivector::<Cl3>::scalar_one();
    let result = registry.transfer(source, target, &g, &table);

    assert_eq!(result.err(), Some(TransferError::MissingDomain(target)));
}
