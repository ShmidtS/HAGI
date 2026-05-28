use clifford_core::{
    get_product_table_cl3, rotor_sandwich_cl3, Cl3, Multivector, ProductTable, Rotor, UnitRotor,
    GRADE_LOOKUP_CL3,
};

fn cl3_table() -> ProductTable {
    ProductTable::generate(3, 0, 0)
}

fn identity_rotor() -> Rotor<Cl3> {
    Rotor::unit(Multivector::<Cl3>::scalar_one())
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
fn unit_rotor_sandwich_identity() {
    let table = cl3_table();
    let rotor = identity_rotor();
    let inverse = rotor.reverse(&table);

    let x = Multivector::<Cl3>::from_coeffs(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let result = rotor.extract_sandwich(&x, &inverse, &table);
    assert_mv_eq(&result, &x, "identity rotor sandwich should preserve x");
}

#[test]
fn transfer_sandwich_identity() {
    let table = cl3_table();
    let rotor = identity_rotor();
    let inverse = rotor.reverse(&table);

    let u = Multivector::<Cl3>::from_coeffs(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let result = rotor.transfer_sandwich(&u, &inverse, &table);
    assert_mv_eq(&result, &u, "identity rotor transfer should preserve u");
}

#[test]
fn reverse_of_scalar_one_is_scalar_one() {
    let table = cl3_table();
    let rev = identity_rotor().reverse(&table);
    assert!((rev.coeffs[0] - 1.0).abs() < 1e-6);
    for i in 1..8 {
        assert!(rev.coeffs[i].abs() < 1e-6);
    }
}

#[test]
fn rotor_sandwich_cl3_matches_existing_transfer_sandwich() {
    let table = cl3_table();
    let rotor: UnitRotor<Cl3> = identity_rotor();
    let inverse = rotor.reverse(&table);
    let input = Multivector::<Cl3>::from_coeffs(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let expected = rotor.transfer_sandwich(&input, &inverse, &table);
    let result = rotor_sandwich_cl3(&rotor, &input);

    assert_mv_eq(
        &result,
        &expected,
        "rotor_sandwich_cl3 should use CL3 transfer sandwich",
    );
}

#[test]
fn cl3_static_constants_match_generated_table() {
    let table = cl3_table();

    assert_eq!(GRADE_LOOKUP_CL3, [0, 1, 1, 2, 1, 2, 2, 3]);
    let static_table = get_product_table_cl3();
    assert_eq!(static_table.dim, table.dim);
    assert_eq!(static_table.blade_count, table.blade_count);
    assert_eq!(static_table.grade, table.grade);
    assert_eq!(static_table.entries, table.entries);
}

#[test]
fn norm_preservation_golden() {
    let table = cl3_table();
    let theta = 0.5f32;
    let rotor = Rotor::unit(Multivector::<Cl3>::from_coeffs(vec![
        (theta * 0.5).cos(),
        0.0,
        0.0,
        (theta * 0.5).sin(),
        0.0,
        0.0,
        0.0,
        0.0,
    ]));
    let inverse = rotor.reverse(&table);
    let input = Multivector::<Cl3>::from_coeffs(vec![0.25, -1.0, 2.0, 0.5, 3.0, -0.75, 1.25, 0.0]);

    let output = rotor.extract_sandwich(&input, &inverse, &table);
    let input_norm = input.coeffs.iter().map(|v| v * v).sum::<f32>().sqrt();
    let output_norm = output.coeffs.iter().map(|v| v * v).sum::<f32>().sqrt();

    assert!((output_norm - input_norm).abs() < 1e-5);
}
