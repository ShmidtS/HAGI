use clifford_core::{get_product_table_cl3, Cl3, Multivector, ProductTable, GRADE_LOOKUP_CL3};

#[test]
fn cl3_product_table_has_expected_dimensions() {
    let table = get_product_table_cl3();

    assert_eq!(table.dim, 3);
    assert_eq!(table.blade_count, 8);
    assert_eq!(table.entries.len(), 64);
    assert_eq!(table.grade, GRADE_LOOKUP_CL3);
}

#[test]
fn generated_cl2_table_has_square_entry_count() {
    let table = ProductTable::generate_checked(2, 0, 0).unwrap();

    assert_eq!(table.dim, 2);
    assert_eq!(table.blade_count, 4);
    assert_eq!(table.entries.len(), 16);
    assert_eq!(table.grade, vec![0, 1, 1, 2]);
}

#[test]
fn scalar_one_is_geometric_identity_for_cl3() {
    let table = get_product_table_cl3();
    let one = Multivector::<Cl3>::scalar_one();
    let mv = Multivector::<Cl3>::from_coeffs(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    assert_eq!(one.geometric_product(&mv, table).coeffs, mv.coeffs);
}
