use clifford_core::{ProductTable, ProductTableError};

fn cl3_table() -> ProductTable {
    ProductTable::generate(3, 0, 0)
}

#[test]
fn grade_lookup_cl3() {
    let table = cl3_table();
    assert_eq!(table.grade, vec![0, 1, 1, 2, 1, 2, 2, 3]);
}

#[test]
fn blade_count_cl3() {
    let table = cl3_table();
    assert_eq!(table.dim, 3);
    assert_eq!(table.blade_count, 8);
    assert_eq!(table.entries.len(), 64);
}

#[test]
fn ei_times_ei_is_scalar() {
    let table = cl3_table();
    for idx in [1, 2] {
        let entry = &table.entries[idx * 8 + idx];
        assert_eq!(entry.result_blade, 0, "blade index {idx}");
        assert_eq!(entry.sign, 1, "blade index {idx}");
        assert_eq!(entry.metric, 1.0, "blade index {idx}");
    }
}

#[test]
fn e1_times_e2_anticommutes() {
    let table = cl3_table();
    let e1e2 = &table.entries[1 * 8 + 2];
    let e2e1 = &table.entries[2 * 8 + 1];
    assert_eq!(e1e2.result_blade, e2e1.result_blade);
    assert_eq!(e1e2.sign, -e2e1.sign);
}

#[test]
fn cl20_negative_signature() {
    let table = ProductTable::generate(0, 2, 0);
    // e1 * e1 with negative signature: blade 1
    let entry = &table.entries[1 * 4 + 1];
    assert_eq!(entry.result_blade, 0);
    assert_eq!(entry.metric, -1.0);
}

#[test]
fn degenerate_signature_zero_metric() {
    let table = ProductTable::generate(1, 0, 1);
    // e2 (degenerate, index 2 = 0b10) times itself
    let entry = &table.entries[2 * 4 + 2];
    assert_eq!(entry.metric, 0.0);
}

#[test]
fn generate_checked_allows_dimension_ten() {
    let table = ProductTable::generate_checked(10, 0, 0).unwrap();
    assert_eq!(table.dim, 10);
    assert_eq!(table.blade_count, 1024);
    assert_eq!(table.entries.len(), 1_048_576);
}

#[test]
fn generate_checked_rejects_dimension_eleven() {
    let err = ProductTable::generate_checked(11, 0, 0).unwrap_err();
    assert_eq!(err, ProductTableError::DimensionTooLarge { dim: 11 });
}

#[test]
fn generate_checked_rejects_dimension_overflow() {
    let err = ProductTable::generate_checked(usize::MAX, 1, 0).unwrap_err();
    assert_eq!(err, ProductTableError::Overflow);
}
