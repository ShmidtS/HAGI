use clifford_core::{Multivector, ProductTable};
use core_types::algebra::Cl;

type Cl3 = Cl<3, 0, 0>;

fn cl3_table() -> ProductTable {
    ProductTable::generate(3, 0, 0)
}

fn basis_vector(idx: usize) -> Multivector<Cl3> {
    let mut coeffs = vec![0.0f32; 8];
    coeffs[idx] = 1.0;
    Multivector::from_coeffs(coeffs)
}

fn test_mv() -> Multivector<Cl3> {
    Multivector::from_coeffs(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
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
fn ei_times_ei_is_positive_scalar() {
    let table = cl3_table();
    for idx in [1, 2, 4] {
        let ei = basis_vector(idx);
        let result = ei.geometric_product(&ei, &table);
        assert!(
            (result.coeffs[0] - 1.0).abs() < 1e-6,
            "e{idx} * e{idx} should equal +1, got scalar {}",
            result.coeffs[0]
        );
        for j in 1..8 {
            assert!(
                result.coeffs[j].abs() < 1e-6,
                "e{idx} * e{idx} should have no grade>0 components"
            );
        }
    }
}

#[test]
fn ei_times_ej_anticommutes() {
    let table = cl3_table();
    let pairs = [(1usize, 2usize), (1, 4), (2, 4)];
    for (i, j) in pairs {
        let ei = basis_vector(i);
        let ej = basis_vector(j);
        let eij = ei.geometric_product(&ej, &table);
        let eji = ej.geometric_product(&ei, &table);
        for k in 0..8 {
            assert!(
                (eij.coeffs[k] + eji.coeffs[k]).abs() < 1e-6,
                "e{i}*e{j} + e{j}*e{i} should be zero at index {k}"
            );
        }
    }
}

#[test]
fn scalar_one_left_identity() {
    let table = cl3_table();
    let one = Multivector::<Cl3>::scalar_one();
    let x = test_mv();
    let result = one.geometric_product(&x, &table);
    assert_mv_eq(&result, &x, "scalar_one * x should equal x");
}

#[test]
fn scalar_one_right_identity() {
    let table = cl3_table();
    let one = Multivector::<Cl3>::scalar_one();
    let x = test_mv();
    let result = x.geometric_product(&one, &table);
    assert_mv_eq(&result, &x, "x * scalar_one should equal x");
}

#[test]
fn reverse_reverse_is_identity() {
    let table = cl3_table();
    let x = test_mv();
    let xrr = x.reverse(&table).reverse(&table);
    assert_mv_eq(&xrr, &x, "reverse(reverse(x)) should equal x");
}

#[test]
fn grade_involution_flips_odd_grades() {
    let table = cl3_table();
    let x = test_mv();
    let xi = x.grade_involution(&table);
    for i in 0..8 {
        let expected = if table.grade[i] % 2 != 0 {
            -x.coeffs[i]
        } else {
            x.coeffs[i]
        };
        assert!(
            (xi.coeffs[i] - expected).abs() < 1e-6,
            "grade_involution at index {i} (grade {}): expected {expected}, got {}",
            table.grade[i],
            xi.coeffs[i]
        );
    }
}
