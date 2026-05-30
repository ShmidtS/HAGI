from __future__ import annotations


def grade_sign(grade: int) -> int:
    grade = int(grade)
    if grade < 0:
        raise ValueError("grade must be non-negative")
    return -1 if (grade * (grade - 1) // 2) % 2 else 1


def clifford_product_sign(a_grade: int, b_grade: int) -> int:
    a_grade = int(a_grade)
    b_grade = int(b_grade)
    if a_grade < 0 or b_grade < 0:
        raise ValueError("grades must be non-negative")
    return -1 if (a_grade * b_grade) % 2 else 1
