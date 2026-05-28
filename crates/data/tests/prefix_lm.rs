use data::{DataError, PrefixLmPacker};

#[test]
fn prefix_lm_prefix_ratio_floor_clamped_to_one() {
    let packer = PrefixLmPacker::new(0.1).unwrap();
    let example = packer.pack(0, &[10, 11, 12]).unwrap();

    assert_eq!(example.prefix_len, 1);
}

#[test]
fn prefix_lm_targets_shift_by_one() {
    let packer = PrefixLmPacker::new(0.5).unwrap();
    let example = packer.pack(3, &[10, 11, 12, 13]).unwrap();

    assert_eq!(example.targets, vec![11, 12, 13, 13]);
}

#[test]
fn prefix_lm_prefix_mask_marks_prefix_only() {
    let packer = PrefixLmPacker::new(0.5).unwrap();
    let example = packer.pack(0, &[10, 11, 12, 13]).unwrap();

    assert_eq!(example.prefix_len, 2);
    assert_eq!(example.prefix_mask, vec![1, 1, 0, 0]);
}

#[test]
fn prefix_lm_rejects_invalid_ratio() {
    assert_eq!(
        PrefixLmPacker::new(0.0),
        Err(DataError::InvalidPrefixRatio { prefix_ratio: 0.0 })
    );
    assert_eq!(
        PrefixLmPacker::new(1.0),
        Err(DataError::InvalidPrefixRatio { prefix_ratio: 1.0 })
    );
}

#[test]
fn prefix_lm_rejects_empty_sequence() {
    let packer = PrefixLmPacker::new(0.5).unwrap();

    assert_eq!(packer.pack(0, &[]), Err(DataError::EmptySequence));
    assert_eq!(packer.pack(0, &[10]), Err(DataError::EmptySequence));
}
