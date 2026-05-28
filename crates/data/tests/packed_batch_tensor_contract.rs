use core_types::dtype::DType;
use data::{MultipackScheduler, PackedBatch, PrefixLmPacker};

fn packed_batch() -> PackedBatch {
    let packer = PrefixLmPacker::new(0.5).unwrap();
    let examples = vec![
        packer.pack(0, &[1, 2, 3]).unwrap(),
        packer.pack(1, &[4, 5]).unwrap(),
        packer.pack(2, &[6, 7, 8]).unwrap(),
    ];
    MultipackScheduler::new(5, 0)
        .unwrap()
        .schedule(&examples)
        .unwrap()
}

#[test]
fn packed_batch_tokens_targets_prefix_mask_have_shape_batch_by_seq_len() {
    let batch = packed_batch();
    let expected = vec![batch.partition.batch_size, batch.partition.seq_len];

    assert_eq!(batch.tokens.shape().dims, expected);
    assert_eq!(batch.targets.shape().dims, expected);
    assert_eq!(batch.prefix_mask.shape().dims, expected);
}

#[test]
fn packed_batch_dtypes_are_u32_u32_u8() {
    let batch = packed_batch();

    assert_eq!(batch.tokens.dtype(), DType::U32);
    assert_eq!(batch.targets.dtype(), DType::U32);
    assert_eq!(batch.prefix_mask.dtype(), DType::U8);
}

#[test]
fn packed_batch_partition_spans_do_not_cross_rows() {
    let batch = packed_batch();

    for span in &batch.partition.spans {
        assert!(span.start + span.len <= batch.partition.seq_len);
    }
}
