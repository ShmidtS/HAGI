use data::{MultipackScheduler, PrefixLmPacker};

#[test]
fn packed_batch_prefix_lens_can_be_derived_per_batch_row() {
    let packer = PrefixLmPacker::new(0.5).unwrap();
    let examples = vec![
        packer.pack(0, &[1, 2, 3, 4]).unwrap(),
        packer.pack(1, &[5, 6]).unwrap(),
        packer.pack(2, &[7, 8, 9]).unwrap(),
    ];
    let batch = MultipackScheduler::new(6, 0)
        .unwrap()
        .schedule(&examples)
        .unwrap();

    assert_eq!(batch.tokens.shape().dims, batch.targets.shape().dims);
    assert_eq!(batch.tokens.shape().dims, batch.prefix_mask.shape().dims);

    let mut per_row_prefix_lens = vec![0usize; batch.partition.batch_size];
    let mut per_row_lens = vec![0usize; batch.partition.batch_size];
    for span in &batch.partition.spans {
        per_row_prefix_lens[span.batch_index] += span.prefix_len;
        per_row_lens[span.batch_index] += span.len;
        assert!(span.prefix_len <= span.len);
    }

    for (prefix_len, len) in per_row_prefix_lens.iter().zip(per_row_lens.iter()) {
        assert!(prefix_len <= len);
    }
}
