use data::{DataError, MultipackScheduler, PackedExample};

fn example(sequence_id: usize, tokens: Vec<u32>, prefix_len: usize) -> PackedExample {
    let len = tokens.len();
    PackedExample {
        sequence_id,
        targets: tokens.clone(),
        prefix_mask: (0..len)
            .map(|i| if i < prefix_len { 1 } else { 0 })
            .collect(),
        tokens,
        prefix_len,
    }
}

#[test]
fn multipack_greedy_first_fit_two_rows() {
    let scheduler = MultipackScheduler::new(5, 0).unwrap();
    let batch = scheduler
        .schedule(&[
            example(0, vec![1, 2, 3], 1),
            example(1, vec![4, 5, 6], 1),
            example(2, vec![7, 8], 1),
        ])
        .unwrap();

    assert_eq!(batch.partition.batch_size, 2);
    assert_eq!(batch.partition.spans[0].batch_index, 0);
    assert_eq!(batch.partition.spans[1].batch_index, 1);
    assert_eq!(batch.partition.spans[2].batch_index, 0);
    assert_eq!(batch.partition.spans[2].start, 3);
}

#[test]
fn multipack_padding_uses_prefix_mask_one_and_pad_targets() {
    let scheduler = MultipackScheduler::new(4, 99).unwrap();
    let batch = scheduler.schedule(&[example(0, vec![1, 2], 1)]).unwrap();

    assert_eq!(batch.tokens.data(), &[1, 2, 99, 99]);
    assert_eq!(batch.targets.data(), &[1, 2, 99, 99]);
    assert_eq!(batch.prefix_mask.data(), &[1, 0, 1, 1]);
}

#[test]
fn multipack_partition_metadata_matches_spans() {
    let scheduler = MultipackScheduler::new(4, 0).unwrap();
    let batch = scheduler
        .schedule(&[example(9, vec![1, 2], 1), example(10, vec![3], 1)])
        .unwrap();

    assert_eq!(batch.partition.batch_size, 1);
    assert_eq!(batch.partition.seq_len, 4);
    assert_eq!(batch.partition.spans.len(), 2);
    assert_eq!(batch.partition.spans[0].sequence_id, 9);
    assert_eq!(batch.partition.spans[1].sequence_id, 10);
    assert_eq!(batch.partition.spans[1].start, 2);
}

#[test]
fn multipack_rejects_sequence_too_long() {
    let scheduler = MultipackScheduler::new(2, 0).unwrap();

    assert!(matches!(
        scheduler.schedule(&[example(0, vec![1, 2, 3], 1)]),
        Err(DataError::SequenceTooLong {
            len: 3,
            max_tokens: 2,
        })
    ));
}

#[test]
fn multipack_rejects_overlapping_partition() {
    let scheduler = MultipackScheduler::new(4, 0).unwrap();
    let batch = scheduler
        .schedule(&[example(0, vec![1, 2], 1), example(1, vec![3, 4], 1)])
        .unwrap();

    let first = &batch.partition.spans[0];
    let second = &batch.partition.spans[1];
    assert!(first.start + first.len <= second.start || second.start + second.len <= first.start);
}
