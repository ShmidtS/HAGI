use data::{DataError, PackedPartition, PackedSpan};

#[test]
fn packed_partition_records_batch_size_and_seq_len() {
    let partition = PackedPartition {
        spans: vec![PackedSpan {
            sequence_id: 7,
            batch_index: 1,
            start: 2,
            len: 3,
            prefix_len: 1,
        }],
        batch_size: 2,
        seq_len: 8,
    };

    assert_eq!(partition.batch_size, 2);
    assert_eq!(partition.seq_len, 8);
    assert_eq!(partition.spans[0].sequence_id, 7);
}

#[test]
fn data_error_formats_sequence_too_long() {
    let error = DataError::SequenceTooLong {
        len: 9,
        max_tokens: 8,
    };

    assert_eq!(error.to_string(), "sequence length 9 exceeds max tokens 8");
}
