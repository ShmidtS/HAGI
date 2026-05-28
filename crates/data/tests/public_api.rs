use data::{
    DataError, MultipackScheduler, PackedBatch, PackedExample, PackedPartition, PackedSpan,
    PrefixLmPacker,
};

#[test]
fn data_public_api_exports_packed_batch() {
    fn accepts_public_api(
        _packer: PrefixLmPacker,
        _example: PackedExample,
        _scheduler: MultipackScheduler,
        _error: DataError,
        _batch: Option<PackedBatch>,
        _partition: PackedPartition,
        _span: PackedSpan,
    ) {
    }

    let packer = PrefixLmPacker::new(0.5).unwrap();
    let example = packer.pack(1, &[10, 11]).unwrap();
    let scheduler = MultipackScheduler::new(2, 0).unwrap();
    let partition = PackedPartition {
        spans: vec![PackedSpan {
            sequence_id: 1,
            batch_index: 0,
            start: 0,
            len: 2,
            prefix_len: 1,
        }],
        batch_size: 1,
        seq_len: 2,
    };
    let span = partition.spans[0].clone();

    accepts_public_api(
        packer,
        example,
        scheduler,
        DataError::EmptySequence,
        None,
        partition,
        span,
    );
}
