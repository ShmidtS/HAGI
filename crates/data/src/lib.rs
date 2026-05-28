//! Data loading, PrefixLM packing, and toy dataset generation.

pub mod batch;
pub mod multipack;
pub mod prefix_lm;
pub mod toy_dataset;

pub use batch::{DataError, PackedBatch, PackedPartition, PackedSpan};
pub use multipack::MultipackScheduler;
pub use prefix_lm::{PackedExample, PrefixLmPacker};
pub use toy_dataset::ToyDataset;
