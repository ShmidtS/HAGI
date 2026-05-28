//! HAGI training infrastructure: optimizer, checkpoint, and training loop.

pub mod checkpoint;
pub mod nars_training_wrapper;
pub mod optimizer;
pub mod train_loop;

pub use checkpoint::{
    load_checkpoint, save_checkpoint, AsyncCheckpointWriter, CheckpointMeta, TensorMeta,
};
pub use nars_hrm::NarsHrmConfig;
pub use nars_training_wrapper::{NarsControlledTrainingLoop, NarsTrainStepReport};
pub use optimizer::{
    adamw_step, AdamW, AdamWConfig, AdamWState, Gradient, OptimizerError, Parameter,
};
pub use train_loop::{
    train_step, NarsTrainingConfig, TrainError, TrainStepReport, TrainingLoop,
};
