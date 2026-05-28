use std::fs;
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::thread::{self, JoinHandle};

use core_types::shape::Shape;
use serde::{Deserialize, Serialize};
use tensor_runtime::Tensor;

const MAGIC: &[u8; 4] = b"HAGI";
const VERSION: u32 = 1;
const CHECKPOINT_BASE: &str = "checkpoints";
const CHECKPOINT_QUEUE_CAP: usize = 2;
const MAX_META_BYTES: usize = 16 * 1024 * 1024;
const MAX_TENSOR_ELEMS: usize = 1_000_000_000;

/// Metadata for a single tensor stored in the checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub numel: usize,
}

/// Checkpoint containing model weights and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub version: u32,
    pub step: u64,
    pub tensors: Vec<TensorMeta>,
}

struct CheckpointJob {
    path: PathBuf,
    step: u64,
    tensors: Vec<(String, Tensor<f32>)>,
}

pub struct AsyncCheckpointWriter {
    sender: SyncSender<CheckpointJob>,
    handle: Option<JoinHandle<io::Result<()>>>,
}

impl AsyncCheckpointWriter {
    pub fn new() -> Self {
        let (sender, receiver): (SyncSender<CheckpointJob>, Receiver<CheckpointJob>) =
            mpsc::sync_channel(CHECKPOINT_QUEUE_CAP);
        let handle = thread::spawn(move || {
            for job in receiver {
                let refs: Vec<(&str, &Tensor<f32>)> = job
                    .tensors
                    .iter()
                    .map(|(name, tensor)| (name.as_str(), tensor))
                    .collect();
                save_checkpoint(&job.path, job.step, &refs)?;
            }
            Ok(())
        });

        Self {
            sender,
            handle: Some(handle),
        }
    }

    pub fn save_snapshot(
        &self,
        path: impl AsRef<Path>,
        step: u64,
        tensors: &[(&str, &Tensor<f32>)],
    ) -> io::Result<()> {
        let snapshot = tensors
            .iter()
            .map(|(name, tensor)| ((*name).to_string(), (*tensor).clone()))
            .collect();
        self.sender
            .try_send(CheckpointJob {
                path: path.as_ref().to_path_buf(),
                step,
                tensors: snapshot,
            })
            .map_err(|e| match e {
                TrySendError::Full(_) => {
                    io::Error::new(io::ErrorKind::WouldBlock, "checkpoint writer queue is full")
                }
                TrySendError::Disconnected(_) => {
                    io::Error::new(io::ErrorKind::BrokenPipe, "checkpoint writer disconnected")
                }
            })
    }

    pub fn finish(mut self) -> io::Result<()> {
        drop(self.sender);
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| io::Error::other("checkpoint writer panicked"))??;
        }
        Ok(())
    }
}

impl Default for AsyncCheckpointWriter {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_checkpoint_path(base: &Path, requested: &Path) -> io::Result<PathBuf> {
    if requested.is_absolute()
        || requested
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::Prefix(_)))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "checkpoint path must be relative and cannot contain '..'",
        ));
    }

    fs::create_dir_all(base)?;
    let canonical_base = base.canonicalize()?;
    let full_path = canonical_base.join(requested);
    if let Some(parent) = full_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let canonical_path = if full_path.exists() {
        full_path.canonicalize()?
    } else {
        let parent = full_path
            .parent()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid checkpoint path"))?
            .canonicalize()?;
        let file_name = full_path.file_name().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "invalid checkpoint path")
        })?;
        parent.join(file_name)
    };

    if !canonical_path.starts_with(&canonical_base) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "checkpoint path escapes checkpoint directory",
        ));
    }

    Ok(canonical_path)
}

fn invalid_data(message: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

/// Saves a collection of named tensors to a binary checkpoint file.
///
/// Format:
/// - 4 bytes magic "HAGI"
/// - 4 bytes version (u32 LE)
/// - 4 bytes metadata length (u32 LE)
/// - N bytes JSON metadata
/// - remaining bytes: concatenated raw f32 data (LE)
pub fn save_checkpoint(path: &Path, step: u64, tensors: &[(&str, &Tensor<f32>)]) -> io::Result<()> {
    let path = validate_checkpoint_path(Path::new(CHECKPOINT_BASE), path)?;

    let mut meta = CheckpointMeta {
        version: VERSION,
        step,
        tensors: Vec::with_capacity(tensors.len()),
    };

    let mut all_data: Vec<f32> = Vec::new();
    for &(name, tensor) in tensors {
        meta.tensors.push(TensorMeta {
            name: name.to_string(),
            shape: tensor.shape().dims.clone(),
            numel: tensor.numel(),
        });
        all_data.extend_from_slice(tensor.data());
    }

    let meta_json = serde_json::to_vec(&meta).map_err(io::Error::other)?;
    if meta_json.len() > u32::MAX as usize {
        return Err(invalid_data("checkpoint metadata too large"));
    }

    let tmp_path = path.with_extension("tmp");
    let mut file = fs::File::create(&tmp_path)?;
    file.write_all(MAGIC)?;
    file.write_all(&VERSION.to_le_bytes())?;
    file.write_all(&(meta_json.len() as u32).to_le_bytes())?;
    file.write_all(&meta_json)?;

    // Write f32 data as little-endian bytes
    for &val in &all_data {
        file.write_all(&val.to_le_bytes())?;
    }
    file.sync_all()?;
    drop(file);
    fs::rename(tmp_path, path)?;

    Ok(())
}

/// Loads a checkpoint from a binary file.
///
/// Returns `(meta, tensors)` where tensors are in the same order as saved.
pub fn load_checkpoint(path: &Path) -> io::Result<(CheckpointMeta, Vec<Tensor<f32>>)> {
    let path = validate_checkpoint_path(Path::new(CHECKPOINT_BASE), path)?;
    let mut file = fs::File::open(path)?;

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
    }

    let mut ver_bytes = [0u8; 4];
    file.read_exact(&mut ver_bytes)?;
    let version = u32::from_le_bytes(ver_bytes);
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported version {}", version),
        ));
    }

    let mut len_bytes = [0u8; 4];
    file.read_exact(&mut len_bytes)?;
    let meta_len = u32::from_le_bytes(len_bytes) as usize;
    if meta_len > MAX_META_BYTES {
        return Err(invalid_data("checkpoint metadata exceeds limit"));
    }

    let mut meta_buf = vec![0u8; meta_len];
    file.read_exact(&mut meta_buf)?;
    let meta: CheckpointMeta = serde_json::from_slice(&meta_buf).map_err(invalid_data)?;

    let mut total_numel = 0usize;
    for tm in &meta.tensors {
        let shape = Shape::new(tm.shape.clone());
        let shape_numel = shape
            .checked_numel()
            .ok_or_else(|| invalid_data("checkpoint tensor shape numel overflow"))?;
        if shape_numel != tm.numel {
            return Err(invalid_data(format!(
                "tensor '{}' shape numel {} != metadata numel {}",
                tm.name, shape_numel, tm.numel
            )));
        }
        total_numel = total_numel
            .checked_add(tm.numel)
            .ok_or_else(|| invalid_data("checkpoint tensor element count overflow"))?;
    }
    if total_numel > MAX_TENSOR_ELEMS {
        return Err(invalid_data(
            "checkpoint tensor element count exceeds limit",
        ));
    }
    let byte_len = total_numel
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| invalid_data("checkpoint tensor byte count overflow"))?;
    let mut raw = vec![0u8; byte_len];
    file.read_exact(&mut raw)?;

    let mut tensors = Vec::with_capacity(meta.tensors.len());
    let mut offset = 0usize;
    for tm in &meta.tensors {
        let mut data = Vec::with_capacity(tm.numel);
        for i in 0..tm.numel {
            let byte_off = (offset + i) * std::mem::size_of::<f32>();
            let bytes: [u8; 4] = raw[byte_off..byte_off + 4]
                .try_into()
                .map_err(|_| invalid_data("truncated data"))?;
            data.push(f32::from_le_bytes(bytes));
        }
        tensors.push(Tensor::from_vec(data, Shape::new(tm.shape.clone())));
        offset += tm.numel;
    }

    Ok((meta, tensors))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> std::path::PathBuf {
        PathBuf::from("tests").join(format!("hagi_test_{}", name))
    }

    fn checkpoint_file(path: &Path) -> PathBuf {
        validate_checkpoint_path(Path::new(CHECKPOINT_BASE), path).unwrap()
    }

    fn cleanup(path: &Path) {
        let _ = fs::remove_file(Path::new(CHECKPOINT_BASE).join(path));
    }

    #[test]
    fn round_trip_single_tensor() {
        let path = temp_path("ckpt_single.bin");
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], Shape::new(vec![3]));

        save_checkpoint(&path, 42, &[("w", &tensor)]).unwrap();
        let (meta, loaded) = load_checkpoint(&path).unwrap();

        assert_eq!(meta.step, 42);
        assert_eq!(meta.tensors.len(), 1);
        assert_eq!(meta.tensors[0].name, "w");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].data(), &[1.0, 2.0, 3.0]);
        assert_eq!(loaded[0].shape().dims, vec![3]);

        cleanup(&path);
    }

    #[test]
    fn round_trip_multiple_tensors() {
        let path = temp_path("ckpt_multi.bin");
        let t1 = Tensor::from_vec(vec![1.0f32, 2.0], Shape::new(vec![2]));
        let t2 = Tensor::from_vec(vec![3.0f32, 4.0, 5.0, 6.0], Shape::new(vec![2, 2]));

        save_checkpoint(&path, 100, &[("a", &t1), ("b", &t2)]).unwrap();
        let (meta, loaded) = load_checkpoint(&path).unwrap();

        assert_eq!(meta.step, 100);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].data(), &[1.0, 2.0]);
        assert_eq!(loaded[1].data(), &[3.0, 4.0, 5.0, 6.0]);
        assert_eq!(loaded[1].shape().dims, vec![2, 2]);

        cleanup(&path);
    }

    #[test]
    fn bad_magic_errors() {
        let path = temp_path("ckpt_bad.bin");
        fs::write(
            checkpoint_file(&path),
            b"NOPE\x01\x00\x00\x00\x00\x00\x00\x00",
        )
        .unwrap();
        let result = load_checkpoint(&path);
        assert!(result.is_err());
        cleanup(&path);
    }

    #[test]
    fn traversal_checkpoint_path_errors() {
        let tensor = Tensor::from_vec(vec![1.0f32], Shape::new(vec![1]));
        let result = save_checkpoint(Path::new("../escape.bin"), 1, &[("w", &tensor)]);

        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn oversized_metadata_errors_before_allocation() {
        let path = temp_path("ckpt_big_meta.bin");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&((MAX_META_BYTES as u32) + 1).to_le_bytes());
        fs::write(checkpoint_file(&path), bytes).unwrap();

        let result = load_checkpoint(&path);
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
        cleanup(&path);
    }

    #[test]
    fn mismatched_tensor_numel_errors_before_data_allocation() {
        let path = temp_path("ckpt_bad_numel.bin");
        let meta = CheckpointMeta {
            version: VERSION,
            step: 1,
            tensors: vec![TensorMeta {
                name: "w".to_string(),
                shape: vec![2],
                numel: 3,
            }],
        };
        let meta_json = serde_json::to_vec(&meta).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&(meta_json.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&meta_json);
        fs::write(checkpoint_file(&path), bytes).unwrap();

        let result = load_checkpoint(&path);
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
        cleanup(&path);
    }
}
