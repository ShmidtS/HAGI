use tensor_runtime::Tensor;

use crate::fetch::fetch_pages;
use crate::kv_cache::HostKvCache;
use crate::registry::SlotRegistry;
use crate::route::{MsaError, RoutingQueryView, SparseRouter};
use crate::sparse_attention::sparse_attention_with_local_context;

pub fn sparse_attention_with_memory(
    local_hidden: &Tensor<f32>,
    local_kv: (&Tensor<f32>, &Tensor<f32>),
    _router: &SparseRouter,
    registry: &SlotRegistry,
    cache: &HostKvCache,
    top_k: usize,
) -> Result<Tensor<f32>, MsaError> {
    let shape = local_hidden.shape();
    if shape.rank() != 3 {
        return Err(MsaError::TensorShape(format!(
            "local_hidden rank {} != 3",
            shape.rank()
        )));
    }
    let hidden = shape.dims[2];
    if hidden == 0 {
        return Err(MsaError::ZeroDimension {
            dim: "hidden".to_string(),
        });
    }

    let query_data = mean_pooled_query(local_hidden)?;
    let selection = crate::route::route_top_k(
        registry,
        RoutingQueryView {
            data: &query_data,
            dim: hidden,
        },
        top_k,
    )?;
    let pages = fetch_pages(cache, selection.slot_ids.as_slice()).wait();
    sparse_attention_with_local_context(local_hidden, local_kv.0, local_kv.1, &pages)
}

fn mean_pooled_query(local_hidden: &Tensor<f32>) -> Result<Vec<f32>, MsaError> {
    let shape = local_hidden.shape();
    let batch = shape.dims[0];
    let tokens = shape.dims[1];
    let hidden = shape.dims[2];
    let count = batch * tokens;
    if count == 0 {
        return Err(MsaError::TensorShape(
            "local_hidden has empty batch or sequence".to_string(),
        ));
    }

    let mut query = vec![0.0f32; hidden];
    let data = local_hidden.data();
    for bt in 0..count {
        let offset = bt * hidden;
        for d in 0..hidden {
            query[d] += data[offset + d];
        }
    }
    for value in &mut query {
        *value /= count as f32;
    }
    Ok(query)
}

pub(crate) fn validate_rope_inputs(
    q: &Tensor<f32>,
    k: &Tensor<f32>,
    doc_ids: &[u16],
) -> Result<(), MsaError> {
    if q.shape().rank() < 1 || k.shape().rank() < 1 {
        return Err(MsaError::TensorShape(
            "q and k must have at least one dimension".to_string(),
        ));
    }
    let batch = q.shape().dims[0];
    if k.shape().dims[0] != batch || doc_ids.len() != batch {
        return Err(MsaError::TensorShape(format!(
            "doc_ids len {} must match batch {}",
            doc_ids.len(),
            batch
        )));
    }
    Ok(())
}
