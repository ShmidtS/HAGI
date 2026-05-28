use core_types::shape::Shape;
use tensor_runtime::Tensor;

use crate::kv_cache::HostKvPage;
use crate::route::MsaError;

/// Sparse attention over fetched memory pages only.
///
/// Router weights are intentionally not accepted here: routing scores are used
/// to select pages, while this CPU reference computes standard attention over
/// the selected memory.
pub fn sparse_attention_over_pages(
    query: &Tensor<f32>,
    pages: &[HostKvPage],
) -> Result<Tensor<f32>, MsaError> {
    let shape = query.shape();
    if shape.rank() != 3 {
        return Err(MsaError::TensorShape(format!(
            "query rank {} != 3",
            shape.rank()
        )));
    }

    let batch = shape.dims[0];
    let tokens = shape.dims[1];
    let hidden = shape.dims[2];
    if hidden == 0 {
        return Err(MsaError::ZeroDimension {
            dim: "hidden".to_string(),
        });
    }

    if pages.is_empty() {
        return Ok(Tensor::from_vec(
            vec![0.0; query.numel()],
            Shape::new(shape.dims.clone()),
        ));
    }

    let mut _total_rows = 0usize;
    for page in pages {
        let key_shape = page.keys.shape();
        let value_shape = page.values.shape();
        let key_rows = page_rows(key_shape, hidden)?;
        let value_rows = page_rows(value_shape, hidden)?;
        if key_rows != value_rows {
            return Err(MsaError::TensorShape(format!(
                "key rows {} != value rows {}",
                key_rows, value_rows
            )));
        }
        _total_rows += key_rows;
    }

    ensure_document_boundaries(pages)?;

    let q_data = query.data();
    let scale = 1.0 / (hidden as f32).sqrt();
    let mut out = vec![0.0f32; batch * tokens * hidden];

    for bt in 0..(batch * tokens) {
        let q_offset = bt * hidden;
        let mut dot_products = Vec::with_capacity(_total_rows);

        for page in pages {
            let key_rows = page_rows(page.keys.shape(), hidden)?;
            let key_data = page.keys.data();
            for row in 0..key_rows {
                let key_offset = row * hidden;
                let mut dot = 0.0f32;
                for d in 0..hidden {
                    dot += q_data[q_offset + d] * key_data[key_offset + d];
                }
                dot_products.push(dot);
            }
        }

        let max_score = dot_products
            .iter()
            .map(|dot| dot * scale)
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = dot_products
            .iter()
            .map(|dot| (dot * scale - max_score).exp())
            .sum();

        let out_offset = bt * hidden;
        let mut dot_index = 0usize;
        for page in pages {
            let rows = page_rows(page.keys.shape(), hidden)?;
            let value_data = page.values.data();
            for row in 0..rows {
                let row_offset = row * hidden;
                let dot = dot_products[dot_index];
                dot_index += 1;
                let weight = if exp_sum > 0.0 && exp_sum.is_finite() {
                    (dot * scale - max_score).exp() / exp_sum
                } else {
                    1.0 / _total_rows as f32
                };
                for d in 0..hidden {
                    out[out_offset + d] += weight * value_data[row_offset + d];
                }
            }
        }
    }

    Ok(Tensor::from_vec(
        out,
        Shape::new(vec![batch, tokens, hidden]),
    ))
}

/// Sparse attention for the full MSA pipeline: local context plus fetched pages.
pub fn sparse_attention_with_local_context(
    query: &Tensor<f32>,
    local_keys: &Tensor<f32>,
    local_values: &Tensor<f32>,
    pages: &[HostKvPage],
) -> Result<Tensor<f32>, MsaError> {
    let local_page = HostKvPage {
        slot_id: u16::MAX,
        page_index: 0,
        keys: local_keys.clone(),
        values: local_values.clone(),
    };
    let mut all_pages = Vec::with_capacity(pages.len() + 1);
    all_pages.push(local_page);
    all_pages.extend_from_slice(pages);
    sparse_attention_over_pages(query, &all_pages)
}

fn page_rows(shape: &Shape, hidden: usize) -> Result<usize, MsaError> {
    match shape.rank() {
        1 if shape.dims[0] == hidden => Ok(1),
        2 if shape.dims[1] == hidden => Ok(shape.dims[0]),
        _ => Err(MsaError::TensorShape(format!(
            "page shape {:?} incompatible with hidden {}",
            shape.dims, hidden
        ))),
    }
}

fn ensure_document_boundaries(pages: &[HostKvPage]) -> Result<(), MsaError> {
    let mut closed_slots = Vec::new();
    let mut current_slot = pages[0].slot_id;
    let mut last_page_index = pages[0].page_index;

    for page in &pages[1..] {
        if page.slot_id == current_slot {
            if page.page_index < last_page_index {
                return Err(MsaError::TensorShape(format!(
                    "page_index {} precedes {} within slot {}",
                    page.page_index, last_page_index, page.slot_id
                )));
            }
            last_page_index = page.page_index;
        } else {
            closed_slots.push(current_slot);
            if closed_slots.contains(&page.slot_id) {
                return Err(MsaError::TensorShape(format!(
                    "slot {} pages are not document-contiguous",
                    page.slot_id
                )));
            }
            current_slot = page.slot_id;
            last_page_index = page.page_index;
        }
    }
    Ok(())
}

/// Sparse attention over a selected subset of memory slots.
///
/// Computes standard scaled dot-product attention restricted to the selected
/// keys and values, producing output with the same shape as the query.
pub struct SparseAttention;

impl SparseAttention {
    pub fn new() -> Self {
        Self
    }

    /// Compute attention over selected slots.
    ///
    /// - `query`: [B, T, hidden]
    /// - `selected_keys`: slice of tensors, each [hidden]
    /// - `selected_values`: slice of tensors, each [hidden]
    ///   Returns output tensor [B, T, hidden].
    pub fn forward(
        &self,
        query: &Tensor<f32>,
        selected_keys: &[Tensor<f32>],
        selected_values: &[Tensor<f32>],
    ) -> Tensor<f32> {
        let shape = query.shape();
        assert_eq!(shape.rank(), 3, "query must be [B, T, hidden]");
        let batch = shape.dims[0];
        let tokens = shape.dims[1];
        let hidden = shape.dims[2];
        assert!(hidden > 0, "hidden dimension must be positive");
        let num_selected = selected_keys.len();
        assert_eq!(
            selected_values.len(),
            num_selected,
            "selected_keys and selected_values must have the same length"
        );
        assert!(num_selected > 0, "must have at least one selected slot");

        let q_data = query.data();
        let scale = 1.0 / (hidden as f32).sqrt();

        let mut out = vec![0.0f32; batch * tokens * hidden];

        for bt in 0..(batch * tokens) {
            let q_off = bt * hidden;

            // Compute attention scores: dot(query, key_i) / sqrt(d)
            let mut scores = Vec::with_capacity(num_selected);
            for selected_key in selected_keys.iter().take(num_selected) {
                let k_data = selected_key.data();
                let mut dot = 0.0f32;
                for d in 0..hidden {
                    dot += q_data[q_off + d] * k_data[d];
                }
                scores.push(dot * scale);
            }

            // Softmax over scores
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            let exp_scores: Vec<f32> = scores
                .iter()
                .map(|&s| {
                    let e = (s - max_score).exp();
                    exp_sum += e;
                    e
                })
                .collect();

            // Weighted sum of values
            let out_off = bt * hidden;
            for s in 0..num_selected {
                let attn_w = if exp_sum > 0.0 {
                    exp_scores[s] / exp_sum
                } else {
                    1.0 / num_selected as f32
                };
                let v_data = selected_values[s].data();
                for d in 0..hidden {
                    out[out_off + d] += attn_w * v_data[d];
                }
            }
        }

        Tensor::from_vec(out, Shape::new(vec![batch, tokens, hidden]))
    }
}

impl Default for SparseAttention {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Reference dense attention over a set of keys/values.
    fn dense_attention(
        query: &Tensor<f32>,
        keys: &[Tensor<f32>],
        values: &[Tensor<f32>],
    ) -> Vec<f32> {
        let shape = query.shape();
        let batch = shape.dims[0];
        let tokens = shape.dims[1];
        let hidden = shape.dims[2];
        let num_kv = keys.len();
        let q_data = query.data();
        let scale = 1.0 / (hidden as f32).sqrt();

        let mut out = vec![0.0f32; batch * tokens * hidden];
        for bt in 0..(batch * tokens) {
            let q_off = bt * hidden;
            let mut scores = Vec::with_capacity(num_kv);
            for s in 0..num_kv {
                let k_data = keys[s].data();
                let mut dot = 0.0f32;
                for d in 0..hidden {
                    dot += q_data[q_off + d] * k_data[d];
                }
                scores.push(dot * scale);
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            let exp_scores: Vec<f32> = scores
                .iter()
                .map(|&s| {
                    let e = (s - max_score).exp();
                    exp_sum += e;
                    e
                })
                .collect();

            let out_off = bt * hidden;
            for s in 0..num_kv {
                let attn_w = if exp_sum > 0.0 {
                    exp_scores[s] / exp_sum
                } else {
                    1.0 / num_kv as f32
                };
                let v_data = values[s].data();
                for d in 0..hidden {
                    out[out_off + d] += attn_w * v_data[d];
                }
            }
        }
        out
    }

    #[test]
    fn output_shape_matches_query() {
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let hidden = 8;
        let query_data: Vec<f32> = (0..2 * 3 * hidden)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let query = Tensor::from_vec(query_data, Shape::new(vec![2, 3, hidden]));

        let keys: Vec<Tensor<f32>> = (0..4)
            .map(|_| {
                let d: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-1.0..1.0)).collect();
                Tensor::from_vec(d, Shape::new(vec![hidden]))
            })
            .collect();
        let values: Vec<Tensor<f32>> = (0..4)
            .map(|_| {
                let d: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-1.0..1.0)).collect();
                Tensor::from_vec(d, Shape::new(vec![hidden]))
            })
            .collect();
        let attn = SparseAttention::new();
        let output = attn.forward(&query, &keys, &values);
        assert_eq!(output.shape().dims, vec![2, 3, hidden]);
    }

    #[test]
    fn sparse_attention_matches_dense_on_same_subset() {
        let mut rng = ChaCha8Rng::seed_from_u64(456);
        let hidden = 16;
        let batch = 2;
        let tokens = 4;
        let num_selected = 5;

        let query_data: Vec<f32> = (0..batch * tokens * hidden)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let query = Tensor::from_vec(query_data, Shape::new(vec![batch, tokens, hidden]));

        let keys: Vec<Tensor<f32>> = (0..num_selected)
            .map(|_| {
                let d: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-1.0..1.0)).collect();
                Tensor::from_vec(d, Shape::new(vec![hidden]))
            })
            .collect();
        let values: Vec<Tensor<f32>> = (0..num_selected)
            .map(|_| {
                let d: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-1.0..1.0)).collect();
                Tensor::from_vec(d, Shape::new(vec![hidden]))
            })
            .collect();
        let attn = SparseAttention::new();
        let sparse_out = attn.forward(&query, &keys, &values);

        let dense_out = dense_attention(&query, &keys, &values);

        let sparse_data = sparse_out.data();
        for i in 0..sparse_data.len() {
            assert!(
                (sparse_data[i] - dense_out[i]).abs() < 1e-5,
                "mismatch at index {}: sparse={} dense={}",
                i,
                sparse_data[i],
                dense_out[i]
            );
        }
    }
}
