use core_types::shape::Shape;
use msa_adapter::{
    sparse_attention_over_pages, sparse_attention_with_local_context, HostKvPage, MsaError,
};
use std::time::Instant;
use tensor_runtime::Tensor;

fn page(slot_id: u16, keys: Vec<f32>, values: Vec<f32>, shape: Vec<usize>) -> HostKvPage {
    HostKvPage {
        slot_id,
        page_index: 0,
        keys: Tensor::from_vec(keys, Shape::new(shape.clone())),
        values: Tensor::from_vec(values, Shape::new(shape)),
    }
}

fn dense_reference(
    query: &Tensor<f32>,
    keys: &[f32],
    values: &[f32],
    rows: usize,
    hidden: usize,
) -> Vec<f32> {
    let q_shape = query.shape();
    let batch = q_shape.dims[0];
    let tokens = q_shape.dims[1];
    let q_data = query.data();
    let scale = 1.0 / (hidden as f32).sqrt();
    let mut out = vec![0.0; query.numel()];

    for bt in 0..(batch * tokens) {
        let q_offset = bt * hidden;
        let mut scores = Vec::with_capacity(rows);
        for row in 0..rows {
            let row_offset = row * hidden;
            let mut dot = 0.0;
            for d in 0..hidden {
                dot += q_data[q_offset + d] * keys[row_offset + d];
            }
            scores.push(dot * scale);
        }
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores
            .iter()
            .map(|score| (*score - max_score).exp())
            .collect();
        let exp_sum: f32 = exps.iter().sum();
        for row in 0..rows {
            let weight = exps[row] / exp_sum;
            let row_offset = row * hidden;
            let out_offset = bt * hidden;
            for d in 0..hidden {
                out[out_offset + d] += weight * values[row_offset + d];
            }
        }
    }

    out
}

#[test]
fn sparse_attention_empty_pages_returns_zero_query_shape() {
    let query = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![1, 2, 2]));

    let output = sparse_attention_over_pages(&query, &[]).unwrap();

    assert_eq!(output.shape().dims, query.shape().dims);
    assert_eq!(output.data(), &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn sparse_attention_zero_hidden_errors() {
    let query = Tensor::from_vec(Vec::<f32>::new(), Shape::new(vec![1, 1, 0]));
    let err = sparse_attention_over_pages(&query, &[]).unwrap_err();

    assert!(matches!(err, MsaError::ZeroDimension { dim } if dim == "hidden"));
}

#[test]
fn sparse_attention_selected_pages_match_dense_subset_reference() {
    let query = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![1, 2, 2]));
    let pages = vec![
        page(1, vec![1.0, 0.0], vec![10.0, 0.0], vec![2]),
        page(
            2,
            vec![0.0, 1.0, 1.0, 1.0],
            vec![0.0, 20.0, 30.0, 30.0],
            vec![2, 2],
        ),
    ];

    let output = sparse_attention_over_pages(&query, &pages).unwrap();
    let keys = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let values = vec![10.0, 0.0, 0.0, 20.0, 30.0, 30.0];
    let expected = dense_reference(&query, &keys, &values, 3, 2);

    for (actual, expected) in output.data().iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-5);
    }
}

#[test]
fn sparse_attention_no_unselected_allocation_smoke() {
    let query = Tensor::from_vec(vec![1.0, 0.0], Shape::new(vec![1, 1, 2]));
    let pages = vec![page(7, vec![1.0, 0.0], vec![42.0, 24.0], vec![2])];

    let output = sparse_attention_over_pages(&query, &pages).unwrap();

    assert_eq!(output.shape().dims, vec![1, 1, 2]);
    assert!(output.data().iter().all(|value| value.is_finite()));
}

#[test]
fn sparse_attention_with_local_context_matches_combined_reference() {
    let query = Tensor::from_vec(vec![1.0, 0.0], Shape::new(vec![1, 1, 2]));
    let local_keys = Tensor::from_vec(vec![1.0, 0.0], Shape::new(vec![1, 2]));
    let local_values = Tensor::from_vec(vec![2.0, 0.0], Shape::new(vec![1, 2]));
    let pages = vec![page(9, vec![0.0, 1.0], vec![0.0, 4.0], vec![2])];

    let output =
        sparse_attention_with_local_context(&query, &local_keys, &local_values, &pages).unwrap();
    let expected = dense_reference(&query, &[1.0, 0.0, 0.0, 1.0], &[2.0, 0.0, 0.0, 4.0], 2, 2);

    for (actual, expected) in output.data().iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-5);
    }
}

#[test]
fn sparse_attention_micro_benchmark_smoke() {
    let batch = 2;
    let tokens = 16;
    let hidden = 64;
    let rows_per_page = 32;
    let pages_len = 4;
    let iterations = 32;

    let query_data: Vec<f32> = (0..batch * tokens * hidden)
        .map(|i| ((i % 17) as f32 - 8.0) / 17.0)
        .collect();
    let query = Tensor::from_vec(query_data, Shape::new(vec![batch, tokens, hidden]));
    let pages: Vec<HostKvPage> = (0..pages_len)
        .map(|page_index| {
            let key_data: Vec<f32> = (0..rows_per_page * hidden)
                .map(|i| ((i + page_index) % 23) as f32 / 23.0)
                .collect();
            let value_data: Vec<f32> = (0..rows_per_page * hidden)
                .map(|i| ((i + page_index * 3) % 29) as f32 / 29.0)
                .collect();
            HostKvPage {
                slot_id: page_index as u16,
                page_index,
                keys: Tensor::from_vec(key_data, Shape::new(vec![rows_per_page, hidden])),
                values: Tensor::from_vec(value_data, Shape::new(vec![rows_per_page, hidden])),
            }
        })
        .collect();

    let start = Instant::now();
    let mut checksum = 0.0f32;
    for _ in 0..iterations {
        let output = sparse_attention_over_pages(&query, &pages).unwrap();
        checksum += output.data()[0];
    }
    let elapsed = start.elapsed();

    assert!(checksum.is_finite());
    println!(
        "sparse_attention_micro_benchmark_smoke: {} iterations in {:?}",
        iterations, elapsed
    );
}

#[test]
fn rope_document_separation_compiles() {
    let rope = msa_adapter::DocumentWiseRoPE::new();
    let doc_a0 = rope.position(10, 0);
    let doc_a1 = rope.position(10, 1);
    let doc_b0 = rope.position(20, 0);

    assert!(rope.same_document(doc_a0, doc_a1));
    assert!(!rope.same_document(doc_a0, doc_b0));
    assert_eq!(doc_a0.position, doc_b0.position);
}
