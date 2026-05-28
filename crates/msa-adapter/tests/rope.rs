use core_types::shape::Shape;
use msa_adapter::{apply_document_wise_rope, DocumentWiseRoPE};
use tensor_runtime::Tensor;

#[test]
fn document_wise_rope_api_compiles() {
    let _rope = DocumentWiseRoPE {
        base: 10_000.0,
        max_len: 128,
    };
    let mut q = Tensor::from_vec(vec![0.0; 2 * 3 * 4], Shape::new(vec![2, 3, 4]));
    let mut k = Tensor::from_vec(vec![0.0; 2 * 3 * 4], Shape::new(vec![2, 3, 4]));

    apply_document_wise_rope(&mut q, &mut k, &[0, 1]);
}
