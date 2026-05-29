use clifford_core::{Multivector, ProductTable, Rotor};
use core_types::algebra::Cl;
use core_types::ids::DomainId;
use core_types::shape::Shape;
use hdim_model::{
    fused_hrm_hdim_inject, project_hidden_to_multivector, transfer_domain, HiddenToMultivector,
    MultivectorBatch, StructuralFusion, TransferRegistry,
};
use hrm_model::HiddenState;
use tensor_runtime::Tensor;

type Cl3 = Cl<3, 0, 0>;

#[test]
fn project_hidden_to_multivector_returns_batch() {
    let dim = 8;
    let structural_heads = 1;
    let blade_count = 8;
    let mut w_data = vec![0.0f32; dim * dim];
    for i in 0..dim {
        w_data[i * dim + i] = 1.0;
    }
    let projector = HiddenToMultivector::with_weights(
        dim,
        structural_heads,
        blade_count,
        Tensor::from_vec(w_data, Shape::new(vec![dim, dim])),
    );
    let input_data: Vec<f32> = (0..dim).map(|i| (i + 1) as f32).collect();
    let hidden = HiddenState::new(Tensor::from_vec(
        input_data.clone(),
        Shape::new(vec![1, 1, dim]),
    ));

    let batch = project_hidden_to_multivector::<Cl3>(&projector, &hidden)
        .expect("projection should succeed");

    assert_eq!(
        batch.shape().dims,
        vec![1, 1, structural_heads, blade_count]
    );
    assert_eq!(batch.structural_heads, structural_heads);
    assert_eq!(batch.multivector_at(0, 0, 0).coeffs, input_data);
}

#[test]
fn transfer_domain_wraps_registry_transfer_for_batches() {
    let table = ProductTable::generate(3, 0, 0);
    let mut registry = TransferRegistry::<Cl3>::new();
    let source = DomainId(0);
    let target = DomainId(1);
    registry.register_domain(source, Rotor::unit(Multivector::<Cl3>::scalar_one()));
    registry.register_domain(target, Rotor::unit(Multivector::<Cl3>::scalar_one()));

    let coeffs: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
    let batch = MultivectorBatch::<Cl3>::new(
        Tensor::from_vec(coeffs.clone(), Shape::new(vec![1, 1, 1, 8])),
        1,
    );

    let transferred = transfer_domain(&mut registry, source, target, &batch, &table)
        .expect("batch transfer should succeed");

    assert_eq!(transferred.shape().dims, vec![1, 1, 1, 8]);
    assert_eq!(transferred.coeffs.data(), coeffs.as_slice());
}

#[test]
fn fused_hrm_hdim_inject_wraps_structural_fusion() {
    let hidden_size = 8;
    let structural_dim = 8;
    let w_gate = Tensor::from_vec(
        vec![0.0f32; (hidden_size + structural_dim) * hidden_size],
        Shape::new(vec![hidden_size + structural_dim, hidden_size]),
    );
    let w_fuse = Tensor::from_vec(
        vec![0.0f32; structural_dim * hidden_size],
        Shape::new(vec![structural_dim, hidden_size]),
    );
    let fusion = StructuralFusion::with_weights(hidden_size, structural_dim, w_gate, w_fuse);
    let hidden_data: Vec<f32> = (0..hidden_size).map(|i| i as f32 * 0.25).collect();
    let hidden = Tensor::from_vec(hidden_data.clone(), Shape::new(vec![1, 1, hidden_size]));
    let hdim_signal = MultivectorBatch::<Cl3>::new(
        Tensor::from_vec(
            vec![1.0f32; structural_dim],
            Shape::new(vec![1, 1, 1, structural_dim]),
        ),
        1,
    );

    let output =
        fused_hrm_hdim_inject(&fusion, &hidden, &hdim_signal).expect("fusion should succeed");

    assert_eq!(output.shape().dims, vec![1, 1, hidden_size]);
    assert_eq!(output.data(), hidden_data.as_slice());
}
