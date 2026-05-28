use clifford_core::{Multivector, ProductTable, Rotor};
use core_types::algebra::Cl;
use core_types::ids::DomainId;
use core_types::shape::Shape;
use hdim_model::{HiddenToMultivector, StructuralFusion, TransferRegistry};
use hrm_model::HiddenState;
use tensor_runtime::Tensor;

type Cl3 = Cl<3, 0, 0>;

/// Integration test: project -> transfer -> fuse roundtrip preserves [B, T, hidden] shape.
#[test]
fn project_transfer_fuse_roundtrip_shape() {
    let hidden_size = 16;
    let structural_heads = 2;
    let blade_count = 8; // Cl<3,0,0>
    let batch = 2;
    let tokens = 3;

    // 1. Project hidden -> multivector coefficients
    let proj = HiddenToMultivector::new(hidden_size, structural_heads, blade_count);
    let h_data = vec![0.3f32; batch * tokens * hidden_size];
    let h_tensor = Tensor::from_vec(h_data.clone(), Shape::new(vec![batch, tokens, hidden_size]));
    let hidden = HiddenState::new(h_tensor.clone());
    let projected = proj.forward(&hidden);

    assert_eq!(
        projected.shape().dims,
        vec![batch, tokens, structural_heads, blade_count],
        "projected shape mismatch"
    );

    // 2. Transfer each (batch, token, head) multivector between domains
    let table = ProductTable::generate(3, 0, 0);
    let mut registry = TransferRegistry::<Cl3>::new();
    let source = DomainId(0);
    let target = DomainId(1);
    registry.register_domain(source, Rotor::unit(Multivector::<Cl3>::scalar_one()));
    registry.register_domain(target, Rotor::unit(Multivector::<Cl3>::scalar_one()));

    let proj_data = projected.data();
    let structural_dim = structural_heads * blade_count;
    let mut transferred = vec![0.0f32; batch * tokens * structural_dim];

    for bt in 0..(batch * tokens) {
        for head in 0..structural_heads {
            let offset = bt * structural_dim + head * blade_count;
            let coeffs = proj_data[offset..offset + blade_count].to_vec();
            let mv = Multivector::<Cl3>::from_coeffs(coeffs);
            let result = registry
                .transfer(source, target, &mv, &table)
                .expect("transfer should succeed");
            transferred[offset..offset + blade_count].copy_from_slice(&result.coeffs);
        }
    }

    let structural_tensor = Tensor::from_vec(
        transferred,
        Shape::new(vec![batch, tokens, structural_heads, blade_count]),
    );

    // 3. Fuse back into hidden state
    let fusion = StructuralFusion::new(hidden_size, structural_dim);
    let output = fusion.forward(&h_tensor, &structural_tensor);

    assert_eq!(
        output.shape().dims,
        vec![batch, tokens, hidden_size],
        "integration roundtrip: output shape must be [B, T, hidden]"
    );
}

/// Integration test with identity-like configuration: projection copies, transfer is identity, fusion passes through.
#[test]
fn identity_roundtrip_preserves_values() {
    let dim = 8;
    let structural_heads = 1;
    let blade_count = 8;
    let batch = 1;
    let tokens = 1;

    // Identity projection
    let mut w_proj_data = vec![0.0f32; dim * dim];
    for i in 0..dim {
        w_proj_data[i * dim + i] = 1.0;
    }
    let w_proj = Tensor::from_vec(w_proj_data, Shape::new(vec![dim, dim]));
    let proj = HiddenToMultivector::with_weights(dim, structural_heads, blade_count, w_proj);

    // Input
    let input_data: Vec<f32> = (0..dim).map(|i| (i + 1) as f32 * 0.1).collect();
    let h_tensor = Tensor::from_vec(input_data.clone(), Shape::new(vec![batch, tokens, dim]));
    let hidden = HiddenState::new(h_tensor.clone());

    // Project
    let projected = proj.forward(&hidden);

    // Transfer with identity rotors (same domain = no-op)
    let table = ProductTable::generate(3, 0, 0);
    let mut registry = TransferRegistry::<Cl3>::new();
    let domain = DomainId(0);
    registry.register_domain(domain, Rotor::unit(Multivector::<Cl3>::scalar_one()));

    let proj_data = projected.data();
    let mv = Multivector::<Cl3>::from_coeffs(proj_data.to_vec());
    let result = registry
        .transfer(domain, domain, &mv, &table)
        .expect("same-domain transfer should succeed");
    let structural_tensor = Tensor::from_vec(
        result.coeffs,
        Shape::new(vec![batch, tokens, structural_heads, blade_count]),
    );

    // Fuse with zero gate and zero fuse -> output = h_state
    let w_gate = Tensor::from_vec(
        vec![0.0f32; (dim + dim) * dim],
        Shape::new(vec![dim + dim, dim]),
    );
    let w_fuse = Tensor::from_vec(vec![0.0f32; dim * dim], Shape::new(vec![dim, dim]));
    let fusion = StructuralFusion::with_weights(dim, dim, w_gate, w_fuse);
    let output = fusion.forward(&h_tensor, &structural_tensor);

    let out_data = output.data();
    for i in 0..dim {
        assert!(
            (out_data[i] - input_data[i]).abs() < 1e-5,
            "identity roundtrip mismatch at index {}: expected {}, got {}",
            i,
            input_data[i],
            out_data[i]
        );
    }
}
