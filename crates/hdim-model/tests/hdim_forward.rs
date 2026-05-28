use clifford_core::{Multivector, ProductTable, Rotor};
use core_types::algebra::Cl;
use core_types::ids::DomainId;
use core_types::shape::Shape;
use hdim_model::{
    hdim_forward, DomainRotor, HiddenToMultivector, InvariantExtractor, StructuralFusion,
    TransferRegistry,
};
use tensor_runtime::Tensor;

type Cl3 = Cl<3, 0, 0>;

#[test]
fn bridge_preserves_hidden_shape() {
    let hidden = make_hidden();
    let (source, target, projector, extractor, transfer, fusion) = make_identity_bridge();

    let output = hdim_forward(
        &hidden, &source, &target, &projector, &extractor, &transfer, &fusion,
    )
    .unwrap();

    assert_eq!(output.shape().dims, hidden.shape().dims);
}

#[test]
fn same_rotor_transfer_identity() {
    let hidden = make_hidden();
    let (source, target, projector, extractor, transfer, fusion) = make_identity_bridge();

    let output = hdim_forward(
        &hidden, &source, &target, &projector, &extractor, &transfer, &fusion,
    )
    .unwrap();

    for (actual, expected) in output.data().iter().zip(hidden.data()) {
        assert!((actual - expected).abs() < 1e-5);
    }
}

fn make_hidden() -> Tensor<f32> {
    Tensor::from_vec(
        (0..2 * 3 * 8).map(|i| (i as f32 * 0.125).sin()).collect(),
        Shape::new(vec![2, 3, 8]),
    )
}

fn make_identity_bridge() -> (
    DomainRotor<Cl3>,
    DomainRotor<Cl3>,
    HiddenToMultivector,
    InvariantExtractor,
    TransferRegistry<Cl3>,
    StructuralFusion,
) {
    let hidden_size = 8;
    let structural_heads = 1;
    let blade_count = 8;
    let mut w_proj_data = vec![0.0f32; hidden_size * blade_count];
    for i in 0..hidden_size {
        w_proj_data[i * blade_count + i] = 1.0;
    }
    let projector = HiddenToMultivector::with_weights(
        hidden_size,
        structural_heads,
        blade_count,
        Tensor::from_vec(w_proj_data, Shape::new(vec![hidden_size, blade_count])),
    );
    let extractor = InvariantExtractor::new(ProductTable::generate(3, 0, 0));
    let unit = Rotor::unit(Multivector::<Cl3>::scalar_one());
    let source = DomainRotor {
        domain_id: DomainId(0),
        rotor: unit,
    };
    let target = DomainRotor {
        domain_id: DomainId(0),
        rotor: Rotor::unit(Multivector::<Cl3>::scalar_one()),
    };
    let transfer = TransferRegistry::<Cl3>::new();
    let w_gate = Tensor::from_vec(
        vec![0.0f32; (hidden_size + blade_count) * hidden_size],
        Shape::new(vec![hidden_size + blade_count, hidden_size]),
    );
    let w_fuse = Tensor::from_vec(
        vec![0.0f32; blade_count * hidden_size],
        Shape::new(vec![blade_count, hidden_size]),
    );
    let fusion = StructuralFusion::with_weights(hidden_size, blade_count, w_gate, w_fuse);
    (source, target, projector, extractor, transfer, fusion)
}
