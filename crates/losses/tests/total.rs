use core_types::shape::Shape;
use losses::{lambda_iso, total_loss, AuxTargets, IsoPairBatch, LossWeights};
use tensor_runtime::Tensor;

#[test]
fn lambda_iso_endpoints() {
    assert_eq!(lambda_iso(0, 0.25, 10), 0.0);
    assert!((lambda_iso(5, 0.25, 10) - 0.125).abs() < 1e-6);
    assert!((lambda_iso(10, 0.25, 10) - 0.25).abs() < 1e-6);
    assert!((lambda_iso(11, 0.25, 10) - 0.25).abs() < 1e-6);
}

#[test]
fn response_only_ce() {
    let logits = Tensor::from_vec(
        vec![-100.0, 100.0, 100.0, -100.0],
        Shape::new(vec![1, 2, 2]),
    );
    let targets = Tensor::from_vec(vec![0u32, 0], Shape::new(vec![1, 2]));
    let prefix_mask = Tensor::from_vec(vec![1u8, 0], Shape::new(vec![1, 2]));
    let src = Tensor::from_vec(vec![0.0f32], Shape::new(vec![1]));
    let tgt = Tensor::from_vec(vec![0.0f32], Shape::new(vec![1]));
    let aux = AuxTargets {
        positive_pairs: &[],
        negative_pairs: &[],
        margin: 1.0,
    };
    let iso = IsoPairBatch {
        u_src: src.as_view(),
        u_tgt: tgt.as_view(),
    };

    let out = total_loss(
        logits.as_view(),
        targets.as_view(),
        &aux,
        &iso,
        &LossWeights {
            lambda_aux: 0.0,
            lambda_iso_target: 0.0,
            iso_warmup_steps: 10,
        },
        0,
        Some(prefix_mask.as_view()),
    )
    .unwrap();

    assert!(
        out.l_ce < 1e-6,
        "prefix token should be ignored, ce={}",
        out.l_ce
    );
    assert_eq!(out.response_token_count, 1);
}

#[test]
fn hand_computed_total_loss() {
    let logits = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], Shape::new(vec![1, 2, 2]));
    let targets = Tensor::from_vec(vec![0u32, 1], Shape::new(vec![1, 2]));
    let prefix_mask = Tensor::from_vec(vec![1u8, 0], Shape::new(vec![1, 2]));
    let pos_a = Tensor::from_vec(vec![1.0f32, 0.0], Shape::new(vec![2]));
    let pos_b = Tensor::from_vec(vec![1.0f32, 0.0], Shape::new(vec![2]));
    let neg_a = Tensor::from_vec(vec![1.0f32, 0.0], Shape::new(vec![2]));
    let neg_b = Tensor::from_vec(vec![1.0f32, 0.0], Shape::new(vec![2]));
    let positives = [(pos_a.as_view(), pos_b.as_view())];
    let negatives = [(neg_a.as_view(), neg_b.as_view())];
    let aux = AuxTargets {
        positive_pairs: &positives,
        negative_pairs: &negatives,
        margin: 2.0,
    };
    let src = Tensor::from_vec(vec![0.0f32, 2.0], Shape::new(vec![2]));
    let tgt = Tensor::from_vec(vec![0.0f32, 0.0], Shape::new(vec![2]));
    let iso = IsoPairBatch {
        u_src: src.as_view(),
        u_tgt: tgt.as_view(),
    };

    let out = total_loss(
        logits.as_view(),
        targets.as_view(),
        &aux,
        &iso,
        &LossWeights {
            lambda_aux: 0.5,
            lambda_iso_target: 0.25,
            iso_warmup_steps: 10,
        },
        5,
        Some(prefix_mask.as_view()),
    )
    .unwrap();

    let expected_ce = 2.0f32.ln();
    let expected_aux = 2.0;
    let expected_iso = 2.0;
    let expected_lambda_iso = 0.125;
    let expected_total = expected_ce + 0.5 * expected_aux + expected_lambda_iso * expected_iso;

    assert!((out.l_ce - expected_ce).abs() < 1e-6);
    assert!((out.l_aux - expected_aux).abs() < 1e-6);
    assert!((out.l_iso - expected_iso).abs() < 1e-6);
    assert!((out.lambda_iso - expected_lambda_iso).abs() < 1e-6);
    assert!((out.l_total - expected_total).abs() < 1e-6);
    assert_eq!(out.response_token_count, 1);
}

#[test]
fn composite_loss_finite_and_grad_non_nan() {
    fn eval_loss(first_logit: f32) -> f32 {
        let mut logits_data: Vec<f32> = (0..2 * 4 * 8)
            .map(|i| (i as f32 % 7.0 - 3.0) * 0.1)
            .collect();
        logits_data[0] = first_logit;
        let logits = Tensor::from_vec(logits_data, Shape::new(vec![2, 4, 8]));
        let targets = Tensor::from_vec(vec![0u32, 1, 2, 3, 4, 5, 6, 7], Shape::new(vec![2, 4]));
        let prefix_mask = Tensor::from_vec(vec![1u8, 0, 0, 0, 1, 0, 0, 0], Shape::new(vec![2, 4]));
        let pos_a = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0], Shape::new(vec![4]));
        let pos_b = Tensor::from_vec(vec![0.9f32, 0.1, 0.0, 0.0], Shape::new(vec![4]));
        let neg_a = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0], Shape::new(vec![4]));
        let neg_b = Tensor::from_vec(vec![0.0f32, 1.0, 0.0, 0.0], Shape::new(vec![4]));
        let positives = [(pos_a.as_view(), pos_b.as_view())];
        let negatives = [(neg_a.as_view(), neg_b.as_view())];
        let aux = AuxTargets {
            positive_pairs: &positives,
            negative_pairs: &negatives,
            margin: 0.5,
        };
        let src = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], Shape::new(vec![4]));
        let tgt = Tensor::from_vec(vec![0.1f32, 0.9, 1.8, 3.2], Shape::new(vec![4]));
        let iso = IsoPairBatch {
            u_src: src.as_view(),
            u_tgt: tgt.as_view(),
        };

        total_loss(
            logits.as_view(),
            targets.as_view(),
            &aux,
            &iso,
            &LossWeights {
                lambda_aux: 0.1,
                lambda_iso_target: 0.2,
                iso_warmup_steps: 2,
            },
            2,
            Some(prefix_mask.as_view()),
        )
        .unwrap()
        .l_total
    }

    let base = eval_loss(-0.3);
    let eps = 1e-3;
    let grad = (eval_loss(-0.3 + eps) - eval_loss(-0.3 - eps)) / (2.0 * eps);

    assert!(base.is_finite());
    assert!(!grad.is_nan());
}
