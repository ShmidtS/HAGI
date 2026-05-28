use config::HrmConfig;

#[test]
fn hrm_default_matches_cpu_spec() {
    let cfg = HrmConfig::default();

    assert_eq!(cfg.total_layers, 24);
    assert_eq!(cfg.n_layers(), 24);
    assert_eq!(cfg.h_layers, 8);
    assert_eq!(cfg.l_layers, 16);
    assert_eq!(cfg.hidden_size, 1280);
    assert_eq!(cfg.num_heads, 10);
    assert_eq!(cfg.expansion, 4);
    assert_eq!(cfg.h_cycles, 2);
    assert_eq!(cfg.l_cycles, 3);
    assert_eq!(cfg.vocab_size, 50000);
    assert_eq!(cfg.max_seq_len, 2048);
    assert_eq!(cfg.convergence_eps, 1e-5);
    assert_eq!(cfg.bp_warmup_ratio, 0.2);
    assert_eq!(cfg.bp_max_steps, 5);
    assert_eq!(cfg.warmup_steps, 1000);
    cfg.validate().expect("default HRM config must be valid");
}

#[test]
fn hrm_validate_rejects_bad_layer_sum() {
    let mut cfg = HrmConfig::default();
    cfg.l_layers = 15;

    let err = cfg.validate().expect_err("bad layer sum must fail");
    assert!(err.to_string().contains("must equal total_layers"));
}

#[test]
fn hrm_validate_rejects_hidden_not_divisible_by_heads() {
    let mut cfg = HrmConfig::default();
    cfg.hidden_size = 1281;

    let err = cfg
        .validate()
        .expect_err("hidden_size not divisible by num_heads must fail");
    assert!(err.to_string().contains("must be divisible by num_heads"));
}
