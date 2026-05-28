use config::ModelConfig;

#[test]
fn model_default_config_valid_after_hrm_default_change() {
    let cfg = ModelConfig::default();

    cfg.validate()
        .expect("default model config must remain valid after HRM default change");
}

#[test]
fn model_config_reports_hrm_validation_error() {
    let mut cfg = ModelConfig::default();
    cfg.hrm.h_layers = 9;

    let err = cfg
        .validate()
        .expect_err("invalid HRM config must fail through model validation");
    assert!(err.to_string().contains("must equal total_layers"));
}
