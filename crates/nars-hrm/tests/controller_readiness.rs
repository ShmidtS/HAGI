use config::HrmConfig;
use nars_hrm::{HrmTrainStepFeedback, NarsHrmController};

struct TestTrainStepReport {
    loss: f32,
}

impl HrmTrainStepFeedback for TestTrainStepReport {
    fn total_loss(&self) -> f32 {
        self.loss
    }

    fn grad_norm(&self) -> f32 {
        self.loss
    }

    fn eval_loss(&self) -> Option<f32> {
        None
    }

    fn should_stop(&self) -> bool {
        false
    }
}

fn report(_step: usize, loss: f32) -> TestTrainStepReport {
    TestTrainStepReport { loss }
}

#[test]
fn controller_checkpoint_roundtrips_state() {
    let mut controller = NarsHrmController::default();
    controller.observe_train_step(&report(0, 1.0));
    controller.observe_train_step(&report(1, 0.25));

    let path = std::env::temp_dir().join(format!(
        "nars-hrm-controller-{}-{}.json",
        std::process::id(),
        controller.concept_store.len()
    ));

    controller.save(&path).unwrap();
    let loaded = NarsHrmController::load(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(loaded.concept_store, controller.concept_store);
    assert_eq!(loaded.goals, controller.goals);
    assert_eq!(loaded.limits, controller.limits);
}

#[test]
fn decreasing_loss_sequence_changes_h_cycle_budget() {
    let mut controller = NarsHrmController::default();
    let config = HrmConfig {
        warmup_steps: 0,
        ..HrmConfig::default()
    };
    let mut h_cycles = Vec::new();

    for (step, loss) in [5.0, 2.0, 1.0, 0.5, 0.1].into_iter().enumerate() {
        controller.observe_train_step(&report(step, loss));
        let policy = controller.begin_step(step + 1, &config);
        h_cycles.push(policy.resolve(&config, &controller.limits).h_cycles);
    }

    assert!(h_cycles.windows(2).any(|pair| pair[0] != pair[1]));
}
