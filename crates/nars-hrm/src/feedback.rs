use nars_core::{Term, TruthValue};

pub trait HrmTrainStepFeedback {
    fn total_loss(&self) -> f32;
    fn grad_norm(&self) -> f32;
    fn eval_loss(&self) -> Option<f32>;
    fn should_stop(&self) -> bool;
}

#[derive(Debug, Clone, PartialEq)]
pub struct HrmExecutionObservation {
    pub step: usize,
    pub h_cycles_used: usize,
    pub l_cycles_used: usize,
    pub converged: bool,
    pub convergence_delta: f32,
    pub efficiency: f32,
    pub stable_state: bool,
    pub bp_steps: usize,
}

pub fn train_step_judgments(report: &impl HrmTrainStepFeedback) -> Vec<(Term, TruthValue)> {
    let prediction_error = report.total_loss();
    let mut judgments = vec![
        (
            low_prediction_error_term(),
            TruthValue::new(loss_to_frequency(prediction_error), 0.9),
        ),
        (
            trainable_term(),
            TruthValue::new(grad_norm_to_frequency(report.grad_norm()), 0.8),
        ),
        (
            efficient_term(),
            TruthValue::new(if report.should_stop() { 0.2 } else { 0.8 }, 0.7),
        ),
    ];

    if let Some(eval_loss) = report.eval_loss() {
        judgments.push((
            generalizes_term(),
            TruthValue::new(eval_loss_to_frequency(eval_loss), 0.8),
        ));
    }

    judgments
}

pub fn execution_observation_judgments(
    observation: &HrmExecutionObservation,
) -> Vec<(Term, TruthValue)> {
    vec![
        (
            stable_state_term(),
            TruthValue::new(stability_frequency(observation), 0.85),
        ),
        (
            efficient_term(),
            TruthValue::new(efficiency_frequency(observation), 0.75),
        ),
    ]
}

pub fn loss_to_frequency(loss: f32) -> f64 {
    bounded_inverse(loss, 1.0)
}

pub fn grad_norm_to_frequency(grad_norm: f32) -> f64 {
    bounded_inverse(grad_norm, 5.0)
}

pub fn eval_loss_to_frequency(eval_loss: f32) -> f64 {
    bounded_inverse(eval_loss, 1.0)
}

pub fn low_prediction_error_term() -> Term {
    Term::atom("hrm_low_prediction_error")
}

pub fn stable_state_term() -> Term {
    Term::atom("hrm_stable_state")
}

pub fn efficient_term() -> Term {
    Term::atom("hrm_efficient")
}

pub fn trainable_term() -> Term {
    Term::atom("hrm_trainable")
}

pub fn generalizes_term() -> Term {
    Term::atom("hrm_generalizes")
}

fn bounded_inverse(value: f32, scale: f32) -> f64 {
    if !value.is_finite() || value < 0.0 {
        return 0.0;
    }
    (scale / (scale + value)).clamp(0.0, 1.0) as f64
}

fn stability_frequency(observation: &HrmExecutionObservation) -> f64 {
    if !observation.stable_state {
        return 0.0;
    }
    bounded_inverse(observation.convergence_delta, 1.0)
}

fn efficiency_frequency(observation: &HrmExecutionObservation) -> f64 {
    if !observation.efficiency.is_finite() {
        return 0.0;
    }
    observation.efficiency.clamp(0.0, 1.0) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loss_to_frequency_decreases_as_loss_increases() {
        assert!(loss_to_frequency(0.1) > loss_to_frequency(2.0));
    }

    #[test]
    fn early_convergence_creates_stable_state_judgment() {
        let observation = HrmExecutionObservation {
            step: 3,
            h_cycles_used: 1,
            l_cycles_used: 1,
            converged: true,
            convergence_delta: 0.01,
            efficiency: 0.9,
            stable_state: true,
            bp_steps: 1,
        };

        let judgments = execution_observation_judgments(&observation);
        let stable = judgments
            .iter()
            .find(|(term, _)| *term == stable_state_term())
            .expect("stable-state judgment missing");

        assert!(stable.1.frequency() > 0.9);
    }
}
