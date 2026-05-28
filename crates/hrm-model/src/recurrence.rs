use config::hrm::HrmConfig;
use tensor_runtime::Tensor;

/// HRM recurrence state: high-level and low-level hidden states.
#[derive(Debug, Clone)]
pub struct HRMState {
    pub z_h: Tensor<f32>,
    pub z_l: Tensor<f32>,
}

impl HRMState {
    pub fn new(z_h: Tensor<f32>, z_l: Tensor<f32>) -> Self {
        assert_eq!(
            z_h.shape(),
            z_l.shape(),
            "z_h and z_l must have the same shape"
        );
        Self { z_h, z_l }
    }
}

/// Scheduled truncated backprop depth.
///
/// Returns `floor(bp_max_steps * min(1.0, step / warmup_steps) / bp_warmup_ratio)`
/// clamped to `0..=bp_max_steps`.
pub fn scheduled_bp_steps(config: &HrmConfig, step: usize) -> usize {
    let warmup = config.warmup_steps as f64;
    let ratio = config.bp_warmup_ratio as f64;
    let max_steps = config.bp_max_steps;

    if warmup <= 0.0 || ratio <= 0.0 {
        return max_steps;
    }

    let progress = (step as f64 / warmup).min(1.0);
    let raw = (max_steps as f64 * progress / ratio).floor() as usize;
    raw.min(max_steps)
}

/// Check convergence: `norm(z_new - z_prev) < eps * max(norm(z_prev), tiny)`.
pub fn check_convergence(z_new: &Tensor<f32>, z_prev: &Tensor<f32>, eps: f32) -> bool {
    assert_eq!(
        z_new.shape(),
        z_prev.shape(),
        "convergence check: shapes must match"
    );
    let new_data = z_new.data();
    let prev_data = z_prev.data();
    let tiny = 1e-8f32;

    let diff_sq: f32 = new_data
        .iter()
        .zip(prev_data.iter())
        .map(|(&n, &p)| (n - p) * (n - p))
        .sum();
    let diff_norm = diff_sq.sqrt();

    let prev_sq: f32 = prev_data.iter().map(|&x| x * x).sum();
    let prev_norm = prev_sq.sqrt();

    diff_norm < eps * prev_norm.max(tiny)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::shape::Shape;

    fn test_config() -> HrmConfig {
        HrmConfig {
            bp_warmup_ratio: 0.2,
            bp_max_steps: 5,
            warmup_steps: 1000,
            ..HrmConfig::default()
        }
    }

    #[test]
    fn scheduled_bp_steps_zero_at_step_zero() {
        let config = test_config();
        assert_eq!(scheduled_bp_steps(&config, 0), 0);
    }

    #[test]
    fn scheduled_bp_steps_max_after_warmup() {
        let config = test_config();
        assert_eq!(scheduled_bp_steps(&config, 1000), 5);
        assert_eq!(scheduled_bp_steps(&config, 2000), 5);
    }

    #[test]
    fn convergence_identical_tensors() {
        let a = Tensor::from_vec(vec![1.0f32; 10], Shape::new(vec![10]));
        let b = Tensor::from_vec(vec![1.0f32; 10], Shape::new(vec![10]));
        assert!(check_convergence(&a, &b, 1e-5));
    }

    #[test]
    fn convergence_different_tensors() {
        let a = Tensor::from_vec(vec![1.0f32; 10], Shape::new(vec![10]));
        let b = Tensor::from_vec(vec![2.0f32; 10], Shape::new(vec![10]));
        assert!(!check_convergence(&a, &b, 1e-5));
    }
}
