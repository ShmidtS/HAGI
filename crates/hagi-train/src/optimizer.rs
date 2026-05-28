use tensor_runtime::Tensor;

pub type Parameter = Tensor<f32>;
pub type Gradient = Tensor<f32>;

#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq)]
pub enum OptimizerError {
    #[error("invalid AdamW config: {0}")]
    InvalidConfig(&'static str),
}

/// Configuration for the free-function AdamW optimizer update.
///
/// Expects finite scalar hyperparameters: `lr > 0`, `beta1` and `beta2` in `(0, 1)`, `eps > 0`,
/// `max_norm > 0`, and `weight_decay >= 0`. Validation returns [`OptimizerError::InvalidConfig`]
/// before mutating parameters; updates run on CPU tensors with no CUDA fallback.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamWConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub max_norm: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.01,
            max_norm: 1.0,
        }
    }
}

impl AdamWConfig {
    /// Validates AdamW hyperparameters before a CPU update step mutates parameter tensors.
    pub fn validate(&self) -> Result<(), OptimizerError> {
        if !self.lr.is_finite() || self.lr <= 0.0 {
            return Err(OptimizerError::InvalidConfig("lr must be finite and > 0"));
        }
        if !(0.0..1.0).contains(&self.beta1) {
            return Err(OptimizerError::InvalidConfig("beta1 must be in (0, 1)"));
        }
        if !(0.0..1.0).contains(&self.beta2) {
            return Err(OptimizerError::InvalidConfig("beta2 must be in (0, 1)"));
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err(OptimizerError::InvalidConfig("eps must be finite and > 0"));
        }
        if !self.max_norm.is_finite() || self.max_norm <= 0.0 {
            return Err(OptimizerError::InvalidConfig(
                "max_norm must be finite and > 0",
            ));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(OptimizerError::InvalidConfig(
                "weight_decay must be finite and >= 0",
            ));
        }
        Ok(())
    }
}

/// Mutable AdamW moment buffers for the free-function optimizer update.
///
/// Stores one first- and second-moment vector per parameter tensor, matching each tensor's flat
/// element count. Buffers are initialized lazily on CPU; shape changes between steps panic via
/// assertions in [`adamw_step`].
#[derive(Debug, Clone, Default)]
pub struct AdamWState {
    pub step: u64,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    initialized: bool,
}

impl AdamWState {
    pub fn current_step(&self) -> u64 {
        self.step
    }

    pub fn reset(&mut self) {
        self.step = 0;
        self.m.clear();
        self.v.clear();
        self.initialized = false;
    }
}

/// Performs one in-place AdamW update over matching parameter and gradient tensors.
///
/// `params` and `grads` must have the same length and each parameter must match its gradient's flat
/// element count; mismatches panic. Invalid hyperparameters return [`OptimizerError::InvalidConfig`]
/// before state or parameters are mutated. This is a CPU tensor implementation and does not dispatch
/// to CUDA or provide a CUDA fallback.
pub fn adamw_step(
    params: &mut [Parameter],
    grads: &[Gradient],
    state: &mut AdamWState,
    config: AdamWConfig,
) -> Result<(), OptimizerError> {
    config.validate()?;
    assert_eq!(
        params.len(),
        grads.len(),
        "params and grads must have same length"
    );
    let n = params.len();
    let mut clipped_grads = grads.to_vec();
    clip_gradients(&mut clipped_grads, config.max_norm);

    if !state.initialized || state.m.len() != n {
        state.m = clipped_grads
            .iter()
            .map(|g| vec![0.0f32; g.numel()])
            .collect();
        state.v = clipped_grads
            .iter()
            .map(|g| vec![0.0f32; g.numel()])
            .collect();
        state.initialized = true;
    }

    state.step += 1;
    let t = state.step as f32;
    let bc1 = 1.0 - config.beta1.powf(t);
    let bc2 = 1.0 - config.beta2.powf(t);

    for i in 0..n {
        let g_data = clipped_grads[i].data();
        let numel = g_data.len();
        assert_eq!(
            state.m[i].len(),
            numel,
            "grad shape changed between steps for param {}",
            i
        );

        let m = &mut state.m[i];
        let v = &mut state.v[i];

        for j in 0..numel {
            let g = g_data[j];
            m[j] = config.beta1 * m[j] + (1.0 - config.beta1) * g;
            v[j] = config.beta2 * v[j] + (1.0 - config.beta2) * g * g;
        }

        let mut p_view = params[i].as_mut();
        let p_data = p_view.data_mut();
        assert_eq!(p_data.len(), numel, "param shape mismatch");

        for j in 0..numel {
            let m_hat = m[j] / bc1;
            let v_hat = v[j] / bc2;
            let update = m_hat / (v_hat.sqrt() + config.eps) + config.weight_decay * p_data[j];
            p_data[j] -= config.lr * update;
        }
    }
    Ok(())
}

fn clip_gradients(grads: &mut [Tensor<f32>], max_norm: f32) {
    let mut norm_sq = 0.0f64;
    for grad in grads.iter() {
        for value in grad.data() {
            norm_sq += (*value as f64) * (*value as f64);
        }
    }

    let norm = norm_sq.sqrt() as f32;
    if norm <= max_norm || norm <= 1e-12 {
        return;
    }

    let scale = max_norm / norm;
    for grad in grads.iter_mut() {
        let mut view = grad.as_mut();
        for value in view.data_mut().iter_mut() {
            *value *= scale;
        }
    }
}

/// AdamW optimizer with weight decay decoupled from gradient update.
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub max_norm: f32,
    step: u64,
    /// First moment estimates (one Vec<f32> per parameter tensor).
    m: Vec<Vec<f32>>,
    /// Second moment estimates.
    v: Vec<Vec<f32>>,
    /// Whether buffers have been initialized for the current parameter set.
    initialized: bool,
}

impl AdamW {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            max_norm: 1.0,
            step: 0,
            m: Vec::new(),
            v: Vec::new(),
            initialized: false,
        }
    }

    /// Returns the current step count.
    pub fn current_step(&self) -> u64 {
        self.step
    }

    /// Resets optimizer state (buffers will be re-initialized on next step).
    pub fn reset(&mut self) {
        self.step = 0;
        self.m.clear();
        self.v.clear();
        self.initialized = false;
    }

    pub fn clip_gradients(&self, grads: &mut [Tensor<f32>]) {
        let mut norm_sq = 0.0f64;
        for grad in grads.iter() {
            for value in grad.data() {
                norm_sq += (*value as f64) * (*value as f64);
            }
        }

        let norm = norm_sq.sqrt() as f32;
        if norm <= self.max_norm || norm <= 1e-12 {
            return;
        }

        let scale = self.max_norm / norm;
        for grad in grads.iter_mut() {
            let mut view = grad.as_mut();
            for value in view.data_mut().iter_mut() {
                *value *= scale;
            }
        }
    }

    /// Performs one AdamW update.
    ///
    /// - `params`: mutable parameter tensors
    /// - `grads`: gradient tensors (same shape/count as params)
    ///
    /// Updates params in-place: `param -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * param)`
    pub fn step(&mut self, params: &mut [Tensor<f32>], grads: &[Tensor<f32>]) {
        assert_eq!(
            params.len(),
            grads.len(),
            "params and grads must have same length"
        );
        let n = params.len();
        let mut clipped_grads = grads.to_vec();
        self.clip_gradients(&mut clipped_grads);

        // Lazily initialize momentum buffers
        if !self.initialized || self.m.len() != n {
            self.m = clipped_grads
                .iter()
                .map(|g| vec![0.0f32; g.numel()])
                .collect();
            self.v = clipped_grads
                .iter()
                .map(|g| vec![0.0f32; g.numel()])
                .collect();
            self.initialized = true;
        }

        self.step += 1;
        let t = self.step as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        for i in 0..n {
            let g_data = clipped_grads[i].data();
            let numel = g_data.len();
            assert_eq!(
                self.m[i].len(),
                numel,
                "grad shape changed between steps for param {}",
                i
            );

            let m = &mut self.m[i];
            let v = &mut self.v[i];

            // Update biased first and second moment estimates
            for j in 0..numel {
                let g = g_data[j];
                m[j] = self.beta1 * m[j] + (1.0 - self.beta1) * g;
                v[j] = self.beta2 * v[j] + (1.0 - self.beta2) * g * g;
            }

            // Bias-corrected estimates
            let mut p_view = params[i].as_mut();
            let p_data = p_view.data_mut();
            assert_eq!(p_data.len(), numel, "param shape mismatch");

            for j in 0..numel {
                let m_hat = m[j] / bc1;
                let v_hat = v[j] / bc2;
                let update = m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * p_data[j];
                p_data[j] -= self.lr * update;
            }
        }
    }
}

impl Default for AdamW {
    fn default() -> Self {
        Self::new(1e-3, 0.9, 0.95, 1e-8, 0.01)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::shape::Shape;

    #[test]
    fn one_step_matches_hand_computation() {
        // Single param tensor with 2 elements, single grad tensor
        let mut params = vec![Tensor::from_vec(vec![1.0f32, 2.0], Shape::new(vec![2]))];
        let grads = vec![Tensor::from_vec(vec![0.5f32, -0.5], Shape::new(vec![2]))];

        let lr = 0.1;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let wd = 0.0; // no weight decay for this test
        let mut opt = AdamW::new(lr, beta1, beta2, eps, wd);

        opt.step(&mut params, &grads);

        // Hand computation for step 1:
        // m = 0.1 * 0.5 = 0.05
        // v = 0.001 * 0.25 = 0.00025
        // bc1 = 1 - 0.9 = 0.1
        // bc2 = 1 - 0.999 = 0.001
        // m_hat = 0.05 / 0.1 = 0.5
        // v_hat = 0.00025 / 0.001 = 0.25
        // update = 0.5 / (sqrt(0.25) + 1e-8) = 0.5 / 0.50000001 ≈ 0.99999998
        // param[0] = 1.0 - 0.1 * 0.99999998 ≈ 0.9
        let data = params[0].data();
        let expected_0 = 1.0 - lr * (0.5 / (0.25f32.sqrt() + eps));
        let expected_1 = 2.0 - lr * (-0.5 / (0.25f32.sqrt() + eps));
        assert!(
            (data[0] - expected_0).abs() < 1e-4,
            "param[0]: expected {}, got {}",
            expected_0,
            data[0]
        );
        assert!(
            (data[1] - expected_1).abs() < 1e-4,
            "param[1]: expected {}, got {}",
            expected_1,
            data[1]
        );
        assert_eq!(opt.current_step(), 1);
    }

    #[test]
    fn weight_decay_applied() {
        let mut params = vec![Tensor::from_vec(vec![1.0f32], Shape::new(vec![1]))];
        let grads = vec![Tensor::from_vec(vec![0.0f32], Shape::new(vec![1]))];

        let mut opt = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.1);
        opt.step(&mut params, &grads);

        // grad=0 => m=0, m_hat=0 => update = 0 + 0.1*1.0 = 0.1
        // param = 1.0 - 0.1 * 0.1 = 0.99
        let data = params[0].data();
        assert!(
            (data[0] - 0.99).abs() < 1e-5,
            "expected 0.99, got {}",
            data[0]
        );
    }

    #[test]
    fn multiple_steps_update() {
        let mut params = vec![Tensor::from_vec(vec![1.0f32], Shape::new(vec![1]))];
        let grads = vec![Tensor::from_vec(vec![1.0f32], Shape::new(vec![1]))];

        let mut opt = AdamW::default();
        for _ in 0..10 {
            opt.step(&mut params, &grads);
        }
        assert_eq!(opt.current_step(), 10);
        // After 10 steps with constant positive grad, param should decrease
        assert!(params[0].data()[0] < 1.0);
    }

    #[test]
    fn reset_clears_state() {
        let mut opt = AdamW::default();
        let mut params = vec![Tensor::from_vec(vec![1.0f32], Shape::new(vec![1]))];
        let grads = vec![Tensor::from_vec(vec![1.0f32], Shape::new(vec![1]))];
        opt.step(&mut params, &grads);
        assert_eq!(opt.current_step(), 1);
        opt.reset();
        assert_eq!(opt.current_step(), 0);
    }
}
