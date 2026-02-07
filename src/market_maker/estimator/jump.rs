//! Jump process estimation.
//!
//! - JumpEstimatorConfig: Configuration for jump detection
//! - JumpEstimator: Poisson jump process parameter estimation

use std::collections::VecDeque;

// ============================================================================
// Jump Process Estimator (First Principles Gap 1)
// ============================================================================

/// Configuration for jump process estimation.
#[derive(Debug, Clone)]
pub struct JumpEstimatorConfig {
    /// Threshold for jump detection (in sigma units, e.g., 3.0 = 3σ)
    pub jump_threshold_sigmas: f64,
    /// Window for jump intensity estimation (ms)
    pub window_ms: u64,
    /// EWMA alpha for online parameter updates
    pub alpha: f64,
    /// Minimum jumps before estimates are valid
    pub min_jumps: usize,
}

impl Default for JumpEstimatorConfig {
    fn default() -> Self {
        Self {
            jump_threshold_sigmas: 3.0,
            window_ms: 300_000, // 5 minutes
            alpha: 0.1,
            min_jumps: 5,
        }
    }
}

/// Jump process estimator for explicit λ (intensity), μ_j (mean), σ_j (std dev).
///
/// Implements the jump component of the price process:
/// dP = μ dt + σ dW + J dN where J ~ N(μ_j, σ_j²), N ~ Poisson(λ)
///
/// Key formulas:
/// - Total variance over horizon h: Var[P(t+h) - P(t)] = σ²h + λh×E[J²]
/// - E[J²] = μ_j² + σ_j²
#[derive(Debug)]
pub struct JumpEstimator {
    /// Jump intensity (jumps per second)
    lambda_jump: f64,
    /// Mean jump size (in log-return units)
    mu_jump: f64,
    /// Jump size standard deviation
    sigma_jump: f64,
    /// Recent detected jumps: (timestamp_ms, size, is_positive)
    recent_jumps: VecDeque<(u64, f64, bool)>,
    /// Online mean tracker
    sum_sizes: f64,
    sum_sq_sizes: f64,
    jump_count: usize,
    /// Configuration
    config: JumpEstimatorConfig,
    /// Last update timestamp
    last_update_ms: u64,
}

impl JumpEstimator {
    /// Create a new jump estimator with default configuration.
    pub fn new() -> Self {
        Self::with_config(JumpEstimatorConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: JumpEstimatorConfig) -> Self {
        Self {
            lambda_jump: 0.0,
            mu_jump: 0.0,
            sigma_jump: 0.0001, // Small default
            recent_jumps: VecDeque::with_capacity(100),
            sum_sizes: 0.0,
            sum_sq_sizes: 0.0,
            jump_count: 0,
            config,
            last_update_ms: 0,
        }
    }

    /// Check if a return qualifies as a jump.
    ///
    /// A return is a "jump" if |return| > threshold × σ_clean
    pub fn is_jump(&self, log_return: f64, sigma_clean: f64) -> bool {
        let threshold = self.config.jump_threshold_sigmas * sigma_clean;
        log_return.abs() > threshold
    }

    /// Record a detected jump and update parameters.
    pub fn record_jump(&mut self, timestamp_ms: u64, log_return: f64) {
        let size = log_return.abs();
        let is_positive = log_return > 0.0;

        self.recent_jumps
            .push_back((timestamp_ms, size, is_positive));

        // Update online statistics
        self.sum_sizes += size;
        self.sum_sq_sizes += size * size;
        self.jump_count += 1;

        // Expire old jumps
        let cutoff = timestamp_ms.saturating_sub(self.config.window_ms);
        while self
            .recent_jumps
            .front()
            .map(|(t, _, _)| *t < cutoff)
            .unwrap_or(false)
        {
            if let Some((_, old_size, _)) = self.recent_jumps.pop_front() {
                self.sum_sizes -= old_size;
                self.sum_sq_sizes -= old_size * old_size;
                self.jump_count = self.jump_count.saturating_sub(1);
            }
        }

        // Update parameters using EWMA
        self.update_parameters(timestamp_ms);
        self.last_update_ms = timestamp_ms;
    }

    /// Update lambda, mu, sigma from recent jumps.
    fn update_parameters(&mut self, _timestamp_ms: u64) {
        if self.jump_count < self.config.min_jumps {
            return;
        }

        let n = self.jump_count as f64;
        let window_secs = self.config.window_ms as f64 / 1000.0;

        // Lambda: jumps per second
        let new_lambda = n / window_secs;
        self.lambda_jump =
            self.config.alpha * new_lambda + (1.0 - self.config.alpha) * self.lambda_jump;

        // Mu: mean jump size (signed average would be near 0, use unsigned)
        let new_mu = self.sum_sizes / n;
        self.mu_jump = self.config.alpha * new_mu + (1.0 - self.config.alpha) * self.mu_jump;

        // Sigma: standard deviation of jump sizes
        let variance = (self.sum_sq_sizes / n) - (new_mu * new_mu);
        if variance > 0.0 {
            let new_sigma = variance.sqrt();
            self.sigma_jump =
                self.config.alpha * new_sigma + (1.0 - self.config.alpha) * self.sigma_jump;
        }
    }

    /// Process a return observation (called on each volume bucket).
    pub fn on_return(&mut self, timestamp_ms: u64, log_return: f64, sigma_clean: f64) {
        if self.is_jump(log_return, sigma_clean) {
            self.record_jump(timestamp_ms, log_return);
        }
    }

    /// Get jump intensity (λ) - jumps per second.
    pub fn lambda(&self) -> f64 {
        self.lambda_jump
    }

    /// Get mean jump size (μ_j) in log-return units.
    pub fn mu(&self) -> f64 {
        self.mu_jump
    }

    /// Get jump size standard deviation (σ_j).
    pub fn sigma(&self) -> f64 {
        self.sigma_jump
    }

    /// Get expected jump variance contribution: E[J²] = μ² + σ².
    pub fn expected_jump_variance(&self) -> f64 {
        self.mu_jump.powi(2) + self.sigma_jump.powi(2)
    }

    /// Get total variance over horizon including jumps.
    ///
    /// Var[P(t+h) - P(t)] = σ_diffusion² × h + λ × h × E[J²]
    pub fn total_variance(&self, sigma_diffusion: f64, horizon_secs: f64) -> f64 {
        let diffusion_var = sigma_diffusion.powi(2) * horizon_secs;
        let jump_var = self.lambda_jump * horizon_secs * self.expected_jump_variance();
        diffusion_var + jump_var
    }

    /// Get total volatility (sqrt of total variance).
    pub fn total_sigma(&self, sigma_diffusion: f64, horizon_secs: f64) -> f64 {
        self.total_variance(sigma_diffusion, horizon_secs).sqrt()
    }

    /// Check if estimator has enough data.
    pub fn is_warmed_up(&self) -> bool {
        self.jump_count >= self.config.min_jumps
    }

    /// Get number of jumps in current window.
    pub fn jump_count(&self) -> usize {
        self.jump_count
    }

    /// Get fraction of recent returns that were jumps.
    pub fn jump_fraction(&self, total_returns: usize) -> f64 {
        if total_returns == 0 {
            0.0
        } else {
            self.jump_count as f64 / total_returns as f64
        }
    }
}

impl Default for JumpEstimator {
    fn default() -> Self {
        Self::new()
    }
}
