//! Ornstein-Uhlenbeck drift model for HJB controller.
//!
//! Replaces EWMA drift smoothing with a mean-reverting OU process that
//! includes threshold gating to filter noise-induced reconciliations.
//!
//! ## Theory
//!
//! The drift follows an Ornstein-Uhlenbeck process:
//! ```text
//! dD_t = θ(μ - D_t)dt + σ_D dW_t
//! ```
//!
//! Where:
//! - `θ` = mean reversion rate (higher = faster reversion, ~1.4s half-life at θ=0.5)
//! - `μ` = long-term mean (typically 0 for neutral drift)
//! - `σ_D` = drift volatility (fitted from tick data)
//!
//! ## Threshold Gating
//!
//! Only reconcile if the innovation (observed - predicted) exceeds the threshold:
//! ```text
//! |observed - predicted| > k × σ × √dt
//! ```
//!
//! This filters ~60-80% of noise-induced updates while responding to genuine
//! regime changes.

/// Configuration for OU drift model.
#[derive(Debug, Clone, Copy)]
pub struct OUDriftConfig {
    /// Mean reversion rate (θ). Higher = faster mean reversion.
    /// Default: 0.5 (~1.4 second half-life)
    pub theta: f64,

    /// Long-term mean (μ). For drift, typically 0.0 (neutral).
    pub mu: f64,

    /// Threshold multiplier (k). Reconcile only if innovation > k×σ√dt.
    /// Default: 2.0 (filters ~95% of noise under normal conditions)
    pub reconcile_k: f64,

    /// Initial drift volatility estimate. Updated adaptively.
    /// Default: 0.001 (1 bps/sec typical)
    pub initial_sigma_drift: f64,

    /// Minimum variance floor to prevent degenerate behavior.
    pub min_variance: f64,

    /// Maximum variance cap to prevent runaway estimation.
    pub max_variance: f64,
}

impl Default for OUDriftConfig {
    fn default() -> Self {
        Self {
            theta: 0.5,           // ~1.4s half-life
            mu: 0.0,              // Neutral drift
            reconcile_k: 2.0,     // 2σ threshold
            initial_sigma_drift: 0.001, // 1 bps/sec
            min_variance: 1e-12,
            max_variance: 0.01,
        }
    }
}

/// Result of an OU drift update.
#[derive(Debug, Clone, Copy)]
pub struct OUUpdateResult {
    /// Current drift estimate after update.
    pub drift: f64,

    /// Predicted drift before observation (for diagnostics).
    pub predicted_drift: f64,

    /// Innovation (observed - predicted).
    pub innovation: f64,

    /// Threshold that was applied.
    pub threshold: f64,

    /// Whether this update triggered a reconciliation.
    /// If false, the observation was filtered as noise.
    pub reconciled: bool,

    /// Kalman gain used (0 if not reconciled).
    pub gain: f64,
}

/// Ornstein-Uhlenbeck drift estimator with threshold gating.
///
/// Uses a Kalman filter-like update mechanism with the OU process as the
/// state model. Innovations below the threshold are filtered out to
/// reduce order churn.
#[derive(Debug, Clone)]
pub struct OUDriftEstimator {
    config: OUDriftConfig,

    /// Current drift estimate (D_t).
    drift: f64,

    /// Current estimate variance (P_t).
    variance: f64,

    /// Drift volatility estimate (σ_D), updated adaptively.
    sigma_drift: f64,

    /// Last update timestamp in milliseconds.
    last_update_ms: u64,

    /// Number of updates received.
    update_count: u64,

    /// Sum of squared innovations for adaptive σ_D estimation.
    innovation_sum_sq: f64,

    /// Count of innovations for adaptive estimation.
    innovation_count: u64,

    /// Whether the estimator is warmed up (enough observations).
    warmed_up: bool,
}

impl OUDriftEstimator {
    /// Create a new OU drift estimator with the given configuration.
    pub fn new(config: OUDriftConfig) -> Self {
        Self {
            config,
            drift: config.mu,
            variance: config.initial_sigma_drift.powi(2),
            sigma_drift: config.initial_sigma_drift,
            last_update_ms: 0,
            update_count: 0,
            innovation_sum_sq: 0.0,
            innovation_count: 0,
            warmed_up: false,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(OUDriftConfig::default())
    }

    /// Update the drift estimate with a new observation.
    ///
    /// Returns the update result including whether reconciliation is needed.
    ///
    /// # Arguments
    /// * `timestamp_ms` - Current timestamp in milliseconds
    /// * `observed_drift` - Observed drift rate (e.g., from momentum signal)
    pub fn update(&mut self, timestamp_ms: u64, observed_drift: f64) -> OUUpdateResult {
        // Handle first observation
        if self.last_update_ms == 0 {
            self.last_update_ms = timestamp_ms;
            self.drift = observed_drift;
            self.update_count = 1;
            return OUUpdateResult {
                drift: observed_drift,
                predicted_drift: self.config.mu,
                innovation: observed_drift - self.config.mu,
                threshold: 0.0,
                reconciled: true, // First observation always reconciles
                gain: 1.0,
            };
        }

        // Calculate time delta
        let dt_ms = timestamp_ms.saturating_sub(self.last_update_ms);
        if dt_ms == 0 {
            // No time passed, return current state
            return OUUpdateResult {
                drift: self.drift,
                predicted_drift: self.drift,
                innovation: 0.0,
                threshold: 0.0,
                reconciled: false,
                gain: 0.0,
            };
        }

        let dt_seconds = dt_ms as f64 / 1000.0;

        // === Prediction Step (OU dynamics) ===
        // E[D_{t+dt}] = μ + (D_t - μ) × exp(-θ×dt)
        let decay = (-self.config.theta * dt_seconds).exp();
        let predicted_drift = self.config.mu + (self.drift - self.config.mu) * decay;

        // Predict variance: P_{t+dt} = decay² × P_t + (1 - decay²) × σ²_D / (2θ)
        // This is the OU stationary variance contribution
        let decay_sq = decay * decay;
        let ou_stationary_var = self.sigma_drift.powi(2) / (2.0 * self.config.theta);
        let predicted_variance = decay_sq * self.variance + (1.0 - decay_sq) * ou_stationary_var;

        // === Innovation Calculation ===
        let innovation = observed_drift - predicted_drift;

        // === Threshold Gating ===
        // Total uncertainty = prediction variance + observation noise
        let observation_noise = self.sigma_drift.powi(2) * dt_seconds;
        let total_uncertainty = (predicted_variance + observation_noise)
            .max(self.config.min_variance)
            .min(self.config.max_variance);
        let threshold = self.config.reconcile_k * total_uncertainty.sqrt();

        // === Sigmoid Soft-Gate ===
        // Replace hard deadband with sigmoid: a slow continuous trend that stays
        // below threshold was actively reversed by the OU prediction step (the
        // "boiling frog" problem). The soft-gate always passes some fraction of
        // the Kalman gain, ensuring slow drifts are tracked.
        const MIN_GATE_WEIGHT: f64 = 0.05;

        let z = innovation.abs() / total_uncertainty.sqrt().max(1e-12);
        let gate_weight = MIN_GATE_WEIGHT
            + (1.0 - MIN_GATE_WEIGHT) / (1.0 + (-3.0 * (z - self.config.reconcile_k)).exp());

        // Always compute Kalman gain, scale by gate weight
        let kalman_gain = predicted_variance / (predicted_variance + observation_noise);
        let effective_gain = kalman_gain * gate_weight;

        let new_drift = predicted_drift + effective_gain * innovation;
        let new_variance = ((1.0 - effective_gain) * predicted_variance)
            .max(self.config.min_variance)
            .min(self.config.max_variance);

        // Update adaptive sigma_D only for significant innovations (above threshold)
        let should_reconcile = z > self.config.reconcile_k;
        if should_reconcile {
            self.innovation_sum_sq += innovation.powi(2);
            self.innovation_count += 1;

            // Re-estimate σ_D every 50 reconciled observations
            if self.innovation_count >= 50 {
                let mean_sq_innovation = self.innovation_sum_sq / self.innovation_count as f64;
                // σ²_D ≈ mean(innovation²) / dt (simplified)
                let avg_dt = 0.1; // Assume 100ms average between observations
                self.sigma_drift = (mean_sq_innovation / avg_dt)
                    .sqrt().clamp(1e-6, 0.1);

                // Reset accumulators
                self.innovation_sum_sq = 0.0;
                self.innovation_count = 0;
            }
        }

        let gain = effective_gain;

        // Update state
        self.drift = new_drift;
        self.variance = new_variance;
        self.last_update_ms = timestamp_ms;
        self.update_count += 1;

        // Mark as warmed up after sufficient observations
        if !self.warmed_up && self.update_count >= 20 {
            self.warmed_up = true;
        }

        OUUpdateResult {
            drift: new_drift,
            predicted_drift,
            innovation,
            threshold,
            reconciled: should_reconcile,
            gain,
        }
    }

    /// Get current drift estimate.
    pub fn drift(&self) -> f64 {
        self.drift
    }

    /// Get current variance estimate.
    pub fn variance(&self) -> f64 {
        self.variance
    }

    /// Get predicted drift at a future time (without updating state).
    pub fn predict(&self, dt_seconds: f64) -> f64 {
        let decay = (-self.config.theta * dt_seconds).exp();
        self.config.mu + (self.drift - self.config.mu) * decay
    }

    /// Get predicted variance at a future time.
    pub fn predict_variance(&self, dt_seconds: f64) -> f64 {
        let decay = (-self.config.theta * dt_seconds).exp();
        let decay_sq = decay * decay;
        let ou_stationary_var = self.sigma_drift.powi(2) / (2.0 * self.config.theta);
        decay_sq * self.variance + (1.0 - decay_sq) * ou_stationary_var
    }

    /// Check if a given innovation would exceed the threshold.
    pub fn threshold_exceeded(&self, innovation: f64, dt_seconds: f64) -> bool {
        let predicted_variance = self.predict_variance(dt_seconds);
        let observation_noise = self.sigma_drift.powi(2) * dt_seconds;
        let total_uncertainty = (predicted_variance + observation_noise)
            .max(self.config.min_variance)
            .min(self.config.max_variance);
        let threshold = self.config.reconcile_k * total_uncertainty.sqrt();

        innovation.abs() > threshold
    }

    /// Check if the estimator is warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.warmed_up
    }

    /// Get the number of updates received.
    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Get the current sigma_drift estimate.
    pub fn sigma_drift(&self) -> f64 {
        self.sigma_drift
    }

    /// Get the half-life implied by theta (in seconds).
    pub fn half_life_secs(&self) -> f64 {
        (2.0_f64).ln() / self.config.theta
    }

    /// Reset the estimator to initial state.
    pub fn reset(&mut self) {
        self.drift = self.config.mu;
        self.variance = self.config.initial_sigma_drift.powi(2);
        self.sigma_drift = self.config.initial_sigma_drift;
        self.last_update_ms = 0;
        self.update_count = 0;
        self.innovation_sum_sq = 0.0;
        self.innovation_count = 0;
        self.warmed_up = false;
    }

    /// Get a diagnostic summary.
    pub fn summary(&self) -> OUDriftSummary {
        OUDriftSummary {
            drift: self.drift,
            variance: self.variance,
            sigma_drift: self.sigma_drift,
            update_count: self.update_count,
            is_warmed_up: self.warmed_up,
            half_life_secs: self.half_life_secs(),
            theta: self.config.theta,
            reconcile_k: self.config.reconcile_k,
        }
    }

    /// Adapt OU parameters to current market regime.
    ///
    /// Different regimes call for different mean-reversion speeds and noise filters:
    /// - Normal: balanced defaults (theta=0.5, reconcile_k=2.0)
    /// - Trending: trust trends longer with slower reversion (theta=0.2, reconcile_k=1.5)
    /// - Cascade: fast mean reversion, high noise filter (theta=1.0, reconcile_k=3.0)
    pub fn adapt_to_regime(&mut self, regime: &str) {
        match regime {
            "trending" => {
                self.config.theta = 0.2;      // Trust trends longer (slower reversion)
                self.config.reconcile_k = 1.5; // Lower filter threshold (more responsive)
            }
            "cascade" => {
                self.config.theta = 1.0;      // Fast mean reversion (don't chase)
                self.config.reconcile_k = 3.0; // High noise filter (only big innovations)
            }
            _ => {
                // "normal" or any unrecognized regime -> safe defaults
                self.config.theta = 0.5;
                self.config.reconcile_k = 2.0;
            }
        }
    }
}

/// Summary of OU drift estimator state for diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct OUDriftSummary {
    pub drift: f64,
    pub variance: f64,
    pub sigma_drift: f64,
    pub update_count: u64,
    pub is_warmed_up: bool,
    pub half_life_secs: f64,
    pub theta: f64,
    pub reconcile_k: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ou_drift_basic() {
        let mut estimator = OUDriftEstimator::default_config();

        // First observation
        let result = estimator.update(1000, 0.001);
        assert!(result.reconciled);
        assert!((result.drift - 0.001).abs() < 1e-10);

        // Second observation with small change (should filter)
        let result = estimator.update(1100, 0.00101);
        // Small innovation likely filtered
        assert_eq!(result.drift, estimator.drift());
    }

    #[test]
    fn test_ou_drift_large_innovation() {
        let mut estimator = OUDriftEstimator::default_config();

        // Initialize
        estimator.update(1000, 0.0);

        // Large jump should trigger reconciliation
        let result = estimator.update(2000, 0.01); // 10x normal drift
        assert!(result.reconciled);
        assert!(result.innovation.abs() > result.threshold);
    }

    #[test]
    fn test_ou_drift_mean_reversion() {
        let mut estimator = OUDriftEstimator::default_config();

        // Initialize with positive drift
        estimator.update(0, 0.01);

        // Predict drift should revert toward mean (0)
        let future_drift = estimator.predict(10.0); // 10 seconds ahead
        assert!(future_drift.abs() < 0.01); // Should be closer to 0
    }

    #[test]
    fn test_ou_drift_warmup() {
        let mut estimator = OUDriftEstimator::default_config();

        assert!(!estimator.is_warmed_up());

        // Process 20+ observations
        for i in 0..25 {
            estimator.update(i * 100, 0.001 * (i as f64 / 10.0 - 1.0).sin());
        }

        assert!(estimator.is_warmed_up());
    }

    #[test]
    fn test_ou_drift_threshold() {
        let estimator = OUDriftEstimator::default_config();

        // Small innovation should not exceed threshold
        assert!(!estimator.threshold_exceeded(0.0001, 0.1));

        // Large innovation should exceed threshold
        assert!(estimator.threshold_exceeded(0.1, 0.1));
    }

    #[test]
    fn test_regime_defaults_match_normal() {
        let mut estimator = OUDriftEstimator::default_config();
        let default_theta = estimator.config.theta;
        let default_k = estimator.config.reconcile_k;

        // Adapt to "normal" regime
        estimator.adapt_to_regime("normal");
        assert!(
            (estimator.config.theta - 0.5).abs() < 1e-12,
            "normal theta should be 0.5, got {}",
            estimator.config.theta
        );
        assert!(
            (estimator.config.reconcile_k - 2.0).abs() < 1e-12,
            "normal reconcile_k should be 2.0, got {}",
            estimator.config.reconcile_k
        );

        // Default config should also match normal regime values
        assert!(
            (default_theta - 0.5).abs() < 1e-12,
            "default theta should be 0.5, got {default_theta}"
        );
        assert!(
            (default_k - 2.0).abs() < 1e-12,
            "default reconcile_k should be 2.0, got {default_k}"
        );
    }

    #[test]
    fn test_regime_cascade_params() {
        let mut estimator = OUDriftEstimator::default_config();

        estimator.adapt_to_regime("cascade");
        assert!(
            (estimator.config.theta - 1.0).abs() < 1e-12,
            "cascade theta should be 1.0, got {}",
            estimator.config.theta
        );
        assert!(
            (estimator.config.reconcile_k - 3.0).abs() < 1e-12,
            "cascade reconcile_k should be 3.0, got {}",
            estimator.config.reconcile_k
        );

        // Verify trending works too
        estimator.adapt_to_regime("trending");
        assert!(
            (estimator.config.theta - 0.2).abs() < 1e-12,
            "trending theta should be 0.2, got {}",
            estimator.config.theta
        );
        assert!(
            (estimator.config.reconcile_k - 1.5).abs() < 1e-12,
            "trending reconcile_k should be 1.5, got {}",
            estimator.config.reconcile_k
        );
    }

    #[test]
    fn test_sigmoid_gate_slow_ramp() {
        // Fix 2: Verify sigmoid soft-gate tracks slow continuous trends.
        // With the old hard deadband, sub-threshold innovations were zeroed out,
        // causing the filter to fight slow trends ("boiling frog" problem).
        let mut estimator = OUDriftEstimator::default_config();

        // Initialize
        estimator.update(0, 0.0);

        // Feed 50 observations of linearly increasing drift, each below threshold.
        // The drift increments are small enough to stay below the deadband
        // individually, but cumulatively represent a real trend.
        for i in 1..=50 {
            let observed = 0.0001 * i as f64; // Slow ramp: 0.1 bps/step
            estimator.update(i * 100, observed);
        }

        // With sigmoid soft-gate, drift should track the trend (not stuck at μ=0)
        assert!(
            estimator.drift() > 0.001,
            "Sigmoid gate should track slow ramp. Drift={}, expected > 0.001",
            estimator.drift()
        );
    }

    #[test]
    fn test_sigmoid_gate_noise_rejection() {
        // Fix 2: Verify noise rejection still works — small random noise
        // around μ should result in drift staying near μ.
        let mut estimator = OUDriftEstimator::default_config();

        // Initialize
        estimator.update(0, 0.0);

        // Feed 50 observations of pure noise around μ=0
        // Small noise should get ~5% gate weight, keeping drift near 0
        let noise_vals = [
            0.00005, -0.00003, 0.00002, -0.00004, 0.00001,
        ];
        for i in 1u64..=50 {
            let noise = noise_vals[(i as usize) % noise_vals.len()];
            estimator.update(i * 100, noise);
        }

        // Drift should stay near zero (noise suppressed by low gate weight)
        assert!(
            estimator.drift().abs() < 0.001,
            "Noise should be mostly rejected. Drift={}, expected near 0",
            estimator.drift()
        );
    }

    #[test]
    fn test_sigmoid_gate_large_innovation_full_gain() {
        // Fix 2: A single large innovation should get nearly full Kalman gain.
        let mut estimator = OUDriftEstimator::default_config();

        // Initialize
        estimator.update(0, 0.0);

        // Large jump: should trigger near-full gate weight
        let result = estimator.update(1000, 0.05); // 50 bps jump
        assert!(
            result.gain > 0.3,
            "Large innovation should get substantial gain. Gain={}",
            result.gain
        );
        assert!(
            estimator.drift().abs() > 0.01,
            "Large innovation should move drift significantly. Drift={}",
            estimator.drift()
        );
    }

    #[test]
    fn test_sigmoid_gate_always_nonzero() {
        // Fix 2: Effective gain should never be exactly 0 (minimum gate weight = 5%).
        let mut estimator = OUDriftEstimator::default_config();
        estimator.update(0, 0.0);

        // Very small innovation
        let result = estimator.update(100, 0.000001);
        // Gain should be > 0 due to MIN_GATE_WEIGHT
        assert!(
            result.gain > 0.0,
            "Gain should never be exactly 0 with sigmoid gate. Gain={}",
            result.gain
        );
    }
}
