//! Kalman filtering and noise estimation components.
//!
//! - NoiseFilter: Roll model-based bid-ask bounce filtering
//! - KalmanPriceFilter: Latent true price estimation

use std::collections::VecDeque;

// ============================================================================
// Noise Filter (Roll Model)
// ============================================================================

/// Filters bid-ask bounce noise from price series.
///
/// Uses the Roll model: Cov(r_t, r_{t-1}) = -noise_var
/// to estimate microstructure noise and provide cleaner price signals.
#[derive(Debug)]
pub struct NoiseFilter {
    /// Estimated noise variance (from autocovariance)
    noise_variance: f64,
    /// Recent returns for autocovariance calculation
    returns: VecDeque<f64>,
    /// Maximum history size
    max_history: usize,
    /// EWMA alpha for noise estimation
    alpha: f64,
}

impl NoiseFilter {
    /// Create a new noise filter.
    pub fn new(max_history: usize, alpha: f64) -> Self {
        Self {
            noise_variance: 0.0,
            returns: VecDeque::with_capacity(max_history),
            max_history,
            alpha,
        }
    }

    /// Create with default parameters.
    pub fn default_config() -> Self {
        Self::new(100, 0.05)
    }

    /// Record a new return observation.
    pub fn on_return(&mut self, log_return: f64) {
        self.returns.push_back(log_return);

        // Trim to max size
        while self.returns.len() > self.max_history {
            self.returns.pop_front();
        }

        // Update noise variance from autocovariance
        if self.returns.len() >= 2 {
            self.update_noise_estimate();
        }
    }

    /// Update noise estimate using Roll model.
    ///
    /// Roll model: Cov(r_t, r_{t-1}) = -noise_var
    fn update_noise_estimate(&mut self) {
        if self.returns.len() < 2 {
            return;
        }

        // Calculate lag-1 autocovariance
        let n = self.returns.len();
        let returns_vec: Vec<f64> = self.returns.iter().cloned().collect();

        let mean = returns_vec.iter().sum::<f64>() / n as f64;

        let mut cov = 0.0;
        for i in 1..n {
            cov += (returns_vec[i] - mean) * (returns_vec[i - 1] - mean);
        }
        cov /= (n - 1) as f64;

        // Roll model: noise_var = -Cov(r_t, r_{t-1})
        // Only update if covariance is negative (expected from bid-ask bounce)
        if cov < 0.0 {
            let new_estimate = -cov;
            self.noise_variance =
                self.alpha * new_estimate + (1.0 - self.alpha) * self.noise_variance;
        }
    }

    /// Get estimated noise standard deviation.
    pub fn noise_sigma(&self) -> f64 {
        self.noise_variance.sqrt()
    }

    /// Get estimated noise variance.
    pub fn noise_variance(&self) -> f64 {
        self.noise_variance
    }

    /// Clean a return by removing estimated noise component.
    ///
    /// Returns a filtered return with reduced microstructure noise.
    pub fn filter_return(&self, raw_return: f64) -> f64 {
        if self.noise_variance < 1e-20 {
            return raw_return;
        }

        // Simple shrinkage toward 0 based on signal-to-noise ratio
        let total_var = raw_return.powi(2);
        let signal_var = (total_var - self.noise_variance).max(0.0);

        if total_var < 1e-20 {
            return 0.0;
        }

        let shrinkage = signal_var / total_var;
        raw_return * shrinkage.sqrt()
    }

    /// Check if filter is warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.returns.len() >= 20
    }
}

// ============================================================================
// Kalman Filter for Latent True Price (Phase 3)
// ============================================================================

/// Kalman filter for estimating the latent "true" price from noisy observations.
///
/// State-space model:
/// - State equation: x_t = x_{t-1} + σ_true × ε_t (random walk)
/// - Observation: y_t = x_t + η_t (observed mid with bid-ask bounce noise)
///
/// Posterior: x_t | y_{1:t} ~ N(μ_t, Σ_t)
///
/// This provides:
/// 1. Filtered estimate of true price (μ)
/// 2. Uncertainty measure (σ = √Σ) for adaptive spread widening
/// 3. Signal smoothing that filters bid-ask bounce
///
/// Theory from the manual:
/// - Process noise Q comes from actual volatility
/// - Observation noise R comes from bid-ask bounce (~25% of spread typical)
/// - When uncertain (high Σ), widen spreads
#[derive(Debug, Clone)]
pub struct KalmanPriceFilter {
    /// Posterior mean of true price
    mu: f64,
    /// Posterior variance
    sigma_sq: f64,
    /// Process noise variance (true price volatility per tick)
    q: f64,
    /// Observation noise variance (bid-ask bounce)
    r: f64,
    /// EWMA smoothing factor for Q estimation
    q_alpha: f64,
    /// Estimated process noise from price changes
    q_estimate: f64,
    /// Count of updates for warmup
    update_count: usize,
    /// Last observation for noise estimation
    last_observation: Option<f64>,
}

impl KalmanPriceFilter {
    /// Create a new Kalman filter.
    ///
    /// # Arguments
    /// * `initial_price` - Initial price estimate
    /// * `initial_variance` - Initial uncertainty (σ²)
    /// * `process_noise` - Process noise Q (volatility per tick)
    /// * `observation_noise` - Observation noise R (bid-ask bounce)
    pub fn new(
        initial_price: f64,
        initial_variance: f64,
        process_noise: f64,
        observation_noise: f64,
    ) -> Self {
        Self {
            mu: initial_price,
            sigma_sq: initial_variance,
            q: process_noise,
            r: observation_noise,
            q_alpha: 0.05, // 20-tick half-life for Q estimation
            q_estimate: process_noise,
            update_count: 0,
            last_observation: None,
        }
    }

    /// Create with sensible defaults for crypto.
    ///
    /// Uses:
    /// - Q = (0.0001)² = 1 bp per tick variance
    /// - R = (0.00005)² = 0.5 bp observation noise
    pub fn default_crypto() -> Self {
        Self::new(
            0.0,    // Will be set on first observation
            1e-6,   // Initial σ² = 10 bp²
            1e-8,   // Q = 1 bp² per tick
            2.5e-9, // R = 0.5 bp² (bid-ask bounce)
        )
    }

    /// Predict step: propagate state forward in time.
    ///
    /// μ_{t|t-1} = μ_{t-1} (random walk)
    /// Σ_{t|t-1} = Σ_{t-1} + Q (uncertainty grows)
    pub fn predict(&mut self) {
        // Mean stays same (random walk has no drift)
        // Variance grows by process noise
        self.sigma_sq += self.q;
    }

    /// Update step: incorporate new observation.
    ///
    /// K = Σ_{t|t-1} / (Σ_{t|t-1} + R) (Kalman gain)
    /// μ_t = μ_{t|t-1} + K × (y_t - μ_{t|t-1}) (update mean)
    /// Σ_t = (1 - K) × Σ_{t|t-1} (update variance)
    pub fn update(&mut self, observation: f64) {
        // Handle first observation specially
        if self.update_count == 0 {
            self.mu = observation;
            self.last_observation = Some(observation);
            self.update_count = 1;
            return;
        }

        // Kalman gain: K = Σ / (Σ + R)
        let k = self.sigma_sq / (self.sigma_sq + self.r);

        // Innovation (measurement residual)
        let innovation = observation - self.mu;

        // State update: μ = μ + K × innovation
        self.mu += k * innovation;

        // Variance update: Σ = (1 - K) × Σ
        self.sigma_sq *= 1.0 - k;

        // Adaptive Q estimation from squared innovations
        if let Some(last) = self.last_observation {
            let price_change = (observation - last).powi(2);
            self.q_estimate = self.q_alpha * price_change + (1.0 - self.q_alpha) * self.q_estimate;

            // Slowly adapt Q to observed volatility
            if self.update_count > 20 {
                self.q = 0.9 * self.q + 0.1 * self.q_estimate;
            }
        }

        self.last_observation = Some(observation);
        self.update_count += 1;
    }

    /// Combined predict + update for time-series filtering.
    pub fn filter(&mut self, observation: f64) {
        self.predict();
        self.update(observation);
    }

    /// Get fair price estimate (posterior mean).
    pub fn fair_price(&self) -> f64 {
        self.mu
    }

    /// Get uncertainty (posterior standard deviation).
    pub fn uncertainty(&self) -> f64 {
        self.sigma_sq.sqrt()
    }

    /// Get uncertainty in basis points (relative to price).
    pub fn uncertainty_bps(&self) -> f64 {
        if self.mu.abs() > 1e-10 {
            (self.sigma_sq.sqrt() / self.mu) * 10000.0
        } else {
            0.0
        }
    }

    /// Get fair price with uncertainty bounds.
    ///
    /// Returns (fair_price, uncertainty) where uncertainty is σ (std dev).
    pub fn fair_price_with_uncertainty(&self) -> (f64, f64) {
        (self.mu, self.sigma_sq.sqrt())
    }

    /// Compute recommended spread widening from uncertainty.
    ///
    /// Uses: spread_add = γ × σ × √(time_horizon)
    /// Higher uncertainty → wider spreads
    pub fn uncertainty_spread(&self, gamma: f64, time_horizon: f64) -> f64 {
        gamma * self.sigma_sq.sqrt() * time_horizon.sqrt()
    }

    /// Check if filter is warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.update_count >= 10
    }

    /// Get update count.
    pub fn update_count(&self) -> usize {
        self.update_count
    }

    /// Get estimated process noise Q.
    pub fn estimated_q(&self) -> f64 {
        self.q_estimate
    }

    /// Get current Kalman gain (for diagnostics).
    pub fn current_kalman_gain(&self) -> f64 {
        self.sigma_sq / (self.sigma_sq + self.r)
    }

    /// Reset filter with new initial conditions.
    pub fn reset(&mut self, initial_price: f64, initial_variance: f64) {
        self.mu = initial_price;
        self.sigma_sq = initial_variance;
        self.update_count = 0;
        self.last_observation = None;
    }

    /// Set process noise Q from external volatility estimate.
    ///
    /// # Arguments
    /// * `sigma` - Per-second volatility (e.g., from bipower variation)
    /// * `dt_seconds` - Time step in seconds (typical: 0.5-2.0 for volume ticks)
    /// * `activity_mult` - Hawkes intensity ratio (current/baseline), clamped [0.5, 5.0].
    ///   During toxic sweeps (100 trades/sec), activity_mult >> 1 → larger Q → faster adaptation.
    ///   During quiet periods, activity_mult < 1 → smaller Q → more stable estimates.
    ///
    /// Scales Q as sigma^2 * dt * activity_mult to match actual market volatility
    /// and activity level. This provides faster adaptation than learning Q from innovations.
    pub fn set_process_noise_from_volatility(
        &mut self,
        sigma: f64,
        dt_seconds: f64,
        activity_mult: f64,
    ) {
        // Q = sigma^2 * dt * activity_mult
        let clamped_mult = activity_mult.clamp(0.5, 5.0);
        let new_q = sigma.powi(2) * dt_seconds * clamped_mult;

        // Use faster adaptation during warmup, slower after
        let alpha = if self.update_count < 20 { 0.3 } else { 0.05 };

        self.q = alpha * new_q + (1.0 - alpha) * self.q;
        self.q_estimate = self.q; // Keep estimate in sync
    }

    /// Get current process noise Q.
    pub fn process_noise(&self) -> f64 {
        self.q
    }

    /// Get observation noise R.
    pub fn observation_noise(&self) -> f64 {
        self.r
    }
}
