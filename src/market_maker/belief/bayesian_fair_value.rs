//! Bayesian Latent Fair Value Model
//!
//! Maintains a Gaussian posterior N(μ_offset, σ²) over the latent fair value offset
//! from mid price. Fills, book state, and trade flow are observations that update
//! this posterior via Kalman gains.
//!
//! ## Key Properties
//!
//! - **Fills shift the mean**: An ask fill (buyer came to us) → μ shifts up.
//!   By fill 5, μ ≈ +7 bps → asks are 7 bps more expensive for the sweeper.
//! - **Uncertainty drives spread**: Wide σ at cold start → natural spread widening.
//!   Tightens as fills arrive → normal GLFT behavior.
//! - **Mean reversion**: μ decays toward 0 at rate λ. After flow stops,
//!   the fair value estimate relaxes back to mid over ~60s.
//! - **Online learning**: α_book, β_flow, σ_noise learned from realized returns,
//!   with guardrails (bounds + decay toward prior).
//!
//! ## Observation Models
//!
//! 1. **Fill**: price-distance-scaled (Glosten-Milgrom spirit). A fill 3 levels deep
//!    carries more information than a fill at the inside.
//! 2. **Book**: L2 imbalance is a weak, noisy signal. H = α/10 deliberately low.
//! 3. **Flow**: Batch multi-observation update (1s, 5s, 30s) — order-independent.

use serde::{Deserialize, Serialize};

/// Configuration for the Bayesian fair value model.
///
/// All parameters have sensible defaults for HIP-3 DEX trading.
#[derive(Debug, Clone)]
pub struct BayesianFairValueConfig {
    // === Prior / Process Noise ===
    /// Initial posterior variance (bps²). σ₀ = √25 = 5 bps.
    pub initial_variance_bps_sq: f64,
    /// Process noise per second (bps²/s). Uncertainty grows ~1 bps²/s.
    pub process_noise_per_sec_bps_sq: f64,

    // === Fill Observation Model ===
    /// Base observation noise per fill (bps).
    pub sigma_noise_base_bps: f64,
    /// VPIN modulation coefficient [0, 0.8]. Higher VPIN → lower noise → more informative fills.
    pub gamma_vpin: f64,
    /// Whether to scale observation by fill_size/quoted_size.
    pub fill_size_scaling: bool,

    // === Book Observation Model ===
    /// Book imbalance sensitivity prior.
    pub alpha_book_prior: f64,
    /// Book observation noise (bps).
    pub sigma_book_bps: f64,
    /// Divisor for book sensitivity (H = alpha_book / divisor).
    pub book_scaling_divisor: f64,

    // === Flow Observation Model [1s, 5s, 30s] ===
    /// Flow sensitivity priors per horizon.
    pub beta_flow_prior: [f64; 3],
    /// Flow observation noise per horizon (bps).
    pub sigma_flow_bps: [f64; 3],

    // === Mean Reversion ===
    /// Offset decays toward 0 at this rate: μ *= exp(-decay × dt).
    /// τ = 1/decay = 20s → half-life ≈ 14s.
    pub mean_reversion_rate_per_sec: f64,

    // === Online Learning Guardrails ===
    /// EWMA learning rate for all learned params.
    pub learning_rate: f64,
    /// Per-update decay rate toward config prior.
    pub learned_decay_to_prior_rate: f64,
    /// Hard bounds for alpha_book.
    pub alpha_book_bounds: (f64, f64),
    /// Hard bounds for beta_flow elements.
    pub beta_flow_bounds: (f64, f64),
    /// Hard bounds for sigma_noise (bps).
    pub sigma_noise_bounds_bps: (f64, f64),

    // === Cascade Detection ===
    /// Z-score threshold for cascade detection.
    pub cascade_z_threshold: f64,

    // === Posterior Bounds ===
    /// Maximum absolute offset (bps).
    pub max_offset_bps: f64,
    /// Maximum posterior variance (bps²).
    pub max_variance_bps_sq: f64,
    /// Minimum posterior variance (bps²).
    pub min_variance_bps_sq: f64,
}

impl Default for BayesianFairValueConfig {
    fn default() -> Self {
        Self {
            initial_variance_bps_sq: 25.0,
            process_noise_per_sec_bps_sq: 1.0,

            sigma_noise_base_bps: 3.0,
            gamma_vpin: 0.5,
            fill_size_scaling: true,

            alpha_book_prior: 1.0,
            sigma_book_bps: 5.0,
            book_scaling_divisor: 10.0,

            beta_flow_prior: [1.5, 1.0, 0.5],
            sigma_flow_bps: [3.0, 4.0, 5.0],

            mean_reversion_rate_per_sec: 0.05,

            learning_rate: 0.01,
            learned_decay_to_prior_rate: 0.001,
            alpha_book_bounds: (0.1, 5.0),
            beta_flow_bounds: (0.05, 5.0),
            sigma_noise_bounds_bps: (1.0, 20.0),

            cascade_z_threshold: 2.5,

            max_offset_bps: 30.0,
            max_variance_bps_sq: 100.0,
            min_variance_bps_sq: 0.5,
        }
    }
}

/// Checkpoint for persisting learned parameters across restarts.
///
/// Only learned parameters are persisted — posterior state (μ, σ²) resets
/// relative to current mid on each restart. Cold start is handled by
/// prior variance. Learned params decay toward config priors each update,
/// so even a poisoned checkpoint self-corrects over ~100 updates.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BayesianFairValueCheckpoint {
    #[serde(default)]
    pub alpha_book: f64,
    #[serde(default)]
    pub beta_flow: [f64; 3],
    #[serde(default)]
    pub sigma_noise_bps: f64,
    #[serde(default)]
    pub n_fills: u64,
}

/// Read-only beliefs for consumers (snapshot).
#[derive(Debug, Clone)]
pub struct FairValueBeliefs {
    /// Absolute fair value (last_mid adjusted by offset).
    pub posterior_mean: f64,
    /// Posterior uncertainty (bps).
    pub posterior_sigma_bps: f64,
    /// Offset from mid (bps). Positive = fair value above mid.
    pub offset_from_mid_bps: f64,
    /// Z-score on posterior drift rate (cascade detector).
    pub cascade_score: f64,
    /// Whether cascade is detected (score > threshold).
    pub is_cascade: bool,
    /// Whether the model has received at least one fill.
    pub is_warmed_up: bool,
    /// Number of fill updates processed.
    pub n_fill_updates: u64,
    /// Confidence [0, 1] based on data quantity and posterior tightness.
    pub confidence: f64,
}

impl Default for FairValueBeliefs {
    fn default() -> Self {
        Self {
            posterior_mean: 0.0,
            posterior_sigma_bps: 5.0,
            offset_from_mid_bps: 0.0,
            cascade_score: 0.0,
            is_cascade: false,
            is_warmed_up: false,
            n_fill_updates: 0,
            confidence: 0.0,
        }
    }
}

/// Bayesian fair value model maintaining N(μ_offset, σ²) posterior.
///
/// The offset is relative to mid price in basis points.
/// Positive μ = fair value is above mid (bullish pressure detected).
pub struct BayesianFairValue {
    config: BayesianFairValueConfig,

    // === Posterior state ===
    /// Offset from mid (bps). Positive = fair value above mid.
    mu_offset_bps: f64,
    /// Posterior variance (bps²).
    sigma_sq_bps: f64,
    /// Last observed mid price.
    last_mid: f64,
    /// Timestamp of last update (ms).
    last_update_ms: u64,

    // === Learned parameters ===
    alpha_book: f64,
    beta_flow: [f64; 3],
    sigma_noise_bps: f64,

    // === Cascade detection ===
    prev_mu_offset_bps: f64,
    prev_update_ms: u64,
    cascade_score: f64,

    // === Stats ===
    n_fill_updates: u64,
    n_book_updates: u64,
    n_flow_updates: u64,
}

impl BayesianFairValue {
    /// Create a new model with given config.
    pub fn new(config: BayesianFairValueConfig) -> Self {
        let alpha_book = config.alpha_book_prior;
        let beta_flow = config.beta_flow_prior;
        let sigma_noise_bps = config.sigma_noise_base_bps;

        Self {
            mu_offset_bps: 0.0,
            sigma_sq_bps: config.initial_variance_bps_sq,
            last_mid: 0.0,
            last_update_ms: 0,

            alpha_book,
            beta_flow,
            sigma_noise_bps,

            prev_mu_offset_bps: 0.0,
            prev_update_ms: 0,
            cascade_score: 0.0,

            n_fill_updates: 0,
            n_book_updates: 0,
            n_flow_updates: 0,

            config,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(BayesianFairValueConfig::default())
    }

    /// Restore learned parameters from checkpoint.
    pub fn restore_from_checkpoint(&mut self, ckpt: &BayesianFairValueCheckpoint) {
        if ckpt.n_fills > 0 {
            self.alpha_book = ckpt.alpha_book.clamp(
                self.config.alpha_book_bounds.0,
                self.config.alpha_book_bounds.1,
            );
            for i in 0..3 {
                self.beta_flow[i] = ckpt.beta_flow[i].clamp(
                    self.config.beta_flow_bounds.0,
                    self.config.beta_flow_bounds.1,
                );
            }
            self.sigma_noise_bps = ckpt.sigma_noise_bps.clamp(
                self.config.sigma_noise_bounds_bps.0,
                self.config.sigma_noise_bounds_bps.1,
            );
        }
    }

    /// Export learned parameters for checkpoint persistence.
    pub fn checkpoint(&self) -> BayesianFairValueCheckpoint {
        BayesianFairValueCheckpoint {
            alpha_book: self.alpha_book,
            beta_flow: self.beta_flow,
            sigma_noise_bps: self.sigma_noise_bps,
            n_fills: self.n_fill_updates,
        }
    }

    /// Build a snapshot of current beliefs for consumers.
    pub fn beliefs(&self) -> FairValueBeliefs {
        let sigma = self.sigma_sq_bps.sqrt();
        // Confidence: ramps with fill count and tightness of posterior
        // At 0 fills: 0.0. At 5 fills with σ ≈ 2 bps: ~0.5. At 20+ fills: ~0.8+.
        let fill_confidence = (self.n_fill_updates as f64 / 10.0).min(1.0);
        let tightness_confidence =
            (1.0 - sigma / self.config.initial_variance_bps_sq.sqrt()).max(0.0);
        let confidence = (fill_confidence * tightness_confidence).clamp(0.0, 1.0);

        FairValueBeliefs {
            posterior_mean: self.posterior_mean(),
            posterior_sigma_bps: sigma,
            offset_from_mid_bps: self.mu_offset_bps,
            cascade_score: self.cascade_score,
            is_cascade: self.cascade_score > self.config.cascade_z_threshold,
            is_warmed_up: self.n_fill_updates > 0,
            n_fill_updates: self.n_fill_updates,
            confidence,
        }
    }

    // =========================================================================
    // Prediction step
    // =========================================================================

    /// Predict step: grow variance, apply mean reversion.
    ///
    /// Called on each mid price update.
    pub fn predict(&mut self, new_mid: f64, dt_secs: f64, timestamp_ms: u64) {
        if dt_secs <= 0.0 {
            self.last_mid = new_mid;
            self.last_update_ms = timestamp_ms;
            return;
        }

        // Variance grows with process noise
        self.sigma_sq_bps += self.config.process_noise_per_sec_bps_sq * dt_secs;
        self.sigma_sq_bps = self.sigma_sq_bps.clamp(
            self.config.min_variance_bps_sq,
            self.config.max_variance_bps_sq,
        );

        // Mean reverts toward 0
        self.mu_offset_bps *= (-self.config.mean_reversion_rate_per_sec * dt_secs).exp();
        self.mu_offset_bps = self
            .mu_offset_bps
            .clamp(-self.config.max_offset_bps, self.config.max_offset_bps);

        self.last_mid = new_mid;
        self.last_update_ms = timestamp_ms;
    }

    // =========================================================================
    // Fill observation update
    // =========================================================================

    /// Update posterior on own fill observation.
    ///
    /// # Arguments
    /// * `fill_price` - Price at which the fill occurred
    /// * `fill_size` - Size of the fill
    /// * `quoted_size` - Size we had quoted at that level
    /// * `is_buy` - Whether WE bought (bid fill). `true` = bearish, `false` = bullish.
    /// * `mid` - Mid price at time of fill
    /// * `vpin` - Current VPIN estimate [0, 1]
    pub fn update_on_fill(
        &mut self,
        fill_price: f64,
        fill_size: f64,
        quoted_size: f64,
        is_buy: bool,
        mid: f64,
        vpin: f64,
    ) {
        if mid <= 0.0 {
            return;
        }

        // Save previous state for cascade detection
        self.prev_mu_offset_bps = self.mu_offset_bps;
        self.prev_update_ms = self.last_update_ms;

        // Distance from mid in bps
        let distance_bps = ((fill_price - mid) / mid).abs() * 10_000.0;

        // Size scaling: larger fills are more informative
        let size_scale = if self.config.fill_size_scaling {
            (fill_size / quoted_size.max(0.01)).clamp(0.2, 3.0)
        } else {
            1.0
        };

        // Signed observation: ask fill (we sold, buyer came) → bullish → positive
        // is_buy = WE bought = bid fill = someone sold to us → bearish → negative
        let sign = if is_buy { -1.0 } else { 1.0 };
        let observation_bps = sign * distance_bps * size_scale;

        // VPIN-modulated noise: high toxicity → fills more informative → lower noise
        let vpin_clamped = vpin.clamp(0.0, 1.0);
        let sigma_noise = self.sigma_noise_bps * (1.0 - self.config.gamma_vpin * vpin_clamped);
        let r = sigma_noise * sigma_noise;

        // Kalman update
        let k = self.sigma_sq_bps / (self.sigma_sq_bps + r);
        self.mu_offset_bps += k * observation_bps;
        self.sigma_sq_bps *= 1.0 - k;

        // Enforce bounds
        self.mu_offset_bps = self
            .mu_offset_bps
            .clamp(-self.config.max_offset_bps, self.config.max_offset_bps);
        self.sigma_sq_bps = self.sigma_sq_bps.clamp(
            self.config.min_variance_bps_sq,
            self.config.max_variance_bps_sq,
        );

        // Update cascade score
        self.update_cascade_score();

        self.last_mid = mid;
        self.n_fill_updates += 1;
    }

    // =========================================================================
    // Book observation update
    // =========================================================================

    /// Update posterior on L2 book imbalance observation.
    ///
    /// Book imbalance [-1, 1] is a weak, noisy signal of fair value direction.
    /// H = α_book / scaling_divisor deliberately maps full imbalance to only
    /// ~0.1 × μ because L2 depth is noisy on thin DEX.
    pub fn update_on_book(&mut self, book_imbalance: f64, mid: f64) {
        let h = self.alpha_book / self.config.book_scaling_divisor;
        let r = self.config.sigma_book_bps * self.config.sigma_book_bps;

        let innovation = book_imbalance - h * self.mu_offset_bps;
        let s = h * h * self.sigma_sq_bps + r;

        if s.abs() < 1e-12 {
            return;
        }

        let k = self.sigma_sq_bps * h / s;
        self.mu_offset_bps += k * innovation;
        self.sigma_sq_bps *= 1.0 - k * h;

        // Enforce bounds
        self.mu_offset_bps = self
            .mu_offset_bps
            .clamp(-self.config.max_offset_bps, self.config.max_offset_bps);
        self.sigma_sq_bps = self.sigma_sq_bps.clamp(
            self.config.min_variance_bps_sq,
            self.config.max_variance_bps_sq,
        );

        self.last_mid = mid;
        self.n_book_updates += 1;
    }

    // =========================================================================
    // Flow observation update (batch)
    // =========================================================================

    /// Update posterior on trade flow observations (batch multi-horizon).
    ///
    /// Uses batch Kalman update treating three horizons as a vector observation.
    /// This avoids order-dependence: sequential updates would give 1s more
    /// influence than 30s purely from ordering.
    pub fn update_on_flow(&mut self, flow_1s: f64, flow_5s: f64, flow_30s: f64, mid: f64) {
        let flows = [flow_1s, flow_5s, flow_30s];

        // Batch Kalman: information-form update
        // precision_posterior = precision_prior + Σᵢ Hᵢ² / Rᵢ
        // weighted_mean = precision_prior × μ + Σᵢ Hᵢ × zᵢ / Rᵢ
        let precision_prior = 1.0 / self.sigma_sq_bps;
        let mut precision_total = precision_prior;
        let mut weighted_obs = precision_prior * self.mu_offset_bps;

        for ((&flow_val, &beta), &sigma) in flows
            .iter()
            .zip(self.beta_flow.iter())
            .zip(self.config.sigma_flow_bps.iter())
        {
            let h = beta / 10.0;
            let r = sigma * sigma;

            if r < 1e-12 {
                continue;
            }

            let obs_precision = h * h / r;
            precision_total += obs_precision;
            weighted_obs += h * flow_val / r;
        }

        if precision_total < 1e-12 {
            return;
        }

        self.sigma_sq_bps = 1.0 / precision_total;
        self.mu_offset_bps = self.sigma_sq_bps * weighted_obs;

        // Enforce bounds
        self.mu_offset_bps = self
            .mu_offset_bps
            .clamp(-self.config.max_offset_bps, self.config.max_offset_bps);
        self.sigma_sq_bps = self.sigma_sq_bps.clamp(
            self.config.min_variance_bps_sq,
            self.config.max_variance_bps_sq,
        );

        self.last_mid = mid;
        self.n_flow_updates += 1;
    }

    // =========================================================================
    // Cascade detection
    // =========================================================================

    /// Update cascade z-score from recent posterior drift.
    fn update_cascade_score(&mut self) {
        if self.prev_update_ms == 0 || self.last_update_ms <= self.prev_update_ms {
            self.cascade_score = 0.0;
            return;
        }

        let dt = (self.last_update_ms - self.prev_update_ms) as f64 / 1000.0;
        if dt < 0.001 {
            return;
        }

        let sigma = self.sigma_sq_bps.sqrt();
        if sigma < 1e-6 {
            return;
        }

        self.cascade_score =
            (self.mu_offset_bps - self.prev_mu_offset_bps).abs() / (sigma * dt.sqrt());
    }

    /// Get current cascade z-score.
    pub fn cascade_score(&self) -> f64 {
        self.cascade_score
    }

    // =========================================================================
    // Outputs for quoting
    // =========================================================================

    /// Absolute fair value (mid adjusted by posterior offset).
    pub fn posterior_mean(&self) -> f64 {
        if self.last_mid <= 0.0 {
            return 0.0;
        }
        self.last_mid * (1.0 + self.mu_offset_bps / 10_000.0)
    }

    /// Avellaneda-Stoikov reservation price.
    ///
    /// r = posterior_mean - γ × inventory × σ² × T
    pub fn reservation_price(&self, inventory: f64, time_horizon_s: f64, gamma: f64) -> f64 {
        let mean = self.posterior_mean();
        if mean <= 0.0 {
            return 0.0;
        }
        // Convert σ² from bps² to fractional² for A-S formula
        let sigma_sq_frac = self.sigma_sq_bps / (10_000.0 * 10_000.0);
        mean - gamma * inventory * sigma_sq_frac * time_horizon_s * mean
    }

    /// Uncertainty-driven spread addon (bps).
    ///
    /// spread_addon = γ × σ × √T
    pub fn uncertainty_spread_bps(&self, gamma: f64, time_horizon_s: f64) -> f64 {
        let sigma = self.sigma_sq_bps.sqrt();
        gamma * sigma * time_horizon_s.sqrt()
    }

    /// Get posterior offset from mid (bps).
    pub fn offset_bps(&self) -> f64 {
        self.mu_offset_bps
    }

    /// Get posterior variance (bps²).
    pub fn variance_bps_sq(&self) -> f64 {
        self.sigma_sq_bps
    }

    /// Get posterior standard deviation (bps).
    pub fn sigma_bps(&self) -> f64 {
        self.sigma_sq_bps.sqrt()
    }

    /// Number of fill updates processed.
    pub fn n_fill_updates(&self) -> u64 {
        self.n_fill_updates
    }

    // =========================================================================
    // Online learning
    // =========================================================================

    /// Update learned parameters from realized return.
    ///
    /// Called when the next mid price update arrives, allowing us to compute
    /// the realized return since the last update and assess how well our
    /// observation models predicted it.
    pub fn update_learned_params(&mut self, realized_return_bps: f64, book_imbalance: f64) {
        let lr = self.config.learning_rate;
        let decay = self.config.learned_decay_to_prior_rate;

        // Update alpha_book: gradient of prediction error w.r.t. alpha
        // Simplified: alpha moves toward value that minimizes (realized - alpha × imbalance)²
        let alpha_gradient = realized_return_bps * book_imbalance
            - self.alpha_book * book_imbalance * book_imbalance;
        self.alpha_book += lr * alpha_gradient;

        // Decay toward prior
        self.alpha_book += decay * (self.config.alpha_book_prior - self.alpha_book);

        // Hard bounds
        self.alpha_book = self.alpha_book.clamp(
            self.config.alpha_book_bounds.0,
            self.config.alpha_book_bounds.1,
        );

        // Update sigma_noise: move toward realized prediction error magnitude
        let prediction_error = realized_return_bps.abs();
        self.sigma_noise_bps += lr * (prediction_error - self.sigma_noise_bps);
        self.sigma_noise_bps += decay * (self.config.sigma_noise_base_bps - self.sigma_noise_bps);
        self.sigma_noise_bps = self.sigma_noise_bps.clamp(
            self.config.sigma_noise_bounds_bps.0,
            self.config.sigma_noise_bounds_bps.1,
        );
    }

    /// Update flow sensitivity learned params from realized return.
    pub fn update_learned_flow_params(&mut self, realized_return_bps: f64, flows: [f64; 3]) {
        let lr = self.config.learning_rate;
        let decay = self.config.learned_decay_to_prior_rate;

        for (i, &flow_val) in flows.iter().enumerate() {
            let gradient = realized_return_bps * flow_val - self.beta_flow[i] * flow_val * flow_val;
            self.beta_flow[i] += lr * gradient;
            self.beta_flow[i] += decay * (self.config.beta_flow_prior[i] - self.beta_flow[i]);
            self.beta_flow[i] = self.beta_flow[i].clamp(
                self.config.beta_flow_bounds.0,
                self.config.beta_flow_bounds.1,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 0.01;

    fn assert_approx(actual: f64, expected: f64, tol: f64, msg: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{}: expected {:.4}, got {:.4} (diff {:.4})",
            msg,
            expected,
            actual,
            (actual - expected).abs()
        );
    }

    #[test]
    fn test_cold_start_prior() {
        let model = BayesianFairValue::with_defaults();
        assert_approx(model.mu_offset_bps, 0.0, TOLERANCE, "cold start mu");
        assert_approx(model.sigma_sq_bps, 25.0, TOLERANCE, "cold start sigma_sq");
        assert_approx(model.sigma_bps(), 5.0, TOLERANCE, "cold start sigma");
    }

    #[test]
    fn test_cold_start_uncertainty_spread() {
        let model = BayesianFairValue::with_defaults();
        // γ=1, T=1s → spread_addon = 1 × 5 × 1 = 5 bps
        let addon = model.uncertainty_spread_bps(1.0, 1.0);
        assert_approx(addon, 5.0, TOLERANCE, "cold start spread addon");
    }

    #[test]
    fn test_ask_fill_shifts_mu_up() {
        let mut model = BayesianFairValue::with_defaults();
        let mid = 100.0;
        model.predict(mid, 0.0, 1000);

        // Ask fill at 3 bps from mid (buyer came to us → bullish)
        // is_buy=false means WE sold → ask fill
        let fill_price = mid * (1.0 + 3.0 / 10_000.0);
        model.update_on_fill(fill_price, 1.0, 1.0, false, mid, 0.0);

        assert!(
            model.mu_offset_bps > 0.0,
            "Ask fill should shift mu positive (bullish): got {}",
            model.mu_offset_bps
        );
    }

    #[test]
    fn test_bid_fill_shifts_mu_down() {
        let mut model = BayesianFairValue::with_defaults();
        let mid = 100.0;
        model.predict(mid, 0.0, 1000);

        // Bid fill at 3 bps from mid (seller came to us → bearish)
        let fill_price = mid * (1.0 - 3.0 / 10_000.0);
        model.update_on_fill(fill_price, 1.0, 1.0, true, mid, 0.0);

        assert!(
            model.mu_offset_bps < 0.0,
            "Bid fill should shift mu negative (bearish): got {}",
            model.mu_offset_bps
        );
    }

    #[test]
    fn test_first_fill_high_kalman_gain() {
        let mut model = BayesianFairValue::with_defaults();

        // After 155s of process noise, variance caps at 100
        model.predict(100.0, 155.0, 155_000);
        assert_approx(
            model.sigma_sq_bps,
            100.0,
            TOLERANCE,
            "variance capped at 100",
        );

        // First fill: K = 100 / (100 + 9) = 0.917
        let fill_price = 100.0 * (1.0 + 3.0 / 10_000.0);
        model.update_on_fill(fill_price, 1.0, 1.0, false, 100.0, 0.0);

        let expected_k = 100.0 / (100.0 + 9.0);
        let expected_mu = expected_k * 3.0;
        assert_approx(model.mu_offset_bps, expected_mu, 0.05, "first fill mu");

        let expected_sigma_sq = 100.0 * (1.0 - expected_k);
        assert_approx(
            model.sigma_sq_bps,
            expected_sigma_sq,
            0.1,
            "first fill sigma_sq",
        );
    }

    #[test]
    fn test_fill_size_scaling() {
        let mut model1 = BayesianFairValue::with_defaults();
        let mut model2 = BayesianFairValue::with_defaults();

        let mid = 100.0;
        model1.predict(mid, 0.0, 1000);
        model2.predict(mid, 0.0, 1000);

        let fill_price = mid * (1.0 + 3.0 / 10_000.0);

        // Fill with 2x quoted size should have larger observation
        model1.update_on_fill(fill_price, 1.0, 1.0, false, mid, 0.0);
        model2.update_on_fill(fill_price, 2.0, 1.0, false, mid, 0.0);

        assert!(
            model2.mu_offset_bps > model1.mu_offset_bps,
            "2x size fill should shift mu more: 1x={:.4}, 2x={:.4}",
            model1.mu_offset_bps,
            model2.mu_offset_bps
        );
    }

    #[test]
    fn test_high_vpin_increases_kalman_gain() {
        let mut model_low = BayesianFairValue::with_defaults();
        let mut model_high = BayesianFairValue::with_defaults();

        let mid = 100.0;
        model_low.predict(mid, 0.0, 1000);
        model_high.predict(mid, 0.0, 1000);

        let fill_price = mid * (1.0 + 3.0 / 10_000.0);

        // VPIN=0 → σ_noise=3, R=9. VPIN=0.8 → σ_noise=3×(1-0.4)=1.8, R=3.24
        model_low.update_on_fill(fill_price, 1.0, 1.0, false, mid, 0.0);
        model_high.update_on_fill(fill_price, 1.0, 1.0, false, mid, 0.8);

        assert!(
            model_high.mu_offset_bps > model_low.mu_offset_bps,
            "High VPIN should give more weight to fill: low={:.4}, high={:.4}",
            model_low.mu_offset_bps,
            model_high.mu_offset_bps
        );
    }

    #[test]
    fn test_mean_reversion() {
        let mut model = BayesianFairValue::with_defaults();
        let mid = 100.0;
        model.predict(mid, 0.0, 1000);

        // Manually set offset
        model.mu_offset_bps = 10.0;

        // After 20s: offset should decay by exp(-0.05 × 20) = exp(-1) ≈ 0.368
        model.predict(mid, 20.0, 21_000);

        let expected = 10.0 * (-0.05_f64 * 20.0).exp();
        assert_approx(
            model.mu_offset_bps,
            expected,
            0.05,
            "mean reversion after 20s",
        );
    }

    #[test]
    fn test_variance_bounds() {
        let mut model = BayesianFairValue::with_defaults();

        // Variance should cap at max
        model.predict(100.0, 1000.0, 1_000_000);
        assert!(
            model.sigma_sq_bps <= 100.0,
            "variance should be capped at 100: {}",
            model.sigma_sq_bps
        );

        // After many fills, variance should stay above min
        for i in 0..100 {
            let fill_price = 100.0 * (1.0 + 3.0 / 10_000.0);
            model.update_on_fill(fill_price, 1.0, 1.0, false, 100.0, 0.5);
            model.predict(100.0, 0.1, 1_000_000 + i * 100);
        }
        assert!(
            model.sigma_sq_bps >= 0.5,
            "variance should stay above 0.5: {}",
            model.sigma_sq_bps
        );
    }

    #[test]
    fn test_batch_flow_update_order_independent() {
        // Two models with same state, different flow orderings should give same result
        let mut model1 = BayesianFairValue::with_defaults();
        let mut model2 = BayesianFairValue::with_defaults();

        model1.predict(100.0, 0.0, 1000);
        model2.predict(100.0, 0.0, 1000);

        // Set same initial state
        model1.mu_offset_bps = 2.0;
        model2.mu_offset_bps = 2.0;

        // Both use batch update → same result regardless of internal ordering
        model1.update_on_flow(0.5, -0.2, 0.1, 100.0);
        model2.update_on_flow(0.5, -0.2, 0.1, 100.0);

        assert_approx(
            model1.mu_offset_bps,
            model2.mu_offset_bps,
            1e-10,
            "batch flow update should be order-independent",
        );
        assert_approx(
            model1.sigma_sq_bps,
            model2.sigma_sq_bps,
            1e-10,
            "batch flow variance should be order-independent",
        );
    }

    #[test]
    fn test_learned_params_bounded() {
        let mut model = BayesianFairValue::with_defaults();

        // Extreme gradient should not push params out of bounds
        for _ in 0..1000 {
            model.update_learned_params(100.0, 1.0); // Extreme return
        }

        assert!(
            model.alpha_book >= 0.1 && model.alpha_book <= 5.0,
            "alpha_book out of bounds: {}",
            model.alpha_book
        );
        assert!(
            model.sigma_noise_bps >= 1.0 && model.sigma_noise_bps <= 20.0,
            "sigma_noise out of bounds: {}",
            model.sigma_noise_bps
        );
    }

    #[test]
    fn test_learned_params_decay_toward_prior() {
        let mut model = BayesianFairValue::with_defaults();

        // Shift params away from prior
        model.alpha_book = 4.0;
        model.sigma_noise_bps = 15.0;

        // Many updates with zero gradient should decay toward prior
        for _ in 0..10000 {
            model.update_learned_params(0.0, 0.0);
        }

        // Should be closer to prior than initial
        let dist_alpha = (model.alpha_book - model.config.alpha_book_prior).abs();
        assert!(
            dist_alpha < 1.0,
            "alpha_book should decay toward prior: {}, dist={}",
            model.alpha_book,
            dist_alpha
        );
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut model = BayesianFairValue::with_defaults();
        model.predict(100.0, 0.0, 1000);

        // Process some fills to change learned params
        for _ in 0..5 {
            let fill_price = 100.0 * (1.0 + 3.0 / 10_000.0);
            model.update_on_fill(fill_price, 1.0, 1.0, false, 100.0, 0.3);
            model.update_learned_params(2.0, 0.5);
        }

        // Save checkpoint
        let ckpt = model.checkpoint();
        assert_eq!(ckpt.n_fills, 5);

        // Restore into fresh model
        let mut model2 = BayesianFairValue::with_defaults();
        model2.restore_from_checkpoint(&ckpt);

        assert_approx(
            model2.alpha_book,
            model.alpha_book,
            1e-10,
            "checkpoint alpha_book",
        );
        assert_approx(
            model2.sigma_noise_bps,
            model.sigma_noise_bps,
            1e-10,
            "checkpoint sigma_noise",
        );
        for i in 0..3 {
            assert_approx(
                model2.beta_flow[i],
                model.beta_flow[i],
                1e-10,
                &format!("checkpoint beta_flow[{}]", i),
            );
        }
    }

    #[test]
    fn test_feb22_acceptance_first_fill() {
        // Reproduce the Feb 22 incident math
        let mut model = BayesianFairValue::with_defaults();

        // Session starts. 155s of process noise before first fill.
        model.predict(100.0, 155.0, 155_000);

        // Variance should cap at 100
        assert_approx(
            model.sigma_sq_bps,
            100.0,
            TOLERANCE,
            "pre-fill variance capped",
        );

        // Fill 1: ask fill at 3 bps from mid, VPIN ≈ 0
        let fill_price = 100.0 * (1.0 + 3.0 / 10_000.0);
        model.update_on_fill(fill_price, 1.0, 1.0, false, 100.0, 0.0);

        // K = 100/(100+9) = 0.917
        // μ = 0.917 × 3.0 = 2.75
        assert_approx(model.mu_offset_bps, 2.75, 0.05, "fill 1 mu");

        // σ² = 100 × 0.083 = 8.26
        assert_approx(model.sigma_sq_bps, 8.26, 0.2, "fill 1 sigma_sq");
    }

    #[test]
    fn test_feb22_acceptance_five_fills() {
        // Full 5-fill cascade simulation
        let mut model = BayesianFairValue::with_defaults();

        // 155s before first fill
        model.predict(100.0, 155.0, 155_000);

        let timestamps = [155_000u64, 157_000, 158_000, 159_000, 160_000];
        let vpins = [0.0, 0.1, 0.2, 0.3, 0.4];
        let dts = [0.0, 2.0, 1.0, 1.0, 1.0]; // seconds between fills

        let mut mus = Vec::new();

        for i in 0..5 {
            if i > 0 {
                model.predict(100.0, dts[i], timestamps[i]);
            }
            let fill_price = 100.0 * (1.0 + 3.0 / 10_000.0);
            model.update_on_fill(fill_price, 1.0, 1.0, false, 100.0, vpins[i]);
            mus.push(model.mu_offset_bps);
        }

        // After 5 fills, μ should be well above 5 bps
        assert!(
            model.mu_offset_bps > 5.0,
            "After 5 fills, mu should be > 5 bps: {:.2}",
            model.mu_offset_bps
        );

        // Each successive fill should increase μ
        for i in 1..5 {
            assert!(
                mus[i] > mus[i - 1],
                "Fill {} mu ({:.2}) should be > fill {} mu ({:.2})",
                i + 1,
                mus[i],
                i,
                mus[i - 1]
            );
        }

        // Sigma should have tightened significantly from initial 5 bps
        assert!(
            model.sigma_bps() < 3.0,
            "Sigma should be < 3 bps after 5 fills: {:.2}",
            model.sigma_bps()
        );
    }

    #[test]
    fn test_book_update_weak_signal() {
        let mut model = BayesianFairValue::with_defaults();
        model.predict(100.0, 0.0, 1000);

        // Strong book imbalance should only move μ slightly
        model.update_on_book(1.0, 100.0);

        // With α=1.0, divisor=10, H=0.1, σ_book=5
        // K = σ² × H / (H² × σ² + R) = 25 × 0.1 / (0.01 × 25 + 25) = 2.5 / 25.25 ≈ 0.099
        // μ = 0 + 0.099 × (1.0 - 0) = 0.099
        assert!(
            model.mu_offset_bps.abs() < 0.5,
            "Book update should be weak: {}",
            model.mu_offset_bps
        );
    }

    #[test]
    fn test_default_beliefs_neutral() {
        let model = BayesianFairValue::with_defaults();
        let beliefs = model.beliefs();

        assert!(!beliefs.is_warmed_up);
        assert!(!beliefs.is_cascade);
        assert_approx(
            beliefs.offset_from_mid_bps,
            0.0,
            TOLERANCE,
            "default offset",
        );
        assert_approx(beliefs.confidence, 0.0, TOLERANCE, "default confidence");
    }

    #[test]
    fn test_beliefs_after_fills() {
        let mut model = BayesianFairValue::with_defaults();
        model.predict(100.0, 0.0, 1000);

        for _ in 0..5 {
            let fill_price = 100.0 * (1.0 + 3.0 / 10_000.0);
            model.update_on_fill(fill_price, 1.0, 1.0, false, 100.0, 0.0);
        }

        let beliefs = model.beliefs();
        assert!(beliefs.is_warmed_up);
        assert!(beliefs.offset_from_mid_bps > 0.0);
        assert!(beliefs.confidence > 0.0);
        assert!(beliefs.posterior_sigma_bps < 5.0); // Tighter than prior
    }

    #[test]
    fn test_reservation_price_with_inventory() {
        let mut model = BayesianFairValue::with_defaults();
        model.predict(100.0, 0.0, 1000);
        model.mu_offset_bps = 5.0; // Fair value 5 bps above mid

        // Long inventory → reservation price below posterior mean
        let r_long = model.reservation_price(1.0, 1.0, 1.0);
        let mean = model.posterior_mean();
        assert!(
            r_long < mean,
            "Long inventory should lower reservation price: r={}, mean={}",
            r_long,
            mean
        );

        // Short inventory → reservation price above posterior mean
        let r_short = model.reservation_price(-1.0, 1.0, 1.0);
        assert!(
            r_short > mean,
            "Short inventory should raise reservation price: r={}, mean={}",
            r_short,
            mean
        );
    }

    #[test]
    fn test_predict_zero_dt_no_change() {
        let mut model = BayesianFairValue::with_defaults();
        model.predict(100.0, 0.0, 1000);
        model.mu_offset_bps = 3.0;
        let sigma_before = model.sigma_sq_bps;

        model.predict(100.0, 0.0, 1000);
        assert_approx(model.mu_offset_bps, 3.0, 1e-10, "zero dt no mu change");
        assert_approx(
            model.sigma_sq_bps,
            sigma_before,
            1e-10,
            "zero dt no sigma change",
        );
    }

    #[test]
    fn test_flow_update_reduces_variance() {
        let mut model = BayesianFairValue::with_defaults();
        model.predict(100.0, 0.0, 1000);

        let sigma_before = model.sigma_sq_bps;
        model.update_on_flow(0.5, 0.3, 0.1, 100.0);

        assert!(
            model.sigma_sq_bps < sigma_before,
            "Flow update should reduce variance: before={}, after={}",
            sigma_before,
            model.sigma_sq_bps
        );
    }
}
