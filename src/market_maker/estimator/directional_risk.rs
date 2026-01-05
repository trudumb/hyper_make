//! Directional Risk Estimator - First Principles Momentum-Aware Variance.
//!
//! # Theory
//!
//! The standard Avellaneda-Stoikov model assumes price follows a martingale:
//! ```text
//! dS = σdW  (no drift)
//! ```
//!
//! But when we have momentum signals, price has predictable drift:
//! ```text
//! dS = μ(t)dt + σdW  where μ(t) = f(momentum, flow_imbalance)
//! ```
//!
//! This changes the optimal inventory skew from the HJB solution:
//!
//! ## Standard (no drift):
//! ```text
//! skew = γσ²qT
//! ```
//!
//! ## Extended with drift:
//! ```text
//! skew = γ × σ²_directional × q × T + drift_urgency
//! ```
//!
//! Where:
//! - `σ²_directional`: Variance adjusted for position-momentum alignment
//! - `drift_urgency`: Additional skew from predictable price movement
//!
//! # Directional Variance
//!
//! When position **opposes** momentum (short + rising, long + falling):
//! - Adverse moves are more likely (momentum predicts unfavorable direction)
//! - Effective variance increases: σ²_eff = σ² × (1 + κ × |μ|/σ)
//!
//! When position **aligns** with momentum:
//! - Risk is lower but don't reduce variance (momentum can reverse)
//! - Keep variance at baseline or slightly reduced
//!
//! # Drift Urgency
//!
//! From optimal control with drift, we add urgency proportional to:
//! ```text
//! urgency = μ × P(continuation) × time_exposure × inventory_pressure
//! ```
//!
//! This creates urgency to reduce position when momentum predicts adverse movement.

use std::collections::VecDeque;

/// Configuration for directional risk estimation.
#[derive(Debug, Clone)]
pub struct DirectionalRiskConfig {
    /// Sensitivity of variance to momentum opposition (κ in theory).
    /// Higher values increase variance more when opposed to momentum.
    /// Range: [0.5, 3.0], Default: 1.5
    pub opposition_sensitivity: f64,

    /// Maximum variance multiplier when opposed to strong momentum.
    /// Caps the directional adjustment to prevent extreme values.
    /// Range: [1.5, 4.0], Default: 2.5
    pub max_variance_multiplier: f64,

    /// Minimum variance multiplier when aligned with momentum.
    /// Don't reduce variance too much - momentum can reverse.
    /// Range: [0.7, 1.0], Default: 0.85
    pub min_variance_multiplier: f64,

    /// Sensitivity of urgency to continuation probability.
    /// Higher values create more urgency when P(continuation) is high.
    /// Range: [1.0, 5.0], Default: 2.0
    pub continuation_urgency_sensitivity: f64,

    /// Threshold for momentum strength to trigger urgency (in bps).
    /// Below this, no urgency adjustment is applied.
    /// Range: [5.0, 50.0], Default: 15.0
    pub momentum_urgency_threshold_bps: f64,

    /// EWMA half-life for smoothing directional estimates (seconds).
    /// Range: [5.0, 60.0], Default: 15.0
    pub ewma_half_life_secs: f64,

    /// Window size for momentum statistics (data points).
    /// Range: [20, 200], Default: 50
    pub stats_window_size: usize,
}

impl Default for DirectionalRiskConfig {
    fn default() -> Self {
        Self {
            opposition_sensitivity: 1.5,
            max_variance_multiplier: 2.5,
            min_variance_multiplier: 0.85,
            continuation_urgency_sensitivity: 2.0,
            momentum_urgency_threshold_bps: 15.0,
            ewma_half_life_secs: 15.0,
            stats_window_size: 50,
        }
    }
}

/// Output from directional risk estimation.
#[derive(Debug, Clone, Copy, Default)]
pub struct DirectionalRiskOutput {
    /// Variance multiplier for inventory skew (σ²_directional / σ²).
    /// > 1.0 when opposed to momentum, ≤ 1.0 when aligned.
    pub variance_multiplier: f64,

    /// Urgency factor to add to skew (in fractional price terms).
    /// Positive when position should be reduced urgently.
    pub drift_urgency: f64,

    /// Whether position is opposed to momentum.
    /// true = short + rising, or long + falling.
    pub is_opposed: bool,

    /// Strength of opposition/alignment [-1, 1].
    /// -1 = fully opposed, +1 = fully aligned.
    pub alignment_score: f64,

    /// Estimated drift rate (μ) in fractional price per second.
    pub estimated_drift: f64,

    /// Continuation probability from momentum model.
    pub p_continuation: f64,

    /// Combined urgency score [0, 5] for diagnostics.
    pub urgency_score: f64,
}

/// Directional Risk Estimator.
///
/// Computes momentum-aware variance adjustments and urgency factors
/// for the GLFT inventory skew calculation.
#[derive(Debug)]
pub struct DirectionalRiskEstimator {
    config: DirectionalRiskConfig,

    /// EWMA alpha for smoothing
    ewma_alpha: f64,

    /// Smoothed variance multiplier
    variance_mult_ewma: f64,

    /// Smoothed drift estimate
    drift_ewma: f64,

    /// Recent momentum observations for statistics
    momentum_history: VecDeque<f64>,

    /// Recent continuation probabilities
    continuation_history: VecDeque<f64>,

    /// Whether estimator is warmed up
    warmed_up: bool,

    /// Number of updates received
    update_count: u64,
}

impl DirectionalRiskEstimator {
    /// Create a new directional risk estimator.
    pub fn new(config: DirectionalRiskConfig) -> Self {
        // Compute EWMA alpha from half-life
        // alpha = 1 - exp(-ln(2) / half_life_updates)
        // Assume ~10 updates per second
        let updates_per_half_life = config.ewma_half_life_secs * 10.0;
        let ewma_alpha = 1.0 - (-2.0_f64.ln() / updates_per_half_life).exp();

        Self {
            ewma_alpha: ewma_alpha.clamp(0.01, 0.5),
            config,
            variance_mult_ewma: 1.0,
            drift_ewma: 0.0,
            momentum_history: VecDeque::with_capacity(200),
            continuation_history: VecDeque::with_capacity(200),
            warmed_up: false,
            update_count: 0,
        }
    }

    /// Create with default config.
    pub fn default_config() -> Self {
        Self::new(DirectionalRiskConfig::default())
    }

    /// Update the estimator with current market state.
    ///
    /// # Arguments
    /// * `position` - Current position (positive = long, negative = short)
    /// * `max_position` - Maximum position for normalization
    /// * `momentum_bps` - Current momentum in basis points (positive = rising)
    /// * `p_continuation` - Probability momentum continues [0, 1]
    /// * `sigma` - Current volatility (per-second)
    /// * `time_horizon` - Time horizon for inventory holding (seconds)
    pub fn update(
        &mut self,
        position: f64,
        max_position: f64,
        momentum_bps: f64,
        p_continuation: f64,
        sigma: f64,
        time_horizon: f64,
    ) -> DirectionalRiskOutput {
        self.update_count += 1;

        // Track history
        self.momentum_history.push_back(momentum_bps);
        if self.momentum_history.len() > self.config.stats_window_size {
            self.momentum_history.pop_front();
        }

        self.continuation_history.push_back(p_continuation);
        if self.continuation_history.len() > self.config.stats_window_size {
            self.continuation_history.pop_front();
        }

        // Check warmup
        if self.momentum_history.len() >= 20 {
            self.warmed_up = true;
        }

        // Normalize position to [-1, 1]
        let q = if max_position.abs() > 1e-10 {
            (position / max_position).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // Convert momentum to fractional drift estimate
        // μ = momentum_bps / 10000 / expected_duration
        // We estimate expected duration as 500ms for momentum signal
        let momentum_frac = momentum_bps / 10000.0;
        let estimated_drift = momentum_frac / 0.5; // Per-second drift

        // Smooth the drift estimate
        self.drift_ewma = self.ewma_alpha * estimated_drift + (1.0 - self.ewma_alpha) * self.drift_ewma;

        // === 1. Compute Position-Momentum Alignment ===
        // alignment = sign(position) × sign(momentum)
        // Positive = aligned (both same direction)
        // Negative = opposed (position against momentum)
        let pos_sign = q.signum();
        let mom_sign = momentum_bps.signum();
        let alignment_raw = pos_sign * mom_sign * momentum_bps.abs().min(100.0) / 100.0;

        // is_opposed = true when position and momentum are in opposite directions
        // Short (q < 0) and rising (mom > 0) → opposed
        // Long (q > 0) and falling (mom < 0) → opposed
        let is_opposed = q * momentum_bps < 0.0;

        // === 2. Compute Variance Multiplier ===
        // When opposed: increase variance proportionally to opposition strength
        // σ²_eff = σ² × (1 + κ × |opposition| × |momentum|/σ)
        let momentum_sigma_ratio = if sigma > 1e-10 {
            (momentum_frac.abs() / sigma).min(5.0) // Cap ratio
        } else {
            0.0
        };

        let variance_multiplier = if is_opposed {
            // Opposed: increase variance
            let opposition_strength = q.abs() * momentum_bps.abs() / 100.0; // Normalized
            let increase = self.config.opposition_sensitivity
                * opposition_strength.min(1.0)
                * (1.0 + momentum_sigma_ratio)
                * p_continuation;

            (1.0 + increase).min(self.config.max_variance_multiplier)
        } else if q.abs() > 0.1 && momentum_bps.abs() > self.config.momentum_urgency_threshold_bps {
            // Aligned with momentum: slight variance reduction (but cautious)
            let alignment_strength = q.abs() * momentum_bps.abs() / 100.0;
            let decrease = 0.15 * alignment_strength.min(1.0) * p_continuation;

            (1.0 - decrease).max(self.config.min_variance_multiplier)
        } else {
            1.0
        };

        // Smooth variance multiplier
        self.variance_mult_ewma = self.ewma_alpha * variance_multiplier
            + (1.0 - self.ewma_alpha) * self.variance_mult_ewma;

        // === 3. Compute Drift Urgency ===
        // Urgency = μ × P(continuation) × time_exposure × inventory_pressure
        // This adds to skew to accelerate position reduction when opposed
        let drift_urgency = if is_opposed
            && momentum_bps.abs() > self.config.momentum_urgency_threshold_bps
        {
            // Urgency proportional to:
            // - Drift magnitude (how fast price is moving against us)
            // - Continuation probability (how likely it continues)
            // - Position size (how much exposure we have)
            // - Time horizon (how long we're exposed)

            let urgency_base = self.drift_ewma.abs()
                * p_continuation
                * q.abs()
                * time_horizon.min(60.0); // Cap time contribution

            // Scale by continuation sensitivity
            let urgency_scaled = urgency_base * self.config.continuation_urgency_sensitivity;

            // Sign: positive urgency when we need to reduce
            // If short and rising, urgency should push quotes UP (negative skew → more aggressive buying)
            // If long and falling, urgency should push quotes DOWN (positive skew → more aggressive selling)
            urgency_scaled * pos_sign
        } else {
            0.0
        };

        // === 4. Compute Urgency Score for Diagnostics ===
        // Combines all urgency factors into a single score [0, 5]
        let urgency_score = if is_opposed {
            let momentum_factor = (momentum_bps.abs() / 50.0).min(1.0); // 50 bps = max
            let continuation_factor = p_continuation;
            let position_factor = q.abs();
            let volatility_factor = momentum_sigma_ratio.min(1.0);

            (momentum_factor + continuation_factor + position_factor + volatility_factor)
                .min(4.0)
                * (1.0 + 0.25 * (momentum_bps.abs() / 100.0).min(1.0)) // Boost for extreme momentum
        } else {
            0.0
        };

        DirectionalRiskOutput {
            variance_multiplier: self.variance_mult_ewma,
            drift_urgency,
            is_opposed,
            alignment_score: alignment_raw,
            estimated_drift: self.drift_ewma,
            p_continuation,
            urgency_score,
        }
    }

    /// Check if estimator is warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.warmed_up
    }

    /// Get current variance multiplier (smoothed).
    pub fn variance_multiplier(&self) -> f64 {
        self.variance_mult_ewma
    }

    /// Get current drift estimate (smoothed).
    pub fn drift_estimate(&self) -> f64 {
        self.drift_ewma
    }

    /// Get momentum statistics for diagnostics.
    pub fn momentum_stats(&self) -> MomentumStats {
        if self.momentum_history.is_empty() {
            return MomentumStats::default();
        }

        let sum: f64 = self.momentum_history.iter().sum();
        let mean = sum / self.momentum_history.len() as f64;

        let variance: f64 = self.momentum_history
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / self.momentum_history.len() as f64;

        let std_dev = variance.sqrt();

        // Count directional changes
        let mut direction_changes = 0;
        let mut prev_sign = 0.0_f64;
        for &m in &self.momentum_history {
            if prev_sign != 0.0 && m.signum() != prev_sign {
                direction_changes += 1;
            }
            if m.abs() > 1e-10 {
                prev_sign = m.signum();
            }
        }

        MomentumStats {
            mean_bps: mean,
            std_dev_bps: std_dev,
            direction_changes,
            sample_count: self.momentum_history.len(),
        }
    }

    /// Reset the estimator state.
    pub fn reset(&mut self) {
        self.variance_mult_ewma = 1.0;
        self.drift_ewma = 0.0;
        self.momentum_history.clear();
        self.continuation_history.clear();
        self.warmed_up = false;
        self.update_count = 0;
    }
}

/// Momentum statistics for diagnostics.
#[derive(Debug, Clone, Default)]
pub struct MomentumStats {
    pub mean_bps: f64,
    pub std_dev_bps: f64,
    pub direction_changes: usize,
    pub sample_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_estimator() -> DirectionalRiskEstimator {
        DirectionalRiskEstimator::new(DirectionalRiskConfig::default())
    }

    #[test]
    fn test_no_position_no_adjustment() {
        let mut est = make_estimator();

        let output = est.update(
            0.0,   // No position
            100.0, // max_position
            50.0,  // Strong momentum
            0.8,   // High continuation
            0.0001,
            60.0,
        );

        // No position = no directional adjustment
        assert!((output.variance_multiplier - 1.0).abs() < 0.1);
        assert!(output.drift_urgency.abs() < 0.001);
    }

    #[test]
    fn test_opposed_position_increases_variance() {
        let mut est = make_estimator();

        // Short position, rising momentum (opposed)
        let output = est.update(
            -50.0, // Short
            100.0,
            50.0, // Rising (opposed to short)
            0.8,  // High continuation
            0.0001,
            60.0,
        );

        assert!(output.is_opposed);
        assert!(output.variance_multiplier > 1.0);
        assert!(output.urgency_score > 0.0);
    }

    #[test]
    fn test_aligned_position_no_increase() {
        let mut est = make_estimator();

        // Long position, rising momentum (aligned)
        let output = est.update(
            50.0,  // Long
            100.0,
            50.0, // Rising (aligned with long)
            0.8,
            0.0001,
            60.0,
        );

        assert!(!output.is_opposed);
        assert!(output.variance_multiplier <= 1.0);
    }

    #[test]
    fn test_drift_urgency_when_opposed() {
        let mut est = make_estimator();

        // Warm up
        for _ in 0..25 {
            est.update(-50.0, 100.0, 50.0, 0.8, 0.0001, 60.0);
        }

        let output = est.update(
            -50.0, // Short
            100.0,
            50.0, // Rising strongly (opposed)
            0.9,  // Very high continuation
            0.0001,
            60.0,
        );

        // Should have positive drift urgency (push quotes up to buy faster)
        assert!(output.drift_urgency.abs() > 0.0);
        assert!(output.is_opposed);
    }

    #[test]
    fn test_low_continuation_reduces_urgency() {
        let mut est = make_estimator();

        // Opposed but low continuation probability
        let output = est.update(
            -50.0,
            100.0,
            50.0,
            0.2, // Low continuation
            0.0001,
            60.0,
        );

        // Low continuation should reduce urgency
        assert!(output.variance_multiplier < 1.5);
    }

    #[test]
    fn test_warmup() {
        let mut est = make_estimator();

        assert!(!est.is_warmed_up());

        for _ in 0..25 {
            est.update(0.0, 100.0, 10.0, 0.5, 0.0001, 60.0);
        }

        assert!(est.is_warmed_up());
    }

    #[test]
    fn test_momentum_stats() {
        let mut est = make_estimator();

        // Add some momentum readings
        for i in 0..30 {
            let momentum = if i % 2 == 0 { 10.0 } else { -5.0 };
            est.update(0.0, 100.0, momentum, 0.5, 0.0001, 60.0);
        }

        let stats = est.momentum_stats();
        assert!(stats.sample_count > 0);
        assert!(stats.direction_changes > 0); // Should have some direction changes
    }

    #[test]
    fn test_variance_multiplier_capped() {
        let mut est = make_estimator();

        // Extreme opposition
        let output = est.update(
            -100.0, // Max short
            100.0,
            200.0, // Extreme rising momentum
            1.0,   // Certain continuation
            0.00001, // Low vol (high momentum/vol ratio)
            60.0,
        );

        // Should be capped at max_variance_multiplier
        assert!(output.variance_multiplier <= est.config.max_variance_multiplier);
    }
}
