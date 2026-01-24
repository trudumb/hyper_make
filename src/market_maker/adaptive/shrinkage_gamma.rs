//! Shrinkage Gamma - Component 2
//!
//! Implements log-additive gamma adjustment with horseshoe-inspired shrinkage.
//! Replaces the multiplicative gamma scaling that causes spread explosion.
//!
//! # Mathematical Model
//!
//! Instead of: γ_eff = γ_base × vol_scalar × tox_scalar × inv_scalar × ...
//!
//! We use: log(γ_eff) = log(γ_base) + Σᵢ wᵢ × zᵢ
//!
//! Where:
//! - zᵢ are standardized signals (mean 0, variance ~1)
//! - wᵢ are learned weights with shrinkage prior
//!
//! # Shrinkage Prior
//!
//! The horseshoe prior:
//! ```text
//! wᵢ | λᵢ, τ ~ Normal(0, λᵢ² × τ²)
//! λᵢ ~ Half-Cauchy(0, 1)     # Local shrinkage
//! τ ~ Half-Cauchy(0, τ₀)     # Global shrinkage
//! ```
//!
//! For computational tractability, we use an empirical Bayes approximation
//! with online learning of weights and global shrinkage.

use tracing::debug;

use super::config::GammaSignal;
use super::standardizer::SignalStandardizers;

/// Shrinkage gamma with log-additive signal adjustment.
///
/// Key properties:
/// - Additive in log-space prevents multiplicative explosion
/// - Shrinkage prior pulls weights toward 0 when signals are noisy
/// - Online learning adapts weights based on PnL outcomes
#[derive(Debug, Clone)]
pub struct ShrinkageGamma {
    /// Base log-gamma (log of risk aversion in normal conditions)
    log_gamma_base: f64,

    /// Signal names (for indexing and logging)
    signals: Vec<GammaSignal>,

    /// Signal weights (learned, one per signal)
    weights: Vec<f64>,

    /// Local shrinkage λᵢ² per signal
    lambda_squared: Vec<f64>,

    /// Global shrinkage τ²
    tau_squared: f64,

    /// Learning rate for weight updates
    learning_rate: f64,

    /// Minimum gamma bound
    gamma_min: f64,

    /// Maximum gamma bound
    gamma_max: f64,

    /// Signal standardizers
    standardizers: SignalStandardizers,

    /// Update count (for logging)
    update_count: usize,
}

impl ShrinkageGamma {
    /// Create a new shrinkage gamma estimator.
    ///
    /// # Arguments
    /// * `gamma_base` - Base gamma (not log)
    /// * `signals` - Which signals to include
    /// * `tau_initial` - Initial global shrinkage
    /// * `learning_rate` - Learning rate for weight updates
    /// * `gamma_min` - Minimum gamma bound
    /// * `gamma_max` - Maximum gamma bound
    pub fn new(
        gamma_base: f64,
        signals: Vec<GammaSignal>,
        tau_initial: f64,
        learning_rate: f64,
        gamma_min: f64,
        gamma_max: f64,
    ) -> Self {
        let n_signals = signals.len();

        Self {
            log_gamma_base: gamma_base.ln(),
            signals,
            weights: vec![0.0; n_signals], // Start at 0 (no adjustment)
            lambda_squared: vec![1.0; n_signals], // Start with unit local shrinkage
            tau_squared: tau_initial * tau_initial,
            learning_rate,
            gamma_min,
            gamma_max,
            standardizers: SignalStandardizers::default(),
            update_count: 0,
        }
    }

    /// Create from config.
    pub fn from_config(config: &super::AdaptiveBayesianConfig) -> Self {
        Self::new(
            config.gamma_base,
            config.gamma_signals.clone(),
            config.tau_initial,
            config.gamma_learning_rate,
            config.gamma_min,
            config.gamma_max,
        )
    }

    /// Compute effective gamma from raw signal values.
    ///
    /// # Arguments
    /// * `raw_signals` - Map of signal type to raw value
    ///
    /// # Returns
    /// Effective gamma, clamped to [gamma_min, gamma_max]
    pub fn effective_gamma(&mut self, raw_signals: &[(GammaSignal, f64)]) -> f64 {
        // Standardize signals and compute weighted sum
        let mut log_adjustment = 0.0;

        for (i, signal) in self.signals.iter().enumerate() {
            // Find the raw value for this signal
            let raw = raw_signals
                .iter()
                .find(|(s, _)| s == signal)
                .map(|(_, v)| *v)
                .unwrap_or(0.0);

            // Standardize
            let z = self.standardizers.standardize(signal, raw);

            // Weighted contribution
            log_adjustment += self.weights[i] * z;
        }

        // Compute gamma
        let gamma = (self.log_gamma_base + log_adjustment).exp();
        gamma.clamp(self.gamma_min, self.gamma_max)
    }

    /// Update weights based on PnL outcome.
    ///
    /// # Arguments
    /// * `standardized_signals` - The standardized signals used for the quote
    /// * `pnl_gradient` - Direction of improvement:
    ///   - Positive: wider spread would have been better (increase weights)
    ///   - Negative: tighter spread would have been better (decrease weights)
    /// * `magnitude` - Magnitude of the PnL outcome (for scaling update)
    pub fn update(&mut self, standardized_signals: &[f64], pnl_gradient: f64, magnitude: f64) {
        if standardized_signals.len() != self.weights.len() {
            return;
        }

        self.update_count += 1;

        // Scale learning rate by magnitude (larger PnL = stronger signal)
        let scaled_lr = self.learning_rate * magnitude.abs().min(1.0);

        for (i, &std_signal) in standardized_signals
            .iter()
            .enumerate()
            .take(self.weights.len())
        {
            // Gradient of PnL w.r.t. weight
            let grad = pnl_gradient * std_signal;

            // Ridge penalty from shrinkage prior
            // penalty = w / (λ² × τ²)
            let shrinkage_penalty =
                self.weights[i] / (self.lambda_squared[i] * self.tau_squared).max(1e-9);

            // Update weight
            self.weights[i] += scaled_lr * (grad - 0.1 * shrinkage_penalty);

            // Update local shrinkage λᵢ² based on weight magnitude
            // If weight is large, increase λ² (less shrinkage)
            // If weight is small, decrease λ² (more shrinkage)
            let weight_sq = self.weights[i].powi(2);
            self.lambda_squared[i] = 0.99 * self.lambda_squared[i] + 0.01 * (weight_sq + 0.01);
        }

        // Update global shrinkage τ² based on overall weight variance
        let weight_variance: f64 =
            self.weights.iter().map(|w| w * w).sum::<f64>() / self.weights.len() as f64;
        self.tau_squared = 0.99 * self.tau_squared + 0.01 * (weight_variance + 0.01);

        // Log periodically
        if self.update_count.is_multiple_of(100) {
            debug!(
                update_count = self.update_count,
                tau_squared = self.tau_squared,
                weights = ?self.weights.iter().map(|w| format!("{w:.4}")).collect::<Vec<_>>(),
                "Shrinkage gamma updated"
            );
        }
    }

    /// Get current standardized signals for a raw signal map.
    ///
    /// This is useful for passing to `update()` after knowing the outcome.
    pub fn get_standardized_signals(&self, raw_signals: &[(GammaSignal, f64)]) -> Vec<f64> {
        self.signals
            .iter()
            .map(|signal| {
                let raw = raw_signals
                    .iter()
                    .find(|(s, _)| s == signal)
                    .map(|(_, v)| *v)
                    .unwrap_or(0.0);

                // Use peek (don't update) for this read
                self.standardizers.standardize_peek(signal, raw)
            })
            .collect()
    }

    /// Get the current weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get the global shrinkage parameter.
    pub fn tau_squared(&self) -> f64 {
        self.tau_squared
    }

    /// Get base gamma (not log).
    pub fn gamma_base(&self) -> f64 {
        self.log_gamma_base.exp()
    }

    /// Check if standardizers are warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.standardizers.all_warmed_up()
    }

    /// Reset weights to zero (but keep standardizer statistics).
    pub fn reset_weights(&mut self) {
        for w in &mut self.weights {
            *w = 0.0;
        }
        for l in &mut self.lambda_squared {
            *l = 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_shrinkage() -> ShrinkageGamma {
        ShrinkageGamma::new(
            0.3, // gamma_base
            vec![
                GammaSignal::VolatilityRatio,
                GammaSignal::JumpRatio,
                GammaSignal::InventoryUtilization,
            ],
            0.1,  // tau_initial
            0.01, // learning_rate
            0.05, // gamma_min
            2.0,  // gamma_max
        )
    }

    #[test]
    fn test_initial_gamma_equals_base() {
        let mut sg = default_shrinkage();

        // With zero weights, gamma should equal base
        let signals = vec![
            (GammaSignal::VolatilityRatio, 1.0),
            (GammaSignal::JumpRatio, 1.5),
            (GammaSignal::InventoryUtilization, 0.5),
        ];

        let gamma = sg.effective_gamma(&signals);

        // Should be close to 0.3 (base)
        assert!(
            (gamma - 0.3).abs() < 0.1,
            "Initial gamma should be ~0.3, got {}",
            gamma
        );
    }

    #[test]
    fn test_gamma_bounds() {
        let mut sg = default_shrinkage();

        // Manually set extreme weights
        sg.weights = vec![10.0, 10.0, 10.0];

        let signals = vec![
            (GammaSignal::VolatilityRatio, 100.0), // Very high
            (GammaSignal::JumpRatio, 100.0),
            (GammaSignal::InventoryUtilization, 100.0),
        ];

        let gamma = sg.effective_gamma(&signals);
        assert!(gamma <= 2.0, "Gamma should be capped at max: {}", gamma);

        // Test lower bound
        sg.weights = vec![-10.0, -10.0, -10.0];
        let gamma = sg.effective_gamma(&signals);
        assert!(gamma >= 0.05, "Gamma should be floored at min: {}", gamma);
    }

    #[test]
    fn test_weight_update_direction() {
        let mut sg = default_shrinkage();

        let standardized = vec![1.0, 0.5, 0.2]; // All positive signals

        // Positive gradient = wider spread would be better
        sg.update(&standardized, 1.0, 0.5);

        // Weights should increase (to widen spread)
        assert!(
            sg.weights[0] > 0.0,
            "Weight should increase on positive gradient"
        );
    }

    #[test]
    fn test_shrinkage_pulls_toward_zero() {
        let mut sg = default_shrinkage();

        // Set a large weight
        sg.weights[0] = 1.0;

        // Update with zero gradient (no signal)
        for _ in 0..100 {
            let standardized = vec![0.0, 0.0, 0.0];
            sg.update(&standardized, 0.0, 0.5);
        }

        // Shrinkage should pull weight toward zero
        assert!(
            sg.weights[0].abs() < 1.0,
            "Shrinkage should reduce weight: {}",
            sg.weights[0]
        );
    }

    #[test]
    fn test_log_additive_no_explosion() {
        let mut sg = default_shrinkage();

        // Even with all weights positive
        sg.weights = vec![0.5, 0.5, 0.5];

        // And high signal values
        let signals = vec![
            (GammaSignal::VolatilityRatio, 3.0),
            (GammaSignal::JumpRatio, 5.0),
            (GammaSignal::InventoryUtilization, 1.0),
        ];

        // Gamma should not explode like multiplicative would
        let gamma = sg.effective_gamma(&signals);

        // With multiplicative: 0.3 × 3 × 5 × 2 = 9.0
        // With log-additive: exp(log(0.3) + 0.5×z1 + 0.5×z2 + 0.5×z3) ≈ bounded

        assert!(
            gamma <= 2.0,
            "Log-additive should prevent explosion: {}",
            gamma
        );
    }

    #[test]
    fn test_interaction_terms() {
        // Create shrinkage gamma with interaction terms
        let mut sg = ShrinkageGamma::new(
            0.3, // gamma_base
            vec![
                GammaSignal::VolatilityRatio,
                GammaSignal::VolatilityXMomentum,
                GammaSignal::RegimeXInventory,
                GammaSignal::JumpXFlow,
            ],
            0.1,  // tau_initial
            0.01, // learning_rate
            0.05, // gamma_min
            2.0,  // gamma_max
        );

        // Set weights to test interaction effects
        // Base signal + 3 interaction signals = 4 weights
        sg.weights = vec![0.2, 0.3, 0.3, 0.3];

        // Simulate high volatility + high momentum (dangerous cascade scenario)
        // Interaction term vol_x_momentum should increase gamma
        let signals_dangerous = vec![
            (GammaSignal::VolatilityRatio, 2.0),     // High vol
            (GammaSignal::VolatilityXMomentum, 2.0), // High vol×momentum product
            (GammaSignal::RegimeXInventory, 0.0),    // Neutral
            (GammaSignal::JumpXFlow, 0.0),           // Neutral
        ];

        let gamma_dangerous = sg.effective_gamma(&signals_dangerous);

        // Simulate normal conditions
        let signals_normal = vec![
            (GammaSignal::VolatilityRatio, 1.0),     // Normal vol
            (GammaSignal::VolatilityXMomentum, 0.0), // Low interaction
            (GammaSignal::RegimeXInventory, 0.0),    // Neutral
            (GammaSignal::JumpXFlow, 0.0),           // Neutral
        ];

        let gamma_normal = sg.effective_gamma(&signals_normal);

        // Gamma should be higher in dangerous conditions due to interaction term
        assert!(
            gamma_dangerous > gamma_normal,
            "Gamma should be higher in dangerous conditions: {} vs {}",
            gamma_dangerous,
            gamma_normal
        );
    }
}
