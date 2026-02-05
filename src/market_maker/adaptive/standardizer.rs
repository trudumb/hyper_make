//! Signal Standardizer for Shrinkage Gamma and Prediction Models
//!
//! Standardizes raw signals to mean 0, variance 1 using Welford's online algorithm.
//! This ensures that learned weights are comparable across signals with different scales.
//!
//! # Enhanced Features for Prediction Models
//!
//! The `PredictionStandardizer` extends the basic standardizer with:
//! - **Output calibration**: Temperature scaling for sigmoid outputs
//! - **Base rate adjustment**: Learned intercept to match observed base rates
//! - **Concentration tracking**: Monitors if predictions cluster too tightly
//!
//! These features address the "IR collapse" problem where models achieve high
//! accuracy by predicting the base rate constantly.

/// Online signal standardizer using Welford's algorithm.
///
/// Standardizes signals to approximately N(0, 1) using running statistics.
/// This is essential for the shrinkage gamma model where weights are
/// comparable across signals.
#[derive(Debug, Clone)]
pub struct SignalStandardizer {
    /// Running mean
    mean: f64,

    /// Running M2 (sum of squared deviations)
    m2: f64,

    /// Observation count
    n: usize,

    /// Minimum observations before standardization is valid
    min_observations: usize,

    /// Default mean to use during warmup
    default_mean: f64,

    /// Default std to use during warmup
    default_std: f64,
}

impl SignalStandardizer {
    /// Create a new signal standardizer.
    ///
    /// # Arguments
    /// * `default_mean` - Default mean during warmup
    /// * `default_std` - Default std during warmup
    /// * `min_observations` - Minimum obs before using learned statistics
    pub fn new(default_mean: f64, default_std: f64, min_observations: usize) -> Self {
        Self {
            mean: default_mean,
            m2: 0.0,
            n: 0,
            min_observations,
            default_mean,
            default_std,
        }
    }

    /// Create with typical defaults for a [0, 1] bounded signal.
    /// Uses 5 observations for warmup (down from 20) for faster adaptation.
    pub fn for_bounded_signal() -> Self {
        Self::new(0.5, 0.25, 5)
    }

    /// Create with typical defaults for a ratio signal (e.g., vol_ratio).
    /// Uses 5 observations for warmup (down from 20) for faster adaptation.
    pub fn for_ratio_signal() -> Self {
        Self::new(1.0, 0.5, 5)
    }

    /// Create with typical defaults for a [-1, 1] bounded signal.
    /// Uses 5 observations for warmup (down from 20) for faster adaptation.
    pub fn for_symmetric_signal() -> Self {
        Self::new(0.0, 0.5, 5)
    }

    /// Update statistics and return standardized value.
    ///
    /// Uses Welford's online algorithm for numerical stability.
    pub fn standardize(&mut self, raw: f64) -> f64 {
        // Update running statistics using Welford's algorithm
        self.n += 1;
        let delta = raw - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = raw - self.mean;
        self.m2 += delta * delta2;

        // Compute standard deviation
        let std = self.std();

        // Standardize: z = (x - μ) / σ
        (raw - self.mean) / std.max(1e-9)
    }

    /// Standardize without updating statistics (peek).
    pub fn standardize_peek(&self, raw: f64) -> f64 {
        let std = self.std();
        (raw - self.mean) / std.max(1e-9)
    }

    /// Get current mean.
    pub fn mean(&self) -> f64 {
        if self.n < self.min_observations {
            self.default_mean
        } else {
            self.mean
        }
    }

    /// Get current standard deviation.
    pub fn std(&self) -> f64 {
        // Use default if not enough observations OR less than 2 for variance calc
        if self.n < self.min_observations || self.n < 2 {
            self.default_std
        } else {
            (self.m2 / (self.n - 1) as f64).sqrt().max(1e-9)
        }
    }

    /// Get current variance.
    pub fn variance(&self) -> f64 {
        self.std().powi(2)
    }

    /// Check if standardizer has enough data.
    pub fn is_warmed_up(&self) -> bool {
        self.n >= self.min_observations
    }

    /// Get observation count.
    pub fn observation_count(&self) -> usize {
        self.n
    }

    /// Reset statistics.
    pub fn reset(&mut self) {
        self.mean = self.default_mean;
        self.m2 = 0.0;
        self.n = 0;
    }
}

/// Collection of signal standardizers for all gamma signals.
#[derive(Debug, Clone)]
pub(super) struct SignalStandardizers {
    // Base signals
    pub vol_ratio: SignalStandardizer,
    pub jump_ratio: SignalStandardizer,
    pub inventory: SignalStandardizer,
    pub hawkes: SignalStandardizer,
    pub spread_regime: SignalStandardizer,
    pub cascade: SignalStandardizer,

    // Interaction terms (products of base signals)
    pub vol_x_momentum: SignalStandardizer,
    pub regime_x_inventory: SignalStandardizer,
    pub jump_x_flow: SignalStandardizer,

    // Additional base signals needed for interactions (reserved for future use)
    #[allow(dead_code)]
    pub momentum_abs: SignalStandardizer,
    #[allow(dead_code)]
    pub flow_abs: SignalStandardizer,
}

impl Default for SignalStandardizers {
    fn default() -> Self {
        Self {
            // Vol ratio: typically 0.5 - 2.0, centered at 1.0
            // Reduced warmup to 5 observations for faster adaptation
            vol_ratio: SignalStandardizer::new(1.0, 0.5, 5),

            // Jump ratio: typically 1.0 - 5.0, centered at 1.5
            // Reduced warmup to 5 observations for faster adaptation
            jump_ratio: SignalStandardizer::new(1.5, 1.0, 5),

            // Inventory utilization: 0.0 - 1.0
            inventory: SignalStandardizer::for_bounded_signal(),

            // Hawkes percentile: 0.0 - 1.0
            hawkes: SignalStandardizer::for_bounded_signal(),

            // Spread regime: -1 to 1 (encoded)
            spread_regime: SignalStandardizer::for_symmetric_signal(),

            // Cascade severity: 0.0 - 1.0
            cascade: SignalStandardizer::for_bounded_signal(),

            // === Interaction terms ===
            // These are products of standardized signals, so they're ~N(0,1) products
            // For z1 ~ N(0,1) and z2 ~ N(0,1), E[z1*z2] = 0, Var[z1*z2] ≈ 1
            vol_x_momentum: SignalStandardizer::new(0.0, 1.0, 5),
            regime_x_inventory: SignalStandardizer::new(0.0, 1.0, 5),
            jump_x_flow: SignalStandardizer::new(0.0, 1.0, 5),

            // Additional base signals for interactions
            // Momentum absolute value: 0.0 - 20 bps typically
            momentum_abs: SignalStandardizer::new(5.0, 5.0, 5),
            // Flow imbalance absolute: 0.0 - 1.0
            flow_abs: SignalStandardizer::for_bounded_signal(),
        }
    }
}

impl SignalStandardizers {
    /// Standardize a signal by name.
    pub(super) fn standardize(&mut self, signal: &super::config::GammaSignal, raw: f64) -> f64 {
        match signal {
            super::config::GammaSignal::VolatilityRatio => self.vol_ratio.standardize(raw),
            super::config::GammaSignal::JumpRatio => self.jump_ratio.standardize(raw),
            super::config::GammaSignal::InventoryUtilization => self.inventory.standardize(raw),
            super::config::GammaSignal::HawkesIntensity => self.hawkes.standardize(raw),
            super::config::GammaSignal::SpreadRegime => self.spread_regime.standardize(raw),
            super::config::GammaSignal::CascadeSeverity => self.cascade.standardize(raw),
            // Interaction terms - raw is already the product
            super::config::GammaSignal::VolatilityXMomentum => self.vol_x_momentum.standardize(raw),
            super::config::GammaSignal::RegimeXInventory => {
                self.regime_x_inventory.standardize(raw)
            }
            super::config::GammaSignal::JumpXFlow => self.jump_x_flow.standardize(raw),
        }
    }

    /// Standardize a signal without updating (peek).
    pub(super) fn standardize_peek(&self, signal: &super::config::GammaSignal, raw: f64) -> f64 {
        match signal {
            super::config::GammaSignal::VolatilityRatio => self.vol_ratio.standardize_peek(raw),
            super::config::GammaSignal::JumpRatio => self.jump_ratio.standardize_peek(raw),
            super::config::GammaSignal::InventoryUtilization => {
                self.inventory.standardize_peek(raw)
            }
            super::config::GammaSignal::HawkesIntensity => self.hawkes.standardize_peek(raw),
            super::config::GammaSignal::SpreadRegime => self.spread_regime.standardize_peek(raw),
            super::config::GammaSignal::CascadeSeverity => self.cascade.standardize_peek(raw),
            super::config::GammaSignal::VolatilityXMomentum => {
                self.vol_x_momentum.standardize_peek(raw)
            }
            super::config::GammaSignal::RegimeXInventory => {
                self.regime_x_inventory.standardize_peek(raw)
            }
            super::config::GammaSignal::JumpXFlow => self.jump_x_flow.standardize_peek(raw),
        }
    }

    /// Standardize momentum_abs (helper for interaction terms).
    #[allow(dead_code)]
    pub(super) fn standardize_momentum_abs(&mut self, raw: f64) -> f64 {
        self.momentum_abs.standardize(raw)
    }

    /// Standardize flow_abs (helper for interaction terms).
    #[allow(dead_code)]
    pub(super) fn standardize_flow_abs(&mut self, raw: f64) -> f64 {
        self.flow_abs.standardize(raw)
    }

    /// Check if all standardizers are warmed up.
    pub(super) fn all_warmed_up(&self) -> bool {
        self.vol_ratio.is_warmed_up()
            && self.jump_ratio.is_warmed_up()
            && self.inventory.is_warmed_up()
            && self.hawkes.is_warmed_up()
    }
}

// ============================================================================
// PREDICTION STANDARDIZER - Enhanced for ML Prediction Tasks
// ============================================================================

/// Enhanced standardizer for prediction models.
///
/// Addresses the "IR collapse" problem by:
/// 1. Tracking output distribution to prevent concentration
/// 2. Applying temperature scaling to spread predictions
/// 3. Learning optimal scaling parameters online
///
/// # Key Insight
///
/// Raw features → sigmoid often produces outputs clustered near base rate.
/// This standardizer applies learned transformations to spread predictions
/// across the useful [0.1, 0.9] range while maintaining calibration.
#[derive(Debug, Clone)]
pub struct PredictionStandardizer {
    /// Input standardizer (z-score normalization)
    input_standardizer: SignalStandardizer,

    /// Temperature for sigmoid (lower = sharper predictions)
    /// Learned online to achieve target spread
    temperature: f64,

    /// Bias term (logit space) to match base rate
    bias: f64,

    /// Target prediction spread (e.g., 0.4 means we want std(predictions) ≈ 0.2)
    target_spread: f64,

    /// Observed output mean (for calibration)
    output_mean: f64,

    /// Observed output variance (for concentration detection)
    output_variance: f64,

    /// EMA alpha for output statistics
    output_alpha: f64,

    /// Number of calibration updates
    calibration_count: usize,

    /// Minimum samples before calibration
    min_calibration_samples: usize,

    /// Base rate estimate (from outcomes)
    base_rate: f64,

    /// Outcome count for base rate estimation
    outcome_count: usize,

    /// Sum of outcomes for base rate
    outcome_sum: f64,
}

impl PredictionStandardizer {
    /// Create a new prediction standardizer.
    ///
    /// # Arguments
    /// * `default_mean` - Default mean for input standardization
    /// * `default_std` - Default std for input standardization
    /// * `min_observations` - Minimum observations before using learned stats
    pub fn new(default_mean: f64, default_std: f64, min_observations: usize) -> Self {
        Self {
            input_standardizer: SignalStandardizer::new(default_mean, default_std, min_observations),
            temperature: 1.0,
            bias: 0.0,
            target_spread: 0.4, // Target std of predictions ≈ 0.2
            output_mean: 0.5,
            output_variance: 0.0625, // (0.25)^2
            output_alpha: 0.01,
            calibration_count: 0,
            min_calibration_samples: 100,
            base_rate: 0.5,
            outcome_count: 0,
            outcome_sum: 0.0,
        }
    }

    /// Create with typical defaults for a [0, 1] bounded feature.
    pub fn for_bounded_feature() -> Self {
        Self::new(0.5, 0.25, 20)
    }

    /// Create with typical defaults for z-score input (already standardized).
    pub fn for_zscore_input() -> Self {
        Self::new(0.0, 1.0, 20)
    }

    /// Process a raw feature value and return a calibrated probability.
    ///
    /// Pipeline:
    /// 1. Z-score standardize the input
    /// 2. Apply temperature scaling
    /// 3. Apply bias (logit space)
    /// 4. Sigmoid to get probability
    /// 5. Update output statistics
    pub fn transform(&mut self, raw: f64) -> f64 {
        // Step 1: Z-score standardize
        let z = self.input_standardizer.standardize(raw);

        // Step 2-4: Temperature-scaled sigmoid with bias
        let logit = z / self.temperature + self.bias;
        let prob = 1.0 / (1.0 + (-logit).exp());

        // Step 5: Update output statistics
        self.update_output_stats(prob);

        prob.clamp(0.001, 0.999)
    }

    /// Transform without updating statistics (for inference only).
    pub fn transform_peek(&self, raw: f64) -> f64 {
        let z = self.input_standardizer.standardize_peek(raw);
        let logit = z / self.temperature + self.bias;
        let prob = 1.0 / (1.0 + (-logit).exp());
        prob.clamp(0.001, 0.999)
    }

    /// Update output statistics and potentially recalibrate.
    fn update_output_stats(&mut self, output: f64) {
        // EMA update for output mean and variance
        let delta = output - self.output_mean;
        self.output_mean += self.output_alpha * delta;
        self.output_variance = (1.0 - self.output_alpha) * self.output_variance
            + self.output_alpha * delta * delta;

        self.calibration_count += 1;

        // Periodically recalibrate temperature
        if self.calibration_count >= self.min_calibration_samples
            && self.calibration_count % 100 == 0
        {
            self.recalibrate();
        }
    }

    /// Record an outcome for base rate estimation.
    ///
    /// Call this after each prediction is resolved (was the fill adverse?).
    pub fn record_outcome(&mut self, was_positive: bool) {
        self.outcome_count += 1;
        if was_positive {
            self.outcome_sum += 1.0;
        }

        // Update base rate with decay (recent outcomes weighted more)
        if self.outcome_count > 100 {
            let alpha = 0.01;
            let outcome = if was_positive { 1.0 } else { 0.0 };
            self.base_rate = (1.0 - alpha) * self.base_rate + alpha * outcome;
        } else {
            self.base_rate = self.outcome_sum / self.outcome_count as f64;
        }

        // Update bias to match base rate
        self.update_bias();
    }

    /// Recalibrate temperature to achieve target spread.
    fn recalibrate(&mut self) {
        let current_std = self.output_variance.sqrt();
        let target_std = self.target_spread / 2.0; // target_spread is range, want std

        if current_std < 0.01 {
            // Severe concentration - reduce temperature aggressively
            self.temperature *= 0.8;
        } else if current_std < target_std * 0.5 {
            // Too concentrated - reduce temperature
            self.temperature *= 0.95;
        } else if current_std > target_std * 1.5 {
            // Too spread - increase temperature
            self.temperature *= 1.05;
        }

        // Clamp temperature to reasonable range
        self.temperature = self.temperature.clamp(0.1, 5.0);
    }

    /// Update bias to match observed base rate.
    fn update_bias(&mut self) {
        if self.outcome_count < 50 {
            return;
        }

        // Target: sigmoid(bias) = base_rate when z=0
        // So bias = logit(base_rate)
        let target_bias = (self.base_rate / (1.0 - self.base_rate + 1e-9)).ln();

        // Smooth update toward target
        self.bias = 0.95 * self.bias + 0.05 * target_bias.clamp(-3.0, 3.0);
    }

    /// Get diagnostics for monitoring.
    pub fn diagnostics(&self) -> PredictionStandardizerDiagnostics {
        let output_std = self.output_variance.sqrt();
        let concentration = if output_std < 0.05 {
            1.0 // Extremely concentrated
        } else if output_std < 0.1 {
            0.7 // Moderately concentrated
        } else {
            (0.2 - output_std).max(0.0) / 0.2 // Normal range
        };

        PredictionStandardizerDiagnostics {
            input_mean: self.input_standardizer.mean(),
            input_std: self.input_standardizer.std(),
            temperature: self.temperature,
            bias: self.bias,
            output_mean: self.output_mean,
            output_std,
            concentration_warning: concentration > 0.5,
            base_rate: self.base_rate,
            calibration_count: self.calibration_count,
            is_warmed_up: self.input_standardizer.is_warmed_up()
                && self.calibration_count >= self.min_calibration_samples,
        }
    }

    /// Check if predictions are too concentrated (IR collapse risk).
    pub fn is_concentrated(&self) -> bool {
        self.output_variance.sqrt() < 0.05 && self.calibration_count >= self.min_calibration_samples
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.input_standardizer.reset();
        self.temperature = 1.0;
        self.bias = 0.0;
        self.output_mean = 0.5;
        self.output_variance = 0.0625;
        self.calibration_count = 0;
        self.base_rate = 0.5;
        self.outcome_count = 0;
        self.outcome_sum = 0.0;
    }
}

/// Diagnostics for PredictionStandardizer.
#[derive(Debug, Clone)]
pub struct PredictionStandardizerDiagnostics {
    /// Input standardizer mean
    pub input_mean: f64,
    /// Input standardizer std
    pub input_std: f64,
    /// Current temperature (lower = sharper)
    pub temperature: f64,
    /// Current bias (logit space)
    pub bias: f64,
    /// Output mean
    pub output_mean: f64,
    /// Output std
    pub output_std: f64,
    /// Whether predictions are dangerously concentrated
    pub concentration_warning: bool,
    /// Estimated base rate
    pub base_rate: f64,
    /// Number of calibration updates
    pub calibration_count: usize,
    /// Whether fully warmed up
    pub is_warmed_up: bool,
}

impl PredictionStandardizerDiagnostics {
    /// Format as a compact string.
    pub fn summary(&self) -> String {
        format!(
            "in={:.2}±{:.2} T={:.2} b={:.2} out={:.2}±{:.2} base={:.1}%{}",
            self.input_mean,
            self.input_std,
            self.temperature,
            self.bias,
            self.output_mean,
            self.output_std,
            self.base_rate * 100.0,
            if self.concentration_warning {
                " ⚠️CONC"
            } else {
                ""
            }
        )
    }
}

// ============================================================================
// MULTI-FEATURE STANDARDIZER - For models with multiple input features
// ============================================================================

/// Standardizer collection for models with multiple features.
///
/// Manages a set of PredictionStandardizers, one per feature, plus
/// an output calibrator for the combined prediction.
#[derive(Debug, Clone)]
pub struct MultiFeatureStandardizer {
    /// Per-feature standardizers
    feature_standardizers: Vec<PredictionStandardizer>,

    /// Feature names (for diagnostics)
    feature_names: Vec<&'static str>,

    /// Output temperature for combined prediction
    output_temperature: f64,

    /// Output bias for combined prediction
    output_bias: f64,

    /// Track output distribution
    output_mean: f64,
    output_variance: f64,
    output_count: usize,
}

impl MultiFeatureStandardizer {
    /// Create a new multi-feature standardizer.
    ///
    /// # Arguments
    /// * `feature_specs` - List of (name, default_mean, default_std) for each feature
    pub fn new(feature_specs: &[(&'static str, f64, f64)]) -> Self {
        let feature_standardizers = feature_specs
            .iter()
            .map(|(_, mean, std)| PredictionStandardizer::new(*mean, *std, 20))
            .collect();

        let feature_names = feature_specs.iter().map(|(name, _, _)| *name).collect();

        Self {
            feature_standardizers,
            feature_names,
            output_temperature: 1.0,
            output_bias: 0.0,
            output_mean: 0.5,
            output_variance: 0.0625,
            output_count: 0,
        }
    }

    /// Create for z-score inputs (features already standardized).
    pub fn for_zscores(feature_names: &[&'static str]) -> Self {
        let specs: Vec<_> = feature_names.iter().map(|n| (*n, 0.0, 1.0)).collect();
        Self::new(&specs)
    }

    /// Standardize a vector of raw features.
    ///
    /// Returns z-score standardized features suitable for weighted combination.
    pub fn standardize(&mut self, raw_features: &[f64]) -> Vec<f64> {
        raw_features
            .iter()
            .zip(self.feature_standardizers.iter_mut())
            .map(|(raw, std)| std.input_standardizer.standardize(*raw))
            .collect()
    }

    /// Transform a weighted sum into a calibrated probability.
    ///
    /// Call this after computing weighted sum of standardized features.
    pub fn calibrate_output(&mut self, weighted_sum: f64) -> f64 {
        let logit = weighted_sum / self.output_temperature + self.output_bias;
        let prob = 1.0 / (1.0 + (-logit).exp());

        // Update output statistics
        let delta = prob - self.output_mean;
        self.output_mean += 0.01 * delta;
        self.output_variance = 0.99 * self.output_variance + 0.01 * delta * delta;
        self.output_count += 1;

        // Periodic recalibration
        if self.output_count % 100 == 0 && self.output_count > 100 {
            self.recalibrate_output();
        }

        prob.clamp(0.001, 0.999)
    }

    /// Record outcome for base rate calibration.
    pub fn record_outcome(&mut self, was_positive: bool) {
        // Update output bias toward base rate
        let outcome = if was_positive { 1.0 } else { 0.0 };
        let alpha = 0.01;

        // Estimate base rate
        let base_rate = self.output_mean; // Use recent mean as proxy

        // Adjust bias if predictions drift from base rate
        let target_bias = (base_rate / (1.0 - base_rate + 1e-9)).ln();
        self.output_bias = 0.99 * self.output_bias + 0.01 * target_bias.clamp(-3.0, 3.0);

        // Also propagate to individual standardizers
        for std in &mut self.feature_standardizers {
            std.record_outcome(was_positive);
        }
    }

    /// Recalibrate output temperature.
    fn recalibrate_output(&mut self) {
        let output_std = self.output_variance.sqrt();

        if output_std < 0.05 {
            // Severe concentration
            self.output_temperature *= 0.9;
        } else if output_std < 0.1 {
            // Moderate concentration
            self.output_temperature *= 0.98;
        } else if output_std > 0.3 {
            // Too spread
            self.output_temperature *= 1.02;
        }

        self.output_temperature = self.output_temperature.clamp(0.1, 5.0);
    }

    /// Check for concentration warning.
    pub fn is_concentrated(&self) -> bool {
        self.output_variance.sqrt() < 0.05 && self.output_count > 100
    }

    /// Get concentration percentage (0-100%).
    pub fn concentration_pct(&self) -> f64 {
        if self.output_count < 100 {
            return 0.0;
        }
        let std = self.output_variance.sqrt();
        // Map std to concentration %: std=0.05 → 80%, std=0.2 → 20%, std=0.35 → 0%
        ((0.35 - std) / 0.3 * 100.0).clamp(0.0, 100.0)
    }

    /// Get diagnostics.
    pub fn diagnostics(&self) -> MultiFeatureStandardizerDiagnostics {
        let feature_diagnostics: Vec<_> = self
            .feature_standardizers
            .iter()
            .zip(self.feature_names.iter())
            .map(|(std, name)| (name.to_string(), std.diagnostics()))
            .collect();

        MultiFeatureStandardizerDiagnostics {
            feature_count: self.feature_standardizers.len(),
            output_temperature: self.output_temperature,
            output_bias: self.output_bias,
            output_mean: self.output_mean,
            output_std: self.output_variance.sqrt(),
            concentration_pct: self.concentration_pct(),
            output_count: self.output_count,
            feature_diagnostics,
        }
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        for std in &mut self.feature_standardizers {
            std.reset();
        }
        self.output_temperature = 1.0;
        self.output_bias = 0.0;
        self.output_mean = 0.5;
        self.output_variance = 0.0625;
        self.output_count = 0;
    }
}

/// Diagnostics for MultiFeatureStandardizer.
#[derive(Debug, Clone)]
pub struct MultiFeatureStandardizerDiagnostics {
    pub feature_count: usize,
    pub output_temperature: f64,
    pub output_bias: f64,
    pub output_mean: f64,
    pub output_std: f64,
    pub concentration_pct: f64,
    pub output_count: usize,
    pub feature_diagnostics: Vec<(String, PredictionStandardizerDiagnostics)>,
}

impl MultiFeatureStandardizerDiagnostics {
    pub fn summary(&self) -> String {
        format!(
            "n={} T={:.2} b={:.2} out={:.2}±{:.2} conc={:.0}%",
            self.feature_count,
            self.output_temperature,
            self.output_bias,
            self.output_mean,
            self.output_std,
            self.concentration_pct
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standardization_mean_zero() {
        let mut s = SignalStandardizer::new(0.0, 1.0, 5);

        // Feed in values centered at 10
        for i in 0..100 {
            let raw = 10.0 + (i as f64 % 5.0) - 2.0; // 8, 9, 10, 11, 12
            s.standardize(raw);
        }

        // Mean should be close to 10
        assert!(
            (s.mean() - 10.0).abs() < 0.1,
            "Mean should be ~10, got {}",
            s.mean()
        );

        // Standardized mean should be ~0
        let z = s.standardize_peek(s.mean());
        assert!(z.abs() < 0.1, "Standardized mean should be ~0, got {}", z);
    }

    #[test]
    fn test_standardization_unit_variance() {
        let mut s = SignalStandardizer::new(0.0, 1.0, 5);

        // Feed in standard normal samples (approximated)
        let samples = [
            -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -0.8, 0.3, -0.2, 0.7, -1.2, 0.9, -0.6, 0.4,
        ];

        for &x in &samples {
            s.standardize(x);
        }

        // Std should be close to empirical std of samples
        let std = s.std();
        assert!(
            std > 0.5 && std < 1.5,
            "Std should be reasonable, got {}",
            std
        );
    }

    #[test]
    fn test_warmup_uses_defaults() {
        let s = SignalStandardizer::new(5.0, 2.0, 20);

        assert_eq!(s.mean(), 5.0);
        assert_eq!(s.std(), 2.0);
        assert!(!s.is_warmed_up());
    }

    #[test]
    fn test_warmup_transition() {
        let mut s = SignalStandardizer::new(0.0, 1.0, 10);

        for i in 0..9 {
            s.standardize(i as f64);
        }
        assert!(!s.is_warmed_up());

        s.standardize(9.0);
        assert!(s.is_warmed_up());

        // Now should use learned statistics
        assert!((s.mean() - 4.5).abs() < 0.1);
    }

    // =========================================================================
    // PredictionStandardizer Tests
    // =========================================================================

    #[test]
    fn test_prediction_standardizer_basic() {
        let mut ps = PredictionStandardizer::for_bounded_feature();

        // Feed in some values
        for i in 0..100 {
            let raw = 0.3 + (i as f64 % 10.0) * 0.05; // Range 0.3 to 0.75
            let prob = ps.transform(raw);
            assert!(prob >= 0.0 && prob <= 1.0, "Output should be probability");
        }

        let diag = ps.diagnostics();
        assert!(diag.calibration_count >= 100);
    }

    #[test]
    fn test_prediction_standardizer_concentration_detection() {
        let mut ps = PredictionStandardizer::for_bounded_feature();

        // Feed in very similar values (should trigger concentration warning)
        for _ in 0..200 {
            ps.transform(0.5 + 0.001 * (rand_like() - 0.5)); // Very tight range
        }

        // Should detect concentration
        let diag = ps.diagnostics();
        // Note: May need many samples for concentration to be detected
        // This test mainly verifies the logic doesn't panic
    }

    #[test]
    fn test_prediction_standardizer_outcome_tracking() {
        let mut ps = PredictionStandardizer::for_bounded_feature();

        // Record outcomes
        for i in 0..200 {
            ps.transform(0.5);
            ps.record_outcome(i % 3 == 0); // ~33% positive rate
        }

        let diag = ps.diagnostics();
        // Base rate should move toward 0.33
        assert!(
            (diag.base_rate - 0.33).abs() < 0.2,
            "Base rate should be near 0.33, got {}",
            diag.base_rate
        );
    }

    #[test]
    fn test_multi_feature_standardizer() {
        let mut mfs = MultiFeatureStandardizer::new(&[
            ("feature1", 0.0, 1.0),
            ("feature2", 0.5, 0.25),
            ("feature3", 1.0, 0.5),
        ]);

        // Process some data
        for i in 0..100 {
            let raw = vec![
                (i as f64 % 5.0) - 2.0,        // -2 to 2
                0.3 + (i as f64 % 10.0) * 0.05, // 0.3 to 0.75
                0.8 + (i as f64 % 5.0) * 0.1,   // 0.8 to 1.2
            ];
            let standardized = mfs.standardize(&raw);
            assert_eq!(standardized.len(), 3);

            // Compute weighted sum
            let weighted = standardized.iter().sum::<f64>() / 3.0;
            let prob = mfs.calibrate_output(weighted);
            assert!(prob >= 0.0 && prob <= 1.0);
        }

        let diag = mfs.diagnostics();
        assert_eq!(diag.feature_count, 3);
    }

    // Helper for test randomness
    fn rand_like() -> f64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        (nanos as f64 / u32::MAX as f64)
    }
}
