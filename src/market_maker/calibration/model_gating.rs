//! Model Gating - IR-based model confidence and spread adjustment.
//!
//! This module provides confidence-weighted model gating based on Information Ratio
//! metrics. Models with IR < 1.0 are downweighted or disabled, and spreads are
//! widened defensively when model confidence is low.
//!
//! # Key Concepts
//!
//! - **Information Ratio (IR)**: Measures model value-add over base rate
//!   - IR > 1.0: Strong standalone predictor
//!   - IR 0.5-1.0: Feature informs priors (useful for Bayesian updates)
//!   - IR < 0.5: Insufficient signal (should be disabled)
//!
//! - **Model Confidence**: Probability that IR > threshold (default 0.5)
//!   - P(IR > 0.5) > 0.7: High confidence, use model fully
//!   - P(IR > 0.5) < 0.3: Low confidence, disable model
//!
//! # Usage
//!
//! ```ignore
//! let mut gating = ModelGating::new(ModelGatingConfig::default());
//!
//! // Update with prediction outcomes
//! gating.update_adverse_selection(predicted_as_bps, actual_as_bps);
//!
//! // Check if models should be used
//! let weights = gating.model_weights();
//! let spread_mult = gating.spread_multiplier();
//! ```

use super::information_ratio::{ExponentialIRTracker, InformationRatioTracker};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Minimum signal weight to prevent death spiral.
/// Even unproven models contribute at 5% to allow IR accumulation.
const MIN_SIGNAL_WEIGHT: f64 = 0.05;

/// Configuration for model gating.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGatingConfig {
    /// Minimum samples before IR is considered reliable.
    pub min_samples: usize,

    /// IR threshold for model value (default: 1.0).
    pub ir_threshold: f64,

    /// Confidence threshold for "high confidence" (default: 0.7).
    pub high_confidence_threshold: f64,

    /// Confidence threshold for "low confidence" / disable (default: 0.3).
    pub low_confidence_threshold: f64,

    /// Spread multiplier when confidence is low (default: 1.5x).
    pub low_confidence_spread_mult: f64,

    /// Spread multiplier when all models unreliable (default: 2.0x).
    pub no_confidence_spread_mult: f64,

    /// Prior mean for IR (Bayesian shrinkage).
    pub prior_mean: f64,

    /// Prior degrees of freedom (Bayesian shrinkage strength).
    pub prior_df: f64,

    /// Number of IR bins for tracking.
    pub n_bins: usize,

    /// Exponential IR decay factor (default 0.998 ~ 500 effective samples).
    pub exp_ir_decay: f64,

    /// Exponential IR threshold below which model is degrading (default 0.3).
    /// If exp IR drops below this while flat IR is still above threshold,
    /// prefer the exp IR (faster regime shift detection).
    pub exp_ir_threshold: f64,
}

impl Default for ModelGatingConfig {
    fn default() -> Self {
        Self {
            min_samples: 500,
            ir_threshold: 0.5,
            high_confidence_threshold: 0.7,
            low_confidence_threshold: 0.3,
            low_confidence_spread_mult: 1.5,
            no_confidence_spread_mult: 2.0,
            prior_mean: 0.5, // Neutral prior; let data drive confidence
            prior_df: 10.0,  // Moderate prior strength
            n_bins: 10,
            exp_ir_decay: 0.998,
            exp_ir_threshold: 0.3,
        }
    }
}

/// Model confidence levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelConfidence {
    /// High confidence - model is adding value.
    High,
    /// Medium confidence - model may be adding value.
    Medium,
    /// Low confidence - model may be adding noise.
    Low,
    /// No confidence - insufficient data or model is harmful.
    None,
}

impl ModelConfidence {
    /// Get weight for this confidence level.
    pub fn weight(&self) -> f64 {
        match self {
            ModelConfidence::High => 1.0,
            ModelConfidence::Medium => 0.7,
            ModelConfidence::Low => 0.3,
            ModelConfidence::None => 0.0,
        }
    }
}

/// Individual model IR tracker.
#[derive(Debug)]
pub struct ModelTracker {
    #[allow(dead_code)]
    name: String,
    ir_tracker: InformationRatioTracker,
    exp_ir_tracker: ExponentialIRTracker,
    last_confidence: ModelConfidence,
    disabled: bool,
}

impl ModelTracker {
    fn new(name: &str, n_bins: usize, exp_ir_decay: f64) -> Self {
        Self {
            name: name.to_string(),
            ir_tracker: InformationRatioTracker::new(n_bins),
            exp_ir_tracker: ExponentialIRTracker::new(exp_ir_decay),
            last_confidence: ModelConfidence::None,
            disabled: false,
        }
    }

    fn update(&mut self, predicted: f64, outcome: bool) {
        self.ir_tracker.update(predicted, outcome);
        // For exp IR: outcome=true -> 1.0, outcome=false -> 0.0
        self.exp_ir_tracker
            .update(predicted, if outcome { 1.0 } else { 0.0 });
    }

    fn confidence(&self, config: &ModelGatingConfig) -> ModelConfidence {
        if self.disabled {
            return ModelConfidence::None;
        }

        if !self.ir_tracker.is_reliable(config.min_samples) {
            return ModelConfidence::None;
        }

        let p_above = self.ir_tracker.posterior_prob_ir_above(
            config.ir_threshold,
            config.prior_mean,
            config.prior_df,
        );

        // Check exponential IR for early degradation detection.
        // If exp IR drops below threshold while flat IR is still above,
        // downgrade confidence to detect regime shifts faster.
        let exp_ir = self.exp_ir_tracker.information_ratio();
        let exp_degraded = self.exp_ir_tracker.effective_sample_size() >= 50.0
            && exp_ir.abs() < config.exp_ir_threshold;

        if exp_degraded {
            // Exp IR detects degradation — cap at Medium regardless of flat IR
            if p_above >= config.low_confidence_threshold {
                return ModelConfidence::Medium;
            } else {
                return ModelConfidence::Low;
            }
        }

        if p_above >= config.high_confidence_threshold {
            ModelConfidence::High
        } else if p_above >= config.low_confidence_threshold {
            ModelConfidence::Medium
        } else {
            ModelConfidence::Low
        }
    }
}

/// Model weights for blending.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelWeights {
    /// Weight for adverse selection model.
    pub adverse_selection: f64,
    /// Weight for informed flow model.
    pub informed_flow: f64,
    /// Weight for lead-lag signal.
    pub lead_lag: f64,
    /// Weight for regime detection.
    pub regime: f64,
    /// Weight for kappa estimation.
    pub kappa: f64,
}

impl ModelWeights {
    /// All models at full weight.
    pub fn full() -> Self {
        Self {
            adverse_selection: 1.0,
            informed_flow: 1.0,
            lead_lag: 1.0,
            regime: 1.0,
            kappa: 1.0,
        }
    }

    /// All models disabled.
    pub fn disabled() -> Self {
        Self::default()
    }

    /// Overall confidence (minimum of all weights).
    pub fn overall_confidence(&self) -> f64 {
        self.adverse_selection
            .min(self.informed_flow)
            .min(self.lead_lag)
            .min(self.regime)
            .min(self.kappa)
    }

    /// Average weight across all models.
    pub fn average_weight(&self) -> f64 {
        (self.adverse_selection + self.informed_flow + self.lead_lag + self.regime + self.kappa)
            / 5.0
    }
}

/// Model gating system.
///
/// Tracks IR for each model component and provides confidence-weighted
/// model weights and spread adjustments.
#[derive(Debug)]
pub struct ModelGating {
    config: ModelGatingConfig,

    /// Adverse selection model IR tracker.
    as_tracker: ModelTracker,

    /// Informed flow model IR tracker.
    informed_flow_tracker: ModelTracker,

    /// Lead-lag signal IR tracker.
    lead_lag_tracker: ModelTracker,

    /// Regime detection IR tracker.
    regime_tracker: ModelTracker,

    /// Kappa estimation IR tracker.
    kappa_tracker: ModelTracker,

    /// Cached model weights.
    cached_weights: ModelWeights,

    /// Last update time (for logging).
    updates_since_log: usize,

    /// Total observations received (for cold-start gating).
    observation_count: usize,
}

impl ModelGating {
    /// Create a new model gating system.
    pub fn new(config: ModelGatingConfig) -> Self {
        let n_bins = config.n_bins;
        let decay = config.exp_ir_decay;
        Self {
            config,
            as_tracker: ModelTracker::new("adverse_selection", n_bins, decay),
            informed_flow_tracker: ModelTracker::new("informed_flow", n_bins, decay),
            lead_lag_tracker: ModelTracker::new("lead_lag", n_bins, decay),
            regime_tracker: ModelTracker::new("regime", n_bins, decay),
            kappa_tracker: ModelTracker::new("kappa", n_bins, decay),
            cached_weights: ModelWeights::full(),
            updates_since_log: 0,
            observation_count: 0,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(ModelGatingConfig::default())
    }

    // =========================================================================
    // Model Updates
    // =========================================================================

    /// Update adverse selection model with prediction and outcome.
    ///
    /// # Arguments
    /// * `predicted_prob` - P(adverse move > threshold) from model
    /// * `actual_adverse` - Whether adverse move actually occurred
    pub fn update_adverse_selection(&mut self, predicted_prob: f64, actual_adverse: bool) {
        self.as_tracker.update(predicted_prob, actual_adverse);
        self.maybe_update_weights();
    }

    /// Update informed flow model with prediction and outcome.
    ///
    /// # Arguments
    /// * `p_informed` - P(informed) from model
    /// * `was_informed` - Whether trade was actually informed (high AS)
    pub fn update_informed_flow(&mut self, p_informed: f64, was_informed: bool) {
        self.informed_flow_tracker.update(p_informed, was_informed);
        self.maybe_update_weights();
    }

    /// Update lead-lag model with prediction and outcome.
    ///
    /// # Arguments
    /// * `predicted_direction_prob` - P(price moves in predicted direction)
    /// * `correct_direction` - Whether price moved in predicted direction
    pub fn update_lead_lag(&mut self, predicted_direction_prob: f64, correct_direction: bool) {
        self.lead_lag_tracker
            .update(predicted_direction_prob, correct_direction);
        self.maybe_update_weights();
    }

    /// Update regime model with prediction and outcome.
    ///
    /// # Arguments
    /// * `predicted_regime_prob` - P(regime change)
    /// * `regime_changed` - Whether regime actually changed
    pub fn update_regime(&mut self, predicted_regime_prob: f64, regime_changed: bool) {
        self.regime_tracker
            .update(predicted_regime_prob, regime_changed);
        self.maybe_update_weights();
    }

    /// Update kappa model with prediction and outcome.
    ///
    /// # Arguments
    /// * `predicted_fill_prob` - P(fill) at given depth
    /// * `filled` - Whether order was filled
    pub fn update_kappa(&mut self, predicted_fill_prob: f64, filled: bool) {
        self.kappa_tracker.update(predicted_fill_prob, filled);
        self.maybe_update_weights();
    }

    /// Periodic weight update and logging.
    fn maybe_update_weights(&mut self) {
        self.updates_since_log += 1;
        self.observation_count += 1;

        // Update weights every 100 observations
        if self.updates_since_log >= 100 {
            self.update_weights();
            self.log_status();
            self.updates_since_log = 0;
        }
    }

    /// Update cached weights from IR trackers.
    fn update_weights(&mut self) {
        self.cached_weights = ModelWeights {
            adverse_selection: self.as_tracker.confidence(&self.config).weight(),
            informed_flow: self.informed_flow_tracker.confidence(&self.config).weight(),
            lead_lag: self.lead_lag_tracker.confidence(&self.config).weight(),
            regime: self.regime_tracker.confidence(&self.config).weight(),
            kappa: self.kappa_tracker.confidence(&self.config).weight(),
        };

        // Store last confidence for each tracker
        self.as_tracker.last_confidence = self.as_tracker.confidence(&self.config);
        self.informed_flow_tracker.last_confidence =
            self.informed_flow_tracker.confidence(&self.config);
        self.lead_lag_tracker.last_confidence = self.lead_lag_tracker.confidence(&self.config);
        self.regime_tracker.last_confidence = self.regime_tracker.confidence(&self.config);
        self.kappa_tracker.last_confidence = self.kappa_tracker.confidence(&self.config);
    }

    /// Log current status.
    fn log_status(&self) {
        let as_ir = self.as_tracker.ir_tracker.information_ratio();
        let if_ir = self.informed_flow_tracker.ir_tracker.information_ratio();
        let ll_ir = self.lead_lag_tracker.ir_tracker.information_ratio();
        let rg_ir = self.regime_tracker.ir_tracker.information_ratio();
        let kp_ir = self.kappa_tracker.ir_tracker.information_ratio();

        let as_exp_ir = self.as_tracker.exp_ir_tracker.information_ratio();
        let if_exp_ir = self
            .informed_flow_tracker
            .exp_ir_tracker
            .information_ratio();
        let ll_exp_ir = self.lead_lag_tracker.exp_ir_tracker.information_ratio();
        let rg_exp_ir = self.regime_tracker.exp_ir_tracker.information_ratio();
        let kp_exp_ir = self.kappa_tracker.exp_ir_tracker.information_ratio();

        let spread_mult = self.spread_multiplier();

        if spread_mult > 1.2 {
            warn!(
                as_ir = %format!("{:.2}", as_ir),
                as_exp_ir = %format!("{:.2}", as_exp_ir),
                as_weight = %format!("{:.1}", self.cached_weights.adverse_selection),
                informed_ir = %format!("{:.2}", if_ir),
                informed_exp_ir = %format!("{:.2}", if_exp_ir),
                informed_weight = %format!("{:.1}", self.cached_weights.informed_flow),
                lead_lag_ir = %format!("{:.2}", ll_ir),
                lead_lag_exp_ir = %format!("{:.2}", ll_exp_ir),
                lead_lag_weight = %format!("{:.1}", self.cached_weights.lead_lag),
                regime_ir = %format!("{:.2}", rg_ir),
                regime_exp_ir = %format!("{:.2}", rg_exp_ir),
                regime_weight = %format!("{:.1}", self.cached_weights.regime),
                kappa_ir = %format!("{:.2}", kp_ir),
                kappa_exp_ir = %format!("{:.2}", kp_exp_ir),
                kappa_weight = %format!("{:.1}", self.cached_weights.kappa),
                spread_mult = %format!("{:.2}x", spread_mult),
                "Model gating: LOW CONFIDENCE - widening spreads"
            );
        } else {
            info!(
                as_ir = %format!("{:.2}", as_ir),
                as_exp_ir = %format!("{:.2}", as_exp_ir),
                informed_ir = %format!("{:.2}", if_ir),
                informed_exp_ir = %format!("{:.2}", if_exp_ir),
                lead_lag_ir = %format!("{:.2}", ll_ir),
                lead_lag_exp_ir = %format!("{:.2}", ll_exp_ir),
                regime_ir = %format!("{:.2}", rg_ir),
                regime_exp_ir = %format!("{:.2}", rg_exp_ir),
                kappa_ir = %format!("{:.2}", kp_ir),
                kappa_exp_ir = %format!("{:.2}", kp_exp_ir),
                avg_weight = %format!("{:.2}", self.cached_weights.average_weight()),
                "Model gating status"
            );
        }
    }

    // =========================================================================
    // Getters
    // =========================================================================

    /// Get current model weights.
    pub fn model_weights(&self) -> ModelWeights {
        self.cached_weights
    }

    /// Get spread multiplier based on model confidence.
    ///
    /// Returns a multiplier >= 1.0 that should be applied to spreads
    /// when model confidence is low. During cold start (insufficient
    /// observations), returns the maximum defensive multiplier.
    pub fn spread_multiplier(&self) -> f64 {
        // Cold start: not enough data to trust any model
        if self.observation_count < self.config.min_samples {
            return self.config.no_confidence_spread_mult;
        }

        let avg_weight = self.cached_weights.average_weight();

        if avg_weight >= 0.7 {
            1.0 // High confidence, no adjustment
        } else if avg_weight >= 0.3 {
            // Linear interpolation between 1.0 and low_confidence_mult
            let t = (0.7 - avg_weight) / 0.4;
            1.0 + t * (self.config.low_confidence_spread_mult - 1.0)
        } else {
            self.config.no_confidence_spread_mult
        }
    }

    /// Check if a specific model should be used.
    pub fn should_use_model(&self, model: &str) -> bool {
        let weight = match model {
            "adverse_selection" => self.cached_weights.adverse_selection,
            "informed_flow" => self.cached_weights.informed_flow,
            "lead_lag" => self.cached_weights.lead_lag,
            "regime" => self.cached_weights.regime,
            "kappa" => self.cached_weights.kappa,
            _ => 0.0,
        };
        weight > 0.3
    }

    /// Get weight for a specific model.
    pub fn model_weight(&self, model: &str) -> f64 {
        match model {
            "adverse_selection" => self.cached_weights.adverse_selection,
            "informed_flow" => self.cached_weights.informed_flow,
            "lead_lag" => self.cached_weights.lead_lag,
            "regime" => self.cached_weights.regime,
            "kappa" => self.cached_weights.kappa,
            _ => 0.0,
        }
    }

    /// Get graduated weight for a model — continuous [MIN_SIGNAL_WEIGHT, 1.0].
    ///
    /// Unlike `should_use_model()` which is binary (zero or full weight),
    /// this returns a continuous weight proportional to the model's IR track record,
    /// with a floor that prevents the death spiral:
    ///   zero weight -> zero predictions -> zero IR -> zero weight
    pub fn graduated_weight(&self, model: &str) -> f64 {
        self.model_weight(model).max(MIN_SIGNAL_WEIGHT)
    }

    /// Get IR for a specific model.
    pub fn model_ir(&self, model: &str) -> f64 {
        match model {
            "adverse_selection" => self.as_tracker.ir_tracker.information_ratio(),
            "informed_flow" => self.informed_flow_tracker.ir_tracker.information_ratio(),
            "lead_lag" => self.lead_lag_tracker.ir_tracker.information_ratio(),
            "regime" => self.regime_tracker.ir_tracker.information_ratio(),
            "kappa" => self.kappa_tracker.ir_tracker.information_ratio(),
            _ => 0.0,
        }
    }

    /// Get exponential IR for a specific model.
    pub fn model_exp_ir(&self, model: &str) -> f64 {
        match model {
            "adverse_selection" => self.as_tracker.exp_ir_tracker.information_ratio(),
            "informed_flow" => self
                .informed_flow_tracker
                .exp_ir_tracker
                .information_ratio(),
            "lead_lag" => self.lead_lag_tracker.exp_ir_tracker.information_ratio(),
            "regime" => self.regime_tracker.exp_ir_tracker.information_ratio(),
            "kappa" => self.kappa_tracker.exp_ir_tracker.information_ratio(),
            _ => 0.0,
        }
    }

    /// Get sample count for a specific model.
    pub fn model_samples(&self, model: &str) -> usize {
        match model {
            "adverse_selection" => self.as_tracker.ir_tracker.n_samples(),
            "informed_flow" => self.informed_flow_tracker.ir_tracker.n_samples(),
            "lead_lag" => self.lead_lag_tracker.ir_tracker.n_samples(),
            "regime" => self.regime_tracker.ir_tracker.n_samples(),
            "kappa" => self.kappa_tracker.ir_tracker.n_samples(),
            _ => 0,
        }
    }

    /// Manually disable a model.
    pub fn disable_model(&mut self, model: &str) {
        match model {
            "adverse_selection" => self.as_tracker.disabled = true,
            "informed_flow" => self.informed_flow_tracker.disabled = true,
            "lead_lag" => self.lead_lag_tracker.disabled = true,
            "regime" => self.regime_tracker.disabled = true,
            "kappa" => self.kappa_tracker.disabled = true,
            _ => {}
        }
        self.update_weights();
    }

    /// Re-enable a model.
    pub fn enable_model(&mut self, model: &str) {
        match model {
            "adverse_selection" => self.as_tracker.disabled = false,
            "informed_flow" => self.informed_flow_tracker.disabled = false,
            "lead_lag" => self.lead_lag_tracker.disabled = false,
            "regime" => self.regime_tracker.disabled = false,
            "kappa" => self.kappa_tracker.disabled = false,
            _ => {}
        }
        self.update_weights();
    }

    /// Reset all trackers.
    pub fn reset(&mut self) {
        self.as_tracker.ir_tracker.clear();
        self.informed_flow_tracker.ir_tracker.clear();
        self.lead_lag_tracker.ir_tracker.clear();
        self.regime_tracker.ir_tracker.clear();
        self.kappa_tracker.ir_tracker.clear();
        self.as_tracker.exp_ir_tracker.reset();
        self.informed_flow_tracker.exp_ir_tracker.reset();
        self.lead_lag_tracker.exp_ir_tracker.reset();
        self.regime_tracker.exp_ir_tracker.reset();
        self.kappa_tracker.exp_ir_tracker.reset();
        self.cached_weights = ModelWeights::full();
        self.updates_since_log = 0;
        self.observation_count = 0;
    }
}

/// Informed flow spread adjustment.
///
/// Adjusts spreads based on P(informed) from the flow decomposition model.
#[derive(Debug, Clone, Copy)]
pub struct InformedFlowAdjustment {
    /// Threshold for widening spreads (default: 0.2).
    pub widen_threshold: f64,
    /// Threshold for tightening spreads (default: 0.05).
    pub tighten_threshold: f64,
    /// Maximum spread multiplier when P(informed) is high.
    pub max_widen_mult: f64,
    /// Minimum spread multiplier when P(informed) is low.
    pub min_tighten_mult: f64,
}

impl Default for InformedFlowAdjustment {
    fn default() -> Self {
        Self {
            widen_threshold: 0.2,
            tighten_threshold: 0.05,
            max_widen_mult: 1.5,
            min_tighten_mult: 1.0,
        }
    }
}

impl InformedFlowAdjustment {
    /// Compute spread multiplier from P(informed).
    ///
    /// - P(informed) > widen_threshold: widen spreads
    /// - P(informed) < tighten_threshold: tighten spreads
    /// - Otherwise: no adjustment
    pub fn spread_multiplier(&self, p_informed: f64) -> f64 {
        if p_informed > self.widen_threshold {
            // Linear interpolation from 1.0 to max_widen_mult
            let t = (p_informed - self.widen_threshold) / (1.0 - self.widen_threshold);
            1.0 + t * (self.max_widen_mult - 1.0)
        } else if p_informed < self.tighten_threshold {
            // Linear interpolation from min_tighten_mult to 1.0
            let t = p_informed / self.tighten_threshold;
            self.min_tighten_mult + t * (1.0 - self.min_tighten_mult)
        } else {
            1.0
        }
    }

    /// Compute asymmetric spread adjustment.
    ///
    /// When P(informed) is high on one side, widen that side more.
    ///
    /// Returns (bid_mult, ask_mult).
    pub fn asymmetric_multiplier(
        &self,
        p_informed: f64,
        flow_imbalance: f64, // Positive = buying pressure
    ) -> (f64, f64) {
        let base_mult = self.spread_multiplier(p_informed);

        if p_informed < self.widen_threshold {
            // No asymmetry when not widening
            return (base_mult, base_mult);
        }

        // Widen the side that faces the informed flow more
        // flow_imbalance > 0 = buying pressure = widen asks more
        let asymmetry = flow_imbalance.clamp(-0.5, 0.5);

        let bid_mult = base_mult * (1.0 - asymmetry * 0.3);
        let ask_mult = base_mult * (1.0 + asymmetry * 0.3);

        (bid_mult.max(1.0), ask_mult.max(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_gating_default() {
        let gating = ModelGating::default_config();
        let weights = gating.model_weights();

        // Should start with full weights (optimistic)
        assert_eq!(weights.adverse_selection, 1.0);
        assert_eq!(weights.informed_flow, 1.0);
    }

    #[test]
    fn test_spread_multiplier_high_confidence() {
        // Use a small min_samples so we can pass cold-start without
        // triggering weight recalculation (which needs all 5 trackers fed)
        let config = ModelGatingConfig {
            min_samples: 10,
            ..ModelGatingConfig::default()
        };
        let mut gating = ModelGating::new(config);

        // Feed enough observations to pass cold-start gate (< 100 avoids weight recalc)
        for _ in 0..10 {
            gating.update_adverse_selection(0.5, true);
        }

        let mult = gating.spread_multiplier();

        // Weights are still full (no weight recalc yet), so no spread widening
        assert!(
            (mult - 1.0).abs() < 0.01,
            "Expected ~1.0 spread mult with full weights, got {mult}"
        );
    }

    #[test]
    fn test_spread_multiplier_cold_start_defensive() {
        let gating = ModelGating::default_config();

        // Cold start: zero observations should return max defensive multiplier
        let mult = gating.spread_multiplier();
        assert!(
            (mult - gating.config.no_confidence_spread_mult).abs() < 0.001,
            "Cold start should return no_confidence_spread_mult (2.0), got {mult}"
        );

        // Feed some data but less than min_samples - still defensive
        let mut gating2 = ModelGating::default_config();
        for _ in 0..10 {
            gating2.update_adverse_selection(0.5, true);
        }
        let mult2 = gating2.spread_multiplier();
        assert!(
            (mult2 - gating2.config.no_confidence_spread_mult).abs() < 0.001,
            "Should still be defensive with only 10 observations, got {mult2}"
        );
    }

    #[test]
    fn test_model_confidence_levels() {
        assert_eq!(ModelConfidence::High.weight(), 1.0);
        assert_eq!(ModelConfidence::Medium.weight(), 0.7);
        assert_eq!(ModelConfidence::Low.weight(), 0.3);
        assert_eq!(ModelConfidence::None.weight(), 0.0);
    }

    #[test]
    fn test_informed_flow_adjustment_widen() {
        let adj = InformedFlowAdjustment::default();

        // High P(informed) should widen
        let mult = adj.spread_multiplier(0.5);
        assert!(mult > 1.0);
        assert!(mult <= adj.max_widen_mult);
    }

    #[test]
    fn test_informed_flow_adjustment_tighten() {
        let adj = InformedFlowAdjustment::default();

        // Low P(informed) should NOT tighten (min_tighten_mult = 1.0 disables tightening)
        let mult = adj.spread_multiplier(0.01);
        assert!(mult >= 1.0);
        assert!(mult >= adj.min_tighten_mult);
    }

    #[test]
    fn test_informed_flow_adjustment_neutral() {
        let adj = InformedFlowAdjustment::default();

        // Medium P(informed) should be neutral
        let mult = adj.spread_multiplier(0.1);
        assert!((mult - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_asymmetric_multiplier() {
        let adj = InformedFlowAdjustment::default();

        // High P(informed) with buying pressure should widen asks more
        let (bid_mult, ask_mult) = adj.asymmetric_multiplier(0.5, 0.5);
        assert!(ask_mult > bid_mult);

        // High P(informed) with selling pressure should widen bids more
        let (bid_mult, ask_mult) = adj.asymmetric_multiplier(0.5, -0.5);
        assert!(bid_mult > ask_mult);
    }

    #[test]
    fn test_model_weights_overall_confidence() {
        let weights = ModelWeights {
            adverse_selection: 1.0,
            informed_flow: 0.5,
            lead_lag: 0.3,
            regime: 0.8,
            kappa: 0.9,
        };

        // Overall confidence is the minimum
        assert!((weights.overall_confidence() - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_model_weights_average() {
        let weights = ModelWeights {
            adverse_selection: 1.0,
            informed_flow: 0.5,
            lead_lag: 0.5,
            regime: 0.5,
            kappa: 0.5,
        };

        // Average should be 0.6
        assert!((weights.average_weight() - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_graduated_weight_floor() {
        let gating = ModelGating::new(ModelGatingConfig::default());
        // Cold start: all weights should be at floor, not zero
        assert!(gating.graduated_weight("lead_lag") >= MIN_SIGNAL_WEIGHT);
        assert!(gating.graduated_weight("informed_flow") >= MIN_SIGNAL_WEIGHT);
        assert!(gating.graduated_weight("adverse_selection") >= MIN_SIGNAL_WEIGHT);
        assert!(gating.graduated_weight("regime") >= MIN_SIGNAL_WEIGHT);
        assert!(gating.graduated_weight("kappa") >= MIN_SIGNAL_WEIGHT);
        // Unknown model also gets floor (not zero)
        assert!(gating.graduated_weight("nonexistent") >= MIN_SIGNAL_WEIGHT);
    }

    #[test]
    fn test_graduated_weight_proportional_to_ir() {
        let config = ModelGatingConfig {
            min_samples: 50, // Lower threshold so 100 updates builds confidence
            ..ModelGatingConfig::default()
        };
        let mut gating = ModelGating::new(config);
        // Feed positive IR data for lead_lag — need 100 to trigger weight update
        for _ in 0..100 {
            gating.update_lead_lag(1.0, true);
        }

        let weight = gating.graduated_weight("lead_lag");
        assert!(
            weight > MIN_SIGNAL_WEIGHT,
            "weight with positive IR should exceed floor: {}",
            weight
        );
    }
}
