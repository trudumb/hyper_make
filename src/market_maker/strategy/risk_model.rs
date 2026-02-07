//! Calibrated risk model using log-additive gamma.
//!
//! This module replaces the multiplicative gamma explosion problem with a principled
//! log-additive approach:
//!
//! ```text
//! log(γ) = log(γ_base) + Σ βᵢ × xᵢ
//! γ = exp(log_gamma).clamp(γ_min, γ_max)
//! ```
//!
//! ## Why Log-Additive?
//!
//! - No explosion: sum of bounded terms instead of product
//! - Coefficients calibrated via regression: `realized_as_bps ~ features`
//! - Each β has economic interpretation (additional log-gamma per unit risk)
//!
//! ## Calibration State Machine
//!
//! ```text
//! Cold → Warming(n/100) → Calibrated(r²) → Stale(hours)
//! ```

use serde::{Deserialize, Serialize};

use super::MarketParams;

/// Calibration state for the risk model.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CalibrationState {
    /// No samples yet, using conservative defaults
    #[default]
    Cold,
    /// Collecting samples, not yet enough for calibration
    Warming { samples: usize, required: usize },
    /// Model is calibrated with sufficient data
    Calibrated { r_squared: f64 },
    /// Calibration is stale, blending toward defaults
    Stale { hours_since_calibration: f64 },
}


/// Calibrated risk model using log-additive gamma.
///
/// Replaces the multiplicative scalar explosion with principled log-space addition:
/// ```text
/// log(γ) = log(γ_base) + β_vol × excess_vol + β_tox × toxicity + ...
/// γ = exp(log_gamma).clamp(γ_min, γ_max)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibratedRiskModel {
    /// Base log-gamma (ln(γ_base))
    pub log_gamma_base: f64,

    // === Calibrated coefficients (from regression) ===
    /// Per unit excess_volatility: (σ - σ_baseline) / σ_baseline
    pub beta_volatility: f64,

    /// Per unit toxicity_score [0, 1]
    pub beta_toxicity: f64,

    /// Per unit inventory_fraction: |pos| / max_pos [0, 1]
    pub beta_inventory: f64,

    /// Per unit excess_intensity: (κ_activity - κ_baseline) / κ_baseline
    pub beta_hawkes: f64,

    /// Per unit depth_depletion: 1 - depth/depth_baseline [0, 1]
    pub beta_book_depth: f64,

    /// Per unit model_uncertainty from kappa_ci_width [0, 1]
    pub beta_uncertainty: f64,

    /// Per unit position_direction_confidence [0, 1]
    /// NEGATIVE coefficient: high confidence → LOWER gamma → more two-sided quoting
    /// confidence=1 → exp(-0.4) ≈ 0.67× gamma
    /// This replaces magic threshold logic in quote_gate with principled gamma modulation
    pub beta_confidence: f64,

    // === Bounds ===
    /// Minimum gamma (floor)
    pub gamma_min: f64,

    /// Maximum gamma (ceiling)
    pub gamma_max: f64,

    // === Calibration metadata ===
    /// Number of samples used in calibration
    pub n_samples: usize,

    /// Timestamp of last calibration (milliseconds since epoch)
    pub last_calibration_ms: u64,

    /// R-squared of the calibration fit
    pub r_squared: f64,

    /// Current calibration state
    #[serde(skip)]
    pub state: CalibrationState,
}

impl Default for CalibratedRiskModel {
    fn default() -> Self {
        Self {
            // ln(0.15) ≈ -1.897
            log_gamma_base: 0.15_f64.ln(),

            // Conservative defaults until calibrated:
            // 100% excess vol → exp(1.0) ≈ 2.7× gamma
            beta_volatility: 1.0,
            // toxicity=1 → exp(0.5) ≈ 1.65× gamma
            beta_toxicity: 0.5,
            // full inventory → exp(0.3) ≈ 1.35× gamma
            beta_inventory: 0.3,
            // 100% excess intensity → exp(0.4) ≈ 1.5× gamma
            beta_hawkes: 0.4,
            // empty book → exp(0.3) ≈ 1.35× gamma
            beta_book_depth: 0.3,
            // full uncertainty → exp(0.2) ≈ 1.2× gamma
            beta_uncertainty: 0.2,
            // NEGATIVE: high confidence → lower gamma (more two-sided quoting)
            // confidence=1 → exp(-0.4) ≈ 0.67× gamma
            beta_confidence: -0.4,

            gamma_min: 0.05,
            gamma_max: 5.0,

            n_samples: 0,
            last_calibration_ms: 0,
            r_squared: 0.0,
            state: CalibrationState::Cold,
        }
    }
}

impl CalibratedRiskModel {
    /// Create a new calibrated risk model with default conservative coefficients.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom base gamma.
    pub fn with_gamma_base(gamma_base: f64) -> Self {
        Self {
            log_gamma_base: gamma_base.max(0.01).ln(),
            ..Default::default()
        }
    }

    /// Create conservative defaults for warmup (50% higher coefficients).
    pub fn conservative_defaults() -> Self {
        Self {
            log_gamma_base: 0.20_f64.ln(), // Higher base during warmup
            beta_volatility: 1.5,          // 50% more conservative
            beta_toxicity: 0.75,
            beta_inventory: 0.45,
            beta_hawkes: 0.6,
            beta_book_depth: 0.45,
            beta_uncertainty: 0.3,
            // Less negative during warmup (more cautious about confidence)
            beta_confidence: -0.2,
            ..Default::default()
        }
    }

    /// Compute gamma from risk features using log-additive model.
    ///
    /// Formula:
    /// ```text
    /// log(γ) = log(γ_base) + Σ βᵢ × xᵢ
    /// γ = exp(log_gamma).clamp(γ_min, γ_max)
    /// ```
    ///
    /// Note: beta_confidence is NEGATIVE, so high confidence DECREASES gamma,
    /// leading to tighter two-sided quotes when position is from informed flow.
    pub fn compute_gamma(&self, features: &RiskFeatures) -> f64 {
        let log_gamma = self.log_gamma_base
            + self.beta_volatility * features.excess_volatility
            + self.beta_toxicity * features.toxicity_score
            + self.beta_inventory * features.inventory_fraction
            + self.beta_hawkes * features.excess_intensity
            + self.beta_book_depth * features.depth_depletion
            + self.beta_uncertainty * features.model_uncertainty
            + self.beta_confidence * features.position_direction_confidence;

        log_gamma.exp().clamp(self.gamma_min, self.gamma_max)
    }

    /// Check if the model is calibrated (has enough samples and recent data).
    pub fn is_calibrated(&self) -> bool {
        matches!(self.state, CalibrationState::Calibrated { .. })
    }

    /// Check if calibration is stale.
    pub fn is_stale(&self, current_ms: u64, staleness_hours: f64) -> bool {
        if self.last_calibration_ms == 0 {
            return true;
        }
        let hours_elapsed =
            (current_ms.saturating_sub(self.last_calibration_ms)) as f64 / (3600.0 * 1000.0);
        hours_elapsed > staleness_hours
    }

    /// Update calibration state based on current conditions.
    pub fn update_state(&mut self, current_ms: u64, staleness_hours: f64, min_samples: usize) {
        if self.n_samples < min_samples {
            self.state = CalibrationState::Warming {
                samples: self.n_samples,
                required: min_samples,
            };
        } else if self.is_stale(current_ms, staleness_hours) {
            let hours = (current_ms.saturating_sub(self.last_calibration_ms)) as f64
                / (3600.0 * 1000.0);
            self.state = CalibrationState::Stale {
                hours_since_calibration: hours,
            };
        } else {
            self.state = CalibrationState::Calibrated {
                r_squared: self.r_squared,
            };
        }
    }

    /// Get warmup progress [0.0, 1.0].
    pub fn warmup_progress(&self, min_samples: usize) -> f64 {
        if min_samples == 0 {
            return 1.0;
        }
        (self.n_samples as f64 / min_samples as f64).min(1.0)
    }

    /// Blend this model with conservative defaults based on staleness.
    ///
    /// Returns a new model with blended coefficients.
    pub fn blend_with_defaults(&self, blend_factor: f64) -> Self {
        let defaults = Self::conservative_defaults();
        let alpha = blend_factor.clamp(0.0, 1.0);

        Self {
            log_gamma_base: self.log_gamma_base * (1.0 - alpha) + defaults.log_gamma_base * alpha,
            beta_volatility: self.beta_volatility * (1.0 - alpha)
                + defaults.beta_volatility * alpha,
            beta_toxicity: self.beta_toxicity * (1.0 - alpha) + defaults.beta_toxicity * alpha,
            beta_inventory: self.beta_inventory * (1.0 - alpha) + defaults.beta_inventory * alpha,
            beta_hawkes: self.beta_hawkes * (1.0 - alpha) + defaults.beta_hawkes * alpha,
            beta_book_depth: self.beta_book_depth * (1.0 - alpha)
                + defaults.beta_book_depth * alpha,
            beta_uncertainty: self.beta_uncertainty * (1.0 - alpha)
                + defaults.beta_uncertainty * alpha,
            beta_confidence: self.beta_confidence * (1.0 - alpha)
                + defaults.beta_confidence * alpha,
            gamma_min: self.gamma_min,
            gamma_max: self.gamma_max,
            n_samples: self.n_samples,
            last_calibration_ms: self.last_calibration_ms,
            r_squared: self.r_squared * (1.0 - alpha),
            state: self.state,
        }
    }
}

/// Normalized risk features (single source of truth for each risk factor).
///
/// Each feature is normalized to a standard range for consistent coefficient interpretation:
/// - Excess features: (current - baseline) / baseline, clamped to [-1, 3]
/// - Fraction features: [0, 1]
/// - Score features: [0, 1]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RiskFeatures {
    /// Excess volatility: (σ - σ_baseline) / σ_baseline, clamped [-1, 3]
    pub excess_volatility: f64,

    /// Toxicity score [0, 1] from mixture model or jump ratio
    pub toxicity_score: f64,

    /// Inventory fraction: |pos| / max_pos, clamped [0, 1]
    pub inventory_fraction: f64,

    /// Excess intensity: (κ_activity - κ_baseline) / κ_baseline, clamped [-1, 3]
    pub excess_intensity: f64,

    /// Depth depletion: 1 - depth/depth_baseline, clamped [0, 1]
    pub depth_depletion: f64,

    /// Model uncertainty from kappa_ci_width, normalized [0, 1]
    pub model_uncertainty: f64,

    /// Position direction confidence [0, 1].
    /// High confidence (>0.5) = position likely from informed flow.
    /// Based on: fill alignment, belief drift alignment, time held without adverse move.
    /// When high, beta_confidence (negative) REDUCES gamma → more two-sided quoting.
    /// When low, gamma stays high → natural urgency to reduce position.
    pub position_direction_confidence: f64,
}

impl RiskFeatures {
    /// Build risk features from MarketParams and RiskModelConfig.
    pub fn from_params(
        params: &MarketParams,
        position: f64,
        max_position: f64,
        config: &RiskModelConfig,
    ) -> Self {
        // === Excess Volatility ===
        // (σ - σ_baseline) / σ_baseline
        let excess_volatility = if config.sigma_baseline > 1e-9 {
            ((params.sigma_effective - config.sigma_baseline) / config.sigma_baseline)
                .clamp(-1.0, 3.0)
        } else {
            0.0
        };

        // === Toxicity Score ===
        // Use the soft toxicity score from mixture model, or derive from jump_ratio
        let toxicity_score = if params.toxicity_score > 0.0 {
            params.toxicity_score.clamp(0.0, 1.0)
        } else {
            // Fallback: convert jump_ratio to [0, 1] score
            // jump_ratio=1 → 0, jump_ratio=2 → 0.5, jump_ratio=3+ → 1.0
            ((params.jump_ratio - 1.0) / 2.0).clamp(0.0, 1.0)
        };

        // === Inventory Fraction ===
        let inventory_fraction = if max_position > 1e-9 {
            (position.abs() / max_position).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // === Excess Intensity (Hawkes) ===
        // Use hawkes_activity_percentile as a proxy for excess intensity
        // percentile=0.5 (median) → 0, percentile=1.0 → 1.0
        let excess_intensity = ((params.hawkes_activity_percentile - 0.5) * 2.0).clamp(-1.0, 3.0);

        // === Depth Depletion ===
        // 1 - depth/depth_baseline
        let depth_depletion = if config.book_depth_baseline_usd > 0.0 {
            (1.0 - params.near_touch_depth_usd / config.book_depth_baseline_usd).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // === Model Uncertainty ===
        // Normalize kappa_ci_width to [0, 1]
        // CI width of 0.3 (converged) → 0.1, CI width of 3.0 → 1.0
        let model_uncertainty = (params.kappa_ci_width / 3.0).clamp(0.0, 1.0);

        // === Position Direction Confidence ===
        // Use pre-computed value from MarketParams if available, else compute from params
        let position_direction_confidence = if params.position_direction_confidence > 0.01
            && params.position_direction_confidence < 0.99
        {
            // Use pre-computed value
            params.position_direction_confidence
        } else {
            // Compute on-the-fly using the MarketParams method
            params.compute_position_direction_confidence(position, max_position)
        };

        Self {
            excess_volatility,
            toxicity_score,
            inventory_fraction,
            excess_intensity,
            depth_depletion,
            model_uncertainty,
            position_direction_confidence,
        }
    }

    /// Create features with all values at their neutral/safe defaults.
    pub fn neutral() -> Self {
        Self {
            excess_volatility: 0.0,
            toxicity_score: 0.0,
            inventory_fraction: 0.0,
            excess_intensity: 0.0,
            depth_depletion: 0.0,
            model_uncertainty: 0.0,
            position_direction_confidence: 0.5, // Neutral confidence
        }
    }

    /// Create features representing extreme risk (for testing bounds).
    pub fn extreme() -> Self {
        Self {
            excess_volatility: 3.0,
            toxicity_score: 1.0,
            inventory_fraction: 1.0,
            excess_intensity: 3.0,
            depth_depletion: 1.0,
            model_uncertainty: 1.0,
            position_direction_confidence: 0.0, // No confidence → high gamma
        }
    }

    /// Build risk features from MarketState (for use in LearningModule calibration).
    ///
    /// Takes config to ensure baselines match those used in `from_params()`.
    /// This is critical for calibration consistency.
    pub fn from_state(
        state: &crate::market_maker::learning::MarketState,
        config: &RiskModelConfig,
    ) -> Self {
        // Use config baselines for consistency with from_params()
        let sigma_baseline = config.sigma_baseline;
        let kappa_baseline = config.kappa_baseline;

        // Excess volatility from sigma_effective
        let excess_volatility = if sigma_baseline > 1e-9 {
            ((state.sigma_effective - sigma_baseline) / sigma_baseline).clamp(-1.0, 3.0)
        } else {
            0.0
        };

        // Toxicity from toxicity_score or jump_ratio
        let toxicity_score = if state.toxicity_score > 0.0 {
            state.toxicity_score.clamp(0.0, 1.0)
        } else {
            ((state.jump_ratio - 1.0) / 2.0).clamp(0.0, 1.0)
        };

        // Inventory fraction
        let inventory_fraction = if state.max_position > 1e-9 {
            (state.position.abs() / state.max_position).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Excess intensity from kappa (higher kappa = more activity)
        // Use kappa relative to baseline
        let excess_intensity = if kappa_baseline > 1e-9 {
            ((state.kappa - kappa_baseline) / kappa_baseline).clamp(-1.0, 3.0)
        } else {
            0.0
        };

        // Depth depletion - not available in MarketState, use 0
        let depth_depletion = 0.0;

        // Model uncertainty - not directly in MarketState, approximate from p_informed
        // Higher p_informed suggests more uncertainty about flow
        let model_uncertainty = state.p_informed.clamp(0.0, 1.0);

        // Position direction confidence - not available in MarketState, use neutral
        // This is only used in live trading context anyway
        let position_direction_confidence = 0.5;

        Self {
            excess_volatility,
            toxicity_score,
            inventory_fraction,
            excess_intensity,
            depth_depletion,
            model_uncertainty,
            position_direction_confidence,
        }
    }
}

/// Configuration for the calibrated risk model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskModelConfig {
    /// Feature flag: use log-additive calibrated model
    pub use_calibrated_risk_model: bool,

    /// Blend factor for gradual rollout (0=old multiplicative, 1=new log-additive)
    pub risk_model_blend: f64,

    /// Baseline volatility for feature normalization (per-second σ)
    pub sigma_baseline: f64,

    /// Baseline kappa for feature normalization
    pub kappa_baseline: f64,

    /// Baseline book depth for feature normalization (USD)
    pub book_depth_baseline_usd: f64,

    /// Minimum samples before using calibrated coefficients
    pub min_calibration_samples: usize,

    /// Hours after which calibration is considered stale
    pub calibration_staleness_hours: f64,
}

impl Default for RiskModelConfig {
    fn default() -> Self {
        Self {
            use_calibrated_risk_model: false, // Conservative: disabled by default
            risk_model_blend: 0.0,            // Start with old model
            sigma_baseline: 0.0002,           // 2 bps per √second
            kappa_baseline: 2500.0,           // Prior for liquid markets
            book_depth_baseline_usd: 100_000.0, // $100k baseline
            min_calibration_samples: 100,
            calibration_staleness_hours: 4.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_additive_gamma_bounded() {
        let model = CalibratedRiskModel::default();

        // Extreme features should still produce bounded gamma
        let extreme = RiskFeatures::extreme();
        let gamma = model.compute_gamma(&extreme);

        assert!(gamma <= 10.0, "Gamma should not explode: got {}", gamma);
        assert!(gamma >= 0.01, "Gamma should not collapse: got {}", gamma);
    }

    #[test]
    fn test_neutral_features_give_expected_gamma() {
        let model = CalibratedRiskModel::default();
        let neutral = RiskFeatures::neutral();

        let gamma = model.compute_gamma(&neutral);

        // With neutral features:
        // - All risk features are 0
        // - position_direction_confidence = 0.5 (neutral)
        // - beta_confidence = -0.4
        // So: log(gamma) = log(0.15) + (-0.4 * 0.5) = log(0.15) - 0.2
        // gamma = exp(log(0.15) - 0.2) ≈ 0.122
        let expected = (0.15_f64.ln() - 0.4 * 0.5).exp();
        assert!(
            (gamma - expected).abs() < 0.01,
            "Neutral features should give expected gamma: got {}, expected {}",
            gamma,
            expected
        );
    }

    #[test]
    fn test_gamma_increases_with_risk() {
        let model = CalibratedRiskModel::default();
        let neutral = RiskFeatures::neutral();
        let risky = RiskFeatures {
            excess_volatility: 1.0, // Double baseline vol
            toxicity_score: 0.5,    // 50% toxicity
            inventory_fraction: 0.5,
            excess_intensity: 0.5,
            depth_depletion: 0.5,
            model_uncertainty: 0.5,
            position_direction_confidence: 0.5, // Neutral confidence
        };

        let gamma_neutral = model.compute_gamma(&neutral);
        let gamma_risky = model.compute_gamma(&risky);

        assert!(
            gamma_risky > gamma_neutral,
            "Risky features should produce higher gamma: {} vs {}",
            gamma_risky,
            gamma_neutral
        );
    }

    #[test]
    fn test_conservative_defaults_higher() {
        let normal = CalibratedRiskModel::default();
        let conservative = CalibratedRiskModel::conservative_defaults();
        let features = RiskFeatures {
            excess_volatility: 0.5,
            toxicity_score: 0.3,
            ..Default::default()
        };

        let gamma_normal = normal.compute_gamma(&features);
        let gamma_conservative = conservative.compute_gamma(&features);

        assert!(
            gamma_conservative > gamma_normal,
            "Conservative defaults should produce higher gamma"
        );
    }

    #[test]
    fn test_blend_with_defaults() {
        let model = CalibratedRiskModel::default();
        let defaults = CalibratedRiskModel::conservative_defaults();

        // Full blend should give conservative defaults
        let blended_full = model.blend_with_defaults(1.0);
        assert!(
            (blended_full.beta_volatility - defaults.beta_volatility).abs() < 0.01,
            "Full blend should match defaults"
        );

        // Zero blend should preserve original
        let blended_zero = model.blend_with_defaults(0.0);
        assert!(
            (blended_zero.beta_volatility - model.beta_volatility).abs() < 0.01,
            "Zero blend should preserve original"
        );
    }

    #[test]
    fn test_high_confidence_reduces_gamma() {
        let model = CalibratedRiskModel::default();

        // Low confidence (adverse position) → higher gamma
        let low_confidence = RiskFeatures {
            position_direction_confidence: 0.0,
            ..Default::default()
        };

        // High confidence (informed position) → lower gamma
        let high_confidence = RiskFeatures {
            position_direction_confidence: 1.0,
            ..Default::default()
        };

        let gamma_low_conf = model.compute_gamma(&low_confidence);
        let gamma_high_conf = model.compute_gamma(&high_confidence);

        // beta_confidence is NEGATIVE, so:
        // high confidence → exp(beta * 1.0) = exp(-0.4) ≈ 0.67× gamma
        // low confidence → exp(beta * 0.0) = 1.0× gamma
        assert!(
            gamma_high_conf < gamma_low_conf,
            "High confidence should REDUCE gamma: high={}, low={}",
            gamma_high_conf,
            gamma_low_conf
        );

        // The ratio should be approximately exp(-0.4) ≈ 0.67
        let ratio = gamma_high_conf / gamma_low_conf;
        let expected_ratio = (-0.4_f64).exp();
        assert!(
            (ratio - expected_ratio).abs() < 0.01,
            "Ratio should be exp(-0.4) ≈ 0.67: got {}",
            ratio
        );
    }
}
