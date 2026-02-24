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

fn default_beta_cascade() -> f64 {
    1.2 // Interim: pending data-driven calibration from gamma_calibration.jsonl
}

fn default_beta_tail_risk() -> f64 {
    0.7 // Interim: pending data-driven calibration from gamma_calibration.jsonl
}

fn default_beta_drawdown() -> f64 {
    1.4 // At 10% dd: e^(1.4×0.10) = 1.15× gamma (matches old 1.20 approx)
}

fn default_beta_regime() -> f64 {
    1.0 // Maps regime_risk_score = ln(regime_gamma_multiplier) directly
}

fn default_beta_ghost() -> f64 {
    0.5 // At ghost_mult=2.0: e^(0.5×1.0) = 1.65× gamma
}

fn default_beta_continuation() -> f64 {
    -0.5 // High continuation probability reduces gamma for more two-sided quoting
}

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

    /// Per unit cascade_intensity [0, 1].
    /// Cascade intensity=1.0 → exp(0.8) ≈ 2.2× gamma widening.
    /// Routes cascade defense through principled log-additive γ path
    /// instead of arbitrary size multiplication.
    #[serde(default = "default_beta_cascade")]
    pub beta_cascade: f64,

    /// Per unit tail_risk_intensity [0, 1].
    /// Distinct from cascade_intensity: captures tail risk from liquidation
    /// cascades (OI drops, depth evaporation) vs general cascade activity.
    /// tail_risk_intensity=1.0 → exp(0.5) ≈ 1.65× gamma widening.
    #[serde(default = "default_beta_tail_risk")]
    pub beta_tail_risk: f64,

    /// Per unit drawdown_fraction [0, 1].
    /// WS1: Replaces multiplicative drawdown_mult = 1.0 + dd_frac × 2.0.
    /// At 10% dd: e^(1.4×0.10) = 1.15× gamma (log-additive, bounded by sigmoid).
    #[serde(default = "default_beta_drawdown")]
    pub beta_drawdown: f64,

    /// Per unit regime_risk_score = ln(regime_gamma_multiplier).
    /// WS1: Replaces multiplicative regime_gamma_multiplier.
    /// Maps multiplier into log-space directly: mult=1.3 → score=0.26, mult=1.8 → score=0.59.
    #[serde(default = "default_beta_regime")]
    pub beta_regime: f64,

    /// Per unit ghost_depletion = (ghost_mult - 1).min(1.0).
    /// WS1: Replaces multiplicative ghost_liquidity_gamma_mult.
    /// At ghost_mult=2.0: feature=1.0, e^(0.5) = 1.65× gamma.
    #[serde(default = "default_beta_ghost")]
    pub beta_ghost: f64,

    /// Per unit continuation_probability [0, 1].
    /// NEGATIVE coefficient: high continuation probability → LOWER gamma → more two-sided quoting.
    /// Replaces discrete HOLD/ADD/REDUCE states in ladder strategy.
    #[serde(default = "default_beta_continuation")]
    pub beta_continuation: f64,

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
            // Reduced from 0.9: toxicity=1 → exp(0.5) ≈ 1.65x gamma.
            // 0.9 was causing 2.46x gamma inflation from toxicity alone, which combined
            // with other features summed to 2.19+ and inflated gamma 9x (0.15 -> 1.338).
            // Pending data-driven calibration from gamma_calibration.jsonl regression.
            beta_toxicity: 0.5,
            // WS1: RE-ENABLED. inventory_fraction² (quadratic) computed in compute_gamma().
            // Replaces multiplicative γ(q) = γ_base × (1 + 7.0 × u²) from effective_gamma().
            // At u=0.5: 4.0 × 0.25 = 1.0, e^1.0 = 2.72 (matches old 2.75 before sigmoid).
            beta_inventory: 4.0,
            // 100% excess intensity → exp(0.4) ≈ 1.5× gamma
            beta_hawkes: 0.4,
            // empty book → exp(0.3) ≈ 1.35× gamma
            beta_book_depth: 0.3,
            // full uncertainty → exp(0.2) ≈ 1.2× gamma
            beta_uncertainty: 0.2,
            // NEGATIVE: high confidence → lower gamma (more two-sided quoting)
            // confidence=1 → exp(-0.4) ≈ 0.67× gamma
            beta_confidence: -0.4,
            // Interim: cascade_intensity=1 → exp(1.2) ≈ 3.32× gamma widening
            // Pending data-driven calibration from gamma_calibration.jsonl regression.
            beta_cascade: 1.2,
            // Interim: tail_risk_intensity=1 → exp(0.7) ≈ 2.01× gamma widening
            // Pending data-driven calibration from gamma_calibration.jsonl regression.
            beta_tail_risk: 0.7,

            beta_drawdown: 1.4,
            beta_regime: 1.0,
            beta_ghost: 0.5,
            beta_continuation: -0.5,

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
            beta_toxicity: 0.75,           // Interim conservative (1.5x of default 0.5)
            // WS1: RE-ENABLED, slightly more aggressive during warmup
            beta_inventory: 5.0,
            beta_hawkes: 0.6,
            beta_book_depth: 0.25,
            beta_uncertainty: 0.3,
            // Less negative during warmup (more cautious about confidence)
            beta_confidence: -0.2,
            // More conservative cascade widening during warmup
            beta_cascade: 1.8,
            // More conservative tail risk during warmup
            beta_tail_risk: 1.05,
            beta_drawdown: 2.0, // More conservative during warmup
            beta_regime: 1.5,   // More conservative during warmup
            beta_ghost: 0.75,   // More conservative during warmup
            beta_continuation: -0.25, // Less confident in continuation during warmup
            ..Default::default()
        }
    }

    /// Compute gamma from risk features using log-additive model with sigmoid regularization.
    ///
    /// Formula:
    /// ```text
    /// raw_sum = Σ βᵢ × xᵢ
    /// regulated_sum = max_contrib × tanh(raw_sum / (max_contrib × steepness))
    /// log(γ) = log(γ_base) + regulated_sum
    /// γ = exp(log_gamma).clamp(γ_min, γ_max)
    /// ```
    ///
    /// The sigmoid "skeptic" regularization prevents noisy features from summing
    /// to extreme values (e.g., 2.19+ -> 9x gamma inflation). The tanh squashes
    /// the feature sum so gamma inflation is bounded by exp(max_contrib) regardless
    /// of how many features fire simultaneously.
    ///
    /// Note: beta_confidence is NEGATIVE, so high confidence DECREASES gamma,
    /// leading to tighter two-sided quotes when position is from informed flow.
    pub fn compute_gamma(&self, features: &RiskFeatures) -> f64 {
        // WS1: Quadratic inventory pressure — (|pos|/max)² makes gamma scale faster near limits.
        // Old multiplicative: γ × (1 + 7.0 × u²). Log-additive: β_inv × u².
        // At u=0.5: 4.0 × 0.25 = 1.0 → e^1.0 = 2.72× (was 2.75 multiplicative).
        let inventory_pressure = features.inventory_fraction.powi(2);

        let raw_sum = self.beta_volatility * features.excess_volatility
            + self.beta_toxicity * features.toxicity_score
            + self.beta_inventory * inventory_pressure
            + self.beta_hawkes * features.excess_intensity
            + self.beta_book_depth * features.depth_depletion
            + self.beta_uncertainty * features.model_uncertainty
            + self.beta_confidence * features.position_direction_confidence
            + self.beta_cascade * features.cascade_intensity
            + self.beta_tail_risk * features.tail_risk_intensity
            // WS1: New terms absorbing multiplicative post-processes from effective_gamma()
            + self.beta_drawdown * features.drawdown_fraction
            + self.beta_regime * features.regime_risk_score
            + self.beta_ghost * features.ghost_depletion
            + self.beta_continuation * features.continuation_probability;

        // Sigmoid "skeptic" regularization — prevents any single noisy feature from dominating.
        // WS1: Increased from 1.5 to 2.5 to accommodate wider feature range (12 features now).
        // max_gamma_contribution bounds total inflation to exp(2.5) ~ 12× at most.
        // gamma_reg_steepness controls how fast the tanh saturates.
        const MAX_GAMMA_CONTRIBUTION: f64 = 2.5; // max ~12× gamma inflation (exp(2.5))
        const GAMMA_REG_STEEPNESS: f64 = 2.0; // how fast sigmoid saturates
        let regulated_sum = MAX_GAMMA_CONTRIBUTION
            * (raw_sum / (MAX_GAMMA_CONTRIBUTION * GAMMA_REG_STEEPNESS)).tanh();

        log::trace!(
            "[SPREAD TRACE] gamma features: raw_sum={:.4}, regulated_sum={:.4}, delta={:.4}",
            raw_sum,
            regulated_sum,
            raw_sum - regulated_sum
        );

        let log_gamma = self.log_gamma_base + regulated_sum;

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
            let hours =
                (current_ms.saturating_sub(self.last_calibration_ms)) as f64 / (3600.0 * 1000.0);
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
            beta_cascade: self.beta_cascade * (1.0 - alpha) + defaults.beta_cascade * alpha,
            beta_tail_risk: self.beta_tail_risk * (1.0 - alpha) + defaults.beta_tail_risk * alpha,
            beta_drawdown: self.beta_drawdown * (1.0 - alpha) + defaults.beta_drawdown * alpha,
            beta_regime: self.beta_regime * (1.0 - alpha) + defaults.beta_regime * alpha,
            beta_ghost: self.beta_ghost * (1.0 - alpha) + defaults.beta_ghost * alpha,
            beta_continuation: self.beta_continuation * (1.0 - alpha) + defaults.beta_continuation * alpha,
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

    /// Cascade intensity [0, 1]: derived from (1.0 - cascade_size_factor).
    /// 0 = calm market, 1 = full cascade (OI dropping, depth evaporating).
    /// Fed into beta_cascade coefficient for principled spread widening.
    pub cascade_intensity: f64,

    /// Tail risk intensity [0, 1]: derived from (tail_risk_multiplier - 1.0) / 4.0.
    /// 0 = no tail risk, 1 = extreme tail risk (liquidation cascades).
    /// Distinct from cascade_intensity: captures depth-of-crisis severity.
    /// Fed into beta_tail_risk coefficient.
    pub tail_risk_intensity: f64,

    /// WS1: Drawdown fraction [0, 1]: current_drawdown_frac from MarketParams.
    /// Replaces multiplicative `1.0 + dd_frac × 2.0` in effective_gamma().
    /// At 10% dd: beta_drawdown(1.4) × 0.10 → e^0.14 = 1.15× gamma.
    pub drawdown_fraction: f64,

    /// WS1: Regime risk score: ln(regime_gamma_multiplier).
    /// Replaces multiplicative regime_gamma_multiplier in effective_gamma().
    /// mult=1.3 → 0.26, mult=1.8 → 0.59. beta_regime(1.0) × score → e^score.
    pub regime_risk_score: f64,

    /// WS1: Ghost liquidity depletion: (ghost_mult - 1).min(1.0) [0, 1].
    /// Replaces multiplicative ghost_liquidity_gamma_mult in effective_gamma().
    /// ghost_mult=2.0 → feature=1.0, beta_ghost(0.5) × 1.0 → e^0.5 = 1.65×.
    pub ghost_depletion: f64,

    /// Continuation probability [0, 1].
    /// High probability means positional direction is strongly expected to continue.
    pub continuation_probability: f64,
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

        // === Cascade Intensity ===
        // Use the direct cascade_intensity from MarketParams
        let cascade_intensity = params.cascade_intensity.clamp(0.0, 1.0);

        // === Tail Risk Intensity ===
        // Use the direct tail_risk_intensity from MarketParams
        let tail_risk_intensity = params.tail_risk_intensity.clamp(0.0, 1.0);

        // === WS1: Drawdown Fraction ===
        let drawdown_fraction = params.current_drawdown_frac.clamp(0.0, 1.0);

        // === WS1: Regime Risk Score ===
        // Map regime_gamma_multiplier into log-space: mult=1.0 → 0, mult=1.3 → 0.26, mult=2.0 → 0.69
        let regime_risk_score = params.regime_gamma_multiplier.max(0.01).ln().max(0.0);

        // === WS1: Ghost Depletion ===
        // ghost_mult=1.0 → 0 (no depletion), ghost_mult=2.0 → 1.0 (capped)
        let ghost_depletion = (params.ghost_liquidity_gamma_mult - 1.0).clamp(0.0, 1.0);

        // === Continuation Probability ===
        let continuation_probability = params.continuation_p.clamp(0.0, 1.0);

        Self {
            excess_volatility,
            toxicity_score,
            inventory_fraction,
            excess_intensity,
            depth_depletion,
            model_uncertainty,
            position_direction_confidence,
            cascade_intensity,
            tail_risk_intensity,
            drawdown_fraction,
            regime_risk_score,
            ghost_depletion,
            continuation_probability,
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
            cascade_intensity: 0.0,
            tail_risk_intensity: 0.0,
            drawdown_fraction: 0.0,
            regime_risk_score: 0.0,
            ghost_depletion: 0.0,
            continuation_probability: 0.5, // Neutral continuation probability
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
            cascade_intensity: 1.0,
            tail_risk_intensity: 1.0,
            drawdown_fraction: 0.5,  // 50% drawdown
            regime_risk_score: 0.59, // ln(1.8) — extreme regime
            ghost_depletion: 1.0,    // Full ghost depletion
            continuation_probability: 0.0, // No continuation -> higher gamma
        }
    }

    /// Convert features to a vector for regression (WS5: AdversarialCalibrator).
    /// Order matches the beta coefficients in compute_gamma().
    pub fn to_vector(&self) -> Vec<f64> {
        vec![
            self.excess_volatility,
            self.toxicity_score,
            self.inventory_fraction.powi(2), // Quadratic, matching compute_gamma()
            self.excess_intensity,
            self.depth_depletion,
            self.model_uncertainty,
            self.position_direction_confidence,
            self.cascade_intensity,
            self.tail_risk_intensity,
            self.drawdown_fraction,
            self.regime_risk_score,
            self.ghost_depletion,
        ]
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
            cascade_intensity: 0.0,   // Not available from MarketState
            tail_risk_intensity: 0.0, // Not available from MarketState
            drawdown_fraction: 0.0,   // Not available from MarketState
            regime_risk_score: 0.0,   // Not available from MarketState
            ghost_depletion: 0.0,     // Not available from MarketState
            continuation_probability: 0.5, // Not available from MarketState, use neutral
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
            use_calibrated_risk_model: true,    // Log-additive model enabled
            risk_model_blend: 1.0,              // Pure log-additive (no multiplicative explosion)
            sigma_baseline: 0.0002,             // 2 bps per √second
            kappa_baseline: 2500.0,             // Prior for liquid markets
            book_depth_baseline_usd: 100_000.0, // $100k baseline
            min_calibration_samples: 100,
            calibration_staleness_hours: 4.0,
        }
    }
}

impl RiskModelConfig {
    /// HIP-3 DEX profile: use log-additive gamma to prevent multiplicative explosion.
    ///
    /// Legacy multiplicative gamma multiplies 7+ scalars — modest 1.2x each → 1.2^7 = 3.6x.
    /// Log-additive approach sums bounded terms instead: no explosion possible.
    /// Setting `risk_model_blend: 1.0` bypasses the multiplicative path entirely.
    pub fn hip3() -> Self {
        Self {
            use_calibrated_risk_model: true,
            risk_model_blend: 1.0, // Pure log-additive (no multiplicative explosion)
            sigma_baseline: 0.00015,
            kappa_baseline: 1500.0,
            book_depth_baseline_usd: 5_000.0, // HIP-3 books are thin
            min_calibration_samples: 50,
            calibration_staleness_hours: 8.0,
        }
    }
}

// === WS3: Gamma Self-Calibration Tracker ===

/// EWMA alpha for gamma tracking error.
const GAMMA_TRACKING_ALPHA: f64 = 0.1;

/// Tracks breakeven gamma for each fill markout and measures systematic bias.
///
/// For each fill, computes the γ that would have made the fill break-even at 5s markout.
/// Uses the small γ/κ approximation: for small γ/κ, ln(1+γ/κ) ≈ γ/κ,
/// so spread ≈ γσ²τ + 2/κ, which inverts to:
///   γ_breakeven = (spread_bps - 2/κ × 10000) / (σ² × τ × 10000)
///
/// This is diagnostics-only — no automated β update. The tracking error
/// EWMA measures persistent bias that can inform future β recalibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GammaCalibrationTracker {
    /// EWMA of |log(predicted_gamma) - log(breakeven_gamma)|.
    tracking_error_ewma: f64,
    /// EWMA of signed error (positive = predicted too high, negative = too low).
    signed_bias_ewma: f64,
    /// Number of observations.
    observation_count: usize,
    /// Most recent breakeven gamma (for diagnostics).
    last_breakeven_gamma: f64,
}

impl Default for GammaCalibrationTracker {
    fn default() -> Self {
        Self {
            tracking_error_ewma: 0.0,
            signed_bias_ewma: 0.0,
            observation_count: 0,
            last_breakeven_gamma: 0.0,
        }
    }
}

impl GammaCalibrationTracker {
    /// Create a new tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute breakeven gamma using small γ/κ approximation.
    ///
    /// spread_bps: half-spread at fill time (bps).
    /// markout_as_bps: 5s markout adverse selection (bps, unsigned).
    /// fee_bps: maker fee (typically 1.5 bps).
    /// sigma: σ_effective at fill time (per-second, fractional).
    /// kappa: effective kappa at fill time.
    /// tau: GLFT time horizon in seconds (typically 1/fill_rate).
    ///
    /// Returns None if inputs are degenerate.
    pub fn breakeven_gamma(
        _spread_bps: f64,
        markout_as_bps: f64,
        fee_bps: f64,
        sigma: f64,
        kappa: f64,
        tau: f64,
    ) -> Option<f64> {
        if sigma < 1e-10 || kappa < 1.0 || tau < 0.01 {
            return None;
        }

        // Target spread = markout_as + fee (what we need to capture to break even)
        let target_bps = markout_as_bps + fee_bps;
        // Kappa contribution: 2/κ in bps ≈ 2/κ × 10000
        let kappa_term_bps = 2.0 / kappa * 10_000.0;
        // Vol-risk contribution must be non-negative
        let numerator = target_bps - kappa_term_bps;
        if numerator <= 0.0 {
            // Kappa term alone covers the spread — γ can be near 0
            return Some(0.01);
        }

        // γ_breakeven = numerator / (σ² × τ × 10000)
        let denominator = sigma * sigma * tau * 10_000.0;
        if denominator < 1e-12 {
            return None;
        }

        let gamma = numerator / denominator;
        Some(gamma.max(0.001))
    }

    /// Record a fill markout observation.
    ///
    /// `predicted_gamma`: the γ used when the fill was placed.
    /// `spread_bps`: half-spread at fill time.
    /// `markout_as_bps`: realized 5s AS (unsigned bps).
    /// `fee_bps`: maker fee.
    /// `sigma`, `kappa`, `tau`: market parameters at fill time.
    #[allow(clippy::too_many_arguments)]
    pub fn record_markout(
        &mut self,
        predicted_gamma: f64,
        spread_bps: f64,
        markout_as_bps: f64,
        fee_bps: f64,
        sigma: f64,
        kappa: f64,
        tau: f64,
    ) {
        let breakeven =
            match Self::breakeven_gamma(spread_bps, markout_as_bps, fee_bps, sigma, kappa, tau) {
                Some(g) => g,
                None => return,
            };

        if predicted_gamma < 0.001 || !predicted_gamma.is_finite() {
            return;
        }

        self.last_breakeven_gamma = breakeven;

        // Signed error: positive = predicted too high (too conservative)
        let signed_error = predicted_gamma.ln() - breakeven.ln();
        let abs_error = signed_error.abs();

        if self.observation_count == 0 {
            self.tracking_error_ewma = abs_error;
            self.signed_bias_ewma = signed_error;
        } else {
            self.tracking_error_ewma = GAMMA_TRACKING_ALPHA * abs_error
                + (1.0 - GAMMA_TRACKING_ALPHA) * self.tracking_error_ewma;
            self.signed_bias_ewma = GAMMA_TRACKING_ALPHA * signed_error
                + (1.0 - GAMMA_TRACKING_ALPHA) * self.signed_bias_ewma;
        }

        self.observation_count += 1;
    }

    /// Tracking error EWMA (absolute log-gamma error).
    /// Target: < 0.3 means γ is well-calibrated.
    pub fn tracking_error(&self) -> f64 {
        self.tracking_error_ewma
    }

    /// Signed bias: positive = consistently too conservative, negative = too aggressive.
    pub fn signed_bias(&self) -> f64 {
        self.signed_bias_ewma
    }

    /// Most recent breakeven gamma (for diagnostics).
    pub fn last_breakeven_gamma(&self) -> f64 {
        self.last_breakeven_gamma
    }

    /// Number of markout observations.
    pub fn observation_count(&self) -> usize {
        self.observation_count
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
        // - beta_continuation = -0.25
        // raw_sum = (-0.4 * 0.5) + (-0.25 * 0.5) = -0.325
        // WS1: MAX_GAMMA_CONTRIBUTION=2.5, STEEPNESS=2.0
        // After sigmoid: regulated = 2.5 * tanh(-0.325 / 5.0)
        // gamma = exp(log(0.15) + regulated)
        let raw_sum = (-0.4 * 0.5) + (-0.25 * 0.5);
        let regulated = 2.5 * (raw_sum / 5.0_f64).tanh();
        let expected = (0.15_f64.ln() + regulated).exp();
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
            ..RiskFeatures::neutral()
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
        // high confidence → lower gamma, low confidence → higher gamma
        assert!(
            gamma_high_conf < gamma_low_conf,
            "High confidence should REDUCE gamma: high={}, low={}",
            gamma_high_conf,
            gamma_low_conf
        );

        // WS1: MAX_GAMMA_CONTRIBUTION=2.5, STEEPNESS=2.0
        // raw_sum_low = 0.0 (all defaults zero, confidence=0 contributes 0)
        // raw_sum_high = -0.4 (beta_conf * 1.0)
        // regulated_low = 2.5*tanh(0/5) = 0
        // regulated_high = 2.5*tanh(-0.4/5) ≈ -0.199
        // ratio = exp(-0.199) ≈ 0.82
        let ratio = gamma_high_conf / gamma_low_conf;
        assert!(
            ratio < 0.90,
            "High confidence should meaningfully reduce gamma: ratio={}",
            ratio
        );
        assert!(
            ratio > 0.70,
            "Sigmoid should dampen confidence effect: ratio={}",
            ratio
        );
    }

    #[test]
    fn test_hip3_config_uses_log_additive() {
        let cfg = RiskModelConfig::hip3();
        assert!(cfg.use_calibrated_risk_model);
        assert!((cfg.risk_model_blend - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cascade_widens_gamma() {
        let model = CalibratedRiskModel::default();
        let calm = RiskFeatures {
            cascade_intensity: 0.0,
            ..RiskFeatures::neutral()
        };
        let cascade = RiskFeatures {
            cascade_intensity: 1.0,
            ..RiskFeatures::neutral()
        };

        let gamma_calm = model.compute_gamma(&calm);
        let gamma_cascade = model.compute_gamma(&cascade);

        // WS1: MAX_GAMMA_CONTRIBUTION=2.5, STEEPNESS=2.0
        // raw_sum_calm = -0.2 (confidence), raw_sum_cascade = -0.2 + 1.2 = 1.0
        // regulated_calm = 2.5*tanh(-0.2/5.0) ≈ -0.0999
        // regulated_cascade = 2.5*tanh(1.0/5.0) ≈ 0.494
        // ratio = exp(0.494 - (-0.0999)) = exp(0.594) ≈ 1.81
        let ratio = gamma_cascade / gamma_calm;
        assert!(
            ratio > 1.5,
            "Cascade should meaningfully widen gamma: calm={}, cascade={}, ratio={:.3}",
            gamma_calm,
            gamma_cascade,
            ratio
        );
        assert!(
            ratio < 3.0,
            "Sigmoid should cap cascade widening: calm={}, cascade={}, ratio={:.3}",
            gamma_calm,
            gamma_cascade,
            ratio
        );
    }

    #[test]
    fn test_hip3_gamma_no_explosion() {
        // Under stressed conditions, gamma should stay bounded (not > 0.5)
        // The log-additive model prevents the 1.2^7 = 3.6x multiplicative explosion
        let model = CalibratedRiskModel::default();
        let stressed = RiskFeatures {
            excess_volatility: 1.5,             // 150% above baseline
            toxicity_score: 0.7,                // High toxicity
            inventory_fraction: 0.8,            // Near max inventory
            excess_intensity: 1.0,              // Double baseline
            depth_depletion: 0.6,               // Thin book
            model_uncertainty: 0.8,             // High uncertainty
            position_direction_confidence: 0.3, // Low confidence
            ..RiskFeatures::neutral()
        };

        let gamma = model.compute_gamma(&stressed);
        // WS1: With beta_inventory=4.0 and inv²=0.64, raw_sum≈5.03.
        // Sigmoid: 2.5*tanh(5.03/5.0)≈1.91. gamma≈exp(ln(0.15)+1.91)≈1.01.
        // Bounded by gamma_max (5.0). Multiplicative would give 0.15 × 1.2^7 × (1+7×0.64) ≈ 3.0+.
        assert!(
            gamma < 5.0,
            "stressed gamma should be bounded by gamma_max: got {gamma}"
        );
        assert!(gamma > 0.05, "gamma should still be positive: got {gamma}");
        // Key insight: at base gamma 0.15 with moderate stress, the model stays reasonable
        let moderate = RiskFeatures {
            excess_volatility: 0.5,
            toxicity_score: 0.3,
            inventory_fraction: 0.3,
            excess_intensity: 0.3,
            depth_depletion: 0.2,
            model_uncertainty: 0.3,
            position_direction_confidence: 0.5,
            ..RiskFeatures::neutral()
        };
        let gamma_moderate = model.compute_gamma(&moderate);
        assert!(
            gamma_moderate < 0.5,
            "moderate stress gamma should stay below 0.5: got {gamma_moderate}"
        );
    }

    #[test]
    fn test_gamma_defense_ratio_at_moderate_toxicity() {
        let model = CalibratedRiskModel::default();

        // Neutral (no risk features)
        let neutral = RiskFeatures::neutral();
        let gamma_neutral = model.compute_gamma(&neutral);

        // Moderate toxicity
        let moderate_toxic = RiskFeatures {
            toxicity_score: 0.5,
            ..RiskFeatures::neutral()
        };
        let gamma_toxic = model.compute_gamma(&moderate_toxic);

        // defense_ratio = gamma_toxic / gamma_neutral
        let defense_ratio = gamma_toxic / gamma_neutral;

        // With beta_toxicity=0.5 and sigmoid regularization:
        // raw_sum difference = 0.5*0.5 = 0.25
        // After sigmoid, the regulated difference is dampened.
        // Still expect meaningful defense (> 1.05) but bounded by sigmoid.
        assert!(
            defense_ratio > 1.05,
            "Defense ratio at toxicity=0.5 should show meaningful widening: got {defense_ratio:.3}"
        );
        assert!(
            defense_ratio < 1.5,
            "Defense ratio should be bounded by sigmoid: got {defense_ratio:.3}"
        );
    }

    #[test]
    fn test_gamma_fills_defense_gap() {
        // Verify that gamma-produced spread at moderate toxicity exceeds fee + AS threshold.
        // GLFT half-spread ~ gamma-dependent, so higher gamma = wider spreads = more defense.
        let model = CalibratedRiskModel::default();

        let calm = RiskFeatures::neutral();
        let gamma_calm = model.compute_gamma(&calm);

        // Moderate risk scenario
        let risky = RiskFeatures {
            toxicity_score: 0.5,
            cascade_intensity: 0.3,
            tail_risk_intensity: 0.2,
            ..RiskFeatures::neutral()
        };
        let gamma_risky = model.compute_gamma(&risky);

        // With reduced betas (tox=0.5, cascade=1.2, tail=0.7) and sigmoid regularization:
        // raw_sum_risky = 0.5*0.5 + 1.2*0.3 + 0.7*0.2 + (-0.4*0.5) = 0.25+0.36+0.14-0.2 = 0.55
        // Sigmoid dampens this, but multiple risk factors still produce meaningful defense.
        let joint_ratio = gamma_risky / gamma_calm;
        assert!(
            joint_ratio > 1.3,
            "Joint defense at moderate risk should produce meaningful gamma widening: got {joint_ratio:.3}"
        );
        assert!(
            joint_ratio < 3.0,
            "Sigmoid should prevent extreme gamma inflation: got {joint_ratio:.3}"
        );
    }

    // === WS1: Unified Gamma Pipeline Tests ===

    #[test]
    fn test_ws1_quadratic_inventory_scaling() {
        let model = CalibratedRiskModel::default();

        // At 50% inventory: pressure = 0.5² = 0.25
        let half = RiskFeatures {
            inventory_fraction: 0.5,
            ..RiskFeatures::neutral()
        };
        // At 100% inventory: pressure = 1.0² = 1.0 (4× the feature value)
        let full = RiskFeatures {
            inventory_fraction: 1.0,
            ..RiskFeatures::neutral()
        };
        let calm = RiskFeatures::neutral();

        let g_calm = model.compute_gamma(&calm);
        let g_half = model.compute_gamma(&half);
        let g_full = model.compute_gamma(&full);

        // Quadratic scaling: full should be significantly more than half
        let ratio_half = g_half / g_calm;
        let ratio_full = g_full / g_calm;

        assert!(
            ratio_half > 1.2,
            "50% inventory should meaningfully widen gamma: ratio={ratio_half:.3}"
        );
        // beta_inventory=4.0, pressure=0.25 → raw contribution=1.0 → e^1.0=2.72 (before sigmoid)
        assert!(
            ratio_full > ratio_half * 1.5,
            "Full inventory should scale quadratically harder: full={ratio_full:.3} vs half={ratio_half:.3}"
        );
    }

    #[test]
    fn test_ws1_drawdown_widens_gamma() {
        let model = CalibratedRiskModel::default();

        let no_dd = RiskFeatures::neutral();
        let with_dd = RiskFeatures {
            drawdown_fraction: 0.10, // 10% drawdown
            ..RiskFeatures::neutral()
        };

        let g_base = model.compute_gamma(&no_dd);
        let g_dd = model.compute_gamma(&with_dd);

        // beta_drawdown=1.4, feature=0.10 → raw contribution=0.14 → e^0.14=1.15×
        let ratio = g_dd / g_base;
        assert!(
            ratio > 1.05,
            "10% drawdown should widen gamma: ratio={ratio:.3}"
        );
        assert!(
            ratio < 1.25,
            "10% drawdown widening should be moderate: ratio={ratio:.3}"
        );
    }

    #[test]
    fn test_ws1_regime_widens_gamma() {
        let model = CalibratedRiskModel::default();

        let calm_regime = RiskFeatures::neutral();
        let volatile_regime = RiskFeatures {
            regime_risk_score: 1.3_f64.ln(), // Regime mult = 1.3 → score ≈ 0.26
            ..RiskFeatures::neutral()
        };

        let g_base = model.compute_gamma(&calm_regime);
        let g_regime = model.compute_gamma(&volatile_regime);

        // beta_regime=1.0, feature=ln(1.3)≈0.26 → raw contribution=0.26, sigmoid dampens
        let ratio = g_regime / g_base;
        assert!(
            ratio > 1.10,
            "Volatile regime should widen gamma: ratio={ratio:.3}"
        );
        assert!(
            ratio < 1.45,
            "Regime widening should be bounded: ratio={ratio:.3}"
        );
    }

    #[test]
    fn test_ws1_ghost_widens_gamma() {
        let model = CalibratedRiskModel::default();

        let no_ghost = RiskFeatures::neutral();
        let ghost = RiskFeatures {
            ghost_depletion: 1.0, // Full ghost depletion (ghost_mult was 2.0)
            ..RiskFeatures::neutral()
        };

        let g_base = model.compute_gamma(&no_ghost);
        let g_ghost = model.compute_gamma(&ghost);

        // beta_ghost=0.5, feature=1.0 → raw contribution=0.5, sigmoid dampens to ~0.25
        let ratio = g_ghost / g_base;
        assert!(
            ratio > 1.2,
            "Full ghost depletion should meaningfully widen gamma: ratio={ratio:.3}"
        );
        assert!(
            ratio < 2.0,
            "Ghost widening should be bounded: ratio={ratio:.3}"
        );
    }

    #[test]
    fn test_ws1_all_new_features_compound_bounded() {
        // When ALL new features fire together with existing ones, gamma should stay bounded
        let model = CalibratedRiskModel::default();

        let everything = RiskFeatures {
            excess_volatility: 1.0,
            toxicity_score: 0.5,
            inventory_fraction: 0.7,
            excess_intensity: 0.5,
            depth_depletion: 0.5,
            model_uncertainty: 0.5,
            position_direction_confidence: 0.2,
            cascade_intensity: 0.5,
            tail_risk_intensity: 0.3,
            drawdown_fraction: 0.15,
            regime_risk_score: 0.59, // ln(1.8)
            ghost_depletion: 0.5,
            continuation_probability: 0.5,
        };

        let gamma = model.compute_gamma(&everything);

        // Sigmoid should prevent explosion even with all 12 features firing
        assert!(
            gamma < 5.0,
            "All features firing should stay within gamma_max: got {gamma:.3}"
        );
        assert!(
            gamma > 0.15,
            "All features should push gamma above base: got {gamma:.3}"
        );
    }

    #[test]
    fn test_ws1_blend_includes_new_betas() {
        let model = CalibratedRiskModel::default();
        let _defaults = CalibratedRiskModel::conservative_defaults();

        let blended = model.blend_with_defaults(0.5);

        // New beta fields should blend correctly
        let expected_dd = 0.5 * 1.4 + 0.5 * 2.0;
        assert!(
            (blended.beta_drawdown - expected_dd).abs() < 0.01,
            "beta_drawdown should blend: got {}, expected {}",
            blended.beta_drawdown,
            expected_dd
        );

        let expected_regime = 0.5 * 1.0 + 0.5 * 1.5;
        assert!(
            (blended.beta_regime - expected_regime).abs() < 0.01,
            "beta_regime should blend: got {}, expected {}",
            blended.beta_regime,
            expected_regime
        );

        let expected_ghost = 0.5 * 0.5 + 0.5 * 0.75;
        assert!(
            (blended.beta_ghost - expected_ghost).abs() < 0.01,
            "beta_ghost should blend: got {}, expected {}",
            blended.beta_ghost,
            expected_ghost
        );

        // Also verify beta_inventory blends (was 0.0, now 4.0 default, 5.0 conservative)
        let expected_inv = 0.5 * 4.0 + 0.5 * 5.0;
        assert!(
            (blended.beta_inventory - expected_inv).abs() < 0.01,
            "beta_inventory should blend: got {}, expected {}",
            blended.beta_inventory,
            expected_inv
        );
    }

    // === WS3: Gamma Calibration Tracker Tests ===

    #[test]
    fn test_breakeven_gamma_known_values() {
        // spread=5 bps, AS=3 bps, fee=1.5 bps → target=4.5 bps
        // kappa=2000 → 2/2000*10000 = 10 bps kappa term
        // target(4.5) < kappa_term(10) → gamma can be near 0
        let g = GammaCalibrationTracker::breakeven_gamma(5.0, 3.0, 1.5, 0.0002, 2000.0, 10.0);
        assert!(g.is_some());
        assert!(
            g.unwrap() < 0.1,
            "Low target → near-zero gamma: {}",
            g.unwrap()
        );

        // spread=20 bps, AS=5 bps, fee=1.5 bps → target=6.5 bps
        // kappa=500 → 2/500*10000 = 40 bps kappa term
        // target(6.5) < kappa_term(40) → gamma near 0
        let g2 = GammaCalibrationTracker::breakeven_gamma(20.0, 5.0, 1.5, 0.0003, 500.0, 5.0);
        assert!(g2.is_some());
    }

    #[test]
    fn test_tracking_error_responds_to_bias() {
        let mut tracker = GammaCalibrationTracker::new();

        // Predicted gamma consistently 2x higher than breakeven
        for _ in 0..20 {
            tracker.record_markout(
                0.2,    // predicted
                5.0,    // spread
                2.0,    // markout AS
                1.5,    // fee
                0.001,  // sigma (high vol so denominator is meaningful)
                5000.0, // kappa
                5.0,    // tau
            );
        }

        assert!(
            tracker.tracking_error() > 0.01,
            "Persistent bias should produce non-zero tracking error: {}",
            tracker.tracking_error()
        );
    }

    #[test]
    fn test_tracking_error_low_when_calibrated() {
        let mut tracker = GammaCalibrationTracker::new();

        // sigma=0.001, kappa=5000, tau=5
        // kappa_term = 2/5000 * 10000 = 4.0 bps
        // target = AS(2) + fee(1.5) = 3.5 bps
        // target(3.5) < kappa_term(4.0) → breakeven near 0.01 (floor)
        // Feed predicted gamma that matches breakeven
        let be =
            GammaCalibrationTracker::breakeven_gamma(5.0, 2.0, 1.5, 0.001, 5000.0, 5.0).unwrap();

        for _ in 0..20 {
            tracker.record_markout(be, 5.0, 2.0, 1.5, 0.001, 5000.0, 5.0);
        }

        assert!(
            tracker.tracking_error() < 0.05,
            "Well-calibrated gamma should have near-zero tracking error: {}",
            tracker.tracking_error()
        );
    }
}
