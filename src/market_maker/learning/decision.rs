//! Level 3: Formal Decision Criterion
//!
//! Not "quote if edge > X" but "quote if P(edge > 0) > Y given uncertainty"
//! Uses Kelly criterion for position sizing.

use super::types::{EnsemblePrediction, Health, ModelHealth, QuoteDecision};

/// Decision engine configuration.
///
/// Implements stochastic control framework for market making (Avellaneda-Stoikov):
/// - Edge → reservation shift (not existence)
/// - Inverse information asymmetry for sizing
/// - Uncertainty → spread widening
#[derive(Debug, Clone)]
pub struct DecisionEngineConfig {
    /// Risk aversion parameter (γ in A-S framework)
    pub risk_aversion: f64,
    /// Maximum drawdown before stopping (fraction of capital)
    pub survival_threshold: f64,
    /// Minimum model health required (the ONLY hard gate besides drawdown)
    pub min_model_health: Health,
    /// Maximum size fraction
    pub max_size_fraction: f64,

    // === Inverse Information Asymmetry (A-S framework) ===
    /// Sensitivity to information asymmetry (0.0 = ignore, 1.0 = aggressive reduction)
    /// When p → 0 or p → 1, someone is informed → reduce size by this factor × asymmetry
    pub adverse_selection_sensitivity: f64,
    /// Minimum size fraction (never go below this, ensures continuous quoting)
    pub min_size_fraction: f64,

    // === Reservation Shift (A-S framework) ===
    /// Baseline realized volatility for normalizing reservation shift
    /// Used when ensemble doesn't provide realized_vol directly
    pub baseline_sigma: f64,

    // === Uncertainty Premium (Bayesian spread widening) ===
    /// Baseline edge std for uncertainty scaling (in bps)
    /// Spread multiplier = 1 + uncertainty_spread_scaling × (σ_μ / baseline_edge_std)
    pub baseline_edge_std: f64,
    /// How much to widen spread per unit uncertainty
    pub uncertainty_spread_scaling: f64,
    /// Maximum spread multiplier (cap on widening)
    pub max_spread_multiplier: f64,
}

impl Default for DecisionEngineConfig {
    fn default() -> Self {
        Self {
            risk_aversion: 0.5,
            survival_threshold: 0.05, // 5% max drawdown
            min_model_health: Health::Warning,
            max_size_fraction: 1.0,

            // Inverse information asymmetry
            adverse_selection_sensitivity: 0.5, // Moderate reduction when informed flow detected
            min_size_fraction: 0.5, // Never go below 50% - ensures continuous quoting

            // Reservation shift
            baseline_sigma: 0.0001, // ~10 bps per second as baseline

            // Uncertainty premium
            baseline_edge_std: 1.0, // 1 bps edge uncertainty as baseline
            uncertainty_spread_scaling: 0.2, // 20% spread widening per unit uncertainty
            max_spread_multiplier: 2.0, // Cap spread widening at 2x
        }
    }
}

/// Decision engine that determines when and how much to quote.
pub struct DecisionEngine {
    config: DecisionEngineConfig,
}

impl Default for DecisionEngine {
    fn default() -> Self {
        Self::new(DecisionEngineConfig::default())
    }
}

impl DecisionEngine {
    /// Create a new decision engine with config.
    pub fn new(config: DecisionEngineConfig) -> Self {
        Self { config }
    }

    /// Stochastic-control-based decision for market making.
    ///
    /// Implements Avellaneda-Stoikov framework:
    /// 1. Model health and drawdown gates (the ONLY hard stops)
    /// 2. Inverse information asymmetry for sizing (quote MORE when uncertain)
    /// 3. Reservation price shift from edge (δ_μ = μ/(γσ²))
    /// 4. Uncertainty premium for spread widening
    ///
    /// Key insight: Edge modulates WHERE you quote (reservation), not WHETHER you quote.
    pub fn should_quote(
        &self,
        ensemble: &EnsemblePrediction,
        model_health: &ModelHealth,
        current_drawdown: f64,
        realized_vol: f64, // σ from market params (per-second volatility)
    ) -> QuoteDecision {
        // === HARD GATES (the only reasons to stop quoting) ===

        // 1. Model health gate
        if model_health.overall == Health::Degraded {
            return QuoteDecision::NoQuote {
                reason: "Model degraded".into(),
            };
        }

        if model_health.overall == Health::Warning
            && self.config.min_model_health == Health::Good
        {
            return QuoteDecision::NoQuote {
                reason: "Model health warning, require good".into(),
            };
        }

        // 2. Drawdown gate
        if current_drawdown > self.config.survival_threshold {
            return QuoteDecision::NoQuote {
                reason: format!(
                    "Drawdown {:.1}% exceeds limit {:.1}%",
                    current_drawdown * 100.0,
                    self.config.survival_threshold * 100.0
                ),
            };
        }

        // === SIZING: Inverse Information Asymmetry ===
        // Key insight: We want to quote LARGER when p ≈ 0.5 (uncertain about direction)
        // because spread capture dominates over adverse selection.
        // When p → 0 or p → 1, someone is informed → reduce size to protect.
        //
        // This is the INVERSE of Kelly which sizes larger with higher confidence.
        // For market makers, neutral uncertainty means safe spread capture.

        let mu = ensemble.mean; // Expected edge (drift) in bps
        let sigma_mu = ensemble.std.max(0.001); // Uncertainty in edge estimate

        // P(edge > 0) = Φ(μ / σ_μ)
        let z = mu / sigma_mu;
        let p_positive_edge = normal_cdf(z);

        // Information asymmetry: 0 at p=0.5 (neutral), 1 at p=0 or p=1 (informed)
        let information_asymmetry = (2.0 * (p_positive_edge - 0.5)).abs();

        // Size fraction: high when uncertain, reduced when informed flow detected
        let adverse_selection_factor =
            1.0 - self.config.adverse_selection_sensitivity * information_asymmetry;
        let size_fraction =
            adverse_selection_factor.clamp(self.config.min_size_fraction, self.config.max_size_fraction);

        // High epistemic uncertainty (model disagreement) → reduce size further
        let epistemicity_ratio = ensemble.disagreement / sigma_mu;
        let size_fraction = if epistemicity_ratio > 0.5 {
            // Reduce by half when models disagree significantly
            (size_fraction * 0.5).max(self.config.min_size_fraction)
        } else {
            size_fraction
        };

        // === RESERVATION SHIFT: From A-S ===
        // δ_μ = μ / (γσ²) - positive edge → shift UP → aggressive asks
        // This shifts where we center quotes, not whether we quote.
        let gamma = self.config.risk_aversion;
        let sigma_sq = realized_vol
            .max(self.config.baseline_sigma)
            .powi(2)
            .max(1e-12);

        // Base reservation shift from A-S formula
        // Note: mu is in bps, convert to price fraction for shift
        let reservation_shift_raw = (mu / 10000.0) / (gamma * sigma_sq);

        // Confidence-weight: don't shift aggressively if uncertain about edge
        // When σ_μ >> |μ|, we're uncertain about the edge direction → reduce shift
        let shift_confidence = 1.0 - (sigma_mu / mu.abs().max(0.001)).clamp(0.0, 1.0);
        let reservation_shift = reservation_shift_raw * shift_confidence;

        // === SPREAD: Uncertainty Premium ===
        // High σ_μ (noisy edge estimate) → widen spread (Bayesian uncertainty premium)
        let edge_uncertainty_ratio = sigma_mu / self.config.baseline_edge_std.max(0.001);
        let spread_multiplier = (1.0 + self.config.uncertainty_spread_scaling * edge_uncertainty_ratio)
            .clamp(1.0, self.config.max_spread_multiplier);

        // Confidence is HIGH when p ≈ 0.5 (we're confident that it's uncertain)
        // This is the inverse of traditional "directional confidence"
        let confidence = 1.0 - information_asymmetry;

        QuoteDecision::Quote {
            size_fraction,
            confidence,
            expected_edge: mu,
            reservation_shift,
            spread_multiplier,
        }
    }

    /// Get current config.
    pub fn config(&self) -> &DecisionEngineConfig {
        &self.config
    }

    /// Update config.
    pub fn set_config(&mut self, config: DecisionEngineConfig) {
        self.config = config;
    }

    /// Set adverse selection sensitivity.
    pub fn set_adverse_selection_sensitivity(&mut self, sensitivity: f64) {
        self.config.adverse_selection_sensitivity = sensitivity.clamp(0.0, 1.0);
    }

    /// Set minimum size fraction.
    pub fn set_min_size_fraction(&mut self, fraction: f64) {
        self.config.min_size_fraction = fraction.clamp(0.1, 1.0);
    }
}

/// Standard normal CDF approximation.
///
/// Uses Abramowitz and Stegun approximation (7.1.26).
fn normal_cdf(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }

    // Constants for the approximation
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x / 2.0).exp();

    0.5 * (1.0 + sign * y)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SIGMA: f64 = 0.0001; // 10 bps per-second volatility

    #[test]
    fn test_normal_cdf() {
        // CDF(0) = 0.5
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.001);

        // CDF(-infinity) → 0
        assert!(normal_cdf(-10.0) < 0.001);

        // CDF(+infinity) → 1
        assert!(normal_cdf(10.0) > 0.999);

        // CDF(1.96) ≈ 0.975
        assert!((normal_cdf(1.96) - 0.975).abs() < 0.01);
    }

    #[test]
    fn test_always_quotes_with_healthy_model() {
        // Key test: Market makers should ALWAYS quote when model is healthy
        // Edge direction should NOT prevent quoting
        let engine = DecisionEngine::default();
        let health = ModelHealth::default();

        // Test with positive edge
        let positive_edge = EnsemblePrediction {
            mean: 3.0,  // 3bp expected edge
            std: 1.0,
            disagreement: 0.1,
            model_contributions: vec![],
        };
        let decision = engine.should_quote(&positive_edge, &health, 0.0, TEST_SIGMA);
        assert!(matches!(decision, QuoteDecision::Quote { .. }), "Should quote with positive edge");

        // Test with NEGATIVE edge - should STILL quote (key A-S insight)
        let negative_edge = EnsemblePrediction {
            mean: -3.0,  // Negative edge
            std: 1.0,
            disagreement: 0.1,
            model_contributions: vec![],
        };
        let decision = engine.should_quote(&negative_edge, &health, 0.0, TEST_SIGMA);
        assert!(matches!(decision, QuoteDecision::Quote { .. }), "Should STILL quote with negative edge");

        // Test with zero edge - should quote
        let zero_edge = EnsemblePrediction {
            mean: 0.0,
            std: 1.0,
            disagreement: 0.1,
            model_contributions: vec![],
        };
        let decision = engine.should_quote(&zero_edge, &health, 0.0, TEST_SIGMA);
        assert!(matches!(decision, QuoteDecision::Quote { .. }), "Should quote with zero edge");
    }

    #[test]
    fn test_inverse_information_asymmetry() {
        // Key test: Size should be LARGER when p ≈ 0.5 (uncertain), SMALLER at extremes
        // This is the INVERSE of Kelly criterion
        let engine = DecisionEngine::default();
        let health = ModelHealth::default();

        // Near-zero edge = p ≈ 0.5 = uncertain = larger size
        let uncertain = EnsemblePrediction {
            mean: 0.1,  // Near zero edge
            std: 1.0,   // z = 0.1 → p ≈ 0.54 (close to 0.5)
            disagreement: 0.1,
            model_contributions: vec![],
        };
        let decision_uncertain = engine.should_quote(&uncertain, &health, 0.0, TEST_SIGMA);

        // Large positive edge = p → 1 = informed = smaller size
        let confident = EnsemblePrediction {
            mean: 5.0,  // Large edge
            std: 1.0,   // z = 5 → p ≈ 1.0 (very confident direction)
            disagreement: 0.1,
            model_contributions: vec![],
        };
        let decision_confident = engine.should_quote(&confident, &health, 0.0, TEST_SIGMA);

        match (decision_uncertain, decision_confident) {
            (QuoteDecision::Quote { size_fraction: s_uncertain, confidence: c_uncertain, .. },
             QuoteDecision::Quote { size_fraction: s_confident, confidence: c_confident, .. }) => {
                // Uncertain prediction should have LARGER size (inverse of Kelly)
                assert!(s_uncertain > s_confident,
                    "Uncertain (p≈0.5) should have larger size than confident (p→1): {} vs {}",
                    s_uncertain, s_confident);
                // Confidence is HIGH when p ≈ 0.5 (we're confident it's a coin flip)
                assert!(c_uncertain > c_confident,
                    "Confidence should be higher when p≈0.5: {} vs {}",
                    c_uncertain, c_confident);
            }
            _ => panic!("Expected both to be Quote decisions"),
        }
    }

    #[test]
    fn test_reservation_shift_direction() {
        // Positive edge → shift UP → aggressive asks
        // Negative edge → shift DOWN → aggressive bids
        let engine = DecisionEngine::default();
        let health = ModelHealth::default();

        let positive_edge = EnsemblePrediction {
            mean: 5.0,
            std: 0.5,  // High confidence in direction
            disagreement: 0.1,
            model_contributions: vec![],
        };
        let decision_pos = engine.should_quote(&positive_edge, &health, 0.0, TEST_SIGMA);

        let negative_edge = EnsemblePrediction {
            mean: -5.0,
            std: 0.5,
            disagreement: 0.1,
            model_contributions: vec![],
        };
        let decision_neg = engine.should_quote(&negative_edge, &health, 0.0, TEST_SIGMA);

        match (decision_pos, decision_neg) {
            (QuoteDecision::Quote { reservation_shift: shift_pos, .. },
             QuoteDecision::Quote { reservation_shift: shift_neg, .. }) => {
                assert!(shift_pos > 0.0, "Positive edge should shift UP: {}", shift_pos);
                assert!(shift_neg < 0.0, "Negative edge should shift DOWN: {}", shift_neg);
            }
            _ => panic!("Expected both to be Quote decisions"),
        }
    }

    #[test]
    fn test_spread_multiplier_uncertainty() {
        // High σ_μ (edge uncertainty) → wider spreads
        let engine = DecisionEngine::default();
        let health = ModelHealth::default();

        let certain_edge = EnsemblePrediction {
            mean: 2.0,
            std: 0.5,   // Low uncertainty
            disagreement: 0.1,
            model_contributions: vec![],
        };
        let decision_certain = engine.should_quote(&certain_edge, &health, 0.0, TEST_SIGMA);

        let uncertain_edge = EnsemblePrediction {
            mean: 2.0,  // Same edge
            std: 5.0,   // High uncertainty
            disagreement: 0.1,
            model_contributions: vec![],
        };
        let decision_uncertain = engine.should_quote(&uncertain_edge, &health, 0.0, TEST_SIGMA);

        match (decision_certain, decision_uncertain) {
            (QuoteDecision::Quote { spread_multiplier: sm_certain, .. },
             QuoteDecision::Quote { spread_multiplier: sm_uncertain, .. }) => {
                assert!(sm_uncertain > sm_certain,
                    "Higher edge uncertainty should widen spread: {} vs {}",
                    sm_uncertain, sm_certain);
                assert!(sm_certain >= 1.0, "Spread multiplier should be >= 1.0");
            }
            _ => panic!("Expected both to be Quote decisions"),
        }
    }

    #[test]
    fn test_no_quote_degraded_model() {
        let engine = DecisionEngine::default();
        let mut health = ModelHealth::default();
        health.overall = Health::Degraded;

        let prediction = EnsemblePrediction {
            mean: 5.0,
            std: 1.0,
            disagreement: 0.5,
            model_contributions: vec![],
        };

        let decision = engine.should_quote(&prediction, &health, 0.0, TEST_SIGMA);

        match decision {
            QuoteDecision::NoQuote { reason } => {
                assert!(reason.contains("degraded"));
            }
            _ => panic!("Expected NoQuote due to degraded model"),
        }
    }

    #[test]
    fn test_reduced_size_high_disagreement() {
        // High model disagreement → reduced size (but still quote!)
        let engine = DecisionEngine::default();
        let health = ModelHealth::default();

        let low_disagreement = EnsemblePrediction {
            mean: 2.0,
            std: 1.0,
            disagreement: 0.1, // Low disagreement
            model_contributions: vec![],
        };
        let decision_low = engine.should_quote(&low_disagreement, &health, 0.0, TEST_SIGMA);

        let high_disagreement = EnsemblePrediction {
            mean: 2.0,
            std: 1.0,
            disagreement: 2.0, // High disagreement (2x std)
            model_contributions: vec![],
        };
        let decision_high = engine.should_quote(&high_disagreement, &health, 0.0, TEST_SIGMA);

        match (decision_low, decision_high) {
            (QuoteDecision::Quote { size_fraction: s_low, .. },
             QuoteDecision::Quote { size_fraction: s_high, .. }) => {
                assert!(s_high < s_low,
                    "High disagreement should reduce size: {} vs {}",
                    s_high, s_low);
                assert!(s_high > 0.0, "Should still quote with high disagreement");
            }
            _ => panic!("Expected both to be Quote decisions"),
        }
    }

    #[test]
    fn test_no_quote_drawdown() {
        let engine = DecisionEngine::default();
        let health = ModelHealth::default();

        let prediction = EnsemblePrediction {
            mean: 5.0,
            std: 1.0,
            disagreement: 0.5,
            model_contributions: vec![],
        };

        // 10% drawdown exceeds 5% threshold
        let decision = engine.should_quote(&prediction, &health, 0.10, TEST_SIGMA);

        match decision {
            QuoteDecision::NoQuote { reason } => {
                assert!(reason.contains("Drawdown"));
            }
            _ => panic!("Expected NoQuote due to drawdown"),
        }
    }

    #[test]
    fn test_minimum_size_fraction() {
        // Even with extreme information asymmetry, size should not go below min_size_fraction
        let engine = DecisionEngine::default();
        let health = ModelHealth::default();

        let extreme_certainty = EnsemblePrediction {
            mean: 10.0, // Very strong edge signal
            std: 0.1,   // Very low uncertainty → p → 1
            disagreement: 0.01,
            model_contributions: vec![],
        };

        let decision = engine.should_quote(&extreme_certainty, &health, 0.0, TEST_SIGMA);

        match decision {
            QuoteDecision::Quote { size_fraction, .. } => {
                assert!(size_fraction >= engine.config().min_size_fraction,
                    "Size should never go below min: {} vs {}",
                    size_fraction, engine.config().min_size_fraction);
            }
            _ => panic!("Expected Quote decision"),
        }
    }
}
