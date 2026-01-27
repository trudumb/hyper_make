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
    // NOTE: min_size_fraction has been REMOVED. Size is now controlled by GLFT's
    // inventory_scalar and cascade_size_factor, not arbitrary floor values.

    // === Reservation Shift (A-S framework) ===
    /// Baseline realized volatility for normalizing reservation shift
    /// Used when ensemble doesn't provide realized_vol directly
    pub baseline_sigma: f64,

    /// Horizon for A-S reservation shift calculation (seconds).
    ///
    /// FIRST PRINCIPLES: The A-S formula δ = μ/(2γσ²) requires ALL units to be
    /// consistent per-τ horizon:
    /// - μ = drift in return-per-τ (e.g., 0.001 = 10 bps per horizon)
    /// - σ = volatility in return-per-√τ
    /// - Result: shift in return units
    ///
    /// Default: 60.0 seconds (1 minute horizon)
    pub tau_horizon_seconds: f64,

    // === Uncertainty Premium (Bayesian spread widening) ===
    /// Baseline edge std for uncertainty scaling (in bps)
    /// NOTE: spread_multiplier has been REMOVED. All uncertainty now flows through
    /// gamma scaling (kappa_ci_width → uncertainty_scalar). This field is kept only
    /// for reservation_shift confidence weighting.
    pub baseline_edge_std: f64,
    // NOTE: uncertainty_spread_scaling and max_spread_multiplier have been REMOVED.
    // All uncertainty is now handled through gamma scaling (kappa_ci_width flows
    // through uncertainty_scalar). The GLFT formula naturally widens spreads
    // when gamma increases due to uncertainty.
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
            // NOTE: min_size_fraction removed - size is controlled by GLFT inventory_scalar

            // Reservation shift
            baseline_sigma: 0.0001, // ~10 bps per second as baseline
            tau_horizon_seconds: 60.0, // 1 minute horizon for A-S formula

            // Uncertainty (for reservation shift confidence weighting only)
            baseline_edge_std: 1.0, // 1 bps edge uncertainty as baseline
            // NOTE: uncertainty_spread_scaling and max_spread_multiplier removed
            // All uncertainty flows through gamma scaling (kappa_ci_width → uncertainty_scalar)
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

        if model_health.overall == Health::Warning && self.config.min_model_health == Health::Good {
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
        // NOTE: min_size_fraction removed - GLFT's inventory_scalar and cascade_size_factor
        // now handle minimum sizing. We just report the adverse_selection_factor to L3.
        let adverse_selection_factor =
            1.0 - self.config.adverse_selection_sensitivity * information_asymmetry;
        let size_fraction = adverse_selection_factor.clamp(0.0, self.config.max_size_fraction);

        // High epistemic uncertainty (model disagreement) → reduce size further
        let epistemicity_ratio = ensemble.disagreement / sigma_mu;
        let size_fraction = if epistemicity_ratio > 0.5 {
            // Reduce by half when models disagree significantly
            size_fraction * 0.5
        } else {
            size_fraction
        };

        // === RESERVATION SHIFT: From A-S (First Principles) ===
        //
        // The A-S reservation price formula for drift is:
        //     δ_μ = μ / (2 × γ × σ²)
        //
        // CRITICAL: All units must be consistent per-τ horizon:
        // - μ = drift in return-per-τ (e.g., 0.001 = 10 bps per horizon)
        // - σ = volatility in return-per-√τ (e.g., 0.005 for 50 bps/√τ)
        // - γ = dimensionless risk aversion
        // - Result: shift in return units (multiply by price for price offset)
        //
        // With proper dimensional analysis, positive edge → shift UP → aggressive asks.
        let gamma = self.config.risk_aversion.max(0.01);
        let tau_seconds = self.config.tau_horizon_seconds;
        let sigma_per_second = realized_vol.max(self.config.baseline_sigma);

        // Convert inputs to consistent per-τ units
        // μ: edge prediction in bps → return fraction per τ
        let mu_per_tau = mu / 10000.0;

        // σ: per-second volatility → per-√τ volatility
        // σ_per_√τ = σ_per_√s × √τ
        let sigma_per_sqrt_tau = sigma_per_second * tau_seconds.sqrt();
        let sigma_sq_per_tau = sigma_per_sqrt_tau.powi(2).max(1e-12);

        // A-S reservation shift: δ = μ / (2γσ²) in return units
        let shift_return_raw = mu_per_tau / (2.0 * gamma * sigma_sq_per_tau);

        // Confidence-weight: don't shift aggressively if uncertain about edge
        // When σ_μ >> |μ|, we're uncertain about the edge direction → reduce shift
        let shift_confidence = 1.0 - (sigma_mu / mu.abs().max(0.001)).clamp(0.0, 1.0);
        let shift_return_weighted = shift_return_raw * shift_confidence;

        // Clamp to ±100 bps (0.01) - beyond this, model is likely wrong
        let reservation_shift = shift_return_weighted.clamp(-0.01, 0.01);

        // === REMOVED: SPREAD UNCERTAINTY PREMIUM ===
        // spread_multiplier has been REMOVED. All uncertainty is now handled through
        // gamma scaling (kappa_ci_width flows through uncertainty_scalar). The GLFT
        // formula naturally widens spreads when gamma increases due to uncertainty.
        //
        // The edge uncertainty (σ_μ) still feeds into confidence for diagnostic purposes.

        // Confidence is HIGH when p ≈ 0.5 (we're confident that it's uncertain)
        // This is the inverse of traditional "directional confidence"
        let confidence = 1.0 - information_asymmetry;

        QuoteDecision::Quote {
            size_fraction,
            confidence,
            expected_edge: mu,
            reservation_shift,
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

    // NOTE: set_min_size_fraction has been REMOVED.
    // Size is now controlled by GLFT's inventory_scalar and cascade_size_factor.
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
            mean: 3.0, // 3bp expected edge
            std: 1.0,
            disagreement: 0.1,
            model_contributions: vec![],
        };
        let decision = engine.should_quote(&positive_edge, &health, 0.0, TEST_SIGMA);
        assert!(
            matches!(decision, QuoteDecision::Quote { .. }),
            "Should quote with positive edge"
        );

        // Test with NEGATIVE edge - should STILL quote (key A-S insight)
        let negative_edge = EnsemblePrediction {
            mean: -3.0, // Negative edge
            std: 1.0,
            disagreement: 0.1,
            model_contributions: vec![],
        };
        let decision = engine.should_quote(&negative_edge, &health, 0.0, TEST_SIGMA);
        assert!(
            matches!(decision, QuoteDecision::Quote { .. }),
            "Should STILL quote with negative edge"
        );

        // Test with zero edge - should quote
        let zero_edge = EnsemblePrediction {
            mean: 0.0,
            std: 1.0,
            disagreement: 0.1,
            model_contributions: vec![],
        };
        let decision = engine.should_quote(&zero_edge, &health, 0.0, TEST_SIGMA);
        assert!(
            matches!(decision, QuoteDecision::Quote { .. }),
            "Should quote with zero edge"
        );
    }

    #[test]
    fn test_inverse_information_asymmetry() {
        // Key test: Size should be LARGER when p ≈ 0.5 (uncertain), SMALLER at extremes
        // This is the INVERSE of Kelly criterion
        let engine = DecisionEngine::default();
        let health = ModelHealth::default();

        // Near-zero edge = p ≈ 0.5 = uncertain = larger size
        let uncertain = EnsemblePrediction {
            mean: 0.1, // Near zero edge
            std: 1.0,  // z = 0.1 → p ≈ 0.54 (close to 0.5)
            disagreement: 0.1,
            model_contributions: vec![],
        };
        let decision_uncertain = engine.should_quote(&uncertain, &health, 0.0, TEST_SIGMA);

        // Large positive edge = p → 1 = informed = smaller size
        let confident = EnsemblePrediction {
            mean: 5.0, // Large edge
            std: 1.0,  // z = 5 → p ≈ 1.0 (very confident direction)
            disagreement: 0.1,
            model_contributions: vec![],
        };
        let decision_confident = engine.should_quote(&confident, &health, 0.0, TEST_SIGMA);

        match (decision_uncertain, decision_confident) {
            (
                QuoteDecision::Quote {
                    size_fraction: s_uncertain,
                    confidence: c_uncertain,
                    ..
                },
                QuoteDecision::Quote {
                    size_fraction: s_confident,
                    confidence: c_confident,
                    ..
                },
            ) => {
                // Uncertain prediction should have LARGER size (inverse of Kelly)
                assert!(
                    s_uncertain > s_confident,
                    "Uncertain (p≈0.5) should have larger size than confident (p→1): {} vs {}",
                    s_uncertain,
                    s_confident
                );
                // Confidence is HIGH when p ≈ 0.5 (we're confident it's a coin flip)
                assert!(
                    c_uncertain > c_confident,
                    "Confidence should be higher when p≈0.5: {} vs {}",
                    c_uncertain,
                    c_confident
                );
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
            std: 0.5, // High confidence in direction
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
            (
                QuoteDecision::Quote {
                    reservation_shift: shift_pos,
                    ..
                },
                QuoteDecision::Quote {
                    reservation_shift: shift_neg,
                    ..
                },
            ) => {
                assert!(
                    shift_pos > 0.0,
                    "Positive edge should shift UP: {}",
                    shift_pos
                );
                assert!(
                    shift_neg < 0.0,
                    "Negative edge should shift DOWN: {}",
                    shift_neg
                );
            }
            _ => panic!("Expected both to be Quote decisions"),
        }
    }

    // NOTE: test_spread_multiplier_uncertainty has been REMOVED.
    // spread_multiplier is no longer computed - all uncertainty flows through gamma.
    // The test was verifying spread_multiplier increases with edge uncertainty,
    // which is now handled by kappa_ci_width → uncertainty_scalar in GLFT.

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
            (
                QuoteDecision::Quote {
                    size_fraction: s_low,
                    ..
                },
                QuoteDecision::Quote {
                    size_fraction: s_high,
                    ..
                },
            ) => {
                assert!(
                    s_high < s_low,
                    "High disagreement should reduce size: {} vs {}",
                    s_high,
                    s_low
                );
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
    fn test_size_fraction_bounds() {
        // NOTE: min_size_fraction has been REMOVED - size is controlled by GLFT.
        // This test now verifies size_fraction is in [0.0, max_size_fraction] range.
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
                assert!(
                    size_fraction >= 0.0 && size_fraction <= engine.config().max_size_fraction,
                    "Size should be in [0.0, max]: {} vs [0.0, {}]",
                    size_fraction,
                    engine.config().max_size_fraction
                );
            }
            _ => panic!("Expected Quote decision"),
        }
    }

    #[test]
    fn test_reservation_shift_first_principles_as() {
        // Test that the A-S formula produces dimensionally correct values
        // with the new tau_horizon implementation.
        //
        // Formula: δ = μ/(2γσ²) where all units are per-τ
        // With τ=60s, σ_per_sec=0.0001 (10 bps/sec), μ=10 bps, γ=0.5:
        // - σ_per_√τ = 0.0001 × √60 = 0.000775
        // - σ² = 6e-7
        // - δ = 0.001 / (2 × 0.5 × 6e-7) = 0.001 / 6e-7 ≈ 1667 (before clamp)
        // This would be huge, so we clamp to ±0.01 (100 bps)
        let engine = DecisionEngine::default();
        let health = ModelHealth::default();

        let prediction = EnsemblePrediction {
            mean: 10.0, // 10 bps expected edge
            std: 1.0,   // Low uncertainty (high confidence)
            disagreement: 0.1,
            model_contributions: vec![],
        };

        let decision = engine.should_quote(&prediction, &health, 0.0, 0.0001);

        match decision {
            QuoteDecision::Quote {
                reservation_shift, ..
            } => {
                // Should be clamped to max 0.01 (100 bps)
                assert!(
                    reservation_shift <= 0.01,
                    "Reservation shift should be clamped to ±0.01: {}",
                    reservation_shift
                );
                assert!(
                    reservation_shift >= -0.01,
                    "Reservation shift should be clamped to ±0.01: {}",
                    reservation_shift
                );
                // With positive edge and high confidence, shift should be positive
                assert!(
                    reservation_shift > 0.0,
                    "Positive edge should produce positive shift: {}",
                    reservation_shift
                );
            }
            _ => panic!("Expected Quote decision"),
        }
    }

    #[test]
    fn test_reservation_shift_bounded_reasonable() {
        // Test that typical market conditions produce reasonable shifts
        // (not the 30× explosion that happened before)
        let engine = DecisionEngine::default();
        let health = ModelHealth::default();

        // Typical scenario: 5 bps edge, moderate confidence
        let prediction = EnsemblePrediction {
            mean: 5.0,
            std: 2.0, // Moderate uncertainty
            disagreement: 0.5,
            model_contributions: vec![],
        };

        // Typical BTC volatility: ~20 bps/sec
        let decision = engine.should_quote(&prediction, &health, 0.0, 0.0002);

        match decision {
            QuoteDecision::Quote {
                reservation_shift, ..
            } => {
                // Shift should be bounded and reasonable
                // Should NOT explode to 30× like the old bug
                let shift_bps = reservation_shift * 10000.0;
                assert!(
                    shift_bps.abs() < 100.0,
                    "Shift should be reasonable (< 100 bps): {} bps",
                    shift_bps
                );
            }
            _ => panic!("Expected Quote decision"),
        }
    }
}
