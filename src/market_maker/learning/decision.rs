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

    // === Realized Edge Feedback (feeds back into A-S information asymmetry) ===
    /// Mean realized gross edge in bps from EdgeTracker.
    /// Fed externally each cycle. Used to compute P(edge > 0) from outcomes.
    pub realized_edge_mean: f64,
    /// Std of realized gross edge in bps from EdgeTracker.
    pub realized_edge_std: f64,
    /// Count of resolved edge snapshots.
    pub realized_edge_n: usize,
    /// Minimum edge observations before realized feedback activates.
    /// Below this threshold, p_positive is purely prediction-based.
    pub min_edge_observations: usize,
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
            baseline_sigma: 0.0001,    // ~10 bps per second as baseline
            tau_horizon_seconds: 60.0, // 1 minute horizon for A-S formula

            // Uncertainty (for reservation shift confidence weighting only)
            baseline_edge_std: 1.0, // 1 bps edge uncertainty as baseline
            // NOTE: uncertainty_spread_scaling and max_spread_multiplier removed
            // All uncertainty flows through gamma scaling (kappa_ci_width → uncertainty_scalar)

            // Realized edge feedback (inactive until enough fills)
            realized_edge_mean: 0.0,
            realized_edge_std: 1.0,
            realized_edge_n: 0,
            min_edge_observations: 20,
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

        // 1. Model health gate — degraded models quote defensively, never stop
        if model_health.overall == Health::Degraded {
            // Scale size by calibration ratio: worse calibration → smaller presence
            // ratio near 0.7 (threshold) → size ~0.15, ratio near 1.0 → size 0.05
            let ratio = model_health.edge_calibration_ratio;
            let size = (0.2 - 0.15 * ((ratio - 0.7) / 0.3).clamp(0.0, 1.0)).max(0.05);
            return QuoteDecision::Quote {
                size_fraction: size,
                confidence: 0.0, // Zero confidence → max defensive widening downstream
                expected_edge: 0.0,
                reservation_shift: 0.0,
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

        // P(edge > 0) from ensemble predictions
        let z = mu / sigma_mu;
        let p_positive_predicted = normal_cdf(z);

        // === Realized edge feedback: blend prediction-based p_positive with
        // realized-edge p_positive (Bayesian update from EdgeTracker) ===
        //
        // The bug this fixes: p_positive stayed near 0.5 because it was computed
        // ONLY from ensemble predictions. Realized edge outcomes were never
        // incorporated. If EdgeTracker shows persistent negative realized edge,
        // the posterior P(edge > 0) should shift below 0.5, increasing information
        // asymmetry and reducing size — all within the A-S framework.
        let p_positive_realized = if self.config.realized_edge_n
            >= self.config.min_edge_observations
        {
            // Floor SE at 0.5 bps to prevent unreasonably sharp posteriors on small
            // samples with tight clustering (e.g., narrow spread regime with consistent
            // small negative marks that may reverse). Without this floor, n=25 fills
            // with std=0.3 gives SE=0.06, and mean=-0.5 gives z=-8.3 → p≈0.
            let se = (self.config.realized_edge_std / (self.config.realized_edge_n as f64).sqrt())
                .max(0.5);
            normal_cdf(self.config.realized_edge_mean / se)
        } else {
            0.5 // Uninformative: insufficient data
        };

        // Blend: trust realized outcomes more as fill count grows.
        // At 0 fills: pure prediction. At min_edge_obs: 50/50. At 2×min: ~67% realized.
        // Cap at 80%: always retain 20% weight on ensemble predictions to avoid
        // purely backward-looking behavior. The ensemble may detect regime changes
        // before realized edge statistics update.
        let realized_trust = if self.config.realized_edge_n >= self.config.min_edge_observations {
            let excess = (self.config.realized_edge_n - self.config.min_edge_observations) as f64;
            let scale = self.config.min_edge_observations as f64;
            (0.5 + 0.5 * (excess / scale).min(1.0)).min(0.8)
        } else {
            0.0 // Pure prediction before threshold
        };

        let p_positive_edge =
            (1.0 - realized_trust) * p_positive_predicted + realized_trust * p_positive_realized;

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
    fn test_defensive_quote_degraded_model() {
        let engine = DecisionEngine::default();
        let mut health = ModelHealth::default();
        health.overall = Health::Degraded;
        health.edge_calibration_ratio = 0.85; // Mid-range degradation

        let prediction = EnsemblePrediction {
            mean: 5.0,
            std: 1.0,
            disagreement: 0.5,
            model_contributions: vec![],
        };

        let decision = engine.should_quote(&prediction, &health, 0.0, TEST_SIGMA);

        match decision {
            QuoteDecision::Quote {
                size_fraction,
                confidence,
                expected_edge,
                reservation_shift,
            } => {
                // Size should be scaled by calibration ratio
                assert!(
                    size_fraction >= 0.05 && size_fraction <= 0.2,
                    "Defensive size_fraction should be in [0.05, 0.2], got {}",
                    size_fraction
                );
                // Confidence should be zero for maximum defensive widening
                assert_eq!(confidence, 0.0);
                assert_eq!(expected_edge, 0.0);
                assert_eq!(reservation_shift, 0.0);
            }
            _ => panic!("Expected defensive Quote, not NoQuote, for degraded model"),
        }
    }

    #[test]
    fn test_defensive_quote_size_scales_with_ratio() {
        let engine = DecisionEngine::default();

        // Mildly degraded (ratio near 0.7 threshold) → larger size (~0.2)
        let mut health_mild = ModelHealth::default();
        health_mild.overall = Health::Degraded;
        health_mild.edge_calibration_ratio = 0.7;

        // Severely degraded (ratio near 1.0) → smaller size (~0.05)
        let mut health_severe = ModelHealth::default();
        health_severe.overall = Health::Degraded;
        health_severe.edge_calibration_ratio = 1.0;

        let prediction = EnsemblePrediction::default();

        let decision_mild = engine.should_quote(&prediction, &health_mild, 0.0, TEST_SIGMA);
        let decision_severe = engine.should_quote(&prediction, &health_severe, 0.0, TEST_SIGMA);

        match (decision_mild, decision_severe) {
            (
                QuoteDecision::Quote {
                    size_fraction: s_mild,
                    ..
                },
                QuoteDecision::Quote {
                    size_fraction: s_severe,
                    ..
                },
            ) => {
                assert!(
                    s_mild > s_severe,
                    "Mild degradation should have larger size than severe: {} vs {}",
                    s_mild,
                    s_severe
                );
                assert!(
                    (s_mild - 0.2).abs() < 0.01,
                    "Mild should be ~0.2, got {}",
                    s_mild
                );
                assert!(
                    (s_severe - 0.05).abs() < 0.01,
                    "Severe should be ~0.05, got {}",
                    s_severe
                );
            }
            _ => panic!("Expected both to be defensive Quote decisions"),
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
                // Note: Shift is clamped to ±10% of mid (100 bps), so <= 100 is valid
                let shift_bps = reservation_shift * 10000.0;
                assert!(
                    shift_bps.abs() <= 100.0,
                    "Shift should be reasonable (<= 100 bps): {:.1} bps",
                    shift_bps
                );
            }
            _ => panic!("Expected Quote decision"),
        }
    }

    // === Realized edge feedback tests ===

    #[test]
    fn test_p_positive_blends_with_realized_negative_edge() {
        // Negative realized edge should reduce p_blended below 0.5
        let config = DecisionEngineConfig {
            realized_edge_mean: -2.0, // Persistently negative
            realized_edge_std: 4.0,
            realized_edge_n: 30, // Above threshold (20)
            min_edge_observations: 20,
            ..Default::default()
        };
        let engine = DecisionEngine::new(config);
        let health = ModelHealth::default();

        // Ensemble says neutral (p_predicted ≈ 0.5)
        let neutral = EnsemblePrediction {
            mean: 0.0,
            std: 1.0,
            disagreement: 0.1,
            model_contributions: vec![],
        };
        let decision = engine.should_quote(&neutral, &health, 0.0, TEST_SIGMA);

        match decision {
            QuoteDecision::Quote {
                size_fraction,
                confidence,
                ..
            } => {
                // With negative realized edge, information asymmetry should increase
                // → size_fraction should decrease and confidence should decrease
                // vs. pure prediction (where neutral → size ≈ 1.0, confidence ≈ 1.0)
                assert!(
                    size_fraction < 0.95,
                    "Negative realized edge should reduce size: {}",
                    size_fraction
                );
                assert!(
                    confidence < 0.95,
                    "Negative realized edge should reduce confidence: {}",
                    confidence
                );
            }
            _ => panic!("Expected Quote"),
        }
    }

    #[test]
    fn test_p_positive_no_blend_below_threshold() {
        // Below min observations → pure prediction, realized edge ignored
        let config = DecisionEngineConfig {
            realized_edge_mean: -10.0, // Very negative, but insufficient data
            realized_edge_std: 1.0,
            realized_edge_n: 5, // Below threshold (20)
            min_edge_observations: 20,
            ..Default::default()
        };
        let engine_insufficient = DecisionEngine::new(config);

        // Compare against engine with zero realized data
        let engine_default = DecisionEngine::default();
        let health = ModelHealth::default();

        let prediction = EnsemblePrediction {
            mean: 0.1,
            std: 1.0,
            disagreement: 0.1,
            model_contributions: vec![],
        };

        let decision_insufficient =
            engine_insufficient.should_quote(&prediction, &health, 0.0, TEST_SIGMA);
        let decision_default = engine_default.should_quote(&prediction, &health, 0.0, TEST_SIGMA);

        match (decision_insufficient, decision_default) {
            (
                QuoteDecision::Quote {
                    size_fraction: s1,
                    confidence: c1,
                    ..
                },
                QuoteDecision::Quote {
                    size_fraction: s2,
                    confidence: c2,
                    ..
                },
            ) => {
                // Should be identical: insufficient data → pure prediction
                assert!(
                    (s1 - s2).abs() < 0.001,
                    "Below threshold should match default: {} vs {}",
                    s1,
                    s2
                );
                assert!(
                    (c1 - c2).abs() < 0.001,
                    "Confidence should match: {} vs {}",
                    c1,
                    c2
                );
            }
            _ => panic!("Expected both to be Quote"),
        }
    }

    #[test]
    fn test_p_positive_blended_positive_realized() {
        // Positive realized edge → p_blended > 0.5 → higher asymmetry → smaller size
        // (This is the A-S framework: knowing direction reduces quoting size)
        let config = DecisionEngineConfig {
            realized_edge_mean: 3.0, // Positive realized edge
            realized_edge_std: 2.0,
            realized_edge_n: 40, // Well above threshold
            min_edge_observations: 20,
            ..Default::default()
        };
        let engine = DecisionEngine::new(config);
        let health = ModelHealth::default();

        let neutral = EnsemblePrediction {
            mean: 0.0,
            std: 1.0,
            disagreement: 0.1,
            model_contributions: vec![],
        };
        let decision = engine.should_quote(&neutral, &health, 0.0, TEST_SIGMA);

        match decision {
            QuoteDecision::Quote { size_fraction, .. } => {
                // Positive realized edge → p_blended > 0.5 → information asymmetry > 0
                // → size reduced (but not by much since A-S sensitivity is 0.5)
                assert!(
                    size_fraction < 1.0,
                    "Positive realized edge should increase asymmetry → smaller size: {}",
                    size_fraction
                );
            }
            _ => panic!("Expected Quote"),
        }
    }

    #[test]
    fn test_realized_trust_caps_at_80_percent() {
        // Even with many fills, 20% prediction weight retained
        let config = DecisionEngineConfig {
            realized_edge_mean: -5.0, // Very negative
            realized_edge_std: 1.0,
            realized_edge_n: 1000, // Way above threshold
            min_edge_observations: 20,
            ..Default::default()
        };
        let engine_many = DecisionEngine::new(config.clone());

        let config_2x = DecisionEngineConfig {
            realized_edge_n: 40, // Exactly 2x threshold
            ..config
        };
        let engine_2x = DecisionEngine::new(config_2x);
        let health = ModelHealth::default();

        let neutral = EnsemblePrediction {
            mean: 0.0,
            std: 1.0,
            disagreement: 0.1,
            model_contributions: vec![],
        };

        let decision_many = engine_many.should_quote(&neutral, &health, 0.0, TEST_SIGMA);
        let decision_2x = engine_2x.should_quote(&neutral, &health, 0.0, TEST_SIGMA);

        match (decision_many, decision_2x) {
            (
                QuoteDecision::Quote {
                    size_fraction: s_many,
                    ..
                },
                QuoteDecision::Quote {
                    size_fraction: s_2x,
                    ..
                },
            ) => {
                // At n=1000 and n=40 (2x threshold), trust is capped at 0.8
                // so 1000 fills should give same result as 40 fills
                // (both have realized_trust capped at 0.8)
                assert!(
                    (s_many - s_2x).abs() < 0.01,
                    "Cap at 80%: n=1000 and n=40 should give similar size: {} vs {}",
                    s_many,
                    s_2x
                );
            }
            _ => panic!("Expected both to be Quote"),
        }
    }

    #[test]
    fn test_existing_inverse_kelly_preserved() {
        // With zero realized_edge_n, behavior is identical to original code
        let engine = DecisionEngine::default();
        let health = ModelHealth::default();

        // Verify config defaults
        assert_eq!(engine.config().realized_edge_n, 0);
        assert_eq!(engine.config().min_edge_observations, 20);

        // Near-neutral edge → high size (inverse kelly)
        let neutral = EnsemblePrediction {
            mean: 0.1,
            std: 1.0,
            disagreement: 0.1,
            model_contributions: vec![],
        };
        let decision = engine.should_quote(&neutral, &health, 0.0, TEST_SIGMA);

        match decision {
            QuoteDecision::Quote {
                size_fraction,
                confidence,
                ..
            } => {
                // Neutral prediction → high size, high confidence (inverse of Kelly)
                assert!(
                    size_fraction > 0.9,
                    "Neutral prediction with no realized data should have high size: {}",
                    size_fraction
                );
                assert!(
                    confidence > 0.85,
                    "Neutral prediction should have high confidence: {}",
                    confidence
                );
            }
            _ => panic!("Expected Quote"),
        }
    }
}
