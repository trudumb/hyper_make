//! Level 3: Formal Decision Criterion
//!
//! Not "quote if edge > X" but "quote if P(edge > 0) > Y given uncertainty"
//! Uses Kelly criterion for position sizing.

use super::types::{EnsemblePrediction, Health, ModelHealth, QuoteDecision};

/// Decision engine configuration.
#[derive(Debug, Clone)]
pub struct DecisionEngineConfig {
    /// Risk aversion parameter
    pub risk_aversion: f64,
    /// Maximum drawdown before stopping (fraction of capital)
    pub survival_threshold: f64,
    /// Minimum P(edge > 0) to quote
    pub min_edge_confidence: f64,
    /// Minimum model health required
    pub min_model_health: Health,
    /// Kelly fraction (typically 0.25 for quarter Kelly)
    pub kelly_fraction: f64,
    /// Maximum size fraction
    pub max_size_fraction: f64,
    /// Minimum edge to quote (bps)
    pub min_edge_bps: f64,
}

impl Default for DecisionEngineConfig {
    fn default() -> Self {
        Self {
            risk_aversion: 0.5,
            survival_threshold: 0.05, // 5% max drawdown
            min_edge_confidence: 0.55, // 55% confidence edge > 0 (permissive for warmup)
            min_model_health: Health::Warning,
            kelly_fraction: 0.25, // Quarter Kelly
            max_size_fraction: 1.0,
            min_edge_bps: 0.3, // Minimum 0.3bp expected edge (allow tighter spreads)
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

    /// Decide whether to quote and how much.
    ///
    /// Decision flow:
    /// 1. Check model health
    /// 2. Check drawdown
    /// 3. Compute P(edge > 0)
    /// 4. Check model disagreement
    /// 5. Kelly sizing
    pub fn should_quote(
        &self,
        ensemble: &EnsemblePrediction,
        model_health: &ModelHealth,
        current_drawdown: f64,
    ) -> QuoteDecision {
        // 1. Check model health
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

        // 2. Check drawdown
        if current_drawdown > self.config.survival_threshold {
            return QuoteDecision::NoQuote {
                reason: format!(
                    "Drawdown {:.1}% exceeds limit {:.1}%",
                    current_drawdown * 100.0,
                    self.config.survival_threshold * 100.0
                ),
            };
        }

        // 3. Compute P(edge > 0) = Φ(mean / std)
        let z = ensemble.mean / ensemble.std.max(0.001);
        let p_positive_edge = normal_cdf(z);

        if p_positive_edge < self.config.min_edge_confidence {
            return QuoteDecision::NoQuote {
                reason: format!(
                    "Low confidence: {:.1}% < {:.1}%",
                    p_positive_edge * 100.0,
                    self.config.min_edge_confidence * 100.0
                ),
            };
        }

        // Check minimum edge
        if ensemble.mean < self.config.min_edge_bps {
            return QuoteDecision::NoQuote {
                reason: format!(
                    "Edge {:.2}bp < minimum {:.2}bp",
                    ensemble.mean,
                    self.config.min_edge_bps
                ),
            };
        }

        // 4. Check model disagreement
        // If models strongly disagree, reduce size
        if ensemble.disagreement > ensemble.mean.abs() {
            return QuoteDecision::ReducedSize {
                fraction: 0.5,
                reason: format!(
                    "High disagreement: {:.2}bp > mean {:.2}bp",
                    ensemble.disagreement,
                    ensemble.mean.abs()
                ),
            };
        }

        // 5. Kelly sizing: f* = μ / σ²
        // For continuous case with mean μ and variance σ²
        let kelly_full = ensemble.mean / ensemble.std.powi(2);
        let kelly_adjusted = kelly_full * self.config.kelly_fraction;
        let size_fraction = kelly_adjusted.clamp(0.0, self.config.max_size_fraction);

        QuoteDecision::Quote {
            size_fraction,
            confidence: p_positive_edge,
            expected_edge: ensemble.mean,
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

    /// Set minimum edge confidence.
    pub fn set_min_edge_confidence(&mut self, confidence: f64) {
        self.config.min_edge_confidence = confidence.clamp(0.5, 0.99);
    }

    /// Set Kelly fraction.
    pub fn set_kelly_fraction(&mut self, fraction: f64) {
        self.config.kelly_fraction = fraction.clamp(0.1, 1.0);
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
    fn test_quote_decision_high_confidence() {
        let engine = DecisionEngine::default();
        let health = ModelHealth::default();

        let prediction = EnsemblePrediction {
            mean: 3.0,  // 3bp expected edge
            std: 1.0,   // 1bp uncertainty
            disagreement: 0.5,
            model_contributions: vec![],
        };

        let decision = engine.should_quote(&prediction, &health, 0.0);

        match decision {
            QuoteDecision::Quote { confidence, expected_edge, .. } => {
                assert!(confidence > 0.9); // z = 3 → very high confidence
                assert!((expected_edge - 3.0).abs() < 0.01);
            }
            _ => panic!("Expected Quote decision"),
        }
    }

    #[test]
    fn test_no_quote_low_confidence() {
        let engine = DecisionEngine::default();
        let health = ModelHealth::default();

        // z = 0.1 / 1.0 = 0.1, giving P(edge > 0) ≈ 54% < 60% threshold
        let prediction = EnsemblePrediction {
            mean: 0.1,  // 0.1bp expected edge (very small)
            std: 1.0,   // 1bp uncertainty (z = 0.1)
            disagreement: 0.05,
            model_contributions: vec![],
        };

        let decision = engine.should_quote(&prediction, &health, 0.0);

        match decision {
            QuoteDecision::NoQuote { reason } => {
                // Should fail either on low confidence or low edge
                assert!(
                    reason.contains("Low confidence") || reason.contains("Edge"),
                    "Unexpected reason: {}",
                    reason
                );
            }
            _ => panic!("Expected NoQuote decision"),
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

        let decision = engine.should_quote(&prediction, &health, 0.0);

        match decision {
            QuoteDecision::NoQuote { reason } => {
                assert!(reason.contains("degraded"));
            }
            _ => panic!("Expected NoQuote due to degraded model"),
        }
    }

    #[test]
    fn test_reduced_size_high_disagreement() {
        let engine = DecisionEngine::default();
        let health = ModelHealth::default();

        let prediction = EnsemblePrediction {
            mean: 2.0,
            std: 0.5,
            disagreement: 3.0, // Disagreement > mean
            model_contributions: vec![],
        };

        let decision = engine.should_quote(&prediction, &health, 0.0);

        match decision {
            QuoteDecision::ReducedSize { fraction, reason } => {
                assert!((fraction - 0.5).abs() < 0.01);
                assert!(reason.contains("disagreement"));
            }
            _ => panic!("Expected ReducedSize decision"),
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
        let decision = engine.should_quote(&prediction, &health, 0.10);

        match decision {
            QuoteDecision::NoQuote { reason } => {
                assert!(reason.contains("Drawdown"));
            }
            _ => panic!("Expected NoQuote due to drawdown"),
        }
    }

    #[test]
    fn test_kelly_sizing() {
        let engine = DecisionEngine::default();
        let health = ModelHealth::default();

        // High edge, low uncertainty → larger size
        let prediction1 = EnsemblePrediction {
            mean: 4.0,
            std: 1.0,
            disagreement: 0.5,
            model_contributions: vec![],
        };

        let decision1 = engine.should_quote(&prediction1, &health, 0.0);

        // Lower edge, same uncertainty → smaller size
        let prediction2 = EnsemblePrediction {
            mean: 2.0,
            std: 1.0,
            disagreement: 0.5,
            model_contributions: vec![],
        };

        let decision2 = engine.should_quote(&prediction2, &health, 0.0);

        match (decision1, decision2) {
            (QuoteDecision::Quote { size_fraction: s1, .. }, QuoteDecision::Quote { size_fraction: s2, .. }) => {
                assert!(s1 > s2, "Higher edge should have larger size");
            }
            _ => panic!("Expected both to be Quote decisions"),
        }
    }
}
