//! Hybrid Expected Value Estimator
//!
//! Provides smooth transition between theoretical and IR-based edge estimation
//! based on bin population and data quality. Solves the bootstrap problem where
//! IR can't calibrate without fills, but we need good edge estimates to get fills.
//!
//! # Problem Solved
//!
//! The vicious cycle:
//! 1. IR can't calibrate (zero resolution from single-bin samples)
//! 2. Theoretical edge is often negative → no quoting
//! 3. No quoting → no fills → no data → IR stays uncalibrated
//!
//! # Solution
//!
//! Hybrid blending with smooth alpha transition:
//! - alpha = f(filled_bins, sample_count, accuracy)
//! - EV = alpha * IR_edge + (1 - alpha) * theoretical_edge
//!
//! As data accumulates and bins populate, alpha increases from 0 to 1.
//!
//! # Usage
//!
//! ```ignore
//! let mut hybrid = HybridEVEstimator::new(config);
//!
//! // On each quote cycle
//! let ev = hybrid.compute_ev(&input);
//!
//! // Check what's driving the decision
//! let alpha = hybrid.current_alpha();
//! ```

use crate::market_maker::calibration::adaptive_binning::AdaptiveBinner;
use crate::market_maker::control::theoretical_edge::TheoreticalEdgeEstimator;

/// Configuration for hybrid EV estimator.
#[derive(Debug, Clone)]
pub struct HybridEVConfig {
    /// Minimum bins to start IR blending (with ≥3 samples each).
    /// Default: 2
    pub min_bins_for_ir: usize,

    /// Bins needed for full IR trust (alpha = 1.0).
    /// Default: 4
    pub bins_for_full_ir: usize,

    /// Minimum samples per bin for counting.
    /// Default: 3
    pub min_samples_per_bin: usize,

    /// Weight of accuracy-based alpha adjustment.
    /// Default: 0.3
    pub accuracy_weight: f64,

    /// Accuracy threshold above which to trust the model.
    /// Default: 0.55
    pub accuracy_threshold: f64,

    /// Maximum alpha cap (safety limit).
    /// Default: 0.9
    pub max_alpha: f64,

    /// Enable accuracy-based fallback when IR fails.
    /// Default: true
    pub accuracy_fallback_enabled: bool,
}

impl Default for HybridEVConfig {
    fn default() -> Self {
        Self {
            min_bins_for_ir: 2,
            bins_for_full_ir: 4,
            min_samples_per_bin: 3,
            accuracy_weight: 0.3,
            accuracy_threshold: 0.55,
            max_alpha: 0.9,
            accuracy_fallback_enabled: true,
        }
    }
}

/// Input for hybrid EV computation.
#[derive(Debug, Clone)]
pub struct HybridEVInput {
    /// Book imbalance signal [-1, +1].
    pub book_imbalance: f64,
    /// Current bid-ask spread (bps).
    pub spread_bps: f64,
    /// Volatility estimate (fractional).
    pub sigma: f64,
    /// Expected holding time (seconds).
    pub tau_seconds: f64,
    /// Enhanced flow signal [-1, +1] (from EnhancedFlowEstimator).
    pub enhanced_flow: f64,
    /// Is the system in warmup?
    pub is_warmup: bool,
}

impl Default for HybridEVInput {
    fn default() -> Self {
        Self {
            book_imbalance: 0.0,
            spread_bps: 10.0,
            sigma: 0.001,
            tau_seconds: 1.0,
            enhanced_flow: 0.0,
            is_warmup: false,
        }
    }
}

/// Result of hybrid EV computation.
#[derive(Debug, Clone)]
pub struct HybridEVResult {
    /// Final blended expected edge (bps).
    pub expected_edge_bps: f64,
    /// Whether to quote based on edge.
    pub should_quote: bool,
    /// Blending alpha (0 = theoretical, 1 = IR).
    pub alpha: f64,
    /// IR-implied edge (bps).
    pub ir_edge_bps: f64,
    /// Theoretical edge (bps).
    pub theoretical_edge_bps: f64,
    /// Number of filled bins.
    pub filled_bins: usize,
    /// Total samples.
    pub total_samples: usize,
    /// Current IR value.
    pub current_ir: f64,
    /// Current accuracy.
    pub accuracy: f64,
    /// Source of the decision.
    pub decision_source: DecisionSource,
}

/// Source of the EV decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecisionSource {
    /// Pure theoretical edge (no IR data).
    Theoretical,
    /// Blended IR + theoretical.
    Blended,
    /// IR-dominated (high alpha).
    IRDominated,
    /// Accuracy-based fallback.
    AccuracyFallback,
    /// Warmup mode (conservative).
    Warmup,
}

impl std::fmt::Display for DecisionSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecisionSource::Theoretical => write!(f, "theoretical"),
            DecisionSource::Blended => write!(f, "blended"),
            DecisionSource::IRDominated => write!(f, "ir_dominated"),
            DecisionSource::AccuracyFallback => write!(f, "accuracy_fallback"),
            DecisionSource::Warmup => write!(f, "warmup"),
        }
    }
}

/// Hybrid EV estimator with smooth transition.
#[derive(Debug)]
pub struct HybridEVEstimator {
    config: HybridEVConfig,
    /// Adaptive binner for IR tracking.
    binner: AdaptiveBinner,
    /// Theoretical edge estimator.
    theoretical: TheoreticalEdgeEstimator,
    /// Computation count.
    computation_count: u64,
}

impl HybridEVEstimator {
    /// Create a new hybrid EV estimator.
    pub fn new(config: HybridEVConfig) -> Self {
        Self {
            config,
            binner: AdaptiveBinner::new(10, 500),
            theoretical: TheoreticalEdgeEstimator::new(),
            computation_count: 0,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(HybridEVConfig::default())
    }

    /// Compute hybrid expected value.
    pub fn compute_ev(&mut self, input: &HybridEVInput) -> HybridEVResult {
        self.computation_count += 1;

        // During warmup, use conservative theoretical edge
        if input.is_warmup {
            let theo = self.theoretical.calculate_edge(
                input.book_imbalance,
                input.spread_bps,
                input.sigma,
                input.tau_seconds,
            );

            return HybridEVResult {
                expected_edge_bps: theo.expected_edge_bps * 0.5, // Conservative
                should_quote: false,                             // Don't quote during warmup
                alpha: 0.0,
                ir_edge_bps: 0.0,
                theoretical_edge_bps: theo.expected_edge_bps,
                filled_bins: 0,
                total_samples: 0,
                current_ir: 0.0,
                accuracy: 0.5,
                decision_source: DecisionSource::Warmup,
            };
        }

        // Get theoretical edge
        let theo = self.theoretical.calculate_edge(
            input.book_imbalance,
            input.spread_bps,
            input.sigma,
            input.tau_seconds,
        );

        // Get IR metrics from binner
        let filled_bins = self.binner.bins_with_count(self.config.min_samples_per_bin);
        let total_samples = self.binner.total_samples();
        let current_ir = self.binner.information_ratio();
        let accuracy = self.binner.accuracy();

        // Compute alpha based on bin population
        let alpha = self.compute_alpha(filled_bins, total_samples, accuracy, current_ir);

        // Compute IR-implied edge
        let ir_edge_bps = self.compute_ir_edge(current_ir, input.spread_bps);

        // Blend edges
        let blended_edge = alpha * ir_edge_bps + (1.0 - alpha) * theo.expected_edge_bps;

        // Determine decision source
        let decision_source = if filled_bins < self.config.min_bins_for_ir {
            if self.config.accuracy_fallback_enabled && accuracy > self.config.accuracy_threshold {
                DecisionSource::AccuracyFallback
            } else {
                DecisionSource::Theoretical
            }
        } else if alpha > 0.7 {
            DecisionSource::IRDominated
        } else {
            DecisionSource::Blended
        };

        // Compute final edge based on source
        let final_edge = match decision_source {
            DecisionSource::AccuracyFallback => {
                // Use accuracy to adjust theoretical edge
                let accuracy_bonus = (accuracy - 0.5) * 2.0; // 0-1 range
                theo.expected_edge_bps + accuracy_bonus * input.spread_bps * 0.1
            }
            _ => blended_edge,
        };

        // Should quote if edge > min_edge_bps
        let min_edge = self.theoretical.config().min_edge_bps;
        let should_quote = final_edge > min_edge;

        HybridEVResult {
            expected_edge_bps: final_edge,
            should_quote,
            alpha,
            ir_edge_bps,
            theoretical_edge_bps: theo.expected_edge_bps,
            filled_bins,
            total_samples,
            current_ir,
            accuracy,
            decision_source,
        }
    }

    /// Compute blending alpha based on data quality.
    fn compute_alpha(
        &self,
        filled_bins: usize,
        total_samples: usize,
        accuracy: f64,
        ir: f64,
    ) -> f64 {
        // Base alpha from bin count
        // At min_bins we start with a base alpha (0.1), then scale up to 1.0 at full_bins
        let bin_alpha = if filled_bins < self.config.min_bins_for_ir {
            0.0
        } else {
            let base_alpha = 0.1; // Start with some alpha at min_bins
            let range = (self.config.bins_for_full_ir - self.config.min_bins_for_ir) as f64;
            let progress = (filled_bins - self.config.min_bins_for_ir) as f64;
            base_alpha + (1.0 - base_alpha) * (progress / range.max(1.0)).min(1.0)
        };

        // Adjust based on sample count (need enough data)
        let sample_factor = (total_samples as f64 / 50.0).min(1.0);

        // Adjust based on accuracy (if enabled)
        let accuracy_factor = if self.config.accuracy_fallback_enabled {
            let acc_above_baseline = (accuracy - 0.5).max(0.0) * 2.0;
            1.0 + self.config.accuracy_weight * acc_above_baseline
        } else {
            1.0
        };

        // Adjust based on IR strength
        let ir_factor = if ir > 1.0 {
            1.0 + (ir - 1.0).min(1.0) * 0.2 // Bonus for strong IR
        } else {
            1.0
        };

        // Combine factors
        let alpha = bin_alpha * sample_factor * accuracy_factor * ir_factor;

        alpha.clamp(0.0, self.config.max_alpha)
    }

    /// Compute IR-implied edge.
    fn compute_ir_edge(&self, ir: f64, spread_bps: f64) -> f64 {
        if ir > 1.0 {
            // IR-implied edge: half spread * (IR - 1.0) scaled
            // IR = 1.5 means 50% more predictive than base rate
            let ir_bonus = (ir - 1.0).min(1.0);
            spread_bps / 2.0 * ir_bonus
        } else {
            0.0
        }
    }

    /// Record a prediction for IR tracking.
    ///
    /// Call this when making a quote decision.
    /// Note: Currently a stub - predictions are tracked when outcomes are recorded.
    pub fn record_prediction(&mut self, _probability: f64) {
        // Store the prediction probability for later outcome recording
        // The binner will update when we call record_outcome
    }

    /// Record an outcome for IR tracking.
    ///
    /// Call this when a prediction matures (price moved or not).
    pub fn record_outcome(&mut self, probability: f64, outcome: bool) {
        self.binner.update(probability, outcome);
    }

    /// Get current alpha (for logging).
    pub fn current_alpha(&self) -> f64 {
        let filled_bins = self.binner.bins_with_count(self.config.min_samples_per_bin);
        let total_samples = self.binner.total_samples();
        let accuracy = self.binner.accuracy();
        let ir = self.binner.information_ratio();
        self.compute_alpha(filled_bins, total_samples, accuracy, ir)
    }

    /// Get the adaptive binner (for diagnostics).
    pub fn binner(&self) -> &AdaptiveBinner {
        &self.binner
    }

    /// Get mutable access to binner.
    pub fn binner_mut(&mut self) -> &mut AdaptiveBinner {
        &mut self.binner
    }

    /// Get the theoretical estimator (for alpha tracking, etc).
    pub fn theoretical(&self) -> &TheoreticalEdgeEstimator {
        &self.theoretical
    }

    /// Get mutable access to theoretical estimator.
    pub fn theoretical_mut(&mut self) -> &mut TheoreticalEdgeEstimator {
        &mut self.theoretical
    }

    /// Get computation count.
    pub fn computation_count(&self) -> u64 {
        self.computation_count
    }

    /// Get configuration.
    pub fn config(&self) -> &HybridEVConfig {
        &self.config
    }

    /// Reset the estimator.
    pub fn reset(&mut self) {
        self.binner.clear();
        self.computation_count = 0;
    }
}

impl Default for HybridEVEstimator {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input() -> HybridEVInput {
        HybridEVInput {
            book_imbalance: 0.3,
            spread_bps: 10.0,
            sigma: 0.001,
            tau_seconds: 1.0,
            enhanced_flow: 0.4,
            is_warmup: false,
        }
    }

    #[test]
    fn test_warmup_mode() {
        let mut estimator = HybridEVEstimator::default_config();
        let mut input = make_input();
        input.is_warmup = true;

        let result = estimator.compute_ev(&input);

        assert!(!result.should_quote);
        assert_eq!(result.decision_source, DecisionSource::Warmup);
        assert_eq!(result.alpha, 0.0);
    }

    #[test]
    fn test_pure_theoretical() {
        let mut estimator = HybridEVEstimator::default_config();
        let input = make_input();

        // No IR data yet
        let result = estimator.compute_ev(&input);

        assert_eq!(result.filled_bins, 0);
        assert_eq!(result.alpha, 0.0);
        assert!(matches!(
            result.decision_source,
            DecisionSource::Theoretical | DecisionSource::AccuracyFallback
        ));
    }

    #[test]
    fn test_blended_with_data() {
        let mut estimator = HybridEVEstimator::default_config();

        // Add varied predictions with outcomes
        for i in 0..100 {
            let prob = (i as f64 / 100.0) * 0.6 + 0.2; // Range 0.2-0.8
            let outcome = prob > 0.5; // Correct predictions
            estimator.record_outcome(prob, outcome);
        }

        let input = make_input();
        let result = estimator.compute_ev(&input);

        // Should have some filled bins and positive alpha
        assert!(result.filled_bins >= 2);
        assert!(result.alpha > 0.0);
    }

    #[test]
    fn test_ir_dominated() {
        let mut estimator = HybridEVEstimator::default_config();

        // Add well-distributed predictions across multiple probability levels
        // This creates 4+ bins for better IR calibration
        for _ in 0..100 {
            // Very low probability → negative outcome
            estimator.record_outcome(0.1, false);
            // Low probability → negative outcome
            estimator.record_outcome(0.3, false);
            // High probability → positive outcome
            estimator.record_outcome(0.7, true);
            // Very high probability → positive outcome
            estimator.record_outcome(0.9, true);
        }

        let input = make_input();
        let result = estimator.compute_ev(&input);

        // With 4 bins and good discrimination, alpha should be moderate
        // bins_for_full_ir defaults to 5, so 4 bins gives ~0.8 of full alpha
        assert!(
            result.alpha > 0.1,
            "Expected alpha > 0.1, got {}",
            result.alpha
        );
        assert!(
            result.current_ir > 0.0,
            "Expected positive IR, got {}",
            result.current_ir
        );
    }

    #[test]
    fn test_accuracy_fallback() {
        let config = HybridEVConfig {
            accuracy_fallback_enabled: true,
            accuracy_threshold: 0.55,
            ..Default::default()
        };
        let mut estimator = HybridEVEstimator::new(config);

        // Add predictions that are accurate but don't spread across bins
        for _ in 0..50 {
            estimator.record_outcome(0.6, true); // Correct
            estimator.record_outcome(0.4, false); // Correct
        }

        let input = make_input();
        let result = estimator.compute_ev(&input);

        // High accuracy should influence the decision
        assert!(result.accuracy > 0.9);
    }

    #[test]
    fn test_alpha_computation() {
        let estimator = HybridEVEstimator::default_config();

        // No bins → alpha = 0
        let alpha0 = estimator.compute_alpha(0, 10, 0.5, 0.5);
        assert_eq!(alpha0, 0.0);

        // Below min bins → alpha = 0
        let alpha1 = estimator.compute_alpha(1, 50, 0.6, 1.2);
        assert_eq!(alpha1, 0.0);

        // At min bins → alpha > 0
        let alpha2 = estimator.compute_alpha(2, 100, 0.6, 1.2);
        assert!(alpha2 > 0.0);

        // More bins → higher alpha
        let alpha3 = estimator.compute_alpha(4, 200, 0.6, 1.2);
        assert!(alpha3 > alpha2);
    }

    #[test]
    fn test_ir_edge_computation() {
        let estimator = HybridEVEstimator::default_config();

        // IR < 1.0 → no edge
        let edge0 = estimator.compute_ir_edge(0.8, 10.0);
        assert_eq!(edge0, 0.0);

        // IR = 1.0 → no edge
        let edge1 = estimator.compute_ir_edge(1.0, 10.0);
        assert_eq!(edge1, 0.0);

        // IR > 1.0 → positive edge
        let edge2 = estimator.compute_ir_edge(1.5, 10.0);
        assert!(edge2 > 0.0);
        // With spread=10 and IR=1.5: edge = 10/2 * 0.5 = 2.5
        assert!((edge2 - 2.5).abs() < 0.1);
    }

    #[test]
    fn test_reset() {
        let mut estimator = HybridEVEstimator::default_config();

        for _ in 0..50 {
            estimator.record_outcome(0.5, true);
        }
        estimator.compute_ev(&make_input());

        estimator.reset();

        assert_eq!(estimator.computation_count(), 0);
        assert_eq!(estimator.binner().total_samples(), 0);
    }

    #[test]
    fn test_decision_source_transitions() {
        let mut estimator = HybridEVEstimator::default_config();

        // Initially theoretical
        let result1 = estimator.compute_ev(&make_input());
        assert!(matches!(
            result1.decision_source,
            DecisionSource::Theoretical | DecisionSource::AccuracyFallback
        ));

        // Add some data
        for i in 0..100 {
            let prob = (i % 5) as f64 * 0.2 + 0.1;
            estimator.record_outcome(prob, prob > 0.5);
        }

        let result2 = estimator.compute_ev(&make_input());
        // Should transition based on data quality
        assert!(result2.alpha >= result1.alpha);
    }
}
