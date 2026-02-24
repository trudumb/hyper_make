//! Position Continuation Model using Beta-Binomial Bayesian inference.
//!
//! This module implements the mathematical foundation for HOLD/ADD/REDUCE
//! position decisions. Instead of always mean-reverting via inventory skew,
//! we estimate P(continuation | fills, regime) to decide whether positions
//! should be held, added to, or reduced.
//!
//! ## Key Insight
//!
//! GLFT inventory skew formula `skew = q × γ × σ² × T` always mean-reverts.
//! This causes repeated "buy at P, sell at P-Δ" cycles with negative EV after costs.
//!
//! The Position Continuation Model solves this by:
//! - **HOLD**: Set `inventory_ratio = 0` → no skew → symmetric quotes
//! - **ADD**: Set `inventory_ratio < 0` → REVERSE skew → tighter on position-building side
//! - **REDUCE**: Normal behavior with urgency scaling
//!
//! ## Mathematical Foundation
//!
//! Beta-Binomial Conjugate Prior for P(continuation | fills, regime):
//!
//! ```text
//! P(cont) ~ Beta(α, β)
//! Prior: α₀, β₀ from regime (cascade=0.8, normal=0.5, quiet=0.3)
//! Update: α += aligned_fills, β += adverse_fills
//! Posterior mean: p_cont = α / (α + β)
//! Confidence: conf = 1 - Var[Beta(α,β)] / Var[Beta(1,1)]
//! ```
//!
//! ## Enhanced Multi-Signal Fusion (v2)
//!
//! The enhanced model fuses multiple information sources:
//!
//! ```text
//! P(cont) = fuse(
//!     p_fill,       # Beta-Binomial from fill observations
//!     p_momentum,   # Momentum continuation probability
//!     p_regime,     # Regime-based prior from HMM
//!     p_trend       # Multi-timeframe trend agreement
//! ) × (1 - changepoint_discount)
//! ```
//!
//! **Changepoint-Aware Discounting**: When BOCD detects a regime change,
//! the fill-based posterior is discounted and blended toward the regime prior.
//! This prevents stale fill history from dominating after market structure changes.

/// Configuration for multi-signal fusion weights.
#[derive(Debug, Clone)]
pub struct ContinuationFusionConfig {
    /// Weight for fill-based posterior (default: 0.4)
    pub weight_fills: f64,
    /// Weight for momentum continuation signal (default: 0.25)
    pub weight_momentum: f64,
    /// Weight for trend agreement signal (default: 0.2)
    pub weight_trend: f64,
    /// Weight for regime prior (default: 0.15)
    pub weight_regime: f64,
    /// Scale factor for changepoint entropy (default: 1.0)
    pub changepoint_entropy_scale: f64,
}

impl Default for ContinuationFusionConfig {
    fn default() -> Self {
        Self {
            weight_fills: 0.4,
            weight_momentum: 0.25,
            weight_trend: 0.2,
            weight_regime: 0.15,
            changepoint_entropy_scale: 1.0,
        }
    }
}

/// Beta-Binomial model for position continuation probability.
///
/// Maintains a Beta distribution posterior over the probability that
/// the current position direction will continue to be profitable.
///
/// ## Enhanced Features (v2)
///
/// The model now supports multi-signal fusion via `update_signals()` and
/// `prob_continuation_fused()`. External signals include:
/// - BOCD changepoint probability and run-length entropy
/// - Momentum continuation probability
/// - Multi-timeframe trend agreement
/// - HMM regime probabilities
#[derive(Debug, Clone)]
pub struct ContinuationPosterior {
    /// Alpha parameter (aligned fill successes + prior)
    pub alpha: f64,
    /// Beta parameter (adverse fill failures + prior)
    pub beta: f64,

    // === External Signal Inputs (set by orchestrator) ===
    /// Changepoint probability from BOCD (P(changepoint in last k ticks))
    pub changepoint_prob: f64,
    /// Run-length entropy from BOCD (measures distribution uncertainty)
    pub changepoint_entropy: f64,
    /// Momentum continuation probability from MomentumModel
    pub momentum_continuation: f64,
    /// Multi-timeframe trend agreement [0, 1] from TrendPersistenceTracker
    pub trend_agreement: f64,
    /// Trend confidence [0, 1] from TrendSignal
    pub trend_confidence: f64,
    /// Regime probabilities [quiet, normal, bursty, cascade] from HMM
    pub regime_probs: [f64; 4],

    // === Fusion Configuration ===
    /// Fusion weights and configuration
    pub fusion_config: ContinuationFusionConfig,
}

impl Default for ContinuationPosterior {
    fn default() -> Self {
        // Start with neutral prior (Beta(2.5, 2.5) → mean = 0.5)
        Self {
            alpha: 2.5,
            beta: 2.5,
            // Initialize external signals to neutral values
            changepoint_prob: 0.0,
            changepoint_entropy: 2.0, // Moderate entropy
            momentum_continuation: 0.5,
            trend_agreement: 0.0,
            trend_confidence: 0.0,
            regime_probs: [0.2, 0.5, 0.2, 0.1], // Default: mostly normal
            fusion_config: ContinuationFusionConfig::default(),
        }
    }
}

impl ContinuationPosterior {
    /// Initialize from regime with appropriate prior.
    ///
    /// Different regimes have different continuation probabilities:
    /// - **cascade**: High continuation (momentum persists), prior mean = 0.8
    /// - **bursty**: Moderate-high continuation, prior mean = 0.6
    /// - **normal**: Neutral, prior mean = 0.5
    /// - **quiet**: Low continuation (mean-reversion dominates), prior mean = 0.3
    pub fn from_regime(regime: &str) -> Self {
        let (alpha, beta) = match regime.to_lowercase().as_str() {
            "cascade" | "extreme" => (4.0, 1.0), // Prior mean = 0.8
            "bursty" | "high" => (3.0, 2.0),     // Prior mean = 0.6
            "normal" => (2.5, 2.5),              // Prior mean = 0.5
            "quiet" | "low" => (1.5, 3.5),       // Prior mean = 0.3
            _ => (2.5, 2.5),                     // Default to neutral
        };
        Self {
            alpha,
            beta,
            ..Default::default()
        }
    }

    /// Create with explicit parameters.
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self {
            alpha: alpha.max(0.1),
            beta: beta.max(0.1),
            ..Default::default()
        }
    }

    /// Create with custom fusion configuration.
    pub fn with_fusion_config(alpha: f64, beta: f64, config: ContinuationFusionConfig) -> Self {
        Self {
            alpha: alpha.max(0.1),
            beta: beta.max(0.1),
            fusion_config: config,
            ..Default::default()
        }
    }

    /// Update posterior with fill observation.
    ///
    /// # Arguments
    /// * `is_aligned` - True if fill was in the direction of the position
    /// * `weight` - Observation weight (typically sqrt(size) for size-weighted updates)
    pub fn observe_fill(&mut self, is_aligned: bool, weight: f64) {
        let w = weight.clamp(0.01, 5.0); // Clamp weight to reasonable range
        if is_aligned {
            self.alpha += w;
        } else {
            self.beta += w;
        }
    }

    /// Posterior mean P(continuation).
    ///
    /// This is the expected probability that the position direction
    /// will continue to be profitable.
    pub fn prob_continuation(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Confidence in the estimate [0, 1].
    ///
    /// Based on variance reduction from uniform prior:
    /// conf = 1 - Var[Beta(α,β)] / Var[Beta(1,1)]
    ///
    /// Higher α + β → lower variance → higher confidence.
    pub fn confidence(&self) -> f64 {
        let n = self.alpha + self.beta;
        let uniform_var = 1.0 / 12.0; // Var[Beta(1,1)] = 1/12
        let posterior_var = (self.alpha * self.beta) / (n * n * (n + 1.0));
        (1.0 - posterior_var / uniform_var).max(0.0)
    }

    // =========================================================================
    // Enhanced Multi-Signal Fusion (v2)
    // =========================================================================

    /// Update external signals from orchestrator.
    ///
    /// Call this each quote cycle with the latest signal values:
    /// - `changepoint_prob`: From BOCD `changepoint_probability(5)`
    /// - `changepoint_entropy`: From BOCD `run_length_entropy()`
    /// - `momentum_continuation`: From MomentumModel `continuation_probability()`
    /// - `trend_agreement`: From TrendSignal `timeframe_agreement`
    /// - `trend_confidence`: From TrendSignal `trend_confidence`
    /// - `regime_probs`: From HMM `regime_probabilities()` [quiet, normal, bursty, cascade]
    pub fn update_signals(
        &mut self,
        changepoint_prob: f64,
        changepoint_entropy: f64,
        momentum_continuation: f64,
        trend_agreement: f64,
        trend_confidence: f64,
        regime_probs: [f64; 4],
    ) {
        self.changepoint_prob = changepoint_prob.clamp(0.0, 1.0);
        self.changepoint_entropy = changepoint_entropy.max(0.01);
        self.momentum_continuation = momentum_continuation.clamp(0.0, 1.0);
        self.trend_agreement = trend_agreement.clamp(0.0, 1.0);
        self.trend_confidence = trend_confidence.clamp(0.0, 1.0);
        self.regime_probs = regime_probs;
    }

    /// Compute regime-based continuation prior from HMM probabilities.
    ///
    /// Different regimes have different base continuation rates:
    /// - Quiet: 0.3 (mean-reversion dominates)
    /// - Normal: 0.5 (neutral)
    /// - Bursty: 0.65 (moderate momentum)
    /// - Cascade: 0.8 (strong momentum persistence)
    fn regime_prior(&self) -> f64 {
        self.regime_probs[0] * 0.3   // quiet
            + self.regime_probs[1] * 0.5   // normal
            + self.regime_probs[2] * 0.65  // bursty
            + self.regime_probs[3] * 0.8 // cascade
    }

    /// Compute changepoint discount factor.
    ///
    /// When BOCD detects a changepoint (high probability + low entropy),
    /// we discount the fill-based posterior and blend toward the regime prior.
    ///
    /// Returns: discount factor [0, 1] where 0 = no discount, 1 = full discount
    fn changepoint_discount(&self) -> f64 {
        // Entropy-based confidence: low entropy = concentrated distribution = trust the signal
        // High entropy = spread distribution = uncertain about changepoint
        let scaled_entropy =
            self.changepoint_entropy * self.fusion_config.changepoint_entropy_scale;
        let entropy_confidence = 1.0 / (1.0 + scaled_entropy);

        // Discount = probability × confidence
        // High changepoint prob + low entropy = strong discount
        (self.changepoint_prob * entropy_confidence).clamp(0.0, 0.8) // Cap at 80% discount
    }

    /// Fused continuation probability using all available signals.
    ///
    /// This is the PREFERRED method for position decisions. It combines:
    /// 1. Fill-based posterior (discounted by changepoint detection)
    /// 2. Momentum continuation probability
    /// 3. Multi-timeframe trend agreement
    /// 4. Regime-based prior from HMM
    ///
    /// The fusion uses confidence-weighted averaging where each signal
    /// contributes according to its configured weight and confidence level.
    pub fn prob_continuation_fused(&self) -> f64 {
        // 1. Fill-based posterior (simple Beta mean)
        let p_fill = self.alpha / (self.alpha + self.beta);
        let fill_conf = self.confidence();

        // 2. Regime-based prior
        let p_regime = self.regime_prior();

        // 3. Apply changepoint discount to fill posterior
        // When changepoint detected, blend fill posterior toward regime prior
        let cp_discount = self.changepoint_discount();
        let p_fill_discounted = (1.0 - cp_discount) * p_fill + cp_discount * p_regime;

        // 4. Convert trend agreement to continuation probability
        // trend_agreement in [0, 1] → p_trend in [0.5, 1.0]
        let p_trend = 0.5 + 0.5 * self.trend_agreement;

        // 5. Confidence-weighted fusion
        // Each signal contributes: weight × confidence × probability
        let cfg = &self.fusion_config;
        let signals = [
            (p_fill_discounted, cfg.weight_fills * fill_conf),
            (self.momentum_continuation, cfg.weight_momentum),
            (p_trend, cfg.weight_trend * self.trend_confidence),
            (p_regime, cfg.weight_regime),
        ];

        let (total_weight, weighted_sum) = signals
            .iter()
            .fold((0.0, 0.0), |(w, s), (p, conf)| (w + conf, s + p * conf));

        let p_fused = if total_weight > 1e-6 {
            weighted_sum / total_weight
        } else {
            p_regime // Fallback to regime prior if no confident signals
        };

        p_fused.clamp(0.0, 1.0)
    }

    /// Fused confidence across all signals.
    ///
    /// Combines confidence from:
    /// - Fill posterior confidence (variance-based)
    /// - Changepoint confidence (inverse: low prob = high conf)
    /// - Trend confidence
    pub fn confidence_fused(&self) -> f64 {
        let fill_conf = self.confidence();
        let cp_conf = 1.0 - self.changepoint_prob; // High changepoint = low confidence
        let trend_conf = self.trend_confidence;

        // Geometric mean of confidences
        let product = fill_conf * cp_conf * (0.3 + 0.7 * trend_conf); // Trend has floor of 0.3
        product.powf(1.0 / 3.0).clamp(0.0, 1.0)
    }

    /// Check if the model is receiving external signal updates.
    ///
    /// Returns true if signals have been updated from default values.
    pub fn has_signal_updates(&self) -> bool {
        // Check if any signals differ from defaults
        self.changepoint_prob > 0.01
            || (self.changepoint_entropy - 2.0).abs() > 0.1
            || (self.momentum_continuation - 0.5).abs() > 0.01
            || self.trend_agreement > 0.01
            || self.trend_confidence > 0.01
    }

    /// Get diagnostic summary of signal contributions.
    pub fn signal_summary(&self) -> ContinuationSignalSummary {
        let p_fill = self.alpha / (self.alpha + self.beta);
        ContinuationSignalSummary {
            p_fill_raw: p_fill,
            p_fill_discounted: (1.0 - self.changepoint_discount()) * p_fill
                + self.changepoint_discount() * self.regime_prior(),
            p_regime: self.regime_prior(),
            p_momentum: self.momentum_continuation,
            p_trend: 0.5 + 0.5 * self.trend_agreement,
            p_fused: self.prob_continuation_fused(),
            changepoint_discount: self.changepoint_discount(),
            fill_confidence: self.confidence(),
            fused_confidence: self.confidence_fused(),
        }
    }

    /// Effective sample size (α + β - 2).
    ///
    /// Represents how many observations have been incorporated
    /// beyond the uniform prior.
    pub fn effective_n(&self) -> f64 {
        (self.alpha + self.beta - 2.0).max(0.0)
    }

    /// Decay posterior toward prior over time.
    ///
    /// Applies exponential decay to prevent stale beliefs from dominating.
    /// Useful for non-stationary markets where continuation probability changes.
    ///
    /// # Arguments
    /// * `half_life_fills` - Number of fills for half-life decay
    pub fn decay(&mut self, half_life_fills: f64) {
        if half_life_fills <= 0.0 {
            return;
        }
        let factor = 0.5_f64.powf(1.0 / half_life_fills.max(1.0));
        // Decay toward prior (α=1, β=1 for uniform, but we use regime prior)
        // Keep at least 1.0 for both to maintain proper Beta distribution
        self.alpha = 1.0 + (self.alpha - 1.0) * factor;
        self.beta = 1.0 + (self.beta - 1.0) * factor;
    }

    /// Reset to regime prior.
    pub fn reset_to_regime(&mut self, regime: &str) {
        let fresh = Self::from_regime(regime);
        self.alpha = fresh.alpha;
        self.beta = fresh.beta;
    }

    /// Get the 95% credible interval for P(continuation).
    ///
    /// Returns (lower, upper) bounds using the beta quantile approximation.
    pub fn credible_interval_95(&self) -> (f64, f64) {
        // Use normal approximation for Beta quantiles when α, β > 1
        let mean = self.prob_continuation();
        let var = (self.alpha * self.beta)
            / ((self.alpha + self.beta).powi(2) * (self.alpha + self.beta + 1.0));
        let std = var.sqrt();

        // 95% CI ≈ mean ± 1.96 × std
        let lower = (mean - 1.96 * std).max(0.0);
        let upper = (mean + 1.96 * std).min(1.0);
        (lower, upper)
    }

    /// Set fusion configuration.
    pub fn set_fusion_config(&mut self, config: ContinuationFusionConfig) {
        self.fusion_config = config;
    }
}

/// Diagnostic summary of signal contributions to continuation probability.
#[derive(Debug, Clone)]
pub struct ContinuationSignalSummary {
    /// Raw fill-based posterior mean
    pub p_fill_raw: f64,
    /// Fill posterior after changepoint discount
    pub p_fill_discounted: f64,
    /// Regime-based prior
    pub p_regime: f64,
    /// Momentum continuation probability
    pub p_momentum: f64,
    /// Trend-based probability
    pub p_trend: f64,
    /// Final fused probability
    pub p_fused: f64,
    /// Changepoint discount applied
    pub changepoint_discount: f64,
    /// Fill-based confidence
    pub fill_confidence: f64,
    /// Fused confidence
    pub fused_confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_neutral() {
        let post = ContinuationPosterior::default();
        let p = post.prob_continuation();
        assert!(
            (p - 0.5).abs() < 0.01,
            "Default should be neutral, got {}",
            p
        );
    }

    #[test]
    fn test_regime_priors() {
        let cascade = ContinuationPosterior::from_regime("cascade");
        let quiet = ContinuationPosterior::from_regime("quiet");

        assert!(
            cascade.prob_continuation() > 0.7,
            "Cascade prior should be high"
        );
        assert!(quiet.prob_continuation() < 0.4, "Quiet prior should be low");
    }

    #[test]
    fn test_observe_fill_updates() {
        let mut post = ContinuationPosterior::default();
        let initial_p = post.prob_continuation();

        // Observe aligned fill
        post.observe_fill(true, 1.0);
        assert!(
            post.prob_continuation() > initial_p,
            "Aligned fill should increase p_cont"
        );

        // Observe adverse fill
        let p_after_aligned = post.prob_continuation();
        post.observe_fill(false, 1.0);
        assert!(
            post.prob_continuation() < p_after_aligned,
            "Adverse fill should decrease p_cont"
        );
    }

    #[test]
    fn test_confidence_increases_with_observations() {
        let mut post = ContinuationPosterior::default();
        let initial_conf = post.confidence();

        for _ in 0..10 {
            post.observe_fill(true, 1.0);
        }

        assert!(
            post.confidence() > initial_conf,
            "Confidence should increase with observations"
        );
    }

    #[test]
    fn test_decay_moves_toward_prior() {
        let mut post = ContinuationPosterior::from_regime("normal");

        // Add lots of aligned observations
        for _ in 0..20 {
            post.observe_fill(true, 1.0);
        }
        let p_before_decay = post.prob_continuation();

        // Decay multiple times
        for _ in 0..50 {
            post.decay(10.0);
        }

        // Should move back toward 0.5 (normal regime prior mean)
        let p_after_decay = post.prob_continuation();
        assert!(
            (p_after_decay - 0.5).abs() < (p_before_decay - 0.5).abs(),
            "Decay should move toward prior: before={}, after={}",
            p_before_decay,
            p_after_decay
        );
    }

    #[test]
    fn test_credible_interval() {
        let post = ContinuationPosterior::new(10.0, 10.0);
        let (lower, upper) = post.credible_interval_95();

        assert!(
            lower < 0.5 && upper > 0.5,
            "CI should contain 0.5 for symmetric Beta(10,10)"
        );
        assert!(lower >= 0.0 && upper <= 1.0, "CI should be in [0,1]");
    }

    // =========================================================================
    // Enhanced Multi-Signal Fusion Tests
    // =========================================================================

    #[test]
    fn test_fused_defaults_to_fill_based() {
        let post = ContinuationPosterior::default();
        let p_fill = post.prob_continuation();
        let p_fused = post.prob_continuation_fused();

        // With default signals, fused should be close to fill-based
        // (regime prior is 0.5, which equals default fill prior)
        assert!(
            (p_fill - p_fused).abs() < 0.1,
            "Default fused should be close to fill-based: fill={}, fused={}",
            p_fill,
            p_fused
        );
    }

    #[test]
    fn test_changepoint_discounts_fill_history() {
        let mut post = ContinuationPosterior::default();

        // Build up high continuation probability from fills
        for _ in 0..30 {
            post.observe_fill(true, 1.0);
        }

        // Also set supportive signals to get a high baseline
        post.update_signals(
            0.0,                  // No changepoint initially
            2.0,                  // Normal entropy
            0.7,                  // Supportive momentum
            0.5,                  // Some trend agreement
            0.5,                  // Some trend confidence
            [0.1, 0.4, 0.3, 0.2], // Trending regime (bursty + cascade)
        );

        let p_before_cp = post.prob_continuation_fused();
        assert!(
            p_before_cp > 0.55,
            "Should have elevated p_cont before changepoint: {}",
            p_before_cp
        );

        // Simulate changepoint detection (high probability, low entropy)
        post.update_signals(
            0.9,                  // High changepoint probability
            0.5,                  // Low entropy (confident)
            0.5,                  // Neutral momentum
            0.0,                  // No trend agreement
            0.0,                  // No trend confidence
            [0.2, 0.5, 0.2, 0.1], // Normal regime
        );

        let p_after_cp = post.prob_continuation_fused();

        // With changepoint, should discount fill history toward regime prior (0.5)
        assert!(
            p_after_cp < p_before_cp,
            "Changepoint should discount fill history: before={}, after={}",
            p_before_cp,
            p_after_cp
        );
    }

    #[test]
    fn test_momentum_agreement_boosts_continuation() {
        let mut post = ContinuationPosterior::default();

        // Neutral fills
        let p_neutral = post.prob_continuation_fused();

        // Add momentum continuation signal
        post.update_signals(
            0.0,                  // No changepoint
            2.0,                  // Normal entropy
            0.8,                  // High momentum continuation
            0.0,                  // No trend
            0.0,                  // No trend confidence
            [0.2, 0.5, 0.2, 0.1], // Normal regime
        );

        let p_with_momentum = post.prob_continuation_fused();

        assert!(
            p_with_momentum > p_neutral,
            "Momentum should boost continuation: neutral={}, with_momentum={}",
            p_neutral,
            p_with_momentum
        );
    }

    #[test]
    fn test_trend_agreement_boosts_continuation() {
        let mut post = ContinuationPosterior::default();

        // Add strong trend agreement with confidence
        post.update_signals(
            0.0,                  // No changepoint
            2.0,                  // Normal entropy
            0.5,                  // Neutral momentum
            0.9,                  // Strong trend agreement
            0.8,                  // High trend confidence
            [0.2, 0.5, 0.2, 0.1], // Normal regime
        );

        let p_fused = post.prob_continuation_fused();

        // Strong trend agreement should push toward higher continuation
        assert!(
            p_fused > 0.55,
            "Strong trend should boost continuation above neutral: {}",
            p_fused
        );
    }

    #[test]
    fn test_cascade_regime_has_high_prior() {
        let mut post = ContinuationPosterior::default();

        // Set cascade regime
        post.update_signals(
            0.0,                  // No changepoint
            2.0,                  // Normal entropy
            0.5,                  // Neutral momentum
            0.0,                  // No trend
            0.0,                  // No trend confidence
            [0.0, 0.0, 0.0, 1.0], // Pure cascade regime
        );

        let p_fused = post.prob_continuation_fused();
        let regime_prior = post.regime_prior();

        assert!(
            regime_prior > 0.75,
            "Cascade regime should have high prior: {}",
            regime_prior
        );
        // Fused should be influenced by high regime prior
        assert!(
            p_fused > 0.55,
            "Cascade regime should boost fused probability: {}",
            p_fused
        );
    }

    #[test]
    fn test_fused_confidence_drops_on_changepoint() {
        let mut post = ContinuationPosterior::default();

        // Build up confidence with fills
        for _ in 0..20 {
            post.observe_fill(true, 1.0);
        }
        let conf_before = post.confidence_fused();

        // Simulate changepoint
        post.update_signals(
            0.8, // High changepoint
            0.3, // Low entropy
            0.5,
            0.0,
            0.0,
            [0.2, 0.5, 0.2, 0.1],
        );

        let conf_after = post.confidence_fused();

        assert!(
            conf_after < conf_before,
            "Confidence should drop on changepoint: before={}, after={}",
            conf_before,
            conf_after
        );
    }

    #[test]
    fn test_signal_summary() {
        let mut post = ContinuationPosterior::default();
        post.update_signals(0.3, 1.5, 0.7, 0.6, 0.5, [0.1, 0.4, 0.3, 0.2]);

        let summary = post.signal_summary();

        assert!(summary.p_fill_raw > 0.0 && summary.p_fill_raw < 1.0);
        assert!(summary.p_fused > 0.0 && summary.p_fused < 1.0);
        assert!(summary.changepoint_discount >= 0.0 && summary.changepoint_discount <= 1.0);
    }

    #[test]
    fn test_has_signal_updates() {
        let mut post = ContinuationPosterior::default();
        assert!(
            !post.has_signal_updates(),
            "Fresh posterior should not have updates"
        );

        post.update_signals(0.5, 2.0, 0.5, 0.0, 0.0, [0.2, 0.5, 0.2, 0.1]);
        assert!(post.has_signal_updates(), "Should detect signal updates");
    }
}
