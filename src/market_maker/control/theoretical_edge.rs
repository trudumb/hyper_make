//! Theoretical Edge Estimator for Illiquid Markets
//!
//! Uses market microstructure theory to compute expected value from book_imbalance
//! WITHOUT requiring price calibration. This provides a fallback for the IR-based
//! edge signal when price movement is insufficient for calibration.
//!
//! # Theory
//!
//! Book imbalance has a priori predictive validity:
//! - When bid_depth >> ask_depth, there's net demand
//! - Market clearing will move price toward the imbalance
//! - P(direction correct | imbalance) ≈ 0.50 + α × |book_imbalance|
//!
//! # Expected Value Formula
//!
//! ```text
//! E[value] = spread/2 + σ√τ × (P(correct) - 0.5) - σ√τ × P(adverse)
//! ```
//!
//! Where:
//! - spread/2: Expected spread capture from passive side
//! - σ√τ × (P(correct) - 0.5): Directional edge (positive when imbalance strong)
//! - σ√τ × P(adverse): Adverse selection cost
//!
//! # References
//!
//! - Easley & O'Hara PIN model: P(informed) typically 0.1-0.2
//! - Cont et al. (2014): Order book imbalance predicts direction

use tracing::debug;

/// Cross-asset signal for incorporating BTC momentum into edge calculation.
#[derive(Debug, Clone, Default)]
pub struct CrossAssetSignal {
    /// BTC short-term return (bps) over lookback window
    pub btc_return_bps: f64,
    /// BTC flow imbalance (if available)
    pub btc_flow_imbalance: f64,
    /// Estimated correlation with target asset (0.0-1.0)
    pub correlation: f64,
    /// Last update timestamp (ms)
    pub last_update_ms: u64,
    /// Whether signal is fresh (updated within threshold)
    pub is_fresh: bool,
}

impl CrossAssetSignal {
    /// Create a new cross-asset signal
    pub fn new(btc_return_bps: f64, btc_flow_imbalance: f64, correlation: f64) -> Self {
        Self {
            btc_return_bps,
            btc_flow_imbalance,
            correlation,
            last_update_ms: 0,
            is_fresh: false,
        }
    }

    /// Update the signal with fresh data
    pub fn update(&mut self, btc_return_bps: f64, btc_flow_imbalance: f64, now_ms: u64) {
        self.btc_return_bps = btc_return_bps;
        self.btc_flow_imbalance = btc_flow_imbalance;
        self.last_update_ms = now_ms;
        self.is_fresh = true;
    }

    /// Mark signal as stale if too old
    pub fn check_freshness(&mut self, now_ms: u64, max_age_ms: u64) {
        self.is_fresh = (now_ms - self.last_update_ms) < max_age_ms;
    }

    /// Get the directional boost from BTC signal
    /// Returns adjustment to P(correct) based on BTC momentum
    pub fn directional_boost(&self) -> f64 {
        if !self.is_fresh || self.correlation < 0.5 {
            return 0.0;
        }

        // BTC return in same direction as position = higher P(correct)
        // Scale: 10 bps BTC return with 0.8 correlation adds ~2% to P(correct)
        (self.btc_return_bps / 50.0).clamp(-0.10, 0.10) * self.correlation
    }
}

// ============================================================================
// Bayesian Alpha Tracker
// ============================================================================

/// Bayesian tracker for directional accuracy (alpha parameter).
///
/// Uses Beta(α, β) posterior to learn P(direction correct | fill).
/// Starts with informative prior Beta(2, 6) → mean 0.25, and updates from fills.
#[derive(Debug, Clone)]
pub struct BayesianAlphaTracker {
    /// Beta distribution α parameter (successes + prior)
    alpha_successes: f64,
    /// Beta distribution β parameter (failures + prior)
    beta_failures: f64,
    /// Total fills observed
    total_fills: u64,
    /// Correct direction predictions
    correct_directions: u64,
}

impl BayesianAlphaTracker {
    /// Create with informative prior Beta(2, 6) → mean 0.25
    pub fn new() -> Self {
        Self {
            alpha_successes: 2.0,  // Prior successes
            beta_failures: 6.0,    // Prior failures → mean = 2/(2+6) = 0.25
            total_fills: 0,
            correct_directions: 0,
        }
    }
    
    /// Update from fill outcome.
    ///
    /// # Arguments
    /// * `imbalance` - Book imbalance at time of fill (for weighting, future use)
    /// * `was_correct` - True if price moved in predicted direction
    pub fn update(&mut self, _imbalance: f64, was_correct: bool) {
        self.total_fills += 1;
        if was_correct {
            self.correct_directions += 1;
            self.alpha_successes += 1.0;
        } else {
            self.beta_failures += 1.0;
        }
    }
    
    /// Posterior mean of alpha (expected directional accuracy scalar).
    pub fn mean(&self) -> f64 {
        self.alpha_successes / (self.alpha_successes + self.beta_failures)
    }
    
    /// Posterior variance (uncertainty measure).
    pub fn variance(&self) -> f64 {
        let ab = self.alpha_successes + self.beta_failures;
        (self.alpha_successes * self.beta_failures) / (ab * ab * (ab + 1.0))
    }
    
    /// Uncertainty penalty for edge calculation.
    /// Subtracts from expected edge when posterior is uncertain.
    pub fn uncertainty_cost(&self) -> f64 {
        // Scale by sqrt(variance) with dampening factor
        self.variance().sqrt() * 0.5
    }
    
    /// Get total fills observed.
    pub fn total_fills(&self) -> u64 {
        self.total_fills
    }
    
    /// Get correct prediction count.
    pub fn correct_directions(&self) -> u64 {
        self.correct_directions
    }
    
    /// Get accuracy rate (empirical, ignoring prior).
    pub fn empirical_accuracy(&self) -> f64 {
        if self.total_fills == 0 {
            return 0.5;  // Neutral
        }
        self.correct_directions as f64 / self.total_fills as f64
    }
    
    /// Decay posterior toward prior (for changepoint handling).
    ///
    /// When a regime change is detected, we want to "forget" old data
    /// by shrinking the posterior toward the prior.
    ///
    /// # Arguments
    /// * `retention` - Fraction of posterior to keep (0.0 = full reset, 1.0 = no change)
    pub fn decay(&mut self, retention: f64) {
        let retention = retention.clamp(0.0, 1.0);
        // Prior is Beta(2, 6)
        const PRIOR_ALPHA: f64 = 2.0;
        const PRIOR_BETA: f64 = 6.0;
        
        // Shrink toward prior
        self.alpha_successes = PRIOR_ALPHA + (self.alpha_successes - PRIOR_ALPHA) * retention;
        self.beta_failures = PRIOR_BETA + (self.beta_failures - PRIOR_BETA) * retention;
        
        // Decay fill counts proportionally
        self.total_fills = (self.total_fills as f64 * retention) as u64;
        self.correct_directions = (self.correct_directions as f64 * retention) as u64;
    }
    
    /// Weighted update from fill outcome.
    ///
    /// Allows weighting by fill quality (e.g., low AS = high weight).
    ///
    /// # Arguments
    /// * `was_correct` - True if price moved in predicted direction
    /// * `weight` - Weight for this observation (0.0-2.0 typical)
    pub fn update_weighted(&mut self, was_correct: bool, weight: f64) {
        let weight = weight.clamp(0.1, 2.0);
        self.total_fills += 1;
        if was_correct {
            self.correct_directions += 1;
            self.alpha_successes += weight;
        } else {
            self.beta_failures += weight;
        }
    }
    
    /// Extract posterior as prior for cross-asset sharing.
    ///
    /// Creates a new tracker initialized with this tracker's posterior
    /// (shrunk toward prior by `shrinkage` factor).
    ///
    /// # Arguments
    /// * `shrinkage` - How much of the learned posterior to transfer (0.0-1.0)
    pub fn as_prior(&self, shrinkage: f64) -> Self {
        let shrinkage = shrinkage.clamp(0.0, 1.0);
        const PRIOR_ALPHA: f64 = 2.0;
        const PRIOR_BETA: f64 = 6.0;
        
        Self {
            alpha_successes: PRIOR_ALPHA + (self.alpha_successes - PRIOR_ALPHA) * shrinkage,
            beta_failures: PRIOR_BETA + (self.beta_failures - PRIOR_BETA) * shrinkage,
            total_fills: 0,  // New asset starts with 0 fills
            correct_directions: 0,
        }
    }
}

impl Default for BayesianAlphaTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Regime-Aware Alpha Tracker
// ============================================================================

/// Regime-aware wrapper for BayesianAlphaTracker.
///
/// Maintains separate alpha posteriors for each volatility regime:
/// - Regime 0: Low volatility (calm markets)
/// - Regime 1: Normal volatility
/// - Regime 2: High volatility (cascade/stress)
///
/// This prevents high-vol experiences from contaminating calm-regime beliefs.
#[derive(Debug, Clone)]
pub struct RegimeAwareAlphaTracker {
    /// Per-regime alpha trackers
    trackers: [BayesianAlphaTracker; 3],
    /// Current regime index (0=Low, 1=Normal, 2=High)
    current_regime: usize,
}

impl RegimeAwareAlphaTracker {
    /// Create with default priors for all regimes.
    pub fn new() -> Self {
        Self {
            trackers: [
                BayesianAlphaTracker::new(),
                BayesianAlphaTracker::new(),
                BayesianAlphaTracker::new(),
            ],
            current_regime: 1,  // Start in normal regime
        }
    }
    
    /// Set the current regime.
    pub fn set_regime(&mut self, regime: usize) {
        self.current_regime = regime.min(2);
    }
    
    /// Get current regime.
    pub fn current_regime(&self) -> usize {
        self.current_regime
    }
    
    /// Update from fill outcome (updates current regime's tracker).
    pub fn update(&mut self, imbalance: f64, was_correct: bool) {
        self.trackers[self.current_regime].update(imbalance, was_correct);
    }
    
    /// Weighted update (updates current regime's tracker).
    pub fn update_weighted(&mut self, was_correct: bool, weight: f64) {
        self.trackers[self.current_regime].update_weighted(was_correct, weight);
    }
    
    /// Get posterior mean for current regime.
    pub fn mean(&self) -> f64 {
        self.trackers[self.current_regime].mean()
    }
    
    /// Get posterior variance for current regime.
    pub fn variance(&self) -> f64 {
        self.trackers[self.current_regime].variance()
    }
    
    /// Get total fills across all regimes.
    pub fn total_fills(&self) -> u64 {
        self.trackers.iter().map(|t| t.total_fills()).sum()
    }
    
    /// Decay all trackers (for hard changepoint reset).
    pub fn decay_all(&mut self, retention: f64) {
        for tracker in &mut self.trackers {
            tracker.decay(retention);
        }
    }
    
    /// Decay only current regime (for soft regime-local reset).
    pub fn decay_current(&mut self, retention: f64) {
        self.trackers[self.current_regime].decay(retention);
    }
    
    /// Get tracker for a specific regime.
    pub fn tracker(&self, regime: usize) -> &BayesianAlphaTracker {
        &self.trackers[regime.min(2)]
    }
    
    /// Uncertainty penalty for edge calculation (current regime).
    pub fn uncertainty_cost(&self) -> f64 {
        self.trackers[self.current_regime].uncertainty_cost()
    }
    
    /// Get empirical accuracy rate (current regime, ignoring prior).
    pub fn empirical_accuracy(&self) -> f64 {
        self.trackers[self.current_regime].empirical_accuracy()
    }
}

impl Default for RegimeAwareAlphaTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Fill Rate Estimator
// ============================================================================

/// Estimator for fill arrival rate (λ) using exponential moving average.
///
/// Used to compute P(fill within τ) = 1 - exp(-λ*τ) for illiquid assets.
#[derive(Debug, Clone)]
pub struct FillRateEstimator {
    /// Exponential moving average of fills per hour
    lambda_fills_per_hour: f64,
    /// EMA decay factor (0.95 ≈ 20-fill half-life)
    decay: f64,
    /// Last fill timestamp (ms)
    last_fill_ms: u64,
    /// Total fills observed
    total_fills: u64,
}

impl FillRateEstimator {
    /// Create with conservative prior (5 fills/hour).
    pub fn new() -> Self {
        Self {
            lambda_fills_per_hour: 5.0,  // Conservative prior
            decay: 0.95,
            last_fill_ms: 0,
            total_fills: 0,
        }
    }
    
    /// Record a fill and update rate estimate.
    pub fn on_fill(&mut self, now_ms: u64) {
        self.total_fills += 1;
        
        if self.last_fill_ms > 0 {
            let hours_since = (now_ms.saturating_sub(self.last_fill_ms)) as f64 / 3_600_000.0;
            if hours_since > 0.0001 {  // Avoid division by tiny values
                let implied_rate = 1.0 / hours_since;
                // EMA update
                self.lambda_fills_per_hour = self.decay * self.lambda_fills_per_hour 
                    + (1.0 - self.decay) * implied_rate;
            }
        }
        
        self.last_fill_ms = now_ms;
    }
    
    /// P(fill within τ seconds) assuming Poisson process.
    pub fn p_fill(&self, tau_seconds: f64) -> f64 {
        let lambda_per_sec = self.lambda_fills_per_hour / 3600.0;
        1.0 - (-lambda_per_sec * tau_seconds).exp()
    }
    
    /// Get current rate estimate (fills per hour).
    pub fn lambda(&self) -> f64 {
        self.lambda_fills_per_hour
    }
    
    /// Get total fills observed.
    pub fn total_fills(&self) -> u64 {
        self.total_fills
    }
}

impl Default for FillRateEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Enhanced Edge Input
// ============================================================================

/// Enhanced input for edge calculation with depth and momentum signals.
///
/// Provides richer signal variance than book_imbalance alone, helping
/// with P(correct) discrimination and breaking calibration bootstrap cycles.
#[derive(Debug, Clone, Default)]
pub struct EnhancedEdgeInput {
    /// Base order book imbalance [-1, +1]
    pub book_imbalance: f64,
    /// Bid depth as fraction of total depth [0, 1]
    pub bid_depth_ratio: f64,
    /// Ask depth as fraction of total depth [0, 1]  
    pub ask_depth_ratio: f64,
    /// Short-term price momentum signal [-1, +1]
    pub short_momentum: f64,
    /// Spread in basis points
    pub spread_bps: f64,
    /// Volatility (fractional)
    pub sigma: f64,
    /// Expected holding time (seconds)
    pub tau_seconds: f64,
}

impl EnhancedEdgeInput {
    /// Create from basic parameters.
    pub fn new(book_imbalance: f64, spread_bps: f64, sigma: f64, tau_seconds: f64) -> Self {
        Self {
            book_imbalance,
            bid_depth_ratio: 0.5,
            ask_depth_ratio: 0.5,
            short_momentum: 0.0,
            spread_bps,
            sigma,
            tau_seconds,
        }
    }
    
    /// Set depth ratios.
    pub fn with_depth(mut self, bid_ratio: f64, ask_ratio: f64) -> Self {
        self.bid_depth_ratio = bid_ratio.clamp(0.0, 1.0);
        self.ask_depth_ratio = ask_ratio.clamp(0.0, 1.0);
        self
    }
    
    /// Set momentum signal.
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.short_momentum = momentum.clamp(-1.0, 1.0);
        self
    }
    
    /// Compute enhanced imbalance by blending signals.
    ///
    /// Formula: imbalance + 0.3 * depth_diff + 0.2 * momentum
    /// This widens the signal range from clustered values (0.3-0.5) to (0.1-0.9).
    pub fn enhanced_imbalance(&self) -> f64 {
        let depth_diff = self.bid_depth_ratio - self.ask_depth_ratio;
        let enhanced = self.book_imbalance
            + 0.3 * depth_diff
            + 0.2 * self.short_momentum;
        enhanced.clamp(-1.0, 1.0)
    }
}

// ============================================================================
// Bayesian Adverse Selection Tracker  
// ============================================================================

/// Bayesian tracker for adverse selection probability.
///
/// Uses Beta(α, β) posterior to learn P(adverse fill | fill).
/// Starts with informative prior Beta(3, 17) → mean 0.15 (from PIN model estimates).
///
/// An "adverse" fill is one where price moved against the position after fill.
#[derive(Debug, Clone)]
pub struct BayesianAdverseTracker {
    /// Beta α (adverse fills + prior)
    alpha: f64,
    /// Beta β (non-adverse fills + prior)
    beta: f64,
    /// Total fills observed
    total_fills: u64,
    /// Adverse fills observed
    adverse_fills: u64,
}

impl BayesianAdverseTracker {
    /// Create with prior Beta(3, 17) → mean 0.15.
    pub fn new() -> Self {
        Self {
            alpha: 3.0,   // Prior adverse
            beta: 17.0,   // Prior non-adverse → mean = 3/(3+17) = 0.15
            total_fills: 0,
            adverse_fills: 0,
        }
    }
    
    /// Update from fill outcome.
    ///
    /// # Arguments
    /// * `was_adverse` - True if price moved against position after fill
    pub fn update(&mut self, was_adverse: bool) {
        self.total_fills += 1;
        if was_adverse {
            self.adverse_fills += 1;
            self.alpha += 1.0;
        } else {
            self.beta += 1.0;
        }
    }
    
    /// Posterior mean of adverse selection probability.
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }
    
    /// Posterior variance (uncertainty measure).
    pub fn variance(&self) -> f64 {
        let ab = self.alpha + self.beta;
        (self.alpha * self.beta) / (ab * ab * (ab + 1.0))
    }
    
    /// Get total fills observed.
    pub fn total_fills(&self) -> u64 {
        self.total_fills
    }
    
    /// Get adverse fill count.
    pub fn adverse_fills(&self) -> u64 {
        self.adverse_fills
    }
    
    /// Empirical adverse rate (ignoring prior).
    pub fn empirical_rate(&self) -> f64 {
        if self.total_fills == 0 {
            return 0.15; // Return prior mean
        }
        self.adverse_fills as f64 / self.total_fills as f64
    }
    
    /// Decay posterior toward prior (for regime changes).
    pub fn decay(&mut self, retention: f64) {
        let retention = retention.clamp(0.0, 1.0);
        const PRIOR_ALPHA: f64 = 3.0;
        const PRIOR_BETA: f64 = 17.0;
        
        self.alpha = PRIOR_ALPHA + (self.alpha - PRIOR_ALPHA) * retention;
        self.beta = PRIOR_BETA + (self.beta - PRIOR_BETA) * retention;
        self.total_fills = (self.total_fills as f64 * retention) as u64;
        self.adverse_fills = (self.adverse_fills as f64 * retention) as u64;
    }
}

impl Default for BayesianAdverseTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Regime-Aware Bayesian Adverse Selection Tracker
// ============================================================================

/// Prior configurations for each regime.
/// Per-regime priors reflect structural expectations:
/// - Calm regime: lower adverse selection (E[p] = 0.10)
/// - Normal regime: moderate adverse selection (E[p] = 0.15)
/// - Volatile regime: higher adverse selection (E[p] = 0.25)
const REGIME_ADVERSE_PRIORS: [(f64, f64); 3] = [
    (3.0, 27.0),  // Calm: E[p] = 3/30 = 0.10
    (3.0, 17.0),  // Normal: E[p] = 3/20 = 0.15
    (5.0, 15.0),  // Volatile: E[p] = 5/20 = 0.25
];

/// Summary of regime-aware adverse selection state.
#[derive(Debug, Clone)]
pub struct AdverseSummary {
    /// Current regime index (0=Calm, 1=Normal, 2=Volatile).
    pub current_regime: usize,
    /// Posterior mean for each regime.
    pub regime_means: [f64; 3],
    /// Fills observed per regime.
    pub fills_per_regime: [u64; 3],
    /// Empirical adverse rate per regime (ignoring prior).
    pub adverse_rate_per_regime: [f64; 3],
    /// Posterior variance for current regime.
    pub current_variance: f64,
    /// 95% credible interval for current regime.
    pub credible_interval_95: (f64, f64),
}

/// Regime-aware Bayesian tracker for adverse selection probability.
///
/// Maintains separate Beta posteriors for each volatility regime:
/// - Regime 0: Calm (low volatility)
/// - Regime 1: Normal
/// - Regime 2: Volatile (high volatility, cascades)
///
/// This prevents high-volatility adverse fills from contaminating calm-regime
/// beliefs, allowing tighter spreads in calm markets while maintaining
/// appropriate conservatism in volatile conditions.
#[derive(Debug, Clone)]
pub struct RegimeAwareBayesianAdverse {
    /// Per-regime Beta posteriors: [(alpha, beta); 3]
    regimes: [(f64, f64); 3],
    /// Current regime index
    current_regime: usize,
    /// Fill counts per regime
    fills_per_regime: [u64; 3],
    /// Adverse fill counts per regime
    adverse_per_regime: [u64; 3],
}

impl RegimeAwareBayesianAdverse {
    /// Create with regime-specific priors.
    pub fn new() -> Self {
        Self {
            regimes: REGIME_ADVERSE_PRIORS,
            current_regime: 1, // Start in normal regime
            fills_per_regime: [0; 3],
            adverse_per_regime: [0; 3],
        }
    }

    /// Classify and update current regime based on market conditions.
    ///
    /// # Arguments
    /// * `vol_ratio` - Current volatility / baseline volatility
    /// * `spread_ratio` - Current spread / typical spread
    /// * `cascade_prob` - Probability of cascade from changepoint detector
    pub fn update_regime(&mut self, vol_ratio: f64, spread_ratio: f64, cascade_prob: f64) {
        self.current_regime = if cascade_prob > 0.5 || vol_ratio > 2.0 {
            2 // Volatile
        } else if vol_ratio < 0.7 && spread_ratio < 0.8 {
            0 // Calm
        } else {
            1 // Normal
        };
    }

    /// Set the current regime directly.
    pub fn set_regime(&mut self, regime: usize) {
        self.current_regime = regime.min(2);
    }

    /// Get current regime index.
    pub fn current_regime(&self) -> usize {
        self.current_regime
    }

    /// Update with fill outcome (standard update).
    ///
    /// # Arguments
    /// * `was_adverse` - True if price moved against position after fill
    pub fn update(&mut self, was_adverse: bool) {
        self.update_weighted(was_adverse, 0.0, 1.0);
    }

    /// Update with impact-weighted fill outcome.
    ///
    /// Uses impact weighting to emphasize painful adverse fills:
    /// - High urgency score → higher weight (quadratic)
    /// - Large adverse selection bps → higher weight (log scale)
    ///
    /// # Arguments
    /// * `was_adverse` - True if price moved against position after fill
    /// * `urgency_score` - Urgency score from L3 controller [0, 4+]
    /// * `realized_as_bps` - Realized adverse selection in basis points
    pub fn update_weighted(
        &mut self,
        was_adverse: bool,
        urgency_score: f64,
        realized_as_bps: f64,
    ) {
        // Impact weight: emphasize painful adverse fills
        let urgency_weight = urgency_score.powi(2) / 4.0; // Quadratic [0, ~4]
        let size_weight = (1.0 + realized_as_bps.abs() / 5.0).ln().max(0.0); // Log scale
        let w = 1.0 + urgency_weight + size_weight;

        let (alpha, beta) = &mut self.regimes[self.current_regime];

        if was_adverse {
            *alpha += w;
            self.adverse_per_regime[self.current_regime] += 1;
        } else {
            *beta += w;
        }
        self.fills_per_regime[self.current_regime] += 1;
    }

    /// Get posterior mean for current regime.
    pub fn mean(&self) -> f64 {
        let (alpha, beta) = self.regimes[self.current_regime];
        alpha / (alpha + beta)
    }

    /// Get posterior mean for a specific regime.
    pub fn mean_for_regime(&self, regime: usize) -> f64 {
        let regime = regime.min(2);
        let (alpha, beta) = self.regimes[regime];
        alpha / (alpha + beta)
    }

    /// Get posterior variance for current regime (uncertainty).
    pub fn variance(&self) -> f64 {
        let (alpha, beta) = self.regimes[self.current_regime];
        let n = alpha + beta;
        alpha * beta / (n * n * (n + 1.0))
    }

    /// Get 95% credible interval for current regime.
    ///
    /// Uses normal approximation for Beta distribution when α, β are large enough.
    pub fn credible_interval_95(&self) -> (f64, f64) {
        let mean = self.mean();
        let std = self.variance().sqrt();
        let lower = (mean - 1.96 * std).max(0.0);
        let upper = (mean + 1.96 * std).min(1.0);
        (lower, upper)
    }

    /// Get total fills across all regimes.
    pub fn total_fills(&self) -> u64 {
        self.fills_per_regime.iter().sum()
    }

    /// Get adverse fills for current regime.
    pub fn adverse_fills_current(&self) -> u64 {
        self.adverse_per_regime[self.current_regime]
    }

    /// Get empirical adverse rate for current regime (ignoring prior).
    pub fn empirical_rate(&self) -> f64 {
        let fills = self.fills_per_regime[self.current_regime];
        if fills == 0 {
            return self.mean(); // Return prior mean
        }
        self.adverse_per_regime[self.current_regime] as f64 / fills as f64
    }

    /// Decay all regime posteriors toward their priors.
    ///
    /// Call this when a hard changepoint is detected.
    ///
    /// # Arguments
    /// * `retention` - Fraction of posterior to keep [0, 1]
    pub fn decay_all(&mut self, retention: f64) {
        let retention = retention.clamp(0.0, 1.0);

        for (i, (alpha, beta)) in self.regimes.iter_mut().enumerate() {
            let (prior_alpha, prior_beta) = REGIME_ADVERSE_PRIORS[i];
            *alpha = prior_alpha + (*alpha - prior_alpha) * retention;
            *beta = prior_beta + (*beta - prior_beta) * retention;
        }

        for fills in &mut self.fills_per_regime {
            *fills = (*fills as f64 * retention) as u64;
        }
        for adverse in &mut self.adverse_per_regime {
            *adverse = (*adverse as f64 * retention) as u64;
        }
    }

    /// Decay only current regime (for soft regime-local reset).
    pub fn decay_current(&mut self, retention: f64) {
        let retention = retention.clamp(0.0, 1.0);
        let r = self.current_regime;

        let (prior_alpha, prior_beta) = REGIME_ADVERSE_PRIORS[r];
        let (alpha, beta) = &mut self.regimes[r];
        *alpha = prior_alpha + (*alpha - prior_alpha) * retention;
        *beta = prior_beta + (*beta - prior_beta) * retention;

        self.fills_per_regime[r] = (self.fills_per_regime[r] as f64 * retention) as u64;
        self.adverse_per_regime[r] = (self.adverse_per_regime[r] as f64 * retention) as u64;
    }

    // =========================================================================
    // Phase 7: Hawkes-Bayesian Fusion
    // =========================================================================

    /// Update regime considering Hawkes excitation state.
    ///
    /// Hawkes high excitation (high branching ratio, high intensity) is a leading
    /// indicator of regime transitions. When Hawkes signals excitation:
    /// - Prefer volatile regime classification
    /// - Faster transition to volatile regime
    /// - Slower transition back to calm regime
    ///
    /// # Arguments
    /// * `vol_ratio` - Current volatility / baseline volatility
    /// * `spread_ratio` - Current spread / typical spread
    /// * `cascade_prob` - Probability of cascade from changepoint detector
    /// * `hawkes_is_high_excitation` - Whether Hawkes predictor indicates high excitation
    /// * `hawkes_branching_ratio` - Current n = α/β from Hawkes calibration
    pub fn update_regime_with_hawkes(
        &mut self,
        vol_ratio: f64,
        spread_ratio: f64,
        cascade_prob: f64,
        hawkes_is_high_excitation: bool,
        hawkes_branching_ratio: f64,
    ) {
        // Hawkes excitation boost: when n > 0.5, add to cascade_prob
        let hawkes_cascade_boost = if hawkes_is_high_excitation {
            // Quadratic boost: n=0.7 → +0.12, n=0.9 → +0.32
            (hawkes_branching_ratio - 0.5).max(0.0).powi(2) * 2.0
        } else {
            0.0
        };

        // Effective cascade probability with Hawkes contribution
        let effective_cascade_prob = (cascade_prob + hawkes_cascade_boost).min(1.0);

        // Regime classification with Hawkes-aware thresholds
        self.current_regime = if effective_cascade_prob > 0.4 || vol_ratio > 1.8 || hawkes_is_high_excitation {
            2 // Volatile - enter earlier with Hawkes signal
        } else if vol_ratio < 0.6 && spread_ratio < 0.7 && hawkes_branching_ratio < 0.4 {
            0 // Calm - require low Hawkes excitation to be calm
        } else {
            1 // Normal
        };
    }

    /// Update with Hawkes-weighted fill outcome.
    ///
    /// During high Hawkes excitation, fills carry more information about
    /// adverse selection probability. This method increases update weight
    /// when excitation is high, allowing faster posterior adaptation
    /// during volatile periods.
    ///
    /// # Arguments
    /// * `was_adverse` - True if price moved against position after fill
    /// * `urgency_score` - Urgency score from L3 controller [0, 4+]
    /// * `realized_as_bps` - Realized adverse selection in basis points
    /// * `hawkes_excitation_penalty` - From HawkesExcitationPredictor [0.5, 1.0]
    /// * `hawkes_p_cluster` - P(cluster) from predictor [0, 1]
    pub fn update_weighted_with_hawkes(
        &mut self,
        was_adverse: bool,
        urgency_score: f64,
        realized_as_bps: f64,
        hawkes_excitation_penalty: f64,
        hawkes_p_cluster: f64,
    ) {
        // Base weight from urgency and size
        let urgency_weight = urgency_score.powi(2) / 4.0;
        let size_weight = (1.0 + realized_as_bps.abs() / 5.0).ln().max(0.0);
        let base_w = 1.0 + urgency_weight + size_weight;

        // Hawkes excitation boost:
        // Lower penalty (high excitation) → higher weight
        // Higher P(cluster) → higher weight for adverse fills
        let excitation_factor = 1.0 / hawkes_excitation_penalty.max(0.5); // [1.0, 2.0]
        
        // Cluster probability adds information especially for adverse fills
        let cluster_boost = if was_adverse {
            // Adverse during high cluster prob = highly informative
            1.0 + hawkes_p_cluster * 2.0 // [1.0, 3.0]
        } else {
            // Non-adverse during high cluster prob = also informative (market absorbing flow)
            1.0 + hawkes_p_cluster * 0.5 // [1.0, 1.5]
        };

        let w = base_w * excitation_factor * cluster_boost;

        let (alpha, beta) = &mut self.regimes[self.current_regime];

        if was_adverse {
            *alpha += w;
            self.adverse_per_regime[self.current_regime] += 1;
        } else {
            *beta += w;
        }
        self.fills_per_regime[self.current_regime] += 1;
    }

    /// Get effective adverse selection rate incorporating Hawkes state.
    ///
    /// When Hawkes excitation is high, use more conservative estimate
    /// (closer to credible interval upper bound).
    ///
    /// # Arguments
    /// * `hawkes_p_cluster` - P(cluster) from predictor [0, 1]
    /// * `hawkes_is_high_excitation` - Whether in high excitation state
    pub fn effective_adverse_with_hawkes(
        &self,
        hawkes_p_cluster: f64,
        hawkes_is_high_excitation: bool,
    ) -> f64 {
        let mean = self.mean();
        let (_, upper_ci) = self.credible_interval_95();

        // During high excitation, blend toward upper credible interval
        if hawkes_is_high_excitation {
            // Linear blend: p_cluster=0.5 → 75% upper, p_cluster=1.0 → 100% upper
            let blend_factor = (hawkes_p_cluster - 0.5).max(0.0) * 2.0; // [0, 1]
            let conservative_blend = 0.5 + blend_factor * 0.5; // [0.5, 1.0]
            mean * (1.0 - conservative_blend) + upper_ci * conservative_blend
        } else {
            mean
        }
    }

    /// Get diagnostic summary.
    pub fn summary(&self) -> AdverseSummary {
        let regime_means = [
            self.mean_for_regime(0),
            self.mean_for_regime(1),
            self.mean_for_regime(2),
        ];

        let adverse_rate_per_regime = [
            if self.fills_per_regime[0] > 0 {
                self.adverse_per_regime[0] as f64 / self.fills_per_regime[0] as f64
            } else {
                regime_means[0]
            },
            if self.fills_per_regime[1] > 0 {
                self.adverse_per_regime[1] as f64 / self.fills_per_regime[1] as f64
            } else {
                regime_means[1]
            },
            if self.fills_per_regime[2] > 0 {
                self.adverse_per_regime[2] as f64 / self.fills_per_regime[2] as f64
            } else {
                regime_means[2]
            },
        ];

        AdverseSummary {
            current_regime: self.current_regime,
            regime_means,
            fills_per_regime: self.fills_per_regime,
            adverse_rate_per_regime,
            current_variance: self.variance(),
            credible_interval_95: self.credible_interval_95(),
        }
    }

    /// Get regime posterior (alpha, beta) for a specific regime.
    pub fn regime_posterior(&self, regime: usize) -> (f64, f64) {
        self.regimes[regime.min(2)]
    }
}

impl Default for RegimeAwareBayesianAdverse {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for theoretical edge estimation.
#[derive(Debug, Clone)]
pub struct TheoreticalEdgeConfig {
    /// Directional accuracy coefficient: P(correct) = 0.5 + alpha * |imbalance|
    /// From market microstructure literature, ~0.25 is typical.
    pub alpha: f64,

    /// Adverse selection prior: P(informed trader | fill)
    /// From PIN models, typically 0.10-0.20 in crypto markets.
    pub adverse_prior: f64,

    /// Minimum expected edge (bps) to trigger quoting
    pub min_edge_bps: f64,

    /// Minimum book imbalance to consider (noise filter)
    pub min_imbalance: f64,

    /// BTC correlation threshold for cross-asset boost
    pub btc_correlation_threshold: f64,

    /// Initial assumed correlation with BTC (before learning)
    pub btc_correlation_prior: f64,

    /// Trading fees in basis points (maker + taker combined for round-trip estimate)
    /// HIP-3 DEX typically 5-10 bps, standard perps 1.5-3.5 bps
    pub fee_bps: f64,
}

impl Default for TheoreticalEdgeConfig {
    fn default() -> Self {
        Self {
            alpha: 0.25,           // From order book imbalance studies
            adverse_prior: 0.25,   // BTC-calibrated: 20-30% informed flow typical on Hyperliquid
            min_edge_bps: 1.0,     // Minimum edge AFTER fees
            min_imbalance: 0.10,   // Ignore very weak imbalances
            btc_correlation_threshold: 0.5, // Only use BTC signal if correlated
            btc_correlation_prior: 0.7,     // Assume HYPE correlates with BTC
            fee_bps: 1.5,          // Standard maker fee (matches main config)
        }
    }
}

/// Theoretical edge estimation result.
#[derive(Debug, Clone)]
pub struct TheoreticalEdgeResult {
    /// Expected edge in basis points
    pub expected_edge_bps: f64,
    /// Directional edge component (from imbalance)
    pub directional_edge_bps: f64,
    /// Spread capture component
    pub spread_edge_bps: f64,
    /// Adverse selection cost
    pub adverse_cost_bps: f64,
    /// Trading fees (deducted from edge)
    pub fee_cost_bps: f64,
    /// Cross-asset boost (from BTC)
    pub btc_boost_bps: f64,
    /// P(direction correct) used in calculation
    pub p_correct: f64,
    /// Whether edge exceeds minimum threshold
    pub should_quote: bool,
    /// Recommended direction (1=long, -1=short, 0=neutral)
    pub direction: i8,
}

/// Theoretical edge estimator using market microstructure priors.
///
/// Provides a fallback for IR-based edge detection when price movement
/// is insufficient for calibration. Uses book_imbalance as the primary
/// signal with optional BTC cross-asset enhancement.
///
/// Now includes Bayesian learning for adaptive alpha and fill rate estimation.
#[derive(Debug)]
pub struct TheoreticalEdgeEstimator {
    config: TheoreticalEdgeConfig,
    cross_asset: CrossAssetSignal,
    /// Bayesian tracker for directional accuracy (regime-aware)
    alpha_tracker: RegimeAwareAlphaTracker,
    /// Bayesian tracker for adverse selection probability (regime-aware)
    adverse_tracker: RegimeAwareBayesianAdverse,
    /// Fill rate estimator for P(fill) calculation
    fill_rate: FillRateEstimator,
    /// Number of edge calculations performed
    calculations: u64,
    /// Number of quotes triggered
    quotes_triggered: u64,
}

impl TheoreticalEdgeEstimator {
    /// Create a new theoretical edge estimator with default config
    pub fn new() -> Self {
        Self::with_config(TheoreticalEdgeConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: TheoreticalEdgeConfig) -> Self {
        Self {
            config,
            cross_asset: CrossAssetSignal::default(),
            alpha_tracker: RegimeAwareAlphaTracker::new(),
            adverse_tracker: RegimeAwareBayesianAdverse::new(),
            fill_rate: FillRateEstimator::new(),
            calculations: 0,
            quotes_triggered: 0,
        }
    }

    /// Update BTC cross-asset signal
    pub fn update_btc_signal(&mut self, btc_return_bps: f64, btc_flow_imbalance: f64, now_ms: u64) {
        self.cross_asset.update(btc_return_bps, btc_flow_imbalance, now_ms);
    }

    /// Set BTC correlation (can be learned over time)
    pub fn set_btc_correlation(&mut self, correlation: f64) {
        self.cross_asset.correlation = correlation.clamp(0.0, 1.0);
    }

    /// Calculate expected edge from book imbalance and market conditions.
    ///
    /// # Arguments
    /// * `book_imbalance` - Book imbalance signal [-1, 1]
    /// * `spread_bps` - Current bid-ask spread in basis points
    /// * `sigma` - Volatility (fractional, e.g., 0.001 = 0.1%)
    /// * `tau_seconds` - Expected holding time in seconds
    ///
    /// # Returns
    /// `TheoreticalEdgeResult` with expected edge and recommendation
    pub fn calculate_edge(
        &mut self,
        book_imbalance: f64,
        spread_bps: f64,
        sigma: f64,
        tau_seconds: f64,
    ) -> TheoreticalEdgeResult {
        self.calculations += 1;

        // Noise filter: require minimum imbalance
        let abs_imbalance = book_imbalance.abs();
        if abs_imbalance < self.config.min_imbalance {
            return TheoreticalEdgeResult {
                expected_edge_bps: 0.0,
                directional_edge_bps: 0.0,
                spread_edge_bps: spread_bps / 2.0,
                adverse_cost_bps: 0.0,
                fee_cost_bps: self.config.fee_bps,
                btc_boost_bps: 0.0,
                p_correct: 0.5,
                should_quote: false,
                direction: 0,
            };
        }

        // Use Bayesian posterior alpha instead of static config
        // This learns from fills: starts at prior mean 0.25, updates on each outcome
        let posterior_alpha = self.alpha_tracker.mean();
        
        // P(direction correct | imbalance) = 0.5 + alpha * |imbalance|
        let base_p_correct = 0.5 + posterior_alpha * abs_imbalance;

        // Cross-asset boost from BTC signal
        let btc_boost = self.cross_asset.directional_boost();
        
        // If BTC signal is aligned with imbalance, boost P(correct)
        // If opposed, reduce P(correct)
        let btc_aligned = book_imbalance.signum() == self.cross_asset.btc_return_bps.signum();
        let btc_adjustment = if btc_aligned { btc_boost } else { -btc_boost.abs() * 0.5 };
        
        let p_correct = (base_p_correct + btc_adjustment).clamp(0.5, 0.85);

        // Expected price move magnitude (σ√τ in bps)
        let expected_move_bps = sigma * tau_seconds.sqrt() * 10_000.0;

        // Components of expected edge:
        // 1. Spread capture (passive side advantage)
        let spread_edge_bps = spread_bps / 2.0;

        // 2. Directional edge = expected_move × (P(correct) - 0.5)
        //    Subtract uncertainty cost when posterior is uncertain
        let uncertainty_cost_bps = self.alpha_tracker.uncertainty_cost() * expected_move_bps;
        let directional_edge_bps = expected_move_bps * (p_correct - 0.5) - uncertainty_cost_bps;

        // 3. BTC boost contribution (separate tracking for diagnostics)
        let btc_boost_bps = expected_move_bps * btc_adjustment;

        // 4. Adverse selection cost = expected_move × P(informed)
        let adverse_cost_bps = expected_move_bps * self.config.adverse_prior;

        // 5. Trading fees (explicit deduction)
        let fee_cost_bps = self.config.fee_bps;

        // 6. P(fill) scaling for illiquidity
        // On illiquid assets, low fill probability reduces expected edge
        let p_fill = self.fill_rate.p_fill(tau_seconds);
        
        // Raw edge before fill probability scaling
        let raw_edge_bps = spread_edge_bps + directional_edge_bps - adverse_cost_bps - fee_cost_bps;
        
        // Scale edge by fill probability: E[edge] = P(fill) × edge_if_filled
        // Use blend: at p_fill=1.0 use full edge, at p_fill→0 use dampened edge
        // Floor of 0.3 prevents complete zeroing on illiquids while properly
        // penalizing uncertain fill probability (previously 0.5 was too generous)
        let fill_dampening = 0.3 + 0.7 * p_fill;  // Range: [0.3, 1.0]
        let expected_edge_bps = raw_edge_bps * fill_dampening;

        // Should we quote?
        let should_quote = expected_edge_bps >= self.config.min_edge_bps;

        // Direction based on imbalance sign
        // Positive imbalance = more bids = expect price to rise = be long (buy)
        let direction = if should_quote {
            if book_imbalance > 0.0 { 1 } else { -1 }
        } else {
            0
        };

        if should_quote {
            self.quotes_triggered += 1;
        }

        // Periodic logging
        if self.calculations.is_multiple_of(100) {
            debug!(
                calculations = self.calculations,
                quotes_triggered = self.quotes_triggered,
                hit_rate = %format!("{:.1}%", self.quotes_triggered as f64 / self.calculations as f64 * 100.0),
                "Theoretical edge estimator stats"
            );
        }

        TheoreticalEdgeResult {
            expected_edge_bps,
            directional_edge_bps,
            spread_edge_bps,
            adverse_cost_bps,
            fee_cost_bps,
            btc_boost_bps,
            p_correct,
            should_quote,
            direction,
        }
    }

    /// Calculate expected edge using enhanced inputs with depth and momentum blending.
    ///
    /// This method provides better signal variance than the basic `calculate_edge` by:
    /// 1. Blending book_imbalance with depth ratios and momentum
    /// 2. Using Bayesian-learned adverse_prior instead of static config value
    ///
    /// # Arguments
    /// * `input` - Enhanced edge input with depth/momentum signals
    ///
    /// # Returns
    /// `TheoreticalEdgeResult` with edge estimate using blended signals
    pub fn calculate_edge_enhanced(&mut self, input: &EnhancedEdgeInput) -> TheoreticalEdgeResult {
        self.calculations += 1;

        // Compute enhanced imbalance by blending signals
        let enhanced_imbalance = input.enhanced_imbalance();
        let abs_imbalance = enhanced_imbalance.abs();

        // Noise filter: require minimum imbalance
        if abs_imbalance < self.config.min_imbalance {
            return TheoreticalEdgeResult {
                expected_edge_bps: 0.0,
                directional_edge_bps: 0.0,
                spread_edge_bps: input.spread_bps / 2.0,
                adverse_cost_bps: 0.0,
                fee_cost_bps: self.config.fee_bps,
                btc_boost_bps: 0.0,
                p_correct: 0.5,
                should_quote: false,
                direction: 0,
            };
        }

        // Use Bayesian posterior alpha for directional accuracy
        let posterior_alpha = self.alpha_tracker.mean();
        
        // P(direction correct | imbalance) = 0.5 + alpha * |imbalance|
        let base_p_correct = 0.5 + posterior_alpha * abs_imbalance;

        // Cross-asset boost from BTC signal
        let btc_boost = self.cross_asset.directional_boost();
        let btc_aligned = enhanced_imbalance.signum() == self.cross_asset.btc_return_bps.signum();
        let btc_adjustment = if btc_aligned { btc_boost } else { -btc_boost.abs() * 0.5 };
        
        let p_correct = (base_p_correct + btc_adjustment).clamp(0.5, 0.85);

        // Expected price move magnitude (σ√τ in bps)
        let expected_move_bps = input.sigma * input.tau_seconds.sqrt() * 10_000.0;

        // Components of expected edge:
        // 1. Spread capture
        let spread_edge_bps = input.spread_bps / 2.0;

        // 2. Directional edge with uncertainty cost
        let uncertainty_cost_bps = self.alpha_tracker.uncertainty_cost() * expected_move_bps;
        let directional_edge_bps = expected_move_bps * (p_correct - 0.5) - uncertainty_cost_bps;

        // 3. BTC boost contribution
        let btc_boost_bps = expected_move_bps * btc_adjustment;

        // 4. Adverse selection cost using REGIME-AWARE BAYESIAN POSTERIOR
        // Use posterior mean, but be conservative when uncertainty is high
        let posterior_adverse = self.adverse_tracker.mean();
        let adverse_uncertainty = self.adverse_tracker.variance().sqrt();
        let (_, upper_ci) = self.adverse_tracker.credible_interval_95();

        // Conservative adjustment: use upper bound of credible interval in high-uncertainty situations
        let effective_adverse = if adverse_uncertainty > 0.05 {
            // High uncertainty: use 75th percentile (mean + 0.67 std)
            (posterior_adverse + 0.67 * adverse_uncertainty).min(upper_ci)
        } else {
            posterior_adverse
        };

        let adverse_cost_bps = expected_move_bps * effective_adverse;

        // 5. Trading fees
        let fee_cost_bps = self.config.fee_bps;

        // 6. P(fill) scaling
        let p_fill = self.fill_rate.p_fill(input.tau_seconds);
        
        // Raw edge
        let raw_edge_bps = spread_edge_bps + directional_edge_bps - adverse_cost_bps - fee_cost_bps;
        
        // Scale by fill probability
        let fill_dampening = 0.5 + 0.5 * p_fill;
        let expected_edge_bps = raw_edge_bps * fill_dampening;

        // Should we quote?
        let should_quote = expected_edge_bps >= self.config.min_edge_bps;

        // Direction based on enhanced imbalance sign
        let direction = if should_quote {
            if enhanced_imbalance > 0.0 { 1 } else { -1 }
        } else {
            0
        };

        if should_quote {
            self.quotes_triggered += 1;
        }

        // Periodic logging with Bayesian diagnostics
        if self.calculations.is_multiple_of(100) {
            debug!(
                calculations = self.calculations,
                quotes_triggered = self.quotes_triggered,
                posterior_adverse = %format!("{:.3}", posterior_adverse),
                posterior_alpha = %format!("{:.3}", posterior_alpha),
                enhanced_imbalance = %format!("{:.3}", enhanced_imbalance),
                "Enhanced edge estimator stats"
            );
        }

        TheoreticalEdgeResult {
            expected_edge_bps,
            directional_edge_bps,
            spread_edge_bps,
            adverse_cost_bps,
            fee_cost_bps,
            btc_boost_bps,
            p_correct,
            should_quote,
            direction,
        }
    }

    /// Quick check if edge signal is active (for warmup status)
    pub fn is_active(&self) -> bool {
        true // Always active - uses priors, no warmup needed
    }

    /// Get configuration
    pub fn config(&self) -> &TheoreticalEdgeConfig {
        &self.config
    }

    /// Get calculation stats
    pub fn stats(&self) -> (u64, u64) {
        (self.calculations, self.quotes_triggered)
    }
    
    // ==================== Bayesian Learning Methods ====================
    
    /// Update from fill outcome for Bayesian learning.
    ///
    /// Call this after each fill with:
    /// - `imbalance`: Book imbalance at time of quote placement  
    /// - `predicted_direction`: Expected direction (1=up, -1=down)
    /// - `actual_direction`: Actual price movement direction (1=up, -1=down)
    /// - `now_ms`: Current timestamp
    ///
    /// This updates the posterior alpha and fill rate estimates.
    pub fn update_from_fill(
        &mut self,
        imbalance: f64,
        predicted_direction: i8,
        actual_direction: i8,
        now_ms: u64,
    ) {
        // Update fill rate estimator
        self.fill_rate.on_fill(now_ms);
        
        // Update Bayesian alpha tracker
        let was_correct = predicted_direction == actual_direction;
        self.alpha_tracker.update(imbalance, was_correct);
        
        debug!(
            was_correct = was_correct,
            posterior_alpha = %format!("{:.3}", self.alpha_tracker.mean()),
            uncertainty = %format!("{:.4}", self.alpha_tracker.uncertainty_cost()),
            total_fills = self.alpha_tracker.total_fills(),
            fill_rate_per_hour = %format!("{:.1}", self.fill_rate.lambda()),
            "Bayesian alpha updated from fill"
        );
    }
    
    /// Update from fill with price change (convenience method).
    ///
    /// - `imbalance`: Book imbalance at quote time
    /// - `price_change_bps`: Realized price change in bps (positive = up)
    /// - `now_ms`: Timestamp
    pub fn update_from_fill_with_price(
        &mut self,
        imbalance: f64,
        price_change_bps: f64,
        now_ms: u64,
    ) {
        // Determine directions from signs
        let predicted_direction = if imbalance > 0.0 { 1i8 } else { -1 };
        let actual_direction = if price_change_bps > 0.0 { 1i8 } else if price_change_bps < 0.0 { -1 } else { 0 };
        
        // Only update if there was meaningful price movement
        if actual_direction != 0 {
            self.update_from_fill(imbalance, predicted_direction, actual_direction, now_ms);
        } else {
            // Still record fill for rate estimation, but don't update alpha
            self.fill_rate.on_fill(now_ms);
        }
    }
    
    /// Update adverse selection tracker from fill outcome.
    ///
    /// Call this when determining if a fill was adverse (price moved against position).
    ///
    /// # Arguments
    /// * `was_adverse` - True if price moved against position after fill
    pub fn update_adverse(&mut self, was_adverse: bool) {
        self.adverse_tracker.update(was_adverse);

        let summary = self.adverse_tracker.summary();
        debug!(
            was_adverse = was_adverse,
            posterior_adverse = %format!("{:.3}", self.adverse_tracker.mean()),
            total_fills = self.adverse_tracker.total_fills(),
            regime = summary.current_regime,
            regime_means = ?summary.regime_means,
            "Adverse selection tracker updated (regime-aware)"
        );
    }

    /// Update adverse selection tracker with impact weighting (regime-aware).
    ///
    /// Uses impact weighting to emphasize painful fills:
    /// - High urgency score → higher weight
    /// - Large realized adverse selection → higher weight
    ///
    /// # Arguments
    /// * `was_adverse` - True if price moved against position after fill
    /// * `urgency_score` - L3 controller urgency score [0, 4+]
    /// * `realized_as_bps` - Realized adverse selection in basis points
    pub fn update_adverse_weighted(&mut self, was_adverse: bool, urgency_score: f64, realized_as_bps: f64) {
        self.adverse_tracker.update_weighted(was_adverse, urgency_score, realized_as_bps);

        let summary = self.adverse_tracker.summary();
        debug!(
            was_adverse = was_adverse,
            urgency_score = %format!("{:.2}", urgency_score),
            realized_as_bps = %format!("{:.2}", realized_as_bps),
            posterior_adverse = %format!("{:.3}", self.adverse_tracker.mean()),
            regime = summary.current_regime,
            credible_interval = ?summary.credible_interval_95,
            "Adverse selection tracker updated (weighted, regime-aware)"
        );
    }
    
    /// Update both alpha and adverse trackers from fill with full context.
    ///
    /// # Arguments
    /// * `imbalance` - Book imbalance at quote time
    /// * `fill_side` - Side of fill (1=bought, -1=sold)
    /// * `price_change_bps` - Price change since fill (positive = up)
    /// * `now_ms` - Timestamp
    pub fn update_from_fill_complete(
        &mut self,
        imbalance: f64,
        fill_side: i8,
        price_change_bps: f64,
        now_ms: u64,
    ) {
        // Update alpha tracker
        let predicted_direction = if imbalance > 0.0 { 1i8 } else { -1 };
        let actual_direction = if price_change_bps > 0.0 { 1i8 } else if price_change_bps < 0.0 { -1 } else { 0 };
        
        if actual_direction != 0 {
            self.update_from_fill(imbalance, predicted_direction, actual_direction, now_ms);
            
            // Update adverse tracker
            // Adverse = price moved against our position
            // If we bought (fill_side=1) and price fell (actual_direction=-1) → adverse
            // If we sold (fill_side=-1) and price rose (actual_direction=1) → adverse
            let was_adverse = fill_side != actual_direction;
            self.adverse_tracker.update(was_adverse);
        } else {
            self.fill_rate.on_fill(now_ms);
        }
    }
    
    /// Get Bayesian adverse selection probability (posterior mean).
    pub fn bayesian_adverse(&self) -> f64 {
        self.adverse_tracker.mean()
    }
    
    /// Get adverse tracker stats (total fills, adverse fills for current regime).
    pub fn adverse_stats(&self) -> (u64, u64) {
        (self.adverse_tracker.total_fills(), self.adverse_tracker.adverse_fills_current())
    }

    /// Get full adverse selection summary for diagnostics.
    pub fn adverse_summary(&self) -> AdverseSummary {
        self.adverse_tracker.summary()
    }

    /// Get adverse selection credible interval (95%) for current regime.
    pub fn adverse_credible_interval(&self) -> (f64, f64) {
        self.adverse_tracker.credible_interval_95()
    }

    /// Update adverse tracker regime based on market conditions.
    ///
    /// # Arguments
    /// * `vol_ratio` - Current volatility / baseline volatility
    /// * `spread_ratio` - Current spread / typical spread
    /// * `cascade_prob` - Probability of cascade from changepoint detector
    pub fn update_adverse_regime(&mut self, vol_ratio: f64, spread_ratio: f64, cascade_prob: f64) {
        self.adverse_tracker.update_regime(vol_ratio, spread_ratio, cascade_prob);
    }

    /// Decay adverse posteriors (for changepoint handling).
    ///
    /// # Arguments
    /// * `retention` - Fraction of posterior to keep (0.0-1.0)
    /// * `all_regimes` - If true, decay all regimes; if false, only current
    pub fn decay_adverse(&mut self, retention: f64, all_regimes: bool) {
        if all_regimes {
            self.adverse_tracker.decay_all(retention);
        } else {
            self.adverse_tracker.decay_current(retention);
        }
    }
    
    /// Get current Bayesian alpha (posterior mean).
    pub fn bayesian_alpha(&self) -> f64 {
        self.alpha_tracker.mean()
    }
    
    /// Get alpha uncertainty cost (for logging/diagnostics).
    pub fn alpha_uncertainty(&self) -> f64 {
        self.alpha_tracker.uncertainty_cost()
    }
    
    /// Get empirical accuracy rate (ignoring prior).
    pub fn empirical_accuracy(&self) -> f64 {
        self.alpha_tracker.empirical_accuracy()
    }
    
    /// Get fill rate estimate (fills per hour).
    pub fn fill_rate_per_hour(&self) -> f64 {
        self.fill_rate.lambda()
    }
    
    /// Get P(fill within tau seconds).
    pub fn p_fill(&self, tau_seconds: f64) -> f64 {
        self.fill_rate.p_fill(tau_seconds)
    }
    
    /// Get total fills observed by Bayesian tracker.
    pub fn bayesian_fills(&self) -> u64 {
        self.alpha_tracker.total_fills()
    }
    
    /// Get Bayesian stats summary for logging.
    pub fn bayesian_summary(&self) -> BayesianSummary {
        BayesianSummary {
            alpha: self.alpha_tracker.mean(),
            uncertainty: self.alpha_tracker.variance().sqrt(),
            total_fills: self.alpha_tracker.total_fills(),
            correct_fills: self.alpha_tracker.tracker(self.alpha_tracker.current_regime()).correct_directions(),
            empirical_accuracy: self.alpha_tracker.tracker(self.alpha_tracker.current_regime()).empirical_accuracy(),
            fill_rate_per_hour: self.fill_rate.lambda(),
            current_regime: self.alpha_tracker.current_regime(),
        }
    }
    
    // ==================== Regime & Changepoint Methods ====================
    
    /// Set the current volatility regime for regime-aware learning.
    ///
    /// Call this when regime changes are detected (from BeliefState.most_likely_regime()
    /// or ChangepointDetector signals).
    ///
    /// # Arguments
    /// * `regime` - Regime index (0=Low vol, 1=Normal, 2=High vol)
    pub fn set_regime(&mut self, regime: usize) {
        self.alpha_tracker.set_regime(regime);
    }
    
    /// Get current regime.
    pub fn current_regime(&self) -> usize {
        self.alpha_tracker.current_regime()
    }
    
    /// Decay alpha posteriors toward prior (for changepoint handling).
    ///
    /// Call this when a changepoint is detected to "forget" old regime data.
    /// The `retention` factor controls how much to keep:
    /// - 0.0 = Full reset to prior
    /// - 0.5 = Keep half of learned information
    /// - 1.0 = No change
    ///
    /// # Arguments
    /// * `retention` - Fraction of posterior to keep (0.0-1.0)
    /// * `all_regimes` - If true, decay all regimes; if false, only current
    pub fn decay_alpha(&mut self, retention: f64, all_regimes: bool) {
        if all_regimes {
            self.alpha_tracker.decay_all(retention);
        } else {
            self.alpha_tracker.decay_current(retention);
        }
    }
    
    /// Update from fill with quality weighting.
    ///
    /// Weights the Bayesian update based on fill quality metrics:
    /// - Low adverse selection = higher weight (more informative)
    /// - Larger fill size = higher weight (stronger signal)
    ///
    /// # Arguments
    /// * `imbalance` - Book imbalance at quote time
    /// * `predicted_direction` - Expected direction (1=up, -1=down)  
    /// * `actual_direction` - Actual price movement (1=up, -1=down)
    /// * `now_ms` - Timestamp
    /// * `realized_as_bps` - Realized adverse selection in basis points
    /// * `fill_size` - Size of the fill (for weighting)
    pub fn update_from_fill_weighted(
        &mut self,
        _imbalance: f64,
        predicted_direction: i8,
        actual_direction: i8,
        now_ms: u64,
        realized_as_bps: f64,
        fill_size: f64,
    ) {
        // Update fill rate estimator
        self.fill_rate.on_fill(now_ms);
        
        // Calculate quality-based weight
        // High AS = less informative (toxic flow), weight down
        let as_weight = 1.0 / (1.0 + realized_as_bps.abs() / 5.0);
        
        // Larger fills = more informative, weight up (with cap)
        let size_weight = (fill_size / 0.01).sqrt().clamp(0.5, 2.0);
        
        let weight = as_weight * size_weight;
        
        // Weighted update to alpha tracker
        let was_correct = predicted_direction == actual_direction;
        self.alpha_tracker.update_weighted(was_correct, weight);
        
        debug!(
            was_correct = was_correct,
            weight = %format!("{:.2}", weight),
            as_weight = %format!("{:.2}", as_weight),
            size_weight = %format!("{:.2}", size_weight),
            posterior_alpha = %format!("{:.3}", self.alpha_tracker.mean()),
            regime = self.alpha_tracker.current_regime(),
            "Weighted Bayesian alpha update"
        );
    }
    
    /// Get alpha tracker (for cross-asset prior sharing).
    pub fn alpha_tracker(&self) -> &RegimeAwareAlphaTracker {
        &self.alpha_tracker
    }
}

/// Summary of Bayesian learning state for logging.
#[derive(Debug, Clone)]
pub struct BayesianSummary {
    /// Posterior alpha (directional accuracy scalar)
    pub alpha: f64,
    /// Uncertainty (std dev of posterior)
    pub uncertainty: f64,
    /// Total fills observed
    pub total_fills: u64,
    /// Correct direction predictions
    pub correct_fills: u64,
    /// Empirical accuracy (ignoring prior)
    pub empirical_accuracy: f64,
    /// Estimated fill rate (per hour)
    pub fill_rate_per_hour: f64,
    /// Current volatility regime (0=Low, 1=Normal, 2=High)
    pub current_regime: usize,
}

impl Default for TheoreticalEdgeEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_imbalance_no_edge() {
        let mut estimator = TheoreticalEdgeEstimator::new();
        let result = estimator.calculate_edge(0.0, 10.0, 0.001, 1.0);
        
        // Zero imbalance = no directional edge (below min_imbalance)
        assert!(!result.should_quote);
        assert_eq!(result.direction, 0);
    }

    #[test]
    fn test_weak_imbalance_filtered() {
        let mut estimator = TheoreticalEdgeEstimator::new();
        // 0.05 imbalance is below min_imbalance (0.10)
        let result = estimator.calculate_edge(0.05, 10.0, 0.001, 1.0);
        
        assert!(!result.should_quote);
        assert_eq!(result.p_correct, 0.5);
    }

    #[test]
    fn test_strong_imbalance_positive_edge() {
        let mut estimator = TheoreticalEdgeEstimator::new();
        // Strong imbalance (0.4) with wide spread (20 bps) and normal vol
        let result = estimator.calculate_edge(0.40, 20.0, 0.001, 1.0);
        
        // P(correct) = 0.5 + 0.25 * 0.4 = 0.60
        assert!((result.p_correct - 0.60).abs() < 0.01);
        
        // Expected edge should be positive (spread capture + directional - adverse)
        assert!(result.expected_edge_bps > 0.0);
        assert!(result.should_quote);
        assert_eq!(result.direction, 1); // Long because positive imbalance
    }

    #[test]
    fn test_negative_imbalance_short_direction() {
        let mut estimator = TheoreticalEdgeEstimator::new();
        let result = estimator.calculate_edge(-0.40, 20.0, 0.001, 1.0);
        
        assert!(result.should_quote);
        assert_eq!(result.direction, -1); // Short because negative imbalance
    }

    #[test]
    fn test_high_volatility_increases_edge() {
        let mut estimator = TheoreticalEdgeEstimator::new();
        
        // Low vol
        let result_low_vol = estimator.calculate_edge(0.40, 10.0, 0.0005, 1.0);
        
        // High vol
        let result_high_vol = estimator.calculate_edge(0.40, 10.0, 0.002, 1.0);
        
        // Higher vol = higher directional edge (but also higher adverse cost)
        // Net effect depends on P(correct) - P(adverse)
        // With P(correct)=0.60 and adverse_prior=0.15, directional wins
        assert!(result_high_vol.directional_edge_bps > result_low_vol.directional_edge_bps);
    }

    #[test]
    fn test_tight_spread_may_not_quote() {
        let mut estimator = TheoreticalEdgeEstimator::new();
        // Very tight spread (2 bps), moderate imbalance
        let result = estimator.calculate_edge(0.20, 2.0, 0.001, 1.0);
        
        // With only 1 bps spread capture and 5% directional edge,
        // may not clear min_edge_bps threshold
        // (Actually depends on exact math - checking behavior)
        println!("Expected edge: {:.2} bps", result.expected_edge_bps);
    }

    #[test]
    fn test_btc_signal_boost() {
        let mut estimator = TheoreticalEdgeEstimator::new();
        estimator.set_btc_correlation(0.8);
        
        // Update with positive BTC return
        estimator.update_btc_signal(20.0, 0.3, 1000);
        estimator.cross_asset.is_fresh = true;
        
        // Positive imbalance aligned with positive BTC return
        let result_aligned = estimator.calculate_edge(0.30, 15.0, 0.001, 1.0);
        
        // Reset for opposed case
        estimator.update_btc_signal(-20.0, -0.3, 1000);
        estimator.cross_asset.is_fresh = true;
        
        // Positive imbalance opposed to negative BTC return
        let result_opposed = estimator.calculate_edge(0.30, 15.0, 0.001, 1.0);
        
        // Aligned should have higher P(correct)
        assert!(result_aligned.p_correct > result_opposed.p_correct);
    }

    // ==================== Bayesian Alpha Tracker Tests ====================

    #[test]
    fn test_bayesian_alpha_prior() {
        let tracker = BayesianAlphaTracker::new();
        
        // Prior mean should be 0.25 (Beta(2,6))
        assert!((tracker.mean() - 0.25).abs() < 0.01);
        assert_eq!(tracker.total_fills(), 0);
    }

    #[test]
    fn test_bayesian_alpha_updates_on_correct() {
        let mut tracker = BayesianAlphaTracker::new();
        
        // 8 correct predictions
        for _ in 0..8 {
            tracker.update(0.5, true);
        }
        
        // Posterior mean should increase: (2+8)/(2+6+8) = 10/16 = 0.625
        assert!(tracker.mean() > 0.50);
        assert!(tracker.mean() < 0.70);
        assert_eq!(tracker.total_fills(), 8);
        assert_eq!(tracker.correct_directions(), 8);
    }

    #[test]
    fn test_bayesian_alpha_updates_on_incorrect() {
        let mut tracker = BayesianAlphaTracker::new();
        
        // 8 incorrect predictions
        for _ in 0..8 {
            tracker.update(0.5, false);
        }
        
        // Posterior mean should decrease: 2/(8+6+8) ≈ 0.14
        assert!(tracker.mean() < 0.20);
        assert_eq!(tracker.total_fills(), 8);
        assert_eq!(tracker.correct_directions(), 0);
    }

    #[test]
    fn test_bayesian_uncertainty_decreases_with_data() {
        let mut tracker = BayesianAlphaTracker::new();
        let initial_uncertainty = tracker.uncertainty_cost();
        
        // Add data
        for _ in 0..20 {
            tracker.update(0.5, true);
        }
        
        // Uncertainty should decrease with more data
        assert!(tracker.uncertainty_cost() < initial_uncertainty);
    }

    #[test]
    fn test_fill_rate_estimator() {
        let mut estimator = FillRateEstimator::new();
        
        // Prior: 5 fills/hour
        assert!((estimator.lambda() - 5.0).abs() < 0.1);
        
        // Simulate fills every 6 minutes (10 fills/hour)
        for i in 0..5 {
            estimator.on_fill(i * 360_000);  // 6 min = 360,000 ms
        }
        
        // Rate should move toward 10 fills/hour
        assert!(estimator.lambda() > 5.0);
    }

    #[test]
    fn test_estimator_update_from_fill() {
        let mut estimator = TheoreticalEdgeEstimator::new();
        
        // Initial alpha
        let initial_alpha = estimator.bayesian_alpha();
        
        // Correct fill (imbalance positive, price went up)
        estimator.update_from_fill(0.5, 1, 1, 1000);
        
        // Alpha should increase
        assert!(estimator.bayesian_alpha() > initial_alpha);
        assert_eq!(estimator.bayesian_fills(), 1);
    }

    #[test]
    fn test_edge_uses_bayesian_alpha() {
        let mut estimator = TheoreticalEdgeEstimator::new();
        
        // Get edge with prior alpha (0.25)
        let result_before = estimator.calculate_edge(0.40, 20.0, 0.001, 1.0);
        
        // Add 10 correct fills to boost alpha
        for i in 0..10 {
            estimator.update_from_fill(0.5, 1, 1, (i + 1) * 1000);
        }
        
        // Get edge with updated alpha (should be higher)
        let result_after = estimator.calculate_edge(0.40, 20.0, 0.001, 1.0);
        
        // Higher alpha → higher p_correct → higher directional edge
        assert!(result_after.p_correct > result_before.p_correct);
    }

    // ==================== Regime-Aware Bayesian Adverse Tests ====================

    #[test]
    fn test_regime_adverse_priors() {
        let tracker = RegimeAwareBayesianAdverse::new();

        // Check regime-specific priors
        // Calm (0): E[p] = 0.10
        assert!((tracker.mean_for_regime(0) - 0.10).abs() < 0.01);
        // Normal (1): E[p] = 0.15
        assert!((tracker.mean_for_regime(1) - 0.15).abs() < 0.01);
        // Volatile (2): E[p] = 0.25
        assert!((tracker.mean_for_regime(2) - 0.25).abs() < 0.01);

        // Default regime is Normal (1)
        assert_eq!(tracker.current_regime(), 1);
    }

    #[test]
    fn test_regime_adverse_regime_switching() {
        let mut tracker = RegimeAwareBayesianAdverse::new();

        // Initial mean is normal regime
        let normal_mean = tracker.mean();

        // Switch to volatile
        tracker.set_regime(2);
        let volatile_mean = tracker.mean();

        // Volatile should have higher adverse prior
        assert!(volatile_mean > normal_mean);
        assert!((volatile_mean - 0.25).abs() < 0.01);

        // Switch to calm
        tracker.set_regime(0);
        let calm_mean = tracker.mean();

        // Calm should have lower adverse prior
        assert!(calm_mean < normal_mean);
        assert!((calm_mean - 0.10).abs() < 0.01);
    }

    #[test]
    fn test_regime_adverse_auto_classification() {
        let mut tracker = RegimeAwareBayesianAdverse::new();

        // High volatility and/or cascade → Volatile regime
        tracker.update_regime(2.5, 1.0, 0.3); // vol_ratio=2.5 > 2.0
        assert_eq!(tracker.current_regime(), 2);

        tracker.update_regime(1.0, 1.0, 0.6); // cascade_prob=0.6 > 0.5
        assert_eq!(tracker.current_regime(), 2);

        // Low volatility and tight spread → Calm regime
        tracker.update_regime(0.6, 0.7, 0.1); // vol<0.7, spread<0.8
        assert_eq!(tracker.current_regime(), 0);

        // Normal conditions
        tracker.update_regime(1.0, 1.0, 0.2);
        assert_eq!(tracker.current_regime(), 1);
    }

    #[test]
    fn test_regime_adverse_updates_in_current_regime() {
        let mut tracker = RegimeAwareBayesianAdverse::new();
        tracker.set_regime(1); // Normal

        // Add adverse fills in normal regime
        for _ in 0..10 {
            tracker.update(true); // Adverse
        }

        // Normal regime mean should increase
        assert!(tracker.mean() > 0.15);

        // Other regimes should be unchanged (still at prior)
        assert!((tracker.mean_for_regime(0) - 0.10).abs() < 0.01);
        assert!((tracker.mean_for_regime(2) - 0.25).abs() < 0.01);

        // Fill count should be tracked per regime
        let summary = tracker.summary();
        assert_eq!(summary.fills_per_regime[1], 10);
        assert_eq!(summary.fills_per_regime[0], 0);
        assert_eq!(summary.fills_per_regime[2], 0);
    }

    #[test]
    fn test_regime_adverse_weighted_updates() {
        let mut tracker = RegimeAwareBayesianAdverse::new();
        tracker.set_regime(1);

        let _initial_mean = tracker.mean();

        // High-impact adverse fill (high urgency, large AS)
        tracker.update_weighted(true, 3.0, 15.0); // urgency=3, as_bps=15

        let after_weighted = tracker.mean();

        // Reset and do normal update
        tracker = RegimeAwareBayesianAdverse::new();
        tracker.set_regime(1);
        tracker.update(true);

        let after_normal = tracker.mean();

        // Weighted adverse update should shift posterior more than normal
        assert!(after_weighted > after_normal);
    }

    #[test]
    fn test_regime_adverse_credible_interval() {
        let tracker = RegimeAwareBayesianAdverse::new();

        let (lower, upper) = tracker.credible_interval_95();

        // CI should contain the mean
        let mean = tracker.mean();
        assert!(lower <= mean);
        assert!(upper >= mean);

        // CI bounds should be valid probabilities
        assert!(lower >= 0.0);
        assert!(upper <= 1.0);
    }

    #[test]
    fn test_regime_adverse_decay() {
        let mut tracker = RegimeAwareBayesianAdverse::new();
        tracker.set_regime(1);

        // Add data to shift away from prior
        for _ in 0..20 {
            tracker.update(true);
        }

        let before_decay = tracker.mean();
        assert!(before_decay > 0.15); // Should have moved up from prior

        // Decay with 50% retention
        tracker.decay_current(0.5);

        let after_decay = tracker.mean();

        // Should move back toward prior but not fully
        assert!(after_decay < before_decay);
        assert!(after_decay > 0.15);
    }

    #[test]
    fn test_regime_adverse_variance_decreases() {
        let mut tracker = RegimeAwareBayesianAdverse::new();
        tracker.set_regime(1);

        let initial_variance = tracker.variance();

        // Add data
        for _ in 0..50 {
            tracker.update(false); // Favorable fills
        }

        // Variance should decrease with more observations
        assert!(tracker.variance() < initial_variance);
    }

    #[test]
    fn test_estimator_uses_regime_adverse() {
        let mut estimator = TheoreticalEdgeEstimator::new();

        // Get edge in normal regime
        estimator.set_regime(1);
        let result_normal = estimator.calculate_edge_enhanced(&EnhancedEdgeInput {
            book_imbalance: 0.4,
            spread_bps: 20.0,
            sigma: 0.001,
            tau_seconds: 1.0,
            ..Default::default()
        });

        // Switch to volatile regime (higher adverse prior)
        estimator.update_adverse_regime(2.5, 1.5, 0.6); // Triggers volatile

        let result_volatile = estimator.calculate_edge_enhanced(&EnhancedEdgeInput {
            book_imbalance: 0.4,
            spread_bps: 20.0,
            sigma: 0.001,
            tau_seconds: 1.0,
            ..Default::default()
        });

        // Volatile regime should have higher adverse cost → lower expected edge
        assert!(result_volatile.adverse_cost_bps > result_normal.adverse_cost_bps);
        assert!(result_volatile.expected_edge_bps < result_normal.expected_edge_bps);
    }

    // ==================== Phase 7: Hawkes-Bayesian Fusion Tests ====================

    #[test]
    fn test_regime_adverse_hawkes_update_regime() {
        let mut tracker = RegimeAwareBayesianAdverse::new();

        // Without Hawkes excitation, normal volatility → Normal regime
        tracker.update_regime_with_hawkes(1.0, 1.0, 0.2, false, 0.3);
        assert_eq!(tracker.current_regime(), 1); // Normal

        // With Hawkes high excitation, should shift to Volatile
        tracker.update_regime_with_hawkes(1.0, 1.0, 0.2, true, 0.8);
        assert_eq!(tracker.current_regime(), 2); // Volatile

        // Low vol + low Hawkes → Calm
        tracker.update_regime_with_hawkes(0.5, 0.6, 0.1, false, 0.2);
        assert_eq!(tracker.current_regime(), 0); // Calm

        // Low vol but high Hawkes branching → can't be calm
        tracker.update_regime_with_hawkes(0.5, 0.6, 0.1, false, 0.6);
        assert_eq!(tracker.current_regime(), 1); // Normal (not calm due to Hawkes)
    }

    #[test]
    fn test_regime_adverse_hawkes_cascade_boost() {
        let mut tracker = RegimeAwareBayesianAdverse::new();

        // Low cascade prob but high Hawkes excitation → should become volatile
        // Hawkes boost: (0.8 - 0.5)^2 * 2.0 = 0.09 * 2.0 = 0.18
        // Effective cascade = 0.3 + 0.18 = 0.48 > 0.4 threshold
        tracker.update_regime_with_hawkes(1.5, 1.0, 0.3, true, 0.8);
        assert_eq!(tracker.current_regime(), 2); // Volatile
    }

    #[test]
    fn test_regime_adverse_hawkes_weighted_update() {
        let mut tracker = RegimeAwareBayesianAdverse::new();
        tracker.set_regime(1);

        // High excitation adverse fill (low penalty, high p_cluster)
        tracker.update_weighted_with_hawkes(true, 2.0, 10.0, 0.6, 0.8);
        let high_excitation_mean = tracker.mean();

        // Reset
        let mut tracker2 = RegimeAwareBayesianAdverse::new();
        tracker2.set_regime(1);

        // Normal excitation adverse fill (high penalty, low p_cluster)
        tracker2.update_weighted_with_hawkes(true, 2.0, 10.0, 1.0, 0.1);
        let normal_excitation_mean = tracker2.mean();

        // High excitation should update more strongly
        assert!(high_excitation_mean > normal_excitation_mean);
    }

    #[test]
    fn test_regime_adverse_hawkes_favorable_update() {
        let mut tracker = RegimeAwareBayesianAdverse::new();
        tracker.set_regime(1);

        let initial_mean = tracker.mean();

        // High excitation favorable fill (market absorbing flow well)
        tracker.update_weighted_with_hawkes(false, 1.0, 5.0, 0.5, 0.9);

        // Should decrease adverse mean (favorable fill during stress = informative)
        assert!(tracker.mean() < initial_mean);
    }

    #[test]
    fn test_effective_adverse_with_hawkes_calm() {
        let mut tracker = RegimeAwareBayesianAdverse::new();
        tracker.set_regime(1);

        let base_mean = tracker.mean();

        // No excitation: should return base mean
        let effective = tracker.effective_adverse_with_hawkes(0.0, false);
        assert!((effective - base_mean).abs() < 0.01);
    }

    #[test]
    fn test_effective_adverse_with_hawkes_excited() {
        let mut tracker = RegimeAwareBayesianAdverse::new();
        tracker.set_regime(1);

        let base_mean = tracker.mean();
        let (_, upper_ci) = tracker.credible_interval_95();

        // High excitation: should blend toward upper CI
        let effective = tracker.effective_adverse_with_hawkes(0.9, true);

        // Should be between mean and upper CI
        assert!(effective > base_mean);
        assert!(effective <= upper_ci);
    }

    #[test]
    fn test_effective_adverse_hawkes_p_cluster_scaling() {
        let mut tracker = RegimeAwareBayesianAdverse::new();
        tracker.set_regime(1);

        // Moderate cluster prob
        let effective_moderate = tracker.effective_adverse_with_hawkes(0.6, true);

        // High cluster prob
        let effective_high = tracker.effective_adverse_with_hawkes(0.95, true);

        // Higher p_cluster should give more conservative estimate
        assert!(effective_high > effective_moderate);
    }
}
