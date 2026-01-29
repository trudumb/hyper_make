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
        let boost = (self.btc_return_bps / 50.0).clamp(-0.10, 0.10) * self.correlation;
        boost
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
            adverse_prior: 0.15,   // From PIN model estimates (lowered from 0.30)
            min_edge_bps: 1.0,     // Minimum edge AFTER fees
            min_imbalance: 0.10,   // Ignore very weak imbalances
            btc_correlation_threshold: 0.5, // Only use BTC signal if correlated
            btc_correlation_prior: 0.7,     // Assume HYPE correlates with BTC
            fee_bps: 5.0,          // Conservative: 5 bps for HIP-3 (maker + buffer)
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
        // This prevents complete zeroing on illiquids while still penalizing low fill probability
        let fill_dampening = 0.5 + 0.5 * p_fill;  // Range: [0.5, 1.0]
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
        if self.calculations % 100 == 0 {
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
        imbalance: f64,
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
        let size_weight = (fill_size / 0.01).sqrt().min(2.0).max(0.5);
        
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
}
