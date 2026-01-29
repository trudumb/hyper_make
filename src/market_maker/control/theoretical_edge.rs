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
#[derive(Debug)]
pub struct TheoreticalEdgeEstimator {
    config: TheoreticalEdgeConfig,
    cross_asset: CrossAssetSignal,
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

        // P(direction correct | imbalance) = 0.5 + alpha * |imbalance|
        let base_p_correct = 0.5 + self.config.alpha * abs_imbalance;

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
        //    Positive when we have directional advantage
        let directional_edge_bps = expected_move_bps * (p_correct - 0.5);

        // 3. BTC boost contribution (separate tracking for diagnostics)
        let btc_boost_bps = expected_move_bps * btc_adjustment;

        // 4. Adverse selection cost = expected_move × P(informed)
        let adverse_cost_bps = expected_move_bps * self.config.adverse_prior;

        // 5. Trading fees (explicit deduction)
        let fee_cost_bps = self.config.fee_bps;

        // Total expected edge = spread capture + directional - adverse - fees
        let expected_edge_bps = spread_edge_bps + directional_edge_bps - adverse_cost_bps - fee_cost_bps;

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
}
