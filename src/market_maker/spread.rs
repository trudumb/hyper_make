//! Spread Process Estimator - tracks and models bid-ask spread dynamics.
//!
//! The spread is not constant but follows a stochastic process that
//! mean-reverts to a volatility-dependent fair value.
//!
//! # Key Observations
//! - Spreads widen during high volatility
//! - Spreads mean-revert to an asset-specific baseline
//! - Extreme spreads (tight or wide) tend to normalize
//! - Spread percentile helps identify opportunities
//!
//! # The Model
//! ```text
//! dS = κ_s(θ_s(σ) - S)dt + ξ_s × dW_s
//! ```
//! where:
//! - S: current spread
//! - κ_s: mean-reversion speed
//! - θ_s(σ): fair spread (function of volatility)
//! - ξ_s: spread volatility

use std::collections::VecDeque;
use std::time::Instant;

/// Configuration for the spread process estimator.
#[derive(Debug, Clone)]
pub struct SpreadConfig {
    /// Mean-reversion speed for spread
    /// Default: 0.1 (slow mean reversion)
    pub mean_reversion_speed: f64,

    /// Base fair spread (as a fraction of mid)
    /// Default: 0.0002 (2 bps)
    pub base_fair_spread: f64,

    /// How much volatility increases fair spread
    /// fair_spread = base + vol_sensitivity × σ
    /// Default: 0.1
    pub vol_sensitivity: f64,

    /// Maximum history size for percentile calculation
    pub max_history: usize,

    /// EWMA decay factor for spread averaging
    /// Default: 0.95
    pub ewma_lambda: f64,

    /// Minimum observations for warmup
    pub min_observations: usize,
}

impl Default for SpreadConfig {
    fn default() -> Self {
        Self {
            mean_reversion_speed: 0.1,
            base_fair_spread: 0.0002, // 2 bps
            vol_sensitivity: 0.1,
            max_history: 1000,
            ewma_lambda: 0.95,
            min_observations: 20,
        }
    }
}

/// A single spread observation.
#[derive(Debug, Clone, Copy)]
struct SpreadObservation {
    spread: f64, // As a fraction of mid
    #[allow(dead_code)]
    timestamp: Instant,
    volatility: f64, // Current volatility when observed
}

/// Spread process estimator.
pub struct SpreadProcessEstimator {
    config: SpreadConfig,

    /// Current spread (as fraction of mid)
    current_spread: f64,

    /// EWMA of spread
    ewma_spread: f64,

    /// Historical spreads for percentile
    spread_history: VecDeque<SpreadObservation>,

    /// Last update time
    last_update: Instant,

    /// Observation count
    observation_count: usize,

    /// Current volatility (for fair spread calculation)
    current_volatility: f64,

    /// Minimum spread observed
    min_spread: f64,

    /// Maximum spread observed
    max_spread: f64,
}

impl SpreadProcessEstimator {
    /// Create a new spread process estimator.
    pub fn new(config: SpreadConfig) -> Self {
        Self {
            current_spread: config.base_fair_spread,
            ewma_spread: config.base_fair_spread,
            config,
            spread_history: VecDeque::with_capacity(1000),
            last_update: Instant::now(),
            observation_count: 0,
            current_volatility: 0.0001,
            min_spread: f64::MAX,
            max_spread: 0.0,
        }
    }

    /// Update with a new spread observation.
    ///
    /// # Arguments
    /// - `best_bid`: Current best bid price
    /// - `best_ask`: Current best ask price
    /// - `volatility`: Current volatility estimate (for fair spread)
    pub fn update(&mut self, best_bid: f64, best_ask: f64, volatility: f64) {
        if best_bid <= 0.0 || best_ask <= best_bid {
            return; // Invalid prices
        }

        let now = Instant::now();
        let mid = (best_bid + best_ask) / 2.0;
        let spread = (best_ask - best_bid) / mid; // As fraction

        // Update current spread
        self.current_spread = spread;
        self.current_volatility = volatility;

        // Update EWMA
        if self.observation_count == 0 {
            self.ewma_spread = spread;
        } else {
            self.ewma_spread = self.config.ewma_lambda * self.ewma_spread
                + (1.0 - self.config.ewma_lambda) * spread;
        }

        // Track min/max
        self.min_spread = self.min_spread.min(spread);
        self.max_spread = self.max_spread.max(spread);

        // Store observation
        let obs = SpreadObservation {
            spread,
            timestamp: now,
            volatility,
        };
        self.spread_history.push_back(obs);

        // Trim old history
        while self.spread_history.len() > self.config.max_history {
            self.spread_history.pop_front();
        }

        self.observation_count += 1;
        self.last_update = now;
    }

    /// Check if estimator is warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.observation_count >= self.config.min_observations
    }

    /// Get current spread (as fraction of mid).
    pub fn current_spread(&self) -> f64 {
        self.current_spread
    }

    /// Get current spread in basis points.
    pub fn current_spread_bps(&self) -> f64 {
        self.current_spread * 10000.0
    }

    /// Get EWMA spread.
    pub fn ewma_spread(&self) -> f64 {
        self.ewma_spread
    }

    /// Get the fair spread given current volatility.
    pub fn fair_spread(&self) -> f64 {
        self.config.base_fair_spread + self.config.vol_sensitivity * self.current_volatility
    }

    /// Get the spread percentile (0 to 1).
    ///
    /// - 0.0: Spread is at minimum (very tight)
    /// - 0.5: Spread is at median
    /// - 1.0: Spread is at maximum (very wide)
    pub fn spread_percentile(&self) -> f64 {
        if self.spread_history.len() < 2 {
            return 0.5;
        }

        let mut spreads: Vec<f64> = self.spread_history.iter().map(|o| o.spread).collect();
        spreads.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let current = self.current_spread;
        let position = spreads.iter().filter(|&&s| s < current).count();

        position as f64 / spreads.len() as f64
    }

    /// Check if spread is tight (below fair value).
    pub fn is_spread_tight(&self) -> bool {
        self.current_spread < self.fair_spread() * 0.8
    }

    /// Check if spread is wide (above fair value).
    pub fn is_spread_wide(&self) -> bool {
        self.current_spread > self.fair_spread() * 1.5
    }

    /// Get expected spread over a time horizon using mean-reversion.
    ///
    /// E[S(t+T)] = S(t) × exp(-κT) + θ × (1 - exp(-κT))
    pub fn expected_spread(&self, horizon_secs: f64) -> f64 {
        let kappa = self.config.mean_reversion_speed;
        let theta = self.fair_spread();

        let decay = (-kappa * horizon_secs).exp();
        self.current_spread * decay + theta * (1.0 - decay)
    }

    /// Predicted spread - alias for expected_spread (First Principles Gap 4).
    pub fn predicted_spread(&self, horizon_secs: f64) -> f64 {
        self.expected_spread(horizon_secs)
    }

    /// Should we defer placing a quote because spread will improve?
    ///
    /// Returns (should_defer, expected_improvement_fraction) where:
    /// - should_defer: True if we should wait before quoting
    /// - expected_improvement: Fraction of current spread we expect to save
    ///
    /// Logic: If the spread is significantly wider than fair value and
    /// expected to improve by more than `improvement_threshold` within
    /// `horizon_secs`, we should defer.
    ///
    /// # Arguments
    /// - `horizon_secs`: How far to look ahead (default: 10 seconds)
    /// - `improvement_threshold`: Minimum improvement to defer (default: 0.2 = 20%)
    pub fn should_defer_quote(&self, horizon_secs: f64, improvement_threshold: f64) -> (bool, f64) {
        let current = self.current_spread;
        let expected = self.expected_spread(horizon_secs);

        if current <= 0.0 {
            return (false, 0.0);
        }

        // Calculate expected improvement (positive = spread will tighten)
        let improvement = (current - expected) / current;

        // Defer if:
        // 1. Spread is wide (above fair)
        // 2. Expected improvement exceeds threshold
        let should_defer = self.is_spread_wide() && improvement > improvement_threshold;

        (should_defer, improvement)
    }

    /// Should defer with default parameters.
    /// Uses 10 second horizon and 20% improvement threshold.
    pub fn should_defer_quote_default(&self) -> (bool, f64) {
        self.should_defer_quote(10.0, 0.2)
    }

    /// Get the optimal time to wait before quoting.
    ///
    /// Returns the time in seconds at which spread is expected to be
    /// closest to fair value, up to max_wait_secs.
    pub fn optimal_wait_time(&self, max_wait_secs: f64) -> f64 {
        // For OU process, spread converges monotonically to fair
        // If current > fair, waiting is always better (up to max)
        // If current < fair, don't wait at all
        let fair = self.fair_spread();

        if self.current_spread <= fair {
            return 0.0; // Already at or below fair
        }

        // Find time when expected spread = fair (solving for T)
        // S(t) × e^(-κT) + θ × (1 - e^(-κT)) = fair
        // Since fair = θ (by definition), this would take infinite time
        // Instead, find time when we're within 10% of fair

        let target = fair * 1.1; // Within 10% of fair
        let kappa = self.config.mean_reversion_speed;

        if kappa <= 1e-9 {
            return max_wait_secs;
        }

        // Solve: S(t) × e^(-κT) + θ × (1 - e^(-κT)) = target
        // S(t) × e^(-κT) - θ × e^(-κT) = target - θ
        // (S(t) - θ) × e^(-κT) = target - θ
        // e^(-κT) = (target - θ) / (S(t) - θ)

        let s0 = self.current_spread;
        let theta = fair;

        if (s0 - theta).abs() < 1e-12 {
            return 0.0; // Already at fair
        }

        let ratio = (target - theta) / (s0 - theta);
        if ratio <= 0.0 || ratio >= 1.0 {
            return max_wait_secs;
        }

        let t = -ratio.ln() / kappa;
        t.min(max_wait_secs).max(0.0)
    }

    /// Get spread volatility from recent history.
    pub fn spread_volatility(&self) -> f64 {
        if self.spread_history.len() < 2 {
            return 0.0;
        }

        let spreads: Vec<f64> = self.spread_history.iter().map(|o| o.spread).collect();
        let mean = spreads.iter().sum::<f64>() / spreads.len() as f64;

        let variance =
            spreads.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / spreads.len() as f64;

        variance.sqrt()
    }

    /// Get spread regime classification.
    pub fn spread_regime(&self) -> SpreadRegime {
        let _percentile = self.spread_percentile(); // Used for potential future percentile-based regime
        let fair = self.fair_spread();

        if self.current_spread < fair * 0.7 {
            SpreadRegime::VeryTight
        } else if self.current_spread < fair * 0.9 {
            SpreadRegime::Tight
        } else if self.current_spread < fair * 1.2 {
            SpreadRegime::Normal
        } else if self.current_spread < fair * 2.0 {
            SpreadRegime::Wide
        } else {
            SpreadRegime::VeryWide
        }
    }

    /// Get adjustment factor for quote sizing based on spread regime.
    ///
    /// - Tight spread: Reduce size (more competition)
    /// - Wide spread: Increase size (capture opportunity)
    pub fn sizing_factor(&self) -> f64 {
        match self.spread_regime() {
            SpreadRegime::VeryTight => 0.5,
            SpreadRegime::Tight => 0.75,
            SpreadRegime::Normal => 1.0,
            SpreadRegime::Wide => 1.25,
            SpreadRegime::VeryWide => 1.5,
        }
    }

    /// Get the correlation between spread and volatility.
    pub fn spread_vol_correlation(&self) -> f64 {
        if self.spread_history.len() < 10 {
            return 0.0;
        }

        let spreads: Vec<f64> = self.spread_history.iter().map(|o| o.spread).collect();
        let vols: Vec<f64> = self.spread_history.iter().map(|o| o.volatility).collect();

        // Simple correlation calculation
        let n = spreads.len() as f64;
        let mean_s = spreads.iter().sum::<f64>() / n;
        let mean_v = vols.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_s = 0.0;
        let mut var_v = 0.0;

        for (s, v) in spreads.iter().zip(vols.iter()) {
            let ds = s - mean_s;
            let dv = v - mean_v;
            cov += ds * dv;
            var_s += ds * ds;
            var_v += dv * dv;
        }

        if var_s < 1e-12 || var_v < 1e-12 {
            return 0.0;
        }

        cov / (var_s.sqrt() * var_v.sqrt())
    }

    /// Get summary statistics.
    pub fn summary(&self) -> SpreadSummary {
        SpreadSummary {
            is_warmed_up: self.is_warmed_up(),
            current_spread_bps: self.current_spread_bps(),
            ewma_spread_bps: self.ewma_spread * 10000.0,
            fair_spread_bps: self.fair_spread() * 10000.0,
            percentile: self.spread_percentile(),
            regime: self.spread_regime(),
            observation_count: self.observation_count,
            min_spread_bps: self.min_spread * 10000.0,
            max_spread_bps: self.max_spread * 10000.0,
        }
    }

    /// Reset the estimator.
    pub fn reset(&mut self) {
        self.spread_history.clear();
        self.current_spread = self.config.base_fair_spread;
        self.ewma_spread = self.config.base_fair_spread;
        self.observation_count = 0;
        self.min_spread = f64::MAX;
        self.max_spread = 0.0;
        self.last_update = Instant::now();
    }
}

/// Spread regime classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpreadRegime {
    VeryTight,
    Tight,
    Normal,
    Wide,
    VeryWide,
}

impl std::fmt::Display for SpreadRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpreadRegime::VeryTight => write!(f, "VeryTight"),
            SpreadRegime::Tight => write!(f, "Tight"),
            SpreadRegime::Normal => write!(f, "Normal"),
            SpreadRegime::Wide => write!(f, "Wide"),
            SpreadRegime::VeryWide => write!(f, "VeryWide"),
        }
    }
}

/// Summary of spread status.
#[derive(Debug, Clone)]
pub struct SpreadSummary {
    pub is_warmed_up: bool,
    pub current_spread_bps: f64,
    pub ewma_spread_bps: f64,
    pub fair_spread_bps: f64,
    pub percentile: f64,
    pub regime: SpreadRegime,
    pub observation_count: usize,
    pub min_spread_bps: f64,
    pub max_spread_bps: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SpreadConfig::default();
        assert_eq!(config.base_fair_spread, 0.0002);
        assert_eq!(config.min_observations, 20);
    }

    #[test]
    fn test_new_estimator() {
        let estimator = SpreadProcessEstimator::new(SpreadConfig::default());
        assert!(!estimator.is_warmed_up());
    }

    #[test]
    fn test_update_spread() {
        let mut estimator = SpreadProcessEstimator::new(SpreadConfig::default());

        estimator.update(99.0, 101.0, 0.001); // 2% spread at $100 mid

        assert!((estimator.current_spread_bps() - 200.0).abs() < 1.0); // ~200 bps
    }

    #[test]
    fn test_fair_spread() {
        let config = SpreadConfig {
            base_fair_spread: 0.0002,
            vol_sensitivity: 0.1,
            ..Default::default()
        };
        let mut estimator = SpreadProcessEstimator::new(config);

        estimator.update(99.0, 101.0, 0.001);

        // fair = 0.0002 + 0.1 * 0.001 = 0.0003 = 3 bps
        assert!((estimator.fair_spread() - 0.0003).abs() < 0.00001);
    }

    #[test]
    fn test_spread_percentile() {
        let config = SpreadConfig {
            min_observations: 5,
            ..Default::default()
        };
        let mut estimator = SpreadProcessEstimator::new(config);

        // Add increasing spreads
        for i in 1..=10 {
            let spread_pct = i as f64 * 0.001;
            let mid = 100.0;
            let half_spread = mid * spread_pct / 2.0;
            estimator.update(mid - half_spread, mid + half_spread, 0.001);
        }

        // Current spread is the largest, so percentile should be high
        assert!(estimator.spread_percentile() > 0.8);
    }

    #[test]
    fn test_spread_regime_classification() {
        let config = SpreadConfig {
            base_fair_spread: 0.001, // 10 bps base
            vol_sensitivity: 0.0,
            min_observations: 1,
            ..Default::default()
        };
        let mut estimator = SpreadProcessEstimator::new(config);

        // Very tight spread (5 bps with 10 bps fair)
        estimator.update(99.975, 100.025, 0.0);
        assert_eq!(estimator.spread_regime(), SpreadRegime::VeryTight);

        // Normal spread (10 bps)
        estimator.update(99.95, 100.05, 0.0);
        assert_eq!(estimator.spread_regime(), SpreadRegime::Normal);

        // Wide spread (25 bps)
        estimator.update(99.875, 100.125, 0.0);
        assert_eq!(estimator.spread_regime(), SpreadRegime::VeryWide);
    }

    #[test]
    fn test_expected_spread_mean_reversion() {
        let config = SpreadConfig {
            mean_reversion_speed: 0.5,
            base_fair_spread: 0.001,
            vol_sensitivity: 0.0,
            min_observations: 1,
            ..Default::default()
        };
        let mut estimator = SpreadProcessEstimator::new(config);

        // Set current spread far from fair
        estimator.update(99.9, 100.1, 0.0); // 20 bps vs 10 bps fair

        // Expected spread should converge toward fair over time
        let expected_10s = estimator.expected_spread(10.0);
        let expected_100s = estimator.expected_spread(100.0);

        assert!(expected_10s < estimator.current_spread);
        assert!(expected_100s < expected_10s);
    }

    #[test]
    fn test_sizing_factor() {
        let config = SpreadConfig {
            base_fair_spread: 0.001,
            vol_sensitivity: 0.0,
            min_observations: 1,
            ..Default::default()
        };
        let mut estimator = SpreadProcessEstimator::new(config);

        // Very tight spread -> reduce size
        estimator.update(99.975, 100.025, 0.0);
        assert!(estimator.sizing_factor() < 1.0);

        // Very wide spread -> increase size
        estimator.update(99.8, 100.2, 0.0);
        assert!(estimator.sizing_factor() > 1.0);
    }

    #[test]
    fn test_min_max_tracking() {
        let config = SpreadConfig {
            min_observations: 1,
            ..Default::default()
        };
        let mut estimator = SpreadProcessEstimator::new(config);

        estimator.update(99.95, 100.05, 0.0); // 10 bps
        estimator.update(99.99, 100.01, 0.0); // 2 bps
        estimator.update(99.90, 100.10, 0.0); // 20 bps

        let summary = estimator.summary();
        assert!((summary.min_spread_bps - 2.0).abs() < 0.1);
        assert!((summary.max_spread_bps - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_warmup() {
        let config = SpreadConfig {
            min_observations: 5,
            ..Default::default()
        };
        let mut estimator = SpreadProcessEstimator::new(config);

        assert!(!estimator.is_warmed_up());

        for _ in 0..5 {
            estimator.update(99.95, 100.05, 0.001);
        }

        assert!(estimator.is_warmed_up());
    }

    #[test]
    fn test_summary() {
        let config = SpreadConfig {
            min_observations: 3,
            ..Default::default()
        };
        let mut estimator = SpreadProcessEstimator::new(config);

        for _ in 0..5 {
            estimator.update(99.95, 100.05, 0.001);
        }

        let summary = estimator.summary();
        assert!(summary.is_warmed_up);
        assert_eq!(summary.observation_count, 5);
        assert!(summary.current_spread_bps > 0.0);
    }

    #[test]
    fn test_reset() {
        let config = SpreadConfig {
            min_observations: 3,
            ..Default::default()
        };
        let mut estimator = SpreadProcessEstimator::new(config);

        for _ in 0..5 {
            estimator.update(99.95, 100.05, 0.001);
        }

        estimator.reset();

        assert!(!estimator.is_warmed_up());
        assert_eq!(estimator.observation_count, 0);
    }
}
