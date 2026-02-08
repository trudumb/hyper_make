//! Kelly criterion position sizer.
//!
//! Implements Kelly optimal sizing for market making:
//!
//! ```text
//! f* = (p × b - q) / b
//! optimal_size = kelly_fraction × bankroll × f*
//! ```
//!
//! Where:
//! - p = P(edge > 0) from Bayesian model
//! - b = E[win] / E[loss] from realized fills (odds ratio)
//! - kelly_fraction = 0.25 (fractional Kelly for safety)

use serde::{Deserialize, Serialize};

/// Standard normal CDF approximation (Abramowitz and Stegun).
/// Internal function for Kelly calculations.
fn normal_cdf(x: f64) -> f64 {
    // Constants for approximation
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / std::f64::consts::SQRT_2;

    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();

    0.5 * (1.0 + sign * y)
}

/// Kelly criterion position sizer.
///
/// Uses realized win/loss ratios and edge probability to compute optimal position size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KellySizer {
    /// Kelly fraction (0.25 = quarter Kelly, conservative)
    pub kelly_fraction: f64,

    /// Minimum P(win) required to take a position
    pub min_p_win: f64,

    /// Minimum edge (bps) required to take a position
    pub min_edge_bps: f64,

    /// Maximum position fraction (cap on f*)
    pub max_position_fraction: f64,

    /// Win/loss ratio tracker
    pub win_loss_tracker: WinLossTracker,

    /// Whether Kelly sizing is enabled
    pub enabled: bool,
}

impl Default for KellySizer {
    fn default() -> Self {
        Self {
            kelly_fraction: 0.25,
            min_p_win: 0.55,
            min_edge_bps: 2.0,
            max_position_fraction: 0.5,
            win_loss_tracker: WinLossTracker::default(),
            enabled: false, // Conservative: disabled by default
        }
    }
}

impl KellySizer {
    /// Create a new Kelly sizer with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom Kelly fraction.
    pub fn with_kelly_fraction(kelly_fraction: f64) -> Self {
        Self {
            kelly_fraction: kelly_fraction.clamp(0.05, 1.0),
            ..Default::default()
        }
    }

    /// Check if the sizer is warmed up (has enough win/loss data).
    pub fn is_warmed_up(&self) -> bool {
        self.win_loss_tracker.is_warmed_up()
    }

    /// Record a winning trade.
    pub fn record_win(&mut self, win_bps: f64) {
        self.win_loss_tracker.record_win(win_bps);
    }

    /// Record a losing trade.
    pub fn record_loss(&mut self, loss_bps: f64) {
        self.win_loss_tracker.record_loss(loss_bps);
    }

    /// Calculate Kelly-optimal position size.
    ///
    /// # Arguments
    /// * `edge_mean_bps` - Expected edge in basis points
    /// * `edge_std_bps` - Standard deviation of edge estimate in basis points
    /// * `p_fill` - Probability of fill (0 to 1)
    /// * `margin_available` - Available margin in USD
    /// * `leverage` - Account leverage
    /// * `price` - Current asset price
    ///
    /// # Returns
    /// Optimal position size in asset units.
    pub fn kelly_size(
        &self,
        edge_mean_bps: f64,
        edge_std_bps: f64,
        p_fill: f64,
        margin_available: f64,
        leverage: f64,
        price: f64,
    ) -> f64 {
        if !self.enabled {
            return 0.0;
        }

        // 1. P(edge > 0) from normal CDF
        let z_score = edge_mean_bps / edge_std_bps.max(0.01);
        let p_positive = normal_cdf(z_score);

        // Check minimum win probability
        if p_positive < self.min_p_win {
            return 0.0;
        }

        // Check minimum edge
        if edge_mean_bps < self.min_edge_bps {
            return 0.0;
        }

        // 2. Win/loss ratio from tracker (odds ratio b)
        let b = self.win_loss_tracker.odds_ratio();
        if b <= 0.0 {
            return 0.0;
        }

        // 3. Kelly formula: f* = (p × b - q) / b
        let q = 1.0 - p_positive;
        let f_full = (p_positive * b - q) / b;
        if f_full <= 0.0 {
            return 0.0;
        }

        // 4. Fractional Kelly with cap
        let f_optimal = (self.kelly_fraction * f_full).min(self.max_position_fraction);

        // 5. Convert to position size
        // bankroll = margin_available × leverage
        let bankroll = margin_available * leverage;
        let size_usd = f_optimal * bankroll;

        // 6. Adjust for fill probability and convert to asset units
        (size_usd / price.max(0.01)) * p_fill
    }

    /// Calculate position size during warmup (before Kelly is calibrated).
    ///
    /// Returns a conservative fraction of the position limit.
    pub fn warmup_size(&self, position_limit: f64) -> f64 {
        position_limit * 0.5 // Conservative 50% during warmup
    }

    /// Get Kelly sizing decision for a given edge estimate.
    ///
    /// Returns (should_trade, optimal_fraction, confidence).
    pub fn sizing_decision(
        &self,
        edge_mean_bps: f64,
        edge_std_bps: f64,
    ) -> (bool, f64, f64) {
        if !self.enabled {
            return (false, 0.0, 0.0);
        }

        let z_score = edge_mean_bps / edge_std_bps.max(0.01);
        let p_positive = normal_cdf(z_score);

        // Confidence is how sure we are about positive edge
        let confidence = (p_positive - 0.5) * 2.0; // Maps [0.5, 1.0] to [0, 1]

        if p_positive < self.min_p_win || edge_mean_bps < self.min_edge_bps {
            return (false, 0.0, confidence.max(0.0));
        }

        let b = self.win_loss_tracker.odds_ratio();
        if b <= 0.0 {
            return (false, 0.0, confidence.max(0.0));
        }

        let q = 1.0 - p_positive;
        let f_full = (p_positive * b - q) / b;
        if f_full <= 0.0 {
            return (false, 0.0, confidence.max(0.0));
        }

        let f_optimal = (self.kelly_fraction * f_full).min(self.max_position_fraction);

        (true, f_optimal, confidence.max(0.0))
    }
}

/// Tracks realized win/loss ratio from fills using EWMA.
///
/// Uses exponentially weighted moving averages to track:
/// - Average win size (bps)
/// - Average loss size (bps)
/// - Win rate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WinLossTracker {
    /// EWMA of win sizes (bps)
    ewma_wins: f64,

    /// Count of wins
    n_wins: u64,

    /// EWMA of loss sizes (bps)
    ewma_losses: f64,

    /// Count of losses
    n_losses: u64,

    /// EWMA decay factor (0.99 = ~100 trade half-life)
    decay: f64,

    /// Prior odds ratio (used during warmup)
    prior_odds: f64,

    /// Minimum samples before using tracked values
    min_samples: usize,
}

impl Default for WinLossTracker {
    fn default() -> Self {
        Self {
            ewma_wins: 5.0,   // Prior: 5 bps average win
            n_wins: 0,
            ewma_losses: 3.0, // Prior: 3 bps average loss
            n_losses: 0,
            decay: 0.99,
            prior_odds: 1.5, // Prior odds ratio (conservative)
            min_samples: 20,
        }
    }
}

impl WinLossTracker {
    /// Create a new tracker with custom decay.
    pub fn with_decay(decay: f64) -> Self {
        Self {
            decay: decay.clamp(0.9, 0.999),
            ..Default::default()
        }
    }

    /// Record a winning trade.
    pub fn record_win(&mut self, win_bps: f64) {
        let win = win_bps.abs().max(0.1); // Floor at 0.1 bps
        self.ewma_wins = self.decay * self.ewma_wins + (1.0 - self.decay) * win;
        self.n_wins += 1;
    }

    /// Record a losing trade.
    pub fn record_loss(&mut self, loss_bps: f64) {
        let loss = loss_bps.abs().max(0.1); // Floor at 0.1 bps
        self.ewma_losses = self.decay * self.ewma_losses + (1.0 - self.decay) * loss;
        self.n_losses += 1;
    }

    /// Check if tracker has enough samples.
    pub fn is_warmed_up(&self) -> bool {
        (self.n_wins + self.n_losses) as usize >= self.min_samples
    }

    /// Get the odds ratio b = E[win] / E[loss].
    ///
    /// Returns blended prior/observed based on sample count.
    pub fn odds_ratio(&self) -> f64 {
        let total = self.n_wins + self.n_losses;

        if total == 0 {
            return self.prior_odds;
        }

        // Calculate observed odds ratio
        let observed_odds = self.ewma_wins / self.ewma_losses.max(0.1);

        // Blend with prior based on sample count
        // Weight shifts from prior to observed as samples increase
        let sample_weight = (total as f64 / self.min_samples as f64).min(1.0);
        let prior_weight = 1.0 - sample_weight;

        prior_weight * self.prior_odds + sample_weight * observed_odds
    }

    /// Get the empirical win rate.
    pub fn win_rate(&self) -> f64 {
        let total = self.n_wins + self.n_losses;
        if total == 0 {
            return 0.5; // Prior
        }
        self.n_wins as f64 / total as f64
    }

    /// Get average win size (bps).
    pub fn avg_win(&self) -> f64 {
        self.ewma_wins
    }

    /// Get average loss size (bps).
    pub fn avg_loss(&self) -> f64 {
        self.ewma_losses
    }

    /// Get total number of trades recorded.
    pub fn total_trades(&self) -> u64 {
        self.n_wins + self.n_losses
    }

    /// Reset tracker to initial state.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Restore tracker state from checkpoint data.
    pub fn restore_from_checkpoint(
        &mut self,
        ewma_wins: f64,
        n_wins: u64,
        ewma_losses: f64,
        n_losses: u64,
        decay: f64,
    ) {
        self.ewma_wins = ewma_wins;
        self.n_wins = n_wins;
        self.ewma_losses = ewma_losses;
        self.n_losses = n_losses;
        self.decay = decay;
    }
}

/// Configuration for Kelly sizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KellyConfig {
    /// Whether Kelly sizing is enabled
    pub use_kelly_sizing: bool,

    /// Kelly fraction (0.25 = quarter Kelly)
    pub kelly_fraction: f64,

    /// Minimum P(edge > 0) to trade
    pub min_p_win: f64,

    /// Minimum expected edge (bps) to trade
    pub min_edge_bps: f64,

    /// Maximum position as fraction of bankroll
    pub max_position_fraction: f64,

    /// EWMA decay for win/loss tracker
    pub tracker_decay: f64,

    /// Minimum trades before using Kelly
    pub min_warmup_trades: usize,
}

impl Default for KellyConfig {
    fn default() -> Self {
        Self {
            use_kelly_sizing: false, // Conservative: disabled by default
            kelly_fraction: 0.25,
            min_p_win: 0.55,
            min_edge_bps: 2.0,
            max_position_fraction: 0.5,
            tracker_decay: 0.99,
            min_warmup_trades: 20,
        }
    }
}

impl From<KellyConfig> for KellySizer {
    fn from(config: KellyConfig) -> Self {
        Self {
            kelly_fraction: config.kelly_fraction,
            min_p_win: config.min_p_win,
            min_edge_bps: config.min_edge_bps,
            max_position_fraction: config.max_position_fraction,
            win_loss_tracker: WinLossTracker {
                decay: config.tracker_decay,
                min_samples: config.min_warmup_trades,
                ..Default::default()
            },
            enabled: config.use_kelly_sizing,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kelly_requires_edge() {
        let sizer = KellySizer {
            enabled: true,
            ..Default::default()
        };

        // No edge -> no size
        let size = sizer.kelly_size(0.0, 1.0, 0.5, 10000.0, 5.0, 100.0);
        assert_eq!(size, 0.0, "Zero edge should give zero size");

        // Negative edge -> no size
        let size = sizer.kelly_size(-5.0, 1.0, 0.5, 10000.0, 5.0, 100.0);
        assert_eq!(size, 0.0, "Negative edge should give zero size");
    }

    #[test]
    fn test_kelly_positive_edge() {
        let mut sizer = KellySizer {
            enabled: true,
            ..Default::default()
        };

        // Warm up the tracker
        for _ in 0..30 {
            sizer.record_win(10.0);
            sizer.record_loss(5.0);
        }

        // Positive edge with high confidence -> positive size
        let size = sizer.kelly_size(
            10.0, // 10 bps mean edge
            2.0,  // 2 bps std
            1.0,  // 100% fill probability
            10000.0,
            5.0,
            100.0,
        );
        assert!(size > 0.0, "Positive edge should give positive size: {}", size);
    }

    #[test]
    fn test_kelly_low_confidence() {
        let mut sizer = KellySizer {
            enabled: true,
            min_p_win: 0.55,
            ..Default::default()
        };

        // Warm up
        for _ in 0..30 {
            sizer.record_win(10.0);
            sizer.record_loss(5.0);
        }

        // Edge with high uncertainty -> low confidence -> no trade
        // z_score = 2.0 / 20.0 = 0.1 -> p_positive ≈ 0.54 < min_p_win (0.55)
        let size = sizer.kelly_size(
            2.0,  // 2 bps mean edge
            20.0, // 20 bps std (very high uncertainty)
            1.0,
            10000.0,
            5.0,
            100.0,
        );
        assert_eq!(size, 0.0, "Low confidence should give zero size");
    }

    #[test]
    fn test_win_loss_tracker() {
        let mut tracker = WinLossTracker::default();

        // Initial state uses priors
        assert_eq!(tracker.win_rate(), 0.5);
        assert!(tracker.odds_ratio() > 0.0);

        // Record some trades
        for _ in 0..10 {
            tracker.record_win(10.0);
        }
        for _ in 0..5 {
            tracker.record_loss(5.0);
        }

        // Should have 2:1 win rate
        assert!(
            (tracker.win_rate() - 0.666).abs() < 0.1,
            "Win rate should be ~66%: {}",
            tracker.win_rate()
        );

        // Odds ratio should be high (wins are larger than losses)
        assert!(
            tracker.odds_ratio() > 1.5,
            "Odds ratio should be > 1.5: {}",
            tracker.odds_ratio()
        );
    }

    #[test]
    fn test_warmup_detection() {
        let sizer = KellySizer::default();
        assert!(!sizer.is_warmed_up(), "Should not be warmed up initially");

        let mut sizer = KellySizer::default();
        for _ in 0..25 {
            sizer.record_win(5.0);
        }
        assert!(sizer.is_warmed_up(), "Should be warmed up after 25 trades");
    }

    #[test]
    fn test_normal_cdf() {
        // Test known values
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.001, "CDF(0) should be 0.5");
        assert!(normal_cdf(3.0) > 0.99, "CDF(3) should be > 0.99");
        assert!(normal_cdf(-3.0) < 0.01, "CDF(-3) should be < 0.01");
    }

    #[test]
    fn test_sizing_decision() {
        let mut sizer = KellySizer {
            enabled: true,
            ..Default::default()
        };

        // Warm up
        for _ in 0..30 {
            sizer.record_win(8.0);
            sizer.record_loss(4.0);
        }

        // High edge, low uncertainty -> should trade
        let (should_trade, fraction, confidence) = sizer.sizing_decision(10.0, 2.0);
        assert!(should_trade, "Should trade with high edge");
        assert!(fraction > 0.0, "Fraction should be positive");
        assert!(confidence > 0.5, "Confidence should be high");

        // Low edge, high uncertainty -> should not trade
        let (should_trade, _, _) = sizer.sizing_decision(1.0, 5.0);
        assert!(!should_trade, "Should not trade with low edge ratio");
    }
}
