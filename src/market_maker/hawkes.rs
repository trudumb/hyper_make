//! Hawkes Order Flow Estimator - self-exciting point process for trade arrivals.
//!
//! Models order flow as a self-exciting point process where each trade increases
//! the probability of subsequent trades. This captures the clustering/momentum
//! behavior observed in real markets.
//!
//! # The Model
//! ```text
//! λ_buy(t)  = μ_buy  + ∫ α × e^(-β(t-s)) dN_buy(s)  + γ × e^(-β(t-s)) dN_sell(s)
//! λ_sell(t) = μ_sell + ∫ α × e^(-β(t-s)) dN_sell(s) + γ × e^(-β(t-s)) dN_buy(s)
//! ```
//!
//! where:
//! - λ(t): Instantaneous intensity (expected trades per second)
//! - μ: Baseline intensity (background rate)
//! - α: Self-excitation (how much each trade excites same-side intensity)
//! - β: Decay rate (how fast excitation fades)
//! - γ: Cross-excitation (how much each trade excites opposite-side intensity)
//!
//! # Key Properties
//! - **Clustering**: High activity begets more activity
//! - **Mean-reversion**: Intensity reverts to baseline μ
//! - **Asymmetry**: Buy and sell intensities can differ

use std::collections::VecDeque;
use std::time::Instant;

/// Configuration for the Hawkes order flow estimator.
#[derive(Debug, Clone)]
pub struct HawkesConfig {
    /// Baseline intensity (trades per second)
    /// Default: 0.5 (1 trade per 2 seconds)
    pub mu: f64,

    /// Self-excitation parameter (0 to 1)
    /// Higher values = more clustering
    /// Default: 0.3
    pub alpha: f64,

    /// Decay rate (per second)
    /// Higher values = faster decay of excitation
    /// Default: 0.1 (10 second half-life)
    pub beta: f64,

    /// Cross-excitation parameter (0 to 1)
    /// How much opposite-side trades increase intensity
    /// Default: 0.1
    pub gamma: f64,

    /// Maximum history to keep (in seconds)
    /// Default: 60
    pub max_history_secs: f64,

    /// Minimum trades for warmup
    pub min_trades: usize,
}

impl Default for HawkesConfig {
    fn default() -> Self {
        Self {
            mu: 0.5,
            alpha: 0.3,
            beta: 0.1,
            gamma: 0.1,
            max_history_secs: 60.0,
            min_trades: 10,
        }
    }
}

/// A single trade event for the Hawkes process.
#[derive(Debug, Clone, Copy)]
struct TradeEvent {
    timestamp: Instant,
    is_buy: bool,
    size: f64,
}

/// Hawkes order flow estimator.
pub struct HawkesOrderFlowEstimator {
    config: HawkesConfig,

    /// Recent trade events
    events: VecDeque<TradeEvent>,

    /// Current buy intensity
    lambda_buy: f64,

    /// Current sell intensity
    lambda_sell: f64,

    /// Last intensity update time
    last_update: Instant,

    /// Total trade count
    trade_count: usize,

    /// Running sum of buy volume
    buy_volume: f64,

    /// Running sum of sell volume
    sell_volume: f64,
}

impl HawkesOrderFlowEstimator {
    /// Create a new Hawkes order flow estimator.
    pub fn new(config: HawkesConfig) -> Self {
        Self {
            lambda_buy: config.mu,
            lambda_sell: config.mu,
            config,
            events: VecDeque::with_capacity(1000),
            last_update: Instant::now(),
            trade_count: 0,
            buy_volume: 0.0,
            sell_volume: 0.0,
        }
    }

    /// Record a trade event.
    ///
    /// # Arguments
    /// - `is_buy`: Whether this was a buy aggressor trade
    /// - `size`: Trade size (for volume-weighted analysis)
    pub fn record_trade(&mut self, is_buy: bool, size: f64) {
        let now = Instant::now();

        // Update intensities before adding new event
        self.update_intensities(now);

        // Record event
        let event = TradeEvent {
            timestamp: now,
            is_buy,
            size,
        };
        self.events.push_back(event);
        self.trade_count += 1;

        if is_buy {
            self.buy_volume += size;
        } else {
            self.sell_volume += size;
        }

        // Add excitation from this trade
        if is_buy {
            self.lambda_buy += self.config.alpha;
            self.lambda_sell += self.config.gamma;
        } else {
            self.lambda_sell += self.config.alpha;
            self.lambda_buy += self.config.gamma;
        }

        // Cleanup old events
        self.cleanup_old_events(now);

        self.last_update = now;
    }

    /// Update intensities based on time decay.
    fn update_intensities(&mut self, now: Instant) {
        let dt = now.duration_since(self.last_update).as_secs_f64();
        if dt <= 0.0 {
            return;
        }

        let decay = (-self.config.beta * dt).exp();

        // Decay intensities toward baseline
        self.lambda_buy = self.config.mu + (self.lambda_buy - self.config.mu) * decay;
        self.lambda_sell = self.config.mu + (self.lambda_sell - self.config.mu) * decay;
    }

    /// Remove events older than max_history_secs.
    fn cleanup_old_events(&mut self, now: Instant) {
        let cutoff_duration = std::time::Duration::from_secs_f64(self.config.max_history_secs);

        while let Some(front) = self.events.front() {
            if now.duration_since(front.timestamp) > cutoff_duration {
                let removed = self.events.pop_front().unwrap();
                if removed.is_buy {
                    self.buy_volume -= removed.size;
                } else {
                    self.sell_volume -= removed.size;
                }
            } else {
                break;
            }
        }
    }

    /// Update without a trade (just decay).
    pub fn update(&mut self) {
        let now = Instant::now();
        self.update_intensities(now);
        self.cleanup_old_events(now);
        self.last_update = now;
    }

    /// Check if estimator is warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.trade_count >= self.config.min_trades
    }

    /// Get current buy intensity (trades per second).
    pub fn lambda_buy(&self) -> f64 {
        self.lambda_buy
    }

    /// Get current sell intensity (trades per second).
    pub fn lambda_sell(&self) -> f64 {
        self.lambda_sell
    }

    /// Get total intensity.
    pub fn lambda_total(&self) -> f64 {
        self.lambda_buy + self.lambda_sell
    }

    /// Get flow imbalance from intensities.
    ///
    /// Returns value in [-1, 1]:
    /// - Positive: More buy intensity
    /// - Negative: More sell intensity
    pub fn flow_imbalance(&self) -> f64 {
        let total = self.lambda_buy + self.lambda_sell;
        if total < 1e-9 {
            return 0.0;
        }
        (self.lambda_buy - self.lambda_sell) / total
    }

    /// Get the expected number of trades in the next N seconds.
    ///
    /// Integrates the expected intensity decay.
    pub fn expected_trades(&self, horizon_secs: f64) -> (f64, f64) {
        let beta = self.config.beta;
        let mu = self.config.mu;

        // For exponential decay from current level to baseline:
        // ∫₀^T [μ + (λ₀ - μ)e^(-βt)] dt = μT + (λ₀ - μ)(1 - e^(-βT))/β

        let decay_factor = (1.0 - (-beta * horizon_secs).exp()) / beta;

        let expected_buy = mu * horizon_secs + (self.lambda_buy - mu) * decay_factor;
        let expected_sell = mu * horizon_secs + (self.lambda_sell - mu) * decay_factor;

        (expected_buy.max(0.0), expected_sell.max(0.0))
    }

    /// Get the intensity percentile (how unusual is current activity).
    ///
    /// Returns value in [0, 1] where:
    /// - 0.5: Normal activity (at baseline)
    /// - 1.0: Very high activity
    /// - 0.0: Very low activity
    pub fn intensity_percentile(&self) -> f64 {
        let total = self.lambda_total();
        let baseline = 2.0 * self.config.mu;

        // Simple mapping: 1 at baseline, higher above, lower below
        // Use sigmoid-like mapping
        let ratio = total / baseline;
        1.0 / (1.0 + (-2.0 * (ratio - 1.0)).exp())
    }

    /// Get optimal quote duration based on expected flow.
    ///
    /// In high-intensity periods, quotes should be shorter-lived.
    /// In low-intensity periods, quotes can persist longer.
    pub fn optimal_quote_duration(&self) -> f64 {
        let intensity = self.lambda_total();
        let baseline = 2.0 * self.config.mu;

        // Base duration of 5 seconds, scaled by activity
        let base_duration = 5.0;
        let ratio = intensity / baseline;

        // Higher intensity -> shorter duration (but cap at 1 second minimum)
        (base_duration / ratio).clamp(1.0, 30.0)
    }

    /// Get position sizing factor based on flow uncertainty.
    ///
    /// In high-intensity periods with imbalanced flow, reduce size.
    pub fn position_sizing_factor(&self) -> f64 {
        let imbalance = self.flow_imbalance().abs();
        let intensity_ratio = self.lambda_total() / (2.0 * self.config.mu);

        // Reduce size when: high intensity AND imbalanced flow
        // Factor of 1.0 = normal, 0.5 = half size
        if intensity_ratio > 2.0 && imbalance > 0.3 {
            0.5
        } else if intensity_ratio > 1.5 || imbalance > 0.2 {
            0.75
        } else {
            1.0
        }
    }

    /// Get recent trade rate (trades per second in last window).
    pub fn recent_trade_rate(&self) -> f64 {
        let window_secs = self.config.max_history_secs.min(60.0);
        self.events.len() as f64 / window_secs
    }

    /// Get volume imbalance from recent history.
    pub fn volume_imbalance(&self) -> f64 {
        let total = self.buy_volume + self.sell_volume;
        if total < 1e-9 {
            return 0.0;
        }
        (self.buy_volume - self.sell_volume) / total
    }

    /// Get summary statistics.
    pub fn summary(&self) -> HawkesSummary {
        let (expected_buy, expected_sell) = self.expected_trades(60.0);

        HawkesSummary {
            is_warmed_up: self.is_warmed_up(),
            lambda_buy: self.lambda_buy,
            lambda_sell: self.lambda_sell,
            lambda_total: self.lambda_total(),
            flow_imbalance: self.flow_imbalance(),
            intensity_percentile: self.intensity_percentile(),
            trade_count: self.trade_count,
            recent_events: self.events.len(),
            expected_trades_1m: expected_buy + expected_sell,
            buy_volume: self.buy_volume,
            sell_volume: self.sell_volume,
        }
    }

    /// Reset the estimator.
    pub fn reset(&mut self) {
        self.events.clear();
        self.lambda_buy = self.config.mu;
        self.lambda_sell = self.config.mu;
        self.trade_count = 0;
        self.buy_volume = 0.0;
        self.sell_volume = 0.0;
        self.last_update = Instant::now();
    }
}

/// Summary of Hawkes order flow status.
#[derive(Debug, Clone)]
pub struct HawkesSummary {
    pub is_warmed_up: bool,
    pub lambda_buy: f64,
    pub lambda_sell: f64,
    pub lambda_total: f64,
    pub flow_imbalance: f64,
    pub intensity_percentile: f64,
    pub trade_count: usize,
    pub recent_events: usize,
    pub expected_trades_1m: f64,
    pub buy_volume: f64,
    pub sell_volume: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_default_config() {
        let config = HawkesConfig::default();
        assert_eq!(config.mu, 0.5);
        assert_eq!(config.alpha, 0.3);
    }

    #[test]
    fn test_new_estimator() {
        let estimator = HawkesOrderFlowEstimator::new(HawkesConfig::default());
        assert!(!estimator.is_warmed_up());
        assert!((estimator.lambda_buy() - 0.5).abs() < 1e-9); // At baseline
    }

    #[test]
    fn test_record_trade_buy() {
        let mut estimator = HawkesOrderFlowEstimator::new(HawkesConfig::default());

        estimator.record_trade(true, 1.0); // Buy trade

        // Buy intensity should increase
        assert!(estimator.lambda_buy() > 0.5);
        // Sell intensity should also increase (cross-excitation)
        assert!(estimator.lambda_sell() > 0.5);
        // But buy should increase more
        assert!(estimator.lambda_buy() > estimator.lambda_sell());
    }

    #[test]
    fn test_record_trade_sell() {
        let mut estimator = HawkesOrderFlowEstimator::new(HawkesConfig::default());

        estimator.record_trade(false, 1.0); // Sell trade

        // Sell intensity should increase more than buy
        assert!(estimator.lambda_sell() > estimator.lambda_buy());
    }

    #[test]
    fn test_flow_imbalance() {
        let mut estimator = HawkesOrderFlowEstimator::new(HawkesConfig::default());

        // Record several buy trades
        for _ in 0..5 {
            estimator.record_trade(true, 1.0);
        }

        // Should have positive imbalance (more buy intensity)
        assert!(estimator.flow_imbalance() > 0.0);
    }

    #[test]
    fn test_intensity_decay() {
        let config = HawkesConfig {
            beta: 10.0, // Very fast decay
            ..Default::default()
        };
        let mut estimator = HawkesOrderFlowEstimator::new(config);

        estimator.record_trade(true, 1.0);
        let initial = estimator.lambda_buy();

        // Wait a bit and update
        thread::sleep(Duration::from_millis(200));
        estimator.update();

        // Intensity should have decayed
        assert!(estimator.lambda_buy() < initial);
    }

    #[test]
    fn test_expected_trades() {
        let estimator = HawkesOrderFlowEstimator::new(HawkesConfig::default());

        let (buy, sell) = estimator.expected_trades(60.0);

        // At baseline (0.5 each), expect ~30 trades each in 60 seconds
        assert!((buy - 30.0).abs() < 5.0);
        assert!((sell - 30.0).abs() < 5.0);
    }

    #[test]
    fn test_warmup() {
        let config = HawkesConfig {
            min_trades: 5,
            ..Default::default()
        };
        let mut estimator = HawkesOrderFlowEstimator::new(config);

        assert!(!estimator.is_warmed_up());

        for i in 0..5 {
            estimator.record_trade(i % 2 == 0, 1.0);
        }

        assert!(estimator.is_warmed_up());
    }

    #[test]
    fn test_volume_tracking() {
        let mut estimator = HawkesOrderFlowEstimator::new(HawkesConfig::default());

        estimator.record_trade(true, 2.0);
        estimator.record_trade(false, 1.0);

        assert!((estimator.buy_volume - 2.0).abs() < 1e-9);
        assert!((estimator.sell_volume - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_position_sizing_factor() {
        let mut estimator = HawkesOrderFlowEstimator::new(HawkesConfig::default());

        // At baseline, should return 1.0
        let factor = estimator.position_sizing_factor();
        assert!((factor - 1.0).abs() < 0.01);

        // After many buy trades, should reduce
        for _ in 0..20 {
            estimator.record_trade(true, 1.0);
        }

        let factor_high = estimator.position_sizing_factor();
        assert!(factor_high < 1.0);
    }

    #[test]
    fn test_summary() {
        let mut estimator = HawkesOrderFlowEstimator::new(HawkesConfig::default());

        for i in 0..10 {
            estimator.record_trade(i % 2 == 0, 1.0);
        }

        let summary = estimator.summary();
        assert!(summary.is_warmed_up);
        assert_eq!(summary.trade_count, 10);
    }

    #[test]
    fn test_reset() {
        let mut estimator = HawkesOrderFlowEstimator::new(HawkesConfig::default());

        for i in 0..10 {
            estimator.record_trade(i % 2 == 0, 1.0);
        }

        estimator.reset();

        assert!(!estimator.is_warmed_up());
        assert_eq!(estimator.trade_count, 0);
        assert!(estimator.events.is_empty());
    }
}
