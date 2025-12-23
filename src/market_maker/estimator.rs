//! Parameter estimation for GLFT strategy from live market data.

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

/// Seconds per year (365.25 days)
const SECONDS_PER_YEAR: f64 = 365.25 * 24.0 * 60.0 * 60.0;

/// Configuration for parameter estimation.
#[derive(Debug, Clone)]
pub struct EstimatorConfig {
    /// Rolling window size in milliseconds for volatility calculation
    pub window_ms: u64,
    /// Minimum number of trades required before volatility estimate is valid
    pub min_trades: usize,
    /// Default sigma to use during warmup
    pub default_sigma: f64,
    /// Default kappa to use during warmup
    pub default_kappa: f64,
    /// Default tau to use during warmup (time horizon in years)
    pub default_tau: f64,
    /// Decay period in seconds for adaptive warmup threshold (0 = no decay)
    pub decay_secs: u64,
    /// Floor for adaptive warmup threshold (minimum trades required even after full decay)
    pub min_warmup_trades: usize,
}

impl Default for EstimatorConfig {
    fn default() -> Self {
        Self {
            window_ms: 300_000, // 5 minutes
            min_trades: 50,
            default_sigma: 0.5,
            default_kappa: 1.5,
            default_tau: 0.0001,   // ~1 hour in years
            decay_secs: 300,       // 5 minutes
            min_warmup_trades: 5,  // floor
        }
    }
}

/// Estimates GLFT parameters (σ and κ) from live market data.
#[derive(Debug)]
pub struct ParameterEstimator {
    /// Configuration
    config: EstimatorConfig,
    /// Recent trade prices for volatility calculation: (timestamp_ms, price)
    trade_prices: VecDeque<(u64, f64)>,
    /// Cached volatility estimate
    cached_sigma: f64,
    /// Cached order intensity estimate
    cached_kappa: f64,
    /// Whether we have enough data for valid estimates
    is_warmed_up: bool,
    /// Timestamp when estimator was created (for adaptive threshold)
    start_time_ms: u64,
}

impl ParameterEstimator {
    /// Create a new parameter estimator with the given config.
    pub fn new(config: EstimatorConfig) -> Self {
        let start_time_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            cached_sigma: config.default_sigma,
            cached_kappa: config.default_kappa,
            config,
            trade_prices: VecDeque::new(),
            is_warmed_up: false,
            start_time_ms,
        }
    }

    /// Calculate the effective minimum trades threshold based on elapsed time.
    /// Linearly decays from `min_trades` to `min_warmup_trades` over `decay_secs`.
    fn effective_min_trades(&self, current_time_ms: u64) -> usize {
        // If decay is disabled, use fixed threshold
        if self.config.decay_secs == 0 {
            return self.config.min_trades;
        }

        let elapsed_ms = current_time_ms.saturating_sub(self.start_time_ms);
        let elapsed_secs = elapsed_ms / 1000;

        // If past decay period, use floor
        if elapsed_secs >= self.config.decay_secs {
            return self.config.min_warmup_trades;
        }

        // Linear interpolation from min_trades down to min_warmup_trades
        let progress = elapsed_secs as f64 / self.config.decay_secs as f64;
        let range = self.config.min_trades.saturating_sub(self.config.min_warmup_trades);
        let reduction = (progress * range as f64) as usize;
        let threshold = self.config.min_trades.saturating_sub(reduction);

        threshold.max(self.config.min_warmup_trades)
    }

    /// Process a new trade and update volatility estimate.
    pub fn on_trade(&mut self, timestamp_ms: u64, price: f64) {
        // Add new trade
        self.trade_prices.push_back((timestamp_ms, price));

        // Remove old trades outside the window
        let cutoff = timestamp_ms.saturating_sub(self.config.window_ms);
        while let Some(&(ts, _)) = self.trade_prices.front() {
            if ts < cutoff {
                self.trade_prices.pop_front();
            } else {
                break;
            }
        }

        // Use adaptive threshold that decays over time
        let effective_min = self.effective_min_trades(timestamp_ms);

        // Update warmup status and volatility estimate based on current trade count
        if self.trade_prices.len() >= effective_min {
            self.cached_sigma = self.calculate_volatility();
            self.is_warmed_up = true;
        } else {
            // Reset warmup if trades expired below threshold
            self.is_warmed_up = false;
        }
    }

    /// Process L2 book update and estimate order flow intensity.
    ///
    /// levels: Vec of (price, size) tuples, sorted by distance from mid
    pub fn on_l2_book(&mut self, mid_price: f64, bids: &[(f64, f64)], asks: &[(f64, f64)]) {
        // Estimate kappa from book depth decay
        // κ measures how quickly fill probability decays with distance from mid
        // Higher κ = liquidity concentrated near mid = tighter optimal spread

        let bid_kappa = self.estimate_kappa_from_side(mid_price, bids, true);
        let ask_kappa = self.estimate_kappa_from_side(mid_price, asks, false);

        // Average of bid and ask side estimates
        if bid_kappa > 0.0 && ask_kappa > 0.0 {
            self.cached_kappa = (bid_kappa + ask_kappa) / 2.0;
        } else if bid_kappa > 0.0 {
            self.cached_kappa = bid_kappa;
        } else if ask_kappa > 0.0 {
            self.cached_kappa = ask_kappa;
        }
        // Otherwise keep cached value
    }

    /// Estimate kappa from one side of the book.
    fn estimate_kappa_from_side(&self, mid_price: f64, levels: &[(f64, f64)], is_bid: bool) -> f64 {
        if levels.len() < 3 {
            return 0.0;
        }

        // Calculate cumulative size and distance from mid for each level
        let mut cumulative_size = 0.0;
        let mut points: Vec<(f64, f64)> = Vec::new(); // (distance, ln_cumulative_size)

        for &(price, size) in levels.iter().take(10) {
            // Use top 10 levels
            cumulative_size += size;
            let distance = if is_bid {
                (mid_price - price) / mid_price
            } else {
                (price - mid_price) / mid_price
            };

            if distance > 0.0 && cumulative_size > 0.0 {
                points.push((distance, cumulative_size.ln()));
            }
        }

        if points.len() < 2 {
            return 0.0;
        }

        // Linear regression to find slope of ln(cumulative_size) vs distance
        // κ ≈ -1 / slope (higher slope = faster decay = higher kappa)
        let n = points.len() as f64;
        let sum_x: f64 = points.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = points.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
        let sum_xx: f64 = points.iter().map(|(x, _)| x * x).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);

        if slope.abs() > 1e-10 {
            // κ = 1 / |slope| with reasonable bounds
            (1.0 / slope.abs()).clamp(0.1, 100.0)
        } else {
            0.0
        }
    }

    /// Calculate volatility from recent trades using log returns.
    fn calculate_volatility(&self) -> f64 {
        if self.trade_prices.len() < 2 {
            return self.config.default_sigma;
        }

        // Calculate log returns
        let prices: Vec<f64> = self.trade_prices.iter().map(|(_, p)| *p).collect();
        let log_returns: Vec<f64> = prices
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        if log_returns.is_empty() {
            return self.config.default_sigma;
        }

        // Calculate standard deviation of log returns
        let n = log_returns.len() as f64;
        let mean = log_returns.iter().sum::<f64>() / n;
        let variance = log_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        // Annualize: multiply by sqrt(trades_per_day)
        // Estimate trades per day from our sample
        if let (Some(&(first_ts, _)), Some(&(last_ts, _))) =
            (self.trade_prices.front(), self.trade_prices.back())
        {
            let duration_ms = last_ts.saturating_sub(first_ts).max(1);
            let trades_per_ms = self.trade_prices.len() as f64 / duration_ms as f64;
            let trades_per_day = trades_per_ms * 86_400_000.0;

            // Annualized volatility (assuming 365 trading days)
            let annualized = std_dev * trades_per_day.sqrt();

            // Clamp to reasonable bounds
            annualized.clamp(0.01, 5.0)
        } else {
            self.config.default_sigma
        }
    }

    /// Get current volatility estimate (σ).
    pub fn sigma(&self) -> f64 {
        self.cached_sigma
    }

    /// Get current order flow intensity estimate (κ).
    pub fn kappa(&self) -> f64 {
        self.cached_kappa
    }

    /// Estimate tau (time horizon) from observed trade rate.
    /// Returns time in "years" representing expected holding period.
    /// Faster markets → smaller τ → less inventory skew.
    pub fn tau(&self) -> f64 {
        let trade_count = self.trade_prices.len();
        if trade_count < 2 {
            return self.config.default_tau;
        }

        // Calculate average time between trades
        if let (Some(&(first_ts, _)), Some(&(last_ts, _))) =
            (self.trade_prices.front(), self.trade_prices.back())
        {
            let duration_ms = last_ts.saturating_sub(first_ts).max(1);
            let duration_secs = duration_ms as f64 / 1000.0;
            let avg_interval_secs = duration_secs / (trade_count - 1) as f64;

            // Expected holding time = avg interval * multiplier (expect fill in ~10 trades)
            let fill_multiplier = 10.0;
            let expected_holding_secs = avg_interval_secs * fill_multiplier;

            // Convert to years, clamp to reasonable range
            let tau = expected_holding_secs / SECONDS_PER_YEAR;
            tau.clamp(1e-8, 0.01) // Min ~0.3s, Max ~3.6 days
        } else {
            self.config.default_tau
        }
    }

    /// Check if estimator has collected enough data.
    pub fn is_warmed_up(&self) -> bool {
        self.is_warmed_up
    }

    /// Get current warmup progress (trades collected / effective min required).
    /// The effective minimum threshold decays over time for low-activity markets.
    pub fn warmup_progress(&self) -> (usize, usize) {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let effective_min = self.effective_min_trades(now_ms);
        (self.trade_prices.len(), effective_min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warmup_progress() {
        // Disable decay to test fixed threshold behavior
        let config = EstimatorConfig {
            min_trades: 10,
            decay_secs: 0, // disable adaptive decay
            ..Default::default()
        };
        let mut estimator = ParameterEstimator::new(config);

        assert!(!estimator.is_warmed_up());

        // Get current time to use realistic timestamps
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Add trades with realistic timestamps
        for i in 0..10 {
            estimator.on_trade(now + i as u64 * 1000, 100.0 + i as f64 * 0.1);
        }

        assert!(estimator.is_warmed_up());
    }

    #[test]
    fn test_volatility_estimation() {
        let config = EstimatorConfig {
            min_trades: 5,
            window_ms: 60_000,
            decay_secs: 0, // disable adaptive decay
            ..Default::default()
        };
        let mut estimator = ParameterEstimator::new(config);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Add trades with increasing prices (some volatility)
        let prices = [100.0, 100.5, 99.8, 100.2, 100.1, 99.9, 100.3];
        for (i, price) in prices.iter().enumerate() {
            estimator.on_trade(now + i as u64 * 1000, *price);
        }

        let sigma = estimator.sigma();
        assert!(sigma > 0.0);
        assert!(sigma < 5.0); // Reasonable bounds
    }

    #[test]
    fn test_window_expiry() {
        let config = EstimatorConfig {
            min_trades: 3,
            window_ms: 5000, // 5 second window
            decay_secs: 0,   // disable adaptive decay
            ..Default::default()
        };
        let mut estimator = ParameterEstimator::new(config);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Add trades
        estimator.on_trade(now, 100.0);
        estimator.on_trade(now + 1000, 100.1);
        estimator.on_trade(now + 2000, 100.2);

        assert!(estimator.is_warmed_up());

        // Add new trade that should expire old ones (10 seconds later)
        estimator.on_trade(now + 10000, 100.5);

        // Only the new trade should remain (old ones expired)
        assert_eq!(estimator.trade_prices.len(), 1);
        assert!(!estimator.is_warmed_up()); // Back to not warmed up
    }

    #[test]
    fn test_adaptive_threshold() {
        // Test that threshold decays over time
        let config = EstimatorConfig {
            min_trades: 50,
            min_warmup_trades: 5,
            decay_secs: 300, // 5 minute decay
            window_ms: 600_000,
            ..Default::default()
        };
        let mut estimator = ParameterEstimator::new(config);

        // At t=0, effective threshold should be 50
        let now = estimator.start_time_ms;
        assert_eq!(estimator.effective_min_trades(now), 50);

        // At t=150s (halfway), effective threshold should be ~27
        let halfway = now + 150_000;
        let halfway_threshold = estimator.effective_min_trades(halfway);
        assert!(halfway_threshold > 20 && halfway_threshold < 35);

        // At t=300s (full decay), effective threshold should be 5
        let full_decay = now + 300_000;
        assert_eq!(estimator.effective_min_trades(full_decay), 5);

        // At t=600s (past decay), effective threshold should still be 5
        let past_decay = now + 600_000;
        assert_eq!(estimator.effective_min_trades(past_decay), 5);

        // Test that warmup triggers with fewer trades after decay
        // Add only 10 trades (not enough at t=0, but enough after some decay)
        for i in 0..10 {
            estimator.on_trade(full_decay + i as u64 * 100, 100.0 + i as f64 * 0.1);
        }
        assert!(estimator.is_warmed_up()); // Should be warmed up with 10 trades after full decay
    }
}
