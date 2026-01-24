//! Lag Analysis for Cross-Exchange Signals
//!
//! Implements time-lag detection for cross-exchange price discovery.
//! The primary use case is Binance → Hyperliquid lead-lag estimation.
//!
//! # Key Features
//!
//! - **Optimal lag detection**: Finds the lag that maximizes mutual information
//! - **Cross-correlation function**: Traditional CCF for comparison
//! - **Regime-aware**: Lag varies with volatility and market conditions
//! - **Decay tracking**: Monitors lead-lag decay as arbitrageurs compete
//!
//! # Usage
//!
//! ```ignore
//! let mut analyzer = LagAnalyzer::new(LagAnalyzerConfig::default());
//!
//! // Feed in signal-target pairs with timestamps
//! analyzer.add_observation(binance_mid_ts, binance_mid, hyperliquid_mid_ts, hyperliquid_mid);
//!
//! // Find optimal lag
//! if let Some((lag_ms, mi)) = analyzer.optimal_lag() {
//!     println!("Binance leads by {}ms with MI={:.3} bits", lag_ms, mi);
//! }
//! ```

use std::collections::VecDeque;

use super::mutual_info::MutualInfoEstimator;

/// Configuration for lag analysis.
#[derive(Debug, Clone)]
pub struct LagAnalyzerConfig {
    /// Candidate lags to test (in milliseconds)
    pub candidate_lags_ms: Vec<i64>,
    /// Buffer capacity (number of observations to retain)
    pub buffer_capacity: usize,
    /// Minimum observations before computing lag
    pub min_observations: usize,
    /// k for k-NN MI estimation
    pub mi_k: usize,
    /// Interpolation tolerance (ms) - observations within this window are matched
    pub interpolation_tolerance_ms: i64,
}

impl Default for LagAnalyzerConfig {
    fn default() -> Self {
        Self {
            // Test lags from -500ms to +500ms in 50ms increments
            // Negative = signal leads target, Positive = signal lags target
            candidate_lags_ms: vec![
                -500, -400, -300, -250, -200, -150, -100, -75, -50, -25, 0, 25, 50, 75, 100, 150,
                200, 250, 300, 400, 500,
            ],
            buffer_capacity: 2000,
            min_observations: 100,
            mi_k: 5,
            interpolation_tolerance_ms: 100,
        }
    }
}

/// Timestamped observation.
#[derive(Debug, Clone, Copy)]
struct TimedValue {
    timestamp_ms: i64,
    value: f64,
}

/// Lag analyzer for cross-exchange signals.
///
/// Maintains buffered observations of signal and target time series,
/// then computes optimal lag using mutual information.
#[derive(Debug, Clone)]
pub struct LagAnalyzer {
    config: LagAnalyzerConfig,
    /// Signal buffer (e.g., Binance mid prices)
    signal_buffer: VecDeque<TimedValue>,
    /// Target buffer (e.g., Hyperliquid mid prices)
    target_buffer: VecDeque<TimedValue>,
    /// MI estimator
    mi_estimator: MutualInfoEstimator,
    /// Cached optimal lag (updated periodically)
    cached_optimal_lag: Option<(i64, f64)>,
    /// Observations since last cache update
    observations_since_update: usize,
    /// Update frequency (recompute optimal lag every N observations)
    update_frequency: usize,
}

impl LagAnalyzer {
    /// Create a new lag analyzer with default config.
    pub fn new(config: LagAnalyzerConfig) -> Self {
        let mi_estimator = MutualInfoEstimator::new(config.mi_k);
        Self {
            config,
            signal_buffer: VecDeque::with_capacity(2000),
            target_buffer: VecDeque::with_capacity(2000),
            mi_estimator,
            cached_optimal_lag: None,
            observations_since_update: 0,
            update_frequency: 50, // Recompute every 50 observations
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(LagAnalyzerConfig::default())
    }

    /// Add a signal observation (e.g., Binance mid price).
    pub fn add_signal(&mut self, timestamp_ms: i64, value: f64) {
        if !value.is_finite() {
            return;
        }

        self.signal_buffer.push_back(TimedValue {
            timestamp_ms,
            value,
        });

        // Trim buffer if too large
        while self.signal_buffer.len() > self.config.buffer_capacity {
            self.signal_buffer.pop_front();
        }
    }

    /// Add a target observation (e.g., Hyperliquid mid price).
    pub fn add_target(&mut self, timestamp_ms: i64, value: f64) {
        if !value.is_finite() {
            return;
        }

        self.target_buffer.push_back(TimedValue {
            timestamp_ms,
            value,
        });

        // Trim buffer if too large
        while self.target_buffer.len() > self.config.buffer_capacity {
            self.target_buffer.pop_front();
        }

        // Periodically update cached lag
        self.observations_since_update += 1;
        if self.observations_since_update >= self.update_frequency {
            self.cached_optimal_lag = self.compute_optimal_lag();
            self.observations_since_update = 0;
        }
    }

    /// Add paired signal-target observation.
    ///
    /// Convenience method when both arrive together.
    pub fn add_observation(
        &mut self,
        signal_ts: i64,
        signal_value: f64,
        target_ts: i64,
        target_value: f64,
    ) {
        self.add_signal(signal_ts, signal_value);
        self.add_target(target_ts, target_value);
    }

    /// Get the optimal lag (cached, updated periodically).
    ///
    /// Returns (lag_ms, mutual_information_bits) or None if insufficient data.
    ///
    /// Negative lag means signal leads target (the expected case for Binance → Hyperliquid).
    pub fn optimal_lag(&self) -> Option<(i64, f64)> {
        self.cached_optimal_lag
    }

    /// Compute optimal lag (expensive, called periodically).
    fn compute_optimal_lag(&self) -> Option<(i64, f64)> {
        if self.signal_buffer.len() < self.config.min_observations
            || self.target_buffer.len() < self.config.min_observations
        {
            return None;
        }

        let mut best_lag = 0i64;
        let mut best_mi = 0.0f64;

        for &lag_ms in &self.config.candidate_lags_ms {
            let mi = self.mi_at_lag(lag_ms);
            if mi > best_mi {
                best_mi = mi;
                best_lag = lag_ms;
            }
        }

        if best_mi > 0.0 {
            Some((best_lag, best_mi))
        } else {
            None
        }
    }

    /// Compute MI at a specific lag.
    ///
    /// # Arguments
    /// * `lag_ms` - Lag in milliseconds. Negative = signal leads target.
    fn mi_at_lag(&self, lag_ms: i64) -> f64 {
        // Build paired observations at this lag
        let (x, y) = self.build_lagged_pairs(lag_ms);

        if x.len() < self.config.min_observations / 2 {
            return 0.0;
        }

        self.mi_estimator.estimate_bits(&x, &y)
    }

    /// Build paired observations at a specific lag using nearest-neighbor interpolation.
    fn build_lagged_pairs(&self, lag_ms: i64) -> (Vec<f64>, Vec<f64>) {
        let mut x = Vec::new();
        let mut y = Vec::new();

        let tolerance = self.config.interpolation_tolerance_ms;

        // For each target observation, find the signal observation at (target_time - lag)
        for target in &self.target_buffer {
            let lagged_time = target.timestamp_ms - lag_ms;

            // Binary search for nearest signal observation
            if let Some(signal_value) = self.interpolate_signal(lagged_time, tolerance) {
                // Use returns rather than levels (more stationary)
                x.push(signal_value);
                y.push(target.value);
            }
        }

        // Convert to returns for stationarity
        if x.len() > 1 {
            let x_returns: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
            let y_returns: Vec<f64> = y.windows(2).map(|w| w[1] - w[0]).collect();
            (x_returns, y_returns)
        } else {
            (x, y)
        }
    }

    /// Interpolate signal value at a specific timestamp.
    fn interpolate_signal(&self, target_ts: i64, tolerance: i64) -> Option<f64> {
        // Find the closest signal observation within tolerance
        let mut best_value = None;
        let mut best_distance = i64::MAX;

        for signal in &self.signal_buffer {
            let distance = (signal.timestamp_ms - target_ts).abs();
            if distance < best_distance && distance <= tolerance {
                best_distance = distance;
                best_value = Some(signal.value);
            }
        }

        best_value
    }

    /// Compute cross-correlation function at multiple lags.
    ///
    /// Returns vector of (lag_ms, correlation) pairs.
    pub fn ccf(&self, max_lag_ms: i64, step_ms: i64) -> Vec<(i64, f64)> {
        let mut result = Vec::new();

        let mut lag = -max_lag_ms;
        while lag <= max_lag_ms {
            let (x, y) = self.build_lagged_pairs(lag);
            if x.len() >= 10 {
                let corr = pearson_correlation(&x, &y);
                result.push((lag, corr));
            }
            lag += step_ms;
        }

        result
    }

    /// Get MI at each candidate lag for analysis.
    pub fn mi_by_lag(&self) -> Vec<(i64, f64)> {
        self.config
            .candidate_lags_ms
            .iter()
            .map(|&lag| (lag, self.mi_at_lag(lag)))
            .collect()
    }

    /// Check if analyzer has enough data.
    pub fn is_ready(&self) -> bool {
        self.signal_buffer.len() >= self.config.min_observations
            && self.target_buffer.len() >= self.config.min_observations
    }

    /// Get observation counts.
    pub fn observation_counts(&self) -> (usize, usize) {
        (self.signal_buffer.len(), self.target_buffer.len())
    }

    /// Get the current best lag in milliseconds (0 if unknown).
    pub fn best_lag_ms(&self) -> i64 {
        self.cached_optimal_lag.map(|(lag, _)| lag).unwrap_or(0)
    }

    /// Get the MI at the best lag (0 if unknown).
    pub fn best_lag_mi(&self) -> f64 {
        self.cached_optimal_lag.map(|(_, mi)| mi).unwrap_or(0.0)
    }

    /// Clear all buffers and reset state.
    pub fn reset(&mut self) {
        self.signal_buffer.clear();
        self.target_buffer.clear();
        self.cached_optimal_lag = None;
        self.observations_since_update = 0;
    }
}

/// Compute Pearson correlation coefficient.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x > 0.0 && var_y > 0.0 {
        cov / (var_x.sqrt() * var_y.sqrt())
    } else {
        0.0
    }
}

/// Lag decay tracker for monitoring lead-lag edge decay.
///
/// Tracks how the optimal lag and MI change over time to detect
/// when arbitrageurs are closing the gap.
#[derive(Debug, Clone)]
pub struct LagDecayTracker {
    /// Historical (timestamp, lag_ms, mi) observations
    history: VecDeque<(i64, i64, f64)>,
    /// Maximum history length
    max_history: usize,
}

impl LagDecayTracker {
    /// Create a new decay tracker.
    pub fn new(max_history: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    /// Record an optimal lag observation.
    pub fn record(&mut self, timestamp_ms: i64, lag_ms: i64, mi: f64) {
        self.history.push_back((timestamp_ms, lag_ms, mi));
        while self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Compute MI decay trend (slope of MI over time).
    ///
    /// Negative value means MI is decaying (edge eroding).
    pub fn mi_trend(&self) -> f64 {
        if self.history.len() < 10 {
            return 0.0;
        }

        // Simple linear regression of MI on time
        let n = self.history.len() as f64;
        let first_ts = self.history.front().map(|(t, _, _)| *t).unwrap_or(0);

        let mut sum_t = 0.0;
        let mut sum_mi = 0.0;
        let mut sum_t2 = 0.0;
        let mut sum_t_mi = 0.0;

        for (ts, _, mi) in &self.history {
            let t = (*ts - first_ts) as f64 / 1000.0 / 3600.0; // Hours since start
            sum_t += t;
            sum_mi += mi;
            sum_t2 += t * t;
            sum_t_mi += t * mi;
        }

        let denom = n * sum_t2 - sum_t * sum_t;
        if denom.abs() < 1e-10 {
            return 0.0;
        }

        (n * sum_t_mi - sum_t * sum_mi) / denom
    }

    /// Compute lag drift trend (is lag getting shorter?).
    ///
    /// Negative value means lag is shrinking (arbitrageurs closing gap).
    pub fn lag_trend(&self) -> f64 {
        if self.history.len() < 10 {
            return 0.0;
        }

        let n = self.history.len() as f64;
        let first_ts = self.history.front().map(|(t, _, _)| *t).unwrap_or(0);

        let mut sum_t = 0.0;
        let mut sum_lag = 0.0;
        let mut sum_t2 = 0.0;
        let mut sum_t_lag = 0.0;

        for (ts, lag, _) in &self.history {
            let t = (*ts - first_ts) as f64 / 1000.0 / 3600.0; // Hours since start
            sum_t += t;
            sum_lag += *lag as f64;
            sum_t2 += t * t;
            sum_t_lag += t * (*lag as f64);
        }

        let denom = n * sum_t2 - sum_t * sum_t;
        if denom.abs() < 1e-10 {
            return 0.0;
        }

        (n * sum_t_lag - sum_t * sum_lag) / denom
    }

    /// Estimate MI half-life in hours.
    ///
    /// Returns None if can't be estimated or MI is not decaying.
    pub fn mi_half_life_hours(&self) -> Option<f64> {
        let trend = self.mi_trend();
        if trend >= 0.0 {
            return None; // Not decaying
        }

        // Current average MI
        let avg_mi: f64 =
            self.history.iter().map(|(_, _, mi)| mi).sum::<f64>() / self.history.len() as f64;

        if avg_mi <= 0.0 {
            return None;
        }

        // Time to decay to half: half_life = avg_mi / (2 * |trend|)
        Some(avg_mi / (2.0 * trend.abs()))
    }

    /// Check if edge is stale (MI decayed significantly).
    pub fn is_edge_stale(&self, min_mi: f64) -> bool {
        if let Some((_, _, mi)) = self.history.back() {
            *mi < min_mi
        } else {
            true // No data = stale
        }
    }

    /// Get history length.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lag_analyzer_basic() {
        let mut analyzer = LagAnalyzer::default_config();

        // Add some synthetic data where signal leads target by 100ms
        let mut ts = 0i64;
        for i in 0..200 {
            let signal_value = (i as f64 * 0.01).sin();
            let target_value = signal_value; // Will be matched with lag

            analyzer.add_signal(ts, signal_value);
            analyzer.add_target(ts + 100, target_value); // Target lags by 100ms

            ts += 50; // 50ms between observations
        }

        assert!(analyzer.is_ready());
        let (signal_count, target_count) = analyzer.observation_counts();
        assert!(signal_count >= 100);
        assert!(target_count >= 100);
    }

    #[test]
    fn test_ccf() {
        let mut analyzer = LagAnalyzer::default_config();

        // Add correlated data
        for i in 0..300 {
            let ts = i * 50;
            let value = (i as f64 * 0.05).sin();
            analyzer.add_signal(ts, value);
            analyzer.add_target(ts + 50, value * 0.9 + 0.1 * (i as f64 * 0.1).cos());
        }

        let ccf = analyzer.ccf(200, 50);
        assert!(!ccf.is_empty());

        // All correlations should be bounded
        for (_, corr) in &ccf {
            assert!(corr.abs() <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn test_pearson_correlation() {
        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = pearson_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-6);

        // Perfect negative correlation
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let corr_neg = pearson_correlation(&x, &y_neg);
        assert!((corr_neg + 1.0).abs() < 1e-6);

        // Uncorrelated (approximately)
        let x_unc = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_unc = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let corr_unc = pearson_correlation(&x_unc, &y_unc);
        assert!(corr_unc.abs() < 0.5);
    }

    #[test]
    fn test_lag_decay_tracker() {
        let mut tracker = LagDecayTracker::new(100);

        // Add decaying MI observations
        for i in 0..50 {
            let ts = i * 3600 * 1000; // 1 hour apart
            let mi = 0.1 - (i as f64) * 0.001; // Decaying MI
            tracker.record(ts, -100, mi);
        }

        let trend = tracker.mi_trend();
        assert!(trend < 0.0, "MI trend should be negative (decaying)");

        let half_life = tracker.mi_half_life_hours();
        assert!(half_life.is_some(), "Should compute half-life");
    }
}
