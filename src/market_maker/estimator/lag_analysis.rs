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

/// Timestamp range as (first, last) for a buffer. Used for diagnostics.
pub type TimestampRange = (Option<i64>, Option<i64>);

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

    /// Whether to use MI significance test (shuffle-based null distribution).
    /// When true, MI must exceed the 95th percentile of shuffled null to be actionable.
    /// This filters spurious MI from the KSG estimator's positive bias with small samples.
    pub use_mi_significance_test: bool,
    /// Number of shuffles for null distribution estimation.
    pub n_shuffles: usize,
    /// Significance level (percentile of null distribution MI must exceed).
    /// 0.95 means MI must be above the 95th percentile of the shuffled null.
    pub significance_level: f64,
}

impl Default for LagAnalyzerConfig {
    fn default() -> Self {
        Self {
            // Test lags from -500ms to +500ms in 50ms increments
            // Positive = signal leads target (look at past signal), Negative = target leads signal
            candidate_lags_ms: vec![
                -500, -400, -300, -250, -200, -150, -100, -75, -50, -25, 0, 25, 50, 75, 100, 150,
                200, 250, 300, 400, 500,
            ],
            buffer_capacity: 2000,
            min_observations: 100,
            mi_k: 5,
            interpolation_tolerance_ms: 100,
            use_mi_significance_test: true,
            n_shuffles: 50,
            significance_level: 0.95,
        }
    }
}

/// Timestamped observation.
#[derive(Debug, Clone, Copy)]
struct TimedValue {
    timestamp_ms: i64,
    value: f64,
}

/// Cached null MI distribution from shuffle test.
///
/// Used to determine whether observed MI is statistically significant
/// or just an artifact of the KSG estimator's positive bias.
#[derive(Debug, Clone)]
struct NullMIDistribution {
    /// 95th percentile of null MI values.
    p95: f64,
    /// 99th percentile of null MI values.
    p99: f64,
    /// Mean of null MI values.
    mean: f64,
    /// Number of shuffles used.
    _n_shuffles: usize,
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
    /// Cached null MI distribution (from shuffle test)
    null_mi_dist: Option<NullMIDistribution>,
    /// Total target observations received (monotonically increasing)
    total_target_observations: usize,
    /// Next observation count at which to recompute null distribution.
    /// Follows logarithmic schedule: 200, 400, 800, 1600, ...
    /// Since null MI is a property of sample size (not data content),
    /// there's no need to recompute at fixed intervals.
    next_null_update_at: usize,
}

impl LagAnalyzer {
    /// Create a new lag analyzer with default config.
    pub fn new(config: LagAnalyzerConfig) -> Self {
        let mi_estimator = MutualInfoEstimator::new(config.mi_k);
        Self {
            signal_buffer: VecDeque::with_capacity(2000),
            target_buffer: VecDeque::with_capacity(2000),
            mi_estimator,
            cached_optimal_lag: None,
            observations_since_update: 0,
            update_frequency: 50, // Recompute every 50 observations
            null_mi_dist: None,
            total_target_observations: 0,
            next_null_update_at: 200,
            config,
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

        // Recompute null MI distribution on logarithmic schedule (200, 400, 800, 1600...)
        // Null MI depends on sample size, not data content, so exponential spacing is optimal.
        self.total_target_observations += 1;
        if self.config.use_mi_significance_test
            && self.total_target_observations >= self.next_null_update_at
            && self.target_buffer.len() >= self.config.min_observations
            && self.signal_buffer.len() >= self.config.min_observations
        {
            self.null_mi_dist = self.compute_null_distribution(self.config.n_shuffles);
            // Double the interval for next recomputation (logarithmic schedule)
            self.next_null_update_at = self.total_target_observations.saturating_mul(2);
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
    /// Positive lag means signal leads target (the expected case for Binance → Hyperliquid).
    pub fn optimal_lag(&self) -> Option<(i64, f64)> {
        self.cached_optimal_lag
    }

    /// Compute optimal lag (expensive, called periodically).
    ///
    /// When MI significance testing is enabled, the best MI must exceed
    /// the null distribution's significance percentile (default p95).
    /// This filters spurious MI from the KSG estimator's positive bias.
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

        if best_mi <= 0.0 {
            return None;
        }

        // Apply MI significance test: check if best MI exceeds null distribution
        if self.config.use_mi_significance_test {
            if let Some(ref null_dist) = self.null_mi_dist {
                // Use the configured significance level to pick threshold
                let threshold = if self.config.significance_level >= 0.99 {
                    null_dist.p99
                } else {
                    null_dist.p95
                };
                if best_mi <= threshold {
                    return None; // Not significant — MI is within noise range
                }
            }
            // If null_dist hasn't been computed yet, allow the signal through
            // (it will be filtered once enough data accumulates)
        }

        Some((best_lag, best_mi))
    }

    /// Compute null MI distribution by shuffling target values.
    ///
    /// Uses Fisher-Yates shuffle with a fast xorshift PRNG to generate
    /// a null distribution of MI values where any mutual information
    /// is due to estimator bias rather than genuine dependence.
    fn compute_null_distribution(&self, n_shuffles: usize) -> Option<NullMIDistribution> {
        // Build lagged pairs at lag=0 for the null test
        let (x, y) = self.build_lagged_pairs(0);
        if x.len() < self.config.min_observations / 2 {
            return None;
        }

        let mut null_mis = Vec::with_capacity(n_shuffles);
        let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_1234; // xorshift seed

        for _ in 0..n_shuffles {
            // Shuffle y using Fisher-Yates with xorshift64
            let mut y_shuffled = y.clone();
            for i in (1..y_shuffled.len()).rev() {
                // xorshift64 step
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let j = (rng_state as usize) % (i + 1);
                y_shuffled.swap(i, j);
            }

            let mi = self.mi_estimator.estimate_bits(&x, &y_shuffled);
            null_mis.push(mi);
        }

        // Sort for percentile computation
        null_mis.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = null_mis.len();
        if n == 0 {
            return None;
        }

        let mean = null_mis.iter().sum::<f64>() / n as f64;
        let p95_idx = ((n as f64 - 1.0) * 0.95) as usize;
        let p99_idx = ((n as f64 - 1.0) * 0.99) as usize;

        Some(NullMIDistribution {
            p95: null_mis[p95_idx.min(n - 1)],
            p99: null_mis[p99_idx.min(n - 1)],
            mean,
            _n_shuffles: n_shuffles,
        })
    }

    /// Compute MI at a specific lag.
    ///
    /// # Arguments
    /// * `lag_ms` - Lag in milliseconds. Positive = signal leads target.
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

    /// Get sample timestamps for debugging (first, last from each buffer).
    pub fn sample_timestamps(&self) -> (TimestampRange, TimestampRange) {
        let signal_first = self.signal_buffer.front().map(|o| o.timestamp_ms);
        let signal_last = self.signal_buffer.back().map(|o| o.timestamp_ms);
        let target_first = self.target_buffer.front().map(|o| o.timestamp_ms);
        let target_last = self.target_buffer.back().map(|o| o.timestamp_ms);
        ((signal_first, signal_last), (target_first, target_last))
    }

    /// Check whether the last computed optimal lag was significant against the null.
    ///
    /// Returns `true` if significance testing is disabled or if MI exceeds null p95.
    /// Returns `false` if MI failed the significance test (lag was filtered).
    pub fn is_lag_significant(&self) -> bool {
        if !self.config.use_mi_significance_test {
            return true;
        }
        // If we have a cached lag, it passed the test
        self.cached_optimal_lag.is_some()
    }

    /// Get the null distribution p95 threshold (for diagnostics).
    ///
    /// Returns 0.0 if null distribution hasn't been computed yet.
    pub fn null_mi_p95(&self) -> f64 {
        self.null_mi_dist.as_ref().map(|d| d.p95).unwrap_or(0.0)
    }

    /// Get the null distribution mean (for diagnostics).
    pub fn null_mi_mean(&self) -> f64 {
        self.null_mi_dist.as_ref().map(|d| d.mean).unwrap_or(0.0)
    }

    /// Clear all buffers and reset state.
    pub fn reset(&mut self) {
        self.signal_buffer.clear();
        self.target_buffer.clear();
        self.cached_optimal_lag = None;
        self.observations_since_update = 0;
        self.null_mi_dist = None;
        self.total_target_observations = 0;
        self.next_null_update_at = 200;
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

/// Lead-lag stability gate.
///
/// Requires lag to be stable and causal before allowing signals.
/// This addresses the oscillating lag problem (-500ms, +150ms, -300ms)
/// by requiring consistent positive lag over a window.
///
/// # Algorithm
/// 1. Track lag history over a sliding window
/// 2. Compute mean and variance of lag
/// 3. Require: variance < threshold AND mean > min_lag_ms
/// 4. Also track MI confidence decay
#[derive(Debug, Clone)]
pub struct LeadLagStabilityGate {
    /// Recent lag observations
    lag_history: VecDeque<i64>,
    /// Recent MI observations
    mi_history: VecDeque<f64>,
    /// Window size for stability check
    window_size: usize,
    /// Maximum allowed lag variance (ms²)
    max_variance_ms2: f64,
    /// Minimum mean lag required (ms) - must be positive/causal
    min_mean_lag_ms: f64,
    /// Minimum MI threshold
    min_mi: f64,
    /// Number of consecutive stable observations required
    min_stable_count: usize,
    /// Current consecutive stable count
    stable_count: usize,
    /// Total observations
    total_observations: u64,
}

impl Default for LeadLagStabilityGate {
    fn default() -> Self {
        Self {
            lag_history: VecDeque::with_capacity(50),
            mi_history: VecDeque::with_capacity(50),
            window_size: 20,
            max_variance_ms2: 10000.0, // 100ms std dev max
            min_mean_lag_ms: 25.0,     // At least 25ms lead
            min_mi: 0.02,              // Minimum 0.02 bits MI
            min_stable_count: 10,      // 10 consecutive stable readings
            stable_count: 0,
            total_observations: 0,
        }
    }
}

impl LeadLagStabilityGate {
    /// Create with custom parameters.
    pub fn new(
        window_size: usize,
        max_variance_ms2: f64,
        min_mean_lag_ms: f64,
        min_mi: f64,
        min_stable_count: usize,
    ) -> Self {
        Self {
            lag_history: VecDeque::with_capacity(window_size),
            mi_history: VecDeque::with_capacity(window_size),
            window_size,
            max_variance_ms2,
            min_mean_lag_ms,
            min_mi,
            min_stable_count,
            stable_count: 0,
            total_observations: 0,
        }
    }

    /// Record a new lag observation.
    pub fn record(&mut self, lag_ms: i64, mi: f64) {
        self.lag_history.push_back(lag_ms);
        self.mi_history.push_back(mi);
        self.total_observations += 1;

        while self.lag_history.len() > self.window_size {
            self.lag_history.pop_front();
        }
        while self.mi_history.len() > self.window_size {
            self.mi_history.pop_front();
        }

        // Update stable count
        if self.check_instant_stability() {
            self.stable_count += 1;
        } else {
            self.stable_count = 0;
        }
    }

    /// Check if lag is currently stable (instant check without consecutive requirement).
    fn check_instant_stability(&self) -> bool {
        if self.lag_history.len() < 5 {
            return false;
        }

        let (mean, var) = self.lag_stats();
        let avg_mi = self.mi_stats().0;

        mean > self.min_mean_lag_ms && var < self.max_variance_ms2 && avg_mi > self.min_mi
    }

    /// Check if lag is stable (requires consecutive stable observations).
    pub fn is_stable(&self) -> bool {
        self.stable_count >= self.min_stable_count
    }

    /// Compute lag statistics (mean, variance).
    fn lag_stats(&self) -> (f64, f64) {
        if self.lag_history.is_empty() {
            return (0.0, f64::MAX);
        }

        let n = self.lag_history.len() as f64;
        let mean = self.lag_history.iter().map(|&x| x as f64).sum::<f64>() / n;
        let variance = if n > 1.0 {
            self.lag_history
                .iter()
                .map(|&x| {
                    let diff = x as f64 - mean;
                    diff * diff
                })
                .sum::<f64>()
                / (n - 1.0)
        } else {
            f64::MAX
        };

        (mean, variance)
    }

    /// Compute MI statistics (mean, variance).
    fn mi_stats(&self) -> (f64, f64) {
        if self.mi_history.is_empty() {
            return (0.0, 0.0);
        }

        let n = self.mi_history.len() as f64;
        let mean = self.mi_history.iter().sum::<f64>() / n;
        let variance = if n > 1.0 {
            self.mi_history
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .sum::<f64>()
                / (n - 1.0)
        } else {
            0.0
        };

        (mean, variance)
    }

    /// Get stability confidence [0, 1].
    ///
    /// Higher confidence when:
    /// - Lag variance is low
    /// - Mean lag is positive
    /// - MI is high and stable
    pub fn stability_confidence(&self) -> f64 {
        if self.lag_history.len() < 5 {
            return 0.0;
        }

        let (mean_lag, var_lag) = self.lag_stats();
        let (mean_mi, _var_mi) = self.mi_stats();

        // Causality component: requires mean lag > threshold
        let causality = if mean_lag > self.min_mean_lag_ms {
            (mean_lag / (self.min_mean_lag_ms * 4.0)).min(1.0)
        } else {
            0.0
        };

        // Stability component: lower variance = higher confidence
        let stability = if var_lag < self.max_variance_ms2 {
            1.0 - (var_lag / self.max_variance_ms2).sqrt()
        } else {
            0.0
        };

        // MI component: higher MI = higher confidence
        let mi_conf = (mean_mi / 0.1).min(1.0); // Saturates at 0.1 bits

        // Combined: geometric mean to require all components
        (causality * stability * mi_conf).powf(1.0 / 3.0)
    }

    /// Get diagnostics for logging.
    pub fn diagnostics(&self) -> LeadLagStabilityDiagnostics {
        let (mean_lag, var_lag) = self.lag_stats();
        let (mean_mi, var_mi) = self.mi_stats();

        LeadLagStabilityDiagnostics {
            mean_lag_ms: mean_lag,
            lag_std_ms: var_lag.sqrt(),
            mean_mi,
            mi_std: var_mi.sqrt(),
            is_stable: self.is_stable(),
            confidence: self.stability_confidence(),
            stable_count: self.stable_count,
            total_observations: self.total_observations,
        }
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.lag_history.clear();
        self.mi_history.clear();
        self.stable_count = 0;
        self.total_observations = 0;
    }
}

/// Diagnostics for lead-lag stability gate.
#[derive(Debug, Clone)]
pub struct LeadLagStabilityDiagnostics {
    pub mean_lag_ms: f64,
    pub lag_std_ms: f64,
    pub mean_mi: f64,
    pub mi_std: f64,
    pub is_stable: bool,
    pub confidence: f64,
    pub stable_count: usize,
    pub total_observations: u64,
}

impl LeadLagStabilityDiagnostics {
    pub fn summary(&self) -> String {
        format!(
            "lag={:.0}±{:.0}ms mi={:.4}±{:.4} stable={} conf={:.2} n={}",
            self.mean_lag_ms,
            self.lag_std_ms,
            self.mean_mi,
            self.mi_std,
            self.is_stable,
            self.confidence,
            self.total_observations
        )
    }
}

/// CUSUM changepoint detector for rapid divergence detection.
///
/// Detects when the Binance-HL price difference shifts significantly
/// within 1-2 observations, before the MI-based lag analyzer can confirm.
/// Used for preemptive (graduated) skew in `signal_integration.rs`.
#[derive(Debug, Clone)]
pub struct CusumDetector {
    /// Cumulative sum (positive direction)
    s_pos: f64,
    /// Cumulative sum (negative direction)
    s_neg: f64,
    /// Running mean of divergence
    mean: f64,
    /// Running variance of divergence
    variance: f64,
    /// Number of observations
    count: u64,
    /// Detection threshold in standard deviations
    threshold_sigma: f64,
    /// EWMA smoothing for mean/variance
    alpha: f64,
}

impl Default for CusumDetector {
    fn default() -> Self {
        Self::new(3.0, 0.05)
    }
}

impl CusumDetector {
    /// Create a CUSUM detector.
    ///
    /// - `threshold_sigma`: Number of standard deviations for detection (default 3.0)
    /// - `alpha`: EWMA smoothing for running statistics (default 0.05)
    pub fn new(threshold_sigma: f64, alpha: f64) -> Self {
        Self {
            s_pos: 0.0,
            s_neg: 0.0,
            mean: 0.0,
            variance: 1.0, // Start with unit variance to avoid divide-by-zero
            count: 0,
            threshold_sigma,
            alpha,
        }
    }

    /// Feed a new divergence observation (e.g., Binance_mid - HL_mid in bps).
    ///
    /// Returns `Some(divergence_bps)` if a changepoint is detected, `None` otherwise.
    pub fn observe(&mut self, divergence_bps: f64) -> Option<f64> {
        self.count += 1;

        // Update running statistics — track variance around zero (fair pricing baseline)
        self.variance =
            (1.0 - self.alpha) * self.variance + self.alpha * divergence_bps * divergence_bps;

        // Track mean for informational purposes (used externally)
        if self.count == 1 {
            self.mean = divergence_bps;
        } else {
            self.mean += self.alpha * (divergence_bps - self.mean);
        }

        // Don't fire during warmup — statistics unreliable
        if self.count < 10 {
            return None;
        }

        // Normalize by RMS of divergence (std dev around zero, not around mean)
        let std_dev = self.variance.sqrt().max(0.1); // Floor to avoid noise amplification
        let normalized = divergence_bps / std_dev;

        // Update CUSUM statistics
        self.s_pos = (self.s_pos + normalized).max(0.0);
        self.s_neg = (self.s_neg - normalized).max(0.0);

        let threshold = self.threshold_sigma;

        if self.s_pos > threshold {
            self.s_pos = 0.0; // Reset after detection
            Some(divergence_bps)
        } else if self.s_neg > threshold {
            self.s_neg = 0.0;
            Some(divergence_bps)
        } else {
            None
        }
    }

    /// Whether the detector has enough data to be reliable.
    pub fn is_warmed_up(&self) -> bool {
        self.count >= 10
    }

    /// Reset the detector state.
    pub fn reset(&mut self) {
        self.s_pos = 0.0;
        self.s_neg = 0.0;
        self.count = 0;
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

    #[test]
    fn test_cusum_detects_shift() {
        let mut detector = CusumDetector::new(3.0, 0.05);

        // Feed neutral observations to warm up
        for _ in 0..20 {
            assert!(detector.observe(0.0).is_none());
        }

        // Sudden shift should be detected
        let mut detected = false;
        for _ in 0..5 {
            if detector.observe(10.0).is_some() {
                detected = true;
                break;
            }
        }
        assert!(
            detected,
            "CUSUM should detect a large shift within 5 observations"
        );
    }

    #[test]
    fn test_cusum_ignores_noise() {
        let mut detector = CusumDetector::new(3.0, 0.05);

        // Feed small fluctuations — should not trigger
        for i in 0..100 {
            let noise = if i % 2 == 0 { 0.1 } else { -0.1 };
            assert!(
                detector.observe(noise).is_none(),
                "CUSUM should not trigger on small noise"
            );
        }
    }

    #[test]
    fn test_cusum_warmup() {
        let detector = CusumDetector::new(3.0, 0.05);
        assert!(!detector.is_warmed_up());

        let mut d2 = detector;
        for _ in 0..10 {
            d2.observe(0.0);
        }
        assert!(d2.is_warmed_up());
    }

    #[test]
    fn test_cusum_reset() {
        let mut detector = CusumDetector::new(3.0, 0.05);
        for _ in 0..20 {
            detector.observe(5.0);
        }
        detector.reset();
        assert!(!detector.is_warmed_up());
        assert_eq!(detector.count, 0);
    }
}
