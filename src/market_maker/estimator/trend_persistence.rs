//! Multi-timeframe trend persistence tracking.
//!
//! Detects sustained trends by combining:
//! - Multi-timeframe momentum (500ms, 30s, 5min windows)
//! - Underwater P&L tracking (position losing money)
//!
//! This addresses the "bounce within trend" problem where short-term
//! momentum flips mask sustained price moves.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// Configuration for trend persistence tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendConfig {
    /// Short-term momentum window in milliseconds (existing momentum detector).
    pub short_window_ms: u64,

    /// Medium-term momentum window in milliseconds.
    pub medium_window_ms: u64,

    /// Long-term momentum window in milliseconds.
    pub long_window_ms: u64,

    /// Minimum unrealized P&L loss (in bps of position value) to be considered "underwater".
    pub underwater_threshold_bps: f64,

    /// Multiplier for drift urgency when multiple timeframes agree.
    pub agreement_boost: f64,

    /// Number of consecutive underwater ticks before considering it significant.
    pub underwater_min_ticks: u32,

    /// EWMA alpha for drift velocity smoothing (0.0 = slow, 1.0 = raw).
    #[serde(default = "default_drift_velocity_alpha")]
    pub drift_velocity_alpha: f64,

    /// Capacity of the recent returns ring buffer for autocorrelation.
    #[serde(default = "default_autocorrelation_window")]
    pub autocorrelation_window: usize,
}

fn default_drift_velocity_alpha() -> f64 {
    0.1
}

fn default_autocorrelation_window() -> usize {
    50
}

impl Default for TrendConfig {
    fn default() -> Self {
        Self {
            short_window_ms: 500,           // 500ms (matches existing)
            medium_window_ms: 30_000,       // 30 seconds
            long_window_ms: 300_000,        // 5 minutes
            underwater_threshold_bps: 20.0, // 20 bps loss = underwater
            agreement_boost: 2.0,           // 2x urgency when timeframes agree
            underwater_min_ticks: 5,        // 5 ticks = ~500ms sustained
            drift_velocity_alpha: 0.1,      // EWMA smoothing for drift velocity
            autocorrelation_window: 50,     // ring buffer capacity for autocorrelation
        }
    }
}

/// Output signal from trend persistence analysis.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrendSignal {
    /// Short-term momentum in basis points (500ms window).
    pub short_momentum_bps: f64,

    /// Medium-term momentum in basis points (30s window).
    pub medium_momentum_bps: f64,

    /// Long-term momentum in basis points (5min window).
    pub long_momentum_bps: f64,

    /// How aligned the timeframes are (0.0 = all different signs, 1.0 = all same sign).
    pub timeframe_agreement: f64,

    /// Underwater severity (0.0 = at/above high water, 1.0 = significantly underwater).
    pub underwater_severity: f64,

    /// Combined confidence in sustained trend (0.0 to 1.0).
    pub trend_confidence: f64,

    /// Whether the trend tracker is warmed up (enough data in all windows).
    pub is_warmed_up: bool,

    /// EWMA-smoothed rate of change of long momentum (bps per second).
    #[serde(default)]
    pub drift_velocity_bps_per_s: f64,

    /// True when drift velocity has the same sign as long momentum AND |velocity| > 1.0 bps/s.
    #[serde(default)]
    pub drift_accelerating: bool,

    /// EWMA-smoothed lag-1 return autocorrelation.
    #[serde(default)]
    pub return_autocorrelation: f64,

    /// True when autocorrelation > 0.1 (genuine trending behavior).
    #[serde(default)]
    pub is_trending: bool,
}

/// Multi-timeframe trend persistence tracker.
#[derive(Debug)]
pub struct TrendPersistenceTracker {
    /// Returns in medium window: (timestamp_ms, log_return).
    medium_returns: VecDeque<(u64, f64)>,

    /// Returns in long window: (timestamp_ms, log_return).
    long_returns: VecDeque<(u64, f64)>,

    /// High-water mark for unrealized P&L.
    pnl_high_water: f64,

    /// Current unrealized P&L.
    current_unrealized_pnl: f64,

    /// Consecutive ticks spent underwater.
    underwater_ticks: u32,

    /// Observation count for warmup.
    observation_count: u64,

    /// Configuration.
    config: TrendConfig,

    // --- Drift velocity state ---
    /// Previous long-term momentum value (bps) for velocity computation.
    prev_long_momentum_bps: f64,

    /// Timestamp (ms) of the previous update (for dt computation).
    prev_update_ms: u64,

    /// EWMA-smoothed drift velocity (bps/s).
    drift_velocity_bps_per_s: f64,

    // --- Autocorrelation state ---
    /// Ring buffer of recent log returns for lag-1 autocorrelation.
    recent_returns: VecDeque<f64>,

    /// EWMA-smoothed lag-1 autocorrelation.
    autocorrelation_ewma: f64,
}

impl TrendPersistenceTracker {
    /// Create a new trend persistence tracker.
    pub fn new(config: TrendConfig) -> Self {
        let autocorrelation_cap = config.autocorrelation_window;
        Self {
            medium_returns: VecDeque::with_capacity(1000),
            long_returns: VecDeque::with_capacity(5000),
            pnl_high_water: 0.0,
            current_unrealized_pnl: 0.0,
            underwater_ticks: 0,
            observation_count: 0,
            prev_long_momentum_bps: 0.0,
            prev_update_ms: 0,
            drift_velocity_bps_per_s: 0.0,
            recent_returns: VecDeque::with_capacity(autocorrelation_cap),
            autocorrelation_ewma: 0.0,
            config,
        }
    }

    /// Record a new VWAP return observation.
    ///
    /// # Arguments
    /// * `timestamp_ms` - Current timestamp in milliseconds
    /// * `log_return` - Signed log return (price change)
    pub fn on_bucket(&mut self, timestamp_ms: u64, log_return: f64) {
        self.observation_count += 1;

        // Add to both windows
        self.medium_returns.push_back((timestamp_ms, log_return));
        self.long_returns.push_back((timestamp_ms, log_return));

        // --- Drift velocity update ---
        let current_long_bps =
            self.momentum_bps(&self.long_returns, timestamp_ms, self.config.long_window_ms);
        if self.prev_update_ms > 0 {
            let dt_ms = timestamp_ms.saturating_sub(self.prev_update_ms);
            if dt_ms > 0 {
                let dt_s = dt_ms as f64 / 1000.0;
                let raw_velocity = (current_long_bps - self.prev_long_momentum_bps) / dt_s;
                let alpha = self.config.drift_velocity_alpha;
                self.drift_velocity_bps_per_s =
                    alpha * raw_velocity + (1.0 - alpha) * self.drift_velocity_bps_per_s;
            }
        }
        self.prev_long_momentum_bps = current_long_bps;
        self.prev_update_ms = timestamp_ms;

        // --- Autocorrelation update ---
        // Push return into ring buffer, cap at configured window
        if self.recent_returns.len() >= self.config.autocorrelation_window {
            self.recent_returns.pop_front();
        }
        self.recent_returns.push_back(log_return);
        self.update_autocorrelation();

        // Expire old entries
        self.expire_old_returns(timestamp_ms);
    }

    /// Compute lag-1 autocorrelation from the recent returns buffer and EWMA smooth it.
    fn update_autocorrelation(&mut self) {
        let n = self.recent_returns.len();
        if n < 3 {
            return;
        }

        // Compute mean
        let mean: f64 = self.recent_returns.iter().sum::<f64>() / n as f64;

        // Compute variance
        let var: f64 = self
            .recent_returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / n as f64;

        if var < 1e-30 {
            // Near-zero variance — autocorrelation undefined, leave unchanged
            return;
        }

        // Compute lag-1 covariance: cov(r_t, r_{t-1})
        let cov: f64 = self
            .recent_returns
            .iter()
            .skip(1)
            .zip(self.recent_returns.iter())
            .map(|(r_t, r_t1)| (r_t - mean) * (r_t1 - mean))
            .sum::<f64>()
            / (n - 1) as f64;

        let raw_autocorr = (cov / var).clamp(-1.0, 1.0);

        // EWMA smooth using the same alpha as drift velocity
        let alpha = self.config.drift_velocity_alpha;
        self.autocorrelation_ewma =
            alpha * raw_autocorr + (1.0 - alpha) * self.autocorrelation_ewma;
    }

    /// Update unrealized P&L for underwater tracking.
    ///
    /// # Arguments
    /// * `unrealized_pnl` - Current unrealized P&L in USD
    pub fn update_pnl(&mut self, unrealized_pnl: f64) {
        self.current_unrealized_pnl = unrealized_pnl;

        // Update high-water mark
        if unrealized_pnl > self.pnl_high_water {
            self.pnl_high_water = unrealized_pnl;
            self.underwater_ticks = 0;
        } else {
            // Track how long we've been underwater
            let depth = self.pnl_high_water - unrealized_pnl;
            if depth > 0.0 {
                self.underwater_ticks = self.underwater_ticks.saturating_add(1);
            }
        }
    }

    /// Reset high-water mark (call when position is closed).
    pub fn reset_high_water(&mut self) {
        self.pnl_high_water = 0.0;
        self.current_unrealized_pnl = 0.0;
        self.underwater_ticks = 0;
    }

    /// Evaluate trend signals.
    ///
    /// # Arguments
    /// * `now_ms` - Current timestamp
    /// * `short_momentum_bps` - Short-term momentum from existing MomentumDetector
    /// * `position_value` - Absolute position value in USD (for underwater severity calc)
    pub fn evaluate(
        &self,
        now_ms: u64,
        short_momentum_bps: f64,
        position_value: f64,
    ) -> TrendSignal {
        let medium_momentum_bps =
            self.momentum_bps(&self.medium_returns, now_ms, self.config.medium_window_ms);
        let long_momentum_bps =
            self.momentum_bps(&self.long_returns, now_ms, self.config.long_window_ms);

        let timeframe_agreement =
            self.calculate_agreement(short_momentum_bps, medium_momentum_bps, long_momentum_bps);

        let underwater_severity = self.calculate_underwater_severity(position_value);

        let trend_confidence = self.calculate_trend_confidence(
            timeframe_agreement,
            underwater_severity,
            medium_momentum_bps.abs(),
            long_momentum_bps.abs(),
        );

        let is_warmed_up = self.is_warmed_up();

        // Drift velocity: accelerating if velocity same sign as long momentum and |v| > 1.0 bps/s
        let drift_velocity_bps_per_s = self.drift_velocity_bps_per_s;
        let drift_accelerating = drift_velocity_bps_per_s.signum() == long_momentum_bps.signum()
            && long_momentum_bps.abs() > 1.0
            && drift_velocity_bps_per_s.abs() > 1.0;

        // Autocorrelation: trending if positively autocorrelated
        let return_autocorrelation = self.autocorrelation_ewma;
        let is_trending = return_autocorrelation > 0.1;

        TrendSignal {
            short_momentum_bps,
            medium_momentum_bps,
            long_momentum_bps,
            timeframe_agreement,
            underwater_severity,
            trend_confidence,
            is_warmed_up,
            drift_velocity_bps_per_s,
            drift_accelerating,
            return_autocorrelation,
            is_trending,
        }
    }

    /// Calculate momentum in bps from a returns window.
    fn momentum_bps(&self, returns: &VecDeque<(u64, f64)>, now_ms: u64, window_ms: u64) -> f64 {
        let cutoff = now_ms.saturating_sub(window_ms);
        let sum: f64 = returns
            .iter()
            .filter(|(t, _)| *t >= cutoff)
            .map(|(_, r)| r)
            .sum();
        sum * 10_000.0
    }

    /// Calculate timeframe agreement (0.0 to 1.0).
    fn calculate_agreement(&self, short_bps: f64, medium_bps: f64, long_bps: f64) -> f64 {
        // Treat near-zero as neutral (no opinion)
        const THRESHOLD: f64 = 1.0; // 1 bps minimum to count

        let short_sign = if short_bps.abs() > THRESHOLD {
            short_bps.signum()
        } else {
            0.0
        };
        let medium_sign = if medium_bps.abs() > THRESHOLD {
            medium_bps.signum()
        } else {
            0.0
        };
        let long_sign = if long_bps.abs() > THRESHOLD {
            long_bps.signum()
        } else {
            0.0
        };

        let mut agreements = 0;
        let mut comparisons = 0;

        // Only count comparisons where both have opinions
        if short_sign != 0.0 && medium_sign != 0.0 {
            comparisons += 1;
            if short_sign == medium_sign {
                agreements += 1;
            }
        }
        if medium_sign != 0.0 && long_sign != 0.0 {
            comparisons += 1;
            if medium_sign == long_sign {
                agreements += 1;
            }
        }
        if short_sign != 0.0 && long_sign != 0.0 {
            comparisons += 1;
            if short_sign == long_sign {
                agreements += 1;
            }
        }

        if comparisons == 0 {
            0.0
        } else {
            agreements as f64 / comparisons as f64
        }
    }

    /// Calculate underwater severity (0.0 to 1.0).
    fn calculate_underwater_severity(&self, position_value: f64) -> f64 {
        if position_value < 1.0 || self.underwater_ticks < self.config.underwater_min_ticks {
            return 0.0;
        }

        let depth = self.pnl_high_water - self.current_unrealized_pnl;
        if depth <= 0.0 {
            return 0.0;
        }

        // Convert depth to bps of position value
        let depth_bps = (depth / position_value) * 10_000.0;

        // Severity scales from 0 at threshold to 1 at 5x threshold
        (depth_bps / self.config.underwater_threshold_bps - 1.0).clamp(0.0, 1.0)
    }

    /// Calculate overall trend confidence.
    fn calculate_trend_confidence(
        &self,
        agreement: f64,
        underwater: f64,
        medium_magnitude: f64,
        long_magnitude: f64,
    ) -> f64 {
        // Base confidence from timeframe agreement
        let agreement_conf = agreement;

        // Magnitude confidence (stronger momentum = more confident)
        // Scale: 10 bps = 0.5, 50 bps = 1.0
        let magnitude_conf = ((medium_magnitude + long_magnitude) / 2.0 / 50.0).min(1.0);

        // Underwater adds urgency
        let underwater_conf = underwater;

        // Combined: agreement is primary, magnitude and underwater are secondary
        let combined = 0.5 * agreement_conf + 0.3 * magnitude_conf + 0.2 * underwater_conf;

        combined.min(1.0)
    }

    /// Check if tracker has enough data.
    pub fn is_warmed_up(&self) -> bool {
        // Need at least 30 observations (roughly 3-6 seconds of data)
        // and some data in both medium and long windows
        self.observation_count >= 30
            && !self.medium_returns.is_empty()
            && !self.long_returns.is_empty()
    }

    /// Expire old returns from windows.
    fn expire_old_returns(&mut self, now_ms: u64) {
        // Medium window: keep 2x window for safety
        let medium_cutoff = now_ms.saturating_sub(self.config.medium_window_ms * 2);
        while let Some((t, _)) = self.medium_returns.front() {
            if *t < medium_cutoff {
                self.medium_returns.pop_front();
            } else {
                break;
            }
        }

        // Long window: keep 1.5x window for safety
        let long_cutoff = now_ms.saturating_sub(self.config.long_window_ms * 3 / 2);
        while let Some((t, _)) = self.long_returns.front() {
            if *t < long_cutoff {
                self.long_returns.pop_front();
            } else {
                break;
            }
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &TrendConfig {
        &self.config
    }

    /// Get medium-term momentum directly.
    pub fn medium_momentum_bps(&self, now_ms: u64) -> f64 {
        self.momentum_bps(&self.medium_returns, now_ms, self.config.medium_window_ms)
    }

    /// Get long-term momentum directly.
    pub fn long_momentum_bps(&self, now_ms: u64) -> f64 {
        self.momentum_bps(&self.long_returns, now_ms, self.config.long_window_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_tracker() -> TrendPersistenceTracker {
        TrendPersistenceTracker::new(TrendConfig::default())
    }

    #[test]
    fn test_config_defaults() {
        let config = TrendConfig::default();
        assert_eq!(config.short_window_ms, 500);
        assert_eq!(config.medium_window_ms, 30_000);
        assert_eq!(config.long_window_ms, 300_000);
        assert!((config.underwater_threshold_bps - 20.0).abs() < f64::EPSILON);
        assert!((config.agreement_boost - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_warmup() {
        let mut tracker = default_tracker();
        assert!(!tracker.is_warmed_up());

        // Add 30 observations
        for i in 0..30 {
            tracker.on_bucket(i * 100, 0.0001);
        }
        assert!(tracker.is_warmed_up());
    }

    #[test]
    fn test_momentum_accumulation() {
        let mut tracker = default_tracker();
        let now = 50_000u64;

        // Add positive returns over 30s
        for i in 0..100 {
            let t = now - 30_000 + i * 300;
            tracker.on_bucket(t, 0.0001); // +1 bps each
        }

        let medium = tracker.medium_momentum_bps(now);
        assert!(medium > 50.0, "Expected >50 bps, got {}", medium);
    }

    #[test]
    fn test_agreement_all_positive() {
        let tracker = default_tracker();
        let agreement = tracker.calculate_agreement(10.0, 20.0, 30.0);
        assert!((agreement - 1.0).abs() < 0.01, "All positive should agree");
    }

    #[test]
    fn test_agreement_mixed() {
        let tracker = default_tracker();
        // Short positive, medium/long negative
        let agreement = tracker.calculate_agreement(10.0, -20.0, -30.0);
        // 2/3 agree (medium-long), 1/3 disagree (short-medium, short-long)
        assert!(
            agreement < 0.5,
            "Mixed signs should have low agreement: {}",
            agreement
        );
    }

    #[test]
    fn test_underwater_tracking() {
        let mut tracker = default_tracker();

        // Start with profit
        tracker.update_pnl(100.0);
        assert_eq!(tracker.pnl_high_water, 100.0);

        // Go underwater
        for _ in 0..10 {
            tracker.update_pnl(50.0);
        }
        assert!(tracker.underwater_ticks >= 5);

        // Check severity (50 drop on 1000 position = 50 bps = significant)
        let severity = tracker.calculate_underwater_severity(1000.0);
        assert!(severity > 0.0);
    }

    #[test]
    fn test_high_water_reset() {
        let mut tracker = default_tracker();

        tracker.update_pnl(100.0);
        tracker.update_pnl(50.0);
        tracker.update_pnl(50.0);

        tracker.reset_high_water();
        assert_eq!(tracker.pnl_high_water, 0.0);
        assert_eq!(tracker.underwater_ticks, 0);
    }

    #[test]
    fn test_trend_signal_structure() {
        let mut tracker = default_tracker();

        // Add some data
        for i in 0..50 {
            tracker.on_bucket(i * 100, 0.0001);
        }
        tracker.update_pnl(100.0);

        let signal = tracker.evaluate(5000, 10.0, 1000.0);
        assert!(signal.is_warmed_up);
        assert!(signal.medium_momentum_bps > 0.0);
        assert!(signal.trend_confidence >= 0.0 && signal.trend_confidence <= 1.0);
    }

    #[test]
    fn test_drift_acceleration_detection() {
        let mut tracker = default_tracker();

        // Feed increasing positive returns — momentum should accelerate.
        // Phase 1: small positive returns to build baseline
        for i in 0..40 {
            let t = 1000 + i * 100; // 100ms apart
            tracker.on_bucket(t, 0.0001); // +1 bps each
        }

        // Phase 2: larger positive returns — momentum increasing (accelerating)
        for i in 0..20 {
            let t = 5000 + i * 100;
            tracker.on_bucket(t, 0.001); // +10 bps each
        }

        let now = 7000;
        let signal = tracker.evaluate(now, 50.0, 1000.0);

        // Velocity should be positive (momentum increasing in positive direction)
        assert!(
            signal.drift_velocity_bps_per_s > 0.0,
            "Expected positive drift velocity, got {}",
            signal.drift_velocity_bps_per_s
        );
        // Long momentum is positive and velocity is positive → accelerating
        assert!(
            signal.drift_accelerating,
            "Expected drift_accelerating=true, velocity={}, long_mom={}",
            signal.drift_velocity_bps_per_s, signal.long_momentum_bps
        );
    }

    #[test]
    fn test_drift_deceleration_detection() {
        let mut tracker = default_tracker();

        // Phase 1: strong positive returns
        for i in 0..30 {
            let t = 1000 + i * 100;
            tracker.on_bucket(t, 0.001); // +10 bps each
        }

        // Phase 2: returns flip negative — momentum decelerating/reversing
        for i in 0..30 {
            let t = 4000 + i * 100;
            tracker.on_bucket(t, -0.001); // -10 bps each
        }

        let now = 7000;
        let signal = tracker.evaluate(now, -20.0, 1000.0);

        // Velocity should be negative (momentum was positive, now decreasing)
        assert!(
            signal.drift_velocity_bps_per_s < 0.0,
            "Expected negative drift velocity during deceleration, got {}",
            signal.drift_velocity_bps_per_s
        );
    }

    #[test]
    fn test_positive_autocorrelation_sustained_trend() {
        let mut tracker = default_tracker();

        // Feed consistently positive returns with small random-walk noise.
        // True trending: each return ≈ previous return (positive autocorrelation).
        let mut ret = 0.0005;
        for i in 0..60 {
            let t = 1000 + i * 100;
            // Small perturbation that preserves sign — simulates genuine trend
            let noise = 0.00001 * ((i % 5) as f64 - 2.0); // ±0.00002
            ret = (ret + noise).max(0.0001); // always positive, slowly drifting
            tracker.on_bucket(t, ret);
        }

        let now = 7000;
        let signal = tracker.evaluate(now, 20.0, 1000.0);

        assert!(
            signal.return_autocorrelation > 0.0,
            "Expected positive autocorrelation in sustained trend, got {}",
            signal.return_autocorrelation
        );
        assert!(
            signal.is_trending,
            "Expected is_trending=true for positive autocorrelation {}",
            signal.return_autocorrelation
        );
    }

    #[test]
    fn test_negative_autocorrelation_choppy_market() {
        let mut tracker = default_tracker();

        // Feed alternating positive/negative returns — classic mean-reverting bounce.
        for i in 0..60 {
            let t = 1000 + i * 100;
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            tracker.on_bucket(t, sign * 0.001);
        }

        let now = 7000;
        let signal = tracker.evaluate(now, 0.0, 1000.0);

        assert!(
            signal.return_autocorrelation < 0.0,
            "Expected negative autocorrelation in choppy market, got {}",
            signal.return_autocorrelation
        );
        assert!(
            !signal.is_trending,
            "Expected is_trending=false for negative autocorrelation {}",
            signal.return_autocorrelation
        );
    }
}
