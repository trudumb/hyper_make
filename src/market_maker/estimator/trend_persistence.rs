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
        }
    }
}

/// Output signal from trend persistence analysis.
#[derive(Debug, Clone, Default)]
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
}

impl TrendPersistenceTracker {
    /// Create a new trend persistence tracker.
    pub fn new(config: TrendConfig) -> Self {
        Self {
            medium_returns: VecDeque::with_capacity(1000),
            long_returns: VecDeque::with_capacity(5000),
            pnl_high_water: 0.0,
            current_unrealized_pnl: 0.0,
            underwater_ticks: 0,
            observation_count: 0,
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

        // Expire old entries
        self.expire_old_returns(timestamp_ms);
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

        TrendSignal {
            short_momentum_bps,
            medium_momentum_bps,
            long_momentum_bps,
            timeframe_agreement,
            underwater_severity,
            trend_confidence,
            is_warmed_up,
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
}
