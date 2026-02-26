//! Rejection-aware rate limiting.
//!
//! Tracks consecutive order rejections and implements exponential backoff
//! to avoid hammering the exchange with invalid orders.

use std::time::{Duration, Instant};

use super::RejectionErrorType;

/// Configuration for rejection-aware rate limiting.
#[derive(Debug, Clone)]
pub struct RejectionRateLimitConfig {
    /// Number of rejections before starting backoff
    pub backoff_start_threshold: u32,
    /// Initial backoff duration after threshold
    pub initial_backoff: Duration,
    /// Maximum backoff duration
    pub max_backoff: Duration,
    /// Backoff multiplier per rejection
    pub backoff_multiplier: f64,
    /// Whether to track only position-related rejections
    pub position_errors_only: bool,
}

impl Default for RejectionRateLimitConfig {
    fn default() -> Self {
        Self {
            backoff_start_threshold: 3,
            initial_backoff: Duration::from_secs(5),
            max_backoff: Duration::from_secs(120),
            backoff_multiplier: 2.0,
            position_errors_only: true,
        }
    }
}

/// Per-side rejection tracking state.
#[derive(Debug, Clone, Default)]
pub(crate) struct SideState {
    /// Consecutive rejections on this side
    pub consecutive_rejections: u32,
    /// When backoff expires (if in backoff)
    pub backoff_until: Option<Instant>,
    /// Total rejections this session
    pub total_rejections: u64,
    /// Total successful orders this session
    pub total_successes: u64,
}

/// Rejection-aware rate limiter.
///
/// Tracks rejections per side and enforces exponential backoff.
#[derive(Debug)]
pub struct RejectionRateLimiter {
    /// Bid side state
    pub(crate) bid_state: SideState,
    /// Ask side state
    pub(crate) ask_state: SideState,
    /// Configuration
    config: RejectionRateLimitConfig,
}

impl Default for RejectionRateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

impl RejectionRateLimiter {
    /// Create a new rate limiter with default configuration.
    pub fn new() -> Self {
        Self {
            bid_state: SideState::default(),
            ask_state: SideState::default(),
            config: RejectionRateLimitConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: RejectionRateLimitConfig) -> Self {
        Self {
            bid_state: SideState::default(),
            ask_state: SideState::default(),
            config,
        }
    }

    /// Record an order rejection.
    ///
    /// # Arguments
    /// - `is_buy`: Whether the rejected order was a buy
    /// - `error`: Error message from exchange
    ///
    /// # Returns
    /// The new backoff duration (if any)
    pub fn record_rejection(&mut self, is_buy: bool, error: &str) -> Option<Duration> {
        // Only track position-related rejections if configured
        if self.config.position_errors_only
            && !error.contains("position")
            && !error.contains("exceed")
            && !error.contains("leverage")
        {
            return None;
        }

        let state = if is_buy {
            &mut self.bid_state
        } else {
            &mut self.ask_state
        };

        state.consecutive_rejections += 1;
        state.total_rejections += 1;

        // Calculate backoff if over threshold
        if state.consecutive_rejections >= self.config.backoff_start_threshold {
            let rejections_over =
                state.consecutive_rejections - self.config.backoff_start_threshold;
            let multiplier = self.config.backoff_multiplier.powi(rejections_over as i32);
            let backoff_secs = (self.config.initial_backoff.as_secs_f64() * multiplier)
                .min(self.config.max_backoff.as_secs_f64());
            let backoff = Duration::from_secs_f64(backoff_secs);

            state.backoff_until = Some(Instant::now() + backoff);

            Some(backoff)
        } else {
            None
        }
    }

    /// Record a batch of order rejections as a single rejection event.
    ///
    /// When a bulk order has multiple rejections of the same type, this counts
    /// as ONE rejection for backoff purposes, but the total count is preserved
    /// for metrics accuracy.
    ///
    /// # Arguments
    /// - `is_buy`: Whether the rejected orders were buys
    /// - `error`: Error message from exchange (representative sample)
    /// - `count`: Number of rejections in this batch
    ///
    /// # Returns
    /// The new backoff duration (if any)
    pub fn record_batch_rejection(
        &mut self,
        is_buy: bool,
        error: &str,
        count: u32,
    ) -> Option<Duration> {
        if count == 0 {
            return None;
        }

        // Only track position-related rejections if configured
        if self.config.position_errors_only
            && !error.contains("position")
            && !error.contains("exceed")
            && !error.contains("leverage")
        {
            return None;
        }

        let state = if is_buy {
            &mut self.bid_state
        } else {
            &mut self.ask_state
        };

        // Increment consecutive_rejections by 1 (batch = single event for backoff)
        // but track the actual count in total_rejections for metrics
        state.consecutive_rejections += 1;
        state.total_rejections += count as u64;

        // Calculate backoff if over threshold
        if state.consecutive_rejections >= self.config.backoff_start_threshold {
            let rejections_over =
                state.consecutive_rejections - self.config.backoff_start_threshold;
            let multiplier = self.config.backoff_multiplier.powi(rejections_over as i32);
            let backoff_secs = (self.config.initial_backoff.as_secs_f64() * multiplier)
                .min(self.config.max_backoff.as_secs_f64());
            let backoff = Duration::from_secs_f64(backoff_secs);

            state.backoff_until = Some(Instant::now() + backoff);

            Some(backoff)
        } else {
            None
        }
    }

    /// Record a batch of rejections with error classification.
    ///
    /// This is the preferred method for batch rejection handling as it provides
    /// both backoff control and error type classification for smarter handling.
    ///
    /// # Arguments
    /// - `is_buy`: Whether the rejected orders were buys
    /// - `error`: Error message from exchange (representative sample)
    /// - `count`: Number of rejections in this batch
    ///
    /// # Returns
    /// A tuple of (backoff_duration, error_type, should_skip_side)
    pub fn classify_and_record_batch(
        &mut self,
        is_buy: bool,
        error: &str,
        count: u32,
    ) -> (Option<Duration>, RejectionErrorType, bool) {
        let error_type = RejectionErrorType::classify(error);
        let backoff = self.record_batch_rejection(is_buy, error, count);
        let should_skip = error_type.should_skip_side();

        (backoff, error_type, should_skip)
    }

    /// Record a successful order placement.
    ///
    /// Resets the consecutive rejection counter for that side.
    pub fn record_success(&mut self, is_buy: bool) {
        let state = if is_buy {
            &mut self.bid_state
        } else {
            &mut self.ask_state
        };

        if state.consecutive_rejections > 0 {
            tracing::debug!(
                is_buy = is_buy,
                was_in_backoff = state.backoff_until.is_some(),
                "Resetting rejection counter after successful order"
            );
        }

        state.consecutive_rejections = 0;
        state.backoff_until = None;
        state.total_successes += 1;
    }

    /// Check if a side should skip order placement due to backoff.
    ///
    /// # Arguments
    /// - `is_buy`: Whether we're checking the buy side
    ///
    /// # Returns
    /// `true` if orders should be skipped, `false` if OK to proceed
    pub fn should_skip(&self, is_buy: bool) -> bool {
        let state = if is_buy {
            &self.bid_state
        } else {
            &self.ask_state
        };

        if let Some(until) = state.backoff_until {
            if Instant::now() < until {
                return true;
            }
        }

        false
    }

    /// Get remaining backoff time for a side.
    ///
    /// # Returns
    /// `Some(duration)` if in backoff, `None` otherwise
    pub fn remaining_backoff(&self, is_buy: bool) -> Option<Duration> {
        let state = if is_buy {
            &self.bid_state
        } else {
            &self.ask_state
        };

        state.backoff_until.and_then(|until| {
            let now = Instant::now();
            if now < until {
                Some(until - now)
            } else {
                None
            }
        })
    }

    /// Reset all state (e.g., on reconnection).
    pub fn reset(&mut self) {
        self.bid_state = SideState::default();
        self.ask_state = SideState::default();
    }

    /// Get metrics for observability.
    pub fn get_metrics(&self) -> RejectionRateLimitMetrics {
        RejectionRateLimitMetrics {
            bid_consecutive_rejections: self.bid_state.consecutive_rejections,
            ask_consecutive_rejections: self.ask_state.consecutive_rejections,
            bid_in_backoff: self
                .bid_state
                .backoff_until
                .map(|u| Instant::now() < u)
                .unwrap_or(false),
            ask_in_backoff: self
                .ask_state
                .backoff_until
                .map(|u| Instant::now() < u)
                .unwrap_or(false),
            bid_total_rejections: self.bid_state.total_rejections,
            ask_total_rejections: self.ask_state.total_rejections,
            bid_total_successes: self.bid_state.total_successes,
            ask_total_successes: self.ask_state.total_successes,
        }
    }
}

/// Metrics for rejection rate limiting.
#[derive(Debug, Clone)]
pub struct RejectionRateLimitMetrics {
    pub bid_consecutive_rejections: u32,
    pub ask_consecutive_rejections: u32,
    pub bid_in_backoff: bool,
    pub ask_in_backoff: bool,
    pub bid_total_rejections: u64,
    pub ask_total_rejections: u64,
    pub bid_total_successes: u64,
    pub ask_total_successes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_backoff_under_threshold() {
        let config = RejectionRateLimitConfig {
            backoff_start_threshold: 3,
            ..Default::default()
        };
        let mut limiter = RejectionRateLimiter::with_config(config);

        // Two rejections should not trigger backoff
        assert!(limiter
            .record_rejection(true, "position exceeded")
            .is_none());
        assert!(limiter
            .record_rejection(true, "position exceeded")
            .is_none());

        assert!(!limiter.should_skip(true));
    }

    #[test]
    fn test_backoff_at_threshold() {
        let config = RejectionRateLimitConfig {
            backoff_start_threshold: 2,
            initial_backoff: Duration::from_secs(5),
            ..Default::default()
        };
        let mut limiter = RejectionRateLimiter::with_config(config);

        // First rejection - no backoff
        assert!(limiter
            .record_rejection(true, "position exceeded")
            .is_none());

        // Second rejection - triggers backoff
        let backoff = limiter.record_rejection(true, "position exceeded");
        assert!(backoff.is_some());
        assert_eq!(backoff.unwrap(), Duration::from_secs(5));

        assert!(limiter.should_skip(true));
    }

    #[test]
    fn test_exponential_backoff() {
        let config = RejectionRateLimitConfig {
            backoff_start_threshold: 1,
            initial_backoff: Duration::from_secs(1),
            backoff_multiplier: 2.0,
            max_backoff: Duration::from_secs(10),
            ..Default::default()
        };
        let mut limiter = RejectionRateLimiter::with_config(config);

        // First rejection - 1s backoff
        let b1 = limiter.record_rejection(true, "position exceeded");
        assert_eq!(b1, Some(Duration::from_secs(1)));

        // Second rejection - 2s backoff
        let b2 = limiter.record_rejection(true, "position exceeded");
        assert_eq!(b2, Some(Duration::from_secs(2)));

        // Third rejection - 4s backoff
        let b3 = limiter.record_rejection(true, "position exceeded");
        assert_eq!(b3, Some(Duration::from_secs(4)));
    }

    #[test]
    fn test_backoff_capped() {
        let config = RejectionRateLimitConfig {
            backoff_start_threshold: 1,
            initial_backoff: Duration::from_secs(10),
            backoff_multiplier: 10.0,
            max_backoff: Duration::from_secs(30),
            ..Default::default()
        };
        let mut limiter = RejectionRateLimiter::with_config(config);

        // First - 10s
        limiter.record_rejection(true, "position exceeded");

        // Second - would be 100s but capped to 30s
        let b2 = limiter.record_rejection(true, "position exceeded");
        assert_eq!(b2, Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_success_resets() {
        let config = RejectionRateLimitConfig {
            backoff_start_threshold: 2,
            ..Default::default()
        };
        let mut limiter = RejectionRateLimiter::with_config(config);

        // Get into backoff
        limiter.record_rejection(true, "position exceeded");
        limiter.record_rejection(true, "position exceeded");
        assert!(limiter.should_skip(true));

        // Success resets
        limiter.record_success(true);
        assert!(!limiter.should_skip(true));
        assert_eq!(limiter.bid_state.consecutive_rejections, 0);
    }

    #[test]
    fn test_sides_independent() {
        let config = RejectionRateLimitConfig {
            backoff_start_threshold: 2,
            ..Default::default()
        };
        let mut limiter = RejectionRateLimiter::with_config(config);

        // Buy side in backoff
        limiter.record_rejection(true, "position exceeded");
        limiter.record_rejection(true, "position exceeded");
        assert!(limiter.should_skip(true));

        // Sell side unaffected
        assert!(!limiter.should_skip(false));
    }

    #[test]
    fn test_ignores_non_position_errors() {
        let mut limiter = RejectionRateLimiter::new();

        // Non-position error - should be ignored
        assert!(limiter
            .record_rejection(true, "insufficient margin")
            .is_none());
        assert_eq!(limiter.bid_state.consecutive_rejections, 0);
    }

    #[test]
    fn test_batch_rejection_counts_as_single_event() {
        let config = RejectionRateLimitConfig {
            backoff_start_threshold: 3,
            initial_backoff: Duration::from_secs(5),
            ..Default::default()
        };
        let mut limiter = RejectionRateLimiter::with_config(config);

        // A batch of 25 rejections should only increment consecutive_rejections by 1
        let backoff = limiter.record_batch_rejection(true, "position exceeded", 25);

        // Should NOT trigger backoff (threshold is 3, we're only at 1)
        assert!(backoff.is_none());
        assert_eq!(limiter.bid_state.consecutive_rejections, 1);
        // But total_rejections should reflect all 25
        assert_eq!(limiter.bid_state.total_rejections, 25);
        assert!(!limiter.should_skip(true));
    }

    #[test]
    fn test_batch_rejection_triggers_backoff_at_threshold() {
        let config = RejectionRateLimitConfig {
            backoff_start_threshold: 2,
            initial_backoff: Duration::from_secs(5),
            ..Default::default()
        };
        let mut limiter = RejectionRateLimiter::with_config(config);

        // First batch of 10 rejections - counts as 1 rejection
        let b1 = limiter.record_batch_rejection(true, "position exceeded", 10);
        assert!(b1.is_none());
        assert_eq!(limiter.bid_state.consecutive_rejections, 1);

        // Second batch of 15 rejections - counts as 2nd rejection, triggers backoff
        let b2 = limiter.record_batch_rejection(true, "position exceeded", 15);
        assert_eq!(b2, Some(Duration::from_secs(5)));
        assert_eq!(limiter.bid_state.consecutive_rejections, 2);
        // Total rejections = 10 + 15 = 25
        assert_eq!(limiter.bid_state.total_rejections, 25);
        assert!(limiter.should_skip(true));
    }

    #[test]
    fn test_batch_rejection_zero_count() {
        let mut limiter = RejectionRateLimiter::new();

        // Zero count should not change anything
        let backoff = limiter.record_batch_rejection(true, "position exceeded", 0);
        assert!(backoff.is_none());
        assert_eq!(limiter.bid_state.consecutive_rejections, 0);
        assert_eq!(limiter.bid_state.total_rejections, 0);
    }

    #[test]
    fn test_classify_and_record_batch() {
        let config = RejectionRateLimitConfig {
            backoff_start_threshold: 2,
            initial_backoff: Duration::from_secs(5),
            ..Default::default()
        };
        let mut limiter = RejectionRateLimiter::with_config(config);

        // First batch with position limit error
        let (backoff, err_type, should_skip) =
            limiter.classify_and_record_batch(true, "PerpMaxPosition: exceeded", 10);

        assert!(backoff.is_none()); // Not at threshold yet
        assert_eq!(err_type, RejectionErrorType::PositionLimit);
        assert!(should_skip); // Should skip side for position limit

        // Second batch
        let (backoff, err_type, should_skip) =
            limiter.classify_and_record_batch(true, "PerpMaxPosition: exceeded", 5);

        assert_eq!(backoff, Some(Duration::from_secs(5))); // At threshold now
        assert_eq!(err_type, RejectionErrorType::PositionLimit);
        assert!(should_skip);
    }
}
