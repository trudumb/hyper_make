//! Rejection-Aware Rate Limiting (Phase 5 Fix)
//!
//! Tracks consecutive order rejections and implements exponential backoff
//! to avoid hammering the exchange with invalid orders.
//!
//! # Design
//!
//! - Separate tracking for bid and ask sides
//! - Exponential backoff with configurable thresholds
//! - Automatic reset on successful order placement
//! - Metrics for monitoring backoff state

use std::time::{Duration, Instant};

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
struct SideState {
    /// Consecutive rejections on this side
    consecutive_rejections: u32,
    /// When backoff expires (if in backoff)
    backoff_until: Option<Instant>,
    /// Total rejections this session
    total_rejections: u64,
    /// Total successful orders this session
    total_successes: u64,
}

/// Rejection-aware rate limiter.
///
/// Tracks rejections per side and enforces exponential backoff.
#[derive(Debug)]
pub struct RejectionRateLimiter {
    /// Bid side state
    bid_state: SideState,
    /// Ask side state
    ask_state: SideState,
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

            tracing::warn!(
                is_buy = is_buy,
                consecutive_rejections = state.consecutive_rejections,
                backoff_secs = %format!("{:.1}", backoff_secs),
                "Entering backoff due to consecutive rejections"
            );

            Some(backoff)
        } else {
            None
        }
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

// ============================================================================
// Proactive Rate Limit Tracker
// ============================================================================

/// Configuration for proactive rate limit tracking.
///
/// Hyperliquid rate limits (per docs):
/// - IP: 1200 weight/minute
/// - Address: 1 request per 1 USDC traded (cumulative) + 10K buffer
/// - Batched: 1 IP weight but n address requests
#[derive(Debug, Clone)]
pub struct ProactiveRateLimitConfig {
    /// IP weight limit per minute (default: 1200)
    pub ip_weight_per_minute: u32,
    /// Warning threshold as fraction of limit (default: 0.8)
    pub ip_warning_threshold: f64,
    /// Address request buffer (initial budget)
    pub address_initial_buffer: u64,
    /// Requests per USDC traded (address budget accumulation)
    pub requests_per_usd_traded: f64,
    /// Minimum delay between requotes in ms
    pub min_requote_interval_ms: u64,
}

impl Default for ProactiveRateLimitConfig {
    fn default() -> Self {
        Self {
            ip_weight_per_minute: 1200,
            ip_warning_threshold: 0.8,
            address_initial_buffer: 10_000,
            requests_per_usd_traded: 1.0,
            min_requote_interval_ms: 100, // 10 requotes/second max
        }
    }
}

/// Proactive rate limit tracker.
///
/// Tracks API usage to avoid hitting limits rather than reacting to errors.
#[derive(Debug)]
pub struct ProactiveRateLimitTracker {
    config: ProactiveRateLimitConfig,
    /// Rolling window of IP weights (last 60 seconds)
    ip_weights: Vec<(Instant, u32)>,
    /// Total address requests made this session
    address_requests: u64,
    /// Total USD volume traded (for address budget calculation)
    usd_volume_traded: f64,
    /// Last requote timestamp
    last_requote: Instant,
}

impl Default for ProactiveRateLimitTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ProactiveRateLimitTracker {
    /// Create a new tracker with default config.
    pub fn new() -> Self {
        Self {
            config: ProactiveRateLimitConfig::default(),
            ip_weights: Vec::new(),
            address_requests: 0,
            usd_volume_traded: 0.0,
            last_requote: Instant::now(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: ProactiveRateLimitConfig) -> Self {
        Self {
            config,
            ip_weights: Vec::new(),
            address_requests: 0,
            usd_volume_traded: 0.0,
            last_requote: Instant::now(),
        }
    }

    /// Record an API call with its weight.
    ///
    /// # Arguments
    /// - `ip_weight`: Weight for IP rate limit (1 for most calls)
    /// - `address_requests`: Number of address-level requests (n for batched)
    pub fn record_call(&mut self, ip_weight: u32, address_requests: u32) {
        let now = Instant::now();
        self.ip_weights.push((now, ip_weight));
        self.address_requests += address_requests as u64;

        // Prune old entries (> 60 seconds)
        let cutoff = now - Duration::from_secs(60);
        self.ip_weights.retain(|(t, _)| *t > cutoff);
    }

    /// Record a fill volume for address budget calculation.
    pub fn record_fill_volume(&mut self, usd_value: f64) {
        self.usd_volume_traded += usd_value;
    }

    /// Get current IP weight usage in the last minute.
    pub fn ip_weight_used(&self) -> u32 {
        let now = Instant::now();
        let cutoff = now - Duration::from_secs(60);
        self.ip_weights
            .iter()
            .filter(|(t, _)| *t > cutoff)
            .map(|(_, w)| w)
            .sum()
    }

    /// Check if we're approaching IP rate limit.
    pub fn ip_rate_warning(&self) -> bool {
        let used = self.ip_weight_used();
        let threshold =
            (self.config.ip_weight_per_minute as f64 * self.config.ip_warning_threshold) as u32;
        used >= threshold
    }

    /// Calculate remaining address budget.
    pub fn address_budget_remaining(&self) -> i64 {
        let budget = self.config.address_initial_buffer as f64
            + self.usd_volume_traded * self.config.requests_per_usd_traded;
        (budget as i64) - (self.address_requests as i64)
    }

    /// Check if address budget is low (< 1000 remaining).
    pub fn address_budget_low(&self) -> bool {
        self.address_budget_remaining() < 1000
    }

    /// Check if minimum requote interval has passed.
    pub fn can_requote(&self) -> bool {
        self.last_requote.elapsed() >= Duration::from_millis(self.config.min_requote_interval_ms)
    }

    /// Mark that a requote was done.
    pub fn mark_requote(&mut self) {
        self.last_requote = Instant::now();
    }

    /// Get metrics for logging/monitoring.
    pub fn get_metrics(&self) -> ProactiveRateLimitMetrics {
        ProactiveRateLimitMetrics {
            ip_weight_used_per_minute: self.ip_weight_used(),
            ip_weight_limit: self.config.ip_weight_per_minute,
            address_requests_used: self.address_requests,
            address_budget_remaining: self.address_budget_remaining(),
            usd_volume_traded: self.usd_volume_traded,
            ip_warning: self.ip_rate_warning(),
            address_warning: self.address_budget_low(),
        }
    }
}

/// Metrics from proactive rate limit tracker.
#[derive(Debug, Clone)]
pub struct ProactiveRateLimitMetrics {
    pub ip_weight_used_per_minute: u32,
    pub ip_weight_limit: u32,
    pub address_requests_used: u64,
    pub address_budget_remaining: i64,
    pub usd_volume_traded: f64,
    pub ip_warning: bool,
    pub address_warning: bool,
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
}
