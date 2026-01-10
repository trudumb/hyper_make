//! Proactive rate limit tracking.
//!
//! Tracks API usage to avoid hitting limits rather than reacting to errors.
//! Also handles 429 response backoff with exponential increase.

use std::time::{Duration, Instant};

/// Configuration for proactive rate limit tracking.
///
/// Hyperliquid rate limits (per docs):
/// - IP: 1200 weight/minute
/// - Address: 1 request per 1 USDC traded (cumulative) + 10K buffer
/// - Batched: 1 IP weight but n address requests
/// - On 429: 1 request per 10 seconds only!
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
    /// Minimum delay between modify operations in ms (prevents rapid OID churn)
    pub min_modify_interval_ms: u64,
    /// Initial backoff duration when 429 received
    pub initial_429_backoff: Duration,
    /// Maximum backoff duration for 429 errors
    pub max_429_backoff: Duration,
    /// Backoff multiplier for consecutive 429 errors
    pub backoff_429_multiplier: f64,
}

impl Default for ProactiveRateLimitConfig {
    fn default() -> Self {
        Self {
            ip_weight_per_minute: 1200,
            ip_warning_threshold: 0.8,
            address_initial_buffer: 10_000,
            requests_per_usd_traded: 1.0,
            min_requote_interval_ms: 100, // 10 requotes/second max
            min_modify_interval_ms: 2000, // Modifies at most every 2 seconds
            initial_429_backoff: Duration::from_secs(10), // Per Hyperliquid docs: 1 req/10s when limited
            max_429_backoff: Duration::from_secs(60),     // Cap at 1 minute
            backoff_429_multiplier: 1.5,                  // Moderate exponential increase
        }
    }
}

/// Proactive rate limit tracker.
///
/// Tracks API usage to avoid hitting limits rather than reacting to errors.
/// Also handles 429 response backoff with exponential increase.
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
    /// Last modify timestamp (for modify debouncing)
    last_modify: Instant,
    /// When 429 backoff expires (if in backoff)
    backoff_until: Option<Instant>,
    /// Consecutive 429 errors (for exponential backoff)
    consecutive_429s: u32,
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
            last_modify: Instant::now() - Duration::from_secs(10), // Allow immediate first modify
            backoff_until: None,
            consecutive_429s: 0,
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
            last_modify: Instant::now() - Duration::from_secs(10), // Allow immediate first modify
            backoff_until: None,
            consecutive_429s: 0,
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
        // Don't requote if we're in 429 backoff
        if self.is_rate_limited() {
            return false;
        }
        self.last_requote.elapsed() >= Duration::from_millis(self.config.min_requote_interval_ms)
    }

    /// Mark that a requote was done.
    pub fn mark_requote(&mut self) {
        self.last_requote = Instant::now();
    }

    /// Check if minimum modify interval has passed.
    ///
    /// Modifies are expensive operations that generate new OIDs on Hyperliquid.
    /// Rate limiting modifies prevents rapid OID churn and rate limit exhaustion.
    pub fn can_modify(&self) -> bool {
        // Don't modify if we're in 429 backoff
        if self.is_rate_limited() {
            return false;
        }
        self.last_modify.elapsed() >= Duration::from_millis(self.config.min_modify_interval_ms)
    }

    /// Mark that a modify operation was done.
    pub fn mark_modify(&mut self) {
        self.last_modify = Instant::now();
    }

    /// Record a 429 (Too Many Requests) response.
    ///
    /// Triggers exponential backoff. Per Hyperliquid docs, when rate limited
    /// you can only make 1 request per 10 seconds.
    ///
    /// # Returns
    /// The backoff duration that was set
    pub fn record_429(&mut self) -> Duration {
        self.consecutive_429s += 1;

        // Calculate backoff with exponential increase
        let multiplier = self
            .config
            .backoff_429_multiplier
            .powi((self.consecutive_429s - 1) as i32);
        let backoff_secs = (self.config.initial_429_backoff.as_secs_f64() * multiplier)
            .min(self.config.max_429_backoff.as_secs_f64());
        let backoff = Duration::from_secs_f64(backoff_secs);

        self.backoff_until = Some(Instant::now() + backoff);

        tracing::error!(
            consecutive_429s = self.consecutive_429s,
            backoff_secs = %format!("{:.1}", backoff_secs),
            "Rate limited (429) - entering backoff. Per Hyperliquid docs: 1 req/10s when limited"
        );

        backoff
    }

    /// Record a successful API call (resets 429 counter).
    pub fn record_api_success(&mut self) {
        if self.consecutive_429s > 0 {
            tracing::info!(
                previous_429s = self.consecutive_429s,
                "API success after rate limiting - resetting 429 counter"
            );
        }
        self.consecutive_429s = 0;
        self.backoff_until = None;
    }

    /// Check if we're currently in 429 backoff.
    pub fn is_rate_limited(&self) -> bool {
        if let Some(until) = self.backoff_until {
            if Instant::now() < until {
                return true;
            }
        }
        false
    }

    /// Get remaining backoff time if rate limited.
    pub fn remaining_backoff(&self) -> Option<Duration> {
        self.backoff_until.and_then(|until| {
            let now = Instant::now();
            if now < until {
                Some(until - now)
            } else {
                None
            }
        })
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
            is_rate_limited: self.is_rate_limited(),
            consecutive_429s: self.consecutive_429s,
        }
    }

    /// Get adaptive queue value config adjusted for current budget pressure.
    ///
    /// When address budget is running low, we want to be MORE conservative
    /// about replacing orders (require higher improvement threshold, lock more
    /// orders). This prevents burning through the remaining budget.
    ///
    /// # Arguments
    /// * `base_config` - The base QueueValueConfig to adapt
    ///
    /// # Returns
    /// An adapted QueueValueConfig with tighter thresholds under budget pressure.
    pub fn get_adaptive_queue_config(
        &self,
        base_config: &crate::market_maker::QueueValueConfig,
    ) -> crate::market_maker::QueueValueConfig {
        let budget = self.address_budget_remaining();

        // Adaptive thresholds based on budget pressure
        let (improvement_threshold, queue_lock_threshold) = match budget {
            // Healthy budget (>5000 remaining): use base config
            r if r > 5000 => (
                base_config.improvement_threshold,
                base_config.queue_lock_threshold,
            ),
            // Moderate pressure (2000-5000): increase improvement threshold by 10%
            r if r > 2000 => (
                base_config.improvement_threshold + 0.10,
                base_config.queue_lock_threshold - 0.05,
            ),
            // High pressure (500-2000): increase by 20%, lock more orders
            r if r > 500 => (
                base_config.improvement_threshold + 0.20,
                base_config.queue_lock_threshold - 0.15,
            ),
            // Critical pressure (<500): very conservative
            _ => (0.50, 0.30),
        };

        crate::market_maker::QueueValueConfig {
            improvement_threshold: improvement_threshold.min(0.50),
            queue_lock_threshold: queue_lock_threshold.max(0.25),
            ..base_config.clone()
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
    pub is_rate_limited: bool,
    pub consecutive_429s: u32,
}
