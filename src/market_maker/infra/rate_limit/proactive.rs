//! Proactive rate limit tracking.
//!
//! Tracks API usage to avoid hitting limits rather than reacting to errors.
//! Also handles 429 response backoff with exponential increase.

use std::time::{Duration, Instant};

// =============================================================================
// Cumulative Quota State - Death Spiral Prevention
// =============================================================================

/// Tracks cumulative quota exhaustion and implements exponential backoff with jitter.
///
/// When the rate limit headroom is exhausted (<5%), the system records this as a
/// "cumulative exhaustion" event. Consecutive exhaustions trigger exponentially
/// increasing backoff periods to break the death spiral pattern:
///
/// - 1st exhaustion: 30s backoff (± 50% jitter)
/// - 2nd exhaustion: 60s backoff
/// - 3rd exhaustion: 120s backoff
/// - 4th exhaustion: 240s backoff
/// - Max: 600s (10 minutes)
///
/// The jitter prevents herd behavior when multiple systems hit limits simultaneously.
#[derive(Debug, Clone)]
pub struct CumulativeQuotaState {
    /// Consecutive exhaustion events
    pub consecutive_exhaustions: u32,
    /// Timestamp of last exhaustion
    pub last_exhaustion: Option<Instant>,
    /// When the backoff expires
    pub backoff_until: Option<Instant>,
}

impl Default for CumulativeQuotaState {
    fn default() -> Self {
        Self::new()
    }
}

impl CumulativeQuotaState {
    /// Create new quota state.
    pub fn new() -> Self {
        Self {
            consecutive_exhaustions: 0,
            last_exhaustion: None,
            backoff_until: None,
        }
    }

    /// Record a quota exhaustion event and compute backoff.
    ///
    /// Returns the backoff duration that was set (includes jitter).
    pub fn record_exhaustion(&mut self) -> Duration {
        self.consecutive_exhaustions += 1;
        self.last_exhaustion = Some(Instant::now());

        // Exponential backoff: 30s, 60s, 120s, 240s... max 10min
        let base = Duration::from_secs(30);
        let multiplier = 2.0_f64.powi((self.consecutive_exhaustions - 1).min(4) as i32);
        let backoff = Duration::from_secs_f64(
            (base.as_secs_f64() * multiplier).min(600.0)
        );

        // Add jitter: ± 50% uniform noise (queuing theory: avoid herd behavior)
        let jitter_factor = 0.5 + rand::random::<f64>(); // [0.5, 1.5]
        let jittered = Duration::from_secs_f64(backoff.as_secs_f64() * jitter_factor);

        self.backoff_until = Some(Instant::now() + jittered);

        tracing::warn!(
            consecutive_exhaustions = self.consecutive_exhaustions,
            base_backoff_secs = %format!("{:.1}", backoff.as_secs_f64()),
            jittered_backoff_secs = %format!("{:.1}", jittered.as_secs_f64()),
            "Cumulative quota exhaustion - entering exponential backoff"
        );

        jittered
    }

    /// Check if currently in backoff period.
    pub fn is_in_backoff(&self) -> bool {
        self.backoff_until
            .map(|until| Instant::now() < until)
            .unwrap_or(false)
    }

    /// Get remaining backoff time (if in backoff).
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

    /// Reset on successful API call after recovery.
    pub fn reset_on_success(&mut self) {
        if self.consecutive_exhaustions > 0 {
            tracing::info!(
                previous_exhaustions = self.consecutive_exhaustions,
                "Quota recovered - resetting exhaustion counter"
            );
            self.consecutive_exhaustions = 0;
            self.backoff_until = None;
        }
    }
}

// =============================================================================
// Proactive Rate Limit Configuration
// =============================================================================

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
/// Now also tracks cumulative quota exhaustion for death spiral prevention.
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
    /// Cumulative quota exhaustion state (death spiral prevention)
    cumulative_quota: CumulativeQuotaState,
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
            cumulative_quota: CumulativeQuotaState::new(),
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
            cumulative_quota: CumulativeQuotaState::new(),
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
        // Also reset cumulative quota exhaustion on success
        self.cumulative_quota.reset_on_success();
    }

    // =========================================================================
    // Cumulative Quota Methods (Death Spiral Prevention)
    // =========================================================================

    /// Record cumulative quota exhaustion and enter exponential backoff.
    ///
    /// Call this when headroom < 5% and we would otherwise try to place orders.
    /// Returns the backoff duration that was set.
    pub fn record_cumulative_exhaustion(&mut self) -> Duration {
        self.cumulative_quota.record_exhaustion()
    }

    /// Check if we're in cumulative quota backoff.
    ///
    /// When true, the system should NOT attempt to place orders - let fills
    /// accumulate to restore quota before resuming.
    pub fn is_cumulative_backoff(&self) -> bool {
        self.cumulative_quota.is_in_backoff()
    }

    /// Get remaining cumulative backoff time.
    pub fn remaining_cumulative_backoff(&self) -> Option<Duration> {
        self.cumulative_quota.remaining_backoff()
    }

    /// Get consecutive exhaustion count (for monitoring/alerting).
    pub fn cumulative_exhaustion_count(&self) -> u32 {
        self.cumulative_quota.consecutive_exhaustions
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

// =============================================================================
// Budget Pacer - EV-aware API budget allocation
// =============================================================================

/// Priority levels for API operations.
///
/// Higher priority operations get access to more of the budget.
/// Emergency operations always succeed. Low-value operations are
/// deprioritized when budget is tight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationPriority {
    /// Emergency: cancel during cascade, close positions on kill switch.
    /// Always allowed regardless of budget.
    Emergency,

    /// High-value: fills, response to significant price moves.
    /// Allowed when budget > reserve threshold.
    HighValue,

    /// Normal: regular quote updates, standard modifications.
    /// Allowed when budget > 2x reserve threshold.
    Normal,

    /// Low-value: small price drift corrections, minor size adjustments.
    /// Allowed when budget > 3x reserve threshold.
    LowValue,
}

/// Configuration for budget pacer.
#[derive(Debug, Clone)]
pub struct BudgetPacerConfig {
    /// API budget per minute (subset of total IP weight limit).
    /// Default: 600 (half of 1200 limit - leaves room for variability).
    pub api_budget_per_minute: u32,

    /// Fraction of budget reserved for high-value operations.
    /// Default: 0.2 (20% reserved for emergencies and fills).
    pub high_value_reserve: f64,

    /// Enable budget pacing. When disabled, all operations are allowed.
    /// Default: true.
    pub enabled: bool,
}

impl Default for BudgetPacerConfig {
    fn default() -> Self {
        Self {
            api_budget_per_minute: 600, // Half of 1200 limit
            high_value_reserve: 0.2,    // 20% reserved
            enabled: true,
        }
    }
}

/// Budget-aware API call pacer.
///
/// Allocates API budget to highest-value operations first.
/// This prevents low-value operations (small price corrections) from
/// consuming budget that should be reserved for high-value operations
/// (fills, emergency cancels).
///
/// # Usage
/// ```ignore
/// let pacer = BudgetPacer::new(BudgetPacerConfig::default());
///
/// // Check before making API call
/// if pacer.should_spend(OperationPriority::Normal) {
///     pacer.record_spend(1);
///     // ... make API call ...
/// } else {
///     // Skip low-priority operation to conserve budget
/// }
/// ```
#[derive(Debug)]
pub struct BudgetPacer {
    config: BudgetPacerConfig,
    /// Rolling window of API calls with timestamps
    calls: Vec<Instant>,
}

impl Default for BudgetPacer {
    fn default() -> Self {
        Self::new(BudgetPacerConfig::default())
    }
}

impl BudgetPacer {
    /// Create a new budget pacer with the given configuration.
    pub fn new(config: BudgetPacerConfig) -> Self {
        let capacity = config.api_budget_per_minute as usize;
        Self {
            config,
            calls: Vec::with_capacity(capacity),
        }
    }

    /// Get the number of API calls made in the last minute.
    pub fn spent_this_minute(&self) -> u32 {
        let now = Instant::now();
        let cutoff = now - Duration::from_secs(60);
        self.calls.iter().filter(|&&t| t > cutoff).count() as u32
    }

    /// Get the remaining budget for this minute.
    pub fn remaining_budget(&self) -> u32 {
        self.config
            .api_budget_per_minute
            .saturating_sub(self.spent_this_minute())
    }

    /// Get the reserve threshold (minimum budget to maintain for high-value ops).
    fn reserve_threshold(&self) -> u32 {
        (self.config.api_budget_per_minute as f64 * self.config.high_value_reserve) as u32
    }

    /// Check if an operation at the given priority should be allowed.
    ///
    /// Returns true if the budget allows this operation, false if it should be skipped.
    ///
    /// # Priority Thresholds
    /// - Emergency: Always allowed
    /// - HighValue: Requires > 1x reserve threshold
    /// - Normal: Requires > 2x reserve threshold
    /// - LowValue: Requires > 3x reserve threshold
    pub fn should_spend(&self, priority: OperationPriority) -> bool {
        if !self.config.enabled {
            return true;
        }

        let remaining = self.remaining_budget();
        let reserve = self.reserve_threshold();

        match priority {
            OperationPriority::Emergency => true,
            OperationPriority::HighValue => remaining > reserve,
            OperationPriority::Normal => remaining > reserve * 2,
            OperationPriority::LowValue => remaining > reserve * 3,
        }
    }

    /// Record an API call (call after making the call).
    pub fn record_spend(&mut self, count: u32) {
        let now = Instant::now();
        for _ in 0..count {
            self.calls.push(now);
        }

        // Prune old entries periodically (when > 2x budget)
        if self.calls.len() > self.config.api_budget_per_minute as usize * 2 {
            let cutoff = now - Duration::from_secs(60);
            self.calls.retain(|&t| t > cutoff);
        }
    }

    /// Get metrics for logging/monitoring.
    pub fn get_metrics(&self) -> BudgetPacerMetrics {
        BudgetPacerMetrics {
            budget_per_minute: self.config.api_budget_per_minute,
            spent_this_minute: self.spent_this_minute(),
            remaining: self.remaining_budget(),
            reserve_threshold: self.reserve_threshold(),
            enabled: self.config.enabled,
        }
    }

    /// Determine the appropriate priority for a price modification.
    ///
    /// Uses price drift (in bps) to categorize the operation:
    /// - Large moves (> 20 bps): HighValue - price discovery is important
    /// - Medium moves (10-20 bps): Normal - standard quote maintenance
    /// - Small moves (< 10 bps): LowValue - can be skipped if budget is tight
    pub fn priority_for_price_drift(price_drift_bps: f64) -> OperationPriority {
        if price_drift_bps > 20.0 {
            OperationPriority::HighValue
        } else if price_drift_bps > 10.0 {
            OperationPriority::Normal
        } else {
            OperationPriority::LowValue
        }
    }

    /// Determine the appropriate priority for a new order placement.
    ///
    /// New placements to cover gaps are important for market making.
    pub fn priority_for_placement(is_best_level: bool) -> OperationPriority {
        if is_best_level {
            OperationPriority::HighValue // Best bid/ask must be covered
        } else {
            OperationPriority::Normal // Outer levels are less critical
        }
    }

    /// Determine the appropriate priority for a cancel operation.
    pub fn priority_for_cancel(is_emergency: bool, is_stale: bool) -> OperationPriority {
        if is_emergency {
            OperationPriority::Emergency
        } else if is_stale {
            OperationPriority::Normal // Clean up stale orders
        } else {
            OperationPriority::LowValue // Routine cleanup
        }
    }
}

/// Metrics from budget pacer.
#[derive(Debug, Clone)]
pub struct BudgetPacerMetrics {
    pub budget_per_minute: u32,
    pub spent_this_minute: u32,
    pub remaining: u32,
    pub reserve_threshold: u32,
    pub enabled: bool,
}
