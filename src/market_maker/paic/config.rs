//! PAIC Framework configuration.

/// Configuration for the PAIC framework.
#[derive(Debug, Clone)]
pub struct PAICConfig {
    // === Observer Layer ===
    /// Volatility regime thresholds (relative to baseline σ)
    pub volatility_quiet_threshold: f64,
    pub volatility_turbulent_threshold: f64,

    /// Flow toxicity threshold (VPIN/OFI based)
    pub toxicity_threshold: f64,

    /// Queue tracking config
    pub queue_config: VirtualQueueConfig,

    // === Controller Layer ===
    /// Minimum drift (in bps) before considering action
    pub min_drift_bps: f64,

    /// Priority premium multiplier (scales with spread)
    pub priority_premium_multiplier: f64,

    /// Threshold for "high priority" (π < this = high priority)
    pub high_priority_threshold: f64,

    /// Leak size reduction factor (reduce to this fraction of size)
    pub leak_size_factor: f64,

    // === Executor Layer ===
    pub rate_limit_config: RateLimitConfig,

    /// Batch window (microseconds)
    pub batch_window_us: u64,

    /// Maximum orders per batch
    pub max_batch_size: usize,
}

impl Default for PAICConfig {
    fn default() -> Self {
        Self {
            // Observer
            volatility_quiet_threshold: 0.5,    // σ < 0.5 × baseline = quiet
            volatility_turbulent_threshold: 2.0, // σ > 2.0 × baseline = turbulent
            toxicity_threshold: 0.1,            // Toxicity > 0.1 = toxic flow

            queue_config: VirtualQueueConfig::default(),

            // Controller
            min_drift_bps: 1.0,              // 1 bps minimum drift
            priority_premium_multiplier: 0.8, // 80% of spread as max premium
            high_priority_threshold: 0.3,    // π < 0.3 = high priority
            leak_size_factor: 0.5,           // Reduce to 50% of size

            // Executor
            rate_limit_config: RateLimitConfig::default(),
            batch_window_us: 5_000, // 5ms batch window
            max_batch_size: 4,      // Max 4 orders per batch
        }
    }
}

/// Configuration for virtual queue tracking.
#[derive(Debug, Clone)]
pub struct VirtualQueueConfig {
    /// Half-life for cumulative volume decay (seconds)
    pub volume_decay_half_life_secs: f64,

    /// Minimum queue position (floor to prevent division issues)
    pub min_queue_position: f64,

    /// Window for tracking volume at level (milliseconds)
    pub volume_window_ms: u64,

    /// EWMA alpha for priority index smoothing
    pub priority_ewma_alpha: f64,
}

impl Default for VirtualQueueConfig {
    fn default() -> Self {
        Self {
            volume_decay_half_life_secs: 30.0, // 30 second half-life
            min_queue_position: 0.01,          // Floor at 1%
            volume_window_ms: 60_000,          // 60 second window
            priority_ewma_alpha: 0.1,          // Smooth priority updates
        }
    }
}

/// Configuration for rate limit shadow pricing.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum tokens (requests per minute capacity)
    pub max_tokens: u64,

    /// Token refill rate (tokens per second)
    pub refill_rate: f64,

    /// Shadow price exponent (cost = (max/current)^exponent)
    pub shadow_price_exponent: f64,

    /// Minimum importance threshold to consume tokens
    pub min_importance: f64,

    /// Reserve ratio (fraction of tokens to keep in reserve)
    pub reserve_ratio: f64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_tokens: 10_000,        // Hyperliquid limit
            refill_rate: 166.67,       // 10,000 per minute = 166.67/s
            shadow_price_exponent: 2.0, // Quadratic cost curve
            min_importance: 0.1,       // Minimum importance to trade
            reserve_ratio: 0.2,        // Keep 20% in reserve
        }
    }
}
