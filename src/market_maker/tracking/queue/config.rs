//! Queue position tracking configuration.

/// Configuration for queue position tracking.
#[derive(Debug, Clone)]
pub struct QueueConfig {
    /// Decay rate for queue position from cancels (fraction per second)
    /// Typical: 0.1-0.3 (10-30% of queue cancels per second)
    pub cancel_decay_rate: f64,

    /// Expected volume at touch per second (in asset units)
    /// Used for execution probability calculation
    pub expected_volume_per_second: f64,

    /// Minimum queue position (floor to prevent division issues)
    pub min_queue_position: f64,

    /// Default queue position when we can't estimate (conservative)
    pub default_queue_position: f64,

    /// Fill probability threshold below which we should consider refreshing
    pub refresh_threshold: f64,

    /// Minimum time before considering a refresh (seconds)
    /// Prevents thrashing on fast markets
    pub min_order_age_for_refresh: f64,

    /// EV cost of refreshing in basis points.
    /// Refresh is only approved if: EV(new_position) - EV(current) > refresh_cost_bps
    /// This accounts for queue loss and API call cost.
    /// Default: 5.0 bps (queue loss ~3-5 bps + API cost ~1-2 bps)
    pub refresh_cost_bps: f64,

    /// Spread capture estimate in bps for EV calculation.
    /// Used to convert P(fill) to expected value: EV = P(fill) * spread_capture_bps
    /// Default: 8.0 bps (half-spread capture on typical MM trade)
    pub spread_capture_bps: f64,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            cancel_decay_rate: 0.2,          // 20% decay per second
            expected_volume_per_second: 1.0, // 1 unit per second at touch
            min_queue_position: 0.01,
            default_queue_position: 10.0,
            refresh_threshold: 0.1,         // Refresh if P(fill) < 10%
            min_order_age_for_refresh: 0.5, // Wait 500ms before refresh
            refresh_cost_bps: 5.0,          // EV cost of refresh (queue loss + API)
            spread_capture_bps: 8.0,        // Expected spread capture per fill
        }
    }
}
