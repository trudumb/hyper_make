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
        }
    }
}
