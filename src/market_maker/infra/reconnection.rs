//! WebSocket Reconnection and Health Monitoring
//!
//! Provides robust reconnection logic with:
//! - **Exponential Backoff**: Increasing delays between reconnection attempts
//! - **Jitter**: Random variation to prevent thundering herd
//! - **Health Tracking**: Monitor connection state and detect stale data
//! - **Configurable Limits**: Max retries, timeout thresholds

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for WebSocket reconnection behavior.
#[derive(Debug, Clone)]
pub struct ReconnectionConfig {
    /// Initial delay before first reconnection attempt (default: 1s)
    pub initial_delay: Duration,
    /// Maximum delay between reconnection attempts (default: 60s)
    pub max_delay: Duration,
    /// Backoff multiplier (default: 2.0)
    pub backoff_multiplier: f64,
    /// Jitter factor (0.0-1.0, default: 0.2 = ±20%)
    pub jitter_factor: f64,
    /// Maximum consecutive failures before giving up (0 = unlimited)
    pub max_consecutive_failures: u32,
    /// Threshold for considering data stale (default: 30s)
    pub stale_data_threshold: Duration,
    /// Interval for health checks (default: 5s)
    pub health_check_interval: Duration,
}

impl Default for ReconnectionConfig {
    fn default() -> Self {
        Self {
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            backoff_multiplier: 2.0,
            jitter_factor: 0.2,
            max_consecutive_failures: 10,
            stale_data_threshold: Duration::from_secs(30),
            health_check_interval: Duration::from_secs(5),
        }
    }
}

/// Connection health state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Connected and receiving data
    Healthy,
    /// Connected but data may be stale
    Stale,
    /// Currently reconnecting
    Reconnecting,
    /// Disconnected, not attempting reconnection
    Disconnected,
    /// Failed after max retries
    Failed,
}

impl std::fmt::Display for ConnectionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectionState::Healthy => write!(f, "healthy"),
            ConnectionState::Stale => write!(f, "stale"),
            ConnectionState::Reconnecting => write!(f, "reconnecting"),
            ConnectionState::Disconnected => write!(f, "disconnected"),
            ConnectionState::Failed => write!(f, "failed"),
        }
    }
}

/// WebSocket connection health monitor.
///
/// Tracks connection state, data freshness, and reconnection attempts.
/// Thread-safe for use across async tasks.
#[derive(Clone)]
pub struct ConnectionHealthMonitor {
    inner: Arc<HealthMonitorInner>,
}

struct HealthMonitorInner {
    config: ReconnectionConfig,
    /// Timestamp of last received data (nanos since start)
    last_data_time: AtomicU64,
    /// Consecutive reconnection failures
    consecutive_failures: AtomicU64,
    /// Current reconnection attempt (0 = not reconnecting)
    current_attempt: AtomicU64,
    /// Whether we're currently reconnecting
    is_reconnecting: AtomicBool,
    /// Whether connection has failed permanently
    has_failed: AtomicBool,
    /// Start time for calculating durations
    start_time: Instant,
}

impl ConnectionHealthMonitor {
    /// Create a new health monitor with default config.
    pub fn new() -> Self {
        Self::with_config(ReconnectionConfig::default())
    }

    /// Create a new health monitor with custom config.
    pub fn with_config(config: ReconnectionConfig) -> Self {
        Self {
            inner: Arc::new(HealthMonitorInner {
                config,
                last_data_time: AtomicU64::new(0),
                consecutive_failures: AtomicU64::new(0),
                current_attempt: AtomicU64::new(0),
                is_reconnecting: AtomicBool::new(false),
                has_failed: AtomicBool::new(false),
                start_time: Instant::now(),
            }),
        }
    }

    /// Record that data was received.
    ///
    /// Call this whenever a message is received from the WebSocket.
    pub fn record_data_received(&self) {
        let nanos = self.inner.start_time.elapsed().as_nanos() as u64;
        self.inner.last_data_time.store(nanos, Ordering::Relaxed);
        // Successful data resets failure count
        if self.inner.consecutive_failures.load(Ordering::Relaxed) > 0 {
            self.inner.consecutive_failures.store(0, Ordering::Relaxed);
        }
    }

    /// Get time since last data was received.
    pub fn time_since_last_data(&self) -> Duration {
        let last_nanos = self.inner.last_data_time.load(Ordering::Relaxed);
        if last_nanos == 0 {
            // Never received data - return time since creation
            self.inner.start_time.elapsed()
        } else {
            let current_nanos = self.inner.start_time.elapsed().as_nanos() as u64;
            Duration::from_nanos(current_nanos.saturating_sub(last_nanos))
        }
    }

    /// Check if data is considered stale.
    pub fn is_data_stale(&self) -> bool {
        self.time_since_last_data() > self.inner.config.stale_data_threshold
    }

    /// Get current connection state.
    pub fn state(&self) -> ConnectionState {
        if self.inner.has_failed.load(Ordering::Relaxed) {
            return ConnectionState::Failed;
        }
        if self.inner.is_reconnecting.load(Ordering::Relaxed) {
            return ConnectionState::Reconnecting;
        }
        if self.is_data_stale() {
            return ConnectionState::Stale;
        }
        let last_data = self.inner.last_data_time.load(Ordering::Relaxed);
        if last_data == 0 {
            // Never connected
            return ConnectionState::Disconnected;
        }
        ConnectionState::Healthy
    }

    /// Record start of reconnection attempt.
    pub fn start_reconnection(&self) {
        self.inner.is_reconnecting.store(true, Ordering::Relaxed);
        self.inner.current_attempt.fetch_add(1, Ordering::Relaxed);
    }

    /// Record successful reconnection.
    pub fn reconnection_success(&self) {
        self.inner.is_reconnecting.store(false, Ordering::Relaxed);
        self.inner.current_attempt.store(0, Ordering::Relaxed);
        self.inner.consecutive_failures.store(0, Ordering::Relaxed);
        self.record_data_received();
    }

    /// Record failed reconnection attempt.
    ///
    /// Returns `true` if should continue trying, `false` if max retries exceeded.
    pub fn reconnection_failed(&self) -> bool {
        let failures = self
            .inner
            .consecutive_failures
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        let max = self.inner.config.max_consecutive_failures;

        if max > 0 && failures >= max as u64 {
            self.inner.has_failed.store(true, Ordering::Relaxed);
            self.inner.is_reconnecting.store(false, Ordering::Relaxed);
            return false;
        }
        true
    }

    /// Calculate delay before next reconnection attempt.
    ///
    /// Uses exponential backoff with jitter.
    pub fn next_reconnection_delay(&self) -> Duration {
        let attempt = self.inner.current_attempt.load(Ordering::Relaxed) as u32;
        calculate_backoff_delay(
            attempt,
            self.inner.config.initial_delay,
            self.inner.config.max_delay,
            self.inner.config.backoff_multiplier,
            self.inner.config.jitter_factor,
        )
    }

    /// Get number of consecutive failures.
    pub fn consecutive_failures(&self) -> u64 {
        self.inner.consecutive_failures.load(Ordering::Relaxed)
    }

    /// Get current reconnection attempt number.
    pub fn current_attempt(&self) -> u64 {
        self.inner.current_attempt.load(Ordering::Relaxed)
    }

    /// Reset the monitor state (e.g., after manual reconnect).
    pub fn reset(&self) {
        self.inner.consecutive_failures.store(0, Ordering::Relaxed);
        self.inner.current_attempt.store(0, Ordering::Relaxed);
        self.inner.is_reconnecting.store(false, Ordering::Relaxed);
        self.inner.has_failed.store(false, Ordering::Relaxed);
    }

    /// Get health summary for monitoring.
    pub fn summary(&self) -> HealthSummary {
        HealthSummary {
            state: self.state(),
            time_since_last_data_secs: self.time_since_last_data().as_secs_f64(),
            consecutive_failures: self.consecutive_failures(),
            current_attempt: self.current_attempt(),
            is_data_stale: self.is_data_stale(),
        }
    }
}

impl Default for ConnectionHealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of connection health.
#[derive(Debug, Clone)]
pub struct HealthSummary {
    pub state: ConnectionState,
    pub time_since_last_data_secs: f64,
    pub consecutive_failures: u64,
    pub current_attempt: u64,
    pub is_data_stale: bool,
}

/// Calculate exponential backoff delay with jitter.
///
/// # Arguments
/// - `attempt`: Current attempt number (0-based)
/// - `initial`: Initial delay
/// - `max`: Maximum delay cap
/// - `multiplier`: Backoff multiplier (typically 2.0)
/// - `jitter`: Jitter factor (0.0-1.0, e.g., 0.2 for ±20%)
pub fn calculate_backoff_delay(
    attempt: u32,
    initial: Duration,
    max: Duration,
    multiplier: f64,
    jitter: f64,
) -> Duration {
    // Calculate base delay: initial * multiplier^attempt
    let base_secs = initial.as_secs_f64() * multiplier.powi(attempt as i32);
    let capped_secs = base_secs.min(max.as_secs_f64());

    // Add jitter: delay * (1 ± jitter)
    // Use a simple deterministic "random" based on attempt for reproducibility
    let jitter_factor = if attempt.is_multiple_of(2) {
        1.0 + jitter * 0.5
    } else {
        1.0 - jitter * 0.5
    };
    let jittered_secs = capped_secs * jitter_factor;

    Duration::from_secs_f64(jittered_secs.max(0.0))
}

/// Helper to run reconnection logic with proper backoff.
///
/// # Arguments
/// - `monitor`: Health monitor to track state
/// - `connect_fn`: Async function that attempts connection
///
/// # Returns
/// `Ok(())` if connection succeeded, `Err` if all retries exhausted
pub async fn reconnect_with_backoff<F, Fut, E>(
    monitor: &ConnectionHealthMonitor,
    mut connect_fn: F,
) -> std::result::Result<(), E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = std::result::Result<(), E>>,
{
    monitor.start_reconnection();

    loop {
        let delay = monitor.next_reconnection_delay();
        tokio::time::sleep(delay).await;

        match connect_fn().await {
            Ok(()) => {
                monitor.reconnection_success();
                return Ok(());
            }
            Err(e) => {
                if !monitor.reconnection_failed() {
                    // Max retries exceeded
                    return Err(e);
                }
                monitor.start_reconnection(); // Increment attempt counter
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ReconnectionConfig::default();
        assert_eq!(config.initial_delay, Duration::from_secs(1));
        assert_eq!(config.max_delay, Duration::from_secs(60));
        assert_eq!(config.backoff_multiplier, 2.0);
    }

    #[test]
    fn test_backoff_calculation() {
        let initial = Duration::from_secs(1);
        let max = Duration::from_secs(60);

        // Attempt 0: 1s
        let d0 = calculate_backoff_delay(0, initial, max, 2.0, 0.0);
        assert!((d0.as_secs_f64() - 1.0).abs() < 0.01);

        // Attempt 1: 2s
        let d1 = calculate_backoff_delay(1, initial, max, 2.0, 0.0);
        assert!((d1.as_secs_f64() - 2.0).abs() < 0.01);

        // Attempt 2: 4s
        let d2 = calculate_backoff_delay(2, initial, max, 2.0, 0.0);
        assert!((d2.as_secs_f64() - 4.0).abs() < 0.01);

        // Attempt 6: 64s -> capped to 60s
        let d6 = calculate_backoff_delay(6, initial, max, 2.0, 0.0);
        assert!((d6.as_secs_f64() - 60.0).abs() < 0.01);
    }

    #[test]
    fn test_backoff_with_jitter() {
        let initial = Duration::from_secs(1);
        let max = Duration::from_secs(60);

        // With 20% jitter, delay should be within ±10% of base
        let d0 = calculate_backoff_delay(0, initial, max, 2.0, 0.2);
        assert!(d0.as_secs_f64() >= 0.8 && d0.as_secs_f64() <= 1.2);
    }

    #[test]
    fn test_health_monitor_creation() {
        let monitor = ConnectionHealthMonitor::new();
        assert_eq!(monitor.state(), ConnectionState::Disconnected);
        assert_eq!(monitor.consecutive_failures(), 0);
    }

    #[test]
    fn test_data_received() {
        let monitor = ConnectionHealthMonitor::new();

        // Initially no data
        assert!(monitor.time_since_last_data() > Duration::ZERO);

        // Record data
        monitor.record_data_received();

        // Now we have recent data
        assert!(monitor.time_since_last_data() < Duration::from_millis(100));
        assert_eq!(monitor.state(), ConnectionState::Healthy);
    }

    #[test]
    fn test_stale_detection() {
        let config = ReconnectionConfig {
            stale_data_threshold: Duration::from_millis(10),
            ..Default::default()
        };
        let monitor = ConnectionHealthMonitor::with_config(config);

        monitor.record_data_received();
        assert!(!monitor.is_data_stale());

        // Wait for data to become stale
        std::thread::sleep(Duration::from_millis(20));
        assert!(monitor.is_data_stale());
        assert_eq!(monitor.state(), ConnectionState::Stale);
    }

    #[test]
    fn test_reconnection_tracking() {
        let monitor = ConnectionHealthMonitor::new();

        // Start reconnecting
        monitor.start_reconnection();
        assert_eq!(monitor.state(), ConnectionState::Reconnecting);
        assert_eq!(monitor.current_attempt(), 1);

        // Fail first attempt
        assert!(monitor.reconnection_failed()); // Should continue
        assert_eq!(monitor.consecutive_failures(), 1);

        // Start second attempt
        monitor.start_reconnection();
        assert_eq!(monitor.current_attempt(), 2);

        // Succeed
        monitor.reconnection_success();
        assert_eq!(monitor.consecutive_failures(), 0);
        assert_eq!(monitor.current_attempt(), 0);
        assert_eq!(monitor.state(), ConnectionState::Healthy);
    }

    #[test]
    fn test_max_failures() {
        let config = ReconnectionConfig {
            max_consecutive_failures: 3,
            ..Default::default()
        };
        let monitor = ConnectionHealthMonitor::with_config(config);

        monitor.start_reconnection();
        assert!(monitor.reconnection_failed()); // 1
        assert!(monitor.reconnection_failed()); // 2
        assert!(!monitor.reconnection_failed()); // 3 -> exceeded

        assert_eq!(monitor.state(), ConnectionState::Failed);
    }

    #[test]
    fn test_reset() {
        let config = ReconnectionConfig {
            max_consecutive_failures: 3,
            ..Default::default()
        };
        let monitor = ConnectionHealthMonitor::with_config(config);

        // Get to failed state
        monitor.start_reconnection();
        monitor.reconnection_failed();
        monitor.reconnection_failed();
        monitor.reconnection_failed();
        assert_eq!(monitor.state(), ConnectionState::Failed);

        // Reset
        monitor.reset();
        assert_eq!(monitor.consecutive_failures(), 0);
        // After reset, state depends on data freshness
        // (since we never recorded data, it's Disconnected)
        assert_eq!(monitor.state(), ConnectionState::Disconnected);
    }

    #[test]
    fn test_summary() {
        let monitor = ConnectionHealthMonitor::new();
        monitor.record_data_received();

        let summary = monitor.summary();
        assert_eq!(summary.state, ConnectionState::Healthy);
        assert!(summary.time_since_last_data_secs < 0.1);
        assert_eq!(summary.consecutive_failures, 0);
        assert!(!summary.is_data_stale);
    }

    #[test]
    fn test_connection_state_display() {
        assert_eq!(format!("{}", ConnectionState::Healthy), "healthy");
        assert_eq!(format!("{}", ConnectionState::Stale), "stale");
        assert_eq!(format!("{}", ConnectionState::Reconnecting), "reconnecting");
        assert_eq!(format!("{}", ConnectionState::Failed), "failed");
    }
}
