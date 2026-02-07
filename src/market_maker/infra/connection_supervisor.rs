//! Connection Supervisor - Proactive WebSocket Health Monitoring
//!
//! Provides application-level supervision of WebSocket connections:
//! - Stale data detection based on message flow
//! - Proactive reconnection signaling
//! - Connection quality tracking for metrics
//! - Integration with ConnectionHealthMonitor for state tracking
//!
//! This complements WsManager's ping/pong-based detection with
//! higher-level application logic that understands market data flow.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::reconnection::{ConnectionHealthMonitor, ConnectionState, ReconnectionConfig};

/// Configuration for connection supervision.
#[derive(Debug, Clone)]
pub struct SupervisorConfig {
    /// How often to check connection health (default: 5s)
    pub check_interval: Duration,
    /// Threshold for market data staleness (default: 10s)
    /// This is aggressive - if no market updates in 10s, something is wrong
    pub market_data_stale_threshold: Duration,
    /// Threshold for user event staleness (default: 60s)
    /// User events are less frequent, so more tolerance
    pub user_event_stale_threshold: Duration,
    /// Enable automatic reconnection signaling (default: true)
    pub auto_reconnect_signal: bool,
    /// Number of stale checks before signaling reconnect (default: 2)
    /// Requires multiple consecutive stale readings to avoid false positives
    pub stale_count_threshold: u32,
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(5),
            market_data_stale_threshold: Duration::from_secs(10),
            user_event_stale_threshold: Duration::from_secs(60),
            auto_reconnect_signal: true,
            stale_count_threshold: 2,
        }
    }
}

/// Events that the supervisor can signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupervisorEvent {
    /// Connection is healthy
    Healthy,
    /// Market data is becoming stale
    MarketDataStale,
    /// User events are becoming stale
    UserEventStale,
    /// Recommending reconnection
    ReconnectRecommended,
    /// Connection has been stale too long, critical
    Critical,
}

/// Statistics about connection supervision.
#[derive(Debug, Clone)]
pub struct SupervisorStats {
    /// Time since last market data update
    pub time_since_market_data: Duration,
    /// Time since last user event
    pub time_since_user_event: Duration,
    /// Number of times reconnection was signaled
    pub reconnect_signal_count: u64,
    /// Current consecutive stale count
    pub consecutive_stale_count: u32,
    /// Current supervisor assessment
    pub current_event: SupervisorEvent,
    /// Whether we're in a reconnecting state
    pub is_reconnecting: bool,
    /// Underlying connection state from health monitor
    pub connection_state: ConnectionState,
}

/// Connection Supervisor for proactive health monitoring.
///
/// Runs independently and monitors data flow to detect issues
/// before they become critical.
pub struct ConnectionSupervisor {
    config: SupervisorConfig,
    health_monitor: ConnectionHealthMonitor,
    inner: Arc<SupervisorInner>,
}

struct SupervisorInner {
    start_time: Instant,
    /// Last market data received (nanos since start)
    last_market_data_nanos: AtomicU64,
    /// Last user event received (nanos since start)
    last_user_event_nanos: AtomicU64,
    /// Consecutive stale readings
    consecutive_stale_count: AtomicU64,
    /// Total reconnect signals sent
    reconnect_signal_count: AtomicU64,
    /// Flag indicating reconnect is recommended
    reconnect_recommended: AtomicBool,
}

impl ConnectionSupervisor {
    /// Create a new connection supervisor with default config.
    pub fn new() -> Self {
        Self::with_config(SupervisorConfig::default())
    }

    /// Create a new connection supervisor with custom config.
    pub fn with_config(config: SupervisorConfig) -> Self {
        // Create a health monitor with matching stale threshold
        let reconnection_config = ReconnectionConfig {
            stale_data_threshold: config.market_data_stale_threshold,
            health_check_interval: config.check_interval,
            ..Default::default()
        };

        Self {
            config,
            health_monitor: ConnectionHealthMonitor::with_config(reconnection_config),
            inner: Arc::new(SupervisorInner {
                start_time: Instant::now(),
                last_market_data_nanos: AtomicU64::new(0),
                last_user_event_nanos: AtomicU64::new(0),
                consecutive_stale_count: AtomicU64::new(0),
                reconnect_signal_count: AtomicU64::new(0),
                reconnect_recommended: AtomicBool::new(false),
            }),
        }
    }

    /// Get a clone of the health monitor for external use.
    pub fn health_monitor(&self) -> ConnectionHealthMonitor {
        self.health_monitor.clone()
    }

    /// Record that market data was received (AllMids, Trades, L2Book).
    ///
    /// This is the critical function that confirms WebSocket health.
    /// When data arrives after a reconnection attempt, this automatically
    /// clears the reconnecting state to prevent kill switch false positives.
    pub fn record_market_data(&self) {
        // Check if we were reconnecting BEFORE recording data
        // (state check must happen before record_data_received updates timestamps)
        let was_reconnecting = self.health_monitor.state() == ConnectionState::Reconnecting;

        let nanos = self.inner.start_time.elapsed().as_nanos() as u64;
        self.inner
            .last_market_data_nanos
            .store(nanos, Ordering::Relaxed);
        self.health_monitor.record_data_received();

        // Reset stale count on data receipt
        self.inner
            .consecutive_stale_count
            .store(0, Ordering::Relaxed);
        self.inner
            .reconnect_recommended
            .store(false, Ordering::Relaxed);

        // CRITICAL FIX: Auto-clear reconnection state when data resumes
        // This prevents the attempt counter from accumulating across reconnection
        // cycles, which would eventually trigger the kill switch erroneously.
        if was_reconnecting {
            self.health_monitor.reconnection_success();
        }
    }

    /// Record that a user event was received (UserFills, OrderUpdates).
    pub fn record_user_event(&self) {
        let nanos = self.inner.start_time.elapsed().as_nanos() as u64;
        self.inner
            .last_user_event_nanos
            .store(nanos, Ordering::Relaxed);
        self.health_monitor.record_data_received();
    }

    /// Get time since last market data.
    pub fn time_since_market_data(&self) -> Duration {
        let last_nanos = self.inner.last_market_data_nanos.load(Ordering::Relaxed);
        if last_nanos == 0 {
            self.inner.start_time.elapsed()
        } else {
            let now_nanos = self.inner.start_time.elapsed().as_nanos() as u64;
            Duration::from_nanos(now_nanos.saturating_sub(last_nanos))
        }
    }

    /// Get time since last user event.
    pub fn time_since_user_event(&self) -> Duration {
        let last_nanos = self.inner.last_user_event_nanos.load(Ordering::Relaxed);
        if last_nanos == 0 {
            self.inner.start_time.elapsed()
        } else {
            let now_nanos = self.inner.start_time.elapsed().as_nanos() as u64;
            Duration::from_nanos(now_nanos.saturating_sub(last_nanos))
        }
    }

    /// Check connection health and return current event.
    ///
    /// Call this periodically (at check_interval) to monitor health.
    pub fn check_health(&self) -> SupervisorEvent {
        let time_since_market = self.time_since_market_data();
        let time_since_user = self.time_since_user_event();

        // Check for market data staleness (most critical)
        let market_stale = time_since_market > self.config.market_data_stale_threshold;
        let user_stale = time_since_user > self.config.user_event_stale_threshold;

        if market_stale {
            let stale_count = self
                .inner
                .consecutive_stale_count
                .fetch_add(1, Ordering::Relaxed)
                + 1;

            if stale_count >= self.config.stale_count_threshold as u64 {
                if self.config.auto_reconnect_signal {
                    self.inner
                        .reconnect_recommended
                        .store(true, Ordering::Relaxed);
                    self.inner
                        .reconnect_signal_count
                        .fetch_add(1, Ordering::Relaxed);
                }

                // Check if critically stale (3x threshold)
                if time_since_market > self.config.market_data_stale_threshold * 3 {
                    return SupervisorEvent::Critical;
                }

                return SupervisorEvent::ReconnectRecommended;
            }

            return SupervisorEvent::MarketDataStale;
        }

        if user_stale {
            return SupervisorEvent::UserEventStale;
        }

        SupervisorEvent::Healthy
    }

    /// Check if reconnection is recommended.
    pub fn is_reconnect_recommended(&self) -> bool {
        self.inner.reconnect_recommended.load(Ordering::Relaxed)
    }

    /// Clear reconnection recommendation (after reconnection is initiated).
    pub fn clear_reconnect_recommendation(&self) {
        self.inner
            .reconnect_recommended
            .store(false, Ordering::Relaxed);
    }

    /// Record that reconnection was started.
    pub fn record_reconnection_start(&self) {
        self.health_monitor.start_reconnection();
    }

    /// Record that reconnection succeeded.
    pub fn record_reconnection_success(&self) {
        self.health_monitor.reconnection_success();
        self.inner
            .consecutive_stale_count
            .store(0, Ordering::Relaxed);
        self.inner
            .reconnect_recommended
            .store(false, Ordering::Relaxed);
    }

    /// Record that reconnection failed.
    pub fn record_reconnection_failed(&self) -> bool {
        self.health_monitor.reconnection_failed()
    }

    /// Get current statistics.
    pub fn stats(&self) -> SupervisorStats {
        SupervisorStats {
            time_since_market_data: self.time_since_market_data(),
            time_since_user_event: self.time_since_user_event(),
            reconnect_signal_count: self.inner.reconnect_signal_count.load(Ordering::Relaxed),
            consecutive_stale_count: self.inner.consecutive_stale_count.load(Ordering::Relaxed)
                as u32,
            current_event: self.check_health(),
            is_reconnecting: self.health_monitor.state() == ConnectionState::Reconnecting,
            connection_state: self.health_monitor.state(),
        }
    }

    /// Get underlying health monitor state.
    pub fn connection_state(&self) -> ConnectionState {
        self.health_monitor.state()
    }

    /// Check if connection is healthy.
    pub fn is_healthy(&self) -> bool {
        matches!(self.check_health(), SupervisorEvent::Healthy)
            && self.health_monitor.state() == ConnectionState::Healthy
    }
}

impl Default for ConnectionSupervisor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supervisor_creation() {
        let supervisor = ConnectionSupervisor::new();
        // Freshly created supervisor with no data is healthy initially
        // (elapsed time is 0, which is less than the 10 second stale threshold)
        assert_eq!(supervisor.check_health(), SupervisorEvent::Healthy);
        // But is_healthy() also checks connection_health state which starts as Disconnected
        // For a pure supervisor check without health monitor dependency:
        assert!(supervisor.time_since_market_data() < Duration::from_secs(1));
    }

    #[test]
    fn test_market_data_tracking() {
        let supervisor = ConnectionSupervisor::new();

        // Record market data
        supervisor.record_market_data();

        // Should be healthy now
        assert!(supervisor.time_since_market_data() < Duration::from_millis(100));
        assert_eq!(supervisor.check_health(), SupervisorEvent::Healthy);
    }

    #[test]
    fn test_user_event_tracking() {
        let supervisor = ConnectionSupervisor::new();

        // Record both types
        supervisor.record_market_data();
        supervisor.record_user_event();

        assert!(supervisor.time_since_user_event() < Duration::from_millis(100));
    }

    #[test]
    fn test_stale_detection() {
        let config = SupervisorConfig {
            market_data_stale_threshold: Duration::from_millis(10),
            stale_count_threshold: 1,
            ..Default::default()
        };
        let supervisor = ConnectionSupervisor::with_config(config);

        // Initially healthy (just created, elapsed time near 0)
        assert_eq!(supervisor.check_health(), SupervisorEvent::Healthy);

        // Record data
        supervisor.record_market_data();
        assert_eq!(supervisor.check_health(), SupervisorEvent::Healthy);

        // Wait for staleness
        std::thread::sleep(Duration::from_millis(20));
        assert!(matches!(
            supervisor.check_health(),
            SupervisorEvent::ReconnectRecommended | SupervisorEvent::MarketDataStale
        ));
    }

    #[test]
    fn test_reconnect_recommendation() {
        let config = SupervisorConfig {
            market_data_stale_threshold: Duration::from_millis(10),
            stale_count_threshold: 1,
            auto_reconnect_signal: true,
            ..Default::default()
        };
        let supervisor = ConnectionSupervisor::with_config(config);

        // Record data first so we have a baseline
        supervisor.record_market_data();

        // Wait for data to become stale
        std::thread::sleep(Duration::from_millis(20));

        // Check health to trigger recommendation
        supervisor.check_health();
        assert!(supervisor.is_reconnect_recommended());

        // Clear it
        supervisor.clear_reconnect_recommendation();
        assert!(!supervisor.is_reconnect_recommended());
    }

    #[test]
    fn test_stats() {
        let supervisor = ConnectionSupervisor::new();
        supervisor.record_market_data();

        let stats = supervisor.stats();
        assert!(stats.time_since_market_data < Duration::from_millis(100));
        assert_eq!(stats.consecutive_stale_count, 0);
    }
}
