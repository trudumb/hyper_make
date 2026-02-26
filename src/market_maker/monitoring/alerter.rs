//! Thread-safe alerting system with deduplication.
//!
//! Provides a comprehensive alerting framework for market making operations
//! with configurable thresholds, severity levels, and alert deduplication.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Severity level of an alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AlertSeverity {
    /// Informational - no action required
    Info,
    /// Warning - attention recommended
    Warning,
    /// Critical - action required
    Critical,
    /// Emergency - immediate action required, may trigger kill switch
    Emergency,
}

impl AlertSeverity {
    /// Get display string for the severity.
    pub fn as_str(&self) -> &'static str {
        match self {
            AlertSeverity::Info => "INFO",
            AlertSeverity::Warning => "WARN",
            AlertSeverity::Critical => "CRIT",
            AlertSeverity::Emergency => "EMRG",
        }
    }

    /// Check if this severity is actionable (Critical or higher).
    pub fn is_actionable(&self) -> bool {
        matches!(self, AlertSeverity::Critical | AlertSeverity::Emergency)
    }
}

/// Type of alert for categorization and deduplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlertType {
    /// Model calibration has degraded (IR below threshold)
    CalibrationDegraded,
    /// Drawdown approaching warning threshold
    DrawdownWarning,
    /// Drawdown at critical level
    DrawdownCritical,
    /// Circuit breaker has been triggered
    CircuitBreakerTriggered,
    /// Fill rate has collapsed (significantly below normal)
    FillRateCollapse,
    /// Latency spike detected
    LatencySpike,
    /// Position approaching limit
    PositionLimitApproaching,
    /// Model data is stale
    ModelStale,
    /// Connection to exchange lost
    ConnectionLost,
    /// Signal quality decaying (half-life below threshold)
    SignalDecaying,
    /// Insufficient samples for reliable calibration
    InsufficientSamples,
    /// Market regime has shifted
    RegimeShift,
    /// Information ratio dropped below 1.0 (model adding noise)
    InformationRatioBelowOne,
    /// Spread has blown up beyond acceptable range
    SpreadBlowup,
    /// Inventory imbalance approaching dangerous levels
    InventoryImbalance,
    /// Kill switch is armed (one or more monitors at HIGH severity)
    KillSwitchArmed,
    /// No fills received within expected timeframe
    NoFills,
}

impl AlertType {
    /// Get display name for the alert type.
    pub fn as_str(&self) -> &'static str {
        match self {
            AlertType::CalibrationDegraded => "CalibrationDegraded",
            AlertType::DrawdownWarning => "DrawdownWarning",
            AlertType::DrawdownCritical => "DrawdownCritical",
            AlertType::CircuitBreakerTriggered => "CircuitBreakerTriggered",
            AlertType::FillRateCollapse => "FillRateCollapse",
            AlertType::LatencySpike => "LatencySpike",
            AlertType::PositionLimitApproaching => "PositionLimitApproaching",
            AlertType::ModelStale => "ModelStale",
            AlertType::ConnectionLost => "ConnectionLost",
            AlertType::SignalDecaying => "SignalDecaying",
            AlertType::InsufficientSamples => "InsufficientSamples",
            AlertType::RegimeShift => "RegimeShift",
            AlertType::InformationRatioBelowOne => "InformationRatioBelowOne",
            AlertType::SpreadBlowup => "SpreadBlowup",
            AlertType::InventoryImbalance => "InventoryImbalance",
            AlertType::KillSwitchArmed => "KillSwitchArmed",
            AlertType::NoFills => "NoFills",
        }
    }
}

/// An alert instance.
#[derive(Debug, Clone)]
pub struct Alert {
    /// Unique identifier for this alert
    pub id: u64,
    /// Unix timestamp in milliseconds when alert was created
    pub timestamp: u64,
    /// Type of alert
    pub alert_type: AlertType,
    /// Severity level
    pub severity: AlertSeverity,
    /// Human-readable message
    pub message: String,
    /// Optional metric value that triggered the alert
    pub value: Option<f64>,
    /// Optional threshold that was breached
    pub threshold: Option<f64>,
    /// Whether the alert has been acknowledged
    pub acknowledged: bool,
}

impl Alert {
    /// Create a new alert.
    pub fn new(
        id: u64,
        timestamp: u64,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: impl Into<String>,
    ) -> Self {
        Self {
            id,
            timestamp,
            alert_type,
            severity,
            message: message.into(),
            value: None,
            threshold: None,
            acknowledged: false,
        }
    }

    /// Builder method to add metric value.
    pub fn with_value(mut self, value: f64) -> Self {
        self.value = Some(value);
        self
    }

    /// Builder method to add threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Format alert for display.
    pub fn format(&self) -> String {
        let value_str = self
            .value
            .map(|v| format!(" (value: {v:.4})"))
            .unwrap_or_default();
        let threshold_str = self
            .threshold
            .map(|t| format!(" (threshold: {t:.4})"))
            .unwrap_or_default();
        let ack_str = if self.acknowledged { " [ACK]" } else { "" };

        format!(
            "[{}] {} - {}: {}{}{}{}",
            format_timestamp_short(self.timestamp),
            self.severity.as_str(),
            self.alert_type.as_str(),
            self.message,
            value_str,
            threshold_str,
            ack_str,
        )
    }
}

/// Configuration for the alerter.
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// IR below this triggers warning (default: 1.0)
    pub ir_warning_threshold: f64,
    /// IR below this triggers critical (default: 0.8)
    pub ir_critical_threshold: f64,
    /// Drawdown percentage for warning (default: 0.01 = 1%)
    pub drawdown_warning_pct: f64,
    /// Drawdown percentage for critical (default: 0.02 = 2%)
    pub drawdown_critical_pct: f64,
    /// Fill rate warning when current is below this fraction of baseline (default: 0.5)
    pub fill_rate_warning_pct: f64,
    /// Latency warning threshold in ms (default: 50.0)
    pub latency_warning_ms: f64,
    /// Position warning when at this fraction of limit (default: 0.8 = 80%)
    pub position_warning_pct: f64,
    /// Spread warning threshold in bps (default: 20.0)
    pub spread_warning_bps: f64,
    /// Inventory imbalance warning when at this fraction of max (default: 0.7)
    pub inventory_warning_pct: f64,
    /// No-fill warning threshold in seconds (default: 300 = 5 min)
    pub no_fill_warning_s: u64,
    /// Signal MI threshold below which signal is considered degraded (default: 0.05)
    pub signal_mi_warning: f64,
    /// Deduplication window in seconds (default: 300 = 5 min)
    pub dedup_window_s: u64,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            ir_warning_threshold: 1.0,
            ir_critical_threshold: 0.8,
            drawdown_warning_pct: 0.01,
            drawdown_critical_pct: 0.02,
            fill_rate_warning_pct: 0.5,
            latency_warning_ms: 50.0,
            position_warning_pct: 0.8,
            spread_warning_bps: 20.0,
            inventory_warning_pct: 0.7,
            no_fill_warning_s: 300,
            signal_mi_warning: 0.05,
            dedup_window_s: 300,
        }
    }
}

/// Thread-safe alerter with deduplication.
///
/// The alerter maintains a history of alerts and provides methods to:
/// - Check various thresholds and create alerts
/// - Deduplicate alerts within a configurable time window
/// - Manage alert lifecycle (acknowledge, query)
///
/// # Thread Safety
///
/// The alerter is fully thread-safe and can be shared across threads.
/// All internal state is protected by appropriate synchronization primitives.
pub struct Alerter {
    config: AlertConfig,
    alerts: Arc<RwLock<VecDeque<Alert>>>,
    max_alerts: usize,
    next_id: AtomicU64,
    last_alert_time: Arc<RwLock<HashMap<AlertType, u64>>>,
}

impl Alerter {
    /// Create a new alerter.
    ///
    /// # Arguments
    ///
    /// * `config` - Alert configuration with thresholds
    /// * `max_alerts` - Maximum number of alerts to retain in history
    pub fn new(config: AlertConfig, max_alerts: usize) -> Self {
        Self {
            config,
            alerts: Arc::new(RwLock::new(VecDeque::with_capacity(max_alerts))),
            max_alerts,
            next_id: AtomicU64::new(1),
            last_alert_time: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get the next unique alert ID.
    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Check if an alert of this type should be deduplicated.
    fn should_dedupe(&self, alert_type: AlertType, timestamp: u64) -> bool {
        let last_times = self.last_alert_time.read().unwrap();
        if let Some(&last_time) = last_times.get(&alert_type) {
            let elapsed_s = (timestamp.saturating_sub(last_time)) / 1000;
            elapsed_s < self.config.dedup_window_s
        } else {
            false
        }
    }

    /// Record that an alert of this type was created.
    fn record_alert_time(&self, alert_type: AlertType, timestamp: u64) {
        let mut last_times = self.last_alert_time.write().unwrap();
        last_times.insert(alert_type, timestamp);
    }

    /// Check calibration and return alert if threshold breached.
    ///
    /// # Arguments
    ///
    /// * `ir` - Current information ratio
    /// * `component` - Name of the component (e.g., "fill_prob", "adverse_sel")
    /// * `timestamp` - Current timestamp in milliseconds
    pub fn check_calibration(&self, ir: f64, component: &str, timestamp: u64) -> Option<Alert> {
        if self.should_dedupe(AlertType::CalibrationDegraded, timestamp) {
            return None;
        }

        if ir < self.config.ir_critical_threshold {
            self.record_alert_time(AlertType::CalibrationDegraded, timestamp);
            Some(
                Alert::new(
                    self.next_id(),
                    timestamp,
                    AlertType::CalibrationDegraded,
                    AlertSeverity::Critical,
                    format!("Calibration critically degraded for {component}"),
                )
                .with_value(ir)
                .with_threshold(self.config.ir_critical_threshold),
            )
        } else if ir < self.config.ir_warning_threshold {
            self.record_alert_time(AlertType::CalibrationDegraded, timestamp);
            Some(
                Alert::new(
                    self.next_id(),
                    timestamp,
                    AlertType::CalibrationDegraded,
                    AlertSeverity::Warning,
                    format!("Calibration degraded for {component}"),
                )
                .with_value(ir)
                .with_threshold(self.config.ir_warning_threshold),
            )
        } else {
            None
        }
    }

    /// Check drawdown and return alert if threshold breached.
    ///
    /// # Arguments
    ///
    /// * `drawdown_pct` - Current drawdown as a fraction (0.01 = 1%)
    /// * `timestamp` - Current timestamp in milliseconds
    pub fn check_drawdown(&self, drawdown_pct: f64, timestamp: u64) -> Option<Alert> {
        if drawdown_pct >= self.config.drawdown_critical_pct {
            if self.should_dedupe(AlertType::DrawdownCritical, timestamp) {
                return None;
            }
            self.record_alert_time(AlertType::DrawdownCritical, timestamp);
            Some(
                Alert::new(
                    self.next_id(),
                    timestamp,
                    AlertType::DrawdownCritical,
                    AlertSeverity::Critical,
                    format!("Critical drawdown: {:.2}%", drawdown_pct * 100.0),
                )
                .with_value(drawdown_pct)
                .with_threshold(self.config.drawdown_critical_pct),
            )
        } else if drawdown_pct >= self.config.drawdown_warning_pct {
            if self.should_dedupe(AlertType::DrawdownWarning, timestamp) {
                return None;
            }
            self.record_alert_time(AlertType::DrawdownWarning, timestamp);
            Some(
                Alert::new(
                    self.next_id(),
                    timestamp,
                    AlertType::DrawdownWarning,
                    AlertSeverity::Warning,
                    format!("Drawdown warning: {:.2}%", drawdown_pct * 100.0),
                )
                .with_value(drawdown_pct)
                .with_threshold(self.config.drawdown_warning_pct),
            )
        } else {
            None
        }
    }

    /// Check fill rate and return alert if collapsed.
    ///
    /// # Arguments
    ///
    /// * `current` - Current fill rate
    /// * `baseline` - Normal/expected fill rate
    /// * `timestamp` - Current timestamp in milliseconds
    pub fn check_fill_rate(&self, current: f64, baseline: f64, timestamp: u64) -> Option<Alert> {
        if baseline <= 0.0 {
            return None;
        }

        let ratio = current / baseline;
        if ratio < self.config.fill_rate_warning_pct {
            if self.should_dedupe(AlertType::FillRateCollapse, timestamp) {
                return None;
            }
            self.record_alert_time(AlertType::FillRateCollapse, timestamp);
            Some(
                Alert::new(
                    self.next_id(),
                    timestamp,
                    AlertType::FillRateCollapse,
                    AlertSeverity::Warning,
                    format!("Fill rate collapse: {current:.1}/hr (baseline: {baseline:.1}/hr)"),
                )
                .with_value(ratio)
                .with_threshold(self.config.fill_rate_warning_pct),
            )
        } else {
            None
        }
    }

    /// Check latency and return alert if spiked.
    ///
    /// # Arguments
    ///
    /// * `latency_ms` - Current latency in milliseconds
    /// * `timestamp` - Current timestamp in milliseconds
    pub fn check_latency(&self, latency_ms: f64, timestamp: u64) -> Option<Alert> {
        if latency_ms > self.config.latency_warning_ms {
            if self.should_dedupe(AlertType::LatencySpike, timestamp) {
                return None;
            }
            self.record_alert_time(AlertType::LatencySpike, timestamp);
            Some(
                Alert::new(
                    self.next_id(),
                    timestamp,
                    AlertType::LatencySpike,
                    AlertSeverity::Warning,
                    format!("Latency spike: {latency_ms:.1}ms"),
                )
                .with_value(latency_ms)
                .with_threshold(self.config.latency_warning_ms),
            )
        } else {
            None
        }
    }

    /// Check position and return alert if approaching limit.
    ///
    /// # Arguments
    ///
    /// * `current` - Current position size (absolute)
    /// * `limit` - Position limit
    /// * `timestamp` - Current timestamp in milliseconds
    pub fn check_position(&self, current: f64, limit: f64, timestamp: u64) -> Option<Alert> {
        if limit <= 0.0 {
            return None;
        }

        let utilization = current.abs() / limit;
        if utilization >= self.config.position_warning_pct {
            if self.should_dedupe(AlertType::PositionLimitApproaching, timestamp) {
                return None;
            }
            self.record_alert_time(AlertType::PositionLimitApproaching, timestamp);
            Some(
                Alert::new(
                    self.next_id(),
                    timestamp,
                    AlertType::PositionLimitApproaching,
                    AlertSeverity::Warning,
                    format!(
                        "Position approaching limit: {:.2} / {:.2} ({:.0}%)",
                        current.abs(),
                        limit,
                        utilization * 100.0
                    ),
                )
                .with_value(utilization)
                .with_threshold(self.config.position_warning_pct),
            )
        } else {
            None
        }
    }

    /// Check spread and return alert if blown up.
    ///
    /// # Arguments
    ///
    /// * `spread_bps` - Current average spread in basis points
    /// * `timestamp` - Current timestamp in milliseconds
    pub fn check_spread(&self, spread_bps: f64, timestamp: u64) -> Option<Alert> {
        if spread_bps > self.config.spread_warning_bps {
            if self.should_dedupe(AlertType::SpreadBlowup, timestamp) {
                return None;
            }
            self.record_alert_time(AlertType::SpreadBlowup, timestamp);
            Some(
                Alert::new(
                    self.next_id(),
                    timestamp,
                    AlertType::SpreadBlowup,
                    AlertSeverity::Warning,
                    format!(
                        "Spread blow-up: {:.1} bps (threshold: {:.1} bps)",
                        spread_bps, self.config.spread_warning_bps
                    ),
                )
                .with_value(spread_bps)
                .with_threshold(self.config.spread_warning_bps),
            )
        } else {
            None
        }
    }

    /// Check inventory imbalance and return alert if approaching dangerous levels.
    ///
    /// # Arguments
    ///
    /// * `inventory_abs` - Absolute inventory size
    /// * `max_inventory` - Maximum allowed inventory
    /// * `timestamp` - Current timestamp in milliseconds
    pub fn check_inventory(
        &self,
        inventory_abs: f64,
        max_inventory: f64,
        timestamp: u64,
    ) -> Option<Alert> {
        if max_inventory <= 0.0 {
            return None;
        }
        let utilization = inventory_abs / max_inventory;
        if utilization >= self.config.inventory_warning_pct {
            if self.should_dedupe(AlertType::InventoryImbalance, timestamp) {
                return None;
            }
            self.record_alert_time(AlertType::InventoryImbalance, timestamp);
            Some(
                Alert::new(
                    self.next_id(),
                    timestamp,
                    AlertType::InventoryImbalance,
                    AlertSeverity::Warning,
                    format!(
                        "Inventory imbalance: {:.4} / {:.4} ({:.0}%)",
                        inventory_abs,
                        max_inventory,
                        utilization * 100.0
                    ),
                )
                .with_value(utilization)
                .with_threshold(self.config.inventory_warning_pct),
            )
        } else {
            None
        }
    }

    /// Check signal health (MI) and return alert if degraded.
    ///
    /// # Arguments
    ///
    /// * `mi` - Current mutual information value
    /// * `signal_name` - Name of the signal (e.g., "lead_lag")
    /// * `timestamp` - Current timestamp in milliseconds
    pub fn check_signal_health(&self, mi: f64, signal_name: &str, timestamp: u64) -> Option<Alert> {
        if mi < self.config.signal_mi_warning {
            if self.should_dedupe(AlertType::SignalDecaying, timestamp) {
                return None;
            }
            self.record_alert_time(AlertType::SignalDecaying, timestamp);
            Some(
                Alert::new(
                    self.next_id(),
                    timestamp,
                    AlertType::SignalDecaying,
                    AlertSeverity::Warning,
                    format!(
                        "Signal {} degraded: MI={:.4} < {:.4}",
                        signal_name, mi, self.config.signal_mi_warning
                    ),
                )
                .with_value(mi)
                .with_threshold(self.config.signal_mi_warning),
            )
        } else {
            None
        }
    }

    /// Check no-fill condition.
    ///
    /// # Arguments
    ///
    /// * `seconds_since_last_fill` - Time since last fill in seconds
    /// * `timestamp` - Current timestamp in milliseconds
    pub fn check_no_fills(&self, seconds_since_last_fill: u64, timestamp: u64) -> Option<Alert> {
        if seconds_since_last_fill >= self.config.no_fill_warning_s {
            if self.should_dedupe(AlertType::NoFills, timestamp) {
                return None;
            }
            self.record_alert_time(AlertType::NoFills, timestamp);
            Some(
                Alert::new(
                    self.next_id(),
                    timestamp,
                    AlertType::NoFills,
                    AlertSeverity::Warning,
                    format!("No fills for {seconds_since_last_fill} seconds"),
                )
                .with_value(seconds_since_last_fill as f64)
                .with_threshold(self.config.no_fill_warning_s as f64),
            )
        } else {
            None
        }
    }

    /// Alert that kill switch is armed (a monitor is at HIGH severity).
    ///
    /// # Arguments
    ///
    /// * `monitor_name` - Name of the monitor that is at HIGH
    /// * `timestamp` - Current timestamp in milliseconds
    pub fn check_kill_switch_armed(&self, monitor_name: &str, timestamp: u64) -> Option<Alert> {
        if self.should_dedupe(AlertType::KillSwitchArmed, timestamp) {
            return None;
        }
        self.record_alert_time(AlertType::KillSwitchArmed, timestamp);
        Some(Alert::new(
            self.next_id(),
            timestamp,
            AlertType::KillSwitchArmed,
            AlertSeverity::Critical,
            format!("Kill switch armed: {monitor_name} at HIGH severity"),
        ))
    }

    /// Create a manual alert.
    ///
    /// # Arguments
    ///
    /// * `alert_type` - Type of alert
    /// * `severity` - Severity level
    /// * `message` - Alert message
    /// * `timestamp` - Current timestamp in milliseconds
    pub fn create_alert(
        &self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: String,
        timestamp: u64,
    ) -> Alert {
        Alert::new(self.next_id(), timestamp, alert_type, severity, message)
    }

    /// Add an alert to the history.
    pub fn add_alert(&self, alert: Alert) {
        let mut alerts = self.alerts.write().unwrap();
        alerts.push_back(alert);

        // Trim to max size
        while alerts.len() > self.max_alerts {
            alerts.pop_front();
        }
    }

    /// Acknowledge an alert by ID.
    pub fn acknowledge(&self, alert_id: u64) {
        let mut alerts = self.alerts.write().unwrap();
        for alert in alerts.iter_mut() {
            if alert.id == alert_id {
                alert.acknowledged = true;
                break;
            }
        }
    }

    /// Get the most recent N alerts.
    pub fn recent_alerts(&self, n: usize) -> Vec<Alert> {
        let alerts = self.alerts.read().unwrap();
        alerts.iter().rev().take(n).cloned().collect()
    }

    /// Get all unacknowledged alerts.
    pub fn unacknowledged_alerts(&self) -> Vec<Alert> {
        let alerts = self.alerts.read().unwrap();
        alerts.iter().filter(|a| !a.acknowledged).cloned().collect()
    }

    /// Get alerts by severity.
    pub fn alerts_by_severity(&self, severity: AlertSeverity) -> Vec<Alert> {
        let alerts = self.alerts.read().unwrap();
        alerts
            .iter()
            .filter(|a| a.severity == severity)
            .cloned()
            .collect()
    }

    /// Check if there are any critical or emergency unacknowledged alerts.
    pub fn has_critical_alerts(&self) -> bool {
        let alerts = self.alerts.read().unwrap();
        alerts.iter().any(|a| {
            !a.acknowledged
                && matches!(
                    a.severity,
                    AlertSeverity::Critical | AlertSeverity::Emergency
                )
        })
    }

    /// Get count of unacknowledged alerts by severity.
    pub fn unacked_count_by_severity(&self) -> HashMap<AlertSeverity, usize> {
        let alerts = self.alerts.read().unwrap();
        let mut counts = HashMap::new();
        for alert in alerts.iter() {
            if !alert.acknowledged {
                *counts.entry(alert.severity).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Clear all alerts.
    pub fn clear(&self) {
        let mut alerts = self.alerts.write().unwrap();
        alerts.clear();
    }

    /// Get total alert count.
    pub fn count(&self) -> usize {
        self.alerts.read().unwrap().len()
    }
}

// Alerter is Send + Sync
unsafe impl Send for Alerter {}
unsafe impl Sync for Alerter {}

/// Trait for alert handlers.
///
/// Implement this trait to handle alerts (e.g., logging, SMS, email, Slack).
pub trait AlertHandler: Send + Sync {
    /// Handle an alert.
    fn handle(&self, alert: &Alert);
}

/// Simple logging alert handler.
///
/// Logs alerts using the tracing crate at appropriate log levels.
pub struct LoggingAlertHandler;

impl AlertHandler for LoggingAlertHandler {
    fn handle(&self, alert: &Alert) {
        match alert.severity {
            AlertSeverity::Info => {
                tracing::info!(
                    alert_type = alert.alert_type.as_str(),
                    value = ?alert.value,
                    "{}",
                    alert.message
                );
            }
            AlertSeverity::Warning => {
                tracing::warn!(
                    alert_type = alert.alert_type.as_str(),
                    value = ?alert.value,
                    threshold = ?alert.threshold,
                    "{}",
                    alert.message
                );
            }
            AlertSeverity::Critical => {
                tracing::error!(
                    alert_type = alert.alert_type.as_str(),
                    value = ?alert.value,
                    threshold = ?alert.threshold,
                    "CRITICAL: {}",
                    alert.message
                );
            }
            AlertSeverity::Emergency => {
                tracing::error!(
                    alert_type = alert.alert_type.as_str(),
                    value = ?alert.value,
                    threshold = ?alert.threshold,
                    "EMERGENCY: {}",
                    alert.message
                );
            }
        }
    }
}

/// Format timestamp for alert display (HH:MM:SS).
fn format_timestamp_short(millis: u64) -> String {
    let secs = millis / 1000;
    let hours = (secs / 3600) % 24;
    let mins = (secs / 60) % 60;
    let s = secs % 60;
    format!("{hours:02}:{mins:02}:{s:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    #[test]
    fn test_alert_severity_ordering() {
        assert!(AlertSeverity::Info < AlertSeverity::Warning);
        assert!(AlertSeverity::Warning < AlertSeverity::Critical);
        assert!(AlertSeverity::Critical < AlertSeverity::Emergency);
    }

    #[test]
    fn test_alert_severity_actionable() {
        assert!(!AlertSeverity::Info.is_actionable());
        assert!(!AlertSeverity::Warning.is_actionable());
        assert!(AlertSeverity::Critical.is_actionable());
        assert!(AlertSeverity::Emergency.is_actionable());
    }

    #[test]
    fn test_alert_creation() {
        let alert = Alert::new(
            1,
            1000,
            AlertType::DrawdownWarning,
            AlertSeverity::Warning,
            "Test alert",
        )
        .with_value(0.015)
        .with_threshold(0.01);

        assert_eq!(alert.id, 1);
        assert_eq!(alert.timestamp, 1000);
        assert_eq!(alert.alert_type, AlertType::DrawdownWarning);
        assert_eq!(alert.severity, AlertSeverity::Warning);
        assert_eq!(alert.value, Some(0.015));
        assert_eq!(alert.threshold, Some(0.01));
        assert!(!alert.acknowledged);
    }

    #[test]
    fn test_alerter_new() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        assert_eq!(alerter.count(), 0);
        assert!(!alerter.has_critical_alerts());
    }

    #[test]
    fn test_check_calibration_warning() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        // Below warning threshold
        let alert = alerter.check_calibration(0.9, "fill_prob", ts);
        assert!(alert.is_some());
        let alert = alert.unwrap();
        assert_eq!(alert.severity, AlertSeverity::Warning);
        assert_eq!(alert.alert_type, AlertType::CalibrationDegraded);
    }

    #[test]
    fn test_check_calibration_critical() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        // Below critical threshold
        let alert = alerter.check_calibration(0.7, "fill_prob", ts);
        assert!(alert.is_some());
        let alert = alert.unwrap();
        assert_eq!(alert.severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_check_calibration_ok() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        // Above thresholds
        let alert = alerter.check_calibration(1.5, "fill_prob", ts);
        assert!(alert.is_none());
    }

    #[test]
    fn test_check_drawdown_warning() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        let alert = alerter.check_drawdown(0.015, ts);
        assert!(alert.is_some());
        let alert = alert.unwrap();
        assert_eq!(alert.alert_type, AlertType::DrawdownWarning);
        assert_eq!(alert.severity, AlertSeverity::Warning);
    }

    #[test]
    fn test_check_drawdown_critical() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        let alert = alerter.check_drawdown(0.025, ts);
        assert!(alert.is_some());
        let alert = alert.unwrap();
        assert_eq!(alert.alert_type, AlertType::DrawdownCritical);
        assert_eq!(alert.severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_check_fill_rate() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        // Fill rate at 30% of baseline
        let alert = alerter.check_fill_rate(3.0, 10.0, ts);
        assert!(alert.is_some());
        let alert = alert.unwrap();
        assert_eq!(alert.alert_type, AlertType::FillRateCollapse);
    }

    #[test]
    fn test_check_fill_rate_ok() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        // Fill rate at 80% of baseline - OK
        let alert = alerter.check_fill_rate(8.0, 10.0, ts);
        assert!(alert.is_none());
    }

    #[test]
    fn test_check_latency() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        let alert = alerter.check_latency(75.0, ts);
        assert!(alert.is_some());
        let alert = alert.unwrap();
        assert_eq!(alert.alert_type, AlertType::LatencySpike);
    }

    #[test]
    fn test_check_position() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        // 85% of limit
        let alert = alerter.check_position(8.5, 10.0, ts);
        assert!(alert.is_some());
        let alert = alert.unwrap();
        assert_eq!(alert.alert_type, AlertType::PositionLimitApproaching);
    }

    #[test]
    fn test_check_position_ok() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        // 50% of limit - OK
        let alert = alerter.check_position(5.0, 10.0, ts);
        assert!(alert.is_none());
    }

    #[test]
    fn test_deduplication() {
        let config = AlertConfig {
            dedup_window_s: 60,
            ..Default::default()
        };
        let alerter = Alerter::new(config, 100);
        let ts = now();

        // First alert should pass
        let alert1 = alerter.check_drawdown(0.015, ts);
        assert!(alert1.is_some());

        // Second alert within window should be deduplicated
        let alert2 = alerter.check_drawdown(0.015, ts + 1000);
        assert!(alert2.is_none());

        // Alert after window should pass
        let alert3 = alerter.check_drawdown(0.015, ts + 61000);
        assert!(alert3.is_some());
    }

    #[test]
    fn test_add_and_query_alerts() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        let alert1 = alerter.create_alert(
            AlertType::DrawdownWarning,
            AlertSeverity::Warning,
            "Warning 1".to_string(),
            ts,
        );
        let alert2 = alerter.create_alert(
            AlertType::DrawdownCritical,
            AlertSeverity::Critical,
            "Critical 1".to_string(),
            ts + 1000,
        );

        alerter.add_alert(alert1);
        alerter.add_alert(alert2);

        assert_eq!(alerter.count(), 2);
        assert!(alerter.has_critical_alerts());

        let recent = alerter.recent_alerts(10);
        assert_eq!(recent.len(), 2);
        // Most recent first
        assert_eq!(recent[0].message, "Critical 1");
    }

    #[test]
    fn test_acknowledge() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        let alert = alerter.create_alert(
            AlertType::DrawdownWarning,
            AlertSeverity::Warning,
            "Test".to_string(),
            ts,
        );
        let id = alert.id;
        alerter.add_alert(alert);

        assert_eq!(alerter.unacknowledged_alerts().len(), 1);

        alerter.acknowledge(id);

        assert_eq!(alerter.unacknowledged_alerts().len(), 0);
    }

    #[test]
    fn test_alerts_by_severity() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        alerter.add_alert(alerter.create_alert(
            AlertType::DrawdownWarning,
            AlertSeverity::Warning,
            "W1".to_string(),
            ts,
        ));
        alerter.add_alert(alerter.create_alert(
            AlertType::DrawdownCritical,
            AlertSeverity::Critical,
            "C1".to_string(),
            ts,
        ));
        alerter.add_alert(alerter.create_alert(
            AlertType::DrawdownWarning,
            AlertSeverity::Warning,
            "W2".to_string(),
            ts,
        ));

        let warnings = alerter.alerts_by_severity(AlertSeverity::Warning);
        assert_eq!(warnings.len(), 2);

        let criticals = alerter.alerts_by_severity(AlertSeverity::Critical);
        assert_eq!(criticals.len(), 1);
    }

    #[test]
    fn test_max_alerts_limit() {
        let alerter = Alerter::new(AlertConfig::default(), 3);
        let ts = now();

        for i in 0..5 {
            alerter.add_alert(alerter.create_alert(
                AlertType::DrawdownWarning,
                AlertSeverity::Warning,
                format!("Alert {}", i),
                ts + i as u64 * 1000,
            ));
        }

        assert_eq!(alerter.count(), 3);

        // Should have the most recent 3
        let recent = alerter.recent_alerts(10);
        assert_eq!(recent[0].message, "Alert 4");
        assert_eq!(recent[2].message, "Alert 2");
    }

    #[test]
    fn test_clear() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        alerter.add_alert(alerter.create_alert(
            AlertType::DrawdownWarning,
            AlertSeverity::Warning,
            "Test".to_string(),
            ts,
        ));

        assert_eq!(alerter.count(), 1);
        alerter.clear();
        assert_eq!(alerter.count(), 0);
    }

    #[test]
    fn test_unacked_count_by_severity() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        let alert1 = alerter.create_alert(
            AlertType::DrawdownWarning,
            AlertSeverity::Warning,
            "W1".to_string(),
            ts,
        );
        let id1 = alert1.id;
        alerter.add_alert(alert1);

        alerter.add_alert(alerter.create_alert(
            AlertType::DrawdownWarning,
            AlertSeverity::Warning,
            "W2".to_string(),
            ts,
        ));
        alerter.add_alert(alerter.create_alert(
            AlertType::DrawdownCritical,
            AlertSeverity::Critical,
            "C1".to_string(),
            ts,
        ));

        let counts = alerter.unacked_count_by_severity();
        assert_eq!(counts.get(&AlertSeverity::Warning), Some(&2));
        assert_eq!(counts.get(&AlertSeverity::Critical), Some(&1));

        // Acknowledge one warning
        alerter.acknowledge(id1);

        let counts = alerter.unacked_count_by_severity();
        assert_eq!(counts.get(&AlertSeverity::Warning), Some(&1));
    }

    #[test]
    fn test_alert_format() {
        let alert = Alert::new(
            1,
            3661000, // 01:01:01
            AlertType::DrawdownWarning,
            AlertSeverity::Warning,
            "Drawdown at 1.5%",
        )
        .with_value(0.015)
        .with_threshold(0.01);

        let formatted = alert.format();
        assert!(formatted.contains("01:01:01"));
        assert!(formatted.contains("WARN"));
        assert!(formatted.contains("DrawdownWarning"));
        assert!(formatted.contains("Drawdown at 1.5%"));
        assert!(formatted.contains("0.0150"));
        assert!(formatted.contains("0.0100"));
    }

    #[test]
    fn test_logging_handler() {
        // Just verify it compiles and runs without panic
        let handler = LoggingAlertHandler;
        let alert = Alert::new(
            1,
            1000,
            AlertType::DrawdownWarning,
            AlertSeverity::Warning,
            "Test",
        );
        handler.handle(&alert);
    }

    #[test]
    fn test_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let alerter = Arc::new(Alerter::new(AlertConfig::default(), 1000));
        let mut handles = vec![];

        // Spawn multiple threads adding alerts
        for i in 0..10 {
            let alerter = Arc::clone(&alerter);
            handles.push(thread::spawn(move || {
                for j in 0..10 {
                    let ts = (i * 1000000 + j * 1000) as u64;
                    let alert = alerter.create_alert(
                        AlertType::DrawdownWarning,
                        AlertSeverity::Warning,
                        format!("Thread {} Alert {}", i, j),
                        ts,
                    );
                    alerter.add_alert(alert);
                }
            }));
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Should have 100 alerts
        assert_eq!(alerter.count(), 100);
    }

    #[test]
    fn test_check_spread_blowup() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        // Spread at 25 bps > 20 bps threshold
        let alert = alerter.check_spread(25.0, ts);
        assert!(alert.is_some());
        let alert = alert.unwrap();
        assert_eq!(alert.alert_type, AlertType::SpreadBlowup);
        assert_eq!(alert.severity, AlertSeverity::Warning);
    }

    #[test]
    fn test_check_spread_ok() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        // Spread at 10 bps < 20 bps threshold
        let alert = alerter.check_spread(10.0, ts);
        assert!(alert.is_none());
    }

    #[test]
    fn test_check_inventory_imbalance() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        // 80% of max inventory
        let alert = alerter.check_inventory(0.008, 0.01, ts);
        assert!(alert.is_some());
        let alert = alert.unwrap();
        assert_eq!(alert.alert_type, AlertType::InventoryImbalance);
    }

    #[test]
    fn test_check_inventory_ok() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        // 50% of max inventory - OK
        let alert = alerter.check_inventory(0.005, 0.01, ts);
        assert!(alert.is_none());
    }

    #[test]
    fn test_check_signal_health() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        let alert = alerter.check_signal_health(0.03, "lead_lag", ts);
        assert!(alert.is_some());
        let alert = alert.unwrap();
        assert_eq!(alert.alert_type, AlertType::SignalDecaying);
    }

    #[test]
    fn test_check_no_fills() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        let alert = alerter.check_no_fills(310, ts);
        assert!(alert.is_some());
        let alert = alert.unwrap();
        assert_eq!(alert.alert_type, AlertType::NoFills);
    }

    #[test]
    fn test_check_kill_switch_armed() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        let alert = alerter.check_kill_switch_armed("drawdown_monitor", ts);
        assert!(alert.is_some());
        let alert = alert.unwrap();
        assert_eq!(alert.alert_type, AlertType::KillSwitchArmed);
        assert_eq!(alert.severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_has_critical_alerts_with_ack() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now();

        let alert = alerter.create_alert(
            AlertType::DrawdownCritical,
            AlertSeverity::Critical,
            "Critical".to_string(),
            ts,
        );
        let id = alert.id;
        alerter.add_alert(alert);

        assert!(alerter.has_critical_alerts());

        alerter.acknowledge(id);

        assert!(!alerter.has_critical_alerts());
    }
}
