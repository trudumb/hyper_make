//! Circuit breaker system for market conditions.
//!
//! Monitors multiple market conditions and triggers protective actions:
//! - OI drop cascades (liquidation events)
//! - Extreme funding rates
//! - Spread blowouts
//! - Fill rate collapses
//! - Model degradation
//!
//! Defense-first: when in doubt, widen spreads or pause trading.

use std::collections::{HashSet, VecDeque};

/// Types of circuit breakers that can be triggered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CircuitBreakerType {
    /// Open interest dropped significantly within window - liquidation cascade likely.
    OIDropCascade,

    /// Funding rate is extremely positive or negative.
    FundingExtreme,

    /// Bid-ask spread has blown out beyond normal.
    SpreadBlowout,

    /// Fill rate has collapsed relative to normal.
    FillRateCollapse,

    /// Model information ratio has degraded below threshold.
    ModelDegradation,
}

impl std::fmt::Display for CircuitBreakerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitBreakerType::OIDropCascade => write!(f, "OI Drop Cascade"),
            CircuitBreakerType::FundingExtreme => write!(f, "Extreme Funding"),
            CircuitBreakerType::SpreadBlowout => write!(f, "Spread Blowout"),
            CircuitBreakerType::FillRateCollapse => write!(f, "Fill Rate Collapse"),
            CircuitBreakerType::ModelDegradation => write!(f, "Model Degradation"),
        }
    }
}

/// Actions to take when a circuit breaker is triggered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitBreakerAction {
    /// Widen spreads by a multiplier (e.g., 2.0 = double spreads).
    WidenSpreads { multiplier: f64 },

    /// Cancel all resting quotes immediately.
    CancelAllQuotes,

    /// Pause trading entirely until conditions improve.
    PauseTrading,
}

impl CircuitBreakerAction {
    /// Returns the severity level (higher = more severe).
    pub fn severity(&self) -> u8 {
        match self {
            CircuitBreakerAction::WidenSpreads { .. } => 1,
            CircuitBreakerAction::CancelAllQuotes => 2,
            CircuitBreakerAction::PauseTrading => 3,
        }
    }

    /// Returns true if this action is more severe than another.
    pub fn is_more_severe_than(&self, other: &CircuitBreakerAction) -> bool {
        self.severity() > other.severity()
    }
}

/// Configuration for circuit breakers.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// OI drop threshold for cascade detection (e.g., -0.02 = 2% drop).
    /// Negative value indicates a drop.
    pub oi_drop_threshold: f64,

    /// Time window in seconds for OI drop calculation.
    pub oi_drop_window_s: u64,

    /// Funding rate threshold (absolute value, 8h rate).
    /// Example: 0.001 = 0.1% per 8h is considered extreme.
    pub funding_extreme_threshold: f64,

    /// Spread blowout threshold in basis points.
    /// Example: 50.0 = 50 bps spread triggers breaker.
    pub spread_blowout_bps: f64,

    /// Fill rate collapse as fraction of normal.
    /// Example: 0.1 = 10% of normal fill rate.
    pub fill_rate_collapse_pct: f64,

    /// Model information ratio threshold.
    /// Below this, model is considered degraded.
    pub model_ir_threshold: f64,

    /// Spread widening multiplier for moderate conditions.
    pub default_widen_multiplier: f64,

    /// Spread widening multiplier for severe conditions.
    pub severe_widen_multiplier: f64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            oi_drop_threshold: -0.02,         // 2% OI drop
            oi_drop_window_s: 60,             // 60 second window
            funding_extreme_threshold: 0.001, // 0.1% per 8h
            spread_blowout_bps: 50.0,         // 50 bps
            fill_rate_collapse_pct: 0.1,      // 10% of normal
            model_ir_threshold: 0.8,          // IR < 0.8 is degraded
            default_widen_multiplier: 2.0,    // 2x spreads for moderate
            severe_widen_multiplier: 5.0,     // 5x spreads for severe
        }
    }
}

impl CircuitBreakerConfig {
    /// Create with custom thresholds.
    pub fn new(oi_drop_threshold: f64, funding_threshold: f64, spread_bps: f64) -> Self {
        Self {
            oi_drop_threshold,
            funding_extreme_threshold: funding_threshold,
            spread_blowout_bps: spread_bps,
            ..Default::default()
        }
    }

    /// Builder method for OI drop window.
    pub fn with_oi_window(mut self, window_s: u64) -> Self {
        self.oi_drop_window_s = window_s.max(1);
        self
    }

    /// Builder method for fill rate threshold.
    pub fn with_fill_rate_threshold(mut self, pct: f64) -> Self {
        self.fill_rate_collapse_pct = pct.clamp(0.0, 1.0);
        self
    }

    /// Builder method for model IR threshold.
    pub fn with_model_ir_threshold(mut self, ir: f64) -> Self {
        self.model_ir_threshold = ir;
        self
    }

    /// Builder method for widen multipliers.
    pub fn with_widen_multipliers(mut self, default: f64, severe: f64) -> Self {
        self.default_widen_multiplier = default.max(1.0);
        self.severe_widen_multiplier = severe.max(default);
        self
    }
}

/// Monitor for circuit breaker conditions.
///
/// Tracks market conditions and triggers circuit breakers when thresholds are breached.
/// Thread-safe when used with appropriate synchronization.
#[derive(Debug)]
pub struct CircuitBreakerMonitor {
    /// Configuration
    config: CircuitBreakerConfig,

    /// Currently triggered breakers
    triggered: HashSet<CircuitBreakerType>,

    /// Last OI observation (timestamp_ms, value)
    last_oi: Option<(u64, f64)>,

    /// OI history for window calculation (timestamp_ms, value)
    oi_history: VecDeque<(u64, f64)>,
}

impl CircuitBreakerMonitor {
    /// Create a new circuit breaker monitor.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            triggered: HashSet::new(),
            last_oi: None,
            oi_history: VecDeque::with_capacity(128),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &CircuitBreakerConfig {
        &self.config
    }

    /// Update OI observation.
    ///
    /// # Arguments
    /// - `timestamp`: Unix timestamp in milliseconds
    /// - `oi`: Current open interest value
    pub fn update_oi(&mut self, timestamp: u64, oi: f64) {
        // Add to history
        self.oi_history.push_back((timestamp, oi));

        // Update last observation
        self.last_oi = Some((timestamp, oi));

        // Prune old entries outside the window
        let window_ms = self.config.oi_drop_window_s * 1000;
        let cutoff = timestamp.saturating_sub(window_ms);

        while let Some(&(ts, _)) = self.oi_history.front() {
            if ts < cutoff {
                self.oi_history.pop_front();
            } else {
                break;
            }
        }
    }

    /// Check for OI cascade condition.
    ///
    /// Returns `Some(OIDropCascade)` if OI has dropped beyond threshold within window.
    pub fn check_oi_cascade(&mut self, _timestamp: u64) -> Option<CircuitBreakerType> {
        if self.oi_history.len() < 2 {
            return None;
        }

        // Get oldest and newest OI in window
        let oldest = self.oi_history.front()?;
        let newest = self.oi_history.back()?;

        if oldest.1 <= 0.0 {
            return None;
        }

        // Calculate percentage change
        let pct_change = (newest.1 - oldest.1) / oldest.1;

        // Check if drop exceeds threshold (threshold is negative)
        if pct_change < self.config.oi_drop_threshold {
            self.triggered.insert(CircuitBreakerType::OIDropCascade);
            Some(CircuitBreakerType::OIDropCascade)
        } else {
            // Clear if condition resolved
            if self.triggered.contains(&CircuitBreakerType::OIDropCascade) {
                // Only clear if we've recovered significantly (half the drop threshold)
                if pct_change > self.config.oi_drop_threshold / 2.0 {
                    self.triggered.remove(&CircuitBreakerType::OIDropCascade);
                }
            }
            None
        }
    }

    /// Check for extreme funding condition.
    ///
    /// # Arguments
    /// - `funding_rate_8h`: 8-hour funding rate as a fraction
    ///
    /// Returns `Some(FundingExtreme)` if funding is beyond threshold.
    pub fn check_funding(&self, funding_rate_8h: f64) -> Option<CircuitBreakerType> {
        if funding_rate_8h.abs() > self.config.funding_extreme_threshold {
            Some(CircuitBreakerType::FundingExtreme)
        } else {
            None
        }
    }

    /// Check for spread blowout condition.
    ///
    /// # Arguments
    /// - `spread_bps`: Current bid-ask spread in basis points
    ///
    /// Returns `Some(SpreadBlowout)` if spread exceeds threshold.
    pub fn check_spread(&self, spread_bps: f64) -> Option<CircuitBreakerType> {
        if spread_bps > self.config.spread_blowout_bps {
            Some(CircuitBreakerType::SpreadBlowout)
        } else {
            None
        }
    }

    /// Check for fill rate collapse condition.
    ///
    /// # Arguments
    /// - `current_rate`: Current fill rate (fills per time unit)
    /// - `normal_rate`: Normal/expected fill rate
    ///
    /// Returns `Some(FillRateCollapse)` if fill rate has collapsed.
    pub fn check_fill_rate(
        &self,
        current_rate: f64,
        normal_rate: f64,
    ) -> Option<CircuitBreakerType> {
        if normal_rate <= 0.0 {
            return None;
        }

        let ratio = current_rate / normal_rate;
        if ratio < self.config.fill_rate_collapse_pct {
            Some(CircuitBreakerType::FillRateCollapse)
        } else {
            None
        }
    }

    /// Check for model degradation condition.
    ///
    /// # Arguments
    /// - `ir`: Information ratio of the model
    ///
    /// Returns `Some(ModelDegradation)` if IR is below threshold.
    pub fn check_model_health(&self, ir: f64) -> Option<CircuitBreakerType> {
        if ir < self.config.model_ir_threshold {
            Some(CircuitBreakerType::ModelDegradation)
        } else {
            None
        }
    }

    /// Check if a specific breaker type is currently triggered.
    pub fn is_triggered(&self, breaker: CircuitBreakerType) -> bool {
        self.triggered.contains(&breaker)
    }

    /// Get the recommended action for a breaker type.
    pub fn get_action(&self, breaker: CircuitBreakerType) -> CircuitBreakerAction {
        match breaker {
            CircuitBreakerType::OIDropCascade => CircuitBreakerAction::CancelAllQuotes,
            CircuitBreakerType::FundingExtreme => CircuitBreakerAction::WidenSpreads {
                multiplier: self.config.default_widen_multiplier,
            },
            CircuitBreakerType::SpreadBlowout => CircuitBreakerAction::WidenSpreads {
                multiplier: self.config.severe_widen_multiplier,
            },
            CircuitBreakerType::FillRateCollapse => CircuitBreakerAction::WidenSpreads {
                multiplier: self.config.default_widen_multiplier,
            },
            CircuitBreakerType::ModelDegradation => CircuitBreakerAction::PauseTrading,
        }
    }

    /// Clear a specific breaker.
    pub fn clear(&mut self, breaker: CircuitBreakerType) {
        self.triggered.remove(&breaker);
    }

    /// Clear all breakers.
    pub fn clear_all(&mut self) {
        self.triggered.clear();
    }

    /// Get all currently triggered breakers.
    pub fn triggered_breakers(&self) -> Vec<CircuitBreakerType> {
        self.triggered.iter().copied().collect()
    }

    /// Get the most severe action required by any triggered breaker.
    pub fn most_severe_action(&self) -> Option<CircuitBreakerAction> {
        self.triggered
            .iter()
            .map(|&b| self.get_action(b))
            .max_by_key(|a| a.severity())
    }

    /// Trigger a breaker manually (for testing or external signals).
    pub fn trigger(&mut self, breaker: CircuitBreakerType) {
        self.triggered.insert(breaker);
    }

    /// Run all checks and return newly triggered breakers.
    ///
    /// # Arguments
    /// - `timestamp`: Current timestamp in milliseconds
    /// - `funding_rate_8h`: Optional funding rate
    /// - `spread_bps`: Optional current spread
    /// - `fill_rate`: Optional (current_rate, normal_rate) tuple
    /// - `model_ir`: Optional model information ratio
    pub fn check_all(
        &mut self,
        timestamp: u64,
        funding_rate_8h: Option<f64>,
        spread_bps: Option<f64>,
        fill_rate: Option<(f64, f64)>,
        model_ir: Option<f64>,
    ) -> Vec<CircuitBreakerType> {
        let mut newly_triggered = Vec::new();

        // Check OI cascade
        if let Some(breaker) = self.check_oi_cascade(timestamp) {
            if !self.triggered.contains(&breaker) {
                newly_triggered.push(breaker);
            }
            self.triggered.insert(breaker);
        }

        // Check funding
        if let Some(funding) = funding_rate_8h {
            if let Some(breaker) = self.check_funding(funding) {
                if !self.triggered.contains(&breaker) {
                    newly_triggered.push(breaker);
                }
                self.triggered.insert(breaker);
            } else {
                // Clear if recovered
                self.triggered.remove(&CircuitBreakerType::FundingExtreme);
            }
        }

        // Check spread
        if let Some(spread) = spread_bps {
            if let Some(breaker) = self.check_spread(spread) {
                if !self.triggered.contains(&breaker) {
                    newly_triggered.push(breaker);
                }
                self.triggered.insert(breaker);
            } else {
                // Clear if recovered
                self.triggered.remove(&CircuitBreakerType::SpreadBlowout);
            }
        }

        // Check fill rate
        if let Some((current, normal)) = fill_rate {
            if let Some(breaker) = self.check_fill_rate(current, normal) {
                if !self.triggered.contains(&breaker) {
                    newly_triggered.push(breaker);
                }
                self.triggered.insert(breaker);
            } else {
                // Clear if recovered
                self.triggered.remove(&CircuitBreakerType::FillRateCollapse);
            }
        }

        // Check model health
        if let Some(ir) = model_ir {
            if let Some(breaker) = self.check_model_health(ir) {
                if !self.triggered.contains(&breaker) {
                    newly_triggered.push(breaker);
                }
                self.triggered.insert(breaker);
            } else {
                // Clear if recovered
                self.triggered.remove(&CircuitBreakerType::ModelDegradation);
            }
        }

        newly_triggered
    }
}

// Ensure Send + Sync for thread safety
unsafe impl Send for CircuitBreakerMonitor {}
unsafe impl Sync for CircuitBreakerMonitor {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CircuitBreakerConfig::default();
        assert_eq!(config.oi_drop_threshold, -0.02);
        assert_eq!(config.oi_drop_window_s, 60);
        assert_eq!(config.spread_blowout_bps, 50.0);
    }

    #[test]
    fn test_oi_cascade_detection() {
        let config = CircuitBreakerConfig::default();
        let mut monitor = CircuitBreakerMonitor::new(config);

        let base_time = 1000000u64;

        // Add initial OI
        monitor.update_oi(base_time, 100.0);

        // Add OI after 30 seconds - small drop (no trigger)
        monitor.update_oi(base_time + 30_000, 99.0);
        assert!(monitor.check_oi_cascade(base_time + 30_000).is_none());

        // Add OI after 50 seconds - large drop (trigger)
        monitor.update_oi(base_time + 50_000, 95.0); // 5% drop
        let result = monitor.check_oi_cascade(base_time + 50_000);
        assert_eq!(result, Some(CircuitBreakerType::OIDropCascade));
        assert!(monitor.is_triggered(CircuitBreakerType::OIDropCascade));
    }

    #[test]
    fn test_funding_check() {
        let config = CircuitBreakerConfig::default();
        let monitor = CircuitBreakerMonitor::new(config);

        // Normal funding - no trigger
        assert!(monitor.check_funding(0.0005).is_none());

        // Extreme funding - trigger
        assert_eq!(
            monitor.check_funding(0.002),
            Some(CircuitBreakerType::FundingExtreme)
        );

        // Extreme negative funding - also trigger
        assert_eq!(
            monitor.check_funding(-0.002),
            Some(CircuitBreakerType::FundingExtreme)
        );
    }

    #[test]
    fn test_spread_check() {
        let config = CircuitBreakerConfig::default();
        let monitor = CircuitBreakerMonitor::new(config);

        // Normal spread - no trigger
        assert!(monitor.check_spread(30.0).is_none());

        // Wide spread - trigger
        assert_eq!(
            monitor.check_spread(60.0),
            Some(CircuitBreakerType::SpreadBlowout)
        );
    }

    #[test]
    fn test_fill_rate_check() {
        let config = CircuitBreakerConfig::default().with_fill_rate_threshold(0.1);
        let monitor = CircuitBreakerMonitor::new(config);

        // Normal fill rate - no trigger
        assert!(monitor.check_fill_rate(0.5, 1.0).is_none());

        // Collapsed fill rate - trigger
        assert_eq!(
            monitor.check_fill_rate(0.05, 1.0),
            Some(CircuitBreakerType::FillRateCollapse)
        );

        // Zero normal rate - no trigger (avoid div by zero)
        assert!(monitor.check_fill_rate(0.05, 0.0).is_none());
    }

    #[test]
    fn test_model_health_check() {
        let config = CircuitBreakerConfig::default().with_model_ir_threshold(0.8);
        let monitor = CircuitBreakerMonitor::new(config);

        // Healthy model - no trigger
        assert!(monitor.check_model_health(1.2).is_none());

        // Degraded model - trigger
        assert_eq!(
            monitor.check_model_health(0.5),
            Some(CircuitBreakerType::ModelDegradation)
        );
    }

    #[test]
    fn test_get_action() {
        let config = CircuitBreakerConfig::default();
        let monitor = CircuitBreakerMonitor::new(config);

        // OI cascade should cancel quotes
        assert_eq!(
            monitor.get_action(CircuitBreakerType::OIDropCascade),
            CircuitBreakerAction::CancelAllQuotes
        );

        // Model degradation should pause trading
        assert_eq!(
            monitor.get_action(CircuitBreakerType::ModelDegradation),
            CircuitBreakerAction::PauseTrading
        );

        // Funding should widen spreads
        match monitor.get_action(CircuitBreakerType::FundingExtreme) {
            CircuitBreakerAction::WidenSpreads { multiplier } => {
                assert!(multiplier > 1.0);
            }
            _ => panic!("Expected WidenSpreads action"),
        }
    }

    #[test]
    fn test_action_severity() {
        let widen = CircuitBreakerAction::WidenSpreads { multiplier: 2.0 };
        let cancel = CircuitBreakerAction::CancelAllQuotes;
        let pause = CircuitBreakerAction::PauseTrading;

        assert!(cancel.is_more_severe_than(&widen));
        assert!(pause.is_more_severe_than(&cancel));
        assert!(pause.is_more_severe_than(&widen));
    }

    #[test]
    fn test_clear_breaker() {
        let config = CircuitBreakerConfig::default();
        let mut monitor = CircuitBreakerMonitor::new(config);

        monitor.trigger(CircuitBreakerType::FundingExtreme);
        assert!(monitor.is_triggered(CircuitBreakerType::FundingExtreme));

        monitor.clear(CircuitBreakerType::FundingExtreme);
        assert!(!monitor.is_triggered(CircuitBreakerType::FundingExtreme));
    }

    #[test]
    fn test_triggered_breakers() {
        let config = CircuitBreakerConfig::default();
        let mut monitor = CircuitBreakerMonitor::new(config);

        assert!(monitor.triggered_breakers().is_empty());

        monitor.trigger(CircuitBreakerType::FundingExtreme);
        monitor.trigger(CircuitBreakerType::SpreadBlowout);

        let triggered = monitor.triggered_breakers();
        assert_eq!(triggered.len(), 2);
        assert!(triggered.contains(&CircuitBreakerType::FundingExtreme));
        assert!(triggered.contains(&CircuitBreakerType::SpreadBlowout));
    }

    #[test]
    fn test_most_severe_action() {
        let config = CircuitBreakerConfig::default();
        let mut monitor = CircuitBreakerMonitor::new(config);

        // No triggers - no action
        assert!(monitor.most_severe_action().is_none());

        // Only widen spreads
        monitor.trigger(CircuitBreakerType::FundingExtreme);
        match monitor.most_severe_action() {
            Some(CircuitBreakerAction::WidenSpreads { .. }) => {}
            other => panic!("Expected WidenSpreads, got {:?}", other),
        }

        // Add pause trading - should be most severe
        monitor.trigger(CircuitBreakerType::ModelDegradation);
        assert_eq!(
            monitor.most_severe_action(),
            Some(CircuitBreakerAction::PauseTrading)
        );
    }

    #[test]
    fn test_check_all() {
        let config = CircuitBreakerConfig::default();
        let mut monitor = CircuitBreakerMonitor::new(config);

        // Normal conditions - no triggers
        let newly = monitor.check_all(
            1000000,
            Some(0.0001),     // Normal funding
            Some(20.0),       // Normal spread
            Some((0.8, 1.0)), // Normal fill rate
            Some(1.5),        // Healthy model
        );
        assert!(newly.is_empty());

        // Extreme conditions
        let newly = monitor.check_all(
            1000000,
            Some(0.005),       // Extreme funding
            Some(100.0),       // Wide spread
            Some((0.01, 1.0)), // Collapsed fills
            Some(0.3),         // Degraded model
        );
        assert!(!newly.is_empty());
        assert!(newly.contains(&CircuitBreakerType::FundingExtreme));
        assert!(newly.contains(&CircuitBreakerType::SpreadBlowout));
        assert!(newly.contains(&CircuitBreakerType::FillRateCollapse));
        assert!(newly.contains(&CircuitBreakerType::ModelDegradation));
    }

    #[test]
    fn test_oi_window_pruning() {
        let config = CircuitBreakerConfig::default().with_oi_window(60); // 60 second window
        let mut monitor = CircuitBreakerMonitor::new(config);

        let base_time = 1000000u64;

        // Add entries over time
        for i in 0..10 {
            monitor.update_oi(base_time + i * 10_000, 100.0 - i as f64);
        }

        // After 90 seconds, entries older than 30s should be pruned
        monitor.update_oi(base_time + 90_000, 85.0);

        // Window should only contain recent entries
        assert!(monitor.oi_history.len() < 10);
    }

    #[test]
    fn test_circuit_breaker_type_display() {
        assert_eq!(
            format!("{}", CircuitBreakerType::OIDropCascade),
            "OI Drop Cascade"
        );
        assert_eq!(
            format!("{}", CircuitBreakerType::FundingExtreme),
            "Extreme Funding"
        );
    }
}
