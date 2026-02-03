//! Kill Switch - Emergency shutdown mechanism for the market maker.
//!
//! Provides automatic circuit breaker functionality to prevent catastrophic losses.
//! Monitors multiple risk metrics and triggers shutdown when thresholds are breached.
//!
//! # Kill Reasons
//! - **MaxLoss**: Daily P&L exceeds configured loss limit
//! - **MaxDrawdown**: Peak-to-trough drawdown exceeds threshold
//! - **MaxPosition**: Position value exceeds configured maximum
//! - **StaleData**: No market data received within timeout
//! - **RateLimit**: Exchange rate limit errors detected
//! - **CascadeDetected**: Liquidation cascade intensity too high
//! - **Manual**: Operator-triggered shutdown

use portable_atomic::{AtomicF64, Ordering as AtomicOrdering};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::market_maker::MarginMode;

/// Configuration for the kill switch.
#[derive(Debug, Clone)]
pub struct KillSwitchConfig {
    /// Maximum allowed daily loss (absolute USD value)
    /// Example: 500.0 means shutdown if daily loss exceeds $500
    pub max_daily_loss: f64,

    /// Maximum allowed drawdown as a fraction (0.0 to 1.0)
    /// Example: 0.05 means shutdown if drawdown exceeds 5%
    pub max_drawdown: f64,

    /// Maximum allowed position value in USD
    /// Example: 10000.0 means shutdown if position notional exceeds $10,000
    pub max_position_value: f64,

    /// Maximum allowed position size in contracts
    /// Example: 0.05 means soft limit at 0.05 contracts, kill at 0.10 (2x)
    pub max_position_contracts: f64,

    /// Maximum time without market data before considering data stale
    pub stale_data_threshold: Duration,

    /// Maximum number of rate limit errors before shutdown
    pub max_rate_limit_errors: u32,

    /// Enable/disable the kill switch (for testing)
    pub enabled: bool,
}

impl Default for KillSwitchConfig {
    fn default() -> Self {
        Self {
            max_daily_loss: 500.0,
            max_drawdown: 0.05,
            max_position_value: 10_000.0,
            max_position_contracts: 1.0, // Default 1.0 contracts, will be overridden by config
            // 30 seconds: allows time for ladder order placement (10 orders × ~1-2s each)
            // plus network latency and exchange processing
            stale_data_threshold: Duration::from_secs(30),
            max_rate_limit_errors: 3,
            enabled: true,
        }
    }
}

impl KillSwitchConfig {
    /// Create config derived from account value using Kelly-scaled risk limits.
    ///
    /// DERIVATION: Instead of arbitrary magic numbers, limits are derived from:
    /// 1. **max_daily_loss**: account_value × kelly_fraction × daily_vol × 2 (covers 95% of days)
    /// 2. **max_drawdown**: VaR_99(daily) × sqrt(horizon) for multi-day accumulation
    /// 3. **max_position_value**: account_value × max_leverage
    ///
    /// # Arguments
    /// * `account_value` - Current account equity in USD
    /// * `kelly_fraction` - Fraction of full Kelly (typically 0.25 for quarter Kelly)
    /// * `daily_volatility` - Expected daily volatility (fraction, e.g., 0.02 for 2%)
    /// * `max_leverage` - Maximum leverage allowed by exchange
    ///
    /// # Example
    /// ```rust,ignore
    /// // $10k account, quarter Kelly, 2% daily vol, 10x max leverage
    /// let config = KillSwitchConfig::from_account_kelly(10_000.0, 0.25, 0.02, 10.0);
    /// // max_daily_loss ≈ $100 (10k × 0.25 × 0.02 × 2)
    /// // max_position_value ≈ $100k (10k × 10)
    /// ```
    pub fn from_account_kelly(
        account_value: f64,
        kelly_fraction: f64,
        daily_volatility: f64,
        max_leverage: f64,
    ) -> Self {
        // Max daily loss = account × f × 2σ (2σ covers 95% of days)
        let max_daily_loss = account_value * kelly_fraction * daily_volatility * 2.0;

        // Max drawdown = VaR_99 × sqrt(horizon)
        // VaR_99 ≈ 2.33σ for normal distribution
        // Assume 5-day recovery horizon, but with mean reversion use T^0.4
        let z_99 = 2.33;
        let horizon_factor = 5.0_f64.powf(0.4); // ~1.9
        let max_drawdown = (z_99 * daily_volatility * horizon_factor).clamp(0.02, 0.20);

        // Max position from leverage
        let max_position_value = account_value * max_leverage;

        Self {
            max_daily_loss,
            max_drawdown,
            max_position_value,
            ..Default::default()
        }
    }

    /// Create config appropriate for margin mode.
    ///
    /// HIP-3 assets use isolated margin which is more dangerous because:
    /// 1. Margin cannot be shared with other positions
    /// 2. Liquidation happens per-position, not account-wide
    /// 3. Position can be liquidated even if overall account is healthy
    ///
    /// For isolated margin, we apply tighter limits as a safety measure.
    pub fn for_margin_mode(margin_mode: MarginMode, base_config: Self) -> Self {
        match margin_mode {
            MarginMode::Cross => base_config,
            MarginMode::Isolated => {
                // Isolated margin is more dangerous - apply tighter limits
                Self {
                    // 20% tighter drawdown limit for isolated positions
                    max_drawdown: base_config.max_drawdown * 0.8,
                    // 30% tighter position value limit (isolated can liquidate faster)
                    max_position_value: base_config.max_position_value * 0.7,
                    // Everything else stays the same
                    ..base_config
                }
            }
        }
    }
}

use crate::market_maker::config::DynamicRiskConfig;

/// Calculate dynamic max_position_value from first principles.
///
/// Two constraints applied:
/// 1. **Leverage constraint**: position_value ≤ account_value × max_leverage
/// 2. **Volatility constraint**: position_value ≤ (equity × risk_fraction) / (num_sigmas × σ × √T)
///
/// The final limit is min(leverage_limit, volatility_limit).
/// Volatility can only reduce the limit, never exceed leverage.
///
/// Uses Bayesian blend to regularize volatility limit towards prior when sigma confidence is low.
///
/// # Arguments
/// * `account_value` - Current account equity in USD
/// * `sigma` - Per-second volatility estimate (from estimator)
/// * `time_horizon` - Expected holding time in seconds (1 / arrival_intensity)
/// * `sigma_confidence` - Confidence in sigma estimate, 0.0-1.0
/// * `risk_config` - Dynamic risk configuration with priors and leverage
///
/// # Returns
/// Position value limit in USD, capped by leverage
pub fn calculate_dynamic_max_position_value(
    account_value: f64,
    sigma: f64,
    time_horizon: f64,
    sigma_confidence: f64,
    risk_config: &DynamicRiskConfig,
) -> f64 {
    // Hard constraint from leverage - this is the ceiling
    let leverage_limit = account_value * risk_config.max_leverage;

    // Guard against degenerate inputs - fall back to leverage limit
    if account_value <= 0.0 {
        return 0.0;
    }
    if sigma <= 1e-9 || time_horizon <= 0.0 {
        return leverage_limit;
    }

    // Prior estimate: what we'd use with no market data (but still capped by leverage)
    let prior_move = risk_config.sigma_prior * time_horizon.sqrt();
    let prior_volatility_limit = if prior_move > 1e-9 {
        (account_value * risk_config.risk_fraction) / (risk_config.num_sigmas * prior_move)
    } else {
        leverage_limit
    };

    // Raw limit from observed volatility
    let expected_move = sigma * time_horizon.sqrt();
    let raw_volatility_limit =
        (account_value * risk_config.risk_fraction) / (risk_config.num_sigmas * expected_move);

    // Bayesian blend: posterior = (confidence × raw + (1-confidence) × prior)
    let confidence = sigma_confidence.clamp(0.0, 1.0);
    let volatility_limit =
        confidence * raw_volatility_limit + (1.0 - confidence) * prior_volatility_limit;

    // Final limit: min of leverage and volatility constraints
    // Volatility can reduce limit during high-risk periods, but never exceed leverage
    leverage_limit.min(volatility_limit)
}

/// Reasons why the kill switch can be triggered.
#[derive(Debug, Clone, PartialEq)]
pub enum KillReason {
    /// Daily loss exceeded configured maximum
    MaxLoss { loss: f64, limit: f64 },
    /// Drawdown exceeded configured maximum
    MaxDrawdown { drawdown: f64, limit: f64 },
    /// Position value exceeded configured maximum
    MaxPosition { value: f64, limit: f64 },
    /// Position contracts exceeded 2x configured maximum (runaway position)
    PositionRunaway { contracts: f64, limit: f64 },
    /// Market data is stale (no updates within threshold)
    StaleData {
        elapsed: Duration,
        threshold: Duration,
    },
    /// Too many rate limit errors from exchange
    RateLimit { count: u32, limit: u32 },
    /// Liquidation cascade detected with high severity
    CascadeDetected { severity: f64 },
    /// Manual shutdown triggered by operator
    Manual { reason: String },
}

impl std::fmt::Display for KillReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KillReason::MaxLoss { loss, limit } => {
                write!(f, "Max daily loss exceeded: ${:.2} > ${:.2}", loss, limit)
            }
            KillReason::MaxDrawdown { drawdown, limit } => {
                write!(
                    f,
                    "Max drawdown exceeded: {:.2}% > {:.2}%",
                    drawdown * 100.0,
                    limit * 100.0
                )
            }
            KillReason::MaxPosition { value, limit } => {
                write!(
                    f,
                    "Max position value exceeded: ${:.2} > ${:.2}",
                    value, limit
                )
            }
            KillReason::PositionRunaway { contracts, limit } => {
                write!(
                    f,
                    "Position runaway: {:.6} contracts > {:.6} (2x limit)",
                    contracts, limit
                )
            }
            KillReason::StaleData { elapsed, threshold } => {
                write!(
                    f,
                    "Stale data: no update for {:.1}s > {:.1}s threshold",
                    elapsed.as_secs_f64(),
                    threshold.as_secs_f64()
                )
            }
            KillReason::RateLimit { count, limit } => {
                write!(f, "Rate limit errors: {} > {} limit", count, limit)
            }
            KillReason::CascadeDetected { severity } => {
                write!(f, "Liquidation cascade detected: severity {:.2}", severity)
            }
            KillReason::Manual { reason } => {
                write!(f, "Manual shutdown: {}", reason)
            }
        }
    }
}

/// Current state being monitored by the kill switch.
#[derive(Debug, Clone)]
pub struct KillSwitchState {
    /// Current daily P&L (negative = loss)
    pub daily_pnl: f64,
    /// Peak P&L since start of day (for drawdown calculation)
    pub peak_pnl: f64,
    /// Current position size
    pub position: f64,
    /// Current mid price (for position value calculation)
    pub mid_price: f64,
    /// Last time market data was received
    pub last_data_time: Instant,
    /// Count of rate limit errors
    pub rate_limit_errors: u32,
    /// Current liquidation cascade severity (0.0 to 1.0+)
    pub cascade_severity: f64,
    /// Current account value in USD (for margin-based position limits)
    pub account_value: f64,
    /// Current leverage setting
    pub leverage: f64,
}

impl Default for KillSwitchState {
    fn default() -> Self {
        Self {
            daily_pnl: 0.0,
            peak_pnl: 0.0,
            position: 0.0,
            mid_price: 0.0,
            last_data_time: Instant::now(),
            rate_limit_errors: 0,
            cascade_severity: 0.0,
            account_value: 0.0,
            leverage: 1.0,
        }
    }
}

/// Kill switch for emergency market maker shutdown.
///
/// Uses atomic operations for thread-safe trigger checking.
/// Once triggered, the kill switch remains triggered until manually reset.
pub struct KillSwitch {
    /// Whether the kill switch has been triggered
    triggered: AtomicBool,
    /// Reasons why the kill switch was triggered (may have multiple)
    trigger_reasons: Mutex<Vec<KillReason>>,
    /// Configuration (uses Mutex for dynamic limit updates)
    config: Mutex<KillSwitchConfig>,
    /// Current state
    state: Mutex<KillSwitchState>,
    /// Max position value limit - atomic for lock-free hot path reads.
    /// Uses portable_atomic::AtomicF64 for cross-platform f64 atomics.
    atomic_max_position_value: AtomicF64,
}

impl KillSwitch {
    /// Create a new kill switch with the given configuration.
    pub fn new(config: KillSwitchConfig) -> Self {
        let initial_max_position_value = config.max_position_value;
        Self {
            triggered: AtomicBool::new(false),
            trigger_reasons: Mutex::new(Vec::new()),
            config: Mutex::new(config),
            state: Mutex::new(KillSwitchState::default()),
            atomic_max_position_value: AtomicF64::new(initial_max_position_value),
        }
    }

    /// Check if the kill switch has been triggered.
    ///
    /// This is a fast atomic check suitable for hot paths.
    #[inline]
    pub fn is_triggered(&self) -> bool {
        self.triggered.load(Ordering::Relaxed)
    }

    /// Get the reasons why the kill switch was triggered.
    pub fn trigger_reasons(&self) -> Vec<KillReason> {
        self.trigger_reasons.lock().unwrap().clone()
    }

    /// Manually trigger the kill switch.
    pub fn trigger_manual(&self, reason: String) {
        self.trigger(KillReason::Manual { reason });
    }

    /// Reset the kill switch (use with caution!).
    ///
    /// This clears the triggered state and all reasons.
    /// Only use after investigating and resolving the trigger cause.
    pub fn reset(&self) {
        self.triggered.store(false, Ordering::SeqCst);
        self.trigger_reasons.lock().unwrap().clear();
    }

    /// Update the state and check all kill conditions.
    ///
    /// Returns `Some(reason)` if a new trigger occurred, `None` otherwise.
    /// Note: Even if already triggered, this will continue to check and log
    /// additional reasons.
    pub fn check(&self, state: &KillSwitchState) -> Option<KillReason> {
        // Lock config once for all checks
        let config = self.config.lock().unwrap();

        if !config.enabled {
            return None;
        }

        // Update internal state
        {
            let mut s = self.state.lock().unwrap();
            *s = state.clone();
        }

        // Check each condition
        if let Some(reason) = self.check_daily_loss(state, &config) {
            self.trigger(reason.clone());
            return Some(reason);
        }

        if let Some(reason) = self.check_drawdown(state, &config) {
            self.trigger(reason.clone());
            return Some(reason);
        }

        if let Some(reason) = self.check_position_value(state, &config) {
            self.trigger(reason.clone());
            return Some(reason);
        }

        if let Some(reason) = self.check_position_runaway(state, &config) {
            self.trigger(reason.clone());
            return Some(reason);
        }

        if let Some(reason) = self.check_stale_data(state, &config) {
            self.trigger(reason.clone());
            return Some(reason);
        }

        if let Some(reason) = self.check_rate_limit(state, &config) {
            self.trigger(reason.clone());
            return Some(reason);
        }

        if let Some(reason) = self.check_cascade(state, &config) {
            self.trigger(reason.clone());
            return Some(reason);
        }

        None
    }

    /// Update P&L and check kill conditions.
    ///
    /// Call this after each fill or periodic P&L update.
    pub fn update_pnl(&self, realized_pnl_delta: f64) {
        let mut state = self.state.lock().unwrap();
        state.daily_pnl += realized_pnl_delta;

        // Update peak for drawdown calculation
        if state.daily_pnl > state.peak_pnl {
            state.peak_pnl = state.daily_pnl;
        }
    }

    /// Update position and mid price.
    pub fn update_position(&self, position: f64, mid_price: f64) {
        let mut state = self.state.lock().unwrap();
        state.position = position;
        state.mid_price = mid_price;
        state.last_data_time = Instant::now();
    }

    /// Record a rate limit error.
    pub fn record_rate_limit_error(&self) {
        let mut state = self.state.lock().unwrap();
        state.rate_limit_errors += 1;
    }

    /// Update cascade severity.
    pub fn update_cascade_severity(&self, severity: f64) {
        let mut state = self.state.lock().unwrap();
        state.cascade_severity = severity;
    }

    /// Reset daily P&L counters (call at start of trading day).
    pub fn reset_daily(&self) {
        let mut state = self.state.lock().unwrap();
        state.daily_pnl = 0.0;
        state.peak_pnl = 0.0;
        state.rate_limit_errors = 0;
    }

    /// Get current state snapshot.
    pub fn state(&self) -> KillSwitchState {
        self.state.lock().unwrap().clone()
    }

    /// Get configuration snapshot.
    pub fn config(&self) -> KillSwitchConfig {
        self.config.lock().unwrap().clone()
    }

    /// Update the dynamic position value limit.
    ///
    /// Call this periodically to adjust limits based on current account equity,
    /// volatility, and sigma confidence. Uses Bayesian blend to regularize.
    pub fn update_dynamic_limit(&self, new_max_value: f64) {
        // Update atomic for lock-free reads
        self.atomic_max_position_value
            .store(new_max_value, AtomicOrdering::Release);
        // Also update config for consistency in check() calls
        let mut config = self.config.lock().unwrap();
        config.max_position_value = new_max_value;
    }

    /// Get the current max position value limit.
    ///
    /// This is a fast lock-free read suitable for hot paths.
    #[inline]
    pub fn max_position_value(&self) -> f64 {
        self.atomic_max_position_value.load(AtomicOrdering::Acquire)
    }

    // === Private helper methods ===

    fn trigger(&self, reason: KillReason) {
        // Set atomic flag
        self.triggered.store(true, Ordering::SeqCst);

        // Add reason to list
        let mut reasons = self.trigger_reasons.lock().unwrap();
        if !reasons.contains(&reason) {
            reasons.push(reason);
        }
    }

    fn check_daily_loss(
        &self,
        state: &KillSwitchState,
        config: &KillSwitchConfig,
    ) -> Option<KillReason> {
        let loss = -state.daily_pnl; // Convert to positive loss
        if loss > config.max_daily_loss {
            return Some(KillReason::MaxLoss {
                loss,
                limit: config.max_daily_loss,
            });
        }
        None
    }

    fn check_drawdown(
        &self,
        state: &KillSwitchState,
        config: &KillSwitchConfig,
    ) -> Option<KillReason> {
        if state.peak_pnl <= 0.0 {
            return None; // No peak to draw down from
        }

        let drawdown = (state.peak_pnl - state.daily_pnl) / state.peak_pnl;
        if drawdown > config.max_drawdown {
            return Some(KillReason::MaxDrawdown {
                drawdown,
                limit: config.max_drawdown,
            });
        }
        None
    }

    fn check_position_value(
        &self,
        state: &KillSwitchState,
        config: &KillSwitchConfig,
    ) -> Option<KillReason> {
        let value = state.position.abs() * state.mid_price;
        // Kill switch at 2x soft limit (reduce-only mode handles 1x-2x range in update_quotes)
        let hard_limit = config.max_position_value * 2.0;
        if value > hard_limit {
            return Some(KillReason::MaxPosition {
                value,
                limit: hard_limit,
            });
        }
        None
    }

    fn check_position_runaway(
        &self,
        state: &KillSwitchState,
        config: &KillSwitchConfig,
    ) -> Option<KillReason> {
        // CAPITAL-EFFICIENT: Use margin-based limit when available
        // Position runaway = contracts exceed 2× what margin SHOULD allow
        // This catches cases where reduce-only mode fails to control position growth
        //
        // Priority:
        // 1. If margin data available (account_value > 0): use margin-derived limit
        // 2. Else: fall back to config.max_position_contracts (for backwards compat)
        let contracts = state.position.abs();

        // Skip runaway check if no valid price yet (startup condition)
        // The reduce-only mode in update_quotes handles existing oversized positions.
        // We only want to kill on RUNTIME runaway, not pre-existing positions at startup.
        if state.mid_price <= 0.0 {
            return None;
        }

        let margin_based_limit = if state.account_value > 0.0 && state.leverage > 0.0 {
            // Same formula as startup: (account_value × leverage × 0.5) / price
            let safety_factor = 0.5;
            (state.account_value * state.leverage * safety_factor) / state.mid_price
        } else {
            // Fall back to configured max_position if margin data not available
            config.max_position_contracts
        };

        // Runaway = 2× the effective limit
        let hard_limit = margin_based_limit * 2.0;
        if contracts > hard_limit {
            return Some(KillReason::PositionRunaway {
                contracts,
                limit: hard_limit,
            });
        }
        None
    }

    fn check_stale_data(
        &self,
        state: &KillSwitchState,
        config: &KillSwitchConfig,
    ) -> Option<KillReason> {
        let elapsed = state.last_data_time.elapsed();
        if elapsed > config.stale_data_threshold {
            return Some(KillReason::StaleData {
                elapsed,
                threshold: config.stale_data_threshold,
            });
        }
        None
    }

    fn check_rate_limit(
        &self,
        state: &KillSwitchState,
        config: &KillSwitchConfig,
    ) -> Option<KillReason> {
        if state.rate_limit_errors > config.max_rate_limit_errors {
            return Some(KillReason::RateLimit {
                count: state.rate_limit_errors,
                limit: config.max_rate_limit_errors,
            });
        }
        None
    }

    fn check_cascade(
        &self,
        state: &KillSwitchState,
        _config: &KillSwitchConfig,
    ) -> Option<KillReason> {
        // Cascade severity > 1.0 means intensity is above normal
        // We trigger at severity > 5.0 (5x normal intensity)
        if state.cascade_severity > 5.0 {
            return Some(KillReason::CascadeDetected {
                severity: state.cascade_severity,
            });
        }
        None
    }
}

/// Summary of kill switch status for logging/monitoring.
#[derive(Debug, Clone)]
pub struct KillSwitchSummary {
    pub triggered: bool,
    pub reasons: Vec<String>,
    pub daily_pnl: f64,
    pub drawdown_pct: f64,
    pub position_value: f64,
    pub data_age_secs: f64,
    pub rate_limit_errors: u32,
    pub cascade_severity: f64,
}

impl KillSwitch {
    /// Get a summary of the current kill switch status.
    pub fn summary(&self) -> KillSwitchSummary {
        let state = self.state.lock().unwrap();
        let reasons = self.trigger_reasons();

        let drawdown_pct = if state.peak_pnl > 0.0 {
            (state.peak_pnl - state.daily_pnl) / state.peak_pnl * 100.0
        } else {
            0.0
        };

        KillSwitchSummary {
            triggered: self.is_triggered(),
            reasons: reasons.iter().map(|r| r.to_string()).collect(),
            daily_pnl: state.daily_pnl,
            drawdown_pct,
            position_value: state.position.abs() * state.mid_price,
            data_age_secs: state.last_data_time.elapsed().as_secs_f64(),
            rate_limit_errors: state.rate_limit_errors,
            cascade_severity: state.cascade_severity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = KillSwitchConfig::default();
        assert_eq!(config.max_daily_loss, 500.0);
        assert_eq!(config.max_drawdown, 0.05);
        assert!(config.enabled);
    }

    #[test]
    fn test_not_triggered_initially() {
        let ks = KillSwitch::new(KillSwitchConfig::default());
        assert!(!ks.is_triggered());
        assert!(ks.trigger_reasons().is_empty());
    }

    #[test]
    fn test_manual_trigger() {
        let ks = KillSwitch::new(KillSwitchConfig::default());
        ks.trigger_manual("test shutdown".to_string());

        assert!(ks.is_triggered());
        assert_eq!(ks.trigger_reasons().len(), 1);
        match &ks.trigger_reasons()[0] {
            KillReason::Manual { reason } => assert_eq!(reason, "test shutdown"),
            _ => panic!("Expected Manual reason"),
        }
    }

    #[test]
    fn test_max_loss_trigger() {
        let config = KillSwitchConfig {
            max_daily_loss: 100.0,
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        let state = KillSwitchState {
            daily_pnl: -150.0, // Loss of $150
            ..Default::default()
        };

        let reason = ks.check(&state);
        assert!(reason.is_some());
        assert!(ks.is_triggered());
        match reason.unwrap() {
            KillReason::MaxLoss { loss, limit } => {
                assert_eq!(loss, 150.0);
                assert_eq!(limit, 100.0);
            }
            _ => panic!("Expected MaxLoss reason"),
        }
    }

    #[test]
    fn test_drawdown_trigger() {
        let config = KillSwitchConfig {
            max_drawdown: 0.10, // 10%
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        let state = KillSwitchState {
            daily_pnl: 80.0,
            peak_pnl: 100.0, // 20% drawdown from peak
            ..Default::default()
        };

        let reason = ks.check(&state);
        assert!(reason.is_some());
        match reason.unwrap() {
            KillReason::MaxDrawdown { drawdown, limit } => {
                assert!((drawdown - 0.20).abs() < 0.01);
                assert_eq!(limit, 0.10);
            }
            _ => panic!("Expected MaxDrawdown reason"),
        }
    }

    #[test]
    fn test_max_position_trigger() {
        // Kill switch triggers at 2x the soft limit (reduce-only handles 1x-2x)
        let config = KillSwitchConfig {
            max_position_value: 5000.0, // Soft limit $5k, hard limit $10k
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        // Position value $11k > hard limit $10k should trigger
        let state = KillSwitchState {
            position: 0.11,       // 0.11 BTC
            mid_price: 100_000.0, // $100k per BTC = $11k position
            last_data_time: Instant::now(),
            ..Default::default()
        };

        let reason = ks.check(&state);
        assert!(reason.is_some());
        match reason.unwrap() {
            KillReason::MaxPosition { value, limit } => {
                assert_eq!(value, 11_000.0);
                assert_eq!(limit, 10_000.0); // 2x soft limit
            }
            _ => panic!("Expected MaxPosition reason"),
        }
    }

    #[test]
    fn test_max_position_reduce_only_zone_does_not_trigger() {
        // Between 1x and 2x limit, reduce-only mode handles it, not kill switch
        let config = KillSwitchConfig {
            max_position_value: 5000.0, // Soft limit $5k, hard limit $10k
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        // Position value $8k is in reduce-only zone (1x-2x), should NOT trigger kill switch
        let state = KillSwitchState {
            position: 0.08,       // 0.08 BTC
            mid_price: 100_000.0, // $100k per BTC = $8k position
            last_data_time: Instant::now(),
            ..Default::default()
        };

        let reason = ks.check(&state);
        assert!(
            reason.is_none(),
            "Kill switch should not trigger at 1.6x limit (reduce-only zone)"
        );
    }

    #[test]
    fn test_stale_data_trigger() {
        let config = KillSwitchConfig {
            stale_data_threshold: Duration::from_millis(100),
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        let state = KillSwitchState {
            last_data_time: Instant::now() - Duration::from_secs(1),
            ..Default::default()
        };

        let reason = ks.check(&state);
        assert!(reason.is_some());
        match reason.unwrap() {
            KillReason::StaleData { elapsed, threshold } => {
                assert!(elapsed > threshold);
            }
            _ => panic!("Expected StaleData reason"),
        }
    }

    #[test]
    fn test_cascade_trigger() {
        let ks = KillSwitch::new(KillSwitchConfig::default());

        let state = KillSwitchState {
            cascade_severity: 6.0, // 6x normal intensity
            last_data_time: Instant::now(),
            ..Default::default()
        };

        let reason = ks.check(&state);
        assert!(reason.is_some());
        match reason.unwrap() {
            KillReason::CascadeDetected { severity } => {
                assert_eq!(severity, 6.0);
            }
            _ => panic!("Expected CascadeDetected reason"),
        }
    }

    #[test]
    fn test_reset() {
        let ks = KillSwitch::new(KillSwitchConfig::default());
        ks.trigger_manual("test".to_string());
        assert!(ks.is_triggered());

        ks.reset();
        assert!(!ks.is_triggered());
        assert!(ks.trigger_reasons().is_empty());
    }

    #[test]
    fn test_disabled_kill_switch() {
        let config = KillSwitchConfig {
            enabled: false,
            max_daily_loss: 100.0,
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        let state = KillSwitchState {
            daily_pnl: -1000.0, // Way over limit
            ..Default::default()
        };

        // Should not trigger when disabled
        let reason = ks.check(&state);
        assert!(reason.is_none());
        assert!(!ks.is_triggered());
    }

    #[test]
    fn test_pnl_tracking() {
        let ks = KillSwitch::new(KillSwitchConfig::default());

        // Simulate some P&L updates
        ks.update_pnl(100.0); // Made $100
        ks.update_pnl(-30.0); // Lost $30
        ks.update_pnl(20.0); // Made $20

        let state = ks.state();
        assert_eq!(state.daily_pnl, 90.0); // Net $90
        assert_eq!(state.peak_pnl, 100.0); // Peak was $100
    }

    #[test]
    fn test_summary() {
        let ks = KillSwitch::new(KillSwitchConfig::default());
        ks.update_pnl(100.0);
        ks.update_pnl(-20.0);
        ks.update_position(0.5, 50000.0);

        let summary = ks.summary();
        assert!(!summary.triggered);
        assert_eq!(summary.daily_pnl, 80.0);
        assert!((summary.drawdown_pct - 20.0).abs() < 0.1);
        assert_eq!(summary.position_value, 25000.0);
    }
}
