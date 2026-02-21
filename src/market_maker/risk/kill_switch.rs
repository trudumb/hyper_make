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

use crate::market_maker::checkpoint::types::KillSwitchCheckpoint;
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

    /// Cascade severity threshold for kill switch (default 5.0, production 1.5)
    pub cascade_severity_threshold: f64,

    /// Price velocity threshold (fraction/second) for pulling quotes.
    /// Default: 0.05 (5% per second). At BTC $100k, that's $5k/s.
    pub price_velocity_threshold: f64,

    /// Fraction of max_position that constitutes a "jump" for liquidation detection.
    /// Default: 0.20 (20%). If position changes by >20% of max with no recent fill, possible liquidation.
    pub liquidation_position_jump_fraction: f64,

    /// Time window (seconds) within which a fill must have occurred to explain a position jump.
    /// Default: 5 seconds.
    pub liquidation_fill_timeout_s: u64,

    /// Minimum peak PnL (USD) before drawdown check activates.
    ///
    /// Drawdown from peak is only statistically meaningful when the peak
    /// represents a significant sample of fills, not spread noise.
    /// Below this threshold, `check_daily_loss()` provides the safety net.
    ///
    /// Default: $1.00 (roughly 40 fills at 5 bps capture on $50 notional).
    /// Production: `max(1.0, max_position_value * 0.02)`.
    pub min_peak_for_drawdown: f64,

    /// Position velocity threshold: max_position fraction per minute before action.
    ///
    /// Measures how fast position is changing. Detects rapid accumulation or whipsaws.
    /// Default: 0.50 (50% of max_position per minute triggers warning).
    /// Widen at 1x, pull quotes at 2x, kill at 4x.
    pub position_velocity_threshold: f64,

    /// Maximum absolute drawdown in USD before kill switch triggers.
    ///
    /// This catches cases where percentage drawdown is disabled due to low peak
    /// (peak < min_peak_for_drawdown). For example, a session with $100 capital
    /// might only reach $0.20 peak PnL, bypassing the percentage check entirely.
    /// Without this absolute check, the session could hit 63% drawdown undetected.
    ///
    /// Default: $5.00 (conservative). Production: 2% of max position notional, min $1.
    pub max_absolute_drawdown: f64,

    /// Enable/disable the kill switch (for testing)
    pub enabled: bool,

    // === Q20: Stuck Inventory Detection ===
    /// Maximum consecutive cycles with significant position but no reducing quotes
    /// before kill switch triggers. Default: 30 (~5 min at 10s/cycle).
    pub max_stuck_cycles: u32,
    /// Warning threshold in stuck cycles — triggers `ForceReducingQuotes`.
    /// Default: 10 (~100s at 10s/cycle).
    pub stuck_warning_cycles: u32,
    /// Minimum position fraction of max to consider "stuck".
    /// Default: 0.10 (10% of max_position).
    pub position_stuck_threshold_fraction: f64,
    /// Unrealized AS cost (fraction of max_position_notional) that triggers warning.
    /// Dollar threshold = max_position_value × this fraction.
    /// Default: 0.01 (1% → ~$2 for $200 max).
    pub unrealized_as_warn_fraction: f64,
    /// Unrealized AS cost (fraction of max_position_notional) that triggers kill.
    /// Default: 0.05 (5% → ~$10 for $200 max).
    pub unrealized_as_kill_fraction: f64,
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
            cascade_severity_threshold: 5.0,
            price_velocity_threshold: 0.05,
            liquidation_position_jump_fraction: 0.20,
            liquidation_fill_timeout_s: 5,
            min_peak_for_drawdown: 1.0,
            // Tightened from 0.50: catches one-sided fill cascades earlier.
            // Widen at 0.20x, pull at 0.40x, kill at 0.80x of max_position/min.
            position_velocity_threshold: 0.20,
            max_absolute_drawdown: 5.0, // Conservative $5 default
            enabled: true,
            // Q20: Stuck inventory defaults
            max_stuck_cycles: 30,
            stuck_warning_cycles: 10,
            position_stuck_threshold_fraction: 0.10,
            unrealized_as_warn_fraction: 0.01,
            unrealized_as_kill_fraction: 0.05,
        }
    }
}

impl KillSwitchConfig {
    /// Production preset for initial live deployment.
    ///
    /// Ultra-conservative thresholds from Phase 3 plan:
    /// - $50 daily loss limit (0.5% of $10K account)
    /// - 2% max drawdown
    /// - 10s stale data threshold (tighter than default 30s)
    /// - 2 rate limit errors max
    /// - Position limit from `max_position_usd` (asset-agnostic)
    ///
    /// Cascade threshold is set via `check_cascade()` method (1.5x).
    ///
    /// # Arguments
    /// * `_account_value_usd` - Account equity (reserved for future Kelly scaling)
    /// * `max_position_usd` - Maximum position notional in USD (e.g., 1000.0)
    pub fn production(_account_value_usd: f64, max_position_usd: f64) -> Self {
        Self {
            max_daily_loss: 50.0,
            max_drawdown: 0.02,
            max_position_value: max_position_usd,
            max_position_contracts: f64::MAX, // Derived at runtime from USD/price
            stale_data_threshold: Duration::from_secs(10),
            max_rate_limit_errors: 2,
            cascade_severity_threshold: 1.5,
            price_velocity_threshold: 0.05,
            liquidation_position_jump_fraction: 0.20,
            liquidation_fill_timeout_s: 5,
            // Drawdown is meaningless when peak is spread noise.
            // min_peak = max(1.0, 2% of max position notional).
            // With max_position_usd=$1000: min_peak=$20 (~800 fills to activate).
            // check_daily_loss ($50) still protects against catastrophic loss.
            min_peak_for_drawdown: 1.0_f64.max(max_position_usd * 0.02),
            position_velocity_threshold: 0.20,
            // Absolute drawdown = 2% of max position notional, min $1.
            // Catches small-capital scenarios where percentage check is bypassed.
            max_absolute_drawdown: (max_position_usd * 0.02).max(1.0),
            enabled: true,
            // Q20 defaults
            max_stuck_cycles: 30,
            stuck_warning_cycles: 10,
            position_stuck_threshold_fraction: 0.10,
            unrealized_as_warn_fraction: 0.01,
            unrealized_as_kill_fraction: 0.05,
        }
    }

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
            min_peak_for_drawdown: 1.0_f64.max(max_position_value * 0.02),
            max_absolute_drawdown: (max_position_value * 0.02).max(1.0),
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

    /// Create config scaled to account capital.
    /// Uses conservative defaults: 5% daily loss, 3% drawdown.
    pub fn for_capital(account_value: f64) -> Self {
        let max_daily_loss = (account_value * 0.05).max(1.0);
        let max_absolute_drawdown = (account_value * 0.03).max(0.50);
        let min_peak = (account_value * 0.005).max(0.50);
        let max_position_value = account_value * 3.0; // 3x leverage default
        Self {
            max_daily_loss,
            max_absolute_drawdown,
            min_peak_for_drawdown: min_peak,
            max_position_value,
            ..Default::default()
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
    /// Drawdown exceeded configured maximum (percentage-based)
    MaxDrawdown { drawdown: f64, limit: f64 },
    /// Absolute drawdown exceeded configured maximum (USD-based)
    /// Safety net for small-capital scenarios where percentage check is bypassed.
    Drawdown {
        drawdown: f64,
        threshold: f64,
        reason: String,
    },
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
    /// Possible liquidation detected: position jumped without a corresponding local fill
    LiquidationDetected {
        position_delta: f64,
        max_position: f64,
    },
    /// Inventory stuck: position cannot be reduced through normal quoting.
    /// Triggered when stuck_cycles or unrealized_as_cost exceeds thresholds.
    InventoryStuck {
        stuck_cycles: u32,
        unrealized_as_cost_usd: f64,
        cause: StuckCause,
    },
}

/// Cause of stuck inventory for diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub enum StuckCause {
    /// E[PnL] filter says all reducing quotes are negative EV
    EpnlBlocking,
    /// Reducing quotes exist but keep getting adversely filled
    AdverseSelection,
    /// Reducing side has no takers
    NoLiquidity,
    /// Unknown / multiple causes
    Unknown,
}

impl std::fmt::Display for StuckCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EpnlBlocking => write!(f, "E[PnL] filter blocking"),
            Self::AdverseSelection => write!(f, "adverse selection"),
            Self::NoLiquidity => write!(f, "no liquidity"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Escalation level for stuck inventory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StuckEscalation {
    /// No stuck condition detected
    None,
    /// Warning: force reducing quotes at progressively wider spreads
    ForceReducingQuotes,
    /// Kill: trigger kill switch
    Kill,
}

impl std::fmt::Display for KillReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KillReason::MaxLoss { loss, limit } => {
                write!(f, "Max daily loss exceeded: ${loss:.2} > ${limit:.2}")
            }
            KillReason::MaxDrawdown { drawdown, limit } => {
                let dd_pct = drawdown * 100.0;
                let lim_pct = limit * 100.0;
                write!(f, "Max drawdown exceeded: {dd_pct:.2}% > {lim_pct:.2}%")
            }
            KillReason::Drawdown {
                drawdown,
                threshold,
                reason,
            } => {
                write!(
                    f,
                    "Absolute drawdown ${drawdown:.2} > ${threshold:.2}: {reason}"
                )
            }
            KillReason::MaxPosition { value, limit } => {
                write!(f, "Max position value exceeded: ${value:.2} > ${limit:.2}")
            }
            KillReason::PositionRunaway { contracts, limit } => {
                write!(
                    f,
                    "Position runaway: {contracts:.6} contracts > {limit:.6} (2x limit)"
                )
            }
            KillReason::StaleData { elapsed, threshold } => {
                let el = elapsed.as_secs_f64();
                let th = threshold.as_secs_f64();
                write!(f, "Stale data: no update for {el:.1}s > {th:.1}s threshold")
            }
            KillReason::RateLimit { count, limit } => {
                write!(f, "Rate limit errors: {count} > {limit} limit")
            }
            KillReason::CascadeDetected { severity } => {
                write!(f, "Liquidation cascade detected: severity {severity:.2}")
            }
            KillReason::Manual { reason } => {
                write!(f, "Manual shutdown: {reason}")
            }
            KillReason::LiquidationDetected {
                position_delta,
                max_position,
            } => {
                write!(
                    f,
                    "Possible liquidation detected: position jump {position_delta:.6} (>{:.0}% of max {max_position:.6})",
                    20.0 // display the threshold percentage
                )
            }
            KillReason::InventoryStuck {
                stuck_cycles,
                unrealized_as_cost_usd,
                cause,
            } => {
                write!(
                    f,
                    "Inventory stuck: {stuck_cycles} cycles, ${unrealized_as_cost_usd:.2} unrealized AS cost ({cause})"
                )
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
    /// Maximum one-sided ladder depth in contracts, updated each quote cycle.
    /// Position runaway threshold will be at least this x 1.5.
    pub max_ladder_one_side_contracts: f64,
    /// Position at startup (pre-existing). Used to exempt inherited positions from
    /// the runaway check — reduce-only mode handles these, not the kill switch.
    pub initial_position: f64,

    // === Q20: Stuck Inventory Detection ===
    /// Consecutive cycles with significant position but no reducing quotes.
    pub position_stuck_cycles: u32,
    /// Cumulative adverse mid-move against position (USD).
    /// Only counts moves against the position (conservative: never credits favorable moves).
    pub unrealized_as_cost_usd: f64,
    /// Whether reducing-side quotes exist in the current ladder.
    pub has_reducing_quotes: bool,
    /// Previous mid price for computing mid-move per cycle.
    pub prev_mid_for_stuck: f64,
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
            max_ladder_one_side_contracts: 0.0,
            initial_position: 0.0,
            // Q20: Stuck inventory
            position_stuck_cycles: 0,
            unrealized_as_cost_usd: 0.0,
            has_reducing_quotes: true,
            prev_mid_for_stuck: 0.0,
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
    /// Timestamp of last own fill, for liquidation self-detection.
    /// If position jumps without a recent fill, it may be a liquidation.
    last_fill_time: Mutex<Option<Instant>>,
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
            last_fill_time: Mutex::new(None),
        }
    }

    /// Validate kill switch config against account capital.
    /// Logs warnings if thresholds seem misconfigured.
    pub fn validate_for_capital(&self, account_value: f64) {
        let config = self.config.lock().unwrap();
        if account_value > 0.0 {
            if config.max_daily_loss > account_value * 0.20 {
                tracing::warn!(
                    max_daily_loss = config.max_daily_loss,
                    account_value = account_value,
                    pct = %format!("{:.0}%", config.max_daily_loss / account_value * 100.0),
                    "Kill switch max_daily_loss > 20% of account — may be misconfigured"
                );
            }
            if config.max_daily_loss > account_value {
                tracing::error!(
                    max_daily_loss = config.max_daily_loss,
                    account_value = account_value,
                    "Kill switch max_daily_loss EXCEEDS account value — effectively disabled!"
                );
            }
            if config.min_peak_for_drawdown > account_value * 0.10 {
                tracing::warn!(
                    min_peak = config.min_peak_for_drawdown,
                    account_value = account_value,
                    "Kill switch min_peak_for_drawdown > 10% of account — drawdown check may never activate"
                );
            }
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

    /// Update position and mid price, with liquidation self-detection.
    ///
    /// If position jumps by more than `liquidation_position_jump_fraction` of
    /// `max_position_contracts` and no fill was recorded in the last
    /// `liquidation_fill_timeout_s` seconds, trigger kill switch.
    pub fn update_position(&self, position: f64, mid_price: f64) {
        let old_position;
        {
            let mut state = self.state.lock().unwrap();
            old_position = state.position;
            state.position = position;
            state.mid_price = mid_price;
            state.last_data_time = Instant::now();
        }

        // Liquidation self-detection: check for unexplained position jumps
        let config = self.config.lock().unwrap();
        if config.enabled {
            if let Some(reason) = self.check_liquidation(old_position, position, &config) {
                self.trigger(reason);
            }
        }
    }

    /// Record that we executed an own fill. Updates the fill timestamp used
    /// for liquidation self-detection.
    pub fn record_own_fill(&self) {
        let mut last_fill = self.last_fill_time.lock().unwrap();
        *last_fill = Some(Instant::now());
    }

    /// Check for possible liquidation: position jumped without a corresponding fill.
    fn check_liquidation(
        &self,
        old_position: f64,
        new_position: f64,
        config: &KillSwitchConfig,
    ) -> Option<KillReason> {
        let position_delta = (new_position - old_position).abs();
        let max_pos = config.max_position_contracts;

        // Skip if max_position not meaningful (e.g., f64::MAX during startup)
        if max_pos <= 0.0 || max_pos >= f64::MAX / 2.0 {
            return None;
        }

        let jump_threshold = max_pos * config.liquidation_position_jump_fraction;
        if position_delta <= jump_threshold {
            return None; // Small change, not suspicious
        }

        // Large position jump — check if a recent fill explains it
        let last_fill = self.last_fill_time.lock().unwrap();
        let fill_timeout = Duration::from_secs(config.liquidation_fill_timeout_s);

        // If we haven't recorded any fill yet, we're in startup/initialization.
        // Position changes during startup (from exchange sync) are expected.
        if last_fill.is_none() {
            return None;
        }

        let has_recent_fill = last_fill
            .map(|t| t.elapsed() < fill_timeout)
            .unwrap_or(false);

        if has_recent_fill {
            return None; // Position jump is explained by a recent fill
        }

        // Defense first: position jumped with no recent fill — possible liquidation
        Some(KillReason::LiquidationDetected {
            position_delta,
            max_position: max_pos,
        })
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

    /// Record the initial position at startup for runaway exemption.
    /// Pre-existing positions are handled by reduce-only mode, not the kill switch.
    pub fn set_initial_position(&self, position: f64) {
        self.state.lock().unwrap().initial_position = position;
    }

    /// Update the maximum one-sided ladder depth for position runaway calculation.
    /// Called by the quote engine after each ladder generation.
    pub fn set_ladder_depth(&self, one_side_contracts: f64) {
        self.state.lock().unwrap().max_ladder_one_side_contracts = one_side_contracts;
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

        // Absolute drawdown safety net: catches small-capital scenarios where
        // percentage check is disabled (peak < min_peak_for_drawdown).
        // For example: $100 capital, $0.20 peak, $2.50 loss = $2.70 absolute drawdown.
        let absolute_drawdown = state.peak_pnl - state.daily_pnl;
        if absolute_drawdown > config.max_absolute_drawdown && absolute_drawdown > 0.0 {
            return Some(KillReason::Drawdown {
                drawdown: absolute_drawdown,
                threshold: config.max_absolute_drawdown,
                reason: format!(
                    "Absolute drawdown ${:.2} exceeds max ${:.2}",
                    absolute_drawdown, config.max_absolute_drawdown
                ),
            });
        }

        // Skip percentage drawdown check when peak is below the noise floor.
        // Percentage drawdown from a tiny peak (e.g., $0.02 from one fill) is
        // statistically meaningless — any tick against you looks like 100%+ drawdown.
        // The absolute check above and daily loss check still protect.
        if state.peak_pnl < config.min_peak_for_drawdown {
            return None;
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
        if state.mid_price <= 0.0 {
            return None;
        }

        // Exempt pre-existing positions: if position hasn't grown beyond what was
        // inherited at startup, it's not a runaway — reduce-only mode handles these.
        // Only trigger kill switch when position grows BEYOND initial + buffer.
        if contracts <= state.initial_position.abs() + 0.001 {
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

        // Ladder-aware floor: a full one-sided sweep is NORMAL, not runaway.
        // Threshold must exceed max possible single-sweep exposure.
        // 1.5× ladder depth = one full sweep + 50% buffer for partial refills.
        let ladder_floor = state.max_ladder_one_side_contracts * 1.5;
        let hard_limit = (margin_based_limit * 2.0).max(ladder_floor);

        if ladder_floor > margin_based_limit * 2.0 {
            tracing::debug!(
                ladder_floor = %format!("{:.4}", ladder_floor),
                margin_limit = %format!("{:.4}", margin_based_limit * 2.0),
                "Position runaway using ladder floor (margin limit < ladder depth)"
            );
        }

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
        config: &KillSwitchConfig,
    ) -> Option<KillReason> {
        // Cascade severity > threshold means liquidation cascade is too intense
        // Default: 5.0 (5x normal), Production: 1.5 (tighter for live capital)
        if state.cascade_severity > config.cascade_severity_threshold {
            return Some(KillReason::CascadeDetected {
                severity: state.cascade_severity,
            });
        }
        None
    }

    /// Q20: Report reducing quote status and update stuck inventory detection.
    ///
    /// Called each quote cycle with:
    /// - `position`: current inventory
    /// - `mid_price`: current mid price
    /// - `max_position_contracts`: configured max position
    /// - `has_reducing_quotes`: whether the ladder has quotes on the reducing side
    ///
    /// Returns the escalation level: None, ForceReducingQuotes, or Kill.
    pub fn report_reducing_quote_status(
        &self,
        position: f64,
        mid_price: f64,
        max_position_contracts: f64,
        has_reducing_quotes: bool,
    ) -> StuckEscalation {
        let config = self.config.lock().unwrap();
        let mut state = self.state.lock().unwrap();

        let position_threshold = max_position_contracts * config.position_stuck_threshold_fraction;

        // If position is below threshold, reset stuck detection
        if position.abs() < position_threshold {
            state.position_stuck_cycles = 0;
            state.unrealized_as_cost_usd = 0.0;
            state.has_reducing_quotes = true;
            state.prev_mid_for_stuck = mid_price;
            return StuckEscalation::None;
        }

        // If we have reducing quotes, reset cycle counter (but keep AS cost tracking)
        if has_reducing_quotes {
            state.position_stuck_cycles = 0;
            state.has_reducing_quotes = true;
        } else {
            state.position_stuck_cycles += 1;
            state.has_reducing_quotes = false;
        }

        // Compute unrealized adverse mid-move
        if state.prev_mid_for_stuck > 0.0 && mid_price > 0.0 {
            let mid_move_against = if position > 0.0 {
                (state.prev_mid_for_stuck - mid_price).max(0.0) // long, market dropping
            } else {
                (mid_price - state.prev_mid_for_stuck).max(0.0) // short, market rising
            };
            state.unrealized_as_cost_usd += position.abs() * mid_move_against;
        }
        state.prev_mid_for_stuck = mid_price;

        // Compute dollar thresholds from config fractions
        let warn_usd = config.max_position_value * config.unrealized_as_warn_fraction;
        let kill_usd = config.max_position_value * config.unrealized_as_kill_fraction;

        // Check kill conditions (either cycles OR cost)
        if state.position_stuck_cycles >= config.max_stuck_cycles
            || state.unrealized_as_cost_usd >= kill_usd
        {
            let cause = if !has_reducing_quotes {
                StuckCause::EpnlBlocking
            } else {
                StuckCause::Unknown
            };
            tracing::error!(
                stuck_cycles = state.position_stuck_cycles,
                unrealized_as_cost_usd = %format!("{:.2}", state.unrealized_as_cost_usd),
                kill_threshold_usd = %format!("{:.2}", kill_usd),
                position = %format!("{:.4}", position),
                cause = %cause,
                "Q20: Inventory stuck — triggering kill switch"
            );
            return StuckEscalation::Kill;
        }

        // Check warning conditions
        if state.position_stuck_cycles >= config.stuck_warning_cycles
            || state.unrealized_as_cost_usd >= warn_usd
        {
            tracing::warn!(
                stuck_cycles = state.position_stuck_cycles,
                unrealized_as_cost_usd = %format!("{:.2}", state.unrealized_as_cost_usd),
                warn_threshold_usd = %format!("{:.2}", warn_usd),
                position = %format!("{:.4}", position),
                "Q20: Inventory stuck — forcing reducing quotes"
            );
            return StuckEscalation::ForceReducingQuotes;
        }

        StuckEscalation::None
    }

    /// Q20: Get current stuck inventory state (for diagnostics).
    pub fn stuck_state(&self) -> (u32, f64) {
        let state = self.state.lock().unwrap();
        (state.position_stuck_cycles, state.unrealized_as_cost_usd)
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

    /// Capture current kill switch state as a checkpoint.
    pub fn to_checkpoint(&self) -> KillSwitchCheckpoint {
        let state = self.state.lock().unwrap();
        let reasons = self.trigger_reasons.lock().unwrap();
        let triggered = self.is_triggered();

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let triggered_at_ms = if triggered { now_ms } else { 0 };

        KillSwitchCheckpoint {
            triggered,
            trigger_reasons: reasons.iter().map(|r| r.to_string()).collect(),
            daily_pnl: state.daily_pnl,
            peak_pnl: state.peak_pnl,
            triggered_at_ms,
            saved_at_ms: now_ms,
            // Q20: Stuck inventory state
            position_stuck_cycles: state.position_stuck_cycles,
            unrealized_as_cost_usd: state.unrealized_as_cost_usd,
        }
    }

    /// Restore kill switch state from a checkpoint.
    ///
    /// Daily P&L and peak P&L are only restored if the checkpoint was saved on the
    /// same UTC trading day. On a new day, counters reset to zero — a fresh day means
    /// a fresh P&L slate. This prevents stale losses from blocking new sessions.
    ///
    /// Re-triggers the kill switch if the checkpoint was triggered within
    /// the last 24 hours, but daily-scoped reasons (daily loss, drawdown) are
    /// skipped on a new trading day since the P&L they reference has been reset.
    /// Stale triggers (>24h) are always ignored.
    pub fn restore_from_checkpoint(&self, checkpoint: &KillSwitchCheckpoint) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let same_day = is_same_utc_day(checkpoint.saved_at_ms, now_ms);

        // Only restore daily P&L if checkpoint is from the same UTC trading day.
        // On a new day, counters start fresh at zero.
        if same_day && checkpoint.saved_at_ms > 0 {
            let mut state = self.state.lock().unwrap();
            state.daily_pnl = checkpoint.daily_pnl;
            state.peak_pnl = checkpoint.peak_pnl;
            // Q20: Restore stuck inventory state (same-day only)
            state.position_stuck_cycles = checkpoint.position_stuck_cycles;
            state.unrealized_as_cost_usd = checkpoint.unrealized_as_cost_usd;
        } else {
            let checkpoint_day = checkpoint.saved_at_ms / MS_PER_DAY;
            let current_day = now_ms / MS_PER_DAY;
            tracing::info!(
                checkpoint_day,
                current_day,
                stale_daily_pnl = checkpoint.daily_pnl,
                "Checkpoint from different trading day — daily P&L reset to zero"
            );
        }

        if !checkpoint.triggered {
            return;
        }

        // Check if the trigger is within 24 hours
        const TWENTY_FOUR_HOURS_MS: u64 = 24 * 60 * 60 * 1000;
        let age_ms = now_ms.saturating_sub(checkpoint.triggered_at_ms);

        if age_ms > TWENTY_FOUR_HOURS_MS {
            // Stale trigger — don't re-trigger after extended maintenance
            return;
        }

        // Re-trigger with saved reasons, filtering out:
        // - Position runaway: transient, live checks handle it with initial_position exemption
        // - Daily loss / drawdown on new day: P&L has been reset, these reasons are stale
        let persistent_reasons: Vec<_> = checkpoint
            .trigger_reasons
            .iter()
            .filter(|r| !r.contains("Position runaway"))
            .filter(|r| same_day || !is_daily_scoped_reason(r))
            .collect();

        if persistent_reasons.is_empty() {
            tracing::info!(
                same_day,
                "Checkpoint had kill switch trigger but no reasons survive for this session — not re-triggering"
            );
            return;
        }

        self.triggered.store(true, Ordering::SeqCst);
        let mut reasons = self.trigger_reasons.lock().unwrap();
        for reason_str in &persistent_reasons {
            let reason = KillReason::Manual {
                reason: format!("restored from checkpoint: {reason_str}"),
            };
            if !reasons.contains(&reason) {
                reasons.push(reason);
            }
        }
    }
}

const MS_PER_DAY: u64 = 24 * 60 * 60 * 1000;

/// Check if two timestamps (ms since epoch) fall on the same UTC calendar day.
fn is_same_utc_day(ts1_ms: u64, ts2_ms: u64) -> bool {
    (ts1_ms / MS_PER_DAY) == (ts2_ms / MS_PER_DAY)
}

/// Returns true if the kill reason string is daily-scoped (tied to daily P&L counters).
/// These reasons should not survive across trading day boundaries.
fn is_daily_scoped_reason(reason: &str) -> bool {
    reason.contains("daily loss") || reason.contains("drawdown")
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
    fn test_production_config() {
        let config = KillSwitchConfig::production(10_000.0, 1_000.0);
        assert_eq!(config.max_daily_loss, 50.0);
        assert_eq!(config.max_drawdown, 0.02);
        assert_eq!(config.max_position_contracts, f64::MAX); // Derived at runtime from USD/price
        assert_eq!(config.max_position_value, 1_000.0); // USD limit passed directly
        assert_eq!(config.stale_data_threshold, Duration::from_secs(10));
        assert_eq!(config.max_rate_limit_errors, 2);
        assert_eq!(config.cascade_severity_threshold, 1.5);
        // min_peak = max(1.0, 1000 * 0.02) = 20.0
        assert_eq!(config.min_peak_for_drawdown, 20.0);
        assert!(config.enabled);
    }

    #[test]
    fn test_cascade_with_production_threshold() {
        let config = KillSwitchConfig::production(10_000.0, 1_000.0);
        let ks = KillSwitch::new(config);

        // 2.0 severity > 1.5 production threshold → should trigger
        let state = KillSwitchState {
            cascade_severity: 2.0,
            last_data_time: Instant::now(),
            ..Default::default()
        };
        let reason = ks.check(&state);
        assert!(reason.is_some());
        match reason.unwrap() {
            KillReason::CascadeDetected { severity } => assert_eq!(severity, 2.0),
            _ => panic!("Expected CascadeDetected"),
        }
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
            max_absolute_drawdown: f64::MAX, // Bypass absolute check to test percentage
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
    fn test_drawdown_skipped_when_peak_below_threshold() {
        // Reproduces the real incident: 1 fill for $0.02, price moves against,
        // drawdown = (0.02 - (-0.0168)) / 0.02 = 184% — but peak is noise.
        let config = KillSwitchConfig {
            max_drawdown: 0.02,         // 2% threshold
            min_peak_for_drawdown: 1.0, // $1 minimum peak
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        let state = KillSwitchState {
            daily_pnl: -0.0168,
            peak_pnl: 0.02, // Single fill spread capture — below $1 threshold
            last_data_time: Instant::now(),
            ..Default::default()
        };

        // Drawdown is 184% but peak is below threshold → should NOT trigger
        let reason = ks.check(&state);
        assert!(
            reason.is_none(),
            "Drawdown should not fire when peak_pnl ({:.4}) < min_peak_for_drawdown (1.0)",
            state.peak_pnl
        );
        assert!(!ks.is_triggered());
    }

    #[test]
    fn test_drawdown_fires_when_peak_above_threshold() {
        // Same drawdown percentage, but peak is above the noise floor → fires correctly
        let config = KillSwitchConfig {
            max_drawdown: 0.10,         // 10% threshold
            min_peak_for_drawdown: 1.0, // $1 minimum peak
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        let state = KillSwitchState {
            daily_pnl: 0.80,
            peak_pnl: 1.50, // Above $1 threshold — drawdown is meaningful
            last_data_time: Instant::now(),
            ..Default::default()
        };

        // Drawdown = (1.50 - 0.80) / 1.50 = 46.7% > 10% → should trigger
        let reason = ks.check(&state);
        assert!(reason.is_some());
        match reason.unwrap() {
            KillReason::MaxDrawdown { drawdown, limit } => {
                assert!((drawdown - 0.467).abs() < 0.01);
                assert_eq!(limit, 0.10);
            }
            _ => panic!("Expected MaxDrawdown reason"),
        }
    }

    #[test]
    fn test_daily_loss_fires_even_when_drawdown_relaxed() {
        // Defense in depth: when peak is below threshold (drawdown relaxed),
        // the absolute daily loss check still catches catastrophic losses.
        let config = KillSwitchConfig {
            max_daily_loss: 50.0,
            max_drawdown: 0.02,
            min_peak_for_drawdown: 100.0, // High threshold → drawdown never fires
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        let state = KillSwitchState {
            daily_pnl: -75.0, // $75 loss exceeds $50 limit
            peak_pnl: 5.0,    // Below min_peak → drawdown check skipped
            last_data_time: Instant::now(),
            ..Default::default()
        };

        let reason = ks.check(&state);
        assert!(reason.is_some());
        match reason.unwrap() {
            KillReason::MaxLoss { loss, limit } => {
                assert_eq!(loss, 75.0);
                assert_eq!(limit, 50.0);
            }
            _ => panic!("Expected MaxLoss (defense in depth), not drawdown"),
        }
    }

    #[test]
    fn test_default_min_peak_for_drawdown() {
        let config = KillSwitchConfig::default();
        assert_eq!(config.min_peak_for_drawdown, 1.0);
    }

    #[test]
    fn test_kelly_config_min_peak_derived() {
        // from_account_kelly should derive min_peak from max_position_value
        let config = KillSwitchConfig::from_account_kelly(10_000.0, 0.25, 0.02, 10.0);
        // max_position_value = 10000 * 10 = 100000
        // min_peak = max(1.0, 100000 * 0.02) = 2000.0
        assert_eq!(config.min_peak_for_drawdown, 2000.0);
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

    // === Checkpoint persistence tests ===

    #[test]
    fn test_kill_switch_checkpoint_roundtrip() {
        let ks = KillSwitch::new(KillSwitchConfig::default());
        ks.update_pnl(50.0);
        ks.trigger_manual("test trigger".to_string());

        let checkpoint = ks.to_checkpoint();
        assert!(checkpoint.triggered);
        assert_eq!(checkpoint.daily_pnl, 50.0);
        assert_eq!(checkpoint.peak_pnl, 50.0);
        assert!(checkpoint.triggered_at_ms > 0);
        assert!(!checkpoint.trigger_reasons.is_empty());

        // Restore into a fresh kill switch
        let ks2 = KillSwitch::new(KillSwitchConfig::default());
        ks2.restore_from_checkpoint(&checkpoint);

        assert!(ks2.is_triggered());
        let state = ks2.state();
        assert_eq!(state.daily_pnl, 50.0);
        assert_eq!(state.peak_pnl, 50.0);
    }

    #[test]
    fn test_kill_switch_checkpoint_triggered_state_persists() {
        let ks = KillSwitch::new(KillSwitchConfig::default());
        // Trigger via MaxLoss check
        let state = KillSwitchState {
            daily_pnl: -600.0,
            last_data_time: Instant::now(),
            ..Default::default()
        };
        ks.check(&state);
        assert!(ks.is_triggered());

        let checkpoint = ks.to_checkpoint();
        assert!(checkpoint.triggered);

        let ks2 = KillSwitch::new(KillSwitchConfig::default());
        ks2.restore_from_checkpoint(&checkpoint);
        assert!(ks2.is_triggered());
    }

    #[test]
    fn test_kill_switch_checkpoint_reasons_preserved() {
        let ks = KillSwitch::new(KillSwitchConfig::default());
        ks.trigger_manual("reason one".to_string());
        // Trigger a second reason via cascade check
        let state = KillSwitchState {
            cascade_severity: 6.0,
            last_data_time: Instant::now(),
            ..Default::default()
        };
        ks.check(&state);

        let checkpoint = ks.to_checkpoint();
        assert!(checkpoint.trigger_reasons.len() >= 2);

        // Verify reasons contain both triggers
        let reasons_joined = checkpoint.trigger_reasons.join("; ");
        assert!(reasons_joined.contains("reason one"));
        assert!(reasons_joined.contains("cascade"));
    }

    #[test]
    fn test_kill_switch_restore_from_default_does_not_trigger() {
        let ks = KillSwitch::new(KillSwitchConfig::default());
        let default_checkpoint = KillSwitchCheckpoint::default();

        ks.restore_from_checkpoint(&default_checkpoint);
        assert!(!ks.is_triggered());
    }

    #[test]
    fn test_kill_switch_restore_expired_trigger_ignored() {
        let ks = KillSwitch::new(KillSwitchConfig::default());

        // Create a checkpoint that was triggered 25 hours ago
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let twenty_five_hours_ago = now_ms.saturating_sub(25 * 60 * 60 * 1000);

        let checkpoint = KillSwitchCheckpoint {
            triggered: true,
            trigger_reasons: vec!["old trigger".to_string()],
            daily_pnl: -100.0,
            peak_pnl: 50.0,
            triggered_at_ms: twenty_five_hours_ago,
            saved_at_ms: twenty_five_hours_ago,
            position_stuck_cycles: 0,
            unrealized_as_cost_usd: 0.0,
        };

        ks.restore_from_checkpoint(&checkpoint);
        // Should NOT re-trigger because trigger is >24h old
        assert!(!ks.is_triggered());
        // P&L should NOT be restored because checkpoint is from a different day
        let state = ks.state();
        assert_eq!(state.daily_pnl, 0.0);
        assert_eq!(state.peak_pnl, 0.0);
    }

    #[test]
    fn test_kill_switch_restore_triggered_blocks_trading() {
        let ks = KillSwitch::new(KillSwitchConfig::default());

        // Create a recent triggered checkpoint
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let checkpoint = KillSwitchCheckpoint {
            triggered: true,
            trigger_reasons: vec!["Max daily loss exceeded: $600.00 > $500.00".to_string()],
            daily_pnl: -600.0,
            peak_pnl: 0.0,
            triggered_at_ms: now_ms,
            saved_at_ms: now_ms,
            position_stuck_cycles: 0,
            unrealized_as_cost_usd: 0.0,
        };

        ks.restore_from_checkpoint(&checkpoint);
        assert!(ks.is_triggered());
        // The kill switch being triggered means is_triggered() returns true,
        // which blocks all trading in the event loop
    }

    // === Trading day boundary tests ===

    #[test]
    fn test_kill_switch_new_day_resets_daily_pnl() {
        let ks = KillSwitch::new(KillSwitchConfig::default());

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Checkpoint from yesterday with daily loss trigger
        let yesterday_ms = now_ms.saturating_sub(MS_PER_DAY);

        let checkpoint = KillSwitchCheckpoint {
            triggered: true,
            trigger_reasons: vec!["Max daily loss exceeded: $5.02 > $5.00".to_string()],
            daily_pnl: -5.02,
            peak_pnl: 0.0,
            triggered_at_ms: yesterday_ms,
            saved_at_ms: yesterday_ms,
            position_stuck_cycles: 0,
            unrealized_as_cost_usd: 0.0,
        };

        ks.restore_from_checkpoint(&checkpoint);

        // Should NOT re-trigger: daily loss is day-scoped and this is a new day
        assert!(!ks.is_triggered());

        // Daily P&L should be reset to zero, not restored
        let state = ks.state();
        assert_eq!(state.daily_pnl, 0.0);
        assert_eq!(state.peak_pnl, 0.0);
    }

    #[test]
    fn test_kill_switch_same_day_preserves_daily_pnl() {
        let ks = KillSwitch::new(KillSwitchConfig::default());

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Checkpoint from 1 minute ago (same day)
        let one_min_ago = now_ms.saturating_sub(60_000);

        let checkpoint = KillSwitchCheckpoint {
            triggered: true,
            trigger_reasons: vec!["Max daily loss exceeded: $600.00 > $500.00".to_string()],
            daily_pnl: -600.0,
            peak_pnl: 10.0,
            triggered_at_ms: one_min_ago,
            saved_at_ms: one_min_ago,
            position_stuck_cycles: 0,
            unrealized_as_cost_usd: 0.0,
        };

        ks.restore_from_checkpoint(&checkpoint);

        // Same day: should re-trigger and restore P&L
        assert!(ks.is_triggered());
        let state = ks.state();
        assert_eq!(state.daily_pnl, -600.0);
        assert_eq!(state.peak_pnl, 10.0);
    }

    #[test]
    fn test_kill_switch_new_day_non_daily_reason_persists_within_24h() {
        let ks = KillSwitch::new(KillSwitchConfig::default());

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // 12 hours ago — different day if we crossed midnight, still within 24h
        let twelve_hours_ago = now_ms.saturating_sub(12 * 3600 * 1000);
        let same_day = is_same_utc_day(twelve_hours_ago, now_ms);

        let checkpoint = KillSwitchCheckpoint {
            triggered: true,
            trigger_reasons: vec!["Manual shutdown: operator requested".to_string()],
            daily_pnl: -50.0,
            peak_pnl: 0.0,
            triggered_at_ms: twelve_hours_ago,
            saved_at_ms: twelve_hours_ago,
            position_stuck_cycles: 0,
            unrealized_as_cost_usd: 0.0,
        };

        ks.restore_from_checkpoint(&checkpoint);

        // Manual shutdown is NOT daily-scoped → persists across day boundaries within 24h
        assert!(ks.is_triggered());

        // But P&L depends on day boundary
        let state = ks.state();
        if same_day {
            assert_eq!(state.daily_pnl, -50.0);
        } else {
            assert_eq!(state.daily_pnl, 0.0);
        }
    }

    #[test]
    fn test_kill_switch_new_day_daily_loss_dropped_but_manual_kept() {
        let ks = KillSwitch::new(KillSwitchConfig::default());

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Guaranteed to be a different day but within 24h
        let yesterday_ms = now_ms.saturating_sub(MS_PER_DAY - 1000);

        let checkpoint = KillSwitchCheckpoint {
            triggered: true,
            trigger_reasons: vec![
                "Max daily loss exceeded: $600.00 > $500.00".to_string(),
                "Manual shutdown: operator requested".to_string(),
            ],
            daily_pnl: -600.0,
            peak_pnl: 0.0,
            triggered_at_ms: yesterday_ms,
            saved_at_ms: yesterday_ms,
            position_stuck_cycles: 0,
            unrealized_as_cost_usd: 0.0,
        };

        ks.restore_from_checkpoint(&checkpoint);

        // Should still trigger because manual reason persists
        assert!(ks.is_triggered());

        // But daily P&L is reset (different day)
        let state = ks.state();
        assert_eq!(state.daily_pnl, 0.0);

        // Only the manual reason should survive, not the daily loss
        let reasons = ks.trigger_reasons.lock().unwrap();
        assert_eq!(reasons.len(), 1);
        assert!(reasons[0].to_string().contains("operator requested"));
    }

    #[test]
    fn test_kill_switch_new_day_drawdown_not_restored() {
        let ks = KillSwitch::new(KillSwitchConfig::default());

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let yesterday_ms = now_ms.saturating_sub(MS_PER_DAY);

        let checkpoint = KillSwitchCheckpoint {
            triggered: true,
            trigger_reasons: vec!["Max drawdown exceeded: 15.00% > 10.00%".to_string()],
            daily_pnl: -30.0,
            peak_pnl: 200.0,
            triggered_at_ms: yesterday_ms,
            saved_at_ms: yesterday_ms,
            position_stuck_cycles: 0,
            unrealized_as_cost_usd: 0.0,
        };

        ks.restore_from_checkpoint(&checkpoint);

        // Drawdown is daily-scoped → should not re-trigger on new day
        assert!(!ks.is_triggered());
        // P&L reset
        let state = ks.state();
        assert_eq!(state.daily_pnl, 0.0);
        assert_eq!(state.peak_pnl, 0.0);
    }

    #[test]
    fn test_kill_switch_old_checkpoint_without_saved_at_resets_pnl() {
        // Old checkpoints (pre-upgrade) have saved_at_ms=0 via #[serde(default)]
        let ks = KillSwitch::new(KillSwitchConfig::default());

        let checkpoint = KillSwitchCheckpoint {
            triggered: false,
            trigger_reasons: vec![],
            daily_pnl: -5.02,
            peak_pnl: 1.0,
            triggered_at_ms: 0,
            saved_at_ms: 0, // simulates old checkpoint without this field
            position_stuck_cycles: 0,
            unrealized_as_cost_usd: 0.0,
        };

        ks.restore_from_checkpoint(&checkpoint);

        // saved_at_ms=0 is never "same day" as now → P&L should reset
        assert!(!ks.is_triggered());
        let state = ks.state();
        assert_eq!(state.daily_pnl, 0.0);
        assert_eq!(state.peak_pnl, 0.0);
    }

    #[test]
    fn test_is_same_utc_day() {
        // Same day
        let base = 1_707_782_400_000_u64; // 2024-02-13 00:00:00 UTC
        assert!(is_same_utc_day(base, base + 1000));
        assert!(is_same_utc_day(base, base + 23 * 3600 * 1000));

        // Different days
        assert!(!is_same_utc_day(base, base + 25 * 3600 * 1000));
        assert!(!is_same_utc_day(0, base));
    }

    #[test]
    fn test_is_daily_scoped_reason() {
        assert!(is_daily_scoped_reason(
            "Max daily loss exceeded: $5.02 > $5.00"
        ));
        assert!(is_daily_scoped_reason(
            "Max drawdown exceeded: 15.00% > 10.00%"
        ));
        assert!(!is_daily_scoped_reason("Manual shutdown: test"));
        assert!(!is_daily_scoped_reason("Position runaway: 5.0 > 3.0"));
        assert!(!is_daily_scoped_reason(
            "Liquidation cascade detected: severity 3.00"
        ));
    }

    // === Liquidation self-detection tests ===

    #[test]
    fn test_liquidation_normal_fill_no_trigger() {
        let config = KillSwitchConfig {
            max_position_contracts: 1.0,
            liquidation_position_jump_fraction: 0.20,
            liquidation_fill_timeout_s: 5,
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        // Record a recent fill
        ks.record_own_fill();

        // Position update within the jump threshold + recent fill
        ks.update_position(0.15, 50000.0); // 15% of max — below 20% threshold
        assert!(!ks.is_triggered());
    }

    #[test]
    fn test_liquidation_position_jump_no_fill_triggers() {
        let config = KillSwitchConfig {
            max_position_contracts: 1.0,
            liquidation_position_jump_fraction: 0.20,
            liquidation_fill_timeout_s: 5,
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        // Set initial position (startup — no trigger because no fill recorded yet)
        ks.update_position(0.0, 50000.0);
        assert!(!ks.is_triggered());

        // Record a fill to leave startup state, then make it old (expired)
        {
            let mut last_fill = ks.last_fill_time.lock().unwrap();
            *last_fill = Some(Instant::now() - Duration::from_secs(10));
        }

        // Position jumps by 30% of max with no recent fill (>5s ago)
        ks.update_position(0.3, 50000.0);
        assert!(ks.is_triggered());

        let reasons = ks.trigger_reasons();
        assert!(reasons.iter().any(|r| matches!(r, KillReason::LiquidationDetected { .. })));
    }

    #[test]
    fn test_liquidation_jump_with_recent_fill_safe() {
        let config = KillSwitchConfig {
            max_position_contracts: 1.0,
            liquidation_position_jump_fraction: 0.20,
            liquidation_fill_timeout_s: 5,
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        // Set initial position
        ks.update_position(0.0, 50000.0);

        // Record a fill — this explains the coming position jump
        ks.record_own_fill();

        // Position jumps by 30% of max, but we have a recent fill
        ks.update_position(0.3, 50000.0);
        assert!(!ks.is_triggered());
    }

    #[test]
    fn test_liquidation_small_change_ignored() {
        let config = KillSwitchConfig {
            max_position_contracts: 1.0,
            liquidation_position_jump_fraction: 0.20,
            liquidation_fill_timeout_s: 5,
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        // Set initial position
        ks.update_position(0.1, 50000.0);

        // Small position change (5% of max) — no fill needed
        ks.update_position(0.15, 50000.0);
        assert!(!ks.is_triggered());
    }

    #[test]
    fn test_runaway_ladder_aware() {
        // Margin-derived hard_limit = 3.32 but ladder depth = 3.33
        // A full sweep to 3.32 contracts should NOT trigger.
        let config = KillSwitchConfig {
            max_position_contracts: 10.0,
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        // Margin-based limit: (account_value * leverage * 0.5) / mid_price
        // We want margin_based_limit * 2.0 = 3.32
        // => margin_based_limit = 1.66
        // => (account_value * 1.0 * 0.5) / 100.0 = 1.66
        // => account_value = 332.0
        let state = KillSwitchState {
            position: 3.32,
            mid_price: 100.0,
            account_value: 332.0,
            leverage: 1.0,
            last_data_time: Instant::now(),
            max_ladder_one_side_contracts: 3.33,
            ..Default::default()
        };

        // Without ladder awareness this would trigger (3.32 > 3.32 is false, but let's
        // use 3.321 to be slightly above the margin limit).
        let state_just_over = KillSwitchState {
            position: 3.321,
            ..state.clone()
        };

        // ladder_floor = 3.33 * 1.5 = 4.995, hard_limit = max(3.32, 4.995) = 4.995
        // 3.321 < 4.995 => no trigger
        let reason = ks.check(&state_just_over);
        assert!(reason.is_none(), "Ladder-aware threshold should prevent false positive for full sweep");
    }

    #[test]
    fn test_runaway_extreme_still_triggers() {
        // Even with ladder awareness, truly extreme positions must trigger.
        let config = KillSwitchConfig {
            max_position_contracts: 10.0,
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        // ladder_floor = 3.33 * 1.5 = 4.995
        // Position = 3.33 * 2.0 = 6.66 > 4.995 => triggers
        let state = KillSwitchState {
            position: 6.66,
            mid_price: 100.0,
            account_value: 332.0,
            leverage: 1.0,
            last_data_time: Instant::now(),
            max_ladder_one_side_contracts: 3.33,
            ..Default::default()
        };

        let reason = ks.check(&state);
        assert!(reason.is_some(), "Extreme position should still trigger kill switch");
        match reason.unwrap() {
            KillReason::PositionRunaway { contracts, limit } => {
                assert!((contracts - 6.66).abs() < 1e-6);
                assert!((limit - 4.995).abs() < 1e-6);
            }
            other => panic!("Expected PositionRunaway, got {:?}", other),
        }
    }

    #[test]
    fn test_runaway_backward_compat() {
        // With ladder_depth = 0.0 (default), behavior matches old code exactly.
        let config = KillSwitchConfig {
            max_position_contracts: 5.0,
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        // Do NOT call set_ladder_depth — default is 0.0
        // ladder_floor = 0.0 * 1.5 = 0.0, so hard_limit = max(margin*2, 0.0) = margin*2

        // margin_based_limit = (200 * 1.0 * 0.5) / 100.0 = 1.0
        // hard_limit = max(2.0, 0.0) = 2.0
        let state_under = KillSwitchState {
            position: 1.99,
            mid_price: 100.0,
            account_value: 200.0,
            leverage: 1.0,
            last_data_time: Instant::now(),
            ..Default::default()
        };
        assert!(ks.check(&state_under).is_none(), "Under limit should not trigger");

        let state_over = KillSwitchState {
            position: 2.01,
            mid_price: 100.0,
            account_value: 200.0,
            leverage: 1.0,
            last_data_time: Instant::now(),
            ..Default::default()
        };
        let reason = ks.check(&state_over);
        assert!(reason.is_some(), "Over margin-based limit should trigger with default ladder depth");
        assert!(matches!(reason.unwrap(), KillReason::PositionRunaway { .. }));
    }

    #[test]
    fn test_for_capital_100_usd() {
        let config = KillSwitchConfig::for_capital(100.0);
        assert!((config.max_daily_loss - 5.0).abs() < 0.01, "5% of $100 = $5");
        assert!((config.max_absolute_drawdown - 3.0).abs() < 0.01, "3% of $100 = $3");
        assert!((config.min_peak_for_drawdown - 0.50).abs() < 0.01, "0.5% of $100 = $0.50");
        assert!((config.max_position_value - 300.0).abs() < 0.01, "3x leverage on $100");
    }

    #[test]
    fn test_for_capital_10000_usd() {
        let config = KillSwitchConfig::for_capital(10000.0);
        assert!((config.max_daily_loss - 500.0).abs() < 0.01, "5% of $10K = $500");
        assert!((config.max_absolute_drawdown - 300.0).abs() < 0.01, "3% of $10K = $300");
        assert!((config.min_peak_for_drawdown - 50.0).abs() < 0.01, "0.5% of $10K = $50");
    }

    #[test]
    fn test_for_capital_very_small() {
        let config = KillSwitchConfig::for_capital(5.0);
        assert!(config.max_daily_loss >= 1.0, "Min $1 daily loss");
        assert!(config.max_absolute_drawdown >= 0.50, "Min $0.50 absolute drawdown");
        assert!(config.min_peak_for_drawdown >= 0.50, "Min $0.50 peak");
    }

    #[test]
    fn test_validate_for_capital_warns() {
        // This test just verifies validate_for_capital doesn't panic
        let ks = KillSwitch::new(KillSwitchConfig::default());
        ks.validate_for_capital(100.0); // $500 max_daily_loss > $100 account -> should warn
    }

    // === Q20: Stuck Inventory Detection Tests ===

    #[test]
    fn test_stuck_counter_increments_and_resets() {
        let config = KillSwitchConfig {
            max_stuck_cycles: 30,
            stuck_warning_cycles: 10,
            position_stuck_threshold_fraction: 0.10,
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        // Position below threshold — no stuck
        let result = ks.report_reducing_quote_status(0.05, 100.0, 1.0, false);
        assert_eq!(result, StuckEscalation::None);
        assert_eq!(ks.stuck_state().0, 0);

        // Position above threshold, no reducing quotes — starts counting
        let result = ks.report_reducing_quote_status(0.5, 100.0, 1.0, false);
        assert_eq!(result, StuckEscalation::None);
        assert_eq!(ks.stuck_state().0, 1);

        // Another cycle without reducing quotes
        let result = ks.report_reducing_quote_status(0.5, 100.0, 1.0, false);
        assert_eq!(result, StuckEscalation::None);
        assert_eq!(ks.stuck_state().0, 2);

        // Reducing quotes appear — resets cycle counter
        let result = ks.report_reducing_quote_status(0.5, 100.0, 1.0, true);
        assert_eq!(result, StuckEscalation::None);
        assert_eq!(ks.stuck_state().0, 0);
    }

    #[test]
    fn test_unrealized_as_cost_accumulates() {
        let config = KillSwitchConfig {
            max_position_value: 200.0,
            unrealized_as_warn_fraction: 0.01,
            unrealized_as_kill_fraction: 0.05,
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        // Long position, market dropping
        let _ = ks.report_reducing_quote_status(3.0, 100.0, 10.0, false);
        // Mid drops from 100 to 99.9 = 0.1 drop, position=3 → cost = 3 * 0.1 = 0.3
        let _ = ks.report_reducing_quote_status(3.0, 99.9, 10.0, false);
        let (_, cost) = ks.stuck_state();
        assert!((cost - 0.3).abs() < 0.01, "Expected ~$0.30, got ${cost:.2}");

        // Mid drops more: 99.9 → 99.5 = 0.4 drop → cost += 3 * 0.4 = 1.2
        let _ = ks.report_reducing_quote_status(3.0, 99.5, 10.0, false);
        let (_, cost) = ks.stuck_state();
        assert!((cost - 1.5).abs() < 0.01, "Expected ~$1.50, got ${cost:.2}");
    }

    #[test]
    fn test_unrealized_as_cost_resets_on_small_position() {
        let config = KillSwitchConfig::default();
        let ks = KillSwitch::new(config);

        // Build up cost
        let _ = ks.report_reducing_quote_status(0.5, 100.0, 1.0, false);
        let _ = ks.report_reducing_quote_status(0.5, 99.0, 1.0, false);
        assert!(ks.stuck_state().1 > 0.0);

        // Position drops below threshold → resets
        let _ = ks.report_reducing_quote_status(0.05, 99.0, 1.0, false);
        assert_eq!(ks.stuck_state().1, 0.0);
    }

    #[test]
    fn test_cost_threshold_triggers_before_cycle_threshold() {
        let config = KillSwitchConfig {
            max_position_value: 100.0,
            max_stuck_cycles: 30,
            stuck_warning_cycles: 10,
            unrealized_as_warn_fraction: 0.01,  // $1 warn
            unrealized_as_kill_fraction: 0.05,   // $5 kill
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        // Big position, market moves fast against us
        let _ = ks.report_reducing_quote_status(5.0, 100.0, 10.0, false);
        // Drop from 100 to 99.0 → cost = 5 * 1.0 = $5.0 (kill threshold)
        let result = ks.report_reducing_quote_status(5.0, 99.0, 10.0, false);
        assert_eq!(result, StuckEscalation::Kill);
        // Only 2 cycles, not 30 — cost triggered first
        assert!(ks.stuck_state().0 < 30);
    }

    #[test]
    fn test_cycle_threshold_triggers_in_flat_market() {
        let config = KillSwitchConfig {
            max_position_value: 10_000.0,
            max_stuck_cycles: 5,
            stuck_warning_cycles: 3,
            unrealized_as_warn_fraction: 0.01,
            unrealized_as_kill_fraction: 0.05,
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        // Flat market: mid barely moves, but position stuck without reducing quotes
        for _ in 0..3 {
            let result = ks.report_reducing_quote_status(0.5, 100.0, 1.0, false);
            if result == StuckEscalation::ForceReducingQuotes {
                // Warning triggered by cycle count
                return;
            }
        }
        let result = ks.report_reducing_quote_status(0.5, 100.0, 1.0, false);
        assert_eq!(result, StuckEscalation::ForceReducingQuotes);
    }

    #[test]
    fn test_stuck_kill_trigger() {
        let config = KillSwitchConfig {
            max_stuck_cycles: 3,
            stuck_warning_cycles: 2,
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        // Accumulate to kill threshold
        for _ in 0..3 {
            let _ = ks.report_reducing_quote_status(0.5, 100.0, 1.0, false);
        }
        let result = ks.report_reducing_quote_status(0.5, 100.0, 1.0, false);
        assert_eq!(result, StuckEscalation::Kill);
    }

    #[test]
    fn test_checkpoint_stuck_persistence() {
        let config = KillSwitchConfig::default();
        let ks = KillSwitch::new(config);

        // Accumulate some stuck state
        let _ = ks.report_reducing_quote_status(0.5, 100.0, 1.0, false);
        let _ = ks.report_reducing_quote_status(0.5, 99.5, 1.0, false);

        let checkpoint = ks.to_checkpoint();
        assert!(checkpoint.position_stuck_cycles > 0);
        assert!(checkpoint.unrealized_as_cost_usd > 0.0);
    }
}
