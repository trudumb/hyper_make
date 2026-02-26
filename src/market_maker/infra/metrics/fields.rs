//! MetricsInner struct containing all atomic metric fields.

use std::sync::atomic::AtomicU64;
use std::time::Instant;

use super::atomic::AtomicF64;

/// Inner metrics storage with all atomic fields.
pub(super) struct MetricsInner {
    // === Position Metrics ===
    /// Current position in base asset
    pub position: AtomicF64,
    /// Maximum allowed position
    pub max_position: AtomicF64,
    /// Inventory utilization (position / max_position)
    pub inventory_utilization: AtomicF64,
    /// Pending bid exposure (total remaining size on buy orders)
    pub pending_bid_exposure: AtomicF64,
    /// Pending ask exposure (total remaining size on sell orders)
    pub pending_ask_exposure: AtomicF64,
    /// Net pending change (bid - ask exposure)
    pub net_pending_change: AtomicF64,
    /// Worst-case max position if all bids fill
    pub worst_case_max_position: AtomicF64,
    /// Worst-case min position if all asks fill
    pub worst_case_min_position: AtomicF64,

    // === P&L Metrics ===
    /// Daily P&L in USD
    pub daily_pnl: AtomicF64,
    /// Peak P&L for drawdown calculation
    pub peak_pnl: AtomicF64,
    /// Current drawdown percentage
    pub drawdown_pct: AtomicF64,
    /// Total realized P&L
    pub realized_pnl: AtomicF64,
    /// Total unrealized P&L
    pub unrealized_pnl: AtomicF64,

    // === Order Metrics ===
    /// Total orders placed
    pub orders_placed: AtomicU64,
    /// Total orders filled
    pub orders_filled: AtomicU64,
    /// Total orders cancelled
    pub orders_cancelled: AtomicU64,
    /// Total orders modified (queue-preserving updates)
    pub orders_modified: AtomicU64,
    /// Total modify fallbacks (modify failed, fell back to cancel+place)
    pub modify_fallbacks: AtomicU64,
    /// Quote cycles skipped due to BBO crossing detection
    pub bbo_crossing_skips: AtomicU64,
    /// Total fill volume (base asset)
    pub fill_volume: AtomicF64,
    /// Buy fill volume
    pub buy_volume: AtomicF64,
    /// Sell fill volume
    pub sell_volume: AtomicF64,

    // === Market Metrics ===
    /// Current mid price
    pub mid_price: AtomicF64,
    /// Current spread in bps
    pub spread_bps: AtomicF64,
    /// Volatility (sigma)
    pub sigma: AtomicF64,
    /// Jump ratio (RV/BV)
    pub jump_ratio: AtomicF64,
    /// Kappa (fill intensity)
    pub kappa: AtomicF64,

    // === V2 Bayesian Estimator Metrics ===
    /// Kappa posterior standard deviation (uncertainty)
    pub kappa_uncertainty: AtomicF64,
    /// 95% credible interval lower bound
    pub kappa_95_lower: AtomicF64,
    /// 95% credible interval upper bound
    pub kappa_95_upper: AtomicF64,
    /// Soft toxicity score [0, 1] from mixture model
    pub toxicity_score: AtomicF64,
    /// (κ, σ) correlation coefficient [-1, 1]
    pub param_correlation: AtomicF64,
    /// Adverse selection factor φ(AS) [0.5, 1.0]
    pub as_factor: AtomicF64,

    // === Robust Kappa Metrics ===
    /// Robust kappa effective sample size (accounts for outlier downweighting)
    pub robust_kappa_ess: AtomicF64,
    /// Robust kappa outlier count (observations with weight < 0.5)
    pub robust_kappa_outliers: AtomicU64,
    /// Robust kappa degrees of freedom (tail heaviness, ν parameter)
    pub robust_kappa_nu: AtomicF64,
    /// Robust kappa observation count
    pub robust_kappa_obs_count: AtomicU64,

    // === Estimator Metrics ===
    /// Microprice deviation from mid
    pub microprice_deviation_bps: AtomicF64,
    /// Book imbalance [-1, 1]
    pub book_imbalance: AtomicF64,
    /// Flow imbalance [-1, 1]
    pub flow_imbalance: AtomicF64,
    /// Beta book coefficient
    pub beta_book: AtomicF64,
    /// Beta flow coefficient
    pub beta_flow: AtomicF64,

    // === Risk Metrics ===
    /// Kill switch triggered (1 = triggered, 0 = not)
    pub kill_switch_triggered: AtomicU64,
    /// Cascade severity [0, 1]
    pub cascade_severity: AtomicF64,
    /// Adverse selection in bps
    pub adverse_selection_bps: AtomicF64,
    /// AS at 500ms horizon (bps)
    pub as_500ms_bps: AtomicF64,
    /// AS at 1000ms horizon (bps)
    pub as_1000ms_bps: AtomicF64,
    /// AS at 2000ms horizon (bps)
    pub as_2000ms_bps: AtomicF64,
    /// Current best AS horizon (ms)
    pub as_best_horizon_ms: AtomicU64,
    /// Tail risk multiplier
    pub tail_risk_multiplier: AtomicF64,

    // === Timing Metrics ===
    /// Time since last market data update (seconds)
    pub data_staleness_secs: AtomicF64,
    /// Quote cycle latency (milliseconds)
    pub quote_cycle_latency_ms: AtomicF64,

    // === Volatility Regime ===
    /// Current volatility regime (0=Low, 1=Normal, 2=High, 3=Extreme)
    pub volatility_regime: AtomicU64,

    // === Kelly-Stochastic Metrics ===
    /// Whether Kelly-Stochastic allocation is enabled (1 = enabled, 0 = disabled)
    pub kelly_stochastic_enabled: AtomicU64,
    /// Calibrated alpha (informed probability) at touch [0, 1]
    pub kelly_alpha_touch: AtomicF64,
    /// Current Kelly fraction being used [0, 1]
    pub kelly_fraction: AtomicF64,
    /// Characteristic depth for alpha decay in bps
    pub kelly_alpha_decay_bps: AtomicF64,

    // === Connection Health Metrics ===
    /// WebSocket connected (1 = connected, 0 = disconnected)
    pub websocket_connected: AtomicU64,
    /// Time since last trade update (ms)
    pub last_trade_age_ms: AtomicU64,
    /// Time since last L2 book update (ms)
    pub last_book_age_ms: AtomicU64,
    /// Total WebSocket reconnections since start
    pub ws_reconnection_count: AtomicU64,
    /// Number of pong timeout events (connection died before server responded)
    pub ws_pong_timeout_count: AtomicU64,
    /// Average ping round-trip latency (ms)
    pub ws_ping_latency_ms: AtomicF64,
    /// Time since last successful pong (ms) - indicates connection liveness
    pub ws_time_since_pong_ms: AtomicU64,
    /// Connection supervisor consecutive stale readings
    pub supervisor_stale_count: AtomicU64,
    /// Connection supervisor reconnect signal count
    pub supervisor_reconnect_signals: AtomicU64,

    // === Data Quality Metrics ===
    /// Total data quality issues detected
    pub data_quality_issues_total: AtomicU64,
    /// Cumulative sequence gaps detected
    pub message_loss_count: AtomicU64,
    /// Crossed book incidents
    pub crossed_book_incidents: AtomicU64,

    // === Exchange Position Limits Metrics ===
    /// Exchange max long position allowed
    pub exchange_max_long: AtomicF64,
    /// Exchange max short position allowed
    pub exchange_max_short: AtomicF64,
    /// Exchange available capacity to buy
    pub exchange_available_buy: AtomicF64,
    /// Exchange available capacity to sell
    pub exchange_available_sell: AtomicF64,
    /// Effective bid limit (min of local and exchange)
    pub exchange_effective_bid: AtomicF64,
    /// Effective ask limit (min of local and exchange)
    pub exchange_effective_ask: AtomicF64,
    /// Age of exchange limits data in milliseconds
    pub exchange_limits_age_ms: AtomicU64,
    /// Whether exchange limits are valid (1 = valid, 0 = not fetched)
    pub exchange_limits_valid: AtomicU64,

    // === Calibration Fill Rate Controller Metrics ===
    /// Calibration gamma multiplier [0.3, 1.0] - lower = more fill-hungry
    pub calibration_gamma_mult: AtomicF64,
    /// Calibration progress [0.0, 1.0]
    pub calibration_progress: AtomicF64,
    /// Fill count in lookback window
    pub calibration_fill_count: AtomicU64,
    /// Whether calibration is complete (1 = yes, 0 = no)
    pub calibration_complete: AtomicU64,

    // === Learned Parameters Metrics ===
    /// Learned alpha_touch (informed trader probability at touch) [0, 1]
    pub learned_alpha_touch: AtomicF64,
    /// Learned kappa (fill intensity)
    pub learned_kappa: AtomicF64,
    /// Learned spread floor in bps
    pub learned_spread_floor_bps: AtomicF64,
    /// Total observations for learned parameters
    pub learned_params_observations: AtomicU64,
    /// Whether learned parameters are calibrated (tier1_ready) (1 = yes, 0 = no)
    pub learned_params_calibrated: AtomicU64,

    // === Impulse Control Metrics ===
    /// Whether impulse control is enabled (1 = enabled, 0 = disabled)
    pub impulse_control_enabled: AtomicU64,
    /// Available execution budget tokens
    pub impulse_budget_available: AtomicF64,
    /// Budget utilization (spent / (earned + initial))
    pub impulse_budget_utilization: AtomicF64,
    /// Total tokens earned from fills
    pub impulse_budget_earned: AtomicF64,
    /// Total tokens spent on actions
    pub impulse_budget_spent: AtomicF64,
    /// Actions blocked by Δλ filter (improvement too small)
    pub impulse_filter_blocked: AtomicU64,
    /// Actions blocked by queue lock (high P(fill) orders)
    pub impulse_queue_locked: AtomicU64,
    /// Full cycles skipped due to insufficient budget
    pub impulse_budget_skipped: AtomicU64,

    /// Start time for uptime calculation
    pub start_time: Instant,
}

impl MetricsInner {
    pub(super) fn new() -> Self {
        Self {
            position: AtomicF64::new(0.0),
            max_position: AtomicF64::new(0.0),
            inventory_utilization: AtomicF64::new(0.0),
            pending_bid_exposure: AtomicF64::new(0.0),
            pending_ask_exposure: AtomicF64::new(0.0),
            net_pending_change: AtomicF64::new(0.0),
            worst_case_max_position: AtomicF64::new(0.0),
            worst_case_min_position: AtomicF64::new(0.0),
            daily_pnl: AtomicF64::new(0.0),
            peak_pnl: AtomicF64::new(0.0),
            drawdown_pct: AtomicF64::new(0.0),
            realized_pnl: AtomicF64::new(0.0),
            unrealized_pnl: AtomicF64::new(0.0),
            orders_placed: AtomicU64::new(0),
            orders_filled: AtomicU64::new(0),
            orders_cancelled: AtomicU64::new(0),
            orders_modified: AtomicU64::new(0),
            modify_fallbacks: AtomicU64::new(0),
            bbo_crossing_skips: AtomicU64::new(0),
            fill_volume: AtomicF64::new(0.0),
            buy_volume: AtomicF64::new(0.0),
            sell_volume: AtomicF64::new(0.0),
            mid_price: AtomicF64::new(0.0),
            spread_bps: AtomicF64::new(0.0),
            sigma: AtomicF64::new(0.0),
            jump_ratio: AtomicF64::new(0.0),
            kappa: AtomicF64::new(0.0),
            // V2 Bayesian Estimator defaults
            kappa_uncertainty: AtomicF64::new(0.0),
            kappa_95_lower: AtomicF64::new(0.0),
            kappa_95_upper: AtomicF64::new(0.0),
            toxicity_score: AtomicF64::new(0.0),
            param_correlation: AtomicF64::new(0.0),
            as_factor: AtomicF64::new(1.0),
            // Robust kappa defaults
            robust_kappa_ess: AtomicF64::new(0.0),
            robust_kappa_outliers: AtomicU64::new(0),
            robust_kappa_nu: AtomicF64::new(4.0), // Default Student-t ν
            robust_kappa_obs_count: AtomicU64::new(0),
            microprice_deviation_bps: AtomicF64::new(0.0),
            book_imbalance: AtomicF64::new(0.0),
            flow_imbalance: AtomicF64::new(0.0),
            beta_book: AtomicF64::new(0.0),
            beta_flow: AtomicF64::new(0.0),
            kill_switch_triggered: AtomicU64::new(0),
            cascade_severity: AtomicF64::new(0.0),
            adverse_selection_bps: AtomicF64::new(0.0),
            as_500ms_bps: AtomicF64::new(0.0),
            as_1000ms_bps: AtomicF64::new(0.0),
            as_2000ms_bps: AtomicF64::new(0.0),
            as_best_horizon_ms: AtomicU64::new(1000),
            tail_risk_multiplier: AtomicF64::new(1.0),
            data_staleness_secs: AtomicF64::new(0.0),
            quote_cycle_latency_ms: AtomicF64::new(0.0),
            volatility_regime: AtomicU64::new(1), // Normal
            // Kelly-Stochastic defaults
            kelly_stochastic_enabled: AtomicU64::new(0),
            kelly_alpha_touch: AtomicF64::new(0.15), // Default 15%
            kelly_fraction: AtomicF64::new(0.25),    // Default quarter Kelly
            kelly_alpha_decay_bps: AtomicF64::new(10.0), // Default 10 bps
            websocket_connected: AtomicU64::new(0),
            last_trade_age_ms: AtomicU64::new(0),
            last_book_age_ms: AtomicU64::new(0),
            ws_reconnection_count: AtomicU64::new(0),
            ws_pong_timeout_count: AtomicU64::new(0),
            ws_ping_latency_ms: AtomicF64::new(0.0),
            ws_time_since_pong_ms: AtomicU64::new(0),
            supervisor_stale_count: AtomicU64::new(0),
            supervisor_reconnect_signals: AtomicU64::new(0),
            data_quality_issues_total: AtomicU64::new(0),
            message_loss_count: AtomicU64::new(0),
            // Exchange Position Limits defaults — 0.0 is conservative (no capacity
            // until exchange confirms), preventing f64::MAX from leaking into logs/metrics.
            exchange_max_long: AtomicF64::new(0.0),
            exchange_max_short: AtomicF64::new(0.0),
            exchange_available_buy: AtomicF64::new(0.0),
            exchange_available_sell: AtomicF64::new(0.0),
            exchange_effective_bid: AtomicF64::new(0.0),
            exchange_effective_ask: AtomicF64::new(0.0),
            exchange_limits_age_ms: AtomicU64::new(u64::MAX),
            exchange_limits_valid: AtomicU64::new(0),
            crossed_book_incidents: AtomicU64::new(0),
            // Calibration Fill Rate Controller defaults
            calibration_gamma_mult: AtomicF64::new(0.3), // Start fill-hungry
            calibration_progress: AtomicF64::new(0.0),
            calibration_fill_count: AtomicU64::new(0),
            calibration_complete: AtomicU64::new(0),
            // Learned Parameters defaults
            learned_alpha_touch: AtomicF64::new(0.25), // Default prior
            learned_kappa: AtomicF64::new(2000.0),     // Default prior
            learned_spread_floor_bps: AtomicF64::new(5.0), // Default prior
            learned_params_observations: AtomicU64::new(0),
            learned_params_calibrated: AtomicU64::new(0),
            // Impulse Control defaults
            impulse_control_enabled: AtomicU64::new(0),
            impulse_budget_available: AtomicF64::new(0.0),
            impulse_budget_utilization: AtomicF64::new(0.0),
            impulse_budget_earned: AtomicF64::new(0.0),
            impulse_budget_spent: AtomicF64::new(0.0),
            impulse_filter_blocked: AtomicU64::new(0),
            impulse_queue_locked: AtomicU64::new(0),
            impulse_budget_skipped: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }
}
