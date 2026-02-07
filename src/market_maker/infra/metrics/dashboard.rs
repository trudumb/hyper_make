//! Dashboard API data structures and aggregation.
//!
//! Provides JSON-serializable structures for the web dashboard,
//! aggregating data from various system components.

use crate::market_maker::tracking::{CalibrationMetrics, CalibrationTracker};
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Live quote metrics for dashboard header.
#[derive(Clone, Debug, Serialize)]
pub struct LiveQuotes {
    /// Current mid price.
    pub mid: f64,
    /// Current spread in basis points.
    pub spread_bps: f64,
    /// Current inventory position.
    pub inventory: f64,
    /// Current regime classification.
    pub regime: String,
    /// Fill intensity (kappa) estimate.
    pub kappa: f64,
    /// Risk aversion parameter (gamma).
    pub gamma: f64,
    /// Probability of fill at current depth.
    pub fill_prob: f64,
    /// Probability of adverse selection.
    pub adverse_prob: f64,
}

impl Default for LiveQuotes {
    fn default() -> Self {
        Self {
            mid: 0.0,
            spread_bps: 0.0,
            inventory: 0.0,
            regime: "Quiet".to_string(),
            kappa: 0.0,
            gamma: 0.0,
            fill_prob: 0.0,
            adverse_prob: 0.0,
        }
    }
}

/// P&L attribution breakdown.
#[derive(Clone, Debug, Serialize)]
pub struct PnLAttribution {
    /// Revenue from capturing bid-ask spread.
    pub spread_capture: f64,
    /// Loss from adverse selection (informed flow).
    pub adverse_selection: f64,
    /// Gain/loss from inventory carry.
    pub inventory_cost: f64,
    /// Exchange fees paid.
    pub fees: f64,
    /// Total P&L.
    pub total: f64,
}

impl Default for PnLAttribution {
    fn default() -> Self {
        Self {
            spread_capture: 0.0,
            adverse_selection: 0.0,
            inventory_cost: 0.0,
            fees: 0.0,
            total: 0.0,
        }
    }
}

/// Regime probability distribution.
#[derive(Clone, Debug, Serialize)]
pub struct RegimeProbabilities {
    pub quiet: f64,
    pub trending: f64,
    pub volatile: f64,
    pub cascade: f64,
}

impl Default for RegimeProbabilities {
    fn default() -> Self {
        Self {
            quiet: 1.0,
            trending: 0.0,
            volatile: 0.0,
            cascade: 0.0,
        }
    }
}

/// Historical regime snapshot for time series.
#[derive(Clone, Debug, Serialize)]
pub struct RegimeSnapshot {
    /// Time label (HH:MM format).
    pub time: String,
    /// Unix timestamp in milliseconds.
    pub timestamp_ms: i64,
    /// Probability of Quiet regime.
    #[serde(rename = "Quiet")]
    pub quiet: f64,
    /// Probability of Trending regime.
    #[serde(rename = "Trending")]
    pub trending: f64,
    /// Probability of Volatile regime.
    #[serde(rename = "Volatile")]
    pub volatile: f64,
    /// Probability of Cascade regime.
    #[serde(rename = "Cascade")]
    pub cascade: f64,
}

/// Current regime state with history.
#[derive(Clone, Debug, Serialize)]
pub struct RegimeState {
    /// Current regime classification.
    pub current: String,
    /// Current regime probabilities.
    pub probabilities: RegimeProbabilities,
    /// Historical regime probabilities (60 minutes, 1-min intervals).
    pub history: Vec<RegimeSnapshot>,
}

impl Default for RegimeState {
    fn default() -> Self {
        Self {
            current: "Quiet".to_string(),
            probabilities: RegimeProbabilities::default(),
            history: Vec::new(),
        }
    }
}

/// Simplified fill record for dashboard.
#[derive(Clone, Debug, Serialize)]
pub struct FillRecord {
    /// Time label (HH:MM format).
    pub time: String,
    /// Unix timestamp in milliseconds.
    pub timestamp_ms: i64,
    /// P&L from this fill.
    pub pnl: f64,
    /// Cumulative P&L.
    #[serde(rename = "cumPnl")]
    pub cum_pnl: f64,
    /// Fill side ("BID" or "ASK").
    pub side: String,
    /// Adverse selection measurement (bps).
    #[serde(rename = "as")]
    pub adverse_selection: String,
}

/// Signal information for audit display.
#[derive(Clone, Debug, Serialize)]
pub struct SignalInfo {
    /// Signal name.
    pub signal: String,
    /// Mutual information (bits).
    pub mi: f64,
    /// Correlation with outcome.
    pub corr: f64,
    /// Lag in milliseconds.
    pub lag: i64,
    /// Regime variance multiplier.
    pub rv: f64,
}

/// Calibration state for dashboard.
#[derive(Clone, Debug, Default, Serialize)]
pub struct CalibrationState {
    pub fill: CalibrationMetrics,
    pub adverse_selection: CalibrationMetrics,
}

// ============================================================================
// New Dashboard Visualization Structures
// ============================================================================

/// Single order book level for heat map.
#[derive(Clone, Debug, Serialize)]
pub struct BookLevel {
    pub price: f64,
    pub size: f64,
}

/// Order book snapshot for 2D heat map visualization.
#[derive(Clone, Debug, Serialize)]
pub struct BookSnapshot {
    pub time: String,
    pub timestamp_ms: i64,
    pub bids: Vec<BookLevel>,
    pub asks: Vec<BookLevel>,
}

/// Mid price point for live price chart.
#[derive(Clone, Debug, Serialize)]
pub struct PricePoint {
    pub time: String,
    pub timestamp_ms: i64,
    pub price: f64,
}

/// Our quote snapshot for overlay visualization.
#[derive(Clone, Debug, Serialize)]
pub struct QuoteSnapshot {
    pub time: String,
    pub timestamp_ms: i64,
    pub bid_prices: Vec<f64>,
    pub ask_prices: Vec<f64>,
    pub spread_bps: f64,
}

/// Quote fill statistics by level.
#[derive(Clone, Debug, Serialize)]
pub struct QuoteFillStats {
    pub level: usize,
    pub side: String,
    pub fill_count: u32,
    pub total_size: f64,
}

/// Spread distribution bucket.
#[derive(Clone, Debug, Serialize)]
pub struct SpreadBucket {
    pub range_bps: String,
    pub count: u32,
    pub percentage: f64,
}

// ============================================================================
// Signal & Decision Visualization Structures
// ============================================================================

/// Snapshot of all signal values at a point in time.
/// Used to visualize signal evolution and understand decision inputs.
#[derive(Clone, Debug, Serialize)]
pub struct SignalSnapshot {
    /// Unix timestamp in milliseconds.
    pub timestamp_ms: i64,
    /// Time label (HH:MM:SS format).
    pub time: String,

    // Kappa (fill intensity) estimates
    /// Posterior mean of kappa.
    pub kappa: f64,
    /// Confidence in kappa estimate [0, 1].
    pub kappa_confidence: f64,
    /// 95% credible interval lower bound.
    pub kappa_ci_lower: f64,
    /// 95% credible interval upper bound.
    pub kappa_ci_upper: f64,

    // Microprice signals
    /// Fair price incorporating book/flow imbalance.
    pub microprice: f64,
    /// Book imbalance weight.
    pub beta_book: f64,
    /// Flow imbalance weight.
    pub beta_flow: f64,

    // Momentum signals
    /// Momentum in basis points (signed).
    pub momentum_bps: f64,
    /// Buy/sell aggressor imbalance [-1, 1].
    pub flow_imbalance: f64,
    /// Falling knife score [0, 3].
    pub falling_knife: f64,

    // Volatility signals
    /// Clean volatility (bipower variation).
    pub sigma: f64,
    /// Jump ratio (RV/BV) for toxic flow detection.
    pub jump_ratio: f64,

    // Regime probabilities
    /// Probability of Quiet regime.
    pub regime_quiet: f64,
    /// Probability of Trending regime.
    pub regime_trending: f64,
    /// Probability of Volatile regime.
    pub regime_volatile: f64,
    /// Probability of Cascade regime.
    pub regime_cascade: f64,

    // Changepoint detection
    /// Probability of changepoint in last 5 observations.
    pub cp_prob_5: f64,
    /// Whether changepoint was detected.
    pub cp_detected: bool,
}

impl Default for SignalSnapshot {
    fn default() -> Self {
        Self {
            timestamp_ms: 0,
            time: String::new(),
            kappa: 0.0,
            kappa_confidence: 0.0,
            kappa_ci_lower: 0.0,
            kappa_ci_upper: 0.0,
            microprice: 0.0,
            beta_book: 0.0,
            beta_flow: 0.0,
            momentum_bps: 0.0,
            flow_imbalance: 0.0,
            falling_knife: 0.0,
            sigma: 0.0,
            jump_ratio: 0.0,
            regime_quiet: 1.0,
            regime_trending: 0.0,
            regime_volatile: 0.0,
            regime_cascade: 0.0,
            cp_prob_5: 0.0,
            cp_detected: false,
        }
    }
}

/// Record of a quote decision with context.
/// Used to understand WHY a specific spread was chosen.
#[derive(Clone, Debug, Serialize)]
pub struct QuoteDecisionRecord {
    /// Unix timestamp in milliseconds.
    pub timestamp_ms: i64,
    /// Time label (HH:MM:SS format).
    pub time: String,

    // What spread was set
    /// Bid spread in basis points.
    pub bid_spread_bps: f64,
    /// Ask spread in basis points.
    pub ask_spread_bps: f64,

    // Key inputs to the decision
    /// Kappa (fill intensity) used.
    pub input_kappa: f64,
    /// Gamma (risk aversion) used.
    pub input_gamma: f64,
    /// Sigma (volatility) used.
    pub input_sigma: f64,
    /// Inventory position.
    pub input_inventory: f64,

    // Decision factors
    /// Current regime classification.
    pub regime: String,
    /// Momentum-based spread adjustment (multiplier).
    pub momentum_adjustment: f64,
    /// Inventory skew applied (bps).
    pub inventory_skew: f64,
    /// Reason for defensive quoting (if any).
    pub defensive_reason: Option<String>,
}

impl Default for QuoteDecisionRecord {
    fn default() -> Self {
        Self {
            timestamp_ms: 0,
            time: String::new(),
            bid_spread_bps: 0.0,
            ask_spread_bps: 0.0,
            input_kappa: 0.0,
            input_gamma: 0.0,
            input_sigma: 0.0,
            input_inventory: 0.0,
            regime: "Unknown".to_string(),
            momentum_adjustment: 1.0,
            inventory_skew: 0.0,
            defensive_reason: None,
        }
    }
}

/// Current kappa diagnostics for detailed view.
#[derive(Clone, Debug, Serialize)]
pub struct KappaDiagnostics {
    /// Posterior mean.
    pub posterior_mean: f64,
    /// Posterior standard deviation.
    pub posterior_std: f64,
    /// Confidence [0, 1].
    pub confidence: f64,
    /// 95% CI lower bound.
    pub ci_95_lower: f64,
    /// 95% CI upper bound.
    pub ci_95_upper: f64,
    /// Coefficient of variation.
    pub cv: f64,
    /// Whether heavy-tailed distribution detected.
    pub is_heavy_tailed: bool,
    /// Number of observations.
    pub observation_count: usize,
    /// Mean fill distance in bps.
    pub mean_distance_bps: f64,
}

impl Default for KappaDiagnostics {
    fn default() -> Self {
        Self {
            posterior_mean: 0.0,
            posterior_std: 0.0,
            confidence: 0.0,
            ci_95_lower: 0.0,
            ci_95_upper: 0.0,
            cv: 0.0,
            is_heavy_tailed: false,
            observation_count: 0,
            mean_distance_bps: 0.0,
        }
    }
}

/// Current changepoint/regime diagnostics.
#[derive(Clone, Debug, Serialize)]
pub struct ChangepointDiagnostics {
    /// P(changepoint 1 obs ago).
    pub cp_prob_1: f64,
    /// P(changepoint 5 obs ago).
    pub cp_prob_5: f64,
    /// P(changepoint 10 obs ago).
    pub cp_prob_10: f64,
    /// Most likely run length (age of current regime).
    pub run_length: u32,
    /// Entropy (uncertainty over run length).
    pub entropy: f64,
    /// Threshold detection status.
    pub detected: bool,
}

impl Default for ChangepointDiagnostics {
    fn default() -> Self {
        Self {
            cp_prob_1: 0.0,
            cp_prob_5: 0.0,
            cp_prob_10: 0.0,
            run_length: 0,
            entropy: 0.0,
            detected: false,
        }
    }
}

// ============================================================================
// Feature Health Visualization Structures
// ============================================================================

/// Individual signal health information.
#[derive(Clone, Debug, Serialize)]
pub struct SignalHealthInfo {
    /// Signal name.
    pub name: String,
    /// Current mutual information (bits).
    pub mi: f64,
    /// Trend direction: "up", "down", or "stable".
    pub trend: String,
    /// Estimated half-life in days.
    pub half_life_days: Option<f64>,
    /// Health status: "healthy", "warning", "critical".
    pub status: String,
    /// MI change percentage over last period.
    pub mi_change_pct: f64,
}

impl Default for SignalHealthInfo {
    fn default() -> Self {
        Self {
            name: String::new(),
            mi: 0.0,
            trend: "stable".to_string(),
            half_life_days: None,
            status: "healthy".to_string(),
            mi_change_pct: 0.0,
        }
    }
}

/// Signal decay alert.
#[derive(Clone, Debug, Serialize)]
pub struct SignalAlertInfo {
    /// Signal name.
    pub signal_name: String,
    /// Severity: "warning" or "critical".
    pub severity: String,
    /// Alert message.
    pub message: String,
    /// Unix timestamp when alert was triggered.
    pub timestamp_ms: i64,
}

/// Signal decay tracking state for dashboard.
#[derive(Clone, Debug, Default, Serialize)]
pub struct SignalDecayState {
    /// All tracked signals with health info.
    pub signals: Vec<SignalHealthInfo>,
    /// Active alerts.
    pub alerts: Vec<SignalAlertInfo>,
    /// Total number of tracked signals.
    pub signal_count: usize,
    /// Number of signals in warning/critical state.
    pub issue_count: usize,
}

/// Feature correlation matrix state.
#[derive(Clone, Debug, Serialize)]
pub struct FeatureCorrelationState {
    /// Feature names in matrix order.
    pub feature_names: Vec<String>,
    /// NxN correlation matrix (row-major).
    pub correlation_matrix: Vec<f64>,
    /// Variance Inflation Factors.
    pub vif: Vec<f64>,
    /// Matrix condition number (stability indicator).
    pub condition_number: f64,
    /// Pairs with correlation above threshold.
    pub highly_correlated_pairs: Vec<CorrelatedPair>,
    /// Whether enough observations for valid matrix.
    pub is_valid: bool,
}

impl Default for FeatureCorrelationState {
    fn default() -> Self {
        Self {
            feature_names: Vec::new(),
            correlation_matrix: Vec::new(),
            vif: Vec::new(),
            condition_number: 1.0,
            highly_correlated_pairs: Vec::new(),
            is_valid: false,
        }
    }
}

/// A highly correlated feature pair.
#[derive(Clone, Debug, Serialize)]
pub struct CorrelatedPair {
    pub feature_a: String,
    pub feature_b: String,
    pub correlation: f64,
}

/// Single feature validation info.
#[derive(Clone, Debug, Serialize)]
pub struct FeatureValidationInfo {
    /// Feature name.
    pub name: String,
    /// Current value.
    pub value: f64,
    /// Lower bound (learned from percentiles).
    pub lower_bound: f64,
    /// Upper bound (learned from percentiles).
    pub upper_bound: f64,
    /// Status: "valid", "out_of_bounds", "stale", "invalid".
    pub status: String,
    /// Milliseconds since last update.
    pub staleness_ms: u64,
}

impl Default for FeatureValidationInfo {
    fn default() -> Self {
        Self {
            name: String::new(),
            value: 0.0,
            lower_bound: f64::NEG_INFINITY,
            upper_bound: f64::INFINITY,
            status: "valid".to_string(),
            staleness_ms: 0,
        }
    }
}

/// Feature validation state for dashboard.
#[derive(Clone, Debug, Default, Serialize)]
pub struct FeatureValidationState {
    /// All validated features.
    pub features: Vec<FeatureValidationInfo>,
    /// Total issue count.
    pub issue_count: usize,
    /// Issue rate (issues / total validations).
    pub issue_rate: f64,
    /// Total validation count.
    pub validation_count: usize,
}

/// Lag analysis state for cross-exchange signals.
#[derive(Clone, Debug, Serialize)]
pub struct LagAnalysisState {
    /// Optimal lag in milliseconds (negative = signal leads).
    pub optimal_lag_ms: i64,
    /// Mutual information at optimal lag.
    pub mi_at_lag: f64,
    /// Cross-correlation function: (lag_ms, correlation).
    pub ccf: Vec<LagPoint>,
    /// Whether signal has enough data to be ready.
    pub signal_ready: bool,
    /// Signal name being analyzed.
    pub signal_name: String,
    /// Target name.
    pub target_name: String,
}

impl Default for LagAnalysisState {
    fn default() -> Self {
        Self {
            optimal_lag_ms: 0,
            mi_at_lag: 0.0,
            ccf: Vec::new(),
            signal_ready: false,
            signal_name: "binance_mid".to_string(),
            target_name: "hyperliquid_mid".to_string(),
        }
    }
}

/// Single point in cross-correlation function.
#[derive(Clone, Debug, Serialize)]
pub struct LagPoint {
    pub lag_ms: i64,
    pub correlation: f64,
}

/// Current values of interaction signals.
#[derive(Clone, Debug, Serialize)]
pub struct InteractionSignalState {
    /// Volatility × Momentum interaction.
    pub vol_x_momentum: f64,
    /// Regime × Inventory interaction.
    pub regime_x_inventory: f64,
    /// Jump × Flow interaction.
    pub jump_x_flow: f64,
    /// Timestamp of last update.
    pub timestamp_ms: i64,
}

impl Default for InteractionSignalState {
    fn default() -> Self {
        Self {
            vol_x_momentum: 0.0,
            regime_x_inventory: 0.0,
            jump_x_flow: 0.0,
            timestamp_ms: 0,
        }
    }
}

/// Combined feature health state for dashboard.
#[derive(Clone, Debug, Default, Serialize)]
pub struct FeatureHealthState {
    /// Signal decay tracking.
    pub signal_decay: SignalDecayState,
    /// Feature correlation matrix.
    pub correlation: FeatureCorrelationState,
    /// Feature validation status.
    pub validation: FeatureValidationState,
    /// Cross-exchange lag analysis.
    pub lag_analysis: LagAnalysisState,
    /// Interaction signal values.
    pub interactions: InteractionSignalState,
}

// ============================================================================
// Dashboard State
// ============================================================================

/// Complete dashboard state for API response.
#[derive(Clone, Debug, Serialize)]
pub struct DashboardState {
    pub quotes: LiveQuotes,
    pub pnl: PnLAttribution,
    pub regime: RegimeState,
    pub fills: Vec<FillRecord>,
    pub calibration: CalibrationState,
    pub signals: Vec<SignalInfo>,
    pub timestamp_ms: i64,

    // Visualization data
    pub book_history: Vec<BookSnapshot>,
    pub price_history: Vec<PricePoint>,
    pub quote_history: Vec<QuoteSnapshot>,
    pub quote_fill_stats: Vec<QuoteFillStats>,
    pub spread_distribution: Vec<SpreadBucket>,

    // Signal & decision visualization (NEW)
    /// Time series of signal values for pattern visualization.
    pub signal_history: Vec<SignalSnapshot>,
    /// Recent quote decisions with context for debugging.
    pub decision_history: Vec<QuoteDecisionRecord>,
    /// Current kappa estimation diagnostics.
    pub kappa_diagnostics: KappaDiagnostics,
    /// Current changepoint detection state.
    pub changepoint_diagnostics: ChangepointDiagnostics,

    // Feature health visualization
    /// Combined feature health state (decay, correlation, validation, lag, interactions).
    pub feature_health: FeatureHealthState,
}

impl Default for DashboardState {
    fn default() -> Self {
        Self {
            quotes: LiveQuotes::default(),
            pnl: PnLAttribution::default(),
            regime: RegimeState::default(),
            fills: Vec::new(),
            calibration: CalibrationState::default(),
            signals: default_signals(),
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
            // Visualization data
            book_history: Vec::new(),
            price_history: Vec::new(),
            quote_history: Vec::new(),
            quote_fill_stats: Vec::new(),
            spread_distribution: Vec::new(),
            // Signal & decision visualization
            signal_history: Vec::new(),
            decision_history: Vec::new(),
            kappa_diagnostics: KappaDiagnostics::default(),
            changepoint_diagnostics: ChangepointDiagnostics::default(),
            // Feature health
            feature_health: FeatureHealthState::default(),
        }
    }
}

/// Default signal list (static for now).
pub fn default_signals() -> Vec<SignalInfo> {
    vec![
        SignalInfo {
            signal: "binance_lead".to_string(),
            mi: 0.089,
            corr: 0.31,
            lag: -150,
            rv: 2.3,
        },
        SignalInfo {
            signal: "trade_imb_1s".to_string(),
            mi: 0.067,
            corr: 0.24,
            lag: 0,
            rv: 1.2,
        },
        SignalInfo {
            signal: "microprice_imb".to_string(),
            mi: 0.045,
            corr: 0.19,
            lag: 0,
            rv: 0.8,
        },
        SignalInfo {
            signal: "funding_x_imb".to_string(),
            mi: 0.041,
            corr: 0.15,
            lag: 0,
            rv: 3.1,
        },
        SignalInfo {
            signal: "oi_change_1m".to_string(),
            mi: 0.023,
            corr: 0.08,
            lag: 0,
            rv: 0.6,
        },
    ]
}

/// Classify regime from continuous signals.
pub fn classify_regime(cascade_severity: f64, jump_ratio: f64, sigma: f64) -> String {
    if cascade_severity > 0.5 || jump_ratio > 5.0 {
        "Cascade".to_string()
    } else if jump_ratio > 3.0 || cascade_severity > 0.2 {
        "Volatile".to_string()
    } else if sigma > 0.002 {
        // High volatility = trending
        "Trending".to_string()
    } else {
        "Quiet".to_string()
    }
}

/// Compute soft regime probabilities using sigmoid-like functions.
pub fn compute_regime_probabilities(
    cascade_severity: f64,
    jump_ratio: f64,
    sigma: f64,
) -> RegimeProbabilities {
    // Use sigmoid functions for smooth transitions
    let cascade_prob = sigmoid(cascade_severity, 0.3, 10.0) + sigmoid(jump_ratio, 4.0, 2.0);
    let volatile_prob = sigmoid(jump_ratio, 2.5, 3.0) + sigmoid(cascade_severity, 0.15, 8.0);
    let trending_prob = sigmoid(sigma, 0.0015, 2000.0);

    // Normalize to sum to 1
    let total = cascade_prob + volatile_prob + trending_prob + 1.0; // +1 for quiet baseline
    let quiet_prob = 1.0 / total;

    RegimeProbabilities {
        quiet: quiet_prob,
        trending: trending_prob / total,
        volatile: volatile_prob / total,
        cascade: cascade_prob / total,
    }
}

/// Sigmoid function for smooth transitions.
fn sigmoid(x: f64, center: f64, steepness: f64) -> f64 {
    1.0 / (1.0 + (-steepness * (x - center)).exp())
}

/// Configuration for dashboard aggregator.
#[derive(Clone, Debug)]
pub struct DashboardConfig {
    /// Maximum regime history entries (1 per minute).
    pub max_regime_history: usize,
    /// Maximum fill records to keep.
    pub max_fills: usize,
    /// Regime snapshot interval.
    pub regime_snapshot_interval: Duration,
    /// Maximum price history entries (30 min at 1/sec).
    pub max_price_history: usize,
    /// Maximum book history entries (6 min at 1/sec).
    pub max_book_history: usize,
    /// Price snapshot interval.
    pub price_snapshot_interval: Duration,
    /// Book snapshot interval.
    pub book_snapshot_interval: Duration,
    /// Maximum signal history entries (5 min at 1/sec).
    pub max_signal_history: usize,
    /// Maximum decision history entries.
    pub max_decision_history: usize,
    /// Signal snapshot interval.
    pub signal_snapshot_interval: Duration,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            max_regime_history: 60, // 1 hour at 1-minute intervals
            max_fills: 100,
            regime_snapshot_interval: Duration::from_secs(5), // More responsive for dashboard
            max_price_history: 1800,                          // 30 min at 1/sec
            max_book_history: 360,                            // 6 min at 1/sec
            price_snapshot_interval: Duration::from_secs(1),
            book_snapshot_interval: Duration::from_secs(1),
            max_signal_history: 300,   // 5 min at 1/sec
            max_decision_history: 100, // Last 100 quote decisions
            signal_snapshot_interval: Duration::from_secs(1),
        }
    }
}

/// Input data for generating a dashboard snapshot.
pub struct DashboardSnapshotParams {
    pub mid_price: f64,
    pub spread_bps: f64,
    pub position: f64,
    pub kappa: f64,
    pub gamma: f64,
    pub sigma: f64,
    pub cascade_severity: f64,
    pub jump_ratio: f64,
    pub fill_prob: f64,
    pub adverse_prob: f64,
    pub spread_capture: f64,
    pub adverse_selection: f64,
    pub inventory_cost: f64,
    pub fees: f64,
}

/// Aggregates data from system components for dashboard API.
pub struct DashboardAggregator {
    /// Configuration.
    config: DashboardConfig,
    /// Regime history (thread-safe).
    regime_history: RwLock<VecDeque<RegimeSnapshot>>,
    /// Fill history (thread-safe).
    fill_history: RwLock<VecDeque<FillRecord>>,
    /// Calibration tracker (shared reference).
    calibration: Arc<RwLock<CalibrationTracker>>,
    /// Last regime snapshot time.
    last_regime_snapshot: RwLock<Instant>,
    /// Cumulative P&L tracking.
    cumulative_pnl: RwLock<f64>,
    /// Price history (thread-safe).
    price_history: RwLock<VecDeque<PricePoint>>,
    /// Book history (thread-safe).
    book_history: RwLock<VecDeque<BookSnapshot>>,
    /// Last price snapshot time.
    last_price_snapshot: RwLock<Instant>,
    /// Last book snapshot time.
    last_book_snapshot: RwLock<Instant>,
    /// Accumulated spread capture P&L.
    total_spread_capture: RwLock<f64>,
    /// Accumulated adverse selection losses.
    total_adverse_selection: RwLock<f64>,
    /// Accumulated inventory cost.
    total_inventory_cost: RwLock<f64>,
    /// Accumulated fees.
    total_fees: RwLock<f64>,
    /// Signal snapshot history (thread-safe).
    signal_history: RwLock<VecDeque<SignalSnapshot>>,
    /// Quote decision history (thread-safe).
    decision_history: RwLock<VecDeque<QuoteDecisionRecord>>,
    /// Last signal snapshot time.
    last_signal_snapshot: RwLock<Instant>,
    /// Latest kappa diagnostics.
    kappa_diagnostics: RwLock<KappaDiagnostics>,
    /// Latest changepoint diagnostics.
    changepoint_diagnostics: RwLock<ChangepointDiagnostics>,
}

impl DashboardAggregator {
    pub fn new(config: DashboardConfig, calibration: Arc<RwLock<CalibrationTracker>>) -> Self {
        Self {
            config,
            regime_history: RwLock::new(VecDeque::new()),
            fill_history: RwLock::new(VecDeque::new()),
            calibration,
            // Initialize to 60s in the past to trigger immediate first snapshot
            last_regime_snapshot: RwLock::new(Instant::now() - Duration::from_secs(60)),
            cumulative_pnl: RwLock::new(0.0),
            price_history: RwLock::new(VecDeque::new()),
            book_history: RwLock::new(VecDeque::new()),
            last_price_snapshot: RwLock::new(Instant::now()),
            last_book_snapshot: RwLock::new(Instant::now()),
            total_spread_capture: RwLock::new(0.0),
            total_adverse_selection: RwLock::new(0.0),
            total_inventory_cost: RwLock::new(0.0),
            total_fees: RwLock::new(0.0),
            signal_history: RwLock::new(VecDeque::new()),
            decision_history: RwLock::new(VecDeque::new()),
            last_signal_snapshot: RwLock::new(Instant::now()),
            kappa_diagnostics: RwLock::new(KappaDiagnostics::default()),
            changepoint_diagnostics: RwLock::new(ChangepointDiagnostics::default()),
        }
    }

    /// Record a regime snapshot if interval has elapsed.
    pub fn maybe_record_regime(&self, probs: &RegimeProbabilities) {
        let mut last_time = self.last_regime_snapshot.write().unwrap();
        if last_time.elapsed() >= self.config.regime_snapshot_interval {
            let snapshot = RegimeSnapshot {
                time: chrono::Local::now().format("%H:%M").to_string(),
                timestamp_ms: chrono::Utc::now().timestamp_millis(),
                quiet: probs.quiet,
                trending: probs.trending,
                volatile: probs.volatile,
                cascade: probs.cascade,
            };

            let mut history = self.regime_history.write().unwrap();
            history.push_back(snapshot);
            while history.len() > self.config.max_regime_history {
                history.pop_front();
            }

            *last_time = Instant::now();
        }
    }

    /// Record a fill event.
    pub fn record_fill(&self, pnl: f64, is_buy: bool, as_bps: f64) {
        let mut cum_pnl = self.cumulative_pnl.write().unwrap();
        *cum_pnl += pnl;

        let record = FillRecord {
            time: chrono::Local::now().format("%H:%M").to_string(),
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
            pnl,
            cum_pnl: *cum_pnl,
            side: if is_buy { "BID" } else { "ASK" }.to_string(),
            adverse_selection: format!("{:.1}", as_bps),
        };

        let mut history = self.fill_history.write().unwrap();
        history.push_back(record);
        while history.len() > self.config.max_fills {
            history.pop_front();
        }
    }

    /// Record a price snapshot if interval has elapsed.
    pub fn record_price(&self, mid: f64) {
        let mut last_time = self.last_price_snapshot.write().unwrap();
        if last_time.elapsed() >= self.config.price_snapshot_interval {
            let now = chrono::Utc::now();
            let snapshot = PricePoint {
                time: now.format("%H:%M:%S").to_string(),
                timestamp_ms: now.timestamp_millis(),
                price: mid,
            };

            let mut history = self.price_history.write().unwrap();
            history.push_back(snapshot);
            while history.len() > self.config.max_price_history {
                history.pop_front();
            }

            *last_time = Instant::now();
        }
    }

    /// Record a book snapshot if interval has elapsed.
    pub fn record_book(&self, bids: &[(f64, f64)], asks: &[(f64, f64)]) {
        let mut last_time = self.last_book_snapshot.write().unwrap();
        if last_time.elapsed() >= self.config.book_snapshot_interval {
            let now = chrono::Utc::now();
            let snapshot = BookSnapshot {
                time: now.format("%H:%M:%S").to_string(),
                timestamp_ms: now.timestamp_millis(),
                bids: bids
                    .iter()
                    .take(10)
                    .map(|(p, s)| BookLevel {
                        price: *p,
                        size: *s,
                    })
                    .collect(),
                asks: asks
                    .iter()
                    .take(10)
                    .map(|(p, s)| BookLevel {
                        price: *p,
                        size: *s,
                    })
                    .collect(),
            };

            let mut history = self.book_history.write().unwrap();
            history.push_back(snapshot);
            while history.len() > self.config.max_book_history {
                history.pop_front();
            }

            *last_time = Instant::now();
        }
    }

    /// Record P&L attribution from a fill.
    ///
    /// Accumulates the components of P&L for dashboard display:
    /// - spread: Revenue from capturing bid-ask spread
    /// - adverse: Loss from adverse selection (negative value)
    /// - inv_cost: Cost from inventory carry (negative value)
    /// - fees: Exchange fees paid (negative value)
    pub fn record_pnl_attribution(&self, spread: f64, adverse: f64, inv_cost: f64, fees: f64) {
        *self.total_spread_capture.write().unwrap() += spread;
        *self.total_adverse_selection.write().unwrap() += adverse;
        *self.total_inventory_cost.write().unwrap() += inv_cost;
        *self.total_fees.write().unwrap() += fees;
    }

    /// Get accumulated spread capture.
    pub fn total_spread_capture(&self) -> f64 {
        *self.total_spread_capture.read().unwrap()
    }

    /// Get accumulated adverse selection losses.
    pub fn total_adverse_selection(&self) -> f64 {
        *self.total_adverse_selection.read().unwrap()
    }

    /// Get accumulated inventory cost.
    pub fn total_inventory_cost(&self) -> f64 {
        *self.total_inventory_cost.read().unwrap()
    }

    /// Get accumulated fees.
    pub fn total_fees(&self) -> f64 {
        *self.total_fees.read().unwrap()
    }

    /// Record a signal snapshot if interval has elapsed.
    pub fn record_signal_snapshot(&self, snapshot: SignalSnapshot) {
        let mut last_time = self.last_signal_snapshot.write().unwrap();
        if last_time.elapsed() >= self.config.signal_snapshot_interval {
            let mut history = self.signal_history.write().unwrap();
            history.push_back(snapshot);
            while history.len() > self.config.max_signal_history {
                history.pop_front();
            }
            *last_time = Instant::now();
        }
    }

    /// Record a quote decision (always recorded, no interval).
    pub fn record_quote_decision(&self, decision: QuoteDecisionRecord) {
        let mut history = self.decision_history.write().unwrap();
        history.push_back(decision);
        while history.len() > self.config.max_decision_history {
            history.pop_front();
        }
    }

    /// Update kappa diagnostics.
    pub fn update_kappa_diagnostics(&self, diagnostics: KappaDiagnostics) {
        *self.kappa_diagnostics.write().unwrap() = diagnostics;
    }

    /// Update changepoint diagnostics.
    pub fn update_changepoint_diagnostics(&self, diagnostics: ChangepointDiagnostics) {
        *self.changepoint_diagnostics.write().unwrap() = diagnostics;
    }

    /// Generate dashboard snapshot from current state.
    pub fn snapshot(&self, params: &DashboardSnapshotParams) -> DashboardState {
        // Compute regime
        let regime = classify_regime(params.cascade_severity, params.jump_ratio, params.sigma);
        let probs =
            compute_regime_probabilities(params.cascade_severity, params.jump_ratio, params.sigma);

        // Record regime snapshot if needed
        self.maybe_record_regime(&probs);

        // Get calibration metrics
        let calibration_summary = {
            let calib = self.calibration.read().unwrap();
            calib.summary()
        };

        // Get regime history
        let regime_history: Vec<RegimeSnapshot> = {
            let history = self.regime_history.read().unwrap();
            history.iter().cloned().collect()
        };

        // Get fill history
        let fills: Vec<FillRecord> = {
            let history = self.fill_history.read().unwrap();
            history.iter().cloned().collect()
        };

        // Get price history
        let price_history: Vec<PricePoint> = {
            let history = self.price_history.read().unwrap();
            history.iter().cloned().collect()
        };

        // Get book history
        let book_history: Vec<BookSnapshot> = {
            let history = self.book_history.read().unwrap();
            history.iter().cloned().collect()
        };

        // Get signal history
        let signal_history: Vec<SignalSnapshot> = {
            let history = self.signal_history.read().unwrap();
            history.iter().cloned().collect()
        };

        // Get decision history
        let decision_history: Vec<QuoteDecisionRecord> = {
            let history = self.decision_history.read().unwrap();
            history.iter().cloned().collect()
        };

        // Get current diagnostics
        let kappa_diagnostics = self.kappa_diagnostics.read().unwrap().clone();
        let changepoint_diagnostics = self.changepoint_diagnostics.read().unwrap().clone();

        DashboardState {
            quotes: LiveQuotes {
                mid: params.mid_price,
                spread_bps: params.spread_bps,
                inventory: params.position,
                regime: regime.clone(),
                kappa: params.kappa,
                gamma: params.gamma,
                fill_prob: params.fill_prob,
                adverse_prob: params.adverse_prob,
            },
            pnl: PnLAttribution {
                spread_capture: params.spread_capture,
                adverse_selection: params.adverse_selection,
                inventory_cost: params.inventory_cost,
                fees: params.fees,
                total: params.spread_capture
                    + params.adverse_selection
                    + params.inventory_cost
                    + params.fees,
            },
            regime: RegimeState {
                current: regime,
                probabilities: probs,
                history: regime_history,
            },
            fills,
            calibration: CalibrationState {
                fill: calibration_summary.fill,
                adverse_selection: calibration_summary.adverse_selection,
            },
            signals: default_signals(),
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
            // Time series visualization data
            book_history,
            price_history,
            quote_history: Vec::new(), // TODO: implement quote history tracking
            quote_fill_stats: Vec::new(),
            spread_distribution: Vec::new(),
            // Signal & decision visualization
            signal_history,
            decision_history,
            kappa_diagnostics,
            changepoint_diagnostics,
            // Feature health (populated separately via update_feature_health)
            feature_health: FeatureHealthState::default(),
        }
    }

    /// Update feature health state.
    pub fn update_feature_health(&self, _state: FeatureHealthState) {
        // Feature health is stored in the aggregator and returned via snapshot_with_feature_health
        // For now, the basic snapshot uses default values
        // The full integration will use snapshot_with_feature_health
    }

    /// Generate dashboard snapshot with feature health data.
    pub fn snapshot_with_feature_health(
        &self,
        params: &DashboardSnapshotParams,
        feature_health: FeatureHealthState,
    ) -> DashboardState {
        let mut state = self.snapshot(params);
        state.feature_health = feature_health;
        state
    }

    /// Get the calibration tracker for recording predictions.
    pub fn calibration(&self) -> Arc<RwLock<CalibrationTracker>> {
        Arc::clone(&self.calibration)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_classification() {
        // Quiet
        assert_eq!(classify_regime(0.0, 1.0, 0.001), "Quiet");

        // Trending
        assert_eq!(classify_regime(0.0, 1.0, 0.003), "Trending");

        // Volatile
        assert_eq!(classify_regime(0.3, 4.0, 0.001), "Volatile");

        // Cascade
        assert_eq!(classify_regime(0.6, 3.0, 0.001), "Cascade");
        assert_eq!(classify_regime(0.3, 6.0, 0.001), "Cascade");
    }

    #[test]
    fn test_regime_probabilities_sum_to_one() {
        let probs = compute_regime_probabilities(0.2, 2.0, 0.001);
        let sum = probs.quiet + probs.trending + probs.volatile + probs.cascade;
        assert!((sum - 1.0).abs() < 0.01, "Sum: {}", sum);
    }

    #[test]
    fn test_dashboard_state_serialization() {
        let state = DashboardState::default();
        let json = serde_json::to_string(&state).expect("Failed to serialize");
        assert!(json.contains("quotes"));
        assert!(json.contains("pnl"));
        assert!(json.contains("regime"));
    }

    #[test]
    fn test_dashboard_aggregator() {
        let calibration = Arc::new(RwLock::new(CalibrationTracker::default()));
        let aggregator = DashboardAggregator::new(DashboardConfig::default(), calibration);

        // Record some fills
        aggregator.record_fill(5.0, true, 2.5);
        aggregator.record_fill(-3.0, false, 4.0);

        // Generate snapshot
        let params = DashboardSnapshotParams {
            mid_price: 50000.0,
            spread_bps: 5.0,
            position: 0.1,
            kappa: 500.0,
            gamma: 0.2,
            sigma: 0.001,
            cascade_severity: 0.1,
            jump_ratio: 1.5,
            fill_prob: 0.2,
            adverse_prob: 0.15,
            spread_capture: 100.0,
            adverse_selection: -50.0,
            inventory_cost: -10.0,
            fees: -5.0,
        };
        let state = aggregator.snapshot(&params);

        assert_eq!(state.quotes.mid, 50000.0);
        assert_eq!(state.fills.len(), 2);
        assert_eq!(state.pnl.total, 35.0);
    }

    #[test]
    fn test_fill_record() {
        let calibration = Arc::new(RwLock::new(CalibrationTracker::default()));
        let aggregator = DashboardAggregator::new(DashboardConfig::default(), calibration);

        aggregator.record_fill(10.0, true, 2.0);
        aggregator.record_fill(-5.0, false, 3.5);

        let fills = aggregator.fill_history.read().unwrap();
        assert_eq!(fills.len(), 2);
        assert_eq!(fills[0].pnl, 10.0);
        assert_eq!(fills[0].cum_pnl, 10.0);
        assert_eq!(fills[1].pnl, -5.0);
        assert_eq!(fills[1].cum_pnl, 5.0);
    }

    #[test]
    fn test_feature_health_serialization() {
        let mut health = FeatureHealthState::default();

        // Add some test data
        health.signal_decay.signals.push(SignalHealthInfo {
            name: "test_signal".to_string(),
            mi: 0.05,
            trend: "down".to_string(),
            half_life_days: Some(14.0),
            status: "warning".to_string(),
            mi_change_pct: -10.0,
        });

        health.correlation.feature_names = vec!["kappa".to_string(), "sigma".to_string()];
        health.correlation.correlation_matrix = vec![1.0, 0.5, 0.5, 1.0];
        health.correlation.is_valid = true;

        health.lag_analysis.optimal_lag_ms = -150;
        health.lag_analysis.mi_at_lag = 0.03;
        health.lag_analysis.signal_ready = true;

        health.interactions.vol_x_momentum = 0.8;
        health.interactions.regime_x_inventory = 0.3;
        health.interactions.jump_x_flow = 0.1;

        let json = serde_json::to_string(&health).expect("Failed to serialize");
        assert!(json.contains("test_signal"));
        assert!(json.contains("optimal_lag_ms"));
        assert!(json.contains("vol_x_momentum"));
        assert!(json.contains("correlation_matrix"));
    }

    #[test]
    fn test_dashboard_with_feature_health() {
        let calibration = Arc::new(RwLock::new(CalibrationTracker::default()));
        let aggregator = DashboardAggregator::new(DashboardConfig::default(), calibration);

        let feature_health = FeatureHealthState {
            interactions: InteractionSignalState {
                vol_x_momentum: 0.5,
                regime_x_inventory: 0.3,
                jump_x_flow: 0.2,
                timestamp_ms: 1000,
            },
            ..Default::default()
        };

        let params = DashboardSnapshotParams {
            mid_price: 50000.0,
            spread_bps: 5.0,
            position: 0.1,
            kappa: 500.0,
            gamma: 0.2,
            sigma: 0.001,
            cascade_severity: 0.1,
            jump_ratio: 1.5,
            fill_prob: 0.2,
            adverse_prob: 0.15,
            spread_capture: 100.0,
            adverse_selection: -50.0,
            inventory_cost: -10.0,
            fees: -5.0,
        };
        let state = aggregator.snapshot_with_feature_health(&params, feature_health);

        assert_eq!(state.feature_health.interactions.vol_x_momentum, 0.5);
        assert_eq!(state.feature_health.interactions.regime_x_inventory, 0.3);
    }
}
