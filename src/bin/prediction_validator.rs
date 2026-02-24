//! Prediction Validator Binary
//!
//! A diagnostic tool for validating model predictive power WITHOUT placing orders.
//! Streams market data, makes predictions in real-time, resolves outcomes from price
//! movements, and reports calibration metrics.
//!
//! ## Purpose
//!
//! After implementing model calibration fixes, we need a way to validate if models
//! actually have predictive power without:
//! 1. Requiring real money at risk
//! 2. Relying on sparse fill data
//! 3. Confusing model performance with execution quality
//!
//! ## What it measures
//!
//! | Model | Prediction | Outcome | Markout |
//! |-------|------------|---------|---------|
//! | InformedFlow | P(informed trade) | Price moves against fill direction | 1s |
//! | PreFillToxicity | P(toxic fill) | Price moves against > threshold | 1s |
//! | RegimeHMM | P(high volatility) | Realized vol > threshold | 30s |
//! | Momentum | P(price continues) | Sign of price change matches | 5s |
//!
//! ## Usage
//!
//! ```bash
//! cargo run --bin prediction_validator -- --asset BTC
//! cargo run --bin prediction_validator -- --asset BTC --report-interval 30 --markout 1000
//! ```

use clap::Parser;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::unbounded_channel;
use tracing::{debug, info, warn};
use tracing_appender::non_blocking::WorkerGuard;

use hyperliquid_rust_sdk::{BaseUrl, InfoClient, Message, Subscription};

// Import calibration infrastructure
use hyperliquid_rust_sdk::market_maker::{BrierScoreTracker, InformationRatioTracker};

// Import estimator components
use hyperliquid_rust_sdk::market_maker::{
    EnhancedASClassifier, InformedFlowEstimator, PreFillASClassifier, TradeFeatures,
    TradeObservation as MicroTradeObs, VolFilterConfig, VolatilityFilter,
};

// Import regime HMM
use hyperliquid_rust_sdk::market_maker::regime_hmm::{Observation as RegimeObservation, RegimeHMM};

// Import lead-lag infrastructure
use hyperliquid_rust_sdk::market_maker::{
    resolve_binance_symbol, BinanceFeed, BinanceFlowConfig, BinancePriceUpdate, BinanceTradeUpdate,
    BinanceUpdate, FlowFeatureVec, LagAnalyzerConfig, SignalIntegrator, SignalIntegratorConfig,
};

// Import logging infrastructure
use hyperliquid_rust_sdk::{init_logging, LogConfig, LogFormat};

// Global log guards to keep logging alive
static LOG_GUARDS: std::sync::OnceLock<Vec<WorkerGuard>> = std::sync::OnceLock::new();

// ============================================================================
// CLI Arguments
// ============================================================================

#[derive(Parser)]
#[command(name = "prediction_validator")]
#[command(version, about = "Validate model predictive power without placing orders", long_about = None)]
struct Cli {
    /// Asset to analyze (e.g., BTC, ETH, SOL)
    #[arg(short, long)]
    asset: String,

    /// Duration to run (e.g., "1h", "4h", "24h")
    #[arg(long, default_value = "24h")]
    duration: String,

    /// Report interval in seconds
    #[arg(long, default_value = "30")]
    report_interval: u64,

    /// Markout window in milliseconds (default: 1000 = 1s)
    #[arg(long, default_value = "1000")]
    markout: u64,

    /// Network: mainnet, testnet
    #[arg(long, default_value = "mainnet")]
    network: String,

    /// HIP-3 DEX name (optional)
    #[arg(long)]
    dex: Option<String>,

    /// Log level: trace, debug, info, warn, error
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Show per-regime metrics breakdown
    #[arg(long)]
    regime_breakdown: bool,

    /// Verbose output (print every message)
    #[arg(long)]
    verbose: bool,

    /// Threshold in bps for adverse movement detection (default: 8.0)
    /// NOTE: 2 bps is below noise floor; 8 bps is more realistic for BTC
    #[arg(long, default_value = "8.0")]
    adverse_threshold_bps: f64,

    /// Warmup period - number of samples before recording predictions (default: 100)
    /// Models need time to converge from priors before predictions are meaningful
    #[arg(long, default_value = "100")]
    warmup_samples: usize,

    /// Directory for log files
    #[arg(long)]
    log_dir: Option<PathBuf>,

    /// Single log file path (legacy mode)
    #[arg(long)]
    log_file: Option<String>,

    /// Enable multi-stream logging (operational/diagnostic/errors)
    #[arg(long, default_value = "true")]
    multi_stream: bool,

    /// Log format: json or pretty
    #[arg(long, default_value = "pretty")]
    log_format: String,

    // === Cross-Exchange Lead-Lag (Binance Feed) ===
    /// Disable Binance price feed for cross-exchange lead-lag validation.
    /// The feed is enabled by default to validate lead-lag signal quality.
    #[arg(long)]
    disable_binance_feed: bool,

    /// Binance symbol to subscribe to for lead-lag validation.
    /// If not specified, auto-derived from asset (e.g., HYPE -> hypeusdt, BTC -> btcusdt).
    #[arg(long)]
    binance_symbol: Option<String>,
}

// ============================================================================
// Prediction Types for Validation
// ============================================================================

/// Extended prediction types for this validator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValidatorPredictionType {
    /// P(informed trade) from InformedFlowEstimator
    InformedFlow,
    /// P(toxic fill) from PreFillASClassifier
    PreFillToxicity,
    /// P(toxic fill) from EnhancedASClassifier (microstructure features)
    EnhancedToxicity,
    /// P(high volatility) from RegimeHMM
    RegimeHighVol,
    /// P(price continues direction) - momentum signal
    Momentum,
    /// P(buy pressure) - next trade direction
    BuyPressure,
    /// P(HL price moves in direction of Binance move) - cross-exchange lead-lag
    LeadLag,
    // === Cross-Venue Features ===
    /// P(price moves in direction of cross-venue agreement) - when both venues agree
    CrossVenueAgreement,
    /// P(adverse fill) when cross-venue toxicity is high
    CrossVenueToxicity,
    /// P(price moves in direction predicted by cross-venue direction signal)
    CrossVenueDirection,
}

impl ValidatorPredictionType {
    pub fn all() -> &'static [ValidatorPredictionType] {
        &[
            ValidatorPredictionType::InformedFlow,
            ValidatorPredictionType::PreFillToxicity,
            ValidatorPredictionType::EnhancedToxicity,
            ValidatorPredictionType::RegimeHighVol,
            ValidatorPredictionType::Momentum,
            ValidatorPredictionType::BuyPressure,
            ValidatorPredictionType::LeadLag,
            // Cross-venue features
            ValidatorPredictionType::CrossVenueAgreement,
            ValidatorPredictionType::CrossVenueToxicity,
            ValidatorPredictionType::CrossVenueDirection,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            ValidatorPredictionType::InformedFlow => "InformedFlow",
            ValidatorPredictionType::PreFillToxicity => "PreFillToxicity",
            ValidatorPredictionType::EnhancedToxicity => "EnhancedTox",
            ValidatorPredictionType::RegimeHighVol => "RegimeHighVol",
            ValidatorPredictionType::Momentum => "Momentum",
            ValidatorPredictionType::BuyPressure => "BuyPressure",
            ValidatorPredictionType::LeadLag => "LeadLag",
            ValidatorPredictionType::CrossVenueAgreement => "CV_Agreement",
            ValidatorPredictionType::CrossVenueToxicity => "CV_Toxicity",
            ValidatorPredictionType::CrossVenueDirection => "CV_Direction",
        }
    }

    pub fn markout_ms(&self) -> u64 {
        match self {
            ValidatorPredictionType::InformedFlow => 1000,    // 1s
            ValidatorPredictionType::PreFillToxicity => 1000, // 1s
            ValidatorPredictionType::EnhancedToxicity => 1000, // 1s
            ValidatorPredictionType::RegimeHighVol => 30000,  // 30s
            ValidatorPredictionType::Momentum => 5000,        // 5s
            ValidatorPredictionType::BuyPressure => 0,        // Next trade (special case)
            ValidatorPredictionType::LeadLag => 500,          // 500ms - typical lead-lag window
            // Cross-venue: 1s markout to measure prediction quality
            ValidatorPredictionType::CrossVenueAgreement => 1000,
            ValidatorPredictionType::CrossVenueToxicity => 1000,
            ValidatorPredictionType::CrossVenueDirection => 1000,
        }
    }
}

// ============================================================================
// Synthetic Fill Detection
// ============================================================================

/// Represents a "synthetic" fill - when a trade would have filled our theoretical quote
#[derive(Debug, Clone)]
pub struct SyntheticFill {
    pub timestamp_ms: u64,
    pub side: Side,
    pub price: f64,
    pub trade_size: f64,
    pub spread_bps: f64,
    pub regime: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Bid, // We would buy
    Ask, // We would sell
}

// ============================================================================
// Pending Outcome Tracking
// ============================================================================

/// A prediction awaiting outcome resolution
#[derive(Debug, Clone)]
pub struct PendingOutcome {
    pub prediction_id: u64,
    pub prediction_type: ValidatorPredictionType,
    pub predicted_prob: f64,
    pub fill_side: Option<Side>,
    /// Reference price - should be MID PRICE at prediction time, not trade price
    pub reference_price: f64,
    pub timestamp_ms: u64,
    pub markout_ms: u64,
    pub regime: usize,
}

/// Input for recording a new prediction outcome.
struct PendingOutcomeInput {
    prediction_type: ValidatorPredictionType,
    predicted_prob: f64,
    fill_side: Option<Side>,
    reference_price: f64,
    timestamp_ms: u64,
    markout_ms: u64,
    regime: usize,
}

// ============================================================================
// Regime Statistics
// ============================================================================

/// Calibration metrics by regime
#[derive(Debug, Clone)]
pub struct RegimeMetrics {
    pub brier: BrierScoreTracker,
    pub ir: InformationRatioTracker,
    pub n_samples: usize,
}

impl Default for RegimeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl RegimeMetrics {
    pub fn new() -> Self {
        Self {
            brier: BrierScoreTracker::new(10000),
            ir: InformationRatioTracker::new(10),
            n_samples: 0,
        }
    }

    pub fn update(&mut self, predicted: f64, outcome: bool) {
        self.brier.update(predicted, outcome);
        self.ir.update(predicted, outcome);
        self.n_samples += 1;
    }
}

// ============================================================================
// Model Tracker
// ============================================================================

/// Tracks calibration metrics for a single prediction type
#[derive(Debug)]
pub struct ModelTracker {
    pub name: ValidatorPredictionType,
    pub brier: BrierScoreTracker,
    pub ir: InformationRatioTracker,
    pub by_regime: [RegimeMetrics; 4], // Calm, Normal, Volatile, Cascade
}

impl ModelTracker {
    pub fn new(name: ValidatorPredictionType) -> Self {
        Self {
            name,
            brier: BrierScoreTracker::new(10000),
            ir: InformationRatioTracker::new(10),
            by_regime: [
                RegimeMetrics::new(),
                RegimeMetrics::new(),
                RegimeMetrics::new(),
                RegimeMetrics::new(),
            ],
        }
    }

    pub fn update(&mut self, predicted: f64, outcome: bool, regime: usize) {
        self.brier.update(predicted, outcome);
        self.ir.update(predicted, outcome);

        let regime_idx = regime.min(3);
        self.by_regime[regime_idx].update(predicted, outcome);
    }

    pub fn n_samples(&self) -> usize {
        self.brier.n_samples()
    }

    pub fn bias(&self) -> f64 {
        // Bias = average predicted - base rate
        // Approximate: if we predicted perfectly calibrated, avg_predicted ≈ base_rate
        // Positive bias means we over-predict, negative means under-predict
        // This is a simplification - proper bias needs prediction mean tracking
        let _base_rate = self.ir.base_rate();
        0.0 // For now, return 0; can be enhanced later
    }

    pub fn health_status(&self) -> &'static str {
        let ir = self.ir.information_ratio();
        let n = self.n_samples();

        if n < 50 {
            "WARMING"
        } else if ir > 1.0 {
            "★ STRONG"
        } else if ir > 0.7 {
            "✓ GOOD"
        } else if ir > 0.5 {
            "~ OK"
        } else if ir > 0.3 {
            "✗ MARGINAL"
        } else {
            "✗ REMOVE"
        }
    }
}

// ============================================================================
// Observation Buffers
// ============================================================================

#[derive(Debug, Clone)]
pub struct TradeObservation {
    pub timestamp_ms: u64,
    pub price: f64,
    pub size: f64,
    pub is_buy: bool,
}

#[derive(Debug, Clone)]
pub struct BookSnapshot {
    pub timestamp_ms: u64,
    pub mid: f64,
    pub best_bid: f64,
    pub best_ask: f64,
    pub bid_depth: f64,
    pub ask_depth: f64,
}

#[derive(Debug, Default)]
pub struct ObservationBuffers {
    pub trades: VecDeque<TradeObservation>,
    pub books: VecDeque<BookSnapshot>,
    pub mids: VecDeque<(u64, f64)>,
    pub max_buffer_size: usize,
}

impl ObservationBuffers {
    pub fn new(max_size: usize) -> Self {
        Self {
            trades: VecDeque::with_capacity(max_size),
            books: VecDeque::with_capacity(max_size),
            mids: VecDeque::with_capacity(max_size),
            max_buffer_size: max_size,
        }
    }

    pub fn add_trade(&mut self, obs: TradeObservation) {
        if self.trades.len() >= self.max_buffer_size {
            self.trades.pop_front();
        }
        self.trades.push_back(obs);
    }

    pub fn add_book(&mut self, snapshot: BookSnapshot) {
        if self.books.len() >= self.max_buffer_size {
            self.books.pop_front();
        }
        self.books.push_back(snapshot);
    }

    pub fn add_mid(&mut self, timestamp_ms: u64, mid: f64) {
        if self.mids.len() >= self.max_buffer_size {
            self.mids.pop_front();
        }
        self.mids.push_back((timestamp_ms, mid));
    }

    /// Calculate book imbalance from last snapshot
    pub fn book_imbalance(&self) -> f64 {
        if let Some(book) = self.books.back() {
            let total = book.bid_depth + book.ask_depth;
            if total > 0.0 {
                (book.bid_depth - book.ask_depth) / total
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Calculate flow imbalance from recent trades
    pub fn flow_imbalance(&self) -> f64 {
        if self.trades.is_empty() {
            return 0.0;
        }

        let (buy_vol, sell_vol) = self.trades.iter().fold((0.0, 0.0), |(b, s), t| {
            if t.is_buy {
                (b + t.size, s)
            } else {
                (b, s + t.size)
            }
        });

        let total = buy_vol + sell_vol;
        if total > 0.0 {
            (buy_vol - sell_vol) / total
        } else {
            0.0
        }
    }

    /// Get recent price momentum (signed)
    pub fn momentum(&self, window: usize) -> f64 {
        if self.mids.len() < 2 {
            return 0.0;
        }

        let recent: Vec<_> = self.mids.iter().rev().take(window.max(2)).collect();
        if recent.len() < 2 {
            return 0.0;
        }

        let latest = recent[0].1;
        let oldest = recent[recent.len() - 1].1;

        if oldest > 0.0 {
            (latest - oldest) / oldest * 10_000.0 // Return in bps
        } else {
            0.0
        }
    }

    /// Get buy ratio from recent trades
    pub fn buy_ratio(&self, window: usize) -> f64 {
        if self.trades.is_empty() {
            return 0.5;
        }

        let recent: Vec<_> = self.trades.iter().rev().take(window).collect();
        let buys = recent.iter().filter(|t| t.is_buy).count();

        buys as f64 / recent.len() as f64
    }

    /// Get volume-weighted buy ratio from recent trades.
    ///
    /// This is more informative than simple buy count because large
    /// trades carry more information than small ones.
    pub fn buy_ratio_volume_weighted(&self, window: usize) -> f64 {
        if self.trades.is_empty() {
            return 0.5;
        }

        let recent: Vec<_> = self.trades.iter().rev().take(window).collect();

        let mut buy_volume = 0.0;
        let mut total_volume = 0.0;

        for trade in &recent {
            total_volume += trade.size;
            if trade.is_buy {
                buy_volume += trade.size;
            }
        }

        if total_volume > 1e-12 {
            buy_volume / total_volume
        } else {
            0.5
        }
    }

    /// Get buy pressure with clustering detection.
    ///
    /// Returns (buy_ratio, clustering_boost) where clustering_boost [0, 1]
    /// indicates how consecutive the buy/sell direction has been.
    pub fn buy_pressure_with_clustering(&self, window: usize) -> (f64, f64) {
        if self.trades.is_empty() {
            return (0.5, 0.0);
        }

        let recent: Vec<_> = self.trades.iter().rev().take(window).collect();

        // Volume-weighted ratio
        let vw_ratio = self.buy_ratio_volume_weighted(window);

        // Clustering: count consecutive same-direction trades
        let mut max_streak = 0;
        let mut current_streak = 0;
        let mut last_is_buy = None;

        for trade in recent.iter().rev() {
            if let Some(last) = last_is_buy {
                if trade.is_buy == last {
                    current_streak += 1;
                    max_streak = max_streak.max(current_streak);
                } else {
                    current_streak = 1;
                }
            } else {
                current_streak = 1;
            }
            last_is_buy = Some(trade.is_buy);
        }

        // Normalize clustering boost (streak of 5+ is considered strong)
        let clustering_boost = (max_streak as f64 / 5.0).min(1.0);

        (vw_ratio, clustering_boost)
    }

    /// Compute HL flow features for cross-venue analysis.
    ///
    /// This creates a FlowFeatureVec from HL trade data that can be
    /// compared with Binance flow features for cross-venue signals.
    pub fn compute_hl_flow_features(&self, timestamp_ms: u64) -> FlowFeatureVec {
        if self.trades.is_empty() {
            return FlowFeatureVec::default();
        }

        // Compute time-windowed imbalances
        let imbalance_1s = self.imbalance_in_window(timestamp_ms, 1000);
        let imbalance_5s = self.imbalance_in_window(timestamp_ms, 5000);
        let imbalance_30s = self.imbalance_in_window(timestamp_ms, 30000);
        let imbalance_5m = self.imbalance_in_window(timestamp_ms, 300_000);

        // Compute trade intensity (trades per second, last 5s)
        let intensity = self.intensity_in_window(timestamp_ms, 5000);

        // Compute size metrics
        let (avg_buy_size, avg_sell_size) = self.avg_trade_sizes(20);
        let size_ratio = if avg_sell_size > 1e-12 {
            avg_buy_size / avg_sell_size
        } else {
            1.0
        };

        // Confidence based on data sufficiency
        let confidence = (self.trades.len().min(100) as f64 / 100.0).min(1.0);

        FlowFeatureVec {
            vpin: 0.0, // HL doesn't have VPIN computed here
            vpin_velocity: 0.0,
            imbalance_1s,
            imbalance_5s,
            imbalance_30s,
            imbalance_5m,
            intensity,
            avg_buy_size,
            avg_sell_size,
            size_ratio,
            order_flow_direction: imbalance_5s, // Use 5s imbalance as direction
            timestamp_ms: timestamp_ms as i64,
            trade_count: self.trades.len() as u64,
            confidence,
        }
    }

    /// Compute imbalance in a specific time window.
    fn imbalance_in_window(&self, now_ms: u64, window_ms: u64) -> f64 {
        let cutoff = now_ms.saturating_sub(window_ms);
        let mut buy_vol = 0.0;
        let mut sell_vol = 0.0;

        for trade in self.trades.iter().rev() {
            if trade.timestamp_ms < cutoff {
                break;
            }
            if trade.is_buy {
                buy_vol += trade.size;
            } else {
                sell_vol += trade.size;
            }
        }

        let total = buy_vol + sell_vol;
        if total > 1e-12 {
            (buy_vol - sell_vol) / total
        } else {
            0.0
        }
    }

    /// Compute trade intensity (trades per second) in a window.
    fn intensity_in_window(&self, now_ms: u64, window_ms: u64) -> f64 {
        let cutoff = now_ms.saturating_sub(window_ms);
        let count = self
            .trades
            .iter()
            .rev()
            .take_while(|t| t.timestamp_ms >= cutoff)
            .count();

        count as f64 / (window_ms as f64 / 1000.0)
    }

    /// Compute average buy and sell trade sizes.
    fn avg_trade_sizes(&self, window: usize) -> (f64, f64) {
        let mut buy_sum = 0.0;
        let mut buy_count = 0;
        let mut sell_sum = 0.0;
        let mut sell_count = 0;

        for trade in self.trades.iter().rev().take(window) {
            if trade.is_buy {
                buy_sum += trade.size;
                buy_count += 1;
            } else {
                sell_sum += trade.size;
                sell_count += 1;
            }
        }

        let avg_buy = if buy_count > 0 {
            buy_sum / buy_count as f64
        } else {
            0.0
        };
        let avg_sell = if sell_count > 0 {
            sell_sum / sell_count as f64
        } else {
            0.0
        };

        (avg_buy, avg_sell)
    }
}

// ============================================================================
// Statistics
// ============================================================================

#[derive(Debug, Default)]
pub struct ValidatorStats {
    pub trades_processed: u64,
    pub books_processed: u64,
    pub mids_processed: u64,
    pub synthetic_fills: u64,
    pub predictions_made: u64,
    pub predictions_resolved: u64,
    pub start_time: Option<Instant>,
    pub last_report_time: Option<Instant>,
    pub last_mid: Option<f64>,
    pub last_spread_bps: Option<f64>,
}

impl ValidatorStats {
    pub fn new() -> Self {
        Self {
            start_time: Some(Instant::now()),
            last_report_time: Some(Instant::now()),
            ..Default::default()
        }
    }

    pub fn uptime_secs(&self) -> f64 {
        self.start_time
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0)
    }

    pub fn uptime_formatted(&self) -> String {
        let secs = self.uptime_secs() as u64;
        let hours = secs / 3600;
        let mins = (secs % 3600) / 60;
        let secs = secs % 60;
        format!("{hours:02}:{mins:02}:{secs:02}")
    }
}

// ============================================================================
// Main Validator
// ============================================================================

pub struct PredictionValidator {
    // Configuration
    pub asset: String,
    pub markout_ms: u64,
    pub adverse_threshold_bps: f64,
    pub show_regime_breakdown: bool,
    pub report_interval_secs: u64,
    pub warmup_samples: usize,

    // Observation buffers
    pub buffers: ObservationBuffers,
    pub stats: ValidatorStats,

    // Estimators
    pub volatility_filter: VolatilityFilter,
    pub flow_estimator: InformedFlowEstimator,
    pub regime_hmm: RegimeHMM,
    pub pre_fill_classifier: PreFillASClassifier,
    pub enhanced_classifier: EnhancedASClassifier,

    // Lead-lag signal integrator (optional - only when Binance feed enabled)
    pub signal_integrator: Option<SignalIntegrator>,
    /// Whether Binance feed is active
    pub binance_enabled: bool,

    // State
    pub last_price: Option<f64>,
    pub last_trade_time_ms: u64,
    /// Tracks total observations for warmup detection
    pub total_observations: usize,

    // Pending predictions awaiting outcome
    pub pending_outcomes: VecDeque<PendingOutcome>,
    pub next_prediction_id: u64,

    // Model trackers
    pub trackers: HashMap<ValidatorPredictionType, ModelTracker>,
}

impl PredictionValidator {
    pub fn new(
        asset: String,
        markout_ms: u64,
        adverse_threshold_bps: f64,
        show_regime_breakdown: bool,
        report_interval_secs: u64,
        warmup_samples: usize,
    ) -> Self {
        // Initialize trackers for each prediction type
        let mut trackers = HashMap::new();
        for &pred_type in ValidatorPredictionType::all() {
            trackers.insert(pred_type, ModelTracker::new(pred_type));
        }

        Self {
            asset,
            markout_ms,
            adverse_threshold_bps,
            show_regime_breakdown,
            report_interval_secs,
            warmup_samples,
            buffers: ObservationBuffers::new(1000),
            stats: ValidatorStats::new(),
            volatility_filter: VolatilityFilter::new(VolFilterConfig::default()),
            flow_estimator: InformedFlowEstimator::default_config(),
            regime_hmm: RegimeHMM::default(),
            pre_fill_classifier: PreFillASClassifier::default(),
            enhanced_classifier: EnhancedASClassifier::default_config(),
            signal_integrator: None,
            binance_enabled: false,
            last_price: None,
            last_trade_time_ms: 0,
            total_observations: 0,
            pending_outcomes: VecDeque::with_capacity(10000),
            next_prediction_id: 0,
            trackers,
        }
    }

    /// Check if models have warmed up enough to record predictions
    pub fn is_warmed_up(&self) -> bool {
        self.total_observations >= self.warmup_samples
    }

    /// Enable Binance lead-lag signal integration
    ///
    /// Uses ADAPTIVE VPIN bucket sizing per Easley, Lopez de Prado, O'Hara:
    /// - Bucket volume dynamically calibrates to maintain ~30s bucket duration
    /// - Self-adjusts to market conditions without hardcoded per-asset constants
    /// - Initial bucket sizes are just starting points, will adapt automatically
    pub fn enable_binance_feed(&mut self, asset: &str) {
        // Use a larger buffer for Binance since bookTicker updates are very fast (~10ms)
        // Default 2000 only holds ~1.5s of Binance data; need 60000 for ~60s
        let mut config = SignalIntegratorConfig {
            lag_config: LagAnalyzerConfig {
                buffer_capacity: 60000, // ~60 seconds at 10ms intervals
                ..LagAnalyzerConfig::default()
            },
            ..SignalIntegratorConfig::default()
        };

        // ADAPTIVE VPIN bucket sizing
        // Instead of static asset-specific buckets, we use adaptive sizing that
        // calibrates to maintain approximately equal-time buckets (~30s target).
        //
        // KEY: Buckets must be large enough to contain multiple trades for meaningful
        // imbalance. Adaptive sizing will adjust based on actual volume rate.
        let initial_bucket_volume = match asset.to_uppercase().as_str() {
            "BTC" => 1.0, // BTC: ~$100k/bucket, ~20-60 trades per bucket
            "ETH" => 5.0, // ETH: similar trade density per bucket
            "SOL" | "AVAX" | "MATIC" | "ARB" | "OP" => 2.0, // Mid-cap
            _ => 0.5,     // Small-cap - adaptive will adjust
        };

        // Use adaptive VPIN configuration - the principled approach
        config.binance_flow_config = BinanceFlowConfig {
            vpin_bucket_volume: initial_bucket_volume,
            vpin_n_buckets: 50,  // More buckets for better VPIN stability
            vpin_adaptive: true, // ENABLED: Self-calibrating bucket sizing
            vpin_target_bucket_seconds: 30.0, // Target ~30s per bucket
            ..BinanceFlowConfig::default()
        };

        self.signal_integrator = Some(SignalIntegrator::new(config));
        self.binance_enabled = true;
        info!(
            asset = %asset,
            initial_bucket_volume = %initial_bucket_volume,
            target_bucket_seconds = 30.0,
            "Lead-lag signal integrator enabled with ADAPTIVE VPIN bucket sizing"
        );
    }

    /// Process a Binance price update
    pub fn on_binance_price(&mut self, update: &BinancePriceUpdate) {
        // Extract signal info first to avoid borrow issues
        let signal_info = if let Some(ref mut integrator) = self.signal_integrator {
            integrator.on_binance_price(update.mid_price, update.timestamp_ms);
            let signals = integrator.get_signals();
            if signals.lead_lag_actionable {
                Some(signals.combined_skew_bps)
            } else {
                None
            }
        } else {
            None
        };

        // Record prediction outside the borrow
        if let Some(skew_bps) = signal_info {
            if self.is_warmed_up() {
                let timestamp_u64 = update.timestamp_ms.max(0) as u64;
                self.record_lead_lag_prediction(skew_bps, timestamp_u64);
            }
        }
    }

    /// Process a Binance trade update for cross-venue analysis
    pub fn on_binance_trade(&mut self, update: &BinanceTradeUpdate) {
        if let Some(ref mut integrator) = self.signal_integrator {
            integrator.on_binance_trade(update);
        }
    }

    /// Record a lead-lag prediction when Binance signal is actionable
    fn record_lead_lag_prediction(&mut self, skew_bps: f64, timestamp_ms: u64) {
        // Only record if we have a reference mid price
        let mid = match self.stats.last_mid {
            Some(m) => m,
            None => return,
        };

        let regime = self.regime_hmm.most_likely_regime();

        // Convert skew to probability and direction
        // Positive skew = expect HL to go up, negative = expect down
        // Map skew magnitude to confidence (sigmoid)
        let p_direction = 1.0 / (1.0 + (-skew_bps.abs() / 5.0).exp());

        // Direction: Ask = up (positive skew), Bid = down (negative skew)
        let predicted_direction = if skew_bps >= 0.0 {
            Some(Side::Ask)
        } else {
            Some(Side::Bid)
        };

        self.record_pending_outcome(PendingOutcomeInput {
            prediction_type: ValidatorPredictionType::LeadLag,
            predicted_prob: p_direction,
            fill_side: predicted_direction,
            reference_price: mid,
            timestamp_ms,
            markout_ms: ValidatorPredictionType::LeadLag.markout_ms(),
            regime,
        });

        self.stats.predictions_made += 1;
    }

    /// Process a trade message
    pub fn on_trade(&mut self, timestamp_ms: u64, price: f64, size: f64, is_buy: bool) {
        let obs = TradeObservation {
            timestamp_ms,
            price,
            size,
            is_buy,
        };
        self.buffers.add_trade(obs);
        self.stats.trades_processed += 1;
        self.total_observations += 1;

        // Update volatility filter
        if let Some(last_price) = self.last_price {
            if last_price > 0.0 {
                let ret = (price / last_price).ln();
                let dt = if self.last_trade_time_ms > 0 {
                    (timestamp_ms.saturating_sub(self.last_trade_time_ms) as f64 / 1000.0)
                        .max(0.001)
                } else {
                    1.0
                };
                self.volatility_filter.on_return(ret, dt);
            }
        }
        self.last_price = Some(price);

        // Calculate inter-arrival time
        let inter_arrival_ms = if self.last_trade_time_ms > 0 {
            timestamp_ms.saturating_sub(self.last_trade_time_ms)
        } else {
            1000
        };
        self.last_trade_time_ms = timestamp_ms;

        // Compute realized price impact from recent trades for the flow estimator
        let price_impact_bps = self.compute_realized_impact_bps();

        // Update flow estimator with computed impact
        let features = TradeFeatures {
            size,
            inter_arrival_ms,
            price_impact_bps,
            book_imbalance: self.buffers.book_imbalance(),
            is_buy,
            timestamp_ms,
        };
        self.flow_estimator.on_trade(&features);

        // Update pre-fill classifier with trade flow
        let buy_ratio = self.buffers.buy_ratio(20);
        self.pre_fill_classifier
            .update_trade_flow(buy_ratio, 1.0 - buy_ratio);

        // Update enhanced microstructure classifier
        self.enhanced_classifier.on_trade(MicroTradeObs {
            timestamp_ms,
            price,
            size,
            is_buy,
        });

        // Feed HL flow features to SignalIntegrator for cross-venue analysis
        // This is critical - without this, CV_Agreement is always 0 because HL imbalances are empty
        if let Some(ref mut integrator) = self.signal_integrator {
            let hl_flow = self.buffers.compute_hl_flow_features(timestamp_ms);
            integrator.set_hl_flow_features(hl_flow);
            // Feed trades to SignalIntegrator's BuyPressure tracker
            integrator.on_trade_for_pressure(size, is_buy);
        }

        // Resolve any BuyPressure predictions (next trade resolution)
        self.resolve_buy_pressure_predictions(is_buy, timestamp_ms);

        // Only record predictions after warmup period
        // This gives models time to converge from priors
        if self.is_warmed_up() {
            // Check for synthetic fill opportunity
            if let Some(fill) = self.detect_fill_opportunity(timestamp_ms, price, size, is_buy) {
                self.stats.synthetic_fills += 1;
                self.record_predictions(&fill, timestamp_ms);
            }
        }

        // Resolve pending outcomes using MID price, not trade price
        // This avoids bid-ask bounce bias
        if let Some(mid) = self.stats.last_mid {
            self.resolve_pending_outcomes(mid, timestamp_ms);
        }
    }

    /// Compute realized price impact from recent trade history
    /// Returns the average absolute price impact in bps of recent trades
    fn compute_realized_impact_bps(&self) -> f64 {
        if self.buffers.trades.len() < 2 {
            return 0.0;
        }

        // Look at last 10 trades and compute average absolute impact
        let recent: Vec<_> = self.buffers.trades.iter().rev().take(11).collect();
        if recent.len() < 2 {
            return 0.0;
        }

        let mut total_impact = 0.0;
        let mut count = 0;

        for window in recent.windows(2) {
            let curr = window[0];
            let prev = window[1];
            if prev.price > 0.0 {
                let impact_bps = ((curr.price - prev.price) / prev.price * 10_000.0).abs();
                total_impact += impact_bps;
                count += 1;
            }
        }

        if count > 0 {
            total_impact / count as f64
        } else {
            0.0
        }
    }

    /// Process a book update
    pub fn on_book(
        &mut self,
        timestamp_ms: u64,
        mid: f64,
        best_bid: f64,
        best_ask: f64,
        bid_depth: f64,
        ask_depth: f64,
    ) {
        let spread_bps = if mid > 0.0 {
            (best_ask - best_bid) / mid * 10_000.0
        } else {
            0.0
        };

        let snapshot = BookSnapshot {
            timestamp_ms,
            mid,
            best_bid,
            best_ask,
            bid_depth,
            ask_depth,
        };
        self.buffers.add_book(snapshot);
        self.stats.books_processed += 1;
        self.stats.last_spread_bps = Some(spread_bps);

        // Update pre-fill classifier
        self.pre_fill_classifier
            .update_orderbook(bid_depth, ask_depth);

        // Update enhanced microstructure classifier
        self.enhanced_classifier.on_book_update(
            best_bid,
            best_ask,
            bid_depth,
            ask_depth,
            timestamp_ms,
        );

        // Update regime HMM
        // CRITICAL: RegimeObservation expects volatility in FRACTIONAL units (e.g., 0.0031 for 31 bps)
        // sigma_bps_per_sqrt_s() returns bps (e.g., 31), so we must convert to fractional
        let sigma_bps = self.volatility_filter.sigma_bps_per_sqrt_s();
        let sigma_fractional = sigma_bps / 10_000.0; // Convert bps to fractional
        let flow_imbalance = self.buffers.flow_imbalance();
        let regime_obs = RegimeObservation::new(sigma_fractional, spread_bps, flow_imbalance);
        self.regime_hmm.forward_update(&regime_obs);

        // Feed HL mid to SignalIntegrator from L2Book (more frequent than AllMids)
        if let Some(ref mut integrator) = self.signal_integrator {
            integrator.on_hl_price(mid, timestamp_ms as i64);
        }
    }

    /// Process a mid price update
    pub fn on_mid(&mut self, timestamp_ms: u64, mid: f64) {
        self.buffers.add_mid(timestamp_ms, mid);
        self.stats.mids_processed += 1;
        self.stats.last_mid = Some(mid);

        // Feed Hyperliquid price to SignalIntegrator for lead-lag calculation
        if let Some(ref mut integrator) = self.signal_integrator {
            integrator.on_hl_price(mid, timestamp_ms as i64);
        }
    }

    /// Detect synthetic fill opportunity
    fn detect_fill_opportunity(
        &self,
        timestamp_ms: u64,
        trade_price: f64,
        trade_size: f64,
        trade_is_buy: bool,
    ) -> Option<SyntheticFill> {
        let book = self.buffers.books.back()?;
        let spread_bps = self.stats.last_spread_bps?;

        // Compute theoretical quotes based on current mid and typical spread
        let theoretical_half_spread_bps = spread_bps / 2.0 + 1.5; // +1.5 bps for maker rebate
        let mid = book.mid;
        let theoretical_bid = mid * (1.0 - theoretical_half_spread_bps / 10_000.0);
        let theoretical_ask = mid * (1.0 + theoretical_half_spread_bps / 10_000.0);

        // Determine regime
        let regime = self.regime_hmm.most_likely_regime();

        // Trade executed at or through our theoretical bid (we would buy)
        if trade_price <= theoretical_bid && !trade_is_buy {
            // Aggressive sell hit our bid
            return Some(SyntheticFill {
                timestamp_ms,
                side: Side::Bid,
                price: theoretical_bid,
                trade_size,
                spread_bps,
                regime,
            });
        }

        // Trade executed at or through our theoretical ask (we would sell)
        if trade_price >= theoretical_ask && trade_is_buy {
            // Aggressive buy lifted our ask
            return Some(SyntheticFill {
                timestamp_ms,
                side: Side::Ask,
                price: theoretical_ask,
                trade_size,
                spread_bps,
                regime,
            });
        }

        None
    }

    /// Record predictions for a synthetic fill
    fn record_predictions(&mut self, fill: &SyntheticFill, timestamp_ms: u64) {
        let regime = fill.regime;
        let mid = self.stats.last_mid.unwrap_or(fill.price);

        // 1. InformedFlow prediction
        let p_informed = self.flow_estimator.decomposition().p_informed;
        self.record_pending_outcome(PendingOutcomeInput {
            prediction_type: ValidatorPredictionType::InformedFlow,
            predicted_prob: p_informed,
            fill_side: Some(fill.side),
            reference_price: mid,
            timestamp_ms,
            markout_ms: self
                .markout_ms
                .max(ValidatorPredictionType::InformedFlow.markout_ms()),
            regime,
        });

        // 2. PreFillToxicity prediction (old classifier)
        // Pipe blended toxicity from IntegratedSignals (includes VPIN when valid)
        if let Some(ref integrator) = self.signal_integrator {
            let signals = integrator.get_signals();
            self.pre_fill_classifier
                .set_blended_toxicity(signals.toxicity_score);
        }
        let p_toxic = match fill.side {
            Side::Bid => self.pre_fill_classifier.predict_toxicity(true),
            Side::Ask => self.pre_fill_classifier.predict_toxicity(false),
        };
        self.record_pending_outcome(PendingOutcomeInput {
            prediction_type: ValidatorPredictionType::PreFillToxicity,
            predicted_prob: p_toxic,
            fill_side: Some(fill.side),
            reference_price: mid,
            timestamp_ms,
            markout_ms: self
                .markout_ms
                .max(ValidatorPredictionType::PreFillToxicity.markout_ms()),
            regime,
        });

        // 3. EnhancedToxicity prediction (microstructure features)
        let p_enhanced_toxic = match fill.side {
            Side::Bid => self.enhanced_classifier.predict_toxicity(true),
            Side::Ask => self.enhanced_classifier.predict_toxicity(false),
        };
        self.record_pending_outcome(PendingOutcomeInput {
            prediction_type: ValidatorPredictionType::EnhancedToxicity,
            predicted_prob: p_enhanced_toxic,
            fill_side: Some(fill.side),
            reference_price: mid,
            timestamp_ms,
            markout_ms: self
                .markout_ms
                .max(ValidatorPredictionType::EnhancedToxicity.markout_ms()),
            regime,
        });

        // 4. RegimeHighVol prediction
        let regime_probs = self.regime_hmm.regime_probabilities();
        // P(high vol) = P(volatile) + P(cascade) - assuming regime indices 2 and 3 are volatile/extreme
        let p_high_vol = regime_probs.get(2).copied().unwrap_or(0.0)
            + regime_probs.get(3).copied().unwrap_or(0.0);
        self.record_pending_outcome(PendingOutcomeInput {
            prediction_type: ValidatorPredictionType::RegimeHighVol,
            predicted_prob: p_high_vol,
            fill_side: None,
            reference_price: mid,
            timestamp_ms,
            markout_ms: ValidatorPredictionType::RegimeHighVol.markout_ms(),
            regime,
        });

        // 4. Momentum prediction
        let momentum_bps = self.buffers.momentum(10);
        // P(price continues) = sigmoid of momentum
        let p_continues = 1.0 / (1.0 + (-momentum_bps / 5.0).exp()); // Scale by 5 bps
        self.record_pending_outcome(PendingOutcomeInput {
            prediction_type: ValidatorPredictionType::Momentum,
            predicted_prob: p_continues,
            fill_side: if momentum_bps >= 0.0 {
                Some(Side::Ask)
            } else {
                Some(Side::Bid)
            },
            reference_price: mid,
            timestamp_ms,
            markout_ms: ValidatorPredictionType::Momentum.markout_ms(),
            regime,
        });

        // 5. BuyPressure prediction (enhanced with volume-weighting and clustering)
        let (vw_buy_ratio, clustering_boost) = self.buffers.buy_pressure_with_clustering(20);
        // Boost prediction confidence when clustering is detected
        let buy_pressure = vw_buy_ratio + (vw_buy_ratio - 0.5) * clustering_boost * 0.3;
        let buy_pressure = buy_pressure.clamp(0.01, 0.99);
        self.record_pending_outcome(PendingOutcomeInput {
            prediction_type: ValidatorPredictionType::BuyPressure,
            predicted_prob: buy_pressure,
            fill_side: None, // Resolved on next trade
            reference_price: mid,
            timestamp_ms,
            markout_ms: 0, // Special: resolved on next trade
            regime,
        });

        let mut prediction_count = 6;

        // 6-8. Cross-Venue predictions (if enabled and valid)
        if let Some(ref integrator) = self.signal_integrator {
            let signals = integrator.get_signals();

            if signals.cross_venue_valid {
                // CrossVenueAgreement: When venues agree, predict price moves in that direction
                // High agreement (>0.5) = high confidence in direction
                let p_agreement = (signals.cross_venue_agreement.abs() + 1.0) / 2.0; // map [-1,1] to [0,1]
                let agreement_direction = if signals.cross_venue_direction >= 0.0 {
                    Some(Side::Ask) // Bullish
                } else {
                    Some(Side::Bid) // Bearish
                };
                self.record_pending_outcome(PendingOutcomeInput {
                    prediction_type: ValidatorPredictionType::CrossVenueAgreement,
                    predicted_prob: p_agreement * signals.cross_venue_confidence,
                    fill_side: agreement_direction,
                    reference_price: mid,
                    timestamp_ms,
                    markout_ms: ValidatorPredictionType::CrossVenueAgreement.markout_ms(),
                    regime,
                });
                prediction_count += 1;

                // CrossVenueToxicity: When cross-venue toxicity is high, predict adverse fill
                // Use max toxicity as the probability of adverse fill
                self.record_pending_outcome(PendingOutcomeInput {
                    prediction_type: ValidatorPredictionType::CrossVenueToxicity,
                    predicted_prob: signals.cross_venue_max_toxicity,
                    fill_side: Some(fill.side), // Same side as the synthetic fill
                    reference_price: mid,
                    timestamp_ms,
                    markout_ms: ValidatorPredictionType::CrossVenueToxicity.markout_ms(),
                    regime,
                });
                prediction_count += 1;

                // CrossVenueDirection: Predict price moves in direction of cross-venue signal
                // Convert direction [-1,1] to probability and direction
                let p_direction = (signals.cross_venue_direction.abs() + 1.0) / 2.0;
                let predicted_direction = if signals.cross_venue_direction >= 0.0 {
                    Some(Side::Ask) // Bullish
                } else {
                    Some(Side::Bid) // Bearish
                };
                self.record_pending_outcome(PendingOutcomeInput {
                    prediction_type: ValidatorPredictionType::CrossVenueDirection,
                    predicted_prob: p_direction * signals.cross_venue_confidence,
                    fill_side: predicted_direction,
                    reference_price: mid,
                    timestamp_ms,
                    markout_ms: ValidatorPredictionType::CrossVenueDirection.markout_ms(),
                    regime,
                });
                prediction_count += 1;
            }
        }

        self.stats.predictions_made += prediction_count;
    }

    /// Record a pending outcome
    fn record_pending_outcome(&mut self, input: PendingOutcomeInput) {
        let id = self.next_prediction_id;
        self.next_prediction_id += 1;

        self.pending_outcomes.push_back(PendingOutcome {
            prediction_id: id,
            prediction_type: input.prediction_type,
            predicted_prob: input.predicted_prob.clamp(0.0, 1.0),
            fill_side: input.fill_side,
            reference_price: input.reference_price,
            timestamp_ms: input.timestamp_ms,
            markout_ms: input.markout_ms,
            regime: input.regime,
        });
    }

    /// Resolve BuyPressure predictions (special case - next trade)
    fn resolve_buy_pressure_predictions(&mut self, trade_is_buy: bool, now_ms: u64) {
        // Find and resolve BuyPressure predictions
        let mut to_resolve = Vec::new();

        for (idx, pending) in self.pending_outcomes.iter().enumerate() {
            if pending.prediction_type == ValidatorPredictionType::BuyPressure
                && pending.markout_ms == 0
                && now_ms > pending.timestamp_ms
            {
                to_resolve.push((idx, pending.predicted_prob, pending.regime));
            }
        }

        // Process in reverse order to maintain indices
        for (idx, predicted_prob, regime) in to_resolve.into_iter().rev() {
            self.pending_outcomes.remove(idx);

            // Outcome: was next trade a buy?
            let outcome = trade_is_buy;

            if let Some(tracker) = self.trackers.get_mut(&ValidatorPredictionType::BuyPressure) {
                tracker.update(predicted_prob, outcome, regime);
            }
            self.stats.predictions_resolved += 1;
        }
    }

    /// Resolve pending outcomes based on price movement
    /// Uses MID PRICE for outcome resolution to avoid bid-ask bounce bias
    fn resolve_pending_outcomes(&mut self, current_mid: f64, now_ms: u64) {
        // Use regime-dependent threshold: base threshold scaled by volatility
        // In high vol regimes, price needs to move more to be "adverse"
        let vol_scale = (self.volatility_filter.sigma_bps_per_sqrt_s() / 10.0).clamp(0.5, 3.0);
        let threshold_bps = self.adverse_threshold_bps * vol_scale;

        while let Some(pending) = self.pending_outcomes.front() {
            // Skip BuyPressure (handled separately)
            if pending.prediction_type == ValidatorPredictionType::BuyPressure {
                self.pending_outcomes.pop_front();
                continue;
            }

            // Check if markout window has elapsed
            if now_ms < pending.timestamp_ms + pending.markout_ms {
                break; // Pending outcomes are in chronological order
            }

            let pending = self.pending_outcomes.pop_front().unwrap();

            // Calculate price movement using mid price (not trade price)
            // This avoids bid-ask bounce bias in outcome measurement
            let price_move_bps = if pending.reference_price > 0.0 {
                (current_mid - pending.reference_price) / pending.reference_price * 10_000.0
            } else {
                0.0
            };

            // Determine outcome based on prediction type
            let outcome = match pending.prediction_type {
                ValidatorPredictionType::InformedFlow
                | ValidatorPredictionType::PreFillToxicity
                | ValidatorPredictionType::EnhancedToxicity => {
                    // Outcome: did price move against our fill?
                    match pending.fill_side {
                        Some(Side::Bid) => price_move_bps < -threshold_bps, // Bought, price dropped
                        Some(Side::Ask) => price_move_bps > threshold_bps,  // Sold, price rose
                        None => false,
                    }
                }
                ValidatorPredictionType::RegimeHighVol => {
                    // Outcome: is the current HMM regime high-vol (regime 2 or 3)?
                    // This aligns with the prediction which uses HMM regime probabilities.
                    // Using the same source for prediction and outcome ensures consistency.
                    let current_regime = self.regime_hmm.most_likely_regime();
                    current_regime >= 2 // Regimes 2 (volatile) and 3 (cascade) are high-vol
                }
                ValidatorPredictionType::Momentum => {
                    // Outcome: did price continue in predicted direction?
                    match pending.fill_side {
                        Some(Side::Ask) => price_move_bps > 0.0, // Predicted up, went up
                        Some(Side::Bid) => price_move_bps < 0.0, // Predicted down, went down
                        None => false,
                    }
                }
                ValidatorPredictionType::BuyPressure => {
                    // Already handled separately
                    false
                }
                ValidatorPredictionType::LeadLag => {
                    // Outcome: did HL price move in predicted direction (from Binance signal)?
                    // fill_side encodes the predicted direction: Ask = up, Bid = down
                    match pending.fill_side {
                        Some(Side::Ask) => price_move_bps > 0.0, // Predicted up, went up
                        Some(Side::Bid) => price_move_bps < 0.0, // Predicted down, went down
                        None => false,
                    }
                }
                // === Cross-Venue Predictions ===
                ValidatorPredictionType::CrossVenueAgreement => {
                    // Outcome: when venues agreed, did price move in the predicted direction?
                    // fill_side encodes predicted direction: Ask = up, Bid = down
                    match pending.fill_side {
                        Some(Side::Ask) => price_move_bps > 0.0,
                        Some(Side::Bid) => price_move_bps < 0.0,
                        None => false,
                    }
                }
                ValidatorPredictionType::CrossVenueToxicity => {
                    // Outcome: did high cross-venue toxicity predict adverse fill?
                    // Same as other toxicity predictors - price moved against fill side
                    match pending.fill_side {
                        Some(Side::Bid) => price_move_bps < -threshold_bps,
                        Some(Side::Ask) => price_move_bps > threshold_bps,
                        None => false,
                    }
                }
                ValidatorPredictionType::CrossVenueDirection => {
                    // Outcome: did price move in the direction predicted by cross-venue signal?
                    match pending.fill_side {
                        Some(Side::Ask) => price_move_bps > 0.0,
                        Some(Side::Bid) => price_move_bps < 0.0,
                        None => false,
                    }
                }
            };

            // Update tracker
            if let Some(tracker) = self.trackers.get_mut(&pending.prediction_type) {
                tracker.update(pending.predicted_prob, outcome, pending.regime);
            }

            // === Online Learning for PreFillASClassifier ===
            // When we resolve a PreFillToxicity prediction, feed the outcome back
            // to the classifier so it can learn optimal signal weights via SGD.
            if pending.prediction_type == ValidatorPredictionType::PreFillToxicity {
                if let Some(side) = pending.fill_side {
                    let is_bid = matches!(side, Side::Bid);
                    let adverse_magnitude = price_move_bps.abs();
                    self.pre_fill_classifier.record_outcome(
                        is_bid,
                        outcome, // was_adverse
                        Some(adverse_magnitude),
                    );
                }
            }

            // === Online Learning for EnhancedASClassifier ===
            // When we resolve an EnhancedToxicity prediction, feed the outcome back
            // to the classifier so it can learn optimal feature weights via SGD.
            if pending.prediction_type == ValidatorPredictionType::EnhancedToxicity {
                if let Some(side) = pending.fill_side {
                    let is_bid = matches!(side, Side::Bid);
                    let adverse_magnitude = price_move_bps.abs();
                    self.enhanced_classifier.record_outcome(
                        is_bid,
                        outcome, // was_adverse
                        Some(adverse_magnitude),
                    );
                }
            }

            self.stats.predictions_resolved += 1;
        }
    }

    /// Check if it's time to print a report
    pub fn should_report(&mut self) -> bool {
        if let Some(last) = self.stats.last_report_time {
            if last.elapsed().as_secs() >= self.report_interval_secs {
                self.stats.last_report_time = Some(Instant::now());
                return true;
            }
        }
        false
    }

    /// Print the validation report
    pub fn print_report(&self) {
        println!();
        println!("┌─────────────────────────────────────────────────────────────────┐");
        println!(
            "│             PREDICTION VALIDATOR - {:<10}                   │",
            self.asset
        );
        println!(
            "│             Runtime: {} | Samples: {:>7}                   │",
            self.stats.uptime_formatted(),
            self.stats.predictions_resolved
        );
        if !self.is_warmed_up() {
            println!(
                "│  ⏳ WARMUP: {}/{} observations (predictions not recorded)   │",
                self.total_observations, self.warmup_samples
            );
        }
        println!("├─────────────────────────────────────────────────────────────────┤");
        println!("│                                                                 │");
        println!("│  Model                Brier   IR      Bias    N      Health     │");
        println!("│  ─────────────────────────────────────────────────────────────  │");

        for pred_type in ValidatorPredictionType::all() {
            if let Some(tracker) = self.trackers.get(pred_type) {
                let brier = tracker.brier.score();
                let ir = tracker.ir.information_ratio();
                let bias = tracker.bias();
                let n = tracker.n_samples();
                let health = tracker.health_status();

                println!(
                    "│  {:20} {:.3}   {:.2}    {:+.1}%   {:>5}  {:10} │",
                    pred_type.name(),
                    brier,
                    ir,
                    bias * 100.0,
                    n,
                    health
                );
            }
        }

        println!("│                                                                 │");

        if self.show_regime_breakdown {
            self.print_regime_breakdown();
        }

        println!("└─────────────────────────────────────────────────────────────────┘");
        println!();

        // Print interpretation guide
        self.print_interpretation();
    }

    fn print_regime_breakdown(&self) {
        let regime_names = ["Calm", "Normal", "Volatile", "Cascade"];

        println!("├─────────────────────────────────────────────────────────────────┤");
        println!("│  BY REGIME:                                                     │");
        println!("│                                                                 │");

        // Aggregate across all models by regime
        let mut regime_brier = [0.0f64; 4];
        let mut regime_ir = [0.0f64; 4];
        let mut regime_n = [0usize; 4];

        for tracker in self.trackers.values() {
            for (i, regime_metrics) in tracker.by_regime.iter().enumerate() {
                if regime_metrics.n_samples > 0 {
                    regime_brier[i] += regime_metrics.brier.score();
                    regime_ir[i] += regime_metrics.ir.information_ratio();
                    regime_n[i] += regime_metrics.n_samples;
                }
            }
        }

        // Average across models
        let n_models = self.trackers.len() as f64;
        for i in 0..4 {
            if regime_n[i] > 0 {
                regime_brier[i] /= n_models;
                regime_ir[i] /= n_models;
            }
        }

        for i in 0..4 {
            let indicator = if regime_ir[i] > 0.5 { "✓" } else { "✗" };
            let note = match i {
                0 if regime_ir[i] > 0.7 => "← Models work best here",
                3 if regime_ir[i] < 0.3 => "← Expected degradation",
                _ => "",
            };
            println!(
                "│  {:8} (n={:>5})    │ Brier: {:.3}  IR: {:.2}  {} {:20} │",
                regime_names[i], regime_n[i], regime_brier[i], regime_ir[i], indicator, note
            );
        }

        println!("│                                                                 │");
    }

    fn print_interpretation(&self) {
        // Count models by health status
        let mut good = 0;
        let mut _marginal = 0;
        let mut bad = 0;
        let mut warming = 0;

        for tracker in self.trackers.values() {
            match tracker.health_status() {
                "★ STRONG" | "✓ GOOD" | "~ OK" => good += 1,
                "✗ MARGINAL" => _marginal += 1,
                "✗ REMOVE" => bad += 1,
                _ => warming += 1,
            }
        }

        if warming == self.trackers.len() {
            println!("⏳ Collecting predictions... Need more samples for reliable metrics.");
        } else if bad > 2 {
            println!("⚠️  Multiple models showing IR < 0.3. Review model assumptions.");
        } else if good >= 3 {
            println!("✅ Models showing predictive power. Consider live validation.");
        } else {
            println!("📊 Mixed results. Continue monitoring.");
        }

        // Print IR diagnostic information
        println!();
        println!("IR Diagnostics:");
        for pred_type in ValidatorPredictionType::all() {
            if let Some(tracker) = self.trackers.get(pred_type) {
                let n = tracker.n_samples();
                if n >= 50 {
                    let base_rate = tracker.ir.base_rate();
                    let resolution = tracker.ir.resolution();
                    let uncertainty = tracker.ir.uncertainty();
                    let bin_counts = tracker.ir.bin_counts();

                    // Check for bin concentration (>80% in one bin is suspicious)
                    let max_bin = *bin_counts.iter().max().unwrap_or(&0);
                    let concentration = if n > 0 {
                        max_bin as f64 / n as f64
                    } else {
                        0.0
                    };

                    let warning = if n < 500 {
                        " ⚠️ <500 samples"
                    } else if concentration > 0.8 {
                        " ⚠️ concentrated"
                    } else {
                        ""
                    };

                    println!(
                        "  {:16} base={:.3} res={:.4} unc={:.4} conc={:.0}%{}",
                        pred_type.name(),
                        base_rate,
                        resolution,
                        uncertainty,
                        concentration * 100.0,
                        warning
                    );
                }
            }
        }

        // Print PreFillASClassifier learning diagnostics
        let learning_diag = self.pre_fill_classifier.learning_diagnostics();
        let weights = if learning_diag.is_using_learned {
            &learning_diag.learned_weights
        } else {
            &learning_diag.default_weights
        };
        println!();
        println!(
            "PreFill Learning: {}/{} samples | weights: [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}]{}",
            learning_diag.samples,
            learning_diag.min_samples,
            weights[0],
            weights[1],
            weights[2],
            weights[3],
            weights[4],
            if learning_diag.is_using_learned {
                " (LEARNED)"
            } else {
                " (default)"
            }
        );

        // Print EnhancedASClassifier diagnostics
        let enhanced_diag = self.enhanced_classifier.diagnostics();
        println!(
            "Enhanced Learning: {}/{} samples | acc={:.1}% | top features: {}{}",
            enhanced_diag.learning_samples,
            500,
            enhanced_diag.accuracy * 100.0,
            enhanced_diag.feature_summary(),
            if enhanced_diag.is_using_learned {
                " (LEARNED)"
            } else {
                " (default)"
            }
        );

        // Print lead-lag diagnostics if enabled
        if let Some(ref integrator) = self.signal_integrator {
            let signals = integrator.get_signals();
            let status = integrator.lag_analyzer_status();
            let ((sig_first, sig_last), (tgt_first, tgt_last)) = status.sample_timestamps;
            println!();
            println!(
                "LeadLag: ready={} | signal_obs={} target_obs={} | lag={:?}ms mi={:.3?}",
                status.is_ready,
                status.observation_counts.0,
                status.observation_counts.1,
                status.optimal_lag_ms,
                status.mi_bits,
            );
            println!("  timestamps: signal=[{sig_first:?}..{sig_last:?}] target=[{tgt_first:?}..{tgt_last:?}]");
            if let (Some(sf), Some(sl), Some(tf), Some(tl)) =
                (sig_first, sig_last, tgt_first, tgt_last)
            {
                let overlap = sl >= tf && tl >= sf;
                println!(
                    "  overlap={} | signal_range={}ms target_range={}ms gap={}ms",
                    overlap,
                    sl - sf,
                    tl - tf,
                    if tf > sl {
                        tf - sl
                    } else if sf > tl {
                        sf - tl
                    } else {
                        0
                    },
                );
            }
            println!(
                "  last_signal: actionable={} diff={:.1}bps skew_dir={} skew_mag={:.1}bps",
                status.last_signal.is_actionable,
                status.last_signal.diff_bps,
                status.last_signal.skew_direction,
                status.last_signal.skew_magnitude_bps,
            );

            // Print cross-venue diagnostics
            if signals.cross_venue_valid {
                println!(
                    "CrossVenue: valid=true | dir={:.2} conf={:.2} agree={:.2}",
                    signals.cross_venue_direction,
                    signals.cross_venue_confidence,
                    signals.cross_venue_agreement,
                );
                println!(
                    "  toxicity: max={:.2} avg={:.2} | intensity_ratio={:.2} divergence={:.2}",
                    signals.cross_venue_max_toxicity,
                    signals.cross_venue_avg_toxicity,
                    signals.cross_venue_intensity_ratio,
                    signals.cross_venue_divergence,
                );
            } else {
                println!("CrossVenue: valid=false (warming up or no Binance trades)");
            }

            // === New Signal Integration Diagnostics (post-degenerate-fix) ===
            println!();
            println!("Signal Fix Diagnostics:");
            println!(
                "  VPIN: value={:.3} velocity={:.3} (blended into toxicity={:.3})",
                signals.hl_vpin, signals.hl_vpin_velocity, signals.toxicity_score,
            );
            println!(
                "  BuyPressure: z={:.2} | skew_contribution={:.2}bps",
                signals.buy_pressure_z,
                signals.combined_skew_bps
                    - if signals.lead_lag_actionable {
                        signals.lead_lag_skew_bps * signals.skew_direction as f64
                    } else {
                        0.0
                    },
            );
            println!(
                "  Gating: lead_lag_w={:.2} informed_flow_w={:.2} | model_conf={:.2}",
                signals.lead_lag_gating_weight,
                signals.informed_flow_gating_weight,
                signals.model_confidence,
            );
            println!(
                "  LeadLag MI: significant={} null_p95={:.4}",
                signals.lead_lag_significant, signals.lead_lag_null_p95,
            );
        }

        // Print HMM calibration status
        {
            let regime_probs = self.regime_hmm.regime_probabilities();
            println!(
                "RegimeHMM: calibrated={} recalibrations={} | belief=[{:.2},{:.2},{:.2},{:.2}]",
                self.regime_hmm.is_calibrated(),
                self.regime_hmm.recalibration_count(),
                regime_probs[0],
                regime_probs[1],
                regime_probs[2],
                regime_probs[3],
            );
        }

        // Print stats
        println!();
        println!(
            "Stats: {} trades | {} L2 updates | {} synthetic fills | {} pending",
            self.stats.trades_processed,
            self.stats.books_processed,
            self.stats.synthetic_fills,
            self.pending_outcomes.len()
        );
        println!(
            "Config: threshold={:.1}bps (vol-scaled) | warmup={}/{}",
            self.adverse_threshold_bps,
            self.total_observations.min(self.warmup_samples),
            self.warmup_samples
        );
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn parse_duration(s: &str) -> Result<Duration, String> {
    let s = s.trim().to_lowercase();
    if let Some(hours) = s.strip_suffix('h') {
        let h: u64 = hours
            .parse()
            .map_err(|_| format!("Invalid hours: {hours}"))?;
        Ok(Duration::from_secs(h * 3600))
    } else if let Some(mins) = s.strip_suffix('m') {
        let m: u64 = mins
            .parse()
            .map_err(|_| format!("Invalid minutes: {mins}"))?;
        Ok(Duration::from_secs(m * 60))
    } else if let Some(secs) = s.strip_suffix('s') {
        let sec: u64 = secs
            .parse()
            .map_err(|_| format!("Invalid seconds: {secs}"))?;
        Ok(Duration::from_secs(sec))
    } else {
        let sec: u64 = s.parse().map_err(|_| format!("Invalid duration: {s}"))?;
        Ok(Duration::from_secs(sec))
    }
}

fn get_current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn print_startup_banner(asset: &str, network: &str, duration: &Duration, dex: &Option<String>) {
    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════╗");
    eprintln!(
        "║          Prediction Validator v{}                    ║",
        env!("CARGO_PKG_VERSION")
    );
    eprintln!("╠═══════════════════════════════════════════════════════════╣");
    eprintln!("║  Asset: {asset:<15}  Network: {network:<17} ║");
    if let Some(d) = dex {
        eprintln!(
            "║  DEX: {:<17}  Duration: {:<14} ║",
            d,
            format!("{}h", duration.as_secs() / 3600)
        );
    } else {
        eprintln!(
            "║  DEX: Validator Perps     Duration: {:<14} ║",
            format!("{}h", duration.as_secs() / 3600)
        );
    }
    eprintln!("╚═══════════════════════════════════════════════════════════╝");
    eprintln!();
}

// ============================================================================
// Logging Setup
// ============================================================================

fn setup_logging(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    // Determine log level
    let level = match cli.log_level.as_str() {
        "trace" => "trace",
        "debug" => "debug",
        "info" => "info",
        "warn" => "warn",
        "error" => "error",
        _ => "info",
    };

    // Determine log directory
    let log_dir = cli.log_dir.clone().unwrap_or_else(|| {
        let mut dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        dir.push("logs");
        dir.push("prediction_validator");
        dir
    });

    // Determine stdout format
    let stdout_format = match cli.log_format.as_str() {
        "json" => LogFormat::Json,
        _ => LogFormat::Pretty,
    };

    // Build LogConfig
    let log_config = LogConfig {
        log_dir,
        enable_multi_stream: cli.multi_stream,
        operational_level: "info".to_string(),
        diagnostic_level: "debug".to_string(),
        error_level: "warn".to_string(),
        enable_stdout: true,
        stdout_format,
        log_file: cli.log_file.clone(),
    };

    // Initialize logging
    let guards = init_logging(&log_config, Some(level))?;

    // Store guards to keep them alive
    let _ = LOG_GUARDS.set(guards);

    Ok(())
}

// ============================================================================
// Main Entry Point
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load .env if present
    dotenvy::dotenv().ok();

    // Parse CLI arguments
    let cli = Cli::parse();

    // Setup logging
    setup_logging(&cli)?;

    // Parse duration
    let duration = parse_duration(&cli.duration)?;

    // Determine network
    let base_url = match cli.network.to_lowercase().as_str() {
        "mainnet" => BaseUrl::Mainnet,
        "testnet" => BaseUrl::Testnet,
        _ => {
            warn!("Unknown network '{}', defaulting to testnet", cli.network);
            BaseUrl::Testnet
        }
    };

    // Print startup banner
    print_startup_banner(&cli.asset, &cli.network, &duration, &cli.dex);

    // Create validator
    let mut validator = PredictionValidator::new(
        cli.asset.clone(),
        cli.markout,
        cli.adverse_threshold_bps,
        cli.regime_breakdown,
        cli.report_interval,
        cli.warmup_samples,
    );

    // Auto-derive Binance symbol from asset using the canonical mapping.
    // HL-native tokens (HYPE, PURR, etc.) will get None and skip the feed.
    let mut binance_receiver: Option<tokio::sync::mpsc::Receiver<BinanceUpdate>> = None;
    if !cli.disable_binance_feed {
        let binance_symbol = resolve_binance_symbol(&cli.asset, cli.binance_symbol.as_deref());
        if let Some(ref sym) = binance_symbol {
            let (tx, rx) = tokio::sync::mpsc::channel(1000);
            let feed = BinanceFeed::for_symbol(sym, tx);
            tokio::spawn(async move {
                feed.run().await;
                tracing::warn!("Binance feed task terminated");
            });
            binance_receiver = Some(rx);
            validator.enable_binance_feed(&cli.asset);
            info!(
                asset = %cli.asset,
                binance_symbol = %sym,
                "Binance lead-lag feed active (auto-derived from asset)"
            );
        } else {
            warn!(
                asset = %cli.asset,
                "No Binance equivalent for asset — LeadLag model will not be validated. \
                 Use --binance-symbol to override."
            );
        }
    } else {
        warn!("Binance lead-lag feed DISABLED - LeadLag model will not be validated");
    }

    // Create InfoClient for WebSocket
    info!("Connecting to {} WebSocket...", cli.network);
    let mut info_client = InfoClient::new(None, Some(base_url)).await?;

    // Create message channel
    let (sender, mut receiver) = unbounded_channel::<Arc<Message>>();

    // Determine asset name
    let coin = if let Some(ref dex) = cli.dex {
        format!("{}:{}", dex, cli.asset)
    } else {
        cli.asset.clone()
    };

    // Subscribe to channels
    info!("Subscribing to market data for {}...", coin);

    let dex_param = cli.dex.clone();

    // AllMids subscription
    info_client
        .subscribe(
            Subscription::AllMids {
                dex: dex_param.clone(),
            },
            sender.clone(),
        )
        .await?;

    // Trades subscription
    info_client
        .subscribe(
            Subscription::Trades {
                coin: coin.clone(),
                dex: dex_param.clone(),
            },
            sender.clone(),
        )
        .await?;

    // L2Book subscription
    info_client
        .subscribe(
            Subscription::L2Book {
                coin: coin.clone(),
                dex: dex_param.clone(),
            },
            sender.clone(),
        )
        .await?;

    info!("Subscriptions active. Starting validation...");

    // Track start time
    let start_time = Instant::now();

    // Main event loop
    loop {
        // Check duration limit
        if start_time.elapsed() >= duration {
            info!("Duration limit reached. Shutting down...");
            break;
        }

        tokio::select! {
            // Binance lead-lag price feed (optional)
            // Process both price and trade updates for cross-venue analysis
            Some(update) = async {
                match binance_receiver.as_mut() {
                    Some(rx) => rx.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                match update {
                    BinanceUpdate::Price(price_update) => {
                        validator.on_binance_price(&price_update);
                        if cli.verbose {
                            debug!("Binance: mid=${:.2}", price_update.mid_price);
                        }
                    }
                    BinanceUpdate::Trade(trade_update) => {
                        // Feed Binance trades for cross-venue flow analysis
                        validator.on_binance_trade(&trade_update);
                        if cli.verbose {
                            debug!("Binance trade: {} {} @ ${:.2}",
                                trade_update.quantity,
                                if trade_update.is_buyer_maker { "SELL" } else { "BUY" },
                                trade_update.price
                            );
                        }
                    }
                }
            }

            Some(arc_msg) = receiver.recv() => {
                let msg = Arc::try_unwrap(arc_msg).unwrap_or_else(|arc| (*arc).clone());

                match msg {
                    Message::AllMids(all_mids) => {
                        if let Some(mid_str) = all_mids.data.mids.get(&cli.asset) {
                            if let Ok(mid) = mid_str.parse::<f64>() {
                                let ts = get_current_timestamp_ms();
                                validator.on_mid(ts, mid);

                                if cli.verbose {
                                    debug!("AllMids: {} = ${:.2}", cli.asset, mid);
                                }
                            }
                        }
                    }

                    Message::Trades(trades) => {
                        for trade in &trades.data {
                            if let (Ok(price), Ok(size)) = (
                                trade.px.parse::<f64>(),
                                trade.sz.parse::<f64>(),
                            ) {
                                let is_buy = trade.side == "B";
                                validator.on_trade(trade.time, price, size, is_buy);

                                if cli.verbose {
                                    debug!(
                                        "Trade: {} {} @ ${:.2}",
                                        size,
                                        if is_buy { "BUY" } else { "SELL" },
                                        price
                                    );
                                }
                            }
                        }
                    }

                    Message::L2Book(book) => {
                        let ts = book.data.time;

                        if book.data.levels.len() >= 2 {
                            let bids = &book.data.levels[0];
                            let asks = &book.data.levels[1];

                            if !bids.is_empty() && !asks.is_empty() {
                                let best_bid: f64 = bids[0].px.parse().unwrap_or(0.0);
                                let best_ask: f64 = asks[0].px.parse().unwrap_or(0.0);

                                if best_bid > 0.0 && best_ask > 0.0 {
                                    let mid = (best_bid + best_ask) / 2.0;

                                    let bid_depth: f64 = bids.iter()
                                        .take(5)
                                        .filter_map(|l| l.sz.parse::<f64>().ok())
                                        .sum();
                                    let ask_depth: f64 = asks.iter()
                                        .take(5)
                                        .filter_map(|l| l.sz.parse::<f64>().ok())
                                        .sum();

                                    validator.on_book(ts, mid, best_bid, best_ask, bid_depth, ask_depth);

                                    if cli.verbose {
                                        debug!(
                                            "L2Book: bid=${:.2} / ask=${:.2}",
                                            best_bid, best_ask
                                        );
                                    }
                                }
                            }
                        }
                    }

                    _ => {}
                }

                // Check if we should print a report
                if validator.should_report() {
                    validator.print_report();
                }
            }

            _ = tokio::signal::ctrl_c() => {
                info!("Received Ctrl+C. Shutting down...");
                break;
            }
        }
    }

    // Final report
    info!("Generating final report...");
    validator.print_report();

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_duration_hours() {
        assert_eq!(parse_duration("1h").unwrap(), Duration::from_secs(3600));
        assert_eq!(parse_duration("24h").unwrap(), Duration::from_secs(86400));
    }

    #[test]
    fn test_parse_duration_minutes() {
        assert_eq!(parse_duration("5m").unwrap(), Duration::from_secs(300));
        assert_eq!(parse_duration("30m").unwrap(), Duration::from_secs(1800));
    }

    #[test]
    fn test_parse_duration_seconds() {
        assert_eq!(parse_duration("60s").unwrap(), Duration::from_secs(60));
        assert_eq!(parse_duration("120").unwrap(), Duration::from_secs(120));
    }

    #[test]
    fn test_validator_creation() {
        let validator = PredictionValidator::new(
            "BTC".to_string(),
            1000,
            8.0, // Updated default threshold
            false,
            30,
            100, // warmup_samples
        );

        assert_eq!(validator.asset, "BTC");
        assert_eq!(validator.markout_ms, 1000);
        assert_eq!(validator.trackers.len(), 10); // 6 original + LeadLag + 3 cross-venue
        assert!(!validator.is_warmed_up());
    }

    #[test]
    fn test_model_tracker_health_status() {
        let mut tracker = ModelTracker::new(ValidatorPredictionType::InformedFlow);

        // Warming up
        assert_eq!(tracker.health_status(), "WARMING");

        // Add some samples
        for i in 0..100 {
            let predicted = (i as f64) / 100.0;
            let outcome = i % 2 == 0;
            tracker.update(predicted, outcome, 0);
        }

        // Should have a status now
        assert_ne!(tracker.health_status(), "WARMING");
    }

    #[test]
    fn test_observation_buffers() {
        let mut buffers = ObservationBuffers::new(100);

        // Add trades
        buffers.add_trade(TradeObservation {
            timestamp_ms: 1000,
            price: 100.0,
            size: 1.0,
            is_buy: true,
        });
        buffers.add_trade(TradeObservation {
            timestamp_ms: 2000,
            price: 101.0,
            size: 1.0,
            is_buy: false,
        });

        assert_eq!(buffers.trades.len(), 2);

        // Buy ratio should be 0.5
        let ratio = buffers.buy_ratio(10);
        assert!((ratio - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_prediction_type_markout() {
        assert_eq!(ValidatorPredictionType::InformedFlow.markout_ms(), 1000);
        assert_eq!(ValidatorPredictionType::RegimeHighVol.markout_ms(), 30000);
        assert_eq!(ValidatorPredictionType::BuyPressure.markout_ms(), 0);
    }
}
