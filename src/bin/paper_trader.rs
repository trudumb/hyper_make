//! Paper Trading Simulation Tool
//!
//! Tracks quote generation and simulates fills WITHOUT placing real orders.
//! Provides:
//! - Full quote generation pipeline (same as live trading)
//! - Prediction logging for calibration analysis
//! - Simulated fills based on market data
//! - PnL attribution and calibration reports
//!
//! # Usage
//!
//! ```bash
//! # Basic paper trading (60 seconds)
//! cargo run --bin paper_trader -- --asset BTC --duration 60
//!
//! # With verbose logging
//! cargo run --bin paper_trader -- --asset BTC --duration 300 --verbose
//!
//! # Generate calibration report
//! cargo run --bin paper_trader -- --asset BTC --duration 3600 --report
//! ```

use axum::{routing::get, Json, Router};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info, warn};

use hyperliquid_rust_sdk::market_maker::infra::{
    ws_handler, DashboardThrottle, DashboardWsConfig, DashboardWsState,
};

use hyperliquid_rust_sdk::{
    BaseUrl, EstimatorConfig, InfoClient, Ladder, LadderConfig, LadderStrategy, MarketParams,
    Message, OrderExecutor, ParameterEstimator, QuoteConfig, RiskConfig, Side, Subscription,
};
use tokio::sync::mpsc::unbounded_channel;

use hyperliquid_rust_sdk::market_maker::infra::metrics::dashboard::{
    classify_regime, compute_regime_probabilities, BookLevel, BookSnapshot, CalibrationState,
    DashboardState, FeatureCorrelationState, FeatureHealthState, FeatureValidationState,
    FillRecord, InteractionSignalState, LagAnalysisState, LiveQuotes, PnLAttribution, PricePoint,
    QuoteFillStats, QuoteSnapshot, RegimeSnapshot, RegimeState, SignalDecayState, SignalHealthInfo,
    SpreadBucket,
};
use hyperliquid_rust_sdk::market_maker::simulation::{
    CalibrationAnalyzer, FillOutcome, FillSimulator, FillSimulatorConfig, MarketStateSnapshot,
    ModelPredictions, ObservedOutcomes, OutcomeTracker, PredictionLogger, SimulatedFill,
    SimulationExecutor,
};
use std::collections::{HashMap, VecDeque};

// Adaptive GLFT + Process models
use hyperliquid_rust_sdk::market_maker::adaptive::{
    AdaptiveBayesianConfig, AdaptiveSpreadCalculator,
};
use hyperliquid_rust_sdk::market_maker::process_models::{
    FundingRateEstimator, HJBConfig, HJBInventoryController,
    HawkesOrderFlowEstimator, LiquidationCascadeDetector, SpreadProcessEstimator,
};

// Full signal pipeline components
use hyperliquid_rust_sdk::market_maker::{
    // Tier 1: Adverse selection
    AdverseSelectionConfig, AdverseSelectionEstimator, DepthDecayAS, EnhancedASClassifier,
    PreFillASClassifier, TradeObservation as MicroTradeObs,
    // Tier 2: Process model configs
    FundingConfig, HawkesConfig, LiquidationConfig, SpreadConfig,
    // Config
    StochasticConfig,
    // Estimator types
    BookLevel as EstimatorBookLevel, CumulativeOFI, CumulativeOFIConfig, EnhancedFlowConfig,
    EnhancedFlowEstimator, HmmObservation, LiquidityEvaporationConfig,
    LiquidityEvaporationDetector, RegimeHMM, ThresholdKappa, ThresholdKappaConfig,
    TradeFeatures, TradeSizeDistribution, TradeSizeDistributionConfig, VpinConfig, VpinEstimator,
    // Infra
    MarginAwareSizer, MarginConfig,
    // Strategy/params
    ParameterAggregator, ParameterSources, SignalIntegrator, SignalIntegratorConfig,
    PositionDecisionConfig, PositionDecisionEngine, action_to_inventory_ratio,
};
use hyperliquid_rust_sdk::market_maker::belief::{BeliefUpdate, CentralBeliefState};
use hyperliquid_rust_sdk::market_maker::risk::{RiskState, RiskAggregator, RiskSeverity};
use hyperliquid_rust_sdk::market_maker::risk::monitors::*;
use hyperliquid_rust_sdk::round_to_significant_and_decimal;
use hyperliquid_rust_sdk::market_maker::checkpoint::{
    CheckpointBundle, CheckpointManager, CheckpointMetadata,
    EnsembleWeightsCheckpoint, KellyTrackerCheckpoint, KillSwitchCheckpoint,
};
use hyperliquid_rust_sdk::market_maker::learning::{
    ExperienceLogger, ExperienceParams, ExperienceRecord, ExperienceSource,
    ExplorationStrategy, MDPAction, MDPState, QLearningAgent, QLearningConfig,
    RLPolicyRecommendation, Reward,
};
use hyperliquid_rust_sdk::market_maker::stochastic::{
    StochasticControlBuilder, StochasticControlConfig,
};
use hyperliquid_rust_sdk::market_maker::calibration::LearnedParameters;
use hyperliquid_rust_sdk::market_maker::{CalibrationController, CalibrationControllerConfig};
use hyperliquid_rust_sdk::market_maker::{BinanceFeed, BinanceTradeUpdate, BinanceUpdate, resolve_binance_symbol};
use hyperliquid_rust_sdk::market_maker::analytics::{
    SharpeTracker, PerSignalSharpeTracker, SignalPnLAttributor,
    EdgeTracker, EdgeSnapshot, AnalyticsLogger, CycleContributions, SignalContribution,
};

// ============================================================================
// CLI Arguments
// ============================================================================

#[derive(Parser)]
#[command(name = "paper_trader")]
#[command(version, about = "Paper Trading Simulation Tool", long_about = None)]
struct Cli {
    /// Asset to simulate trading for (e.g., BTC, ETH, SOL)
    #[arg(long, default_value = "BTC")]
    asset: String,

    /// Duration to run simulation in seconds
    #[arg(long, default_value = "300")]
    duration: u64,

    /// Network (mainnet, testnet)
    #[arg(long, default_value = "mainnet")]
    network: String,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Enable verbose simulation logging
    #[arg(long)]
    verbose: bool,

    /// Output directory for logs and reports
    #[arg(long, default_value = "logs/paper_trading")]
    output_dir: String,

    /// Generate calibration report at end
    #[arg(long)]
    report: bool,

    /// Risk aversion parameter (gamma)
    #[arg(long, default_value = "0.5")]
    gamma: f64,

    /// Target spread in basis points
    #[arg(long, default_value = "10.0")]
    target_spread_bps: f64,

    /// Number of ladder levels per side
    #[arg(long, default_value = "3")]
    ladder_levels: usize,

    /// Target liquidity per level
    #[arg(long, default_value = "0.01")]
    target_liquidity: f64,

    /// Enable live dashboard
    #[arg(long)]
    dashboard: bool,

    /// Metrics HTTP port for dashboard API (default 8080)
    #[arg(long, default_value = "8080")]
    metrics_port: u16,

    /// Quote refresh interval in milliseconds (default 500)
    #[arg(long, default_value = "500")]
    quote_interval_ms: u64,

    /// Disable Binance price feed for cross-exchange lead-lag signal
    #[arg(long)]
    disable_binance_feed: bool,

    /// Binance symbol override for lead-lag signal (e.g., btcusdt, ethusdt).
    /// If not set, auto-derived from the trading asset. Assets without a
    /// Binance equivalent (HYPE, PURR, etc.) will have the feed disabled.
    #[arg(long)]
    binance_symbol: Option<String>,

    /// Checkpoint directory for warm-starting models (default: data/checkpoints/paper/{asset})
    #[arg(long)]
    checkpoint_dir: Option<String>,

    /// Disable checkpoint persistence
    #[arg(long)]
    disable_checkpoint: bool,

    /// Maximum checkpoint age in hours before skipping restore
    #[arg(long, default_value = "4")]
    max_checkpoint_age_hours: f64,

    /// Force restore checkpoint regardless of age (useful for iterative paper trading)
    #[arg(long)]
    force_restore: bool,

    /// Paper trading calibration mode: tighter spreads, faster warmup for generating fills
    #[arg(long)]
    paper_mode: bool,

    /// Quote lifetime in seconds for fill probability model
    #[arg(long, default_value_t = 10.0)]
    quote_lifetime_s: f64,

    /// Minimum order notional in USD
    #[arg(long, default_value_t = 10.0)]
    min_notional: f64,

    /// Max position size in notional USD (asset-agnostic).
    /// Converted to contracts on first valid mid price: max_position = max_position_usd / mid_price.
    /// Default: $200 (~6.4 HYPE at $31). Forces inventory management to engage sooner.
    #[arg(long, default_value_t = 200.0)]
    max_position_usd: f64,
}

// ============================================================================
// Feature Health Mock Data Generator
// ============================================================================

/// Generate mock feature health data for dashboard display.
/// In production, this would come from actual trackers.
fn generate_mock_feature_health(cycle_count: u64) -> FeatureHealthState {
    let timestamp_ms = chrono::Utc::now().timestamp_millis();

    // Simulate slowly varying signal health
    let base_mi = 0.04 + 0.01 * ((cycle_count as f64 / 100.0).sin());

    FeatureHealthState {
        signal_decay: SignalDecayState {
            signals: vec![
                SignalHealthInfo {
                    name: "book_imbalance".to_string(),
                    mi: base_mi + 0.005,
                    trend: "stable".to_string(),
                    half_life_days: Some(45.0),
                    status: "healthy".to_string(),
                    mi_change_pct: 2.0,
                },
                SignalHealthInfo {
                    name: "trade_flow".to_string(),
                    mi: base_mi - 0.005,
                    trend: "up".to_string(),
                    half_life_days: Some(35.0),
                    status: "healthy".to_string(),
                    mi_change_pct: 5.0,
                },
                SignalHealthInfo {
                    name: "momentum_10s".to_string(),
                    mi: base_mi - 0.015,
                    trend: "down".to_string(),
                    half_life_days: Some(18.0),
                    status: "warning".to_string(),
                    mi_change_pct: -12.0,
                },
                SignalHealthInfo {
                    name: "binance_lead".to_string(),
                    mi: base_mi + 0.02,
                    trend: "down".to_string(),
                    half_life_days: Some(25.0),
                    status: "healthy".to_string(),
                    mi_change_pct: -8.0,
                },
            ],
            alerts: vec![],
            signal_count: 4,
            issue_count: 1,
        },
        correlation: FeatureCorrelationState {
            feature_names: vec![
                "kappa".to_string(),
                "sigma".to_string(),
                "momentum".to_string(),
                "flow".to_string(),
                "book_imb".to_string(),
            ],
            // 5x5 correlation matrix (row-major)
            correlation_matrix: vec![
                1.0, 0.3, -0.1, 0.2, 0.7, 0.3, 1.0, 0.4, -0.2, 0.1, -0.1, 0.4, 1.0, 0.6, 0.0, 0.2,
                -0.2, 0.6, 1.0, 0.3, 0.7, 0.1, 0.0, 0.3, 1.0,
            ],
            vif: vec![2.1, 1.3, 1.8, 1.5, 2.0],
            condition_number: 8.5,
            highly_correlated_pairs: vec![],
            is_valid: true,
        },
        validation: FeatureValidationState {
            features: vec![],
            issue_count: 0,
            issue_rate: 0.0,
            validation_count: cycle_count as usize,
        },
        lag_analysis: LagAnalysisState {
            optimal_lag_ms: -150,
            mi_at_lag: 0.032,
            ccf: vec![], // Would contain cross-correlation function
            signal_ready: true,
            signal_name: "binance_mid".to_string(),
            target_name: "hyperliquid_mid".to_string(),
        },
        interactions: InteractionSignalState {
            vol_x_momentum: 0.3 + 0.1 * ((cycle_count as f64 / 50.0).sin()),
            regime_x_inventory: 0.2 + 0.05 * ((cycle_count as f64 / 30.0).cos()),
            jump_x_flow: 0.1 + 0.05 * ((cycle_count as f64 / 70.0).sin()),
            timestamp_ms,
        },
    }
}

// ============================================================================
// Simulation State (using real LadderStrategy)
// ============================================================================

/// Configuration for SimulationState construction
struct SimulationConfig {
    asset: String,
    gamma: f64,
    target_spread_bps: f64,
    ladder_levels: usize,
    target_liquidity: f64,
    quote_lifetime_s: f64,
    min_notional: f64,
    paper_mode: bool,
    /// Max position in notional USD (converted to contracts on first valid mid)
    max_position_usd: f64,
}

/// Pending fill for adverse selection outcome tracking
struct PendingFillOutcome {
    timestamp_ns: u64,
    #[allow(dead_code)]
    fill_price: f64,
    is_buy: bool,
    mid_at_fill: f64,
}

/// Complete simulation state using production LadderStrategy
struct SimulationState {
    /// Asset being simulated
    #[allow(dead_code)]
    asset: String,
    /// Current mid price
    mid_price: f64,
    /// Previous mid price (for return computation)
    prev_mid: f64,
    /// Best bid
    best_bid: f64,
    /// Best ask
    best_ask: f64,
    /// Book depth (bid levels, ask levels)
    bid_levels: Vec<(f64, f64)>,
    ask_levels: Vec<(f64, f64)>,

    /// Parameter estimator (same as production)
    estimator: ParameterEstimator,
    /// Real ladder strategy (same as production)
    strategy: LadderStrategy,
    /// Simulated inventory
    inventory: f64,
    /// Quote cycle counter
    cycle_count: u64,
    /// Risk aversion
    gamma: f64,

    /// Configuration
    max_position: f64,
    /// Max position in notional USD (source-of-truth, asset-agnostic)
    max_position_usd: f64,
    /// Whether max_position has been derived from USD (on first valid mid)
    max_position_usd_resolved: bool,
    target_liquidity: f64,
    /// Price decimals (BTC=1, ETH=2, etc.)
    decimals: u32,
    /// Size decimals
    sz_decimals: u32,
    /// Quote lifetime in seconds for fill probability model
    quote_lifetime_s: f64,
    /// Minimum order notional in USD
    min_notional: f64,

    // === Adaptive GLFT Components ===
    adaptive_spreads: AdaptiveSpreadCalculator,
    hjb_controller: HJBInventoryController,

    // === Tier 1: Adverse Selection ===
    adverse_selection: AdverseSelectionEstimator,
    depth_decay_as: DepthDecayAS,
    pre_fill_classifier: PreFillASClassifier,
    enhanced_classifier: EnhancedASClassifier,
    liquidation_detector: LiquidationCascadeDetector,

    // === Tier 2: Process Models ===
    hawkes: HawkesOrderFlowEstimator,
    funding: FundingRateEstimator,
    spread_tracker: SpreadProcessEstimator,

    // === Stochastic Signals ===
    vpin: VpinEstimator,
    regime_hmm: RegimeHMM,
    signal_integrator: SignalIntegrator,
    enhanced_flow: EnhancedFlowEstimator,
    liquidity_evaporation: LiquidityEvaporationDetector,
    cofi: CumulativeOFI,
    trade_size_dist: TradeSizeDistribution,
    position_decision: PositionDecisionEngine,
    threshold_kappa: ThresholdKappa,

    // === Central Belief System ===
    central_beliefs: CentralBeliefState,

    // === Deprecated but needed for ParameterSources ===
    beliefs_builder: StochasticControlBuilder,
    stochastic_config: StochasticConfig,
    margin_sizer: MarginAwareSizer,

    // === Cached state for signal processing ===
    cached_bid_sizes: Vec<f64>,
    cached_ask_sizes: Vec<f64>,
    cached_trades: VecDeque<(f64, bool, u64)>,
    /// Last trade timestamp for inter-arrival computation
    last_trade_ms: u64,
    /// Last computed book imbalance for RL state (updated on each trade)
    last_book_imbalance: f64,

    // === Signal health logging ===
    last_signal_log: Instant,

    // === Checkpoint persistence ===
    checkpoint_manager: Option<CheckpointManager>,
    session_start: Instant,
    last_checkpoint_save: Instant,
    learned_params: LearnedParameters,

    // === Learning feedback loops ===
    calibration_controller: CalibrationController,
    pending_fill_outcomes: VecDeque<PendingFillOutcome>,
    total_fill_count: u64,

    /// Paper trading calibration mode (tighter spreads, faster warmup)
    #[allow(dead_code)]
    paper_mode: bool,

    // === RL Agent for Policy Learning ===
    /// Q-Learning agent for adaptive quoting policy.
    /// Paper trader uses Thompson sampling exploration for aggressive learning.
    rl_agent: QLearningAgent,

    // === Experience Logging ===
    /// JSONL experience logger for offline RL training.
    experience_logger: Option<ExperienceLogger>,
    /// Session identifier for experience records.
    session_id: String,
}

impl SimulationState {
    fn new(config: SimulationConfig) -> Self {
        let SimulationConfig {
            asset,
            gamma,
            target_spread_bps,
            ladder_levels,
            target_liquidity,
            quote_lifetime_s,
            min_notional,
            paper_mode,
            max_position_usd,
        } = config;

        // Paper mode: use competitive gamma for tighter theoretical spreads.
        // GLFT: δ* = (1/γ)ln(1 + γ/κ) + fee
        // γ=0.07, κ=8000 → δ* ≈ 2.75 bps/side (down from 9.0 at γ=0.24)
        let effective_gamma = if paper_mode { 0.07 } else { gamma };

        // Paper mode: lower spread floor for more aggressive quoting
        let min_spread = if paper_mode {
            (target_spread_bps / 10_000.0 / 2.0).min(0.0002) // Cap at 2 bps per side
        } else {
            target_spread_bps / 10_000.0 / 2.0
        };

        // Create production-style RiskConfig
        let risk_config = RiskConfig {
            gamma_base: effective_gamma.clamp(0.01, 10.0),
            min_spread_floor: min_spread,
            ..Default::default()
        };

        // Create LadderConfig with specified levels
        let mut ladder_config = LadderConfig {
            num_levels: ladder_levels,
            ..Default::default()
        };
        if paper_mode {
            // Lower min depth for tighter quoting in calibration mode
            ladder_config.min_depth_bps = 2.0;
        }

        // Create the real LadderStrategy
        let strategy = LadderStrategy::with_config(risk_config, ladder_config);

        // Determine decimals based on asset
        let (decimals, sz_decimals) = match asset.as_str() {
            "BTC" => (1, 4),
            "ETH" => (2, 3),
            "SOL" => (3, 2),
            _ => (2, 3),
        };

        // Create adaptive components — paper mode lowers the floor for more fills
        // Default floor = fee(1.5) + AS_mean(3.0) + k(1.17) × AS_std(3.0) = 8.01 bps
        // Paper mode floor = fee(1.5) + AS_mean(1.0) + k(0.0) × AS_std(2.0) = 2.5 bps
        let adaptive_config = if paper_mode {
            AdaptiveBayesianConfig {
                as_prior_mean: 0.0001,       // 1 bps AS prior (vs 3 bps default)
                as_prior_std: 0.0002,        // 2 bps uncertainty (vs 3 bps default)
                floor_risk_k: 0.0,           // No safety margin — learn from fills
                floor_absolute_min: 0.0001,  // 1 bp hard floor
                ..AdaptiveBayesianConfig::default()
            }
        } else {
            AdaptiveBayesianConfig::default()
        };
        let adaptive_spreads = AdaptiveSpreadCalculator::new(adaptive_config);
        let mut hjb_controller = HJBInventoryController::new(HJBConfig::default());
        hjb_controller.start_session();

        // Paper mode: faster warmup for quicker convergence
        let estimator_config = if paper_mode {
            EstimatorConfig {
                min_volume_ticks: 3,
                min_trade_observations: 2,
                ..EstimatorConfig::default()
            }
        } else {
            EstimatorConfig::default()
        };

        let session_id = uuid::Uuid::new_v4().to_string();

        Self {
            asset,
            mid_price: 0.0,
            prev_mid: 0.0,
            best_bid: 0.0,
            best_ask: 0.0,
            bid_levels: Vec::new(),
            ask_levels: Vec::new(),
            estimator: ParameterEstimator::new(estimator_config),
            strategy,
            inventory: 0.0,
            cycle_count: 0,
            gamma: effective_gamma,
            max_position: 1.0, // Placeholder; resolved from max_position_usd on first valid mid
            max_position_usd,
            max_position_usd_resolved: false,
            target_liquidity,
            decimals,
            sz_decimals,
            quote_lifetime_s,
            min_notional,
            // Adaptive components
            adaptive_spreads,
            hjb_controller,
            // Tier 1: Adverse selection
            adverse_selection: AdverseSelectionEstimator::new(AdverseSelectionConfig::default()),
            depth_decay_as: DepthDecayAS::default(),
            pre_fill_classifier: PreFillASClassifier::default(),
            enhanced_classifier: EnhancedASClassifier::default_config(),
            liquidation_detector: LiquidationCascadeDetector::new(LiquidationConfig::default()),
            // Tier 2: Process models
            hawkes: HawkesOrderFlowEstimator::new(HawkesConfig::default()),
            funding: FundingRateEstimator::new(FundingConfig::default()),
            spread_tracker: SpreadProcessEstimator::new(SpreadConfig::default()),
            // Stochastic signals
            vpin: VpinEstimator::new(VpinConfig::default()),
            regime_hmm: RegimeHMM::new(),
            signal_integrator: SignalIntegrator::new(SignalIntegratorConfig::default()),
            enhanced_flow: EnhancedFlowEstimator::new(EnhancedFlowConfig::default()),
            liquidity_evaporation: LiquidityEvaporationDetector::new(LiquidityEvaporationConfig::default()),
            cofi: CumulativeOFI::new(CumulativeOFIConfig::default()),
            trade_size_dist: TradeSizeDistribution::new(TradeSizeDistributionConfig::default()),
            position_decision: PositionDecisionEngine::new(PositionDecisionConfig::default()),
            threshold_kappa: ThresholdKappa::new(ThresholdKappaConfig::default()),
            // Central belief system
            central_beliefs: CentralBeliefState::default_config(),
            // Needed for ParameterSources
            beliefs_builder: StochasticControlBuilder::new(StochasticControlConfig::default()),
            stochastic_config: StochasticConfig::default(),
            margin_sizer: MarginAwareSizer::new(MarginConfig::default()),
            // Cached state
            cached_bid_sizes: Vec::new(),
            cached_ask_sizes: Vec::new(),
            cached_trades: VecDeque::new(),
            last_trade_ms: 0,
            last_book_imbalance: 0.0,
            // Signal health logging
            last_signal_log: Instant::now(),
            // Checkpoint persistence
            checkpoint_manager: None,
            session_start: Instant::now(),
            last_checkpoint_save: Instant::now(),
            learned_params: LearnedParameters::default(),

            // Learning feedback loops
            calibration_controller: CalibrationController::new(CalibrationControllerConfig {
                enabled: true,
                target_fill_rate_per_hour: 60.0,
                min_gamma_mult: 0.3,
                ..CalibrationControllerConfig::default()
            }),
            pending_fill_outcomes: VecDeque::with_capacity(64),
            total_fill_count: 0,
            paper_mode,
            // RL agent: Thompson sampling for aggressive exploration in paper trading
            rl_agent: QLearningAgent::new(QLearningConfig {
                exploration: ExplorationStrategy::ThompsonSampling,
                min_observations: 5, // Lower threshold for paper — explore more
                ..Default::default()
            }),
            // Experience logging for offline RL training
            experience_logger: {
                ExperienceLogger::new("logs/experience", ExperienceSource::Paper, &session_id)
                    .ok()
            },
            session_id,
        }
    }

    /// Update state from all mids message
    fn update_mid(&mut self, mid: f64) {
        if mid > 0.0 {
            self.prev_mid = self.mid_price;
            self.mid_price = mid;

            // Resolve USD→contracts on first valid mid price
            if !self.max_position_usd_resolved && self.max_position_usd > 0.0 {
                self.max_position = self.max_position_usd / mid;
                self.max_position_usd_resolved = true;
                info!(
                    max_position_usd = %format!("{:.2}", self.max_position_usd),
                    mid_price = %format!("{:.2}", mid),
                    max_position_contracts = %format!("{:.6}", self.max_position),
                    "Resolved USD position limit → contracts"
                );
            }

            // Check pending fill outcomes for adverse selection learning
            self.check_adverse_selection_outcomes();

            // === FIX: Update AS estimator with current mid to resolve pending fills ===
            // Without this, horizon_stats.as_ewma stays at 0.0 and edge prediction has no AS correction
            self.adverse_selection.update(mid);

            let now_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;

            // Update HJB controller with current volatility estimate
            let sigma = self.estimator.sigma_clean().max(0.00001);
            self.hjb_controller.update_sigma(sigma);

            // === Replicate live signal processing from handlers.rs ===

            // Threshold kappa: compute return and update TAR model
            if self.prev_mid > 0.0 {
                let return_bps = (mid - self.prev_mid) / self.prev_mid * 10_000.0;
                self.threshold_kappa.update(return_bps);
            }

            // Signal integrator: track HL price
            self.signal_integrator.on_hl_price(mid, now_ms as i64);

            // Central beliefs: price return update
            if self.prev_mid > 0.0 {
                let return_frac = (mid - self.prev_mid) / self.prev_mid;
                // Estimate dt from mid update frequency (~100ms typical)
                let dt_secs = 0.1;
                self.central_beliefs.update(BeliefUpdate::PriceReturn {
                    return_frac,
                    dt_secs,
                    timestamp_ms: now_ms,
                });
            }

            // Liquidation detector: update internal state
            self.liquidation_detector.update();
        }
    }

    /// Update state from L2 book
    fn update_book(&mut self, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) {
        self.bid_levels = bids.clone();
        self.ask_levels = asks.clone();

        if let Some((price, _)) = self.bid_levels.first() {
            self.best_bid = *price;
        }
        if let Some((price, _)) = self.ask_levels.first() {
            self.best_ask = *price;
        }

        // Cache top 5 bid/ask sizes for depth imbalance
        self.cached_bid_sizes = bids.iter().take(5).map(|(_, sz)| *sz).collect();
        self.cached_ask_sizes = asks.iter().take(5).map(|(_, sz)| *sz).collect();

        // Update adaptive kappa from book depth
        if self.mid_price > 0.0 {
            self.adaptive_spreads
                .on_l2_update(&bids, &asks, self.mid_price);

            // Feed L2 to parameter estimator for book-structure kappa estimation
            // CRITICAL: Live system does this in l2_book.rs:88 but paper trader was missing it,
            // causing book_kappa to stay at prior (2500) and never learn from actual book depth
            self.estimator.on_l2_book(&bids, &asks, self.mid_price);
        }

        // === Replicate live L2 signal processing from handlers.rs:793-860 ===

        // Spread tracker: update from BBO + volatility
        if self.best_bid > 0.0 && self.best_ask > self.best_bid {
            self.spread_tracker
                .update(self.best_bid, self.best_ask, self.estimator.sigma());
        }

        // Pre-fill classifier: orderbook imbalance
        let bid_depth: f64 = bids.iter().take(5).map(|(_, sz)| sz).sum();
        let ask_depth: f64 = asks.iter().take(5).map(|(_, sz)| sz).sum();
        self.pre_fill_classifier.update_orderbook(bid_depth, ask_depth);

        // Enhanced classifier: BBO + depth update
        if let (Some(&(best_bid, _)), Some(&(best_ask, _))) = (bids.first(), asks.first()) {
            let timestamp_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            self.enhanced_classifier.on_book_update(
                best_bid, best_ask, bid_depth, ask_depth, timestamp_ms,
            );
        }

        // Liquidity evaporation: near-touch depth
        let near_bid_depth: f64 = bids.iter().take(3).map(|(_, sz)| sz).sum();
        let near_ask_depth: f64 = asks.iter().take(3).map(|(_, sz)| sz).sum();
        let near_touch_depth = near_bid_depth + near_ask_depth;
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.liquidity_evaporation.on_book(near_touch_depth, timestamp_ms);

        // COFI delta update
        let bid_delta = near_bid_depth * 0.1;
        let ask_delta = near_ask_depth * 0.1;
        self.cofi.on_book_update(bid_delta, ask_delta, timestamp_ms);
    }

    /// Update from trade — full signal pipeline replicating live handlers.rs:135-328
    fn on_trade(&mut self, price: f64, size: f64, is_buy: bool) {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // 1. Core estimator (existing)
        self.estimator.on_trade(now_ms, price, size, Some(is_buy));

        // 2. Enhanced microstructure classifier
        self.enhanced_classifier.on_trade(MicroTradeObs {
            timestamp_ms: now_ms,
            price,
            size,
            is_buy,
        });

        // 3. Signal integrator: buy pressure tracking
        self.signal_integrator.on_trade_for_pressure(size, is_buy);

        // 3a. Signal integrator: InformedFlow EM estimator
        {
            let inter_arrival_ms = if self.last_trade_ms > 0 {
                now_ms.saturating_sub(self.last_trade_ms)
            } else {
                1000 // default 1s for first trade
            };
            // Book imbalance from cached sizes: (bid_depth - ask_depth) / (bid_depth + ask_depth)
            let bid_depth: f64 = self.cached_bid_sizes.iter().sum();
            let ask_depth: f64 = self.cached_ask_sizes.iter().sum();
            let book_imbalance = if bid_depth + ask_depth > 0.0 {
                (bid_depth - ask_depth) / (bid_depth + ask_depth)
            } else {
                0.0
            };
            // Cache for RL state at fill time
            self.last_book_imbalance = book_imbalance;
            // Price impact: how much the mid moved since last trade (crude proxy)
            let price_impact_bps = if self.prev_mid > 0.0 {
                ((self.mid_price - self.prev_mid) / self.prev_mid).abs() * 10_000.0
            } else {
                0.0
            };
            let features = TradeFeatures {
                size,
                inter_arrival_ms,
                price_impact_bps,
                book_imbalance,
                is_buy,
                timestamp_ms: now_ms,
            };
            self.signal_integrator.on_trade(&features);
            self.last_trade_ms = now_ms;
        }

        // 3b. Hawkes order flow (feeds flow_imbalance for HMM observations)
        self.hawkes.record_trade(is_buy, size);

        // 4. Trade size distribution
        self.trade_size_dist.on_trade(size);

        // 5. Central beliefs: market trade
        if price > 0.0 {
            self.central_beliefs.update(BeliefUpdate::MarketTrade {
                price,
                mid: self.mid_price,
                timestamp_ms: now_ms,
            });
        }

        // 6. VPIN bucket update
        if let Some(_vpin_value) = self.vpin.on_trade(size, price, self.mid_price, now_ms) {
            // Bucket completed - wire to signal integrator + central beliefs
            let vpin = self.vpin.vpin();
            let vpin_velocity = self.vpin.vpin_velocity();

            let vpin_fresh = self.vpin.is_valid() && !self.vpin.is_stale(now_ms);
            self.signal_integrator.set_vpin(
                vpin,
                vpin_velocity,
                vpin_fresh,
            );

            let order_flow_direction = self.vpin.order_flow_direction();
            let vpin_buckets = self.vpin.bucket_count();

            // Depth-weighted imbalance from cached book
            let bid_levels: Vec<EstimatorBookLevel> = self.cached_bid_sizes.iter()
                .map(|&sz| EstimatorBookLevel { size: sz })
                .collect();
            let ask_levels: Vec<EstimatorBookLevel> = self.cached_ask_sizes.iter()
                .map(|&sz| EstimatorBookLevel { size: sz })
                .collect();
            let depth_ofi = self.enhanced_flow.depth_weighted_imbalance(
                &bid_levels,
                &ask_levels,
            );

            let liquidity_evaporation = self.liquidity_evaporation.evaporation_score();
            let confidence = (vpin_buckets as f64 / 50.0).min(1.0);

            // Trade size anomaly + COFI
            let trade_size_sigma = self.trade_size_dist.median_sigma();
            let toxicity_acceleration = self.trade_size_dist.toxicity_acceleration(vpin);
            let cofi = self.cofi.cofi();
            let cofi_velocity = self.cofi.cofi_velocity();
            let is_sustained_shift = self.cofi.is_sustained_shift();

            self.central_beliefs.update(BeliefUpdate::MicrostructureUpdate {
                vpin,
                vpin_velocity,
                depth_ofi,
                liquidity_evaporation,
                order_flow_direction,
                confidence,
                vpin_buckets,
                trade_size_sigma,
                toxicity_acceleration,
                cofi,
                cofi_velocity,
                is_sustained_shift,
            });
        }

        // 7. Cache trade
        const MAX_CACHED_TRADES: usize = 500;
        self.cached_trades.push_back((size, is_buy, now_ms));
        while self.cached_trades.len() > MAX_CACHED_TRADES {
            self.cached_trades.pop_front();
        }

        // 8. Regime HMM forward update (with leading indicators)
        // OI and liquidation data not yet available in paper mode — wire the code path
        // so it activates when the data feeds are connected
        let oi_level = 1.0_f64; // TODO: wire from Hyperliquid OI feed
        let oi_velocity = 0.0_f64; // TODO: wire from OI rate-of-change tracker
        let liquidation_pressure = 0.0_f64; // TODO: wire from OI drop + extreme funding
        let obs = HmmObservation::new_full(
            self.estimator.sigma(),
            self.spread_tracker.current_spread_bps(),
            self.hawkes.flow_imbalance(),
            oi_level,
            oi_velocity,
            liquidation_pressure,
        );
        self.regime_hmm.forward_update(&obs);

        // 8b. Feed regime probabilities to signal integrator for regime-aware kappa blending
        {
            let probs = self.regime_hmm.regime_probabilities();
            self.signal_integrator.set_regime_probabilities(probs);
        }

        // 9. Position continuation model: sync regime
        let regime_probs = self.regime_hmm.regime_probabilities();
        let continuation_regime = if regime_probs[3] > 0.3 {
            "cascade"
        } else if regime_probs[2] > 0.4 {
            "bursty"
        } else if regime_probs[0] > 0.5 {
            "quiet"
        } else {
            "normal"
        };
        if self.position_decision.current_regime() != continuation_regime {
            self.position_decision.reset_for_regime(continuation_regime);
        }
    }

    /// Generate quotes using the REAL LadderStrategy (same as production)
    fn generate_quotes(&mut self) -> Option<Ladder> {
        if self.mid_price <= 0.0 {
            return None;
        }

        self.cycle_count += 1;

        // Build QuoteConfig
        let quote_config = QuoteConfig {
            mid_price: self.mid_price,
            decimals: self.decimals,
            sz_decimals: self.sz_decimals,
            min_notional: self.min_notional,
        };

        // Update calibration controller with current estimator state
        let as_fills = self.pre_fill_classifier.learning_diagnostics().samples as u64;
        let kappa_conf = self.estimator.kappa_confidence();
        self.calibration_controller.update_calibration_status(as_fills, kappa_conf);

        // Build MarketParams from estimator (same as production)
        let mut market_params = self.build_market_params();

        // Edge surface diagnostic logging
        if !market_params.should_quote_edge {
            tracing::debug!(
                edge_bps = %format!("{:.2}", market_params.current_edge_bps),
                flow_confidence = %format!("{:.2}", market_params.flow_decomp_confidence),
                "Edge surface indicates no edge - GLFT circuit breaker may trigger"
            );
        }

        // Paper mode: confidence-gated kappa floor for competitive spreads.
        // During warmup (low confidence): force kappa=8000 for tight spreads and fast learning.
        // As confidence grows (>0.75): blend floor with learned kappa to transition smoothly.
        // GLFT: δ* ≈ 1/κ + fee → κ=8000 gives 2.75 bps/side.
        if self.paper_mode {
            const KAPPA_WARMUP_FLOOR: f64 = 8000.0;
            const KAPPA_CONFIDENCE_THRESHOLD: f64 = 0.75;
            const KAPPA_TRANSITION_MIN: f64 = 2000.0;

            let paper_kappa_floor = if kappa_conf < KAPPA_CONFIDENCE_THRESHOLD {
                // During warmup: force competitive kappa to bootstrap fill data
                KAPPA_WARMUP_FLOOR
            } else {
                // Transition: blend floor with learned value as confidence grows
                let learned = market_params.kappa_robust;
                (KAPPA_WARMUP_FLOOR * (1.0 - kappa_conf) + learned * kappa_conf)
                    .max(KAPPA_TRANSITION_MIN)
            };

            if market_params.kappa < paper_kappa_floor {
                market_params.kappa = paper_kappa_floor;
            }
            if market_params.adaptive_kappa < paper_kappa_floor {
                market_params.adaptive_kappa = paper_kappa_floor;
            }
            if market_params.kappa_robust < paper_kappa_floor {
                market_params.kappa_robust = paper_kappa_floor;
            }

            // Disable regime kappa blending during warmup to prevent the floor from
            // being blended down (70% floor + 30% regime_kappa ≈ 2000 → kappa drops to ~6200).
            // Once past confidence threshold, regime blending can resume safely.
            if kappa_conf < KAPPA_CONFIDENCE_THRESHOLD {
                market_params.regime_kappa = None;
            }
        }

        // === RL Agent: Observe state and get exploration action ===
        {
            let vol_ratio = market_params.sigma / market_params.sigma_effective.max(0.0001);
            let mdp_state = MDPState::from_continuous(
                self.inventory,
                self.max_position,
                market_params.book_imbalance,
                vol_ratio,
                self.adverse_selection.best_horizon_as_bps() / 10.0, // normalize to ~[0,1]
                market_params.hawkes_branching_ratio,
            );
            // Always explore in paper trading
            let rl_rec = RLPolicyRecommendation::from_agent(&mut self.rl_agent, mdp_state.to_index(), true);
            self.rl_agent.push_state_action_idx(rl_rec.state_idx, rl_rec.action_idx);

            // Store RL recommendations in market_params for logging
            market_params.rl_spread_delta_bps = rl_rec.spread_delta_bps;
            market_params.rl_bid_skew_bps = rl_rec.bid_skew_bps;
            market_params.rl_ask_skew_bps = rl_rec.ask_skew_bps;
            market_params.rl_confidence = rl_rec.confidence;
            market_params.rl_is_exploration = rl_rec.is_exploration;
            market_params.rl_expected_q = rl_rec.expected_q;

            if self.cycle_count.is_multiple_of(100) {
                let summary = self.rl_agent.summary();
                tracing::info!(
                    episodes = summary.episodes,
                    states_visited = summary.states_visited,
                    avg_reward = %format!("{:.3}", summary.recent_avg_reward),
                    "Paper RL agent summary"
                );
            }
        }

        // Update strategy's fill model with current volatility
        let sigma = market_params.sigma.max(0.00001);
        let tau = self.quote_lifetime_s;
        self.strategy.update_fill_model_params(sigma, tau);

        // Use the REAL LadderStrategy to generate quotes
        let ladder = self.strategy.generate_ladder(
            &quote_config,
            self.inventory,
            self.max_position,
            self.target_liquidity,
            &market_params,
        );

        Some(ladder)
    }

    /// Build market params using ParameterAggregator — identical to live system.
    fn build_market_params(&mut self) -> MarketParams {
        // Update pre-fill classifier with latest flow and regime data (same as quote_engine.rs:380-399)
        {
            let flow_imb = self.estimator.flow_imbalance();
            let buy_volume = (1.0 + flow_imb) / 2.0;
            let sell_volume = 1.0 - buy_volume;
            self.pre_fill_classifier.update_trade_flow(buy_volume, sell_volume);

            let belief_snapshot = self.central_beliefs.snapshot();
            let hmm_confidence = belief_snapshot.regime.confidence;
            let changepoint_prob = belief_snapshot.changepoint.prob_5;
            self.pre_fill_classifier.update_regime(hmm_confidence, changepoint_prob);

            let funding_rate_8h = self.funding.current_rate();
            self.pre_fill_classifier.update_funding(funding_rate_8h);
        }

        // === Fix #2: Compute DriftAdjustedSkew from momentum/trend (replicate quote_engine.rs:324-376) ===
        let momentum_bps = self.estimator.momentum_bps();
        let p_continuation = self.estimator.momentum_continuation_probability();
        let position = self.inventory;

        self.hjb_controller.update_momentum_signals(
            momentum_bps,
            p_continuation,
            position,
            self.max_position,
        );

        let position_value = (position.abs() * self.mid_price).max(1.0);
        let trend_signal = self.estimator.trend_signal(position_value);

        let drift_adjusted_skew = self.hjb_controller.optimal_skew_with_trend(
            position,
            self.max_position,
            momentum_bps,
            p_continuation,
            &trend_signal,
        );

        let belief_snapshot = self.central_beliefs.snapshot();

        // === Fix #4: Forward regime and changepoint to central beliefs (replicate quote_engine.rs:700-711) ===
        let regime_probs = self.regime_hmm.regime_probabilities();
        self.central_beliefs.update(BeliefUpdate::RegimeUpdate {
            probs: regime_probs,
            features: None,
        });

        let sigma = self.estimator.sigma();
        if sigma > 0.0 {
            self.central_beliefs
                .update(BeliefUpdate::ChangepointObs { observation: sigma });
        }

        // Simulate margin state: assume enough capital for max_position at current leverage
        // Live system gets this from WebData2 clearinghouse_state; paper trader must synthesize it.
        {
            let sim_account_value = self.max_position_usd; // Account value ≈ max notional
            let position_notional = self.inventory.abs() * self.mid_price;
            let margin_used = position_notional / 3.0; // ~3x leverage
            self.margin_sizer.update_state(sim_account_value, margin_used, position_notional);
        }

        let sources = ParameterSources {
            estimator: &self.estimator,
            adverse_selection: &self.adverse_selection,
            depth_decay_as: &self.depth_decay_as,
            pre_fill_classifier: &self.pre_fill_classifier,
            liquidation_detector: &self.liquidation_detector,
            hawkes: &self.hawkes,
            funding: &self.funding,
            spread_tracker: &self.spread_tracker,
            hjb_controller: &self.hjb_controller,
            margin_sizer: &self.margin_sizer,
            stochastic_config: &self.stochastic_config,
            drift_adjusted_skew,
            adaptive_spreads: &self.adaptive_spreads,
            beliefs_builder: &self.beliefs_builder,
            beliefs: Some(&belief_snapshot),
            position: self.inventory,
            max_position: self.max_position,
            latest_mid: self.mid_price,
            risk_aversion: self.gamma,
            // Paper mode — no real exchange limits
            exchange_limits_valid: false,
            exchange_effective_bid_limit: f64::MAX,
            exchange_effective_ask_limit: f64::MAX,
            exchange_limits_age_ms: 0,
            pending_bid_exposure: 0.0,
            pending_ask_exposure: 0.0,
            dynamic_max_position_value: self.max_position_usd,
            dynamic_limit_valid: false,
            tick_size_bps: 10.0,
            near_touch_depth_usd: self.estimator.near_touch_depth_usd(),
            calibration_gamma_mult: self.calibration_controller.gamma_multiplier(),
            calibration_progress: self.calibration_controller.calibration_progress(),
            calibration_complete: self.calibration_controller.is_calibrated(),
            use_dynamic_kappa_floor: true,
            use_dynamic_spread_ceiling: true,
            learned_params: None,
        };
        let mut market_params = ParameterAggregator::build(&sources);

        // === Fix #1: Wire lead-lag signal into market_params (replicate quote_engine.rs:724-841) ===
        let signals = self.signal_integrator.get_signals();

        if signals.lead_lag_actionable {
            market_params.lead_lag_signal_bps = signals.combined_skew_bps;
            market_params.lead_lag_confidence = signals.model_confidence;
        }

        // Apply cross-venue spread multiplier and skew when valid
        if signals.cross_venue_valid {
            market_params.spread_widening_mult *= signals.cross_venue_spread_mult;

            let cv_skew_contribution = signals.cross_venue_skew
                * signals.cross_venue_confidence
                * 5.0;

            if signals.cross_venue_agreement > 0.5 {
                market_params.lead_lag_signal_bps += cv_skew_contribution;
            }

            let timestamp_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);

            self.central_beliefs.update(BeliefUpdate::CrossVenueUpdate {
                direction: signals.cross_venue_direction,
                confidence: signals.cross_venue_confidence,
                discovery_venue: signals.cross_venue_intensity_ratio,
                max_toxicity: signals.cross_venue_max_toxicity,
                avg_toxicity: signals.cross_venue_avg_toxicity,
                agreement: signals.cross_venue_agreement,
                divergence: signals.cross_venue_divergence,
                intensity_ratio: signals.cross_venue_intensity_ratio,
                imbalance_correlation: signals.cross_venue_imbalance_correlation,
                toxicity_alert: signals.cross_venue_max_toxicity > 0.7,
                divergence_alert: signals.cross_venue_divergence > 0.4,
                timestamp_ms,
            });
        }

        // Wire regime-conditioned kappa from SignalIntegration into MarketParams.
        // This allows the ladder strategy to blend regime kappa with robust/adaptive kappa.
        if signals.kappa_effective > 0.0 {
            market_params.regime_kappa = Some(signals.kappa_effective);
            market_params.regime_kappa_current_regime = signals.current_regime;
        }

        // === Fix #3: Position continuation decision (replicate quote_engine.rs:581-682) ===
        {
            let changepoint_prob = if belief_snapshot.changepoint.is_warmed_up {
                belief_snapshot.changepoint.prob_5
            } else {
                0.0
            };
            let changepoint_entropy = belief_snapshot.changepoint.entropy;
            let momentum_p = self.estimator.momentum_continuation_probability();
            let position_value = (position.abs() * self.mid_price).max(1.0);
            let trend_signal = self.estimator.trend_signal(position_value);
            let regime_probs = belief_snapshot.regime.probs;

            self.position_decision.update_signals(
                changepoint_prob,
                changepoint_entropy,
                momentum_p,
                trend_signal.timeframe_agreement,
                trend_signal.trend_confidence,
                regime_probs,
            );

            let belief_drift = market_params.belief_predictive_bias;
            let belief_confidence = market_params.belief_confidence;
            let edge_bps = market_params.current_edge_bps;

            let position_action = self.position_decision.decide(
                position,
                self.max_position,
                belief_drift,
                belief_confidence,
                edge_bps,
            );

            let raw_inventory_ratio = if self.max_position > 1e-9 {
                position / self.max_position
            } else {
                0.0
            };

            market_params.position_action = position_action;
            market_params.continuation_p = self.position_decision.prob_continuation();
            market_params.continuation_confidence = self.position_decision.confidence();
            market_params.effective_inventory_ratio =
                action_to_inventory_ratio(position_action, raw_inventory_ratio);
        }

        // === Fix #7: Apply ThresholdKappa spread multiplier (replicate quote_engine.rs:849-860) ===
        let threshold_kappa_mult = self.threshold_kappa.regime().spread_multiplier();
        if threshold_kappa_mult > 1.0 && self.threshold_kappa.is_warmed_up() {
            market_params.spread_widening_mult *= threshold_kappa_mult;
        }

        market_params
    }

    /// Log signal health periodically (every 60s)
    fn log_signal_health(&mut self) {
        if self.last_signal_log.elapsed() < Duration::from_secs(60) {
            return;
        }
        self.last_signal_log = Instant::now();

        let regime_probs = self.regime_hmm.regime_probabilities();
        let vpin = self.vpin.vpin();
        let _pre_fill_tox = self.pre_fill_classifier.summary();
        let hawkes_imb = self.hawkes.flow_imbalance();
        let spread_bps = self.spread_tracker.current_spread_bps();
        let liq_det = self.liquidation_detector.summary();

        let as_diag = self.pre_fill_classifier.learning_diagnostics();
        let cal_progress = self.calibration_controller.calibration_progress();
        let cal_gamma = self.calibration_controller.gamma_multiplier();

        info!(
            "[SIGNAL CHECK] vpin={:.3} regime=[{:.2},{:.2},{:.2},{:.2}] \
             hawkes_imb={:.3} spread_bps={:.1} \
             should_pull={} cascade_severity={:.2} tail_risk={:.2}",
            vpin,
            regime_probs[0], regime_probs[1], regime_probs[2], regime_probs[3],
            hawkes_imb, spread_bps,
            liq_det.should_pull_quotes,
            liq_det.cascade_severity,
            liq_det.tail_risk_multiplier,
        );
        info!(
            "[LEARNING] fills={} kappa_conf={:.3} as_samples={} as_learned={} \
             cal_progress={:.0}% cal_gamma={:.2} pending_as_outcomes={}",
            self.total_fill_count,
            self.estimator.kappa_confidence(),
            as_diag.samples,
            as_diag.is_using_learned,
            cal_progress * 100.0,
            cal_gamma,
            self.pending_fill_outcomes.len(),
        );

        // Warmup progression diagnostics
        let warmup = self.adaptive_spreads.warmup_progress();
        let uncertainty = self.adaptive_spreads.warmup_uncertainty_factor();
        let status = self.adaptive_spreads.status();
        info!(
            "[WARMUP] progress={:.0}% floor_obs={} kappa_fills={} floor_bps={:.1} uncertainty={:.3}",
            warmup * 100.0,
            status.floor_observations,
            status.kappa_own_fills,
            status.floor_bps,
            uncertainty,
        );
    }

    /// Update inventory from simulated fill and record for Bayesian learning
    fn on_simulated_fill(&mut self, fill: &SimulatedFill) {
        let direction = if fill.side == Side::Buy { 1.0 } else { -1.0 };
        self.inventory += direction * fill.fill_size;
        self.total_fill_count += 1;

        // Record fill for Bayesian learning in the strategy
        let depth_bps = if self.mid_price > 0.0 {
            ((fill.fill_price - self.mid_price).abs() / self.mid_price) * 10_000.0
        } else {
            5.0
        };
        self.strategy.record_fill_observation(depth_bps, true);

        // === FIX: Wire kappa learning from own fills ===
        let timestamp_ms = fill.timestamp_ns / 1_000_000;
        let is_buy = fill.side == Side::Buy;
        self.estimator.on_own_fill(
            timestamp_ms,
            fill.fill_price, // placement_price = fill_price for limit orders
            fill.fill_price,
            fill.fill_size,
            is_buy,
        );

        // === FIX: Wire calibration controller fill tracking ===
        self.calibration_controller.record_fill();

        // === FIX: Wire signal integrator fill tracking (kappa + model gating) ===
        self.signal_integrator.on_fill(
            timestamp_ms,
            fill.fill_price,
            fill.fill_size,
            self.mid_price,
        );

        // === FIX: Wire adverse selection estimator fill tracking ===
        // Without this, the AS estimator never sees fills and best_horizon_as_bps() returns 0.0 forever
        self.adverse_selection.record_fill(
            fill.oid,
            fill.fill_size,
            is_buy,
            self.mid_price,
        );

        // === FIX: Queue fill for adverse selection outcome tracking ===
        self.pending_fill_outcomes.push_back(PendingFillOutcome {
            timestamp_ns: fill.timestamp_ns,
            fill_price: fill.fill_price,
            is_buy,
            mid_at_fill: self.mid_price,
        });

        // === FIX: Wire adaptive spread learning from fills (ROOT CAUSE of stuck warmup) ===
        // Without this, adaptive_spreads never sees fills and warmup_progress() stays at ~10%.
        // Mirrors the live system's handlers.rs:456-484 logic exactly.
        if self.mid_price > 0.0 {
            let direction = if is_buy { 1.0 } else { -1.0 };
            let as_realized = (self.mid_price - fill.fill_price) * direction / fill.fill_price;
            let depth_from_mid = (fill.fill_price - self.mid_price).abs() / self.mid_price;
            let fill_pnl = -as_realized;
            self.adaptive_spreads.on_fill_simple(
                as_realized,
                depth_from_mid,
                fill_pnl,
                self.estimator.kappa(),
            );
        }

        // Per-fill spread capture: positive if we captured spread (bought below mid / sold above mid)
        let spread_capture = if is_buy {
            (self.mid_price - fill.fill_price) * fill.fill_size
        } else {
            (fill.fill_price - self.mid_price) * fill.fill_size
        };

        info!(
            "[FILL FEEDBACK] oid={} side={:?} price={:.4} size={:.4} depth_bps={:.1} total_fills={} kappa_conf={:.3}",
            fill.oid, fill.side, fill.fill_price, fill.fill_size, depth_bps,
            self.total_fill_count, self.estimator.kappa_confidence(),
        );
        info!(
            "[FILL PNL] side={:?} price={:.2} size={:.4} spread_capture={:.4} mid_at_fill={:.2} depth_bps={:.1} inventory_after={:.4}",
            fill.side, fill.fill_price, fill.fill_size, spread_capture,
            self.mid_price, depth_bps, self.inventory,
        );

        // === RL Agent: Update Q-values from simulated fill ===
        {
            let vol_ratio = self.estimator.sigma() / self.estimator.sigma_clean().max(0.0001);
            let inventory_risk = (self.inventory.abs() / self.max_position.max(1.0)).clamp(0.0, 1.0);

            // FIX P0-1: Reward uses spread capture (positive = good), not negated AS
            // spread_capture_bps = how far from mid we filled, converted to bps
            let notional = fill.fill_price * fill.fill_size;
            let spread_capture_bps = if notional > 0.0 {
                (spread_capture / notional) * 10_000.0
            } else {
                0.0
            };
            const MAKER_FEE_BPS: f64 = 1.5;
            let realized_edge_bps = spread_capture_bps - MAKER_FEE_BPS;
            let _was_adverse = realized_edge_bps < 0.0;

            // FIX P0-2: Use cached book_imbalance instead of hardcoded 0.0
            let fill_state = MDPState::from_continuous(
                self.inventory,
                self.max_position,
                self.last_book_imbalance,
                vol_ratio,
                self.adverse_selection.best_horizon_as_bps() / 10.0, // normalize to ~[0,1]
                self.hawkes.intensity_percentile(),
            );

            let reward = Reward::compute(
                &self.rl_agent.reward_config().clone(),
                realized_edge_bps,
                inventory_risk,
                vol_ratio,
                inventory_risk, // prev_inventory_risk approximated by current
            );

            // FIX P1-2: Drain ALL pending state-actions (handles clustered fills)
            while let Some((state_idx, action_idx)) = self.rl_agent.take_next_state_action() {
                self.rl_agent.update_idx(state_idx, action_idx, reward, fill_state.to_index(), false);

                // Log experience for offline RL training
                if let Some(ref mut logger) = self.experience_logger {
                    let regime_probs = self.regime_hmm.regime_probabilities();
                    let regime_label = if regime_probs[3] > 0.3 {
                        "Cascade"
                    } else if regime_probs[2] > 0.3 {
                        "Volatile"
                    } else if regime_probs[1] > 0.3 {
                        "Trending"
                    } else {
                        "Quiet"
                    };
                    let record = ExperienceRecord::from_params(ExperienceParams {
                        state: MDPState::from_index(state_idx),
                        action: MDPAction::from_index(action_idx),
                        reward,
                        next_state: fill_state,
                        done: false,
                        timestamp_ms,
                        session_id: self.session_id.clone(),
                        source: ExperienceSource::Paper,
                        side: if is_buy { "buy".to_string() } else { "sell".to_string() },
                        fill_price: fill.fill_price,
                        mid_price: self.mid_price,
                        fill_size: fill.fill_size,
                        inventory: self.inventory,
                        regime: regime_label.to_string(),
                    });
                    let _ = logger.log(&record);
                }

                tracing::debug!(
                    realized_edge_bps = %format!("{:.2}", realized_edge_bps),
                    inventory_risk = %format!("{:.3}", inventory_risk),
                    reward_total = %format!("{:.3}", reward.total),
                    book_imbalance = %format!("{:.3}", self.last_book_imbalance),
                    "Paper RL agent updated from simulated fill"
                );
            }
        }

        // === FIX P0-3: Update Learned Parameters from fill data ===
        // Mirrors event_loop.rs:690-738 — transfers AS classifications to Bayesian params
        if self.stochastic_config.use_learned_parameters {
            let (informed, uninformed) = self.adverse_selection.take_informed_counts();
            if informed > 0 || uninformed > 0 {
                self.learned_params.alpha_touch.observe_beta(informed, uninformed);
                tracing::debug!(
                    informed = informed,
                    uninformed = uninformed,
                    total_obs = self.learned_params.alpha_touch.n_observations,
                    alpha_touch = %format!("{:.4}", self.learned_params.alpha_touch.estimate()),
                    "Paper trader: updated learned alpha_touch from AS classifier"
                );
            }

            // Kappa update from fill rate (mirrors components.rs:739-748)
            let session_secs = self.session_start.elapsed().as_secs_f64();
            let avg_spread_bps = self.spread_tracker.current_spread_bps();
            let total_fills = self.total_fill_count as usize;
            if total_fills >= 10 && session_secs > 60.0 && avg_spread_bps > 0.0 {
                let fill_rate = total_fills as f64 / session_secs;
                let kappa_obs = fill_rate / (avg_spread_bps / 10_000.0);
                if kappa_obs > 100.0 && kappa_obs < 100_000.0 {
                    let exposure = session_secs * (avg_spread_bps / 10_000.0);
                    self.learned_params.kappa.observe_gamma_poisson(total_fills, exposure);
                }
            }
            self.learned_params.total_fills_observed = total_fills;
        }

        // === FIX P1-3: Wire Kelly tracker from fill outcomes ===
        // LadderStrategy.record_win/record_loss feeds WinLossTracker for Kelly sizing.
        // Without this, n_wins=0/n_losses=0 → odds_ratio always returns prior (1.5).
        {
            let notional = fill.fill_price * fill.fill_size;
            let edge_for_kelly = if notional > 0.0 {
                (spread_capture / notional) * 10_000.0 - 1.5 // MAKER_FEE_BPS
            } else {
                0.0
            };
            if edge_for_kelly > 0.0 {
                self.strategy.record_win(edge_for_kelly);
            } else if edge_for_kelly < 0.0 {
                self.strategy.record_loss(-edge_for_kelly);
            }
        }
    }

    /// Check pending fills for adverse selection outcomes (call after mid updates)
    fn check_adverse_selection_outcomes(&mut self) {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let outcome_delay_ns = 5_000_000_000u64; // 5 seconds
        let markout_s: f64 = 5.0; // Must match outcome_delay_ns / 1e9

        // Volatility-scaled threshold: 2σ × √τ prevents noise misclassification.
        // Random walk E[|ΔP|] = σ × √τ. At 2σ, only ~5% of noise moves are misclassified.
        let sigma_bps = self.estimator.sigma() * 10_000.0;
        let adverse_threshold_bps = (2.0 * sigma_bps * markout_s.sqrt()).max(1.0);

        while let Some(front) = self.pending_fill_outcomes.front() {
            if now_ns.saturating_sub(front.timestamp_ns) < outcome_delay_ns {
                break; // Not old enough yet
            }

            let pending = self.pending_fill_outcomes.pop_front().unwrap();
            if self.mid_price <= 0.0 || pending.mid_at_fill <= 0.0 {
                continue;
            }

            // Compute adverse selection:
            // Buy fill is adverse if mid dropped (we overpaid)
            // Sell fill is adverse if mid rose (we undersold)
            let mid_change_bps = ((self.mid_price - pending.mid_at_fill) / pending.mid_at_fill) * 10_000.0;
            let was_adverse = if pending.is_buy {
                mid_change_bps < -adverse_threshold_bps
            } else {
                mid_change_bps > adverse_threshold_bps
            };
            let magnitude_bps = mid_change_bps.abs();

            self.pre_fill_classifier.record_outcome(
                pending.is_buy, // is_bid = is_buy (our bid was filled)
                was_adverse,
                Some(magnitude_bps),
            );

            // === FIX: Wire model gating AS prediction feedback ===
            // This lets model_gating track how well the AS classifier predicts adverse fills
            let predicted_as_prob = self.pre_fill_classifier.cached_toxicity();
            self.signal_integrator.update_as_prediction(predicted_as_prob, was_adverse);

            if was_adverse {
                info!(
                    "[AS OUTCOME] side={} adverse=true magnitude={:.1}bps mid_at_fill={:.4} mid_now={:.4}",
                    if pending.is_buy { "Buy" } else { "Sell" },
                    magnitude_bps, pending.mid_at_fill, self.mid_price,
                );
            }
        }
    }

    /// Record cancelled (non-filled) order for Bayesian learning
    #[allow(dead_code)]
    fn on_order_cancelled(&mut self, order_price: f64) {
        if self.mid_price > 0.0 {
            let depth_bps = ((order_price - self.mid_price).abs() / self.mid_price) * 10_000.0;
            self.strategy.record_fill_observation(depth_bps, false);
        }
    }

    /// Handle Binance price update -> feed to lag analyzer
    fn on_binance_price(&mut self, mid: f64, timestamp_ms: i64) {
        self.signal_integrator.on_binance_price(mid, timestamp_ms);
    }

    /// Handle Binance trade update -> feed to cross-venue flow
    fn on_binance_trade(&mut self, trade: &BinanceTradeUpdate) {
        self.signal_integrator.on_binance_trade(trade);
    }

    /// Assemble checkpoint bundle from current state
    fn assemble_checkpoint_bundle(&self) -> CheckpointBundle {
        let (vol_filter, informed_flow, fill_rate, kappa_own, kappa_bid, kappa_ask, momentum) =
            self.estimator.to_checkpoint();

        CheckpointBundle {
            metadata: CheckpointMetadata {
                version: 1,
                timestamp_ms: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
                asset: self.asset.clone(),
                session_duration_s: self.session_start.elapsed().as_secs_f64(),
            },
            learned_params: self.learned_params.clone(),
            pre_fill: self.pre_fill_classifier.to_checkpoint(),
            enhanced: self.enhanced_classifier.to_checkpoint(),
            vol_filter,
            regime_hmm: self.regime_hmm.to_checkpoint(),
            informed_flow,
            fill_rate,
            kappa_own,
            kappa_bid,
            kappa_ask,
            momentum,
            kelly_tracker: {
                let wlt = &self.strategy.kelly_sizer.win_loss_tracker;
                KellyTrackerCheckpoint {
                    ewma_wins: wlt.avg_win(),
                    n_wins: (wlt.total_trades() as f64 * wlt.win_rate()) as u64,
                    ewma_losses: wlt.avg_loss(),
                    n_losses: (wlt.total_trades() as f64 * (1.0 - wlt.win_rate())) as u64,
                    decay: 0.99, // matches WinLossTracker default
                }
            },
            ensemble_weights: EnsembleWeightsCheckpoint::default(),
            rl_q_table: self.rl_agent.to_checkpoint(),
            kill_switch: KillSwitchCheckpoint::default(),
        }
    }

    /// Restore learned state from checkpoint bundle
    fn restore_from_bundle(&mut self, bundle: &CheckpointBundle) {
        self.pre_fill_classifier
            .restore_checkpoint(&bundle.pre_fill);
        self.enhanced_classifier
            .restore_checkpoint(&bundle.enhanced);
        self.estimator.restore_checkpoint(bundle);
        self.regime_hmm.restore_checkpoint(&bundle.regime_hmm);
        self.learned_params = bundle.learned_params.clone();
        self.rl_agent.restore_from_checkpoint(&bundle.rl_q_table);

        // Restore Kelly tracker state from checkpoint
        let kt = &bundle.kelly_tracker;
        self.strategy.kelly_sizer.win_loss_tracker.restore_from_checkpoint(
            kt.ewma_wins,
            kt.n_wins,
            kt.ewma_losses,
            kt.n_losses,
            kt.decay,
        );
    }
}

// ============================================================================
// Simulation Statistics
// ============================================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct SimulationStats {
    /// Total quote cycles
    quote_cycles: u64,
    /// Total quotes generated (bid + ask levels)
    quotes_generated: u64,
    /// Simulated fills
    simulated_fills: u64,
    /// Total simulated volume
    simulated_volume: f64,
    /// Estimated spread capture
    estimated_spread_capture: f64,
    /// Estimated adverse selection
    estimated_adverse_selection: f64,
    /// Runtime in seconds
    runtime_secs: f64,

    /// Quotes per side
    bid_quotes: u64,
    ask_quotes: u64,

    /// Average spread in bps
    avg_spread_bps: f64,
    /// Max spread seen
    max_spread_bps: f64,
    /// Min spread seen
    min_spread_bps: f64,
}

impl SimulationStats {
    fn format(&self) -> String {
        let mut output = String::new();

        output.push('\n');
        output.push_str("╔══════════════════════════════════════════════════════════════╗\n");
        output.push_str("║           PAPER TRADING SIMULATION SUMMARY                   ║\n");
        output.push_str("╠══════════════════════════════════════════════════════════════╣\n");

        output.push_str(&format!(
            "║ Runtime:              {:>10.1} seconds                     ║\n",
            self.runtime_secs
        ));
        output.push_str(&format!(
            "║ Quote Cycles:         {:>10}                              ║\n",
            self.quote_cycles
        ));
        output.push_str(&format!(
            "║ Quotes/Second:        {:>10.1}                              ║\n",
            if self.runtime_secs > 0.0 {
                self.quote_cycles as f64 / self.runtime_secs
            } else {
                0.0
            }
        ));

        output.push_str("╠══════════════════════════════════════════════════════════════╣\n");

        output.push_str(&format!(
            "║ Simulated Fills:      {:>10}                              ║\n",
            self.simulated_fills
        ));
        output.push_str(&format!(
            "║ Simulated Volume:     {:>10.4}                              ║\n",
            self.simulated_volume
        ));

        output.push_str("╠══════════════════════════════════════════════════════════════╣\n");

        output.push_str(&format!(
            "║ Avg Spread (bps):     {:>10.2}                              ║\n",
            self.avg_spread_bps
        ));
        output.push_str(&format!(
            "║ Min Spread (bps):     {:>10.2}                              ║\n",
            self.min_spread_bps
        ));
        output.push_str(&format!(
            "║ Max Spread (bps):     {:>10.2}                              ║\n",
            self.max_spread_bps
        ));

        output.push_str("╠══════════════════════════════════════════════════════════════╣\n");

        output.push_str(&format!(
            "║ Est. Spread Capture:  ${:>9.2}                              ║\n",
            self.estimated_spread_capture
        ));
        output.push_str(&format!(
            "║ Est. Adverse Select:  ${:>9.2}                              ║\n",
            self.estimated_adverse_selection
        ));
        output.push_str(&format!(
            "║ Est. Net PnL:         ${:>9.2}                              ║\n",
            self.estimated_spread_capture + self.estimated_adverse_selection
        ));

        output.push_str("╚══════════════════════════════════════════════════════════════╝\n");

        output
    }
}

// ============================================================================
// Market Data Types
// ============================================================================

#[derive(Debug, Clone)]
enum MarketDataMessage {
    Mid(f64),
    Book {
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
    },
    Trade {
        price: f64,
        size: f64,
        is_buy: bool,
    },
}

/// Helper to create fill simulator trade
fn create_market_trade(
    price: f64,
    size: f64,
    is_buy: bool,
) -> hyperliquid_rust_sdk::market_maker::simulation::fill_sim::MarketTrade {
    hyperliquid_rust_sdk::market_maker::simulation::fill_sim::MarketTrade {
        timestamp_ns: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64,
        price,
        size,
        side: if is_buy { Side::Buy } else { Side::Sell },
    }
}

// ============================================================================
// Risk Aggregator for Paper Trading
// ============================================================================

fn build_paper_risk_aggregator() -> RiskAggregator {
    // Paper mode: relaxed limits to survive adverse periods and enable learning.
    // Live mode keeps conservative limits via KillSwitchConfig::from_account_kelly().
    let max_daily_loss = 2000.0; // USD (was 500 — 5% drawdown killed at 218s)
    let max_drawdown_frac = 0.15; // 15% (was 5% — too strict for learning)
    let stale_data_threshold = Duration::from_secs(30);
    let cascade_pull = 0.8;
    let cascade_kill = 2.0; // (was 0.95 — paper mode should survive cascades)
    let max_rate_limit_errors = 3;

    RiskAggregator::new()
        .with_monitor(Box::new(LossMonitor::new(max_daily_loss)))
        .with_monitor(Box::new(DrawdownMonitor::new(max_drawdown_frac)))
        .with_monitor(Box::new(PositionMonitor::new()))
        .with_monitor(Box::new(DataStalenessMonitor::new(stale_data_threshold)))
        .with_monitor(Box::new(CascadeMonitor::new(cascade_pull, cascade_kill)))
        .with_monitor(Box::new(RateLimitMonitor::new(max_rate_limit_errors)))
}

// ============================================================================
// Main Simulation Loop
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Initialize logging using env_logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(&cli.log_level))
        .init();

    // Create output directory
    std::fs::create_dir_all(&cli.output_dir)?;

    info!(
        "Starting paper trading simulation: asset={}, duration={}s, network={}",
        cli.asset, cli.duration, cli.network
    );

    if cli.paper_mode {
        warn!("Paper trading calibration mode active: tighter spreads, faster warmup");
        warn!("Results are NOT representative of production quoting");
    }

    // Determine base URL
    let base_url = match cli.network.as_str() {
        "mainnet" => BaseUrl::Mainnet,
        "testnet" => BaseUrl::Testnet,
        _ => {
            error!("Invalid network: {}", cli.network);
            return Err(format!("Invalid network: {}", cli.network).into());
        }
    };

    // Create simulation components
    let executor = Arc::new(SimulationExecutor::new(cli.verbose));
    // Paper mode: realistic fill simulation for meaningful learning.
    // 50% touch probability models back-of-book queue position realistically
    // (our quotes are 15 bps from mid when top-of-book is 1-3 bps).
    // Default mode uses conservative probabilities (30% touch, 50% queue).
    let fill_config = if cli.paper_mode {
        FillSimulatorConfig {
            touch_fill_probability: 0.5,   // Realistic fill rate (was 0.9)
            queue_position_factor: 0.6,    // Back-of-book position (was 0.9)
            ignore_book_depth: true,       // Sim orders aren't in real book
            ..Default::default()
        }
    } else {
        FillSimulatorConfig {
            touch_fill_probability: 0.3,
            queue_position_factor: 0.5,
            ignore_book_depth: true,
            ..Default::default()
        }
    };
    let mut fill_simulator = FillSimulator::new(executor.clone(), fill_config);

    // Create prediction logger
    let prediction_log_path = PathBuf::from(&cli.output_dir).join("predictions.jsonl");
    let mut prediction_logger = PredictionLogger::new(prediction_log_path)?;

    // Create calibration analyzer
    let mut calibration_analyzer = CalibrationAnalyzer::new(20);

    // Create outcome tracker
    let mut outcome_tracker = OutcomeTracker::new(0.00015); // 1.5 bps maker fee

    // Analytics trackers for Sharpe, attribution, and edge validation
    let mut sharpe_tracker = SharpeTracker::new();
    let mut signal_sharpe = PerSignalSharpeTracker::new();
    let mut signal_attributor = SignalPnLAttributor::new();
    let mut edge_tracker = EdgeTracker::new();
    let mut analytics_logger = match AnalyticsLogger::new(&cli.output_dir) {
        Ok(logger) => Some(logger),
        Err(e) => {
            warn!("Failed to initialize analytics logger: {}", e);
            None
        }
    };
    let mut last_cycle_contributions: Option<CycleContributions> = None;

    // Calibration pipeline state: track current cycle and its fills
    let mut current_cycle_id: u64 = 0;
    let mut cycle_fills: Vec<(usize, SimulatedFill)> = Vec::new();
    let mut last_calibration_feed = Instant::now();

    // Initialize simulation state
    let mut state = SimulationState::new(SimulationConfig {
        asset: cli.asset.clone(),
        gamma: cli.gamma,
        target_spread_bps: cli.target_spread_bps,
        ladder_levels: cli.ladder_levels,
        target_liquidity: cli.target_liquidity,
        quote_lifetime_s: cli.quote_lifetime_s,
        min_notional: cli.min_notional,
        paper_mode: cli.paper_mode,
        max_position_usd: cli.max_position_usd,
    });

    // Restore from checkpoint if enabled
    if !cli.disable_checkpoint {
        let dir = cli
            .checkpoint_dir
            .clone()
            .unwrap_or_else(|| format!("data/checkpoints/paper/{}", cli.asset));
        let dir = PathBuf::from(dir);

        match CheckpointManager::new(dir) {
            Ok(mgr) => {
                match mgr.load_latest() {
                    Ok(Some(bundle)) => {
                        if bundle.metadata.asset != cli.asset {
                            warn!(
                                checkpoint_asset = %bundle.metadata.asset,
                                our_asset = %cli.asset,
                                "Checkpoint asset mismatch, starting fresh"
                            );
                        } else {
                            let now_ms = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .map(|d| d.as_millis() as u64)
                                .unwrap_or(0);
                            let checkpoint_age_hours = (now_ms.saturating_sub(bundle.metadata.timestamp_ms)) as f64 / (3600.0 * 1000.0);
                            if !cli.force_restore && checkpoint_age_hours > cli.max_checkpoint_age_hours {
                                warn!(
                                    "Checkpoint is {:.1} hours old (max: {}h), skipping restore (use --force-restore to override)",
                                    checkpoint_age_hours, cli.max_checkpoint_age_hours
                                );
                            } else if cli.force_restore && checkpoint_age_hours > cli.max_checkpoint_age_hours {
                                warn!(
                                    "Force-restoring checkpoint despite being {:.1} hours old (max: {}h)",
                                    checkpoint_age_hours, cli.max_checkpoint_age_hours
                                );
                                state.restore_from_bundle(&bundle);
                                info!(
                                    asset = %bundle.metadata.asset,
                                    samples = bundle.pre_fill.learning_samples,
                                    session_s = bundle.metadata.session_duration_s,
                                    age_hours = checkpoint_age_hours,
                                    "Force-restored from old checkpoint"
                                );
                            } else {
                                state.restore_from_bundle(&bundle);
                                info!(
                                    asset = %bundle.metadata.asset,
                                    samples = bundle.pre_fill.learning_samples,
                                    session_s = bundle.metadata.session_duration_s,
                                    age_hours = checkpoint_age_hours,
                                    "Restored from checkpoint"
                                );
                            }
                        }
                    }
                    Ok(None) => info!("No checkpoint found, starting fresh"),
                    Err(e) => warn!("Failed to load checkpoint: {e}, starting fresh"),
                }
                if let Err(e) = mgr.cleanup_old(7) {
                    warn!("Checkpoint cleanup failed: {e}");
                }
                state.checkpoint_manager = Some(mgr);
            }
            Err(e) => warn!("Checkpoint init failed: {e}"),
        }
    }

    let mut stats = SimulationStats::default();
    let mut spread_samples: Vec<f64> = Vec::new();

    // Create message channel
    let (tx, mut rx) = mpsc::channel::<MarketDataMessage>(1000);

    // Spawn WebSocket subscription tasks
    let asset = cli.asset.clone();
    let tx_mids = tx.clone();
    tokio::spawn(async move {
        if let Err(e) = subscribe_all_mids(base_url, &asset, tx_mids).await {
            error!("AllMids subscription error: {}", e);
        }
    });

    let asset = cli.asset.clone();
    let tx_book = tx.clone();
    tokio::spawn(async move {
        if let Err(e) = subscribe_l2_book(base_url, &asset, tx_book).await {
            error!("L2Book subscription error: {}", e);
        }
    });

    let asset = cli.asset.clone();
    let tx_trades = tx.clone();
    tokio::spawn(async move {
        if let Err(e) = subscribe_trades(base_url, &asset, tx_trades).await {
            error!("Trades subscription error: {}", e);
        }
    });

    // Spawn Binance feed for cross-exchange lead-lag signal.
    // Auto-derive Binance symbol from asset; disable for HL-native tokens.
    let mut binance_rx: Option<tokio::sync::mpsc::Receiver<BinanceUpdate>> =
        if !cli.disable_binance_feed {
            let binance_symbol = resolve_binance_symbol(
                &cli.asset,
                cli.binance_symbol.as_deref(),
            );
            if let Some(ref sym) = binance_symbol {
                let (btx, brx) = tokio::sync::mpsc::channel(1000);
                let feed = BinanceFeed::for_symbol(sym, btx);
                tokio::spawn(async move {
                    feed.run().await;
                    warn!("Binance feed task terminated");
                });
                info!(
                    asset = %cli.asset,
                    binance_symbol = %sym,
                    "Binance lead-lag feed active (auto-derived from asset)"
                );
                Some(brx)
            } else {
                warn!(
                    asset = %cli.asset,
                    "No Binance equivalent for asset — cross-venue signal disabled. \
                     Use --binance-symbol to override if a correlated pair exists."
                );
                None
            }
        } else {
            warn!("Binance feed DISABLED - no cross-exchange signal");
            None
        };

    // Run simulation loop
    let start_time = Instant::now();
    let duration = Duration::from_secs(cli.duration);
    let shutdown = Arc::new(AtomicBool::new(false));

    // Risk aggregator for paper trading
    let risk_aggregator = build_paper_risk_aggregator();
    let mut peak_pnl: f64 = 0.0;
    let mut last_risk_log = Instant::now();

    // PnL summary and fill balance tracking (30s periodic logging)
    let mut last_pnl_summary = Instant::now();
    let mut summary_fill_count: u64 = 0;
    let mut summary_spread_pnl: f64 = 0.0;
    let mut summary_fee_pnl: f64 = 0.0;
    let mut summary_wins: u64 = 0;
    let mut buy_fill_count: u64 = 0;
    let mut sell_fill_count: u64 = 0;
    let mut buy_depth_sum_bps: f64 = 0.0;
    let mut sell_depth_sum_bps: f64 = 0.0;

    // Ctrl+C handler
    let shutdown_clone = shutdown.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        info!("Shutdown signal received");
        shutdown_clone.store(true, Ordering::SeqCst);
    });

    // Dashboard setup
    let dashboard_state: Arc<RwLock<DashboardState>> =
        Arc::new(RwLock::new(DashboardState::default()));

    // WebSocket dashboard state (None if dashboard disabled)
    let dashboard_ws_state: Option<Arc<DashboardWsState>> = if cli.dashboard {
        Some(Arc::new(
            DashboardWsState::new(DashboardWsConfig::default()),
        ))
    } else {
        None
    };

    // Throttle for WebSocket updates (100ms)
    let mut ws_throttle = DashboardThrottle::new(100);

    if cli.dashboard {
        let dashboard_state_for_server = dashboard_state.clone();
        let ws_state_for_server = dashboard_ws_state.clone().unwrap();
        let metrics_port = cli.metrics_port;

        tokio::spawn(async move {
            // CORS layer to allow dashboard at localhost:3000 to fetch from localhost:8080
            let cors = CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any);

            let app = Router::new()
                // HTTP endpoint (kept for debugging and fallback)
                .route(
                    "/api/dashboard",
                    get({
                        let state = dashboard_state_for_server.clone();
                        move || {
                            let state = state.clone();
                            async move {
                                let state_guard = state.read().unwrap();
                                Json(state_guard.clone())
                            }
                        }
                    }),
                )
                // WebSocket endpoint
                .route("/ws/dashboard", get(ws_handler))
                .with_state(ws_state_for_server)
                .layer(cors);

            let addr = format!("127.0.0.1:{metrics_port}");
            info!(
                port = metrics_port,
                "Dashboard server starting: HTTP at /api/dashboard, WS at /ws/dashboard"
            );

            if let Ok(listener) = tokio::net::TcpListener::bind(&addr).await {
                if let Err(e) = axum::serve(listener, app).await {
                    warn!(error = %e, "Dashboard server error");
                }
            } else {
                warn!(port = metrics_port, "Failed to bind dashboard port");
            }
        });

        info!("Dashboard enabled at http://localhost:3000/mm-dashboard-fixed.html");
        info!(
            "HTTP: http://localhost:{}/api/dashboard  WS: ws://localhost:{}/ws/dashboard",
            cli.metrics_port, cli.metrics_port
        );
    }

    info!("Simulation running for {} seconds...", cli.duration);

    let mut last_quote_time = Instant::now();
    let quote_interval = Duration::from_millis(cli.quote_interval_ms);

    // Dashboard accumulators - persist across loop iterations
    let mut accumulated_fills: Vec<FillRecord> = Vec::new();
    let mut regime_history: Vec<RegimeSnapshot> = Vec::new();
    let mut last_regime_snapshot = Instant::now();
    let mut cumulative_pnl: f64 = 0.0;

    // New visualization buffers
    let mut book_history: VecDeque<BookSnapshot> = VecDeque::with_capacity(360); // 5 min at 1/sec
    let mut price_history: VecDeque<PricePoint> = VecDeque::with_capacity(1800); // 30 min at 1/sec
    let mut quote_history: VecDeque<QuoteSnapshot> = VecDeque::with_capacity(360); // 30 min at 5/sec
    let mut spread_counts: HashMap<String, u32> = HashMap::new();
    let fill_stats: HashMap<(String, usize), (u32, f64)> = HashMap::new(); // (side, level) -> (count, size)
    let mut last_price_snapshot = Instant::now();
    let mut last_book_snapshot = Instant::now();
    let mut last_quote_snapshot = Instant::now();

    // Watchdog: track when we last received market data
    let mut last_data_received = Instant::now();
    let stale_data_threshold = Duration::from_secs(30);
    let mut stale_warning_logged = false;
    let mut data_stale = false;

    while !shutdown.load(Ordering::SeqCst) && start_time.elapsed() < duration {
        tokio::select! {
            Some(msg) = rx.recv() => {
                // Reset watchdog on any message
                last_data_received = Instant::now();
                if stale_warning_logged || data_stale {
                    info!("Market data stream recovered");
                    stale_warning_logged = false;
                    data_stale = false;
                }

                match msg {
                    MarketDataMessage::Mid(mid) => {
                        state.update_mid(mid);
                        executor.update_mid(mid);

                        // Capture price history (every 1 second)
                        if last_price_snapshot.elapsed() >= Duration::from_secs(1) {
                            last_price_snapshot = Instant::now();
                            let now = chrono::Utc::now();
                            if price_history.len() >= 1800 {
                                price_history.pop_front();
                            }
                            price_history.push_back(PricePoint {
                                time: now.format("%H:%M:%S").to_string(),
                                timestamp_ms: now.timestamp_millis(),
                                price: mid,
                            });
                        }
                    }
                    MarketDataMessage::Book { bids, asks } => {
                        state.update_book(bids.clone(), asks.clone());
                        fill_simulator.update_book(&bids, &asks);

                        // Capture book history (every 1 second)
                        if last_book_snapshot.elapsed() >= Duration::from_secs(1) {
                            last_book_snapshot = Instant::now();
                            let now = chrono::Utc::now();
                            if book_history.len() >= 360 {
                                book_history.pop_front();
                            }
                            book_history.push_back(BookSnapshot {
                                time: now.format("%H:%M:%S").to_string(),
                                timestamp_ms: now.timestamp_millis(),
                                bids: bids.iter().take(10).map(|(p, s)| BookLevel { price: *p, size: *s }).collect(),
                                asks: asks.iter().take(10).map(|(p, s)| BookLevel { price: *p, size: *s }).collect(),
                            });
                        }
                    }
                    MarketDataMessage::Trade { price, size, is_buy } => {
                        state.on_trade(price, size, is_buy);

                        // Check for simulated fills
                        let fills = fill_simulator.on_trade(&create_market_trade(price, size, is_buy));
                        for fill in fills {
                            stats.simulated_fills += 1;
                            stats.simulated_volume += fill.fill_size;

                            // Update inventory
                            state.on_simulated_fill(&fill);

                            // Track for outcome analysis
                            outcome_tracker.on_fill(fill.clone(), state.mid_price);

                            // Track fill for calibration pipeline (level_index resolved at cycle end)
                            cycle_fills.push((0, fill.clone()));

                            info!(
                                "[SIM FILL] side={:?} price={:.2} size={:.4} inventory={:.4}",
                                fill.side, fill.fill_price, fill.fill_size, state.inventory
                            );

                            // Accumulate fills for dashboard
                            let now = chrono::Utc::now();
                            let fill_pnl = if fill.side == Side::Buy {
                                (state.mid_price - fill.fill_price) * fill.fill_size
                            } else {
                                (fill.fill_price - state.mid_price) * fill.fill_size
                            };
                            cumulative_pnl += fill_pnl;

                            // Update PnL summary and fill balance counters
                            let fill_depth_bps = if state.mid_price > 0.0 {
                                ((fill.fill_price - state.mid_price).abs() / state.mid_price) * 10_000.0
                            } else {
                                0.0
                            };
                            let fill_fee = fill.fill_price * fill.fill_size * 0.000_15; // 1.5 bps maker fee
                            summary_fill_count += 1;
                            summary_spread_pnl += fill_pnl;
                            summary_fee_pnl += fill_fee;
                            if fill_pnl > 0.0 {
                                summary_wins += 1;
                            }
                            if fill.side == Side::Buy {
                                buy_fill_count += 1;
                                buy_depth_sum_bps += fill_depth_bps;
                            } else {
                                sell_fill_count += 1;
                                sell_depth_sum_bps += fill_depth_bps;
                            }

                            // Analytics: update Sharpe and attribution trackers
                            let notional_value = fill.fill_price * fill.fill_size;
                            let fill_pnl_bps = if notional_value > 0.0 {
                                (fill_pnl / notional_value) * 10_000.0
                            } else {
                                0.0
                            };
                            let fill_timestamp_ns = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap()
                                .as_nanos() as u64;

                            sharpe_tracker.add_return(fill_pnl_bps, fill_timestamp_ns);

                            if let Some(ref contribs) = last_cycle_contributions {
                                for contrib in &contribs.contributions {
                                    if contrib.was_active {
                                        signal_sharpe.add_signal_return(&contrib.signal_name, fill_pnl_bps, fill_timestamp_ns);
                                    }
                                }
                                signal_attributor.record_cycle(contribs, fill_pnl_bps);
                            }

                            // Analytics: track predicted vs realized edge
                            let predicted_as = state.adverse_selection.best_horizon_as_bps();
                            let edge_snap = EdgeSnapshot {
                                timestamp_ns: fill_timestamp_ns,
                                predicted_spread_bps: fill_depth_bps * 2.0,
                                realized_spread_bps: fill_depth_bps * 2.0,
                                predicted_as_bps: predicted_as,
                                realized_as_bps: fill_depth_bps.min(predicted_as),
                                fee_bps: 1.5,
                                predicted_edge_bps: fill_depth_bps * 2.0 - predicted_as - 1.5,
                                realized_edge_bps: fill_pnl_bps - 1.5,
                            };
                            edge_tracker.add_snapshot(edge_snap.clone());

                            if let Some(ref mut logger) = analytics_logger {
                                let _ = logger.log_edge(&edge_snap);
                            }

                            let fill_record = FillRecord {
                                time: now.format("%H:%M:%S").to_string(),
                                timestamp_ms: chrono::Utc::now().timestamp_millis(),
                                pnl: fill_pnl,
                                cum_pnl: cumulative_pnl,
                                side: if fill.side == Side::Buy { "BID".to_string() } else { "ASK".to_string() },
                                adverse_selection: format!("{predicted_as:.2}"),
                            };

                            // Keep last 100 fills
                            if accumulated_fills.len() >= 100 {
                                accumulated_fills.remove(0);
                            }
                            accumulated_fills.push(fill_record.clone());

                            // Push fill to WebSocket clients (immediate)
                            if let Some(ref ws_state) = dashboard_ws_state {
                                ws_state.push_fill(fill_record);
                            }
                        }
                    }
                }
            }
            Some(update) = async {
                match binance_rx.as_mut() {
                    Some(rx) => rx.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                match update {
                    BinanceUpdate::Price(p) => state.on_binance_price(p.mid_price, p.timestamp_ms),
                    BinanceUpdate::Trade(t) => state.on_binance_trade(&t),
                }
                last_data_received = Instant::now();
            }
            _ = tokio::time::sleep(Duration::from_millis(10)) => {
                // Watchdog: check for stale market data
                if last_data_received.elapsed() > stale_data_threshold && !data_stale {
                    warn!(
                        "No market data received for {:?}. Cancelling all orders and pausing quotes.",
                        last_data_received.elapsed()
                    );
                    stale_warning_logged = true;
                    data_stale = true;
                    // Cancel all active orders defensively
                    let active = executor.get_active_orders();
                    for order in &active {
                        executor.cancel_order(&cli.asset, order.oid).await;
                    }
                }

                // Generate quotes on interval (skip if data is stale)
                if data_stale {
                    // Skip quoting entirely while data is stale
                } else if last_quote_time.elapsed() >= quote_interval {
                    last_quote_time = Instant::now();

                    // Periodic signal health check
                    state.log_signal_health();

                    // Risk evaluation before quote generation
                    peak_pnl = peak_pnl.max(cumulative_pnl);
                    let risk_state = RiskState::new(
                        cumulative_pnl,
                        peak_pnl,
                        state.inventory,
                        state.max_position,
                        state.mid_price,
                        state.max_position_usd, // USD-based position limit (asset-agnostic)
                        10_000.0, // paper account value
                        state.estimator.sigma_clean(),
                        state.liquidation_detector.cascade_severity(),
                        last_data_received,
                    );
                    let risk = risk_aggregator.evaluate(&risk_state);

                    if risk.should_kill() {
                        warn!("[RISK] Kill switch triggered: {:?}", risk.kill_reasons);
                        shutdown.store(true, Ordering::SeqCst);
                        break;
                    }
                    if risk.pull_quotes {
                        warn!("[RISK] Pulling all quotes: {}", risk.summary());
                        continue;
                    }
                    if last_risk_log.elapsed() > Duration::from_secs(10) || risk.max_severity >= RiskSeverity::High {
                        info!("[RISK] {}", risk.summary());
                        last_risk_log = Instant::now();
                    }

                    // 30-second PnL summary and fill balance logging
                    if last_pnl_summary.elapsed() >= Duration::from_secs(30) {
                        let win_rate_pct = if summary_fill_count > 0 {
                            (summary_wins as f64 / summary_fill_count as f64) * 100.0
                        } else {
                            0.0
                        };
                        let net_pnl = summary_spread_pnl - summary_fee_pnl;
                        info!(
                            "[PNL SUMMARY] fills={} spread={:+.2} fee=-{:.2} net={:+.2} cum={:+.2} win_rate={:.0}%",
                            summary_fill_count, summary_spread_pnl, summary_fee_pnl, net_pnl, cumulative_pnl, win_rate_pct,
                        );

                        let total_fills = buy_fill_count + sell_fill_count;
                        let imbalance_pct = if total_fills > 0 {
                            let max_side = buy_fill_count.max(sell_fill_count) as f64;
                            (max_side / total_fills as f64) * 100.0
                        } else {
                            50.0
                        };
                        let avg_buy_depth = if buy_fill_count > 0 { buy_depth_sum_bps / buy_fill_count as f64 } else { 0.0 };
                        let avg_sell_depth = if sell_fill_count > 0 { sell_depth_sum_bps / sell_fill_count as f64 } else { 0.0 };
                        if imbalance_pct > 70.0 && total_fills > 5 {
                            warn!(
                                "[FILL BALANCE] total={} buys={} sells={} imbalance={:.0}% avg_buy_depth={:.1}bps avg_sell_depth={:.1}bps (WARNING: >70% imbalance)",
                                total_fills, buy_fill_count, sell_fill_count, imbalance_pct, avg_buy_depth, avg_sell_depth,
                            );
                        } else {
                            info!(
                                "[FILL BALANCE] total={} buys={} sells={} imbalance={:.0}% avg_buy_depth={:.1}bps avg_sell_depth={:.1}bps",
                                total_fills, buy_fill_count, sell_fill_count, imbalance_pct, avg_buy_depth, avg_sell_depth,
                            );
                        }

                        // Analytics summary (every 30s alongside PnL)
                        let sharpe_summary = sharpe_tracker.summary();
                        info!(
                            "[ANALYTICS] Sharpe(1h)={:.2} Sharpe(24h)={:.2} Sharpe(all)={:.2} Fills={} Edge={:.1}bps",
                            sharpe_summary.sharpe_1h,
                            sharpe_summary.sharpe_24h,
                            sharpe_summary.sharpe_all,
                            sharpe_summary.count,
                            edge_tracker.mean_realized_edge(),
                        );

                        if !signal_attributor.signal_names().is_empty() {
                            let signal_parts: Vec<String> = signal_attributor.signal_names()
                                .iter()
                                .map(|name| format!("{}={:.1}bps", name, signal_attributor.marginal_value(name)))
                                .collect();
                            info!("[ANALYTICS] Signal marginal: {}", signal_parts.join(" "));
                        }

                        if let Some(ref mut logger) = analytics_logger {
                            let _ = logger.log_sharpe(&sharpe_summary);
                            let _ = logger.log_signal_pnl(&signal_attributor);
                            let _ = logger.flush();
                        }

                        // Reset period counters
                        last_pnl_summary = Instant::now();
                        summary_fill_count = 0;
                        summary_spread_pnl = 0.0;
                        summary_fee_pnl = 0.0;
                        summary_wins = 0;
                    }

                    if let Some(mut ladder) = state.generate_quotes() {
                        // --- Risk-based quote gating ---

                        // Reduce-only mode: only quote the side that reduces inventory
                        if risk.reduce_only {
                            if state.inventory > 0.0 {
                                // Long: only keep asks (selling reduces position)
                                ladder.bids.clear();
                                info!("[RISK] Reduce-only: clearing bids (long inventory {:.4})", state.inventory);
                            } else if state.inventory < 0.0 {
                                // Short: only keep bids (buying reduces position)
                                ladder.asks.clear();
                                info!("[RISK] Reduce-only: clearing asks (short inventory {:.4})", state.inventory);
                            } else {
                                // Flat: skip quoting entirely
                                info!("[RISK] Reduce-only: flat inventory, skipping quotes");
                                continue;
                            }
                        }

                        // Side-specific pulls
                        if risk.pull_buys {
                            ladder.bids.clear();
                            info!("[RISK] Pulling buy-side quotes");
                        }
                        if risk.pull_sells {
                            ladder.asks.clear();
                            info!("[RISK] Pulling sell-side quotes");
                        }

                        // Spread widening: push prices away from mid
                        if risk.spread_factor > 1.0 {
                            let mid = state.mid_price;
                            let decimals = state.decimals;
                            for level in ladder.bids.iter_mut() {
                                level.price = round_to_significant_and_decimal(
                                    mid - (mid - level.price) * risk.spread_factor,
                                    5,
                                    decimals,
                                );
                            }
                            for level in ladder.asks.iter_mut() {
                                level.price = round_to_significant_and_decimal(
                                    mid + (level.price - mid) * risk.spread_factor,
                                    5,
                                    decimals,
                                );
                            }
                            info!("[RISK] Spread widened by factor {:.2}", risk.spread_factor);
                        }

                        // Skew: shift all prices by skew offset
                        if risk.skew_bps != 0.0 {
                            let skew_offset = state.mid_price * risk.skew_bps / 10_000.0;
                            let decimals = state.decimals;
                            for level in ladder.bids.iter_mut() {
                                level.price = round_to_significant_and_decimal(
                                    level.price + skew_offset,
                                    5,
                                    decimals,
                                );
                            }
                            for level in ladder.asks.iter_mut() {
                                level.price = round_to_significant_and_decimal(
                                    level.price + skew_offset,
                                    5,
                                    decimals,
                                );
                            }
                            info!("[RISK] Skew applied: {:.1} bps", risk.skew_bps);
                        }

                        // --- End risk-based quote gating ---

                        stats.quote_cycles += 1;

                        // Queue-aware order management:
                        // Only cancel/replace orders if price moved beyond threshold
                        // This preserves queue position for realistic fill simulation
                        let price_threshold_bps = 5.0; // Only refresh if price moved 5+ bps (preserves queue position)
                        let active_orders = executor.get_active_orders();

                        // Build set of desired prices from new ladder
                        let desired_bid_prices: Vec<f64> = ladder.bids.iter().map(|l| l.price).collect();
                        let desired_ask_prices: Vec<f64> = ladder.asks.iter().map(|l| l.price).collect();

                        // Cancel orders that are too far from any desired price
                        for order in &active_orders {
                            let prices_to_check = if order.is_buy { &desired_bid_prices } else { &desired_ask_prices };

                            // Find closest desired price
                            let min_diff_bps = prices_to_check.iter()
                                .map(|p| (order.price - p).abs() / state.mid_price * 10_000.0)
                                .fold(f64::MAX, f64::min);

                            // Cancel if no desired price is within threshold
                            if min_diff_bps > price_threshold_bps {
                                executor.cancel_order(&cli.asset, order.oid).await;
                                state.on_order_cancelled(order.price);
                            }
                        }

                        // Get updated active orders after cancellations
                        let active_orders = executor.get_active_orders();
                        let active_bid_prices: Vec<f64> = active_orders.iter()
                            .filter(|o| o.is_buy)
                            .map(|o| o.price)
                            .collect();
                        let active_ask_prices: Vec<f64> = active_orders.iter()
                            .filter(|o| !o.is_buy)
                            .map(|o| o.price)
                            .collect();

                        // Count quotes
                        stats.bid_quotes += ladder.bids.len() as u64;
                        stats.ask_quotes += ladder.asks.len() as u64;
                        stats.quotes_generated += (ladder.bids.len() + ladder.asks.len()) as u64;

                        // Track spread
                        if let (Some(best_bid), Some(best_ask)) = (ladder.bids.first(), ladder.asks.first()) {
                            let spread_bps = (best_ask.price - best_bid.price) / state.mid_price * 10000.0;
                            spread_samples.push(spread_bps);

                            if stats.min_spread_bps == 0.0 || spread_bps < stats.min_spread_bps {
                                stats.min_spread_bps = spread_bps;
                            }
                            if spread_bps > stats.max_spread_bps {
                                stats.max_spread_bps = spread_bps;
                            }

                            // Track spread distribution
                            let bucket = if spread_bps < 5.0 { "0-5" }
                                else if spread_bps < 10.0 { "5-10" }
                                else if spread_bps < 15.0 { "10-15" }
                                else if spread_bps < 20.0 { "15-20" }
                                else if spread_bps < 30.0 { "20-30" }
                                else { "30+" };
                            *spread_counts.entry(bucket.to_string()).or_insert(0) += 1;

                            // Capture quote history (every 5 seconds)
                            if last_quote_snapshot.elapsed() >= Duration::from_secs(5) {
                                last_quote_snapshot = Instant::now();
                                let now = chrono::Utc::now();
                                if quote_history.len() >= 360 {
                                    quote_history.pop_front();
                                }
                                quote_history.push_back(QuoteSnapshot {
                                    time: now.format("%H:%M:%S").to_string(),
                                    timestamp_ms: now.timestamp_millis(),
                                    bid_prices: ladder.bids.iter().map(|l| l.price).collect(),
                                    ask_prices: ladder.asks.iter().map(|l| l.price).collect(),
                                    spread_bps,
                                });
                            }
                        }

                        // Build market params and predictions
                        let params = state.build_market_params();

                        // Extract signal contributions for analytics attribution
                        let integrated_signals = state.signal_integrator.get_signals();
                        if let Some(ref contributions_record) = integrated_signals.signal_contributions {
                            let timestamp_ns = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap()
                                .as_nanos() as u64;
                            let contributions = CycleContributions {
                                cycle_id: state.cycle_count,
                                timestamp_ns,
                                contributions: vec![
                                    SignalContribution {
                                        signal_name: "LeadLag".to_string(),
                                        spread_adjustment_bps: 0.0,
                                        skew_adjustment_bps: contributions_record.lead_lag_skew_bps,
                                        was_active: contributions_record.lead_lag_active,
                                        gating_weight: contributions_record.lead_lag_gating_weight,
                                        raw_value: contributions_record.lead_lag_skew_bps,
                                    },
                                    SignalContribution {
                                        signal_name: "InformedFlow".to_string(),
                                        spread_adjustment_bps: (contributions_record.informed_flow_spread_mult - 1.0) * 100.0,
                                        skew_adjustment_bps: 0.0,
                                        was_active: contributions_record.informed_flow_active,
                                        gating_weight: contributions_record.informed_flow_gating_weight,
                                        raw_value: contributions_record.informed_flow_spread_mult,
                                    },
                                    SignalContribution {
                                        signal_name: "RegimeDetection".to_string(),
                                        spread_adjustment_bps: 0.0,
                                        skew_adjustment_bps: 0.0,
                                        was_active: contributions_record.regime_active,
                                        gating_weight: if contributions_record.regime_active { 1.0 } else { 0.0 },
                                        raw_value: contributions_record.regime_kappa_effective,
                                    },
                                    SignalContribution {
                                        signal_name: "CrossVenue".to_string(),
                                        spread_adjustment_bps: (contributions_record.cross_venue_spread_mult - 1.0) * 100.0,
                                        skew_adjustment_bps: contributions_record.cross_venue_skew_bps,
                                        was_active: contributions_record.cross_venue_active,
                                        gating_weight: if contributions_record.cross_venue_active { 1.0 } else { 0.0 },
                                        raw_value: contributions_record.cross_venue_spread_mult,
                                    },
                                    SignalContribution {
                                        signal_name: "VPIN".to_string(),
                                        spread_adjustment_bps: (contributions_record.vpin_spread_mult - 1.0) * 100.0,
                                        skew_adjustment_bps: 0.0,
                                        was_active: contributions_record.vpin_active,
                                        gating_weight: if contributions_record.vpin_active { 1.0 } else { 0.0 },
                                        raw_value: contributions_record.vpin_spread_mult,
                                    },
                                    SignalContribution {
                                        signal_name: "BuyPressure".to_string(),
                                        spread_adjustment_bps: 0.0,
                                        skew_adjustment_bps: contributions_record.buy_pressure_skew_bps,
                                        was_active: contributions_record.buy_pressure_active,
                                        gating_weight: if contributions_record.buy_pressure_active { 1.0 } else { 0.0 },
                                        raw_value: contributions_record.buy_pressure_skew_bps,
                                    },
                                ],
                                total_spread_mult: integrated_signals.total_spread_mult,
                                combined_skew_bps: integrated_signals.combined_skew_bps,
                            };

                            if let Some(ref mut logger) = analytics_logger {
                                let _ = logger.log_contributions(&contributions);
                            }

                            last_cycle_contributions = Some(contributions);
                        }

                        let market_snapshot = MarketStateSnapshot::from_params(
                            &params,
                            state.bid_levels.clone(),
                            state.ask_levels.clone(),
                            state.inventory,
                            cli.gamma,
                        );

                        let predictions = ModelPredictions::from_ladder(
                            &ladder,
                            &params,
                            cli.target_spread_bps / 2.0,
                            cli.target_spread_bps / 2.0,
                            0.0,
                            Some(&fill_simulator),
                        );

                        // Finalize previous cycle: attach outcomes if fills occurred
                        if current_cycle_id > 0 && !cycle_fills.is_empty() {
                            let prev_record = prediction_logger.get_record(current_cycle_id);
                            let mid = state.mid_price;

                            let fill_outcomes: Vec<FillOutcome> = cycle_fills
                                .iter()
                                .map(|(_, fill)| {
                                    // Match fill to closest prediction level on same side
                                    let level_index = prev_record
                                        .as_ref()
                                        .and_then(|r| {
                                            r.predictions
                                                .levels
                                                .iter()
                                                .enumerate()
                                                .filter(|(_, l)| l.side == fill.side)
                                                .min_by(|(_, a), (_, b)| {
                                                    (a.price - fill.fill_price)
                                                        .abs()
                                                        .partial_cmp(
                                                            &(b.price - fill.fill_price).abs(),
                                                        )
                                                        .unwrap_or(std::cmp::Ordering::Equal)
                                                })
                                                .map(|(i, _)| i)
                                        })
                                        .unwrap_or(0);

                                    let as_bps = if fill.side == Side::Buy {
                                        (fill.fill_price - mid) / mid * 10_000.0
                                    } else {
                                        (mid - fill.fill_price) / mid * 10_000.0
                                    };

                                    FillOutcome {
                                        level_index,
                                        fill_timestamp_ns: fill.timestamp_ns,
                                        fill_price: fill.fill_price,
                                        fill_size: fill.fill_size,
                                        mark_price_at_fill: mid,
                                        mark_price_100ms_later: mid,
                                        mark_price_1s_later: mid,
                                        mark_price_10s_later: mid,
                                        realized_as_bps: as_bps,
                                    }
                                })
                                .collect();

                            let outcomes = ObservedOutcomes {
                                fills: fill_outcomes,
                                price_1s_later: mid,
                                price_10s_later: mid,
                                price_60s_later: mid,
                                adverse_selection_realized_bps: 0.0,
                                realized_pnl: 0.0,
                            };
                            prediction_logger.attach_outcomes(current_cycle_id, outcomes);
                        }
                        cycle_fills.clear();

                        // Log prediction and capture cycle_id
                        current_cycle_id = prediction_logger.log_prediction(market_snapshot, predictions);

                        // Place simulated orders only at NEW price levels
                        // (preserves queue position at existing levels)
                        for level in &ladder.bids {
                            // Check if we already have an order near this price
                            let has_order = active_bid_prices.iter()
                                .any(|p| ((level.price - p).abs() / state.mid_price * 10_000.0) < price_threshold_bps);
                            if !has_order {
                                executor.place_order(
                                    &cli.asset,
                                    level.price,
                                    level.size,
                                    true,
                                    None,
                                    true,
                                ).await;
                            }
                        }
                        for level in &ladder.asks {
                            let has_order = active_ask_prices.iter()
                                .any(|p| ((level.price - p).abs() / state.mid_price * 10_000.0) < price_threshold_bps);
                            if !has_order {
                                executor.place_order(
                                    &cli.asset,
                                    level.price,
                                    level.size,
                                    false,
                                    None,
                                    true,
                                ).await;
                            }
                        }

                        if cli.verbose && stats.quote_cycles % 100 == 0 {
                            info!(
                                "Quote cycle {} - mid={:.2} sigma={:.6} kappa={:.0} inventory={:.4} fills={}",
                                stats.quote_cycles,
                                state.mid_price,
                                state.estimator.sigma_clean(),
                                state.estimator.kappa(),
                                state.inventory,
                                stats.simulated_fills
                            );
                        }

                        // Update dashboard state
                        if cli.dashboard {
                            let sigma = state.estimator.sigma_clean();
                            let kappa = state.estimator.kappa();
                            let spread_bps = if let (Some(bid), Some(ask)) = (ladder.bids.first(), ladder.asks.first()) {
                                (ask.price - bid.price) / state.mid_price * 10000.0
                            } else {
                                0.0
                            };

                            // Compute regime probabilities
                            let regime_probs = compute_regime_probabilities(0.0, 1.0, sigma);
                            let regime = classify_regime(0.0, 1.0, sigma);

                            let outcome_summary = outcome_tracker.get_summary();

                            // Build current quotes and PnL for updates
                            let live_quotes = LiveQuotes {
                                mid: state.mid_price,
                                spread_bps,
                                inventory: state.inventory,
                                regime: regime.clone(),
                                kappa,
                                gamma: cli.gamma,
                                fill_prob: 0.2, // Estimated
                                adverse_prob: 0.1, // Estimated
                            };

                            let pnl_attribution = PnLAttribution {
                                spread_capture: outcome_summary.spread_capture,
                                adverse_selection: outcome_summary.adverse_selection,
                                inventory_cost: outcome_summary.inventory_cost,
                                fees: outcome_summary.fee_cost,
                                total: outcome_summary.total_pnl,
                            };

                            // Add regime snapshot every 5 seconds (keep last 60)
                            if last_regime_snapshot.elapsed() >= Duration::from_secs(5) {
                                last_regime_snapshot = Instant::now();
                                let now = chrono::Utc::now();
                                if regime_history.len() >= 60 {
                                    regime_history.remove(0);
                                }
                                regime_history.push(RegimeSnapshot {
                                    time: now.format("%H:%M:%S").to_string(),
                                    timestamp_ms: now.timestamp_millis(),
                                    quiet: regime_probs.quiet,
                                    trending: regime_probs.trending,
                                    volatile: regime_probs.volatile,
                                    cascade: regime_probs.cascade,
                                });
                            }

                            let regime_state = RegimeState {
                                current: regime,
                                probabilities: regime_probs,
                                history: regime_history.clone(),
                            };

                            // Build spread distribution from counts
                            let total_spread_samples: u32 = spread_counts.values().sum();
                            let spread_distribution: Vec<SpreadBucket> = ["0-5", "5-10", "10-15", "15-20", "20-30", "30+"]
                                .iter()
                                .map(|range| {
                                    let count = *spread_counts.get(*range).unwrap_or(&0);
                                    SpreadBucket {
                                        range_bps: range.to_string(),
                                        count,
                                        percentage: if total_spread_samples > 0 {
                                            count as f64 / total_spread_samples as f64 * 100.0
                                        } else { 0.0 },
                                    }
                                })
                                .collect();

                            // Build fill stats
                            let quote_fill_stats: Vec<QuoteFillStats> = fill_stats
                                .iter()
                                .map(|((side, level), (count, size))| QuoteFillStats {
                                    level: *level,
                                    side: side.clone(),
                                    fill_count: *count,
                                    total_size: *size,
                                })
                                .collect();

                            // Generate mock feature health data for dashboard visualization
                            let feature_health = generate_mock_feature_health(stats.quote_cycles);

                            // Update HTTP dashboard state
                            {
                                let mut dashboard = dashboard_state.write().unwrap();
                                *dashboard = DashboardState {
                                    quotes: live_quotes.clone(),
                                    pnl: pnl_attribution.clone(),
                                    regime: regime_state.clone(),
                                    fills: accumulated_fills.clone(),
                                    calibration: CalibrationState::default(),
                                    signals: Vec::new(),
                                    timestamp_ms: chrono::Utc::now().timestamp_millis(),
                                    book_history: book_history.iter().cloned().collect(),
                                    price_history: price_history.iter().cloned().collect(),
                                    quote_history: quote_history.iter().cloned().collect(),
                                    quote_fill_stats: quote_fill_stats.clone(),
                                    spread_distribution: spread_distribution.clone(),
                                    feature_health: feature_health.clone(),
                                    ..Default::default()
                                };
                            }

                            // Update WebSocket state for snapshots and push throttled updates
                            if let Some(ref ws_state) = dashboard_ws_state {
                                // Keep WebSocket state in sync for new client snapshots
                                ws_state.update_state(DashboardState {
                                    quotes: live_quotes.clone(),
                                    pnl: pnl_attribution.clone(),
                                    regime: regime_state.clone(),
                                    fills: accumulated_fills.clone(),
                                    calibration: CalibrationState::default(),
                                    signals: Vec::new(),
                                    timestamp_ms: chrono::Utc::now().timestamp_millis(),
                                    book_history: book_history.iter().cloned().collect(),
                                    price_history: price_history.iter().cloned().collect(),
                                    quote_history: quote_history.iter().cloned().collect(),
                                    quote_fill_stats,
                                    spread_distribution,
                                    feature_health,
                                    ..Default::default()
                                });

                                // Push throttled incremental update
                                if ws_throttle.should_push() {
                                    ws_state.push_update(
                                        Some(live_quotes),
                                        Some(pnl_attribution),
                                        Some(regime_state),
                                    );
                                }
                            }
                        }
                    }
                }

                // Update outcome tracker with price
                outcome_tracker.on_price_update(state.mid_price, SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64);

                // Periodic calibration feed: drain completed prediction records (every 10s)
                if last_calibration_feed.elapsed() >= Duration::from_secs(10) {
                    last_calibration_feed = Instant::now();
                    let completed = prediction_logger.drain_completed_records();
                    for record in &completed {
                        calibration_analyzer.add_record(record);
                    }
                    if !completed.is_empty() {
                        let counts = calibration_analyzer.get_sample_counts();
                        let total: usize = counts.values().sum();
                        info!(
                            "[CALIBRATION] fed {} records, total samples={}",
                            completed.len(),
                            total
                        );
                    }
                }

                // Periodic checkpoint save (every 5 minutes)
                if state.last_checkpoint_save.elapsed() >= Duration::from_secs(300) {
                    if let Some(ref mgr) = state.checkpoint_manager {
                        let bundle = state.assemble_checkpoint_bundle();
                        match mgr.save_all(&bundle) {
                            Ok(()) => info!("Checkpoint saved"),
                            Err(e) => warn!("Checkpoint save failed: {e}"),
                        }
                    }
                    state.last_checkpoint_save = Instant::now();
                }
            }
        }
    }

    // Final checkpoint save on shutdown
    if let Some(ref mgr) = state.checkpoint_manager {
        let bundle = state.assemble_checkpoint_bundle();
        match mgr.save_all(&bundle) {
            Ok(()) => info!("Final checkpoint saved on shutdown"),
            Err(e) => warn!("Final checkpoint save failed: {e}"),
        }
    }

    // Flush experience logger on shutdown
    if let Some(ref mut logger) = state.experience_logger {
        let _ = logger.flush();
        info!("Experience logger flushed on shutdown");
    }

    // Compute final statistics
    stats.runtime_secs = start_time.elapsed().as_secs_f64();
    if !spread_samples.is_empty() {
        stats.avg_spread_bps = spread_samples.iter().sum::<f64>() / spread_samples.len() as f64;
    }

    // Get executor stats
    let exec_stats = executor.get_stats();
    let outcome_summary = outcome_tracker.get_summary();

    stats.estimated_spread_capture = outcome_summary.spread_capture;
    stats.estimated_adverse_selection = outcome_summary.adverse_selection;

    // Print results
    println!("{}", stats.format());

    // Print executor statistics
    println!("\nSimulation Executor Stats:");
    println!("  Orders Placed:    {}", exec_stats.orders_placed);
    println!("  Orders Cancelled: {}", exec_stats.orders_cancelled);
    println!("  Orders Modified:  {}", exec_stats.orders_modified);
    println!("  Orders Rejected:  {}", exec_stats.orders_rejected);
    println!(
        "  Total Notional:   ${:.2}",
        exec_stats.total_notional_placed
    );

    // Print outcome summary
    println!("\n{}", outcome_summary.format());

    // Generate calibration report if requested
    if cli.report {
        info!("Generating calibration report...");
        let report = calibration_analyzer.generate_report();
        println!("\n{}", report.format());

        // Save report to file
        let report_path = PathBuf::from(&cli.output_dir).join("calibration_report.json");
        let report_json = serde_json::to_string_pretty(&report)?;
        std::fs::write(&report_path, report_json)?;
        info!("Calibration report saved to {:?}", report_path);
    }

    // Flush prediction logger
    prediction_logger.flush_all();

    // Edge Validation Report
    info!("=== EDGE VALIDATION REPORT ===");
    info!("Overall Sharpe: {:.2} (annualized, {} fills)",
        sharpe_tracker.sharpe_ratio(), sharpe_tracker.count());
    let final_summary = sharpe_tracker.summary();
    info!("  1h: {:.2}  24h: {:.2}  7d: {:.2}",
        final_summary.sharpe_1h, final_summary.sharpe_24h, final_summary.sharpe_7d);
    info!("  Mean return: {:.2} bps  Std: {:.2} bps",
        final_summary.mean_return_bps, final_summary.std_return_bps);
    info!("");
    info!("{}", edge_tracker.format_report());
    info!("");
    info!("Per-Signal Attribution:");
    info!("{}", signal_attributor.format_report());
    info!("");
    info!("Per-Signal Sharpe:");
    info!("{}", signal_sharpe.format_report());
    info!("");
    info!("Analytics files written to {}/", cli.output_dir);

    // Final analytics flush
    if let Some(ref mut logger) = analytics_logger {
        let _ = logger.flush();
    }

    info!("Paper trading simulation complete");

    Ok(())
}

// ============================================================================
// WebSocket Subscription Functions
// ============================================================================

async fn subscribe_all_mids(
    base_url: BaseUrl,
    asset: &str,
    tx: mpsc::Sender<MarketDataMessage>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let asset = asset.to_string();
    let mut retry_count = 0;
    let max_retries = 100; // Keep trying for a long time
    let mut backoff_ms = 1000u64;
    let max_backoff_ms = 30000u64;

    loop {
        match subscribe_all_mids_inner(base_url, &asset, tx.clone()).await {
            Ok(()) => {
                // Clean exit (channel closed) - likely shutdown
                info!("AllMids subscription ended gracefully");
                break;
            }
            Err(e) => {
                retry_count += 1;
                if retry_count > max_retries {
                    error!(
                        "AllMids subscription failed after {} retries: {}",
                        max_retries, e
                    );
                    return Err(e);
                }
                warn!(
                    "AllMids subscription error (attempt {}/{}): {}. Reconnecting in {}ms...",
                    retry_count, max_retries, e, backoff_ms
                );
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(max_backoff_ms);
            }
        }
    }
    Ok(())
}

async fn subscribe_all_mids_inner(
    base_url: BaseUrl,
    asset: &str,
    tx: mpsc::Sender<MarketDataMessage>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut info_client = InfoClient::new(None, Some(base_url)).await?;

    let (sender, mut receiver) = unbounded_channel::<Arc<Message>>();
    let _subscription_id = info_client
        .subscribe(Subscription::AllMids { dex: None }, sender)
        .await?;

    let asset = asset.to_string();
    while let Some(arc_msg) = receiver.recv().await {
        if let Message::AllMids(all_mids) = &*arc_msg {
            if let Some(mid_str) = all_mids.data.mids.get(&asset) {
                if let Ok(mid) = mid_str.parse::<f64>() {
                    // If send fails, the main loop has shut down
                    if tx.send(MarketDataMessage::Mid(mid)).await.is_err() {
                        return Ok(()); // Clean exit
                    }
                }
            }
        }
    }

    // Channel closed - return error to trigger reconnect
    Err("AllMids channel closed unexpectedly".into())
}

async fn subscribe_l2_book(
    base_url: BaseUrl,
    asset: &str,
    tx: mpsc::Sender<MarketDataMessage>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let asset = asset.to_string();
    let mut retry_count = 0;
    let max_retries = 100;
    let mut backoff_ms = 1000u64;
    let max_backoff_ms = 30000u64;

    loop {
        match subscribe_l2_book_inner(base_url, &asset, tx.clone()).await {
            Ok(()) => {
                info!("L2Book subscription ended gracefully");
                break;
            }
            Err(e) => {
                retry_count += 1;
                if retry_count > max_retries {
                    error!(
                        "L2Book subscription failed after {} retries: {}",
                        max_retries, e
                    );
                    return Err(e);
                }
                warn!(
                    "L2Book subscription error (attempt {}/{}): {}. Reconnecting in {}ms...",
                    retry_count, max_retries, e, backoff_ms
                );
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(max_backoff_ms);
            }
        }
    }
    Ok(())
}

async fn subscribe_l2_book_inner(
    base_url: BaseUrl,
    asset: &str,
    tx: mpsc::Sender<MarketDataMessage>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut info_client = InfoClient::new(None, Some(base_url)).await?;

    let (sender, mut receiver) = unbounded_channel::<Arc<Message>>();
    let _subscription_id = info_client
        .subscribe(
            Subscription::L2Book {
                coin: asset.to_string(),
                dex: None,
            },
            sender,
        )
        .await?;

    while let Some(arc_msg) = receiver.recv().await {
        if let Message::L2Book(book) = &*arc_msg {
            // levels is Vec<Vec<OrderBookLevel>> - [0] = bids, [1] = asks
            let bids: Vec<(f64, f64)> = if !book.data.levels.is_empty() {
                book.data.levels[0]
                    .iter()
                    .filter_map(|l| {
                        let px = l.px.parse::<f64>().ok()?;
                        let sz = l.sz.parse::<f64>().ok()?;
                        Some((px, sz))
                    })
                    .collect()
            } else {
                Vec::new()
            };

            let asks: Vec<(f64, f64)> = if book.data.levels.len() > 1 {
                book.data.levels[1]
                    .iter()
                    .filter_map(|l| {
                        let px = l.px.parse::<f64>().ok()?;
                        let sz = l.sz.parse::<f64>().ok()?;
                        Some((px, sz))
                    })
                    .collect()
            } else {
                Vec::new()
            };

            if tx
                .send(MarketDataMessage::Book { bids, asks })
                .await
                .is_err()
            {
                return Ok(()); // Clean exit
            }
        }
    }

    Err("L2Book channel closed unexpectedly".into())
}

async fn subscribe_trades(
    base_url: BaseUrl,
    asset: &str,
    tx: mpsc::Sender<MarketDataMessage>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let asset = asset.to_string();
    let mut retry_count = 0;
    let max_retries = 100;
    let mut backoff_ms = 1000u64;
    let max_backoff_ms = 30000u64;

    loop {
        match subscribe_trades_inner(base_url, &asset, tx.clone()).await {
            Ok(()) => {
                info!("Trades subscription ended gracefully");
                break;
            }
            Err(e) => {
                retry_count += 1;
                if retry_count > max_retries {
                    error!(
                        "Trades subscription failed after {} retries: {}",
                        max_retries, e
                    );
                    return Err(e);
                }
                warn!(
                    "Trades subscription error (attempt {}/{}): {}. Reconnecting in {}ms...",
                    retry_count, max_retries, e, backoff_ms
                );
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(max_backoff_ms);
            }
        }
    }
    Ok(())
}

async fn subscribe_trades_inner(
    base_url: BaseUrl,
    asset: &str,
    tx: mpsc::Sender<MarketDataMessage>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut info_client = InfoClient::new(None, Some(base_url)).await?;

    let (sender, mut receiver) = unbounded_channel::<Arc<Message>>();
    let _subscription_id = info_client
        .subscribe(
            Subscription::Trades {
                coin: asset.to_string(),
                dex: None,
            },
            sender,
        )
        .await?;

    let asset = asset.to_string();
    while let Some(arc_msg) = receiver.recv().await {
        if let Message::Trades(trades) = &*arc_msg {
            for trade in &trades.data {
                // Filter to our asset
                if trade.coin != asset {
                    continue;
                }
                if let (Ok(px), Ok(sz)) = (trade.px.parse::<f64>(), trade.sz.parse::<f64>()) {
                    let is_buy = trade.side == "B";
                    if tx
                        .send(MarketDataMessage::Trade {
                            price: px,
                            size: sz,
                            is_buy,
                        })
                        .await
                        .is_err()
                    {
                        return Ok(()); // Clean exit
                    }
                }
            }
        }
    }

    Err("Trades channel closed unexpectedly".into())
}
