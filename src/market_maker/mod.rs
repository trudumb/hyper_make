//! Modular market maker implementation.
//!
//! This module provides a market making system with pluggable components:
//! - **Strategy**: Determines quote prices and sizes (symmetric, inventory-aware, etc.)
//! - **OrderManager**: Tracks resting orders
//! - **PositionTracker**: Tracks position and deduplicates fills
//! - **Executor**: Handles order placement and cancellation

pub mod adaptive;
mod adverse_selection;
pub mod analytics;
pub mod belief;
pub mod calibration;
mod config;
pub mod core;
mod estimator;
pub mod events;
pub mod execution;
pub mod features;
pub mod fills;
pub mod infra;
pub mod messages;
pub mod multi;
pub mod process_models;
pub mod quoting;
pub mod risk;
pub mod safety;
mod strategy;
pub mod tracking;

pub mod control;
pub mod edge;
pub mod latent;
pub mod learning;
pub mod models;
pub mod monitoring;
pub mod simulation;
pub mod stochastic;

pub mod checkpoint;
pub mod environment;

mod orchestrator;

#[cfg(test)]
mod tests;

pub use adverse_selection::*;
pub use calibration::{
    BrierScoreTracker, ConditionalCalibration, InformationRatioTracker, PredictionLog,
    PredictionRecord, PredictionType,
};
pub use config::*;
pub use edge::{
    ABMetrics, ABTest, ABTestConfig, ABTestManager, ABVariant, EdgeSignalKind, HealthStatus,
    SignalHealth, SignalHealthMonitor, SignalHealthSummary,
};
pub use estimator::regime_hmm;
pub use estimator::*;
pub use execution::{
    CancelAnalysis, FillMetrics, FillRecord, FillStatistics, FillTracker, OrderEvent,
    OrderLifecycle, OrderLifecycleTracker, OrderState as ExecutionOrderState,
    Side as ExecutionSide,
};
pub use infra::*;
pub use monitoring::{
    Alert, AlertConfig, AlertHandler, AlertSeverity, AlertType, Alerter,
    DashboardState as MonitoringDashboardState, LoggingAlertHandler, PositionSide,
};
pub use process_models::*;
pub use quoting::*;
pub use risk::*;
pub use strategy::*;
pub use tracking::*;

use alloy::primitives::Address;
use belief::{CentralBeliefConfig, CentralBeliefState};
use environment::TradingEnvironment;
use tracing::{error, info, warn};

use crate::InfoClient;

/// Market maker orchestrator.
///
/// Coordinates strategy, order management, position tracking, and execution.
/// Includes live parameter estimation for GLFT strategy (σ from trades, κ from L2 book).
/// Tier 1 modules provide adverse selection measurement, queue tracking, and cascade detection.
///
/// # Component Organization
///
/// Fields are organized into logical bundles:
/// - **Core**: config, strategy, executor, orders, position, estimator
/// - **Tier1**: adverse selection, queue tracking, liquidation detection
/// - **Tier2**: hawkes, funding, spread, P&L
/// - **Safety**: kill switch, risk aggregator, fill processor
/// - **Infra**: margin, prometheus, connection health, data quality
/// - **Stochastic**: HJB controller, stochastic config, dynamic risk
pub struct MarketMaker<S: QuotingStrategy, Env: TradingEnvironment> {
    // === Core Fields ===
    /// Configuration
    config: MarketMakerConfig,
    /// Quoting strategy
    strategy: S,
    /// Trading environment (order execution + observation stream)
    pub(crate) environment: Env,
    /// Order state manager
    orders: OrderManager,
    /// WebSocket-based order state manager for improved state tracking
    ws_state: WsOrderStateManager,
    /// Position tracker
    position: PositionTracker,
    /// Info client for subscriptions
    info_client: InfoClient,
    /// User address
    user_address: Address,
    /// Latest mid price
    latest_mid: f64,
    /// Parameter estimator for σ and κ
    estimator: ParameterEstimator,
    /// Last logged warmup progress (to avoid spam)
    last_warmup_log: usize,
    /// Last time warmup blocked quoting was logged (Instant for time-based throttling)
    last_warmup_block_log: Option<std::time::Instant>,

    // === Component Bundles ===
    /// Tier 1: Production resilience (AS, queue, liquidation)
    tier1: core::Tier1Components,
    /// Tier 2: Process models (hawkes, funding, spread, P&L)
    tier2: core::Tier2Components,
    /// Safety: Kill switch, risk aggregation, fill processing
    safety: core::SafetyComponents,
    /// Infrastructure: Margin, metrics, health, data quality
    infra: core::InfraComponents,
    /// Stochastic: HJB control, dynamic risk
    stochastic: core::StochasticComponents,

    // === Inventory Governor ===
    /// Hard ceiling on position size from config.max_position.
    /// FIRST check in every quote cycle — never overridden by margin-derived limits.
    pub(crate) inventory_governor: risk::InventoryGovernor,

    /// Capital-aware policy, updated each cycle from CapacityBudget.
    /// Accessible by reconciler and other components that don't receive MarketParams.
    capital_policy: config::CapitalAwarePolicy,

    /// Options-theoretic volatility floor config.
    /// Stored on MarketMaker because the strategy trait is generic (no risk_config access).
    option_floor: strategy::OptionFloor,

    /// Self-impact estimator: tracks our book dominance and computes spread addon.
    self_impact: estimator::self_impact::SelfImpactEstimator,

    // === Directional Flow Defense Configs (retained for checkpoint compat) ===
    #[allow(dead_code)]
    staleness_config: config::StalenessConfig,
    #[allow(dead_code)]
    flow_toxicity_config: config::FlowToxicityConfig,

    // === Bootstrap from Book ===
    /// L2-derived market microstructure profile for kappa/sigma estimation.
    /// Fed by handle_l2_book(), seeds CalibrationCoordinator during warmup.
    market_profile: estimator::market_profile::MarketProfile,
    /// Coordinates L2-derived and fill-derived kappa estimates.
    /// Provides conservative kappa during warmup before robust estimators converge.
    calibration_coordinator: estimator::calibration_coordinator::CalibrationCoordinator,

    // === First-Principles Position Limit ===
    /// Effective max position SIZE (contracts), updated each quote cycle.
    ///
    /// Derived from first principles:
    /// - During warmup: Uses config.max_position as fallback
    /// - After warmup: Uses dynamic_max_position_value / mid_price
    ///
    /// Formula: max_position = min(leverage_limit, volatility_limit) / mid_price
    /// where:
    ///   leverage_limit = account_value × max_leverage
    ///   volatility_limit = (equity × risk_fraction) / (num_sigmas × σ × √T)
    ///
    /// This is THE source of truth for all position limit checks.
    /// NEVER use config.max_position directly for runtime decisions.
    effective_max_position: f64,

    /// Effective target liquidity SIZE (contracts), updated each quote cycle.
    ///
    /// Derived from first principles:
    /// - min_viable = MIN_ORDER_NOTIONAL / mid_price (exchange requirement)
    /// - effective = max(config.target_liquidity, min_viable).min(effective_max_position)
    ///
    /// This ensures orders are viable for ANY asset on ANY DEX, regardless of price.
    /// NEVER use config.target_liquidity directly for runtime quoting.
    effective_target_liquidity: f64,

    // === WebSocket State Sync (Phase 2) ===
    /// Latest OpenOrders snapshot received via WebSocket.
    /// Used by safety_sync to avoid REST API race conditions.
    last_ws_open_orders_snapshot: Option<std::collections::HashSet<u64>>,
    /// Timestamp of the last WS snapshot.
    last_ws_snapshot_time: Option<std::time::Instant>,
    /// Timestamp of last ActiveAssetData WS message (for exchange limits).
    last_active_asset_data_time: Option<std::time::Instant>,
    /// Timestamp of last WebData2 WS message (for user/margin state).
    last_web_data2_time: Option<std::time::Instant>,

    /// Local cache of spot balances (coin -> balance).
    /// Initialized via REST, updated via WebSocket ledger updates.
    spot_balance_cache: std::collections::HashMap<String, f64>,

    /// Tracks whether we're in high risk state to avoid log spam.
    /// Only logs transitions (normal → high risk, high risk → normal).
    last_high_risk_state: bool,

    // === Closed-Loop Learning ===
    /// Learning module for model confidence tracking and ensemble predictions
    learning: learning::LearningModule,

    // === Session Tracking ===
    /// Session start time for controller terminal condition handling
    session_start_time: std::time::Instant,
    /// Time when first valid market data arrived (for warmup timeout).
    /// `None` until the data quality gate first passes. Using session_start_time
    /// includes WS connection time (~40s for HYPE), defeating the 30s warmup timeout.
    first_data_time: Option<std::time::Instant>,

    // === L2 Book & Trade Cache (for EnhancedFlowContext) ===
    /// Cached L2 bid sizes (top 5 levels) for EnhancedFlowContext depth imbalance.
    /// Updated by handle_l2_book().
    cached_bid_sizes: Vec<f64>,
    /// Cached L2 ask sizes (top 5 levels) for EnhancedFlowContext depth imbalance.
    /// Updated by handle_l2_book().
    cached_ask_sizes: Vec<f64>,

    // === BBO Cache (for pre-placement crossing validation) ===
    /// Best bid price from latest L2 book update.
    /// Used to validate orders don't cross the BBO before placement.
    cached_best_bid: f64,
    /// Best ask price from latest L2 book update.
    /// Used to validate orders don't cross the BBO before placement.
    cached_best_ask: f64,
    /// Timestamp of the last L2 book update for staleness detection.
    last_l2_update_time: std::time::Instant,
    /// Recent trades buffer for EnhancedFlowContext momentum calculation.
    /// Stores (size, is_buy, timestamp_ms). Updated by handle_trades().
    /// Bounded to MAX_CACHED_TRADES entries.
    cached_trades: std::collections::VecDeque<(f64, bool, u64)>,

    // === Exchange Price Validation ===
    /// Exchange pricing rules for quote validation.
    /// Used by `place_bulk_ladder_orders` to debug_assert price validity.
    #[allow(dead_code)] // Read only in #[cfg(debug_assertions)] blocks
    exchange_rules: quoting::ExchangeRules,

    // === Event-Driven Churn Reduction (Phase 3) ===
    /// Event accumulator for event-driven quote updates.
    /// Reduces order churn by only reconciling when meaningful events occur.
    #[allow(dead_code)] // WIP: Will be used when event handlers are wired
    event_accumulator: orchestrator::EventAccumulator,

    // === Latency-Aware Mid (WS5) ===
    /// EWMA of order acknowledgement latency in milliseconds.
    /// Used to compute anticipated mid = latest_mid + drift_rate × latency.
    latency_ewma_ms: f64,

    /// Mid price when quotes were last placed. Used for latency-aware mid computation.
    mid_at_last_quote: f64,

    // === First-Principles Belief System (Bayesian Posterior Updates) ===
    /// Previous mid price for computing returns fed to beliefs_builder.
    /// Used to calculate price_return = (current_mid - prev_mid) / prev_mid.
    prev_mid_for_beliefs: f64,
    /// Last time beliefs were updated (for computing dt).
    last_beliefs_update: Option<std::time::Instant>,

    // === Centralized Belief State (Phase 2) ===
    /// Single source of truth for all Bayesian beliefs.
    /// Replaces fragmented belief modules with unified state management.
    /// Consumers read via snapshot() for consistent point-in-time views.
    central_beliefs: CentralBeliefState,

    // === Price Velocity Tracking (Flash Crash Detection) ===
    /// Previous mid price for velocity computation.
    last_mid_for_velocity: f64,
    /// Timestamp of last mid price used for velocity computation.
    last_mid_velocity_time: std::time::Instant,
    /// Current price velocity (abs(delta_mid / mid) per second).
    price_velocity_1s: f64,

    // === Position Velocity Tracking (Rapid Accumulation Detection) ===
    /// Rolling position history for velocity computation (Instant, position).
    position_history: std::collections::VecDeque<(std::time::Instant, f64)>,
    /// Current position velocity (abs change / max_position per minute).
    position_velocity_1m: f64,
    /// Fill cascade tracker: detects and mitigates same-side fill runs.
    fill_cascade_tracker: FillCascadeTracker,

    // === WS4: Bayesian Hawkes Cascade Defense ===
    /// Bayesian Hawkes process with Gamma posteriors on excitation parameters.
    /// Provides cascade_score_upper() for defense-first gamma widening.
    bayesian_hawkes: process_models::BayesianHawkes,

    // === WS3: Bayesian Model Averaging for Sigma ===
    /// BMA over 3 sigma sources (clean_bv, leverage_adjusted, particle_filter).
    /// Provides sigma_bma() and sigma_variance_bma() for PPIP ambiguity aversion.
    sigma_bma: estimator::BayesianModelAverager,

    // === WS2: Tau Inventory Tracking ===
    /// EWMA of reducing-fill holding durations (seconds).
    tau_inventory_ewma_s: f64,
    /// Online variance of reducing-fill holding durations (seconds²).
    tau_inventory_variance_s2: f64,
    /// Timestamp of last position-reducing fill for tau computation.
    last_reducing_fill_time: Option<std::time::Instant>,

    // === Fix 4: Hawkes σ_conditional High Water Mark ===
    // WS4: sigma_cascade_hwm REMOVED from quoting path.
    // CovarianceTracker (WS2) handles realized vol feedback via Bayesian posterior.
    // Fields kept for struct compatibility but unused in quoting.
    #[allow(dead_code)]
    sigma_cascade_hwm: f64,
    #[allow(dead_code)]
    sigma_cascade_hwm_set_at: u64,
    /// AS floor HWM — same pattern as sigma cascade.
    as_floor_hwm: f64,

    /// Last time a quote cycle completed (for quota-aware frequency throttling).
    last_quote_cycle_time: Option<std::time::Instant>,

    /// Quote cycle counter for warm/cold tier feature scheduling (Phase 8).
    quote_cycle_count: u64,
    /// Last time an empty ladder recovery was attempted (2s cooldown prevents churn).
    last_empty_ladder_recovery: Option<std::time::Instant>,

    // === Signal Diagnostics Cache (Phase 1) ===
    /// Cached market params from last quote cycle.
    /// Used by fill handler to capture signal state at fill time.
    /// Updated at end of each update_quotes() cycle.
    cached_market_params: Option<strategy::MarketParams>,

    // === Cross-Exchange Lead-Lag (Binance Feed) ===
    /// Receiver for Binance updates (optional).
    /// When enabled, feeds BinanceUpdate (price + trades) to SignalIntegrator
    /// for lead-lag skew and cross-venue flow analysis.
    binance_receiver: Option<tokio::sync::mpsc::Receiver<infra::BinanceUpdate>>,

    // === Checkpoint Persistence ===
    /// Checkpoint manager for saving/loading learned state across sessions.
    /// Enabled via `with_checkpoint_dir()`. Saves every 5 minutes + on shutdown.
    checkpoint_manager: Option<checkpoint::CheckpointManager>,
    /// Last time a checkpoint was saved (for periodic save interval).
    last_checkpoint_save: std::time::Instant,
    /// Prior confidence [0,1] from injection — populated by inject_prior.
    /// Used by quote_engine to blend with fill-count warmup.
    /// 0.0 = cold-start (no prior), 1.0 = fully calibrated fresh prior.
    prior_confidence: f64,
    /// Source mode of the injected prior ("paper", "live", or "").
    /// Used for ConvergenceSnapshot logging.
    prior_source_mode: String,

    // === RL Agent Control ===
    // REMOVED — RL superseded by SpreadBandit (contextual bandit).
    // RL hot-reload infrastructure removed in feedback-loop cleanup.

    // === Experience Logging ===
    /// JSONL logger for RL experience records (SARSA tuples + metadata).
    /// Enabled via `with_experience_logging()`. Writes to `logs/experience/`.
    experience_logger: Option<learning::experience::ExperienceLogger>,
    /// Unique session identifier for correlating experience records.
    experience_session_id: String,
    /// Live analytics bundle (Sharpe, signal attribution, persistence).
    /// Enabled by default for both paper and live environments.
    pub live_analytics: analytics::live::LiveAnalytics,

    // === Phase 5: Quote Outcome Tracking ===
    /// Tracks outcomes of all quotes (filled AND unfilled) for unbiased edge estimation.
    /// Enables learning from expired quotes and computing optimal spread = argmax(edge × fill_rate).
    quote_outcome_tracker: learning::quote_outcome::QuoteOutcomeTracker,

    // === Queue Value ===
    /// Heuristic for level-by-level quote filtering based on expected edge.
    /// Used by ExecutionMode selection and per-level filtering in quote_engine.rs.
    queue_value_heuristic: models::QueueValueHeuristic,

    // === Price Grid (Phase 2b: Churn Reduction) ===
    /// Configuration for price grid quantization.
    /// Snaps ladder prices to grid points to prevent sub-tick oscillations
    /// from triggering cancel+replace cycles.
    price_grid_config: quoting::PriceGridConfig,

    // === Parameter Smoothing (L1: Churn Reduction) ===
    /// EWMA + deadband smoothing for kappa/sigma/gamma before they reach the ladder.
    /// Suppresses sub-deadband oscillations that cause unnecessary requotes.
    /// Gated by `SmootherConfig.enabled` (default: false).
    parameter_smoother: ParameterSmoother,

    // === Churn Observability (L6: Churn Reduction) ===
    /// Rolling-window tracker for cancel/fill ratios, latch effectiveness,
    /// and budget suppression diagnostics.
    churn_tracker: analytics::churn_tracker::ChurnTracker,

    // === Reconcile Outcome Learning (Economic Reconciliation) ===
    /// Tracks reconcile decisions and correlates with fill outcomes to learn
    /// fill rates, edge, and calibration bias per action type.
    reconcile_outcome_tracker: tracking::ReconcileOutcomeTracker,

    // === Toxicity-Based Cancel (Sprint 2.2) === REMOVED
    // Binary side-clearing replaced by GLFT parameter routing (AS multipliers).
    // See quote_engine.rs TOXICITY HANDLING comment.

    // === Cross-Asset Signals (Sprint 4.1) ===
    /// BTC lead-lag, funding divergence, and OI-vol signals for altcoins.
    /// Provides aggregate directional signal and volatility multiplier.
    pub cross_asset_signals: learning::cross_asset::CrossAssetSignals,
    /// Last Binance mid price for computing BTC returns.
    last_binance_mid: f64,
    /// Last reference perp mid price for HIP-3 cross-venue drift.
    /// When trading hyna:HYPE, this tracks the HYPE perp mid from AllMids.
    reference_perp_mid: f64,
    /// Previous reference perp mid for computing returns.
    prev_reference_perp_mid: f64,
    /// EMA-smoothed drift rate per second from reference perp returns.
    /// Fed to DriftEstimator as one of the signal observations.
    reference_perp_drift_ema: f64,
    /// Timestamp (ms) of last reference perp mid update for dt computation.
    reference_perp_last_update_ms: i64,
    /// Bayesian drift estimator: fuses all directional signals into posterior μ.
    drift_estimator: strategy::drift_estimator::KalmanDriftEstimator,

    // === Trend Echo Attenuation (Fix 13) ===
    /// Hysteresis gate for trend feeding drift Kalman.
    /// Must earn past return_autocorrelation > 0.12 before activating.
    /// Cold start = false (gate off).
    trend_gate_active: bool,
    /// Tracks self-echo vs external signal contribution to trend changes.
    echo_estimator: strategy::echo_estimator::EchoEstimator,

    // === Cancel-on-Toxicity (Session 2 Sprint 2.2) ===
    /// Last time bid ladder was cleared due to high toxicity.
    last_toxicity_cancel_bid: Option<std::time::Instant>,
    /// Last time ask ladder was cleared due to high toxicity.
    last_toxicity_cancel_ask: Option<std::time::Instant>,

    // === Phase 14: Information-Gain Cycle Timer === //
    pub cycle_state_changes:
        crate::market_maker::orchestrator::event_accumulator::CycleStateChanges,
    pub cycle_timer: crate::market_maker::orchestrator::event_accumulator::AdaptiveCycleTimer,

    // === Cancel-Race AS Tracking (Sprint 6.3) ===
    /// Tracks adverse selection from cancel-race events (cancel sent, fill arrived first).
    /// Excess race AS feeds into spread floor to protect against latency disadvantage.
    pub cancel_race_tracker: adverse_selection::CancelRaceTracker,

    // === Bayesian Sigma Correction (CovarianceTracker) ===
    /// Tracks realized vs predicted sigma via markout feedback.
    /// `sigma_correction_factor()` multiplies into sigma_effective.
    covariance_tracker: estimator::covariance_tracker::CovarianceTracker,

    // === Q18: Vol Sampling Bias Tracker ===
    /// Tracks fill-conditioned sampling bias in volatility estimation.
    /// σ̂ conditioned on fills overweights volatile regimes.
    /// Diagnostics-only for Phase 1; Phase 2 adds importance weighting.
    sampling_bias_tracker: estimator::SamplingBiasTracker,
    /// Fill count at the start of the current quote cycle (for had_fill detection).
    last_bias_fill_count: usize,

    // === Shadow Tuner (Continuous Background Optimization) ===
    /// Producer for feeding market events to background CMA-ES optimizer.
    /// Lock-free `flume::try_send` — never blocks the event loop.
    shadow_buffer_producer: Option<simulation::ShadowBufferProducer>,
    /// Receiver for hot-swapped dynamic params from Shadow Tuner.
    /// Checked once per quote cycle via `has_changed()`.
    dynamic_params_rx: Option<tokio::sync::watch::Receiver<strategy::DynamicParams>>,
    /// Current blended dynamic params (graduated blend over 10 cycles).
    dynamic_params_current: Option<strategy::DynamicParams>,
    /// Cycles since last dynamic params update (for graduated blending).
    dynamic_params_blend_cycles: u64,
    /// Signal to Shadow Tuner that calibration gate has passed.
    shadow_tuner_gate: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// Shared live kappa for Shadow Tuner replay evaluation.
    shadow_tuner_kappa: Option<std::sync::Arc<portable_atomic::AtomicF64>>,
    /// Shared live sigma for Shadow Tuner replay evaluation.
    shadow_tuner_sigma: Option<std::sync::Arc<portable_atomic::AtomicF64>>,
}

impl<S: QuotingStrategy, Env: TradingEnvironment> MarketMaker<S, Env> {
    /// Create a new market maker.
    ///
    /// # Parameters
    /// - `config`: Market maker configuration (asset, position limits, etc.)
    /// - `strategy`: Quoting strategy (GLFT, Symmetric, etc.)
    /// - `executor`: Order executor (Hyperliquid or mock)
    /// - `info_client`: WebSocket client for market data
    /// - `user_address`: Trader's Ethereum address
    /// - `initial_position`: Starting position (from exchange sync)
    /// - `metrics`: Optional metrics recorder
    /// - `estimator_config`: Configuration for parameter estimation
    /// - `as_config`: Adverse selection estimator configuration
    /// - `queue_config`: Queue position tracker configuration
    /// - `liquidation_config`: Liquidation cascade detector configuration
    /// - `kill_switch_config`: Kill switch configuration for emergency shutdown
    /// - `hawkes_config`: Hawkes order flow estimator configuration
    /// - `funding_config`: Funding rate estimator configuration
    /// - `spread_config`: Spread process estimator configuration
    /// - `pnl_config`: P&L tracker configuration
    /// - `margin_config`: Margin-aware sizer configuration
    /// - `data_quality_config`: Data quality monitor configuration
    /// - `recovery_config`: Recovery manager configuration (Phase 3)
    /// - `reconciliation_config`: Position reconciler configuration (Phase 4)
    /// - `rate_limit_config`: Rejection rate limiter configuration (Phase 5)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: MarketMakerConfig,
        strategy: S,
        environment: Env,
        info_client: InfoClient,
        user_address: Address,
        initial_position: f64,
        metrics: MetricsRecorder,
        estimator_config: EstimatorConfig,
        as_config: AdverseSelectionConfig,
        queue_config: QueueConfig,
        liquidation_config: LiquidationConfig,
        kill_switch_config: KillSwitchConfig,
        hawkes_config: HawkesConfig,
        funding_config: FundingConfig,
        spread_config: SpreadConfig,
        pnl_config: PnLConfig,
        margin_config: MarginConfig,
        data_quality_config: infra::DataQualityConfig,
        recovery_config: infra::RecoveryConfig,
        reconciliation_config: infra::ReconciliationConfig,
        rate_limit_config: infra::RejectionRateLimitConfig,
    ) -> Self {
        // Capture config values before moving config
        let stochastic_config = config.stochastic.clone();
        let hjb_config = HJBConfig {
            session_duration_secs: stochastic_config.hjb_session_duration,
            terminal_penalty: stochastic_config.hjb_terminal_penalty,
            gamma_base: config.risk_aversion,
            funding_ewma_half_life: stochastic_config.hjb_funding_half_life,
            ..HJBConfig::default()
        };

        // Build component bundles
        let tier1 = core::Tier1Components::new(as_config, queue_config, liquidation_config);
        let tier2 =
            core::Tier2Components::new(hawkes_config, funding_config, spread_config, pnl_config);
        let safety = core::SafetyComponents::new(
            kill_switch_config.clone(),
            Self::build_risk_aggregator(&kill_switch_config),
            config.max_position,
            config.risk_aversion,
        );
        let infra = core::InfraComponents::new(
            margin_config,
            data_quality_config,
            metrics,
            recovery_config,
            reconciliation_config,
            rate_limit_config,
            infra::ProactiveRateLimitConfig::default(),
        );
        let stochastic = core::StochasticComponents::new(
            hjb_config,
            stochastic_config,
            DynamicRiskConfig::default(),
        );
        // WS5: Extract latency config before stochastic is moved into struct
        let initial_latency_ms = stochastic.stochastic_config.quote_update_latency_ms;

        // Initialize inventory governor from user's max_position (hard ceiling)
        let inventory_governor = risk::InventoryGovernor::new(config.max_position);

        // Initialize effective_max_position to static fallback (used during warmup)
        let effective_max_position = config.max_position;

        // Initialize effective_target_liquidity to static fallback (used during warmup)
        // Will be updated with first-principles min_viable floor once we have price data
        let effective_target_liquidity = config.target_liquidity;

        // Construct exchange rules before moving config into Self
        let exchange_rules = quoting::ExchangeRules::new(
            config.decimals,
            config.sz_decimals,
            10.0, // MIN_ORDER_NOTIONAL
        );

        // Build fill cascade tracker before moving config into Self
        let mut fill_cascade_tracker = FillCascadeTracker::new_with_config(&config.cascade);
        // Phase 2B: Wire configurable sigma boost from stochastic config
        fill_cascade_tracker.set_sigma_boost(config.stochastic.burst_sigma_boost);

        // Capture asset name before config is moved (Sprint 4.1)
        // BTC has no lead-lag against itself — use for_btc() to avoid self-referential model
        let cross_asset_signals = if config.asset.to_uppercase().contains("BTC") {
            learning::cross_asset::CrossAssetSignals::for_btc()
        } else {
            learning::cross_asset::CrossAssetSignals::for_altcoin(&config.asset)
        };

        Self {
            config,
            strategy,
            environment,
            orders: OrderManager::new(),
            ws_state: WsOrderStateManager::new(),
            position: PositionTracker::new(initial_position),
            info_client,
            user_address,
            latest_mid: -1.0,
            estimator: ParameterEstimator::new(estimator_config),
            last_warmup_log: 0,
            last_warmup_block_log: None,
            tier1,
            tier2,
            safety,
            infra,
            stochastic,
            inventory_governor,
            capital_policy: config::CapitalAwarePolicy::default(),
            option_floor: strategy::OptionFloor::default(),
            self_impact: estimator::self_impact::SelfImpactEstimator::new(
                estimator::self_impact::SelfImpactConfig::default(),
            ),
            market_profile: estimator::market_profile::MarketProfile::new(),
            calibration_coordinator:
                estimator::calibration_coordinator::CalibrationCoordinator::new(),
            effective_max_position,
            effective_target_liquidity,
            last_ws_open_orders_snapshot: None,
            last_ws_snapshot_time: None,
            last_active_asset_data_time: None,
            last_web_data2_time: None,
            spot_balance_cache: std::collections::HashMap::new(),
            last_high_risk_state: false,
            learning: learning::LearningModule::default(),
            session_start_time: std::time::Instant::now(),
            first_data_time: None,
            // L2 book & trade cache for EnhancedFlowContext
            cached_bid_sizes: Vec::with_capacity(5),
            cached_ask_sizes: Vec::with_capacity(5),
            // BBO cache for pre-placement crossing validation
            cached_best_bid: 0.0,
            cached_best_ask: 0.0,
            last_l2_update_time: std::time::Instant::now(),
            cached_trades: std::collections::VecDeque::with_capacity(500),
            // Exchange price validation rules
            exchange_rules,
            // Event-driven churn reduction
            event_accumulator: orchestrator::EventAccumulator::default_config(),
            // Directional flow defense configs
            staleness_config: config::StalenessConfig::default(),
            flow_toxicity_config: config::FlowToxicityConfig::default(),
            // WS5: Latency-aware mid — initialized from stochastic config
            latency_ewma_ms: initial_latency_ms,
            mid_at_last_quote: 0.0,
            // First-principles belief system
            prev_mid_for_beliefs: 0.0,
            last_beliefs_update: None,
            // Centralized belief state (single source of truth)
            central_beliefs: CentralBeliefState::new(CentralBeliefConfig::default()),
            // Price velocity tracking for flash crash detection
            last_mid_for_velocity: 0.0,
            last_mid_velocity_time: std::time::Instant::now(),
            price_velocity_1s: 0.0,
            // Position velocity tracking for rapid accumulation detection
            position_history: std::collections::VecDeque::with_capacity(120),
            position_velocity_1m: 0.0,
            // Fill cascade tracker for same-side fill run detection (config-driven)
            fill_cascade_tracker,
            // WS4: Bayesian Hawkes cascade defense (default priors, defense-first)
            bayesian_hawkes: process_models::BayesianHawkes::new(),
            // WS3: BMA sigma (starts with equal weights, learns from realized variance)
            sigma_bma: estimator::BayesianModelAverager::default(),
            // WS2: Tau inventory tracking (starts at 60s default, learns from fills)
            tau_inventory_ewma_s: 60.0,
            tau_inventory_variance_s2: 900.0,
            last_reducing_fill_time: None,
            sigma_cascade_hwm: 1.0,
            sigma_cascade_hwm_set_at: 0,
            as_floor_hwm: 0.0,
            last_quote_cycle_time: None,
            quote_cycle_count: 0,
            last_empty_ladder_recovery: None,
            // Signal diagnostics cache (updated each quote cycle)
            cached_market_params: None,
            // Cross-exchange lead-lag (disabled by default, enabled via with_binance_receiver)
            binance_receiver: None,
            // Checkpoint persistence (disabled by default, enabled via with_checkpoint_dir)
            checkpoint_manager: None,
            last_checkpoint_save: std::time::Instant::now(),
            prior_confidence: 0.0, // Updated by inject_prior
            prior_source_mode: String::new(),
            // Phase 4: RL agent control fields removed (now SpreadBandit)
            // Experience logging (disabled by default, enabled via with_experience_logging)
            experience_logger: None,
            experience_session_id: format!(
                "live_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0)
            ),
            // Live analytics (enabled by default — Sharpe, signal attribution, logging)
            live_analytics: analytics::live::LiveAnalytics::new(Some(std::path::PathBuf::from(
                "data/analytics",
            ))),
            // Phase 5: Quote outcome tracking for unbiased edge estimation
            quote_outcome_tracker: learning::quote_outcome::QuoteOutcomeTracker::new(),
            queue_value_heuristic: models::QueueValueHeuristic::new(),
            price_grid_config: quoting::PriceGridConfig {
                enabled: true,
                ..quoting::PriceGridConfig::default()
            },
            parameter_smoother: ParameterSmoother::new(SmootherConfig {
                enabled: true,
                ..SmootherConfig::default()
            }),
            churn_tracker: analytics::churn_tracker::ChurnTracker::new(60),
            reconcile_outcome_tracker: tracking::ReconcileOutcomeTracker::new(),
            cross_asset_signals,
            last_binance_mid: 0.0,
            reference_perp_mid: 0.0,
            prev_reference_perp_mid: 0.0,
            reference_perp_drift_ema: 0.0,
            reference_perp_last_update_ms: 0,
            drift_estimator: strategy::drift_estimator::KalmanDriftEstimator::default(),
            trend_gate_active: false,
            echo_estimator: strategy::echo_estimator::EchoEstimator::default(),
            last_toxicity_cancel_bid: None,
            last_toxicity_cancel_ask: None,
            cycle_state_changes:
                crate::market_maker::orchestrator::event_accumulator::CycleStateChanges::default(),
            cycle_timer:
                crate::market_maker::orchestrator::event_accumulator::AdaptiveCycleTimer::new(),
            cancel_race_tracker: adverse_selection::CancelRaceTracker::default(),
            // Bayesian sigma correction from markout feedback
            covariance_tracker: estimator::covariance_tracker::CovarianceTracker::new(),
            // Q18: Vol sampling bias tracker (diagnostics Phase 1)
            sampling_bias_tracker: estimator::SamplingBiasTracker::new(),
            last_bias_fill_count: 0,
            // Shadow Tuner (disabled by default, enabled via with_shadow_tuner)
            shadow_buffer_producer: None,
            dynamic_params_rx: None,
            dynamic_params_current: None,
            dynamic_params_blend_cycles: 0,
            shadow_tuner_gate: None,
            shadow_tuner_kappa: None,
            shadow_tuner_sigma: None,
        }
    }

    /// Set the dynamic risk configuration.
    pub fn with_dynamic_risk_config(mut self, config: DynamicRiskConfig) -> Self {
        self.stochastic.dynamic_risk_config = config;
        self
    }

    /// Configure options-theoretic volatility floor.
    pub fn with_option_floor(mut self, option_floor: strategy::OptionFloor) -> Self {
        self.option_floor = option_floor;
        self
    }

    /// Configure self-impact estimator.
    pub fn with_self_impact(mut self, config: estimator::self_impact::SelfImpactConfig) -> Self {
        self.self_impact = estimator::self_impact::SelfImpactEstimator::new(config);
        self
    }

    /// Configure microprice EMA smoothing from StochasticConfig.
    pub fn with_microprice_ema(mut self, alpha: f64, min_change_bps: f64) -> Self {
        self.estimator.set_microprice_ema(alpha, min_change_bps);
        self
    }

    /// Enable Binance feed for lead-lag signal and cross-venue flow analysis.
    ///
    /// When enabled, the event loop will receive BinanceUpdate messages
    /// (both price and trade updates) and feed them to the SignalIntegrator
    /// for cross-exchange skew calculation and bivariate flow analysis.
    ///
    /// # Arguments
    /// * `receiver` - Channel receiver for BinanceUpdate messages
    pub fn with_binance_receiver(
        mut self,
        receiver: tokio::sync::mpsc::Receiver<infra::BinanceUpdate>,
    ) -> Self {
        self.binance_receiver = Some(receiver);
        info!("Binance lead-lag feed enabled (with cross-venue flow analysis)");
        self
    }

    /// Enable checkpoint persistence for warm-starting across sessions.
    ///
    /// When enabled:
    /// - On startup: loads latest checkpoint and restores learned state
    /// - Every 5 minutes: saves checkpoint to disk
    /// - On shutdown: saves final checkpoint
    ///
    /// # Arguments
    /// * `dir` - Directory for checkpoint files (e.g., `data/checkpoints/ETH`)
    pub fn with_checkpoint_dir(mut self, dir: std::path::PathBuf) -> Self {
        match checkpoint::CheckpointManager::new(dir.clone()) {
            Ok(mgr) => {
                // Try to load and restore latest checkpoint
                match mgr.load_latest() {
                    Ok(Some(bundle)) => {
                        if checkpoint::asset_identity::assets_match(
                            &bundle.metadata.asset,
                            &self.config.asset,
                        ) {
                            self.restore_from_bundle(&bundle);
                            info!(
                                asset = %bundle.metadata.asset,
                                samples = bundle.pre_fill.learning_samples,
                                session_duration_s = bundle.metadata.session_duration_s,
                                "Restored from checkpoint"
                            );
                        } else {
                            warn!(
                                checkpoint_asset = %bundle.metadata.asset,
                                our_asset = %self.config.asset,
                                "Checkpoint asset mismatch, starting fresh"
                            );
                        }
                    }
                    Ok(None) => {
                        info!("No checkpoint found, starting fresh");
                    }
                    Err(e) => {
                        warn!("Failed to load checkpoint: {e}, starting fresh");
                    }
                }
                // Clean up old backups (keep 7 days)
                if let Err(e) = mgr.cleanup_old(7) {
                    warn!("Checkpoint cleanup failed: {e}");
                }
                self.checkpoint_manager = Some(mgr);
            }
            Err(e) => {
                warn!("Checkpoint init failed: {e}");
            }
        }
        self
    }

    /// Set WebSocket dashboard state for real-time push updates.
    pub fn with_dashboard_ws(
        mut self,
        ws_state: std::sync::Arc<infra::dashboard_ws::DashboardWsState>,
    ) -> Self {
        self.infra.dashboard_ws = Some(ws_state);
        info!("WebSocket dashboard enabled");
        self
    }

    /// Seed the margin sizer and exchange limits for paper trading.
    ///
    /// Without this, the MarketMaker sees `margin_available=0.00` and
    /// `limits_initialized=false`, generating empty ladders (no orders, no fills).
    /// This provides the synthetic account state that would normally come from
    /// WebData2 and ActiveAssetData in live mode.
    pub fn with_paper_balance(mut self, paper_balance_usd: f64) -> Self {
        // Seed margin sizer
        self.infra.margin_sizer.update_state(
            paper_balance_usd,
            0.0, // no margin used initially
            0.0, // no notional initially
        );

        // Seed exchange limits so place_bulk_ladder_orders doesn't block on
        // limits_initialized=false. Use a generous position limit derived from
        // the paper balance, leverage, and a fallback price.
        let leverage = self.infra.margin_sizer.max_leverage().max(1.0);
        let price = if self.latest_mid > 0.0 {
            self.latest_mid
        } else {
            1.0
        };
        let paper_max_position = (paper_balance_usd * leverage / price).max(1.0);
        self.infra
            .exchange_limits
            .initialize_for_paper(paper_max_position);

        info!(
            paper_balance_usd = paper_balance_usd,
            paper_max_position = %format!("{:.4}", paper_max_position),
            "Initialized paper balance for margin sizing and exchange limits"
        );
        self
    }

    /// Configure parameter smoothing for churn reduction.
    ///
    /// When enabled, EWMA + deadband filtering on kappa/sigma/gamma
    /// suppresses sub-threshold oscillations before they reach the ladder.
    pub fn with_smoother_config(mut self, config: SmootherConfig) -> Self {
        self.parameter_smoother = ParameterSmoother::new(config);
        self
    }

    // RL hot-reload and Q-table transfer removed — SpreadBandit is the optimizer now.

    /// Disable Binance-dependent signals (lead-lag, cross-venue).
    /// Call when no Binance feed is available for the asset to prevent
    /// permanent signal staleness widening.
    pub fn disable_binance_signals(&mut self) {
        self.stochastic.signal_integrator.disable_binance_signals();
    }

    /// Disable only cross-venue trade flow, keep lead-lag active for reference perp.
    pub fn disable_cross_venue_only(&mut self) {
        self.stochastic.signal_integrator.disable_cross_venue_only();
    }

    /// Set changepoint detector to ThinDex regime for HIP-3 assets.
    ///
    /// Raises threshold from 0.50 → 0.85 and requires 2 confirmations,
    /// reducing false changepoints from noisy thin-book data.
    pub fn set_changepoint_regime_thin_dex(&mut self) {
        use crate::market_maker::config::RegimeProfile;
        self.stochastic
            .controller
            .set_changepoint_regime_profile(&RegimeProfile::thin_dex());
    }

    /// Enable experience logging for RL SARSA tuples.
    ///
    /// Creates a JSONL file writer in the specified directory. Each fill
    /// produces one experience record containing state, action, reward,
    /// next_state, and market context metadata.
    ///
    /// # Arguments
    /// * `output_dir` - Directory for experience JSONL files (e.g., `logs/experience`)
    pub fn with_experience_logging(mut self, output_dir: &str) -> Self {
        match learning::experience::ExperienceLogger::new(
            output_dir,
            learning::experience::ExperienceSource::Live,
            &self.experience_session_id,
        ) {
            Ok(logger) => {
                self.experience_logger = Some(logger);
                info!(
                    session_id = %self.experience_session_id,
                    output_dir,
                    "Experience logging enabled (live)"
                );
            }
            Err(e) => {
                warn!("Failed to initialize experience logger: {e}");
            }
        }
        self
    }

    /// Enable the Shadow Tuner: continuous background CMA-ES optimization.
    ///
    /// Creates a lock-free producer/consumer pair (via flume), a watch channel for
    /// hot-swapping `DynamicParams`, and shared atomics for live kappa/sigma.
    /// Returns the `ShadowTuner` instance which the caller must spawn on a dedicated
    /// OS thread via `std::thread::spawn`.
    ///
    /// # Returns
    /// `(self, ShadowTuner)` — the tuner must be moved into `std::thread::spawn(move || tuner.run())`
    pub fn with_shadow_tuner(
        mut self,
        config: config::ShadowTunerConfig,
        initial_params: Option<strategy::DynamicParams>,
        checkpoint: Option<simulation::ShadowTunerCheckpoint>,
    ) -> (Self, simulation::ShadowTuner) {
        let max_duration_ns = config.buffer_duration_min * 60 * 1_000_000_000;
        let (producer, consumer) =
            simulation::create_shadow_buffer(config.max_buffer_events, max_duration_ns);

        let initial = initial_params.clone().unwrap_or_default();
        let (params_tx, params_rx) = tokio::sync::watch::channel(initial);

        let gate = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let kappa = std::sync::Arc::new(portable_atomic::AtomicF64::new(2000.0));
        let sigma = std::sync::Arc::new(portable_atomic::AtomicF64::new(0.001));

        let tuner_config = simulation::TunerConfig {
            cycle_interval_s: config.cycle_interval_s,
            min_events_for_replay: config.min_events_for_replay,
            improvement_threshold: config.improvement_threshold,
            convergence_sigma: config.convergence_sigma,
            max_generations_before_reset: config.max_generations_before_reset,
            param_bounds: crate::market_maker::simulation::shadow_tuner::default_param_bounds(),
            rayon_threads: config.rayon_threads,
            replay_max_position: self.config.max_position,
        };

        let tuner = simulation::ShadowTuner::new(
            consumer,
            params_tx,
            tuner_config,
            simulation::SharedEstimators {
                calibration_gate_passed: gate.clone(),
                live_kappa: kappa.clone(),
                live_sigma: sigma.clone(),
            },
            initial_params,
            checkpoint,
        );

        self.shadow_buffer_producer = Some(producer);
        self.dynamic_params_rx = Some(params_rx);
        self.shadow_tuner_gate = Some(gate);
        self.shadow_tuner_kappa = Some(kappa);
        self.shadow_tuner_sigma = Some(sigma);

        info!(
            "Shadow Tuner enabled (cycle={}s, buffer={}min)",
            config.cycle_interval_s, config.buffer_duration_min
        );

        (self, tuner)
    }

    /// Feed a market event to the Shadow Tuner's buffer (lock-free, non-blocking).
    ///
    /// Called from event loop handlers on L2Book and Trade events.
    /// Returns silently if the shadow tuner is not enabled.
    #[inline]
    fn feed_shadow_buffer(&self, event: simulation::ReplayEvent) {
        if let Some(ref producer) = self.shadow_buffer_producer {
            let _ = producer.push(event);
        }
    }

    /// Signal the Shadow Tuner that calibration gate has passed.
    fn signal_shadow_tuner_gate(&self) {
        if let Some(ref gate) = self.shadow_tuner_gate {
            if !gate.load(std::sync::atomic::Ordering::Relaxed) {
                gate.store(true, std::sync::atomic::Ordering::Relaxed);
                info!("Shadow Tuner calibration gate activated");
            }
        }
    }

    /// Update shared live kappa/sigma atomics for Shadow Tuner replay evaluation.
    fn update_shadow_tuner_estimators(&self) {
        if let Some(ref kappa) = self.shadow_tuner_kappa {
            kappa.store(self.estimator.kappa(), portable_atomic::Ordering::Relaxed);
        }
        if let Some(ref sigma) = self.shadow_tuner_sigma {
            sigma.store(
                self.estimator.sigma_clean(),
                portable_atomic::Ordering::Relaxed,
            );
        }
    }

    /// Check for new DynamicParams from Shadow Tuner and apply with graduated blending.
    fn check_dynamic_params(&mut self) {
        let rx = match self.dynamic_params_rx.as_mut() {
            Some(rx) => rx,
            None => return,
        };

        if rx.has_changed().unwrap_or(false) {
            let new_params = rx.borrow_and_update().clone();
            if let Err(e) = new_params.validate() {
                warn!("Shadow Tuner sent invalid params, rejecting: {e}");
                return;
            }
            info!(
                version = new_params.version,
                gamma = new_params.gamma_base,
                inv_beta = new_params.inventory_beta,
                floor = new_params.spread_floor_bps,
                "Shadow Tuner params update received, starting blend"
            );
            self.dynamic_params_current = Some(new_params);
            self.dynamic_params_blend_cycles = 0;
        }

        // Graduated blend over 10 cycles
        if let Some(ref target) = self.dynamic_params_current.clone() {
            if self.dynamic_params_blend_cycles < 10 {
                self.dynamic_params_blend_cycles += 1;
                let alpha = self.dynamic_params_blend_cycles as f64 / 10.0;
                let position_ratio = (self.position.position().abs()
                    / self.effective_max_position.max(0.001))
                .clamp(0.0, 1.0);
                let blended = strategy::DynamicParams::blend(
                    &strategy::DynamicParams::default(),
                    target,
                    alpha,
                    position_ratio,
                );
                self.dynamic_params_current = Some(blended);
            }
        }
    }

    /// Assemble a checkpoint bundle from current state.
    fn assemble_checkpoint_bundle(&self) -> checkpoint::CheckpointBundle {
        let (
            mut vol_filter,
            informed_flow,
            fill_rate,
            kappa_own,
            kappa_bid,
            kappa_ask,
            momentum,
            kappa_orchestrator_cp,
        ) = self.estimator.to_checkpoint();

        // Q18: Populate bias tracker fields on vol_filter checkpoint
        vol_filter.bias_fill_intervals = self.sampling_bias_tracker.fill_intervals();
        vol_filter.bias_nonfill_intervals = self.sampling_bias_tracker.nonfill_intervals();

        checkpoint::CheckpointBundle {
            metadata: checkpoint::CheckpointMetadata {
                version: 1,
                timestamp_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
                asset: self.config.asset.to_string(),
                session_duration_s: self.session_start_time.elapsed().as_secs_f64(),
                base_symbol: checkpoint::asset_identity::base_symbol(&self.config.asset)
                    .to_string(),
                source_mode: String::new(),
                parent_timestamp_ms: 0,
                chain_depth: 0,
            },
            learned_params: self.stochastic.learned_params.clone(),
            pre_fill: self.tier1.pre_fill_classifier.to_checkpoint(),
            enhanced: self.tier1.enhanced_classifier.to_checkpoint(),
            vol_filter,
            regime_hmm: self.stochastic.regime_hmm.to_checkpoint(),
            informed_flow,
            fill_rate,
            kappa_own,
            kappa_bid,
            kappa_ask,
            momentum,
            kelly_tracker: {
                let kt = self.learning.kelly_tracker();
                let total = kt.total_trades();
                let n_wins = (total as f64 * kt.win_rate()).round() as u64;
                checkpoint::KellyTrackerCheckpoint {
                    ewma_wins: kt.avg_win(),
                    n_wins,
                    ewma_losses: kt.avg_loss(),
                    n_losses: total.saturating_sub(n_wins),
                    decay: 0.99,
                    horizon_ms: self.learning.config().prediction_horizon_ms,
                }
            },
            ensemble_weights: checkpoint::EnsembleWeightsCheckpoint {
                model_weights: self.learning.ensemble_weights(),
                total_updates: self.learning.ensemble_total_updates(),
            },
            quote_outcomes: self.quote_outcome_tracker.to_checkpoint(),
            spread_bandit: self.stochastic.spread_bandit.to_checkpoint(),
            baseline_tracker: checkpoint::BaselineTrackerCheckpoint {
                ewma_reward: self.stochastic.baseline_tracker.baseline(),
                n_observations: self.stochastic.baseline_tracker.n_observations(),
            },
            kill_switch: self.safety.kill_switch.to_checkpoint(),
            readiness: checkpoint::PriorReadiness::default(),
            calibration_coordinator: self.calibration_coordinator.clone(),
            kappa_orchestrator: kappa_orchestrator_cp,
            prior_confidence: 0.0,
            bayesian_fair_value: Default::default(),
            shadow_tuner: None,
        }
    }

    /// Restore state from a checkpoint bundle.
    fn restore_from_bundle(&mut self, bundle: &checkpoint::CheckpointBundle) {
        self.tier1
            .pre_fill_classifier
            .restore_checkpoint(&bundle.pre_fill);
        self.tier1
            .enhanced_classifier
            .restore_checkpoint(&bundle.enhanced);
        self.estimator.restore_checkpoint(bundle);
        self.stochastic
            .regime_hmm
            .restore_checkpoint(&bundle.regime_hmm);
        self.stochastic.learned_params = bundle.learned_params.clone();
        self.learning
            .restore_from_checkpoint(&bundle.kelly_tracker, &bundle.ensemble_weights);
        self.quote_outcome_tracker
            .restore_from_checkpoint(&bundle.quote_outcomes);
        self.stochastic
            .spread_bandit
            .restore_from_checkpoint(&bundle.spread_bandit);
        self.stochastic.baseline_tracker.restore(
            bundle.baseline_tracker.ewma_reward,
            bundle.baseline_tracker.n_observations,
        );
        self.safety
            .kill_switch
            .restore_from_checkpoint(&bundle.kill_switch);
        // Restore calibration coordinator (L2-derived kappa blending state)
        self.calibration_coordinator = bundle.calibration_coordinator.clone();
        // Q18: Restore bias tracker intervals from checkpoint
        self.sampling_bias_tracker.restore(
            bundle.vol_filter.bias_fill_intervals,
            bundle.vol_filter.bias_nonfill_intervals,
        );
    }

    // =========================================================================
    // Prior Transfer Protocol
    // =========================================================================
}

/// φ: S → P — extract learned priors.
impl<S: QuotingStrategy, Env: TradingEnvironment> checkpoint::PriorExtract for MarketMaker<S, Env> {
    fn extract_prior(&self) -> checkpoint::CheckpointBundle {
        self.assemble_checkpoint_bundle()
    }
}

/// ψ: P × S → S — inject priors with configurable blending.
impl<S: QuotingStrategy, Env: TradingEnvironment> checkpoint::PriorInject for MarketMaker<S, Env> {
    fn inject_prior(
        &mut self,
        prior: &checkpoint::CheckpointBundle,
        config: &checkpoint::InjectionConfig,
    ) -> usize {
        use tracing::{info, warn};

        // Validate asset match (using base symbol for cross-DEX compatibility).
        if config.require_asset_match
            && !checkpoint::asset_identity::assets_match(&prior.metadata.asset, &self.config.asset)
        {
            warn!(
                prior_asset = %prior.metadata.asset,
                our_asset = %self.config.asset,
                "Prior asset mismatch — skipping injection"
            );
            return 0;
        }

        // Compute age and age-based confidence (soft decay replaces hard cutoff).
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let age_s = (now_ms.saturating_sub(prior.metadata.timestamp_ms)) as f64 / 1000.0;
        let age_confidence = checkpoint::transfer::prior_age_confidence(age_s, &config.age_policy);

        if age_confidence <= 0.0 {
            warn!(
                age_s = age_s,
                hard_reject_s = config.age_policy.hard_reject_age_s,
                "Prior beyond hard reject age — skipping injection"
            );
            return 0;
        }

        if age_confidence < 1.0 {
            info!(
                age_s = %format!("{:.0}", age_s),
                age_confidence = %format!("{:.2}", age_confidence),
                "Prior age-decayed — injecting with reduced confidence"
            );
        }

        // Inject all components via restore_from_bundle.
        // Create a modified bundle that zeroes out excluded components.
        let mut injection_bundle = prior.clone();
        if config.skip_kill_switch {
            injection_bundle.kill_switch = checkpoint::KillSwitchCheckpoint::default();
        }

        // Restore all standard components (including quote outcomes).
        self.restore_from_bundle(&injection_bundle);

        // Compute and store continuous confidence for quote_engine blending.
        let gate = crate::market_maker::calibration::gate::CalibrationGate::new(
            crate::market_maker::calibration::gate::CalibrationGateConfig::default(),
        );
        let prior_conf = gate.confidence(prior, age_s);
        self.prior_confidence = prior_conf.overall;
        self.prior_source_mode = prior.metadata.source_mode.clone();

        info!(
            asset = %prior.metadata.asset,
            prior_age_s = %format!("{:.0}", age_s),
            age_confidence = %format!("{:.2}", age_confidence),
            prior_confidence = %format!("{:.2}", prior_conf.overall),
            structural = %format!("{:.2}", prior_conf.structural),
            execution = %format!("{:.2}", prior_conf.execution),
            session_duration_s = %format!("{:.0}", prior.metadata.session_duration_s),
            skip_kill_switch = config.skip_kill_switch,
            "Prior injected successfully"
        );

        0 // RL states no longer injected
    }
}

impl<S: QuotingStrategy, Env: TradingEnvironment> MarketMaker<S, Env> {
    // =========================================================================
    // Accessor Methods
    // =========================================================================

    /// Get current position.
    pub fn position(&self) -> f64 {
        self.position.position()
    }

    /// Get latest mid price.
    pub fn latest_mid(&self) -> f64 {
        self.latest_mid
    }

    /// Get prior confidence [0,1] from injection.
    /// 0.0 = cold-start, >0 = prior was injected with this confidence level.
    pub fn prior_confidence(&self) -> f64 {
        self.prior_confidence
    }

    /// Get kill switch reference for monitoring.
    pub fn kill_switch(&self) -> &KillSwitch {
        &self.safety.kill_switch
    }

    /// Get prometheus metrics for external monitoring.
    pub fn prometheus(&self) -> &PrometheusMetrics {
        &self.infra.prometheus
    }

    /// Get asset name for metrics labeling.
    pub fn asset(&self) -> &str {
        &self.config.asset
    }

    /// Get count of tracked orders.
    pub fn orders_count(&self) -> usize {
        self.orders.order_ids().len()
    }

    /// Get centralized belief state reference for monitoring.
    pub fn central_beliefs(&self) -> &CentralBeliefState {
        &self.central_beliefs
    }

    /// Get mutable centralized belief state reference for updates.
    pub fn central_beliefs_mut(&mut self) -> &mut CentralBeliefState {
        &mut self.central_beliefs
    }

    /// Configure signal diagnostics export path.
    /// Call this at startup to enable fill signal export to JSON.
    pub fn with_signal_export_path(mut self, path: String) -> Self {
        self.safety.signal_store.set_export_path(path);
        self
    }

    /// Get signal store stats (total_recorded, total_exported, in_memory, pending_markouts).
    pub fn signal_store_stats(&self) -> (u64, u64, usize, usize) {
        self.safety.signal_store.stats()
    }

    // =========================================================================
    // Risk Methods
    // =========================================================================

    /// Build a unified RiskState snapshot from current MarketMaker state.
    ///
    /// This provides a single point-in-time view of all risk-relevant data
    /// for use with the RiskAggregator system.
    pub fn build_risk_state(&self) -> risk::RiskState {
        let pnl_summary = self.tier2.pnl_tracker.summary(self.latest_mid);
        let data_age = self.infra.connection_health.time_since_last_data();
        let last_data_time = std::time::Instant::now()
            .checked_sub(data_age)
            .unwrap_or_else(std::time::Instant::now);

        let position = self.position.position();

        // config.max_position is ALWAYS the hard ceiling (InventoryGovernor principle).
        // Margin can only LOWER the limit, never raise it above config.
        let risk_max_position = self.effective_max_position.min(self.config.max_position);
        risk::RiskState::new(
            pnl_summary.total_pnl,
            pnl_summary.peak_pnl,
            position,
            risk_max_position, // min(effective, config) — config is always hard ceiling
            self.latest_mid,
            self.safety.kill_switch.max_position_value(),
            self.infra.margin_sizer.state().account_value,
            self.estimator.sigma_clean(),
            self.tier1.liquidation_detector.cascade_severity(),
            last_data_time,
        )
        .with_pnl_breakdown(pnl_summary.realized_pnl, pnl_summary.unrealized_pnl)
        .with_volatility(
            self.estimator.sigma_clean(),
            self.estimator.sigma_confidence(),
            self.estimator.jump_ratio(),
        )
        .with_cascade(
            self.tier1.liquidation_detector.cascade_severity(),
            self.tier1.liquidation_detector.tail_risk_multiplier(),
            self.tier1.liquidation_detector.should_pull_quotes(),
        )
        .with_adverse_selection(
            self.tier1.adverse_selection.realized_as_bps(),
            self.tier1.adverse_selection.predicted_alpha(),
        )
        .with_pending_exposure(
            self.orders.pending_exposure().0,
            self.orders.pending_exposure().1,
        )
        .with_connection_state(
            self.infra.connection_health.state()
                == crate::market_maker::infra::ConnectionState::Reconnecting,
            self.infra.connection_health.current_attempt() as u32,
            self.infra.connection_health.state()
                == crate::market_maker::infra::ConnectionState::Failed,
        )
        .with_rate_limit_errors(self.safety.kill_switch.state().rate_limit_errors)
        .with_price_velocity(self.price_velocity_1s)
        .with_position_velocity(self.position_velocity_1m)
    }

    /// Get current session time as fraction [0, 1].
    ///
    /// Returns 0.0 at session start, 1.0 at session end.
    /// Used by the stochastic controller for terminal condition handling.
    pub fn session_time_fraction(&self) -> f64 {
        // Use elapsed time since session start
        let elapsed_secs = self.session_start_time.elapsed().as_secs_f64();

        // Assume 24-hour session (can be configured later)
        const SESSION_DURATION_SECS: f64 = 24.0 * 60.0 * 60.0;

        (elapsed_secs / SESSION_DURATION_SECS).clamp(0.0, 1.0)
    }

    /// Check kill switch conditions and update state.
    /// Called after each message and periodically.
    fn check_kill_switch(&mut self) {
        // Calculate data age from connection health
        let data_age = self.infra.connection_health.time_since_last_data();
        let last_data_time = std::time::Instant::now()
            .checked_sub(data_age)
            .unwrap_or_else(std::time::Instant::now);

        // Build current state for kill switch check using actual P&L
        // Use summary_update_peak() to properly track peak PnL for drawdown calculation
        let pnl_summary = self.tier2.pnl_tracker.summary_update_peak(self.latest_mid);
        let margin_state = self.infra.margin_sizer.state();
        let state = KillSwitchState {
            daily_pnl: pnl_summary.total_pnl,
            peak_pnl: pnl_summary.peak_pnl,
            position: self.position.position(),
            mid_price: self.latest_mid,
            last_data_time,
            rate_limit_errors: self.safety.kill_switch.state().rate_limit_errors,
            cascade_severity: self.tier1.liquidation_detector.cascade_severity(),
            // Capital-efficient: margin data for position runaway check
            account_value: margin_state.account_value,
            leverage: self.infra.margin_sizer.max_leverage(),
            // Ladder depth updated separately via set_ladder_depth()
            max_ladder_one_side_contracts: self
                .safety
                .kill_switch
                .state()
                .max_ladder_one_side_contracts,
            // Preserve initial position from startup for runaway exemption
            initial_position: self.safety.kill_switch.state().initial_position,
            // Q20: Stuck inventory state (preserved from kill switch state)
            position_stuck_cycles: self.safety.kill_switch.state().position_stuck_cycles,
            unrealized_as_cost_usd: self.safety.kill_switch.state().unrealized_as_cost_usd,
            has_reducing_quotes: self.safety.kill_switch.state().has_reducing_quotes,
            prev_mid_for_stuck: self.safety.kill_switch.state().prev_mid_for_stuck,
        };

        // Update position in kill switch (for value calculation)
        self.safety
            .kill_switch
            .update_position(self.position.position(), self.latest_mid);

        // Update cascade severity
        self.safety
            .kill_switch
            .update_cascade_severity(self.tier1.liquidation_detector.cascade_severity());

        // Warn if data is stale
        if self.infra.connection_health.is_data_stale() {
            warn!(
                data_age_secs = %format!("{:.1}", data_age.as_secs_f64()),
                "WebSocket data is stale - connection may be unhealthy"
            );
        }

        // Check all conditions
        if let Some(reason) = self.safety.kill_switch.check(&state) {
            warn!("Kill switch condition detected: {}", reason);
        }

        // Evaluate using RiskAggregator for unified risk assessment
        let risk_state = self.build_risk_state();
        let aggregated = self.safety.risk_aggregator.evaluate(&risk_state);

        if aggregated.should_kill() {
            error!(
                reasons = ?aggregated.kill_reasons,
                "RiskAggregator kill condition triggered"
            );
        } else {
            // Use state-transition logging to reduce spam
            let is_high_risk = aggregated.max_severity >= risk::RiskSeverity::High;
            if is_high_risk && !self.last_high_risk_state {
                // Transition: normal → high risk
                warn!(
                    summary = %aggregated.summary(),
                    "High risk detected by RiskAggregator"
                );
            } else if !is_high_risk && self.last_high_risk_state {
                // Transition: high risk → normal
                info!("Risk returned to normal");
            }
            self.last_high_risk_state = is_high_risk;
        }
    }

    /// Build RiskAggregator with monitors from kill switch configuration.
    fn build_risk_aggregator(config: &KillSwitchConfig) -> risk::RiskAggregator {
        use risk::monitors::*;

        // Default cascade thresholds (pull at 0.8, kill at 0.95)
        const DEFAULT_CASCADE_PULL: f64 = 0.8;
        const DEFAULT_CASCADE_KILL: f64 = 0.95;

        risk::RiskAggregator::new()
            .with_monitor(Box::new(LossMonitor::new(config.max_daily_loss)))
            .with_monitor(Box::new(
                DrawdownMonitor::new(config.max_drawdown)
                    .with_min_peak(config.min_peak_for_drawdown)
                    .with_max_absolute_drawdown(config.max_absolute_drawdown),
            ))
            .with_monitor(Box::new(PositionMonitor::new()))
            .with_monitor(Box::new(DataStalenessMonitor::new(
                config.stale_data_threshold,
            )))
            .with_monitor(Box::new(CascadeMonitor::new(
                DEFAULT_CASCADE_PULL,
                DEFAULT_CASCADE_KILL,
            )))
            .with_monitor(Box::new(RateLimitMonitor::new(
                config.max_rate_limit_errors,
            )))
            .with_monitor(Box::new(PriceVelocityMonitor::new(
                config.price_velocity_threshold,
                config.price_velocity_threshold * 3.0, // Kill at 3x pull threshold
            )))
            .with_monitor(Box::new(PositionVelocityMonitor::from_base_threshold(
                config.position_velocity_threshold,
            )))
    }

    /// Evaluate risk using the unified RiskAggregator.
    ///
    /// Returns an aggregated risk assessment from all monitors.
    pub fn evaluate_risk(&self) -> risk::AggregatedRisk {
        let state = self.build_risk_state();
        self.safety.risk_aggregator.evaluate(&state)
    }

    /// Update position velocity tracking after a fill or position change.
    /// Computes abs(position_change) / max_position per minute over a 60s window.
    pub fn update_position_velocity(&mut self) {
        let now = std::time::Instant::now();
        let current_pos = self.position.position();
        self.position_history.push_back((now, current_pos));

        // Prune entries older than 60 seconds
        let window = std::time::Duration::from_secs(60);
        while let Some(&(t, _)) = self.position_history.front() {
            if now.duration_since(t) > window {
                self.position_history.pop_front();
            } else {
                break;
            }
        }

        // Compute velocity: max position swing in window / max_position
        if self.position_history.len() >= 2 {
            let oldest_pos = self
                .position_history
                .front()
                .map(|&(_, p)| p)
                .unwrap_or(current_pos);
            let max_pos = self.effective_max_position.max(0.001); // avoid div by zero
            self.position_velocity_1m = (current_pos - oldest_pos).abs() / max_pos;
        }
    }
}

/// Event emitted when a fill cascade threshold is newly crossed.
/// Used by handlers to trigger immediate cancel of resting accumulating orders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CascadeEvent {
    /// Intensity ratio crossed widen threshold: widen that side's spread
    Widen(Side),
    /// Intensity ratio crossed suppress threshold: suppress (reduce-only) that side
    Suppress(Side),
}

/// Cascade severity level for tracking escalation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum CascadeLevel {
    None,
    Widen,
    Suppress,
}

/// Size-weighted Hawkes cascade tracker.
///
/// Replaces naive VecDeque+count with two `IncrementalHawkes` instances (one per side).
/// Intensity ratio λ/μ drives graduated response: burst → widen → suppress.
/// Large fills contribute proportionally more via `size_scale` normalization.
#[derive(Debug)]
pub struct FillCascadeTracker {
    hawkes_buy: process_models::IncrementalHawkes,
    hawkes_sell: process_models::IncrementalHawkes,
    baseline_intensity: f64,
    cooldown_until: Option<(std::time::Instant, Side, CascadeLevel)>,
    widen_cooldown: std::time::Duration,
    suppress_cooldown: std::time::Duration,
    widen_intensity_ratio: f64,
    suppress_intensity_ratio: f64,
    burst_intensity_ratio: f64,
    widen_addon_bps: f64,
    suppress_addon_bps: f64,
    burst_sigma_boost_until: Option<std::time::Instant>,
    burst_sigma_boost_duration: std::time::Duration,
    /// Phase 2B: Configurable sigma boost factor (default 1.5, was hardcoded 2.0).
    configured_sigma_boost: f64,
    last_level_buy: CascadeLevel,
    last_level_sell: CascadeLevel,
    // Count-based burst detection (parallel to Hawkes intensity)
    burst_ring_buy: std::collections::VecDeque<std::time::Instant>,
    burst_ring_sell: std::collections::VecDeque<std::time::Instant>,
    burst_count_threshold: usize,
    burst_count_window: std::time::Duration,
    burst_cancel_cooldown_until: Option<std::time::Instant>,
    burst_cancel_cooldown: std::time::Duration,
}

impl FillCascadeTracker {
    /// Create a fill cascade tracker from a CascadeConfig.
    pub fn new_with_config(cfg: &CascadeConfig) -> Self {
        Self {
            hawkes_buy: process_models::IncrementalHawkes::new(
                cfg.baseline_intensity,
                cfg.alpha,
                cfg.beta,
                cfg.size_scale,
            ),
            hawkes_sell: process_models::IncrementalHawkes::new(
                cfg.baseline_intensity,
                cfg.alpha,
                cfg.beta,
                cfg.size_scale,
            ),
            baseline_intensity: cfg.baseline_intensity,
            cooldown_until: None,
            widen_cooldown: std::time::Duration::from_secs(cfg.widen_cooldown_secs),
            suppress_cooldown: std::time::Duration::from_secs(cfg.suppress_cooldown_secs),
            widen_intensity_ratio: cfg.widen_intensity_ratio,
            suppress_intensity_ratio: cfg.suppress_intensity_ratio,
            burst_intensity_ratio: cfg.burst_intensity_ratio,
            widen_addon_bps: cfg.widen_addon_bps,
            suppress_addon_bps: cfg.suppress_addon_bps,
            burst_sigma_boost_until: None,
            burst_sigma_boost_duration: std::time::Duration::from_secs(cfg.burst_sigma_boost_secs),
            configured_sigma_boost: 1.5, // Default; override via set_sigma_boost()
            last_level_buy: CascadeLevel::None,
            last_level_sell: CascadeLevel::None,
            burst_ring_buy: std::collections::VecDeque::new(),
            burst_ring_sell: std::collections::VecDeque::new(),
            burst_count_threshold: cfg.burst_count_threshold,
            burst_count_window: std::time::Duration::from_secs(cfg.burst_count_window_secs),
            burst_cancel_cooldown_until: None,
            burst_cancel_cooldown: std::time::Duration::from_secs(cfg.burst_cancel_cooldown_secs),
        }
    }

    /// Create a new fill cascade tracker with default parameters.
    pub fn new() -> Self {
        Self::new_with_config(&CascadeConfig::default())
    }

    /// Record a size-weighted fill event.
    /// Returns `Some(CascadeEvent)` when intensity ratio newly crosses a threshold.
    pub fn record_fill(&mut self, side: Side, _position: f64, size: f64) -> Option<CascadeEvent> {
        self.clear_expired_cooldown();

        let hawkes = match side {
            Side::Buy => &mut self.hawkes_buy,
            Side::Sell => &mut self.hawkes_sell,
        };

        // Record event and compute ratio from returned intensity (avoids double Instant::now())
        let intensity = hawkes.record_event(size);
        let ratio = if self.baseline_intensity > 0.0 {
            intensity / self.baseline_intensity
        } else {
            1.0
        };

        // Determine current cascade level from ratio
        let current_level = if ratio >= self.suppress_intensity_ratio {
            CascadeLevel::Suppress
        } else if ratio >= self.widen_intensity_ratio {
            CascadeLevel::Widen
        } else {
            CascadeLevel::None
        };

        // Check burst: if ratio >= burst threshold, activate sigma boost
        if ratio >= self.burst_intensity_ratio {
            let was_active = self.sigma_boost_active();
            self.burst_sigma_boost_until =
                Some(std::time::Instant::now() + self.burst_sigma_boost_duration);
            if !was_active {
                tracing::warn!(
                    side = ?side,
                    ratio = format!("{:.2}", ratio),
                    boost_secs = self.burst_sigma_boost_duration.as_secs(),
                    "Fill BURST detected: intensity ratio {:.2} >= {:.2} — sigma boost activated",
                    ratio,
                    self.burst_intensity_ratio,
                );
            }
        }

        let last_level = match side {
            Side::Buy => &mut self.last_level_buy,
            Side::Sell => &mut self.last_level_sell,
        };

        // Only emit event when level escalates (current > previous)
        let event = if current_level > *last_level {
            let now = std::time::Instant::now();
            match current_level {
                CascadeLevel::Suppress => {
                    self.cooldown_until =
                        Some((now + self.suppress_cooldown, side, CascadeLevel::Suppress));
                    tracing::warn!(
                        side = ?side,
                        ratio = format!("{:.2}", ratio),
                        suppress_secs = self.suppress_cooldown.as_secs(),
                        "Fill cascade SUPPRESS: intensity ratio {:.2} >= {:.2}",
                        ratio,
                        self.suppress_intensity_ratio,
                    );
                    Some(CascadeEvent::Suppress(side))
                }
                CascadeLevel::Widen => {
                    self.cooldown_until =
                        Some((now + self.widen_cooldown, side, CascadeLevel::Widen));
                    tracing::warn!(
                        side = ?side,
                        ratio = format!("{:.2}", ratio),
                        widen_secs = self.widen_cooldown.as_secs(),
                        "Fill cascade WIDEN: intensity ratio {:.2} >= {:.2}",
                        ratio,
                        self.widen_intensity_ratio,
                    );
                    Some(CascadeEvent::Widen(side))
                }
                CascadeLevel::None => None,
            }
        } else {
            None
        };

        // Update tracked level — decay back to None if ratio dropped below thresholds
        *last_level = current_level;

        event
    }

    /// Get spread widening for a side (0.0 = normal, configurable widening due to cascade).
    pub fn spread_addon_bps(&self, side: Side) -> f64 {
        // Check active cooldown first
        if let Some((expiry, cooldown_side, level)) = self.cooldown_until {
            if std::time::Instant::now() < expiry && cooldown_side == side {
                return match level {
                    CascadeLevel::Suppress => self.suppress_addon_bps,
                    CascadeLevel::Widen => self.widen_addon_bps,
                    CascadeLevel::None => 0.0,
                };
            }
        }
        // Fallback: check live intensity ratio via snapshot (no &mut needed)
        let ratio = match side {
            Side::Buy => self.hawkes_buy.intensity_ratio_snapshot(),
            Side::Sell => self.hawkes_sell.intensity_ratio_snapshot(),
        };
        if ratio >= self.suppress_intensity_ratio {
            self.suppress_addon_bps
        } else if ratio >= self.widen_intensity_ratio {
            self.widen_addon_bps
        } else {
            0.0
        }
    }

    /// Check if a side is suppressed (reduce-only).
    pub fn is_suppressed(&self, side: Side) -> bool {
        // Check active cooldown
        if let Some((expiry, cooldown_side, level)) = self.cooldown_until {
            if std::time::Instant::now() < expiry
                && cooldown_side == side
                && level == CascadeLevel::Suppress
            {
                return true;
            }
        }
        // Fallback: check live intensity ratio via snapshot
        let ratio = match side {
            Side::Buy => self.hawkes_buy.intensity_ratio_snapshot(),
            Side::Sell => self.hawkes_sell.intensity_ratio_snapshot(),
        };
        ratio >= self.suppress_intensity_ratio
    }

    /// Check for fill burst (intensity ratio above burst threshold).
    /// Returns true when sigma boost is newly activated.
    pub fn check_burst(&mut self, side: Side) -> bool {
        let hawkes = match side {
            Side::Buy => &mut self.hawkes_buy,
            Side::Sell => &mut self.hawkes_sell,
        };
        let ratio = hawkes.intensity_ratio();
        if ratio >= self.burst_intensity_ratio {
            let was_active = self.sigma_boost_active();
            self.burst_sigma_boost_until =
                Some(std::time::Instant::now() + self.burst_sigma_boost_duration);
            if !was_active {
                tracing::warn!(
                    side = ?side,
                    ratio = format!("{:.2}", ratio),
                    boost_secs = self.burst_sigma_boost_duration.as_secs(),
                    "Fill BURST detected: intensity ratio {:.2} >= {:.2} — sigma boost activated",
                    ratio,
                    self.burst_intensity_ratio,
                );
                return true;
            }
        }
        false
    }

    /// Count-based burst detection: N+ same-side fills within window → emergency cancel trigger.
    /// Has its own cooldown to prevent repeated cancel storms.
    /// Complements Hawkes intensity-based detection by catching sustained-level bursts.
    pub fn check_count_burst(&mut self, side: Side) -> bool {
        // Respect cooldown
        if let Some(expiry) = self.burst_cancel_cooldown_until {
            if std::time::Instant::now() < expiry {
                return false;
            }
        }
        let ring = match side {
            Side::Buy => &mut self.burst_ring_buy,
            Side::Sell => &mut self.burst_ring_sell,
        };
        let now = std::time::Instant::now();
        ring.push_back(now);
        // Evict expired entries
        while ring
            .front()
            .is_some_and(|t| now.duration_since(*t) > self.burst_count_window)
        {
            ring.pop_front();
        }
        if ring.len() >= self.burst_count_threshold {
            self.burst_cancel_cooldown_until = Some(now + self.burst_cancel_cooldown);
            ring.clear();
            true
        } else {
            false
        }
    }

    /// Accessor for burst count threshold (for logging in handler).
    pub fn burst_count_threshold(&self) -> usize {
        self.burst_count_threshold
    }

    /// Accessor for burst count window in seconds (for logging in handler).
    pub fn burst_count_window_secs(&self) -> u64 {
        self.burst_count_window.as_secs()
    }

    /// Phase 2C: Current max intensity ratio across both sides (snapshot, no mutation).
    /// Used to gate side-cancel: below cascade_cancel_threshold → gamma boost only.
    pub fn max_intensity_ratio(&self) -> f64 {
        let buy_ratio = self.hawkes_buy.intensity_ratio_snapshot();
        let sell_ratio = self.hawkes_sell.intensity_ratio_snapshot();
        buy_ratio.max(sell_ratio)
    }

    /// Whether sigma boost from burst detection is currently active.
    pub fn sigma_boost_active(&self) -> bool {
        if let Some(expiry) = self.burst_sigma_boost_until {
            std::time::Instant::now() < expiry
        } else {
            false
        }
    }

    /// Sigma multiplier from burst detection: configured value when active, 1.0 otherwise.
    /// Phase 2B: Configurable (default 1.5, was hardcoded 2.0).
    pub fn sigma_boost_factor(&self) -> f64 {
        if self.sigma_boost_active() {
            self.configured_sigma_boost
        } else {
            1.0
        }
    }

    /// Set the configured sigma boost factor from StochasticConfig.
    pub fn set_sigma_boost(&mut self, factor: f64) {
        self.configured_sigma_boost = factor.max(1.0);
    }

    /// Clear expired cooldown.
    fn clear_expired_cooldown(&mut self) {
        if let Some((expiry, _, _)) = self.cooldown_until {
            if std::time::Instant::now() >= expiry {
                self.cooldown_until = None;
            }
        }
    }
}

impl Default for FillCascadeTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod fill_cascade_tests {
    use super::*;

    /// Test config: baseline=1.0, alpha=1.1, beta=2.0, size_scale=1.0
    /// Each unit-size fill adds 1.1 to weighted_sum. With fast tests (micro-second gaps),
    /// ratio after N fills ≈ 1 + N*1.1 (comfortably above thresholds, avoids exact boundaries).
    fn test_cascade_config() -> CascadeConfig {
        CascadeConfig {
            baseline_intensity: 1.0,
            alpha: 1.1,
            beta: 2.0,
            size_scale: 1.0,
            widen_intensity_ratio: 3.0,
            suppress_intensity_ratio: 5.0,
            burst_intensity_ratio: 2.0,
            widen_addon_bps: 10.0,
            suppress_addon_bps: 20.0,
            widen_cooldown_secs: 15,
            suppress_cooldown_secs: 30,
            burst_sigma_boost_secs: 30,
            burst_count_threshold: 4,
            burst_count_window_secs: 2,
            burst_cancel_cooldown_secs: 5,
        }
    }

    #[test]
    fn test_fill_cascade_small_fills_no_cascade() {
        // 1 fill → ratio ≈ 2.1, below widen (3.0)
        let mut tracker = FillCascadeTracker::new_with_config(&test_cascade_config());
        let event = tracker.record_fill(Side::Buy, 0.0, 1.0);
        assert!(event.is_none(), "single fill should not trigger cascade");
    }

    #[test]
    fn test_fill_cascade_widen_at_threshold() {
        // 2 fills → ratio ≈ 3.2, triggers Widen
        let mut tracker = FillCascadeTracker::new_with_config(&test_cascade_config());
        let e1 = tracker.record_fill(Side::Buy, 0.0, 1.0);
        assert!(e1.is_none(), "first fill below widen");
        let e2 = tracker.record_fill(Side::Buy, 0.0, 1.0);
        assert_eq!(e2, Some(CascadeEvent::Widen(Side::Buy)));
    }

    #[test]
    fn test_fill_cascade_suppress_at_threshold() {
        // 4 fills → ratio ≈ 5.4, triggers Suppress (after Widen at 2)
        let mut tracker = FillCascadeTracker::new_with_config(&test_cascade_config());
        let e1 = tracker.record_fill(Side::Buy, 0.0, 1.0);
        assert!(e1.is_none());
        let e2 = tracker.record_fill(Side::Buy, 0.0, 1.0);
        assert_eq!(e2, Some(CascadeEvent::Widen(Side::Buy)));
        let e3 = tracker.record_fill(Side::Buy, 0.0, 1.0);
        assert!(e3.is_none(), "already at widen, not yet suppress");
        let e4 = tracker.record_fill(Side::Buy, 0.0, 1.0);
        assert_eq!(e4, Some(CascadeEvent::Suppress(Side::Buy)));
        assert!(tracker.is_suppressed(Side::Buy));
    }

    #[test]
    fn test_fill_cascade_size_weighting() {
        // 1 fill of size 3.0 ≈ 3 fills of size 1.0, hits widen
        let mut tracker = FillCascadeTracker::new_with_config(&test_cascade_config());
        // size_scale=1.0, alpha=1.1, so size 3.0 contributes 3.3 to weighted_sum
        // ratio = (1.0 + 3.3) / 1.0 = 4.3 ≥ widen(3.0)
        let event = tracker.record_fill(Side::Sell, 0.0, 3.0);
        assert_eq!(
            event,
            Some(CascadeEvent::Widen(Side::Sell)),
            "single large fill should trigger widen"
        );
    }

    #[test]
    fn test_fill_cascade_sides_independent() {
        // 10 buy fills, sell ratio stays ≈ 1.0
        let mut tracker = FillCascadeTracker::new_with_config(&test_cascade_config());
        for _ in 0..10 {
            tracker.record_fill(Side::Buy, 0.0, 1.0);
        }
        // Sell side should be unaffected
        assert!(
            !tracker.is_suppressed(Side::Sell),
            "sell side should not be suppressed"
        );
        assert!(
            tracker.spread_addon_bps(Side::Sell) < f64::EPSILON,
            "sell side should have no addon"
        );
    }

    #[test]
    fn test_fill_cascade_newly_crossed_only() {
        // After Widen, additional fill at same level returns None
        let mut tracker = FillCascadeTracker::new_with_config(&test_cascade_config());
        tracker.record_fill(Side::Buy, 0.0, 1.0); // ratio ≈ 2.1
        let e2 = tracker.record_fill(Side::Buy, 0.0, 1.0); // ratio ≈ 3.2 → Widen
        assert_eq!(e2, Some(CascadeEvent::Widen(Side::Buy)));
        // 3rd fill: ratio ≈ 4.3, still Widen level, no re-emit
        let e3 = tracker.record_fill(Side::Buy, 0.0, 1.0);
        assert!(e3.is_none(), "should not re-emit Widen");
    }

    #[test]
    fn test_fill_cascade_re_trigger_after_decay() {
        // Use high beta so events decay very fast, with alpha < beta
        let mut cfg = test_cascade_config();
        cfg.beta = 50.0;
        cfg.alpha = 1.5; // alpha < beta, each fill adds 1.5 to weighted_sum
        let mut tracker = FillCascadeTracker::new_with_config(&cfg);

        // First burst: 2 fills → ratio ≈ 1 + 1.5 + 1.5 = 4.0 → Widen (3.0)
        tracker.record_fill(Side::Buy, 0.0, 1.0);
        let e2 = tracker.record_fill(Side::Buy, 0.0, 1.0);
        assert_eq!(e2, Some(CascadeEvent::Widen(Side::Buy)));

        // Sleep briefly to let intensity decay (beta=50 → exp(-50*0.2) ≈ 0)
        std::thread::sleep(std::time::Duration::from_millis(200));

        // After decay, first fill resets level to below widen
        // ratio ≈ 1 + 1.5 = 2.5 < widen(3.0) → CascadeLevel::None
        // last_level updates from Widen → None
        let e3 = tracker.record_fill(Side::Buy, 0.0, 1.0);
        assert!(e3.is_none(), "first fill after decay below widen");

        // Second fill re-triggers Widen: ratio ≈ 1 + 1.5 + 1.5 = 4.0 ≥ widen(3.0)
        let e4 = tracker.record_fill(Side::Buy, 0.0, 1.0);
        assert_eq!(
            e4,
            Some(CascadeEvent::Widen(Side::Buy)),
            "should re-trigger widen after decay"
        );
    }

    #[test]
    fn test_fill_cascade_burst_activates_sigma() {
        // Fill above burst ratio → sigma_boost_active + factor 2.0
        let mut tracker = FillCascadeTracker::new_with_config(&test_cascade_config());
        // 1 fill of size 1.0 → ratio ≈ 2.1 ≥ burst(2.0)
        tracker.record_fill(Side::Buy, 0.0, 1.0);
        assert!(
            tracker.sigma_boost_active(),
            "burst should activate sigma boost"
        );
        assert!(
            (tracker.sigma_boost_factor() - 1.5).abs() < f64::EPSILON,
            "sigma boost factor should be 1.5 (Phase 2B default)"
        );
    }

    #[test]
    fn test_fill_cascade_burst_below_threshold() {
        // 1 tiny fill → ratio < burst threshold → no sigma boost
        let mut cfg = test_cascade_config();
        cfg.burst_intensity_ratio = 3.0; // raise burst threshold
        let mut tracker = FillCascadeTracker::new_with_config(&cfg);
        tracker.record_fill(Side::Sell, 0.0, 0.5);
        // ratio = 1.0 + 1.1*0.5 = 1.55 < burst(3.0)
        assert!(
            !tracker.sigma_boost_active(),
            "tiny fill should not trigger burst"
        );
        assert!(
            (tracker.sigma_boost_factor() - 1.0).abs() < f64::EPSILON,
            "no boost without burst"
        );
    }

    #[test]
    fn test_fill_cascade_cooldown_spread_addon() {
        // After Widen, spread_addon_bps returns widen_addon_bps
        let mut tracker = FillCascadeTracker::new_with_config(&test_cascade_config());
        tracker.record_fill(Side::Sell, 0.0, 1.0);
        tracker.record_fill(Side::Sell, 0.0, 1.0); // triggers Widen
        let addon = tracker.spread_addon_bps(Side::Sell);
        assert!(
            (addon - 10.0).abs() < f64::EPSILON,
            "widen addon should be 10.0, got {}",
            addon
        );
    }

    #[test]
    fn test_fill_cascade_config_default_validates() {
        assert!(CascadeConfig::default().validate().is_ok());
    }

    #[test]
    fn test_fill_cascade_config_rejects_bad_params() {
        // alpha >= beta (non-stationary)
        let mut cfg = CascadeConfig::default();
        cfg.alpha = cfg.beta;
        assert!(cfg.validate().is_err(), "alpha >= beta should fail");

        // baseline <= 0
        let cfg2 = CascadeConfig {
            baseline_intensity: 0.0,
            ..Default::default()
        };
        assert!(cfg2.validate().is_err(), "zero baseline should fail");

        // suppress <= widen ratio
        let mut cfg3 = CascadeConfig::default();
        cfg3.suppress_intensity_ratio = cfg3.widen_intensity_ratio;
        assert!(
            cfg3.validate().is_err(),
            "suppress <= widen ratio should fail"
        );
    }

    // === Count-based burst detection tests ===

    fn count_burst_config() -> CascadeConfig {
        let mut cfg = test_cascade_config();
        cfg.burst_count_threshold = 4;
        cfg.burst_count_window_secs = 2;
        cfg.burst_cancel_cooldown_secs = 5;
        cfg
    }

    #[test]
    fn test_count_burst_below_threshold_no_trigger() {
        // 3 fills in window (threshold is 4) → no trigger
        let mut tracker = FillCascadeTracker::new_with_config(&count_burst_config());
        assert!(!tracker.check_count_burst(Side::Buy));
        assert!(!tracker.check_count_burst(Side::Buy));
        assert!(!tracker.check_count_burst(Side::Buy));
    }

    #[test]
    fn test_count_burst_at_threshold_triggers() {
        // 4 fills in window → triggers
        let mut tracker = FillCascadeTracker::new_with_config(&count_burst_config());
        assert!(!tracker.check_count_burst(Side::Buy));
        assert!(!tracker.check_count_burst(Side::Buy));
        assert!(!tracker.check_count_burst(Side::Buy));
        assert!(
            tracker.check_count_burst(Side::Buy),
            "4th fill should trigger count burst"
        );
    }

    #[test]
    fn test_count_burst_exceeds_window_no_trigger() {
        // 4 fills but spread across > 2s window → no trigger
        let mut cfg = count_burst_config();
        cfg.burst_count_window_secs = 1; // tight 1s window
        let mut tracker = FillCascadeTracker::new_with_config(&cfg);
        assert!(!tracker.check_count_burst(Side::Sell));
        assert!(!tracker.check_count_burst(Side::Sell));
        // Sleep past window
        std::thread::sleep(std::time::Duration::from_millis(1100));
        assert!(!tracker.check_count_burst(Side::Sell));
        // Only 1 fill in window now (previous 2 expired), 4th is actually 2nd in window
        assert!(
            !tracker.check_count_burst(Side::Sell),
            "fills outside window should not count"
        );
    }

    #[test]
    fn test_count_burst_cooldown_blocks_retrigger() {
        // After trigger, 4 more fills within 5s cooldown → no trigger
        let mut tracker = FillCascadeTracker::new_with_config(&count_burst_config());
        // First burst
        for _ in 0..3 {
            assert!(!tracker.check_count_burst(Side::Buy));
        }
        assert!(
            tracker.check_count_burst(Side::Buy),
            "first burst should trigger"
        );
        // Second burst during cooldown
        for _ in 0..4 {
            assert!(
                !tracker.check_count_burst(Side::Buy),
                "should not trigger during cooldown"
            );
        }
    }

    #[test]
    fn test_count_burst_cooldown_expires_allows_retrigger() {
        // After cooldown expires, next burst triggers
        let mut cfg = count_burst_config();
        cfg.burst_cancel_cooldown_secs = 1; // short cooldown for test
        let mut tracker = FillCascadeTracker::new_with_config(&cfg);
        // First burst
        for _ in 0..3 {
            assert!(!tracker.check_count_burst(Side::Sell));
        }
        assert!(tracker.check_count_burst(Side::Sell), "first burst");
        // Wait for cooldown to expire
        std::thread::sleep(std::time::Duration::from_millis(1100));
        // Second burst should trigger
        for _ in 0..3 {
            assert!(!tracker.check_count_burst(Side::Sell));
        }
        assert!(
            tracker.check_count_burst(Side::Sell),
            "should trigger after cooldown expires"
        );
    }

    #[test]
    fn test_count_burst_sides_independent() {
        // 3 buy + 3 sell = no trigger on either side (threshold is 4)
        let mut tracker = FillCascadeTracker::new_with_config(&count_burst_config());
        for _ in 0..3 {
            assert!(!tracker.check_count_burst(Side::Buy));
            assert!(!tracker.check_count_burst(Side::Sell));
        }
    }
}
