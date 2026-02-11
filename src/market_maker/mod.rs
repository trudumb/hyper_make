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
pub use estimator::*;
pub use estimator::regime_hmm;
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

    // === Event-Driven Churn Reduction (Phase 3) ===
    /// Event accumulator for event-driven quote updates.
    /// Reduces order churn by only reconciling when meaningful events occur.
    #[allow(dead_code)] // WIP: Will be used when event handlers are wired
    event_accumulator: orchestrator::EventAccumulator,

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

    // === RL Agent Control ===
    /// Whether RL agent controls quoting actions (vs observe-only)
    pub(crate) rl_enabled: bool,
    /// Minimum real fills before RL controls actions
    rl_min_real_fills: usize,
    /// Auto-disable RL if mean reward < threshold after this many fills
    rl_auto_disable_fills: usize,
    /// Threshold for RL auto-disable (default: -1.5 bps, matching maker fee)
    rl_auto_disable_threshold_bps: f64,

    // === RL Hot-Reload ===
    /// Watch channel receiver for Q-table hot-reload from an offline trainer.
    /// When the trainer writes a new checkpoint, the file watcher sends it here.
    q_table_reload_rx:
        Option<tokio::sync::watch::Receiver<Option<checkpoint::types::RLCheckpoint>>>,
    /// Blend weight for hot-reloaded Q-table (0.0 = ignore, 1.0 = replace).
    q_table_reload_weight: f64,
    /// Quote cycle counter for gating periodic RL reload checks.
    rl_reload_cycle_count: usize,

    // === Experience Logging ===
    /// JSONL logger for RL experience records (SARSA tuples + metadata).
    /// Enabled via `with_experience_logging()`. Writes to `logs/experience/`.
    experience_logger: Option<learning::experience::ExperienceLogger>,
    /// Unique session identifier for correlating experience records.
    experience_session_id: String,
    /// Live analytics bundle (Sharpe, signal attribution, persistence).
    /// Enabled by default for both paper and live environments.
    pub live_analytics: analytics::live::LiveAnalytics,
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

        // Initialize effective_max_position to static fallback (used during warmup)
        let effective_max_position = config.max_position;

        // Initialize effective_target_liquidity to static fallback (used during warmup)
        // Will be updated with first-principles min_viable floor once we have price data
        let effective_target_liquidity = config.target_liquidity;

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
            // Event-driven churn reduction
            event_accumulator: orchestrator::EventAccumulator::default_config(),
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
            // Signal diagnostics cache (updated each quote cycle)
            cached_market_params: None,
            // Cross-exchange lead-lag (disabled by default, enabled via with_binance_receiver)
            binance_receiver: None,
            // Checkpoint persistence (disabled by default, enabled via with_checkpoint_dir)
            checkpoint_manager: None,
            last_checkpoint_save: std::time::Instant::now(),
            // RL agent control (enabled by default, gated by min_real_fills)
            rl_enabled: true,
            rl_min_real_fills: 20,
            rl_auto_disable_fills: 100,
            rl_auto_disable_threshold_bps: -1.5,
            // RL hot-reload (disabled by default, enabled via with_rl_reload)
            q_table_reload_rx: None,
            q_table_reload_weight: 0.3,
            rl_reload_cycle_count: 0,
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
            live_analytics: analytics::live::LiveAnalytics::new(
                Some(std::path::PathBuf::from("data/analytics")),
            ),
        }
    }

    /// Set the dynamic risk configuration.
    pub fn with_dynamic_risk_config(mut self, config: DynamicRiskConfig) -> Self {
        self.stochastic.dynamic_risk_config = config;
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
                        if bundle.metadata.asset == *self.config.asset {
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
            0.0,  // no margin used initially
            0.0,  // no notional initially
        );

        // Seed exchange limits so place_bulk_ladder_orders doesn't block on
        // limits_initialized=false. Use a generous position limit derived from
        // the paper balance, leverage, and a fallback price.
        let leverage = self.infra.margin_sizer.max_leverage().max(1.0);
        let price = if self.latest_mid > 0.0 { self.latest_mid } else { 1.0 };
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

    /// Enable Q-table hot-reload from an offline trainer via a watch channel.
    ///
    /// The file watcher task sends new `RLCheckpoint` values through the channel
    /// whenever the checkpoint file changes. The live RL agent blends the new
    /// Q-table with its current state using `weight` (0.0 = ignore, 1.0 = replace).
    pub fn with_rl_reload(
        mut self,
        rx: tokio::sync::watch::Receiver<Option<checkpoint::types::RLCheckpoint>>,
        weight: f64,
    ) -> Self {
        self.q_table_reload_rx = Some(rx);
        self.q_table_reload_weight = weight;
        info!(weight, "RL Q-table hot-reload enabled");
        self
    }

    /// Check for a new Q-table from the hot-reload watch channel.
    ///
    /// Called periodically from the quote cycle (every 100 cycles).
    /// If the watch channel has a new value, loads it as a prior into the RL agent.
    pub fn check_rl_reload(&mut self) {
        if let Some(ref mut rx) = self.q_table_reload_rx {
            if rx.has_changed().unwrap_or(false) {
                let checkpoint_opt = rx.borrow_and_update().clone();
                if let Some(ref checkpoint) = checkpoint_opt {
                    let n_states =
                        self.load_paper_rl_prior(checkpoint, self.q_table_reload_weight);
                    info!(
                        n_states,
                        weight = self.q_table_reload_weight,
                        "RL Q-table hot-reloaded"
                    );
                }
            }
        }
    }

    /// Enable or disable RL agent control of quoting actions (builder).
    pub fn with_rl_enabled(mut self, enabled: bool) -> Self {
        self.rl_enabled = enabled;
        self
    }

    /// Enable or disable RL agent control of quoting actions (setter).
    pub fn set_rl_enabled(&mut self, enabled: bool) {
        self.rl_enabled = enabled;
    }

    /// Disable Binance-dependent signals (lead-lag, cross-venue).
    /// Call when no Binance feed is available for the asset to prevent
    /// permanent signal staleness widening.
    pub fn disable_binance_signals(&mut self) {
        self.stochastic.signal_integrator.disable_binance_signals();
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

    /// Load a paper trader's Q-table as a discounted prior for the live RL agent.
    ///
    /// Returns the number of states loaded.
    pub fn load_paper_rl_prior(
        &mut self,
        rl_checkpoint: &checkpoint::types::RLCheckpoint,
        weight: f64,
    ) -> usize {
        let mut paper_agent = learning::QLearningAgent::default();
        paper_agent.restore_from_checkpoint(rl_checkpoint);
        let paper_q = paper_agent.export_q_table();
        let n_states = paper_q.len();
        self.stochastic.rl_agent.import_q_table_as_prior(&paper_q, weight);
        n_states
    }

    /// Assemble a checkpoint bundle from current state.
    fn assemble_checkpoint_bundle(&self) -> checkpoint::CheckpointBundle {
        let (vol_filter, informed_flow, fill_rate, kappa_own, kappa_bid, kappa_ask, momentum) =
            self.estimator.to_checkpoint();

        checkpoint::CheckpointBundle {
            metadata: checkpoint::CheckpointMetadata {
                version: 1,
                timestamp_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
                asset: self.config.asset.to_string(),
                session_duration_s: self.session_start_time.elapsed().as_secs_f64(),
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
                }
            },
            ensemble_weights: checkpoint::EnsembleWeightsCheckpoint {
                model_weights: self.learning.ensemble_weights(),
                total_updates: self.learning.ensemble_total_updates(),
            },
            rl_q_table: self.stochastic.rl_agent.to_checkpoint(),
            kill_switch: self.safety.kill_switch.to_checkpoint(),
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
        self.stochastic
            .rl_agent
            .restore_from_checkpoint(&bundle.rl_q_table);
        self.safety
            .kill_switch
            .restore_from_checkpoint(&bundle.kill_switch);
    }

    // =========================================================================
    // Prior Transfer Protocol
    // =========================================================================
}

/// φ: S → P — extract learned priors.
impl<S: QuotingStrategy, Env: TradingEnvironment> checkpoint::PriorExtract
    for MarketMaker<S, Env>
{
    fn extract_prior(&self) -> checkpoint::CheckpointBundle {
        self.assemble_checkpoint_bundle()
    }
}

/// ψ: P × S → S — inject priors with configurable blending.
impl<S: QuotingStrategy, Env: TradingEnvironment> checkpoint::PriorInject
    for MarketMaker<S, Env>
{
    fn inject_prior(
        &mut self,
        prior: &checkpoint::CheckpointBundle,
        config: &checkpoint::InjectionConfig,
    ) -> usize {
        use tracing::{info, warn};

        // Validate asset match.
        if config.require_asset_match && *prior.metadata.asset != *self.config.asset {
            warn!(
                prior_asset = %prior.metadata.asset,
                our_asset = %self.config.asset,
                "Prior asset mismatch — skipping injection"
            );
            return 0;
        }

        // Validate age.
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let age_s = (now_ms.saturating_sub(prior.metadata.timestamp_ms)) as f64 / 1000.0;
        if age_s > config.max_prior_age_s {
            warn!(
                age_s = age_s,
                max_age_s = config.max_prior_age_s,
                "Prior too old — skipping injection"
            );
            return 0;
        }

        // Inject all non-RL, non-kill-switch components via restore_from_bundle.
        // Create a modified bundle that zeroes out excluded components.
        let mut injection_bundle = prior.clone();
        if config.skip_kill_switch {
            injection_bundle.kill_switch = checkpoint::KillSwitchCheckpoint::default();
        }

        // Restore all standard components.
        self.restore_from_bundle(&injection_bundle);

        // RL Q-table injection with configurable blend weight.
        let rl_states = if config.skip_rl {
            0
        } else {
            self.load_paper_rl_prior(&prior.rl_q_table, config.rl_blend_weight)
        };

        info!(
            asset = %prior.metadata.asset,
            prior_age_s = %format!("{:.0}", age_s),
            session_duration_s = %format!("{:.0}", prior.metadata.session_duration_s),
            rl_states = rl_states,
            rl_weight = config.rl_blend_weight,
            skip_kill_switch = config.skip_kill_switch,
            "Prior injected successfully"
        );

        rl_states
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

        // When user explicitly specified max_position (CLI/TOML), enforce as hard ceiling.
        // Otherwise, let dynamic margin-based effective_max_position be the sole limit.
        let risk_max_position = if self.config.max_position_user_specified {
            self.effective_max_position.min(self.config.max_position)
        } else {
            self.effective_max_position
        };
        risk::RiskState::new(
            pnl_summary.total_pnl,
            pnl_summary.peak_pnl,
            position,
            risk_max_position, // min(effective, config) — user config is hard constraint
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
            max_ladder_one_side_contracts: self.safety.kill_switch.state().max_ladder_one_side_contracts,
            // Preserve initial position from startup for runaway exemption
            initial_position: self.safety.kill_switch.state().initial_position,
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
                    .with_min_peak(config.min_peak_for_drawdown),
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
            let oldest_pos = self.position_history.front().map(|&(_, p)| p).unwrap_or(current_pos);
            let max_pos = self.effective_max_position.max(0.001); // avoid div by zero
            self.position_velocity_1m = (current_pos - oldest_pos).abs() / max_pos;
        }
    }
}
