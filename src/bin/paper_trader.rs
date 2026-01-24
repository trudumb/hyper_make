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
    BaseUrl, EstimatorConfig, InfoClient, Message, ParameterEstimator, Side, Subscription,
};
use tokio::sync::mpsc::unbounded_channel;

use hyperliquid_rust_sdk::market_maker::infra::metrics::dashboard::{
    classify_regime,
    compute_regime_probabilities,
    BookLevel,
    BookSnapshot,
    CalibrationState,
    DashboardState,
    FeatureCorrelationState,
    // Feature health visualization
    FeatureHealthState,
    FeatureValidationState,
    FillRecord,
    InteractionSignalState,
    LagAnalysisState,
    LiveQuotes,
    PnLAttribution,
    PricePoint,
    QuoteFillStats,
    QuoteSnapshot,
    RegimeSnapshot,
    RegimeState,
    SignalDecayState,
    SignalHealthInfo,
    SpreadBucket,
};
use hyperliquid_rust_sdk::market_maker::simulation::{
    CalibrationAnalyzer, FillSimulator, FillSimulatorConfig, MarketStateSnapshot, ModelPredictions,
    OutcomeTracker, PredictionLogger, SimulatedFill, SimulationExecutor,
};
use std::collections::{HashMap, VecDeque};

use hyperliquid_rust_sdk::{
    Ladder, LadderConfig, LadderStrategy, MarketParams, OrderExecutor, QuoteConfig, RiskConfig,
};

// Adaptive GLFT components
use hyperliquid_rust_sdk::market_maker::adaptive::{
    AdaptiveBayesianConfig, AdaptiveSpreadCalculator,
};
use hyperliquid_rust_sdk::market_maker::process_models::{HJBConfig, HJBInventoryController};

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
    #[arg(long, default_value = "60")]
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

/// Complete simulation state using production LadderStrategy
struct SimulationState {
    /// Asset being simulated
    #[allow(dead_code)]
    asset: String,
    /// Current mid price
    mid_price: f64,
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

    /// Configuration
    max_position: f64,
    target_liquidity: f64,
    /// Price decimals (BTC=1, ETH=2, etc.)
    decimals: u32,
    /// Size decimals
    sz_decimals: u32,

    // === Adaptive GLFT Components ===
    /// Adaptive Bayesian spread calculator (learned kappa, gamma, floor)
    adaptive_spreads: AdaptiveSpreadCalculator,
    /// HJB inventory controller (optimal skew from HJB solution)
    hjb_controller: HJBInventoryController,
}

impl SimulationState {
    fn new(
        asset: String,
        gamma: f64,
        target_spread_bps: f64,
        ladder_levels: usize,
        target_liquidity: f64,
    ) -> Self {
        // Create production-style RiskConfig
        let risk_config = RiskConfig {
            gamma_base: gamma.clamp(0.01, 10.0),
            min_spread_floor: target_spread_bps / 10_000.0 / 2.0, // Convert bps to fraction, half-spread
            ..Default::default()
        };

        // Create LadderConfig with specified levels
        let ladder_config = LadderConfig {
            num_levels: ladder_levels,
            ..Default::default()
        };

        // Create the real LadderStrategy
        let strategy = LadderStrategy::with_config(risk_config, ladder_config);

        // Determine decimals based on asset
        let (decimals, sz_decimals) = match asset.as_str() {
            "BTC" => (1, 4), // $0.1 tick, 0.0001 BTC
            "ETH" => (2, 3), // $0.01 tick, 0.001 ETH
            "SOL" => (3, 2), // $0.001 tick, 0.01 SOL
            _ => (2, 3),     // Default
        };

        // Create adaptive components with defaults
        let adaptive_spreads = AdaptiveSpreadCalculator::new(AdaptiveBayesianConfig::default());
        let mut hjb_controller = HJBInventoryController::new(HJBConfig::default());
        hjb_controller.start_session(); // Start HJB session timing

        Self {
            asset,
            mid_price: 0.0,
            best_bid: 0.0,
            best_ask: 0.0,
            bid_levels: Vec::new(),
            ask_levels: Vec::new(),
            estimator: ParameterEstimator::new(EstimatorConfig::default()),
            strategy,
            inventory: 0.0,
            cycle_count: 0,
            max_position: 1.0, // 1 BTC max position for simulation
            target_liquidity,
            decimals,
            sz_decimals,
            // Adaptive components
            adaptive_spreads,
            hjb_controller,
        }
    }

    /// Update state from all mids message
    fn update_mid(&mut self, mid: f64) {
        if mid > 0.0 {
            self.mid_price = mid;

            // Update HJB controller with current volatility estimate
            let sigma = self.estimator.sigma_clean().max(0.00001);
            self.hjb_controller.update_sigma(sigma);
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

        // Update adaptive kappa from book depth
        if self.mid_price > 0.0 {
            self.adaptive_spreads
                .on_l2_update(&bids, &asks, self.mid_price);
        }
    }

    /// Update from trade
    fn on_trade(&mut self, price: f64, size: f64, is_buy: bool) {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        self.estimator.on_trade(now_ms, price, size, Some(is_buy));
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
            min_notional: 10.0, // $10 minimum
        };

        // Build MarketParams from estimator (same as production)
        let market_params = self.build_market_params();

        // Update strategy's fill model with current volatility
        let sigma = market_params.sigma.max(0.00001);
        let tau = 10.0; // Typical quote lifetime
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

    /// Build market params from estimator (populates all fields needed by strategy)
    fn build_market_params(&self) -> MarketParams {
        let mut params = MarketParams::default();

        // Core pricing
        params.microprice = self.mid_price;
        params.market_mid = self.mid_price;

        // Volatility from estimator
        params.sigma = self.estimator.sigma_clean().max(0.00001);
        params.sigma_total = self.estimator.sigma_total();
        params.sigma_effective = self.estimator.sigma_effective();

        // Kappa from estimator - use robust path
        let kappa = self.estimator.kappa().max(100.0);
        params.kappa = kappa;
        params.kappa_robust = kappa;
        params.kappa_bid = kappa;
        params.kappa_ask = kappa;
        params.use_kappa_robust = true; // Use robust path

        // Jump/momentum
        params.jump_ratio = self.estimator.jump_ratio();
        params.momentum_bps = self.estimator.momentum_bps();

        // Book state
        params.book_imbalance = self.compute_book_imbalance();
        params.market_spread_bps = if self.best_ask > 0.0 && self.best_bid > 0.0 {
            (self.best_ask - self.best_bid) / self.mid_price * 10000.0
        } else {
            10.0 // Default 10 bps if no book
        };

        // Arrival intensity (for GLFT time horizon)
        params.arrival_intensity = 0.5; // Conservative default

        // === ENABLE FULL ADAPTIVE GLFT ===
        // This is the key change: enable adaptive features to use learned parameters
        params.use_adaptive_spreads = true;
        params.use_hjb_skew = true;
        params.use_kalman_filter = false; // Not wired up yet
        params.use_dynamic_bounds = false;

        // Populate adaptive values from our components
        params.adaptive_gamma = self.adaptive_spreads.current_gamma();
        params.adaptive_kappa = self.adaptive_spreads.current_kappa();
        params.adaptive_spread_floor = self.adaptive_spreads.current_floor();
        params.adaptive_can_estimate = self.adaptive_spreads.can_provide_estimates();
        params.adaptive_warmed_up = self.adaptive_spreads.is_warmed_up();
        params.adaptive_warmup_progress = self.adaptive_spreads.warmup_progress();

        // HJB optimal skew for inventory control
        params.hjb_optimal_skew = self
            .hjb_controller
            .optimal_skew(self.inventory, self.max_position);

        // No cascade protection in paper trading
        params.should_pull_quotes = false;
        params.cascade_size_factor = 1.0;
        params.tail_risk_multiplier = 1.0;

        params
    }

    fn compute_book_imbalance(&self) -> f64 {
        let bid_size: f64 = self.bid_levels.iter().take(3).map(|(_, s)| s).sum();
        let ask_size: f64 = self.ask_levels.iter().take(3).map(|(_, s)| s).sum();
        let total = bid_size + ask_size;
        if total > 0.0 {
            (bid_size - ask_size) / total
        } else {
            0.0
        }
    }

    /// Update inventory from simulated fill and record for Bayesian learning
    fn on_simulated_fill(&mut self, fill: &SimulatedFill) {
        let direction = if fill.side == Side::Buy { 1.0 } else { -1.0 };
        self.inventory += direction * fill.fill_size;

        // Record fill for Bayesian learning in the strategy
        // Compute depth in bps from mid
        let depth_bps = if self.mid_price > 0.0 {
            ((fill.fill_price - self.mid_price).abs() / self.mid_price) * 10_000.0
        } else {
            5.0 // Default 5 bps if no mid
        };
        self.strategy.record_fill_observation(depth_bps, true);
    }

    /// Record cancelled (non-filled) order for Bayesian learning
    #[allow(dead_code)]
    fn on_order_cancelled(&mut self, order_price: f64) {
        if self.mid_price > 0.0 {
            let depth_bps = ((order_price - self.mid_price).abs() / self.mid_price) * 10_000.0;
            self.strategy.record_fill_observation(depth_bps, false);
        }
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

        output.push_str("\n");
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
    let mut fill_simulator = FillSimulator::new(
        executor.clone(),
        FillSimulatorConfig {
            touch_fill_probability: 0.25,
            queue_position_factor: 0.5,
            ..Default::default()
        },
    );

    // Create prediction logger
    let prediction_log_path = PathBuf::from(&cli.output_dir).join("predictions.jsonl");
    let mut prediction_logger = PredictionLogger::new(prediction_log_path)?;

    // Create calibration analyzer
    let calibration_analyzer = CalibrationAnalyzer::new(20);

    // Create outcome tracker
    let mut outcome_tracker = OutcomeTracker::new(0.00015); // 1.5 bps maker fee

    // Initialize simulation state
    let mut state = SimulationState::new(
        cli.asset.clone(),
        cli.gamma,
        cli.target_spread_bps,
        cli.ladder_levels,
        cli.target_liquidity,
    );

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

    // Run simulation loop
    let start_time = Instant::now();
    let duration = Duration::from_secs(cli.duration);
    let shutdown = Arc::new(AtomicBool::new(false));

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

            let addr = format!("127.0.0.1:{}", metrics_port);
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

    while !shutdown.load(Ordering::SeqCst) && start_time.elapsed() < duration {
        tokio::select! {
            Some(msg) = rx.recv() => {
                // Reset watchdog on any message
                last_data_received = Instant::now();
                if stale_warning_logged {
                    info!("Market data stream recovered");
                    stale_warning_logged = false;
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

                            let fill_record = FillRecord {
                                time: now.format("%H:%M:%S").to_string(),
                                timestamp_ms: chrono::Utc::now().timestamp_millis(),
                                pnl: fill_pnl,
                                cum_pnl: cumulative_pnl,
                                side: if fill.side == Side::Buy { "BID".to_string() } else { "ASK".to_string() },
                                adverse_selection: "0.0".to_string(),
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
            _ = tokio::time::sleep(Duration::from_millis(10)) => {
                // Watchdog: check for stale market data
                if last_data_received.elapsed() > stale_data_threshold && !stale_warning_logged {
                    warn!(
                        "No market data received for {:?}. WebSocket subscriptions may have disconnected.",
                        last_data_received.elapsed()
                    );
                    stale_warning_logged = true;
                }

                // Generate quotes on interval
                if last_quote_time.elapsed() >= quote_interval {
                    last_quote_time = Instant::now();

                    if let Some(ladder) = state.generate_quotes() {
                        stats.quote_cycles += 1;

                        // Queue-aware order management:
                        // Only cancel/replace orders if price moved beyond threshold
                        // This preserves queue position for realistic fill simulation
                        let price_threshold_bps = 2.0; // Only refresh if price moved 2+ bps
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
                        );

                        // Log prediction
                        prediction_logger.log_prediction(market_snapshot, predictions);

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
            }
        }
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
            let bids: Vec<(f64, f64)> = if book.data.levels.len() > 0 {
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
