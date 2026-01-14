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

use hyperliquid_rust_sdk::{
    BaseUrl, EstimatorConfig, InfoClient, Message, ParameterEstimator, Side, Subscription,
};
use tokio::sync::mpsc::unbounded_channel;

use hyperliquid_rust_sdk::market_maker::infra::metrics::dashboard::{
    CalibrationState, DashboardState, LiveQuotes, PnLAttribution, RegimeState,
    classify_regime, compute_regime_probabilities,
};
use hyperliquid_rust_sdk::market_maker::simulation::{
    CalibrationAnalyzer, FillSimulator, FillSimulatorConfig, MarketStateSnapshot, ModelPredictions,
    OutcomeTracker, PredictionLogger, SimulatedFill, SimulationExecutor,
};

use hyperliquid_rust_sdk::{Ladder, LadderLevel, MarketParams, OrderExecutor};

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
}

// ============================================================================
// Simulation State
// ============================================================================

/// Complete simulation state
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

    /// Parameter estimator
    estimator: ParameterEstimator,
    /// Simulated inventory
    inventory: f64,
    /// Quote cycle counter
    cycle_count: u64,

    /// Configuration
    gamma: f64,
    target_spread_bps: f64,
    ladder_levels: usize,
    target_liquidity: f64,
}

impl SimulationState {
    fn new(asset: String, gamma: f64, target_spread_bps: f64, ladder_levels: usize, target_liquidity: f64) -> Self {
        Self {
            asset,
            mid_price: 0.0,
            best_bid: 0.0,
            best_ask: 0.0,
            bid_levels: Vec::new(),
            ask_levels: Vec::new(),
            estimator: ParameterEstimator::new(EstimatorConfig::default()),
            inventory: 0.0,
            cycle_count: 0,
            gamma,
            target_spread_bps,
            ladder_levels,
            target_liquidity,
        }
    }

    /// Update state from all mids message
    fn update_mid(&mut self, mid: f64) {
        if mid > 0.0 {
            self.mid_price = mid;
        }
    }

    /// Update state from L2 book
    fn update_book(&mut self, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) {
        self.bid_levels = bids;
        self.ask_levels = asks;

        if let Some((price, _)) = self.bid_levels.first() {
            self.best_bid = *price;
        }
        if let Some((price, _)) = self.ask_levels.first() {
            self.best_ask = *price;
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

    /// Generate quotes for this cycle
    fn generate_quotes(&mut self) -> Option<Ladder> {
        if self.mid_price <= 0.0 {
            return None;
        }

        self.cycle_count += 1;

        // Get estimated parameters
        let sigma = self.estimator.sigma_clean().max(0.00001);
        let kappa = self.estimator.kappa().max(100.0);

        // GLFT optimal half-spread: δ* = (1/γ) * ln(1 + γ/κ) + fee
        let maker_fee = 0.00015; // 1.5 bps
        let glft_half_spread = (1.0 / self.gamma) * (1.0 + self.gamma / kappa).ln() + maker_fee;

        // Apply floor
        let min_half_spread = self.target_spread_bps / 10000.0 / 2.0;
        let half_spread = glft_half_spread.max(min_half_spread);

        // Inventory skew: shift quotes when carrying inventory
        let inventory_skew = self.gamma * sigma * sigma * self.inventory * 10.0;

        // Generate ladder levels
        let mut bids = Vec::new();
        let mut asks = Vec::new();

        let level_spacing_bps = 3.0; // 3 bps between levels

        for level in 0..self.ladder_levels {
            let depth_offset = (level as f64 * level_spacing_bps) / 10000.0;

            // Bid: below mid
            let bid_depth = half_spread + depth_offset + inventory_skew.max(0.0);
            let bid_price = self.mid_price * (1.0 - bid_depth);
            let bid_size = self.target_liquidity * (0.8_f64.powi(level as i32)); // Decay size at further levels

            bids.push(LadderLevel {
                price: bid_price,
                size: bid_size,
                depth_bps: bid_depth * 10000.0,
            });

            // Ask: above mid
            let ask_depth = half_spread + depth_offset - inventory_skew.min(0.0);
            let ask_price = self.mid_price * (1.0 + ask_depth);
            let ask_size = self.target_liquidity * (0.8_f64.powi(level as i32));

            asks.push(LadderLevel {
                price: ask_price,
                size: ask_size,
                depth_bps: ask_depth * 10000.0,
            });
        }

        Some(Ladder { bids: bids.into(), asks: asks.into() })
    }

    /// Build market params for prediction logging
    fn build_market_params(&self) -> MarketParams {
        let mut params = MarketParams::default();

        params.microprice = self.mid_price;
        params.market_mid = self.mid_price;
        params.sigma = self.estimator.sigma_clean();
        params.sigma_total = self.estimator.sigma_total();
        params.sigma_effective = self.estimator.sigma_effective();
        params.kappa = self.estimator.kappa();
        params.kappa_robust = self.estimator.kappa();
        params.jump_ratio = self.estimator.jump_ratio();
        params.book_imbalance = self.compute_book_imbalance();
        params.market_spread_bps = if self.best_ask > 0.0 && self.best_bid > 0.0 {
            (self.best_ask - self.best_bid) / self.mid_price * 10000.0
        } else {
            0.0
        };

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

    /// Update inventory from simulated fill
    fn on_simulated_fill(&mut self, fill: &SimulatedFill) {
        let direction = if fill.side == Side::Buy { 1.0 } else { -1.0 };
        self.inventory += direction * fill.fill_size;
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
fn create_market_trade(price: f64, size: f64, is_buy: bool) -> hyperliquid_rust_sdk::market_maker::simulation::fill_sim::MarketTrade {
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
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or(&cli.log_level)
    ).init();

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
    let dashboard_state: Arc<RwLock<DashboardState>> = Arc::new(RwLock::new(DashboardState::default()));

    if cli.dashboard {
        let dashboard_state_for_server = dashboard_state.clone();
        let metrics_port = cli.metrics_port;

        tokio::spawn(async move {
            // CORS layer to allow dashboard at localhost:3000 to fetch from localhost:8080
            let cors = CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any);

            let app = Router::new()
                .route(
                    "/api/dashboard",
                    get(move || {
                        let state = dashboard_state_for_server.clone();
                        async move {
                            let state_guard = state.read().unwrap();
                            Json(state_guard.clone())
                        }
                    }),
                )
                .layer(cors);

            let addr = format!("127.0.0.1:{}", metrics_port);
            info!(
                port = metrics_port,
                "Dashboard API server starting at http://{}/api/dashboard", addr
            );

            if let Ok(listener) = tokio::net::TcpListener::bind(&addr).await {
                if let Err(e) = axum::serve(listener, app).await {
                    warn!(error = %e, "Dashboard server error");
                }
            } else {
                warn!(port = metrics_port, "Failed to bind dashboard port");
            }
        });

        info!(
            "Dashboard enabled at http://localhost:3000/mm-dashboard-fixed.html"
        );
        info!(
            "API endpoint at http://localhost:{}/api/dashboard",
            cli.metrics_port
        );
    }

    info!("Simulation running for {} seconds...", cli.duration);

    let mut last_quote_time = Instant::now();
    let quote_interval = Duration::from_millis(100); // Quote every 100ms

    while !shutdown.load(Ordering::SeqCst) && start_time.elapsed() < duration {
        tokio::select! {
            Some(msg) = rx.recv() => {
                match msg {
                    MarketDataMessage::Mid(mid) => {
                        state.update_mid(mid);
                        executor.update_mid(mid);
                    }
                    MarketDataMessage::Book { bids, asks } => {
                        state.update_book(bids, asks);
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
                        }
                    }
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(10)) => {
                // Generate quotes on interval
                if last_quote_time.elapsed() >= quote_interval {
                    last_quote_time = Instant::now();

                    if let Some(ladder) = state.generate_quotes() {
                        stats.quote_cycles += 1;

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
                            state.target_spread_bps / 2.0,
                            state.target_spread_bps / 2.0,
                            0.0,
                        );

                        // Log prediction
                        prediction_logger.log_prediction(market_snapshot, predictions);

                        // Place simulated orders
                        for level in &ladder.bids {
                            executor.place_order(
                                &cli.asset,
                                level.price,
                                level.size,
                                true,
                                None,
                                true,
                            ).await;
                        }
                        for level in &ladder.asks {
                            executor.place_order(
                                &cli.asset,
                                level.price,
                                level.size,
                                false,
                                None,
                                true,
                            ).await;
                        }

                        // Cancel old orders (simulate quote refresh)
                        let active_orders = executor.get_active_orders();
                        for order in active_orders {
                            let age_ms = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap()
                                .as_millis() as u64
                                - order.created_at_ns / 1_000_000;

                            if age_ms > 200 {
                                executor.cancel_order(&cli.asset, order.oid).await;
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

                            let mut dashboard = dashboard_state.write().unwrap();
                            *dashboard = DashboardState {
                                quotes: LiveQuotes {
                                    mid: state.mid_price,
                                    spread_bps,
                                    inventory: state.inventory,
                                    regime: regime.clone(),
                                    kappa,
                                    gamma: cli.gamma,
                                    fill_prob: 0.2, // Estimated
                                    adverse_prob: 0.1, // Estimated
                                },
                                pnl: PnLAttribution {
                                    spread_capture: outcome_summary.spread_capture,
                                    adverse_selection: outcome_summary.adverse_selection,
                                    inventory_cost: outcome_summary.inventory_cost,
                                    fees: outcome_summary.fee_cost,
                                    total: outcome_summary.total_pnl,
                                },
                                regime: RegimeState {
                                    current: regime,
                                    probabilities: regime_probs,
                                    history: Vec::new(), // Could add history tracking
                                },
                                fills: Vec::new(), // Could add fill history
                                calibration: CalibrationState::default(),
                                signals: Vec::new(),
                                timestamp_ms: chrono::Utc::now().timestamp_millis(),
                            };
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
    println!("  Total Notional:   ${:.2}", exec_stats.total_notional_placed);

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
                    tx.send(MarketDataMessage::Mid(mid)).await.ok();
                }
            }
        }
    }

    Ok(())
}

async fn subscribe_l2_book(
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

            tx.send(MarketDataMessage::Book { bids, asks }).await.ok();
        }
    }

    Ok(())
}

async fn subscribe_trades(
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
                    tx.send(MarketDataMessage::Trade {
                        price: px,
                        size: sz,
                        is_buy,
                    })
                    .await
                    .ok();
                }
            }
        }
    }

    Ok(())
}
