//! Latent State Parameter Estimator
//!
//! A diagnostic binary for estimating latent market state parameters:
//! - Stochastic volatility with regime switching (particle filter)
//! - Trade flow decomposition (informed/noise/forced)
//! - Conditional fill rate modeling
//! - Adverse selection decomposition
//! - Joint dynamics and edge surface quantification
//!
//! Usage:
//! ```bash
//! cargo run --bin parameter_estimator -- --asset BTC --duration 4h --report-interval 5m
//! ```

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::File;
use std::io::Write as IoWrite;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::unbounded_channel;
use tracing::{debug, info, warn};
use tracing_subscriber::FmtSubscriber;

use hyperliquid_rust_sdk::{BaseUrl, InfoClient, Message, Subscription};

// Import estimator components (re-exported from market_maker)
use hyperliquid_rust_sdk::market_maker::{
    ASDecomposition, ASFillInfo, FillObservation, FillRateMarketState, FillRateModel,
    InformedFlowEstimator, TradeFeatures, VolFilterConfig, VolatilityFilter,
};

// Import latent state components
use hyperliquid_rust_sdk::market_maker::latent::{
    EdgeObservation, EdgeSurface, JointDynamics, JointObservation, MarketCondition,
};

// ============================================================================
// CLI Arguments
// ============================================================================

#[derive(Parser)]
#[command(name = "parameter_estimator")]
#[command(version, about = "Latent State Parameter Estimator for Market Making", long_about = None)]
struct Cli {
    /// Asset to analyze (e.g., BTC, ETH, SOL)
    #[arg(short, long)]
    asset: String,

    /// Duration to run (e.g., "1h", "4h", "24h")
    #[arg(long, default_value = "24h")]
    duration: String,

    /// Output file path for JSON export
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Report interval (e.g., "1m", "5m", "15m")
    #[arg(long, default_value = "5m")]
    report_interval: String,

    /// Network: mainnet, testnet
    #[arg(long, default_value = "mainnet")]
    network: String,

    /// HIP-3 DEX name (optional, e.g., "hyna")
    #[arg(long)]
    dex: Option<String>,

    /// Log level: trace, debug, info, warn, error
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Number of particles for volatility filter
    #[arg(long, default_value = "500")]
    n_particles: usize,

    /// Verbose output (print every message)
    #[arg(long)]
    verbose: bool,
}

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LatentEstimatorConfig {
    /// Asset being analyzed
    pub asset: String,

    /// Number of particles for volatility filter
    pub n_particles: usize,

    /// Report interval in milliseconds
    pub report_interval_ms: u64,

    /// Duration in milliseconds
    pub duration_ms: u64,

    /// Volume bucket window size (trades)
    pub volume_bucket_window: usize,

    /// Minimum trades before warmup complete
    pub min_warmup_trades: usize,

    /// Minimum L2 updates before warmup complete
    pub min_warmup_l2_updates: usize,
}

impl Default for LatentEstimatorConfig {
    fn default() -> Self {
        Self {
            asset: "BTC".to_string(),
            n_particles: 500,
            report_interval_ms: 5 * 60 * 1000, // 5 minutes
            duration_ms: 24 * 60 * 60 * 1000,  // 24 hours
            volume_bucket_window: 100,
            min_warmup_trades: 50,
            min_warmup_l2_updates: 20,
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
}

// ============================================================================
// Statistics Tracking
// ============================================================================

#[derive(Debug, Default)]
pub struct EstimatorStats {
    pub trades_processed: u64,
    pub books_processed: u64,
    pub mids_processed: u64,
    pub start_time: Option<Instant>,
    pub last_report_time: Option<Instant>,
    pub last_mid: Option<f64>,
    pub last_spread_bps: Option<f64>,
}

impl EstimatorStats {
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

    pub fn trades_per_minute(&self) -> f64 {
        let mins = self.uptime_secs() / 60.0;
        if mins > 0.0 {
            self.trades_processed as f64 / mins
        } else {
            0.0
        }
    }
}

// ============================================================================
// JSON Export Types
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct ParameterEstimateReport {
    pub timestamp_ms: u64,
    pub asset: String,
    pub uptime_secs: f64,
    pub trades_processed: u64,
    pub volatility: VolatilityReport,
    pub flow: FlowReport,
    pub fill_rate: FillRateReport,
    pub adverse_selection: ASReport,
    pub joint_dynamics: JointDynamicsReport,
    pub edge: EdgeReport,
}

#[derive(Debug, Clone, Serialize)]
pub struct VolatilityReport {
    pub sigma_bps_per_sqrt_s: f64,
    pub regime: String,
    pub regime_confidence: f64,
    pub credible_interval: (f64, f64),
    pub is_warmed_up: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct FlowReport {
    pub p_informed: f64,
    pub p_noise: f64,
    pub p_forced: f64,
    pub confidence: f64,
    pub is_warmed_up: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct FillRateReport {
    pub lambda_0: f64,
    pub delta_char_bps: f64,
    pub fill_rate_at_8bps: f64,
    pub is_warmed_up: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ASReport {
    pub total_bps: f64,
    pub permanent_bps: f64,
    pub temporary_bps: f64,
    pub timing_bps: f64,
    pub fills_measured: usize,
    pub is_warmed_up: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct JointDynamicsReport {
    pub sigma_kappa_corr: f64,
    pub sigma_as_corr: f64,
    pub kappa_informed_corr: f64,
    pub regime: u8,
    pub is_toxic: bool,
    pub is_warmed_up: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct EdgeReport {
    pub mean_edge_bps: f64,
    pub cells_with_data: usize,
    pub should_quote: bool,
    pub current_edge_bps: f64,
    pub optimal_spread_bps: f64,
    pub is_warmed_up: bool,
}

// ============================================================================
// Main Estimator Structure
// ============================================================================

pub struct LatentStateEstimator {
    config: LatentEstimatorConfig,
    buffers: ObservationBuffers,
    stats: EstimatorStats,

    // Phase 2: Volatility Filter
    volatility_filter: VolatilityFilter,
    last_price: Option<f64>,

    // Phase 3: Flow Decomposition
    flow_estimator: InformedFlowEstimator,
    last_trade_time_ms: u64,

    // Phase 4: Fill Rate Model
    fill_rate_model: FillRateModel,

    // Phase 5: Adverse Selection
    as_decomposition: ASDecomposition,

    // Phase 7: Joint Dynamics
    joint_dynamics: JointDynamics,

    // Phase 6: Edge Surface
    edge_surface: EdgeSurface,

    warmed_up: bool,
}

impl LatentStateEstimator {
    pub fn new(config: LatentEstimatorConfig) -> Self {
        // Configure volatility filter
        let vol_config = VolFilterConfig {
            n_particles: config.n_particles,
            ..VolFilterConfig::default()
        };

        Self {
            buffers: ObservationBuffers::new(config.volume_bucket_window * 10),
            volatility_filter: VolatilityFilter::new(vol_config),
            flow_estimator: InformedFlowEstimator::default_config(),
            fill_rate_model: FillRateModel::default_config(),
            as_decomposition: ASDecomposition::default_config(),
            joint_dynamics: JointDynamics::default_config(),
            edge_surface: EdgeSurface::default_config(),
            config,
            stats: EstimatorStats::new(),
            last_price: None,
            last_trade_time_ms: 0,
            warmed_up: false,
        }
    }

    pub fn on_trade(&mut self, timestamp_ms: u64, price: f64, size: f64, is_buy: bool) {
        // Store observation
        let obs = TradeObservation {
            timestamp_ms,
            price,
            size,
            is_buy,
        };
        self.buffers.add_trade(obs);
        self.stats.trades_processed += 1;

        // Update volatility filter with returns
        if let Some(last_price) = self.last_price {
            if last_price > 0.0 {
                let ret = (price / last_price).ln();
                let dt = if self.last_trade_time_ms > 0 {
                    ((timestamp_ms - self.last_trade_time_ms) as f64 / 1000.0).max(0.001)
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
            1000 // Default 1 second
        };
        self.last_trade_time_ms = timestamp_ms;

        // Update flow estimator
        let features = TradeFeatures {
            size,
            inter_arrival_ms,
            price_impact_bps: 0.0, // We don't measure impact immediately
            book_imbalance: self.buffers.book_imbalance(),
            is_buy,
            timestamp_ms,
        };
        self.flow_estimator.on_trade(&features);

        // Update joint dynamics
        let decomp = self.flow_estimator.decomposition();
        let sigma = self.volatility_filter.sigma_bps_per_sqrt_s();

        let joint_obs = JointObservation {
            sigma,
            kappa: 100.0, // Default kappa, would come from book depth analysis
            p_informed: decomp.p_informed,
            as_bps: self.as_decomposition.total_as_bps(),
            flow_momentum: self.buffers.flow_imbalance(),
            timestamp_ms,
        };
        self.joint_dynamics.update(&joint_obs);

        // Check warmup
        if !self.warmed_up {
            self.check_warmup();
        }
    }

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
    }

    pub fn on_mid(&mut self, timestamp_ms: u64, mid: f64) {
        self.buffers.add_mid(timestamp_ms, mid);
        self.stats.mids_processed += 1;
        self.stats.last_mid = Some(mid);

        // Update AS decomposition with price updates for pending measurements
        self.as_decomposition.on_price_update(mid, timestamp_ms);
    }

    /// Record a fill observation (for testing/simulated fills)
    pub fn on_fill(&mut self, timestamp_ms: u64, price: f64, size: f64, side_is_buy: bool) {
        // Get current market state
        let sigma = self.volatility_filter.sigma_bps_per_sqrt_s();
        let regime = self.volatility_filter.regime() as u8;
        let hour_utc = ((timestamp_ms / 3600000) % 24) as u8;
        let flow = self.buffers.flow_imbalance();

        // Record fill for AS decomposition
        let fill_info = ASFillInfo {
            fill_id: timestamp_ms, // Use timestamp as unique ID
            timestamp_ms,
            fill_mid: price,
            size,
            is_buy: side_is_buy,
            sigma_bps: Some(sigma),
            regime: Some(regime),
        };
        self.as_decomposition.on_fill(&fill_info);

        let condition = MarketCondition::from_state(sigma, regime, hour_utc, flow);

        // Record fill rate observation
        let market_state = FillRateMarketState {
            sigma_bps: sigma,
            spread_bps: self.stats.last_spread_bps.unwrap_or(8.0),
            book_imbalance: self.buffers.book_imbalance(),
            flow_imbalance: flow,
            regime,
            hour_utc,
            is_bid: !side_is_buy,
        };
        let fill_obs = FillObservation {
            depth_bps: 8.0, // Assume 8 bps depth
            filled: true,
            duration_s: 1.0, // Default duration
            state: market_state,
            size,
        };
        self.fill_rate_model.observe(&fill_obs);

        // Record edge observation
        let edge_obs = EdgeObservation {
            condition,
            spread_bps: self.stats.last_spread_bps.unwrap_or(8.0),
            filled: true,
            realized_as_bps: self.as_decomposition.total_as_bps(),
            timestamp_ms,
        };
        self.edge_surface.observe(&edge_obs);
    }

    fn check_warmup(&mut self) {
        if self.stats.trades_processed >= self.config.min_warmup_trades as u64
            && self.stats.books_processed >= self.config.min_warmup_l2_updates as u64
            && self.volatility_filter.is_warmed_up()
        {
            self.warmed_up = true;
            info!(
                "Estimator warmed up after {} trades, {} L2 updates",
                self.stats.trades_processed, self.stats.books_processed
            );
        }
    }

    pub fn is_warmed_up(&self) -> bool {
        self.warmed_up
    }

    pub fn should_report(&mut self) -> bool {
        if let Some(last) = self.stats.last_report_time {
            if last.elapsed().as_millis() as u64 >= self.config.report_interval_ms {
                self.stats.last_report_time = Some(Instant::now());
                return true;
            }
        }
        false
    }

    /// Generate JSON report
    pub fn generate_report(&self) -> ParameterEstimateReport {
        let sigma = self.volatility_filter.sigma_bps_per_sqrt_s();
        let regime = self.volatility_filter.regime();
        let ci_raw = self.volatility_filter.sigma_credible_interval(0.95);
        let ci = (ci_raw.0 * 10000.0, ci_raw.1 * 10000.0); // Convert to bps

        let decomp = self.flow_estimator.decomposition();
        let correlations = self.joint_dynamics.correlations();
        let joint_state = self.joint_dynamics.current_state();
        let surface_stats = self.edge_surface.surface_statistics();

        // Get current condition for edge estimate
        let hour_utc = (get_current_timestamp_ms() / 3600000 % 24) as u8;
        let condition = MarketCondition::from_state(
            sigma,
            regime as u8,
            hour_utc,
            self.buffers.flow_imbalance() as f32 as f64,
        );
        let edge_estimate = self.edge_surface.edge_estimate(&condition);

        ParameterEstimateReport {
            timestamp_ms: get_current_timestamp_ms(),
            asset: self.config.asset.clone(),
            uptime_secs: self.stats.uptime_secs(),
            trades_processed: self.stats.trades_processed,
            volatility: VolatilityReport {
                sigma_bps_per_sqrt_s: sigma,
                regime: format!("{:?}", regime),
                regime_confidence: self.volatility_filter.regime_confidence(),
                credible_interval: ci,
                is_warmed_up: self.volatility_filter.is_warmed_up(),
            },
            flow: FlowReport {
                p_informed: decomp.p_informed,
                p_noise: decomp.p_noise,
                p_forced: decomp.p_forced,
                confidence: decomp.confidence,
                is_warmed_up: self.flow_estimator.is_warmed_up(),
            },
            fill_rate: FillRateReport {
                lambda_0: self.fill_rate_model.lambda_0().mean,
                delta_char_bps: self.fill_rate_model.delta_char().mean,
                fill_rate_at_8bps: self
                    .fill_rate_model
                    .fill_rate(8.0, &FillRateMarketState::default()),
                is_warmed_up: self.fill_rate_model.is_warmed_up(),
            },
            adverse_selection: ASReport {
                total_bps: self.as_decomposition.total_as_bps(),
                permanent_bps: self.as_decomposition.permanent_as_bps(),
                temporary_bps: self.as_decomposition.temporary_as_bps(),
                timing_bps: self.as_decomposition.timing_as_bps(),
                fills_measured: self.as_decomposition.fills_measured(),
                is_warmed_up: self.as_decomposition.is_warmed_up(),
            },
            joint_dynamics: JointDynamicsReport {
                sigma_kappa_corr: correlations.sigma_kappa,
                sigma_as_corr: correlations.sigma_as,
                kappa_informed_corr: correlations.kappa_p_informed,
                regime: joint_state.regime,
                is_toxic: correlations.is_toxic(),
                is_warmed_up: self.joint_dynamics.is_warmed_up(),
            },
            edge: EdgeReport {
                mean_edge_bps: surface_stats.mean_edge_bps,
                cells_with_data: surface_stats.cells_with_data,
                should_quote: self.edge_surface.should_quote(&condition),
                current_edge_bps: edge_estimate.edge_bps,
                optimal_spread_bps: edge_estimate.optimal_spread_bps,
                is_warmed_up: self.edge_surface.is_warmed_up(),
            },
        }
    }

    pub fn print_report(&self) {
        let report = self.generate_report();

        println!();
        println!("============================================================");
        println!("PARAMETER ESTIMATION REPORT");
        println!("============================================================");
        println!();
        println!("OBSERVATION STATISTICS:");
        println!("  Uptime: {:.1} minutes", report.uptime_secs / 60.0);
        println!("  Trades processed: {}", report.trades_processed);
        println!("  L2 books processed: {}", self.stats.books_processed);
        println!("  Mids processed: {}", self.stats.mids_processed);
        println!(
            "  Trade rate: {:.1} trades/min",
            self.stats.trades_per_minute()
        );
        println!("  Warmed up: {}", self.warmed_up);
        if let Some(mid) = self.stats.last_mid {
            println!("  Last mid: ${:.2}", mid);
        }
        println!();

        // Volatility
        println!("VOLATILITY:");
        if report.volatility.is_warmed_up {
            println!(
                "  Current: {:.2} bps/√s",
                report.volatility.sigma_bps_per_sqrt_s
            );
            println!(
                "  Regime: {} (P={:.2})",
                report.volatility.regime, report.volatility.regime_confidence
            );
            println!(
                "  95% CI: [{:.2}, {:.2}] bps/√s",
                report.volatility.credible_interval.0, report.volatility.credible_interval.1
            );
        } else {
            println!(
                "  Warming up... ({} observations)",
                self.volatility_filter.observation_count()
            );
        }
        println!();

        // Flow Decomposition
        println!("FLOW DECOMPOSITION:");
        if report.flow.is_warmed_up {
            println!("  P(informed): {:.1}%", report.flow.p_informed * 100.0);
            println!("  P(noise):    {:.1}%", report.flow.p_noise * 100.0);
            println!("  P(forced):   {:.1}%", report.flow.p_forced * 100.0);
            println!("  Confidence:  {:.2}", report.flow.confidence);
        } else {
            println!(
                "  Warming up... ({} observations)",
                self.flow_estimator.observation_count()
            );
        }
        println!();

        // Fill Rate Model
        println!("FILL RATE MODEL:");
        if report.fill_rate.is_warmed_up {
            println!("  λ_0 = {:.2} fills/min", report.fill_rate.lambda_0);
            println!("  δ_char = {:.2} bps", report.fill_rate.delta_char_bps);
            println!(
                "  Fill rate at 8 bps: {:.2} fills/min",
                report.fill_rate.fill_rate_at_8bps
            );
        } else {
            println!(
                "  Warming up... ({} observations)",
                self.fill_rate_model.observation_count()
            );
        }
        println!();

        // Adverse Selection
        println!("ADVERSE SELECTION:");
        if report.adverse_selection.is_warmed_up {
            let total = report.adverse_selection.total_bps;
            let perm = report.adverse_selection.permanent_bps;
            let temp = report.adverse_selection.temporary_bps;
            let timing = report.adverse_selection.timing_bps;
            let perm_pct = if total > 0.0 {
                perm / total * 100.0
            } else {
                0.0
            };
            let temp_pct = if total > 0.0 {
                temp / total * 100.0
            } else {
                0.0
            };
            let timing_pct = if total > 0.0 {
                timing / total * 100.0
            } else {
                0.0
            };

            println!("  Total: {:.2} bps", total);
            println!("  Permanent: {:.2} bps ({:.0}%)", perm, perm_pct);
            println!("  Temporary: {:.2} bps ({:.0}%)", temp, temp_pct);
            println!("  Timing: {:.2} bps ({:.0}%)", timing, timing_pct);
            println!(
                "  Fills measured: {}",
                report.adverse_selection.fills_measured
            );
        } else {
            println!(
                "  Warming up... ({} fills pending)",
                self.as_decomposition.pending_count()
            );
        }
        println!();

        // Joint Dynamics
        println!("JOINT DYNAMICS:");
        if report.joint_dynamics.is_warmed_up {
            println!("  Regime: {}", report.joint_dynamics.regime);
            println!("  ρ(σ, κ): {:.2}", report.joint_dynamics.sigma_kappa_corr);
            println!("  ρ(σ, AS): {:.2}", report.joint_dynamics.sigma_as_corr);
            println!(
                "  ρ(κ, P_inf): {:.2}",
                report.joint_dynamics.kappa_informed_corr
            );
            if report.joint_dynamics.is_toxic {
                println!("  ⚠️  TOXIC CONDITIONS DETECTED");
            }
        } else {
            println!(
                "  Warming up... ({} observations)",
                self.joint_dynamics.observation_count()
            );
        }
        println!();

        // Edge Analysis
        println!("============================================================");
        println!("EDGE ANALYSIS");
        println!("============================================================");
        println!();
        if report.edge.is_warmed_up {
            println!("  Mean edge: {:.2} bps", report.edge.mean_edge_bps);
            println!("  Cells with data: {}/225", report.edge.cells_with_data);
            println!();
            println!("Current state:");
            println!("  Edge: {:.2} bps", report.edge.current_edge_bps);
            println!(
                "  Optimal spread: {:.2} bps",
                report.edge.optimal_spread_bps
            );
            if report.edge.should_quote {
                println!("  ✓ SHOULD QUOTE");
            } else {
                println!("  ✗ DO NOT QUOTE");
            }
        } else {
            println!(
                "  Warming up... ({} observations)",
                self.edge_surface.observation_count()
            );
        }
        println!();
        println!("============================================================");
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
            .map_err(|_| format!("Invalid hours: {}", hours))?;
        Ok(Duration::from_secs(h * 3600))
    } else if let Some(mins) = s.strip_suffix('m') {
        let m: u64 = mins
            .parse()
            .map_err(|_| format!("Invalid minutes: {}", mins))?;
        Ok(Duration::from_secs(m * 60))
    } else if let Some(secs) = s.strip_suffix('s') {
        let sec: u64 = secs
            .parse()
            .map_err(|_| format!("Invalid seconds: {}", secs))?;
        Ok(Duration::from_secs(sec))
    } else {
        // Try parsing as seconds
        let sec: u64 = s.parse().map_err(|_| format!("Invalid duration: {}", s))?;
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
        "║     Latent State Parameter Estimator v{}              ║",
        env!("CARGO_PKG_VERSION")
    );
    eprintln!("╠═══════════════════════════════════════════════════════════╣");
    eprintln!("║  Asset: {:<15}  Network: {:<17} ║", asset, network);
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
// Main Entry Point
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load .env if present
    dotenvy::dotenv().ok();

    // Parse CLI arguments
    let cli = Cli::parse();

    // Setup logging
    let log_level = match cli.log_level.as_str() {
        "trace" => tracing::Level::TRACE,
        "debug" => tracing::Level::DEBUG,
        "info" => tracing::Level::INFO,
        "warn" => tracing::Level::WARN,
        "error" => tracing::Level::ERROR,
        _ => tracing::Level::INFO,
    };

    let subscriber = FmtSubscriber::builder().with_max_level(log_level).finish();
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set tracing subscriber");

    // Parse duration and report interval
    let duration = parse_duration(&cli.duration)?;
    let report_interval = parse_duration(&cli.report_interval)?;

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

    // Create config
    let config = LatentEstimatorConfig {
        asset: cli.asset.clone(),
        n_particles: cli.n_particles,
        report_interval_ms: report_interval.as_millis() as u64,
        duration_ms: duration.as_millis() as u64,
        ..Default::default()
    };

    // Create estimator
    let mut estimator = LatentStateEstimator::new(config);

    // Create InfoClient for WebSocket
    info!("Connecting to {} WebSocket...", cli.network);
    let mut info_client = InfoClient::new(None, Some(base_url)).await?;

    // Create message channel
    let (sender, mut receiver) = unbounded_channel::<Arc<Message>>();

    // Determine asset name (with DEX prefix if applicable)
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

    info!("Subscriptions active. Starting data collection...");

    // Track start time for duration limit
    let start_time = Instant::now();

    // Main event loop
    loop {
        // Check duration limit
        if start_time.elapsed() >= duration {
            info!("Duration limit reached. Shutting down...");
            break;
        }

        tokio::select! {
            // Handle incoming messages
            Some(arc_msg) = receiver.recv() => {
                let msg = Arc::try_unwrap(arc_msg).unwrap_or_else(|arc| (*arc).clone());

                match msg {
                    Message::AllMids(all_mids) => {
                        // Extract mid for our asset
                        if let Some(mid_str) = all_mids.data.mids.get(&cli.asset) {
                            if let Ok(mid) = mid_str.parse::<f64>() {
                                let ts = get_current_timestamp_ms();
                                estimator.on_mid(ts, mid);

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
                                estimator.on_trade(trade.time, price, size, is_buy);

                                if cli.verbose {
                                    debug!(
                                        "Trade: {} {} @ ${:.2} ({})",
                                        size,
                                        if is_buy { "BUY" } else { "SELL" },
                                        price,
                                        trade.time
                                    );
                                }
                            }
                        }
                    }

                    Message::L2Book(book) => {
                        let ts = book.data.time;

                        // Parse best bid/ask and depths
                        if book.data.levels.len() >= 2 {
                            let bids = &book.data.levels[0];
                            let asks = &book.data.levels[1];

                            if !bids.is_empty() && !asks.is_empty() {
                                let best_bid: f64 = bids[0].px.parse().unwrap_or(0.0);
                                let best_ask: f64 = asks[0].px.parse().unwrap_or(0.0);

                                if best_bid > 0.0 && best_ask > 0.0 {
                                    let mid = (best_bid + best_ask) / 2.0;

                                    // Calculate depth (top 5 levels)
                                    let bid_depth: f64 = bids.iter()
                                        .take(5)
                                        .filter_map(|l| l.sz.parse::<f64>().ok())
                                        .sum();
                                    let ask_depth: f64 = asks.iter()
                                        .take(5)
                                        .filter_map(|l| l.sz.parse::<f64>().ok())
                                        .sum();

                                    estimator.on_book(ts, mid, best_bid, best_ask, bid_depth, ask_depth);

                                    if cli.verbose {
                                        debug!(
                                            "L2Book: bid=${:.2} ({:.4}) / ask=${:.2} ({:.4})",
                                            best_bid, bid_depth, best_ask, ask_depth
                                        );
                                    }
                                }
                            }
                        }
                    }

                    _ => {
                        // Ignore other message types
                    }
                }

                // Check if we should print a report
                if estimator.should_report() {
                    estimator.print_report();
                }
            }

            // Handle Ctrl+C
            _ = tokio::signal::ctrl_c() => {
                info!("Received Ctrl+C. Shutting down...");
                break;
            }
        }
    }

    // Final report
    info!("Generating final report...");
    estimator.print_report();

    // Export to JSON if output path specified
    if let Some(output_path) = cli.output {
        info!("Exporting results to {:?}...", output_path);
        let report = estimator.generate_report();
        let json = serde_json::to_string_pretty(&report)?;

        let mut file = File::create(&output_path)?;
        file.write_all(json.as_bytes())?;
        info!("Results exported successfully.");
    }

    info!("Parameter estimator shutdown complete.");
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
        assert_eq!(parse_duration("4h").unwrap(), Duration::from_secs(4 * 3600));
        assert_eq!(
            parse_duration("24h").unwrap(),
            Duration::from_secs(24 * 3600)
        );
    }

    #[test]
    fn test_parse_duration_minutes() {
        assert_eq!(parse_duration("1m").unwrap(), Duration::from_secs(60));
        assert_eq!(parse_duration("5m").unwrap(), Duration::from_secs(5 * 60));
        assert_eq!(parse_duration("30m").unwrap(), Duration::from_secs(30 * 60));
    }

    #[test]
    fn test_parse_duration_seconds() {
        assert_eq!(parse_duration("30s").unwrap(), Duration::from_secs(30));
        assert_eq!(parse_duration("120s").unwrap(), Duration::from_secs(120));
    }

    #[test]
    fn test_observation_buffers() {
        let mut buffers = ObservationBuffers::new(3);

        buffers.add_trade(TradeObservation {
            timestamp_ms: 1,
            price: 100.0,
            size: 1.0,
            is_buy: true,
        });
        buffers.add_trade(TradeObservation {
            timestamp_ms: 2,
            price: 101.0,
            size: 2.0,
            is_buy: false,
        });
        buffers.add_trade(TradeObservation {
            timestamp_ms: 3,
            price: 102.0,
            size: 3.0,
            is_buy: true,
        });

        assert_eq!(buffers.trades.len(), 3);

        // Adding fourth should evict first
        buffers.add_trade(TradeObservation {
            timestamp_ms: 4,
            price: 103.0,
            size: 4.0,
            is_buy: true,
        });

        assert_eq!(buffers.trades.len(), 3);
        assert_eq!(buffers.trades.front().unwrap().timestamp_ms, 2);
    }

    #[test]
    fn test_estimator_creation() {
        let config = LatentEstimatorConfig::default();
        let estimator = LatentStateEstimator::new(config);

        assert!(!estimator.is_warmed_up());
        assert_eq!(estimator.stats.trades_processed, 0);
    }

    #[test]
    fn test_book_imbalance() {
        let mut buffers = ObservationBuffers::new(10);

        buffers.add_book(BookSnapshot {
            timestamp_ms: 1000,
            mid: 100.0,
            best_bid: 99.9,
            best_ask: 100.1,
            bid_depth: 10.0,
            ask_depth: 5.0,
        });

        let imbalance = buffers.book_imbalance();
        assert!((imbalance - 0.333).abs() < 0.01); // (10-5)/(10+5) = 0.333
    }

    #[test]
    fn test_flow_imbalance() {
        let mut buffers = ObservationBuffers::new(10);

        buffers.add_trade(TradeObservation {
            timestamp_ms: 1,
            price: 100.0,
            size: 10.0,
            is_buy: true,
        });
        buffers.add_trade(TradeObservation {
            timestamp_ms: 2,
            price: 100.0,
            size: 5.0,
            is_buy: false,
        });

        let imbalance = buffers.flow_imbalance();
        assert!((imbalance - 0.333).abs() < 0.01); // (10-5)/(10+5) = 0.333
    }
}
