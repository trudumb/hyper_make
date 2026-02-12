//! Enhanced Market Maker Tool for Hyperliquid
//!
//! A production-ready market making tool with:
//! - CLI arguments and TOML config file support
//! - Multiple quoting strategies
//! - Structured logging with tracing
//! - Metrics collection
//! - Prometheus HTTP metrics endpoint

use alloy::signers::local::PrivateKeySigner;
use axum::{routing::get, Json, Router};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info, warn};

use hyperliquid_rust_sdk::{
    init_logging, AdverseSelectionConfig, AssetRuntimeConfig, BaseUrl, CollateralInfo,
    DataQualityConfig, DynamicRiskConfig, EstimatorConfig, ExchangeClient, ExchangeResponseStatus,
    ExchangeContext, FundingConfig, GLFTStrategy, HawkesConfig, HyperliquidExecutor,
    ImpulseControlConfig, InfoClient, InventoryAwareStrategy, KillSwitchConfig, LadderConfig,
    LadderStrategy, LiquidationConfig, LogConfig, LogFormat as MmLogFormat, MarginConfig,
    MarketMaker, MarketMakerConfig as MmConfig, MarketMakerMetricsRecorder, PnLConfig,
    QueueConfig, QuotingStrategy, ReconcileConfig, ReconciliationConfig, RecoveryConfig,
    RejectionRateLimitConfig, RiskConfig, RiskModelConfig, SpreadConfig, SpreadProfile,
    StochasticConfig, SymmetricStrategy,
    market_maker::{BinanceFeed, resolve_binance_symbol,
    auto_derive::auto_derive, environment::live::LiveEnvironment},
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Parse the actual allowed leverage from an exchange error message.
///
/// The exchange returns error messages like:
/// - "Max leverage at current position size is 25x. To increase leverage, reduce position size."
///
/// Returns the parsed leverage value if found, None otherwise.
fn parse_leverage_from_error(error_msg: &str) -> Option<u32> {
    // Pattern: "...is Nx..." where N is the leverage
    // Look for "is " followed by digits and "x"
    if let Some(start) = error_msg.find("is ") {
        let rest = &error_msg[start + 3..];
        // Find the number followed by 'x'
        let mut num_str = String::new();
        for c in rest.chars() {
            if c.is_ascii_digit() {
                num_str.push(c);
            } else if c == 'x' || c == 'X' {
                break;
            } else if !num_str.is_empty() {
                // Non-digit, non-x after digits - stop
                break;
            }
        }
        if !num_str.is_empty() {
            return num_str.parse().ok();
        }
    }
    None
}

// ============================================================================
// CLI Arguments
// ============================================================================

#[derive(Parser)]
#[command(name = "market_maker")]
#[command(version, about = "Hyperliquid Market Maker Tool", long_about = None)]
struct Cli {
    /// Path to config file
    #[arg(short, long, default_value = "market_maker.toml")]
    config: String,

    /// Override asset from config (default: BTC)
    #[arg(long)]
    asset: Option<String>,

    /// Override target liquidity
    #[arg(long)]
    target_liquidity: Option<f64>,

    /// Override risk aversion (gamma) for GLFT strategy
    /// Typical values: 0.1 (aggressive) to 2.0 (conservative)
    #[arg(long)]
    risk_aversion: Option<f64>,

    /// Override max BPS diff before requoting
    #[arg(long)]
    max_bps_diff: Option<u16>,

    /// Override max position size (in contracts)
    #[arg(long)]
    max_position: Option<f64>,

    /// Max position size in notional USD (asset-agnostic, preferred over --max-position)
    /// Converted to contracts at startup via: max_position = max_position_usd / mark_price
    /// Example: 1000.0 → 0.01 BTC at $100K, 40.0 HYPE at $25
    #[arg(long)]
    max_position_usd: Option<f64>,

    /// Capital to deploy in USD (auto-derives position limits, liquidity, gamma, and max_bps_diff)
    /// This is the primary sizing input. All other trading params are derived from this.
    #[arg(long)]
    capital_usd: Option<f64>,

    /// Set leverage for the asset (default: max available)
    #[arg(long)]
    leverage: Option<u32>,

    /// Override decimals for price rounding
    #[arg(long)]
    decimals: Option<u32>,

    /// Override network (mainnet, testnet, localhost)
    #[arg(long)]
    network: Option<String>,

    /// Private key (overrides config and env var)
    #[arg(long, env = "HYPERLIQUID_PRIVATE_KEY")]
    private_key: Option<String>,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long)]
    log_level: Option<String>,

    /// Output format (pretty, json, compact)
    #[arg(long)]
    log_format: Option<String>,

    /// Log file path (logs to both file and stdout)
    #[arg(long)]
    log_file: Option<String>,

    /// Enable multi-stream logging (operational/diagnostic/errors)
    #[arg(long)]
    multi_stream_logs: bool,

    /// Log directory for multi-stream logging (default: logs/)
    #[arg(long)]
    log_dir: Option<String>,

    /// Metrics HTTP port (0 to disable)
    #[arg(long)]
    metrics_port: Option<u16>,

    /// Dry run mode: validate config and connect but don't place orders
    #[arg(long)]
    dry_run: bool,

    // === HIP-3 Support Flags ===
    /// Initial isolated margin allocation in USD for HIP-3 assets
    /// Only used when trading builder-deployed assets
    #[arg(long, default_value = "1000.0")]
    initial_isolated_margin: f64,

    /// Force isolated margin mode even for cross-capable assets
    /// Useful for testing or when you want isolated-style risk
    #[arg(long)]
    force_isolated: bool,

    /// HIP-3 DEX name (e.g., "hyena", "felix")
    /// If not specified, trades on validator perps (default)
    #[arg(long)]
    dex: Option<String>,

    /// Spread profile for target spread ranges (default, hip3, aggressive)
    /// - default: 40-50 bps for liquid perps
    /// - hip3: 15-25 bps for HIP-3 DEX (tighter spreads)
    /// - aggressive: 10-20 bps (experimental)
    #[arg(long, default_value = "default")]
    spread_profile: String,

    /// List available HIP-3 DEXs and exit
    #[arg(long)]
    list_dexs: bool,

    // === Reconciliation Tuning Flags ===
    /// Price tolerance in bps for skipping order updates (default: 10)
    /// Higher values reduce API calls but allow more price drift
    #[arg(long)]
    skip_price_tolerance_bps: Option<u16>,

    /// Max price change in bps for using MODIFY vs CANCEL+PLACE (default: 50)
    /// On Hyperliquid, price modifications reset queue, so this only affects API call count
    #[arg(long)]
    max_modify_price_bps: Option<u16>,

    /// Size tolerance as fraction for skipping order updates (default: 0.05)
    /// E.g., 0.05 means skip if size change is ≤ 5%
    #[arg(long)]
    skip_size_tolerance_pct: Option<f64>,

    /// Max size change as fraction for using MODIFY vs CANCEL+PLACE (default: 0.50)
    #[arg(long)]
    max_modify_size_pct: Option<f64>,

    /// Enable queue-aware reconciliation that uses fill probability to decide refreshes
    /// When enabled, orders with low fill probability may be refreshed even if price is close
    #[arg(long)]
    use_queue_aware: bool,

    /// Time horizon in seconds for queue fill probability calculation (default: 1.0)
    #[arg(long)]
    queue_horizon_seconds: Option<f64>,

    // === Statistical Impulse Control Flags ===
    /// Enable statistical impulse control to reduce API churn by 50-70%
    /// Uses token budget + Δλ filtering to gate order updates
    #[arg(long)]
    use_impulse_control: bool,

    /// Minimum fill probability improvement (Δλ) required to justify an update (default: 0.10)
    /// E.g., 0.10 means 10% improvement in P(fill) required to update an order
    #[arg(long)]
    impulse_threshold: Option<f64>,

    /// P(fill) threshold above which orders are "locked" (protected from updates) (default: 0.30)
    /// E.g., 0.30 means orders with >30% fill probability are protected
    #[arg(long)]
    queue_lock_threshold: Option<f64>,

    /// Price movement in bps that overrides queue lock (default: 25.0)
    /// If price moved more than this, update even if order is locked
    #[arg(long)]
    queue_lock_override_bps: Option<f64>,

    // === Queue Value Comparison Flags ===
    /// Enable EV-based queue value comparison for reconciliation decisions
    /// When enabled, orders are kept if their queue position EV exceeds replacement EV
    /// This provides more principled rate limit optimization than impulse control
    #[arg(long)]
    use_queue_value_comparison: bool,

    /// Minimum improvement percentage required to justify replacing an order (default: 0.15)
    /// E.g., 0.15 means 15% improvement in EV required to justify the API call
    #[arg(long)]
    queue_improvement_threshold: Option<f64>,

    /// Estimated spread capture in bps for EV calculation (default: 8.0)
    /// Used to convert fill probability to expected value
    #[arg(long)]
    queue_spread_capture_bps: Option<f64>,

    /// Minimum order age in seconds before considering replacement (default: 0.3)
    /// Very young orders are protected to give them time to build queue position
    #[arg(long)]
    queue_min_order_age_secs: Option<f64>,

    /// Time horizon in seconds for fill probability calculation (default: 1.0)
    #[arg(long)]
    queue_fill_horizon_secs: Option<f64>,

    // === Competitive Spread Control Flags ===
    /// Minimum kappa floor for order arrival intensity estimation.
    /// Higher kappa = tighter GLFT spreads. Use to force competitive spreads on illiquid assets.
    /// GLFT formula: δ* = (1/γ) × ln(1 + γ/κ) + maker_fee
    /// Example: --kappa-floor 2000 → spreads ~5-8 bps instead of 44 bps with low natural kappa
    #[arg(long)]
    kappa_floor: Option<f64>,

    /// Maximum spread per side in basis points (hard cap regardless of GLFT output).
    /// Use as a safety valve to prevent extremely wide spreads on illiquid assets.
    /// Example: --max-spread-bps 15 → never quote wider than 15 bps per side
    #[arg(long)]
    max_spread_bps: Option<f64>,

    // === Quote Gate (API Budget Conservation) Flags ===
    /// Quote when flat without directional edge signal.
    /// When false (default): conserves API budget by only quoting when there's an edge signal.
    /// When true: market-making mode, quotes both sides even without edge.
    #[arg(long)]
    quote_flat_without_edge: bool,

    /// Disable quote gate entirely (always quote both sides).
    /// Overrides quote_flat_without_edge setting.
    #[arg(long)]
    disable_quote_gate: bool,

    /// Minimum |flow_imbalance| to have directional edge (default: 0.15)
    #[arg(long)]
    quote_gate_min_edge_signal: Option<f64>,

    /// Minimum momentum confidence to trust weak edge signals (default: 0.45)
    #[arg(long)]
    quote_gate_min_edge_confidence: Option<f64>,

    // === Kill Switch Override Flags ===
    /// Override max daily loss in USD from TOML config.
    /// Critical safety parameter — lower is safer. Example: --max-daily-loss 5.0
    #[arg(long)]
    max_daily_loss: Option<f64>,

    /// Override max drawdown as fraction (0.0–1.0) from TOML config.
    /// Example: --max-drawdown 0.10 (= 10% max drawdown)
    #[arg(long)]
    max_drawdown: Option<f64>,

    // === Signal Diagnostics Flags ===
    /// Path to export fill signal snapshots for calibration analysis.
    /// Enables diagnostic infrastructure that captures all signal values at fill time
    /// and tracks markouts at 500ms, 2s, 10s horizons.
    /// Example: --signal-export-path fills_with_signals.json
    #[arg(long)]
    signal_export_path: Option<String>,

    // === Cross-Exchange Lead-Lag (Binance Feed) ===
    /// Disable Binance price feed for cross-exchange lead-lag signal.
    /// The feed is enabled by default as it's a core source of alpha.
    #[arg(long)]
    disable_binance_feed: bool,

    /// Binance symbol override for lead-lag signal (e.g., btcusdt, ethusdt).
    /// If not set, auto-derived from the trading asset. Assets without a
    /// Binance equivalent (HYPE, PURR, etc.) will have the feed disabled.
    #[arg(long)]
    binance_symbol: Option<String>,

    // === RL Agent (Sim-to-Real Transfer) ===
    /// Path to paper trader checkpoint directory for loading Q-table as prior.
    /// The paper trader's learned Q-table becomes an informative prior for the
    /// live RL agent, discounted by paper_prior_weight (default 0.3).
    /// Example: --paper-checkpoint data/checkpoints/HYPE
    #[arg(long)]
    paper_checkpoint: Option<String>,

    /// Enable RL agent to control quoting actions (default: observe only).
    /// When enabled, the RL agent applies spread/skew adjustments after
    /// accumulating enough real fills (default 20).
    #[arg(long)]
    enable_rl: bool,

    /// Path to RL checkpoint file to watch for hot-reload.
    /// When the offline trainer writes a new checkpoint, the live RL agent
    /// blends the updated Q-table into its current state (weight 0.3).
    /// Example: --rl-watch data/checkpoints/paper/BTC/latest/checkpoint.json
    #[arg(long)]
    rl_watch: Option<String>,

    // === Auto-Calibration Pipeline Flags ===
    /// Skip calibration gate — run live with whatever priors exist.
    /// WARNING: uncalibrated priors risk real money.
    #[arg(long)]
    skip_calibration: bool,

    /// Force cold-start live with no prior. Implies --skip-calibration.
    #[arg(long)]
    force: bool,

    /// Auto-paper calibration duration in seconds (default: 1800 = 30 min).
    #[arg(long, default_value = "1800")]
    calibration_duration: u64,

    /// Don't auto-paper — just refuse to start if no calibrated prior exists.
    #[arg(long)]
    no_auto_paper: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a sample config file
    GenerateConfig {
        /// Output file path
        #[arg(short, long, default_value = "market_maker.toml")]
        output: String,
    },
    /// Validate config without running
    ValidateConfig,
    /// Show account status (position, balance, open orders)
    Status,
    /// Run the market maker (default)
    Run,
    /// Run in paper trading mode using the unified observation loop.
    /// Uses the same MarketMaker core as live, with a PaperEnvironment
    /// that synthesizes fills from market data.
    Paper {
        /// Paper trading duration in seconds (0 = run indefinitely).
        #[arg(long, default_value = "0")]
        duration: u64,
    },
}

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct AppConfig {
    #[serde(default)]
    pub network: NetworkConfig,
    #[serde(default)]
    pub trading: TradingConfig,
    #[serde(default)]
    pub strategy: StrategyConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub monitoring: MonitoringAppConfig,
    #[serde(default)]
    pub kill_switch: KillSwitchAppConfig,
}

/// Kill switch configuration from TOML.
/// Maps to `KillSwitchConfig` at runtime, with TOML-friendly field types.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct KillSwitchAppConfig {
    /// Maximum allowed daily loss in USD (default: 500.0 — override this!)
    #[serde(default = "default_ks_max_daily_loss")]
    pub max_daily_loss: f64,
    /// Maximum allowed drawdown as fraction 0.0–1.0 (default: 0.05 = 5%)
    #[serde(default = "default_ks_max_drawdown")]
    pub max_drawdown: f64,
    /// Maximum position value in USD (default: 10000.0)
    #[serde(default = "default_ks_max_position_value")]
    pub max_position_value: f64,
    /// Maximum position size in contracts (default: 1.0)
    #[serde(default = "default_ks_max_position_contracts")]
    pub max_position_contracts: f64,
    /// Seconds without data before stale shutdown (default: 30)
    #[serde(default = "default_ks_stale_data_secs")]
    pub stale_data_threshold_secs: u64,
    /// Cascade severity threshold (default: 5.0)
    #[serde(default = "default_ks_cascade_severity")]
    pub cascade_severity_threshold: f64,
    /// Maximum rate limit errors before shutdown (default: 3)
    #[serde(default = "default_ks_max_rate_limit_errors")]
    pub max_rate_limit_errors: u32,
    /// Enable/disable kill switch (default: true)
    #[serde(default = "default_ks_enabled")]
    pub enabled: bool,
}

fn default_ks_max_daily_loss() -> f64 { 500.0 }
fn default_ks_max_drawdown() -> f64 { 0.05 }
fn default_ks_max_position_value() -> f64 { 10_000.0 }
fn default_ks_max_position_contracts() -> f64 { 1.0 }
fn default_ks_stale_data_secs() -> u64 { 30 }
fn default_ks_cascade_severity() -> f64 { 5.0 }
fn default_ks_max_rate_limit_errors() -> u32 { 3 }
fn default_ks_enabled() -> bool { true }

impl Default for KillSwitchAppConfig {
    fn default() -> Self {
        Self {
            max_daily_loss: default_ks_max_daily_loss(),
            max_drawdown: default_ks_max_drawdown(),
            max_position_value: default_ks_max_position_value(),
            max_position_contracts: default_ks_max_position_contracts(),
            stale_data_threshold_secs: default_ks_stale_data_secs(),
            cascade_severity_threshold: default_ks_cascade_severity(),
            max_rate_limit_errors: default_ks_max_rate_limit_errors(),
            enabled: default_ks_enabled(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MonitoringAppConfig {
    /// HTTP port for Prometheus metrics endpoint (0 to disable)
    #[serde(default = "default_metrics_port")]
    pub metrics_port: u16,
    /// Enable HTTP metrics endpoint
    #[serde(default = "default_enable_metrics")]
    pub enable_http_metrics: bool,
}

fn default_metrics_port() -> u16 {
    9090
}

fn default_enable_metrics() -> bool {
    true
}

impl Default for MonitoringAppConfig {
    fn default() -> Self {
        Self {
            metrics_port: default_metrics_port(),
            enable_http_metrics: default_enable_metrics(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NetworkConfig {
    /// Network to connect to: mainnet, testnet, localhost
    #[serde(default = "default_network")]
    pub base_url: String,
    /// Private key (prefer using HYPERLIQUID_PRIVATE_KEY env var)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub private_key: Option<String>,
}

fn default_network() -> String {
    "testnet".to_string()
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            base_url: default_network(),
            private_key: None,
        }
    }
}

/// Trading configuration with evidence-based defaults.
///
/// **Important**: These values are MAXIMUMS that get capped at runtime based on account equity:
/// - `max_position` capped by: `(account_value × leverage × 0.5) / price`
/// - `target_liquidity` capped by: `min(target_liquidity, max_position × 0.4)`
///
/// This ensures safe operation regardless of configured values.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TradingConfig {
    /// Asset to market make on (e.g., "BTC", "ETH", "SOL")
    /// Default: BTC (most liquid, tightest spreads)
    #[serde(default = "default_asset")]
    pub asset: String,

    /// Capital to deploy in USD. Primary sizing input.
    /// When set, max_position, target_liquidity, risk_aversion, and max_bps_diff
    /// are auto-derived from this + exchange metadata + spread profile.
    /// Any explicitly set parameter overrides the derived value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capital_usd: Option<f64>,

    /// Target liquidity per side in asset units (optional, auto-derived from capital_usd).
    /// When None: auto-derived as fraction of max_position (profile-dependent).
    /// Capped at runtime to max_position based on account equity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_liquidity: Option<f64>,

    /// Risk aversion parameter (gamma) - controls spread width and inventory skew (optional).
    /// When None: auto-derived from spread_profile (Default=0.3, Hip3=0.15, Aggressive=0.10).
    /// Range: 0.1 (aggressive, tight spreads) to 1.0+ (conservative, wide spreads)
    /// GLFT formula: δ = (1/γ) × ln(1 + γ/κ)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub risk_aversion: Option<f64>,

    /// Maximum price deviation (in basis points) before requoting (optional).
    /// When None: auto-derived from fee structure (fee_bps × 2, clamped [3, 15]).
    /// Lower values (2 bps) cause excessive cancels in volatile crypto markets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_bps_diff: Option<u16>,

    /// Maximum position in contracts (optional).
    /// If not specified, defaults to margin-based limit: (account_value × leverage × 0.5) / price
    /// This is capital-efficient: use gamma to control risk, not arbitrary position limits.
    /// Prefer `max_position_usd` for asset-agnostic configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_absolute_position_size: Option<f64>,

    /// Maximum position in notional USD (asset-agnostic, preferred).
    /// Converted to contracts at startup: max_position = max_position_usd / mark_price.
    /// Example: 1000.0 → 0.01 BTC at $100K, 40.0 HYPE at $25, 0.33 ETH at $3K.
    /// Takes priority over `max_absolute_position_size` when both are set.
    /// Also serves as backward-compat `capital_usd` when capital_usd is not set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_position_usd: Option<f64>,

    /// Leverage to use (set on exchange at startup).
    /// Default: max available for the asset (from exchange metadata).
    /// Higher leverage = larger position capacity but more liquidation risk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leverage: Option<u32>,

    /// Decimals for price rounding.
    /// Default: auto-calculated from asset metadata.
    /// Hyperliquid: price_decimals + sz_decimals = 6 (perps) or 8 (spot)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decimals: Option<u32>,
}

/// Default asset: BTC - most liquid crypto, tightest spreads, best for market making
fn default_asset() -> String {
    "BTC".to_string()
}

impl Default for TradingConfig {
    fn default() -> Self {
        Self {
            asset: default_asset(),
            capital_usd: None,                  // Auto-derived from account value
            target_liquidity: None,             // Auto-derived from capital_usd
            risk_aversion: None,                // Auto-derived from spread_profile
            max_bps_diff: None,                 // Auto-derived from fee structure
            max_absolute_position_size: None,   // Auto-derived from margin
            max_position_usd: None,
            leverage: None,
            decimals: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StrategyConfig {
    /// Strategy type: symmetric, inventory_aware, glft
    #[serde(default)]
    pub strategy_type: StrategyType,
    /// Half spread in BPS for symmetric/inventory-aware strategies
    #[serde(default = "default_half_spread_bps")]
    pub half_spread_bps: u16,
    /// Skew factor for inventory-aware strategy (BPS per unit position)
    #[serde(default)]
    pub inventory_skew_factor: f64,
    /// Rolling window for parameter estimation (in seconds)
    #[serde(default = "default_estimation_window_secs")]
    pub estimation_window_secs: u64,
    /// Minimum trades before volatility estimate is valid (warmup period)
    #[serde(default = "default_min_trades")]
    pub min_trades: usize,
    /// Default sigma to use during warmup (before enough trades collected)
    #[serde(default = "default_sigma")]
    pub default_sigma: f64,
    /// Default kappa to use during warmup (order book decay)
    #[serde(default = "default_kappa")]
    pub default_kappa: f64,
    /// Default arrival intensity during warmup (trades per second)
    #[serde(default = "default_arrival_intensity")]
    pub default_arrival_intensity: f64,
    /// Decay period for adaptive warmup threshold (in seconds). 0 = disabled.
    /// On low-activity markets, the min_trades threshold decays linearly to
    /// min_warmup_trades over this period.
    #[serde(default = "default_warmup_decay_secs")]
    pub warmup_decay_secs: u64,
    /// Floor for adaptive warmup threshold (minimum trades even after full decay)
    #[serde(default = "default_min_warmup_trades")]
    pub min_warmup_trades: usize,
    /// Full risk configuration for GLFT dynamic gamma (optional).
    /// If not set, uses risk_aversion from [trading] as gamma_base with defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub risk_config: Option<RiskConfig>,
    /// Ladder configuration for multi-level quoting (optional).
    /// Only used when strategy_type = "ladder".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ladder_config: Option<LadderConfig>,
}

fn default_half_spread_bps() -> u16 {
    10 // 10 bps for simpler strategies
}
fn default_estimation_window_secs() -> u64 {
    300 // 5 minutes
}
fn default_min_trades() -> usize {
    50
}
fn default_sigma() -> f64 {
    0.0001 // 0.01% per-second volatility
}
fn default_kappa() -> f64 {
    100.0 // Moderate order book decay
}
fn default_arrival_intensity() -> f64 {
    0.5 // 1 trade per 2 seconds
}
fn default_warmup_decay_secs() -> u64 {
    300 // 5 minutes
}
fn default_min_warmup_trades() -> usize {
    5
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            strategy_type: StrategyType::default(),
            half_spread_bps: default_half_spread_bps(),
            inventory_skew_factor: 0.0,
            estimation_window_secs: default_estimation_window_secs(),
            min_trades: default_min_trades(),
            default_sigma: default_sigma(),
            default_kappa: default_kappa(),
            default_arrival_intensity: default_arrival_intensity(),
            warmup_decay_secs: default_warmup_decay_secs(),
            min_warmup_trades: default_min_warmup_trades(),
            risk_config: None,
            ladder_config: None,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StrategyType {
    Symmetric,
    InventoryAware,
    Glft,
    /// Multi-level ladder quoting with depth-dependent sizing (GLFT-based)
    #[default]
    Ladder,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoggingConfig {
    /// Log level: trace, debug, info, warn, error
    #[serde(default = "default_log_level")]
    pub level: String,
    /// Output format: pretty, json, compact
    #[serde(default)]
    pub format: LogFormat,
    /// Optional log file path (logs to both file and stdout)
    #[serde(default)]
    pub log_file: Option<String>,
    /// Enable multi-stream logging (operational/diagnostic/errors)
    #[serde(default)]
    pub multi_stream: bool,
    /// Log directory for multi-stream logging
    #[serde(default = "default_log_dir")]
    pub log_dir: String,
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_log_dir() -> String {
    "logs".to_string()
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            format: LogFormat::default(),
            log_file: None,
            multi_stream: false,
            log_dir: default_log_dir(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    #[default]
    Pretty,
    Json,
    Compact,
}

// ============================================================================
// Metrics
// ============================================================================

#[derive(Debug, Default)]
pub struct MarketMakerMetrics {
    pub orders_placed: AtomicU64,
    pub orders_cancelled: AtomicU64,
    pub orders_filled: AtomicU64,
    pub volume_bought_scaled: AtomicU64,
    pub volume_sold_scaled: AtomicU64,
    pub current_position_scaled: AtomicI64,
    pub start_time_ms: AtomicU64,
}

impl MarketMakerMetrics {
    pub fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let metrics = Self::default();
        metrics.start_time_ms.store(now, Ordering::Relaxed);
        metrics
    }

    pub fn record_order_placed(&self) {
        self.orders_placed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_order_cancelled(&self) {
        self.orders_cancelled.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_fill(&self, amount: f64, is_buy: bool) {
        self.orders_filled.fetch_add(1, Ordering::Relaxed);
        let scaled = (amount * 1e8) as u64;
        if is_buy {
            self.volume_bought_scaled
                .fetch_add(scaled, Ordering::Relaxed);
        } else {
            self.volume_sold_scaled.fetch_add(scaled, Ordering::Relaxed);
        }
    }

    pub fn update_position(&self, position: f64) {
        let scaled = (position * 1e8) as i64;
        self.current_position_scaled
            .store(scaled, Ordering::Relaxed);
    }

    pub fn log_summary(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let uptime_secs = (now - self.start_time_ms.load(Ordering::Relaxed)) / 1000;

        info!(
            orders_placed = self.orders_placed.load(Ordering::Relaxed),
            orders_cancelled = self.orders_cancelled.load(Ordering::Relaxed),
            orders_filled = self.orders_filled.load(Ordering::Relaxed),
            volume_bought = self.volume_bought_scaled.load(Ordering::Relaxed) as f64 / 1e8,
            volume_sold = self.volume_sold_scaled.load(Ordering::Relaxed) as f64 / 1e8,
            position = self.current_position_scaled.load(Ordering::Relaxed) as f64 / 1e8,
            uptime_secs = uptime_secs,
            "Market maker metrics"
        );
    }

    pub fn to_json(&self) -> serde_json::Value {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        serde_json::json!({
            "orders_placed": self.orders_placed.load(Ordering::Relaxed),
            "orders_cancelled": self.orders_cancelled.load(Ordering::Relaxed),
            "orders_filled": self.orders_filled.load(Ordering::Relaxed),
            "volume_bought": self.volume_bought_scaled.load(Ordering::Relaxed) as f64 / 1e8,
            "volume_sold": self.volume_sold_scaled.load(Ordering::Relaxed) as f64 / 1e8,
            "current_position": self.current_position_scaled.load(Ordering::Relaxed) as f64 / 1e8,
            "uptime_seconds": (now - self.start_time_ms.load(Ordering::Relaxed)) / 1000,
        })
    }
}

impl MarketMakerMetricsRecorder for MarketMakerMetrics {
    fn record_order_placed(&self) {
        self.orders_placed.fetch_add(1, Ordering::Relaxed);
    }

    fn record_order_cancelled(&self) {
        self.orders_cancelled.fetch_add(1, Ordering::Relaxed);
    }

    fn record_fill(&self, amount: f64, is_buy: bool) {
        self.orders_filled.fetch_add(1, Ordering::Relaxed);
        let scaled = (amount * 1e8) as u64;
        if is_buy {
            self.volume_bought_scaled
                .fetch_add(scaled, Ordering::Relaxed);
        } else {
            self.volume_sold_scaled.fetch_add(scaled, Ordering::Relaxed);
        }
    }

    fn update_position(&self, position: f64) {
        let scaled = (position * 1e8) as i64;
        self.current_position_scaled
            .store(scaled, Ordering::Relaxed);
    }
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load .env file if it exists (before parsing CLI args)
    let _ = dotenvy::dotenv();

    let cli = Cli::parse();

    // Handle subcommands
    match &cli.command {
        Some(Commands::GenerateConfig { output }) => {
            generate_sample_config(output)?;
            return Ok(());
        }
        Some(Commands::ValidateConfig) => {
            let config = load_config(&cli)?;
            println!("Configuration is valid:\n{config:#?}");
            return Ok(());
        }
        Some(Commands::Status) => {
            return show_account_status(&cli).await;
        }
        Some(Commands::Paper { duration }) => {
            return run_paper_mode(&cli, *duration).await;
        }
        Some(Commands::Run) | None => {
            // === Auto-Calibration Pipeline ===
            // Unless --force or --skip-calibration, check for calibrated priors
            // and auto-paper if needed.
            if !cli.force && !cli.skip_calibration {
                use hyperliquid_rust_sdk::market_maker::calibration::gate::{
                    CalibrationGate, CalibrationGateConfig,
                };
                use hyperliquid_rust_sdk::market_maker::checkpoint::types::CheckpointBundle;

                let asset_raw = cli.asset.clone().unwrap_or_else(|| "BTC".to_string());
                let resolved_asset = if let Some(ref dex_name) = cli.dex {
                    if asset_raw.contains(':') {
                        asset_raw.clone()
                    } else {
                        format!("{dex_name}:{asset_raw}")
                    }
                } else {
                    asset_raw.clone()
                };

                // Auto-resolve prior path from --paper-checkpoint or default location
                let prior_path = cli
                    .paper_checkpoint
                    .as_ref()
                    .map(|p| std::path::PathBuf::from(p).join("prior.json"))
                    .unwrap_or_else(|| {
                        std::path::PathBuf::from(format!(
                            "data/checkpoints/paper/{resolved_asset}/prior.json"
                        ))
                    });

                let gate = CalibrationGate::new(CalibrationGateConfig::default());

                // Try to load and assess existing prior
                let prior_ok = if prior_path.exists() {
                    match std::fs::read_to_string(&prior_path) {
                        Ok(json) => match serde_json::from_str::<CheckpointBundle>(&json) {
                            Ok(bundle) => {
                                let readiness = &bundle.readiness;
                                // Check age
                                let now_ms = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_millis() as u64;
                                let age_s = (now_ms.saturating_sub(bundle.metadata.timestamp_ms))
                                    as f64
                                    / 1000.0;
                                if age_s > gate.config().max_prior_age_s {
                                    println!(
                                        "Prior is stale ({:.0}s old, max {:.0}s)",
                                        age_s,
                                        gate.config().max_prior_age_s
                                    );
                                    false
                                } else if gate.passes(readiness) {
                                    println!(
                                        "Prior calibration: {:?} ({}/5 estimators ready, {:.0}s session)",
                                        readiness.verdict,
                                        readiness.estimators_ready,
                                        readiness.session_duration_s
                                    );
                                    true
                                } else {
                                    println!(
                                        "Prior insufficient: {}",
                                        gate.explain_failure(readiness)
                                    );
                                    false
                                }
                            }
                            Err(e) => {
                                println!("Failed to parse prior: {e}");
                                false
                            }
                        },
                        Err(_) => false,
                    }
                } else {
                    println!("No prior found at {}", prior_path.display());
                    false
                };

                if !prior_ok {
                    if cli.no_auto_paper {
                        return Err(
                            "No calibrated prior exists and --no-auto-paper was set. \
                             Run `market_maker paper --duration 1800` first, or use --force to cold-start."
                                .into(),
                        );
                    }

                    println!(
                        "\n=== Auto-calibration: paper trading for {}s ===\n",
                        cli.calibration_duration
                    );
                    run_paper_mode(&cli, cli.calibration_duration).await?;
                    println!("\n=== Auto-calibration complete, proceeding to live ===\n");

                    // Re-check the saved prior
                    if prior_path.exists() {
                        if let Ok(json) = std::fs::read_to_string(&prior_path) {
                            if let Ok(bundle) = serde_json::from_str::<CheckpointBundle>(&json) {
                                if !gate.passes(&bundle.readiness) {
                                    return Err(format!(
                                        "Auto-calibration completed but prior still insufficient: {}\n\
                                         Use --force to override.",
                                        gate.explain_failure(&bundle.readiness)
                                    )
                                    .into());
                                }
                                println!(
                                    "Post-calibration verdict: {:?}",
                                    bundle.readiness.verdict
                                );
                            }
                        }
                    }
                }
            }
            // Continue to run the market maker
        }
    }

    // Handle --list-dexs flag (before normal startup)
    if cli.list_dexs {
        return list_available_dexs(&cli).await;
    }

    // Load and merge configuration
    let config = load_config(&cli)?;

    // Setup tracing/logging
    setup_logging(&config, &cli)?;

    // Get private key from CLI, config, or fail
    let private_key = cli
        .private_key
        .clone()
        .or(config.network.private_key.clone())
        .ok_or("Private key required. Set via --private-key, HYPERLIQUID_PRIVATE_KEY env var, or config file.")?;

    let wallet: PrivateKeySigner = private_key
        .parse()
        .map_err(|_| "Invalid private key format")?;

    // Parse network
    let base_url = parse_base_url(cli.network.as_ref().unwrap_or(&config.network.base_url))?;

    // Build market maker input, CLI args override config
    let asset = cli.asset.clone().unwrap_or(config.trading.asset.clone());

    // === PARAMETER RESOLUTION ===
    // All trading params are now Optional. Resolution priority:
    //   CLI flag → TOML explicit value → auto_derive(capital_usd, exchange_data)
    // These overrides are resolved later (after mark_px is known) in the auto_derive pipeline.
    let target_liquidity_override: Option<f64> = cli.target_liquidity.or(config.trading.target_liquidity);
    let risk_aversion_override: Option<f64> = cli.risk_aversion.or(config.trading.risk_aversion);
    let max_bps_diff_override: Option<u16> = cli.max_bps_diff.or(config.trading.max_bps_diff);

    // Resolve capital_usd: CLI > TOML capital_usd > TOML max_position_usd (backward compat)
    let capital_usd_resolved: Option<f64> = cli.capital_usd
        .or(config.trading.capital_usd)
        .or(config.trading.max_position_usd); // backward compat: treat max_position_usd as capital

    // max_position is optional - will default to margin-based limit later
    // Priority: CLI --max-position-usd > TOML max_position_usd > CLI --max-position > TOML max_absolute_position_size
    // USD values are converted to contracts later when mark_px is known
    let max_position_usd_override: Option<f64> = cli
        .max_position_usd
        .or(config.trading.max_position_usd);
    let max_position_contracts_override: Option<f64> = cli
        .max_position
        .or(config.trading.max_absolute_position_size);
    // Will be resolved after mark_px is known (USD→contracts conversion happens below)
    let max_position_override: Option<f64> = if max_position_usd_override.is_some() {
        None // Defer: USD will be converted to contracts once mark_px is available
    } else {
        max_position_contracts_override
    };

    // Create HTTP client with proper timeouts to prevent 502 errors from stale connections
    // This client is shared across InfoClient and ExchangeClient for REST fallback
    let http_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30)) // Total request timeout
        .connect_timeout(std::time::Duration::from_secs(10)) // Connection establishment
        .pool_idle_timeout(std::time::Duration::from_secs(60)) // Close idle connections before server does
        .pool_max_idle_per_host(8) // Limit pooled connections per host
        .tcp_keepalive(std::time::Duration::from_secs(30)) // Detect dead connections
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {e}"))?;

    // Query metadata to get sz_decimals (always needed for size precision)
    // Use with_reconnect to automatically reconnect if WebSocket disconnects
    let info_client = InfoClient::with_reconnect(Some(http_client.clone()), Some(base_url)).await?;

    // Use DEX-specific metadata query if --dex is specified
    let dex = cli.dex.clone();

    // Auto-prefix asset with DEX name for HIP-3 DEXs
    // API returns prefixed names like "hyna:BTC" for HIP-3 assets
    let asset = if let Some(ref dex_name) = dex {
        if asset.contains(':') {
            // Already prefixed (e.g., "hyna:BTC"), use as-is
            asset
        } else {
            // Auto-prefix: "BTC" + "hyna" → "hyna:BTC"
            let prefixed = format!("{dex_name}:{asset}");
            info!(
                original = %asset,
                prefixed = %prefixed,
                dex = %dex_name,
                "Auto-prefixed asset name for HIP-3 DEX"
            );
            prefixed
        }
    } else {
        // No DEX specified, use plain asset name (validator perps)
        asset
    };

    let meta = info_client
        .meta_for_dex(dex.as_deref())
        .await
        .map_err(|e| {
            format!(
                "Failed to get metadata for DEX '{}': {e}",
                dex.as_deref().unwrap_or("validator")
            )
        })?;

    let asset_meta = meta
        .universe
        .iter()
        .find(|a| a.name == asset)
        .ok_or_else(|| {
            format!(
                "Asset '{}' not found in DEX '{}'. Use --list-dexs to see available DEXs.",
                asset,
                dex.as_deref().unwrap_or("validator")
            )
        })?;

    let sz_decimals = asset_meta.sz_decimals;

    // Resolve collateral/quote asset for HIP-3 DEXs
    // Different HIP-3 DEXs use different stablecoins: USDC, USDE, USDH, etc.
    let collateral = if let Some(token_index) = meta.collateral_token {
        // Fetch spot metadata to resolve token index to name/decimals
        let spot_meta = info_client
            .spot_meta()
            .await
            .map_err(|e| format!("Failed to get spot metadata for collateral resolution: {e}"))?;
        let collateral_info = CollateralInfo::from_token_index(token_index, &spot_meta);
        info!(
            dex = %dex.as_deref().unwrap_or("validator"),
            token_index = token_index,
            symbol = %collateral_info.symbol,
            "Resolved collateral token for DEX"
        );
        collateral_info
    } else {
        // No collateral_token in meta = validator perps = USDC
        CollateralInfo::usdc()
    };

    // Log DEX selection with collateral info
    if let Some(ref dex_name) = dex {
        info!(
            dex = %dex_name,
            asset = %asset,
            collateral = %collateral.symbol,
            "Using HIP-3 DEX with collateral"
        );
    }

    // Create runtime config from asset metadata (HIP-3 detection happens here, ONCE at startup)
    let mut runtime_config = AssetRuntimeConfig::from_asset_meta(asset_meta);

    // Handle --force-isolated flag: override is_cross to false
    if cli.force_isolated && runtime_config.is_cross {
        info!(
            asset = %asset,
            "Force isolated mode: overriding cross margin to isolated (--force-isolated)"
        );
        runtime_config.is_cross = false;
    }

    // Log HIP-3/margin status at startup
    if runtime_config.is_hip3 {
        info!(
            asset = %asset,
            is_hip3 = true,
            is_cross = runtime_config.is_cross,
            oi_cap_usd = %runtime_config.oi_cap_usd,
            deployer = ?runtime_config.deployer,
            "HIP-3 builder-deployed asset detected (isolated margin only)"
        );
    } else if cli.force_isolated {
        info!(
            asset = %asset,
            is_hip3 = false,
            is_cross = false,
            "Standard validator perp using forced isolated margin mode"
        );
    } else {
        info!(
            asset = %asset,
            is_hip3 = false,
            is_cross = runtime_config.is_cross,
            "Standard validator perp (cross margin supported)"
        );
    }

    // Use CLI/config leverage if specified, otherwise use max available from asset metadata
    let leverage = cli
        .leverage
        .or(config.trading.leverage)
        .unwrap_or(asset_meta.max_leverage as u32);

    info!(
        asset = %asset,
        max_leverage = asset_meta.max_leverage,
        using_leverage = leverage,
        "Leverage: using {} (max available: {}x)",
        if cli.leverage.is_some() || config.trading.leverage.is_some() { "configured" } else { "max available" },
        asset_meta.max_leverage
    );

    // Auto-calculate price decimals from asset metadata if not explicitly set
    let decimals = match cli.decimals.or(config.trading.decimals) {
        Some(d) => d,
        None => {
            // Hyperliquid requires: price_decimals + sz_decimals = 6 (perpetuals) or 8 (spot)
            // Perpetual assets have index < 10000, spot assets >= 10000
            let is_spot = meta
                .universe
                .iter()
                .position(|a| a.name == asset)
                .map(|i| i >= 10000)
                .unwrap_or(false);
            let max_decimals: u32 = if is_spot { 8 } else { 6 };
            let price_decimals = max_decimals.saturating_sub(sz_decimals);

            info!(
                asset = %asset,
                sz_decimals = sz_decimals,
                price_decimals = price_decimals,
                "Auto-calculated price precision from asset metadata"
            );

            price_decimals
        }
    };

    // Print startup banner
    print_startup_banner(&asset, &collateral.symbol, &base_url, cli.dry_run);

    info!(
        asset = %asset,
        target_liquidity = ?target_liquidity_override,
        risk_aversion = ?risk_aversion_override,
        max_bps_diff = ?max_bps_diff_override,
        capital_usd = ?capital_usd_resolved,
        max_position_override = ?max_position_override,
        leverage = %leverage,
        decimals = %decimals,
        strategy = ?config.strategy.strategy_type,
        network = ?base_url,
        dry_run = cli.dry_run,
        "Starting market maker (None values will be auto-derived)"
    );

    // Log strategy info
    if config.strategy.strategy_type == StrategyType::InventoryAware {
        info!(
            skew_factor = config.strategy.inventory_skew_factor,
            "Using inventory-aware strategy"
        );
    }

    // Create metrics
    let metrics = Arc::new(MarketMakerMetrics::new());

    // Spawn periodic metrics logging
    let metrics_clone = metrics.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
        loop {
            interval.tick().await;
            metrics_clone.log_summary();
        }
    });

    // Create exchange client (pass DEX-specific meta and DEX name for HIP-3 support)
    // For HIP-3 DEXs, this applies the correct asset index formula:
    // asset_index = 100000 + (perp_dex_index × 10000) + index_in_meta
    // Use the same HTTP client with timeouts for REST fallback
    let mut exchange_client = ExchangeClient::new_for_dex(
        Some(http_client.clone()),
        wallet.clone(),
        Some(base_url),
        Some(meta.clone()),
        None,
        dex.as_deref(),
    )
    .await
    .map_err(|e| format!("Failed to create exchange client: {e}"))?;

    // Enable WebSocket posting for low-latency order execution
    // Create a separate InfoClient for WS POST sharing (MarketMaker keeps the original)
    // Use the same HTTP client with timeouts for REST fallback
    let mut ws_post_info_client =
        InfoClient::with_reconnect(Some(http_client.clone()), Some(base_url))
            .await
            .map_err(|e| format!("Failed to create WS POST InfoClient: {e}"))?;
    // Explicitly connect the WS client since we won't be adding subscriptions
    ws_post_info_client
        .ensure_connected()
        .await
        .map_err(|e| format!("Failed to connect WS POST client: {e}"))?;

    let ws_post_info_shared = Arc::new(tokio::sync::RwLock::new(ws_post_info_client));
    exchange_client.enable_ws_post(
        Arc::clone(&ws_post_info_shared),
        Some(std::time::Duration::from_secs(3)),
    );
    info!("WebSocket posting enabled for low-latency order execution");

    // Set leverage on the exchange (use is_cross from runtime config for HIP-3 support)
    info!(
        leverage = leverage,
        asset = %asset,
        is_cross = runtime_config.is_cross,
        "Setting leverage on exchange"
    );

    // Track effective leverage - may be reduced by exchange due to margin tiers
    let mut effective_leverage = leverage;

    match exchange_client
        .update_leverage(leverage, &asset, runtime_config.is_cross, None)
        .await
    {
        Ok(response) => {
            // Check if response contains a leverage limit error
            // The exchange returns Ok(ExchangeResponseStatus::Err(msg)) when leverage is capped
            if let ExchangeResponseStatus::Err(ref err_msg) = response {
                // Parse "Max leverage at current position size is 25x" pattern
                if let Some(actual) = parse_leverage_from_error(err_msg) {
                    warn!(
                        requested = leverage,
                        actual = actual,
                        "Exchange capped leverage due to position size - using actual"
                    );
                    effective_leverage = actual;
                } else {
                    warn!(
                        requested = leverage,
                        error = %err_msg,
                        "Leverage request returned error - using requested"
                    );
                }
            } else {
                info!(leverage = leverage, "Leverage set successfully");
            }
        }
        Err(e) => {
            warn!(leverage = leverage, error = %e, "Failed to set leverage, using existing");
        }
    }

    // Use effective_leverage for all subsequent calculations
    let leverage = effective_leverage;

    // Query initial position (use DEX-specific clearinghouse state for HIP-3)
    let user_address = wallet.address();
    let user_state = info_client
        .user_state_for_dex(user_address, dex.as_deref())
        .await
        .map_err(|e| format!("Failed to get user state: {e}"))?;
    let initial_position = user_state
        .asset_positions
        .iter()
        .find(|p| p.position.coin == asset)
        .map(|p| p.position.szi.parse::<f64>().unwrap_or(0.0))
        .unwrap_or(0.0);

    info!(
        initial_position = %initial_position,
        "Queried initial position"
    );

    // Query account equity for position limits and auto-derive trading params
    let (target_liquidity, risk_aversion, max_bps_diff, max_position) = {
        let asset_data_result = info_client
            .active_asset_data(user_address, asset.clone())
            .await;

        // For HIP-3 DEXs, collateral is in spot balance (USDE, USDH, etc.)
        // For validator perps, collateral is in perps clearinghouse (USDC)
        let account_value: f64 = if dex.is_some() {
            // HIP-3: Get account value from spot balance (total, not available)
            // The collateral token (e.g., USDE) holds the full deposited amount.
            match info_client.user_token_balances(user_address).await {
                Ok(balances) => collateral
                    .balance_from_spot(&balances.balances)
                    .map(|(total, _hold)| total) // Use total, not total-hold
                    .unwrap_or_else(|| {
                        warn!(
                            "Collateral {} not found in spot balances, using 0",
                            collateral.symbol
                        );
                        0.0
                    }),
                Err(e) => {
                    warn!("Failed to query spot balances for HIP-3: {e}, using 0");
                    0.0
                }
            }
        } else {
            // Validator perps: unified account value = perps clearinghouse + spot balances.
            // Hyperliquid unified margin uses ALL collateral (perps + spot) for margin.
            // cross_margin_summary.account_value only reflects the perps side.
            let perps_value = match info_client.user_state(user_address).await {
                Ok(state) => {
                    let cross: f64 = state
                        .cross_margin_summary
                        .account_value
                        .parse()
                        .unwrap_or(0.0);
                    let margin: f64 = state
                        .margin_summary
                        .account_value
                        .parse()
                        .unwrap_or(0.0);
                    cross.max(margin)
                }
                Err(e) => {
                    warn!("Failed to query user state: {e}, using 0");
                    0.0
                }
            };

            // Always check spot balances — unified margin counts all collateral
            let spot_value = match info_client.user_token_balances(user_address).await {
                Ok(balances) => {
                    // Sum USDC + other stablecoin spot balances used as collateral
                    let spot_usdc: f64 = balances
                        .balances
                        .iter()
                        .filter(|b| b.coin == "USDC" || b.coin == "USDT" || b.coin == "USDE")
                        .map(|b| b.total.parse::<f64>().unwrap_or(0.0))
                        .sum();
                    spot_usdc
                }
                Err(e) => {
                    warn!("Failed to query spot balances: {e}, using 0");
                    0.0
                }
            };

            let total = perps_value + spot_value;
            if total > 0.0 {
                info!(
                    perps_value = %format!("{:.2}", perps_value),
                    spot_value = %format!("{:.2}", spot_value),
                    total = %format!("{:.2}", total),
                    "Unified account value (perps clearinghouse + spot collateral)"
                );
            } else {
                warn!(
                    "No funds found in perps clearinghouse or spot wallet. \
                     Deposit USDC to trade."
                );
            }
            total
        };

        match asset_data_result {
            Ok(asset_data) => {
                let mark_px: f64 = asset_data.mark_px.parse().unwrap_or(1.0);
                let spread_profile = SpreadProfile::from_str(&cli.spread_profile);

                // === AUTO-DERIVE PIPELINE ===
                // 1. Resolve capital_usd (the ONE sizing input)
                let capital_usd = capital_usd_resolved.unwrap_or_else(|| {
                    let frac = 0.10; // Use 10% of account as default allocation
                    let derived = account_value * frac;
                    info!(
                        account_value = %format!("${:.2}", account_value),
                        fraction = "10%",
                        derived_capital = %format!("${:.2}", derived),
                        "No capital_usd specified, using 10% of account value"
                    );
                    derived
                });

                // 2. Auto-derive all parameters from first principles
                let exchange_ctx = ExchangeContext {
                    mark_px,
                    account_value,
                    available_margin: account_value,
                    max_leverage: leverage as f64,
                    fee_bps: 1.5,
                    sz_decimals,
                    min_notional: 10.0,
                };
                let derived = auto_derive(
                    capital_usd, spread_profile, &exchange_ctx,
                );

                // 3. Viability check
                if !derived.viable {
                    if let Some(ref diag) = derived.diagnostic {
                        error!("{}", diag);
                    }
                    if !cli.force {
                        return Err(format!(
                            "Cannot trade: {}",
                            derived.diagnostic.as_deref().unwrap_or("insufficient capital")
                        ).into());
                    }
                    warn!("--force: proceeding despite insufficient capital");
                }

                // 4. Apply overrides (explicit config/CLI values take priority over derived)
                let risk_aversion = risk_aversion_override.unwrap_or(derived.risk_aversion);
                let max_bps_diff = max_bps_diff_override.unwrap_or(derived.max_bps_diff);

                // Resolve USD→contracts for explicit position overrides
                let effective_override: Option<f64> = if let Some(usd_limit) = max_position_usd_override {
                    let contracts = usd_limit / mark_px;
                    info!(
                        max_position_usd = %format!("{:.2}", usd_limit),
                        mark_px = %format!("{:.2}", mark_px),
                        derived_contracts = %format!("{:.6}", contracts),
                        "USD position limit → contracts (explicit override)"
                    );
                    Some(contracts)
                } else {
                    max_position_override
                };

                // Max position: explicit override capped by margin, or auto-derived
                let safety_factor = 0.5;
                let max_from_leverage = (account_value * leverage as f64 * safety_factor) / mark_px;
                let max_position = match effective_override {
                    Some(user_limit) => {
                        let capped = user_limit.min(max_from_leverage);
                        info!(
                            user_limit = %format!("{:.6}", user_limit),
                            margin_limit = %format!("{:.6}", max_from_leverage),
                            effective = %format!("{:.6}", capped),
                            "User-specified max_position (capped by margin)"
                        );
                        capped
                    }
                    None => derived.max_position,
                };

                // Target liquidity: explicit override or auto-derived, then cap by margin
                let min_notional = 10.0;
                let num_ladder_levels = 5_u8;
                let smallest_level_fraction = 0.05;
                let min_for_ladder = min_notional / (mark_px * smallest_level_fraction);

                let raw_target = target_liquidity_override.unwrap_or(derived.target_liquidity);
                let capped_liquidity = raw_target
                    .max(min_for_ladder) // Ensure multi-level ladder support
                    .min(max_from_leverage); // Cap by margin

                if raw_target < min_for_ladder {
                    info!(
                        configured_target = %format!("{:.6}", raw_target),
                        min_for_ladder = %format!("{:.6}", min_for_ladder),
                        using = %format!("{:.6}", capped_liquidity),
                        "Auto-increased target_liquidity to support {} ladder levels",
                        num_ladder_levels
                    );
                }

                info!(
                    capital_usd = %format!("${:.2}", capital_usd),
                    max_position = %format!("{:.4}", max_position),
                    target_liquidity = %format!("{:.4}", capped_liquidity),
                    risk_aversion = %format!("{:.3}", risk_aversion),
                    max_bps_diff = %max_bps_diff,
                    auto_derived = derived.diagnostic.is_none(),
                    "Trading parameters (auto-derived, override with explicit config)"
                );

                (capped_liquidity, risk_aversion, max_bps_diff, max_position)
            }
            Err(e) => {
                // Fallback: use config values or conservative defaults
                let fallback_max = max_position_override.unwrap_or(0.05);
                let fallback_liquidity = target_liquidity_override.unwrap_or(0.01);
                let fallback_gamma = risk_aversion_override.unwrap_or(0.3);
                let fallback_bps = max_bps_diff_override.unwrap_or(5);
                tracing::warn!(
                    fallback_max = %format!("{:.6}", fallback_max),
                    "Failed to query asset data: {e}, using fallback values"
                );
                (fallback_liquidity, fallback_gamma, fallback_bps, fallback_max)
            }
        }
    };

    // Create dynamic risk config with the leverage we set
    let dynamic_risk_config = DynamicRiskConfig::default().with_max_leverage(leverage as f64);
    info!(
        max_leverage = leverage,
        risk_fraction = dynamic_risk_config.risk_fraction,
        num_sigmas = dynamic_risk_config.num_sigmas,
        sigma_prior = dynamic_risk_config.sigma_prior,
        "Dynamic risk config (position limit = min(equity × leverage, volatility-based))"
    );

    // Initial isolated margin from CLI (default $1000)
    let initial_isolated_margin = cli.initial_isolated_margin;

    // Save collateral symbol for metrics before it's moved into config
    let collateral_symbol = collateral.symbol.clone();

    // Create market maker config
    // Build StochasticConfig with CLI overrides for quote gate
    let stochastic_config = {
        let mut cfg = StochasticConfig::default();
        // Quote Gate settings (API budget conservation)
        if cli.disable_quote_gate {
            cfg.enable_quote_gate = false;
        }
        if cli.quote_flat_without_edge {
            cfg.quote_gate_flat_without_edge = true;
        }
        if let Some(v) = cli.quote_gate_min_edge_signal {
            cfg.quote_gate_min_edge_signal = v;
        }
        if let Some(v) = cli.quote_gate_min_edge_confidence {
            cfg.quote_gate_min_edge_confidence = v;
        }
        cfg
    };

    let mm_config = MmConfig {
        asset: Arc::from(asset.as_str()),
        target_liquidity,
        risk_aversion,
        max_bps_diff,
        max_position,
        max_position_usd: max_position_usd_override.unwrap_or(0.0),
        max_position_user_specified: max_position_usd_override.is_some()
            || max_position_contracts_override.is_some(),
        decimals,
        sz_decimals,
        multi_asset: false, // Single-asset mode by default
        stochastic: stochastic_config,
        smart_reconcile: true, // Enable queue-preserving ORDER MODIFY by default
        reconcile: {
            let mut rc = ReconcileConfig::default();
            if let Some(v) = cli.skip_price_tolerance_bps {
                rc.skip_price_tolerance_bps = v;
            }
            if let Some(v) = cli.max_modify_price_bps {
                rc.max_modify_price_bps = v;
            }
            if let Some(v) = cli.skip_size_tolerance_pct {
                rc.skip_size_tolerance_pct = v;
            }
            if let Some(v) = cli.max_modify_size_pct {
                rc.max_modify_size_pct = v;
            }
            if cli.use_queue_aware {
                rc.use_queue_aware = true;
            }
            if let Some(v) = cli.queue_horizon_seconds {
                rc.queue_horizon_seconds = v;
            }
            // Enable impulse filtering by default (matches ImpulseControlConfig default)
            // Can be explicitly enabled via --use-impulse-control even if default changes
            rc.use_impulse_filter =
                ImpulseControlConfig::default().enabled || cli.use_impulse_control;

            // Queue value comparison (EV-based reconciliation) - enabled by default
            // Supersedes impulse filtering with more principled EV-based decisions
            if cli.use_queue_value_comparison {
                rc.use_queue_value_comparison = true;
            }
            // Apply queue value config parameters from CLI
            if let Some(v) = cli.queue_improvement_threshold {
                rc.queue_value_config.improvement_threshold = v;
            }
            if let Some(v) = cli.queue_lock_threshold {
                // Use this for both impulse filter and queue value config
                rc.queue_value_config.queue_lock_threshold = v;
            }
            if let Some(v) = cli.queue_lock_override_bps {
                rc.queue_value_config.queue_lock_override_bps = v;
            }
            if let Some(v) = cli.queue_spread_capture_bps {
                rc.queue_value_config.spread_capture_bps = v;
            }
            if let Some(v) = cli.queue_min_order_age_secs {
                rc.queue_value_config.min_order_age_secs = v;
            }
            if let Some(v) = cli.queue_fill_horizon_secs {
                rc.queue_value_config.fill_horizon_secs = v;
            }
            rc
        },
        // HIP-3 support: pre-computed runtime config for zero hot-path overhead
        runtime: runtime_config,
        initial_isolated_margin,
        // HIP-3 DEX selection
        dex: dex.clone(),
        // Collateral/quote asset (USDC, USDE, USDH, etc.)
        collateral,
        // Statistical impulse control
        impulse_control: {
            let mut ic = ImpulseControlConfig::default();
            if cli.use_impulse_control {
                ic.enabled = true;
            }
            if let Some(v) = cli.impulse_threshold {
                ic.filter.improvement_threshold = v;
            }
            if let Some(v) = cli.queue_lock_threshold {
                ic.filter.queue_lock_threshold = v;
            }
            if let Some(v) = cli.queue_lock_override_bps {
                ic.filter.queue_lock_override_bps = v;
            }
            ic
        },
        // Spread profile for target spread ranges
        spread_profile: SpreadProfile::from_str(&cli.spread_profile),
        // Trading fee in basis points (maker fee)
        fee_bps: 1.5,
    };

    // Extract EMA config before mm_config is moved
    let microprice_ema_alpha = mm_config.stochastic.microprice_ema_alpha;
    let microprice_ema_min_change_bps = mm_config.stochastic.microprice_ema_min_change_bps;

    // Create strategy based on config
    let strategy: Box<dyn QuotingStrategy> = match config.strategy.strategy_type {
        StrategyType::Symmetric => {
            Box::new(SymmetricStrategy::new(config.strategy.half_spread_bps))
        }
        StrategyType::InventoryAware => Box::new(InventoryAwareStrategy::new(
            config.strategy.half_spread_bps,
            config.strategy.inventory_skew_factor,
        )),
        StrategyType::Glft => {
            if let Some(ref risk_cfg) = config.strategy.risk_config {
                info!(
                    gamma_base = risk_cfg.gamma_base,
                    volatility_weight = risk_cfg.volatility_weight,
                    toxicity_sensitivity = risk_cfg.toxicity_sensitivity,
                    inventory_sensitivity = risk_cfg.inventory_sensitivity,
                    "Using full RiskConfig for dynamic gamma"
                );
                Box::new(GLFTStrategy::with_config(risk_cfg.clone()))
            } else {
                Box::new(GLFTStrategy::new(risk_aversion))
            }
        }
        StrategyType::Ladder => {
            // Parse spread profile to determine risk configuration
            let spread_profile = SpreadProfile::from_str(&cli.spread_profile);

            // Risk config priority: spread_profile > config file > default with gamma from CLI
            let risk_cfg = match spread_profile {
                SpreadProfile::Hip3 => {
                    // HIP-3 profile: use optimized risk config for 15-25 bps spreads
                    info!(
                        spread_profile = "hip3",
                        gamma_base = 0.15,
                        "Using HIP-3 RiskConfig for tighter spreads"
                    );
                    RiskConfig::hip3()
                }
                SpreadProfile::Aggressive => {
                    // Aggressive profile: use low gamma for 10-20 bps spreads
                    info!(
                        spread_profile = "aggressive",
                        gamma_base = 0.10,
                        "Using Aggressive RiskConfig (experimental)"
                    );
                    RiskConfig {
                        gamma_base: 0.10,
                        gamma_min: 0.05,
                        gamma_max: 1.5,
                        min_spread_floor: 0.0004, // 4 bps floor
                        enable_time_of_day_scaling: false,
                        enable_book_depth_scaling: false,
                        max_warmup_gamma_mult: 1.02,
                        ..RiskConfig::hip3() // Base on HIP-3 for other params
                    }
                }
                SpreadProfile::Default => {
                    // Default: use config file or CLI gamma
                    config
                        .strategy
                        .risk_config
                        .clone()
                        .unwrap_or_else(|| RiskConfig {
                            gamma_base: risk_aversion,
                            ..Default::default()
                        })
                }
            };
            let mut ladder_cfg = config.strategy.ladder_config.clone().unwrap_or_default();

            // Max spread handling: CLI override takes precedence, otherwise use dynamic bounds
            // Dynamic bounds are computed at runtime from fill rate controller + market spread p80
            if let Some(max_spread) = cli.max_spread_bps {
                // Explicit CLI override - use static cap
                info!(
                    max_spread_per_side_bps = %max_spread,
                    mode = "CLI override",
                    "Max spread cap: using explicit CLI value"
                );
                ladder_cfg.max_spread_per_side_bps = max_spread;
            } else {
                // No CLI arg - use dynamic bounds (model-driven spread ceiling)
                // Set to 0.0 to signal that dynamic ceiling should be used at runtime
                ladder_cfg.max_spread_per_side_bps = 0.0;
                info!(
                    dex = ?dex,
                    mode = "dynamic",
                    "Max spread cap: using model-driven dynamic ceiling"
                );
            }

            // Use log-additive risk model for Hip3 to prevent gamma explosion
            let risk_model_cfg = match spread_profile {
                SpreadProfile::Hip3 | SpreadProfile::Aggressive => {
                    info!(
                        risk_model_blend = 1.0,
                        "Using log-additive CalibratedRiskModel (no multiplicative gamma explosion)"
                    );
                    RiskModelConfig::hip3()
                }
                SpreadProfile::Default => RiskModelConfig::default(),
            };

            info!(
                gamma_base = risk_cfg.gamma_base,
                num_levels = ladder_cfg.num_levels,
                min_depth_bps = ladder_cfg.min_depth_bps,
                max_depth_bps = ladder_cfg.max_depth_bps,
                geometric_spacing = ladder_cfg.geometric_spacing,
                max_spread_per_side_bps = ladder_cfg.max_spread_per_side_bps,
                risk_model_blend = risk_model_cfg.risk_model_blend,
                "Using LadderStrategy (multi-level GLFT)"
            );
            Box::new(LadderStrategy::with_full_config(
                risk_cfg,
                ladder_cfg,
                risk_model_cfg,
                Default::default(), // KellySizer
            ))
        }
    };

    // Create estimator config for live parameter estimation (using legacy compatibility)
    let mut estimator_config = EstimatorConfig::from_legacy(
        config.strategy.estimation_window_secs * 1000,
        config.strategy.min_trades,
        config.strategy.default_sigma,
        config.strategy.default_kappa,
        config.strategy.default_arrival_intensity,
        config.strategy.warmup_decay_secs,
        config.strategy.min_warmup_trades,
    );

    // Kappa prior handling based on spread profile (GLFT first principles)
    // GLFT: δ* ≈ 1/κ when γ/κ is small, so κ directly controls target spread
    let spread_profile = SpreadProfile::from_str(&cli.spread_profile);
    match spread_profile {
        SpreadProfile::Hip3 => {
            // HIP-3 profile: Target 15-25 bps spreads
            // κ=1500 → 1/1500 = 6.7 bps GLFT + 1.5 bps fee ≈ 16 bps total
            estimator_config.kappa_prior_mean = 1500.0;
            info!(
                spread_profile = "hip3",
                kappa_prior = 1500.0,
                target_spread_bps = "15-25",
                "Using HIP-3 spread profile for tighter spreads"
            );
        }
        SpreadProfile::Aggressive => {
            // Aggressive profile: Target 10-20 bps spreads (EXPERIMENTAL)
            // κ=2000 → 1/2000 = 5 bps GLFT + 1.5 bps fee ≈ 13 bps total
            estimator_config.kappa_prior_mean = 2000.0;
            info!(
                spread_profile = "aggressive",
                kappa_prior = 2000.0,
                target_spread_bps = "10-20",
                "Using AGGRESSIVE spread profile (experimental)"
            );
        }
        SpreadProfile::Default => {
            // Default profile: Target 40-50 bps for liquid perps
            // Legacy behavior: adjust for DEX vs perps
            if dex.is_some() {
                // HIP-3 DEX without explicit profile: use conservative default
                estimator_config.kappa_prior_mean = 500.0;
                info!(
                    spread_profile = "default",
                    kappa_prior = 500.0,
                    dex = ?dex,
                    "Using default DEX kappa (use --spread-profile hip3 for tighter spreads)"
                );
            } else {
                // Standard perps: κ=2500 → ~4 bps GLFT spread
                // (Already default from EstimatorConfig)
                info!(
                    spread_profile = "default",
                    kappa_prior = %estimator_config.kappa_prior_mean,
                    "Using default perps kappa"
                );
            }
        }
    }

    if let Some(floor) = cli.kappa_floor {
        // Explicit CLI override - use static floor
        info!(
            kappa_floor = %floor,
            mode = "CLI override",
            "Kappa floor: using explicit CLI value"
        );
        estimator_config.kappa_floor = Some(floor);
    } else {
        // No CLI arg - use dynamic bounds (model-driven kappa floor)
        // Leave kappa_floor as None to signal dynamic floor should be computed at runtime
        estimator_config.kappa_floor = None;
        info!(
            dex = ?dex,
            kappa_prior_mean = %estimator_config.kappa_prior_mean,
            mode = "dynamic",
            "Kappa floor: using model-driven dynamic floor from Bayesian CI"
        );
    }

    // Set max spread ceiling on estimator config for dynamic bounds detection
    if let Some(max_spread) = cli.max_spread_bps {
        estimator_config.max_spread_ceiling_bps = Some(max_spread);
    } else {
        estimator_config.max_spread_ceiling_bps = None;
    }

    info!(
        estimation_window_secs = config.strategy.estimation_window_secs,
        min_trades = config.strategy.min_trades,
        warmup_decay_secs = config.strategy.warmup_decay_secs,
        min_warmup_trades = config.strategy.min_warmup_trades,
        kappa_floor = ?estimator_config.kappa_floor,
        kappa_prior_mean = %estimator_config.kappa_prior_mean,
        "Live parameter estimation enabled (adaptive warmup)"
    );

    // Create executor
    let executor = HyperliquidExecutor::new(exchange_client, Some(metrics.clone()));

    // Create Tier 1 module configs with defaults
    let as_config = AdverseSelectionConfig::default();
    let queue_config = QueueConfig::default();
    let liquidation_config = LiquidationConfig::default();

    // Create kill switch config (production safety)
    // Priority: CLI flags > TOML [kill_switch] > KillSwitchConfig::default()
    let ks_toml = &config.kill_switch;
    let kill_switch_config = KillSwitchConfig {
        max_daily_loss: cli.max_daily_loss.unwrap_or(ks_toml.max_daily_loss),
        max_drawdown: cli.max_drawdown.unwrap_or(ks_toml.max_drawdown),
        max_position_value: if let Some(usd_limit) = max_position_usd_override {
            usd_limit // USD position limit overrides TOML position value
        } else {
            ks_toml.max_position_value
        },
        max_position_contracts: max_position, // from CLI/config position sizing
        stale_data_threshold: std::time::Duration::from_secs(ks_toml.stale_data_threshold_secs),
        max_rate_limit_errors: ks_toml.max_rate_limit_errors,
        cascade_severity_threshold: ks_toml.cascade_severity_threshold,
        price_velocity_threshold: 0.05, // 5%/s — default from KillSwitchConfig
        position_velocity_threshold: 0.50, // 50% of max_position/min — default
        liquidation_position_jump_fraction: 0.20, // 20% of max position
        liquidation_fill_timeout_s: 5,
        // Drawdown is meaningless when peak is spread noise ($0.02).
        // min_peak = max(1.0, 2% of max position notional).
        // check_daily_loss still protects against catastrophic loss.
        min_peak_for_drawdown: {
            let pos_val = if let Some(usd_limit) = max_position_usd_override {
                usd_limit
            } else {
                ks_toml.max_position_value
            };
            1.0_f64.max(pos_val * 0.02)
        },
        enabled: ks_toml.enabled,
    };
    info!(
        max_daily_loss = %kill_switch_config.max_daily_loss,
        max_drawdown = %format!("{:.1}%", kill_switch_config.max_drawdown * 100.0),
        max_position_value = %kill_switch_config.max_position_value,
        max_position_contracts = %kill_switch_config.max_position_contracts,
        stale_data_threshold = ?kill_switch_config.stale_data_threshold,
        "Kill switch enabled"
    );

    // Create Tier 2 module configs with defaults
    let hawkes_config = HawkesConfig::default();
    let funding_config = FundingConfig::default();
    let spread_config = SpreadConfig::default();
    let pnl_config = PnLConfig::default();
    // Use API-derived leverage from asset metadata (first-principles: exchange sets limits)
    let margin_config = MarginConfig::from_asset_meta(asset_meta);
    info!(
        asset = %asset,
        max_leverage = margin_config.max_leverage,
        isolated_only = margin_config.leverage_config.as_ref().map(|c| c.isolated_only).unwrap_or(false),
        "Margin config from API metadata"
    );
    let data_quality_config = DataQualityConfig::default();
    let recovery_config = RecoveryConfig::default();
    let reconciliation_config = ReconciliationConfig::default();
    let rate_limit_config = RejectionRateLimitConfig::default();

    // Validate config before constructing MarketMaker
    mm_config
        .validate()
        .map_err(|e| format!("Invalid MarketMakerConfig: {e}"))
        .expect("config validation failed");

    // Create and start market maker
    let environment = LiveEnvironment::from_executor(executor);
    let mut market_maker = MarketMaker::new(
        mm_config,
        strategy,
        environment,
        info_client,
        user_address,
        initial_position,
        Some(metrics),
        estimator_config,
        as_config,
        queue_config,
        liquidation_config,
        kill_switch_config,
        hawkes_config,
        funding_config,
        spread_config,
        pnl_config,
        margin_config,
        data_quality_config,
        recovery_config,
        reconciliation_config,
        rate_limit_config,
    )
    .with_dynamic_risk_config(dynamic_risk_config)
    .with_microprice_ema(microprice_ema_alpha, microprice_ema_min_change_bps);

    // Wire checkpoint persistence for warm-starting across sessions
    let checkpoint_dir = PathBuf::from(format!("data/checkpoints/{asset}"));
    market_maker = market_maker.with_checkpoint_dir(checkpoint_dir);

    // Wire signal export path for diagnostic infrastructure
    if let Some(ref path) = cli.signal_export_path {
        market_maker = market_maker.with_signal_export_path(path.clone());
        tracing::info!(
            export_path = %path,
            "Signal diagnostic infrastructure enabled - fills will be exported with all signal values"
        );
    }

    // Wire Binance feed for cross-exchange lead-lag signal.
    // Auto-derive Binance symbol from asset; disable for HL-native tokens.
    if !cli.disable_binance_feed {
        let binance_symbol = resolve_binance_symbol(
            &asset,
            cli.binance_symbol.as_deref(),
        );
        if let Some(ref sym) = binance_symbol {
            let (tx, rx) = tokio::sync::mpsc::channel(1000);
            let feed = BinanceFeed::for_symbol(sym, tx);
            tokio::spawn(async move {
                feed.run().await;
                tracing::warn!("Binance feed task terminated");
            });
            market_maker = market_maker.with_binance_receiver(rx);
            tracing::info!(
                asset = %asset,
                binance_symbol = %sym,
                "Binance lead-lag feed active (auto-derived from asset)"
            );
        } else {
            tracing::warn!(
                asset = %asset,
                "No Binance equivalent for asset — cross-venue signal disabled. \
                 Use --binance-symbol to override if a correlated pair exists."
            );
            market_maker.disable_binance_signals();
            tracing::info!("Disabled Binance-dependent signals to prevent staleness widening");
        }
    } else {
        tracing::warn!("Binance lead-lag feed DISABLED - running without cross-exchange signal");
    }

    // === Prior Injection: Load paper checkpoint and inject full prior (φ→ψ) ===
    // Auto-resolve: use --paper-checkpoint if given, else check default paper prior path
    let paper_ckpt_path_resolved = cli.paper_checkpoint.clone().or_else(|| {
        let default_path = format!("data/checkpoints/paper/{asset}");
        if std::path::Path::new(&default_path).join("prior.json").exists() {
            Some(default_path)
        } else {
            None
        }
    });
    if let Some(ref paper_ckpt_path) = paper_ckpt_path_resolved {
        use hyperliquid_rust_sdk::market_maker::checkpoint::{PriorInject, transfer::InjectionConfig};

        // Try multiple candidate paths (paper mode saves to prior.json, old path used latest/checkpoint.json)
        let candidates = [
            std::path::Path::new(paper_ckpt_path).join("prior.json"),
            std::path::Path::new(paper_ckpt_path).join("latest/checkpoint.json"),
            std::path::PathBuf::from(paper_ckpt_path), // direct file path
        ];

        let mut loaded = false;
        for candidate in &candidates {
            if let Ok(json) = std::fs::read_to_string(candidate) {
                match serde_json::from_str::<hyperliquid_rust_sdk::market_maker::checkpoint::CheckpointBundle>(&json) {
                    Ok(bundle) => {
                        let config = InjectionConfig::default();
                        let rl_states = market_maker.inject_prior(&bundle, &config);
                        tracing::info!(
                            path = %candidate.display(),
                            rl_states_injected = rl_states,
                            rl_blend_weight = config.rl_blend_weight,
                            "Injected paper prior into live state (full φ→ψ transfer)"
                        );
                        loaded = true;
                        break;
                    }
                    Err(e) => tracing::warn!(
                        path = %candidate.display(),
                        error = %e,
                        "Failed to parse checkpoint JSON, trying next candidate"
                    ),
                }
            }
        }
        if !loaded {
            tracing::warn!(
                paper_ckpt_path = %paper_ckpt_path,
                "No valid paper checkpoint found (tried prior.json, latest/checkpoint.json, and direct path)"
            );
        }
    }

    // Phase 4: RL now flows through ensemble as RLEdgeModel, weighted by IR.
    // No separate enable/disable — RL influence is proportional to its track record.
    tracing::info!(
        "RL agent integrated into ensemble pipeline (no separate override)"
    );

    // === RL Hot-Reload: Watch checkpoint file for offline trainer updates ===
    if let Some(ref rl_watch_path) = cli.rl_watch {
        let (rl_watch_tx, rl_watch_rx) = tokio::sync::watch::channel(None);
        market_maker = market_maker.with_rl_reload(rl_watch_rx, 0.3);

        let watch_path = std::path::PathBuf::from(rl_watch_path);
        tokio::spawn(async move {
            let mut last_modified = None;
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                if let Ok(metadata) = tokio::fs::metadata(&watch_path).await {
                    if let Ok(modified) = metadata.modified() {
                        if last_modified.as_ref() != Some(&modified) {
                            last_modified = Some(modified);
                            if let Ok(contents) = tokio::fs::read_to_string(&watch_path).await {
                                if let Ok(checkpoint) = serde_json::from_str::<
                                    hyperliquid_rust_sdk::market_maker::checkpoint::types::RLCheckpoint,
                                >(&contents) {
                                    let _ = rl_watch_tx.send(Some(checkpoint));
                                    tracing::info!(
                                        path = %watch_path.display(),
                                        "RL checkpoint file changed, sent for hot-reload"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    // Sync open orders
    market_maker
        .sync_open_orders()
        .await
        .map_err(|e| format!("Failed to sync open orders: {e}"))?;

    // Start HTTP metrics endpoint if enabled
    let metrics_port = cli.metrics_port.unwrap_or(config.monitoring.metrics_port);
    if config.monitoring.enable_http_metrics && metrics_port > 0 {
        let prometheus = market_maker.prometheus().clone();
        let asset_for_metrics = asset.clone();
        let quote_for_metrics = collateral_symbol.clone();

        tokio::spawn(async move {
            // Clone prometheus for dashboard endpoint
            let prometheus_for_dashboard = prometheus.clone();

            // CORS layer to allow dashboard at localhost:3000 to fetch from localhost:8080
            let cors = CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any);

            let app = Router::new()
                .route(
                    "/metrics",
                    get(move || {
                        let prom = prometheus.clone();
                        let asset = asset_for_metrics.clone();
                        let quote = quote_for_metrics.clone();
                        async move { prom.to_prometheus_text(&asset, &quote) }
                    }),
                )
                .route(
                    "/api/dashboard",
                    get(move || {
                        let prom = prometheus_for_dashboard.clone();
                        async move { Json(prom.to_dashboard_state()) }
                    }),
                )
                // Health check endpoints for orchestration (k8s, load balancers)
                .route("/healthz", get(|| async { axum::http::StatusCode::OK }))
                .route("/readyz", get(|| async { axum::http::StatusCode::OK }))
                .layer(cors);

            // Security: Bind to localhost only to prevent metrics exposure to network
            let addr = format!("127.0.0.1:{metrics_port}");
            info!(
                port = metrics_port,
                bind = "127.0.0.1",
                "Starting Prometheus metrics endpoint (localhost only)"
            );

            match tokio::net::TcpListener::bind(&addr).await {
                Ok(listener) => {
                    if let Err(e) = axum::serve(listener, app).await {
                        warn!(error = %e, "Metrics server error");
                    }
                }
                Err(e) => {
                    warn!(error = %e, port = metrics_port, "Failed to bind metrics port");
                }
            }
        });
    }

    // Dry-run mode: validate everything but don't start trading
    if cli.dry_run {
        info!("=== DRY RUN MODE ===");
        info!("Configuration validated successfully");
        info!("Exchange connection established");
        info!(
            position = %initial_position,
            open_orders = market_maker.orders_count(),
            "Account state verified"
        );
        info!("Exiting dry-run mode (no orders placed)");
        return Ok(());
    }

    info!("Market maker initialized, starting main loop...");
    market_maker
        .start()
        .await
        .map_err(|e| format!("Market maker error: {e}"))?;

    Ok(())
}

// ============================================================================
// Helper Functions
// ============================================================================

fn load_config(cli: &Cli) -> Result<AppConfig, Box<dyn std::error::Error>> {
    let config_path = &cli.config;
    if Path::new(config_path).exists() {
        let content = std::fs::read_to_string(config_path)?;
        let config: AppConfig = toml::from_str(&content)?;
        Ok(config)
    } else {
        // Return default config if file doesn't exist
        Ok(AppConfig::default())
    }
}

/// Static storage for log guards to keep them alive for the program duration.
/// These guards ensure async log writes are flushed before program exit.
static LOG_GUARDS: std::sync::OnceLock<Vec<tracing_appender::non_blocking::WorkerGuard>> =
    std::sync::OnceLock::new();

fn setup_logging(config: &AppConfig, cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let level = cli.log_level.as_ref().unwrap_or(&config.logging.level);

    // Determine stdout format from CLI or config
    let stdout_format = match cli.log_format.as_deref() {
        Some("json") => MmLogFormat::Json,
        Some("compact") => MmLogFormat::Compact,
        Some("pretty") | Some(_) => MmLogFormat::Pretty,
        None => match config.logging.format {
            LogFormat::Json => MmLogFormat::Json,
            LogFormat::Compact => MmLogFormat::Compact,
            LogFormat::Pretty => MmLogFormat::Pretty,
        },
    };

    // Check for multi-stream mode (CLI flag takes precedence)
    let enable_multi_stream = cli.multi_stream_logs || config.logging.multi_stream;

    // Get log directory
    let log_dir = cli
        .log_dir
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(&config.logging.log_dir));

    // Get log file path from CLI or config (legacy single-file mode)
    let log_file = cli
        .log_file
        .as_ref()
        .or(config.logging.log_file.as_ref())
        .cloned();

    // Build LogConfig
    let log_config = LogConfig {
        log_dir,
        enable_multi_stream,
        operational_level: "info".to_string(),
        diagnostic_level: "debug".to_string(),
        error_level: "warn".to_string(),
        enable_stdout: true,
        stdout_format,
        log_file,
    };

    // Initialize logging using the new module
    let guards = init_logging(&log_config, Some(level))?;

    // Store guards to keep them alive
    let _ = LOG_GUARDS.set(guards);

    Ok(())
}

fn parse_base_url(s: &str) -> Result<BaseUrl, Box<dyn std::error::Error>> {
    match s.to_lowercase().as_str() {
        "mainnet" => Ok(BaseUrl::Mainnet),
        "testnet" => Ok(BaseUrl::Testnet),
        "localhost" => Ok(BaseUrl::Localhost),
        _ => Err(format!("Unknown network '{s}'. Use: mainnet, testnet, localhost").into()),
    }
}

fn generate_sample_config(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let sample = AppConfig::default();
    let content = toml::to_string_pretty(&sample)?;

    let with_comments = format!(
        r#"# Hyperliquid Market Maker Configuration
# See: cargo run --bin market_maker -- --help

{content}

# Note: Set private key via HYPERLIQUID_PRIVATE_KEY environment variable
# or uncomment below (not recommended for security):
# [network]
# private_key = "your_private_key_here"
"#
    );

    std::fs::write(path, with_comments)?;
    println!("Sample config written to: {path}");
    Ok(())
}

/// Print startup banner with version and configuration summary.
fn print_startup_banner(asset: &str, quote_asset: &str, network: &BaseUrl, dry_run: bool) {
    let version = env!("CARGO_PKG_VERSION");
    let mode = if dry_run { " [DRY RUN]" } else { "" };
    let network_str = match network {
        BaseUrl::Mainnet => "MAINNET",
        BaseUrl::Testnet => "testnet",
        BaseUrl::Localhost => "localhost",
    };

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════╗");
    eprintln!(
        "║     Hyperliquid Market Maker v{version:<10}{mode}              ║"
    );
    eprintln!("║                                                           ║");
    eprintln!(
        "║  Asset:   {asset:<15}  Network: {network_str:<15}   ║"
    );
    eprintln!(
        "║  Quote:   {quote_asset:<15}                                ║"
    );
    eprintln!("╚═══════════════════════════════════════════════════════════╝");
    eprintln!();
}

/// List available HIP-3 DEXs.
async fn list_available_dexs(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config(cli)?;
    let base_url = parse_base_url(cli.network.as_ref().unwrap_or(&config.network.base_url))?;

    println!("Connecting to {base_url:?}...");

    let info_client = InfoClient::with_reconnect(None, Some(base_url)).await?;

    // Query available DEXs
    let dexs = info_client
        .perp_dexs()
        .await
        .map_err(|e| format!("Failed to query DEXs: {e}"))?;

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  Available HIP-3 DEXs");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let mut found_any = false;
    for (idx, dex_opt) in dexs.iter().enumerate() {
        if let Some(dex) = dex_opt {
            found_any = true;
            println!("  [{:>2}] {} - {}", idx, dex.name, dex.full_name);
            println!("       Deployer: {}", dex.deployer);
            if let Some(ref oracle) = dex.oracle_updater {
                println!("       Oracle:   {oracle}");
            }
            println!();
        }
    }

    if !found_any {
        println!("  No HIP-3 DEXs found on this network.");
        println!();
    }

    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("Usage: cargo run --bin market_maker -- --asset BTC --dex <name>");
    println!();

    Ok(())
}

/// Run paper trading mode using the unified MarketMaker core + PaperEnvironment.
///
/// This uses the same MarketMaker<S, PaperEnvironment> as the unified architecture,
/// consuming market data and synthesizing fills through the FillSimulator.
async fn run_paper_mode(cli: &Cli, duration: u64) -> Result<(), Box<dyn std::error::Error>> {
    use hyperliquid_rust_sdk::market_maker::{
        environment::paper::{PaperEnvironment, PaperEnvironmentConfig},
        simulation::fill_sim::FillSimulatorConfig,
        checkpoint::PriorExtract,
    };

    let config = load_config(cli)?;

    // Setup logging (reuse the same setup as live mode)
    setup_logging(&config, cli)?;

    // Get private key (same priority as live: CLI > config > env)
    let private_key = cli
        .private_key
        .clone()
        .or(config.network.private_key.clone())
        .ok_or("Private key required. Set via --private-key, HYPERLIQUID_PRIVATE_KEY env var, or config file.")?;
    let wallet: PrivateKeySigner = private_key
        .parse()
        .map_err(|_| "Invalid private key format")?;
    let user_address = wallet.address();

    // Parse network (same as live)
    let base_url = parse_base_url(cli.network.as_ref().unwrap_or(&config.network.base_url))?;

    // Create info clients — one for PaperEnvironment subscriptions, one for MarketMaker state queries
    let info_client = InfoClient::with_reconnect(None, Some(base_url)).await?;
    let info_client_for_mm = InfoClient::with_reconnect(None, Some(base_url)).await?;

    // Resolve asset (same as live: CLI > config, with DEX prefix)
    let dex = cli.dex.clone();
    let asset = cli.asset.clone().unwrap_or(config.trading.asset.clone());
    let asset = if let Some(ref dex_name) = dex {
        if asset.contains(':') {
            asset
        } else {
            format!("{dex_name}:{asset}")
        }
    } else {
        asset
    };

    // Query metadata for sz_decimals (use the MM client, not the paper env one)
    let meta = info_client_for_mm
        .meta_for_dex(dex.as_deref())
        .await
        .map_err(|e| format!("Failed to get metadata: {e}"))?;

    let asset_meta = meta
        .universe
        .iter()
        .find(|a| a.name == asset)
        .ok_or_else(|| format!("Asset '{asset}' not found"))?;
    let sz_decimals = asset_meta.sz_decimals;
    let runtime_config = AssetRuntimeConfig::from_asset_meta(asset_meta);

    // Auto-calculate decimals
    let decimals = cli.decimals.or(config.trading.decimals).unwrap_or_else(|| {
        let is_spot = meta.universe.iter().position(|a| a.name == asset).map(|i| i >= 10000).unwrap_or(false);
        let max_decimals: u32 = if is_spot { 8 } else { 6 };
        max_decimals.saturating_sub(sz_decimals)
    });

    // Position limit: conservative default for paper
    let max_position = cli.max_position.unwrap_or(
        config.trading.max_absolute_position_size.unwrap_or(1.0),
    );

    // Resolve collateral
    let collateral = if let Some(token_index) = meta.collateral_token {
        let spot_meta = info_client_for_mm.spot_meta().await
            .map_err(|e| format!("Failed to get spot metadata: {e}"))?;
        CollateralInfo::from_token_index(token_index, &spot_meta)
    } else {
        CollateralInfo::usdc()
    };

    info!(
        asset = %asset,
        duration_s = duration,
        max_position = %max_position,
        decimals = decimals,
        "Starting paper trading mode"
    );

    // Create PaperEnvironment
    let paper_config = PaperEnvironmentConfig {
        asset: asset.clone(),
        user_address,
        fill_sim_config: FillSimulatorConfig::default(),
        dex: dex.clone(),
    };
    let paper_env = PaperEnvironment::new(paper_config, info_client);

    // Build MmConfig (same struct literal pattern as live, simplified)
    // Paper trader uses conservative defaults when config values are None
    let risk_aversion = cli.risk_aversion.or(config.trading.risk_aversion).unwrap_or(0.3);
    let mm_config = MmConfig {
        asset: Arc::from(asset.as_str()),
        target_liquidity: cli.target_liquidity.or(config.trading.target_liquidity).unwrap_or(0.01),
        risk_aversion,
        max_bps_diff: cli.max_bps_diff.or(config.trading.max_bps_diff).unwrap_or(5),
        max_position,
        max_position_usd: 0.0,
        max_position_user_specified: cli.max_position.is_some(),
        decimals,
        sz_decimals,
        multi_asset: false,
        stochastic: StochasticConfig::default(),
        smart_reconcile: true,
        reconcile: ReconcileConfig::default(),
        runtime: runtime_config,
        initial_isolated_margin: cli.initial_isolated_margin,
        dex: dex.clone(),
        collateral,
        impulse_control: ImpulseControlConfig::default(),
        spread_profile: SpreadProfile::from_str(&cli.spread_profile),
        fee_bps: 1.5,
    };

    // Create strategy (Ladder with sensible defaults)
    let strategy: Box<dyn QuotingStrategy> = Box::new(
        LadderStrategy::new(risk_aversion),
    );

    // Create metrics
    let metrics = Arc::new(MarketMakerMetrics::new());

    // Module configs (all defaults for paper)
    let estimator_config = EstimatorConfig::from_legacy(
        config.strategy.estimation_window_secs * 1000,
        config.strategy.min_trades,
        config.strategy.default_sigma,
        config.strategy.default_kappa,
        config.strategy.default_arrival_intensity,
        config.strategy.warmup_decay_secs,
        config.strategy.min_warmup_trades,
    );

    let mut market_maker = MarketMaker::new(
        mm_config,
        strategy,
        paper_env,
        info_client_for_mm,
        user_address,
        0.0, // Initial position = 0 for paper
        Some(metrics),
        estimator_config,
        AdverseSelectionConfig::default(),
        QueueConfig::default(),
        LiquidationConfig::default(),
        KillSwitchConfig::default(),
        HawkesConfig::default(),
        FundingConfig::default(),
        SpreadConfig::default(),
        PnLConfig::default(),
        MarginConfig::default(),
        DataQualityConfig::default(),
        RecoveryConfig::default(),
        ReconciliationConfig::default(),
        RejectionRateLimitConfig::default(),
    );

    // Wire checkpoint persistence
    let checkpoint_dir = PathBuf::from(format!("data/checkpoints/paper/{asset}"));
    market_maker = market_maker.with_checkpoint_dir(checkpoint_dir);

    // Disable Binance signals for paper (no cross-venue feed)
    market_maker.disable_binance_signals();

    // Seed paper balance — without this, margin_available=0 and no orders are placed.
    // Paper balance should be a realistic account size, NOT the max_position_usd (which
    // is just the position limit). Default $1000 ensures orders pass the $10 minimum
    // notional check even for low-priced assets.
    let paper_balance = 1000.0_f64;
    market_maker = market_maker.with_paper_balance(paper_balance);

    // Run with optional duration
    if duration > 0 {
        info!(duration_s = duration, "Paper trading with time limit");
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(duration),
            market_maker.run(),
        )
        .await;

        match result {
            Ok(Ok(())) => info!("Paper trading completed normally"),
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => {
                info!("Paper trading duration elapsed, extracting prior...");
                let mut prior = market_maker.extract_prior();
                // Stamp readiness assessment before saving
                let gate = hyperliquid_rust_sdk::market_maker::calibration::gate::CalibrationGate::new(
                    hyperliquid_rust_sdk::market_maker::calibration::gate::CalibrationGateConfig::default(),
                );
                prior.readiness = gate.assess(&prior);
                info!(
                    verdict = ?prior.readiness.verdict,
                    estimators_ready = prior.readiness.estimators_ready,
                    "Prior readiness assessed"
                );
                let checkpoint_path = format!("data/checkpoints/paper/{asset}/prior.json");
                if let Ok(json) = serde_json::to_string_pretty(&prior) {
                    std::fs::create_dir_all(format!("data/checkpoints/paper/{asset}"))?;
                    std::fs::write(&checkpoint_path, json)?;
                    info!(path = %checkpoint_path, "Paper prior saved");
                }
            }
        }
    } else {
        market_maker.run().await?;
    }

    Ok(())
}

/// Show account status without starting the market maker.
async fn show_account_status(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config(cli)?;

    // Get private key
    let private_key = cli
        .private_key
        .clone()
        .or(config.network.private_key.clone())
        .ok_or("Private key required for status command")?;

    let wallet: PrivateKeySigner = private_key
        .parse()
        .map_err(|_| "Invalid private key format")?;

    let base_url = parse_base_url(cli.network.as_ref().unwrap_or(&config.network.base_url))?;
    let asset = cli.asset.clone().unwrap_or(config.trading.asset.clone());

    println!("Connecting to {base_url:?}...");

    let info_client = InfoClient::with_reconnect(None, Some(base_url)).await?;
    let user_address = wallet.address();

    // Get user state (use DEX-specific clearinghouse state for HIP-3)
    let dex = cli.dex.as_deref();
    let user_state = info_client.user_state_for_dex(user_address, dex).await?;

    // Get open orders (use DEX-aware for HIP-3)
    let open_orders = info_client.open_orders_for_dex(user_address, dex).await?;
    let asset_orders: Vec<_> = open_orders.iter().filter(|o| o.coin == asset).collect();

    // Get asset-specific data
    let asset_data = info_client
        .active_asset_data(user_address, asset.clone())
        .await
        .ok();

    // Print status
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  Account Status for {asset}");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("  Wallet:          {user_address}");
    println!();

    // Account summary
    // NOTE: Use cross_margin_summary for cross margin accounts (default)
    println!(
        "  Account Value:   ${}",
        user_state.cross_margin_summary.account_value
    );
    println!(
        "  Total Notional:  ${}",
        user_state.cross_margin_summary.total_ntl_pos
    );
    println!(
        "  Margin Used:     ${}",
        user_state.cross_margin_summary.total_margin_used
    );
    println!();

    // Position
    let position = user_state
        .asset_positions
        .iter()
        .find(|p| p.position.coin == asset);

    if let Some(pos) = position {
        println!("  Position:        {} {}", pos.position.szi, asset);
        if let Some(ref entry_px) = pos.position.entry_px {
            println!("  Entry Price:     ${entry_px}");
        }
        println!("  Unrealized PnL:  ${}", pos.position.unrealized_pnl);
        println!("  Leverage:        {}x", pos.position.leverage.value);
    } else {
        println!("  Position:        None");
    }
    println!();

    // Mark price
    if let Some(data) = asset_data {
        println!("  Mark Price:      ${}", data.mark_px);
    }

    // Open orders
    println!("  Open Orders:     {}", asset_orders.len());
    if !asset_orders.is_empty() {
        println!();
        for order in &asset_orders {
            let side = if order.side == "B" { "BUY " } else { "SELL" };
            println!("    {} {} {} @ ${}", side, order.sz, asset, order.limit_px);
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!();

    Ok(())
}
