//! Enhanced Market Maker Tool for Hyperliquid
//!
//! A production-ready market making tool with:
//! - CLI arguments and TOML config file support
//! - Multiple quoting strategies
//! - Structured logging with tracing
//! - Metrics collection
//! - Prometheus HTTP metrics endpoint

use alloy::signers::local::PrivateKeySigner;
use axum::{routing::get, Router};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

use hyperliquid_rust_sdk::{
    init_logging, AdverseSelectionConfig, AssetRuntimeConfig, BaseUrl, CollateralInfo,
    DataQualityConfig, DynamicRiskConfig, EstimatorConfig, ExchangeClient, FundingConfig,
    GLFTStrategy, HawkesConfig, HyperliquidExecutor, InfoClient, InventoryAwareStrategy,
    KillSwitchConfig, LadderConfig, LadderStrategy, LiquidationConfig, LogConfig,
    LogFormat as MmLogFormat, MarginConfig, MarketMaker, MarketMakerConfig as MmConfig,
    MarketMakerMetricsRecorder, PnLConfig, QueueConfig, QuotingStrategy, ReconcileConfig,
    ReconciliationConfig, RecoveryConfig, RejectionRateLimitConfig, RiskConfig, SpreadConfig,
    StochasticConfig, SymmetricStrategy,
};

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

    /// Override max position size
    #[arg(long)]
    max_position: Option<f64>,

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

    /// Target liquidity per side in asset units.
    /// Default: 0.01 BTC (~$1K notional) - conservative, safe for any account.
    /// Capped at runtime to 40% of max_position based on account equity.
    #[serde(default = "default_target_liquidity")]
    pub target_liquidity: f64,

    /// Risk aversion parameter (gamma) - controls spread width and inventory skew.
    /// Default: 0.3 (matches RiskConfig.gamma_base for consistency)
    /// Range: 0.1 (aggressive, tight spreads) to 1.0+ (conservative, wide spreads)
    /// GLFT formula: δ = (1/γ) × ln(1 + γ/κ)
    #[serde(default = "default_risk_aversion")]
    pub risk_aversion: f64,

    /// Maximum price deviation (in basis points) before requoting.
    /// Default: 5 bps - balances tight quoting with reduced order churn.
    /// Lower values (2 bps) cause excessive cancels in volatile crypto markets.
    #[serde(default = "default_max_bps_diff")]
    pub max_bps_diff: u16,

    /// Maximum position in contracts (optional).
    /// If not specified, defaults to margin-based limit: (account_value × leverage × 0.5) / price
    /// This is capital-efficient: use gamma to control risk, not arbitrary position limits.
    #[serde(
        default = "default_max_position",
        skip_serializing_if = "Option::is_none"
    )]
    pub max_absolute_position_size: Option<f64>,

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

/// Default target liquidity: 0.01 BTC (~$1K at $100K BTC)
/// Conservative default safe for any account size.
/// Will be capped at runtime to 40% of max_position based on account equity.
fn default_target_liquidity() -> f64 {
    0.01
}

/// Default risk aversion (gamma): 0.3
/// Matches RiskConfig.gamma_base for consistency.
/// Range: 0.1 (aggressive) to 1.0+ (conservative)
/// In GLFT: δ = (1/γ) × ln(1 + γ/κ) - higher γ means wider spreads
fn default_risk_aversion() -> f64 {
    0.3
}

/// Default max BPS diff before requoting: 5 bps
/// Crypto markets are volatile - 2 bps causes excessive order churn.
/// 5 bps balances tight quoting with practical order management.
fn default_max_bps_diff() -> u16 {
    5
}

/// Default max position: 0.05 BTC (~$5K at $100K BTC)
/// Conservative default below kill switch limit ($10K).
/// Will be capped at runtime by (account_value × leverage × 0.5) / price.
fn default_max_position() -> Option<f64> {
    None // Default to margin-based limit (capital-efficient)
}

impl Default for TradingConfig {
    fn default() -> Self {
        Self {
            asset: default_asset(),
            target_liquidity: default_target_liquidity(),
            risk_aversion: default_risk_aversion(),
            max_bps_diff: default_max_bps_diff(),
            max_absolute_position_size: default_max_position(),
            leverage: None, // Uses max available from asset metadata
            decimals: None, // Auto-calculated from asset metadata
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
            println!("Configuration is valid:\n{:#?}", config);
            return Ok(());
        }
        Some(Commands::Status) => {
            return show_account_status(&cli).await;
        }
        Some(Commands::Run) | None => {
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
    let target_liquidity = cli
        .target_liquidity
        .unwrap_or(config.trading.target_liquidity);
    let risk_aversion = cli.risk_aversion.unwrap_or(config.trading.risk_aversion);
    let max_bps_diff = cli.max_bps_diff.unwrap_or(config.trading.max_bps_diff);
    // max_position is now optional - will default to margin-based limit later
    // Priority: CLI arg > config file > margin-based default
    let max_position_override: Option<f64> = cli
        .max_position
        .or(config.trading.max_absolute_position_size);

    // Query metadata to get sz_decimals (always needed for size precision)
    // Use with_reconnect to automatically reconnect if WebSocket disconnects
    let info_client = InfoClient::with_reconnect(None, Some(base_url)).await?;

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
            let prefixed = format!("{}:{}", dex_name, asset);
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
        target_liquidity = %target_liquidity,
        risk_aversion = %risk_aversion,
        max_bps_diff = %max_bps_diff,
        max_position_override = ?max_position_override,
        leverage = %leverage,
        decimals = %decimals,
        strategy = ?config.strategy.strategy_type,
        network = ?base_url,
        dry_run = cli.dry_run,
        "Starting market maker"
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
    let exchange_client = ExchangeClient::new_for_dex(
        None,
        wallet.clone(),
        Some(base_url),
        Some(meta.clone()),
        None,
        dex.as_deref(),
    )
    .await
    .map_err(|e| format!("Failed to create exchange client: {e}"))?;

    // Set leverage on the exchange (use is_cross from runtime config for HIP-3 support)
    info!(
        leverage = leverage,
        asset = %asset,
        is_cross = runtime_config.is_cross,
        "Setting leverage on exchange"
    );
    match exchange_client
        .update_leverage(leverage, &asset, runtime_config.is_cross, None)
        .await
    {
        Ok(response) => {
            info!(leverage = leverage, response = ?response, "Leverage set successfully");
        }
        Err(e) => {
            warn!(leverage = leverage, error = %e, "Failed to set leverage, using existing");
        }
    }

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

    // Query account equity for position limits (leverage was already set above)
    let (target_liquidity, max_position) = {
        let asset_data_result = info_client
            .active_asset_data(user_address, asset.clone())
            .await;

        // For HIP-3 DEXs, collateral is in spot balance (USDE, USDH, etc.)
        // For validator perps, collateral is in perps clearinghouse (USDC)
        let account_value: f64 = if dex.is_some() {
            // HIP-3: Get account value from spot balance
            match info_client.user_token_balances(user_address).await {
                Ok(balances) => collateral
                    .available_balance_from_spot(&balances.balances)
                    .unwrap_or(0.0),
                Err(e) => {
                    warn!("Failed to query spot balances for HIP-3: {e}, using 0");
                    0.0
                }
            }
        } else {
            // Validator perps: Get account value from perps clearinghouse
            match info_client.user_state(user_address).await {
                Ok(state) => state.margin_summary.account_value.parse().unwrap_or(0.0),
                Err(e) => {
                    warn!("Failed to query user state: {e}, using 0");
                    0.0
                }
            }
        };

        match asset_data_result {
            Ok(asset_data) => {
                let mark_px: f64 = asset_data.mark_px.parse().unwrap_or(1.0);

                // === CAPITAL-EFFICIENT POSITION LIMITS ===
                // From GLFT theory: max_position should be derived from MARGIN, not arbitrary config.
                // The user controls risk via gamma (utility), not position caps.
                //
                // Formula: max_from_leverage = (account_value × leverage × safety_factor) / price
                // This is the TRUE hard constraint from solvency.
                let safety_factor = 0.5; // Leave 50% margin buffer for adverse moves
                let max_from_leverage = (account_value * leverage as f64 * safety_factor) / mark_px;

                // Effective max_position:
                // - If user specified: use min(user_value, margin_based) - user can only REDUCE
                // - If not specified: use full margin capacity (capital-efficient default)
                let capped_max_pos = match max_position_override {
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
                    None => {
                        info!(
                            margin_limit = %format!("{:.6}", max_from_leverage),
                            "Using margin-based max_position (capital-efficient default)"
                        );
                        max_from_leverage
                    }
                };

                // Minimum notional for multi-level ladder
                // min_notional = $10 (Hyperliquid minimum order notional)
                let min_notional = 10.0;
                let min_viable_liquidity = min_notional / mark_px;

                // Calculate minimum liquidity needed for multi-level ladder
                // With 5 levels and decay, smallest fraction ≈ 5% of total
                let num_ladder_levels = 5_u8;
                let smallest_level_fraction = 0.05;
                let min_for_ladder = min_notional / (mark_px * smallest_level_fraction);

                // Target liquidity: auto-scale for ladder, cap by margin (NOT by max_position)
                // This allows full margin utilization for quoting even with reduce-only active
                let capped_liquidity = target_liquidity
                    .max(min_for_ladder) // Ensure multi-level ladder support
                    .min(max_from_leverage); // Cap by margin, not max_position

                // Log if we auto-increased target_liquidity for multi-level ladder
                if target_liquidity < min_for_ladder {
                    info!(
                        configured_target = %format!("{:.6}", target_liquidity),
                        min_for_ladder = %format!("{:.6}", min_for_ladder),
                        using = %format!("{:.6}", capped_liquidity),
                        "Auto-increased target_liquidity to support {} ladder levels",
                        num_ladder_levels
                    );
                }

                info!(
                    account_value = %format!("{:.2}", account_value),
                    leverage = %leverage,
                    mark_px = %format!("{:.4}", mark_px),
                    max_from_leverage = %format!("{:.6}", max_from_leverage),
                    capped_max_pos = %format!("{:.6}", capped_max_pos),
                    capped_liquidity = %format!("{:.6}", capped_liquidity),
                    min_viable = %format!("{:.6}", min_viable_liquidity),
                    "Capital-efficient position limits (use gamma to control risk)"
                );

                (capped_liquidity, capped_max_pos)
            }
            Err(e) => {
                // Fallback: use config values or conservative defaults
                let fallback_max = max_position_override.unwrap_or(0.05);
                tracing::warn!(
                    fallback_max = %format!("{:.6}", fallback_max),
                    "Failed to query asset data: {e}, using fallback position limit"
                );
                (target_liquidity, fallback_max)
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
    let mm_config = MmConfig {
        asset: Arc::from(asset.as_str()),
        target_liquidity,
        risk_aversion,
        max_bps_diff,
        max_position,
        decimals,
        sz_decimals,
        multi_asset: false, // Single-asset mode by default
        stochastic: StochasticConfig::default(),
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
            rc
        },
        // HIP-3 support: pre-computed runtime config for zero hot-path overhead
        runtime: runtime_config,
        initial_isolated_margin,
        // HIP-3 DEX selection
        dex: dex.clone(),
        // Collateral/quote asset (USDC, USDE, USDH, etc.)
        collateral,
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
            let risk_cfg = config
                .strategy
                .risk_config
                .clone()
                .unwrap_or_else(|| RiskConfig {
                    gamma_base: risk_aversion,
                    ..Default::default()
                });
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

            info!(
                gamma_base = risk_cfg.gamma_base,
                num_levels = ladder_cfg.num_levels,
                min_depth_bps = ladder_cfg.min_depth_bps,
                max_depth_bps = ladder_cfg.max_depth_bps,
                geometric_spacing = ladder_cfg.geometric_spacing,
                max_spread_per_side_bps = ladder_cfg.max_spread_per_side_bps,
                "Using LadderStrategy (multi-level GLFT)"
            );
            Box::new(LadderStrategy::with_config(risk_cfg, ladder_cfg))
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

    // Kappa floor handling: CLI override takes precedence, otherwise use dynamic bounds
    // Dynamic bounds are computed at runtime from Bayesian confidence + credible intervals
    if dex.is_some() {
        // Lower kappa prior for DEX assets (illiquid → trades execute far from mid)
        // Standard perps: 2500 (trades at 4 bps from mid)
        // HIP-3 DEX: 500 (trades at 20 bps from mid)
        estimator_config.kappa_prior_mean = 500.0;
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
    let kill_switch_config = KillSwitchConfig {
        max_position_contracts: max_position, // Use the configured max_position for runaway detection
        ..Default::default()
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

    // Create and start market maker
    let mut market_maker = MarketMaker::new(
        mm_config,
        strategy,
        executor,
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
            let app = Router::new().route(
                "/metrics",
                get(move || {
                    let prom = prometheus.clone();
                    let asset = asset_for_metrics.clone();
                    let quote = quote_for_metrics.clone();
                    async move { prom.to_prometheus_text(&asset, &quote) }
                }),
            );

            // Security: Bind to localhost only to prevent metrics exposure to network
            let addr = format!("127.0.0.1:{}", metrics_port);
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
        _ => Err(format!("Unknown network '{}'. Use: mainnet, testnet, localhost", s).into()),
    }
}

fn generate_sample_config(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let sample = AppConfig::default();
    let content = toml::to_string_pretty(&sample)?;

    let with_comments = format!(
        r#"# Hyperliquid Market Maker Configuration
# See: cargo run --bin market_maker -- --help

{}

# Note: Set private key via HYPERLIQUID_PRIVATE_KEY environment variable
# or uncomment below (not recommended for security):
# [network]
# private_key = "your_private_key_here"
"#,
        content
    );

    std::fs::write(path, with_comments)?;
    println!("Sample config written to: {}", path);
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
        "║     Hyperliquid Market Maker v{:<10}{}              ║",
        version, mode
    );
    eprintln!("║                                                           ║");
    eprintln!(
        "║  Asset:   {:<15}  Network: {:<15}   ║",
        asset, network_str
    );
    eprintln!(
        "║  Quote:   {:<15}                                ║",
        quote_asset
    );
    eprintln!("╚═══════════════════════════════════════════════════════════╝");
    eprintln!();
}

/// List available HIP-3 DEXs.
async fn list_available_dexs(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config(cli)?;
    let base_url = parse_base_url(cli.network.as_ref().unwrap_or(&config.network.base_url))?;

    println!("Connecting to {:?}...", base_url);

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
                println!("       Oracle:   {}", oracle);
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

    println!("Connecting to {:?}...", base_url);

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
    println!("  Account Status for {}", asset);
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("  Wallet:          {}", user_address);
    println!();

    // Account summary
    println!(
        "  Account Value:   ${}",
        user_state.margin_summary.account_value
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
            println!("  Entry Price:     ${}", entry_px);
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
