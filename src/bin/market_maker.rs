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
use std::path::Path;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{info, warn};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

use hyperliquid_rust_sdk::{
    AdverseSelectionConfig, BaseUrl, DataQualityConfig, DynamicRiskConfig, EstimatorConfig,
    ExchangeClient, FundingConfig, GLFTStrategy, HawkesConfig, HyperliquidExecutor, InfoClient,
    InventoryAwareStrategy, KillSwitchConfig, LadderConfig, LadderStrategy, LiquidationConfig,
    MarginConfig, MarketMaker, MarketMakerConfig as MmConfig, MarketMakerMetricsRecorder,
    PnLConfig, QueueConfig, QuotingStrategy, RiskConfig, SpreadConfig, StochasticConfig,
    SymmetricStrategy,
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

    /// Metrics HTTP port (0 to disable)
    #[arg(long)]
    metrics_port: Option<u16>,

    /// Dry run mode: validate config and connect but don't place orders
    #[arg(long)]
    dry_run: bool,

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

    /// Maximum position size in asset units.
    /// Default: 0.05 BTC (~$5K notional) - below kill switch limit.
    /// Capped at runtime by: (account_value × leverage × 0.5) / price
    #[serde(default = "default_max_position")]
    pub max_absolute_position_size: f64,

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
fn default_max_position() -> f64 {
    0.05
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
}

fn default_log_level() -> String {
    "info".to_string()
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            format: LogFormat::default(),
            log_file: None,
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
    let max_position = cli
        .max_position
        .unwrap_or(config.trading.max_absolute_position_size);

    // Query metadata to get sz_decimals (always needed for size precision)
    // Use with_reconnect to automatically reconnect if WebSocket disconnects
    let info_client = InfoClient::with_reconnect(None, Some(base_url)).await?;
    let meta = info_client
        .meta()
        .await
        .map_err(|e| format!("Failed to get metadata: {e}"))?;

    let asset_meta = meta
        .universe
        .iter()
        .find(|a| a.name == asset)
        .ok_or_else(|| format!("Asset '{}' not found in metadata", asset))?;

    let sz_decimals = asset_meta.sz_decimals;

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
    print_startup_banner(&asset, &base_url, cli.dry_run);

    info!(
        asset = %asset,
        target_liquidity = %target_liquidity,
        risk_aversion = %risk_aversion,
        max_bps_diff = %max_bps_diff,
        max_position = %max_position,
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

    // Create exchange client
    let exchange_client = ExchangeClient::new(None, wallet.clone(), Some(base_url), None, None)
        .await
        .map_err(|e| format!("Failed to create exchange client: {e}"))?;

    // Set leverage on the exchange
    info!(leverage = leverage, asset = %asset, "Setting leverage on exchange");
    match exchange_client
        .update_leverage(leverage, &asset, true, None)
        .await
    {
        Ok(response) => {
            info!(leverage = leverage, response = ?response, "Leverage set successfully");
        }
        Err(e) => {
            warn!(leverage = leverage, error = %e, "Failed to set leverage, using existing");
        }
    }

    // Query initial position
    let user_address = wallet.address();
    let user_state = info_client
        .user_state(user_address)
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
        let user_state_result = info_client.user_state(user_address).await;

        match (asset_data_result, user_state_result) {
            (Ok(asset_data), Ok(user_state)) => {
                let mark_px: f64 = asset_data.mark_px.parse().unwrap_or(1.0);
                let account_value: f64 = user_state
                    .margin_summary
                    .account_value
                    .parse()
                    .unwrap_or(0.0);

                // Calculate max position from account equity × leverage (we set leverage above)
                // Use 50% safety factor to leave room for adverse moves
                let max_from_leverage = (account_value * leverage as f64 * 0.5) / mark_px;

                // Cap to configured max_position (if user wants smaller)
                let capped_max_pos = max_position.min(max_from_leverage);

                // Cap liquidity to 40% of max position for each side
                let capped_liquidity = target_liquidity.min(capped_max_pos * 0.4);

                info!(
                    account_value = %account_value,
                    leverage = %leverage,
                    mark_px = %mark_px,
                    max_from_leverage = %format!("{:.6}", max_from_leverage),
                    capped_max_pos = %format!("{:.6}", capped_max_pos),
                    capped_liquidity = %format!("{:.6}", capped_liquidity),
                    "Position limits (equity × leverage × 0.5 / price)"
                );

                (capped_liquidity, capped_max_pos)
            }
            (Err(e), _) => {
                tracing::warn!("Failed to query asset data: {e}, using config defaults");
                (target_liquidity, max_position)
            }
            (_, Err(e)) => {
                tracing::warn!("Failed to query user state: {e}, using config defaults");
                (target_liquidity, max_position)
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

    // Create market maker config
    let mm_config = MmConfig {
        asset: asset.clone(),
        target_liquidity,
        risk_aversion,
        max_bps_diff,
        max_position,
        decimals,
        sz_decimals,
        multi_asset: false, // Single-asset mode by default
        stochastic: StochasticConfig::default(),
    };

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
            let ladder_cfg = config.strategy.ladder_config.clone().unwrap_or_default();
            info!(
                gamma_base = risk_cfg.gamma_base,
                num_levels = ladder_cfg.num_levels,
                min_depth_bps = ladder_cfg.min_depth_bps,
                max_depth_bps = ladder_cfg.max_depth_bps,
                geometric_spacing = ladder_cfg.geometric_spacing,
                "Using LadderStrategy (multi-level GLFT)"
            );
            Box::new(LadderStrategy::with_config(risk_cfg, ladder_cfg))
        }
    };

    // Create estimator config for live parameter estimation (using legacy compatibility)
    let estimator_config = EstimatorConfig::from_legacy(
        config.strategy.estimation_window_secs * 1000,
        config.strategy.min_trades,
        config.strategy.default_sigma,
        config.strategy.default_kappa,
        config.strategy.default_arrival_intensity,
        config.strategy.warmup_decay_secs,
        config.strategy.min_warmup_trades,
    );

    info!(
        estimation_window_secs = config.strategy.estimation_window_secs,
        min_trades = config.strategy.min_trades,
        warmup_decay_secs = config.strategy.warmup_decay_secs,
        min_warmup_trades = config.strategy.min_warmup_trades,
        "Live parameter estimation enabled (adaptive warmup)"
    );

    // Create executor
    let executor = HyperliquidExecutor::new(exchange_client, Some(metrics.clone()));

    // Create Tier 1 module configs with defaults
    let as_config = AdverseSelectionConfig::default();
    let queue_config = QueueConfig::default();
    let liquidation_config = LiquidationConfig::default();

    // Create kill switch config (production safety)
    let kill_switch_config = KillSwitchConfig::default();
    info!(
        max_daily_loss = %kill_switch_config.max_daily_loss,
        max_drawdown = %format!("{:.1}%", kill_switch_config.max_drawdown * 100.0),
        max_position_value = %kill_switch_config.max_position_value,
        stale_data_threshold = ?kill_switch_config.stale_data_threshold,
        "Kill switch enabled"
    );

    // Create Tier 2 module configs with defaults
    let hawkes_config = HawkesConfig::default();
    let funding_config = FundingConfig::default();
    let spread_config = SpreadConfig::default();
    let pnl_config = PnLConfig::default();
    let margin_config = MarginConfig::default();
    let data_quality_config = DataQualityConfig::default();

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
    )
    .with_dynamic_risk_config(dynamic_risk_config);

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

        tokio::spawn(async move {
            let app = Router::new().route(
                "/metrics",
                get(move || {
                    let prom = prometheus.clone();
                    let asset = asset_for_metrics.clone();
                    async move { prom.to_prometheus_text(&asset) }
                }),
            );

            let addr = format!("0.0.0.0:{}", metrics_port);
            info!(port = metrics_port, "Starting Prometheus metrics endpoint");

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

fn setup_logging(config: &AppConfig, cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let level = cli.log_level.as_ref().unwrap_or(&config.logging.level);

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(level)
            .add_directive("hyper::=warn".parse().unwrap())
            .add_directive("reqwest=warn".parse().unwrap())
            .add_directive("tokio_tungstenite=warn".parse().unwrap())
    });

    let format = cli
        .log_format
        .as_deref()
        .unwrap_or(match config.logging.format {
            LogFormat::Json => "json",
            LogFormat::Compact => "compact",
            LogFormat::Pretty => "pretty",
        });

    // Get log file path from CLI or config
    let log_file = cli.log_file.as_ref().or(config.logging.log_file.as_ref());

    if let Some(log_path) = log_file {
        // Create file writer
        let file = std::fs::File::create(log_path)?;
        let file = Mutex::new(file);

        // When logging to file, use JSON format for both (easier to parse)
        let stdout_layer = tracing_subscriber::fmt::layer().json();
        let file_layer = tracing_subscriber::fmt::layer()
            .with_writer(file)
            .with_ansi(false)
            .json();

        tracing_subscriber::registry()
            .with(filter)
            .with(stdout_layer)
            .with(file_layer)
            .init();

        eprintln!(
            "Logging to file: {} (using JSON format for both stdout and file)",
            log_path
        );
    } else {
        // No file, just stdout with requested format
        match format {
            "json" => {
                tracing_subscriber::fmt()
                    .with_env_filter(filter)
                    .json()
                    .init();
            }
            "compact" => {
                tracing_subscriber::fmt()
                    .with_env_filter(filter)
                    .compact()
                    .init();
            }
            _ => {
                tracing_subscriber::fmt()
                    .with_env_filter(filter)
                    .with_target(false)
                    .init();
            }
        }
    }

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
fn print_startup_banner(asset: &str, network: &BaseUrl, dry_run: bool) {
    let version = env!("CARGO_PKG_VERSION");
    let mode = if dry_run { " [DRY RUN]" } else { "" };
    let network_str = match network {
        BaseUrl::Mainnet => "MAINNET",
        BaseUrl::Testnet => "testnet",
        BaseUrl::Localhost => "localhost",
    };

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════╗");
    eprintln!("║     Hyperliquid Market Maker v{:<10}{}              ║", version, mode);
    eprintln!("║                                                           ║");
    eprintln!("║  Asset:   {:<15}  Network: {:<15}   ║", asset, network_str);
    eprintln!("╚═══════════════════════════════════════════════════════════╝");
    eprintln!();
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

    // Get user state
    let user_state = info_client.user_state(user_address).await?;

    // Get open orders
    let open_orders = info_client.open_orders(user_address).await?;
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
    println!("  Account Value:   ${}", user_state.margin_summary.account_value);
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
        println!(
            "  Unrealized PnL:  ${}",
            pos.position.unrealized_pnl
        );
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
            println!(
                "    {} {} {} @ ${}",
                side, order.sz, asset, order.limit_px
            );
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!();

    Ok(())
}
