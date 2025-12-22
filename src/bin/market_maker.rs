//! Enhanced Market Maker Tool for Hyperliquid
//!
//! A production-ready market making tool with:
//! - CLI arguments and TOML config file support
//! - Multiple quoting strategies
//! - Structured logging with tracing
//! - Metrics collection

use alloy::signers::local::PrivateKeySigner;
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;
use tracing_subscriber::EnvFilter;

use hyperliquid_rust_sdk::{
    BaseUrl, ExchangeClient, HyperliquidExecutor, InfoClient, InventoryAwareStrategy,
    MarketMaker, MarketMakerConfig as MmConfig, MarketMakerMetricsRecorder, QuotingStrategy,
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

    /// Override asset from config
    #[arg(long)]
    asset: Option<String>,

    /// Override target liquidity
    #[arg(long)]
    target_liquidity: Option<f64>,

    /// Override half spread in BPS
    #[arg(long)]
    half_spread: Option<u16>,

    /// Override max BPS diff before requoting
    #[arg(long)]
    max_bps_diff: Option<u16>,

    /// Override max position size
    #[arg(long)]
    max_position: Option<f64>,

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

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TradingConfig {
    /// Asset to market make on (e.g., "ETH", "BTC")
    #[serde(default = "default_asset")]
    pub asset: String,
    /// Amount of liquidity to target on each side
    #[serde(default = "default_target_liquidity")]
    pub target_liquidity: f64,
    /// Half spread in basis points
    #[serde(default = "default_half_spread")]
    pub half_spread_bps: u16,
    /// Max deviation before requoting (in BPS)
    #[serde(default = "default_max_bps_diff")]
    pub max_bps_diff: u16,
    /// Maximum absolute position size
    #[serde(default = "default_max_position")]
    pub max_absolute_position_size: f64,
    /// Decimals for price rounding (auto-calculated from asset metadata if not set)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decimals: Option<u32>,
}

fn default_asset() -> String {
    "ETH".to_string()
}
fn default_target_liquidity() -> f64 {
    0.25
}
fn default_half_spread() -> u16 {
    1
}
fn default_max_bps_diff() -> u16 {
    2
}
fn default_max_position() -> f64 {
    0.5
}

impl Default for TradingConfig {
    fn default() -> Self {
        Self {
            asset: default_asset(),
            target_liquidity: default_target_liquidity(),
            half_spread_bps: default_half_spread(),
            max_bps_diff: default_max_bps_diff(),
            max_absolute_position_size: default_max_position(),
            decimals: None, // Auto-calculated from asset metadata
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StrategyConfig {
    /// Strategy type: symmetric, inventory_aware
    #[serde(default)]
    pub strategy_type: StrategyType,
    /// Skew factor for inventory-aware strategy (BPS per unit position)
    #[serde(default)]
    pub inventory_skew_factor: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            strategy_type: StrategyType::default(),
            inventory_skew_factor: 0.0,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StrategyType {
    #[default]
    Symmetric,
    InventoryAware,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoggingConfig {
    /// Log level: trace, debug, info, warn, error
    #[serde(default = "default_log_level")]
    pub level: String,
    /// Output format: pretty, json, compact
    #[serde(default)]
    pub format: LogFormat,
}

fn default_log_level() -> String {
    "info".to_string()
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            format: LogFormat::default(),
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
            self.volume_bought_scaled.fetch_add(scaled, Ordering::Relaxed);
        } else {
            self.volume_sold_scaled.fetch_add(scaled, Ordering::Relaxed);
        }
    }

    pub fn update_position(&self, position: f64) {
        let scaled = (position * 1e8) as i64;
        self.current_position_scaled.store(scaled, Ordering::Relaxed);
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
            self.volume_bought_scaled.fetch_add(scaled, Ordering::Relaxed);
        } else {
            self.volume_sold_scaled.fetch_add(scaled, Ordering::Relaxed);
        }
    }

    fn update_position(&self, position: f64) {
        let scaled = (position * 1e8) as i64;
        self.current_position_scaled.store(scaled, Ordering::Relaxed);
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
    let target_liquidity = cli.target_liquidity.unwrap_or(config.trading.target_liquidity);
    let half_spread = cli.half_spread.unwrap_or(config.trading.half_spread_bps);
    let max_bps_diff = cli.max_bps_diff.unwrap_or(config.trading.max_bps_diff);
    let max_position = cli.max_position.unwrap_or(config.trading.max_absolute_position_size);

    // Query metadata to get sz_decimals (always needed for size precision)
    let info_client = InfoClient::new(None, Some(base_url)).await?;
    let meta = info_client.meta().await
        .map_err(|e| format!("Failed to get metadata: {e}"))?;

    let asset_meta = meta.universe.iter()
        .find(|a| a.name == asset)
        .ok_or_else(|| format!("Asset '{}' not found in metadata", asset))?;

    let sz_decimals = asset_meta.sz_decimals;

    // Auto-calculate price decimals from asset metadata if not explicitly set
    let decimals = match cli.decimals.or(config.trading.decimals) {
        Some(d) => d,
        None => {
            // Hyperliquid requires: price_decimals + sz_decimals = 6 (perpetuals) or 8 (spot)
            // Perpetual assets have index < 10000, spot assets >= 10000
            let is_spot = meta.universe.iter()
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

    info!(
        asset = %asset,
        target_liquidity = %target_liquidity,
        half_spread_bps = %half_spread,
        max_bps_diff = %max_bps_diff,
        max_position = %max_position,
        decimals = %decimals,
        strategy = ?config.strategy.strategy_type,
        network = ?base_url,
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

    // Query initial position
    let user_address = wallet.address();
    let user_state = info_client.user_state(user_address).await
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

    // Query available margin and cap liquidity/position accordingly
    let (target_liquidity, max_position) = {
        match info_client.active_asset_data(user_address, asset.clone()).await {
            Ok(data) => {
                let mark_px: f64 = data.mark_px.parse().unwrap_or(1.0);
                let available_usdc: f64 = data
                    .available_to_trade
                    .first()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                let available_asset = available_usdc / mark_px;

                info!(
                    available_usdc = %available_usdc,
                    available_asset = %available_asset,
                    mark_px = %mark_px,
                    "Queried available margin"
                );

                // Cap to 40% for each side's liquidity, 80% for max position
                let capped_liquidity = target_liquidity.min(available_asset * 0.4);
                let capped_max_pos = max_position.min(available_asset * 0.8);

                if capped_liquidity < target_liquidity {
                    info!(
                        requested = %target_liquidity,
                        capped = %capped_liquidity,
                        "Capped target_liquidity to available margin"
                    );
                }

                (capped_liquidity, capped_max_pos)
            }
            Err(e) => {
                tracing::warn!("Failed to query margin: {e}, using defaults");
                (target_liquidity, max_position)
            }
        }
    };

    // Create market maker config
    let mm_config = MmConfig {
        asset: asset.clone(),
        target_liquidity,
        half_spread_bps: half_spread,
        max_bps_diff,
        max_position,
        decimals,
        sz_decimals,
    };

    // Create strategy based on config
    let strategy: Box<dyn QuotingStrategy> = match config.strategy.strategy_type {
        StrategyType::Symmetric => Box::new(SymmetricStrategy::new()),
        StrategyType::InventoryAware => Box::new(InventoryAwareStrategy::new(
            config.strategy.inventory_skew_factor,
        )),
    };

    // Create executor
    let executor = HyperliquidExecutor::new(exchange_client, Some(metrics.clone()));

    // Create and start market maker
    let mut market_maker = MarketMaker::new(
        mm_config,
        strategy,
        executor,
        info_client,
        user_address,
        initial_position,
        Some(metrics),
    );

    // Sync open orders
    market_maker.sync_open_orders().await
        .map_err(|e| format!("Failed to sync open orders: {e}"))?;

    info!("Market maker initialized, starting main loop...");
    market_maker.start().await
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

fn setup_logging(
    config: &AppConfig,
    cli: &Cli,
) -> Result<(), Box<dyn std::error::Error>> {
    let level = cli.log_level.as_ref().unwrap_or(&config.logging.level);

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(level)
            .add_directive("hyper=warn".parse().unwrap())
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
