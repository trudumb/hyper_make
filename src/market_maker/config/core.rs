//! Core market maker configuration types.

use std::sync::Arc;

use crate::market_maker::tracking::ReconcileConfig;
use crate::meta::CollateralInfo;

use super::impulse::ImpulseControlConfig;
use super::runtime::AssetRuntimeConfig;
use super::spread_profile::SpreadProfile;
use super::stochastic::StochasticConfig;

/// Trait for recording market maker metrics.
/// Implement this trait to collect statistics about orders and fills.
pub trait MarketMakerMetricsRecorder: Send + Sync {
    /// Called when an order is successfully placed
    fn record_order_placed(&self);
    /// Called when an order is successfully cancelled
    fn record_order_cancelled(&self);
    /// Called when a fill is received
    fn record_fill(&self, amount: f64, is_buy: bool);
    /// Called when position changes
    fn update_position(&self, position: f64);
    /// Get measured WebSocket ping latency in milliseconds.
    /// Returns 0.0 if not available or not measured yet.
    fn ws_ping_latency_ms(&self) -> f64 {
        0.0 // Default implementation for backwards compatibility
    }
}

/// Configuration for the market maker.
#[derive(Debug, Clone)]
pub struct MarketMakerConfig {
    /// Asset to market make on (e.g., "ETH", "BTC").
    /// Uses Arc<str> for cheap cloning in hot paths.
    pub asset: Arc<str>,
    /// Amount of liquidity to target on each side
    pub target_liquidity: f64,
    /// Risk aversion parameter (gamma) - controls spread and inventory skew
    /// Typical values: 0.1 (aggressive) to 2.0 (conservative)
    /// The market (kappa, sigma) determines actual spread via GLFT formula
    pub risk_aversion: f64,
    /// Max deviation before requoting (in BPS)
    pub max_bps_diff: u16,
    /// Maximum absolute position size (in contracts, derived at startup)
    pub max_position: f64,
    /// Maximum position in notional USD (source-of-truth when set).
    /// Contracts value (`max_position`) is derived from this at startup.
    /// When zero, the system was configured via contracts directly (backward-compat).
    pub max_position_usd: f64,
    /// Whether max_position was explicitly set by the user (CLI or TOML).
    /// When false, max_position is margin-derived and should NOT cap effective_max_position.
    pub max_position_user_specified: bool,
    /// Decimals for price rounding
    pub decimals: u32,
    /// Decimals for size rounding (from asset metadata)
    pub sz_decimals: u32,
    /// Enable multi-asset correlation tracking (First Principles Gap 5)
    /// When true, tracks correlations for portfolio risk management
    pub multi_asset: bool,
    /// Stochastic module integration settings.
    /// Controls HJB skew, Kalman filter, constrained optimizer, and depth AS calibration.
    pub stochastic: StochasticConfig,
    /// Enable smart ladder reconciliation with ORDER MODIFY for queue preservation.
    /// When true, uses differential updates (SKIP/MODIFY/CANCEL+PLACE) to preserve
    /// queue position when possible. This improves spread capturing competitiveness.
    /// Default: true (recommended for production)
    pub smart_reconcile: bool,

    /// Reconciliation thresholds for smart ladder updates.
    /// Controls when to SKIP (unchanged), MODIFY (small change), or CANCEL+PLACE.
    /// Only used when smart_reconcile = true.
    ///
    /// NOTE: On Hyperliquid, price modifications always reset queue position (new OID).
    /// Only SIZE-only modifications preserve queue. These tolerances primarily affect
    /// API call frequency rather than queue preservation.
    pub reconcile: ReconcileConfig,

    // === HIP-3 Support Fields ===
    /// Pre-computed runtime config (resolved at startup).
    /// All hot-path reads go through this - no Option unwraps or string comparisons.
    pub runtime: AssetRuntimeConfig,

    /// Initial isolated margin allocation in USD.
    /// Only used when runtime.is_cross == false (HIP-3 assets).
    /// Default: $1000.0
    pub initial_isolated_margin: f64,

    /// HIP-3 DEX name (e.g., "hyena", "felix").
    /// If None, trades on validator perps (default).
    /// Used for WebSocket subscriptions and asset index lookups.
    pub dex: Option<String>,

    /// Collateral/quote asset information for this DEX.
    /// Resolved at startup from meta.collateral_token and spot metadata.
    /// - Validator perps: USDC (index 0)
    /// - HIP-3 DEXs: May use USDE, USDH, or other stablecoins
    pub collateral: CollateralInfo,

    /// Statistical impulse control configuration.
    /// When enabled, reduces API churn by only updating orders when
    /// the improvement in fill probability exceeds the cost.
    pub impulse_control: ImpulseControlConfig,

    /// Spread profile for target spread ranges.
    /// Controls kappa and gamma configurations:
    /// - Default: 40-50 bps (liquid perps)
    /// - Hip3: 15-25 bps (HIP-3 DEX)
    /// - Aggressive: 10-20 bps (experimental)
    pub spread_profile: SpreadProfile,

    /// Trading fee in basis points.
    /// Used for edge calculation: edge = spread_captured - AS - fee_bps.
    /// Default: 1.5 bps (maker fee on Hyperliquid)
    pub fee_bps: f64,
}

impl MarketMakerConfig {
    /// Validate invariants that would cause runtime panics or incorrect behavior.
    ///
    /// Must be called before constructing MarketMaker. Returns a descriptive error
    /// for the first violated invariant.
    pub fn validate(&self) -> Result<(), String> {
        if self.risk_aversion <= 0.0 {
            return Err(format!(
                "risk_aversion (gamma) must be > 0.0, got {}. GLFT formula diverges at zero.",
                self.risk_aversion
            ));
        }
        if self.max_position <= 0.0 {
            return Err(format!(
                "max_position must be > 0.0, got {}",
                self.max_position
            ));
        }
        if self.target_liquidity <= 0.0 {
            return Err(format!(
                "target_liquidity must be > 0.0, got {}",
                self.target_liquidity
            ));
        }
        if self.fee_bps < 0.0 || self.fee_bps >= 100.0 {
            return Err(format!(
                "fee_bps must be in [0.0, 100.0), got {}",
                self.fee_bps
            ));
        }
        if self.max_position_usd < 0.0 {
            return Err(format!(
                "max_position_usd must be >= 0.0, got {}",
                self.max_position_usd
            ));
        }
        Ok(())
    }
}

/// Configuration passed to strategy for quote calculation.
#[derive(Debug, Clone, Copy)]
pub struct QuoteConfig {
    /// Current mid price
    pub mid_price: f64,
    /// Decimals for price rounding
    pub decimals: u32,
    /// Decimals for size rounding
    pub sz_decimals: u32,
    /// Minimum order notional value (USD)
    pub min_notional: f64,
}

/// A quote with price and size.
#[derive(Debug, Clone, Copy)]
pub struct Quote {
    /// Price of the quote
    pub price: f64,
    /// Size of the quote
    pub size: f64,
}

impl Quote {
    /// Create a new quote.
    pub fn new(price: f64, size: f64) -> Self {
        Self { price, size }
    }

    /// Calculate the notional value.
    pub fn notional(&self) -> f64 {
        self.price * self.size
    }
}

/// Type alias for optional metrics recorder.
pub type MetricsRecorder = Option<Arc<dyn MarketMakerMetricsRecorder>>;

/// Monitoring and metrics export configuration.
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Port for HTTP metrics endpoint
    pub metrics_port: u16,
    /// Whether to enable HTTP metrics endpoint
    pub enable_http_metrics: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            metrics_port: 9090,
            enable_http_metrics: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::config::ImpulseControlConfig;
    use crate::market_maker::config::runtime::AssetRuntimeConfig;
    use crate::market_maker::config::stochastic::StochasticConfig;
    use crate::market_maker::tracking::ReconcileConfig;
    use crate::meta::CollateralInfo;

    /// Create a valid MarketMakerConfig for testing.
    fn test_valid_config() -> MarketMakerConfig {
        MarketMakerConfig {
            asset: Arc::from("ETH"),
            target_liquidity: 1.0,
            risk_aversion: 0.3,
            max_bps_diff: 10,
            max_position: 5.0,
            max_position_usd: 10_000.0,
            max_position_user_specified: false,
            decimals: 2,
            sz_decimals: 3,
            multi_asset: false,
            stochastic: StochasticConfig::default(),
            smart_reconcile: true,
            reconcile: ReconcileConfig::default(),
            runtime: AssetRuntimeConfig::default(),
            initial_isolated_margin: 1000.0,
            dex: None,
            collateral: CollateralInfo::default(),
            impulse_control: ImpulseControlConfig::default(),
            spread_profile: SpreadProfile::Default,
            fee_bps: 1.5,
        }
    }

    #[test]
    fn test_config_validate_accepts_valid_config() {
        let cfg = test_valid_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_rejects_zero_gamma() {
        let mut cfg = test_valid_config();
        cfg.risk_aversion = 0.0;
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("risk_aversion"), "error should mention risk_aversion: {err}");
    }

    #[test]
    fn test_config_validate_rejects_negative_max_position() {
        let mut cfg = test_valid_config();
        cfg.max_position = -1.0;
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("max_position"), "error should mention max_position: {err}");
    }

    #[test]
    fn test_config_validate_rejects_zero_target_liquidity() {
        let mut cfg = test_valid_config();
        cfg.target_liquidity = 0.0;
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("target_liquidity"), "error should mention target_liquidity: {err}");
    }

    #[test]
    fn test_config_validate_rejects_negative_fee() {
        let mut cfg = test_valid_config();
        cfg.fee_bps = -0.5;
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("fee_bps"), "error should mention fee_bps: {err}");
    }
}
