//! Configuration types for the market maker.

use std::sync::Arc;

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
}

/// Configuration for the market maker.
#[derive(Debug, Clone)]
pub struct MarketMakerConfig {
    /// Asset to market make on (e.g., "ETH", "BTC")
    pub asset: String,
    /// Amount of liquidity to target on each side
    pub target_liquidity: f64,
    /// Risk aversion parameter (gamma) - controls spread and inventory skew
    /// Typical values: 0.1 (aggressive) to 2.0 (conservative)
    /// The market (kappa, sigma) determines actual spread via GLFT formula
    pub risk_aversion: f64,
    /// Max deviation before requoting (in BPS)
    pub max_bps_diff: u16,
    /// Maximum absolute position size
    pub max_position: f64,
    /// Decimals for price rounding
    pub decimals: u32,
    /// Decimals for size rounding (from asset metadata)
    pub sz_decimals: u32,
    /// Enable multi-asset correlation tracking (First Principles Gap 5)
    /// When true, tracks correlations for portfolio risk management
    pub multi_asset: bool,
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

/// Configuration for first-principles dynamic risk limits.
///
/// All parameters are derived from mathematical principles - no arbitrary clamps.
/// Position limits adapt to account equity and volatility via Bayesian regularization.
/// Skew adjustments respond to order flow using exponential modifiers.
#[derive(Debug, Clone)]
pub struct DynamicRiskConfig {
    /// Fraction of capital to risk in a num_sigmas move.
    /// Derived from Kelly criterion: risk_fraction ≈ edge / variance
    /// At 0.5, a 5-sigma move leaves 50% of capital intact.
    pub risk_fraction: f64,

    /// Confidence level in standard deviations.
    /// 5.0 = 99.99997% confidence (5-sigma)
    pub num_sigmas: f64,

    /// Prior volatility when estimator has low confidence.
    /// Use historical baseline (e.g., 0.0002 = 2bp/sec for BTC)
    pub sigma_prior: f64,

    /// Flow sensitivity β for skew adjustment.
    /// exp(-β × alignment) is the modifier.
    /// β = 0.5 → ±39% adjustment at perfect alignment
    /// β = 1.0 → ±63% adjustment at perfect alignment
    pub flow_sensitivity: f64,

    /// Maximum leverage from exchange (queried from asset metadata).
    /// Caps position_value to account_value × max_leverage.
    /// This is the hard constraint - volatility can only reduce, never exceed.
    pub max_leverage: f64,
}

impl Default for DynamicRiskConfig {
    fn default() -> Self {
        Self {
            risk_fraction: 0.5,
            num_sigmas: 5.0,
            sigma_prior: 0.0002, // 2bp/sec baseline
            flow_sensitivity: 0.5,
            max_leverage: 20.0, // Conservative default, should be queried from exchange
        }
    }
}

impl DynamicRiskConfig {
    /// Create a new dynamic risk config with custom risk fraction.
    pub fn with_risk_fraction(mut self, risk_fraction: f64) -> Self {
        self.risk_fraction = risk_fraction;
        self
    }

    /// Create a new dynamic risk config with custom sigma prior.
    pub fn with_sigma_prior(mut self, sigma_prior: f64) -> Self {
        self.sigma_prior = sigma_prior;
        self
    }

    /// Create a new dynamic risk config with custom flow sensitivity.
    pub fn with_flow_sensitivity(mut self, flow_sensitivity: f64) -> Self {
        self.flow_sensitivity = flow_sensitivity;
        self
    }

    /// Create a new dynamic risk config with custom max leverage.
    pub fn with_max_leverage(mut self, max_leverage: f64) -> Self {
        self.max_leverage = max_leverage;
        self
    }
}
