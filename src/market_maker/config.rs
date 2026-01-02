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
    /// Maximum absolute position size
    pub max_position: f64,
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

// ============================================================================
// Stochastic Module Integration Configuration
// ============================================================================

/// Method for calculating Kelly time horizon for first-passage fill probability.
///
/// The Kelly-Stochastic optimizer needs a time horizon τ to compute fill probabilities.
/// This is semantically different from the GLFT inventory time horizon T = 1/λ.
///
/// For first-passage probability P(fill) = 2Φ(-δ/(σ√τ)) to be meaningful:
/// - τ must be long enough for price to diffuse to quote depth
/// - τ = (δ/σ)² gives P(fill at δ) ≈ 15.9%
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum KellyTimeHorizonMethod {
    /// Fixed time horizon in seconds.
    /// Use when you want predictable behavior regardless of volatility.
    Fixed,
    /// Diffusion-based: τ = (δ_char / σ)² clamped to [τ_min, τ_max].
    /// Automatically scales with volatility to maintain meaningful fill probabilities.
    /// This is the recommended default for first-principles correctness.
    #[default]
    DiffusionBased,
    /// Use arrival intensity (1/λ) - legacy behavior.
    /// WARNING: This typically produces τ ~milliseconds, causing P(fill) ≈ 0.
    ArrivalIntensity,
}

/// Configuration for stochastic module integration.
///
/// Controls feature flags for first-principles stochastic components:
/// - HJB optimal inventory skew
/// - Kalman filter for price denoising
/// - Constrained variational ladder optimization
/// - Depth-dependent adverse selection calibration
/// - Kelly-Stochastic optimal allocation
#[derive(Debug, Clone)]
pub struct StochasticConfig {
    /// Use HJB optimal_skew instead of heuristic inventory_skew_with_flow.
    /// When true: skew = γσ²qT + terminal_penalty × q × urgency + funding_bias
    /// When false: skew = inventory_ratio × γ × σ² × T × flow_modifier (existing)
    pub use_hjb_skew: bool,

    /// Use Kalman-filtered price for microprice base and spread widening.
    /// Adds uncertainty-based spread widening: half_spread += γ × σ_kalman × √T
    pub use_kalman_filter: bool,

    /// Use ConstrainedLadderOptimizer for ladder sizing.
    /// Optimizes: max Σ λ(δᵢ) × SC(δᵢ) × sᵢ s.t. margin/position constraints
    /// When false: uses geometric decay allocation
    pub use_constrained_optimizer: bool,

    /// Use Kelly-Stochastic allocation instead of proportional MV allocation.
    /// Only active when use_constrained_optimizer is true.
    /// When true: uses first-passage fill probability and Kelly criterion
    /// When false: uses proportional allocation by marginal value (λ × SC)
    pub use_kelly_stochastic: bool,

    /// Feed fills to DepthDecayAS for calibration.
    /// Records fill depth and realized AS for exponential decay model fitting.
    /// Safe to enable - passive data collection only.
    pub calibrate_depth_as: bool,

    /// Use calibrated DepthDecayAS model in ladder generation.
    /// When true and calibrated: AS(δ) = AS₀ × exp(-δ/δ_char)
    /// When false: uses config-based flat AS adjustment
    pub use_calibrated_as: bool,

    /// Kalman filter process noise Q (price variance per tick).
    /// Higher Q = trust observations more (reactive).
    /// Typical: 1e-8 (1 bp² per tick)
    pub kalman_q: f64,

    /// Kalman filter observation noise R (bid-ask bounce variance).
    /// Higher R = trust model more (smooth).
    /// Typical: 2.5e-9 (0.5 bp²)
    pub kalman_r: f64,

    /// HJB controller session duration (seconds).
    /// For 24/7 markets, use daily session (86400) or shorter sub-sessions.
    pub hjb_session_duration: f64,

    /// HJB terminal inventory penalty ($/unit²).
    /// Higher = more aggressive position reduction near session end.
    pub hjb_terminal_penalty: f64,

    /// HJB funding rate EWMA half-life (seconds).
    pub hjb_funding_half_life: f64,

    // ==================== Kelly-Stochastic Parameters ====================
    /// Informed trader probability at the touch (0.0-1.0).
    /// Estimated from fill data or set conservatively.
    /// Default: 0.15 (15% of trades at touch are informed)
    pub kelly_alpha_touch: f64,

    /// Characteristic depth for alpha decay in bps.
    /// α(δ) = α_touch × exp(-δ/alpha_decay_bps)
    /// Default: 10 bps
    pub kelly_alpha_decay_bps: f64,

    /// Kelly fraction (0.25 = quarter Kelly, recommended 0.25-0.5).
    /// Lower values are more conservative.
    /// Default: 0.25
    pub kelly_fraction: f64,

    // ==================== Kelly Time Horizon Parameters ====================
    /// Method for calculating Kelly time horizon.
    /// Default: DiffusionBased (scales τ with volatility for correct fill probabilities)
    pub kelly_time_horizon_method: KellyTimeHorizonMethod,

    /// Fixed tau value in seconds (used when method = Fixed).
    /// Default: 60.0 (1 minute)
    pub kelly_tau_fixed: f64,

    /// Minimum tau in seconds (clamp floor for DiffusionBased).
    /// Prevents τ from becoming too short in high-vol regimes.
    /// Default: 10.0 (10 seconds)
    pub kelly_tau_min: f64,

    /// Maximum tau in seconds (clamp ceiling for DiffusionBased/ArrivalIntensity).
    /// Prevents τ from becoming too long in low-vol regimes.
    /// Default: 600.0 (10 minutes)
    pub kelly_tau_max: f64,

    /// Characteristic depth in bps for diffusion-based tau calculation.
    /// τ = (kelly_char_depth_bps / σ)² gives P(fill at δ_char) ≈ 15.9%.
    /// Should be set to mid-ladder depth for balanced allocation.
    /// Default: 25.0 bps
    pub kelly_char_depth_bps: f64,
}

impl Default for StochasticConfig {
    fn default() -> Self {
        Self {
            // Feature flags - all ON by default
            use_hjb_skew: true,
            use_kalman_filter: true,
            use_constrained_optimizer: true,
            use_kelly_stochastic: true, // Use Kelly-Stochastic by default

            // Calibration flags - ON by default (passive, safe)
            calibrate_depth_as: true,
            use_calibrated_as: true,

            // Kalman parameters
            kalman_q: 1e-8,   // 1 bp² per tick
            kalman_r: 2.5e-9, // 0.5 bp² observation noise

            // HJB parameters
            hjb_session_duration: 86400.0, // 24 hour session
            hjb_terminal_penalty: 0.0005,  // 0.05% per unit²
            hjb_funding_half_life: 3600.0, // 1 hour

            // Kelly-Stochastic parameters
            // CALIBRATED from trade_history.csv (2,038 trades, Dec 2025):
            // - 42.5% win rate suggests ~25% informed flow at touch
            // - Large trades (>$2k) had -11.58 bps edge, 14.3% win rate
            // - Conservatively set alpha higher than measured AS
            kelly_alpha_touch: 0.25,     // 25% informed at touch (was 0.15)
            kelly_alpha_decay_bps: 15.0, // Slower decay for wider protection (was 10)
            kelly_fraction: 0.20,        // Conservative 1/5 Kelly (was 0.25)

            // Kelly time horizon parameters
            kelly_time_horizon_method: KellyTimeHorizonMethod::DiffusionBased,
            kelly_tau_fixed: 60.0,      // 1 minute (for Fixed method)
            kelly_tau_min: 10.0,        // 10 seconds floor
            kelly_tau_max: 600.0,       // 10 minutes ceiling
            kelly_char_depth_bps: 25.0, // Mid-ladder characteristic depth
        }
    }
}

impl StochasticConfig {
    /// Create config with all stochastic features enabled.
    pub fn all_enabled() -> Self {
        Self {
            use_hjb_skew: true,
            use_kalman_filter: true,
            use_constrained_optimizer: true,
            use_kelly_stochastic: true,
            calibrate_depth_as: true,
            use_calibrated_as: true,
            ..Default::default()
        }
    }

    /// Create config with only passive calibration (no quote changes).
    pub fn passive_only() -> Self {
        Self {
            use_hjb_skew: false,
            use_kalman_filter: false,
            use_constrained_optimizer: false,
            use_kelly_stochastic: false,
            calibrate_depth_as: true,
            use_calibrated_as: false,
            ..Default::default()
        }
    }

    /// Builder: enable HJB skew.
    pub fn with_hjb_skew(mut self) -> Self {
        self.use_hjb_skew = true;
        self
    }

    /// Builder: enable Kalman filter.
    pub fn with_kalman_filter(mut self) -> Self {
        self.use_kalman_filter = true;
        self
    }

    /// Builder: enable constrained optimizer.
    pub fn with_constrained_optimizer(mut self) -> Self {
        self.use_constrained_optimizer = true;
        self
    }

    /// Builder: enable Kelly-Stochastic allocation.
    pub fn with_kelly_stochastic(mut self) -> Self {
        self.use_kelly_stochastic = true;
        self
    }

    /// Builder: set Kelly alpha at touch.
    pub fn with_kelly_alpha_touch(mut self, alpha: f64) -> Self {
        self.kelly_alpha_touch = alpha;
        self
    }

    /// Builder: set Kelly fraction.
    pub fn with_kelly_fraction(mut self, fraction: f64) -> Self {
        self.kelly_fraction = fraction;
        self
    }

    /// Builder: set Kalman Q (process noise).
    pub fn with_kalman_q(mut self, q: f64) -> Self {
        self.kalman_q = q;
        self
    }

    /// Builder: set Kalman R (observation noise).
    pub fn with_kalman_r(mut self, r: f64) -> Self {
        self.kalman_r = r;
        self
    }

    /// Builder: set Kelly time horizon method.
    pub fn with_kelly_time_horizon_method(mut self, method: KellyTimeHorizonMethod) -> Self {
        self.kelly_time_horizon_method = method;
        self
    }

    /// Builder: set Kelly characteristic depth for diffusion-based tau.
    pub fn with_kelly_char_depth_bps(mut self, depth: f64) -> Self {
        self.kelly_char_depth_bps = depth;
        self
    }

    /// Builder: set Kelly tau min (floor for diffusion-based).
    pub fn with_kelly_tau_min(mut self, tau_min: f64) -> Self {
        self.kelly_tau_min = tau_min;
        self
    }

    /// Builder: set Kelly tau max (ceiling).
    pub fn with_kelly_tau_max(mut self, tau_max: f64) -> Self {
        self.kelly_tau_max = tau_max;
        self
    }

    /// Builder: set fixed Kelly tau (for Fixed method).
    pub fn with_kelly_tau_fixed(mut self, tau: f64) -> Self {
        self.kelly_tau_fixed = tau;
        self
    }
}
