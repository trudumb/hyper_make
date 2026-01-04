//! Econometric parameter estimation for GLFT strategy from live market data.
//!
//! Features a robust HFT pipeline:
//! - Volume Clock: Adaptive volume-based sampling (normalizes by economic activity)
//! - VWAP Pre-Averaging: Filters bid-ask bounce noise
//! - Bipower Variation: Jump-robust volatility estimation (σ = √BV)
//! - Regime Detection: RV/BV ratio identifies toxic jump regimes
//! - Weighted Kappa: Proximity-weighted L2 book regression
//!
//! # V2 Improvements (January 2026)
//!
//! - **Corrected Conjugacy**: Kappa estimation now uses observation COUNT (not volume)
//! - **Tick-Based EWMA**: Half-lives measured in ticks, not wall-clock seconds
//! - **Credible Intervals**: 95% CI for kappa posterior uncertainty
//! - **Soft Jump Classification**: P(jump) ∈ [0,1] mixture model (planned)
//! - **Parameter Covariance**: Joint (κ, σ) tracking (planned)
//!
//! # Module Structure
//!
//! - `volatility`: Bipower variation, regime detection, stochastic vol
//! - `volume`: Volume clock, bucket accumulation, arrival estimation
//! - `momentum`: Momentum detection, trade flow tracking
//! - `kappa`: Bayesian kappa estimation (V2 - corrected), book structure
//! - `microprice`: Fair price estimation from signals
//! - `kalman`: Price filtering, noise reduction
//! - `jump`: Jump process estimation
//! - `tick_ewma`: Tick-based EWMA for volume-clock aligned estimation
//! - `parameter_estimator`: Main orchestrator

mod calibration_controller;
mod covariance;
mod hierarchical_kappa;
mod jump;
mod kalman;
mod kappa;
mod microprice;
mod mock;
mod momentum;
mod parameter_estimator;
mod soft_jump;
pub(crate) mod tick_ewma;
mod volatility;
mod volume;

// V2 re-exports (will be used when integrated)
#[allow(unused_imports)]
pub(crate) use covariance::{MultiParameterCovariance, ParameterCovariance};
#[allow(unused_imports)]
pub(crate) use hierarchical_kappa::{HierarchicalKappa, HierarchicalKappaConfig};
#[allow(unused_imports)]
pub(crate) use soft_jump::SoftJumpClassifier;

// Re-export public types
pub use calibration_controller::{CalibrationController, CalibrationControllerConfig};
pub use jump::{JumpEstimator, JumpEstimatorConfig};
pub use kalman::{KalmanPriceFilter, NoiseFilter};
pub use mock::MockEstimator;
pub use momentum::MomentumModel;
pub use parameter_estimator::ParameterEstimator;
pub use volatility::{StochasticVolParams, VolatilityRegime};

// ============================================================================
// MarketEstimator Trait - Abstraction for Testability
// ============================================================================

/// Trait for market parameter estimation.
///
/// This trait provides a read-only interface to market parameter estimators,
/// allowing for mock implementations in tests. The concrete implementation
/// is `ParameterEstimator`.
///
/// # Usage in Tests
///
/// ```ignore
/// use hyperliquid_rust_sdk::market_maker::MarketEstimator;
///
/// struct MockEstimator { sigma: f64, kappa: f64 }
///
/// impl MarketEstimator for MockEstimator {
///     fn sigma_clean(&self) -> f64 { self.sigma }
///     fn kappa(&self) -> f64 { self.kappa }
///     // ... implement other methods
/// }
/// ```
pub trait MarketEstimator: Send + Sync {
    // === Volatility ===
    /// Clean volatility (per-second, bipower variation based).
    fn sigma_clean(&self) -> f64;
    /// Total volatility (includes jumps).
    fn sigma_total(&self) -> f64;
    /// Blended effective volatility.
    fn sigma_effective(&self) -> f64;
    /// Leverage-adjusted effective volatility (wider during down moves).
    fn sigma_leverage_adjusted(&self) -> f64;
    /// Current volatility regime.
    fn volatility_regime(&self) -> VolatilityRegime;

    // === Order Book / Liquidity ===
    /// Blended kappa (fill rate decay).
    fn kappa(&self) -> f64;
    /// Bid-side kappa.
    fn kappa_bid(&self) -> f64;
    /// Ask-side kappa.
    fn kappa_ask(&self) -> f64;
    /// Whether fill distance distribution is heavy-tailed (CV > 1.2).
    fn is_heavy_tailed(&self) -> bool;
    /// Kappa coefficient of variation (CV=1 for exponential, >1 for heavy tail).
    fn kappa_cv(&self) -> f64;
    /// Order arrival intensity (ticks/second).
    fn arrival_intensity(&self) -> f64;
    /// Liquidity-based gamma multiplier.
    fn liquidity_gamma_multiplier(&self) -> f64;

    // === Regime Detection ===
    /// Whether in toxic jump regime.
    fn is_toxic_regime(&self) -> bool;
    /// RV/BV jump ratio.
    fn jump_ratio(&self) -> f64;

    // === Directional Flow ===
    /// Signed momentum in basis points.
    fn momentum_bps(&self) -> f64;
    /// Trade flow imbalance [-1, 1].
    fn flow_imbalance(&self) -> f64;
    /// Falling knife score [0, 3].
    fn falling_knife_score(&self) -> f64;
    /// Rising knife score [0, 3].
    fn rising_knife_score(&self) -> f64;

    // === Probabilistic Momentum Model ===
    /// Learned momentum continuation probability.
    fn momentum_continuation_probability(&self) -> f64;
    /// Bid protection factor from learned model.
    fn bid_protection_factor(&self) -> f64;
    /// Ask protection factor from learned model.
    fn ask_protection_factor(&self) -> f64;
    /// Overall momentum strength [0, 1].
    fn momentum_strength(&self) -> f64;
    /// Whether momentum model is calibrated.
    fn momentum_model_calibrated(&self) -> bool;

    // === L2 Book Structure ===
    /// Book imbalance [-1, 1].
    fn book_imbalance(&self) -> f64;

    // === Microprice ===
    /// Data-driven fair price.
    fn microprice(&self) -> f64;
    /// Book imbalance coefficient.
    fn beta_book(&self) -> f64;
    /// Flow imbalance coefficient.
    fn beta_flow(&self) -> f64;

    // === Jump Process (Gap 1) ===
    /// Jump intensity (λ).
    fn lambda_jump(&self) -> f64;
    /// Mean jump size.
    fn mu_jump(&self) -> f64;
    /// Jump size standard deviation.
    fn sigma_jump(&self) -> f64;

    // === Stochastic Volatility (Gap 2) ===
    /// Volatility mean-reversion speed.
    fn kappa_vol(&self) -> f64;
    /// Long-run volatility.
    fn theta_vol_sigma(&self) -> f64;
    /// Vol-of-vol.
    fn xi_vol(&self) -> f64;
    /// Price-vol correlation.
    fn rho_price_vol(&self) -> f64;

    // === Warmup Status ===
    /// Whether estimator is warmed up.
    fn is_warmed_up(&self) -> bool;
    /// Confidence in sigma estimate [0, 1].
    fn sigma_confidence(&self) -> f64;
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for econometric parameter estimation pipeline.
#[derive(Debug, Clone)]
pub struct EstimatorConfig {
    // === Volume Clock ===
    /// Initial volume bucket threshold (in asset units, e.g., 1.0 BTC)
    pub initial_bucket_volume: f64,
    /// Rolling window for adaptive bucket calculation (seconds)
    pub volume_window_secs: f64,
    /// Percentile of rolling volume for adaptive bucket (e.g., 0.01 = 1%)
    pub volume_percentile: f64,
    /// Minimum bucket volume floor
    pub min_bucket_volume: f64,
    /// Maximum bucket volume ceiling
    pub max_bucket_volume: f64,

    // === Multi-Timescale EWMA ===
    /// Fast half-life for variance EWMA (~2 seconds, reacts to crashes)
    pub fast_half_life_ticks: f64,
    /// Medium half-life for variance EWMA (~10 seconds)
    pub medium_half_life_ticks: f64,
    /// Slow half-life for variance EWMA (~60 seconds, baseline)
    pub slow_half_life_ticks: f64,
    /// Half-life for kappa EWMA (in L2 updates)
    pub kappa_half_life_updates: f64,

    // === Momentum Detection ===
    /// Window for momentum calculation (milliseconds)
    pub momentum_window_ms: u64,

    // === Trade Flow Tracking ===
    /// Window for trade flow imbalance calculation (milliseconds)
    pub trade_flow_window_ms: u64,
    /// EWMA alpha for smoothing flow imbalance
    pub trade_flow_alpha: f64,

    // === Regime Detection ===
    /// Jump detection threshold: RV/BV ratio > threshold = toxic regime
    pub jump_ratio_threshold: f64,

    // === Bayesian Kappa Estimation ===
    /// Prior mean for κ (default 500 = 20 bps avg fill distance)
    /// Higher κ = trades execute closer to mid (tighter markets)
    pub kappa_prior_mean: f64,
    /// Prior strength (effective sample size, default 10)
    /// Higher = more confident prior, slower adaptation to data
    pub kappa_prior_strength: f64,
    /// Observation window for kappa estimation (ms, default 300000 = 5 min)
    pub kappa_window_ms: u64,

    // === Warmup ===
    /// Minimum volume ticks before volatility estimates are valid
    pub min_volume_ticks: usize,
    /// Minimum trade observations before kappa is valid
    /// (Note: Originally named min_l2_updates, but actually counts trades via market_kappa)
    pub min_trade_observations: usize,

    // === Defaults ===
    /// Default sigma during warmup (per-second volatility)
    pub default_sigma: f64,
    /// Default kappa during warmup (order book decay)
    pub default_kappa: f64,
    /// Default arrival intensity during warmup (ticks per second)
    pub default_arrival_intensity: f64,
}

impl Default for EstimatorConfig {
    fn default() -> Self {
        Self {
            // Volume Clock - tuned for testnet/low-activity markets
            initial_bucket_volume: 0.01, // Start very small (0.01 BTC)
            volume_window_secs: 300.0,   // 5 minutes
            volume_percentile: 0.01,     // 1% of 5-min volume
            min_bucket_volume: 0.001,    // Floor at 0.001 BTC
            max_bucket_volume: 10.0,     // Cap at 10 BTC

            // Multi-Timescale EWMA
            fast_half_life_ticks: 5.0,     // ~2 seconds - reacts to crashes
            medium_half_life_ticks: 20.0,  // ~10 seconds
            slow_half_life_ticks: 100.0,   // ~60 seconds - baseline
            kappa_half_life_updates: 30.0, // 30 L2 updates

            // Momentum Detection
            momentum_window_ms: 500, // Track signed returns over 500ms

            // Trade Flow Tracking
            trade_flow_window_ms: 1000, // Track buy/sell imbalance over 1s
            trade_flow_alpha: 0.1,      // EWMA smoothing for flow

            // Regime Detection - true toxic flow detection
            jump_ratio_threshold: 3.0, // RV/BV > 3.0 = toxic (lower values are normal bid-ask bounce)

            // Bayesian Kappa (First Principles) - Calibrated for liquid markets
            // Prior: κ ~ Gamma(α₀, β₀) with mean = 2500, strength = 5
            // κ = 2500 implies avg fill distance = 1/2500 = 0.0004 = 4 bps
            // This is appropriate for BTC/ETH where trades execute very close to mid.
            // For illiquid assets, use asset-specific config with lower kappa.
            //
            // GLFT spread formula: δ* = (1/γ) × ln(1 + γ/κ)
            // With κ=2500, γ=0.3: δ* = 3.33 × ln(1.00012) ≈ 4bp + fees ≈ 6bp
            kappa_prior_mean: 2500.0,
            kappa_prior_strength: 5.0, // Lower strength = faster adaptation to real fills
            kappa_window_ms: 300_000,  // 5 minutes

            // Warmup - reasonable for testnet/low-activity
            min_volume_ticks: 10,
            min_trade_observations: 5,

            // Defaults
            default_sigma: 0.0001,          // 0.01% per-second
            default_kappa: 100.0,           // Moderate depth decay
            default_arrival_intensity: 0.5, // 0.5 ticks per second
        }
    }
}

impl EstimatorConfig {
    /// Create config from legacy fields (for backward compatibility with bin/market_maker.rs)
    pub fn from_legacy(
        _window_ms: u64,
        _min_trades: usize,
        default_sigma: f64,
        default_kappa: f64,
        default_arrival_intensity: f64,
        _decay_secs: u64,
        min_warmup_trades: usize,
    ) -> Self {
        Self {
            default_sigma,
            default_kappa,
            default_arrival_intensity,
            min_volume_ticks: min_warmup_trades.max(10),
            ..Default::default()
        }
    }

    /// Create config with custom toxic threshold
    pub fn with_toxic_threshold(mut self, threshold: f64) -> Self {
        self.jump_ratio_threshold = threshold;
        self
    }
}

// ============================================================================
// V2 Feature Flags
// ============================================================================

/// Feature flags for V2 estimator components.
///
/// These allow incremental rollout of the corrected Bayesian estimation
/// without breaking existing functionality. Each feature can be enabled
/// independently for testing and validation.
///
/// # Usage
///
/// ```rust,ignore
/// let config = EstimatorConfig::default();
/// let v2_flags = EstimatorV2Flags::disabled(); // Start conservative
///
/// // Enable features one at a time after validation
/// let v2_flags = EstimatorV2Flags {
///     use_hierarchical_kappa: true,
///     ..EstimatorV2Flags::disabled()
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct EstimatorV2Flags {
    /// Use hierarchical kappa model (market kappa as prior for own kappa).
    /// This is the core fix for the linear blending bug.
    pub use_hierarchical_kappa: bool,

    /// Use soft jump classification (P(jump) ∈ [0,1] instead of binary).
    /// Provides smoother toxicity score without threshold discontinuities.
    pub use_soft_jumps: bool,

    /// Track (κ, σ) covariance for spread uncertainty quantification.
    /// Enables proper error propagation through GLFT formula.
    pub use_param_covariance: bool,

    /// Expose uncertainty metrics (credible intervals, posterior variance).
    /// These are always computed; this flag controls whether they're exported.
    pub export_uncertainty_metrics: bool,
}

impl Default for EstimatorV2Flags {
    fn default() -> Self {
        Self::disabled()
    }
}

impl EstimatorV2Flags {
    /// All V2 features disabled (production default).
    pub fn disabled() -> Self {
        Self {
            use_hierarchical_kappa: false,
            use_soft_jumps: false,
            use_param_covariance: false,
            export_uncertainty_metrics: false,
        }
    }

    /// All V2 features enabled (for testing/validation).
    pub fn all_enabled() -> Self {
        Self {
            use_hierarchical_kappa: true,
            use_soft_jumps: true,
            use_param_covariance: true,
            export_uncertainty_metrics: true,
        }
    }

    /// Conservative rollout: only hierarchical kappa enabled.
    pub fn hierarchical_only() -> Self {
        Self {
            use_hierarchical_kappa: true,
            use_soft_jumps: false,
            use_param_covariance: false,
            export_uncertainty_metrics: true,
        }
    }

    /// Check if any V2 feature is enabled.
    pub fn any_enabled(&self) -> bool {
        self.use_hierarchical_kappa
            || self.use_soft_jumps
            || self.use_param_covariance
            || self.export_uncertainty_metrics
    }
}
