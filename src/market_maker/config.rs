//! Configuration types for the market maker.

use std::sync::Arc;

use crate::meta::{AssetMeta, CollateralInfo};

// ============================================================================
// Pre-Computed Asset Runtime Configuration (HIP-3 Support)
// ============================================================================

/// Pre-computed asset configuration for zero-overhead hot paths.
///
/// All HIP-3 detection is done ONCE at startup. Quote cycle uses only
/// primitive fields (bool, f64) with no Option unwraps or string comparisons.
///
/// # Design Principle: ZERO HOT-PATH OVERHEAD
///
/// This struct resolves all conditional logic at construction time:
/// - `is_cross`: Pre-computed from margin mode detection
/// - `oi_cap_usd`: Pre-resolved (f64::MAX if no cap)
/// - `sz_multiplier`: Pre-computed 10^sz_decimals
///
/// # Fee Handling
///
/// Fees are NOT pre-computed. HIP-3 builder fees vary per deployer
/// (0-300% share) and are included in the `fee` field of each fill.
/// The `builderFee` field in fills contains the builder's portion.
#[derive(Debug, Clone)]
pub struct AssetRuntimeConfig {
    // === Pre-computed margin fields (NO OPTIONS) ===
    /// Whether to use cross margin (pre-computed from AssetMeta).
    /// HOT PATH: Used directly in margin calculations and leverage API calls.
    pub is_cross: bool,

    /// Open interest cap in USD (f64::MAX if no cap).
    /// HOT PATH: Pre-flight check before order placement.
    pub oi_cap_usd: f64,

    /// Pre-computed sz_decimals as f64 power for truncation.
    /// HOT PATH: Avoids powi() call in size formatting.
    pub sz_multiplier: f64,

    /// Pre-computed price decimals multiplier.
    /// For perps: 10^5 (5 significant figures).
    pub price_multiplier: f64,

    // === Cold path metadata (startup only) ===
    /// Asset name (Arc for cheap cloning).
    pub asset: Arc<str>,

    /// Maximum leverage (from API).
    pub max_leverage: f64,

    /// Whether this is a HIP-3 builder-deployed asset.
    pub is_hip3: bool,

    /// Deployer address (for logging/display only).
    pub deployer: Option<Arc<str>>,
}

impl AssetRuntimeConfig {
    /// Build from API metadata - called ONCE at startup.
    ///
    /// This resolves all HIP-3 detection and margin mode logic upfront
    /// so the hot path has zero conditional overhead.
    pub fn from_asset_meta(meta: &AssetMeta) -> Self {
        let is_hip3 = meta.is_hip3();

        Self {
            // Pre-compute for hot path
            is_cross: !is_hip3,
            oi_cap_usd: meta.oi_cap_usd.unwrap_or(f64::MAX),
            sz_multiplier: 10_f64.powi(meta.sz_decimals as i32),
            price_multiplier: 10_f64.powi(5), // 5 sig figs for perps

            // Cold path storage
            asset: Arc::from(meta.name.as_str()),
            max_leverage: meta.max_leverage as f64,
            is_hip3,
            deployer: meta.deployer.as_ref().map(|d| Arc::from(d.as_str())),
        }
    }

    /// Fast size truncation (hot path).
    ///
    /// Uses pre-computed multiplier to avoid powi() in hot path.
    #[inline(always)]
    pub fn truncate_size(&self, size: f64) -> f64 {
        (size * self.sz_multiplier).trunc() / self.sz_multiplier
    }

    /// Check OI cap (hot path) - returns max additional notional allowed.
    ///
    /// Returns 0.0 if current_oi >= cap, otherwise returns remaining capacity.
    /// For unlimited assets (oi_cap_usd == f64::MAX), returns f64::MAX.
    #[inline(always)]
    pub fn remaining_oi_capacity(&self, current_oi: f64) -> f64 {
        (self.oi_cap_usd - current_oi).max(0.0)
    }

    /// Format OI cap for display (cold path).
    pub fn oi_cap_display(&self) -> String {
        if self.oi_cap_usd == f64::MAX {
            "unlimited".to_string()
        } else {
            format!("${:.0}", self.oi_cap_usd)
        }
    }
}

impl Default for AssetRuntimeConfig {
    /// Default config for testing - represents a standard validator perp.
    fn default() -> Self {
        Self {
            is_cross: true,
            oi_cap_usd: f64::MAX,
            sz_multiplier: 100_000.0, // 5 decimals
            price_multiplier: 100_000.0,
            asset: Arc::from("BTC"),
            max_leverage: 50.0,
            is_hip3: false,
            deployer: None,
        }
    }
}

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

    // ==================== Stochastic Constraints (First Principles) ====================
    /// Enable latency-aware spread floor: δ_min = σ × √(2×τ_update) + fee
    /// When enabled, spread floor dynamically scales with volatility and quote latency.
    /// Default: true
    pub use_latency_spread_floor: bool,

    /// Expected quote update latency in milliseconds.
    /// Used in latency-aware spread floor: δ_min = σ × √(2×τ_update)
    /// Lower latency allows tighter spreads.
    /// Default: 50.0 ms
    pub quote_update_latency_ms: f64,

    /// Enable book depth threshold for tight quoting.
    /// When enabled, tight spreads require sufficient book depth to ensure fills.
    /// Default: true
    pub use_book_depth_constraint: bool,

    /// Minimum book depth (USD) within 5 bps of mid to allow tight quoting.
    /// Below this, spreads widen to protect against thin-book slippage.
    /// Default: $50,000
    pub min_book_depth_usd: f64,

    /// Book depth (USD) required for tightest spreads (3-5 bps).
    /// Interpolate between min and tight thresholds for spread adjustment.
    /// Default: $200,000
    pub tight_spread_book_depth_usd: f64,

    /// Enable conditional tight quoting logic.
    /// When true, enforces all prerequisites for tight spreads:
    /// - Calm volatility regime
    /// - Low toxicity (< toxicity_threshold)
    /// - Sufficient book depth
    /// - Low inventory utilization
    ///
    /// Default: true
    pub use_conditional_tight_quoting: bool,

    /// Maximum inventory utilization (fraction of max_position) for tight quoting.
    /// Above this, spreads widen to reduce inventory risk.
    /// Default: 0.3 (30%)
    pub tight_quoting_max_inventory: f64,

    /// Maximum toxicity (predicted alpha) for tight quoting.
    /// Above this, spreads widen to protect against informed flow.
    /// Default: 0.1 (10%)
    pub tight_quoting_max_toxicity: f64,

    /// Hours (UTC) to exclude from tight quoting due to high volatility.
    /// Common: US open (14:30 UTC), EU open (7-8 UTC), Asia session (0-2 UTC).
    /// Default: [7, 14] (EU open, US open)
    pub tight_quoting_excluded_hours: Vec<u8>,

    // ==================== Adaptive Bayesian System ====================
    /// Enable adaptive Bayesian spread calculator.
    /// When true, uses learned spread floor, blended kappa, and shrinkage gamma.
    /// This replaces static parameters with online-learned values.
    /// Default: false (conservative - enable after testing)
    pub use_adaptive_spreads: bool,

    /// Baseline volatility for adaptive gamma scaling (per-second σ).
    /// This is duplicated from RiskConfig for module isolation.
    /// Default: 0.0002 (20 bps per second)
    pub sigma_baseline: f64,

    // ==================== Entropy-Based Distribution ====================
    /// Enable entropy-based stochastic order distribution.
    /// Completely replaces the concentration fallback with a diversity-preserving system.
    ///
    /// Key features:
    /// - Minimum entropy floor ensures at least N effective levels always active
    /// - Softmax temperature controls distribution spread
    /// - Thompson sampling adds controlled randomness
    /// - Dirichlet smoothing prevents zero allocations
    ///
    /// Default: true (the new system is superior to concentration fallback)
    pub use_entropy_distribution: bool,

    /// Minimum entropy floor (bits).
    /// H_min = 1.5 → at least exp(1.5) ≈ 4.5 effective levels always active.
    /// H_min = 2.0 → at least exp(2.0) ≈ 7.4 effective levels always active.
    /// Default: 1.5
    pub entropy_min_entropy: f64,

    /// Base temperature for softmax distribution.
    /// Higher = more uniform distribution, lower = more concentrated.
    /// T = 1.0: Standard softmax
    /// T = 2.0: Twice as uniform
    /// T = 0.5: Twice as concentrated
    /// Default: 1.0
    pub entropy_base_temperature: f64,

    /// Minimum allocation floor per level (prevents zero allocations).
    /// floor = 0.02 → each level gets at least 2% of total capacity.
    /// Default: 0.02
    pub entropy_min_allocation_floor: f64,

    /// Number of Thompson samples for stochastic allocation.
    /// Higher = more stable but less explorative.
    /// n = 1: Pure Thompson (most random)
    /// n = 5: Averaged Thompson (moderate)
    /// n = 20: Quasi-deterministic
    /// Default: 5
    pub entropy_thompson_samples: usize,

    // ==================== Calibration Fill Rate Controller ====================
    /// Enable calibration-aware fill rate targeting.
    /// During warmup, reduces gamma to attract fills for parameter calibration.
    /// Automatically phases out as calibration completes.
    ///
    /// Default: true
    pub enable_calibration_fill_rate: bool,

    /// Target fill rate per hour (across all levels).
    /// The controller adjusts gamma to achieve this fill rate during warmup.
    /// 10 fills/hour ≈ 2 fills/level for a 5-level ladder.
    ///
    /// Default: 10.0
    pub target_fill_rate_per_hour: f64,

    /// Minimum gamma multiplier during fill-hungry mode.
    /// 0.3 = allow up to 70% gamma reduction (tighter quotes).
    /// Lower values = more aggressive fill seeking.
    ///
    /// Default: 0.3
    pub min_fill_hungry_gamma: f64,
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

            // Stochastic Constraints (First Principles)
            use_latency_spread_floor: true,
            quote_update_latency_ms: 50.0, // 50ms expected round-trip
            use_book_depth_constraint: true,
            min_book_depth_usd: 20_000.0, // $20k minimum (reduced from $50k)
            tight_spread_book_depth_usd: 100_000.0, // $100k for tightest spreads (reduced from $200k)
            use_conditional_tight_quoting: true,
            tight_quoting_max_inventory: 0.3, // 30% of max position
            tight_quoting_max_toxicity: 0.1,  // 10% predicted alpha
            tight_quoting_excluded_hours: vec![7, 14], // EU open, US open

            // Adaptive Bayesian System
            // ENABLED by default - uses learned parameters instead of static ones:
            // - Learned spread floor from Bayesian AS estimation
            // - Blended kappa from book depth + own fills
            // - Log-additive shrinkage gamma (prevents multiplicative explosion)
            // - Fill rate targeting with spread ceiling
            use_adaptive_spreads: true,
            sigma_baseline: 0.0002, // 20 bps per second (matches RiskConfig)

            // Entropy-Based Distribution
            // ENABLED by default - replaces concentration fallback with diversity-preserving system
            use_entropy_distribution: true,
            entropy_min_entropy: 1.5,      // At least ~4.5 effective levels
            entropy_base_temperature: 1.0, // Standard softmax
            entropy_min_allocation_floor: 0.02, // 2% minimum per level
            entropy_thompson_samples: 5,   // Moderate stochasticity

            // Calibration Fill Rate Controller
            // ENABLED by default - ensures fills for parameter calibration
            enable_calibration_fill_rate: true,
            target_fill_rate_per_hour: 10.0, // 10 fills/hour = ~2/level for 5 levels
            min_fill_hungry_gamma: 0.3,      // Max 70% gamma reduction
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::AssetMeta;

    fn make_hip3_asset_meta() -> AssetMeta {
        AssetMeta {
            name: "HIP3COIN".to_string(),
            sz_decimals: 2,
            max_leverage: 10,
            only_isolated: Some(true),
            margin_mode: Some("noCross".to_string()),
            is_delisted: None,
            deployer: Some("0xbuilder".to_string()),
            dex_id: Some(5),
            oi_cap_usd: Some(5_000_000.0),
            is_builder_deployed: Some(true),
        }
    }

    fn make_validator_asset_meta() -> AssetMeta {
        AssetMeta {
            name: "BTC".to_string(),
            sz_decimals: 5,
            max_leverage: 50,
            only_isolated: None,
            margin_mode: None,
            is_delisted: None,
            deployer: None,
            dex_id: None,
            oi_cap_usd: None,
            is_builder_deployed: None,
        }
    }

    #[test]
    fn test_runtime_config_from_hip3_asset() {
        let meta = make_hip3_asset_meta();
        let config = AssetRuntimeConfig::from_asset_meta(&meta);

        assert!(config.is_hip3);
        assert!(!config.is_cross); // HIP-3 = isolated only
        assert_eq!(config.oi_cap_usd, 5_000_000.0);
        assert_eq!(config.max_leverage, 10.0);
        assert!(config.deployer.is_some());
        assert_eq!(config.deployer.as_deref(), Some("0xbuilder"));
    }

    #[test]
    fn test_runtime_config_from_validator_perp() {
        let meta = make_validator_asset_meta();
        let config = AssetRuntimeConfig::from_asset_meta(&meta);

        assert!(!config.is_hip3);
        assert!(config.is_cross); // Validator perps support cross margin
        assert_eq!(config.oi_cap_usd, f64::MAX); // No OI cap
        assert_eq!(config.max_leverage, 50.0);
        assert!(config.deployer.is_none());
    }

    #[test]
    fn test_oi_cap_remaining_capacity() {
        let meta = make_hip3_asset_meta();
        let config = AssetRuntimeConfig::from_asset_meta(&meta);

        // With 1M position, remaining = 5M - 1M = 4M
        let remaining = config.remaining_oi_capacity(1_000_000.0);
        assert!((remaining - 4_000_000.0).abs() < 0.01);

        // At cap, remaining = 0
        let remaining = config.remaining_oi_capacity(5_000_000.0);
        assert_eq!(remaining, 0.0);

        // Over cap, remaining = 0 (clamped)
        let remaining = config.remaining_oi_capacity(6_000_000.0);
        assert_eq!(remaining, 0.0);
    }

    #[test]
    fn test_oi_cap_no_limit_validator_perp() {
        let meta = make_validator_asset_meta();
        let config = AssetRuntimeConfig::from_asset_meta(&meta);

        // With no cap (f64::MAX), remaining is effectively infinite
        let remaining = config.remaining_oi_capacity(1_000_000_000.0);
        assert!(remaining > 1e15); // Still huge
    }

    #[test]
    fn test_size_truncation() {
        let meta = make_hip3_asset_meta(); // sz_decimals = 2
        let config = AssetRuntimeConfig::from_asset_meta(&meta);

        // 1.234 should truncate to 1.23
        let truncated = config.truncate_size(1.234);
        assert!((truncated - 1.23).abs() < 1e-10);

        // 0.999 should truncate to 0.99
        let truncated = config.truncate_size(0.999);
        assert!((truncated - 0.99).abs() < 1e-10);
    }
}
