//! Focused parameter sets for market making.
//!
//! Decomposes the 62-field MarketParams into logical groups for clarity
//! and maintainability. Each struct represents a domain of market state.

use crate::market_maker::adverse_selection::DepthDecayAS;
use crate::market_maker::estimator::VolatilityRegime;
use crate::market_maker::process_models::SpreadRegime;

// === Volatility Parameters ===

/// Volatility metrics from the estimator.
#[derive(Debug, Clone, Default)]
pub struct VolatilityParams {
    /// Clean volatility (σ_clean) - √BV, robust to jumps.
    /// Use for base spread calculation (continuous risk).
    pub sigma: f64,

    /// Total volatility (σ_total) - √RV, includes jumps.
    /// Captures full price variance including discontinuities.
    pub sigma_total: f64,

    /// Effective volatility (σ_effective) - blended clean/total.
    /// Reacts to jump regime; use for inventory skew.
    pub sigma_effective: f64,

    /// Volatility regime (Low/Normal/High/Extreme).
    pub regime: VolatilityRegime,

    // Stochastic volatility model parameters (Gap 2)
    /// Vol mean-reversion speed κ_vol.
    pub kappa_vol: f64,
    /// Long-run volatility θ_vol.
    pub theta_vol: f64,
    /// Vol-of-vol ξ.
    pub xi_vol: f64,
    /// Price-vol correlation ρ (leverage effect).
    pub rho_price_vol: f64,
}

impl VolatilityParams {
    /// Create with basic volatility metrics.
    pub fn new(sigma: f64, sigma_total: f64, sigma_effective: f64) -> Self {
        Self {
            sigma,
            sigma_total,
            sigma_effective,
            regime: VolatilityRegime::Normal,
            kappa_vol: 0.5,
            theta_vol: 0.0001_f64.powi(2),
            xi_vol: 0.1,
            rho_price_vol: -0.3,
        }
    }
}

// === Liquidity Parameters ===

/// Order book depth and liquidity metrics.
#[derive(Debug, Clone)]
pub struct LiquidityParams {
    /// Estimated order book depth decay (κ) - from weighted L2 book regression.
    pub kappa: f64,

    /// Directional kappa for bid side (our buy fills).
    /// When informed flow is selling, κ_bid < κ_ask → wider bid spread.
    pub kappa_bid: f64,

    /// Directional kappa for ask side (our sell fills).
    /// When informed flow is buying, κ_ask < κ_bid → wider ask spread.
    pub kappa_ask: f64,

    /// Whether fill distance distribution is heavy-tailed (CV > 1.2).
    pub is_heavy_tailed: bool,

    /// Coefficient of variation of fill distances (CV = σ/μ).
    pub kappa_cv: f64,

    /// Order arrival intensity (A) - volume ticks per second.
    pub arrival_intensity: f64,

    /// Liquidity-based gamma multiplier [1.0, 2.0].
    /// > 1.0 when near-touch liquidity is below average (thin book).
    pub liquidity_gamma_mult: f64,

    // Queue model parameters (Gap 3)
    /// Calibrated volume at touch rate (units/sec).
    pub calibrated_volume_rate: f64,
    /// Calibrated cancel rate (fraction/sec).
    pub calibrated_cancel_rate: f64,
}

impl Default for LiquidityParams {
    fn default() -> Self {
        Self {
            kappa: 100.0,
            kappa_bid: 100.0,
            kappa_ask: 100.0,
            is_heavy_tailed: false,
            kappa_cv: 1.0,
            arrival_intensity: 0.5,
            liquidity_gamma_mult: 1.0,
            calibrated_volume_rate: 1.0,
            calibrated_cancel_rate: 0.2,
        }
    }
}

// === Flow Parameters ===

/// Order flow and momentum signals.
#[derive(Debug, Clone, Default)]
pub struct FlowParams {
    /// Order flow imbalance [-1, 1].
    /// Negative = sell pressure, Positive = buy pressure.
    pub flow_imbalance: f64,

    /// L2 book imbalance [-1, 1].
    /// Positive = more bids, Negative = more asks.
    pub book_imbalance: f64,

    /// Signed momentum over 500ms window (in bps).
    /// Negative = market falling, Positive = market rising.
    pub momentum_bps: f64,

    /// Falling knife score [0, 3].
    /// > 0.5 = some downward momentum, > 1.0 = severe (protect bids!).
    pub falling_knife_score: f64,

    /// Rising knife score [0, 3].
    /// > 0.5 = some upward momentum, > 1.0 = severe (protect asks!).
    pub rising_knife_score: f64,

    // Hawkes order flow (Tier 2)
    /// Hawkes buy intensity (λ_buy) - self-exciting arrival rate.
    pub hawkes_buy_intensity: f64,
    /// Hawkes sell intensity (λ_sell) - self-exciting arrival rate.
    pub hawkes_sell_intensity: f64,
    /// Hawkes flow imbalance [-1, 1].
    pub hawkes_imbalance: f64,
    /// Hawkes activity percentile [0, 1].
    pub hawkes_activity_percentile: f64,

    // Momentum protection (Gap 10)
    /// Bid protection factor (>= 1.0 when market falling).
    pub bid_protection_factor: f64,
    /// Ask protection factor (>= 1.0 when market rising).
    pub ask_protection_factor: f64,
    /// Probability momentum continues.
    pub p_momentum_continue: f64,
}

impl FlowParams {
    /// Create with default protection factors.
    pub fn new() -> Self {
        Self {
            bid_protection_factor: 1.0,
            ask_protection_factor: 1.0,
            p_momentum_continue: 0.5,
            hawkes_activity_percentile: 0.5,
            ..Default::default()
        }
    }
}

// === Regime Parameters ===

/// Market regime detection signals.
#[derive(Debug, Clone)]
pub struct RegimeParams {
    /// Whether market is in toxic (jump) regime: RV/BV > 1.5.
    pub is_toxic_regime: bool,

    /// RV/BV jump ratio: ≈1.0 = normal diffusion, >1.5 = toxic.
    pub jump_ratio: f64,

    /// Spread regime (Tight/Normal/Wide).
    pub spread_regime: SpreadRegime,

    /// Spread percentile [0, 1].
    pub spread_percentile: f64,

    /// Fair spread from vol-adjusted model.
    pub fair_spread: f64,

    // Jump-diffusion parameters (Gap 1)
    /// Jump intensity λ (expected jumps per second).
    pub lambda_jump: f64,
    /// Mean jump size μ_j.
    pub mu_jump: f64,
    /// Jump size standard deviation σ_j.
    pub sigma_jump: f64,
}

impl Default for RegimeParams {
    fn default() -> Self {
        Self {
            is_toxic_regime: false,
            jump_ratio: 1.0,
            spread_regime: SpreadRegime::Normal,
            spread_percentile: 0.5,
            fair_spread: 0.0,
            lambda_jump: 0.0,
            mu_jump: 0.0,
            sigma_jump: 0.0001,
        }
    }
}

// === Fair Price Parameters ===

/// Microprice and fair value estimation.
#[derive(Debug, Clone, Default)]
pub struct FairPriceParams {
    /// Microprice - fair price incorporating signal predictions.
    /// Quote around this instead of raw mid.
    pub microprice: f64,

    /// β_book coefficient (return prediction per unit book imbalance).
    pub beta_book: f64,

    /// β_flow coefficient (return prediction per unit flow imbalance).
    pub beta_flow: f64,

    // Kalman filter (stochastic integration)
    /// Whether to use Kalman filter spread widening.
    pub use_kalman_filter: bool,
    /// Kalman-filtered fair price (denoised mid).
    pub kalman_fair_price: f64,
    /// Kalman filter uncertainty (posterior std dev).
    pub kalman_uncertainty: f64,
    /// Kalman-based spread widening: γ × σ_kalman × √T.
    pub kalman_spread_widening: f64,
    /// Whether Kalman filter is warmed up.
    pub kalman_warmed_up: bool,
}

// === Adverse Selection Parameters ===

/// Adverse selection measurement and adjustment.
#[derive(Debug, Clone, Default)]
pub struct AdverseSelectionParams {
    /// Spread adjustment from AS estimator (as fraction of mid price).
    /// Add to half-spread to compensate for informed flow.
    pub as_spread_adjustment: f64,

    /// Predicted alpha: P(next trade is informed) in [0, 1].
    pub predicted_alpha: f64,

    /// Is AS estimator warmed up with enough fills?
    pub as_warmed_up: bool,

    /// Depth-dependent AS model (calibrated from fills).
    /// AS(δ) = AS₀ × exp(-δ/δ_char).
    pub depth_decay_as: Option<DepthDecayAS>,
}

// === Cascade Parameters ===

/// Liquidation cascade detection and response.
#[derive(Debug, Clone)]
pub struct CascadeParams {
    /// Tail risk multiplier for gamma [1.0, 5.0].
    /// Multiply gamma by this during cascade conditions.
    pub tail_risk_multiplier: f64,

    /// Should pull all quotes due to extreme cascade?
    pub should_pull_quotes: bool,

    /// Size reduction factor [0, 1] for graceful degradation.
    /// 1.0 = full size, 0.0 = no quotes (cascade severe).
    pub cascade_size_factor: f64,
}

impl Default for CascadeParams {
    fn default() -> Self {
        Self {
            tail_risk_multiplier: 1.0,
            should_pull_quotes: false,
            cascade_size_factor: 1.0,
        }
    }
}

// === Funding Parameters ===

/// Perpetual funding rate metrics.
#[derive(Debug, Clone, Default)]
pub struct FundingParams {
    /// Current funding rate (annualized).
    pub funding_rate: f64,

    /// Predicted funding cost for holding period.
    pub predicted_funding_cost: f64,

    /// Mark-index premium.
    pub premium: f64,

    /// Premium alpha signal.
    pub premium_alpha: f64,
}

// === HJB Controller Parameters ===

/// Stochastic inventory control from HJB solution.
#[derive(Debug, Clone)]
pub struct HJBParams {
    /// Whether to use HJB optimal skew instead of heuristic.
    pub use_hjb_skew: bool,

    /// HJB optimal inventory skew (from Avellaneda-Stoikov HJB solution).
    /// Formula: γσ²qT + terminal_penalty × q × urgency + funding_bias.
    pub hjb_optimal_skew: f64,

    /// HJB gamma multiplier (for logging/diagnostics).
    pub hjb_gamma_multiplier: f64,

    /// HJB inventory target (optimal q* for current session state).
    pub hjb_inventory_target: f64,

    /// Whether HJB controller is in terminal zone (near session end).
    pub hjb_is_terminal_zone: bool,
}

impl Default for HJBParams {
    fn default() -> Self {
        Self {
            use_hjb_skew: false,
            hjb_optimal_skew: 0.0,
            hjb_gamma_multiplier: 1.0,
            hjb_inventory_target: 0.0,
            hjb_is_terminal_zone: false,
        }
    }
}

// === Margin Parameters ===

/// Margin and leverage constraints.
#[derive(Debug, Clone)]
pub struct MarginConstraintParams {
    /// Whether to use constrained ladder optimizer.
    pub use_constrained_optimizer: bool,

    /// Available margin for order placement (USD).
    pub margin_available: f64,

    /// Current leverage ratio.
    pub leverage: f64,
}

impl Default for MarginConstraintParams {
    fn default() -> Self {
        Self {
            use_constrained_optimizer: false,
            margin_available: 0.0,
            leverage: 1.0,
        }
    }
}

// === Kelly-Stochastic Parameters ===

/// Kelly-Stochastic config parameters (for MarketParams extraction).
///
/// This is a view type for extracting Kelly parameters from MarketParams.
/// The actual optimization parameters are in `quoting::ladder::optimizer::KellyStochasticParams`.
#[derive(Debug, Clone)]
pub struct KellyStochasticConfigParams {
    /// Whether to use Kelly-Stochastic allocation.
    pub use_kelly_stochastic: bool,

    /// Informed trader probability at the touch (0.0-1.0).
    pub alpha_touch: f64,

    /// Characteristic depth for alpha decay in bps.
    /// α(δ) = α_touch × exp(-δ/alpha_decay_bps).
    pub alpha_decay_bps: f64,

    /// Kelly fraction (0.25 = quarter Kelly).
    pub kelly_fraction: f64,

    /// Kelly-specific time horizon for first-passage probability (seconds).
    /// Semantically different from GLFT inventory time horizon.
    pub time_horizon: f64,
}

impl Default for KellyStochasticConfigParams {
    fn default() -> Self {
        Self {
            use_kelly_stochastic: false,
            alpha_touch: 0.15,
            alpha_decay_bps: 10.0,
            kelly_fraction: 0.25,
            time_horizon: 60.0,
        }
    }
}

/// Exchange position limit parameters (from active_asset_data API).
///
/// Used to prevent order rejections due to exchange-enforced position limits.
#[derive(Debug, Clone, Copy)]
pub struct ExchangeLimitsParams {
    /// Whether exchange limits have been fetched and are valid.
    pub valid: bool,

    /// Effective bid limit: min(local_max, exchange_available_buy).
    /// Use this to cap bid ladder sizes.
    pub effective_bid_limit: f64,

    /// Effective ask limit: min(local_max, exchange_available_sell).
    /// Use this to cap ask ladder sizes.
    pub effective_ask_limit: f64,

    /// Age of exchange limits in milliseconds.
    /// > 120,000 (2 min) = stale, consider reducing sizes.
    /// > 300,000 (5 min) = critically stale, pause quoting.
    pub age_ms: u64,
}

impl Default for ExchangeLimitsParams {
    fn default() -> Self {
        Self {
            valid: false,
            effective_bid_limit: f64::MAX,
            effective_ask_limit: f64::MAX,
            age_ms: u64::MAX,
        }
    }
}

impl ExchangeLimitsParams {
    /// Check if limits are stale (> 2 minutes old).
    pub fn is_stale(&self) -> bool {
        self.age_ms > 120_000
    }

    /// Check if limits are critically stale (> 5 minutes old).
    pub fn is_critically_stale(&self) -> bool {
        self.age_ms > 300_000
    }

    /// Get effective bid limit with staleness factor applied.
    pub fn safe_bid_limit(&self) -> f64 {
        if !self.valid {
            return f64::MAX;
        }
        match self.age_ms {
            0..=120_000 => self.effective_bid_limit,
            120_001..=300_000 => self.effective_bid_limit * 0.5,
            _ => 0.0,
        }
    }

    /// Get effective ask limit with staleness factor applied.
    pub fn safe_ask_limit(&self) -> f64 {
        if !self.valid {
            return f64::MAX;
        }
        match self.age_ms {
            0..=120_000 => self.effective_ask_limit,
            120_001..=300_000 => self.effective_ask_limit * 0.5,
            _ => 0.0,
        }
    }
}

/// Stochastic constraint parameters extracted from MarketParams.
///
/// Contains first-principles constraints for safe tight spread quoting:
/// - Tick size constraint
/// - Latency-based spread floor
/// - Book depth thresholds
/// - Conditional tight quoting logic
#[derive(Debug, Clone, Copy)]
pub struct StochasticConstraintParams {
    /// Asset tick size in basis points.
    /// Spread floor must be >= tick_size_bps.
    pub tick_size_bps: f64,

    /// Latency-aware spread floor: δ_min = σ × √(2×τ_update).
    /// In fractional terms (multiply by 10000 for bps).
    pub latency_spread_floor: f64,

    /// Near-touch book depth (USD) within 5 bps of mid.
    pub near_touch_depth_usd: f64,

    /// Whether tight quoting is currently allowed.
    pub tight_quoting_allowed: bool,

    /// Combined spread widening factor [1.0, 2.0+].
    pub stochastic_spread_multiplier: f64,
}

impl Default for StochasticConstraintParams {
    fn default() -> Self {
        Self {
            tick_size_bps: 10.0,
            latency_spread_floor: 0.0003,
            near_touch_depth_usd: 0.0,
            tight_quoting_allowed: false,
            stochastic_spread_multiplier: 1.0,
        }
    }
}

impl StochasticConstraintParams {
    /// Get effective spread floor as fraction.
    pub fn effective_floor(&self, risk_config_floor: f64) -> f64 {
        let tick_floor = self.tick_size_bps / 10_000.0;
        risk_config_floor
            .max(tick_floor)
            .max(self.latency_spread_floor)
    }

    /// Check if constraints allow tight spreads.
    pub fn can_quote_tight(&self) -> bool {
        self.tight_quoting_allowed && self.stochastic_spread_multiplier < 1.2
    }
}

// === Parameter Aggregation ===

use crate::market_maker::adaptive::AdaptiveSpreadCalculator;
use crate::market_maker::adverse_selection::AdverseSelectionEstimator;
use crate::market_maker::config::{KellyTimeHorizonMethod, StochasticConfig};
use crate::market_maker::estimator::ParameterEstimator;
use crate::market_maker::infra::MarginAwareSizer;
use crate::market_maker::process_models::{
    FundingRateEstimator, HJBInventoryController, HawkesOrderFlowEstimator,
    LiquidationCascadeDetector, SpreadProcessEstimator,
};

use super::MarketParams;

/// Sources for parameter aggregation.
///
/// Bundles references to all modules that provide market parameters.
/// This replaces the 95-line parameter gathering in update_quotes.
pub struct ParameterSources<'a> {
    // Core estimator
    pub estimator: &'a ParameterEstimator,

    // Tier 1: Adverse selection
    pub adverse_selection: &'a AdverseSelectionEstimator,
    pub depth_decay_as: &'a DepthDecayAS,
    pub liquidation_detector: &'a LiquidationCascadeDetector,

    // Tier 2: Process models
    pub hawkes: &'a HawkesOrderFlowEstimator,
    pub funding: &'a FundingRateEstimator,
    pub spread_tracker: &'a SpreadProcessEstimator,

    // Stochastic modules
    pub hjb_controller: &'a HJBInventoryController,
    pub margin_sizer: &'a MarginAwareSizer,
    pub stochastic_config: &'a StochasticConfig,

    // Adaptive Bayesian system
    pub adaptive_spreads: &'a AdaptiveSpreadCalculator,

    // Context
    pub position: f64,
    pub max_position: f64,
    pub latest_mid: f64,
    pub risk_aversion: f64,

    // Exchange position limits
    pub exchange_limits_valid: bool,
    pub exchange_effective_bid_limit: f64,
    pub exchange_effective_ask_limit: f64,
    pub exchange_limits_age_ms: u64,

    // Pending exposure from resting orders
    pub pending_bid_exposure: f64,
    pub pending_ask_exposure: f64,

    // Dynamic position limits (first principles)
    /// Dynamic max position VALUE from kill switch (USD).
    /// Derived from: min(leverage_limit, volatility_limit)
    pub dynamic_max_position_value: f64,
    /// Whether dynamic limit has been calculated from valid margin state.
    pub dynamic_limit_valid: bool,

    // Stochastic constraints (first principles)
    /// Asset tick size in basis points (from asset metadata).
    pub tick_size_bps: f64,
    /// Near-touch book depth (USD) within 5 bps of mid.
    pub near_touch_depth_usd: f64,
}

/// Calculate Kelly time horizon based on config method.
///
/// For first-passage fill probability P(fill) = 2Φ(-δ/(σ√τ)) to be meaningful,
/// τ must be long enough for price to diffuse to quote depth.
///
/// # Methods
/// - **Fixed**: Use config-specified fixed tau
/// - **DiffusionBased**: τ = (δ_char / σ)² gives P(δ_char) ≈ 15.9%
/// - **ArrivalIntensity**: τ = 1/λ (legacy, typically too short)
fn calculate_kelly_time_horizon(
    config: &StochasticConfig,
    sigma: f64,
    arrival_intensity: f64,
) -> f64 {
    match config.kelly_time_horizon_method {
        KellyTimeHorizonMethod::Fixed => config.kelly_tau_fixed,
        KellyTimeHorizonMethod::DiffusionBased => {
            // τ = (δ_char / σ)² gives P(fill at δ_char) ≈ 15.9%
            // This scales with volatility: low vol → longer tau, high vol → shorter tau
            let depth_frac = config.kelly_char_depth_bps / 10000.0;
            let safe_sigma = sigma.max(1e-9);
            let tau = (depth_frac / safe_sigma).powi(2);
            tau.clamp(config.kelly_tau_min, config.kelly_tau_max)
        }
        KellyTimeHorizonMethod::ArrivalIntensity => {
            // Legacy behavior: τ = 1/λ
            // WARNING: This typically produces τ ~milliseconds, causing P(fill) ≈ 0
            (1.0 / arrival_intensity.max(0.01)).min(config.kelly_tau_max)
        }
    }
}

/// Aggregates parameters from multiple sources into MarketParams.
///
/// This struct provides a single method to build MarketParams from all
/// the various estimators and modules, replacing scattered parameter
/// gathering code.
pub struct ParameterAggregator;

impl ParameterAggregator {
    /// Build MarketParams from all sources.
    ///
    /// This replaces the 95-line parameter gathering block in update_quotes.
    pub fn build(sources: &ParameterSources) -> MarketParams {
        let est = sources.estimator;

        // Calculate derived values
        let arrival_intensity = est.arrival_intensity();
        let holding_time = 1.0 / arrival_intensity.max(0.01);

        MarketParams {
            // === Volatility (dual-sigma architecture) ===
            sigma: est.sigma_clean(),
            sigma_total: est.sigma_total(),
            sigma_effective: est.sigma_effective(),
            sigma_leverage_adjusted: est.sigma_leverage_adjusted(),
            volatility_regime: est.volatility_regime(),

            // === Order book / Liquidity ===
            kappa: est.kappa(),
            kappa_bid: est.kappa_bid(),
            kappa_ask: est.kappa_ask(),
            is_heavy_tailed: est.is_heavy_tailed(),
            kappa_cv: est.kappa_cv(),
            arrival_intensity,
            liquidity_gamma_mult: est.liquidity_gamma_multiplier(),

            // === Regime detection ===
            is_toxic_regime: est.is_toxic_regime(),
            jump_ratio: est.jump_ratio(),

            // === Directional flow ===
            momentum_bps: est.momentum_bps(),
            flow_imbalance: est.flow_imbalance(),
            falling_knife_score: est.falling_knife_score(),
            rising_knife_score: est.rising_knife_score(),

            // === L2 book structure ===
            book_imbalance: est.book_imbalance(),

            // === Microprice: data-driven fair price ===
            microprice: est.microprice(),
            beta_book: est.beta_book(),
            beta_flow: est.beta_flow(),

            // === Tier 1: Adverse Selection ===
            as_spread_adjustment: sources.adverse_selection.spread_adjustment(),
            predicted_alpha: sources.adverse_selection.predicted_alpha(),
            as_warmed_up: sources.adverse_selection.is_warmed_up(),
            depth_decay_as: Some(sources.depth_decay_as.clone()),

            // === Tier 1: Liquidation Cascade ===
            tail_risk_multiplier: sources.liquidation_detector.tail_risk_multiplier(),
            should_pull_quotes: sources.liquidation_detector.should_pull_quotes(),
            cascade_size_factor: sources.liquidation_detector.size_reduction_factor(),

            // === Tier 2: Hawkes Order Flow ===
            hawkes_buy_intensity: sources.hawkes.lambda_buy(),
            hawkes_sell_intensity: sources.hawkes.lambda_sell(),
            hawkes_imbalance: sources.hawkes.flow_imbalance(),
            hawkes_activity_percentile: sources.hawkes.intensity_percentile(),

            // === Tier 2: Funding Rate ===
            funding_rate: sources.funding.current_rate(),
            predicted_funding_cost: sources.funding.funding_cost(
                sources.position,
                sources.latest_mid,
                3600.0, // 1 hour holding period
            ),
            premium: sources.funding.current_premium(),
            premium_alpha: sources.funding.premium_alpha(),

            // === Tier 2: Spread Process ===
            fair_spread: sources.spread_tracker.fair_spread(),
            spread_percentile: sources.spread_tracker.spread_percentile(),
            spread_regime: sources.spread_tracker.spread_regime(),
            // Market spread from observed best bid/ask
            market_spread_bps: sources.spread_tracker.current_spread_bps(),

            // === First Principles Extensions (Gaps 1-10) ===
            // Jump-Diffusion (Gap 1)
            lambda_jump: est.lambda_jump(),
            mu_jump: est.mu_jump(),
            sigma_jump: est.sigma_jump(),

            // Stochastic Volatility (Gap 2)
            kappa_vol: est.kappa_vol(),
            theta_vol: est.theta_vol_sigma().powi(2),
            xi_vol: est.xi_vol(),
            rho_price_vol: est.rho_price_vol(),

            // Queue Model (Gap 3) - defaults until CalibratedQueueModel integrated
            calibrated_volume_rate: 1.0,
            calibrated_cancel_rate: 0.2,

            // Momentum Protection (Gap 10) - defaults until MomentumModel integrated
            bid_protection_factor: 1.0,
            ask_protection_factor: 1.0,
            p_momentum_continue: 0.5,

            // === Stochastic Module: HJB Controller ===
            use_hjb_skew: sources.stochastic_config.use_hjb_skew,
            hjb_optimal_skew: sources
                .hjb_controller
                .optimal_skew(sources.position, sources.max_position),
            hjb_gamma_multiplier: sources.hjb_controller.gamma_multiplier(),
            hjb_inventory_target: sources.hjb_controller.optimal_inventory_target(),
            hjb_is_terminal_zone: sources.hjb_controller.is_terminal_zone(),

            // === Stochastic Module: Kalman Filter ===
            use_kalman_filter: sources.stochastic_config.use_kalman_filter,
            kalman_fair_price: est.kalman_fair_price(),
            kalman_uncertainty: est.kalman_uncertainty(),
            kalman_spread_widening: est.kalman_spread_widening(sources.risk_aversion, holding_time),
            kalman_warmed_up: est.kalman_warmed_up(),

            // === Stochastic Module: Constrained Optimizer ===
            use_constrained_optimizer: sources.stochastic_config.use_constrained_optimizer,
            margin_available: sources.margin_sizer.state().available_margin,
            // NOTE: Use max_leverage (allowed leverage) not current_leverage (how levered we are)
            // This determines how much margin is needed per unit of position
            leverage: sources.margin_sizer.summary().max_leverage,

            // === Stochastic Module: Kelly-Stochastic Allocation ===
            use_kelly_stochastic: sources.stochastic_config.use_kelly_stochastic,
            kelly_alpha_touch: sources.stochastic_config.kelly_alpha_touch,
            kelly_alpha_decay_bps: sources.stochastic_config.kelly_alpha_decay_bps,
            kelly_fraction: sources.stochastic_config.kelly_fraction,
            kelly_time_horizon: calculate_kelly_time_horizon(
                sources.stochastic_config,
                est.sigma_clean(),
                arrival_intensity,
            ),

            // === Exchange Position Limits ===
            // These are populated from infra.exchange_limits in the MarketMaker
            // Default to "not valid" - will be set explicitly when exchange limits are fetched
            exchange_limits_valid: sources.exchange_limits_valid,
            exchange_effective_bid_limit: sources.exchange_effective_bid_limit,
            exchange_effective_ask_limit: sources.exchange_effective_ask_limit,
            exchange_limits_age_ms: sources.exchange_limits_age_ms,

            // === Pending Exposure ===
            // Resting orders that would change position if filled
            pending_bid_exposure: sources.pending_bid_exposure,
            pending_ask_exposure: sources.pending_ask_exposure,

            // === Dynamic Position Limits (First Principles) ===
            // Convert VALUE limit to SIZE limit: max_position = max_value / price
            // Get margin-derived values for fallback
            dynamic_max_position: {
                let margin_available = sources.margin_sizer.state().available_margin;
                let leverage = sources.margin_sizer.summary().max_leverage;

                if sources.dynamic_limit_valid && sources.latest_mid > 0.0 {
                    sources.dynamic_max_position_value / sources.latest_mid
                } else if margin_available > 0.0 && leverage > 0.0 && sources.latest_mid > 0.0 {
                    // CONTROLLER-DERIVED: Use margin-based capacity, not user's arbitrary limit
                    // This ensures Kelly optimizer can allocate across full margin capacity
                    (margin_available * leverage / sources.latest_mid).max(0.0)
                } else {
                    sources.max_position // Last resort: static limit during early warmup
                }
            },
            dynamic_limit_valid: sources.dynamic_limit_valid,

            // Margin-based quoting capacity (pure solvency constraint)
            // This is the HARD limit from available margin, ignoring user's max_position
            margin_quoting_capacity: {
                let margin_available = sources.margin_sizer.state().available_margin;
                let leverage = sources.margin_sizer.summary().max_leverage;

                if margin_available > 0.0 && leverage > 0.0 && sources.latest_mid > 0.0 {
                    (margin_available * leverage / sources.latest_mid).max(0.0)
                } else {
                    0.0 // Will be computed dynamically via quoting_capacity()
                }
            },

            // === Stochastic Constraints (First Principles) ===
            // These are computed dynamically via compute_stochastic_constraints()
            tick_size_bps: sources.tick_size_bps,
            latency_spread_floor: 0.0, // Computed dynamically
            near_touch_depth_usd: sources.near_touch_depth_usd,
            tight_quoting_allowed: false, // Conservative default, computed by caller
            tight_quoting_block_reason: Some("Warmup".to_string()),
            stochastic_spread_multiplier: 1.0, // Computed dynamically

            // === Adaptive Bayesian System ===
            use_adaptive_spreads: sources.stochastic_config.use_adaptive_spreads,
            adaptive_spread_floor: sources.adaptive_spreads.spread_floor(),
            adaptive_kappa: sources.adaptive_spreads.kappa(sources.estimator.kappa()),
            adaptive_gamma: sources.adaptive_spreads.gamma(
                sources.risk_aversion,
                sources.estimator.sigma_effective()
                    / sources.stochastic_config.sigma_baseline.max(1e-9),
                sources.estimator.jump_ratio(),
                sources
                    .adaptive_spreads
                    .inventory_utilization(sources.position, sources.max_position),
                sources.hawkes.intensity_percentile(),
                sources.liquidation_detector.cascade_severity(),
            ),
            adaptive_spread_ceiling: sources.adaptive_spreads.spread_ceiling(),
            adaptive_warmed_up: sources.adaptive_spreads.is_warmed_up(),
            adaptive_can_estimate: sources.adaptive_spreads.can_provide_estimates(),
            adaptive_warmup_progress: sources.adaptive_spreads.warmup_progress(),
            adaptive_uncertainty_factor: sources.adaptive_spreads.warmup_uncertainty_factor(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volatility_params_new() {
        let params = VolatilityParams::new(0.001, 0.002, 0.0015);
        assert!((params.sigma - 0.001).abs() < f64::EPSILON);
        assert!((params.sigma_total - 0.002).abs() < f64::EPSILON);
        assert!((params.sigma_effective - 0.0015).abs() < f64::EPSILON);
    }

    #[test]
    fn test_liquidity_params_default() {
        let params = LiquidityParams::default();
        assert!((params.kappa - 100.0).abs() < f64::EPSILON);
        assert!((params.liquidity_gamma_mult - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_flow_params_new() {
        let params = FlowParams::new();
        assert!((params.bid_protection_factor - 1.0).abs() < f64::EPSILON);
        assert!((params.ask_protection_factor - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cascade_params_default() {
        let params = CascadeParams::default();
        assert!(!params.should_pull_quotes);
        assert!((params.tail_risk_multiplier - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hjb_params_default() {
        let params = HJBParams::default();
        assert!(!params.use_hjb_skew);
        assert!(!params.hjb_is_terminal_zone);
    }
}
