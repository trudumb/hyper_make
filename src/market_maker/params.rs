//! Focused parameter sets for market making.
//!
//! Decomposes the 62-field MarketParams into logical groups for clarity
//! and maintainability. Each struct represents a domain of market state.

use super::adverse_selection::DepthDecayAS;
use super::{SpreadRegime, VolatilityRegime};

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

// === Parameter Aggregation ===

use super::adverse_selection::AdverseSelectionEstimator;
use super::estimator::ParameterEstimator;
use super::funding::FundingRateEstimator;
use super::hawkes::HawkesOrderFlowEstimator;
use super::hjb_control::HJBInventoryController;
use super::liquidation::LiquidationCascadeDetector;
use super::margin::MarginAwareSizer;
use super::spread::SpreadProcessEstimator;
use super::config::StochasticConfig;
use super::strategy::MarketParams;

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

    // Context
    pub position: f64,
    pub max_position: f64,
    pub latest_mid: f64,
    pub risk_aversion: f64,
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
            volatility_regime: est.volatility_regime(),

            // === Order book / Liquidity ===
            kappa: est.kappa(),
            kappa_bid: est.kappa_bid(),
            kappa_ask: est.kappa_ask(),
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
            hjb_optimal_skew: sources.hjb_controller.optimal_skew(
                sources.position,
                sources.max_position,
            ),
            hjb_gamma_multiplier: sources.hjb_controller.gamma_multiplier(),
            hjb_inventory_target: sources.hjb_controller.optimal_inventory_target(),
            hjb_is_terminal_zone: sources.hjb_controller.is_terminal_zone(),

            // === Stochastic Module: Kalman Filter ===
            use_kalman_filter: sources.stochastic_config.use_kalman_filter,
            kalman_fair_price: est.kalman_fair_price(),
            kalman_uncertainty: est.kalman_uncertainty(),
            kalman_spread_widening: est.kalman_spread_widening(
                sources.risk_aversion,
                holding_time,
            ),
            kalman_warmed_up: est.kalman_warmed_up(),

            // === Stochastic Module: Constrained Optimizer ===
            use_constrained_optimizer: sources.stochastic_config.use_constrained_optimizer,
            margin_available: sources.margin_sizer.state().available_margin,
            leverage: sources.margin_sizer.state().current_leverage,
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
