//! Market parameters for quoting strategies.

use crate::market_maker::adverse_selection::DepthDecayAS;
use crate::market_maker::estimator::VolatilityRegime;
use crate::market_maker::process_models::SpreadRegime;

use super::params;

/// Parameters estimated from live market data.
///
/// For the infinite-horizon GLFT model with regime detection and directional protection:
/// - Dual-sigma: sigma_clean (BV-based) for spreads, sigma_effective (blended) for skew
/// - kappa: order book depth decay constant (from weighted L2 book regression)
/// - arrival_intensity: volume ticks per second
/// - Regime detection: jump_ratio > 1.5 = toxic
/// - Directional flow: momentum_bps, flow_imbalance, falling/rising knife scores
#[derive(Debug, Clone)]
pub struct MarketParams {
    // === Volatility (all per-second, NOT annualized) ===
    /// Clean volatility (σ_clean) - √BV, robust to jumps
    /// Use for base spread calculation (continuous risk)
    pub sigma: f64,

    /// Total volatility (σ_total) - √RV, includes jumps
    /// Captures full price variance including discontinuities
    pub sigma_total: f64,

    /// Effective volatility (σ_effective) - blended clean/total
    /// Reacts to jump regime; use for inventory skew
    pub sigma_effective: f64,

    // === Order Book ===
    /// Estimated order book depth decay (κ) - from weighted L2 book regression
    pub kappa: f64,

    /// Directional kappa for bid side (our buy fills).
    /// Theory: When informed flow is selling, κ_bid < κ_ask → wider bid spread.
    /// Use for asymmetric GLFT: δ_bid = (1/γ) × ln(1 + γ/κ_bid)
    pub kappa_bid: f64,

    /// Directional kappa for ask side (our sell fills).
    /// Theory: When informed flow is buying, κ_ask < κ_bid → wider ask spread.
    /// Use for asymmetric GLFT: δ_ask = (1/γ) × ln(1 + γ/κ_ask)
    pub kappa_ask: f64,

    /// Order arrival intensity (A) - volume ticks per second
    pub arrival_intensity: f64,

    // === Regime Detection ===
    /// Whether market is in toxic (jump) regime: RV/BV > 1.5
    pub is_toxic_regime: bool,

    /// RV/BV jump ratio: ≈1.0 = normal diffusion, >1.5 = toxic
    pub jump_ratio: f64,

    // === Directional Flow ===
    /// Signed momentum over 500ms window (in bps)
    /// Negative = market falling, Positive = market rising
    pub momentum_bps: f64,

    /// Order flow imbalance [-1, 1]
    /// Negative = sell pressure, Positive = buy pressure
    pub flow_imbalance: f64,

    /// Falling knife score [0, 3]
    /// > 0.5 = some downward momentum, > 1.0 = severe (protect bids!)
    pub falling_knife_score: f64,

    /// Rising knife score [0, 3]
    /// > 0.5 = some upward momentum, > 1.0 = severe (protect asks!)
    pub rising_knife_score: f64,

    // === L2 Book Structure ===
    /// L2 book imbalance [-1, 1]
    /// Positive = more bids (buying pressure), Negative = more asks (selling pressure)
    /// Use for directional quote skew adjustment
    pub book_imbalance: f64,

    /// Liquidity-based gamma multiplier [1.0, 2.0]
    /// Returns values greater than 1.0 when near-touch liquidity is below average (thin book).
    /// Scales gamma up for wider spreads in thin conditions.
    pub liquidity_gamma_mult: f64,

    // === Microprice (Data-Driven Fair Price) ===
    /// Microprice - fair price incorporating signal predictions.
    /// Quote around this instead of raw mid.
    pub microprice: f64,

    /// β_book coefficient (return prediction per unit book imbalance).
    /// For diagnostics: expected range ~0.001-0.01 (1-10 bps per unit signal).
    pub beta_book: f64,

    /// β_flow coefficient (return prediction per unit flow imbalance).
    /// For diagnostics: expected range ~0.001-0.01 (1-10 bps per unit signal).
    pub beta_flow: f64,

    // === Tier 1: Adverse Selection ===
    /// Spread adjustment from AS estimator (as fraction of mid price).
    /// Add to half-spread to compensate for informed flow.
    pub as_spread_adjustment: f64,

    /// Predicted alpha: P(next trade is informed) in [0, 1].
    /// Use for diagnostics and future alpha-aware sizing.
    pub predicted_alpha: f64,

    /// Is AS estimator warmed up with enough fills?
    pub as_warmed_up: bool,

    /// Depth-dependent AS model (calibrated from fills).
    /// First-principles: AS(δ) = AS₀ × exp(-δ/δ_char)
    /// Used by ladder for depth-aware spread capture.
    pub depth_decay_as: Option<DepthDecayAS>,

    // === Tier 1: Liquidation Cascade ===
    /// Tail risk multiplier for gamma [1.0, 5.0].
    /// Multiply gamma by this during cascade conditions.
    pub tail_risk_multiplier: f64,

    /// Should pull all quotes due to extreme cascade?
    pub should_pull_quotes: bool,

    /// Size reduction factor [0, 1] for graceful degradation.
    /// 1.0 = full size, 0.0 = no quotes (cascade severe).
    pub cascade_size_factor: f64,

    // === Tier 2: Hawkes Order Flow ===
    /// Hawkes buy intensity (λ_buy) - self-exciting arrival rate
    pub hawkes_buy_intensity: f64,

    /// Hawkes sell intensity (λ_sell) - self-exciting arrival rate
    pub hawkes_sell_intensity: f64,

    /// Hawkes flow imbalance [-1, 1] - normalized buy/sell intensity difference
    pub hawkes_imbalance: f64,

    /// Hawkes activity percentile [0, 1] - where current intensity sits in history
    pub hawkes_activity_percentile: f64,

    // === Tier 2: Funding Rate ===
    /// Current funding rate (annualized)
    pub funding_rate: f64,

    /// Predicted funding cost for holding period
    pub predicted_funding_cost: f64,

    // === Tier 2: Spread Process ===
    /// Fair spread from vol-adjusted model
    pub fair_spread: f64,

    /// Spread percentile [0, 1] - where current spread sits in history
    pub spread_percentile: f64,

    /// Spread regime (Tight/Normal/Wide)
    pub spread_regime: SpreadRegime,

    // === Volatility Regime ===
    /// Volatility regime (Low/Normal/High/Extreme)
    pub volatility_regime: VolatilityRegime,

    // === First Principles Extensions (Gaps 1-10) ===

    // Jump-Diffusion (Gap 1)
    /// Jump intensity λ (expected jumps per second)
    pub lambda_jump: f64,
    /// Mean jump size μ_j
    pub mu_jump: f64,
    /// Jump size standard deviation σ_j
    pub sigma_jump: f64,

    // Stochastic Volatility (Gap 2)
    /// Vol mean-reversion speed κ_vol
    pub kappa_vol: f64,
    /// Long-run volatility θ_vol
    pub theta_vol: f64,
    /// Vol-of-vol ξ
    pub xi_vol: f64,
    /// Price-vol correlation ρ (leverage effect)
    pub rho_price_vol: f64,

    // Queue Model (Gap 3)
    /// Calibrated volume at touch rate (units/sec)
    pub calibrated_volume_rate: f64,
    /// Calibrated cancel rate (fraction/sec)
    pub calibrated_cancel_rate: f64,

    // Funding Enhancement (Gap 6)
    /// Mark-index premium
    pub premium: f64,
    /// Premium alpha signal
    pub premium_alpha: f64,

    // Momentum Protection (Gap 10)
    /// Bid protection factor (>= 1.0 when market falling)
    pub bid_protection_factor: f64,
    /// Ask protection factor (>= 1.0 when market rising)
    pub ask_protection_factor: f64,
    /// Probability momentum continues
    pub p_momentum_continue: f64,

    // === Stochastic Module Integration (HJB Controller) ===
    /// Whether to use HJB optimal skew instead of heuristic inventory_skew_with_flow
    pub use_hjb_skew: bool,
    /// HJB optimal inventory skew (from Avellaneda-Stoikov HJB solution)
    /// Formula: γσ²qT + terminal_penalty × q × urgency + funding_bias
    pub hjb_optimal_skew: f64,
    /// HJB gamma multiplier (for logging/diagnostics)
    pub hjb_gamma_multiplier: f64,
    /// HJB inventory target (optimal q* for current session state)
    pub hjb_inventory_target: f64,
    /// Whether HJB controller is in terminal zone (near session end)
    pub hjb_is_terminal_zone: bool,

    // === Stochastic Module Integration (Kalman Filter) ===
    /// Whether to use Kalman filter spread widening
    pub use_kalman_filter: bool,
    /// Kalman-filtered fair price (denoised mid)
    pub kalman_fair_price: f64,
    /// Kalman filter uncertainty (posterior std dev)
    pub kalman_uncertainty: f64,
    /// Kalman-based spread widening: γ × σ_kalman × √T
    pub kalman_spread_widening: f64,
    /// Whether Kalman filter is warmed up
    pub kalman_warmed_up: bool,

    // === Stochastic Module Integration (Constrained Optimizer) ===
    /// Whether to use constrained ladder optimizer
    pub use_constrained_optimizer: bool,
    /// Available margin for order placement (USD)
    pub margin_available: f64,
    /// Current leverage ratio
    pub leverage: f64,

    // === Stochastic Module Integration (Kelly-Stochastic Allocation) ===
    /// Whether to use Kelly-Stochastic allocation (requires use_constrained_optimizer=true)
    pub use_kelly_stochastic: bool,
    /// Informed trader probability at the touch (0.0-1.0)
    pub kelly_alpha_touch: f64,
    /// Characteristic depth for alpha decay in bps
    pub kelly_alpha_decay_bps: f64,
    /// Kelly fraction (0.25 = quarter Kelly)
    pub kelly_fraction: f64,
    /// Kelly-specific time horizon for first-passage probability (seconds).
    /// Semantically different from GLFT inventory time horizon T = 1/λ.
    /// Computed from KellyTimeHorizonMethod in config.
    pub kelly_time_horizon: f64,

    // === Exchange Position Limits (Pre-flight Rejection Prevention) ===
    /// Whether exchange limits have been fetched and are valid
    pub exchange_limits_valid: bool,
    /// Effective bid limit: min(local_max, exchange_available_buy)
    /// Use this to cap bid ladder sizes
    pub exchange_effective_bid_limit: f64,
    /// Effective ask limit: min(local_max, exchange_available_sell)
    /// Use this to cap ask ladder sizes
    pub exchange_effective_ask_limit: f64,
    /// Age of exchange limits in milliseconds (for staleness warnings)
    pub exchange_limits_age_ms: u64,

    // === Pending Exposure (Resting Order State) ===
    /// Total remaining size on resting buy orders (would increase long if filled)
    pub pending_bid_exposure: f64,
    /// Total remaining size on resting sell orders (would increase short if filled)
    pub pending_ask_exposure: f64,

    // === Dynamic Position Limits (First Principles) ===
    /// Dynamic max position SIZE derived from first-principles VALUE limit.
    /// Formula: dynamic_max_position = max_position_value / mid_price
    /// where max_position_value = min(leverage_limit, volatility_limit)
    /// - leverage_limit = account_value × max_leverage
    /// - volatility_limit = (equity × risk_fraction) / (num_sigmas × σ × √T)
    ///
    /// This replaces the arbitrary static max_position with an equity/volatility-derived limit.
    /// Use this for all sizing calculations instead of config.max_position.
    pub dynamic_max_position: f64,

    /// Whether the dynamic limit is valid (has been calculated from margin state).
    /// If false, strategies should fall back to config.max_position.
    pub dynamic_limit_valid: bool,
}

impl Default for MarketParams {
    fn default() -> Self {
        Self {
            sigma: 0.0001,             // 0.01% per-second volatility (clean)
            sigma_total: 0.0001,       // Same initially
            sigma_effective: 0.0001,   // Same initially
            kappa: 100.0,              // Moderate depth decay
            kappa_bid: 100.0,          // Same as kappa initially
            kappa_ask: 100.0,          // Same as kappa initially
            arrival_intensity: 0.5,    // 0.5 volume ticks per second
            is_toxic_regime: false,    // Default: not toxic
            jump_ratio: 1.0,           // Default: normal diffusion
            momentum_bps: 0.0,         // Default: no momentum
            flow_imbalance: 0.0,       // Default: balanced flow
            falling_knife_score: 0.0,  // Default: no falling knife
            rising_knife_score: 0.0,   // Default: no rising knife
            book_imbalance: 0.0,       // Default: balanced book
            liquidity_gamma_mult: 1.0, // Default: normal liquidity
            microprice: 0.0,           // Will be set from estimator
            beta_book: 0.0,            // Will be learned from data
            beta_flow: 0.0,            // Will be learned from data
            // Tier 1: Adverse Selection
            as_spread_adjustment: 0.0, // No adjustment until warmed up
            predicted_alpha: 0.0,      // Default: no informed flow detected
            as_warmed_up: false,       // Starts not warmed up
            depth_decay_as: None,      // No calibrated model initially
            // Tier 1: Liquidation Cascade
            tail_risk_multiplier: 1.0, // Default: no tail risk scaling
            should_pull_quotes: false, // Default: don't pull quotes
            cascade_size_factor: 1.0,  // Default: full size
            // Tier 2: Hawkes Order Flow
            hawkes_buy_intensity: 0.0,
            hawkes_sell_intensity: 0.0,
            hawkes_imbalance: 0.0,
            hawkes_activity_percentile: 0.5,
            // Tier 2: Funding Rate
            funding_rate: 0.0,
            predicted_funding_cost: 0.0,
            // Tier 2: Spread Process
            fair_spread: 0.0,
            spread_percentile: 0.5,
            spread_regime: SpreadRegime::Normal,
            // Volatility Regime
            volatility_regime: VolatilityRegime::Normal,
            // First Principles Extensions
            lambda_jump: 0.0,
            mu_jump: 0.0,
            sigma_jump: 0.0001,
            kappa_vol: 0.5,
            theta_vol: 0.0001_f64.powi(2), // Default variance
            xi_vol: 0.1,
            rho_price_vol: -0.3,
            calibrated_volume_rate: 1.0,
            calibrated_cancel_rate: 0.2,
            premium: 0.0,
            premium_alpha: 0.0,
            bid_protection_factor: 1.0,
            ask_protection_factor: 1.0,
            p_momentum_continue: 0.5,
            // HJB Controller (stochastic integration)
            use_hjb_skew: false,         // Default OFF for safety
            hjb_optimal_skew: 0.0,       // Will be computed from HJB controller
            hjb_gamma_multiplier: 1.0,   // No multiplier by default
            hjb_inventory_target: 0.0,   // Zero inventory target
            hjb_is_terminal_zone: false, // Not in terminal zone
            // Kalman Filter (stochastic integration)
            use_kalman_filter: false,    // Default OFF for safety
            kalman_fair_price: 0.0,      // Will be computed from Kalman filter
            kalman_uncertainty: 0.0,     // Will be computed from Kalman filter
            kalman_spread_widening: 0.0, // Will be computed from Kalman filter
            kalman_warmed_up: false,     // Not warmed up initially
            // Constrained Optimizer (stochastic integration)
            use_constrained_optimizer: false, // Default OFF for safety
            margin_available: 0.0,            // Will be fetched from margin sizer
            leverage: 1.0,                    // Default 1x leverage
            // Kelly-Stochastic Allocation (stochastic integration)
            use_kelly_stochastic: false, // Default OFF for safety
            kelly_alpha_touch: 0.15,     // 15% informed at touch
            kelly_alpha_decay_bps: 10.0, // Alpha decays with 10bp characteristic
            kelly_fraction: 0.25,        // Quarter Kelly (conservative)
            kelly_time_horizon: 60.0,    // Default 60s (will be computed from config method)
            // Exchange Position Limits
            exchange_limits_valid: false,           // Not fetched yet
            exchange_effective_bid_limit: f64::MAX, // No constraint until fetched
            exchange_effective_ask_limit: f64::MAX, // No constraint until fetched
            exchange_limits_age_ms: u64::MAX,       // Never fetched
            // Pending Exposure
            pending_bid_exposure: 0.0, // No resting orders initially
            pending_ask_exposure: 0.0, // No resting orders initially
            // Dynamic Position Limits
            dynamic_max_position: 0.0,   // Will be set from kill switch
            dynamic_limit_valid: false,  // Not valid until margin state refreshed
        }
    }
}

impl MarketParams {
    /// Get net pending exposure (positive = net long exposure from resting orders).
    pub fn net_pending_exposure(&self) -> f64 {
        self.pending_bid_exposure - self.pending_ask_exposure
    }

    /// Get effective max_position for sizing calculations.
    ///
    /// Returns dynamic_max_position if valid, otherwise falls back to provided static limit.
    /// This ensures sizing adapts to equity/volatility when margin state is available,
    /// while maintaining safe defaults during warmup.
    pub fn effective_max_position(&self, static_fallback: f64) -> f64 {
        if self.dynamic_limit_valid && self.dynamic_max_position > 0.0 {
            self.dynamic_max_position
        } else {
            static_fallback
        }
    }
}

impl MarketParams {
    /// Extract volatility parameters as a focused struct.
    pub fn volatility(&self) -> params::VolatilityParams {
        params::VolatilityParams {
            sigma: self.sigma,
            sigma_total: self.sigma_total,
            sigma_effective: self.sigma_effective,
            regime: self.volatility_regime,
            kappa_vol: self.kappa_vol,
            theta_vol: self.theta_vol,
            xi_vol: self.xi_vol,
            rho_price_vol: self.rho_price_vol,
        }
    }

    /// Extract liquidity parameters as a focused struct.
    pub fn liquidity(&self) -> params::LiquidityParams {
        params::LiquidityParams {
            kappa: self.kappa,
            kappa_bid: self.kappa_bid,
            kappa_ask: self.kappa_ask,
            arrival_intensity: self.arrival_intensity,
            liquidity_gamma_mult: self.liquidity_gamma_mult,
            calibrated_volume_rate: self.calibrated_volume_rate,
            calibrated_cancel_rate: self.calibrated_cancel_rate,
        }
    }

    /// Extract flow parameters as a focused struct.
    pub fn flow(&self) -> params::FlowParams {
        params::FlowParams {
            flow_imbalance: self.flow_imbalance,
            book_imbalance: self.book_imbalance,
            momentum_bps: self.momentum_bps,
            falling_knife_score: self.falling_knife_score,
            rising_knife_score: self.rising_knife_score,
            hawkes_buy_intensity: self.hawkes_buy_intensity,
            hawkes_sell_intensity: self.hawkes_sell_intensity,
            hawkes_imbalance: self.hawkes_imbalance,
            hawkes_activity_percentile: self.hawkes_activity_percentile,
            bid_protection_factor: self.bid_protection_factor,
            ask_protection_factor: self.ask_protection_factor,
            p_momentum_continue: self.p_momentum_continue,
        }
    }

    /// Extract regime parameters as a focused struct.
    pub fn regime(&self) -> params::RegimeParams {
        params::RegimeParams {
            is_toxic_regime: self.is_toxic_regime,
            jump_ratio: self.jump_ratio,
            spread_regime: self.spread_regime,
            spread_percentile: self.spread_percentile,
            fair_spread: self.fair_spread,
            lambda_jump: self.lambda_jump,
            mu_jump: self.mu_jump,
            sigma_jump: self.sigma_jump,
        }
    }

    /// Extract fair price parameters as a focused struct.
    pub fn fair_price(&self) -> params::FairPriceParams {
        params::FairPriceParams {
            microprice: self.microprice,
            beta_book: self.beta_book,
            beta_flow: self.beta_flow,
            use_kalman_filter: self.use_kalman_filter,
            kalman_fair_price: self.kalman_fair_price,
            kalman_uncertainty: self.kalman_uncertainty,
            kalman_spread_widening: self.kalman_spread_widening,
            kalman_warmed_up: self.kalman_warmed_up,
        }
    }

    /// Extract adverse selection parameters as a focused struct.
    pub fn adverse_selection(&self) -> params::AdverseSelectionParams {
        params::AdverseSelectionParams {
            as_spread_adjustment: self.as_spread_adjustment,
            predicted_alpha: self.predicted_alpha,
            as_warmed_up: self.as_warmed_up,
            depth_decay_as: self.depth_decay_as.clone(),
        }
    }

    /// Extract cascade parameters as a focused struct.
    pub fn cascade(&self) -> params::CascadeParams {
        params::CascadeParams {
            tail_risk_multiplier: self.tail_risk_multiplier,
            should_pull_quotes: self.should_pull_quotes,
            cascade_size_factor: self.cascade_size_factor,
        }
    }

    /// Extract funding parameters as a focused struct.
    pub fn funding(&self) -> params::FundingParams {
        params::FundingParams {
            funding_rate: self.funding_rate,
            predicted_funding_cost: self.predicted_funding_cost,
            premium: self.premium,
            premium_alpha: self.premium_alpha,
        }
    }

    /// Extract HJB controller parameters as a focused struct.
    pub fn hjb(&self) -> params::HJBParams {
        params::HJBParams {
            use_hjb_skew: self.use_hjb_skew,
            hjb_optimal_skew: self.hjb_optimal_skew,
            hjb_gamma_multiplier: self.hjb_gamma_multiplier,
            hjb_inventory_target: self.hjb_inventory_target,
            hjb_is_terminal_zone: self.hjb_is_terminal_zone,
        }
    }

    /// Extract margin constraint parameters as a focused struct.
    pub fn margin_constraints(&self) -> params::MarginConstraintParams {
        params::MarginConstraintParams {
            use_constrained_optimizer: self.use_constrained_optimizer,
            margin_available: self.margin_available,
            leverage: self.leverage,
        }
    }

    /// Extract Kelly-Stochastic parameters as a focused struct.
    pub fn kelly_stochastic(&self) -> params::KellyStochasticConfigParams {
        params::KellyStochasticConfigParams {
            use_kelly_stochastic: self.use_kelly_stochastic,
            alpha_touch: self.kelly_alpha_touch,
            alpha_decay_bps: self.kelly_alpha_decay_bps,
            kelly_fraction: self.kelly_fraction,
            time_horizon: self.kelly_time_horizon,
        }
    }

    /// Extract exchange position limit parameters as a focused struct.
    pub fn exchange_limits(&self) -> params::ExchangeLimitsParams {
        params::ExchangeLimitsParams {
            valid: self.exchange_limits_valid,
            effective_bid_limit: self.exchange_effective_bid_limit,
            effective_ask_limit: self.exchange_effective_ask_limit,
            age_ms: self.exchange_limits_age_ms,
        }
    }
}
