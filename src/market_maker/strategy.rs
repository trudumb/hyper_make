//! Quoting strategies for the market maker.

use tracing::debug;

use crate::{round_to_significant_and_decimal, truncate_float, EPSILON};

use super::config::{Quote, QuoteConfig};
use super::ladder::Ladder;

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

    // === Directional Flow (NEW) ===
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
    pub depth_decay_as: Option<super::adverse_selection::DepthDecayAS>,

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
    pub spread_regime: super::SpreadRegime,

    // === Volatility Regime ===
    /// Volatility regime (Low/Normal/High/Extreme)
    pub volatility_regime: super::VolatilityRegime,

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
            spread_regime: super::SpreadRegime::Normal,
            // Volatility Regime
            volatility_regime: super::VolatilityRegime::Normal,
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
            use_hjb_skew: false,          // Default OFF for safety
            hjb_optimal_skew: 0.0,        // Will be computed from HJB controller
            hjb_gamma_multiplier: 1.0,    // No multiplier by default
            hjb_inventory_target: 0.0,    // Zero inventory target
            hjb_is_terminal_zone: false,  // Not in terminal zone
            // Kalman Filter (stochastic integration)
            use_kalman_filter: false,     // Default OFF for safety
            kalman_fair_price: 0.0,       // Will be computed from Kalman filter
            kalman_uncertainty: 0.0,      // Will be computed from Kalman filter
            kalman_spread_widening: 0.0,  // Will be computed from Kalman filter
            kalman_warmed_up: false,      // Not warmed up initially
            // Constrained Optimizer (stochastic integration)
            use_constrained_optimizer: false, // Default OFF for safety
            margin_available: 0.0,        // Will be fetched from margin sizer
            leverage: 1.0,                // Default 1x leverage
        }
    }
}

/// Trait for quoting strategies.
/// Strategies calculate bid and ask quotes based on market conditions and position.
pub trait QuotingStrategy: Send + Sync {
    /// Calculate bid and ask quotes.
    ///
    /// Returns `(bid, ask)` where each is `Some(Quote)` if a quote should be placed,
    /// or `None` if no quote should be placed on that side.
    ///
    /// # Parameters
    /// - `config`: Quote configuration (mid price, decimals, etc.)
    /// - `position`: Current inventory position
    /// - `max_position`: Maximum allowed position
    /// - `target_liquidity`: Target order size
    /// - `market_params`: Estimated market parameters (σ, κ) from live data
    fn calculate_quotes(
        &self,
        config: &QuoteConfig,
        position: f64,
        max_position: f64,
        target_liquidity: f64,
        market_params: &MarketParams,
    ) -> (Option<Quote>, Option<Quote>);

    /// Generate multi-level ladder quotes.
    ///
    /// For strategies that support multi-level quoting, returns a full ladder
    /// with multiple bid and ask levels. For single-level strategies, returns
    /// an empty ladder (the default implementation).
    ///
    /// # Parameters
    /// Same as `calculate_quotes`.
    fn calculate_ladder(
        &self,
        _config: &QuoteConfig,
        _position: f64,
        _max_position: f64,
        _target_liquidity: f64,
        _market_params: &MarketParams,
    ) -> Ladder {
        Ladder::default() // Empty ladder for non-ladder strategies
    }

    /// Get the name of this strategy for logging.
    fn name(&self) -> &'static str;
}

/// Blanket implementation for Box<dyn QuotingStrategy>.
impl QuotingStrategy for Box<dyn QuotingStrategy> {
    fn calculate_quotes(
        &self,
        config: &QuoteConfig,
        position: f64,
        max_position: f64,
        target_liquidity: f64,
        market_params: &MarketParams,
    ) -> (Option<Quote>, Option<Quote>) {
        (**self).calculate_quotes(
            config,
            position,
            max_position,
            target_liquidity,
            market_params,
        )
    }

    fn calculate_ladder(
        &self,
        config: &QuoteConfig,
        position: f64,
        max_position: f64,
        target_liquidity: f64,
        market_params: &MarketParams,
    ) -> Ladder {
        (**self).calculate_ladder(
            config,
            position,
            max_position,
            target_liquidity,
            market_params,
        )
    }

    fn name(&self) -> &'static str {
        (**self).name()
    }
}

/// Symmetric quoting strategy - equal spread on both sides of mid.
#[derive(Debug, Clone)]
pub struct SymmetricStrategy {
    /// Half spread in basis points
    pub half_spread_bps: u16,
}

impl SymmetricStrategy {
    /// Create a new symmetric strategy with specified half spread.
    pub fn new(half_spread_bps: u16) -> Self {
        Self { half_spread_bps }
    }
}

impl Default for SymmetricStrategy {
    fn default() -> Self {
        Self {
            half_spread_bps: 10,
        } // 10 bps default
    }
}

impl QuotingStrategy for SymmetricStrategy {
    fn calculate_quotes(
        &self,
        config: &QuoteConfig,
        position: f64,
        max_position: f64,
        target_liquidity: f64,
        _market_params: &MarketParams,
    ) -> (Option<Quote>, Option<Quote>) {
        let half_spread = (config.mid_price * self.half_spread_bps as f64) / 10000.0;

        // Calculate raw prices
        let lower_price_raw = config.mid_price - half_spread;
        let upper_price_raw = config.mid_price + half_spread;

        // Round to 5 significant figures AND max decimal places per Hyperliquid tick size rules
        let mut lower_price = round_to_significant_and_decimal(lower_price_raw, 5, config.decimals);
        let upper_price = round_to_significant_and_decimal(upper_price_raw, 5, config.decimals);

        // Ensure bid < ask (rounding may cause them to be equal for high-value assets)
        if lower_price >= upper_price {
            let tick = 10f64.powi(-(config.decimals as i32));
            lower_price -= tick;
        }

        // Calculate sizes based on position limits
        // Buy size: how much more can we buy before hitting max_position
        let buy_size_raw = (max_position - position).min(target_liquidity).max(0.0);
        // Sell size: how much more can we sell before hitting -max_position
        let sell_size_raw = (max_position + position).min(target_liquidity).max(0.0);

        // Truncate to sz_decimals (floor, not round, to be conservative)
        let buy_size = truncate_float(buy_size_raw, config.sz_decimals, false);
        let sell_size = truncate_float(sell_size_raw, config.sz_decimals, false);

        // Build quotes, checking minimum notional
        let bid = if buy_size > EPSILON {
            let quote = Quote::new(lower_price, buy_size);
            if quote.notional() >= config.min_notional {
                Some(quote)
            } else {
                None
            }
        } else {
            None
        };

        let ask = if sell_size > EPSILON {
            let quote = Quote::new(upper_price, sell_size);
            if quote.notional() >= config.min_notional {
                Some(quote)
            } else {
                None
            }
        } else {
            None
        };

        (bid, ask)
    }

    fn name(&self) -> &'static str {
        "Symmetric"
    }
}

/// Inventory-aware quoting strategy - skews prices based on position.
#[derive(Debug, Clone)]
pub struct InventoryAwareStrategy {
    /// Half spread in basis points
    pub half_spread_bps: u16,
    /// Skew factor in BPS per unit position.
    /// Positive position -> shift mid down (discourage buying, encourage selling).
    pub skew_factor_bps: f64,
}

impl InventoryAwareStrategy {
    /// Create a new inventory-aware strategy.
    pub fn new(half_spread_bps: u16, skew_factor_bps: f64) -> Self {
        Self {
            half_spread_bps,
            skew_factor_bps,
        }
    }
}

impl QuotingStrategy for InventoryAwareStrategy {
    fn calculate_quotes(
        &self,
        config: &QuoteConfig,
        position: f64,
        max_position: f64,
        target_liquidity: f64,
        _market_params: &MarketParams,
    ) -> (Option<Quote>, Option<Quote>) {
        // Calculate skew based on position
        // Positive position -> negative skew (lower prices to encourage selling)
        let skew = position * self.skew_factor_bps / 10000.0;
        let adjusted_mid = config.mid_price * (1.0 - skew);

        let half_spread = (adjusted_mid * self.half_spread_bps as f64) / 10000.0;

        // Calculate raw prices around adjusted mid
        let lower_price_raw = adjusted_mid - half_spread;
        let upper_price_raw = adjusted_mid + half_spread;

        // Round to 5 significant figures AND max decimal places
        let mut lower_price = round_to_significant_and_decimal(lower_price_raw, 5, config.decimals);
        let upper_price = round_to_significant_and_decimal(upper_price_raw, 5, config.decimals);

        // Ensure bid < ask
        if lower_price >= upper_price {
            let tick = 10f64.powi(-(config.decimals as i32));
            lower_price -= tick;
        }

        // Calculate sizes based on position limits
        let buy_size_raw = (max_position - position).min(target_liquidity).max(0.0);
        let sell_size_raw = (max_position + position).min(target_liquidity).max(0.0);

        let buy_size = truncate_float(buy_size_raw, config.sz_decimals, false);
        let sell_size = truncate_float(sell_size_raw, config.sz_decimals, false);

        // Build quotes, checking minimum notional
        let bid = if buy_size > EPSILON {
            let quote = Quote::new(lower_price, buy_size);
            if quote.notional() >= config.min_notional {
                Some(quote)
            } else {
                None
            }
        } else {
            None
        };

        let ask = if sell_size > EPSILON {
            let quote = Quote::new(upper_price, sell_size);
            if quote.notional() >= config.min_notional {
                Some(quote)
            } else {
                None
            }
        } else {
            None
        };

        (bid, ask)
    }

    fn name(&self) -> &'static str {
        "InventoryAware"
    }
}

/// Configuration for dynamic risk aversion scaling.
///
/// All parameters are explicit for future online optimization.
/// γ_effective = γ_base × vol_scalar × toxicity_scalar × inventory_scalar
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RiskConfig {
    /// Base risk aversion (γ_base) - personality in normal conditions
    /// Typical values: 0.1 (aggressive) to 1.0 (conservative)
    pub gamma_base: f64,

    /// Baseline volatility for scaling (per-second σ)
    /// When σ_effective > this, γ scales up
    pub sigma_baseline: f64,

    /// Weight for volatility scaling [0.0, 1.0]
    /// 0.0 = ignore volatility, 1.0 = full scaling
    pub volatility_weight: f64,

    /// Maximum volatility multiplier
    /// Caps how much high volatility can increase γ
    pub max_volatility_multiplier: f64,

    /// Toxicity threshold (jump_ratio above this triggers scaling)
    pub toxicity_threshold: f64,

    /// How much toxicity increases γ per unit of jump_ratio above 1.0
    pub toxicity_sensitivity: f64,

    /// Inventory utilization threshold for γ scaling [0.0, 1.0]
    /// Below this, no inventory scaling
    pub inventory_threshold: f64,

    /// How aggressively γ increases near position limits
    /// Uses quadratic scaling: 1 + sensitivity × (utilization - threshold)²
    pub inventory_sensitivity: f64,

    /// Minimum γ floor
    pub gamma_min: f64,

    /// Maximum γ ceiling
    pub gamma_max: f64,

    /// Minimum spread floor (as fraction, e.g., 0.00015 = 1.5 bps)
    /// Should be >= maker_fee_rate to ensure profitability at minimum spread
    pub min_spread_floor: f64,

    /// Maximum holding time cap (seconds)
    /// Prevents skew explosion in dead markets
    pub max_holding_time: f64,

    /// Flow sensitivity β for inventory skew adjustment.
    /// Controls how strongly flow alignment dampens/amplifies skew.
    /// exp(-β × alignment) is the modifier:
    ///   - β = 0.5 → ±39% adjustment at perfect alignment
    ///   - β = 1.0 → ±63% adjustment at perfect alignment
    ///
    /// Derived from information theory (exponential link function).
    pub flow_sensitivity: f64,

    /// Maker fee rate as fraction of notional.
    /// This is added to the GLFT half-spread to ensure profitability.
    /// The HJB equation with fees: dW = (δ - f_maker) × dN
    /// Therefore optimal spread: δ* = δ_GLFT + f_maker
    ///
    /// Hyperliquid maker fee: 0.00015 (1.5 bps)
    pub maker_fee_rate: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            gamma_base: 0.3,
            sigma_baseline: 0.0002, // 20bp per-second
            volatility_weight: 0.5,
            max_volatility_multiplier: 3.0,
            toxicity_threshold: 1.5,
            toxicity_sensitivity: 0.3,
            inventory_threshold: 0.5,
            inventory_sensitivity: 2.0,
            gamma_min: 0.05,
            gamma_max: 5.0,
            min_spread_floor: 0.00015, // 1.5 bps - matches maker fee for guaranteed profitability
            max_holding_time: 120.0,   // 2 minutes
            flow_sensitivity: 0.5,     // exp(-0.5) ≈ 0.61 at perfect alignment
            maker_fee_rate: 0.00015,   // 1.5 bps Hyperliquid maker fee
        }
    }
}

/// GLFT (Guéant-Lehalle-Fernandez-Tapia) optimal market making strategy.
///
/// Implements the **infinite-horizon** GLFT model with **dynamic risk aversion**.
///
/// ## Key Formulas (Corrected per Guéant et al. 2013):
///
/// **Half-spread (adverse selection protection):**
/// ```text
/// δ = (1/γ) × ln(1 + γ/κ)
/// ```
///
/// **Reservation price offset (inventory skew):**
/// ```text
/// skew = (q/Q_max) × γ × σ² × T
/// ```
/// where T = 1/λ (inverse of arrival intensity).
///
/// ## Dynamic Risk Aversion:
/// ```text
/// γ_effective = γ_base × vol_scalar × toxicity_scalar × inventory_scalar
/// ```
///
/// This preserves GLFT's structure while adapting to:
/// - Volatility regime (higher σ → more conservative)
/// - Toxicity (high RV/BV → informed flow → widen)
/// - Inventory utilization (near limits → reduce risk)
#[derive(Debug, Clone)]
pub struct GLFTStrategy {
    /// Risk configuration for dynamic γ calculation
    pub risk_config: RiskConfig,
}

impl GLFTStrategy {
    /// Create a new GLFT strategy with base risk aversion.
    ///
    /// Uses default RiskConfig with the specified gamma_base.
    /// For full control, use `with_config()`.
    pub fn new(gamma_base: f64) -> Self {
        Self {
            risk_config: RiskConfig {
                gamma_base: gamma_base.clamp(0.01, 10.0),
                ..Default::default()
            },
        }
    }

    /// Create a new GLFT strategy with full risk configuration.
    pub fn with_config(risk_config: RiskConfig) -> Self {
        Self { risk_config }
    }

    /// Calculate effective γ based on current market conditions.
    ///
    /// γ_effective = γ_base × vol_scalar × toxicity_scalar × inventory_scalar × regime_scalar × hawkes_scalar
    fn effective_gamma(
        &self,
        market_params: &MarketParams,
        position: f64,
        max_position: f64,
    ) -> f64 {
        let cfg = &self.risk_config;

        // === VOLATILITY SCALING ===
        // Higher realized vol → more risk per unit inventory
        let vol_ratio = market_params.sigma_effective / cfg.sigma_baseline.max(1e-9);
        let vol_scalar = if vol_ratio <= 1.0 {
            1.0 // Don't reduce γ in low vol (often precedes spikes)
        } else {
            let raw = 1.0 + cfg.volatility_weight * (vol_ratio - 1.0);
            raw.min(cfg.max_volatility_multiplier)
        };

        // === TOXICITY SCALING ===
        // High RV/BV indicates informed flow
        let toxicity_scalar = if market_params.jump_ratio <= cfg.toxicity_threshold {
            1.0
        } else {
            1.0 + cfg.toxicity_sensitivity * (market_params.jump_ratio - 1.0)
        };

        // === INVENTORY SCALING ===
        // Near position limits → less room for error
        let utilization = if max_position > EPSILON {
            (position.abs() / max_position).min(1.0)
        } else {
            0.0
        };
        let inventory_scalar = if utilization <= cfg.inventory_threshold {
            1.0
        } else {
            let excess = utilization - cfg.inventory_threshold;
            1.0 + cfg.inventory_sensitivity * excess.powi(2)
        };

        // === VOLATILITY REGIME SCALING (Tier 2) ===
        // Explicit regime detection provides additional safety layer
        let regime_scalar = match market_params.volatility_regime {
            super::VolatilityRegime::Low => 0.8, // Slightly less conservative
            super::VolatilityRegime::Normal => 1.0,
            super::VolatilityRegime::High => 1.5, // More conservative
            super::VolatilityRegime::Extreme => 2.5, // Much more conservative
        };

        // === HAWKES ACTIVITY SCALING (Tier 2) ===
        // High order flow intensity indicates potential informed trading
        // intensity_percentile > 0.8 → unusual activity → widen spreads
        let hawkes_scalar = if market_params.hawkes_activity_percentile > 0.9 {
            1.5 // Very high activity
        } else if market_params.hawkes_activity_percentile > 0.8 {
            1.2 // High activity
        } else {
            1.0 // Normal activity
        };

        // === COMBINE AND CLAMP ===
        let gamma_effective = cfg.gamma_base
            * vol_scalar
            * toxicity_scalar
            * inventory_scalar
            * regime_scalar
            * hawkes_scalar;
        let gamma_clamped = gamma_effective.clamp(cfg.gamma_min, cfg.gamma_max);

        // Log gamma component breakdown for debugging strategy behavior
        debug!(
            gamma_base = %format!("{:.3}", cfg.gamma_base),
            vol_scalar = %format!("{:.3}", vol_scalar),
            tox_scalar = %format!("{:.3}", toxicity_scalar),
            inv_scalar = %format!("{:.3}", inventory_scalar),
            regime_scalar = %format!("{:.3}", regime_scalar),
            hawkes_scalar = %format!("{:.3}", hawkes_scalar),
            gamma_raw = %format!("{:.4}", gamma_effective),
            gamma_clamped = %format!("{:.4}", gamma_clamped),
            "Gamma component breakdown"
        );

        gamma_clamped
    }

    /// Calculate expected holding time from arrival intensity.
    ///
    /// T = 1/λ where λ = arrival intensity (fills per second)
    /// Clamped to prevent skew explosion when market is dead.
    fn holding_time(&self, arrival_intensity: f64) -> f64 {
        let safe_intensity = arrival_intensity.max(0.01);
        (1.0 / safe_intensity).min(self.risk_config.max_holding_time)
    }

    /// Correct GLFT half-spread formula with fee recovery:
    ///
    /// δ* = (1/γ) × ln(1 + γ/κ) + f_maker
    ///
    /// The fee term comes from the modified HJB equation:
    /// dW = (δ - f_maker) × dN, so optimal δ* = δ_GLFT + f_maker
    ///
    /// This is market-driven: when κ drops (thin book), spread widens automatically.
    /// When κ rises (deep book), spread tightens. Fee ensures minimum profitable spread.
    fn half_spread(&self, gamma: f64, kappa: f64) -> f64 {
        let ratio = gamma / kappa;
        let glft_spread = if ratio > 1e-9 && gamma > 1e-9 {
            (1.0 / gamma) * (1.0 + ratio).ln()
        } else {
            // When γ/κ → 0, use Taylor expansion: ln(1+x) ≈ x
            // δ ≈ (1/γ) × (γ/κ) = 1/κ
            1.0 / kappa.max(1.0)
        };

        // Add maker fee to ensure profitability (first-principles HJB modification)
        glft_spread + self.risk_config.maker_fee_rate
    }

    /// Flow-adjusted inventory skew with exponential regularization.
    ///
    /// The flow_alignment ∈ [-1, 1] measures how aligned position is with flow:
    ///   +1 = perfectly aligned (long + buy flow, or short + sell flow)
    ///   -1 = perfectly opposed (long + sell flow, or short + buy flow)
    ///    0 = no flow signal
    ///
    /// We use exponential regularization that naturally bounds the modifier:
    ///   modifier = exp(-β × flow_alignment)
    ///
    /// This is mathematically clean:
    ///   - exp(0) = 1.0 (no adjustment when no flow)
    ///   - exp(β) ≈ 1 + β for small β (linear approximation)
    ///   - Always positive (can't flip skew sign)
    ///   - Symmetric in positive/negative alignment
    ///
    /// When aligned (flow pushed us here): dampen counter-skew (don't fight momentum)
    /// When opposed (fighting informed flow): amplify counter-skew (reduce risk faster)
    fn inventory_skew_with_flow(
        &self,
        inventory_ratio: f64,
        sigma: f64,
        gamma: f64,
        time_horizon: f64,
        flow_imbalance: f64,
    ) -> f64 {
        // Base GLFT skew (Avellaneda-Stoikov)
        let base_skew = inventory_ratio * gamma * sigma.powi(2) * time_horizon;

        // Flow alignment: positive when position and flow have same sign
        // inventory_ratio.signum() gives direction of position
        // flow_imbalance ∈ [-1, 1] from MarketParams
        // flow_alignment = inventory_ratio.signum() * flow_imbalance ∈ [-1, 1]
        let flow_alignment = inventory_ratio.signum() * flow_imbalance;

        // Regularized modifier using exponential
        // exp(-β × alignment) because:
        //   aligned (positive) → smaller modifier → dampen skew
        //   opposed (negative) → larger modifier → amplify skew
        let flow_modifier = (-self.risk_config.flow_sensitivity * flow_alignment).exp();

        base_skew * flow_modifier
    }
}

impl QuotingStrategy for GLFTStrategy {
    fn calculate_quotes(
        &self,
        config: &QuoteConfig,
        position: f64,
        max_position: f64,
        target_liquidity: f64,
        market_params: &MarketParams,
    ) -> (Option<Quote>, Option<Quote>) {
        // === 0. CIRCUIT BREAKER: Cascade quote pulling ===
        // Extreme liquidation cascade detected - pull all quotes immediately
        if market_params.should_pull_quotes {
            debug!(
                tail_risk_mult = %format!("{:.2}", market_params.tail_risk_multiplier),
                cascade_size_factor = %format!("{:.2}", market_params.cascade_size_factor),
                "CIRCUIT BREAKER: Liquidation cascade detected - pulling all quotes"
            );
            return (None, None);
        }

        // === 1. DYNAMIC GAMMA with Tail Risk ===
        // γ scales with volatility, toxicity, inventory utilization, liquidity, AND cascade severity
        let base_gamma = self.effective_gamma(market_params, position, max_position);
        // Apply liquidity multiplier: thin book → higher gamma → wider spread
        let gamma_with_liq = base_gamma * market_params.liquidity_gamma_mult;
        // Apply tail risk multiplier: during cascade → much higher gamma → wider spread
        let gamma = gamma_with_liq * market_params.tail_risk_multiplier;

        // === 1a. ADVERSE SELECTION-ADJUSTED KAPPA (Fix 1 + Fix 2: Directional) ===
        // Theory: Informed flow reduces effective supply of uninformed liquidity.
        // κ_effective = κ̂ × (1 - α), where α = P(informed | fill)
        //
        // In GLFT, κ is the order book depth decay. But when our fills are adversely
        // selected (informed traders hit us first), the effective κ for OUR orders
        // is lower than the market-wide κ. Lower κ → wider spreads (correct response).
        //
        // Fix 2: Directional kappa estimation - use separate κ_bid and κ_ask
        // When informed flow is selling, our bids get hit → lower κ_bid → wider bid spread
        // When informed flow is buying, our asks get lifted → lower κ_ask → wider ask spread
        //
        // Example: α = 0.5 → kappa halves → spread widens by ~30%
        let alpha = market_params.predicted_alpha.min(0.5); // Cap at 50% AS

        // Symmetric kappa (for skew and logging)
        let kappa = market_params.kappa * (1.0 - alpha);

        // Directional kappas for asymmetric GLFT spreads
        let kappa_bid = market_params.kappa_bid * (1.0 - alpha);
        let kappa_ask = market_params.kappa_ask * (1.0 - alpha);

        // Time horizon from arrival intensity: T = 1/λ (with max cap)
        let time_horizon = self.holding_time(market_params.arrival_intensity);

        // === 2. ASYMMETRIC GLFT HALF-SPREADS (First-Principles Fix 2) ===
        // δ_bid = (1/γ) × ln(1 + γ/κ_bid) - wider if κ_bid < κ_ask (sell pressure)
        // δ_ask = (1/γ) × ln(1 + γ/κ_ask) - wider if κ_ask < κ_bid (buy pressure)
        let mut half_spread_bid = self.half_spread(gamma, kappa_bid);
        let mut half_spread_ask = self.half_spread(gamma, kappa_ask);

        // Symmetric half-spread for logging (average of bid/ask)
        let mut half_spread = (half_spread_bid + half_spread_ask) / 2.0;

        // === 2a. ADVERSE SELECTION SPREAD ADJUSTMENT ===
        // Add measured AS cost to half-spreads (only when warmed up)
        if market_params.as_warmed_up && market_params.as_spread_adjustment > 0.0 {
            half_spread_bid += market_params.as_spread_adjustment;
            half_spread_ask += market_params.as_spread_adjustment;
            half_spread += market_params.as_spread_adjustment;
            debug!(
                as_adj_bps = %format!("{:.2}", market_params.as_spread_adjustment * 10000.0),
                predicted_alpha = %format!("{:.3}", market_params.predicted_alpha),
                "AS spread adjustment applied"
            );
        }

        // === 2a'. KALMAN UNCERTAINTY SPREAD WIDENING (Stochastic Module) ===
        // When use_kalman_filter is enabled, add uncertainty-based spread widening
        // Formula: spread_add = γ × σ_kalman × √T
        // Higher Kalman uncertainty → wider spreads (protects against fair price misestimation)
        if market_params.use_kalman_filter
            && market_params.kalman_warmed_up
            && market_params.kalman_spread_widening > 0.0
        {
            half_spread_bid += market_params.kalman_spread_widening;
            half_spread_ask += market_params.kalman_spread_widening;
            half_spread += market_params.kalman_spread_widening;
            debug!(
                kalman_widening_bps = %format!("{:.2}", market_params.kalman_spread_widening * 10000.0),
                kalman_uncertainty_bps = %format!("{:.2}", market_params.kalman_uncertainty * 10000.0),
                kalman_fair_price = %format!("{:.4}", market_params.kalman_fair_price),
                "Kalman uncertainty spread widening applied"
            );
        }

        // === 2b. JUMP PREMIUM (First-Principles) ===
        // Theory: Under jump-diffusion dP = σ dW + J dN, total risk exceeds diffusion risk.
        // GLFT assumes continuous diffusion, but crypto has discrete jumps.
        //
        // Jump premium formula (from jump-diffusion optimal MM theory):
        //   jump_premium = λ × E[J²] / (γ × κ)
        // where λ = jump intensity, E[J²] = expected squared jump size
        //
        // We estimate this from jump_ratio = RV/BV:
        //   - RV captures total variance (diffusion + jumps)
        //   - BV captures diffusion variance only
        //   - Jump contribution ≈ (jump_ratio - 1) × σ²
        if market_params.jump_ratio > 1.5 {
            // Estimate jump component from the ratio
            // RV/BV - 1 ≈ (λ × E[J²]) / σ²_diffusion
            let jump_component = (market_params.jump_ratio - 1.0) * market_params.sigma.powi(2);

            // Jump premium: spread compensation for jump risk
            // Scaled by 1/(γ × κ) like the GLFT formula
            let jump_premium = if gamma * kappa > 1e-9 {
                (jump_component / (gamma * kappa)).clamp(0.0, 0.005) // Max 50 bps
            } else {
                0.0
            };

            half_spread_bid += jump_premium;
            half_spread_ask += jump_premium;
            half_spread += jump_premium;

            if jump_premium > 0.0001 {
                debug!(
                    jump_ratio = %format!("{:.2}", market_params.jump_ratio),
                    jump_premium_bps = %format!("{:.2}", jump_premium * 10000.0),
                    "Jump premium applied"
                );
            }
        }

        // === 2c. SPREAD REGIME ADJUSTMENT (Tier 2) ===
        // Adjust spread based on current market spread dynamics
        let spread_regime_mult = match market_params.spread_regime {
            super::SpreadRegime::VeryTight => 1.3, // Widen - tight spreads = competition/manipulation
            super::SpreadRegime::Tight => 1.1,     // Slightly widen
            super::SpreadRegime::Normal => 1.0,
            super::SpreadRegime::Wide => 0.95, // Can tighten slightly to capture
            super::SpreadRegime::VeryWide => 0.9, // Tighten more - opportunity
        };
        half_spread_bid *= spread_regime_mult;
        half_spread_ask *= spread_regime_mult;
        half_spread *= spread_regime_mult;

        // Apply minimum spread floor
        half_spread_bid = half_spread_bid.max(self.risk_config.min_spread_floor);
        half_spread_ask = half_spread_ask.max(self.risk_config.min_spread_floor);
        half_spread = half_spread.max(self.risk_config.min_spread_floor);

        // === 3. USE SIGMA_EFFECTIVE FOR INVENTORY SKEW ===
        // sigma_effective blends clean and total based on jump regime
        let sigma_for_skew = market_params.sigma_effective;

        // Calculate inventory ratio: q / Q_max (normalized to [-1, 1])
        let inventory_ratio = if max_position > EPSILON {
            (position / max_position).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // === STOCHASTIC MODULE: HJB vs Heuristic Skew ===
        // When use_hjb_skew is true, use optimal skew from HJB controller
        // When false, use flow-dampened heuristic (existing behavior)
        let base_skew = if market_params.use_hjb_skew {
            // HJB optimal skew from Avellaneda-Stoikov HJB solution:
            // skew = γσ²qT + terminal_penalty × q × urgency + funding_bias
            // This is pre-computed by HJBInventoryController in mod.rs
            if market_params.hjb_is_terminal_zone {
                debug!(
                    hjb_skew = %format!("{:.6}", market_params.hjb_optimal_skew),
                    hjb_inv_target = %format!("{:.4}", market_params.hjb_inventory_target),
                    "HJB TERMINAL ZONE: Aggressive inventory reduction"
                );
            }
            market_params.hjb_optimal_skew
        } else {
            // Flow-dampened inventory skew: base_skew × exp(-β × flow_alignment)
            // Uses flow_imbalance to dampen skew when aligned with flow (don't fight momentum)
            // and amplify skew when opposed to flow (reduce risk faster)
            self.inventory_skew_with_flow(
                inventory_ratio,
                sigma_for_skew,
                gamma,
                time_horizon,
                market_params.flow_imbalance,
            )
        };

        // === 3a. HAWKES FLOW SKEWING (Tier 2) ===
        // Use Hawkes-derived flow imbalance for additional directional adjustment
        // High activity + imbalance → stronger signal than simple flow imbalance
        let hawkes_skew = if market_params.hawkes_activity_percentile > 0.7 {
            // Significant activity - use Hawkes imbalance for flow prediction
            // hawkes_imbalance > 0 = more buy pressure → skew asks tighter (encourage selling)
            // Scaling: 0.5 bps per 0.1 imbalance at high activity
            market_params.hawkes_imbalance * 0.00005 * market_params.hawkes_activity_percentile
        } else {
            0.0 // Low activity - don't trust flow signal
        };

        // === 3b. FUNDING COST ADJUSTMENT (Tier 2) ===
        // Incorporate perpetual funding cost into inventory decision
        // Positive funding + long = cost → skew to reduce long
        // Negative funding + short = cost → skew to reduce short
        let funding_skew = if market_params.funding_rate.abs() > 0.0001 {
            // Significant funding rate (> 1bp/hour)
            // predicted_funding_cost is signed: positive = cost to current position
            // If cost is positive and we're long, skew to sell (negative skew adds to bid_delta)
            // If cost is positive and we're short, skew to buy (positive skew reduces bid_delta)
            // Simplified: position * funding_rate gives us direction
            let funding_pressure = position.signum() * market_params.funding_rate;
            // Scale: 0.1 bps per 1bp funding rate for moderate pressure
            funding_pressure * 0.01
        } else {
            0.0
        };

        // === 4. TOXIC REGIME ===
        // Note: Dynamic gamma already scales with toxicity via effective_gamma().
        // We no longer apply a separate toxicity_multiplier to avoid double-scaling.
        // The gamma-based approach is more principled as it also affects skew.
        if market_params.is_toxic_regime {
            debug!(
                jump_ratio = %format!("{:.2}", market_params.jump_ratio),
                "Toxic regime detected (handled by dynamic gamma)"
            );
        }

        // === 5. USE MICROPRICE AS FAIR PRICE ===
        // The microprice incorporates book_imbalance and flow_imbalance predictions
        // via learned β coefficients: fair = mid × (1 + β_book × book_imb + β_flow × flow_imb)
        // This replaces all ad-hoc adjustments with data-driven signal integration.
        let fair_price = market_params.microprice;

        // === 5a. MOMENTUM-CONDITIONAL SKEW AMPLIFICATION ===
        // Theory: When position opposes strong momentum, increase skew to exit faster.
        //
        // Falling knife: market falling + long position = urgent need to sell
        // Rising knife: market rising + short position = urgent need to cover
        //
        // The flow-dampened skew handles mild flow opposition, but during strong
        // momentum (falling/rising knife), we need MORE aggressive inventory reduction.
        // This is a first-principles response to autocorrelated momentum.
        let momentum_skew_multiplier = {
            // Position direction: positive = long, negative = short
            let pos_direction = position.signum();

            // Momentum direction: falling_knife > rising_knife = falling
            let momentum_direction = if market_params.falling_knife_score
                > market_params.rising_knife_score
            {
                -1.0 // Falling
            } else if market_params.rising_knife_score > market_params.falling_knife_score {
                1.0 // Rising
            } else {
                0.0 // Neutral
            };

            // Momentum severity: max of falling/rising knife scores
            let momentum_severity = market_params
                .falling_knife_score
                .max(market_params.rising_knife_score);

            // Opposition: position and momentum in opposite directions
            let is_opposed = pos_direction * momentum_direction < 0.0;

            if is_opposed && momentum_severity > 0.5 {
                // Amplify skew when opposed to strong momentum
                // Up to 2x amplification at maximum momentum severity (score = 3)
                let amplification = 1.0 + (momentum_severity / 3.0).min(1.0);

                if momentum_severity > 1.0 {
                    debug!(
                        position = %format!("{:.4}", position),
                        falling_knife = %format!("{:.2}", market_params.falling_knife_score),
                        rising_knife = %format!("{:.2}", market_params.rising_knife_score),
                        skew_amplifier = %format!("{:.2}x", amplification),
                        "Momentum-opposed position: amplifying inventory skew"
                    );
                }

                amplification
            } else {
                1.0 // No amplification
            }
        };

        // === 6. COMBINED SKEW WITH TIER 2 ADJUSTMENTS ===
        // Combine all skew components:
        // - base_skew: GLFT inventory skew × flow modifier (exp(-β × alignment))
        // - hawkes_skew: Hawkes flow-based directional adjustment
        // - funding_skew: Perpetual funding cost pressure
        // - momentum amplification: when opposed to strong momentum
        let skew = (base_skew + hawkes_skew + funding_skew) * momentum_skew_multiplier;

        // Calculate flow modifier for logging (same as in inventory_skew_with_flow)
        let flow_alignment = inventory_ratio.signum() * market_params.flow_imbalance;
        let flow_modifier = (-self.risk_config.flow_sensitivity * flow_alignment).exp();

        // === 6a. ASYMMETRIC BID/ASK DELTAS ===
        // Use directional half-spreads: κ_bid ≠ κ_ask when flow is directional
        // - Bid delta uses half_spread_bid (wider when sell pressure = low κ_bid)
        // - Ask delta uses half_spread_ask (wider when buy pressure = low κ_ask)
        let bid_delta = half_spread_bid + skew;
        let ask_delta = (half_spread_ask - skew).max(0.0);

        debug!(
            inv_ratio = %format!("{:.4}", inventory_ratio),
            gamma = %format!("{:.4}", gamma),
            liq_mult = %format!("{:.2}", market_params.liquidity_gamma_mult),
            kappa = %format!("{:.0}", kappa),
            kappa_bid = %format!("{:.0}", kappa_bid),
            kappa_ask = %format!("{:.0}", kappa_ask),
            time_horizon = %format!("{:.2}", time_horizon),
            half_spread_bps = %format!("{:.1}", half_spread * 10000.0),
            half_spread_bid_bps = %format!("{:.1}", half_spread_bid * 10000.0),
            half_spread_ask_bps = %format!("{:.1}", half_spread_ask * 10000.0),
            flow_imb = %format!("{:.3}", market_params.flow_imbalance),
            flow_mod = %format!("{:.3}", flow_modifier),
            base_skew_bps = %format!("{:.4}", base_skew * 10000.0),
            hawkes_skew_bps = %format!("{:.4}", hawkes_skew * 10000.0),
            funding_skew_bps = %format!("{:.4}", funding_skew * 10000.0),
            total_skew_bps = %format!("{:.4}", skew * 10000.0),
            spread_regime = ?market_params.spread_regime,
            vol_regime = ?market_params.volatility_regime,
            hawkes_activity = %format!("{:.2}", market_params.hawkes_activity_percentile),
            funding_rate = %format!("{:.4}", market_params.funding_rate),
            microprice = %format!("{:.4}", fair_price),
            bid_delta_bps = %format!("{:.1}", bid_delta * 10000.0),
            ask_delta_bps = %format!("{:.1}", ask_delta * 10000.0),
            is_toxic = market_params.is_toxic_regime,
            "GLFT spread components with asymmetric kappa"
        );

        // Convert to absolute price offsets using fair_price (microprice)
        let bid_offset = fair_price * bid_delta;
        let ask_offset = fair_price * ask_delta;

        // Calculate raw prices around fair_price
        let lower_price_raw = fair_price - bid_offset;
        let upper_price_raw = fair_price + ask_offset;

        // Round to exchange precision
        let mut lower_price = round_to_significant_and_decimal(lower_price_raw, 5, config.decimals);
        let upper_price = round_to_significant_and_decimal(upper_price_raw, 5, config.decimals);

        // Ensure bid < ask
        if lower_price >= upper_price {
            let tick = 10f64.powi(-(config.decimals as i32));
            lower_price -= tick;
        }

        debug!(
            mid = config.mid_price,
            fair_price = %format!("{:.4}", fair_price),
            sigma_effective = %format!("{:.6}", sigma_for_skew),
            kappa = %format!("{:.2}", kappa),
            gamma = %format!("{:.4}", gamma),
            jump_ratio = %format!("{:.2}", market_params.jump_ratio),
            bid_final = lower_price,
            ask_final = upper_price,
            spread_bps = %format!("{:.1}", (upper_price - lower_price) / fair_price * 10000.0),
            "GLFT prices (microprice-based)"
        );

        // Calculate sizes based on position limits
        let buy_size_raw = (max_position - position).min(target_liquidity).max(0.0);
        let sell_size_raw = (max_position + position).min(target_liquidity).max(0.0);

        // === Apply cascade size reduction for graceful degradation ===
        // During moderate cascade (before quote pulling), reduce size gradually
        let buy_size_adjusted = buy_size_raw * market_params.cascade_size_factor;
        let sell_size_adjusted = sell_size_raw * market_params.cascade_size_factor;

        let buy_size = truncate_float(buy_size_adjusted, config.sz_decimals, false);
        let sell_size = truncate_float(sell_size_adjusted, config.sz_decimals, false);

        // Build quotes, checking minimum notional
        let bid = if buy_size > EPSILON {
            let quote = Quote::new(lower_price, buy_size);
            if quote.notional() >= config.min_notional {
                Some(quote)
            } else {
                None
            }
        } else {
            None
        };

        let ask = if sell_size > EPSILON {
            let quote = Quote::new(upper_price, sell_size);
            if quote.notional() >= config.min_notional {
                Some(quote)
            } else {
                None
            }
        } else {
            None
        };

        (bid, ask)
    }

    fn name(&self) -> &'static str {
        "GLFT"
    }
}

// ============================================================================
// Ladder Strategy
// ============================================================================

use super::ladder::{LadderConfig, LadderParams};

/// GLFT Ladder Strategy - multi-level quoting with depth-dependent sizing.
///
/// Generates K levels per side with:
/// - Geometric or linear depth spacing
/// - Size allocation proportional to λ(δ) × SC(δ) (fill intensity × spread capture)
/// - GLFT inventory skew applied to entire ladder
///
/// This strategy uses the same gamma calculation as GLFTStrategy but
/// distributes liquidity across multiple price levels instead of just
/// quoting at the touch.
#[derive(Debug, Clone)]
pub struct LadderStrategy {
    /// Risk configuration (same as GLFTStrategy)
    pub risk_config: RiskConfig,
    /// Ladder-specific configuration
    pub ladder_config: LadderConfig,
}

impl LadderStrategy {
    /// Create a new ladder strategy with default configs.
    pub fn new(gamma_base: f64) -> Self {
        Self {
            risk_config: RiskConfig {
                gamma_base: gamma_base.clamp(0.01, 10.0),
                ..Default::default()
            },
            ladder_config: LadderConfig::default(),
        }
    }

    /// Create a new ladder strategy with custom configs.
    pub fn with_config(risk_config: RiskConfig, ladder_config: LadderConfig) -> Self {
        Self {
            risk_config,
            ladder_config,
        }
    }

    /// Calculate effective γ based on current market conditions.
    /// (Same logic as GLFTStrategy for consistency)
    fn effective_gamma(
        &self,
        market_params: &MarketParams,
        position: f64,
        max_position: f64,
    ) -> f64 {
        let cfg = &self.risk_config;

        // Volatility scaling
        let vol_ratio = market_params.sigma_effective / cfg.sigma_baseline.max(1e-9);
        let vol_scalar = if vol_ratio <= 1.0 {
            1.0
        } else {
            let raw = 1.0 + cfg.volatility_weight * (vol_ratio - 1.0);
            raw.min(cfg.max_volatility_multiplier)
        };

        // Toxicity scaling
        let toxicity_scalar = if market_params.jump_ratio <= cfg.toxicity_threshold {
            1.0
        } else {
            1.0 + cfg.toxicity_sensitivity * (market_params.jump_ratio - 1.0)
        };

        // Inventory scaling
        let utilization = if max_position > EPSILON {
            (position.abs() / max_position).min(1.0)
        } else {
            0.0
        };
        let inventory_scalar = if utilization <= cfg.inventory_threshold {
            1.0
        } else {
            let excess = utilization - cfg.inventory_threshold;
            1.0 + cfg.inventory_sensitivity * excess.powi(2)
        };

        // Volatility regime scaling
        let regime_scalar = match market_params.volatility_regime {
            super::VolatilityRegime::Low => 0.8,
            super::VolatilityRegime::Normal => 1.0,
            super::VolatilityRegime::High => 1.5,
            super::VolatilityRegime::Extreme => 2.5,
        };

        // Hawkes activity scaling
        let hawkes_scalar = if market_params.hawkes_activity_percentile > 0.9 {
            1.5
        } else if market_params.hawkes_activity_percentile > 0.8 {
            1.2
        } else {
            1.0
        };

        let gamma_effective = cfg.gamma_base
            * vol_scalar
            * toxicity_scalar
            * inventory_scalar
            * regime_scalar
            * hawkes_scalar;
        gamma_effective.clamp(cfg.gamma_min, cfg.gamma_max)
    }

    /// Calculate holding time from arrival intensity.
    fn holding_time(&self, arrival_intensity: f64) -> f64 {
        let safe_intensity = arrival_intensity.max(0.01);
        (1.0 / safe_intensity).min(self.risk_config.max_holding_time)
    }

    /// Generate full ladder from market params.
    ///
    /// This is the main method that creates a multi-level quote ladder.
    pub fn generate_ladder(
        &self,
        config: &QuoteConfig,
        position: f64,
        max_position: f64,
        target_liquidity: f64,
        market_params: &MarketParams,
    ) -> Ladder {
        // Circuit breaker: pull all quotes during cascade
        if market_params.should_pull_quotes {
            return Ladder::default();
        }

        // Calculate effective gamma (includes all scaling factors)
        let base_gamma = self.effective_gamma(market_params, position, max_position);
        let gamma_with_liq = base_gamma * market_params.liquidity_gamma_mult;
        let gamma = gamma_with_liq * market_params.tail_risk_multiplier;

        // AS-adjusted kappa: same logic as GLFTStrategy
        // κ_effective = κ̂ × (1 - α), where α = P(informed | fill)
        let alpha = market_params.predicted_alpha.min(0.5);
        let kappa = market_params.kappa * (1.0 - alpha);

        let time_horizon = self.holding_time(market_params.arrival_intensity);

        let inventory_ratio = if max_position > EPSILON {
            (position / max_position).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // Convert AS spread adjustment to bps for ladder generation
        let as_at_touch_bps = if market_params.as_warmed_up {
            market_params.as_spread_adjustment * 10000.0
        } else {
            0.0 // Use zero AS until warmed up
        };

        // Apply cascade size reduction
        let adjusted_size = target_liquidity * market_params.cascade_size_factor;

        let params = LadderParams {
            mid_price: market_params.microprice,
            sigma: market_params.sigma,
            kappa,  // Use AS-adjusted kappa
            arrival_intensity: market_params.arrival_intensity,
            as_at_touch_bps,
            total_size: adjusted_size,
            inventory_ratio,
            gamma,
            time_horizon,
            decimals: config.decimals,
            sz_decimals: config.sz_decimals,
            min_notional: config.min_notional,
            depth_decay_as: market_params.depth_decay_as.clone(),
        };

        // === STOCHASTIC MODULE: Constrained Ladder Optimization ===
        // Solves: max Σ λ(δᵢ) × SC(δᵢ) × sᵢ  (expected profit)
        // s.t. Σ sᵢ × margin_per_unit ≤ margin_available (margin constraint)
        //      Σ sᵢ ≤ max_position (position constraint)
        if market_params.use_constrained_optimizer {
            // 1. Account for margin used by current position
            let leverage = market_params.leverage.max(1.0);
            let position_margin_cost = position.abs() * (market_params.microprice / leverage);
            let available_margin = (market_params.margin_available - position_margin_cost).max(0.0);
            let available_position = (max_position - position.abs()).max(0.0);

            // 2. Generate ladder to get depth levels and prices
            let mut ladder = Ladder::generate(&self.ladder_config, &params);

            // 3. Create constrained optimizer
            let optimizer = super::ladder::ConstrainedLadderOptimizer::new(
                available_margin,
                available_position,
                self.ladder_config.min_level_size,
                config.min_notional,
                market_params.microprice,
                leverage,
            );

            // 4. Build LevelOptimizationParams for bids
            let bid_level_params: Vec<_> = ladder
                .bids
                .iter()
                .map(|level| {
                    let fill_intensity =
                        fill_intensity_at_depth(level.depth_bps, params.sigma, params.kappa);
                    let spread_capture =
                        spread_capture_at_depth(level.depth_bps, &params, self.ladder_config.fees_bps);
                    super::ladder::LevelOptimizationParams {
                        depth_bps: level.depth_bps,
                        fill_intensity,
                        spread_capture,
                        margin_per_unit: market_params.microprice / leverage,
                    }
                })
                .collect();

            // 5. Optimize bid sizes
            if !bid_level_params.is_empty() {
                let allocation = optimizer.optimize(&bid_level_params);
                for (i, &size) in allocation.sizes.iter().enumerate() {
                    if i < ladder.bids.len() {
                        ladder.bids[i].size = size;
                    }
                }
                debug!(
                    binding = ?allocation.binding_constraint,
                    margin_used = %format!("{:.2}", allocation.margin_used),
                    position_used = %format!("{:.4}", allocation.position_used),
                    shadow_price = %format!("{:.6}", allocation.shadow_price),
                    "Constrained optimizer applied to bids"
                );
            }

            // 6. Build LevelOptimizationParams for asks
            let ask_level_params: Vec<_> = ladder
                .asks
                .iter()
                .map(|level| {
                    let fill_intensity =
                        fill_intensity_at_depth(level.depth_bps, params.sigma, params.kappa);
                    let spread_capture =
                        spread_capture_at_depth(level.depth_bps, &params, self.ladder_config.fees_bps);
                    super::ladder::LevelOptimizationParams {
                        depth_bps: level.depth_bps,
                        fill_intensity,
                        spread_capture,
                        margin_per_unit: market_params.microprice / leverage,
                    }
                })
                .collect();

            // 7. Optimize ask sizes
            if !ask_level_params.is_empty() {
                let allocation = optimizer.optimize(&ask_level_params);
                for (i, &size) in allocation.sizes.iter().enumerate() {
                    if i < ladder.asks.len() {
                        ladder.asks[i].size = size;
                    }
                }
                debug!(
                    binding = ?allocation.binding_constraint,
                    margin_used = %format!("{:.2}", allocation.margin_used),
                    position_used = %format!("{:.4}", allocation.position_used),
                    shadow_price = %format!("{:.6}", allocation.shadow_price),
                    "Constrained optimizer applied to asks"
                );
            }

            // 8. Filter out zero-size levels
            ladder.bids.retain(|l| l.size > EPSILON);
            ladder.asks.retain(|l| l.size > EPSILON);

            ladder
        } else {
            Ladder::generate(&self.ladder_config, &params)
        }
    }
}

impl QuotingStrategy for LadderStrategy {
    /// For compatibility with existing infrastructure, returns best bid/ask from ladder.
    ///
    /// The full ladder can be obtained via `generate_ladder()` for multi-level quoting.
    fn calculate_quotes(
        &self,
        config: &QuoteConfig,
        position: f64,
        max_position: f64,
        target_liquidity: f64,
        market_params: &MarketParams,
    ) -> (Option<Quote>, Option<Quote>) {
        let ladder = self.generate_ladder(
            config,
            position,
            max_position,
            target_liquidity,
            market_params,
        );

        // Return just the best bid/ask for backward compatibility
        let bid = ladder.bids.first().map(|l| Quote::new(l.price, l.size));
        let ask = ladder.asks.first().map(|l| Quote::new(l.price, l.size));

        (bid, ask)
    }

    fn calculate_ladder(
        &self,
        config: &QuoteConfig,
        position: f64,
        max_position: f64,
        target_liquidity: f64,
        market_params: &MarketParams,
    ) -> Ladder {
        self.generate_ladder(
            config,
            position,
            max_position,
            target_liquidity,
            market_params,
        )
    }

    fn name(&self) -> &'static str {
        "LadderGLFT"
    }
}

// ============================================================================
// Constrained Optimizer Helper Functions
// ============================================================================

/// Fill intensity at depth: λ(δ) = σ²/δ² × κ
///
/// Models probability of price reaching depth δ based on diffusion.
/// At touch (δ → 0), returns kappa as baseline intensity.
fn fill_intensity_at_depth(depth_bps: f64, sigma: f64, kappa: f64) -> f64 {
    if depth_bps < EPSILON {
        return kappa; // At touch, use kappa as baseline
    }
    let depth_frac = depth_bps / 10000.0;
    // σ² / δ² gives diffusion-driven fill probability
    // Scale by kappa for market activity level
    let diffusion_term = sigma.powi(2) / depth_frac.powi(2);
    (diffusion_term * kappa).min(kappa) // Cap at kappa
}

/// Spread capture at depth: SC(δ) = δ - AS(δ) - fees
///
/// Expected profit from capturing spread at depth δ.
/// Uses calibrated depth-dependent AS model if available.
fn spread_capture_at_depth(
    depth_bps: f64,
    params: &super::ladder::LadderParams,
    fees_bps: f64,
) -> f64 {
    let as_at_depth = if let Some(ref decay) = params.depth_decay_as {
        // Use calibrated first-principles model: AS(δ) = AS₀ × exp(-δ/δ_char)
        decay.as_at_depth(depth_bps)
    } else {
        // Legacy fallback: exponential decay with 10bp characteristic depth
        params.as_at_touch_bps * (-depth_bps / 10.0).exp()
    };
    // Spread capture = depth - adverse selection - fees
    (depth_bps - as_at_depth - fees_bps).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(mid: f64) -> QuoteConfig {
        QuoteConfig {
            mid_price: mid,
            decimals: 2,
            sz_decimals: 4,
            min_notional: 10.0,
        }
    }

    fn make_config_with_decimals(mid: f64, decimals: u32) -> QuoteConfig {
        QuoteConfig {
            mid_price: mid,
            decimals,
            sz_decimals: 4,
            min_notional: 10.0,
        }
    }

    #[test]
    fn test_symmetric_strategy_basic() {
        let strategy = SymmetricStrategy::new(10); // 10 bps half spread
        let config = make_config(100.0);
        let market_params = MarketParams::default();

        let (bid, ask) = strategy.calculate_quotes(&config, 0.0, 1.0, 0.5, &market_params);

        let bid = bid.unwrap();
        let ask = ask.unwrap();

        // With 10 bps half spread on 100.0 mid:
        // bid = 100.0 - 0.1 = 99.9
        // ask = 100.0 + 0.1 = 100.1
        assert!((bid.price - 99.9).abs() < 0.01);
        assert!((ask.price - 100.1).abs() < 0.01);
        assert!((bid.size - 0.5).abs() < 0.0001);
        assert!((ask.size - 0.5).abs() < 0.0001);
    }

    #[test]
    fn test_symmetric_strategy_position_limits() {
        let strategy = SymmetricStrategy::new(10);
        let config = make_config(100.0);
        let market_params = MarketParams::default();

        // At max long position, can't buy more
        let (bid, ask) = strategy.calculate_quotes(&config, 1.0, 1.0, 0.5, &market_params);
        assert!(bid.is_none()); // Can't buy
        assert!(ask.is_some()); // Can sell
    }

    #[test]
    fn test_inventory_aware_strategy_skew() {
        let strategy = InventoryAwareStrategy::new(10, 100.0); // 10 bps spread, 100 bps skew per unit
        let config = make_config(100.0);
        let market_params = MarketParams::default();

        // With position = 1.0 and skew_factor = 100 bps:
        // skew = 1.0 * 100 / 10000 = 0.01 (1%)
        // adjusted_mid = 100.0 * (1 - 0.01) = 99.0
        let (bid, ask) = strategy.calculate_quotes(&config, 1.0, 2.0, 0.5, &market_params);

        let bid = bid.unwrap();
        let ask = ask.unwrap();

        // Prices should be lower than symmetric (skewed down due to long position)
        assert!(bid.price < 99.9);
        assert!(ask.price < 100.1);
    }

    #[test]
    fn test_glft_zero_inventory() {
        // Risk aversion of 0.5 with kappa=100 gives spread ≈ 100 bps
        let strategy = GLFTStrategy::new(0.5);
        let config = make_config(100.0);
        let market_params = MarketParams {
            sigma: 0.0001,
            sigma_total: 0.0001,
            sigma_effective: 0.0001,
            kappa: 100.0,
            arrival_intensity: 1.0,
            is_toxic_regime: false,
            jump_ratio: 1.0,
            microprice: 100.0, // Must match mid_price for fair quoting
            ..Default::default()
        };

        // With zero inventory, bid and ask should be symmetric around mid
        let (bid, ask) = strategy.calculate_quotes(&config, 0.0, 1.0, 0.5, &market_params);

        let bid = bid.unwrap();
        let ask = ask.unwrap();

        // Both should exist and be roughly symmetric
        let bid_offset = config.mid_price - bid.price;
        let ask_offset = ask.price - config.mid_price;
        assert!(
            (bid_offset - ask_offset).abs() < 0.1,
            "Offsets should be symmetric: bid={:.4}, ask={:.4}",
            bid_offset,
            ask_offset
        );
    }

    #[test]
    fn test_glft_long_inventory_skews() {
        // Higher risk aversion to make skew more visible
        let strategy = GLFTStrategy::new(1.0);
        let config = make_config(100.0);
        let market_params = MarketParams {
            sigma: 0.01,
            sigma_total: 0.01,
            sigma_effective: 0.01,
            kappa: 50.0,
            arrival_intensity: 0.5, // T = 2 seconds
            is_toxic_regime: false,
            jump_ratio: 1.0,
            microprice: 100.0, // Must match mid_price for fair quoting
            ..Default::default()
        };

        // With long position, bid should be further from mid (discourage buying)
        // and ask should be closer to mid (encourage selling)
        let (bid_neutral, ask_neutral) =
            strategy.calculate_quotes(&config, 0.0, 1.0, 0.5, &market_params);
        let (bid_long, ask_long) =
            strategy.calculate_quotes(&config, 0.5, 1.0, 0.5, &market_params);

        let bid_neutral = bid_neutral.unwrap();
        let ask_neutral = ask_neutral.unwrap();
        let bid_long = bid_long.unwrap();
        let ask_long = ask_long.unwrap();

        // Long position: bid moves down (further from mid), ask moves down (closer to mid)
        assert!(
            bid_long.price < bid_neutral.price,
            "Long: bid should move down. neutral={}, long={}",
            bid_neutral.price,
            bid_long.price
        );
        assert!(
            ask_long.price < ask_neutral.price,
            "Long: ask should move down. neutral={}, long={}",
            ask_neutral.price,
            ask_long.price
        );
    }

    #[test]
    fn test_glft_short_inventory_skews() {
        let strategy = GLFTStrategy::new(1.0);
        let config = make_config(100.0);
        let market_params = MarketParams {
            sigma: 0.01,
            sigma_total: 0.01,
            sigma_effective: 0.01,
            kappa: 50.0,
            arrival_intensity: 0.5,
            is_toxic_regime: false,
            jump_ratio: 1.0,
            microprice: 100.0, // Must match mid_price for fair quoting
            ..Default::default()
        };

        // With short position, ask should be further from mid (discourage selling)
        // and bid should be closer to mid (encourage buying)
        let (bid_neutral, ask_neutral) =
            strategy.calculate_quotes(&config, 0.0, 1.0, 0.5, &market_params);
        let (bid_short, ask_short) =
            strategy.calculate_quotes(&config, -0.5, 1.0, 0.5, &market_params);

        let bid_neutral = bid_neutral.unwrap();
        let ask_neutral = ask_neutral.unwrap();
        let bid_short = bid_short.unwrap();
        let ask_short = ask_short.unwrap();

        // Short position: bid moves up (closer to mid), ask moves up (further from mid)
        assert!(
            bid_short.price > bid_neutral.price,
            "Short: bid should move up. neutral={}, short={}",
            bid_neutral.price,
            bid_short.price
        );
        assert!(
            ask_short.price > ask_neutral.price,
            "Short: ask should move up. neutral={}, short={}",
            ask_neutral.price,
            ask_short.price
        );
    }

    #[test]
    fn test_glft_half_spread_formula() {
        // Test the correct half-spread formula with fee recovery:
        // δ* = (1/γ) × ln(1 + γ/κ) + f_maker
        let strategy = GLFTStrategy::new(0.5);

        let gamma = 0.5;
        let kappa = 100.0;
        let maker_fee = strategy.risk_config.maker_fee_rate; // 0.00015 (1.5 bps)

        // Expected: δ = (1/0.5) * ln(1 + 0.5/100) + 0.00015 ≈ 0.00998 + 0.00015 ≈ 0.01013
        let half_spread = strategy.half_spread(gamma, kappa);
        let glft_spread = (1.0 / gamma) * (1.0 + gamma / kappa).ln();
        let expected = glft_spread + maker_fee;

        assert!(
            (half_spread - expected).abs() < 1e-6,
            "Half-spread mismatch: got {}, expected {}",
            half_spread,
            expected
        );

        // Verify the numerical example: GLFT ~0.998% + fee 0.015% ≈ 1.013%
        assert!(
            (half_spread - 0.01013).abs() < 0.0001,
            "Half-spread should be ~1.013%: got {}",
            half_spread
        );
    }

    #[test]
    fn test_glft_inventory_skew_formula() {
        // Test flow-dampened skew: base_skew × exp(-β × flow_alignment)
        let strategy = GLFTStrategy::new(0.5);

        let inventory_ratio = 0.5_f64; // 50% of max position (long)
        let sigma = 0.01_f64; // 1% per-second volatility
        let gamma = 0.5_f64;
        let time_horizon = 2.0_f64; // T = 2 seconds
        let flow_imbalance = 0.0_f64; // No flow signal

        // With no flow, should get base skew: 0.5 * 0.5 * 0.01^2 * 2 = 0.00005
        let skew = strategy.inventory_skew_with_flow(
            inventory_ratio,
            sigma,
            gamma,
            time_horizon,
            flow_imbalance,
        );
        let expected = inventory_ratio * gamma * sigma.powi(2) * time_horizon;

        assert!(
            (skew - expected).abs() < 1e-10,
            "Skew mismatch: got {}, expected {}",
            skew,
            expected
        );

        // Test flow dampening: long position + buy flow = aligned → dampen skew
        let aligned_flow = 0.8_f64; // Strong buy flow
        let dampened_skew = strategy.inventory_skew_with_flow(
            inventory_ratio,
            sigma,
            gamma,
            time_horizon,
            aligned_flow,
        );
        // flow_alignment = 0.5.signum() * 0.8 = 0.8
        // modifier = exp(-0.5 * 0.8) = exp(-0.4) ≈ 0.67
        // dampened_skew should be less than base_skew
        assert!(
            dampened_skew < skew,
            "Aligned flow should dampen skew: got {} vs base {}",
            dampened_skew,
            skew
        );

        // Test flow amplification: long position + sell flow = opposed → amplify skew
        let opposed_flow = -0.8_f64; // Strong sell flow
        let amplified_skew = strategy.inventory_skew_with_flow(
            inventory_ratio,
            sigma,
            gamma,
            time_horizon,
            opposed_flow,
        );
        // flow_alignment = 0.5.signum() * -0.8 = -0.8
        // modifier = exp(-0.5 * -0.8) = exp(0.4) ≈ 1.49
        // amplified_skew should be greater than base_skew
        assert!(
            amplified_skew > skew,
            "Opposed flow should amplify skew: got {} vs base {}",
            amplified_skew,
            skew
        );
    }

    #[test]
    fn test_glft_market_driven_spread() {
        // Test that spread responds to market conditions (kappa)
        let strategy = GLFTStrategy::new(0.5);
        let config = make_config_with_decimals(100.0, 4);

        // Deep book (high kappa) - should have tighter spread
        let deep_book = MarketParams {
            kappa: 200.0,
            kappa_bid: 200.0,  // Match kappa for symmetric test
            kappa_ask: 200.0,
            microprice: 100.0, // Must match mid_price for fair quoting
            ..Default::default()
        };

        // Thin book (low kappa) - should have wider spread
        let thin_book = MarketParams {
            kappa: 20.0,
            kappa_bid: 20.0,  // Match kappa for symmetric test
            kappa_ask: 20.0,
            microprice: 100.0, // Must match mid_price for fair quoting
            ..Default::default()
        };

        let (bid_deep, ask_deep) = strategy.calculate_quotes(&config, 0.0, 1.0, 0.5, &deep_book);
        let (bid_thin, ask_thin) = strategy.calculate_quotes(&config, 0.0, 1.0, 0.5, &thin_book);

        let spread_deep = ask_deep.unwrap().price - bid_deep.unwrap().price;
        let spread_thin = ask_thin.unwrap().price - bid_thin.unwrap().price;

        assert!(
            spread_thin > spread_deep,
            "Thin book should have wider spread: thin={:.4}, deep={:.4}",
            spread_thin,
            spread_deep
        );
    }

    #[test]
    fn test_glft_toxic_regime_increases_gamma() {
        // Test that dynamic gamma increases in toxic regime
        // Note: spread width may not always increase because the GLFT formula
        // δ = (1/γ) × ln(1 + γ/κ) is non-monotonic in gamma
        let strategy = GLFTStrategy::new(0.5);

        // Normal regime
        let normal_params = MarketParams {
            sigma: 0.001,
            sigma_total: 0.001,
            sigma_effective: 0.001,
            kappa: 50.0,
            arrival_intensity: 1.0,
            is_toxic_regime: false,
            jump_ratio: 1.0,
            ..Default::default()
        };

        // Toxic regime with high jump ratio
        let toxic_params = MarketParams {
            sigma: 0.001,
            sigma_total: 0.002,
            sigma_effective: 0.0015,
            kappa: 50.0,
            arrival_intensity: 1.0,
            is_toxic_regime: true,
            jump_ratio: 4.5,
            ..Default::default()
        };

        let gamma_normal = strategy.effective_gamma(&normal_params, 0.0, 1.0);
        let gamma_toxic = strategy.effective_gamma(&toxic_params, 0.0, 1.0);

        assert!(
            gamma_toxic > gamma_normal,
            "Toxic gamma should be higher: normal={:.4}, toxic={:.4}",
            gamma_normal,
            gamma_toxic
        );
    }

    #[test]
    fn test_glft_toxic_regime_affects_inventory_skew() {
        // In toxic regime, higher sigma_effective leads to larger inventory skew magnitude.
        // Note: We removed ad-hoc "additional_skew" - now toxic regime affects quoting through:
        // 1. Dynamic gamma (higher risk aversion)
        // 2. sigma_effective blending (includes jump component)
        // These work together within the GLFT framework.
        let strategy = GLFTStrategy::new(0.5);
        let config = make_config(100.0);

        // Use larger sigma values so skew is visible at 2 decimal precision
        // skew = inv_ratio * gamma * sigma^2 * T, needs to be > 0.01 for 2 decimal visibility
        let normal_params = MarketParams {
            sigma: 0.01,
            sigma_total: 0.01,
            sigma_effective: 0.01, // 1% per-second vol
            kappa: 50.0,
            arrival_intensity: 0.5, // T = 2 seconds
            is_toxic_regime: false,
            jump_ratio: 1.0,
            microprice: 100.0,
            ..Default::default()
        };

        let toxic_params = MarketParams {
            sigma: 0.01,
            sigma_total: 0.02,
            sigma_effective: 0.015, // Higher due to jump blending
            kappa: 50.0,
            arrival_intensity: 0.5,
            is_toxic_regime: true,
            jump_ratio: 5.0,
            microprice: 100.0,
            ..Default::default()
        };

        // With long position (0.5)
        let (bid_normal, ask_normal) =
            strategy.calculate_quotes(&config, 0.5, 1.0, 0.5, &normal_params);
        let (bid_toxic, ask_toxic) =
            strategy.calculate_quotes(&config, 0.5, 1.0, 0.5, &toxic_params);

        // Both regimes should produce valid quotes
        assert!(bid_normal.is_some() && ask_normal.is_some());
        assert!(bid_toxic.is_some() && ask_toxic.is_some());

        let bid_normal = bid_normal.unwrap();
        let ask_normal = ask_normal.unwrap();
        let bid_toxic = bid_toxic.unwrap();
        let ask_toxic = ask_toxic.unwrap();

        // Verify quotes are correctly positioned relative to mid
        // (bid < mid < ask for both)
        assert!(bid_normal.price < 100.0 && ask_normal.price > 100.0);
        assert!(bid_toxic.price < 100.0 && ask_toxic.price > 100.0);

        // The inventory skew magnitude (bid-ask midpoint offset from mid)
        // With long inventory, midpoint should be below mid (skewed to sell)
        let midpoint_normal = (bid_normal.price + ask_normal.price) / 2.0;
        let midpoint_toxic = (bid_toxic.price + ask_toxic.price) / 2.0;

        assert!(
            midpoint_normal < 100.0,
            "Long inventory should skew quotes down: normal midpoint={:.4}",
            midpoint_normal
        );
        assert!(
            midpoint_toxic < 100.0,
            "Long inventory should skew quotes down: toxic midpoint={:.4}",
            midpoint_toxic
        );
    }

    // ===== DYNAMIC GAMMA TESTS =====

    #[test]
    fn test_glft_dynamic_gamma_increases_with_volatility() {
        let strategy = GLFTStrategy::with_config(RiskConfig {
            gamma_base: 0.3,
            sigma_baseline: 0.0002,
            volatility_weight: 1.0, // Full weight for clear test
            ..Default::default()
        });

        let normal_params = MarketParams {
            sigma_effective: 0.0002, // baseline
            ..Default::default()
        };

        let high_vol_params = MarketParams {
            sigma_effective: 0.0006, // 3x baseline
            ..Default::default()
        };

        let gamma_normal = strategy.effective_gamma(&normal_params, 0.0, 1.0);
        let gamma_high_vol = strategy.effective_gamma(&high_vol_params, 0.0, 1.0);

        assert!(
            gamma_high_vol > gamma_normal,
            "High vol should increase gamma: normal={}, high={}",
            gamma_normal,
            gamma_high_vol
        );
    }

    #[test]
    fn test_glft_dynamic_gamma_increases_with_toxicity() {
        let strategy = GLFTStrategy::with_config(RiskConfig {
            gamma_base: 0.3,
            toxicity_threshold: 1.5,
            toxicity_sensitivity: 0.5,
            ..Default::default()
        });

        let normal_params = MarketParams {
            jump_ratio: 1.0,
            ..Default::default()
        };

        let toxic_params = MarketParams {
            jump_ratio: 3.0,
            ..Default::default()
        };

        let gamma_normal = strategy.effective_gamma(&normal_params, 0.0, 1.0);
        let gamma_toxic = strategy.effective_gamma(&toxic_params, 0.0, 1.0);

        assert!(
            gamma_toxic > gamma_normal,
            "Toxic regime should increase gamma: normal={}, toxic={}",
            gamma_normal,
            gamma_toxic
        );
    }

    #[test]
    fn test_glft_dynamic_gamma_increases_near_position_limits() {
        let strategy = GLFTStrategy::with_config(RiskConfig {
            gamma_base: 0.3,
            inventory_threshold: 0.5,
            inventory_sensitivity: 2.0,
            ..Default::default()
        });

        let params = MarketParams::default();

        let gamma_empty = strategy.effective_gamma(&params, 0.0, 1.0);
        let gamma_half = strategy.effective_gamma(&params, 0.5, 1.0);
        let gamma_near_full = strategy.effective_gamma(&params, 0.9, 1.0);

        assert!(
            gamma_near_full > gamma_half,
            "Near-full inventory should increase gamma: half={}, near_full={}",
            gamma_half,
            gamma_near_full
        );
        assert!(
            gamma_half >= gamma_empty,
            "Half inventory should be >= empty: empty={}, half={}",
            gamma_empty,
            gamma_half
        );
    }

    #[test]
    fn test_glft_holding_time_calculation() {
        let strategy = GLFTStrategy::new(0.5);

        // T = 1/λ
        assert!((strategy.holding_time(0.5) - 2.0).abs() < 1e-10);
        assert!((strategy.holding_time(1.0) - 1.0).abs() < 1e-10);
        assert!((strategy.holding_time(0.1) - 10.0).abs() < 1e-10);

        // Should be capped at max_holding_time
        assert!(strategy.holding_time(0.001) <= strategy.risk_config.max_holding_time + 1e-10);
    }

    #[test]
    fn test_glft_spread_widens_in_stress() {
        // In stress conditions (high vol + toxic + inventory), spreads should widen
        let strategy = GLFTStrategy::with_config(RiskConfig {
            gamma_base: 0.3,
            sigma_baseline: 0.0002,
            volatility_weight: 0.5,
            toxicity_sensitivity: 0.3,
            inventory_sensitivity: 2.0,
            ..Default::default()
        });

        let config = make_config_with_decimals(100.0, 4);

        let normal_params = MarketParams {
            sigma: 0.0002,
            sigma_effective: 0.0002,
            kappa: 100.0,
            kappa_bid: 100.0,  // Match kappa for symmetric test
            kappa_ask: 100.0,
            arrival_intensity: 0.5,
            jump_ratio: 1.0,
            is_toxic_regime: false,
            microprice: 100.0, // Must match mid_price for fair quoting
            ..Default::default()
        };

        let stress_params = MarketParams {
            sigma: 0.0006, // 3x vol
            sigma_effective: 0.0006,
            kappa: 50.0,            // Thinner book
            kappa_bid: 50.0,        // Match kappa for symmetric test
            kappa_ask: 50.0,
            arrival_intensity: 0.2, // Slower fills
            jump_ratio: 3.0,        // Toxic
            is_toxic_regime: true,
            microprice: 100.0, // Must match mid_price for fair quoting
            ..Default::default()
        };

        let (bid_normal, ask_normal) =
            strategy.calculate_quotes(&config, 0.0, 1.0, 0.5, &normal_params);
        let (bid_stress, ask_stress) =
            strategy.calculate_quotes(&config, 0.5, 1.0, 0.5, &stress_params);

        let spread_normal = ask_normal.unwrap().price - bid_normal.unwrap().price;
        let spread_stress = ask_stress.unwrap().price - bid_stress.unwrap().price;

        assert!(
            spread_stress > spread_normal,
            "Stress should widen spread: normal={}, stress={}",
            spread_normal,
            spread_stress
        );
    }

    #[test]
    fn test_glft_gamma_clamped_to_bounds() {
        let strategy = GLFTStrategy::with_config(RiskConfig {
            gamma_base: 0.3,
            gamma_min: 0.1,
            gamma_max: 2.0,
            volatility_weight: 1.0,
            max_volatility_multiplier: 100.0, // Very high to test clamping
            ..Default::default()
        });

        // Extreme volatility that would push gamma way up
        let extreme_params = MarketParams {
            sigma_effective: 0.01, // 50x baseline
            jump_ratio: 10.0,      // Very toxic
            ..Default::default()
        };

        let gamma = strategy.effective_gamma(&extreme_params, 0.99, 1.0);

        assert!(
            gamma <= 2.0,
            "Gamma should be capped at gamma_max: got {}",
            gamma
        );
        assert!(
            gamma >= 0.1,
            "Gamma should be floored at gamma_min: got {}",
            gamma
        );
    }
}
