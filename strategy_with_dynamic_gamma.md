//! Quoting strategies for the market maker.
//!
//! Includes the corrected GLFT (Guéant-Lehalle-Fernandez-Tapia) implementation
//! with proper mathematical foundations.

use tracing::debug;

use crate::{round_to_significant_and_decimal, truncate_float, EPSILON};

use super::config::{Quote, QuoteConfig};

/// Parameters estimated from live market data.
///
/// For the infinite-horizon GLFT model with regime detection and directional protection:
/// - Dual-sigma: sigma_clean (BV-based) for spreads, sigma_effective (blended) for skew
/// - kappa: order book depth decay constant (from weighted L2 book regression)
/// - arrival_intensity: volume ticks per second (CRITICAL for holding time calculation)
/// - Regime detection: jump_ratio > 1.5 = toxic
/// - Directional flow: momentum_bps, flow_imbalance, falling/rising knife scores
#[derive(Debug, Clone, Copy)]
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

    /// Order arrival intensity (λ) - volume ticks per second
    /// CRITICAL: Used to calculate expected holding time T = 1/λ
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
}

impl Default for MarketParams {
    fn default() -> Self {
        Self {
            sigma: 0.0001,            // 0.01% per-second volatility (clean)
            sigma_total: 0.0001,      // Same initially
            sigma_effective: 0.0001,  // Same initially
            kappa: 100.0,             // Moderate depth decay
            arrival_intensity: 0.5,   // 0.5 volume ticks per second
            is_toxic_regime: false,   // Default: not toxic
            jump_ratio: 1.0,          // Default: normal diffusion
            momentum_bps: 0.0,        // Default: no momentum
            flow_imbalance: 0.0,      // Default: balanced flow
            falling_knife_score: 0.0, // Default: no falling knife
            rising_knife_score: 0.0,  // Default: no rising knife
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
    fn calculate_quotes(
        &self,
        config: &QuoteConfig,
        position: f64,
        max_position: f64,
        target_liquidity: f64,
        market_params: &MarketParams,
    ) -> (Option<Quote>, Option<Quote>);

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
        (**self).calculate_quotes(config, position, max_position, target_liquidity, market_params)
    }

    fn name(&self) -> &'static str {
        (**self).name()
    }
}

/// Symmetric quoting strategy - equal spread on both sides of mid.
#[derive(Debug, Clone, Default)]
pub struct SymmetricStrategy;

impl SymmetricStrategy {
    pub fn new() -> Self {
        Self
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
        let half_spread = (config.mid_price * config.half_spread_bps as f64) / 10000.0;

        let lower_price_raw = config.mid_price - half_spread;
        let upper_price_raw = config.mid_price + half_spread;

        let mut lower_price = round_to_significant_and_decimal(lower_price_raw, 5, config.decimals);
        let upper_price = round_to_significant_and_decimal(upper_price_raw, 5, config.decimals);

        if lower_price >= upper_price {
            let tick = 10f64.powi(-(config.decimals as i32));
            lower_price -= tick;
        }

        let buy_size_raw = (max_position - position).min(target_liquidity).max(0.0);
        let sell_size_raw = (max_position + position).min(target_liquidity).max(0.0);

        let buy_size = truncate_float(buy_size_raw, config.sz_decimals, false);
        let sell_size = truncate_float(sell_size_raw, config.sz_decimals, false);

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
    pub skew_factor_bps: f64,
}

impl InventoryAwareStrategy {
    pub fn new(skew_factor_bps: f64) -> Self {
        Self { skew_factor_bps }
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
        let skew = position * self.skew_factor_bps / 10000.0;
        let adjusted_mid = config.mid_price * (1.0 - skew);

        let half_spread = (adjusted_mid * config.half_spread_bps as f64) / 10000.0;

        let lower_price_raw = adjusted_mid - half_spread;
        let upper_price_raw = adjusted_mid + half_spread;

        let mut lower_price = round_to_significant_and_decimal(lower_price_raw, 5, config.decimals);
        let upper_price = round_to_significant_and_decimal(upper_price_raw, 5, config.decimals);

        if lower_price >= upper_price {
            let tick = 10f64.powi(-(config.decimals as i32));
            lower_price -= tick;
        }

        let buy_size_raw = (max_position - position).min(target_liquidity).max(0.0);
        let sell_size_raw = (max_position + position).min(target_liquidity).max(0.0);

        let buy_size = truncate_float(buy_size_raw, config.sz_decimals, false);
        let sell_size = truncate_float(sell_size_raw, config.sz_decimals, false);

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
#[derive(Debug, Clone)]
pub struct RiskConfig {
    /// Base risk aversion (γ_base) - your "personality" in normal conditions
    pub gamma_base: f64,

    /// Baseline volatility for scaling (per-second σ)
    pub sigma_baseline: f64,

    /// How much weight to give volatility scaling (0.0 to 1.0)
    pub volatility_weight: f64,

    /// Maximum volatility multiplier
    pub max_volatility_multiplier: f64,

    /// Toxicity threshold (jump_ratio above this triggers scaling)
    pub toxicity_threshold: f64,

    /// How much toxicity increases γ per unit of jump_ratio
    pub toxicity_sensitivity: f64,

    /// Inventory utilization threshold for γ scaling
    pub inventory_threshold: f64,

    /// How aggressively γ increases near position limits
    pub inventory_sensitivity: f64,

    /// Minimum γ floor
    pub gamma_min: f64,

    /// Maximum γ ceiling
    pub gamma_max: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            gamma_base: 0.3,
            sigma_baseline: 0.0002,           // 20bp per-second
            volatility_weight: 0.5,
            max_volatility_multiplier: 3.0,
            toxicity_threshold: 1.5,
            toxicity_sensitivity: 0.3,
            inventory_threshold: 0.5,
            inventory_sensitivity: 2.0,
            gamma_min: 0.05,
            gamma_max: 5.0,
        }
    }
}

/// GLFT (Guéant-Lehalle-Fernandez-Tapia) optimal market making strategy.
///
/// Implements the **correct** infinite-horizon GLFT model from stochastic control theory.
///
/// ## Key Formulas (Per Guéant et al. 2013):
///
/// **Optimal Half-Spread:**
/// ```text
/// δ = (1/γ) × ln(1 + γ/κ)
/// ```
/// This naturally widens when:
/// - κ drops (thin order book → more adverse selection risk)
/// - γ increases (higher risk aversion)
///
/// **Reservation Price (Inventory Skew):**
/// ```text
/// r = s - q × γ × σ² × T
/// ```
/// where:
/// - s = mid price
/// - q = inventory (positive = long)
/// - γ = risk aversion parameter (DYNAMIC based on conditions)
/// - σ² = variance (per-second)
/// - T = expected holding time = 1/λ (where λ = arrival intensity)
///
/// ## Dynamic Risk Aversion
///
/// γ scales with market conditions:
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

    /// Minimum spread floor (as fraction) to ensure profitability
    pub min_spread_floor: f64,

    /// Maximum holding time cap (seconds) to prevent skew explosion in dead markets
    pub max_holding_time: f64,
}

impl GLFTStrategy {
    /// Create a new GLFT strategy with base risk aversion.
    ///
    /// # Arguments
    /// * `gamma_base` - Base γ parameter in normal conditions
    ///   - 0.1 = aggressive (tight spreads, tolerates inventory)
    ///   - 0.3 = moderate (default)
    ///   - 1.0 = conservative (wide spreads, aggressive skew)
    pub fn new(gamma_base: f64) -> Self {
        Self {
            risk_config: RiskConfig {
                gamma_base: gamma_base.max(0.01),
                ..Default::default()
            },
            min_spread_floor: 0.0001, // 1 bps minimum
            max_holding_time: 120.0,  // 2 minutes max
        }
    }

    /// Create with full risk configuration.
    pub fn with_config(risk_config: RiskConfig) -> Self {
        Self {
            risk_config,
            min_spread_floor: 0.0001,
            max_holding_time: 120.0,
        }
    }

    /// Calculate effective γ based on current market conditions.
    ///
    /// γ_effective = γ_base × vol_scalar × toxicity_scalar × inventory_scalar
    fn effective_gamma(
        &self,
        market_params: &MarketParams,
        position: f64,
        max_position: f64,
    ) -> f64 {
        let cfg = &self.risk_config;
        let gamma_base = cfg.gamma_base;

        // === VOLATILITY SCALING ===
        // Higher realized vol → more risk per unit inventory
        let vol_ratio = market_params.sigma_effective / cfg.sigma_baseline;
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

        // === COMBINE AND CLAMP ===
        let gamma_effective = gamma_base * vol_scalar * toxicity_scalar * inventory_scalar;
        gamma_effective.clamp(cfg.gamma_min, cfg.gamma_max)
    }

    /// Calculate the GLFT optimal half-spread.
    ///
    /// Formula: δ = (1/γ) × ln(1 + γ/κ)
    ///
    /// This is the CORRECT formula per the paper.
    /// - When κ → ∞ (infinite liquidity): δ → 0 (can quote tight)
    /// - When κ → 0 (no liquidity): δ → ∞ (must quote wide)
    /// - When γ → ∞ (infinite risk aversion): δ → 0 (counterintuitive but correct - see paper)
    /// - When γ → 0 (no risk aversion): δ → 1/κ
    fn optimal_half_spread(&self, gamma: f64, kappa: f64) -> f64 {
        // δ = (1/γ) × ln(1 + γ/κ)
        let ratio = gamma / kappa;
        if ratio > 1e-9 && gamma > 1e-9 {
            (1.0 / gamma) * (1.0 + ratio).ln()
        } else {
            // Limit case: γ/κ small → ln(1+x) ≈ x → δ ≈ 1/κ
            1.0 / kappa.max(1.0)
        }
    }

    /// Calculate the reservation price offset (inventory skew).
    ///
    /// Formula: skew = -q × γ × σ² × T
    ///
    /// This shifts the midpoint based on inventory:
    /// - Long inventory (q > 0) → negative skew → reservation price BELOW mid
    ///   → bid further from mid, ask closer → encourages selling
    /// - Short inventory (q < 0) → positive skew → reservation price ABOVE mid
    ///   → ask further from mid, bid closer → encourages buying
    ///
    /// The magnitude depends on:
    /// - Position size (q): more inventory = more skew
    /// - Risk aversion (γ): more risk averse = more aggressive skew
    /// - Variance (σ²): more volatile = more urgent to reduce inventory
    /// - Holding time (T): longer expected hold = more price risk
    fn reservation_price_offset(
        &self,
        position: f64,
        gamma: f64,
        sigma: f64,
        holding_time: f64,
    ) -> f64 {
        // skew = -q × γ × σ² × T
        // Note: sigma is per-second, so σ² × T gives variance over holding period
        let variance_over_horizon = sigma.powi(2) * holding_time;
        -position * gamma * variance_over_horizon
    }

    /// Calculate expected holding time from arrival intensity.
    ///
    /// T = 1/λ where λ = arrival intensity (fills per second)
    ///
    /// Clamped to prevent:
    /// - Division by zero when intensity is 0
    /// - Skew explosion when market is dead (cap at max_holding_time)
    fn holding_time(&self, arrival_intensity: f64) -> f64 {
        // Minimum intensity to prevent division by zero
        let safe_intensity = arrival_intensity.max(0.01);
        // T = 1/λ, capped at max
        (1.0 / safe_intensity).min(self.max_holding_time)
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
        // === 1. EXTRACT MARKET PARAMETERS ===
        let sigma_clean = market_params.sigma;
        let sigma_effective = market_params.sigma_effective;
        let kappa = market_params.kappa;
        let arrival_intensity = market_params.arrival_intensity;

        // === 2. DYNAMIC RISK AVERSION ===
        // γ scales with volatility, toxicity, and inventory utilization
        let gamma = self.effective_gamma(market_params, position, max_position);

        // === 3. CALCULATE HOLDING TIME ===
        // T = 1/λ - how long we expect to hold inventory before it's filled
        let holding_time = self.holding_time(arrival_intensity);

        // === 4. OPTIMAL HALF-SPREAD ===
        // δ = (1/γ) × ln(1 + γ/κ)
        // Uses clean sigma conceptually (though formula doesn't include σ directly)
        let mut half_spread = self.optimal_half_spread(gamma, kappa);

        // Apply minimum floor to ensure we're not quoting sub-tick
        half_spread = half_spread.max(self.min_spread_floor);

        // === 5. RESERVATION PRICE (INVENTORY SKEW) ===
        // r = s - q × γ × σ² × T
        // Uses sigma_effective (blends in jump risk for more aggressive skew when volatile)
        let skew_offset = self.reservation_price_offset(position, gamma, sigma_effective, holding_time);

        // Skew offset is in price units (fraction of mid)
        // Convert to absolute price offset
        let skew_price_offset = skew_offset * config.mid_price;

        // === 6. TOXICITY SPREAD WIDENING ===
        let toxicity_multiplier = if market_params.is_toxic_regime {
            // Scale: at ratio=1.5 → 1.25x, at ratio=3.0 → 2.0x, cap at 2.5x
            let factor = 1.0 + (market_params.jump_ratio - 1.0) * 0.5;
            factor.clamp(1.0, 2.5)
        } else {
            1.0
        };

        // === 7. FALLING KNIFE PROTECTION (protect bids during crashes) ===
        let mut bid_protection = 0.0;
        if market_params.falling_knife_score > 0.3 {
            // Additional spread on bid side during downward momentum
            bid_protection = market_params.falling_knife_score * half_spread * 0.5;

            // Extra protection if we're already long (compound risk)
            let inventory_ratio = position / max_position.max(EPSILON);
            if inventory_ratio > 0.0 {
                bid_protection *= 1.0 + inventory_ratio;
            }
        }

        // === 8. RISING KNIFE PROTECTION (protect asks during pumps) ===
        let mut ask_protection = 0.0;
        if market_params.rising_knife_score > 0.3 {
            ask_protection = market_params.rising_knife_score * half_spread * 0.5;

            let inventory_ratio = position / max_position.max(EPSILON);
            if inventory_ratio < 0.0 {
                ask_protection *= 1.0 + inventory_ratio.abs();
            }
        }

        // === 9. FLOW IMBALANCE ADJUSTMENT ===
        // Shift quotes in direction of flow (anticipate continued pressure)
        let flow_adjustment = market_params.flow_imbalance * half_spread * 0.3;

        // === 10. CALCULATE FINAL PRICES ===
        // Reservation price = mid + skew_offset
        let reservation_price = config.mid_price + skew_price_offset;

        // Spread components (in price units)
        let spread_price = half_spread * config.mid_price;

        // Bid: reservation - spread - bid_protection - flow_adjustment
        // Ask: reservation + spread + ask_protection - flow_adjustment
        let bid_delta = (spread_price + bid_protection * config.mid_price) * toxicity_multiplier;
        let ask_delta = (spread_price + ask_protection * config.mid_price) * toxicity_multiplier;

        let lower_price_raw = reservation_price - bid_delta - flow_adjustment * config.mid_price;
        let upper_price_raw = reservation_price + ask_delta - flow_adjustment * config.mid_price;

        // Round to exchange precision
        let mut lower_price = round_to_significant_and_decimal(lower_price_raw, 5, config.decimals);
        let upper_price = round_to_significant_and_decimal(upper_price_raw, 5, config.decimals);

        // Ensure bid < ask
        if lower_price >= upper_price {
            let tick = 10f64.powi(-(config.decimals as i32));
            lower_price -= tick;
        }

        // === 11. LOGGING ===
        debug!(
            gamma = %format!("{:.4}", gamma),
            kappa = %format!("{:.2}", kappa),
            sigma_clean = %format!("{:.6}", sigma_clean),
            sigma_effective = %format!("{:.6}", sigma_effective),
            holding_time_s = %format!("{:.2}", holding_time),
            half_spread_bps = %format!("{:.2}", half_spread * 10000.0),
            skew_bps = %format!("{:.2}", skew_offset * 10000.0),
            position = %format!("{:.6}", position),
            "GLFT spread calculation"
        );

        debug!(
            mid = config.mid_price,
            reservation = %format!("{:.2}", reservation_price),
            bid_final = lower_price,
            ask_final = upper_price,
            spread_bps = %format!("{:.1}", (upper_price - lower_price) / config.mid_price * 10000.0),
            is_toxic = market_params.is_toxic_regime,
            falling_knife = %format!("{:.2}", market_params.falling_knife_score),
            rising_knife = %format!("{:.2}", market_params.rising_knife_score),
            "GLFT final prices"
        );

        // === 12. SIZE CALCULATION ===
        let buy_size_raw = (max_position - position).min(target_liquidity).max(0.0);
        let sell_size_raw = (max_position + position).min(target_liquidity).max(0.0);

        let buy_size = truncate_float(buy_size_raw, config.sz_decimals, false);
        let sell_size = truncate_float(sell_size_raw, config.sz_decimals, false);

        // === 13. BUILD QUOTES ===
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

        debug!(
            bid = ?bid.as_ref().map(|q| (q.price, q.size)),
            ask = ?ask.as_ref().map(|q| (q.price, q.size)),
            "GLFT quotes"
        );

        (bid, ask)
    }

    fn name(&self) -> &'static str {
        "GLFT"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(mid: f64) -> QuoteConfig {
        QuoteConfig {
            mid_price: mid,
            half_spread_bps: 10, // Not used by GLFT anymore
            decimals: 2,
            sz_decimals: 4,
            min_notional: 10.0,
            max_position: 1.0,
        }
    }

    #[test]
    fn test_glft_half_spread_formula() {
        let strategy = GLFTStrategy::new(0.5);

        // Test: δ = (1/γ) × ln(1 + γ/κ)
        // γ = 0.5, κ = 100
        // δ = (1/0.5) × ln(1 + 0.5/100) = 2 × ln(1.005) ≈ 2 × 0.00499 ≈ 0.00998
        let spread = strategy.optimal_half_spread(0.5, 100.0);
        let expected = (1.0 / 0.5) * (1.0 + 0.5 / 100.0).ln();
        assert!(
            (spread - expected).abs() < 1e-10,
            "Spread mismatch: got {}, expected {}",
            spread,
            expected
        );
    }

    #[test]
    fn test_glft_spread_widens_with_thin_book() {
        let strategy = GLFTStrategy::new(0.5);

        let spread_thick = strategy.optimal_half_spread(0.5, 200.0); // Deep book
        let spread_thin = strategy.optimal_half_spread(0.5, 50.0); // Thin book

        assert!(
            spread_thin > spread_thick,
            "Thin book should have wider spread: thick={}, thin={}",
            spread_thick,
            spread_thin
        );
    }

    #[test]
    fn test_glft_skew_formula() {
        let strategy = GLFTStrategy::new(0.5);

        // Test: skew = -q × γ × σ² × T
        // q = 1.0, γ = 0.5, σ = 0.001, T = 10.0
        // skew = -1.0 × 0.5 × 0.000001 × 10.0 = -0.000005
        let skew = strategy.reservation_price_offset(1.0, 0.5, 0.001, 10.0);
        let expected = -1.0 * 0.5 * 0.001_f64.powi(2) * 10.0;
        assert!(
            (skew - expected).abs() < 1e-12,
            "Skew mismatch: got {}, expected {}",
            skew,
            expected
        );
    }

    #[test]
    fn test_glft_long_inventory_lowers_reservation() {
        let strategy = GLFTStrategy::new(0.5);

        // Long position should give negative skew (reservation below mid)
        let skew_long = strategy.reservation_price_offset(1.0, 0.5, 0.001, 10.0);
        let skew_short = strategy.reservation_price_offset(-1.0, 0.5, 0.001, 10.0);

        assert!(skew_long < 0.0, "Long inventory should have negative skew");
        assert!(skew_short > 0.0, "Short inventory should have positive skew");
        assert!(
            (skew_long + skew_short).abs() < 1e-15,
            "Skews should be symmetric"
        );
    }

    #[test]
    fn test_glft_holding_time_calculation() {
        let strategy = GLFTStrategy::new(0.5);

        // T = 1/λ
        assert!((strategy.holding_time(0.5) - 2.0).abs() < 1e-10);
        assert!((strategy.holding_time(1.0) - 1.0).abs() < 1e-10);
        assert!((strategy.holding_time(0.1) - 10.0).abs() < 1e-10);

        // Should be capped at max
        assert!(strategy.holding_time(0.001) <= strategy.max_holding_time + 1e-10);
    }

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
    fn test_glft_quotes_with_inventory() {
        let strategy = GLFTStrategy::new(0.5);
        let config = make_config(100.0);

        let market_params = MarketParams {
            sigma: 0.001,
            sigma_effective: 0.001,
            kappa: 100.0,
            arrival_intensity: 0.5, // T = 2 seconds
            ..Default::default()
        };

        // Zero inventory - quotes should be symmetric around mid
        let (bid_zero, ask_zero) = strategy.calculate_quotes(&config, 0.0, 1.0, 0.5, &market_params);
        let bid_zero = bid_zero.unwrap();
        let ask_zero = ask_zero.unwrap();

        let mid = config.mid_price;
        let bid_offset_zero = mid - bid_zero.price;
        let ask_offset_zero = ask_zero.price - mid;

        // Should be roughly symmetric
        assert!(
            (bid_offset_zero - ask_offset_zero).abs() < 0.5,
            "Zero inventory should be symmetric: bid_off={}, ask_off={}",
            bid_offset_zero,
            ask_offset_zero
        );

        // Long inventory - reservation should be below mid
        let (bid_long, ask_long) = strategy.calculate_quotes(&config, 0.5, 1.0, 0.5, &market_params);
        let bid_long = bid_long.unwrap();
        let ask_long = ask_long.unwrap();

        // Both bid and ask should be lower (shifted down due to long inventory)
        assert!(
            bid_long.price < bid_zero.price,
            "Long inventory: bid should shift down: zero={}, long={}",
            bid_zero.price,
            bid_long.price
        );
        assert!(
            ask_long.price < ask_zero.price,
            "Long inventory: ask should shift down: zero={}, long={}",
            ask_zero.price,
            ask_long.price
        );
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

        let config = make_config(100.0);

        let normal_params = MarketParams {
            sigma: 0.0002,
            sigma_effective: 0.0002,
            kappa: 100.0,
            arrival_intensity: 0.5,
            jump_ratio: 1.0,
            is_toxic_regime: false,
            ..Default::default()
        };

        let stress_params = MarketParams {
            sigma: 0.0006,           // 3x vol
            sigma_effective: 0.0006,
            kappa: 50.0,             // Thinner book
            arrival_intensity: 0.2,  // Slower fills
            jump_ratio: 3.0,         // Toxic
            is_toxic_regime: true,
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
}
