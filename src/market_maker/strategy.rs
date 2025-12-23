//! Quoting strategies for the market maker.

use tracing::debug;

use crate::{round_to_significant_and_decimal, truncate_float, EPSILON};

use super::config::{Quote, QuoteConfig};

/// Parameters estimated from live market data.
///
/// For the infinite-horizon GLFT model with regime detection and directional protection:
/// - Dual-sigma: sigma_clean (BV-based) for spreads, sigma_effective (blended) for skew
/// - kappa: order book depth decay constant (from weighted L2 book regression)
/// - arrival_intensity: volume ticks per second
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

    fn name(&self) -> &'static str {
        (**self).name()
    }
}

/// Symmetric quoting strategy - equal spread on both sides of mid.
#[derive(Debug, Clone, Default)]
pub struct SymmetricStrategy;

impl SymmetricStrategy {
    /// Create a new symmetric strategy.
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
    /// Skew factor in BPS per unit position.
    /// Positive position -> shift mid down (discourage buying, encourage selling).
    pub skew_factor_bps: f64,
}

impl InventoryAwareStrategy {
    /// Create a new inventory-aware strategy.
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
        // Calculate skew based on position
        // Positive position -> negative skew (lower prices to encourage selling)
        let skew = position * self.skew_factor_bps / 10000.0;
        let adjusted_mid = config.mid_price * (1.0 - skew);

        let half_spread = (adjusted_mid * config.half_spread_bps as f64) / 10000.0;

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

/// GLFT (Guéant-Lehalle-Fernandez-Tapia) optimal market making strategy.
///
/// Implements the **infinite-horizon** GLFT model from stochastic control theory.
///
/// ## Key Formulas (Corrected per Guéant et al. 2013):
///
/// **Half-spread (adverse selection protection):**
/// ```text
/// ψ = (1/γ) × ln(1 + γ/κ)
/// ```
///
/// **Reservation price offset (inventory skew):**
/// ```text
/// skew = (q/Q_max) × γ × σ² / κ
/// ```
///
/// **Gamma derivation (from max inventory constraint):**
/// At max inventory, we want skew ≈ half_spread, so:
/// ```text
/// γ = δ × κ / (Q_max × σ²)
/// ```
/// where δ = target half-spread as fraction.
///
/// ## Parameters:
/// - σ (sigma): per-second volatility (NOT annualized)
/// - κ (kappa): order book depth decay (from L2 book)
/// - Q_max: maximum position size
#[derive(Debug, Clone)]
pub struct GLFTStrategy {
    /// Minimum gamma floor (prevents extreme behavior with tiny σ)
    pub min_gamma: f64,
    /// Maximum gamma ceiling (prevents extreme behavior with huge σ)
    pub max_gamma: f64,
}

impl GLFTStrategy {
    /// Create a new GLFT strategy.
    /// Gamma is derived dynamically from the max inventory constraint.
    pub fn new(_target_spread: f64) -> Self {
        // Note: target_spread is now derived from half_spread_bps in config
        Self {
            min_gamma: 0.001,   // Allow very small gamma for large positions
            max_gamma: 10000.0, // Allow large gamma for small positions/low vol
        }
    }

    /// Derive gamma from the max inventory constraint.
    ///
    /// We want: at max inventory (q = Q_max), the skew should equal the half-spread.
    /// From: skew = q × γ × σ² / κ
    /// At q = Q_max, skew = δ (target half-spread)
    /// So: γ = δ × κ / (Q_max × σ²)
    fn derive_gamma(
        &self,
        target_half_spread: f64,
        kappa: f64,
        sigma: f64,
        max_position: f64,
    ) -> f64 {
        // Guard against division by zero
        let sigma_sq = sigma.powi(2).max(1e-12);
        let max_pos = max_position.abs().max(1e-9);

        let gamma = target_half_spread * kappa / (max_pos * sigma_sq);
        gamma.clamp(self.min_gamma, self.max_gamma)
    }

    /// Correct GLFT half-spread formula: ψ = (1/γ) × ln(1 + γ/κ)
    ///
    /// This compensates for adverse selection risk.
    fn half_spread(&self, gamma: f64, kappa: f64) -> f64 {
        let ratio = gamma / kappa;
        if ratio > 1e-6 {
            (1.0 / gamma) * (1.0 + ratio).ln()
        } else {
            // Taylor expansion for small γ/κ: ln(1+x) ≈ x, so ψ ≈ 1/κ
            1.0 / kappa
        }
    }

    /// Infinite-horizon inventory skew: skew = (q/Q_max) × γ × σ² / κ
    ///
    /// Positive inventory → positive skew → wider bid, tighter ask (encourage selling)
    fn inventory_skew(&self, inventory_ratio: f64, sigma: f64, gamma: f64, kappa: f64) -> f64 {
        inventory_ratio * gamma * sigma.powi(2) / kappa
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
        // === 1. USE SIGMA_CLEAN FOR BASE HALF-SPREAD ===
        // sigma (clean) is BV-based, robust to jumps - good for continuous pricing
        let sigma_for_spread = market_params.sigma;
        let kappa = market_params.kappa;

        // Target half-spread from config (as fraction, not bps)
        let target_half_spread = config.half_spread_bps as f64 / 10000.0;

        // Derive gamma for spread calculation using clean sigma
        let gamma_spread = self.derive_gamma(
            target_half_spread,
            kappa,
            sigma_for_spread,
            config.max_position,
        );

        // GLFT half-spread: ψ = (1/γ) × ln(1 + γ/κ)
        let half_spread = self.half_spread(gamma_spread, kappa);

        // === 2. USE SIGMA_EFFECTIVE FOR INVENTORY SKEW ===
        // sigma_effective blends clean and total based on jump regime
        // This makes skew react to jumps (larger skew when RV >> BV)
        let sigma_for_skew = market_params.sigma_effective;
        let gamma_skew = self.derive_gamma(
            target_half_spread,
            kappa,
            sigma_for_skew,
            config.max_position,
        );

        // Calculate inventory ratio: q / Q_max (normalized to [-1, 1])
        let inventory_ratio = if max_position > EPSILON {
            (position / max_position).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // Base inventory skew using sigma_effective
        let base_skew = self.inventory_skew(inventory_ratio, sigma_for_skew, gamma_skew, kappa);

        // === 3. TOXIC REGIME SPREAD WIDENING ===
        // Now triggers at lower threshold (1.5 instead of 3.0)
        let toxicity_multiplier = if market_params.is_toxic_regime {
            // Scale spread multiplier: at ratio=1.5 → 1.0x, at ratio=3.0 → 1.5x, capped at 2.5x
            let factor = (market_params.jump_ratio / 2.0).clamp(1.0, 2.5);
            debug!(
                jump_ratio = %format!("{:.2}", market_params.jump_ratio),
                toxicity_factor = %format!("{:.2}", factor),
                "Toxic regime: widening spread"
            );
            factor
        } else {
            1.0
        };

        // === 4. FALLING KNIFE PROTECTION (protect bids during crashes) ===
        let mut bid_protection = 0.0;
        if market_params.falling_knife_score > 0.5 {
            // Base protection: score * half_spread * 0.5
            bid_protection = market_params.falling_knife_score * half_spread * 0.5;

            // Extra protection if we're already long (compound risk!)
            if inventory_ratio > 0.0 {
                bid_protection *= 1.0 + inventory_ratio;
            }
        }

        // === 5. RISING KNIFE PROTECTION (protect asks during pumps) ===
        let mut ask_protection = 0.0;
        if market_params.rising_knife_score > 0.5 {
            ask_protection = market_params.rising_knife_score * half_spread * 0.5;

            // Extra protection if we're already short
            if inventory_ratio < 0.0 {
                ask_protection *= 1.0 + inventory_ratio.abs();
            }
        }

        // === 6. FLOW IMBALANCE ADJUSTMENT ===
        // If sell pressure (negative), shift quotes down (anticipate continued decline)
        // If buy pressure (positive), shift quotes up
        let flow_adjustment = market_params.flow_imbalance * half_spread * 0.2;

        // === 7. ADDITIONAL TOXIC SKEW ===
        let additional_skew = if market_params.is_toxic_regime {
            let toxicity_excess = (market_params.jump_ratio - 1.0).max(0.0) * 0.1;
            inventory_ratio * toxicity_excess * half_spread
        } else {
            0.0
        };
        let skew = base_skew + additional_skew;

        // === 8. COMBINE ALL ADJUSTMENTS ===
        // bid_delta: base spread + inventory skew + falling knife protection - flow adjustment
        // ask_delta: base spread - inventory skew + rising knife protection - flow adjustment
        let bid_delta =
            (half_spread + skew + bid_protection - flow_adjustment) * toxicity_multiplier;
        let ask_delta = ((half_spread - skew + ask_protection - flow_adjustment).max(0.0))
            * toxicity_multiplier;

        debug!(
            inv_ratio = %format!("{:.4}", inventory_ratio),
            gamma_spread = %format!("{:.2}", gamma_spread),
            gamma_skew = %format!("{:.2}", gamma_skew),
            half_spread = %format!("{:.6}", half_spread),
            skew = %format!("{:.6}", skew),
            bid_protection = %format!("{:.6}", bid_protection),
            ask_protection = %format!("{:.6}", ask_protection),
            flow_adj = %format!("{:.6}", flow_adjustment),
            bid_delta = %format!("{:.6}", bid_delta),
            ask_delta = %format!("{:.6}", ask_delta),
            is_toxic = market_params.is_toxic_regime,
            falling_knife = %format!("{:.2}", market_params.falling_knife_score),
            rising_knife = %format!("{:.2}", market_params.rising_knife_score),
            "GLFT spread components with directional protection"
        );

        // Convert to absolute price offsets
        let bid_offset = config.mid_price * bid_delta;
        let ask_offset = config.mid_price * ask_delta;

        // Calculate raw prices
        let lower_price_raw = config.mid_price - bid_offset;
        let upper_price_raw = config.mid_price + ask_offset;

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
            sigma_clean = %format!("{:.6}", sigma_for_spread),
            sigma_effective = %format!("{:.6}", sigma_for_skew),
            kappa = %format!("{:.2}", kappa),
            jump_ratio = %format!("{:.2}", market_params.jump_ratio),
            momentum_bps = %format!("{:.1}", market_params.momentum_bps),
            flow = %format!("{:.2}", market_params.flow_imbalance),
            bid_final = lower_price,
            ask_final = upper_price,
            spread_bps = %format!("{:.1}", (upper_price - lower_price) / config.mid_price * 10000.0),
            "GLFT prices with directional protection"
        );

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
        "GLFT"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(mid: f64) -> QuoteConfig {
        QuoteConfig {
            mid_price: mid,
            half_spread_bps: 10, // 0.1% = 10 bps
            decimals: 2,
            sz_decimals: 4,
            min_notional: 10.0,
            max_position: 1.0, // Default max position for gamma derivation
        }
    }

    fn make_config_with_max_pos(mid: f64, max_pos: f64) -> QuoteConfig {
        QuoteConfig {
            mid_price: mid,
            half_spread_bps: 100, // 1% = 100 bps for GLFT tests
            decimals: 2,
            sz_decimals: 4,
            min_notional: 10.0,
            max_position: max_pos,
        }
    }

    #[test]
    fn test_symmetric_strategy_basic() {
        let strategy = SymmetricStrategy::new();
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
        let strategy = SymmetricStrategy::new();
        let config = make_config(100.0);
        let market_params = MarketParams::default();

        // At max long position, can't buy more
        let (bid, ask) = strategy.calculate_quotes(&config, 1.0, 1.0, 0.5, &market_params);
        assert!(bid.is_none()); // Can't buy
        assert!(ask.is_some()); // Can sell
    }

    #[test]
    fn test_inventory_aware_strategy_skew() {
        let strategy = InventoryAwareStrategy::new(100.0); // 100 bps per unit
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
        let strategy = GLFTStrategy::new(0.01);
        let config = make_config_with_max_pos(100.0, 1.0);
        // Per-second sigma (√BV), kappa from L2 book
        let market_params = MarketParams {
            sigma: 0.0001, // 0.01% per-second volatility (jump-robust)
            sigma_total: 0.0001,
            sigma_effective: 0.0001,
            kappa: 100.0, // Moderate depth decay
            arrival_intensity: 1.0,
            is_toxic_regime: false,
            jump_ratio: 1.0,
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
        let strategy = GLFTStrategy::new(0.01);
        let config = make_config_with_max_pos(100.0, 1.0);
        // Higher sigma to make skew more visible
        let market_params = MarketParams {
            sigma: 0.001, // 0.1% per-second volatility (higher)
            sigma_total: 0.001,
            sigma_effective: 0.001,
            kappa: 50.0, // Lower kappa = more skew
            arrival_intensity: 1.0,
            is_toxic_regime: false,
            jump_ratio: 1.0,
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
        let strategy = GLFTStrategy::new(0.01);
        let config = make_config_with_max_pos(100.0, 1.0);
        // Higher sigma to make skew more visible
        let market_params = MarketParams {
            sigma: 0.001, // 0.1% per-second volatility
            sigma_total: 0.001,
            sigma_effective: 0.001,
            kappa: 50.0, // Lower kappa = more skew
            arrival_intensity: 1.0,
            is_toxic_regime: false,
            jump_ratio: 1.0,
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
    fn test_glft_gamma_derivation() {
        // Test that gamma is derived correctly from max inventory constraint
        // γ = δ × κ / (Q_max × σ²)
        let strategy = GLFTStrategy::new(0.01);

        let target_half_spread = 0.01; // 1%
        let kappa = 100.0;
        let sigma = 0.001; // 0.1% per-second
        let max_position = 1.0;

        // Expected: γ = 0.01 * 100 / (1.0 * 0.001^2) = 1 / 0.000001 = 1,000,000
        // But clamped to max_gamma (10000)
        let gamma = strategy.derive_gamma(target_half_spread, kappa, sigma, max_position);
        assert!(
            (gamma - 10000.0).abs() < 0.1,
            "Gamma should be clamped to max: {}",
            gamma
        );

        // With larger sigma, gamma should be smaller
        let sigma_large = 0.01; // 1% per-second
                                // Expected: γ = 0.01 * 100 / (1.0 * 0.01^2) = 1 / 0.0001 = 10,000
        let gamma_large_vol =
            strategy.derive_gamma(target_half_spread, kappa, sigma_large, max_position);
        assert!(
            (gamma_large_vol - 10000.0).abs() < 100.0,
            "Gamma with large vol: {}",
            gamma_large_vol
        );
    }

    #[test]
    fn test_glft_half_spread_formula() {
        // Test the correct half-spread formula: ψ = (1/γ) × ln(1 + γ/κ)
        let strategy = GLFTStrategy::new(0.01);

        let gamma = 100.0;
        let kappa = 50.0;

        // Expected: ψ = (1/100) * ln(1 + 100/50) = 0.01 * ln(3) = 0.01 * 1.0986 ≈ 0.011
        let half_spread = strategy.half_spread(gamma, kappa);
        let expected = (1.0 / gamma) * (1.0 + gamma / kappa).ln();

        assert!(
            (half_spread - expected).abs() < 1e-6,
            "Half-spread mismatch: got {}, expected {}",
            half_spread,
            expected
        );
    }

    #[test]
    fn test_glft_inventory_skew_formula() {
        // Test the infinite-horizon skew: skew = (q/Q_max) × γ × σ² / κ
        let strategy = GLFTStrategy::new(0.01);

        let inventory_ratio = 0.5; // 50% of max position
        let sigma = 0.001; // 0.1% per-second
        let gamma = 1000.0;
        let kappa = 100.0;

        // Expected: skew = 0.5 * 1000 * 0.001^2 / 100 = 0.5 * 1000 * 0.000001 / 100 = 0.000005
        let skew = strategy.inventory_skew(inventory_ratio, sigma, gamma, kappa);
        let expected = inventory_ratio * gamma * sigma.powi(2) / kappa;

        assert!(
            (skew - expected).abs() < 1e-10,
            "Skew mismatch: got {}, expected {}",
            skew,
            expected
        );
    }

    #[test]
    fn test_glft_toxic_regime_widens_spread() {
        let strategy = GLFTStrategy::new(0.01);
        // Use more decimals to avoid rounding issues
        let config = QuoteConfig {
            mid_price: 100.0,
            half_spread_bps: 100, // 1% spread
            decimals: 4,          // More precision to see spread difference
            sz_decimals: 4,
            min_notional: 10.0,
            max_position: 1.0,
        };

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

        // Toxic regime with high jump ratio (should multiply spread by ~1.5x)
        let toxic_params = MarketParams {
            sigma: 0.001,            // Same clean volatility
            sigma_total: 0.002,      // Higher total (includes jumps)
            sigma_effective: 0.0015, // Blended
            kappa: 50.0,
            arrival_intensity: 1.0,
            is_toxic_regime: true,
            jump_ratio: 4.5, // RV/BV = 4.5 -> multiplier = 4.5/2.0 = 2.25 (capped at 2.5)
            ..Default::default()
        };

        let (bid_normal, ask_normal) =
            strategy.calculate_quotes(&config, 0.0, 1.0, 0.5, &normal_params);
        let (bid_toxic, ask_toxic) =
            strategy.calculate_quotes(&config, 0.0, 1.0, 0.5, &toxic_params);

        let bid_normal = bid_normal.unwrap();
        let ask_normal = ask_normal.unwrap();
        let bid_toxic = bid_toxic.unwrap();
        let ask_toxic = ask_toxic.unwrap();

        // Toxic regime should have wider spreads
        let spread_normal = ask_normal.price - bid_normal.price;
        let spread_toxic = ask_toxic.price - bid_toxic.price;

        assert!(
            spread_toxic > spread_normal,
            "Toxic spread should be wider: normal={:.4}, toxic={:.4}",
            spread_normal,
            spread_toxic
        );
    }

    #[test]
    fn test_glft_toxic_regime_extra_skew() {
        let strategy = GLFTStrategy::new(0.01);
        let config = make_config_with_max_pos(100.0, 1.0);

        // Long position in toxic regime should have more aggressive skew
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

        let toxic_params = MarketParams {
            sigma: 0.001,
            sigma_total: 0.002,
            sigma_effective: 0.0015,
            kappa: 50.0,
            arrival_intensity: 1.0,
            is_toxic_regime: true,
            jump_ratio: 5.0,
            ..Default::default()
        };

        // With long position (0.5)
        let (bid_normal, ask_normal) =
            strategy.calculate_quotes(&config, 0.5, 1.0, 0.5, &normal_params);
        let (bid_toxic, ask_toxic) =
            strategy.calculate_quotes(&config, 0.5, 1.0, 0.5, &toxic_params);

        let bid_normal = bid_normal.unwrap();
        let _ask_normal = ask_normal.unwrap();
        let bid_toxic = bid_toxic.unwrap();
        let _ask_toxic = ask_toxic.unwrap();

        // In toxic regime with long position:
        // - Bid should be even lower (more skew away from buying)
        // - Ask should be lower (more aggressive selling)
        assert!(
            bid_toxic.price <= bid_normal.price,
            "Toxic bid should be lower: normal={:.4}, toxic={:.4}",
            bid_normal.price,
            bid_toxic.price
        );
    }
}
