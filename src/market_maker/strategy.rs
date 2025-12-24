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

/// GLFT (Guéant-Lehalle-Fernandez-Tapia) optimal market making strategy.
///
/// Implements the **infinite-horizon** GLFT model from stochastic control theory.
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
/// ## Parameters:
/// - γ (gamma): risk aversion - fixed parameter, not derived
/// - σ (sigma): per-second volatility (NOT annualized)
/// - κ (kappa): order book depth decay (from L2 book)
/// - λ (lambda): arrival intensity (volume ticks per second)
#[derive(Debug, Clone)]
pub struct GLFTStrategy {
    /// Risk aversion parameter (gamma) - controls spread and inventory sensitivity
    /// Typical values: 0.1 (aggressive) to 2.0 (conservative)
    pub risk_aversion: f64,
    /// Minimum gamma floor (safety bound)
    pub min_gamma: f64,
    /// Maximum gamma ceiling (safety bound)
    pub max_gamma: f64,
}

impl GLFTStrategy {
    /// Create a new GLFT strategy with fixed risk aversion.
    /// The market (kappa, sigma) determines the spread, not a target spread.
    pub fn new(risk_aversion: f64) -> Self {
        Self {
            risk_aversion: risk_aversion.clamp(0.01, 100.0),
            min_gamma: 0.01,
            max_gamma: 100.0,
        }
    }

    /// Correct GLFT half-spread formula: δ = (1/γ) × ln(1 + γ/κ)
    ///
    /// This is market-driven: when κ drops (thin book), spread widens automatically.
    /// When κ rises (deep book), spread tightens.
    fn half_spread(&self, gamma: f64, kappa: f64) -> f64 {
        let ratio = gamma / kappa;
        if ratio > 1e-9 {
            (1.0 / gamma) * (1.0 + ratio).ln()
        } else {
            // When γ/κ → 0, use Taylor expansion: ln(1+x) ≈ x
            // δ ≈ (1/γ) × (γ/κ) = 1/κ
            1.0 / kappa
        }
    }

    /// Correct GLFT inventory skew: skew = (q/Q_max) × γ × σ² × T
    ///
    /// Where T = 1/λ (time horizon from arrival intensity).
    /// Positive inventory → positive skew → wider bid, tighter ask (encourage selling)
    fn inventory_skew(
        &self,
        inventory_ratio: f64,
        sigma: f64,
        gamma: f64,
        time_horizon: f64,
    ) -> f64 {
        inventory_ratio * gamma * sigma.powi(2) * time_horizon
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
        // === 1. USE FIXED GAMMA (RISK AVERSION) ===
        // Gamma is a personality parameter, not derived from target spread
        let gamma = self.risk_aversion.clamp(self.min_gamma, self.max_gamma);
        let kappa = market_params.kappa;

        // Time horizon from arrival intensity: T = 1/λ
        let time_horizon = 1.0 / market_params.arrival_intensity.max(0.01);

        // === 2. CORRECT GLFT HALF-SPREAD: δ = (1/γ) × ln(1 + γ/κ) ===
        // This is market-driven: thin book (low κ) → wider spread
        let half_spread = self.half_spread(gamma, kappa);

        // === 3. USE SIGMA_EFFECTIVE FOR INVENTORY SKEW ===
        // sigma_effective blends clean and total based on jump regime
        let sigma_for_skew = market_params.sigma_effective;

        // Calculate inventory ratio: q / Q_max (normalized to [-1, 1])
        let inventory_ratio = if max_position > EPSILON {
            (position / max_position).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // Correct inventory skew: (q/Q_max) × γ × σ² × T
        let base_skew = self.inventory_skew(inventory_ratio, sigma_for_skew, gamma, time_horizon);

        // === 4. TOXIC REGIME SPREAD WIDENING ===
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

        // === 5. FALLING KNIFE PROTECTION (protect bids during crashes) ===
        let mut bid_protection = 0.0;
        if market_params.falling_knife_score > 0.5 {
            bid_protection = market_params.falling_knife_score * half_spread * 0.5;
            if inventory_ratio > 0.0 {
                bid_protection *= 1.0 + inventory_ratio;
            }
        }

        // === 6. RISING KNIFE PROTECTION (protect asks during pumps) ===
        let mut ask_protection = 0.0;
        if market_params.rising_knife_score > 0.5 {
            ask_protection = market_params.rising_knife_score * half_spread * 0.5;
            if inventory_ratio < 0.0 {
                ask_protection *= 1.0 + inventory_ratio.abs();
            }
        }

        // === 7. FLOW IMBALANCE ADJUSTMENT ===
        let flow_adjustment = market_params.flow_imbalance * half_spread * 0.2;

        // === 8. ADDITIONAL TOXIC SKEW ===
        let additional_skew = if market_params.is_toxic_regime {
            let toxicity_excess = (market_params.jump_ratio - 1.0).max(0.0) * 0.1;
            inventory_ratio * toxicity_excess * half_spread
        } else {
            0.0
        };
        let skew = base_skew + additional_skew;

        // === 9. COMBINE ALL ADJUSTMENTS ===
        let bid_delta =
            (half_spread + skew + bid_protection - flow_adjustment) * toxicity_multiplier;
        let ask_delta = ((half_spread - skew + ask_protection - flow_adjustment).max(0.0))
            * toxicity_multiplier;

        debug!(
            inv_ratio = %format!("{:.4}", inventory_ratio),
            gamma = %format!("{:.4}", gamma),
            kappa = %format!("{:.2}", kappa),
            time_horizon = %format!("{:.2}", time_horizon),
            half_spread_bps = %format!("{:.1}", half_spread * 10000.0),
            skew_bps = %format!("{:.4}", skew * 10000.0),
            bid_protection = %format!("{:.6}", bid_protection),
            ask_protection = %format!("{:.6}", ask_protection),
            flow_adj = %format!("{:.6}", flow_adjustment),
            bid_delta_bps = %format!("{:.1}", bid_delta * 10000.0),
            ask_delta_bps = %format!("{:.1}", ask_delta * 10000.0),
            is_toxic = market_params.is_toxic_regime,
            falling_knife = %format!("{:.2}", market_params.falling_knife_score),
            rising_knife = %format!("{:.2}", market_params.rising_knife_score),
            "GLFT spread components (market-driven)"
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
            sigma_effective = %format!("{:.6}", sigma_for_skew),
            kappa = %format!("{:.2}", kappa),
            gamma = %format!("{:.4}", gamma),
            jump_ratio = %format!("{:.2}", market_params.jump_ratio),
            momentum_bps = %format!("{:.1}", market_params.momentum_bps),
            flow = %format!("{:.2}", market_params.flow_imbalance),
            bid_final = lower_price,
            ask_final = upper_price,
            spread_bps = %format!("{:.1}", (upper_price - lower_price) / config.mid_price * 10000.0),
            "GLFT prices (market-driven spread)"
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
        // Test the correct half-spread formula: δ = (1/γ) × ln(1 + γ/κ)
        let strategy = GLFTStrategy::new(0.5);

        let gamma = 0.5;
        let kappa = 100.0;

        // Expected: δ = (1/0.5) * ln(1 + 0.5/100) = 2 * ln(1.005) ≈ 2 * 0.00499 ≈ 0.00998
        let half_spread = strategy.half_spread(gamma, kappa);
        let expected = (1.0 / gamma) * (1.0 + gamma / kappa).ln();

        assert!(
            (half_spread - expected).abs() < 1e-6,
            "Half-spread mismatch: got {}, expected {}",
            half_spread,
            expected
        );

        // Verify the numerical example from GLFT_CORRECTIONS.md
        assert!(
            (half_spread - 0.00998).abs() < 0.0001,
            "Half-spread should be ~0.998%: got {}",
            half_spread
        );
    }

    #[test]
    fn test_glft_inventory_skew_formula() {
        // Test the correct skew formula: skew = (q/Q_max) × γ × σ² × T
        let strategy = GLFTStrategy::new(0.5);

        let inventory_ratio = 0.5; // 50% of max position
        let sigma = 0.01; // 1% per-second volatility
        let gamma = 0.5;
        let time_horizon = 2.0; // T = 2 seconds

        // Expected: skew = 0.5 * 0.5 * 0.01^2 * 2 = 0.5 * 0.5 * 0.0001 * 2 = 0.00005
        let skew = strategy.inventory_skew(inventory_ratio, sigma, gamma, time_horizon);
        let expected = inventory_ratio * gamma * sigma.powi(2) * time_horizon;

        assert!(
            (skew - expected).abs() < 1e-10,
            "Skew mismatch: got {}, expected {}",
            skew,
            expected
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
            ..Default::default()
        };

        // Thin book (low kappa) - should have wider spread
        let thin_book = MarketParams {
            kappa: 20.0,
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
    fn test_glft_toxic_regime_widens_spread() {
        let strategy = GLFTStrategy::new(0.5);
        let config = make_config_with_decimals(100.0, 4);

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

        let (bid_normal, ask_normal) =
            strategy.calculate_quotes(&config, 0.0, 1.0, 0.5, &normal_params);
        let (bid_toxic, ask_toxic) =
            strategy.calculate_quotes(&config, 0.0, 1.0, 0.5, &toxic_params);

        let spread_normal = ask_normal.unwrap().price - bid_normal.unwrap().price;
        let spread_toxic = ask_toxic.unwrap().price - bid_toxic.unwrap().price;

        assert!(
            spread_toxic > spread_normal,
            "Toxic spread should be wider: normal={:.4}, toxic={:.4}",
            spread_normal,
            spread_toxic
        );
    }

    #[test]
    fn test_glft_toxic_regime_extra_skew() {
        let strategy = GLFTStrategy::new(0.5);
        let config = make_config(100.0);

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
        // Bid should be even lower (more skew away from buying)
        assert!(
            bid_toxic.price <= bid_normal.price,
            "Toxic bid should be lower: normal={:.4}, toxic={:.4}",
            bid_normal.price,
            bid_toxic.price
        );
    }
}
