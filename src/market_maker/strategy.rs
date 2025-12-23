//! Quoting strategies for the market maker.

use crate::{round_to_significant_and_decimal, truncate_float, EPSILON};

use super::config::{Quote, QuoteConfig};

/// Parameters estimated from live market data.
#[derive(Debug, Clone, Copy)]
pub struct MarketParams {
    /// Estimated volatility (σ)
    pub sigma: f64,
    /// Estimated order flow intensity (κ)
    pub kappa: f64,
    /// Estimated time horizon (τ) in years
    pub tau: f64,
}

impl Default for MarketParams {
    fn default() -> Self {
        Self {
            sigma: 0.5,
            kappa: 1.5,
            tau: 0.0001, // ~1 hour in years
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
        let mut lower_price =
            round_to_significant_and_decimal(lower_price_raw, 5, config.decimals);
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
        let mut lower_price =
            round_to_significant_and_decimal(lower_price_raw, 5, config.decimals);
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
/// Uses stochastic control theory to derive optimal bid/ask quotes based on:
/// - Target spread (δ) - user-configured target half-spread
/// - Order flow intensity (κ) - fill probability decay rate (estimated from L2 book)
/// - Volatility (σ) - asset price volatility (estimated from trades)
/// - Time horizon (τ) - estimated from trade rate (faster markets → smaller τ)
///
/// γ (risk aversion) is calculated dynamically to achieve the target spread:
/// ```text
/// γ = κ / (exp(δ*κ) - 1)
/// ```
///
/// The optimal spread deltas are:
/// ```text
/// δ_bid = (1/κ) * ln(1 + κ/γ) + (q/Q_max) * γ * σ² * τ
/// δ_ask = (1/κ) * ln(1 + κ/γ) - (q/Q_max) * γ * σ² * τ
/// ```
#[derive(Debug, Clone)]
pub struct GLFTStrategy {
    /// Target half-spread (e.g., 0.005 = 0.5%)
    pub target_spread: f64,
    /// Minimum gamma floor (prevents extreme behavior)
    pub min_gamma: f64,
    /// Maximum gamma ceiling
    pub max_gamma: f64,
}

impl GLFTStrategy {
    /// Create a new GLFT strategy with the given target spread.
    /// γ is calculated dynamically from κ to achieve this target.
    /// τ is estimated from market trade rate via MarketParams.
    pub fn new(target_spread: f64) -> Self {
        Self {
            target_spread,
            min_gamma: 1.0,
            max_gamma: 1000.0,
        }
    }

    /// Calculate gamma to achieve target spread given kappa.
    /// Formula: γ = κ / (exp(δ*κ) - 1)
    fn calculate_gamma(&self, kappa: f64) -> f64 {
        let delta_kappa = self.target_spread * kappa;
        let gamma = if delta_kappa > 0.001 {
            kappa / (delta_kappa.exp() - 1.0)
        } else {
            // For very small δ*κ, use Taylor expansion: γ ≈ 1/δ
            1.0 / self.target_spread
        };
        gamma.clamp(self.min_gamma, self.max_gamma)
    }

    /// Calculate the base spread component from order flow intensity.
    /// This is the minimum spread needed to compensate for adverse selection.
    fn base_spread(&self, kappa: f64, gamma: f64) -> f64 {
        (1.0 / kappa) * (1.0 + kappa / gamma).ln()
    }

    /// Calculate the inventory adjustment component.
    /// Positive inventory -> positive adjustment (widen bid, tighten ask)
    fn inventory_adjustment(&self, inventory_ratio: f64, sigma: f64, gamma: f64, tau: f64) -> f64 {
        inventory_ratio * gamma * sigma.powi(2) * tau
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
        // Calculate inventory ratio: q / Q_max (normalized to [-1, 1])
        let inventory_ratio = if max_position > EPSILON {
            position / max_position
        } else {
            0.0
        };

        // Use estimated parameters from market data
        let sigma = market_params.sigma;
        let kappa = market_params.kappa;
        let tau = market_params.tau;

        // Calculate gamma dynamically to achieve target spread
        let gamma = self.calculate_gamma(kappa);

        // GLFT optimal spread components
        let base = self.base_spread(kappa, gamma);
        let inv_adj = self.inventory_adjustment(inventory_ratio, sigma, gamma, tau);

        // Optimal deltas (as fraction of price)
        // When long (inv_adj > 0): bid delta increases, ask delta decreases
        let bid_delta = base + inv_adj;
        let ask_delta = base - inv_adj;

        // Convert to absolute price offsets
        let bid_offset = config.mid_price * bid_delta;
        let ask_offset = config.mid_price * ask_delta;

        // Enforce minimum spread from config
        let min_offset = (config.mid_price * config.half_spread_bps as f64) / 10000.0;
        let bid_offset = bid_offset.max(min_offset);
        let ask_offset = ask_offset.max(min_offset);

        // Calculate raw prices
        let lower_price_raw = config.mid_price - bid_offset;
        let upper_price_raw = config.mid_price + ask_offset;

        // Round to exchange precision
        let mut lower_price =
            round_to_significant_and_decimal(lower_price_raw, 5, config.decimals);
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
        "GLFT"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(mid: f64) -> QuoteConfig {
        QuoteConfig {
            mid_price: mid,
            half_spread_bps: 10, // 0.1%
            decimals: 2,
            sz_decimals: 4,
            min_notional: 10.0,
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
        // target_spread=0.01 (1% half-spread)
        let strategy = GLFTStrategy::new(0.01);
        let config = make_config(100.0);
        // kappa=100.0 (high order flow), sigma=0.01 (low volatility), tau from trade rate
        let market_params = MarketParams {
            sigma: 0.01,
            kappa: 100.0,
            tau: 0.0001, // ~1 hour
        };

        // With zero inventory, bid and ask should be symmetric around mid
        let (bid, ask) = strategy.calculate_quotes(&config, 0.0, 1.0, 0.5, &market_params);

        let bid = bid.unwrap();
        let ask = ask.unwrap();

        // Both should exist and be roughly symmetric
        let bid_offset = config.mid_price - bid.price;
        let ask_offset = ask.price - config.mid_price;
        assert!((bid_offset - ask_offset).abs() < 0.1);
    }

    #[test]
    fn test_glft_long_inventory_skews() {
        // target_spread=0.01 (1% half-spread)
        let strategy = GLFTStrategy::new(0.01);
        let config = make_config(100.0);
        // Use larger tau to make inventory effect visible in test
        // In production, tau is estimated from trade rate (~0.0001)
        let market_params = MarketParams {
            sigma: 0.30,   // 30% annual vol (realistic for crypto)
            kappa: 100.0,
            tau: 0.01,     // larger tau to make effect visible
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
        assert!(bid_long.price < bid_neutral.price);
        assert!(ask_long.price < ask_neutral.price);
    }

    #[test]
    fn test_glft_short_inventory_skews() {
        // target_spread=0.01 (1% half-spread)
        let strategy = GLFTStrategy::new(0.01);
        let config = make_config(100.0);
        // Use larger tau to make inventory effect visible in test
        // In production, tau is estimated from trade rate (~0.0001)
        let market_params = MarketParams {
            sigma: 0.30,   // 30% annual vol (realistic for crypto)
            kappa: 100.0,
            tau: 0.01,     // larger tau to make effect visible
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
        assert!(bid_short.price > bid_neutral.price);
        assert!(ask_short.price > ask_neutral.price);
    }

    #[test]
    fn test_glft_respects_min_spread() {
        // Even with tiny target_spread, min spread from config should be enforced
        let strategy = GLFTStrategy::new(0.0001); // 0.01% target
        let config = make_config(100.0);
        let market_params = MarketParams {
            sigma: 0.001,
            kappa: 1000.0,
            tau: 0.0001,
        };

        let (bid, ask) = strategy.calculate_quotes(&config, 0.0, 1.0, 0.5, &market_params);

        let bid = bid.unwrap();
        let ask = ask.unwrap();

        // Spread should be at least 2 * half_spread_bps
        let spread = ask.price - bid.price;
        let min_spread = 100.0 * 10.0 / 10000.0 * 2.0; // 0.2
        assert!(spread >= min_spread - 0.01);
    }

    #[test]
    fn test_glft_adaptive_gamma() {
        // Test that gamma is calculated correctly from target spread and kappa
        let strategy = GLFTStrategy::new(0.005); // 0.5% target spread

        // With different kappa values, gamma should adjust to maintain target spread
        // γ = κ / (exp(δ*κ) - 1)

        // kappa = 1.0: gamma = 1.0 / (exp(0.005) - 1) ≈ 199
        let gamma_low_kappa = strategy.calculate_gamma(1.0);
        assert!(gamma_low_kappa > 100.0 && gamma_low_kappa < 300.0);

        // kappa = 100.0: gamma = 100 / (exp(0.5) - 1) ≈ 154
        let gamma_high_kappa = strategy.calculate_gamma(100.0);
        assert!(gamma_high_kappa > 100.0 && gamma_high_kappa < 200.0);
    }
}
