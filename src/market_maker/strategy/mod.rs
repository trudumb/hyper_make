//! Quoting strategies for the market maker.
//!
//! This module provides different quoting strategies that implement the
//! `QuotingStrategy` trait.

mod glft;
mod ladder_strat;
mod market_params;
mod params;
mod risk_config;
mod simple;

pub use glft::*;
pub use ladder_strat::*;
pub use market_params::*;
pub use params::*;
pub use risk_config::*;
pub use simple::*;

use crate::{round_to_significant_and_decimal, truncate_float, EPSILON};

use super::config::{Quote, QuoteConfig};
use super::quoting::Ladder;

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

/// Shared helper: Calculate size with position-aware limits.
pub(crate) fn calculate_position_limited_size(
    position: f64,
    max_position: f64,
    target_liquidity: f64,
    sz_decimals: u32,
) -> (f64, f64) {
    // Buy size: how much more can we buy before hitting max_position
    let buy_size_raw = (max_position - position).min(target_liquidity).max(0.0);
    // Sell size: how much more can we sell before hitting -max_position
    let sell_size_raw = (max_position + position).min(target_liquidity).max(0.0);

    // Truncate to sz_decimals (floor, not round, to be conservative)
    let buy_size = truncate_float(buy_size_raw, sz_decimals, false);
    let sell_size = truncate_float(sell_size_raw, sz_decimals, false);

    (buy_size, sell_size)
}

/// Shared helper: Build quotes checking minimum notional.
pub(crate) fn build_quotes_with_min_notional(
    bid_price: f64,
    ask_price: f64,
    buy_size: f64,
    sell_size: f64,
    min_notional: f64,
) -> (Option<Quote>, Option<Quote>) {
    let bid = if buy_size > EPSILON {
        let quote = Quote::new(bid_price, buy_size);
        if quote.notional() >= min_notional {
            Some(quote)
        } else {
            None
        }
    } else {
        None
    };

    let ask = if sell_size > EPSILON {
        let quote = Quote::new(ask_price, sell_size);
        if quote.notional() >= min_notional {
            Some(quote)
        } else {
            None
        }
    } else {
        None
    };

    (bid, ask)
}

/// Shared helper: Ensure bid < ask after rounding.
pub(crate) fn ensure_bid_less_than_ask(mut bid_price: f64, ask_price: f64, decimals: u32) -> f64 {
    if bid_price >= ask_price {
        let tick = 10f64.powi(-(decimals as i32));
        bid_price -= tick;
    }
    bid_price
}

/// Shared helper: Round price to exchange constraints.
pub(crate) fn round_price_for_exchange(price: f64, decimals: u32) -> f64 {
    round_to_significant_and_decimal(price, 5, decimals)
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

        // GLFT spread at γ=0.5, κ=100: δ = (1/0.5) * ln(1 + 0.5/100) ≈ 0.01
        // So spread ≈ 1% of mid = 1.0 each side
        assert!(bid.price < 100.0);
        assert!(ask.price > 100.0);

        // Spread should be roughly symmetric (small rounding differences)
        let bid_spread = 100.0 - bid.price;
        let ask_spread = ask.price - 100.0;
        assert!((bid_spread - ask_spread).abs() < 0.1);
    }

    #[test]
    fn test_glft_positive_inventory_skew() {
        let strategy = GLFTStrategy::new(0.3);
        let config = make_config(100.0);
        let market_params = MarketParams {
            sigma: 0.0001,
            sigma_total: 0.0001,
            sigma_effective: 0.0001,
            kappa: 100.0,
            arrival_intensity: 1.0,
            is_toxic_regime: false,
            jump_ratio: 1.0,
            microprice: 100.0,
            ..Default::default()
        };

        // With positive position, both quotes should be skewed down
        // (lower bid to not buy more, lower ask to sell faster)
        let (bid_pos, ask_pos) = strategy.calculate_quotes(&config, 0.5, 1.0, 0.5, &market_params);
        let (bid_zero, ask_zero) =
            strategy.calculate_quotes(&config, 0.0, 1.0, 0.5, &market_params);

        let bid_pos = bid_pos.unwrap();
        let ask_pos = ask_pos.unwrap();
        let bid_zero = bid_zero.unwrap();
        let ask_zero = ask_zero.unwrap();

        // With positive inventory, prices should be lower (skewed towards selling)
        assert!(bid_pos.price <= bid_zero.price);
        assert!(ask_pos.price <= ask_zero.price);
    }

    #[test]
    fn test_glft_negative_inventory_skew() {
        let strategy = GLFTStrategy::new(0.3);
        let config = make_config(100.0);
        let market_params = MarketParams {
            sigma: 0.0001,
            sigma_total: 0.0001,
            sigma_effective: 0.0001,
            kappa: 100.0,
            arrival_intensity: 1.0,
            is_toxic_regime: false,
            jump_ratio: 1.0,
            microprice: 100.0,
            ..Default::default()
        };

        // With negative position, both quotes should be skewed up
        let (bid_neg, ask_neg) = strategy.calculate_quotes(&config, -0.5, 1.0, 0.5, &market_params);
        let (bid_zero, ask_zero) =
            strategy.calculate_quotes(&config, 0.0, 1.0, 0.5, &market_params);

        let bid_neg = bid_neg.unwrap();
        let ask_neg = ask_neg.unwrap();
        let bid_zero = bid_zero.unwrap();
        let ask_zero = ask_zero.unwrap();

        // With negative inventory, prices should be higher (skewed towards buying)
        assert!(bid_neg.price >= bid_zero.price);
        assert!(ask_neg.price >= ask_zero.price);
    }

    #[test]
    fn test_glft_minimum_spread_floor() {
        let mut config = RiskConfig::default();
        config.gamma_base = 0.01; // Very low gamma = tight spread
        config.min_spread_floor = 0.001; // 10 bps minimum

        let strategy = GLFTStrategy::with_config(config);
        let quote_config = make_config(100.0);
        let market_params = MarketParams {
            sigma: 0.00001, // Very low volatility
            sigma_total: 0.00001,
            sigma_effective: 0.00001,
            kappa: 1000.0, // High kappa = tight spread
            arrival_intensity: 1.0,
            is_toxic_regime: false,
            jump_ratio: 1.0,
            microprice: 100.0,
            ..Default::default()
        };

        let (bid, ask) = strategy.calculate_quotes(&quote_config, 0.0, 1.0, 0.5, &market_params);

        let bid = bid.unwrap();
        let ask = ask.unwrap();

        // Spread should be at least 10 bps (the floor)
        let spread = ask.price - bid.price;
        let spread_bps = spread / 100.0 * 10000.0;
        assert!(spread_bps >= 9.0); // Allow for rounding
    }

    #[test]
    fn test_glft_market_driven_spread() {
        // Test that spread responds to market conditions (kappa)
        let strategy = GLFTStrategy::new(0.5);
        let config = make_config_with_decimals(100.0, 4);

        // Deep book (high kappa) - should have tighter spread
        let deep_book = MarketParams {
            kappa: 200.0,
            kappa_bid: 200.0, // Match kappa for symmetric test
            kappa_ask: 200.0,
            microprice: 100.0, // Must match mid_price for fair quoting
            ..Default::default()
        };

        // Thin book (low kappa) - should have wider spread
        let thin_book = MarketParams {
            kappa: 20.0,
            kappa_bid: 20.0, // Match kappa for symmetric test
            kappa_ask: 20.0,
            microprice: 100.0, // Must match mid_price for fair quoting
            ..Default::default()
        };

        let (bid_deep, ask_deep) = strategy.calculate_quotes(&config, 0.0, 1.0, 0.5, &deep_book);
        let (bid_thin, ask_thin) = strategy.calculate_quotes(&config, 0.0, 1.0, 0.5, &thin_book);

        let spread_deep = ask_deep.unwrap().price - bid_deep.unwrap().price;
        let spread_thin = ask_thin.unwrap().price - bid_thin.unwrap().price;

        // Thin book should have wider spread (lower kappa -> wider spread)
        assert!(
            spread_thin > spread_deep,
            "Thin book should have wider spread: thin={:.4}, deep={:.4}",
            spread_thin,
            spread_deep
        );
    }

    #[test]
    fn test_symmetric_no_quote_below_min_notional() {
        let strategy = SymmetricStrategy::new(10);
        let config = QuoteConfig {
            mid_price: 100.0,
            decimals: 2,
            sz_decimals: 4,
            min_notional: 100.0, // High minimum
        };
        let market_params = MarketParams::default();

        // Size 0.5 at 100.0 = $50 notional, below $100 minimum
        let (bid, ask) = strategy.calculate_quotes(&config, 0.0, 1.0, 0.5, &market_params);
        assert!(bid.is_none());
        assert!(ask.is_none());
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
    fn test_glft_high_price_precision() {
        let strategy = GLFTStrategy::new(0.3);
        let config = make_config_with_decimals(50000.0, 1); // BTC-like
        let market_params = MarketParams {
            sigma: 0.0001,
            sigma_effective: 0.0001,
            kappa: 100.0,
            arrival_intensity: 1.0,
            microprice: 50000.0,
            ..Default::default()
        };

        let (bid, ask) = strategy.calculate_quotes(&config, 0.0, 1.0, 0.0005, &market_params);

        let bid = bid.unwrap();
        let ask = ask.unwrap();

        // Should have at most 1 decimal place
        let bid_str = format!("{:.1}", bid.price);
        let parsed: f64 = bid_str.parse().unwrap();
        assert!((parsed - bid.price).abs() < 0.01);

        // Bid should be less than ask
        assert!(bid.price < ask.price);
    }

    #[test]
    fn test_glft_max_holding_time_cap() {
        let config = RiskConfig {
            gamma_base: 0.3,
            max_holding_time: 10.0, // Very short cap
            ..Default::default()
        };

        let strategy = GLFTStrategy::with_config(config);
        let quote_config = make_config(100.0);

        // Very slow market = huge uncapped T
        let slow_market = MarketParams {
            sigma: 0.0001,
            sigma_effective: 0.0001,
            kappa: 100.0,
            arrival_intensity: 0.001, // Almost no activity
            microprice: 100.0,
            ..Default::default()
        };

        // With 50% utilization
        let (bid, ask) = strategy.calculate_quotes(&quote_config, 0.5, 1.0, 0.5, &slow_market);

        // Should still produce valid quotes (skew capped by max_holding_time)
        assert!(bid.is_some());
        assert!(ask.is_some());

        // Skew should be bounded, not astronomical
        let spread = ask.unwrap().price - bid.unwrap().price;
        assert!(spread < 10.0); // Less than 10% spread
    }

    #[test]
    fn test_glft_with_as_spread_adjustment() {
        let strategy = GLFTStrategy::new(0.3);
        let quote_config = make_config(100.0);

        let base_params = MarketParams {
            sigma: 0.0001,
            sigma_effective: 0.0001,
            kappa: 100.0,
            arrival_intensity: 1.0,
            microprice: 100.0,
            as_spread_adjustment: 0.0,
            as_warmed_up: true,
            ..Default::default()
        };

        let as_params = MarketParams {
            as_spread_adjustment: 0.001, // 10 bps AS adjustment
            ..base_params.clone()
        };

        let (bid_base, ask_base) =
            strategy.calculate_quotes(&quote_config, 0.0, 1.0, 0.5, &base_params);
        let (bid_as, ask_as) = strategy.calculate_quotes(&quote_config, 0.0, 1.0, 0.5, &as_params);

        let base_spread = ask_base.unwrap().price - bid_base.unwrap().price;
        let as_spread = ask_as.unwrap().price - bid_as.unwrap().price;

        // AS adjustment should widen spread
        assert!(as_spread > base_spread);
    }

    #[test]
    fn test_glft_cascade_size_factor() {
        let strategy = GLFTStrategy::new(0.3);
        let quote_config = make_config(100.0);

        let normal_params = MarketParams {
            sigma: 0.0001,
            sigma_effective: 0.0001,
            kappa: 100.0,
            arrival_intensity: 1.0,
            microprice: 100.0,
            cascade_size_factor: 1.0,
            ..Default::default()
        };

        let cascade_params = MarketParams {
            cascade_size_factor: 0.5, // 50% size reduction
            ..normal_params.clone()
        };

        let (bid_normal, ask_normal) =
            strategy.calculate_quotes(&quote_config, 0.0, 1.0, 0.5, &normal_params);
        let (bid_cascade, ask_cascade) =
            strategy.calculate_quotes(&quote_config, 0.0, 1.0, 0.5, &cascade_params);

        // Cascade should have smaller sizes
        assert!(bid_cascade.unwrap().size < bid_normal.unwrap().size);
        assert!(ask_cascade.unwrap().size < ask_normal.unwrap().size);
    }

    #[test]
    fn test_glft_should_pull_quotes() {
        let strategy = GLFTStrategy::new(0.3);
        let quote_config = make_config(100.0);

        let pull_params = MarketParams {
            sigma: 0.0001,
            sigma_effective: 0.0001,
            kappa: 100.0,
            arrival_intensity: 1.0,
            microprice: 100.0,
            should_pull_quotes: true,
            ..Default::default()
        };

        let (bid, ask) = strategy.calculate_quotes(&quote_config, 0.0, 1.0, 0.5, &pull_params);

        // Should return no quotes when pull flag is set
        assert!(bid.is_none());
        assert!(ask.is_none());
    }
}
