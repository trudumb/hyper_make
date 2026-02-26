//! Simple quoting strategies.

use crate::market_maker::config::{Quote, QuoteConfig};

use super::{
    build_quotes_with_min_notional, calculate_position_limited_size, ensure_bid_less_than_ask,
    round_price_for_exchange, MarketParams, QuotingStrategy,
};

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
        let lower_price = round_price_for_exchange(lower_price_raw, config.decimals);
        let upper_price = round_price_for_exchange(upper_price_raw, config.decimals);

        // Ensure bid < ask (rounding may cause them to be equal for high-value assets)
        let lower_price = ensure_bid_less_than_ask(lower_price, upper_price, config.decimals);

        // Calculate sizes based on position limits
        let (buy_size, sell_size) = calculate_position_limited_size(
            position,
            max_position,
            target_liquidity,
            config.sz_decimals,
        );

        // Build quotes, checking minimum notional
        build_quotes_with_min_notional(
            lower_price,
            upper_price,
            buy_size,
            sell_size,
            config.min_notional,
        )
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

        // Round to 5 significant figures
        let lower_price = round_price_for_exchange(lower_price_raw, config.decimals);
        let upper_price = round_price_for_exchange(upper_price_raw, config.decimals);

        // Ensure bid < ask
        let lower_price = ensure_bid_less_than_ask(lower_price, upper_price, config.decimals);

        // Calculate sizes based on position limits
        let (buy_size, sell_size) = calculate_position_limited_size(
            position,
            max_position,
            target_liquidity,
            config.sz_decimals,
        );

        // Build quotes, checking minimum notional
        build_quotes_with_min_notional(
            lower_price,
            upper_price,
            buy_size,
            sell_size,
            config.min_notional,
        )
    }

    fn name(&self) -> &'static str {
        "InventoryAware"
    }
}
