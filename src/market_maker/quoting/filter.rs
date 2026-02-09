//! Quote filtering for position limits and reduce-only mode.
//!
//! Extracts the duplicated reduce-only logic from mod.rs into a unified filter.
//!
//! # Phase 3 Fix: Exchange-Aware Reduce-Only
//!
//! The original reduce-only logic only considered local position limits.
//! When position exceeded limits, it would allow orders to reduce exposure,
//! but those orders could still be rejected by the exchange if there was
//! no available capacity.
//!
//! The enhanced logic now:
//! 1. Checks local position limits (original behavior)
//! 2. Checks exchange available capacity
//! 3. Signals when escalation to market orders may be needed

use crate::market_maker::config::Quote;
use crate::market_maker::infra::ExchangePositionLimits;
use tracing::warn;

/// Reason for entering reduce-only mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReduceOnlyReason {
    /// Approaching liquidation (buffer ratio below threshold).
    /// HIGHEST PRIORITY - based on exchange-provided liquidation price.
    ApproachingLiquidation,
    /// Margin utilization exceeds threshold (capital-efficient approach).
    OverMarginUtilization,
    /// Position value exceeds max_position_value limit.
    OverValueLimit,
    /// Position exceeds max_position limit (fallback/override).
    OverPositionLimit,
    /// Position exceeds limit but is underwater - widen spreads instead of forcing exit.
    /// This prevents selling at a loss when not at liquidation risk.
    UnderwaterWidenOnly,
}

/// Result of reduce-only filtering.
#[derive(Debug, Clone)]
pub struct ReduceOnlyResult {
    /// Whether quotes were filtered.
    pub was_filtered: bool,
    /// Reason for filtering (if any).
    pub reason: Option<ReduceOnlyReason>,
    /// Side that was filtered (true = bid, false = ask).
    pub filtered_bids: bool,
    /// Side that was filtered (true = ask, false = bid).
    pub filtered_asks: bool,
    /// Whether escalation to market orders is needed (Phase 3 Fix).
    /// True when reduce-only mode is active but exchange has no capacity
    /// to place reducing limit orders.
    pub needs_escalation: bool,
    /// Close urgency: 0.0 = at threshold, 1.0 = maximum urgency.
    /// Used to tighten the closing side spread to attract fills that reduce position.
    pub close_urgency: f64,
}

impl ReduceOnlyResult {
    fn no_filtering() -> Self {
        Self {
            was_filtered: false,
            reason: None,
            filtered_bids: false,
            filtered_asks: false,
            needs_escalation: false,
            close_urgency: 0.0,
        }
    }

    fn filtered_bids(reason: ReduceOnlyReason) -> Self {
        Self {
            was_filtered: true,
            reason: Some(reason),
            filtered_bids: true,
            filtered_asks: false,
            needs_escalation: false,
            close_urgency: 0.0,
        }
    }

    fn filtered_asks(reason: ReduceOnlyReason) -> Self {
        Self {
            was_filtered: true,
            reason: Some(reason),
            filtered_bids: false,
            filtered_asks: true,
            needs_escalation: false,
            close_urgency: 0.0,
        }
    }

    fn filtered_bids_needs_escalation(reason: ReduceOnlyReason) -> Self {
        Self {
            was_filtered: true,
            reason: Some(reason),
            filtered_bids: true,
            filtered_asks: false,
            needs_escalation: true,
            close_urgency: 0.0,
        }
    }

    fn filtered_asks_needs_escalation(reason: ReduceOnlyReason) -> Self {
        Self {
            was_filtered: true,
            reason: Some(reason),
            filtered_bids: false,
            filtered_asks: true,
            needs_escalation: true,
            close_urgency: 0.0,
        }
    }

    /// Compute close urgency based on how far we are over the limit.
    ///
    /// Factors:
    /// 1. Excess position ratio: (|position| - max_position) / max_position
    /// 2. Unrealized PnL (more urgency when underwater)
    ///
    /// Result is clamped to [0.0, 1.0].
    pub fn compute_urgency(
        position_abs: f64,
        max_position: f64,
        unrealized_pnl: f64,
    ) -> f64 {
        if position_abs <= max_position {
            return 0.0;
        }

        // Base urgency from excess position
        let excess_ratio = (position_abs - max_position) / max_position.max(1e-9);
        let base_urgency = excess_ratio.min(1.0);

        // Increase urgency when underwater (losing money on position)
        let pnl_factor = if unrealized_pnl < 0.0 {
            1.0 + (-unrealized_pnl / 100.0).min(0.5) // Up to 50% boost when underwater
        } else {
            1.0
        };

        (base_urgency * pnl_factor).clamp(0.0, 1.0)
    }
}

/// Margin utilization threshold for reduce-only mode (80%).
/// When margin_used/account_value exceeds this, reduce-only activates.
const MARGIN_UTILIZATION_THRESHOLD: f64 = 0.8;

/// Default threshold for liquidation proximity trigger.
/// The buffer_ratio formula: distance / (distance + maintenance_margin_rate)
/// - 0.5 = distance equals maintenance margin rate (too aggressive for high leverage)
/// - 0.3 = distance is ~43% of maintenance rate (better for 10x+ leverage)
/// - 0.2 = distance is ~25% of maintenance rate (conservative)
///
/// For 10x leverage (5% maintenance), thresholds translate to:
/// - 0.5 → triggers at 5% price distance (too early)
/// - 0.3 → triggers at ~2.1% price distance (reasonable)
/// - 0.2 → triggers at ~1.25% price distance (conservative)
pub const DEFAULT_LIQUIDATION_TRIGGER_THRESHOLD: f64 = 0.3;

/// Configuration for reduce-only filtering.
pub struct ReduceOnlyConfig {
    /// Current position.
    pub position: f64,
    /// Maximum allowed position (fallback/override only).
    pub max_position: f64,
    /// Current mid price (for value calculation).
    pub mid_price: f64,
    /// Maximum allowed position value.
    pub max_position_value: f64,
    /// Asset name (for logging).
    pub asset: String,
    /// Current margin used (USD).
    pub margin_used: f64,
    /// Current account value (USD).
    pub account_value: f64,

    // === Liquidation Proximity Fields ===
    /// Exchange-provided liquidation price (if available).
    /// Used for dynamic reduce-only based on proximity to liquidation.
    pub liquidation_price: Option<f64>,
    /// Liquidation buffer ratio (0.0 = at liquidation, 1.0 = very safe).
    /// Calculated as: distance / (distance + maintenance_margin_rate)
    /// None if no position or liquidation price not available.
    pub liquidation_buffer_ratio: Option<f64>,
    /// Threshold below which reduce-only triggers (default 0.5).
    /// Higher = more conservative (triggers earlier).
    pub liquidation_trigger_threshold: f64,

    // === Underwater Position Protection ===
    /// Unrealized P&L (USD). Negative = underwater.
    /// Used to prevent forcing sales at a loss when not at liquidation risk.
    pub unrealized_pnl: f64,
}

/// Quote filtering utilities.
///
/// Provides reduce-only filtering for both single quotes and ladders.
pub struct QuoteFilter;

impl QuoteFilter {
    /// Apply reduce-only logic to ladder quotes.
    ///
    /// When position exceeds limits or margin utilization is too high,
    /// clears the side that would increase exposure.
    /// Returns a result describing what was filtered.
    ///
    /// Priority (highest to lowest):
    /// 1. Approaching liquidation (buffer_ratio < threshold) - DYNAMIC, exchange-derived
    ///    → ALWAYS force reduce, even at a loss (survival > P&L)
    /// 2. Margin utilization > 80% (capital constraint)
    /// 3. Position value exceeds max_position_value
    /// 4. Position exceeds max_position (fallback/override)
    ///
    /// Special case: Underwater position protection
    /// When position exceeds limits but is underwater (unrealized_pnl < 0) AND
    /// not approaching liquidation, we only clear the side that would ADD to
    /// the position (don't buy more when long underwater), but keep the reducing
    /// side passive to wait for better exit prices. This is the UnderwaterWidenOnly mode.
    pub fn apply_reduce_only_ladder(
        bids: &mut Vec<Quote>,
        asks: &mut Vec<Quote>,
        config: &ReduceOnlyConfig,
    ) -> ReduceOnlyResult {
        let position = config.position;
        let position_value = position.abs() * config.mid_price;

        // Check margin utilization (capital-efficient approach)
        let margin_utilization = if config.account_value > 0.0 {
            config.margin_used / config.account_value
        } else {
            0.0
        };

        // === Priority 1: Liquidation Proximity (HIGHEST PRIORITY) ===
        // This is the dynamic, exchange-derived trigger that replaces static max_position
        let approaching_liquidation = config
            .liquidation_buffer_ratio
            .map(|ratio| ratio < config.liquidation_trigger_threshold)
            .unwrap_or(false);

        // === Priority 2-4: Fallback triggers ===
        let over_margin_limit = margin_utilization > MARGIN_UTILIZATION_THRESHOLD;
        let over_value_limit = position_value > config.max_position_value;
        let over_position_limit = position.abs() > config.max_position;

        // Check if any trigger is active
        if !approaching_liquidation
            && !over_margin_limit
            && !over_position_limit
            && !over_value_limit
        {
            return ReduceOnlyResult::no_filtering();
        }

        // Check if position is underwater (unrealized P&L is negative)
        let is_underwater = config.unrealized_pnl < 0.0;

        // === UNDERWATER POSITION PROTECTION ===
        // When position exceeds limit but is underwater AND not at liquidation risk,
        // DON'T force reduce (which would realize a loss).
        // Instead: clear the side that would ADD to the losing position,
        // but keep the reducing side passive to wait for better exit prices.
        //
        // Priority 1 (liquidation) always overrides this - survival > P&L.
        if is_underwater && !approaching_liquidation && !over_margin_limit {
            if position > 0.0 {
                // Long and underwater: clear bids (don't add to losing position)
                // Keep asks for potential exit at better prices
                bids.clear();
                warn!(
                    asset = %config.asset,
                    position = %format!("{:.6}", position),
                    unrealized_pnl = %format!("${:.2}", config.unrealized_pnl),
                    position_value = %format!("${:.2}", position_value),
                    max_position_value = %format!("${:.2}", config.max_position_value),
                    "Position over limit but UNDERWATER - filtering bids only, not forcing exit"
                );
                return ReduceOnlyResult {
                    was_filtered: true,
                    reason: Some(ReduceOnlyReason::UnderwaterWidenOnly),
                    filtered_bids: true,
                    filtered_asks: false,
                    needs_escalation: false,
                    close_urgency: 0.0,
                };
            } else {
                // Short and underwater: clear asks (don't add to losing position)
                // Keep bids for potential exit at better prices
                asks.clear();
                warn!(
                    asset = %config.asset,
                    position = %format!("{:.6}", position),
                    unrealized_pnl = %format!("${:.2}", config.unrealized_pnl),
                    position_value = %format!("${:.2}", position_value),
                    max_position_value = %format!("${:.2}", config.max_position_value),
                    "Position over limit but UNDERWATER - filtering asks only, not forcing exit"
                );
                return ReduceOnlyResult {
                    was_filtered: true,
                    reason: Some(ReduceOnlyReason::UnderwaterWidenOnly),
                    filtered_bids: false,
                    filtered_asks: true,
                    needs_escalation: false,
                    close_urgency: 0.0,
                };
            }
        }

        // === STANDARD REDUCE-ONLY BEHAVIOR ===
        // Position is over limit and either:
        // - Approaching liquidation (force reduce regardless of P&L)
        // - Over margin limit (capital constraint)
        // - Not underwater (can reduce at profit or break-even)

        // Determine primary reason (priority order)
        let reason = if approaching_liquidation {
            ReduceOnlyReason::ApproachingLiquidation
        } else if over_margin_limit {
            ReduceOnlyReason::OverMarginUtilization
        } else if over_value_limit {
            ReduceOnlyReason::OverValueLimit
        } else {
            ReduceOnlyReason::OverPositionLimit
        };

        if position > 0.0 {
            // Long position over max: only allow sells (no bids)
            bids.clear();
            Self::log_reduce_only(position, config, reason, true, margin_utilization);
            ReduceOnlyResult::filtered_bids(reason)
        } else {
            // Short position over max: only allow buys (no asks)
            asks.clear();
            Self::log_reduce_only(position, config, reason, false, margin_utilization);
            ReduceOnlyResult::filtered_asks(reason)
        }
    }

    /// Apply reduce-only logic to single quotes.
    ///
    /// When position exceeds limits or margin utilization is too high,
    /// clears the side that would increase exposure.
    /// Returns a result describing what was filtered.
    ///
    /// Priority (highest to lowest):
    /// 1. Approaching liquidation (buffer_ratio < threshold) - DYNAMIC, exchange-derived
    ///    → ALWAYS force reduce, even at a loss (survival > P&L)
    /// 2. Margin utilization > 80% (capital constraint)
    /// 3. Position value exceeds max_position_value
    /// 4. Position exceeds max_position (fallback/override)
    ///
    /// Special case: Underwater position protection (same as ladder version)
    pub fn apply_reduce_only_single(
        bid: &mut Option<Quote>,
        ask: &mut Option<Quote>,
        config: &ReduceOnlyConfig,
    ) -> ReduceOnlyResult {
        let position = config.position;
        let position_value = position.abs() * config.mid_price;

        // Check margin utilization (capital-efficient approach)
        let margin_utilization = if config.account_value > 0.0 {
            config.margin_used / config.account_value
        } else {
            0.0
        };

        // === Priority 1: Liquidation Proximity (HIGHEST PRIORITY) ===
        let approaching_liquidation = config
            .liquidation_buffer_ratio
            .map(|ratio| ratio < config.liquidation_trigger_threshold)
            .unwrap_or(false);

        // === Priority 2-4: Fallback triggers ===
        let over_margin_limit = margin_utilization > MARGIN_UTILIZATION_THRESHOLD;
        let over_value_limit = position_value > config.max_position_value;
        let over_position_limit = position.abs() > config.max_position;

        if !approaching_liquidation
            && !over_margin_limit
            && !over_position_limit
            && !over_value_limit
        {
            return ReduceOnlyResult::no_filtering();
        }

        // Check if position is underwater (unrealized P&L is negative)
        let is_underwater = config.unrealized_pnl < 0.0;

        // === UNDERWATER POSITION PROTECTION ===
        // Same logic as apply_reduce_only_ladder
        if is_underwater && !approaching_liquidation && !over_margin_limit {
            if position > 0.0 {
                *bid = None;
                warn!(
                    asset = %config.asset,
                    position = %format!("{:.6}", position),
                    unrealized_pnl = %format!("${:.2}", config.unrealized_pnl),
                    "Position over limit but UNDERWATER (single) - filtering bid only"
                );
                return ReduceOnlyResult {
                    was_filtered: true,
                    reason: Some(ReduceOnlyReason::UnderwaterWidenOnly),
                    filtered_bids: true,
                    filtered_asks: false,
                    needs_escalation: false,
                    close_urgency: 0.0,
                };
            } else {
                *ask = None;
                warn!(
                    asset = %config.asset,
                    position = %format!("{:.6}", position),
                    unrealized_pnl = %format!("${:.2}", config.unrealized_pnl),
                    "Position over limit but UNDERWATER (single) - filtering ask only"
                );
                return ReduceOnlyResult {
                    was_filtered: true,
                    reason: Some(ReduceOnlyReason::UnderwaterWidenOnly),
                    filtered_bids: false,
                    filtered_asks: true,
                    needs_escalation: false,
                    close_urgency: 0.0,
                };
            }
        }

        // Determine primary reason (priority order)
        let reason = if approaching_liquidation {
            ReduceOnlyReason::ApproachingLiquidation
        } else if over_margin_limit {
            ReduceOnlyReason::OverMarginUtilization
        } else if over_value_limit {
            ReduceOnlyReason::OverValueLimit
        } else {
            ReduceOnlyReason::OverPositionLimit
        };

        if position > 0.0 {
            // Long position over max: only allow sells (no bids)
            *bid = None;
            Self::log_reduce_only(position, config, reason, true, margin_utilization);
            ReduceOnlyResult::filtered_bids(reason)
        } else {
            // Short position over max: only allow buys (no asks)
            *ask = None;
            Self::log_reduce_only(position, config, reason, false, margin_utilization);
            ReduceOnlyResult::filtered_asks(reason)
        }
    }

    /// Check if we should be in reduce-only mode.
    ///
    /// Priority (highest to lowest):
    /// 1. Approaching liquidation (buffer_ratio < threshold) - DYNAMIC, exchange-derived
    /// 2. Margin utilization > 80% (capital constraint)
    /// 3. Position value exceeds max_position_value
    /// 4. Position exceeds max_position (fallback/override)
    pub fn is_reduce_only(config: &ReduceOnlyConfig) -> Option<ReduceOnlyReason> {
        let position = config.position;
        let position_value = position.abs() * config.mid_price;

        // Check margin utilization
        let margin_utilization = if config.account_value > 0.0 {
            config.margin_used / config.account_value
        } else {
            0.0
        };

        // Priority 1: Liquidation proximity (highest priority)
        let approaching_liquidation = config
            .liquidation_buffer_ratio
            .map(|ratio| ratio < config.liquidation_trigger_threshold)
            .unwrap_or(false);

        if approaching_liquidation {
            Some(ReduceOnlyReason::ApproachingLiquidation)
        } else if margin_utilization > MARGIN_UTILIZATION_THRESHOLD {
            Some(ReduceOnlyReason::OverMarginUtilization)
        } else if position_value > config.max_position_value {
            Some(ReduceOnlyReason::OverValueLimit)
        } else if position.abs() > config.max_position {
            Some(ReduceOnlyReason::OverPositionLimit)
        } else {
            None
        }
    }

    /// Apply reduce-only logic with exchange limit awareness (Phase 3 Fix).
    ///
    /// This enhanced version checks both local limits AND exchange capacity.
    /// If we're in reduce-only mode but the exchange has no capacity to place
    /// reducing orders, we signal that escalation (e.g., market orders) is needed.
    ///
    /// Priority (highest to lowest):
    /// 1. Approaching liquidation (buffer_ratio < threshold) - DYNAMIC, exchange-derived
    ///    → ALWAYS force reduce, even at a loss (survival > P&L)
    /// 2. Margin utilization > 80% (capital constraint)
    /// 3. Position value exceeds max_position_value
    /// 4. Position exceeds max_position (fallback/override)
    ///
    /// Special case: Underwater position protection (same as ladder version)
    ///
    /// # Arguments
    /// - `bids`: Mutable bid quotes to filter
    /// - `asks`: Mutable ask quotes to filter
    /// - `config`: Reduce-only configuration
    /// - `exchange_limits`: Exchange position limits (for capacity check)
    ///
    /// # Returns
    /// `ReduceOnlyResult` with `needs_escalation = true` if stuck
    pub fn apply_reduce_only_with_exchange_limits(
        bids: &mut Vec<Quote>,
        asks: &mut Vec<Quote>,
        config: &ReduceOnlyConfig,
        exchange_limits: &ExchangePositionLimits,
    ) -> ReduceOnlyResult {
        let position = config.position;
        let position_value = position.abs() * config.mid_price;

        // Check margin utilization (capital-efficient approach)
        let margin_utilization = if config.account_value > 0.0 {
            config.margin_used / config.account_value
        } else {
            0.0
        };

        // === Priority 1: Liquidation Proximity (HIGHEST PRIORITY) ===
        let approaching_liquidation = config
            .liquidation_buffer_ratio
            .map(|ratio| ratio < config.liquidation_trigger_threshold)
            .unwrap_or(false);

        // === Priority 2-4: Fallback triggers ===
        let over_margin_limit = margin_utilization > MARGIN_UTILIZATION_THRESHOLD;
        let over_value_limit = position_value > config.max_position_value;
        let over_position_limit = position.abs() > config.max_position;

        if !approaching_liquidation
            && !over_margin_limit
            && !over_position_limit
            && !over_value_limit
        {
            return ReduceOnlyResult::no_filtering();
        }

        // Check if position is underwater (unrealized P&L is negative)
        let is_underwater = config.unrealized_pnl < 0.0;

        // === UNDERWATER POSITION PROTECTION ===
        // Same logic as apply_reduce_only_ladder
        if is_underwater && !approaching_liquidation && !over_margin_limit {
            if position > 0.0 {
                bids.clear();
                warn!(
                    asset = %config.asset,
                    position = %format!("{:.6}", position),
                    unrealized_pnl = %format!("${:.2}", config.unrealized_pnl),
                    "Position over limit but UNDERWATER (exchange limits) - filtering bids only"
                );
                return ReduceOnlyResult {
                    was_filtered: true,
                    reason: Some(ReduceOnlyReason::UnderwaterWidenOnly),
                    filtered_bids: true,
                    filtered_asks: false,
                    needs_escalation: false,
                    close_urgency: 0.0,
                };
            } else {
                asks.clear();
                warn!(
                    asset = %config.asset,
                    position = %format!("{:.6}", position),
                    unrealized_pnl = %format!("${:.2}", config.unrealized_pnl),
                    "Position over limit but UNDERWATER (exchange limits) - filtering asks only"
                );
                return ReduceOnlyResult {
                    was_filtered: true,
                    reason: Some(ReduceOnlyReason::UnderwaterWidenOnly),
                    filtered_bids: false,
                    filtered_asks: true,
                    needs_escalation: false,
                    close_urgency: 0.0,
                };
            }
        }

        // Determine primary reason (priority order)
        let reason = if approaching_liquidation {
            ReduceOnlyReason::ApproachingLiquidation
        } else if over_margin_limit {
            ReduceOnlyReason::OverMarginUtilization
        } else if over_value_limit {
            ReduceOnlyReason::OverValueLimit
        } else {
            ReduceOnlyReason::OverPositionLimit
        };

        // Minimum capacity to consider "usable" (avoid micro-orders)
        const MIN_CAPACITY: f64 = 0.001;

        if position > 0.0 {
            // Long position over max: only allow sells (no bids)
            bids.clear();

            // Phase 3 Fix: Check if we can actually place asks (sells to reduce)
            let available_sell = exchange_limits.available_sell();
            if available_sell < MIN_CAPACITY && exchange_limits.is_initialized() {
                warn!(
                    position = %format!("{:.6}", position),
                    available_sell = %format!("{:.6}", available_sell),
                    "Reduce-only mode but no exchange capacity to sell - needs escalation"
                );
                Self::log_reduce_only(position, config, reason, true, margin_utilization);
                return ReduceOnlyResult::filtered_bids_needs_escalation(reason);
            }

            Self::log_reduce_only(position, config, reason, true, margin_utilization);
            ReduceOnlyResult::filtered_bids(reason)
        } else {
            // Short position over max: only allow buys (no asks)
            asks.clear();

            // Phase 3 Fix: Check if we can actually place bids (buys to reduce)
            let available_buy = exchange_limits.available_buy();
            if available_buy < MIN_CAPACITY && exchange_limits.is_initialized() {
                warn!(
                    position = %format!("{:.6}", position),
                    available_buy = %format!("{:.6}", available_buy),
                    "Reduce-only mode but no exchange capacity to buy - needs escalation"
                );
                Self::log_reduce_only(position, config, reason, false, margin_utilization);
                return ReduceOnlyResult::filtered_asks_needs_escalation(reason);
            }

            Self::log_reduce_only(position, config, reason, false, margin_utilization);
            ReduceOnlyResult::filtered_asks(reason)
        }
    }

    fn log_reduce_only(
        position: f64,
        config: &ReduceOnlyConfig,
        reason: ReduceOnlyReason,
        is_bid_side: bool,
        margin_utilization: f64,
    ) {
        let side_name = if is_bid_side { "bids" } else { "asks" };
        let position_type = if position > 0.0 { "long" } else { "short" };

        match reason {
            ReduceOnlyReason::ApproachingLiquidation => {
                warn!(
                    asset = %config.asset,
                    position = %format!("{:.6}", position),
                    liquidation_price = ?config.liquidation_price.map(|p| format!("{p:.4}")),
                    buffer_ratio = ?config.liquidation_buffer_ratio.map(|r| format!("{:.2}%", r * 100.0)),
                    threshold = %format!("{:.1}%", config.liquidation_trigger_threshold * 100.0),
                    mid_price = %format!("{:.4}", config.mid_price),
                    "APPROACHING LIQUIDATION ({}) - reduce-only mode, cancelling {}",
                    position_type,
                    side_name
                );
            }
            ReduceOnlyReason::OverMarginUtilization => {
                warn!(
                    position = %format!("{:.6}", position),
                    margin_used = %format!("${:.2}", config.margin_used),
                    account_value = %format!("${:.2}", config.account_value),
                    utilization = %format!("{:.1}%", margin_utilization * 100.0),
                    threshold = %format!("{:.1}%", MARGIN_UTILIZATION_THRESHOLD * 100.0),
                    "Margin utilization over threshold ({}) - reduce-only mode, cancelling {}",
                    position_type,
                    side_name
                );
            }
            ReduceOnlyReason::OverValueLimit => {
                let position_value = position.abs() * config.mid_price;
                warn!(
                    position = %format!("{:.6}", position),
                    position_value = %format!("${:.2}", position_value),
                    limit = %format!("${:.2}", config.max_position_value),
                    "Position value over limit ({}) - reduce-only mode, cancelling {}",
                    position_type,
                    side_name
                );
            }
            ReduceOnlyReason::OverPositionLimit => {
                warn!(
                    position = %format!("{:.6}", position),
                    max_position = %format!("{:.6}", config.max_position),
                    "Over max position ({}) - reduce-only mode, cancelling {}",
                    position_type,
                    side_name
                );
            }
            ReduceOnlyReason::UnderwaterWidenOnly => {
                // This case is handled inline in the apply_* methods with more detailed logging
                // This branch should not be reached from log_reduce_only
                warn!(
                    asset = %config.asset,
                    position = %format!("{:.6}", position),
                    unrealized_pnl = %format!("${:.2}", config.unrealized_pnl),
                    "Position over limit but UNDERWATER - widening spreads, not forcing exit"
                );
            }
        }
    }
}

/// Apply close bias to tighten the closing side spread.
///
/// When we're long and need to reduce: tighten ask (sell) side
/// When we're short and need to reduce: tighten bid (buy) side
///
/// INVARIANT: Never crosses mid price. The tightened price is always
/// on the correct side of mid.
///
/// # Arguments
/// * `bid_price` - Current bid price
/// * `ask_price` - Current ask price
/// * `mid_price` - Current mid price
/// * `position` - Current position (positive = long)
/// * `urgency` - Close urgency [0.0, 1.0]
///
/// # Returns
/// (adjusted_bid, adjusted_ask)
pub fn apply_close_bias(
    bid_price: f64,
    ask_price: f64,
    mid_price: f64,
    position: f64,
    urgency: f64,
) -> (f64, f64) {
    if urgency <= 0.0 || urgency > 1.0 {
        return (bid_price, ask_price);
    }

    let half_spread = (ask_price - bid_price) / 2.0;
    let tightening = urgency * 0.5 * half_spread; // Max 50% of half-spread

    if position > 0.0 {
        // Long position: tighten ask (sell side) to attract buyers
        let new_ask = (ask_price - tightening).max(mid_price + 1e-10); // Never cross mid
        (bid_price, new_ask)
    } else if position < 0.0 {
        // Short position: tighten bid (buy side) to attract sellers
        let new_bid = (bid_price + tightening).min(mid_price - 1e-10); // Never cross mid
        (new_bid, ask_price)
    } else {
        (bid_price, ask_price)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(position: f64) -> ReduceOnlyConfig {
        // Set value limit high enough that it won't trigger by default
        // Position 10 * 50000 = $500k, so set limit to $1M
        // Margin utilization low enough to not trigger (50% < 80% threshold)
        ReduceOnlyConfig {
            position,
            max_position: 10.0,
            mid_price: 50000.0,
            max_position_value: 1_000_000.0, // $1M (high enough to not trigger)
            asset: "BTC".to_string(),
            margin_used: 5000.0,    // $5k margin used
            account_value: 10000.0, // $10k account = 50% utilization (below 80% threshold)
            // Liquidation fields - default to None (disabled) for backward compat tests
            liquidation_price: None,
            liquidation_buffer_ratio: None,
            liquidation_trigger_threshold: DEFAULT_LIQUIDATION_TRIGGER_THRESHOLD,
            // Unrealized P&L - default to positive (not underwater)
            unrealized_pnl: 1000.0, // $1000 profit by default
        }
    }

    #[test]
    fn test_no_filtering_when_within_limits() {
        let config = make_config(5.0); // 5 BTC, under max_position of 10
        let mut bids = vec![Quote::new(49000.0, 0.1)];
        let mut asks = vec![Quote::new(51000.0, 0.1)];

        let result = QuoteFilter::apply_reduce_only_ladder(&mut bids, &mut asks, &config);

        assert!(!result.was_filtered);
        assert_eq!(bids.len(), 1);
        assert_eq!(asks.len(), 1);
    }

    #[test]
    fn test_filter_bids_when_long_over_limit() {
        let config = make_config(15.0); // Over max_position of 10
        let mut bids = vec![Quote::new(49000.0, 0.1)];
        let mut asks = vec![Quote::new(51000.0, 0.1)];

        let result = QuoteFilter::apply_reduce_only_ladder(&mut bids, &mut asks, &config);

        assert!(result.was_filtered);
        assert!(result.filtered_bids);
        assert!(!result.filtered_asks);
        assert_eq!(result.reason, Some(ReduceOnlyReason::OverPositionLimit));
        assert!(bids.is_empty());
        assert_eq!(asks.len(), 1);
    }

    #[test]
    fn test_filter_asks_when_short_over_limit() {
        let config = make_config(-15.0); // Short over max_position of 10
        let mut bids = vec![Quote::new(49000.0, 0.1)];
        let mut asks = vec![Quote::new(51000.0, 0.1)];

        let result = QuoteFilter::apply_reduce_only_ladder(&mut bids, &mut asks, &config);

        assert!(result.was_filtered);
        assert!(!result.filtered_bids);
        assert!(result.filtered_asks);
        assert_eq!(result.reason, Some(ReduceOnlyReason::OverPositionLimit));
        assert_eq!(bids.len(), 1);
        assert!(asks.is_empty());
    }

    #[test]
    fn test_value_limit_takes_precedence() {
        // Position value: 5 * 50000 = $250k
        // Set value limit to $100k so it triggers
        let mut config = make_config(5.0);
        config.max_position_value = 100_000.0; // Lower than position value

        let mut bids = vec![Quote::new(49000.0, 0.1)];
        let mut asks = vec![Quote::new(51000.0, 0.1)];

        let result = QuoteFilter::apply_reduce_only_ladder(&mut bids, &mut asks, &config);

        assert!(result.was_filtered);
        assert_eq!(result.reason, Some(ReduceOnlyReason::OverValueLimit));
    }

    #[test]
    fn test_single_quote_filtering() {
        let config = make_config(15.0); // Over max_position of 10
        let mut bid = Some(Quote::new(49000.0, 0.1));
        let mut ask = Some(Quote::new(51000.0, 0.1));

        let result = QuoteFilter::apply_reduce_only_single(&mut bid, &mut ask, &config);

        assert!(result.was_filtered);
        assert!(bid.is_none());
        assert!(ask.is_some());
    }

    #[test]
    fn test_is_reduce_only() {
        let normal_config = make_config(5.0);
        assert!(QuoteFilter::is_reduce_only(&normal_config).is_none());

        let over_position = make_config(15.0);
        assert_eq!(
            QuoteFilter::is_reduce_only(&over_position),
            Some(ReduceOnlyReason::OverPositionLimit)
        );
    }

    #[test]
    fn test_underwater_protection_long() {
        // Position over limit AND underwater (unrealized P&L negative)
        let mut config = make_config(15.0); // Over max_position of 10
        config.unrealized_pnl = -500.0; // $500 loss (underwater)

        let mut bids = vec![Quote::new(49000.0, 0.1)];
        let mut asks = vec![Quote::new(51000.0, 0.1)];

        let result = QuoteFilter::apply_reduce_only_ladder(&mut bids, &mut asks, &config);

        // Should filter bids (don't add to losing long) but keep asks passive
        assert!(result.was_filtered);
        assert_eq!(result.reason, Some(ReduceOnlyReason::UnderwaterWidenOnly));
        assert!(result.filtered_bids);
        assert!(!result.filtered_asks);
        assert!(bids.is_empty()); // Bids cleared
        assert_eq!(asks.len(), 1); // Asks kept for potential exit
    }

    #[test]
    fn test_underwater_protection_short() {
        // Short position over limit AND underwater
        let mut config = make_config(-15.0); // Short over max_position of 10
        config.unrealized_pnl = -500.0; // $500 loss (underwater)

        let mut bids = vec![Quote::new(49000.0, 0.1)];
        let mut asks = vec![Quote::new(51000.0, 0.1)];

        let result = QuoteFilter::apply_reduce_only_ladder(&mut bids, &mut asks, &config);

        // Should filter asks (don't add to losing short) but keep bids passive
        assert!(result.was_filtered);
        assert_eq!(result.reason, Some(ReduceOnlyReason::UnderwaterWidenOnly));
        assert!(!result.filtered_bids);
        assert!(result.filtered_asks);
        assert_eq!(bids.len(), 1); // Bids kept for potential exit
        assert!(asks.is_empty()); // Asks cleared
    }

    #[test]
    fn test_underwater_but_approaching_liquidation_forces_exit() {
        // Underwater but approaching liquidation - survival > P&L
        let mut config = make_config(15.0);
        config.unrealized_pnl = -500.0; // Underwater
        config.liquidation_buffer_ratio = Some(0.1); // Below threshold (0.3)
        config.liquidation_trigger_threshold = DEFAULT_LIQUIDATION_TRIGGER_THRESHOLD;

        let mut bids = vec![Quote::new(49000.0, 0.1)];
        let mut asks = vec![Quote::new(51000.0, 0.1)];

        let result = QuoteFilter::apply_reduce_only_ladder(&mut bids, &mut asks, &config);

        // Should force reduce (survival > P&L), NOT underwater protection
        assert!(result.was_filtered);
        assert_eq!(result.reason, Some(ReduceOnlyReason::ApproachingLiquidation));
        assert!(result.filtered_bids);
        assert!(!result.filtered_asks);
    }

    #[test]
    fn test_underwater_but_over_margin_forces_exit() {
        // Underwater but over margin limit - capital constraint > P&L optimization
        let mut config = make_config(15.0);
        config.unrealized_pnl = -500.0; // Underwater
        config.margin_used = 9000.0; // 90% utilization (above 80% threshold)

        let mut bids = vec![Quote::new(49000.0, 0.1)];
        let mut asks = vec![Quote::new(51000.0, 0.1)];

        let result = QuoteFilter::apply_reduce_only_ladder(&mut bids, &mut asks, &config);

        // Should force reduce due to margin, NOT underwater protection
        assert!(result.was_filtered);
        assert_eq!(result.reason, Some(ReduceOnlyReason::OverMarginUtilization));
        assert!(result.filtered_bids);
        assert!(!result.filtered_asks);
    }

    #[test]
    fn test_not_underwater_forces_exit() {
        // Over position limit but profitable - can reduce at profit
        let mut config = make_config(15.0);
        config.unrealized_pnl = 1000.0; // $1000 profit (not underwater)

        let mut bids = vec![Quote::new(49000.0, 0.1)];
        let mut asks = vec![Quote::new(51000.0, 0.1)];

        let result = QuoteFilter::apply_reduce_only_ladder(&mut bids, &mut asks, &config);

        // Should force reduce since we can exit profitably
        assert!(result.was_filtered);
        assert_eq!(result.reason, Some(ReduceOnlyReason::OverPositionLimit));
        assert!(result.filtered_bids);
        assert!(!result.filtered_asks);
    }

    // === Close Urgency Tests ===

    #[test]
    fn test_reduce_only_close_urgency_zero_at_threshold() {
        // Position exactly at max -> urgency = 0.0
        let urgency = ReduceOnlyResult::compute_urgency(10.0, 10.0, 0.0);
        assert_eq!(urgency, 0.0);
    }

    #[test]
    fn test_reduce_only_close_urgency_increases_with_excess() {
        // Position = 1.5 * max -> excess_ratio = 0.5 -> urgency = 0.5
        let urgency = ReduceOnlyResult::compute_urgency(15.0, 10.0, 0.0);
        assert!((urgency - 0.5).abs() < 1e-9, "Expected 0.5, got {urgency}");

        // Position = 2.0 * max -> excess_ratio = 1.0 -> urgency = 1.0
        let urgency = ReduceOnlyResult::compute_urgency(20.0, 10.0, 0.0);
        assert!((urgency - 1.0).abs() < 1e-9, "Expected 1.0, got {urgency}");
    }

    #[test]
    fn test_reduce_only_close_urgency_increases_when_underwater() {
        // Same excess (1.5x) but underwater
        let urgency_dry = ReduceOnlyResult::compute_urgency(15.0, 10.0, 100.0);
        let urgency_wet = ReduceOnlyResult::compute_urgency(15.0, 10.0, -50.0);
        assert!(urgency_wet > urgency_dry, "Underwater should increase urgency: wet={urgency_wet} dry={urgency_dry}");
    }

    #[test]
    fn test_reduce_only_close_urgency_capped_at_one() {
        // Extreme position (10x max) + very underwater
        let urgency = ReduceOnlyResult::compute_urgency(100.0, 10.0, -10000.0);
        assert_eq!(urgency, 1.0, "Urgency must be capped at 1.0");
    }

    // === Close Bias Tests ===

    #[test]
    fn test_reduce_only_close_bias_tightens_correct_side_long() {
        let mid = 50000.0;
        let bid = 49990.0;
        let ask = 50010.0;
        let urgency = 0.5;

        let (new_bid, new_ask) = apply_close_bias(bid, ask, mid, 5.0, urgency);
        // Long: ask should decrease (tighten sell side), bid unchanged
        assert_eq!(new_bid, bid);
        assert!(new_ask < ask, "Ask should tighten: new_ask={new_ask} < old_ask={ask}");
        assert!(new_ask > mid, "Ask must stay above mid: new_ask={new_ask} > mid={mid}");
    }

    #[test]
    fn test_reduce_only_close_bias_tightens_correct_side_short() {
        let mid = 50000.0;
        let bid = 49990.0;
        let ask = 50010.0;
        let urgency = 0.5;

        let (new_bid, new_ask) = apply_close_bias(bid, ask, mid, -5.0, urgency);
        // Short: bid should increase (tighten buy side), ask unchanged
        assert_eq!(new_ask, ask);
        assert!(new_bid > bid, "Bid should tighten: new_bid={new_bid} > old_bid={bid}");
        assert!(new_bid < mid, "Bid must stay below mid: new_bid={new_bid} < mid={mid}");
    }

    #[test]
    fn test_reduce_only_close_bias_no_effect_at_zero_urgency() {
        let mid = 50000.0;
        let bid = 49990.0;
        let ask = 50010.0;

        let (new_bid, new_ask) = apply_close_bias(bid, ask, mid, 5.0, 0.0);
        assert_eq!(new_bid, bid);
        assert_eq!(new_ask, ask);
    }

    #[test]
    fn test_reduce_only_close_bias_never_crosses_mid() {
        let mid = 50000.0;
        // Very tight spread to test boundary
        let bid = 49999.0;
        let ask = 50001.0;

        // Long with max urgency
        let (_, new_ask) = apply_close_bias(bid, ask, mid, 5.0, 1.0);
        assert!(new_ask > mid, "Ask must never cross mid: new_ask={new_ask} > mid={mid}");

        // Short with max urgency
        let (new_bid, _) = apply_close_bias(bid, ask, mid, -5.0, 1.0);
        assert!(new_bid < mid, "Bid must never cross mid: new_bid={new_bid} < mid={mid}");
    }
}
