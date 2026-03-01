//! Exchange-Enforced Position Limits
//!
//! Queries and caches the exchange's actual position limits from `active_asset_data` API.
//!
//! # Problem Solved
//!
//! The exchange enforces position limits based on:
//! - Account equity × leverage / mark price
//! - Current position already counts against this limit
//!
//! Without querying these limits, orders can be rejected with:
//! "Order would exceed maximum position size for current leverage"
//!
//! # Design
//!
//! - **Lock-free**: Uses atomic operations for O(1) hot path access
//! - **Pre-computed**: Effective limits calculated on refresh, not on each quote
//! - **Staleness-aware**: Graceful degradation when data is stale
//!
//! # Usage
//!
//! ```ignore
//! // On startup and in safety_sync (every 60s)
//! let response = info_client.active_asset_data(user, coin).await?;
//! limits.update_from_response(&response, local_max_position);
//!
//! // In hot path (generate_ladder)
//! let bid_limit = limits.effective_bid_limit();  // O(1) atomic load
//! let ask_limit = limits.effective_ask_limit();
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::info::response_structs::ActiveAssetDataResponse;
use tracing::{info, warn};

/// Exchange-enforced position limits with lock-free access.
///
/// Thread-safe via atomic operations. Pre-computes effective limits
/// to avoid branching in the hot path.
#[derive(Clone)]
pub struct ExchangePositionLimits {
    inner: Arc<ExchangeLimitsInner>,
}

struct ExchangeLimitsInner {
    // === Raw exchange values ===
    /// Maximum long position allowed (max_trade_szs[0])
    max_long: AtomicF64,
    /// Maximum short position allowed (max_trade_szs[1])
    max_short: AtomicF64,
    /// Available capacity to INCREASE long exposure (available_to_trade[0])
    /// NOTE: This is capacity for new long positions, NOT capacity to close shorts.
    /// Closing a short (buying to flat) releases margin, doesn't consume this.
    available_buy: AtomicF64,
    /// Available capacity to INCREASE short exposure (available_to_trade[1])
    /// NOTE: This is capacity for new short positions, NOT capacity to close longs.
    /// Closing a long (selling to flat) releases margin, doesn't consume this.
    available_sell: AtomicF64,

    // === Pre-computed effective limits ===
    /// Effective bid limit: min(local_max, available_buy)
    effective_bid_limit: AtomicF64,
    /// Effective ask limit: min(local_max, available_sell)
    effective_ask_limit: AtomicF64,

    // === Metadata ===
    /// Last refresh timestamp (epoch milliseconds)
    last_refresh_epoch_ms: AtomicU64,
    /// Local max position used in last computation
    local_max_position: AtomicF64,
    /// Paper trading mode: limits never go stale (no websocket refresh)
    paper_mode: std::sync::atomic::AtomicBool,
    /// Whether the last update had complete exchange data (no parse failures).
    /// `false` when any API field was missing/unparseable (e.g. HIP-3 assets).
    has_full_exchange_data: std::sync::atomic::AtomicBool,
}

impl ExchangePositionLimits {
    /// Create new exchange position limits (uninitialized).
    ///
    /// Call `update_from_response` before using in production.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ExchangeLimitsInner {
                max_long: AtomicF64::new(0.0),
                max_short: AtomicF64::new(0.0),
                available_buy: AtomicF64::new(0.0),
                available_sell: AtomicF64::new(0.0),
                effective_bid_limit: AtomicF64::new(0.0),
                effective_ask_limit: AtomicF64::new(0.0),
                last_refresh_epoch_ms: AtomicU64::new(0),
                local_max_position: AtomicF64::new(0.0),
                paper_mode: std::sync::atomic::AtomicBool::new(false),
                has_full_exchange_data: std::sync::atomic::AtomicBool::new(false),
            }),
        }
    }

    /// Initialize limits for paper trading with synthetic capacity.
    ///
    /// In paper mode there is no exchange to query, so we set generous
    /// limits derived from the paper balance and price.
    pub fn initialize_for_paper(&self, max_position: f64) {
        self.inner.max_long.store(max_position);
        self.inner.max_short.store(max_position);
        self.inner.available_buy.store(max_position);
        self.inner.available_sell.store(max_position);
        self.inner.effective_bid_limit.store(max_position);
        self.inner.effective_ask_limit.store(max_position);
        self.inner.local_max_position.store(max_position);
        self.inner.paper_mode.store(true, Ordering::Relaxed);
        self.inner
            .has_full_exchange_data
            .store(true, Ordering::Relaxed);
        self.inner
            .last_refresh_epoch_ms
            .store(now_epoch_ms(), Ordering::Relaxed);

        info!(
            max_position = %format!("{:.6}", max_position),
            "Exchange limits initialized for paper trading"
        );
    }

    /// Update limits from exchange API response.
    ///
    /// # Arguments
    /// - `response`: Response from `info_client.active_asset_data()`
    /// - `local_max_position`: User-configured max position limit
    ///
    /// # Note
    /// This should be called:
    /// 1. On startup (before first quote)
    /// 2. Every 60s in safety_sync
    /// 3. After significant position changes (>10%)
    /// 4. After order rejection due to position limits
    pub fn update_from_response(
        &self,
        response: &ActiveAssetDataResponse,
        local_max_position: f64,
    ) {
        // Parse exchange limits using Option-based parsing.
        // On parse failure (empty, non-finite, negative), fall back to local_max_position
        // instead of f64::MAX, so the exchange guardrail remains meaningful.
        let max_long_opt = parse_f64_opt(&response.max_trade_szs, 0);
        let max_short_opt = parse_f64_opt(&response.max_trade_szs, 1);
        let available_buy_usd_opt = parse_f64_opt(&response.available_to_trade, 0);
        let available_sell_usd_opt = parse_f64_opt(&response.available_to_trade, 1);
        let mark_price_opt = parse_f64_opt_str(&response.mark_px);

        let max_long = max_long_opt.unwrap_or(local_max_position);
        let max_short = max_short_opt.unwrap_or(local_max_position);

        // Convert available notional (USD) to position size (contracts).
        // If either USD amount or mark_px is missing, fall back to local_max_position.
        let available_buy = match (available_buy_usd_opt, mark_price_opt) {
            (Some(usd), Some(px)) if px > 0.0 => usd / px,
            _ => local_max_position,
        };
        let available_sell = match (available_sell_usd_opt, mark_price_opt) {
            (Some(usd), Some(px)) if px > 0.0 => usd / px,
            _ => local_max_position,
        };

        // Track whether we got complete data from the exchange
        let has_full_data = max_long_opt.is_some()
            && max_short_opt.is_some()
            && available_buy_usd_opt.is_some()
            && available_sell_usd_opt.is_some()
            && mark_price_opt.is_some();
        self.inner
            .has_full_exchange_data
            .store(has_full_data, Ordering::Relaxed);

        if !has_full_data {
            warn!(
                max_long_parsed = max_long_opt.is_some(),
                max_short_parsed = max_short_opt.is_some(),
                available_buy_parsed = available_buy_usd_opt.is_some(),
                available_sell_parsed = available_sell_usd_opt.is_some(),
                mark_px_parsed = mark_price_opt.is_some(),
                fallback = %format!("{:.6}", local_max_position),
                "Exchange limits: partial/missing data, using local_max_position as fallback"
            );
        }

        // Store raw values (in contracts for consistency)
        self.inner.max_long.store(max_long);
        self.inner.max_short.store(max_short);
        self.inner.available_buy.store(available_buy);
        self.inner.available_sell.store(available_sell);

        // Pre-compute effective limits: min(local_max, max_long, available_buy)
        // All three values are now sensible (no f64::MAX), so min() always produces
        // a real limit.
        let effective_bid = local_max_position.min(max_long).min(available_buy);
        let effective_ask = local_max_position.min(max_short).min(available_sell);
        self.inner.effective_bid_limit.store(effective_bid);
        self.inner.effective_ask_limit.store(effective_ask);

        // Update metadata
        self.inner.local_max_position.store(local_max_position);
        self.inner
            .last_refresh_epoch_ms
            .store(now_epoch_ms(), Ordering::Relaxed);

        tracing::debug!(
            max_long = %format!("{:.6}", max_long),
            max_short = %format!("{:.6}", max_short),
            available_buy = %format!("{:.6}", available_buy),
            available_sell = %format!("{:.6}", available_sell),
            effective_bid = %format!("{:.6}", effective_bid),
            effective_ask = %format!("{:.6}", effective_ask),
            has_full_data = has_full_data,
            "Exchange position limits updated"
        );
    }

    /// Update limits from WebSocket message.
    ///
    /// Unlike `update_from_response`, the WebSocket `ActiveAssetData` message
    /// does NOT include `mark_px`, so we must pass it from local state.
    ///
    /// # Arguments
    /// - `data`: Data from `ActiveAssetData` WebSocket message.
    /// - `mark_price`: Current mark price from `latest_mid`.
    /// - `local_max_position`: User-configured max position limit.
    pub fn update_from_ws(
        &self,
        data: &crate::types::ActiveAssetDataData,
        mark_price: f64,
        local_max_position: f64,
    ) {
        let max_long_opt = parse_f64_opt(&data.max_trade_szs, 0);
        let max_short_opt = parse_f64_opt(&data.max_trade_szs, 1);
        let available_buy_usd_opt = parse_f64_opt(&data.available_to_trade, 0);
        let available_sell_usd_opt = parse_f64_opt(&data.available_to_trade, 1);

        let max_long = max_long_opt.unwrap_or(local_max_position);
        let max_short = max_short_opt.unwrap_or(local_max_position);

        // Convert available notional (USD) to position size (contracts).
        // mark_price is passed from local state; if invalid, fall back to local_max.
        let mark_price_valid = mark_price > 0.0 && mark_price.is_finite();
        let available_buy = match (available_buy_usd_opt, mark_price_valid) {
            (Some(usd), true) => usd / mark_price,
            _ => local_max_position,
        };
        let available_sell = match (available_sell_usd_opt, mark_price_valid) {
            (Some(usd), true) => usd / mark_price,
            _ => local_max_position,
        };

        // Track whether we got complete data
        let has_full_data = max_long_opt.is_some()
            && max_short_opt.is_some()
            && available_buy_usd_opt.is_some()
            && available_sell_usd_opt.is_some()
            && mark_price_valid;
        self.inner
            .has_full_exchange_data
            .store(has_full_data, Ordering::Relaxed);

        if !has_full_data {
            warn!(
                max_long_parsed = max_long_opt.is_some(),
                max_short_parsed = max_short_opt.is_some(),
                available_buy_parsed = available_buy_usd_opt.is_some(),
                available_sell_parsed = available_sell_usd_opt.is_some(),
                mark_price_valid = mark_price_valid,
                fallback = %format!("{:.6}", local_max_position),
                "Exchange limits WS: partial/missing data, using local_max_position as fallback"
            );
        }

        // Store raw values
        self.inner.max_long.store(max_long);
        self.inner.max_short.store(max_short);
        self.inner.available_buy.store(available_buy);
        self.inner.available_sell.store(available_sell);

        // Pre-compute effective limits
        let effective_bid = local_max_position.min(max_long).min(available_buy);
        let effective_ask = local_max_position.min(max_short).min(available_sell);
        self.inner.effective_bid_limit.store(effective_bid);
        self.inner.effective_ask_limit.store(effective_ask);

        // Update metadata
        self.inner.local_max_position.store(local_max_position);
        self.inner
            .last_refresh_epoch_ms
            .store(now_epoch_ms(), Ordering::Relaxed);

        tracing::debug!(
            max_long = %format!("{:.6}", max_long),
            max_short = %format!("{:.6}", max_short),
            available_buy = %format!("{:.6}", available_buy),
            available_sell = %format!("{:.6}", available_sell),
            mark_price = %format!("{:.2}", mark_price),
            effective_bid = %format!("{:.6}", effective_bid),
            effective_ask = %format!("{:.6}", effective_ask),
            has_full_data = has_full_data,
            source = "WebSocket",
            "Exchange position limits updated from WS"
        );
    }

    /// Update only the local max position (re-computes effective limits).
    ///
    /// Use when local config changes but exchange limits are still valid.
    /// Uses the same 3-factor formula as `update_from_response`/`update_from_ws`:
    /// `effective = min(local_max, max_long/short, available_buy/sell)`
    pub fn update_local_max(&self, local_max_position: f64) {
        let max_long = self.inner.max_long.load();
        let max_short = self.inner.max_short.load();
        let available_buy = self.inner.available_buy.load();
        let available_sell = self.inner.available_sell.load();

        self.inner
            .effective_bid_limit
            .store(local_max_position.min(max_long).min(available_buy));
        self.inner
            .effective_ask_limit
            .store(local_max_position.min(max_short).min(available_sell));
        self.inner.local_max_position.store(local_max_position);
    }

    // =========================================================================
    // Hot Path Accessors (O(1) atomic loads)
    // =========================================================================

    /// Get effective bid limit (for buy orders).
    ///
    /// Returns `min(local_max_position, exchange_available_buy)`.
    /// This is the maximum size we can bid without risking rejection.
    #[inline]
    pub fn effective_bid_limit(&self) -> f64 {
        self.inner.effective_bid_limit.load()
    }

    /// Get effective ask limit (for sell orders).
    ///
    /// Returns `min(local_max_position, exchange_available_sell)`.
    /// This is the maximum size we can ask without risking rejection.
    #[inline]
    pub fn effective_ask_limit(&self) -> f64 {
        self.inner.effective_ask_limit.load()
    }

    /// Get safe bid limit with staleness degradation.
    ///
    /// Applies safety factors based on data age:
    /// - < 2 min: 100% (use as-is)
    /// - 2-5 min: 50% (reduce sizing)
    /// - > 5 min: 0% (pause quoting)
    pub fn safe_bid_limit(&self) -> f64 {
        let base = self.inner.effective_bid_limit.load();
        apply_staleness_factor(base, self.age_ms())
    }

    /// Get safe ask limit with staleness degradation.
    pub fn safe_ask_limit(&self) -> f64 {
        let base = self.inner.effective_ask_limit.load();
        apply_staleness_factor(base, self.age_ms())
    }

    // =========================================================================
    // Raw Value Accessors (for logging/metrics)
    // =========================================================================

    /// Maximum long position allowed by exchange.
    pub fn max_long(&self) -> f64 {
        self.inner.max_long.load()
    }

    /// Maximum short position allowed by exchange.
    pub fn max_short(&self) -> f64 {
        self.inner.max_short.load()
    }

    /// Available capacity to buy (contracts).
    pub fn available_buy(&self) -> f64 {
        self.inner.available_buy.load()
    }

    /// Available capacity to sell (contracts).
    pub fn available_sell(&self) -> f64 {
        self.inner.available_sell.load()
    }

    // =========================================================================
    // Staleness & Validation
    // =========================================================================

    /// Time since last refresh in milliseconds.
    ///
    /// Whether limits are in paper trading mode.
    ///
    /// In paper mode, limits are synthetically initialized and should NOT
    /// be overwritten by WebSocket `ActiveAssetData` messages (which carry
    /// real account data that would zero out the synthetic capacity).
    pub fn is_paper_mode(&self) -> bool {
        self.inner.paper_mode.load(Ordering::Relaxed)
    }

    /// In paper mode, always returns 0 (never stale) because limits are
    /// synthetically initialized and never refreshed via websocket.
    pub fn age_ms(&self) -> u64 {
        if self.inner.paper_mode.load(Ordering::Relaxed) {
            return 0; // Paper mode: limits never go stale
        }
        let last = self.inner.last_refresh_epoch_ms.load(Ordering::Relaxed);
        if last == 0 {
            return u64::MAX; // Never initialized
        }
        now_epoch_ms().saturating_sub(last)
    }

    /// Time since last refresh as Duration.
    pub fn age(&self) -> Duration {
        Duration::from_millis(self.age_ms())
    }

    /// Check if limits have been initialized.
    pub fn is_initialized(&self) -> bool {
        self.inner.last_refresh_epoch_ms.load(Ordering::Relaxed) > 0
    }

    /// Check if limits are stale (> 2 minutes old).
    pub fn is_stale(&self) -> bool {
        self.age_ms() > 120_000 // 2 minutes
    }

    /// Check if limits are critically stale (> 5 minutes old).
    pub fn is_critically_stale(&self) -> bool {
        self.age_ms() > 300_000 // 5 minutes
    }

    /// Whether the last update had complete exchange data (no parse failures).
    ///
    /// `false` means some fields were missing/unparseable (e.g. HIP-3 assets
    /// where `available_to_trade` or `mark_px` come back empty), and limits
    /// fell back to `local_max_position`.
    pub fn has_full_exchange_data(&self) -> bool {
        self.inner.has_full_exchange_data.load(Ordering::Relaxed)
    }

    /// Calculate safe order size that won't exceed exchange limits.
    ///
    /// This is the key pre-flight check that prevents order rejections (Phase 2 Fix).
    ///
    /// # Arguments
    /// - `requested_size`: Desired order size
    /// - `is_buy`: Whether this is a buy order
    /// - `current_position`: Current position (positive = long)
    ///
    /// # Returns
    /// `(safe_size, was_clamped, reason)` where:
    /// - `safe_size`: Size that is safe to place (may be 0 if no capacity)
    /// - `was_clamped`: Whether the size was reduced
    /// - `reason`: Why the size was clamped (if applicable)
    pub fn calculate_safe_order_size(
        &self,
        requested_size: f64,
        is_buy: bool,
        current_position: f64,
    ) -> (f64, bool, Option<String>) {
        if !self.is_initialized() {
            // Not initialized — block all orders until limits are fetched
            return (
                0.0,
                true,
                Some("Exchange limits not yet initialized".to_string()),
            );
        }

        let available = if is_buy {
            self.inner.available_buy.load()
        } else {
            self.inner.available_sell.load()
        };

        // Clamp to available capacity
        if requested_size <= available {
            (requested_size, false, None)
        } else if available > 0.0 {
            let reason = format!(
                "{} size {:.6} clamped to available {:.6} (position: {:.6})",
                if is_buy { "Buy" } else { "Sell" },
                requested_size,
                available,
                current_position
            );
            (available, true, Some(reason))
        } else {
            let reason = format!(
                "{} blocked: no available capacity (position: {:.6})",
                if is_buy { "Buy" } else { "Sell" },
                current_position
            );
            (0.0, true, Some(reason))
        }
    }

    /// Check if a proposed order would likely be rejected.
    ///
    /// # Arguments
    /// - `size`: Order size in contracts
    /// - `is_buy`: Whether this is a buy order
    /// - `current_position`: Current position (positive = long)
    ///
    /// # Returns
    /// `(would_exceed, reason)` - true if order would exceed limits
    pub fn would_exceed_limit(
        &self,
        size: f64,
        is_buy: bool,
        current_position: f64,
    ) -> (bool, Option<String>) {
        if !self.is_initialized() {
            return (
                true,
                Some("Exchange limits not yet initialized".to_string()),
            );
        }

        if is_buy {
            // Buying increases long exposure
            let available = self.inner.available_buy.load();
            if size > available {
                return (
                    true,
                    Some(format!(
                        "Buy size {size:.6} exceeds available {available:.6}"
                    )),
                );
            }
            // Check against max long
            let new_position = current_position + size;
            let max_long = self.inner.max_long.load();
            if new_position > max_long {
                return (
                    true,
                    Some(format!(
                        "New position {new_position:.6} would exceed max long {max_long:.6}"
                    )),
                );
            }
        } else {
            // Selling increases short exposure
            let available = self.inner.available_sell.load();
            if size > available {
                return (
                    true,
                    Some(format!(
                        "Sell size {size:.6} exceeds available {available:.6}"
                    )),
                );
            }
            // Check against max short
            let new_position = current_position - size;
            let max_short = self.inner.max_short.load();
            let neg_max_short = -max_short;
            if new_position < neg_max_short {
                return (
                    true,
                    Some(format!(
                        "New position {new_position:.6} would exceed max short {neg_max_short:.6}"
                    )),
                );
            }
        }

        (false, None)
    }

    /// Get a summary for logging.
    pub fn summary(&self) -> ExchangeLimitsSummary {
        ExchangeLimitsSummary {
            max_long: self.inner.max_long.load(),
            max_short: self.inner.max_short.load(),
            available_buy: self.inner.available_buy.load(),
            available_sell: self.inner.available_sell.load(),
            effective_bid_limit: self.inner.effective_bid_limit.load(),
            effective_ask_limit: self.inner.effective_ask_limit.load(),
            local_max_position: self.inner.local_max_position.load(),
            age_ms: self.age_ms(),
            is_stale: self.is_stale(),
            has_full_exchange_data: self.has_full_exchange_data(),
        }
    }

    /// Filter a batch of orders to stay within exchange limits.
    ///
    /// This method processes orders in sequence, tracking cumulative exposure
    /// and filtering out orders that would cause the total to exceed limits.
    ///
    /// # Arguments
    /// - `orders`: Slice of (size, is_buy) tuples representing proposed orders
    /// - `current_position`: Current position (positive = long)
    ///
    /// # Returns
    /// Vector of booleans indicating which orders are allowed (true = OK, false = skip)
    pub fn filter_orders_to_limits(
        &self,
        orders: &[(f64, bool)],
        current_position: f64,
    ) -> Vec<bool> {
        if !self.is_initialized() {
            // Not initialized — block all orders until limits are fetched
            return vec![false; orders.len()];
        }

        let available_buy = self.inner.available_buy.load();
        let available_sell = self.inner.available_sell.load();
        let max_long = self.inner.max_long.load();
        let max_short = self.inner.max_short.load();

        let mut cumulative_buy = 0.0;
        let mut cumulative_sell = 0.0;
        let mut result = Vec::with_capacity(orders.len());

        for &(size, is_buy) in orders {
            let allowed = if is_buy {
                // Check if cumulative buys + this order would exceed available
                let new_cumulative = cumulative_buy + size;
                let within_available = new_cumulative <= available_buy;

                // Check if new position would exceed max long
                let new_position = current_position + new_cumulative;
                let within_max = new_position <= max_long;

                within_available && within_max
            } else {
                // Check if cumulative sells + this order would exceed available
                let new_cumulative = cumulative_sell + size;
                let within_available = new_cumulative <= available_sell;

                // Check if new position would exceed max short
                let new_position = current_position - new_cumulative;
                let within_max = new_position >= -max_short;

                within_available && within_max
            };

            if allowed {
                if is_buy {
                    cumulative_buy += size;
                } else {
                    cumulative_sell += size;
                }
            }

            result.push(allowed);
        }

        result
    }

    /// Calculate remaining capacity after accounting for proposed orders.
    ///
    /// Use this to check how much more exposure we can add after a batch of orders.
    ///
    /// # Arguments
    /// - `buy_exposure`: Total buy order size being placed
    /// - `sell_exposure`: Total sell order size being placed
    ///
    /// # Returns
    /// (remaining_buy_capacity, remaining_sell_capacity)
    pub fn remaining_capacity(&self, buy_exposure: f64, sell_exposure: f64) -> (f64, f64) {
        let available_buy = self.inner.available_buy.load();
        let available_sell = self.inner.available_sell.load();

        let remaining_buy = (available_buy - buy_exposure).max(0.0);
        let remaining_sell = (available_sell - sell_exposure).max(0.0);

        (remaining_buy, remaining_sell)
    }

    /// Log current exchange limits status for diagnostic purposes.
    ///
    /// Call this at key points (e.g., before quote generation, after refresh)
    /// to help debug position limit issues.
    pub fn log_status(&self, context: &str) {
        info!(
            context = %context,
            available_buy = %format!("{:.6}", self.available_buy()),
            available_sell = %format!("{:.6}", self.available_sell()),
            effective_bid = %format!("{:.6}", self.effective_bid_limit()),
            effective_ask = %format!("{:.6}", self.effective_ask_limit()),
            max_long = %format!("{:.6}", self.max_long()),
            max_short = %format!("{:.6}", self.max_short()),
            age_ms = self.age_ms(),
            initialized = self.is_initialized(),
            stale = self.is_stale(),
            has_full_exchange_data = self.has_full_exchange_data(),
            "Exchange limits status"
        );
    }
}

impl Default for ExchangePositionLimits {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of exchange position limits for logging.
#[derive(Debug, Clone)]
pub struct ExchangeLimitsSummary {
    pub max_long: f64,
    pub max_short: f64,
    pub available_buy: f64,
    pub available_sell: f64,
    pub effective_bid_limit: f64,
    pub effective_ask_limit: f64,
    pub local_max_position: f64,
    pub age_ms: u64,
    pub is_stale: bool,
    pub has_full_exchange_data: bool,
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Parse f64 from string array, returning `None` on failure.
///
/// Rejects empty strings, non-finite values (NaN, Inf), and negative values.
/// `None` means "field unavailable" — callers should fall back to `local_max_position`.
fn parse_f64_opt(arr: &[String], index: usize) -> Option<f64> {
    let s = arr.get(index)?;
    parse_f64_opt_str(s)
}

/// Parse f64 from a single string, returning `None` on failure.
///
/// Rejects empty strings, non-finite values (NaN, Inf), and negative values.
fn parse_f64_opt_str(s: &str) -> Option<f64> {
    if s.is_empty() {
        return None;
    }
    let v = s.parse::<f64>().ok()?;
    if !v.is_finite() || v < 0.0 {
        return None;
    }
    Some(v)
}

/// Get current epoch time in milliseconds.
fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Apply staleness degradation factor to a limit.
fn apply_staleness_factor(base: f64, age_ms: u64) -> f64 {
    match age_ms {
        0..=120_000 => base,             // < 2 min: 100%
        120_001..=300_000 => base * 0.5, // 2-5 min: 50%
        _ => 0.0,                        // > 5 min: pause
    }
}

// =============================================================================
// AtomicF64 (lock-free f64 via AtomicU64)
// =============================================================================

/// Atomic f64 wrapper using bit-level AtomicU64.
///
/// Provides lock-free read/write for f64 values.
struct AtomicF64(AtomicU64);

impl AtomicF64 {
    fn new(val: f64) -> Self {
        Self(AtomicU64::new(val.to_bits()))
    }

    #[inline]
    fn load(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }

    #[inline]
    fn store(&self, val: f64) {
        self.0.store(val.to_bits(), Ordering::Relaxed);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create test response.
    ///
    /// NOTE: avail_buy_usd and avail_sell_usd are in USD, and will be converted
    /// to contract size using mark_px = 100000.0
    ///
    /// Example: avail_buy_usd = 100000.0 → available_buy = 1.0 BTC
    fn make_response(
        max_long: f64,
        max_short: f64,
        avail_buy_usd: f64,
        avail_sell_usd: f64,
    ) -> ActiveAssetDataResponse {
        use crate::types::Leverage;
        use alloy::primitives::Address;

        ActiveAssetDataResponse {
            user: Address::ZERO,
            coin: "BTC".to_string(),
            leverage: Leverage {
                type_string: "cross".to_string(),
                value: 10,
                raw_usd: None,
            },
            max_trade_szs: vec![max_long.to_string(), max_short.to_string()],
            available_to_trade: vec![avail_buy_usd.to_string(), avail_sell_usd.to_string()],
            mark_px: "100000.0".to_string(),
        }
    }

    #[test]
    fn test_new_uninitialized() {
        let limits = ExchangePositionLimits::new();
        assert!(!limits.is_initialized());
        assert!(limits.is_stale());
        // Conservative default: 0.0 blocks all orders until real limits are fetched
        assert_eq!(limits.effective_bid_limit(), 0.0);
        assert_eq!(limits.effective_ask_limit(), 0.0);
        assert_eq!(limits.max_long(), 0.0);
        assert_eq!(limits.max_short(), 0.0);
        assert_eq!(limits.available_buy(), 0.0);
        assert_eq!(limits.available_sell(), 0.0);
    }

    #[test]
    fn test_update_from_response() {
        let limits = ExchangePositionLimits::new();
        // mark_px = 100000, so 100000 USD = 1.0 BTC, 50000 USD = 0.5 BTC
        let response = make_response(1.5, 1.5, 100000.0, 50000.0);

        limits.update_from_response(&response, 2.0);

        assert!(limits.is_initialized());
        assert!(!limits.is_stale());
        assert_eq!(limits.max_long(), 1.5);
        assert_eq!(limits.max_short(), 1.5);
        assert_eq!(limits.available_buy(), 1.0); // 100000 / 100000 = 1.0
        assert_eq!(limits.available_sell(), 0.5); // 50000 / 100000 = 0.5
                                                  // Effective limits are min(local, max_pos, available)
                                                  // min(2.0, 1.5, 1.0) = 1.0 for bids
                                                  // min(2.0, 1.5, 0.5) = 0.5 for asks
        assert_eq!(limits.effective_bid_limit(), 1.0);
        assert_eq!(limits.effective_ask_limit(), 0.5);
    }

    #[test]
    fn test_local_max_constrains() {
        let limits = ExchangePositionLimits::new();
        // mark_px = 100000, so 100000 USD = 1.0 BTC
        let response = make_response(1.5, 1.5, 100000.0, 100000.0);

        // Local max (0.5) is smaller than exchange limits (1.5, 1.0)
        limits.update_from_response(&response, 0.5);

        // min(0.5, 1.5, 1.0) = 0.5 for both
        assert_eq!(limits.effective_bid_limit(), 0.5);
        assert_eq!(limits.effective_ask_limit(), 0.5);
    }

    #[test]
    fn test_would_exceed_buy() {
        let limits = ExchangePositionLimits::new();
        // mark_px = 100000, so 50000 USD = 0.5 BTC, 100000 USD = 1.0 BTC
        let response = make_response(1.5, 1.5, 50000.0, 100000.0);
        limits.update_from_response(&response, 2.0);

        // Buy within limit (available_buy = 0.5)
        let (exceed, _) = limits.would_exceed_limit(0.3, true, 0.0);
        assert!(!exceed);

        // Buy exceeds available (0.6 > 0.5)
        let (exceed, reason) = limits.would_exceed_limit(0.6, true, 0.0);
        assert!(exceed);
        assert!(reason.unwrap().contains("available"));
    }

    #[test]
    fn test_would_exceed_sell() {
        let limits = ExchangePositionLimits::new();
        // mark_px = 100000, so 100000 USD = 1.0 BTC, 30000 USD = 0.3 BTC
        let response = make_response(1.5, 1.5, 100000.0, 30000.0);
        limits.update_from_response(&response, 2.0);

        // Sell within limit (available_sell = 0.3)
        let (exceed, _) = limits.would_exceed_limit(0.2, false, 0.0);
        assert!(!exceed);

        // Sell exceeds available (0.5 > 0.3)
        let (exceed, reason) = limits.would_exceed_limit(0.5, false, 0.0);
        assert!(exceed);
        assert!(reason.unwrap().contains("available"));
    }

    #[test]
    fn test_update_local_max() {
        let limits = ExchangePositionLimits::new();
        // mark_px = 100000, so 100000 USD = 1.0 BTC
        // max_long=1.5, max_short=1.5, available_buy=100000 USD (=1.0 BTC), available_sell=100000 USD (=1.0 BTC)
        let response = make_response(1.5, 1.5, 100000.0, 100000.0);
        limits.update_from_response(&response, 2.0);

        // min(2.0, 1.5, 1.0) = 1.0
        assert_eq!(limits.effective_bid_limit(), 1.0);

        // Update local max to smaller value — should use 3-factor formula:
        // effective_bid = min(local_max=0.3, max_long=1.5, available_buy=1.0) = 0.3
        limits.update_local_max(0.3);
        assert_eq!(limits.effective_bid_limit(), 0.3);
        assert_eq!(limits.effective_ask_limit(), 0.3);

        // Update local max to larger value — max_long/max_short still constrain:
        // effective_bid = min(local_max=5.0, max_long=1.5, available_buy=1.0) = 1.0
        limits.update_local_max(5.0);
        assert_eq!(limits.effective_bid_limit(), 1.0);
        assert_eq!(limits.effective_ask_limit(), 1.0);
    }

    #[test]
    fn test_max_trade_szs_constrains() {
        // Test that max_trade_szs (position limits) properly constrain effective limits
        let limits = ExchangePositionLimits::new();
        // max_long = 0.1, max_short = 0.05
        // available_usd = 1000000 -> 10.0 BTC (much larger than max)
        let response = make_response(0.1, 0.05, 1000000.0, 1000000.0);
        limits.update_from_response(&response, 5.0); // local max = 5.0

        // Effective should be min(local=5.0, max_long=0.1, available=10.0) = 0.1
        assert_eq!(limits.effective_bid_limit(), 0.1);
        // Effective should be min(local=5.0, max_short=0.05, available=10.0) = 0.05
        assert_eq!(limits.effective_ask_limit(), 0.05);
    }

    #[test]
    fn test_atomic_f64() {
        let af = AtomicF64::new(1.5);
        assert_eq!(af.load(), 1.5);

        af.store(2.5);
        assert_eq!(af.load(), 2.5);
    }

    // =========================================================================
    // parse_f64_opt tests
    // =========================================================================

    #[test]
    fn test_parse_f64_opt_valid() {
        let arr = vec!["1.5".to_string(), "100.0".to_string()];
        assert_eq!(parse_f64_opt(&arr, 0), Some(1.5));
        assert_eq!(parse_f64_opt(&arr, 1), Some(100.0));
        // Zero is valid (means no capacity)
        let arr_zero = vec!["0.0".to_string()];
        assert_eq!(parse_f64_opt(&arr_zero, 0), Some(0.0));
    }

    #[test]
    fn test_parse_f64_opt_empty_string() {
        let arr = vec!["".to_string()];
        assert_eq!(parse_f64_opt(&arr, 0), None);
    }

    #[test]
    fn test_parse_f64_opt_missing_index() {
        let arr: Vec<String> = vec![];
        assert_eq!(parse_f64_opt(&arr, 0), None);
        let arr2 = vec!["1.0".to_string()];
        assert_eq!(parse_f64_opt(&arr2, 1), None);
    }

    #[test]
    fn test_parse_f64_opt_negative() {
        let arr = vec!["-5.0".to_string()];
        assert_eq!(parse_f64_opt(&arr, 0), None);
    }

    #[test]
    fn test_parse_f64_opt_infinity() {
        let arr = vec!["inf".to_string(), "Infinity".to_string()];
        assert_eq!(parse_f64_opt(&arr, 0), None);
        assert_eq!(parse_f64_opt(&arr, 1), None);
    }

    #[test]
    fn test_parse_f64_opt_nan() {
        let arr = vec!["NaN".to_string()];
        assert_eq!(parse_f64_opt(&arr, 0), None);
    }

    #[test]
    fn test_parse_f64_opt_str_garbage() {
        assert_eq!(parse_f64_opt_str("abc"), None);
        assert_eq!(parse_f64_opt_str(""), None);
        assert_eq!(parse_f64_opt_str("--3"), None);
    }

    // =========================================================================
    // Fallback behavior tests
    // =========================================================================

    #[test]
    fn test_update_from_response_valid_data_has_full_flag() {
        let limits = ExchangePositionLimits::new();
        let response = make_response(1.5, 1.5, 100000.0, 50000.0);
        limits.update_from_response(&response, 2.0);

        // Standard perp data: all fields parse, flag should be true
        assert!(limits.has_full_exchange_data());
        assert_eq!(limits.max_long(), 1.5);
        assert_eq!(limits.available_buy(), 1.0); // 100000 / 100000
    }

    /// Simulates HIP-3 asset where API returns empty fields.
    #[test]
    fn test_update_from_response_missing_all_fields() {
        use crate::types::Leverage;
        use alloy::primitives::Address;

        let limits = ExchangePositionLimits::new();
        let response = ActiveAssetDataResponse {
            user: Address::ZERO,
            coin: "HYPE".to_string(),
            leverage: Leverage {
                type_string: "cross".to_string(),
                value: 5,
                raw_usd: None,
            },
            max_trade_szs: vec![],      // empty array
            available_to_trade: vec![], // empty array
            mark_px: "".to_string(),    // empty mark price
        };

        let local_max = 10.0;
        limits.update_from_response(&response, local_max);

        // All fields should fall back to local_max_position, NOT f64::MAX
        assert!(!limits.has_full_exchange_data());
        assert_eq!(limits.max_long(), local_max);
        assert_eq!(limits.max_short(), local_max);
        assert_eq!(limits.available_buy(), local_max);
        assert_eq!(limits.available_sell(), local_max);
        assert_eq!(limits.effective_bid_limit(), local_max);
        assert_eq!(limits.effective_ask_limit(), local_max);
    }

    /// Simulates partial data: max_trade_szs present but available_to_trade or mark_px missing.
    #[test]
    fn test_update_from_response_partial_data_mark_px_missing() {
        use crate::types::Leverage;
        use alloy::primitives::Address;

        let limits = ExchangePositionLimits::new();
        let response = ActiveAssetDataResponse {
            user: Address::ZERO,
            coin: "HYPE".to_string(),
            leverage: Leverage {
                type_string: "cross".to_string(),
                value: 5,
                raw_usd: None,
            },
            max_trade_szs: vec!["50.0".to_string(), "50.0".to_string()],
            available_to_trade: vec!["10000.0".to_string(), "10000.0".to_string()],
            mark_px: "".to_string(), // mark price missing — can't convert USD to contracts
        };

        let local_max = 20.0;
        limits.update_from_response(&response, local_max);

        // max_long/max_short parsed fine
        assert_eq!(limits.max_long(), 50.0);
        assert_eq!(limits.max_short(), 50.0);
        // available_buy/sell can't be computed without mark_px → fall back to local_max
        assert_eq!(limits.available_buy(), local_max);
        assert_eq!(limits.available_sell(), local_max);
        // Effective = min(local_max=20, max_long=50, available=20) = 20
        assert_eq!(limits.effective_bid_limit(), local_max);
        assert_eq!(limits.effective_ask_limit(), local_max);
        // Partial data
        assert!(!limits.has_full_exchange_data());
    }

    #[test]
    fn test_update_from_ws_missing_fields_uses_fallback() {
        use crate::types::Leverage;
        use alloy::primitives::Address;

        let limits = ExchangePositionLimits::new();

        let data = crate::types::ActiveAssetDataData {
            user: Address::ZERO,
            coin: "HYPE".to_string(),
            leverage: Leverage {
                type_string: "cross".to_string(),
                value: 5,
                raw_usd: None,
            },
            max_trade_szs: vec![],
            available_to_trade: vec![],
        };

        let local_max = 15.0;
        limits.update_from_ws(&data, 0.0, local_max); // mark_price = 0.0 → invalid

        assert!(!limits.has_full_exchange_data());
        assert_eq!(limits.max_long(), local_max);
        assert_eq!(limits.max_short(), local_max);
        assert_eq!(limits.available_buy(), local_max);
        assert_eq!(limits.available_sell(), local_max);
        assert_eq!(limits.effective_bid_limit(), local_max);
        assert_eq!(limits.effective_ask_limit(), local_max);
    }

    #[test]
    fn test_paper_init_has_full_data() {
        let limits = ExchangePositionLimits::new();
        limits.initialize_for_paper(100.0);
        assert!(limits.has_full_exchange_data());
        assert!(limits.is_initialized());
        assert_eq!(limits.effective_bid_limit(), 100.0);
    }
}
