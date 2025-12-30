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
    /// Available capacity to buy (available_to_trade[0])
    available_buy: AtomicF64,
    /// Available capacity to sell (available_to_trade[1])
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
}

impl ExchangePositionLimits {
    /// Create new exchange position limits (uninitialized).
    ///
    /// Call `update_from_response` before using in production.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ExchangeLimitsInner {
                max_long: AtomicF64::new(f64::MAX),
                max_short: AtomicF64::new(f64::MAX),
                available_buy: AtomicF64::new(f64::MAX),
                available_sell: AtomicF64::new(f64::MAX),
                effective_bid_limit: AtomicF64::new(f64::MAX),
                effective_ask_limit: AtomicF64::new(f64::MAX),
                last_refresh_epoch_ms: AtomicU64::new(0),
                local_max_position: AtomicF64::new(0.0),
            }),
        }
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
        // Parse exchange limits
        let max_long = parse_f64_or_max(&response.max_trade_szs, 0);
        let max_short = parse_f64_or_max(&response.max_trade_szs, 1);
        let available_buy = parse_f64_or_max(&response.available_to_trade, 0);
        let available_sell = parse_f64_or_max(&response.available_to_trade, 1);

        // Store raw values
        self.inner.max_long.store(max_long);
        self.inner.max_short.store(max_short);
        self.inner.available_buy.store(available_buy);
        self.inner.available_sell.store(available_sell);

        // Pre-compute effective limits (avoids branching in hot path)
        let effective_bid = local_max_position.min(available_buy);
        let effective_ask = local_max_position.min(available_sell);
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
            "Exchange position limits updated"
        );
    }

    /// Update only the local max position (re-computes effective limits).
    ///
    /// Use when local config changes but exchange limits are still valid.
    pub fn update_local_max(&self, local_max_position: f64) {
        let available_buy = self.inner.available_buy.load();
        let available_sell = self.inner.available_sell.load();

        self.inner
            .effective_bid_limit
            .store(local_max_position.min(available_buy));
        self.inner
            .effective_ask_limit
            .store(local_max_position.min(available_sell));
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
    pub fn age_ms(&self) -> u64 {
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
            // Not initialized - allow optimistically but warn
            return (requested_size, false, None);
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
            return (false, None); // Can't check, allow optimistically
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
        }
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
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Parse f64 from string array, returning MAX on error.
fn parse_f64_or_max(arr: &[String], index: usize) -> f64 {
    arr.get(index)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(f64::MAX)
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

    fn make_response(
        max_long: f64,
        max_short: f64,
        avail_buy: f64,
        avail_sell: f64,
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
            available_to_trade: vec![avail_buy.to_string(), avail_sell.to_string()],
            mark_px: "100000.0".to_string(),
        }
    }

    #[test]
    fn test_new_uninitialized() {
        let limits = ExchangePositionLimits::new();
        assert!(!limits.is_initialized());
        assert!(limits.is_stale());
        assert_eq!(limits.effective_bid_limit(), f64::MAX);
    }

    #[test]
    fn test_update_from_response() {
        let limits = ExchangePositionLimits::new();
        let response = make_response(1.5, 1.5, 1.0, 0.5);

        limits.update_from_response(&response, 2.0);

        assert!(limits.is_initialized());
        assert!(!limits.is_stale());
        assert_eq!(limits.max_long(), 1.5);
        assert_eq!(limits.max_short(), 1.5);
        assert_eq!(limits.available_buy(), 1.0);
        assert_eq!(limits.available_sell(), 0.5);
        // Effective limits are min(local, exchange)
        assert_eq!(limits.effective_bid_limit(), 1.0); // min(2.0, 1.0)
        assert_eq!(limits.effective_ask_limit(), 0.5); // min(2.0, 0.5)
    }

    #[test]
    fn test_local_max_constrains() {
        let limits = ExchangePositionLimits::new();
        let response = make_response(1.5, 1.5, 1.0, 1.0);

        // Local max is smaller than exchange limits
        limits.update_from_response(&response, 0.5);

        assert_eq!(limits.effective_bid_limit(), 0.5); // min(0.5, 1.0)
        assert_eq!(limits.effective_ask_limit(), 0.5); // min(0.5, 1.0)
    }

    #[test]
    fn test_would_exceed_buy() {
        let limits = ExchangePositionLimits::new();
        let response = make_response(1.5, 1.5, 0.5, 1.0);
        limits.update_from_response(&response, 2.0);

        // Buy within limit
        let (exceed, _) = limits.would_exceed_limit(0.3, true, 0.0);
        assert!(!exceed);

        // Buy exceeds available
        let (exceed, reason) = limits.would_exceed_limit(0.6, true, 0.0);
        assert!(exceed);
        assert!(reason.unwrap().contains("available"));
    }

    #[test]
    fn test_would_exceed_sell() {
        let limits = ExchangePositionLimits::new();
        let response = make_response(1.5, 1.5, 1.0, 0.3);
        limits.update_from_response(&response, 2.0);

        // Sell within limit
        let (exceed, _) = limits.would_exceed_limit(0.2, false, 0.0);
        assert!(!exceed);

        // Sell exceeds available
        let (exceed, reason) = limits.would_exceed_limit(0.5, false, 0.0);
        assert!(exceed);
        assert!(reason.unwrap().contains("available"));
    }

    #[test]
    fn test_update_local_max() {
        let limits = ExchangePositionLimits::new();
        let response = make_response(1.5, 1.5, 1.0, 1.0);
        limits.update_from_response(&response, 2.0);

        assert_eq!(limits.effective_bid_limit(), 1.0);

        // Update local max to smaller value
        limits.update_local_max(0.3);
        assert_eq!(limits.effective_bid_limit(), 0.3);
        assert_eq!(limits.effective_ask_limit(), 0.3);
    }

    #[test]
    fn test_atomic_f64() {
        let af = AtomicF64::new(1.5);
        assert_eq!(af.load(), 1.5);

        af.store(2.5);
        assert_eq!(af.load(), 2.5);
    }
}
