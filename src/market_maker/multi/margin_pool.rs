//! Shared margin pool for multi-asset market making.
//!
//! Tracks margin usage across all assets and enforces concentration limits
//! to prevent any single asset from consuming too much of the capital pool.
//!
//! # Design
//!
//! The margin pool implements two key constraints:
//! 1. **Total margin limit**: Prevents over-leveraging (default 80% utilization)
//! 2. **Per-asset concentration**: No single asset > 30% of total capital
//!
//! # Example
//!
//! ```ignore
//! let mut pool = SharedMarginPool::new(10000.0, 0.30);
//!
//! // Check if we can allocate margin for an asset
//! if pool.can_allocate(btc_id, 2000.0) {
//!     pool.allocate(btc_id, 2000.0);
//! }
//!
//! // Get quoting capacity for an asset
//! let capacity = pool.quoting_capacity(btc_id, 50000.0, 10.0);
//! ```

use std::collections::HashMap;

use crate::market_maker::tracking::AssetId;

/// Shared margin pool for cross-asset capital management.
#[derive(Debug, Clone)]
pub struct SharedMarginPool {
    /// Total account value (equity).
    total_account_value: f64,

    /// Total margin currently in use across all assets.
    total_margin_used: f64,

    /// Margin usage per asset.
    per_asset_usage: HashMap<AssetId, f64>,

    /// Maximum margin utilization (default: 0.80 = 80%).
    max_utilization: f64,

    /// Maximum per-asset concentration (default: 0.30 = 30%).
    max_per_asset_pct: f64,

    /// Reduce-only threshold (default: 0.80 = 80%).
    reduce_only_threshold: f64,
}

impl SharedMarginPool {
    /// Create a new margin pool.
    ///
    /// # Arguments
    ///
    /// * `account_value` - Total account equity
    /// * `max_per_asset_pct` - Maximum concentration per asset (0.0-1.0)
    pub fn new(account_value: f64, max_per_asset_pct: f64) -> Self {
        Self {
            total_account_value: account_value,
            total_margin_used: 0.0,
            per_asset_usage: HashMap::new(),
            max_utilization: 0.80,
            max_per_asset_pct,
            reduce_only_threshold: 0.80,
        }
    }

    /// Create with default settings.
    pub fn with_defaults(account_value: f64) -> Self {
        Self::new(account_value, 0.30)
    }

    /// Update the total account value.
    pub fn set_account_value(&mut self, value: f64) {
        self.total_account_value = value;
    }

    /// Get total account value.
    pub fn account_value(&self) -> f64 {
        self.total_account_value
    }

    /// Get total margin currently in use.
    pub fn margin_used(&self) -> f64 {
        self.total_margin_used
    }

    /// Get available margin (total - used).
    pub fn margin_available(&self) -> f64 {
        (self.total_account_value * self.max_utilization - self.total_margin_used).max(0.0)
    }

    /// Get current margin utilization (0.0-1.0).
    pub fn utilization(&self) -> f64 {
        if self.total_account_value <= 0.0 {
            return 1.0;
        }
        self.total_margin_used / self.total_account_value
    }

    /// Check if we should enter reduce-only mode.
    pub fn is_reduce_only(&self) -> bool {
        self.utilization() >= self.reduce_only_threshold
    }

    /// Get margin usage for a specific asset.
    pub fn asset_usage(&self, asset_id: AssetId) -> f64 {
        self.per_asset_usage.get(&asset_id).copied().unwrap_or(0.0)
    }

    /// Get concentration for a specific asset (0.0-1.0).
    pub fn asset_concentration(&self, asset_id: AssetId) -> f64 {
        if self.total_account_value <= 0.0 {
            return 0.0;
        }
        self.asset_usage(asset_id) / self.total_account_value
    }

    /// Check if we can allocate additional margin for an asset.
    pub fn can_allocate(&self, asset_id: AssetId, amount: f64) -> bool {
        // Check total margin limit
        if self.total_margin_used + amount > self.total_account_value * self.max_utilization {
            return false;
        }

        // Check per-asset concentration limit
        let current = self.asset_usage(asset_id);
        if current + amount > self.total_account_value * self.max_per_asset_pct {
            return false;
        }

        true
    }

    /// Allocate margin for an asset.
    pub fn allocate(&mut self, asset_id: AssetId, amount: f64) {
        let current = self.per_asset_usage.entry(asset_id).or_insert(0.0);
        *current += amount;
        self.total_margin_used += amount;
    }

    /// Release margin for an asset.
    pub fn release(&mut self, asset_id: AssetId, amount: f64) {
        let current = self.per_asset_usage.entry(asset_id).or_insert(0.0);
        let release_amount = amount.min(*current);
        *current -= release_amount;
        self.total_margin_used -= release_amount;

        // Clean up zero entries
        if *current <= 0.0 {
            self.per_asset_usage.remove(&asset_id);
        }
    }

    /// Set absolute margin usage for an asset (replaces previous value).
    pub fn set_usage(&mut self, asset_id: AssetId, amount: f64) {
        let old = self.per_asset_usage.insert(asset_id, amount).unwrap_or(0.0);
        self.total_margin_used = self.total_margin_used - old + amount;
    }

    /// Clear all usage for an asset.
    pub fn clear_asset(&mut self, asset_id: AssetId) {
        if let Some(amount) = self.per_asset_usage.remove(&asset_id) {
            self.total_margin_used -= amount;
        }
    }

    /// Compute quoting capacity for an asset.
    ///
    /// Returns the maximum position size in base units that can be quoted
    /// while respecting margin and concentration constraints.
    ///
    /// # Arguments
    ///
    /// * `asset_id` - Asset identifier
    /// * `mid_price` - Current mid price
    /// * `leverage` - Available leverage
    pub fn quoting_capacity(&self, asset_id: AssetId, mid_price: f64, leverage: f64) -> f64 {
        if mid_price <= 0.0 || leverage <= 0.0 {
            return 0.0;
        }

        // Available margin headroom (total)
        let available = self.margin_available();

        // Per-asset headroom
        let current_usage = self.asset_usage(asset_id);
        let max_asset_margin = self.total_account_value * self.max_per_asset_pct;
        let asset_headroom = (max_asset_margin - current_usage).max(0.0);

        // Use the smaller of the two constraints
        let margin_headroom = available.min(asset_headroom);

        // Convert margin to position size
        // position = margin * leverage / price
        // Using 50% of headroom for safety buffer
        (margin_headroom * 0.5 * leverage / mid_price).max(0.0)
    }

    /// Get the maximum position value allowed for an asset.
    pub fn max_position_value(&self, asset_id: AssetId) -> f64 {
        let current = self.asset_usage(asset_id);
        let max_allowed = self.total_account_value * self.max_per_asset_pct;
        max_allowed - current
    }

    /// Get summary of current state.
    pub fn summary(&self) -> MarginPoolSummary {
        MarginPoolSummary {
            account_value: self.total_account_value,
            margin_used: self.total_margin_used,
            margin_available: self.margin_available(),
            utilization: self.utilization(),
            is_reduce_only: self.is_reduce_only(),
            asset_count: self.per_asset_usage.len(),
        }
    }

    /// Get all per-asset usage.
    pub fn all_usage(&self) -> impl Iterator<Item = (AssetId, f64)> + '_ {
        self.per_asset_usage.iter().map(|(&id, &usage)| (id, usage))
    }

    /// Find the asset with highest concentration.
    pub fn highest_concentration_asset(&self) -> Option<(AssetId, f64)> {
        self.per_asset_usage
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&id, &usage)| (id, usage / self.total_account_value))
    }
}

/// Summary of margin pool state.
#[derive(Debug, Clone)]
pub struct MarginPoolSummary {
    pub account_value: f64,
    pub margin_used: f64,
    pub margin_available: f64,
    pub utilization: f64,
    pub is_reduce_only: bool,
    pub asset_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_allocation() {
        let mut pool = SharedMarginPool::new(10000.0, 0.30);
        let btc = AssetId::new("BTC", None);

        assert!(pool.can_allocate(btc, 2000.0));
        pool.allocate(btc, 2000.0);

        assert_eq!(pool.asset_usage(btc), 2000.0);
        assert_eq!(pool.margin_used(), 2000.0);
    }

    #[test]
    fn test_concentration_limit() {
        let mut pool = SharedMarginPool::new(10000.0, 0.30);
        let btc = AssetId::new("BTC", None);

        // Can allocate up to 30%
        assert!(pool.can_allocate(btc, 3000.0));

        // Cannot allocate more than 30%
        assert!(!pool.can_allocate(btc, 3001.0));

        pool.allocate(btc, 3000.0);

        // Now at 30%, can't add more
        assert!(!pool.can_allocate(btc, 100.0));
    }

    #[test]
    fn test_total_utilization_limit() {
        let mut pool = SharedMarginPool::new(10000.0, 0.50); // 50% per asset allowed
        let btc = AssetId::new("BTC", None);
        let eth = AssetId::new("ETH", None);

        // Allocate 4000 to BTC
        pool.allocate(btc, 4000.0);

        // Allocate 4000 to ETH
        pool.allocate(eth, 4000.0);

        // Now at 80% total, can't add more even though concentration allows it
        let sol = AssetId::new("SOL", None);
        assert!(!pool.can_allocate(sol, 1000.0));
    }

    #[test]
    fn test_release() {
        let mut pool = SharedMarginPool::new(10000.0, 0.30);
        let btc = AssetId::new("BTC", None);

        pool.allocate(btc, 2000.0);
        assert_eq!(pool.margin_used(), 2000.0);

        pool.release(btc, 1000.0);
        assert_eq!(pool.margin_used(), 1000.0);
        assert_eq!(pool.asset_usage(btc), 1000.0);
    }

    #[test]
    fn test_reduce_only() {
        let mut pool = SharedMarginPool::new(10000.0, 1.0); // No concentration limit
        let btc = AssetId::new("BTC", None);

        pool.allocate(btc, 7900.0);
        assert!(!pool.is_reduce_only());

        pool.allocate(btc, 100.0); // Now at 80%
        assert!(pool.is_reduce_only());
    }

    #[test]
    fn test_quoting_capacity() {
        let pool = SharedMarginPool::new(10000.0, 0.30);
        let btc = AssetId::new("BTC", None);

        // With 10x leverage, 50k price, and 30% concentration limit:
        // max margin = 3000, headroom = 3000 * 0.5 = 1500
        // capacity = 1500 * 10 / 50000 = 0.3 BTC
        let capacity = pool.quoting_capacity(btc, 50000.0, 10.0);
        assert!(capacity > 0.0);
        assert!(capacity < 1.0); // Should be around 0.3 BTC
    }

    #[test]
    fn test_set_usage() {
        let mut pool = SharedMarginPool::new(10000.0, 0.30);
        let btc = AssetId::new("BTC", None);

        pool.set_usage(btc, 2000.0);
        assert_eq!(pool.margin_used(), 2000.0);

        pool.set_usage(btc, 1500.0);
        assert_eq!(pool.margin_used(), 1500.0);
    }
}
