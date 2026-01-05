//! Volatility-weighted asset allocator for multi-asset market making.
//!
//! Allocates order budget across multiple assets using inverse volatility weighting,
//! ensuring lower-volatility assets receive more orders (higher capital efficiency).
//!
//! # Algorithm
//!
//! 1. Compute inverse volatility weights: `weight_i = 1/σ_i`
//! 2. Normalize weights: `weight_i /= Σ weights`
//! 3. Enforce concentration cap: `weight_i = min(weight_i, max_concentration)`
//! 4. Allocate minimum levels to all assets
//! 5. Distribute remaining budget by weight
//! 6. Apply regime scaling (high vol → more levels for defense)
//!
//! # Example
//!
//! ```ignore
//! let config = AllocationConfig::default();
//! let mut allocator = AssetAllocator::new(config);
//!
//! // Rebalance with current volatilities
//! let vols = hashmap! {
//!     btc_id => 0.0015,  // Lower vol → higher weight
//!     eth_id => 0.0025,  // Higher vol → lower weight
//! };
//! allocator.rebalance(&vols);
//!
//! // Get allocation for an asset
//! let budget = allocator.get_budget(btc_id);
//! assert!(budget.allocated_levels > 5);  // BTC gets more levels due to lower vol
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::market_maker::estimator::VolatilityRegime;
use crate::market_maker::tracking::AssetId;

/// Configuration for the asset allocator.
#[derive(Debug, Clone)]
pub struct AllocationConfig {
    /// Total order limit across all assets (default: 1000).
    pub total_order_limit: usize,

    /// Minimum levels per asset (default: 5).
    pub min_levels_per_asset: usize,

    /// Maximum levels per asset (default: 25).
    pub max_levels_per_asset: usize,

    /// Maximum concentration per asset (default: 0.30 = 30%).
    pub max_concentration: f64,

    /// Rebalance interval (default: 5 minutes).
    pub rebalance_interval: Duration,

    /// Minimum volatility for weight calculation (prevents division by zero).
    pub min_volatility: f64,
}

impl Default for AllocationConfig {
    fn default() -> Self {
        Self {
            total_order_limit: 1000,
            min_levels_per_asset: 5,
            max_levels_per_asset: 25,
            max_concentration: 0.30,
            rebalance_interval: Duration::from_secs(300),
            min_volatility: 0.0001, // 1 bp minimum
        }
    }
}

/// Budget allocation for a single asset.
#[derive(Debug, Clone)]
pub struct AssetBudget {
    /// Asset identifier.
    pub asset_id: AssetId,

    /// Base minimum levels (always allocated).
    pub base_levels: usize,

    /// Current allocated levels (may exceed base due to weight).
    pub allocated_levels: usize,

    /// Normalized inverse-volatility weight (0.0-1.0).
    pub weight: f64,

    /// Current volatility regime.
    pub regime: VolatilityRegime,

    /// Orders currently in use (tracking).
    pub orders_in_use: usize,

    /// Last volatility reading.
    pub last_sigma: f64,
}

impl AssetBudget {
    /// Create a new budget with minimum allocation.
    pub fn new(asset_id: AssetId, min_levels: usize) -> Self {
        Self {
            asset_id,
            base_levels: min_levels,
            allocated_levels: min_levels,
            weight: 0.0,
            regime: VolatilityRegime::Normal,
            orders_in_use: 0,
            last_sigma: 0.0,
        }
    }

    /// Total orders for this asset (levels × 2 sides).
    pub fn total_orders(&self) -> usize {
        self.allocated_levels * 2
    }

    /// Available orders (allocated - in use).
    pub fn available_orders(&self) -> usize {
        self.total_orders().saturating_sub(self.orders_in_use)
    }
}

/// Manages order budget allocation across multiple assets.
pub struct AssetAllocator {
    /// Configuration.
    config: AllocationConfig,

    /// Per-asset budgets.
    budgets: HashMap<AssetId, AssetBudget>,

    /// Ordered list of asset IDs (for deterministic iteration).
    asset_order: Vec<AssetId>,

    /// Last rebalance timestamp.
    last_rebalance: Instant,

    /// Total orders currently allocated.
    total_allocated: usize,
}

impl AssetAllocator {
    /// Create a new allocator with the given configuration.
    pub fn new(config: AllocationConfig) -> Self {
        Self {
            config,
            budgets: HashMap::new(),
            asset_order: Vec::new(),
            last_rebalance: Instant::now(),
            total_allocated: 0,
        }
    }

    /// Add an asset to the allocator.
    pub fn add_asset(&mut self, asset_id: AssetId) {
        if self.budgets.contains_key(&asset_id) {
            return;
        }

        let budget = AssetBudget::new(asset_id, self.config.min_levels_per_asset);
        self.budgets.insert(asset_id, budget);
        self.asset_order.push(asset_id);
    }

    /// Remove an asset from the allocator.
    pub fn remove_asset(&mut self, asset_id: AssetId) {
        self.budgets.remove(&asset_id);
        self.asset_order.retain(|&id| id != asset_id);
    }

    /// Check if rebalance is due.
    pub fn should_rebalance(&self) -> bool {
        self.last_rebalance.elapsed() >= self.config.rebalance_interval
    }

    /// Rebalance allocations based on current volatilities.
    ///
    /// # Arguments
    ///
    /// * `volatilities` - Current per-second volatility (σ) for each asset
    pub fn rebalance(&mut self, volatilities: &HashMap<AssetId, f64>) {
        if self.budgets.is_empty() {
            return;
        }

        // Step 1: Compute inverse volatility weights
        let mut weights: HashMap<AssetId, f64> = HashMap::new();
        let mut total_weight = 0.0;

        for &asset_id in &self.asset_order {
            let sigma = volatilities
                .get(&asset_id)
                .copied()
                .unwrap_or(self.config.min_volatility)
                .max(self.config.min_volatility);

            let inv_weight = 1.0 / sigma;
            weights.insert(asset_id, inv_weight);
            total_weight += inv_weight;

            // Update last sigma in budget
            if let Some(budget) = self.budgets.get_mut(&asset_id) {
                budget.last_sigma = sigma;
            }
        }

        // Step 2: Normalize weights with concentration cap
        for weight in weights.values_mut() {
            *weight /= total_weight;
            *weight = weight.min(self.config.max_concentration);
        }

        // Renormalize after capping
        let capped_sum: f64 = weights.values().sum();
        for weight in weights.values_mut() {
            *weight /= capped_sum;
        }

        // Step 3: Calculate minimum allocation
        let n_assets = self.budgets.len();
        let min_orders_per_asset = self.config.min_levels_per_asset * 2;
        let total_min_orders = n_assets * min_orders_per_asset;

        // Step 4: Handle case where minimum exceeds limit
        if total_min_orders >= self.config.total_order_limit {
            // Distribute evenly at reduced minimum
            let effective_min = self.config.total_order_limit / n_assets / 2;
            for budget in self.budgets.values_mut() {
                budget.allocated_levels = effective_min.max(1);
                budget.weight = 1.0 / n_assets as f64;
            }
            self.total_allocated = n_assets * effective_min * 2;
            self.last_rebalance = Instant::now();
            return;
        }

        // Step 5: Distribute remaining by weight
        let remaining = self.config.total_order_limit - total_min_orders;

        self.total_allocated = 0;
        for &asset_id in &self.asset_order {
            if let Some(budget) = self.budgets.get_mut(&asset_id) {
                let weight = weights.get(&asset_id).copied().unwrap_or(0.0);
                let extra_orders = (remaining as f64 * weight) as usize;
                let extra_levels = extra_orders / 2;

                budget.weight = weight;
                budget.allocated_levels = (self.config.min_levels_per_asset + extra_levels)
                    .min(self.config.max_levels_per_asset);

                self.total_allocated += budget.total_orders();
            }
        }

        self.last_rebalance = Instant::now();
    }

    /// Apply regime-based scaling to an asset's allocation.
    ///
    /// High volatility regimes get more levels for defensive depth.
    pub fn apply_regime_scaling(&mut self, asset_id: AssetId, regime: VolatilityRegime) {
        if let Some(budget) = self.budgets.get_mut(&asset_id) {
            let base = budget.allocated_levels;
            budget.regime = regime;

            let scaled = match regime {
                VolatilityRegime::Low => base,
                VolatilityRegime::Normal => base,
                VolatilityRegime::High => (base * 3 / 2).min(self.config.max_levels_per_asset),
                VolatilityRegime::Extreme => (base * 2).min(self.config.max_levels_per_asset),
            };

            // Update total allocated
            self.total_allocated -= budget.total_orders();
            budget.allocated_levels = scaled;
            self.total_allocated += budget.total_orders();
        }
    }

    /// Get the budget for an asset.
    pub fn get_budget(&self, asset_id: AssetId) -> Option<&AssetBudget> {
        self.budgets.get(&asset_id)
    }

    /// Get mutable budget for an asset.
    pub fn get_budget_mut(&mut self, asset_id: AssetId) -> Option<&mut AssetBudget> {
        self.budgets.get_mut(&asset_id)
    }

    /// Get all budgets.
    pub fn budgets(&self) -> impl Iterator<Item = &AssetBudget> {
        self.budgets.values()
    }

    /// Get ordered asset IDs.
    pub fn asset_ids(&self) -> &[AssetId] {
        &self.asset_order
    }

    /// Total orders currently allocated.
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Orders remaining (limit - allocated).
    pub fn orders_remaining(&self) -> usize {
        self.config.total_order_limit.saturating_sub(self.total_allocated)
    }

    /// Number of assets.
    pub fn asset_count(&self) -> usize {
        self.budgets.len()
    }

    /// Update orders in use for an asset.
    pub fn set_orders_in_use(&mut self, asset_id: AssetId, count: usize) {
        if let Some(budget) = self.budgets.get_mut(&asset_id) {
            budget.orders_in_use = count;
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &AllocationConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_allocator() -> AssetAllocator {
        let config = AllocationConfig {
            total_order_limit: 100,
            min_levels_per_asset: 5,
            max_levels_per_asset: 20,
            max_concentration: 0.40,
            rebalance_interval: Duration::from_secs(1),
            min_volatility: 0.0001,
        };
        AssetAllocator::new(config)
    }

    #[test]
    fn test_add_remove_asset() {
        let mut allocator = make_test_allocator();
        let btc = AssetId::new("BTC", None);
        let eth = AssetId::new("ETH", None);

        allocator.add_asset(btc);
        allocator.add_asset(eth);

        assert_eq!(allocator.asset_count(), 2);

        allocator.remove_asset(btc);
        assert_eq!(allocator.asset_count(), 1);
    }

    #[test]
    fn test_inverse_volatility_weighting() {
        let mut allocator = make_test_allocator();
        let btc = AssetId::new("BTC", None);
        let eth = AssetId::new("ETH", None);

        allocator.add_asset(btc);
        allocator.add_asset(eth);

        // BTC has lower volatility (15 bps) → higher weight
        // ETH has higher volatility (30 bps) → lower weight
        let vols = HashMap::from([(btc, 0.0015), (eth, 0.0030)]);
        allocator.rebalance(&vols);

        let btc_budget = allocator.get_budget(btc).unwrap();
        let eth_budget = allocator.get_budget(eth).unwrap();

        // BTC should have higher weight (inverse vol)
        assert!(btc_budget.weight > eth_budget.weight);

        // BTC should have more levels allocated
        assert!(btc_budget.allocated_levels >= eth_budget.allocated_levels);
    }

    #[test]
    fn test_concentration_cap() {
        let mut allocator = make_test_allocator();
        let btc = AssetId::new("BTC", None);
        let eth = AssetId::new("ETH", None);

        allocator.add_asset(btc);
        allocator.add_asset(eth);

        // Extreme volatility difference (BTC 10x lower than ETH)
        let vols = HashMap::from([(btc, 0.001), (eth, 0.010)]);
        allocator.rebalance(&vols);

        let btc_budget = allocator.get_budget(btc).unwrap();

        // BTC weight should be capped at 40% (config)
        assert!(btc_budget.weight <= 0.40 + 0.01); // Small tolerance
    }

    #[test]
    fn test_minimum_allocation() {
        let mut allocator = make_test_allocator();
        let btc = AssetId::new("BTC", None);
        let eth = AssetId::new("ETH", None);

        allocator.add_asset(btc);
        allocator.add_asset(eth);

        let vols = HashMap::from([(btc, 0.001), (eth, 0.001)]);
        allocator.rebalance(&vols);

        // Both should have at least minimum levels
        let btc_budget = allocator.get_budget(btc).unwrap();
        let eth_budget = allocator.get_budget(eth).unwrap();

        assert!(btc_budget.allocated_levels >= 5);
        assert!(eth_budget.allocated_levels >= 5);
    }

    #[test]
    fn test_regime_scaling() {
        let mut allocator = make_test_allocator();
        let btc = AssetId::new("BTC", None);

        allocator.add_asset(btc);

        let vols = HashMap::from([(btc, 0.001)]);
        allocator.rebalance(&vols);

        let base_levels = allocator.get_budget(btc).unwrap().allocated_levels;

        // Apply high volatility regime
        allocator.apply_regime_scaling(btc, VolatilityRegime::High);

        let scaled_levels = allocator.get_budget(btc).unwrap().allocated_levels;

        // High regime should increase levels (1.5x)
        assert!(scaled_levels >= base_levels);
    }

    #[test]
    fn test_total_allocation_within_limit() {
        let mut allocator = make_test_allocator();
        let btc = AssetId::new("BTC", None);
        let eth = AssetId::new("ETH", None);
        let sol = AssetId::new("SOL", None);

        allocator.add_asset(btc);
        allocator.add_asset(eth);
        allocator.add_asset(sol);

        let vols = HashMap::from([(btc, 0.001), (eth, 0.002), (sol, 0.003)]);
        allocator.rebalance(&vols);

        // Total allocated should not exceed limit
        assert!(allocator.total_allocated() <= 100);
    }

    #[test]
    fn test_too_many_assets() {
        let config = AllocationConfig {
            total_order_limit: 20, // Very small limit
            min_levels_per_asset: 5,
            ..Default::default()
        };
        let mut allocator = AssetAllocator::new(config);

        // Add 3 assets (would require 30 min orders, but limit is 20)
        let btc = AssetId::new("BTC", None);
        let eth = AssetId::new("ETH", None);
        let sol = AssetId::new("SOL", None);

        allocator.add_asset(btc);
        allocator.add_asset(eth);
        allocator.add_asset(sol);

        let vols = HashMap::from([(btc, 0.001), (eth, 0.002), (sol, 0.003)]);
        allocator.rebalance(&vols);

        // Should handle gracefully (reduced minimum)
        assert!(allocator.total_allocated() <= 20);
        assert!(allocator.get_budget(btc).unwrap().allocated_levels >= 1);
    }
}
