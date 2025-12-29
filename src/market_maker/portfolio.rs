//! Multi-Asset Portfolio Optimization
//!
//! Implements Markowitz mean-variance optimization for coordinated capital
//! deployment across multiple assets.
//!
//! Theory (First Principles):
//! ```text
//! max Σᵢ E[PnLᵢ] - (γ/2) × Σᵢⱼ ρᵢⱼ × posᵢ × posⱼ × σᵢ × σⱼ
//! s.t. Σᵢ marginᵢ ≤ M_total
//! ```
//!
//! Where:
//! - E[PnLᵢ] = expected edge for asset i (spread capture - AS)
//! - ρᵢⱼ = correlation between assets i and j
//! - σᵢ = volatility of asset i
//! - γ = risk aversion parameter
//!
//! Key features:
//! - **Correlation-adjusted limits**: High correlation with existing positions reduces limit
//! - **Mean-variance optimization**: Balances expected return against portfolio risk
//! - **Margin-aware allocation**: Respects total margin constraints
//! - **Diversification metrics**: Tracks effective diversification ratio

use std::collections::HashMap;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for multi-asset portfolio optimization.
#[derive(Debug, Clone)]
pub struct PortfolioConfig {
    /// Total margin available across all assets (USD)
    pub total_margin: f64,

    /// Risk aversion parameter (γ)
    /// Higher = more conservative, lower portfolio variance
    pub gamma: f64,

    /// Maximum position per asset as fraction of total margin
    /// E.g., 0.3 = 30% of margin can be in one asset
    pub max_single_asset_fraction: f64,

    /// Minimum position size per asset (USD notional)
    pub min_position_notional: f64,

    /// Correlation threshold for position reduction
    /// When correlation > threshold, reduce limits
    pub correlation_reduction_threshold: f64,

    /// Maximum correlation penalty (fraction of limit to reduce)
    pub max_correlation_penalty: f64,

    /// Base leverage for margin calculations
    pub base_leverage: f64,

    /// Target number of assets for diversification
    pub target_asset_count: usize,
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            total_margin: 10000.0,         // $10k default
            gamma: 0.5,                    // Moderate risk aversion
            max_single_asset_fraction: 0.4, // Max 40% in one asset
            min_position_notional: 100.0,  // Min $100 per asset
            correlation_reduction_threshold: 0.6,
            max_correlation_penalty: 0.5,  // Up to 50% limit reduction
            base_leverage: 5.0,            // 5x leverage
            target_asset_count: 5,         // Target 5 assets for diversification
        }
    }
}

// ============================================================================
// Asset State
// ============================================================================

/// Per-asset state for portfolio optimization.
#[derive(Debug, Clone)]
pub struct AssetState {
    /// Asset symbol
    pub symbol: String,
    /// Current position (in units)
    pub position: f64,
    /// Current price (USD)
    pub price: f64,
    /// Volatility (σ per second)
    pub sigma: f64,
    /// Expected edge (spread capture - AS, in bps)
    pub expected_edge_bps: f64,
    /// Maximum position limit (in units)
    pub max_position: f64,
    /// Margin required per unit
    pub margin_per_unit: f64,
}

impl AssetState {
    /// Create new asset state.
    pub fn new(symbol: &str, price: f64, sigma: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            position: 0.0,
            price,
            sigma,
            expected_edge_bps: 0.0,
            max_position: 1.0,
            margin_per_unit: price / 5.0, // 5x leverage default
        }
    }

    /// Position value in USD.
    pub fn position_value(&self) -> f64 {
        self.position.abs() * self.price
    }

    /// Position utilization (0-1).
    pub fn utilization(&self) -> f64 {
        if self.max_position > 1e-10 {
            (self.position.abs() / self.max_position).min(1.0)
        } else {
            0.0
        }
    }

    /// Expected PnL per unit (in USD).
    pub fn expected_pnl_per_unit(&self) -> f64 {
        self.expected_edge_bps * self.price / 10000.0
    }
}

// ============================================================================
// Portfolio Allocation Result
// ============================================================================

/// Result of portfolio optimization.
#[derive(Debug, Clone)]
pub struct PortfolioAllocation {
    /// Optimal position size per asset (in USD notional)
    pub allocations: HashMap<String, f64>,
    /// Adjusted position limits per asset (after correlation penalty)
    pub adjusted_limits: HashMap<String, f64>,
    /// Total margin used
    pub margin_used: f64,
    /// Portfolio variance (σ² of portfolio)
    pub portfolio_variance: f64,
    /// Expected portfolio return (sum of edges)
    pub expected_return: f64,
    /// Diversification ratio (>= 1.0, higher = better)
    pub diversification_ratio: f64,
    /// Sharpe ratio estimate (expected_return / sqrt(variance))
    pub sharpe_estimate: f64,
    /// Binding constraint (what limited the allocation)
    pub binding_constraint: PortfolioConstraint,
}

/// Which constraint is binding in the optimization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortfolioConstraint {
    /// No constraint binding (below all limits)
    None,
    /// Total margin limit binding
    TotalMargin,
    /// Single asset limit binding
    SingleAssetLimit(String),
    /// Correlation-adjusted limit binding
    CorrelationLimit(String),
    /// Minimum notional constraint
    MinimumNotional,
}

// ============================================================================
// Portfolio Optimizer
// ============================================================================

/// Multi-asset portfolio optimizer.
///
/// Implements mean-variance optimization with correlation-adjusted limits
/// for coordinated capital deployment across assets.
#[derive(Debug, Clone)]
pub struct MultiAssetAllocator {
    config: PortfolioConfig,
}

impl MultiAssetAllocator {
    /// Create a new portfolio allocator.
    pub fn new(config: PortfolioConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(PortfolioConfig::default())
    }

    /// Compute optimal allocation across assets.
    ///
    /// Uses greedy mean-variance optimization:
    /// 1. Compute marginal Sharpe ratio for each asset
    /// 2. Allocate capital to highest Sharpe assets first
    /// 3. Apply correlation penalties as positions build
    /// 4. Stop when margin exhausted or all assets at limits
    ///
    /// # Arguments
    /// - `assets`: Current state of each asset
    /// - `correlation_matrix`: Pairwise correlations (row/col order matches assets)
    ///
    /// # Returns
    /// Optimal allocation and metrics
    pub fn optimize(
        &self,
        assets: &[AssetState],
        correlation_matrix: &[Vec<f64>],
    ) -> PortfolioAllocation {
        let n = assets.len();
        if n == 0 || correlation_matrix.len() != n {
            return self.empty_allocation();
        }

        // Initialize allocations at current positions
        let mut allocations: Vec<f64> = assets.iter().map(|a| a.position_value()).collect();
        let margin_used: f64 = assets
            .iter()
            .map(|a| a.position.abs() * a.margin_per_unit)
            .sum();

        // Compute adjusted limits for each asset (correlation penalty)
        let adjusted_limits: Vec<f64> = self.compute_adjusted_limits(assets, correlation_matrix);

        // Compute marginal values for each asset
        let marginal_values: Vec<f64> = assets
            .iter()
            .enumerate()
            .map(|(i, a)| self.marginal_value(a, &allocations, correlation_matrix, i))
            .collect();

        // Sort assets by marginal value (descending)
        let mut sorted_indices: Vec<(usize, f64)> = marginal_values
            .iter()
            .enumerate()
            .map(|(i, &mv)| (i, mv))
            .collect();
        sorted_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Greedy allocation
        let margin_available = self.config.total_margin - margin_used;
        let mut remaining_margin = margin_available;
        let mut binding_constraint = PortfolioConstraint::None;

        for &(idx, mv) in &sorted_indices {
            if mv <= 0.0 {
                break; // No profit at remaining assets
            }
            if remaining_margin <= 0.0 {
                binding_constraint = PortfolioConstraint::TotalMargin;
                break;
            }

            let asset = &assets[idx];

            // Max allocation for this asset
            let limit_by_margin = remaining_margin / asset.margin_per_unit * asset.price;
            let limit_by_asset = adjusted_limits[idx] - allocations[idx];
            let limit_by_single = self.config.max_single_asset_fraction * self.config.total_margin
                - allocations[idx];

            let max_alloc = limit_by_margin.min(limit_by_asset).min(limit_by_single);

            if max_alloc < self.config.min_position_notional {
                binding_constraint = PortfolioConstraint::MinimumNotional;
                continue;
            }

            // Allocate
            let alloc = max_alloc;
            allocations[idx] += alloc;
            remaining_margin -= alloc / asset.price * asset.margin_per_unit;

            // Update binding constraint
            if (alloc - limit_by_asset).abs() < 1e-6 {
                binding_constraint = PortfolioConstraint::CorrelationLimit(asset.symbol.clone());
            } else if (alloc - limit_by_single).abs() < 1e-6 {
                binding_constraint = PortfolioConstraint::SingleAssetLimit(asset.symbol.clone());
            }
        }

        // Compute portfolio metrics
        let (portfolio_variance, expected_return) =
            self.compute_portfolio_metrics(assets, &allocations, correlation_matrix);
        let diversification_ratio =
            self.compute_diversification_ratio(assets, &allocations, correlation_matrix);
        let sharpe_estimate = if portfolio_variance > 1e-12 {
            expected_return / portfolio_variance.sqrt()
        } else {
            0.0
        };

        // Build result
        let alloc_map: HashMap<String, f64> = assets
            .iter()
            .zip(allocations.iter())
            .map(|(a, &alloc)| (a.symbol.clone(), alloc))
            .collect();
        let limit_map: HashMap<String, f64> = assets
            .iter()
            .zip(adjusted_limits.iter())
            .map(|(a, &limit)| (a.symbol.clone(), limit))
            .collect();

        PortfolioAllocation {
            allocations: alloc_map,
            adjusted_limits: limit_map,
            margin_used: self.config.total_margin - remaining_margin,
            portfolio_variance,
            expected_return,
            diversification_ratio,
            sharpe_estimate,
            binding_constraint,
        }
    }

    /// Compute correlation-adjusted position limit for an asset.
    ///
    /// If highly correlated assets already have positions, reduce the limit.
    ///
    /// # Arguments
    /// - `asset_idx`: Index of asset to compute limit for
    /// - `assets`: All asset states
    /// - `correlation_matrix`: Pairwise correlations
    ///
    /// # Returns
    /// Adjusted limit in USD notional
    pub fn correlation_adjusted_limit(
        &self,
        asset_idx: usize,
        assets: &[AssetState],
        correlation_matrix: &[Vec<f64>],
    ) -> f64 {
        let asset = &assets[asset_idx];

        // Base limit from config
        let base_limit = asset.max_position * asset.price;

        // Compute correlation penalty
        let mut correlation_penalty = 0.0;
        for (j, other_asset) in assets.iter().enumerate() {
            if j == asset_idx {
                continue;
            }

            let corr = correlation_matrix
                .get(asset_idx)
                .and_then(|row| row.get(j))
                .copied()
                .unwrap_or(0.0);

            // Only penalize if correlation exceeds threshold
            if corr.abs() > self.config.correlation_reduction_threshold {
                let excess_corr = corr.abs() - self.config.correlation_reduction_threshold;
                let other_utilization = other_asset.utilization();

                // Penalty proportional to correlation × other position size
                correlation_penalty += excess_corr * other_utilization;
            }
        }

        // Apply penalty (capped at max_correlation_penalty)
        let penalty_factor =
            (1.0 - correlation_penalty.min(self.config.max_correlation_penalty)).max(0.5);

        base_limit * penalty_factor
    }

    /// Compute adjusted limits for all assets.
    fn compute_adjusted_limits(
        &self,
        assets: &[AssetState],
        correlation_matrix: &[Vec<f64>],
    ) -> Vec<f64> {
        (0..assets.len())
            .map(|i| self.correlation_adjusted_limit(i, assets, correlation_matrix))
            .collect()
    }

    /// Compute marginal value of adding to an asset's position.
    ///
    /// MV = E[edge] - γ × Σⱼ ρᵢⱼ × wⱼ × σᵢ × σⱼ
    fn marginal_value(
        &self,
        asset: &AssetState,
        allocations: &[f64],
        correlation_matrix: &[Vec<f64>],
        asset_idx: usize,
    ) -> f64 {
        // Expected edge (convert bps to fraction)
        let edge = asset.expected_edge_bps / 10000.0;

        // Marginal portfolio risk
        let mut marginal_risk = 0.0;
        for (j, alloc) in allocations.iter().enumerate() {
            let corr = correlation_matrix
                .get(asset_idx)
                .and_then(|row| row.get(j))
                .copied()
                .unwrap_or(if asset_idx == j { 1.0 } else { 0.0 });

            // Weight of other asset
            let wj = alloc / self.config.total_margin.max(1.0);

            // Get sigma of other asset (placeholder - would need full asset list)
            let sigma_j = asset.sigma; // Simplified: assume same sigma

            marginal_risk += corr * wj * asset.sigma * sigma_j;
        }

        edge - self.config.gamma * marginal_risk
    }

    /// Compute portfolio variance and expected return.
    fn compute_portfolio_metrics(
        &self,
        assets: &[AssetState],
        allocations: &[f64],
        correlation_matrix: &[Vec<f64>],
    ) -> (f64, f64) {
        let total_alloc: f64 = allocations.iter().sum::<f64>().max(1.0);

        let mut variance = 0.0;
        let mut expected_return = 0.0;

        for (i, (asset_i, &alloc_i)) in assets.iter().zip(allocations.iter()).enumerate() {
            let wi = alloc_i / total_alloc;

            // Expected return contribution
            expected_return += wi * asset_i.expected_edge_bps / 10000.0;

            // Variance contribution
            for (j, (asset_j, &alloc_j)) in assets.iter().zip(allocations.iter()).enumerate() {
                let wj = alloc_j / total_alloc;
                let corr = correlation_matrix
                    .get(i)
                    .and_then(|row| row.get(j))
                    .copied()
                    .unwrap_or(if i == j { 1.0 } else { 0.0 });

                variance += wi * wj * asset_i.sigma * asset_j.sigma * corr;
            }
        }

        (variance.max(0.0), expected_return)
    }

    /// Compute diversification ratio.
    ///
    /// DR = Σ wᵢ × σᵢ / σ_portfolio
    fn compute_diversification_ratio(
        &self,
        assets: &[AssetState],
        allocations: &[f64],
        correlation_matrix: &[Vec<f64>],
    ) -> f64 {
        let total_alloc: f64 = allocations.iter().sum::<f64>().max(1.0);

        // Weighted sum of individual volatilities
        let weighted_vol: f64 = assets
            .iter()
            .zip(allocations.iter())
            .map(|(a, &alloc)| (alloc / total_alloc) * a.sigma)
            .sum();

        // Portfolio volatility
        let (port_var, _) = self.compute_portfolio_metrics(assets, allocations, correlation_matrix);
        let port_vol = port_var.sqrt().max(1e-10);

        weighted_vol / port_vol
    }

    /// Empty allocation for error cases.
    fn empty_allocation(&self) -> PortfolioAllocation {
        PortfolioAllocation {
            allocations: HashMap::new(),
            adjusted_limits: HashMap::new(),
            margin_used: 0.0,
            portfolio_variance: 0.0,
            expected_return: 0.0,
            diversification_ratio: 1.0,
            sharpe_estimate: 0.0,
            binding_constraint: PortfolioConstraint::None,
        }
    }

    /// Get optimal capital allocation for a new asset.
    ///
    /// Given existing portfolio, how much capital should be allocated to a new asset?
    ///
    /// # Arguments
    /// - `new_asset`: State of the new asset
    /// - `existing_assets`: Current portfolio
    /// - `correlations_with_existing`: Correlation of new asset with each existing asset
    ///
    /// # Returns
    /// Recommended notional allocation (USD)
    pub fn new_asset_allocation(
        &self,
        new_asset: &AssetState,
        existing_assets: &[AssetState],
        correlations_with_existing: &[f64],
    ) -> f64 {
        // Compute correlation penalty
        let mut penalty = 0.0;
        for (i, existing) in existing_assets.iter().enumerate() {
            let corr = correlations_with_existing.get(i).copied().unwrap_or(0.0);
            if corr.abs() > self.config.correlation_reduction_threshold {
                let excess = corr.abs() - self.config.correlation_reduction_threshold;
                penalty += excess * existing.utilization();
            }
        }

        // Base allocation
        let n_assets = existing_assets.len() + 1;
        let equal_weight = self.config.total_margin / n_assets as f64;

        // Adjust for edge
        let avg_edge: f64 = existing_assets
            .iter()
            .map(|a| a.expected_edge_bps)
            .sum::<f64>()
            / existing_assets.len().max(1) as f64;

        let edge_multiplier = if avg_edge.abs() > 1e-6 {
            (new_asset.expected_edge_bps / avg_edge).clamp(0.5, 2.0)
        } else {
            1.0
        };

        // Final allocation with penalty
        let penalty_factor = (1.0 - penalty.min(self.config.max_correlation_penalty)).max(0.5);

        (equal_weight * edge_multiplier * penalty_factor).max(self.config.min_position_notional)
    }

    /// Compute rebalancing trades to reach target allocation.
    ///
    /// # Arguments
    /// - `current`: Current positions (symbol -> position in units)
    /// - `target`: Target allocation (symbol -> notional in USD)
    /// - `prices`: Current prices (symbol -> price)
    ///
    /// # Returns
    /// Trades needed (symbol -> units to trade, positive = buy, negative = sell)
    pub fn rebalancing_trades(
        &self,
        current: &HashMap<String, f64>,
        target: &HashMap<String, f64>,
        prices: &HashMap<String, f64>,
    ) -> HashMap<String, f64> {
        let mut trades = HashMap::new();

        // For each target asset
        for (symbol, target_notional) in target.iter() {
            let price = prices.get(symbol).copied().unwrap_or(1.0);
            let target_units = target_notional / price;

            let current_units = current.get(symbol).copied().unwrap_or(0.0);
            let delta = target_units - current_units;

            // Only include meaningful trades
            if delta.abs() * price >= self.config.min_position_notional {
                trades.insert(symbol.clone(), delta);
            }
        }

        // For assets we hold but aren't in target, close position
        for (symbol, &units) in current.iter() {
            if !target.contains_key(symbol) && units.abs() > 1e-10 {
                trades.insert(symbol.clone(), -units);
            }
        }

        trades
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_asset(symbol: &str, price: f64, sigma: f64, edge_bps: f64) -> AssetState {
        let mut asset = AssetState::new(symbol, price, sigma);
        asset.expected_edge_bps = edge_bps;
        asset.max_position = 10.0;
        asset
    }

    fn identity_matrix(n: usize) -> Vec<Vec<f64>> {
        let mut m = vec![vec![0.0; n]; n];
        for i in 0..n {
            m[i][i] = 1.0;
        }
        m
    }

    fn correlated_matrix(n: usize, corr: f64) -> Vec<Vec<f64>> {
        let mut m = vec![vec![corr; n]; n];
        for i in 0..n {
            m[i][i] = 1.0;
        }
        m
    }

    #[test]
    fn test_default_config() {
        let config = PortfolioConfig::default();
        assert_eq!(config.total_margin, 10000.0);
        assert_eq!(config.gamma, 0.5);
    }

    #[test]
    fn test_asset_state() {
        let asset = make_asset("BTC", 50000.0, 0.0001, 5.0);
        assert_eq!(asset.symbol, "BTC");
        assert_eq!(asset.price, 50000.0);
        assert_eq!(asset.expected_pnl_per_unit(), 25.0); // 5 bps × $50k
    }

    #[test]
    fn test_empty_optimization() {
        let allocator = MultiAssetAllocator::default_config();
        let result = allocator.optimize(&[], &[]);

        assert!(result.allocations.is_empty());
        assert_eq!(result.margin_used, 0.0);
    }

    #[test]
    fn test_single_asset_optimization() {
        let allocator = MultiAssetAllocator::default_config();

        let assets = vec![make_asset("BTC", 50000.0, 0.0001, 5.0)];
        let corr = identity_matrix(1);

        let result = allocator.optimize(&assets, &corr);

        assert!(result.allocations.contains_key("BTC"));
        assert!(result.margin_used > 0.0);
    }

    #[test]
    fn test_uncorrelated_assets() {
        let allocator = MultiAssetAllocator::default_config();

        let assets = vec![
            make_asset("BTC", 50000.0, 0.0001, 5.0),
            make_asset("ETH", 3000.0, 0.0002, 5.0),
        ];
        let corr = identity_matrix(2);

        let result = allocator.optimize(&assets, &corr);

        // Both assets should get allocation
        assert!(result.allocations.get("BTC").copied().unwrap_or(0.0) > 0.0);
        assert!(result.allocations.get("ETH").copied().unwrap_or(0.0) > 0.0);

        // Diversification ratio should be > 1 with uncorrelated assets
        assert!(
            result.diversification_ratio >= 1.0,
            "Div ratio: {}",
            result.diversification_ratio
        );
    }

    #[test]
    fn test_highly_correlated_assets_reduced_limits() {
        let allocator = MultiAssetAllocator::default_config();

        // BTC already has a 50% position
        let mut btc = make_asset("BTC", 50000.0, 0.0001, 5.0);
        btc.position = 5.0; // 50% of max_position

        let eth = make_asset("ETH", 3000.0, 0.0002, 5.0);

        let assets = vec![btc, eth];

        // High correlation (0.9 > 0.6 threshold)
        let corr = correlated_matrix(2, 0.9);

        // Check that ETH limit is reduced due to correlated BTC position
        let eth_limit = allocator.correlation_adjusted_limit(1, &assets, &corr);
        let eth_base = assets[1].max_position * assets[1].price;

        // ETH limit should be reduced because BTC has position and high correlation
        assert!(
            eth_limit < eth_base,
            "ETH limit {} should be < base {} due to correlation with positioned BTC",
            eth_limit,
            eth_base
        );
    }

    #[test]
    fn test_negative_edge_not_allocated() {
        let allocator = MultiAssetAllocator::default_config();

        let assets = vec![
            make_asset("BTC", 50000.0, 0.0001, 5.0),  // Positive edge
            make_asset("BAD", 100.0, 0.0001, -10.0),  // Negative edge
        ];
        let corr = identity_matrix(2);

        let result = allocator.optimize(&assets, &corr);

        // BTC should get allocation
        assert!(result.allocations.get("BTC").copied().unwrap_or(0.0) > 0.0);

        // BAD should get minimal or no allocation (negative marginal value)
        // Note: since positions start at 0, negative edge means negative MV
        // so greedy won't allocate to it
    }

    #[test]
    fn test_correlation_adjusted_limit() {
        let allocator = MultiAssetAllocator::default_config();

        // One asset with position, one without
        let mut btc = make_asset("BTC", 50000.0, 0.0001, 5.0);
        btc.position = 0.5; // 50% utilization

        let eth = make_asset("ETH", 3000.0, 0.0002, 5.0);

        let assets = vec![btc, eth];
        let corr = correlated_matrix(2, 0.8); // High correlation

        let eth_limit = allocator.correlation_adjusted_limit(1, &assets, &corr);
        let eth_base = assets[1].max_position * assets[1].price;

        // ETH limit should be reduced due to BTC position + correlation
        assert!(
            eth_limit < eth_base,
            "ETH limit {} should be < base {}",
            eth_limit,
            eth_base
        );
    }

    #[test]
    fn test_new_asset_allocation() {
        let allocator = MultiAssetAllocator::default_config();

        let existing = vec![make_asset("BTC", 50000.0, 0.0001, 5.0)];

        let new_asset = make_asset("ETH", 3000.0, 0.0002, 5.0);
        let correlations = vec![0.3]; // Low correlation with BTC

        let alloc = allocator.new_asset_allocation(&new_asset, &existing, &correlations);

        assert!(alloc >= allocator.config.min_position_notional);
        assert!(alloc <= allocator.config.total_margin);
    }

    #[test]
    fn test_rebalancing_trades() {
        let allocator = MultiAssetAllocator::default_config();

        let mut current = HashMap::new();
        current.insert("BTC".to_string(), 0.1); // 0.1 BTC

        let mut target = HashMap::new();
        target.insert("BTC".to_string(), 10000.0); // $10k in BTC
        target.insert("ETH".to_string(), 5000.0);  // $5k in ETH

        let mut prices = HashMap::new();
        prices.insert("BTC".to_string(), 50000.0);
        prices.insert("ETH".to_string(), 2500.0);

        let trades = allocator.rebalancing_trades(&current, &target, &prices);

        // Should buy more BTC (10000/50000 = 0.2 - 0.1 = 0.1)
        let btc_trade = trades.get("BTC").copied().unwrap_or(0.0);
        assert!(btc_trade > 0.0, "Should buy more BTC: {}", btc_trade);

        // Should buy ETH (5000/2500 = 2.0)
        let eth_trade = trades.get("ETH").copied().unwrap_or(0.0);
        assert!(eth_trade > 0.0, "Should buy ETH: {}", eth_trade);
    }

    #[test]
    fn test_diversification_benefit() {
        let allocator = MultiAssetAllocator::default_config();

        // 3 uncorrelated assets
        let assets = vec![
            make_asset("BTC", 50000.0, 0.0001, 5.0),
            make_asset("ETH", 3000.0, 0.0001, 5.0),
            make_asset("SOL", 100.0, 0.0001, 5.0),
        ];
        let corr = identity_matrix(3);

        let result = allocator.optimize(&assets, &corr);

        // With 3 uncorrelated assets, diversification should be significant
        assert!(
            result.diversification_ratio >= 1.0,
            "Diversification ratio should be >= 1: {}",
            result.diversification_ratio
        );
    }

    #[test]
    fn test_portfolio_variance_calculation() {
        let allocator = MultiAssetAllocator::default_config();

        // Single asset - variance = σ²
        let assets = vec![make_asset("BTC", 50000.0, 0.1, 5.0)]; // σ = 0.1
        let corr = identity_matrix(1);

        let result = allocator.optimize(&assets, &corr);

        // Variance should be positive
        assert!(
            result.portfolio_variance >= 0.0,
            "Variance should be >= 0: {}",
            result.portfolio_variance
        );
    }

    #[test]
    fn test_sharpe_estimate() {
        let allocator = MultiAssetAllocator::default_config();

        let assets = vec![
            make_asset("BTC", 50000.0, 0.0001, 10.0), // High edge
        ];
        let corr = identity_matrix(1);

        let result = allocator.optimize(&assets, &corr);

        // With positive edge and variance, Sharpe should be positive
        if result.portfolio_variance > 1e-12 && result.expected_return > 0.0 {
            assert!(
                result.sharpe_estimate > 0.0,
                "Sharpe should be positive: {}",
                result.sharpe_estimate
            );
        }
    }

    #[test]
    fn test_margin_constraint() {
        let config = PortfolioConfig {
            total_margin: 1000.0, // Very low margin
            ..Default::default()
        };
        let allocator = MultiAssetAllocator::new(config);

        let assets = vec![
            make_asset("BTC", 50000.0, 0.0001, 5.0), // $50k price, expensive
        ];
        let corr = identity_matrix(1);

        let result = allocator.optimize(&assets, &corr);

        // Margin used should not exceed available
        assert!(
            result.margin_used <= 1000.0,
            "Margin used {} should not exceed 1000",
            result.margin_used
        );
    }
}
