//! Execution budget system for statistical impulse control.
//!
//! Implements a token-based budget that gates API calls:
//! - Earns tokens on fills (aligned with Hyperliquid's rate limit model)
//! - Spends tokens on order actions (place/modify/cancel)
//! - Provides budget metrics for observability
//!
//! Based on Hyperliquid's rate limit: 1 request per $1 USDC traded + 10K initial buffer.

use serde::{Deserialize, Serialize};

/// Configuration for the execution budget system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionBudgetConfig {
    /// Initial token budget to start with (provides initial liquidity for quoting)
    pub initial_tokens: f64,
    /// Maximum tokens that can be accumulated (prevents hoarding)
    pub max_tokens: f64,
    /// Emergency reserve for risk management cancels (kill switch, etc.)
    pub emergency_reserve: f64,
    /// Tokens earned per $1000 USD filled
    pub reward_per_1000_usd: f64,
    /// Minimum tokens earned per fill (floor for small HIP-3 fills)
    pub min_tokens_per_fill: f64,
    /// Cost per API action (place/modify/cancel)
    pub cost_per_action: f64,
    /// Discount factor for bulk operations (e.g., 0.8 = 20% discount)
    pub bulk_discount_factor: f64,
}

impl Default for ExecutionBudgetConfig {
    fn default() -> Self {
        Self {
            initial_tokens: 100.0,
            max_tokens: 500.0,
            emergency_reserve: 20.0,
            reward_per_1000_usd: 0.5,
            min_tokens_per_fill: 0.5,  // Floor for HIP-3 small notional fills
            cost_per_action: 1.0,
            bulk_discount_factor: 0.8,
        }
    }
}

/// Metrics snapshot for the execution budget.
#[derive(Debug, Clone, Default)]
pub struct ExecutionBudgetMetrics {
    /// Current available tokens
    pub available_tokens: f64,
    /// Total tokens earned from fills
    pub total_earned: f64,
    /// Total tokens spent on actions
    pub total_spent: f64,
    /// Number of reconciliation cycles skipped due to budget
    pub skipped_count: u64,
    /// Number of emergency reserve uses
    pub emergency_uses: u64,
    /// Budget utilization ratio: spent / (earned + initial)
    pub utilization: f64,
}

/// Token-based execution budget for gating API calls.
///
/// The budget system aligns with Hyperliquid's rate limit model:
/// - Each fill replenishes the budget based on notional volume
/// - Each API action (place/modify/cancel) costs tokens
/// - Emergency reserve allows risk management actions even when budget is low
#[derive(Debug)]
pub struct ExecutionBudget {
    config: ExecutionBudgetConfig,
    /// Current available tokens (excluding emergency reserve)
    available_tokens: f64,
    /// Lifetime tokens earned from fills
    total_earned: f64,
    /// Lifetime tokens spent on actions
    total_spent: f64,
    /// Count of times reconciliation was skipped due to insufficient budget
    skipped_count: u64,
    /// Count of times emergency reserve was used
    emergency_uses: u64,
}

impl ExecutionBudget {
    /// Create a new execution budget with the given configuration.
    pub fn new(config: ExecutionBudgetConfig) -> Self {
        let initial = config.initial_tokens;
        Self {
            config,
            available_tokens: initial,
            total_earned: 0.0,
            total_spent: 0.0,
            skipped_count: 0,
            emergency_uses: 0,
        }
    }

    /// Earn tokens from a fill based on notional USD volume.
    ///
    /// Formula: max(min_tokens_per_fill, (notional_usd / 1000) * reward_per_1000_usd)
    pub fn on_fill(&mut self, notional_usd: f64) {
        let volume_reward = (notional_usd / 1000.0) * self.config.reward_per_1000_usd;
        let reward = volume_reward.max(self.config.min_tokens_per_fill);

        self.total_earned += reward;
        self.available_tokens = (self.available_tokens + reward).min(self.config.max_tokens);
    }

    /// Check if we can afford a single action.
    pub fn can_afford_update(&self) -> bool {
        self.available_tokens >= self.config.cost_per_action
    }

    /// Check if we can afford a bulk operation of `count` actions.
    ///
    /// Bulk operations get a discount (bulk_discount_factor).
    pub fn can_afford_bulk(&self, count: usize) -> bool {
        if count == 0 {
            return true;
        }
        let cost = self.bulk_cost(count);
        self.available_tokens >= cost
    }

    /// Calculate the cost of a bulk operation.
    fn bulk_cost(&self, count: usize) -> f64 {
        if count == 0 {
            return 0.0;
        }
        // First action at full cost, rest at discounted rate
        let first_cost = self.config.cost_per_action;
        let rest_cost = (count - 1) as f64 * self.config.cost_per_action * self.config.bulk_discount_factor;
        first_cost + rest_cost
    }

    /// Spend tokens for `count` actions.
    ///
    /// Returns `true` if the spend was successful, `false` if insufficient budget.
    pub fn spend(&mut self, count: usize) -> bool {
        if count == 0 {
            return true;
        }

        let cost = self.bulk_cost(count);
        if self.available_tokens >= cost {
            self.available_tokens -= cost;
            self.total_spent += cost;
            true
        } else {
            self.skipped_count += 1;
            false
        }
    }

    /// Emergency spend that dips into reserve.
    ///
    /// Used for risk management actions (kill switch cancels, reduce-only mode).
    /// Always succeeds but tracks emergency usage.
    pub fn emergency_spend(&mut self, count: usize) {
        if count == 0 {
            return;
        }

        let cost = self.bulk_cost(count);

        // Use emergency reserve if needed
        let total_available = self.available_tokens + self.config.emergency_reserve;
        if cost > self.available_tokens {
            self.emergency_uses += 1;
        }

        // Spend what we can, potentially going negative (up to reserve)
        let actual_cost = cost.min(total_available);
        self.available_tokens = (self.available_tokens - actual_cost).max(-self.config.emergency_reserve);
        self.total_spent += actual_cost;
    }

    /// Record a skipped reconciliation cycle (for metrics).
    pub fn record_skip(&mut self) {
        self.skipped_count += 1;
    }

    /// Get current budget metrics.
    pub fn get_metrics(&self) -> ExecutionBudgetMetrics {
        let total_budget = self.total_earned + self.config.initial_tokens;
        let utilization = if total_budget > 0.0 {
            self.total_spent / total_budget
        } else {
            0.0
        };

        ExecutionBudgetMetrics {
            available_tokens: self.available_tokens,
            total_earned: self.total_earned,
            total_spent: self.total_spent,
            skipped_count: self.skipped_count,
            emergency_uses: self.emergency_uses,
            utilization,
        }
    }

    /// Get current available tokens.
    pub fn available(&self) -> f64 {
        self.available_tokens
    }

    /// Get count of skipped cycles.
    pub fn skipped_count(&self) -> u64 {
        self.skipped_count
    }

    /// Get count of emergency reserve uses.
    pub fn emergency_uses(&self) -> u64 {
        self.emergency_uses
    }

    /// Reset the budget (for testing or restart scenarios).
    #[cfg(test)]
    pub fn reset(&mut self) {
        self.available_tokens = self.config.initial_tokens;
        self.total_earned = 0.0;
        self.total_spent = 0.0;
        self.skipped_count = 0;
        self.emergency_uses = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_budget() -> ExecutionBudget {
        ExecutionBudget::new(ExecutionBudgetConfig::default())
    }

    #[test]
    fn test_initial_budget() {
        let budget = default_budget();
        assert!((budget.available() - 100.0).abs() < f64::EPSILON);
        assert!(budget.can_afford_update());
        assert!(budget.can_afford_bulk(10));
    }

    #[test]
    fn test_on_fill_standard() {
        let mut budget = default_budget();
        budget.available_tokens = 50.0;  // Start lower to see the effect

        // $2000 fill = 2 * 0.5 = 1.0 tokens
        budget.on_fill(2000.0);
        assert!((budget.available() - 51.0).abs() < f64::EPSILON);
        assert!((budget.total_earned - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_on_fill_small_notional_floor() {
        let mut budget = default_budget();
        budget.available_tokens = 50.0;

        // $10 fill would be 0.005 tokens, but floor is 0.5
        budget.on_fill(10.0);
        assert!((budget.available() - 50.5).abs() < f64::EPSILON);
        assert!((budget.total_earned - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_on_fill_respects_max() {
        let mut budget = default_budget();
        budget.available_tokens = 495.0;

        // Large fill that would exceed max
        budget.on_fill(100_000.0);  // 50 tokens
        assert!((budget.available() - 500.0).abs() < f64::EPSILON);  // Capped at max
    }

    #[test]
    fn test_spend_single_action() {
        let mut budget = default_budget();

        assert!(budget.spend(1));
        assert!((budget.available() - 99.0).abs() < f64::EPSILON);
        assert!((budget.total_spent - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_spend_bulk_with_discount() {
        let mut budget = default_budget();

        // 10 actions: 1.0 + 9 * 0.8 = 8.2 tokens
        assert!(budget.can_afford_bulk(10));
        assert!(budget.spend(10));

        let expected_cost = 1.0 + 9.0 * 0.8;  // 8.2
        assert!((budget.total_spent - expected_cost).abs() < 0.001);
    }

    #[test]
    fn test_spend_fails_insufficient_budget() {
        let config = ExecutionBudgetConfig {
            initial_tokens: 5.0,
            ..Default::default()
        };
        let mut budget = ExecutionBudget::new(config);

        // Try to spend more than available
        assert!(!budget.can_afford_bulk(10));
        assert!(!budget.spend(10));
        assert_eq!(budget.skipped_count(), 1);
        assert!((budget.available() - 5.0).abs() < f64::EPSILON);  // Unchanged
    }

    #[test]
    fn test_emergency_spend() {
        let config = ExecutionBudgetConfig {
            initial_tokens: 5.0,
            emergency_reserve: 20.0,
            ..Default::default()
        };
        let mut budget = ExecutionBudget::new(config);

        // Emergency spend that dips into reserve
        budget.emergency_spend(10);

        assert_eq!(budget.emergency_uses(), 1);
        // Should have negative balance (up to reserve)
        assert!(budget.available() < 0.0);
        assert!(budget.available() >= -20.0);
    }

    #[test]
    fn test_metrics() {
        let mut budget = default_budget();

        budget.on_fill(5000.0);  // 2.5 tokens
        budget.spend(5);  // 1.0 + 4*0.8 = 4.2 tokens

        let metrics = budget.get_metrics();
        assert!((metrics.total_earned - 2.5).abs() < f64::EPSILON);
        assert!((metrics.total_spent - 4.2).abs() < 0.001);
        assert!(metrics.utilization > 0.0);
    }

    #[test]
    fn test_bulk_cost_calculation() {
        let budget = default_budget();

        // 0 actions = 0 cost
        assert!((budget.bulk_cost(0) - 0.0).abs() < f64::EPSILON);

        // 1 action = 1.0 cost (no discount)
        assert!((budget.bulk_cost(1) - 1.0).abs() < f64::EPSILON);

        // 2 actions = 1.0 + 0.8 = 1.8
        assert!((budget.bulk_cost(2) - 1.8).abs() < f64::EPSILON);

        // 5 actions = 1.0 + 4*0.8 = 4.2
        assert!((budget.bulk_cost(5) - 4.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_zero_actions() {
        let mut budget = default_budget();

        assert!(budget.can_afford_bulk(0));
        assert!(budget.spend(0));
        assert!((budget.available() - 100.0).abs() < f64::EPSILON);
    }
}
