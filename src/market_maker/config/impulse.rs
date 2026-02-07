//! Statistical impulse control configuration.

use crate::market_maker::infra::{ExecutionBudget, ExecutionBudgetConfig};
use crate::market_maker::tracking::ImpulseFilterConfig;

/// Configuration for statistical impulse control.
///
/// Combines token budget and impulse filter to reduce API churn while maintaining
/// fill rate. The system only updates orders when the improvement in fill probability
/// exceeds the cost of the update.
///
/// Formula: Update only when `Δλ = (λ_new - λ_current) / λ_current > threshold`
#[derive(Debug, Clone)]
pub struct ImpulseControlConfig {
    /// Enable statistical impulse control.
    /// When false, all filtering is bypassed and orders update normally.
    pub enabled: bool,

    /// Token budget configuration for API call gating.
    pub budget: ExecutionBudgetConfig,

    /// Impulse filter configuration for Δλ-based filtering.
    pub filter: ImpulseFilterConfig,
}

impl Default for ImpulseControlConfig {
    fn default() -> Self {
        Self {
            enabled: true, // Enabled by default - reduces API churn by 50-70%
            budget: ExecutionBudgetConfig::default(),
            filter: ImpulseFilterConfig::default(),
        }
    }
}

impl ImpulseControlConfig {
    /// Create a new impulse control config with default settings, enabled.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }

    /// Builder: set improvement threshold (Δλ required for update).
    pub fn with_improvement_threshold(mut self, threshold: f64) -> Self {
        self.filter.improvement_threshold = threshold;
        self
    }

    /// Builder: set queue lock threshold (P(fill) above which orders are locked).
    pub fn with_queue_lock_threshold(mut self, threshold: f64) -> Self {
        self.filter.queue_lock_threshold = threshold;
        self
    }

    /// Builder: set initial tokens for budget.
    pub fn with_initial_tokens(mut self, tokens: f64) -> Self {
        self.budget.initial_tokens = tokens;
        self
    }

    /// Builder: set cost per action.
    pub fn with_cost_per_action(mut self, cost: f64) -> Self {
        self.budget.cost_per_action = cost;
        self
    }

    /// Create the ExecutionBudget from this config.
    pub fn create_budget(&self) -> ExecutionBudget {
        ExecutionBudget::new(self.budget.clone())
    }
}
