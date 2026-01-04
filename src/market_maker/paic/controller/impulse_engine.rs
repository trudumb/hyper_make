//! Impulse Control Engine - The brain of PAIC.
//!
//! This engine calculates two values for every active order and intervenes
//! only when the spread between them exceeds the cost.
//!
//! # Core Inequality
//!
//! We trigger an update only if:
//! ```text
//! V_move > V_hold
//! ```
//!
//! # Strategy Matrix
//!
//! |                    | Small Drift | Large Drift |
//! |--------------------|-------------|-------------|
//! | High Priority (π≈0) | HOLD        | LEAK        |
//! | Low Priority (π≈1)  | SHADOW      | RESET       |

use super::actions::{ImpulseAction, ImpulseDecision};
use super::priority_value::PriorityValueCalculator;
use super::super::config::PAICConfig;
use super::super::observer::{MarketState, PriorityClass, StateEstimator};

/// Order state for impulse decision.
#[derive(Debug, Clone)]
pub struct OrderState {
    /// Order ID
    pub oid: u64,
    /// Current order price
    pub price: f64,
    /// Current order size
    pub size: f64,
    /// Is this a bid order?
    pub is_bid: bool,
    /// Original size (for leak calculation)
    pub original_size: f64,
}

/// Impulse Control Engine.
#[derive(Debug)]
pub struct ImpulseEngine {
    /// Configuration
    config: PAICConfig,

    /// Priority value calculator
    value_calc: PriorityValueCalculator,
}

impl ImpulseEngine {
    /// Create a new impulse engine.
    pub fn new(config: PAICConfig) -> Self {
        let value_calc = PriorityValueCalculator {
            priority_premium_multiplier: config.priority_premium_multiplier,
            high_priority_threshold: config.high_priority_threshold,
            ..Default::default()
        };

        Self { config, value_calc }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(PAICConfig::default())
    }

    /// Decide action for a single order.
    ///
    /// The core logic:
    /// 1. Calculate drift from fair price
    /// 2. Get priority index (π) from state estimator
    /// 3. Calculate thresholds based on priority
    /// 4. Apply strategy matrix to decide action
    pub fn decide_action(
        &self,
        order: &OrderState,
        fair_price: f64,
        state_estimator: &StateEstimator,
        market_state: &MarketState,
    ) -> ImpulseDecision {
        // Get priority index
        let pi = state_estimator.get_priority(order.oid).unwrap_or(1.0);
        let priority_class = self.value_calc.classify_priority(pi);

        // Calculate drift in basis points
        let drift = (fair_price - order.price).abs();
        let drift_bps = if order.price > 0.0 {
            drift / order.price * 10_000.0
        } else {
            0.0
        };

        // Calculate priority premium
        let priority_premium_bps = self.value_calc.priority_premium_bps(pi, market_state.spread_bps);

        // Calculate dynamic thresholds
        let modify_threshold_bps = market_state.modify_threshold(pi, &self.config);
        let leak_threshold_bps = market_state.leak_threshold(pi, &self.config);

        // Check toxicity for our side
        let is_toxic = state_estimator.is_toxic_for_side(order.is_bid);

        // Apply strategy matrix
        self.apply_strategy_matrix(
            order,
            fair_price,
            pi,
            priority_class,
            drift_bps,
            priority_premium_bps,
            modify_threshold_bps,
            leak_threshold_bps,
            is_toxic,
        )
    }

    /// Apply the PAIC strategy matrix.
    ///
    /// |                    | Small Drift | Large Drift |
    /// |--------------------|-------------|-------------|
    /// | High Priority (π≈0) | HOLD        | LEAK        |
    /// | Low Priority (π≈1)  | SHADOW      | RESET       |
    #[allow(clippy::too_many_arguments)]
    fn apply_strategy_matrix(
        &self,
        order: &OrderState,
        fair_price: f64,
        pi: f64,
        priority_class: PriorityClass,
        drift_bps: f64,
        priority_premium_bps: f64,
        modify_threshold_bps: f64,
        leak_threshold_bps: f64,
        is_toxic: bool,
    ) -> ImpulseDecision {
        // Large drift threshold check
        let is_large_drift = drift_bps > modify_threshold_bps;
        let is_medium_drift = drift_bps > leak_threshold_bps;

        // is_toxic_medium indicates toxic flow AND medium drift (worth considering leak)
        let is_toxic_medium = is_toxic && is_medium_drift;

        match (priority_class.is_high(), is_large_drift, is_toxic_medium) {
            // High priority + toxic medium/large drift → LEAK
            // Market moving against us with toxic flow, reduce exposure
            (true, _, true) => {
                let new_size = order.size * self.config.leak_size_factor;
                ImpulseDecision::leak(
                    order.oid,
                    new_size,
                    pi,
                    drift_bps,
                    priority_premium_bps,
                    modify_threshold_bps,
                )
            }

            // High priority + large drift (not toxic) → HOLD
            // We value the queue position highly even with drift
            (true, true, false) => ImpulseDecision::hold(
                order.oid,
                pi,
                drift_bps,
                priority_premium_bps,
                modify_threshold_bps,
            ),

            // High priority + small drift + not toxic → HOLD
            (true, false, false) => ImpulseDecision::hold(
                order.oid,
                pi,
                drift_bps,
                priority_premium_bps,
                modify_threshold_bps,
            ),

            // Low priority + small drift → SHADOW
            (false, false, _) => ImpulseDecision::shadow(
                order.oid,
                fair_price,
                pi,
                drift_bps,
                priority_premium_bps,
                modify_threshold_bps,
            ),

            // Low priority + large drift → RESET
            (false, true, _) => ImpulseDecision::reset(
                order.oid,
                fair_price,
                pi,
                drift_bps,
                priority_premium_bps,
                modify_threshold_bps,
            ),
        }
    }

    /// Decide actions for all orders in a batch.
    pub fn decide_batch(
        &self,
        orders: &[OrderState],
        fair_price: f64,
        state_estimator: &StateEstimator,
        market_state: &MarketState,
    ) -> Vec<ImpulseDecision> {
        orders
            .iter()
            .map(|order| self.decide_action(order, fair_price, state_estimator, market_state))
            .collect()
    }

    /// Filter decisions that require action (exclude HOLDs).
    pub fn filter_actionable(decisions: &[ImpulseDecision]) -> Vec<&ImpulseDecision> {
        decisions
            .iter()
            .filter(|d| d.action != ImpulseAction::Hold)
            .collect()
    }

    /// Get the most urgent decision from a batch.
    pub fn most_urgent(decisions: &[ImpulseDecision]) -> Option<&ImpulseDecision> {
        decisions.iter().max_by_key(|d| d.action.urgency())
    }

    /// Calculate aggregate importance for rate limiting.
    pub fn aggregate_importance(decisions: &[ImpulseDecision]) -> f64 {
        decisions.iter().map(|d| d.importance).sum()
    }

    /// Get configuration reference.
    pub fn config(&self) -> &PAICConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_order(oid: u64, price: f64, is_bid: bool) -> OrderState {
        OrderState {
            oid,
            price,
            size: 1.0,
            is_bid,
            original_size: 1.0,
        }
    }

    #[test]
    fn test_impulse_engine_new() {
        let engine = ImpulseEngine::default_config();
        assert!((engine.config.min_drift_bps - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_high_priority_small_drift_holds() {
        let engine = ImpulseEngine::default_config();
        let mut state_estimator = StateEstimator::default_config();

        // Place order and advance to high priority
        state_estimator.order_placed(1, 100.0, 1.0, 10.0, true);
        // Simulate lots of trades to advance priority
        for _ in 0..20 {
            state_estimator.on_trade(100.0, 1.0, true);
        }

        let market_state = state_estimator.market_state();
        let order = make_order(1, 100.0, true);

        // Fair price very close to order price
        let fair_price = 100.0005; // 0.5 bps drift

        let decision = engine.decide_action(&order, fair_price, &state_estimator, &market_state);

        assert_eq!(decision.action, ImpulseAction::Hold);
    }

    #[test]
    fn test_low_priority_small_drift_shadows() {
        let engine = ImpulseEngine::default_config();
        let mut state_estimator = StateEstimator::default_config();

        // Place order at back of queue (no trades to advance)
        state_estimator.order_placed(1, 100.0, 1.0, 10.0, true);

        let market_state = state_estimator.market_state();
        let order = make_order(1, 100.0, true);

        // Fair price close but different
        let fair_price = 100.002; // 2 bps drift

        let decision = engine.decide_action(&order, fair_price, &state_estimator, &market_state);

        // Low priority + small drift = SHADOW
        assert_eq!(decision.action, ImpulseAction::Shadow);
    }

    #[test]
    fn test_low_priority_large_drift_resets() {
        let engine = ImpulseEngine::default_config();
        let mut state_estimator = StateEstimator::default_config();

        // Place order at back of queue
        state_estimator.order_placed(1, 100.0, 1.0, 10.0, true);

        // Update book to establish spread
        state_estimator.update_book(100.0, 100.1);

        let market_state = state_estimator.market_state();
        let order = make_order(1, 100.0, true);

        // Large drift from fair price
        let fair_price = 100.05; // 50 bps drift

        let decision = engine.decide_action(&order, fair_price, &state_estimator, &market_state);

        // Low priority + large drift = RESET
        assert_eq!(decision.action, ImpulseAction::Reset);
    }

    #[test]
    fn test_filter_actionable() {
        let decisions = vec![
            ImpulseDecision::hold(1, 0.1, 1.0, 5.0, 10.0),
            ImpulseDecision::shadow(2, 100.0, 0.9, 2.0, 1.0, 5.0),
            ImpulseDecision::reset(3, 101.0, 0.95, 20.0, 0.5, 5.0),
        ];

        let actionable = ImpulseEngine::filter_actionable(&decisions);
        assert_eq!(actionable.len(), 2);
    }

    #[test]
    fn test_most_urgent() {
        let decisions = vec![
            ImpulseDecision::hold(1, 0.1, 1.0, 5.0, 10.0),
            ImpulseDecision::shadow(2, 100.0, 0.9, 2.0, 1.0, 5.0),
            ImpulseDecision::reset(3, 101.0, 0.95, 20.0, 0.5, 5.0),
        ];

        let most_urgent = ImpulseEngine::most_urgent(&decisions);
        assert!(most_urgent.is_some());
        assert_eq!(most_urgent.unwrap().action, ImpulseAction::Reset);
    }
}
