//! Queue value estimation for HJB controller.
//!
//! Extends the HJB value function to V(t, I, S, q) where q is queue position.
//! This allows the controller to make economically-informed decisions about
//! when to cancel orders based on the value of their queue position.
//!
//! ## Theory
//!
//! The queue value formula captures two opposing effects:
//! 1. **Exponential decay**: Orders deeper in queue have less chance of filling
//! 2. **Linear cost**: Maintaining queue position has opportunity cost
//!
//! ```text
//! v(q) = (s/2) × exp(-α×q) - β×q
//! ```
//!
//! Where:
//! - `s` = half-spread (bps) - the reward for filling
//! - `q` = queue depth ahead - number of units in front
//! - `α` = decay rate - how fast value decreases with depth
//! - `β` = linear cost - opportunity cost per unit depth
//!
//! ## Decision Rule
//!
//! Preserve an order if: `v(q) > modify_cost_bps`
//!
//! This ensures we only cancel orders when the economic benefit of the new
//! position exceeds the cost of losing queue priority.

/// Configuration for HJB queue value calculations.
///
/// Note: Named `HJBQueueValueConfig` to avoid conflict with the existing
/// `QueueValueConfig` in tracking/queue module which has different fields.
#[derive(Debug, Clone, Copy)]
pub struct HJBQueueValueConfig {
    /// Decay rate (α). Higher = queue value decays faster with depth.
    /// Typical range: [0.05, 0.3], Default: 0.1
    pub alpha: f64,

    /// Linear cost (β). Penalizes orders deep in queue.
    /// Typical range: [0.01, 0.05], Default: 0.02
    pub beta: f64,

    /// Cost of modifying an order (in bps).
    /// This is the break-even threshold for cancel+replace decisions.
    /// Typical range: [1.0, 10.0], Default: 3.0
    pub modify_cost_bps: f64,

    /// Minimum queue value to consider (bps).
    /// Orders with value below this are candidates for refresh.
    pub min_queue_value_bps: f64,

    /// Maximum queue depth to track.
    /// Beyond this, queue value is effectively zero.
    pub max_queue_depth: f64,
}

impl Default for HJBQueueValueConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            beta: 0.02,
            modify_cost_bps: 3.0,
            min_queue_value_bps: 0.5,
            max_queue_depth: 100.0,
        }
    }
}

impl HJBQueueValueConfig {
    /// Create from HJB config parameters.
    pub fn from_hjb_config(
        queue_alpha: f64,
        queue_beta: f64,
        queue_modify_cost_bps: f64,
    ) -> Self {
        Self {
            alpha: queue_alpha,
            beta: queue_beta,
            modify_cost_bps: queue_modify_cost_bps,
            ..Default::default()
        }
    }
}

/// Computed queue value for an order.
#[derive(Debug, Clone, Copy)]
pub struct OrderQueueValue {
    /// Order ID
    pub oid: u64,

    /// Queue depth ahead (units)
    pub queue_depth: f64,

    /// Computed queue value (bps)
    pub value_bps: f64,

    /// Whether to preserve this order
    pub should_preserve: bool,

    /// Half-spread used in calculation (bps)
    pub half_spread_bps: f64,
}

/// HJB queue value calculator.
///
/// Provides methods for computing queue value and making preservation decisions.
#[derive(Debug, Clone)]
pub struct HJBQueueValueCalculator {
    config: HJBQueueValueConfig,
}

impl HJBQueueValueCalculator {
    /// Create a new HJB queue value calculator.
    pub fn new(config: HJBQueueValueConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(HJBQueueValueConfig::default())
    }

    /// Compute the queue value for a given position and spread.
    ///
    /// # Arguments
    /// * `queue_depth` - Number of units ahead in queue
    /// * `half_spread_bps` - Half-spread in basis points (reward for filling)
    ///
    /// # Returns
    /// Queue value in basis points. Higher = more valuable to preserve.
    ///
    /// # Formula
    /// ```text
    /// v(q) = (s/2) × exp(-α×q) - β×q
    /// ```
    pub fn queue_value(&self, queue_depth: f64, half_spread_bps: f64) -> f64 {
        if queue_depth < 0.0 || queue_depth > self.config.max_queue_depth {
            return 0.0;
        }

        // Exponential term: expected fill value decays with queue depth
        let exp_term = (half_spread_bps / 2.0) * (-self.config.alpha * queue_depth).exp();

        // Linear term: opportunity cost of maintaining queue position
        let lin_term = self.config.beta * queue_depth;

        // Queue value = expected value minus cost
        (exp_term - lin_term).max(0.0)
    }

    /// Check if an order should be preserved based on its queue value.
    ///
    /// # Arguments
    /// * `queue_depth` - Number of units ahead in queue
    /// * `half_spread_bps` - Half-spread in basis points
    ///
    /// # Returns
    /// True if the order's queue value exceeds the modify cost threshold.
    pub fn should_preserve(&self, queue_depth: f64, half_spread_bps: f64) -> bool {
        self.queue_value(queue_depth, half_spread_bps) >= self.config.modify_cost_bps
    }

    /// Compute full queue value analysis for an order.
    ///
    /// # Arguments
    /// * `oid` - Order ID
    /// * `queue_depth` - Number of units ahead in queue
    /// * `half_spread_bps` - Half-spread in basis points
    pub fn compute(&self, oid: u64, queue_depth: f64, half_spread_bps: f64) -> OrderQueueValue {
        let value_bps = self.queue_value(queue_depth, half_spread_bps);
        let should_preserve = value_bps >= self.config.modify_cost_bps;

        OrderQueueValue {
            oid,
            queue_depth,
            value_bps,
            should_preserve,
            half_spread_bps,
        }
    }

    /// Compute the break-even queue depth where value equals modify cost.
    ///
    /// Orders at or beyond this depth have no economic value to preserve.
    ///
    /// # Arguments
    /// * `half_spread_bps` - Half-spread in basis points
    pub fn break_even_depth(&self, half_spread_bps: f64) -> f64 {
        // Solve: (s/2) × exp(-α×q) - β×q = modify_cost
        // This is transcendental, so we use Newton-Raphson iteration

        let target = self.config.modify_cost_bps;
        let s2 = half_spread_bps / 2.0;
        let alpha = self.config.alpha;
        let beta = self.config.beta;

        // Initial guess: where exp term roughly equals target
        let mut q = if s2 > target {
            -((2.0 * target) / s2).ln() / alpha
        } else {
            0.0
        };

        // Newton-Raphson iterations
        for _ in 0..10 {
            let exp_aq = (-alpha * q).exp();
            let f = s2 * exp_aq - beta * q - target;
            let f_prime = -alpha * s2 * exp_aq - beta;

            if f_prime.abs() < 1e-10 {
                break;
            }

            let delta = f / f_prime;
            q -= delta;

            if delta.abs() < 1e-6 {
                break;
            }
        }

        q.max(0.0).min(self.config.max_queue_depth)
    }

    /// Get the configuration.
    pub fn config(&self) -> &HJBQueueValueConfig {
        &self.config
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: HJBQueueValueConfig) {
        self.config = config;
    }
}

/// Batch queue value analysis for multiple orders.
#[derive(Debug, Clone)]
pub struct BatchQueueValue {
    /// Individual order values
    pub orders: Vec<OrderQueueValue>,

    /// Total queue value across all orders (bps)
    pub total_value_bps: f64,

    /// Number of orders that should be preserved
    pub preserve_count: usize,

    /// Number of orders that can be cancelled
    pub cancel_count: usize,
}

impl BatchQueueValue {
    /// Compute batch queue value for multiple orders.
    ///
    /// # Arguments
    /// * `calculator` - HJB queue value calculator
    /// * `orders` - Vector of (oid, queue_depth, half_spread_bps) tuples
    pub fn compute(calculator: &HJBQueueValueCalculator, orders: Vec<(u64, f64, f64)>) -> Self {
        let mut order_values = Vec::with_capacity(orders.len());
        let mut total_value = 0.0;
        let mut preserve_count = 0;

        for (oid, queue_depth, half_spread_bps) in orders {
            let value = calculator.compute(oid, queue_depth, half_spread_bps);
            total_value += value.value_bps;
            if value.should_preserve {
                preserve_count += 1;
            }
            order_values.push(value);
        }

        let cancel_count = order_values.len() - preserve_count;

        Self {
            orders: order_values,
            total_value_bps: total_value,
            preserve_count,
            cancel_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_value_basic() {
        let calc = HJBQueueValueCalculator::default_config();

        // Zero queue depth = maximum value
        let v0 = calc.queue_value(0.0, 10.0);
        assert!(v0 > 0.0);
        assert!((v0 - 5.0).abs() < 0.01); // s/2 when q=0

        // Queue value decreases with depth
        let v5 = calc.queue_value(5.0, 10.0);
        let v10 = calc.queue_value(10.0, 10.0);
        assert!(v0 > v5);
        assert!(v5 > v10);
    }

    #[test]
    fn test_should_preserve() {
        let calc = HJBQueueValueCalculator::default_config();

        // At front of queue with decent spread: preserve
        assert!(calc.should_preserve(0.0, 10.0));

        // Deep in queue: don't preserve
        assert!(!calc.should_preserve(50.0, 10.0));
    }

    #[test]
    fn test_break_even_depth() {
        let calc = HJBQueueValueCalculator::default_config();

        // Break-even depth should be positive for reasonable spreads
        let be_depth = calc.break_even_depth(10.0);
        assert!(be_depth > 0.0);

        // Orders at break-even should have value ≈ modify_cost
        let value_at_be = calc.queue_value(be_depth, 10.0);
        assert!((value_at_be - calc.config().modify_cost_bps).abs() < 0.1);
    }

    #[test]
    fn test_batch_queue_value() {
        let calc = HJBQueueValueCalculator::default_config();

        let orders = vec![
            (1, 0.0, 10.0),  // Front of queue, should preserve
            (2, 5.0, 10.0),  // Mid queue
            (3, 50.0, 10.0), // Deep queue, should not preserve
        ];

        let batch = BatchQueueValue::compute(&calc, orders);

        assert_eq!(batch.orders.len(), 3);
        assert!(batch.orders[0].should_preserve);
        assert!(!batch.orders[2].should_preserve);
        assert!(batch.total_value_bps > 0.0);
    }

    #[test]
    fn test_edge_cases() {
        let calc = HJBQueueValueCalculator::default_config();

        // Negative depth should return 0
        assert_eq!(calc.queue_value(-1.0, 10.0), 0.0);

        // Beyond max depth should return 0
        assert_eq!(calc.queue_value(200.0, 10.0), 0.0);

        // Zero spread should have low value
        let v = calc.queue_value(0.0, 0.0);
        assert!(v <= 0.0);
    }
}
