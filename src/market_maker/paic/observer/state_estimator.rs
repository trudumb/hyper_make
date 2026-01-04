//! State Estimator - Synthesizes market state for PAIC controller.
//!
//! Combines volatility regime, queue priority, and flow toxicity into
//! a unified market state that drives impulse control decisions.

use super::toxicity::ToxicityEstimator;
use super::virtual_queue::{PriorityClass, VirtualQueueTracker};
use super::super::config::PAICConfig;

/// Volatility regime state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolatilityState {
    /// Very quiet market (σ < 0.5 × baseline)
    Quiet,
    /// Normal market conditions
    Normal,
    /// Elevated volatility (σ > 2 × baseline)
    Turbulent,
}

impl VolatilityState {
    /// Create from volatility ratio (current / baseline).
    pub fn from_ratio(ratio: f64, quiet_threshold: f64, turbulent_threshold: f64) -> Self {
        if ratio < quiet_threshold {
            Self::Quiet
        } else if ratio > turbulent_threshold {
            Self::Turbulent
        } else {
            Self::Normal
        }
    }

    /// Get width multiplier for no-action band.
    pub fn band_multiplier(&self) -> f64 {
        match self {
            Self::Quiet => 0.5,    // Tighter band in quiet markets
            Self::Normal => 1.0,   // Base case
            Self::Turbulent => 2.0, // Wider band in turbulent markets
        }
    }
}

/// Synthesized market state for PAIC decisions.
#[derive(Debug, Clone)]
pub struct MarketState {
    // === Volatility ===
    /// Current volatility (per-second)
    pub sigma: f64,
    /// Baseline volatility for regime classification
    pub sigma_baseline: f64,
    /// Volatility regime
    pub volatility_state: VolatilityState,
    /// Current spread in basis points
    pub spread_bps: f64,

    // === Toxicity ===
    /// Toxicity score [0, 1]
    pub toxicity: f64,
    /// Order flow imbalance [-1, 1]
    pub ofi: f64,
    /// Is flow currently toxic?
    pub is_toxic: bool,

    // === Price ===
    /// Current mid price
    pub mid_price: f64,
    /// Microprice (if available)
    pub microprice: Option<f64>,
}

impl MarketState {
    /// Calculate dynamic threshold for modify action.
    ///
    /// Threshold = volatility_band × (1 + priority_premium)
    pub fn modify_threshold(&self, priority: f64, config: &PAICConfig) -> f64 {
        let base_band = self.volatility_state.band_multiplier() * config.min_drift_bps;
        let priority_premium = (1.0 - priority) * self.spread_bps * config.priority_premium_multiplier;
        base_band + priority_premium
    }

    /// Calculate threshold for leak action (more aggressive than modify).
    pub fn leak_threshold(&self, priority: f64, config: &PAICConfig) -> f64 {
        self.modify_threshold(priority, config) * 0.5
    }

    /// Check if conditions favor tight quoting.
    ///
    /// Returns true if:
    /// - Volatility is quiet or normal
    /// - Not toxic flow
    pub fn can_quote_tight(&self) -> bool {
        !self.is_toxic && self.volatility_state != VolatilityState::Turbulent
    }
}

/// State Estimator - combines all observer components.
#[derive(Debug)]
pub struct StateEstimator {
    /// Configuration
    config: PAICConfig,

    /// Virtual queue tracker
    queue_tracker: VirtualQueueTracker,

    /// Toxicity estimator
    toxicity_estimator: ToxicityEstimator,

    /// Current volatility
    sigma: f64,

    /// Baseline volatility (long-term EWMA)
    sigma_baseline: f64,

    /// EWMA alpha for baseline updates
    baseline_alpha: f64,

    /// Current mid price
    mid_price: f64,

    /// Current spread in bps
    spread_bps: f64,

    /// Microprice (if available)
    microprice: Option<f64>,

    /// Tick count for warmup
    tick_count: usize,

    /// Minimum ticks for warmup
    min_warmup_ticks: usize,
}

impl StateEstimator {
    /// Create a new state estimator.
    pub fn new(config: PAICConfig) -> Self {
        let queue_config = config.queue_config.clone();
        Self {
            config,
            queue_tracker: VirtualQueueTracker::new(queue_config),
            toxicity_estimator: ToxicityEstimator::default_config(),
            sigma: 0.0001,          // Default 1 bps/s
            sigma_baseline: 0.0001, // Default baseline
            baseline_alpha: 0.001,  // Slow baseline update
            mid_price: 0.0,
            spread_bps: 10.0,
            microprice: None,
            tick_count: 0,
            min_warmup_ticks: 20,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(PAICConfig::default())
    }

    // === Updates ===

    /// Update with new volatility observation.
    pub fn update_volatility(&mut self, sigma: f64) {
        self.sigma = sigma;
        // Update baseline with slow EWMA
        self.sigma_baseline = self.baseline_alpha * sigma + (1.0 - self.baseline_alpha) * self.sigma_baseline;
        self.tick_count += 1;
    }

    /// Update with trade observation.
    pub fn on_trade(&mut self, price: f64, volume: f64, is_buy: bool) {
        // Update toxicity
        self.toxicity_estimator.on_trade(volume, is_buy);

        // Update queue tracker
        self.queue_tracker.on_trade(price, volume, is_buy);
    }

    /// Update with L2 book.
    pub fn update_book(&mut self, best_bid: f64, best_ask: f64) {
        self.queue_tracker.update_bbo(best_bid, best_ask);
        self.mid_price = (best_bid + best_ask) / 2.0;
        if self.mid_price > 0.0 {
            self.spread_bps = (best_ask - best_bid) / self.mid_price * 10_000.0;
        }
    }

    /// Update microprice.
    pub fn update_microprice(&mut self, microprice: f64) {
        self.microprice = Some(microprice);
    }

    /// Apply decay (call periodically).
    pub fn apply_decay(&mut self) {
        self.queue_tracker.apply_decay();
    }

    // === Queue Management ===

    /// Register a new order for tracking.
    pub fn order_placed(
        &mut self,
        oid: u64,
        price: f64,
        size: f64,
        depth_ahead: f64,
        is_bid: bool,
    ) {
        self.queue_tracker.order_placed(oid, price, size, depth_ahead, is_bid);
    }

    /// Remove an order from tracking.
    pub fn order_removed(&mut self, oid: u64) {
        self.queue_tracker.order_removed(oid);
    }

    /// Update order after partial fill.
    pub fn order_partially_filled(&mut self, oid: u64, filled_amount: f64) {
        self.queue_tracker.order_partially_filled(oid, filled_amount);
    }

    // === State Access ===

    /// Get current synthesized market state.
    pub fn market_state(&self) -> MarketState {
        let ratio = if self.sigma_baseline > 1e-10 {
            self.sigma / self.sigma_baseline
        } else {
            1.0
        };

        MarketState {
            sigma: self.sigma,
            sigma_baseline: self.sigma_baseline,
            volatility_state: VolatilityState::from_ratio(
                ratio,
                self.config.volatility_quiet_threshold,
                self.config.volatility_turbulent_threshold,
            ),
            spread_bps: self.spread_bps,
            toxicity: self.toxicity_estimator.toxicity(),
            ofi: self.toxicity_estimator.ofi(),
            is_toxic: self.toxicity_estimator.is_toxic(),
            mid_price: self.mid_price,
            microprice: self.microprice,
        }
    }

    /// Get priority index for an order.
    pub fn get_priority(&self, oid: u64) -> Option<f64> {
        self.queue_tracker.get_priority(oid)
    }

    /// Get priority class for an order.
    pub fn get_priority_class(&self, oid: u64) -> Option<PriorityClass> {
        self.queue_tracker.get_priority_class(oid)
    }

    /// Get option value for an order.
    pub fn get_option_value(&self, oid: u64) -> Option<f64> {
        let spread = self.spread_bps / 10_000.0 * self.mid_price;
        self.queue_tracker.get_option_value(oid, spread)
    }

    /// Calculate priority premium for an order.
    pub fn priority_premium(&self, oid: u64) -> Option<f64> {
        let spread = self.spread_bps;
        self.queue_tracker.priority_premium(oid, spread, self.config.priority_premium_multiplier)
    }

    /// Check if order should hold position.
    pub fn should_hold(&self, oid: u64, drift_bps: f64) -> bool {
        self.queue_tracker.should_hold(oid, drift_bps, self.spread_bps)
    }

    /// Check if flow is toxic for a specific side.
    pub fn is_toxic_for_side(&self, is_bid: bool) -> bool {
        self.toxicity_estimator.is_toxic_for_side(is_bid)
    }

    /// Check if estimator is warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.tick_count >= self.min_warmup_ticks
    }

    /// Get reference to queue tracker.
    pub fn queue_tracker(&self) -> &VirtualQueueTracker {
        &self.queue_tracker
    }

    /// Get mutable reference to queue tracker.
    pub fn queue_tracker_mut(&mut self) -> &mut VirtualQueueTracker {
        &mut self.queue_tracker
    }

    /// Get reference to toxicity estimator.
    pub fn toxicity_estimator(&self) -> &ToxicityEstimator {
        &self.toxicity_estimator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volatility_state_from_ratio() {
        assert_eq!(
            VolatilityState::from_ratio(0.3, 0.5, 2.0),
            VolatilityState::Quiet
        );
        assert_eq!(
            VolatilityState::from_ratio(1.0, 0.5, 2.0),
            VolatilityState::Normal
        );
        assert_eq!(
            VolatilityState::from_ratio(3.0, 0.5, 2.0),
            VolatilityState::Turbulent
        );
    }

    #[test]
    fn test_state_estimator_basic() {
        let mut estimator = StateEstimator::default_config();

        // Update book
        estimator.update_book(100.0, 100.1);
        let state = estimator.market_state();
        assert!((state.mid_price - 100.05).abs() < 0.01);
        assert!(state.spread_bps > 0.0);

        // Update volatility
        estimator.update_volatility(0.0002);
        let state = estimator.market_state();
        assert!((state.sigma - 0.0002).abs() < 0.0001);
    }

    #[test]
    fn test_order_tracking() {
        let mut estimator = StateEstimator::default_config();

        // Place order
        estimator.order_placed(1, 100.0, 1.0, 10.0, true);
        assert!(estimator.get_priority(1).is_some());

        // Simulate trades
        estimator.on_trade(100.0, 5.0, true);
        let pi = estimator.get_priority(1).unwrap();
        assert!(pi < 1.0);

        // Remove order
        estimator.order_removed(1);
        assert!(estimator.get_priority(1).is_none());
    }

    #[test]
    fn test_modify_threshold() {
        let estimator = StateEstimator::default_config();
        let state = estimator.market_state();
        let config = PAICConfig::default();

        // High priority (π ≈ 0) should have higher threshold
        let threshold_high = state.modify_threshold(0.1, &config);
        // Low priority (π ≈ 1) should have lower threshold
        let threshold_low = state.modify_threshold(0.9, &config);

        assert!(threshold_high > threshold_low);
    }
}
