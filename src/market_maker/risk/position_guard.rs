//! Preemptive Position Guard with Inventory Skew.
//!
//! This module provides proactive position management that:
//! 1. Reduces quote capacity BEFORE position limits are breached
//! 2. Applies inventory skew based on current position (Guéant-Lehalle)
//! 3. Pulls quotes on one side when approaching limits
//!
//! # Problem Solved
//!
//! The existing PositionMonitor only checks AFTER fills occur, allowing
//! position to breach limits (e.g., 406% of limit in HIP-3 logs).
//!
//! This guard provides:
//! - **Preemptive capacity reduction**: Reduce headroom as position grows
//! - **Asymmetric skew**: Shift spreads away from position direction
//! - **Side pulling**: Remove quotes on one side at high utilization
//!
//! # Example
//!
//! ```ignore
//! use crate::market_maker::tracking::Side;
//!
//! let guard = PositionGuard::new(0.5, 0.15); // max_position=0.5, gamma=0.15
//!
//! // Current position is +0.35 (70% of limit)
//! guard.update_position(0.35);
//!
//! // Get capacity per side (buy reduced since we're long)
//! let buy_capacity = guard.quote_capacity(Side::Buy);   // ~0.075
//! let sell_capacity = guard.quote_capacity(Side::Sell); // ~0.15
//!
//! // Should we pull buy quotes entirely?
//! if guard.should_pull_side(Side::Buy) {
//!     // At 90%+ utilization, pull the side that would increase position
//! }
//!
//! // Get inventory skew to apply to spreads
//! let skew_bps = guard.inventory_skew_bps(); // ~+35 bps (widen asks, tighten bids)
//! ```

use crate::market_maker::tracking::Side;
use tracing::debug;

/// Configuration for position guard thresholds.
#[derive(Debug, Clone)]
pub struct PositionGuardConfig {
    /// Maximum position (contracts) - the "soft" limit
    pub max_position: f64,
    /// Risk aversion parameter for skew calculation (Guéant-Lehalle γ)
    pub gamma: f64,
    /// Warning threshold (fraction of max_position)
    /// Start reducing capacity at this level
    pub warning_threshold: f64,
    /// Pull threshold (fraction of max_position)
    /// Pull quotes on one side at this level
    pub pull_threshold: f64,
    /// Maximum skew in basis points (GL formula)
    pub max_skew_bps: f64,
    /// Direct linear skew at max position (bps).
    /// Added on top of GL formula to provide meaningful skew even when
    /// gamma*sigma^2*q*tau is negligible (~0.28 bps).
    /// At 50% position utilization, adds direct_skew_max_bps/2 to GL skew.
    pub direct_skew_max_bps: f64,
    /// Volatility estimate for skew calculation (bps per sqrt(second))
    pub sigma_bps: f64,
    /// Time horizon for skew calculation (seconds)
    pub tau_seconds: f64,
    /// Hard entry gate threshold (fraction of max_position).
    /// Orders are rejected if worst-case position exceeds this fraction of max_position.
    /// Default: 0.95 (reject when >95% utilized).
    pub hard_entry_threshold: f64,
}

impl Default for PositionGuardConfig {
    fn default() -> Self {
        Self {
            max_position: 1.0,
            gamma: 0.15,
            warning_threshold: 0.7,
            pull_threshold: 0.9,
            max_skew_bps: 100.0,
            direct_skew_max_bps: 15.0, // 15 bps at max position
            sigma_bps: 50.0,     // 50 bps volatility
            tau_seconds: 300.0,  // 5 minute horizon
            hard_entry_threshold: 0.95,
        }
    }
}

/// Result of a pre-order entry check.
#[derive(Debug, Clone, PartialEq)]
pub enum OrderEntryCheck {
    /// Order is allowed.
    Allowed,
    /// Order is rejected because worst-case position exceeds hard limit.
    Rejected {
        current_position: f64,
        proposed_size: f64,
        worst_case_position: f64,
        hard_limit: f64,
        reason: String,
    },
}

impl OrderEntryCheck {
    /// Returns true if the order is allowed.
    pub fn is_allowed(&self) -> bool {
        matches!(self, OrderEntryCheck::Allowed)
    }
}

/// Preemptive position guard with inventory skew.
///
/// Provides proactive position management to prevent limit breaches.
#[derive(Debug)]
pub struct PositionGuard {
    config: PositionGuardConfig,
    /// Current position (signed: positive = long, negative = short)
    position: f64,
    /// Last computed skew in basis points
    last_skew_bps: f64,
}

impl PositionGuard {
    /// Create a new position guard with default config.
    pub fn new(max_position: f64, gamma: f64) -> Self {
        Self::with_config(PositionGuardConfig {
            max_position,
            gamma,
            ..Default::default()
        })
    }

    /// Create with full configuration.
    pub fn with_config(config: PositionGuardConfig) -> Self {
        Self {
            config,
            position: 0.0,
            last_skew_bps: 0.0,
        }
    }

    /// Update current position.
    pub fn update_position(&mut self, position: f64) {
        self.position = position;
        self.last_skew_bps = self.compute_inventory_skew_bps();
    }

    /// Update volatility estimate.
    pub fn update_sigma(&mut self, sigma_bps: f64) {
        self.config.sigma_bps = sigma_bps.max(1.0);
    }

    /// Update time horizon.
    pub fn update_tau(&mut self, tau_seconds: f64) {
        self.config.tau_seconds = tau_seconds.max(1.0);
    }

    /// Get current position utilization [0, 1+].
    ///
    /// Values > 1.0 indicate limit breach.
    pub fn utilization(&self) -> f64 {
        if self.config.max_position <= 0.0 {
            return 0.0;
        }
        self.position.abs() / self.config.max_position
    }

    /// Get remaining capacity for a given side.
    ///
    /// This implements preemptive capacity reduction:
    /// - Full capacity when utilization < warning_threshold
    /// - Reduced capacity between warning and pull thresholds
    /// - Zero capacity at pull_threshold (quotes should be pulled)
    ///
    /// Additionally, if position is already in the same direction as the side,
    /// capacity is halved to encourage mean reversion.
    pub fn quote_capacity(&self, side: Side) -> f64 {
        let utilization = self.utilization();
        let headroom = (self.config.max_position - self.position.abs()).max(0.0);

        // Base capacity is remaining headroom
        let mut capacity = headroom;

        // Reduce capacity if above warning threshold
        if utilization > self.config.warning_threshold {
            // Linear reduction from warning to pull threshold
            let reduction_range = self.config.pull_threshold - self.config.warning_threshold;
            if reduction_range > 0.0 {
                let progress = ((utilization - self.config.warning_threshold) / reduction_range)
                    .clamp(0.0, 1.0);
                capacity *= 1.0 - progress;
            }
        }

        // Halve capacity if side would increase position in same direction
        let same_direction = match side {
            Side::Buy if self.position > 0.0 => true,
            Side::Sell if self.position < 0.0 => true,
            _ => false,
        };

        if same_direction {
            capacity *= 0.5;
        }

        capacity
    }

    /// Check if quotes should be pulled on a given side.
    ///
    /// Returns true when:
    /// - Utilization >= pull_threshold (90% by default)
    /// - AND the side would increase position in the same direction
    pub fn should_pull_side(&self, side: Side) -> bool {
        let utilization = self.utilization();

        if utilization < self.config.pull_threshold {
            return false;
        }

        // Only pull the side that would increase position
        match side {
            Side::Buy if self.position > 0.0 => true,  // Long, pull buys
            Side::Sell if self.position < 0.0 => true, // Short, pull sells
            _ => false,
        }
    }

    /// Check if either side should be pulled.
    pub fn should_pull_any(&self) -> bool {
        self.should_pull_side(Side::Buy) || self.should_pull_side(Side::Sell)
    }

    /// Hard pre-order entry gate.
    ///
    /// Checks whether placing an order of `proposed_size` on `side` would
    /// cause the worst-case position to exceed `hard_entry_threshold` of max_position.
    ///
    /// This is a stateless check using the provided `current_position` (not cached state)
    /// to avoid stale-data races.
    ///
    /// Orders that REDUCE position are always allowed.
    pub fn check_order_entry(
        &self,
        current_position: f64,
        proposed_size: f64,
        side: Side,
    ) -> OrderEntryCheck {
        let worst_case = match side {
            Side::Buy => current_position + proposed_size,
            Side::Sell => current_position - proposed_size,
        };

        let hard_limit = self.config.max_position * self.config.hard_entry_threshold;

        // Allow orders that reduce position toward zero
        if worst_case.abs() < current_position.abs() {
            return OrderEntryCheck::Allowed;
        }

        if worst_case.abs() > hard_limit {
            OrderEntryCheck::Rejected {
                current_position,
                proposed_size,
                worst_case_position: worst_case,
                hard_limit,
                reason: format!(
                    "Hard entry gate: worst-case position {:.6} exceeds {:.1}% limit ({:.6})",
                    worst_case.abs(),
                    self.config.hard_entry_threshold * 100.0,
                    hard_limit,
                ),
            }
        } else {
            OrderEntryCheck::Allowed
        }
    }

    /// Get inventory skew in basis points.
    ///
    /// Implements Guéant-Lehalle-Fernandez-Tapia inventory skew:
    /// ```text
    /// skew = γ × σ² × q × τ
    /// ```
    ///
    /// Where:
    /// - γ = risk aversion (higher = more skew)
    /// - σ = volatility (bps per sqrt(second))
    /// - q = inventory (signed position / max_position)
    /// - τ = time horizon (seconds)
    ///
    /// Positive skew = widen asks, tighten bids (encourage selling)
    /// Negative skew = widen bids, tighten asks (encourage buying)
    ///
    /// NOTE: This is now DEPRECATED for quote skewing purposes.
    /// Inventory skew is handled solely by the GLFT q-term in `glft.rs`.
    /// The value computed here was previously double-counted with both
    /// the GLFT q-term and signal_integration.inventory_skew_bps.
    /// The quote_engine no longer adds this to lead_lag_signal_bps.
    pub fn inventory_skew_bps(&self) -> f64 {
        self.last_skew_bps
    }

    /// Compute inventory skew (internal).
    ///
    /// Combines two components:
    /// 1. Guéant-Lehalle formula: γ × σ² × q × τ (typically ~0.28 bps — negligible)
    /// 2. Direct linear skew: position_fraction × direct_skew_max_bps (up to 15 bps)
    ///
    /// The direct term ensures meaningful skew even when GL parameters produce
    /// negligible values. At 50% position utilization → 7.5 bps skew.
    fn compute_inventory_skew_bps(&self) -> f64 {
        if self.config.max_position <= 0.0 {
            return 0.0;
        }

        // Normalized inventory [-1, 1]
        let q = self.position / self.config.max_position;

        // Component 1: Guéant-Lehalle skew: γ × σ² × q × τ
        let sigma = self.config.sigma_bps / 10000.0; // Convert bps to fraction
        let sigma_squared = sigma * sigma;
        let gl_skew = self.config.gamma * sigma_squared * q * self.config.tau_seconds * 10000.0;

        // Component 2: Direct linear skew — position-proportional
        let position_fraction = q.clamp(-1.0, 1.0);
        let direct_skew = position_fraction * self.config.direct_skew_max_bps;

        // Combine and clamp
        let skew_bps = gl_skew + direct_skew;
        skew_bps.clamp(-self.config.max_skew_bps, self.config.max_skew_bps)
    }

    /// Get bid-side spread adjustment in basis points.
    ///
    /// Negative skew means tighten bids (more aggressive buying).
    /// Positive skew means widen bids (less aggressive buying).
    pub fn bid_adjustment_bps(&self) -> f64 {
        // Positive position (long) → positive skew → widen bids
        self.last_skew_bps
    }

    /// Get ask-side spread adjustment in basis points.
    ///
    /// Positive skew means tighten asks (more aggressive selling).
    /// Negative skew means widen asks (less aggressive selling).
    pub fn ask_adjustment_bps(&self) -> f64 {
        // Positive position (long) → positive skew → tighten asks
        -self.last_skew_bps
    }

    /// Get a summary of current guard state.
    pub fn summary(&self) -> PositionGuardSummary {
        PositionGuardSummary {
            position: self.position,
            max_position: self.config.max_position,
            utilization: self.utilization(),
            skew_bps: self.last_skew_bps,
            buy_capacity: self.quote_capacity(Side::Buy),
            sell_capacity: self.quote_capacity(Side::Sell),
            should_pull_buy: self.should_pull_side(Side::Buy),
            should_pull_sell: self.should_pull_side(Side::Sell),
            warning_threshold: self.config.warning_threshold,
            pull_threshold: self.config.pull_threshold,
        }
    }

    /// Log current state for diagnostics.
    pub fn log_state(&self) {
        let summary = self.summary();
        debug!(
            position = %format!("{:.4}", summary.position),
            utilization = %format!("{:.1}%", summary.utilization * 100.0),
            skew_bps = %format!("{:.2}", summary.skew_bps),
            buy_capacity = %format!("{:.4}", summary.buy_capacity),
            sell_capacity = %format!("{:.4}", summary.sell_capacity),
            pull_buy = summary.should_pull_buy,
            pull_sell = summary.should_pull_sell,
            "[PositionGuard] State"
        );
    }
}

impl Default for PositionGuard {
    fn default() -> Self {
        Self::with_config(PositionGuardConfig::default())
    }
}

/// Summary of position guard state.
#[derive(Debug, Clone)]
pub struct PositionGuardSummary {
    pub position: f64,
    pub max_position: f64,
    pub utilization: f64,
    pub skew_bps: f64,
    pub buy_capacity: f64,
    pub sell_capacity: f64,
    pub should_pull_buy: bool,
    pub should_pull_sell: bool,
    pub warning_threshold: f64,
    pub pull_threshold: f64,
}

impl PositionGuardSummary {
    /// Check if position is in warning zone.
    pub fn in_warning_zone(&self) -> bool {
        self.utilization >= self.warning_threshold && self.utilization < self.pull_threshold
    }

    /// Check if position is in pull zone.
    pub fn in_pull_zone(&self) -> bool {
        self.utilization >= self.pull_threshold
    }

    /// Check if position is over limit.
    pub fn over_limit(&self) -> bool {
        self.utilization > 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_creation() {
        let guard = PositionGuard::new(1.0, 0.15);
        assert_eq!(guard.position, 0.0);
        assert_eq!(guard.utilization(), 0.0);
    }

    #[test]
    fn test_utilization() {
        let mut guard = PositionGuard::new(1.0, 0.15);

        guard.update_position(0.5);
        assert!((guard.utilization() - 0.5).abs() < 1e-10);

        guard.update_position(-0.5);
        assert!((guard.utilization() - 0.5).abs() < 1e-10);

        guard.update_position(1.5); // Over limit
        assert!((guard.utilization() - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_quote_capacity_flat() {
        let guard = PositionGuard::new(1.0, 0.15);
        // With zero position, full capacity on both sides
        assert!((guard.quote_capacity(Side::Buy) - 1.0).abs() < 1e-10);
        assert!((guard.quote_capacity(Side::Sell) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quote_capacity_long() {
        let mut guard = PositionGuard::new(1.0, 0.15);
        guard.update_position(0.5); // 50% long

        // Buy capacity should be halved (same direction)
        let buy_cap = guard.quote_capacity(Side::Buy);
        let sell_cap = guard.quote_capacity(Side::Sell);

        assert!(buy_cap < sell_cap);
        assert!((buy_cap - 0.25).abs() < 1e-10); // 0.5 headroom * 0.5 same-direction
        assert!((sell_cap - 0.5).abs() < 1e-10); // 0.5 headroom
    }

    #[test]
    fn test_quote_capacity_warning_zone() {
        let mut guard = PositionGuard::with_config(PositionGuardConfig {
            max_position: 1.0,
            warning_threshold: 0.7,
            pull_threshold: 0.9,
            ..Default::default()
        });

        // At 80% utilization (in warning zone)
        guard.update_position(0.8);

        let buy_cap = guard.quote_capacity(Side::Buy);
        let sell_cap = guard.quote_capacity(Side::Sell);

        // Capacity should be reduced
        assert!(buy_cap < 0.1);
        assert!(sell_cap < 0.2);
    }

    #[test]
    fn test_should_pull_side() {
        let mut guard = PositionGuard::with_config(PositionGuardConfig {
            max_position: 1.0,
            pull_threshold: 0.9,
            ..Default::default()
        });

        // Below threshold - no pulling
        guard.update_position(0.8);
        assert!(!guard.should_pull_side(Side::Buy));
        assert!(!guard.should_pull_side(Side::Sell));

        // At threshold, long position - pull buys
        guard.update_position(0.95);
        assert!(guard.should_pull_side(Side::Buy));
        assert!(!guard.should_pull_side(Side::Sell));

        // At threshold, short position - pull sells
        guard.update_position(-0.95);
        assert!(!guard.should_pull_side(Side::Buy));
        assert!(guard.should_pull_side(Side::Sell));
    }

    #[test]
    fn test_inventory_skew() {
        let mut guard = PositionGuard::with_config(PositionGuardConfig {
            max_position: 1.0,
            gamma: 0.15,
            sigma_bps: 50.0,
            tau_seconds: 300.0,
            max_skew_bps: 100.0,
            ..Default::default()
        });

        // Zero position - zero skew
        guard.update_position(0.0);
        assert!((guard.inventory_skew_bps()).abs() < 1e-10);

        // Long position - positive skew (widen bids, tighten asks)
        guard.update_position(0.5);
        let skew = guard.inventory_skew_bps();
        assert!(skew > 0.0);

        // Short position - negative skew (tighten bids, widen asks)
        guard.update_position(-0.5);
        let skew = guard.inventory_skew_bps();
        assert!(skew < 0.0);
    }

    #[test]
    fn test_direct_skew_at_half_position() {
        let mut guard = PositionGuard::with_config(PositionGuardConfig {
            max_position: 1.0,
            gamma: 0.15,
            direct_skew_max_bps: 15.0,
            sigma_bps: 50.0,
            tau_seconds: 300.0,
            max_skew_bps: 100.0,
            ..Default::default()
        });

        // At 50% position:
        //   direct_skew = 0.5 * 15.0 = 7.5 bps
        //   GL = 0.15 * (0.005)^2 * 0.5 * 300 * 10000 = 5.625 bps
        //   total = ~13.1 bps
        guard.update_position(0.5);
        let skew = guard.inventory_skew_bps();
        assert!(skew > 12.0, "Expected >12 bps skew at 50% position, got {skew}");
        assert!(skew < 14.0, "Expected <14 bps skew at 50% position, got {skew}");

        // Negative position should give negative skew of same magnitude
        guard.update_position(-0.5);
        let skew = guard.inventory_skew_bps();
        assert!(skew < -12.0, "Expected <-12 bps skew at -50% position, got {skew}");
    }

    #[test]
    fn test_direct_skew_at_max_position() {
        let mut guard = PositionGuard::with_config(PositionGuardConfig {
            max_position: 10.0,
            gamma: 0.15,
            direct_skew_max_bps: 15.0,
            sigma_bps: 50.0,
            tau_seconds: 300.0,
            max_skew_bps: 100.0,
            ..Default::default()
        });

        // At 100% position utilization: direct_skew = 1.0 * 15.0 = 15.0 bps + GL
        guard.update_position(10.0);
        let skew = guard.inventory_skew_bps();
        assert!(skew > 15.0, "Expected >15 bps skew at max position, got {skew}");
    }

    #[test]
    fn test_skew_clamping() {
        let mut guard = PositionGuard::with_config(PositionGuardConfig {
            max_position: 1.0,
            gamma: 1.0, // Very high gamma
            sigma_bps: 200.0, // Very high volatility
            tau_seconds: 1000.0, // Long horizon
            max_skew_bps: 50.0, // But limited skew
            ..Default::default()
        });

        guard.update_position(1.0); // Full position
        let skew = guard.inventory_skew_bps();
        assert!(skew <= 50.0); // Clamped to max
    }

    #[test]
    fn test_bid_ask_adjustments() {
        let mut guard = PositionGuard::new(1.0, 0.15);

        // Long position: widen bids, tighten asks
        guard.update_position(0.5);
        let skew = guard.inventory_skew_bps();
        assert!(skew > 0.0);
        assert_eq!(guard.bid_adjustment_bps(), skew);   // Widen bids
        assert_eq!(guard.ask_adjustment_bps(), -skew);  // Tighten asks

        // Short position: tighten bids, widen asks
        guard.update_position(-0.5);
        let skew = guard.inventory_skew_bps();
        assert!(skew < 0.0);
        assert_eq!(guard.bid_adjustment_bps(), skew);   // Tighten bids
        assert_eq!(guard.ask_adjustment_bps(), -skew);  // Widen asks
    }

    #[test]
    fn test_summary() {
        let mut guard = PositionGuard::with_config(PositionGuardConfig {
            max_position: 1.0,
            warning_threshold: 0.7,
            pull_threshold: 0.9,
            ..Default::default()
        });

        guard.update_position(0.8);
        let summary = guard.summary();

        assert!((summary.position - 0.8).abs() < 1e-10);
        assert!((summary.utilization - 0.8).abs() < 1e-10);
        assert!(summary.in_warning_zone());
        assert!(!summary.in_pull_zone());
        assert!(!summary.over_limit());
    }

    // ====================================================================
    // Hard Entry Gate Tests
    // ====================================================================

    #[test]
    fn test_entry_gate_small_order_allowed() {
        let guard = PositionGuard::with_config(PositionGuardConfig {
            max_position: 1.0,
            hard_entry_threshold: 0.95,
            ..Default::default()
        });
        let check = guard.check_order_entry(0.0, 0.5, Side::Buy);
        assert!(check.is_allowed());
    }

    #[test]
    fn test_entry_gate_buy_overshoot_rejected() {
        let guard = PositionGuard::with_config(PositionGuardConfig {
            max_position: 1.0,
            hard_entry_threshold: 0.95,
            ..Default::default()
        });
        // Already at 0.9, buying 0.1 would put us at 1.0 > 0.95 limit
        let check = guard.check_order_entry(0.9, 0.1, Side::Buy);
        assert!(!check.is_allowed());
        if let OrderEntryCheck::Rejected { worst_case_position, hard_limit, .. } = check {
            assert!((worst_case_position - 1.0).abs() < 1e-10);
            assert!((hard_limit - 0.95).abs() < 1e-10);
        }
    }

    #[test]
    fn test_entry_gate_sell_overshoot_rejected() {
        let guard = PositionGuard::with_config(PositionGuardConfig {
            max_position: 1.0,
            hard_entry_threshold: 0.95,
            ..Default::default()
        });
        // Already short -0.9, selling 0.1 more would put us at -1.0
        let check = guard.check_order_entry(-0.9, 0.1, Side::Sell);
        assert!(!check.is_allowed());
    }

    #[test]
    fn test_entry_gate_reducing_position_always_allowed() {
        let guard = PositionGuard::with_config(PositionGuardConfig {
            max_position: 1.0,
            hard_entry_threshold: 0.95,
            ..Default::default()
        });
        // Long 0.99, selling reduces position → always allowed
        let check = guard.check_order_entry(0.99, 0.5, Side::Sell);
        assert!(check.is_allowed());

        // Short -0.99, buying reduces position → always allowed
        let check = guard.check_order_entry(-0.99, 0.5, Side::Buy);
        assert!(check.is_allowed());
    }

    #[test]
    fn test_entry_gate_exactly_at_boundary_allowed() {
        let guard = PositionGuard::with_config(PositionGuardConfig {
            max_position: 1.0,
            hard_entry_threshold: 0.95,
            ..Default::default()
        });
        // Worst case = exactly 0.95 (not strictly greater) → allowed
        let check = guard.check_order_entry(0.0, 0.95, Side::Buy);
        assert!(check.is_allowed());
    }

    #[test]
    fn test_entry_gate_just_over_boundary_rejected() {
        let guard = PositionGuard::with_config(PositionGuardConfig {
            max_position: 1.0,
            hard_entry_threshold: 0.95,
            ..Default::default()
        });
        // Worst case = 0.951 > 0.95 → rejected
        let check = guard.check_order_entry(0.0, 0.951, Side::Buy);
        assert!(!check.is_allowed());
    }

    #[test]
    fn test_entry_gate_custom_threshold() {
        let guard = PositionGuard::with_config(PositionGuardConfig {
            max_position: 10.0,
            hard_entry_threshold: 0.80,
            ..Default::default()
        });
        // 80% of 10.0 = 8.0 hard limit
        let check = guard.check_order_entry(7.0, 1.5, Side::Buy);
        assert!(!check.is_allowed()); // 8.5 > 8.0

        let check = guard.check_order_entry(7.0, 0.5, Side::Buy);
        assert!(check.is_allowed()); // 7.5 < 8.0
    }

    #[test]
    fn test_entry_gate_reason_string() {
        let guard = PositionGuard::with_config(PositionGuardConfig {
            max_position: 1.0,
            hard_entry_threshold: 0.95,
            ..Default::default()
        });
        let check = guard.check_order_entry(0.9, 0.2, Side::Buy);
        if let OrderEntryCheck::Rejected { reason, .. } = check {
            assert!(reason.contains("Hard entry gate"));
            assert!(reason.contains("95.0%"));
        } else {
            panic!("Expected Rejected");
        }
    }
}
