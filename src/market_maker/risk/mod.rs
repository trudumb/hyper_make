//! Unified risk management system.
//!
//! This module provides a single source of truth for risk state and evaluation.
//!
//! # Architecture
//!
//! ```text
//! +------------------+
//! |    RiskState     |  <- Unified snapshot of all risk-relevant data
//! +------------------+
//!          |
//!          v
//! +------------------+
//! |  RiskAggregator  |  <- Evaluates all monitors
//! +------------------+
//!     |    |    |
//!     v    v    v
//! +------+ +------+ +------+
//! |Loss  | |Pos.  | |Cascade|  <- RiskMonitor implementations
//! +------+ +------+ +------+
//! ```
//!
//! # Components
//!
//! - [`RiskState`]: Unified snapshot of all risk-relevant data
//! - [`RiskAggregator`]: Evaluates all monitors against risk state
//! - [`RiskMonitor`]: Trait for implementing risk monitors
//! - [`RiskLimits`] / [`RiskChecker`]: Position and order size limit checking
//! - [`CircuitBreakerMonitor`]: Market condition circuit breakers
//! - [`DrawdownTracker`]: Equity drawdown tracking and position sizing
//! - [`KillSwitch`]: Emergency shutdown mechanism
//!
//! # Benefits
//!
//! - **Single state snapshot**: All monitors evaluate the same point-in-time data
//! - **Extensible**: Add new monitors without modifying aggregator
//! - **Testable**: Monitors can be tested in isolation
//! - **Defense-first**: Defaults to safer options when uncertain

mod aggregator;
mod circuit_breaker;
mod drawdown;
mod inventory_governor;
mod kill_switch;
mod limits;
mod monitor;
pub mod monitors;
mod position_guard;
mod reentry;
mod state;

pub use aggregator::{AggregatedRisk, RiskAggregator};
pub use inventory_governor::{
    InventoryGovernor, PositionAssessment, PositionBudget, PositionLimits, PositionZone,
};
pub use circuit_breaker::{
    CircuitBreakerAction, CircuitBreakerConfig, CircuitBreakerMonitor, CircuitBreakerType,
};
pub use drawdown::{DrawdownConfig, DrawdownLevel, DrawdownSummary, DrawdownTracker};
pub use kill_switch::*;
pub use limits::{RiskCheckResult, RiskChecker, RiskLimits};
pub use monitor::{RiskAction, RiskAssessment, RiskMonitor, RiskMonitorBox, RiskSeverity};
pub use position_guard::{OrderEntryCheck, PositionGuard, PositionGuardConfig, PositionGuardSummary};
// Note: Side enum is NOT re-exported to avoid conflict with tracking::Side
// Use position_guard::Side explicitly if needed
pub use reentry::{ReentryConfig, ReentryManager, ReentryPhase, ReentrySummary};
pub use state::RiskState;

/// Unified emergency decision — resolves conflicts between 6 competing mechanisms.
/// Priority: Kill > Governor > Circuit > Overlay > Cascade > ReduceOnly.
/// Key invariant: Position-reducing quotes NEVER blocked when reduce_only=true.
#[derive(Debug, Clone)]
pub struct EmergencyDecision {
    /// Whether only position-reducing orders are allowed
    pub reduce_only: bool,
    /// Additive spread widening (max of all mechanisms, NOT product)
    pub spread_addon_bps: f64,
    /// Whether bid quotes are allowed
    pub allowed_bid: bool,
    /// Whether ask quotes are allowed
    pub allowed_ask: bool,
    /// Human-readable reason for the decision
    pub reason: String,
}

impl Default for EmergencyDecision {
    fn default() -> Self {
        Self {
            reduce_only: false,
            spread_addon_bps: 0.0,
            allowed_bid: true,
            allowed_ask: true,
            reason: "normal".to_string(),
        }
    }
}

/// Input parameters for emergency decision evaluation.
#[derive(Debug, Clone, Default)]
pub struct EmergencyInput {
    pub kill_switch_triggered: bool,
    pub governor_reduce_only: bool,
    pub circuit_breaker_active: bool,
    pub circuit_breaker_spread_addon_bps: f64,
    pub fill_cascade_active: bool,
    pub fill_cascade_spread_addon_bps: f64,
    pub risk_overlay_spread_addon_bps: f64,
    pub position: f64,
}

impl EmergencyDecision {
    /// Evaluate emergency state from all sources.
    /// Priority: Kill > Governor > Circuit > Cascade > ReduceOnly.
    ///
    /// INVARIANT: When reduce_only is true, the position-reducing side
    /// (asks when long, bids when short) is NEVER blocked.
    pub fn evaluate(input: &EmergencyInput) -> Self {
        let EmergencyInput {
            kill_switch_triggered,
            governor_reduce_only,
            circuit_breaker_active,
            circuit_breaker_spread_addon_bps,
            fill_cascade_active,
            fill_cascade_spread_addon_bps,
            risk_overlay_spread_addon_bps,
            position,
        } = *input;
        // Kill switch: cancel everything — highest priority
        if kill_switch_triggered {
            return Self {
                reduce_only: true,
                spread_addon_bps: 0.0,
                allowed_bid: false,
                allowed_ask: false,
                reason: "kill_switch".to_string(),
            };
        }

        let mut decision = Self::default();
        let mut reasons: Vec<&str> = Vec::new();

        // Governor: reduce-only
        if governor_reduce_only {
            decision.reduce_only = true;
            reasons.push("governor_reduce_only");
        }

        // Circuit breaker: widen spreads (additive max)
        if circuit_breaker_active {
            decision.spread_addon_bps =
                decision.spread_addon_bps.max(circuit_breaker_spread_addon_bps);
            reasons.push("circuit_breaker");
        }

        // Risk overlay: widen spreads (additive max)
        if risk_overlay_spread_addon_bps > 0.0 {
            decision.spread_addon_bps =
                decision.spread_addon_bps.max(risk_overlay_spread_addon_bps);
            reasons.push("risk_overlay");
        }

        // Fill cascade: widen spreads (additive max)
        if fill_cascade_active {
            decision.spread_addon_bps =
                decision.spread_addon_bps.max(fill_cascade_spread_addon_bps);
            reasons.push("fill_cascade");
        }

        // Apply reduce-only side restrictions LAST
        // INVARIANT: reducing side never blocked
        if decision.reduce_only {
            if position > 0.0 {
                // Long → only asks allowed (to reduce)
                decision.allowed_bid = false;
                // allowed_ask stays true — NEVER block reducing side
            } else if position < 0.0 {
                // Short → only bids allowed (to reduce)
                decision.allowed_ask = false;
                // allowed_bid stays true — NEVER block reducing side
            }
            // Flat (position == 0.0) → both sides allowed (nothing to reduce)
        }

        decision.reason = if reasons.is_empty() {
            "normal".to_string()
        } else {
            reasons.join("+")
        };

        decision
    }
}

#[cfg(test)]
mod emergency_tests {
    use super::*;

    /// Helper: construct EmergencyInput from positional args for test convenience.
    fn eval(
        kill: bool, governor: bool, cb: bool, cb_bps: f64,
        cascade: bool, cascade_bps: f64, overlay_bps: f64, pos: f64,
    ) -> EmergencyDecision {
        EmergencyDecision::evaluate(&EmergencyInput {
            kill_switch_triggered: kill,
            governor_reduce_only: governor,
            circuit_breaker_active: cb,
            circuit_breaker_spread_addon_bps: cb_bps,
            fill_cascade_active: cascade,
            fill_cascade_spread_addon_bps: cascade_bps,
            risk_overlay_spread_addon_bps: overlay_bps,
            position: pos,
        })
    }

    // === Kill switch tests ===

    #[test]
    fn test_kill_switch_overrides_everything() {
        let d = eval(
            true,  // kill switch triggered
            true,  // governor also says reduce-only
            true,  // circuit breaker also active
            50.0,  // circuit breaker addon
            true,  // cascade active
            30.0,  // cascade addon
            20.0,  // risk overlay addon
            1.5,   // long position
        );

        assert!(!d.allowed_bid, "kill switch must block bids");
        assert!(!d.allowed_ask, "kill switch must block asks");
        assert!(d.reduce_only);
        assert_eq!(d.reason, "kill_switch");
        // Kill switch doesn't apply spread addon — no point, everything cancelled
        assert_eq!(d.spread_addon_bps, 0.0);
    }

    #[test]
    fn test_kill_switch_blocks_even_reducing_side() {
        // Kill switch is the ONLY mechanism that blocks the reducing side
        let d = eval(
            true, false, false, 0.0, false, 0.0, 0.0, 5.0,
        );
        assert!(!d.allowed_ask, "kill switch blocks even reducing asks");
        assert!(!d.allowed_bid);
    }

    // === Governor reduce-only tests ===

    #[test]
    fn test_governor_reduce_only_long_preserves_asks() {
        let d = eval(
            false, true, false, 0.0, false, 0.0, 0.0, 2.0,
        );

        assert!(d.reduce_only);
        assert!(!d.allowed_bid, "long + reduce-only → no bids");
        assert!(d.allowed_ask, "long + reduce-only → asks MUST be allowed");
        assert_eq!(d.reason, "governor_reduce_only");
    }

    #[test]
    fn test_governor_reduce_only_short_preserves_bids() {
        let d = eval(
            false, true, false, 0.0, false, 0.0, 0.0, -3.0,
        );

        assert!(d.reduce_only);
        assert!(d.allowed_bid, "short + reduce-only → bids MUST be allowed");
        assert!(!d.allowed_ask, "short + reduce-only → no asks");
    }

    #[test]
    fn test_governor_reduce_only_flat_allows_both() {
        let d = eval(
            false, true, false, 0.0, false, 0.0, 0.0, 0.0,
        );

        assert!(d.reduce_only);
        assert!(d.allowed_bid, "flat + reduce-only → bids allowed");
        assert!(d.allowed_ask, "flat + reduce-only → asks allowed");
    }

    // === Fill cascade + reduce-only interaction (THE KEY BUG) ===

    #[test]
    fn test_cascade_plus_reduce_only_no_paralysis_long() {
        // The bug: InventoryGovernor says "reduce-only, keep closing quotes"
        // but fill cascade would clear the closing side → stuck
        // Fix: reduce-only reducing side is NEVER blocked
        let d = eval(
            false,
            true,   // governor: reduce-only
            false,
            0.0,
            true,   // cascade active
            15.0,   // cascade widens
            0.0,
            3.0,    // long position
        );

        assert!(d.reduce_only);
        assert!(!d.allowed_bid, "long → bids blocked");
        assert!(d.allowed_ask, "CRITICAL: asks (reducing side) must survive cascade");
        assert_eq!(d.spread_addon_bps, 15.0);
        assert!(d.reason.contains("governor_reduce_only"));
        assert!(d.reason.contains("fill_cascade"));
    }

    #[test]
    fn test_cascade_plus_reduce_only_no_paralysis_short() {
        let d = eval(
            false, true, false, 0.0, true, 10.0, 0.0, -2.0,
        );

        assert!(d.reduce_only);
        assert!(d.allowed_bid, "CRITICAL: bids (reducing side) must survive cascade");
        assert!(!d.allowed_ask, "short → asks blocked");
        assert_eq!(d.spread_addon_bps, 10.0);
    }

    // === Spread addon tests ===

    #[test]
    fn test_spread_addon_takes_max_not_product() {
        let d = eval(
            false,
            false,
            true,  // circuit breaker
            20.0,  // circuit breaker addon
            true,  // cascade
            30.0,  // cascade addon
            15.0,  // risk overlay addon
            0.0,
        );

        // Max(20, 30, 15) = 30, NOT 20 * 30 * 15
        assert_eq!(d.spread_addon_bps, 30.0);
        assert!(!d.reduce_only);
        assert!(d.allowed_bid);
        assert!(d.allowed_ask);
    }

    #[test]
    fn test_circuit_breaker_only_widens_no_side_restriction() {
        let d = eval(
            false, false, true, 25.0, false, 0.0, 0.0, 1.0,
        );

        assert!(!d.reduce_only);
        assert!(d.allowed_bid);
        assert!(d.allowed_ask);
        assert_eq!(d.spread_addon_bps, 25.0);
        assert_eq!(d.reason, "circuit_breaker");
    }

    #[test]
    fn test_risk_overlay_only_widens() {
        let d = eval(
            false, false, false, 0.0, false, 0.0, 12.5, 0.0,
        );

        assert_eq!(d.spread_addon_bps, 12.5);
        assert_eq!(d.reason, "risk_overlay");
        assert!(d.allowed_bid);
        assert!(d.allowed_ask);
    }

    #[test]
    fn test_zero_risk_overlay_not_reported() {
        let d = eval(
            false, false, false, 0.0, false, 0.0, 0.0, 0.0,
        );

        assert_eq!(d.reason, "normal");
        assert_eq!(d.spread_addon_bps, 0.0);
    }

    // === Default / normal state ===

    #[test]
    fn test_default_normal_operation() {
        let d = eval(
            false, false, false, 0.0, false, 0.0, 0.0, 0.5,
        );

        assert!(!d.reduce_only);
        assert!(d.allowed_bid);
        assert!(d.allowed_ask);
        assert_eq!(d.spread_addon_bps, 0.0);
        assert_eq!(d.reason, "normal");
    }

    #[test]
    fn test_default_impl() {
        let d = EmergencyDecision::default();
        assert!(!d.reduce_only);
        assert!(d.allowed_bid);
        assert!(d.allowed_ask);
        assert_eq!(d.spread_addon_bps, 0.0);
        assert_eq!(d.reason, "normal");
    }

    // === Combination tests ===

    #[test]
    fn test_all_mechanisms_except_kill() {
        let d = eval(
            false,
            true,   // governor reduce-only
            true,   // circuit breaker
            20.0,
            true,   // cascade
            35.0,
            10.0,   // overlay
            -1.0,   // short
        );

        assert!(d.reduce_only);
        assert!(d.allowed_bid, "short → bids reduce position");
        assert!(!d.allowed_ask, "short → asks increase position");
        assert_eq!(d.spread_addon_bps, 35.0); // max(20, 35, 10)
        assert!(d.reason.contains("governor_reduce_only"));
        assert!(d.reason.contains("circuit_breaker"));
        assert!(d.reason.contains("fill_cascade"));
        assert!(d.reason.contains("risk_overlay"));
    }

    #[test]
    fn test_cascade_without_governor_no_side_restriction() {
        // Fill cascade alone should NOT restrict sides — only widen
        let d = eval(
            false, false, false, 0.0, true, 20.0, 0.0, 5.0,
        );

        assert!(!d.reduce_only);
        assert!(d.allowed_bid, "cascade alone doesn't block sides");
        assert!(d.allowed_ask, "cascade alone doesn't block sides");
        assert_eq!(d.spread_addon_bps, 20.0);
    }

    #[test]
    fn test_governor_with_circuit_breaker_spreads_compound() {
        let d = eval(
            false, true, true, 40.0, false, 0.0, 0.0, 2.0,
        );

        assert!(d.reduce_only);
        assert!(!d.allowed_bid, "long → bids blocked");
        assert!(d.allowed_ask, "long → asks (reducing) allowed");
        assert_eq!(d.spread_addon_bps, 40.0);
    }
}
