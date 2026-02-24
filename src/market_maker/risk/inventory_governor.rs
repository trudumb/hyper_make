//! Inventory Governor — enforces config.max_position as a HARD ceiling.
//!
//! This is the FIRST check in every quote cycle. Position safety is structural.
//! `config.max_position` is NEVER overridden by margin-derived limits.
//!
//! # Position Zones
//!
//! ```text
//! |--- Green (< 50%) ---|--- Yellow (50-80%) ---|--- Red (80-100%) ---|--- Kill (> 100%) ---|
//! | full two-sided       | bias toward reducing  | reduce-only         | cancel all          |
//! ```

use crate::market_maker::config::GovernorConfig;
use serde::{Deserialize, Serialize};

/// Position zone classification for the inventory governor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PositionZone {
    /// < yellow_threshold (default 50%) — full two-sided quoting
    #[default]
    Green,
    /// yellow..red (default 50-80%) — bias toward reducing, wider on increasing side
    Yellow,
    /// red..100% (default 80-100%) — reduce-only both sides
    Red,
    /// > 100% — cancel all, kill switch territory
    Kill,
}

/// Assessment result from the inventory governor.
#[derive(Debug, Clone)]
pub struct PositionAssessment {
    /// Classified zone
    pub zone: PositionZone,
    /// Maximum new exposure allowed on position-increasing side (0.0 in Red/Kill)
    pub max_new_exposure: f64,
    /// Current position ratio abs(position) / max_position
    pub position_ratio: f64,
    /// Current position (signed)
    pub position: f64,
    /// Is the position long? (for determining which side is reducing)
    pub is_long: bool,
    /// Additive spread widening (bps) for position-increasing side.
    /// 0.0 in Green, ramps in Yellow, fixed in Red/Kill.
    pub increasing_side_addon_bps: f64,
    /// Additive spread adjustment (bps) for position-REDUCING side.
    /// Negative values TIGHTEN the reducing side to attract fills (Kill/Red zones).
    /// 0.0 in Green/Yellow (no tightening needed).
    pub reducing_side_addon_bps: f64,
}

/// Position budget with a min-viable floor that prevents compound reductions
/// from crushing position below exchange minimums.
///
/// Multiple safety systems apply multiplicative reductions (regime, signal, proactive).
/// Without a floor, three independent 0.3x reductions compound to 0.027x — pushing
/// a $100 account's position below exchange min_notional ($10).
///
/// The floor is the mathematical minimum for placing one valid exchange order.
#[derive(Debug, Clone)]
pub struct PositionBudget {
    /// Starting max position (config.max_position or margin-derived)
    base_max: f64,
    /// Floor: minimum position that produces a valid exchange order
    min_viable: f64,
    /// Applied reductions: (source_name, multiplier)
    reductions: Vec<(&'static str, f64)>,
}

/// Single source of truth for position limits — computed ONCE per cycle.
/// Replaces 6 independent position limit computations scattered across the codebase.
///
/// `hard_max` (from user config) is an absolute ceiling that can NEVER be exceeded.
/// `margin_max` (from exchange solvency) may lower the effective limit.
/// `regime_fraction` and `signal_fraction` apply further reductions in adverse conditions.
#[derive(Debug, Clone, Copy)]
pub struct PositionLimits {
    /// Absolute maximum from user config — NEVER exceeded
    pub hard_max: f64,
    /// Solvency floor from exchange margin — may be lower than hard_max
    pub margin_max: f64,
    /// Regime-dependent fraction [0.3, 1.0] — reduces limits in volatile regimes
    pub regime_fraction: f64,
    /// Signal-availability fraction [0.3, 1.0] — reduces limits when signals unavailable
    pub signal_fraction: f64,
}

impl PositionLimits {
    /// The effective position limit used for all quoting decisions.
    /// Always <= hard_max (config.max_position).
    pub fn effective(&self) -> f64 {
        (self.hard_max.min(self.margin_max) * self.regime_fraction * self.signal_fraction).max(0.0)
    }

    /// Whether current position exceeds effective limit (reduce-only mode).
    pub fn is_reduce_only(&self, current_position: f64) -> bool {
        current_position.abs() >= self.effective()
    }

    /// Create with defaults (no regime/signal reduction).
    pub fn new(hard_max: f64, margin_max: f64) -> Self {
        Self {
            hard_max,
            margin_max,
            regime_fraction: 1.0,
            signal_fraction: 1.0,
        }
    }
}

impl PositionBudget {
    /// Create a new position budget.
    ///
    /// `base_max` is the starting position limit (pre-reductions).
    /// `min_viable` is the floor — typically (min_notional * 1.15) / mark_px.
    pub fn new(base_max: f64, min_viable: f64) -> Self {
        Self {
            base_max,
            min_viable,
            reductions: Vec::new(),
        }
    }

    /// Apply a multiplicative reduction from a named source.
    ///
    /// Multiplier is clamped to [0.0, 1.0]. Multiple reductions compound
    /// but `effective()` never drops below `min_viable`.
    pub fn apply_reduction(&mut self, source: &'static str, multiplier: f64) {
        self.reductions.push((source, multiplier.clamp(0.0, 1.0)));
    }

    /// Effective position limit after all reductions, floored at min_viable.
    ///
    /// `max(base * product_of_reductions, min_viable)` — unless base_max
    /// itself is below min_viable (startup should reject this).
    pub fn effective(&self) -> f64 {
        let product: f64 = self.reductions.iter().map(|(_, m)| m).product();
        let reduced = self.base_max * product;
        reduced.max(self.min_viable).min(self.base_max)
    }

    /// Returns true if the floor is binding (reductions wanted to go lower).
    pub fn floor_is_binding(&self) -> bool {
        let product: f64 = self.reductions.iter().map(|(_, m)| m).product();
        let reduced = self.base_max * product;
        reduced < self.min_viable
    }

    /// Human-readable diagnostic of all reductions and the effective result.
    pub fn diagnostic(&self) -> String {
        let mut parts = vec![format!("base={:.4}", self.base_max)];
        for (source, mult) in &self.reductions {
            parts.push(format!("{source}={mult:.2}x"));
        }
        let product: f64 = self.reductions.iter().map(|(_, m)| m).product();
        let raw = self.base_max * product;
        let eff = self.effective();
        parts.push(format!("raw={raw:.4}"));
        if self.floor_is_binding() {
            parts.push(format!(
                "FLOORED→{eff:.4} (min_viable={:.4})",
                self.min_viable
            ));
        } else {
            parts.push(format!("effective={eff:.4}"));
        }
        parts.join(" | ")
    }
}

/// Inventory governor — enforces config.max_position as HARD ceiling.
///
/// This is the FIRST check in every quote cycle. Position safety is structural.
/// `config.max_position` is NEVER overridden by margin-derived limits.
#[derive(Debug, Clone)]
pub struct InventoryGovernor {
    /// Hard position ceiling from user config
    max_position: f64,
    /// Governor zone thresholds and addon configuration.
    config: GovernorConfig,
}

impl InventoryGovernor {
    /// Create a new inventory governor with the given max position.
    ///
    /// # Panics
    /// Panics if `max_position <= 0.0`.
    pub fn new(max_position: f64) -> Self {
        assert!(
            max_position > 0.0,
            "InventoryGovernor: max_position must be > 0.0, got {max_position}"
        );
        Self {
            max_position,
            config: GovernorConfig::default(),
        }
    }

    /// Create a new inventory governor with custom config.
    pub fn with_config(max_position: f64, config: GovernorConfig) -> Self {
        assert!(
            max_position > 0.0,
            "InventoryGovernor: max_position must be > 0.0, got {max_position}"
        );
        Self {
            max_position,
            config,
        }
    }

    /// Returns the hard position ceiling.
    pub fn max_position(&self) -> f64 {
        self.max_position
    }

    /// Classify current position into a zone and compute constraints.
    pub fn assess(&self, position: f64) -> PositionAssessment {
        let abs_pos = position.abs();
        let ratio = abs_pos / self.max_position;
        let is_long = position > 0.0;
        let remaining = (self.max_position - abs_pos).max(0.0);

        let yellow = self.config.yellow_threshold;
        let red = self.config.red_threshold;

        let (zone, max_new_exposure, addon_bps, reducing_addon) = if ratio >= 1.0 {
            // Kill zone: position at or above max.
            // Aggressively tighten reducing side to attract fills and escape overexposed state.
            (
                PositionZone::Kill,
                0.0,
                self.config.kill_addon_bps,
                -self.config.kill_reducing_addon_bps,
            )
        } else if ratio >= red {
            // Red zone: reduce-only. Tighten reducing side to attract fills.
            (
                PositionZone::Red,
                0.0,
                self.config.red_addon_bps,
                -self.config.red_reducing_addon_bps,
            )
        } else if ratio >= yellow {
            // Yellow zone: bias toward reducing, cap new exposure
            let capped_exposure = remaining * 0.5;
            // Linear ramp: 0.0 at yellow_threshold, yellow_max_addon_bps at red_threshold
            let fraction = (ratio - yellow) / (red - yellow);
            let addon = self.config.yellow_max_addon_bps * fraction;
            (PositionZone::Yellow, capped_exposure, addon, 0.0)
        } else {
            // Green zone: full two-sided
            (PositionZone::Green, remaining, 0.0, 0.0)
        };

        PositionAssessment {
            zone,
            max_new_exposure,
            position_ratio: ratio,
            position,
            is_long,
            increasing_side_addon_bps: addon_bps,
            reducing_side_addon_bps: reducing_addon,
        }
    }

    /// Check if placing an order of `order_size` would push position past max.
    ///
    /// Returns `true` if the resulting |position| would exceed max_position.
    pub fn would_exceed(&self, current_position: f64, order_size: f64, is_buy: bool) -> bool {
        let new_position = if is_buy {
            current_position + order_size
        } else {
            current_position - order_size
        };
        new_position.abs() > self.max_position
    }

    /// Returns `true` if the order reduces |position|.
    pub fn is_reducing(&self, position: f64, is_buy: bool) -> bool {
        if position > 0.0 {
            // Long position — selling reduces
            !is_buy
        } else if position < 0.0 {
            // Short position — buying reduces
            is_buy
        } else {
            // Flat — nothing reduces
            false
        }
    }

    /// Get current position zone based on position vs max.
    ///
    /// Uses 60/80/100% boundaries for Green/Yellow/Red/Kill classification.
    /// This is a lightweight zone check without the full assessment.
    pub fn current_zone(&self, position_abs: f64) -> PositionZone {
        let ratio = position_abs / self.max_position;
        if ratio >= 1.0 {
            PositionZone::Kill
        } else if ratio >= self.config.red_threshold {
            PositionZone::Red
        } else if ratio >= self.config.yellow_threshold {
            PositionZone::Yellow
        } else {
            PositionZone::Green
        }
    }

    /// Adjust max position based on regime volatility fraction.
    ///
    /// In calm markets (fraction=0.0), max stays at config value.
    /// In extreme regime (fraction=1.0), max reduces to 30% of config max.
    /// Linear interpolation between: tightening = 1.0 - 0.7 * fraction.
    pub fn zone_adjusted_max(&self, regime_volatility_fraction: f64) -> f64 {
        let fraction = regime_volatility_fraction.clamp(0.0, 1.0);
        let tightening = 1.0 - 0.7 * fraction; // [0.3, 1.0]
        self.max_position * tightening
    }

    /// Compute a single `PositionLimits` from exchange margin, regime, and signal state.
    ///
    /// This is the ONE place position limits are computed each cycle.
    /// `regime_fraction` and `signal_fraction` are clamped to [0.3, 1.0] — even in
    /// the worst conditions, we never reduce below 30% of hard_max (avoids zero-quoting).
    pub fn compute_position_limits(
        &self,
        margin_max: f64,
        regime_fraction: f64,
        signal_fraction: f64,
    ) -> PositionLimits {
        PositionLimits {
            hard_max: self.max_position,
            margin_max,
            regime_fraction: regime_fraction.clamp(0.3, 1.0),
            signal_fraction: signal_fraction.clamp(0.3, 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_green_zone() {
        let gov = InventoryGovernor::new(10.0);
        let a = gov.assess(0.0);
        assert_eq!(a.zone, PositionZone::Green);
        assert!((a.max_new_exposure - 10.0).abs() < 1e-9);
        assert!((a.position_ratio).abs() < 1e-9);
        assert!(a.increasing_side_addon_bps.abs() < 1e-9);
    }

    #[test]
    fn test_green_zone_partial() {
        let gov = InventoryGovernor::new(10.0);
        let a = gov.assess(3.0);
        assert_eq!(a.zone, PositionZone::Green);
        assert!((a.max_new_exposure - 7.0).abs() < 1e-9);
        assert!((a.position_ratio - 0.3).abs() < 1e-9);
        assert!(a.is_long);
    }

    #[test]
    fn test_yellow_zone_lower_bound() {
        let gov = InventoryGovernor::new(10.0);
        // ratio = 0.5 is at yellow_threshold (exact boundary → Yellow)
        let a = gov.assess(5.0);
        assert_eq!(a.zone, PositionZone::Yellow);
        // remaining = 5.0, capped = 2.5
        assert!((a.max_new_exposure - 2.5).abs() < 1e-9);
        // addon_bps at yellow lower bound should be 0.0
        assert!(a.increasing_side_addon_bps.abs() < 0.01);
    }

    #[test]
    fn test_yellow_zone_mid() {
        let gov = InventoryGovernor::new(10.0);
        // ratio = 0.6 is mid-Yellow
        let a = gov.assess(6.0);
        assert_eq!(a.zone, PositionZone::Yellow);
        // remaining = 4.0, capped = 2.0
        assert!((a.max_new_exposure - 2.0).abs() < 1e-9);
        // addon_bps: 10.0 × (0.6 - 0.5) / (0.8 - 0.5) = 3.33
        let expected = 10.0 * (0.6 - 0.5) / (0.8 - 0.5);
        assert!((a.increasing_side_addon_bps - expected).abs() < 0.1);
    }

    #[test]
    fn test_green_zone_below_yellow() {
        let gov = InventoryGovernor::new(10.0);
        // ratio = 0.4 is Green (below yellow_threshold 0.5)
        let a = gov.assess(4.0);
        assert_eq!(a.zone, PositionZone::Green);
        assert!((a.max_new_exposure - 6.0).abs() < 1e-9);
        assert!(a.increasing_side_addon_bps.abs() < 0.01);
    }

    #[test]
    fn test_yellow_zone_upper_bound() {
        let gov = InventoryGovernor::new(10.0);
        // ratio = 0.79 — still yellow
        let a = gov.assess(7.9);
        assert_eq!(a.zone, PositionZone::Yellow);
        // remaining = 2.1, capped = 1.05
        assert!((a.max_new_exposure - 1.05).abs() < 1e-9);
        // addon_bps: 10.0 × (0.79 - 0.5) / (0.8 - 0.5) = 10.0 × 0.9667 = 9.667
        let expected_addon = 10.0 * (0.79 - 0.5) / (0.8 - 0.5);
        assert!((a.increasing_side_addon_bps - expected_addon).abs() < 0.1);
    }

    #[test]
    fn test_red_zone() {
        let gov = InventoryGovernor::new(10.0);
        let a = gov.assess(8.5);
        assert_eq!(a.zone, PositionZone::Red);
        assert!((a.max_new_exposure).abs() < 1e-9);
        assert!((a.increasing_side_addon_bps - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_kill_zone() {
        let gov = InventoryGovernor::new(10.0);
        let a = gov.assess(10.5);
        assert_eq!(a.zone, PositionZone::Kill);
        assert!((a.max_new_exposure).abs() < 1e-9);
        assert!((a.increasing_side_addon_bps - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_kill_zone_exact_boundary() {
        let gov = InventoryGovernor::new(10.0);
        let a = gov.assess(10.0);
        assert_eq!(a.zone, PositionZone::Kill);
    }

    #[test]
    fn test_red_zone_exact_boundary() {
        let gov = InventoryGovernor::new(10.0);
        let a = gov.assess(8.0);
        assert_eq!(a.zone, PositionZone::Red);
    }

    #[test]
    fn test_negative_position_zones() {
        let gov = InventoryGovernor::new(10.0);

        // Short position, green zone
        let a = gov.assess(-3.0);
        assert_eq!(a.zone, PositionZone::Green);
        assert!(!a.is_long);
        assert!((a.max_new_exposure - 7.0).abs() < 1e-9);

        // Short position, yellow zone
        let a = gov.assess(-6.0);
        assert_eq!(a.zone, PositionZone::Yellow);
        assert!(!a.is_long);

        // Short position, red zone
        let a = gov.assess(-9.0);
        assert_eq!(a.zone, PositionZone::Red);

        // Short position, kill zone
        let a = gov.assess(-11.0);
        assert_eq!(a.zone, PositionZone::Kill);
    }

    #[test]
    fn test_would_exceed_long() {
        let gov = InventoryGovernor::new(10.0);
        // Current position 8.0, buying 3.0 -> 11.0 > 10.0
        assert!(gov.would_exceed(8.0, 3.0, true));
        // Current position 8.0, buying 1.0 -> 9.0 <= 10.0
        assert!(!gov.would_exceed(8.0, 1.0, true));
        // Current position 8.0, selling 3.0 -> 5.0 <= 10.0
        assert!(!gov.would_exceed(8.0, 3.0, false));
    }

    #[test]
    fn test_would_exceed_short() {
        let gov = InventoryGovernor::new(10.0);
        // Current position -8.0, selling 3.0 -> -11.0, abs = 11.0 > 10.0
        assert!(gov.would_exceed(-8.0, 3.0, false));
        // Current position -8.0, selling 1.0 -> -9.0, abs = 9.0 <= 10.0
        assert!(!gov.would_exceed(-8.0, 1.0, false));
        // Current position -8.0, buying 3.0 -> -5.0, abs = 5.0 <= 10.0
        assert!(!gov.would_exceed(-8.0, 3.0, true));
    }

    #[test]
    fn test_would_exceed_zero_position() {
        let gov = InventoryGovernor::new(10.0);
        // From flat, buying 11 would exceed
        assert!(gov.would_exceed(0.0, 11.0, true));
        // From flat, buying 10 would NOT exceed (10.0 == 10.0, not >)
        assert!(!gov.would_exceed(0.0, 10.0, true));
        // From flat, selling 11 would exceed
        assert!(gov.would_exceed(0.0, 11.0, false));
    }

    #[test]
    fn test_would_exceed_reducing_can_flip() {
        let gov = InventoryGovernor::new(10.0);
        // Long 8.0, selling 100 -> -92 -> abs 92 > 10 — exceeds by flipping past max
        assert!(gov.would_exceed(8.0, 100.0, false));
        // Long 8.0, selling 5.0 -> 3.0, abs = 3.0 <= 10.0 — safe reduce
        assert!(!gov.would_exceed(8.0, 5.0, false));
        // Long 8.0, selling 18.0 -> -10.0, abs = 10.0 <= 10.0 — exact boundary ok
        assert!(!gov.would_exceed(8.0, 18.0, false));
        // Short -8.0, buying 100 -> 92 > 10 — exceeds by flipping
        assert!(gov.would_exceed(-8.0, 100.0, true));
        // Short -8.0, buying 5.0 -> -3.0, abs = 3.0 <= 10.0
        assert!(!gov.would_exceed(-8.0, 5.0, true));
    }

    #[test]
    fn test_is_reducing_long() {
        let gov = InventoryGovernor::new(10.0);
        assert!(gov.is_reducing(5.0, false)); // selling reduces long
        assert!(!gov.is_reducing(5.0, true)); // buying increases long
    }

    #[test]
    fn test_is_reducing_short() {
        let gov = InventoryGovernor::new(10.0);
        assert!(gov.is_reducing(-5.0, true)); // buying reduces short
        assert!(!gov.is_reducing(-5.0, false)); // selling increases short
    }

    #[test]
    fn test_is_reducing_flat() {
        let gov = InventoryGovernor::new(10.0);
        assert!(!gov.is_reducing(0.0, true)); // flat, nothing reduces
        assert!(!gov.is_reducing(0.0, false));
    }

    #[test]
    #[should_panic(expected = "max_position must be > 0.0")]
    fn test_new_zero_panics() {
        InventoryGovernor::new(0.0);
    }

    #[test]
    #[should_panic(expected = "max_position must be > 0.0")]
    fn test_new_negative_panics() {
        InventoryGovernor::new(-1.0);
    }

    #[test]
    fn test_addon_bps_linear_ramp() {
        let gov = InventoryGovernor::new(100.0);
        // At ratio=0.5: addon should be 0.0 (Yellow lower bound)
        let a = gov.assess(50.0);
        assert!(a.increasing_side_addon_bps.abs() < 0.01);

        // At ratio=0.65: addon should be 5.0 (midpoint Yellow, 10.0 * 0.5)
        let a = gov.assess(65.0);
        assert!((a.increasing_side_addon_bps - 5.0).abs() < 0.01);

        // Just below 0.8: addon should approach 10.0
        let a = gov.assess(79.9);
        assert!((a.increasing_side_addon_bps - 10.0).abs() < 0.1);
    }

    // --- current_zone tests ---

    #[test]
    fn test_current_zone_green() {
        let gov = InventoryGovernor::new(10.0);
        // 0% — Green
        assert_eq!(gov.current_zone(0.0), PositionZone::Green);
        // 49% — still Green
        assert_eq!(gov.current_zone(4.9), PositionZone::Green);
    }

    #[test]
    fn test_current_zone_yellow() {
        let gov = InventoryGovernor::new(10.0);
        // 50% exactly — Yellow (>= 0.5)
        assert_eq!(gov.current_zone(5.0), PositionZone::Yellow);
        // 60% — Yellow
        assert_eq!(gov.current_zone(6.0), PositionZone::Yellow);
        // 79% — Yellow
        assert_eq!(gov.current_zone(7.9), PositionZone::Yellow);
    }

    #[test]
    fn test_current_zone_red() {
        let gov = InventoryGovernor::new(10.0);
        // 80% exactly — Red (>= 0.8)
        assert_eq!(gov.current_zone(8.0), PositionZone::Red);
        // 81% — Red
        assert_eq!(gov.current_zone(8.1), PositionZone::Red);
        // 99% — Red
        assert_eq!(gov.current_zone(9.9), PositionZone::Red);
    }

    #[test]
    fn test_current_zone_kill() {
        let gov = InventoryGovernor::new(10.0);
        // 100% — Kill (>= 1.0)
        assert_eq!(gov.current_zone(10.0), PositionZone::Kill);
        // 101% — Kill
        assert_eq!(gov.current_zone(10.1), PositionZone::Kill);
    }

    // --- zone_adjusted_max tests ---

    #[test]
    fn test_zone_adjusted_max_calm() {
        let gov = InventoryGovernor::new(10.0);
        // fraction=0.0: tightening=1.0, max=10.0
        let adjusted = gov.zone_adjusted_max(0.0);
        assert!((adjusted - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_zone_adjusted_max_moderate() {
        let gov = InventoryGovernor::new(10.0);
        // fraction=0.5: tightening = 1.0 - 0.35 = 0.65, max = 6.5
        let adjusted = gov.zone_adjusted_max(0.5);
        assert!((adjusted - 6.5).abs() < 1e-9);
    }

    #[test]
    fn test_zone_adjusted_max_extreme() {
        let gov = InventoryGovernor::new(10.0);
        // fraction=1.0: tightening = 1.0 - 0.7 = 0.3, max = 3.0
        let adjusted = gov.zone_adjusted_max(1.0);
        assert!((adjusted - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_zone_adjusted_max_clamps_input() {
        let gov = InventoryGovernor::new(10.0);
        // fraction > 1.0 clamped to 1.0
        let adjusted = gov.zone_adjusted_max(2.0);
        assert!((adjusted - 3.0).abs() < 1e-9);
        // fraction < 0.0 clamped to 0.0
        let adjusted = gov.zone_adjusted_max(-1.0);
        assert!((adjusted - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_position_zone_default() {
        let zone = PositionZone::default();
        assert_eq!(zone, PositionZone::Green);
    }

    // === PositionBudget tests ===

    #[test]
    fn test_budget_no_reductions() {
        let budget = PositionBudget::new(10.0, 0.5);
        assert!((budget.effective() - 10.0).abs() < 1e-9);
        assert!(!budget.floor_is_binding());
    }

    #[test]
    fn test_budget_single_reduction() {
        let mut budget = PositionBudget::new(10.0, 0.5);
        budget.apply_reduction("regime", 0.3);
        // 10.0 * 0.3 = 3.0, > min_viable 0.5
        assert!((budget.effective() - 3.0).abs() < 1e-9);
        assert!(!budget.floor_is_binding());
    }

    #[test]
    fn test_budget_compound_reductions_floored() {
        let mut budget = PositionBudget::new(10.0, 0.5);
        budget.apply_reduction("regime", 0.3);
        budget.apply_reduction("signal", 0.3);
        budget.apply_reduction("proactive", 0.3);
        // Without floor: 10.0 * 0.3 * 0.3 * 0.3 = 0.27 < 0.5
        // With floor: max(0.27, 0.5) = 0.5
        assert!((budget.effective() - 0.5).abs() < 1e-9);
        assert!(budget.floor_is_binding());
    }

    #[test]
    fn test_budget_effective_never_exceeds_base() {
        let mut budget = PositionBudget::new(1.0, 5.0);
        // min_viable > base_max: effective capped at base_max
        assert!((budget.effective() - 1.0).abs() < 1e-9);
        budget.apply_reduction("regime", 0.5);
        // 1.0 * 0.5 = 0.5, max(0.5, 5.0) = 5.0, min(5.0, 1.0) = 1.0
        assert!((budget.effective() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_budget_diagnostic_readable() {
        let mut budget = PositionBudget::new(3.24, 0.383);
        budget.apply_reduction("regime", 0.3);
        budget.apply_reduction("signal", 0.3);
        let diag = budget.diagnostic();
        assert!(diag.contains("base=3.2400"));
        assert!(diag.contains("regime=0.30x"));
        assert!(diag.contains("signal=0.30x"));
        assert!(diag.contains("FLOORED"));
    }

    #[test]
    fn test_budget_multiplier_clamped() {
        let mut budget = PositionBudget::new(10.0, 0.5);
        budget.apply_reduction("bad", 1.5); // Clamped to 1.0
        assert!((budget.effective() - 10.0).abs() < 1e-9);
        budget.apply_reduction("bad2", -0.5); // Clamped to 0.0
        assert!((budget.effective() - 0.5).abs() < 1e-9); // Floored at min_viable
    }

    #[test]
    fn test_budget_zero_base() {
        let budget = PositionBudget::new(0.0, 0.5);
        // max(0.0, 0.5) = 0.5, min(0.5, 0.0) = 0.0
        assert!((budget.effective()).abs() < 1e-9);
    }

    // === PositionLimits tests ===

    #[test]
    fn test_position_limits_effective_respects_hard_max() {
        // effective() should never exceed hard_max, regardless of margin_max
        let limits = PositionLimits::new(1.0, 100.0);
        assert!(limits.effective() <= limits.hard_max);

        let limits = PositionLimits::new(1.0, 0.5);
        assert!(limits.effective() <= limits.hard_max);
        assert!((limits.effective() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_position_limits_effective_uses_min_of_hard_and_margin() {
        let limits = PositionLimits::new(2.0, 1.5);
        assert!((limits.effective() - 1.5).abs() < 1e-9);

        let limits = PositionLimits::new(1.5, 2.0);
        assert!((limits.effective() - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_position_limits_regime_fraction_reduces() {
        let limits = PositionLimits {
            hard_max: 10.0,
            margin_max: 10.0,
            regime_fraction: 0.5,
            signal_fraction: 1.0,
        };
        assert!((limits.effective() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_position_limits_signal_fraction_reduces() {
        let limits = PositionLimits {
            hard_max: 10.0,
            margin_max: 10.0,
            regime_fraction: 1.0,
            signal_fraction: 0.3,
        };
        assert!((limits.effective() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_position_limits_compound_reductions() {
        // Both regime and signal reduce independently
        let limits = PositionLimits {
            hard_max: 10.0,
            margin_max: 10.0,
            regime_fraction: 0.5,
            signal_fraction: 0.5,
        };
        assert!((limits.effective() - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_position_limits_effective_never_negative() {
        let limits = PositionLimits {
            hard_max: -1.0, // pathological
            margin_max: 10.0,
            regime_fraction: 1.0,
            signal_fraction: 1.0,
        };
        assert!(limits.effective() >= 0.0);
    }

    #[test]
    fn test_position_limits_is_reduce_only() {
        let limits = PositionLimits::new(1.0, 1.0);
        assert!(!limits.is_reduce_only(0.5));
        assert!(!limits.is_reduce_only(-0.5));
        assert!(limits.is_reduce_only(1.0));
        assert!(limits.is_reduce_only(-1.0));
        assert!(limits.is_reduce_only(1.5));
        assert!(limits.is_reduce_only(-1.5));
    }

    #[test]
    fn test_position_limits_is_reduce_only_with_reductions() {
        let limits = PositionLimits {
            hard_max: 10.0,
            margin_max: 10.0,
            regime_fraction: 0.5,
            signal_fraction: 1.0,
        };
        // effective = 5.0
        assert!(!limits.is_reduce_only(4.9));
        assert!(limits.is_reduce_only(5.0));
        assert!(limits.is_reduce_only(5.1));
    }

    #[test]
    fn test_compute_position_limits_clamps_fractions() {
        let gov = InventoryGovernor::new(10.0);

        // Below 0.3 gets clamped to 0.3
        let limits = gov.compute_position_limits(10.0, 0.0, 0.0);
        assert!((limits.regime_fraction - 0.3).abs() < 1e-9);
        assert!((limits.signal_fraction - 0.3).abs() < 1e-9);
        assert!((limits.effective() - 10.0 * 0.3 * 0.3).abs() < 1e-9);

        // Above 1.0 gets clamped to 1.0
        let limits = gov.compute_position_limits(10.0, 2.0, 2.0);
        assert!((limits.regime_fraction - 1.0).abs() < 1e-9);
        assert!((limits.signal_fraction - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_position_limits_uses_config_max() {
        let gov = InventoryGovernor::new(3.0);
        let limits = gov.compute_position_limits(100.0, 1.0, 1.0);
        // hard_max should be 3.0 from config, margin_max is 100.0
        // effective = min(3.0, 100.0) * 1.0 * 1.0 = 3.0
        assert!((limits.hard_max - 3.0).abs() < 1e-9);
        assert!((limits.effective() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_kill_zone_reducing_addon_25bps() {
        let gov = InventoryGovernor::new(10.0);
        let a = gov.assess(10.5); // Kill zone
        assert_eq!(a.zone, PositionZone::Kill);
        assert!(
            (a.reducing_side_addon_bps - (-25.0)).abs() < 1e-9,
            "Kill zone reducing addon should be -25.0: {}",
            a.reducing_side_addon_bps
        );
    }

    #[test]
    fn test_red_zone_reducing_addon_15bps() {
        let gov = InventoryGovernor::new(10.0);
        let a = gov.assess(8.5); // Red zone
        assert_eq!(a.zone, PositionZone::Red);
        assert!(
            (a.reducing_side_addon_bps - (-15.0)).abs() < 1e-9,
            "Red zone reducing addon should be -15.0: {}",
            a.reducing_side_addon_bps
        );
    }

    #[test]
    fn test_position_limits_effective_always_le_hard_max() {
        // Exhaustive property: effective() <= hard_max for any inputs
        let test_cases = [
            (1.0, 100.0, 1.0, 1.0),
            (1.0, 0.5, 0.3, 0.3),
            (10.0, 10.0, 0.5, 0.5),
            (5.0, 3.0, 1.0, 0.3),
            (0.1, 1000.0, 1.0, 1.0),
        ];
        for (hard, margin, regime, signal) in test_cases {
            let limits = PositionLimits {
                hard_max: hard,
                margin_max: margin,
                regime_fraction: regime,
                signal_fraction: signal,
            };
            assert!(
                limits.effective() <= limits.hard_max + 1e-12,
                "effective {} > hard_max {} for ({hard}, {margin}, {regime}, {signal})",
                limits.effective(),
                limits.hard_max
            );
        }
    }
}
