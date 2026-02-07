//! Pre-computed asset runtime configuration for zero-overhead hot paths.

use std::sync::Arc;
use std::time::Instant;

use crate::meta::AssetMeta;

// =============================================================================
// Session Position Ramp
// =============================================================================

/// Ramp curve for position capacity growth over time.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum RampCurve {
    /// Linear: fraction = t / T
    Linear,
    /// Square root: fraction = sqrt(t / T) - reaches 70% at half time
    #[default]
    Sqrt,
    /// Logarithmic: fraction = ln(1 + t) / ln(1 + T) - very gradual
    Log,
}

impl RampCurve {
    /// Calculate the fraction at time t for total duration T.
    pub fn fraction_at(&self, t_secs: f64, total_secs: f64) -> f64 {
        if total_secs <= 0.0 || t_secs >= total_secs {
            return 1.0;
        }
        if t_secs <= 0.0 {
            return 0.0;
        }

        let ratio = t_secs / total_secs;
        match self {
            RampCurve::Linear => ratio,
            RampCurve::Sqrt => ratio.sqrt(),
            RampCurve::Log => {
                // ln(1 + t) / ln(1 + T)
                (1.0 + t_secs).ln() / (1.0 + total_secs).ln()
            }
        }
    }
}

/// Time-based position ramp that limits max position based on session time.
///
/// Prevents the system from immediately taking full position at session start.
/// Instead, position capacity grows gradually, allowing time to:
/// - Calibrate model parameters
/// - Observe market conditions
/// - Build confidence before full exposure
///
/// # Example
/// With 30-minute ramp using sqrt curve:
/// | Time | Ramp Fraction | If max=10 BTC |
/// |------|---------------|---------------|
/// | 0 min | 10% | 1.0 BTC |
/// | 2 min | 36% | 3.6 BTC |
/// | 5 min | 51% | 5.1 BTC |
/// | 15 min | 81% | 8.1 BTC |
/// | 30 min | 100% | 10.0 BTC |
#[derive(Debug, Clone)]
pub struct SessionPositionRamp {
    /// Time to reach full position capacity (seconds).
    /// Default: 1800 (30 minutes)
    pub ramp_duration_secs: f64,

    /// Starting fraction of max position (0.0 - 1.0).
    /// Default: 0.1 (10%)
    pub initial_fraction: f64,

    /// Ramp curve shape.
    /// Default: Sqrt (fast start, slow finish)
    pub ramp_curve: RampCurve,

    /// Session start time (set when session begins).
    session_start: Option<Instant>,
}

impl Default for SessionPositionRamp {
    fn default() -> Self {
        Self {
            ramp_duration_secs: 1800.0, // 30 minutes
            initial_fraction: 0.1,      // Start at 10%
            ramp_curve: RampCurve::Sqrt,
            session_start: None,
        }
    }
}

impl SessionPositionRamp {
    /// Create a new position ramp with custom parameters.
    pub fn new(ramp_duration_secs: f64, initial_fraction: f64, ramp_curve: RampCurve) -> Self {
        Self {
            ramp_duration_secs: ramp_duration_secs.max(0.0),
            initial_fraction: initial_fraction.clamp(0.0, 1.0),
            ramp_curve,
            session_start: None,
        }
    }

    /// Start the session (call once at session begin).
    pub fn start_session(&mut self) {
        self.session_start = Some(Instant::now());
    }

    /// Check if session has started.
    pub fn is_session_started(&self) -> bool {
        self.session_start.is_some()
    }

    /// Get the current ramp fraction [initial_fraction, 1.0].
    ///
    /// Returns 1.0 if session hasn't started (fail-open behavior).
    pub fn current_fraction(&self) -> f64 {
        let Some(start) = self.session_start else {
            return 1.0; // Fail-open: full capacity if not initialized
        };

        let elapsed_secs = start.elapsed().as_secs_f64();
        self.fraction_at(elapsed_secs)
    }

    /// Get ramp fraction at a specific elapsed time.
    pub fn fraction_at(&self, elapsed_secs: f64) -> f64 {
        if self.ramp_duration_secs <= 0.0 {
            return 1.0;
        }

        // Calculate base fraction from curve
        let curve_fraction = self
            .ramp_curve
            .fraction_at(elapsed_secs, self.ramp_duration_secs);

        // Scale from initial_fraction to 1.0
        // fraction = initial + (1 - initial) * curve_fraction
        self.initial_fraction + (1.0 - self.initial_fraction) * curve_fraction
    }

    /// Get elapsed session time in seconds.
    pub fn elapsed_secs(&self) -> f64 {
        self.session_start
            .map(|s| s.elapsed().as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Check if ramp is complete (at full capacity).
    pub fn is_complete(&self) -> bool {
        self.current_fraction() >= 0.999
    }

    /// Apply ramp to a max position value.
    #[inline]
    pub fn apply(&self, max_position: f64) -> f64 {
        max_position * self.current_fraction()
    }
}

/// Pre-computed asset configuration for zero-overhead hot paths.
///
/// All HIP-3 detection is done ONCE at startup. Quote cycle uses only
/// primitive fields (bool, f64) with no Option unwraps or string comparisons.
///
/// # Design Principle: ZERO HOT-PATH OVERHEAD
///
/// This struct resolves all conditional logic at construction time:
/// - `is_cross`: Pre-computed from margin mode detection
/// - `oi_cap_usd`: Pre-resolved (f64::MAX if no cap)
/// - `sz_multiplier`: Pre-computed 10^sz_decimals
///
/// # Fee Handling
///
/// Fees are NOT pre-computed. HIP-3 builder fees vary per deployer
/// (0-300% share) and are included in the `fee` field of each fill.
/// The `builderFee` field in fills contains the builder's portion.
#[derive(Debug, Clone)]
pub struct AssetRuntimeConfig {
    // === Pre-computed margin fields (NO OPTIONS) ===
    /// Whether to use cross margin (pre-computed from AssetMeta).
    /// HOT PATH: Used directly in margin calculations and leverage API calls.
    pub is_cross: bool,

    /// Open interest cap in USD (f64::MAX if no cap).
    /// HOT PATH: Pre-flight check before order placement.
    pub oi_cap_usd: f64,

    /// Pre-computed sz_decimals as f64 power for truncation.
    /// HOT PATH: Avoids powi() call in size formatting.
    pub sz_multiplier: f64,

    /// Pre-computed price decimals multiplier.
    /// For perps: 10^5 (5 significant figures).
    pub price_multiplier: f64,

    // === Cold path metadata (startup only) ===
    /// Asset name (Arc for cheap cloning).
    pub asset: Arc<str>,

    /// Maximum leverage (from API).
    pub max_leverage: f64,

    /// Whether this is a HIP-3 builder-deployed asset.
    pub is_hip3: bool,

    /// Deployer address (for logging/display only).
    pub deployer: Option<Arc<str>>,
}

impl AssetRuntimeConfig {
    /// Build from API metadata - called ONCE at startup.
    ///
    /// This resolves all HIP-3 detection and margin mode logic upfront
    /// so the hot path has zero conditional overhead.
    pub fn from_asset_meta(meta: &AssetMeta) -> Self {
        let is_hip3 = meta.is_hip3();

        Self {
            // Pre-compute for hot path
            is_cross: !is_hip3,
            oi_cap_usd: meta.oi_cap_usd.unwrap_or(f64::MAX),
            sz_multiplier: 10_f64.powi(meta.sz_decimals as i32),
            price_multiplier: 10_f64.powi(5), // 5 sig figs for perps

            // Cold path storage
            asset: Arc::from(meta.name.as_str()),
            max_leverage: meta.max_leverage as f64,
            is_hip3,
            deployer: meta.deployer.as_ref().map(|d| Arc::from(d.as_str())),
        }
    }

    /// Fast size truncation (hot path).
    ///
    /// Uses pre-computed multiplier to avoid powi() in hot path.
    #[inline(always)]
    pub fn truncate_size(&self, size: f64) -> f64 {
        (size * self.sz_multiplier).trunc() / self.sz_multiplier
    }

    /// Check OI cap (hot path) - returns max additional notional allowed.
    ///
    /// Returns 0.0 if current_oi >= cap, otherwise returns remaining capacity.
    /// For unlimited assets (oi_cap_usd == f64::MAX), returns f64::MAX.
    #[inline(always)]
    pub fn remaining_oi_capacity(&self, current_oi: f64) -> f64 {
        (self.oi_cap_usd - current_oi).max(0.0)
    }

    /// Format OI cap for display (cold path).
    pub fn oi_cap_display(&self) -> String {
        if self.oi_cap_usd == f64::MAX {
            "unlimited".to_string()
        } else {
            format!("${:.0}", self.oi_cap_usd)
        }
    }
}

impl Default for AssetRuntimeConfig {
    /// Default config for testing - represents a standard validator perp.
    fn default() -> Self {
        Self {
            is_cross: true,
            oi_cap_usd: f64::MAX,
            sz_multiplier: 100_000.0, // 5 decimals
            price_multiplier: 100_000.0,
            asset: Arc::from("BTC"),
            max_leverage: 50.0,
            is_hip3: false,
            deployer: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =============================================================================
    // SessionPositionRamp Tests
    // =============================================================================

    #[test]
    fn test_ramp_curve_linear() {
        let curve = RampCurve::Linear;
        assert!((curve.fraction_at(0.0, 100.0) - 0.0).abs() < 0.01);
        assert!((curve.fraction_at(50.0, 100.0) - 0.5).abs() < 0.01);
        assert!((curve.fraction_at(100.0, 100.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ramp_curve_sqrt() {
        let curve = RampCurve::Sqrt;
        assert!((curve.fraction_at(0.0, 100.0) - 0.0).abs() < 0.01);
        // sqrt(0.5) ≈ 0.707
        assert!((curve.fraction_at(50.0, 100.0) - 0.707).abs() < 0.01);
        assert!((curve.fraction_at(100.0, 100.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ramp_curve_log() {
        let curve = RampCurve::Log;
        assert!((curve.fraction_at(0.0, 100.0) - 0.0).abs() < 0.01);
        // Log curve: ln(1+t)/ln(1+T) - actually faster than linear early on
        // At t=50, T=100: ln(51)/ln(101) ≈ 0.85
        assert!(curve.fraction_at(50.0, 100.0) > 0.8);
        assert!((curve.fraction_at(100.0, 100.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_session_position_ramp_default() {
        let ramp = SessionPositionRamp::default();
        assert_eq!(ramp.ramp_duration_secs, 1800.0);
        assert_eq!(ramp.initial_fraction, 0.1);
        assert!(!ramp.is_session_started());
        // Without session start, returns 1.0 (fail-open)
        assert_eq!(ramp.current_fraction(), 1.0);
    }

    #[test]
    fn test_session_position_ramp_fraction_at() {
        let ramp = SessionPositionRamp {
            ramp_duration_secs: 1800.0, // 30 minutes
            initial_fraction: 0.1,
            ramp_curve: RampCurve::Sqrt,
            session_start: None,
        };

        // At t=0: should be initial_fraction (10%)
        assert!((ramp.fraction_at(0.0) - 0.1).abs() < 0.01);

        // At t=2min (120s): sqrt(120/1800) ≈ 0.258, scaled = 0.1 + 0.9*0.258 ≈ 0.33
        let frac_2min = ramp.fraction_at(120.0);
        assert!(frac_2min > 0.3 && frac_2min < 0.4, "At 2min: {}", frac_2min);

        // At t=30min: should be ~1.0
        assert!((ramp.fraction_at(1800.0) - 1.0).abs() < 0.01);

        // Beyond ramp duration: should be 1.0
        assert!((ramp.fraction_at(3600.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_session_position_ramp_apply() {
        let mut ramp = SessionPositionRamp::new(1800.0, 0.1, RampCurve::Linear);

        // Before session start, apply returns full amount (fail-open)
        assert_eq!(ramp.apply(10.0), 10.0);

        // After session start
        ramp.start_session();

        // Immediately after start: ~10% of 10 = 1.0
        // (with some tolerance for the tiny elapsed time)
        let applied = ramp.apply(10.0);
        assert!(applied >= 0.99 && applied <= 1.5, "Immediate: {}", applied);
    }

    // =============================================================================
    // AssetRuntimeConfig Tests
    // =============================================================================

    fn make_hip3_asset_meta() -> AssetMeta {
        AssetMeta {
            name: "HIP3COIN".to_string(),
            sz_decimals: 2,
            max_leverage: 10,
            only_isolated: Some(true),
            margin_mode: Some("noCross".to_string()),
            is_delisted: None,
            deployer: Some("0xbuilder".to_string()),
            dex_id: Some(5),
            oi_cap_usd: Some(5_000_000.0),
            is_builder_deployed: Some(true),
        }
    }

    fn make_validator_asset_meta() -> AssetMeta {
        AssetMeta {
            name: "BTC".to_string(),
            sz_decimals: 5,
            max_leverage: 50,
            only_isolated: None,
            margin_mode: None,
            is_delisted: None,
            deployer: None,
            dex_id: None,
            oi_cap_usd: None,
            is_builder_deployed: None,
        }
    }

    #[test]
    fn test_runtime_config_from_hip3_asset() {
        let meta = make_hip3_asset_meta();
        let config = AssetRuntimeConfig::from_asset_meta(&meta);

        assert!(config.is_hip3);
        assert!(!config.is_cross); // HIP-3 = isolated only
        assert_eq!(config.oi_cap_usd, 5_000_000.0);
        assert_eq!(config.max_leverage, 10.0);
        assert!(config.deployer.is_some());
        assert_eq!(config.deployer.as_deref(), Some("0xbuilder"));
    }

    #[test]
    fn test_runtime_config_from_validator_perp() {
        let meta = make_validator_asset_meta();
        let config = AssetRuntimeConfig::from_asset_meta(&meta);

        assert!(!config.is_hip3);
        assert!(config.is_cross); // Validator perps support cross margin
        assert_eq!(config.oi_cap_usd, f64::MAX); // No OI cap
        assert_eq!(config.max_leverage, 50.0);
        assert!(config.deployer.is_none());
    }

    #[test]
    fn test_oi_cap_remaining_capacity() {
        let meta = make_hip3_asset_meta();
        let config = AssetRuntimeConfig::from_asset_meta(&meta);

        // With 1M position, remaining = 5M - 1M = 4M
        let remaining = config.remaining_oi_capacity(1_000_000.0);
        assert!((remaining - 4_000_000.0).abs() < 0.01);

        // At cap, remaining = 0
        let remaining = config.remaining_oi_capacity(5_000_000.0);
        assert_eq!(remaining, 0.0);

        // Over cap, remaining = 0 (clamped)
        let remaining = config.remaining_oi_capacity(6_000_000.0);
        assert_eq!(remaining, 0.0);
    }

    #[test]
    fn test_oi_cap_no_limit_validator_perp() {
        let meta = make_validator_asset_meta();
        let config = AssetRuntimeConfig::from_asset_meta(&meta);

        // With no cap (f64::MAX), remaining is effectively infinite
        let remaining = config.remaining_oi_capacity(1_000_000_000.0);
        assert!(remaining > 1e15); // Still huge
    }

    #[test]
    fn test_size_truncation() {
        let meta = make_hip3_asset_meta(); // sz_decimals = 2
        let config = AssetRuntimeConfig::from_asset_meta(&meta);

        // 1.234 should truncate to 1.23
        let truncated = config.truncate_size(1.234);
        assert!((truncated - 1.23).abs() < 1e-10);

        // 0.999 should truncate to 0.99
        let truncated = config.truncate_size(0.999);
        assert!((truncated - 0.99).abs() < 1e-10);
    }
}
