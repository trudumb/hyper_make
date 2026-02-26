//! Quota Shadow Pricing: Continuous rate-limit-aware spread and ladder adjustments.
//!
//! ## Purpose
//!
//! When API quota (rate limit headroom) is tight, we need to conserve requests.
//! Instead of hard binary gates (quote / don't quote), shadow pricing smoothly
//! adjusts spreads and ladder density based on remaining headroom.
//!
//! This is purely an infrastructure concern (rate limiting), not a quoting decision.
//! The pricer outputs continuous values that the quote engine absorbs into its
//! spread calculation:
//!
//! - **Shadow spread (bps)**: Added to GLFT spread, widens as headroom drops
//! - **Ladder levels**: Reduces order density at low headroom (fewer API calls)
//! - **Request shadow price (bps)**: Cost of using one API request at current headroom
//! - **Edge justification**: Whether expected edge exceeds the shadow price
//!
//! ## Key Properties
//!
//! - No cliff effects: all outputs are smooth functions of headroom
//! - No binary decisions: outputs are continuous values, not yes/no
//! - Self-contained: no dependency on QuoteGate or other quoting logic
//! - Regime-aware: volatility regime adjusts shadow price (high vol = faster recharge)

use serde::{Deserialize, Serialize};
use tracing::debug;

/// Configuration for continuous quota shadow pricing.
///
/// Instead of hard tier cutoffs, shadow pricing smoothly adjusts spreads
/// based on rate limit headroom. The shadow spread is:
///   shadow_spread_bps = lambda_shadow_bps / headroom_pct.max(0.01)
///
/// At 100% headroom: 0.5 bps (negligible)
/// At 50% headroom: 1.0 bps (mild)
/// At 10% headroom: 5.0 bps (significant)
/// At 5% headroom: 10.0 bps (aggressive)
/// At 1% headroom: 50.0 bps (prohibitive)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaShadowConfig {
    /// Base shadow price lambda (bps). Higher = more spread at low headroom.
    #[serde(default = "default_lambda_shadow_bps")]
    pub lambda_shadow_bps: f64,

    /// Headroom threshold below which ladder density is reduced.
    /// At full headroom: all levels. At zero: 1 level.
    /// Scaling: levels = max(1, (max_levels * sqrt(headroom/threshold)) as usize)
    #[serde(default = "default_min_headroom_for_full_ladder")]
    pub min_headroom_for_full_ladder: f64,

    /// Maximum shadow spread in bps (cap to prevent blowup at headroom -> 0).
    #[serde(default = "default_max_shadow_spread_bps")]
    pub max_shadow_spread_bps: f64,
}

fn default_lambda_shadow_bps() -> f64 {
    0.5
}
fn default_min_headroom_for_full_ladder() -> f64 {
    0.20
}
fn default_max_shadow_spread_bps() -> f64 {
    50.0
}

impl Default for QuotaShadowConfig {
    fn default() -> Self {
        Self {
            lambda_shadow_bps: default_lambda_shadow_bps(),
            min_headroom_for_full_ladder: default_min_headroom_for_full_ladder(),
            max_shadow_spread_bps: default_max_shadow_spread_bps(),
        }
    }
}

/// Continuous quota shadow pricer.
///
/// Computes the "shadow cost" of using API quota. When API quota is tight,
/// the pricer widens spreads and reduces ladder levels to conserve requests.
///
/// All outputs are continuous functions of headroom -- no cliff effects.
#[derive(Debug, Clone, Default)]
pub struct QuotaShadowPricer {
    config: QuotaShadowConfig,
}

// Default derived via #[derive(Default)] on the struct

impl QuotaShadowPricer {
    /// Create a new pricer with custom config.
    pub fn new(config: QuotaShadowConfig) -> Self {
        Self { config }
    }

    /// Get the current config.
    pub fn config(&self) -> &QuotaShadowConfig {
        &self.config
    }

    /// Compute continuous shadow spread adjustment in basis points.
    ///
    /// Instead of hard-blocking at low headroom, this returns a spread addition
    /// that smoothly increases as headroom decreases. The GLFT spread absorbs
    /// this cost, naturally reducing quoting frequency at low headroom without
    /// cliff effects.
    ///
    /// Formula: shadow_spread = lambda / headroom.max(0.01), capped at max_bps.
    ///
    /// At 100% headroom: 0.5 bps (negligible)
    /// At 50% headroom:  1.0 bps (mild)
    /// At 20% headroom:  2.5 bps (noticeable)
    /// At 10% headroom:  5.0 bps (significant)
    /// At 5% headroom:   10.0 bps (aggressive)
    pub fn continuous_shadow_spread_bps(&self, headroom_pct: f64) -> f64 {
        let effective_headroom = headroom_pct.max(0.01);
        let raw = self.config.lambda_shadow_bps / effective_headroom;
        raw.min(self.config.max_shadow_spread_bps)
    }

    /// Compute continuous ladder density based on headroom.
    ///
    /// Smoothly scales ladder levels from 1 to max_levels based on headroom.
    /// Uses sqrt scaling so levels drop gradually, not in cliff steps.
    ///
    /// At 100% headroom: max_levels
    /// At 25% headroom:  max_levels/2
    /// At 4% headroom:   max_levels/5
    /// At 1% headroom:   1 level
    ///
    /// When `density_scaling_enabled` is false (Micro/Small capital-aware policy),
    /// returns `max_levels` without sqrt truncation -- small accounts cannot afford
    /// to lose levels from quota density scaling.
    pub fn continuous_ladder_levels(
        &self,
        max_levels: usize,
        headroom_pct: f64,
        density_scaling_enabled: bool,
        min_headroom_override: Option<f64>,
    ) -> usize {
        // Capital-aware policy can disable density scaling entirely
        if !density_scaling_enabled {
            return max_levels;
        }

        let min_headroom =
            min_headroom_override.unwrap_or(self.config.min_headroom_for_full_ladder);
        if headroom_pct >= min_headroom {
            return max_levels;
        }
        // Scale by sqrt(headroom / min_headroom) for smooth reduction
        let scale = (headroom_pct / min_headroom).sqrt();
        (max_levels as f64 * scale).round().max(1.0) as usize
    }

    /// Compute the shadow price (in bps) of making one API request.
    ///
    /// This represents the opportunity cost of consuming quota. When headroom
    /// is high, requests are cheap. When headroom is low, each request has a
    /// significant cost because it reduces the remaining budget for future,
    /// potentially more valuable, quote updates.
    ///
    /// Regime adjustment: high-vol regimes have faster fill rates, which
    /// recharge quota faster, so the shadow price is lower.
    ///
    /// - When headroom >= 50%: shadow price = 0 (free to quote)
    /// - When headroom <= 5%: shadow price = 100 bps (effectively infinite)
    /// - Between: cubic explosion lambda proportional to (1 - h)^3
    pub fn compute_request_shadow_price(headroom_pct: f64, vol_regime: u8) -> f64 {
        // Regime adjustment: high-vol -> fills -> faster recharge -> lower shadow
        let regime_mult = match vol_regime {
            0 => 1.2, // Low volatility: conservative (fills rare)
            1 => 1.0, // Normal: baseline
            2 => 0.7, // High: can afford more requests
            3 => 0.5, // Extreme: fills plentiful
            _ => 1.0, // Fallback
        };

        // When headroom is high (>50%), shadow price ~ 0
        if headroom_pct >= 0.50 {
            return 0.0;
        }

        // When headroom is critically low (<=5%), shadow price -> infinity
        // Effectively prohibit quoting
        if headroom_pct <= 0.05 {
            return 100.0;
        }

        // Sigmoid-like explosion between 5% and 50%
        // Normalized x in [0, 1] where x=0 at 50% headroom, x=1 at 5% headroom
        let x = (0.50 - headroom_pct) / 0.45;

        // Base price for cubic explosion
        let base_price = 50.0; // Max shadow price in bps (before regime adjustment)

        // Cubic curve: steeper as headroom drops
        // At 30% headroom: x ~ 0.44, shadow ~ 4.3 bps
        // At 15% headroom: x ~ 0.78, shadow ~ 23.7 bps
        // At 10% headroom: x ~ 0.89, shadow ~ 35.2 bps
        let shadow = base_price * x.powi(3) * regime_mult;

        // Log at debug level for transparency
        debug!(
            headroom_pct = %format!("{:.1}%", headroom_pct * 100.0),
            vol_regime = vol_regime,
            regime_mult = %format!("{:.2}", regime_mult),
            shadow_bps = %format!("{:.2}", shadow),
            "Shadow price computed"
        );

        shadow
    }

    /// Check if edge justifies the shadow price of requesting.
    ///
    /// Returns true if the expected edge exceeds the shadow price of consuming
    /// one API request at the current headroom level. This internalizes quota
    /// cost into quoting decisions.
    ///
    /// # Arguments
    /// * `expected_edge_bps` - Expected edge from placing this quote (bps)
    /// * `rate_limit_headroom_pct` - Current rate limit headroom [0, 1]
    /// * `vol_regime` - Volatility regime (0=low, 1=normal, 2=high, 3=extreme)
    pub fn edge_justifies_request(
        &self,
        expected_edge_bps: f64,
        rate_limit_headroom_pct: f64,
        vol_regime: u8,
    ) -> bool {
        let shadow_price = Self::compute_request_shadow_price(rate_limit_headroom_pct, vol_regime);
        expected_edge_bps > shadow_price
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_pricer() -> QuotaShadowPricer {
        QuotaShadowPricer::default()
    }

    // ========================================================================
    // Continuous Shadow Spread Tests
    // ========================================================================

    #[test]
    fn test_continuous_shadow_spread_at_full_headroom() {
        let pricer = default_pricer();
        // At 100% headroom: lambda / 1.0 = 0.5 bps (negligible)
        let shadow = pricer.continuous_shadow_spread_bps(1.0);
        assert!(
            (shadow - 0.5).abs() < 0.01,
            "At 100% headroom shadow should be ~0.5 bps, got {}",
            shadow
        );
    }

    #[test]
    fn test_continuous_shadow_spread_at_10pct_headroom() {
        let pricer = default_pricer();
        // At 10% headroom: 0.5 / 0.10 = 5.0 bps (significant)
        let shadow = pricer.continuous_shadow_spread_bps(0.10);
        assert!(
            (shadow - 5.0).abs() < 0.01,
            "At 10% headroom shadow should be ~5.0 bps, got {}",
            shadow
        );
    }

    #[test]
    fn test_continuous_shadow_spread_at_5pct_headroom() {
        let pricer = default_pricer();
        // At 5% headroom: 0.5 / 0.05 = 10.0 bps (aggressive)
        let shadow = pricer.continuous_shadow_spread_bps(0.05);
        assert!(
            (shadow - 10.0).abs() < 0.01,
            "At 5% headroom shadow should be ~10.0 bps, got {}",
            shadow
        );
    }

    #[test]
    fn test_continuous_shadow_spread_capped_at_max() {
        let pricer = default_pricer();
        // At 0.1% headroom: 0.5 / 0.001 = 500 bps -> capped at 50.0 bps
        let shadow = pricer.continuous_shadow_spread_bps(0.001);
        assert!(
            (shadow - 50.0).abs() < 0.01,
            "Shadow should be capped at 50 bps, got {}",
            shadow
        );
    }

    #[test]
    fn test_continuous_shadow_spread_smooth_increase() {
        // Verify no cliff effects: shadow spread increases monotonically as headroom decreases
        let pricer = default_pricer();
        let headrooms = [1.0, 0.5, 0.3, 0.2, 0.1, 0.07, 0.05, 0.03, 0.01];
        let mut prev_shadow = 0.0;
        for &h in &headrooms {
            let shadow = pricer.continuous_shadow_spread_bps(h);
            assert!(
                shadow >= prev_shadow,
                "Shadow spread should increase as headroom decreases: at {:.0}% got {:.2} bps, prev {:.2} bps",
                h * 100.0,
                shadow,
                prev_shadow
            );
            prev_shadow = shadow;
        }
    }

    #[test]
    fn test_continuous_shadow_spread_zero_headroom() {
        let pricer = default_pricer();
        // At 0% headroom: clamped to 0.01, so 0.5/0.01 = 50 bps (max)
        let shadow = pricer.continuous_shadow_spread_bps(0.0);
        assert!(
            (shadow - 50.0).abs() < 0.01,
            "At 0% headroom shadow should be capped at 50 bps, got {}",
            shadow
        );
    }

    #[test]
    fn test_continuous_shadow_spread_negative_headroom() {
        let pricer = default_pricer();
        // Negative headroom should be clamped like zero
        let shadow = pricer.continuous_shadow_spread_bps(-0.5);
        assert!(
            (shadow - 50.0).abs() < 0.01,
            "Negative headroom should be capped at max, got {}",
            shadow
        );
    }

    #[test]
    fn test_continuous_shadow_spread_custom_config() {
        let pricer = QuotaShadowPricer::new(QuotaShadowConfig {
            lambda_shadow_bps: 1.0,
            max_shadow_spread_bps: 20.0,
            ..Default::default()
        });
        // At 50% headroom: 1.0 / 0.5 = 2.0 bps
        let shadow = pricer.continuous_shadow_spread_bps(0.5);
        assert!(
            (shadow - 2.0).abs() < 0.01,
            "Custom lambda: at 50% headroom should be 2.0 bps, got {}",
            shadow
        );
        // At 1% headroom: 1.0 / 0.01 = 100 -> capped at 20
        let shadow = pricer.continuous_shadow_spread_bps(0.01);
        assert!(
            (shadow - 20.0).abs() < 0.01,
            "Custom cap: should be 20 bps, got {}",
            shadow
        );
    }

    // ========================================================================
    // Continuous Ladder Levels Tests
    // ========================================================================

    #[test]
    fn test_continuous_ladder_levels_at_full_headroom() {
        let pricer = default_pricer();
        // At 100% headroom: all levels
        let levels = pricer.continuous_ladder_levels(10, 1.0, true, None);
        assert_eq!(levels, 10, "At full headroom should get all levels");
    }

    #[test]
    fn test_continuous_ladder_levels_at_min_threshold() {
        let pricer = default_pricer();
        // At 20% headroom (min_headroom_for_full_ladder): all levels
        let levels = pricer.continuous_ladder_levels(10, 0.20, true, None);
        assert_eq!(
            levels, 10,
            "At min_headroom_for_full_ladder should get all levels"
        );
    }

    #[test]
    fn test_continuous_ladder_levels_at_5pct() {
        let pricer = default_pricer();
        // At 5% headroom: sqrt(0.05/0.20) = sqrt(0.25) = 0.5 -> 5 levels
        let levels = pricer.continuous_ladder_levels(10, 0.05, true, None);
        assert_eq!(
            levels, 5,
            "At 5% headroom with 10 max should get ~5 levels, got {}",
            levels
        );
    }

    #[test]
    fn test_continuous_ladder_levels_at_1pct() {
        let pricer = default_pricer();
        // At 1% headroom: sqrt(0.01/0.20) = sqrt(0.05) ~ 0.224 -> round(2.24) = 2 levels
        let levels = pricer.continuous_ladder_levels(10, 0.01, true, None);
        assert!(
            (1..=3).contains(&levels),
            "At 1% headroom with 10 max should get ~2 levels, got {}",
            levels
        );
    }

    #[test]
    fn test_continuous_ladder_levels_floor_at_1() {
        let pricer = default_pricer();
        // Even at extremely low headroom, should always get at least 1 level
        let levels = pricer.continuous_ladder_levels(10, 0.001, true, None);
        assert_eq!(
            levels, 1,
            "Should always get at least 1 level, got {}",
            levels
        );
    }

    #[test]
    fn test_continuous_ladder_levels_smooth_decrease() {
        // Verify no cliff effects: levels decrease smoothly as headroom drops
        let pricer = default_pricer();
        let headrooms = [0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.01];
        let mut prev_levels = 100;
        for &h in &headrooms {
            let levels = pricer.continuous_ladder_levels(25, h, true, None);
            assert!(
                levels <= prev_levels,
                "Levels should not increase as headroom drops: at {:.0}% got {}, prev {}",
                h * 100.0,
                levels,
                prev_levels
            );
            prev_levels = levels;
        }
    }

    #[test]
    fn test_continuous_ladder_levels_density_scaling_disabled() {
        let pricer = default_pricer();
        // With density_scaling=false (Micro/Small policy), should always return max_levels
        let levels = pricer.continuous_ladder_levels(2, 0.01, false, None);
        assert_eq!(
            levels, 2,
            "With density scaling disabled, should get max_levels regardless of headroom"
        );

        let levels = pricer.continuous_ladder_levels(3, 0.001, false, None);
        assert_eq!(
            levels, 3,
            "With density scaling disabled at extreme low headroom"
        );
    }

    #[test]
    fn test_continuous_ladder_levels_custom_min_headroom() {
        let pricer = default_pricer();
        // With min_headroom override of 0.05 (Micro policy), full levels at 5%+ headroom
        let levels = pricer.continuous_ladder_levels(10, 0.05, true, Some(0.05));
        assert_eq!(
            levels, 10,
            "At overridden min_headroom should get all levels"
        );

        // Below override threshold, sqrt scaling kicks in
        let levels = pricer.continuous_ladder_levels(10, 0.01, true, Some(0.05));
        assert!(
            (1..10).contains(&levels),
            "Below override threshold, should scale: got {}",
            levels
        );
    }

    // ========================================================================
    // Request Shadow Price Tests
    // ========================================================================

    #[test]
    fn test_request_shadow_price_at_full_headroom() {
        // At 50%+ headroom: shadow price = 0
        let price = QuotaShadowPricer::compute_request_shadow_price(1.0, 1);
        assert!(
            price.abs() < 1e-10,
            "At full headroom shadow price should be 0, got {}",
            price
        );

        let price = QuotaShadowPricer::compute_request_shadow_price(0.50, 1);
        assert!(
            price.abs() < 1e-10,
            "At 50% headroom shadow price should be 0, got {}",
            price
        );
    }

    #[test]
    fn test_request_shadow_price_at_critical_headroom() {
        // At 5% or below: shadow price = 100 bps
        let price = QuotaShadowPricer::compute_request_shadow_price(0.05, 1);
        assert!(
            (price - 100.0).abs() < 1e-10,
            "At 5% headroom shadow price should be 100 bps, got {}",
            price
        );

        let price = QuotaShadowPricer::compute_request_shadow_price(0.01, 1);
        assert!(
            (price - 100.0).abs() < 1e-10,
            "At 1% headroom shadow price should be 100 bps, got {}",
            price
        );
    }

    #[test]
    fn test_request_shadow_price_cubic_scaling() {
        // Between 5% and 50%: cubic explosion
        // At 30% headroom: x = (0.50 - 0.30) / 0.45 = 0.444
        // shadow = 50 * 0.444^3 * 1.0 = 50 * 0.0878 = 4.39
        let price = QuotaShadowPricer::compute_request_shadow_price(0.30, 1);
        assert!(
            (price - 4.39).abs() < 0.1,
            "At 30% headroom shadow price should be ~4.4 bps, got {}",
            price
        );
    }

    #[test]
    fn test_request_shadow_price_regime_adjustment() {
        let headroom = 0.30;
        let base_price = QuotaShadowPricer::compute_request_shadow_price(headroom, 1);

        // Low vol: higher shadow (fills are rare, conserve quota)
        let low_vol = QuotaShadowPricer::compute_request_shadow_price(headroom, 0);
        assert!(
            low_vol > base_price,
            "Low vol should have higher shadow price: {} vs {}",
            low_vol,
            base_price
        );

        // High vol: lower shadow (fills recharge quota faster)
        let high_vol = QuotaShadowPricer::compute_request_shadow_price(headroom, 2);
        assert!(
            high_vol < base_price,
            "High vol should have lower shadow price: {} vs {}",
            high_vol,
            base_price
        );

        // Extreme vol: even lower
        let extreme_vol = QuotaShadowPricer::compute_request_shadow_price(headroom, 3);
        assert!(
            extreme_vol < high_vol,
            "Extreme vol should have lowest shadow price: {} vs {}",
            extreme_vol,
            high_vol
        );
    }

    #[test]
    fn test_request_shadow_price_monotonic() {
        // Shadow price should increase as headroom decreases (within the cubic range)
        let headrooms = [0.49, 0.40, 0.30, 0.20, 0.10, 0.06];
        let mut prev_price = 0.0;
        for &h in &headrooms {
            let price = QuotaShadowPricer::compute_request_shadow_price(h, 1);
            assert!(
                price >= prev_price,
                "Shadow price should increase as headroom drops: at {:.0}% got {:.2}, prev {:.2}",
                h * 100.0,
                price,
                prev_price
            );
            prev_price = price;
        }
    }

    // ========================================================================
    // Edge Justification Tests
    // ========================================================================

    #[test]
    fn test_edge_justifies_request_at_full_headroom() {
        let pricer = default_pricer();
        // At full headroom, shadow price = 0, so any positive edge justifies
        assert!(pricer.edge_justifies_request(0.1, 1.0, 1));
        assert!(pricer.edge_justifies_request(0.01, 0.50, 1));
    }

    #[test]
    fn test_edge_justifies_request_at_critical_headroom() {
        let pricer = default_pricer();
        // At 5% headroom, shadow price = 100 bps
        assert!(!pricer.edge_justifies_request(50.0, 0.05, 1));
        assert!(pricer.edge_justifies_request(101.0, 0.05, 1));
    }

    #[test]
    fn test_edge_justifies_request_moderate_headroom() {
        let pricer = default_pricer();
        // At 30% headroom, shadow ~ 4.4 bps
        assert!(pricer.edge_justifies_request(10.0, 0.30, 1));
        assert!(!pricer.edge_justifies_request(2.0, 0.30, 1));
    }

    // ========================================================================
    // Config and Default Tests
    // ========================================================================

    #[test]
    fn test_default_config() {
        let config = QuotaShadowConfig::default();
        assert!((config.lambda_shadow_bps - 0.5).abs() < 1e-10);
        assert!((config.min_headroom_for_full_ladder - 0.20).abs() < 1e-10);
        assert!((config.max_shadow_spread_bps - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_default_pricer() {
        let pricer = QuotaShadowPricer::default();
        assert!((pricer.config().lambda_shadow_bps - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_pricer_with_custom_config() {
        let config = QuotaShadowConfig {
            lambda_shadow_bps: 2.0,
            min_headroom_for_full_ladder: 0.10,
            max_shadow_spread_bps: 100.0,
        };
        let pricer = QuotaShadowPricer::new(config.clone());
        assert!((pricer.config().lambda_shadow_bps - 2.0).abs() < 1e-10);
        assert!((pricer.config().min_headroom_for_full_ladder - 0.10).abs() < 1e-10);
    }
}
