//! First-principles parameter derivation from capital and exchange metadata.
//!
//! All trading parameters can be auto-derived from a single input (`capital_usd`)
//! plus exchange context (mark price, margin, leverage, fees). This eliminates
//! arbitrary config numbers and ensures parameters are always self-consistent.

use serde::{Deserialize, Serialize};

use super::spread_profile::SpreadProfile;

/// Capital tier classification based on viable ladder levels per side.
///
/// Determines how the system adapts quoting behavior to available capital.
/// Tier boundaries are based on viable levels (each meeting exchange min_notional).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapitalTier {
    /// 1-2 viable levels/side — concentrated quoting, no multi-level ladder
    Micro,
    /// 3-5 viable levels/side — reduced ladder, wider spreads compensate
    Small,
    /// 6-15 viable levels/side — standard operation
    Medium,
    /// 16+ viable levels/side — full ladder capacity
    Large,
}

/// Capital-aware profile computed from position limits and exchange minimums.
///
/// Everything flows from this: ladder depth, execution mode fallbacks,
/// position budget floors, and spread compensation.
#[derive(Debug, Clone)]
pub struct CapitalProfile {
    /// Classification tier
    pub tier: CapitalTier,
    /// How many ladder levels per side can meet min_notional
    pub viable_levels_per_side: usize,
    /// Floor position size: (min_notional * 1.15) / mark_px
    /// This is the minimum position that produces a valid exchange order.
    pub min_viable_position: f64,
    /// min_notional * 1.15 in USD
    pub min_viable_notional_usd: f64,
    /// Total notional available per side in USD
    pub notional_per_side_usd: f64,
    /// Exchange minimum notional in USD
    pub min_notional_usd: f64,
    /// Mark price at time of computation (for staleness detection)
    pub mark_px_at_computation: f64,
}

impl CapitalProfile {
    /// Returns true if the mark price has moved enough to warrant recomputation.
    pub fn is_stale(&self, current_mark_px: f64) -> bool {
        if self.mark_px_at_computation <= 0.0 {
            return true;
        }
        let pct_change =
            ((current_mark_px - self.mark_px_at_computation) / self.mark_px_at_computation).abs();
        pct_change > 0.10 // >10% price move
    }
}

/// Exchange metadata needed for parameter derivation.
#[derive(Debug, Clone)]
pub struct ExchangeContext {
    /// Current mark price in USD.
    pub mark_px: f64,
    /// Account value in USD (equity).
    pub account_value: f64,
    /// Available margin in USD.
    pub available_margin: f64,
    /// Maximum leverage for this asset.
    pub max_leverage: f64,
    /// Maker fee in basis points (1.5 for Hyperliquid).
    pub fee_bps: f64,
    /// Size decimals for order rounding.
    pub sz_decimals: u32,
    /// Minimum order notional in USD (10.0 for Hyperliquid).
    pub min_notional: f64,
}

/// Auto-derived trading parameters.
#[derive(Debug, Clone)]
pub struct DerivedParams {
    /// Maximum position in contracts.
    pub max_position: f64,
    /// Target liquidity per side in contracts.
    pub target_liquidity: f64,
    /// Risk aversion (gamma) for GLFT formula.
    pub risk_aversion: f64,
    /// Maximum price deviation before requoting (basis points).
    pub max_bps_diff: u16,
    /// Whether the derived parameters allow viable trading.
    /// False when max_position < min_order_size.
    pub viable: bool,
    /// Human-readable explanation when not viable.
    pub diagnostic: Option<String>,
    /// Capital-aware profile for adaptive quoting behavior.
    pub capital_profile: CapitalProfile,
}

/// Derive all trading parameters from first principles.
///
/// Given a capital allocation, spread profile, and exchange metadata,
/// computes self-consistent parameters that respect exchange minimums
/// and margin constraints.
///
/// # Arguments
/// * `capital_usd` - Capital to deploy in USD (the ONE sizing input)
/// * `spread_profile` - Market type (Default/Hip3/Aggressive)
/// * `ctx` - Exchange metadata (price, margin, leverage, fees)
pub fn auto_derive(
    capital_usd: f64,
    spread_profile: SpreadProfile,
    ctx: &ExchangeContext,
) -> DerivedParams {
    let safety_factor = 0.5; // Leave 50% margin buffer for adverse moves

    // === MAX POSITION: from capital and margin solvency ===
    let max_from_capital = capital_usd / ctx.mark_px;
    let max_from_margin =
        (ctx.available_margin * ctx.max_leverage * safety_factor) / ctx.mark_px;
    let max_position = max_from_capital.min(max_from_margin).max(0.0);

    // === MINIMUM ORDER: from exchange min notional with safety margin ===
    // 1.15x buffer ensures truncation to sz_decimals doesn't drop below minimum
    let min_order = (ctx.min_notional * 1.15) / ctx.mark_px;

    // === VIABILITY CHECK ===
    let viable = max_position >= min_order;
    let diagnostic = if !viable {
        Some(format!(
            "Capital too small: min order ${:.2} ({:.4} contracts) > \
             max position ${:.2} ({:.4} contracts). \
             Need ${:.0}+ capital_usd for this asset at ${:.2}.",
            min_order * ctx.mark_px,
            min_order,
            max_position * ctx.mark_px,
            max_position,
            ctx.min_notional * 1.15 * 2.0, // Need 2x min for any meaningful quoting
            ctx.mark_px,
        ))
    } else {
        None
    };

    // === RISK AVERSION: from spread profile ===
    let risk_aversion = match spread_profile {
        SpreadProfile::Default => 0.3,
        SpreadProfile::Hip3 => 0.15,
        SpreadProfile::Aggressive => 0.10,
    };

    // === TARGET LIQUIDITY: GLFT-inspired sizing ===
    // Profile-dependent fraction of max position
    let sizing_fraction = match spread_profile {
        SpreadProfile::Default => 0.20,
        SpreadProfile::Hip3 => 0.30,
        SpreadProfile::Aggressive => 0.40,
    };
    let target_liquidity = (max_position * sizing_fraction)
        .max(min_order * 5.0) // At least 5 levels × min order
        .min(max_position); // Never exceed position limit

    // === MAX BPS DIFF: from fee structure ===
    // Cold-start estimate; runtime DynamicReconcileConfig refines from sigma
    let max_bps_diff = (ctx.fee_bps * 2.0).clamp(3.0, 15.0) as u16;

    // === CAPITAL PROFILE: from position limits and exchange minimums ===
    let min_viable_notional_usd = ctx.min_notional * 1.15;
    let min_viable_position = min_viable_notional_usd / ctx.mark_px;
    let notional_per_side_usd = max_position * ctx.mark_px / 2.0;
    let viable_levels = if min_viable_notional_usd > 0.0 {
        (notional_per_side_usd / min_viable_notional_usd).floor() as usize
    } else {
        0
    };
    let tier = match viable_levels {
        0..=2 => CapitalTier::Micro,
        3..=5 => CapitalTier::Small,
        6..=15 => CapitalTier::Medium,
        _ => CapitalTier::Large,
    };
    let capital_profile = CapitalProfile {
        tier,
        viable_levels_per_side: viable_levels,
        min_viable_position,
        min_viable_notional_usd,
        notional_per_side_usd,
        min_notional_usd: ctx.min_notional,
        mark_px_at_computation: ctx.mark_px,
    };

    DerivedParams {
        max_position,
        target_liquidity,
        risk_aversion,
        max_bps_diff,
        viable,
        diagnostic,
        capital_profile,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hype_context() -> ExchangeContext {
        ExchangeContext {
            mark_px: 20.0,
            account_value: 100.0,
            available_margin: 100.0,
            max_leverage: 3.0,
            fee_bps: 1.5,
            sz_decimals: 2,
            min_notional: 10.0,
        }
    }

    fn btc_context() -> ExchangeContext {
        ExchangeContext {
            mark_px: 100_000.0,
            account_value: 50_000.0,
            available_margin: 50_000.0,
            max_leverage: 50.0,
            fee_bps: 1.5,
            sz_decimals: 5,
            min_notional: 10.0,
        }
    }

    #[test]
    fn test_auto_derive_hype_small_capital() {
        let ctx = hype_context();
        let d = auto_derive(50.0, SpreadProfile::Hip3, &ctx);

        // max_position = min(50/20, 100*3*0.5/20) = min(2.5, 7.5) = 2.5
        assert!((d.max_position - 2.5).abs() < 0.01, "max_pos={}", d.max_position);
        assert!(d.viable, "Should be viable with $50 for HYPE at $20");
        assert!(d.diagnostic.is_none());

        // target_liquidity = max(2.5 * 0.30, min_order * 5) = max(0.75, 2.875)
        // min_order = (10 * 1.15) / 20 = 0.575
        // 0.575 * 5 = 2.875 > 0.75, so target_liquidity = 2.5 (capped by max_position)
        assert!(d.target_liquidity <= d.max_position + 0.01,
            "target_liquidity ({}) should not exceed max_position ({})",
            d.target_liquidity, d.max_position);

        // Hip3 gamma
        assert!((d.risk_aversion - 0.15).abs() < 0.01);
        // max_bps_diff = (1.5 * 2.0).max(3.0) = 3
        assert_eq!(d.max_bps_diff, 3);
    }

    #[test]
    fn test_auto_derive_btc_large_capital() {
        let ctx = btc_context();
        let d = auto_derive(10_000.0, SpreadProfile::Default, &ctx);

        // max_position = min(10000/100000, 50000*50*0.5/100000) = min(0.1, 12.5) = 0.1
        assert!((d.max_position - 0.1).abs() < 0.001, "max_pos={}", d.max_position);
        assert!(d.viable);

        // Default gamma
        assert!((d.risk_aversion - 0.3).abs() < 0.01);
        // target_liquidity = max(0.1 * 0.20, min_order * 5)
        // min_order = (10 * 1.15) / 100000 = 0.000115
        // 0.000115 * 5 = 0.000575
        // 0.1 * 0.20 = 0.02 > 0.000575, so target_liquidity = 0.02
        assert!(d.target_liquidity > 0.0);
        assert!(d.target_liquidity <= d.max_position + 0.001);
    }

    #[test]
    fn test_auto_derive_insufficient_capital() {
        let ctx = btc_context();
        let d = auto_derive(5.0, SpreadProfile::Default, &ctx);

        // max_position = 5/100000 = 0.00005 BTC = $5 notional
        // min_order = 11.5/100000 = 0.000115 BTC = $11.5 notional
        assert!(!d.viable, "Should NOT be viable with $5 for BTC");
        assert!(d.diagnostic.is_some());
        let diag = d.diagnostic.unwrap();
        assert!(diag.contains("Capital too small"), "diagnostic: {}", diag);
    }

    #[test]
    fn test_auto_derive_profiles_differ() {
        // Use large capital + margin so sizing_fraction dominates over min_order floor
        let ctx = ExchangeContext {
            mark_px: 20.0,
            account_value: 10_000.0,
            available_margin: 10_000.0,
            max_leverage: 3.0,
            fee_bps: 1.5,
            sz_decimals: 2,
            min_notional: 10.0,
        };
        let d_default = auto_derive(500.0, SpreadProfile::Default, &ctx);
        let d_hip3 = auto_derive(500.0, SpreadProfile::Hip3, &ctx);
        let d_aggressive = auto_derive(500.0, SpreadProfile::Aggressive, &ctx);

        // Gamma: Default > Hip3 > Aggressive
        assert!(d_default.risk_aversion > d_hip3.risk_aversion);
        assert!(d_hip3.risk_aversion > d_aggressive.risk_aversion);

        // max_position = min(500/20, 10000*3*0.5/20) = min(25, 750) = 25
        // Default: 25 * 0.20 = 5.0 > min_order*5 = 2.875
        // Hip3: 25 * 0.30 = 7.5
        // Aggressive: 25 * 0.40 = 10.0
        assert!(d_aggressive.target_liquidity > d_hip3.target_liquidity,
            "aggressive {} > hip3 {}", d_aggressive.target_liquidity, d_hip3.target_liquidity);
        assert!(d_hip3.target_liquidity > d_default.target_liquidity,
            "hip3 {} > default {}", d_hip3.target_liquidity, d_default.target_liquidity);
    }

    #[test]
    fn test_auto_derive_margin_caps_position() {
        // When margin is limited, it should cap max_position below capital/price
        let ctx = ExchangeContext {
            mark_px: 20.0,
            account_value: 10.0, // Only $10 in account
            available_margin: 10.0,
            max_leverage: 3.0,
            fee_bps: 1.5,
            sz_decimals: 2,
            min_notional: 10.0,
        };
        let d = auto_derive(1000.0, SpreadProfile::Default, &ctx);

        // max_from_capital = 1000/20 = 50
        // max_from_margin = 10*3*0.5/20 = 0.75
        // max_position = min(50, 0.75) = 0.75
        assert!((d.max_position - 0.75).abs() < 0.01, "max_pos={}", d.max_position);
    }

    #[test]
    fn test_auto_derive_max_bps_diff_bounds() {
        // Low fee: floor at 3
        let ctx = ExchangeContext {
            fee_bps: 0.5,
            ..hype_context()
        };
        let d = auto_derive(100.0, SpreadProfile::Default, &ctx);
        assert_eq!(d.max_bps_diff, 3, "Should floor at 3 bps");

        // High fee: cap at 15
        let ctx = ExchangeContext {
            fee_bps: 10.0,
            ..hype_context()
        };
        let d = auto_derive(100.0, SpreadProfile::Default, &ctx);
        assert_eq!(d.max_bps_diff, 15, "Should cap at 15 bps");
    }

    #[test]
    fn test_auto_derive_zero_capital() {
        let ctx = hype_context();
        let d = auto_derive(0.0, SpreadProfile::Default, &ctx);

        assert!(!d.viable);
        assert_eq!(d.max_position, 0.0);
        assert_eq!(d.capital_profile.tier, CapitalTier::Micro);
        assert_eq!(d.capital_profile.viable_levels_per_side, 0);
    }

    // === Capital Profile / Tier Tests ===

    fn hype30_context() -> ExchangeContext {
        ExchangeContext {
            mark_px: 30.0,
            account_value: 100_000.0, // Large margin so capital_usd is the binding constraint
            available_margin: 100_000.0,
            max_leverage: 10.0,
            fee_bps: 1.5,
            sz_decimals: 2,
            min_notional: 10.0,
        }
    }

    #[test]
    fn test_capital_tier_micro_25_usd() {
        let ctx = hype30_context();
        let d = auto_derive(25.0, SpreadProfile::Hip3, &ctx);
        // max_position = 25/30 = 0.833
        // notional_per_side = 0.833 * 30 / 2 = 12.5
        // min_viable_notional = 10 * 1.15 = 11.5
        // viable_levels = floor(12.5 / 11.5) = 1
        assert_eq!(d.capital_profile.tier, CapitalTier::Micro);
        assert_eq!(d.capital_profile.viable_levels_per_side, 1);
        assert!(d.viable);
    }

    #[test]
    fn test_capital_tier_micro_50_usd() {
        let ctx = hype30_context();
        let d = auto_derive(50.0, SpreadProfile::Hip3, &ctx);
        // max_position = 50/30 = 1.667
        // notional_per_side = 1.667 * 30 / 2 = 25.0
        // viable_levels = floor(25.0 / 11.5) = 2
        assert_eq!(d.capital_profile.tier, CapitalTier::Micro);
        assert_eq!(d.capital_profile.viable_levels_per_side, 2);
    }

    #[test]
    fn test_capital_tier_small_100_usd() {
        let ctx = hype30_context();
        let d = auto_derive(100.0, SpreadProfile::Hip3, &ctx);
        // max_position = 100/30 = 3.333
        // notional_per_side = 3.333 * 30 / 2 = 50.0
        // viable_levels = floor(50.0 / 11.5) = 4
        assert_eq!(d.capital_profile.tier, CapitalTier::Small, "tier={:?} levels={}", d.capital_profile.tier, d.capital_profile.viable_levels_per_side);
        assert_eq!(d.capital_profile.viable_levels_per_side, 4);
    }

    #[test]
    fn test_capital_tier_medium_500_usd() {
        let ctx = hype30_context();
        let d = auto_derive(500.0, SpreadProfile::Hip3, &ctx);
        // max_position = 500/30 = 16.667
        // notional_per_side = 16.667 * 30 / 2 = 250.0
        // viable_levels = floor(250.0 / 11.5) = 21
        // Wait, 21 > 15, that's Large
        assert_eq!(d.capital_profile.tier, CapitalTier::Large, "tier={:?} levels={}", d.capital_profile.tier, d.capital_profile.viable_levels_per_side);
    }

    #[test]
    fn test_capital_tier_large_5000_usd() {
        let ctx = hype30_context();
        let d = auto_derive(5000.0, SpreadProfile::Hip3, &ctx);
        assert_eq!(d.capital_profile.tier, CapitalTier::Large);
        assert!(d.capital_profile.viable_levels_per_side >= 16);
    }

    #[test]
    fn test_capital_profile_min_viable_position() {
        let ctx = hype30_context();
        let d = auto_derive(100.0, SpreadProfile::Hip3, &ctx);
        // min_viable_position = (10 * 1.15) / 30 = 0.383
        let expected = (10.0 * 1.15) / 30.0;
        assert!((d.capital_profile.min_viable_position - expected).abs() < 0.001,
            "min_viable={}, expected={}", d.capital_profile.min_viable_position, expected);
    }

    #[test]
    fn test_capital_profile_staleness_detection() {
        let ctx = hype30_context();
        let d = auto_derive(100.0, SpreadProfile::Hip3, &ctx);
        // Not stale at same price
        assert!(!d.capital_profile.is_stale(30.0));
        // Not stale at 5% move
        assert!(!d.capital_profile.is_stale(31.5));
        // Stale at 15% move
        assert!(d.capital_profile.is_stale(34.5));
        // Stale at -12% move
        assert!(d.capital_profile.is_stale(26.0));
    }

    #[test]
    fn test_capital_tier_5_usd_not_viable() {
        let ctx = hype30_context();
        let d = auto_derive(5.0, SpreadProfile::Hip3, &ctx);
        // max_position = 5/30 = 0.167
        // notional_per_side = 0.167 * 30 / 2 = 2.5
        // viable_levels = floor(2.5 / 11.5) = 0
        assert_eq!(d.capital_profile.tier, CapitalTier::Micro);
        assert_eq!(d.capital_profile.viable_levels_per_side, 0);
        assert!(!d.viable);
    }
}
