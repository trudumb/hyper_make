//! First-principles parameter derivation from capital and exchange metadata.
//!
//! All trading parameters can be auto-derived from a single input (`capital_usd`)
//! plus exchange context (mark price, margin, leverage, fees). This eliminates
//! arbitrary config numbers and ensures parameters are always self-consistent.

use super::spread_profile::SpreadProfile;

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

    DerivedParams {
        max_position,
        target_liquidity,
        risk_aversion,
        max_bps_diff,
        viable,
        diagnostic,
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
    }
}
