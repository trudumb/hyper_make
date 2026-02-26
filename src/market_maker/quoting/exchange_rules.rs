//! Exchange price/size validation rules for Hyperliquid.
//!
//! Provides a single source of truth for Hyperliquid's pricing constraints:
//! - Prices: max 5 significant figures, max `price_decimals` decimal places
//! - Sizes: truncated to `sz_decimals` decimal places
//! - Notional: price * size >= min_notional ($10 USD)
//!
//! # Types
//!
//! - [`ExchangeRules`]: Created once at startup from asset metadata.
//!   Methods: `round_price()`, `truncate_size()`, `is_valid_price()`, `validate_quote()`.
//!
//! - [`ValidatedQuote`]: Private fields, constructed only via `ExchangeRules::validate_quote()`.
//!   Once you have one, it is guaranteed exchange-valid.
//!
//! - [`QuoteRejection`]: Why a quote failed validation — NaN, below notional, zero size, etc.
//!   `PriceFixable` means the quote can be auto-fixed by re-rounding.
//!
//! - [`ValidationReport`]: Per-cycle stats (proposed / valid / fixed / rejected).

use std::fmt;

use crate::{round_to_significant_and_decimal, truncate_float};

/// Maximum significant figures allowed by Hyperliquid for prices.
const MAX_SIG_FIGS: u32 = 5;

/// Single source of truth for Hyperliquid pricing constraints.
///
/// Created once at startup from asset metadata (sz_decimals, price_decimals).
/// Thread this through the pipeline — replaces raw `decimals` param.
#[derive(Debug, Clone, Copy)]
pub struct ExchangeRules {
    /// Maximum significant figures for prices (always 5 on HL).
    pub max_sig_figs: u32,
    /// Maximum decimal places for prices (= MAX_DECIMALS - sz_decimals).
    pub price_decimals: u32,
    /// Decimal places for sizes (from exchange metadata).
    pub sz_decimals: u32,
    /// Tick size: 10^(-price_decimals), precomputed.
    pub tick_size: f64,
    /// Size step: 10^(-sz_decimals), precomputed.
    pub size_step: f64,
    /// Minimum order notional in USD ($10).
    pub min_notional: f64,
}

impl ExchangeRules {
    /// Construct exchange rules for perps (MAX_DECIMALS = 6).
    ///
    /// `price_decimals` = 6 - sz_decimals (perps).
    /// For spot, use `new_spot()` with MAX_DECIMALS = 8.
    pub fn new_perps(sz_decimals: u32, min_notional: f64) -> Self {
        let price_decimals = 6_u32.saturating_sub(sz_decimals);
        Self::new(price_decimals, sz_decimals, min_notional)
    }

    /// Construct exchange rules with explicit price_decimals.
    pub fn new(price_decimals: u32, sz_decimals: u32, min_notional: f64) -> Self {
        Self {
            max_sig_figs: MAX_SIG_FIGS,
            price_decimals,
            sz_decimals,
            tick_size: 10f64.powi(-(price_decimals as i32)),
            size_step: 10f64.powi(-(sz_decimals as i32)),
            min_notional,
        }
    }

    /// Round a raw price to exchange-valid precision.
    ///
    /// Applies both 5-sig-fig and max-decimal-place constraints.
    /// Idempotent: `round(round(x)) == round(x)`.
    #[inline]
    pub fn round_price(&self, price: f64) -> f64 {
        round_to_significant_and_decimal(price, self.max_sig_figs, self.price_decimals)
    }

    /// Truncate a raw size to exchange-valid precision (floor).
    #[inline]
    pub fn truncate_size(&self, size: f64) -> f64 {
        truncate_float(size, self.sz_decimals, false)
    }

    /// Round a raw size UP to exchange-valid precision (ceiling).
    ///
    /// Returns at least `size_step` for any positive input.
    /// This is the dual of `truncate_size` — use it when rounding down
    /// would produce zero and you want the minimum valid size instead.
    #[inline]
    pub fn ceil_size(&self, size: f64) -> f64 {
        if size <= 0.0 {
            return 0.0;
        }
        let truncated = self.truncate_size(size);
        if truncated >= size - 1e-12 {
            // Already at or above a grid point
            truncated
        } else {
            // Round up to next grid point (at least size_step)
            (truncated + self.size_step).max(self.size_step)
        }
    }

    /// Minimum valid order size (one size step).
    #[inline]
    pub fn min_order_size(&self) -> f64 {
        self.size_step
    }

    /// Check if a price is already exchange-valid (round-trip stable).
    ///
    /// A price is valid iff rounding it produces the same value (within float tolerance).
    #[inline]
    pub fn is_valid_price(&self, price: f64) -> bool {
        if !price.is_finite() || price <= 0.0 {
            return false;
        }
        let rounded = self.round_price(price);
        (rounded - price).abs() < 1e-12
    }

    /// Check if a price is on the tick grid: `(price / tick_size)` is integer.
    #[inline]
    pub fn is_on_tick_grid(&self, price: f64) -> bool {
        if !price.is_finite() || price <= 0.0 {
            return false;
        }
        let ticks = price / self.tick_size;
        (ticks - ticks.round()).abs() < 1e-6
    }

    /// Validate and produce a `ValidatedQuote`, auto-fixing prices if needed.
    ///
    /// Returns `Ok(ValidatedQuote)` if the quote passes all checks (possibly after price rounding).
    /// Returns `Err(QuoteRejection)` if the quote is fundamentally invalid.
    pub fn validate_quote(
        &self,
        price: f64,
        size: f64,
        is_buy: bool,
    ) -> Result<ValidatedQuote, QuoteRejection> {
        // Check for NaN/Inf
        if !price.is_finite() {
            return Err(QuoteRejection::InvalidPrice {
                raw_price: price,
                reason: "NaN or Inf",
            });
        }
        if !size.is_finite() {
            return Err(QuoteRejection::InvalidSize {
                raw_size: size,
                reason: "NaN or Inf",
            });
        }

        // Check for zero/negative
        if price <= 0.0 {
            return Err(QuoteRejection::InvalidPrice {
                raw_price: price,
                reason: "zero or negative",
            });
        }
        if size <= 0.0 {
            return Err(QuoteRejection::InvalidSize {
                raw_size: size,
                reason: "zero or negative",
            });
        }

        // Round price and round size to exchange precision.
        // Use ceil_size (round up) instead of truncate (round down) so that
        // positive sizes never become zero. A size of 0.003 with sz_decimals=2
        // produces 0.01 (one step) instead of 0.00.
        let rounded_price = self.round_price(price);
        let rounded_size = self.ceil_size(size);

        // Should never happen with ceil_size for positive input, but defensive
        if rounded_size <= 0.0 {
            return Err(QuoteRejection::InvalidSize {
                raw_size: size,
                reason: "rounded to zero",
            });
        }

        // Notional check
        let notional = rounded_price * rounded_size;
        if notional < self.min_notional - 0.01 {
            return Err(QuoteRejection::BelowNotional {
                notional,
                min_notional: self.min_notional,
            });
        }

        // Determine if we had to fix the price or size
        let was_fixed =
            (rounded_price - price).abs() > 1e-12 || (rounded_size - size).abs() > 1e-12;

        Ok(ValidatedQuote {
            price: rounded_price,
            size: rounded_size,
            is_buy,
            was_fixed,
        })
    }
}

/// A quote that has been validated against exchange rules.
///
/// Private fields — can only be constructed through `ExchangeRules::validate_quote()`.
/// Once you have one, the price is guaranteed 5-sig-fig + tick-grid valid,
/// the size is truncation-stable, and notional >= min_notional.
#[derive(Debug, Clone, Copy)]
pub struct ValidatedQuote {
    price: f64,
    size: f64,
    is_buy: bool,
    /// Whether the price was auto-rounded during validation.
    was_fixed: bool,
}

impl ValidatedQuote {
    /// The validated price (5 sig figs, on tick grid).
    #[inline]
    pub fn price(&self) -> f64 {
        self.price
    }

    /// The validated size (truncated to sz_decimals).
    #[inline]
    pub fn size(&self) -> f64 {
        self.size
    }

    /// Whether this is a buy order.
    #[inline]
    pub fn is_buy(&self) -> bool {
        self.is_buy
    }

    /// Whether the price was auto-fixed (re-rounded) during validation.
    #[inline]
    pub fn was_fixed(&self) -> bool {
        self.was_fixed
    }

    /// Deconstruct into (price, size) tuple for backward compatibility.
    #[inline]
    pub fn into_price_size(self) -> (f64, f64) {
        (self.price, self.size)
    }

    /// Create a ValidatedQuote from raw values without exchange rules validation.
    ///
    /// Used for backward compatibility when ExchangeRules is not available.
    /// The caller is responsible for ensuring the values are valid.
    pub fn from_raw(price: f64, size: f64, is_buy: bool) -> Self {
        Self {
            price,
            size,
            is_buy,
            was_fixed: false,
        }
    }
}

/// Why a quote failed validation.
#[derive(Debug, Clone)]
pub enum QuoteRejection {
    /// Price is NaN, Inf, zero, or negative.
    InvalidPrice {
        raw_price: f64,
        reason: &'static str,
    },
    /// Size is NaN, Inf, zero, negative, or truncated to zero.
    InvalidSize { raw_size: f64, reason: &'static str },
    /// Notional (price * size) below exchange minimum.
    BelowNotional { notional: f64, min_notional: f64 },
}

impl fmt::Display for QuoteRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuoteRejection::InvalidPrice { raw_price, reason } => {
                write!(f, "invalid price {raw_price}: {reason}")
            }
            QuoteRejection::InvalidSize { raw_size, reason } => {
                write!(f, "invalid size {raw_size}: {reason}")
            }
            QuoteRejection::BelowNotional {
                notional,
                min_notional,
            } => {
                write!(f, "notional {notional:.2} below minimum {min_notional:.2}")
            }
        }
    }
}

/// Per-cycle validation statistics.
///
/// Logged as `[VALIDATE]` every quote cycle.
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    /// Total quotes proposed (before validation).
    pub proposed: usize,
    /// Quotes that passed validation unchanged.
    pub valid: usize,
    /// Quotes that required price re-rounding (auto-fixed).
    pub fixed: usize,
    /// Quotes rejected (not submittable).
    pub rejected: usize,
    /// Rejection reasons for diagnostics.
    pub rejection_reasons: Vec<String>,
}

impl ValidationReport {
    /// Merge another report into this one.
    pub fn merge(&mut self, other: &ValidationReport) {
        self.proposed += other.proposed;
        self.valid += other.valid;
        self.fixed += other.fixed;
        self.rejected += other.rejected;
        self.rejection_reasons
            .extend(other.rejection_reasons.iter().cloned());
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "proposed={} valid={} fixed={} rejected={}",
            self.proposed, self.valid, self.fixed, self.rejected,
        )?;
        if !self.rejection_reasons.is_empty() {
            write!(f, " reasons={:?}", self.rejection_reasons)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ETH-like rules: sz_decimals=4, price_decimals=2
    fn eth_rules() -> ExchangeRules {
        ExchangeRules::new_perps(4, 10.0)
    }

    /// HYPE-like rules: sz_decimals=2, price_decimals=4
    fn hype_rules() -> ExchangeRules {
        ExchangeRules::new_perps(2, 10.0)
    }

    /// BTC-like rules: sz_decimals=5, price_decimals=1
    fn btc_rules() -> ExchangeRules {
        ExchangeRules::new_perps(5, 10.0)
    }

    // =========================================================
    // ExchangeRules construction
    // =========================================================

    #[test]
    fn test_perps_rules_construction() {
        let rules = eth_rules();
        assert_eq!(rules.max_sig_figs, 5);
        assert_eq!(rules.price_decimals, 2); // 6 - 4
        assert_eq!(rules.sz_decimals, 4);
        assert!((rules.tick_size - 0.01).abs() < 1e-12);
        assert!((rules.size_step - 0.0001).abs() < 1e-12);
        assert!((rules.min_notional - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_hype_rules_construction() {
        let rules = hype_rules();
        assert_eq!(rules.price_decimals, 4); // 6 - 2
        assert_eq!(rules.sz_decimals, 2);
        assert!((rules.tick_size - 0.0001).abs() < 1e-12);
        assert!((rules.size_step - 0.01).abs() < 1e-12);
    }

    // =========================================================
    // round_price tests
    // =========================================================

    #[test]
    fn test_round_price_eth_2_decimals() {
        let rules = eth_rules();
        // ETH at ~$2000 has 4-digit integer part. With 5 sig figs → 1 decimal place.
        // 1981.4537 → 1981.5 (5 sig figs: 1,9,8,1,5)
        let rounded = rules.round_price(1981.4537);
        assert!(
            (rounded - 1981.5).abs() < 1e-10,
            "Expected 1981.5, got {rounded}",
        );

        // A price with fewer integer digits gets more decimal precision.
        // 123.4567 → 123.46 (5 sig figs: 1,2,3,4,6; 2 decimals OK)
        let rounded2 = rules.round_price(123.4567);
        assert!(
            (rounded2 - 123.46).abs() < 1e-10,
            "Expected 123.46, got {rounded2}",
        );
    }

    #[test]
    fn test_round_price_hype_4_decimals() {
        let rules = hype_rules();
        // HYPE with price_decimals=4: 25.123456 → 25.123 (5 sig figs)
        let rounded = rules.round_price(25.123456);
        assert!(
            (rounded - 25.123).abs() < 1e-10,
            "Expected 25.123, got {rounded}",
        );
    }

    #[test]
    fn test_round_price_idempotent() {
        // round(round(x)) == round(x) for many random-ish prices
        let rules = eth_rules();
        let prices = [
            1000.0,
            1234.5678,
            99999.99,
            0.01,
            50000.12345,
            1.23456789,
            100.0,
            3000.001,
            2500.999,
            50.12345,
            0.12345,
            99.999,
        ];
        for &price in &prices {
            let r1 = rules.round_price(price);
            let r2 = rules.round_price(r1);
            assert!(
                (r1 - r2).abs() < 1e-12,
                "Idempotency failed for price={price}: round1={r1}, round2={r2}",
            );
        }

        let hype = hype_rules();
        for &price in &prices {
            let r1 = hype.round_price(price);
            let r2 = hype.round_price(r1);
            assert!(
                (r1 - r2).abs() < 1e-12,
                "Idempotency failed (HYPE) for price={price}: round1={r1}, round2={r2}",
            );
        }
    }

    #[test]
    fn test_is_valid_after_round() {
        // round(x) always passes is_valid_price()
        let rules = eth_rules();
        let prices = [1000.0, 1234.5678, 50000.12345, 0.01, 99999.99, 3000.001];
        for &price in &prices {
            let rounded = rules.round_price(price);
            assert!(
                rules.is_valid_price(rounded),
                "round({price}) = {rounded} should be valid",
            );
        }
    }

    #[test]
    fn test_tick_grid_alignment() {
        // round(x) / tick_size should be integer (within float tolerance)
        let rules = hype_rules(); // tick_size = 0.0001
        let prices = [25.0, 25.1234, 30.99876, 100.12345];
        for &price in &prices {
            let rounded = rules.round_price(price);
            assert!(
                rules.is_on_tick_grid(rounded),
                "round({price}) = {rounded} should be on tick grid (tick={})",
                rules.tick_size,
            );
        }
    }

    // =========================================================
    // is_valid_price edge cases
    // =========================================================

    #[test]
    fn test_invalid_price_nan_inf() {
        let rules = eth_rules();
        assert!(!rules.is_valid_price(f64::NAN));
        assert!(!rules.is_valid_price(f64::INFINITY));
        assert!(!rules.is_valid_price(f64::NEG_INFINITY));
        assert!(!rules.is_valid_price(0.0));
        assert!(!rules.is_valid_price(-1.0));
    }

    // =========================================================
    // validate_quote tests
    // =========================================================

    #[test]
    fn test_validated_quote_rejects_nan_inf_zero() {
        let rules = eth_rules();

        // NaN price
        assert!(rules.validate_quote(f64::NAN, 1.0, true).is_err());
        // Inf price
        assert!(rules.validate_quote(f64::INFINITY, 1.0, true).is_err());
        // Zero price
        assert!(rules.validate_quote(0.0, 1.0, true).is_err());
        // Negative price
        assert!(rules.validate_quote(-100.0, 1.0, true).is_err());

        // NaN size
        assert!(rules.validate_quote(100.0, f64::NAN, true).is_err());
        // Inf size
        assert!(rules.validate_quote(100.0, f64::INFINITY, true).is_err());
        // Zero size
        assert!(rules.validate_quote(100.0, 0.0, true).is_err());
        // Negative size
        assert!(rules.validate_quote(100.0, -1.0, true).is_err());
    }

    #[test]
    fn test_validated_quote_accepts_valid() {
        let rules = eth_rules();
        // ETH at $2000, size 0.01 → notional $20 > $10 ✓
        let result = rules.validate_quote(2000.0, 0.01, true);
        assert!(result.is_ok());
        let vq = result.unwrap();
        assert!((vq.price() - 2000.0).abs() < 1e-10);
        assert!((vq.size() - 0.01).abs() < 1e-10);
        assert!(vq.is_buy());
        assert!(!vq.was_fixed());
    }

    #[test]
    fn test_validated_quote_auto_fixes_price() {
        let rules = eth_rules();
        // Price with too many decimals: 1981.4537 → 1981.5 (5 sig figs, auto-fixed)
        let result = rules.validate_quote(1981.4537, 0.01, false);
        assert!(result.is_ok());
        let vq = result.unwrap();
        assert!((vq.price() - 1981.5).abs() < 1e-10);
        assert!(vq.was_fixed());
    }

    #[test]
    fn test_validated_quote_rejects_below_notional() {
        let rules = eth_rules();
        // ETH at $2000, size 0.0001 → notional $0.20 < $10 ✗
        let result = rules.validate_quote(2000.0, 0.0001, true);
        assert!(result.is_err());
        match result.unwrap_err() {
            QuoteRejection::BelowNotional { .. } => {} // expected
            other => panic!("Expected BelowNotional, got {other}"),
        }
    }

    #[test]
    fn test_validated_quote_into_price_size() {
        let rules = eth_rules();
        let vq = rules.validate_quote(2000.0, 0.01, true).unwrap();
        let (p, s) = vq.into_price_size();
        assert!((p - 2000.0).abs() < 1e-10);
        assert!((s - 0.01).abs() < 1e-10);
    }

    // =========================================================
    // ValidationReport tests
    // =========================================================

    #[test]
    fn test_validation_report_accounting() {
        let report = ValidationReport {
            proposed: 10,
            valid: 7,
            fixed: 2,
            rejected: 1,
            ..Default::default()
        };

        // proposed == valid + fixed + rejected
        assert_eq!(
            report.proposed,
            report.valid + report.fixed + report.rejected
        );
    }

    #[test]
    fn test_validation_report_merge() {
        let mut a = ValidationReport {
            proposed: 5,
            valid: 3,
            fixed: 1,
            rejected: 1,
            rejection_reasons: vec!["below notional".to_string()],
        };
        let b = ValidationReport {
            proposed: 3,
            valid: 2,
            fixed: 1,
            rejected: 0,
            rejection_reasons: vec![],
        };
        a.merge(&b);
        assert_eq!(a.proposed, 8);
        assert_eq!(a.valid, 5);
        assert_eq!(a.fixed, 2);
        assert_eq!(a.rejected, 1);
    }

    #[test]
    fn test_validation_report_display() {
        let report = ValidationReport {
            proposed: 10,
            valid: 8,
            fixed: 2,
            rejected: 0,
            rejection_reasons: vec![],
        };
        let s = format!("{report}");
        assert!(s.contains("proposed=10"));
        assert!(s.contains("valid=8"));
        assert!(s.contains("fixed=2"));
        assert!(s.contains("rejected=0"));
    }

    // =========================================================
    // Regression: ETH hyna first-cycle ask (the actual bug)
    // =========================================================

    #[test]
    fn test_eth_hyna_defense_widening_price_valid() {
        // Reproduce: ETH at $1983.45 mid, defense widening mult=1.54.
        // Ask at 5 bps offset → widened offset → must still be valid.
        let rules = eth_rules(); // price_decimals=2
        let mid = 1983.45;
        let original_offset = mid * 5.0 / 10000.0; // ~0.9917
        let widened_offset = original_offset * 1.54; // ~1.5273

        let raw_ask_price = mid + widened_offset; // ~1984.977...
                                                  // Without re-rounding, this price has too many decimals → exchange rejects!
        assert!(
            !rules.is_valid_price(raw_ask_price),
            "Raw widened price {raw_ask_price} should NOT be valid",
        );

        // After rounding, it should be valid
        let rounded = rules.round_price(raw_ask_price);
        assert!(
            rules.is_valid_price(rounded),
            "Rounded price {rounded} should be valid",
        );

        // And the validated quote should auto-fix it
        let vq = rules.validate_quote(raw_ask_price, 0.01, false).unwrap();
        assert!(vq.was_fixed());
        assert!(rules.is_valid_price(vq.price()));
    }

    #[test]
    fn test_btc_high_price_validation() {
        // BTC at ~$100,000 — 5 sig figs means only 1 decimal place matters
        let rules = btc_rules(); // price_decimals=1
        let price = 100_123.456;
        let rounded = rules.round_price(price);
        assert!(rules.is_valid_price(rounded));

        // Integer part of BTC prices can exceed 5 sig figs — that's OK
        // per HL docs: "Integer prices are always allowed regardless of sig figs"
        let integer_price = 100_123.0;
        let rounded_int = rules.round_price(integer_price);
        // This should round to 100120.0 due to 5 sig fig constraint
        assert!(rules.is_valid_price(rounded_int));
    }

    // =========================================================
    // ceil_size tests
    // =========================================================

    #[test]
    fn test_ceil_size_on_grid() {
        let rules = hype_rules(); // sz_decimals=2, size_step=0.01
                                  // Already on grid — no change
        assert!((rules.ceil_size(0.50) - 0.50).abs() < 1e-12);
        assert!((rules.ceil_size(0.01) - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_ceil_size_rounds_up() {
        let rules = hype_rules(); // sz_decimals=2, size_step=0.01
                                  // 0.003 rounds up to 0.01 (one step)
        assert!((rules.ceil_size(0.003) - 0.01).abs() < 1e-12);
        // 0.015 rounds up to 0.02
        assert!((rules.ceil_size(0.015) - 0.02).abs() < 1e-12);
    }

    #[test]
    fn test_ceil_size_zero_stays_zero() {
        let rules = eth_rules();
        assert!((rules.ceil_size(0.0)).abs() < 1e-12);
        assert!((rules.ceil_size(-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_ceil_size_min_order_size() {
        // Any positive size produces at least size_step
        let rules = hype_rules(); // size_step=0.01
        assert!((rules.ceil_size(0.001) - 0.01).abs() < 1e-12);
        assert_eq!(rules.min_order_size(), 0.01);
    }

    #[test]
    fn test_validate_quote_never_truncates_positive_to_zero() {
        // THE critical fix: a positive size should never become zero after validation.
        // Before this fix, size 0.003 with sz_decimals=2 would truncate to 0.00 → rejected.
        // Now it rounds UP to 0.01.
        let rules = hype_rules(); // sz_decimals=2, price at ~$25
                                  // Size 0.003 at $25 → rounds up to 0.01. Notional = $25 * 0.01 = $0.25 < $10
                                  // → BelowNotional (not "truncated to zero")
        let result = rules.validate_quote(25.0, 0.003, true);
        assert!(result.is_err());
        match result.unwrap_err() {
            QuoteRejection::BelowNotional { .. } => {} // Correct: below notional, NOT truncated to zero
            other => panic!("Expected BelowNotional, got {other}"),
        }
    }

    #[test]
    fn test_validate_quote_ceil_preserves_viable_size() {
        // A size that is viable after ceiling should pass validation.
        // ETH at $2700, size 0.004 with sz_decimals=4 → ceil(0.004) = 0.004 (already on grid)
        // Notional = $2700 * 0.004 = $10.80 > $10 ✓
        let rules = eth_rules(); // sz_decimals=4
        let result = rules.validate_quote(2700.0, 0.004, true);
        assert!(result.is_ok(), "Viable size should pass: {result:?}");
        let vq = result.unwrap();
        assert!((vq.size() - 0.004).abs() < 1e-12);
    }
}
