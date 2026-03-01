//! Viable quote ladder: exchange-minimum-aware quote management.
//!
//! Wraps quote vectors with a type-level guarantee that all surviving quotes
//! meet the exchange minimum notional requirement. When a size reduction
//! would push below minimum, the system either removes the level (if others
//! exist) or converts the defense intent to spread widening (if it's the
//! last level on a side).
//!
//! # Design Principle
//!
//! The exchange minimum notional is a *foundational constraint*, not a
//! *terminal filter*. Every stage in the pipeline respects it, or converts
//! its defense intent to an alternative mechanism (wider spread).

use crate::market_maker::config::{Quote, SizeQuantum};
use crate::market_maker::quoting::exchange_rules::{
    ExchangeRules, ValidatedQuote, ValidationReport,
};
use crate::market_maker::quoting::ladder::Ladder;

/// Maximum spread widening multiplier from defense-via-spread conversions.
/// Prevents quotes from moving unreasonably far from mid.
const MAX_DEFENSE_SPREAD_MULT: f64 = 3.0;

/// One side (bids or asks) of a viable quote ladder.
///
/// All quotes are guaranteed to have `size >= quantum.min_viable_size`
/// after any mutation, unless the side has been explicitly cleared.
#[derive(Debug, Clone)]
pub struct ViableQuoteSide {
    quotes: Vec<Quote>,
    /// Accumulated spread widening from defense-via-spread conversions.
    /// When a size reduction is blocked (last level at minimum),
    /// the factor is absorbed here and applied as spread widening on finalize.
    defense_spread_mult: f64,
}

impl ViableQuoteSide {
    /// Create a new viable quote side from existing quotes.
    pub fn new(quotes: Vec<Quote>) -> Self {
        Self {
            quotes,
            defense_spread_mult: 1.0,
        }
    }

    /// Reduce all quote sizes by `factor` (0 < factor < 1).
    ///
    /// For each quote:
    /// - If `size * factor >= min_viable_size`: normal reduction
    /// - If other levels exist: remove this level (equivalent exposure reduction)
    /// - If this is the last level: keep at `min_viable_size`, convert remaining
    ///   reduction to spread widening (capped at 3x total)
    ///
    /// No-op if factor >= 1.0 or side is empty.
    pub fn reduce_sizes(&mut self, factor: f64, quantum: &SizeQuantum) {
        if factor >= 1.0 || self.quotes.is_empty() {
            return;
        }

        // Clamp factor to avoid negative/zero sizes
        let factor = factor.max(0.01);

        // Partition: which quotes survive the reduction?
        let mut survive_indices = Vec::new();
        let mut casualty_indices = Vec::new();

        for (i, q) in self.quotes.iter().enumerate() {
            if q.size * factor >= quantum.min_viable_size {
                survive_indices.push(i);
            } else {
                casualty_indices.push(i);
            }
        }

        if casualty_indices.is_empty() {
            // All survive — normal reduction
            for q in &mut self.quotes {
                q.size *= factor;
            }
        } else if !survive_indices.is_empty() {
            // Some survive, some don't — apply reduction, remove casualties
            for q in &mut self.quotes {
                q.size *= factor;
            }
            // Remove casualties in reverse order to maintain indices
            for &i in casualty_indices.iter().rev() {
                self.quotes.remove(i);
            }
        } else {
            // NO quotes would survive — floor-protect the best (first) quote
            self.quotes.truncate(1);
            self.quotes[0].size = quantum.min_viable_size;

            // Convert the intended reduction to spread widening
            // factor=0.82 → widening = 1/0.82 ≈ 1.22x
            let widening = 1.0 / factor;
            self.defense_spread_mult =
                (self.defense_spread_mult * widening).min(MAX_DEFENSE_SPREAD_MULT);
        }
    }

    /// Retain quotes matching a predicate, protecting the last level.
    ///
    /// If the predicate would remove ALL quotes, the best (first) quote
    /// is preserved. This prevents filter stages like QueueValue from
    /// emptying a side that was intentionally kept.
    ///
    /// Use `clear()` for intentional side removal.
    pub fn retain<F: Fn(&Quote) -> bool>(&mut self, predicate: F) {
        if self.quotes.is_empty() {
            return;
        }

        // Save the best quote before retain (Quote is Copy)
        let best = self.quotes[0];
        self.quotes.retain(|q| predicate(q));

        // Protect last level: if retain emptied the side, keep best
        if self.quotes.is_empty() {
            self.quotes.push(best);
        }
    }

    /// Clear all quotes on this side (intentional removal).
    pub fn clear(&mut self) {
        self.quotes.clear();
    }

    /// Truncate to at most `n` quotes (keeps the best/tightest).
    pub fn truncate(&mut self, n: usize) {
        self.quotes.truncate(n);
    }

    /// Whether this side has no quotes.
    pub fn is_empty(&self) -> bool {
        self.quotes.is_empty()
    }

    /// Number of quotes on this side.
    pub fn len(&self) -> usize {
        self.quotes.len()
    }

    /// Get the best (tightest) quote, if any.
    pub fn first(&self) -> Option<&Quote> {
        self.quotes.first()
    }

    /// Iterate over quotes (immutable).
    pub fn iter(&self) -> impl Iterator<Item = &Quote> {
        self.quotes.iter()
    }

    /// Iterate over quotes (mutable) for price adjustments.
    ///
    /// Safe for price modifications; do NOT use to reduce sizes below minimum.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Quote> {
        self.quotes.iter_mut()
    }

    /// Access the underlying quote vector (immutable).
    pub fn quotes(&self) -> &[Quote] {
        &self.quotes
    }

    /// Access the underlying quote vector (mutable).
    ///
    /// Exposed for `QuoteFilter::apply_reduce_only_*` which only calls `clear()`.
    /// Do NOT use to reduce individual sizes below minimum.
    pub fn quotes_mut(&mut self) -> &mut Vec<Quote> {
        &mut self.quotes
    }

    /// Current accumulated defense spread multiplier.
    pub fn defense_spread_mult(&self) -> f64 {
        self.defense_spread_mult
    }

    /// Apply defense spread widening to surviving quotes.
    ///
    /// Moves quotes farther from mid proportionally to `defense_spread_mult`.
    /// `is_bid`: true for bid side (prices move down), false for ask (prices move up).
    /// Re-rounds prices to exchange precision after widening — this fixes the
    /// "Order has invalid price" rejection that occurred when widened offsets
    /// produced prices with too many decimal places.
    fn apply_defense_spread_widening(
        &mut self,
        mid: f64,
        is_bid: bool,
        rules: Option<&ExchangeRules>,
    ) {
        if self.defense_spread_mult <= 1.0 {
            return;
        }
        for q in &mut self.quotes {
            let offset = (q.price - mid).abs();
            let widened_offset = offset * self.defense_spread_mult;
            let raw_price = if is_bid {
                mid - widened_offset
            } else {
                mid + widened_offset
            };
            // Re-round to exchange precision after widening.
            // Without this, widened prices can have arbitrary decimal places.
            q.price = match rules {
                Some(r) => r.round_price(raw_price),
                None => raw_price,
            };
        }
    }
}

/// A viable quote ladder with exchange-minimum guarantees.
///
/// Wraps bid and ask sides with size-floor protection. All mutations
/// through `reduce_sizes` and `retain` guarantee that surviving quotes
/// meet the exchange minimum notional, or convert defense intent to
/// spread widening.
///
/// Call `finalize()` to extract validated quotes for submission.
#[derive(Debug, Clone)]
pub struct ViableQuoteLadder {
    /// Bid side (highest price first).
    pub bids: ViableQuoteSide,
    /// Ask side (lowest price first).
    pub asks: ViableQuoteSide,
    /// Exchange minimum size quantum.
    quantum: SizeQuantum,
    /// Mid price for spread widening calculations.
    mid_px: f64,
    /// Exchange price/size validation rules (if available).
    /// When set, `finalize()` validates all quotes and returns `ValidatedQuote`.
    rules: Option<ExchangeRules>,
}

impl ViableQuoteLadder {
    /// Create a viable quote ladder from a raw Ladder.
    ///
    /// Sizes are clamped up to `quantum.min_viable_size` if slightly below.
    /// Levels with zero or negative size are dropped.
    ///
    /// When `rules` is provided, `finalize()` will validate all quotes against
    /// exchange precision rules and re-round prices after defense spread widening.
    pub fn from_ladder(
        ladder: &Ladder,
        quantum: SizeQuantum,
        mid_px: f64,
        rules: Option<ExchangeRules>,
    ) -> Self {
        // Helper: ensure size meets min_notional at THIS level's price.
        // Quantum's min_viable_size was computed at mark_px, but price grid
        // snapping or other post-ladder adjustments can shift prices, making
        // min_viable_size * actual_price < min_notional.
        let ensure_notional = |size: f64, price: f64| -> f64 {
            if price > 0.0 && size * price < quantum.min_notional {
                let raw = quantum.min_notional / price;
                let steps = (raw / quantum.step).ceil() as u64;
                (steps as f64 * quantum.step).max(size)
            } else {
                size
            }
        };

        let bid_quotes: Vec<Quote> = ladder
            .bids
            .iter()
            .filter_map(|l| {
                quantum
                    .clamp_to_viable(l.size, true)
                    .map(|size| Quote::new(l.price, ensure_notional(size, l.price)))
            })
            .collect();
        let ask_quotes: Vec<Quote> = ladder
            .asks
            .iter()
            .filter_map(|l| {
                quantum
                    .clamp_to_viable(l.size, true)
                    .map(|size| Quote::new(l.price, ensure_notional(size, l.price)))
            })
            .collect();

        Self {
            bids: ViableQuoteSide::new(bid_quotes),
            asks: ViableQuoteSide::new(ask_quotes),
            quantum,
            mid_px,
            rules,
        }
    }

    /// Get a reference to the SizeQuantum.
    pub fn quantum(&self) -> &SizeQuantum {
        &self.quantum
    }

    /// Whether both sides are empty.
    pub fn is_empty(&self) -> bool {
        self.bids.is_empty() && self.asks.is_empty()
    }

    /// Finalize the ladder: apply defense spread widening, validate, and return quotes.
    ///
    /// When `ExchangeRules` was provided at construction, returns validated quotes
    /// with a validation report. Auto-fixes prices that need re-rounding.
    ///
    /// Returns `(bid_quotes, ask_quotes, validation_report)`.
    /// All returned quotes meet notional minimum and exchange precision rules.
    pub fn finalize(mut self) -> (Vec<ValidatedQuote>, Vec<ValidatedQuote>, ValidationReport) {
        let rules_ref = self.rules.as_ref();
        self.bids
            .apply_defense_spread_widening(self.mid_px, true, rules_ref);
        self.asks
            .apply_defense_spread_widening(self.mid_px, false, rules_ref);

        let mut report = ValidationReport::default();

        let validate_side = |quotes: Vec<Quote>,
                             is_buy: bool,
                             rules: Option<&ExchangeRules>,
                             report: &mut ValidationReport|
         -> Vec<ValidatedQuote> {
            report.proposed += quotes.len();

            match rules {
                Some(r) => {
                    let mut validated = Vec::with_capacity(quotes.len());
                    for q in &quotes {
                        match r.validate_quote(q.price, q.size, is_buy) {
                            Ok(vq) => {
                                if vq.was_fixed() {
                                    report.fixed += 1;
                                } else {
                                    report.valid += 1;
                                }
                                validated.push(vq);
                            }
                            Err(rejection) => {
                                report.rejected += 1;
                                report.rejection_reasons.push(rejection.to_string());
                            }
                        }
                    }
                    validated
                }
                None => {
                    // No rules: wrap quotes as-is (backward compat)
                    report.valid += quotes.len();
                    quotes
                        .iter()
                        .map(|q| {
                            // Use validate_quote-like construction but without ExchangeRules.
                            // Create a minimal ExchangeRules just for wrapping.
                            ValidatedQuote::from_raw(q.price, q.size, is_buy)
                        })
                        .collect()
                }
            }
        };

        // Debug assertion: all surviving quotes meet notional minimum
        #[cfg(debug_assertions)]
        {
            for q in &self.bids.quotes {
                debug_assert!(
                    q.size * q.price >= self.quantum.min_notional - 0.01,
                    "Bid quote notional violation: size={}, price={}, notional={}, min={}",
                    q.size,
                    q.price,
                    q.size * q.price,
                    self.quantum.min_notional,
                );
            }
            for q in &self.asks.quotes {
                debug_assert!(
                    q.size * q.price >= self.quantum.min_notional - 0.01,
                    "Ask quote notional violation: size={}, price={}, notional={}, min={}",
                    q.size,
                    q.price,
                    q.size * q.price,
                    self.quantum.min_notional,
                );
            }
        }

        let bid_validated = validate_side(self.bids.quotes, true, rules_ref, &mut report);
        let ask_validated = validate_side(self.asks.quotes, false, rules_ref, &mut report);

        (bid_validated, ask_validated, report)
    }

    /// Legacy finalize: return raw `Vec<Quote>` for backward compatibility.
    ///
    /// Applies defense spread widening and extracts raw quotes without validation.
    /// Prefer `finalize()` with `ExchangeRules` in production paths.
    pub fn finalize_raw(mut self) -> (Vec<Quote>, Vec<Quote>) {
        self.bids
            .apply_defense_spread_widening(self.mid_px, true, self.rules.as_ref());
        self.asks
            .apply_defense_spread_widening(self.mid_px, false, self.rules.as_ref());

        (self.bids.quotes, self.asks.quotes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a SizeQuantum for HYPE at ~$30, sz_decimals=2.
    /// min_viable_size = 0.34 (ceil(10.0/30.30/0.01) = 34 steps → 34 * 0.01 = 0.34).
    fn hype_quantum() -> SizeQuantum {
        SizeQuantum::compute(10.0, 30.30, 2)
    }

    /// Helper: create a SizeQuantum for BTC at $50k, sz_decimals=5.
    fn btc_quantum() -> SizeQuantum {
        SizeQuantum::compute(10.0, 50_000.0, 5)
    }

    // =========================================================
    // reduce_sizes tests
    // =========================================================

    #[test]
    fn test_reduce_above_minimum() {
        // Normal path: all quotes stay above minimum after reduction
        let quantum = hype_quantum(); // min_viable = 0.34
        let mut side = ViableQuoteSide::new(vec![
            Quote::new(30.0, 1.0),
            Quote::new(29.9, 0.80),
            Quote::new(29.8, 0.70),
        ]);

        side.reduce_sizes(0.5, &quantum);

        // All should survive at half size (0.50, 0.40, 0.35 — all >= 0.34)
        assert_eq!(side.len(), 3);
        assert!((side.quotes[0].size - 0.5).abs() < 1e-10);
        assert!((side.quotes[1].size - 0.40).abs() < 1e-10);
        assert!((side.quotes[2].size - 0.35).abs() < 1e-10);
        // No defense widening needed
        assert!((side.defense_spread_mult - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reduce_below_minimum_removes_level() {
        // Multi-level: some quotes would go below min, they get removed
        let quantum = hype_quantum(); // min_viable = 0.34
        let mut side = ViableQuoteSide::new(vec![
            Quote::new(30.0, 1.0),  // 1.0 * 0.5 = 0.50 >= 0.34 ✓
            Quote::new(29.9, 0.50), // 0.50 * 0.5 = 0.25 < 0.34 ✗
            Quote::new(29.8, 0.40), // 0.40 * 0.5 = 0.20 < 0.34 ✗
        ]);

        side.reduce_sizes(0.5, &quantum);

        // Only the first level should survive
        assert_eq!(side.len(), 1);
        assert!((side.quotes[0].size - 0.50).abs() < 1e-10);
        assert!((side.quotes[0].price - 30.0).abs() < 1e-10);
        // No defense widening (we had survivors)
        assert!((side.defense_spread_mult - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reduce_below_minimum_last_level_floor_protected() {
        // Single level: can't remove, so floor-protect and widen spread
        let quantum = hype_quantum(); // min_viable = 0.34
        let mut side = ViableQuoteSide::new(vec![
            Quote::new(30.0, 0.34), // Already at minimum
        ]);

        side.reduce_sizes(0.82, &quantum);

        // Should be floor-protected at min_viable_size
        assert_eq!(side.len(), 1);
        assert!(
            (side.quotes[0].size - quantum.min_viable_size).abs() < 1e-10,
            "Size should be floor-protected at {}, got {}",
            quantum.min_viable_size,
            side.quotes[0].size,
        );
        // Defense spread widening should absorb the reduction
        // factor=0.82 → widening = 1/0.82 ≈ 1.2195
        let expected_mult = 1.0 / 0.82;
        assert!(
            (side.defense_spread_mult - expected_mult).abs() < 0.01,
            "Expected defense_spread_mult ≈ {:.4}, got {:.4}",
            expected_mult,
            side.defense_spread_mult,
        );
    }

    #[test]
    fn test_defense_spread_capped_at_3x() {
        // Multiple reductions accumulate, but cap at 3x
        let quantum = hype_quantum(); // min_viable = 0.34
        let mut side = ViableQuoteSide::new(vec![
            Quote::new(30.0, 0.34), // At minimum
        ]);

        // First reduction: 0.5x → widening = 2.0x
        side.reduce_sizes(0.5, &quantum);
        assert!(
            side.defense_spread_mult > 1.9 && side.defense_spread_mult < 2.1,
            "After 0.5x: expected ~2.0, got {:.4}",
            side.defense_spread_mult,
        );

        // Second reduction: 0.5x → widening would be 4.0x, capped at 3.0x
        side.reduce_sizes(0.5, &quantum);
        assert!(
            (side.defense_spread_mult - MAX_DEFENSE_SPREAD_MULT).abs() < 1e-10,
            "After two 0.5x: expected {}, got {:.4}",
            MAX_DEFENSE_SPREAD_MULT,
            side.defense_spread_mult,
        );
    }

    #[test]
    fn test_reduce_no_op_for_factor_ge_1() {
        let quantum = hype_quantum();
        let mut side = ViableQuoteSide::new(vec![Quote::new(30.0, 0.50)]);
        let original_size = side.quotes[0].size;

        side.reduce_sizes(1.0, &quantum);
        assert!((side.quotes[0].size - original_size).abs() < 1e-10);

        side.reduce_sizes(1.5, &quantum);
        assert!((side.quotes[0].size - original_size).abs() < 1e-10);
    }

    #[test]
    fn test_reduce_empty_side() {
        let quantum = hype_quantum();
        let mut side = ViableQuoteSide::new(vec![]);

        side.reduce_sizes(0.5, &quantum); // Should not panic
        assert!(side.is_empty());
    }

    // =========================================================
    // retain tests
    // =========================================================

    #[test]
    fn test_retain_normal() {
        let mut side = ViableQuoteSide::new(vec![
            Quote::new(30.0, 0.50),
            Quote::new(29.9, 0.50),
            Quote::new(29.8, 0.50),
        ]);

        // Keep only quotes with price > 29.85
        side.retain(|q| q.price > 29.85);

        assert_eq!(side.len(), 2);
        assert!((side.quotes[0].price - 30.0).abs() < 1e-10);
        assert!((side.quotes[1].price - 29.9).abs() < 1e-10);
    }

    #[test]
    fn test_retain_last_level_protection() {
        // If retain would remove ALL quotes, the best (first) is preserved
        let mut side = ViableQuoteSide::new(vec![Quote::new(30.0, 0.50), Quote::new(29.9, 0.50)]);

        // Predicate that matches nothing
        side.retain(|_| false);

        // Best quote should be preserved
        assert_eq!(side.len(), 1);
        assert!((side.quotes[0].price - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_retain_empty_side() {
        let mut side = ViableQuoteSide::new(vec![]);
        side.retain(|_| false); // Should not panic
        assert!(side.is_empty());
    }

    // =========================================================
    // ViableQuoteLadder tests
    // =========================================================

    #[test]
    fn test_hype_100_dollar_full_pipeline() {
        // THE scenario: $100 capital, HYPE at ~$30
        // Concentrated 0.34 contract order at 5 bps from mid.
        // After ToxicityRegime::Normal (0.82x), should survive via floor protection.
        let quantum = hype_quantum(); // min_viable = 0.34
        let mid = 30.30;

        let mut ladder = ViableQuoteLadder {
            bids: ViableQuoteSide::new(vec![Quote::new(
                mid * (1.0 - 5.0 / 10_000.0), // 5 bps below mid
                quantum.min_viable_size,      // 0.34 — exactly at minimum
            )]),
            asks: ViableQuoteSide::new(vec![Quote::new(
                mid * (1.0 + 5.0 / 10_000.0), // 5 bps above mid
                quantum.min_viable_size,      // 0.34
            )]),
            quantum,
            mid_px: mid,
            rules: None,
        };

        // Simulate ToxicityRegime::Normal — bid_mult=0.82, ask_mult=0.82
        ladder.bids.reduce_sizes(0.82, &ladder.quantum);
        ladder.asks.reduce_sizes(0.82, &ladder.quantum);

        // Both sides should survive (floor-protected)
        assert_eq!(ladder.bids.len(), 1, "Bid should survive floor protection");
        assert_eq!(ladder.asks.len(), 1, "Ask should survive floor protection");
        assert!(
            (ladder.bids.quotes()[0].size - quantum.min_viable_size).abs() < 1e-10,
            "Bid size should be floor-protected at {}",
            quantum.min_viable_size,
        );

        // Spread widening should be ~1/0.82 = 1.22x
        let expected_widening = 1.0 / 0.82;
        assert!(
            (ladder.bids.defense_spread_mult() - expected_widening).abs() < 0.01,
            "Expected bid defense_spread_mult ≈ {:.2}, got {:.2}",
            expected_widening,
            ladder.bids.defense_spread_mult(),
        );

        // Finalize: spread widening applied
        let (bids, asks, _report) = ladder.finalize();
        assert_eq!(bids.len(), 1);
        assert_eq!(asks.len(), 1);

        // Bid should have moved farther from mid (lower price)
        let original_bid_offset = mid * 5.0 / 10_000.0; // 5 bps
        let actual_bid_offset = mid - bids[0].price();
        assert!(
            actual_bid_offset > original_bid_offset,
            "Bid should be wider: original_offset={:.6}, actual_offset={:.6}",
            original_bid_offset,
            actual_bid_offset,
        );

        // Notional must meet exchange minimum
        assert!(
            bids[0].size() * bids[0].price() >= 10.0 - 0.01,
            "Bid notional {:.2} must be >= $10",
            bids[0].size() * bids[0].price(),
        );
        assert!(
            asks[0].size() * asks[0].price() >= 10.0 - 0.01,
            "Ask notional {:.2} must be >= $10",
            asks[0].size() * asks[0].price(),
        );
    }

    #[test]
    fn test_large_capital_no_protection() {
        // $10k+ capital: all reductions stay above minimum. No overhead.
        let quantum = btc_quantum(); // min_viable = 0.00020
        let mid = 50_000.0;

        let mut ladder = ViableQuoteLadder {
            bids: ViableQuoteSide::new(vec![
                Quote::new(49_990.0, 0.01),
                Quote::new(49_980.0, 0.01),
                Quote::new(49_970.0, 0.01),
            ]),
            asks: ViableQuoteSide::new(vec![
                Quote::new(50_010.0, 0.01),
                Quote::new(50_020.0, 0.01),
                Quote::new(50_030.0, 0.01),
            ]),
            quantum,
            mid_px: mid,
            rules: None,
        };

        // ToxicityRegime::Normal — 0.82x
        ladder.bids.reduce_sizes(0.82, &ladder.quantum);
        ladder.asks.reduce_sizes(0.82, &ladder.quantum);

        // All 3 levels per side should survive (0.01 * 0.82 = 0.0082 >> 0.00020)
        assert_eq!(ladder.bids.len(), 3);
        assert_eq!(ladder.asks.len(), 3);

        // No defense widening needed
        assert!(
            (ladder.bids.defense_spread_mult() - 1.0).abs() < 1e-10,
            "Large capital should have no defense widening",
        );

        // Finalize: prices unchanged (no widening)
        let (bids, asks, _report) = ladder.finalize();
        assert!((bids[0].price() - 49_990.0).abs() < 1e-10);
        assert!((asks[0].price() - 50_010.0).abs() < 1e-10);
    }

    #[test]
    fn test_finalize_applies_spread_widening() {
        let mid = 100.0;
        let _quantum = SizeQuantum::compute(10.0, mid, 2); // min_viable = 0.10

        let mut bids = ViableQuoteSide::new(vec![Quote::new(99.90, 0.50)]); // 10 bps from mid
        let mut asks = ViableQuoteSide::new(vec![Quote::new(100.10, 0.50)]); // 10 bps from mid

        // Manually set defense spread mult to 2.0x
        bids.defense_spread_mult = 2.0;
        asks.defense_spread_mult = 1.5;

        bids.apply_defense_spread_widening(mid, true, None);
        asks.apply_defense_spread_widening(mid, false, None);

        // Bid: offset was 0.10, widened by 2.0x → 0.20 → new price = 99.80
        assert!(
            (bids.quotes[0].price - 99.80).abs() < 1e-10,
            "Bid should widen to 99.80, got {:.4}",
            bids.quotes[0].price,
        );

        // Ask: offset was 0.10, widened by 1.5x → 0.15 → new price = 100.15
        assert!(
            (asks.quotes[0].price - 100.15).abs() < 1e-10,
            "Ask should widen to 100.15, got {:.4}",
            asks.quotes[0].price,
        );
    }

    #[test]
    fn test_multiple_stacked_reductions() {
        // Toxicity (0.82x) + Risk Emergency (0.5x) = both convert to spread widening
        let quantum = hype_quantum(); // min_viable = 0.34
        let mut side = ViableQuoteSide::new(vec![
            Quote::new(30.0, 0.34), // At minimum
        ]);

        // First: Toxicity 0.82x
        side.reduce_sizes(0.82, &quantum);
        let after_first = side.defense_spread_mult;

        // Second: Risk Emergency 0.5x
        side.reduce_sizes(0.5, &quantum);
        let after_second = side.defense_spread_mult;

        // Size should still be at minimum
        assert!(
            (side.quotes[0].size - quantum.min_viable_size).abs() < 1e-10,
            "Size should stay at minimum: {}",
            side.quotes[0].size,
        );

        // Widening should accumulate: ~1/0.82 * 1/0.5 ≈ 2.44x
        let expected: f64 = (1.0 / 0.82) * (1.0 / 0.5);
        assert!(
            (after_second - expected.min(MAX_DEFENSE_SPREAD_MULT)).abs() < 0.1,
            "Expected accumulated widening ≈ {:.2}, got {:.2} (first={:.2})",
            expected.min(MAX_DEFENSE_SPREAD_MULT),
            after_second,
            after_first,
        );
    }

    #[test]
    fn test_clear_does_not_protect() {
        // clear() is an intentional removal — no protection
        let mut side = ViableQuoteSide::new(vec![Quote::new(30.0, 0.50)]);

        side.clear();
        assert!(side.is_empty(), "clear() should empty the side");
    }

    #[test]
    fn test_from_ladder_clamps_sizes() {
        // Ladder with a level slightly below min_viable — should be clamped up
        let quantum = hype_quantum(); // min_viable = 0.34

        let mut ladder = Ladder::default();
        ladder.bids.push(crate::market_maker::quoting::LadderLevel {
            price: 30.0,
            size: 0.30, // Below min_viable (0.34)
            depth_bps: 5.0,
        });
        ladder.asks.push(crate::market_maker::quoting::LadderLevel {
            price: 30.60,
            size: 0.50, // Above min_viable
            depth_bps: 5.0,
        });

        let viable = ViableQuoteLadder::from_ladder(&ladder, quantum, 30.30, None);

        // Bid should be clamped up to 0.34 (min_viable)
        assert_eq!(viable.bids.len(), 1);
        assert!(
            (viable.bids.quotes()[0].size - 0.34).abs() < 1e-10,
            "Should clamp up to min_viable: got {}",
            viable.bids.quotes()[0].size,
        );

        // Ask should stay at 0.50
        assert_eq!(viable.asks.len(), 1);
        assert!((viable.asks.quotes()[0].size - 0.50).abs() < 1e-10);
    }

    #[test]
    fn test_from_ladder_drops_zero_size() {
        let quantum = hype_quantum();

        let mut ladder = Ladder::default();
        ladder.bids.push(crate::market_maker::quoting::LadderLevel {
            price: 30.0,
            size: 0.0, // Zero size — should be dropped
            depth_bps: 5.0,
        });

        let viable = ViableQuoteLadder::from_ladder(&ladder, quantum, 30.30, None);
        assert!(viable.bids.is_empty());
    }

    // =========================================================
    // Defense widening + validation (regression tests)
    // =========================================================

    #[test]
    fn test_defense_widening_prices_valid_with_rules() {
        // After defense spread widening with ExchangeRules, all prices must be exchange-valid.
        // This is the bug that caused "Order has invalid price" on line 67 of the log.
        let quantum = hype_quantum(); // min_viable = 0.34
        let mid = 30.30;
        let rules = ExchangeRules::new_perps(2, 10.0); // HYPE: price_decimals=4

        let mut ladder = ViableQuoteLadder {
            bids: ViableQuoteSide::new(vec![Quote::new(
                mid * (1.0 - 5.0 / 10_000.0),
                quantum.min_viable_size,
            )]),
            asks: ViableQuoteSide::new(vec![Quote::new(
                mid * (1.0 + 5.0 / 10_000.0),
                quantum.min_viable_size,
            )]),
            quantum,
            mid_px: mid,
            rules: Some(rules),
        };

        // Force max defense widening (3x) to stress-test rounding
        ladder.bids.reduce_sizes(0.5, &quantum);
        ladder.bids.reduce_sizes(0.5, &quantum); // 4x → capped at 3x

        let (bids, _asks, report) = ladder.finalize();
        assert_eq!(bids.len(), 1);

        // The widened price MUST be exchange-valid
        assert!(
            rules.is_valid_price(bids[0].price()),
            "Widened bid price {} must be exchange-valid",
            bids[0].price(),
        );

        // Report should show the price was fixed (or was already valid after re-rounding)
        assert_eq!(
            report.proposed,
            report.valid + report.fixed + report.rejected,
            "Accounting must balance: {report}",
        );
    }

    #[test]
    fn test_finalize_with_rules_validates_quotes() {
        // With ExchangeRules, finalize() produces ValidatedQuotes with valid prices.
        let quantum = SizeQuantum::compute(10.0, 100.0, 2); // min_viable = 0.10
        let mid = 100.0;
        let rules = ExchangeRules::new_perps(2, 10.0); // price_decimals=4

        let ladder = ViableQuoteLadder {
            bids: ViableQuoteSide::new(vec![Quote::new(99.90, 0.50)]),
            asks: ViableQuoteSide::new(vec![Quote::new(100.10, 0.50)]),
            quantum,
            mid_px: mid,
            rules: Some(rules),
        };

        let (bids, asks, report) = ladder.finalize();
        assert_eq!(report.proposed, 2);
        assert_eq!(report.rejected, 0);

        // All quotes must have valid prices
        for vq in &bids {
            assert!(
                rules.is_valid_price(vq.price()),
                "Bid price {} invalid",
                vq.price()
            );
        }
        for vq in &asks {
            assert!(
                rules.is_valid_price(vq.price()),
                "Ask price {} invalid",
                vq.price()
            );
        }
    }
}
