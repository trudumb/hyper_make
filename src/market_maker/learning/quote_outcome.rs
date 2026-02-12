//! Quote outcome tracking for unbiased edge estimation.
//!
//! Problem: Learning only from fills creates survivorship bias — fills are
//! biased toward adverse flow. Unfilled quotes (which survived the market)
//! are invisible to learning, making edge estimates permanently pessimistic.
//!
//! Solution: Track outcomes of ALL quotes — filled AND unfilled.
//! E[edge | state, spread] = P(fill) × E[edge | fill] + P(no fill) × 0
//! This enables finding the optimal spread = argmax(edge × fill_rate).

use std::collections::VecDeque;

/// Compact market state for pending quote tracking (~100 bytes).
#[derive(Debug, Clone)]
pub struct CompactMarketState {
    /// Microprice at quote time
    pub microprice: f64,
    /// Sigma at quote time
    pub sigma: f64,
    /// Book imbalance [-1, 1]
    pub book_imbalance: f64,
    /// Flow imbalance [-1, 1]
    pub flow_imbalance: f64,
    /// Toxicity score [0, 1]
    pub toxicity_score: f64,
    /// Kappa (arrival intensity)
    pub kappa: f64,
}

/// A quote that is awaiting its outcome (fill or expiry).
#[derive(Debug, Clone)]
pub struct PendingQuote {
    /// Timestamp when quote was placed (ms since epoch)
    pub timestamp_ms: u64,
    /// Quoted half-spread in bps
    pub half_spread_bps: f64,
    /// Side: true = bid, false = ask
    pub is_bid: bool,
    /// Market state snapshot at quote time
    pub state: CompactMarketState,
}

/// Outcome of a quote cycle.
#[derive(Debug, Clone)]
pub enum QuoteOutcome {
    /// Quote was filled — we have realized edge data.
    Filled {
        /// Realized edge in bps (spread_captured - AS - fees)
        edge_bps: f64,
        /// Market state at quote time
        state: CompactMarketState,
        /// Spread in bps
        spread_bps: f64,
    },
    /// Quote expired unfilled — counterfactual edge is 0.
    Expired {
        /// Market state at quote time
        state: CompactMarketState,
        /// Spread in bps (how wide we quoted)
        spread_bps: f64,
        /// Price move after the quote expired (bps, absolute)
        subsequent_move_bps: f64,
    },
}

impl QuoteOutcome {
    /// Whether this outcome was a fill.
    pub fn is_filled(&self) -> bool {
        matches!(self, Self::Filled { .. })
    }

    /// The spread at which this quote was placed.
    pub fn spread_bps(&self) -> f64 {
        match self {
            Self::Filled { spread_bps, .. } => *spread_bps,
            Self::Expired { spread_bps, .. } => *spread_bps,
        }
    }
}

/// Bin for tracking fill rates at different spread levels.
#[derive(Debug, Clone)]
struct SpreadBin {
    /// Lower bound of bin (bps)
    lo_bps: f64,
    /// Upper bound of bin (bps)
    hi_bps: f64,
    /// Number of fills in this bin
    fills: u64,
    /// Total quotes in this bin
    total: u64,
}

impl SpreadBin {
    fn fill_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.fills as f64 / self.total as f64
        }
    }
}

/// Tracks P(fill | spread_bin) empirically.
#[derive(Debug, Clone)]
pub struct BinnedFillRate {
    bins: Vec<SpreadBin>,
}

impl BinnedFillRate {
    /// Create binned tracker with default bin edges.
    /// Bins: [0,2), [2,4), [4,6), [6,8), [8,10), [10,15), [15,20), [20,∞)
    pub fn new() -> Self {
        let edges = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, f64::INFINITY];
        let bins = edges
            .windows(2)
            .map(|w| SpreadBin {
                lo_bps: w[0],
                hi_bps: w[1],
                fills: 0,
                total: 0,
            })
            .collect();
        Self { bins }
    }

    /// Record a quote outcome in the appropriate spread bin.
    pub fn record(&mut self, spread_bps: f64, filled: bool) {
        for bin in &mut self.bins {
            if spread_bps >= bin.lo_bps && spread_bps < bin.hi_bps {
                bin.total += 1;
                if filled {
                    bin.fills += 1;
                }
                break;
            }
        }
    }

    /// Get fill rate for a given spread.
    pub fn fill_rate_at(&self, spread_bps: f64) -> Option<f64> {
        for bin in &self.bins {
            if spread_bps >= bin.lo_bps && spread_bps < bin.hi_bps {
                if bin.total >= 5 {
                    return Some(bin.fill_rate());
                } else {
                    return None; // Insufficient data
                }
            }
        }
        None
    }

    /// Get (spread_midpoint, fill_rate, sample_count) for all bins with data.
    pub fn all_rates(&self) -> Vec<(f64, f64, u64)> {
        self.bins
            .iter()
            .filter(|b| b.total >= 5 && b.hi_bps.is_finite())
            .map(|b| ((b.lo_bps + b.hi_bps) / 2.0, b.fill_rate(), b.total))
            .collect()
    }

    /// Total quotes tracked across all bins.
    pub fn total_quotes(&self) -> u64 {
        self.bins.iter().map(|b| b.total).sum()
    }

    /// Total fills tracked across all bins.
    pub fn total_fills(&self) -> u64 {
        self.bins.iter().map(|b| b.fills).sum()
    }
}

impl Default for BinnedFillRate {
    fn default() -> Self {
        Self::new()
    }
}

/// Maximum age of a pending quote before it's considered expired (ms).
const MAX_PENDING_AGE_MS: u64 = 30_000; // 30 seconds

/// Maximum pending quotes to track simultaneously.
const MAX_PENDING_QUOTES: usize = 100;

/// Maximum outcome log size.
const MAX_OUTCOME_LOG: usize = 1000;

/// Tracks outcomes of quotes (filled and unfilled) for unbiased edge estimation.
#[derive(Debug, Clone)]
pub struct QuoteOutcomeTracker {
    /// Quotes awaiting outcome resolution.
    pending_quotes: VecDeque<PendingQuote>,
    /// Recent outcomes for learning.
    outcome_log: VecDeque<QuoteOutcome>,
    /// Empirical P(fill | spread_bin).
    fill_rate: BinnedFillRate,
    /// Current mid price for computing subsequent moves on expiry.
    last_mid: f64,
}

impl QuoteOutcomeTracker {
    /// Create a new tracker.
    pub fn new() -> Self {
        Self {
            pending_quotes: VecDeque::with_capacity(MAX_PENDING_QUOTES),
            outcome_log: VecDeque::with_capacity(MAX_OUTCOME_LOG),
            fill_rate: BinnedFillRate::new(),
            last_mid: 0.0,
        }
    }

    /// Update the current mid price (call each quote cycle).
    pub fn update_mid(&mut self, mid: f64) {
        self.last_mid = mid;
    }

    /// Register a new pending quote.
    pub fn register_quote(&mut self, quote: PendingQuote) {
        if self.pending_quotes.len() >= MAX_PENDING_QUOTES {
            // Expire oldest as "unfilled"
            if let Some(old) = self.pending_quotes.pop_front() {
                self.resolve_as_expired(old);
            }
        }
        self.pending_quotes.push_back(quote);
    }

    /// Resolve a pending quote as filled.
    /// Returns true if a matching quote was found and resolved.
    pub fn on_fill(&mut self, is_bid: bool, edge_bps: f64) -> bool {
        // Find the most recent matching pending quote on the same side
        let idx = self
            .pending_quotes
            .iter()
            .rposition(|q| q.is_bid == is_bid);

        if let Some(idx) = idx {
            let quote = self.pending_quotes.remove(idx).unwrap();
            let outcome = QuoteOutcome::Filled {
                edge_bps,
                state: quote.state,
                spread_bps: quote.half_spread_bps * 2.0,
            };
            self.fill_rate
                .record(quote.half_spread_bps * 2.0, true);
            self.push_outcome(outcome);
            true
        } else {
            false
        }
    }

    /// Expire old pending quotes (call periodically, e.g. each quote cycle).
    pub fn expire_old_quotes(&mut self, now_ms: u64) {
        while let Some(front) = self.pending_quotes.front() {
            if now_ms.saturating_sub(front.timestamp_ms) > MAX_PENDING_AGE_MS {
                let quote = self.pending_quotes.pop_front().unwrap();
                self.resolve_as_expired(quote);
            } else {
                break; // VecDeque is ordered by time
            }
        }
    }

    /// Resolve a quote as expired/unfilled.
    fn resolve_as_expired(&mut self, quote: PendingQuote) {
        let subsequent_move_bps = if self.last_mid > 0.0 && quote.state.microprice > 0.0 {
            ((self.last_mid - quote.state.microprice) / quote.state.microprice * 10000.0).abs()
        } else {
            0.0
        };

        let outcome = QuoteOutcome::Expired {
            state: quote.state,
            spread_bps: quote.half_spread_bps * 2.0,
            subsequent_move_bps,
        };
        self.fill_rate
            .record(quote.half_spread_bps * 2.0, false);
        self.push_outcome(outcome);
    }

    /// Push an outcome to the log, evicting oldest if at capacity.
    fn push_outcome(&mut self, outcome: QuoteOutcome) {
        if self.outcome_log.len() >= MAX_OUTCOME_LOG {
            self.outcome_log.pop_front();
        }
        self.outcome_log.push_back(outcome);
    }

    /// Get the binned fill rate tracker.
    pub fn fill_rate(&self) -> &BinnedFillRate {
        &self.fill_rate
    }

    /// Get recent outcome statistics.
    pub fn outcome_stats(&self) -> OutcomeStats {
        let n_total = self.outcome_log.len();
        let n_filled = self.outcome_log.iter().filter(|o| o.is_filled()).count();
        let n_expired = n_total - n_filled;

        let mean_edge = if n_filled > 0 {
            self.outcome_log
                .iter()
                .filter_map(|o| match o {
                    QuoteOutcome::Filled { edge_bps, .. } => Some(*edge_bps),
                    _ => None,
                })
                .sum::<f64>()
                / n_filled as f64
        } else {
            0.0
        };

        // Expected edge = P(fill) × E[edge|fill]
        let fill_rate = if n_total > 0 {
            n_filled as f64 / n_total as f64
        } else {
            0.0
        };
        let expected_edge = fill_rate * mean_edge;

        OutcomeStats {
            n_total,
            n_filled,
            n_expired,
            fill_rate,
            mean_edge_given_fill: mean_edge,
            expected_edge,
        }
    }

    /// Number of pending quotes.
    pub fn pending_count(&self) -> usize {
        self.pending_quotes.len()
    }

    /// Total outcomes recorded.
    pub fn total_outcomes(&self) -> usize {
        self.outcome_log.len()
    }

    /// Find the optimal spread (bps) that maximizes expected edge for a target fill rate.
    ///
    /// Expected edge = P(fill | spread) × E[edge | fill, spread].
    /// Returns the spread bin midpoint with the highest expected edge,
    /// or None if insufficient data.
    pub fn optimal_spread_bps(&self, _fill_rate_target: f64) -> Option<f64> {
        let rates = self.fill_rate.all_rates();
        if rates.is_empty() {
            return None;
        }

        let mut best_spread = None;
        let mut best_expected = f64::NEG_INFINITY;

        for (spread_mid, fill_rate, _count) in &rates {
            let edge_at_spread = self.mean_edge_at_spread(*spread_mid);
            let expected = fill_rate * edge_at_spread;

            if expected > best_expected {
                best_expected = expected;
                best_spread = Some(*spread_mid);
            }
        }

        best_spread
    }

    /// Estimate the expected edge (bps) at a given spread level.
    ///
    /// Uses the outcome log to find filled outcomes near this spread
    /// and returns P(fill) × E[edge | fill]. Returns 0.0 if no data.
    pub fn expected_edge_at(&self, spread_bps: f64) -> f64 {
        let fill_rate = self.fill_rate.fill_rate_at(spread_bps).unwrap_or(0.0);
        let mean_edge = self.mean_edge_at_spread(spread_bps);
        fill_rate * mean_edge
    }

    /// Mean realized edge for fills near a given spread level.
    fn mean_edge_at_spread(&self, spread_bps: f64) -> f64 {
        let tolerance_bps = 3.0;
        let mut sum = 0.0;
        let mut count = 0;

        for outcome in &self.outcome_log {
            if let QuoteOutcome::Filled {
                edge_bps,
                spread_bps: s,
                ..
            } = outcome
            {
                if (*s - spread_bps).abs() < tolerance_bps {
                    sum += edge_bps;
                    count += 1;
                }
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }
}

impl Default for QuoteOutcomeTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics from quote outcome tracking.
#[derive(Debug, Clone)]
pub struct OutcomeStats {
    /// Total outcomes tracked
    pub n_total: usize,
    /// Number of fills
    pub n_filled: usize,
    /// Number of expired/unfilled quotes
    pub n_expired: usize,
    /// Overall fill rate P(fill)
    pub fill_rate: f64,
    /// Mean edge conditioned on fill: E[edge | fill]
    pub mean_edge_given_fill: f64,
    /// Unbiased expected edge: P(fill) × E[edge | fill]
    pub expected_edge: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(microprice: f64) -> CompactMarketState {
        CompactMarketState {
            microprice,
            sigma: 0.01,
            book_imbalance: 0.0,
            flow_imbalance: 0.0,
            toxicity_score: 0.0,
            kappa: 1000.0,
        }
    }

    #[test]
    fn test_empty_tracker() {
        let tracker = QuoteOutcomeTracker::new();
        let stats = tracker.outcome_stats();
        assert_eq!(stats.n_total, 0);
        assert_eq!(stats.fill_rate, 0.0);
        assert_eq!(stats.expected_edge, 0.0);
    }

    #[test]
    fn test_fill_resolution() {
        let mut tracker = QuoteOutcomeTracker::new();
        tracker.update_mid(100.0);

        tracker.register_quote(PendingQuote {
            timestamp_ms: 1000,
            half_spread_bps: 5.0,
            is_bid: true,
            state: make_state(100.0),
        });

        assert_eq!(tracker.pending_count(), 1);

        // Resolve as fill
        let resolved = tracker.on_fill(true, 2.5);
        assert!(resolved);
        assert_eq!(tracker.pending_count(), 0);

        let stats = tracker.outcome_stats();
        assert_eq!(stats.n_filled, 1);
        assert_eq!(stats.n_expired, 0);
        assert!((stats.mean_edge_given_fill - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_expiry_resolution() {
        let mut tracker = QuoteOutcomeTracker::new();
        tracker.update_mid(100.0);

        tracker.register_quote(PendingQuote {
            timestamp_ms: 1000,
            half_spread_bps: 5.0,
            is_bid: true,
            state: make_state(100.0),
        });

        // Move time forward past expiry
        tracker.update_mid(100.05); // Price moved up 5 bps
        tracker.expire_old_quotes(40_000); // 39 seconds later

        assert_eq!(tracker.pending_count(), 0);

        let stats = tracker.outcome_stats();
        assert_eq!(stats.n_filled, 0);
        assert_eq!(stats.n_expired, 1);
        assert_eq!(stats.expected_edge, 0.0); // No fills => no expected edge
    }

    #[test]
    fn test_expected_edge_unbiased() {
        let mut tracker = QuoteOutcomeTracker::new();
        tracker.update_mid(100.0);

        // Simulate: 3 fills with +2 bps edge, 7 unfilled
        for i in 0..10 {
            tracker.register_quote(PendingQuote {
                timestamp_ms: i * 1000,
                half_spread_bps: 5.0,
                is_bid: true,
                state: make_state(100.0),
            });
        }

        // Fill 3 of them
        for _ in 0..3 {
            tracker.on_fill(true, 2.0);
        }
        // Expire the remaining 7
        tracker.expire_old_quotes(100_000);

        let stats = tracker.outcome_stats();
        assert_eq!(stats.n_filled, 3);
        assert_eq!(stats.n_expired, 7);
        assert!((stats.fill_rate - 0.3).abs() < 0.01);
        assert!((stats.mean_edge_given_fill - 2.0).abs() < 0.01);
        // Expected edge = 0.3 * 2.0 = 0.6
        assert!((stats.expected_edge - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_binned_fill_rate() {
        let mut bfr = BinnedFillRate::new();

        // 10 quotes at 5 bps spread, 3 filled
        for _ in 0..10 {
            bfr.record(5.0, false);
        }
        for _ in 0..3 {
            bfr.record(5.0, true);
        }

        // Bin [4,6) should have 13 total, 3 fills
        let rate = bfr.fill_rate_at(5.0).unwrap();
        assert!((rate - 3.0 / 13.0).abs() < 0.01);

        assert_eq!(bfr.total_quotes(), 13);
        assert_eq!(bfr.total_fills(), 3);
    }

    #[test]
    fn test_wrong_side_fill_not_resolved() {
        let mut tracker = QuoteOutcomeTracker::new();
        tracker.update_mid(100.0);

        tracker.register_quote(PendingQuote {
            timestamp_ms: 1000,
            half_spread_bps: 5.0,
            is_bid: true,
            state: make_state(100.0),
        });

        // Try to fill the ask side — should not match
        let resolved = tracker.on_fill(false, 2.0);
        assert!(!resolved);
        assert_eq!(tracker.pending_count(), 1);
    }

    #[test]
    fn test_capacity_eviction() {
        let mut tracker = QuoteOutcomeTracker::new();
        tracker.update_mid(100.0);

        // Fill beyond MAX_PENDING_QUOTES
        for i in 0..(MAX_PENDING_QUOTES + 5) {
            tracker.register_quote(PendingQuote {
                timestamp_ms: i as u64 * 1000,
                half_spread_bps: 5.0,
                is_bid: true,
                state: make_state(100.0),
            });
        }

        // Should have evicted 5 oldest as expired
        assert_eq!(tracker.pending_count(), MAX_PENDING_QUOTES);
        assert_eq!(tracker.total_outcomes(), 5);
    }
}
