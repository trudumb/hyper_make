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

use crate::market_maker::checkpoint::types::QuoteOutcomeCheckpoint;

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
    /// E[PnL] prediction at registration time (bps), for reconciliation against realized markout.
    pub epnl_at_registration: Option<f64>,
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

/// Posterior estimate of fill rate from Beta distribution.
#[derive(Debug, Clone, Copy)]
pub struct FillRateEstimate {
    /// Posterior mean = α/(α+β)
    pub mean: f64,
    /// Posterior variance = αβ/((α+β)²(α+β+1))
    pub variance: f64,
    /// Total observations in this bin
    pub n_observations: u64,
    /// Whether hierarchical blend is active (variance may be understated)
    pub blend_active: bool,
}

impl FillRateEstimate {
    /// Variance with 1.5x multiplier when hierarchical blend is active.
    /// Use this instead of raw `variance` — accounts for heuristic blending
    /// understating true uncertainty during the transition regime.
    pub fn variance_adjusted(&self) -> f64 {
        if self.blend_active {
            self.variance * 1.5
        } else {
            self.variance
        }
    }

    /// Standard deviation (adjusted for blend).
    pub fn std_adjusted(&self) -> f64 {
        self.variance_adjusted().sqrt()
    }
}

/// Bin for tracking fill rates at different spread levels using Beta posteriors.
#[derive(Debug, Clone)]
pub struct SpreadBin {
    /// Lower bound of bin (bps)
    pub lo_bps: f64,
    /// Upper bound of bin (bps)
    pub hi_bps: f64,
    /// Beta posterior alpha (pseudo-count of fills = prior + observed)
    pub alpha: f64,
    /// Beta posterior beta (pseudo-count of misses = prior + observed)
    pub beta: f64,
    /// Raw observed fills (for checkpoint/diagnostics, excludes prior)
    pub observed_fills: u64,
    /// Raw observed total (for checkpoint/diagnostics, excludes prior)
    pub observed_total: u64,
}

/// Shrinkage rate: fine bin accumulates ~50 obs before mostly ignoring coarse parent.
const HIERARCHICAL_TAU: f64 = 50.0;

/// Initial prior weight for hierarchical blending.
const HIERARCHICAL_PRIOR_WEIGHT: f64 = 5.0;

impl SpreadBin {
    /// Create a new bin with uniform Beta(1,1) prior.
    fn new(lo_bps: f64, hi_bps: f64) -> Self {
        Self {
            lo_bps,
            hi_bps,
            alpha: 1.0,
            beta: 1.0,
            observed_fills: 0,
            observed_total: 0,
        }
    }

    /// Record an observation: fill or miss.
    fn record(&mut self, filled: bool) {
        if filled {
            self.alpha += 1.0;
            self.observed_fills += 1;
        } else {
            self.beta += 1.0;
        }
        self.observed_total += 1;
    }

    /// Posterior mean of fill rate (own data only, no hierarchical blend).
    fn posterior_mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Compute fill rate estimate with hierarchical shrinkage toward coarse parent.
    fn fill_rate_estimate(&self, coarse_alpha: f64, coarse_beta: f64) -> FillRateEstimate {
        let n_fine = self.observed_total as f64;
        let w = HIERARCHICAL_PRIOR_WEIGHT / (1.0 + n_fine / HIERARCHICAL_TAU);
        let blend_active = w > 0.01; // blend is negligible below 1%

        let eff_alpha = self.alpha + w * coarse_alpha;
        let eff_beta = self.beta + w * coarse_beta;
        let total = eff_alpha + eff_beta;

        let mean = eff_alpha / total;
        let variance = (eff_alpha * eff_beta) / (total * total * (total + 1.0));

        FillRateEstimate {
            mean,
            variance,
            n_observations: self.observed_total,
            blend_active,
        }
    }

    /// Simple fill rate (for backward compat with all_rates).
    fn fill_rate(&self) -> f64 {
        self.posterior_mean()
    }
}

/// Coarse bin for hierarchical prior aggregation.
#[derive(Debug, Clone)]
struct CoarseBin {
    /// Lower bound of coarse bin (bps)
    lo_bps: f64,
    /// Upper bound of coarse bin (bps)
    hi_bps: f64,
    /// Beta posterior alpha (fills + prior)
    alpha: f64,
    /// Beta posterior beta (misses + prior)
    beta: f64,
    /// Indices of fine bins that belong to this coarse bin
    fine_indices: Vec<usize>,
}

impl CoarseBin {
    fn new(lo_bps: f64, hi_bps: f64) -> Self {
        Self {
            lo_bps,
            hi_bps,
            alpha: 1.0, // Uniform prior
            beta: 1.0,
            fine_indices: Vec::new(),
        }
    }

    fn record(&mut self, filled: bool) {
        if filled {
            self.alpha += 1.0;
        } else {
            self.beta += 1.0;
        }
    }
}

/// Tracks P(fill | spread_bin) with Beta posteriors and hierarchical shrinkage.
///
/// Two-layer hierarchy:
/// - **Coarse layer** (4 bins: `[0,5), [5,10), [10,20), [20,∞)`): aggregated posteriors
/// - **Fine layer** (8 bins): each bin's prior shrinks toward its coarse parent
///
/// No hard switching threshold — shrinkage is continuous via prior strength
/// that decays as fine-bin observations accumulate.
#[derive(Debug, Clone)]
pub struct BinnedFillRate {
    /// Fine bins (8 bins, same edges as before)
    pub bins: Vec<SpreadBin>,
    /// Coarse bins (4 bins) for hierarchical prior
    coarse_bins: Vec<CoarseBin>,
    /// Mapping: fine_bin_index -> coarse_bin_index
    fine_to_coarse: Vec<usize>,
}

impl BinnedFillRate {
    /// Create binned tracker with default bin edges and hierarchical structure.
    /// Fine bins: [0,2), [2,4), [4,6), [6,8), [8,10), [10,15), [15,20), [20,∞)
    /// Coarse bins: [0,5), [5,10), [10,20), [20,∞)
    pub fn new() -> Self {
        let fine_edges = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 1e9];
        let bins: Vec<SpreadBin> = fine_edges
            .windows(2)
            .map(|w| SpreadBin::new(w[0], w[1]))
            .collect();

        let coarse_edges = [(0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 1e9)];
        let mut coarse_bins: Vec<CoarseBin> = coarse_edges
            .iter()
            .map(|&(lo, hi)| CoarseBin::new(lo, hi))
            .collect();

        // Map fine bins to coarse parents
        let mut fine_to_coarse = Vec::with_capacity(bins.len());
        for (fine_idx, fine_bin) in bins.iter().enumerate() {
            let fine_mid = (fine_bin.lo_bps + fine_bin.hi_bps.min(100.0)) / 2.0;
            let coarse_idx = coarse_bins
                .iter()
                .position(|c| fine_mid >= c.lo_bps && fine_mid < c.hi_bps)
                .unwrap_or(coarse_bins.len() - 1);
            fine_to_coarse.push(coarse_idx);
            coarse_bins[coarse_idx].fine_indices.push(fine_idx);
        }

        Self {
            bins,
            coarse_bins,
            fine_to_coarse,
        }
    }

    /// Record a quote outcome in the appropriate spread bin.
    pub fn record(&mut self, spread_bps: f64, filled: bool) {
        for (fine_idx, bin) in self.bins.iter_mut().enumerate() {
            if spread_bps >= bin.lo_bps && spread_bps < bin.hi_bps {
                bin.record(filled);
                // Update coarse parent
                let coarse_idx = self.fine_to_coarse[fine_idx];
                self.coarse_bins[coarse_idx].record(filled);
                break;
            }
        }
    }

    /// Get fill rate estimate for a given spread — always returns a value.
    ///
    /// Uses Beta posterior with hierarchical shrinkage: weak prior gives
    /// uncertain but usable estimate even with 0 observations.
    pub fn fill_rate_at(&self, spread_bps: f64) -> FillRateEstimate {
        for (fine_idx, bin) in self.bins.iter().enumerate() {
            if spread_bps >= bin.lo_bps && spread_bps < bin.hi_bps {
                let coarse_idx = self.fine_to_coarse[fine_idx];
                let coarse = &self.coarse_bins[coarse_idx];
                return bin.fill_rate_estimate(coarse.alpha, coarse.beta);
            }
        }
        // Fallback: uniform prior
        FillRateEstimate {
            mean: 0.5,
            variance: 1.0 / 12.0,
            n_observations: 0,
            blend_active: true,
        }
    }

    /// Get fill rate as Option<f64> for backward compatibility.
    /// Returns None only when there are fewer than 5 observations.
    pub fn fill_rate_at_optional(&self, spread_bps: f64) -> Option<f64> {
        let est = self.fill_rate_at(spread_bps);
        if est.n_observations >= 5 {
            Some(est.mean)
        } else {
            None
        }
    }

    /// Get (spread_midpoint, fill_rate, sample_count) for all bins with data.
    pub fn all_rates(&self) -> Vec<(f64, f64, u64)> {
        self.bins
            .iter()
            .filter(|b| b.observed_total >= 5 && b.hi_bps.is_finite())
            .map(|b| ((b.lo_bps + b.hi_bps) / 2.0, b.fill_rate(), b.observed_total))
            .collect()
    }

    /// Total quotes tracked across all bins.
    pub fn total_quotes(&self) -> u64 {
        self.bins.iter().map(|b| b.observed_total).sum()
    }

    /// Total fills tracked across all bins.
    pub fn total_fills(&self) -> u64 {
        self.bins.iter().map(|b| b.observed_fills).sum()
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
    /// Running sum of (epnl_predicted - realized_edge) for bias calculation.
    epnl_error_sum: f64,
    /// Running sum of (epnl_predicted - realized_edge)^2 for RMSE.
    epnl_error_sq_sum: f64,
    /// Number of fills with E[PnL] predictions for reconciliation.
    epnl_reconciliation_count: u64,
}

impl QuoteOutcomeTracker {
    /// Create a new tracker.
    pub fn new() -> Self {
        Self {
            pending_quotes: VecDeque::with_capacity(MAX_PENDING_QUOTES),
            outcome_log: VecDeque::with_capacity(MAX_OUTCOME_LOG),
            fill_rate: BinnedFillRate::new(),
            last_mid: 0.0,
            epnl_error_sum: 0.0,
            epnl_error_sq_sum: 0.0,
            epnl_reconciliation_count: 0,
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
        let idx = self.pending_quotes.iter().rposition(|q| q.is_bid == is_bid);

        if let Some(idx) = idx {
            let quote = self.pending_quotes.remove(idx).unwrap();

            // WS7c: Reconcile E[PnL] prediction against realized markout
            if let Some(predicted) = quote.epnl_at_registration {
                let error = predicted - edge_bps;
                self.epnl_error_sum += error;
                self.epnl_error_sq_sum += error * error;
                self.epnl_reconciliation_count += 1;
            }

            let outcome = QuoteOutcome::Filled {
                edge_bps,
                state: quote.state,
                spread_bps: quote.half_spread_bps * 2.0,
            };
            self.fill_rate.record(quote.half_spread_bps * 2.0, true);
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
        self.fill_rate.record(quote.half_spread_bps * 2.0, false);
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

    /// E[PnL] prediction accuracy: (bias_bps, rmse_bps).
    /// Bias > 0 means predictions are optimistic, < 0 means pessimistic.
    /// Returns (0.0, 0.0) if no reconciliation data yet.
    pub fn epnl_prediction_accuracy(&self) -> (f64, f64) {
        if self.epnl_reconciliation_count == 0 {
            return (0.0, 0.0);
        }
        let n = self.epnl_reconciliation_count as f64;
        let bias = self.epnl_error_sum / n;
        let rmse = (self.epnl_error_sq_sum / n).sqrt();
        (bias, rmse)
    }

    /// Create a checkpoint of the fill rate bins.
    pub fn to_checkpoint(&self) -> QuoteOutcomeCheckpoint {
        QuoteOutcomeCheckpoint {
            bins: self
                .fill_rate
                .bins
                .iter()
                .map(|b| (b.lo_bps, b.hi_bps, b.observed_fills, b.observed_total))
                .collect(),
        }
    }

    /// Restore fill rate bins from a checkpoint.
    ///
    /// Handles backward compatibility: old checkpoints store raw (fills, total),
    /// which are migrated to Beta posteriors: alpha = fills + 1.0, beta = total - fills + 1.0.
    pub fn restore_from_checkpoint(&mut self, cp: &QuoteOutcomeCheckpoint) {
        if cp.bins.len() == self.fill_rate.bins.len() {
            for (bin, &(lo, hi, fills, total)) in self.fill_rate.bins.iter_mut().zip(cp.bins.iter())
            {
                bin.lo_bps = lo;
                bin.hi_bps = hi;
                bin.observed_fills = fills;
                bin.observed_total = total;
                // Migrate from count format to Beta posterior:
                // alpha = prior(1) + observed fills, beta = prior(1) + observed misses
                bin.alpha = 1.0 + fills as f64;
                bin.beta = 1.0 + (total - fills) as f64;
            }
            // Rebuild coarse bins from restored fine bins
            for coarse in &mut self.fill_rate.coarse_bins {
                coarse.alpha = 1.0;
                coarse.beta = 1.0;
                for &fine_idx in &coarse.fine_indices {
                    let fine = &self.fill_rate.bins[fine_idx];
                    coarse.alpha += fine.observed_fills as f64;
                    coarse.beta += (fine.observed_total - fine.observed_fills) as f64;
                }
            }
        }
        // If bin count doesn't match (format change), start fresh — safe default
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
        let fill_rate_est = self.fill_rate.fill_rate_at(spread_bps);
        let mean_edge = self.mean_edge_at_spread(spread_bps);
        fill_rate_est.mean * mean_edge
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
            epnl_at_registration: None,
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
            epnl_at_registration: None,
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
                epnl_at_registration: None,
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
        // Beta posterior: alpha = 1 + 3, beta = 1 + 10, plus hierarchical blend
        let est = bfr.fill_rate_at(5.0);
        assert_eq!(est.n_observations, 13);
        // Allow tolerance for prior + hierarchical influence
        assert!(
            (est.mean - 3.0 / 13.0).abs() < 0.1,
            "Mean should be near 3/13. Got: {}",
            est.mean
        );

        assert_eq!(bfr.total_quotes(), 13);
        assert_eq!(bfr.total_fills(), 3);
    }

    #[test]
    fn test_beta_posterior_with_zero_observations() {
        let bfr = BinnedFillRate::new();

        // With no data, should return prior mean (~0.5) with high variance
        let est = bfr.fill_rate_at(5.0);
        assert_eq!(est.n_observations, 0);
        assert!(
            (est.mean - 0.5).abs() < 0.1,
            "Empty bin should have mean ~0.5. Got: {}",
            est.mean
        );
        assert!(
            est.variance > 0.01,
            "Empty bin should have high variance. Got: {}",
            est.variance
        );
        assert!(
            est.blend_active,
            "Blend should be active with zero observations"
        );
    }

    #[test]
    fn test_beta_posterior_converges() {
        let mut bfr = BinnedFillRate::new();

        // 100 fills out of 200 → mean should converge to ≈ 0.5
        for _ in 0..100 {
            bfr.record(5.0, true);
        }
        for _ in 0..100 {
            bfr.record(5.0, false);
        }

        let est = bfr.fill_rate_at(5.0);
        assert!(
            (est.mean - 0.5).abs() < 0.05,
            "200 obs should converge to ~0.5. Got: {}",
            est.mean
        );
        assert!(
            est.variance < 0.005,
            "200 obs should have tight variance. Got: {}",
            est.variance
        );
    }

    #[test]
    fn test_hierarchical_shrinkage() {
        let mut bfr = BinnedFillRate::new();

        // Populate coarse parent [0,5) with data in [0,2) bin
        for _ in 0..200 {
            bfr.record(1.0, true);
        }
        for _ in 0..200 {
            bfr.record(1.0, false);
        }

        // Check [2,4) bin with only 3 fills (sparse)
        for _ in 0..3 {
            bfr.record(3.0, true);
        }
        let sparse_est = bfr.fill_rate_at(3.0);
        // Should be pulled below 1.0 by coarse prior (~0.5)
        assert!(
            sparse_est.mean < 0.95,
            "3 obs should shrink toward parent. Got: {}",
            sparse_est.mean
        );
        assert!(
            sparse_est.blend_active,
            "Blend should be active with 3 observations"
        );

        // Add 1000 more observations to [2,4) — should mostly ignore parent
        for _ in 0..1000 {
            bfr.record(3.0, true);
        }
        let dense_est = bfr.fill_rate_at(3.0);
        assert!(
            dense_est.mean > 0.8,
            "Many obs should mostly ignore parent. Got: {}",
            dense_est.mean
        );
    }

    #[test]
    fn test_fill_rate_always_returns_value() {
        let bfr = BinnedFillRate::new();

        for spread in [0.5, 3.0, 5.0, 7.0, 9.0, 12.0, 18.0, 25.0, 100.0] {
            let est = bfr.fill_rate_at(spread);
            assert!(
                est.mean >= 0.0 && est.mean <= 1.0,
                "Mean must be in [0,1]. Got: {} at {} bps",
                est.mean,
                spread
            );
            assert!(
                est.variance >= 0.0,
                "Variance must be non-negative at {} bps",
                spread
            );
        }
    }

    #[test]
    fn test_variance_adjusted_multiplier() {
        let with_blend = FillRateEstimate {
            mean: 0.5,
            variance: 0.10,
            n_observations: 5,
            blend_active: true,
        };
        assert!(
            (with_blend.variance_adjusted() - 0.15).abs() < 1e-9,
            "1.5x when blend active"
        );

        let without_blend = FillRateEstimate {
            mean: 0.5,
            variance: 0.10,
            n_observations: 100,
            blend_active: false,
        };
        assert!(
            (without_blend.variance_adjusted() - 0.10).abs() < 1e-9,
            "1.0x when no blend"
        );
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
            epnl_at_registration: None,
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
                epnl_at_registration: None,
            });
        }

        // Should have evicted 5 oldest as expired
        assert_eq!(tracker.pending_count(), MAX_PENDING_QUOTES);
        assert_eq!(tracker.total_outcomes(), 5);
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut tracker = QuoteOutcomeTracker::new();
        tracker.update_mid(100.0);

        // Record some data
        for _ in 0..10 {
            tracker.register_quote(PendingQuote {
                timestamp_ms: 1000,
                half_spread_bps: 3.0,
                is_bid: true,
                state: make_state(100.0),
                epnl_at_registration: None,
            });
        }
        for _ in 0..3 {
            tracker.on_fill(true, 1.5);
        }
        tracker.expire_old_quotes(100_000);

        // Checkpoint
        let cp = tracker.to_checkpoint();
        assert_eq!(cp.bins.len(), 8); // 8 default bins

        // Restore into fresh tracker
        let mut restored = QuoteOutcomeTracker::new();
        restored.restore_from_checkpoint(&cp);

        // Verify fill rates match
        let orig_est = tracker.fill_rate().fill_rate_at(3.0);
        let restored_est = restored.fill_rate().fill_rate_at(3.0);
        assert!(
            (orig_est.mean - restored_est.mean).abs() < 0.01,
            "Restored mean should match original. Got: {} vs {}",
            restored_est.mean,
            orig_est.mean
        );
        assert_eq!(orig_est.n_observations, restored_est.n_observations);
    }

    #[test]
    fn test_epnl_reconciliation() {
        let mut tracker = QuoteOutcomeTracker::new();
        tracker.update_mid(100.0);

        // Register quote with E[PnL] prediction of 2.0 bps
        tracker.register_quote(PendingQuote {
            timestamp_ms: 1000,
            half_spread_bps: 5.0,
            is_bid: true,
            state: make_state(100.0),
            epnl_at_registration: Some(2.0),
        });

        // Fill with realized edge of 1.5 bps → error = 2.0 - 1.5 = 0.5
        tracker.on_fill(true, 1.5);

        let (bias, rmse) = tracker.epnl_prediction_accuracy();
        assert!(
            (bias - 0.5).abs() < 0.01,
            "Bias should be +0.5 (optimistic). Got: {}",
            bias
        );
        assert!(
            (rmse - 0.5).abs() < 0.01,
            "RMSE should be 0.5. Got: {}",
            rmse
        );

        // Register another with None — should not affect reconciliation
        tracker.register_quote(PendingQuote {
            timestamp_ms: 2000,
            half_spread_bps: 5.0,
            is_bid: true,
            state: make_state(100.0),
            epnl_at_registration: None,
        });
        tracker.on_fill(true, 3.0);

        // Still only 1 reconciliation data point
        let (bias2, _) = tracker.epnl_prediction_accuracy();
        assert!((bias2 - 0.5).abs() < 0.01, "Bias unchanged. Got: {}", bias2);
    }

    #[test]
    fn test_checkpoint_migration_from_counts() {
        // Simulate an old checkpoint with raw (fills=10, total=20)
        let old_cp = QuoteOutcomeCheckpoint {
            bins: vec![
                (0.0, 2.0, 10, 20),
                (2.0, 4.0, 5, 15),
                (4.0, 6.0, 0, 0),
                (6.0, 8.0, 0, 0),
                (8.0, 10.0, 0, 0),
                (10.0, 15.0, 0, 0),
                (15.0, 20.0, 0, 0),
                (20.0, 1e9, 0, 0),
            ],
        };

        let mut tracker = QuoteOutcomeTracker::new();
        tracker.restore_from_checkpoint(&old_cp);

        // Bin [0,2): fills=10, total=20 → alpha = 1+10 = 11, beta = 1+10 = 11
        // Posterior mean ≈ 0.5 (with small hierarchical shift)
        let est = tracker.fill_rate().fill_rate_at(1.0);
        assert_eq!(est.n_observations, 20);
        assert!(
            (est.mean - 0.5).abs() < 0.1,
            "Migrated bin should have mean ~0.5. Got: {}",
            est.mean
        );

        // Bin [2,4): fills=5, total=15 → alpha = 1+5 = 6, beta = 1+10 = 11
        // Posterior mean ≈ 5/15 ≈ 0.33 (shifted by hierarchy)
        let est2 = tracker.fill_rate().fill_rate_at(3.0);
        assert_eq!(est2.n_observations, 15);
        assert!(
            (est2.mean - 0.33).abs() < 0.15,
            "Migrated bin should have mean ~0.33. Got: {}",
            est2.mean
        );
    }
}
