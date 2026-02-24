//! Momentum detection and trade flow tracking.
//!
//! - MomentumDetector: Directional momentum from signed VWAP returns
//! - MomentumModel: Probabilistic continuation/reversal prediction
//! - TradeFlowTracker: Buy/sell aggressor imbalance tracking

use std::collections::VecDeque;

// ============================================================================
// Momentum Detector
// ============================================================================

/// Detects directional momentum from signed VWAP returns.
///
/// Tracks signed (not absolute) returns to detect falling/rising knife patterns.
#[derive(Debug)]
pub(crate) struct MomentumDetector {
    /// Recent (timestamp_ms, log_return) pairs
    returns: VecDeque<(u64, f64)>,
    /// Window for momentum calculation (ms)
    window_ms: u64,
}

impl MomentumDetector {
    pub(crate) fn new(window_ms: u64) -> Self {
        Self {
            returns: VecDeque::with_capacity(100),
            window_ms,
        }
    }

    /// Add a new VWAP-based return
    pub(crate) fn on_bucket(&mut self, end_time_ms: u64, log_return: f64) {
        self.returns.push_back((end_time_ms, log_return));

        // Expire old returns (keep 2x window for safety)
        let cutoff = end_time_ms.saturating_sub(self.window_ms * 2);
        while self
            .returns
            .front()
            .map(|(t, _)| *t < cutoff)
            .unwrap_or(false)
        {
            self.returns.pop_front();
        }
    }

    /// Signed momentum in bps over the configured window
    pub(crate) fn momentum_bps(&self, now_ms: u64) -> f64 {
        let cutoff = now_ms.saturating_sub(self.window_ms);
        let sum: f64 = self
            .returns
            .iter()
            .filter(|(t, _)| *t >= cutoff)
            .map(|(_, r)| r)
            .sum();
        sum * 10_000.0 // Convert to bps
    }

    /// Falling knife score: 0 = normal, 1+ = severe downward momentum
    pub(crate) fn falling_knife_score(&self, now_ms: u64) -> f64 {
        let momentum = self.momentum_bps(now_ms);

        // Only trigger on negative momentum
        if momentum >= 0.0 {
            return 0.0;
        }

        // Score: -20 bps = 1.0, -40 bps = 2.0, etc.
        (momentum.abs() / 20.0).clamp(0.0, 3.0)
    }

    /// Rising knife score (for protecting asks during pumps)
    pub(crate) fn rising_knife_score(&self, now_ms: u64) -> f64 {
        let momentum = self.momentum_bps(now_ms);

        if momentum <= 0.0 {
            return 0.0;
        }

        (momentum / 20.0).clamp(0.0, 3.0)
    }
}

// ============================================================================
// Probabilistic Momentum Model (First Principles Gap 10)
// ============================================================================

/// Probabilistic momentum model for continuation/reversal prediction.
///
/// Replaces heuristic knife scores with Bayesian probability estimates
/// of momentum continuation.
#[derive(Debug)]
pub struct MomentumModel {
    /// Prior probability of momentum continuation
    prior_continuation: f64,
    /// Likelihood ratio observations: (timestamp_ms, momentum_bps, continued)
    observations: VecDeque<(u64, f64, bool)>,
    /// Learned continuation probability by momentum magnitude
    continuation_by_magnitude: [f64; 10], // Buckets: 0-10, 10-20, ..., 90+ bps
    /// Observation counts per bucket
    counts_by_magnitude: [usize; 10],
    /// Window for observations (ms)
    window_ms: u64,
    /// Minimum observations per bucket
    min_observations: usize,
    /// EWMA alpha for updates
    alpha: f64,
}

impl MomentumModel {
    /// Create a new momentum model.
    pub(crate) fn new(window_ms: u64, alpha: f64) -> Self {
        Self {
            prior_continuation: 0.5, // 50% prior
            observations: VecDeque::with_capacity(1000),
            continuation_by_magnitude: [0.5; 10], // Start at 50%
            counts_by_magnitude: [0; 10],
            window_ms,
            min_observations: 10,
            alpha,
        }
    }

    /// Create with default parameters.
    pub(crate) fn default_config() -> Self {
        Self::new(300_000, 0.1) // 5 minute window
    }

    /// Record an observation of momentum and whether it continued.
    ///
    /// # Arguments
    /// - `timestamp_ms`: Current timestamp
    /// - `momentum_bps`: Momentum in basis points (can be negative)
    /// - `continued`: Whether the momentum continued (same sign return)
    pub(crate) fn record_observation(
        &mut self,
        timestamp_ms: u64,
        momentum_bps: f64,
        continued: bool,
    ) {
        self.observations
            .push_back((timestamp_ms, momentum_bps, continued));

        // Update bucket statistics
        let bucket = self.magnitude_to_bucket(momentum_bps.abs());
        self.counts_by_magnitude[bucket] += 1;

        let obs = if continued { 1.0 } else { 0.0 };
        self.continuation_by_magnitude[bucket] =
            self.alpha * obs + (1.0 - self.alpha) * self.continuation_by_magnitude[bucket];

        // Expire old observations
        let cutoff = timestamp_ms.saturating_sub(self.window_ms);
        while self
            .observations
            .front()
            .map(|(t, _, _)| *t < cutoff)
            .unwrap_or(false)
        {
            self.observations.pop_front();
        }
    }

    /// Map momentum magnitude to bucket index.
    fn magnitude_to_bucket(&self, abs_momentum_bps: f64) -> usize {
        ((abs_momentum_bps / 10.0) as usize).min(9)
    }

    /// Get probability of momentum continuation.
    ///
    /// Returns P(next_return has same sign as momentum).
    pub(crate) fn continuation_probability(&self, momentum_bps: f64) -> f64 {
        let bucket = self.magnitude_to_bucket(momentum_bps.abs());

        // Use learned probability if we have enough data
        if self.counts_by_magnitude[bucket] >= self.min_observations {
            self.continuation_by_magnitude[bucket]
        } else {
            self.prior_continuation
        }
    }

    /// Get bid protection factor based on momentum.
    ///
    /// Returns multiplier > 1 if we should protect bids (falling market).
    pub(crate) fn bid_protection_factor(&self, momentum_bps: f64) -> f64 {
        if momentum_bps >= 0.0 {
            return 1.0; // Not falling, no protection needed
        }

        let p_continue = self.continuation_probability(momentum_bps);
        let magnitude_factor = (momentum_bps.abs() / 50.0).min(1.0); // Scale by magnitude

        // Protection factor: 1.0 to 2.0 based on continuation prob and magnitude
        1.0 + p_continue * magnitude_factor
    }

    /// Get ask protection factor based on momentum.
    ///
    /// Returns multiplier > 1 if we should protect asks (rising market).
    pub(crate) fn ask_protection_factor(&self, momentum_bps: f64) -> f64 {
        if momentum_bps <= 0.0 {
            return 1.0; // Not rising, no protection needed
        }

        let p_continue = self.continuation_probability(momentum_bps);
        let magnitude_factor = (momentum_bps.abs() / 50.0).min(1.0);

        1.0 + p_continue * magnitude_factor
    }

    /// Get overall momentum strength [0, 1].
    pub(crate) fn momentum_strength(&self, momentum_bps: f64) -> f64 {
        let p_continue = self.continuation_probability(momentum_bps);
        let magnitude = (momentum_bps.abs() / 100.0).min(1.0);

        p_continue * magnitude
    }

    /// Check if model is calibrated.
    pub(crate) fn is_calibrated(&self) -> bool {
        self.observations.len() >= self.min_observations * 3
    }

    // === Checkpoint persistence ===

    /// Extract learned state for checkpoint persistence.
    ///
    /// The observation VecDeque is NOT persisted — it's a rolling window.
    /// The per-magnitude continuation probabilities are the valuable state.
    pub(crate) fn to_checkpoint(&self) -> crate::market_maker::checkpoint::MomentumCheckpoint {
        crate::market_maker::checkpoint::MomentumCheckpoint {
            continuation_by_magnitude: self.continuation_by_magnitude,
            counts_by_magnitude: self.counts_by_magnitude,
            prior_continuation: self.prior_continuation,
        }
    }

    /// Restore learned state from a checkpoint.
    pub(crate) fn restore_checkpoint(
        &mut self,
        cp: &crate::market_maker::checkpoint::MomentumCheckpoint,
    ) {
        self.continuation_by_magnitude = cp.continuation_by_magnitude;
        self.counts_by_magnitude = cp.counts_by_magnitude;
        self.prior_continuation = cp.prior_continuation;
    }
}

// ============================================================================
// Trade Flow Tracker (Buy/Sell Imbalance)
// ============================================================================

/// Tracks buy vs sell aggressor imbalance from trade tape.
///
/// Uses the trade side field from Hyperliquid ("B" = buy aggressor, "S" = sell aggressor)
/// to detect directional order flow before it shows up in price.
#[derive(Debug)]
pub(crate) struct TradeFlowTracker {
    /// (timestamp_ms, signed_volume): positive = buy aggressor
    trades: VecDeque<(u64, f64)>,
    /// Rolling window (ms)
    window_ms: u64,
    /// EWMA smoothed imbalance
    ewma_imbalance: f64,
    /// EWMA alpha
    alpha: f64,
}

impl TradeFlowTracker {
    pub(crate) fn new(window_ms: u64, alpha: f64) -> Self {
        Self {
            trades: VecDeque::with_capacity(500),
            window_ms,
            ewma_imbalance: 0.0,
            alpha,
        }
    }

    /// Add a trade from the tape.
    /// is_buy_aggressor: true if buyer was taker (lifted the ask)
    pub(crate) fn on_trade(&mut self, timestamp_ms: u64, size: f64, is_buy_aggressor: bool) {
        let signed = if is_buy_aggressor { size } else { -size };
        self.trades.push_back((timestamp_ms, signed));

        // Expire old trades
        let cutoff = timestamp_ms.saturating_sub(self.window_ms);
        while self
            .trades
            .front()
            .map(|(t, _)| *t < cutoff)
            .unwrap_or(false)
        {
            self.trades.pop_front();
        }

        // Update EWMA
        let instant = self.compute_instant_imbalance();
        self.ewma_imbalance = self.alpha * instant + (1.0 - self.alpha) * self.ewma_imbalance;
    }

    /// Compute instantaneous imbalance: (buy - sell) / total
    fn compute_instant_imbalance(&self) -> f64 {
        let (buy_vol, sell_vol) =
            self.trades.iter().fold(
                (0.0, 0.0),
                |(b, s), (_, v)| {
                    if *v > 0.0 {
                        (b + v, s)
                    } else {
                        (b, s - v)
                    }
                },
            );
        let total = buy_vol + sell_vol;
        if total < 1e-12 {
            0.0
        } else {
            (buy_vol - sell_vol) / total
        }
    }

    /// Smoothed flow imbalance [-1, 1]
    /// Negative = sell pressure, Positive = buy pressure
    pub(crate) fn imbalance(&self) -> f64 {
        self.ewma_imbalance.clamp(-1.0, 1.0)
    }

    /// Is there dominant selling (for bid protection)?
    #[allow(dead_code)]
    pub(crate) fn is_sell_pressure(&self) -> bool {
        self.ewma_imbalance < -0.25
    }

    /// Is there dominant buying (for ask protection)?
    #[allow(dead_code)]
    pub(crate) fn is_buy_pressure(&self) -> bool {
        self.ewma_imbalance > 0.25
    }
}
