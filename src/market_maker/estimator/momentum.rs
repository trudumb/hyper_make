//! Momentum detector for directional flow analysis.
//!
//! Detects falling knife and rising knife patterns from signed VWAP returns.

use std::collections::VecDeque;

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
    pub(super) fn new(window_ms: u64) -> Self {
        Self {
            returns: VecDeque::with_capacity(100),
            window_ms,
        }
    }

    /// Add a new VWAP-based return
    pub(super) fn on_bucket(&mut self, end_time_ms: u64, log_return: f64) {
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
    pub(super) fn momentum_bps(&self, now_ms: u64) -> f64 {
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
    pub(super) fn falling_knife_score(&self, now_ms: u64) -> f64 {
        let momentum = self.momentum_bps(now_ms);

        // Only trigger on negative momentum
        if momentum >= 0.0 {
            return 0.0;
        }

        // Score: -20 bps = 1.0, -40 bps = 2.0, etc.
        (momentum.abs() / 20.0).clamp(0.0, 3.0)
    }

    /// Rising knife score (for protecting asks during pumps)
    pub(super) fn rising_knife_score(&self, now_ms: u64) -> f64 {
        let momentum = self.momentum_bps(now_ms);

        if momentum <= 0.0 {
            return 0.0;
        }

        (momentum / 20.0).clamp(0.0, 3.0)
    }
}
