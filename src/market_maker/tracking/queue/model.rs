//! Calibrated queue model for fill probability estimation.
//!
//! Estimates queue parameters from market data using MLE:
//! - volume_at_touch_rate: Calibrated from trade tape (volume executed at best)
//! - cancel_rate: Calibrated from book dynamics (depth reduction not from fills)

use std::collections::VecDeque;

use super::normal_cdf;

/// Calibrated queue model for fill probability estimation.
///
/// Key formulas:
/// - P(touch) = 2Φ(-δ/(σ√T))  (Black-Scholes barrier touching probability)
/// - P(execute|touch) = exp(-queue_ahead / (volume_rate × horizon))
#[derive(Debug, Clone)]
pub struct CalibratedQueueModel {
    /// Calibrated volume at touch per second (from trade tape)
    volume_at_touch_rate: f64,
    /// Calibrated cancel rate (fraction of queue cancelled per second)
    cancel_rate: f64,
    /// History of touch volumes: (timestamp_ms, volume)
    touch_volume_history: VecDeque<(u64, f64)>,
    /// History of cancel observations: (timestamp_ms, queue_delta_not_from_fills)
    cancel_history: VecDeque<(u64, f64)>,
    /// History window (ms)
    window_ms: u64,
    /// Minimum observations for calibration
    min_observations: usize,
    /// EWMA alpha for online updates
    alpha: f64,
    /// Last recorded best bid/ask depth
    last_best_bid_depth: f64,
    last_best_ask_depth: f64,
    last_update_ms: u64,
}

impl CalibratedQueueModel {
    /// Create a new calibrated queue model.
    pub fn new(window_ms: u64, min_observations: usize) -> Self {
        Self {
            volume_at_touch_rate: 1.0, // Conservative default
            cancel_rate: 0.2,          // 20% per second default
            touch_volume_history: VecDeque::with_capacity(500),
            cancel_history: VecDeque::with_capacity(500),
            window_ms,
            min_observations,
            alpha: 0.05,
            last_best_bid_depth: 0.0,
            last_best_ask_depth: 0.0,
            last_update_ms: 0,
        }
    }

    /// Create with default parameters.
    pub fn default_config() -> Self {
        Self::new(300_000, 50) // 5 minute window, 50 min observations
    }

    /// Record a trade at the touch level.
    ///
    /// This is used to calibrate volume_at_touch_rate.
    /// Only call for trades that execute at best bid or ask.
    pub fn on_touch_trade(&mut self, timestamp_ms: u64, size: f64) {
        self.touch_volume_history.push_back((timestamp_ms, size));
        self.expire_old_entries(timestamp_ms);
        self.calibrate_volume_rate(timestamp_ms);
    }

    /// Record book depth update.
    ///
    /// Used to calibrate cancel_rate by observing queue shrinkage
    /// that is not explained by fills.
    pub fn on_book_update(
        &mut self,
        timestamp_ms: u64,
        best_bid_depth: f64,
        best_ask_depth: f64,
        fill_volume_since_last: f64,
    ) {
        if self.last_update_ms == 0 {
            // First update, just store
            self.last_best_bid_depth = best_bid_depth;
            self.last_best_ask_depth = best_ask_depth;
            self.last_update_ms = timestamp_ms;
            return;
        }

        // Calculate queue changes not explained by fills
        let bid_delta = self.last_best_bid_depth - best_bid_depth;
        let ask_delta = self.last_best_ask_depth - best_ask_depth;

        // Positive delta after removing fills = cancellations
        let cancel_bid = (bid_delta - fill_volume_since_last / 2.0).max(0.0);
        let cancel_ask = (ask_delta - fill_volume_since_last / 2.0).max(0.0);

        if cancel_bid > 0.0 || cancel_ask > 0.0 {
            self.cancel_history
                .push_back((timestamp_ms, cancel_bid + cancel_ask));
        }

        self.last_best_bid_depth = best_bid_depth;
        self.last_best_ask_depth = best_ask_depth;
        self.last_update_ms = timestamp_ms;

        self.expire_old_entries(timestamp_ms);
        self.calibrate_cancel_rate(timestamp_ms);
    }

    /// Calibrate volume rate from touch trade history.
    fn calibrate_volume_rate(&mut self, timestamp_ms: u64) {
        if self.touch_volume_history.len() < self.min_observations {
            return;
        }

        // Total volume and time span
        let total_volume: f64 = self.touch_volume_history.iter().map(|(_, v)| v).sum();
        let oldest = self
            .touch_volume_history
            .front()
            .map(|(t, _)| *t)
            .unwrap_or(timestamp_ms);
        let span_secs = (timestamp_ms.saturating_sub(oldest)) as f64 / 1000.0;

        if span_secs > 0.0 {
            let new_rate = total_volume / span_secs;
            // EWMA update
            self.volume_at_touch_rate =
                self.alpha * new_rate + (1.0 - self.alpha) * self.volume_at_touch_rate;
        }
    }

    /// Calibrate cancel rate from cancel history.
    fn calibrate_cancel_rate(&mut self, timestamp_ms: u64) {
        if self.cancel_history.len() < self.min_observations {
            return;
        }

        // Average cancel volume per second
        let total_cancels: f64 = self.cancel_history.iter().map(|(_, v)| v).sum();
        let oldest = self
            .cancel_history
            .front()
            .map(|(t, _)| *t)
            .unwrap_or(timestamp_ms);
        let span_secs = (timestamp_ms.saturating_sub(oldest)) as f64 / 1000.0;

        if span_secs > 0.0 && self.last_best_bid_depth + self.last_best_ask_depth > 0.0 {
            let avg_depth = (self.last_best_bid_depth + self.last_best_ask_depth) / 2.0;
            // Cancel rate = fraction of queue cancelled per second
            let new_rate = (total_cancels / span_secs) / avg_depth.max(1e-9);
            // EWMA update, clamp to reasonable range
            self.cancel_rate =
                (self.alpha * new_rate + (1.0 - self.alpha) * self.cancel_rate).clamp(0.05, 0.8);
        }
    }

    /// Expire old entries from history.
    fn expire_old_entries(&mut self, timestamp_ms: u64) {
        let cutoff = timestamp_ms.saturating_sub(self.window_ms);

        while self
            .touch_volume_history
            .front()
            .map(|(t, _)| *t < cutoff)
            .unwrap_or(false)
        {
            self.touch_volume_history.pop_front();
        }

        while self
            .cancel_history
            .front()
            .map(|(t, _)| *t < cutoff)
            .unwrap_or(false)
        {
            self.cancel_history.pop_front();
        }
    }

    /// Probability that price will touch a level δ away within horizon.
    ///
    /// Uses Black-Scholes barrier probability: P(touch) = 2Φ(-δ/(σ√T))
    ///
    /// # Arguments
    /// - `delta`: Distance from best (in price units, already converted to fraction)
    /// - `sigma`: Volatility per second
    /// - `horizon_secs`: Time horizon in seconds
    pub fn p_touch(&self, delta: f64, sigma: f64, horizon_secs: f64) -> f64 {
        if delta <= 0.0 {
            return 1.0; // Already at or through level
        }

        let sigma_sqrt_t = sigma * horizon_secs.sqrt();
        if sigma_sqrt_t < 1e-12 {
            return 0.0;
        }

        let z = -delta / sigma_sqrt_t;
        (2.0 * normal_cdf(z)).min(1.0)
    }

    /// Probability of execution given price touches our level.
    ///
    /// P(execute|touch) = exp(-queue_ahead / expected_volume_at_touch)
    ///
    /// # Arguments
    /// - `queue_ahead`: Depth ahead of us in the queue
    /// - `horizon_secs`: Time horizon in seconds
    pub fn p_execute_given_touch(&self, queue_ahead: f64, horizon_secs: f64) -> f64 {
        let expected_volume = self.volume_at_touch_rate * horizon_secs;
        if expected_volume <= 0.0 {
            return 0.0;
        }

        (-queue_ahead / expected_volume).exp().min(1.0)
    }

    /// Combined fill probability.
    ///
    /// P(fill) = P(touch) × P(execute|touch)
    pub fn p_fill(&self, delta: f64, sigma: f64, queue_ahead: f64, horizon_secs: f64) -> f64 {
        let p_t = self.p_touch(delta, sigma, horizon_secs);
        let p_e = self.p_execute_given_touch(queue_ahead, horizon_secs);
        p_t * p_e
    }

    /// Get expected queue decay over time (for position estimation).
    ///
    /// Returns the expected queue position after `dt` seconds.
    pub fn expected_queue_decay(&self, current_queue: f64, dt_secs: f64) -> f64 {
        current_queue * (-self.cancel_rate * dt_secs).exp()
    }

    // === Getters ===

    /// Get calibrated volume at touch rate (units per second).
    pub fn volume_at_touch_rate(&self) -> f64 {
        self.volume_at_touch_rate
    }

    /// Get calibrated cancel rate (fraction per second).
    pub fn cancel_rate(&self) -> f64 {
        self.cancel_rate
    }

    /// Check if model is calibrated (has enough data).
    pub fn is_calibrated(&self) -> bool {
        self.touch_volume_history.len() >= self.min_observations
    }

    /// Get number of touch volume observations.
    pub fn touch_observation_count(&self) -> usize {
        self.touch_volume_history.len()
    }
}
