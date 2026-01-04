//! Tick-Based EWMA Module
//!
//! Fixes the fundamental timing bug in EWMA calculations where half-life
//! was specified in wall-clock seconds but the volume clock produces
//! irregular time intervals between observations.
//!
//! This module provides two EWMA variants:
//! - `TickEWMA`: Pure tick-based, half-life in number of observations
//! - `HybridEWMA`: Combines tick decay with time-based decay for stale data

// Allow dead code since this is V2 infrastructure being built incrementally
#![allow(dead_code)]

/// Pure tick-based EWMA where half-life is measured in number of observations.
///
/// When using volume-clock sampling, each "tick" represents a fixed volume bucket,
/// so half-life in ticks gives consistent economic weighting regardless of
/// wall-clock timing variations.
#[derive(Debug, Clone)]
pub(crate) struct TickEWMA {
    value: f64,
    half_life_ticks: f64,
    alpha: f64,
    tick_count: usize,
    initialized: bool,
}

impl TickEWMA {
    /// Create a new tick-based EWMA.
    ///
    /// # Arguments
    /// * `half_life_ticks` - Number of ticks for value to decay by 50%
    /// * `initial` - Initial value (used if `initialize_on_first` is false)
    pub(crate) fn new(half_life_ticks: f64, initial: f64) -> Self {
        // α such that (1-α)^half_life = 0.5
        // => 1-α = 0.5^(1/half_life)
        // => α = 1 - 2^(-1/half_life)
        let alpha = 1.0 - 2.0_f64.powf(-1.0 / half_life_ticks);
        Self {
            value: initial,
            half_life_ticks,
            alpha,
            tick_count: 0,
            initialized: initial != 0.0,
        }
    }

    /// Create with delayed initialization (first observation becomes the value)
    pub(crate) fn new_uninitialized(half_life_ticks: f64) -> Self {
        Self {
            value: 0.0,
            half_life_ticks,
            alpha: 1.0 - 2.0_f64.powf(-1.0 / half_life_ticks),
            tick_count: 0,
            initialized: false,
        }
    }

    /// Update with a new observation
    pub(crate) fn update(&mut self, observation: f64) {
        if !self.initialized {
            self.value = observation;
            self.initialized = true;
        } else {
            self.value = self.alpha * observation + (1.0 - self.alpha) * self.value;
        }
        self.tick_count += 1;
    }

    /// Get current EWMA value
    #[inline]
    pub(crate) fn value(&self) -> f64 {
        self.value
    }

    /// Get number of observations processed
    #[inline]
    pub(crate) fn tick_count(&self) -> usize {
        self.tick_count
    }

    /// Check if initialized with at least one observation
    #[inline]
    pub(crate) fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get the decay factor (1 - alpha)
    #[inline]
    pub(crate) fn decay_factor(&self) -> f64 {
        1.0 - self.alpha
    }

    /// Get the half-life in ticks
    #[inline]
    pub(crate) fn half_life(&self) -> f64 {
        self.half_life_ticks
    }

    /// Reset to uninitialized state
    pub(crate) fn reset(&mut self) {
        self.value = 0.0;
        self.tick_count = 0;
        self.initialized = false;
    }
}

/// Hybrid EWMA combining tick-based updates with time-based decay.
///
/// This handles the case where we want:
/// 1. Normal updates to use tick-based half-life (volume-normalized)
/// 2. Stale data (long gaps between updates) to decay toward a baseline
///
/// Useful for parameters like kappa where we want fresh observations
/// to dominate but also want the estimate to decay if no data arrives.
#[derive(Debug, Clone)]
pub(crate) struct HybridEWMA {
    value: f64,
    tick_alpha: f64,
    time_half_life_ms: u64,
    last_update_ms: u64,
    baseline: f64,
    tick_count: usize,
    initialized: bool,
}

impl HybridEWMA {
    /// Create a new hybrid EWMA.
    ///
    /// # Arguments
    /// * `tick_half_life` - Half-life in number of observations
    /// * `time_half_life_ms` - Half-life in milliseconds for time decay
    /// * `baseline` - Value to decay toward when stale
    pub(crate) fn new(tick_half_life: f64, time_half_life_ms: u64, baseline: f64) -> Self {
        Self {
            value: baseline,
            tick_alpha: 1.0 - 2.0_f64.powf(-1.0 / tick_half_life),
            time_half_life_ms,
            last_update_ms: 0,
            baseline,
            tick_count: 0,
            initialized: false,
        }
    }

    /// Update with a new observation and timestamp
    pub(crate) fn update(&mut self, observation: f64, timestamp_ms: u64) {
        if !self.initialized {
            self.value = observation;
            self.last_update_ms = timestamp_ms;
            self.initialized = true;
            self.tick_count = 1;
            return;
        }

        // First apply time-based decay toward baseline
        let elapsed = timestamp_ms.saturating_sub(self.last_update_ms);
        if elapsed > 0 && self.time_half_life_ms > 0 {
            let time_decay = 2.0_f64.powf(-(elapsed as f64) / (self.time_half_life_ms as f64));
            self.value = time_decay * self.value + (1.0 - time_decay) * self.baseline;
        }

        // Then apply tick-based update
        self.value = self.tick_alpha * observation + (1.0 - self.tick_alpha) * self.value;
        self.last_update_ms = timestamp_ms;
        self.tick_count += 1;
    }

    /// Apply only time decay without a new observation (for staleness handling)
    pub(crate) fn apply_time_decay(&mut self, current_time_ms: u64) {
        if !self.initialized || self.time_half_life_ms == 0 {
            return;
        }

        let elapsed = current_time_ms.saturating_sub(self.last_update_ms);
        if elapsed > 0 {
            let time_decay = 2.0_f64.powf(-(elapsed as f64) / (self.time_half_life_ms as f64));
            self.value = time_decay * self.value + (1.0 - time_decay) * self.baseline;
            self.last_update_ms = current_time_ms;
        }
    }

    /// Get current value
    #[inline]
    pub(crate) fn value(&self) -> f64 {
        self.value
    }

    /// Get value with time decay applied (non-mutating)
    pub(crate) fn value_at(&self, current_time_ms: u64) -> f64 {
        if !self.initialized || self.time_half_life_ms == 0 {
            return self.value;
        }

        let elapsed = current_time_ms.saturating_sub(self.last_update_ms);
        if elapsed == 0 {
            return self.value;
        }

        let time_decay = 2.0_f64.powf(-(elapsed as f64) / (self.time_half_life_ms as f64));
        time_decay * self.value + (1.0 - time_decay) * self.baseline
    }

    /// Get tick count
    #[inline]
    pub(crate) fn tick_count(&self) -> usize {
        self.tick_count
    }

    /// Check if initialized
    #[inline]
    pub(crate) fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get milliseconds since last update
    pub(crate) fn staleness_ms(&self, current_time_ms: u64) -> u64 {
        current_time_ms.saturating_sub(self.last_update_ms)
    }

    /// Reset to uninitialized state
    pub(crate) fn reset(&mut self) {
        self.value = self.baseline;
        self.last_update_ms = 0;
        self.tick_count = 0;
        self.initialized = false;
    }
}

/// EWMA for tracking variance/second moment (for computing standard deviation)
#[derive(Debug, Clone)]
pub(crate) struct TickEWMAVariance {
    mean: TickEWMA,
    mean_sq: TickEWMA,
}

impl TickEWMAVariance {
    pub(crate) fn new(half_life_ticks: f64) -> Self {
        Self {
            mean: TickEWMA::new_uninitialized(half_life_ticks),
            mean_sq: TickEWMA::new_uninitialized(half_life_ticks),
        }
    }

    pub(crate) fn update(&mut self, observation: f64) {
        self.mean.update(observation);
        self.mean_sq.update(observation * observation);
    }

    pub(crate) fn mean(&self) -> f64 {
        self.mean.value()
    }

    pub(crate) fn variance(&self) -> f64 {
        let m = self.mean.value();
        (self.mean_sq.value() - m * m).max(0.0)
    }

    pub(crate) fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    pub(crate) fn coefficient_of_variation(&self) -> f64 {
        let m = self.mean();
        if m.abs() < 1e-12 {
            0.0
        } else {
            self.std() / m.abs()
        }
    }

    pub(crate) fn tick_count(&self) -> usize {
        self.mean.tick_count()
    }

    pub(crate) fn is_initialized(&self) -> bool {
        self.mean.is_initialized()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tick_ewma_half_life() {
        // With half_life = 10 ticks, after 10 updates of 0,
        // an initial value of 1.0 should decay to ~0.5
        let mut ewma = TickEWMA::new(10.0, 1.0);
        for _ in 0..10 {
            ewma.update(0.0);
        }
        // Should be close to 0.5
        assert!((ewma.value() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_tick_ewma_convergence() {
        let mut ewma = TickEWMA::new(5.0, 0.0);
        // Feed constant value, should converge
        for _ in 0..100 {
            ewma.update(1.0);
        }
        assert!((ewma.value() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_hybrid_ewma_time_decay() {
        let mut ewma = HybridEWMA::new(10.0, 1000, 0.0); // 1 second half-life
        ewma.update(1.0, 0);

        // After 1 second with no updates, should decay toward 0
        let decayed = ewma.value_at(1000);
        assert!((decayed - 0.5).abs() < 0.01);

        // After 2 seconds, should be ~0.25
        let decayed2 = ewma.value_at(2000);
        assert!((decayed2 - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_hybrid_ewma_tick_update() {
        let mut ewma = HybridEWMA::new(5.0, 10000, 0.5);
        ewma.update(1.0, 0);
        ewma.update(1.0, 100); // 100ms later, minimal time decay
        ewma.update(1.0, 200);

        // Should be close to 1.0 after multiple 1.0 updates
        assert!(ewma.value() > 0.9);
    }

    #[test]
    fn test_variance_tracking() {
        let mut var = TickEWMAVariance::new(20.0);

        // Feed values 0, 1, 0, 1, ... (variance should be ~0.25)
        for i in 0..100 {
            var.update((i % 2) as f64);
        }

        assert!((var.mean() - 0.5).abs() < 0.1);
        assert!((var.variance() - 0.25).abs() < 0.1);
    }

    #[test]
    fn test_coefficient_of_variation() {
        let mut var = TickEWMAVariance::new(50.0);

        // Exponential distribution has CV = 1
        // Simulate with values centered around mean with some spread
        for i in 0..200 {
            let v = 1.0 + 0.5 * ((i as f64) * 0.1).sin();
            var.update(v);
        }

        let cv = var.coefficient_of_variation();
        assert!(cv > 0.0 && cv < 1.0);
    }
}
