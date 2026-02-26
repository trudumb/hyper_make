//! Baseline tracker for counterfactual RL reward computation.
//!
//! The RL agent's raw reward is typically -1.5 bps (fee drag), making it
//! impossible to learn which actions are better or worse. By tracking an
//! EWMA baseline of rewards and subtracting it, the counterfactual reward
//! centers around zero, enabling meaningful Q-value differentiation.

use serde::{Deserialize, Serialize};

/// Tracks EWMA baseline of rewards for counterfactual RL comparison.
///
/// Problem: RL reward = -1.5 bps (fee drag) => agent cannot learn.
/// Solution: counterfactual_reward = actual - baseline => centers at ~0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineTracker {
    /// Exponentially weighted moving average of observed rewards.
    ewma_reward: f64,
    /// EWMA decay factor (e.g., 0.99 = slow adaptation).
    decay: f64,
    /// Total number of reward observations.
    n_observations: u64,
    /// Minimum observations before baseline subtraction is applied.
    min_observations: u64,
}

impl BaselineTracker {
    /// Create a new baseline tracker.
    ///
    /// # Arguments
    /// * `decay` - EWMA decay factor, higher = slower adaptation (e.g. 0.99)
    /// * `min_observations` - warmup period before baseline subtraction kicks in
    pub fn new(decay: f64, min_observations: u64) -> Self {
        Self {
            ewma_reward: 0.0,
            decay: decay.clamp(0.0, 1.0),
            n_observations: 0,
            min_observations,
        }
    }

    /// Update baseline with a new reward observation.
    pub fn observe(&mut self, reward: f64) {
        if self.n_observations == 0 {
            self.ewma_reward = reward;
        } else {
            self.ewma_reward = self.decay * self.ewma_reward + (1.0 - self.decay) * reward;
        }
        self.n_observations += 1;
    }

    /// Get counterfactual reward (actual - baseline).
    ///
    /// Returns raw reward when not enough observations have been collected,
    /// so the agent can still learn from early data without baseline bias.
    pub fn counterfactual_reward(&self, actual: f64) -> f64 {
        if self.n_observations < self.min_observations {
            actual
        } else {
            actual - self.ewma_reward
        }
    }

    /// Current EWMA baseline value.
    pub fn baseline(&self) -> f64 {
        self.ewma_reward
    }

    /// Total number of observations recorded.
    pub fn n_observations(&self) -> u64 {
        self.n_observations
    }

    /// Whether the tracker has enough observations to subtract baseline.
    pub fn is_warmed_up(&self) -> bool {
        self.n_observations >= self.min_observations
    }

    /// Restore state from checkpoint fields.
    pub fn restore(&mut self, ewma: f64, n: u64) {
        self.ewma_reward = ewma;
        self.n_observations = n;
    }
}

impl Default for BaselineTracker {
    fn default() -> Self {
        Self::new(0.99, 10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baseline_tracker_default() {
        let tracker = BaselineTracker::default();
        assert_eq!(tracker.baseline(), 0.0);
        assert_eq!(tracker.n_observations(), 0);
        assert!(!tracker.is_warmed_up());
    }

    #[test]
    fn test_first_observation_sets_baseline() {
        let mut tracker = BaselineTracker::new(0.99, 5);
        tracker.observe(-1.5);
        assert_eq!(tracker.baseline(), -1.5);
        assert_eq!(tracker.n_observations(), 1);
    }

    #[test]
    fn test_ewma_converges() {
        let mut tracker = BaselineTracker::new(0.9, 1);
        // Feed constant -1.5 rewards
        for _ in 0..100 {
            tracker.observe(-1.5);
        }
        // Should converge to -1.5
        assert!((tracker.baseline() - (-1.5)).abs() < 0.01);
    }

    #[test]
    fn test_counterfactual_removes_baseline() {
        let mut tracker = BaselineTracker::new(0.9, 5);
        // Warm up with constant -1.5
        for _ in 0..20 {
            tracker.observe(-1.5);
        }
        assert!(tracker.is_warmed_up());

        // Counterfactual of -1.5 should be ~0
        let cf = tracker.counterfactual_reward(-1.5);
        assert!(cf.abs() < 0.1, "expected ~0, got {cf}");

        // Counterfactual of 0.5 should be ~2.0 (0.5 - (-1.5))
        let cf_positive = tracker.counterfactual_reward(0.5);
        assert!(
            (cf_positive - 2.0).abs() < 0.1,
            "expected ~2.0, got {cf_positive}"
        );
    }

    #[test]
    fn test_warmup_returns_raw_reward() {
        let mut tracker = BaselineTracker::new(0.99, 10);
        // Only 5 observations - not warmed up
        for _ in 0..5 {
            tracker.observe(-1.5);
        }
        assert!(!tracker.is_warmed_up());

        // Should return raw reward during warmup
        let cf = tracker.counterfactual_reward(3.0);
        assert_eq!(cf, 3.0);
    }

    #[test]
    fn test_restore_from_checkpoint() {
        let mut tracker = BaselineTracker::new(0.99, 10);
        tracker.restore(-1.5, 100);
        assert_eq!(tracker.baseline(), -1.5);
        assert_eq!(tracker.n_observations(), 100);
        assert!(tracker.is_warmed_up());
    }

    #[test]
    fn test_decay_clamped() {
        let tracker = BaselineTracker::new(1.5, 10);
        assert_eq!(tracker.decay, 1.0);

        let tracker2 = BaselineTracker::new(-0.5, 10);
        assert_eq!(tracker2.decay, 0.0);
    }

    #[test]
    fn test_positive_reward_shifts_baseline() {
        let mut tracker = BaselineTracker::new(0.5, 1);
        tracker.observe(-2.0);
        tracker.observe(4.0);
        // EWMA: 0.5 * (-2.0) + 0.5 * 4.0 = 1.0
        assert!((tracker.baseline() - 1.0).abs() < 0.01);
    }
}
