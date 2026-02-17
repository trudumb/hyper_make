//! Order lifecycle churn tracking and diagnostics.
//!
//! Tracks cancel/fill ratios, latch effectiveness, and budget suppression
//! across a rolling window to diagnose pathologies in the order lifecycle.

use std::collections::VecDeque;

/// Per-cycle summary of order lifecycle actions.
#[derive(Debug, Clone, Default)]
pub struct CycleSummary {
    pub placed: u32,
    pub cancelled: u32,
    pub filled: u32,
    pub modified: u32,
    pub latched: u32,
    pub grid_preserved: u32,
    pub budget_suppressed: u32,
}

/// Threshold for cancel/fill ratio that indicates excessive churn.
const HIGH_CHURN_RATIO: f64 = 10.0;
/// Number of consecutive high-churn cycles before alerting.
const HIGH_CHURN_STREAK_ALERT: u32 = 50;

/// Rolling-window churn tracker for order lifecycle diagnostics.
pub struct ChurnTracker {
    window: VecDeque<CycleSummary>,
    max_window: usize,
    high_churn_streak: u32,
}

impl ChurnTracker {
    pub fn new(max_window: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(max_window.min(1024)),
            max_window,
            high_churn_streak: 0,
        }
    }

    /// Record a cycle summary into the rolling window.
    ///
    /// Trims to `max_window` and updates the high-churn streak counter.
    pub fn record_cycle(&mut self, summary: CycleSummary) {
        self.window.push_back(summary);
        while self.window.len() > self.max_window {
            self.window.pop_front();
        }

        // Track consecutive high-churn cycles
        if self.cancel_fill_ratio() > HIGH_CHURN_RATIO {
            self.high_churn_streak += 1;
            if self.high_churn_streak == HIGH_CHURN_STREAK_ALERT {
                tracing::warn!(
                    cancel_fill_ratio = %format!("{:.1}", self.cancel_fill_ratio()),
                    streak = self.high_churn_streak,
                    "Sustained high churn: cancel/fill ratio > {} for {} consecutive cycles",
                    HIGH_CHURN_RATIO,
                    HIGH_CHURN_STREAK_ALERT,
                );
            }
        } else {
            self.high_churn_streak = 0;
        }
    }

    /// Ratio of total cancellations to total fills across the window.
    ///
    /// A ratio > 10 sustained over 50+ cycles indicates a pathological cancel loop.
    pub fn cancel_fill_ratio(&self) -> f64 {
        let total_cancelled: u32 = self.window.iter().map(|s| s.cancelled).sum();
        let total_filled: u32 = self.window.iter().map(|s| s.filled).sum();
        total_cancelled as f64 / total_filled.max(1) as f64
    }

    /// Churn rate: (cancels + placements) per minute.
    ///
    /// High values indicate the reconciler is thrashing rather than latching.
    pub fn churn_rate_per_minute(&self, cycle_interval_secs: f64) -> f64 {
        if self.window.is_empty() || cycle_interval_secs <= 0.0 {
            return 0.0;
        }
        let total_churn: u32 = self
            .window
            .iter()
            .map(|s| s.cancelled + s.placed)
            .sum();
        let window_duration_secs = self.window.len() as f64 * cycle_interval_secs;
        total_churn as f64 / window_duration_secs * 60.0
    }

    /// Fraction of decisions where an order was latched (preserved) vs total decisions.
    ///
    /// Higher = better; indicates the latch mechanism is reducing unnecessary churn.
    pub fn latch_effectiveness(&self) -> f64 {
        let total_latched: u32 = self.window.iter().map(|s| s.latched).sum();
        let total_decisions: u32 = self
            .window
            .iter()
            .map(|s| s.latched + s.cancelled + s.modified + s.placed)
            .sum();
        total_latched as f64 / total_decisions.max(1) as f64
    }

    /// Total actions (all types) across the rolling window.
    pub fn total_actions(&self) -> u32 {
        self.window
            .iter()
            .map(|s| {
                s.placed
                    + s.cancelled
                    + s.filled
                    + s.modified
                    + s.latched
                    + s.grid_preserved
                    + s.budget_suppressed
            })
            .sum()
    }

    /// Current high-churn streak length (consecutive cycles with cancel/fill > 10).
    pub fn high_churn_streak(&self) -> u32 {
        self.high_churn_streak
    }

    /// Number of cycles in the current window.
    pub fn window_len(&self) -> usize {
        self.window.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancel_fill_ratio() {
        let mut tracker = ChurnTracker::new(100);

        // 10 cancels, 2 fills → ratio = 5.0
        tracker.record_cycle(CycleSummary {
            cancelled: 6,
            filled: 1,
            ..Default::default()
        });
        tracker.record_cycle(CycleSummary {
            cancelled: 4,
            filled: 1,
            ..Default::default()
        });

        let ratio = tracker.cancel_fill_ratio();
        assert!(
            (ratio - 5.0).abs() < f64::EPSILON,
            "Expected ratio 5.0, got {:.2}",
            ratio
        );
    }

    #[test]
    fn test_churn_rate_per_minute() {
        let mut tracker = ChurnTracker::new(100);

        // 3 cycles, each with 2 cancels + 2 places = 4 churn each
        // Total churn = 12, window = 3 cycles × 0.5s = 1.5s
        // Rate = 12 / 1.5 * 60 = 480 per minute
        for _ in 0..3 {
            tracker.record_cycle(CycleSummary {
                placed: 2,
                cancelled: 2,
                ..Default::default()
            });
        }

        let rate = tracker.churn_rate_per_minute(0.5);
        assert!(
            (rate - 480.0).abs() < 1e-6,
            "Expected 480/min, got {:.2}",
            rate
        );
    }

    #[test]
    fn test_latch_effectiveness() {
        let mut tracker = ChurnTracker::new(100);

        // 5 latched, 3 cancelled, 2 placed = 10 total decisions
        // Effectiveness = 5/10 = 0.5
        tracker.record_cycle(CycleSummary {
            latched: 5,
            cancelled: 3,
            placed: 2,
            ..Default::default()
        });

        let eff = tracker.latch_effectiveness();
        assert!(
            (eff - 0.5).abs() < f64::EPSILON,
            "Expected 0.5 effectiveness, got {:.2}",
            eff
        );
    }

    #[test]
    fn test_window_trimming() {
        let mut tracker = ChurnTracker::new(3);

        for i in 0..5 {
            tracker.record_cycle(CycleSummary {
                placed: i,
                ..Default::default()
            });
        }

        assert_eq!(tracker.window_len(), 3);
        // Window should contain cycles 2, 3, 4 (placed = 2, 3, 4)
        assert_eq!(tracker.total_actions(), 2 + 3 + 4);
    }

    #[test]
    fn test_high_churn_streak_resets() {
        let mut tracker = ChurnTracker::new(100);

        // Drive up the streak with high cancel/fill
        for _ in 0..5 {
            tracker.record_cycle(CycleSummary {
                cancelled: 20,
                filled: 1,
                ..Default::default()
            });
        }
        assert!(tracker.high_churn_streak() > 0);

        // A cycle with fills resets the ratio and streak
        tracker.record_cycle(CycleSummary {
            cancelled: 0,
            filled: 100,
            ..Default::default()
        });
        assert_eq!(tracker.high_churn_streak(), 0);
    }

    #[test]
    fn test_empty_tracker() {
        let tracker = ChurnTracker::new(100);
        assert_eq!(tracker.cancel_fill_ratio(), 0.0);
        assert_eq!(tracker.churn_rate_per_minute(0.5), 0.0);
        assert_eq!(tracker.latch_effectiveness(), 0.0);
        assert_eq!(tracker.total_actions(), 0);
    }
}
