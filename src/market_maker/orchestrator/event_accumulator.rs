//! Event accumulator for event-driven quote updates.
//!
//! Accumulates events and decides when to trigger reconciliation.
//! This replaces the timed polling approach with event-driven updates
//! that only reconcile when meaningful changes occur.
//!
//! ## Features
//!
//! - **Event Buffering**: Accumulates multiple events before reconciling
//! - **Priority Handling**: Higher priority events trigger faster
//! - **Scope Merging**: Combines event scopes for efficient reconciliation
//! - **Debouncing**: Prevents too-frequent updates
//! - **Fallback Timer**: Safety mechanism if no events occur

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::market_maker::events::quote_trigger::{
    EventDrivenConfig, QuoteUpdateEvent, QuoteUpdateTrigger, ReconcileScope,
};
use crate::market_maker::tracking::Side;

/// Tracks which parts of the ladder are affected by pending events.
#[allow(dead_code)] // WIP: Will be used when event handlers are wired
#[derive(Debug, Clone, Default)]
struct AffectedTracker {
    /// Whether bids are affected.
    bids_affected: bool,
    /// Whether asks are affected.
    asks_affected: bool,
    /// Specific order IDs affected.
    affected_oids: Vec<u64>,
    /// Accumulated price move (bps).
    accumulated_price_move_bps: f64,
    /// Whether a fill occurred.
    had_fill: bool,
    /// Maximum event priority seen.
    max_priority: u8,
}

#[allow(dead_code)] // WIP: Methods will be used when event handlers are wired
impl AffectedTracker {
    /// Create a new affected tracker.
    fn new() -> Self {
        Self::default()
    }

    /// Record an event that affects a side.
    fn record_side(&mut self, side: Side, priority: u8) {
        match side {
            Side::Buy => self.bids_affected = true,
            Side::Sell => self.asks_affected = true,
        }
        self.max_priority = self.max_priority.max(priority);
    }

    /// Record an event that affects a specific order.
    fn record_order(&mut self, oid: u64, side: Side, priority: u8) {
        if !self.affected_oids.contains(&oid) {
            self.affected_oids.push(oid);
        }
        self.record_side(side, priority);
    }

    /// Record a price move.
    fn record_price_move(&mut self, delta_bps: f64, priority: u8) {
        self.accumulated_price_move_bps += delta_bps;
        self.bids_affected = true;
        self.asks_affected = true;
        self.max_priority = self.max_priority.max(priority);
    }

    /// Record a fill.
    fn record_fill(&mut self, side: Side, oid: u64, priority: u8) {
        self.had_fill = true;
        self.record_order(oid, side, priority);
    }

    /// Convert to reconcile scope.
    fn to_scope(&self) -> ReconcileScope {
        if !self.bids_affected && !self.asks_affected {
            return ReconcileScope::None;
        }

        if self.bids_affected && self.asks_affected {
            if self.affected_oids.is_empty() {
                ReconcileScope::Full
            } else {
                // Both sides but specific orders - still do full for safety
                ReconcileScope::Full
            }
        } else if self.bids_affected {
            if self.affected_oids.is_empty() {
                ReconcileScope::SideOnly { side: Side::Buy }
            } else {
                ReconcileScope::SideAndLevels {
                    side: Side::Buy,
                    oids: self.affected_oids.clone(),
                }
            }
        } else if self.affected_oids.is_empty() {
            ReconcileScope::SideOnly { side: Side::Sell }
        } else {
            ReconcileScope::SideAndLevels {
                side: Side::Sell,
                oids: self.affected_oids.clone(),
            }
        }
    }

    /// Reset the tracker.
    fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Accumulates events and decides when to trigger reconciliation.
#[allow(dead_code)] // WIP: Fields will be used when event handlers are wired
#[derive(Debug)]
pub(crate) struct EventAccumulator {
    config: EventDrivenConfig,

    /// Pending events.
    events: VecDeque<QuoteUpdateTrigger>,

    /// Affected parts tracker.
    affected: AffectedTracker,

    /// Last reconciliation time.
    last_reconcile: Instant,

    /// Last mid price (for tracking moves).
    last_mid: f64,

    /// Last sigma (for tracking volatility changes).
    last_sigma: f64,

    /// Total events processed.
    total_events: u64,

    /// Total reconciliations triggered.
    total_reconciles: u64,

    /// Events filtered (not triggering reconcile).
    events_filtered: u64,
}

#[allow(dead_code)] // WIP: Methods will be used when event handlers are wired
impl EventAccumulator {
    /// Create a new event accumulator.
    pub(crate) fn new(config: EventDrivenConfig) -> Self {
        Self {
            config,
            events: VecDeque::with_capacity(20),
            affected: AffectedTracker::new(),
            last_reconcile: Instant::now(),
            last_mid: 0.0,
            last_sigma: 0.0,
            total_events: 0,
            total_reconciles: 0,
            events_filtered: 0,
        }
    }

    /// Create with default configuration.
    pub(crate) fn default_config() -> Self {
        Self::new(EventDrivenConfig::default())
    }

    /// Record a mid price update.
    ///
    /// Returns the event if it triggered, None if filtered.
    pub(crate) fn on_mid_update(&mut self, new_mid: f64) -> Option<QuoteUpdateTrigger> {
        if !self.config.enabled {
            return None;
        }

        if self.last_mid <= 0.0 {
            self.last_mid = new_mid;
            return None;
        }

        let delta_bps = ((new_mid - self.last_mid) / self.last_mid).abs() * 10000.0;

        if delta_bps >= self.config.mid_move_threshold_bps {
            let event = QuoteUpdateEvent::MidPriceMove {
                delta_bps,
                new_mid,
                old_mid: self.last_mid,
            };
            self.last_mid = new_mid;
            let trigger = self.record_event(event, ReconcileScope::Full);
            return Some(trigger);
        }

        // Still track the price for accumulated moves
        self.affected.accumulated_price_move_bps += delta_bps;
        self.last_mid = new_mid;

        None
    }

    /// Record a fill event.
    pub(crate) fn on_fill(&mut self, side: Side, oid: u64, size: f64, is_full_fill: bool) -> QuoteUpdateTrigger {
        let event = QuoteUpdateEvent::FillReceived {
            side,
            oid,
            size,
            is_full_fill,
        };

        let scope = if self.config.fill_immediate_trigger {
            ReconcileScope::SideOnly { side }
        } else {
            ReconcileScope::LevelsOnly { oids: vec![oid] }
        };

        self.affected.record_fill(side, oid, event.priority());
        self.record_event(event, scope)
    }

    /// Record a queue depletion event.
    pub(crate) fn on_queue_depletion(
        &mut self,
        oid: u64,
        side: Side,
        fill_prob: f64,
        prev_fill_prob: f64,
    ) -> Option<QuoteUpdateTrigger> {
        if !self.config.enabled {
            return None;
        }

        if fill_prob < self.config.queue_depletion_p_fill && prev_fill_prob >= self.config.queue_depletion_p_fill {
            let event = QuoteUpdateEvent::QueueDepletion {
                oid,
                fill_prob,
                prev_fill_prob,
                side,
            };
            self.affected.record_order(oid, side, event.priority());
            return Some(self.record_event(event, ReconcileScope::LevelsOnly { oids: vec![oid] }));
        }

        None
    }

    /// Record a signal change event.
    pub(crate) fn on_signal_change(&mut self, signal_name: &str, magnitude: f64) -> Option<QuoteUpdateTrigger> {
        if !self.config.enabled {
            return None;
        }

        if magnitude >= self.config.signal_change_threshold {
            let event = QuoteUpdateEvent::SignalChange {
                signal_name: signal_name.to_string(),
                magnitude,
            };
            self.affected.record_price_move(0.0, event.priority()); // Mark both sides
            return Some(self.record_event(event, ReconcileScope::Full));
        }

        None
    }

    /// Record a volatility update.
    pub(crate) fn on_volatility_update(&mut self, sigma: f64) -> Option<QuoteUpdateTrigger> {
        if !self.config.enabled {
            return None;
        }

        if self.last_sigma <= 0.0 {
            self.last_sigma = sigma;
            return None;
        }

        let ratio = sigma / self.last_sigma;
        self.last_sigma = sigma;

        if ratio >= self.config.volatility_spike_ratio || ratio <= (1.0 / self.config.volatility_spike_ratio) {
            let event = QuoteUpdateEvent::VolatilitySpike {
                sigma,
                prev_sigma: self.last_sigma,
                ratio,
            };
            return Some(self.record_event(event, ReconcileScope::Full));
        }

        None
    }

    /// Check the fallback timer.
    pub(crate) fn check_fallback(&mut self) -> Option<QuoteUpdateTrigger> {
        if !self.config.enabled {
            return None;
        }

        let elapsed = self.last_reconcile.elapsed();
        if elapsed >= self.config.fallback_interval {
            // Only trigger if there's something pending
            if self.affected.bids_affected || self.affected.asks_affected || !self.events.is_empty() {
                let event = QuoteUpdateEvent::FallbackTimer { elapsed };
                return Some(self.record_event(event, self.affected.to_scope()));
            }
        }

        None
    }

    /// Record an event and return the trigger.
    fn record_event(&mut self, event: QuoteUpdateEvent, scope: ReconcileScope) -> QuoteUpdateTrigger {
        self.total_events += 1;

        let trigger = QuoteUpdateTrigger::new(event, scope);

        if self.events.len() >= self.config.max_pending_events {
            self.events.pop_front();
        }
        self.events.push_back(trigger.clone());

        trigger
    }

    /// Check if reconciliation should be triggered now.
    ///
    /// Returns the trigger if reconciliation is needed, None otherwise.
    pub(crate) fn should_trigger(&self) -> Option<QuoteUpdateTrigger> {
        if !self.config.enabled {
            return None;
        }

        // Debounce check
        if self.last_reconcile.elapsed() < self.config.min_reconcile_interval {
            return None;
        }

        // No events = no trigger
        if self.events.is_empty() && !self.affected.had_fill {
            return None;
        }

        // High priority events trigger immediately
        if self.affected.max_priority >= 80 {
            return Some(QuoteUpdateTrigger::new(
                QuoteUpdateEvent::FallbackTimer { elapsed: Duration::ZERO },
                self.affected.to_scope(),
            ));
        }

        // Accumulated price move exceeds threshold
        if self.affected.accumulated_price_move_bps >= self.config.mid_move_threshold_bps {
            return Some(QuoteUpdateTrigger::new(
                QuoteUpdateEvent::MidPriceMove {
                    delta_bps: self.affected.accumulated_price_move_bps,
                    new_mid: self.last_mid,
                    old_mid: 0.0, // Not tracked for accumulated
                },
                ReconcileScope::Full,
            ));
        }

        // Too many pending events
        if self.events.len() >= self.config.max_pending_events {
            return Some(QuoteUpdateTrigger::full(QuoteUpdateEvent::FallbackTimer {
                elapsed: self.last_reconcile.elapsed(),
            }));
        }

        // Fill events trigger side update
        if self.affected.had_fill {
            return Some(QuoteUpdateTrigger::new(
                QuoteUpdateEvent::FallbackTimer { elapsed: Duration::ZERO },
                self.affected.to_scope(),
            ));
        }

        None
    }

    /// Reset after reconciliation.
    pub(crate) fn reset(&mut self) {
        self.events.clear();
        self.affected.reset();
        self.last_reconcile = Instant::now();
        self.total_reconciles += 1;
    }

    /// Mark events as filtered (not triggering reconcile).
    pub(crate) fn mark_filtered(&mut self, count: usize) {
        self.events_filtered += count as u64;
    }

    /// Get statistics.
    pub(crate) fn stats(&self) -> EventAccumulatorStats {
        EventAccumulatorStats {
            total_events: self.total_events,
            total_reconciles: self.total_reconciles,
            events_filtered: self.events_filtered,
            pending_events: self.events.len(),
            time_since_last_reconcile: self.last_reconcile.elapsed(),
            accumulated_price_move_bps: self.affected.accumulated_price_move_bps,
            max_pending_priority: self.affected.max_priority,
        }
    }

    /// Check if event-driven mode is enabled.
    pub(crate) fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the configuration.
    pub(crate) fn config(&self) -> &EventDrivenConfig {
        &self.config
    }

    /// Update configuration.
    pub(crate) fn set_config(&mut self, config: EventDrivenConfig) {
        self.config = config;
    }
}

/// Statistics for event accumulator.
#[allow(dead_code)] // WIP: Will be used for monitoring when event handlers are wired
#[derive(Debug, Clone)]
pub(crate) struct EventAccumulatorStats {
    pub(crate) total_events: u64,
    pub(crate) total_reconciles: u64,
    pub(crate) events_filtered: u64,
    pub(crate) pending_events: usize,
    pub(crate) time_since_last_reconcile: Duration,
    pub(crate) accumulated_price_move_bps: f64,
    pub(crate) max_pending_priority: u8,
}

#[allow(dead_code)] // WIP: Methods will be used for monitoring
impl EventAccumulatorStats {
    /// Calculate filter ratio (events filtered / total events).
    pub(crate) fn filter_ratio(&self) -> f64 {
        if self.total_events == 0 {
            0.0
        } else {
            self.events_filtered as f64 / self.total_events as f64
        }
    }

    /// Calculate reconciliation frequency (reconciles per event).
    pub(crate) fn reconcile_frequency(&self) -> f64 {
        if self.total_events == 0 {
            0.0
        } else {
            self.total_reconciles as f64 / self.total_events as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_mid_update() {
        let mut acc = EventAccumulator::default_config();

        // First update just sets baseline
        assert!(acc.on_mid_update(100.0).is_none());

        // Small move should not trigger
        assert!(acc.on_mid_update(100.01).is_none());

        // Large move should trigger
        let trigger = acc.on_mid_update(100.6).unwrap();
        assert!(matches!(trigger.event, QuoteUpdateEvent::MidPriceMove { .. }));
    }

    #[test]
    fn test_accumulator_fill() {
        let mut acc = EventAccumulator::default_config();

        let trigger = acc.on_fill(Side::Buy, 123, 1.0, true);
        assert!(matches!(trigger.event, QuoteUpdateEvent::FillReceived { .. }));
        assert!(acc.affected.had_fill);
    }

    #[test]
    fn test_should_trigger() {
        let mut acc = EventAccumulator::default_config();

        // No events = no trigger
        assert!(acc.should_trigger().is_none());

        // Fill should trigger
        acc.on_fill(Side::Buy, 1, 1.0, true);

        // Wait for debounce
        std::thread::sleep(Duration::from_millis(150));

        assert!(acc.should_trigger().is_some());
    }

    #[test]
    fn test_reset() {
        let mut acc = EventAccumulator::default_config();

        acc.on_fill(Side::Buy, 1, 1.0, true);
        assert!(acc.affected.had_fill);

        acc.reset();
        assert!(!acc.affected.had_fill);
        assert!(acc.events.is_empty());
    }

    #[test]
    fn test_affected_tracker() {
        let mut tracker = AffectedTracker::new();

        tracker.record_side(Side::Buy, 50);
        assert!(tracker.bids_affected);
        assert!(!tracker.asks_affected);

        tracker.record_side(Side::Sell, 60);
        assert!(tracker.bids_affected);
        assert!(tracker.asks_affected);

        let scope = tracker.to_scope();
        assert!(scope.is_full());
    }
}
