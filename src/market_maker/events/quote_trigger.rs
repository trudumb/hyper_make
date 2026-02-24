//! Quote update event triggers for event-driven architecture.
//!
//! Replaces timed polling with event-based reconciliation triggers.
//! This reduces order churn by only reconciling when meaningful events occur.
//!
//! ## Event Types
//!
//! - **MidPriceMove**: Price moved significantly (> threshold bps)
//! - **FillReceived**: An order was filled (immediate trigger)
//! - **QueueDepletion**: Queue position degraded significantly
//! - **SignalChange**: Model parameters changed significantly
//! - **FallbackTimer**: Safety timer (5-10 seconds)
//!
//! ## Reconciliation Scope
//!
//! Events specify which parts of the ladder need reconciliation:
//! - **Full**: Both sides, all levels
//! - **SideOnly**: One side only (e.g., after a fill)
//! - **LevelsOnly**: Specific orders only
//! - **SideAndLevels**: One side, specific orders

use std::time::{Duration, Instant};

use crate::market_maker::tracking::Side;

/// Types of events that can trigger quote updates.
#[derive(Debug, Clone)]
pub enum QuoteUpdateEvent {
    /// Mid price moved significantly.
    MidPriceMove {
        /// How much the mid moved (absolute bps)
        delta_bps: f64,
        /// New mid price
        new_mid: f64,
        /// Previous mid price
        old_mid: f64,
    },

    /// A fill was received.
    FillReceived {
        /// Which side was filled
        side: Side,
        /// Order ID that was filled
        oid: u64,
        /// Size that was filled
        size: f64,
        /// Whether this was a full fill
        is_full_fill: bool,
    },

    /// Queue position degraded significantly.
    QueueDepletion {
        /// Order ID with degraded queue
        oid: u64,
        /// Current fill probability estimate
        fill_prob: f64,
        /// Previous fill probability
        prev_fill_prob: f64,
        /// Side of the order
        side: Side,
    },

    /// Model signal changed significantly.
    SignalChange {
        /// Name of the signal that changed
        signal_name: String,
        /// Magnitude of change (relative)
        magnitude: f64,
    },

    /// Fallback timer triggered (safety mechanism).
    FallbackTimer {
        /// Time since last reconciliation
        elapsed: Duration,
    },

    /// Volatility spike detected.
    VolatilitySpike {
        /// Current sigma
        sigma: f64,
        /// Previous sigma
        prev_sigma: f64,
        /// Ratio of change
        ratio: f64,
    },

    /// Regime change detected.
    RegimeChange {
        /// New regime name/description
        new_regime: String,
        /// Confidence in regime change
        confidence: f64,
    },
}

impl QuoteUpdateEvent {
    /// Get the event type name for logging.
    pub fn event_type(&self) -> &'static str {
        match self {
            QuoteUpdateEvent::MidPriceMove { .. } => "MidPriceMove",
            QuoteUpdateEvent::FillReceived { .. } => "FillReceived",
            QuoteUpdateEvent::QueueDepletion { .. } => "QueueDepletion",
            QuoteUpdateEvent::SignalChange { .. } => "SignalChange",
            QuoteUpdateEvent::FallbackTimer { .. } => "FallbackTimer",
            QuoteUpdateEvent::VolatilitySpike { .. } => "VolatilitySpike",
            QuoteUpdateEvent::RegimeChange { .. } => "RegimeChange",
        }
    }

    /// Get the priority of this event (higher = more urgent).
    pub fn priority(&self) -> u8 {
        match self {
            QuoteUpdateEvent::FillReceived { is_full_fill, .. } => {
                if *is_full_fill {
                    100
                } else {
                    90
                }
            }
            QuoteUpdateEvent::VolatilitySpike { .. } => 85,
            QuoteUpdateEvent::RegimeChange { .. } => 80,
            QuoteUpdateEvent::MidPriceMove { delta_bps, .. } => {
                // Scale priority by price move magnitude
                (50.0 + (*delta_bps / 2.0).min(40.0)) as u8
            }
            QuoteUpdateEvent::QueueDepletion { .. } => 40,
            QuoteUpdateEvent::SignalChange { magnitude, .. } => {
                (30.0 + (*magnitude * 20.0).min(30.0)) as u8
            }
            QuoteUpdateEvent::FallbackTimer { .. } => 10,
        }
    }
}

/// Scope of reconciliation needed.
#[derive(Debug, Clone)]
pub enum ReconcileScope {
    /// Full reconciliation of both sides, all levels.
    Full,

    /// Only reconcile one side.
    SideOnly { side: Side },

    /// Only reconcile specific orders.
    LevelsOnly { oids: Vec<u64> },

    /// One side, specific orders.
    SideAndLevels { side: Side, oids: Vec<u64> },

    /// No reconciliation needed (event was filtered).
    None,
}

impl ReconcileScope {
    /// Check if this scope includes a given side.
    pub fn includes_side(&self, side: Side) -> bool {
        match self {
            ReconcileScope::Full => true,
            ReconcileScope::SideOnly { side: s } => *s == side,
            ReconcileScope::LevelsOnly { .. } => true, // Levels can be on either side
            ReconcileScope::SideAndLevels { side: s, .. } => *s == side,
            ReconcileScope::None => false,
        }
    }

    /// Check if this scope includes a given order.
    pub fn includes_order(&self, oid: u64) -> bool {
        match self {
            ReconcileScope::Full => true,
            ReconcileScope::SideOnly { .. } => true, // All orders on the side
            ReconcileScope::LevelsOnly { oids } => oids.contains(&oid),
            ReconcileScope::SideAndLevels { oids, .. } => oids.contains(&oid),
            ReconcileScope::None => false,
        }
    }

    /// Check if this is a full reconciliation.
    pub fn is_full(&self) -> bool {
        matches!(self, ReconcileScope::Full)
    }

    /// Check if no reconciliation is needed.
    pub fn is_none(&self) -> bool {
        matches!(self, ReconcileScope::None)
    }

    /// Merge two scopes (result is the union).
    pub fn merge(self, other: ReconcileScope) -> ReconcileScope {
        match (&self, &other) {
            (ReconcileScope::None, _) => other,
            (_, ReconcileScope::None) => self,
            (ReconcileScope::Full, _) | (_, ReconcileScope::Full) => ReconcileScope::Full,
            (ReconcileScope::SideOnly { side: s1 }, ReconcileScope::SideOnly { side: s2 }) => {
                if s1 == s2 {
                    self
                } else {
                    ReconcileScope::Full
                }
            }
            _ => ReconcileScope::Full, // Default to full for complex merges
        }
    }
}

/// Combined trigger with event and scope.
#[derive(Debug, Clone)]
pub struct QuoteUpdateTrigger {
    /// The event that triggered the update.
    pub event: QuoteUpdateEvent,

    /// The scope of reconciliation needed.
    pub scope: ReconcileScope,

    /// When the event occurred.
    pub timestamp: Instant,
}

impl QuoteUpdateTrigger {
    /// Create a new trigger.
    pub fn new(event: QuoteUpdateEvent, scope: ReconcileScope) -> Self {
        Self {
            event,
            scope,
            timestamp: Instant::now(),
        }
    }

    /// Create a full reconciliation trigger.
    pub fn full(event: QuoteUpdateEvent) -> Self {
        Self::new(event, ReconcileScope::Full)
    }

    /// Create a side-only trigger.
    pub fn side_only(event: QuoteUpdateEvent, side: Side) -> Self {
        Self::new(event, ReconcileScope::SideOnly { side })
    }

    /// Create a no-op trigger (event filtered).
    pub fn none(event: QuoteUpdateEvent) -> Self {
        Self::new(event, ReconcileScope::None)
    }

    /// Get the age of this trigger.
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed()
    }

    /// Get the priority of this trigger.
    pub fn priority(&self) -> u8 {
        self.event.priority()
    }
}

/// Configuration for event-driven updates.
#[derive(Debug, Clone)]
pub struct EventDrivenConfig {
    /// Minimum mid price move to trigger reconciliation (bps).
    pub mid_move_threshold_bps: f64,

    /// Whether fills should immediately trigger reconciliation.
    pub fill_immediate_trigger: bool,

    /// Minimum fill probability before queue depletion triggers.
    pub queue_depletion_p_fill: f64,

    /// Minimum signal change magnitude to trigger (relative).
    pub signal_change_threshold: f64,

    /// Fallback timer interval (seconds).
    pub fallback_interval: Duration,

    /// Maximum events to accumulate before forcing reconciliation.
    pub max_pending_events: usize,

    /// Minimum time between reconciliations (debounce).
    pub min_reconcile_interval: Duration,

    /// Volatility ratio threshold to trigger spike event.
    pub volatility_spike_ratio: f64,

    /// Enable event-driven mode (false = legacy timed mode).
    pub enabled: bool,
}

impl Default for EventDrivenConfig {
    fn default() -> Self {
        Self {
            mid_move_threshold_bps: 5.0,
            fill_immediate_trigger: true,
            queue_depletion_p_fill: 0.05,
            signal_change_threshold: 0.20, // 20% change
            fallback_interval: Duration::from_secs(5),
            max_pending_events: 10,
            min_reconcile_interval: Duration::from_millis(100),
            volatility_spike_ratio: 1.5, // 50% increase
            enabled: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_priority() {
        let fill = QuoteUpdateEvent::FillReceived {
            side: Side::Buy,
            oid: 1,
            size: 1.0,
            is_full_fill: true,
        };
        let timer = QuoteUpdateEvent::FallbackTimer {
            elapsed: Duration::from_secs(5),
        };

        assert!(fill.priority() > timer.priority());
    }

    #[test]
    fn test_scope_merge() {
        let buy_only = ReconcileScope::SideOnly { side: Side::Buy };
        let sell_only = ReconcileScope::SideOnly { side: Side::Sell };
        let none = ReconcileScope::None;

        // Merging opposite sides = full
        assert!(buy_only.clone().merge(sell_only).is_full());

        // Merging with none = other
        assert!(matches!(
            none.clone().merge(buy_only.clone()),
            ReconcileScope::SideOnly { .. }
        ));

        // Same side = same
        let merged = buy_only
            .clone()
            .merge(ReconcileScope::SideOnly { side: Side::Buy });
        assert!(matches!(
            merged,
            ReconcileScope::SideOnly { side: Side::Buy }
        ));
    }

    #[test]
    fn test_scope_includes() {
        let buy_only = ReconcileScope::SideOnly { side: Side::Buy };
        let levels = ReconcileScope::LevelsOnly {
            oids: vec![1, 2, 3],
        };

        assert!(buy_only.includes_side(Side::Buy));
        assert!(!buy_only.includes_side(Side::Sell));

        assert!(levels.includes_order(1));
        assert!(!levels.includes_order(4));
    }
}
