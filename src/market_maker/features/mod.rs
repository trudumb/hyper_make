//! Compact feature extraction for downstream models.
//!
//! [`FeatureSnapshot`] is a ~20-field projection of [`MarketParams`] +
//! [`BeliefSnapshot`] + [`TradeFlowTracker`], computed once per quote cycle.
//! It provides a clean input contract for toxicity classification,
//! queue-value estimation, and execution mode selection.

mod snapshot;

pub use snapshot::FeatureSnapshot;
