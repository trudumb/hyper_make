//! Models for execution-level decision making.
//!
//! - [`QueueValueHeuristic`]: Expected value of quoting at a given depth level

mod queue_value;

pub use queue_value::QueueValueHeuristic;
