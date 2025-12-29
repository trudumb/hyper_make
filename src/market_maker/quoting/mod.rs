//! Quote filtering and transformation.
//!
//! Provides utilities for filtering quotes based on position limits,
//! reduce-only mode, and other constraints.

mod filter;

pub use filter::{QuoteFilter, ReduceOnlyConfig, ReduceOnlyResult, ReduceOnlyReason};
