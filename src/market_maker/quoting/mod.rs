//! Quote generation and filtering.
//!
//! Provides utilities for:
//! - **Ladder**: Multi-level quote generation with depth-dependent sizing
//! - **Filter**: Quote filtering based on position limits, reduce-only mode

mod filter;
mod ladder;

pub use filter::{QuoteFilter, ReduceOnlyConfig, ReduceOnlyReason, ReduceOnlyResult};
pub use ladder::*;
