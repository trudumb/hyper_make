//! Quote generation and filtering.
//!
//! Provides utilities for:
//! - **Ladder**: Multi-level quote generation with depth-dependent sizing
//! - **Filter**: Quote filtering based on position limits, reduce-only mode
//! - **KappaSpread**: Kappa-driven dynamic spread adjustment

mod filter;
pub mod kappa_spread;
mod ladder;

pub use filter::{
    apply_close_bias, QuoteFilter, ReduceOnlyConfig, ReduceOnlyReason, ReduceOnlyResult,
    DEFAULT_LIQUIDATION_TRIGGER_THRESHOLD,
};
pub use kappa_spread::{
    KappaRegime, KappaSpreadConfig, KappaSpreadController, KappaSpreadDiagnostics,
    KappaSpreadResult,
};
pub use ladder::*;
