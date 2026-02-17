//! Quote generation and filtering.
//!
//! Provides utilities for:
//! - **Ladder**: Multi-level quote generation with depth-dependent sizing
//! - **Filter**: Quote filtering based on position limits, reduce-only mode
//! - **KappaSpread**: Kappa-driven dynamic spread adjustment

pub mod exchange_rules;
mod filter;
pub mod kappa_spread;
mod ladder;
pub mod price_grid;
pub mod viable;

pub use exchange_rules::{ExchangeRules, QuoteRejection, ValidatedQuote, ValidationReport};
pub use filter::{
    apply_close_bias, QuoteFilter, ReduceOnlyConfig, ReduceOnlyReason, ReduceOnlyResult,
    DEFAULT_LIQUIDATION_TRIGGER_THRESHOLD,
};
pub use kappa_spread::{
    KappaRegime, KappaSpreadConfig, KappaSpreadController, KappaSpreadDiagnostics,
    KappaSpreadResult,
};
pub use ladder::*;
pub use price_grid::{PriceGrid, PriceGridConfig};
pub use viable::{ViableQuoteLadder, ViableQuoteSide};
