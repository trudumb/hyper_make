//! Event types and infrastructure for market maker.
//!
//! This module provides unified event types for market data and fill events.
//! It centralizes event definitions that were previously scattered across modules.
//!
//! ## Modules
//!
//! - `types`: Core market data event types (AllMids, Trades, L2Book)
//! - `quote_trigger`: Event-driven quote update triggers (Phase 3: Churn Reduction)

mod types;
pub mod quote_trigger;

pub use quote_trigger::{
    EventDrivenConfig, QuoteUpdateEvent, QuoteUpdateTrigger, ReconcileScope,
};
pub use types::*;
