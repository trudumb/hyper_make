//! Core state and component bundles for MarketMaker.
//!
//! This module provides helper structs that group related fields,
//! preparing for eventual MarketMaker field reduction.
//!
//! # Design
//!
//! The goal is to reduce MarketMaker from ~29 fields to ~12 via composition:
//!
//! ```text
//! MarketMaker (before: 29 fields)
//! ├── config, strategy, executor, info_client, user_address
//! ├── position, orders, latest_mid, last_warmup_log
//! ├── Tier 1: adverse_selection, depth_decay_as, queue_tracker, liquidation_detector
//! ├── Tier 2: hawkes, funding, spread_tracker, pnl_tracker
//! ├── Safety: kill_switch, risk_aggregator, fill_processor
//! ├── Infra: margin_sizer, prometheus, connection_health, data_quality, metrics
//! └── Stochastic: hjb_controller, stochastic_config, dynamic_risk_config
//!
//! MarketMaker (after: ~14 fields)
//! ├── config, strategy, executor, info_client, user_address
//! ├── core_state: CoreState
//! ├── tier1: Tier1Components
//! ├── tier2: Tier2Components
//! ├── safety: SafetyComponents
//! ├── infra: InfraComponents
//! ├── stochastic: StochasticComponents
//! └── estimator
//! ```

mod state;
mod components;

pub use state::CoreState;
pub use components::{Tier1Components, Tier2Components, InfraComponents, StochasticComponents, SafetyComponents};
