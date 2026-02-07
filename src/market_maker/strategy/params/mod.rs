//! Focused parameter sets for market making.
//!
//! Decomposes the 62-field MarketParams into logical groups for clarity
//! and maintainability. Each struct represents a domain of market state.
//!
//! This module is organized into focused submodules:
//!
//! - `volatility`: Volatility metrics from the estimator
//! - `liquidity`: Order book depth and liquidity metrics
//! - `flow`: Order flow and momentum signals
//! - `regime`: Market regime detection signals
//! - `fair_price`: Microprice and fair value estimation
//! - `adverse_selection`: Adverse selection measurement
//! - `cascade`: Liquidation cascade detection
//! - `funding`: Perpetual funding rate metrics
//! - `hjb`: HJB controller parameters
//! - `margin`: Margin and leverage constraints
//! - `kelly`: Kelly-Stochastic config parameters
//! - `exchange_limits`: Exchange position limits
//! - `stochastic_constraints`: First-principles constraints
//! - `entropy`: Entropy distribution parameters
//! - `aggregator`: Parameter aggregation from multiple sources

mod adverse_selection;
mod aggregator;
mod cascade;
mod entropy;
mod exchange_limits;
mod fair_price;
mod flow;
mod funding;
mod hjb;
mod kelly;
mod liquidity;
mod margin;
mod regime;
mod stochastic_constraints;
mod volatility;

// Re-export everything for backward compatibility
pub use adverse_selection::*;
pub use aggregator::*;
pub use cascade::*;
pub use entropy::*;
pub use exchange_limits::*;
pub use fair_price::*;
pub use flow::*;
pub use funding::*;
pub use hjb::*;
pub use kelly::*;
pub use liquidity::*;
pub use margin::*;
pub use regime::*;
pub use stochastic_constraints::*;
pub use volatility::*;
