//! Risk monitor implementations.
//!
//! Each monitor evaluates one aspect of risk:
//! - `LossMonitor`: Daily loss limit
//! - `DrawdownMonitor`: Peak-to-trough drawdown
//! - `PositionMonitor`: Position size and value limits
//! - `DataStalenessMonitor`: Market data freshness
//! - `CascadeMonitor`: Liquidation cascade detection
//! - `RateLimitMonitor`: Exchange rate limit errors

mod cascade;
mod data_staleness;
mod drawdown;
mod loss;
mod position;
mod rate_limit;

pub use cascade::CascadeMonitor;
pub use data_staleness::DataStalenessMonitor;
pub use drawdown::DrawdownMonitor;
pub use loss::LossMonitor;
pub use position::PositionMonitor;
pub use rate_limit::RateLimitMonitor;
