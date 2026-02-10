//! Risk monitor implementations.
//!
//! Each monitor evaluates one aspect of risk:
//! - `LossMonitor`: Daily loss limit
//! - `DrawdownMonitor`: Peak-to-trough drawdown
//! - `PositionMonitor`: Position size and value limits
//! - `PositionVelocityMonitor`: Rapid position accumulation / whipsaws
//! - `DataStalenessMonitor`: Market data freshness
//! - `CascadeMonitor`: Liquidation cascade detection
//! - `PriceVelocityMonitor`: Flash crash / price velocity detection
//! - `RateLimitMonitor`: Exchange rate limit errors

mod cascade;
mod data_staleness;
mod drawdown;
mod loss;
mod position;
mod position_velocity;
mod price_velocity;
mod rate_limit;

pub use cascade::CascadeMonitor;
pub use data_staleness::DataStalenessMonitor;
pub use drawdown::DrawdownMonitor;
pub use loss::LossMonitor;
pub use position::PositionMonitor;
pub use position_velocity::PositionVelocityMonitor;
pub use price_velocity::PriceVelocityMonitor;
pub use rate_limit::RateLimitMonitor;
