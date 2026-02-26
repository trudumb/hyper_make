//! Process models for market dynamics.
//!
//! This module contains stochastic process estimators and models:
//! - **Hawkes**: Self-exciting order flow intensity
//! - **Spread**: Bid-ask spread dynamics
//! - **Funding**: Funding rate estimation
//! - **HJB Control**: Hamilton-Jacobi-Bellman inventory control
//! - **Liquidation**: Cascade detection

mod funding;
mod hawkes;
mod hjb;
mod liquidation;
mod spread;

pub use funding::*;
pub use hawkes::*;
pub use hjb::*;
pub use liquidation::*;
pub use spread::*;
