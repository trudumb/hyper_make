//! Infrastructure modules.
//!
//! This module contains infrastructure components:
//! - **Reconnection**: WebSocket connection health monitoring
//! - **DataQuality**: Market data validation and anomaly detection
//! - **Margin**: Margin-aware position sizing
//! - **ExchangeLimits**: Exchange-enforced position limits from API
//! - **Metrics**: Prometheus metrics for observability
//! - **Executor**: Order execution abstraction

mod data_quality;
mod exchange_limits;
mod executor;
mod margin;
mod metrics;
mod reconnection;

pub use data_quality::*;
pub use exchange_limits::*;
pub use executor::*;
pub use margin::*;
pub use metrics::*;
pub use reconnection::*;
