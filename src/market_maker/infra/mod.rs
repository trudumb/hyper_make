//! Infrastructure modules.
//!
//! This module contains infrastructure components:
//! - **Reconnection**: WebSocket connection health monitoring
//! - **DataQuality**: Market data validation and anomaly detection
//! - **Margin**: Margin-aware position sizing
//! - **ExchangeLimits**: Exchange-enforced position limits from API
//! - **Metrics**: Prometheus metrics for observability
//! - **Executor**: Order execution abstraction
//! - **Recovery**: Stuck position recovery state machine (Phase 3)
//! - **Reconciliation**: Event-driven position sync (Phase 4)
//! - **RateLimit**: Rejection-aware rate limiting (Phase 5)
//! - **Logging**: Multi-stream structured logging (Phase 6)

mod data_quality;
mod exchange_limits;
mod executor;
mod logging;
mod margin;
mod metrics;
mod rate_limit;
mod reconciliation;
mod reconnection;
mod recovery;

pub use data_quality::*;
pub use exchange_limits::*;
pub use executor::*;
pub use logging::*;
pub use margin::*;
pub use metrics::*;
pub use rate_limit::*;
pub use reconciliation::*;
pub use reconnection::*;
pub use recovery::*;
