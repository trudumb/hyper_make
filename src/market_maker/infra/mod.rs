//! Infrastructure modules.
//!
//! This module contains infrastructure components:
//! - **Reconnection**: WebSocket connection health monitoring
//! - **ConnectionSupervisor**: Proactive connection supervision
//! - **DataQuality**: Market data validation and anomaly detection
//! - **Margin**: Margin-aware position sizing
//! - **ExchangeLimits**: Exchange-enforced position limits from API
//! - **Metrics**: Prometheus metrics for observability
//! - **Executor**: Order execution abstraction
//! - **Recovery**: Stuck position recovery state machine (Phase 3)
//! - **Reconciliation**: Event-driven position sync (Phase 4)
//! - **RateLimit**: Rejection-aware rate limiting (Phase 5)
//! - **Logging**: Multi-stream structured logging (Phase 6)
//! - **Capacity**: Pre-allocation constants for latency optimization
//! - **Arena**: Quote cycle arena allocator for latency optimization
//! - **OrphanTracker**: Prevents false orphan detection during order lifecycle
//! - **ExecutionBudget**: Token-based budget for statistical impulse control

mod arena;
pub mod capacity;
mod connection_supervisor;
mod data_quality;
mod exchange_limits;
mod execution_budget;
mod executor;
mod logging;
mod margin;
mod metrics;
mod orphan_tracker;
mod rate_limit;
mod reconciliation;
mod reconnection;
mod recovery;

pub use arena::*;
pub use connection_supervisor::*;
pub use data_quality::*;
pub use exchange_limits::*;
pub use execution_budget::*;
pub use executor::*;
pub use logging::*;
pub use margin::*;
pub use metrics::*;
pub use orphan_tracker::*;
pub use rate_limit::*;
pub use reconciliation::*;
pub use reconnection::*;
pub use recovery::*;
