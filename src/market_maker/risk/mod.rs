//! Unified risk management system.
//!
//! This module provides a single source of truth for risk state and evaluation.
//!
//! # Architecture
//!
//! ```text
//! +------------------+
//! |    RiskState     |  <- Unified snapshot of all risk-relevant data
//! +------------------+
//!          |
//!          v
//! +------------------+
//! |  RiskAggregator  |  <- Evaluates all monitors
//! +------------------+
//!     |    |    |
//!     v    v    v
//! +------+ +------+ +------+
//! |Loss  | |Pos.  | |Cascade|  <- RiskMonitor implementations
//! +------+ +------+ +------+
//! ```
//!
//! # Components
//!
//! - [`RiskState`]: Unified snapshot of all risk-relevant data
//! - [`RiskAggregator`]: Evaluates all monitors against risk state
//! - [`RiskMonitor`]: Trait for implementing risk monitors
//! - [`RiskLimits`] / [`RiskChecker`]: Position and order size limit checking
//! - [`CircuitBreakerMonitor`]: Market condition circuit breakers
//! - [`DrawdownTracker`]: Equity drawdown tracking and position sizing
//! - [`KillSwitch`]: Emergency shutdown mechanism
//!
//! # Benefits
//!
//! - **Single state snapshot**: All monitors evaluate the same point-in-time data
//! - **Extensible**: Add new monitors without modifying aggregator
//! - **Testable**: Monitors can be tested in isolation
//! - **Defense-first**: Defaults to safer options when uncertain

mod aggregator;
mod circuit_breaker;
mod drawdown;
mod kill_switch;
mod limits;
mod monitor;
pub mod monitors;
mod state;

pub use aggregator::{AggregatedRisk, RiskAggregator};
pub use circuit_breaker::{
    CircuitBreakerAction, CircuitBreakerConfig, CircuitBreakerMonitor, CircuitBreakerType,
};
pub use drawdown::{DrawdownConfig, DrawdownLevel, DrawdownSummary, DrawdownTracker};
pub use kill_switch::*;
pub use limits::{RiskCheckResult, RiskChecker, RiskLimits};
pub use monitor::{RiskAction, RiskAssessment, RiskMonitor, RiskMonitorBox, RiskSeverity};
pub use state::RiskState;
