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
//! # Benefits
//!
//! - **Single state snapshot**: All monitors evaluate the same point-in-time data
//! - **Extensible**: Add new monitors without modifying aggregator
//! - **Testable**: Monitors can be tested in isolation

mod aggregator;
mod monitor;
pub mod monitors;
mod state;

pub use aggregator::{AggregatedRisk, RiskAggregator};
pub use monitor::{RiskAction, RiskAssessment, RiskMonitor, RiskMonitorBox, RiskSeverity};
pub use state::RiskState;
