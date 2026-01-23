//! Monitoring and alerting module for market making operations.
//!
//! This module provides:
//! - **DashboardState**: Terminal-friendly real-time dashboard state
//! - **Alerter**: Thread-safe alerting system with deduplication
//!
//! # Architecture
//!
//! ```text
//! +------------------+     +------------------+
//! |  DashboardState  |     |     Alerter      |
//! +------------------+     +------------------+
//! | PnL metrics      |     | Alert config     |
//! | Position state   |     | Alert history    |
//! | Market regime    |     | Deduplication    |
//! | Execution stats  |     | Handlers         |
//! | Calibration IRs  |     +------------------+
//! +------------------+
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::market_maker::monitoring::{DashboardState, Alerter, AlertConfig};
//!
//! // Create dashboard state
//! let mut dashboard = DashboardState::new();
//! dashboard.update_pnl(100.0, 0.5);
//! dashboard.update_position(1.5, 75000.0);
//!
//! // Create alerter
//! let alerter = Alerter::new(AlertConfig::default(), 1000);
//!
//! // Check thresholds
//! if let Some(alert) = alerter.check_drawdown(0.015, now()) {
//!     alerter.add_alert(alert);
//! }
//!
//! // Get ASCII dashboard
//! println!("{}", dashboard.format_summary());
//! ```

mod alerter;
mod dashboard;

pub use alerter::{
    Alert, AlertConfig, AlertHandler, AlertSeverity, AlertType, Alerter, LoggingAlertHandler,
};
pub use dashboard::{DashboardState, PositionSide};
