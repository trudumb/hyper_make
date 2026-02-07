//! Analytics module for performance measurement, signal attribution, and edge validation.
//!
//! Provides:
//! - **SharpeTracker**: Return-based Sharpe ratio computation with rolling windows
//! - **Attribution**: Per-signal PnL attribution via active/inactive conditional analysis
//! - **EdgeMetrics**: Predicted vs realized edge tracking
//! - **Persistence**: JSONL file logging for offline analysis

pub mod attribution;
pub mod edge_metrics;
pub mod persistence;
pub mod sharpe;

pub use attribution::{CycleContributions, SignalContribution, SignalPnLAttributor};
pub use edge_metrics::{EdgeSnapshot, EdgeTracker};
pub use persistence::AnalyticsLogger;
pub use sharpe::{PerSignalSharpeTracker, SharpeSummary, SharpeTracker};
