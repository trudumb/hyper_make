//! Analytics module for performance measurement, signal attribution, and edge validation.
//!
//! Provides:
//! - **SharpeTracker**: Return-based Sharpe ratio computation with rolling windows
//! - **Attribution**: Per-signal PnL attribution via active/inactive conditional analysis
//! - **EdgeMetrics**: Predicted vs realized edge tracking
//! - **Persistence**: JSONL file logging for offline analysis
//! - **LiveAnalytics**: Bundled analytics for the live market maker

pub mod attribution;
pub mod drift_calibration;
pub mod edge_metrics;
pub mod live;
pub mod market_toxicity;
pub mod persistence;
pub mod sharpe;

pub use drift_calibration::DriftCalibrationTracker;
pub use market_toxicity::{
    MarketToxicityComposite, MarketToxicityConfig, ToxicityAssessment, ToxicityComponents,
    ToxicityInput,
};
pub use attribution::{CycleContributions, SignalContribution, SignalPnLAttributor};
pub use edge_metrics::{EdgeSnapshot, EdgeTracker};
pub use live::{LiveAnalytics, LiveAnalyticsSummary};
pub use persistence::AnalyticsLogger;
pub use sharpe::{PerSignalSharpeTracker, SharpeSummary, SharpeTracker};
