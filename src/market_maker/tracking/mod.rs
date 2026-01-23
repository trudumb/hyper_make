//! State tracking modules.
//!
//! This module contains components for tracking trading state:
//! - **OrderManager**: Tracks resting orders and their lifecycle
//! - **Position**: Position state tracking
//! - **Queue**: Queue position tracking for fill probability
//! - **PnL**: Profit and loss attribution
//! - **WsOrderState**: WebSocket-based order state management (new)
//! - **AssetId**: Unique identifier for assets (multi-asset support)
//! - **Calibration**: Prediction calibration tracking for model validation

mod asset_id;
pub mod calibration;
mod order_manager;
mod pnl;
mod position;
mod queue;
pub mod signal_decay;
pub mod ws_order_state;

pub use asset_id::AssetId;
pub use calibration::{
    CalibrationConfig, CalibrationMetrics, CalibrationSummary, CalibrationTracker,
    // Small Fish Strategy prediction logging infrastructure
    LinkedPredictionOutcome, OutcomeLog, PredictionLog, PredictionOutcomeStore, PredictionType,
    // Small Fish Strategy signal MI (Mutual Information) tracking
    SignalMiTracker,
};
pub use order_manager::*;
pub use pnl::*;
pub use position::*;
pub use queue::*;
pub use signal_decay::{AlertSeverity, SignalAlert, SignalDecayConfig, SignalDecayReport, SignalDecayTracker};
pub use ws_order_state::{WsOrderSpec, WsOrderStateConfig, WsOrderStateManager};
