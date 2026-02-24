//! Paper Trading Simulation Infrastructure
//!
//! This module provides comprehensive simulation capabilities for the market maker,
//! enabling:
//! - Quote generation tracking without real order submission
//! - Fill simulation based on market data
//! - Prediction logging for calibration analysis
//! - Outcome attribution for PnL decomposition
//! - Quick Monte Carlo EV simulation for proactive quoting
//!
//! # Architecture
//!
//! ```text
//! Market Data → Quote Engine → SimulationExecutor → PredictionLogger
//!                                    ↓
//!                              FillSimulator ← Market Trades
//!                                    ↓
//!                              OutcomeTracker → CalibrationMetrics
//! ```
//!
//! # Usage
//!
//! Instead of using `HyperliquidExecutor`, inject `SimulationExecutor` to run
//! in paper trading mode. All quotes are logged but not submitted to the exchange.

pub mod calibration;
pub mod cma_es;
pub mod executor;
pub mod fill_sim;
pub mod latency_model;
pub mod outcome;
pub mod prediction;
pub mod quick_mc;
pub mod replay;
pub mod shadow_buffer;
pub mod shadow_tuner;

pub use calibration::{
    BrierDecomposition, CalibrationAnalyzer, CalibrationCurve, ConditionalSlice,
};
pub use cma_es::{CmaEsOptimizer, ParamBound};
pub use executor::SimulationExecutor;
pub use fill_sim::{FillSimulator, FillSimulatorConfig, QueuePositionEstimator, SimulatedFill};
pub use latency_model::{LatencyDistribution, LatencyModel};
pub use outcome::{CycleAttribution, OutcomeTracker, PnLDecomposition};
pub use prediction::{
    FillOutcome, LevelPrediction, MarketStateSnapshot, ModelPredictions, ObservedOutcomes,
    PredictionLogger, PredictionRecord,
};
pub use quick_mc::{MCSimulationResult, QuickMCConfig, QuickMCSimulator};
pub use replay::{ReplayConfig, ReplayEngine, ReplayEvent, ReplayFill, ReplayReport};
pub use shadow_buffer::{create_shadow_buffer, ShadowBufferConsumer, ShadowBufferProducer};
pub use shadow_tuner::{ShadowTuner, ShadowTunerCheckpoint, SharedEstimators, TunerConfig};
