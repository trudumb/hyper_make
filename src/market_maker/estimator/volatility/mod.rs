//! Volatility estimation components.
//!
//! This module provides volatility estimation with multiple approaches:
//!
//! - `bipower`: Single-scale bipower variation (RV/BV tracker)
//! - `multi_scale`: Multi-timescale volatility (fast/medium/slow components)
//! - `regime`: 4-state volatility regime classification with hysteresis
//! - `stochastic`: Heston-style stochastic volatility parameters
//! - `incremental`: O(1) incremental statistics for calibration
//!
//! # Design
//!
//! - Bipower variation separates continuous volatility from jumps
//! - Multi-scale blending adapts to market acceleration
//! - Regime tracking with asymmetric hysteresis (fast escalation, slow de-escalation)
//! - O(1) incremental calibration instead of O(n²) batch computation

mod bipower;
mod incremental;
mod multi_scale;
mod regime;
mod stochastic;

// Re-export pub(crate) types for internal use within the crate
pub(crate) use multi_scale::MultiScaleBipowerEstimator;
pub(crate) use regime::VolatilityRegimeTracker;

// Test-only re-export (used by parameter_estimator tests)
#[cfg(test)]
pub(crate) use bipower::SingleScaleBipower;

// Re-export public types for external access
pub use regime::VolatilityRegime;
pub use regime::{
    BlendedParameters, RegimeBeliefState, RegimeParameterBlender, RegimeParameterConfig,
};
pub use stochastic::StochasticVolParams;
