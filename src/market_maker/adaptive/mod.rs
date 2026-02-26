//! Adaptive Bayesian Market Maker (ABMM) Module
//!
//! This module implements online Bayesian learning for dynamic parameter tuning
//! in market making. Key components:
//!
//! 1. **LearnedSpreadFloor**: Bayesian estimation of break-even spread from AS
//! 2. **ShrinkageGamma**: Log-additive gamma with horseshoe shrinkage prior
//! 3. **BlendedKappa**: Sigmoid blend of book-based and own-fill kappa
//! 4. **FillRateController**: Gamma-Poisson model for fill rate targeting
//! 5. **AdaptiveSpreadCalculator**: Orchestrates all components
//!
//! Mathematical foundations documented in docs/design/adaptive_bayesian_mm.md

mod blended_kappa;
mod calculator;
mod config;
mod fill_controller;
mod learned_floor;
mod shrinkage_gamma;
mod standardizer;

pub use blended_kappa::BlendedKappaEstimator;
pub use calculator::AdaptiveSpreadCalculator;
pub use config::AdaptiveBayesianConfig;
pub use fill_controller::FillRateController;
pub use learned_floor::LearnedSpreadFloor;
pub use shrinkage_gamma::ShrinkageGamma;
pub use standardizer::{
    MultiFeatureStandardizer, MultiFeatureStandardizerDiagnostics, PredictionStandardizer,
    PredictionStandardizerDiagnostics, SignalStandardizer,
};
