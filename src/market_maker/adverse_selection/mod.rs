//! Adverse selection measurement and prediction for market making.
//!
//! Measures the ground truth adverse selection cost (E[Δp | fill]) and provides
//! spread adjustment recommendations to compensate for informed flow.
//!
//! Key components:
//! - Ground truth measurement: Track price movement 1 second after each fill
//! - Realized AS tracking: EWMA of signed price movements conditional on fills
//! - Predicted α(t): Probability that next trade is informed (from signals)
//! - Spread adjustment: Recommended spread widening based on realized AS
//! - **Depth-Dependent AS**: Exponential decay model AS(δ) = AS₀ × exp(-δ/δ_char)
//!   calibrated from fill history by depth bucket
//! - **Pre-Fill Classifier**: Predicts toxicity BEFORE fills using microstructure signals
//!
//! # Module Structure
//!
//! - `depth_decay`: Depth-dependent adverse selection model
//! - `estimator`: Main adverse selection estimator (post-fill measurement)
//! - `pre_fill_classifier`: Pre-fill toxicity prediction

pub mod book_dynamics;
mod cancel_race;
mod depth_decay;
mod enhanced_classifier;
mod estimator;
pub mod microstructure_features;
mod pre_fill_classifier;
pub mod toxicity_regime;

pub use book_dynamics::{BookDynamicsTracker, IcebergDetector, SweepDetector};
pub use cancel_race::{CancelRaceConfig, CancelRaceSummary, CancelRaceTracker};
pub use depth_decay::{DepthDecayAS, DepthDecayASSummary, FillWithDepth};
pub use enhanced_classifier::{
    EnhancedASClassifier, EnhancedClassifierConfig, EnhancedClassifierDiagnostics,
};
pub use estimator::{AdverseSelectionEstimator, AdverseSelectionSummary};
pub use microstructure_features::{
    MicrostructureConfig, MicrostructureDiagnostics, MicrostructureExtractor,
    MicrostructureFeatures, TradeObservation,
};
pub use pre_fill_classifier::{PreFillASClassifier, PreFillClassifierConfig, PreFillSummary};
pub use toxicity_regime::{OfiAccelerationSignals, ToxicityRegime, ToxicityThresholds};

/// Configuration for adverse selection estimation.
#[derive(Debug, Clone)]
pub struct AdverseSelectionConfig {
    /// Horizon for measuring price impact after fill (milliseconds)
    /// Default: 1000ms (1 second)
    pub measurement_horizon_ms: u64,

    /// EWMA decay factor for realized AS (0.0 = no decay, 1.0 = instant decay)
    /// Default: 0.05 (20-trade half-life)
    pub ewma_alpha: f64,

    /// Minimum fills required before AS estimates are valid
    /// Default: 50
    pub min_fills_warmup: usize,

    /// Maximum pending fills to track (memory bound)
    /// Default: 1000
    pub max_pending_fills: usize,

    /// Maximum spread adjustment (as fraction of mid price)
    /// Default: 0.005 (0.5%)
    pub max_spread_adjustment: f64,

    /// Minimum spread adjustment (floor)
    /// Default: 0.0 (no floor)
    pub min_spread_adjustment: f64,

    // === Alpha Prediction Weights ===
    /// Weight for volatility surprise signal in α prediction
    pub alpha_volatility_weight: f64,
    /// Weight for flow imbalance magnitude signal in α prediction
    pub alpha_flow_weight: f64,
    /// Weight for jump ratio signal in α prediction
    pub alpha_jump_weight: f64,
}

impl Default for AdverseSelectionConfig {
    fn default() -> Self {
        Self {
            // MAINNET OPTIMIZED: Shorter horizon for liquid markets with faster mean reversion
            measurement_horizon_ms: 500, // Reduced from 1000ms
            // MAINNET OPTIMIZED: Faster adaptation to changing AS patterns
            ewma_alpha: 0.12,     // Increased from 0.05 - faster adaptation
            min_fills_warmup: 50, // Increased from 20 — need statistically meaningful sample
            // Increased from 1000 to handle 10-minute tracking window.
            // At ~2 fills/sec peak, 10 min = ~1200 fills. Use 5000 for safety margin.
            max_pending_fills: 5000,
            max_spread_adjustment: 0.005,
            min_spread_adjustment: 0.0,
            // Alpha prediction weights (simple linear combination)
            alpha_volatility_weight: 0.3,
            alpha_flow_weight: 0.4,
            alpha_jump_weight: 0.3,
        }
    }
}
