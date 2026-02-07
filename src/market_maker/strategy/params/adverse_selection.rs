//! Adverse selection parameter set.

use crate::market_maker::adverse_selection::DepthDecayAS;

/// Adverse selection measurement and adjustment.
#[derive(Debug, Clone, Default)]
pub struct AdverseSelectionParams {
    /// Spread adjustment from AS estimator (as fraction of mid price).
    /// Add to half-spread to compensate for informed flow.
    pub as_spread_adjustment: f64,

    /// Predicted alpha: P(next trade is informed) in [0, 1].
    pub predicted_alpha: f64,

    /// Is AS estimator warmed up with enough fills?
    pub as_warmed_up: bool,

    /// Depth-dependent AS model (calibrated from fills).
    /// AS(δ) = AS₀ × exp(-δ/δ_char).
    pub depth_decay_as: Option<DepthDecayAS>,
}
