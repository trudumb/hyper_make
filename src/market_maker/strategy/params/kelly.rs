//! Kelly-Stochastic parameter set.

/// Kelly-Stochastic config parameters (for MarketParams extraction).
///
/// This is a view type for extracting Kelly parameters from MarketParams.
#[derive(Debug, Clone)]
pub struct KellyStochasticConfigParams {
    /// Informed trader probability at the touch (0.0-1.0).
    pub alpha_touch: f64,

    /// Characteristic depth for alpha decay in bps.
    /// α(δ) = α_touch × exp(-δ/alpha_decay_bps).
    pub alpha_decay_bps: f64,

    /// Kelly fraction (0.25 = quarter Kelly).
    pub kelly_fraction: f64,

    /// Kelly-specific time horizon for first-passage probability (seconds).
    /// Semantically different from GLFT inventory time horizon.
    pub time_horizon: f64,
}

impl Default for KellyStochasticConfigParams {
    fn default() -> Self {
        Self {
            alpha_touch: 0.15,
            alpha_decay_bps: 10.0,
            kelly_fraction: 0.25,
            time_horizon: 60.0,
        }
    }
}
