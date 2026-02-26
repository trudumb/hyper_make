//! Funding parameter set.

/// Perpetual funding rate metrics.
#[derive(Debug, Clone, Default)]
pub struct FundingParams {
    /// Current funding rate (annualized).
    pub funding_rate: f64,

    /// Predicted funding cost for holding period.
    pub predicted_funding_cost: f64,

    /// Mark-index premium.
    pub premium: f64,

    /// Premium alpha signal.
    pub premium_alpha: f64,
}
