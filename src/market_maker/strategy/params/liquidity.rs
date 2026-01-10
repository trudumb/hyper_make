//! Liquidity parameter set.

/// Order book depth and liquidity metrics.
#[derive(Debug, Clone)]
pub struct LiquidityParams {
    /// Estimated order book depth decay (κ) - from weighted L2 book regression.
    pub kappa: f64,

    /// Directional kappa for bid side (our buy fills).
    /// When informed flow is selling, κ_bid < κ_ask → wider bid spread.
    pub kappa_bid: f64,

    /// Directional kappa for ask side (our sell fills).
    /// When informed flow is buying, κ_ask < κ_bid → wider ask spread.
    pub kappa_ask: f64,

    /// Whether fill distance distribution is heavy-tailed (CV > 1.2).
    pub is_heavy_tailed: bool,

    /// Coefficient of variation of fill distances (CV = σ/μ).
    pub kappa_cv: f64,

    /// Order arrival intensity (A) - volume ticks per second.
    pub arrival_intensity: f64,

    /// Liquidity-based gamma multiplier [1.0, 2.0].
    /// > 1.0 when near-touch liquidity is below average (thin book).
    pub liquidity_gamma_mult: f64,

    // Queue model parameters (Gap 3)
    /// Calibrated volume at touch rate (units/sec).
    pub calibrated_volume_rate: f64,
    /// Calibrated cancel rate (fraction/sec).
    pub calibrated_cancel_rate: f64,
}

impl Default for LiquidityParams {
    fn default() -> Self {
        Self {
            kappa: 100.0,
            kappa_bid: 100.0,
            kappa_ask: 100.0,
            is_heavy_tailed: false,
            kappa_cv: 1.0,
            arrival_intensity: 0.5,
            liquidity_gamma_mult: 1.0,
            calibrated_volume_rate: 1.0,
            calibrated_cancel_rate: 0.2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_liquidity_params_default() {
        let params = LiquidityParams::default();
        assert!((params.kappa - 100.0).abs() < f64::EPSILON);
        assert!((params.liquidity_gamma_mult - 1.0).abs() < f64::EPSILON);
    }
}
