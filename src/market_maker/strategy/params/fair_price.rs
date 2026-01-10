//! Fair price parameter set.

/// Microprice and fair value estimation.
#[derive(Debug, Clone, Default)]
pub struct FairPriceParams {
    /// Microprice - fair price incorporating signal predictions.
    /// Quote around this instead of raw mid.
    pub microprice: f64,

    /// β_book coefficient (return prediction per unit book imbalance).
    pub beta_book: f64,

    /// β_flow coefficient (return prediction per unit flow imbalance).
    pub beta_flow: f64,

    // Kalman filter (stochastic integration)
    /// Whether to use Kalman filter spread widening.
    pub use_kalman_filter: bool,
    /// Kalman-filtered fair price (denoised mid).
    pub kalman_fair_price: f64,
    /// Kalman filter uncertainty (posterior std dev).
    pub kalman_uncertainty: f64,
    /// Kalman-based spread widening: γ × σ_kalman × √T.
    pub kalman_spread_widening: f64,
    /// Whether Kalman filter is warmed up.
    pub kalman_warmed_up: bool,
}
