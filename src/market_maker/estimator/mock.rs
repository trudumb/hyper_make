//! Mock estimator for testing.

use super::{MarketEstimator, VolatilityRegime};

/// Default mock estimator for testing.
///
/// Returns neutral/safe values for all parameters.
#[derive(Debug, Clone, Default)]
pub struct MockEstimator {
    pub sigma: f64,
    pub kappa: f64,
    pub is_warmed: bool,
}

impl MockEstimator {
    /// Create a mock estimator with default safe values.
    pub fn new() -> Self {
        Self {
            sigma: 0.0002, // 2 bps/sec
            kappa: 500.0,  // Moderate fill rate
            is_warmed: true,
        }
    }

    /// Create with custom sigma and kappa.
    pub fn with_params(sigma: f64, kappa: f64) -> Self {
        Self {
            sigma,
            kappa,
            is_warmed: true,
        }
    }
}

impl MarketEstimator for MockEstimator {
    fn sigma_clean(&self) -> f64 {
        self.sigma
    }
    fn sigma_total(&self) -> f64 {
        self.sigma
    }
    fn sigma_effective(&self) -> f64 {
        self.sigma
    }
    fn sigma_leverage_adjusted(&self) -> f64 {
        self.sigma // Mock returns same as sigma (no leverage effect)
    }
    fn volatility_regime(&self) -> VolatilityRegime {
        VolatilityRegime::Normal
    }
    fn kappa(&self) -> f64 {
        self.kappa
    }
    fn kappa_bid(&self) -> f64 {
        self.kappa
    }
    fn kappa_ask(&self) -> f64 {
        self.kappa
    }
    fn is_heavy_tailed(&self) -> bool {
        false // Mock assumes exponential tails
    }
    fn kappa_cv(&self) -> f64 {
        1.0 // CV=1 for exponential
    }
    fn arrival_intensity(&self) -> f64 {
        1.0
    }
    fn liquidity_gamma_multiplier(&self) -> f64 {
        1.0
    }
    fn is_toxic_regime(&self) -> bool {
        false
    }
    fn jump_ratio(&self) -> f64 {
        1.0
    }
    fn momentum_bps(&self) -> f64 {
        0.0
    }
    fn flow_imbalance(&self) -> f64 {
        0.0
    }
    fn falling_knife_score(&self) -> f64 {
        0.0
    }
    fn rising_knife_score(&self) -> f64 {
        0.0
    }
    fn momentum_continuation_probability(&self) -> f64 {
        0.5 // 50% prior - no learned data
    }
    fn bid_protection_factor(&self) -> f64 {
        1.0 // No protection needed
    }
    fn ask_protection_factor(&self) -> f64 {
        1.0 // No protection needed
    }
    fn momentum_strength(&self) -> f64 {
        0.0 // No momentum
    }
    fn momentum_model_calibrated(&self) -> bool {
        false // Not calibrated
    }
    fn book_imbalance(&self) -> f64 {
        0.0
    }
    fn microprice(&self) -> f64 {
        0.0
    }
    fn beta_book(&self) -> f64 {
        0.0
    }
    fn beta_flow(&self) -> f64 {
        0.0
    }
    fn lambda_jump(&self) -> f64 {
        0.01
    }
    fn mu_jump(&self) -> f64 {
        0.0
    }
    fn sigma_jump(&self) -> f64 {
        0.001
    }
    fn kappa_vol(&self) -> f64 {
        0.5
    }
    fn theta_vol_sigma(&self) -> f64 {
        self.sigma
    }
    fn xi_vol(&self) -> f64 {
        0.1
    }
    fn rho_price_vol(&self) -> f64 {
        -0.5
    }
    fn is_warmed_up(&self) -> bool {
        self.is_warmed
    }
    fn sigma_confidence(&self) -> f64 {
        if self.is_warmed {
            1.0
        } else {
            0.0
        }
    }

    // =========================================================================
    // New Latent State Estimators (Phases 2-7) - Mock implementations
    // =========================================================================

    // --- Particle Filter Volatility (Phase 2) ---
    fn sigma_particle_filter(&self) -> f64 {
        self.sigma * 10_000.0 // Convert to bps
    }

    fn sigma_credible_interval(&self, _level: f64) -> (f64, f64) {
        let sigma_bps = self.sigma * 10_000.0;
        (sigma_bps * 0.8, sigma_bps * 1.2) // Mock 80-120% range
    }

    fn regime_probabilities(&self) -> [f64; 4] {
        [0.1, 0.7, 0.15, 0.05] // Mock: mostly Normal regime
    }

    // --- Informed Flow Model (Phase 3) ---
    fn p_informed(&self) -> f64 {
        0.05 // 5% informed baseline
    }

    fn p_noise(&self) -> f64 {
        0.90 // 90% noise
    }

    fn p_forced(&self) -> f64 {
        0.05 // 5% forced
    }

    fn flow_decomposition_confidence(&self) -> f64 {
        0.5 // Medium confidence
    }

    // --- Fill Rate Model (Phase 4) ---
    fn fill_rate_at_depth(&self, depth_bps: f64) -> f64 {
        // Simple exponential decay: λ(d) = exp(-d / 10)
        (-depth_bps / 10.0).exp()
    }

    fn optimal_depth_for_fill_rate(&self, target_rate: f64) -> f64 {
        // Inverse of fill_rate_at_depth
        -10.0 * target_rate.ln().max(-10.0)
    }

    // --- Adverse Selection Decomposition (Phase 5) ---
    fn as_permanent_bps(&self) -> f64 {
        1.0 // 1 bps permanent
    }

    fn as_temporary_bps(&self) -> f64 {
        0.5 // 0.5 bps temporary
    }

    fn as_timing_bps(&self) -> f64 {
        0.5 // 0.5 bps timing
    }

    fn total_as_bps(&self) -> f64 {
        2.0 // 2 bps total
    }

    // --- Edge Surface (Phase 6) ---
    fn current_edge_bps(&self) -> f64 {
        2.0 // 2 bps expected edge
    }

    fn should_quote_edge(&self) -> bool {
        true // Always quote in mock
    }

    // --- Joint Dynamics (Phase 7) ---
    fn is_toxic_joint(&self) -> bool {
        false // Never toxic in mock
    }

    fn sigma_kappa_correlation(&self) -> f64 {
        -0.3 // Typical negative correlation
    }
}
