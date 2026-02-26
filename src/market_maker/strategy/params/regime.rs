//! Regime parameter set.

use crate::market_maker::process_models::SpreadRegime;

/// Market regime detection signals.
#[derive(Debug, Clone)]
pub struct RegimeParams {
    /// Whether market is in toxic (jump) regime: RV/BV > 1.5.
    pub is_toxic_regime: bool,

    /// RV/BV jump ratio: ≈1.0 = normal diffusion, >1.5 = toxic.
    pub jump_ratio: f64,

    /// Spread regime (Tight/Normal/Wide).
    pub spread_regime: SpreadRegime,

    /// Spread percentile [0, 1].
    pub spread_percentile: f64,

    /// Fair spread from vol-adjusted model.
    pub fair_spread: f64,

    // Jump-diffusion parameters (Gap 1)
    /// Jump intensity λ (expected jumps per second).
    pub lambda_jump: f64,
    /// Mean jump size μ_j.
    pub mu_jump: f64,
    /// Jump size standard deviation σ_j.
    pub sigma_jump: f64,
}

impl Default for RegimeParams {
    fn default() -> Self {
        Self {
            is_toxic_regime: false,
            jump_ratio: 1.0,
            spread_regime: SpreadRegime::Normal,
            spread_percentile: 0.5,
            fair_spread: 0.0,
            lambda_jump: 0.0,
            mu_jump: 0.0,
            sigma_jump: 0.0001,
        }
    }
}
