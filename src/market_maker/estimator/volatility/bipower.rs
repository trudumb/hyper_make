//! Single-scale bipower variation estimator.
//!
//! Building block for multi-scale volatility estimation.

/// Single-timescale RV/BV tracker - building block for multi-scale estimator.
#[derive(Debug)]
pub(crate) struct SingleScaleBipower {
    /// EWMA decay factor (per tick)
    alpha: f64,
    /// Realized variance (includes jumps): EWMA of r²
    rv: f64,
    /// Bipower variation (excludes jumps): EWMA of (π/2)|r_t||r_{t-1}|
    bv: f64,
    /// Last absolute log return (for BV calculation)
    last_abs_return: Option<f64>,
}

impl SingleScaleBipower {
    pub(crate) fn new(half_life_ticks: f64, default_var: f64) -> Self {
        Self {
            alpha: (2.0_f64.ln() / half_life_ticks).clamp(0.001, 1.0),
            rv: default_var,
            bv: default_var,
            last_abs_return: None,
        }
    }

    pub(crate) fn update(&mut self, log_return: f64) {
        let abs_return = log_return.abs();

        // RV: EWMA of r²
        let rv_obs = log_return.powi(2);
        self.rv = self.alpha * rv_obs + (1.0 - self.alpha) * self.rv;

        // BV: EWMA of (π/2) × |r_t| × |r_{t-1}|
        if let Some(last_abs) = self.last_abs_return {
            let bv_obs = std::f64::consts::FRAC_PI_2 * abs_return * last_abs;
            self.bv = self.alpha * bv_obs + (1.0 - self.alpha) * self.bv;
        }

        self.last_abs_return = Some(abs_return);
    }

    /// Total volatility including jumps (√RV)
    pub(crate) fn sigma_total(&self) -> f64 {
        self.rv.sqrt().clamp(1e-7, 0.05)
    }

    /// Clean volatility excluding jumps (√BV)
    pub(crate) fn sigma_clean(&self) -> f64 {
        self.bv.sqrt().clamp(1e-7, 0.05)
    }

    /// Jump ratio: RV/BV (1.0 = normal, >2 = jumps)
    pub(crate) fn jump_ratio(&self) -> f64 {
        if self.bv > 1e-12 {
            (self.rv / self.bv).clamp(0.1, 100.0)
        } else {
            1.0
        }
    }
}
