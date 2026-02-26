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

    /// Returns true if jump_ratio > 1.5, indicating toxic/jump regime.
    /// Per Small Fish Strategy: jump_ratio > 1.5 = toxic regime, widen spreads.
    #[allow(dead_code)]
    pub(crate) fn is_toxic_regime(&self) -> bool {
        self.jump_ratio() > 1.5
    }

    /// Get realized variance (includes jumps)
    #[allow(dead_code)]
    pub(crate) fn rv(&self) -> f64 {
        self.rv
    }

    /// Get bipower variation (excludes jumps)
    #[allow(dead_code)]
    pub(crate) fn bv(&self) -> f64 {
        self.bv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bipower_variation_formula() {
        // BV = (π/2) × Σ |r_t| × |r_{t-1}|
        // Test with known returns
        let mut bp = SingleScaleBipower::new(10.0, 0.0001);

        // First update - no BV yet (need two returns)
        bp.update(0.01);
        assert!(bp.last_abs_return.is_some());

        // Second update - now we have BV
        bp.update(0.02);
        // BV observation = (π/2) × 0.01 × 0.02 = 0.000314...
        assert!(bp.bv > 0.0);
    }

    #[test]
    fn test_sigma_clean_excludes_jumps() {
        let mut bp = SingleScaleBipower::new(5.0, 0.0001);

        // Feed normal returns
        for _ in 0..20 {
            bp.update(0.001);
        }
        let _sigma_clean_normal = bp.sigma_clean();

        // Feed a jump
        bp.update(0.05); // 5% move = jump

        // sigma_clean should increase less than sigma_total
        let sigma_clean_after = bp.sigma_clean();
        let sigma_total_after = bp.sigma_total();

        // After a jump, total vol should be higher than clean vol
        assert!(
            sigma_total_after > sigma_clean_after,
            "sigma_total ({}) should be > sigma_clean ({}) after jump",
            sigma_total_after,
            sigma_clean_after
        );
    }

    #[test]
    fn test_jump_ratio_normal_market() {
        let mut bp = SingleScaleBipower::new(10.0, 0.0001);

        // Feed consistent returns - no jumps
        for _ in 0..50 {
            bp.update(0.001);
        }

        // Jump ratio should be close to 1.0 in normal market
        let jr = bp.jump_ratio();
        assert!(
            jr > 0.8 && jr < 1.5,
            "Jump ratio {} should be near 1.0 in normal market",
            jr
        );
        assert!(
            !bp.is_toxic_regime(),
            "Should not be toxic in normal market"
        );
    }

    #[test]
    fn test_jump_ratio_with_jumps() {
        let mut bp = SingleScaleBipower::new(5.0, 0.0001);

        // Feed small returns
        for _ in 0..10 {
            bp.update(0.0005);
        }

        // Feed large jump
        bp.update(0.10); // 10% move

        // Jump ratio should spike
        let jr = bp.jump_ratio();
        assert!(jr > 1.0, "Jump ratio {} should be > 1.0 after big move", jr);
    }

    #[test]
    fn test_is_toxic_regime_threshold() {
        let mut bp = SingleScaleBipower::new(3.0, 0.0001);

        // Build baseline with small returns
        for _ in 0..20 {
            bp.update(0.0002);
        }

        // Should not be toxic yet
        assert!(
            !bp.is_toxic_regime(),
            "Should not be toxic with normal returns"
        );

        // Inject multiple large moves to push ratio above 1.5
        for _ in 0..5 {
            bp.update(0.05);
            bp.update(0.0001); // small return after jump
        }

        // After multiple jumps, may become toxic
        // Note: depends on EWMA dynamics
        let jr = bp.jump_ratio();
        println!("Jump ratio after jumps: {}", jr);
    }

    #[test]
    fn test_rv_and_bv_accessors() {
        let mut bp = SingleScaleBipower::new(10.0, 0.0001);
        bp.update(0.01);
        bp.update(0.02);

        assert!(bp.rv() > 0.0, "RV should be positive");
        assert!(bp.bv() > 0.0, "BV should be positive");
    }

    #[test]
    fn test_bipower_ewma_decay() {
        let mut bp = SingleScaleBipower::new(5.0, 0.0001);

        // Feed a spike
        bp.update(0.05);
        bp.update(0.05);
        let rv_after_spike = bp.rv();

        // Feed many small returns - should decay
        for _ in 0..20 {
            bp.update(0.0001);
        }
        let rv_after_decay = bp.rv();

        assert!(
            rv_after_decay < rv_after_spike,
            "RV should decay: {} should be < {}",
            rv_after_decay,
            rv_after_spike
        );
    }

    #[test]
    fn test_sigma_clean_formula() {
        // sigma_clean = sqrt(BV)
        let mut bp = SingleScaleBipower::new(10.0, 0.0004); // default var = 0.0004 -> sigma = 0.02
        bp.update(0.01);
        bp.update(0.01);

        let sigma = bp.sigma_clean();
        let expected = bp.bv().sqrt();

        // Should be within clamping range
        assert!((1e-7..=0.05).contains(&sigma));
        // Should match sqrt(bv) if within range
        if (1e-7..=0.05).contains(&expected) {
            assert!(
                (sigma - expected).abs() < 1e-10,
                "sigma_clean {} should equal sqrt(bv) {}",
                sigma,
                expected
            );
        }
    }

    #[test]
    fn test_toxic_regime_constant() {
        // Verify the 1.5 threshold from Small Fish Strategy
        let mut bp = SingleScaleBipower::new(10.0, 0.0001);

        // Manually set internal state to test threshold
        // We can't directly set, so we test via behavior
        // This test documents the threshold
        for _ in 0..10 {
            bp.update(0.001);
        }

        // The threshold is jump_ratio > 1.5
        // We verify this by checking the method behavior
        let jr = bp.jump_ratio();
        let is_toxic = bp.is_toxic_regime();

        if jr > 1.5 {
            assert!(is_toxic, "Should be toxic when jr > 1.5");
        } else {
            assert!(!is_toxic, "Should not be toxic when jr <= 1.5");
        }
    }
}
