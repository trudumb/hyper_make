//! Echo estimator for distinguishing self-impact from genuine market signals.
//!
//! On thin DEX venues (HIP-3), our own fills dominate the tape. The trend detector
//! then picks up our fills as "market moves", creating a feedback loop where:
//! 1. We fill → price moves (our impact)
//! 2. Trend detector sees "trend" → feeds drift Kalman
//! 3. Drift skews quotes → we fill again in same direction
//!
//! The EchoEstimator tracks what fraction of observed trend changes correlate
//! with our own fills vs external/reference moves. High echo fraction → heavily
//! distrust trend observations (inflate R in Kalman).
//!
//! Cold start: echo_fraction=0.5 (uninformative prior), α=0.02 needs ~50 obs
//! to shift meaningfully. Until then, venue_base R multiplier (V1 behavior).

/// Tracks self-echo vs external signal contribution to trend changes.
#[derive(Debug, Clone)]
pub struct EchoEstimator {
    /// EMA of |trend_change| within dt of own fill
    fill_trend_ema: f64,
    /// EMA of |trend_change| within dt of reference/external move
    external_trend_ema: f64,
    /// EMA decay rate
    alpha: f64,
}

impl Default for EchoEstimator {
    fn default() -> Self {
        Self::new(0.02)
    }
}

impl EchoEstimator {
    /// Create with specified decay rate.
    /// α=0.02 → ~50 observations to shift from uninformative prior.
    pub fn new(alpha: f64) -> Self {
        Self {
            // Initialize both to equal small values → echo_fraction=0.5 (uninformative)
            fill_trend_ema: 1e-6,
            external_trend_ema: 1e-6,
            alpha: alpha.clamp(0.001, 0.5),
        }
    }

    /// Record trend change magnitude observed around our own fill.
    pub fn update_on_fill(&mut self, trend_change_abs: f64) {
        self.fill_trend_ema =
            self.alpha * trend_change_abs + (1.0 - self.alpha) * self.fill_trend_ema;
    }

    /// Record trend change magnitude observed around external/reference move.
    pub fn update_on_reference_move(&mut self, trend_change_abs: f64) {
        self.external_trend_ema =
            self.alpha * trend_change_abs + (1.0 - self.alpha) * self.external_trend_ema;
    }

    /// Fraction of trend changes attributable to self-echo [0, 1].
    /// 0.5 = uninformative (cold start), 1.0 = all echo, 0.0 = all external.
    pub fn echo_fraction(&self) -> f64 {
        let total = self.fill_trend_ema + self.external_trend_ema;
        if total < 1e-12 {
            0.5 // uninformative prior
        } else {
            self.fill_trend_ema / total
        }
    }

    /// R multiplier combining echo fraction with venue base.
    /// - echo=0.0 (all external) → 0.5× base (trust trend more)
    /// - echo=0.5 (equal/cold) → 1.0× base (neutral)
    /// - echo=1.0 (all self-echo) → 3.0× base (heavily distrust)
    pub fn r_multiplier(&self, venue_base: f64) -> f64 {
        venue_base * (0.5 + 2.5 * self.echo_fraction())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cold_start_uninformative() {
        let est = EchoEstimator::default();
        assert!(
            (est.echo_fraction() - 0.5).abs() < 0.01,
            "Cold start should be ~0.5, got {}",
            est.echo_fraction()
        );
    }

    #[test]
    fn test_pure_echo_high_fraction() {
        let mut est = EchoEstimator::new(0.1);
        // Feed only fill-correlated trend changes
        for _ in 0..100 {
            est.update_on_fill(1.0);
        }
        assert!(
            est.echo_fraction() > 0.95,
            "Pure fill echo should be near 1.0, got {}",
            est.echo_fraction()
        );
    }

    #[test]
    fn test_pure_external_low_fraction() {
        let mut est = EchoEstimator::new(0.1);
        // Feed only external trend changes
        for _ in 0..100 {
            est.update_on_reference_move(1.0);
        }
        assert!(
            est.echo_fraction() < 0.05,
            "Pure external should be near 0.0, got {}",
            est.echo_fraction()
        );
    }

    #[test]
    fn test_r_multiplier_scaling() {
        let est = EchoEstimator::default();
        let base = 5.0;
        let r = est.r_multiplier(base);
        // echo=0.5 → r = 5.0 * (0.5 + 2.5*0.5) = 5.0 * 1.75 = 8.75
        assert!(
            (r - base * 1.75).abs() < 0.1,
            "Neutral echo r_mult should be ~{}, got {r}",
            base * 1.75
        );

        // High echo
        let mut high_echo = EchoEstimator::new(0.5);
        high_echo.fill_trend_ema = 1.0;
        high_echo.external_trend_ema = 0.0;
        let r_high = high_echo.r_multiplier(base);
        // echo=1.0 → r = 5.0 * (0.5 + 2.5) = 5.0 * 3.0 = 15.0
        assert!(
            (r_high - base * 3.0).abs() < 0.1,
            "Full echo r_mult should be ~{}, got {r_high}",
            base * 3.0
        );

        // Low echo
        let mut low_echo = EchoEstimator::new(0.5);
        low_echo.fill_trend_ema = 0.0;
        low_echo.external_trend_ema = 1.0;
        let r_low = low_echo.r_multiplier(base);
        // echo=0.0 → r = 5.0 * (0.5 + 0) = 5.0 * 0.5 = 2.5
        assert!(
            (r_low - base * 0.5).abs() < 0.1,
            "No echo r_mult should be ~{}, got {r_low}",
            base * 0.5
        );
    }
}
