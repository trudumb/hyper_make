//! Bayesian signal fusion for directional drift estimation.
//!
//! Precision-weighted conjugate normal update fusing all directional signals
//! into a single posterior drift rate (μ). Each signal contributes a
//! `SignalObservation` with value and variance; the estimator produces a
//! posterior mean that enters the GLFT pricer via the μ·τ term.
//!
//! Architecture: every directional signal touches quotes through exactly
//! one channel — μ (drift). Variance modulation by regime happens at the
//! observation level, not inside the estimator.

/// A single directional signal observation.
#[derive(Debug, Clone, Copy)]
pub struct SignalObservation {
    /// Drift signal value in bps/sec.
    pub value_bps_per_sec: f64,
    /// Variance of this signal estimate. Lower variance = higher precision = more weight.
    /// CALIBRATION TARGET — bootstrap from paper trading MSE.
    pub variance: f64,
}

// Base variance constants for relative signal weighting.
// These determine how much each signal contributes to the posterior.
// CALIBRATION TARGET: replace with empirical MSE from paper trading.

/// Base variance for short-term momentum signal.
pub const BASE_MOMENTUM_VAR: f64 = 100.0; // CALIBRATION TARGET
/// Base variance for long-term trend signal.
pub const BASE_TREND_VAR: f64 = 200.0; // CALIBRATION TARGET
/// Base variance for lead-lag signal.
pub const BASE_LL_VAR: f64 = 50.0; // CALIBRATION TARGET
/// Base variance for flow imbalance signal.
pub const BASE_FLOW_VAR: f64 = 200.0; // CALIBRATION TARGET
/// Base variance for belief system drift signal.
pub const BASE_BELIEF_VAR: f64 = 150.0; // CALIBRATION TARGET

/// Bayesian drift estimator using precision-weighted conjugate normal updates.
///
/// Fuses multiple directional signals into a single posterior drift estimate.
/// Between updates, the posterior decays toward the prior (zero drift) to
/// prevent stale signals from persisting.
#[derive(Debug, Clone)]
pub struct DriftEstimator {
    /// Posterior mean drift (bps/sec).
    posterior_mean: f64,
    /// Posterior precision (1/variance).
    posterior_precision: f64,
    /// Prior precision — posterior decays toward this between updates.
    prior_precision: f64,
}

impl DriftEstimator {
    /// Create a new DriftEstimator with uninformative prior (zero drift).
    ///
    /// `prior_precision` controls how quickly the posterior reverts to zero
    /// when no signals are present. Higher = faster reversion.
    pub fn new(prior_precision: f64) -> Self {
        Self {
            posterior_mean: 0.0,
            posterior_precision: prior_precision,
            prior_precision,
        }
    }

    /// Bayesian conjugate normal update: fuse all signal observations.
    ///
    /// Posterior = Σ(precision_i × μ_i) / Σ(precision_i)
    /// where precision_i = 1/variance_i for each signal, plus the prior.
    ///
    /// When no signals are provided, posterior reverts to prior (zero drift).
    pub fn update(&mut self, signals: &[SignalObservation]) {
        if signals.is_empty() {
            // No signals: decay posterior toward prior
            // Blend: posterior = 0.9 × old_posterior + 0.1 × prior
            // This gives ~10 update half-life for stale signals
            self.posterior_mean *= 0.9;
            self.posterior_precision = self.prior_precision
                + 0.9 * (self.posterior_precision - self.prior_precision);
            return;
        }

        // Start with prior: zero mean, prior_precision
        let mut total_precision = self.prior_precision;
        let mut weighted_sum = 0.0; // prior_mean = 0.0, so prior contribution = 0

        for obs in signals {
            // Skip degenerate observations
            if obs.variance <= 0.0 || !obs.variance.is_finite() || !obs.value_bps_per_sec.is_finite() {
                continue;
            }
            let precision = 1.0 / obs.variance;
            total_precision += precision;
            weighted_sum += precision * obs.value_bps_per_sec;
        }

        if total_precision > 0.0 {
            self.posterior_mean = weighted_sum / total_precision;
            self.posterior_precision = total_precision;
        }
    }

    /// Posterior drift rate in fractional units per second (NOT bps).
    /// This is what enters the GLFT formula: r = S + μτ − γΣqτ
    pub fn drift_rate_per_sec(&self) -> f64 {
        self.posterior_mean / 10_000.0
    }

    /// Posterior drift in bps/sec (for logging).
    pub fn drift_bps_per_sec(&self) -> f64 {
        self.posterior_mean
    }

    /// Confidence in the drift estimate [0, 1].
    /// Higher when posterior precision >> prior precision (strong signal agreement).
    pub fn drift_confidence(&self) -> f64 {
        if self.posterior_precision <= self.prior_precision {
            0.0
        } else {
            // Confidence = 1 - prior_precision/posterior_precision
            // At prior only: 0.0. With strong signals: approaches 1.0.
            (1.0 - self.prior_precision / self.posterior_precision).clamp(0.0, 1.0)
        }
    }

    /// Posterior precision (for diagnostics).
    pub fn posterior_precision(&self) -> f64 {
        self.posterior_precision
    }
}

impl Default for DriftEstimator {
    fn default() -> Self {
        // Default prior precision: relatively uninformative
        // 1/100 = variance of 100 bps²/sec² → wide prior
        Self::new(0.01)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_signals_returns_zero_drift() {
        let est = DriftEstimator::default();
        assert_eq!(est.drift_rate_per_sec(), 0.0);
        assert_eq!(est.drift_confidence(), 0.0);
    }

    #[test]
    fn test_single_signal() {
        let mut est = DriftEstimator::default();
        est.update(&[SignalObservation {
            value_bps_per_sec: 10.0,
            variance: 100.0,
        }]);
        // With prior_precision=0.01 and signal precision=0.01,
        // posterior = (0.01*0 + 0.01*10) / 0.02 = 5.0 bps/sec
        let drift = est.drift_bps_per_sec();
        assert!(drift > 0.0, "Drift should be positive: {drift}");
        assert!(drift < 10.0, "Drift should be pulled toward zero by prior: {drift}");
    }

    #[test]
    fn test_multiple_signals_precision_weighted() {
        let mut est = DriftEstimator::new(0.001); // Very weak prior
        est.update(&[
            SignalObservation {
                value_bps_per_sec: 10.0,
                variance: 10.0, // High precision (0.1)
            },
            SignalObservation {
                value_bps_per_sec: -5.0,
                variance: 100.0, // Low precision (0.01)
            },
        ]);
        // High-precision signal (10.0) should dominate
        let drift = est.drift_bps_per_sec();
        assert!(drift > 0.0, "High-precision positive signal should dominate: {drift}");
    }

    #[test]
    fn test_decay_without_signals() {
        let mut est = DriftEstimator::default();
        est.update(&[SignalObservation {
            value_bps_per_sec: 10.0,
            variance: 1.0, // Very precise
        }]);
        let initial_drift = est.drift_bps_per_sec();
        assert!(initial_drift > 0.0);

        // Multiple updates with no signals should decay toward zero
        for _ in 0..50 {
            est.update(&[]);
        }
        let decayed_drift = est.drift_bps_per_sec();
        assert!(
            decayed_drift.abs() < initial_drift.abs() * 0.1,
            "Drift should decay toward zero: initial={initial_drift}, decayed={decayed_drift}"
        );
    }

    #[test]
    fn test_zero_drift_with_no_input_regression() {
        // μ=0 regression test: when DriftEstimator receives no signals,
        // drift_rate_per_sec must be exactly 0.0
        let est = DriftEstimator::default();
        assert_eq!(est.drift_rate_per_sec(), 0.0);
        assert_eq!(est.posterior_mean, 0.0);
    }

    #[test]
    fn test_degenerate_observations_ignored() {
        let mut est = DriftEstimator::default();
        est.update(&[
            SignalObservation {
                value_bps_per_sec: 10.0,
                variance: 0.0, // Degenerate: zero variance
            },
            SignalObservation {
                value_bps_per_sec: f64::NAN,
                variance: 100.0, // Degenerate: NaN value
            },
            SignalObservation {
                value_bps_per_sec: 5.0,
                variance: f64::INFINITY, // Degenerate: infinite variance
            },
        ]);
        // All observations are degenerate → should stay at prior (zero)
        assert_eq!(est.drift_bps_per_sec(), 0.0);
    }

    #[test]
    fn test_confidence_increases_with_signals() {
        let mut est = DriftEstimator::default();
        assert_eq!(est.drift_confidence(), 0.0);

        est.update(&[SignalObservation {
            value_bps_per_sec: 5.0,
            variance: 10.0,
        }]);
        let conf = est.drift_confidence();
        assert!(conf > 0.0, "Confidence should increase with signal: {conf}");
        assert!(conf < 1.0, "Confidence should not be 1.0 yet: {conf}");
    }

    #[test]
    fn test_opposing_signals_cancel() {
        let mut est = DriftEstimator::new(0.001); // Very weak prior
        est.update(&[
            SignalObservation {
                value_bps_per_sec: 10.0,
                variance: 50.0,
            },
            SignalObservation {
                value_bps_per_sec: -10.0,
                variance: 50.0,
            },
        ]);
        // Equal and opposite signals should roughly cancel
        assert!(
            est.drift_bps_per_sec().abs() < 1.0,
            "Opposing signals should cancel: {}",
            est.drift_bps_per_sec()
        );
    }
}
