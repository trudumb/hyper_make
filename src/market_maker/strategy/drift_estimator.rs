//! Kalman-filtered drift estimator with OU mean-reversion.
//!
//! Replaces the batch conjugate DriftEstimator with a Kalman filter using
//! Ornstein-Uhlenbeck dynamics. Drift decays toward zero between observations
//! via OU mean-reversion (rate θ), not ad-hoc 0.9× decay.
//!
//! Architecture: all directional signals enter as (z, R) observation pairs.
//! The filter produces a posterior μ̂ that enters the GLFT pricer via the μ·τ term.
//! No drift cap — posterior variance P naturally bounds the estimate.
//!
//! Feature tiers:
//! - Hot (every cycle): BIM, ΔBIM, BPG, sweep → composite observation (Phase 7)
//! - Warm (every 2-3 cycles): Hawkes, fill size, basis (Phase 8)
//! - Cold (every 5-10 cycles): cross-asset lead, funding, OI (Phase 6/8)

use std::collections::VecDeque;

/// Default autocorrelation prior during warmup (< 20 fills).
/// Moderate value slightly dampens fill impact until enough data to measure.
/// Gets washed out by data within a few minutes.
const AUTOCORRELATION_WARMUP_PRIOR: f64 = 0.3;

/// Number of fills needed for meaningful autocorrelation measurement.
const MIN_FILLS_FOR_AUTOCORRELATION: usize = 20;

/// A single directional signal observation (kept for backward compat with signal collection).
#[derive(Debug, Clone, Copy)]
pub struct SignalObservation {
    /// Drift signal value in bps/sec.
    pub value_bps_per_sec: f64,
    /// Variance of this signal estimate. Lower variance = higher precision = more weight.
    pub variance: f64,
}

/// Snapshot of a signal's contribution to the Kalman posterior.
#[derive(Debug, Clone)]
pub struct SignalContribution {
    /// Signal name identifier.
    pub name: String,
    /// Current signal value (bps/sec).
    pub value_bps_per_sec: f64,
    /// Variance used in the filter for this signal.
    pub variance: f64,
}

// Base variance constants for relative signal weighting.
// These determine how much each signal contributes to the posterior.
// CALIBRATION TARGET: replace with empirical MSE from paper trading.

/// Base variance for short-term momentum signal.
pub const BASE_MOMENTUM_VAR: f64 = 100.0;
/// Base variance for long-term trend signal.
pub const BASE_TREND_VAR: f64 = 200.0;
/// Base variance for lead-lag signal.
pub const BASE_LL_VAR: f64 = 50.0;
/// Base variance for flow imbalance signal.
pub const BASE_FLOW_VAR: f64 = 200.0;
/// Base variance for belief system drift signal.
pub const BASE_BELIEF_VAR: f64 = 150.0;

/// Minimum posterior variance floor ((bps/sec)²).
/// Prevents overconfidence when many features feed the filter.
/// At P_MIN=2.0, uncertainty floor is √2 ≈ 1.4 bps/sec — filter can still track
/// 5+ bps/sec shifts but retains enough uncertainty to reverse within a few cycles.
const P_MIN: f64 = 2.0;

/// Kalman-filtered drift estimator with OU mean-reversion dynamics.
///
/// State model: dμ = -θ·μ·dt + σ_μ·dW (OU process)
/// Observation: z_k = μ_k + ε_k, ε_k ~ N(0, R_k)
///
/// Predict (every cycle): μ̂⁻ = μ̂·exp(-θΔt), P⁻ = P·exp(-2θΔt) + Q(Δt)
/// Update (per observation): K = P⁻/(P⁻+R), μ̂ = μ̂⁻ + K(z-μ̂⁻), P = (1-K)P⁻
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KalmanDriftEstimator {
    /// Posterior mean drift (bps/sec).
    state_mean: f64,
    /// Posterior variance P ((bps/sec)²). Kalman state is drift in bps/sec.
    state_variance: f64,
    /// OU mean-reversion rate (1/sec). ~0.02 = 35s half-life.
    theta: f64,
    /// Process noise σ_μ² (bps²/sec³).
    process_noise: f64,
    /// Initial/prior variance P₀. Used for confidence calculation.
    prior_variance: f64,
    /// Last update timestamp (ms).
    last_update_ms: u64,

    // === Fill-Quote Autocorrelation (WS1: Observation-Quality-Aware R) ===
    /// Rolling window of (fill_direction, quote_skew_at_fill_time) pairs.
    /// fill_direction: +1.0 for buy fill, -1.0 for sell fill.
    /// quote_skew: sign(ask_depth - bid_depth) at fill time, range [-1, 1].
    /// High correlation = fills are echoes of our own quotes (less informative).
    #[serde(default)]
    fill_skew_history: VecDeque<(f64, f64)>,

    /// Last observed signal contributions (for calibration logging).
    /// Each entry is (name, value_bps_per_sec, variance).
    #[serde(default)]
    last_signals: Vec<(String, f64, f64)>,

    /// Last Kalman innovation (z - predicted) for diagnostics.
    #[serde(default)]
    last_innovation: f64,
}

impl KalmanDriftEstimator {
    /// Create a new Kalman drift estimator.
    ///
    /// # Parameters
    /// - `theta`: OU mean-reversion rate (1/sec). 0.02 = 35s half-life.
    /// - `process_noise`: σ_μ² (bps²/sec³). Higher = more responsive to signals.
    pub fn new(theta: f64, process_noise: f64) -> Self {
        // Stationary variance = σ_μ²/(2θ). Initialize at 2× for faster cold-start:
        // higher P means larger Kalman gain on first observations.
        let stationary_var = process_noise / (2.0 * theta.max(1e-6));
        let prior_variance = stationary_var * 2.0;
        Self {
            state_mean: 0.0,
            state_variance: prior_variance,
            theta,
            process_noise,
            prior_variance,
            last_update_ms: 0,
            fill_skew_history: VecDeque::with_capacity(MIN_FILLS_FOR_AUTOCORRELATION + 5),
            last_signals: Vec::new(),
            last_innovation: 0.0,
        }
    }

    /// OU propagation: drift decays toward zero, variance grows.
    ///
    /// μ̂⁻ = μ̂·exp(-θΔt)
    /// P⁻ = P·exp(-2θΔt) + σ_μ²/(2θ)·(1 - exp(-2θΔt))
    pub fn predict(&mut self, now_ms: u64) {
        if self.last_update_ms == 0 {
            self.last_update_ms = now_ms;
            return;
        }

        let dt_ms = now_ms.saturating_sub(self.last_update_ms);
        if dt_ms == 0 {
            return;
        }
        let dt = dt_ms as f64 / 1000.0;

        // OU mean-reversion: drift decays toward zero
        let decay = (-self.theta * dt).exp();
        self.state_mean *= decay;

        // Variance propagation: P decays + process noise accumulates
        let decay2 = (-2.0 * self.theta * dt).exp();
        let stationary_var = self.process_noise / (2.0 * self.theta.max(1e-6));
        self.state_variance = self.state_variance * decay2 + stationary_var * (1.0 - decay2);

        // P_min floor prevents overconfidence from many feature updates
        self.state_variance = self.state_variance.max(P_MIN);

        self.last_update_ms = now_ms;
    }

    /// Kalman update with a single (z, R) observation.
    ///
    /// K = P⁻/(P⁻+R)
    /// μ̂ = μ̂⁻ + K(z - μ̂⁻)
    /// P = (1-K)P⁻
    pub fn update_single_observation(&mut self, z: f64, r: f64) {
        if !z.is_finite() || !r.is_finite() || r <= 0.0 {
            return;
        }

        let innovation = z - self.state_mean;
        self.last_innovation = innovation;

        let k = self.state_variance / (self.state_variance + r);
        self.state_mean += k * innovation;
        self.state_variance *= 1.0 - k;

        // P_min floor
        self.state_variance = self.state_variance.max(P_MIN);
    }

    /// Batch update from legacy SignalObservation vec.
    ///
    /// Converts each signal to a (z, R) pair and runs sequential Kalman updates.
    /// For hot-tier features, prefer composite observation (see Phase 7) to
    /// prevent P collapse from too many updates per cycle.
    pub fn update(&mut self, signals: &[SignalObservation], now_ms: u64) {
        // Run predict step first
        self.predict(now_ms);

        if signals.is_empty() {
            return;
        }

        for obs in signals {
            if obs.variance <= 0.0
                || !obs.variance.is_finite()
                || !obs.value_bps_per_sec.is_finite()
            {
                continue;
            }
            self.update_single_observation(obs.value_bps_per_sec, obs.variance);
        }
    }

    /// Fill observation: bid fill → bearish signal, ask fill → bullish.
    ///
    /// Bid fill means someone sold to us → selling pressure → bearish.
    /// Ask fill means someone bought from us → buying pressure → bullish.
    /// Scale by fill-to-mid distance: fills far from mid are more informative.
    ///
    /// R is scaled by fill-quote autocorrelation: when fills are echoes of our
    /// own quote skew, R increases and the Kalman gain drops. This replaces
    /// the FillCascadeTracker's influence on drift with measured autocorrelation.
    pub fn update_fill(&mut self, is_buy: bool, fill_price: f64, mid: f64, sigma: f64) {
        if sigma < 1e-12 || mid < 1e-12 {
            return;
        }

        // Distance from mid in sigma units (how aggressively someone hit our quote)
        let distance_sigma = ((mid - fill_price).abs() / mid) / sigma.max(1e-10);
        let scale = distance_sigma.min(3.0) * 0.5; // cap at 3σ, scale by 0.5

        // Bid fill (we bought) → someone sold to us → bearish signal (negative z)
        // Ask fill (we sold) → someone bought from us → bullish signal (positive z)
        let z = if is_buy { -scale } else { scale };

        let sigma_fill = 3.0; // Fill signal is noisy
        let base_r = sigma_fill * sigma_fill;
        // WS1: Scale R by fill-quote autocorrelation
        let r = self.observation_variance_scaled(base_r);
        self.update_single_observation(z, r);
    }

    /// Trend observation from multi-timeframe trend detector.
    ///
    /// z = -magnitude × agreement × p_continuation
    /// R = σ_trend² / agreement² × r_multiplier
    ///
    /// `r_multiplier` inflates observation noise to attenuate echo signals on thin venues.
    /// - 1.0 = default (LiquidCex), 5.0+ = ThinDex (heavily distrust trend)
    /// - Combined from venue_base, lambda-adaptive, and echo estimator
    pub fn update_trend(
        &mut self,
        magnitude: f64,
        agreement: f64,
        p_continuation: f64,
        r_multiplier: f64,
    ) {
        if agreement < 0.1 || !magnitude.is_finite() {
            return;
        }

        let z = -magnitude * agreement * p_continuation;
        let sigma_trend = 2.0;
        let base_r = sigma_trend * sigma_trend / (agreement * agreement).max(0.01);
        let r = base_r * r_multiplier.max(0.1);
        self.update_single_observation(z, r);
    }

    /// Funding rate as drift observation.
    ///
    /// Extreme funding rates signal expected price pressure:
    /// High positive funding → longs capitulate → expect downward pressure.
    /// At low |funding_zscore|, R is large so Kalman gain is small (negligible update).
    pub fn update_funding(&mut self, funding_zscore: f64, scale_factor: f64) {
        if !funding_zscore.is_finite() || funding_zscore.abs() < 1e-6 {
            return;
        }

        let z = -funding_zscore.signum() * funding_zscore.abs() * scale_factor;
        let sigma_funding = 2.0;
        // R inversely proportional to |zscore|: extreme funding = higher confidence.
        let r = sigma_funding * sigma_funding / funding_zscore.abs().max(0.1);
        self.update_single_observation(z, r);
    }

    /// Posterior drift rate in fractional units per second (NOT bps).
    /// This is what enters the GLFT formula: r = S + μτ − γΣqτ
    pub fn drift_rate_per_sec(&self) -> f64 {
        self.state_mean / 10_000.0
    }

    /// James-Stein shrunken drift rate (fractional per second).
    /// drift_shrunk = drift × max(0, 1 - P/drift²)
    /// SNR < 1 → returns 0. SNR >> 1 → returns near-raw.
    /// Prevents noisy sub-threshold drift from creating phantom skew.
    pub fn shrunken_drift_rate_per_sec(&self) -> f64 {
        let drift_bps = self.state_mean;
        let drift_sq = drift_bps * drift_bps;
        if drift_sq < 1e-12 {
            return 0.0;
        }
        let shrinkage = (1.0 - self.state_variance / drift_sq).max(0.0);
        (drift_bps * shrinkage) / 10_000.0
    }

    /// Posterior drift in bps/sec (for logging).
    pub fn drift_bps_per_sec(&self) -> f64 {
        self.state_mean
    }

    /// Posterior uncertainty: √P (in bps).
    /// Returns √P in bps/sec (NOT bps, despite the name).
    /// The Kalman state tracks drift in bps/sec, so P is (bps/sec)².
    pub fn drift_uncertainty_bps(&self) -> f64 {
        self.state_variance.max(0.0).sqrt()
    }

    /// Confidence in the drift estimate [0, 1].
    /// 1 - P/P₀: at prior → 0.0, with strong signals → approaches 1.0.
    pub fn drift_confidence(&self) -> f64 {
        if self.prior_variance <= 0.0 {
            return 0.0;
        }
        (1.0 - self.state_variance / self.prior_variance).clamp(0.0, 1.0)
    }

    /// Posterior variance (for diagnostics).
    pub fn state_variance(&self) -> f64 {
        self.state_variance
    }

    /// Posterior mean (for diagnostics).
    pub fn state_mean(&self) -> f64 {
        self.state_mean
    }

    /// Online parameter adaptation from realized drift.
    ///
    /// After markout window completes, compare predicted drift with realized.
    /// Prediction errors > expected → increase process_noise (more responsive).
    /// Prediction errors < expected → decrease process_noise (more stable).
    /// Also adapts theta: overshoots → increase θ (faster mean-reversion),
    /// undershoots → decrease θ (slower decay).
    ///
    /// Bounded: θ ∈ [0.01, 1.0], process_noise ∈ [0.1, 10.0].
    pub fn update_parameters(&mut self, realized_drift_bps: f64) {
        if !realized_drift_bps.is_finite() {
            return;
        }

        let prediction_error = (realized_drift_bps - self.state_mean).abs();
        let expected_error = self.state_variance.max(0.01).sqrt();

        // Normalized surprise: how many σ off was the prediction?
        let surprise = prediction_error / expected_error.max(0.01);

        if surprise > 1.5 {
            // Under-confident: prediction errors larger than expected
            // → increase process noise (more responsive to signals)
            // → decrease θ (slower mean-reversion, trust drift longer)
            self.process_noise = (self.process_noise * 1.05).min(10.0);
            self.theta = (self.theta * 0.99).max(0.01);
        } else if surprise < 0.5 {
            // Over-confident: prediction errors smaller than expected
            // → decrease process noise (less jittery)
            // → increase θ (faster mean-reversion, less drift tracking)
            self.process_noise = (self.process_noise * 0.95).max(0.1);
            self.theta = (self.theta * 1.01).min(1.0);
        }
        // surprise ∈ [0.5, 1.5]: well-calibrated, no change
    }

    /// Current theta value (for diagnostics).
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Current process noise (for diagnostics).
    pub fn process_noise(&self) -> f64 {
        self.process_noise
    }

    /// Temporarily boost responsiveness after a cascade event.
    ///
    /// Multiplies process_noise by `factor` (bounded at 10.0) so the Kalman
    /// filter trusts new observations more heavily. `update_parameters()` will
    /// naturally decay it back when realized drift matches predictions.
    pub fn boost_responsiveness(&mut self, factor: f64) {
        self.process_noise = (self.process_noise * factor).min(10.0);
    }

    // === Fill-Quote Autocorrelation (WS1) ===

    /// Record a fill event for autocorrelation tracking.
    ///
    /// `fill_direction`: +1.0 for buy fill, -1.0 for sell fill.
    /// `quote_skew`: current quote asymmetry at fill time, range [-1, 1].
    ///   Computed as sign(ask_depth - bid_depth) or normalized inventory skew.
    pub fn record_fill_for_autocorrelation(&mut self, fill_direction: f64, quote_skew: f64) {
        if !fill_direction.is_finite() || !quote_skew.is_finite() {
            return;
        }
        self.fill_skew_history
            .push_back((fill_direction.signum(), quote_skew.clamp(-1.0, 1.0)));
        // Keep rolling window at MIN_FILLS_FOR_AUTOCORRELATION size
        while self.fill_skew_history.len() > MIN_FILLS_FOR_AUTOCORRELATION {
            self.fill_skew_history.pop_front();
        }
    }

    /// Measured fill-quote autocorrelation [0, 1].
    ///
    /// High correlation = fills are echoes of our own quote skew (less informative).
    /// Range [0, 1]: 0 = independent (fully informative), 1 = pure echo.
    ///
    /// During warmup (< MIN_FILLS_FOR_AUTOCORRELATION fills), returns
    /// AUTOCORRELATION_WARMUP_PRIOR (0.3) to slightly dampen fills
    /// until enough data to measure. Gets washed out by data quickly.
    pub fn fill_quote_autocorrelation(&self) -> f64 {
        if self.fill_skew_history.len() < MIN_FILLS_FOR_AUTOCORRELATION {
            return AUTOCORRELATION_WARMUP_PRIOR;
        }

        let n = self.fill_skew_history.len() as f64;
        let (sum_x, sum_y, sum_xy, sum_x2, sum_y2) = self.fill_skew_history.iter().fold(
            (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64),
            |(sx, sy, sxy, sx2, sy2), &(x, y)| {
                (sx + x, sy + y, sxy + x * y, sx2 + x * x, sy2 + y * y)
            },
        );

        let denom = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        if denom < 1e-10 {
            // All fills same direction → zero variance in fill_direction.
            // This IS the cascade/echo pattern: every fill confirms the same skew.
            // Return high autocorrelation to suppress Kalman gain (fills uninformative).
            if self.fill_skew_history.len() >= MIN_FILLS_FOR_AUTOCORRELATION {
                return 0.8;
            }
            return AUTOCORRELATION_WARMUP_PRIOR;
        }

        let r = (n * sum_xy - sum_x * sum_y) / denom;
        // Map correlation [-1, 1] to echo metric [0, 1]:
        // positive correlation = fills align with skew = echo
        // negative/zero correlation = independent
        r.clamp(0.0, 1.0)
    }

    /// Compute observation variance R scaled by fill-quote autocorrelation.
    ///
    /// When fills are echoes of our own quotes (high autocorrelation),
    /// R increases → Kalman gain drops → drift updates slower.
    /// At 80% echo: R is 5× base → Kalman gain drops to ~20%.
    /// At 0% echo: R is 1× base → full Kalman gain.
    fn observation_variance_scaled(&self, base_r: f64) -> f64 {
        let echo = self.fill_quote_autocorrelation();
        let informativeness = (1.0 - echo).max(0.05); // floor at 5%
        base_r / informativeness
    }

    /// Current fill-skew history length (for diagnostics).
    pub fn fill_skew_history_len(&self) -> usize {
        self.fill_skew_history.len()
    }

    // === Signal Contribution Tracking ===

    /// Record a signal contribution for calibration logging.
    ///
    /// Tracks the last 10 signals per cycle. When the buffer is full it
    /// clears to make room for the next cycle's signals, preventing
    /// unbounded growth.
    pub fn record_signal(&mut self, name: &str, value_bps_per_sec: f64, variance: f64) {
        if self.last_signals.len() >= 10 {
            self.last_signals.clear();
        }
        self.last_signals
            .push((name.to_string(), value_bps_per_sec, variance));
    }

    /// Returns the last observed signal contributions for calibration logging.
    pub fn signal_contributions(&self) -> Vec<SignalContribution> {
        self.last_signals
            .iter()
            .map(|(name, value, var)| SignalContribution {
                name: name.clone(),
                value_bps_per_sec: *value,
                variance: *var,
            })
            .collect()
    }

    /// Returns the last Kalman innovation (observation - predicted).
    ///
    /// Autocorrelation in the innovation sequence indicates model
    /// miscalibration: a well-specified filter produces white-noise
    /// innovations.
    pub fn last_innovation(&self) -> f64 {
        self.last_innovation
    }
}

impl Default for KalmanDriftEstimator {
    fn default() -> Self {
        // theta=0.02 (35s half-life), process_noise=1.0
        // Stationary variance = 1.0/(2×0.02) = 25.0
        // Prior variance = 50.0 (2× stationary for cold-start)
        Self::new(0.02, 1.0)
    }
}

// === Legacy estimator kept for feature-gate rollback ===

/// Legacy batch conjugate normal drift estimator (pre-Kalman).
/// Kept for feature-gate rollback. Use `KalmanDriftEstimator` instead.
#[derive(Debug, Clone)]
pub struct LegacyDriftEstimator {
    posterior_mean: f64,
    posterior_precision: f64,
    prior_precision: f64,
}

impl LegacyDriftEstimator {
    pub fn new(prior_precision: f64) -> Self {
        Self {
            posterior_mean: 0.0,
            posterior_precision: prior_precision,
            prior_precision,
        }
    }

    pub fn update(&mut self, signals: &[SignalObservation]) {
        if signals.is_empty() {
            self.posterior_mean *= 0.9;
            self.posterior_precision =
                self.prior_precision + 0.9 * (self.posterior_precision - self.prior_precision);
            return;
        }

        let mut total_precision = self.prior_precision;
        let mut weighted_sum = 0.0;

        for obs in signals {
            if obs.variance <= 0.0
                || !obs.variance.is_finite()
                || !obs.value_bps_per_sec.is_finite()
            {
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

    pub fn drift_rate_per_sec(&self) -> f64 {
        self.posterior_mean / 10_000.0
    }

    pub fn drift_bps_per_sec(&self) -> f64 {
        self.posterior_mean
    }

    pub fn drift_confidence(&self) -> f64 {
        if self.posterior_precision <= self.prior_precision {
            0.0
        } else {
            (1.0 - self.prior_precision / self.posterior_precision).clamp(0.0, 1.0)
        }
    }
}

impl Default for LegacyDriftEstimator {
    fn default() -> Self {
        Self::new(0.01)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now_ms() -> u64 {
        1_000_000 // Arbitrary starting time
    }

    #[test]
    fn test_kalman_predict_decays_toward_zero() {
        let mut est = KalmanDriftEstimator {
            state_mean: 10.0,
            last_update_ms: now_ms(),
            ..Default::default()
        };

        // Predict 35 seconds later (one half-life at θ=0.02)
        est.predict(now_ms() + 35_000);

        // Should decay to ~half (exp(-0.02*35) ≈ 0.497)
        assert!(
            est.state_mean < 6.0 && est.state_mean > 4.0,
            "Expected ~5.0 after one half-life, got {}",
            est.state_mean
        );
    }

    #[test]
    fn test_kalman_update_moves_toward_observation() {
        let mut est = KalmanDriftEstimator {
            last_update_ms: now_ms(),
            ..Default::default()
        };

        // Single precise observation
        est.update_single_observation(10.0, 5.0);

        assert!(
            est.state_mean > 0.0,
            "Should shift toward positive observation: {}",
            est.state_mean
        );
        assert!(
            est.state_mean < 10.0,
            "Should be pulled toward zero by prior: {}",
            est.state_mean
        );
    }

    #[test]
    fn test_kalman_fill_observation_bearish() {
        let mut est = KalmanDriftEstimator {
            last_update_ms: now_ms(),
            ..Default::default()
        };

        // Bid fill at 99 when mid is 100 → someone sold to us → bearish
        est.update_fill(true, 99.0, 100.0, 0.001);

        assert!(
            est.state_mean < 0.0,
            "Bid fill should produce bearish (negative) drift: {}",
            est.state_mean
        );
    }

    #[test]
    fn test_kalman_trend_observation_high_agreement() {
        let mut est = KalmanDriftEstimator {
            last_update_ms: now_ms(),
            ..Default::default()
        };

        // Strong bearish trend with high agreement (r_multiplier=1.0 for LiquidCex)
        est.update_trend(5.0, 0.9, 0.8, 1.0);

        // z = -5.0 * 0.9 * 0.8 = -3.6, R = 4.0 / 0.81 ≈ 4.94
        assert!(
            est.state_mean < 0.0,
            "Bearish trend should produce negative drift: {}",
            est.state_mean
        );
    }

    #[test]
    fn test_drift_unclamped_exceeds_3bps() {
        let mut est = KalmanDriftEstimator {
            last_update_ms: now_ms(),
            ..Default::default()
        };

        // Multiple strong bearish signals
        for _ in 0..5 {
            est.update_single_observation(-10.0, 5.0); // Strong, precise bearish
        }

        let drift_bps = est.drift_bps_per_sec().abs();
        assert!(
            drift_bps > 3.0,
            "Strong drift should exceed old ±3 cap: {} bps",
            drift_bps
        );
    }

    #[test]
    fn test_no_signals_decays_via_ou() {
        let mut est = KalmanDriftEstimator {
            state_mean: 10.0,
            last_update_ms: now_ms(),
            ..Default::default()
        };

        // 100 seconds of decay (no signals)
        est.predict(now_ms() + 100_000);

        // exp(-0.02 * 100) = exp(-2) ≈ 0.135
        assert!(
            est.state_mean < 2.0,
            "Should decay significantly via OU: {}",
            est.state_mean
        );
        assert!(
            est.state_mean > 0.0,
            "Should still be positive (not zero): {}",
            est.state_mean
        );
    }

    #[test]
    fn test_p_min_floor_prevents_overconfidence() {
        let mut est = KalmanDriftEstimator {
            last_update_ms: now_ms(),
            ..Default::default()
        };

        // Many precise observations — P should not drop below P_MIN
        for _ in 0..100 {
            est.update_single_observation(5.0, 1.0);
        }

        assert!(
            est.state_variance >= P_MIN,
            "Variance should not drop below P_MIN: {}",
            est.state_variance
        );
    }

    #[test]
    fn test_confidence_increases_with_signals() {
        let mut est = KalmanDriftEstimator::default();
        let initial_conf = est.drift_confidence();
        assert!(
            initial_conf < 0.01,
            "Initial confidence should be ~0: {initial_conf}"
        );

        est.last_update_ms = now_ms();
        est.update_single_observation(5.0, 10.0);

        let conf = est.drift_confidence();
        assert!(
            conf > initial_conf,
            "Confidence should increase: {} > {}",
            conf,
            initial_conf
        );
    }

    #[test]
    fn test_opposing_signals_cancel() {
        let mut est = KalmanDriftEstimator {
            last_update_ms: now_ms(),
            ..Default::default()
        };

        est.update_single_observation(10.0, 5.0);
        est.update_single_observation(-10.0, 5.0);

        assert!(
            est.state_mean.abs() < 3.0,
            "Opposing signals should roughly cancel: {}",
            est.state_mean
        );
    }

    #[test]
    fn test_degenerate_observations_ignored() {
        let mut est = KalmanDriftEstimator {
            last_update_ms: now_ms(),
            ..Default::default()
        };
        let initial = est.state_mean;

        est.update_single_observation(f64::NAN, 5.0);
        est.update_single_observation(5.0, 0.0);
        est.update_single_observation(5.0, f64::INFINITY);
        est.update_single_observation(5.0, -1.0);

        assert_eq!(
            est.state_mean, initial,
            "Degenerate observations should be ignored"
        );
    }

    #[test]
    fn test_update_with_legacy_signals() {
        let mut est = KalmanDriftEstimator::default();

        est.update(
            &[SignalObservation {
                value_bps_per_sec: 10.0,
                variance: 100.0,
            }],
            now_ms(),
        );

        assert!(est.state_mean > 0.0, "Legacy signal update should work");
    }

    #[test]
    fn test_funding_observation_extreme_bearish() {
        let mut est = KalmanDriftEstimator {
            last_update_ms: now_ms(),
            ..Default::default()
        };

        // funding_zscore = +3.0 → longs paying heavily → expect downward pressure
        est.update_funding(3.0, 0.5);

        assert!(
            est.state_mean < 0.0,
            "High positive funding should produce bearish drift: {}",
            est.state_mean
        );
    }

    #[test]
    fn test_default_starts_at_zero() {
        let est = KalmanDriftEstimator::default();
        assert_eq!(est.drift_rate_per_sec(), 0.0);
        assert_eq!(est.drift_bps_per_sec(), 0.0);
    }

    // === Phase 5: Online Parameter Adaptation Tests ===

    #[test]
    fn test_theta_adapts_on_undershoot() {
        let mut est = KalmanDriftEstimator::default();
        let initial_theta = est.theta();
        let initial_noise = est.process_noise();

        // Large prediction error → surprise > 1.5 → increase noise, decrease theta
        est.update_parameters(100.0);

        assert!(
            est.process_noise() > initial_noise,
            "Large error should increase process_noise"
        );
        assert!(
            est.theta() < initial_theta,
            "Large error should decrease theta"
        );
    }

    #[test]
    fn test_theta_adapts_on_overshoot() {
        // Push state_mean close to 10 with high confidence
        let mut est = KalmanDriftEstimator {
            state_mean: 10.0,
            state_variance: P_MIN, // minimum variance = high confidence
            ..Default::default()
        };

        let initial_noise = est.process_noise();

        // Tiny prediction error → surprise < 0.5 → decrease noise, increase theta
        est.update_parameters(10.1);

        assert!(
            est.process_noise() < initial_noise,
            "Small error should decrease process_noise: {} vs {}",
            est.process_noise(),
            initial_noise
        );
    }

    #[test]
    fn test_parameters_bounded() {
        let mut est = KalmanDriftEstimator::default();

        // Hammer with large errors to drive noise up
        for _ in 0..1000 {
            est.update_parameters(1000.0);
        }
        assert!(est.process_noise() <= 10.0, "process_noise bounded at 10.0");
        assert!(est.theta() >= 0.01, "theta bounded at 0.01");

        // Hammer with tiny errors to drive noise down
        est.state_mean = 0.0;
        est.state_variance = P_MIN;
        for _ in 0..1000 {
            est.update_parameters(0.0);
        }
        assert!(est.process_noise() >= 0.1, "process_noise bounded at 0.1");
        assert!(est.theta() <= 1.0, "theta bounded at 1.0");
    }

    // === WS1: Fill-Quote Autocorrelation Tests ===

    #[test]
    fn test_autocorrelation_zero_when_independent() {
        let mut est = KalmanDriftEstimator {
            last_update_ms: now_ms(),
            ..Default::default()
        };

        // Alternating: buy fills with positive skew, sell fills with negative skew
        // This is anti-correlated (fills oppose skew), so autocorrelation should be low
        for i in 0..20 {
            let dir = if i % 2 == 0 { 1.0 } else { -1.0 };
            let skew = if i % 2 == 0 { -0.5 } else { 0.5 }; // opposite of fill dir
            est.record_fill_for_autocorrelation(dir, skew);
        }

        let ac = est.fill_quote_autocorrelation();
        assert!(
            ac < 0.2,
            "Anti-correlated fills should have near-zero autocorrelation: {}",
            ac
        );
    }

    #[test]
    fn test_autocorrelation_high_when_echo() {
        let mut est = KalmanDriftEstimator {
            last_update_ms: now_ms(),
            ..Default::default()
        };

        // Fills align with skew: buy fills when skew is positive,
        // sell fills when skew is negative. This is echo behavior.
        // Need variance in both x and y for Pearson correlation to be defined.
        for _ in 0..10 {
            est.record_fill_for_autocorrelation(1.0, 0.8); // buy fill, positive skew
            est.record_fill_for_autocorrelation(-1.0, -0.8); // sell fill, negative skew
        }

        let ac = est.fill_quote_autocorrelation();
        assert!(
            ac > 0.7,
            "Echo fills should have high autocorrelation: {}",
            ac
        );
    }

    #[test]
    fn test_autocorrelation_r_scaling() {
        let mut est = KalmanDriftEstimator {
            last_update_ms: now_ms(),
            ..Default::default()
        };

        // No fills yet → warmup prior (0.3)
        assert!(
            (est.fill_quote_autocorrelation() - AUTOCORRELATION_WARMUP_PRIOR).abs() < 0.01,
            "Should return warmup prior with no fills"
        );

        // Base R = 9.0 (sigma_fill=3.0)
        let base_r = 9.0;

        // With warmup prior (0.3): informativeness = 0.7, scaled_r = 9/0.7 ≈ 12.9
        let scaled_r = est.observation_variance_scaled(base_r);
        assert!(
            scaled_r > base_r,
            "Warmup prior should slightly inflate R: {} > {}",
            scaled_r,
            base_r
        );

        // Fill with high echo (fills align with skew direction — echo behavior)
        // Need variance in both x and y for Pearson r to be defined
        for _ in 0..10 {
            est.record_fill_for_autocorrelation(1.0, 0.9); // buy fill, positive skew
            est.record_fill_for_autocorrelation(-1.0, -0.9); // sell fill, negative skew
        }

        let high_echo_r = est.observation_variance_scaled(base_r);
        assert!(
            high_echo_r > 3.0 * base_r,
            "High echo should inflate R by 3x+: {} vs base {}",
            high_echo_r,
            base_r
        );
    }

    #[test]
    fn test_autocorrelation_mixed_fills_near_zero() {
        let mut est = KalmanDriftEstimator {
            last_update_ms: now_ms(),
            ..Default::default()
        };

        // Random-ish pattern: fills and skew uncorrelated
        let fills = [
            (1.0, 0.3),
            (-1.0, 0.5),
            (1.0, -0.2),
            (-1.0, -0.8),
            (1.0, 0.1),
            (-1.0, 0.4),
            (1.0, -0.6),
            (-1.0, 0.2),
            (1.0, 0.7),
            (-1.0, -0.3),
            (1.0, -0.1),
            (-1.0, 0.6),
            (1.0, 0.2),
            (-1.0, -0.5),
            (1.0, -0.4),
            (-1.0, 0.1),
            (1.0, 0.5),
            (-1.0, -0.7),
            (1.0, -0.3),
            (-1.0, 0.8),
        ];
        for (dir, skew) in &fills {
            est.record_fill_for_autocorrelation(*dir, *skew);
        }

        let ac = est.fill_quote_autocorrelation();
        assert!(
            ac < 0.4,
            "Mixed fills should have low autocorrelation: {}",
            ac
        );
    }

    #[test]
    fn test_drift_recovery_after_cascade_echo() {
        let mut est = KalmanDriftEstimator {
            last_update_ms: now_ms(),
            ..Default::default()
        };

        // Simulate cascade: 20 same-direction fills with matching skew (echo)
        for _ in 0..20 {
            est.record_fill_for_autocorrelation(-1.0, -0.7); // sell fills with sell skew
        }

        // Push drift negative with fill observations
        for _ in 0..5 {
            est.update_fill(true, 99.0, 100.0, 0.001); // buy fills = bearish
        }
        let cascade_drift = est.drift_bps_per_sec();

        // Now let OU decay for 35s (one half-life)
        est.predict(now_ms() + 35_000);
        let recovered_drift = est.drift_bps_per_sec();

        assert!(
            recovered_drift.abs() < cascade_drift.abs(),
            "Drift should recover after cascade: cascade={:.2}, recovered={:.2}",
            cascade_drift,
            recovered_drift
        );
    }

    #[test]
    fn test_shrunken_drift_zeros_low_snr() {
        // SNR < 1: drift=0.78, P=2.0 → drift²=0.608, P/drift²=3.29 → shrinkage=0
        let est = KalmanDriftEstimator {
            state_mean: 0.78,
            state_variance: 2.0,
            ..Default::default()
        };
        assert_eq!(
            est.shrunken_drift_rate_per_sec(),
            0.0,
            "Sub-threshold drift (SNR<1) should be shrunk to zero"
        );
    }

    #[test]
    fn test_shrunken_drift_partial_at_moderate_snr() {
        // SNR = 2: drift=2.0, P=2.0 → drift²=4.0, shrinkage=0.5 → effective=1.0 bps
        let est = KalmanDriftEstimator {
            state_mean: 2.0,
            state_variance: 2.0,
            ..Default::default()
        };
        let shrunk = est.shrunken_drift_rate_per_sec();
        let expected_bps = 1.0; // 2.0 * 0.5
        let expected_frac = expected_bps / 10_000.0;
        assert!(
            (shrunk - expected_frac).abs() < 1e-10,
            "Moderate SNR should give partial shrinkage: got {shrunk}, expected {expected_frac}"
        );
    }

    #[test]
    fn test_shrunken_drift_preserves_strong_signal() {
        let est = KalmanDriftEstimator {
            state_mean: 5.0,
            state_variance: 2.0,
            ..Default::default()
        };
        // SNR = 12.5: drift=5.0, P=2.0 → drift²=25.0, shrinkage=0.92
        let shrunk = est.shrunken_drift_rate_per_sec();
        let raw = est.drift_rate_per_sec();
        let retention = shrunk / raw;
        assert!(
            retention > 0.9,
            "Strong signal should be mostly preserved: retention={retention:.3}"
        );
    }

    #[test]
    fn test_shrunken_drift_zero_drift_returns_zero() {
        let est = KalmanDriftEstimator {
            state_mean: 0.0,
            state_variance: 2.0,
            ..Default::default()
        };
        assert_eq!(est.shrunken_drift_rate_per_sec(), 0.0);
    }

    #[test]
    fn test_same_direction_fills_high_autocorrelation() {
        let mut est = KalmanDriftEstimator {
            last_update_ms: now_ms(),
            ..Default::default()
        };

        // 20 same-direction buy fills (zero variance in fill direction)
        for _ in 0..20 {
            est.record_fill_for_autocorrelation(1.0, 0.5);
        }

        let ac = est.fill_quote_autocorrelation();
        assert!(
            ac >= 0.7,
            "All same-direction fills should return high autocorrelation (cascade pattern): {}",
            ac
        );
    }

    #[test]
    fn test_same_direction_fills_below_threshold_returns_warmup() {
        let mut est = KalmanDriftEstimator {
            last_update_ms: now_ms(),
            ..Default::default()
        };

        // Only 10 same-direction fills (below MIN_FILLS_FOR_AUTOCORRELATION=20)
        for _ in 0..10 {
            est.record_fill_for_autocorrelation(1.0, 0.5);
        }

        let ac = est.fill_quote_autocorrelation();
        assert!(
            (ac - AUTOCORRELATION_WARMUP_PRIOR).abs() < 0.01,
            "Below threshold should return warmup prior: {}",
            ac
        );
    }

    #[test]
    fn test_mixed_fills_proper_pearson() {
        let mut est = KalmanDriftEstimator {
            last_update_ms: now_ms(),
            ..Default::default()
        };

        // Alternating fills with matching skew = high echo
        for _ in 0..10 {
            est.record_fill_for_autocorrelation(1.0, 0.8);
            est.record_fill_for_autocorrelation(-1.0, -0.8);
        }

        let ac = est.fill_quote_autocorrelation();
        assert!(
            ac > 0.6,
            "Mixed fills with matching skew should compute proper Pearson r: {}",
            ac
        );
    }
}
