//! Volatility estimation components.
//!
//! - SingleScaleBipower: EWMA-based realized/bipower variation
//! - MultiScaleBipowerEstimator: Multi-timescale volatility (fast/medium/slow)
//! - VolatilityRegime: 4-state regime classification
//! - VolatilityRegimeTracker: Regime transitions with hysteresis
//! - StochasticVolParams: Heston-style stochastic volatility parameters

use std::collections::VecDeque;
use tracing::debug;

use super::volume::VolumeBucket;
use super::EstimatorConfig;

// ============================================================================
// O(1) Incremental Statistics for Stochastic Volatility Calibration
// ============================================================================

/// O(1) incremental statistics tracker using Welford's algorithm.
///
/// Eliminates the O(n) Vec allocations and iterations in calibration by
/// maintaining running statistics that update in O(1) per observation.
///
/// Tracks:
/// - Mean (E[X])
/// - Variance (Var[X])
/// - Covariance (Cov[X, Y])
/// - Autocorrelation (Corr[X_t, X_{t-1}])
///
/// Key insight: Welford's online algorithm computes variance in a single pass
/// without storing all values:
///   M_n = M_{n-1} + (x_n - M_{n-1})/n
///   S_n = S_{n-1} + (x_n - M_{n-1})(x_n - M_n)
///   Var = S_n / (n-1)
#[derive(Debug, Clone)]
pub(crate) struct IncrementalVolStats {
    // === Mean and Variance (Welford's algorithm) ===
    /// Number of observations
    n: usize,
    /// Running mean of variance values
    mean_var: f64,
    /// Running M2 (sum of squared deviations) for variance
    m2_var: f64,

    // === Variance of variance changes (for ξ estimation) ===
    /// Running mean of variance changes (dV = V_t - V_{t-1})
    mean_dv: f64,
    /// Running M2 for variance changes
    m2_dv: f64,
    /// Previous variance (for computing dV)
    prev_variance: Option<f64>,
    /// Count of variance change observations
    n_dv: usize,

    // === Autocorrelation (for κ estimation) ===
    /// Running sum of (V_t - mean)(V_{t-1} - mean) for autocovariance
    autocov_sum: f64,
    /// Previous deviation from mean (for autocov)
    prev_deviation: Option<f64>,

    // === Return-variance correlation (for ρ estimation) ===
    /// Running mean of returns
    mean_ret: f64,
    /// Running M2 for returns
    m2_ret: f64,
    /// Running covariance accumulator: sum of (r - mean_r)(dV - mean_dV)
    cov_ret_dv: f64,
    /// Previous return (for correlation with variance change)
    prev_return: Option<f64>,
    /// Count of return observations
    n_ret: usize,
}

impl IncrementalVolStats {
    /// Create a new incremental stats tracker.
    pub(crate) fn new() -> Self {
        Self {
            n: 0,
            mean_var: 0.0,
            m2_var: 0.0,
            mean_dv: 0.0,
            m2_dv: 0.0,
            prev_variance: None,
            n_dv: 0,
            autocov_sum: 0.0,
            prev_deviation: None,
            mean_ret: 0.0,
            m2_ret: 0.0,
            cov_ret_dv: 0.0,
            prev_return: None,
            n_ret: 0,
        }
    }

    /// Update with a new variance observation.
    ///
    /// This is O(1) - no allocations, just arithmetic.
    fn update_variance(&mut self, variance: f64) {
        self.n += 1;

        // Welford's online mean and variance
        let delta = variance - self.mean_var;
        self.mean_var += delta / self.n as f64;
        let delta2 = variance - self.mean_var;
        self.m2_var += delta * delta2;

        // Update autocorrelation (deviation from current mean estimate)
        let deviation = variance - self.mean_var;
        if let Some(prev_dev) = self.prev_deviation {
            self.autocov_sum += prev_dev * deviation;
        }
        self.prev_deviation = Some(deviation);

        // Update variance change statistics
        if let Some(prev_var) = self.prev_variance {
            let dv = variance - prev_var;
            self.n_dv += 1;

            let delta_dv = dv - self.mean_dv;
            self.mean_dv += delta_dv / self.n_dv as f64;
            let delta2_dv = dv - self.mean_dv;
            self.m2_dv += delta_dv * delta2_dv;
        }
        self.prev_variance = Some(variance);
    }

    /// Update with a return observation (for ρ estimation).
    ///
    /// Should be called in sync with update_variance.
    fn update_return(&mut self, log_return: f64) {
        self.n_ret += 1;

        // Welford for returns
        let delta_r = log_return - self.mean_ret;
        self.mean_ret += delta_r / self.n_ret as f64;
        let delta2_r = log_return - self.mean_ret;
        self.m2_ret += delta_r * delta2_r;

        // Covariance between return and variance change
        // Using online covariance: Cov(X,Y) ≈ E[XY] - E[X]E[Y]
        // Simplified: accumulate (r - mean_r)(dV - mean_dV) incrementally
        if let (Some(prev_ret), Some(prev_var)) = (self.prev_return, self.prev_variance) {
            if let Some(current_var) = self.prev_variance {
                // We have a valid variance change
                let dv = current_var - prev_var;
                // Online covariance update
                self.cov_ret_dv += (prev_ret - self.mean_ret) * (dv - self.mean_dv);
            }
        }
        self.prev_return = Some(log_return);
    }

    /// Get estimated theta (long-run mean variance).
    ///
    /// θ = E[V]
    #[inline]
    fn theta(&self) -> Option<f64> {
        if self.n > 0 {
            Some(self.mean_var)
        } else {
            None
        }
    }

    /// Get estimated xi (vol-of-vol).
    ///
    /// ξ = std(dV) = √Var(V_t - V_{t-1})
    #[inline]
    fn xi(&self) -> Option<f64> {
        if self.n_dv > 1 {
            let var_dv = self.m2_dv / (self.n_dv - 1) as f64;
            Some(var_dv.sqrt().max(1e-9))
        } else {
            None
        }
    }

    /// Get estimated kappa (mean-reversion speed).
    ///
    /// For OU process: autocorr ≈ exp(-κ × Δt)
    /// κ ≈ -ln(autocorr) / Δt
    ///
    /// Assumes Δt ≈ 1 second between observations.
    #[inline]
    fn kappa(&self) -> Option<f64> {
        if self.n < 3 {
            return None;
        }

        let variance = self.m2_var / (self.n - 1) as f64;
        if variance < 1e-12 {
            return None;
        }

        let autocov = self.autocov_sum / (self.n - 1) as f64;
        let autocorr = autocov / variance;

        if autocorr > 0.0 && autocorr < 1.0 {
            Some((-autocorr.ln()).clamp(0.01, 5.0))
        } else {
            None
        }
    }

    /// Get estimated rho (price-vol correlation).
    ///
    /// ρ = Corr(r, dV) = Cov(r, dV) / (σ_r × σ_dV)
    #[inline]
    fn rho(&self) -> Option<f64> {
        if self.n_ret < 3 || self.n_dv < 3 {
            return None;
        }

        let var_ret = self.m2_ret / (self.n_ret - 1) as f64;
        let var_dv = self.m2_dv / (self.n_dv - 1) as f64;

        if var_ret < 1e-12 || var_dv < 1e-12 {
            return None;
        }

        let std_ret = var_ret.sqrt();
        let std_dv = var_dv.sqrt();

        let n_pairs = self.n_ret.min(self.n_dv) - 1;
        if n_pairs == 0 {
            return None;
        }

        let cov = self.cov_ret_dv / n_pairs as f64;
        Some((cov / (std_ret * std_dv)).clamp(-0.95, 0.95))
    }

    /// Get number of variance observations.
    #[inline]
    #[allow(dead_code)]
    fn count(&self) -> usize {
        self.n
    }

    /// Check if enough observations for calibration.
    #[inline]
    fn is_ready(&self, min_obs: usize) -> bool {
        self.n >= min_obs
    }

    /// Reset all statistics.
    #[allow(dead_code)]
    fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for IncrementalVolStats {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Single-Scale Bipower Variation (Building Block)
// ============================================================================

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

// ============================================================================
// Multi-Timescale Bipower Estimator
// ============================================================================

/// Multi-timescale volatility with fast/medium/slow components.
///
/// Fast (~2s): Reacts quickly to crashes, used for early warning
/// Medium (~10s): Balanced responsiveness
/// Slow (~60s): Stable baseline for pricing
#[derive(Debug)]
pub(crate) struct MultiScaleBipowerEstimator {
    fast: SingleScaleBipower,   // ~5 ticks / ~2 seconds
    medium: SingleScaleBipower, // ~20 ticks / ~10 seconds
    slow: SingleScaleBipower,   // ~100 ticks / ~60 seconds
    last_vwap: Option<f64>,
    tick_count: usize,
}

impl MultiScaleBipowerEstimator {
    pub(crate) fn new(config: &EstimatorConfig) -> Self {
        let default_var = config.default_sigma.powi(2);
        Self {
            fast: SingleScaleBipower::new(config.fast_half_life_ticks, default_var),
            medium: SingleScaleBipower::new(config.medium_half_life_ticks, default_var),
            slow: SingleScaleBipower::new(config.slow_half_life_ticks, default_var),
            last_vwap: None,
            tick_count: 0,
        }
    }

    /// Process a completed volume bucket.
    pub(crate) fn on_bucket(&mut self, bucket: &VolumeBucket) {
        if let Some(prev_vwap) = self.last_vwap {
            if bucket.vwap > 0.0 && prev_vwap > 0.0 {
                let log_return = (bucket.vwap / prev_vwap).ln();
                self.fast.update(log_return);
                self.medium.update(log_return);
                self.slow.update(log_return);
                self.tick_count += 1;
            }
        }
        self.last_vwap = Some(bucket.vwap);
    }

    /// Get the log return for the most recent bucket (for momentum tracking)
    pub(crate) fn last_log_return(&self, bucket: &VolumeBucket) -> Option<f64> {
        self.last_vwap.and_then(|prev| {
            if bucket.vwap > 0.0 && prev > 0.0 {
                Some((bucket.vwap / prev).ln())
            } else {
                None
            }
        })
    }

    /// Clean sigma (BV-based) for spread pricing.
    /// Uses slow timescale for stability.
    pub(crate) fn sigma_clean(&self) -> f64 {
        self.slow.sigma_clean()
    }

    /// Total sigma (RV-based) for risk assessment.
    /// Blends fast + slow: uses fast when market is accelerating.
    pub(crate) fn sigma_total(&self) -> f64 {
        let fast = self.fast.sigma_total();
        let slow = self.slow.sigma_total();

        // If fast >> slow, market is accelerating - trust fast more
        let ratio = fast / slow.max(1e-9);

        if ratio > 1.5 {
            // Acceleration: blend toward fast
            let weight = ((ratio - 1.0) / 3.0).clamp(0.0, 0.7);
            weight * fast + (1.0 - weight) * slow
        } else {
            // Stable: prefer slow for less noise
            0.2 * fast + 0.8 * slow
        }
    }

    /// Effective sigma for inventory skew.
    /// Blends clean and total based on jump regime.
    pub(crate) fn sigma_effective(&self) -> f64 {
        let clean = self.sigma_clean();
        let total = self.sigma_total();
        let jump_ratio = self.jump_ratio_fast();

        // At ratio=1: pure clean (no jumps)
        // At ratio=3: 67% total (jumps dominant)
        // At ratio=5: 80% total
        let jump_weight = 1.0 - (1.0 / jump_ratio.max(1.0));
        let jump_weight = jump_weight.clamp(0.0, 0.85);

        (1.0 - jump_weight) * clean + jump_weight * total
    }

    /// Fast jump ratio (detects recent jumps quickly)
    pub(crate) fn jump_ratio_fast(&self) -> f64 {
        self.fast.jump_ratio()
    }

    /// Medium jump ratio (more stable signal)
    #[allow(dead_code)]
    pub(crate) fn jump_ratio_medium(&self) -> f64 {
        self.medium.jump_ratio()
    }

    pub(crate) fn tick_count(&self) -> usize {
        self.tick_count
    }
}

// ============================================================================
// 4-State Volatility Regime with Hysteresis
// ============================================================================

/// Volatility regime classification.
///
/// Four states with hysteresis to prevent rapid switching:
/// - Low: Very quiet market (σ < 0.5 × baseline)
/// - Normal: Standard market conditions
/// - High: Elevated volatility (σ > 1.5 × baseline)
/// - Extreme: Crisis/toxic conditions (σ > 3 × baseline OR high jump ratio)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum VolatilityRegime {
    /// Very quiet market - can tighten spreads
    Low,
    /// Normal market conditions
    #[default]
    Normal,
    /// Elevated volatility - widen spreads
    High,
    /// Crisis/toxic - maximum caution, consider pulling quotes
    Extreme,
}

impl VolatilityRegime {
    /// Get spread multiplier for this regime.
    ///
    /// Used to scale spreads based on volatility state.
    pub(crate) fn spread_multiplier(&self) -> f64 {
        match self {
            Self::Low => 0.8,     // Tighter spreads in quiet markets
            Self::Normal => 1.0,  // Base case
            Self::High => 1.5,    // Wider spreads for elevated vol
            Self::Extreme => 2.5, // Much wider spreads in crisis
        }
    }

    /// Get gamma multiplier for this regime.
    ///
    /// Risk aversion increases with volatility.
    pub(crate) fn gamma_multiplier(&self) -> f64 {
        match self {
            Self::Low => 0.8,
            Self::Normal => 1.0,
            Self::High => 1.5,
            Self::Extreme => 3.0,
        }
    }

    /// Get Kelly fraction multiplier for this regime.
    ///
    /// In high volatility, we want to be more conservative (lower Kelly fraction).
    /// In low volatility, we can be more aggressive (higher Kelly fraction).
    ///
    /// Returns a multiplier to apply to the base Kelly fraction (0.25 default):
    /// - Low: 1.5x → 0.375 effective (can be more aggressive in quiet markets)
    /// - Normal: 1.0x → 0.25 effective (standard quarter Kelly)
    /// - High: 0.5x → 0.125 effective (more conservative)
    /// - Extreme: 0.25x → 0.0625 effective (very conservative, near flat)
    pub fn kelly_fraction_multiplier(&self) -> f64 {
        match self {
            Self::Low => 1.5,      // More aggressive in quiet markets
            Self::Normal => 1.0,   // Standard
            Self::High => 0.5,     // More conservative
            Self::Extreme => 0.25, // Very conservative
        }
    }

    /// Check if quotes should be pulled (extreme regime).
    #[allow(dead_code)]
    pub(crate) fn should_consider_pulling_quotes(&self) -> bool {
        matches!(self, Self::Extreme)
    }
}

/// Tracks volatility regime with asymmetric hysteresis to prevent rapid switching.
///
/// Transitions between states require sustained conditions to trigger,
/// preventing oscillation at boundaries.
///
/// Asymmetric hysteresis captures real market behavior:
/// - Vol spikes fast: Escalation to High/Extreme is faster (2 ticks)
/// - Vol mean-reverts slow: De-escalation to Normal/Low is slower (8 ticks)
#[derive(Debug)]
pub(crate) struct VolatilityRegimeTracker {
    /// Current regime state
    regime: VolatilityRegime,
    /// Baseline volatility (for regime thresholds)
    baseline_sigma: f64,
    /// Consecutive updates in potential new regime (for hysteresis)
    transition_count: u32,
    /// Minimum transitions for escalating to higher regime (fast: vol spikes)
    min_transitions_escalate: u32,
    /// Minimum transitions for de-escalating to lower regime (slow: vol mean-reverts)
    min_transitions_deescalate: u32,
    /// Thresholds relative to baseline
    low_threshold: f64, // σ < baseline × low_threshold → Low
    high_threshold: f64,    // σ > baseline × high_threshold → High
    extreme_threshold: f64, // σ > baseline × extreme_threshold → Extreme
    /// Jump ratio threshold for Extreme regime
    jump_threshold: f64,
    /// Pending regime (for hysteresis tracking)
    pending_regime: Option<VolatilityRegime>,
}

impl VolatilityRegimeTracker {
    pub(crate) fn new(baseline_sigma: f64) -> Self {
        Self {
            regime: VolatilityRegime::Normal,
            baseline_sigma,
            transition_count: 0,
            // Asymmetric hysteresis: escalate fast (2 ticks), de-escalate slow (8 ticks)
            // This matches market behavior: vol spikes quickly, mean-reverts slowly
            min_transitions_escalate: 2,
            min_transitions_deescalate: 8,
            low_threshold: 0.5,
            high_threshold: 1.5,
            extreme_threshold: 3.0,
            jump_threshold: 3.0,
            pending_regime: None,
        }
    }

    /// Determine if a transition is escalating (moving to higher risk regime).
    fn is_escalation(&self, from: VolatilityRegime, to: VolatilityRegime) -> bool {
        let from_level = match from {
            VolatilityRegime::Low => 0,
            VolatilityRegime::Normal => 1,
            VolatilityRegime::High => 2,
            VolatilityRegime::Extreme => 3,
        };
        let to_level = match to {
            VolatilityRegime::Low => 0,
            VolatilityRegime::Normal => 1,
            VolatilityRegime::High => 2,
            VolatilityRegime::Extreme => 3,
        };
        to_level > from_level
    }

    /// Get required transitions for a regime change (asymmetric).
    fn required_transitions(&self, from: VolatilityRegime, to: VolatilityRegime) -> u32 {
        if self.is_escalation(from, to) {
            self.min_transitions_escalate
        } else {
            self.min_transitions_deescalate
        }
    }

    /// Update regime based on current volatility and jump ratio.
    ///
    /// Uses asymmetric hysteresis:
    /// - Escalation (to higher risk): 2 ticks - react quickly to vol spikes
    /// - De-escalation (to lower risk): 8 ticks - confirm vol has truly subsided
    pub(crate) fn update(&mut self, sigma: f64, jump_ratio: f64) {
        // Determine target regime based on current conditions
        let target = self.classify(sigma, jump_ratio);

        // Check if target matches pending transition
        if let Some(pending) = self.pending_regime {
            if pending == target {
                self.transition_count += 1;
                // Use asymmetric hysteresis: fast escalation, slow de-escalation
                let required = self.required_transitions(self.regime, target);
                if self.transition_count >= required {
                    // Transition confirmed
                    if self.regime != target {
                        let direction = if self.is_escalation(self.regime, target) {
                            "escalation"
                        } else {
                            "de-escalation"
                        };
                        debug!(
                            from = ?self.regime,
                            to = ?target,
                            sigma = %format!("{:.6}", sigma),
                            jump_ratio = %format!("{:.2}", jump_ratio),
                            direction = direction,
                            ticks_required = required,
                            "Volatility regime transition"
                        );
                    }
                    self.regime = target;
                    self.pending_regime = None;
                    self.transition_count = 0;
                }
            } else {
                // Target changed, reset hysteresis
                self.pending_regime = Some(target);
                self.transition_count = 1;
            }
        } else if target != self.regime {
            // Start new pending transition
            self.pending_regime = Some(target);
            self.transition_count = 1;
        }
    }

    /// Classify conditions into target regime.
    fn classify(&self, sigma: f64, jump_ratio: f64) -> VolatilityRegime {
        // Jump ratio overrides to Extreme
        if jump_ratio > self.jump_threshold {
            return VolatilityRegime::Extreme;
        }

        // Volatility-based classification
        let sigma_ratio = sigma / self.baseline_sigma.max(1e-9);

        if sigma_ratio < self.low_threshold {
            VolatilityRegime::Low
        } else if sigma_ratio > self.extreme_threshold {
            VolatilityRegime::Extreme
        } else if sigma_ratio > self.high_threshold {
            VolatilityRegime::High
        } else {
            VolatilityRegime::Normal
        }
    }

    /// Get current regime.
    pub(crate) fn regime(&self) -> VolatilityRegime {
        self.regime
    }

    /// Update baseline volatility (e.g., from long-term EWMA).
    pub(crate) fn update_baseline(&mut self, new_baseline: f64) {
        if new_baseline > 1e-9 {
            // Slow update to baseline (EWMA with long half-life)
            self.baseline_sigma = 0.99 * self.baseline_sigma + 0.01 * new_baseline;
        }
    }
}

// ============================================================================
// Stochastic Volatility Parameters (First Principles Gap 2)
// ============================================================================

/// Stochastic volatility model parameters (Heston-style OU process).
///
/// Models volatility as a mean-reverting process:
/// dσ² = κ(θ - σ²)dt + ξσ² dZ  with Corr(dW, dZ) = ρ
///
/// Key formulas:
/// - Expected avg variance: E[∫₀ᵀ σ² dt]/T = θ + (σ₀² - θ)(1 - e^(-κT))/(κT)
/// - Leverage effect: Vol increases when returns are negative (ρ < 0)
///
/// Uses O(1) incremental statistics for calibration instead of O(n²) batch.
#[derive(Debug, Clone)]
pub struct StochasticVolParams {
    /// Current instantaneous variance (σ²)
    v_t: f64,
    /// Mean-reversion speed (κ)
    kappa_vol: f64,
    /// Long-run variance (θ)
    theta_vol: f64,
    /// Vol-of-vol (ξ)
    xi_vol: f64,
    /// Price-vol correlation (ρ, typically negative "leverage effect")
    rho: f64,
    /// O(1) incremental statistics tracker (replaces Vec-based calibration)
    incremental_stats: IncrementalVolStats,
    /// Variance history for calibration: (timestamp_ms, variance)
    /// NOTE: Kept for window expiration only, not for calibration computation
    variance_history: VecDeque<(u64, f64)>,
    /// Return history for correlation estimation
    /// NOTE: Kept for window expiration only, not for calibration computation
    return_history: VecDeque<(u64, f64)>,
    /// EWMA alpha for updates
    alpha: f64,
    /// History window (ms)
    window_ms: u64,
    /// Minimum observations for calibration
    min_observations: usize,
}

impl StochasticVolParams {
    /// Create with default parameters.
    pub(crate) fn new(default_sigma: f64) -> Self {
        let default_var = default_sigma.powi(2);
        Self {
            v_t: default_var,
            kappa_vol: 0.5,         // Moderate mean-reversion
            theta_vol: default_var, // Long-run = current
            xi_vol: 0.1,            // 10% vol-of-vol
            rho: -0.3,              // Typical leverage effect
            incremental_stats: IncrementalVolStats::new(),
            variance_history: VecDeque::with_capacity(500),
            return_history: VecDeque::with_capacity(500),
            alpha: 0.05,
            window_ms: 300_000, // 5 minutes
            min_observations: 20,
        }
    }

    /// Update with new variance observation.
    ///
    /// Uses O(1) incremental statistics instead of O(n²) batch calibration.
    pub(crate) fn on_variance(&mut self, timestamp_ms: u64, variance: f64, log_return: f64) {
        // Update current variance with EWMA
        self.v_t = self.alpha * variance + (1.0 - self.alpha) * self.v_t;

        // Update O(1) incremental statistics
        self.incremental_stats.update_variance(variance);
        self.incremental_stats.update_return(log_return);

        // Store history (for window expiration tracking only)
        self.variance_history.push_back((timestamp_ms, variance));
        self.return_history.push_back((timestamp_ms, log_return));

        // Expire old entries
        let cutoff = timestamp_ms.saturating_sub(self.window_ms);
        while self
            .variance_history
            .front()
            .map(|(t, _)| *t < cutoff)
            .unwrap_or(false)
        {
            self.variance_history.pop_front();
        }
        while self
            .return_history
            .front()
            .map(|(t, _)| *t < cutoff)
            .unwrap_or(false)
        {
            self.return_history.pop_front();
        }

        // Apply incremental calibration (O(1) instead of O(n²))
        if self.incremental_stats.is_ready(self.min_observations) {
            self.apply_incremental_calibration();
        }
    }

    /// Apply parameters from incremental statistics.
    ///
    /// This is O(1) - just reads from the pre-computed running statistics.
    fn apply_incremental_calibration(&mut self) {
        // Theta: long-run mean variance
        if let Some(theta) = self.incremental_stats.theta() {
            self.theta_vol = theta;
        }

        // Xi: vol-of-vol
        if let Some(xi) = self.incremental_stats.xi() {
            self.xi_vol = xi;
        }

        // Kappa: mean-reversion speed
        if let Some(kappa) = self.incremental_stats.kappa() {
            self.kappa_vol = kappa;
        }

        // Rho: price-vol correlation (leverage effect)
        if let Some(rho) = self.incremental_stats.rho() {
            self.rho = rho;
        }
    }

    /// Calibrate κ, θ, ξ, ρ from history using O(n²) brute-force method.
    ///
    /// NOTE: This is the original implementation kept for testing/validation.
    /// Production code uses `apply_incremental_calibration()` which is O(1).
    #[cfg(test)]
    #[allow(dead_code)]
    fn calibrate_bruteforce(&mut self) {
        let n = self.variance_history.len();
        if n < self.min_observations {
            return;
        }

        // Theta: long-run mean of variance
        let sum_var: f64 = self.variance_history.iter().map(|(_, v)| v).sum();
        self.theta_vol = sum_var / n as f64;

        // Xi (vol-of-vol): std dev of variance changes
        let var_changes: Vec<f64> = self
            .variance_history
            .iter()
            .zip(self.variance_history.iter().skip(1))
            .map(|((_, v1), (_, v2))| v2 - v1)
            .collect();
        if var_changes.len() > 1 {
            let mean_change: f64 = var_changes.iter().sum::<f64>() / var_changes.len() as f64;
            let sum_sq: f64 = var_changes.iter().map(|c| (c - mean_change).powi(2)).sum();
            self.xi_vol = (sum_sq / var_changes.len() as f64).sqrt();
        }

        // Kappa: estimate from autocorrelation decay
        // Simplified: use ratio of consecutive variance changes
        let mean_var = self.theta_vol;
        let deviations: Vec<f64> = self
            .variance_history
            .iter()
            .map(|(_, v)| v - mean_var)
            .collect();
        if deviations.len() > 1 {
            let autocov: f64 = deviations
                .iter()
                .zip(deviations.iter().skip(1))
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / (deviations.len() - 1) as f64;
            let variance: f64 =
                deviations.iter().map(|d| d.powi(2)).sum::<f64>() / deviations.len() as f64;
            if variance > 1e-12 {
                let autocorr = autocov / variance;
                // For OU process: autocorr ≈ exp(-κ × Δt)
                // Assuming Δt ≈ 1 second average between observations
                if autocorr > 0.0 && autocorr < 1.0 {
                    self.kappa_vol = -autocorr.ln().clamp(0.01, 5.0);
                }
            }
        }

        // Rho: correlation between returns and variance changes
        if self.return_history.len() == self.variance_history.len() && n > 2 {
            let returns: Vec<f64> = self.return_history.iter().map(|(_, r)| *r).collect();
            let var_deltas: Vec<f64> = self
                .variance_history
                .iter()
                .zip(self.variance_history.iter().skip(1))
                .map(|((_, v1), (_, v2))| v2 - v1)
                .collect();

            if !var_deltas.is_empty() {
                let n_pairs = var_deltas.len().min(returns.len() - 1);
                let mean_r: f64 = returns[..n_pairs].iter().sum::<f64>() / n_pairs as f64;
                let mean_dv: f64 = var_deltas[..n_pairs].iter().sum::<f64>() / n_pairs as f64;

                let cov: f64 = returns[..n_pairs]
                    .iter()
                    .zip(var_deltas[..n_pairs].iter())
                    .map(|(r, dv)| (r - mean_r) * (dv - mean_dv))
                    .sum::<f64>()
                    / n_pairs as f64;

                let std_r = (returns[..n_pairs]
                    .iter()
                    .map(|r| (r - mean_r).powi(2))
                    .sum::<f64>()
                    / n_pairs as f64)
                    .sqrt();
                let std_dv = (var_deltas[..n_pairs]
                    .iter()
                    .map(|dv| (dv - mean_dv).powi(2))
                    .sum::<f64>()
                    / n_pairs as f64)
                    .sqrt();

                if std_r > 1e-12 && std_dv > 1e-12 {
                    self.rho = (cov / (std_r * std_dv)).clamp(-0.95, 0.95);
                }
            }
        }
    }

    /// Get expected average variance over horizon using OU dynamics.
    ///
    /// E[∫₀ᵀ σ² dt]/T = θ + (σ₀² - θ)(1 - e^(-κT))/(κT)
    pub(crate) fn expected_avg_variance(&self, horizon_secs: f64) -> f64 {
        if horizon_secs < 1e-9 || self.kappa_vol < 1e-9 {
            return self.v_t;
        }

        let kt = self.kappa_vol * horizon_secs;
        let decay = 1.0 - (-kt).exp();
        self.theta_vol + (self.v_t - self.theta_vol) * decay / kt
    }

    /// Get expected average volatility (sqrt of variance).
    pub(crate) fn expected_avg_sigma(&self, horizon_secs: f64) -> f64 {
        self.expected_avg_variance(horizon_secs).sqrt()
    }

    /// Get leverage-adjusted volatility.
    ///
    /// When returns are negative and ρ < 0, volatility increases.
    pub(crate) fn leverage_adjusted_vol(&self, recent_return: f64) -> f64 {
        let base_vol = self.v_t.sqrt();

        // If return and rho have same sign, vol decreases
        // If opposite signs, vol increases (leverage effect)
        let adjustment = if recent_return * self.rho < 0.0 {
            // Return is negative, rho is negative: vol increases
            0.2 * recent_return.abs() / base_vol.max(1e-9)
        } else {
            0.0
        };

        base_vol * (1.0 + adjustment.clamp(0.0, 0.5))
    }

    // === Getters ===

    /// Current instantaneous variance.
    #[allow(dead_code)]
    pub(crate) fn v_t(&self) -> f64 {
        self.v_t
    }

    /// Current instantaneous volatility.
    pub(crate) fn sigma_t(&self) -> f64 {
        self.v_t.sqrt()
    }

    /// Mean-reversion speed.
    pub(crate) fn kappa(&self) -> f64 {
        self.kappa_vol
    }

    /// Long-run variance.
    #[allow(dead_code)]
    pub(crate) fn theta(&self) -> f64 {
        self.theta_vol
    }

    /// Long-run volatility.
    pub(crate) fn theta_sigma(&self) -> f64 {
        self.theta_vol.sqrt()
    }

    /// Vol-of-vol.
    pub(crate) fn xi(&self) -> f64 {
        self.xi_vol
    }

    /// Price-vol correlation (leverage effect).
    pub(crate) fn rho(&self) -> f64 {
        self.rho
    }

    /// Check if calibrated.
    pub(crate) fn is_calibrated(&self) -> bool {
        self.variance_history.len() >= self.min_observations
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // IncrementalVolStats Tests
    // ========================================================================

    #[test]
    fn test_incremental_stats_new() {
        let stats = IncrementalVolStats::new();
        assert_eq!(stats.count(), 0);
        assert!(!stats.is_ready(1));
        assert!(stats.theta().is_none());
        assert!(stats.xi().is_none());
        assert!(stats.kappa().is_none());
        assert!(stats.rho().is_none());
    }

    #[test]
    fn test_incremental_stats_single_observation() {
        let mut stats = IncrementalVolStats::new();
        stats.update_variance(0.0004); // σ² = 0.0004 (σ = 0.02 = 2%)

        assert_eq!(stats.count(), 1);
        assert!(stats.is_ready(1));

        // Theta should be the single value
        assert!((stats.theta().unwrap() - 0.0004).abs() < 1e-9);

        // Xi and kappa need more observations
        assert!(stats.xi().is_none()); // Need at least 2 for dV
        assert!(stats.kappa().is_none()); // Need at least 3 for autocorr
    }

    #[test]
    fn test_incremental_stats_theta_mean() {
        let mut stats = IncrementalVolStats::new();

        // Add variances: 0.0001, 0.0002, 0.0003, 0.0004
        // Mean should be 0.00025
        for i in 1..=4 {
            stats.update_variance(0.0001 * i as f64);
        }

        let theta = stats.theta().unwrap();
        assert!(
            (theta - 0.00025).abs() < 1e-9,
            "Expected theta=0.00025, got {}",
            theta
        );
    }

    #[test]
    fn test_incremental_stats_xi_volatility_of_volatility() {
        let mut stats = IncrementalVolStats::new();

        // Constant variance changes should give low xi
        // Variances: 0.0001, 0.0002, 0.0003, 0.0004, 0.0005
        // Changes: 0.0001, 0.0001, 0.0001, 0.0001 (constant)
        for i in 1..=5 {
            stats.update_variance(0.0001 * i as f64);
        }

        let xi = stats.xi().unwrap();
        // With constant changes, std dev of changes should be near 0
        assert!(
            xi < 1e-6,
            "Expected low xi for constant changes, got {}",
            xi
        );
    }

    #[test]
    fn test_incremental_stats_xi_varying() {
        let mut stats = IncrementalVolStats::new();

        // Variable variance changes
        let variances = [0.0001, 0.0004, 0.0002, 0.0005, 0.0001, 0.0006];
        for v in variances {
            stats.update_variance(v);
        }

        let xi = stats.xi().unwrap();
        // With varying changes, xi should be positive
        assert!(xi > 0.0, "Expected positive xi for varying changes");
    }

    #[test]
    fn test_incremental_stats_kappa_mean_reversion() {
        let mut stats = IncrementalVolStats::new();

        // Simulate mean-reverting process
        // Variances oscillating around 0.0004
        let variances = [
            0.0006, 0.0005, 0.0004, 0.0003, 0.0004, 0.0005, 0.0004, 0.0003, 0.0004, 0.0005,
        ];
        for v in variances {
            stats.update_variance(v);
        }

        // Should get some kappa estimate
        if let Some(kappa) = stats.kappa() {
            assert!(
                kappa > 0.0 && kappa < 5.0,
                "Kappa should be in valid range, got {}",
                kappa
            );
        }
    }

    #[test]
    fn test_incremental_stats_reset() {
        let mut stats = IncrementalVolStats::new();

        // Add some data
        for i in 1..=10 {
            stats.update_variance(0.0001 * i as f64);
        }

        assert!(stats.count() > 0);

        // Reset
        stats.reset();

        assert_eq!(stats.count(), 0);
        assert!(stats.theta().is_none());
    }

    #[test]
    fn test_incremental_stats_return_tracking() {
        let mut stats = IncrementalVolStats::new();

        // Add synchronized variance and return observations
        for i in 0..20 {
            let variance = 0.0004 + 0.0001 * (i as f64 / 10.0).sin();
            let log_return = 0.001 * (i as f64 / 5.0).cos();
            stats.update_variance(variance);
            stats.update_return(log_return);
        }

        // Should have all estimates
        assert!(stats.theta().is_some());
        assert!(stats.xi().is_some());
        // Rho might or might not be computable depending on data
    }

    // ========================================================================
    // VolatilityRegime Tests
    // ========================================================================

    #[test]
    fn test_volatility_regime_multipliers() {
        assert!((VolatilityRegime::Low.spread_multiplier() - 0.8).abs() < 0.01);
        assert!((VolatilityRegime::Normal.spread_multiplier() - 1.0).abs() < 0.01);
        assert!((VolatilityRegime::High.spread_multiplier() - 1.5).abs() < 0.01);
        assert!((VolatilityRegime::Extreme.spread_multiplier() - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_volatility_regime_gamma_multipliers() {
        assert!((VolatilityRegime::Low.gamma_multiplier() - 0.8).abs() < 0.01);
        assert!((VolatilityRegime::Normal.gamma_multiplier() - 1.0).abs() < 0.01);
        assert!((VolatilityRegime::High.gamma_multiplier() - 1.5).abs() < 0.01);
        assert!((VolatilityRegime::Extreme.gamma_multiplier() - 3.0).abs() < 0.01);
    }

    // ========================================================================
    // StochasticVolParams Tests
    // ========================================================================

    #[test]
    fn test_stochastic_vol_params_new() {
        let params = StochasticVolParams::new(0.02); // 2% default vol

        assert!((params.sigma_t() - 0.02).abs() < 0.001);
        assert!((params.theta_sigma() - 0.02).abs() < 0.001);
    }

    #[test]
    fn test_stochastic_vol_params_variance_update() {
        let mut params = StochasticVolParams::new(0.02);

        // Add variance observations
        for i in 0..30 {
            let timestamp = i as u64 * 1000; // 1 second apart
            let variance = 0.0004 + 0.0001 * ((i as f64) / 10.0).sin();
            let log_return = 0.001 * ((i as f64) / 5.0).cos();
            params.on_variance(timestamp, variance, log_return);
        }

        // Should be calibrated now
        assert!(params.is_calibrated());
    }

    #[test]
    fn test_stochastic_vol_expected_avg_variance() {
        let params = StochasticVolParams::new(0.02);

        // At t=0, expected avg variance should be near current
        let short_horizon = params.expected_avg_variance(0.1);
        let current_var = 0.02_f64.powi(2);
        assert!(
            (short_horizon - current_var).abs() < 0.0001,
            "Short horizon expected variance should be near current"
        );
    }
}
