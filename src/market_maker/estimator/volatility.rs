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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VolatilityRegime {
    /// Very quiet market - can tighten spreads
    Low,
    /// Normal market conditions
    Normal,
    /// Elevated volatility - widen spreads
    High,
    /// Crisis/toxic - maximum caution, consider pulling quotes
    Extreme,
}

impl Default for VolatilityRegime {
    fn default() -> Self {
        Self::Normal
    }
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
            Self::Low => 1.5,     // More aggressive in quiet markets
            Self::Normal => 1.0,  // Standard
            Self::High => 0.5,    // More conservative
            Self::Extreme => 0.25, // Very conservative
        }
    }

    /// Check if quotes should be pulled (extreme regime).
    #[allow(dead_code)]
    pub(crate) fn should_consider_pulling_quotes(&self) -> bool {
        matches!(self, Self::Extreme)
    }
}

/// Tracks volatility regime with hysteresis to prevent rapid switching.
///
/// Transitions between states require sustained conditions to trigger,
/// preventing oscillation at boundaries.
#[derive(Debug)]
pub(crate) struct VolatilityRegimeTracker {
    /// Current regime state
    regime: VolatilityRegime,
    /// Baseline volatility (for regime thresholds)
    baseline_sigma: f64,
    /// Consecutive updates in potential new regime (for hysteresis)
    transition_count: u32,
    /// Minimum transitions before state change (hysteresis parameter)
    min_transitions: u32,
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
            min_transitions: 5, // Require 5 consecutive updates before transition
            low_threshold: 0.5,
            high_threshold: 1.5,
            extreme_threshold: 3.0,
            jump_threshold: 3.0,
            pending_regime: None,
        }
    }

    /// Update regime based on current volatility and jump ratio.
    pub(crate) fn update(&mut self, sigma: f64, jump_ratio: f64) {
        // Determine target regime based on current conditions
        let target = self.classify(sigma, jump_ratio);

        // Check if target matches pending transition
        if let Some(pending) = self.pending_regime {
            if pending == target {
                self.transition_count += 1;
                if self.transition_count >= self.min_transitions {
                    // Transition confirmed
                    if self.regime != target {
                        debug!(
                            from = ?self.regime,
                            to = ?target,
                            sigma = %format!("{:.6}", sigma),
                            jump_ratio = %format!("{:.2}", jump_ratio),
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
    /// Variance history for calibration: (timestamp_ms, variance)
    variance_history: VecDeque<(u64, f64)>,
    /// Return history for correlation estimation
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
            variance_history: VecDeque::with_capacity(500),
            return_history: VecDeque::with_capacity(500),
            alpha: 0.05,
            window_ms: 300_000, // 5 minutes
            min_observations: 20,
        }
    }

    /// Update with new variance observation.
    pub(crate) fn on_variance(&mut self, timestamp_ms: u64, variance: f64, log_return: f64) {
        // Update current variance with EWMA
        self.v_t = self.alpha * variance + (1.0 - self.alpha) * self.v_t;

        // Store history
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

        // Periodically calibrate parameters
        if self.variance_history.len() >= self.min_observations
            && self.variance_history.len().is_multiple_of(10)
        {
            self.calibrate();
        }
    }

    /// Calibrate κ, θ, ξ, ρ from history.
    fn calibrate(&mut self) {
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
