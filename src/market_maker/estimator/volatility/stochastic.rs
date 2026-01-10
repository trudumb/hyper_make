//! Stochastic volatility model parameters.
//!
//! Heston-style OU process for volatility modeling with O(1) incremental calibration.

use std::collections::VecDeque;

use super::incremental::IncrementalVolStats;

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

#[cfg(test)]
mod tests {
    use super::*;

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
