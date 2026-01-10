//! O(1) incremental statistics for stochastic volatility calibration.
//!
//! Uses Welford's algorithm to maintain running statistics that update in O(1) per observation.

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
    pub(crate) fn update_variance(&mut self, variance: f64) {
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
    pub(crate) fn update_return(&mut self, log_return: f64) {
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
    pub(crate) fn theta(&self) -> Option<f64> {
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
    pub(crate) fn xi(&self) -> Option<f64> {
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
    pub(crate) fn kappa(&self) -> Option<f64> {
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
    pub(crate) fn rho(&self) -> Option<f64> {
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
    pub(crate) fn count(&self) -> usize {
        self.n
    }

    /// Check if enough observations for calibration.
    #[inline]
    pub(crate) fn is_ready(&self, min_obs: usize) -> bool {
        self.n >= min_obs
    }

    /// Reset all statistics.
    #[allow(dead_code)]
    pub(crate) fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for IncrementalVolStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
