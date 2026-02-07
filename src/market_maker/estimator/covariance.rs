//! Parameter Covariance Tracking Module
//!
//! Tracks joint (κ, σ) uncertainty for proper spread uncertainty quantification.

// Allow dead code since this is V2 infrastructure being built incrementally
#![allow(dead_code)]
//!
//! In GLFT, the optimal spread δ* depends on both κ and σ:
//! δ* = (1/γ) × ln(1 + γ/κ)
//!
//! If κ and σ are correlated (they often are during volatility regimes),
//! the uncertainty in spread calculations should account for this correlation.

use super::tick_ewma::TickEWMA;

/// Tracks rolling covariance between two parameters using EWMA.
///
/// Maintains online estimates of:
/// - Mean of each parameter
/// - Variance of each parameter
/// - Covariance between parameters
/// - Correlation coefficient
#[derive(Debug, Clone)]
pub(crate) struct ParameterCovariance {
    /// EWMA of κ
    mean_kappa: f64,
    /// EWMA of σ
    mean_sigma: f64,
    /// EWMA of κ²
    mean_kappa_sq: f64,
    /// EWMA of σ²
    mean_sigma_sq: f64,
    /// EWMA of κ × σ
    mean_kappa_sigma: f64,
    /// EWMA decay factor
    alpha: f64,
    /// Observation count
    observation_count: usize,
    /// Minimum observations before covariance is valid
    min_observations: usize,
}

impl ParameterCovariance {
    /// Create a new covariance tracker.
    ///
    /// # Arguments
    /// * `alpha` - EWMA decay factor (e.g., 0.02 for 50-tick half-life)
    pub(crate) fn new(alpha: f64) -> Self {
        Self {
            mean_kappa: 0.0,
            mean_sigma: 0.0,
            mean_kappa_sq: 0.0,
            mean_sigma_sq: 0.0,
            mean_kappa_sigma: 0.0,
            alpha,
            observation_count: 0,
            min_observations: 20,
        }
    }

    /// Create with half-life in ticks.
    pub(crate) fn with_half_life(half_life_ticks: f64) -> Self {
        let alpha = 1.0 - 2.0_f64.powf(-1.0 / half_life_ticks);
        Self::new(alpha)
    }

    /// Update with new (κ, σ) observation.
    pub(crate) fn update(&mut self, kappa: f64, sigma: f64) {
        if self.observation_count == 0 {
            // Initialize with first observation
            self.mean_kappa = kappa;
            self.mean_sigma = sigma;
            self.mean_kappa_sq = kappa * kappa;
            self.mean_sigma_sq = sigma * sigma;
            self.mean_kappa_sigma = kappa * sigma;
        } else {
            // EWMA update
            self.mean_kappa = self.alpha * kappa + (1.0 - self.alpha) * self.mean_kappa;
            self.mean_sigma = self.alpha * sigma + (1.0 - self.alpha) * self.mean_sigma;
            self.mean_kappa_sq =
                self.alpha * (kappa * kappa) + (1.0 - self.alpha) * self.mean_kappa_sq;
            self.mean_sigma_sq =
                self.alpha * (sigma * sigma) + (1.0 - self.alpha) * self.mean_sigma_sq;
            self.mean_kappa_sigma =
                self.alpha * (kappa * sigma) + (1.0 - self.alpha) * self.mean_kappa_sigma;
        }
        self.observation_count += 1;
    }

    /// Get variance of κ.
    pub(crate) fn variance_kappa(&self) -> f64 {
        (self.mean_kappa_sq - self.mean_kappa.powi(2)).max(0.0)
    }

    /// Get variance of σ.
    pub(crate) fn variance_sigma(&self) -> f64 {
        (self.mean_sigma_sq - self.mean_sigma.powi(2)).max(0.0)
    }

    /// Get covariance Cov(κ, σ).
    pub(crate) fn covariance(&self) -> f64 {
        self.mean_kappa_sigma - self.mean_kappa * self.mean_sigma
    }

    /// Get correlation coefficient ρ(κ, σ) ∈ [-1, 1].
    pub(crate) fn correlation(&self) -> f64 {
        let var_k = self.variance_kappa();
        let var_s = self.variance_sigma();
        let denom = (var_k * var_s).sqrt();
        if denom < 1e-12 {
            return 0.0;
        }
        (self.covariance() / denom).clamp(-1.0, 1.0)
    }

    /// Get standard deviation of κ.
    pub(crate) fn std_kappa(&self) -> f64 {
        self.variance_kappa().sqrt()
    }

    /// Get standard deviation of σ.
    pub(crate) fn std_sigma(&self) -> f64 {
        self.variance_sigma().sqrt()
    }

    /// Get mean κ.
    pub(crate) fn mean_kappa(&self) -> f64 {
        self.mean_kappa
    }

    /// Get mean σ.
    pub(crate) fn mean_sigma(&self) -> f64 {
        self.mean_sigma
    }

    /// Check if covariance estimate is valid.
    pub(crate) fn is_valid(&self) -> bool {
        self.observation_count >= self.min_observations
    }

    /// Compute spread uncertainty from parameter uncertainty.
    ///
    /// Using delta method for δ* = (1/γ) × ln(1 + γ/κ):
    /// ∂δ*/∂κ ≈ -1/(κ(κ+γ)) for small γ/κ
    ///
    /// Var(δ*) ≈ (∂δ*/∂κ)² Var(κ)
    pub(crate) fn spread_uncertainty(&self, gamma: f64) -> f64 {
        let kappa = self.mean_kappa;
        if kappa < 1e-6 || gamma < 1e-12 {
            return 0.0;
        }

        // Derivative of GLFT spread w.r.t. kappa
        // δ* = (1/γ) × ln(1 + γ/κ)
        // dδ*/dκ = (1/γ) × (-γ/κ²) / (1 + γ/κ) = -1/(κ(κ + γ))
        let d_delta_d_kappa = -1.0 / (kappa * (kappa + gamma));

        // Standard error of spread
        (d_delta_d_kappa.powi(2) * self.variance_kappa()).sqrt()
    }

    /// Reset to initial state.
    pub(crate) fn reset(&mut self) {
        self.mean_kappa = 0.0;
        self.mean_sigma = 0.0;
        self.mean_kappa_sq = 0.0;
        self.mean_sigma_sq = 0.0;
        self.mean_kappa_sigma = 0.0;
        self.observation_count = 0;
    }
}

/// Extended covariance tracker for multiple parameters.
///
/// Tracks covariances among (κ, σ, λ_jump) for full uncertainty propagation.
#[derive(Debug, Clone)]
pub(crate) struct MultiParameterCovariance {
    /// (κ, σ) covariance
    kappa_sigma: ParameterCovariance,

    /// EWMA of λ (jump intensity)
    mean_lambda: TickEWMA,

    /// EWMA of λ × κ
    mean_lambda_kappa: TickEWMA,

    /// EWMA of λ × σ
    mean_lambda_sigma: TickEWMA,
}

impl MultiParameterCovariance {
    pub(crate) fn new(half_life_ticks: f64) -> Self {
        Self {
            kappa_sigma: ParameterCovariance::with_half_life(half_life_ticks),
            mean_lambda: TickEWMA::new_uninitialized(half_life_ticks),
            mean_lambda_kappa: TickEWMA::new_uninitialized(half_life_ticks),
            mean_lambda_sigma: TickEWMA::new_uninitialized(half_life_ticks),
        }
    }

    pub(crate) fn update(&mut self, kappa: f64, sigma: f64, lambda: f64) {
        self.kappa_sigma.update(kappa, sigma);
        self.mean_lambda.update(lambda);
        self.mean_lambda_kappa.update(lambda * kappa);
        self.mean_lambda_sigma.update(lambda * sigma);
    }

    pub(crate) fn correlation_kappa_sigma(&self) -> f64 {
        self.kappa_sigma.correlation()
    }

    pub(crate) fn is_valid(&self) -> bool {
        self.kappa_sigma.is_valid()
    }
}

// ============================================================================
// Feature Correlation Matrix (Phase 4: Feature Engineering)
// ============================================================================

/// Feature names for correlation tracking.
pub(crate) const FEATURE_NAMES: [&str; 10] = [
    "kappa",
    "sigma",
    "momentum",
    "flow_imbalance",
    "book_imbalance",
    "jump_ratio",
    "hawkes_intensity",
    "spread_regime",
    "funding_rate",
    "volatility_ratio",
];

/// Full feature correlation matrix tracker.
///
/// Tracks NxN correlation matrix using online EWMA updates.
/// Use cases:
/// - Detect redundant signals (correlation > 0.8)
/// - Weight decorrelated signals higher
/// - Alert when regime causes correlation breakdown
#[derive(Debug, Clone)]
pub(crate) struct FeatureCorrelationTracker {
    /// Feature count
    n: usize,
    /// Feature names
    feature_names: Vec<String>,
    /// EWMA means for each feature
    means: Vec<f64>,
    /// EWMA of squared features (for variance)
    mean_squares: Vec<f64>,
    /// EWMA of products (flattened upper triangle, for covariance)
    /// Index: i * n + j for i < j
    mean_products: Vec<f64>,
    /// EWMA decay factor
    alpha: f64,
    /// Observation count
    observation_count: usize,
    /// Minimum observations before valid
    min_observations: usize,
}

impl FeatureCorrelationTracker {
    /// Create a new tracker for N features.
    pub(crate) fn new(feature_names: &[&str], alpha: f64) -> Self {
        let n = feature_names.len();
        let n_products = n * (n - 1) / 2; // Upper triangle count

        Self {
            n,
            feature_names: feature_names.iter().map(|s| s.to_string()).collect(),
            means: vec![0.0; n],
            mean_squares: vec![0.0; n],
            mean_products: vec![0.0; n_products],
            alpha,
            observation_count: 0,
            min_observations: 30,
        }
    }

    /// Create with default features and half-life.
    pub(crate) fn default_features(half_life_ticks: f64) -> Self {
        let alpha = 1.0 - 2.0_f64.powf(-1.0 / half_life_ticks);
        Self::new(&FEATURE_NAMES, alpha)
    }

    /// Get index into mean_products for pair (i, j) where i < j.
    fn product_index(&self, i: usize, j: usize) -> usize {
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        // Upper triangle: sum of (n-1) + (n-2) + ... + (n-i) elements before row i
        // Then add (j - i - 1) for position within row
        i * self.n - (i * (i + 1)) / 2 + j - i - 1
    }

    /// Update with a new feature vector.
    ///
    /// # Arguments
    /// * `features` - Feature values in same order as feature_names
    pub(crate) fn update(&mut self, features: &[f64]) {
        if features.len() != self.n {
            return;
        }

        // Validate all features are finite
        if !features.iter().all(|f| f.is_finite()) {
            return;
        }

        if self.observation_count == 0 {
            // Initialize with first observation
            for (i, &feat) in features.iter().enumerate().take(self.n) {
                self.means[i] = feat;
                self.mean_squares[i] = feat * feat;
            }
            for i in 0..self.n {
                for j in (i + 1)..self.n {
                    let idx = self.product_index(i, j);
                    self.mean_products[idx] = features[i] * features[j];
                }
            }
        } else {
            // EWMA update
            let one_minus_alpha = 1.0 - self.alpha;
            for (i, &feat) in features.iter().enumerate().take(self.n) {
                self.means[i] = self.alpha * feat + one_minus_alpha * self.means[i];
                self.mean_squares[i] =
                    self.alpha * feat * feat + one_minus_alpha * self.mean_squares[i];
            }
            for i in 0..self.n {
                for j in (i + 1)..self.n {
                    let idx = self.product_index(i, j);
                    self.mean_products[idx] = self.alpha * features[i] * features[j]
                        + one_minus_alpha * self.mean_products[idx];
                }
            }
        }
        self.observation_count += 1;
    }

    /// Get variance of feature i.
    pub(crate) fn variance(&self, i: usize) -> f64 {
        if i >= self.n {
            return 0.0;
        }
        (self.mean_squares[i] - self.means[i].powi(2)).max(0.0)
    }

    /// Get covariance between features i and j.
    pub(crate) fn covariance(&self, i: usize, j: usize) -> f64 {
        if i >= self.n || j >= self.n {
            return 0.0;
        }
        if i == j {
            return self.variance(i);
        }

        let idx = self.product_index(i, j);
        self.mean_products[idx] - self.means[i] * self.means[j]
    }

    /// Get correlation between features i and j.
    pub(crate) fn correlation(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return 1.0;
        }

        let var_i = self.variance(i);
        let var_j = self.variance(j);
        let denom = (var_i * var_j).sqrt();

        if denom < 1e-12 {
            return 0.0;
        }

        (self.covariance(i, j) / denom).clamp(-1.0, 1.0)
    }

    /// Get full NxN correlation matrix.
    pub(crate) fn correlation_matrix(&self) -> Vec<Vec<f64>> {
        let mut matrix = vec![vec![0.0; self.n]; self.n];
        for (i, row) in matrix.iter_mut().enumerate().take(self.n) {
            for (j, cell) in row.iter_mut().enumerate().take(self.n) {
                *cell = self.correlation(i, j);
            }
        }
        matrix
    }

    /// Compute condition number of correlation matrix.
    ///
    /// High condition number (> 30) indicates multicollinearity.
    /// Uses power iteration to estimate largest/smallest eigenvalues.
    pub(crate) fn condition_number(&self) -> f64 {
        if self.observation_count < self.min_observations {
            return 1.0;
        }

        let matrix = self.correlation_matrix();

        // Power iteration for largest eigenvalue
        let lambda_max = self.power_iteration(&matrix, 50);

        // For condition number, we need smallest eigenvalue too
        // Use inverse power iteration (solve Ax = b iteratively)
        let lambda_min = self.inverse_power_iteration(&matrix, 50);

        if lambda_min.abs() < 1e-10 {
            return f64::INFINITY; // Singular matrix
        }

        lambda_max.abs() / lambda_min.abs()
    }

    /// Power iteration to find largest eigenvalue.
    fn power_iteration(&self, matrix: &[Vec<f64>], iterations: usize) -> f64 {
        let n = matrix.len();
        let mut v: Vec<f64> = (0..n).map(|i| 1.0 / (i + 1) as f64).collect();

        for _ in 0..iterations {
            // Matrix-vector multiply
            let mut w = vec![0.0; n];
            for i in 0..n {
                for (j, &vj) in v.iter().enumerate().take(n) {
                    w[i] += matrix[i][j] * vj;
                }
            }

            // Normalize
            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-10 {
                return 0.0;
            }
            v = w.iter().map(|x| x / norm).collect();
        }

        // Rayleigh quotient for eigenvalue estimate
        let mut numer = 0.0;
        let mut denom = 0.0;
        for (i, &vi) in v.iter().enumerate().take(n) {
            let mut av_i = 0.0;
            for (j, &vj) in v.iter().enumerate().take(n) {
                av_i += matrix[i][j] * vj;
            }
            numer += vi * av_i;
            denom += vi * vi;
        }

        numer / denom.max(1e-10)
    }

    /// Inverse power iteration to find smallest eigenvalue.
    fn inverse_power_iteration(&self, matrix: &[Vec<f64>], iterations: usize) -> f64 {
        let n = matrix.len();

        // Shift matrix slightly to avoid singularity
        let mut shifted = matrix.to_vec();
        for (i, row) in shifted.iter_mut().enumerate().take(n) {
            row[i] += 0.01;
        }

        let mut v: Vec<f64> = (0..n).map(|i| 1.0 / (i + 1) as f64).collect();

        for _ in 0..iterations {
            // Solve (A + 0.01*I) * w = v using Gauss-Seidel (simple iterative solver)
            let mut w = v.clone();
            for _ in 0..10 {
                for i in 0..n {
                    let mut sum = v[i];
                    for (j, &wj) in w.iter().enumerate().take(n) {
                        if j != i {
                            sum -= shifted[i][j] * wj;
                        }
                    }
                    w[i] = sum / shifted[i][i].max(1e-10);
                }
            }

            // Normalize
            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-10 {
                return 0.0;
            }
            v = w.iter().map(|x| x / norm).collect();
        }

        // Inverse of Rayleigh quotient gives smallest eigenvalue of original matrix
        let mut numer = 0.0;
        let mut denom = 0.0;
        for (i, &vi) in v.iter().enumerate().take(n) {
            let mut av_i = 0.0;
            for (j, &vj) in v.iter().enumerate().take(n) {
                av_i += matrix[i][j] * vj;
            }
            numer += vi * av_i;
            denom += vi * vi;
        }

        numer / denom.max(1e-10)
    }

    /// Compute variance inflation factors.
    ///
    /// VIF > 5 indicates potential multicollinearity.
    /// VIF > 10 is severe multicollinearity.
    pub(crate) fn variance_inflation_factors(&self) -> Vec<f64> {
        let mut vifs = vec![1.0; self.n];

        if self.observation_count < self.min_observations {
            return vifs;
        }

        for (i, vif) in vifs.iter_mut().enumerate().take(self.n) {
            // VIF_i = 1 / (1 - R²_i)
            // where R²_i is the R² from regressing feature i on all others
            // Approximate with max correlation squared
            let mut max_corr_sq: f64 = 0.0;
            for j in 0..self.n {
                if i != j {
                    let corr = self.correlation(i, j);
                    max_corr_sq = max_corr_sq.max(corr * corr);
                }
            }

            // VIF approximation using max correlation
            if max_corr_sq < 0.999 {
                *vif = 1.0 / (1.0 - max_corr_sq);
            } else {
                *vif = f64::INFINITY;
            }
        }

        vifs
    }

    /// Find highly correlated feature pairs.
    ///
    /// Returns pairs with |correlation| > threshold.
    pub(crate) fn highly_correlated_pairs(&self, threshold: f64) -> Vec<(usize, usize, f64)> {
        let mut pairs = Vec::new();

        if self.observation_count < self.min_observations {
            return pairs;
        }

        for i in 0..self.n {
            for j in (i + 1)..self.n {
                let corr = self.correlation(i, j);
                if corr.abs() > threshold {
                    pairs.push((i, j, corr));
                }
            }
        }

        // Sort by absolute correlation (highest first)
        pairs.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());
        pairs
    }

    /// Get feature name by index.
    pub(crate) fn feature_name(&self, i: usize) -> &str {
        if i < self.feature_names.len() {
            &self.feature_names[i]
        } else {
            "unknown"
        }
    }

    /// Get index by feature name.
    pub(crate) fn feature_index(&self, name: &str) -> Option<usize> {
        self.feature_names.iter().position(|n| n == name)
    }

    /// Check if enough observations.
    pub(crate) fn is_valid(&self) -> bool {
        self.observation_count >= self.min_observations
    }

    /// Get observation count.
    pub(crate) fn observation_count(&self) -> usize {
        self.observation_count
    }

    /// Reset tracker.
    pub(crate) fn reset(&mut self) {
        self.means.fill(0.0);
        self.mean_squares.fill(0.0);
        self.mean_products.fill(0.0);
        self.observation_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_covariance_positive_correlation() {
        let mut cov = ParameterCovariance::with_half_life(10.0);

        // Feed positively correlated data: high κ with high σ
        for i in 0..100 {
            let kappa = 500.0 + 100.0 * (i as f64 / 100.0);
            let sigma = 0.0001 + 0.00005 * (i as f64 / 100.0);
            cov.update(kappa, sigma);
        }

        assert!(cov.is_valid());
        assert!(
            cov.correlation() > 0.5,
            "Positively correlated data should have positive correlation"
        );
    }

    #[test]
    fn test_covariance_negative_correlation() {
        let mut cov = ParameterCovariance::with_half_life(10.0);

        // Feed negatively correlated data: high κ with low σ
        for i in 0..100 {
            let kappa = 500.0 + 100.0 * (i as f64 / 100.0);
            let sigma = 0.00015 - 0.00005 * (i as f64 / 100.0);
            cov.update(kappa, sigma);
        }

        assert!(cov.is_valid());
        assert!(
            cov.correlation() < -0.5,
            "Negatively correlated data should have negative correlation"
        );
    }

    #[test]
    fn test_covariance_uncorrelated() {
        let mut cov = ParameterCovariance::with_half_life(20.0);

        // Feed uncorrelated data (κ oscillates, σ is random-ish)
        for i in 0..200 {
            let kappa = 500.0 + 50.0 * ((i as f64 * 0.3).sin());
            let sigma = 0.0001 + 0.00002 * ((i as f64 * 0.7).cos());
            cov.update(kappa, sigma);
        }

        assert!(cov.is_valid());
        // Should be roughly uncorrelated (close to 0)
        assert!(
            cov.correlation().abs() < 0.5,
            "Uncorrelated data should have near-zero correlation"
        );
    }

    #[test]
    fn test_spread_uncertainty() {
        let mut cov = ParameterCovariance::with_half_life(10.0);

        // Feed data with variance in kappa
        for i in 0..100 {
            let kappa = 500.0 + 50.0 * ((i as f64 * 0.5).sin());
            cov.update(kappa, 0.0001);
        }

        let spread_std = cov.spread_uncertainty(0.3);
        assert!(spread_std > 0.0, "Should have positive spread uncertainty");
        assert!(spread_std < 0.01, "Spread uncertainty should be reasonable");
    }

    #[test]
    fn test_variance_calculation() {
        let mut cov = ParameterCovariance::new(0.1);

        // Feed constant data - variance should be 0
        for _ in 0..50 {
            cov.update(500.0, 0.0001);
        }

        assert!(
            cov.variance_kappa() < 1e-6,
            "Constant data should have near-zero variance"
        );
        assert!(
            cov.variance_sigma() < 1e-12,
            "Constant data should have near-zero variance"
        );
    }

    #[test]
    fn test_feature_correlation_tracker() {
        let features = ["a", "b", "c"];
        let mut tracker = FeatureCorrelationTracker::new(&features, 0.1);

        // Feed correlated data: a and b positively correlated, c independent
        for i in 0..100 {
            let a = i as f64;
            let b = a * 0.9 + 5.0; // Highly correlated with a
            let c = (i as f64 * 0.3).sin() * 10.0; // Independent
            tracker.update(&[a, b, c]);
        }

        assert!(tracker.is_valid());

        // a-b correlation should be high
        let corr_ab = tracker.correlation(0, 1);
        assert!(
            corr_ab > 0.9,
            "a-b should be highly correlated: {}",
            corr_ab
        );

        // a-c correlation should be low
        let corr_ac = tracker.correlation(0, 2);
        assert!(
            corr_ac.abs() < 0.5,
            "a-c should be weakly correlated: {}",
            corr_ac
        );

        // Diagonal should be 1
        assert!((tracker.correlation(0, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_correlation_matrix() {
        let features = ["x", "y"];
        let mut tracker = FeatureCorrelationTracker::new(&features, 0.1);

        // Perfect positive correlation
        for i in 0..50 {
            let x = i as f64;
            let y = x * 2.0;
            tracker.update(&[x, y]);
        }

        let matrix = tracker.correlation_matrix();
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);

        // Diagonal should be 1
        assert!((matrix[0][0] - 1.0).abs() < 1e-6);
        assert!((matrix[1][1] - 1.0).abs() < 1e-6);

        // Off-diagonal should be ~1 (perfect correlation)
        assert!(matrix[0][1] > 0.99);
        assert!(matrix[1][0] > 0.99);
    }

    #[test]
    fn test_highly_correlated_pairs() {
        let features = ["a", "b", "c", "d"];
        let mut tracker = FeatureCorrelationTracker::new(&features, 0.1);

        // a-b highly correlated, c-d highly correlated, others independent
        for i in 0..100 {
            let a = i as f64;
            let b = a + 1.0; // Same trend as a
            let c = (i as f64 * 0.5).sin() * 10.0;
            let d = c + 0.5; // Same trend as c
            tracker.update(&[a, b, c, d]);
        }

        let pairs = tracker.highly_correlated_pairs(0.9);
        assert!(!pairs.is_empty(), "Should find highly correlated pairs");

        // a-b (0,1) should be in the list
        let has_ab = pairs.iter().any(|(i, j, _)| *i == 0 && *j == 1);
        assert!(has_ab, "Should find a-b pair");
    }

    #[test]
    fn test_vif() {
        let features = ["x", "y", "z"];
        let mut tracker = FeatureCorrelationTracker::new(&features, 0.1);

        // x and y highly correlated, z independent
        for i in 0..100 {
            let x = i as f64;
            let y = x * 1.1 + 2.0;
            let z = (i as f64 * 0.2).cos() * 5.0;
            tracker.update(&[x, y, z]);
        }

        let vifs = tracker.variance_inflation_factors();
        assert_eq!(vifs.len(), 3);

        // x and y should have high VIF (multicollinear)
        assert!(vifs[0] > 5.0, "x should have high VIF: {}", vifs[0]);
        assert!(vifs[1] > 5.0, "y should have high VIF: {}", vifs[1]);

        // z should have low VIF (independent)
        assert!(vifs[2] < 5.0, "z should have low VIF: {}", vifs[2]);
    }
}
