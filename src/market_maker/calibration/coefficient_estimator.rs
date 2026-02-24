//! Coefficient estimator for calibrating risk model from fill outcomes.
//!
//! Fits risk model coefficients via OLS regression:
//! ```text
//! realized_as_bps ~ β₀ + β_vol × excess_vol + β_tox × toxicity + ...
//! ```
//!
//! Then converts AS coefficients to log-gamma via scaling.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::market_maker::strategy::{CalibratedRiskModel, RiskFeatures};

/// A single calibration sample recording features and outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSample {
    /// Timestamp when fill occurred (milliseconds)
    pub timestamp_ms: u64,

    /// Risk features at time of fill
    pub features: RiskFeatures,

    /// Realized adverse selection in basis points
    pub realized_as_bps: f64,

    /// Realized edge in basis points (spread - AS - fees)
    pub realized_edge_bps: f64,

    /// Fill side (true = buy, false = sell)
    pub is_buy: bool,

    /// Depth from mid where fill occurred (bps)
    pub depth_bps: f64,
}

/// Fits risk model coefficients from resolved fill outcomes.
///
/// Uses online OLS regression with exponential forgetting to adapt
/// to changing market conditions.
#[derive(Debug, Clone)]
pub struct CoefficientEstimator {
    /// Minimum samples before fitting
    min_samples: usize,

    /// Maximum sample age in seconds
    max_sample_age_secs: f64,

    /// Ring buffer of recent samples
    samples: VecDeque<CalibrationSample>,

    /// Maximum samples to retain
    max_samples: usize,

    /// Currently fitted model (if any)
    fitted_model: Option<CalibratedRiskModel>,

    /// Exponential forgetting factor (0.99 = slow adaptation)
    forgetting_factor: f64,

    // === Sufficient statistics for incremental OLS ===
    // Using Sherman-Morrison for incremental updates
    /// Sum of X'X (6x6 matrix stored as flat array)
    xtx: Vec<f64>,

    /// Sum of X'y (6x1 vector)
    xty: Vec<f64>,

    /// Sum of y² (for R² calculation)
    yy: f64,

    /// Number of samples in statistics
    n_stats: usize,
}

impl Default for CoefficientEstimator {
    fn default() -> Self {
        Self::new(100, 86400.0) // 100 samples, 24 hour max age
    }
}

impl CoefficientEstimator {
    /// Number of features in the model.
    const N_FEATURES: usize = 6;

    /// Create a new coefficient estimator.
    pub fn new(min_samples: usize, max_sample_age_secs: f64) -> Self {
        Self {
            min_samples,
            max_sample_age_secs,
            samples: VecDeque::with_capacity(1000),
            max_samples: 1000,
            fitted_model: None,
            forgetting_factor: 0.995,
            xtx: vec![0.0; Self::N_FEATURES * Self::N_FEATURES],
            xty: vec![0.0; Self::N_FEATURES],
            yy: 0.0,
            n_stats: 0,
        }
    }

    /// Record a new calibration sample.
    pub fn record_sample(&mut self, sample: CalibrationSample) {
        // Add to ring buffer
        self.samples.push_back(sample.clone());
        while self.samples.len() > self.max_samples {
            self.samples.pop_front();
        }

        // Update sufficient statistics
        self.update_statistics(&sample);

        // Evict old samples
        self.evict_old_samples();

        // Attempt to fit if we have enough samples
        if self.samples.len() >= self.min_samples {
            self.fit();
        }
    }

    /// Update sufficient statistics incrementally.
    fn update_statistics(&mut self, sample: &CalibrationSample) {
        let x = self.features_to_vector(&sample.features);
        let y = sample.realized_as_bps;

        // Apply forgetting factor to existing statistics
        let ff = self.forgetting_factor;
        for v in &mut self.xtx {
            *v *= ff;
        }
        for v in &mut self.xty {
            *v *= ff;
        }
        self.yy *= ff;

        // Add new sample contribution
        for i in 0..Self::N_FEATURES {
            for j in 0..Self::N_FEATURES {
                self.xtx[i * Self::N_FEATURES + j] += x[i] * x[j];
            }
            self.xty[i] += x[i] * y;
        }
        self.yy += y * y;
        self.n_stats += 1;
    }

    /// Convert RiskFeatures to a vector for regression.
    fn features_to_vector(&self, features: &RiskFeatures) -> [f64; Self::N_FEATURES] {
        [
            features.excess_volatility,
            features.toxicity_score,
            features.inventory_fraction,
            features.excess_intensity,
            features.depth_depletion,
            features.model_uncertainty,
        ]
    }

    /// Evict samples older than max_sample_age_secs.
    fn evict_old_samples(&mut self) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let max_age_ms = (self.max_sample_age_secs * 1000.0) as u64;

        while let Some(front) = self.samples.front() {
            if now_ms.saturating_sub(front.timestamp_ms) > max_age_ms {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }

    /// Fit the model using OLS on accumulated samples.
    fn fit(&mut self) {
        if self.samples.len() < self.min_samples {
            return;
        }

        // Solve β = (X'X)^(-1) X'y using Cholesky decomposition
        // For numerical stability, add small regularization
        let mut xtx_reg = self.xtx.clone();
        let ridge = 0.01; // Small L2 regularization
        for i in 0..Self::N_FEATURES {
            xtx_reg[i * Self::N_FEATURES + i] += ridge;
        }

        // Cholesky decomposition and solve
        if let Some(beta) = self.solve_cholesky(&xtx_reg, &self.xty) {
            // Calculate R²
            let y_mean = self.calculate_y_mean();
            let ss_tot = self.yy - (self.samples.len() as f64) * y_mean * y_mean;
            let ss_res = self.calculate_ss_residual(&beta);
            let r_squared = if ss_tot > 1e-9 {
                1.0 - ss_res / ss_tot
            } else {
                0.0
            };

            // Convert AS coefficients to log-gamma coefficients
            // Key insight: higher AS → wider spreads → higher gamma
            // Scale factor converts bps AS to log-gamma units
            let as_to_log_gamma_scale = 0.1; // 10 bps AS → 1.0 log-gamma

            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);

            // Create fitted model
            let fitted = CalibratedRiskModel {
                log_gamma_base: 0.15_f64.ln(), // Keep base from defaults
                beta_volatility: (beta[0] * as_to_log_gamma_scale).clamp(-2.0, 3.0),
                beta_toxicity: (beta[1] * as_to_log_gamma_scale).clamp(-1.0, 2.0),
                beta_inventory: (beta[2] * as_to_log_gamma_scale).clamp(-0.5, 1.5),
                beta_hawkes: (beta[3] * as_to_log_gamma_scale).clamp(-1.0, 2.0),
                beta_book_depth: (beta[4] * as_to_log_gamma_scale).clamp(-0.5, 1.5),
                beta_uncertainty: (beta[5] * as_to_log_gamma_scale).clamp(-0.5, 1.0),
                // Keep default for confidence - not learned from AS data
                // (position direction confidence is orthogonal to AS risk)
                beta_confidence: -0.4,
                // Cascade risk not learned from AS data - use interim default
                beta_cascade: 1.2,
                // Tail risk not learned from AS data - use interim default
                beta_tail_risk: 0.7,
                // WS1: New betas not learned from AS data - use defaults
                beta_drawdown: 1.4,
                beta_regime: 1.0,
                beta_ghost: 0.5,
                beta_continuation: -0.5,
                gamma_min: 0.05,
                gamma_max: 5.0,
                n_samples: self.samples.len(),
                last_calibration_ms: now_ms,
                r_squared: r_squared.max(0.0),
                state: crate::market_maker::strategy::CalibrationState::Calibrated {
                    r_squared: r_squared.max(0.0),
                },
            };

            tracing::info!(
                n_samples = self.samples.len(),
                r_squared = %format!("{:.3}", r_squared),
                beta_vol = %format!("{:.3}", fitted.beta_volatility),
                beta_tox = %format!("{:.3}", fitted.beta_toxicity),
                beta_inv = %format!("{:.3}", fitted.beta_inventory),
                beta_hawkes = %format!("{:.3}", fitted.beta_hawkes),
                beta_depth = %format!("{:.3}", fitted.beta_book_depth),
                beta_unc = %format!("{:.3}", fitted.beta_uncertainty),
                "Coefficient estimator fitted new model"
            );

            self.fitted_model = Some(fitted);
        }
    }

    /// Solve linear system using Cholesky decomposition.
    fn solve_cholesky(&self, a: &[f64], b: &[f64]) -> Option<Vec<f64>> {
        let n = Self::N_FEATURES;

        // Cholesky decomposition: A = L * L'
        let mut l = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = a[i * n + j];
                for k in 0..j {
                    sum -= l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    if sum <= 0.0 {
                        return None; // Not positive definite
                    }
                    l[i * n + j] = sum.sqrt();
                } else {
                    let ljj = l[j * n + j];
                    if ljj.abs() < 1e-12 {
                        return None;
                    }
                    l[i * n + j] = sum / ljj;
                }
            }
        }

        // Forward substitution: L * y = b
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= l[i * n + j] * y[j];
            }
            let lii = l[i * n + i];
            if lii.abs() < 1e-12 {
                return None;
            }
            y[i] = sum / lii;
        }

        // Backward substitution: L' * x = y
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= l[j * n + i] * x[j];
            }
            let lii = l[i * n + i];
            if lii.abs() < 1e-12 {
                return None;
            }
            x[i] = sum / lii;
        }

        Some(x)
    }

    /// Calculate mean of y values (realized AS).
    fn calculate_y_mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().map(|s| s.realized_as_bps).sum::<f64>() / self.samples.len() as f64
    }

    /// Calculate sum of squared residuals.
    fn calculate_ss_residual(&self, beta: &[f64]) -> f64 {
        self.samples
            .iter()
            .map(|s| {
                let x = self.features_to_vector(&s.features);
                let y_hat: f64 = x.iter().zip(beta.iter()).map(|(xi, bi)| xi * bi).sum();
                let residual = s.realized_as_bps - y_hat;
                residual * residual
            })
            .sum()
    }

    /// Get the fitted model (if available).
    pub fn fitted_model(&self) -> Option<&CalibratedRiskModel> {
        self.fitted_model.as_ref()
    }

    /// Get number of samples.
    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    /// Check if estimator is warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.samples.len() >= self.min_samples
    }

    /// Get R² of the current fit.
    pub fn r_squared(&self) -> f64 {
        self.fitted_model
            .as_ref()
            .map(|m| m.r_squared)
            .unwrap_or(0.0)
    }

    /// Reset the estimator.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.fitted_model = None;
        self.xtx = vec![0.0; Self::N_FEATURES * Self::N_FEATURES];
        self.xty = vec![0.0; Self::N_FEATURES];
        self.yy = 0.0;
        self.n_stats = 0;
    }

    /// Get feature importance (absolute beta values normalized).
    pub fn feature_importance(&self) -> Option<[f64; Self::N_FEATURES]> {
        self.fitted_model.as_ref().map(|m| {
            let betas = [
                m.beta_volatility.abs(),
                m.beta_toxicity.abs(),
                m.beta_inventory.abs(),
                m.beta_hawkes.abs(),
                m.beta_book_depth.abs(),
                m.beta_uncertainty.abs(),
            ];
            let total: f64 = betas.iter().sum();
            if total > 1e-9 {
                [
                    betas[0] / total,
                    betas[1] / total,
                    betas[2] / total,
                    betas[3] / total,
                    betas[4] / total,
                    betas[5] / total,
                ]
            } else {
                [0.0; Self::N_FEATURES]
            }
        })
    }
}

/// Configuration for coefficient estimator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoefficientEstimatorConfig {
    /// Minimum samples before fitting
    pub min_samples: usize,

    /// Maximum sample age in seconds
    pub max_sample_age_secs: f64,

    /// Maximum samples to retain
    pub max_samples: usize,

    /// Exponential forgetting factor
    pub forgetting_factor: f64,
}

impl Default for CoefficientEstimatorConfig {
    fn default() -> Self {
        Self {
            min_samples: 100,
            max_sample_age_secs: 86400.0, // 24 hours
            max_samples: 1000,
            forgetting_factor: 0.995,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimator_warmup() {
        let estimator = CoefficientEstimator::new(10, 3600.0);
        assert!(!estimator.is_warmed_up());
        assert_eq!(estimator.n_samples(), 0);
    }

    #[test]
    fn test_sample_recording() {
        let mut estimator = CoefficientEstimator::new(5, 3600.0);

        // Use current timestamps so samples don't get evicted as "old"
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        for i in 0..10 {
            let sample = CalibrationSample {
                timestamp_ms: now_ms + i * 1000, // Offset from now
                features: RiskFeatures {
                    excess_volatility: 0.5,
                    toxicity_score: 0.3,
                    ..Default::default()
                },
                realized_as_bps: 5.0 + (i as f64) * 0.5,
                realized_edge_bps: 2.0,
                is_buy: true,
                depth_bps: 10.0,
            };
            estimator.record_sample(sample);
        }

        assert_eq!(estimator.n_samples(), 10);
        assert!(estimator.is_warmed_up());
    }

    #[test]
    fn test_cholesky_solve() {
        let estimator = CoefficientEstimator::default();

        // Simple 2x2 test case (padded to 6x6 identity)
        let mut a = vec![0.0; 36];
        for i in 0..6 {
            a[i * 6 + i] = 1.0;
        }
        // Set (0,0) = 4, (0,1) = 2, (1,0) = 2, (1,1) = 3
        a[0] = 4.0;
        a[1] = 2.0;
        a[6] = 2.0;
        a[7] = 3.0;

        let b = vec![8.0, 7.0, 1.0, 1.0, 1.0, 1.0];

        let result = estimator.solve_cholesky(&a, &b);
        assert!(result.is_some());

        let x = result.unwrap();
        // For Ax = b with A = [[4,2],[2,3]] and b = [8,7]
        // Solution should be x = [1, 2] for the first two components
        // (The rest are identity, so x[i] = b[i] = 1)
        assert!((x[0] - 1.25).abs() < 0.01, "x[0] = {}", x[0]);
        assert!((x[1] - 1.5).abs() < 0.01, "x[1] = {}", x[1]);
    }

    #[test]
    fn test_fitting() {
        let mut estimator = CoefficientEstimator::new(5, 3600.0);

        // Generate synthetic data where AS correlates with features
        for i in 0..20 {
            let vol = (i as f64) / 20.0;
            let sample = CalibrationSample {
                timestamp_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
                features: RiskFeatures {
                    excess_volatility: vol,
                    toxicity_score: vol * 0.5,
                    inventory_fraction: 0.1,
                    ..Default::default()
                },
                // AS increases with volatility
                realized_as_bps: 2.0 + vol * 10.0,
                realized_edge_bps: 3.0 - vol * 5.0,
                is_buy: i % 2 == 0,
                depth_bps: 8.0,
            };
            estimator.record_sample(sample);
        }

        assert!(estimator.fitted_model().is_some());
        let model = estimator.fitted_model().unwrap();

        // Beta volatility should be positive (higher vol → higher AS → higher gamma)
        assert!(
            model.beta_volatility > 0.0,
            "beta_vol should be positive: {}",
            model.beta_volatility
        );
    }
}
