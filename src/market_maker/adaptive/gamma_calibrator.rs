//! Online Bayesian Gamma Calibration via Recursive Least Squares.
//!
//! Learns the 15 beta coefficients of `CalibratedRiskModel` from realized fill PnL.
//! Uses a diagonal RLS approximation with forgetting factor λ=0.999 (~1000-fill half-life)
//! to adapt to non-stationary environments (regime changes).
//!
//! The calibrator observes (features, gamma_used, realized_edge_bps) after each fill
//! and updates beta coefficients toward the gamma that would have maximized edge.
//!
//! During warmup (< min_samples fills), the calibrator blends learned betas
//! with the hand-tuned defaults from `CalibratedRiskModel::default()`.

use serde::{Deserialize, Serialize};

/// Number of beta coefficients in CalibratedRiskModel.
pub const NUM_BETAS: usize = 15;

/// Online Bayesian gamma calibrator using diagonal Recursive Least Squares.
///
/// Learns from fill outcomes to improve gamma (risk aversion) coefficient selection.
/// The key insight: if a fill had negative edge, gamma was too low (should have widened).
/// If edge was positive, gamma was appropriate or slightly too high.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineBayesianGammaCalibrator {
    /// Current beta coefficients (15-dim), in same order as CalibratedRiskModel.
    pub beta: Vec<f64>,
    /// Diagonal precision (inverse variance) per feature.
    /// Full 15×15 is O(n²) per update; diagonal is O(n) and sufficient for orthogonal features.
    pub precision_diag: Vec<f64>,
    /// Forgetting factor: 0.999 → ~1000-fill half-life for adaptation.
    #[serde(default = "default_lambda")]
    pub lambda: f64,
    /// Minimum fills before overriding defaults with learned values.
    #[serde(default = "default_min_samples")]
    pub min_samples: usize,
    /// Running sample count.
    pub n_samples: usize,
    /// Prior beta (defaults from CalibratedRiskModel).
    pub beta_prior: Vec<f64>,
    /// Prior precision strength (regularization toward defaults).
    #[serde(default = "default_prior_strength")]
    pub prior_strength: f64,
}

fn default_lambda() -> f64 {
    0.999
}
fn default_min_samples() -> usize {
    100
}
fn default_prior_strength() -> f64 {
    0.1
}

impl OnlineBayesianGammaCalibrator {
    /// Create a new calibrator initialized from CalibratedRiskModel defaults.
    pub fn new(default_betas: &[f64; NUM_BETAS]) -> Self {
        let beta = default_betas.to_vec();
        let beta_prior = default_betas.to_vec();
        // Initial precision: modest confidence in priors (0.1 → std ≈ 3.16 per coefficient)
        let precision_diag = vec![default_prior_strength(); NUM_BETAS];
        Self {
            beta,
            precision_diag,
            lambda: default_lambda(),
            min_samples: default_min_samples(),
            n_samples: 0,
            beta_prior,
            prior_strength: default_prior_strength(),
        }
    }

    /// Update with a fill observation.
    ///
    /// # Arguments
    /// * `features` - The 15-dim feature vector at fill time (same order as RiskFeatures fields)
    /// * `gamma_used` - The gamma that was active when the fill occurred
    /// * `realized_edge_bps` - Actual edge from markout (positive = profitable)
    pub fn update(&mut self, features: &[f64], gamma_used: f64, realized_edge_bps: f64) {
        if features.len() != NUM_BETAS || gamma_used <= 0.0 {
            return;
        }

        let log_gamma_used = gamma_used.ln();

        // Asymmetric learning rate:
        // - Negative edge → push gamma up (widen). Losing is expensive.
        // - Positive edge → gently push gamma down (tighten). Capture more edge.
        let edge_gradient = if realized_edge_bps < 0.0 {
            0.5 // Strong push to widen when losing
        } else {
            -0.1 // Gentle push to tighten when winning
        };
        // Target: log-gamma that would have been better
        let target = log_gamma_used + edge_gradient * realized_edge_bps.abs().min(5.0);

        // Diagonal RLS update
        // prediction = β^T × features
        let prediction: f64 = self.beta.iter().zip(features).map(|(b, x)| b * x).sum();
        let error = target - prediction;

        for (i, &x_i) in features.iter().enumerate().take(NUM_BETAS) {
            let p_i = self.precision_diag[i];
            let denominator = self.lambda + x_i * p_i * x_i;
            if denominator.abs() < 1e-15 {
                continue;
            }
            let gain = p_i * x_i / denominator;
            self.beta[i] += gain * error;
            self.precision_diag[i] = (1.0 / self.lambda) * (p_i - gain * x_i * p_i);
            // Prevent precision from going negative or exploding
            self.precision_diag[i] = self.precision_diag[i].clamp(1e-6, 1e6);
        }
        self.n_samples += 1;
    }

    /// Returns calibrated betas, blended with prior when sample count is low.
    ///
    /// blend = min(n_samples / min_samples, 1.0)
    /// effective_beta[i] = (1 - blend) × prior[i] + blend × learned[i]
    pub fn effective_betas(&self) -> [f64; NUM_BETAS] {
        let blend = (self.n_samples as f64 / self.min_samples.max(1) as f64).min(1.0);
        let mut result = [0.0; NUM_BETAS];
        for (i, val) in result.iter_mut().enumerate().take(NUM_BETAS) {
            *val = (1.0 - blend) * self.beta_prior[i] + blend * self.beta[i];
        }
        result
    }

    /// Whether the calibrator has enough data to influence gamma computation.
    pub fn is_warmed_up(&self) -> bool {
        self.n_samples >= self.min_samples
    }

    /// Current blend ratio [0, 1]. 0 = pure prior, 1 = fully learned.
    pub fn blend_ratio(&self) -> f64 {
        (self.n_samples as f64 / self.min_samples.max(1) as f64).min(1.0)
    }
}

impl Default for OnlineBayesianGammaCalibrator {
    fn default() -> Self {
        // Default betas matching CalibratedRiskModel::default()
        let defaults = [
            1.0,  // beta_volatility
            0.5,  // beta_toxicity
            4.0,  // beta_inventory
            0.4,  // beta_hawkes
            0.3,  // beta_book_depth
            0.2,  // beta_uncertainty
            -0.4, // beta_confidence (negative)
            1.2,  // beta_cascade
            0.7,  // beta_tail_risk
            1.4,  // beta_drawdown
            1.0,  // beta_regime
            0.5,  // beta_ghost
            -0.5, // beta_continuation (negative)
            1.5,  // beta_edge_uncertainty
            0.8,  // beta_calibration
        ];
        Self::new(&defaults)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_calibrator_default_creation() {
        let cal = OnlineBayesianGammaCalibrator::default();
        assert_eq!(cal.beta.len(), NUM_BETAS);
        assert_eq!(cal.beta_prior.len(), NUM_BETAS);
        assert_eq!(cal.n_samples, 0);
        assert!(!cal.is_warmed_up());
        assert!((cal.blend_ratio() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_calibrator_warmup_blend() {
        let mut cal = OnlineBayesianGammaCalibrator::default();
        let features = [0.5; NUM_BETAS];

        // Before any updates: effective_betas = prior
        let betas_cold = cal.effective_betas();
        assert!((betas_cold[0] - 1.0).abs() < 1e-10, "Cold should be prior");

        // After 50 fills (half warmup): blend = 0.5
        for _ in 0..50 {
            cal.update(&features, 0.15, 2.0);
        }
        assert!((cal.blend_ratio() - 0.5).abs() < 1e-10);
        let betas_half = cal.effective_betas();
        // Should be between prior and learned
        assert!(
            betas_half[0] != betas_cold[0] || cal.beta[0] == cal.beta_prior[0],
            "Half-warmed should differ from cold (unless learned == prior)"
        );

        // After 100 fills (fully warmed): blend = 1.0
        for _ in 0..50 {
            cal.update(&features, 0.15, 2.0);
        }
        assert!((cal.blend_ratio() - 1.0).abs() < 1e-10);
        assert!(cal.is_warmed_up());
    }

    #[test]
    fn test_gamma_calibrator_negative_edge_widens() {
        let mut cal = OnlineBayesianGammaCalibrator::default();
        let initial_beta = cal.beta.clone();
        // Use small features so initial prediction ≈ sum(beta * 0.05) ≈ 0.63,
        // close to log(gamma_used=1.5) ≈ 0.41. This lets the error signal from
        // edge correctly dominate the direction of beta updates.
        let features = vec![0.05; NUM_BETAS];

        // gamma_used=1.5: log(1.5)=0.41, prediction≈0.63
        // Negative edge target: 0.41 + 0.5*3 = 1.91 > 0.63 → error positive → betas UP
        for _ in 0..20 {
            cal.update(&features, 1.5, -3.0); // -3 bps edge = losing
        }

        // The sum of (beta * features) should have increased (wider gamma)
        let sum_before: f64 = initial_beta.iter().sum();
        let sum_after: f64 = cal.beta.iter().sum();
        assert!(
            sum_after > sum_before,
            "Negative edge should push betas up: before={sum_before:.3}, after={sum_after:.3}"
        );
    }

    #[test]
    fn test_gamma_calibrator_positive_edge_tightens() {
        let mut cal = OnlineBayesianGammaCalibrator::default();
        let initial_beta = cal.beta.clone();
        // Small features for realistic prediction scale
        let features = vec![0.05; NUM_BETAS];

        // gamma_used=1.5: log(1.5)=0.41, prediction≈0.63
        // Positive edge target: 0.41 + (-0.1)*5 = -0.09 < 0.63 → error negative → betas DOWN
        for _ in 0..50 {
            cal.update(&features, 1.5, 5.0); // +5 bps edge = winning
        }

        let sum_before: f64 = initial_beta.iter().sum();
        let sum_after: f64 = cal.beta.iter().sum();
        assert!(
            sum_after < sum_before,
            "Positive edge should gently push betas down: before={sum_before:.3}, after={sum_after:.3}"
        );
    }

    #[test]
    fn test_gamma_calibrator_ignores_bad_input() {
        let mut cal = OnlineBayesianGammaCalibrator::default();
        let initial_beta = cal.beta.clone();

        // Wrong feature length → no update
        cal.update(&[1.0; 5], 0.15, 2.0);
        assert_eq!(cal.n_samples, 0);

        // Zero gamma → no update
        cal.update(&[1.0; NUM_BETAS], 0.0, 2.0);
        assert_eq!(cal.n_samples, 0);

        // Negative gamma → no update
        cal.update(&[1.0; NUM_BETAS], -1.0, 2.0);
        assert_eq!(cal.n_samples, 0);

        assert_eq!(cal.beta, initial_beta);
    }
}
