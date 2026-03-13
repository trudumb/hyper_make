//! Cox process fill model for principled fill-to-drift updates.
//!
//! Replaces heuristic `z = signum * distance_sigma.min(3) * 0.5` with
//! Cox process score function: delta_mu = -beta * Sigma (buy fill) or +beta * Sigma (sell fill).
//!
//! In the Kalman framework, K = Sigma/(Sigma+R). With R = beta^2 * Sigma:
//! K = 1/(1+beta^2), so delta_mu = sign * beta * Sigma/(1+beta^2) -- proportional to Sigma.
//! High uncertainty -> large update. Low uncertainty -> small update.

use serde::{Deserialize, Serialize};

/// Prior beta sensitivity (1/bps).
const PRIOR_BETA: f64 = 0.2;
/// Minimum fills before beta MLE.
const MIN_FILLS_FOR_BETA: u64 = 50;
/// Beta MLE interval (fills).
const BETA_UPDATE_INTERVAL: u64 = 10;

/// Cox process model for fill-to-drift inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoxFillModel {
    /// Sensitivity parameter (1/bps): how much each fill shifts drift.
    beta: f64,
    /// Baseline fill rate (fills/sec), EWMA-smoothed.
    lambda_0: f64,
    /// MLE sufficient stats: sum of drift at buy fills.
    sum_mu_at_buy: f64,
    /// MLE sufficient stats: sum of drift at sell fills.
    sum_mu_at_sell: f64,
    /// Count of buy fills observed.
    n_buy: u64,
    /// Count of sell fills observed.
    n_sell: u64,
    /// EWMA inter-arrival time (seconds).
    ewma_iat_s: f64,
    /// Last fill timestamp for IAT computation.
    last_fill_time_ms: Option<u64>,
    /// Fills since last beta update.
    fills_since_beta_update: u64,
}

impl CoxFillModel {
    /// Create with prior beta and initial lambda.
    pub fn new(beta_prior: f64) -> Self {
        Self {
            beta: beta_prior.clamp(0.05, 2.0),
            lambda_0: 0.1, // Conservative initial estimate
            sum_mu_at_buy: 0.0,
            sum_mu_at_sell: 0.0,
            n_buy: 0,
            n_sell: 0,
            ewma_iat_s: 10.0, // Conservative initial
            last_fill_time_ms: None,
            fills_since_beta_update: 0,
        }
    }

    /// Cox process score function for a fill observation.
    ///
    /// Returns (z, r) suitable for Kalman update:
    /// - z: observation signal (negative for buy = bearish, positive for sell = bullish)
    /// - r: observation variance = beta^2 * current_variance
    ///
    /// With R = beta^2 * Sigma, the Kalman gain K = 1/(1+beta^2) is a fixed fraction,
    /// so the update delta_mu = sign * beta * Sigma/(1+beta^2) scales with uncertainty.
    pub fn fill_observation(&self, is_buy: bool, current_variance: f64) -> (f64, f64) {
        let sign = if is_buy { -1.0 } else { 1.0 };
        let z = sign * self.beta * current_variance;
        let r = self.beta * self.beta * current_variance;
        (z, r)
    }

    /// Record a fill for beta MLE estimation.
    ///
    /// Call after the Kalman update so drift_at_fill reflects post-update state.
    pub fn record_fill(&mut self, is_buy: bool, drift_at_fill: f64, fill_time_ms: u64) {
        if is_buy {
            self.sum_mu_at_buy += drift_at_fill;
            self.n_buy += 1;
        } else {
            self.sum_mu_at_sell += drift_at_fill;
            self.n_sell += 1;
        }

        // Update inter-arrival time EWMA
        if let Some(last_ms) = self.last_fill_time_ms {
            if fill_time_ms > last_ms {
                let iat_s = (fill_time_ms - last_ms) as f64 / 1000.0;
                let alpha = 0.05;
                self.ewma_iat_s = (1.0 - alpha) * self.ewma_iat_s + alpha * iat_s;
                self.lambda_0 = (1.0 / self.ewma_iat_s).clamp(0.01, 100.0);
            }
        }
        self.last_fill_time_ms = Some(fill_time_ms);

        self.fills_since_beta_update += 1;

        // Online beta MLE
        let total_fills = self.n_buy + self.n_sell;
        if total_fills >= MIN_FILLS_FOR_BETA && self.fills_since_beta_update >= BETA_UPDATE_INTERVAL
        {
            self.update_beta();
            self.fills_since_beta_update = 0;
        }
    }

    /// MLE update for beta from sufficient statistics.
    fn update_beta(&mut self) {
        if self.n_buy == 0 || self.n_sell == 0 {
            return;
        }
        let avg_mu_sell = self.sum_mu_at_sell / self.n_sell as f64;
        let avg_mu_buy = self.sum_mu_at_buy / self.n_buy as f64;
        // Beta should be positive: drift at sell fills > drift at buy fills
        // (sell fills are bullish -> drift should be more positive)
        let beta_hat = ((avg_mu_sell - avg_mu_buy) / 2.0).abs().clamp(0.05, 2.0);
        // Blend toward estimate
        self.beta = 0.9 * self.beta + 0.1 * beta_hat;
    }

    /// Current beta sensitivity.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Current baseline fill rate (fills/sec).
    pub fn lambda_0(&self) -> f64 {
        self.lambda_0
    }

    /// Total fills observed.
    pub fn total_fills(&self) -> u64 {
        self.n_buy + self.n_sell
    }

    /// Whether beta has been calibrated from data.
    pub fn is_calibrated(&self) -> bool {
        self.n_buy + self.n_sell >= MIN_FILLS_FOR_BETA
    }
}

impl Default for CoxFillModel {
    fn default() -> Self {
        Self::new(PRIOR_BETA)
    }
}

/// Checkpoint state for Cox fill model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoxFillCheckpoint {
    #[serde(default)]
    pub beta: f64,
    #[serde(default)]
    pub lambda_0: f64,
    #[serde(default)]
    pub sum_mu_at_buy: f64,
    #[serde(default)]
    pub sum_mu_at_sell: f64,
    #[serde(default)]
    pub n_buy: u64,
    #[serde(default)]
    pub n_sell: u64,
    #[serde(default)]
    pub ewma_iat_s: f64,
}

impl CoxFillModel {
    pub fn to_checkpoint(&self) -> CoxFillCheckpoint {
        CoxFillCheckpoint {
            beta: self.beta,
            lambda_0: self.lambda_0,
            sum_mu_at_buy: self.sum_mu_at_buy,
            sum_mu_at_sell: self.sum_mu_at_sell,
            n_buy: self.n_buy,
            n_sell: self.n_sell,
            ewma_iat_s: self.ewma_iat_s,
        }
    }

    pub fn from_checkpoint(checkpoint: &CoxFillCheckpoint) -> Self {
        if checkpoint.n_buy == 0 && checkpoint.n_sell == 0 {
            return Self::default();
        }
        Self {
            beta: checkpoint.beta.clamp(0.05, 2.0),
            lambda_0: checkpoint.lambda_0.clamp(0.01, 100.0),
            sum_mu_at_buy: checkpoint.sum_mu_at_buy,
            sum_mu_at_sell: checkpoint.sum_mu_at_sell,
            n_buy: checkpoint.n_buy,
            n_sell: checkpoint.n_sell,
            ewma_iat_s: checkpoint.ewma_iat_s.max(0.01),
            last_fill_time_ms: None,
            fills_since_beta_update: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buy_fill_produces_negative_z() {
        let model = CoxFillModel::default();
        let (z, _r) = model.fill_observation(true, 1.0);
        assert!(
            z < 0.0,
            "Buy fill should produce negative z (bearish), got {z}"
        );
    }

    #[test]
    fn test_sell_fill_produces_positive_z() {
        let model = CoxFillModel::default();
        let (z, _r) = model.fill_observation(false, 1.0);
        assert!(
            z > 0.0,
            "Sell fill should produce positive z (bullish), got {z}"
        );
    }

    #[test]
    fn test_high_variance_larger_update() {
        let model = CoxFillModel::default();
        let (z_low, _) = model.fill_observation(true, 0.1);
        let (z_high, _) = model.fill_observation(true, 10.0);
        assert!(
            z_high.abs() > z_low.abs(),
            "Higher variance should produce larger |z|: {} vs {}",
            z_high.abs(),
            z_low.abs()
        );
    }

    #[test]
    fn test_kalman_gain_is_fixed_fraction() {
        // With R = beta^2 * Sigma, K = Sigma/(Sigma+R) = 1/(1+beta^2) regardless of Sigma
        let model = CoxFillModel::new(0.5);
        let expected_k = 1.0 / (1.0 + 0.5 * 0.5); // 0.8
        for variance in [0.1, 1.0, 10.0, 100.0] {
            let (_z, r) = model.fill_observation(true, variance);
            let k = variance / (variance + r);
            assert!(
                (k - expected_k).abs() < 1e-10,
                "Kalman gain should be {} for all variances, got {} at P={}",
                expected_k,
                k,
                variance
            );
        }
    }

    #[test]
    fn test_beta_convergence() {
        let mut model = CoxFillModel::new(0.2);
        // Simulate: buy fills when drift is negative, sell fills when positive
        for i in 0..200u64 {
            let drift = if i % 2 == 0 { -0.5 } else { 0.5 };
            let is_buy = i % 2 == 0;
            model.record_fill(is_buy, drift, i * 1000);
        }
        assert!(model.is_calibrated());
        assert!(
            model.beta > 0.05 && model.beta < 2.0,
            "Beta should be in bounds, got {}",
            model.beta
        );
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut model = CoxFillModel::new(0.3);
        for i in 0..60 {
            model.record_fill(i % 2 == 0, 0.1 * (i as f64), i * 500);
        }
        let cp = model.to_checkpoint();
        let restored = CoxFillModel::from_checkpoint(&cp);
        assert_eq!(model.n_buy, restored.n_buy);
        assert_eq!(model.n_sell, restored.n_sell);
        assert!((model.beta - restored.beta).abs() < 1e-10);
    }

    #[test]
    fn test_default_model() {
        let model = CoxFillModel::default();
        assert_eq!(model.total_fills(), 0);
        assert!(!model.is_calibrated());
        assert!((model.beta - PRIOR_BETA).abs() < 1e-10);
    }
}
