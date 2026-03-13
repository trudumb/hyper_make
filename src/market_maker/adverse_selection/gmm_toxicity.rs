//! Gaussian Mixture Model for fill toxicity classification.
//!
//! Replaces heuristic adverse selection weights with a principled posterior:
//! P(informed | ΔP) = π·φ_inf(ΔP) / [π·φ_inf(ΔP) + (1-π)·φ_noise(ΔP)]
//!
//! Online EM updates component parameters from fill markout observations.

use serde::{Deserialize, Serialize};

/// Minimum fills before GMM updates parameters via EM.
const MIN_FILLS_WARMUP: u64 = 30;
/// EM re-estimation interval (fills).
const EM_UPDATE_INTERVAL: u64 = 20;
/// EWMA decay for per-side P(informed) tracking.
const EWMA_ALPHA: f64 = 0.02;

/// GMM toxicity classifier configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GmmToxicityConfig {
    /// Initial noise markout standard deviation (bps).
    pub sigma_noise_bps: f64,
    /// Initial informed markout magnitude (bps).
    pub mu_inf_bps: f64,
    /// Initial informed markout standard deviation (bps).
    pub sigma_inf_bps: f64,
    /// Prior probability of informed trade.
    pub pi_prior: f64,
    /// Minimum fills before EM updates.
    pub min_fills_warmup: u64,
    /// EM update interval (fills).
    pub em_update_interval: u64,
}

impl Default for GmmToxicityConfig {
    fn default() -> Self {
        Self {
            sigma_noise_bps: 3.0,
            mu_inf_bps: 8.0,
            sigma_inf_bps: 5.0,
            pi_prior: 0.15,
            min_fills_warmup: MIN_FILLS_WARMUP,
            em_update_interval: EM_UPDATE_INTERVAL,
        }
    }
}

/// Gaussian Mixture Model for classifying fill toxicity.
///
/// Two components:
/// - Noise: N(0, σ_noise²) — uninformed flow, symmetric markouts
/// - Informed: N(±μ_inf, σ_inf²) — directional adverse selection
///
/// Online EM updates parameters from markout observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GmmToxicityModel {
    // Component parameters
    sigma_noise_bps: f64,
    mu_inf_bps: f64,
    sigma_inf_bps: f64,
    pi: f64,

    // Online EM sufficient statistics (EWMA)
    sum_gamma: f64,
    sum_gamma_dp: f64,
    sum_gamma_dp2: f64,
    sum_1mg_dp2: f64,
    n_observations: u64,
    fills_since_em: u64,

    // Per-side EWMA tracking
    p_informed_buy_ewma: f64,
    p_informed_sell_ewma: f64,

    // Config
    config: GmmToxicityConfig,
}

impl GmmToxicityModel {
    pub fn new(config: GmmToxicityConfig) -> Self {
        let pi = config.pi_prior;
        let sigma_noise = config.sigma_noise_bps;
        let mu_inf = config.mu_inf_bps;
        let sigma_inf = config.sigma_inf_bps;
        Self {
            sigma_noise_bps: sigma_noise,
            mu_inf_bps: mu_inf,
            sigma_inf_bps: sigma_inf,
            pi,
            sum_gamma: 0.0,
            sum_gamma_dp: 0.0,
            sum_gamma_dp2: 0.0,
            sum_1mg_dp2: 0.0,
            n_observations: 0,
            fills_since_em: 0,
            p_informed_buy_ewma: pi,
            p_informed_sell_ewma: pi,
            config,
        }
    }

    /// Compute posterior P(informed | markout_bps, is_buy).
    pub fn posterior_informed(&self, markout_bps: f64, is_buy: bool) -> f64 {
        let mu_inf_signed = if is_buy {
            self.mu_inf_bps
        } else {
            -self.mu_inf_bps
        };
        let ll_noise = gaussian_pdf(markout_bps, 0.0, self.sigma_noise_bps);
        let ll_informed = gaussian_pdf(markout_bps, mu_inf_signed, self.sigma_inf_bps);

        let numerator = self.pi * ll_informed;
        let denominator = numerator + (1.0 - self.pi) * ll_noise;

        if denominator < 1e-300 {
            return self.pi; // Fallback to prior
        }
        (numerator / denominator).clamp(0.0, 1.0)
    }

    /// Update model with a new markout observation.
    /// Call this when a fill's markout measurement becomes available.
    pub fn update(&mut self, markout_bps: f64, is_buy: bool) {
        let gamma_i = self.posterior_informed(markout_bps, is_buy);

        // EWMA sufficient statistics
        let alpha = EWMA_ALPHA;
        self.sum_gamma = (1.0 - alpha) * self.sum_gamma + alpha * gamma_i;
        self.sum_gamma_dp = (1.0 - alpha) * self.sum_gamma_dp + alpha * gamma_i * markout_bps;
        self.sum_gamma_dp2 =
            (1.0 - alpha) * self.sum_gamma_dp2 + alpha * gamma_i * markout_bps * markout_bps;
        self.sum_1mg_dp2 =
            (1.0 - alpha) * self.sum_1mg_dp2 + alpha * (1.0 - gamma_i) * markout_bps * markout_bps;

        // Per-side EWMA
        if is_buy {
            self.p_informed_buy_ewma = (1.0 - alpha) * self.p_informed_buy_ewma + alpha * gamma_i;
        } else {
            self.p_informed_sell_ewma = (1.0 - alpha) * self.p_informed_sell_ewma + alpha * gamma_i;
        }

        self.n_observations += 1;
        self.fills_since_em += 1;

        // Online EM re-estimation
        if self.n_observations >= self.config.min_fills_warmup
            && self.fills_since_em >= self.config.em_update_interval
        {
            self.run_em_step();
            self.fills_since_em = 0;
        }
    }

    /// Online EM M-step: update component parameters from sufficient statistics.
    fn run_em_step(&mut self) {
        let sum_g = self.sum_gamma;
        if sum_g < 1e-10 {
            return;
        }

        // Update pi (mixing weight)
        let new_pi = sum_g.clamp(0.02, 0.80);
        self.pi = new_pi;

        // Update mu_inf (informed component mean magnitude)
        let mu_hat = (self.sum_gamma_dp / sum_g).abs();
        let new_mu = mu_hat.clamp(1.0, 50.0);
        self.mu_inf_bps = 0.9 * self.mu_inf_bps + 0.1 * new_mu;

        // Update sigma_inf (informed component std)
        let var_inf = (self.sum_gamma_dp2 / sum_g) - (self.sum_gamma_dp / sum_g).powi(2);
        if var_inf > 0.0 {
            let sigma_hat = var_inf.sqrt().clamp(0.5, 30.0);
            self.sigma_inf_bps = 0.9 * self.sigma_inf_bps + 0.1 * sigma_hat;
        }

        // Update sigma_noise (noise component std)
        let sum_1mg = 1.0 - sum_g;
        if sum_1mg > 1e-10 {
            let var_noise = self.sum_1mg_dp2 / sum_1mg;
            if var_noise > 0.0 {
                let sigma_hat = var_noise.sqrt().clamp(0.5, 20.0);
                self.sigma_noise_bps = 0.9 * self.sigma_noise_bps + 0.1 * sigma_hat;
            }
        }
    }

    /// Aggregate toxicity score [0, 1] suitable for gamma integration.
    /// Combines per-side posteriors into single score.
    pub fn toxicity_score(&self) -> f64 {
        (self.p_informed_buy_ewma + self.p_informed_sell_ewma) / 2.0
    }

    /// Per-side P(informed) for asymmetric spread widening.
    pub fn p_informed_buy(&self) -> f64 {
        self.p_informed_buy_ewma
    }

    /// Per-side P(informed) for asymmetric spread widening.
    pub fn p_informed_sell(&self) -> f64 {
        self.p_informed_sell_ewma
    }

    /// Whether model has enough observations for reliable posteriors.
    pub fn is_warmed_up(&self) -> bool {
        self.n_observations >= self.config.min_fills_warmup
    }

    /// Number of observations processed.
    pub fn observation_count(&self) -> u64 {
        self.n_observations
    }

    /// Current component parameters for diagnostics.
    pub fn diagnostics(&self) -> GmmToxicityDiagnostics {
        GmmToxicityDiagnostics {
            pi: self.pi,
            mu_inf_bps: self.mu_inf_bps,
            sigma_inf_bps: self.sigma_inf_bps,
            sigma_noise_bps: self.sigma_noise_bps,
            p_informed_buy: self.p_informed_buy_ewma,
            p_informed_sell: self.p_informed_sell_ewma,
            n_observations: self.n_observations,
            is_warmed_up: self.is_warmed_up(),
        }
    }
}

impl Default for GmmToxicityModel {
    fn default() -> Self {
        Self::new(GmmToxicityConfig::default())
    }
}

/// Diagnostic output for GMM toxicity model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GmmToxicityDiagnostics {
    pub pi: f64,
    pub mu_inf_bps: f64,
    pub sigma_inf_bps: f64,
    pub sigma_noise_bps: f64,
    pub p_informed_buy: f64,
    pub p_informed_sell: f64,
    pub n_observations: u64,
    pub is_warmed_up: bool,
}

/// Checkpoint state for GMM toxicity model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GmmToxicityCheckpoint {
    #[serde(default)]
    pub sigma_noise_bps: f64,
    #[serde(default)]
    pub mu_inf_bps: f64,
    #[serde(default)]
    pub sigma_inf_bps: f64,
    #[serde(default)]
    pub pi: f64,
    #[serde(default)]
    pub sum_gamma: f64,
    #[serde(default)]
    pub sum_gamma_dp: f64,
    #[serde(default)]
    pub sum_gamma_dp2: f64,
    #[serde(default)]
    pub sum_1mg_dp2: f64,
    #[serde(default)]
    pub n_observations: u64,
    #[serde(default)]
    pub p_informed_buy_ewma: f64,
    #[serde(default)]
    pub p_informed_sell_ewma: f64,
}

impl GmmToxicityModel {
    /// Create checkpoint for persistence.
    pub fn to_checkpoint(&self) -> GmmToxicityCheckpoint {
        GmmToxicityCheckpoint {
            sigma_noise_bps: self.sigma_noise_bps,
            mu_inf_bps: self.mu_inf_bps,
            sigma_inf_bps: self.sigma_inf_bps,
            pi: self.pi,
            sum_gamma: self.sum_gamma,
            sum_gamma_dp: self.sum_gamma_dp,
            sum_gamma_dp2: self.sum_gamma_dp2,
            sum_1mg_dp2: self.sum_1mg_dp2,
            n_observations: self.n_observations,
            p_informed_buy_ewma: self.p_informed_buy_ewma,
            p_informed_sell_ewma: self.p_informed_sell_ewma,
        }
    }

    /// Restore from checkpoint.
    pub fn from_checkpoint(checkpoint: &GmmToxicityCheckpoint, config: GmmToxicityConfig) -> Self {
        // If checkpoint is empty (all zeros), return fresh model
        if checkpoint.n_observations == 0 {
            return Self::new(config);
        }
        Self {
            sigma_noise_bps: checkpoint.sigma_noise_bps.max(0.5),
            mu_inf_bps: checkpoint.mu_inf_bps.max(1.0),
            sigma_inf_bps: checkpoint.sigma_inf_bps.max(0.5),
            pi: checkpoint.pi.clamp(0.02, 0.80),
            sum_gamma: checkpoint.sum_gamma,
            sum_gamma_dp: checkpoint.sum_gamma_dp,
            sum_gamma_dp2: checkpoint.sum_gamma_dp2,
            sum_1mg_dp2: checkpoint.sum_1mg_dp2,
            n_observations: checkpoint.n_observations,
            fills_since_em: 0,
            p_informed_buy_ewma: checkpoint.p_informed_buy_ewma.clamp(0.0, 1.0),
            p_informed_sell_ewma: checkpoint.p_informed_sell_ewma.clamp(0.0, 1.0),
            config,
        }
    }
}

/// Gaussian PDF: φ(x; μ, σ) = exp(-(x-μ)²/(2σ²)) / (σ√(2π))
fn gaussian_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma < 1e-10 {
        return if (x - mu).abs() < 1e-10 { 1e300 } else { 0.0 };
    }
    let z = (x - mu) / sigma;
    (-0.5 * z * z).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uninformed_markout_low_posterior() {
        let model = GmmToxicityModel::default();
        // Small markout around 0 -> noise component dominates
        let p = model.posterior_informed(0.5, true);
        assert!(
            p < 0.3,
            "Expected low P(informed) for small markout, got {p}"
        );
    }

    #[test]
    fn test_large_adverse_markout_high_posterior() {
        let model = GmmToxicityModel::default();
        // Large adverse markout (buy fill, price went up 15 bps) -> informed
        let p = model.posterior_informed(15.0, true);
        assert!(
            p > 0.5,
            "Expected high P(informed) for large markout, got {p}"
        );
    }

    #[test]
    fn test_symmetric_markout_sign_convention() {
        let model = GmmToxicityModel::default();
        // Buy fill with positive markout (price went up) = informed buy
        let p_buy_up = model.posterior_informed(10.0, true);
        // Sell fill with negative markout (price went down) = informed sell
        let p_sell_down = model.posterior_informed(-10.0, false);
        // Both should show high informed probability
        assert!(
            (p_buy_up - p_sell_down).abs() < 0.05,
            "Symmetric markouts should give similar posteriors"
        );
    }

    #[test]
    fn test_em_convergence() {
        let mut model = GmmToxicityModel::default();
        // Feed 200 fills: 80% noise (small markouts), 20% informed (large markouts)
        for i in 0..200u64 {
            if i % 5 == 0 {
                // Informed fill: large adverse markout
                let markout = 12.0 + (i as f64 * 0.01);
                model.update(markout, true);
            } else {
                // Noise fill: small random markout
                let markout = ((i as f64 * 7.3) % 4.0) - 2.0;
                model.update(markout, i % 2 == 0);
            }
        }
        // pi should converge toward ~0.2
        assert!(
            model.pi > 0.10 && model.pi < 0.40,
            "Expected pi near 0.2, got {}",
            model.pi
        );
        assert!(model.is_warmed_up());
    }

    #[test]
    fn test_parameter_bounds() {
        let mut model = GmmToxicityModel::default();
        // Feed extreme markouts
        for _ in 0..100 {
            model.update(100.0, true); // Extreme
            model.update(-100.0, false);
        }
        // Parameters should stay in bounds
        assert!(model.mu_inf_bps >= 1.0 && model.mu_inf_bps <= 50.0);
        assert!(model.sigma_noise_bps >= 0.5 && model.sigma_noise_bps <= 20.0);
        assert!(model.pi >= 0.02 && model.pi <= 0.80);
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut model = GmmToxicityModel::default();
        for i in 0..50 {
            model.update(5.0 * (i as f64 % 3.0) - 2.5, i % 2 == 0);
        }
        let checkpoint = model.to_checkpoint();
        let restored = GmmToxicityModel::from_checkpoint(&checkpoint, GmmToxicityConfig::default());
        assert_eq!(model.n_observations, restored.n_observations);
        assert!((model.pi - restored.pi).abs() < 1e-10);
        assert!((model.mu_inf_bps - restored.mu_inf_bps).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_pdf_basic() {
        let p = gaussian_pdf(0.0, 0.0, 1.0);
        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((p - expected).abs() < 1e-10);
    }

    #[test]
    fn test_default_model() {
        let model = GmmToxicityModel::default();
        assert_eq!(model.n_observations, 0);
        assert!(!model.is_warmed_up());
        assert!((model.toxicity_score() - 0.15).abs() < 0.01); // Prior
    }
}
