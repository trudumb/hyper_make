//! Calibration gate for paper-to-live transition.
//!
//! Assesses whether a checkpoint has sufficient learned state to safely
//! drive live trading. The gate reads observation counts already stored
//! in checkpoint component structs — no new tracking needed.

use serde::{Deserialize, Serialize};

use crate::market_maker::checkpoint::types::{CheckpointBundle, PriorReadiness, PriorVerdict};

/// Configuration for the calibration gate thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationGateConfig {
    /// Minimum paper session duration in seconds (default: 1800 = 30 min)
    pub min_paper_duration_s: f64,
    /// Minimum observations for vol/regime/fill_rate estimators
    pub min_observations: usize,
    /// Minimum observations for kappa estimator
    pub min_kappa_observations: usize,
    /// Minimum AS learning samples
    pub min_as_samples: usize,
    /// Minimum kelly fills (wins + losses)
    pub min_kelly_fills: usize,
    /// Maximum age of prior in seconds before considered stale (default: 14400 = 4h)
    pub max_prior_age_s: f64,
    /// Whether Marginal verdict is acceptable for live trading
    pub allow_marginal: bool,
}

impl Default for CalibrationGateConfig {
    fn default() -> Self {
        Self {
            min_paper_duration_s: 1800.0,
            min_observations: 200,
            min_kappa_observations: 50,
            min_as_samples: 100,
            min_kelly_fills: 20,
            max_prior_age_s: 14400.0,
            allow_marginal: true,
        }
    }
}

/// Gate that assesses checkpoint readiness for live trading.
pub struct CalibrationGate {
    config: CalibrationGateConfig,
}

impl CalibrationGate {
    pub fn new(config: CalibrationGateConfig) -> Self {
        Self { config }
    }

    /// Read-only access to config (for age checks in CLI code).
    pub fn config(&self) -> &CalibrationGateConfig {
        &self.config
    }

    /// Assess a checkpoint bundle and return a PriorReadiness snapshot.
    pub fn assess(&self, bundle: &CheckpointBundle) -> PriorReadiness {
        let vol_observations = bundle.vol_filter.observation_count;
        let kappa_observations = bundle.kappa_own.observation_count;
        let as_learning_samples = bundle.pre_fill.learning_samples;
        let regime_observations = bundle.regime_hmm.observation_count as usize;
        let fill_rate_observations = bundle.fill_rate.observation_count;
        let kelly_fills = (bundle.kelly_tracker.n_wins + bundle.kelly_tracker.n_losses) as usize;
        let session_duration_s = bundle.metadata.session_duration_s;

        // Count how many of the 5 core estimators meet their thresholds
        let mut estimators_ready: u8 = 0;
        if vol_observations >= self.config.min_observations {
            estimators_ready += 1;
        }
        if kappa_observations >= self.config.min_kappa_observations {
            estimators_ready += 1;
        }
        if as_learning_samples >= self.config.min_as_samples {
            estimators_ready += 1;
        }
        if regime_observations >= self.config.min_observations {
            estimators_ready += 1;
        }
        if fill_rate_observations >= self.config.min_observations {
            estimators_ready += 1;
        }

        let duration_met = session_duration_s >= self.config.min_paper_duration_s;
        let kelly_met = kelly_fills >= self.config.min_kelly_fills;

        let verdict = if duration_met && estimators_ready == 5 && kelly_met {
            PriorVerdict::Ready
        } else if duration_met && estimators_ready >= 3 {
            PriorVerdict::Marginal
        } else {
            PriorVerdict::Insufficient
        };

        PriorReadiness {
            verdict,
            vol_observations,
            kappa_observations,
            as_learning_samples,
            regime_observations,
            fill_rate_observations,
            kelly_fills,
            session_duration_s,
            estimators_ready,
        }
    }

    /// Check if a readiness assessment passes the gate.
    pub fn passes(&self, readiness: &PriorReadiness) -> bool {
        match readiness.verdict {
            PriorVerdict::Ready => true,
            PriorVerdict::Marginal => self.config.allow_marginal,
            PriorVerdict::Insufficient => false,
        }
    }

    /// Human-readable diagnostic explaining why the gate failed.
    pub fn explain_failure(&self, readiness: &PriorReadiness) -> String {
        let mut issues = Vec::new();

        if readiness.session_duration_s < self.config.min_paper_duration_s {
            issues.push(format!(
                "session too short ({:.0}s < {:.0}s required)",
                readiness.session_duration_s, self.config.min_paper_duration_s
            ));
        }
        if readiness.vol_observations < self.config.min_observations {
            issues.push(format!(
                "vol observations insufficient ({} < {})",
                readiness.vol_observations, self.config.min_observations
            ));
        }
        if readiness.kappa_observations < self.config.min_kappa_observations {
            issues.push(format!(
                "kappa observations insufficient ({} < {})",
                readiness.kappa_observations, self.config.min_kappa_observations
            ));
        }
        if readiness.as_learning_samples < self.config.min_as_samples {
            issues.push(format!(
                "AS learning samples insufficient ({} < {})",
                readiness.as_learning_samples, self.config.min_as_samples
            ));
        }
        if readiness.regime_observations < self.config.min_observations {
            issues.push(format!(
                "regime observations insufficient ({} < {})",
                readiness.regime_observations, self.config.min_observations
            ));
        }
        if readiness.fill_rate_observations < self.config.min_observations {
            issues.push(format!(
                "fill rate observations insufficient ({} < {})",
                readiness.fill_rate_observations, self.config.min_observations
            ));
        }
        if readiness.kelly_fills < self.config.min_kelly_fills {
            issues.push(format!(
                "kelly fills insufficient ({} < {})",
                readiness.kelly_fills, self.config.min_kelly_fills
            ));
        }

        if issues.is_empty() {
            "No issues found".to_string()
        } else {
            format!(
                "Verdict {:?} ({}/5 estimators ready): {}",
                readiness.verdict, readiness.estimators_ready,
                issues.join("; ")
            )
        }
    }
}

// === Phase 6: Bootstrap Warmup via Graduated Uncertainty ===
//
// With self-consistent gamma (Phase 1) and graduated gating (Phase 3),
// warmup is handled naturally. The only bootstrap-specific behavior:
// quote slightly tighter during warmup to attract fills for calibration,
// with small size to limit learning cost.

/// Spread discount during warmup to attract fills.
///
/// Returns a multiplier [0.85, 1.0] applied to gamma (lower gamma -> tighter spreads).
/// After 200 fills, the discount vanishes and spreads are fully model-driven.
pub fn warmup_spread_discount(fill_count: usize) -> f64 {
    if fill_count < 50 {
        0.85 // 15% tighter to attract fills quickly
    } else if fill_count < 200 {
        0.95 // 5% tighter, still learning
    } else {
        1.0 // Fully calibrated
    }
}

/// Size multiplier during warmup to limit learning cost.
///
/// Returns a multiplier [0.3, 1.0] applied to target_liquidity.
/// Small size while learning prevents large losses from miscalibrated models.
pub fn warmup_size_multiplier(fill_count: usize) -> f64 {
    if fill_count < 50 {
        0.3 // Small size while learning AS/kappa
    } else if fill_count < 200 {
        0.7 // Medium size, growing confidence
    } else {
        1.0 // Full size
    }
}

/// Result of loading and assessing a prior checkpoint.
#[allow(dead_code)]
pub enum PriorStatus {
    /// Fully calibrated, safe for live at full confidence.
    Ready(CheckpointBundle),
    /// Partially calibrated — live with defensive spreads.
    Marginal(CheckpointBundle, PriorReadiness),
    /// Checkpoint exists but is too old.
    Stale,
    /// No checkpoint found.
    Missing,
    /// Checkpoint exists but has insufficient data.
    Insufficient(PriorReadiness),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::calibration::parameter_learner::LearnedParameters;
    use crate::market_maker::checkpoint::types::*;

    fn make_bundle(
        session_duration_s: f64,
        vol_obs: usize,
        kappa_obs: usize,
        as_samples: usize,
        regime_obs: u64,
        fill_rate_obs: usize,
        n_wins: u64,
        n_losses: u64,
    ) -> CheckpointBundle {
        CheckpointBundle {
            metadata: CheckpointMetadata {
                version: 1,
                timestamp_ms: 1700000000000,
                asset: "HYPE".to_string(),
                session_duration_s,
            },
            learned_params: LearnedParameters::default(),
            pre_fill: PreFillCheckpoint {
                learning_samples: as_samples,
                ..PreFillCheckpoint::default()
            },
            enhanced: EnhancedCheckpoint::default(),
            vol_filter: VolFilterCheckpoint {
                observation_count: vol_obs,
                ..VolFilterCheckpoint::default()
            },
            regime_hmm: RegimeHMMCheckpoint {
                observation_count: regime_obs,
                ..RegimeHMMCheckpoint::default()
            },
            informed_flow: InformedFlowCheckpoint::default(),
            fill_rate: FillRateCheckpoint {
                observation_count: fill_rate_obs,
                ..FillRateCheckpoint::default()
            },
            kappa_own: KappaCheckpoint {
                observation_count: kappa_obs,
                ..KappaCheckpoint::default()
            },
            kappa_bid: KappaCheckpoint::default(),
            kappa_ask: KappaCheckpoint::default(),
            momentum: MomentumCheckpoint::default(),
            kelly_tracker: KellyTrackerCheckpoint {
                n_wins,
                n_losses,
                ..KellyTrackerCheckpoint::default()
            },
            ensemble_weights: EnsembleWeightsCheckpoint::default(),
            quote_outcomes: QuoteOutcomeCheckpoint::default(),
            spread_bandit: crate::market_maker::learning::spread_bandit::SpreadBanditCheckpoint::default(),
            baseline_tracker: crate::market_maker::checkpoint::types::BaselineTrackerCheckpoint::default(),
            kill_switch: KillSwitchCheckpoint::default(),
            readiness: PriorReadiness::default(),
            calibration_coordinator: crate::market_maker::checkpoint::types::CalibrationCoordinatorCheckpoint::default(),
        }
    }

    #[test]
    fn test_ready_verdict() {
        let gate = CalibrationGate::new(CalibrationGateConfig::default());
        let bundle = make_bundle(2000.0, 300, 100, 200, 300, 250, 15, 10);
        let readiness = gate.assess(&bundle);
        assert_eq!(readiness.verdict, PriorVerdict::Ready);
        assert_eq!(readiness.estimators_ready, 5);
        assert!(gate.passes(&readiness));
    }

    #[test]
    fn test_marginal_verdict() {
        let gate = CalibrationGate::new(CalibrationGateConfig::default());
        // 3/5 estimators ready (vol, kappa, AS ok; regime and fill_rate insufficient)
        let bundle = make_bundle(2000.0, 300, 100, 200, 10, 10, 5, 3);
        let readiness = gate.assess(&bundle);
        assert_eq!(readiness.verdict, PriorVerdict::Marginal);
        assert_eq!(readiness.estimators_ready, 3);
        assert!(gate.passes(&readiness)); // allow_marginal = true by default
    }

    #[test]
    fn test_insufficient_verdict() {
        let gate = CalibrationGate::new(CalibrationGateConfig::default());
        let bundle = make_bundle(100.0, 10, 5, 10, 5, 10, 0, 0); // too short, too few
        let readiness = gate.assess(&bundle);
        assert_eq!(readiness.verdict, PriorVerdict::Insufficient);
        assert!(!gate.passes(&readiness));
    }

    #[test]
    fn test_passes_logic() {
        let mut config = CalibrationGateConfig::default();
        config.allow_marginal = false;
        let gate = CalibrationGate::new(config);

        // Marginal should NOT pass when allow_marginal is false
        let bundle = make_bundle(2000.0, 300, 100, 200, 10, 10, 5, 3);
        let readiness = gate.assess(&bundle);
        assert_eq!(readiness.verdict, PriorVerdict::Marginal);
        assert!(!gate.passes(&readiness));
    }

    #[test]
    fn test_explain_failure() {
        let gate = CalibrationGate::new(CalibrationGateConfig::default());
        let bundle = make_bundle(100.0, 10, 5, 10, 5, 10, 0, 0);
        let readiness = gate.assess(&bundle);
        let explanation = gate.explain_failure(&readiness);
        assert!(explanation.contains("session too short"));
        assert!(explanation.contains("vol observations insufficient"));
        assert!(explanation.contains("kappa observations insufficient"));
    }

    #[test]
    fn test_duration_required_for_ready() {
        let gate = CalibrationGate::new(CalibrationGateConfig::default());
        // All estimators have enough data but session is too short
        let bundle = make_bundle(500.0, 300, 100, 200, 300, 250, 15, 10);
        let readiness = gate.assess(&bundle);
        // Short duration prevents Ready, but 5/5 estimators + duration -> Marginal not Ready
        assert_ne!(readiness.verdict, PriorVerdict::Ready);
    }

    #[test]
    fn test_warmup_spread_discount() {
        // Early warmup: 15% tighter to attract fills
        assert!((warmup_spread_discount(0) - 0.85).abs() < 0.001);
        assert!((warmup_spread_discount(25) - 0.85).abs() < 0.001);
        assert!((warmup_spread_discount(49) - 0.85).abs() < 0.001);

        // Mid warmup: 5% tighter
        assert!((warmup_spread_discount(50) - 0.95).abs() < 0.001);
        assert!((warmup_spread_discount(100) - 0.95).abs() < 0.001);
        assert!((warmup_spread_discount(199) - 0.95).abs() < 0.001);

        // Fully calibrated
        assert!((warmup_spread_discount(200) - 1.0).abs() < 0.001);
        assert!((warmup_spread_discount(1000) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_warmup_size_multiplier() {
        // Early warmup: small size
        assert!((warmup_size_multiplier(0) - 0.3).abs() < 0.001);
        assert!((warmup_size_multiplier(49) - 0.3).abs() < 0.001);

        // Mid warmup: medium size
        assert!((warmup_size_multiplier(50) - 0.7).abs() < 0.001);
        assert!((warmup_size_multiplier(199) - 0.7).abs() < 0.001);

        // Fully calibrated: full size
        assert!((warmup_size_multiplier(200) - 1.0).abs() < 0.001);
        assert!((warmup_size_multiplier(500) - 1.0).abs() < 0.001);
    }
}
