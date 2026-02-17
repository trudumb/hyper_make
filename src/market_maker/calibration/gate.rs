//! Calibration gate for paper-to-live transition.
//!
//! Two-phase assessment:
//! - **Phase 1 (Market Readiness)**: vol + regime estimators — converge from
//!   market data alone, no fills required. Sufficient for Marginal (go live
//!   with defensive spreads).
//! - **Phase 2 (Execution Readiness)**: kappa + AS + fill_rate — require our
//!   fills for calibration. All three must converge for Ready verdict.
//!
//! Thresholds are set to be achievable within 30 minutes on low-volume assets
//! (~2 trades/min). The gate reads observation counts from checkpoint structs.

use serde::{Deserialize, Serialize};

use crate::market_maker::checkpoint::types::{CheckpointBundle, PriorReadiness, PriorVerdict};

/// Configuration for the calibration gate thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationGateConfig {
    /// Minimum paper session duration in seconds (default: 1800 = 30 min)
    pub min_paper_duration_s: f64,
    /// Phase 1: Minimum observations for market-data estimators (vol, regime).
    /// These accumulate from market trades, not our fills.
    /// Default 50 — achievable in 30 min even at ~2 trades/min.
    pub min_market_observations: usize,
    /// Phase 2: Minimum fill-based observations for kappa estimator.
    /// Needs our fills (paper or live). Default 10.
    pub min_kappa_observations: usize,
    /// Phase 2: Minimum AS learning samples (needs our fills). Default 15.
    pub min_as_samples: usize,
    /// Phase 2: Minimum fill rate observations (needs our fills). Default 15.
    pub min_fill_rate_observations: usize,
    /// Minimum kelly fills (wins + losses) for Ready verdict. Default 5.
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
            min_market_observations: 50,
            min_kappa_observations: 10,
            min_as_samples: 15,
            min_fill_rate_observations: 15,
            min_kelly_fills: 5,
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
    ///
    /// Two-phase assessment:
    /// - Phase 1 (Market): vol + regime — only needs market data
    /// - Phase 2 (Execution): kappa + AS + fill_rate — needs our fills
    pub fn assess(&self, bundle: &CheckpointBundle) -> PriorReadiness {
        let vol_observations = bundle.vol_filter.observation_count;
        // Use cumulative count (never decremented by rolling window expiry).
        // Backward-compat: old checkpoints have total_observations=0, fall back to rolling.
        let kappa_observations = if bundle.kappa_own.total_observations > 0 {
            bundle.kappa_own.total_observations
        } else {
            bundle.kappa_own.observation_count
        };
        let as_learning_samples = bundle.pre_fill.learning_samples;
        let regime_observations = bundle.regime_hmm.observation_count as usize;
        let fill_rate_observations = bundle.fill_rate.observation_count;
        let kelly_fills = (bundle.kelly_tracker.n_wins + bundle.kelly_tracker.n_losses) as usize;
        let session_duration_s = bundle.metadata.session_duration_s;

        // Phase 1: Market readiness (market data only, no fills needed)
        let vol_ready = vol_observations >= self.config.min_market_observations;
        let regime_ready = regime_observations >= self.config.min_market_observations;
        let phase1_ready = vol_ready && regime_ready;

        // Phase 2: Execution readiness (needs our fills)
        let kappa_ready = kappa_observations >= self.config.min_kappa_observations;
        let as_ready = as_learning_samples >= self.config.min_as_samples;
        let fill_rate_ready = fill_rate_observations >= self.config.min_fill_rate_observations;
        let phase2_ready = kappa_ready && as_ready && fill_rate_ready;

        // Count estimators for diagnostic reporting
        let mut estimators_ready: u8 = 0;
        if vol_ready {
            estimators_ready += 1;
        }
        if kappa_ready {
            estimators_ready += 1;
        }
        if as_ready {
            estimators_ready += 1;
        }
        if regime_ready {
            estimators_ready += 1;
        }
        if fill_rate_ready {
            estimators_ready += 1;
        }

        let duration_met = session_duration_s >= self.config.min_paper_duration_s;
        let kelly_met = kelly_fills >= self.config.min_kelly_fills;

        let verdict = if duration_met && phase1_ready && phase2_ready && kelly_met {
            PriorVerdict::Ready
        } else if duration_met && phase1_ready {
            // Market microstructure understood — safe to go live with defensive spreads.
            // Execution calibration (AS, fill_rate) will converge from live fills.
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
        // Phase 1: Market data estimators
        if readiness.vol_observations < self.config.min_market_observations {
            issues.push(format!(
                "vol observations insufficient ({} < {} [market])",
                readiness.vol_observations, self.config.min_market_observations
            ));
        }
        if readiness.regime_observations < self.config.min_market_observations {
            issues.push(format!(
                "regime observations insufficient ({} < {} [market])",
                readiness.regime_observations, self.config.min_market_observations
            ));
        }
        // Phase 2: Fill-based estimators
        if readiness.kappa_observations < self.config.min_kappa_observations {
            issues.push(format!(
                "kappa observations insufficient ({} < {} [fill])",
                readiness.kappa_observations, self.config.min_kappa_observations
            ));
        }
        if readiness.as_learning_samples < self.config.min_as_samples {
            issues.push(format!(
                "AS learning samples insufficient ({} < {} [fill])",
                readiness.as_learning_samples, self.config.min_as_samples
            ));
        }
        if readiness.fill_rate_observations < self.config.min_fill_rate_observations {
            issues.push(format!(
                "fill rate observations insufficient ({} < {} [fill])",
                readiness.fill_rate_observations, self.config.min_fill_rate_observations
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
        // All estimators above thresholds: vol≥50, regime≥50, kappa≥10, AS≥15, fill_rate≥15, kelly≥5
        let bundle = make_bundle(2000.0, 100, 20, 30, 100, 30, 5, 5);
        let readiness = gate.assess(&bundle);
        assert_eq!(readiness.verdict, PriorVerdict::Ready);
        assert_eq!(readiness.estimators_ready, 5);
        assert!(gate.passes(&readiness));
    }

    #[test]
    fn test_marginal_phase1_only() {
        let gate = CalibrationGate::new(CalibrationGateConfig::default());
        // Phase 1 ready (vol=100≥50, regime=60≥50), Phase 2 not ready (0 fills)
        // This is the typical paper trading scenario: market data converged, no fills yet.
        let bundle = make_bundle(2000.0, 100, 0, 0, 60, 0, 0, 0);
        let readiness = gate.assess(&bundle);
        assert_eq!(readiness.verdict, PriorVerdict::Marginal);
        assert_eq!(readiness.estimators_ready, 2); // only vol + regime
        assert!(gate.passes(&readiness)); // allow_marginal = true by default
    }

    #[test]
    fn test_insufficient_phase1_not_met() {
        let gate = CalibrationGate::new(CalibrationGateConfig::default());
        // Phase 1 not ready: vol=10<50, regime=5<50
        let bundle = make_bundle(2000.0, 10, 5, 10, 5, 10, 0, 0);
        let readiness = gate.assess(&bundle);
        assert_eq!(readiness.verdict, PriorVerdict::Insufficient);
        assert!(!gate.passes(&readiness));
    }

    #[test]
    fn test_insufficient_short_session() {
        let gate = CalibrationGate::new(CalibrationGateConfig::default());
        // All estimators ready, but session too short
        let bundle = make_bundle(100.0, 100, 20, 30, 100, 30, 5, 5);
        let readiness = gate.assess(&bundle);
        assert_eq!(readiness.verdict, PriorVerdict::Insufficient);
        assert_eq!(readiness.estimators_ready, 5);
        assert!(!gate.passes(&readiness));
    }

    #[test]
    fn test_passes_logic() {
        let mut config = CalibrationGateConfig::default();
        config.allow_marginal = false;
        let gate = CalibrationGate::new(config);

        // Marginal should NOT pass when allow_marginal is false
        let bundle = make_bundle(2000.0, 100, 0, 0, 60, 0, 0, 0);
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
        // All estimators ready but session too short → Insufficient (not Ready)
        let bundle = make_bundle(500.0, 100, 20, 30, 100, 30, 5, 5);
        let readiness = gate.assess(&bundle);
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

    /// Phase 1 (vol+regime) ready → Marginal, even with zero fills.
    /// This is the key scenario: low-volume asset, paper trading for 30 min,
    /// market data converged, zero fills — should still pass the gate.
    #[test]
    fn test_marginal_zero_fills_market_converged() {
        let gate = CalibrationGate::new(CalibrationGateConfig::default());

        let bundle = make_bundle(
            2000.0, // session long enough
            80,     // vol_obs: meets 50 threshold (market data)
            0,      // kappa_obs: zero fills
            0,      // as_samples: zero fills
            60,     // regime_obs: meets 50 threshold (market data)
            0,      // fill_rate: zero fills
            0,      // kelly wins
            0,      // kelly losses
        );

        let readiness = gate.assess(&bundle);
        assert_eq!(readiness.verdict, PriorVerdict::Marginal);
        assert_eq!(readiness.estimators_ready, 2); // only vol + regime
        assert!(gate.passes(&readiness)); // allow_marginal = true
    }

    /// Ready requires Phase 2 (fill-based) in addition to Phase 1.
    #[test]
    fn test_ready_requires_fills() {
        let gate = CalibrationGate::new(CalibrationGateConfig::default());

        // Phase 1 ready, Phase 2 partial (only kappa met, AS and fill_rate not)
        let bundle = make_bundle(
            2000.0, // duration OK
            100,    // vol: OK
            15,     // kappa: OK (≥10)
            5,      // AS: NOT OK (<15)
            100,    // regime: OK
            5,      // fill_rate: NOT OK (<15)
            3,      // kelly wins
            3,      // kelly losses (=6, ≥5 OK)
        );

        let readiness = gate.assess(&bundle);
        // Phase 1 ready but Phase 2 not → Marginal, not Ready
        assert_eq!(readiness.verdict, PriorVerdict::Marginal);
        assert_eq!(readiness.estimators_ready, 3); // vol + regime + kappa
    }

    /// Insufficient when session too short despite all estimators ready.
    #[test]
    fn test_insufficient_short_session_all_estimators_ready() {
        let gate = CalibrationGate::new(CalibrationGateConfig::default());

        let bundle = make_bundle(
            500.0,  // too short (need 1800s)
            100,    // vol_obs: ready
            20,     // kappa_obs: ready
            30,     // as_samples: ready
            100,    // regime_obs: ready
            30,     // fill_rate: ready
            5,      // kelly wins
            5,      // kelly losses
        );

        let readiness = gate.assess(&bundle);
        assert_eq!(readiness.verdict, PriorVerdict::Insufficient);
        assert_eq!(readiness.estimators_ready, 5);
    }
}
