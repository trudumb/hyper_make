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

use crate::market_maker::checkpoint::transfer::{PriorAgePolicy, prior_age_confidence};
use crate::market_maker::checkpoint::types::{CheckpointBundle, PriorReadiness, PriorVerdict};

/// Continuous confidence assessment for a prior checkpoint.
///
/// Replaces binary Ready/Marginal/Insufficient with smooth [0,1] scores
/// that drive spread/size multipliers. A prior with 8 kappa observations
/// (threshold=10) gets ~0.8 confidence, not zero.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PriorConfidence {
    /// Structural confidence: vol + regime estimator saturation [0,1].
    /// Geometric mean of (obs / threshold) for market-data estimators.
    pub structural: f64,
    /// Execution confidence: kappa + AS + fill_rate saturation [0,1].
    /// Geometric mean of (obs / threshold) for fill-based estimators.
    pub execution: f64,
    /// Age confidence: freshness of the prior [0,1].
    /// From `prior_age_confidence()`.
    pub age: f64,
    /// Overall confidence: combined score [0,1].
    /// `min(structural, age) × (base + (1 - base) × execution)`.
    /// `base` is source-dependent: paper=0.3, live=0.5, unknown=0.4.
    pub overall: f64,
}

/// Spread multiplier from confidence: wider spreads when confidence is low.
///
/// Returns [1.0, 3.0] — at confidence=1.0 returns 1.0 (no widening),
/// at confidence=0.0 returns 3.0 (3x wider spreads for safety).
pub fn spread_multiplier_from_confidence(confidence: f64) -> f64 {
    let c = confidence.clamp(0.0, 1.0);
    // Linear: 3.0 - 2.0 * c
    3.0 - 2.0 * c
}

/// Size multiplier from confidence: smaller size when confidence is low.
///
/// Returns [0.1, 1.0] — at confidence=1.0 returns 1.0 (full size),
/// at confidence=0.0 returns 0.1 (10% size to limit risk).
pub fn size_multiplier_from_confidence(confidence: f64) -> f64 {
    let c = confidence.clamp(0.0, 1.0);
    // Linear: 0.1 + 0.9 * c
    0.1 + 0.9 * c
}

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
    /// Base credit for execution confidence when prior source is paper.
    /// Paper has no realistic queue/latency/adversarial flow, so execution
    /// confidence gets a lower base than live. Formula:
    /// `overall = min(structural, age) × (base + (1 - base) × execution)`
    /// Default: 0.3 (30% base credit for paper).
    #[serde(default = "default_paper_execution_base_credit")]
    pub paper_execution_base_credit: f64,
    /// Base credit for execution confidence when prior source is live.
    /// Default: 0.5 (50% base credit — unchanged from original formula).
    #[serde(default = "default_live_execution_base_credit")]
    pub live_execution_base_credit: f64,
}

fn default_paper_execution_base_credit() -> f64 {
    0.3
}

fn default_live_execution_base_credit() -> f64 {
    0.5
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
            paper_execution_base_credit: 0.3,
            live_execution_base_credit: 0.5,
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

    /// Compute continuous confidence for a checkpoint bundle.
    ///
    /// Returns a `PriorConfidence` with structural, execution, age, and overall scores.
    /// This replaces the binary Ready/Marginal/Insufficient with smooth [0,1] values.
    pub fn confidence(&self, bundle: &CheckpointBundle, age_s: f64) -> PriorConfidence {
        let age_policy = PriorAgePolicy::default();

        // Structural confidence: geometric mean of vol/regime saturation
        let vol_sat = (bundle.vol_filter.observation_count as f64
            / self.config.min_market_observations.max(1) as f64)
            .min(1.0);
        let regime_obs = bundle.regime_hmm.observation_count as f64;
        let regime_sat =
            (regime_obs / self.config.min_market_observations.max(1) as f64).min(1.0);
        let structural = (vol_sat * regime_sat).sqrt();

        // Execution confidence: geometric mean of kappa/AS/fill_rate saturation
        let kappa_obs = if bundle.kappa_own.total_observations > 0 {
            bundle.kappa_own.total_observations
        } else {
            bundle.kappa_own.observation_count
        };
        let kappa_sat =
            (kappa_obs as f64 / self.config.min_kappa_observations.max(1) as f64).min(1.0);
        let as_sat = (bundle.pre_fill.learning_samples as f64
            / self.config.min_as_samples.max(1) as f64)
            .min(1.0);
        let fill_sat = (bundle.fill_rate.observation_count as f64
            / self.config.min_fill_rate_observations.max(1) as f64)
            .min(1.0);
        // Use cubic root for 3 factors
        let execution = (kappa_sat * as_sat * fill_sat).cbrt();

        // Age confidence from policy
        let age = prior_age_confidence(age_s, &age_policy);

        // Source-aware execution base credit:
        // Paper has no realistic queue/latency/adversarial flow — execution confidence
        // gets a lower base than live. Unknown source (old checkpoints with empty
        // source_mode) gets midpoint as safe default.
        let base_credit = match bundle.metadata.source_mode.as_str() {
            "paper" => self.config.paper_execution_base_credit,
            "live" => self.config.live_execution_base_credit,
            _ => {
                // Unknown source: average of paper and live as safe default
                (self.config.paper_execution_base_credit
                    + self.config.live_execution_base_credit) / 2.0
            }
        };

        // Overall: min(structural, age) × (base + (1 - base) × execution)
        // Structural and age are both required foundations.
        // Execution is additive on top — zero execution gets `base` credit.
        let overall = structural.min(age) * (base_credit + (1.0 - base_credit) * execution);

        PriorConfidence {
            structural,
            execution,
            age,
            overall,
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

/// Snapshot for A/B measurement of prior convergence speed.
///
/// Logged periodically to analytics JSONL for future empirical estimation of
/// the paper→live domain gap. Will eventually replace heuristic base credits
/// with measured precision inflation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceSnapshot {
    /// Total fills received since session start.
    pub fill_count: usize,
    /// Seconds since session start.
    pub session_elapsed_s: f64,
    /// Current prior confidence [0, 1].
    pub prior_confidence: f64,
    /// Current spread multiplier from confidence.
    pub spread_multiplier: f64,
    /// Current size multiplier from confidence.
    pub size_multiplier: f64,
    /// Cumulative PnL in bps.
    pub cumulative_pnl_bps: f64,
    /// Mean realized gross edge in bps.
    pub mean_realized_edge_bps: f64,
    /// Whether a prior was loaded at startup.
    pub has_prior: bool,
    /// Source mode of the prior ("paper", "live", or empty).
    pub prior_source_mode: String,
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
                ..Default::default()
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
            kappa_orchestrator: crate::market_maker::checkpoint::types::KappaOrchestratorCheckpoint::default(),
            prior_confidence: 0.0,
            bayesian_fair_value: Default::default(),
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

    // === PriorConfidence tests ===

    #[test]
    fn test_confidence_zero_observations() {
        let gate = CalibrationGate::new(CalibrationGateConfig::default());
        let bundle = make_bundle(0.0, 0, 0, 0, 0, 0, 0, 0);
        let conf = gate.confidence(&bundle, 0.0);
        assert_eq!(conf.structural, 0.0);
        assert_eq!(conf.execution, 0.0);
        assert_eq!(conf.age, 1.0); // fresh
        assert_eq!(conf.overall, 0.0); // min(0, 1) * (0.5+0) = 0
    }

    #[test]
    fn test_confidence_full_calibration() {
        let gate = CalibrationGate::new(CalibrationGateConfig::default());
        let bundle = make_bundle(2000.0, 100, 20, 30, 100, 30, 5, 5);
        let conf = gate.confidence(&bundle, 100.0); // fresh
        assert!((conf.structural - 1.0).abs() < 0.01);
        assert!((conf.execution - 1.0).abs() < 0.01);
        assert_eq!(conf.age, 1.0);
        assert!((conf.overall - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_confidence_structural_only() {
        let gate = CalibrationGate::new(CalibrationGateConfig::default());
        // Market data converged, zero fills
        let bundle = make_bundle(2000.0, 80, 0, 0, 60, 0, 0, 0);
        let conf = gate.confidence(&bundle, 100.0);
        assert!(conf.structural > 0.9, "structural should be ~1.0: {}", conf.structural);
        assert_eq!(conf.execution, 0.0);
        // overall = min(1.0, 1.0) * (0.4 + 0.6*0) = 0.4  (unknown source → 0.4 base)
        assert!((conf.overall - 0.4).abs() < 0.05, "overall ~0.4: {}", conf.overall);
    }

    #[test]
    fn test_confidence_partial_execution() {
        let gate = CalibrationGate::new(CalibrationGateConfig::default());
        // 8/10 kappa (below threshold but close), other execution at 0
        let bundle = make_bundle(2000.0, 100, 8, 0, 100, 0, 0, 0);
        let conf = gate.confidence(&bundle, 100.0);
        assert!(conf.structural > 0.9);
        // kappa: 8/10=0.8, AS: 0/15=0, fill: 0/15=0 → geomean = 0
        assert_eq!(conf.execution, 0.0);
    }

    #[test]
    fn test_confidence_aged_prior() {
        let gate = CalibrationGate::new(CalibrationGateConfig::default());
        let bundle = make_bundle(2000.0, 100, 20, 30, 100, 30, 5, 5);
        // 48h old
        let conf = gate.confidence(&bundle, 48.0 * 3600.0);
        assert!(conf.age < 1.0, "Should be age-decayed: {}", conf.age);
        assert!(conf.age > 0.15, "Should be above floor: {}", conf.age);
        assert!(conf.overall < 1.0);
        assert!(conf.overall > 0.1);
    }

    #[test]
    fn test_spread_multiplier_boundaries() {
        assert!((spread_multiplier_from_confidence(1.0) - 1.0).abs() < 0.001);
        assert!((spread_multiplier_from_confidence(0.0) - 3.0).abs() < 0.001);
        assert!((spread_multiplier_from_confidence(0.5) - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_size_multiplier_boundaries() {
        assert!((size_multiplier_from_confidence(1.0) - 1.0).abs() < 0.001);
        assert!((size_multiplier_from_confidence(0.0) - 0.1).abs() < 0.001);
        assert!((size_multiplier_from_confidence(0.5) - 0.55).abs() < 0.001);
    }

    // === Source-aware confidence tests ===

    fn make_bundle_with_source(
        source_mode: &str,
        vol_obs: usize,
        kappa_obs: usize,
        as_samples: usize,
        regime_obs: u64,
        fill_rate_obs: usize,
    ) -> CheckpointBundle {
        let mut bundle = make_bundle(2000.0, vol_obs, kappa_obs, as_samples, regime_obs, fill_rate_obs, 5, 5);
        bundle.metadata.source_mode = source_mode.to_string();
        bundle
    }

    #[test]
    fn test_confidence_paper_vs_live_source_mode() {
        let gate = CalibrationGate::new(CalibrationGateConfig::default());

        // Same observations, different source modes
        let paper_bundle = make_bundle_with_source("paper", 100, 5, 15, 100, 15);
        let live_bundle = make_bundle_with_source("live", 100, 5, 15, 100, 15);

        let paper_conf = gate.confidence(&paper_bundle, 100.0);
        let live_conf = gate.confidence(&live_bundle, 100.0);

        // Paper should get lower overall confidence due to lower base credit
        assert!(
            paper_conf.overall < live_conf.overall,
            "Paper confidence {} should be < live confidence {}",
            paper_conf.overall, live_conf.overall
        );

        // Structural and execution should be the same (source doesn't affect these)
        assert!(
            (paper_conf.structural - live_conf.structural).abs() < 0.001,
            "Structural should be equal"
        );
        assert!(
            (paper_conf.execution - live_conf.execution).abs() < 0.001,
            "Execution should be equal"
        );
    }

    #[test]
    fn test_confidence_unknown_source_mode() {
        let gate = CalibrationGate::new(CalibrationGateConfig::default());

        let paper_bundle = make_bundle_with_source("paper", 100, 5, 15, 100, 15);
        let unknown_bundle = make_bundle_with_source("", 100, 5, 15, 100, 15);
        let live_bundle = make_bundle_with_source("live", 100, 5, 15, 100, 15);

        let paper_conf = gate.confidence(&paper_bundle, 100.0);
        let unknown_conf = gate.confidence(&unknown_bundle, 100.0);
        let live_conf = gate.confidence(&live_bundle, 100.0);

        // Unknown should be between paper and live
        assert!(
            unknown_conf.overall > paper_conf.overall,
            "Unknown {} should be > paper {}",
            unknown_conf.overall, paper_conf.overall
        );
        assert!(
            unknown_conf.overall < live_conf.overall,
            "Unknown {} should be < live {}",
            unknown_conf.overall, live_conf.overall
        );
    }

    #[test]
    fn test_convergence_snapshot_serde() {
        let snapshot = ConvergenceSnapshot {
            fill_count: 42,
            session_elapsed_s: 1800.0,
            prior_confidence: 0.35,
            spread_multiplier: 2.3,
            size_multiplier: 0.42,
            cumulative_pnl_bps: -1.5,
            mean_realized_edge_bps: -0.06,
            has_prior: true,
            prior_source_mode: "paper".to_string(),
        };

        let json = serde_json::to_string(&snapshot).unwrap();
        let deserialized: ConvergenceSnapshot = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.fill_count, 42);
        assert!((deserialized.prior_confidence - 0.35).abs() < f64::EPSILON);
        assert_eq!(deserialized.prior_source_mode, "paper");
    }

    #[test]
    fn test_confidence_paper_zero_execution() {
        // Paper with zero execution → base credit = 0.3
        let gate = CalibrationGate::new(CalibrationGateConfig::default());
        let bundle = make_bundle_with_source("paper", 100, 0, 0, 100, 0);

        let conf = gate.confidence(&bundle, 100.0);
        assert!(conf.structural > 0.9);
        assert_eq!(conf.execution, 0.0);
        // overall = min(~1, 1) × (0.3 + 0.7×0) = 0.3
        assert!(
            (conf.overall - 0.3).abs() < 0.05,
            "Paper with zero execution should get ~0.3: {}",
            conf.overall
        );
    }
}
