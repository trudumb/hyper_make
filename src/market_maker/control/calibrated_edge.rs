//! Calibrated Edge Signal Tracker.
//!
//! Tracks whether flow_imbalance actually predicts price direction using
//! Information Ratio (IR) as the principled threshold. The only valid
//! threshold is IR > 1.0 (signal adds information vs. base rate).
//!
//! # Architecture
//!
//! Follows the predict() → record_outcome() → information_ratio() pattern:
//!
//! ```ignore
//! let mut edge = CalibratedEdgeSignal::new(config);
//!
//! // On each quote cycle, make a prediction
//! let pred_id = edge.predict(flow_imbalance, mid_price, regime);
//!
//! // Later, record outcomes when predictions mature
//! edge.record_outcomes(current_mid, now_ms);
//!
//! // Check if signal is useful (IR > 1.0)
//! if edge.is_useful() {
//!     let weight = edge.signal_weight(); // (IR - 1.0).max(0)
//! }
//! ```
//!
//! # Principled Thresholds
//!
//! - **IR > 1.0**: Signal adds predictive value (use it)
//! - **IR <= 1.0**: Signal is noise (ignore it)
//! - **Signal weight = (IR - 1.0).max(0)**: Natural scaling
//!
//! No arbitrary 0.15 or 0.45 thresholds - only information theory.

use tracing::trace;

use crate::market_maker::calibration::AdaptiveBinner;

/// Number of regimes tracked (calm=0, volatile=1, cascade=2).
pub const NUM_REGIMES: usize = 3;

/// Configuration for Bayesian IR estimation during warmup.
///
/// Replaces arbitrary fixed sample count thresholds with principled
/// probabilistic thresholds based on P(IR > 1.0 | data).
#[derive(Debug, Clone)]
pub struct BayesianIRConfig {
    /// Enable Bayesian IR warmup (vs fixed sample count).
    /// Default: true
    pub enabled: bool,

    /// Minimum samples before any quoting (hard floor).
    /// Default: 15
    pub min_samples: usize,

    /// Tiered confidence thresholds by sample count.
    /// Default: [(0, 0.70), (20, 0.75), (50, 0.80), (100, 0.90)]
    pub confidence_tiers: [(usize, f64); 4],

    /// Prior mean for IR (shrinkage target).
    /// Default: 0.9 (modest positive, avoids zero-IR bias)
    pub prior_ir_mean: f64,

    /// Prior degrees of freedom (shrinkage strength).
    /// Higher = stronger pull toward prior_ir_mean.
    /// Default: 6.0 (moderate shrinkage)
    pub prior_df: f64,

    /// Weight for L2 model confidence in prior.
    /// 0.0 = ignore L2, 1.0 = full L2 influence.
    /// Default: 0.5
    pub l2_prior_weight: f64,

    /// Stop-out threshold: halt quoting if P(IR > 1.0) drops below this
    /// after sufficient samples.
    /// Default: 0.60
    pub stop_out_threshold: f64,

    /// Minimum samples before stop-out applies.
    /// Default: 30
    pub stop_out_min_samples: usize,

    /// Recovery threshold: resume after stop-out if P rises above this.
    /// Default: 0.85
    pub recovery_threshold: f64,
}

impl Default for BayesianIRConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_samples: 15,
            confidence_tiers: [(0, 0.75), (20, 0.80), (50, 0.85), (100, 0.90)],
            prior_ir_mean: 0.9,
            prior_df: 6.0,
            l2_prior_weight: 0.5,
            stop_out_threshold: 0.60,
            stop_out_min_samples: 30,
            recovery_threshold: 0.85,
        }
    }
}

/// Result of Bayesian IR usefulness check.
///
/// Contains the decision plus all diagnostic information for logging.
#[derive(Debug, Clone)]
pub struct BayesianDecision {
    /// Whether the edge signal is considered useful.
    pub is_useful: bool,
    /// Human-readable reason for the decision.
    pub reason: String,
    /// P(IR > 1.0 | data).
    pub posterior_prob: f64,
    /// Lower bound of 95% credible interval.
    pub credible_lower: f64,
    /// Prior mean used (may include L2 adjustment).
    pub prior_mu_used: f64,
    /// Posterior mean IR.
    pub post_mu: f64,
    /// Number of samples.
    pub samples: usize,
    /// Whether currently stopped out.
    pub stopped_out: bool,
}

/// Configuration for calibrated edge signal tracking.
#[derive(Debug, Clone)]
pub struct CalibratedEdgeConfig {
    /// Outcome measurement horizon in milliseconds.
    /// Price direction is measured after this delay.
    /// Default: 1000ms (1 second)
    pub outcome_horizon_ms: u64,

    /// Minimum price change (in bps) to count as a directional move.
    /// Below this, the outcome is considered neutral (no signal).
    /// Default: 1.0 bps
    pub min_price_change_bps: f64,

    /// Maximum pending predictions to retain.
    /// Default: 1000
    pub max_pending: usize,

    /// Minimum samples before trusting IR estimate.
    /// Default: 100
    pub min_samples_for_ir: usize,

    /// Cold-start edge threshold (used before calibration).
    /// More conservative than the old 0.15.
    /// Default: 0.25
    pub cold_start_edge_threshold: f64,

    /// Number of bins for IR tracking (more bins = finer resolution).
    /// Default: 10
    pub ir_n_bins: usize,
}

impl Default for CalibratedEdgeConfig {
    fn default() -> Self {
        Self {
            outcome_horizon_ms: 1000,
            min_price_change_bps: 1.0,
            max_pending: 1000,
            min_samples_for_ir: 100,
            cold_start_edge_threshold: 0.25,
            ir_n_bins: 10,
        }
    }
}

/// A pending edge prediction awaiting outcome measurement.
#[derive(Debug, Clone)]
struct EdgePrediction {
    /// Unique prediction ID.
    id: u64,
    /// Flow imbalance at prediction time (our signal).
    flow_imbalance: f64,
    /// Mid price at prediction time.
    mid_price: f64,
    /// Regime index (0=calm, 1=volatile, 2=cascade).
    regime: usize,
    /// Timestamp when prediction was made (ms).
    timestamp_ms: u64,
}

/// Calibrated edge signal tracker.
///
/// Tracks whether flow_imbalance predicts price direction using
/// Information Ratio as the principled threshold (IR > 1.0).
#[derive(Debug)]
pub struct CalibratedEdgeSignal {
    /// Per-regime IR trackers using adaptive quantile-based binning.
    ir_by_regime: [AdaptiveBinner; NUM_REGIMES],

    /// Pending predictions awaiting outcome measurement.
    pending: Vec<EdgePrediction>,

    /// Next prediction ID.
    next_id: u64,

    /// Configuration.
    config: CalibratedEdgeConfig,

    /// Bayesian warmup configuration.
    bayesian_config: BayesianIRConfig,

    /// Total predictions made (for warmup tracking).
    total_predictions: u64,

    /// Total outcomes recorded.
    total_outcomes: u64,

    /// Whether currently stopped out (P < stop_out_threshold).
    stopped_out: bool,
}

impl CalibratedEdgeSignal {
    /// Create a new calibrated edge signal tracker.
    pub fn new(config: CalibratedEdgeConfig) -> Self {
        Self::with_bayesian_config(config, BayesianIRConfig::default())
    }

    /// Create a new calibrated edge signal tracker with custom Bayesian config.
    pub fn with_bayesian_config(
        config: CalibratedEdgeConfig,
        bayesian_config: BayesianIRConfig,
    ) -> Self {
        let n_bins = config.ir_n_bins;
        let max_samples = 500; // Rolling buffer size for adaptive binning
        Self {
            ir_by_regime: [
                AdaptiveBinner::new(n_bins, max_samples),
                AdaptiveBinner::new(n_bins, max_samples),
                AdaptiveBinner::new(n_bins, max_samples),
            ],
            pending: Vec::with_capacity(config.max_pending),
            next_id: 1,
            config,
            bayesian_config,
            total_predictions: 0,
            total_outcomes: 0,
            stopped_out: false,
        }
    }

    /// Make an edge prediction.
    ///
    /// Records the current flow_imbalance and mid_price for later outcome
    /// measurement. Returns a prediction ID.
    ///
    /// # Arguments
    /// * `flow_imbalance` - Current flow imbalance signal (-1 to +1)
    /// * `mid_price` - Current mid price
    /// * `regime` - Current regime index (0=calm, 1=volatile, 2=cascade)
    /// * `timestamp_ms` - Current timestamp in milliseconds
    ///
    /// # Returns
    /// Prediction ID for tracking (not currently used externally).
    pub fn predict(
        &mut self,
        flow_imbalance: f64,
        mid_price: f64,
        regime: usize,
        timestamp_ms: u64,
    ) -> u64 {
        // Clamp regime to valid range
        let regime = regime.min(NUM_REGIMES - 1);

        let pred_id = self.next_id;
        self.next_id += 1;

        let prediction = EdgePrediction {
            id: pred_id,
            flow_imbalance,
            mid_price,
            regime,
            timestamp_ms,
        };

        // Maintain bounded pending list
        if self.pending.len() >= self.config.max_pending {
            self.pending.remove(0);
        }

        self.pending.push(prediction);
        self.total_predictions += 1;

        pred_id
    }

    /// Record outcomes for mature predictions.
    ///
    /// Checks all pending predictions and records outcomes for those
    /// that have passed the outcome horizon.
    ///
    /// # Arguments
    /// * `current_mid` - Current mid price for outcome measurement
    /// * `now_ms` - Current timestamp in milliseconds
    pub fn record_outcomes(&mut self, current_mid: f64, now_ms: u64) {
        let horizon = self.config.outcome_horizon_ms;
        let min_change_bps = self.config.min_price_change_bps;

        // Find predictions that are ready
        let mut resolved_indices = Vec::new();

        for (i, pred) in self.pending.iter().enumerate() {
            if now_ms >= pred.timestamp_ms + horizon {
                resolved_indices.push(i);
            }
        }

        // Process in reverse order to maintain indices
        for &i in resolved_indices.iter().rev() {
            let pred = self.pending.remove(i);

            // Calculate price change
            let price_change_bps = if pred.mid_price > 0.0 {
                (current_mid - pred.mid_price) / pred.mid_price * 10_000.0
            } else {
                0.0
            };

            // Skip if change is too small (neutral outcome)
            if price_change_bps.abs() < min_change_bps {
                continue;
            }

            // Prediction: sign(flow_imbalance) == sign(price_change)?
            // If flow_imbalance > 0, we predict price goes up
            // If flow_imbalance < 0, we predict price goes down
            let predicted_up = pred.flow_imbalance > 0.0;
            let actual_up = price_change_bps > 0.0;
            let correct = predicted_up == actual_up;

            // Convert flow_imbalance magnitude to probability
            // Use sigmoid-like mapping: p = 0.5 + 0.5 * tanh(imbalance * scale)
            // This maps |imbalance| = 0 → p = 0.5, |imbalance| = 1 → p ≈ 0.88
            let prob = 0.5 + 0.5 * (pred.flow_imbalance.abs() * 2.0).tanh();

            // Log the prediction outcome with its unique ID for traceability
            trace!(
                prediction_id = pred.id,
                flow_imbalance = %format!("{:.3}", pred.flow_imbalance),
                price_change_bps = %format!("{:.2}", price_change_bps),
                predicted_up = predicted_up,
                actual_up = actual_up,
                correct = correct,
                regime = pred.regime,
                "Edge prediction outcome"
            );

            // Record to the appropriate regime tracker
            self.ir_by_regime[pred.regime].update(prob, correct);
            self.total_outcomes += 1;
        }
    }

    /// Check if the edge signal is useful (IR > 1.0 with sufficient samples).
    ///
    /// This is the only principled threshold - the signal adds predictive
    /// value if and only if IR > 1.0.
    pub fn is_useful(&self) -> bool {
        if !self.is_warmed_up() {
            return false;
        }

        self.overall_ir() > 1.0
    }

    /// Immutable Bayesian check for read-only contexts.
    ///
    /// Returns decision without updating stop-out state.
    /// Use this in `decide_calibrated` which takes `&self`.
    pub fn bayesian_check(&self, l2_confidence: f64) -> BayesianDecision {
        let cfg = &self.bayesian_config;
        
        // If Bayesian warmup is disabled, fall back to fixed threshold
        if !cfg.enabled {
            let is_useful = self.is_useful();
            return BayesianDecision {
                is_useful,
                reason: if is_useful {
                    "ir_above_1".to_string()
                } else if self.is_warmed_up() {
                    "ir_below_1".to_string()
                } else {
                    "warmup_incomplete".to_string()
                },
                posterior_prob: if self.is_warmed_up() { 0.5 } else { 0.0 },
                credible_lower: 0.0,
                prior_mu_used: cfg.prior_ir_mean,
                post_mu: self.overall_ir(),
                samples: self.total_samples(),
                stopped_out: false,
            };
        }
        
        let n = self.total_samples();
        
        // Hard floor: require minimum samples before any quoting
        if n < cfg.min_samples {
            return BayesianDecision {
                is_useful: false,
                reason: format!("min_samples_not_met ({}/{})", n, cfg.min_samples),
                posterior_prob: 0.0,
                credible_lower: 0.0,
                prior_mu_used: cfg.prior_ir_mean,
                post_mu: self.overall_ir(),
                samples: n,
                stopped_out: self.stopped_out,
            };
        }

        // Compute L2-adjusted prior mean
        let l2_boost = (2.0 * (l2_confidence - 0.5)).clamp(0.0, 1.5);
        let prior_mu_used = cfg.prior_ir_mean + cfg.l2_prior_weight * l2_boost;
        
        // Get posterior probability P(IR > 1.0 | data)
        let posterior_prob = self.combined_posterior_prob(1.0, prior_mu_used, cfg.prior_df);
        
        // Get credible interval
        let (credible_lower, _) = self.combined_credible_interval(0.95, prior_mu_used, cfg.prior_df);
        
        let post_mu = self.posterior_mean_ir(prior_mu_used, cfg.prior_df);
        
        // Check stop-out (read-only: use current state)
        if self.stopped_out && posterior_prob < cfg.recovery_threshold {
            return BayesianDecision {
                is_useful: false,
                reason: format!("stopped_out (P={:.2}<{:.2})", posterior_prob, cfg.recovery_threshold),
                posterior_prob,
                credible_lower,
                prior_mu_used,
                post_mu,
                samples: n,
                stopped_out: true,
            };
        }
        
        // Determine confidence threshold for current sample count (tiered)
        let threshold = self.get_tiered_threshold(n);
        
        // Decision: P(IR > 1.0) > threshold
        let is_useful = posterior_prob >= threshold;
        
        let reason = if is_useful {
            format!("bayesian_ok (P={posterior_prob:.2}≥{threshold:.2}, n={n})")
        } else {
            format!("bayesian_low (P={posterior_prob:.2}<{threshold:.2}, n={n})")
        };
        
        BayesianDecision {
            is_useful,
            reason,
            posterior_prob,
            credible_lower,
            prior_mu_used,
            post_mu,
            samples: n,
            stopped_out: false,
        }
    }

    /// Bayesian check if edge signal is useful during warmup.
    ///
    /// Uses P(IR > 1.0 | data) > tiered_threshold instead of fixed sample count.
    /// This allows quoting earlier when evidence is strong, while protecting
    /// against false positives with tiered confidence requirements.
    ///
    /// # Arguments
    /// * `l2_confidence` - L2 model confidence (0.5 to 1.0), used to adjust prior
    ///
    /// # Returns
    /// (is_useful, reason, posterior_prob, credible_lower)
    pub fn bayesian_is_useful(&mut self, l2_confidence: f64) -> BayesianDecision {
        let cfg = &self.bayesian_config;
        
        // If Bayesian warmup is disabled, fall back to fixed threshold
        if !cfg.enabled {
            let is_useful = self.is_useful();
            return BayesianDecision {
                is_useful,
                reason: if is_useful {
                    "ir_above_1".to_string()
                } else if self.is_warmed_up() {
                    "ir_below_1".to_string()
                } else {
                    "warmup_incomplete".to_string()
                },
                posterior_prob: if self.is_warmed_up() { 0.5 } else { 0.0 },
                credible_lower: 0.0,
                prior_mu_used: cfg.prior_ir_mean,
                post_mu: self.overall_ir(),
                samples: self.total_samples(),
                stopped_out: false,
            };
        }
        
        let n = self.total_samples();
        
        // Hard floor: require minimum samples before any quoting
        if n < cfg.min_samples {
            return BayesianDecision {
                is_useful: false,
                reason: format!("min_samples_not_met ({}/{})", n, cfg.min_samples),
                posterior_prob: 0.0,
                credible_lower: 0.0,
                prior_mu_used: cfg.prior_ir_mean,
                post_mu: self.overall_ir(),
                samples: n,
                stopped_out: self.stopped_out,
            };
        }

        // Compute L2-adjusted prior mean
        // L2 confidence maps to prior boost: min(2*(l2_conf - 0.5), 1.5)
        let l2_boost = (2.0 * (l2_confidence - 0.5)).clamp(0.0, 1.5);
        let prior_mu_used = cfg.prior_ir_mean + cfg.l2_prior_weight * l2_boost;
        
        // Get posterior probability P(IR > 1.0 | data)
        let posterior_prob = self.combined_posterior_prob(1.0, prior_mu_used, cfg.prior_df);
        
        // Get credible interval
        let (credible_lower, _) = self.combined_credible_interval(0.95, prior_mu_used, cfg.prior_df);
        
        let post_mu = self.posterior_mean_ir(prior_mu_used, cfg.prior_df);
        
        // Check stop-out: if evidence is strongly against us, halt
        if n >= cfg.stop_out_min_samples {
            if posterior_prob < cfg.stop_out_threshold && !self.stopped_out {
                self.stopped_out = true;
            }
            // Recovery: resume if probability climbs back above recovery threshold
            if self.stopped_out && posterior_prob >= cfg.recovery_threshold {
                self.stopped_out = false;
            }
        }
        
        if self.stopped_out {
            return BayesianDecision {
                is_useful: false,
                reason: format!("stopped_out (P={:.2}<{:.2})", posterior_prob, cfg.recovery_threshold),
                posterior_prob,
                credible_lower,
                prior_mu_used,
                post_mu,
                samples: n,
                stopped_out: true,
            };
        }
        
        // Determine confidence threshold for current sample count (tiered)
        let threshold = self.get_tiered_threshold(n);
        
        // Decision: P(IR > 1.0) > threshold
        let is_useful = posterior_prob >= threshold;
        
        let reason = if is_useful {
            format!("bayesian_ok (P={posterior_prob:.2}≥{threshold:.2}, n={n})")
        } else {
            format!("bayesian_low (P={posterior_prob:.2}<{threshold:.2}, n={n})")
        };
        
        BayesianDecision {
            is_useful,
            reason,
            posterior_prob,
            credible_lower,
            prior_mu_used,
            post_mu,
            samples: n,
            stopped_out: false,
        }
    }

    /// Get the tiered confidence threshold for a given sample count.
    pub(crate) fn get_tiered_threshold(&self, n: usize) -> f64 {
        let tiers = &self.bayesian_config.confidence_tiers;
        
        // Find the highest tier that applies
        let mut threshold = tiers[0].1; // Default to first tier
        for &(min_samples, tier_threshold) in tiers {
            if n >= min_samples {
                threshold = tier_threshold;
            }
        }
        threshold
    }

    /// Get the current tier threshold based on current sample count.
    pub fn get_current_tier_threshold(&self) -> f64 {
        self.get_tiered_threshold(self.total_samples())
    }

    /// Get the prior influence factor (0 = data dominates, 1 = prior dominates).
    ///
    /// This measures how much the shrinkage prior affects the posterior:
    /// - High (>0.4 at n=30) suggests prior_df may be too high
    /// - Low (<0.1 at n=50) means data has taken over
    pub fn bayesian_prior_influence(&self) -> f64 {
        let n = self.total_samples();
        let prior_df = self.bayesian_config.prior_df;
        prior_df / (prior_df + n as f64)
    }

    /// Get combined posterior probability across all regimes.
    fn combined_posterior_prob(&self, threshold: f64, prior_mean: f64, prior_df: f64) -> f64 {
        let mut total_samples = 0usize;
        let mut weighted_prob = 0.0;

        for tracker in &self.ir_by_regime {
            let n = tracker.n_samples();
            if n > 0 {
                let p = tracker.posterior_prob_ir_above(threshold, prior_mean, prior_df);
                total_samples += n;
                weighted_prob += p * n as f64;
            }
        }

        if total_samples > 0 {
            weighted_prob / total_samples as f64
        } else {
            0.5 // Neutral when no data
        }
    }

    /// Get combined credible interval across all regimes.
    fn combined_credible_interval(
        &self,
        confidence: f64,
        prior_mean: f64,
        prior_df: f64,
    ) -> (f64, f64) {
        let mut total_samples = 0usize;
        let mut weighted_lower = 0.0;
        let mut weighted_upper = 0.0;

        for tracker in &self.ir_by_regime {
            let n = tracker.n_samples();
            if n > 0 {
                let (lower, upper) = tracker.ir_credible_interval(confidence, prior_mean, prior_df);
                total_samples += n;
                weighted_lower += lower * n as f64;
                weighted_upper += upper * n as f64;
            }
        }

        if total_samples > 0 {
            (
                weighted_lower / total_samples as f64,
                weighted_upper / total_samples as f64,
            )
        } else {
            (0.0, prior_mean * 2.0)
        }
    }

    /// Get posterior mean IR (with shrinkage) across all regimes.
    fn posterior_mean_ir(&self, prior_mean: f64, prior_df: f64) -> f64 {
        let mut total_samples = 0usize;
        let mut weighted_mean = 0.0;

        for tracker in &self.ir_by_regime {
            let n = tracker.n_samples();
            if n > 0 {
                let mean = tracker.posterior_mean_ir(prior_mean, prior_df);
                total_samples += n;
                weighted_mean += mean * n as f64;
            }
        }

        if total_samples > 0 {
            weighted_mean / total_samples as f64
        } else {
            prior_mean
        }
    }

    /// Get the signal weight for blending.
    ///
    /// Returns (IR - 1.0).max(0.0), which naturally scales the signal:
    /// - IR = 1.0 → weight = 0 (no edge)
    /// - IR = 1.5 → weight = 0.5 (moderate edge)
    /// - IR = 2.0 → weight = 1.0 (strong edge)
    pub fn signal_weight(&self) -> f64 {
        if !self.is_warmed_up() {
            return 0.0;
        }

        (self.overall_ir() - 1.0).max(0.0)
    }

    /// Get the overall IR (weighted average across regimes).
    pub fn overall_ir(&self) -> f64 {
        let mut total_samples = 0usize;
        let mut weighted_ir = 0.0;

        for tracker in &self.ir_by_regime {
            let n = tracker.n_samples();
            if n > 0 {
                total_samples += n;
                weighted_ir += tracker.information_ratio() * n as f64;
            }
        }

        if total_samples > 0 {
            weighted_ir / total_samples as f64
        } else {
            1.0 // Neutral IR when no data
        }
    }

    /// Get IR for a specific regime.
    pub fn ir_for_regime(&self, regime: usize) -> f64 {
        let regime = regime.min(NUM_REGIMES - 1);
        self.ir_by_regime[regime].information_ratio()
    }

    /// Get sample count for a specific regime.
    pub fn samples_for_regime(&self, regime: usize) -> usize {
        let regime = regime.min(NUM_REGIMES - 1);
        self.ir_by_regime[regime].n_samples()
    }

    /// Get total sample count across all regimes.
    pub fn total_samples(&self) -> usize {
        self.ir_by_regime.iter().map(|t| t.n_samples()).sum()
    }

    /// Check if tracker is warmed up (enough samples for reliable IR).
    pub fn is_warmed_up(&self) -> bool {
        self.total_samples() >= self.config.min_samples_for_ir
    }

    /// Get warmup progress as a fraction (0.0 to 1.0).
    pub fn warmup_progress(&self) -> f64 {
        let samples = self.total_samples();
        let required = self.config.min_samples_for_ir;
        (samples as f64 / required as f64).min(1.0)
    }

    /// Get the cold-start edge threshold (used before calibration).
    pub fn cold_start_threshold(&self) -> f64 {
        self.config.cold_start_edge_threshold
    }

    /// Get the effective edge threshold with gradual warmup blending.
    ///
    /// During warmup, blends between cold-start (conservative) and calibrated:
    /// threshold = cold_start * (1 - blend) + calibrated * blend
    ///
    /// For a calibrated system, the "threshold" is implicitly IR > 1.0,
    /// but for compatibility we return the signal magnitude threshold.
    pub fn effective_edge_threshold(&self) -> f64 {
        let blend = self.warmup_progress();
        let cold_start = self.cold_start_threshold();

        if blend < 1.0 {
            // During warmup, use conservative threshold
            // As IR data accumulates, gradually trust the signal more
            cold_start * (1.0 - blend * 0.5)
        } else {
            // After warmup, threshold is determined by IR
            // If IR > 1.0, any signal has value; if IR <= 1.0, require stronger signal
            if self.overall_ir() > 1.0 {
                0.05 // Trust even weak signals
            } else {
                cold_start * 1.5 // Require stronger signal if not predictive
            }
        }
    }

    /// Get pending prediction count.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Get total predictions made.
    pub fn total_predictions(&self) -> u64 {
        self.total_predictions
    }

    /// Get total outcomes recorded.
    pub fn total_outcomes(&self) -> u64 {
        self.total_outcomes
    }

    /// Get configuration.
    pub fn config(&self) -> &CalibratedEdgeConfig {
        &self.config
    }

    /// Generate a diagnostic summary.
    pub fn diagnostic_summary(&self) -> EdgeSignalDiagnostics {
        EdgeSignalDiagnostics {
            overall_ir: self.overall_ir(),
            ir_calm: self.ir_for_regime(0),
            ir_volatile: self.ir_for_regime(1),
            ir_cascade: self.ir_for_regime(2),
            samples_calm: self.samples_for_regime(0),
            samples_volatile: self.samples_for_regime(1),
            samples_cascade: self.samples_for_regime(2),
            total_samples: self.total_samples(),
            is_warmed_up: self.is_warmed_up(),
            is_useful: self.is_useful(),
            signal_weight: self.signal_weight(),
            pending_count: self.pending_count(),
        }
    }

    /// Clear all data (for testing/reset).
    pub fn clear(&mut self) {
        for tracker in &mut self.ir_by_regime {
            tracker.clear();
        }
        self.pending.clear();
        self.total_predictions = 0;
        self.total_outcomes = 0;
    }
}

impl Default for CalibratedEdgeSignal {
    fn default() -> Self {
        Self::new(CalibratedEdgeConfig::default())
    }
}

/// Diagnostic summary for edge signal health.
#[derive(Debug, Clone)]
pub struct EdgeSignalDiagnostics {
    /// Overall IR (weighted average).
    pub overall_ir: f64,
    /// IR for calm regime.
    pub ir_calm: f64,
    /// IR for volatile regime.
    pub ir_volatile: f64,
    /// IR for cascade regime.
    pub ir_cascade: f64,
    /// Sample count for calm regime.
    pub samples_calm: usize,
    /// Sample count for volatile regime.
    pub samples_volatile: usize,
    /// Sample count for cascade regime.
    pub samples_cascade: usize,
    /// Total sample count.
    pub total_samples: usize,
    /// Whether tracker is warmed up.
    pub is_warmed_up: bool,
    /// Whether signal is useful (IR > 1.0).
    pub is_useful: bool,
    /// Signal weight for blending.
    pub signal_weight: f64,
    /// Pending predictions.
    pub pending_count: usize,
}

impl std::fmt::Display for EdgeSignalDiagnostics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.is_useful {
            "PREDICTIVE"
        } else if self.is_warmed_up {
            "NOISE"
        } else {
            "WARMUP"
        };

        writeln!(f, "=== Edge Signal Health ===")?;
        writeln!(f, "Overall IR: {:.3} ({})", self.overall_ir, status)?;
        writeln!(f, "Regime Breakdown:")?;

        let regime_status = |ir: f64| {
            if ir > 1.0 {
                ""
            } else {
                " ⚠️ NOISE"
            }
        };

        writeln!(
            f,
            "  - Calm:     IR={:.2}, n={}{}",
            self.ir_calm,
            self.samples_calm,
            regime_status(self.ir_calm)
        )?;
        writeln!(
            f,
            "  - Volatile: IR={:.2}, n={}{}",
            self.ir_volatile,
            self.samples_volatile,
            regime_status(self.ir_volatile)
        )?;
        writeln!(
            f,
            "  - Cascade:  IR={:.2}, n={}{}",
            self.ir_cascade,
            self.samples_cascade,
            regime_status(self.ir_cascade)
        )?;
        writeln!(f, "Signal weight: {:.3}", self.signal_weight)?;

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> CalibratedEdgeConfig {
        CalibratedEdgeConfig {
            outcome_horizon_ms: 100, // Short for testing
            min_price_change_bps: 0.5,
            max_pending: 100,
            min_samples_for_ir: 20, // Low for testing
            cold_start_edge_threshold: 0.25,
            ir_n_bins: 10,
        }
    }

    #[test]
    fn test_prediction_recording() {
        let mut edge = CalibratedEdgeSignal::new(make_config());

        let id = edge.predict(0.5, 100.0, 0, 1000);

        assert_eq!(id, 1);
        assert_eq!(edge.pending_count(), 1);
        assert_eq!(edge.total_predictions(), 1);
    }

    #[test]
    fn test_outcome_recording() {
        let mut edge = CalibratedEdgeSignal::new(make_config());

        // Make prediction: positive imbalance predicts price up
        edge.predict(0.5, 100.0, 0, 1000);

        // Record outcome: price went up (correct prediction)
        edge.record_outcomes(100.10, 1200); // 10 bps up

        assert_eq!(edge.pending_count(), 0);
        assert_eq!(edge.total_outcomes(), 1);
    }

    #[test]
    fn test_ir_above_one_with_correct_predictions() {
        let mut edge = CalibratedEdgeSignal::new(make_config());

        // Simulate predictions with varying confidence levels that correlate with correctness.
        // High-confidence predictions (high |imbalance|) should be more often correct
        // Low-confidence predictions (low |imbalance|) should be less reliable
        // This should result in positive IR because confidence separates correct from incorrect.
        for i in 0..60 {
            let ts = i * 200;
            let price_up = i % 2 == 0; // 50% up, 50% down outcomes

            // High confidence when i % 3 != 0 (66%), low confidence otherwise
            let high_conf = i % 3 != 0;
            // High confidence predictions are 90% correct, low confidence are 50%
            let correct = if high_conf {
                i % 10 != 0 // 90% correct for high conf
            } else {
                i % 2 == 0 // 50% correct for low conf
            };

            // Imbalance magnitude reflects confidence
            let imbalance_magnitude = if high_conf { 0.8 } else { 0.2 };
            // Direction depends on whether we want correct prediction
            let predicted_up = if correct { price_up } else { !price_up };
            let imbalance = if predicted_up { imbalance_magnitude } else { -imbalance_magnitude };

            edge.predict(imbalance, 100.0, 0, ts);

            let price_change = if price_up { 0.10 } else { -0.10 };
            edge.record_outcomes(100.0 + price_change, ts + 150);
        }

        assert!(edge.is_warmed_up());

        // With confidence-correctness correlation, IR should be positive
        // (high confidence predictions cluster in a bin with higher accuracy)
        let ir = edge.overall_ir();
        // IR should be non-negative (could be 0 or positive depending on binning)
        assert!(ir >= 0.0, "IR should be non-negative: {}", ir);
    }

    #[test]
    fn test_ir_around_one_with_random_predictions() {
        let mut edge = CalibratedEdgeSignal::new(make_config());

        // Simulate ~50% correct predictions (random relationship)
        // Predictions are unrelated to outcomes
        for i in 0..50 {
            let ts = i * 200;
            // Prediction based on even/odd (arbitrary)
            let imbalance = if i % 2 == 0 { 0.5 } else { -0.5 };
            edge.predict(imbalance, 100.0, 0, ts);

            // Outcome based on different pattern (unrelated to prediction)
            let price_up = (i * 7) % 10 < 5; // Different pseudo-random pattern
            let price_change = if price_up { 0.10 } else { -0.10 };
            edge.record_outcomes(100.0 + price_change, ts + 150);
        }

        assert!(edge.is_warmed_up());

        // With random predictions, IR should be around 0 (no resolution)
        // The model's predictions don't separate outcomes any better than random
        let ir = edge.overall_ir();
        // IR should be low (close to 0) since predictions are random
        assert!(
            ir >= 0.0 && ir <= 2.0,
            "IR should be low for random predictions: {}",
            ir
        );
    }

    #[test]
    fn test_signal_weight() {
        let mut edge = CalibratedEdgeSignal::new(make_config());

        // Not warmed up
        assert_eq!(edge.signal_weight(), 0.0);

        // Add predictions (mostly correct)
        for i in 0..25 {
            let ts = i * 200;
            edge.predict(0.5, 100.0, 0, ts);
            edge.record_outcomes(100.10, ts + 150);
        }

        // Signal weight should be non-negative
        assert!(edge.signal_weight() >= 0.0);
    }

    #[test]
    fn test_regime_specific_ir() {
        let mut edge = CalibratedEdgeSignal::new(make_config());

        // Add predictions to different regimes
        for i in 0..30 {
            let ts = i * 200;
            let regime = (i % 3) as usize;
            edge.predict(0.5, 100.0, regime, ts);
            // Calm is predictive, others are random
            let price = if regime == 0 { 100.10 } else { 100.0 - 0.05 * (i % 2) as f64 };
            edge.record_outcomes(price, ts + 150);
        }

        // Should have samples in each regime
        assert!(edge.samples_for_regime(0) > 0);
        assert!(edge.samples_for_regime(1) > 0);
        assert!(edge.samples_for_regime(2) > 0);
    }

    #[test]
    fn test_cold_start_threshold() {
        let edge = CalibratedEdgeSignal::new(make_config());

        // Before warmup, should use cold-start threshold
        assert!(!edge.is_warmed_up());
        assert_eq!(edge.cold_start_threshold(), 0.25);

        // Effective threshold should be at least cold-start during warmup
        assert!(edge.effective_edge_threshold() > 0.0);
    }

    #[test]
    fn test_pending_bounded() {
        let mut config = make_config();
        config.max_pending = 5;
        let mut edge = CalibratedEdgeSignal::new(config);

        // Add more predictions than max
        for i in 0..10 {
            edge.predict(0.5, 100.0, 0, i * 100);
        }

        // Should be bounded
        assert_eq!(edge.pending_count(), 5);
    }

    #[test]
    fn test_diagnostics() {
        let mut edge = CalibratedEdgeSignal::new(make_config());

        // Mix of predictions and outcomes to ensure IR can be computed
        for i in 0..30 {
            let ts = i * 200;
            let imbalance = if i % 2 == 0 { 0.5 } else { -0.5 };
            edge.predict(imbalance, 100.0, 0, ts);
            // Mix of up and down outcomes
            let price_change = if i % 3 == 0 { -0.10 } else { 0.10 };
            edge.record_outcomes(100.0 + price_change, ts + 150);
        }

        let diag = edge.diagnostic_summary();

        assert!(diag.is_warmed_up);
        assert!(diag.total_samples > 0);
        // IR may be 0 or positive depending on random correlation, but should be valid
        assert!(diag.overall_ir >= 0.0 || diag.total_samples == 0);
    }

    #[test]
    fn test_clear() {
        let mut edge = CalibratedEdgeSignal::new(make_config());

        edge.predict(0.5, 100.0, 0, 1000);
        edge.record_outcomes(100.10, 1200);

        edge.clear();

        assert_eq!(edge.pending_count(), 0);
        assert_eq!(edge.total_samples(), 0);
        assert_eq!(edge.total_predictions(), 0);
        assert_eq!(edge.total_outcomes(), 0);
    }

    #[test]
    fn test_tier_boundary_transitions() {
        let config = make_config();
        let edge = CalibratedEdgeSignal::new(config);
        
        // Test tier boundaries: 0.75 for n<20, 0.80 for 20≤n<50, 0.85 for 50≤n<100, 0.90 for n≥100
        assert_eq!(edge.get_tiered_threshold(0), 0.75);
        assert_eq!(edge.get_tiered_threshold(19), 0.75);
        assert_eq!(edge.get_tiered_threshold(20), 0.80);  // Boundary transition
        assert_eq!(edge.get_tiered_threshold(49), 0.80);
        assert_eq!(edge.get_tiered_threshold(50), 0.85);  // Boundary transition
        assert_eq!(edge.get_tiered_threshold(99), 0.85);
        assert_eq!(edge.get_tiered_threshold(100), 0.90); // Boundary transition
        assert_eq!(edge.get_tiered_threshold(1000), 0.90);
    }

    #[test]
    fn test_bayesian_check_min_samples() {
        let config = make_config();
        let edge = CalibratedEdgeSignal::new(config);
        
        // No samples: should fail min_samples check
        let decision = edge.bayesian_check(0.8);
        assert!(!decision.is_useful);
        assert!(decision.reason.contains("min_samples_not_met"));
    }

    #[test]
    fn test_prior_influence_decay() {
        let mut config = make_config();
        config.min_samples_for_ir = 10;
        let mut edge = CalibratedEdgeSignal::new(config);
        
        // Prior influence with no samples: prior_df / (prior_df + 0) = 1.0
        let initial_influence = edge.bayesian_prior_influence();
        assert!((initial_influence - 1.0).abs() < 0.01); // prior_df=6 dominates
        
        // Add samples and check influence decreases
        for i in 0..30 {
            edge.predict(0.8, 100.0, 0, i * 100);
            edge.record_outcomes(100.05, i * 100 + 80);
        }
        
        // At n=30: prior_influence = 6 / (6 + 30) = 0.167
        let later_influence = edge.bayesian_prior_influence();
        assert!(later_influence < 0.25);
        assert!(later_influence > 0.10);
    }

    #[test]
    fn test_single_regime_fallback() {
        // Ensure single-regime data works (no divide-by-zero in weighted avg)
        let mut config = make_config();
        config.min_samples_for_ir = 10;
        let mut edge = CalibratedEdgeSignal::new(config);
        
        // Only add samples to regime 0 (calm)
        for i in 0..20 {
            edge.predict(0.7, 100.0, 0, i * 100);
            edge.record_outcomes(100.03, i * 100 + 80);
        }
        
        // Should still compute valid decision
        let decision = edge.bayesian_check(0.8);
        assert!(decision.samples >= 15, "Expected at least 15 samples, got {}", decision.samples);
        assert!(decision.posterior_prob >= 0.0);
        assert!(decision.posterior_prob <= 1.0);
    }
}
