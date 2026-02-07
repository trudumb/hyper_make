//! Prediction Logging Infrastructure
//!
//! Records all model predictions and their outcomes for calibration analysis.
//! Every quote cycle produces a PredictionRecord that captures:
//! - Market state at prediction time
//! - Model outputs (fill probabilities, adverse selection, etc.)
//! - Actual outcomes (filled in asynchronously)

use super::fill_sim::FillSimulator;
use crate::{Ladder, LadderLevel, MarketParams, Side};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Regime classification for conditional analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Regime {
    Quiet,
    Active,
    Volatile,
    Cascade,
}

impl From<&MarketParams> for Regime {
    fn from(params: &MarketParams) -> Self {
        if params.should_pull_quotes || params.cascade_size_factor < 0.5 {
            Regime::Cascade
        } else if params.is_toxic_regime || params.jump_ratio > 2.0 {
            Regime::Volatile
        } else if params.arrival_intensity > 2.0 || params.sigma > 0.0002 {
            Regime::Active
        } else {
            Regime::Quiet
        }
    }
}

/// Complete prediction record for a single quote cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRecord {
    /// Timestamp in nanoseconds since epoch
    pub timestamp_ns: u64,
    /// Unique quote cycle identifier
    pub quote_cycle_id: u64,
    /// Market state at prediction time
    pub market_state: MarketStateSnapshot,
    /// Model predictions
    pub predictions: ModelPredictions,
    /// Observed outcomes (filled in async after some time)
    pub outcomes: Option<ObservedOutcomes>,
}

/// Snapshot of market state at prediction time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketStateSnapshot {
    // === L2 Book State ===
    /// Best bid price
    pub best_bid: f64,
    /// Best ask price
    pub best_ask: f64,
    /// Bid levels (price, size) for top N levels
    pub bid_levels: Vec<(f64, f64)>,
    /// Ask levels (price, size) for top N levels
    pub ask_levels: Vec<(f64, f64)>,

    // === Derived Quantities ===
    /// Microprice (inventory-weighted mid)
    pub microprice: f64,
    /// Market mid price
    pub market_mid: f64,
    /// Book imbalance [-1, 1]
    pub book_imbalance: f64,
    /// Market spread in bps
    pub market_spread_bps: f64,

    // === Kappa (Fill Intensity) ===
    /// Book-derived kappa
    pub kappa_book: f64,
    /// Robust kappa (outlier-resistant)
    pub kappa_robust: f64,
    /// Adaptive kappa (blended)
    pub kappa_adaptive: f64,
    /// Final kappa used for GLFT
    pub kappa_final: f64,
    /// Kappa uncertainty (posterior std dev)
    pub kappa_uncertainty: f64,
    /// Kappa credible interval width
    pub kappa_ci_width: f64,

    // === Volatility ===
    /// Clean volatility (bipower variation)
    pub sigma_clean: f64,
    /// Total volatility (realized)
    pub sigma_total: f64,
    /// Effective volatility (regime-blended)
    pub sigma_effective: f64,
    /// Jump ratio (RV/BV)
    pub jump_ratio: f64,

    // === Gamma (Risk Aversion) ===
    /// Base gamma from config
    pub gamma_base: f64,
    /// Adaptive gamma (Bayesian)
    pub gamma_adaptive: f64,
    /// Calibration gamma multiplier
    pub gamma_calibration_mult: f64,
    /// Effective gamma used
    pub gamma_effective: f64,

    // === External State ===
    /// Funding rate (annualized)
    pub funding_rate: f64,
    /// Time to funding settlement (seconds)
    pub time_to_funding_settlement_s: f64,
    /// Open interest
    pub open_interest: f64,
    /// OI change in last minute
    pub oi_change_1m: f64,

    // === Position State ===
    /// Current inventory (signed)
    pub inventory: f64,
    /// Pending bid exposure
    pub pending_bid_exposure: f64,
    /// Pending ask exposure
    pub pending_ask_exposure: f64,
    /// Dynamic max position limit
    pub dynamic_max_position: f64,

    // === Regime Classification ===
    /// Current regime
    pub regime: Regime,
    /// Toxicity score [0, 1]
    pub toxicity_score: f64,
    /// Is in toxic regime
    pub is_toxic_regime: bool,

    // === Flow State ===
    /// Momentum in bps
    pub momentum_bps: f64,
    /// Flow imbalance [-1, 1]
    pub flow_imbalance: f64,
    /// Falling knife score [0, 3]
    pub falling_knife_score: f64,
    /// Rising knife score [0, 3]
    pub rising_knife_score: f64,

    // === Adaptive System State ===
    /// Adaptive spread floor
    pub adaptive_spread_floor: f64,
    /// Adaptive spread ceiling
    pub adaptive_spread_ceiling: f64,
    /// Adaptive warmup progress [0, 1]
    pub adaptive_warmup_progress: f64,
    /// Calibration progress [0, 1]
    pub calibration_progress: f64,
}

impl MarketStateSnapshot {
    /// Create snapshot from MarketParams and current state
    pub fn from_params(
        params: &MarketParams,
        bid_levels: Vec<(f64, f64)>,
        ask_levels: Vec<(f64, f64)>,
        inventory: f64,
        gamma_base: f64,
    ) -> Self {
        let best_bid = bid_levels.first().map(|(p, _)| *p).unwrap_or(0.0);
        let best_ask = ask_levels.first().map(|(p, _)| *p).unwrap_or(0.0);

        Self {
            best_bid,
            best_ask,
            bid_levels,
            ask_levels,
            microprice: params.microprice,
            market_mid: params.market_mid,
            book_imbalance: params.book_imbalance,
            market_spread_bps: params.market_spread_bps,
            kappa_book: params.kappa,
            kappa_robust: params.kappa_robust,
            kappa_adaptive: params.adaptive_kappa,
            kappa_final: if params.use_kappa_robust {
                params.kappa_robust
            } else {
                params.kappa
            },
            kappa_uncertainty: params.kappa_uncertainty,
            kappa_ci_width: params.kappa_ci_width,
            sigma_clean: params.sigma,
            sigma_total: params.sigma_total,
            sigma_effective: params.sigma_effective,
            jump_ratio: params.jump_ratio,
            gamma_base,
            gamma_adaptive: params.adaptive_gamma,
            gamma_calibration_mult: params.calibration_gamma_mult,
            gamma_effective: gamma_base
                * params.adaptive_gamma
                * params.calibration_gamma_mult
                * params.tail_risk_multiplier,
            funding_rate: params.funding_rate,
            time_to_funding_settlement_s: params.time_to_funding_settlement_s,
            open_interest: params.open_interest,
            oi_change_1m: params.oi_change_1m,
            inventory,
            pending_bid_exposure: params.pending_bid_exposure,
            pending_ask_exposure: params.pending_ask_exposure,
            dynamic_max_position: params.dynamic_max_position,
            regime: Regime::from(params),
            toxicity_score: params.toxicity_score,
            is_toxic_regime: params.is_toxic_regime,
            momentum_bps: params.momentum_bps,
            flow_imbalance: params.flow_imbalance,
            falling_knife_score: params.falling_knife_score,
            rising_knife_score: params.rising_knife_score,
            adaptive_spread_floor: params.adaptive_spread_floor,
            adaptive_spread_ceiling: params.adaptive_spread_ceiling,
            calibration_progress: params.calibration_progress,
            adaptive_warmup_progress: params.adaptive_warmup_progress,
        }
    }
}

/// Model predictions for a single quote cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPredictions {
    /// Per-level predictions
    pub levels: Vec<LevelPrediction>,

    // === Aggregate Predictions ===
    /// Expected fill rate in next 1s (fills/second)
    pub expected_fill_rate_1s: f64,
    /// Expected fill rate in next 10s
    pub expected_fill_rate_10s: f64,
    /// Expected adverse selection in bps
    pub expected_adverse_selection_bps: f64,
    /// Regime probabilities
    pub regime_probabilities: HashMap<Regime, f64>,

    // === Quote Decision ===
    /// GLFT optimal half-spread in bps
    pub glft_half_spread_bps: f64,
    /// Final half-spread used (after floors/ceilings)
    pub final_half_spread_bps: f64,
    /// Inventory skew applied in bps
    pub inventory_skew_bps: f64,
    /// Whether quotes were pulled
    pub quotes_pulled: bool,
    /// Reason for pulling quotes
    pub pull_reason: Option<String>,
}

impl ModelPredictions {
    /// Create from ladder and market params
    pub fn from_ladder(
        ladder: &Ladder,
        params: &MarketParams,
        glft_half_spread_bps: f64,
        final_half_spread_bps: f64,
        inventory_skew_bps: f64,
        fill_sim: Option<&FillSimulator>,
    ) -> Self {
        let mut levels = Vec::new();

        // Process bid levels
        for level in &ladder.bids {
            let (queue_pos, queue_total) = fill_sim
                .map(|fs| fs.estimate_queue_position(level.price, true))
                .unwrap_or((0.0, 0.0));
            levels.push(LevelPrediction {
                side: Side::Buy,
                price: level.price,
                size: level.size,
                depth_from_mid_bps: level.depth_bps,
                p_fill_100ms: estimate_fill_probability(level.depth_bps, params, 0.1),
                p_fill_1s: estimate_fill_probability(level.depth_bps, params, 1.0),
                p_fill_10s: estimate_fill_probability(level.depth_bps, params, 10.0),
                p_adverse_given_fill: params.predicted_alpha,
                expected_pnl_given_fill: estimate_expected_pnl(level, params, Side::Buy),
                estimated_queue_position: queue_pos,
                estimated_queue_total: queue_total,
            });
        }

        // Process ask levels
        for level in &ladder.asks {
            let (queue_pos, queue_total) = fill_sim
                .map(|fs| fs.estimate_queue_position(level.price, false))
                .unwrap_or((0.0, 0.0));
            levels.push(LevelPrediction {
                side: Side::Sell,
                price: level.price,
                size: level.size,
                depth_from_mid_bps: level.depth_bps,
                p_fill_100ms: estimate_fill_probability(level.depth_bps, params, 0.1),
                p_fill_1s: estimate_fill_probability(level.depth_bps, params, 1.0),
                p_fill_10s: estimate_fill_probability(level.depth_bps, params, 10.0),
                p_adverse_given_fill: params.predicted_alpha,
                expected_pnl_given_fill: estimate_expected_pnl(level, params, Side::Sell),
                estimated_queue_position: queue_pos,
                estimated_queue_total: queue_total,
            });
        }

        // Compute aggregate predictions
        let total_fill_prob_1s: f64 = levels.iter().map(|l| l.p_fill_1s * l.size).sum();

        Self {
            levels,
            expected_fill_rate_1s: total_fill_prob_1s,
            expected_fill_rate_10s: total_fill_prob_1s * 10.0 * 0.8, // Decay factor
            expected_adverse_selection_bps: params.total_as_bps,
            regime_probabilities: compute_regime_probabilities(params),
            glft_half_spread_bps,
            final_half_spread_bps,
            inventory_skew_bps,
            quotes_pulled: params.should_pull_quotes,
            pull_reason: if params.should_pull_quotes {
                Some("Cascade detected".to_string())
            } else {
                None
            },
        }
    }
}

/// Fill probability estimation using GLFT intensity model
fn estimate_fill_probability(depth_bps: f64, params: &MarketParams, horizon_s: f64) -> f64 {
    // λ(δ) = κ × exp(-δ/δ_char) where δ_char ≈ 10 bps typically
    let delta_char_bps = 10.0;
    let kappa = if params.use_kappa_robust {
        params.kappa_robust
    } else {
        params.kappa
    };

    let intensity = kappa * (-depth_bps / delta_char_bps).exp();

    // P(fill in [0, T]) = 1 - exp(-λT)
    let fill_prob = 1.0 - (-intensity * horizon_s).exp();

    fill_prob.clamp(0.0, 0.99)
}

/// Expected PnL given fill
fn estimate_expected_pnl(level: &LadderLevel, params: &MarketParams, side: Side) -> f64 {
    // PnL = spread_capture - adverse_selection - fees
    let spread_capture_bps = level.depth_bps;
    let as_bps = params.total_as_bps;
    let fee_bps = 1.5; // Maker fee

    let gross_edge_bps = spread_capture_bps - as_bps - fee_bps;

    // Apply directional adjustment
    let directional_penalty = match side {
        Side::Buy => params.falling_knife_score * 2.0, // Extra penalty if falling
        Side::Sell => params.rising_knife_score * 2.0,
    };

    (gross_edge_bps - directional_penalty) * level.size * params.microprice / 10000.0
}

/// Compute regime probabilities from params
fn compute_regime_probabilities(params: &MarketParams) -> HashMap<Regime, f64> {
    let mut probs = HashMap::new();

    // Use HMM-style soft probabilities if available, otherwise heuristic
    let regime_probs = params.regime_probs;

    // Map 4-state regime to our enum
    probs.insert(Regime::Quiet, regime_probs[0]);
    probs.insert(Regime::Active, regime_probs[1]);
    probs.insert(Regime::Volatile, regime_probs[2]);
    probs.insert(Regime::Cascade, regime_probs[3]);

    probs
}

/// Prediction for a single quote level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelPrediction {
    /// Side of the quote
    pub side: Side,
    /// Quote price
    pub price: f64,
    /// Quote size
    pub size: f64,
    /// Distance from mid in basis points
    pub depth_from_mid_bps: f64,

    // === Fill Probability Predictions ===
    /// P(fill within 100ms)
    pub p_fill_100ms: f64,
    /// P(fill within 1s)
    pub p_fill_1s: f64,
    /// P(fill within 10s)
    pub p_fill_10s: f64,

    // === Conditional Predictions ===
    /// P(adverse selection | fill)
    pub p_adverse_given_fill: f64,
    /// E[PnL | fill] in USD
    pub expected_pnl_given_fill: f64,

    // === Queue State ===
    /// Estimated position in queue
    pub estimated_queue_position: f64,
    /// Estimated total queue size
    pub estimated_queue_total: f64,
}

/// Observed outcomes for a prediction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservedOutcomes {
    /// Fill outcomes for each level
    pub fills: Vec<FillOutcome>,

    // === Price Evolution ===
    /// Price 1s after prediction
    pub price_1s_later: f64,
    /// Price 10s after prediction
    pub price_10s_later: f64,
    /// Price 60s after prediction
    pub price_60s_later: f64,

    /// Realized adverse selection in bps
    pub adverse_selection_realized_bps: f64,

    /// Total PnL from this cycle
    pub realized_pnl: f64,
}

/// Outcome for a single fill
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillOutcome {
    /// Index into predictions.levels
    pub level_index: usize,
    /// Fill timestamp (ns since epoch)
    pub fill_timestamp_ns: u64,
    /// Actual fill price
    pub fill_price: f64,
    /// Actual fill size
    pub fill_size: f64,

    // === Post-Fill Price Evolution ===
    /// Mark price at fill time
    pub mark_price_at_fill: f64,
    /// Mark price 100ms after fill
    pub mark_price_100ms_later: f64,
    /// Mark price 1s after fill
    pub mark_price_1s_later: f64,
    /// Mark price 10s after fill
    pub mark_price_10s_later: f64,

    /// Realized adverse selection for this fill (bps)
    pub realized_as_bps: f64,
}

/// Prediction logger that writes records to disk
pub struct PredictionLogger {
    /// Output file writer
    writer: Arc<Mutex<BufWriter<File>>>,
    /// Current quote cycle ID
    cycle_id: u64,
    /// Pending records awaiting outcomes
    pending_records: Arc<Mutex<HashMap<u64, PredictionRecord>>>,
    /// Maximum age before dropping pending records (ms)
    max_pending_age_ms: u64,
}

impl PredictionLogger {
    /// Create a new prediction logger
    pub fn new(output_path: PathBuf) -> std::io::Result<Self> {
        let file = File::create(output_path)?;
        let writer = BufWriter::new(file);

        Ok(Self {
            writer: Arc::new(Mutex::new(writer)),
            cycle_id: 0,
            pending_records: Arc::new(Mutex::new(HashMap::new())),
            max_pending_age_ms: 120_000, // 2 minutes
        })
    }

    /// Log a prediction record
    pub fn log_prediction(
        &mut self,
        market_state: MarketStateSnapshot,
        predictions: ModelPredictions,
    ) -> u64 {
        self.cycle_id += 1;
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let record = PredictionRecord {
            timestamp_ns,
            quote_cycle_id: self.cycle_id,
            market_state,
            predictions,
            outcomes: None,
        };

        // Store for later outcome matching
        let mut pending = self.pending_records.lock().unwrap();
        pending.insert(self.cycle_id, record);

        // Clean up old records
        self.cleanup_old_records(&mut pending);

        self.cycle_id
    }

    /// Attach outcomes to a prediction record
    pub fn attach_outcomes(&self, cycle_id: u64, outcomes: ObservedOutcomes) {
        let mut pending = self.pending_records.lock().unwrap();

        if let Some(record) = pending.remove(&cycle_id) {
            let completed_record = PredictionRecord {
                outcomes: Some(outcomes),
                ..record
            };

            // Write completed record
            self.write_record(&completed_record);
        }
    }

    /// Write a record to disk
    fn write_record(&self, record: &PredictionRecord) {
        let json = serde_json::to_string(record).unwrap();
        let mut writer = self.writer.lock().unwrap();
        writeln!(writer, "{}", json).ok();
        writer.flush().ok();
    }

    /// Clean up records older than max_pending_age_ms
    fn cleanup_old_records(&self, pending: &mut HashMap<u64, PredictionRecord>) {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let max_age_ns = self.max_pending_age_ms * 1_000_000;

        let old_ids: Vec<u64> = pending
            .iter()
            .filter(|(_, r)| now_ns.saturating_sub(r.timestamp_ns) > max_age_ns)
            .map(|(id, _)| *id)
            .collect();

        for id in old_ids {
            if let Some(record) = pending.remove(&id) {
                // Write with empty outcomes
                self.write_record(&record);
            }
        }
    }

    /// Flush all pending records
    pub fn flush_all(&self) {
        let mut pending = self.pending_records.lock().unwrap();
        for (_, record) in pending.drain() {
            self.write_record(&record);
        }
        self.writer.lock().unwrap().flush().ok();
    }

    /// Get statistics about logged predictions
    pub fn get_stats(&self) -> PredictionLoggerStats {
        let pending = self.pending_records.lock().unwrap();
        PredictionLoggerStats {
            total_cycles: self.cycle_id,
            pending_count: pending.len(),
        }
    }
}

/// Statistics about the prediction logger
#[derive(Debug, Clone)]
pub struct PredictionLoggerStats {
    pub total_cycles: u64,
    pub pending_count: usize,
}

impl Drop for PredictionLogger {
    fn drop(&mut self) {
        self.flush_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_probability_estimation() {
        // Use a reasonable kappa for testing (not the high default)
        // kappa=10 means λ(0) = 10 fills/second at the touch
        let mut params = MarketParams::default();
        params.kappa_robust = 10.0; // Reasonable kappa for test
        params.use_kappa_robust = true;

        // At depth 0, probability should be high
        let p_0 = estimate_fill_probability(0.0, &params, 1.0);
        assert!(p_0 > 0.5, "Fill prob at touch should be high, got {}", p_0);

        // At depth 50 bps, probability should be low
        // λ(50) = 10 × exp(-50/10) = 10 × exp(-5) = 0.067
        // P(fill in 1s) = 1 - exp(-0.067) = 0.065
        let p_50 = estimate_fill_probability(50.0, &params, 1.0);
        assert!(
            p_50 < 0.1,
            "Fill prob at 50 bps should be low, got {}",
            p_50
        );

        // Longer horizon = higher probability
        let p_1s = estimate_fill_probability(10.0, &params, 1.0);
        let p_10s = estimate_fill_probability(10.0, &params, 10.0);
        assert!(p_10s > p_1s, "Longer horizon should have higher fill prob");
    }

    #[test]
    fn test_regime_classification() {
        let mut params = MarketParams::default();

        // Quiet regime
        params.should_pull_quotes = false;
        params.is_toxic_regime = false;
        params.arrival_intensity = 0.5;
        params.sigma = 0.0001;
        assert_eq!(Regime::from(&params), Regime::Quiet);

        // Cascade regime
        params.should_pull_quotes = true;
        assert_eq!(Regime::from(&params), Regime::Cascade);
    }
}
