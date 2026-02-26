//! Update messages for the centralized belief state.
//!
//! This module defines the `BeliefUpdate` enum that estimators use to
//! publish updates to the `CentralBeliefState`. All belief updates flow
//! through this unified message system.
//!
//! ## Design Rationale
//!
//! By using a single enum for all updates, we:
//! - Ensure all updates are explicit and type-safe
//! - Make it easy to add new update types
//! - Enable logging/replay of all belief updates
//! - Simplify testing by mocking message sequences

use std::collections::HashMap;

/// Update message for the centralized belief state.
///
/// Estimators publish these updates, and the CentralBeliefState
/// processes them to maintain consistent Bayesian posteriors.
///
/// Note: This enum is not Clone due to RequestSnapshot containing a oneshot sender.
/// Most variants can be cloned manually if needed.
#[derive(Debug)]
pub enum BeliefUpdate {
    // =========================================================================
    // Price and Market Data Observations
    // =========================================================================
    /// Observed price return for drift/volatility updates.
    ///
    /// Published by: Quote engine (on each mid price update)
    PriceReturn {
        /// Fractional return (e.g., 0.001 = 10 bps)
        return_frac: f64,
        /// Time elapsed since last observation (seconds)
        dt_secs: f64,
        /// Timestamp of observation (epoch ms)
        timestamp_ms: u64,
    },

    /// Our own fill observation (gold standard for kappa).
    ///
    /// Published by: Fill processor
    OwnFill {
        /// Fill price
        price: f64,
        /// Fill size
        size: f64,
        /// Mid price at time of fill
        mid: f64,
        /// Was this a buy?
        is_buy: bool,
        /// Was fill aligned with position direction?
        is_aligned: bool,
        /// Realized adverse selection (bps)
        realized_as_bps: f64,
        /// Realized edge (bps)
        realized_edge_bps: f64,
        /// Timestamp (epoch ms)
        timestamp_ms: u64,
        /// Order ID for linking predictions
        order_id: Option<u64>,
        /// Size we had quoted at this level (for fill-size scaling).
        /// Defaults to fill size if unknown.
        quoted_size: f64,
    },

    /// Market trade observation (for robust kappa estimation).
    ///
    /// Published by: Trade handler
    MarketTrade {
        /// Trade price
        price: f64,
        /// Mid price at time of trade
        mid: f64,
        /// Timestamp (epoch ms)
        timestamp_ms: u64,
    },

    /// L2 book update (for book kappa estimation).
    ///
    /// Published by: Book handler
    BookUpdate {
        /// Bid levels [(price, size)]
        bids: Vec<(f64, f64)>,
        /// Ask levels [(price, size)]
        asks: Vec<(f64, f64)>,
        /// Mid price
        mid: f64,
        /// Timestamp (epoch ms)
        timestamp_ms: u64,
    },

    // =========================================================================
    // Regime and Changepoint
    // =========================================================================
    /// Regime probabilities update from HMM.
    ///
    /// Published by: RegimeHMM
    RegimeUpdate {
        /// [quiet, normal, bursty, cascade] probabilities
        probs: [f64; 4],
        /// Features used for this update (for diagnostics)
        features: Option<HashMap<String, f64>>,
    },

    /// Changepoint observation for BOCD.
    ///
    /// Published by: Volatility estimator or changepoint detector
    ChangepointObs {
        /// The observation value (typically volatility or return)
        observation: f64,
    },

    /// Continuation signal updates from external estimators.
    ///
    /// Published by: MomentumModel, TrendPersistenceTracker
    ContinuationSignals {
        /// Momentum continuation probability [0, 1]
        momentum: f64,
        /// Multi-timeframe trend agreement [0, 1]
        trend_agreement: f64,
        /// Trend confidence [0, 1]
        trend_confidence: f64,
    },

    // =========================================================================
    // Calibration
    // =========================================================================
    /// Log a prediction for later calibration.
    ///
    /// Published by: Any component making a prediction
    LogPrediction {
        /// The prediction to log
        prediction: PredictionLog,
    },

    /// Record the outcome for a previous prediction.
    ///
    /// Published by: Fill processor, outcome observer
    RecordOutcome {
        /// ID of the prediction to resolve
        prediction_id: u64,
        /// Actual outcome value [0, 1] for binary, continuous otherwise
        actual_value: f64,
        /// Delay from prediction to outcome (ms)
        delay_ms: u64,
    },

    /// Signal mutual information update.
    ///
    /// Published by: Signal quality monitor
    SignalMiUpdate {
        /// Name of the signal
        signal_name: String,
        /// Mutual information value
        mi: f64,
    },

    /// Microstructure update from VPIN/OFI estimators.
    ///
    /// Published by: VpinEstimator, EnhancedFlowEstimator
    MicrostructureUpdate {
        /// VPIN value [0, 1]
        vpin: f64,
        /// VPIN velocity (rate of change)
        vpin_velocity: f64,
        /// Depth-weighted OFI [-1, 1]
        depth_ofi: f64,
        /// Liquidity evaporation score [0, 1]
        liquidity_evaporation: f64,
        /// Order flow direction [-1, 1]
        order_flow_direction: f64,
        /// Confidence in estimates [0, 1]
        confidence: f64,
        /// Number of VPIN buckets
        vpin_buckets: usize,
        // === Phase 1A: Toxic Volume Refinements ===
        /// Trade size sigma (deviation from baseline)
        trade_size_sigma: f64,
        /// Toxicity acceleration factor [1.0, 2.0]
        toxicity_acceleration: f64,
        /// Cumulative OFI with decay [-1, 1]
        cofi: f64,
        /// COFI velocity
        cofi_velocity: f64,
        /// Whether sustained shift detected
        is_sustained_shift: bool,
    },

    /// Cross-venue update from joint Binance + Hyperliquid analysis.
    ///
    /// Published by: CrossVenueAnalyzer
    CrossVenueUpdate {
        /// Joint direction belief [-1, +1]
        direction: f64,
        /// Confidence based on venue agreement [0, 1]
        confidence: f64,
        /// Where is price discovery? [0=HL, 1=Binance]
        discovery_venue: f64,
        /// Maximum VPIN across venues [0, 1]
        max_toxicity: f64,
        /// Average VPIN across venues [0, 1]
        avg_toxicity: f64,
        /// Agreement score [-1, 1]
        agreement: f64,
        /// Imbalance divergence (binance - hl) [-2, 2]
        divergence: f64,
        /// Intensity ratio λ_B / (λ_B + λ_H) [0, 1]
        intensity_ratio: f64,
        /// Rolling imbalance correlation [-1, 1]
        imbalance_correlation: f64,
        /// Toxicity alert active?
        toxicity_alert: bool,
        /// Divergence alert active?
        divergence_alert: bool,
        /// Timestamp (epoch ms)
        timestamp_ms: u64,
    },

    /// Trade flow imbalance update from TradeFlowTracker.
    ///
    /// Published by: Quote engine (after TradeFlowTracker updates)
    FlowUpdate {
        /// 1-second EWMA trade flow imbalance [-1, 1]
        imbalance_1s: f64,
        /// 5-second EWMA trade flow imbalance [-1, 1]
        imbalance_5s: f64,
        /// 30-second EWMA trade flow imbalance [-1, 1]
        imbalance_30s: f64,
        /// Timestamp (epoch ms)
        timestamp_ms: u64,
    },

    // =========================================================================
    // Control
    // =========================================================================
    /// Soft reset: decay beliefs while retaining some information.
    ///
    /// Used after changepoint detection to blend toward priors.
    SoftReset {
        /// Fraction of information to retain [0, 1]
        retention: f64,
    },

    /// Hard reset: return to prior beliefs.
    ///
    /// Used on catastrophic events or session start.
    HardReset,

    /// Request a snapshot (used for async consumers).
    ///
    /// The callback channel receives the snapshot.
    RequestSnapshot {
        /// Channel to send snapshot back on
        #[allow(dead_code)]
        callback: tokio::sync::oneshot::Sender<super::BeliefSnapshot>,
    },
}

/// Prediction log entry for calibration tracking.
#[derive(Debug, Clone)]
pub struct PredictionLog {
    /// Type of prediction
    pub prediction_type: PredictionType,
    /// Predicted probability [0, 1]
    pub predicted_prob: f64,
    /// Confidence in the prediction [0, 1]
    pub confidence: f64,
    /// Current regime at prediction time
    pub regime: usize,
    /// Additional features/context
    pub features: HashMap<String, f64>,
    /// Timestamp (epoch ms)
    pub timestamp_ms: u64,
}

impl PredictionLog {
    /// Create a new prediction log entry.
    pub fn new(
        prediction_type: PredictionType,
        predicted_prob: f64,
        confidence: f64,
        features: HashMap<String, f64>,
    ) -> Self {
        Self {
            prediction_type,
            predicted_prob: predicted_prob.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            regime: 1, // Default to normal
            features,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        }
    }

    /// Set the regime.
    pub fn with_regime(mut self, regime: usize) -> Self {
        self.regime = regime.min(3);
        self
    }
}

/// Type of prediction being logged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PredictionType {
    /// Probability of fill at given spread
    FillProbability,
    /// Probability of adverse selection
    AdverseSelection,
    /// Probability of regime change
    RegimeChange,
    /// Probability of position continuation
    PositionContinuation,
    /// Predicted edge direction
    EdgeDirection,
}

impl PredictionType {
    /// Get human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            PredictionType::FillProbability => "fill_probability",
            PredictionType::AdverseSelection => "adverse_selection",
            PredictionType::RegimeChange => "regime_change",
            PredictionType::PositionContinuation => "position_continuation",
            PredictionType::EdgeDirection => "edge_direction",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_log_creation() {
        let features = HashMap::from([("spread_bps".to_string(), 5.0)]);
        let log = PredictionLog::new(PredictionType::FillProbability, 0.75, 0.9, features);

        assert_eq!(log.prediction_type, PredictionType::FillProbability);
        assert!((log.predicted_prob - 0.75).abs() < 1e-10);
        assert!((log.confidence - 0.9).abs() < 1e-10);
        assert_eq!(log.regime, 1);
    }

    #[test]
    fn test_prediction_log_clamping() {
        let log = PredictionLog::new(
            PredictionType::FillProbability,
            1.5,  // Should be clamped to 1.0
            -0.5, // Should be clamped to 0.0
            HashMap::new(),
        );

        assert!((log.predicted_prob - 1.0).abs() < 1e-10);
        assert!((log.confidence - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_belief_update_variants() {
        // Test that we can construct each variant
        let _price_return = BeliefUpdate::PriceReturn {
            return_frac: 0.001,
            dt_secs: 1.0,
            timestamp_ms: 12345,
        };

        let _own_fill = BeliefUpdate::OwnFill {
            price: 100.0,
            size: 1.0,
            mid: 99.95,
            is_buy: true,
            is_aligned: true,
            realized_as_bps: 1.5,
            realized_edge_bps: 2.0,
            timestamp_ms: 12345,
            order_id: Some(42),
            quoted_size: 1.0,
        };

        let _regime_update = BeliefUpdate::RegimeUpdate {
            probs: [0.1, 0.6, 0.2, 0.1],
            features: None,
        };

        let _soft_reset = BeliefUpdate::SoftReset { retention: 0.5 };

        let _hard_reset = BeliefUpdate::HardReset;
    }

    #[test]
    fn test_prediction_type_name() {
        assert_eq!(PredictionType::FillProbability.name(), "fill_probability");
        assert_eq!(PredictionType::AdverseSelection.name(), "adverse_selection");
        assert_eq!(PredictionType::RegimeChange.name(), "regime_change");
        assert_eq!(
            PredictionType::PositionContinuation.name(),
            "position_continuation"
        );
        assert_eq!(PredictionType::EdgeDirection.name(), "edge_direction");
    }
}
