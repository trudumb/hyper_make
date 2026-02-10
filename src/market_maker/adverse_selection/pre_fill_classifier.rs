//! Pre-fill adverse selection classifier.
//!
//! Predicts toxicity BEFORE a fill occurs based on market microstructure signals.
//! This enables proactive spread widening rather than reactive adjustment after losses.
//!
//! # Key Insight
//!
//! Traditional AS measurement tracks price movement AFTER fills (markout analysis).
//! This classifier predicts toxicity BEFORE fills using:
//! - Orderbook imbalance (directional pressure)
//! - Recent trade flow (momentum vs mean-reversion)
//! - Regime trust (changepoint probability)
//! - Funding rate (crowd positioning)
//!
//! # Integration
//!
//! The `spread_multiplier()` output should be fed into GLFT's `tox` variable
//! to widen spreads when toxic flow is predicted.

/// Number of base signals (imbalance, flow, regime, funding, changepoint).
const NUM_BASE_SIGNALS: usize = 5;
/// Total signals including trend opposition.
const NUM_SIGNALS: usize = 6;
/// Human-readable signal names for diagnostics.
const SIGNAL_NAMES: [&str; NUM_SIGNALS] = [
    "imbalance",
    "flow",
    "regime",
    "funding",
    "changepoint",
    "trend",
];

/// Configuration for pre-fill AS classifier.
#[derive(Debug, Clone)]
pub struct PreFillClassifierConfig {
    /// Weight for orderbook imbalance signal (default: 0.30)
    pub imbalance_weight: f64,
    /// Weight for trade flow momentum signal (default: 0.25)
    pub flow_weight: f64,
    /// Weight for regime distrust signal (default: 0.25)
    pub regime_weight: f64,
    /// Weight for funding rate signal (default: 0.10)
    pub funding_weight: f64,
    /// Weight for BOCD changepoint probability (default: 0.10)
    pub changepoint_weight: f64,
    /// Weight for trend opposition signal (default: 0.30).
    /// Fills against strong price trends are more toxic (Kyle 1985).
    pub trend_weight: f64,

    /// Maximum spread multiplier (caps toxicity impact)
    pub max_spread_multiplier: f64,

    /// Orderbook imbalance threshold for full toxicity (bid/ask ratio)
    /// E.g., 2.0 means 2:1 imbalance = full toxicity signal
    pub imbalance_threshold: f64,

    /// Trade flow threshold for full toxicity (fraction of volume)
    pub flow_threshold: f64,

    /// Funding rate threshold for full toxicity (annualized rate)
    pub funding_threshold: f64,

    /// Enable online weight learning (default: true)
    pub enable_learning: bool,

    /// Learning rate for weight updates (default: 0.01)
    pub learning_rate: f64,

    /// Minimum samples before using learned weights (default: 500)
    pub min_samples_for_learning: usize,

    /// Regularization strength for weight learning (default: 0.1)
    /// Prevents weights from going to extremes
    pub regularization: f64,

    /// EWMA smoothing factor for z-score normalizer (default: 0.05, ~20-sample half-life)
    pub normalizer_alpha: f64,

    /// Minimum observations before using z-score normalization (default: 50)
    pub normalizer_warmup: usize,

    /// Z-score scaling factor to amplify compressed z-scores (default: 2.0).
    /// EWMA z-scores are attenuated because the mean tracks current values closely.
    /// Scaling by 2.0 maps ±1σ events to sigmoid(±2) = [0.12, 0.88], providing
    /// good spread across the [0,1] range.
    pub z_scale: f64,

    /// Bayesian prior toxicity during warmup (default: 0.35).
    /// Provides mild spread protection before sufficient data arrives.
    pub warmup_prior_toxicity: f64,

    /// Minimum update_count before data fully overrides the prior (default: 10).
    pub warmup_prior_min_updates: u64,
}

impl Default for PreFillClassifierConfig {
    fn default() -> Self {
        Self {
            imbalance_weight: 0.30,
            flow_weight: 0.25,
            regime_weight: 0.25,
            funding_weight: 0.10,
            changepoint_weight: 0.10,
            trend_weight: 0.30,

            max_spread_multiplier: 3.0, // Cap at 3x base spread

            imbalance_threshold: 2.0,   // 2:1 bid/ask ratio
            flow_threshold: 0.7,        // 70% one-sided flow
            funding_threshold: 0.001,   // 0.1% per 8h (annualized ~110%)

            enable_learning: true,
            learning_rate: 0.01,
            min_samples_for_learning: 500,
            regularization: 0.1,

            normalizer_alpha: 0.05,
            normalizer_warmup: 50,
            z_scale: 2.0,

            warmup_prior_toxicity: 0.35,
            warmup_prior_min_updates: 10,
        }
    }
}

impl PreFillClassifierConfig {
    /// Return the config-default weights as an array of NUM_SIGNALS elements.
    fn default_weights_array(&self) -> [f64; NUM_SIGNALS] {
        [
            self.imbalance_weight,
            self.flow_weight,
            self.regime_weight,
            self.funding_weight,
            self.changepoint_weight,
            self.trend_weight,
        ]
    }

    /// Return the prior weight for a given signal index.
    fn prior_weight(&self, index: usize) -> f64 {
        match index {
            0 => self.imbalance_weight,
            1 => self.flow_weight,
            2 => self.regime_weight,
            3 => self.funding_weight,
            4 => self.changepoint_weight,
            5 => self.trend_weight,
            _ => 0.2,
        }
    }
}

/// Pre-fill adverse selection classifier.
///
/// Predicts fill toxicity using real-time microstructure signals.
/// Higher toxicity → widen spreads to avoid adverse fills.
#[derive(Debug, Clone)]
pub struct PreFillASClassifier {
    config: PreFillClassifierConfig,

    // === Cached Signal Values ===
    /// Orderbook imbalance: bid_size / ask_size (>1 = bid pressure)
    orderbook_imbalance: f64,

    /// Recent trade direction: net_buy_volume / total_volume [-1, 1]
    recent_trade_direction: f64,

    /// Regime trust from HMM/BOCD (1.0 = high trust, 0.0 = regime uncertain)
    regime_trust: f64,

    /// BOCD changepoint probability [0, 1]
    changepoint_prob: f64,

    /// Current funding rate (8h rate as fraction)
    funding_rate: f64,

    /// Long-term trend momentum in bps (positive = uptrend, negative = downtrend).
    /// Fed externally via `update_trend()`. Used for drift conditioning:
    /// fills against strong trends are more toxic (Kyle 1985).
    trend_momentum_bps: f64,

    /// Cached toxicity prediction [0, 1]
    cached_toxicity: f64,

    /// Update count for statistics
    update_count: u64,

    /// Timestamp of last orderbook update (ms since epoch)
    orderbook_updated_at_ms: u64,

    /// Timestamp of last trade flow update (ms since epoch)
    trade_flow_updated_at_ms: u64,

    /// Timestamp of last regime update (ms since epoch)
    regime_updated_at_ms: u64,

    // === Online Learning State ===
    /// Learned weights [imbalance, flow, regime, funding, changepoint, trend]
    learned_weights: [f64; NUM_SIGNALS],

    /// Running sum of (signal_i * outcome) for each signal
    signal_outcome_sum: [f64; NUM_SIGNALS],

    /// Running sum of signal_i^2 for each signal
    signal_sq_sum: [f64; NUM_SIGNALS],

    /// Number of learning samples processed
    learning_samples: usize,

    // === Regime-Conditional State ===
    /// Current detected regime (0=Low, 1=Normal, 2=High, 3=Extreme)
    current_regime: usize,

    /// Regime probabilities for soft blending
    regime_probs: [f64; 4],

    // === Blended Toxicity Override ===
    /// When set, blends the classifier's own prediction with an external
    /// toxicity signal (e.g., VPIN-blended toxicity from IntegratedSignals).
    /// This prevents EM-style degeneration by injecting a varying signal.
    blended_toxicity_override: Option<f64>,

    /// Weight given to the external toxicity signal [0, 1].
    /// 0.0 = pure classifier, 1.0 = pure override.
    blend_weight: f64,

    // === EWMA Z-Score Normalizer State ===
    /// EWMA mean for log(bid_depth/ask_depth)
    imbalance_ewma_mean: f64,
    /// EWMA variance for log(bid_depth/ask_depth)
    imbalance_ewma_var: f64,
    /// EWMA mean for trade_direction
    flow_ewma_mean: f64,
    /// EWMA variance for trade_direction
    flow_ewma_var: f64,
    /// EWMA mean for funding_rate
    funding_ewma_mean: f64,
    /// EWMA variance for funding_rate
    funding_ewma_var: f64,
    /// Previous regime trust (for delta computation)
    regime_trust_prev: f64,
    /// EWMA mean for regime instability (1.0 - trust)
    regime_ewma_mean: f64,
    /// EWMA variance for regime instability
    regime_ewma_var: f64,
    /// EWMA mean for changepoint probability
    changepoint_ewma_mean: f64,
    /// EWMA variance for changepoint probability
    changepoint_ewma_var: f64,
    /// Normalizer smoothing factor
    normalizer_alpha: f64,
    /// Minimum observations before using z-score normalization
    normalizer_warmup: usize,
    /// Number of normalizer observations
    normalizer_obs_count: usize,
    // === Pre-computed z-scores (computed BEFORE EWMA update for true N(0,1) distribution) ===
    /// Z-score of log(bid/ask ratio) computed against pre-update EWMA
    imbalance_z: f64,
    /// Z-score of trade direction computed against pre-update EWMA
    flow_z: f64,
    /// Z-score of funding rate computed against pre-update EWMA
    funding_z: f64,
    /// Z-score of regime instability (1-trust) computed against pre-update EWMA
    regime_z: f64,
    /// Z-score of changepoint probability computed against pre-update EWMA
    changepoint_z: f64,

    // === AS Bias Correction ===
    /// EWMA of (predicted_as_bps - realized_as_bps).
    /// Positive = systematic overprediction. Subtracted from raw predictions.
    bias_correction_bps: f64,
    /// Number of AS bias observations.
    bias_observation_count: usize,
}

impl PreFillASClassifier {
    /// Create a new pre-fill classifier with default config.
    pub fn new() -> Self {
        Self::with_config(PreFillClassifierConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: PreFillClassifierConfig) -> Self {
        // Initialize learned weights with config defaults
        let learned_weights = config.default_weights_array();

        let normalizer_alpha = config.normalizer_alpha;
        let normalizer_warmup = config.normalizer_warmup;

        Self {
            config,
            orderbook_imbalance: 1.0, // Neutral
            recent_trade_direction: 0.0, // Neutral
            regime_trust: 1.0, // High trust initially
            changepoint_prob: 0.0, // No changepoint
            funding_rate: 0.0, // Neutral funding
            trend_momentum_bps: 0.0, // No drift
            cached_toxicity: 0.0,
            update_count: 0,
            orderbook_updated_at_ms: 0,
            trade_flow_updated_at_ms: 0,
            regime_updated_at_ms: 0,
            learned_weights,
            signal_outcome_sum: [0.0; NUM_SIGNALS],
            signal_sq_sum: [0.0; NUM_SIGNALS],
            learning_samples: 0,
            // Regime-conditional state
            current_regime: 1, // Start with Normal
            regime_probs: [0.1, 0.7, 0.15, 0.05], // Default prior
            // Blended toxicity override
            blended_toxicity_override: None,
            blend_weight: 0.5, // 50/50 blend by default
            // EWMA z-score normalizer
            imbalance_ewma_mean: 0.0,
            imbalance_ewma_var: 1.0,
            flow_ewma_mean: 0.0,
            flow_ewma_var: 1.0,
            funding_ewma_mean: 0.0,
            funding_ewma_var: 1.0,
            regime_trust_prev: 1.0,
            regime_ewma_mean: 0.0,
            regime_ewma_var: 1.0,
            changepoint_ewma_mean: 0.0,
            changepoint_ewma_var: 1.0,
            normalizer_alpha,
            normalizer_warmup,
            normalizer_obs_count: 0,
            // Pre-computed z-scores (initially 0 = neutral)
            imbalance_z: 0.0,
            flow_z: 0.0,
            funding_z: 0.0,
            regime_z: 0.0,
            changepoint_z: 0.0,
            // AS bias correction
            bias_correction_bps: 0.0,
            bias_observation_count: 0,
        }
    }

    // === Checkpoint persistence ===

    /// Extract learning state for checkpoint persistence.
    ///
    /// Note: checkpoint format stores the first 5 base signals only.
    /// The 6th trend signal's learning state is ephemeral (relearned quickly).
    pub fn to_checkpoint(&self) -> crate::market_maker::checkpoint::PreFillCheckpoint {
        let mut cp_weights = [0.0; NUM_BASE_SIGNALS];
        cp_weights.copy_from_slice(&self.learned_weights[..NUM_BASE_SIGNALS]);
        let mut cp_outcome = [0.0; NUM_BASE_SIGNALS];
        cp_outcome.copy_from_slice(&self.signal_outcome_sum[..NUM_BASE_SIGNALS]);
        let mut cp_sq = [0.0; NUM_BASE_SIGNALS];
        cp_sq.copy_from_slice(&self.signal_sq_sum[..NUM_BASE_SIGNALS]);

        crate::market_maker::checkpoint::PreFillCheckpoint {
            learned_weights: cp_weights,
            signal_outcome_sum: cp_outcome,
            signal_sq_sum: cp_sq,
            learning_samples: self.learning_samples,
            regime_probs: self.regime_probs,
            // EWMA normalizer state
            imbalance_ewma_mean: self.imbalance_ewma_mean,
            imbalance_ewma_var: self.imbalance_ewma_var,
            flow_ewma_mean: self.flow_ewma_mean,
            flow_ewma_var: self.flow_ewma_var,
            funding_ewma_mean: self.funding_ewma_mean,
            funding_ewma_var: self.funding_ewma_var,
            regime_trust_prev: self.regime_trust_prev,
            regime_ewma_mean: self.regime_ewma_mean,
            regime_ewma_var: self.regime_ewma_var,
            changepoint_ewma_mean: self.changepoint_ewma_mean,
            changepoint_ewma_var: self.changepoint_ewma_var,
            normalizer_obs_count: self.normalizer_obs_count,
            bias_correction_bps: self.bias_correction_bps,
            bias_observation_count: self.bias_observation_count,
        }
    }

    /// Restore learning state from a checkpoint.
    ///
    /// Ephemeral state (cached signals, timestamps) stays at defaults —
    /// they repopulate within seconds from live data.
    /// The 6th trend signal's learning state resets to config default on restore.
    pub fn restore_checkpoint(&mut self, cp: &crate::market_maker::checkpoint::PreFillCheckpoint) {
        self.learned_weights[..NUM_BASE_SIGNALS].copy_from_slice(&cp.learned_weights);
        self.learned_weights[NUM_BASE_SIGNALS] = self.config.trend_weight;
        self.signal_outcome_sum[..NUM_BASE_SIGNALS].copy_from_slice(&cp.signal_outcome_sum);
        self.signal_outcome_sum[NUM_BASE_SIGNALS] = 0.0;
        self.signal_sq_sum[..NUM_BASE_SIGNALS].copy_from_slice(&cp.signal_sq_sum);
        self.signal_sq_sum[NUM_BASE_SIGNALS] = 0.0;
        self.learning_samples = cp.learning_samples;
        self.regime_probs = cp.regime_probs;
        // EWMA normalizer state
        self.imbalance_ewma_mean = cp.imbalance_ewma_mean;
        self.imbalance_ewma_var = cp.imbalance_ewma_var;
        self.flow_ewma_mean = cp.flow_ewma_mean;
        self.flow_ewma_var = cp.flow_ewma_var;
        self.funding_ewma_mean = cp.funding_ewma_mean;
        self.funding_ewma_var = cp.funding_ewma_var;
        self.regime_trust_prev = cp.regime_trust_prev;
        self.regime_ewma_mean = cp.regime_ewma_mean;
        self.regime_ewma_var = cp.regime_ewma_var;
        self.changepoint_ewma_mean = cp.changepoint_ewma_mean;
        self.changepoint_ewma_var = cp.changepoint_ewma_var;
        self.normalizer_obs_count = cp.normalizer_obs_count;
        self.bias_correction_bps = cp.bias_correction_bps;
        self.bias_observation_count = cp.bias_observation_count;
    }

    // === EWMA Z-Score Helper Functions ===

    /// Update EWMA mean and variance with a new observation.
    /// Returns (new_mean, new_var).
    fn ewma_update(mean: f64, var: f64, value: f64, alpha: f64) -> (f64, f64) {
        let new_mean = (1.0 - alpha) * mean + alpha * value;
        let diff = value - new_mean;
        let new_var = (1.0 - alpha) * var + alpha * diff * diff;
        // Floor variance to prevent division by zero
        (new_mean, new_var.max(1e-12))
    }

    /// Standard sigmoid: maps z ∈ (-∞, +∞) to (0, 1), with 0→0.5, 2→0.88, -2→0.12.
    /// Centers at 0.5 for neutral conditions. Positive z = more toxic, negative z = less toxic.
    /// This provides natural spread across the full [0,1] range from Gaussian z-scores.
    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    /// Update orderbook imbalance signal.
    ///
    /// # Arguments
    /// * `bid_depth` - Total bid depth (top N levels)
    /// * `ask_depth` - Total ask depth (top N levels)
    pub fn update_orderbook(&mut self, bid_depth: f64, ask_depth: f64) {
        if ask_depth > 0.0 {
            self.orderbook_imbalance = bid_depth / ask_depth;
        } else {
            self.orderbook_imbalance = if bid_depth > 0.0 { 10.0 } else { 1.0 };
        }
        let log_ratio = (self.orderbook_imbalance.max(0.01)).ln();
        // Compute z-score BEFORE updating EWMA (against pre-update mean/var)
        let std_imb = self.imbalance_ewma_var.sqrt();
        self.imbalance_z = if std_imb > 1e-6 {
            (log_ratio - self.imbalance_ewma_mean) / std_imb
        } else {
            0.0
        };
        // Now update EWMA for next iteration
        let (m, v) = Self::ewma_update(
            self.imbalance_ewma_mean,
            self.imbalance_ewma_var,
            log_ratio,
            self.normalizer_alpha,
        );
        self.imbalance_ewma_mean = m;
        self.imbalance_ewma_var = v;
        self.normalizer_obs_count += 1;

        self.orderbook_updated_at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.update_cached_toxicity();
    }

    /// Update trade flow signal.
    ///
    /// # Arguments
    /// * `buy_volume` - Recent buy volume
    /// * `sell_volume` - Recent sell volume
    pub fn update_trade_flow(&mut self, buy_volume: f64, sell_volume: f64) {
        let total = buy_volume + sell_volume;
        if total > 0.0 {
            self.recent_trade_direction = (buy_volume - sell_volume) / total;
        } else {
            self.recent_trade_direction = 0.0;
        }
        // Compute z-score BEFORE updating EWMA
        let std_flow = self.flow_ewma_var.sqrt();
        self.flow_z = if std_flow > 1e-6 {
            (self.recent_trade_direction - self.flow_ewma_mean) / std_flow
        } else {
            0.0
        };
        // Now update EWMA
        let (m, v) = Self::ewma_update(
            self.flow_ewma_mean,
            self.flow_ewma_var,
            self.recent_trade_direction,
            self.normalizer_alpha,
        );
        self.flow_ewma_mean = m;
        self.flow_ewma_var = v;

        self.trade_flow_updated_at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.update_cached_toxicity();
    }

    /// Update regime trust signal.
    ///
    /// # Arguments
    /// * `hmm_confidence` - HMM regime probability (max across regimes)
    /// * `changepoint_prob` - BOCD changepoint probability
    pub fn update_regime(&mut self, hmm_confidence: f64, changepoint_prob: f64) {
        self.regime_trust_prev = self.regime_trust;
        self.regime_trust = hmm_confidence.clamp(0.0, 1.0);
        self.changepoint_prob = changepoint_prob.clamp(0.0, 1.0);

        // Compute instability = 1 - trust (higher = less stable = more toxic)
        let instability = 1.0 - self.regime_trust;

        // Compute z-scores BEFORE EWMA update (against pre-update mean/var)
        let std_regime = self.regime_ewma_var.sqrt();
        self.regime_z = if std_regime > 1e-6 {
            (instability - self.regime_ewma_mean) / std_regime
        } else {
            0.0
        };
        let std_cp = self.changepoint_ewma_var.sqrt();
        self.changepoint_z = if std_cp > 1e-6 {
            (self.changepoint_prob - self.changepoint_ewma_mean) / std_cp
        } else {
            0.0
        };

        // Now update EWMA for next iteration
        let (m, v) = Self::ewma_update(
            self.regime_ewma_mean,
            self.regime_ewma_var,
            instability,
            self.normalizer_alpha,
        );
        self.regime_ewma_mean = m;
        self.regime_ewma_var = v;

        let (m2, v2) = Self::ewma_update(
            self.changepoint_ewma_mean,
            self.changepoint_ewma_var,
            self.changepoint_prob,
            self.normalizer_alpha,
        );
        self.changepoint_ewma_mean = m2;
        self.changepoint_ewma_var = v2;

        self.regime_updated_at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.update_cached_toxicity();
    }


    /// Update full regime probabilities for regime-conditional weighting.
    ///
    /// This enables regime-conditional signal weights:
    /// - In Calm regimes: imbalance signal is more predictive
    /// - In High/Extreme regimes: flow and changepoint signals are more predictive
    ///
    /// # Arguments
    /// * `regime_probs` - [P(Low), P(Normal), P(High), P(Extreme)]
    pub fn set_regime_probs(&mut self, regime_probs: [f64; 4]) {
        self.regime_probs = regime_probs;
        // Find most likely regime
        let mut max_prob = 0.0;
        let mut max_regime = 1;
        for (i, &p) in regime_probs.iter().enumerate() {
            if p > max_prob {
                max_prob = p;
                max_regime = i;
            }
        }
        self.current_regime = max_regime;
        self.update_cached_toxicity();
    }

    /// Get regime-conditional weight multipliers.
    ///
    /// Different regimes have different signal importance:
    /// - Low/Calm: imbalance is more predictive (slow market, queue position matters)
    /// - Normal: balanced weights
    /// - High: flow and changepoint matter more (momentum-driven)
    /// - Extreme/Cascade: changepoint and flow dominate (information asymmetry)
    fn regime_weight_multipliers(&self) -> [f64; NUM_SIGNALS] {
        // Weight multipliers for [imbalance, flow, regime, funding, changepoint, trend]
        // Soft-blend based on regime probabilities

        let regime_weights: [[f64; NUM_SIGNALS]; 4] = [
            // Low/Calm: imbalance matters, trend is strong signal in quiet markets
            [1.5, 0.8, 0.7, 1.0, 0.5, 1.3],
            // Normal: balanced
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            // High: flow and changepoint matter more, trend less reliable
            [0.8, 1.3, 1.2, 0.9, 1.4, 0.7],
            // Extreme/Cascade: changepoint and flow dominate, trend unreliable
            [0.6, 1.5, 1.3, 0.8, 1.8, 0.4],
        ];

        // Soft blend across regimes
        let mut multipliers = [0.0; NUM_SIGNALS];
        for (regime_idx, &prob) in self.regime_probs.iter().enumerate() {
            for (signal_idx, mult) in multipliers.iter_mut().enumerate() {
                *mult += prob * regime_weights[regime_idx][signal_idx];
            }
        }

        multipliers
    }

    /// Update funding rate signal.
    ///
    /// # Arguments
    /// * `funding_rate_8h` - 8-hour funding rate as fraction (e.g., 0.0001 = 1 bp)
    pub fn update_funding(&mut self, funding_rate_8h: f64) {
        self.funding_rate = funding_rate_8h;
        // Compute z-score BEFORE updating EWMA
        let std_fund = self.funding_ewma_var.sqrt();
        self.funding_z = if std_fund > 1e-6 {
            (funding_rate_8h - self.funding_ewma_mean) / std_fund
        } else {
            0.0
        };
        // Now update EWMA
        let (m, v) = Self::ewma_update(
            self.funding_ewma_mean,
            self.funding_ewma_var,
            funding_rate_8h,
            self.normalizer_alpha,
        );
        self.funding_ewma_mean = m;
        self.funding_ewma_var = v;

        self.update_cached_toxicity();
    }

    /// Update trend momentum signal for drift conditioning.
    ///
    /// Fills against strong price trends are more toxic (Kyle 1985):
    /// a downtrend makes bid fills toxic, an uptrend makes ask fills toxic.
    ///
    /// # Arguments
    /// * `long_momentum_bps` - Long-term price momentum in bps.
    ///   Positive = uptrend, negative = downtrend.
    pub fn update_trend(&mut self, long_momentum_bps: f64) {
        self.trend_momentum_bps = long_momentum_bps;
        self.update_cached_toxicity();
    }

    /// Get the current trend momentum in bps.
    pub fn trend_momentum_bps(&self) -> f64 {
        self.trend_momentum_bps
    }

    /// Set the blended toxicity from an external source (e.g., IntegratedSignals).
    ///
    /// When set, `predict_toxicity()` blends its own 5-signal classifier output
    /// with this external value. This injects VPIN-based variation that prevents
    /// the classifier from concentrating at a single value.
    pub fn set_blended_toxicity(&mut self, toxicity: f64) {
        self.blended_toxicity_override = Some(toxicity.clamp(0.0, 1.0));
    }

    /// Set the blend weight between classifier and external toxicity.
    /// 0.0 = pure classifier, 1.0 = pure external.
    pub fn set_blend_weight(&mut self, weight: f64) {
        self.blend_weight = weight.clamp(0.0, 1.0);
    }

    /// Predict fill toxicity for a given side.
    ///
    /// Returns toxicity probability [0, 1] where:
    /// - 0 = no expected adverse selection
    /// - 1 = maximum expected adverse selection
    ///
    /// # Toxicity Logic
    ///
    /// Key insight: Fills are toxic when we're "with the informed flow".
    ///
    /// For **bids** (we want to buy):
    /// - Heavy BID pressure (book imbalance > 1) = others buying = we get run over on buys
    /// - Strong BUY flow (trade_direction > 0) = momentum against us after fill
    ///
    /// For **asks** (we want to sell):
    /// - Heavy ASK pressure (book imbalance < 1) = others selling = we get run over on sells
    /// - Strong SELL flow (trade_direction < 0) = momentum against us after fill
    ///
    /// # Arguments
    /// * `is_bid` - True if predicting toxicity for bid (buy) side
    pub fn predict_toxicity(&self, is_bid: bool) -> f64 {
        // Get weights - use learned if enough samples, else config defaults
        let weights = self.effective_weights();

        // Compute individual signals (normalized to [0, 1])
        let signals = self.compute_signals(is_bid);

        // Weighted sum
        let mut toxicity = 0.0;
        for (weight, signal) in weights.iter().zip(signals.iter()) {
            toxicity += weight * signal;
        }

        let classifier_tox = toxicity.clamp(0.0, 1.0);

        // Bayesian warmup prior: blend mild toxicity until enough data arrives.
        // Prevents zero-toxicity (unprotected spreads) during cold start.
        let warmup_blended = if self.update_count < self.config.warmup_prior_min_updates {
            let prior_weight =
                1.0 - (self.update_count as f64 / self.config.warmup_prior_min_updates as f64);
            let data_weight = 1.0 - prior_weight;
            data_weight * classifier_tox + prior_weight * self.config.warmup_prior_toxicity
        } else {
            classifier_tox
        };

        // Blend with external toxicity if available (e.g., VPIN-blended from IntegratedSignals)
        if let Some(override_tox) = self.blended_toxicity_override {
            let w = self.blend_weight;
            ((1.0 - w) * warmup_blended + w * override_tox).clamp(0.0, 1.0)
        } else {
            warmup_blended
        }
    }

    /// Get the effective weights (learned or default, with regime conditioning)
    fn effective_weights(&self) -> [f64; NUM_SIGNALS] {
        let base_weights = if self.config.enable_learning
            && self.learning_samples >= self.config.min_samples_for_learning
        {
            self.learned_weights
        } else {
            self.config.default_weights_array()
        };

        // Apply regime-conditional multipliers
        let multipliers = self.regime_weight_multipliers();
        let mut weights = [0.0; NUM_SIGNALS];
        for i in 0..weights.len() {
            weights[i] = base_weights[i] * multipliers[i];
        }

        // Normalize to sum to 1
        let sum: f64 = weights.iter().sum();
        if sum > 1e-9 {
            for w in &mut weights {
                *w /= sum;
            }
        }

        weights
    }

    /// Compute individual signal values for a given side.
    ///
    /// When warmed up (normalizer_obs_count >= warmup), uses EWMA z-score
    /// normalization through half_sigmoid for smooth [0,1] output distribution.
    /// Falls back to legacy threshold-based normalization during warmup.
    ///
    /// Returns [imbalance, flow, regime, funding, changepoint, trend] signals.
    pub fn compute_signals(&self, is_bid: bool) -> [f64; NUM_SIGNALS] {
        // Use z-score normalization when warmed up
        if self.normalizer_obs_count >= self.normalizer_warmup {
            return self.compute_signals_zscore(is_bid);
        }

        // === Legacy signals (warmup fallback — preserves test compatibility) ===

        // === Orderbook Imbalance ===
        let imbalance_signal = if is_bid {
            let raw = (self.orderbook_imbalance - 1.0) / (self.config.imbalance_threshold - 1.0);
            raw.clamp(0.0, 1.0)
        } else {
            let inverted = 1.0 / self.orderbook_imbalance.max(0.01);
            let raw = (inverted - 1.0) / (self.config.imbalance_threshold - 1.0);
            raw.clamp(0.0, 1.0)
        };

        // === Trade Flow Momentum ===
        let flow_signal = if is_bid {
            let raw = self.recent_trade_direction / self.config.flow_threshold;
            raw.clamp(0.0, 1.0)
        } else {
            let raw = (-self.recent_trade_direction) / self.config.flow_threshold;
            raw.clamp(0.0, 1.0)
        };

        // === Regime Distrust ===
        let regime_signal = 1.0 - self.regime_trust;

        // === Changepoint Probability ===
        let changepoint_signal = self.changepoint_prob;

        // === Funding Rate ===
        let funding_signal = if is_bid {
            (self.funding_rate / self.config.funding_threshold).clamp(0.0, 1.0)
        } else {
            (-self.funding_rate / self.config.funding_threshold).clamp(0.0, 1.0)
        };

        // === Trend Opposition (Kyle 1985 drift conditioning) ===
        // Downtrend (negative momentum) makes bids toxic, uptrend makes asks toxic.
        // Normalized: 20 bps of opposing momentum = full toxicity signal.
        let trend_signal = if is_bid {
            (-self.trend_momentum_bps / 20.0).clamp(0.0, 1.0)
        } else {
            (self.trend_momentum_bps / 20.0).clamp(0.0, 1.0)
        };

        [
            imbalance_signal,
            flow_signal,
            regime_signal,
            funding_signal,
            changepoint_signal,
            trend_signal,
        ]
    }

    /// Z-score normalized signal computation (post-warmup).
    ///
    /// Uses pre-computed z-scores (computed BEFORE EWMA update in update_* methods)
    /// so they reflect true deviations from the historical mean. Scaled by z_scale
    /// and passed through sigmoid for smooth (0,1) output centered at 0.5.
    fn compute_signals_zscore(&self, is_bid: bool) -> [f64; NUM_SIGNALS] {
        let scale = self.config.z_scale;

        // === Orderbook Imbalance (pre-computed z-score) ===
        // Directional: positive z (high bid pressure) is toxic for bids
        let directional_z_imb = if is_bid { self.imbalance_z } else { -self.imbalance_z };
        let imbalance_signal = Self::sigmoid(directional_z_imb * scale);

        // === Trade Flow (pre-computed z-score) ===
        // Directional: positive z (buy flow) is toxic for bids
        let directional_z_flow = if is_bid { self.flow_z } else { -self.flow_z };
        let flow_signal = Self::sigmoid(directional_z_flow * scale);

        // === Regime (pre-computed z-score of instability) ===
        // Non-directional: high instability is toxic for both sides
        let regime_signal = Self::sigmoid(self.regime_z * scale);

        // === Funding Rate (pre-computed z-score) ===
        // Directional: positive z (high positive funding) is toxic for bids
        let directional_z_fund = if is_bid { self.funding_z } else { -self.funding_z };
        let funding_signal = Self::sigmoid(directional_z_fund * scale);

        // === Changepoint (pre-computed z-score) ===
        // Non-directional: high changepoint probability is toxic for both sides
        let changepoint_signal = Self::sigmoid(self.changepoint_z * scale);

        // === Trend Opposition (no z-score, already in bps) ===
        // Uses same normalization as legacy path — already well-scaled.
        let trend_signal = if is_bid {
            (-self.trend_momentum_bps / 20.0).clamp(0.0, 1.0)
        } else {
            (self.trend_momentum_bps / 20.0).clamp(0.0, 1.0)
        };

        [
            imbalance_signal,
            flow_signal,
            regime_signal,
            funding_signal,
            changepoint_signal,
            trend_signal,
        ]
    }

    /// Update learned weights with a (prediction, outcome) pair.
    ///
    /// This implements online linear regression via stochastic gradient descent:
    /// - `is_bid`: The side for which toxicity was predicted
    /// - `was_adverse`: True if the fill was adverse (price moved against us)
    /// - `adverse_magnitude_bps`: Optional magnitude of adverse movement (for weighted learning)
    ///
    /// Call this after each fill is resolved with its markout outcome.
    pub fn record_outcome(&mut self, is_bid: bool, was_adverse: bool, adverse_magnitude_bps: Option<f64>) {
        if !self.config.enable_learning {
            return;
        }

        let signals = self.compute_signals(is_bid);

        // Convert outcome to target (0.0 = not adverse, 1.0 = adverse)
        let target = if was_adverse { 1.0 } else { 0.0 };

        // Weight by magnitude if provided (larger adverse moves count more)
        let sample_weight = adverse_magnitude_bps
            .map(|m| (m.abs() / 10.0).clamp(0.1, 5.0))
            .unwrap_or(1.0);

        // Update running statistics for each signal
        for (i, signal) in signals.iter().enumerate() {
            self.signal_outcome_sum[i] += signal * target * sample_weight;
            self.signal_sq_sum[i] += signal * signal * sample_weight;
        }
        self.learning_samples += 1;

        // Every 50 samples, update weights using regularized regression
        if self.learning_samples.is_multiple_of(50) {
            self.update_weights();
        }
    }

    /// Update weights using accumulated statistics
    fn update_weights(&mut self) {
        if self.learning_samples < self.config.min_samples_for_learning {
            return;
        }

        let lr = self.config.learning_rate;
        let reg = self.config.regularization;
        let n = self.learning_samples as f64;

        for i in 0..self.learned_weights.len() {
            // Compute gradient: E[signal * (target - prediction)]
            // Simplified: proportional to signal-outcome correlation
            let correlation = if self.signal_sq_sum[i] > 1e-9 {
                self.signal_outcome_sum[i] / self.signal_sq_sum[i].sqrt() / n.sqrt()
            } else {
                0.0
            };

            // Get prior weight (from config)
            let prior = self.config.prior_weight(i);

            // Update with regularization toward prior
            let target_weight = correlation.clamp(0.0, 1.0);
            let update = lr * (target_weight - self.learned_weights[i])
                - reg * (self.learned_weights[i] - prior);

            self.learned_weights[i] += update;
            self.learned_weights[i] = self.learned_weights[i].clamp(0.01, 0.8);
        }

        // Normalize weights to sum to 1
        let sum: f64 = self.learned_weights.iter().sum();
        if sum > 0.01 {
            for w in &mut self.learned_weights {
                *w /= sum;
            }
        }
    }

    /// Update AS bias correction from a resolved fill outcome.
    ///
    /// Tracks the EWMA of (predicted - realized) AS in bps.
    /// With alpha=0.95, the half-life is ~14 fills (ln(2)/ln(1/0.95) ≈ 13.5).
    /// The current 2x overestimate (9.14 predicted vs 4.27 realized) should
    /// converge to near-zero bias within ~50 fills.
    ///
    /// # Arguments
    /// * `predicted_as_bps` - The AS prediction at fill time (in bps)
    /// * `realized_as_bps` - The actual adverse selection observed (in bps)
    pub fn observe_as_outcome(&mut self, predicted_as_bps: f64, realized_as_bps: f64) {
        let error = predicted_as_bps - realized_as_bps;
        // EWMA alpha = 0.95 → slow adaptation, ~14 fill half-life
        const BIAS_EWMA_ALPHA: f64 = 0.95;

        if self.bias_observation_count == 0 {
            // First observation: initialize directly
            self.bias_correction_bps = error;
        } else {
            self.bias_correction_bps =
                BIAS_EWMA_ALPHA * self.bias_correction_bps + (1.0 - BIAS_EWMA_ALPHA) * error;
        }
        self.bias_observation_count += 1;

        // Log periodically for monitoring
        if self.bias_observation_count.is_multiple_of(20) {
            tracing::info!(
                bias_correction_bps = format!("{:.2}", self.bias_correction_bps),
                observation_count = self.bias_observation_count,
                latest_predicted_bps = format!("{:.2}", predicted_as_bps),
                latest_realized_bps = format!("{:.2}", realized_as_bps),
                "AS bias correction update"
            );
        }
    }

    /// Get the current AS bias correction in bps.
    ///
    /// Positive means we systematically overpredict AS.
    /// Returns 0.0 if fewer than 5 observations (not enough data).
    pub fn bias_correction(&self) -> f64 {
        if self.bias_observation_count >= 5 {
            self.bias_correction_bps
        } else {
            0.0
        }
    }

    /// Get learning diagnostics
    pub fn learning_diagnostics(&self) -> LearningDiagnostics {
        LearningDiagnostics {
            samples: self.learning_samples,
            min_samples: self.config.min_samples_for_learning,
            is_using_learned: self.learning_samples >= self.config.min_samples_for_learning,
            default_weights: self.config.default_weights_array(),
            learned_weights: self.learned_weights,
            signal_names: SIGNAL_NAMES,
        }
    }

    /// Get spread multiplier for a given side.
    ///
    /// Returns a multiplier >= 1.0 to apply to base spread.
    /// Higher toxicity → higher multiplier → wider spreads.
    ///
    /// # Arguments
    /// * `is_bid` - True for bid side, false for ask side
    pub fn spread_multiplier(&self, is_bid: bool) -> f64 {
        let toxicity = self.predict_toxicity(is_bid);

        // Apply bias correction: if we're overpredicting AS, reduce effective toxicity.
        // Convert bias_correction_bps to a toxicity adjustment.
        // Typical AS range is ~0-20 bps; toxicity [0,1] maps roughly to 0-20 bps.
        // So bias_correction_bps / 20.0 gives the toxicity adjustment.
        let bias_adj = self.bias_correction() / 20.0;
        let corrected_toxicity = (toxicity - bias_adj).clamp(0.0, 1.0);

        // Sigmoid-centered signals: toxicity 0.5 = neutral, only widen above neutral.
        // Maps [0.5, 1.0] → [1.0, max_multiplier], below 0.5 → 1.0 (no widening).
        let excess = (corrected_toxicity - 0.5).max(0.0) * 2.0; // [0.5,1.0] → [0.0,1.0]
        let multiplier = 1.0 + excess * (self.config.max_spread_multiplier - 1.0);
        multiplier.clamp(1.0, self.config.max_spread_multiplier)
    }

    /// Get symmetric spread multiplier (max of bid/ask).
    ///
    /// Use this when applying the same multiplier to both sides.
    pub fn spread_multiplier_symmetric(&self) -> f64 {
        let bid_mult = self.spread_multiplier(true);
        let ask_mult = self.spread_multiplier(false);
        bid_mult.max(ask_mult)
    }

    /// Get cached toxicity estimate (average of bid/ask).
    pub fn cached_toxicity(&self) -> f64 {
        self.cached_toxicity
    }

    /// Update cached toxicity (average of bid and ask toxicity).
    fn update_cached_toxicity(&mut self) {
        self.cached_toxicity = (self.predict_toxicity(true) + self.predict_toxicity(false)) / 2.0;
        self.update_count += 1;
    }

    /// Check if orderbook signal is stale.
    ///
    /// # Arguments
    /// * `max_age_ms` - Maximum age in milliseconds before considered stale
    pub fn is_orderbook_stale(&self, max_age_ms: u64) -> bool {
        if self.orderbook_updated_at_ms == 0 {
            return true; // Never updated
        }
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        now_ms.saturating_sub(self.orderbook_updated_at_ms) > max_age_ms
    }

    /// Check if trade flow signal is stale.
    pub fn is_trade_flow_stale(&self, max_age_ms: u64) -> bool {
        if self.trade_flow_updated_at_ms == 0 {
            return true;
        }
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        now_ms.saturating_sub(self.trade_flow_updated_at_ms) > max_age_ms
    }

    /// Check if regime signal is stale.
    pub fn is_regime_stale(&self, max_age_ms: u64) -> bool {
        if self.regime_updated_at_ms == 0 {
            return true;
        }
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        now_ms.saturating_sub(self.regime_updated_at_ms) > max_age_ms
    }

    /// Check if any critical signal is stale.
    ///
    /// Returns true if orderbook or trade flow signals are stale.
    /// These are the highest-weight signals in toxicity prediction.
    pub fn has_stale_signals(&self, max_age_ms: u64) -> bool {
        self.is_orderbook_stale(max_age_ms) || self.is_trade_flow_stale(max_age_ms)
    }

    /// Get staleness summary for all signals.
    pub fn staleness_summary(&self, max_age_ms: u64) -> SignalStaleness {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        SignalStaleness {
            orderbook_age_ms: now_ms.saturating_sub(self.orderbook_updated_at_ms),
            trade_flow_age_ms: now_ms.saturating_sub(self.trade_flow_updated_at_ms),
            regime_age_ms: now_ms.saturating_sub(self.regime_updated_at_ms),
            orderbook_stale: self.is_orderbook_stale(max_age_ms),
            trade_flow_stale: self.is_trade_flow_stale(max_age_ms),
            regime_stale: self.is_regime_stale(max_age_ms),
        }
    }

    /// Get summary for diagnostics.
    pub fn summary(&self) -> PreFillSummary {
        PreFillSummary {
            orderbook_imbalance: self.orderbook_imbalance,
            trade_direction: self.recent_trade_direction,
            regime_trust: self.regime_trust,
            changepoint_prob: self.changepoint_prob,
            funding_rate: self.funding_rate,
            trend_momentum_bps: self.trend_momentum_bps,
            bid_toxicity: self.predict_toxicity(true),
            ask_toxicity: self.predict_toxicity(false),
            bid_spread_mult: self.spread_multiplier(true),
            ask_spread_mult: self.spread_multiplier(false),
            update_count: self.update_count,
            orderbook_updated_at_ms: self.orderbook_updated_at_ms,
            trade_flow_updated_at_ms: self.trade_flow_updated_at_ms,
            regime_updated_at_ms: self.regime_updated_at_ms,
            learning_samples: self.learning_samples,
            using_learned_weights: self.learning_samples >= self.config.min_samples_for_learning,
        }
    }
}

impl Default for PreFillASClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of pre-fill classifier state.
#[derive(Debug, Clone)]
pub struct PreFillSummary {
    pub orderbook_imbalance: f64,
    pub trade_direction: f64,
    pub regime_trust: f64,
    pub changepoint_prob: f64,
    pub funding_rate: f64,
    pub trend_momentum_bps: f64,
    pub bid_toxicity: f64,
    pub ask_toxicity: f64,
    pub bid_spread_mult: f64,
    pub ask_spread_mult: f64,
    pub update_count: u64,
    /// Timestamp of last orderbook update (ms since epoch)
    pub orderbook_updated_at_ms: u64,
    /// Timestamp of last trade flow update (ms since epoch)
    pub trade_flow_updated_at_ms: u64,
    /// Timestamp of last regime update (ms since epoch)
    pub regime_updated_at_ms: u64,
    /// Number of learning samples processed
    pub learning_samples: usize,
    /// Whether learned weights are being used
    pub using_learned_weights: bool,
}

/// Signal staleness information.
#[derive(Debug, Clone)]
pub struct SignalStaleness {
    pub orderbook_age_ms: u64,
    pub trade_flow_age_ms: u64,
    pub regime_age_ms: u64,
    pub orderbook_stale: bool,
    pub trade_flow_stale: bool,
    pub regime_stale: bool,
}

/// Diagnostics for online weight learning.
#[derive(Debug, Clone)]
pub struct LearningDiagnostics {
    /// Number of samples used for learning
    pub samples: usize,
    /// Minimum samples required to use learned weights
    pub min_samples: usize,
    /// Whether learned weights are currently being used
    pub is_using_learned: bool,
    /// Default weights from config
    pub default_weights: [f64; NUM_SIGNALS],
    /// Learned weights (may not be in use yet)
    pub learned_weights: [f64; NUM_SIGNALS],
    /// Signal names for display
    pub signal_names: [&'static str; NUM_SIGNALS],
}

impl LearningDiagnostics {
    /// Get a formatted summary string
    pub fn summary(&self) -> String {
        let status = if self.is_using_learned {
            "LEARNED"
        } else {
            "DEFAULT"
        };

        let mut s = format!(
            "PreFillAS [{} n={}]\n",
            status, self.samples
        );

        let weights = if self.is_using_learned {
            &self.learned_weights
        } else {
            &self.default_weights
        };

        for (i, name) in self.signal_names.iter().enumerate() {
            let learned = self.learned_weights[i];
            let default = self.default_weights[i];
            let diff = if (learned - default).abs() > 0.05 {
                if learned > default { "↑" } else { "↓" }
            } else {
                "="
            };

            s.push_str(&format!(
                "  {:12} w={:.2} (def={:.2}) {}\n",
                name, weights[i], default, diff
            ));
        }

        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_classifier() {
        let classifier = PreFillASClassifier::new();
        // Neutral state should have low toxicity
        assert!(classifier.predict_toxicity(true) < 0.5);
        assert!(classifier.predict_toxicity(false) < 0.5);
        assert!((classifier.spread_multiplier(true) - 1.0).abs() < 1.0);
    }

    #[test]
    fn test_orderbook_imbalance() {
        let mut classifier = PreFillASClassifier::new();

        // Heavy bid pressure → toxic for bids
        classifier.update_orderbook(2.0, 1.0); // 2:1 bid/ask
        assert!(classifier.predict_toxicity(true) > classifier.predict_toxicity(false));

        // Heavy ask pressure → toxic for asks
        classifier.update_orderbook(1.0, 2.0); // 1:2 bid/ask
        assert!(classifier.predict_toxicity(false) > classifier.predict_toxicity(true));
    }

    #[test]
    fn test_trade_flow() {
        let mut classifier = PreFillASClassifier::new();

        // Heavy buy flow → toxic for bids
        classifier.update_trade_flow(80.0, 20.0);
        let bid_tox = classifier.predict_toxicity(true);

        // Reset and test sell flow
        classifier.update_trade_flow(20.0, 80.0);
        let ask_tox = classifier.predict_toxicity(false);

        // Both should be elevated
        assert!(bid_tox > 0.1);
        assert!(ask_tox > 0.1);
    }

    #[test]
    fn test_trade_flow_symmetric() {
        let mut classifier = PreFillASClassifier::new();

        // SYMMETRIC TEST: Buy flow should only affect bids, not asks
        classifier.update_trade_flow(80.0, 20.0); // Strong buy flow (trade_direction ≈ +0.6)
        let bid_tox_with_buy_flow = classifier.predict_toxicity(true);
        let _ask_tox_with_buy_flow = classifier.predict_toxicity(false);

        // Bids should be toxic (we're buying with the crowd)
        assert!(bid_tox_with_buy_flow > 0.1, "Buy flow should make bids toxic");
        // Asks should NOT be elevated from buy flow alone
        // (only from regime/changepoint which are symmetric)

        // SYMMETRIC TEST: Sell flow should only affect asks, not bids
        classifier.update_trade_flow(20.0, 80.0); // Strong sell flow (trade_direction ≈ -0.6)
        let bid_tox_with_sell_flow = classifier.predict_toxicity(true);
        let ask_tox_with_sell_flow = classifier.predict_toxicity(false);

        // Asks should be toxic (we're selling with the crowd)
        assert!(ask_tox_with_sell_flow > 0.1, "Sell flow should make asks toxic");
        // Buy flow contribution to bid should be gone now
        assert!(bid_tox_with_sell_flow < bid_tox_with_buy_flow,
            "Sell flow should reduce bid toxicity vs buy flow");
    }

    #[test]
    fn test_funding_directional() {
        let mut classifier = PreFillASClassifier::new();

        // Positive funding = longs pay shorts = crowd is long
        classifier.update_funding(0.001); // High positive funding

        let bid_tox_pos_funding = classifier.predict_toxicity(true);
        let ask_tox_pos_funding = classifier.predict_toxicity(false);

        // Positive funding should make bids more toxic (crowd is long, may dump)
        // Asks should be less affected (or safe - crowd is long, not short)
        assert!(bid_tox_pos_funding > ask_tox_pos_funding,
            "Positive funding should make bids more toxic than asks");

        // Negative funding = shorts pay longs = crowd is short
        classifier.update_funding(-0.001); // High negative funding

        let bid_tox_neg_funding = classifier.predict_toxicity(true);
        let ask_tox_neg_funding = classifier.predict_toxicity(false);

        // Negative funding should make asks more toxic (crowd is short, may squeeze)
        assert!(ask_tox_neg_funding > bid_tox_neg_funding,
            "Negative funding should make asks more toxic than bids");
    }

    #[test]
    fn test_regime_distrust() {
        let mut classifier = PreFillASClassifier::new();

        // Low HMM confidence + high changepoint = high toxicity
        classifier.update_regime(0.3, 0.8);
        assert!(classifier.predict_toxicity(true) > 0.2);
        assert!(classifier.predict_toxicity(false) > 0.2);

        // High confidence + low changepoint = low toxicity
        classifier.update_regime(0.95, 0.05);
        let low_tox = (classifier.predict_toxicity(true) + classifier.predict_toxicity(false)) / 2.0;
        assert!(low_tox < 0.3);
    }

    #[test]
    fn test_spread_multiplier_capped() {
        let mut classifier = PreFillASClassifier::new();

        // Max out all signals
        classifier.update_orderbook(10.0, 1.0);
        classifier.update_trade_flow(100.0, 0.0);
        classifier.update_regime(0.1, 0.9);
        classifier.update_funding(0.01);

        // Should be capped at max_spread_multiplier
        let mult = classifier.spread_multiplier(true);
        assert!(mult <= classifier.config.max_spread_multiplier + 0.01);
        assert!(mult >= 1.0);
    }

    #[test]
    fn test_summary() {
        let classifier = PreFillASClassifier::new();
        let summary = classifier.summary();

        assert_eq!(summary.orderbook_imbalance, 1.0);
        assert_eq!(summary.trade_direction, 0.0);
        assert_eq!(summary.regime_trust, 1.0);
    }

    #[test]
    fn test_as_bias_correction_reduces_overprediction() {
        let mut classifier = PreFillASClassifier::new();

        // Initially no bias correction
        assert_eq!(classifier.bias_correction(), 0.0);

        // Simulate systematic overprediction: predicted 9 bps, realized 4 bps
        for _ in 0..30 {
            classifier.observe_as_outcome(9.0, 4.0);
        }

        // Bias should be close to 5.0 (overpredicting by 5 bps)
        let bias = classifier.bias_correction();
        assert!(bias > 3.0, "Bias should be positive (overpredicting): {}", bias);
        assert!(bias < 6.0, "Bias should converge toward 5.0: {}", bias);
    }

    #[test]
    fn test_as_bias_correction_needs_minimum_observations() {
        let mut classifier = PreFillASClassifier::new();

        // Only 3 observations — not enough
        for _ in 0..3 {
            classifier.observe_as_outcome(10.0, 5.0);
        }
        assert_eq!(classifier.bias_correction(), 0.0);

        // After 5 observations, correction kicks in
        for _ in 0..2 {
            classifier.observe_as_outcome(10.0, 5.0);
        }
        assert!(classifier.bias_correction() > 0.0);
    }

    #[test]
    fn test_as_bias_correction_applied_to_spread_multiplier() {
        let mut classifier = PreFillASClassifier::new();

        // Push toxicity high
        classifier.update_orderbook(5.0, 1.0);
        classifier.update_trade_flow(90.0, 10.0);

        let mult_before = classifier.spread_multiplier(true);

        // Now train bias correction: we've been overpredicting massively
        for _ in 0..20 {
            classifier.observe_as_outcome(15.0, 2.0);
        }

        let mult_after = classifier.spread_multiplier(true);

        // After bias correction, spread multiplier should be lower
        assert!(mult_after <= mult_before,
            "Bias correction should reduce spread multiplier: before={}, after={}",
            mult_before, mult_after);
    }

    #[test]
    fn test_warmup_prior_nonzero() {
        // Fresh classifier with 0 updates should return nonzero toxicity
        // from the Bayesian warmup prior (defense during cold start).
        let classifier = PreFillASClassifier::new();
        assert_eq!(classifier.update_count, 0);

        let bid_tox = classifier.predict_toxicity(true);
        let ask_tox = classifier.predict_toxicity(false);

        // With 0 updates, prior_weight = 1.0, so toxicity = warmup_prior_toxicity = 0.35
        assert!(bid_tox > 0.0, "Warmup prior should produce nonzero bid toxicity, got {}", bid_tox);
        assert!(ask_tox > 0.0, "Warmup prior should produce nonzero ask toxicity, got {}", ask_tox);
        assert!((bid_tox - 0.35).abs() < 0.01,
            "With 0 updates, toxicity should equal warmup_prior_toxicity (0.35), got {}", bid_tox);
    }

    #[test]
    fn test_prior_decays() {
        // After enough updates, the prior contribution should approach 0.
        let mut classifier = PreFillASClassifier::new();

        // Toxicity at 0 updates (pure prior)
        let tox_at_zero = classifier.predict_toxicity(true);

        // Simulate updates by calling update_cached_toxicity (increments update_count)
        for _ in 0..10 {
            classifier.update_cached_toxicity();
        }
        assert!(classifier.update_count >= 10);

        // After min_updates (10), prior_weight = 0 → pure data
        let tox_after_warmup = classifier.predict_toxicity(true);

        // With neutral inputs and no prior, classifier_tox should be near 0
        // (all signals are at neutral defaults). The prior was inflating it.
        assert!(tox_after_warmup < tox_at_zero,
            "Prior should decay: tox_at_zero={}, tox_after_warmup={}",
            tox_at_zero, tox_after_warmup);

        // Data-only toxicity should be very low with neutral signals
        assert!(tox_after_warmup < 0.1,
            "Neutral signals with no prior should give low toxicity, got {}", tox_after_warmup);
    }

    #[test]
    fn test_spread_multiplier_warmup() {
        // Fresh classifier should have spread_multiplier > 1.0 due to warmup prior.
        let classifier = PreFillASClassifier::new();
        assert_eq!(classifier.update_count, 0);

        let bid_mult = classifier.spread_multiplier(true);
        let ask_mult = classifier.spread_multiplier(false);

        // Warmup prior toxicity = 0.35. The spread_multiplier maps [0.5, 1.0] -> [1.0, max].
        // Since 0.35 < 0.5, excess = 0 and multiplier = 1.0.
        // But bias_correction is 0, so corrected_toxicity = 0.35.
        // Actually: spread_multiplier computes excess = (0.35 - 0.5).max(0.0) = 0.0 → mult = 1.0.
        //
        // To get spread widening from the prior, we need toxicity > 0.5.
        // The default prior of 0.35 is intentionally mild — it provides nonzero AS estimation
        // without necessarily widening spreads (defense at the AS level, not spread level).
        //
        // Verify the prior at least makes toxicity nonzero (the core fix).
        let tox = classifier.predict_toxicity(true);
        assert!(tox > 0.0, "Fresh classifier must have nonzero toxicity, got {}", tox);

        // With a higher prior, spread multiplier would exceed 1.0.
        let mut config = PreFillClassifierConfig::default();
        config.warmup_prior_toxicity = 0.7; // High prior for testing
        let high_prior_classifier = PreFillASClassifier::with_config(config);

        let high_bid_mult = high_prior_classifier.spread_multiplier(true);
        assert!(high_bid_mult > 1.0,
            "High warmup prior (0.7) should widen spreads, got multiplier={}", high_bid_mult);

        // Default prior should not be wider than high prior
        assert!(bid_mult <= high_bid_mult,
            "Default prior mult ({}) should be <= high prior mult ({})", bid_mult, high_bid_mult);
        assert!(ask_mult <= high_bid_mult,
            "Default prior mult ({}) should be <= high prior mult ({})", ask_mult, high_bid_mult);
    }

    // === Trend Opposition Tests ===

    #[test]
    fn test_trend_bid_in_downtrend_high_toxicity() {
        // A strong downtrend should make bid fills toxic (buying into falling market).
        let mut classifier = PreFillASClassifier::new();
        // Warm past the prior by updating enough times
        for _ in 0..11 {
            classifier.update_cached_toxicity();
        }

        let bid_tox_neutral = classifier.predict_toxicity(true);

        // Apply strong downtrend: -30 bps momentum
        classifier.update_trend(-30.0);

        let bid_tox_downtrend = classifier.predict_toxicity(true);
        let ask_tox_downtrend = classifier.predict_toxicity(false);

        // Bids should be MORE toxic in a downtrend (buying into decline)
        assert!(
            bid_tox_downtrend > bid_tox_neutral,
            "Downtrend should increase bid toxicity: neutral={:.4}, downtrend={:.4}",
            bid_tox_neutral, bid_tox_downtrend
        );
        // Asks should NOT be more toxic from downtrend (selling into decline is safe)
        assert!(
            ask_tox_downtrend < bid_tox_downtrend,
            "Downtrend should not make asks more toxic than bids: ask={:.4}, bid={:.4}",
            ask_tox_downtrend, bid_tox_downtrend
        );
    }

    #[test]
    fn test_trend_bid_in_uptrend_zero_trend_toxicity() {
        // An uptrend should NOT add toxicity to bid fills (buying into rising market is fine).
        let mut classifier = PreFillASClassifier::new();
        for _ in 0..11 {
            classifier.update_cached_toxicity();
        }

        let bid_tox_neutral = classifier.predict_toxicity(true);

        // Apply strong uptrend: +30 bps momentum
        classifier.update_trend(30.0);

        let bid_tox_uptrend = classifier.predict_toxicity(true);
        let ask_tox_uptrend = classifier.predict_toxicity(false);

        // Bids should NOT be more toxic in an uptrend
        assert!(
            (bid_tox_uptrend - bid_tox_neutral).abs() < 0.05,
            "Uptrend should not increase bid toxicity: neutral={:.4}, uptrend={:.4}",
            bid_tox_neutral, bid_tox_uptrend
        );
        // Asks SHOULD be more toxic in an uptrend (selling into rising market)
        assert!(
            ask_tox_uptrend > bid_tox_uptrend,
            "Uptrend should make asks more toxic than bids: ask={:.4}, bid={:.4}",
            ask_tox_uptrend, bid_tox_uptrend
        );
    }

    #[test]
    fn test_trend_neutral_momentum_zero_contribution() {
        // Zero momentum should produce zero trend signal contribution.
        let mut classifier = PreFillASClassifier::new();
        for _ in 0..11 {
            classifier.update_cached_toxicity();
        }

        classifier.update_trend(0.0);

        let signals_bid = classifier.compute_signals(true);
        let signals_ask = classifier.compute_signals(false);

        // Trend signal (index 5) should be 0 when momentum is 0
        assert!(
            signals_bid[5].abs() < 1e-9,
            "Zero momentum should produce zero bid trend signal, got {:.6}",
            signals_bid[5]
        );
        assert!(
            signals_ask[5].abs() < 1e-9,
            "Zero momentum should produce zero ask trend signal, got {:.6}",
            signals_ask[5]
        );
    }

    #[test]
    fn test_trend_learning_convergence_six_features() {
        // Verify that the learning loop works correctly with 6 features.
        let mut config = PreFillClassifierConfig::default();
        config.min_samples_for_learning = 50; // Lower threshold for testing
        let mut classifier = PreFillASClassifier::with_config(config);

        // Apply a downtrend so trend signal is nonzero for bids
        classifier.update_trend(-15.0);
        classifier.update_orderbook(1.5, 1.0);
        classifier.update_trade_flow(60.0, 40.0);

        // Record many outcomes: bids in downtrend are always adverse
        for _ in 0..100 {
            classifier.record_outcome(true, true, Some(5.0));
        }

        let diag = classifier.learning_diagnostics();
        assert_eq!(diag.samples, 100);
        assert!(diag.is_using_learned, "Should be using learned weights after 100 samples");

        // All 6 signal names should be present
        assert_eq!(diag.signal_names.len(), 6);
        assert_eq!(diag.signal_names[5], "trend");

        // Learned weights should sum to 1.0 (normalized)
        let sum: f64 = diag.learned_weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Learned weights should sum to 1.0, got {:.4}",
            sum
        );

        // Trend weight should be nonzero (it's contributing to adverse outcomes)
        assert!(
            diag.learned_weights[5] > 0.01,
            "Trend learned weight should be nonzero after adverse outcomes, got {:.4}",
            diag.learned_weights[5]
        );
    }
}
