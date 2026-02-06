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
}

impl Default for PreFillClassifierConfig {
    fn default() -> Self {
        Self {
            imbalance_weight: 0.30,
            flow_weight: 0.25,
            regime_weight: 0.25,
            funding_weight: 0.10,
            changepoint_weight: 0.10,

            max_spread_multiplier: 3.0, // Cap at 3x base spread

            imbalance_threshold: 2.0,   // 2:1 bid/ask ratio
            flow_threshold: 0.7,        // 70% one-sided flow
            funding_threshold: 0.001,   // 0.1% per 8h (annualized ~110%)

            enable_learning: true,
            learning_rate: 0.01,
            min_samples_for_learning: 500,
            regularization: 0.1,
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
    /// Learned weights [imbalance, flow, regime, funding, changepoint]
    learned_weights: [f64; 5],

    /// Running sum of (signal_i * outcome) for each signal
    signal_outcome_sum: [f64; 5],

    /// Running sum of signal_i^2 for each signal
    signal_sq_sum: [f64; 5],

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
}

impl PreFillASClassifier {
    /// Create a new pre-fill classifier with default config.
    pub fn new() -> Self {
        Self::with_config(PreFillClassifierConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: PreFillClassifierConfig) -> Self {
        // Initialize learned weights with config defaults
        let learned_weights = [
            config.imbalance_weight,
            config.flow_weight,
            config.regime_weight,
            config.funding_weight,
            config.changepoint_weight,
        ];

        Self {
            config,
            orderbook_imbalance: 1.0, // Neutral
            recent_trade_direction: 0.0, // Neutral
            regime_trust: 1.0, // High trust initially
            changepoint_prob: 0.0, // No changepoint
            funding_rate: 0.0, // Neutral funding
            cached_toxicity: 0.0,
            update_count: 0,
            orderbook_updated_at_ms: 0,
            trade_flow_updated_at_ms: 0,
            regime_updated_at_ms: 0,
            learned_weights,
            signal_outcome_sum: [0.0; 5],
            signal_sq_sum: [0.0; 5],
            learning_samples: 0,
            // Regime-conditional state
            current_regime: 1, // Start with Normal
            regime_probs: [0.1, 0.7, 0.15, 0.05], // Default prior
            // Blended toxicity override
            blended_toxicity_override: None,
            blend_weight: 0.5, // 50/50 blend by default
        }
    }

    // === Checkpoint persistence ===

    /// Extract learning state for checkpoint persistence.
    pub fn to_checkpoint(&self) -> crate::market_maker::checkpoint::PreFillCheckpoint {
        crate::market_maker::checkpoint::PreFillCheckpoint {
            learned_weights: self.learned_weights,
            signal_outcome_sum: self.signal_outcome_sum,
            signal_sq_sum: self.signal_sq_sum,
            learning_samples: self.learning_samples,
            regime_probs: self.regime_probs,
        }
    }

    /// Restore learning state from a checkpoint.
    ///
    /// Ephemeral state (cached signals, timestamps) stays at defaults —
    /// they repopulate within seconds from live data.
    pub fn restore_checkpoint(&mut self, cp: &crate::market_maker::checkpoint::PreFillCheckpoint) {
        self.learned_weights = cp.learned_weights;
        self.signal_outcome_sum = cp.signal_outcome_sum;
        self.signal_sq_sum = cp.signal_sq_sum;
        self.learning_samples = cp.learning_samples;
        self.regime_probs = cp.regime_probs;
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
        self.regime_trust = hmm_confidence.clamp(0.0, 1.0);
        self.changepoint_prob = changepoint_prob.clamp(0.0, 1.0);
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
    fn regime_weight_multipliers(&self) -> [f64; 5] {
        // Weight multipliers for [imbalance, flow, regime, funding, changepoint]
        // Soft-blend based on regime probabilities
        
        let regime_weights: [[f64; 5]; 4] = [
            // Low/Calm: imbalance matters, others matter less
            [1.5, 0.8, 0.7, 1.0, 0.5],
            // Normal: balanced
            [1.0, 1.0, 1.0, 1.0, 1.0],
            // High: flow and changepoint matter more
            [0.8, 1.3, 1.2, 0.9, 1.4],
            // Extreme/Cascade: changepoint and flow dominate
            [0.6, 1.5, 1.3, 0.8, 1.8],
        ];

        // Soft blend across regimes
        let mut multipliers = [0.0; 5];
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
        self.update_cached_toxicity();
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

        // Blend with external toxicity if available (e.g., VPIN-blended from IntegratedSignals)
        if let Some(override_tox) = self.blended_toxicity_override {
            let w = self.blend_weight;
            ((1.0 - w) * classifier_tox + w * override_tox).clamp(0.0, 1.0)
        } else {
            classifier_tox
        }
    }

    /// Get the effective weights (learned or default, with regime conditioning)
    fn effective_weights(&self) -> [f64; 5] {
        let base_weights = if self.config.enable_learning
            && self.learning_samples >= self.config.min_samples_for_learning
        {
            self.learned_weights
        } else {
            [
                self.config.imbalance_weight,
                self.config.flow_weight,
                self.config.regime_weight,
                self.config.funding_weight,
                self.config.changepoint_weight,
            ]
        };

        // Apply regime-conditional multipliers
        let multipliers = self.regime_weight_multipliers();
        let mut weights = [0.0; 5];
        for i in 0..5 {
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

    /// Compute individual signal values for a given side
    /// Returns [imbalance, flow, regime, funding, changepoint] signals
    pub fn compute_signals(&self, is_bid: bool) -> [f64; 5] {
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

        [
            imbalance_signal,
            flow_signal,
            regime_signal,
            funding_signal,
            changepoint_signal,
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
        for i in 0..5 {
            self.signal_outcome_sum[i] += signals[i] * target * sample_weight;
            self.signal_sq_sum[i] += signals[i] * signals[i] * sample_weight;
        }
        self.learning_samples += 1;

        // Every 50 samples, update weights using regularized regression
        if self.learning_samples % 50 == 0 {
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

        for i in 0..5 {
            // Compute gradient: E[signal * (target - prediction)]
            // Simplified: proportional to signal-outcome correlation
            let correlation = if self.signal_sq_sum[i] > 1e-9 {
                self.signal_outcome_sum[i] / self.signal_sq_sum[i].sqrt() / n.sqrt()
            } else {
                0.0
            };

            // Get prior weight (from config)
            let prior = match i {
                0 => self.config.imbalance_weight,
                1 => self.config.flow_weight,
                2 => self.config.regime_weight,
                3 => self.config.funding_weight,
                4 => self.config.changepoint_weight,
                _ => 0.2,
            };

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

    /// Get learning diagnostics
    pub fn learning_diagnostics(&self) -> LearningDiagnostics {
        let default_weights = [
            self.config.imbalance_weight,
            self.config.flow_weight,
            self.config.regime_weight,
            self.config.funding_weight,
            self.config.changepoint_weight,
        ];

        LearningDiagnostics {
            samples: self.learning_samples,
            min_samples: self.config.min_samples_for_learning,
            is_using_learned: self.learning_samples >= self.config.min_samples_for_learning,
            default_weights,
            learned_weights: self.learned_weights,
            signal_names: ["imbalance", "flow", "regime", "funding", "changepoint"],
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
        // Linear scaling: 0 toxicity = 1x, max toxicity = max_multiplier
        let multiplier = 1.0 + toxicity * (self.config.max_spread_multiplier - 1.0);
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
    pub default_weights: [f64; 5],
    /// Learned weights (may not be in use yet)
    pub learned_weights: [f64; 5],
    /// Signal names for display
    pub signal_names: [&'static str; 5],
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
        let ask_tox_with_buy_flow = classifier.predict_toxicity(false);

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
}
