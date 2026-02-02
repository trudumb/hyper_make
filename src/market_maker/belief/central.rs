//! CentralBeliefState - the single source of truth for all beliefs.
//!
//! This is the core struct that maintains all Bayesian posteriors and
//! provides unified access for consumers. It processes BeliefUpdate
//! messages and produces BeliefSnapshots.
//!
//! ## Thread Safety
//!
//! CentralBeliefState uses internal mutability with RwLock for thread-safe
//! access. Reads (via `snapshot()`) take a read lock, while updates take
//! a write lock.
//!
//! ## Processing Updates
//!
//! Updates can be processed in two ways:
//! 1. Synchronously via `update()` method
//! 2. Via a background processor that consumes from a channel
//!
//! ## Example
//!
//! ```ignore
//! let config = CentralBeliefConfig::default();
//! let beliefs = CentralBeliefState::new(config);
//!
//! // Process updates
//! beliefs.update(BeliefUpdate::PriceReturn {
//!     return_frac: 0.001,
//!     dt_secs: 1.0,
//!     timestamp_ms: now_ms(),
//! });
//!
//! // Read snapshot
//! let snapshot = beliefs.snapshot();
//! println!("Kappa: {}", snapshot.kappa.kappa_effective);
//! ```

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Instant;

use super::messages::{BeliefUpdate, PredictionLog, PredictionType};
use super::snapshot::{
    BeliefSnapshot, BeliefStats, CalibrationMetrics, CalibrationState, ChangepointBeliefs,
    ChangepointResult, ContinuationBeliefs, ContinuationSignals, DriftVolatilityBeliefs,
    EdgeBeliefs, KappaBeliefs, KappaComponents, RegimeBeliefs,
};
use super::Regime;

/// Configuration for CentralBeliefState.
#[derive(Debug, Clone)]
pub struct CentralBeliefConfig {
    // === Drift/Volatility Priors ===
    /// Prior standard deviation for drift
    pub drift_prior_sigma: f64,

    // === Kappa Priors ===
    /// Prior kappa value (fill intensity)
    pub kappa_prior: f64,
    /// Prior strength for kappa (pseudo-observations)
    pub kappa_prior_strength: f64,

    // === Warmup ===
    /// Minimum price observations before warmed up
    pub min_price_obs: u64,
    /// Minimum fills before warmed up
    pub min_fills: u64,
    /// Minimum observation time (seconds) before warmed up
    pub min_observation_time: f64,

    // === Decay ===
    /// Decay factor for non-stationarity (per 1000 observations)
    pub decay_factor: f64,

    // === Changepoint ===
    /// Hazard rate for BOCD (probability of changepoint per observation)
    pub changepoint_hazard: f64,
    /// Threshold for changepoint detection
    pub changepoint_threshold: f64,

    // === EWMA ===
    /// EWMA smoothing factor for kappa (0.9 = 90% previous, 10% new)
    pub kappa_ewma_alpha: f64,
}

impl Default for CentralBeliefConfig {
    fn default() -> Self {
        Self {
            drift_prior_sigma: 0.01,
            kappa_prior: 2000.0,
            kappa_prior_strength: 10.0,
            min_price_obs: 50,
            min_fills: 5,
            min_observation_time: 60.0,
            decay_factor: 0.999,
            changepoint_hazard: 1.0 / 250.0,
            changepoint_threshold: 0.7,
            kappa_ewma_alpha: 0.9,
        }
    }
}

impl CentralBeliefConfig {
    /// Config for HIP-3 DEX (thin books, slower fills).
    pub fn hip3() -> Self {
        Self {
            kappa_prior: 1500.0,
            kappa_prior_strength: 15.0,
            min_fills: 3,
            changepoint_hazard: 1.0 / 150.0, // More sensitive
            ..Default::default()
        }
    }

    /// Config for liquid CEX markets.
    pub fn liquid() -> Self {
        Self {
            kappa_prior: 2500.0,
            kappa_prior_strength: 5.0,
            min_observation_time: 30.0,
            ..Default::default()
        }
    }
}

/// Internal state (mutable, protected by RwLock).
struct InternalState {
    // === Drift/Volatility ===
    drift_sum: f64,
    drift_sum_sq: f64,
    drift_n: f64,
    drift_prior_mean: f64,
    drift_prior_precision: f64,

    // === Kappa (fill intensity) ===
    kappa_smoothed: f64,
    kappa_smoothed_initialized: bool,

    // Own-fill kappa (Bayesian)
    own_kappa_alpha: f64,
    own_kappa_beta: f64,
    own_kappa_n_fills: usize,

    // Book kappa (exponential decay regression)
    book_kappa: f64,
    book_kappa_r2: f64,

    // Robust kappa (Student-t)
    robust_kappa: f64,
    robust_kappa_ess: f64,

    // === Continuation (Beta-Binomial) ===
    continuation_alpha: f64,
    continuation_beta: f64,
    momentum_continuation: f64,
    trend_agreement: f64,
    trend_confidence: f64,

    // === Regime (HMM) ===
    regime_probs: [f64; 4],

    // === Changepoint (BOCD) ===
    changepoint_run_probs: Vec<f64>,
    changepoint_obs_count: usize,
    changepoint_consecutive_high: usize,

    // === Edge ===
    edge_sum: f64,
    edge_sum_sq: f64,
    edge_n: f64,
    edge_by_regime: [f64; 3],
    as_bias: f64,

    // === Calibration ===
    pending_predictions: HashMap<u64, StoredPrediction>,
    next_prediction_id: u64,
    fill_brier_sum: f64,
    fill_brier_n: usize,
    fill_base_rate_sum: f64,
    as_brier_sum: f64,
    as_brier_n: usize,
    as_base_rate_sum: f64,
    signal_mi: HashMap<String, f64>,

    // === Statistics ===
    n_price_obs: u64,
    n_fills: u64,
    n_market_trades: u64,
    total_time: f64,
    last_update_ms: u64,
}

/// Stored prediction awaiting outcome.
#[derive(Debug, Clone)]
struct StoredPrediction {
    prediction_type: PredictionType,
    predicted_prob: f64,
    _confidence: f64,
    _regime: usize,
    _timestamp_ms: u64,
}

impl Default for InternalState {
    fn default() -> Self {
        // Initialize changepoint with spread distribution (not all mass on r=0)
        let init_spread = 20;
        let mut changepoint_run_probs = vec![0.0; init_spread];
        for i in 0..init_spread {
            changepoint_run_probs[i] = (0.8_f64).powi(i as i32);
        }
        let sum: f64 = changepoint_run_probs.iter().sum();
        for p in &mut changepoint_run_probs {
            *p /= sum;
        }

        Self {
            // Drift/vol
            drift_sum: 0.0,
            drift_sum_sq: 0.0,
            drift_n: 0.0,
            drift_prior_mean: 0.0,
            drift_prior_precision: 100.0, // Tight prior around zero

            // Kappa
            kappa_smoothed: 2000.0,
            kappa_smoothed_initialized: false,
            own_kappa_alpha: 2.0,
            own_kappa_beta: 0.001,
            own_kappa_n_fills: 0,
            book_kappa: 2000.0,
            book_kappa_r2: 0.0,
            robust_kappa: 2000.0,
            robust_kappa_ess: 0.0,

            // Continuation
            continuation_alpha: 2.5,
            continuation_beta: 2.5,
            momentum_continuation: 0.5,
            trend_agreement: 0.0,
            trend_confidence: 0.0,

            // Regime
            regime_probs: [0.2, 0.5, 0.2, 0.1],

            // Changepoint
            changepoint_run_probs,
            changepoint_obs_count: 0,
            changepoint_consecutive_high: 0,

            // Edge
            edge_sum: 0.0,
            edge_sum_sq: 0.0,
            edge_n: 0.0,
            edge_by_regime: [1.0, 0.5, -0.5],
            as_bias: 0.0,

            // Calibration
            pending_predictions: HashMap::new(),
            next_prediction_id: 0,
            fill_brier_sum: 0.0,
            fill_brier_n: 0,
            fill_base_rate_sum: 0.0,
            as_brier_sum: 0.0,
            as_brier_n: 0,
            as_base_rate_sum: 0.0,
            signal_mi: HashMap::new(),

            // Stats
            n_price_obs: 0,
            n_fills: 0,
            n_market_trades: 0,
            total_time: 0.0,
            last_update_ms: 0,
        }
    }
}

/// Centralized belief state - single source of truth.
pub struct CentralBeliefState {
    config: CentralBeliefConfig,
    state: RwLock<InternalState>,
    start_time: Instant,
}

impl CentralBeliefState {
    /// Create a new centralized belief state.
    pub fn new(config: CentralBeliefConfig) -> Self {
        let mut state = InternalState::default();

        // Initialize kappa priors from config
        state.kappa_smoothed = config.kappa_prior;
        state.own_kappa_alpha = config.kappa_prior_strength;
        state.own_kappa_beta = config.kappa_prior_strength / config.kappa_prior;
        state.book_kappa = config.kappa_prior;
        state.robust_kappa = config.kappa_prior;

        // Initialize drift prior
        state.drift_prior_precision = 1.0 / (config.drift_prior_sigma.powi(2));

        Self {
            config,
            state: RwLock::new(state),
            start_time: Instant::now(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(CentralBeliefConfig::default())
    }

    /// Process a belief update.
    pub fn update(&self, update: BeliefUpdate) {
        let mut state = self.state.write().unwrap();
        self.process_update(&mut state, update);
    }

    /// Get a snapshot of current beliefs.
    pub fn snapshot(&self) -> BeliefSnapshot {
        let state = self.state.read().unwrap();
        self.build_snapshot(&state)
    }

    /// Check if beliefs are warmed up.
    pub fn is_warmed_up(&self) -> bool {
        let state = self.state.read().unwrap();
        state.n_price_obs >= self.config.min_price_obs
            && state.own_kappa_n_fills >= self.config.min_fills as usize
            && state.total_time >= self.config.min_observation_time
    }

    /// Get warmup progress [0, 1].
    pub fn warmup_progress(&self) -> f64 {
        let state = self.state.read().unwrap();

        let price_progress =
            (state.n_price_obs as f64 / self.config.min_price_obs as f64).min(1.0);
        let fill_progress = (state.own_kappa_n_fills as f64 / self.config.min_fills as f64).min(1.0);
        let time_progress = (state.total_time / self.config.min_observation_time).min(1.0);

        // Geometric mean
        (price_progress * fill_progress * time_progress).powf(1.0 / 3.0)
    }

    // =========================================================================
    // Update Processing
    // =========================================================================

    fn process_update(&self, state: &mut InternalState, update: BeliefUpdate) {
        match update {
            BeliefUpdate::PriceReturn {
                return_frac,
                dt_secs,
                timestamp_ms,
            } => {
                self.process_price_return(state, return_frac, dt_secs, timestamp_ms);
            }

            BeliefUpdate::OwnFill {
                price,
                size: _,
                mid,
                is_aligned,
                realized_as_bps,
                realized_edge_bps,
                timestamp_ms,
                order_id,
                ..
            } => {
                self.process_own_fill(
                    state,
                    price,
                    mid,
                    is_aligned,
                    realized_as_bps,
                    realized_edge_bps,
                    timestamp_ms,
                    order_id,
                );
            }

            BeliefUpdate::MarketTrade {
                price,
                mid,
                timestamp_ms: _,
            } => {
                self.process_market_trade(state, price, mid);
            }

            BeliefUpdate::BookUpdate {
                bids: _,
                asks: _,
                mid: _,
                timestamp_ms: _,
            } => {
                // Book kappa updates would go here
                // For now, handled externally by KappaOrchestrator
            }

            BeliefUpdate::RegimeUpdate { probs, features: _ } => {
                state.regime_probs = probs;
            }

            BeliefUpdate::ChangepointObs { observation } => {
                self.process_changepoint_obs(state, observation);
            }

            BeliefUpdate::ContinuationSignals {
                momentum,
                trend_agreement,
                trend_confidence,
            } => {
                state.momentum_continuation = momentum;
                state.trend_agreement = trend_agreement;
                state.trend_confidence = trend_confidence;
            }

            BeliefUpdate::LogPrediction { prediction } => {
                self.log_prediction(state, prediction);
            }

            BeliefUpdate::RecordOutcome {
                prediction_id,
                actual_value,
                delay_ms: _,
            } => {
                self.record_outcome(state, prediction_id, actual_value);
            }

            BeliefUpdate::SignalMiUpdate { signal_name, mi } => {
                state.signal_mi.insert(signal_name, mi);
            }

            BeliefUpdate::SoftReset { retention } => {
                self.soft_reset(state, retention);
            }

            BeliefUpdate::HardReset => {
                *state = InternalState::default();
            }

            BeliefUpdate::RequestSnapshot { callback } => {
                let snapshot = self.build_snapshot(state);
                let _ = callback.send(snapshot);
            }
        }
    }

    fn process_price_return(
        &self,
        state: &mut InternalState,
        return_frac: f64,
        dt_secs: f64,
        timestamp_ms: u64,
    ) {
        // Normalize return to per-second for drift estimate
        let drift_obs = if dt_secs > 0.01 {
            return_frac / dt_secs.sqrt()
        } else {
            return_frac
        };

        // Update sufficient statistics for Normal-Inverse-Gamma
        state.drift_sum += drift_obs;
        state.drift_sum_sq += drift_obs * drift_obs;
        state.drift_n += 1.0;

        state.n_price_obs += 1;
        state.total_time += dt_secs;
        state.last_update_ms = timestamp_ms;

        // Apply decay periodically
        if state.n_price_obs % 1000 == 0 {
            self.apply_decay(state);
        }
    }

    fn process_own_fill(
        &self,
        state: &mut InternalState,
        price: f64,
        mid: f64,
        is_aligned: bool,
        realized_as_bps: f64,
        realized_edge_bps: f64,
        timestamp_ms: u64,
        order_id: Option<u64>,
    ) {
        // Update kappa posterior (Gamma)
        let distance = ((price - mid) / mid).abs();
        state.own_kappa_alpha += 1.0;
        state.own_kappa_beta += distance.max(0.0001); // Avoid zero
        state.own_kappa_n_fills += 1;

        // Update EWMA-smoothed kappa
        let raw_kappa = self.compute_raw_kappa(state);
        if !state.kappa_smoothed_initialized {
            state.kappa_smoothed = raw_kappa;
            state.kappa_smoothed_initialized = true;
        } else {
            state.kappa_smoothed = self.config.kappa_ewma_alpha * state.kappa_smoothed
                + (1.0 - self.config.kappa_ewma_alpha) * raw_kappa;
        }

        // Update continuation posterior (Beta-Binomial)
        if is_aligned {
            state.continuation_alpha += 1.0;
        } else {
            state.continuation_beta += 1.0;
        }

        // Update edge statistics
        state.edge_sum += realized_edge_bps;
        state.edge_sum_sq += realized_edge_bps * realized_edge_bps;
        state.edge_n += 1.0;

        // Update AS bias
        let alpha = 0.1;
        state.as_bias = (1.0 - alpha) * state.as_bias + alpha * realized_as_bps;

        // Link prediction if we have order_id
        if let Some(oid) = order_id {
            self.link_fill_prediction(state, oid, realized_edge_bps > 0.0);
        }

        state.n_fills += 1;
        state.last_update_ms = timestamp_ms;
    }

    fn process_market_trade(&self, state: &mut InternalState, price: f64, mid: f64) {
        // Update robust kappa with market trade distance
        let distance = ((price - mid) / mid).abs();

        // Simple running average for robust kappa (in reality would use Student-t)
        let alpha = 0.01;
        if distance > 0.0001 && distance < 0.05 {
            // 0.1 bps to 500 bps
            let implied_kappa = 1.0 / distance;
            state.robust_kappa = (1.0 - alpha) * state.robust_kappa + alpha * implied_kappa;
            state.robust_kappa_ess += 1.0;
        }

        state.n_market_trades += 1;
    }

    fn process_changepoint_obs(&self, state: &mut InternalState, observation: f64) {
        // Simplified BOCD update
        // In practice this would use the full run-length distribution update
        let n = state.changepoint_run_probs.len();
        if n == 0 {
            return;
        }

        // Calculate changepoint probability as sum of short run lengths
        let cp_prob: f64 = state.changepoint_run_probs.iter().take(5).sum();

        // Track consecutive high probability
        if cp_prob > self.config.changepoint_threshold {
            state.changepoint_consecutive_high += 1;
        } else {
            state.changepoint_consecutive_high = 0;
        }

        // Shift probabilities (simplified growth model)
        let hazard = self.config.changepoint_hazard;
        let mut new_probs = vec![0.0; (n + 1).min(500)];

        // Probability of changepoint
        let mut cp_mass = 0.0;
        for &prob in state.changepoint_run_probs.iter() {
            // Approximate predictive probability based on observation
            let pred_prob = (-observation.abs()).exp().clamp(0.01, 1.0);
            cp_mass += prob * hazard * pred_prob;
        }
        new_probs[0] = cp_mass;

        // Growth probabilities
        for (r, &prob) in state.changepoint_run_probs.iter().enumerate() {
            if r + 1 < new_probs.len() {
                let pred_prob = (-observation.abs()).exp().clamp(0.01, 1.0);
                new_probs[r + 1] = prob * (1.0 - hazard) * pred_prob;
            }
        }

        // Normalize
        let sum: f64 = new_probs.iter().sum();
        if sum > 1e-10 {
            for p in &mut new_probs {
                *p /= sum;
            }
        }

        state.changepoint_run_probs = new_probs;
        state.changepoint_obs_count += 1;
    }

    fn log_prediction(&self, state: &mut InternalState, prediction: PredictionLog) {
        let id = state.next_prediction_id;
        state.next_prediction_id += 1;

        state.pending_predictions.insert(
            id,
            StoredPrediction {
                prediction_type: prediction.prediction_type,
                predicted_prob: prediction.predicted_prob,
                _confidence: prediction.confidence,
                _regime: prediction.regime,
                _timestamp_ms: prediction.timestamp_ms,
            },
        );

        // Limit pending predictions to avoid memory growth
        if state.pending_predictions.len() > 10000 {
            // Remove oldest (smallest IDs)
            let mut to_remove: Vec<u64> = state.pending_predictions.keys().copied().collect();
            to_remove.sort();
            for id in to_remove.iter().take(1000) {
                state.pending_predictions.remove(id);
            }
        }
    }

    fn record_outcome(&self, state: &mut InternalState, prediction_id: u64, actual_value: f64) {
        if let Some(pred) = state.pending_predictions.remove(&prediction_id) {
            let brier = (pred.predicted_prob - actual_value).powi(2);

            match pred.prediction_type {
                PredictionType::FillProbability => {
                    state.fill_brier_sum += brier;
                    state.fill_brier_n += 1;
                    state.fill_base_rate_sum += actual_value;
                }
                PredictionType::AdverseSelection => {
                    state.as_brier_sum += brier;
                    state.as_brier_n += 1;
                    state.as_base_rate_sum += actual_value;
                }
                _ => {}
            }
        }
    }

    fn link_fill_prediction(&self, state: &mut InternalState, order_id: u64, positive_edge: bool) {
        // Look for prediction associated with this order
        // (In practice would have order_id -> prediction_id mapping)
        // For now, just record as a fill outcome
        let actual = if positive_edge { 1.0 } else { 0.0 };

        // Use order_id as prediction_id (simplified)
        if state.pending_predictions.remove(&order_id).is_some() {
            state.fill_base_rate_sum += actual;
            state.fill_brier_n += 1;
        }
    }

    fn soft_reset(&self, state: &mut InternalState, retention: f64) {
        // Decay all posteriors toward prior
        let r = retention.clamp(0.0, 1.0);

        // Drift
        state.drift_sum *= r;
        state.drift_sum_sq *= r;
        state.drift_n *= r;

        // Kappa
        let prior_alpha = self.config.kappa_prior_strength;
        let prior_beta = self.config.kappa_prior_strength / self.config.kappa_prior;
        state.own_kappa_alpha = prior_alpha + (state.own_kappa_alpha - prior_alpha) * r;
        state.own_kappa_beta = prior_beta + (state.own_kappa_beta - prior_beta) * r;

        // Continuation
        state.continuation_alpha = 2.5 + (state.continuation_alpha - 2.5) * r;
        state.continuation_beta = 2.5 + (state.continuation_beta - 2.5) * r;

        // Edge
        state.edge_sum *= r;
        state.edge_sum_sq *= r;
        state.edge_n *= r;
    }

    fn apply_decay(&self, state: &mut InternalState) {
        let d = self.config.decay_factor;
        state.drift_sum *= d;
        state.drift_sum_sq *= d;
        state.drift_n *= d;
    }

    // =========================================================================
    // Snapshot Building
    // =========================================================================

    fn build_snapshot(&self, state: &InternalState) -> BeliefSnapshot {
        let uptime = self.start_time.elapsed().as_secs_f64();

        BeliefSnapshot {
            drift_vol: self.build_drift_vol_beliefs(state),
            kappa: self.build_kappa_beliefs(state),
            continuation: self.build_continuation_beliefs(state),
            regime: self.build_regime_beliefs(state),
            changepoint: self.build_changepoint_beliefs(state),
            edge: self.build_edge_beliefs(state),
            calibration: self.build_calibration_state(state),
            stats: BeliefStats {
                n_price_obs: state.n_price_obs,
                n_fills: state.n_fills,
                n_market_trades: state.n_market_trades,
                last_update_ms: state.last_update_ms,
                uptime_secs: uptime,
                is_warmed_up: state.n_price_obs >= self.config.min_price_obs
                    && state.own_kappa_n_fills >= self.config.min_fills as usize
                    && state.total_time >= self.config.min_observation_time,
                warmup_progress: self.compute_warmup_progress(state),
            },
        }
    }

    fn build_drift_vol_beliefs(&self, state: &InternalState) -> DriftVolatilityBeliefs {
        let n = state.drift_n.max(1.0);
        let mean = state.drift_sum / n;
        let variance = (state.drift_sum_sq / n - mean * mean).max(0.0);

        // Posterior precision combines prior and data
        let posterior_precision = state.drift_prior_precision + n;
        let posterior_mean =
            (state.drift_prior_precision * state.drift_prior_mean + n * mean) / posterior_precision;
        let posterior_std = (1.0 / posterior_precision).sqrt();

        // Probability calculations using normal CDF approximation
        let z = -posterior_mean / posterior_std.max(1e-10);
        let prob_bearish = normal_cdf(z);

        // Confidence based on precision increase
        let confidence = (1.0 - state.drift_prior_precision / posterior_precision).max(0.0);

        DriftVolatilityBeliefs {
            expected_drift: posterior_mean,
            drift_uncertainty: posterior_std,
            expected_sigma: variance.sqrt().max(0.001),
            prob_bearish,
            prob_bullish: 1.0 - prob_bearish,
            confidence: confidence.min(1.0),
            n_observations: state.n_price_obs,
        }
    }

    fn build_kappa_beliefs(&self, state: &InternalState) -> KappaBeliefs {
        let raw_kappa = self.compute_raw_kappa(state);

        let own_kappa = state.own_kappa_alpha / state.own_kappa_beta.max(0.0001);
        let own_conf = (state.own_kappa_n_fills as f64 / 20.0).min(1.0);

        let book_conf = state.book_kappa_r2;
        let robust_conf = (state.robust_kappa_ess / 50.0).min(1.0);

        // Compute weights
        let total = own_conf + book_conf + robust_conf + 0.05;
        let w_own = own_conf / total;
        let w_book = book_conf / total;
        let w_robust = robust_conf / total;
        let w_prior = 0.05 / total;

        KappaBeliefs {
            kappa_effective: state.kappa_smoothed,
            kappa_raw: raw_kappa,
            components: KappaComponents {
                own: (own_kappa, w_own),
                book: (state.book_kappa, w_book),
                robust: (state.robust_kappa, w_robust),
                prior: (self.config.kappa_prior, w_prior),
            },
            confidence: own_conf.max(book_conf).max(robust_conf),
            is_warmup: state.own_kappa_n_fills < self.config.min_fills as usize,
            n_own_fills: state.own_kappa_n_fills,
        }
    }

    fn compute_raw_kappa(&self, state: &InternalState) -> f64 {
        let min_fills = self.config.min_fills as usize;

        if state.own_kappa_n_fills < min_fills {
            // Warmup: blend market signals with prior
            let book_valid = state.book_kappa > 100.0 && state.book_kappa_r2 > 0.1;
            let robust_valid = state.robust_kappa > 100.0 && state.robust_kappa_ess > 5.0;

            let book_weight = if book_valid { 0.4 } else { 0.0 };
            let robust_weight = if robust_valid { 0.3 } else { 0.0 };
            let prior_weight = 1.0 - book_weight - robust_weight;

            (book_weight * state.book_kappa
                + robust_weight * state.robust_kappa
                + prior_weight * self.config.kappa_prior)
                .clamp(50.0, 10000.0)
        } else {
            // Post-warmup: confidence-weighted blend
            let own_kappa = state.own_kappa_alpha / state.own_kappa_beta.max(0.0001);
            let own_conf = (state.own_kappa_n_fills as f64 / 20.0).min(1.0);
            let book_conf = state.book_kappa_r2;
            let robust_conf = (state.robust_kappa_ess / 50.0).min(1.0);

            let total = own_conf + book_conf + robust_conf + 0.05;

            ((own_conf * own_kappa
                + book_conf * state.book_kappa
                + robust_conf * state.robust_kappa
                + 0.05 * self.config.kappa_prior)
                / total)
                .clamp(50.0, 10000.0)
        }
    }

    fn build_continuation_beliefs(&self, state: &InternalState) -> ContinuationBeliefs {
        let p_fill_raw = state.continuation_alpha / (state.continuation_alpha + state.continuation_beta);

        // Regime prior
        let p_regime = state.regime_probs[0] * 0.3
            + state.regime_probs[1] * 0.5
            + state.regime_probs[2] * 0.65
            + state.regime_probs[3] * 0.8;

        // Changepoint discount
        let cp_prob: f64 = state.changepoint_run_probs.iter().take(5).sum();
        let changepoint_discount = (cp_prob * 0.5).clamp(0.0, 0.8);

        // Fused probability
        let p_fill_discounted = (1.0 - changepoint_discount) * p_fill_raw + changepoint_discount * p_regime;
        let p_trend = 0.5 + 0.5 * state.trend_agreement;

        let fill_conf = 1.0 - (state.continuation_alpha * state.continuation_beta)
            / ((state.continuation_alpha + state.continuation_beta).powi(2)
                * (state.continuation_alpha + state.continuation_beta + 1.0));

        // Weighted fusion
        let w_fill = 0.4 * fill_conf;
        let w_momentum = 0.25;
        let w_trend = 0.2 * state.trend_confidence;
        let w_regime = 0.15;
        let total = w_fill + w_momentum + w_trend + w_regime;

        let p_fused = if total > 1e-6 {
            (w_fill * p_fill_discounted
                + w_momentum * state.momentum_continuation
                + w_trend * p_trend
                + w_regime * p_regime)
                / total
        } else {
            p_regime
        };

        ContinuationBeliefs {
            p_fill_raw,
            p_fused: p_fused.clamp(0.0, 1.0),
            confidence_fused: (fill_conf * (1.0 - cp_prob) * (0.3 + 0.7 * state.trend_confidence))
                .powf(1.0 / 3.0),
            changepoint_discount,
            signal_summary: ContinuationSignals {
                p_momentum: state.momentum_continuation,
                p_trend,
                p_regime,
                trend_confidence: state.trend_confidence,
            },
        }
    }

    fn build_regime_beliefs(&self, state: &InternalState) -> RegimeBeliefs {
        let current = Regime::from_probs(&state.regime_probs);
        let max_prob = state.regime_probs.iter().cloned().fold(0.0_f64, f64::max);

        RegimeBeliefs {
            probs: state.regime_probs,
            current,
            confidence: max_prob,
            transition_prob: 0.1, // Default, would be from HMM
        }
    }

    fn build_changepoint_beliefs(&self, state: &InternalState) -> ChangepointBeliefs {
        let prob_1: f64 = state.changepoint_run_probs.iter().take(1).sum();
        let prob_5: f64 = state.changepoint_run_probs.iter().take(5).sum();
        let prob_10: f64 = state.changepoint_run_probs.iter().take(10).sum();

        let run_length = state
            .changepoint_run_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let entropy: f64 = -state
            .changepoint_run_probs
            .iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| p * p.ln())
            .sum::<f64>();

        // Determine result
        let result = if state.changepoint_obs_count < 10 {
            ChangepointResult::None
        } else if prob_5 <= self.config.changepoint_threshold {
            ChangepointResult::None
        } else if state.changepoint_consecutive_high >= 2 {
            ChangepointResult::Confirmed
        } else {
            ChangepointResult::Pending(state.changepoint_consecutive_high)
        };

        // Learning trust decreases with changepoint probability
        let learning_trust = (1.0 - prob_10).max(0.2);

        ChangepointBeliefs {
            prob_1,
            prob_5,
            prob_10,
            run_length,
            entropy,
            result,
            learning_trust,
        }
    }

    fn build_edge_beliefs(&self, state: &InternalState) -> EdgeBeliefs {
        let n = state.edge_n.max(1.0);
        let mean = state.edge_sum / n;
        let variance = (state.edge_sum_sq / n - mean * mean).max(0.0);
        let std = variance.sqrt();

        // P(positive edge)
        let z = mean / std.max(0.1);
        let p_positive = normal_cdf(z);

        EdgeBeliefs {
            expected_edge: mean,
            uncertainty: std,
            by_regime: state.edge_by_regime,
            p_positive,
            as_bias: state.as_bias,
            epistemic_uncertainty: 0.5, // Default
        }
    }

    fn build_calibration_state(&self, state: &InternalState) -> CalibrationState {
        let fill_metrics = if state.fill_brier_n > 0 {
            let brier = state.fill_brier_sum / state.fill_brier_n as f64;
            let base_rate = state.fill_base_rate_sum / state.fill_brier_n as f64;
            let baseline_brier = base_rate * (1.0 - base_rate);
            let ir = if brier > 0.0 {
                baseline_brier / brier
            } else {
                0.0
            };

            CalibrationMetrics {
                brier_score: brier,
                information_ratio: ir,
                base_rate,
                n_samples: state.fill_brier_n,
                is_calibrated: state.fill_brier_n >= 100 && ir > 1.0,
            }
        } else {
            CalibrationMetrics::default()
        };

        let as_metrics = if state.as_brier_n > 0 {
            let brier = state.as_brier_sum / state.as_brier_n as f64;
            let base_rate = state.as_base_rate_sum / state.as_brier_n as f64;
            let baseline_brier = base_rate * (1.0 - base_rate);
            let ir = if brier > 0.0 {
                baseline_brier / brier
            } else {
                0.0
            };

            CalibrationMetrics {
                brier_score: brier,
                information_ratio: ir,
                base_rate,
                n_samples: state.as_brier_n,
                is_calibrated: state.as_brier_n >= 100 && ir > 1.0,
            }
        } else {
            CalibrationMetrics::default()
        };

        CalibrationState {
            fill: fill_metrics,
            adverse_selection: as_metrics,
            signal_quality: state.signal_mi.clone(),
            pending_count: state.pending_predictions.len(),
            linked_count: state.fill_brier_n + state.as_brier_n,
        }
    }

    fn compute_warmup_progress(&self, state: &InternalState) -> f64 {
        let price_progress =
            (state.n_price_obs as f64 / self.config.min_price_obs as f64).min(1.0);
        let fill_progress = (state.own_kappa_n_fills as f64 / self.config.min_fills as f64).min(1.0);
        let time_progress = (state.total_time / self.config.min_observation_time).min(1.0);

        (price_progress * fill_progress * time_progress).powf(1.0 / 3.0)
    }
}

/// Normal CDF approximation using error function approximation.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz and Stegun).
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_central_belief_state_creation() {
        let beliefs = CentralBeliefState::default_config();
        assert!(!beliefs.is_warmed_up());
    }

    #[test]
    fn test_price_return_updates_drift() {
        let beliefs = CentralBeliefState::default_config();

        // Observe negative returns
        for _ in 0..50 {
            beliefs.update(BeliefUpdate::PriceReturn {
                return_frac: -0.002,
                dt_secs: 1.0,
                timestamp_ms: 12345,
            });
        }

        let snapshot = beliefs.snapshot();
        assert!(
            snapshot.drift_vol.expected_drift < 0.0,
            "Expected negative drift: {}",
            snapshot.drift_vol.expected_drift
        );
    }

    #[test]
    fn test_own_fill_updates_kappa() {
        let beliefs = CentralBeliefState::default_config();

        // Add fills
        for i in 0..10 {
            beliefs.update(BeliefUpdate::OwnFill {
                price: 100.01,
                size: 1.0,
                mid: 100.0,
                is_buy: true,
                is_aligned: true,
                realized_as_bps: 1.0,
                realized_edge_bps: 2.0,
                timestamp_ms: i * 1000,
                order_id: Some(i),
            });
        }

        let snapshot = beliefs.snapshot();
        assert!(snapshot.kappa.n_own_fills == 10);
        assert!(!snapshot.kappa.is_warmup);
    }

    #[test]
    fn test_regime_update() {
        let beliefs = CentralBeliefState::default_config();

        beliefs.update(BeliefUpdate::RegimeUpdate {
            probs: [0.1, 0.1, 0.1, 0.7], // Cascade
            features: None,
        });

        let snapshot = beliefs.snapshot();
        assert_eq!(snapshot.regime.current, Regime::Cascade);
    }

    #[test]
    fn test_soft_reset() {
        let beliefs = CentralBeliefState::default_config();

        // Build up state
        for _ in 0..100 {
            beliefs.update(BeliefUpdate::PriceReturn {
                return_frac: 0.001,
                dt_secs: 1.0,
                timestamp_ms: 12345,
            });
        }

        let before = beliefs.snapshot();
        assert!(before.drift_vol.n_observations > 0);

        // Soft reset
        beliefs.update(BeliefUpdate::SoftReset { retention: 0.5 });

        let after = beliefs.snapshot();
        assert!(after.drift_vol.expected_drift.abs() < before.drift_vol.expected_drift.abs());
    }

    #[test]
    fn test_warmup_progress() {
        let config = CentralBeliefConfig {
            min_price_obs: 10,
            min_fills: 2,
            min_observation_time: 5.0,
            ..Default::default()
        };
        let beliefs = CentralBeliefState::new(config);

        assert!(!beliefs.is_warmed_up());
        assert!((beliefs.warmup_progress() - 0.0).abs() < 0.1);

        // Add observations
        for i in 0..10 {
            beliefs.update(BeliefUpdate::PriceReturn {
                return_frac: 0.001,
                dt_secs: 1.0,
                timestamp_ms: i * 1000,
            });
        }

        // Add fills
        for i in 0..2 {
            beliefs.update(BeliefUpdate::OwnFill {
                price: 100.01,
                size: 1.0,
                mid: 100.0,
                is_buy: true,
                is_aligned: true,
                realized_as_bps: 1.0,
                realized_edge_bps: 2.0,
                timestamp_ms: i * 1000,
                order_id: Some(i),
            });
        }

        assert!(beliefs.is_warmed_up());
        assert!(beliefs.warmup_progress() > 0.9);
    }

    #[test]
    fn test_normal_cdf() {
        // CDF(0) should be 0.5
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);

        // CDF(-∞) → 0, CDF(+∞) → 1
        assert!(normal_cdf(-5.0) < 0.01);
        assert!(normal_cdf(5.0) > 0.99);
    }
}
