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

use super::bayesian_fair_value::{BayesianFairValue, BayesianFairValueConfig, FairValueBeliefs};
use super::messages::{BeliefUpdate, PredictionLog, PredictionType};
use super::snapshot::{
    BeliefSnapshot, BeliefStats, CalibrationMetrics, CalibrationState, ChangepointBeliefs,
    ChangepointResult, ContinuationBeliefs, ContinuationSignals, CrossVenueBeliefs,
    DriftVolatilityBeliefs, EdgeBeliefs, KappaBeliefs, KappaComponents, LatencyCalibration,
    MicrostructureBeliefs, RegimeBeliefs,
};
use super::Regime;

/// Configuration for CentralBeliefState.
#[derive(Debug, Clone)]
pub struct CentralBeliefConfig {
    // === Directional Log-Odds Config ===
    /// Decay time constant (seconds). Half-life = tau × ln(2) ≈ 21s at default.
    pub dir_decay_tau_secs: f64,
    /// Max absolute log-odds (sigmoid(5)=0.993). Prevents extreme beliefs.
    pub dir_max_log_odds: f64,
    /// Weight for price-return evidence (z-score, already dimensionless).
    pub dir_lambda_price: f64,
    /// Weight for fill-side evidence (binary ±1 per fill).
    pub dir_lambda_fill: f64,
    /// Weight for AS-direction evidence (magnitude-scaled).
    pub dir_lambda_as: f64,
    /// Weight for flow-direction evidence (order_flow_direction).
    pub dir_lambda_flow: f64,

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
    /// Minimum consecutive high-probability observations to confirm a changepoint.
    /// Higher values reduce false positives from noise. Default: 2.
    pub changepoint_min_confirmations: usize,
    /// Cooldown period (ms) between confirmed changepoints.
    /// Prevents rapid resets that destroy learned parameters. Default: 0 (no cooldown).
    pub changepoint_cooldown_ms: u64,

    // === EWMA ===
    /// EWMA smoothing factor for kappa (0.9 = 90% previous, 10% new)
    pub kappa_ewma_alpha: f64,

    // === Bayesian Fair Value ===
    /// Configuration for the Bayesian fair value model.
    /// Set to None to disable the model entirely.
    pub bayesian_fv_config: Option<BayesianFairValueConfig>,
}

impl Default for CentralBeliefConfig {
    fn default() -> Self {
        Self {
            dir_decay_tau_secs: 30.0,
            dir_max_log_odds: 5.0,
            dir_lambda_price: 1.0,
            dir_lambda_fill: 0.3,
            dir_lambda_as: 0.5,
            dir_lambda_flow: 0.8,
            kappa_prior: 2000.0,
            kappa_prior_strength: 10.0,
            min_price_obs: 50,
            min_fills: 5,
            min_observation_time: 60.0,
            decay_factor: 0.999,
            changepoint_hazard: 1.0 / 250.0,
            changepoint_threshold: 0.7,
            changepoint_min_confirmations: 2,
            changepoint_cooldown_ms: 0,
            kappa_ewma_alpha: 0.9,
            bayesian_fv_config: Some(BayesianFairValueConfig::default()),
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
            changepoint_hazard: 1.0 / 150.0,  // More sensitive
            changepoint_min_confirmations: 4, // 4 consecutive required (vs default 2)
            changepoint_cooldown_ms: 300_000, // 5 minute cooldown between confirmed changepoints
            bayesian_fv_config: Some(BayesianFairValueConfig::default()),
            ..Default::default()
        }
    }

    /// Config for liquid CEX markets.
    pub fn liquid() -> Self {
        Self {
            kappa_prior: 2500.0,
            kappa_prior_strength: 5.0,
            min_observation_time: 30.0,
            bayesian_fv_config: Some(BayesianFairValueConfig::default()),
            ..Default::default()
        }
    }
}

/// Internal state (mutable, protected by RwLock).
struct InternalState {
    // === Bayesian Fair Value ===
    bayesian_fv: Option<BayesianFairValue>,
    /// Latest mid price for Bayesian FV flow updates (BookUpdate/PriceReturn set this).
    last_mid_for_fv: f64,

    // === Directional Log-Odds ===
    /// Log-odds accumulator: L > 0 = bearish, L < 0 = bullish
    dir_log_odds: f64,
    dir_last_update_ms: u64,
    /// Per-source evidence counts (diagnostics + confidence ramp)
    dir_n_price: u64,
    dir_n_fill: u64,
    dir_n_as: u64,
    dir_n_flow: u64,
    /// Running variance for sigma estimation
    sigma_sum_sq: f64,
    sigma_n: f64,
    /// Per-source cumulative LR (decayed, for drift_skewness diagnostic)
    dir_lr_sum_price: f64,
    dir_lr_sum_fill: f64,
    dir_lr_sum_as: f64,
    dir_lr_sum_flow: f64,

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

    // Kappa uncertainty (for spread CI)
    kappa_variance: f64,
    kappa_sigma_covariance: f64,
    sigma_variance: f64,

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
    /// Timestamp (ms) of last confirmed changepoint, for cooldown enforcement.
    last_changepoint_confirmed_ms: u64,

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

    // === Microstructure (Phase 1: Alpha-generating) ===
    vpin: f64,
    vpin_velocity: f64,
    depth_ofi: f64,
    liquidity_evaporation: f64,
    order_flow_direction: f64,
    vpin_confidence: f64,
    vpin_buckets: usize,

    // === Microstructure (Phase 1A: Toxic Volume Refinements) ===
    trade_size_sigma: f64,
    toxicity_acceleration: f64,
    cofi: f64,
    cofi_velocity: f64,
    is_sustained_shift: bool,

    // === Skewness (Phase 2A: Fat-Tail Tracking) ===
    /// Tracked sigma skewness for asymmetric spread adjustment
    sigma_skewness: f64,

    // === Cross-Venue (Bivariate Flow Model) ===
    /// Joint direction belief from bivariate analysis [-1, +1]
    cv_direction: f64,
    /// Confidence in direction based on venue agreement [0, 1]
    cv_confidence: f64,
    /// Where is price discovery happening? [0=HL, 1=Binance]
    cv_discovery_venue: f64,
    /// Maximum toxicity across venues [0, 1]
    cv_max_toxicity: f64,
    /// Average toxicity across venues [0, 1]
    cv_avg_toxicity: f64,
    /// Agreement score between venues [-1, 1]
    cv_agreement: f64,
    /// Imbalance divergence (binance - hl) [-2, 2]
    cv_divergence: f64,
    /// Intensity ratio λ_B / (λ_B + λ_H) [0, 1]
    cv_intensity_ratio: f64,
    /// Rolling imbalance correlation [-1, 1]
    cv_imbalance_correlation: f64,
    /// Toxicity alert active?
    cv_toxicity_alert: bool,
    /// Divergence alert active?
    cv_divergence_alert: bool,
    /// Whether cross-venue beliefs are valid
    cv_is_valid: bool,
    /// Number of cross-venue observations
    cv_observation_count: u64,
    /// Last cross-venue update timestamp
    cv_last_update_ms: u64,

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
        for (i, prob) in changepoint_run_probs
            .iter_mut()
            .enumerate()
            .take(init_spread)
        {
            *prob = (0.8_f64).powi(i as i32);
        }
        let sum: f64 = changepoint_run_probs.iter().sum();
        for p in &mut changepoint_run_probs {
            *p /= sum;
        }

        Self {
            // Bayesian fair value (lazy-init from config)
            bayesian_fv: None,
            last_mid_for_fv: 0.0,

            // Directional log-odds
            dir_log_odds: 0.0,
            dir_last_update_ms: 0,
            dir_n_price: 0,
            dir_n_fill: 0,
            dir_n_as: 0,
            dir_n_flow: 0,
            sigma_sum_sq: 0.0,
            sigma_n: 0.0,
            dir_lr_sum_price: 0.0,
            dir_lr_sum_fill: 0.0,
            dir_lr_sum_as: 0.0,
            dir_lr_sum_flow: 0.0,

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
            // Kappa uncertainty
            kappa_variance: 500.0 * 500.0, // High initial uncertainty
            kappa_sigma_covariance: 0.0,
            sigma_variance: 0.0001 * 0.0001,

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
            last_changepoint_confirmed_ms: 0,

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

            // Microstructure
            vpin: 0.0,
            vpin_velocity: 0.0,
            depth_ofi: 0.0,
            liquidity_evaporation: 0.0,
            order_flow_direction: 0.0,
            vpin_confidence: 0.0,
            vpin_buckets: 0,

            // Microstructure Phase 1A
            trade_size_sigma: 0.0,
            toxicity_acceleration: 1.0,
            cofi: 0.0,
            cofi_velocity: 0.0,
            is_sustained_shift: false,

            // Phase 2A: Skewness tracking
            sigma_skewness: 0.5, // Slight positive (typical for vol)

            // Cross-venue (bivariate flow model)
            cv_direction: 0.0,
            cv_confidence: 0.0,
            cv_discovery_venue: 0.5,
            cv_max_toxicity: 0.0,
            cv_avg_toxicity: 0.0,
            cv_agreement: 0.0,
            cv_divergence: 0.0,
            cv_intensity_ratio: 0.5,
            cv_imbalance_correlation: 0.0,
            cv_toxicity_alert: false,
            cv_divergence_alert: false,
            cv_is_valid: false,
            cv_observation_count: 0,
            cv_last_update_ms: 0,

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

/// Own fill data for belief state updates.
struct OwnFillParams {
    price: f64,
    mid: f64,
    is_buy: bool,
    is_aligned: bool,
    realized_as_bps: f64,
    realized_edge_bps: f64,
    timestamp_ms: u64,
    order_id: Option<u64>,
}

/// Cross-venue analysis results for belief state integration.
struct CrossVenueParams {
    direction: f64,
    confidence: f64,
    discovery_venue: f64,
    max_toxicity: f64,
    avg_toxicity: f64,
    agreement: f64,
    divergence: f64,
    intensity_ratio: f64,
    imbalance_correlation: f64,
    toxicity_alert: bool,
    divergence_alert: bool,
    timestamp_ms: u64,
}

impl CentralBeliefState {
    /// Create a new centralized belief state.
    pub fn new(config: CentralBeliefConfig) -> Self {
        let bayesian_fv = config
            .bayesian_fv_config
            .as_ref()
            .map(|fv_config| BayesianFairValue::new(fv_config.clone()));

        let state = InternalState {
            bayesian_fv,
            kappa_smoothed: config.kappa_prior,
            own_kappa_alpha: config.kappa_prior_strength,
            own_kappa_beta: config.kappa_prior_strength / config.kappa_prior,
            book_kappa: config.kappa_prior,
            robust_kappa: config.kappa_prior,
            ..Default::default()
        };

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

        let price_progress = (state.n_price_obs as f64 / self.config.min_price_obs as f64).min(1.0);
        let fill_progress =
            (state.own_kappa_n_fills as f64 / self.config.min_fills as f64).min(1.0);
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
                size,
                mid,
                is_buy,
                is_aligned,
                realized_as_bps,
                realized_edge_bps,
                timestamp_ms,
                order_id,
                quoted_size,
            } => {
                let fill = OwnFillParams {
                    price,
                    mid,
                    is_buy,
                    is_aligned,
                    realized_as_bps,
                    realized_edge_bps,
                    timestamp_ms,
                    order_id,
                };
                self.process_own_fill(state, &fill);

                // Update Bayesian fair value model
                if let Some(ref mut fv) = state.bayesian_fv {
                    let vpin = state.vpin;
                    fv.update_on_fill(price, size, quoted_size, is_buy, mid, vpin);
                }
            }

            BeliefUpdate::MarketTrade {
                price,
                mid,
                timestamp_ms: _,
            } => {
                self.process_market_trade(state, price, mid);
            }

            BeliefUpdate::BookUpdate {
                bids,
                asks,
                mid,
                timestamp_ms: _,
            } => {
                // Book kappa updates handled externally by KappaOrchestrator.
                // Bayesian fair value uses L2 imbalance.
                // Track mid for FV flow updates and predict step
                state.last_mid_for_fv = mid;
                if let Some(ref mut fv) = state.bayesian_fv {
                    let bid_depth: f64 = bids.iter().map(|(_, s)| s).sum();
                    let ask_depth: f64 = asks.iter().map(|(_, s)| s).sum();
                    let total = bid_depth + ask_depth;
                    if total > 0.0 {
                        let imbalance = (bid_depth - ask_depth) / total;
                        fv.update_on_book(imbalance, mid);
                    }
                }
            }

            BeliefUpdate::FlowUpdate {
                imbalance_1s,
                imbalance_5s,
                imbalance_30s,
                timestamp_ms: _,
            } => {
                if let Some(ref mut fv) = state.bayesian_fv {
                    // Decorrelate flow observations: raw imbalances are nested
                    // (1s ⊆ 5s ⊆ 30s), not independent. Batch Kalman treats them
                    // as orthogonal, overcounting information by ~2-3x.
                    // Compute exclusive increments so each observation is independent.
                    let flow_short = imbalance_1s; // 0-1s (exclusive)
                    let flow_medium = imbalance_5s - imbalance_1s; // 1-5s increment
                    let flow_long = imbalance_30s - imbalance_5s; // 5-30s increment
                    fv.update_on_flow(flow_short, flow_medium, flow_long, state.last_mid_for_fv);
                }
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

            BeliefUpdate::MicrostructureUpdate {
                vpin,
                vpin_velocity,
                depth_ofi,
                liquidity_evaporation,
                order_flow_direction,
                confidence,
                vpin_buckets,
                trade_size_sigma,
                toxicity_acceleration,
                cofi,
                cofi_velocity,
                is_sustained_shift,
            } => {
                state.vpin = vpin;
                state.vpin_velocity = vpin_velocity;
                state.depth_ofi = depth_ofi;
                state.liquidity_evaporation = liquidity_evaporation;
                state.order_flow_direction = order_flow_direction;
                state.vpin_confidence = confidence;
                state.vpin_buckets = vpin_buckets;
                // Phase 1A fields
                state.trade_size_sigma = trade_size_sigma;
                state.toxicity_acceleration = toxicity_acceleration;
                state.cofi = cofi;
                state.cofi_velocity = cofi_velocity;
                state.is_sustained_shift = is_sustained_shift;

                // --- Directional log-odds: order flow evidence ---
                // order_flow_direction ∈ [-1,1], positive = buy pressure = bullish → L↓
                let lr_flow = (-order_flow_direction).clamp(-2.0, 2.0);
                state.dir_log_odds += self.config.dir_lambda_flow * lr_flow;
                state.dir_log_odds = state
                    .dir_log_odds
                    .clamp(-self.config.dir_max_log_odds, self.config.dir_max_log_odds);
                state.dir_n_flow += 1;
                state.dir_lr_sum_flow += lr_flow;
            }

            BeliefUpdate::CrossVenueUpdate {
                direction,
                confidence,
                discovery_venue,
                max_toxicity,
                avg_toxicity,
                agreement,
                divergence,
                intensity_ratio,
                imbalance_correlation,
                toxicity_alert,
                divergence_alert,
                timestamp_ms,
            } => {
                let params = CrossVenueParams {
                    direction,
                    confidence,
                    discovery_venue,
                    max_toxicity,
                    avg_toxicity,
                    agreement,
                    divergence,
                    intensity_ratio,
                    imbalance_correlation,
                    toxicity_alert,
                    divergence_alert,
                    timestamp_ms,
                };
                self.process_cross_venue_update(state, &params);
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

    fn apply_dir_time_decay(&self, state: &mut InternalState, current_ms: u64) {
        if state.dir_last_update_ms == 0 || current_ms <= state.dir_last_update_ms {
            state.dir_last_update_ms = current_ms;
            return;
        }
        let dt_secs = (current_ms - state.dir_last_update_ms) as f64 / 1000.0;
        let decay = (-dt_secs / self.config.dir_decay_tau_secs).exp();
        state.dir_log_odds *= decay;
        state.dir_lr_sum_price *= decay;
        state.dir_lr_sum_fill *= decay;
        state.dir_lr_sum_as *= decay;
        state.dir_lr_sum_flow *= decay;
        state.dir_last_update_ms = current_ms;
    }

    fn process_price_return(
        &self,
        state: &mut InternalState,
        return_frac: f64,
        dt_secs: f64,
        timestamp_ms: u64,
    ) {
        state.n_price_obs += 1;
        state.total_time += dt_secs;
        state.last_update_ms = timestamp_ms;

        // --- Directional log-odds: price return evidence ---
        self.apply_dir_time_decay(state, timestamp_ms);

        let sigma_est = if state.sigma_n > 5.0 {
            (state.sigma_sum_sq / state.sigma_n).sqrt().max(0.0001)
        } else {
            0.002 // ~20 bps conservative default
        };

        // z-score: positive return → bullish evidence → decrease L
        let z = (return_frac / (sigma_est * dt_secs.sqrt().max(0.1))).clamp(-3.0, 3.0);
        state.dir_log_odds -= self.config.dir_lambda_price * z;
        state.dir_log_odds = state
            .dir_log_odds
            .clamp(-self.config.dir_max_log_odds, self.config.dir_max_log_odds);
        state.dir_n_price += 1;
        state.dir_lr_sum_price -= z;

        // Running variance for sigma
        let return_sq = if dt_secs > 0.01 {
            (return_frac / dt_secs.sqrt()).powi(2)
        } else {
            return_frac.powi(2)
        };
        state.sigma_sum_sq += return_sq;
        state.sigma_n += 1.0;

        // Update Bayesian fair value model: predict step (variance grows, mean reverts)
        if let Some(ref mut fv) = state.bayesian_fv {
            // Derive new mid from return: mid_new = mid_old × (1 + return_frac)
            // If last_mid_for_fv is not yet set (0.0), initialize from a reasonable default
            if state.last_mid_for_fv > 0.0 {
                let new_mid = state.last_mid_for_fv * (1.0 + return_frac);
                fv.predict(new_mid, dt_secs, timestamp_ms);
                state.last_mid_for_fv = new_mid;
            }
        }

        // Apply decay periodically
        if state.n_price_obs.is_multiple_of(1000) {
            self.apply_decay(state);
        }
    }

    fn process_own_fill(&self, state: &mut InternalState, fill: &OwnFillParams) {
        // Update kappa posterior (Gamma)
        let distance = ((fill.price - fill.mid) / fill.mid).abs();
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
        if fill.is_aligned {
            state.continuation_alpha += 1.0;
        } else {
            state.continuation_beta += 1.0;
        }

        // Update edge statistics
        state.edge_sum += fill.realized_edge_bps;
        state.edge_sum_sq += fill.realized_edge_bps * fill.realized_edge_bps;
        state.edge_n += 1.0;

        // Update AS bias
        let alpha = 0.1;
        state.as_bias = (1.0 - alpha) * state.as_bias + alpha * fill.realized_as_bps;

        // --- Directional log-odds: fill side + AS evidence ---
        self.apply_dir_time_decay(state, fill.timestamp_ms);

        // Fill side: is_buy=true → sell aggressor hit our bid → bearish evidence (L↑)
        let lr_fill = if fill.is_buy { 1.0 } else { -1.0 };
        state.dir_log_odds += self.config.dir_lambda_fill * lr_fill;
        state.dir_n_fill += 1;
        state.dir_lr_sum_fill += lr_fill;

        // AS direction: large AS + is_buy → confirmed downward move → bearish
        let as_scale_bps = 5.0;
        let as_mag = (fill.realized_as_bps.abs() / as_scale_bps).clamp(0.0, 2.0);
        let lr_as = if fill.is_buy { as_mag } else { -as_mag };
        state.dir_log_odds += self.config.dir_lambda_as * lr_as;
        state.dir_n_as += 1;
        state.dir_lr_sum_as += lr_as;

        state.dir_log_odds = state
            .dir_log_odds
            .clamp(-self.config.dir_max_log_odds, self.config.dir_max_log_odds);

        // Link prediction if we have order_id
        if let Some(oid) = fill.order_id {
            self.link_fill_prediction(state, oid, fill.realized_edge_bps > 0.0);
        }

        state.n_fills += 1;
        state.last_update_ms = fill.timestamp_ms;
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
            // Record timestamp when confirmation threshold is first reached
            if state.changepoint_consecutive_high == self.config.changepoint_min_confirmations {
                state.last_changepoint_confirmed_ms = state.last_update_ms;
            }
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

    fn process_cross_venue_update(&self, state: &mut InternalState, params: &CrossVenueParams) {
        // Update cross-venue state with smoothing
        // Use EMA smoothing for continuous values to reduce noise
        let alpha = 0.3; // Smoothing factor

        if state.cv_is_valid {
            // Smooth updates for continuous values
            state.cv_direction = (1.0 - alpha) * state.cv_direction + alpha * params.direction;
            state.cv_confidence = (1.0 - alpha) * state.cv_confidence + alpha * params.confidence;
            state.cv_discovery_venue =
                (1.0 - alpha) * state.cv_discovery_venue + alpha * params.discovery_venue;
            state.cv_max_toxicity =
                (1.0 - alpha) * state.cv_max_toxicity + alpha * params.max_toxicity;
            state.cv_avg_toxicity =
                (1.0 - alpha) * state.cv_avg_toxicity + alpha * params.avg_toxicity;
            state.cv_agreement = (1.0 - alpha) * state.cv_agreement + alpha * params.agreement;
            state.cv_divergence = (1.0 - alpha) * state.cv_divergence + alpha * params.divergence;
            state.cv_intensity_ratio =
                (1.0 - alpha) * state.cv_intensity_ratio + alpha * params.intensity_ratio;
            state.cv_imbalance_correlation = (1.0 - alpha) * state.cv_imbalance_correlation
                + alpha * params.imbalance_correlation;
        } else {
            // First update - initialize directly
            state.cv_direction = params.direction;
            state.cv_confidence = params.confidence;
            state.cv_discovery_venue = params.discovery_venue;
            state.cv_max_toxicity = params.max_toxicity;
            state.cv_avg_toxicity = params.avg_toxicity;
            state.cv_agreement = params.agreement;
            state.cv_divergence = params.divergence;
            state.cv_intensity_ratio = params.intensity_ratio;
            state.cv_imbalance_correlation = params.imbalance_correlation;
        }

        // Boolean alerts are not smoothed
        state.cv_toxicity_alert = params.toxicity_alert;
        state.cv_divergence_alert = params.divergence_alert;
        state.cv_is_valid = true;
        state.cv_observation_count += 1;
        state.cv_last_update_ms = params.timestamp_ms;
    }

    fn soft_reset(&self, state: &mut InternalState, retention: f64) {
        // Decay all posteriors toward prior
        let r = retention.clamp(0.0, 1.0);

        // Directional log-odds
        state.dir_log_odds *= r;
        state.dir_lr_sum_price *= r;
        state.dir_lr_sum_fill *= r;
        state.dir_lr_sum_as *= r;
        state.dir_lr_sum_flow *= r;
        state.sigma_sum_sq *= r;
        state.sigma_n *= r;

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
        state.dir_log_odds *= d;
        state.sigma_sum_sq *= d;
        state.sigma_n *= d;
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
            microstructure: self.build_microstructure_beliefs(state),
            cross_venue: self.build_cross_venue_beliefs(state),
            calibration: self.build_calibration_state(state),
            fair_value: self.build_fair_value_beliefs(state),
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
        let l = state.dir_log_odds;
        let prob_bearish = sigmoid(l);
        let prob_bullish = 1.0 - prob_bearish;

        // Confidence: directional strength × evidence ramp
        let total_n =
            (state.dir_n_price + state.dir_n_fill + state.dir_n_as + state.dir_n_flow) as f64;
        let dir_conf = 1.0 - 2.0 * prob_bearish.min(prob_bullish);
        let evidence_conf = (total_n / 50.0).min(1.0);
        let confidence = (dir_conf * evidence_conf).clamp(0.0, 1.0);

        // Sigma from running variance
        let expected_sigma = if state.sigma_n > 5.0 {
            (state.sigma_sum_sq / state.sigma_n).sqrt().max(0.001)
        } else {
            0.02
        };

        // Expected drift: smooth mapping from log-odds via tanh
        // L > 0 (bearish) → drift < 0; bounded by ±sigma
        let expected_drift = -expected_sigma * (l / 2.0).tanh();

        // Drift uncertainty: shrinks with evidence
        let drift_uncertainty = expected_sigma / (1.0 + total_n / 10.0).sqrt();

        // Phase 2A: Sigma skewness/kurtosis (depends on n_price_obs)
        let ig_alpha = (state.n_price_obs as f64) / 2.0 + 2.0;
        let sigma_skewness = if ig_alpha > 3.0 {
            (4.0 * (2.0 / (ig_alpha - 3.0)).sqrt()).min(3.0)
        } else {
            2.0
        };
        let sigma_kurtosis = if ig_alpha > 4.0 {
            (6.0 * (5.0 * ig_alpha - 11.0) / ((ig_alpha - 3.0) * (ig_alpha - 4.0))).min(10.0)
        } else {
            5.0
        };

        // Drift skewness from source asymmetry
        let drift_skewness = if total_n > 10.0 {
            let fill_frac = (state.dir_lr_sum_fill.abs() + state.dir_lr_sum_as.abs())
                / (state.dir_lr_sum_price.abs()
                    + state.dir_lr_sum_fill.abs()
                    + state.dir_lr_sum_as.abs()
                    + state.dir_lr_sum_flow.abs()
                    + 1e-10);
            fill_frac.clamp(0.0, 1.0) * l.signum() * 0.5
        } else {
            0.0
        };

        DriftVolatilityBeliefs {
            expected_drift,
            drift_uncertainty,
            expected_sigma,
            prob_bearish,
            prob_bullish,
            confidence,
            n_observations: state.n_price_obs,
            sigma_skewness,
            sigma_kurtosis,
            drift_skewness,
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

        // Compute uncertainty metrics
        let kappa_std = state.kappa_variance.sqrt().max(10.0);

        // Compute spread CI using delta method
        // δ* = (1/γ) × ln(1 + γ/κ)
        // ∂δ*/∂κ ≈ -1/(κ(κ+γ)) for typical γ
        let gamma = 0.5; // Default gamma for CI calculation
        let kappa = state.kappa_smoothed.max(100.0);
        let d_delta_d_kappa = -1.0 / (kappa * (kappa + gamma));
        let spread_std = (d_delta_d_kappa.powi(2) * state.kappa_variance).sqrt() * 10000.0; // Convert to bps

        // 95% CI: mean ± 1.96 × std
        let base_spread_bps = (1.0 / gamma) * (1.0 + gamma / kappa).ln() * 10000.0;
        let spread_ci_lower = (base_spread_bps - 1.96 * spread_std).max(1.0);
        let spread_ci_upper = (base_spread_bps + 1.96 * spread_std).min(50.0);

        // Compute kappa-sigma correlation
        let kappa_sigma_corr = if state.kappa_variance > 1e-12 && state.sigma_variance > 1e-12 {
            let denom = (state.kappa_variance * state.sigma_variance).sqrt();
            (state.kappa_sigma_covariance / denom).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // Phase 2A: Compute skew-adjusted CIs
        // Use sigma_skewness from drift_vol to adjust spread CIs asymmetrically
        // Positive skew (vol spike risk) => widen upper CI, tighten lower CI
        let skew_sensitivity = 0.1; // 10% adjustment per unit of skewness
        let sigma_skew = state.sigma_skewness; // Tracked in InternalState
        let skew_factor = (sigma_skew * skew_sensitivity).tanh();

        // Skew-adjusted CIs: asymmetric based on vol spike risk
        let spread_ci_lower_skew_adjusted = (spread_ci_lower * (1.0 - skew_factor * 0.5)).max(1.0);
        let spread_ci_upper_skew_adjusted = (spread_ci_upper * (1.0 + skew_factor * 0.5)).min(50.0);

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
            // Uncertainty fields
            kappa_std,
            spread_ci_lower,
            spread_ci_upper,
            kappa_sigma_corr,
            // Phase 2A: Skew-adjusted CIs
            spread_ci_lower_skew_adjusted,
            spread_ci_upper_skew_adjusted,
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
        let p_fill_raw =
            state.continuation_alpha / (state.continuation_alpha + state.continuation_beta);

        // Regime prior
        let p_regime = state.regime_probs[0] * 0.3
            + state.regime_probs[1] * 0.5
            + state.regime_probs[2] * 0.65
            + state.regime_probs[3] * 0.8;

        // Changepoint discount
        let cp_prob: f64 = state.changepoint_run_probs.iter().take(5).sum();
        let changepoint_discount = (cp_prob * 0.5).clamp(0.0, 0.8);

        // Fused probability
        let p_fill_discounted =
            (1.0 - changepoint_discount) * p_fill_raw + changepoint_discount * p_regime;
        let p_trend = 0.5 + 0.5 * state.trend_agreement;

        let fill_conf = 1.0
            - (state.continuation_alpha * state.continuation_beta)
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

        // Determine result with configurable confirmation gate and cooldown
        let result = if state.changepoint_obs_count < 10
            || prob_5 <= self.config.changepoint_threshold
        {
            ChangepointResult::None
        } else if state.changepoint_consecutive_high >= self.config.changepoint_min_confirmations {
            // Check cooldown: suppress confirmed if too soon after last one
            let elapsed_ms = state
                .last_update_ms
                .saturating_sub(state.last_changepoint_confirmed_ms);
            if self.config.changepoint_cooldown_ms > 0
                && state.last_changepoint_confirmed_ms > 0
                && elapsed_ms < self.config.changepoint_cooldown_ms
            {
                // Still in cooldown — demote to Pending instead of Confirmed
                ChangepointResult::Pending(state.changepoint_consecutive_high)
            } else {
                // Note: last_changepoint_confirmed_ms is set in process_changepoint_obs
                // when consecutive_high first reaches the confirmation threshold.
                ChangepointResult::Confirmed
            }
        } else {
            ChangepointResult::Pending(state.changepoint_consecutive_high)
        };

        // Learning trust decreases with changepoint probability
        let learning_trust = (1.0 - prob_10).max(0.2);

        // Warmup: BOCD needs ~10 observations for reliable detection
        let observation_count = state.changepoint_obs_count;
        let is_warmed_up = observation_count >= 10;

        ChangepointBeliefs {
            prob_1,
            prob_5,
            prob_10,
            run_length,
            entropy,
            result,
            learning_trust,
            observation_count,
            is_warmed_up,
        }
    }

    fn build_edge_beliefs(&self, state: &InternalState) -> EdgeBeliefs {
        let n = state.edge_n.max(1.0);
        let mean = state.edge_sum / n;
        let variance = (state.edge_sum_sq / n - mean * mean).max(0.0);
        let raw_std = variance.sqrt();

        // Bound the uncertainty to reasonable ranges:
        // - Min 0.5 bps: even with perfect data, there's inherent market uncertainty
        // - Max 20 bps: beyond this, the estimate is essentially uninformative
        // - Scale by 1/sqrt(n) to reflect sample size (shrink uncertainty with more data)
        let sample_factor = (10.0 / n.max(1.0)).sqrt().min(3.0); // Converges to ~1 at n=10
        let std = (raw_std * sample_factor).clamp(0.5, 20.0);

        // P(positive edge)
        let z = mean / std.max(0.1);
        let p_positive = normal_cdf(z);

        // Compute combined toxicity score from VPIN + soft_jump proxy (via liquidity evaporation)
        // Weight: VPIN 0.6, evaporation 0.4
        let toxicity_score = 0.6 * state.vpin + 0.4 * state.liquidity_evaporation;

        // Toxicity-adjusted edge: expected_edge × (1 - α × toxicity)
        // α = 0.7 (toxicity penalty coefficient)
        const TOXICITY_PENALTY_ALPHA: f64 = 0.7;
        let toxicity_penalty = TOXICITY_PENALTY_ALPHA * toxicity_score;
        let toxicity_adjusted_edge = mean * (1.0 - toxicity_penalty);

        // P(positive adjusted edge)
        let z_adj = toxicity_adjusted_edge / std.max(0.1);
        let p_positive_adjusted = normal_cdf(z_adj);

        // Should we quote? Conditions:
        // 1. Toxicity not extreme (< 0.8)
        // 2. Adjusted edge > minimum threshold (-1 bps)
        // 3. P(positive adjusted) > 0.3
        let should_quote =
            toxicity_score < 0.8 && toxicity_adjusted_edge > -1.0 && p_positive_adjusted > 0.3;

        EdgeBeliefs {
            expected_edge: mean,
            toxicity_adjusted_edge,
            toxicity_score,
            uncertainty: std,
            by_regime: state.edge_by_regime,
            p_positive,
            p_positive_adjusted,
            as_bias: state.as_bias,
            epistemic_uncertainty: 0.5, // Default
            should_quote,
        }
    }

    fn build_microstructure_beliefs(&self, state: &InternalState) -> MicrostructureBeliefs {
        let is_valid = state.vpin_buckets >= 10 && state.vpin_confidence > 0.3;

        MicrostructureBeliefs {
            vpin: state.vpin,
            vpin_velocity: state.vpin_velocity,
            depth_ofi: state.depth_ofi,
            liquidity_evaporation: state.liquidity_evaporation,
            order_flow_direction: state.order_flow_direction,
            confidence: state.vpin_confidence,
            vpin_buckets: state.vpin_buckets,
            is_valid,
            // Phase 1A fields
            trade_size_sigma: state.trade_size_sigma,
            toxicity_acceleration: state.toxicity_acceleration,
            cofi: state.cofi,
            cofi_velocity: state.cofi_velocity,
            is_sustained_shift: state.is_sustained_shift,
        }
    }

    fn build_cross_venue_beliefs(&self, state: &InternalState) -> CrossVenueBeliefs {
        CrossVenueBeliefs {
            direction: state.cv_direction,
            confidence: state.cv_confidence,
            discovery_venue: state.cv_discovery_venue,
            max_toxicity: state.cv_max_toxicity,
            avg_toxicity: state.cv_avg_toxicity,
            agreement: state.cv_agreement,
            divergence: state.cv_divergence,
            intensity_ratio: state.cv_intensity_ratio,
            imbalance_correlation: state.cv_imbalance_correlation,
            toxicity_alert: state.cv_toxicity_alert,
            divergence_alert: state.cv_divergence_alert,
            is_valid: state.cv_is_valid,
            observation_count: state.cv_observation_count,
            last_update_ms: state.cv_last_update_ms,
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
            latency: LatencyCalibration::default(),
        }
    }

    fn build_fair_value_beliefs(&self, state: &InternalState) -> FairValueBeliefs {
        match &state.bayesian_fv {
            Some(fv) => fv.beliefs(),
            None => FairValueBeliefs::default(),
        }
    }

    fn compute_warmup_progress(&self, state: &InternalState) -> f64 {
        let price_progress = (state.n_price_obs as f64 / self.config.min_price_obs as f64).min(1.0);
        let fill_progress =
            (state.own_kappa_n_fills as f64 / self.config.min_fills as f64).min(1.0);
        let time_progress = (state.total_time / self.config.min_observation_time).min(1.0);

        (price_progress * fill_progress * time_progress).powf(1.0 / 3.0)
    }
}

/// Sigmoid function: maps log-odds to probability.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
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
                quoted_size: 1.0,
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
                quoted_size: 1.0,
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

    #[test]
    fn test_hip3_needs_4_confirmations() {
        let config = CentralBeliefConfig::hip3();
        assert_eq!(config.changepoint_min_confirmations, 4);
        assert_eq!(config.changepoint_cooldown_ms, 300_000);
    }

    #[test]
    fn test_default_needs_2_confirmations() {
        let config = CentralBeliefConfig::default();
        assert_eq!(config.changepoint_min_confirmations, 2);
        assert_eq!(config.changepoint_cooldown_ms, 0);
    }

    #[test]
    fn test_log_odds_fills_update_direction() {
        let beliefs = CentralBeliefState::default_config();

        // 10 buy fills (sell aggressor hit our bid → bearish → L increases)
        for i in 0..10 {
            beliefs.update(BeliefUpdate::OwnFill {
                price: 100.01,
                size: 1.0,
                mid: 100.0,
                is_buy: true,
                is_aligned: true,
                realized_as_bps: 2.0,
                realized_edge_bps: 1.0,
                timestamp_ms: i * 1000,
                order_id: Some(i),
                quoted_size: 1.0,
            });
        }

        let snapshot = beliefs.snapshot();
        assert!(
            snapshot.drift_vol.prob_bearish > 0.6,
            "Expected bearish bias from buy fills: prob_bearish={}",
            snapshot.drift_vol.prob_bearish
        );
    }

    #[test]
    fn test_log_odds_decay_to_neutral() {
        let beliefs = CentralBeliefState::default_config();

        // Push strong bearish evidence at t=1000
        for _ in 0..20 {
            beliefs.update(BeliefUpdate::PriceReturn {
                return_frac: -0.003,
                dt_secs: 1.0,
                timestamp_ms: 1000,
            });
        }

        let before = beliefs.snapshot();
        assert!(
            before.drift_vol.prob_bearish > 0.7,
            "Should be strongly bearish: {}",
            before.drift_vol.prob_bearish
        );

        // Advance 120 seconds (4 × tau=30s → decay ≈ e^(-4) ≈ 0.018)
        // with a neutral return to trigger decay
        beliefs.update(BeliefUpdate::PriceReturn {
            return_frac: 0.0,
            dt_secs: 0.1,
            timestamp_ms: 121_000,
        });

        let after = beliefs.snapshot();
        assert!(
            after.drift_vol.prob_bearish < before.drift_vol.prob_bearish,
            "Should decay toward neutral: before={}, after={}",
            before.drift_vol.prob_bearish,
            after.drift_vol.prob_bearish
        );
        assert!(
            (after.drift_vol.prob_bearish - 0.5).abs() < 0.15,
            "Should be near neutral after 4×tau decay: {}",
            after.drift_vol.prob_bearish
        );
    }

    #[test]
    fn test_log_odds_mixed_signals_cancel() {
        let beliefs = CentralBeliefState::default_config();

        // Alternating positive and negative returns at same timestamp (no decay)
        for _ in 0..50 {
            beliefs.update(BeliefUpdate::PriceReturn {
                return_frac: -0.001,
                dt_secs: 1.0,
                timestamp_ms: 1000,
            });
            beliefs.update(BeliefUpdate::PriceReturn {
                return_frac: 0.001,
                dt_secs: 1.0,
                timestamp_ms: 1000,
            });
        }

        let snapshot = beliefs.snapshot();
        assert!(
            (snapshot.drift_vol.prob_bearish - 0.5).abs() < 0.15,
            "Mixed signals should roughly cancel: prob_bearish={}",
            snapshot.drift_vol.prob_bearish
        );
    }

    #[test]
    fn test_confidence_ramps() {
        let beliefs = CentralBeliefState::default_config();

        // Zero evidence → confidence should be 0
        let empty = beliefs.snapshot();
        assert!(
            empty.drift_vol.confidence < 0.01,
            "No evidence → confidence ≈ 0: {}",
            empty.drift_vol.confidence
        );

        // 60 observations → confidence should be meaningful
        for i in 0..60 {
            beliefs.update(BeliefUpdate::PriceReturn {
                return_frac: -0.002,
                dt_secs: 1.0,
                timestamp_ms: i * 1000,
            });
        }

        let rich = beliefs.snapshot();
        assert!(
            rich.drift_vol.confidence > 0.3,
            "50+ evidence → confidence > 0.3: {}",
            rich.drift_vol.confidence
        );
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(5.0) > 0.99);
        assert!(sigmoid(-5.0) < 0.01);
        assert!((sigmoid(1.0) + sigmoid(-1.0) - 1.0).abs() < 1e-10);
    }
}
