//! Econometric parameter estimation for GLFT strategy from live market data.
//!
//! Features a robust HFT pipeline:
//! - Volume Clock: Adaptive volume-based sampling (normalizes by economic activity)
//! - VWAP Pre-Averaging: Filters bid-ask bounce noise
//! - Bipower Variation: Jump-robust volatility estimation (σ = √BV)
//! - Regime Detection: RV/BV ratio identifies toxic jump regimes
//! - Weighted Kappa: Proximity-weighted L2 book regression

use std::collections::VecDeque;
use tracing::{debug, warn};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for econometric parameter estimation pipeline.
#[derive(Debug, Clone)]
pub struct EstimatorConfig {
    // === Volume Clock ===
    /// Initial volume bucket threshold (in asset units, e.g., 1.0 BTC)
    pub initial_bucket_volume: f64,
    /// Rolling window for adaptive bucket calculation (seconds)
    pub volume_window_secs: f64,
    /// Percentile of rolling volume for adaptive bucket (e.g., 0.01 = 1%)
    pub volume_percentile: f64,
    /// Minimum bucket volume floor
    pub min_bucket_volume: f64,
    /// Maximum bucket volume ceiling
    pub max_bucket_volume: f64,

    // === Multi-Timescale EWMA ===
    /// Fast half-life for variance EWMA (~2 seconds, reacts to crashes)
    pub fast_half_life_ticks: f64,
    /// Medium half-life for variance EWMA (~10 seconds)
    pub medium_half_life_ticks: f64,
    /// Slow half-life for variance EWMA (~60 seconds, baseline)
    pub slow_half_life_ticks: f64,
    /// Half-life for kappa EWMA (in L2 updates)
    pub kappa_half_life_updates: f64,

    // === Momentum Detection ===
    /// Window for momentum calculation (milliseconds)
    pub momentum_window_ms: u64,

    // === Trade Flow Tracking ===
    /// Window for trade flow imbalance calculation (milliseconds)
    pub trade_flow_window_ms: u64,
    /// EWMA alpha for smoothing flow imbalance
    pub trade_flow_alpha: f64,

    // === Regime Detection ===
    /// Jump detection threshold: RV/BV ratio > threshold = toxic regime
    pub jump_ratio_threshold: f64,

    // === Kappa Estimation ===
    /// Maximum distance from mid for kappa regression (as fraction, e.g., 0.01 = 1%)
    pub kappa_max_distance: f64,
    /// Maximum number of L2 levels to use per side
    pub kappa_max_levels: usize,

    // === Warmup ===
    /// Minimum volume ticks before volatility estimates are valid
    pub min_volume_ticks: usize,
    /// Minimum L2 updates before kappa is valid
    pub min_l2_updates: usize,

    // === Defaults ===
    /// Default sigma during warmup (per-second volatility)
    pub default_sigma: f64,
    /// Default kappa during warmup (order book decay)
    pub default_kappa: f64,
    /// Default arrival intensity during warmup (ticks per second)
    pub default_arrival_intensity: f64,
}

impl Default for EstimatorConfig {
    fn default() -> Self {
        Self {
            // Volume Clock - tuned for testnet/low-activity markets
            initial_bucket_volume: 0.01, // Start very small (0.01 BTC)
            volume_window_secs: 300.0,   // 5 minutes
            volume_percentile: 0.01,     // 1% of 5-min volume
            min_bucket_volume: 0.001,    // Floor at 0.001 BTC
            max_bucket_volume: 10.0,     // Cap at 10 BTC

            // Multi-Timescale EWMA
            fast_half_life_ticks: 5.0,     // ~2 seconds - reacts to crashes
            medium_half_life_ticks: 20.0,  // ~10 seconds
            slow_half_life_ticks: 100.0,   // ~60 seconds - baseline
            kappa_half_life_updates: 30.0, // 30 L2 updates

            // Momentum Detection
            momentum_window_ms: 500, // Track signed returns over 500ms

            // Trade Flow Tracking
            trade_flow_window_ms: 1000, // Track buy/sell imbalance over 1s
            trade_flow_alpha: 0.1,      // EWMA smoothing for flow

            // Regime Detection - true toxic flow detection
            jump_ratio_threshold: 3.0, // RV/BV > 3.0 = toxic (lower values are normal bid-ask bounce)

            // Kappa
            kappa_max_distance: 0.01, // 1% from mid
            kappa_max_levels: 15,

            // Warmup - reasonable for testnet/low-activity
            min_volume_ticks: 10,
            min_l2_updates: 5,

            // Defaults
            default_sigma: 0.0001,          // 0.01% per-second
            default_kappa: 100.0,           // Moderate depth decay
            default_arrival_intensity: 0.5, // 0.5 ticks per second
        }
    }
}

impl EstimatorConfig {
    /// Create config from legacy fields (for backward compatibility with bin/market_maker.rs)
    pub fn from_legacy(
        _window_ms: u64,
        _min_trades: usize,
        default_sigma: f64,
        default_kappa: f64,
        default_arrival_intensity: f64,
        _decay_secs: u64,
        min_warmup_trades: usize,
    ) -> Self {
        Self {
            default_sigma,
            default_kappa,
            default_arrival_intensity,
            min_volume_ticks: min_warmup_trades.max(10),
            ..Default::default()
        }
    }

    /// Create config with custom toxic threshold
    pub fn with_toxic_threshold(mut self, threshold: f64) -> Self {
        self.jump_ratio_threshold = threshold;
        self
    }
}

// ============================================================================
// Volume Clock (Data Normalization)
// ============================================================================

/// A completed volume bucket with VWAP.
#[derive(Debug, Clone)]
struct VolumeBucket {
    /// Start timestamp of bucket (ms)
    start_time_ms: u64,
    /// End timestamp of bucket (ms)
    end_time_ms: u64,
    /// VWAP: sum(price * size) / sum(size)
    vwap: f64,
    /// Total volume in bucket
    volume: f64,
}

/// Accumulates trades until volume threshold is reached.
/// Implements adaptive volume clock for normalized economic sampling.
#[derive(Debug)]
struct VolumeBucketAccumulator {
    /// Current bucket start time
    start_time_ms: Option<u64>,
    /// Sum of (price * size) for current bucket
    price_volume_sum: f64,
    /// Sum of sizes for current bucket
    volume_sum: f64,
    /// Current adaptive threshold
    threshold: f64,
    /// Rolling volume tracker for adaptive threshold: (timestamp_ms, volume)
    rolling_volumes: VecDeque<(u64, f64)>,
    /// Config reference
    initial_bucket_volume: f64,
    volume_window_secs: f64,
    volume_percentile: f64,
    min_bucket_volume: f64,
    max_bucket_volume: f64,
}

impl VolumeBucketAccumulator {
    fn new(config: &EstimatorConfig) -> Self {
        Self {
            start_time_ms: None,
            price_volume_sum: 0.0,
            volume_sum: 0.0,
            threshold: config.initial_bucket_volume,
            rolling_volumes: VecDeque::new(),
            initial_bucket_volume: config.initial_bucket_volume,
            volume_window_secs: config.volume_window_secs,
            volume_percentile: config.volume_percentile,
            min_bucket_volume: config.min_bucket_volume,
            max_bucket_volume: config.max_bucket_volume,
        }
    }

    /// Add a trade. Returns Some(VolumeBucket) if bucket completed.
    fn on_trade(&mut self, time_ms: u64, price: f64, size: f64) -> Option<VolumeBucket> {
        if price <= 0.0 || size <= 0.0 {
            return None;
        }

        // Initialize bucket start if needed
        if self.start_time_ms.is_none() {
            self.start_time_ms = Some(time_ms);
        }

        // Accumulate
        self.price_volume_sum += price * size;
        self.volume_sum += size;

        // Update rolling volumes for adaptive threshold
        self.rolling_volumes.push_back((time_ms, size));
        let cutoff = time_ms.saturating_sub((self.volume_window_secs * 1000.0) as u64);
        while self
            .rolling_volumes
            .front()
            .map(|(t, _)| *t < cutoff)
            .unwrap_or(false)
        {
            self.rolling_volumes.pop_front();
        }

        // Check if bucket is complete
        if self.volume_sum >= self.threshold {
            let vwap = self.price_volume_sum / self.volume_sum;
            let bucket = VolumeBucket {
                start_time_ms: self.start_time_ms.unwrap(),
                end_time_ms: time_ms,
                vwap,
                volume: self.volume_sum,
            };

            // Reset for next bucket
            self.price_volume_sum = 0.0;
            self.volume_sum = 0.0;
            self.start_time_ms = None;

            // Update adaptive threshold based on rolling volume
            self.update_adaptive_threshold();

            Some(bucket)
        } else {
            None
        }
    }

    /// Calculate adaptive threshold as percentile of recent volume.
    fn update_adaptive_threshold(&mut self) {
        let total_rolling: f64 = self.rolling_volumes.iter().map(|(_, v)| v).sum();

        if total_rolling > 0.0 {
            let target = total_rolling * self.volume_percentile;
            self.threshold = target.clamp(self.min_bucket_volume, self.max_bucket_volume);
        } else {
            self.threshold = self.initial_bucket_volume;
        }
    }

    /// Get current bucket fill progress (0.0 to 1.0+)
    #[allow(dead_code)]
    fn bucket_progress(&self) -> f64 {
        if self.threshold > 0.0 {
            self.volume_sum / self.threshold
        } else {
            0.0
        }
    }
}

// ============================================================================
// Single-Scale Bipower Variation (Building Block)
// ============================================================================

/// Single-timescale RV/BV tracker - building block for multi-scale estimator.
#[derive(Debug)]
struct SingleScaleBipower {
    /// EWMA decay factor (per tick)
    alpha: f64,
    /// Realized variance (includes jumps): EWMA of r²
    rv: f64,
    /// Bipower variation (excludes jumps): EWMA of (π/2)|r_t||r_{t-1}|
    bv: f64,
    /// Last absolute log return (for BV calculation)
    last_abs_return: Option<f64>,
}

impl SingleScaleBipower {
    fn new(half_life_ticks: f64, default_var: f64) -> Self {
        Self {
            alpha: (2.0_f64.ln() / half_life_ticks).clamp(0.001, 1.0),
            rv: default_var,
            bv: default_var,
            last_abs_return: None,
        }
    }

    fn update(&mut self, log_return: f64) {
        let abs_return = log_return.abs();

        // RV: EWMA of r²
        let rv_obs = log_return.powi(2);
        self.rv = self.alpha * rv_obs + (1.0 - self.alpha) * self.rv;

        // BV: EWMA of (π/2) × |r_t| × |r_{t-1}|
        if let Some(last_abs) = self.last_abs_return {
            let bv_obs = std::f64::consts::FRAC_PI_2 * abs_return * last_abs;
            self.bv = self.alpha * bv_obs + (1.0 - self.alpha) * self.bv;
        }

        self.last_abs_return = Some(abs_return);
    }

    /// Total volatility including jumps (√RV)
    fn sigma_total(&self) -> f64 {
        self.rv.sqrt().clamp(1e-7, 0.05)
    }

    /// Clean volatility excluding jumps (√BV)
    fn sigma_clean(&self) -> f64 {
        self.bv.sqrt().clamp(1e-7, 0.05)
    }

    /// Jump ratio: RV/BV (1.0 = normal, >2 = jumps)
    fn jump_ratio(&self) -> f64 {
        if self.bv > 1e-12 {
            (self.rv / self.bv).clamp(0.1, 100.0)
        } else {
            1.0
        }
    }
}

// ============================================================================
// Multi-Timescale Bipower Estimator
// ============================================================================

/// Multi-timescale volatility with fast/medium/slow components.
///
/// Fast (~2s): Reacts quickly to crashes, used for early warning
/// Medium (~10s): Balanced responsiveness
/// Slow (~60s): Stable baseline for pricing
#[derive(Debug)]
struct MultiScaleBipowerEstimator {
    fast: SingleScaleBipower,   // ~5 ticks / ~2 seconds
    medium: SingleScaleBipower, // ~20 ticks / ~10 seconds
    slow: SingleScaleBipower,   // ~100 ticks / ~60 seconds
    last_vwap: Option<f64>,
    tick_count: usize,
}

impl MultiScaleBipowerEstimator {
    fn new(config: &EstimatorConfig) -> Self {
        let default_var = config.default_sigma.powi(2);
        Self {
            fast: SingleScaleBipower::new(config.fast_half_life_ticks, default_var),
            medium: SingleScaleBipower::new(config.medium_half_life_ticks, default_var),
            slow: SingleScaleBipower::new(config.slow_half_life_ticks, default_var),
            last_vwap: None,
            tick_count: 0,
        }
    }

    /// Process a completed volume bucket.
    fn on_bucket(&mut self, bucket: &VolumeBucket) {
        if let Some(prev_vwap) = self.last_vwap {
            if bucket.vwap > 0.0 && prev_vwap > 0.0 {
                let log_return = (bucket.vwap / prev_vwap).ln();
                self.fast.update(log_return);
                self.medium.update(log_return);
                self.slow.update(log_return);
                self.tick_count += 1;
            }
        }
        self.last_vwap = Some(bucket.vwap);
    }

    /// Get the log return for the most recent bucket (for momentum tracking)
    fn last_log_return(&self, bucket: &VolumeBucket) -> Option<f64> {
        self.last_vwap.and_then(|prev| {
            if bucket.vwap > 0.0 && prev > 0.0 {
                Some((bucket.vwap / prev).ln())
            } else {
                None
            }
        })
    }

    /// Clean sigma (BV-based) for spread pricing.
    /// Uses slow timescale for stability.
    fn sigma_clean(&self) -> f64 {
        self.slow.sigma_clean()
    }

    /// Total sigma (RV-based) for risk assessment.
    /// Blends fast + slow: uses fast when market is accelerating.
    fn sigma_total(&self) -> f64 {
        let fast = self.fast.sigma_total();
        let slow = self.slow.sigma_total();

        // If fast >> slow, market is accelerating - trust fast more
        let ratio = fast / slow.max(1e-9);

        if ratio > 1.5 {
            // Acceleration: blend toward fast
            let weight = ((ratio - 1.0) / 3.0).clamp(0.0, 0.7);
            weight * fast + (1.0 - weight) * slow
        } else {
            // Stable: prefer slow for less noise
            0.2 * fast + 0.8 * slow
        }
    }

    /// Effective sigma for inventory skew.
    /// Blends clean and total based on jump regime.
    fn sigma_effective(&self) -> f64 {
        let clean = self.sigma_clean();
        let total = self.sigma_total();
        let jump_ratio = self.jump_ratio_fast();

        // At ratio=1: pure clean (no jumps)
        // At ratio=3: 67% total (jumps dominant)
        // At ratio=5: 80% total
        let jump_weight = 1.0 - (1.0 / jump_ratio.max(1.0));
        let jump_weight = jump_weight.clamp(0.0, 0.85);

        (1.0 - jump_weight) * clean + jump_weight * total
    }

    /// Fast jump ratio (detects recent jumps quickly)
    fn jump_ratio_fast(&self) -> f64 {
        self.fast.jump_ratio()
    }

    /// Medium jump ratio (more stable signal)
    #[allow(dead_code)]
    fn jump_ratio_medium(&self) -> f64 {
        self.medium.jump_ratio()
    }

    fn tick_count(&self) -> usize {
        self.tick_count
    }
}

// ============================================================================
// Momentum Detector (Directional Flow)
// ============================================================================

/// Detects directional momentum from signed VWAP returns.
///
/// Tracks signed (not absolute) returns to detect falling/rising knife patterns.
#[derive(Debug)]
struct MomentumDetector {
    /// Recent (timestamp_ms, log_return) pairs
    returns: VecDeque<(u64, f64)>,
    /// Window for momentum calculation (ms)
    window_ms: u64,
}

impl MomentumDetector {
    fn new(window_ms: u64) -> Self {
        Self {
            returns: VecDeque::with_capacity(100),
            window_ms,
        }
    }

    /// Add a new VWAP-based return
    fn on_bucket(&mut self, end_time_ms: u64, log_return: f64) {
        self.returns.push_back((end_time_ms, log_return));

        // Expire old returns (keep 2x window for safety)
        let cutoff = end_time_ms.saturating_sub(self.window_ms * 2);
        while self
            .returns
            .front()
            .map(|(t, _)| *t < cutoff)
            .unwrap_or(false)
        {
            self.returns.pop_front();
        }
    }

    /// Signed momentum in bps over the configured window
    fn momentum_bps(&self, now_ms: u64) -> f64 {
        let cutoff = now_ms.saturating_sub(self.window_ms);
        let sum: f64 = self
            .returns
            .iter()
            .filter(|(t, _)| *t >= cutoff)
            .map(|(_, r)| r)
            .sum();
        sum * 10_000.0 // Convert to bps
    }

    /// Falling knife score: 0 = normal, 1+ = severe downward momentum
    fn falling_knife_score(&self, now_ms: u64) -> f64 {
        let momentum = self.momentum_bps(now_ms);

        // Only trigger on negative momentum
        if momentum >= 0.0 {
            return 0.0;
        }

        // Score: -20 bps = 1.0, -40 bps = 2.0, etc.
        (momentum.abs() / 20.0).clamp(0.0, 3.0)
    }

    /// Rising knife score (for protecting asks during pumps)
    fn rising_knife_score(&self, now_ms: u64) -> f64 {
        let momentum = self.momentum_bps(now_ms);

        if momentum <= 0.0 {
            return 0.0;
        }

        (momentum / 20.0).clamp(0.0, 3.0)
    }
}

// ============================================================================
// Jump Process Estimator (First Principles Gap 1)
// ============================================================================

/// Configuration for jump process estimation.
#[derive(Debug, Clone)]
pub struct JumpEstimatorConfig {
    /// Threshold for jump detection (in sigma units, e.g., 3.0 = 3σ)
    pub jump_threshold_sigmas: f64,
    /// Window for jump intensity estimation (ms)
    pub window_ms: u64,
    /// EWMA alpha for online parameter updates
    pub alpha: f64,
    /// Minimum jumps before estimates are valid
    pub min_jumps: usize,
}

impl Default for JumpEstimatorConfig {
    fn default() -> Self {
        Self {
            jump_threshold_sigmas: 3.0,
            window_ms: 300_000, // 5 minutes
            alpha: 0.1,
            min_jumps: 5,
        }
    }
}

/// Jump process estimator for explicit λ (intensity), μ_j (mean), σ_j (std dev).
///
/// Implements the jump component of the price process:
/// dP = μ dt + σ dW + J dN where J ~ N(μ_j, σ_j²), N ~ Poisson(λ)
///
/// Key formulas:
/// - Total variance over horizon h: Var[P(t+h) - P(t)] = σ²h + λh×E[J²]
/// - E[J²] = μ_j² + σ_j²
#[derive(Debug)]
pub struct JumpEstimator {
    /// Jump intensity (jumps per second)
    lambda_jump: f64,
    /// Mean jump size (in log-return units)
    mu_jump: f64,
    /// Jump size standard deviation
    sigma_jump: f64,
    /// Recent detected jumps: (timestamp_ms, size, is_positive)
    recent_jumps: VecDeque<(u64, f64, bool)>,
    /// Online mean tracker
    sum_sizes: f64,
    sum_sq_sizes: f64,
    jump_count: usize,
    /// Configuration
    config: JumpEstimatorConfig,
    /// Last update timestamp
    last_update_ms: u64,
}

impl JumpEstimator {
    /// Create a new jump estimator with default configuration.
    pub fn new() -> Self {
        Self::with_config(JumpEstimatorConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: JumpEstimatorConfig) -> Self {
        Self {
            lambda_jump: 0.0,
            mu_jump: 0.0,
            sigma_jump: 0.0001, // Small default
            recent_jumps: VecDeque::with_capacity(100),
            sum_sizes: 0.0,
            sum_sq_sizes: 0.0,
            jump_count: 0,
            config,
            last_update_ms: 0,
        }
    }

    /// Check if a return qualifies as a jump.
    ///
    /// A return is a "jump" if |return| > threshold × σ_clean
    pub fn is_jump(&self, log_return: f64, sigma_clean: f64) -> bool {
        let threshold = self.config.jump_threshold_sigmas * sigma_clean;
        log_return.abs() > threshold
    }

    /// Record a detected jump and update parameters.
    pub fn record_jump(&mut self, timestamp_ms: u64, log_return: f64) {
        let size = log_return.abs();
        let is_positive = log_return > 0.0;

        self.recent_jumps
            .push_back((timestamp_ms, size, is_positive));

        // Update online statistics
        self.sum_sizes += size;
        self.sum_sq_sizes += size * size;
        self.jump_count += 1;

        // Expire old jumps
        let cutoff = timestamp_ms.saturating_sub(self.config.window_ms);
        while self
            .recent_jumps
            .front()
            .map(|(t, _, _)| *t < cutoff)
            .unwrap_or(false)
        {
            if let Some((_, old_size, _)) = self.recent_jumps.pop_front() {
                self.sum_sizes -= old_size;
                self.sum_sq_sizes -= old_size * old_size;
                self.jump_count = self.jump_count.saturating_sub(1);
            }
        }

        // Update parameters using EWMA
        self.update_parameters(timestamp_ms);
        self.last_update_ms = timestamp_ms;
    }

    /// Update lambda, mu, sigma from recent jumps.
    fn update_parameters(&mut self, _timestamp_ms: u64) {
        if self.jump_count < self.config.min_jumps {
            return;
        }

        let n = self.jump_count as f64;
        let window_secs = self.config.window_ms as f64 / 1000.0;

        // Lambda: jumps per second
        let new_lambda = n / window_secs;
        self.lambda_jump =
            self.config.alpha * new_lambda + (1.0 - self.config.alpha) * self.lambda_jump;

        // Mu: mean jump size (signed average would be near 0, use unsigned)
        let new_mu = self.sum_sizes / n;
        self.mu_jump = self.config.alpha * new_mu + (1.0 - self.config.alpha) * self.mu_jump;

        // Sigma: standard deviation of jump sizes
        let variance = (self.sum_sq_sizes / n) - (new_mu * new_mu);
        if variance > 0.0 {
            let new_sigma = variance.sqrt();
            self.sigma_jump =
                self.config.alpha * new_sigma + (1.0 - self.config.alpha) * self.sigma_jump;
        }
    }

    /// Process a return observation (called on each volume bucket).
    pub fn on_return(&mut self, timestamp_ms: u64, log_return: f64, sigma_clean: f64) {
        if self.is_jump(log_return, sigma_clean) {
            self.record_jump(timestamp_ms, log_return);
        }
    }

    /// Get jump intensity (λ) - jumps per second.
    pub fn lambda(&self) -> f64 {
        self.lambda_jump
    }

    /// Get mean jump size (μ_j) in log-return units.
    pub fn mu(&self) -> f64 {
        self.mu_jump
    }

    /// Get jump size standard deviation (σ_j).
    pub fn sigma(&self) -> f64 {
        self.sigma_jump
    }

    /// Get expected jump variance contribution: E[J²] = μ² + σ².
    pub fn expected_jump_variance(&self) -> f64 {
        self.mu_jump.powi(2) + self.sigma_jump.powi(2)
    }

    /// Get total variance over horizon including jumps.
    ///
    /// Var[P(t+h) - P(t)] = σ_diffusion² × h + λ × h × E[J²]
    pub fn total_variance(&self, sigma_diffusion: f64, horizon_secs: f64) -> f64 {
        let diffusion_var = sigma_diffusion.powi(2) * horizon_secs;
        let jump_var = self.lambda_jump * horizon_secs * self.expected_jump_variance();
        diffusion_var + jump_var
    }

    /// Get total volatility (sqrt of total variance).
    pub fn total_sigma(&self, sigma_diffusion: f64, horizon_secs: f64) -> f64 {
        self.total_variance(sigma_diffusion, horizon_secs).sqrt()
    }

    /// Check if estimator has enough data.
    pub fn is_warmed_up(&self) -> bool {
        self.jump_count >= self.config.min_jumps
    }

    /// Get number of jumps in current window.
    pub fn jump_count(&self) -> usize {
        self.jump_count
    }

    /// Get fraction of recent returns that were jumps.
    pub fn jump_fraction(&self, total_returns: usize) -> f64 {
        if total_returns == 0 {
            0.0
        } else {
            self.jump_count as f64 / total_returns as f64
        }
    }
}

impl Default for JumpEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Stochastic Volatility Parameters (First Principles Gap 2)
// ============================================================================

/// Stochastic volatility model parameters (Heston-style OU process).
///
/// Models volatility as a mean-reverting process:
/// dσ² = κ(θ - σ²)dt + ξσ² dZ  with Corr(dW, dZ) = ρ
///
/// Key formulas:
/// - Expected avg variance: E[∫₀ᵀ σ² dt]/T = θ + (σ₀² - θ)(1 - e^(-κT))/(κT)
/// - Leverage effect: Vol increases when returns are negative (ρ < 0)
#[derive(Debug, Clone)]
pub struct StochasticVolParams {
    /// Current instantaneous variance (σ²)
    v_t: f64,
    /// Mean-reversion speed (κ)
    kappa_vol: f64,
    /// Long-run variance (θ)
    theta_vol: f64,
    /// Vol-of-vol (ξ)
    xi_vol: f64,
    /// Price-vol correlation (ρ, typically negative "leverage effect")
    rho: f64,
    /// Variance history for calibration: (timestamp_ms, variance)
    variance_history: VecDeque<(u64, f64)>,
    /// Return history for correlation estimation
    return_history: VecDeque<(u64, f64)>,
    /// EWMA alpha for updates
    alpha: f64,
    /// History window (ms)
    window_ms: u64,
    /// Minimum observations for calibration
    min_observations: usize,
}

impl StochasticVolParams {
    /// Create with default parameters.
    pub fn new(default_sigma: f64) -> Self {
        let default_var = default_sigma.powi(2);
        Self {
            v_t: default_var,
            kappa_vol: 0.5,         // Moderate mean-reversion
            theta_vol: default_var, // Long-run = current
            xi_vol: 0.1,            // 10% vol-of-vol
            rho: -0.3,              // Typical leverage effect
            variance_history: VecDeque::with_capacity(500),
            return_history: VecDeque::with_capacity(500),
            alpha: 0.05,
            window_ms: 300_000, // 5 minutes
            min_observations: 20,
        }
    }

    /// Update with new variance observation.
    pub fn on_variance(&mut self, timestamp_ms: u64, variance: f64, log_return: f64) {
        // Update current variance with EWMA
        self.v_t = self.alpha * variance + (1.0 - self.alpha) * self.v_t;

        // Store history
        self.variance_history.push_back((timestamp_ms, variance));
        self.return_history.push_back((timestamp_ms, log_return));

        // Expire old entries
        let cutoff = timestamp_ms.saturating_sub(self.window_ms);
        while self
            .variance_history
            .front()
            .map(|(t, _)| *t < cutoff)
            .unwrap_or(false)
        {
            self.variance_history.pop_front();
        }
        while self
            .return_history
            .front()
            .map(|(t, _)| *t < cutoff)
            .unwrap_or(false)
        {
            self.return_history.pop_front();
        }

        // Periodically calibrate parameters
        if self.variance_history.len() >= self.min_observations
            && self.variance_history.len().is_multiple_of(10)
        {
            self.calibrate();
        }
    }

    /// Calibrate κ, θ, ξ, ρ from history.
    fn calibrate(&mut self) {
        let n = self.variance_history.len();
        if n < self.min_observations {
            return;
        }

        // Theta: long-run mean of variance
        let sum_var: f64 = self.variance_history.iter().map(|(_, v)| v).sum();
        self.theta_vol = sum_var / n as f64;

        // Xi (vol-of-vol): std dev of variance changes
        let var_changes: Vec<f64> = self
            .variance_history
            .iter()
            .zip(self.variance_history.iter().skip(1))
            .map(|((_, v1), (_, v2))| v2 - v1)
            .collect();
        if var_changes.len() > 1 {
            let mean_change: f64 = var_changes.iter().sum::<f64>() / var_changes.len() as f64;
            let sum_sq: f64 = var_changes.iter().map(|c| (c - mean_change).powi(2)).sum();
            self.xi_vol = (sum_sq / var_changes.len() as f64).sqrt();
        }

        // Kappa: estimate from autocorrelation decay
        // Simplified: use ratio of consecutive variance changes
        let mean_var = self.theta_vol;
        let deviations: Vec<f64> = self
            .variance_history
            .iter()
            .map(|(_, v)| v - mean_var)
            .collect();
        if deviations.len() > 1 {
            let autocov: f64 = deviations
                .iter()
                .zip(deviations.iter().skip(1))
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / (deviations.len() - 1) as f64;
            let variance: f64 =
                deviations.iter().map(|d| d.powi(2)).sum::<f64>() / deviations.len() as f64;
            if variance > 1e-12 {
                let autocorr = autocov / variance;
                // For OU process: autocorr ≈ exp(-κ × Δt)
                // Assuming Δt ≈ 1 second average between observations
                if autocorr > 0.0 && autocorr < 1.0 {
                    self.kappa_vol = -autocorr.ln().clamp(0.01, 5.0);
                }
            }
        }

        // Rho: correlation between returns and variance changes
        if self.return_history.len() == self.variance_history.len() && n > 2 {
            let returns: Vec<f64> = self.return_history.iter().map(|(_, r)| *r).collect();
            let var_deltas: Vec<f64> = self
                .variance_history
                .iter()
                .zip(self.variance_history.iter().skip(1))
                .map(|((_, v1), (_, v2))| v2 - v1)
                .collect();

            if !var_deltas.is_empty() {
                let n_pairs = var_deltas.len().min(returns.len() - 1);
                let mean_r: f64 = returns[..n_pairs].iter().sum::<f64>() / n_pairs as f64;
                let mean_dv: f64 = var_deltas[..n_pairs].iter().sum::<f64>() / n_pairs as f64;

                let cov: f64 = returns[..n_pairs]
                    .iter()
                    .zip(var_deltas[..n_pairs].iter())
                    .map(|(r, dv)| (r - mean_r) * (dv - mean_dv))
                    .sum::<f64>()
                    / n_pairs as f64;

                let std_r = (returns[..n_pairs]
                    .iter()
                    .map(|r| (r - mean_r).powi(2))
                    .sum::<f64>()
                    / n_pairs as f64)
                    .sqrt();
                let std_dv = (var_deltas[..n_pairs]
                    .iter()
                    .map(|dv| (dv - mean_dv).powi(2))
                    .sum::<f64>()
                    / n_pairs as f64)
                    .sqrt();

                if std_r > 1e-12 && std_dv > 1e-12 {
                    self.rho = (cov / (std_r * std_dv)).clamp(-0.95, 0.95);
                }
            }
        }
    }

    /// Get expected average variance over horizon using OU dynamics.
    ///
    /// E[∫₀ᵀ σ² dt]/T = θ + (σ₀² - θ)(1 - e^(-κT))/(κT)
    pub fn expected_avg_variance(&self, horizon_secs: f64) -> f64 {
        if horizon_secs < 1e-9 || self.kappa_vol < 1e-9 {
            return self.v_t;
        }

        let kt = self.kappa_vol * horizon_secs;
        let decay = 1.0 - (-kt).exp();
        self.theta_vol + (self.v_t - self.theta_vol) * decay / kt
    }

    /// Get expected average volatility (sqrt of variance).
    pub fn expected_avg_sigma(&self, horizon_secs: f64) -> f64 {
        self.expected_avg_variance(horizon_secs).sqrt()
    }

    /// Get leverage-adjusted volatility.
    ///
    /// When returns are negative and ρ < 0, volatility increases.
    pub fn leverage_adjusted_vol(&self, recent_return: f64) -> f64 {
        let base_vol = self.v_t.sqrt();

        // If return and rho have same sign, vol decreases
        // If opposite signs, vol increases (leverage effect)
        let adjustment = if recent_return * self.rho < 0.0 {
            // Return is negative, rho is negative: vol increases
            0.2 * recent_return.abs() / base_vol.max(1e-9)
        } else {
            0.0
        };

        base_vol * (1.0 + adjustment.clamp(0.0, 0.5))
    }

    // === Getters ===

    /// Current instantaneous variance.
    pub fn v_t(&self) -> f64 {
        self.v_t
    }

    /// Current instantaneous volatility.
    pub fn sigma_t(&self) -> f64 {
        self.v_t.sqrt()
    }

    /// Mean-reversion speed.
    pub fn kappa(&self) -> f64 {
        self.kappa_vol
    }

    /// Long-run variance.
    pub fn theta(&self) -> f64 {
        self.theta_vol
    }

    /// Long-run volatility.
    pub fn theta_sigma(&self) -> f64 {
        self.theta_vol.sqrt()
    }

    /// Vol-of-vol.
    pub fn xi(&self) -> f64 {
        self.xi_vol
    }

    /// Price-vol correlation (leverage effect).
    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// Check if calibrated.
    pub fn is_calibrated(&self) -> bool {
        self.variance_history.len() >= self.min_observations
    }
}

// ============================================================================
// Noise Filter (First Principles Gap 9)
// ============================================================================

/// Filters bid-ask bounce noise from price series.
///
/// Uses the Roll model: Cov(r_t, r_{t-1}) = -noise_var
/// to estimate microstructure noise and provide cleaner price signals.
#[derive(Debug)]
pub struct NoiseFilter {
    /// Estimated noise variance (from autocovariance)
    noise_variance: f64,
    /// Recent returns for autocovariance calculation
    returns: VecDeque<f64>,
    /// Maximum history size
    max_history: usize,
    /// EWMA alpha for noise estimation
    alpha: f64,
}

impl NoiseFilter {
    /// Create a new noise filter.
    pub fn new(max_history: usize, alpha: f64) -> Self {
        Self {
            noise_variance: 0.0,
            returns: VecDeque::with_capacity(max_history),
            max_history,
            alpha,
        }
    }

    /// Create with default parameters.
    pub fn default_config() -> Self {
        Self::new(100, 0.05)
    }

    /// Record a new return observation.
    pub fn on_return(&mut self, log_return: f64) {
        self.returns.push_back(log_return);

        // Trim to max size
        while self.returns.len() > self.max_history {
            self.returns.pop_front();
        }

        // Update noise variance from autocovariance
        if self.returns.len() >= 2 {
            self.update_noise_estimate();
        }
    }

    /// Update noise estimate using Roll model.
    ///
    /// Roll model: Cov(r_t, r_{t-1}) = -noise_var
    fn update_noise_estimate(&mut self) {
        if self.returns.len() < 2 {
            return;
        }

        // Calculate lag-1 autocovariance
        let n = self.returns.len();
        let returns_vec: Vec<f64> = self.returns.iter().cloned().collect();

        let mean = returns_vec.iter().sum::<f64>() / n as f64;

        let mut cov = 0.0;
        for i in 1..n {
            cov += (returns_vec[i] - mean) * (returns_vec[i - 1] - mean);
        }
        cov /= (n - 1) as f64;

        // Roll model: noise_var = -Cov(r_t, r_{t-1})
        // Only update if covariance is negative (expected from bid-ask bounce)
        if cov < 0.0 {
            let new_estimate = -cov;
            self.noise_variance =
                self.alpha * new_estimate + (1.0 - self.alpha) * self.noise_variance;
        }
    }

    /// Get estimated noise standard deviation.
    pub fn noise_sigma(&self) -> f64 {
        self.noise_variance.sqrt()
    }

    /// Get estimated noise variance.
    pub fn noise_variance(&self) -> f64 {
        self.noise_variance
    }

    /// Clean a return by removing estimated noise component.
    ///
    /// Returns a filtered return with reduced microstructure noise.
    pub fn filter_return(&self, raw_return: f64) -> f64 {
        if self.noise_variance < 1e-20 {
            return raw_return;
        }

        // Simple shrinkage toward 0 based on signal-to-noise ratio
        let total_var = raw_return.powi(2);
        let signal_var = (total_var - self.noise_variance).max(0.0);

        if total_var < 1e-20 {
            return 0.0;
        }

        let shrinkage = signal_var / total_var;
        raw_return * shrinkage.sqrt()
    }

    /// Check if filter is warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.returns.len() >= 20
    }
}

// ============================================================================
// Probabilistic Momentum Model (First Principles Gap 10)
// ============================================================================

/// Probabilistic momentum model for continuation/reversal prediction.
///
/// Replaces heuristic knife scores with Bayesian probability estimates
/// of momentum continuation.
#[derive(Debug)]
pub struct MomentumModel {
    /// Prior probability of momentum continuation
    prior_continuation: f64,
    /// Likelihood ratio observations: (timestamp_ms, momentum_bps, continued)
    observations: VecDeque<(u64, f64, bool)>,
    /// Learned continuation probability by momentum magnitude
    continuation_by_magnitude: [f64; 10], // Buckets: 0-10, 10-20, ..., 90+ bps
    /// Observation counts per bucket
    counts_by_magnitude: [usize; 10],
    /// Window for observations (ms)
    window_ms: u64,
    /// Minimum observations per bucket
    min_observations: usize,
    /// EWMA alpha for updates
    alpha: f64,
}

impl MomentumModel {
    /// Create a new momentum model.
    pub fn new(window_ms: u64, alpha: f64) -> Self {
        Self {
            prior_continuation: 0.5, // 50% prior
            observations: VecDeque::with_capacity(1000),
            continuation_by_magnitude: [0.5; 10], // Start at 50%
            counts_by_magnitude: [0; 10],
            window_ms,
            min_observations: 10,
            alpha,
        }
    }

    /// Create with default parameters.
    pub fn default_config() -> Self {
        Self::new(300_000, 0.1) // 5 minute window
    }

    /// Record an observation of momentum and whether it continued.
    ///
    /// # Arguments
    /// - `timestamp_ms`: Current timestamp
    /// - `momentum_bps`: Momentum in basis points (can be negative)
    /// - `continued`: Whether the momentum continued (same sign return)
    pub fn record_observation(&mut self, timestamp_ms: u64, momentum_bps: f64, continued: bool) {
        self.observations
            .push_back((timestamp_ms, momentum_bps, continued));

        // Update bucket statistics
        let bucket = self.magnitude_to_bucket(momentum_bps.abs());
        self.counts_by_magnitude[bucket] += 1;

        let obs = if continued { 1.0 } else { 0.0 };
        self.continuation_by_magnitude[bucket] =
            self.alpha * obs + (1.0 - self.alpha) * self.continuation_by_magnitude[bucket];

        // Expire old observations
        let cutoff = timestamp_ms.saturating_sub(self.window_ms);
        while self
            .observations
            .front()
            .map(|(t, _, _)| *t < cutoff)
            .unwrap_or(false)
        {
            self.observations.pop_front();
        }
    }

    /// Map momentum magnitude to bucket index.
    fn magnitude_to_bucket(&self, abs_momentum_bps: f64) -> usize {
        ((abs_momentum_bps / 10.0) as usize).min(9)
    }

    /// Get probability of momentum continuation.
    ///
    /// Returns P(next_return has same sign as momentum).
    pub fn continuation_probability(&self, momentum_bps: f64) -> f64 {
        let bucket = self.magnitude_to_bucket(momentum_bps.abs());

        // Use learned probability if we have enough data
        if self.counts_by_magnitude[bucket] >= self.min_observations {
            self.continuation_by_magnitude[bucket]
        } else {
            self.prior_continuation
        }
    }

    /// Get bid protection factor based on momentum.
    ///
    /// Returns multiplier > 1 if we should protect bids (falling market).
    pub fn bid_protection_factor(&self, momentum_bps: f64) -> f64 {
        if momentum_bps >= 0.0 {
            return 1.0; // Not falling, no protection needed
        }

        let p_continue = self.continuation_probability(momentum_bps);
        let magnitude_factor = (momentum_bps.abs() / 50.0).min(1.0); // Scale by magnitude

        // Protection factor: 1.0 to 2.0 based on continuation prob and magnitude
        1.0 + p_continue * magnitude_factor
    }

    /// Get ask protection factor based on momentum.
    ///
    /// Returns multiplier > 1 if we should protect asks (rising market).
    pub fn ask_protection_factor(&self, momentum_bps: f64) -> f64 {
        if momentum_bps <= 0.0 {
            return 1.0; // Not rising, no protection needed
        }

        let p_continue = self.continuation_probability(momentum_bps);
        let magnitude_factor = (momentum_bps.abs() / 50.0).min(1.0);

        1.0 + p_continue * magnitude_factor
    }

    /// Get overall momentum strength [0, 1].
    pub fn momentum_strength(&self, momentum_bps: f64) -> f64 {
        let p_continue = self.continuation_probability(momentum_bps);
        let magnitude = (momentum_bps.abs() / 100.0).min(1.0);

        p_continue * magnitude
    }

    /// Check if model is calibrated.
    pub fn is_calibrated(&self) -> bool {
        self.observations.len() >= self.min_observations * 3
    }
}

// ============================================================================
// Trade Flow Tracker (Buy/Sell Imbalance)
// ============================================================================

/// Tracks buy vs sell aggressor imbalance from trade tape.
///
/// Uses the trade side field from Hyperliquid ("B" = buy aggressor, "S" = sell aggressor)
/// to detect directional order flow before it shows up in price.
#[derive(Debug)]
struct TradeFlowTracker {
    /// (timestamp_ms, signed_volume): positive = buy aggressor
    trades: VecDeque<(u64, f64)>,
    /// Rolling window (ms)
    window_ms: u64,
    /// EWMA smoothed imbalance
    ewma_imbalance: f64,
    /// EWMA alpha
    alpha: f64,
}

impl TradeFlowTracker {
    fn new(window_ms: u64, alpha: f64) -> Self {
        Self {
            trades: VecDeque::with_capacity(500),
            window_ms,
            ewma_imbalance: 0.0,
            alpha,
        }
    }

    /// Add a trade from the tape.
    /// is_buy_aggressor: true if buyer was taker (lifted the ask)
    fn on_trade(&mut self, timestamp_ms: u64, size: f64, is_buy_aggressor: bool) {
        let signed = if is_buy_aggressor { size } else { -size };
        self.trades.push_back((timestamp_ms, signed));

        // Expire old trades
        let cutoff = timestamp_ms.saturating_sub(self.window_ms);
        while self
            .trades
            .front()
            .map(|(t, _)| *t < cutoff)
            .unwrap_or(false)
        {
            self.trades.pop_front();
        }

        // Update EWMA
        let instant = self.compute_instant_imbalance();
        self.ewma_imbalance = self.alpha * instant + (1.0 - self.alpha) * self.ewma_imbalance;
    }

    /// Compute instantaneous imbalance: (buy - sell) / total
    fn compute_instant_imbalance(&self) -> f64 {
        let (buy_vol, sell_vol) =
            self.trades.iter().fold(
                (0.0, 0.0),
                |(b, s), (_, v)| {
                    if *v > 0.0 {
                        (b + v, s)
                    } else {
                        (b, s - v)
                    }
                },
            );
        let total = buy_vol + sell_vol;
        if total < 1e-12 {
            0.0
        } else {
            (buy_vol - sell_vol) / total
        }
    }

    /// Smoothed flow imbalance [-1, 1]
    /// Negative = sell pressure, Positive = buy pressure
    fn imbalance(&self) -> f64 {
        self.ewma_imbalance.clamp(-1.0, 1.0)
    }

    /// Is there dominant selling (for bid protection)?
    #[allow(dead_code)]
    fn is_sell_pressure(&self) -> bool {
        self.ewma_imbalance < -0.25
    }

    /// Is there dominant buying (for ask protection)?
    #[allow(dead_code)]
    fn is_buy_pressure(&self) -> bool {
        self.ewma_imbalance > 0.25
    }
}

// ============================================================================
// Weighted Kappa Estimator (Improved L2 Analysis)
// ============================================================================

/// Weighted linear regression kappa estimator.
///
/// Improvements over simple kappa:
/// - Truncates to orders within max_distance of mid (ignores fake far orders)
/// - Uses first N levels only (focuses on relevant liquidity)
/// - Weights by proximity to mid (closer levels matter more)
#[derive(Debug)]
struct WeightedKappaEstimator {
    alpha: f64,
    kappa: f64,
    max_distance: f64,
    max_levels: usize,
    update_count: usize,
}

impl WeightedKappaEstimator {
    fn new(
        half_life_updates: f64,
        default_kappa: f64,
        max_distance: f64,
        max_levels: usize,
    ) -> Self {
        Self {
            alpha: (2.0_f64.ln() / half_life_updates).clamp(0.001, 1.0),
            kappa: default_kappa,
            max_distance,
            max_levels,
            update_count: 0,
        }
    }

    /// Update kappa from L2 order book.
    ///
    /// Uses instantaneous depth (size at each level) to fit exponential decay:
    /// L(δ) = A × exp(-κ × δ)  =>  ln(L) = ln(A) - κ × δ
    fn update(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)], mid: f64) {
        if mid <= 0.0 {
            return;
        }

        // Collect points: (distance, size_at_level, weight)
        let mut points: Vec<(f64, f64, f64)> = Vec::new();

        // Process bids (truncate and limit levels)
        for (i, (price, size)) in bids.iter().enumerate() {
            if i >= self.max_levels || *price <= 0.0 || *size <= 0.0 {
                break;
            }
            let distance = (mid - price) / mid;
            if distance > self.max_distance {
                break; // Too far from mid
            }
            // Weight by proximity: closer to mid = higher weight
            let weight = 1.0 / (1.0 + distance * 100.0);
            if distance > 1e-6 {
                points.push((distance, *size, weight));
            }
        }

        // Process asks (same logic)
        for (i, (price, size)) in asks.iter().enumerate() {
            if i >= self.max_levels || *price <= 0.0 || *size <= 0.0 {
                break;
            }
            let distance = (price - mid) / mid;
            if distance > self.max_distance {
                break;
            }
            let weight = 1.0 / (1.0 + distance * 100.0);
            if distance > 1e-6 {
                points.push((distance, *size, weight));
            }
        }

        // Need at least 4 points for meaningful regression
        if points.len() < 4 {
            return;
        }

        // Weighted linear regression on (distance, ln(size))
        // Model: ln(size) = ln(A) - κ × distance
        // Slope = -κ, so κ = -slope
        if let Some(slope) = weighted_linear_regression_slope(&points) {
            // In real order books, liquidity often INCREASES slightly with distance
            // (more limit orders stacked further from mid), giving positive slope.
            // Use absolute value and a reasonable default if slope is wrong sign.
            let kappa_estimated = if slope < 0.0 {
                // Negative slope = liquidity decays with distance (expected in theory)
                (-slope).clamp(1.0, 10000.0)
            } else {
                // Positive slope = liquidity increases with distance (common in practice)
                // Use a moderate default based on typical market structure
                50.0
            };

            // EWMA update
            self.kappa = self.alpha * kappa_estimated + (1.0 - self.alpha) * self.kappa;
            self.update_count += 1;

            debug!(
                points = points.len(),
                slope = %format!("{:.4}", slope),
                kappa_new = %format!("{:.2}", kappa_estimated),
                kappa_ewma = %format!("{:.2}", self.kappa),
                "Kappa updated from L2 book"
            );
        }
    }

    #[allow(dead_code)]
    fn kappa(&self) -> f64 {
        self.kappa.clamp(1.0, 10000.0)
    }

    #[allow(dead_code)]
    fn update_count(&self) -> usize {
        self.update_count
    }
}

// ============================================================================
// Fill Rate Kappa Estimator (Trade Distance Distribution)
// ============================================================================

/// Estimates κ from trade execution distance distribution with confidence tracking.
///
/// This is the CORRECT κ for the GLFT formula:
/// λ(δ) = A × exp(-κ × δ)  where λ is fill rate at distance δ
///
/// For exponential distribution: E[δ] = 1/κ, so κ = 1/E[δ]
///
/// This measures WHERE trades actually execute relative to mid,
/// NOT the shape of the order book (which is a different concept).
///
/// **Enhancements:**
/// - Variance tracking for confidence estimation
/// - Goodness-of-fit scoring for exponential assumption validation
/// - Sample size weighting in confidence calculation
#[derive(Debug)]
struct FillRateKappaEstimator {
    /// Rolling observations: (distance, volume, timestamp)
    observations: VecDeque<(f64, f64, u64)>,
    /// Rolling window (ms)
    window_ms: u64,
    /// Running volume-weighted distance sum (for efficiency)
    volume_weighted_distance: f64,
    /// Running volume-weighted squared distance sum (for variance)
    volume_weighted_distance_sq: f64,
    /// Running total volume
    total_volume: f64,
    /// EWMA smoothed kappa
    kappa: f64,
    /// EWMA alpha
    alpha: f64,
    /// Update count for warmup
    update_count: usize,
    /// Default kappa for fallback
    #[allow(dead_code)]
    default_kappa: f64,
    /// Coefficient of variation (CV) - measure of relative dispersion
    cv: f64,
    /// Goodness of fit score [0, 1] - how well data fits exponential
    fit_score: f64,
}

impl FillRateKappaEstimator {
    fn new(window_ms: u64, half_life_ticks: f64, default_kappa: f64) -> Self {
        Self {
            observations: VecDeque::with_capacity(1000),
            window_ms,
            volume_weighted_distance: 0.0,
            volume_weighted_distance_sq: 0.0,
            total_volume: 0.0,
            kappa: default_kappa,
            alpha: (2.0_f64.ln() / half_life_ticks).clamp(0.001, 0.5),
            update_count: 0,
            default_kappa,
            cv: 1.0,        // For exponential, CV = 1.0 exactly
            fit_score: 0.5, // Start neutral
        }
    }

    /// Process a trade. mid must be the mid price at time of trade.
    fn on_trade(&mut self, timestamp_ms: u64, price: f64, size: f64, mid: f64) {
        if mid <= 0.0 || size <= 0.0 || price <= 0.0 {
            return;
        }

        // Distance from mid as fraction
        let distance = ((price - mid) / mid).abs();

        // Minimum distance floor to avoid division issues
        // Trades AT the mid get a small floor distance
        let distance = distance.max(0.00001); // 0.1 bps floor

        // Add observation
        self.observations.push_back((distance, size, timestamp_ms));
        self.volume_weighted_distance += distance * size;
        self.volume_weighted_distance_sq += distance * distance * size;
        self.total_volume += size;

        // Expire old observations
        self.expire_old(timestamp_ms);

        // Update kappa estimate and fit metrics
        self.update_kappa();
        self.update_fit_metrics();
    }

    fn expire_old(&mut self, now: u64) {
        let cutoff = now.saturating_sub(self.window_ms);
        while let Some((dist, size, ts)) = self.observations.front() {
            if *ts < cutoff {
                self.volume_weighted_distance -= dist * size;
                self.volume_weighted_distance_sq -= dist * dist * size;
                self.total_volume -= size;
                self.observations.pop_front();
            } else {
                break;
            }
        }

        // Ensure running sums don't go negative due to float precision
        self.volume_weighted_distance = self.volume_weighted_distance.max(0.0);
        self.volume_weighted_distance_sq = self.volume_weighted_distance_sq.max(0.0);
        self.total_volume = self.total_volume.max(0.0);
    }

    fn update_kappa(&mut self) {
        if self.total_volume < 1e-9 {
            return;
        }

        // Volume-weighted average distance
        let avg_distance = self.volume_weighted_distance / self.total_volume;

        // For exponential distribution λ(δ) = A·exp(-κδ):
        // E[δ] = 1/κ (mean of exponential)
        // Therefore: κ = 1/E[δ]
        if avg_distance > 1e-8 {
            let kappa_instant = 1.0 / avg_distance;

            // Clamp to reasonable range
            // κ = 1000 means avg distance = 10 bps
            // κ = 5000 means avg distance = 2 bps
            // κ = 500 means avg distance = 20 bps
            let kappa_clamped = kappa_instant.clamp(100.0, 10000.0);

            // EWMA update
            self.kappa = self.alpha * kappa_clamped + (1.0 - self.alpha) * self.kappa;
            self.update_count += 1;

            if self.update_count.is_multiple_of(100) {
                debug!(
                    observations = self.observations.len(),
                    avg_distance_bps = %format!("{:.2}", avg_distance * 10000.0),
                    kappa_instant = %format!("{:.0}", kappa_instant),
                    kappa_ewma = %format!("{:.0}", self.kappa),
                    "Fill-rate kappa updated from trade distances"
                );
            }
        }
    }

    /// Update fit metrics (coefficient of variation and goodness of fit).
    ///
    /// For exponential distribution, CV = σ/μ = 1.0 exactly.
    /// Deviation from CV = 1 indicates non-exponential behavior.
    fn update_fit_metrics(&mut self) {
        if self.total_volume < 1e-9 || self.observations.len() < 10 {
            return;
        }

        // Calculate mean and variance
        let mean = self.volume_weighted_distance / self.total_volume;
        let mean_sq = self.volume_weighted_distance_sq / self.total_volume;
        let variance = (mean_sq - mean * mean).max(0.0);
        let std_dev = variance.sqrt();

        // Coefficient of variation: CV = σ/μ
        // For exponential: CV = 1.0
        // For power-law: CV > 1.0 (heavy tail)
        // For light-tail: CV < 1.0
        if mean > 1e-9 {
            self.cv = std_dev / mean;

            // Fit score: how close is CV to 1.0?
            // Score = 1 when CV = 1, decreases as CV deviates
            // Using: score = exp(-|CV - 1|²)
            let cv_deviation = (self.cv - 1.0).abs();
            self.fit_score = (-cv_deviation * cv_deviation * 4.0).exp();
        }
    }

    fn kappa(&self) -> f64 {
        self.kappa.clamp(100.0, 10000.0)
    }

    /// Get kappa with confidence-based blending with fallback.
    ///
    /// When confidence is low (poor exponential fit), blend towards fallback.
    fn kappa_with_fallback(&self, fallback_kappa: f64) -> f64 {
        let confidence = self.confidence();
        let primary = self.kappa.clamp(100.0, 10000.0);
        let blended = confidence * primary + (1.0 - confidence) * fallback_kappa;
        blended.clamp(100.0, 10000.0)
    }

    /// Confidence score [0, 1] based on sample size and fit quality.
    ///
    /// High confidence when:
    /// - Many observations (sample size factor)
    /// - Good exponential fit (fit_score)
    /// - CV close to 1.0
    fn confidence(&self) -> f64 {
        // Sample size factor: ramps up to 1.0 at 100+ observations
        let sample_factor = (self.observations.len() as f64 / 100.0).min(1.0);

        // Combine factors
        (sample_factor * self.fit_score).clamp(0.0, 1.0)
    }

    /// Get coefficient of variation (CV).
    /// For exponential: CV ≈ 1.0. Higher = heavy tail, Lower = light tail.
    fn cv(&self) -> f64 {
        self.cv
    }

    /// Get fit score [0, 1]. Higher = better exponential fit.
    #[allow(dead_code)]
    fn fit_score(&self) -> f64 {
        self.fit_score
    }

    fn update_count(&self) -> usize {
        self.update_count
    }
}

/// Weighted linear regression to get slope.
/// Points are (x, y, weight). Fits ln(y) ~ a + b*x, returns b.
fn weighted_linear_regression_slope(points: &[(f64, f64, f64)]) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }

    let mut sum_w = 0.0;
    let mut sum_wx = 0.0;
    let mut sum_wy = 0.0;
    let mut sum_wxx = 0.0;
    let mut sum_wxy = 0.0;

    for (x, y, w) in points {
        if *y <= 0.0 {
            continue;
        }
        let ln_y = y.ln();
        sum_w += w;
        sum_wx += w * x;
        sum_wy += w * ln_y;
        sum_wxx += w * x * x;
        sum_wxy += w * x * ln_y;
    }

    let denominator = sum_w * sum_wxx - sum_wx * sum_wx;
    if denominator.abs() < 1e-12 {
        return None;
    }

    Some((sum_w * sum_wxy - sum_wx * sum_wy) / denominator)
}

// ============================================================================
// Book Structure Estimator (L2 Order Book Analysis)
// ============================================================================

/// Analyzes L2 order book structure for auxiliary quote adjustments.
///
/// Provides two key signals:
/// 1. **Book Imbalance** [-1, 1]: Bid/ask depth asymmetry
///    - Positive = more bids than asks (buying pressure)
///    - Used for directional skew adjustment
///
/// 2. **Liquidity Gamma Multiplier** [1.0, 2.0]: Thin book detection
///    - Scales γ up when near-touch liquidity is below average
///    - Protects against adverse selection in thin markets
#[derive(Debug)]
struct BookStructureEstimator {
    /// EWMA smoothed book imbalance [-1, 1]
    imbalance: f64,
    /// Current near-touch depth (within 10 bps of mid)
    near_touch_depth: f64,
    /// Rolling reference depth for comparison
    reference_depth: f64,
    /// EWMA smoothing factor
    alpha: f64,
    /// Number of levels to consider for imbalance
    imbalance_levels: usize,
    /// Maximum distance for near-touch liquidity (as fraction)
    near_touch_distance: f64,
}

impl BookStructureEstimator {
    fn new() -> Self {
        Self {
            imbalance: 0.0,
            near_touch_depth: 0.0,
            reference_depth: 1.0, // Start with 1.0 to avoid division issues
            alpha: 0.1,
            imbalance_levels: 5,
            near_touch_distance: 0.001, // 10 bps
        }
    }

    /// Update with new L2 book data.
    fn update(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)], mid: f64) {
        if mid <= 0.0 {
            return;
        }

        // 1. Calculate bid/ask imbalance from top N levels
        let bid_depth: f64 = bids
            .iter()
            .take(self.imbalance_levels)
            .map(|(_, sz)| sz)
            .sum();
        let ask_depth: f64 = asks
            .iter()
            .take(self.imbalance_levels)
            .map(|(_, sz)| sz)
            .sum();

        let total = bid_depth + ask_depth;
        if total > 1e-9 {
            let instant_imbalance = (bid_depth - ask_depth) / total;
            self.imbalance = self.alpha * instant_imbalance + (1.0 - self.alpha) * self.imbalance;
        }

        // 2. Calculate near-touch liquidity (depth within 10 bps of mid)
        let bid_near: f64 = bids
            .iter()
            .take_while(|(px, _)| (mid - px) / mid <= self.near_touch_distance)
            .map(|(_, sz)| sz)
            .sum();
        let ask_near: f64 = asks
            .iter()
            .take_while(|(px, _)| (px - mid) / mid <= self.near_touch_distance)
            .map(|(_, sz)| sz)
            .sum();
        self.near_touch_depth = bid_near + ask_near;

        // 3. Update reference depth (slow-moving average)
        // Use very slow decay to establish "normal" liquidity baseline
        self.reference_depth =
            0.99 * self.reference_depth + 0.01 * self.near_touch_depth.max(0.001);
    }

    /// Get current book imbalance [-1, 1].
    /// Positive = more bids (buying pressure), Negative = more asks (selling pressure).
    fn imbalance(&self) -> f64 {
        self.imbalance.clamp(-1.0, 1.0)
    }

    /// Get gamma multiplier for thin book conditions [1.0, 2.0].
    /// Returns > 1.0 when near-touch liquidity is below reference.
    fn gamma_multiplier(&self) -> f64 {
        if self.near_touch_depth >= self.reference_depth {
            1.0
        } else {
            // Thin book → scale gamma up (wider spreads for protection)
            // sqrt scaling: 4x thinner → 2x gamma
            let ratio = self.reference_depth / self.near_touch_depth.max(0.001);
            ratio.sqrt().clamp(1.0, 2.0)
        }
    }
}

// ============================================================================
// Microprice Estimator (Data-Driven Fair Price)
// ============================================================================

/// Observation for microprice regression.
/// Stores signals at time t, matched with realized return at t + horizon.
#[derive(Debug, Clone)]
struct MicropriceObservation {
    timestamp_ms: u64,
    book_imbalance: f64,
    flow_imbalance: f64,
    mid: f64,
}

/// Estimates microprice by learning how book/flow imbalance predict returns.
///
/// Uses rolling online regression to estimate:
/// E[r_{t+Δ}] = β_book × book_imbalance + β_flow × flow_imbalance
///
/// microprice = mid × (1 + β_book × book_imb + β_flow × flow_imb)
///
/// This replaces magic number adjustments with data-driven coefficients.
#[derive(Debug)]
/// Microprice estimator with Ridge regularization.
///
/// Uses online OLS with L2 regularization to learn coefficients for:
/// microprice = mid × (1 + β_book × book_imb + β_flow × flow_imb)
///
/// **Regularization benefits:**
/// - Prevents coefficient explosion when signals are collinear
/// - Biases coefficients toward zero when data is noisy
/// - Produces more stable estimates across market regimes
struct MicropriceEstimator {
    /// Pending observations waiting for forward horizon to elapse
    pending: VecDeque<MicropriceObservation>,
    /// Window for regression data (ms)
    window_ms: u64,
    /// Forward horizon to measure realized return (ms)
    forward_horizon_ms: u64,

    // Running sums for online regression (2-variable linear regression)
    // y = β_book × x_book + β_flow × x_flow + ε
    n: usize,
    sum_x_book: f64,
    sum_x_flow: f64,
    sum_y: f64,
    sum_xx_book: f64,
    sum_xx_flow: f64,
    sum_x_cross: f64, // book * flow
    sum_xy_book: f64,
    sum_xy_flow: f64,
    sum_yy: f64,

    // Estimated coefficients
    beta_book: f64,
    beta_flow: f64,
    r_squared: f64,

    // Warmup threshold
    min_observations: usize,

    /// Ridge regularization parameter (λ)
    /// Higher = more regularization, coefficients biased toward zero
    lambda: f64,

    /// Minimum R² threshold - below this, revert to mid price
    min_r_squared: f64,

    /// Correlation between book and flow signals (for multicollinearity detection)
    signal_correlation: f64,
}

impl MicropriceEstimator {
    fn new(window_ms: u64, forward_horizon_ms: u64, min_observations: usize) -> Self {
        Self {
            pending: VecDeque::with_capacity(2000),
            window_ms,
            forward_horizon_ms,
            n: 0,
            sum_x_book: 0.0,
            sum_x_flow: 0.0,
            sum_y: 0.0,
            sum_xx_book: 0.0,
            sum_xx_flow: 0.0,
            sum_x_cross: 0.0,
            sum_xy_book: 0.0,
            sum_xy_flow: 0.0,
            sum_yy: 0.0,
            beta_book: 0.0,
            beta_flow: 0.0,
            r_squared: 0.0,
            min_observations,
            // Ridge regularization: λ = 0.01 provides moderate regularization
            // This adds λI to X'X, shrinking coefficients toward zero
            lambda: 0.01,
            // Minimum R² threshold: below 1% explained variance, revert to mid
            min_r_squared: 0.01,
            // Initialize correlation to 0 (no correlation assumed)
            signal_correlation: 0.0,
        }
    }

    /// Update with new book state.
    fn on_book_update(
        &mut self,
        timestamp_ms: u64,
        mid: f64,
        book_imbalance: f64,
        flow_imbalance: f64,
    ) {
        // 1. Process pending observations that have reached forward horizon
        self.process_completed(timestamp_ms, mid);

        // 2. Add new observation
        self.pending.push_back(MicropriceObservation {
            timestamp_ms,
            book_imbalance,
            flow_imbalance,
            mid,
        });

        // 3. Expire old data outside regression window
        self.expire_old(timestamp_ms);

        // 4. Update regression coefficients
        self.update_betas();
    }

    /// Match completed observations with their realized returns.
    fn process_completed(&mut self, now: u64, current_mid: f64) {
        // Find observations where forward_horizon has elapsed
        while let Some(obs) = self.pending.front() {
            if now >= obs.timestamp_ms + self.forward_horizon_ms {
                let obs = self.pending.pop_front().unwrap();

                // Calculate realized return
                if obs.mid > 0.0 {
                    let realized_return = (current_mid - obs.mid) / obs.mid;

                    // Add to regression sums
                    self.add_observation(obs.book_imbalance, obs.flow_imbalance, realized_return);
                }
            } else {
                break; // Remaining observations haven't reached horizon yet
            }
        }
    }

    /// Add a completed observation to regression.
    fn add_observation(&mut self, x_book: f64, x_flow: f64, y: f64) {
        let was_warmed_up = self.is_warmed_up();
        self.n += 1;
        self.sum_x_book += x_book;
        self.sum_x_flow += x_flow;
        self.sum_y += y;
        self.sum_xx_book += x_book * x_book;
        self.sum_xx_flow += x_flow * x_flow;
        self.sum_x_cross += x_book * x_flow;
        self.sum_xy_book += x_book * y;
        self.sum_xy_flow += x_flow * y;
        self.sum_yy += y * y;

        // Log when microprice warmup completes
        if !was_warmed_up && self.is_warmed_up() {
            debug!(
                n = self.n,
                min_observations = self.min_observations,
                "Microprice estimator warmup complete"
            );
        }
    }

    /// Expire observations outside the regression window.
    /// Note: We use a simple approach - reset if oldest data is too old.
    /// A more sophisticated approach would subtract old observations.
    fn expire_old(&mut self, now: u64) {
        // Simple windowing: if we have enough data and oldest is too old, decay
        // This is approximate but avoids complexity of exact windowing
        if self.n > self.min_observations * 2 {
            // Apply decay to running sums (approximate window)
            let decay = 0.999; // Slow decay
            self.sum_x_book *= decay;
            self.sum_x_flow *= decay;
            self.sum_y *= decay;
            self.sum_xx_book *= decay;
            self.sum_xx_flow *= decay;
            self.sum_x_cross *= decay;
            self.sum_xy_book *= decay;
            self.sum_xy_flow *= decay;
            self.sum_yy *= decay;
            // Effective n decays too
            self.n = ((self.n as f64) * decay) as usize;
        }

        // Also trim pending queue if it gets too large
        let max_pending = (self.window_ms / 100) as usize; // ~10 obs per second
        while self.pending.len() > max_pending {
            self.pending.pop_front();
        }

        let _ = now; // Used for logging if needed
    }

    /// Solve 2-variable linear regression for β_book and β_flow with Ridge regularization.
    ///
    /// Uses Ridge regression: β = (X'X + λI)⁻¹ X'y
    /// This shrinks coefficients toward zero, preventing explosion with collinear signals.
    fn update_betas(&mut self) {
        if self.n < self.min_observations {
            return;
        }

        let n = self.n as f64;

        // Solve normal equations for: y = β_book × x_book + β_flow × x_flow
        // Using Ridge regression: (X'X + λI)β = X'y
        //
        // X'X + λI = | Σx_book² + λ, Σx_book×x_flow |
        //           | Σx_book×x_flow, Σx_flow² + λ |
        //
        // X'y = | Σx_book×y |
        //       | Σx_flow×y |

        // Center the data (remove means)
        let mean_x_book = self.sum_x_book / n;
        let mean_x_flow = self.sum_x_flow / n;
        let mean_y = self.sum_y / n;

        // Centered sums of squares
        let sxx_book = self.sum_xx_book - n * mean_x_book * mean_x_book;
        let sxx_flow = self.sum_xx_flow - n * mean_x_flow * mean_x_flow;
        let sxy_book = self.sum_xy_book - n * mean_x_book * mean_y;
        let sxy_flow = self.sum_xy_flow - n * mean_x_flow * mean_y;
        let sx_cross = self.sum_x_cross - n * mean_x_book * mean_x_flow;
        let syy = self.sum_yy - n * mean_y * mean_y;

        // Calculate signal correlation (for multicollinearity detection)
        let std_book = sxx_book.sqrt();
        let std_flow = sxx_flow.sqrt();
        if std_book > 1e-9 && std_flow > 1e-9 {
            self.signal_correlation = (sx_cross / (std_book * std_flow)).clamp(-1.0, 1.0);
        }

        // Ridge regularization: add λ to diagonal of X'X
        // Scale λ by the average variance to make it scale-invariant
        let avg_var = (sxx_book + sxx_flow) / 2.0;
        let lambda_scaled = self.lambda * avg_var.max(1e-6);

        // Regularized diagonal elements
        let sxx_book_reg = sxx_book + lambda_scaled;
        let sxx_flow_reg = sxx_flow + lambda_scaled;

        // Determinant of (X'X + λI)
        let det = sxx_book_reg * sxx_flow_reg - sx_cross * sx_cross;

        if det.abs() < 1e-12 {
            // Still singular after regularization - very unusual
            warn!(
                correlation = %format!("{:.3}", self.signal_correlation),
                "Microprice regression singular even with regularization"
            );
            return;
        }

        // Solve using Cramer's rule with regularized matrix
        self.beta_book = (sxy_book * sxx_flow_reg - sxy_flow * sx_cross) / det;
        self.beta_flow = (sxy_flow * sxx_book_reg - sxy_book * sx_cross) / det;

        // Clamp betas to reasonable range (prevent explosion)
        // Expected range: ~0.0001-0.01 (0.1-10 bps per unit signal)
        self.beta_book = self.beta_book.clamp(-0.05, 0.05);
        self.beta_flow = self.beta_flow.clamp(-0.05, 0.05);

        // Calculate R² = 1 - SSE/SST
        if syy > 1e-12 {
            let y_pred_var = self.beta_book * self.beta_book * sxx_book
                + self.beta_flow * self.beta_flow * sxx_flow
                + 2.0 * self.beta_book * self.beta_flow * sx_cross;
            self.r_squared = (y_pred_var / syy).clamp(0.0, 1.0);
        }

        // Warn about high multicollinearity
        if self.signal_correlation.abs() > 0.8 && self.n.is_multiple_of(1000) {
            warn!(
                correlation = %format!("{:.3}", self.signal_correlation),
                "High multicollinearity between book and flow signals"
            );
        }

        // Log periodically (every 100 observations for better visibility)
        if self.n.is_multiple_of(100) {
            debug!(
                n = self.n,
                beta_book_bps = %format!("{:.2}", self.beta_book * 10000.0),
                beta_flow_bps = %format!("{:.2}", self.beta_flow * 10000.0),
                r_squared = %format!("{:.4}", self.r_squared),
                correlation = %format!("{:.3}", self.signal_correlation),
                "Microprice coefficients updated"
            );
        }
    }

    /// Get microprice adjusted for current signals.
    ///
    /// Returns mid price if:
    /// - Not warmed up (insufficient data)
    /// - R² below threshold (model has no predictive power)
    /// - Signals are highly collinear (unreliable coefficients)
    fn microprice(&self, mid: f64, book_imbalance: f64, flow_imbalance: f64) -> f64 {
        if !self.is_warmed_up() {
            return mid;
        }

        // If R² is too low, the model has no predictive power - use mid
        if self.r_squared < self.min_r_squared {
            return mid;
        }

        // If signals are extremely collinear (|ρ| > 0.95), coefficients are unreliable
        if self.signal_correlation.abs() > 0.95 {
            return mid;
        }

        // microprice = mid × (1 + β_book × book_imb + β_flow × flow_imb)
        let adjustment = self.beta_book * book_imbalance + self.beta_flow * flow_imbalance;

        // Clamp adjustment to ±50 bps for safety
        let adjustment_clamped = adjustment.clamp(-0.005, 0.005);

        mid * (1.0 + adjustment_clamped)
    }

    fn is_warmed_up(&self) -> bool {
        self.n >= self.min_observations
    }

    fn beta_book(&self) -> f64 {
        self.beta_book
    }

    fn beta_flow(&self) -> f64 {
        self.beta_flow
    }

    fn r_squared(&self) -> f64 {
        self.r_squared
    }
}

// ============================================================================
// Volume Tick Arrival Estimator
// ============================================================================

/// Arrival intensity estimator based on volume clock ticks.
/// Measures volume ticks per second (not raw trades per second).
#[derive(Debug)]
struct VolumeTickArrivalEstimator {
    alpha: f64,
    intensity: f64, // Volume ticks per second
    last_tick_ms: Option<u64>,
    tick_count: usize,
}

impl VolumeTickArrivalEstimator {
    fn new(half_life_ticks: f64, default_intensity: f64) -> Self {
        Self {
            alpha: (2.0_f64.ln() / half_life_ticks).clamp(0.001, 1.0),
            intensity: default_intensity,
            last_tick_ms: None,
            tick_count: 0,
        }
    }

    fn on_bucket(&mut self, bucket: &VolumeBucket) {
        if let Some(last_ms) = self.last_tick_ms {
            let interval_secs = (bucket.end_time_ms.saturating_sub(last_ms)) as f64 / 1000.0;
            if interval_secs > 0.001 {
                let rate = 1.0 / interval_secs;
                let rate_clamped = rate.clamp(0.001, 100.0);
                self.intensity = self.alpha * rate_clamped + (1.0 - self.alpha) * self.intensity;
                self.tick_count += 1;
            }
        }
        self.last_tick_ms = Some(bucket.end_time_ms);
    }

    fn ticks_per_second(&self) -> f64 {
        self.intensity.clamp(0.001, 100.0)
    }
}

// ============================================================================
// 4-State Volatility Regime with Hysteresis
// ============================================================================

/// Volatility regime classification.
///
/// Four states with hysteresis to prevent rapid switching:
/// - Low: Very quiet market (σ < 0.5 × baseline)
/// - Normal: Standard market conditions
/// - High: Elevated volatility (σ > 1.5 × baseline)
/// - Extreme: Crisis/toxic conditions (σ > 3 × baseline OR high jump ratio)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VolatilityRegime {
    /// Very quiet market - can tighten spreads
    Low,
    /// Normal market conditions
    Normal,
    /// Elevated volatility - widen spreads
    High,
    /// Crisis/toxic - maximum caution, consider pulling quotes
    Extreme,
}

impl Default for VolatilityRegime {
    fn default() -> Self {
        Self::Normal
    }
}

impl VolatilityRegime {
    /// Get spread multiplier for this regime.
    ///
    /// Used to scale spreads based on volatility state.
    pub fn spread_multiplier(&self) -> f64 {
        match self {
            Self::Low => 0.8,     // Tighter spreads in quiet markets
            Self::Normal => 1.0,  // Base case
            Self::High => 1.5,    // Wider spreads for elevated vol
            Self::Extreme => 2.5, // Much wider spreads in crisis
        }
    }

    /// Get gamma multiplier for this regime.
    ///
    /// Risk aversion increases with volatility.
    pub fn gamma_multiplier(&self) -> f64 {
        match self {
            Self::Low => 0.8,
            Self::Normal => 1.0,
            Self::High => 1.5,
            Self::Extreme => 3.0,
        }
    }

    /// Check if quotes should be pulled (extreme regime).
    pub fn should_consider_pulling_quotes(&self) -> bool {
        matches!(self, Self::Extreme)
    }
}

/// Tracks volatility regime with hysteresis to prevent rapid switching.
///
/// Transitions between states require sustained conditions to trigger,
/// preventing oscillation at boundaries.
#[derive(Debug)]
struct VolatilityRegimeTracker {
    /// Current regime state
    regime: VolatilityRegime,
    /// Baseline volatility (for regime thresholds)
    baseline_sigma: f64,
    /// Consecutive updates in potential new regime (for hysteresis)
    transition_count: u32,
    /// Minimum transitions before state change (hysteresis parameter)
    min_transitions: u32,
    /// Thresholds relative to baseline
    low_threshold: f64, // σ < baseline × low_threshold → Low
    high_threshold: f64,    // σ > baseline × high_threshold → High
    extreme_threshold: f64, // σ > baseline × extreme_threshold → Extreme
    /// Jump ratio threshold for Extreme regime
    jump_threshold: f64,
    /// Pending regime (for hysteresis tracking)
    pending_regime: Option<VolatilityRegime>,
}

impl VolatilityRegimeTracker {
    fn new(baseline_sigma: f64) -> Self {
        Self {
            regime: VolatilityRegime::Normal,
            baseline_sigma,
            transition_count: 0,
            min_transitions: 5, // Require 5 consecutive updates before transition
            low_threshold: 0.5,
            high_threshold: 1.5,
            extreme_threshold: 3.0,
            jump_threshold: 3.0,
            pending_regime: None,
        }
    }

    /// Update regime based on current volatility and jump ratio.
    fn update(&mut self, sigma: f64, jump_ratio: f64) {
        // Determine target regime based on current conditions
        let target = self.classify(sigma, jump_ratio);

        // Check if target matches pending transition
        if let Some(pending) = self.pending_regime {
            if pending == target {
                self.transition_count += 1;
                if self.transition_count >= self.min_transitions {
                    // Transition confirmed
                    if self.regime != target {
                        debug!(
                            from = ?self.regime,
                            to = ?target,
                            sigma = %format!("{:.6}", sigma),
                            jump_ratio = %format!("{:.2}", jump_ratio),
                            "Volatility regime transition"
                        );
                    }
                    self.regime = target;
                    self.pending_regime = None;
                    self.transition_count = 0;
                }
            } else {
                // Target changed, reset hysteresis
                self.pending_regime = Some(target);
                self.transition_count = 1;
            }
        } else if target != self.regime {
            // Start new pending transition
            self.pending_regime = Some(target);
            self.transition_count = 1;
        }
    }

    /// Classify conditions into target regime.
    fn classify(&self, sigma: f64, jump_ratio: f64) -> VolatilityRegime {
        // Jump ratio overrides to Extreme
        if jump_ratio > self.jump_threshold {
            return VolatilityRegime::Extreme;
        }

        // Volatility-based classification
        let sigma_ratio = sigma / self.baseline_sigma.max(1e-9);

        if sigma_ratio < self.low_threshold {
            VolatilityRegime::Low
        } else if sigma_ratio > self.extreme_threshold {
            VolatilityRegime::Extreme
        } else if sigma_ratio > self.high_threshold {
            VolatilityRegime::High
        } else {
            VolatilityRegime::Normal
        }
    }

    /// Get current regime.
    fn regime(&self) -> VolatilityRegime {
        self.regime
    }

    /// Update baseline volatility (e.g., from long-term EWMA).
    fn update_baseline(&mut self, new_baseline: f64) {
        if new_baseline > 1e-9 {
            // Slow update to baseline (EWMA with long half-life)
            self.baseline_sigma = 0.99 * self.baseline_sigma + 0.01 * new_baseline;
        }
    }
}

// ============================================================================
// Main Orchestrator: ParameterEstimator
// ============================================================================

/// Econometric parameter estimator with multi-timescale variance, momentum detection,
/// and trade flow tracking for robust adverse selection protection.
///
/// Pipeline:
/// 1. Raw trades → Volume Clock → Normalized volume buckets with VWAP
/// 2. VWAP returns → Multi-Scale Bipower → sigma_clean, sigma_total, sigma_effective
/// 3. VWAP returns → Momentum Detector → falling/rising knife scores
/// 4. Trade tape → Flow Tracker → buy/sell imbalance
/// 5. Trade tape → Fill Rate Kappa → trade distance distribution (CORRECT κ for GLFT)
/// 6. L2 Book → Book analysis for auxiliary adjustments (imbalance, etc.)
/// 7. Regime detection: fast jump_ratio > threshold = toxic
#[derive(Debug)]
pub struct ParameterEstimator {
    config: EstimatorConfig,
    /// Volume bucket accumulator (volume clock)
    bucket_accumulator: VolumeBucketAccumulator,
    /// Multi-timescale bipower estimator (replaces single-scale)
    multi_scale: MultiScaleBipowerEstimator,
    /// Momentum detector for falling/rising knife patterns
    momentum: MomentumDetector,
    /// Trade flow tracker for buy/sell imbalance
    flow: TradeFlowTracker,
    /// Fill-rate kappa estimator from trade distances (PRIMARY κ for GLFT)
    fill_rate_kappa: FillRateKappaEstimator,
    /// Weighted kappa estimator from L2 book (fallback when fill-rate has low confidence)
    book_kappa: WeightedKappaEstimator,
    /// Volume tick arrival estimator
    arrival: VolumeTickArrivalEstimator,
    /// Book structure estimator (imbalance, near-touch liquidity)
    book_structure: BookStructureEstimator,
    /// Microprice estimator (data-driven fair price)
    microprice_estimator: MicropriceEstimator,
    /// Current mid price
    current_mid: f64,
    /// Current timestamp for momentum queries
    current_time_ms: u64,
    /// 4-state volatility regime tracker with hysteresis
    volatility_regime: VolatilityRegimeTracker,
    /// Jump process estimator (λ, μ_j, σ_j)
    jump_estimator: JumpEstimator,
    /// Stochastic volatility parameters (κ_vol, θ_vol, ξ, ρ)
    stoch_vol: StochasticVolParams,
}

impl ParameterEstimator {
    /// Create a new parameter estimator with the given config.
    pub fn new(config: EstimatorConfig) -> Self {
        let bucket_accumulator = VolumeBucketAccumulator::new(&config);
        let multi_scale = MultiScaleBipowerEstimator::new(&config);
        let momentum = MomentumDetector::new(config.momentum_window_ms);
        let flow = TradeFlowTracker::new(config.trade_flow_window_ms, config.trade_flow_alpha);

        // Fill-rate kappa from trade distances (PRIMARY for GLFT)
        // Use 5-minute window, medium half-life, default to 2000 (typical for liquid markets)
        let fill_rate_kappa = FillRateKappaEstimator::new(
            300_000, // 5 minute window
            config.medium_half_life_ticks,
            2000.0, // Default: avg trade distance = 5 bps
        );

        // Book kappa (kept for auxiliary use, may remove later)
        let book_kappa = WeightedKappaEstimator::new(
            config.kappa_half_life_updates,
            config.default_kappa,
            config.kappa_max_distance,
            config.kappa_max_levels,
        );

        let arrival = VolumeTickArrivalEstimator::new(
            config.medium_half_life_ticks, // Use medium timescale
            config.default_arrival_intensity,
        );

        let book_structure = BookStructureEstimator::new();

        // Microprice estimator: 60s window, 300ms forward horizon, 50 min observations
        let microprice_estimator = MicropriceEstimator::new(60_000, 300, 50);

        // Volatility regime tracker with baseline from config
        let volatility_regime = VolatilityRegimeTracker::new(config.default_sigma);

        // Jump process estimator (First Principles Gap 1)
        let jump_estimator = JumpEstimator::new();

        // Stochastic volatility params (First Principles Gap 2)
        let stoch_vol = StochasticVolParams::new(config.default_sigma);

        Self {
            config,
            bucket_accumulator,
            multi_scale,
            momentum,
            flow,
            fill_rate_kappa,
            book_kappa,
            arrival,
            book_structure,
            microprice_estimator,
            current_mid: 0.0,
            current_time_ms: 0,
            volatility_regime,
            jump_estimator,
            stoch_vol,
        }
    }

    /// Update current mid price.
    pub fn on_mid_update(&mut self, mid_price: f64) {
        self.current_mid = mid_price;
    }

    /// Process a new trade (feeds into volume clock, flow tracker, AND fill-rate kappa).
    ///
    /// # Arguments
    /// * `timestamp_ms` - Trade timestamp
    /// * `price` - Trade price
    /// * `size` - Trade size
    /// * `is_buy_aggressor` - Whether buyer was the taker (if available from exchange)
    pub fn on_trade(
        &mut self,
        timestamp_ms: u64,
        price: f64,
        size: f64,
        is_buy_aggressor: Option<bool>,
    ) {
        self.current_time_ms = timestamp_ms;

        // Track trade flow if we know aggressor side
        if let Some(is_buy) = is_buy_aggressor {
            self.flow.on_trade(timestamp_ms, size, is_buy);
        }

        // Feed into fill-rate kappa estimator (trade distance from mid)
        // This is the PRIMARY κ source for GLFT
        if self.current_mid > 0.0 {
            self.fill_rate_kappa
                .on_trade(timestamp_ms, price, size, self.current_mid);
        }

        // Feed into volume bucket accumulator
        if let Some(bucket) = self.bucket_accumulator.on_trade(timestamp_ms, price, size) {
            // Get log return BEFORE updating multi_scale (it will update last_vwap)
            let log_return = self.multi_scale.last_log_return(&bucket);

            // Bucket completed - update estimators
            self.multi_scale.on_bucket(&bucket);
            self.arrival.on_bucket(&bucket);

            // Update momentum detector with signed return
            if let Some(ret) = log_return {
                self.momentum.on_bucket(bucket.end_time_ms, ret);

                // Update jump estimator with return (detects jumps > 3σ)
                let sigma_clean = self.multi_scale.sigma_clean();
                self.jump_estimator
                    .on_return(bucket.end_time_ms, ret, sigma_clean);

                // Update stochastic vol with current variance observation
                let variance = self.multi_scale.sigma_total().powi(2);
                self.stoch_vol
                    .on_variance(bucket.end_time_ms, variance, ret);
            }

            // Update volatility regime with current sigma and jump ratio
            let sigma = self.multi_scale.sigma_clean();
            let jump_ratio = self.multi_scale.jump_ratio_fast();
            self.volatility_regime.update(sigma, jump_ratio);

            // Slowly update baseline from slow sigma (long-term anchor)
            // Using slow sigma as the stable reference for regime thresholds
            self.volatility_regime
                .update_baseline(self.multi_scale.sigma_clean());

            debug!(
                vwap = %format!("{:.4}", bucket.vwap),
                volume = %format!("{:.4}", bucket.volume),
                duration_ms = bucket.end_time_ms.saturating_sub(bucket.start_time_ms),
                tick = self.multi_scale.tick_count(),
                sigma_clean = %format!("{:.6}", self.multi_scale.sigma_clean()),
                sigma_total = %format!("{:.6}", self.multi_scale.sigma_total()),
                jump_ratio = %format!("{:.2}", self.multi_scale.jump_ratio_fast()),
                kappa = %format!("{:.0}", self.fill_rate_kappa.kappa()),
                regime = ?self.volatility_regime.regime(),
                "Volume bucket completed"
            );
        }
    }

    /// Legacy on_trade without aggressor info (backward compatibility).
    pub fn on_trade_legacy(&mut self, timestamp_ms: u64, price: f64, size: f64) {
        self.on_trade(timestamp_ms, price, size, None);
    }

    /// Process L2 order book update for kappa estimation.
    /// bids and asks are slices of (price, size) tuples, best first.
    pub fn on_l2_book(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)], mid: f64) {
        self.current_mid = mid;
        self.book_kappa.update(bids, asks, mid);
        self.book_structure.update(bids, asks, mid);

        // Feed microprice estimator with current signals
        self.microprice_estimator.on_book_update(
            self.current_time_ms,
            mid,
            self.book_structure.imbalance(),
            self.flow.imbalance(),
        );
    }

    // === Volatility Accessors ===

    /// Get clean volatility (σ_clean) - per-second, NOT annualized.
    /// Based on Bipower Variation, robust to jumps.
    /// Use for base spread pricing (continuous risk).
    pub fn sigma(&self) -> f64 {
        self.multi_scale.sigma_clean()
    }

    /// Get clean volatility - alias for sigma()
    pub fn sigma_clean(&self) -> f64 {
        self.multi_scale.sigma_clean()
    }

    /// Get total volatility (σ_total) - includes jumps.
    /// Based on Realized Variance, captures full price risk.
    pub fn sigma_total(&self) -> f64 {
        self.multi_scale.sigma_total()
    }

    /// Get effective volatility (σ_effective) - blended.
    /// Blends clean and total based on jump regime.
    /// Use for inventory skew (reacts appropriately to jumps).
    pub fn sigma_effective(&self) -> f64 {
        self.multi_scale.sigma_effective()
    }

    // === Order Book Accessors ===

    /// Get current kappa estimate (κ) with confidence-based blending.
    ///
    /// Uses fill-rate kappa from trade execution distances as primary estimate.
    /// When confidence is low (poor exponential fit), blends toward book-based kappa.
    pub fn kappa(&self) -> f64 {
        // Use book kappa as fallback when fill-rate confidence is low
        self.fill_rate_kappa
            .kappa_with_fallback(self.book_kappa.kappa())
    }

    /// Get raw fill-rate kappa (without blending).
    pub fn kappa_fill_rate(&self) -> f64 {
        self.fill_rate_kappa.kappa()
    }

    /// Get book-based kappa from L2 order book depth decay.
    pub fn kappa_book(&self) -> f64 {
        self.book_kappa.kappa()
    }

    /// Get kappa estimation confidence [0, 1].
    ///
    /// Based on sample size and how well data fits exponential distribution.
    /// Low confidence indicates book-based fallback is being used more.
    pub fn kappa_confidence(&self) -> f64 {
        self.fill_rate_kappa.confidence()
    }

    /// Get coefficient of variation for fill distance distribution.
    ///
    /// For exponential: CV ≈ 1.0
    /// CV > 1.0: Heavy tail (power-law like) - common in crypto
    /// CV < 1.0: Light tail
    pub fn kappa_cv(&self) -> f64 {
        self.fill_rate_kappa.cv()
    }

    /// Get current order arrival intensity (volume ticks per second).
    pub fn arrival_intensity(&self) -> f64 {
        self.arrival.ticks_per_second()
    }

    /// Get L2 book imbalance [-1, 1].
    /// Positive = more bids (buying pressure), Negative = more asks (selling pressure).
    /// Use for directional quote skew.
    pub fn book_imbalance(&self) -> f64 {
        self.book_structure.imbalance()
    }

    /// Get liquidity-based gamma multiplier [1.0, 2.0].
    /// Returns > 1.0 when near-touch liquidity is below average (thin book).
    /// Use to scale gamma up for wider spreads in thin conditions.
    pub fn liquidity_gamma_multiplier(&self) -> f64 {
        self.book_structure.gamma_multiplier()
    }

    // === Microprice Accessors ===

    /// Get microprice (data-driven fair price).
    /// Incorporates book imbalance and flow imbalance predictions.
    /// Falls back to raw mid if not warmed up.
    pub fn microprice(&self) -> f64 {
        self.microprice_estimator.microprice(
            self.current_mid,
            self.book_structure.imbalance(),
            self.flow.imbalance(),
        )
    }

    /// Get β_book coefficient (return prediction per unit book imbalance).
    pub fn beta_book(&self) -> f64 {
        self.microprice_estimator.beta_book()
    }

    /// Get β_flow coefficient (return prediction per unit flow imbalance).
    pub fn beta_flow(&self) -> f64 {
        self.microprice_estimator.beta_flow()
    }

    /// Get R² of microprice regression.
    pub fn microprice_r_squared(&self) -> f64 {
        self.microprice_estimator.r_squared()
    }

    /// Check if microprice estimator is warmed up.
    pub fn microprice_warmed_up(&self) -> bool {
        self.microprice_estimator.is_warmed_up()
    }

    // === Regime Detection ===

    /// Get fast RV/BV jump ratio.
    /// - ≈ 1.0: Normal diffusion (safe to market make)
    /// - > 1.5: Jumps present (toxic environment)
    pub fn jump_ratio(&self) -> f64 {
        self.multi_scale.jump_ratio_fast()
    }

    /// Check if currently in toxic (jump) regime.
    pub fn is_toxic_regime(&self) -> bool {
        self.multi_scale.jump_ratio_fast() > self.config.jump_ratio_threshold
    }

    /// Get current 4-state volatility regime.
    ///
    /// Regime classification with hysteresis:
    /// - Low: Quiet market (σ < 0.5 × baseline) - can tighten spreads
    /// - Normal: Standard conditions
    /// - High: Elevated volatility (σ > 1.5 × baseline) - widen spreads
    /// - Extreme: Crisis/toxic (σ > 3 × baseline OR high jump ratio) - consider pulling quotes
    pub fn volatility_regime(&self) -> VolatilityRegime {
        self.volatility_regime.regime()
    }

    /// Get spread multiplier based on current volatility regime.
    ///
    /// Ranges from 0.8 (Low) to 2.5 (Extreme).
    pub fn regime_spread_multiplier(&self) -> f64 {
        self.volatility_regime.regime().spread_multiplier()
    }

    /// Get gamma multiplier based on current volatility regime.
    ///
    /// Ranges from 0.8 (Low) to 3.0 (Extreme).
    pub fn regime_gamma_multiplier(&self) -> f64 {
        self.volatility_regime.regime().gamma_multiplier()
    }

    // === Directional Flow Accessors ===

    /// Get signed momentum in bps over momentum window.
    /// Negative = market falling, Positive = market rising.
    pub fn momentum_bps(&self) -> f64 {
        self.momentum.momentum_bps(self.current_time_ms)
    }

    /// Get falling knife score [0, 3].
    /// > 0.5 = some downward momentum
    /// > 1.0 = severe downward momentum (protect bids!)
    pub fn falling_knife_score(&self) -> f64 {
        self.momentum.falling_knife_score(self.current_time_ms)
    }

    /// Get rising knife score [0, 3].
    /// > 0.5 = some upward momentum
    /// > 1.0 = severe upward momentum (protect asks!)
    pub fn rising_knife_score(&self) -> f64 {
        self.momentum.rising_knife_score(self.current_time_ms)
    }

    /// Get trade flow imbalance [-1, 1].
    /// Negative = sell pressure, Positive = buy pressure.
    pub fn flow_imbalance(&self) -> f64 {
        self.flow.imbalance()
    }

    // === Jump Process Accessors (First Principles Gap 1) ===

    /// Get jump intensity (λ) - expected jumps per second.
    pub fn lambda_jump(&self) -> f64 {
        self.jump_estimator.lambda()
    }

    /// Get mean jump size (μ_j) in log-return units.
    pub fn mu_jump(&self) -> f64 {
        self.jump_estimator.mu()
    }

    /// Get jump size standard deviation (σ_j).
    pub fn sigma_jump(&self) -> f64 {
        self.jump_estimator.sigma()
    }

    /// Get total variance including jumps over horizon.
    ///
    /// Var[P(t+h) - P(t)] = σ²h + λh×E[J²]
    pub fn total_variance(&self, horizon_secs: f64) -> f64 {
        let sigma_diffusion = self.multi_scale.sigma_clean();
        self.jump_estimator
            .total_variance(sigma_diffusion, horizon_secs)
    }

    /// Get total volatility including jumps (sqrt of total variance).
    pub fn total_sigma(&self, horizon_secs: f64) -> f64 {
        let sigma_diffusion = self.multi_scale.sigma_clean();
        self.jump_estimator
            .total_sigma(sigma_diffusion, horizon_secs)
    }

    /// Check if jump estimator has enough data.
    pub fn jump_estimator_warmed_up(&self) -> bool {
        self.jump_estimator.is_warmed_up()
    }

    // === Stochastic Volatility Accessors (First Principles Gap 2) ===

    /// Get current instantaneous volatility from stochastic vol model.
    pub fn sigma_stoch_vol(&self) -> f64 {
        self.stoch_vol.sigma_t()
    }

    /// Get volatility mean-reversion speed (κ_vol).
    pub fn kappa_vol(&self) -> f64 {
        self.stoch_vol.kappa()
    }

    /// Get long-run volatility (√θ_vol).
    pub fn theta_vol_sigma(&self) -> f64 {
        self.stoch_vol.theta_sigma()
    }

    /// Get vol-of-vol (ξ).
    pub fn xi_vol(&self) -> f64 {
        self.stoch_vol.xi()
    }

    /// Get price-vol correlation (ρ, typically negative - leverage effect).
    pub fn rho_price_vol(&self) -> f64 {
        self.stoch_vol.rho()
    }

    /// Get expected average volatility over horizon using OU dynamics.
    ///
    /// Accounts for mean-reversion: if σ > θ, vol will decrease toward θ.
    pub fn expected_avg_sigma(&self, horizon_secs: f64) -> f64 {
        self.stoch_vol.expected_avg_sigma(horizon_secs)
    }

    /// Get leverage-adjusted volatility based on recent return.
    ///
    /// When returns are negative and ρ < 0, volatility increases.
    pub fn leverage_adjusted_vol(&self, recent_return: f64) -> f64 {
        self.stoch_vol.leverage_adjusted_vol(recent_return)
    }

    /// Check if stochastic vol is calibrated.
    pub fn stoch_vol_calibrated(&self) -> bool {
        self.stoch_vol.is_calibrated()
    }

    // === Warmup ===

    /// Check if estimator has collected enough data.
    pub fn is_warmed_up(&self) -> bool {
        self.multi_scale.tick_count() >= self.config.min_volume_ticks
            && self.fill_rate_kappa.update_count() >= self.config.min_l2_updates
    }

    /// Get current warmup progress.
    /// Returns (volume_ticks, min_volume_ticks, l2_updates, min_l2_updates)
    pub fn warmup_progress(&self) -> (usize, usize, usize, usize) {
        (
            self.multi_scale.tick_count(),
            self.config.min_volume_ticks,
            self.fill_rate_kappa.update_count(),
            self.config.min_l2_updates,
        )
    }

    /// Get simplified warmup progress for legacy compatibility.
    /// Returns (current_samples, min_samples) based on volume ticks.
    pub fn warmup_progress_simple(&self) -> (usize, usize) {
        (self.multi_scale.tick_count(), self.config.min_volume_ticks)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> EstimatorConfig {
        EstimatorConfig {
            initial_bucket_volume: 1.0,
            min_volume_ticks: 5,
            min_l2_updates: 3,
            fast_half_life_ticks: 5.0,
            medium_half_life_ticks: 10.0,
            slow_half_life_ticks: 50.0,
            kappa_half_life_updates: 10.0,
            ..Default::default()
        }
    }

    #[test]
    fn test_volume_bucket_accumulation() {
        let config = EstimatorConfig {
            initial_bucket_volume: 10.0,
            ..Default::default()
        };
        let mut acc = VolumeBucketAccumulator::new(&config);

        // Should not complete bucket yet
        assert!(acc.on_trade(1000, 100.0, 3.0).is_none());
        assert!(acc.on_trade(2000, 101.0, 3.0).is_none());

        // Should complete bucket at 10+ volume
        let bucket = acc.on_trade(3000, 102.0, 5.0);
        assert!(bucket.is_some());

        let b = bucket.unwrap();
        assert!((b.volume - 11.0).abs() < 0.01);
        // VWAP = (100*3 + 101*3 + 102*5) / 11 = 1113/11 ≈ 101.18
        assert!((b.vwap - 101.18).abs() < 0.1);
    }

    #[test]
    fn test_vwap_calculation() {
        let config = EstimatorConfig {
            initial_bucket_volume: 5.0,
            ..Default::default()
        };
        let mut acc = VolumeBucketAccumulator::new(&config);

        // Trades at different prices
        acc.on_trade(1000, 100.0, 2.0); // 200, vol = 2
        acc.on_trade(2000, 110.0, 2.0); // 220, vol = 4

        // Total: 420 / 4 = 105.0, but need 5 volume
        let bucket = acc.on_trade(2000, 110.0, 0.0); // Force check with zero size
        assert!(bucket.is_none()); // Not enough volume yet

        // Add more to complete (need 1 more)
        let bucket = acc.on_trade(3000, 120.0, 1.0); // 120, vol = 5
        assert!(bucket.is_some());

        let b = bucket.unwrap();
        // VWAP = (200 + 220 + 120) / 5 = 540/5 = 108.0
        assert!((b.vwap - 108.0).abs() < 0.1);
    }

    #[test]
    fn test_single_scale_bipower_no_jumps() {
        let mut bv = SingleScaleBipower::new(10.0, 0.001_f64.powi(2));

        // Feed stable returns (no jumps) - small oscillations
        let vwaps: [f64; 8] = [100.0, 100.1, 100.0, 100.1, 100.0, 100.1, 100.0, 100.1];
        let mut last_vwap: f64 = vwaps[0];
        for vwap in vwaps.iter().skip(1) {
            let log_return = (vwap / last_vwap).ln();
            bv.update(log_return);
            last_vwap = *vwap;
        }

        // Jump ratio should be close to 1.0 (no jumps)
        let ratio = bv.jump_ratio();
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "Expected ratio ~1.0 for no jumps, got {}",
            ratio
        );
    }

    #[test]
    fn test_single_scale_bipower_with_jump() {
        let mut bv = SingleScaleBipower::new(5.0, 0.001_f64.powi(2));

        // Feed returns with a sudden jump
        // Normal, normal, JUMP, normal, normal
        let vwaps: [f64; 7] = [100.0, 100.1, 100.0, 105.0, 105.1, 105.0, 105.1];
        let mut last_vwap: f64 = vwaps[0];
        for vwap in vwaps.iter().skip(1) {
            let log_return = (vwap / last_vwap).ln();
            bv.update(log_return);
            last_vwap = *vwap;
        }

        // Jump ratio should be elevated (RV > BV due to jump)
        let ratio = bv.jump_ratio();
        assert!(
            ratio > 1.5,
            "Expected elevated ratio due to jump, got {}",
            ratio
        );
    }

    #[test]
    fn test_weighted_kappa_estimator() {
        let mut kappa = WeightedKappaEstimator::new(10.0, 100.0, 0.01, 15);
        let mid = 100.0;

        // Synthetic book where depth increases at each level
        // This is typical: more liquidity accumulates further from mid
        // Cumulative depth at level i: sum of sizes from 0 to i
        let bids: Vec<(f64, f64)> = (1..=10)
            .map(|i| {
                let price = mid - i as f64 * 0.05; // 99.95, 99.90, ...
                let size = 1.0; // Constant size at each level
                (price, size)
            })
            .collect();

        let asks: Vec<(f64, f64)> = (1..=10)
            .map(|i| {
                let price = mid + i as f64 * 0.05; // 100.05, 100.10, ...
                let size = 1.0;
                (price, size)
            })
            .collect();

        // Run multiple updates to converge
        for _ in 0..30 {
            kappa.update(&bids, &asks, mid);
        }

        // Kappa should be in a reasonable range (changed from default 100)
        let k = kappa.kappa();
        assert!(
            k > 1.0 && k < 10000.0,
            "Kappa should be in valid range, got {}",
            k
        );
        // Verify it's updating (not stuck at default)
        assert!(
            k != 100.0,
            "Kappa should have changed from default 100, got {}",
            k
        );
    }

    #[test]
    fn test_regime_detection() {
        let mut config = make_config();
        config.jump_ratio_threshold = 2.0;
        let estimator = ParameterEstimator::new(config);

        // Initially not toxic (default ratio = 1.0)
        assert!(!estimator.is_toxic_regime());
    }

    #[test]
    fn test_full_pipeline_warmup() {
        let config = make_config();
        let mut estimator = ParameterEstimator::new(config);

        assert!(!estimator.is_warmed_up());

        // Feed initial L2 book to set current_mid (needed for fill-rate kappa)
        let bids = vec![(99.9, 5.0), (99.8, 10.0), (99.7, 15.0)];
        let asks = vec![(100.1, 5.0), (100.2, 10.0), (100.3, 15.0)];
        estimator.on_l2_book(&bids, &asks, 100.0);

        // Feed trades to fill buckets (need 5 volume ticks)
        // These will also feed into fill_rate_kappa since current_mid is set
        let mut time = 1000u64;
        for i in 0..100 {
            let price = 100.0 + (i as f64 * 0.1).sin() * 0.5;
            // Alternate buy/sell to simulate balanced flow
            let is_buy = i % 2 == 0;
            estimator.on_trade(time, price, 0.5, Some(is_buy)); // 0.5 per trade, 2 trades per bucket
            time += 100;
        }

        // Feed more L2 books (total needs min_l2_updates which is checked via fill_rate_kappa)
        for _ in 0..5 {
            estimator.on_l2_book(&bids, &asks, 100.0);
        }

        // Should be warmed up
        assert!(estimator.is_warmed_up());

        // Params should be in reasonable ranges
        let sigma = estimator.sigma();
        let kappa = estimator.kappa();
        let ratio = estimator.jump_ratio();

        assert!(sigma > 0.0, "sigma should be positive");
        assert!(kappa > 1.0, "kappa should be > 1");
        assert!(ratio > 0.0, "jump_ratio should be positive");
    }

    #[test]
    fn test_adaptive_bucket_threshold() {
        let config = EstimatorConfig {
            initial_bucket_volume: 1.0,
            volume_window_secs: 10.0,
            volume_percentile: 0.1, // 10% of rolling volume
            min_bucket_volume: 0.5,
            max_bucket_volume: 10.0,
            ..Default::default()
        };
        let mut acc = VolumeBucketAccumulator::new(&config);

        // Fill several buckets to build up rolling volume
        let mut time = 0u64;
        for _ in 0..10 {
            while acc.on_trade(time, 100.0, 0.5).is_none() {
                time += 100;
            }
        }

        // Threshold should have adapted based on rolling volume
        // With 10 buckets of ~1.0 volume each in 10 seconds,
        // and 10% percentile, threshold should be around 1.0
        assert!(acc.threshold >= 0.5);
        assert!(acc.threshold <= 10.0);
    }

    #[test]
    fn test_arrival_intensity() {
        let config = make_config();
        let mut estimator = ParameterEstimator::new(config);

        // Feed trades at consistent rate to fill buckets
        let mut time = 0u64;
        for i in 0..50 {
            let is_buy = i % 2 == 0;
            estimator.on_trade(time, 100.0, 1.0, Some(is_buy)); // Each trade = 1 bucket
            time += 500; // 500ms between buckets = 2 ticks/sec
        }

        let intensity = estimator.arrival_intensity();
        // Should be close to 2 ticks/sec
        assert!(
            intensity > 1.0 && intensity < 5.0,
            "Expected ~2 ticks/sec, got {}",
            intensity
        );
    }
}
