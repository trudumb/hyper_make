//! Hawkes Order Flow Estimator - self-exciting point process for trade arrivals.
//!
//! Models order flow as a self-exciting point process where each trade increases
//! the probability of subsequent trades. This captures the clustering/momentum
//! behavior observed in real markets.
//!
//! # The Model
//! ```text
//! λ_buy(t)  = μ_buy  + ∫ α × e^(-β(t-s)) dN_buy(s)  + γ × e^(-β(t-s)) dN_sell(s)
//! λ_sell(t) = μ_sell + ∫ α × e^(-β(t-s)) dN_sell(s) + γ × e^(-β(t-s)) dN_buy(s)
//! ```
//!
//! where:
//! - λ(t): Instantaneous intensity (expected trades per second)
//! - μ: Baseline intensity (background rate)
//! - α: Self-excitation (how much each trade excites same-side intensity)
//! - β: Decay rate (how fast excitation fades)
//! - γ: Cross-excitation (how much each trade excites opposite-side intensity)
//!
//! # Key Properties
//! - **Clustering**: High activity begets more activity
//! - **Mean-reversion**: Intensity reverts to baseline μ
//! - **Asymmetry**: Buy and sell intensities can differ

use std::collections::VecDeque;
use std::time::Instant;

/// Configuration for the Hawkes order flow estimator.
#[derive(Debug, Clone)]
pub struct HawkesConfig {
    /// Baseline intensity (trades per second)
    /// Default: 0.5 (1 trade per 2 seconds)
    pub mu: f64,

    /// Self-excitation parameter (0 to 1)
    /// Higher values = more clustering
    /// Default: 0.3
    pub alpha: f64,

    /// Decay rate (per second)
    /// Higher values = faster decay of excitation
    /// Default: 0.1 (10 second half-life)
    pub beta: f64,

    /// Cross-excitation parameter (0 to 1)
    /// How much opposite-side trades increase intensity
    /// Default: 0.1
    pub gamma: f64,

    /// Maximum history to keep (in seconds)
    /// Default: 60
    pub max_history_secs: f64,

    /// Minimum trades for warmup
    pub min_trades: usize,
}

impl Default for HawkesConfig {
    fn default() -> Self {
        Self {
            mu: 0.5,
            alpha: 0.3,
            beta: 0.1,
            gamma: 0.1,
            max_history_secs: 60.0,
            min_trades: 10,
        }
    }
}

/// A single trade event for the Hawkes process.
#[derive(Debug, Clone, Copy)]
struct TradeEvent {
    timestamp: Instant,
    is_buy: bool,
    size: f64,
}

/// Hawkes order flow estimator.
pub struct HawkesOrderFlowEstimator {
    config: HawkesConfig,

    /// Recent trade events
    events: VecDeque<TradeEvent>,

    /// Current buy intensity
    lambda_buy: f64,

    /// Current sell intensity
    lambda_sell: f64,

    /// Last intensity update time
    last_update: Instant,

    /// Total trade count
    trade_count: usize,

    /// Running sum of buy volume
    buy_volume: f64,

    /// Running sum of sell volume
    sell_volume: f64,
}

impl HawkesOrderFlowEstimator {
    /// Create a new Hawkes order flow estimator.
    pub fn new(config: HawkesConfig) -> Self {
        Self {
            lambda_buy: config.mu,
            lambda_sell: config.mu,
            config,
            events: VecDeque::with_capacity(1000),
            last_update: Instant::now(),
            trade_count: 0,
            buy_volume: 0.0,
            sell_volume: 0.0,
        }
    }

    /// Record a trade event.
    ///
    /// # Arguments
    /// - `is_buy`: Whether this was a buy aggressor trade
    /// - `size`: Trade size (for volume-weighted analysis)
    pub fn record_trade(&mut self, is_buy: bool, size: f64) {
        let now = Instant::now();

        // Update intensities before adding new event
        self.update_intensities(now);

        // Record event
        let event = TradeEvent {
            timestamp: now,
            is_buy,
            size,
        };
        self.events.push_back(event);
        self.trade_count += 1;

        if is_buy {
            self.buy_volume += size;
        } else {
            self.sell_volume += size;
        }

        // Add excitation from this trade
        if is_buy {
            self.lambda_buy += self.config.alpha;
            self.lambda_sell += self.config.gamma;
        } else {
            self.lambda_sell += self.config.alpha;
            self.lambda_buy += self.config.gamma;
        }

        // Cleanup old events
        self.cleanup_old_events(now);

        self.last_update = now;
    }

    /// Update intensities based on time decay.
    fn update_intensities(&mut self, now: Instant) {
        let dt = now.duration_since(self.last_update).as_secs_f64();
        if dt <= 0.0 {
            return;
        }

        let decay = (-self.config.beta * dt).exp();

        // Decay intensities toward baseline
        self.lambda_buy = self.config.mu + (self.lambda_buy - self.config.mu) * decay;
        self.lambda_sell = self.config.mu + (self.lambda_sell - self.config.mu) * decay;
    }

    /// Remove events older than max_history_secs.
    fn cleanup_old_events(&mut self, now: Instant) {
        let cutoff_duration = std::time::Duration::from_secs_f64(self.config.max_history_secs);

        while let Some(front) = self.events.front() {
            if now.duration_since(front.timestamp) > cutoff_duration {
                let removed = self.events.pop_front().unwrap();
                if removed.is_buy {
                    self.buy_volume -= removed.size;
                } else {
                    self.sell_volume -= removed.size;
                }
            } else {
                break;
            }
        }
    }

    /// Update without a trade (just decay).
    pub fn update(&mut self) {
        let now = Instant::now();
        self.update_intensities(now);
        self.cleanup_old_events(now);
        self.last_update = now;
    }

    /// Check if estimator is warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.trade_count >= self.config.min_trades
    }

    /// Get current buy intensity (trades per second).
    pub fn lambda_buy(&self) -> f64 {
        self.lambda_buy
    }

    /// Get current sell intensity (trades per second).
    pub fn lambda_sell(&self) -> f64 {
        self.lambda_sell
    }

    /// Get total intensity.
    pub fn lambda_total(&self) -> f64 {
        self.lambda_buy + self.lambda_sell
    }

    /// Get intensity ratio (current / baseline).
    ///
    /// Used for adaptive hazard rate in BOCPD (First-Principles Gap 1).
    /// - 1.0: Normal activity (at baseline)
    /// - >1.0: Elevated activity (higher changepoint probability)
    /// - <1.0: Low activity
    pub fn intensity_ratio(&self) -> f64 {
        let baseline = 2.0 * self.config.mu;
        self.lambda_total() / baseline
    }

    /// Get flow imbalance from intensities.
    ///
    /// Returns value in [-1, 1]:
    /// - Positive: More buy intensity
    /// - Negative: More sell intensity
    pub fn flow_imbalance(&self) -> f64 {
        let total = self.lambda_buy + self.lambda_sell;
        if total < 1e-9 {
            return 0.0;
        }
        (self.lambda_buy - self.lambda_sell) / total
    }

    /// Get the expected number of trades in the next N seconds.
    ///
    /// Integrates the expected intensity decay.
    pub fn expected_trades(&self, horizon_secs: f64) -> (f64, f64) {
        let beta = self.config.beta;
        let mu = self.config.mu;

        // For exponential decay from current level to baseline:
        // ∫₀^T [μ + (λ₀ - μ)e^(-βt)] dt = μT + (λ₀ - μ)(1 - e^(-βT))/β

        let decay_factor = (1.0 - (-beta * horizon_secs).exp()) / beta;

        let expected_buy = mu * horizon_secs + (self.lambda_buy - mu) * decay_factor;
        let expected_sell = mu * horizon_secs + (self.lambda_sell - mu) * decay_factor;

        (expected_buy.max(0.0), expected_sell.max(0.0))
    }

    /// Get the intensity percentile (how unusual is current activity).
    ///
    /// Returns value in [0, 1] where:
    /// - 0.5: Normal activity (at baseline)
    /// - 1.0: Very high activity
    /// - 0.0: Very low activity
    pub fn intensity_percentile(&self) -> f64 {
        let total = self.lambda_total();
        let baseline = 2.0 * self.config.mu;

        // Simple mapping: 1 at baseline, higher above, lower below
        // Use sigmoid-like mapping
        let ratio = total / baseline;
        1.0 / (1.0 + (-2.0 * (ratio - 1.0)).exp())
    }

    /// Get optimal quote duration based on expected flow.
    ///
    /// In high-intensity periods, quotes should be shorter-lived.
    /// In low-intensity periods, quotes can persist longer.
    pub fn optimal_quote_duration(&self) -> f64 {
        let intensity = self.lambda_total();
        let baseline = 2.0 * self.config.mu;

        // Base duration of 5 seconds, scaled by activity
        let base_duration = 5.0;
        let ratio = intensity / baseline;

        // Higher intensity -> shorter duration (but cap at 1 second minimum)
        (base_duration / ratio).clamp(1.0, 30.0)
    }

    /// Get position sizing factor based on flow uncertainty.
    ///
    /// In high-intensity periods with imbalanced flow, reduce size.
    pub fn position_sizing_factor(&self) -> f64 {
        let imbalance = self.flow_imbalance().abs();
        let intensity_ratio = self.lambda_total() / (2.0 * self.config.mu);

        // Reduce size when: high intensity AND imbalanced flow
        // Factor of 1.0 = normal, 0.5 = half size
        if intensity_ratio > 2.0 && imbalance > 0.3 {
            0.5
        } else if intensity_ratio > 1.5 || imbalance > 0.2 {
            0.75
        } else {
            1.0
        }
    }

    /// Get recent trade rate (trades per second in last window).
    pub fn recent_trade_rate(&self) -> f64 {
        let window_secs = self.config.max_history_secs.min(60.0);
        self.events.len() as f64 / window_secs
    }

    /// Get volume imbalance from recent history.
    pub fn volume_imbalance(&self) -> f64 {
        let total = self.buy_volume + self.sell_volume;
        if total < 1e-9 {
            return 0.0;
        }
        (self.buy_volume - self.sell_volume) / total
    }

    /// Get summary statistics.
    pub fn summary(&self) -> HawkesSummary {
        let (expected_buy, expected_sell) = self.expected_trades(60.0);

        HawkesSummary {
            is_warmed_up: self.is_warmed_up(),
            lambda_buy: self.lambda_buy,
            lambda_sell: self.lambda_sell,
            lambda_total: self.lambda_total(),
            flow_imbalance: self.flow_imbalance(),
            intensity_percentile: self.intensity_percentile(),
            trade_count: self.trade_count,
            recent_events: self.events.len(),
            expected_trades_1m: expected_buy + expected_sell,
            buy_volume: self.buy_volume,
            sell_volume: self.sell_volume,
        }
    }

    /// Excess intensity as drift observation (z, R) for the Kalman filter.
    ///
    /// When λ_sell >> λ_base, we're in a selling burst → bearish drift.
    /// When λ_buy >> λ_base, buying burst → bullish drift.
    /// Returns None if not warmed up.
    pub fn drift_observation(&self) -> Option<(f64, f64)> {
        if !self.is_warmed_up() {
            return None;
        }
        let excess_sell = self.lambda_sell - self.config.mu;
        let excess_buy = self.lambda_buy - self.config.mu;
        // Net excess sell pressure → negative z (bearish)
        let z = -(excess_sell - excess_buy) * 0.3; // 0.3 sensitivity scaling
        let sigma_hawkes = 1.5;
        let total_lambda = self.lambda_total().max(0.1);
        let r = sigma_hawkes * sigma_hawkes / total_lambda;
        Some((z, r))
    }

    /// Reset the estimator.
    pub fn reset(&mut self) {
        self.events.clear();
        self.lambda_buy = self.config.mu;
        self.lambda_sell = self.config.mu;
        self.trade_count = 0;
        self.buy_volume = 0.0;
        self.sell_volume = 0.0;
        self.last_update = Instant::now();
    }
}

/// Summary of Hawkes order flow status.
#[derive(Debug, Clone)]
pub struct HawkesSummary {
    pub is_warmed_up: bool,
    pub lambda_buy: f64,
    pub lambda_sell: f64,
    pub lambda_total: f64,
    pub flow_imbalance: f64,
    pub intensity_percentile: f64,
    pub trade_count: usize,
    pub recent_events: usize,
    pub expected_trades_1m: f64,
    pub buy_volume: f64,
    pub sell_volume: f64,
}

// ============================================================================
// GMM Calibration for Hawkes Process
// ============================================================================

/// Result of GMM calibration for Hawkes process parameters.
#[derive(Debug, Clone, Copy)]
pub struct HawkesGmmResult {
    /// Estimated baseline intensity (λ₀ or μ)
    pub lambda_0: f64,
    /// Estimated self-excitation parameter (α)
    pub alpha: f64,
    /// Estimated decay rate (β)
    pub beta: f64,
    /// Branching ratio α/β (must be < 1 for stationarity)
    pub branching_ratio: f64,
    /// Sample mean of counts
    pub sample_mean: f64,
    /// Sample variance of counts
    pub sample_variance: f64,
    /// Number of windows used for estimation
    pub n_windows: usize,
    /// Estimation quality: ratio of theoretical to empirical variance
    pub fit_quality: f64,
}

impl HawkesGmmResult {
    /// Check if the estimated process is stationary (branching ratio < 1).
    pub fn is_stationary(&self) -> bool {
        self.branching_ratio < 1.0
    }

    /// Check if the fit is reasonable (quality close to 1.0).
    pub fn is_well_fit(&self) -> bool {
        self.fit_quality > 0.5 && self.fit_quality < 2.0
    }

    /// Get the long-run expected intensity.
    /// E[λ] = λ₀ / (1 - α/β)
    pub fn long_run_intensity(&self) -> f64 {
        if self.branching_ratio >= 1.0 {
            return f64::INFINITY;
        }
        self.lambda_0 / (1.0 - self.branching_ratio)
    }
}

/// GMM calibrator for Hawkes process parameters.
///
/// Uses closed-form moment conditions to estimate parameters from fill history:
/// - E[N(T)] = λ₀ × T / (1 - α/β)
/// - Var[N(T)] = E[N(T)] × (1 + 2α/(β - α))
///
/// This is much faster than MLE and works well for real-time calibration.
#[derive(Debug, Clone)]
pub struct HawkesGmmCalibrator {
    /// Window size in seconds for counting
    window_size_secs: f64,
    /// Trade counts per window
    window_counts: Vec<usize>,
    /// Minimum windows needed for estimation
    min_windows: usize,
    /// Prior estimate of beta for initialization
    beta_prior: f64,
}

impl HawkesGmmCalibrator {
    /// Create a new GMM calibrator.
    ///
    /// # Arguments
    /// - `window_size_secs`: Size of counting windows (e.g., 10 seconds)
    /// - `min_windows`: Minimum windows needed before estimation (e.g., 30)
    /// - `beta_prior`: Prior estimate of decay rate for regularization
    pub fn new(window_size_secs: f64, min_windows: usize, beta_prior: f64) -> Self {
        Self {
            window_size_secs,
            window_counts: Vec::with_capacity(min_windows * 2),
            min_windows,
            beta_prior,
        }
    }

    /// Create with default settings (10s windows, 30 minimum, beta=0.1).
    pub fn default_calibrator() -> Self {
        Self::new(10.0, 30, 0.1)
    }

    /// Add a window count observation.
    pub fn add_window_count(&mut self, count: usize) {
        self.window_counts.push(count);

        // Keep only recent history (rolling window of 1000 observations)
        if self.window_counts.len() > 1000 {
            self.window_counts.remove(0);
        }
    }

    /// Add multiple fill timestamps and compute window counts.
    ///
    /// # Arguments
    /// - `timestamps_secs`: Fill timestamps in seconds (relative to some epoch)
    pub fn add_fill_history(&mut self, timestamps_secs: &[f64]) {
        if timestamps_secs.is_empty() {
            return;
        }

        let min_t = timestamps_secs
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_t = timestamps_secs
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let duration = max_t - min_t;
        if duration < self.window_size_secs {
            // Not enough data for even one window
            self.window_counts.push(timestamps_secs.len());
            return;
        }

        let n_windows = (duration / self.window_size_secs).floor() as usize;

        for i in 0..n_windows {
            let window_start = min_t + (i as f64) * self.window_size_secs;
            let window_end = window_start + self.window_size_secs;

            let count = timestamps_secs
                .iter()
                .filter(|&&t| t >= window_start && t < window_end)
                .count();

            self.window_counts.push(count);
        }
    }

    /// Check if we have enough data for calibration.
    pub fn has_sufficient_data(&self) -> bool {
        self.window_counts.len() >= self.min_windows
    }

    /// Get the number of windows collected.
    pub fn n_windows(&self) -> usize {
        self.window_counts.len()
    }

    /// Compute sample mean of window counts.
    fn sample_mean(&self) -> f64 {
        if self.window_counts.is_empty() {
            return 0.0;
        }
        let sum: usize = self.window_counts.iter().sum();
        sum as f64 / self.window_counts.len() as f64
    }

    /// Compute sample variance of window counts.
    fn sample_variance(&self) -> f64 {
        if self.window_counts.len() < 2 {
            return 0.0;
        }
        let mean = self.sample_mean();
        let sum_sq: f64 = self
            .window_counts
            .iter()
            .map(|&c| {
                let diff = c as f64 - mean;
                diff * diff
            })
            .sum();
        sum_sq / (self.window_counts.len() - 1) as f64
    }

    /// Calibrate Hawkes parameters using GMM.
    ///
    /// Uses moment conditions:
    /// - E[N(T)] = λ₀ × T / (1 - α/β)
    /// - Var[N(T)] = E[N(T)] × (1 + 2α/(β - α))
    ///
    /// Returns None if insufficient data or estimation fails.
    pub fn calibrate(&self) -> Option<HawkesGmmResult> {
        if !self.has_sufficient_data() {
            return None;
        }

        let mean = self.sample_mean();
        let variance = self.sample_variance();
        let t = self.window_size_secs;

        // Sanity checks
        if mean < 0.001 || variance < 0.001 {
            // Not enough activity
            return Some(HawkesGmmResult {
                lambda_0: mean / t,
                alpha: 0.0,
                beta: self.beta_prior,
                branching_ratio: 0.0,
                sample_mean: mean,
                sample_variance: variance,
                n_windows: self.window_counts.len(),
                fit_quality: 1.0,
            });
        }

        // From the moment equations:
        // Let m = E[N(T)], v = Var[N(T)]
        // m = λ₀ × T / (1 - α/β)    ... (1)
        // v = m × (1 + 2α/(β - α))  ... (2)
        //
        // Let r = α/β (branching ratio), then:
        // m = λ₀ × T / (1 - r)
        // v/m = 1 + 2r/(1 - r) = 1 + 2r/(1-r) = (1 - r + 2r)/(1-r) = (1 + r)/(1 - r)
        //
        // So: v/m = (1 + r)/(1 - r)
        // Solving: v/m × (1 - r) = 1 + r
        //          v/m - v/m × r = 1 + r
        //          v/m - 1 = r + v/m × r = r(1 + v/m)
        //          r = (v/m - 1) / (1 + v/m) = (v - m) / (v + m)

        let variance_to_mean = variance / mean;

        // Estimate branching ratio
        let r = if variance_to_mean > 1.0 {
            // Overdispersed (typical for Hawkes)
            ((variance - mean) / (variance + mean)).clamp(0.0, 0.95)
        } else {
            // Underdispersed - no self-excitation
            0.0
        };

        // Estimate λ₀ from mean: λ₀ = m × (1 - r) / T
        let lambda_0 = (mean * (1.0 - r) / t).max(0.001);

        // Use prior beta and solve for alpha: α = r × β
        let beta = self.beta_prior;
        let alpha = r * beta;

        // Compute theoretical variance for fit quality
        let theoretical_var = if r < 1.0 {
            mean * (1.0 + r) / (1.0 - r)
        } else {
            f64::INFINITY
        };
        let fit_quality = if theoretical_var > 0.0 && theoretical_var.is_finite() {
            variance / theoretical_var
        } else {
            0.0
        };

        Some(HawkesGmmResult {
            lambda_0,
            alpha,
            beta,
            branching_ratio: r,
            sample_mean: mean,
            sample_variance: variance,
            n_windows: self.window_counts.len(),
            fit_quality,
        })
    }

    /// Calibrate with variable beta estimation.
    ///
    /// Uses additional moment condition to estimate beta from
    /// autocorrelation of window counts.
    pub fn calibrate_with_beta(&self) -> Option<HawkesGmmResult> {
        if self.window_counts.len() < self.min_windows * 2 {
            // Need more data for autocorrelation
            return self.calibrate();
        }

        // Compute lag-1 autocorrelation
        let mean = self.sample_mean();
        let variance = self.sample_variance();

        if variance < 0.001 {
            return self.calibrate();
        }

        let mut autocovar = 0.0;
        for i in 0..(self.window_counts.len() - 1) {
            let x = self.window_counts[i] as f64 - mean;
            let y = self.window_counts[i + 1] as f64 - mean;
            autocovar += x * y;
        }
        autocovar /= (self.window_counts.len() - 1) as f64;

        let autocorr = autocovar / variance;

        // For Hawkes: autocorr ≈ exp(-β × T) for adjacent windows
        // So: β ≈ -ln(autocorr) / T
        let estimated_beta = if autocorr > 0.01 && autocorr < 0.99 {
            (-autocorr.ln() / self.window_size_secs).clamp(0.01, 10.0)
        } else {
            self.beta_prior
        };

        // Now calibrate with estimated beta
        let mean_val = self.sample_mean();
        let variance_val = self.sample_variance();
        let t = self.window_size_secs;

        let variance_to_mean = variance_val / mean_val;
        let r = if variance_to_mean > 1.0 {
            ((variance_val - mean_val) / (variance_val + mean_val)).clamp(0.0, 0.95)
        } else {
            0.0
        };

        let lambda_0 = (mean_val * (1.0 - r) / t).max(0.001);
        let alpha = r * estimated_beta;

        let theoretical_var = if r < 1.0 {
            mean_val * (1.0 + r) / (1.0 - r)
        } else {
            f64::INFINITY
        };
        let fit_quality = if theoretical_var > 0.0 && theoretical_var.is_finite() {
            variance_val / theoretical_var
        } else {
            0.0
        };

        Some(HawkesGmmResult {
            lambda_0,
            alpha,
            beta: estimated_beta,
            branching_ratio: r,
            sample_mean: mean_val,
            sample_variance: variance_val,
            n_windows: self.window_counts.len(),
            fit_quality,
        })
    }

    /// Reset the calibrator.
    pub fn reset(&mut self) {
        self.window_counts.clear();
    }
}

/// Fast online Hawkes calibrator that updates incrementally.
///
/// Maintains running statistics for O(1) parameter updates.
#[derive(Debug, Clone)]
pub struct OnlineHawkesCalibrator {
    /// Window size in seconds
    window_size_secs: f64,
    /// EWMA decay for statistics (fast, for mean)
    ewma_alpha: f64,
    /// Slower EWMA decay for variance (2nd-order moment needs more stability)
    ewma_alpha_var: f64,
    /// Running mean of counts
    ewma_mean: f64,
    /// Running variance (Welford's algorithm adapted for EWMA)
    ewma_variance: f64,
    /// Running autocorrelation
    ewma_autocorr: f64,
    /// Last window count
    last_count: Option<usize>,
    /// Total windows seen
    n_windows: usize,
    /// Prior for beta
    beta_prior: f64,
}

impl OnlineHawkesCalibrator {
    /// Create a new online calibrator.
    ///
    /// # Arguments
    /// - `window_size_secs`: Size of counting windows
    /// - `half_life_windows`: Half-life in number of windows for EWMA
    /// - `beta_prior`: Prior estimate of decay rate
    pub fn new(window_size_secs: f64, half_life_windows: f64, beta_prior: f64) -> Self {
        let ewma_alpha = 1.0 - (-2.0_f64.ln() / half_life_windows).exp();
        // Variance is a 2nd-order moment — needs slower alpha for stability.
        // With fast alpha, variance can momentarily drop near zero, causing
        // autocorrelation spikes. Use ~1/3 of the mean alpha.
        let ewma_alpha_var = ewma_alpha * 0.3;
        Self {
            window_size_secs,
            ewma_alpha,
            ewma_alpha_var,
            ewma_mean: 0.0,
            ewma_variance: 1.0,
            ewma_autocorr: 0.0,
            last_count: None,
            n_windows: 0,
            beta_prior,
        }
    }

    /// Update with a new window count.
    pub fn update(&mut self, count: usize) {
        let x = count as f64;
        self.n_windows += 1;

        if self.n_windows == 1 {
            self.ewma_mean = x;
            self.ewma_variance = 1.0; // Initialize with small variance
            self.last_count = Some(count);
            return;
        }

        // Update mean
        let old_mean = self.ewma_mean;
        self.ewma_mean = self.ewma_alpha * x + (1.0 - self.ewma_alpha) * self.ewma_mean;

        // Update variance with slower alpha (2nd-order moment needs more stability)
        let diff = x - old_mean;
        let var_obs = diff * diff;
        self.ewma_variance =
            self.ewma_alpha_var * var_obs + (1.0 - self.ewma_alpha_var) * self.ewma_variance;

        // Update autocorrelation with bounded observations
        if let Some(last) = self.last_count {
            let last_diff = last as f64 - old_mean;
            // Normalize each diff by sqrt(variance) before multiplying to keep
            // autocorr_obs in [-1, 1]. This prevents spikes when variance
            // momentarily drops near zero.
            let denom = (self.ewma_variance + 0.001).sqrt();
            let autocorr_obs = ((diff / denom) * (last_diff / denom)).clamp(-1.0, 1.0);
            self.ewma_autocorr =
                self.ewma_alpha * autocorr_obs + (1.0 - self.ewma_alpha) * self.ewma_autocorr;
        }

        self.last_count = Some(count);
    }

    /// Get current parameter estimates.
    pub fn estimate(&self) -> Option<HawkesGmmResult> {
        if self.n_windows < 10 {
            return None;
        }

        let mean = self.ewma_mean;
        let variance = self.ewma_variance;
        let t = self.window_size_secs;

        if mean < 0.001 {
            return Some(HawkesGmmResult {
                lambda_0: 0.001,
                alpha: 0.0,
                beta: self.beta_prior,
                branching_ratio: 0.0,
                sample_mean: mean,
                sample_variance: variance,
                n_windows: self.n_windows,
                fit_quality: 1.0,
            });
        }

        // Estimate branching ratio from variance/mean
        let r = if variance > mean {
            ((variance - mean) / (variance + mean)).clamp(0.0, 0.95)
        } else {
            0.0
        };

        // Estimate beta from autocorrelation — robust for all signs
        let beta = if self.ewma_autocorr > 0.01 && self.ewma_autocorr < 0.99 {
            // Normal case: positive autocorrelation → clustered arrivals
            (-self.ewma_autocorr.ln() / t).clamp(0.01, 10.0)
        } else if self.ewma_autocorr <= 0.01 {
            // Negative or near-zero autocorrelation → anti-bunching or Poisson
            // ln(negative) would be NaN; use 2x prior (faster decay than clustered)
            (self.beta_prior * 2.0).min(10.0)
        } else {
            // autocorr >= 0.99 → near-perfect correlation, use prior
            self.beta_prior
        };

        let lambda_0 = (mean * (1.0 - r) / t).max(0.001);
        let alpha = r * beta;

        let theoretical_var = if r < 1.0 {
            mean * (1.0 + r) / (1.0 - r)
        } else {
            variance
        };
        let fit_quality = if theoretical_var > 0.0 {
            variance / theoretical_var
        } else {
            0.0
        };

        Some(HawkesGmmResult {
            lambda_0,
            alpha,
            beta,
            branching_ratio: r,
            sample_mean: mean,
            sample_variance: variance,
            n_windows: self.n_windows,
            fit_quality,
        })
    }

    /// Reset the calibrator.
    pub fn reset(&mut self) {
        self.ewma_mean = 0.0;
        self.ewma_variance = 1.0;
        self.ewma_autocorr = 0.0;
        self.last_count = None;
        self.n_windows = 0;
        // ewma_alpha_var and beta_prior are configuration — don't reset
    }
}

// ============================================================================
// Hawkes Excitation Predictor (Bayesian Fusion)
// ============================================================================

/// Configuration for Hawkes excitation prediction.
#[derive(Debug, Clone)]
pub struct HawkesExcitationConfig {
    /// Threshold branching ratio above which we consider high excitation
    /// (n = α/β, typically concerning when n > 0.7)
    pub high_excitation_branching_ratio: f64,

    /// Threshold intensity percentile for early warning
    pub high_intensity_percentile: f64,

    /// Time horizon for cluster probability (seconds)
    pub cluster_horizon_secs: f64,

    /// Minimum penalty multiplier (never go below this)
    pub min_penalty: f64,

    /// Maximum penalty multiplier (cap at this)
    pub max_penalty: f64,
}

impl Default for HawkesExcitationConfig {
    fn default() -> Self {
        Self {
            high_excitation_branching_ratio: 0.7,
            high_intensity_percentile: 0.8,
            cluster_horizon_secs: 10.0,
            min_penalty: 0.5,
            max_penalty: 1.0,
        }
    }
}

/// Prediction result from Hawkes excitation analysis.
#[derive(Debug, Clone, Copy)]
pub struct HawkesExcitationPrediction {
    /// Probability of cluster occurring in next tau seconds
    pub p_cluster: f64,

    /// Penalty multiplier for edge calculation (0.5-1.0)
    /// Lower values = more conservative quoting
    pub excitation_penalty: f64,

    /// Whether we're in a high excitation state
    pub is_high_excitation: bool,

    /// Current branching ratio from calibration
    pub branching_ratio: f64,

    /// Current intensity percentile
    pub intensity_percentile: f64,

    /// Excess intensity ratio: λ_current / λ_baseline
    pub excess_intensity_ratio: f64,

    /// Expected time to next cluster event (seconds)
    pub expected_cluster_time_secs: f64,

    /// Recommended spread widening factor
    pub spread_widening_factor: f64,
}

impl Default for HawkesExcitationPrediction {
    fn default() -> Self {
        Self {
            p_cluster: 0.0,
            excitation_penalty: 1.0,
            is_high_excitation: false,
            branching_ratio: 0.0,
            intensity_percentile: 0.5,
            excess_intensity_ratio: 1.0,
            expected_cluster_time_secs: f64::INFINITY,
            spread_widening_factor: 1.0,
        }
    }
}

/// Hawkes Excitation Predictor - Bayesian fusion of Hawkes process with quoting decisions.
///
/// Uses the branching ratio (n = α/β) and current intensity to predict:
/// 1. P(cluster in next τ seconds) - probability of cascade/clustering
/// 2. Excitation penalty - multiplier for edge calculation
/// 3. Spread widening factor - how much to widen spreads defensively
///
/// # Theory
///
/// For a Hawkes process with intensity λ(t) = λ₀ + Σᵢ α × e^(-β(t-tᵢ)):
/// - Branching ratio n = α/β determines criticality (n < 1 for stationarity)
/// - When n → 1, the process becomes critical (infinite clustering)
/// - Excess intensity λ_excess = λ(t) - λ₀ measures current excitation level
///
/// The probability of k additional events given one triggering event follows
/// a negative binomial distribution with mean n/(1-n). This allows us to
/// compute P(cluster) = P(≥1 child event | current excitation).
///
/// # Integration with Bayesian Adverse Selection
///
/// The excitation penalty directly modifies expected edge:
/// ```text
/// effective_edge = base_edge × excitation_penalty
/// ```
///
/// When P(cluster) is high, excitation_penalty is low, reducing quoting aggressiveness.
///
/// # Synchronicity Coefficient
///
/// Measures event density entropy — toxic flow arrives in bursts of many trades
/// with low inter-arrival time entropy. Returns [0, 1]: 0 = random noise,
/// 1 = synchronized toxic flow.
#[derive(Debug, Clone)]
pub struct HawkesExcitationPredictor {
    config: HawkesExcitationConfig,

    /// Latest GMM calibration result
    latest_calibration: Option<HawkesGmmResult>,

    /// Latest summary from order flow estimator
    latest_summary: Option<HawkesSummary>,

    /// Baseline intensity (λ₀) from calibration
    baseline_intensity: f64,

    /// EWMA of intensity percentile for smoothing
    ewma_intensity_percentile: f64,

    /// EWMA decay factor
    ewma_alpha: f64,

    // --- Synchronicity fields ---
    /// Recent inter-arrival times (seconds) for entropy computation.
    /// Defaults to empty on construction (checkpoint-compatible).
    inter_arrival_buffer: VecDeque<f64>,

    /// Timestamp (seconds, monotonic) of the last recorded event.
    /// Defaults to None on construction (checkpoint-compatible).
    last_event_time_secs: Option<f64>,

    /// EWMA baseline density (events/sec) for normalizing burst detection.
    /// Defaults to 1.0 on construction (checkpoint-compatible).
    baseline_density: f64,
}

impl HawkesExcitationPredictor {
    /// Create a new predictor with default config.
    pub fn new() -> Self {
        Self::with_config(HawkesExcitationConfig::default())
    }

    /// Create a new predictor with custom config.
    pub fn with_config(config: HawkesExcitationConfig) -> Self {
        Self {
            config,
            latest_calibration: None,
            latest_summary: None,
            baseline_intensity: 0.5, // Default baseline
            ewma_intensity_percentile: 0.5,
            ewma_alpha: 0.1, // Smooth over ~10 updates
            inter_arrival_buffer: VecDeque::with_capacity(20),
            last_event_time_secs: None,
            baseline_density: 1.0, // 1 event/sec default
        }
    }

    /// Update with new GMM calibration result.
    pub fn update_calibration(&mut self, calibration: HawkesGmmResult) {
        self.baseline_intensity = calibration.lambda_0;
        self.latest_calibration = Some(calibration);
    }

    /// Update with new Hawkes summary.
    pub fn update_summary(&mut self, summary: HawkesSummary) {
        // Update EWMA of intensity percentile
        self.ewma_intensity_percentile = self.ewma_alpha * summary.intensity_percentile
            + (1.0 - self.ewma_alpha) * self.ewma_intensity_percentile;

        self.latest_summary = Some(summary);
    }

    /// Record an event timestamp for synchronicity tracking.
    ///
    /// Call this on every trade event. The timestamp must be monotonically
    /// increasing (seconds, e.g. `Instant::now().elapsed().as_secs_f64()`
    /// or any monotonic clock).
    pub fn record_event_time(&mut self, timestamp_secs: f64) {
        const MAX_BUFFER: usize = 20;
        const DENSITY_EWMA_ALPHA: f64 = 0.05;

        if let Some(last_t) = self.last_event_time_secs {
            let dt = (timestamp_secs - last_t).max(1e-9);
            self.inter_arrival_buffer.push_back(dt);
            if self.inter_arrival_buffer.len() > MAX_BUFFER {
                self.inter_arrival_buffer.pop_front();
            }

            // Update baseline density EWMA (events/sec)
            let instant_density = 1.0 / dt;
            self.baseline_density = DENSITY_EWMA_ALPHA * instant_density
                + (1.0 - DENSITY_EWMA_ALPHA) * self.baseline_density;
        }
        self.last_event_time_secs = Some(timestamp_secs);
    }

    /// Synchronicity coefficient: measures event density entropy.
    ///
    /// Toxic flow arrives in bursts with uniform (low-entropy) inter-arrival
    /// times — many trades spaced equally within a short window. Random noise
    /// has high-entropy (varied) inter-arrival times.
    ///
    /// Returns `[0, 1]`:
    /// - `0.0` = random noise (high entropy, no density burst)
    /// - `1.0` = synchronized toxic flow (low entropy, high density burst)
    ///
    /// Formula:
    /// ```text
    /// sync = (1 - H_norm) * min(density / baseline_density, 1.0)
    /// ```
    /// where `H_norm` is the normalized Shannon entropy of inter-arrival times
    /// (binned into equal-width buckets), and density is the current arrival rate.
    pub fn synchronicity_coefficient(&self) -> f64 {
        const MIN_EVENTS: usize = 5;
        const NUM_BINS: usize = 5;

        if self.inter_arrival_buffer.len() < MIN_EVENTS {
            return 0.0;
        }

        // Find range of inter-arrival times
        let mut min_dt = f64::MAX;
        let mut max_dt = f64::MIN;
        for &dt in &self.inter_arrival_buffer {
            if dt < min_dt {
                min_dt = dt;
            }
            if dt > max_dt {
                max_dt = dt;
            }
        }

        let range = max_dt - min_dt;
        if range < 1e-12 {
            // All inter-arrival times are identical — perfectly synchronized
            let density = 1.0 / min_dt.max(1e-9);
            let density_ratio = (density / self.baseline_density.max(1e-9)).min(1.0);
            return density_ratio;
        }

        // Bin inter-arrival times
        let bin_width = range / NUM_BINS as f64;
        let mut bins = [0u32; NUM_BINS];
        let n = self.inter_arrival_buffer.len() as f64;

        for &dt in &self.inter_arrival_buffer {
            let idx = ((dt - min_dt) / bin_width).floor() as usize;
            let idx = idx.min(NUM_BINS - 1);
            bins[idx] += 1;
        }

        // Normalized Shannon entropy
        let max_entropy = (NUM_BINS as f64).ln();
        let mut entropy = 0.0_f64;
        for &count in &bins {
            if count > 0 {
                let p = count as f64 / n;
                entropy -= p * p.ln();
            }
        }
        let normalized_entropy = if max_entropy > 1e-12 {
            (entropy / max_entropy).clamp(0.0, 1.0)
        } else {
            1.0
        };

        // Current density from most recent inter-arrival time
        let last_dt = *self.inter_arrival_buffer.back().unwrap();
        let current_density = 1.0 / last_dt.max(1e-9);
        let density_ratio = (current_density / self.baseline_density.max(1e-9)).min(1.0);

        // Synchronicity = low entropy × high density
        (1.0 - normalized_entropy) * density_ratio
    }

    /// Get current branching ratio (α/β).
    pub fn branching_ratio(&self) -> f64 {
        self.latest_calibration
            .map(|c| c.branching_ratio)
            .unwrap_or(0.3) // Conservative default
    }

    /// Get current intensity percentile.
    pub fn intensity_percentile(&self) -> f64 {
        self.latest_summary
            .as_ref()
            .map(|s| s.intensity_percentile)
            .unwrap_or(0.5)
    }

    /// Compute P(cluster in next τ seconds).
    ///
    /// Uses the branching ratio and current excitation level to estimate
    /// the probability of a clustering event (cascade) occurring.
    ///
    /// # Theory
    ///
    /// For a Hawkes process, the expected number of child events from one
    /// parent event is n = α/β. The excess intensity above baseline
    /// represents "potential parents" that could trigger clusters.
    ///
    /// P(cluster | excitation) ≈ 1 - (1 - n)^(excess_events × decay_factor)
    ///
    /// where:
    /// - excess_events = (λ_current - λ₀) × τ
    /// - decay_factor = (1 - e^(-β×τ)) / (β×τ) accounts for decay during horizon
    pub fn p_cluster(&self, tau_secs: f64) -> f64 {
        let n = self.branching_ratio();

        // Get current total intensity
        let lambda_current = self
            .latest_summary
            .as_ref()
            .map(|s| s.lambda_total)
            .unwrap_or(self.baseline_intensity);

        // Excess intensity above baseline
        let lambda_excess = (lambda_current - self.baseline_intensity).max(0.0);

        // If no excess or no calibration, low cluster probability
        if lambda_excess < 0.01 || n < 0.01 {
            return 0.0;
        }

        // Get beta from calibration (decay rate)
        let beta = self.latest_calibration.map(|c| c.beta).unwrap_or(0.1);

        // Expected excess events in horizon, accounting for decay
        // ∫₀^τ λ_excess × e^(-β×t) dt = λ_excess × (1 - e^(-β×τ)) / β
        let decay_integral = if beta > 0.001 {
            (1.0 - (-beta * tau_secs).exp()) / beta
        } else {
            tau_secs // Linear for very slow decay
        };

        let excess_events = lambda_excess * decay_integral;

        // P(at least one child) = 1 - (1-n)^excess_events
        // For small n, this ≈ n × excess_events
        // For n close to 1, this approaches 1 quickly
        let p_no_cluster = (1.0 - n).powf(excess_events);

        (1.0 - p_no_cluster).clamp(0.0, 1.0)
    }

    /// Compute excitation penalty for edge calculation.
    ///
    /// Returns a multiplier in [min_penalty, max_penalty] that reduces
    /// expected edge when cluster probability is high.
    ///
    /// # Formula
    ///
    /// ```text
    /// penalty = max_penalty - (max_penalty - min_penalty) × p_cluster^γ
    /// ```
    ///
    /// where γ = 0.5 (square root) to make the penalty responsive but not
    /// overly aggressive for moderate cluster probabilities.
    pub fn excitation_penalty(&self) -> f64 {
        let p_cluster = self.p_cluster(self.config.cluster_horizon_secs);

        let penalty_range = self.config.max_penalty - self.config.min_penalty;

        // Square root scaling: responsive but not too aggressive
        let reduction = penalty_range * p_cluster.sqrt();

        (self.config.max_penalty - reduction)
            .clamp(self.config.min_penalty, self.config.max_penalty)
    }

    /// Check if we're in a high excitation state.
    ///
    /// High excitation is defined as:
    /// - Branching ratio > threshold (default 0.7), OR
    /// - Intensity percentile > threshold (default 0.8)
    pub fn is_high_excitation(&self) -> bool {
        let n = self.branching_ratio();
        let intensity_pct = self.intensity_percentile();

        n > self.config.high_excitation_branching_ratio
            || intensity_pct > self.config.high_intensity_percentile
    }

    /// Compute spread widening factor.
    ///
    /// Returns a multiplier >= 1.0 to widen spreads during high excitation.
    ///
    /// # Formula
    ///
    /// Based on excess intensity and branching ratio:
    /// ```text
    /// widening = 1 + k₁ × (intensity_ratio - 1) + k₂ × n²
    /// ```
    ///
    /// where:
    /// - k₁ = 0.3 (intensity sensitivity)
    /// - k₂ = 0.5 (branching ratio sensitivity, squared for non-linearity)
    pub fn spread_widening_factor(&self) -> f64 {
        let lambda_current = self
            .latest_summary
            .as_ref()
            .map(|s| s.lambda_total)
            .unwrap_or(self.baseline_intensity);

        let intensity_ratio = lambda_current / self.baseline_intensity.max(0.01);
        let n = self.branching_ratio();

        // Widening components
        let intensity_component = 0.3 * (intensity_ratio - 1.0).max(0.0);
        let branching_component = 0.5 * n * n;

        (1.0 + intensity_component + branching_component).clamp(1.0, 3.0)
    }

    /// Expected time to next cluster event (seconds).
    ///
    /// Based on the intensity of "cluster-triggering" events and the
    /// conditional probability of cascade given a trigger.
    pub fn expected_cluster_time(&self) -> f64 {
        let p_cluster = self.p_cluster(self.config.cluster_horizon_secs);

        if p_cluster < 0.001 {
            return f64::INFINITY;
        }

        // Rough estimate: horizon / p_cluster (geometric distribution)
        self.config.cluster_horizon_secs / p_cluster
    }

    /// Get full prediction result.
    pub fn predict(&self) -> HawkesExcitationPrediction {
        let tau = self.config.cluster_horizon_secs;
        let p_cluster = self.p_cluster(tau);
        let penalty = self.excitation_penalty();
        let is_high = self.is_high_excitation();
        let n = self.branching_ratio();
        let intensity_pct = self.intensity_percentile();

        let lambda_current = self
            .latest_summary
            .as_ref()
            .map(|s| s.lambda_total)
            .unwrap_or(self.baseline_intensity);

        let excess_ratio = lambda_current / self.baseline_intensity.max(0.01);

        HawkesExcitationPrediction {
            p_cluster,
            excitation_penalty: penalty,
            is_high_excitation: is_high,
            branching_ratio: n,
            intensity_percentile: intensity_pct,
            excess_intensity_ratio: excess_ratio,
            expected_cluster_time_secs: self.expected_cluster_time(),
            spread_widening_factor: self.spread_widening_factor(),
        }
    }

    /// Diagnostic summary for logging.
    pub fn diagnostic_summary(&self) -> String {
        let pred = self.predict();
        format!(
            "Hawkes: p_cluster={:.3} penalty={:.3} n={:.3} intensity_pct={:.2} excess_ratio={:.2} widening={:.2}",
            pred.p_cluster,
            pred.excitation_penalty,
            pred.branching_ratio,
            pred.intensity_percentile,
            pred.excess_intensity_ratio,
            pred.spread_widening_factor,
        )
    }

    /// Reset predictor state.
    pub fn reset(&mut self) {
        self.latest_calibration = None;
        self.latest_summary = None;
        self.baseline_intensity = 0.5;
        self.ewma_intensity_percentile = 0.5;
        self.inter_arrival_buffer.clear();
        self.last_event_time_secs = None;
        self.baseline_density = 1.0;
    }
}

impl Default for HawkesExcitationPredictor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_default_config() {
        let config = HawkesConfig::default();
        assert_eq!(config.mu, 0.5);
        assert_eq!(config.alpha, 0.3);
    }

    #[test]
    fn test_new_estimator() {
        let estimator = HawkesOrderFlowEstimator::new(HawkesConfig::default());
        assert!(!estimator.is_warmed_up());
        assert!((estimator.lambda_buy() - 0.5).abs() < 1e-9); // At baseline
    }

    #[test]
    fn test_record_trade_buy() {
        let mut estimator = HawkesOrderFlowEstimator::new(HawkesConfig::default());

        estimator.record_trade(true, 1.0); // Buy trade

        // Buy intensity should increase
        assert!(estimator.lambda_buy() > 0.5);
        // Sell intensity should also increase (cross-excitation)
        assert!(estimator.lambda_sell() > 0.5);
        // But buy should increase more
        assert!(estimator.lambda_buy() > estimator.lambda_sell());
    }

    #[test]
    fn test_record_trade_sell() {
        let mut estimator = HawkesOrderFlowEstimator::new(HawkesConfig::default());

        estimator.record_trade(false, 1.0); // Sell trade

        // Sell intensity should increase more than buy
        assert!(estimator.lambda_sell() > estimator.lambda_buy());
    }

    #[test]
    fn test_flow_imbalance() {
        let mut estimator = HawkesOrderFlowEstimator::new(HawkesConfig::default());

        // Record several buy trades
        for _ in 0..5 {
            estimator.record_trade(true, 1.0);
        }

        // Should have positive imbalance (more buy intensity)
        assert!(estimator.flow_imbalance() > 0.0);
    }

    #[test]
    fn test_intensity_decay() {
        let config = HawkesConfig {
            beta: 10.0, // Very fast decay
            ..Default::default()
        };
        let mut estimator = HawkesOrderFlowEstimator::new(config);

        estimator.record_trade(true, 1.0);
        let initial = estimator.lambda_buy();

        // Wait a bit and update
        thread::sleep(Duration::from_millis(200));
        estimator.update();

        // Intensity should have decayed
        assert!(estimator.lambda_buy() < initial);
    }

    #[test]
    fn test_expected_trades() {
        let estimator = HawkesOrderFlowEstimator::new(HawkesConfig::default());

        let (buy, sell) = estimator.expected_trades(60.0);

        // At baseline (0.5 each), expect ~30 trades each in 60 seconds
        assert!((buy - 30.0).abs() < 5.0);
        assert!((sell - 30.0).abs() < 5.0);
    }

    #[test]
    fn test_warmup() {
        let config = HawkesConfig {
            min_trades: 5,
            ..Default::default()
        };
        let mut estimator = HawkesOrderFlowEstimator::new(config);

        assert!(!estimator.is_warmed_up());

        for i in 0..5 {
            estimator.record_trade(i % 2 == 0, 1.0);
        }

        assert!(estimator.is_warmed_up());
    }

    #[test]
    fn test_volume_tracking() {
        let mut estimator = HawkesOrderFlowEstimator::new(HawkesConfig::default());

        estimator.record_trade(true, 2.0);
        estimator.record_trade(false, 1.0);

        assert!((estimator.buy_volume - 2.0).abs() < 1e-9);
        assert!((estimator.sell_volume - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_position_sizing_factor() {
        let mut estimator = HawkesOrderFlowEstimator::new(HawkesConfig::default());

        // At baseline, should return 1.0
        let factor = estimator.position_sizing_factor();
        assert!((factor - 1.0).abs() < 0.01);

        // After many buy trades, should reduce
        for _ in 0..20 {
            estimator.record_trade(true, 1.0);
        }

        let factor_high = estimator.position_sizing_factor();
        assert!(factor_high < 1.0);
    }

    #[test]
    fn test_summary() {
        let mut estimator = HawkesOrderFlowEstimator::new(HawkesConfig::default());

        for i in 0..10 {
            estimator.record_trade(i % 2 == 0, 1.0);
        }

        let summary = estimator.summary();
        assert!(summary.is_warmed_up);
        assert_eq!(summary.trade_count, 10);
    }

    #[test]
    fn test_reset() {
        let mut estimator = HawkesOrderFlowEstimator::new(HawkesConfig::default());

        for i in 0..10 {
            estimator.record_trade(i % 2 == 0, 1.0);
        }

        estimator.reset();

        assert!(!estimator.is_warmed_up());
        assert_eq!(estimator.trade_count, 0);
        assert!(estimator.events.is_empty());
    }

    // ========================================================================
    // GMM Calibration Tests
    // ========================================================================

    #[test]
    fn test_gmm_calibrator_new() {
        let calibrator = HawkesGmmCalibrator::new(10.0, 30, 0.1);
        assert_eq!(calibrator.n_windows(), 0);
        assert!(!calibrator.has_sufficient_data());
    }

    #[test]
    fn test_gmm_calibrator_default() {
        let calibrator = HawkesGmmCalibrator::default_calibrator();
        assert_eq!(calibrator.window_size_secs, 10.0);
        assert_eq!(calibrator.min_windows, 30);
    }

    #[test]
    fn test_gmm_add_window_count() {
        let mut calibrator = HawkesGmmCalibrator::new(10.0, 5, 0.1);

        for i in 0..5 {
            calibrator.add_window_count(i + 1);
        }

        assert_eq!(calibrator.n_windows(), 5);
        assert!(calibrator.has_sufficient_data());
    }

    #[test]
    fn test_gmm_sample_statistics() {
        let mut calibrator = HawkesGmmCalibrator::new(10.0, 5, 0.1);

        // Add known counts: 2, 4, 6, 8, 10 -> mean = 6, variance = 10
        for &count in &[2, 4, 6, 8, 10] {
            calibrator.add_window_count(count);
        }

        let mean = calibrator.sample_mean();
        assert!((mean - 6.0).abs() < 0.01, "Mean should be 6, got {}", mean);

        let variance = calibrator.sample_variance();
        // Sample variance of [2,4,6,8,10] = 10
        assert!(
            (variance - 10.0).abs() < 0.01,
            "Variance should be 10, got {}",
            variance
        );
    }

    #[test]
    fn test_gmm_calibrate_poisson_like() {
        // If variance ≈ mean, branching ratio should be ≈ 0 (Poisson, no self-excitation)
        let mut calibrator = HawkesGmmCalibrator::new(1.0, 10, 0.1);

        // Add counts where variance ≈ mean (Poisson-like)
        // Poisson(5) has mean=5, var=5
        for &count in &[4, 6, 5, 3, 7, 5, 4, 6, 5, 5] {
            calibrator.add_window_count(count);
        }

        let result = calibrator.calibrate().unwrap();

        // For Poisson-like data, branching ratio should be near 0
        assert!(
            result.branching_ratio < 0.3,
            "Branching ratio {} should be small for Poisson-like data",
            result.branching_ratio
        );
        assert!(result.is_stationary());
    }

    #[test]
    fn test_gmm_calibrate_overdispersed() {
        // If variance >> mean, branching ratio should be significant (self-excitation)
        let mut calibrator = HawkesGmmCalibrator::new(1.0, 10, 0.1);

        // Add overdispersed counts (variance > mean)
        // Mean ≈ 5, but high variability
        for &count in &[1, 10, 2, 9, 1, 10, 2, 8, 1, 11] {
            calibrator.add_window_count(count);
        }

        let result = calibrator.calibrate().unwrap();

        // For overdispersed data, branching ratio should be positive
        assert!(
            result.branching_ratio > 0.1,
            "Branching ratio {} should be significant for overdispersed data",
            result.branching_ratio
        );
        assert!(result.is_stationary());
    }

    #[test]
    fn test_gmm_result_long_run_intensity() {
        let result = HawkesGmmResult {
            lambda_0: 0.5,
            alpha: 0.2,
            beta: 0.4,
            branching_ratio: 0.5, // alpha/beta = 0.2/0.4 = 0.5
            sample_mean: 10.0,
            sample_variance: 30.0,
            n_windows: 100,
            fit_quality: 1.0,
        };

        // Long-run intensity = λ₀ / (1 - r) = 0.5 / 0.5 = 1.0
        let long_run = result.long_run_intensity();
        assert!(
            (long_run - 1.0).abs() < 0.01,
            "Long-run intensity should be 1.0, got {}",
            long_run
        );
    }

    #[test]
    fn test_gmm_result_stationarity() {
        let stationary = HawkesGmmResult {
            lambda_0: 0.5,
            alpha: 0.08,
            beta: 0.1,
            branching_ratio: 0.8,
            sample_mean: 10.0,
            sample_variance: 30.0,
            n_windows: 100,
            fit_quality: 1.0,
        };
        assert!(stationary.is_stationary());

        let non_stationary = HawkesGmmResult {
            lambda_0: 0.5,
            alpha: 0.15,
            beta: 0.1,
            branching_ratio: 1.5,
            sample_mean: 10.0,
            sample_variance: 30.0,
            n_windows: 100,
            fit_quality: 1.0,
        };
        assert!(!non_stationary.is_stationary());
    }

    #[test]
    fn test_gmm_add_fill_history() {
        let mut calibrator = HawkesGmmCalibrator::new(1.0, 5, 0.1);

        // Add fills over 5 seconds
        let timestamps: Vec<f64> = vec![0.1, 0.5, 1.2, 1.8, 2.5, 3.1, 3.9, 4.2, 4.8];
        calibrator.add_fill_history(&timestamps);

        // Should have created ~4-5 windows
        assert!(calibrator.n_windows() >= 4);
    }

    #[test]
    fn test_gmm_calibrate_with_beta() {
        let mut calibrator = HawkesGmmCalibrator::new(1.0, 20, 0.1);

        // Add enough data for autocorrelation estimation
        for i in 0..100 {
            // Clustered counts with some autocorrelation
            let base = 5 + (i % 10);
            calibrator.add_window_count(base);
        }

        let result = calibrator.calibrate_with_beta();
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.beta > 0.0);
        assert!(r.is_stationary());
    }

    #[test]
    fn test_gmm_calibrator_reset() {
        let mut calibrator = HawkesGmmCalibrator::default_calibrator();

        for i in 0..50 {
            calibrator.add_window_count(i);
        }
        assert!(calibrator.has_sufficient_data());

        calibrator.reset();
        assert!(!calibrator.has_sufficient_data());
        assert_eq!(calibrator.n_windows(), 0);
    }

    #[test]
    fn test_online_calibrator_new() {
        let calibrator = OnlineHawkesCalibrator::new(10.0, 20.0, 0.1);
        assert_eq!(calibrator.n_windows, 0);
        assert!(calibrator.estimate().is_none());
    }

    #[test]
    fn test_online_calibrator_update() {
        let mut calibrator = OnlineHawkesCalibrator::new(1.0, 10.0, 0.1);

        for i in 0..20 {
            calibrator.update(5 + (i % 3));
        }

        let result = calibrator.estimate();
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.lambda_0 > 0.0);
        assert!(r.is_stationary());
    }

    #[test]
    fn test_online_calibrator_reset() {
        let mut calibrator = OnlineHawkesCalibrator::new(1.0, 10.0, 0.1);

        for i in 0..20 {
            calibrator.update(i);
        }
        assert!(calibrator.estimate().is_some());

        calibrator.reset();
        assert!(calibrator.estimate().is_none());
    }

    #[test]
    fn test_gmm_moment_equations() {
        // Verify the moment equations are implemented correctly:
        // E[N(T)] = λ₀ × T / (1 - α/β)
        // Var[N(T)] = E[N(T)] × (1 + 2α/(β - α))
        //
        // Equivalently with r = α/β:
        // Var/Mean = (1 + r)/(1 - r)
        // So r = (Var - Mean)/(Var + Mean)

        let mean: f64 = 10.0;
        let variance: f64 = 30.0;

        // Expected r = (30 - 10)/(30 + 10) = 20/40 = 0.5
        let expected_r: f64 = (variance - mean) / (variance + mean);
        assert!(
            (expected_r - 0.5).abs() < 0.01,
            "Expected r=0.5, got {}",
            expected_r
        );

        // Verify: Var/Mean = (1 + 0.5)/(1 - 0.5) = 1.5/0.5 = 3.0
        let var_to_mean: f64 = variance / mean;
        let theoretical: f64 = (1.0 + expected_r) / (1.0 - expected_r);
        assert!(
            (var_to_mean - theoretical).abs() < 0.01,
            "Var/Mean={} should equal theoretical={}",
            var_to_mean,
            theoretical
        );
    }

    #[test]
    fn test_gmm_result_well_fit() {
        let good_fit = HawkesGmmResult {
            lambda_0: 0.5,
            alpha: 0.05,
            beta: 0.1,
            branching_ratio: 0.5,
            sample_mean: 10.0,
            sample_variance: 30.0,
            n_windows: 100,
            fit_quality: 0.95, // Close to 1.0
        };
        assert!(good_fit.is_well_fit());

        let bad_fit = HawkesGmmResult {
            lambda_0: 0.5,
            alpha: 0.05,
            beta: 0.1,
            branching_ratio: 0.5,
            sample_mean: 10.0,
            sample_variance: 30.0,
            n_windows: 100,
            fit_quality: 0.1, // Far from 1.0
        };
        assert!(!bad_fit.is_well_fit());
    }

    #[test]
    fn test_gmm_low_activity() {
        // Test behavior with very low activity (should still work)
        let mut calibrator = HawkesGmmCalibrator::new(10.0, 10, 0.1);

        for _ in 0..20 {
            calibrator.add_window_count(0); // All zeros
        }

        let result = calibrator.calibrate();
        assert!(result.is_some());
        let r = result.unwrap();
        // Should return minimal lambda_0 and no excitation
        assert!(r.lambda_0 >= 0.0);
        assert!(r.alpha >= 0.0);
    }

    // ============================================================================
    // HawkesExcitationPredictor Tests
    // ============================================================================

    #[test]
    fn test_excitation_predictor_new() {
        let predictor = HawkesExcitationPredictor::new();
        assert!(predictor.branching_ratio() >= 0.0);
        assert!(predictor.intensity_percentile() >= 0.0);
    }

    #[test]
    fn test_excitation_predictor_default_prediction() {
        let predictor = HawkesExcitationPredictor::new();
        let pred = predictor.predict();

        // Without calibration, should return safe defaults
        assert!(pred.p_cluster >= 0.0 && pred.p_cluster <= 1.0);
        assert!(pred.excitation_penalty >= 0.5 && pred.excitation_penalty <= 1.0);
        assert!(!pred.is_high_excitation);
    }

    #[test]
    fn test_excitation_predictor_with_calibration() {
        let mut predictor = HawkesExcitationPredictor::new();

        // Add a calibration with moderate branching ratio
        let calibration = HawkesGmmResult {
            lambda_0: 0.5,
            alpha: 0.03,
            beta: 0.1,
            branching_ratio: 0.3,
            sample_mean: 5.0,
            sample_variance: 8.0,
            n_windows: 100,
            fit_quality: 1.0,
        };
        predictor.update_calibration(calibration);

        assert!((predictor.branching_ratio() - 0.3).abs() < 1e-6);
        assert!(!predictor.is_high_excitation()); // 0.3 < 0.7 threshold
    }

    #[test]
    fn test_excitation_predictor_high_branching_ratio() {
        let mut predictor = HawkesExcitationPredictor::new();

        // Add a calibration with high branching ratio (near critical)
        let calibration = HawkesGmmResult {
            lambda_0: 0.5,
            alpha: 0.08,
            beta: 0.1,
            branching_ratio: 0.8, // > 0.7 threshold
            sample_mean: 5.0,
            sample_variance: 20.0,
            n_windows: 100,
            fit_quality: 1.0,
        };
        predictor.update_calibration(calibration);

        assert!(predictor.is_high_excitation());

        // Add summary with elevated intensity
        let summary = HawkesSummary {
            is_warmed_up: true,
            lambda_buy: 1.0,
            lambda_sell: 1.0,
            lambda_total: 2.0, // 4x baseline
            flow_imbalance: 0.0,
            intensity_percentile: 0.9,
            trade_count: 100,
            recent_events: 50,
            expected_trades_1m: 120.0,
            buy_volume: 100.0,
            sell_volume: 100.0,
        };
        predictor.update_summary(summary);

        let pred = predictor.predict();

        // High branching ratio + elevated intensity = high cluster probability
        assert!(pred.p_cluster > 0.5);
        // Should reduce edge penalty
        assert!(pred.excitation_penalty < 0.8);
        // Should widen spreads
        assert!(pred.spread_widening_factor > 1.0);
    }

    #[test]
    fn test_excitation_predictor_p_cluster_bounds() {
        let mut predictor = HawkesExcitationPredictor::new();

        // Even with extreme parameters, p_cluster should be bounded
        let calibration = HawkesGmmResult {
            lambda_0: 0.5,
            alpha: 0.095,
            beta: 0.1,
            branching_ratio: 0.95, // Very close to critical
            sample_mean: 5.0,
            sample_variance: 50.0,
            n_windows: 100,
            fit_quality: 1.0,
        };
        predictor.update_calibration(calibration);

        let summary = HawkesSummary {
            is_warmed_up: true,
            lambda_buy: 5.0,
            lambda_sell: 5.0,
            lambda_total: 10.0, // 20x baseline
            flow_imbalance: 0.0,
            intensity_percentile: 0.99,
            trade_count: 500,
            recent_events: 100,
            expected_trades_1m: 600.0,
            buy_volume: 500.0,
            sell_volume: 500.0,
        };
        predictor.update_summary(summary);

        let p = predictor.p_cluster(10.0);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_excitation_penalty_range() {
        let mut predictor = HawkesExcitationPredictor::new();

        // Test with various calibrations
        for branching in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let calibration = HawkesGmmResult {
                lambda_0: 0.5,
                alpha: branching * 0.1,
                beta: 0.1,
                branching_ratio: branching,
                sample_mean: 5.0,
                sample_variance: 5.0 * (1.0 + branching) / (1.0 - branching).max(0.01),
                n_windows: 100,
                fit_quality: 1.0,
            };
            predictor.update_calibration(calibration);

            let penalty = predictor.excitation_penalty();
            assert!(
                penalty >= 0.5 && penalty <= 1.0,
                "Penalty {} out of range for branching {}",
                penalty,
                branching
            );
        }
    }

    #[test]
    fn test_spread_widening_factor() {
        let mut predictor = HawkesExcitationPredictor::new();

        // Calm conditions
        let calibration = HawkesGmmResult {
            lambda_0: 0.5,
            alpha: 0.02,
            beta: 0.1,
            branching_ratio: 0.2,
            sample_mean: 5.0,
            sample_variance: 6.0,
            n_windows: 100,
            fit_quality: 1.0,
        };
        predictor.update_calibration(calibration);

        let summary = HawkesSummary {
            is_warmed_up: true,
            lambda_buy: 0.25,
            lambda_sell: 0.25,
            lambda_total: 0.5, // At baseline
            flow_imbalance: 0.0,
            intensity_percentile: 0.5,
            trade_count: 50,
            recent_events: 20,
            expected_trades_1m: 30.0,
            buy_volume: 50.0,
            sell_volume: 50.0,
        };
        predictor.update_summary(summary);

        let widening = predictor.spread_widening_factor();
        // At baseline intensity with low branching, widening should be near 1
        assert!(widening >= 1.0 && widening < 1.3);
    }

    #[test]
    fn test_diagnostic_summary() {
        let predictor = HawkesExcitationPredictor::new();
        let summary = predictor.diagnostic_summary();

        // Should contain key metrics
        assert!(summary.contains("p_cluster="));
        assert!(summary.contains("penalty="));
        assert!(summary.contains("n="));
    }

    #[test]
    fn test_predictor_reset() {
        let mut predictor = HawkesExcitationPredictor::new();

        // Add calibration
        let calibration = HawkesGmmResult {
            lambda_0: 1.0,
            alpha: 0.05,
            beta: 0.1,
            branching_ratio: 0.5,
            sample_mean: 10.0,
            sample_variance: 30.0,
            n_windows: 100,
            fit_quality: 1.0,
        };
        predictor.update_calibration(calibration);

        // Verify it took effect
        assert!((predictor.branching_ratio() - 0.5).abs() < 1e-6);

        // Reset
        predictor.reset();

        // Should be back to defaults
        assert!((predictor.branching_ratio() - 0.3).abs() < 1e-6); // Default is 0.3
    }

    // ============================================================================
    // Synchronicity Coefficient Tests
    // ============================================================================

    #[test]
    fn test_synchronicity_cold_start() {
        let predictor = HawkesExcitationPredictor::new();
        // No events recorded — should return 0
        assert!((predictor.synchronicity_coefficient() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_synchronicity_too_few_events() {
        let mut predictor = HawkesExcitationPredictor::new();
        // Record only 3 events (below MIN_EVENTS = 5)
        for i in 0..3 {
            predictor.record_event_time(i as f64 * 0.1);
        }
        assert!((predictor.synchronicity_coefficient() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_synchronicity_uniform_arrivals() {
        let mut predictor = HawkesExcitationPredictor::new();
        // Perfectly uniform inter-arrival times (all 0.01s apart)
        // → zero entropy → high synchronicity (modulated by density ratio)
        for i in 0..15 {
            predictor.record_event_time(i as f64 * 0.01);
        }
        let sync = predictor.synchronicity_coefficient();
        // All inter-arrivals identical → range ≈ 0 → fast path → density_ratio
        // density = 1/0.01 = 100, baseline starts at 1.0 and converges
        // density_ratio capped at 1.0
        assert!(
            sync > 0.5,
            "Uniform arrivals should have high synchronicity, got {}",
            sync
        );
    }

    #[test]
    fn test_synchronicity_random_arrivals() {
        let mut predictor = HawkesExcitationPredictor::new();
        // Highly varied inter-arrival times → high entropy → low synchronicity
        // Use a manually constructed "random-like" sequence
        let timestamps = [
            0.0, 0.5, 0.52, 1.8, 1.81, 3.5, 3.9, 4.0, 7.0, 7.1, 7.3, 9.0, 12.0, 12.5, 15.0,
        ];
        for &t in &timestamps {
            predictor.record_event_time(t);
        }
        let sync = predictor.synchronicity_coefficient();
        // With varied spacing, entropy should be relatively high → lower sync
        assert!(
            sync < 0.8,
            "Random-like arrivals should have lower synchronicity, got {}",
            sync
        );
    }

    #[test]
    fn test_synchronicity_bounded_zero_one() {
        let mut predictor = HawkesExcitationPredictor::new();
        // Record a mix of events
        for i in 0..20 {
            let t = i as f64 * 0.05 + if i % 3 == 0 { 0.2 } else { 0.0 };
            predictor.record_event_time(t);
        }
        let sync = predictor.synchronicity_coefficient();
        assert!(
            sync >= 0.0 && sync <= 1.0,
            "Sync must be in [0,1], got {}",
            sync
        );
    }

    #[test]
    fn test_synchronicity_reset_clears() {
        let mut predictor = HawkesExcitationPredictor::new();
        for i in 0..10 {
            predictor.record_event_time(i as f64 * 0.01);
        }
        assert!(predictor.synchronicity_coefficient() > 0.0);

        predictor.reset();
        assert!((predictor.synchronicity_coefficient() - 0.0).abs() < 1e-9);
        assert!(predictor.inter_arrival_buffer.is_empty());
        assert!(predictor.last_event_time_secs.is_none());
    }

    // ============================================================================
    // Fix 3: Hawkes Autocorrelation Stabilization Tests
    // ============================================================================

    #[test]
    fn test_online_calibrator_constant_count_bounded_autocorr() {
        // Fix 3A/B: Feed constant counts (zero variance in x).
        // With the old code, variance could drop near zero → autocorr spikes.
        // With slow alpha for variance + bounded autocorr, this stays stable.
        let mut calibrator = OnlineHawkesCalibrator::new(1.0, 10.0, 0.1);

        for _ in 0..30 {
            calibrator.update(5); // All same count
        }

        // Autocorrelation should stay bounded in [-1, 1]
        assert!(
            calibrator.ewma_autocorr >= -1.0 && calibrator.ewma_autocorr <= 1.0,
            "Autocorr must be bounded: got {}",
            calibrator.ewma_autocorr
        );

        // Should still produce a valid estimate
        let result = calibrator.estimate();
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.beta.is_finite(), "Beta must be finite, got {}", r.beta);
    }

    #[test]
    fn test_online_calibrator_antibunching_beta() {
        // Fix 3C: Alternating high/low counts → negative autocorrelation.
        // Beta should use 2x prior (faster decay) rather than NaN.
        let mut calibrator = OnlineHawkesCalibrator::new(1.0, 10.0, 0.1);

        for i in 0..40 {
            let count = if i % 2 == 0 { 10 } else { 1 };
            calibrator.update(count);
        }

        let result = calibrator.estimate();
        assert!(result.is_some());
        let r = result.unwrap();
        // With anti-bunching, beta should be >= beta_prior (faster decay)
        assert!(
            r.beta >= calibrator.beta_prior,
            "Anti-bunching should give beta >= prior: beta={}, prior={}",
            r.beta,
            calibrator.beta_prior
        );
        assert!(r.beta.is_finite(), "Beta must not be NaN: {}", r.beta);
    }

    #[test]
    fn test_online_calibrator_normal_clustering_beta() {
        // Fix 3C: Correlated counts → positive autocorrelation → sensible beta.
        let mut calibrator = OnlineHawkesCalibrator::new(1.0, 10.0, 0.1);

        // Simulate correlated counts (clustering)
        for i in 0..50 {
            // Gradually ramp up then down to create autocorrelation
            let count = 5 + (i / 5) % 5;
            calibrator.update(count);
        }

        let result = calibrator.estimate();
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.beta > 0.0, "Beta should be positive: {}", r.beta);
        assert!(r.beta.is_finite(), "Beta must be finite: {}", r.beta);
    }

    #[test]
    fn test_online_calibrator_slow_variance_alpha() {
        // Fix 3A: Verify the variance uses a slower alpha than the mean.
        let calibrator = OnlineHawkesCalibrator::new(1.0, 10.0, 0.1);
        assert!(
            calibrator.ewma_alpha_var < calibrator.ewma_alpha,
            "Variance alpha ({}) should be slower than mean alpha ({})",
            calibrator.ewma_alpha_var,
            calibrator.ewma_alpha
        );
    }
}
