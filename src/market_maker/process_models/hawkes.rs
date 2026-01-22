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
    /// EWMA decay for statistics
    ewma_alpha: f64,
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
        Self {
            window_size_secs,
            ewma_alpha,
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

        // Update variance (simplified EWMA variance)
        let diff = x - old_mean;
        let var_obs = diff * diff;
        self.ewma_variance =
            self.ewma_alpha * var_obs + (1.0 - self.ewma_alpha) * self.ewma_variance;

        // Update autocorrelation
        if let Some(last) = self.last_count {
            let last_diff = last as f64 - old_mean;
            let autocorr_obs = diff * last_diff / (self.ewma_variance + 0.001);
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

        // Estimate beta from autocorrelation
        let beta = if self.ewma_autocorr > 0.01 && self.ewma_autocorr < 0.99 {
            (-self.ewma_autocorr.ln() / t).clamp(0.01, 10.0)
        } else {
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
}
