//! Microprice estimation from book/flow imbalance signals.
//!
//! Uses rolling online regression to learn how imbalance predicts returns,
//! replacing magic number adjustments with data-driven coefficients.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, warn};

/// Sentinel value for "no EMA value" (NaN bits)
const EMA_NONE: u64 = u64::MAX;

// ============================================================================
// Microprice Observation
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

// ============================================================================
// Correlation Mode
// ============================================================================

/// Mode for handling correlation between book and flow signals.
///
/// When signals are highly correlated (common in thin markets), we switch
/// from two-variable regression to more robust alternatives.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
enum CorrelationMode {
    /// Use both signals independently (correlation < 0.80)
    #[default]
    Independent,
    /// Orthogonalize flow onto book (0.80 <= correlation < 0.95)
    Orthogonalized,
    /// Use combined net_pressure signal (correlation >= 0.95)
    Combined,
}

// ============================================================================
// Microprice Estimator
// ============================================================================

/// Estimates microprice by learning how book/flow imbalance predict returns.
///
/// Uses rolling online regression to estimate:
/// E[r_{t+Δ}] = β_book × book_imbalance + β_flow × flow_imbalance
///
/// microprice = mid × (1 + β_book × book_imb + β_flow × flow_imb)
///
/// This replaces magic number adjustments with data-driven coefficients.
///
/// Microprice estimator with Ridge regularization.
///
/// Uses online OLS with L2 regularization to learn coefficients for:
/// microprice = mid × (1 + β_book × book_imb + β_flow × flow_imb)
///
/// **Regularization benefits:**
/// - Prevents coefficient explosion when signals are collinear
/// - Biases coefficients toward zero when data is noisy
/// - Produces more stable estimates across market regimes
///
/// **Adaptive horizon:**
/// - Horizon adapts to arrival intensity: horizon_ms = 2000 / arrival_intensity
/// - Fast markets (high intensity): shorter horizon (min 100ms)
/// - Slow markets (low intensity): longer horizon (max 500ms)
#[derive(Debug)]
pub(crate) struct MicropriceEstimator {
    /// Pending observations waiting for forward horizon to elapse
    pending: VecDeque<MicropriceObservation>,
    /// Window for regression data (ms)
    window_ms: u64,
    /// Forward horizon to measure realized return (ms) - now adaptive
    forward_horizon_ms: u64,
    /// Default forward horizon (used when arrival intensity unknown)
    default_forward_horizon_ms: u64,
    /// Minimum forward horizon (ms)
    min_horizon_ms: u64,
    /// Maximum forward horizon (ms)
    max_horizon_ms: u64,

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

    /// Mode for handling correlated signals
    correlation_mode: CorrelationMode,

    /// Combined signal coefficient (for high-correlation mode)
    beta_net: f64,

    /// Sum statistics for net_pressure signal (book - flow)
    sum_x_net: f64,
    sum_xx_net: f64,
    sum_xy_net: f64,

    // EMA smoothing for microprice output
    /// EMA smoothing factor (0.0-1.0). Higher = more weight to new observation.
    /// 0.2 = 5-update half-life, 0.1 = 10-update half-life
    ema_alpha: f64,
    /// Current EMA smoothed microprice as bits (uses AtomicU64 for Sync-safe interior mutability)
    /// EMA_NONE (u64::MAX) represents "no value yet"
    ema_microprice_bits: AtomicU64,
    /// Minimum change in bps to update EMA (noise filter)
    ema_min_change_bps: f64,
}

impl MicropriceEstimator {
    pub(crate) fn new(window_ms: u64, forward_horizon_ms: u64, min_observations: usize) -> Self {
        Self {
            pending: VecDeque::with_capacity(2000),
            window_ms,
            forward_horizon_ms,
            default_forward_horizon_ms: forward_horizon_ms,
            min_horizon_ms: 100, // Minimum 100ms for fast markets
            max_horizon_ms: 500, // Maximum 500ms for slow markets
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
            // Ridge regularization: λ = 0.001 (reduced from 0.01)
            // Lower regularization allows coefficients to learn from sparse data
            // This adds λI to X'X, shrinking coefficients toward zero
            lambda: 0.001,
            // Minimum R² threshold: 0.01% explained variance (reduced from 1%)
            // Lower threshold allows microprice to deviate even with weak signals
            min_r_squared: 0.0001,
            // Initialize correlation to 0 (no correlation assumed)
            signal_correlation: 0.0,
            // Start in Independent mode
            correlation_mode: CorrelationMode::Independent,
            // Combined signal coefficient
            beta_net: 0.0,
            // Net pressure statistics (book - flow)
            sum_x_net: 0.0,
            sum_xx_net: 0.0,
            sum_xy_net: 0.0,
            // EMA smoothing (defaults - can be overridden via set_ema_config)
            ema_alpha: 0.2,                                // 5-update half-life
            ema_microprice_bits: AtomicU64::new(EMA_NONE), // No smoothed value until first update
            ema_min_change_bps: 2.0,                       // 2 bps noise filter
        }
    }

    /// Configure EMA smoothing parameters.
    ///
    /// - `alpha`: Smoothing factor (0.0-1.0). Higher = more weight to new observation.
    ///   0.2 = 5-update half-life, 0.1 = 10-update half-life, 0.0 = disabled
    /// - `min_change_bps`: Minimum change in bps to update EMA (noise filter)
    pub(crate) fn set_ema_config(&mut self, alpha: f64, min_change_bps: f64) {
        self.ema_alpha = alpha.clamp(0.0, 1.0);
        self.ema_min_change_bps = min_change_bps.max(0.0);
    }

    /// Update forward horizon based on arrival intensity.
    ///
    /// Adapts the prediction horizon to market activity:
    /// - Fast markets (high intensity): shorter horizon for quicker updates
    /// - Slow markets (low intensity): longer horizon for more stable predictions
    ///
    /// Formula: horizon_ms = 2000 / arrival_intensity
    /// Clamped to [min_horizon_ms, max_horizon_ms]
    pub(crate) fn update_horizon(&mut self, arrival_intensity: f64) {
        if arrival_intensity <= 0.0 {
            self.forward_horizon_ms = self.default_forward_horizon_ms;
            return;
        }

        // Formula: horizon_ms = 2000 / arrival_intensity
        // At intensity 4 ticks/sec: horizon = 500ms
        // At intensity 10 ticks/sec: horizon = 200ms
        // At intensity 20 ticks/sec: horizon = 100ms
        let computed = (2000.0 / arrival_intensity) as u64;
        self.forward_horizon_ms = computed.clamp(self.min_horizon_ms, self.max_horizon_ms);
    }

    /// Get current forward horizon in ms.
    #[allow(dead_code)]
    pub(crate) fn forward_horizon_ms(&self) -> u64 {
        self.forward_horizon_ms
    }

    /// Update with new book state.
    pub(crate) fn on_book_update(
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

        // Track net_pressure = book - flow for combined mode
        let x_net = x_book - x_flow;
        self.sum_x_net += x_net;
        self.sum_xx_net += x_net * x_net;
        self.sum_xy_net += x_net * y;

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
            // Decay net_pressure stats too
            self.sum_x_net *= decay;
            self.sum_xx_net *= decay;
            self.sum_xy_net *= decay;
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

        // Determine correlation mode based on signal correlation
        let abs_corr = self.signal_correlation.abs();
        self.correlation_mode = if abs_corr >= 0.95 {
            CorrelationMode::Combined
        } else if abs_corr >= 0.80 {
            CorrelationMode::Orthogonalized
        } else {
            CorrelationMode::Independent
        };

        match self.correlation_mode {
            CorrelationMode::Combined => {
                // Single-variable regression on net_pressure = book - flow
                // When correlation is extreme, the two signals collapse into one dimension
                let mean_x_net = self.sum_x_net / n;
                let sxx_net = self.sum_xx_net - n * mean_x_net * mean_x_net;
                let sxy_net = self.sum_xy_net - n * mean_x_net * mean_y;

                // Ridge regularization to prevent overfitting in sparse data
                // Scale lambda by variance for scale-invariance
                let lambda_scaled = self.lambda * sxx_net.max(1e-6);
                let sxx_net_reg = sxx_net + lambda_scaled;

                if sxx_net_reg > 1e-9 {
                    // Regularized OLS estimate
                    let beta_raw = sxy_net / sxx_net_reg;

                    // Tight clamp: ±10 bps max coefficient
                    // net_pressure ranges [-2, +2], so max adjustment is ±20 bps
                    // This is economically reasonable for microprice
                    let beta_clamped = beta_raw.clamp(-0.001, 0.001);

                    // Sample-size based confidence: shrink toward 0 when n is small
                    // Full confidence at n = min_observations + 200
                    let confidence = ((n - self.min_observations as f64) / 200.0).clamp(0.0, 1.0);
                    self.beta_net = beta_clamped * confidence;

                    // R² calculation (use regularized estimate)
                    if syy > 1e-12 {
                        let y_pred_var = self.beta_net * self.beta_net * sxx_net;
                        self.r_squared = (y_pred_var / syy).clamp(0.0, 1.0);
                    }
                }

                // Log periodically
                if self.n.is_multiple_of(100) {
                    let confidence = ((n - self.min_observations as f64) / 200.0).clamp(0.0, 1.0);
                    debug!(
                        n = self.n,
                        beta_net_bps = %format!("{:.2}", self.beta_net * 10000.0),
                        r_squared = %format!("{:.4}", self.r_squared),
                        correlation = %format!("{:.3}", self.signal_correlation),
                        confidence = %format!("{:.2}", confidence),
                        mode = "Combined",
                        "Microprice using net_pressure signal"
                    );
                }
                return;
            }
            CorrelationMode::Orthogonalized => {
                // Project flow onto orthogonal space of book
                // flow_residual = flow - proj_coef * book
                let proj_coef = if sxx_book > 1e-9 {
                    sx_cross / sxx_book
                } else {
                    0.0
                };

                // Residual variance and covariance
                let sxx_flow_ortho = sxx_flow - proj_coef * proj_coef * sxx_book;
                let sxy_flow_ortho = sxy_flow - proj_coef * sxy_book;

                // Ridge regularization for book regression
                let lambda_book = self.lambda * sxx_book.max(1e-6);
                let sxx_book_reg = sxx_book + lambda_book;

                // Regress on book first with regularization
                if sxx_book_reg > 1e-9 {
                    // Tight clamp: ±10 bps
                    self.beta_book = (sxy_book / sxx_book_reg).clamp(-0.001, 0.001);
                }

                // Ridge regularization for orthogonalized flow
                let lambda_flow = self.lambda * sxx_flow_ortho.max(1e-6);
                let sxx_flow_ortho_reg = sxx_flow_ortho + lambda_flow;

                // Regress on orthogonalized flow with regularization
                if sxx_flow_ortho_reg > 1e-9 {
                    // Tight clamp: ±10 bps
                    let beta_flow_ortho =
                        (sxy_flow_ortho / sxx_flow_ortho_reg).clamp(-0.001, 0.001);
                    // Transform back: y = beta_book*book + beta_flow_ortho*(flow - proj*book)
                    // y = (beta_book - beta_flow_ortho*proj)*book + beta_flow_ortho*flow
                    self.beta_flow = beta_flow_ortho;
                    self.beta_book -= beta_flow_ortho * proj_coef;
                }

                // Sample-size based confidence scaling
                let confidence = ((n - self.min_observations as f64) / 200.0).clamp(0.0, 1.0);
                self.beta_book *= confidence;
                self.beta_flow *= confidence;

                // Final clamp after transformation (transformation can amplify)
                self.beta_book = self.beta_book.clamp(-0.001, 0.001);
                self.beta_flow = self.beta_flow.clamp(-0.001, 0.001);

                // Calculate R²
                if syy > 1e-12 {
                    let y_pred_var = self.beta_book.powi(2) * sxx_book
                        + self.beta_flow.powi(2) * sxx_flow
                        + 2.0 * self.beta_book * self.beta_flow * sx_cross;
                    self.r_squared = (y_pred_var / syy).clamp(0.0, 1.0);
                }

                // Log periodically
                if self.n.is_multiple_of(100) {
                    debug!(
                        n = self.n,
                        beta_book_bps = %format!("{:.2}", self.beta_book * 10000.0),
                        beta_flow_bps = %format!("{:.2}", self.beta_flow * 10000.0),
                        r_squared = %format!("{:.4}", self.r_squared),
                        correlation = %format!("{:.3}", self.signal_correlation),
                        confidence = %format!("{:.2}", confidence),
                        mode = "Orthogonalized",
                        "Microprice coefficients updated"
                    );
                }
                return;
            }
            CorrelationMode::Independent => {
                // Continue with standard ridge regression below
            }
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
        let beta_book_raw = (sxy_book * sxx_flow_reg - sxy_flow * sx_cross) / det;
        let beta_flow_raw = (sxy_flow * sxx_book_reg - sxy_book * sx_cross) / det;

        // Tight clamp: ±10 bps max coefficient (prevents overfitting)
        let beta_book_clamped = beta_book_raw.clamp(-0.001, 0.001);
        let beta_flow_clamped = beta_flow_raw.clamp(-0.001, 0.001);

        // Sample-size based confidence: shrink toward 0 when n is small
        // Full confidence after 200+ observations beyond minimum
        let confidence = ((n - self.min_observations as f64) / 200.0).clamp(0.0, 1.0);
        self.beta_book = beta_book_clamped * confidence;
        self.beta_flow = beta_flow_clamped * confidence;

        // Calculate R² = 1 - SSE/SST
        if syy > 1e-12 {
            let y_pred_var = self.beta_book * self.beta_book * sxx_book
                + self.beta_flow * self.beta_flow * sxx_flow
                + 2.0 * self.beta_book * self.beta_flow * sx_cross;
            self.r_squared = (y_pred_var / syy).clamp(0.0, 1.0);
        }

        // Log periodically (every 100 observations for better visibility)
        if self.n.is_multiple_of(100) {
            debug!(
                n = self.n,
                beta_book_bps = %format!("{:.2}", self.beta_book * 10000.0),
                beta_flow_bps = %format!("{:.2}", self.beta_flow * 10000.0),
                r_squared = %format!("{:.4}", self.r_squared),
                correlation = %format!("{:.3}", self.signal_correlation),
                mode = "Independent",
                "Microprice coefficients updated"
            );
        }
    }

    /// Get microprice adjusted for current signals with optional EMA smoothing.
    ///
    /// Returns mid price if:
    /// - Not warmed up (insufficient data)
    /// - R² below threshold (model has no predictive power)
    ///
    /// Uses mode-based adjustment depending on signal correlation:
    /// - Combined: uses net_pressure = book - flow when correlation >= 0.95
    /// - Orthogonalized/Independent: uses both signals
    ///
    /// EMA smoothing (if ema_alpha > 0):
    /// - Applies exponential moving average to smooth microprice output
    /// - Noise filter: skips EMA update if change < ema_min_change_bps
    pub(crate) fn microprice(&self, mid: f64, book_imbalance: f64, flow_imbalance: f64) -> f64 {
        if !self.is_warmed_up() {
            return mid;
        }

        // If R² is too low, the model has no predictive power - use mid
        if self.r_squared < self.min_r_squared {
            return mid;
        }

        // Mode-based adjustment (correlation handling is now built into mode selection)
        let adjustment = match self.correlation_mode {
            CorrelationMode::Combined => {
                // Use net_pressure signal when correlation is extreme
                let net_pressure = book_imbalance - flow_imbalance;
                self.beta_net * net_pressure
            }
            CorrelationMode::Orthogonalized | CorrelationMode::Independent => {
                // Standard two-signal adjustment
                self.beta_book * book_imbalance + self.beta_flow * flow_imbalance
            }
        };

        // Clamp adjustment to ±50 bps for safety
        let adjustment_clamped = adjustment.clamp(-0.005, 0.005);

        let raw_microprice = mid * (1.0 + adjustment_clamped);

        // Apply EMA smoothing if enabled (alpha > 0)
        if self.ema_alpha > 0.0 {
            let prev_bits = self.ema_microprice_bits.load(Ordering::Relaxed);

            if prev_bits == EMA_NONE {
                // First observation - initialize EMA with mid (not raw_microprice)
                // This prevents capturing early outliers during warmup
                self.ema_microprice_bits
                    .store(mid.to_bits(), Ordering::Relaxed);
                raw_microprice
            } else {
                let prev = f64::from_bits(prev_bits);

                // Safety check: if EMA diverged too far from mid, reset it
                // This handles cases where market moved significantly while EMA was stale
                let ema_divergence_bps = ((prev - mid) / mid).abs() * 10_000.0;
                if ema_divergence_bps > 100.0 {
                    // EMA is >100 bps from mid - reset to mid
                    tracing::warn!(
                        prev_ema = %format!("{:.2}", prev),
                        mid = %format!("{:.2}", mid),
                        divergence_bps = %format!("{:.1}", ema_divergence_bps),
                        "Microprice EMA diverged too far from mid - resetting"
                    );
                    self.ema_microprice_bits
                        .store(mid.to_bits(), Ordering::Relaxed);
                    return raw_microprice;
                }

                // Calculate change in bps
                let change_bps = ((raw_microprice - prev) / prev).abs() * 10_000.0;

                // Noise filter: skip update if change is too small
                if change_bps < self.ema_min_change_bps {
                    return prev;
                }

                // EMA update: new = alpha * raw + (1 - alpha) * prev
                let smoothed = self.ema_alpha * raw_microprice + (1.0 - self.ema_alpha) * prev;
                self.ema_microprice_bits
                    .store(smoothed.to_bits(), Ordering::Relaxed);
                smoothed
            }
        } else {
            // No smoothing - return raw microprice
            raw_microprice
        }
    }

    /// Get momentum-aware microprice with dynamic adjustment clamps.
    ///
    /// During strong momentum regimes with high continuation probability,
    /// the microprice adjustment clamp is widened to allow larger shifts
    /// in the fair price estimate.
    ///
    /// # Theory (First Principles)
    ///
    /// Standard microprice uses ±50 bps clamp which is conservative.
    /// When momentum signals are strong with high continuation probability:
    /// - The expected price move is larger than normal
    /// - The microprice should shift more aggressively toward the predicted direction
    /// - Dynamic clamp: ±50 bps + (momentum_bps × p_continuation × 0.3)
    ///
    /// # Arguments
    /// * `mid` - Current mid price
    /// * `book_imbalance` - Book imbalance signal [-1, 1]
    /// * `flow_imbalance` - Flow imbalance signal [-1, 1]
    /// * `momentum_bps` - Momentum in basis points (positive = rising)
    /// * `p_continuation` - Probability momentum continues [0, 1]
    /// * `urgency_score` - Urgency score [0, 5] from directional risk
    #[allow(dead_code)] // Reserved for future integration with momentum-aware quoting
    pub(crate) fn microprice_momentum_aware(
        &self,
        mid: f64,
        book_imbalance: f64,
        flow_imbalance: f64,
        momentum_bps: f64,
        p_continuation: f64,
        urgency_score: f64,
    ) -> f64 {
        if !self.is_warmed_up() {
            return mid;
        }

        // If R² is too low, the model has no predictive power - use mid
        if self.r_squared < self.min_r_squared {
            return mid;
        }

        // Mode-based adjustment (same as standard microprice)
        let adjustment = match self.correlation_mode {
            CorrelationMode::Combined => {
                let net_pressure = book_imbalance - flow_imbalance;
                self.beta_net * net_pressure
            }
            CorrelationMode::Orthogonalized | CorrelationMode::Independent => {
                self.beta_book * book_imbalance + self.beta_flow * flow_imbalance
            }
        };

        // === Dynamic Clamp Based on Momentum ===
        // Base clamp: ±50 bps
        // During strong momentum with high continuation:
        // - Widen clamp proportionally to momentum strength
        // - Max widening: 50 bps → 150 bps (3x) at extreme momentum
        let base_clamp_bps = 50.0;

        // Momentum-based widening factor
        // momentum_factor = |momentum_bps| / 100 × p_continuation × (1 + urgency/5)
        // Capped at 2.0 (allows up to 150 bps total clamp)
        let momentum_factor = if momentum_bps.abs() > 15.0 && p_continuation > 0.5 {
            let strength = (momentum_bps.abs() / 100.0).min(1.0);
            let urgency_boost = 1.0 + (urgency_score / 5.0).min(1.0);
            (strength * p_continuation * urgency_boost).min(2.0)
        } else {
            0.0
        };

        let dynamic_clamp_bps = base_clamp_bps * (1.0 + momentum_factor);
        let dynamic_clamp = dynamic_clamp_bps / 10000.0;

        // Clamp adjustment with dynamic bounds
        let adjustment_clamped = adjustment.clamp(-dynamic_clamp, dynamic_clamp);

        let raw_microprice = mid * (1.0 + adjustment_clamped);

        // Apply EMA smoothing if enabled (same as standard)
        if self.ema_alpha > 0.0 {
            let prev_bits = self.ema_microprice_bits.load(Ordering::Relaxed);

            if prev_bits == EMA_NONE {
                self.ema_microprice_bits
                    .store(raw_microprice.to_bits(), Ordering::Relaxed);
                raw_microprice
            } else {
                let prev = f64::from_bits(prev_bits);
                let change_bps = ((raw_microprice - prev) / prev).abs() * 10_000.0;

                if change_bps < self.ema_min_change_bps {
                    return prev;
                }

                let smoothed = self.ema_alpha * raw_microprice + (1.0 - self.ema_alpha) * prev;
                self.ema_microprice_bits
                    .store(smoothed.to_bits(), Ordering::Relaxed);
                smoothed
            }
        } else {
            raw_microprice
        }
    }

    pub(crate) fn is_warmed_up(&self) -> bool {
        self.n >= self.min_observations
    }

    pub(crate) fn beta_book(&self) -> f64 {
        self.beta_book
    }

    pub(crate) fn beta_flow(&self) -> f64 {
        self.beta_flow
    }

    pub(crate) fn r_squared(&self) -> f64 {
        self.r_squared
    }
}
