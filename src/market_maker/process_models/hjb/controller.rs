//! HJB Inventory Controller struct and update methods.

use std::collections::VecDeque;
use std::time::Instant;

use super::config::HJBConfig;
use super::summary::MomentumStats;

/// HJB-derived optimal inventory controller.
///
/// Computes optimal inventory skew using the closed-form solution to the
/// Avellaneda-Stoikov HJB equation. Key features:
///
/// 1. **Diffusion Skew**: γσ²qT - standard A-S formula for inventory risk
/// 2. **Terminal Penalty**: Forces position reduction as session end approaches
/// 3. **Funding Integration**: Accounts for perpetual funding costs in carry
/// 4. **Optimal Inventory Target**: Not always zero (funding affects target)
///
/// The controller is stateful, tracking:
/// - Session timing for terminal penalty
/// - Funding rate EWMA for carry cost estimation
#[derive(Debug, Clone)]
pub struct HJBInventoryController {
    pub(super) config: HJBConfig,

    /// Session start time
    pub(super) session_start: Instant,

    /// Current volatility estimate (per-second)
    pub(super) sigma: f64,

    /// Funding rate EWMA (annualized, positive = longs pay shorts)
    pub(super) funding_rate_ewma: f64,

    /// EWMA alpha for funding rate
    pub(super) funding_alpha: f64,

    /// Whether controller is initialized
    pub(super) initialized: bool,

    // === EWMA Smoothing State ===
    /// EWMA alpha for drift smoothing
    pub(super) drift_alpha: f64,

    /// Smoothed drift estimate (per-second)
    pub(super) drift_ewma: f64,

    /// Smoothed variance multiplier
    pub(super) variance_mult_ewma: f64,

    /// Recent momentum observations (bps)
    pub(super) momentum_history: VecDeque<f64>,

    /// Recent continuation probability observations
    pub(super) continuation_history: VecDeque<f64>,

    /// Number of drift updates received
    pub(super) drift_update_count: u64,
}

impl HJBInventoryController {
    /// Create a new HJB inventory controller.
    pub fn new(config: HJBConfig) -> Self {
        let funding_alpha = (2.0_f64.ln() / config.funding_ewma_half_life).clamp(0.0001, 1.0);

        // Compute drift EWMA alpha from half-life
        // Assume ~10 updates per second
        let updates_per_half_life = config.drift_ewma_half_life_secs * 10.0;
        let drift_alpha = (1.0 - (-2.0_f64.ln() / updates_per_half_life).exp()).clamp(0.01, 0.5);

        let momentum_capacity = config.momentum_stats_window + 10;

        Self {
            config,
            session_start: Instant::now(),
            sigma: 0.0001, // 1 bp/sec default
            funding_rate_ewma: 0.0,
            funding_alpha,
            initialized: false,
            // EWMA state
            drift_alpha,
            drift_ewma: 0.0,
            variance_mult_ewma: 1.0,
            momentum_history: VecDeque::with_capacity(momentum_capacity),
            continuation_history: VecDeque::with_capacity(momentum_capacity),
            drift_update_count: 0,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(HJBConfig::default())
    }

    /// Start a new session (resets terminal penalty timing).
    pub fn start_session(&mut self) {
        self.session_start = Instant::now();
        self.initialized = true;
    }

    /// Update volatility estimate.
    pub fn update_sigma(&mut self, sigma: f64) {
        self.sigma = sigma.max(1e-10); // Floor to avoid zero
    }

    /// Update funding rate.
    ///
    /// # Arguments
    /// * `funding_rate` - Current 8-hour funding rate (as decimal, e.g., 0.0001 = 0.01%)
    pub fn update_funding(&mut self, funding_rate: f64) {
        // Convert 8-hour rate to annualized
        let annualized = funding_rate * 3.0 * 365.0; // 3 periods/day × 365 days

        // EWMA update
        if self.initialized {
            self.funding_rate_ewma = self.funding_alpha * annualized
                + (1.0 - self.funding_alpha) * self.funding_rate_ewma;
        } else {
            self.funding_rate_ewma = annualized;
        }
    }

    /// Update momentum/drift signals with EWMA smoothing.
    ///
    /// This method should be called on each quote cycle to maintain
    /// smooth drift estimates that reduce noise in skew adjustments.
    ///
    /// # Arguments
    /// * `momentum_bps` - Current momentum in basis points (positive = rising)
    /// * `p_continuation` - Probability momentum continues [0, 1]
    /// * `position` - Current position (for variance multiplier calculation)
    /// * `max_position` - Maximum position for normalization
    pub fn update_momentum_signals(
        &mut self,
        momentum_bps: f64,
        p_continuation: f64,
        position: f64,
        max_position: f64,
    ) {
        self.drift_update_count += 1;

        // Track momentum history
        self.momentum_history.push_back(momentum_bps);
        if self.momentum_history.len() > self.config.momentum_stats_window {
            self.momentum_history.pop_front();
        }

        // Track continuation history
        self.continuation_history.push_back(p_continuation);
        if self.continuation_history.len() > self.config.momentum_stats_window {
            self.continuation_history.pop_front();
        }

        // Convert momentum to fractional drift estimate
        // μ = momentum_bps / 10000 / expected_duration
        // We estimate expected duration as 500ms for momentum signal
        let momentum_frac = momentum_bps / 10000.0;
        let estimated_drift = momentum_frac / 0.5; // Per-second drift

        // EWMA smooth the drift estimate
        self.drift_ewma =
            self.drift_alpha * estimated_drift + (1.0 - self.drift_alpha) * self.drift_ewma;

        // Compute and smooth variance multiplier
        let q = if max_position.abs() > 1e-10 {
            (position / max_position).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // is_opposed = position and momentum are in opposite directions
        let is_opposed = q * momentum_bps < 0.0;

        // Compute raw variance multiplier
        let variance_multiplier = if is_opposed && q.abs() > 0.05 {
            // Opposed: increase variance proportionally to opposition strength
            let momentum_sigma_ratio = if self.sigma > 1e-10 {
                (momentum_frac.abs() / self.sigma).min(5.0)
            } else {
                0.0
            };
            let opposition_strength = q.abs() * (momentum_bps.abs() / 100.0).min(1.0);
            let increase = self.config.opposition_sensitivity
                * opposition_strength
                * (1.0 + momentum_sigma_ratio * 0.5)
                * p_continuation;

            (1.0 + increase).min(self.config.max_drift_urgency)
        } else {
            1.0
        };

        // EWMA smooth variance multiplier
        self.variance_mult_ewma = self.drift_alpha * variance_multiplier
            + (1.0 - self.drift_alpha) * self.variance_mult_ewma;
    }

    /// Check if drift smoothing is warmed up.
    pub fn is_drift_warmed_up(&self) -> bool {
        self.drift_update_count >= self.config.min_warmup_observations as u64
    }

    /// Get smoothed drift estimate (per-second).
    pub fn smoothed_drift(&self) -> f64 {
        self.drift_ewma
    }

    /// Get smoothed variance multiplier.
    pub fn smoothed_variance_multiplier(&self) -> f64 {
        self.variance_mult_ewma
    }

    /// Get momentum statistics for diagnostics.
    pub fn momentum_stats(&self) -> MomentumStats {
        if self.momentum_history.is_empty() {
            return MomentumStats::default();
        }

        let sum: f64 = self.momentum_history.iter().sum();
        let mean = sum / self.momentum_history.len() as f64;

        let variance: f64 = self
            .momentum_history
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / self.momentum_history.len() as f64;

        let std_dev = variance.sqrt();

        // Count directional changes
        let mut direction_changes = 0;
        let mut prev_sign = 0.0_f64;
        for &m in &self.momentum_history {
            if prev_sign != 0.0 && m.signum() != prev_sign {
                direction_changes += 1;
            }
            if m.abs() > 1e-10 {
                prev_sign = m.signum();
            }
        }

        // Compute average continuation probability
        let avg_continuation = if self.continuation_history.is_empty() {
            0.5
        } else {
            self.continuation_history.iter().sum::<f64>() / self.continuation_history.len() as f64
        };

        MomentumStats {
            mean_bps: mean,
            std_dev_bps: std_dev,
            direction_changes,
            sample_count: self.momentum_history.len(),
            avg_continuation,
        }
    }

    /// Get time remaining in current session (seconds).
    pub fn time_remaining(&self) -> f64 {
        let elapsed = self.session_start.elapsed().as_secs_f64();
        (self.config.session_duration_secs - elapsed).max(self.config.min_time_remaining)
    }

    /// Get terminal urgency factor (0 = start of session, 1 = end of session).
    pub fn terminal_urgency(&self) -> f64 {
        let elapsed = self.session_start.elapsed().as_secs_f64();
        (elapsed / self.config.session_duration_secs).clamp(0.0, 1.0)
    }
}
