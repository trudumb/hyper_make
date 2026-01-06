//! HJB-derived optimal inventory control for market making.
//!
//! Implements the Avellaneda-Stoikov HJB (Hamilton-Jacobi-Bellman) solution
//! for optimal market making with inventory risk.
//!
//! The HJB equation:
//! ```text
//! ∂V/∂t + max_δ { λ(δ)[δ + V(t,x+δ,q-1,S) - V(t,x,q,S)] } - γσ²q² = 0
//! ```
//!
//! With terminal condition: `V(T,x,q,S) = x + q×S - penalty×q²`
//!
//! This module provides:
//! - Optimal inventory skew from the value function gradient
//! - Terminal penalty that forces position reduction before session end
//! - Funding rate integration for perpetuals (carry cost affects optimal inventory)
//! - Theoretically rigorous position management

use std::collections::VecDeque;
use std::time::Instant;

use crate::market_maker::estimator::TrendSignal;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for HJB inventory controller.
#[derive(Debug, Clone)]
pub struct HJBConfig {
    /// Session duration for terminal penalty (seconds)
    /// For 24/7 markets, use daily session (86400) or shorter sub-sessions
    pub session_duration_secs: f64,

    /// Terminal inventory penalty ($/unit²)
    /// Higher values force more aggressive position reduction near session end
    /// Typical: 0.0001 - 0.001 (0.01% - 0.1% per unit²)
    pub terminal_penalty: f64,

    /// Base risk aversion (γ)
    /// Used in diffusion skew calculation: γσ²qT
    pub gamma_base: f64,

    /// Funding rate half-life for EWMA (seconds)
    pub funding_ewma_half_life: f64,

    /// Minimum time remaining before terminal urgency kicks in (seconds)
    /// Avoids division by zero and extreme values near session end
    pub min_time_remaining: f64,

    /// Maximum terminal skew multiplier
    /// Caps the terminal penalty contribution to prevent extreme skews
    pub max_terminal_multiplier: f64,

    // === Drift-Adjusted Skew (First Principles Extension) ===
    /// Enable drift-adjusted skew from momentum signals.
    /// When true, incorporates predicted price drift into optimal skew.
    pub use_drift_adjusted_skew: bool,

    /// Sensitivity to momentum-position opposition [0.5, 3.0].
    /// Higher values increase skew urgency when position opposes momentum.
    pub opposition_sensitivity: f64,

    /// Maximum drift urgency multiplier [1.5, 5.0].
    /// Caps the drift contribution to prevent extreme skews.
    pub max_drift_urgency: f64,

    /// Minimum continuation probability to apply drift adjustment [0.3, 0.7].
    /// Below this, momentum is considered noise.
    pub min_continuation_prob: f64,

    // === EWMA Smoothing for Stability ===
    /// EWMA half-life for smoothing drift estimates (seconds).
    /// Longer values give more stable but slower-reacting signals.
    /// Range: [5.0, 60.0], Default: 15.0
    pub drift_ewma_half_life_secs: f64,

    /// Window size for momentum statistics (data points).
    /// Range: [20, 200], Default: 50
    pub momentum_stats_window: usize,

    /// Minimum observations before using smoothed signals.
    /// Range: [10, 50], Default: 20
    pub min_warmup_observations: usize,
}

impl Default for HJBConfig {
    fn default() -> Self {
        Self {
            session_duration_secs: 86400.0, // 24 hour session
            terminal_penalty: 0.0005,       // 0.05% per unit²
            gamma_base: 0.3,                // Moderate risk aversion
            funding_ewma_half_life: 3600.0, // 1 hour
            min_time_remaining: 60.0,       // 1 minute minimum
            max_terminal_multiplier: 5.0,   // Cap at 5x normal skew
            // Drift-adjusted skew (enabled by default for first-principles trading)
            use_drift_adjusted_skew: true,
            opposition_sensitivity: 1.5,
            max_drift_urgency: 3.0,
            min_continuation_prob: 0.5,
            // EWMA smoothing for stability
            drift_ewma_half_life_secs: 15.0,
            momentum_stats_window: 50,
            min_warmup_observations: 20,
        }
    }
}

// ============================================================================
// HJB Inventory Controller
// ============================================================================

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
    config: HJBConfig,

    /// Session start time
    session_start: Instant,

    /// Current volatility estimate (per-second)
    sigma: f64,

    /// Funding rate EWMA (annualized, positive = longs pay shorts)
    funding_rate_ewma: f64,

    /// EWMA alpha for funding rate
    funding_alpha: f64,

    /// Whether controller is initialized
    initialized: bool,

    // === EWMA Smoothing State ===
    /// EWMA alpha for drift smoothing
    drift_alpha: f64,

    /// Smoothed drift estimate (per-second)
    drift_ewma: f64,

    /// Smoothed variance multiplier
    variance_mult_ewma: f64,

    /// Recent momentum observations (bps)
    momentum_history: VecDeque<f64>,

    /// Recent continuation probability observations
    continuation_history: VecDeque<f64>,

    /// Number of drift updates received
    drift_update_count: u64,
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

    /// Compute optimal inventory skew from HJB solution.
    ///
    /// The full skew formula:
    /// ```text
    /// skew = γσ²qT + terminal_penalty × q × (1 - t/T) + funding_bias
    /// ```
    ///
    /// Where:
    /// - `γσ²qT`: Diffusion-driven inventory risk (standard A-S)
    /// - `terminal_penalty × q × (1 - t/T)`: Increases as session end approaches
    /// - `funding_bias`: Carry cost shifts optimal inventory target
    ///
    /// Returns skew in fractional units (multiply by price for absolute value).
    pub fn optimal_skew(&self, position: f64, max_position: f64) -> f64 {
        let gamma = self.config.gamma_base;
        let sigma = self.sigma;
        let time_remaining = self.time_remaining();

        // Normalize position to [-1, 1] range
        let q = if max_position > 1e-10 {
            (position / max_position).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // 1. Diffusion skew: γσ²qT
        // This is the standard Avellaneda-Stoikov inventory skew
        let diffusion_skew = gamma * sigma.powi(2) * q * time_remaining;

        // 2. Terminal penalty: increases skew as session end approaches
        // penalty × q × urgency, where urgency = 1 - time_remaining/session_duration
        let urgency = self.terminal_urgency();
        let terminal_skew = self.config.terminal_penalty * q * urgency;

        // Cap terminal contribution
        let terminal_skew_capped = terminal_skew
            .abs()
            .min(diffusion_skew.abs() * self.config.max_terminal_multiplier)
            * terminal_skew.signum();

        // 3. Funding bias: carry cost shifts optimal inventory
        // If funding positive (longs pay), we want to be short → adds positive skew
        // If funding negative (shorts pay), we want to be long → adds negative skew
        let funding_per_second = self.funding_rate_ewma / (365.0 * 24.0 * 3600.0);
        let funding_bias = funding_per_second * time_remaining * q.signum();

        diffusion_skew + terminal_skew_capped + funding_bias
    }

    /// Compute optimal inventory skew in basis points.
    pub fn optimal_skew_bps(&self, position: f64, max_position: f64) -> f64 {
        self.optimal_skew(position, max_position) * 10000.0
    }

    /// Compute drift-adjusted optimal skew from HJB solution with momentum.
    ///
    /// # Theory
    ///
    /// Standard HJB assumes price follows martingale: dS = σdW
    /// With momentum signals, price has drift: dS = μdt + σdW
    ///
    /// The extended HJB optimal skew becomes:
    /// ```text
    /// skew = γσ²qT + terminal_penalty + funding_bias + drift_urgency
    /// ```
    ///
    /// Where drift_urgency = μ × P(continuation) × time_exposure × opposition_factor
    ///
    /// When position **opposes** momentum (short + rising, long + falling):
    /// - Adverse moves are more likely (momentum predicts unfavorable direction)
    /// - We add urgency to accelerate position reduction
    ///
    /// When position **aligns** with momentum:
    /// - Risk is lower, we slightly reduce urgency but stay conservative
    ///
    /// # Arguments
    /// * `position` - Current position (positive = long, negative = short)
    /// * `max_position` - Maximum position for normalization
    /// * `momentum_bps` - Momentum in basis points (positive = rising)
    /// * `p_continuation` - Probability momentum continues [0, 1]
    ///
    /// # Returns
    /// Optimal skew in fractional units (multiply by price for absolute value).
    pub fn optimal_skew_with_drift(
        &self,
        position: f64,
        max_position: f64,
        momentum_bps: f64,
        p_continuation: f64,
    ) -> DriftAdjustedSkew {
        // Start with base HJB skew
        let base_skew = self.optimal_skew(position, max_position);

        if !self.config.use_drift_adjusted_skew {
            return DriftAdjustedSkew {
                total_skew: base_skew,
                base_skew,
                drift_urgency: 0.0,
                variance_multiplier: 1.0,
                is_opposed: false,
                urgency_score: 0.0,
            };
        }

        // Normalize position
        let q = if max_position > 1e-10 {
            (position / max_position).clamp(-1.0, 1.0)
        } else {
            return DriftAdjustedSkew {
                total_skew: base_skew,
                base_skew,
                drift_urgency: 0.0,
                variance_multiplier: 1.0,
                is_opposed: false,
                urgency_score: 0.0,
            };
        };

        // Check if position opposes momentum
        // Short (q < 0) + rising (momentum > 0) = opposed
        // Long (q > 0) + falling (momentum < 0) = opposed
        let is_opposed = q * momentum_bps < 0.0;

        // Only apply drift adjustment if:
        // 1. Position opposes momentum
        // 2. Continuation probability exceeds threshold (or we're using prior during warmup)
        // 3. Momentum is significant (dynamic threshold based on position size)
        //
        // Position-dependent threshold: larger positions need smaller momentum to trigger
        // - |q| = 0.1: threshold = 8 bps (small position, need clearer signal)
        // - |q| = 0.5: threshold = 5 bps (medium position)
        // - |q| = 1.0: threshold = 3 bps (max position, react to any trend)
        let base_threshold = 8.0;
        let position_factor = (1.0 - 0.6 * q.abs()).max(0.3); // Range: 0.3 to 1.0
        let momentum_threshold = base_threshold * position_factor;
        let momentum_significant = momentum_bps.abs() > momentum_threshold;

        // Accept 0.5 prior during warmup - better to react than ignore
        let continuation_confident = p_continuation >= self.config.min_continuation_prob;

        if !is_opposed || !momentum_significant || !continuation_confident {
            // No drift adjustment needed
            return DriftAdjustedSkew {
                total_skew: base_skew,
                base_skew,
                drift_urgency: 0.0,
                variance_multiplier: 1.0,
                is_opposed,
                urgency_score: 0.0,
            };
        }

        // === Compute Drift Urgency ===
        // From optimal control with drift: urgency ∝ μ × P(continue) × |q| × T
        let time_remaining = self.time_remaining().min(300.0); // Cap at 5 min exposure

        // Use EWMA-smoothed drift if warmed up, otherwise compute from raw momentum
        // Smoothed drift gives more stable signals and reduces whipsawing
        let drift_rate = if self.is_drift_warmed_up() {
            // Use smoothed drift for stable signals
            self.drift_ewma
        } else {
            // Fall back to raw momentum during warmup
            (momentum_bps / 10000.0) / 0.5
        };

        // Urgency formula:
        // drift_urgency = sensitivity × drift_rate × P(continue) × |q| × T
        let raw_urgency = self.config.opposition_sensitivity
            * drift_rate.abs()
            * p_continuation
            * q.abs()
            * time_remaining;

        // Cap urgency
        let max_base_urgency = base_skew.abs() * self.config.max_drift_urgency;
        let drift_urgency_magnitude = raw_urgency.min(max_base_urgency);

        // Sign: urgency should amplify the base skew direction
        // If short (q < 0), base skew is negative (quotes shift up)
        // Urgency should make it MORE negative (more aggressive buying)
        let drift_urgency = drift_urgency_magnitude * q.signum();

        // === Compute Variance Multiplier ===
        // Use EWMA-smoothed variance multiplier if warmed up for stability
        let variance_multiplier_capped = if self.is_drift_warmed_up() {
            // Use pre-computed smoothed variance multiplier
            self.variance_mult_ewma
        } else {
            // Compute inline during warmup
            // σ²_eff = σ² × (1 + κ × |momentum/σ| × P(continue))
            let momentum_vol_ratio = if self.sigma > 1e-10 {
                ((momentum_bps / 10000.0) / self.sigma).abs().min(3.0)
            } else {
                0.0
            };

            let variance_multiplier = 1.0
                + self.config.opposition_sensitivity
                    * momentum_vol_ratio
                    * p_continuation
                    * q.abs();
            variance_multiplier.min(self.config.max_drift_urgency)
        };

        // Compute momentum_vol_ratio for urgency_score (needed below)
        let momentum_vol_ratio = if self.sigma > 1e-10 {
            ((momentum_bps / 10000.0) / self.sigma).abs().min(3.0)
        } else {
            0.0
        };

        // Urgency score for diagnostics [0, 5]
        let urgency_score = (momentum_bps.abs() / 50.0).min(1.0) // Momentum strength
            + p_continuation // Continuation confidence
            + q.abs() // Position size
            + momentum_vol_ratio.min(1.0) // Vol-adjusted momentum
            + if self.is_terminal_zone() { 1.0 } else { 0.0 }; // Terminal zone boost

        DriftAdjustedSkew {
            total_skew: base_skew + drift_urgency,
            base_skew,
            drift_urgency,
            variance_multiplier: variance_multiplier_capped,
            is_opposed,
            urgency_score: urgency_score.min(5.0),
        }
    }

    /// Compute drift-adjusted skew in basis points.
    pub fn optimal_skew_with_drift_bps(
        &self,
        position: f64,
        max_position: f64,
        momentum_bps: f64,
        p_continuation: f64,
    ) -> DriftAdjustedSkew {
        let mut result =
            self.optimal_skew_with_drift(position, max_position, momentum_bps, p_continuation);
        result.total_skew *= 10000.0;
        result.base_skew *= 10000.0;
        result.drift_urgency *= 10000.0;
        result
    }

    /// Compute drift-adjusted skew with multi-timeframe trend detection.
    ///
    /// This is an enhanced version that uses the TrendSignal to detect
    /// sustained trends even when short-term momentum flips due to bounces.
    ///
    /// # Opposition Detection (Enhanced)
    ///
    /// Position is considered "opposed to trend" if ANY of:
    /// 1. Short-term momentum opposes position (original logic)
    /// 2. Medium-term (30s) momentum opposes position with agreement > 0.5
    /// 3. Long-term (5min) momentum opposes position
    /// 4. Position is significantly underwater (severity > 0.3)
    ///
    /// # Urgency Scaling
    ///
    /// When trend_confidence is high (multiple timeframes agree), the
    /// drift urgency is boosted by up to 2x (configurable via TrendConfig).
    pub fn optimal_skew_with_trend(
        &self,
        position: f64,
        max_position: f64,
        momentum_bps: f64,
        p_continuation: f64,
        trend: &TrendSignal,
    ) -> DriftAdjustedSkew {
        // Start with base HJB skew
        let base_skew = self.optimal_skew(position, max_position);

        if !self.config.use_drift_adjusted_skew {
            return DriftAdjustedSkew {
                total_skew: base_skew,
                base_skew,
                drift_urgency: 0.0,
                variance_multiplier: 1.0,
                is_opposed: false,
                urgency_score: 0.0,
            };
        }

        // Normalize position
        let q = if max_position > 1e-10 {
            (position / max_position).clamp(-1.0, 1.0)
        } else {
            return DriftAdjustedSkew {
                total_skew: base_skew,
                base_skew,
                drift_urgency: 0.0,
                variance_multiplier: 1.0,
                is_opposed: false,
                urgency_score: 0.0,
            };
        };

        // === Enhanced Opposition Detection ===
        // Check if position opposes momentum at ANY timeframe, or is underwater

        // Short-term opposition (original logic)
        let short_opposed = q * momentum_bps < 0.0;

        // Medium-term opposition (30s window, requires some agreement)
        let medium_opposed = if trend.is_warmed_up && trend.timeframe_agreement > 0.5 {
            q * trend.medium_momentum_bps < 0.0 && trend.medium_momentum_bps.abs() > 3.0
        } else {
            false
        };

        // Long-term opposition (5min window, more authoritative)
        let long_opposed = if trend.is_warmed_up {
            q * trend.long_momentum_bps < 0.0 && trend.long_momentum_bps.abs() > 5.0
        } else {
            false
        };

        // Underwater opposition (position losing money in a sustained way)
        let underwater_opposed = trend.underwater_severity > 0.3;

        // Combined opposition: any trigger counts
        let is_opposed = short_opposed || medium_opposed || long_opposed || underwater_opposed;

        // Position-dependent threshold (same as original)
        let base_threshold = 8.0;
        let position_factor = (1.0 - 0.6 * q.abs()).max(0.3);
        let momentum_threshold = base_threshold * position_factor;

        // Use the strongest momentum signal for threshold check
        let effective_momentum = if trend.is_warmed_up {
            // When long-term has a clear signal, use it
            if trend.long_momentum_bps.abs() > 10.0 {
                trend.long_momentum_bps.abs()
            } else if trend.medium_momentum_bps.abs() > 5.0 {
                trend.medium_momentum_bps.abs()
            } else {
                momentum_bps.abs()
            }
        } else {
            momentum_bps.abs()
        };

        let momentum_significant = effective_momentum > momentum_threshold;
        let continuation_confident = p_continuation >= self.config.min_continuation_prob;

        if !is_opposed || !momentum_significant || !continuation_confident {
            return DriftAdjustedSkew {
                total_skew: base_skew,
                base_skew,
                drift_urgency: 0.0,
                variance_multiplier: 1.0,
                is_opposed,
                urgency_score: 0.0,
            };
        }

        // === Compute Drift Urgency (with trend boost) ===
        let time_remaining = self.time_remaining().min(300.0);

        // Use EWMA-smoothed drift if warmed up
        let drift_rate = if self.is_drift_warmed_up() {
            self.drift_ewma
        } else {
            (momentum_bps / 10000.0) / 0.5
        };

        // Base urgency calculation
        let raw_urgency = self.config.opposition_sensitivity
            * drift_rate.abs()
            * p_continuation
            * q.abs()
            * time_remaining;

        // Boost urgency when trend is confident (multi-timeframe agreement)
        // trend_confidence is 0.0-1.0, boost factor is 1.0-2.0
        let trend_boost = if trend.is_warmed_up {
            1.0 + trend.trend_confidence // 1.0 to 2.0
        } else {
            1.0
        };

        let boosted_urgency = raw_urgency * trend_boost;
        let max_base_urgency = base_skew.abs() * self.config.max_drift_urgency;
        let drift_urgency_magnitude = boosted_urgency.min(max_base_urgency);
        let drift_urgency = drift_urgency_magnitude * q.signum();

        // === Compute Variance Multiplier ===
        let variance_multiplier_capped = if self.is_drift_warmed_up() {
            self.variance_mult_ewma * trend_boost.min(1.5) // Additional boost, capped
        } else {
            let momentum_vol_ratio = if self.sigma > 1e-10 {
                ((momentum_bps / 10000.0) / self.sigma).abs().min(3.0)
            } else {
                0.0
            };
            let variance_multiplier = 1.0
                + self.config.opposition_sensitivity
                    * momentum_vol_ratio
                    * p_continuation
                    * q.abs();
            variance_multiplier.min(self.config.max_drift_urgency)
        };

        // Urgency score for diagnostics
        let momentum_vol_ratio = if self.sigma > 1e-10 {
            ((momentum_bps / 10000.0) / self.sigma).abs().min(3.0)
        } else {
            0.0
        };

        let urgency_score = (momentum_bps.abs() / 50.0).min(1.0)
            + p_continuation
            + q.abs()
            + momentum_vol_ratio.min(1.0)
            + if self.is_terminal_zone() { 1.0 } else { 0.0 }
            + trend.trend_confidence; // Add trend confidence to score

        DriftAdjustedSkew {
            total_skew: base_skew + drift_urgency,
            base_skew,
            drift_urgency,
            variance_multiplier: variance_multiplier_capped,
            is_opposed,
            urgency_score: urgency_score.min(6.0), // Max now 6.0 with trend
        }
    }

    /// Compute drift-adjusted skew with trend in basis points.
    pub fn optimal_skew_with_trend_bps(
        &self,
        position: f64,
        max_position: f64,
        momentum_bps: f64,
        p_continuation: f64,
        trend: &TrendSignal,
    ) -> DriftAdjustedSkew {
        let mut result = self.optimal_skew_with_trend(
            position,
            max_position,
            momentum_bps,
            p_continuation,
            trend,
        );
        result.total_skew *= 10000.0;
        result.base_skew *= 10000.0;
        result.drift_urgency *= 10000.0;
        result
    }

    /// Compute the optimal inventory target (not always zero for perpetuals).
    ///
    /// Theory: With non-zero funding rate, the optimal inventory is:
    /// ```text
    /// q* = -funding_rate / (2 × terminal_penalty)
    /// ```
    ///
    /// Positive funding → optimal to be short
    /// Negative funding → optimal to be long
    ///
    /// Returns target as fraction of max_position.
    pub fn optimal_inventory_target(&self) -> f64 {
        if self.config.terminal_penalty.abs() < 1e-10 {
            return 0.0;
        }

        let funding_per_second = self.funding_rate_ewma / (365.0 * 24.0 * 3600.0);

        // Target = -funding / (2 × penalty)
        let target = -funding_per_second / (2.0 * self.config.terminal_penalty);

        // Clamp to reasonable range
        target.clamp(-0.5, 0.5)
    }

    /// Compute how much to adjust gamma based on session timing.
    ///
    /// Near session end, we want to be more aggressive about reducing
    /// inventory, which means higher gamma.
    ///
    /// Returns a multiplier to apply to base gamma.
    pub fn gamma_multiplier(&self) -> f64 {
        let urgency = self.terminal_urgency();

        // Ramp up gamma near session end
        // At t=0: multiplier = 1.0
        // At t=T: multiplier = 1.0 + max_terminal_multiplier
        1.0 + urgency * (self.config.max_terminal_multiplier - 1.0)
    }

    /// Get effective gamma (base × multiplier).
    pub fn effective_gamma(&self) -> f64 {
        self.config.gamma_base * self.gamma_multiplier()
    }

    /// Compute the value function gradient ∂V/∂q at current state.
    ///
    /// This gives the marginal cost of holding one more unit of inventory.
    /// Useful for diagnostics and sizing decisions.
    ///
    /// ∂V/∂q ≈ -S - 2γσ²qT - 2×penalty×q×urgency + funding×T
    pub fn value_gradient(&self, position: f64, max_position: f64, price: f64) -> f64 {
        let gamma = self.config.gamma_base;
        let sigma = self.sigma;
        let time_remaining = self.time_remaining();
        let urgency = self.terminal_urgency();

        let q = if max_position > 1e-10 {
            (position / max_position).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // The gradient captures the marginal cost of inventory
        let inventory_cost = 2.0 * gamma * sigma.powi(2) * q * time_remaining;
        let terminal_cost = 2.0 * self.config.terminal_penalty * q * urgency;
        let funding_benefit = self.funding_rate_ewma / (365.0 * 24.0 * 3600.0) * time_remaining;

        // Negative of costs (value decreases with costs)
        -price - inventory_cost - terminal_cost + funding_benefit * q.signum()
    }

    /// Check if we're in terminal urgency zone (last portion of session).
    ///
    /// Returns true if urgency > 0.8 (last 20% of session).
    pub fn is_terminal_zone(&self) -> bool {
        self.terminal_urgency() > 0.8
    }

    /// Get summary for diagnostics.
    pub fn summary(&self) -> HJBSummary {
        HJBSummary {
            time_remaining_secs: self.time_remaining(),
            terminal_urgency: self.terminal_urgency(),
            is_terminal_zone: self.is_terminal_zone(),
            gamma_multiplier: self.gamma_multiplier(),
            effective_gamma: self.effective_gamma(),
            funding_rate_ewma: self.funding_rate_ewma,
            optimal_inventory_target: self.optimal_inventory_target(),
            sigma: self.sigma,
        }
    }
}

/// Output from drift-adjusted skew calculation.
///
/// Contains both the total skew and its components for analysis.
#[derive(Debug, Clone, Copy, Default)]
pub struct DriftAdjustedSkew {
    /// Total optimal skew (base + drift urgency).
    pub total_skew: f64,

    /// Base HJB skew (γσ²qT + terminal + funding).
    pub base_skew: f64,

    /// Additional urgency from momentum-position opposition.
    /// Same sign as base_skew when opposed (amplifies reduction).
    pub drift_urgency: f64,

    /// Variance multiplier for inventory risk.
    /// > 1.0 when opposed, used for σ²_effective calculation.
    pub variance_multiplier: f64,

    /// Whether position is opposed to momentum.
    pub is_opposed: bool,

    /// Urgency score [0, 5] for diagnostics.
    pub urgency_score: f64,
}

/// Summary of HJB controller state for diagnostics.
#[derive(Debug, Clone)]
pub struct HJBSummary {
    pub time_remaining_secs: f64,
    pub terminal_urgency: f64,
    pub is_terminal_zone: bool,
    pub gamma_multiplier: f64,
    pub effective_gamma: f64,
    pub funding_rate_ewma: f64,
    pub optimal_inventory_target: f64,
    pub sigma: f64,
}

/// Momentum statistics for diagnostics and confidence estimation.
#[derive(Debug, Clone, Default)]
pub struct MomentumStats {
    /// Mean momentum over window (bps).
    pub mean_bps: f64,
    /// Standard deviation of momentum (bps).
    pub std_dev_bps: f64,
    /// Number of direction changes in window.
    pub direction_changes: usize,
    /// Number of samples in window.
    pub sample_count: usize,
    /// Average continuation probability.
    pub avg_continuation: f64,
}

impl MomentumStats {
    /// Check if momentum signal is noisy (many direction changes).
    pub fn is_noisy(&self) -> bool {
        if self.sample_count < 10 {
            return true;
        }
        // More than 40% direction changes = noisy
        (self.direction_changes as f64 / self.sample_count as f64) > 0.4
    }

    /// Get signal quality score [0, 1].
    /// Higher = more reliable momentum signal.
    pub fn signal_quality(&self) -> f64 {
        if self.sample_count < 10 {
            return 0.0;
        }

        // Factors that increase quality:
        // 1. Mean momentum magnitude (stronger signals)
        let magnitude_score = (self.mean_bps.abs() / 50.0).min(1.0);

        // 2. Low direction changes (consistent trend)
        let consistency =
            1.0 - (self.direction_changes as f64 / self.sample_count as f64).min(0.5) * 2.0;

        // 3. High continuation probability
        let continuation_score = self.avg_continuation;

        // 4. Low volatility relative to mean (high signal-to-noise)
        let snr = if self.std_dev_bps > 0.0 {
            (self.mean_bps.abs() / self.std_dev_bps).min(2.0) / 2.0
        } else {
            0.0
        };

        // Combine factors
        (magnitude_score * 0.2 + consistency * 0.3 + continuation_score * 0.3 + snr * 0.2)
            .clamp(0.0, 1.0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_controller() -> HJBInventoryController {
        let config = HJBConfig {
            session_duration_secs: 100.0, // Short session for testing
            terminal_penalty: 0.001,
            gamma_base: 0.3,
            ..Default::default()
        };
        let mut ctrl = HJBInventoryController::new(config);
        ctrl.start_session();
        ctrl.update_sigma(0.0001); // 1 bp/sec
        ctrl
    }

    #[test]
    fn test_hjb_basic() {
        let ctrl = make_controller();

        // Should be initialized
        assert!(ctrl.initialized);
        assert!(ctrl.time_remaining() > 0.0);
        assert!(ctrl.terminal_urgency() < 0.5); // Early in session
    }

    #[test]
    fn test_hjb_zero_position_zero_skew() {
        let ctrl = make_controller();

        // With zero position, skew should be ~zero (funding aside)
        let skew = ctrl.optimal_skew(0.0, 1.0);
        assert!(
            skew.abs() < 1e-6,
            "Zero position should give ~zero skew: {}",
            skew
        );
    }

    #[test]
    fn test_hjb_long_position_positive_skew() {
        let ctrl = make_controller();

        // Long position should give positive skew (shift quotes down)
        let skew = ctrl.optimal_skew(0.5, 1.0);
        assert!(
            skew > 0.0,
            "Long position should give positive skew: {}",
            skew
        );
    }

    #[test]
    fn test_hjb_short_position_negative_skew() {
        let ctrl = make_controller();

        // Short position should give negative skew (shift quotes up)
        let skew = ctrl.optimal_skew(-0.5, 1.0);
        assert!(
            skew < 0.0,
            "Short position should give negative skew: {}",
            skew
        );
    }

    #[test]
    fn test_hjb_skew_symmetry() {
        let ctrl = make_controller();

        // Skew should be antisymmetric in position
        let skew_long = ctrl.optimal_skew(0.5, 1.0);
        let skew_short = ctrl.optimal_skew(-0.5, 1.0);

        assert!(
            (skew_long + skew_short).abs() < 1e-8,
            "Skew should be antisymmetric: long={}, short={}",
            skew_long,
            skew_short
        );
    }

    #[test]
    fn test_hjb_gamma_multiplier() {
        let ctrl = make_controller();

        // At start of session, multiplier should be ~1.0
        let mult = ctrl.gamma_multiplier();
        assert!(
            mult >= 1.0 && mult < 1.5,
            "Early multiplier should be near 1.0: {}",
            mult
        );

        // Effective gamma should be base × multiplier (computed at same instant)
        // Get both from effective_gamma method which computes mult internally
        let eff = ctrl.effective_gamma();
        let expected = ctrl.config.gamma_base * ctrl.gamma_multiplier();
        assert!(
            (eff - expected).abs() < 0.01,
            "Effective gamma {} should be close to gamma_base {} × multiplier: expected {}",
            eff,
            ctrl.config.gamma_base,
            expected
        );
    }

    #[test]
    fn test_hjb_optimal_inventory_target_no_funding() {
        let ctrl = make_controller();

        // With zero funding, optimal target is zero
        let target = ctrl.optimal_inventory_target();
        assert!(
            target.abs() < 0.01,
            "With zero funding, target should be ~0: {}",
            target
        );
    }

    #[test]
    fn test_hjb_optimal_inventory_target_with_funding() {
        let mut ctrl = make_controller();

        // Positive funding rate (longs pay) → optimal to be short
        ctrl.update_funding(0.001); // 0.1% 8-hour rate
        let target = ctrl.optimal_inventory_target();

        // With positive funding, target should be negative (short)
        assert!(
            target < 0.0,
            "Positive funding should give negative target: {}",
            target
        );
    }

    #[test]
    fn test_hjb_funding_ewma() {
        // Use a faster-converging controller for testing
        let config = HJBConfig {
            session_duration_secs: 100.0,
            terminal_penalty: 0.001,
            gamma_base: 0.3,
            funding_ewma_half_life: 10.0, // Fast EWMA for testing (10 seconds)
            ..Default::default()
        };
        let mut ctrl = HJBInventoryController::new(config);
        ctrl.start_session();

        // Initial funding rate
        ctrl.update_funding(0.001);
        let rate1 = ctrl.funding_rate_ewma;

        // Update with same rate - EWMA should move toward target
        ctrl.update_funding(0.001);
        let rate2 = ctrl.funding_rate_ewma;

        // Expected annualized rate
        let annualized = 0.001 * 3.0 * 365.0; // = 1.095

        // EWMA should be moving toward the annualized rate
        assert!(
            rate2 > rate1,
            "EWMA should increase toward target: {} -> {}",
            rate1,
            rate2
        );
        assert!(rate2 < annualized, "EWMA should not exceed target");

        // After many updates with fast EWMA, should converge
        for _ in 0..100 {
            ctrl.update_funding(0.001);
        }
        let rate_converged = ctrl.funding_rate_ewma;

        // Should be close to annualized after convergence
        assert!(
            (rate_converged - annualized).abs() / annualized < 0.1,
            "EWMA should converge to annualized rate: {} vs {}",
            rate_converged,
            annualized
        );
    }

    #[test]
    fn test_hjb_value_gradient() {
        let ctrl = make_controller();

        // Value gradient at zero position
        let grad_zero = ctrl.value_gradient(0.0, 1.0, 100.0);

        // Value gradient with long position (should be more negative = higher cost)
        let grad_long = ctrl.value_gradient(0.5, 1.0, 100.0);

        // Holding inventory has cost, so gradient should differ
        // (exact relationship depends on parameters)
        assert!(grad_zero != grad_long, "Gradient should depend on position");
    }

    #[test]
    fn test_hjb_terminal_zone() {
        let ctrl = make_controller();

        // Early in session, not in terminal zone
        assert!(!ctrl.is_terminal_zone());
    }

    #[test]
    fn test_hjb_summary() {
        let ctrl = make_controller();
        let summary = ctrl.summary();

        assert!(summary.time_remaining_secs > 0.0);
        assert!(summary.terminal_urgency >= 0.0 && summary.terminal_urgency <= 1.0);
        assert!(summary.gamma_multiplier >= 1.0);
        assert!(summary.sigma > 0.0);
    }

    #[test]
    fn test_hjb_skew_increases_with_position() {
        let ctrl = make_controller();

        // Larger position → larger skew magnitude
        let skew_small = ctrl.optimal_skew(0.1, 1.0);
        let skew_large = ctrl.optimal_skew(0.5, 1.0);

        assert!(
            skew_large.abs() > skew_small.abs(),
            "Larger position should give larger skew: small={}, large={}",
            skew_small,
            skew_large
        );
    }

    #[test]
    fn test_hjb_skew_increases_with_volatility() {
        let mut ctrl = make_controller();

        ctrl.update_sigma(0.0001);
        let skew_low_vol = ctrl.optimal_skew(0.5, 1.0);

        ctrl.update_sigma(0.001); // 10x higher vol
        let skew_high_vol = ctrl.optimal_skew(0.5, 1.0);

        assert!(
            skew_high_vol.abs() > skew_low_vol.abs(),
            "Higher vol should give larger skew: low={}, high={}",
            skew_low_vol,
            skew_high_vol
        );
    }

    #[test]
    fn test_hjb_drift_warmup() {
        let mut ctrl = make_controller();

        // Initially not warmed up
        assert!(!ctrl.is_drift_warmed_up());

        // Update momentum signals
        for _ in 0..25 {
            ctrl.update_momentum_signals(10.0, 0.6, -0.5, 1.0);
        }

        // Now should be warmed up
        assert!(ctrl.is_drift_warmed_up());
    }

    #[test]
    fn test_hjb_ewma_smoothing() {
        let mut ctrl = make_controller();

        // Start with no drift
        assert!((ctrl.smoothed_drift()).abs() < 1e-10);
        assert!((ctrl.smoothed_variance_multiplier() - 1.0).abs() < 1e-10);

        // Add positive momentum (opposed to short position)
        for _ in 0..30 {
            ctrl.update_momentum_signals(20.0, 0.7, -0.5, 1.0);
        }

        // Drift should be positive (rising)
        assert!(
            ctrl.smoothed_drift() > 0.0,
            "Positive momentum should give positive drift: {}",
            ctrl.smoothed_drift()
        );

        // Variance multiplier should be > 1 (opposed position)
        assert!(
            ctrl.smoothed_variance_multiplier() > 1.0,
            "Opposed position should increase variance: {}",
            ctrl.smoothed_variance_multiplier()
        );
    }

    #[test]
    fn test_hjb_momentum_stats() {
        let mut ctrl = make_controller();

        // Add momentum signals
        for i in 0..30 {
            let momentum = if i % 5 == 0 { -5.0 } else { 15.0 };
            ctrl.update_momentum_signals(momentum, 0.6, -0.3, 1.0);
        }

        let stats = ctrl.momentum_stats();

        assert!(stats.sample_count > 0);
        assert!(stats.mean_bps > 0.0); // Mostly positive momentum
        assert!(stats.std_dev_bps > 0.0); // Some variance
        assert!(stats.direction_changes > 0); // Some direction changes
    }

    #[test]
    fn test_hjb_drift_adjusted_skew_uses_smoothed() {
        let mut ctrl = make_controller();
        ctrl.start_session();

        // Warm up with consistent momentum
        for _ in 0..30 {
            ctrl.update_momentum_signals(25.0, 0.7, -0.5, 1.0);
        }

        assert!(ctrl.is_drift_warmed_up());

        // Get drift-adjusted skew
        let result = ctrl.optimal_skew_with_drift(-0.5, 1.0, 25.0, 0.7);

        // Should be opposed (short + rising)
        assert!(result.is_opposed);

        // Should have drift urgency
        assert!(
            result.drift_urgency.abs() > 0.0,
            "Expected drift urgency: {}",
            result.drift_urgency
        );

        // Should use smoothed variance multiplier (> 1.0 for opposed)
        assert!(
            result.variance_multiplier > 1.0,
            "Expected variance > 1.0 for opposed: {}",
            result.variance_multiplier
        );
    }

    #[test]
    fn test_momentum_stats_signal_quality() {
        // Create high quality signal
        let high_quality = MomentumStats {
            mean_bps: 30.0,
            std_dev_bps: 5.0,
            direction_changes: 2,
            sample_count: 50,
            avg_continuation: 0.75,
        };

        // Create low quality signal
        let low_quality = MomentumStats {
            mean_bps: 5.0,
            std_dev_bps: 20.0,
            direction_changes: 25,
            sample_count: 50,
            avg_continuation: 0.45,
        };

        assert!(
            high_quality.signal_quality() > low_quality.signal_quality(),
            "High quality signal should have higher score: {} vs {}",
            high_quality.signal_quality(),
            low_quality.signal_quality()
        );

        assert!(
            !high_quality.is_noisy(),
            "High quality signal should not be noisy"
        );
        assert!(low_quality.is_noisy(), "Low quality signal should be noisy");
    }
}
