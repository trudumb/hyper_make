//! Optimal skew calculations for HJB controller.

use crate::market_maker::estimator::TrendSignal;

use super::controller::HJBInventoryController;
use super::summary::{DriftAdjustedSkew, HJBSummary};

/// Default predictive bias threshold (minimum changepoint_prob to activate).
const DEFAULT_PREDICTIVE_BIAS_THRESHOLD: f64 = 0.3;
/// Default predictive bias sensitivity (σ multiplier).
const DEFAULT_PREDICTIVE_BIAS_SENSITIVITY: f64 = 2.0;

// === Momentum Threshold Constants ===
/// Base momentum threshold (bps) before position scaling.
/// Small positions need clearer signal; large positions react earlier.
const MOMENTUM_THRESHOLD_BASE_BPS: f64 = 8.0;
/// Position-factor slope: reduces threshold as |q| increases.
/// At |q|=1.0: factor = 1 - 0.6 = 0.4 → threshold = 8 * 0.4 = 3.2 bps.
const MOMENTUM_THRESHOLD_POSITION_SLOPE: f64 = 0.6;
/// Floor for position factor (prevents threshold from reaching zero).
const MOMENTUM_THRESHOLD_POSITION_FLOOR: f64 = 0.3;

// === Soft Gate Constants ===
/// Continuation probability below which the soft gate ramps from 0.
const SOFT_GATE_RAMP_START: f64 = 0.2;
/// Gate activation floor: skip drift adjustment when gate < this value.
const SOFT_GATE_MIN_ACTIVATION: f64 = 0.01;

// === Drift Urgency Constants ===
/// Maximum time exposure cap for drift urgency (seconds, = 5 minutes).
const MAX_DRIFT_EXPOSURE_S: f64 = 300.0;
/// Momentum normalization for urgency score (bps).
const URGENCY_MOMENTUM_NORM_BPS: f64 = 50.0;
/// Maximum momentum-to-volatility ratio for variance multiplier.
const MAX_MOMENTUM_VOL_RATIO: f64 = 3.0;

// === Trend Detection Constants ===
/// Minimum timeframe agreement to consider medium-term opposition.
const TREND_MIN_AGREEMENT: f64 = 0.5;
/// Medium-term momentum threshold for opposition detection (bps).
const TREND_MEDIUM_OPPOSITION_BPS: f64 = 3.0;
/// Long-term momentum threshold for opposition detection (bps).
const TREND_LONG_OPPOSITION_BPS: f64 = 5.0;
/// Strong long-term momentum threshold for drift fallback (bps).
const TREND_STRONG_LONG_BPS: f64 = 10.0;
/// Medium-term momentum threshold for drift fallback (bps).
const TREND_MEDIUM_DRIFT_BPS: f64 = 5.0;
/// Underwater severity threshold for opposition detection.
const TREND_UNDERWATER_THRESHOLD: f64 = 0.3;
/// Trend boost factor for drift urgency (scales with trend_confidence).
const TREND_BOOST_FACTOR: f64 = 0.5;
/// Short-term drift threshold (per-sec) below which trend fallback activates.
const SHORT_DRIFT_FALLBACK_THRESHOLD: f64 = 0.0001;

/// CJP signal-aware reservation price shift (Cartea-Jaimungal-Penalva 2015).
///
/// Computes the optimal inventory-independent price shift from a mean-reverting signal:
///
/// ```text
/// signal_shift = α_t / (κ_ou + γ×κ_trade) × (1 - exp(-(κ_ou + γ×κ_trade)×(T-t)))
/// ```
///
/// Properties:
/// 1. **Self-bounding**: converges to α/(κ_ou + γ×κ_trade) as T→∞ — no manual caps needed
/// 2. **Execution-coupled**: higher fill rate → smaller shift (conserve in low liquidity)
/// 3. **Risk-coupled**: higher γ → smaller shift (risk-averse = conservative)
/// 4. **Sign-preserving**: positive α → positive shift, negative α → negative shift
///
/// # Arguments
/// * `alpha_t` - Current drift signal (per-second, signed)
/// * `kappa_ou` - OU mean-reversion rate of the signal process
/// * `gamma` - Risk aversion parameter
/// * `kappa_trade` - Fill intensity (fills/sec)
/// * `time_remaining` - Time to horizon (seconds)
fn cjp_signal_shift(
    alpha_t: f64,
    kappa_ou: f64,
    gamma: f64,
    kappa_trade: f64,
    time_remaining: f64,
) -> f64 {
    let denom = kappa_ou + gamma * kappa_trade;
    if denom < 1e-10 {
        return 0.0;
    }
    let asymptotic = alpha_t / denom;
    let decay = 1.0 - (-denom * time_remaining).exp();
    asymptotic * decay
}

impl HJBInventoryController {
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
        let terminal_skew = self.effective_terminal_penalty() * q * urgency;

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

    /// Compute predictive bias from changepoint probability.
    ///
    /// Returns a bias term β_t to add to the A-S skew formula:
    /// ```text
    /// δ^b = δ - γσ²qT/2 + β_t/2    (bid offset + predictive shift)
    /// δ^a = δ + γσ²qT/2 + β_t/2    (ask offset + predictive shift)
    /// ```
    ///
    /// Where β_t = -sensitivity × prob_excess × σ:
    /// - Negative β_t (predict down): Widen bids, narrow asks → sell aggressively
    /// - Positive β_t (predict up): Narrow bids, widen asks → buy aggressively
    ///
    /// # Arguments
    /// * `changepoint_prob` - Probability of regime change [0, 1] from BOCD
    /// * `sigma` - Current volatility estimate (per-second fractional)
    /// * `sensitivity` - Bias sensitivity (default: 2.0 = expect 2σ move)
    /// * `threshold` - Minimum prob to activate (default: 0.3)
    ///
    /// # Returns
    /// Predictive bias in fractional units. Negative = expect price to fall.
    pub fn predictive_bias(
        changepoint_prob: f64,
        sigma: f64,
        sensitivity: f64,
        threshold: f64,
    ) -> f64 {
        // Ramp function: only activate above baseline prob
        // Maps [threshold, 1.0] → [0.0, 1.0]
        if changepoint_prob <= threshold {
            return 0.0;
        }

        let prob_excess = (changepoint_prob - threshold) / (1.0 - threshold);

        // Negative bias = expect price to fall = widen bids, tighten asks
        // During regime changes (high changepoint prob), we expect downside moves
        // so we preemptively skew to reduce long exposure / avoid buying
        -sensitivity * prob_excess * sigma
    }

    /// Compute predictive bias with default parameters.
    ///
    /// Uses DEFAULT_PREDICTIVE_BIAS_SENSITIVITY (2.0) and DEFAULT_PREDICTIVE_BIAS_THRESHOLD (0.3).
    pub fn predictive_bias_default(changepoint_prob: f64, sigma: f64) -> f64 {
        Self::predictive_bias(
            changepoint_prob,
            sigma,
            DEFAULT_PREDICTIVE_BIAS_SENSITIVITY,
            DEFAULT_PREDICTIVE_BIAS_THRESHOLD,
        )
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
                predictive_bias: 0.0,
                variance_multiplier: 1.0,
                is_opposed: false,
                urgency_score: 0.0,
                ..Default::default()
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
                predictive_bias: 0.0,
                variance_multiplier: 1.0,
                is_opposed: false,
                urgency_score: 0.0,
                ..Default::default()
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
        let position_factor = (1.0 - MOMENTUM_THRESHOLD_POSITION_SLOPE * q.abs())
            .max(MOMENTUM_THRESHOLD_POSITION_FLOOR);
        let momentum_threshold = MOMENTUM_THRESHOLD_BASE_BPS * position_factor;
        let momentum_significant = momentum_bps.abs() > momentum_threshold;

        // Phase 3C: Soft continuation gate replaces binary cliff at p=0.5.
        // beta_continuation=-0.5 already provides continuous scaling.
        // Soft gate: ramp from 0 at SOFT_GATE_RAMP_START to 1.0 at min_continuation_prob
        let continuation_gate = if p_continuation >= self.config.min_continuation_prob {
            1.0
        } else if p_continuation > SOFT_GATE_RAMP_START {
            (p_continuation - SOFT_GATE_RAMP_START)
                / (self.config.min_continuation_prob - SOFT_GATE_RAMP_START)
        } else {
            0.0
        };

        if !is_opposed || !momentum_significant || continuation_gate < SOFT_GATE_MIN_ACTIVATION {
            // No drift adjustment needed
            return DriftAdjustedSkew {
                total_skew: base_skew,
                base_skew,
                drift_urgency: 0.0,
                predictive_bias: 0.0,
                variance_multiplier: 1.0,
                is_opposed,
                urgency_score: 0.0,
                ..Default::default()
            };
        }

        // === Compute Drift Urgency ===
        // From optimal control with drift: urgency ∝ μ × P(continue) × |q| × T
        let time_remaining = self.time_remaining().min(MAX_DRIFT_EXPOSURE_S);

        // Use EWMA-smoothed drift if warmed up, otherwise compute from raw momentum
        // Smoothed drift gives more stable signals and reduces whipsawing
        let drift_rate = if self.is_drift_warmed_up() {
            // Use smoothed drift for stable signals
            self.drift_ewma
        } else {
            // Fall back to raw momentum during warmup
            (momentum_bps / 10000.0) / 0.5
        };

        // CJP signal-aware skew: self-bounding, execution-coupled
        let raw_shift = cjp_signal_shift(
            drift_rate.abs(),
            self.config.cjp_kappa_ou,
            self.config.gamma_base,
            self.kappa_trade,
            time_remaining,
        );
        let drift_urgency = raw_shift * p_continuation * q.signum() * continuation_gate;

        // === Compute Variance Multiplier ===
        // Use EWMA-smoothed variance multiplier if warmed up for stability
        let variance_multiplier_capped = if self.is_drift_warmed_up() {
            // Use pre-computed smoothed variance multiplier
            self.variance_mult_ewma
        } else {
            // Compute inline during warmup
            // σ²_eff = σ² × (1 + κ × |momentum/σ| × P(continue))
            let momentum_vol_ratio = if self.sigma > 1e-10 {
                ((momentum_bps / 10000.0) / self.sigma)
                    .abs()
                    .min(MAX_MOMENTUM_VOL_RATIO)
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
            ((momentum_bps / 10000.0) / self.sigma)
                .abs()
                .min(MAX_MOMENTUM_VOL_RATIO)
        } else {
            0.0
        };

        // Urgency score for diagnostics [0, 5]
        let urgency_score = (momentum_bps.abs() / URGENCY_MOMENTUM_NORM_BPS).min(1.0) // Momentum strength
            + p_continuation // Continuation confidence
            + q.abs() // Position size
            + momentum_vol_ratio.min(1.0) // Vol-adjusted momentum
            + if self.is_terminal_zone() { 1.0 } else { 0.0 }; // Terminal zone boost

        // CJP diagnostics
        let cjp_denom = self.config.cjp_kappa_ou + self.config.gamma_base * self.kappa_trade;
        let cjp_asymptotic = if cjp_denom > 1e-10 {
            drift_rate.abs() / cjp_denom
        } else {
            0.0
        };

        DriftAdjustedSkew {
            total_skew: base_skew + drift_urgency,
            base_skew,
            drift_urgency,
            predictive_bias: 0.0,
            variance_multiplier: variance_multiplier_capped,
            is_opposed,
            urgency_score: urgency_score.min(5.0),
            cjp_signal_shift: drift_urgency.abs(),
            cjp_asymptotic_shift: cjp_asymptotic,
            cjp_kappa_trade: self.kappa_trade,
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
        result.cjp_signal_shift *= 10000.0;
        result.cjp_asymptotic_shift *= 10000.0;
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
                predictive_bias: 0.0,
                variance_multiplier: 1.0,
                is_opposed: false,
                urgency_score: 0.0,
                ..Default::default()
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
                predictive_bias: 0.0,
                variance_multiplier: 1.0,
                is_opposed: false,
                urgency_score: 0.0,
                ..Default::default()
            };
        };

        // === Enhanced Opposition Detection ===
        // Check if position opposes momentum at ANY timeframe, or is underwater

        // Short-term opposition (original logic)
        let short_opposed = q * momentum_bps < 0.0;

        // Medium-term opposition (30s window, requires some agreement)
        let medium_opposed =
            if trend.is_warmed_up && trend.timeframe_agreement > TREND_MIN_AGREEMENT {
                q * trend.medium_momentum_bps < 0.0
                    && trend.medium_momentum_bps.abs() > TREND_MEDIUM_OPPOSITION_BPS
            } else {
                false
            };

        // Long-term opposition (5min window, more authoritative)
        let long_opposed = if trend.is_warmed_up {
            q * trend.long_momentum_bps < 0.0
                && trend.long_momentum_bps.abs() > TREND_LONG_OPPOSITION_BPS
        } else {
            false
        };

        // Underwater opposition (position losing money in a sustained way)
        let underwater_opposed = trend.underwater_severity > TREND_UNDERWATER_THRESHOLD;

        // Combined opposition: any trigger counts
        let is_opposed = short_opposed || medium_opposed || long_opposed || underwater_opposed;

        // Position-dependent threshold (same as original)
        let position_factor = (1.0 - MOMENTUM_THRESHOLD_POSITION_SLOPE * q.abs())
            .max(MOMENTUM_THRESHOLD_POSITION_FLOOR);
        let momentum_threshold = MOMENTUM_THRESHOLD_BASE_BPS * position_factor;

        // Use the strongest momentum signal for threshold check
        let effective_momentum = if trend.is_warmed_up {
            // When long-term has a clear signal, use it
            if trend.long_momentum_bps.abs() > TREND_STRONG_LONG_BPS {
                trend.long_momentum_bps.abs()
            } else if trend.medium_momentum_bps.abs() > TREND_MEDIUM_DRIFT_BPS {
                trend.medium_momentum_bps.abs()
            } else {
                momentum_bps.abs()
            }
        } else {
            momentum_bps.abs()
        };

        let momentum_significant = effective_momentum > momentum_threshold;
        // Inverted soft gate: urgency ramps UP as continuation drops.
        // Low p_continuation means position is in trouble → max reduction pressure.
        // High p_continuation means position is healthy → no extra reduction pressure.
        let min_prob = self.config.min_continuation_prob;
        let reduction_urgency_gate = if p_continuation >= min_prob {
            0.0 // Position healthy, no extra reduction pressure
        } else if p_continuation > SOFT_GATE_RAMP_START {
            // Ramp from 0.0 (at min_prob) to 1.0 (at SOFT_GATE_RAMP_START)
            (min_prob - p_continuation) / (min_prob - SOFT_GATE_RAMP_START)
        } else {
            1.0 // Position strongly opposed, max reduction pressure
        };

        if !is_opposed || !momentum_significant || reduction_urgency_gate < SOFT_GATE_MIN_ACTIVATION
        {
            return DriftAdjustedSkew {
                total_skew: base_skew,
                base_skew,
                drift_urgency: 0.0,
                predictive_bias: 0.0,
                variance_multiplier: 1.0,
                is_opposed,
                urgency_score: 0.0,
                ..Default::default()
            };
        }

        // === Compute Drift Urgency (with trend boost) ===
        let time_remaining = self.time_remaining().min(MAX_DRIFT_EXPOSURE_S);

        // Use EWMA-smoothed drift if warmed up, with long-term trend fallback
        let drift_rate = if self.is_drift_warmed_up() {
            let short_drift = self.drift_ewma;
            // If short-term drift is near zero but long-term trend is strong and opposed,
            // use the long-term trend as drift signal. This prevents the MM from buying
            // into a smooth downtrend where short-term momentum oscillates near zero.
            if trend.is_warmed_up
                && short_drift.abs() < SHORT_DRIFT_FALLBACK_THRESHOLD
                && trend.long_momentum_bps.abs() > TREND_MEDIUM_DRIFT_BPS
                && is_opposed
            {
                // Long momentum is per 5min window, convert to per-second drift rate
                let long_drift = (trend.long_momentum_bps / 10000.0) / 300.0;
                // Also consider medium-term for faster response
                let med_drift = (trend.medium_momentum_bps / 10000.0) / 30.0;
                // Confidence-weighted blend: long is anchor, medium only trusted when aligned
                let long_confidence =
                    (trend.long_momentum_bps.abs() / TREND_STRONG_LONG_BPS).min(1.0);
                let med_confidence = (trend.medium_momentum_bps.abs() / TREND_MEDIUM_DRIFT_BPS)
                    .min(1.0)
                    * trend.timeframe_agreement;
                let total_confidence = long_confidence + med_confidence;
                if total_confidence > 0.001 {
                    (long_confidence * long_drift + med_confidence * med_drift) / total_confidence
                } else {
                    long_drift
                }
            } else {
                short_drift
            }
        } else {
            (momentum_bps / 10000.0) / 0.5
        };

        // Trend boost: multi-timeframe agreement amplifies drift urgency
        let trend_boost = if trend.is_warmed_up {
            1.0 + trend.trend_confidence * TREND_BOOST_FACTOR // 1.0 to 1.5 (conservative with CJP)
        } else {
            1.0
        };

        // CJP signal-aware skew: self-bounding, execution-coupled
        let raw_shift = cjp_signal_shift(
            drift_rate.abs(),
            self.config.cjp_kappa_ou,
            self.config.gamma_base,
            self.kappa_trade,
            time_remaining,
        );
        let drift_urgency = raw_shift * trend_boost * q.signum() * reduction_urgency_gate;

        // === Compute Variance Multiplier ===
        let variance_multiplier_capped = if self.is_drift_warmed_up() {
            self.variance_mult_ewma * trend_boost.min(1.5) // Additional boost, capped
        } else {
            let momentum_vol_ratio = if self.sigma > 1e-10 {
                ((momentum_bps / 10000.0) / self.sigma)
                    .abs()
                    .min(MAX_MOMENTUM_VOL_RATIO)
            } else {
                0.0
            };
            let variance_multiplier = 1.0
                + self.config.opposition_sensitivity
                    * momentum_vol_ratio
                    * reduction_urgency_gate
                    * q.abs();
            variance_multiplier.min(self.config.max_drift_urgency)
        };

        // Urgency score for diagnostics
        let momentum_vol_ratio = if self.sigma > 1e-10 {
            ((momentum_bps / 10000.0) / self.sigma)
                .abs()
                .min(MAX_MOMENTUM_VOL_RATIO)
        } else {
            0.0
        };

        let urgency_score = (momentum_bps.abs() / URGENCY_MOMENTUM_NORM_BPS).min(1.0)
            + reduction_urgency_gate
            + q.abs()
            + momentum_vol_ratio.min(1.0)
            + if self.is_terminal_zone() { 1.0 } else { 0.0 }
            + trend.trend_confidence; // Add trend confidence to score

        // CJP diagnostics
        let cjp_denom = self.config.cjp_kappa_ou + self.config.gamma_base * self.kappa_trade;
        let cjp_asymptotic = if cjp_denom > 1e-10 {
            drift_rate.abs() / cjp_denom
        } else {
            0.0
        };

        DriftAdjustedSkew {
            total_skew: base_skew + drift_urgency,
            base_skew,
            drift_urgency,
            predictive_bias: 0.0,
            variance_multiplier: variance_multiplier_capped,
            is_opposed,
            urgency_score: urgency_score.min(6.0), // Max now 6.0 with trend
            cjp_signal_shift: drift_urgency.abs(),
            cjp_asymptotic_shift: cjp_asymptotic,
            cjp_kappa_trade: self.kappa_trade,
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
        result.cjp_signal_shift *= 10000.0;
        result.cjp_asymptotic_shift *= 10000.0;
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
        if self.effective_terminal_penalty().abs() < 1e-10 {
            return 0.0;
        }

        let funding_per_second = self.funding_rate_ewma / (365.0 * 24.0 * 3600.0);

        // Target = -funding / (2 × penalty)
        let target = -funding_per_second / (2.0 * self.effective_terminal_penalty());

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
        let terminal_cost = 2.0 * self.effective_terminal_penalty() * q * urgency;
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
            funding_horizon_active: self.is_funding_horizon_active(),
            effective_terminal_penalty: self.effective_terminal_penalty(),
        }
    }

    // === Queue Value Methods (Phase 2: Churn Reduction) ===

    /// Compute the queue value for an order at a given depth.
    ///
    /// # Arguments
    /// * `queue_depth` - Number of units ahead in queue
    /// * `half_spread_bps` - Half-spread in basis points (reward for filling)
    ///
    /// # Returns
    /// Queue value in basis points. Higher = more valuable to preserve.
    ///
    /// # Formula
    /// ```text
    /// v(q) = (s/2) × exp(-α×q) - β×q
    /// ```
    pub fn queue_value(&self, queue_depth: f64, half_spread_bps: f64) -> f64 {
        if !self.config.use_queue_value {
            return 0.0;
        }

        let alpha = self.config.queue_alpha;
        let beta = self.config.queue_beta;

        // Exponential term: expected fill value decays with queue depth
        let exp_term = (half_spread_bps / 2.0) * (-alpha * queue_depth).exp();

        // Linear term: opportunity cost of maintaining queue position
        let lin_term = beta * queue_depth;

        // Queue value = expected value minus cost
        (exp_term - lin_term).max(0.0)
    }

    /// Check if an order should be preserved based on its queue value.
    ///
    /// # Arguments
    /// * `queue_depth` - Number of units ahead in queue
    /// * `half_spread_bps` - Half-spread in basis points
    ///
    /// # Returns
    /// True if the order's queue value exceeds the modify cost threshold.
    pub fn should_preserve_queue(&self, queue_depth: f64, half_spread_bps: f64) -> bool {
        if !self.config.use_queue_value {
            return false; // Disabled = don't preserve
        }

        self.queue_value(queue_depth, half_spread_bps) >= self.config.queue_modify_cost_bps
    }

    /// Compute optimal skew with queue position consideration.
    ///
    /// This extends the standard optimal skew by adding a term that accounts
    /// for the value of the current queue position. If the order has significant
    /// queue value, the controller recommends smaller adjustments to preserve
    /// the queue position.
    ///
    /// # Arguments
    /// * `position` - Current inventory position
    /// * `max_position` - Maximum position for normalization
    /// * `queue_depth` - Current queue depth ahead
    /// * `current_spread_bps` - Current spread in basis points
    ///
    /// # Returns
    /// Tuple of (optimal_skew, queue_value_bps, should_preserve)
    pub fn optimal_skew_with_queue(
        &self,
        position: f64,
        max_position: f64,
        queue_depth: f64,
        current_spread_bps: f64,
    ) -> (f64, f64, bool) {
        let base_skew = self.optimal_skew(position, max_position);
        let half_spread = current_spread_bps / 2.0;
        let queue_val = self.queue_value(queue_depth, half_spread);
        let should_preserve = self.should_preserve_queue(queue_depth, half_spread);

        // If queue is valuable, dampen the skew adjustment
        let adjusted_skew = if should_preserve && self.config.use_queue_value {
            // Reduce skew magnitude when queue is valuable
            // The more valuable the queue, the less we want to move
            let damping = 1.0 - (queue_val / (queue_val + self.config.queue_modify_cost_bps));
            base_skew * damping
        } else {
            base_skew
        };

        (adjusted_skew, queue_val, should_preserve)
    }

    /// Compute the break-even queue depth for order preservation.
    ///
    /// Orders at or beyond this depth have no economic value to preserve.
    ///
    /// # Arguments
    /// * `half_spread_bps` - Half-spread in basis points
    pub fn break_even_queue_depth(&self, half_spread_bps: f64) -> f64 {
        // Solve: (s/2) × exp(-α×q) - β×q = modify_cost
        // Using Newton-Raphson iteration

        let target = self.config.queue_modify_cost_bps;
        let s2 = half_spread_bps / 2.0;
        let alpha = self.config.queue_alpha;
        let beta = self.config.queue_beta;

        // Initial guess
        let mut q = if s2 > target {
            -((2.0 * target) / s2).ln() / alpha
        } else {
            0.0
        };

        // Newton-Raphson iterations
        for _ in 0..10 {
            let exp_aq = (-alpha * q).exp();
            let f = s2 * exp_aq - beta * q - target;
            let f_prime = -alpha * s2 * exp_aq - beta;

            if f_prime.abs() < 1e-10 {
                break;
            }

            let delta = f / f_prime;
            q -= delta;

            if delta.abs() < 1e-6 {
                break;
            }
        }

        q.clamp(0.0, 100.0) // Cap at 100 units
    }

    /// Get queue value configuration parameters for diagnostics.
    pub fn queue_config(&self) -> (f64, f64, f64, bool) {
        (
            self.config.queue_alpha,
            self.config.queue_beta,
            self.config.queue_modify_cost_bps,
            self.config.use_queue_value,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::process_models::hjb::config::HJBConfig;

    /// Build a warmed-up controller with drift_ewma = 0 so the trend fallback path activates.
    fn make_controller() -> HJBInventoryController {
        let config = HJBConfig {
            use_ou_drift: false,
            use_drift_adjusted_skew: true,
            min_continuation_prob: 0.5,
            ..HJBConfig::default()
        };
        let mut ctrl = HJBInventoryController::new(config);
        ctrl.drift_update_count = 100; // well past min_warmup (20)
        ctrl.drift_ewma = 0.0; // near-zero so trend fallback activates
        ctrl.sigma = 0.001; // 10 bps/s
        ctrl
    }

    fn make_trend(long_bps: f64, med_bps: f64, agreement: f64) -> TrendSignal {
        TrendSignal {
            long_momentum_bps: long_bps,
            medium_momentum_bps: med_bps,
            timeframe_agreement: agreement,
            trend_confidence: agreement,
            is_warmed_up: true,
            ..TrendSignal::default()
        }
    }

    #[test]
    fn test_drift_blend_long_dominates_when_aligned() {
        let ctrl = make_controller();
        // Long = -20 bps (strong), medium = -3 bps (weak), fully aligned
        let trend = make_trend(-20.0, -3.0, 1.0);
        // Position positive, momentum negative => opposed
        let result = ctrl.optimal_skew_with_trend(5.0, 10.0, -15.0, 0.3, &trend);

        // With long_bps=-20: long_confidence = (20/10).min(1) = 1.0
        // With med_bps=-3: med_confidence = (3/5).min(1) * 1.0 = 0.6
        // long_drift = (-20/10000)/300 = -6.67e-6
        // med_drift = (-3/10000)/30 = -1.0e-5
        // blend = (1.0 * -6.67e-6 + 0.6 * -1.0e-5) / 1.6 = -7.92e-6
        assert!(
            result.drift_urgency != 0.0,
            "drift urgency should be nonzero"
        );
        assert!(result.is_opposed, "should detect opposition");
    }

    #[test]
    fn test_drift_blend_medium_weight_when_aligned_and_stronger() {
        let ctrl = make_controller();
        // Long = -8 bps (moderate), medium = -10 bps (strong, fast), fully aligned
        let trend = make_trend(-8.0, -10.0, 1.0);
        let result_blended = ctrl.optimal_skew_with_trend(5.0, 10.0, -15.0, 0.3, &trend);

        // Now test with zero agreement => medium ignored
        let trend_disagree = make_trend(-8.0, -10.0, 0.0);
        let result_long_only = ctrl.optimal_skew_with_trend(5.0, 10.0, -15.0, 0.3, &trend_disagree);

        // Blended urgency should be larger because medium pulls drift faster
        assert!(
            result_blended.drift_urgency.abs() > result_long_only.drift_urgency.abs(),
            "blended drift urgency ({}) should exceed long-only ({}) when medium is stronger and aligned",
            result_blended.drift_urgency, result_long_only.drift_urgency,
        );
    }

    #[test]
    fn test_drift_blend_ignores_medium_when_disagreeing() {
        let ctrl = make_controller();
        // Long = -15 bps, medium = +8 bps (opposite direction), zero agreement
        let trend_disagree = make_trend(-15.0, 8.0, 0.0);
        let result_disagree = ctrl.optimal_skew_with_trend(5.0, 10.0, -15.0, 0.3, &trend_disagree);

        // Same scenario but with full agreement — medium gets weight and dilutes long
        let trend_agree = make_trend(-15.0, 8.0, 1.0);
        let _result_agree = ctrl.optimal_skew_with_trend(5.0, 10.0, -15.0, 0.3, &trend_agree);

        // When disagreeing (agreement=0), med_confidence=0, blend = pure long_drift
        // long_drift = (-15/10000)/300 = -5e-6
        // When agreeing (agreement=1), med gets weight and pushes drift toward positive
        // med_drift = (+8/10000)/30 = +2.67e-5 (10x normalization advantage)
        // So the disagreeing case should preserve the long signal more faithfully.
        //
        // Key invariant: with disagreement, drift urgency should be nonzero
        // (pure long drift) and the sign should match the long direction.
        assert!(
            result_disagree.drift_urgency != 0.0,
            "with disagreeing medium zeroed out, long drift should still produce urgency",
        );

        // The disagreeing case preserves long direction; the agreeing case lets
        // the faster medium window (10x normalization advantage) dilute or reverse it.
        // Verify: disagreeing result is closer to pure-long behavior
        // (pure long drift is negative for long=-15 bps, so urgency sign should be positive
        // because position is positive and drift is negative => opposed => positive urgency)
        let long_only_trend = make_trend(-15.0, 0.0, 0.0);
        let result_long_only =
            ctrl.optimal_skew_with_trend(5.0, 10.0, -15.0, 0.3, &long_only_trend);

        // Disagreeing result should match long-only result (medium is zeroed)
        assert!(
            (result_disagree.drift_urgency - result_long_only.drift_urgency).abs() < 1e-10,
            "disagreeing medium (urgency={}) should equal long-only (urgency={})",
            result_disagree.drift_urgency,
            result_long_only.drift_urgency,
        );
    }

    // === Inverted soft gate tests ===

    #[test]
    fn test_inverted_gate_increases_urgency_as_continuation_drops() {
        let ctrl = make_controller();
        let trend = make_trend(-20.0, -10.0, 1.0);

        // p=0.45: gate = (0.5-0.45)/(0.5-0.2) = 0.167
        let result_moderate = ctrl.optimal_skew_with_trend(5.0, 10.0, -15.0, 0.45, &trend);
        // p=0.3: gate = (0.5-0.3)/(0.5-0.2) = 0.667
        let result_low = ctrl.optimal_skew_with_trend(5.0, 10.0, -15.0, 0.3, &trend);

        assert!(
            result_low.drift_urgency.abs() > result_moderate.drift_urgency.abs(),
            "lower continuation ({}) should produce more urgency than moderate ({})",
            result_low.drift_urgency,
            result_moderate.drift_urgency,
        );
    }

    #[test]
    fn test_inverted_gate_zero_urgency_when_position_healthy() {
        let ctrl = make_controller();
        let trend = make_trend(-20.0, -10.0, 1.0);

        // p=0.6 >= min_prob=0.5 → gate = 0.0 → no urgency
        let result = ctrl.optimal_skew_with_trend(5.0, 10.0, -15.0, 0.6, &trend);

        assert!(
            result.drift_urgency.abs() < 1e-12,
            "healthy position (p=0.6) should have zero drift urgency, got {}",
            result.drift_urgency,
        );
    }

    #[test]
    fn test_inverted_gate_max_urgency_at_extreme() {
        let ctrl = make_controller();
        let trend = make_trend(-20.0, -10.0, 1.0);

        // p=0.15 < 0.2 → gate = 1.0 (max)
        let result_extreme = ctrl.optimal_skew_with_trend(5.0, 10.0, -15.0, 0.15, &trend);
        // p=0.35: gate = (0.5-0.35)/(0.5-0.2) = 0.5
        let result_mid = ctrl.optimal_skew_with_trend(5.0, 10.0, -15.0, 0.35, &trend);

        assert!(
            result_extreme.drift_urgency.abs() > result_mid.drift_urgency.abs(),
            "extreme opposition (p=0.15) should exceed mid (p=0.35): extreme={}, mid={}",
            result_extreme.drift_urgency,
            result_mid.drift_urgency,
        );

        // Gate at p=0.15 should be 1.0 and at p=0.1 should also be 1.0 (both below 0.2)
        let result_very_extreme = ctrl.optimal_skew_with_trend(5.0, 10.0, -15.0, 0.1, &trend);
        assert!(
            (result_extreme.drift_urgency - result_very_extreme.drift_urgency).abs() < 1e-12,
            "both below 0.2 should have same max urgency: p=0.15={}, p=0.1={}",
            result_extreme.drift_urgency,
            result_very_extreme.drift_urgency,
        );
    }

    // === CJP Signal-Aware Tests ===

    /// Build a CJP-enabled controller with known kappa_trade.
    fn make_cjp_controller() -> HJBInventoryController {
        let config = HJBConfig {
            use_ou_drift: false,
            use_drift_adjusted_skew: true,
            use_cjp_signal: true,
            cjp_kappa_trade_default: 0.5,
            cjp_kappa_ou: 0.1,
            gamma_base: 0.3,
            min_continuation_prob: 0.5,
            ..HJBConfig::default()
        };
        let mut ctrl = HJBInventoryController::new(config);
        ctrl.drift_update_count = 100;
        ctrl.drift_ewma = 0.001; // 10 bps/s drift
        ctrl.sigma = 0.001;
        ctrl
    }

    #[test]
    fn test_cjp_shift_bounded_by_asymptotic() {
        // CJP shift must be bounded by α/(κ_ou + γ×κ_trade) for any T.
        let alpha = 0.001; // 10 bps/s
        let kappa_ou = 0.1;
        let gamma = 0.3;
        let kappa_trade = 0.5;
        let asymptotic = alpha / (kappa_ou + gamma * kappa_trade);

        // Test across a wide range of time horizons
        for &t in &[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0] {
            let shift = cjp_signal_shift(alpha, kappa_ou, gamma, kappa_trade, t);
            assert!(
                shift <= asymptotic + 1e-12,
                "CJP shift ({}) should be bounded by asymptotic ({}) at T={}",
                shift,
                asymptotic,
                t,
            );
            assert!(
                shift >= 0.0,
                "CJP shift should be non-negative for positive alpha, got {} at T={}",
                shift,
                t,
            );
        }
    }

    #[test]
    fn test_cjp_higher_kappa_trade_reduces_shift() {
        let alpha = 0.001;
        let kappa_ou = 0.1;
        let gamma = 0.3;
        let time_remaining = 60.0;

        let shift_low_kappa = cjp_signal_shift(alpha, kappa_ou, gamma, 0.1, time_remaining);
        let shift_high_kappa = cjp_signal_shift(alpha, kappa_ou, gamma, 2.0, time_remaining);

        assert!(
            shift_low_kappa > shift_high_kappa,
            "higher kappa_trade ({}) should reduce shift: low_k={}, high_k={}",
            2.0,
            shift_low_kappa,
            shift_high_kappa,
        );
    }

    #[test]
    fn test_cjp_zero_drift_zero_shift() {
        let shift = cjp_signal_shift(0.0, 0.1, 0.3, 0.5, 60.0);
        assert!(
            shift.abs() < 1e-15,
            "zero drift should produce zero shift, got {}",
            shift,
        );
    }

    #[test]
    fn test_cjp_positive_drift_positive_shift() {
        let shift = cjp_signal_shift(0.001, 0.1, 0.3, 0.5, 60.0);
        assert!(
            shift > 0.0,
            "positive drift should produce positive shift, got {}",
            shift,
        );
    }

    #[test]
    fn test_cjp_with_reduction_gate_zero_yields_zero() {
        let ctrl = make_cjp_controller();
        let trend = make_trend(-20.0, -10.0, 1.0);

        // p=0.6 >= min_prob=0.5 → reduction_urgency_gate = 0.0 → zero urgency
        let result = ctrl.optimal_skew_with_trend(5.0, 10.0, -15.0, 0.6, &trend);
        assert!(
            result.drift_urgency.abs() < 1e-12,
            "CJP with reduction_urgency_gate=0 should yield zero drift urgency, got {}",
            result.drift_urgency,
        );
        // CJP diagnostics should also be zero when urgency is zero
        assert!(
            result.cjp_signal_shift.abs() < 1e-12,
            "CJP signal shift diagnostic should be zero when gate=0, got {}",
            result.cjp_signal_shift,
        );
    }

    #[test]
    fn test_cjp_diagnostics_populated_when_active() {
        let ctrl = make_cjp_controller();
        // p=0.3 → reduction_urgency_gate = (0.5-0.3)/(0.5-0.2) ≈ 0.667
        // Position positive, momentum negative => opposed
        let trend = make_trend(-20.0, -10.0, 1.0);
        let result = ctrl.optimal_skew_with_trend(5.0, 10.0, -15.0, 0.3, &trend);

        assert!(
            result.cjp_signal_shift > 0.0,
            "CJP signal shift should be populated, got {}",
            result.cjp_signal_shift,
        );
        assert!(
            result.cjp_asymptotic_shift > 0.0,
            "CJP asymptotic shift should be populated, got {}",
            result.cjp_asymptotic_shift,
        );
        assert!(
            result.cjp_kappa_trade > 0.0,
            "CJP kappa_trade should be populated, got {}",
            result.cjp_kappa_trade,
        );
    }

    #[test]
    fn test_cjp_diagnostics_populated() {
        // CJP is always on — verify diagnostics are populated
        let ctrl = make_controller();
        let trend = make_trend(-20.0, -10.0, 1.0);
        let result = ctrl.optimal_skew_with_trend(5.0, 10.0, -15.0, 0.3, &trend);

        // CJP diagnostics should be populated (kappa_trade from default config)
        assert!(
            result.cjp_kappa_trade > 0.0,
            "CJP kappa_trade should be populated, got {}",
            result.cjp_kappa_trade,
        );
    }

    #[test]
    fn test_cjp_shift_convergence_at_large_t() {
        // As T → ∞, shift should converge to α/(κ_ou + γ×κ_trade)
        let alpha = 0.001;
        let kappa_ou = 0.1;
        let gamma = 0.3;
        let kappa_trade = 0.5;
        let asymptotic = alpha / (kappa_ou + gamma * kappa_trade);

        let shift_large_t = cjp_signal_shift(alpha, kappa_ou, gamma, kappa_trade, 100000.0);
        let relative_error = (shift_large_t - asymptotic).abs() / asymptotic;
        assert!(
            relative_error < 1e-6,
            "CJP shift at large T should converge to asymptotic: shift={}, asymptotic={}, error={}",
            shift_large_t,
            asymptotic,
            relative_error,
        );
    }

    #[test]
    fn test_cjp_degenerate_denom_returns_zero() {
        // When κ_ou ≈ 0 and γ×κ_trade ≈ 0, should return 0 (not NaN/Inf)
        let shift = cjp_signal_shift(0.001, 0.0, 0.0, 0.0, 60.0);
        assert!(
            shift.abs() < 1e-12,
            "degenerate denominator should produce zero, got {}",
            shift,
        );
        assert!(!shift.is_nan(), "should not be NaN");
        assert!(!shift.is_infinite(), "should not be infinite");
    }

    #[test]
    fn test_update_kappa_trade_ewma_smoothing() {
        let mut ctrl = make_cjp_controller();
        let initial = ctrl.kappa_trade;
        assert!(
            (initial - 0.5).abs() < 1e-10,
            "initial kappa_trade should be 0.5"
        );

        // Update with a much higher value — EWMA should smooth
        ctrl.update_kappa_trade(5.0);
        let after_one = ctrl.kappa_trade;
        assert!(
            after_one > initial,
            "kappa_trade should increase after high observation"
        );
        assert!(
            after_one < 5.0,
            "kappa_trade should be smoothed, not jump to 5.0"
        );

        // Expected: 0.05 * 5.0 + 0.95 * 0.5 = 0.25 + 0.475 = 0.725
        assert!(
            (after_one - 0.725).abs() < 1e-10,
            "expected EWMA value 0.725, got {}",
            after_one,
        );
    }
}
