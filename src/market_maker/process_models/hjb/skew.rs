//! Optimal skew calculations for HJB controller.

use crate::market_maker::estimator::TrendSignal;

use super::controller::HJBInventoryController;
use super::summary::{DriftAdjustedSkew, HJBSummary};

/// Default predictive bias threshold (minimum changepoint_prob to activate).
const DEFAULT_PREDICTIVE_BIAS_THRESHOLD: f64 = 0.3;
/// Default predictive bias sensitivity (σ multiplier).
const DEFAULT_PREDICTIVE_BIAS_SENSITIVITY: f64 = 2.0;

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

        // Phase 3C: Soft continuation gate replaces binary cliff at p=0.5.
        // beta_continuation=-0.5 already provides continuous scaling.
        // Soft gate: ramp from 0 at p=0.2 to 1.0 at min_continuation_prob
        let continuation_gate = if p_continuation >= self.config.min_continuation_prob {
            1.0
        } else if p_continuation > 0.2 {
            (p_continuation - 0.2) / (self.config.min_continuation_prob - 0.2)
        } else {
            0.0
        };

        if !is_opposed || !momentum_significant || continuation_gate < 0.01 {
            // No drift adjustment needed
            return DriftAdjustedSkew {
                total_skew: base_skew,
                base_skew,
                drift_urgency: 0.0,
                predictive_bias: 0.0,
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
        // Phase 3C: Scale by continuation_gate for smooth ramp-in
        let drift_urgency = drift_urgency_magnitude * q.signum() * continuation_gate;

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
            predictive_bias: 0.0,
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
                predictive_bias: 0.0,
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
                predictive_bias: 0.0,
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
        // Inverted soft gate: urgency ramps UP as continuation drops.
        // Low p_continuation means position is in trouble → max reduction pressure.
        // High p_continuation means position is healthy → no extra reduction pressure.
        let min_prob = self.config.min_continuation_prob;
        let reduction_urgency_gate = if p_continuation >= min_prob {
            0.0 // Position healthy, no extra reduction pressure
        } else if p_continuation > 0.2 {
            // Ramp from 0.0 (at min_prob) to 1.0 (at 0.2)
            (min_prob - p_continuation) / (min_prob - 0.2)
        } else {
            1.0 // Position strongly opposed, max reduction pressure
        };

        if !is_opposed || !momentum_significant || reduction_urgency_gate < 0.01 {
            return DriftAdjustedSkew {
                total_skew: base_skew,
                base_skew,
                drift_urgency: 0.0,
                predictive_bias: 0.0,
                variance_multiplier: 1.0,
                is_opposed,
                urgency_score: 0.0,
            };
        }

        // === Compute Drift Urgency (with trend boost) ===
        let time_remaining = self.time_remaining().min(300.0);

        // Use EWMA-smoothed drift if warmed up, with long-term trend fallback
        let drift_rate = if self.is_drift_warmed_up() {
            let short_drift = self.drift_ewma;
            // If short-term drift is near zero but long-term trend is strong and opposed,
            // use the long-term trend as drift signal. This prevents the MM from buying
            // into a smooth downtrend where short-term momentum oscillates near zero.
            if trend.is_warmed_up
                && short_drift.abs() < 0.0001
                && trend.long_momentum_bps.abs() > 5.0
                && is_opposed
            {
                // Long momentum is per 5min window, convert to per-second drift rate
                let long_drift = (trend.long_momentum_bps / 10000.0) / 300.0;
                // Also consider medium-term for faster response
                let med_drift = (trend.medium_momentum_bps / 10000.0) / 30.0;
                // Confidence-weighted blend: long is anchor, medium only trusted when aligned
                let long_confidence = (trend.long_momentum_bps.abs() / 10.0).min(1.0);
                let med_confidence =
                    (trend.medium_momentum_bps.abs() / 5.0).min(1.0) * trend.timeframe_agreement;
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

        // Base urgency calculation (gate handles p_continuation scaling)
        let raw_urgency =
            self.config.opposition_sensitivity * drift_rate.abs() * q.abs() * time_remaining;

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
        let drift_urgency = drift_urgency_magnitude * q.signum() * reduction_urgency_gate;

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
                    * reduction_urgency_gate
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
            + reduction_urgency_gate
            + q.abs()
            + momentum_vol_ratio.min(1.0)
            + if self.is_terminal_zone() { 1.0 } else { 0.0 }
            + trend.trend_confidence; // Add trend confidence to score

        DriftAdjustedSkew {
            total_skew: base_skew + drift_urgency,
            base_skew,
            drift_urgency,
            predictive_bias: 0.0,
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
}
