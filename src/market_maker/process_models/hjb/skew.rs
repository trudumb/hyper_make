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

        // Accept 0.5 prior during warmup - better to react than ignore
        let continuation_confident = p_continuation >= self.config.min_continuation_prob;

        if !is_opposed || !momentum_significant || !continuation_confident {
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
        let continuation_confident = p_continuation >= self.config.min_continuation_prob;

        if !is_opposed || !momentum_significant || !continuation_confident {
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
