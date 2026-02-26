//! Direction Hysteresis — prevents immediate re-accumulation after reducing to flat.
//!
//! After spending 5+ minutes reducing a position to flat, the system should not
//! immediately re-accumulate in the same direction without new information.
//!
//! Mechanism: After a zero-crossing, penalize the side that would re-accumulate
//! the previous direction via gamma scaling (exponential decay).
//!
//! Example: Was long → penalize bids (re-accumulating long) for cooldown period.

/// Direction hysteresis state for preventing uninformed re-accumulation.
#[derive(Debug, Clone)]
pub struct DirectionHysteresis {
    /// Current position sign (+1, -1, or 0).
    last_sign: i8,
    /// Sign of the direction to penalize (the OLD direction before zero-crossing).
    /// +1 means "was long, penalize bids (re-accumulating long)".
    /// -1 means "was short, penalize asks (re-accumulating short)".
    penalized_sign: i8,
    /// Timestamp (ms) of the last zero-crossing.
    zero_cross_time_ms: u64,
    /// Cooldown time constant (ms). Penalty decays as exp(-elapsed/tau).
    cooldown_tau_ms: f64,
    /// Hysteresis strength: peak gamma multiplier at zero-crossing.
    /// Penalty = 1 + strength × exp(-elapsed/tau).
    hysteresis_strength: f64,
}

/// Per-side gamma adjustments from hysteresis.
#[derive(Debug, Clone, Copy)]
pub struct HysteresisAdjustment {
    /// Gamma multiplier for bid side (>1.0 = wider bids).
    pub bid_gamma_mult: f64,
    /// Gamma multiplier for ask side (>1.0 = wider asks).
    pub ask_gamma_mult: f64,
}

impl DirectionHysteresis {
    /// Create with default parameters (90s cooldown, 1.0 strength).
    pub fn new() -> Self {
        Self {
            last_sign: 0,
            penalized_sign: 0,
            zero_cross_time_ms: 0,
            cooldown_tau_ms: 90_000.0, // 90 seconds
            hysteresis_strength: 1.0,  // Peak: 2x gamma on re-accumulating side
        }
    }

    /// Create with custom parameters.
    pub fn with_params(cooldown_tau_ms: f64, hysteresis_strength: f64) -> Self {
        Self {
            last_sign: 0,
            penalized_sign: 0,
            zero_cross_time_ms: 0,
            cooldown_tau_ms: cooldown_tau_ms.max(1000.0),
            hysteresis_strength: hysteresis_strength.clamp(0.0, 5.0),
        }
    }

    /// Update position and detect zero-crossings.
    /// Call after every fill or position update.
    pub fn update_position(&mut self, position: f64, timestamp_ms: u64) {
        let current_sign = if position > 0.01 {
            1
        } else if position < -0.01 {
            -1
        } else {
            0
        };

        // Detect zero-crossing: position changed sign (through or to zero)
        if self.last_sign != 0 && current_sign != self.last_sign {
            tracing::debug!(
                from_sign = self.last_sign,
                to_sign = current_sign,
                position = %format!("{:.4}", position),
                "HYSTERESIS: zero-crossing detected"
            );
            // Record the old direction as the one to penalize
            self.penalized_sign = self.last_sign;
            self.zero_cross_time_ms = timestamp_ms;
            // Update last_sign to current direction
            self.last_sign = current_sign;
        } else if self.last_sign == 0 && current_sign != 0 {
            // First position taken — just record direction
            self.last_sign = current_sign;
        }
    }

    /// Get the penalized direction sign.
    /// +1 = was long (penalizing bids), -1 = was short (penalizing asks), 0 = no penalty.
    pub fn penalized_sign(&self) -> i8 {
        self.penalized_sign
    }

    /// Get per-side gamma adjustments.
    /// Penalty applied to the side that would re-accumulate the previous direction.
    /// If `suppress` is true (informed flip detected), returns neutral 1.0 multipliers.
    pub fn gamma_adjustments(&self, now_ms: u64, suppress: bool) -> HysteresisAdjustment {
        if suppress || self.zero_cross_time_ms == 0 || self.penalized_sign == 0 {
            return HysteresisAdjustment {
                bid_gamma_mult: 1.0,
                ask_gamma_mult: 1.0,
            };
        }

        let elapsed_ms = now_ms.saturating_sub(self.zero_cross_time_ms) as f64;
        let decay = (-elapsed_ms / self.cooldown_tau_ms).exp();
        let penalty = 1.0 + self.hysteresis_strength * decay;

        // Penalized sign is the OLD direction before crossing:
        // If was long (penalized_sign=+1): penalize bids (buying = re-accumulating long)
        // If was short (penalized_sign=-1): penalize asks (selling = re-accumulating short)
        if self.penalized_sign > 0 {
            HysteresisAdjustment {
                bid_gamma_mult: penalty,
                ask_gamma_mult: 1.0,
            }
        } else {
            HysteresisAdjustment {
                bid_gamma_mult: 1.0,
                ask_gamma_mult: penalty,
            }
        }
    }

    /// Check if hysteresis is actively penalizing (for logging).
    pub fn is_active(&self, now_ms: u64) -> bool {
        if self.zero_cross_time_ms == 0 || self.penalized_sign == 0 {
            return false;
        }
        let elapsed_ms = now_ms.saturating_sub(self.zero_cross_time_ms) as f64;
        let decay = (-elapsed_ms / self.cooldown_tau_ms).exp();
        decay > 0.05 // Active if penalty > 5%
    }
}

impl Default for DirectionHysteresis {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_hysteresis_initially() {
        let h = DirectionHysteresis::new();
        let adj = h.gamma_adjustments(1000, false);
        assert_eq!(adj.bid_gamma_mult, 1.0);
        assert_eq!(adj.ask_gamma_mult, 1.0);
    }

    #[test]
    fn test_hysteresis_after_long_to_flat() {
        let mut h = DirectionHysteresis::new();
        // Build long position
        h.update_position(1.0, 1000);
        // Cross to flat/short
        h.update_position(-0.1, 2000);

        // Immediately after crossing: penalty on bids (re-accumulating long)
        let adj = h.gamma_adjustments(2000, false);
        assert!(
            adj.bid_gamma_mult > 1.5,
            "Bids should be penalized after long→short"
        );
        assert_eq!(adj.ask_gamma_mult, 1.0, "Asks should be unaffected");
    }

    #[test]
    fn test_hysteresis_after_short_to_flat() {
        let mut h = DirectionHysteresis::new();
        h.update_position(-1.0, 1000);
        h.update_position(0.1, 2000);

        let adj = h.gamma_adjustments(2000, false);
        assert_eq!(adj.bid_gamma_mult, 1.0);
        assert!(
            adj.ask_gamma_mult > 1.5,
            "Asks should be penalized after short→long"
        );
    }

    #[test]
    fn test_hysteresis_decays() {
        let mut h = DirectionHysteresis::new();
        h.update_position(1.0, 1000);
        h.update_position(-0.1, 2000);

        // At t=2000 (crossing): full penalty
        let adj_immediate = h.gamma_adjustments(2000, false);
        // At t=92000 (90s later, ≈1τ): penalty ≈ 37%
        let adj_later = h.gamma_adjustments(92_000, false);
        // At t=272000 (270s, ≈3τ): penalty ≈ 5%
        let adj_much_later = h.gamma_adjustments(272_000, false);

        assert!(adj_immediate.bid_gamma_mult > adj_later.bid_gamma_mult);
        assert!(adj_later.bid_gamma_mult > adj_much_later.bid_gamma_mult);
        assert!(
            adj_much_later.bid_gamma_mult < 1.1,
            "Should have decayed substantially"
        );
    }

    #[test]
    fn test_suppress_returns_neutral() {
        let mut h = DirectionHysteresis::new();
        h.update_position(1.0, 1000);
        h.update_position(-0.1, 2000);

        // Without suppress: penalty active
        let adj_normal = h.gamma_adjustments(2000, false);
        assert!(adj_normal.bid_gamma_mult > 1.5);

        // With suppress (informed flip): neutral multipliers
        let adj_suppressed = h.gamma_adjustments(2000, true);
        assert_eq!(adj_suppressed.bid_gamma_mult, 1.0);
        assert_eq!(adj_suppressed.ask_gamma_mult, 1.0);
    }

    #[test]
    fn test_penalized_sign_getter() {
        let mut h = DirectionHysteresis::new();
        assert_eq!(h.penalized_sign(), 0);

        h.update_position(1.0, 1000);
        assert_eq!(h.penalized_sign(), 0); // No crossing yet

        h.update_position(-0.1, 2000);
        assert_eq!(h.penalized_sign(), 1); // Was long → penalize bids
    }
}
