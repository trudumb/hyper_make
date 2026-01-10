//! Volatility regime classification and tracking.
//!
//! Four-state regime with asymmetric hysteresis to prevent rapid switching.

use tracing::debug;

/// Volatility regime classification.
///
/// Four states with hysteresis to prevent rapid switching:
/// - Low: Very quiet market (σ < 0.5 × baseline)
/// - Normal: Standard market conditions
/// - High: Elevated volatility (σ > 1.5 × baseline)
/// - Extreme: Crisis/toxic conditions (σ > 3 × baseline OR high jump ratio)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum VolatilityRegime {
    /// Very quiet market - can tighten spreads
    Low,
    /// Normal market conditions
    #[default]
    Normal,
    /// Elevated volatility - widen spreads
    High,
    /// Crisis/toxic - maximum caution, consider pulling quotes
    Extreme,
}

impl VolatilityRegime {
    /// Get spread multiplier for this regime.
    ///
    /// Used to scale spreads based on volatility state.
    pub(crate) fn spread_multiplier(&self) -> f64 {
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
    pub(crate) fn gamma_multiplier(&self) -> f64 {
        match self {
            Self::Low => 0.8,
            Self::Normal => 1.0,
            Self::High => 1.5,
            Self::Extreme => 3.0,
        }
    }

    /// Get Kelly fraction multiplier for this regime.
    ///
    /// In high volatility, we want to be more conservative (lower Kelly fraction).
    /// In low volatility, we can be more aggressive (higher Kelly fraction).
    ///
    /// Returns a multiplier to apply to the base Kelly fraction (0.25 default):
    /// - Low: 1.5x → 0.375 effective (can be more aggressive in quiet markets)
    /// - Normal: 1.0x → 0.25 effective (standard quarter Kelly)
    /// - High: 0.5x → 0.125 effective (more conservative)
    /// - Extreme: 0.25x → 0.0625 effective (very conservative, near flat)
    pub fn kelly_fraction_multiplier(&self) -> f64 {
        match self {
            Self::Low => 1.5,      // More aggressive in quiet markets
            Self::Normal => 1.0,   // Standard
            Self::High => 0.5,     // More conservative
            Self::Extreme => 0.25, // Very conservative
        }
    }

    /// Check if quotes should be pulled (extreme regime).
    #[allow(dead_code)]
    pub(crate) fn should_consider_pulling_quotes(&self) -> bool {
        matches!(self, Self::Extreme)
    }
}

/// Tracks volatility regime with asymmetric hysteresis to prevent rapid switching.
///
/// Transitions between states require sustained conditions to trigger,
/// preventing oscillation at boundaries.
///
/// Asymmetric hysteresis captures real market behavior:
/// - Vol spikes fast: Escalation to High/Extreme is faster (2 ticks)
/// - Vol mean-reverts slow: De-escalation to Normal/Low is slower (8 ticks)
#[derive(Debug)]
pub(crate) struct VolatilityRegimeTracker {
    /// Current regime state
    regime: VolatilityRegime,
    /// Baseline volatility (for regime thresholds)
    baseline_sigma: f64,
    /// Consecutive updates in potential new regime (for hysteresis)
    transition_count: u32,
    /// Minimum transitions for escalating to higher regime (fast: vol spikes)
    min_transitions_escalate: u32,
    /// Minimum transitions for de-escalating to lower regime (slow: vol mean-reverts)
    min_transitions_deescalate: u32,
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
    pub(crate) fn new(baseline_sigma: f64) -> Self {
        Self {
            regime: VolatilityRegime::Normal,
            baseline_sigma,
            transition_count: 0,
            // Asymmetric hysteresis: escalate fast (2 ticks), de-escalate slow (8 ticks)
            // This matches market behavior: vol spikes quickly, mean-reverts slowly
            min_transitions_escalate: 2,
            min_transitions_deescalate: 8,
            low_threshold: 0.5,
            high_threshold: 1.5,
            extreme_threshold: 3.0,
            jump_threshold: 3.0,
            pending_regime: None,
        }
    }

    /// Determine if a transition is escalating (moving to higher risk regime).
    fn is_escalation(&self, from: VolatilityRegime, to: VolatilityRegime) -> bool {
        let from_level = match from {
            VolatilityRegime::Low => 0,
            VolatilityRegime::Normal => 1,
            VolatilityRegime::High => 2,
            VolatilityRegime::Extreme => 3,
        };
        let to_level = match to {
            VolatilityRegime::Low => 0,
            VolatilityRegime::Normal => 1,
            VolatilityRegime::High => 2,
            VolatilityRegime::Extreme => 3,
        };
        to_level > from_level
    }

    /// Get required transitions for a regime change (asymmetric).
    fn required_transitions(&self, from: VolatilityRegime, to: VolatilityRegime) -> u32 {
        if self.is_escalation(from, to) {
            self.min_transitions_escalate
        } else {
            self.min_transitions_deescalate
        }
    }

    /// Update regime based on current volatility and jump ratio.
    ///
    /// Uses asymmetric hysteresis:
    /// - Escalation (to higher risk): 2 ticks - react quickly to vol spikes
    /// - De-escalation (to lower risk): 8 ticks - confirm vol has truly subsided
    pub(crate) fn update(&mut self, sigma: f64, jump_ratio: f64) {
        // Determine target regime based on current conditions
        let target = self.classify(sigma, jump_ratio);

        // Check if target matches pending transition
        if let Some(pending) = self.pending_regime {
            if pending == target {
                self.transition_count += 1;
                // Use asymmetric hysteresis: fast escalation, slow de-escalation
                let required = self.required_transitions(self.regime, target);
                if self.transition_count >= required {
                    // Transition confirmed
                    if self.regime != target {
                        let direction = if self.is_escalation(self.regime, target) {
                            "escalation"
                        } else {
                            "de-escalation"
                        };
                        debug!(
                            from = ?self.regime,
                            to = ?target,
                            sigma = %format!("{:.6}", sigma),
                            jump_ratio = %format!("{:.2}", jump_ratio),
                            direction = direction,
                            ticks_required = required,
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
    pub(crate) fn regime(&self) -> VolatilityRegime {
        self.regime
    }

    /// Update baseline volatility (e.g., from long-term EWMA).
    pub(crate) fn update_baseline(&mut self, new_baseline: f64) {
        if new_baseline > 1e-9 {
            // Slow update to baseline (EWMA with long half-life)
            self.baseline_sigma = 0.99 * self.baseline_sigma + 0.01 * new_baseline;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volatility_regime_multipliers() {
        assert!((VolatilityRegime::Low.spread_multiplier() - 0.8).abs() < 0.01);
        assert!((VolatilityRegime::Normal.spread_multiplier() - 1.0).abs() < 0.01);
        assert!((VolatilityRegime::High.spread_multiplier() - 1.5).abs() < 0.01);
        assert!((VolatilityRegime::Extreme.spread_multiplier() - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_volatility_regime_gamma_multipliers() {
        assert!((VolatilityRegime::Low.gamma_multiplier() - 0.8).abs() < 0.01);
        assert!((VolatilityRegime::Normal.gamma_multiplier() - 1.0).abs() < 0.01);
        assert!((VolatilityRegime::High.gamma_multiplier() - 1.5).abs() < 0.01);
        assert!((VolatilityRegime::Extreme.gamma_multiplier() - 3.0).abs() < 0.01);
    }
}
