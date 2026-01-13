//! Volatility regime classification and tracking.
//!
//! Four-state regime with asymmetric hysteresis to prevent rapid switching.
//! Self-calibrating: learns baseline and thresholds from observed market data.

use tracing::{debug, info};

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
            Self::Extreme => 1.8, // Reduced from 3.0 - still conservative but not punitive
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

/// Result of warmup calibration from observed market data.
#[derive(Debug, Clone)]
struct CalibrationResult {
    /// Baseline volatility (median of observations)
    baseline: f64,
    /// Extreme threshold multiplier (P95/median, min 3.0)
    extreme_threshold: f64,
    /// Low threshold multiplier (P10/median, max 0.7)
    low_threshold: f64,
    /// High threshold multiplier (P75/median, min 1.3)
    high_threshold: f64,
}

/// Self-calibrating warmup that learns volatility thresholds from observed data.
///
/// Instead of hardcoding regime thresholds, this collects observations during warmup
/// and computes percentile-based thresholds that adapt to the actual market.
#[derive(Debug)]
struct WarmupCalibrator {
    /// Collected sigma observations
    observations: Vec<f64>,
    /// Minimum observations before calibration (default: 50)
    min_observations: usize,
    /// Whether calibration has been performed
    is_calibrated: bool,
}

impl WarmupCalibrator {
    fn new(min_observations: usize) -> Self {
        Self {
            observations: Vec::with_capacity(min_observations + 10),
            min_observations,
            is_calibrated: false,
        }
    }

    /// Add an observation and return calibration result when ready.
    fn add_observation(&mut self, sigma: f64) -> Option<CalibrationResult> {
        if self.is_calibrated {
            return None;
        }

        // Filter out clearly invalid observations
        if sigma > 1e-9 && sigma < 0.1 {
            self.observations.push(sigma);
        }

        if self.observations.len() >= self.min_observations {
            let result = self.compute_calibration();
            self.is_calibrated = true;
            Some(result)
        } else {
            None
        }
    }

    /// Compute calibration from collected observations.
    fn compute_calibration(&self) -> CalibrationResult {
        let mut sorted = self.observations.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let len = sorted.len();
        let median = sorted[len / 2];
        let p10 = sorted[(len as f64 * 0.10) as usize];
        let p75 = sorted[(len as f64 * 0.75) as usize];
        let p95 = sorted[(len as f64 * 0.95) as usize];

        CalibrationResult {
            baseline: median,
            // Extreme: P95/median but at least 3.0 to avoid false positives
            extreme_threshold: (p95 / median).max(3.0),
            // Low: P10/median but at most 0.7 to ensure it's truly quiet
            low_threshold: (p10 / median).min(0.7),
            // High: P75/median but at least 1.3 to distinguish from normal
            high_threshold: (p75 / median).max(1.3),
        }
    }

    /// Get warmup progress as (current, required).
    fn progress(&self) -> (usize, usize) {
        (self.observations.len(), self.min_observations)
    }
}

/// Tracks volatility regime with asymmetric hysteresis to prevent rapid switching.
///
/// Transitions between states require sustained conditions to trigger,
/// preventing oscillation at boundaries.
///
/// **Self-calibrating:** During warmup, collects volatility observations and
/// computes percentile-based thresholds that adapt to the actual market.
/// This eliminates the need for hardcoded "magic numbers" that may not match
/// different assets or market conditions.
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
    low_threshold: f64,     // σ < baseline × low_threshold → Low
    high_threshold: f64,    // σ > baseline × high_threshold → High
    extreme_threshold: f64, // σ > baseline × extreme_threshold → Extreme
    /// Jump ratio threshold for Extreme regime
    jump_threshold: f64,
    /// Pending regime (for hysteresis tracking)
    pending_regime: Option<VolatilityRegime>,
    /// Self-calibrating warmup (learns thresholds from observed data)
    calibrator: WarmupCalibrator,
}

impl VolatilityRegimeTracker {
    /// Number of observations required for calibration.
    const CALIBRATION_OBSERVATIONS: usize = 50;

    pub(crate) fn new(baseline_sigma: f64) -> Self {
        Self {
            regime: VolatilityRegime::Normal,
            baseline_sigma,
            transition_count: 0,
            // Balanced hysteresis: escalate (3 ticks), de-escalate (4 ticks)
            min_transitions_escalate: 3,
            min_transitions_deescalate: 4,
            // Initial thresholds (will be overwritten by calibrator)
            // These serve as fallback if calibration fails or during warmup
            low_threshold: 0.5,
            high_threshold: 1.5,
            extreme_threshold: 5.0,
            jump_threshold: 3.0,
            pending_regime: None,
            calibrator: WarmupCalibrator::new(Self::CALIBRATION_OBSERVATIONS),
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
    /// - Escalation (to higher risk): 3 ticks - react quickly to vol spikes
    /// - De-escalation (to lower risk): 4 ticks - confirm vol has truly subsided
    ///
    /// **Self-calibrating:** During warmup, collects sigma observations and
    /// computes percentile-based thresholds when enough data is collected.
    pub(crate) fn update(&mut self, sigma: f64, jump_ratio: f64) {
        // Self-calibration: collect observations and calibrate when ready
        if let Some(calibration) = self.calibrator.add_observation(sigma) {
            self.baseline_sigma = calibration.baseline;
            self.low_threshold = calibration.low_threshold;
            self.high_threshold = calibration.high_threshold;
            self.extreme_threshold = calibration.extreme_threshold;
            info!(
                baseline = %format!("{:.6}", calibration.baseline),
                low_threshold = %format!("{:.2}", calibration.low_threshold),
                high_threshold = %format!("{:.2}", calibration.high_threshold),
                extreme_threshold = %format!("{:.2}", calibration.extreme_threshold),
                observations = self.calibrator.progress().0,
                "Volatility regime self-calibrated from observed market data"
            );
        }

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
        assert!((VolatilityRegime::Extreme.gamma_multiplier() - 1.8).abs() < 0.01); // Reduced from 3.0
    }

    #[test]
    fn test_warmup_calibrator() {
        let mut calibrator = WarmupCalibrator::new(10); // Small for testing

        // Add observations simulating realistic BTC volatility
        let observations = [0.00020, 0.00022, 0.00018, 0.00025, 0.00030,
                           0.00019, 0.00021, 0.00023, 0.00028, 0.00024];

        for (i, &sigma) in observations.iter().enumerate() {
            let result = calibrator.add_observation(sigma);
            if i < 9 {
                assert!(result.is_none(), "Should not calibrate before min observations");
            } else {
                assert!(result.is_some(), "Should calibrate at min observations");
                let cal = result.unwrap();
                // Median should be around 0.000225 (middle of sorted values)
                assert!(cal.baseline > 0.00020 && cal.baseline < 0.00025);
                // Extreme threshold should be at least 3.0
                assert!(cal.extreme_threshold >= 3.0);
            }
        }

        // Calibrator should not produce more results
        assert!(calibrator.add_observation(0.00025).is_none());
    }
}
