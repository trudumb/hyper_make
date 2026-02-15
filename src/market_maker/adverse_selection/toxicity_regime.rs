//! Discrete toxicity regime classification.
//!
//! Classifies current market conditions into `{Benign, Normal, Toxic}` based on
//! the PreFillASClassifier's toxicity score plus OFI acceleration signals.
//! Thresholds are regime-dependent: tighter in cascades, looser in quiet markets.
//!
//! Used by the execution state machine (Phase 4) to select quoting mode.

use serde::{Deserialize, Serialize};

/// Discrete toxicity regime for execution mode selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ToxicityRegime {
    /// Low toxicity — safe to quote both sides
    #[default]
    Benign,
    /// Moderate toxicity — normal quoting with spread adjustment
    Normal,
    /// High toxicity — should go flat or reduce-only
    Toxic,
}

/// Regime-dependent toxicity thresholds.
#[derive(Debug, Clone, Copy)]
pub struct ToxicityThresholds {
    /// Below this → Benign
    pub benign_ceiling: f64,
    /// Above this → Toxic
    pub toxic_floor: f64,
}

impl ToxicityThresholds {
    /// Compute thresholds conditioned on regime probabilities.
    ///
    /// In cascade/bursty regimes, lower the toxic floor to react faster.
    /// `regime_probs`: [quiet, normal, bursty, cascade]
    pub fn for_regime(regime_probs: &[f64; 4]) -> Self {
        let stress_weight = regime_probs[2] + regime_probs[3]; // bursty + cascade

        // Interpolate between quiet and cascade thresholds
        let benign_ceiling = lerp(0.25, 0.20, stress_weight);
        let toxic_floor = lerp(0.55, 0.40, stress_weight);

        Self {
            benign_ceiling,
            toxic_floor,
        }
    }

    /// Classify a toxicity score into a regime.
    pub fn classify(&self, toxicity: f64) -> ToxicityRegime {
        if toxicity <= self.benign_ceiling {
            ToxicityRegime::Benign
        } else if toxicity >= self.toxic_floor {
            ToxicityRegime::Toxic
        } else {
            ToxicityRegime::Normal
        }
    }
}

/// OFI-derived acceleration signals for fast toxicity detection.
///
/// These detect sudden flow onset within 1-2 trades, before the main
/// classifier's 50-trade EWMA can react.
#[derive(Debug, Clone, Copy, Default)]
pub struct OfiAccelerationSignals {
    /// Derivative of 1s OFI — detects sudden flow onset
    pub ofi_acceleration: f64,
    /// Ratio of 5s/1s OFI — sustained vs transient flow
    pub ofi_divergence: f64,
}

impl OfiAccelerationSignals {
    /// Compute acceleration signals from OFI snapshots.
    ///
    /// - `ofi_1s`: Current 1-second OFI
    /// - `ofi_5s`: Current 5-second OFI
    /// - `prev_ofi_1s`: Previous cycle's 1-second OFI
    pub fn compute(ofi_1s: f64, ofi_5s: f64, prev_ofi_1s: f64) -> Self {
        let ofi_acceleration = ofi_1s - prev_ofi_1s;

        // Divergence: how much does short-term differ from medium-term?
        // High magnitude → sustained directional flow
        let ofi_divergence = if ofi_1s.abs() > 1e-6 {
            (ofi_5s / ofi_1s).clamp(-3.0, 3.0)
        } else {
            1.0 // No short-term signal → assume neutral
        };

        Self {
            ofi_acceleration,
            ofi_divergence,
        }
    }

    /// Compute a toxicity boost from OFI acceleration.
    ///
    /// Returns a value in [0, 0.3] that can be added to the base toxicity score
    /// to accelerate Toxic detection during sudden flow.
    pub fn toxicity_boost(&self) -> f64 {
        // Strong acceleration (> 0.3 in absolute terms) adds up to 0.15
        let accel_boost = (self.ofi_acceleration.abs() - 0.3).clamp(0.0, 0.5) * 0.3;

        // Divergence > 1.5 (sustained > transient) adds up to 0.15
        let diverge_boost = (self.ofi_divergence.abs() - 1.5).clamp(0.0, 0.5) * 0.3;

        (accel_boost + diverge_boost).min(0.3)
    }
}

/// Classify current toxicity into a discrete regime.
///
/// Combines the base toxicity score with OFI acceleration for fast detection.
pub fn classify_toxicity(
    base_toxicity: f64,
    ofi_signals: &OfiAccelerationSignals,
    regime_probs: &[f64; 4],
) -> ToxicityRegime {
    let boosted = (base_toxicity + ofi_signals.toxicity_boost()).clamp(0.0, 1.0);
    let thresholds = ToxicityThresholds::for_regime(regime_probs);
    thresholds.classify(boosted)
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

impl std::fmt::Display for ToxicityRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Benign => write!(f, "Benign"),
            Self::Normal => write!(f, "Normal"),
            Self::Toxic => write!(f, "Toxic"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benign_in_quiet_regime() {
        let regime = [0.8, 0.15, 0.04, 0.01]; // quiet
        let thresholds = ToxicityThresholds::for_regime(&regime);
        assert_eq!(thresholds.classify(0.1), ToxicityRegime::Benign);
        assert_eq!(thresholds.classify(0.2), ToxicityRegime::Benign);
    }

    #[test]
    fn test_toxic_in_quiet_regime() {
        let regime = [0.8, 0.15, 0.04, 0.01];
        let thresholds = ToxicityThresholds::for_regime(&regime);
        assert_eq!(thresholds.classify(0.6), ToxicityRegime::Toxic);
    }

    #[test]
    fn test_toxic_threshold_lower_in_cascade() {
        let quiet = [0.8, 0.15, 0.04, 0.01];
        let cascade = [0.05, 0.1, 0.3, 0.55];
        let t_quiet = ToxicityThresholds::for_regime(&quiet);
        let t_cascade = ToxicityThresholds::for_regime(&cascade);

        // Cascade should have lower toxic floor
        assert!(t_cascade.toxic_floor < t_quiet.toxic_floor);
        // 0.45 should be Toxic in cascade but Normal in quiet
        assert_eq!(t_cascade.classify(0.45), ToxicityRegime::Toxic);
        assert_eq!(t_quiet.classify(0.45), ToxicityRegime::Normal);
    }

    #[test]
    fn test_ofi_acceleration_boost() {
        let signals = OfiAccelerationSignals {
            ofi_acceleration: 0.8, // Strong acceleration
            ofi_divergence: 2.0,   // Sustained flow
        };
        let boost = signals.toxicity_boost();
        assert!(boost > 0.0);
        assert!(boost <= 0.3);
    }

    #[test]
    fn test_ofi_no_boost_when_calm() {
        let signals = OfiAccelerationSignals {
            ofi_acceleration: 0.1, // Mild
            ofi_divergence: 1.0,   // Neutral
        };
        let boost = signals.toxicity_boost();
        assert_eq!(boost, 0.0);
    }

    #[test]
    fn test_classify_toxicity_with_boost() {
        let regime = [0.05, 0.1, 0.3, 0.55]; // cascade
        let signals = OfiAccelerationSignals {
            ofi_acceleration: 0.8,
            ofi_divergence: 2.0,
        };
        // Base toxicity 0.3 (Normal), but with boost should tip to Toxic in cascade
        let result = classify_toxicity(0.3, &signals, &regime);
        assert_eq!(result, ToxicityRegime::Toxic);
    }

    #[test]
    fn test_classify_toxicity_without_boost() {
        let regime = [0.8, 0.15, 0.04, 0.01]; // quiet
        let no_signals = OfiAccelerationSignals::default();
        let result = classify_toxicity(0.1, &no_signals, &regime);
        assert_eq!(result, ToxicityRegime::Benign);
    }

    #[test]
    fn test_ofi_compute() {
        let signals = OfiAccelerationSignals::compute(0.5, 0.3, 0.1);
        assert!((signals.ofi_acceleration - 0.4).abs() < 1e-10);
        assert!((signals.ofi_divergence - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_ofi_compute_zero_ofi() {
        let signals = OfiAccelerationSignals::compute(0.0, 0.3, 0.1);
        assert_eq!(signals.ofi_divergence, 1.0); // Neutral when no short-term signal
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", ToxicityRegime::Benign), "Benign");
        assert_eq!(format!("{}", ToxicityRegime::Toxic), "Toxic");
    }
}
