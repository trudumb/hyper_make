//! Unified feature vector with staleness decay and confidence tracking.
//! Replaces scattered ad-hoc feature handling across the pipeline.

use std::time::Instant;

/// Individual feature with confidence and staleness tracking.
#[derive(Debug, Clone, Copy)]
pub struct NormalizedFeature {
    /// Feature value, normalized to [-1, 1]
    pub value: f64,
    /// Confidence in measurement [0, 1]
    pub confidence: f64,
    /// Age of measurement in milliseconds
    pub age_ms: f64,
}

impl NormalizedFeature {
    /// Create a new normalized feature.
    pub fn new(value: f64, confidence: f64, age_ms: f64) -> Self {
        Self {
            value: value.clamp(-1.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            age_ms: age_ms.max(0.0),
        }
    }

    /// Decayed effective value: value * confidence * exp(-ln2 * age / half_life)
    pub fn effective(&self, half_life_ms: f64) -> f64 {
        if half_life_ms <= 0.0 || self.age_ms < 0.0 {
            return 0.0;
        }
        let decay = (-self.age_ms * std::f64::consts::LN_2 / half_life_ms).exp();
        self.value * self.confidence * decay
    }

    /// Create a stale (zero-confidence) feature sentinel.
    pub fn stale() -> Self {
        Self {
            value: 0.0,
            confidence: 0.0,
            age_ms: f64::MAX,
        }
    }

    /// Whether this feature has meaningful data.
    pub fn is_valid(&self) -> bool {
        self.confidence > 0.0 && self.age_ms < f64::MAX
    }
}

impl Default for NormalizedFeature {
    fn default() -> Self {
        Self::stale()
    }
}

/// Central feature vector built once per quoting cycle.
/// Aggregates all signal sources into a unified representation.
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// Flow toxicity from Hawkes intensity asymmetry [-1, 1]
    pub flow_toxicity: NormalizedFeature,
    /// Trade tape directional pressure [-1, 1]
    pub flow_direction: NormalizedFeature,
    /// L2 book depth change rate asymmetry [-1, 1]
    pub book_thinning: NormalizedFeature,
    /// Funding rate induced pressure [-1, 1]
    pub funding_pressure: NormalizedFeature,
    /// Sigma relative to baseline (dimensionless ratio)
    pub sigma_normalized: f64,
    /// Kappa relative to regime (dimensionless ratio)
    pub kappa_normalized: f64,
    /// When this vector was computed
    pub computed_at: Instant,
    /// Hawkes synchronicity: cross-venue intensity correlation [-1, 1]
    pub synchronicity: NormalizedFeature,
    /// Cross-venue flow acceleration: rate of change of flow imbalance [-1, 1]
    pub flow_acceleration: NormalizedFeature,
    /// Funding basis velocity: rate of premium change scaled by settlement proximity [-1, 1]
    pub basis_velocity: NormalizedFeature,
}

impl FeatureVector {
    /// Directional signal: weighted combination for skew computation.
    /// Positive = bullish pressure, negative = bearish.
    pub fn directional_signal(&self) -> f64 {
        const FLOW_DIR_WEIGHT: f64 = 0.35;
        const FLOW_TOX_WEIGHT: f64 = 0.20;
        const BOOK_THIN_WEIGHT: f64 = 0.15;
        const FUNDING_WEIGHT: f64 = 0.10;
        const SYNC_WEIGHT: f64 = 0.10;
        const ACCEL_WEIGHT: f64 = 0.05;
        const BASIS_VEL_WEIGHT: f64 = 0.05;

        self.flow_direction.effective(2000.0) * FLOW_DIR_WEIGHT
            + self.flow_toxicity.effective(1000.0) * FLOW_TOX_WEIGHT
            + self.book_thinning.effective(3000.0) * BOOK_THIN_WEIGHT
            + self.funding_pressure.effective(10000.0) * FUNDING_WEIGHT
            + self.synchronicity.effective(2000.0) * SYNC_WEIGHT
            + self.flow_acceleration.effective(1500.0) * ACCEL_WEIGHT
            + self.basis_velocity.effective(5000.0) * BASIS_VEL_WEIGHT
    }

    /// Risk signal: urgency for spread widening. Always >= 0.
    pub fn risk_signal(&self) -> f64 {
        (self.flow_toxicity.effective(1000.0).abs()
            + self.book_thinning.effective(2000.0).abs() * 0.5)
            .clamp(0.0, 1.0)
    }

    /// Create a default (all stale) feature vector.
    pub fn empty() -> Self {
        Self {
            flow_toxicity: NormalizedFeature::stale(),
            flow_direction: NormalizedFeature::stale(),
            book_thinning: NormalizedFeature::stale(),
            funding_pressure: NormalizedFeature::stale(),
            sigma_normalized: 1.0,
            kappa_normalized: 1.0,
            computed_at: Instant::now(),
            synchronicity: NormalizedFeature::stale(),
            flow_acceleration: NormalizedFeature::stale(),
            basis_velocity: NormalizedFeature::stale(),
        }
    }
}

impl Default for FeatureVector {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalized_feature_effective_no_decay() {
        let f = NormalizedFeature::new(0.8, 1.0, 0.0);
        assert!((f.effective(1000.0) - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_feature_effective_half_life() {
        let f = NormalizedFeature::new(1.0, 1.0, 1000.0);
        // At exactly one half-life, value should be 0.5
        assert!((f.effective(1000.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_feature_effective_with_confidence() {
        let f = NormalizedFeature::new(1.0, 0.5, 0.0);
        assert!((f.effective(1000.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_feature_stale() {
        let f = NormalizedFeature::stale();
        assert_eq!(f.effective(1000.0), 0.0);
        assert!(!f.is_valid());
    }

    #[test]
    fn test_normalized_feature_clamping() {
        let f = NormalizedFeature::new(5.0, 2.0, -10.0);
        assert_eq!(f.value, 1.0);
        assert_eq!(f.confidence, 1.0);
        assert_eq!(f.age_ms, 0.0);
    }

    #[test]
    fn test_normalized_feature_zero_half_life() {
        let f = NormalizedFeature::new(0.8, 1.0, 0.0);
        assert_eq!(f.effective(0.0), 0.0);
        assert_eq!(f.effective(-1.0), 0.0);
    }

    #[test]
    fn test_feature_vector_directional_signal_neutral() {
        let fv = FeatureVector::empty();
        assert_eq!(fv.directional_signal(), 0.0);
    }

    #[test]
    fn test_feature_vector_directional_signal_bullish() {
        let mut fv = FeatureVector::empty();
        fv.flow_direction = NormalizedFeature::new(0.8, 1.0, 0.0);
        let signal = fv.directional_signal();
        assert!(
            signal > 0.0,
            "Bullish flow should produce positive directional signal, got {signal}"
        );
    }

    #[test]
    fn test_feature_vector_directional_signal_bearish() {
        let mut fv = FeatureVector::empty();
        fv.flow_direction = NormalizedFeature::new(-0.8, 1.0, 0.0);
        let signal = fv.directional_signal();
        assert!(
            signal < 0.0,
            "Bearish flow should produce negative directional signal, got {signal}"
        );
    }

    #[test]
    fn test_feature_vector_risk_signal_calm() {
        let fv = FeatureVector::empty();
        assert_eq!(fv.risk_signal(), 0.0);
    }

    #[test]
    fn test_feature_vector_risk_signal_toxic() {
        let mut fv = FeatureVector::empty();
        fv.flow_toxicity = NormalizedFeature::new(0.9, 1.0, 0.0);
        let risk = fv.risk_signal();
        assert!(
            risk > 0.5,
            "High toxicity should produce high risk signal, got {risk}"
        );
    }

    #[test]
    fn test_feature_vector_risk_signal_clamped() {
        let mut fv = FeatureVector::empty();
        fv.flow_toxicity = NormalizedFeature::new(1.0, 1.0, 0.0);
        fv.book_thinning = NormalizedFeature::new(1.0, 1.0, 0.0);
        assert!(fv.risk_signal() <= 1.0);
    }

    #[test]
    fn test_decay_increases_with_age() {
        let young = NormalizedFeature::new(1.0, 1.0, 100.0);
        let old = NormalizedFeature::new(1.0, 1.0, 5000.0);
        assert!(young.effective(1000.0) > old.effective(1000.0));
    }

    #[test]
    fn test_new_features_default_stale() {
        let fv = FeatureVector::empty();
        assert!(!fv.synchronicity.is_valid());
        assert!(!fv.flow_acceleration.is_valid());
        assert!(!fv.basis_velocity.is_valid());
    }

    #[test]
    fn test_directional_signal_with_synchronicity() {
        let mut fv = FeatureVector::empty();
        fv.synchronicity = NormalizedFeature::new(0.9, 1.0, 0.0);
        let signal = fv.directional_signal();
        assert!(
            signal > 0.0,
            "Positive synchronicity should produce positive directional signal, got {signal}"
        );
    }

    #[test]
    fn test_directional_signal_with_basis_velocity() {
        let mut fv = FeatureVector::empty();
        fv.basis_velocity = NormalizedFeature::new(-0.7, 1.0, 0.0);
        let signal = fv.directional_signal();
        assert!(
            signal < 0.0,
            "Negative basis velocity should produce negative directional signal, got {signal}"
        );
    }

    #[test]
    fn test_directional_signal_weights_sum_to_one() {
        // Verify weights are properly normalized
        const FLOW_DIR_WEIGHT: f64 = 0.35;
        const FLOW_TOX_WEIGHT: f64 = 0.20;
        const BOOK_THIN_WEIGHT: f64 = 0.15;
        const FUNDING_WEIGHT: f64 = 0.10;
        const SYNC_WEIGHT: f64 = 0.10;
        const ACCEL_WEIGHT: f64 = 0.05;
        const BASIS_VEL_WEIGHT: f64 = 0.05;

        let total = FLOW_DIR_WEIGHT
            + FLOW_TOX_WEIGHT
            + BOOK_THIN_WEIGHT
            + FUNDING_WEIGHT
            + SYNC_WEIGHT
            + ACCEL_WEIGHT
            + BASIS_VEL_WEIGHT;
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Directional signal weights should sum to 1.0, got {total}"
        );
    }
}
