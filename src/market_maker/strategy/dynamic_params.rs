//! Dynamic hyperparameters for the Shadow Tuner hot-swap mechanism.
//!
//! These are "macro" parameters — they change on ~5 minute timescales via CMA-ES optimization.
//! The "micro" parameters (kappa, sigma, regime HMM) remain under Bayesian online control.
//!
//! # Design
//!
//! The Shadow Tuner runs CMA-ES in simulation, producing candidate `DynamicParams`.
//! The live engine blends them in gradually via `blend()`, with extra safety on
//! `inventory_beta` when position is elevated (>50% utilization) to prevent forced
//! liquidation from sudden penalty changes.

use serde::{Deserialize, Serialize};

/// Number of tunable parameters exposed to CMA-ES.
const NUM_TUNABLE_PARAMS: usize = 8;

/// Dynamic hyperparameters that the Shadow Tuner can hot-swap into the live engine.
///
/// These are "macro" parameters — they change on ~5 minute timescales via CMA-ES optimization.
/// The "micro" parameters (kappa, sigma, regime HMM) remain under Bayesian online control.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DynamicParams {
    /// Base risk aversion — wider = safer (unitless, typically 0.05..1.0)
    pub gamma_base: f64,
    /// Quadratic inventory penalty coefficient (unitless, >= 1.0)
    pub inventory_beta: f64,
    /// Minimum viable spread in basis points
    pub spread_floor_bps: f64,
    /// Time-of-day widening multiplier for toxic hours (>= 1.0)
    pub toxic_hour_gamma_mult: f64,
    /// Informed trader fraction for adverse selection model [0.0, 1.0]
    pub alpha_touch: f64,
    /// Kelly criterion safety factor [0.05, 0.50]
    pub kelly_fraction: f64,
    /// OI drop percentage threshold for cascade defense (fraction, > 0.0)
    pub cascade_threshold: f64,
    /// Momentum-based position building sensitivity (unitless, > 0.0)
    pub proactive_skew_sensitivity: f64,
    /// Monotonically increasing version counter
    pub version: u64,
    /// Timestamp when these params were produced (nanos since epoch)
    pub source_timestamp_ns: u64,
}

impl Default for DynamicParams {
    fn default() -> Self {
        Self {
            gamma_base: 0.15,
            inventory_beta: 7.0,
            spread_floor_bps: 5.0,
            toxic_hour_gamma_mult: 2.0,
            alpha_touch: 0.25,
            kelly_fraction: 0.20,
            cascade_threshold: 0.02,
            proactive_skew_sensitivity: 2.0,
            version: 0,
            source_timestamp_ns: 0,
        }
    }
}

impl DynamicParams {
    /// Validate all parameter invariants.
    ///
    /// Returns `Ok(())` if all constraints are satisfied, or `Err(reason)` describing
    /// the first violated constraint.
    pub fn validate(&self) -> Result<(), String> {
        if self.gamma_base <= 0.0 {
            return Err(format!("gamma_base must be > 0.0, got {}", self.gamma_base));
        }
        if self.inventory_beta < 1.0 {
            return Err(format!(
                "inventory_beta must be >= 1.0, got {}",
                self.inventory_beta
            ));
        }
        // spread_floor_bps must exceed maker fee (1.5 bps)
        if self.spread_floor_bps <= 1.5 {
            return Err(format!(
                "spread_floor_bps must be > 1.5 (maker fee), got {}",
                self.spread_floor_bps
            ));
        }
        if self.toxic_hour_gamma_mult < 1.0 {
            return Err(format!(
                "toxic_hour_gamma_mult must be >= 1.0, got {}",
                self.toxic_hour_gamma_mult
            ));
        }
        if !(0.0..=1.0).contains(&self.alpha_touch) {
            return Err(format!(
                "alpha_touch must be in [0.0, 1.0], got {}",
                self.alpha_touch
            ));
        }
        if !(0.0..=1.0).contains(&self.kelly_fraction) {
            return Err(format!(
                "kelly_fraction must be in [0.0, 1.0], got {}",
                self.kelly_fraction
            ));
        }
        if self.cascade_threshold <= 0.0 {
            return Err(format!(
                "cascade_threshold must be > 0.0, got {}",
                self.cascade_threshold
            ));
        }
        if self.proactive_skew_sensitivity <= 0.0 {
            return Err(format!(
                "proactive_skew_sensitivity must be > 0.0, got {}",
                self.proactive_skew_sensitivity
            ));
        }
        Ok(())
    }

    /// Graduated interpolation between old and new parameter sets.
    ///
    /// For most parameters: `old_val + alpha * (new_val - old_val)`.
    ///
    /// For `inventory_beta`: when `position_ratio > 0.5`, the change rate is clamped
    /// to +/-10% per blend step to prevent forced liquidation from sudden penalty shifts.
    ///
    /// # Arguments
    /// * `old` - Currently active parameters
    /// * `new` - Candidate parameters from Shadow Tuner
    /// * `alpha` - Blend fraction [0.0, 1.0] where 0.0 = all old, 1.0 = all new
    /// * `position_ratio` - `abs(position) / max_position` in [0.0, 1.0]
    pub fn blend(old: &Self, new: &Self, alpha: f64, position_ratio: f64) -> Self {
        let alpha = alpha.clamp(0.0, 1.0);

        let lerp = |old_val: f64, new_val: f64| -> f64 { old_val + alpha * (new_val - old_val) };

        // inventory_beta gets special treatment when position is elevated
        let inventory_beta_raw = lerp(old.inventory_beta, new.inventory_beta);
        let inventory_beta = if position_ratio > 0.5 {
            // Clamp change to +/-10% of old value to prevent forced liquidation
            let lo = old.inventory_beta * 0.9;
            let hi = old.inventory_beta * 1.1;
            inventory_beta_raw.clamp(lo, hi)
        } else {
            inventory_beta_raw
        };

        Self {
            gamma_base: lerp(old.gamma_base, new.gamma_base),
            inventory_beta,
            spread_floor_bps: lerp(old.spread_floor_bps, new.spread_floor_bps),
            toxic_hour_gamma_mult: lerp(old.toxic_hour_gamma_mult, new.toxic_hour_gamma_mult),
            alpha_touch: lerp(old.alpha_touch, new.alpha_touch),
            kelly_fraction: lerp(old.kelly_fraction, new.kelly_fraction),
            cascade_threshold: lerp(old.cascade_threshold, new.cascade_threshold),
            proactive_skew_sensitivity: lerp(
                old.proactive_skew_sensitivity,
                new.proactive_skew_sensitivity,
            ),
            // Metadata always comes from the new params
            version: new.version,
            source_timestamp_ns: new.source_timestamp_ns,
        }
    }

    /// Serialize the 8 tunable parameters into a flat vector for CMA-ES.
    ///
    /// Order: gamma_base, inventory_beta, spread_floor_bps, toxic_hour_gamma_mult,
    ///        alpha_touch, kelly_fraction, cascade_threshold, proactive_skew_sensitivity
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.gamma_base,
            self.inventory_beta,
            self.spread_floor_bps,
            self.toxic_hour_gamma_mult,
            self.alpha_touch,
            self.kelly_fraction,
            self.cascade_threshold,
            self.proactive_skew_sensitivity,
        ]
    }

    /// Reconstruct from a flat parameter vector (from CMA-ES).
    ///
    /// # Panics
    /// Panics if `v.len() < 8`.
    pub fn from_vec(v: &[f64], version: u64, timestamp_ns: u64) -> Self {
        assert!(
            v.len() >= NUM_TUNABLE_PARAMS,
            "from_vec requires at least {} elements, got {}",
            NUM_TUNABLE_PARAMS,
            v.len()
        );
        Self {
            gamma_base: v[0],
            inventory_beta: v[1],
            spread_floor_bps: v[2],
            toxic_hour_gamma_mult: v[3],
            alpha_touch: v[4],
            kelly_fraction: v[5],
            cascade_threshold: v[6],
            proactive_skew_sensitivity: v[7],
            version,
            source_timestamp_ns: timestamp_ns,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values_are_valid() {
        let params = DynamicParams::default();
        assert!(params.validate().is_ok(), "Default params must be valid");
    }

    #[test]
    fn test_validate_rejects_gamma_base_zero() {
        let params = DynamicParams {
            gamma_base: 0.0,
            ..Default::default()
        };
        let err = params.validate().unwrap_err();
        assert!(
            err.contains("gamma_base"),
            "Error should mention gamma_base: {err}"
        );
    }

    #[test]
    fn test_validate_rejects_gamma_base_negative() {
        let params = DynamicParams {
            gamma_base: -0.1,
            ..Default::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_inventory_beta_below_one() {
        let params = DynamicParams {
            inventory_beta: 0.5,
            ..Default::default()
        };
        let err = params.validate().unwrap_err();
        assert!(
            err.contains("inventory_beta"),
            "Error should mention inventory_beta: {err}"
        );
    }

    #[test]
    fn test_validate_rejects_spread_floor_at_maker_fee() {
        // spread_floor_bps=1.5 is exactly the maker fee — must be strictly greater
        let params = DynamicParams {
            spread_floor_bps: 1.5,
            ..Default::default()
        };
        let err = params.validate().unwrap_err();
        assert!(
            err.contains("spread_floor_bps"),
            "Error should mention spread_floor_bps: {err}"
        );
    }

    #[test]
    fn test_validate_rejects_spread_floor_below_maker_fee() {
        let params = DynamicParams {
            spread_floor_bps: 1.0,
            ..Default::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_toxic_hour_mult_below_one() {
        let params = DynamicParams {
            toxic_hour_gamma_mult: 0.9,
            ..Default::default()
        };
        let err = params.validate().unwrap_err();
        assert!(
            err.contains("toxic_hour_gamma_mult"),
            "Error should mention toxic_hour_gamma_mult: {err}"
        );
    }

    #[test]
    fn test_validate_rejects_alpha_touch_out_of_range() {
        let params = DynamicParams {
            alpha_touch: 1.1,
            ..Default::default()
        };
        assert!(params.validate().is_err());

        let params = DynamicParams {
            alpha_touch: -0.01,
            ..Default::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_kelly_fraction_out_of_range() {
        let params = DynamicParams {
            kelly_fraction: 1.5,
            ..Default::default()
        };
        assert!(params.validate().is_err());

        let params = DynamicParams {
            kelly_fraction: -0.1,
            ..Default::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_cascade_threshold_zero() {
        let params = DynamicParams {
            cascade_threshold: 0.0,
            ..Default::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_proactive_skew_sensitivity_zero() {
        let params = DynamicParams {
            proactive_skew_sensitivity: 0.0,
            ..Default::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_validate_accepts_boundary_values() {
        // alpha_touch and kelly_fraction at exact boundaries
        let params = DynamicParams {
            alpha_touch: 0.0,
            kelly_fraction: 0.0,
            ..Default::default()
        };
        assert!(params.validate().is_ok());

        let params = DynamicParams {
            alpha_touch: 1.0,
            kelly_fraction: 1.0,
            ..Default::default()
        };
        assert!(params.validate().is_ok());

        // toxic_hour_gamma_mult at exactly 1.0
        let params = DynamicParams {
            toxic_hour_gamma_mult: 1.0,
            ..Default::default()
        };
        assert!(params.validate().is_ok());

        // inventory_beta at exactly 1.0
        let params = DynamicParams {
            inventory_beta: 1.0,
            ..Default::default()
        };
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_blend_alpha_zero_returns_old() {
        let old = DynamicParams::default();
        let new = DynamicParams {
            gamma_base: 0.50,
            inventory_beta: 12.0,
            spread_floor_bps: 8.0,
            toxic_hour_gamma_mult: 3.0,
            alpha_touch: 0.40,
            kelly_fraction: 0.35,
            cascade_threshold: 0.05,
            proactive_skew_sensitivity: 4.0,
            version: 5,
            source_timestamp_ns: 1_000_000,
        };

        let blended = DynamicParams::blend(&old, &new, 0.0, 0.0);

        assert_eq!(blended.gamma_base, old.gamma_base);
        assert_eq!(blended.inventory_beta, old.inventory_beta);
        assert_eq!(blended.spread_floor_bps, old.spread_floor_bps);
        assert_eq!(blended.toxic_hour_gamma_mult, old.toxic_hour_gamma_mult);
        assert_eq!(blended.alpha_touch, old.alpha_touch);
        assert_eq!(blended.kelly_fraction, old.kelly_fraction);
        assert_eq!(blended.cascade_threshold, old.cascade_threshold);
        assert_eq!(
            blended.proactive_skew_sensitivity,
            old.proactive_skew_sensitivity
        );
        // Metadata comes from new
        assert_eq!(blended.version, new.version);
        assert_eq!(blended.source_timestamp_ns, new.source_timestamp_ns);
    }

    #[test]
    fn test_blend_alpha_one_returns_new() {
        let old = DynamicParams::default();
        let new = DynamicParams {
            gamma_base: 0.50,
            inventory_beta: 12.0,
            spread_floor_bps: 8.0,
            toxic_hour_gamma_mult: 3.0,
            alpha_touch: 0.40,
            kelly_fraction: 0.35,
            cascade_threshold: 0.05,
            proactive_skew_sensitivity: 4.0,
            version: 5,
            source_timestamp_ns: 1_000_000,
        };

        let blended = DynamicParams::blend(&old, &new, 1.0, 0.0);

        assert!((blended.gamma_base - new.gamma_base).abs() < 1e-12);
        assert!((blended.inventory_beta - new.inventory_beta).abs() < 1e-12);
        assert!((blended.spread_floor_bps - new.spread_floor_bps).abs() < 1e-12);
        assert!((blended.toxic_hour_gamma_mult - new.toxic_hour_gamma_mult).abs() < 1e-12);
        assert!((blended.alpha_touch - new.alpha_touch).abs() < 1e-12);
        assert!((blended.kelly_fraction - new.kelly_fraction).abs() < 1e-12);
        assert!((blended.cascade_threshold - new.cascade_threshold).abs() < 1e-12);
        assert!(
            (blended.proactive_skew_sensitivity - new.proactive_skew_sensitivity).abs() < 1e-12
        );
    }

    #[test]
    fn test_blend_partial_interpolation() {
        let old = DynamicParams {
            gamma_base: 0.10,
            inventory_beta: 5.0,
            spread_floor_bps: 4.0,
            toxic_hour_gamma_mult: 1.0,
            alpha_touch: 0.20,
            kelly_fraction: 0.10,
            cascade_threshold: 0.01,
            proactive_skew_sensitivity: 1.0,
            version: 1,
            source_timestamp_ns: 100,
        };
        let new = DynamicParams {
            gamma_base: 0.30,
            inventory_beta: 9.0,
            spread_floor_bps: 8.0,
            toxic_hour_gamma_mult: 3.0,
            alpha_touch: 0.40,
            kelly_fraction: 0.30,
            cascade_threshold: 0.03,
            proactive_skew_sensitivity: 3.0,
            version: 2,
            source_timestamp_ns: 200,
        };

        let blended = DynamicParams::blend(&old, &new, 0.5, 0.0);

        // Each param should be midpoint of old and new
        assert!((blended.gamma_base - 0.20).abs() < 1e-12);
        assert!((blended.inventory_beta - 7.0).abs() < 1e-12);
        assert!((blended.spread_floor_bps - 6.0).abs() < 1e-12);
        assert!((blended.toxic_hour_gamma_mult - 2.0).abs() < 1e-12);
        assert!((blended.alpha_touch - 0.30).abs() < 1e-12);
        assert!((blended.kelly_fraction - 0.20).abs() < 1e-12);
        assert!((blended.cascade_threshold - 0.02).abs() < 1e-12);
        assert!((blended.proactive_skew_sensitivity - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_blend_clamps_inventory_beta_when_position_elevated() {
        let old = DynamicParams {
            inventory_beta: 10.0,
            ..Default::default()
        };
        // New value is a 100% increase — way more than 10%
        let new = DynamicParams {
            inventory_beta: 20.0,
            version: 1,
            source_timestamp_ns: 100,
            ..Default::default()
        };

        // position_ratio = 0.8 > 0.5, so inventory_beta change clamped to +/-10%
        let blended = DynamicParams::blend(&old, &new, 1.0, 0.8);

        // Raw blended = 20.0, but clamped to old * 1.1 = 11.0
        assert!(
            (blended.inventory_beta - 11.0).abs() < 1e-12,
            "inventory_beta should be clamped to 11.0, got {}",
            blended.inventory_beta
        );
    }

    #[test]
    fn test_blend_clamps_inventory_beta_decrease_when_position_elevated() {
        let old = DynamicParams {
            inventory_beta: 10.0,
            ..Default::default()
        };
        let new = DynamicParams {
            inventory_beta: 2.0,
            version: 1,
            source_timestamp_ns: 100,
            ..Default::default()
        };

        // position_ratio = 0.7 > 0.5
        let blended = DynamicParams::blend(&old, &new, 1.0, 0.7);

        // Raw blended = 2.0, but clamped to old * 0.9 = 9.0
        assert!(
            (blended.inventory_beta - 9.0).abs() < 1e-12,
            "inventory_beta should be clamped to 9.0, got {}",
            blended.inventory_beta
        );
    }

    #[test]
    fn test_blend_no_clamp_when_position_low() {
        let old = DynamicParams {
            inventory_beta: 10.0,
            ..Default::default()
        };
        let new = DynamicParams {
            inventory_beta: 20.0,
            version: 1,
            source_timestamp_ns: 100,
            ..Default::default()
        };

        // position_ratio = 0.3 <= 0.5, so no clamping
        let blended = DynamicParams::blend(&old, &new, 1.0, 0.3);

        assert!(
            (blended.inventory_beta - 20.0).abs() < 1e-12,
            "inventory_beta should not be clamped at low position, got {}",
            blended.inventory_beta
        );
    }

    #[test]
    fn test_blend_alpha_clamped_to_valid_range() {
        let old = DynamicParams::default();
        let new = DynamicParams {
            gamma_base: 0.50,
            version: 1,
            ..Default::default()
        };

        // alpha > 1.0 should be clamped to 1.0
        let blended = DynamicParams::blend(&old, &new, 2.0, 0.0);
        assert!((blended.gamma_base - 0.50).abs() < 1e-12);

        // alpha < 0.0 should be clamped to 0.0
        let blended = DynamicParams::blend(&old, &new, -1.0, 0.0);
        assert!((blended.gamma_base - old.gamma_base).abs() < 1e-12);
    }

    #[test]
    fn test_to_vec_from_vec_roundtrip() {
        let original = DynamicParams {
            gamma_base: 0.22,
            inventory_beta: 8.5,
            spread_floor_bps: 6.3,
            toxic_hour_gamma_mult: 2.5,
            alpha_touch: 0.33,
            kelly_fraction: 0.18,
            cascade_threshold: 0.035,
            proactive_skew_sensitivity: 3.1,
            version: 42,
            source_timestamp_ns: 999_999_999,
        };

        let v = original.to_vec();
        assert_eq!(v.len(), NUM_TUNABLE_PARAMS);

        let reconstructed = DynamicParams::from_vec(&v, 42, 999_999_999);

        assert!((reconstructed.gamma_base - original.gamma_base).abs() < 1e-12);
        assert!((reconstructed.inventory_beta - original.inventory_beta).abs() < 1e-12);
        assert!((reconstructed.spread_floor_bps - original.spread_floor_bps).abs() < 1e-12);
        assert!(
            (reconstructed.toxic_hour_gamma_mult - original.toxic_hour_gamma_mult).abs() < 1e-12
        );
        assert!((reconstructed.alpha_touch - original.alpha_touch).abs() < 1e-12);
        assert!((reconstructed.kelly_fraction - original.kelly_fraction).abs() < 1e-12);
        assert!((reconstructed.cascade_threshold - original.cascade_threshold).abs() < 1e-12);
        assert!(
            (reconstructed.proactive_skew_sensitivity - original.proactive_skew_sensitivity).abs()
                < 1e-12
        );
        assert_eq!(reconstructed.version, original.version);
        assert_eq!(
            reconstructed.source_timestamp_ns,
            original.source_timestamp_ns
        );
    }

    #[test]
    fn test_to_vec_ordering() {
        let params = DynamicParams {
            gamma_base: 1.0,
            inventory_beta: 2.0,
            spread_floor_bps: 3.0,
            toxic_hour_gamma_mult: 4.0,
            alpha_touch: 5.0,
            kelly_fraction: 6.0,
            cascade_threshold: 7.0,
            proactive_skew_sensitivity: 8.0,
            version: 0,
            source_timestamp_ns: 0,
        };

        let v = params.to_vec();
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    #[should_panic(expected = "from_vec requires at least 8 elements")]
    fn test_from_vec_panics_on_short_input() {
        DynamicParams::from_vec(&[1.0, 2.0, 3.0], 0, 0);
    }

    #[test]
    fn test_from_vec_ignores_extra_elements() {
        let v = vec![0.15, 7.0, 5.0, 2.0, 0.25, 0.20, 0.02, 2.0, 99.0, 100.0];
        let params = DynamicParams::from_vec(&v, 0, 0);
        // Extra elements (99.0, 100.0) are silently ignored
        assert!((params.proactive_skew_sensitivity - 2.0).abs() < 1e-12);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_serde_roundtrip_json() {
        let original = DynamicParams::default();
        let json = serde_json::to_string(&original).expect("serialize");
        let deserialized: DynamicParams = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_serde_roundtrip_custom_values() {
        let original = DynamicParams {
            gamma_base: 0.33,
            inventory_beta: 11.0,
            spread_floor_bps: 7.5,
            toxic_hour_gamma_mult: 2.8,
            alpha_touch: 0.15,
            kelly_fraction: 0.40,
            cascade_threshold: 0.04,
            proactive_skew_sensitivity: 5.0,
            version: 100,
            source_timestamp_ns: 1_234_567_890,
        };
        let json = serde_json::to_string_pretty(&original).expect("serialize");
        let deserialized: DynamicParams = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(original, deserialized);
    }
}
