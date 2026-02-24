//! Prior transfer protocol: φ (extract) and ψ (inject).
//!
//! These traits formalize the checkpoint transfer between phases:
//! ```text
//! S_paper →[φ: extract_prior()]→ P →[ψ: inject_prior()]→ S_live
//! ```
//!
//! Where P = [`CheckpointBundle`] is the mediating prior object.

use super::types::CheckpointBundle;

/// φ: S → P — extract learned priors from market maker state.
pub trait PriorExtract {
    /// Extract a checkpoint bundle containing all learned priors.
    fn extract_prior(&self) -> CheckpointBundle;
}

/// Policy for prior age decay — replaces hard 4h cutoff with exponential decay.
///
/// Theoretical priors (market structure learned in paper) decay slowly because
/// market microstructure is relatively stable. The curve:
/// - `[0, full_confidence_age_s]`: confidence = 1.0
/// - `(full_confidence_age_s, decay_horizon_s]`: exponential decay to `confidence_floor`
/// - `(decay_horizon_s, hard_reject_age_s]`: held at `confidence_floor`
/// - `> hard_reject_age_s`: confidence = 0.0 (hard reject)
#[derive(Debug, Clone)]
pub struct PriorAgePolicy {
    /// Full confidence below this age (seconds). Default: 4h.
    pub full_confidence_age_s: f64,
    /// Exponential decay horizon (seconds). Default: 72h.
    pub decay_horizon_s: f64,
    /// Floor confidence — even old priors beat cold-start. Default: 0.15.
    pub confidence_floor: f64,
    /// Hard reject age (seconds). Beyond this, prior is truly stale. Default: 168h (7 days).
    pub hard_reject_age_s: f64,
}

impl Default for PriorAgePolicy {
    fn default() -> Self {
        Self {
            full_confidence_age_s: 4.0 * 3600.0, // 4 hours
            decay_horizon_s: 72.0 * 3600.0,      // 72 hours
            confidence_floor: 0.15,
            hard_reject_age_s: 168.0 * 3600.0, // 7 days
        }
    }
}

/// Compute age-based confidence for a prior [0.0, 1.0].
///
/// Returns 1.0 for fresh priors, exponentially decays to `confidence_floor`,
/// and returns 0.0 beyond hard_reject_age_s.
pub fn prior_age_confidence(age_s: f64, policy: &PriorAgePolicy) -> f64 {
    if age_s <= 0.0 {
        return 1.0;
    }
    if age_s > policy.hard_reject_age_s {
        return 0.0;
    }
    if age_s <= policy.full_confidence_age_s {
        return 1.0;
    }

    // Exponential decay from 1.0 to confidence_floor over the decay window
    let decay_window = policy.decay_horizon_s - policy.full_confidence_age_s;
    if decay_window <= 0.0 {
        return policy.confidence_floor;
    }

    let elapsed_in_decay = age_s - policy.full_confidence_age_s;
    let fraction = (elapsed_in_decay / decay_window).min(1.0);

    // Exponential: c(t) = floor + (1 - floor) * exp(-3 * fraction)
    // -3 gives ~95% decay at fraction=1.0 (i.e., at decay_horizon)
    let decay_range = 1.0 - policy.confidence_floor;
    let confidence = policy.confidence_floor + decay_range * (-3.0 * fraction).exp();

    confidence.clamp(policy.confidence_floor, 1.0)
}

/// Configuration for prior injection.
#[derive(Debug, Clone)]
pub struct InjectionConfig {
    /// Deprecated: RL removed. Kept for config compatibility.
    pub rl_blend_weight: f64,
    /// Maximum age of prior before it's considered stale (seconds).
    /// Now only used as legacy fallback — prefer `age_policy` for soft decay.
    pub max_prior_age_s: f64,
    /// If true, reject priors from a different asset.
    pub require_asset_match: bool,
    /// Deprecated: RL removed. Kept for config compatibility.
    pub skip_rl: bool,
    /// Skip kill switch state injection (safety: never inherit kill state from paper).
    pub skip_kill_switch: bool,
    /// Age-based confidence policy (replaces hard max_prior_age_s cutoff).
    pub age_policy: PriorAgePolicy,
}

impl Default for InjectionConfig {
    fn default() -> Self {
        let age_policy = PriorAgePolicy::default();
        Self {
            rl_blend_weight: 0.3,
            max_prior_age_s: age_policy.hard_reject_age_s, // Use hard_reject as legacy compat
            require_asset_match: true,
            skip_rl: false,
            skip_kill_switch: true, // Never inherit kill switch from paper by default
            age_policy,
        }
    }
}

/// ψ: P × S → S — inject priors into market maker state.
pub trait PriorInject {
    /// Inject a checkpoint bundle as prior into the current state.
    ///
    /// Returns the number of RL states injected (0 if RL is skipped or empty).
    fn inject_prior(&mut self, prior: &CheckpointBundle, config: &InjectionConfig) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prior_age_confidence_fresh() {
        let policy = PriorAgePolicy::default();
        assert_eq!(prior_age_confidence(0.0, &policy), 1.0);
        assert_eq!(prior_age_confidence(100.0, &policy), 1.0);
        assert_eq!(prior_age_confidence(3600.0, &policy), 1.0); // 1h < 4h
    }

    #[test]
    fn test_prior_age_confidence_within_full_window() {
        let policy = PriorAgePolicy::default();
        // Exactly at full_confidence boundary
        assert_eq!(prior_age_confidence(4.0 * 3600.0, &policy), 1.0);
    }

    #[test]
    fn test_prior_age_confidence_decaying() {
        let policy = PriorAgePolicy::default();
        // Halfway through decay window (~38h)
        let age_s = 4.0 * 3600.0 + (72.0 - 4.0) * 3600.0 * 0.5;
        let c = prior_age_confidence(age_s, &policy);
        assert!(c > policy.confidence_floor, "Should be above floor: {c}");
        assert!(c < 1.0, "Should be below 1.0: {c}");
        // ~0.15 + 0.85 * exp(-1.5) ≈ 0.15 + 0.85*0.223 ≈ 0.34
        assert!(c > 0.25 && c < 0.45, "Expected ~0.34, got {c}");
    }

    #[test]
    fn test_prior_age_confidence_at_decay_horizon() {
        let policy = PriorAgePolicy::default();
        let c = prior_age_confidence(72.0 * 3600.0, &policy);
        // At decay horizon: floor + (1-floor)*exp(-3) ≈ 0.15 + 0.85*0.05 ≈ 0.19
        assert!(c >= policy.confidence_floor, "Should be >= floor: {c}");
        assert!(c < 0.25, "Should be near floor: {c}");
    }

    #[test]
    fn test_prior_age_confidence_hard_reject() {
        let policy = PriorAgePolicy::default();
        // Beyond 7 days
        assert_eq!(prior_age_confidence(168.0 * 3600.0 + 1.0, &policy), 0.0);
        assert_eq!(prior_age_confidence(200.0 * 3600.0, &policy), 0.0);
    }
}
