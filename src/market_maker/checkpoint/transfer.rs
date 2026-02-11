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

/// Configuration for prior injection.
#[derive(Debug, Clone)]
pub struct InjectionConfig {
    /// RL Q-table blend weight (0.0 = ignore prior, 1.0 = full prior).
    /// Typical: 0.3 for paper→live transfer.
    pub rl_blend_weight: f64,
    /// Maximum age of prior before it's considered stale (seconds).
    /// Priors older than this are rejected.
    pub max_prior_age_s: f64,
    /// If true, reject priors from a different asset.
    pub require_asset_match: bool,
    /// Skip RL Q-table injection entirely.
    pub skip_rl: bool,
    /// Skip kill switch state injection (safety: never inherit kill state from paper).
    pub skip_kill_switch: bool,
}

impl Default for InjectionConfig {
    fn default() -> Self {
        Self {
            rl_blend_weight: 0.3,
            max_prior_age_s: 4.0 * 3600.0, // 4 hours
            require_asset_match: true,
            skip_rl: false,
            skip_kill_switch: true, // Never inherit kill switch from paper by default
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
