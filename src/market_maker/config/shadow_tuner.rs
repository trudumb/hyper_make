//! Configuration for the Shadow Tuner continuous optimization system.
//!
//! The Shadow Tuner runs a CMA-ES optimizer in a background thread,
//! replaying recent market data with candidate hyperparameters to find
//! optimal macro settings (gamma, inventory penalty, spread floors, etc.).

use serde::{Deserialize, Serialize};

/// Configuration for the Shadow Tuner background optimization thread.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowTunerConfig {
    /// Whether the shadow tuner is enabled.
    #[serde(default)]
    pub enabled: bool,

    /// Seconds between CMA-ES generations (default: 300 = 5 min).
    #[serde(default = "default_cycle_interval_s")]
    pub cycle_interval_s: u64,

    /// Maximum buffer duration in minutes (default: 60).
    /// Events older than this are evicted from the replay buffer.
    #[serde(default = "default_buffer_duration_min")]
    pub buffer_duration_min: u64,

    /// Maximum events to buffer (default: 100_000).
    #[serde(default = "default_max_buffer_events")]
    pub max_buffer_events: usize,

    /// Minimum improvement ratio required to inject new params (default: 0.10 = 10%).
    /// New params must beat current best by this factor.
    #[serde(default = "default_improvement_threshold")]
    pub improvement_threshold: f64,

    /// Sigma threshold for convergence detection (default: 0.01).
    /// When CMA-ES step size falls below this, the optimizer resets.
    #[serde(default = "default_convergence_sigma")]
    pub convergence_sigma: f64,

    /// Maximum generations before forced reset (default: 50).
    #[serde(default = "default_max_generations_before_reset")]
    pub max_generations_before_reset: u64,

    /// Number of rayon threads for parallel population evaluation.
    /// None = use rayon default (all cores), Some(N) = cap at N threads.
    #[serde(default)]
    pub rayon_threads: Option<usize>,

    /// Minimum events required in the buffer before running a generation.
    #[serde(default = "default_min_events_for_replay")]
    pub min_events_for_replay: usize,

    /// Number of blend cycles over which new params are gradually applied (default: 10).
    #[serde(default = "default_blend_cycles")]
    pub blend_cycles: u64,
}

impl Default for ShadowTunerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cycle_interval_s: default_cycle_interval_s(),
            buffer_duration_min: default_buffer_duration_min(),
            max_buffer_events: default_max_buffer_events(),
            improvement_threshold: default_improvement_threshold(),
            convergence_sigma: default_convergence_sigma(),
            max_generations_before_reset: default_max_generations_before_reset(),
            rayon_threads: None,
            min_events_for_replay: default_min_events_for_replay(),
            blend_cycles: default_blend_cycles(),
        }
    }
}

fn default_cycle_interval_s() -> u64 {
    300
}
fn default_buffer_duration_min() -> u64 {
    60
}
fn default_max_buffer_events() -> usize {
    100_000
}
fn default_improvement_threshold() -> f64 {
    0.10
}
fn default_convergence_sigma() -> f64 {
    0.01
}
fn default_max_generations_before_reset() -> u64 {
    50
}
fn default_min_events_for_replay() -> usize {
    5000
}
fn default_blend_cycles() -> u64 {
    10
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ShadowTunerConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.cycle_interval_s, 300);
        assert_eq!(config.buffer_duration_min, 60);
        assert_eq!(config.max_buffer_events, 100_000);
        assert!((config.improvement_threshold - 0.10).abs() < 1e-10);
        assert!((config.convergence_sigma - 0.01).abs() < 1e-10);
        assert_eq!(config.max_generations_before_reset, 50);
        assert!(config.rayon_threads.is_none());
        assert_eq!(config.min_events_for_replay, 5000);
        assert_eq!(config.blend_cycles, 10);
    }

    #[test]
    fn test_serde_roundtrip() {
        let config = ShadowTunerConfig {
            enabled: true,
            cycle_interval_s: 120,
            rayon_threads: Some(4),
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let restored: ShadowTunerConfig = serde_json::from_str(&json).unwrap();
        assert!(restored.enabled);
        assert_eq!(restored.cycle_interval_s, 120);
        assert_eq!(restored.rayon_threads, Some(4));
    }

    #[test]
    fn test_serde_defaults_on_missing_fields() {
        let json = r#"{"enabled": true}"#;
        let config: ShadowTunerConfig = serde_json::from_str(json).unwrap();
        assert!(config.enabled);
        assert_eq!(config.cycle_interval_s, 300);
        assert_eq!(config.buffer_duration_min, 60);
    }
}
