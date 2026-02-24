//! Prior discovery: search multiple candidate paths for paper-to-live transfer.
//!
//! Paper saves checkpoints under different directory structures depending on
//! how the asset was specified. This module searches all reasonable candidates
//! and returns the best (newest valid) match.

use std::path::{Path, PathBuf};

use tracing::{debug, info, warn};

use super::asset_identity::base_symbol;
use super::types::CheckpointBundle;

/// A discovered prior with metadata.
#[derive(Debug)]
pub struct DiscoveredPrior {
    /// The parsed checkpoint bundle.
    pub bundle: CheckpointBundle,
    /// Path where the prior was found.
    pub path: PathBuf,
    /// Age of the prior in seconds.
    pub age_s: f64,
}

/// Search for a paper prior across multiple candidate paths.
///
/// Candidate paths searched (in order):
/// 1. Explicit `--paper-checkpoint` override (if Some)
/// 2. `paper/{base_symbol}/prior.json`
/// 3. `paper/{dex}:{base_symbol}/prior.json`
/// 4. `paper/{base_symbol}/latest/checkpoint.json`
/// 5. `paper/{dex}:{base_symbol}/latest/checkpoint.json`
///
/// Returns the newest valid match. If `explicit_path` is provided and valid,
/// it always wins (user override).
pub fn discover_prior(
    asset: &str,
    dex: Option<&str>,
    checkpoint_root: &str,
    explicit_path: Option<&str>,
) -> Option<DiscoveredPrior> {
    let base = base_symbol(asset);
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let mut candidates: Vec<PathBuf> = Vec::new();

    // 1. Explicit override (highest priority)
    if let Some(explicit) = explicit_path {
        let p = Path::new(explicit);
        if p.extension().is_some() {
            // Direct file path
            candidates.push(p.to_path_buf());
        } else {
            // Directory — check for prior.json and latest/checkpoint.json
            candidates.push(p.join("prior.json"));
            candidates.push(p.join("latest/checkpoint.json"));
        }
    }

    // 2. paper/{base_symbol}/prior.json
    candidates.push(Path::new(checkpoint_root).join(base).join("prior.json"));

    // 3. paper/{dex}:{base_symbol}/prior.json (if dex specified)
    if let Some(dex_name) = dex {
        candidates.push(
            Path::new(checkpoint_root)
                .join(format!("{dex_name}_{base}"))
                .join("prior.json"),
        );
    }

    // 4. paper/{base_symbol}/latest/checkpoint.json
    candidates.push(
        Path::new(checkpoint_root)
            .join(base)
            .join("latest/checkpoint.json"),
    );

    // 5. paper/{dex}:{base_symbol}/latest/checkpoint.json
    if let Some(dex_name) = dex {
        candidates.push(
            Path::new(checkpoint_root)
                .join(format!("{dex_name}_{base}"))
                .join("latest/checkpoint.json"),
        );
    }

    let mut best: Option<DiscoveredPrior> = None;

    for candidate in &candidates {
        if !candidate.exists() {
            debug!(path = %candidate.display(), "Prior candidate not found");
            continue;
        }

        match std::fs::read_to_string(candidate) {
            Ok(json) => match serde_json::from_str::<CheckpointBundle>(&json) {
                Ok(bundle) => {
                    let age_s =
                        (now_ms.saturating_sub(bundle.metadata.timestamp_ms)) as f64 / 1000.0;

                    info!(
                        path = %candidate.display(),
                        asset = %bundle.metadata.asset,
                        age_h = %format!("{:.1}", age_s / 3600.0),
                        "Found prior candidate"
                    );

                    // If explicit path provided and valid, always use it
                    if explicit_path.is_some()
                        && candidates
                            .iter()
                            .take(2) // First 2 are explicit-derived
                            .any(|c| c == candidate)
                    {
                        return Some(DiscoveredPrior {
                            bundle,
                            path: candidate.clone(),
                            age_s,
                        });
                    }

                    // Otherwise, keep newest
                    let dominated = best.as_ref().is_some_and(|b| {
                        b.bundle.metadata.timestamp_ms >= bundle.metadata.timestamp_ms
                    });
                    if !dominated {
                        best = Some(DiscoveredPrior {
                            bundle,
                            path: candidate.clone(),
                            age_s,
                        });
                    }
                }
                Err(e) => {
                    warn!(
                        path = %candidate.display(),
                        error = %e,
                        "Failed to parse prior checkpoint"
                    );
                }
            },
            Err(e) => {
                warn!(
                    path = %candidate.display(),
                    error = %e,
                    "Failed to read prior checkpoint file"
                );
            }
        }
    }

    if let Some(ref found) = best {
        info!(
            path = %found.path.display(),
            age_h = %format!("{:.1}", found.age_s / 3600.0),
            "Selected best prior"
        );
    } else {
        info!(
            asset = %asset,
            candidates_searched = candidates.len(),
            "No valid prior found across any candidate path"
        );
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::checkpoint::types::CheckpointBundle;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn test_dir() -> PathBuf {
        let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("discovery_test_{pid}_{id}"));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn make_prior(asset: &str, timestamp_ms: u64) -> CheckpointBundle {
        let json = format!(
            r#"{{"metadata":{{"version":1,"timestamp_ms":{timestamp_ms},"asset":"{asset}","session_duration_s":1800.0}}}}"#
        );
        serde_json::from_str(&json).expect("valid test prior")
    }

    fn save_prior(dir: &Path, filename: &str, bundle: &CheckpointBundle) {
        let path = dir.join(filename);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let json = serde_json::to_string_pretty(bundle).unwrap();
        std::fs::write(path, json).unwrap();
    }

    #[test]
    fn test_discover_base_symbol_path() {
        let root = test_dir();
        let prior = make_prior("HYPE", 1700000000000);
        save_prior(&root, "HYPE/prior.json", &prior);

        let found = discover_prior("hyna:HYPE", Some("hyna"), root.to_str().unwrap(), None);
        assert!(found.is_some());
        assert!(found.unwrap().path.ends_with("HYPE/prior.json"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_discover_dex_prefixed_path() {
        let root = test_dir();
        let prior = make_prior("hyna:HYPE", 1700000000000);
        save_prior(&root, "hyna_HYPE/prior.json", &prior);

        let found = discover_prior("HYPE", Some("hyna"), root.to_str().unwrap(), None);
        assert!(found.is_some());
        assert!(found.unwrap().path.to_str().unwrap().contains("hyna_HYPE"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_discover_explicit_override() {
        let root = test_dir();
        let prior = make_prior("HYPE", 1700000000000);
        let explicit_dir = root.join("custom");
        save_prior(&explicit_dir, "prior.json", &prior);

        let found = discover_prior(
            "HYPE",
            None,
            root.to_str().unwrap(),
            Some(explicit_dir.to_str().unwrap()),
        );
        assert!(found.is_some());
        assert!(found.unwrap().path.to_str().unwrap().contains("custom"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_discover_no_prior() {
        let root = test_dir();
        std::fs::create_dir_all(&root).unwrap();

        let found = discover_prior("HYPE", None, root.to_str().unwrap(), None);
        assert!(found.is_none());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_discover_newest_wins() {
        let root = test_dir();

        // Old prior in base path
        let old_prior = make_prior("HYPE", 1700000000000);
        save_prior(&root, "HYPE/prior.json", &old_prior);

        // Newer prior in dex-prefixed path
        let new_prior = make_prior("hyna:HYPE", 1700000099000);
        save_prior(&root, "hyna_HYPE/prior.json", &new_prior);

        let found = discover_prior("HYPE", Some("hyna"), root.to_str().unwrap(), None);
        assert!(found.is_some());
        let found = found.unwrap();
        assert_eq!(found.bundle.metadata.timestamp_ms, 1700000099000);

        let _ = std::fs::remove_dir_all(&root);
    }
}
