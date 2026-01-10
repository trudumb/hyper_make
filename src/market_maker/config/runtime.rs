//! Pre-computed asset runtime configuration for zero-overhead hot paths.

use std::sync::Arc;

use crate::meta::AssetMeta;

/// Pre-computed asset configuration for zero-overhead hot paths.
///
/// All HIP-3 detection is done ONCE at startup. Quote cycle uses only
/// primitive fields (bool, f64) with no Option unwraps or string comparisons.
///
/// # Design Principle: ZERO HOT-PATH OVERHEAD
///
/// This struct resolves all conditional logic at construction time:
/// - `is_cross`: Pre-computed from margin mode detection
/// - `oi_cap_usd`: Pre-resolved (f64::MAX if no cap)
/// - `sz_multiplier`: Pre-computed 10^sz_decimals
///
/// # Fee Handling
///
/// Fees are NOT pre-computed. HIP-3 builder fees vary per deployer
/// (0-300% share) and are included in the `fee` field of each fill.
/// The `builderFee` field in fills contains the builder's portion.
#[derive(Debug, Clone)]
pub struct AssetRuntimeConfig {
    // === Pre-computed margin fields (NO OPTIONS) ===
    /// Whether to use cross margin (pre-computed from AssetMeta).
    /// HOT PATH: Used directly in margin calculations and leverage API calls.
    pub is_cross: bool,

    /// Open interest cap in USD (f64::MAX if no cap).
    /// HOT PATH: Pre-flight check before order placement.
    pub oi_cap_usd: f64,

    /// Pre-computed sz_decimals as f64 power for truncation.
    /// HOT PATH: Avoids powi() call in size formatting.
    pub sz_multiplier: f64,

    /// Pre-computed price decimals multiplier.
    /// For perps: 10^5 (5 significant figures).
    pub price_multiplier: f64,

    // === Cold path metadata (startup only) ===
    /// Asset name (Arc for cheap cloning).
    pub asset: Arc<str>,

    /// Maximum leverage (from API).
    pub max_leverage: f64,

    /// Whether this is a HIP-3 builder-deployed asset.
    pub is_hip3: bool,

    /// Deployer address (for logging/display only).
    pub deployer: Option<Arc<str>>,
}

impl AssetRuntimeConfig {
    /// Build from API metadata - called ONCE at startup.
    ///
    /// This resolves all HIP-3 detection and margin mode logic upfront
    /// so the hot path has zero conditional overhead.
    pub fn from_asset_meta(meta: &AssetMeta) -> Self {
        let is_hip3 = meta.is_hip3();

        Self {
            // Pre-compute for hot path
            is_cross: !is_hip3,
            oi_cap_usd: meta.oi_cap_usd.unwrap_or(f64::MAX),
            sz_multiplier: 10_f64.powi(meta.sz_decimals as i32),
            price_multiplier: 10_f64.powi(5), // 5 sig figs for perps

            // Cold path storage
            asset: Arc::from(meta.name.as_str()),
            max_leverage: meta.max_leverage as f64,
            is_hip3,
            deployer: meta.deployer.as_ref().map(|d| Arc::from(d.as_str())),
        }
    }

    /// Fast size truncation (hot path).
    ///
    /// Uses pre-computed multiplier to avoid powi() in hot path.
    #[inline(always)]
    pub fn truncate_size(&self, size: f64) -> f64 {
        (size * self.sz_multiplier).trunc() / self.sz_multiplier
    }

    /// Check OI cap (hot path) - returns max additional notional allowed.
    ///
    /// Returns 0.0 if current_oi >= cap, otherwise returns remaining capacity.
    /// For unlimited assets (oi_cap_usd == f64::MAX), returns f64::MAX.
    #[inline(always)]
    pub fn remaining_oi_capacity(&self, current_oi: f64) -> f64 {
        (self.oi_cap_usd - current_oi).max(0.0)
    }

    /// Format OI cap for display (cold path).
    pub fn oi_cap_display(&self) -> String {
        if self.oi_cap_usd == f64::MAX {
            "unlimited".to_string()
        } else {
            format!("${:.0}", self.oi_cap_usd)
        }
    }
}

impl Default for AssetRuntimeConfig {
    /// Default config for testing - represents a standard validator perp.
    fn default() -> Self {
        Self {
            is_cross: true,
            oi_cap_usd: f64::MAX,
            sz_multiplier: 100_000.0, // 5 decimals
            price_multiplier: 100_000.0,
            asset: Arc::from("BTC"),
            max_leverage: 50.0,
            is_hip3: false,
            deployer: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hip3_asset_meta() -> AssetMeta {
        AssetMeta {
            name: "HIP3COIN".to_string(),
            sz_decimals: 2,
            max_leverage: 10,
            only_isolated: Some(true),
            margin_mode: Some("noCross".to_string()),
            is_delisted: None,
            deployer: Some("0xbuilder".to_string()),
            dex_id: Some(5),
            oi_cap_usd: Some(5_000_000.0),
            is_builder_deployed: Some(true),
        }
    }

    fn make_validator_asset_meta() -> AssetMeta {
        AssetMeta {
            name: "BTC".to_string(),
            sz_decimals: 5,
            max_leverage: 50,
            only_isolated: None,
            margin_mode: None,
            is_delisted: None,
            deployer: None,
            dex_id: None,
            oi_cap_usd: None,
            is_builder_deployed: None,
        }
    }

    #[test]
    fn test_runtime_config_from_hip3_asset() {
        let meta = make_hip3_asset_meta();
        let config = AssetRuntimeConfig::from_asset_meta(&meta);

        assert!(config.is_hip3);
        assert!(!config.is_cross); // HIP-3 = isolated only
        assert_eq!(config.oi_cap_usd, 5_000_000.0);
        assert_eq!(config.max_leverage, 10.0);
        assert!(config.deployer.is_some());
        assert_eq!(config.deployer.as_deref(), Some("0xbuilder"));
    }

    #[test]
    fn test_runtime_config_from_validator_perp() {
        let meta = make_validator_asset_meta();
        let config = AssetRuntimeConfig::from_asset_meta(&meta);

        assert!(!config.is_hip3);
        assert!(config.is_cross); // Validator perps support cross margin
        assert_eq!(config.oi_cap_usd, f64::MAX); // No OI cap
        assert_eq!(config.max_leverage, 50.0);
        assert!(config.deployer.is_none());
    }

    #[test]
    fn test_oi_cap_remaining_capacity() {
        let meta = make_hip3_asset_meta();
        let config = AssetRuntimeConfig::from_asset_meta(&meta);

        // With 1M position, remaining = 5M - 1M = 4M
        let remaining = config.remaining_oi_capacity(1_000_000.0);
        assert!((remaining - 4_000_000.0).abs() < 0.01);

        // At cap, remaining = 0
        let remaining = config.remaining_oi_capacity(5_000_000.0);
        assert_eq!(remaining, 0.0);

        // Over cap, remaining = 0 (clamped)
        let remaining = config.remaining_oi_capacity(6_000_000.0);
        assert_eq!(remaining, 0.0);
    }

    #[test]
    fn test_oi_cap_no_limit_validator_perp() {
        let meta = make_validator_asset_meta();
        let config = AssetRuntimeConfig::from_asset_meta(&meta);

        // With no cap (f64::MAX), remaining is effectively infinite
        let remaining = config.remaining_oi_capacity(1_000_000_000.0);
        assert!(remaining > 1e15); // Still huge
    }

    #[test]
    fn test_size_truncation() {
        let meta = make_hip3_asset_meta(); // sz_decimals = 2
        let config = AssetRuntimeConfig::from_asset_meta(&meta);

        // 1.234 should truncate to 1.23
        let truncated = config.truncate_size(1.234);
        assert!((truncated - 1.23).abs() < 1e-10);

        // 0.999 should truncate to 0.99
        let truncated = config.truncate_size(0.999);
        assert!((truncated - 0.99).abs() < 1e-10);
    }
}
