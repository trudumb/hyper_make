//! Multi-asset market making configuration.

use super::spread_profile::SpreadProfile;

/// Configuration for multi-asset market making mode.
///
/// Enables quoting multiple assets from a single capital pool with
/// volatility-weighted order allocation. Utilizes the 1000-order limit
/// for improved capital efficiency.
///
/// # Example
///
/// ```ignore
/// let config = MultiAssetConfig {
///     assets: vec![
///         AssetSpec::new("BTC", None),
///         AssetSpec::new("ETH", None),
///         AssetSpec::new("HYPE", Some("hyna")),
///     ],
///     total_order_limit: 500,
///     min_levels_per_asset: 5,
///     max_levels_per_asset: 15,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct MultiAssetConfig {
    /// List of assets to quote.
    pub assets: Vec<AssetSpec>,

    /// Total order limit across all assets (default: 1000).
    /// Per Hyperliquid docs: 1000 base + 1 per 5M USDC volume, capped at 5000.
    pub total_order_limit: usize,

    /// Minimum ladder levels per asset (default: 5).
    /// Ensures every asset gets meaningful liquidity.
    pub min_levels_per_asset: usize,

    /// Maximum ladder levels per asset (default: 25).
    /// Caps allocation to prevent over-concentration in low-vol assets.
    pub max_levels_per_asset: usize,

    /// Rebalance interval in seconds (default: 300 = 5 minutes).
    /// How often to recompute volatility-weighted allocations.
    pub rebalance_interval_secs: u64,

    /// Maximum concentration per asset (default: 0.30 = 30%).
    /// No single asset can consume more than this fraction of total orders.
    pub max_concentration_pct: f64,

    /// Spread profile to use for all assets (default: Default).
    /// Can be overridden per-asset via AssetSpec.
    pub default_spread_profile: SpreadProfile,
}

impl Default for MultiAssetConfig {
    fn default() -> Self {
        Self {
            assets: Vec::new(),
            total_order_limit: 1000,
            min_levels_per_asset: 5,
            max_levels_per_asset: 25,
            rebalance_interval_secs: 300,
            max_concentration_pct: 0.30,
            default_spread_profile: SpreadProfile::Default,
        }
    }
}

impl MultiAssetConfig {
    /// Create a new multi-asset config with the given assets.
    pub fn new(assets: Vec<AssetSpec>) -> Self {
        Self {
            assets,
            ..Default::default()
        }
    }

    /// Builder: set total order limit.
    pub fn with_total_order_limit(mut self, limit: usize) -> Self {
        self.total_order_limit = limit;
        self
    }

    /// Builder: set min/max levels per asset.
    pub fn with_levels(mut self, min: usize, max: usize) -> Self {
        self.min_levels_per_asset = min;
        self.max_levels_per_asset = max;
        self
    }

    /// Builder: set rebalance interval.
    pub fn with_rebalance_interval(mut self, secs: u64) -> Self {
        self.rebalance_interval_secs = secs;
        self
    }

    /// Builder: set max concentration.
    pub fn with_max_concentration(mut self, pct: f64) -> Self {
        self.max_concentration_pct = pct;
        self
    }

    /// Builder: set default spread profile.
    pub fn with_spread_profile(mut self, profile: SpreadProfile) -> Self {
        self.default_spread_profile = profile;
        self
    }

    /// Check if multi-asset mode is enabled (has at least 2 assets).
    pub fn is_enabled(&self) -> bool {
        self.assets.len() >= 2
    }

    /// Get the number of configured assets.
    pub fn asset_count(&self) -> usize {
        self.assets.len()
    }

    /// Calculate orders per asset if evenly distributed.
    /// Used for fallback when volatility data unavailable.
    pub fn orders_per_asset_even(&self) -> usize {
        if self.assets.is_empty() {
            return 0;
        }
        let orders_per = self.total_order_limit / self.assets.len();
        let levels = orders_per / 2; // 2 sides per level
        levels.clamp(self.min_levels_per_asset, self.max_levels_per_asset)
    }
}

/// Specification for a single asset in multi-asset mode.
#[derive(Debug, Clone)]
pub struct AssetSpec {
    /// Asset symbol (e.g., "BTC", "ETH", "HYPE").
    pub symbol: String,

    /// Optional DEX name for HIP-3 assets (e.g., "hyna", "felix").
    /// If None, uses validator perps.
    pub dex: Option<String>,

    /// Optional weight override (0.0-1.0).
    /// If set, bypasses volatility-weighted allocation for this asset.
    pub weight_override: Option<f64>,

    /// Whether this asset is enabled (default: true).
    pub enabled: bool,

    /// Optional spread profile override for this asset.
    pub spread_profile: Option<SpreadProfile>,
}

impl AssetSpec {
    /// Create a new asset spec.
    pub fn new(symbol: &str, dex: Option<&str>) -> Self {
        Self {
            symbol: symbol.to_string(),
            dex: dex.map(|s| s.to_string()),
            weight_override: None,
            enabled: true,
            spread_profile: None,
        }
    }

    /// Create a HIP-3 asset spec.
    pub fn hip3(symbol: &str, dex: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            dex: Some(dex.to_string()),
            weight_override: None,
            enabled: true,
            spread_profile: Some(SpreadProfile::Hip3),
        }
    }

    /// Builder: set weight override.
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight_override = Some(weight);
        self
    }

    /// Builder: set enabled flag.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Builder: set spread profile.
    pub fn with_spread_profile(mut self, profile: SpreadProfile) -> Self {
        self.spread_profile = Some(profile);
        self
    }

    /// Get the effective asset name for API calls.
    /// For HIP-3: "dex:symbol", for validator perps: "symbol".
    pub fn effective_asset(&self) -> String {
        match &self.dex {
            Some(dex) => format!("{}:{}", dex, self.symbol),
            None => self.symbol.clone(),
        }
    }

    /// Check if this is a HIP-3 asset.
    pub fn is_hip3(&self) -> bool {
        self.dex.is_some()
    }
}
