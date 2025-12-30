use std::collections::HashMap;

use alloy::primitives::B128;
use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct Meta {
    pub universe: Vec<AssetMeta>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct SpotMeta {
    pub universe: Vec<SpotAssetMeta>,
    pub tokens: Vec<TokenInfo>,
}

impl SpotMeta {
    pub fn add_pair_and_name_to_index_map(
        &self,
        mut coin_to_asset: HashMap<String, u32>,
    ) -> HashMap<String, u32> {
        let index_to_name: HashMap<usize, &str> = self
            .tokens
            .iter()
            .map(|info| (info.index, info.name.as_str()))
            .collect();

        for asset in self.universe.iter() {
            let spot_ind: u32 = 10000 + asset.index as u32;
            let name_to_ind = (asset.name.clone(), spot_ind);

            let Some(token_1_name) = index_to_name.get(&asset.tokens[0]) else {
                continue;
            };

            let Some(token_2_name) = index_to_name.get(&asset.tokens[1]) else {
                continue;
            };

            coin_to_asset.insert(format!("{token_1_name}/{token_2_name}"), spot_ind);
            coin_to_asset.insert(name_to_ind.0, name_to_ind.1);
        }

        coin_to_asset
    }
}

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum SpotMetaAndAssetCtxs {
    SpotMeta(SpotMeta),
    Context(Vec<SpotAssetContext>),
}

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum MetaAndAssetCtxs {
    Meta(Meta),
    Context(Vec<AssetContext>),
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SpotAssetContext {
    pub day_ntl_vlm: String,
    pub mark_px: String,
    pub mid_px: Option<String>,
    pub prev_day_px: String,
    pub circulating_supply: String,
    pub coin: String,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AssetContext {
    pub day_ntl_vlm: String,
    pub funding: String,
    pub impact_pxs: Option<Vec<String>>,
    pub mark_px: String,
    pub mid_px: Option<String>,
    pub open_interest: String,
    pub oracle_px: String,
    pub premium: Option<String>,
    pub prev_day_px: String,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AssetMeta {
    pub name: String,
    pub sz_decimals: u32,
    pub max_leverage: usize,
    #[serde(default)]
    pub only_isolated: Option<bool>,
    /// Margin mode: "strictIsolated" = margin cannot be removed, "noCross" = only isolated allowed
    #[serde(default)]
    pub margin_mode: Option<String>,
    /// Whether the asset is delisted
    #[serde(default)]
    pub is_delisted: Option<bool>,
}

// ============================================================================
// Asset Leverage Configuration
// ============================================================================

/// Leverage configuration derived from exchange API metadata.
///
/// This is the **single source of truth** for leverage limits.
/// Never use hardcoded defaults - always derive from API response.
///
/// # First-Principles Design
/// - Leverage limits are set by the exchange, not the trader
/// - Different assets have different max leverage (BTC: 50x, memecoins: 3x)
/// - Some assets have tiered leverage (lower max at higher notional)
/// - Trading without knowing the real limits is dangerous
#[derive(Debug, Clone)]
pub struct AssetLeverageConfig {
    /// Asset name (e.g., "BTC", "ETH")
    pub asset: String,
    /// Maximum leverage at the base tier (from API metadata)
    pub max_leverage: f64,
    /// Whether this asset only supports isolated margin
    pub isolated_only: bool,
    /// Tiered leverage tiers (optional, for assets with position-based limits)
    pub tiers: Vec<LeverageTier>,
}

/// A single tier in a tiered leverage schedule.
///
/// Hyperliquid reduces max leverage at higher position notional values.
/// Example: 10x up to $3M, then 5x above $3M.
#[derive(Debug, Clone)]
pub struct LeverageTier {
    /// Lower bound of notional position for this tier (USD)
    pub lower_bound: f64,
    /// Max leverage allowed in this tier
    pub max_leverage: f64,
}

impl AssetLeverageConfig {
    /// Create leverage config from asset metadata.
    ///
    /// This is the primary constructor - always use API-derived data.
    pub fn from_asset_meta(meta: &AssetMeta) -> Self {
        Self {
            asset: meta.name.clone(),
            max_leverage: meta.max_leverage as f64,
            isolated_only: meta.only_isolated.unwrap_or(false),
            tiers: vec![], // No tiered data in basic metadata
        }
    }

    /// Create leverage config with tiered schedule.
    ///
    /// Use when margin table data is available from the API.
    pub fn with_tiers(mut self, tiers: Vec<LeverageTier>) -> Self {
        // Sort tiers by lower_bound ascending
        self.tiers = tiers;
        self.tiers
            .sort_by(|a, b| a.lower_bound.partial_cmp(&b.lower_bound).unwrap());
        self
    }

    /// Get effective max leverage at a given position notional.
    ///
    /// For tiered assets, leverage decreases at higher notional values.
    /// This returns the applicable leverage limit for the given position size.
    ///
    /// # Arguments
    /// - `notional`: Current position notional in USD
    ///
    /// # Returns
    /// Maximum leverage allowed at this position size
    pub fn leverage_at_notional(&self, notional: f64) -> f64 {
        if self.tiers.is_empty() {
            return self.max_leverage;
        }

        // Find the highest tier where notional >= lower_bound
        for tier in self.tiers.iter().rev() {
            if notional >= tier.lower_bound {
                return tier.max_leverage;
            }
        }

        // Default to max leverage if below all tiers
        self.max_leverage
    }

    /// Check if a position at given notional exceeds leverage limits.
    ///
    /// # Arguments
    /// - `notional`: Position notional in USD
    /// - `account_value`: Account equity in USD
    ///
    /// # Returns
    /// (is_valid, effective_leverage, max_allowed)
    pub fn validate_leverage(
        &self,
        notional: f64,
        account_value: f64,
    ) -> (bool, f64, f64) {
        if account_value <= 0.0 {
            return (false, f64::INFINITY, self.max_leverage);
        }

        let effective_leverage = notional / account_value;
        let max_allowed = self.leverage_at_notional(notional);
        let is_valid = effective_leverage <= max_allowed;

        (is_valid, effective_leverage, max_allowed)
    }
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SpotAssetMeta {
    pub tokens: [usize; 2],
    pub name: String,
    pub index: usize,
    pub is_canonical: bool,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TokenInfo {
    pub name: String,
    pub sz_decimals: u8,
    pub wei_decimals: u8,
    pub index: usize,
    pub token_id: B128,
    pub is_canonical: bool,
}
