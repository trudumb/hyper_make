use std::collections::HashMap;

use alloy::primitives::B128;
use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Meta {
    pub universe: Vec<AssetMeta>,
    /// Collateral token index for HIP-3 DEXs.
    /// Only present when querying with `dex` parameter.
    /// Index 0 = USDC, other indices map to spot tokens.
    #[serde(default)]
    pub collateral_token: Option<u32>,
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

/// Asset deployment type - compact representation for startup logic.
/// Uses repr(u8) for minimal memory footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AssetType {
    /// Standard validator-operated perpetual (supports cross margin)
    ValidatorPerp = 0,
    /// HIP-3 builder-deployed perpetual (isolated margin only)
    BuilderPerp = 1,
    /// Spot trading pair
    Spot = 2,
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

    // === HIP-3 specific fields (parsed from API, used only at startup) ===
    /// Deployer address for HIP-3 builder-deployed assets
    #[serde(default)]
    pub deployer: Option<String>,
    /// DEX ID for HIP-3 assets
    #[serde(default)]
    pub dex_id: Option<u32>,
    /// Open interest cap in USD for HIP-3 assets
    #[serde(default)]
    pub oi_cap_usd: Option<f64>,
    /// Whether this is a HIP-3 builder-deployed asset
    #[serde(default)]
    pub is_builder_deployed: Option<bool>,
}

impl AssetMeta {
    /// Check if this is a HIP-3 builder-deployed asset.
    ///
    /// HIP-3 assets are identified by any of:
    /// - `is_builder_deployed: true`
    /// - `only_isolated: true`
    /// - `margin_mode: "noCross"` or `"strictIsolated"`
    ///
    /// Called ONCE at startup - do NOT use in hot paths.
    #[inline]
    pub fn is_hip3(&self) -> bool {
        self.is_builder_deployed.unwrap_or(false)
            || self.only_isolated.unwrap_or(false)
            || self.margin_mode.as_deref() == Some("noCross")
            || self.margin_mode.as_deref() == Some("strictIsolated")
    }

    /// Check if this asset allows cross margin.
    ///
    /// Returns false for HIP-3 assets which are isolated-only.
    /// Called ONCE at startup - do NOT use in hot paths.
    #[inline]
    pub fn allows_cross_margin(&self) -> bool {
        !self.is_hip3()
    }

    /// Determine asset type - called ONCE at startup.
    /// Do NOT call in hot paths - use `AssetRuntimeConfig.is_hip3` instead.
    pub fn asset_type(&self) -> AssetType {
        if self.is_hip3() {
            AssetType::BuilderPerp
        } else {
            AssetType::ValidatorPerp
        }
    }
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
    pub fn validate_leverage(&self, notional: f64, account_value: f64) -> (bool, f64, f64) {
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

// ============================================================================
// Collateral Token Info (HIP-3 Multi-Quote Asset Support)
// ============================================================================

/// Collateral/quote asset information for a perp DEX.
///
/// Different HIP-3 DEXs can use different stablecoins as collateral:
/// - Validator perps (default): USDC (index 0)
/// - HIP-3 DEXs: May use USDE (USDe), USDH, or other tokens
///
/// This struct holds the resolved token information for margin/P&L display.
#[derive(Debug, Clone)]
pub struct CollateralInfo {
    /// Token index (0 = USDC, 235 = USDE, 360 = USDH, etc.)
    pub token_index: u32,
    /// Token symbol (e.g., "USDC", "USDE", "USDH")
    pub symbol: String,
    /// Full name if available (e.g., "USDe", "USDH")
    pub full_name: Option<String>,
    /// Size decimals for display
    pub sz_decimals: u8,
    /// Wei decimals for on-chain precision
    pub wei_decimals: u8,
}

impl CollateralInfo {
    /// Create CollateralInfo for USDC (the default collateral).
    pub fn usdc() -> Self {
        Self {
            token_index: 0,
            symbol: "USDC".to_string(),
            full_name: None,
            sz_decimals: 8,
            wei_decimals: 8,
        }
    }

    /// Create CollateralInfo from a token index and spot metadata.
    ///
    /// Looks up the token in the spot metadata to get name and decimals.
    /// Falls back to USDC if token not found.
    pub fn from_token_index(token_index: u32, spot_meta: &SpotMeta) -> Self {
        // USDC is always index 0
        if token_index == 0 {
            return Self::usdc();
        }

        // Look up token in spot metadata
        if let Some(token) = spot_meta.tokens.iter().find(|t| t.index == token_index as usize) {
            Self {
                token_index,
                symbol: token.name.clone(),
                full_name: None, // TokenInfo doesn't have full_name
                sz_decimals: token.sz_decimals,
                wei_decimals: token.wei_decimals,
            }
        } else {
            // Token not found - use placeholder
            Self {
                token_index,
                symbol: format!("TOKEN_{}", token_index),
                full_name: None,
                sz_decimals: 8,
                wei_decimals: 8,
            }
        }
    }

    /// Check if this is the default USDC collateral.
    #[inline]
    pub fn is_usdc(&self) -> bool {
        self.token_index == 0
    }

    /// Get display string for logging (e.g., "USDE" or "USDC")
    #[inline]
    pub fn display(&self) -> &str {
        &self.symbol
    }

    /// Extract the balance for this collateral from spot balances.
    ///
    /// Returns (total, hold) for this token, or None if not found.
    /// The `hold` amount represents locked balance (e.g., for open orders).
    pub fn balance_from_spot(
        &self,
        balances: &[crate::types::UserTokenBalance],
    ) -> Option<(f64, f64)> {
        balances
            .iter()
            .find(|b| b.coin == self.symbol)
            .and_then(|b| {
                let total: f64 = b.total.parse().ok()?;
                let hold: f64 = b.hold.parse().ok()?;
                Some((total, hold))
            })
    }

    /// Get available (withdrawable) balance for this collateral.
    ///
    /// Available = Total - Hold
    pub fn available_balance_from_spot(
        &self,
        balances: &[crate::types::UserTokenBalance],
    ) -> Option<f64> {
        self.balance_from_spot(balances).map(|(total, hold)| total - hold)
    }

    /// Get the token symbol for use in spot transfers.
    ///
    /// This is the token name used in `ExchangeClient::spot_transfer()`.
    #[inline]
    pub fn transfer_token(&self) -> &str {
        &self.symbol
    }

    /// Format an amount for transfer using this collateral's decimal precision.
    ///
    /// # Arguments
    /// * `amount` - The amount to format
    ///
    /// # Returns
    /// String formatted with appropriate decimal places for this token.
    pub fn format_amount(&self, amount: f64) -> String {
        let precision = self.sz_decimals as usize;
        format!("{:.prec$}", amount, prec = precision)
    }
}

impl Default for CollateralInfo {
    fn default() -> Self {
        Self::usdc()
    }
}

impl std::fmt::Display for CollateralInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.symbol)
    }
}

// ============================================================================
// HIP-3 DEX Types
// ============================================================================

/// HIP-3 DEX information from the `perpDexs` API endpoint.
///
/// Each entry represents a HIP-3 builder-deployed exchange.
/// The list includes `null` entries for unregistered DEX IDs.
#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PerpDex {
    /// Short DEX name (≤6 characters, e.g., "hyena", "felix")
    pub name: String,
    /// Full DEX display name (e.g., "Hyena Exchange")
    pub full_name: String,
    /// Deployer's Ethereum address
    pub deployer: String,
    /// Optional oracle updater address
    pub oracle_updater: Option<String>,
}

/// HIP-3 DEX open interest limits from the `perpDexLimits` API endpoint.
#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PerpDexLimits {
    /// Total open interest cap across all perps in USD
    pub total_oi_cap: String,
    /// Per-perp open interest size cap
    pub oi_sz_cap_per_perp: String,
    /// Maximum transfer notional allowed
    pub max_transfer_ntl: String,
    /// Per-coin OI caps: Vec<(coin_name, oi_cap)>
    pub coin_to_oi_cap: Vec<(String, String)>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_asset_meta(
        name: &str,
        only_isolated: Option<bool>,
        margin_mode: Option<&str>,
        is_builder_deployed: Option<bool>,
    ) -> AssetMeta {
        AssetMeta {
            name: name.to_string(),
            sz_decimals: 5,
            max_leverage: 50,
            only_isolated,
            margin_mode: margin_mode.map(String::from),
            is_delisted: None,
            deployer: None,
            dex_id: None,
            oi_cap_usd: None,
            is_builder_deployed,
        }
    }

    #[test]
    fn test_hip3_detection_only_isolated() {
        let meta = make_asset_meta("MEMECOIN", Some(true), None, None);
        assert!(meta.is_hip3());
        assert!(!meta.allows_cross_margin());
        assert_eq!(meta.asset_type(), AssetType::BuilderPerp);
    }

    #[test]
    fn test_hip3_detection_margin_mode_no_cross() {
        let meta = make_asset_meta("EXOTIC", None, Some("noCross"), Some(true));
        assert!(meta.is_hip3());
        assert!(!meta.allows_cross_margin());
        assert_eq!(meta.asset_type(), AssetType::BuilderPerp);
    }

    #[test]
    fn test_hip3_detection_margin_mode_strict_isolated() {
        let meta = make_asset_meta("STRICT", None, Some("strictIsolated"), None);
        assert!(meta.is_hip3());
        assert!(!meta.allows_cross_margin());
        assert_eq!(meta.asset_type(), AssetType::BuilderPerp);
    }

    #[test]
    fn test_hip3_detection_is_builder_deployed() {
        let meta = make_asset_meta("BUILDER", None, None, Some(true));
        assert!(meta.is_hip3());
        assert!(!meta.allows_cross_margin());
        assert_eq!(meta.asset_type(), AssetType::BuilderPerp);
    }

    #[test]
    fn test_validator_perp_allows_cross() {
        let meta = make_asset_meta("BTC", None, None, None);
        assert!(!meta.is_hip3());
        assert!(meta.allows_cross_margin());
        assert_eq!(meta.asset_type(), AssetType::ValidatorPerp);
    }

    #[test]
    fn test_validator_perp_explicit_false() {
        let meta = make_asset_meta("ETH", Some(false), None, Some(false));
        assert!(!meta.is_hip3());
        assert!(meta.allows_cross_margin());
        assert_eq!(meta.asset_type(), AssetType::ValidatorPerp);
    }

    // ========================================================================
    // CollateralInfo Tests
    // ========================================================================

    #[test]
    fn test_collateral_info_usdc_default() {
        let info = CollateralInfo::usdc();
        assert_eq!(info.token_index, 0);
        assert_eq!(info.symbol, "USDC");
        assert!(info.is_usdc());
        assert_eq!(info.display(), "USDC");
    }

    #[test]
    fn test_collateral_info_default_is_usdc() {
        let info = CollateralInfo::default();
        assert!(info.is_usdc());
        assert_eq!(info.symbol, "USDC");
    }

    fn make_spot_meta_with_tokens() -> SpotMeta {
        SpotMeta {
            universe: vec![],
            tokens: vec![
                TokenInfo {
                    name: "USDC".to_string(),
                    sz_decimals: 8,
                    wei_decimals: 8,
                    index: 0,
                    token_id: B128::ZERO,
                    is_canonical: true,
                },
                TokenInfo {
                    name: "USDE".to_string(),
                    sz_decimals: 2,
                    wei_decimals: 8,
                    index: 235,
                    token_id: B128::ZERO,
                    is_canonical: false,
                },
                TokenInfo {
                    name: "USDH".to_string(),
                    sz_decimals: 6,
                    wei_decimals: 8,
                    index: 360,
                    token_id: B128::ZERO,
                    is_canonical: false,
                },
            ],
        }
    }

    #[test]
    fn test_collateral_info_from_token_index_usdc() {
        let spot_meta = make_spot_meta_with_tokens();
        let info = CollateralInfo::from_token_index(0, &spot_meta);
        assert!(info.is_usdc());
        assert_eq!(info.symbol, "USDC");
    }

    #[test]
    fn test_collateral_info_from_token_index_usde() {
        let spot_meta = make_spot_meta_with_tokens();
        let info = CollateralInfo::from_token_index(235, &spot_meta);
        assert!(!info.is_usdc());
        assert_eq!(info.token_index, 235);
        assert_eq!(info.symbol, "USDE");
        assert_eq!(info.sz_decimals, 2);
    }

    #[test]
    fn test_collateral_info_from_token_index_usdh() {
        let spot_meta = make_spot_meta_with_tokens();
        let info = CollateralInfo::from_token_index(360, &spot_meta);
        assert!(!info.is_usdc());
        assert_eq!(info.token_index, 360);
        assert_eq!(info.symbol, "USDH");
        assert_eq!(info.sz_decimals, 6);
    }

    #[test]
    fn test_collateral_info_unknown_token() {
        let spot_meta = make_spot_meta_with_tokens();
        let info = CollateralInfo::from_token_index(999, &spot_meta);
        assert!(!info.is_usdc());
        assert_eq!(info.token_index, 999);
        assert_eq!(info.symbol, "TOKEN_999");
    }

    #[test]
    fn test_collateral_info_display() {
        let usdc = CollateralInfo::usdc();
        assert_eq!(format!("{}", usdc), "USDC");

        let spot_meta = make_spot_meta_with_tokens();
        let usde = CollateralInfo::from_token_index(235, &spot_meta);
        assert_eq!(format!("{}", usde), "USDE");
    }

    // ========================================================================
    // Meta with collateral_token Tests
    // ========================================================================

    #[test]
    fn test_meta_deserialize_with_collateral_token() {
        let json = r#"{"universe": [], "collateralToken": 235}"#;
        let meta: Meta = serde_json::from_str(json).unwrap();
        assert_eq!(meta.collateral_token, Some(235));
    }

    #[test]
    fn test_meta_deserialize_without_collateral_token() {
        let json = r#"{"universe": []}"#;
        let meta: Meta = serde_json::from_str(json).unwrap();
        assert_eq!(meta.collateral_token, None);
    }

    // ========================================================================
    // CollateralInfo balance extraction tests
    // ========================================================================

    #[test]
    fn test_collateral_balance_from_spot() {
        use crate::types::UserTokenBalance;

        let usdc = CollateralInfo::usdc();
        let balances = vec![
            UserTokenBalance {
                coin: "USDC".to_string(),
                total: "1000.50".to_string(),
                hold: "100.25".to_string(),
                entry_ntl: "0".to_string(),
            },
            UserTokenBalance {
                coin: "ETH".to_string(),
                total: "5.0".to_string(),
                hold: "0.0".to_string(),
                entry_ntl: "0".to_string(),
            },
        ];

        let (total, hold) = usdc.balance_from_spot(&balances).unwrap();
        assert!((total - 1000.50).abs() < 0.001);
        assert!((hold - 100.25).abs() < 0.001);

        let available = usdc.available_balance_from_spot(&balances).unwrap();
        assert!((available - 900.25).abs() < 0.001);
    }

    #[test]
    fn test_collateral_balance_not_found() {
        use crate::types::UserTokenBalance;

        let usde = CollateralInfo {
            token_index: 235,
            symbol: "USDE".to_string(),
            full_name: None,
            sz_decimals: 2,
            wei_decimals: 18,
        };

        let balances = vec![UserTokenBalance {
            coin: "USDC".to_string(),
            total: "1000.0".to_string(),
            hold: "0.0".to_string(),
            entry_ntl: "0".to_string(),
        }];

        assert!(usde.balance_from_spot(&balances).is_none());
        assert!(usde.available_balance_from_spot(&balances).is_none());
    }

    #[test]
    fn test_collateral_format_amount() {
        // USDC has 8 sz_decimals
        let usdc = CollateralInfo::usdc();
        assert_eq!(usdc.format_amount(100.123456789), "100.12345679");

        // USDE with 2 sz_decimals
        let usde = CollateralInfo {
            token_index: 235,
            symbol: "USDE".to_string(),
            full_name: None,
            sz_decimals: 2,
            wei_decimals: 18,
        };
        assert_eq!(usde.format_amount(100.129), "100.13");
    }

    #[test]
    fn test_collateral_transfer_token() {
        let usdc = CollateralInfo::usdc();
        assert_eq!(usdc.transfer_token(), "USDC");

        let usde = CollateralInfo {
            token_index: 235,
            symbol: "USDE".to_string(),
            full_name: None,
            sz_decimals: 2,
            wei_decimals: 18,
        };
        assert_eq!(usde.transfer_token(), "USDE");
    }
}
