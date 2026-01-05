//! Unique asset identifier for multi-asset market making.
//!
//! `AssetId` provides a compact, hashable identifier for assets that handles
//! both validator perps and HIP-3 builder-deployed assets. The identifier
//! includes the DEX context for HIP-3 assets.
//!
//! # Format
//!
//! - Validator perps: `"validator:BTC"` → AssetId(hash)
//! - HIP-3 assets: `"hyna:HYPE"` → AssetId(hash)
//!
//! # Usage
//!
//! ```ignore
//! let btc = AssetId::new("BTC", None);           // Validator perp
//! let hype = AssetId::new("HYPE", Some("hyna")); // HIP-3 asset
//!
//! // Use in HashMaps for O(1) lookup
//! let mut workers: HashMap<AssetId, AssetWorker> = HashMap::new();
//! workers.insert(btc, worker);
//! ```

use std::fmt;
use std::hash::{Hash, Hasher};

/// Unique identifier for an asset, combining symbol and optional DEX.
///
/// Uses FNV-1a hash for fast, deterministic hashing. The hash is computed
/// from the canonical key format: `"dex:symbol"` or `"validator:symbol"`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct AssetId {
    hash: u64,
}

impl AssetId {
    /// Create a new AssetId from symbol and optional DEX.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let btc = AssetId::new("BTC", None);           // "validator:BTC"
    /// let hype = AssetId::new("HYPE", Some("hyna")); // "hyna:HYPE"
    /// ```
    pub fn new(symbol: &str, dex: Option<&str>) -> Self {
        let key = Self::canonical_key(symbol, dex);
        Self {
            hash: fnv1a_hash(key.as_bytes()),
        }
    }

    /// Create from an effective asset string (e.g., "hyna:HYPE" or "BTC").
    pub fn from_effective(effective: &str) -> Self {
        if let Some((dex, symbol)) = effective.split_once(':') {
            Self::new(symbol, Some(dex))
        } else {
            Self::new(effective, None)
        }
    }

    /// Get the canonical key string.
    ///
    /// Format: `"dex:symbol"` for HIP-3, `"validator:symbol"` for validator perps.
    pub fn canonical_key(symbol: &str, dex: Option<&str>) -> String {
        match dex {
            Some(d) => format!("{}:{}", d, symbol),
            None => format!("validator:{}", symbol),
        }
    }

    /// Get the raw hash value (for debugging).
    pub fn hash_value(&self) -> u64 {
        self.hash
    }
}

impl Hash for AssetId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

impl fmt::Debug for AssetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AssetId({:#018x})", self.hash)
    }
}

impl fmt::Display for AssetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#018x}", self.hash)
    }
}

/// FNV-1a hash implementation for deterministic, fast hashing.
///
/// FNV-1a is used because:
/// - Fast computation (no multiplication)
/// - Good distribution for short strings
/// - Deterministic (same input → same output)
/// - No external dependencies
fn fnv1a_hash(bytes: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET_BASIS;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_id_validator_perp() {
        let btc = AssetId::new("BTC", None);
        let eth = AssetId::new("ETH", None);

        // Different assets have different IDs
        assert_ne!(btc, eth);

        // Same asset has same ID
        let btc2 = AssetId::new("BTC", None);
        assert_eq!(btc, btc2);
    }

    #[test]
    fn test_asset_id_hip3() {
        let hype_hyna = AssetId::new("HYPE", Some("hyna"));
        let hype_felix = AssetId::new("HYPE", Some("felix"));

        // Same symbol on different DEXs have different IDs
        assert_ne!(hype_hyna, hype_felix);

        // Same symbol+DEX has same ID
        let hype_hyna2 = AssetId::new("HYPE", Some("hyna"));
        assert_eq!(hype_hyna, hype_hyna2);
    }

    #[test]
    fn test_asset_id_validator_vs_hip3() {
        // BTC on validator perps vs BTC on HIP-3 DEX
        let btc_validator = AssetId::new("BTC", None);
        let btc_hip3 = AssetId::new("BTC", Some("hyna"));

        // These should be different
        assert_ne!(btc_validator, btc_hip3);
    }

    #[test]
    fn test_from_effective() {
        let hype = AssetId::from_effective("hyna:HYPE");
        let expected = AssetId::new("HYPE", Some("hyna"));
        assert_eq!(hype, expected);

        let btc = AssetId::from_effective("BTC");
        let expected_btc = AssetId::new("BTC", None);
        assert_eq!(btc, expected_btc);
    }

    #[test]
    fn test_canonical_key() {
        assert_eq!(AssetId::canonical_key("BTC", None), "validator:BTC");
        assert_eq!(AssetId::canonical_key("HYPE", Some("hyna")), "hyna:HYPE");
    }

    #[test]
    fn test_hashmap_usage() {
        use std::collections::HashMap;

        let btc = AssetId::new("BTC", None);
        let eth = AssetId::new("ETH", None);
        let hype = AssetId::new("HYPE", Some("hyna"));

        let mut map: HashMap<AssetId, &str> = HashMap::new();
        map.insert(btc, "Bitcoin");
        map.insert(eth, "Ethereum");
        map.insert(hype, "Hype on Hyena");

        assert_eq!(map.get(&btc), Some(&"Bitcoin"));
        assert_eq!(map.get(&eth), Some(&"Ethereum"));
        assert_eq!(map.get(&hype), Some(&"Hype on Hyena"));
    }
}
