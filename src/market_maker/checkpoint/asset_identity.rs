//! Canonical asset identity for cross-mode checkpoint matching.
//!
//! Paper saves checkpoints as "HYPE", live with `--dex hyna` resolves to "hyna:HYPE".
//! These are the same market. This module provides canonical identity functions
//! so that prior discovery and injection work across modes.
//!
//! # Examples
//! ```text
//! base_symbol("hyna:HYPE")   → "HYPE"
//! base_symbol("HYPE")        → "HYPE"
//! base_symbol("hl:BTC")      → "BTC"
//! assets_match("HYPE", "hyna:HYPE") → true
//! ```

use std::path::{Path, PathBuf};

/// Strip DEX prefix from an asset identifier, returning the base symbol.
///
/// `"hyna:HYPE" → "HYPE"`, `"HYPE" → "HYPE"`, `"hl:BTC" → "BTC"`.
pub fn base_symbol(asset: &str) -> &str {
    match asset.find(':') {
        Some(idx) => &asset[idx + 1..],
        None => asset,
    }
}

/// Compare two asset identifiers by their base symbol.
///
/// Returns true if both refer to the same underlying market,
/// regardless of DEX prefix.
pub fn assets_match(a: &str, b: &str) -> bool {
    base_symbol(a) == base_symbol(b)
}

/// Build the canonical checkpoint directory path using base symbol.
///
/// Always uses the base symbol (no DEX prefix) for consistent paths.
/// `checkpoint_dir("data/checkpoints/paper", "hyna:HYPE")` →
/// `"data/checkpoints/paper/HYPE"`.
pub fn checkpoint_dir(root: &str, asset: &str) -> PathBuf {
    Path::new(root).join(base_symbol(asset))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_symbol_strips_dex_prefix() {
        assert_eq!(base_symbol("hyna:HYPE"), "HYPE");
        assert_eq!(base_symbol("hl:BTC"), "BTC");
        assert_eq!(base_symbol("dex:ETH"), "ETH");
    }

    #[test]
    fn test_base_symbol_no_prefix() {
        assert_eq!(base_symbol("HYPE"), "HYPE");
        assert_eq!(base_symbol("BTC"), "BTC");
        assert_eq!(base_symbol(""), "");
    }

    #[test]
    fn test_base_symbol_multiple_colons() {
        // Edge case: multiple colons — strip only the first prefix
        assert_eq!(base_symbol("a:b:c"), "b:c");
    }

    #[test]
    fn test_assets_match_cross_dex() {
        assert!(assets_match("HYPE", "hyna:HYPE"));
        assert!(assets_match("hyna:HYPE", "HYPE"));
        assert!(assets_match("hl:BTC", "BTC"));
        assert!(assets_match("dex1:ETH", "dex2:ETH"));
    }

    #[test]
    fn test_assets_match_same() {
        assert!(assets_match("HYPE", "HYPE"));
        assert!(assets_match("hyna:HYPE", "hyna:HYPE"));
    }

    #[test]
    fn test_assets_match_different() {
        assert!(!assets_match("HYPE", "BTC"));
        assert!(!assets_match("hyna:HYPE", "hyna:BTC"));
        assert!(!assets_match("HYPE", "hyna:BTC"));
    }

    #[test]
    fn test_checkpoint_dir_canonical() {
        let dir = checkpoint_dir("data/checkpoints/paper", "hyna:HYPE");
        assert_eq!(dir, PathBuf::from("data/checkpoints/paper/HYPE"));

        let dir2 = checkpoint_dir("data/checkpoints/paper", "HYPE");
        assert_eq!(dir2, PathBuf::from("data/checkpoints/paper/HYPE"));

        let dir3 = checkpoint_dir("/abs/path", "hl:BTC");
        assert_eq!(dir3, PathBuf::from("/abs/path/BTC"));
    }
}
