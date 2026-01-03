//! Trade and fill types.

use serde::{Deserialize, Serialize};

/// Public trade from the order book.
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Trade {
    pub coin: String,
    pub side: String,
    pub px: String,
    pub sz: String,
    pub time: u64,
    pub hash: String,
    pub tid: u64,
    pub users: (String, String),
}

/// User fill/trade with extended information.
///
/// # HIP-3 Fee Handling
///
/// For HIP-3 builder-deployed assets, the `fee` field contains the **total** fee
/// (protocol fee + builder fee). The `builder_fee` field, when present, contains
/// the builder's portion. For validator-operated perps, `builder_fee` is None.
///
/// Fee breakdown:
/// - `total_fee()` = fee (always present)
/// - `builder_fee_amount()` = builder_fee (0 if not present)
/// - `protocol_fee()` = fee - builder_fee
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct TradeInfo {
    pub coin: String,
    pub side: String,
    pub px: String,
    pub sz: String,
    pub time: u64,
    pub hash: String,
    pub start_position: String,
    pub dir: String,
    pub closed_pnl: String,
    pub oid: u64,
    pub cloid: Option<String>,
    pub crossed: bool,
    /// Total fee (includes builder fee for HIP-3 assets).
    pub fee: String,
    pub fee_token: String,
    pub tid: u64,

    // === HIP-3 Fields ===
    /// Builder fee component (optional, only present if non-zero).
    /// This is the portion of `fee` that goes to the HIP-3 deployer.
    #[serde(default)]
    pub builder_fee: Option<String>,
}

impl TradeInfo {
    /// Get total fee as f64.
    ///
    /// This is the complete fee amount, including any builder fee.
    #[inline]
    pub fn total_fee(&self) -> f64 {
        self.fee.parse().unwrap_or(0.0)
    }

    /// Get builder fee component as f64 (0.0 if not present).
    ///
    /// For HIP-3 assets, this is the fee portion going to the deployer.
    /// For validator perps, this returns 0.0.
    #[inline]
    pub fn builder_fee_amount(&self) -> f64 {
        self.builder_fee
            .as_ref()
            .and_then(|f| f.parse().ok())
            .unwrap_or(0.0)
    }

    /// Get protocol fee component (total - builder).
    ///
    /// This is the fee portion going to Hyperliquid validators/stakers.
    #[inline]
    pub fn protocol_fee(&self) -> f64 {
        self.total_fee() - self.builder_fee_amount()
    }

    /// Check if this is a HIP-3 fill (has builder fee).
    #[inline]
    pub fn is_hip3_fill(&self) -> bool {
        self.builder_fee.is_some()
    }
}

/// Liquidation event.
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Liquidation {
    pub lid: u64,
    pub liquidator: String,
    pub liquidated_user: String,
    pub liquidated_ntl_pos: String,
    pub liquidated_account_value: String,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_validator_fill() -> TradeInfo {
        TradeInfo {
            coin: "BTC".to_string(),
            side: "B".to_string(),
            px: "50000.0".to_string(),
            sz: "0.1".to_string(),
            time: 1000,
            hash: "hash".to_string(),
            start_position: "0".to_string(),
            dir: "L".to_string(),
            closed_pnl: "0".to_string(),
            oid: 1,
            cloid: None,
            crossed: false,
            fee: "0.50".to_string(), // Total fee
            fee_token: "USDC".to_string(),
            tid: 1,
            builder_fee: None, // No builder fee for validator perps
        }
    }

    fn make_hip3_fill() -> TradeInfo {
        TradeInfo {
            coin: "MEMECOIN".to_string(),
            side: "S".to_string(),
            px: "0.001".to_string(),
            sz: "10000".to_string(),
            time: 2000,
            hash: "hash2".to_string(),
            start_position: "10000".to_string(),
            dir: "S".to_string(),
            closed_pnl: "5.0".to_string(),
            oid: 2,
            cloid: Some("cloid123".to_string()),
            crossed: true,
            fee: "0.10".to_string(), // Total fee (protocol + builder)
            fee_token: "USDC".to_string(),
            tid: 2,
            builder_fee: Some("0.03".to_string()), // Builder gets 30% of fee
        }
    }

    #[test]
    fn test_validator_perp_fee_calculation() {
        let fill = make_validator_fill();

        assert!(!fill.is_hip3_fill());
        assert!((fill.total_fee() - 0.50).abs() < 1e-10);
        assert_eq!(fill.builder_fee_amount(), 0.0);
        assert!((fill.protocol_fee() - 0.50).abs() < 1e-10);
    }

    #[test]
    fn test_hip3_fee_calculation() {
        let fill = make_hip3_fill();

        assert!(fill.is_hip3_fill());
        assert!((fill.total_fee() - 0.10).abs() < 1e-10);
        assert!((fill.builder_fee_amount() - 0.03).abs() < 1e-10);
        assert!((fill.protocol_fee() - 0.07).abs() < 1e-10); // 0.10 - 0.03 = 0.07
    }

    #[test]
    fn test_fee_with_invalid_parse() {
        let mut fill = make_validator_fill();
        fill.fee = "invalid".to_string();

        // Should return 0.0 for unparseable fees
        assert_eq!(fill.total_fee(), 0.0);
        assert_eq!(fill.protocol_fee(), 0.0);
    }

    #[test]
    fn test_builder_fee_with_invalid_parse() {
        let mut fill = make_hip3_fill();
        fill.builder_fee = Some("invalid".to_string());

        // total_fee should still work
        assert!((fill.total_fee() - 0.10).abs() < 1e-10);
        // builder_fee should return 0.0 for invalid
        assert_eq!(fill.builder_fee_amount(), 0.0);
        // protocol_fee = total - 0 = total
        assert!((fill.protocol_fee() - 0.10).abs() < 1e-10);
    }
}
