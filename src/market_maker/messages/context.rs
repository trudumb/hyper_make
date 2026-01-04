//! Message processing context.
//!
//! Provides a snapshot of market state for message processors.

use std::sync::Arc;

/// Context for message processing.
///
/// Captures the current state needed by processors without
/// requiring access to the full MarketMaker.
#[derive(Debug, Clone)]
pub struct MessageContext {
    /// Current asset being traded.
    /// Uses Arc<str> for cheap cloning from MarketMakerConfig.
    pub asset: Arc<str>,
    /// Latest mid price
    pub latest_mid: f64,
    /// Current position
    pub position: f64,
    /// Max position limit
    pub max_position: f64,
    /// Whether estimator is warmed up
    pub is_warmed_up: bool,
    /// Expected collateral/quote asset symbol (e.g., "USDC", "USDE", "USDH").
    /// Used to validate fee_token in fills matches expected DEX collateral.
    pub expected_collateral: Arc<str>,
}

impl MessageContext {
    /// Create a new message context.
    pub fn new(
        asset: Arc<str>,
        latest_mid: f64,
        position: f64,
        max_position: f64,
        is_warmed_up: bool,
        expected_collateral: Arc<str>,
    ) -> Self {
        Self {
            asset,
            latest_mid,
            position,
            max_position,
            is_warmed_up,
            expected_collateral,
        }
    }

    /// Check if we have valid mid price.
    pub fn has_mid(&self) -> bool {
        self.latest_mid > 0.0
    }

    /// Calculate position utilization.
    pub fn position_utilization(&self) -> f64 {
        if self.max_position > 0.0 {
            (self.position.abs() / self.max_position).min(1.0)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_utilization() {
        let ctx = MessageContext::new(Arc::from("BTC"), 50000.0, 0.5, 1.0, true, Arc::from("USDC"));
        assert!((ctx.position_utilization() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_has_mid() {
        let ctx = MessageContext::new(Arc::from("BTC"), 50000.0, 0.0, 1.0, true, Arc::from("USDC"));
        assert!(ctx.has_mid());

        let ctx_no_mid =
            MessageContext::new(Arc::from("BTC"), -1.0, 0.0, 1.0, true, Arc::from("USDC"));
        assert!(!ctx_no_mid.has_mid());
    }
}
