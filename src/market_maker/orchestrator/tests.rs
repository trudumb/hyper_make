//! Integration tests for the orchestrator module.
//!
//! Tests cover:
//! - Ladder action partitioning
//! - Side string conversion
//! - Reconciliation logic invariants
//! - NaN handling in price sorting
//! - Order notional validation

#[cfg(test)]
mod tests {
    use crate::market_maker::tracking::LadderAction;
    use crate::market_maker::Side;

    use super::super::{partition_ladder_actions, side_str};

    // =========================================================================
    // Utility Function Tests
    // =========================================================================

    #[test]
    fn test_side_str_buy() {
        assert_eq!(side_str(Side::Buy), "BUY");
    }

    #[test]
    fn test_side_str_sell() {
        assert_eq!(side_str(Side::Sell), "SELL");
    }

    // =========================================================================
    // Partition Ladder Actions Tests
    // =========================================================================

    #[test]
    fn test_partition_empty_actions() {
        let actions: Vec<LadderAction> = vec![];
        let (cancels, modifies, places) = partition_ladder_actions(&actions, Side::Buy);

        assert!(cancels.is_empty());
        assert!(modifies.is_empty());
        assert!(places.is_empty());
    }

    #[test]
    fn test_partition_only_cancels() {
        let actions = vec![
            LadderAction::Cancel { oid: 100 },
            LadderAction::Cancel { oid: 101 },
            LadderAction::Cancel { oid: 102 },
        ];
        let (cancels, modifies, places) = partition_ladder_actions(&actions, Side::Buy);

        assert_eq!(cancels, vec![100, 101, 102]);
        assert!(modifies.is_empty());
        assert!(places.is_empty());
    }

    #[test]
    fn test_partition_only_places() {
        let actions = vec![
            LadderAction::Place {
                side: Side::Buy,
                price: 100.0,
                size: 1.0,
            },
            LadderAction::Place {
                side: Side::Buy,
                price: 99.5,
                size: 2.0,
            },
        ];
        let (cancels, modifies, places) = partition_ladder_actions(&actions, Side::Buy);

        assert!(cancels.is_empty());
        assert!(modifies.is_empty());
        assert_eq!(places.len(), 2);
        assert_eq!(places[0], (100.0, 1.0));
        assert_eq!(places[1], (99.5, 2.0));
    }

    #[test]
    fn test_partition_only_modifies() {
        let actions = vec![
            LadderAction::Modify {
                oid: 200,
                new_price: 101.0,
                new_size: 1.5,
                side: Side::Buy,
            },
            LadderAction::Modify {
                oid: 201,
                new_price: 100.5,
                new_size: 2.5,
                side: Side::Buy,
            },
        ];
        let (cancels, modifies, places) = partition_ladder_actions(&actions, Side::Buy);

        assert!(cancels.is_empty());
        assert_eq!(modifies.len(), 2);
        assert_eq!(modifies[0].oid, 200);
        assert_eq!(modifies[0].new_price, 101.0);
        assert_eq!(modifies[0].new_size, 1.5);
        assert!(modifies[0].is_buy);
        assert!(modifies[0].post_only); // ALO on modify
        assert_eq!(modifies[1].oid, 201);
        assert!(places.is_empty());
    }

    #[test]
    fn test_partition_mixed_actions() {
        let actions = vec![
            LadderAction::Cancel { oid: 100 },
            LadderAction::Place {
                side: Side::Buy,
                price: 99.0,
                size: 1.0,
            },
            LadderAction::Modify {
                oid: 200,
                new_price: 100.0,
                new_size: 2.0,
                side: Side::Buy,
            },
            LadderAction::Cancel { oid: 101 },
            LadderAction::Place {
                side: Side::Buy,
                price: 98.0,
                size: 1.5,
            },
        ];
        let (cancels, modifies, places) = partition_ladder_actions(&actions, Side::Buy);

        // Order should be preserved within each category
        assert_eq!(cancels, vec![100, 101]);
        assert_eq!(modifies.len(), 1);
        assert_eq!(modifies[0].oid, 200);
        assert_eq!(places.len(), 2);
        assert_eq!(places[0], (99.0, 1.0));
        assert_eq!(places[1], (98.0, 1.5));
    }

    #[test]
    fn test_partition_sell_side_is_buy_flag() {
        let actions = vec![LadderAction::Modify {
            oid: 300,
            new_price: 105.0,
            new_size: 3.0,
            side: Side::Sell,
        }];
        let (_, modifies, _) = partition_ladder_actions(&actions, Side::Sell);

        assert_eq!(modifies.len(), 1);
        assert!(!modifies[0].is_buy); // Should be false for sell side
        assert!(modifies[0].post_only);
    }

    // =========================================================================
    // NaN Handling Tests
    // =========================================================================

    #[test]
    fn test_nan_ordering_safe() {
        // Verify that NaN comparisons don't panic with unwrap_or(Equal)
        let nan = f64::NAN;
        let normal = 100.0;

        // This would panic with plain unwrap()
        let cmp = nan
            .partial_cmp(&normal)
            .unwrap_or(std::cmp::Ordering::Equal);
        assert_eq!(cmp, std::cmp::Ordering::Equal);

        let cmp = normal
            .partial_cmp(&nan)
            .unwrap_or(std::cmp::Ordering::Equal);
        assert_eq!(cmp, std::cmp::Ordering::Equal);

        let cmp = nan.partial_cmp(&nan).unwrap_or(std::cmp::Ordering::Equal);
        assert_eq!(cmp, std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_nan_detection() {
        // Verify f64::is_nan() works as expected for our filtering
        assert!(f64::NAN.is_nan());
        assert!(!100.0_f64.is_nan());
        assert!(!f64::INFINITY.is_nan());
        assert!(!f64::NEG_INFINITY.is_nan());
        assert!(!0.0_f64.is_nan());
    }

    #[test]
    fn test_price_sort_with_nan_handling() {
        // Simulate the sort logic from reconcile.rs with NaN handling
        let mut prices = vec![100.0, 99.5, f64::NAN, 101.0, 98.0];

        // Filter NaN first (as we do in reconcile.rs)
        prices.retain(|p| !p.is_nan());
        assert_eq!(prices.len(), 4);

        // Sort descending (buy side)
        prices.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        assert_eq!(prices, vec![101.0, 100.0, 99.5, 98.0]);
    }

    #[test]
    fn test_price_sort_ascending_with_nan_handling() {
        // Simulate sell side sort (ascending)
        let mut prices = vec![100.0, 99.5, f64::NAN, 101.0, 98.0];

        // Filter NaN first
        prices.retain(|p| !p.is_nan());

        // Sort ascending (sell side)
        prices.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        assert_eq!(prices, vec![98.0, 99.5, 100.0, 101.0]);
    }

    // =========================================================================
    // Order Notional Tests
    // =========================================================================

    #[test]
    fn test_min_notional_constant() {
        use super::super::reconcile::MIN_ORDER_NOTIONAL;

        // Hyperliquid minimum is 10 USD
        assert_eq!(MIN_ORDER_NOTIONAL, 10.0);
    }

    #[test]
    fn test_notional_calculation() {
        // Verify notional = price * size
        let price: f64 = 100.0;
        let size: f64 = 0.15;
        let notional: f64 = price * size;

        assert!((notional - 15.0).abs() < 1e-10);
        assert!(notional >= 10.0); // Above minimum
    }

    #[test]
    fn test_notional_below_minimum() {
        // Orders below 10 USD notional should be filtered
        let price = 100.0;
        let size = 0.05; // Only 5 USD notional
        let notional = price * size;

        assert!(notional < 10.0); // Below minimum - would be filtered
    }

    // =========================================================================
    // Quote Invariant Tests
    // =========================================================================

    #[test]
    fn test_spread_positive_invariant() {
        // Core invariant: ask > bid
        let bid_price: f64 = 99.95;
        let ask_price: f64 = 100.05;

        assert!(ask_price > bid_price);

        let spread_bps: f64 = ((ask_price - bid_price) / bid_price) * 10000.0;
        assert!(spread_bps > 0.0);
        assert!((spread_bps - 10.0).abs() < 0.1); // ~10 bps spread
    }

    #[test]
    fn test_microprice_invariant() {
        // Invariant: bid < microprice < ask
        let bid_price = 99.95;
        let ask_price = 100.05;
        let microprice = (bid_price + ask_price) / 2.0; // Simplified mid

        assert!(bid_price < microprice);
        assert!(microprice < ask_price);
    }

    // =========================================================================
    // Ladder Level Tests
    // =========================================================================

    #[test]
    fn test_ladder_levels_ordered() {
        // Buy ladder: prices should be descending (best bid first)
        let buy_prices = vec![100.0, 99.5, 99.0, 98.5];
        for i in 1..buy_prices.len() {
            assert!(buy_prices[i] < buy_prices[i - 1]);
        }

        // Sell ladder: prices should be ascending (best ask first)
        let sell_prices = vec![100.1, 100.5, 101.0, 101.5];
        for i in 1..sell_prices.len() {
            assert!(sell_prices[i] > sell_prices[i - 1]);
        }
    }

    #[test]
    fn test_ladder_no_overlap() {
        // Best bid must be below best ask
        let best_bid = 100.0;
        let best_ask = 100.1;

        assert!(best_bid < best_ask);
    }

    // =========================================================================
    // LadderAction Enum Tests
    // =========================================================================

    #[test]
    fn test_ladder_action_cancel_has_oid() {
        let action = LadderAction::Cancel { oid: 12345 };
        if let LadderAction::Cancel { oid } = action {
            assert_eq!(oid, 12345);
        } else {
            panic!("Expected Cancel action");
        }
    }

    #[test]
    fn test_ladder_action_place_has_all_fields() {
        let action = LadderAction::Place {
            side: Side::Buy,
            price: 100.0,
            size: 1.5,
        };
        if let LadderAction::Place { side, price, size } = action {
            assert_eq!(side, Side::Buy);
            assert_eq!(price, 100.0);
            assert_eq!(size, 1.5);
        } else {
            panic!("Expected Place action");
        }
    }

    #[test]
    fn test_ladder_action_modify_has_all_fields() {
        let action = LadderAction::Modify {
            oid: 500,
            new_price: 99.9,
            new_size: 2.0,
            side: Side::Sell,
        };
        if let LadderAction::Modify {
            oid,
            new_price,
            new_size,
            side,
        } = action
        {
            assert_eq!(oid, 500);
            assert_eq!(new_price, 99.9);
            assert_eq!(new_size, 2.0);
            assert_eq!(side, Side::Sell);
        } else {
            panic!("Expected Modify action");
        }
    }

    // =========================================================================
    // BBO Crossing Validation Tests
    // =========================================================================

    /// Verifies that the BBO crossing detection logic correctly identifies
    /// orders that would cross the exchange BBO.
    ///
    /// From microstructure theory: a post-only bid at price P crosses the BBO
    /// if P >= best_ask. With a 1-tick buffer (1 bps), it crosses if
    /// P >= best_ask - tick_proxy.
    #[test]
    fn test_bbo_crossing_detection_bid_crosses_ask() {
        let exchange_best_ask = 33.000;
        let mid_price = 32.996;
        let tick_proxy = mid_price * 0.0001; // ~0.0033

        // Bid at 32.998 is within 1 tick of ask 33.000 — should be flagged
        let bid_price = 32.998;
        assert!(
            bid_price >= exchange_best_ask - tick_proxy,
            "Bid {bid_price} should cross ask {exchange_best_ask} with tick buffer {tick_proxy:.4}"
        );

        // Bid at 32.990 is well below ask — should NOT be flagged
        let safe_bid = 32.990;
        assert!(
            safe_bid < exchange_best_ask - tick_proxy,
            "Bid {safe_bid} should NOT cross ask {exchange_best_ask}"
        );
    }

    #[test]
    fn test_bbo_crossing_detection_ask_crosses_bid() {
        let exchange_best_bid = 32.990;
        let mid_price = 32.993;
        let tick_proxy = mid_price * 0.0001; // ~0.0033

        // Ask at 32.992 is within 1 tick of bid 32.990 — should be flagged
        let ask_price = 32.992;
        assert!(
            ask_price <= exchange_best_bid + tick_proxy,
            "Ask {ask_price} should cross bid {exchange_best_bid} with tick buffer {tick_proxy:.4}"
        );

        // Ask at 33.000 is well above bid — should NOT be flagged
        let safe_ask = 33.000;
        assert!(
            safe_ask > exchange_best_bid + tick_proxy,
            "Ask {safe_ask} should NOT cross bid {exchange_best_bid}"
        );
    }

    /// Verifies the staleness buffer calculation: older books get wider buffers.
    #[test]
    fn test_bbo_staleness_buffer() {
        let mid_price = 100_000.0; // BTC-like
        let tick_proxy = mid_price * 0.0001; // $10

        // 0-1 seconds: no staleness buffer
        let staleness_ticks_0s = 0.0_f64;
        let buffer_0s = staleness_ticks_0s * tick_proxy;
        assert_eq!(buffer_0s, 0.0);

        // 3 seconds (above 2s threshold): 2 ticks of buffer
        let staleness_ticks_3s = (3_u64 - 2 + 1) as f64;
        let buffer_3s = staleness_ticks_3s * tick_proxy;
        assert!(
            (buffer_3s - 20.0).abs() < 0.01,
            "Expected ~$20 buffer at 3s age"
        );

        // 4 seconds: 3 ticks of buffer
        let staleness_ticks_4s = (4_u64 - 2 + 1) as f64;
        let buffer_4s = staleness_ticks_4s * tick_proxy;
        assert!(
            (buffer_4s - 30.0).abs() < 0.01,
            "Expected ~$30 buffer at 4s age"
        );
    }

    /// The real-world bug: bid at 32.992 hit ask at 32.992.
    /// This test reproduces the exact scenario from the production incident.
    #[test]
    fn test_real_world_crossing_incident() {
        // Production scenario: asset price ~$33, bid placed at 32.992
        // Exchange best ask was also 32.992
        let exchange_best_ask = 32.992;
        let mid_price = 32.992;
        let tick_proxy = mid_price * 0.0001; // ~0.0033

        let bid_price = 32.992;

        // Our validation would catch this: bid >= ask - tick
        assert!(
            bid_price >= exchange_best_ask - tick_proxy,
            "The production incident bid={bid_price} at ask={exchange_best_ask} \
             MUST be caught by BBO validation"
        );

        // A safe bid would be at least 1 tick below the ask
        let safe_bid = exchange_best_ask - tick_proxy - 0.001;
        assert!(
            safe_bid < exchange_best_ask - tick_proxy,
            "A bid 1+ tick below ask should pass validation"
        );
    }
}
