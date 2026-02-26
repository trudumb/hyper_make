//! Phase 3 stress tests for exchange integration and risk management.
//!
//! Tests verify system behavior under failure conditions:
//! - WebSocket reconnection scenarios (A1)
//! - Order lifecycle state transitions (A2)
//! - Exchange outage / stale data detection (A3)
//! - Kill switch + re-entry integration (B1-B3)
//! - Rate limit interaction with reconnection
//! - Post-mortem dump generation

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use crate::market_maker::execution::{
        CreateOrderParams, OrderEvent, OrderLifecycleTracker, OrderState, Side,
    };
    use crate::market_maker::infra::{
        ConnectionHealthMonitor, ConnectionState, ReconnectionConfig, RejectionRateLimiter,
    };
    use crate::market_maker::monitoring::{AlertConfig, Alerter, PostMortemDump};
    use crate::market_maker::risk::{
        KillReason, KillSwitch, KillSwitchConfig, KillSwitchState, ReentryConfig, ReentryManager,
        ReentryPhase,
    };

    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    fn make_order_params(oid: u64, is_buy: bool) -> CreateOrderParams {
        CreateOrderParams {
            order_id: oid,
            client_order_id: format!("test_{}", oid),
            symbol: "BTC".to_string(),
            side: if is_buy { Side::Bid } else { Side::Ask },
            price: 100_000.0,
            size: 0.01,
            timestamp_ms: now_ms(),
        }
    }

    fn make_event(
        state: OrderState,
        filled: f64,
        remaining: f64,
        reason: Option<&str>,
    ) -> OrderEvent {
        OrderEvent {
            timestamp: now_ms(),
            state,
            filled_size: filled,
            remaining_size: remaining,
            reason: reason.map(|s| s.to_string()),
        }
    }

    // =========================================================================
    // A1: WebSocket Reconnection Stress Tests
    // =========================================================================

    #[test]
    fn test_reconnection_multiple_sequential_failures() {
        let config = ReconnectionConfig {
            max_consecutive_failures: 5,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
            ..ReconnectionConfig::default()
        };
        let monitor = ConnectionHealthMonitor::with_config(config);

        // Simulate 5 sequential failures
        for i in 0..5 {
            monitor.start_reconnection();
            let delay = monitor.next_reconnection_delay();
            assert!(delay.as_millis() > 0, "Attempt {} should have delay", i);
            monitor.reconnection_failed();
        }

        // After max failures, should be in Failed state
        assert_eq!(monitor.state(), ConnectionState::Failed);
    }

    #[test]
    fn test_reconnection_recovery_resets_state() {
        let config = ReconnectionConfig {
            max_consecutive_failures: 10,
            initial_delay: Duration::from_millis(1),
            ..ReconnectionConfig::default()
        };
        let monitor = ConnectionHealthMonitor::with_config(config);

        // Fail a few times
        for _ in 0..3 {
            monitor.start_reconnection();
            monitor.reconnection_failed();
        }

        // Successful reconnect
        monitor.reset();
        monitor.record_data_received();

        assert_eq!(monitor.state(), ConnectionState::Healthy);
    }

    #[test]
    fn test_stale_data_detection_fires() {
        let config = ReconnectionConfig {
            stale_data_threshold: Duration::from_millis(50),
            ..ReconnectionConfig::default()
        };
        let monitor = ConnectionHealthMonitor::with_config(config);

        monitor.record_data_received();
        assert!(!monitor.is_data_stale());

        std::thread::sleep(Duration::from_millis(60));
        assert!(monitor.is_data_stale());
        assert_eq!(monitor.state(), ConnectionState::Stale);
    }

    #[test]
    fn test_reconnection_backoff_increases() {
        let config = ReconnectionConfig {
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(500),
            backoff_multiplier: 2.0,
            jitter_factor: 0.0, // No jitter for deterministic test
            max_consecutive_failures: 10,
            ..ReconnectionConfig::default()
        };
        let monitor = ConnectionHealthMonitor::with_config(config);

        let mut delays = Vec::new();
        for _ in 0..5 {
            monitor.start_reconnection();
            delays.push(monitor.next_reconnection_delay());
            monitor.reconnection_failed();
        }

        // Each delay should be >= previous (exponential backoff)
        for i in 1..delays.len() {
            assert!(
                delays[i] >= delays[i - 1],
                "Delay {} ({:?}) should be >= delay {} ({:?})",
                i,
                delays[i],
                i - 1,
                delays[i - 1]
            );
        }
    }

    // =========================================================================
    // A2: Order Lifecycle State Transitions
    // =========================================================================

    #[test]
    fn test_order_place_fill_inventory_update() {
        let tracker = OrderLifecycleTracker::new(100);

        tracker.create_order(make_order_params(1, true));
        tracker.update_order(1, make_event(OrderState::Filled, 0.01, 0.0, None));

        let completed = tracker.completed_orders(1);
        assert_eq!(completed.len(), 1);
        assert!((completed[0].fill_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_order_place_cancel_no_phantom_position() {
        let tracker = OrderLifecycleTracker::new(100);

        tracker.create_order(make_order_params(2, true));
        tracker.update_order(
            2,
            make_event(OrderState::Cancelled, 0.0, 0.01, Some("user_cancel")),
        );

        let completed = tracker.completed_orders(1);
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].fill_ratio(), 0.0);
        assert!(completed[0].was_cancelled_before_fill());
    }

    #[test]
    fn test_order_partial_fill_cancel_remainder() {
        let tracker = OrderLifecycleTracker::new(100);

        tracker.create_order(make_order_params(3, false));
        tracker.update_order(
            3,
            make_event(OrderState::PartiallyFilled, 0.004, 0.006, None),
        );
        tracker.update_order(
            3,
            make_event(
                OrderState::Cancelled,
                0.004,
                0.006,
                Some("remainder_cancel"),
            ),
        );

        let completed = tracker.completed_orders(1);
        assert_eq!(completed.len(), 1);
        assert!((completed[0].fill_ratio() - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_order_rejection_no_position_change() {
        let tracker = OrderLifecycleTracker::new(100);

        tracker.create_order(make_order_params(4, true));
        tracker.update_order(
            4,
            make_event(OrderState::Rejected, 0.0, 0.01, Some("PerpMaxPosition")),
        );

        let completed = tracker.completed_orders(1);
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].fill_ratio(), 0.0);
    }

    #[test]
    fn test_rapid_order_cycle_no_rate_violation() {
        let tracker = OrderLifecycleTracker::new(1000);

        // Simulate 10 rapid order cycles
        for i in 0..10u64 {
            let oid = 100 + i;
            tracker.create_order(make_order_params(oid, i % 2 == 0));
            tracker.update_order(oid, make_event(OrderState::Cancelled, 0.0, 0.01, None));
        }

        let analysis = tracker.cancel_analysis();
        assert_eq!(analysis.cancel_before_fill_ratio(), 1.0);
        assert_eq!(tracker.active_orders().len(), 0);
        assert_eq!(tracker.completed_orders(100).len(), 10);
    }

    #[test]
    fn test_fill_statistics_accuracy() {
        let tracker = OrderLifecycleTracker::new(100);

        // 3 full fills
        for i in 0..3u64 {
            tracker.create_order(make_order_params(i, true));
            tracker.update_order(i, make_event(OrderState::Filled, 0.01, 0.0, None));
        }
        // 2 partial fills + cancel
        for i in 3..5u64 {
            tracker.create_order(make_order_params(i, false));
            tracker.update_order(
                i,
                make_event(OrderState::PartiallyFilled, 0.005, 0.005, None),
            );
            tracker.update_order(i, make_event(OrderState::Cancelled, 0.005, 0.005, None));
        }
        // 1 pure cancel
        tracker.create_order(make_order_params(5, true));
        tracker.update_order(5, make_event(OrderState::Cancelled, 0.0, 0.01, None));

        let stats = tracker.fill_statistics();
        // 3 full fills out of 6 total = 50%
        assert!((stats.full_fill_rate() - 0.5).abs() < 0.01);
    }

    // =========================================================================
    // A3: Exchange Outage / Kill Switch Integration
    // =========================================================================

    #[test]
    fn test_stale_data_kills_after_threshold() {
        let config = KillSwitchConfig {
            stale_data_threshold: Duration::from_millis(50),
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        // Fresh data — no kill
        let state = KillSwitchState {
            last_data_time: Instant::now(),
            ..Default::default()
        };
        assert!(ks.check(&state).is_none());

        // Stale data — kill
        std::thread::sleep(Duration::from_millis(60));
        let state = KillSwitchState {
            last_data_time: Instant::now() - Duration::from_millis(100),
            ..Default::default()
        };
        let reason = ks.check(&state);
        assert!(reason.is_some());
        match reason.unwrap() {
            KillReason::StaleData { .. } => {}
            other => panic!("Expected StaleData, got {:?}", other),
        }
    }

    #[test]
    fn test_rate_limit_errors_accumulate_and_kill() {
        let config = KillSwitchConfig {
            max_rate_limit_errors: 2,
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        // 2 errors — within limit
        let state = KillSwitchState {
            rate_limit_errors: 2,
            last_data_time: Instant::now(),
            ..Default::default()
        };
        assert!(ks.check(&state).is_none());

        // 3 errors — exceeds limit of 2
        let state = KillSwitchState {
            rate_limit_errors: 3,
            last_data_time: Instant::now(),
            ..Default::default()
        };
        let reason = ks.check(&state);
        assert!(reason.is_some());
        match reason.unwrap() {
            KillReason::RateLimit { count: 3, limit: 2 } => {}
            other => panic!("Expected RateLimit, got {:?}", other),
        }
    }

    // =========================================================================
    // B1-B3: Kill Switch + Re-entry Integration
    // =========================================================================

    #[test]
    fn test_kill_then_reentry_with_reduced_position() {
        let config = ReentryConfig {
            cooling_period: Duration::from_millis(10),
            recovery_position_fraction: 0.5,
            recovery_duration: Duration::from_millis(100),
            ..Default::default()
        };
        let mut reentry = ReentryManager::new(config);

        reentry.on_kill(KillReason::MaxLoss {
            loss: 100.0,
            limit: 50.0,
        });
        assert_eq!(reentry.position_multiplier(), 0.0);
        assert!(!reentry.can_trade());

        // Wait for cooling
        std::thread::sleep(Duration::from_millis(15));
        reentry.tick();

        assert_eq!(reentry.phase(), ReentryPhase::Recovery);
        assert!(reentry.can_trade());
        let mult = reentry.position_multiplier();
        assert!((0.45..=0.65).contains(&mult), "Expected ~0.5, got {}", mult);
    }

    #[test]
    fn test_three_kills_blocks_auto_reentry() {
        let config = ReentryConfig {
            cooling_period: Duration::from_millis(1),
            max_daily_kills: 3,
            rapid_kill_window: Duration::from_millis(1),
            ..Default::default()
        };
        let mut reentry = ReentryManager::new(config);

        for i in 0..3 {
            reentry.force_normal();
            reentry.on_kill(KillReason::MaxLoss {
                loss: (i + 1) as f64 * 30.0,
                limit: 50.0,
            });
        }

        assert_eq!(reentry.phase(), ReentryPhase::ManualReviewRequired);
        assert!(!reentry.can_trade());
        assert_eq!(reentry.daily_kill_count(), 3);
    }

    #[test]
    fn test_reentry_daily_reset_clears_kills() {
        let config = ReentryConfig {
            max_daily_kills: 2,
            ..Default::default()
        };
        let mut reentry = ReentryManager::new(config);

        reentry.on_kill(KillReason::Manual {
            reason: "test".into(),
        });
        reentry.force_normal();
        reentry.on_kill(KillReason::Manual {
            reason: "test2".into(),
        });
        assert_eq!(reentry.phase(), ReentryPhase::ManualReviewRequired);

        reentry.reset_daily();
        assert_eq!(reentry.daily_kill_count(), 0);
        assert_eq!(reentry.phase(), ReentryPhase::Normal);
    }

    #[test]
    fn test_production_kill_thresholds() {
        let config = KillSwitchConfig::production(10_000.0, 1_000.0);
        let ks = KillSwitch::new(config);

        let state = KillSwitchState {
            daily_pnl: -60.0,
            last_data_time: Instant::now(),
            ..Default::default()
        };
        let reason = ks.check(&state);
        assert!(reason.is_some());
        match reason.unwrap() {
            KillReason::MaxLoss {
                loss: 60.0,
                limit: 50.0,
            } => {}
            other => panic!("Expected MaxLoss 60>50, got {:?}", other),
        }
    }

    #[test]
    fn test_production_cascade_threshold() {
        let config = KillSwitchConfig::production(10_000.0, 1_000.0);
        let ks = KillSwitch::new(config);

        // Under threshold
        let state = KillSwitchState {
            cascade_severity: 1.0,
            last_data_time: Instant::now(),
            ..Default::default()
        };
        assert!(ks.check(&state).is_none());

        // Above threshold
        ks.reset();
        let state = KillSwitchState {
            cascade_severity: 2.0,
            last_data_time: Instant::now(),
            ..Default::default()
        };
        assert!(ks.check(&state).is_some());
    }

    // =========================================================================
    // Rate Limit + Rejection Interaction Tests
    // =========================================================================

    #[test]
    fn test_rejection_backoff_per_side() {
        let mut limiter = RejectionRateLimiter::new();

        // Record bid rejections (need error string for record_rejection)
        for _ in 0..4 {
            limiter.record_rejection(true, "PerpMaxPosition exceeded");
        }

        assert!(limiter.should_skip(true), "Bid should be in backoff");
        assert!(!limiter.should_skip(false), "Ask should not be in backoff");
    }

    #[test]
    fn test_rejection_success_resets_backoff() {
        let mut limiter = RejectionRateLimiter::new();

        // Trigger backoff
        for _ in 0..4 {
            limiter.record_rejection(true, "PerpMaxPosition exceeded");
        }
        assert!(limiter.should_skip(true));

        // Success resets
        limiter.record_success(true);
        assert!(!limiter.should_skip(true));
    }

    // =========================================================================
    // D3: Post-Mortem Dump Tests
    // =========================================================================

    #[test]
    fn test_postmortem_dump_creation() {
        let dump = PostMortemDump::new("MaxLoss".to_string(), 150.0, 50.0);
        assert_eq!(dump.trigger, "MaxLoss");
        assert_eq!(dump.trigger_value, 150.0);

        let json = serde_json::to_string_pretty(&dump).unwrap();
        assert!(json.contains("MaxLoss"));
        assert!(json.contains("150"));
    }

    #[test]
    fn test_postmortem_dump_write() {
        let dir = std::path::PathBuf::from("/tmp/claude-test-stress-postmortem");
        let _ = std::fs::remove_dir_all(&dir);

        let mut dump = PostMortemDump::new("MaxDrawdown".to_string(), 0.025, 0.02);
        dump.position = 0.005;
        dump.daily_pnl = -45.0;

        let path = dump.write_to_dir(&dir).unwrap();
        assert!(path.exists());

        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("MaxDrawdown"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // =========================================================================
    // Alert Integration Tests
    // =========================================================================

    #[test]
    fn test_production_alerts_fire_at_thresholds() {
        let alerter = Alerter::new(AlertConfig::default(), 100);
        let ts = now_ms();

        // Drawdown at 1.5% (between warning 1% and critical 2%)
        assert!(alerter.check_drawdown(0.015, ts).is_some());

        // Spread at 25 bps (above 20 bps threshold)
        assert!(alerter.check_spread(25.0, ts).is_some());

        // No fills for 310 seconds (above 300s threshold)
        assert!(alerter.check_no_fills(310, ts).is_some());

        // Signal MI at 0.03 (below 0.05 threshold)
        assert!(alerter.check_signal_health(0.03, "lead_lag", ts).is_some());

        // Inventory at 80% (above 70% threshold)
        assert!(alerter.check_inventory(0.008, 0.01, ts).is_some());
    }

    #[test]
    fn test_alert_deduplication_across_types() {
        let config = AlertConfig {
            dedup_window_s: 60,
            ..Default::default()
        };
        let alerter = Alerter::new(config, 100);
        let ts = now_ms();

        // First spread alert fires
        assert!(alerter.check_spread(25.0, ts).is_some());
        // Second within dedup window — suppressed
        assert!(alerter.check_spread(30.0, ts + 1000).is_none());
        // Different type should still fire
        assert!(alerter.check_no_fills(400, ts + 1000).is_some());
    }
}
