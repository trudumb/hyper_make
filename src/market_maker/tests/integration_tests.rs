//! Integration tests for the full market maker component pipeline.
//!
//! These tests verify that components work together correctly:
//! - Circuit breaker cascade detection
//! - HMM regime transitions on volatility spikes
//! - Ensemble model degradation detection
//! - Drawdown triggers and pause conditions
//! - Risk checker position limits
//! - Alert generation on calibration degradation
//! - Full component pipeline orchestration

#[cfg(test)]
mod tests {
    // Use public re-exports from the parent modules
    use crate::market_maker::risk::{
        CircuitBreakerConfig, CircuitBreakerMonitor, CircuitBreakerType,
        DrawdownConfig, DrawdownLevel, DrawdownTracker,
        RiskCheckResult, RiskChecker, RiskLimits,
    };
    use crate::market_maker::monitoring::{AlertConfig, AlertSeverity, AlertType, Alerter};
    use crate::market_maker::estimator::{HmmObservation, RegimeHMM};
    use crate::market_maker::learning::AdaptiveEnsemble;
    use crate::market_maker::edge::{EdgeSignalKind, SignalHealthMonitor};

    // =========================================================================
    // Circuit Breaker Tests
    // =========================================================================

    #[test]
    fn test_circuit_breaker_cascade_detection() {
        // Simulate OI cascade and verify circuit breaker triggers
        let config = CircuitBreakerConfig::default();
        let mut breaker = CircuitBreakerMonitor::new(config);

        let base_time = 1_000_000u64;

        // Feed normal OI - no cascade
        for i in 0..10 {
            breaker.update_oi(base_time + i * 1000, 1000.0);
        }
        assert!(breaker.check_oi_cascade(base_time + 10_000).is_none());

        // Simulate cascade (5% OI drop, exceeds default 2% threshold)
        breaker.update_oi(base_time + 20_000, 950.0);
        let action = breaker.check_oi_cascade(base_time + 20_000);
        assert!(action.is_some());
        assert_eq!(action.unwrap(), CircuitBreakerType::OIDropCascade);
    }

    #[test]
    fn test_circuit_breaker_funding_extreme() {
        let config = CircuitBreakerConfig::default();
        let breaker = CircuitBreakerMonitor::new(config);

        // Normal funding - no trigger
        assert!(breaker.check_funding(0.0005).is_none());

        // Extreme funding (>0.1% per 8h) - trigger
        assert_eq!(
            breaker.check_funding(0.002),
            Some(CircuitBreakerType::FundingExtreme)
        );
    }

    #[test]
    fn test_circuit_breaker_spread_blowout() {
        let config = CircuitBreakerConfig::default();
        let breaker = CircuitBreakerMonitor::new(config);

        // Normal spread - no trigger
        assert!(breaker.check_spread(30.0).is_none());

        // Spread blowout (>50 bps default) - trigger
        assert_eq!(
            breaker.check_spread(60.0),
            Some(CircuitBreakerType::SpreadBlowout)
        );
    }

    #[test]
    fn test_circuit_breaker_most_severe_action() {
        let config = CircuitBreakerConfig::default();
        let mut breaker = CircuitBreakerMonitor::new(config);

        // No triggers - no action
        assert!(breaker.most_severe_action().is_none());

        // Trigger funding (WidenSpreads)
        breaker.trigger(CircuitBreakerType::FundingExtreme);

        // Trigger model degradation (PauseTrading - more severe)
        breaker.trigger(CircuitBreakerType::ModelDegradation);

        // Most severe should be PauseTrading
        let action = breaker.most_severe_action();
        assert!(action.is_some());
    }

    // =========================================================================
    // HMM Regime Detection Tests
    // =========================================================================

    #[test]
    fn test_regime_hmm_volatility_spike() {
        // Verify HMM transitions to high regime on volatility spike
        let mut hmm = RegimeHMM::new();

        // Normal observations to establish baseline
        for _ in 0..20 {
            hmm.forward_update(&HmmObservation {
                volatility: 0.002,
                spread_bps: 5.0,
                flow_imbalance: 0.0,
            });
        }

        let belief_before = hmm.regime_probabilities();
        let high_extreme_before = belief_before[2] + belief_before[3]; // High + Extreme

        // Volatility spike
        for _ in 0..10 {
            hmm.forward_update(&HmmObservation {
                volatility: 0.05,
                spread_bps: 25.0,
                flow_imbalance: 0.5,
            });
        }

        let belief_after = hmm.regime_probabilities();
        let high_extreme_after = belief_after[2] + belief_after[3];

        // High+Extreme probability should increase
        assert!(
            high_extreme_after > high_extreme_before,
            "High+Extreme prob should increase after vol spike: before={}, after={}",
            high_extreme_before,
            high_extreme_after
        );
    }

    #[test]
    fn test_regime_hmm_low_volatility() {
        let mut hmm = RegimeHMM::new();

        // Feed very low volatility observations
        for _ in 0..30 {
            hmm.forward_update(&HmmObservation {
                volatility: 0.0005,
                spread_bps: 2.0,
                flow_imbalance: 0.0,
            });
        }

        let belief = hmm.regime_probabilities();
        // Low regime should be elevated relative to Extreme
        assert!(
            belief[0] > belief[3],
            "Low prob {} should exceed Extreme prob {} for quiet market",
            belief[0],
            belief[3]
        );
    }

    #[test]
    fn test_regime_hmm_belief_sums_to_one() {
        let mut hmm = RegimeHMM::new();

        for _ in 0..10 {
            hmm.forward_update(&HmmObservation {
                volatility: 0.003,
                spread_bps: 7.0,
                flow_imbalance: 0.1,
            });
        }

        let belief = hmm.regime_probabilities();
        let sum: f64 = belief.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "Belief probabilities should sum to 1.0, got {}",
            sum
        );
    }

    // =========================================================================
    // Ensemble Model Tests
    // =========================================================================

    #[test]
    fn test_ensemble_degrades_poor_model() {
        // Verify ensemble reduces weight of poorly performing model
        let mut ensemble = AdaptiveEnsemble::new(1.0);
        ensemble.register_model("good_model");
        ensemble.register_model("bad_model");

        // Update with different IRs (need enough predictions for weighting)
        let timestamp = 1000u64;
        for _ in 0..25 {
            ensemble.update_performance("good_model", 1.5, 0.2, timestamp);
            ensemble.update_performance("bad_model", 0.8, 0.4, timestamp);
        }

        // Good model should have higher weight
        assert!(
            ensemble.get_weight("good_model") > ensemble.get_weight("bad_model"),
            "Good model weight {} should exceed bad model weight {}",
            ensemble.get_weight("good_model"),
            ensemble.get_weight("bad_model")
        );

        // Bad model should be marked as degraded (IR < 1.0)
        assert!(
            ensemble.is_model_degraded("bad_model"),
            "Bad model with IR 0.8 should be degraded"
        );
    }

    #[test]
    fn test_ensemble_weights_sum_to_one() {
        let mut ensemble = AdaptiveEnsemble::new(1.0);
        ensemble.register_model("model_a");
        ensemble.register_model("model_b");
        ensemble.register_model("model_c");

        for _ in 0..25 {
            ensemble.update_performance("model_a", 2.0, 0.1, 1000);
            ensemble.update_performance("model_b", 1.0, 0.25, 1000);
            ensemble.update_performance("model_c", 0.5, 0.4, 1000);
        }

        let total: f64 = ["model_a", "model_b", "model_c"]
            .iter()
            .map(|n| ensemble.get_weight(n))
            .sum();

        assert!(
            (total - 1.0).abs() < 1e-9,
            "Weights should sum to 1.0, got {}",
            total
        );
    }

    #[test]
    fn test_ensemble_minimum_weight_floor() {
        let mut ensemble = AdaptiveEnsemble::new(0.01); // Very low temperature
        ensemble.register_model("best");
        ensemble.register_model("worst");

        for _ in 0..25 {
            ensemble.update_performance("best", 10.0, 0.01, 1000);
            ensemble.update_performance("worst", 0.1, 0.9, 1000);
        }

        // Even the worst model should have at least min_weight
        assert!(
            ensemble.get_weight("worst") >= ensemble.min_weight(),
            "Worst model weight {} should be >= min_weight {}",
            ensemble.get_weight("worst"),
            ensemble.min_weight()
        );
    }

    // =========================================================================
    // Drawdown Tracker Tests
    // =========================================================================

    #[test]
    fn test_drawdown_triggers_pause() {
        let config = DrawdownConfig::default();
        let mut tracker = DrawdownTracker::new(config, 100_000.0);

        // Initial state - no drawdown
        assert_eq!(tracker.level(), DrawdownLevel::Normal);
        assert!(!tracker.should_pause());

        // Simulate losses to emergency level (>3% default)
        tracker.update_equity(96_000.0); // 4% drawdown
        assert_eq!(tracker.level(), DrawdownLevel::Emergency);
        assert!(tracker.should_pause());
    }

    #[test]
    fn test_drawdown_position_multiplier() {
        let config = DrawdownConfig::default()
            .with_position_reduction(0.5, 0.25);
        let mut tracker = DrawdownTracker::new(config, 10_000.0);

        // Normal - full size
        assert!((tracker.position_multiplier() - 1.0).abs() < 1e-9);

        // Warning level (1.5% drawdown)
        tracker.update_equity(9_850.0);
        assert_eq!(tracker.level(), DrawdownLevel::Warning);
        assert!((tracker.position_multiplier() - 0.5).abs() < 1e-9);

        // Critical level (2.5% drawdown)
        tracker.update_equity(9_750.0);
        assert_eq!(tracker.level(), DrawdownLevel::Critical);
        assert!((tracker.position_multiplier() - 0.25).abs() < 1e-9);

        // Emergency level (3.5% drawdown) - zero size
        tracker.update_equity(9_650.0);
        assert_eq!(tracker.level(), DrawdownLevel::Emergency);
        assert!((tracker.position_multiplier() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_drawdown_peak_tracking() {
        let config = DrawdownConfig::default();
        let mut tracker = DrawdownTracker::new(config, 10_000.0);

        // New high
        tracker.update_equity(11_000.0);
        assert_eq!(tracker.peak_equity(), 11_000.0);
        assert!((tracker.drawdown_pct() - 0.0).abs() < 1e-9);

        // Drawdown from new peak
        tracker.update_equity(10_500.0);
        assert_eq!(tracker.peak_equity(), 11_000.0); // Peak unchanged
        assert!((tracker.drawdown_pct() - 500.0 / 11_000.0).abs() < 0.001);
    }

    // =========================================================================
    // Risk Checker Tests
    // =========================================================================

    #[test]
    fn test_risk_checker_position_limits() {
        let limits = RiskLimits::new(100_000.0, 80_000.0); // Hard 100k, soft 80k
        let checker = RiskChecker::new(limits);

        // Within limits
        assert!(matches!(
            checker.check_position(50_000.0),
            RiskCheckResult::Ok
        ));

        // Soft breach (between 80k and 100k)
        let result = checker.check_position(90_000.0);
        assert!(result.is_soft_breach());

        // Hard breach (above 100k)
        let result = checker.check_position(125_000.0);
        assert!(result.is_hard_breach());
    }

    #[test]
    fn test_risk_checker_order_size() {
        let limits = RiskLimits::default().with_max_order_size(1.0);
        let checker = RiskChecker::new(limits);

        assert!(matches!(
            checker.check_order_size(0.5),
            RiskCheckResult::Ok
        ));
        assert!(checker.check_order_size(1.5).is_hard_breach());
    }

    #[test]
    fn test_risk_checker_size_multiplier() {
        let limits = RiskLimits::new(10_000.0, 5_000.0);
        let checker = RiskChecker::new(limits);

        // Below soft limit - full size
        assert!((checker.suggested_size_multiplier(3_000.0) - 1.0).abs() < 1e-9);

        // Midway between soft and hard - 50%
        assert!((checker.suggested_size_multiplier(7_500.0) - 0.5).abs() < 1e-9);

        // At hard limit - zero
        assert!((checker.suggested_size_multiplier(10_000.0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_risk_checker_min_spread() {
        let limits = RiskLimits::default().with_min_spread_bps(5.0);
        let checker = RiskChecker::new(limits);

        // Spread above minimum - unchanged
        assert!((checker.enforce_min_spread(10.0) - 10.0).abs() < 1e-9);

        // Spread below minimum - raised to minimum
        assert!((checker.enforce_min_spread(3.0) - 5.0).abs() < 1e-9);
    }

    // =========================================================================
    // Alerter Tests
    // =========================================================================

    #[test]
    fn test_alert_on_calibration_degradation() {
        let config = AlertConfig::default();
        let alerter = Alerter::new(config, 100);
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Check calibration with low IR (below 1.0 threshold)
        let alert = alerter.check_calibration(0.8, "fill_model", ts);

        assert!(alert.is_some(), "Should generate alert for IR < 1.0");
        let alert = alert.unwrap();
        assert_eq!(alert.alert_type, AlertType::CalibrationDegraded);
        assert!(alert.severity >= AlertSeverity::Warning);
    }

    #[test]
    fn test_alert_on_drawdown() {
        let config = AlertConfig::default();
        let alerter = Alerter::new(config, 100);
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Warning level drawdown
        let alert = alerter.check_drawdown(0.015, ts);
        assert!(alert.is_some());
        let alert = alert.unwrap();
        assert_eq!(alert.alert_type, AlertType::DrawdownWarning);

        // Critical level drawdown
        let alert2 = alerter.check_drawdown(0.025, ts + 400_000); // After dedup window
        assert!(alert2.is_some());
        let alert2 = alert2.unwrap();
        assert_eq!(alert2.alert_type, AlertType::DrawdownCritical);
    }

    #[test]
    fn test_alert_deduplication() {
        let config = AlertConfig {
            dedup_window_s: 60,
            ..Default::default()
        };
        let alerter = Alerter::new(config, 100);
        let ts = 1_000_000u64;

        // First alert should pass
        let alert1 = alerter.check_drawdown(0.015, ts);
        assert!(alert1.is_some());

        // Second alert within window should be deduplicated
        let alert2 = alerter.check_drawdown(0.015, ts + 1000);
        assert!(alert2.is_none());

        // Alert after window should pass
        let alert3 = alerter.check_drawdown(0.015, ts + 61_000);
        assert!(alert3.is_some());
    }

    // =========================================================================
    // Signal Health Monitor Tests
    // =========================================================================

    #[test]
    fn test_signal_health_degradation() {
        let mut monitor = SignalHealthMonitor::new(0.5, 0.75);

        // Register signals
        monitor.register_signal(
            EdgeSignalKind::LeadLag,
            "Lead-Lag Signal".to_string(),
            0.8,
        );
        monitor.register_signal(
            EdgeSignalKind::FillProbability,
            "Fill Probability".to_string(),
            0.6,
        );

        // Initially all healthy
        assert!(monitor.all_healthy());

        // Degrade one signal (60% of baseline = degraded but not stale)
        monitor.update_signal(EdgeSignalKind::LeadLag, 0.48, 1000);

        // Should have one degraded signal
        assert!(!monitor.all_healthy());
        let degraded = monitor.degraded_signals();
        assert!(degraded.contains(&EdgeSignalKind::LeadLag));
    }

    #[test]
    fn test_signal_health_stale_detection() {
        let mut monitor = SignalHealthMonitor::new(0.5, 0.75);

        monitor.register_signal(
            EdgeSignalKind::AdverseSelection,
            "AS Signal".to_string(),
            1.0,
        );

        // Make signal stale (< 50% of baseline)
        monitor.update_signal(EdgeSignalKind::AdverseSelection, 0.3, 1000);

        let stale = monitor.stale_signals();
        assert!(stale.contains(&EdgeSignalKind::AdverseSelection));
    }

    // =========================================================================
    // Full Component Pipeline Test
    // =========================================================================

    #[test]
    fn test_full_component_pipeline() {
        // End-to-end test simulating a quote cycle with all components

        // 1. Initialize components
        let mut circuit_breaker = CircuitBreakerMonitor::new(CircuitBreakerConfig::default());
        let mut hmm = RegimeHMM::new();
        let mut ensemble = AdaptiveEnsemble::new(1.0);
        // Use higher limits for this test (default is $10k)
        let risk_limits = RiskLimits::new(100_000.0, 80_000.0);
        let risk_checker = RiskChecker::new(risk_limits);
        let mut drawdown = DrawdownTracker::new(DrawdownConfig::default(), 100_000.0);
        let alerter = Alerter::new(AlertConfig::default(), 100);

        // 2. Setup models in ensemble
        ensemble.register_model("fill_model");
        ensemble.register_model("as_model");

        // 3. Simulate market update
        let base_time = 1_000_000u64;
        circuit_breaker.update_oi(base_time, 1000.0);
        hmm.forward_update(&HmmObservation {
            volatility: 0.002,
            spread_bps: 5.0,
            flow_imbalance: 0.1,
        });
        drawdown.update_equity(100_000.0);

        // 4. Pre-quote checks
        assert!(
            circuit_breaker.most_severe_action().is_none(),
            "Circuit breaker should not be triggered in normal conditions"
        );
        assert!(
            matches!(risk_checker.check_position(50_000.0), RiskCheckResult::Ok),
            "Position should be within limits"
        );
        assert!(
            !drawdown.should_pause(),
            "Drawdown should not trigger pause"
        );

        // 5. Get regime for quoting
        let regime = hmm.most_likely_regime();
        assert!(
            regime <= 1,
            "Should be in Low or Normal regime, got {}",
            regime
        );

        // 6. Verify alerter has no unacked alerts yet
        assert_eq!(
            alerter.unacknowledged_alerts().len(),
            0,
            "No alerts should be generated in normal conditions"
        );

        // 7. Pipeline passes - would proceed to quote
        println!("Full pipeline check passed, ready to quote");
    }

    #[test]
    fn test_component_pipeline_under_stress() {
        // Test pipeline behavior under stress conditions

        // 1. Initialize components
        let mut circuit_breaker = CircuitBreakerMonitor::new(CircuitBreakerConfig::default());
        let mut hmm = RegimeHMM::new();
        let mut ensemble = AdaptiveEnsemble::new(1.0);
        let _risk_checker = RiskChecker::new(RiskLimits::default());
        let mut drawdown = DrawdownTracker::new(DrawdownConfig::default(), 100_000.0);
        let alerter = Alerter::new(AlertConfig::default(), 100);

        // 2. Setup models
        ensemble.register_model("fill_model");
        ensemble.register_model("as_model");

        // 3. Train ensemble to detect degradation
        for _ in 0..25 {
            ensemble.update_performance("fill_model", 0.7, 0.4, 1000); // Degraded
            ensemble.update_performance("as_model", 1.5, 0.15, 1000); // Good
        }

        // 4. Simulate stress conditions
        let base_time = 1_000_000u64;
        circuit_breaker.update_oi(base_time, 1000.0);
        circuit_breaker.update_oi(base_time + 30_000, 940.0); // 6% OI drop

        // Feed volatile observations to HMM
        for _ in 0..10 {
            hmm.forward_update(&HmmObservation {
                volatility: 0.05,
                spread_bps: 30.0,
                flow_imbalance: 0.6,
            });
        }

        // Simulate drawdown
        drawdown.update_equity(96_500.0); // 3.5% drawdown

        // 5. Check stress responses
        assert!(
            circuit_breaker.check_oi_cascade(base_time + 30_000).is_some(),
            "Should detect OI cascade"
        );

        let belief = hmm.regime_probabilities();
        assert!(
            belief[2] + belief[3] > 0.3,
            "High+Extreme probability should be elevated"
        );

        assert!(
            ensemble.is_model_degraded("fill_model"),
            "Fill model should be marked as degraded"
        );

        assert!(
            drawdown.should_pause(),
            "Should trigger trading pause at 3.5% drawdown"
        );

        // 6. Verify alerts would be generated
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let calibration_alert = alerter.check_calibration(0.7, "fill_model", ts);
        assert!(
            calibration_alert.is_some(),
            "Should generate calibration alert"
        );

        println!("Stress test completed - all defensive mechanisms triggered correctly");
    }

    #[test]
    fn test_regime_and_ensemble_coordination() {
        // Test that regime detection and ensemble weighting work together

        let mut hmm = RegimeHMM::new();
        let mut ensemble = AdaptiveEnsemble::new(1.0);

        // Register models
        ensemble.register_model("normal_regime_model");
        ensemble.register_model("high_volatility_model");

        // Train ensemble (high vol model performs better during stress)
        for _ in 0..25 {
            ensemble.update_performance("normal_regime_model", 1.5, 0.15, 1000);
            ensemble.update_performance("high_volatility_model", 1.2, 0.2, 1000);
        }

        // Initially in normal regime
        for _ in 0..10 {
            hmm.forward_update(&HmmObservation {
                volatility: 0.002,
                spread_bps: 5.0,
                flow_imbalance: 0.0,
            });
        }

        let normal_regime_weight = ensemble.get_weight("normal_regime_model");
        assert!(
            normal_regime_weight > ensemble.get_weight("high_volatility_model"),
            "Normal regime model should have higher weight"
        );

        // Simulate regime transition
        for _ in 0..15 {
            hmm.forward_update(&HmmObservation {
                volatility: 0.03,
                spread_bps: 20.0,
                flow_imbalance: 0.4,
            });
        }

        // Regime should shift toward High/Extreme
        let belief = hmm.regime_probabilities();
        assert!(
            belief[2] + belief[3] > 0.2,
            "Should have elevated probability in High/Extreme regimes"
        );

        // In practice, the quote engine would now:
        // 1. Use HMM belief to blend regime-specific parameters
        // 2. Use ensemble weights to combine model predictions
        // 3. Adjust spreads based on regime confidence
    }
}
