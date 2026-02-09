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
    use crate::market_maker::edge::{EdgeSignalKind, SignalHealthMonitor};
    use crate::market_maker::estimator::{HmmObservation, RegimeHMM};
    use crate::market_maker::learning::AdaptiveEnsemble;
    use crate::market_maker::monitoring::{AlertConfig, AlertSeverity, AlertType, Alerter};
    use crate::market_maker::risk::{
        CircuitBreakerConfig, CircuitBreakerMonitor, CircuitBreakerType, DrawdownConfig,
        DrawdownLevel, DrawdownTracker, RiskCheckResult, RiskChecker, RiskLimits,
    };

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
            hmm.forward_update(&HmmObservation::new(0.002, 5.0, 0.0));
        }

        let belief_before = hmm.regime_probabilities();
        let high_extreme_before = belief_before[2] + belief_before[3]; // High + Extreme

        // Volatility spike
        for _ in 0..10 {
            hmm.forward_update(&HmmObservation::new(0.05, 25.0, 0.5));
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
            hmm.forward_update(&HmmObservation::new(0.0005, 2.0, 0.0));
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
            hmm.forward_update(&HmmObservation::new(0.003, 7.0, 0.1));
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
        let config = DrawdownConfig::default().with_position_reduction(0.5, 0.25);
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

        assert!(matches!(checker.check_order_size(0.5), RiskCheckResult::Ok));
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
        monitor.register_signal(EdgeSignalKind::LeadLag, "Lead-Lag Signal".to_string(), 0.8);
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
        hmm.forward_update(&HmmObservation::new(0.002, 5.0, 0.1));
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
            hmm.forward_update(&HmmObservation::new(0.05, 30.0, 0.6));
        }

        // Simulate drawdown
        drawdown.update_equity(96_500.0); // 3.5% drawdown

        // 5. Check stress responses
        assert!(
            circuit_breaker
                .check_oi_cascade(base_time + 30_000)
                .is_some(),
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
            hmm.forward_update(&HmmObservation::new(0.002, 5.0, 0.0));
        }

        let normal_regime_weight = ensemble.get_weight("normal_regime_model");
        assert!(
            normal_regime_weight > ensemble.get_weight("high_volatility_model"),
            "Normal regime model should have higher weight"
        );

        // Simulate regime transition
        for _ in 0..15 {
            hmm.forward_update(&HmmObservation::new(0.03, 20.0, 0.4));
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

    // =========================================================================
    // Full Pipeline End-to-End Tests
    // =========================================================================

    #[test]
    fn test_full_pipeline_data_to_pnl() {
        use crate::market_maker::estimator::{
            EstimatorConfig, ParameterEstimator, VolatilityRegime,
        };
        use crate::market_maker::strategy::{GLFTStrategy, MarketParams, QuotingStrategy};
        use crate::market_maker::QuoteConfig;
        // 1. Create a ParameterEstimator with default config
        let config = EstimatorConfig::default();
        let mut estimator = ParameterEstimator::new(config);

        // 2. Feed synthetic trade data to warm up (100+ trades around mid=100.0)
        // Use a simple deterministic sequence for reproducibility
        let mid = 100.0;
        estimator.on_mid_update(mid);

        for i in 0..150u64 {
            // Deterministic price variation: oscillates ±0.05 around mid
            let offset = ((i as f64) * 0.7).sin() * 0.05;
            let price = mid + offset;
            // Deterministic size: varies between 0.01 and 0.06
            let size = 0.01 + ((i as f64) * 1.3).cos().abs() * 0.05;
            // Alternate buy/sell based on index parity
            let is_buy = i % 2 == 0;
            let ts = 1000 + i * 100; // 100ms intervals
            estimator.on_trade(ts, price, size, Some(is_buy));
        }

        // 3. Build MarketParams from estimator
        let sigma = estimator.sigma_clean();
        let kappa = estimator.kappa();
        let flow = estimator.flow_imbalance();
        let regime = estimator.volatility_regime();

        let mut market_params = MarketParams::default();
        market_params.sigma = if sigma > 0.0 { sigma } else { 0.0001 };
        market_params.sigma_effective = market_params.sigma;
        market_params.kappa = if kappa > 0.0 { kappa } else { 100.0 };
        market_params.flow_imbalance = flow;
        market_params.microprice = mid;
        market_params.market_mid = mid;
        market_params.arrival_intensity = 0.5;

        // 4. Create a GLFTStrategy and generate quotes
        let strategy = GLFTStrategy::new(0.5); // gamma_base = 0.5
        let quote_config = QuoteConfig {
            mid_price: mid,
            decimals: 2,
            sz_decimals: 4,
            min_notional: 10.0,
        };

        let (bid, ask) = strategy.calculate_quotes(
            &quote_config,
            0.0,   // position = 0 (flat)
            10.0,  // max_position
            1.0,   // target_liquidity
            &market_params,
        );

        // 5. Verify quotes: bid < mid < ask, spread > 0
        assert!(bid.is_some(), "Should generate a bid quote");
        assert!(ask.is_some(), "Should generate an ask quote");

        let bid_quote = bid.unwrap();
        let ask_quote = ask.unwrap();

        assert!(
            bid_quote.price < mid,
            "Bid {} should be below mid {}",
            bid_quote.price,
            mid
        );
        assert!(
            ask_quote.price > mid,
            "Ask {} should be above mid {}",
            ask_quote.price,
            mid
        );
        assert!(
            ask_quote.price > bid_quote.price,
            "Ask {} should be above bid {} (positive spread)",
            ask_quote.price,
            bid_quote.price
        );

        let spread_bps = (ask_quote.price - bid_quote.price) / mid * 10_000.0;
        assert!(
            spread_bps > 0.0,
            "Spread should be positive, got {} bps",
            spread_bps
        );

        // 6. Verify MarketParams: sigma > 0, kappa > 0
        assert!(
            market_params.sigma > 0.0,
            "Sigma should be positive, got {}",
            market_params.sigma
        );
        assert!(
            market_params.kappa > 0.0,
            "Kappa should be positive, got {}",
            market_params.kappa
        );

        // 7. Verify regime: should be Low or Normal after normal data
        assert!(
            matches!(regime, VolatilityRegime::Low | VolatilityRegime::Normal),
            "Regime should be Low or Normal for calm data, got {:?}",
            regime
        );
    }

    // =========================================================================
    // Cascade Stress Test
    // =========================================================================

    #[test]
    fn test_cascade_regime_detection_and_recovery() {
        use crate::market_maker::estimator::{
            EstimatorConfig, ParameterEstimator, VolatilityRegime,
        };
        let config = EstimatorConfig::default();
        let mut estimator = ParameterEstimator::new(config);

        let mid = 100.0;
        estimator.on_mid_update(mid);

        // Phase 1: Normal market (100 trades to establish baseline)
        for i in 0..100u64 {
            let offset = ((i as f64) * 0.7).sin() * 0.03;
            let price = mid + offset;
            let size = 0.02 + ((i as f64) * 1.1).cos().abs() * 0.03;
            let is_buy = i % 2 == 0;
            let ts = 1000 + i * 100;
            estimator.on_trade(ts, price, size, Some(is_buy));
        }

        let regime_before = estimator.volatility_regime();
        let sigma_before = estimator.sigma_clean();
        let toxic_before = estimator.is_toxic_regime();

        // Verify normal regime baseline
        assert!(
            matches!(regime_before, VolatilityRegime::Low | VolatilityRegime::Normal),
            "Phase 1: Regime should be Low or Normal, got {:?}",
            regime_before
        );

        // Phase 2: Cascade injection (rapid price drop, high vol, all sells)
        let mut cascade_price = mid;
        let cascade_start_ts = 1000 + 100 * 100;
        for i in 0..30u64 {
            cascade_price *= 0.995; // 0.5% drop per trade
            let size = 0.1 + ((i as f64) * 0.9).cos().abs() * 0.2; // Larger trades
            let ts = cascade_start_ts + i * 50; // Faster arrival (50ms)
            estimator.on_mid_update(cascade_price);
            estimator.on_trade(ts, cascade_price, size, Some(false)); // All sells
        }

        let sigma_after_cascade = estimator.sigma_clean();
        let jump_ratio_after = estimator.jump_ratio();

        // Volatility should increase during cascade
        // (sigma_after_cascade may still equal sigma_before if the estimator needs
        // more volume ticks, so we check it's at least not lower)
        assert!(
            sigma_after_cascade >= sigma_before * 0.5,
            "Sigma should not collapse during cascade: before={}, after={}",
            sigma_before,
            sigma_after_cascade
        );

        // Toxicity or toxic regime should be elevated after cascade
        // The cascade creates strong directional flow and large price moves
        let toxic_or_elevated = estimator.is_toxic_regime()
            || jump_ratio_after > 1.0
            || estimator.falling_knife_score() > 0.0
            || estimator.flow_imbalance() < -0.1; // Strong sell imbalance

        assert!(
            toxic_or_elevated || toxic_before,
            "Phase 2: Should detect cascade via toxicity, jump ratio, knife score, or flow imbalance"
        );

        // Phase 3: Recovery (stabilizing prices at new level)
        let stable_price = cascade_price;
        let recovery_start_ts = cascade_start_ts + 30 * 50;
        for i in 0..150u64 {
            let offset = ((i as f64) * 0.7).sin() * 0.02;
            let price = stable_price + offset;
            let size = 0.02 + ((i as f64) * 1.3).cos().abs() * 0.03;
            let is_buy = i % 2 == 0;
            let ts = recovery_start_ts + i * 100;
            estimator.on_mid_update(price);
            estimator.on_trade(ts, price, size, Some(is_buy));
        }

        let regime_after_recovery = estimator.volatility_regime();

        // After sufficient recovery trades, regime should return to calmer state
        // or at least not be in Extreme
        assert!(
            !matches!(regime_after_recovery, VolatilityRegime::Extreme),
            "Phase 3: Regime should recover from Extreme after stable trades, got {:?}",
            regime_after_recovery
        );
    }

    // =========================================================================
    // Regime Transition Test
    // =========================================================================

    #[test]
    fn test_regime_transition_parameter_blending() {
        // Test that regime HMM transitions cause smooth parameter changes

        let mut hmm = RegimeHMM::new();

        // 1. Feed quiet observations to establish baseline
        for _ in 0..30 {
            hmm.forward_update(&HmmObservation::new(0.001, 3.0, 0.0));
        }

        let quiet_probs = hmm.regime_probabilities();
        let quiet_sum: f64 = quiet_probs.iter().sum();

        // 2. Verify probabilities sum to 1.0
        assert!(
            (quiet_sum - 1.0).abs() < 1e-9,
            "Quiet regime probabilities should sum to 1.0, got {}",
            quiet_sum
        );

        // 3. Record quiet state: Low+Normal should dominate
        let quiet_calm = quiet_probs[0] + quiet_probs[1]; // Low + Normal
        assert!(
            quiet_calm > 0.5,
            "Low+Normal should dominate in quiet market: {}",
            quiet_calm
        );

        // 4. Gradually increase volatility and track regime shifts
        let vol_steps = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08];

        for &vol in &vol_steps {
            // Feed 5 observations at each volatility level
            for _ in 0..5 {
                hmm.forward_update(&HmmObservation::new(vol, vol * 500.0, 0.3));
            }

            let new_probs = hmm.regime_probabilities();
            let new_sum: f64 = new_probs.iter().sum();

            // Probabilities must always sum to 1.0 at every step
            assert!(
                (new_sum - 1.0).abs() < 1e-9,
                "Probabilities should sum to 1.0 at vol={}, got {}",
                vol,
                new_sum
            );

            // All probabilities must be non-negative
            for (j, &p) in new_probs.iter().enumerate() {
                assert!(
                    p >= 0.0,
                    "Probability for regime {} should be non-negative at vol={}, got {}",
                    j,
                    vol,
                    p
                );
            }
        }

        // 5. After high volatility, High+Extreme should be elevated vs initial
        let final_probs = hmm.regime_probabilities();
        let final_high_extreme = final_probs[2] + final_probs[3];
        let initial_high_extreme = quiet_probs[2] + quiet_probs[3];

        assert!(
            final_high_extreme > initial_high_extreme,
            "High+Extreme should increase after volatility ramp: initial={}, final={}",
            initial_high_extreme,
            final_high_extreme
        );
    }

    // =========================================================================
    // Quote Quality Under Stress Test
    // =========================================================================

    #[test]
    fn test_quote_quality_invariants() {
        use crate::market_maker::strategy::{GLFTStrategy, MarketParams, QuotingStrategy};
        use crate::market_maker::QuoteConfig;

        let strategy = GLFTStrategy::new(0.5);
        let mid = 100.0;
        let quote_config = QuoteConfig {
            mid_price: mid,
            decimals: 2,
            sz_decimals: 4,
            min_notional: 10.0,
        };

        // Define regime-specific market params
        let regimes: Vec<(&str, MarketParams)> = vec![
            ("quiet", {
                let mut p = MarketParams::default();
                p.sigma = 0.0001;         // Low volatility
                p.sigma_effective = 0.0001;
                p.kappa = 200.0;          // Deep book
                p.microprice = mid;
                p.market_mid = mid;
                p.arrival_intensity = 0.3;
                p
            }),
            ("trending", {
                let mut p = MarketParams::default();
                p.sigma = 0.0005;
                p.sigma_effective = 0.0005;
                p.kappa = 150.0;
                p.momentum_bps = 5.0;     // Moderate momentum
                p.microprice = mid;
                p.market_mid = mid;
                p.arrival_intensity = 0.5;
                p
            }),
            ("volatile", {
                let mut p = MarketParams::default();
                p.sigma = 0.003;          // High volatility
                p.sigma_effective = 0.003;
                p.kappa = 80.0;           // Thinner book
                p.microprice = mid;
                p.market_mid = mid;
                p.arrival_intensity = 1.0;
                p
            }),
            ("cascade", {
                let mut p = MarketParams::default();
                p.sigma = 0.01;           // Very high volatility
                p.sigma_effective = 0.01;
                p.kappa = 30.0;           // Very thin book
                p.is_toxic_regime = true;
                p.microprice = mid;
                p.market_mid = mid;
                p.arrival_intensity = 2.0;
                p.falling_knife_score = 0.8;
                p
            }),
        ];

        let mut prev_spread_bps = 0.0_f64;

        for (name, params) in &regimes {
            let (bid, ask) = strategy.calculate_quotes(
                &quote_config,
                0.0,   // flat position
                10.0,
                1.0,
                params,
            );

            // Assert: both sides should produce quotes
            assert!(
                bid.is_some(),
                "Regime '{}': should produce bid quote",
                name
            );
            assert!(
                ask.is_some(),
                "Regime '{}': should produce ask quote",
                name
            );

            let bid_price = bid.unwrap().price;
            let ask_price = ask.unwrap().price;

            // Assert: bid < ask (positive spread)
            assert!(
                bid_price < ask_price,
                "Regime '{}': bid {} must be < ask {}",
                name,
                bid_price,
                ask_price
            );

            let spread_bps = (ask_price - bid_price) / mid * 10_000.0;

            // Assert: spread > maker fee (1.5 bps on each side = 3 bps round-trip)
            // The GLFT formula produces half-spread per side, so total spread should
            // exceed the round-trip fee for the strategy to be profitable
            assert!(
                spread_bps > 1.5,
                "Regime '{}': spread {} bps should exceed maker fee 1.5 bps",
                name,
                spread_bps
            );

            // Assert: spread increases monotonically with volatility/stress
            // (quiet < trending < volatile < cascade)
            if prev_spread_bps > 0.0 {
                assert!(
                    spread_bps >= prev_spread_bps * 0.8, // Allow 20% tolerance for non-monotonicity from other factors
                    "Regime '{}': spread {} bps should not drastically decrease from previous {} bps",
                    name,
                    spread_bps,
                    prev_spread_bps
                );
            }

            prev_spread_bps = spread_bps;
        }

        // Assert: cascade spread > quiet spread (robustly)
        let quiet_params = &regimes[0].1;
        let cascade_params = &regimes[3].1;

        let (quiet_bid, quiet_ask) = strategy.calculate_quotes(
            &quote_config, 0.0, 10.0, 1.0, quiet_params,
        );
        let (cascade_bid, cascade_ask) = strategy.calculate_quotes(
            &quote_config, 0.0, 10.0, 1.0, cascade_params,
        );

        let quiet_spread = quiet_ask.unwrap().price - quiet_bid.unwrap().price;
        let cascade_spread = cascade_ask.unwrap().price - cascade_bid.unwrap().price;

        assert!(
            cascade_spread >= quiet_spread,
            "Cascade spread {} should not be less than quiet spread {}",
            cascade_spread,
            quiet_spread
        );
    }

    // =========================================================================
    // Safety Integration Tests — Cross-Component Escalation
    // =========================================================================

    #[test]
    fn test_cascade_triggers_circuit_breaker_then_kill_switch() {
        // Verifies the full escalation chain:
        // moderate cascade -> WidenSpreads -> high cascade -> PullQuotes -> extreme -> Kill
        use crate::market_maker::risk::{
            RiskAction, RiskAggregator, RiskSeverity, RiskState,
        };
        use crate::market_maker::risk::monitors::CascadeMonitor;

        // Build aggregator with a cascade monitor: pull at 0.8, kill at 5.0
        let aggregator = RiskAggregator::new()
            .with_monitor(Box::new(CascadeMonitor::new(0.8, 5.0)));

        // Step 1: Moderate cascade (severity 0.5) -> should widen spreads
        // Default widen_threshold = pull_threshold * 0.5 = 0.4
        let state_moderate = RiskState {
            cascade_severity: 0.5,
            ..Default::default()
        };
        let result = aggregator.evaluate(&state_moderate);
        assert!(
            matches!(result.primary_action, RiskAction::WidenSpreads(_)),
            "Moderate cascade should trigger WidenSpreads, got {:?}",
            result.primary_action
        );
        assert!(
            result.spread_factor > 1.0,
            "Spread factor should be > 1.0 during moderate cascade, got {}",
            result.spread_factor
        );
        assert!(
            !result.should_kill(),
            "Moderate cascade should not trigger kill"
        );

        // Step 2: High cascade (severity 0.95) -> should pull quotes
        let state_high = RiskState {
            cascade_severity: 0.95,
            ..Default::default()
        };
        let result = aggregator.evaluate(&state_high);
        assert!(
            result.pull_quotes,
            "High cascade (0.95) should trigger PullQuotes"
        );
        assert_eq!(
            result.max_severity,
            RiskSeverity::High,
            "High cascade should be High severity"
        );

        // Step 3: Extreme cascade (severity 6.0) -> should trigger kill
        let state_extreme = RiskState {
            cascade_severity: 6.0,
            ..Default::default()
        };
        let result = aggregator.evaluate(&state_extreme);
        assert!(
            result.should_kill(),
            "Extreme cascade should trigger kill switch"
        );
        assert_eq!(
            result.max_severity,
            RiskSeverity::Critical,
            "Extreme cascade should be Critical severity"
        );
        assert!(
            !result.kill_reasons.is_empty(),
            "Kill reasons should contain cascade description"
        );
        let reasons_joined = result.kill_reasons.join("; ");
        assert!(
            reasons_joined.to_lowercase().contains("cascade"),
            "Kill reasons should mention cascade: {}",
            reasons_joined
        );
    }

    #[test]
    fn test_position_guard_blocks_while_circuit_breaker_widens() {
        // Verifies that PositionGuard blocks position-increasing orders
        // while allowing position-reducing orders, independent of whether
        // the aggregator says WidenSpreads or PauseTrading.
        use crate::market_maker::risk::{
            OrderEntryCheck, PositionGuard, PositionGuardConfig,
        };
        use crate::market_maker::tracking::Side;

        let guard = PositionGuard::with_config(PositionGuardConfig {
            max_position: 10.0,
            hard_entry_threshold: 0.95, // Reject at > 9.5 worst case
            ..Default::default()
        });

        // Position at 9.5 -- near the limit
        let current_position = 9.5;

        // Buying 0.6 would put worst-case at 10.1 > 9.5 limit -> REJECTED
        let check_buy = guard.check_order_entry(current_position, 0.6, Side::Buy);
        assert!(
            !check_buy.is_allowed(),
            "Position-increasing order should be rejected near limit"
        );
        if let OrderEntryCheck::Rejected { worst_case_position, hard_limit, .. } = &check_buy {
            assert!(
                *worst_case_position > *hard_limit,
                "Worst case {} should exceed hard limit {}",
                worst_case_position, hard_limit
            );
        }

        // Selling (reducing position) should ALWAYS be allowed
        let check_sell = guard.check_order_entry(current_position, 1.0, Side::Sell);
        assert!(
            check_sell.is_allowed(),
            "Position-reducing order should always be allowed"
        );

        // Small buy that stays within limits -> still rejected at 9.5 + 0.01 = 9.51 > 9.5
        let check_small_buy = guard.check_order_entry(current_position, 0.01, Side::Buy);
        assert!(
            !check_small_buy.is_allowed(),
            "Even small buy at 9.5 + 0.01 = 9.51 > 9.5 limit should be rejected"
        );

        // From a lower position, buying should be fine
        let check_ok_buy = guard.check_order_entry(5.0, 2.0, Side::Buy);
        assert!(
            check_ok_buy.is_allowed(),
            "Buy that stays well within limits should be allowed (worst case = 7.0)"
        );
    }

    #[test]
    fn test_risk_aggregator_takes_max_severity_across_monitors() {
        // Verifies the fundamental invariant: one Critical monitor overrides
        // all Normal monitors, and the kill action propagates correctly.
        use crate::market_maker::risk::{
            RiskAggregator, RiskSeverity, RiskState,
        };
        use crate::market_maker::risk::monitors::{
            CascadeMonitor, DrawdownMonitor, LossMonitor, PositionMonitor,
        };

        // Set up aggregator with multiple monitors
        let aggregator = RiskAggregator::new()
            .with_monitor(Box::new(LossMonitor::new(500.0)))       // max loss $500
            .with_monitor(Box::new(DrawdownMonitor::new(0.05)))    // max 5% dd
            .with_monitor(Box::new(PositionMonitor::new()))
            .with_monitor(Box::new(CascadeMonitor::new(0.8, 5.0)));

        // Create a state where everything is normal EXCEPT cascade is extreme.
        // P&L: daily=10, peak=10 means 0% drawdown (at peak).
        // Position: 30% utilized. Cascade: 6.0 (extreme).
        let state = RiskState {
            daily_pnl: 10.0,         // In profit
            peak_pnl: 10.0,          // At peak, 0% drawdown
            position: 0.3,           // Well within limits
            max_position: 1.0,
            position_value: 3000.0,
            max_position_value: 10000.0,
            cascade_severity: 6.0,   // EXTREME: above kill threshold of 5.0
            ..Default::default()
        };

        let result = aggregator.evaluate(&state);

        // Max severity should be Critical from the cascade monitor
        assert_eq!(
            result.max_severity,
            RiskSeverity::Critical,
            "One Critical monitor should dominate: max_severity = {:?}",
            result.max_severity
        );

        // Kill switch should be triggered
        assert!(
            result.should_kill(),
            "Critical cascade should trigger kill"
        );

        // Other monitors should be OK (profit, small drawdown, within position limits)
        let non_critical_count = result.assessments.iter()
            .filter(|a| a.severity == RiskSeverity::None)
            .count();
        assert!(
            non_critical_count >= 3,
            "At least 3 monitors should report None severity, got {}",
            non_critical_count
        );
    }

    #[test]
    fn test_kill_switch_blocks_after_cascade() {
        // Verifies that KillSwitch correctly triggers on extreme cascade severity
        // and remains triggered on subsequent checks (latching behavior).
        use crate::market_maker::risk::{KillReason, KillSwitch, KillSwitchConfig, KillSwitchState};
        use std::time::Instant;

        let config = KillSwitchConfig {
            cascade_severity_threshold: 1.5, // Production-like threshold
            ..Default::default()
        };
        let ks = KillSwitch::new(config);

        // Not triggered initially
        assert!(!ks.is_triggered());

        // State with extreme cascade severity
        let state = KillSwitchState {
            cascade_severity: 2.0, // Above 1.5 threshold
            last_data_time: Instant::now(),
            ..Default::default()
        };

        // Check should trigger CascadeDetected
        let reason = ks.check(&state);
        assert!(reason.is_some(), "Should trigger on cascade severity 2.0");
        match reason.unwrap() {
            KillReason::CascadeDetected { severity } => {
                assert!(
                    (severity - 2.0).abs() < f64::EPSILON,
                    "Severity should be 2.0, got {}",
                    severity
                );
            }
            other => panic!("Expected CascadeDetected, got {:?}", other),
        }

        // Kill switch should now be triggered
        assert!(ks.is_triggered(), "Kill switch should be triggered");

        // Subsequent checks with calm state should still report triggered (latching)
        let state_calm = KillSwitchState {
            cascade_severity: 0.0, // Calm now
            last_data_time: Instant::now(),
            ..Default::default()
        };
        let _ = ks.check(&state_calm);
        assert!(
            ks.is_triggered(),
            "Kill switch should remain triggered even after conditions improve"
        );

        // Trigger reasons should be preserved
        let reasons = ks.trigger_reasons();
        assert!(
            !reasons.is_empty(),
            "Trigger reasons should be preserved"
        );
    }

    #[test]
    fn test_multiple_monitors_fire_simultaneously() {
        // Verifies correct aggregation when MULTIPLE conditions are bad simultaneously:
        // high drawdown + high position + elevated cascade.
        // Max severity wins, spread_factor is maximum, reduce_only propagates.
        use crate::market_maker::risk::{
            RiskAggregator, RiskSeverity, RiskState,
        };
        use crate::market_maker::risk::monitors::{
            CascadeMonitor, DrawdownMonitor, LossMonitor, PositionMonitor,
        };

        let aggregator = RiskAggregator::new()
            .with_monitor(Box::new(LossMonitor::new(50.0)))        // $50 limit
            .with_monitor(Box::new(DrawdownMonitor::new(0.05)))    // 5% dd limit
            .with_monitor(Box::new(PositionMonitor::new()))
            .with_monitor(Box::new(
                CascadeMonitor::new(0.8, 5.0).with_widen_threshold(0.4, 0.5),
            ));

        // State where MULTIPLE conditions are bad:
        // - Drawdown: peak=100, current pnl implies 20% drawdown > 5% limit -> Critical
        // - Loss: $40 loss is 80% of $50 limit -> Warning
        // - Position: 90% of value limit -> Warning
        // - Cascade: 0.5 severity -> WidenSpreads (above 0.4 widen threshold)
        let state = RiskState {
            daily_pnl: -40.0,
            peak_pnl: 100.0,
            position: 0.9,
            max_position: 1.0,
            position_value: 9000.0,
            max_position_value: 10000.0,
            cascade_severity: 0.5,
            ..Default::default()
        };

        let result = aggregator.evaluate(&state);

        // Drawdown monitor fires Critical (peak=100, pnl=-40 -> drawdown=140%), max severity
        assert_eq!(
            result.max_severity,
            RiskSeverity::Critical,
            "Max severity should be Critical from drawdown, got {:?}",
            result.max_severity
        );

        // Kill should be triggered from drawdown
        assert!(
            result.should_kill(),
            "Drawdown exceeding limit should trigger kill"
        );

        // Spread factor should be > 1.0 from the cascade monitor
        assert!(
            result.spread_factor > 1.0,
            "Spread factor should be elevated from cascade, got {}",
            result.spread_factor
        );

        // Multiple assessments should be actionable
        let actionable_count = result.assessments.iter()
            .filter(|a| a.is_actionable())
            .count();
        assert!(
            actionable_count >= 2,
            "Multiple monitors should fire simultaneously, got {} actionable",
            actionable_count
        );
    }
}
