//! HJB-derived optimal inventory control for market making.
//!
//! Implements the Avellaneda-Stoikov HJB (Hamilton-Jacobi-Bellman) solution
//! for optimal market making with inventory risk.
//!
//! The HJB equation:
//! ```text
//! ∂V/∂t + max_δ { λ(δ)[δ + V(t,x+δ,q-1,S) - V(t,x,q,S)] } - γσ²q² = 0
//! ```
//!
//! With terminal condition: `V(T,x,q,S) = x + q×S - penalty×q²`
//!
//! This module provides:
//! - Optimal inventory skew from the value function gradient
//! - Terminal penalty that forces position reduction before session end
//! - Funding rate integration for perpetuals (carry cost affects optimal inventory)
//! - Theoretically rigorous position management

mod config;
mod controller;
pub mod ou_drift;
pub mod queue_value;
mod skew;
mod summary;

pub use config::HJBConfig;
pub use controller::HJBInventoryController;
pub use ou_drift::{OUDriftConfig, OUDriftEstimator, OUDriftSummary, OUUpdateResult};
pub use queue_value::{
    BatchQueueValue, HJBQueueValueCalculator, HJBQueueValueConfig, OrderQueueValue,
};
pub use summary::{DriftAdjustedSkew, HJBSummary, MomentumStats};

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_controller() -> HJBInventoryController {
        let config = HJBConfig {
            session_duration_secs: 100.0, // Short session for testing
            terminal_penalty: 0.001,
            gamma_base: 0.3,
            ..Default::default()
        };
        let mut ctrl = HJBInventoryController::new(config);
        ctrl.start_session();
        ctrl.update_sigma(0.0001); // 1 bp/sec
        ctrl
    }

    #[test]
    fn test_hjb_basic() {
        let ctrl = make_controller();

        // Should be initialized
        assert!(ctrl.initialized);
        assert!(ctrl.time_remaining() > 0.0);
        assert!(ctrl.terminal_urgency() < 0.5); // Early in session
    }

    #[test]
    fn test_hjb_zero_position_zero_skew() {
        let ctrl = make_controller();

        // With zero position, skew should be ~zero (funding aside)
        let skew = ctrl.optimal_skew(0.0, 1.0);
        assert!(
            skew.abs() < 1e-6,
            "Zero position should give ~zero skew: {}",
            skew
        );
    }

    #[test]
    fn test_hjb_long_position_positive_skew() {
        let ctrl = make_controller();

        // Long position should give positive skew (shift quotes down)
        let skew = ctrl.optimal_skew(0.5, 1.0);
        assert!(
            skew > 0.0,
            "Long position should give positive skew: {}",
            skew
        );
    }

    #[test]
    fn test_hjb_short_position_negative_skew() {
        let ctrl = make_controller();

        // Short position should give negative skew (shift quotes up)
        let skew = ctrl.optimal_skew(-0.5, 1.0);
        assert!(
            skew < 0.0,
            "Short position should give negative skew: {}",
            skew
        );
    }

    #[test]
    fn test_hjb_skew_symmetry() {
        let ctrl = make_controller();

        // Skew should be antisymmetric in position
        let skew_long = ctrl.optimal_skew(0.5, 1.0);
        let skew_short = ctrl.optimal_skew(-0.5, 1.0);

        assert!(
            (skew_long + skew_short).abs() < 1e-8,
            "Skew should be antisymmetric: long={}, short={}",
            skew_long,
            skew_short
        );
    }

    #[test]
    fn test_hjb_gamma_multiplier() {
        let ctrl = make_controller();

        // At start of session, multiplier should be ~1.0
        let mult = ctrl.gamma_multiplier();
        assert!(
            mult >= 1.0 && mult < 1.5,
            "Early multiplier should be near 1.0: {}",
            mult
        );

        // Effective gamma should be base × multiplier (computed at same instant)
        // Get both from effective_gamma method which computes mult internally
        let eff = ctrl.effective_gamma();
        let expected = ctrl.config.gamma_base * ctrl.gamma_multiplier();
        assert!(
            (eff - expected).abs() < 0.01,
            "Effective gamma {} should be close to gamma_base {} × multiplier: expected {}",
            eff,
            ctrl.config.gamma_base,
            expected
        );
    }

    #[test]
    fn test_hjb_optimal_inventory_target_no_funding() {
        let ctrl = make_controller();

        // With zero funding, optimal target is zero
        let target = ctrl.optimal_inventory_target();
        assert!(
            target.abs() < 0.01,
            "With zero funding, target should be ~0: {}",
            target
        );
    }

    #[test]
    fn test_hjb_optimal_inventory_target_with_funding() {
        let mut ctrl = make_controller();

        // Positive funding rate (longs pay) → optimal to be short
        ctrl.update_funding(0.001); // 0.1% 8-hour rate
        let target = ctrl.optimal_inventory_target();

        // With positive funding, target should be negative (short)
        assert!(
            target < 0.0,
            "Positive funding should give negative target: {}",
            target
        );
    }

    #[test]
    fn test_hjb_funding_ewma() {
        // Use a faster-converging controller for testing
        let config = HJBConfig {
            session_duration_secs: 100.0,
            terminal_penalty: 0.001,
            gamma_base: 0.3,
            funding_ewma_half_life: 10.0, // Fast EWMA for testing (10 seconds)
            ..Default::default()
        };
        let mut ctrl = HJBInventoryController::new(config);
        ctrl.start_session();

        // Initial funding rate
        ctrl.update_funding(0.001);
        let rate1 = ctrl.funding_rate_ewma;

        // Update with same rate - EWMA should move toward target
        ctrl.update_funding(0.001);
        let rate2 = ctrl.funding_rate_ewma;

        // Expected annualized rate
        let annualized = 0.001 * 3.0 * 365.0; // = 1.095

        // EWMA should be moving toward the annualized rate
        assert!(
            rate2 > rate1,
            "EWMA should increase toward target: {} -> {}",
            rate1,
            rate2
        );
        assert!(rate2 < annualized, "EWMA should not exceed target");

        // After many updates with fast EWMA, should converge
        for _ in 0..100 {
            ctrl.update_funding(0.001);
        }
        let rate_converged = ctrl.funding_rate_ewma;

        // Should be close to annualized after convergence
        assert!(
            (rate_converged - annualized).abs() / annualized < 0.1,
            "EWMA should converge to annualized rate: {} vs {}",
            rate_converged,
            annualized
        );
    }

    #[test]
    fn test_hjb_value_gradient() {
        let ctrl = make_controller();

        // Value gradient at zero position
        let grad_zero = ctrl.value_gradient(0.0, 1.0, 100.0);

        // Value gradient with long position (should be more negative = higher cost)
        let grad_long = ctrl.value_gradient(0.5, 1.0, 100.0);

        // Holding inventory has cost, so gradient should differ
        // (exact relationship depends on parameters)
        assert!(grad_zero != grad_long, "Gradient should depend on position");
    }

    #[test]
    fn test_hjb_terminal_zone() {
        let ctrl = make_controller();

        // Early in session, not in terminal zone
        assert!(!ctrl.is_terminal_zone());
    }

    #[test]
    fn test_hjb_summary() {
        let ctrl = make_controller();
        let summary = ctrl.summary();

        assert!(summary.time_remaining_secs > 0.0);
        assert!(summary.terminal_urgency >= 0.0 && summary.terminal_urgency <= 1.0);
        assert!(summary.gamma_multiplier >= 1.0);
        assert!(summary.sigma > 0.0);
    }

    #[test]
    fn test_hjb_skew_increases_with_position() {
        let ctrl = make_controller();

        // Larger position → larger skew magnitude
        let skew_small = ctrl.optimal_skew(0.1, 1.0);
        let skew_large = ctrl.optimal_skew(0.5, 1.0);

        assert!(
            skew_large.abs() > skew_small.abs(),
            "Larger position should give larger skew: small={}, large={}",
            skew_small,
            skew_large
        );
    }

    #[test]
    fn test_hjb_skew_increases_with_volatility() {
        let mut ctrl = make_controller();

        ctrl.update_sigma(0.0001);
        let skew_low_vol = ctrl.optimal_skew(0.5, 1.0);

        ctrl.update_sigma(0.001); // 10x higher vol
        let skew_high_vol = ctrl.optimal_skew(0.5, 1.0);

        assert!(
            skew_high_vol.abs() > skew_low_vol.abs(),
            "Higher vol should give larger skew: low={}, high={}",
            skew_low_vol,
            skew_high_vol
        );
    }

    #[test]
    fn test_hjb_drift_warmup() {
        // Use EWMA mode for deterministic testing (OU requires actual time passing)
        let config = HJBConfig {
            session_duration_secs: 100.0,
            terminal_penalty: 0.001,
            gamma_base: 0.3,
            use_ou_drift: false, // Disable OU for deterministic test
            ..Default::default()
        };
        let mut ctrl = HJBInventoryController::new(config);
        ctrl.start_session();
        ctrl.update_sigma(0.0001);

        // Initially not warmed up
        assert!(!ctrl.is_drift_warmed_up());

        // Update momentum signals
        for _ in 0..25 {
            ctrl.update_momentum_signals(10.0, 0.6, -0.5, 1.0);
        }

        // Now should be warmed up
        assert!(ctrl.is_drift_warmed_up());
    }

    #[test]
    fn test_hjb_ewma_smoothing() {
        let mut ctrl = make_controller();

        // Start with no drift
        assert!((ctrl.smoothed_drift()).abs() < 1e-10);
        assert!((ctrl.smoothed_variance_multiplier() - 1.0).abs() < 1e-10);

        // Add positive momentum (opposed to short position)
        for _ in 0..30 {
            ctrl.update_momentum_signals(20.0, 0.7, -0.5, 1.0);
        }

        // Drift should be positive (rising)
        assert!(
            ctrl.smoothed_drift() > 0.0,
            "Positive momentum should give positive drift: {}",
            ctrl.smoothed_drift()
        );

        // Variance multiplier should be > 1 (opposed position)
        assert!(
            ctrl.smoothed_variance_multiplier() > 1.0,
            "Opposed position should increase variance: {}",
            ctrl.smoothed_variance_multiplier()
        );
    }

    #[test]
    fn test_hjb_momentum_stats() {
        let mut ctrl = make_controller();

        // Add momentum signals
        for i in 0..30 {
            let momentum = if i % 5 == 0 { -5.0 } else { 15.0 };
            ctrl.update_momentum_signals(momentum, 0.6, -0.3, 1.0);
        }

        let stats = ctrl.momentum_stats();

        assert!(stats.sample_count > 0);
        assert!(stats.mean_bps > 0.0); // Mostly positive momentum
        assert!(stats.std_dev_bps > 0.0); // Some variance
        assert!(stats.direction_changes > 0); // Some direction changes
    }

    #[test]
    fn test_hjb_drift_adjusted_skew_uses_smoothed() {
        // Use EWMA mode for deterministic testing (OU requires actual time passing)
        let config = HJBConfig {
            session_duration_secs: 100.0,
            terminal_penalty: 0.001,
            gamma_base: 0.3,
            use_ou_drift: false, // Disable OU for deterministic test
            ..Default::default()
        };
        let mut ctrl = HJBInventoryController::new(config);
        ctrl.start_session();
        ctrl.update_sigma(0.0001);

        // Warm up with consistent momentum
        for _ in 0..30 {
            ctrl.update_momentum_signals(25.0, 0.7, -0.5, 1.0);
        }

        assert!(ctrl.is_drift_warmed_up());

        // Get drift-adjusted skew
        let result = ctrl.optimal_skew_with_drift(-0.5, 1.0, 25.0, 0.7);

        // Should be opposed (short + rising)
        assert!(result.is_opposed);

        // Should have drift urgency
        assert!(
            result.drift_urgency.abs() > 0.0,
            "Expected drift urgency: {}",
            result.drift_urgency
        );

        // Should use smoothed variance multiplier (> 1.0 for opposed)
        assert!(
            result.variance_multiplier > 1.0,
            "Expected variance > 1.0 for opposed: {}",
            result.variance_multiplier
        );
    }

    #[test]
    fn test_momentum_stats_signal_quality() {
        // Create high quality signal
        let high_quality = MomentumStats {
            mean_bps: 30.0,
            std_dev_bps: 5.0,
            direction_changes: 2,
            sample_count: 50,
            avg_continuation: 0.75,
        };

        // Create low quality signal
        let low_quality = MomentumStats {
            mean_bps: 5.0,
            std_dev_bps: 20.0,
            direction_changes: 25,
            sample_count: 50,
            avg_continuation: 0.45,
        };

        assert!(
            high_quality.signal_quality() > low_quality.signal_quality(),
            "High quality signal should have higher score: {} vs {}",
            high_quality.signal_quality(),
            low_quality.signal_quality()
        );

        assert!(
            !high_quality.is_noisy(),
            "High quality signal should not be noisy"
        );
        assert!(low_quality.is_noisy(), "Low quality signal should be noisy");
    }

    // ==========================================================================
    // Funding Horizon Tests
    // ==========================================================================

    #[test]
    fn test_funding_horizon_uses_settlement_time() {
        let config = HJBConfig {
            session_duration_secs: 86400.0, // 24h fallback
            terminal_penalty: 0.001,
            gamma_base: 0.3,
            use_funding_horizon: true,
            ..Default::default()
        };
        let mut ctrl = HJBInventoryController::new(config);
        ctrl.start_session();
        ctrl.update_sigma(0.0001);

        // Without funding settlement, time_remaining uses session duration
        let tr_session = ctrl.time_remaining();
        assert!(
            tr_session > 86000.0,
            "Without funding settlement, should use ~86400s session: {}",
            tr_session,
        );
        assert!(!ctrl.is_funding_horizon_active());

        // Set funding settlement to 2 hours from now
        ctrl.update_funding_settlement(7200.0);

        // Now time_remaining should reflect funding cycle (8h period)
        // With 2h remaining in the cycle, time_remaining ≈ 7200
        let tr_funding = ctrl.time_remaining();
        assert!(
            tr_funding < 8000.0,
            "With funding settlement set to 2h, time_remaining should be ~7200s: {}",
            tr_funding,
        );
        assert!(ctrl.is_funding_horizon_active());

        // Terminal urgency should reflect 75% through the 8h cycle (6h elapsed, 2h remaining)
        let urgency = ctrl.terminal_urgency();
        assert!(
            urgency > 0.7 && urgency < 0.85,
            "With 2h remaining of 8h cycle, urgency should be ~0.75: {}",
            urgency,
        );
    }

    #[test]
    fn test_funding_horizon_gamma_increases_near_settlement() {
        let config = HJBConfig {
            session_duration_secs: 86400.0,
            terminal_penalty: 0.001,
            gamma_base: 0.3,
            use_funding_horizon: true,
            max_terminal_multiplier: 5.0,
            ..Default::default()
        };
        let mut ctrl = HJBInventoryController::new(config);
        ctrl.start_session();
        ctrl.update_sigma(0.0001);

        // Early in cycle (7h remaining of 8h) → low urgency
        ctrl.update_funding_settlement(25200.0); // 7h
        let gamma_early = ctrl.gamma_multiplier();

        // Late in cycle (30 min remaining) → high urgency
        ctrl.update_funding_settlement(1800.0); // 30 min
        let gamma_late = ctrl.gamma_multiplier();

        assert!(
            gamma_late > gamma_early,
            "Gamma should increase near settlement: early={}, late={}",
            gamma_early,
            gamma_late,
        );

        // Late should show significant urgency (>80% through cycle)
        let urgency_late = ctrl.terminal_urgency();
        assert!(
            urgency_late > 0.9,
            "30 min remaining in 8h cycle should show high urgency: {}",
            urgency_late,
        );
    }

    #[test]
    fn test_funding_horizon_disabled_uses_session() {
        let config = HJBConfig {
            session_duration_secs: 100.0,
            terminal_penalty: 0.001,
            gamma_base: 0.3,
            use_funding_horizon: false, // Disabled
            ..Default::default()
        };
        let mut ctrl = HJBInventoryController::new(config);
        ctrl.start_session();
        ctrl.update_sigma(0.0001);

        // Even with funding settlement set, should use session duration
        ctrl.update_funding_settlement(3600.0);

        let tr = ctrl.time_remaining();
        assert!(
            tr < 110.0,
            "With use_funding_horizon=false, should use session duration: {}",
            tr,
        );
        assert!(!ctrl.is_funding_horizon_active());
    }

    #[test]
    fn test_calibrate_terminal_penalty_from_spread() {
        let config = HJBConfig {
            terminal_penalty: 0.0005,
            calibrate_terminal_penalty: true,
            gamma_base: 0.3,
            ..Default::default()
        };
        let mut ctrl = HJBInventoryController::new(config);
        ctrl.start_session();
        ctrl.update_sigma(0.0001); // 1 bp/sec

        // Before calibration, uses config default
        assert!(
            (ctrl.effective_terminal_penalty() - 0.0005).abs() < 1e-6,
            "Before calibration should use config value: {}",
            ctrl.effective_terminal_penalty(),
        );

        // Calibrate with 20 bps market spread
        ctrl.calibrate_terminal_penalty(20.0);

        let penalty = ctrl.effective_terminal_penalty();
        // spread_cost = 20/10000 = 0.002
        // vol_cost = 0.0001 * sqrt(28800) ≈ 0.017
        // total ≈ 0.019, clamped to [0.00005, 0.01] → 0.01
        assert!(
            penalty > 0.0005,
            "Calibrated penalty should differ from default: {}",
            penalty,
        );
        assert!(
            penalty <= 0.01,
            "Calibrated penalty should be clamped to max 0.01: {}",
            penalty,
        );
    }

    #[test]
    fn test_calibrate_terminal_penalty_disabled() {
        let config = HJBConfig {
            terminal_penalty: 0.0005,
            calibrate_terminal_penalty: false, // Disabled
            gamma_base: 0.3,
            ..Default::default()
        };
        let mut ctrl = HJBInventoryController::new(config);
        ctrl.start_session();
        ctrl.update_sigma(0.0001);

        ctrl.calibrate_terminal_penalty(20.0);

        assert!(
            (ctrl.effective_terminal_penalty() - 0.0005).abs() < 1e-6,
            "With calibration disabled, should use config value: {}",
            ctrl.effective_terminal_penalty(),
        );
    }

    #[test]
    fn test_funding_settlement_clamped_to_valid_range() {
        let config = HJBConfig {
            use_funding_horizon: true,
            ..Default::default()
        };
        let mut ctrl = HJBInventoryController::new(config);
        ctrl.start_session();

        // Negative time should be clamped to 0
        ctrl.update_funding_settlement(-100.0);
        assert_eq!(ctrl.time_to_funding_settlement_s, Some(0.0));

        // Over 8h should be clamped to 28800
        ctrl.update_funding_settlement(50000.0);
        assert_eq!(ctrl.time_to_funding_settlement_s, Some(28800.0));
    }

    #[test]
    fn test_funding_horizon_summary_fields() {
        let config = HJBConfig {
            use_funding_horizon: true,
            calibrate_terminal_penalty: true,
            ..Default::default()
        };
        let mut ctrl = HJBInventoryController::new(config);
        ctrl.start_session();
        ctrl.update_sigma(0.0001);
        ctrl.update_funding_settlement(3600.0);
        ctrl.calibrate_terminal_penalty(15.0);

        let summary = ctrl.summary();

        assert!(
            summary.funding_horizon_active,
            "Summary should report funding horizon active"
        );
        assert!(
            summary.effective_terminal_penalty > 0.0,
            "Summary should include effective terminal penalty: {}",
            summary.effective_terminal_penalty,
        );
    }
}
