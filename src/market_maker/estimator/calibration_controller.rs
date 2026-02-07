//! Calibration-aware fill rate controller.
//!
//! During warmup, stochastic features (adverse selection, kappa, fill probability)
//! require fills to calibrate. This controller adjusts gamma to target a minimum
//! fill rate, trading some expected profit for calibration data.
//!
//! Once calibration is complete, the controller phases out and GLFT operates normally.
//!
//! # Bootstrap Problem
//!
//! The market maker has a calibration bootstrap problem:
//! - Stochastic features require fills to calibrate (AS needs 20+, kappa needs confidence)
//! - GLFT quotes "competitively" (tight spreads) with uncalibrated defaults
//! - In illiquid markets, competitive quotes may never fill
//! - Without fills, parameters never calibrate → perpetual uncalibrated state
//!
//! # Solution
//!
//! Target a minimum fill rate. If actual rate < target, reduce gamma (tighter quotes)
//! to attract fills. Phase out as calibration completes.
//!
//! This is classic exploration vs exploitation - we trade expected profit for
//! calibration data during bootstrap.

use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tracing::debug;

/// Tracks fill rate and adjusts gamma to ensure sufficient fills for calibration.
#[derive(Debug)]
pub struct CalibrationController {
    /// Timestamps of recent fills (for rate calculation)
    fill_timestamps: VecDeque<Instant>,

    /// Target fill rate (fills per hour across all levels)
    target_fill_rate_per_hour: f64,

    /// Minimum gamma multiplier (maximum tightening)
    min_gamma_mult: f64,

    /// Lookback window for fill rate calculation
    lookback_secs: u64,

    /// Current calibration status from external estimators
    as_fills_measured: u64,
    as_warmup_threshold: u64,
    kappa_confidence: f64,
    kappa_confidence_threshold: f64,

    /// Cached output
    fill_hungry_gamma_mult: f64,
    calibration_progress: f64,

    /// Whether the controller is enabled
    enabled: bool,
}

/// Configuration for the calibration controller.
#[derive(Debug, Clone)]
pub struct CalibrationControllerConfig {
    /// Enable fill-rate targeting during warmup
    pub enabled: bool,
    /// Target fills per hour (across all levels)
    pub target_fill_rate_per_hour: f64,
    /// Minimum gamma multiplier (0.3 = allow 70% gamma reduction)
    pub min_gamma_mult: f64,
    /// Lookback window in seconds for rate calculation
    pub lookback_secs: u64,
    /// AS warmup threshold (fills needed)
    pub as_warmup_threshold: u64,
    /// Kappa confidence threshold to consider calibrated
    pub kappa_confidence_threshold: f64,
}

impl Default for CalibrationControllerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            // MAINNET OPTIMIZED: Higher fill rate target for liquid markets
            target_fill_rate_per_hour: 60.0, // Increased from 10 - liquid markets fill faster
            // MAINNET OPTIMIZED: Don't sacrifice edge for fills - fills are abundant
            min_gamma_mult: 0.6, // Increased from 0.3 - less aggressive tightening
            // MAINNET OPTIMIZED: Shorter lookback - market regimes change faster
            lookback_secs: 900,      // Reduced from 3600 - 15 minute window
            as_warmup_threshold: 20, // Match AS estimator requirement
            kappa_confidence_threshold: 0.5,
        }
    }
}

impl CalibrationController {
    /// Create a new calibration controller with the given configuration.
    pub fn new(config: CalibrationControllerConfig) -> Self {
        Self {
            fill_timestamps: VecDeque::with_capacity(100),
            target_fill_rate_per_hour: config.target_fill_rate_per_hour,
            min_gamma_mult: config.min_gamma_mult,
            lookback_secs: config.lookback_secs,
            as_fills_measured: 0,
            as_warmup_threshold: config.as_warmup_threshold,
            kappa_confidence: 0.0,
            kappa_confidence_threshold: config.kappa_confidence_threshold,
            fill_hungry_gamma_mult: config.min_gamma_mult, // Start hungry
            calibration_progress: 0.0,
            enabled: config.enabled,
        }
    }

    /// Record a new fill and update the gamma multiplier.
    pub fn record_fill(&mut self) {
        if !self.enabled {
            return;
        }
        self.fill_timestamps.push_back(Instant::now());
        self.update();
    }

    /// Update calibration status from external estimators.
    pub fn update_calibration_status(&mut self, as_fills_measured: u64, kappa_confidence: f64) {
        if !self.enabled {
            return;
        }
        self.as_fills_measured = as_fills_measured;
        self.kappa_confidence = kappa_confidence;
        self.update();
    }

    /// Recalculate gamma multiplier based on current state.
    fn update(&mut self) {
        if !self.enabled {
            self.fill_hungry_gamma_mult = 1.0;
            self.calibration_progress = 1.0;
            return;
        }

        // Prune old fills
        let cutoff = Instant::now() - Duration::from_secs(self.lookback_secs);
        while self
            .fill_timestamps
            .front()
            .map(|t| *t < cutoff)
            .unwrap_or(false)
        {
            self.fill_timestamps.pop_front();
        }

        // Calculate calibration progress (0.0 to 1.0)
        let as_progress =
            (self.as_fills_measured as f64 / self.as_warmup_threshold as f64).min(1.0);
        let kappa_progress = (self.kappa_confidence / self.kappa_confidence_threshold).min(1.0);
        self.calibration_progress = (as_progress + kappa_progress) / 2.0;

        // If fully calibrated, disable fill-hungry mode
        if self.calibration_progress >= 0.95 {
            self.fill_hungry_gamma_mult = 1.0;
            return;
        }

        // Calculate actual fill count in the lookback window
        let actual_fills = self.fill_timestamps.len() as f64;

        // Calculate expected fills in the lookback window
        // (target_fill_rate_per_hour scaled to the window size)
        let window_hours = self.lookback_secs as f64 / 3600.0;
        let expected_fills_in_window = self.target_fill_rate_per_hour * window_hours;

        // Calculate fill-hungry multiplier
        // When ratio < 1: we need more fills, reduce gamma (tighter quotes)
        let ratio = actual_fills / expected_fills_in_window;
        let raw_mult = ratio.min(1.0);
        let clamped_mult = raw_mult.max(self.min_gamma_mult);

        // Blend with calibration progress: as we calibrate, reduce fill-hunger
        self.fill_hungry_gamma_mult =
            clamped_mult + (1.0 - clamped_mult) * self.calibration_progress;

        if self.fill_hungry_gamma_mult < 0.99 {
            debug!(
                actual_fills = %format!("{:.1}", actual_fills),
                expected_fills = %format!("{:.1}", expected_fills_in_window),
                fill_rate_ratio = %format!("{:.2}", ratio),
                calibration_progress = %format!("{:.0}%", self.calibration_progress * 100.0),
                fill_hungry_mult = %format!("{:.2}", self.fill_hungry_gamma_mult),
                as_fills = self.as_fills_measured,
                kappa_conf = %format!("{:.2}", self.kappa_confidence),
                "Fill-hungry mode active (calibration incomplete)"
            );
        }
    }

    /// Get the current gamma multiplier (0.3 to 1.0).
    /// Lower values = tighter quotes to attract fills.
    pub fn gamma_multiplier(&self) -> f64 {
        if !self.enabled {
            return 1.0;
        }
        self.fill_hungry_gamma_mult
    }

    /// Get calibration progress (0.0 to 1.0).
    pub fn calibration_progress(&self) -> f64 {
        self.calibration_progress
    }

    /// Check if calibration is complete.
    pub fn is_calibrated(&self) -> bool {
        !self.enabled || self.calibration_progress >= 0.95
    }

    /// Get fill count in lookback window.
    pub fn fill_count(&self) -> usize {
        self.fill_timestamps.len()
    }

    /// Check if the controller is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_controller_starts_hungry() {
        let config = CalibrationControllerConfig::default();
        let controller = CalibrationController::new(config);

        // Should start with min gamma (most aggressive, now 0.6 for mainnet)
        assert!((controller.gamma_multiplier() - 0.6).abs() < 0.01);
        assert!(!controller.is_calibrated());
        assert_eq!(controller.fill_count(), 0);
    }

    #[test]
    fn test_calibration_controller_disabled() {
        let config = CalibrationControllerConfig {
            enabled: false,
            ..Default::default()
        };
        let controller = CalibrationController::new(config);

        // Should return 1.0 when disabled
        assert!((controller.gamma_multiplier() - 1.0).abs() < 0.01);
        assert!(controller.is_calibrated());
    }

    #[test]
    fn test_calibration_progress_phases_out() {
        let config = CalibrationControllerConfig::default();
        let mut controller = CalibrationController::new(config);

        // At 0% calibration, should be at min gamma (0.6 for mainnet)
        assert!(controller.gamma_multiplier() < 0.7);

        // At 50% calibration (10 AS fills, 0.25 kappa conf)
        controller.update_calibration_status(10, 0.25);
        let mid_mult = controller.gamma_multiplier();

        // Should be between min (0.6) and 1.0
        assert!(mid_mult > 0.6);
        assert!(mid_mult < 1.0);

        // At 100% calibration (20 AS fills, 0.5 kappa conf)
        controller.update_calibration_status(20, 0.5);
        assert!((controller.gamma_multiplier() - 1.0).abs() < 0.01);
        assert!(controller.is_calibrated());
    }

    #[test]
    fn test_fill_recording() {
        let config = CalibrationControllerConfig::default();
        let mut controller = CalibrationController::new(config);

        // Record some fills
        for _ in 0..5 {
            controller.record_fill();
        }

        assert_eq!(controller.fill_count(), 5);

        // With 5 fills (half target), multiplier should increase
        // ratio = 5/10 = 0.5, clamped to 0.5, blended with 0% progress = 0.5
        let mult = controller.gamma_multiplier();
        assert!(mult >= 0.3);
        assert!(mult <= 0.6);
    }

    #[test]
    fn test_multiplier_clamped_to_min() {
        let config = CalibrationControllerConfig {
            min_gamma_mult: 0.4,
            ..Default::default()
        };
        let controller = CalibrationController::new(config);

        // With 0 fills and 0 calibration, should be at min
        assert!((controller.gamma_multiplier() - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_fill_rate_exceeds_target() {
        let config = CalibrationControllerConfig::default();
        let mut controller = CalibrationController::new(config);

        // Record more fills than target (90 fills when target is 60/hour)
        // With 900 second lookback, 90 fills = 360/hour, well above 60
        for _ in 0..90 {
            controller.record_fill();
        }

        // At 0% calibration progress, ratio = 90/(60*900/3600) = 90/15 = 6.0, clamped to 1.0
        // So fill_hungry_mult should be 1.0 (no aggressive tightening needed)
        let mult = controller.gamma_multiplier();
        // With 0% calibration progress: blended = 1.0 + (1-1.0)*0 = 1.0
        assert!(
            (mult - 1.0).abs() < 0.01,
            "When fill rate exceeds target, gamma_mult should be 1.0, got {}",
            mult
        );
    }

    #[test]
    fn test_calibration_near_complete_threshold() {
        let config = CalibrationControllerConfig::default();
        let mut controller = CalibrationController::new(config);

        // At 94% calibration (just below 95% threshold), should still be hungry
        // AS: 18/20 = 0.90, kappa: 0.49/0.5 = 0.98, avg = 0.94
        controller.update_calibration_status(18, 0.49);

        assert!(
            !controller.is_calibrated(),
            "Should not be calibrated at 94%"
        );
        assert!(
            controller.gamma_multiplier() < 1.0,
            "Should still have fill-hungry adjustment below 95%"
        );

        // At 95%+, should be fully calibrated
        controller.update_calibration_status(20, 0.5);
        assert!(controller.is_calibrated(), "Should be calibrated at 100%");
        assert!(
            (controller.gamma_multiplier() - 1.0).abs() < 0.01,
            "Gamma multiplier should be 1.0 when calibrated"
        );
    }

    #[test]
    fn test_blending_formula_correctness() {
        // Test that the blending formula works as documented:
        // blended_mult = fill_hungry_mult + (1 - fill_hungry_mult) × calibration_progress
        let config = CalibrationControllerConfig {
            min_gamma_mult: 0.3,
            target_fill_rate_per_hour: 10.0,
            ..Default::default()
        };
        let mut controller = CalibrationController::new(config);

        // With 0 fills and 50% calibration progress:
        // ratio = 0/10 = 0, raw_mult = 0, clamped = 0.3
        // blended = 0.3 + (1 - 0.3) * 0.5 = 0.3 + 0.35 = 0.65
        controller.update_calibration_status(10, 0.25); // 50% progress

        let mult = controller.gamma_multiplier();
        let expected = 0.3 + (1.0 - 0.3) * 0.5; // = 0.65
        assert!(
            (mult - expected).abs() < 0.05,
            "Blending formula: expected {:.2}, got {:.2}",
            expected,
            mult
        );
    }

    #[test]
    fn test_partial_fill_rate_contribution() {
        let config = CalibrationControllerConfig::default();
        let mut controller = CalibrationController::new(config);

        // Record exactly target fills for the lookback window
        // target = 60/hour, lookback = 900s, so target in window = 60 * 900/3600 = 15
        for _ in 0..15 {
            controller.record_fill();
        }

        // With target fill rate met and 0% calibration:
        // ratio = 15/15 = 1.0, raw_mult = 1.0
        // blended = 1.0 + (1-1.0) * 0 = 1.0
        let mult = controller.gamma_multiplier();
        assert!(
            (mult - 1.0).abs() < 0.01,
            "Meeting target fill rate should yield gamma_mult ~1.0, got {}",
            mult
        );
    }
}
