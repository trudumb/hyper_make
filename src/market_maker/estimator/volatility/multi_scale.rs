//! Multi-timescale bipower estimator.
//!
//! Combines fast/medium/slow volatility components for adaptive estimation.

use super::bipower::SingleScaleBipower;
use crate::market_maker::estimator::volume::VolumeBucket;
use crate::market_maker::estimator::EstimatorConfig;

/// Multi-timescale volatility with fast/medium/slow components.
///
/// Fast (~2s): Reacts quickly to crashes, used for early warning
/// Medium (~10s): Balanced responsiveness
/// Slow (~60s): Stable baseline for pricing
#[derive(Debug)]
pub(crate) struct MultiScaleBipowerEstimator {
    fast: SingleScaleBipower,   // ~5 ticks / ~2 seconds
    medium: SingleScaleBipower, // ~20 ticks / ~10 seconds
    slow: SingleScaleBipower,   // ~100 ticks / ~60 seconds
    last_vwap: Option<f64>,
    tick_count: usize,
}

impl MultiScaleBipowerEstimator {
    pub(crate) fn new(config: &EstimatorConfig) -> Self {
        let default_var = config.default_sigma.powi(2);
        Self {
            fast: SingleScaleBipower::new(config.fast_half_life_ticks, default_var),
            medium: SingleScaleBipower::new(config.medium_half_life_ticks, default_var),
            slow: SingleScaleBipower::new(config.slow_half_life_ticks, default_var),
            last_vwap: None,
            tick_count: 0,
        }
    }

    /// Process a completed volume bucket.
    pub(crate) fn on_bucket(&mut self, bucket: &VolumeBucket) {
        if let Some(prev_vwap) = self.last_vwap {
            if bucket.vwap > 0.0 && prev_vwap > 0.0 {
                let log_return = (bucket.vwap / prev_vwap).ln();
                self.fast.update(log_return);
                self.medium.update(log_return);
                self.slow.update(log_return);
                self.tick_count += 1;
            }
        }
        self.last_vwap = Some(bucket.vwap);
    }

    /// Get the log return for the most recent bucket (for momentum tracking)
    pub(crate) fn last_log_return(&self, bucket: &VolumeBucket) -> Option<f64> {
        self.last_vwap.and_then(|prev| {
            if bucket.vwap > 0.0 && prev > 0.0 {
                Some((bucket.vwap / prev).ln())
            } else {
                None
            }
        })
    }

    /// Clean sigma (BV-based) for spread pricing.
    /// Uses slow timescale for stability.
    pub(crate) fn sigma_clean(&self) -> f64 {
        self.slow.sigma_clean()
    }

    /// Total sigma (RV-based) for risk assessment.
    /// Blends fast + slow: uses fast when market is accelerating.
    pub(crate) fn sigma_total(&self) -> f64 {
        let fast = self.fast.sigma_total();
        let slow = self.slow.sigma_total();

        // If fast >> slow, market is accelerating - trust fast more
        let ratio = fast / slow.max(1e-9);

        if ratio > 1.5 {
            // Acceleration: blend toward fast
            let weight = ((ratio - 1.0) / 3.0).clamp(0.0, 0.7);
            weight * fast + (1.0 - weight) * slow
        } else {
            // Stable: prefer slow for less noise
            0.2 * fast + 0.8 * slow
        }
    }

    /// Effective sigma for inventory skew.
    /// Blends clean and total based on jump regime.
    pub(crate) fn sigma_effective(&self) -> f64 {
        let clean = self.sigma_clean();
        let total = self.sigma_total();
        let jump_ratio = self.jump_ratio_fast();

        // At ratio=1: pure clean (no jumps)
        // At ratio=3: 67% total (jumps dominant)
        // At ratio=5: 80% total
        let jump_weight = 1.0 - (1.0 / jump_ratio.max(1.0));
        let jump_weight = jump_weight.clamp(0.0, 0.85);

        (1.0 - jump_weight) * clean + jump_weight * total
    }

    /// Fast jump ratio (detects recent jumps quickly)
    pub(crate) fn jump_ratio_fast(&self) -> f64 {
        self.fast.jump_ratio()
    }

    /// Medium jump ratio (more stable signal)
    #[allow(dead_code)]
    pub(crate) fn jump_ratio_medium(&self) -> f64 {
        self.medium.jump_ratio()
    }

    pub(crate) fn tick_count(&self) -> usize {
        self.tick_count
    }
}
