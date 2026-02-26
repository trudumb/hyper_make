//! Self-Impact Estimator
//!
//! Models our own market impact by tracking the fraction of visible book depth
//! occupied by our resting orders. On thin HIP-3 books, our flow can be 40%+
//! of visible depth, causing adverse fills against ourselves.
//!
//! Impact model: `addon_bps = coefficient × our_fraction²`
//! - Square-root impact law (squared for maker side): large fractions → disproportionate impact
//! - EWMA smoothing of `our_fraction` for stability
//! - Applied as additive spread widening in the quoting pipeline

use serde::{Deserialize, Serialize};

/// Configuration for the self-impact estimator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfImpactConfig {
    /// Whether self-impact estimation is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Impact coefficient: addon_bps = coefficient × fraction².
    /// At coefficient=5.0: 40% dominance → 0.8 bps, 80% → 3.2 bps.
    #[serde(default = "default_coefficient")]
    pub coefficient: f64,
    /// EWMA half-life in seconds for smoothing our_fraction.
    #[serde(default = "default_ewma_half_life_s")]
    pub ewma_half_life_s: f64,
}

fn default_coefficient() -> f64 {
    5.0
}
fn default_ewma_half_life_s() -> f64 {
    30.0
}

impl Default for SelfImpactConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            coefficient: default_coefficient(),
            ewma_half_life_s: default_ewma_half_life_s(),
        }
    }
}

/// Tracks our book dominance and computes impact addon.
#[derive(Debug, Clone)]
pub struct SelfImpactEstimator {
    config: SelfImpactConfig,
    /// EWMA of our fraction on bid side [0, 1].
    bid_fraction_ewma: f64,
    /// EWMA of our fraction on ask side [0, 1].
    ask_fraction_ewma: f64,
    /// EWMA decay factor per update (computed from half-life).
    alpha: f64,
    /// Whether we've received at least one observation.
    initialized: bool,
}

impl SelfImpactEstimator {
    pub fn new(config: SelfImpactConfig) -> Self {
        // alpha = 1 - exp(-ln2 / half_life_updates)
        // We update once per cycle (~1s), so half_life_updates ≈ half_life_s
        let half_life = config.ewma_half_life_s.max(1.0);
        let alpha = 1.0 - (-core::f64::consts::LN_2 / half_life).exp();
        Self {
            config,
            bid_fraction_ewma: 0.0,
            ask_fraction_ewma: 0.0,
            alpha,
            initialized: false,
        }
    }

    /// Update with current book state.
    ///
    /// # Arguments
    /// * `our_bid_size` - Total resting bid size from our orders
    /// * `other_bid_depth` - Total visible bid depth from other participants
    /// * `our_ask_size` - Total resting ask size from our orders
    /// * `other_ask_depth` - Total visible ask depth from other participants
    pub fn update(
        &mut self,
        our_bid_size: f64,
        other_bid_depth: f64,
        our_ask_size: f64,
        other_ask_depth: f64,
    ) {
        if !self.config.enabled {
            return;
        }

        let total_bid = our_bid_size + other_bid_depth;
        let total_ask = our_ask_size + other_ask_depth;

        let bid_fraction = if total_bid > 0.0 {
            (our_bid_size / total_bid).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let ask_fraction = if total_ask > 0.0 {
            (our_ask_size / total_ask).clamp(0.0, 1.0)
        } else {
            0.0
        };

        if !self.initialized {
            self.bid_fraction_ewma = bid_fraction;
            self.ask_fraction_ewma = ask_fraction;
            self.initialized = true;
        } else {
            self.bid_fraction_ewma += self.alpha * (bid_fraction - self.bid_fraction_ewma);
            self.ask_fraction_ewma += self.alpha * (ask_fraction - self.ask_fraction_ewma);
        }
    }

    /// Compute the self-impact addon in basis points.
    ///
    /// Uses the average of bid and ask fractions for a symmetric addon.
    /// Returns 0.0 if disabled or not yet initialized.
    pub fn impact_addon_bps(&self) -> f64 {
        if !self.config.enabled || !self.initialized {
            return 0.0;
        }
        let avg_fraction = (self.bid_fraction_ewma + self.ask_fraction_ewma) / 2.0;
        self.config.coefficient * avg_fraction * avg_fraction
    }

    /// Compute per-side impact addons in basis points.
    ///
    /// Returns (bid_addon_bps, ask_addon_bps) for asymmetric application.
    pub fn per_side_impact_bps(&self) -> (f64, f64) {
        if !self.config.enabled || !self.initialized {
            return (0.0, 0.0);
        }
        let bid_addon = self.config.coefficient * self.bid_fraction_ewma * self.bid_fraction_ewma;
        let ask_addon = self.config.coefficient * self.ask_fraction_ewma * self.ask_fraction_ewma;
        (bid_addon, ask_addon)
    }

    /// Current bid-side book fraction (EWMA-smoothed).
    pub fn bid_fraction(&self) -> f64 {
        self.bid_fraction_ewma
    }

    /// Current ask-side book fraction (EWMA-smoothed).
    pub fn ask_fraction(&self) -> f64 {
        self.ask_fraction_ewma
    }

    /// Whether the estimator is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_fraction_zero_impact() {
        let config = SelfImpactConfig {
            enabled: true,
            coefficient: 5.0,
            ewma_half_life_s: 30.0,
        };
        let mut est = SelfImpactEstimator::new(config);
        est.update(0.0, 100.0, 0.0, 100.0);
        assert!((est.impact_addon_bps() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_40pct_dominance() {
        let config = SelfImpactConfig {
            enabled: true,
            coefficient: 5.0,
            ewma_half_life_s: 30.0,
        };
        let mut est = SelfImpactEstimator::new(config);
        // 40% of book on both sides
        est.update(40.0, 60.0, 40.0, 60.0);
        // addon = 5.0 × 0.4² = 0.8 bps
        assert!((est.impact_addon_bps() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_80pct_dominance() {
        let config = SelfImpactConfig {
            enabled: true,
            coefficient: 5.0,
            ewma_half_life_s: 30.0,
        };
        let mut est = SelfImpactEstimator::new(config);
        // 80% of book on both sides
        est.update(80.0, 20.0, 80.0, 20.0);
        // addon = 5.0 × 0.8² = 3.2 bps
        assert!((est.impact_addon_bps() - 3.2).abs() < 1e-10);
    }

    #[test]
    fn test_disabled_returns_zero() {
        let config = SelfImpactConfig {
            enabled: false,
            coefficient: 5.0,
            ewma_half_life_s: 30.0,
        };
        let mut est = SelfImpactEstimator::new(config);
        est.update(80.0, 20.0, 80.0, 20.0);
        assert_eq!(est.impact_addon_bps(), 0.0);
    }

    #[test]
    fn test_ewma_smoothing() {
        let config = SelfImpactConfig {
            enabled: true,
            coefficient: 5.0,
            ewma_half_life_s: 1.0, // Fast decay for testing
        };
        let mut est = SelfImpactEstimator::new(config);

        // First observation: 80% dominance
        est.update(80.0, 20.0, 80.0, 20.0);
        let impact_1 = est.impact_addon_bps();
        assert!((impact_1 - 3.2).abs() < 1e-10); // First obs = exact

        // Second observation: 0% dominance — EWMA should decay
        est.update(0.0, 100.0, 0.0, 100.0);
        let impact_2 = est.impact_addon_bps();
        assert!(impact_2 < impact_1); // Should decay
        assert!(impact_2 > 0.0); // But not to zero yet
    }

    #[test]
    fn test_per_side_asymmetric() {
        let config = SelfImpactConfig {
            enabled: true,
            coefficient: 5.0,
            ewma_half_life_s: 30.0,
        };
        let mut est = SelfImpactEstimator::new(config);
        // 80% bid, 20% ask
        est.update(80.0, 20.0, 20.0, 80.0);
        let (bid_addon, ask_addon) = est.per_side_impact_bps();
        // bid: 5.0 × 0.8² = 3.2
        assert!((bid_addon - 3.2).abs() < 1e-10);
        // ask: 5.0 × 0.2² = 0.2
        assert!((ask_addon - 0.2).abs() < 1e-10);
    }
}
