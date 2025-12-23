//! Weighted kappa estimator for order book depth decay.
//!
//! Estimates κ (kappa) from L2 order book data using weighted linear regression.

use tracing::debug;

/// Weighted linear regression kappa estimator.
///
/// Improvements over simple kappa:
/// - Truncates to orders within max_distance of mid (ignores fake far orders)
/// - Uses first N levels only (focuses on relevant liquidity)
/// - Weights by proximity to mid (closer levels matter more)
#[derive(Debug)]
pub(crate) struct WeightedKappaEstimator {
    alpha: f64,
    kappa: f64,
    max_distance: f64,
    max_levels: usize,
    update_count: usize,
}

impl WeightedKappaEstimator {
    pub(crate) fn new(
        half_life_updates: f64,
        default_kappa: f64,
        max_distance: f64,
        max_levels: usize,
    ) -> Self {
        Self {
            alpha: (2.0_f64.ln() / half_life_updates).clamp(0.001, 1.0),
            kappa: default_kappa,
            max_distance,
            max_levels,
            update_count: 0,
        }
    }

    /// Update kappa from L2 order book.
    ///
    /// Uses instantaneous depth (size at each level) to fit exponential decay:
    /// L(δ) = A × exp(-κ × δ)  =>  ln(L) = ln(A) - κ × δ
    pub(crate) fn update(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)], mid: f64) {
        if mid <= 0.0 {
            return;
        }

        // Collect points: (distance, size_at_level, weight)
        let mut points: Vec<(f64, f64, f64)> = Vec::new();

        // Process bids (truncate and limit levels)
        for (i, (price, size)) in bids.iter().enumerate() {
            if i >= self.max_levels || *price <= 0.0 || *size <= 0.0 {
                break;
            }
            let distance = (mid - price) / mid;
            if distance > self.max_distance {
                break; // Too far from mid
            }
            // Weight by proximity: closer to mid = higher weight
            let weight = 1.0 / (1.0 + distance * 100.0);
            if distance > 1e-6 {
                points.push((distance, *size, weight));
            }
        }

        // Process asks (same logic)
        for (i, (price, size)) in asks.iter().enumerate() {
            if i >= self.max_levels || *price <= 0.0 || *size <= 0.0 {
                break;
            }
            let distance = (price - mid) / mid;
            if distance > self.max_distance {
                break;
            }
            let weight = 1.0 / (1.0 + distance * 100.0);
            if distance > 1e-6 {
                points.push((distance, *size, weight));
            }
        }

        // Need at least 4 points for meaningful regression
        if points.len() < 4 {
            return;
        }

        // Weighted linear regression on (distance, ln(size))
        // Model: ln(size) = ln(A) - κ × distance
        // Slope = -κ, so κ = -slope
        if let Some(slope) = weighted_linear_regression_slope(&points) {
            // In real order books, liquidity often INCREASES slightly with distance
            // (more limit orders stacked further from mid), giving positive slope.
            // Use absolute value and a reasonable default if slope is wrong sign.
            let kappa_estimated = if slope < 0.0 {
                // Negative slope = liquidity decays with distance (expected in theory)
                (-slope).clamp(1.0, 10000.0)
            } else {
                // Positive slope = liquidity increases with distance (common in practice)
                // Use a moderate default based on typical market structure
                50.0
            };

            // EWMA update
            self.kappa = self.alpha * kappa_estimated + (1.0 - self.alpha) * self.kappa;
            self.update_count += 1;

            debug!(
                points = points.len(),
                slope = %format!("{:.4}", slope),
                kappa_new = %format!("{:.2}", kappa_estimated),
                kappa_ewma = %format!("{:.2}", self.kappa),
                "Kappa updated from L2 book"
            );
        }
    }

    pub(crate) fn kappa(&self) -> f64 {
        self.kappa.clamp(1.0, 10000.0)
    }

    pub(crate) fn update_count(&self) -> usize {
        self.update_count
    }
}

/// Weighted linear regression to get slope.
/// Points are (x, y, weight). Fits ln(y) ~ a + b*x, returns b.
fn weighted_linear_regression_slope(points: &[(f64, f64, f64)]) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }

    let mut sum_w = 0.0;
    let mut sum_wx = 0.0;
    let mut sum_wy = 0.0;
    let mut sum_wxx = 0.0;
    let mut sum_wxy = 0.0;

    for (x, y, w) in points {
        if *y <= 0.0 {
            continue;
        }
        let ln_y = y.ln();
        sum_w += w;
        sum_wx += w * x;
        sum_wy += w * ln_y;
        sum_wxx += w * x * x;
        sum_wxy += w * x * ln_y;
    }

    let denominator = sum_w * sum_wxx - sum_wx * sum_wx;
    if denominator.abs() < 1e-12 {
        return None;
    }

    Some((sum_w * sum_wxy - sum_wx * sum_wy) / denominator)
}
