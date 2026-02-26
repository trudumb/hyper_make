//! Book-structure kappa estimator using L2 depth decay.
//!
//! This module provides a kappa (κ) estimator based on the structure of the L2 order book.
//! The key insight is that in many markets, the cumulative depth follows an exponential decay:
//!
//! ```text
//! Depth(δ) = D₀ × exp(-κ × δ)
//! ```
//!
//! Where:
//! - δ = distance from mid price (as fraction)
//! - D₀ = depth at the touch
//! - κ = decay parameter (what we estimate)
//!
//! This approach is semantically correct for GLFT because:
//! - GLFT models fill probability as λ(δ) = A × exp(-κδ)
//! - Book depth decay directly reflects where liquidity sits
//! - Unlike market trade distances, book structure is NOT affected by outlier liquidations

use std::collections::VecDeque;
use tracing::debug;

/// Minimum number of levels for a valid regression
const MIN_LEVELS: usize = 3;

/// Maximum κ to prevent instability (corresponds to very tight books)
const MAX_KAPPA: f64 = 10000.0;

/// Minimum κ to prevent instability (corresponds to very sparse books)
const MIN_KAPPA: f64 = 50.0;

/// Book-structure kappa estimator using L2 depth decay.
///
/// Fits exponential decay model: Depth(δ) = D₀ × exp(-κδ)
/// via log-linear regression on cumulative depth at each price level.
///
/// # Example
///
/// ```ignore
/// let mut estimator = BookKappaEstimator::new(2000.0);
/// estimator.on_l2_book(&bids, &asks, mid);
/// let kappa = estimator.kappa();
/// let confidence = estimator.confidence();
/// ```
#[derive(Debug, Clone)]
pub(crate) struct BookKappaEstimator {
    /// Current κ estimate from book structure
    kappa: f64,

    /// R² of log-linear fit (confidence measure)
    r_squared: f64,

    /// EWMA smoothing factor (0 = no smoothing, 1 = full smoothing)
    alpha: f64,

    /// Minimum R² to trust estimate (below this, use prior)
    min_r_squared: f64,

    /// Prior κ when fit is poor or no data
    prior_kappa: f64,

    /// Number of valid updates received
    update_count: u64,

    /// Historical R² values for stability tracking
    r_squared_history: VecDeque<f64>,

    /// Maximum history length
    max_history: usize,

    /// Bid-side κ estimate (for diagnostics)
    kappa_bid: f64,

    /// Ask-side κ estimate (for diagnostics)
    kappa_ask: f64,
}

impl BookKappaEstimator {
    /// Create a new book-structure kappa estimator.
    ///
    /// # Arguments
    /// * `prior_kappa` - Default κ value when fit is poor (e.g., 2000 for liquid markets)
    pub(crate) fn new(prior_kappa: f64) -> Self {
        Self {
            kappa: prior_kappa,
            r_squared: 0.0,
            alpha: 0.2, // 20% weight on new observation
            min_r_squared: 0.3,
            prior_kappa,
            update_count: 0,
            r_squared_history: VecDeque::with_capacity(20),
            max_history: 20,
            kappa_bid: prior_kappa,
            kappa_ask: prior_kappa,
        }
    }

    /// Update from L2 book snapshot.
    ///
    /// # Arguments
    /// * `bids` - Bid levels as (price, size) tuples, best (highest) first
    /// * `asks` - Ask levels as (price, size) tuples, best (lowest) first
    /// * `mid` - Current mid price
    pub(crate) fn on_l2_book(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)], mid: f64) {
        if mid <= 0.0 {
            return;
        }

        // Fit exponential decay on each side
        let (kappa_bid, r2_bid) = self.fit_exponential_decay(bids, mid, true);
        let (kappa_ask, r2_ask) = self.fit_exponential_decay(asks, mid, false);

        // Average the two sides (weighted by R²)
        let total_r2 = r2_bid + r2_ask;
        let (new_kappa, new_r2) = if total_r2 > 0.0 {
            let w_bid = r2_bid / total_r2;
            let w_ask = r2_ask / total_r2;
            let blended_kappa = w_bid * kappa_bid + w_ask * kappa_ask;
            let blended_r2 = (r2_bid + r2_ask) / 2.0;
            (blended_kappa, blended_r2)
        } else {
            (self.prior_kappa, 0.0)
        };

        // Store side-specific values for diagnostics
        self.kappa_bid = kappa_bid;
        self.kappa_ask = kappa_ask;

        // EWMA smooth the estimate
        if self.update_count == 0 {
            self.kappa = new_kappa;
            self.r_squared = new_r2;
        } else {
            self.kappa = self.alpha * new_kappa + (1.0 - self.alpha) * self.kappa;
            self.r_squared = self.alpha * new_r2 + (1.0 - self.alpha) * self.r_squared;
        }

        // Clamp to reasonable bounds
        self.kappa = self.kappa.clamp(MIN_KAPPA, MAX_KAPPA);

        // Track R² history
        if self.r_squared_history.len() >= self.max_history {
            self.r_squared_history.pop_front();
        }
        self.r_squared_history.push_back(new_r2);

        self.update_count += 1;

        // Log periodically
        if self.update_count.is_multiple_of(100) {
            debug!(
                kappa = %format!("{:.0}", self.kappa),
                r_squared = %format!("{:.2}", self.r_squared),
                kappa_bid = %format!("{:.0}", self.kappa_bid),
                kappa_ask = %format!("{:.0}", self.kappa_ask),
                updates = self.update_count,
                "Book kappa updated"
            );
        }
    }

    /// Fit exponential decay model to one side of the book.
    ///
    /// Uses log-linear regression: log(depth) = log(D₀) - κ × distance
    /// where depth is the individual level size (not cumulative).
    ///
    /// # Returns
    /// (κ, R²) tuple
    fn fit_exponential_decay(&self, levels: &[(f64, f64)], mid: f64, is_bid: bool) -> (f64, f64) {
        if levels.len() < MIN_LEVELS {
            return (self.prior_kappa, 0.0);
        }

        // Use individual level depth (not cumulative) - this matches the exponential decay model
        // where marginal liquidity decreases as you move away from mid
        let mut points: Vec<(f64, f64)> = Vec::with_capacity(levels.len());

        for (price, size) in levels {
            if *size <= 0.0 || *price <= 0.0 {
                continue;
            }

            // Distance from mid as fraction
            let distance = if is_bid {
                (mid - price) / mid
            } else {
                (price - mid) / mid
            };

            // Only include positive distances with positive depth
            if distance > 0.0 && *size > 0.0 {
                points.push((distance, size.ln()));
            }
        }

        if points.len() < MIN_LEVELS {
            return (self.prior_kappa, 0.0);
        }

        // Simple linear regression: y = a + b*x
        // Where y = ln(cum_depth), x = distance, b = -κ
        let n = points.len() as f64;
        let sum_x: f64 = points.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = points.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = points.iter().map(|(x, _)| x * x).sum();
        let sum_y2: f64 = points.iter().map(|(_, y)| y * y).sum();

        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return (self.prior_kappa, 0.0);
        }

        // Slope b = -κ
        let b = (n * sum_xy - sum_x * sum_y) / denom;
        let kappa = -b; // κ should be positive (decay)

        // R² calculation
        let mean_y = sum_y / n;
        let ss_tot = sum_y2 - n * mean_y * mean_y;
        let ss_res: f64 = points
            .iter()
            .map(|(x, y)| {
                let a = (sum_y - b * sum_x) / n;
                let y_pred = a + b * x;
                (y - y_pred).powi(2)
            })
            .sum();

        let r_squared = if ss_tot > 1e-10 {
            (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Clamp κ to reasonable bounds
        let kappa = kappa.clamp(MIN_KAPPA, MAX_KAPPA);

        (kappa, r_squared)
    }

    /// Get current κ estimate.
    ///
    /// Returns the EWMA-smoothed κ if fit quality is good,
    /// otherwise returns the prior.
    pub(crate) fn kappa(&self) -> f64 {
        if self.r_squared >= self.min_r_squared && self.update_count > 5 {
            self.kappa
        } else {
            self.prior_kappa
        }
    }

    /// Get raw κ estimate (without prior fallback).
    #[allow(dead_code)] // API completeness for diagnostics
    pub(crate) fn kappa_raw(&self) -> f64 {
        self.kappa
    }

    /// Get confidence in the estimate (R² of exponential fit).
    ///
    /// Returns a value in [0, 1] where:
    /// - 0.0 = no confidence (use prior)
    /// - 1.0 = perfect exponential decay (high confidence)
    pub(crate) fn confidence(&self) -> f64 {
        if self.update_count < 5 {
            return 0.0;
        }

        // Use average R² from recent history for stability
        if self.r_squared_history.is_empty() {
            return self.r_squared.clamp(0.0, 1.0);
        }

        let avg_r2: f64 =
            self.r_squared_history.iter().sum::<f64>() / self.r_squared_history.len() as f64;
        avg_r2.clamp(0.0, 1.0)
    }

    /// Get bid-side κ estimate (for diagnostics).
    #[allow(dead_code)] // API completeness for diagnostics
    pub(crate) fn kappa_bid(&self) -> f64 {
        self.kappa_bid
    }

    /// Get ask-side κ estimate (for diagnostics).
    #[allow(dead_code)] // API completeness for diagnostics
    pub(crate) fn kappa_ask(&self) -> f64 {
        self.kappa_ask
    }

    /// Get number of updates received.
    #[allow(dead_code)] // API completeness for diagnostics
    pub(crate) fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Check if estimator has sufficient data.
    pub(crate) fn is_warmed_up(&self) -> bool {
        self.update_count >= 10 && self.r_squared >= self.min_r_squared
    }
}

impl Default for BookKappaEstimator {
    fn default() -> Self {
        Self::new(2000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_exponential_decay() {
        let mut estimator = BookKappaEstimator::new(2000.0);

        // Create perfect exponential decay with κ = 1000
        // Individual level depth: Depth(δ) = 100 × exp(-1000 × δ)
        let kappa_true = 1000.0;
        let mid = 100.0;

        let mut bids = Vec::new();
        let mut asks = Vec::new();

        for i in 1..=10 {
            let delta = i as f64 * 0.001; // 0.1% steps
                                          // Each level has exponentially decaying depth
            let level_depth = 100.0 * (-kappa_true * delta).exp();

            let bid_price = mid * (1.0 - delta);
            let ask_price = mid * (1.0 + delta);

            bids.push((bid_price, level_depth.max(0.1)));
            asks.push((ask_price, level_depth.max(0.1)));
        }

        // Run multiple updates for stability
        for _ in 0..20 {
            estimator.on_l2_book(&bids, &asks, mid);
        }

        // Should recover approximately the true κ
        let estimated_kappa = estimator.kappa();
        let r_squared = estimator.confidence();

        // Allow 30% error due to numerical issues
        assert!(
            (estimated_kappa - kappa_true).abs() / kappa_true < 0.3,
            "Expected κ ≈ {}, got {}",
            kappa_true,
            estimated_kappa
        );
        assert!(r_squared > 0.7, "Expected high R², got {}", r_squared);
    }

    #[test]
    fn test_warmup() {
        let mut estimator = BookKappaEstimator::new(2000.0);

        // Before any updates
        assert!(!estimator.is_warmed_up());
        assert_eq!(estimator.kappa(), 2000.0); // Prior

        // After a few updates with poor data
        let bids = vec![(99.0, 10.0), (98.0, 10.0)];
        let asks = vec![(101.0, 10.0), (102.0, 10.0)];

        for _ in 0..5 {
            estimator.on_l2_book(&bids, &asks, 100.0);
        }

        // Still should use prior (not enough levels)
        assert_eq!(estimator.kappa(), 2000.0);
    }

    #[test]
    fn test_sparse_book() {
        let mut estimator = BookKappaEstimator::new(2000.0);

        // Very sparse book - only 2 levels
        let bids = vec![(99.0, 10.0), (98.0, 10.0)];
        let asks = vec![(101.0, 10.0), (102.0, 10.0)];

        for _ in 0..20 {
            estimator.on_l2_book(&bids, &asks, 100.0);
        }

        // Should fall back to prior
        assert_eq!(estimator.kappa(), 2000.0);
    }

    #[test]
    fn test_asymmetric_book() {
        let mut estimator = BookKappaEstimator::new(2000.0);

        // Bid side: steep decay (high κ)
        let bids = vec![
            (99.9, 100.0),
            (99.8, 50.0),
            (99.7, 25.0),
            (99.6, 12.0),
            (99.5, 6.0),
        ];

        // Ask side: gradual decay (low κ)
        let asks = vec![
            (100.1, 100.0),
            (100.2, 90.0),
            (100.3, 80.0),
            (100.4, 70.0),
            (100.5, 60.0),
        ];

        for _ in 0..20 {
            estimator.on_l2_book(&bids, &asks, 100.0);
        }

        // Bid κ should be higher than ask κ
        assert!(
            estimator.kappa_bid() > estimator.kappa_ask(),
            "Expected bid κ ({}) > ask κ ({})",
            estimator.kappa_bid(),
            estimator.kappa_ask()
        );
    }
}
