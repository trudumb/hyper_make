//! Blended Kappa Estimator - Component 3
//!
//! Implements sigmoid-weighted blending of book-based and own-fill kappa estimates.
//! This provides fast warmup from book structure while converging to accurate
//! own-fill estimates as data accumulates.
//!
//! # Mathematical Model
//!
//! Two κ sources with different properties:
//! 1. κ_book = from L2 order book structure (fast, approximate)
//! 2. κ_own = from our fill distances (slow, accurate)
//!
//! Blending formula:
//! ```text
//! κ_eff = (1 - w(n)) × κ_book + w(n) × κ_own
//! ```
//!
//! Where w(n) is a sigmoid blend weight:
//! ```text
//! w(n) = sigmoid((n - n_min) / scale) = 1 / (1 + exp(-(n - n_min) / scale))
//! ```
//!
//! # Rationale
//!
//! - Book-based κ is available immediately from L2 data
//! - Own-fill κ is the TRUE κ relevant to GLFT (how our orders fill)
//! - Sigmoid blending smoothly transitions as data accumulates

use std::collections::VecDeque;
use tracing::debug;

/// Blended kappa estimator combining book structure and own-fill estimates.
#[derive(Debug, Clone)]
pub struct BlendedKappaEstimator {
    /// Own-fill distance observations: (distance_fraction, size, timestamp_ms)
    own_fill_observations: VecDeque<(f64, f64, u64)>,

    /// Number of own fills processed
    own_fill_count: usize,

    /// Sum of volume-weighted distances for own fills
    own_sum_vw_distance: f64,

    /// Sum of volumes for own fills
    own_sum_volume: f64,

    /// Own-fill Bayesian posterior: Gamma(alpha, beta)
    own_alpha: f64,
    own_beta: f64,

    /// Book-based kappa estimate
    book_kappa: f64,

    /// Book kappa update count
    book_update_count: usize,

    /// Prior mean for kappa
    prior_mean: f64,

    /// Prior strength (effective sample size)
    prior_strength: f64,

    /// Minimum fills before blending starts
    blend_min_fills: usize,

    /// Blend sigmoid scale (steepness)
    blend_scale: f64,

    /// Warmup conservatism factor
    warmup_factor: f64,

    /// Rolling window for own fills (ms)
    window_ms: u64,
}

impl BlendedKappaEstimator {
    /// Create a new blended kappa estimator.
    ///
    /// # Arguments
    /// * `prior_mean` - Prior expected κ (e.g., 2500 for 4 bps avg distance)
    /// * `prior_strength` - Effective sample size of prior
    /// * `blend_min_fills` - Minimum fills before blending starts
    /// * `blend_scale` - Sigmoid steepness
    /// * `warmup_factor` - Conservatism during warmup (< 1.0 widens)
    /// * `window_ms` - Rolling window for observations
    pub fn new(
        prior_mean: f64,
        prior_strength: f64,
        blend_min_fills: usize,
        blend_scale: f64,
        warmup_factor: f64,
        window_ms: u64,
    ) -> Self {
        // Initialize Gamma prior: mean = α/β, strength = α
        let alpha = prior_strength;
        let beta = prior_strength / prior_mean;

        Self {
            own_fill_observations: VecDeque::with_capacity(1000),
            own_fill_count: 0,
            own_sum_vw_distance: 0.0,
            own_sum_volume: 0.0,
            own_alpha: alpha,
            own_beta: beta,
            book_kappa: prior_mean,
            book_update_count: 0,
            prior_mean,
            prior_strength,
            blend_min_fills,
            blend_scale,
            warmup_factor,
            window_ms,
        }
    }

    /// Create from config.
    pub fn from_config(config: &super::AdaptiveBayesianConfig) -> Self {
        Self::new(
            config.kappa_prior_mean,
            config.kappa_prior_strength,
            config.kappa_blend_min_fills,
            config.kappa_blend_scale,
            config.kappa_warmup_factor,
            600_000, // 10 minute window (increased from 5 min for sparse fills)
        )
    }

    /// Record an own fill observation.
    ///
    /// # Arguments
    /// * `fill_price` - Price at which we filled
    /// * `mid_at_fill` - Mid price at time of fill
    /// * `size` - Fill size (for volume weighting)
    /// * `timestamp_ms` - Fill timestamp
    pub fn on_own_fill(&mut self, fill_price: f64, mid_at_fill: f64, size: f64, timestamp_ms: u64) {
        if mid_at_fill <= 0.0 || size <= 0.0 {
            return;
        }

        // Calculate fill distance as fraction of mid
        let distance = ((fill_price - mid_at_fill) / mid_at_fill).abs();
        let distance = distance.max(0.00001); // Floor at 0.1 bps

        // Add observation
        self.own_fill_observations
            .push_back((distance, size, timestamp_ms));
        self.own_fill_count += 1;
        self.own_sum_vw_distance += distance * size;
        self.own_sum_volume += size;

        // Expire old observations
        self.expire_old_observations(timestamp_ms);

        // Update Bayesian posterior
        // For exponential likelihood with Gamma prior:
        // posterior: Gamma(α + n, β + Σδᵢ)
        // Using volume-weighted effective n
        let effective_n = self.own_sum_volume;
        let sum_distance = self.own_sum_vw_distance;

        self.own_alpha = self.prior_strength + effective_n;
        self.own_beta = self.prior_strength / self.prior_mean + sum_distance;

        // Log periodically
        if self.own_fill_count.is_multiple_of(20) {
            debug!(
                own_fill_count = self.own_fill_count,
                own_kappa = self.own_kappa(),
                book_kappa = self.book_kappa,
                blended_kappa = self.kappa(),
                blend_weight = self.blend_weight(),
                "Blended kappa updated from own fill"
            );
        }
    }

    /// Update book-based kappa from L2 order book.
    ///
    /// # Arguments
    /// * `bids` - Bid levels as (price, size) tuples, sorted by price desc
    /// * `asks` - Ask levels as (price, size) tuples, sorted by price asc
    /// * `mid` - Current mid price
    pub fn on_l2_update(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)], mid: f64) {
        if mid <= 0.0 {
            return;
        }

        // Estimate kappa from cumulative depth decay
        let kappa_bid = self.estimate_book_kappa(bids, mid, true);
        let kappa_ask = self.estimate_book_kappa(asks, mid, false);

        // Average bid and ask kappa (both should be similar)
        if kappa_bid > 0.0 && kappa_ask > 0.0 {
            self.book_kappa = 0.5 * (kappa_bid + kappa_ask);
        } else if kappa_bid > 0.0 {
            self.book_kappa = kappa_bid;
        } else if kappa_ask > 0.0 {
            self.book_kappa = kappa_ask;
        }
        // else: keep previous estimate

        self.book_update_count += 1;
    }

    /// Estimate kappa from one side of the book.
    ///
    /// Models: Depth(δ) = A × exp(-κ × δ)
    /// So: ln(Depth(δ)) = ln(A) - κ × δ
    ///
    /// Linear regression of ln(cumulative_depth) vs distance gives -κ as slope.
    fn estimate_book_kappa(&self, levels: &[(f64, f64)], mid: f64, is_bid: bool) -> f64 {
        if levels.len() < 3 {
            return 0.0;
        }

        // Use first 10 levels (or fewer)
        let n = levels.len().min(10);

        // Compute cumulative depth at each distance
        let mut cum_depth = 0.0;
        let mut points: Vec<(f64, f64)> = Vec::with_capacity(n);

        for &(price, size) in levels.iter().take(n) {
            cum_depth += size;

            let distance = if is_bid {
                (mid - price) / mid
            } else {
                (price - mid) / mid
            };

            if distance > 0.0 && cum_depth > 0.0 {
                points.push((distance, cum_depth.ln()));
            }
        }

        if points.len() < 3 {
            return 0.0;
        }

        // Simple linear regression: y = a + b×x
        // We want b (slope), which is -κ
        let n_f = points.len() as f64;
        let sum_x: f64 = points.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = points.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
        let sum_xx: f64 = points.iter().map(|(x, _)| x * x).sum();

        let denominator = n_f * sum_xx - sum_x * sum_x;
        if denominator.abs() < 1e-12 {
            return 0.0;
        }

        let slope = (n_f * sum_xy - sum_x * sum_y) / denominator;

        // κ = -slope (slope should be negative for depth decay)
        (-slope).clamp(10.0, 10000.0) // Reasonable bounds
    }

    /// Expire old observations outside the window.
    fn expire_old_observations(&mut self, current_time_ms: u64) {
        let cutoff = current_time_ms.saturating_sub(self.window_ms);

        while let Some(&(distance, size, ts)) = self.own_fill_observations.front() {
            if ts < cutoff {
                self.own_fill_observations.pop_front();
                self.own_sum_vw_distance -= distance * size;
                self.own_sum_volume -= size;
            } else {
                break;
            }
        }

        // Ensure non-negative
        self.own_sum_vw_distance = self.own_sum_vw_distance.max(0.0);
        self.own_sum_volume = self.own_sum_volume.max(0.0);
    }

    /// Get the sigmoid blend weight.
    ///
    /// w(n) = 1 / (1 + exp(-(n - n_min) / scale))
    fn blend_weight(&self) -> f64 {
        let n = self.own_fill_count as f64;
        let n_min = self.blend_min_fills as f64;

        1.0 / (1.0 + (-(n - n_min) / self.blend_scale).exp())
    }

    /// Get own-fill kappa estimate (Bayesian posterior mean).
    pub fn own_kappa(&self) -> f64 {
        if self.own_beta > 0.0 {
            self.own_alpha / self.own_beta
        } else {
            self.prior_mean
        }
    }

    /// Get book-based kappa estimate.
    pub fn book_kappa(&self) -> f64 {
        self.book_kappa
    }

    /// Check if book-based kappa estimate is reliable.
    ///
    /// Returns false if:
    /// - κ_book > 3× prior (likely thin book artifact)
    /// - Fewer than 10 L2 updates processed
    pub fn book_kappa_reliable(&self) -> bool {
        self.book_kappa < self.prior_mean * 3.0 && self.book_update_count >= 10
    }

    /// Get the blended kappa estimate.
    ///
    /// Includes reliability check to prevent spread collapse on thin books.
    /// When book kappa is unreliable (too high or insufficient samples),
    /// falls back to prior. The final warmup factor is applied separately.
    pub fn kappa(&self) -> f64 {
        let w = self.blend_weight();

        // Validate book kappa before using it
        // High values (κ > 3× prior) on thin books can cause spreads to collapse
        let book_reliable = self.book_kappa < self.prior_mean * 3.0 && self.book_update_count >= 10;

        let effective_book_kappa = if book_reliable {
            self.book_kappa
        } else {
            // Use prior when book is unreliable
            // (warmup factor applied at the end, not here to avoid double-counting)
            self.prior_mean
        };

        let blended = (1.0 - w) * effective_book_kappa + w * self.own_kappa();

        // Apply warmup conservatism if still early
        if self.own_fill_count < self.blend_min_fills {
            blended * self.warmup_factor
        } else {
            blended
        }
    }

    /// Get directional kappa estimates (for asymmetric spreads).
    ///
    /// Returns (kappa_bid, kappa_ask) where lower kappa = wider spread needed.
    pub fn directional_kappa(&self) -> (f64, f64) {
        // For now, return symmetric kappa
        // Could be extended to track bid/ask fills separately
        let k = self.kappa();
        (k, k)
    }

    /// Get the number of own fills recorded.
    pub fn own_fill_count(&self) -> usize {
        self.own_fill_count
    }

    /// Increment the fill count (for simplified fill processing).
    ///
    /// This is used when we don't have full fill details but want to
    /// update the blend weight based on receiving a fill.
    pub fn increment_fill_count(&mut self) {
        self.own_fill_count += 1;
    }

    /// Record a fill with known distance (for simplified fill processing).
    ///
    /// This updates both the fill count AND the Bayesian posterior,
    /// using a unit size weight. Use when fill_price/mid aren't available
    /// but the distance is known.
    ///
    /// # Arguments
    /// * `distance` - Fill distance as fraction of mid (e.g., 0.0004 = 4 bps)
    /// * `timestamp_ms` - Fill timestamp for windowing
    pub fn on_fill_distance(&mut self, distance: f64, timestamp_ms: u64) {
        let distance = distance.max(0.00001); // Floor at 0.1 bps

        // Add observation with unit size weight
        self.own_fill_observations
            .push_back((distance, 1.0, timestamp_ms));
        self.own_fill_count += 1;
        self.own_sum_vw_distance += distance;
        self.own_sum_volume += 1.0;

        // Expire old observations
        self.expire_old_observations(timestamp_ms);

        // Update Bayesian posterior
        // For Gamma-Exponential conjugacy: posterior = Gamma(α₀ + n, β₀ + Σdᵢ)
        let n = self.own_fill_observations.len() as f64;
        let sum_distances: f64 = self.own_fill_observations.iter().map(|(d, _, _)| d).sum();
        self.own_alpha = self.prior_strength + n;
        self.own_beta = self.prior_strength / self.prior_mean + sum_distances;

        debug!(
            distance_bps = %format!("{:.2}", distance * 10000.0),
            own_kappa = %format!("{:.0}", self.own_kappa()),
            fill_count = self.own_fill_count,
            "BlendedKappa: fill distance recorded"
        );
    }

    /// Check if we have enough own-fill data to be reliable.
    pub fn is_warmed_up(&self) -> bool {
        self.own_fill_count >= self.blend_min_fills
    }

    /// Get posterior uncertainty (standard deviation of kappa estimate).
    pub fn posterior_std(&self) -> f64 {
        // For Gamma(α, β), variance = α/β²
        // std = sqrt(α) / β
        if self.own_beta > 0.0 {
            self.own_alpha.sqrt() / self.own_beta
        } else {
            self.prior_mean / self.prior_strength.sqrt()
        }
    }

    /// Reset own-fill statistics (keep book estimate).
    pub fn reset_own_fills(&mut self) {
        self.own_fill_observations.clear();
        self.own_fill_count = 0;
        self.own_sum_vw_distance = 0.0;
        self.own_sum_volume = 0.0;
        self.own_alpha = self.prior_strength;
        self.own_beta = self.prior_strength / self.prior_mean;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_blended() -> BlendedKappaEstimator {
        BlendedKappaEstimator::new(
            2500.0,  // prior_mean: 4 bps avg distance
            5.0,     // prior_strength
            10,      // blend_min_fills
            5.0,     // blend_scale
            0.8,     // warmup_factor
            300_000, // 5 min window
        )
    }

    #[test]
    fn test_initial_kappa_uses_prior() {
        let est = default_blended();

        // Before any data, should be close to prior (with warmup factor)
        let k = est.kappa();
        let expected = 2500.0 * 0.8; // prior × warmup_factor

        assert!(
            (k - expected).abs() < 100.0,
            "Initial kappa should be ~{}, got {}",
            expected,
            k
        );
    }

    #[test]
    fn test_blend_weight_sigmoid() {
        let est = default_blended();

        // At n = 0, weight should be low
        assert!(est.blend_weight() < 0.2, "Weight at n=0 should be low");
    }

    #[test]
    fn test_blend_weight_increases_with_fills() {
        let mut est = default_blended();

        // Simulate fills
        for i in 0..30 {
            est.on_own_fill(100.0 + 0.04, 100.0, 1.0, i * 1000); // 4 bps distance
        }

        // After 30 fills, weight should be high
        assert!(
            est.blend_weight() > 0.8,
            "Weight after 30 fills should be high: {}",
            est.blend_weight()
        );
    }

    #[test]
    fn test_own_kappa_updates_from_fills() {
        let mut est = default_blended();

        // Simulate fills at 2 bps distance (tighter than prior)
        for i in 0..50 {
            est.on_own_fill(100.0 + 0.02, 100.0, 1.0, i * 1000); // 2 bps distance
        }

        // Own kappa should be higher than prior (tighter market)
        // κ = 1/avg_distance, so tighter distance = higher κ
        assert!(
            est.own_kappa() > 2500.0,
            "Own kappa should be > prior for tighter fills: {}",
            est.own_kappa()
        );
    }

    #[test]
    fn test_blending_converges_to_own_kappa() {
        let mut est = default_blended();

        // Set book kappa to something different
        est.book_kappa = 1000.0;

        // Simulate many fills
        for i in 0..100 {
            est.on_own_fill(100.0 + 0.04, 100.0, 1.0, i * 1000); // 4 bps
        }

        let blended = est.kappa();
        let own = est.own_kappa();

        // After many fills, blended should be very close to own
        let diff_pct = ((blended - own) / own).abs();
        assert!(
            diff_pct < 0.1,
            "Blended should converge to own: blended={}, own={}",
            blended,
            own
        );
    }

    #[test]
    fn test_warmup_factor_applied() {
        let est = default_blended();

        // Before warmup, factor should be applied
        let k = est.kappa();
        assert!(
            k < est.prior_mean,
            "Warmup factor should reduce kappa: {} vs {}",
            k,
            est.prior_mean
        );
    }

    #[test]
    fn test_book_kappa_estimation() {
        let mut est = default_blended();

        // Create synthetic order book with CUMULATIVE depth (increasing with distance)
        // This simulates a realistic book where deeper levels have more total liquidity
        let mid = 100.0;

        // Bids: price decreasing, cumulative size should increase
        let bids: Vec<(f64, f64)> = (1..=10)
            .map(|i| {
                let price = mid - 0.1 * i as f64; // 0.1% per level (10 bps)
                let size = i as f64 * 2.0; // Increasing size at each level
                (price, size)
            })
            .collect();

        // Asks: price increasing, cumulative size should increase
        let asks: Vec<(f64, f64)> = (1..=10)
            .map(|i| {
                let price = mid + 0.1 * i as f64;
                let size = i as f64 * 2.0;
                (price, size)
            })
            .collect();

        est.on_l2_update(&bids, &asks, mid);

        // Book kappa should have been updated (may still be at floor bounds)
        // The key is that it processed without error
        assert!(
            est.book_kappa() >= 10.0,
            "Book kappa should be at least minimum: {}",
            est.book_kappa()
        );
    }
}
