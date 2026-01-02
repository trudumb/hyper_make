//! Kappa (fill rate decay) estimation components.
//!
//! - BayesianKappaEstimator: Gamma conjugate prior for κ estimation
//! - BookStructureEstimator: L2 order book structure analysis

use std::collections::VecDeque;
use tracing::debug;

// ============================================================================
// Bayesian Kappa Estimator (First Principles Implementation)
// ============================================================================

/// Bayesian kappa estimator with Gamma conjugate prior.
///
/// ## First Principles
///
/// In GLFT, κ is the fill rate decay parameter in λ(δ) = A × exp(-κδ).
/// When modeling fill distances as exponential with rate κ:
///
/// - Likelihood: L(δ₁...δₙ | κ) = κⁿ exp(-κ Σδᵢ)
/// - Gamma prior: π(κ | α₀, β₀) ∝ κ^(α₀-1) exp(-β₀ κ)
/// - Posterior: π(κ | data) = Gamma(α₀ + n, β₀ + Σδᵢ)
///
/// This conjugacy gives:
/// - Posterior mean: E[κ | data] = (α₀ + n) / (β₀ + Σδ)
/// - Posterior variance: Var[κ | data] = (α₀ + n) / (β₀ + Σδ)²
/// - Posterior std: σ_κ = κ̂ / √(α₀ + n)
///
/// ## No Clamping Needed
///
/// With proper Bayesian regularization:
/// - Prior provides natural regularization toward reasonable values
/// - Sparse data → posterior ≈ prior (no extreme estimates)
/// - Abundant data → posterior → MLE (data-driven)
/// - Uncertainty is explicit, not hidden by arbitrary clamps
///
/// ## Interpretation
///
/// κ = 500 implies E[distance] = 1/500 = 0.002 = 20 bps average fill distance
/// κ = 1000 implies E[distance] = 10 bps (tighter markets)
/// κ = 200 implies E[distance] = 50 bps (wider markets)
#[derive(Debug)]
pub(crate) struct BayesianKappaEstimator {
    /// Prior shape parameter (α₀). Higher = more confident prior.
    prior_alpha: f64,

    /// Prior rate parameter (β₀). Prior mean = α₀/β₀.
    prior_beta: f64,

    /// Rolling window observations: (distance, volume, timestamp)
    observations: VecDeque<(f64, f64, u64)>,

    /// Rolling window (ms)
    window_ms: u64,

    /// Sum of volume-weighted distances (Σ vᵢ × δᵢ) in current window
    sum_volume_weighted_distance: f64,

    /// Sum of volume-weighted squared distances (for variance/CV)
    sum_volume_weighted_distance_sq: f64,

    /// Sum of volumes (effective n for volume-weighted version)
    sum_volume: f64,

    /// Cached posterior mean κ̂
    kappa_posterior_mean: f64,

    /// Cached posterior standard deviation
    kappa_posterior_std: f64,

    /// Volume-weighted mean distance (for diagnostics)
    mean_distance: f64,

    /// Coefficient of variation (CV = σ/μ). For exponential, CV = 1.0.
    cv: f64,

    /// Update count for logging throttling
    update_count: usize,

    // === Heavy-Tail Detection ===
    /// Rolling count of high-CV observations (CV > 1.2)
    heavy_tail_count: usize,
    /// Total observations for heavy-tail tracking
    heavy_tail_window: usize,
    /// Whether we've detected consistent heavy-tail behavior
    is_heavy_tailed: bool,
}

impl BayesianKappaEstimator {
    /// Create a new Bayesian kappa estimator.
    ///
    /// # Arguments
    /// * `prior_mean` - Prior expected value of κ (e.g., 500 for 20 bps avg distance)
    /// * `prior_strength` - Effective sample size of prior (e.g., 10)
    /// * `window_ms` - Rolling window in milliseconds
    pub(crate) fn new(prior_mean: f64, prior_strength: f64, window_ms: u64) -> Self {
        // Convert prior mean and strength to Gamma parameters
        // Prior mean = α₀/β₀, prior strength = α₀
        // So: α₀ = prior_strength, β₀ = prior_strength / prior_mean
        let prior_alpha = prior_strength;
        let prior_beta = prior_strength / prior_mean;

        Self {
            prior_alpha,
            prior_beta,
            observations: VecDeque::with_capacity(10000),
            window_ms,
            sum_volume_weighted_distance: 0.0,
            sum_volume_weighted_distance_sq: 0.0,
            sum_volume: 0.0,
            kappa_posterior_mean: prior_mean,
            kappa_posterior_std: prior_mean / prior_strength.sqrt(),
            mean_distance: 1.0 / prior_mean,
            cv: 1.0, // Exponential has CV = 1.0
            update_count: 0,
            heavy_tail_count: 0,
            heavy_tail_window: 0,
            is_heavy_tailed: false,
        }
    }

    /// Process a trade and update posterior.
    ///
    /// # Arguments
    /// * `timestamp_ms` - Trade timestamp
    /// * `price` - Trade execution price
    /// * `size` - Trade size (volume weight)
    /// * `mid` - Mid price at time of trade
    pub(crate) fn on_trade(&mut self, timestamp_ms: u64, price: f64, size: f64, mid: f64) {
        if mid <= 0.0 || size <= 0.0 || price <= 0.0 {
            return;
        }

        // Calculate distance as fraction of mid
        let distance = ((price - mid) / mid).abs();

        // Apply small floor to prevent division issues for trades at mid
        // 0.1 bps = 0.00001 is a reasonable floor
        let distance = distance.max(0.00001);

        // Add observation (volume-weighted)
        self.observations.push_back((distance, size, timestamp_ms));
        self.sum_volume_weighted_distance += distance * size;
        self.sum_volume_weighted_distance_sq += distance * distance * size;
        self.sum_volume += size;

        // Expire old observations
        self.expire_old(timestamp_ms);

        // Update posterior
        self.update_posterior();

        self.update_count += 1;

        // Log periodically (every 100 updates)
        if self.update_count.is_multiple_of(100) {
            debug!(
                observations = self.observations.len(),
                sum_volume = %format!("{:.2}", self.sum_volume),
                mean_distance_bps = %format!("{:.2}", self.mean_distance * 10000.0),
                kappa_posterior = %format!("{:.0}", self.kappa_posterior_mean),
                kappa_std = %format!("{:.0}", self.kappa_posterior_std),
                confidence = %format!("{:.2}", self.confidence()),
                cv = %format!("{:.2}", self.cv),
                "Kappa posterior updated (Bayesian)"
            );
        }
    }

    /// Expire old observations outside the rolling window.
    fn expire_old(&mut self, now: u64) {
        let cutoff = now.saturating_sub(self.window_ms);
        while let Some((dist, size, ts)) = self.observations.front() {
            if *ts < cutoff {
                self.sum_volume_weighted_distance -= dist * size;
                self.sum_volume_weighted_distance_sq -= dist * dist * size;
                self.sum_volume -= size;
                self.observations.pop_front();
            } else {
                break;
            }
        }

        // Ensure running sums don't go negative due to float precision
        self.sum_volume_weighted_distance = self.sum_volume_weighted_distance.max(0.0);
        self.sum_volume_weighted_distance_sq = self.sum_volume_weighted_distance_sq.max(0.0);
        self.sum_volume = self.sum_volume.max(0.0);
    }

    /// Update posterior parameters from sufficient statistics.
    fn update_posterior(&mut self) {
        // Calculate mean distance for diagnostics
        if self.sum_volume > 1e-9 {
            self.mean_distance = self.sum_volume_weighted_distance / self.sum_volume;

            // Calculate CV for exponential fit checking
            let mean_sq = self.sum_volume_weighted_distance_sq / self.sum_volume;
            let variance = (mean_sq - self.mean_distance * self.mean_distance).max(0.0);
            if self.mean_distance > 1e-9 {
                self.cv = variance.sqrt() / self.mean_distance;
            }
        }

        // Track heavy-tail detection
        // CV > 1.2 indicates heavy tail (power-law like distribution)
        self.heavy_tail_window += 1;
        if self.cv > 1.2 {
            self.heavy_tail_count += 1;
        }

        // Check for consistent heavy-tail behavior after 100+ observations
        // If more than 80% of recent observations have CV > 1.2, flag as heavy-tailed
        if self.heavy_tail_window >= 100 {
            let heavy_ratio = self.heavy_tail_count as f64 / self.heavy_tail_window as f64;
            let was_heavy = self.is_heavy_tailed;
            self.is_heavy_tailed = heavy_ratio > 0.8;

            // Reset counters periodically to adapt to regime changes
            if self.heavy_tail_window >= 500 {
                // Decay counters by half
                self.heavy_tail_window /= 2;
                self.heavy_tail_count /= 2;
            }

            // Log transition
            if self.is_heavy_tailed != was_heavy {
                debug!(
                    is_heavy = self.is_heavy_tailed,
                    cv = %format!("{:.2}", self.cv),
                    heavy_ratio = %format!("{:.2}", heavy_ratio),
                    "Kappa tail regime change"
                );
            }
        }

        // Posterior parameters with volume weighting
        // Note: Using sum_volume as effective n (volume-weighted sample size)
        // and sum_volume_weighted_distance as the sum of distances
        let posterior_alpha = self.prior_alpha + self.sum_volume;
        let posterior_beta = self.prior_beta + self.sum_volume_weighted_distance;

        // Posterior mean: E[κ | data] = (α₀ + n) / (β₀ + Σδ)
        self.kappa_posterior_mean = posterior_alpha / posterior_beta;

        // Posterior std: σ_κ = κ̂ / √(α₀ + n)
        self.kappa_posterior_std = self.kappa_posterior_mean / posterior_alpha.sqrt();
    }

    /// Get posterior mean of kappa.
    pub(crate) fn posterior_mean(&self) -> f64 {
        self.kappa_posterior_mean
    }

    /// Get posterior standard deviation of kappa.
    pub(crate) fn posterior_std(&self) -> f64 {
        self.kappa_posterior_std
    }

    /// Get confidence score [0, 1] based on sample size.
    ///
    /// Ramps up to 1.0 as effective sample size increases.
    /// With prior_alpha = 10, confidence = √(n) / 10 capped at 1.0.
    pub(crate) fn confidence(&self) -> f64 {
        let effective_n = self.sum_volume;
        (effective_n.sqrt() / 10.0).min(1.0)
    }

    /// Get coefficient of variation (CV = σ/μ of distances).
    ///
    /// For exponential distribution, CV = 1.0 exactly.
    /// CV > 1.0 indicates heavy tail (power-law like)
    /// CV < 1.0 indicates light tail
    pub(crate) fn cv(&self) -> f64 {
        self.cv
    }

    /// Check if distribution is detected as heavy-tailed.
    ///
    /// Returns true if CV has consistently been > 1.2 over recent observations.
    /// Heavy-tailed distributions mean occasional large fills are more likely,
    /// requiring wider spreads to account for tail risk.
    pub(crate) fn is_heavy_tailed(&self) -> bool {
        self.is_heavy_tailed
    }

    /// Get tail-adjusted kappa for GLFT spread calculation.
    ///
    /// When the fill distance distribution is heavy-tailed (CV > 1.2),
    /// the standard exponential model underestimates the probability of
    /// large adverse fills. This method returns a more conservative kappa
    /// by applying a tail risk premium:
    ///
    /// - For exponential (CV ≈ 1.0): κ_adj = κ
    /// - For heavy tail (CV > 1.2): κ_adj = κ × (2 - CV), capped at 0.5×κ
    ///
    /// Lower effective κ → wider spreads → protection against tail risk.
    #[allow(dead_code)]
    pub(crate) fn tail_adjusted_kappa(&self) -> f64 {
        if !self.is_heavy_tailed {
            return self.kappa_posterior_mean;
        }

        // Heavy-tail adjustment: reduce effective kappa based on CV
        // CV = 1.2 → multiplier = 0.8 (20% reduction)
        // CV = 1.5 → multiplier = 0.5 (50% reduction, floor)
        let multiplier = (2.0 - self.cv).clamp(0.5, 1.0);
        self.kappa_posterior_mean * multiplier
    }

    /// Get mean fill distance for diagnostics.
    #[allow(dead_code)]
    pub(crate) fn mean_distance(&self) -> f64 {
        self.mean_distance
    }

    /// Get observation count.
    #[allow(dead_code)]
    pub(crate) fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Get effective sample size (sum of volumes).
    #[allow(dead_code)]
    pub(crate) fn effective_sample_size(&self) -> f64 {
        self.sum_volume
    }

    /// Get update count (for warmup checking).
    pub(crate) fn update_count(&self) -> usize {
        self.update_count
    }

    /// Record a fill observation from our own order.
    ///
    /// This is the CORRECT measurement for OUR fill rate decay:
    /// - placement_price: Where we placed the order
    /// - fill_price: Where it actually filled
    /// - distance: |fill - placement| / placement
    ///
    /// For a market maker, this measures how far price moved
    /// against us before our order got hit. This is exactly what
    /// GLFT's κ models.
    pub(crate) fn record_fill_distance(
        &mut self,
        timestamp_ms: u64,
        placement_price: f64,
        fill_price: f64,
        fill_size: f64,
    ) {
        if placement_price <= 0.0 || fill_price <= 0.0 || fill_size <= 0.0 {
            return;
        }

        // Distance as fraction of placement price
        let distance = ((fill_price - placement_price) / placement_price).abs();

        // Minimum floor (fills exactly at placement price get 0.1 bps)
        let distance = distance.max(0.00001);

        // Add to posterior (same math as on_trade)
        self.observations
            .push_back((distance, fill_size, timestamp_ms));
        self.sum_volume_weighted_distance += distance * fill_size;
        self.sum_volume_weighted_distance_sq += distance * distance * fill_size;
        self.sum_volume += fill_size;

        self.expire_old(timestamp_ms);
        self.update_posterior();
        self.update_count += 1;

        // Log every fill (own fills are valuable data)
        debug!(
            fill_distance_bps = %format!("{:.2}", distance * 10000.0),
            placement_price = %format!("{:.2}", placement_price),
            fill_price = %format!("{:.2}", fill_price),
            fill_size = %format!("{:.4}", fill_size),
            kappa_posterior = %format!("{:.0}", self.kappa_posterior_mean),
            confidence = %format!("{:.2}", self.confidence()),
            "Own fill recorded for kappa estimation"
        );
    }
}

// ============================================================================
// Book Structure Estimator (L2 Order Book Analysis)
// ============================================================================

/// Analyzes L2 order book structure for auxiliary quote adjustments.
///
/// Provides two key signals:
/// 1. **Book Imbalance** [-1, 1]: Bid/ask depth asymmetry
///    - Positive = more bids than asks (buying pressure)
///    - Used for directional skew adjustment
///
/// 2. **Liquidity Gamma Multiplier** [1.0, 2.0]: Thin book detection
///    - Scales γ up when near-touch liquidity is below average
///    - Protects against adverse selection in thin markets
#[derive(Debug)]
pub(crate) struct BookStructureEstimator {
    /// EWMA smoothed book imbalance [-1, 1]
    imbalance: f64,
    /// Current near-touch depth (within 10 bps of mid)
    near_touch_depth: f64,
    /// Rolling reference depth for comparison
    reference_depth: f64,
    /// EWMA smoothing factor
    alpha: f64,
    /// Number of levels to consider for imbalance
    imbalance_levels: usize,
    /// Maximum distance for near-touch liquidity (as fraction)
    near_touch_distance: f64,
}

impl BookStructureEstimator {
    pub(crate) fn new() -> Self {
        Self {
            imbalance: 0.0,
            near_touch_depth: 0.0,
            reference_depth: 1.0, // Start with 1.0 to avoid division issues
            alpha: 0.1,
            imbalance_levels: 5,
            near_touch_distance: 0.001, // 10 bps
        }
    }

    /// Update with new L2 book data.
    pub(crate) fn update(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)], mid: f64) {
        if mid <= 0.0 {
            return;
        }

        // 1. Calculate bid/ask imbalance from top N levels
        let bid_depth: f64 = bids
            .iter()
            .take(self.imbalance_levels)
            .map(|(_, sz)| sz)
            .sum();
        let ask_depth: f64 = asks
            .iter()
            .take(self.imbalance_levels)
            .map(|(_, sz)| sz)
            .sum();

        let total = bid_depth + ask_depth;
        if total > 1e-9 {
            let instant_imbalance = (bid_depth - ask_depth) / total;
            self.imbalance = self.alpha * instant_imbalance + (1.0 - self.alpha) * self.imbalance;
        }

        // 2. Calculate near-touch liquidity (depth within 10 bps of mid)
        let bid_near: f64 = bids
            .iter()
            .take_while(|(px, _)| (mid - px) / mid <= self.near_touch_distance)
            .map(|(_, sz)| sz)
            .sum();
        let ask_near: f64 = asks
            .iter()
            .take_while(|(px, _)| (px - mid) / mid <= self.near_touch_distance)
            .map(|(_, sz)| sz)
            .sum();
        self.near_touch_depth = bid_near + ask_near;

        // 3. Update reference depth (slow-moving average)
        // Use very slow decay to establish "normal" liquidity baseline
        self.reference_depth =
            0.99 * self.reference_depth + 0.01 * self.near_touch_depth.max(0.001);
    }

    /// Get current book imbalance [-1, 1].
    /// Positive = more bids (buying pressure), Negative = more asks (selling pressure).
    pub(crate) fn imbalance(&self) -> f64 {
        self.imbalance.clamp(-1.0, 1.0)
    }

    /// Get gamma multiplier for thin book conditions [1.0, 2.0].
    /// Returns > 1.0 when near-touch liquidity is below reference.
    pub(crate) fn gamma_multiplier(&self) -> f64 {
        if self.near_touch_depth >= self.reference_depth {
            1.0
        } else {
            // Thin book → scale gamma up (wider spreads for protection)
            // sqrt scaling: 4x thinner → 2x gamma
            let ratio = self.reference_depth / self.near_touch_depth.max(0.001);
            ratio.sqrt().clamp(1.0, 2.0)
        }
    }

    /// Get near-touch depth in contracts.
    /// This is the depth within 10 bps (0.1%) of mid.
    pub(crate) fn near_touch_depth(&self) -> f64 {
        self.near_touch_depth
    }
}
