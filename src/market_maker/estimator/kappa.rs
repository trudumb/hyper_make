//! Kappa (fill rate decay) estimation components.
//!
//! - BayesianKappaEstimator: Gamma conjugate prior for κ estimation (V2 - corrected)
//! - BookStructureEstimator: L2 order book structure analysis
//!
//! ## V2 Conjugacy Fix
//!
//! The original implementation incorrectly used volume-weighted sums as the
//! effective sample size, which breaks Gamma-Exponential conjugacy.
//!
//! **Correct conjugate update:**
//! - Prior: κ ~ Gamma(α₀, β₀)
//! - Likelihood: X₁...Xₙ | κ ~ Exp(κ) → L(data|κ) = κⁿ exp(-κ Σxᵢ)
//! - Posterior: κ | data ~ Gamma(α₀ + **n**, β₀ + Σxᵢ)
//!
//! The shape update must use COUNT of observations, not sum of volumes!

use std::collections::VecDeque;
use tracing::debug;

use super::tick_ewma::TickEWMAVariance;

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

    /// Rolling window observations: (distance, timestamp)
    /// V2: Removed volume - each observation counts equally for conjugacy
    observations: VecDeque<(f64, u64)>,

    /// Rolling window (ms)
    window_ms: u64,

    /// V2 FIX: Observation count (n) - the CORRECT sufficient statistic for shape
    observation_count: usize,

    /// Sum of distances (Σδᵢ) - unweighted for proper conjugacy
    sum_distances: f64,

    /// Sum of squared distances (Σδᵢ²) - for variance/CV calculation
    sum_sq_distances: f64,

    /// Legacy: Sum of volumes (kept for backwards compatibility metrics)
    sum_volume: f64,

    /// Cached posterior mean κ̂
    kappa_posterior_mean: f64,

    /// Cached posterior standard deviation
    kappa_posterior_std: f64,

    /// Cached posterior variance
    kappa_posterior_var: f64,

    /// Mean distance (for diagnostics)
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

    // === V2 Additions ===
    /// Tick-based variance tracker for CV (volume-clock aligned)
    distance_variance: TickEWMAVariance,

    /// 95% credible interval lower bound
    ci_95_lower: f64,

    /// 95% credible interval upper bound
    ci_95_upper: f64,
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
        let prior_var = prior_alpha / (prior_beta * prior_beta);

        // Initial credible interval from prior
        let (ci_lower, ci_upper) =
            Self::compute_credible_interval_static(prior_alpha, prior_beta, 0.95);

        Self {
            prior_alpha,
            prior_beta,
            observations: VecDeque::with_capacity(10000),
            window_ms,
            observation_count: 0,
            sum_distances: 0.0,
            sum_sq_distances: 0.0,
            sum_volume: 0.0,
            kappa_posterior_mean: prior_mean,
            kappa_posterior_std: prior_var.sqrt(),
            kappa_posterior_var: prior_var,
            mean_distance: 1.0 / prior_mean,
            cv: 1.0, // Exponential has CV = 1.0
            update_count: 0,
            heavy_tail_count: 0,
            heavy_tail_window: 0,
            is_heavy_tailed: false,
            distance_variance: TickEWMAVariance::new(50.0),
            ci_95_lower: ci_lower,
            ci_95_upper: ci_upper,
        }
    }

    /// Process a trade and update posterior.
    ///
    /// # Arguments
    /// * `timestamp_ms` - Trade timestamp
    /// * `price` - Trade execution price
    /// * `size` - Trade size (kept for volume tracking, but not used in conjugate update)
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

        // V2 FIX: Add observation with UNWEIGHTED distance
        // Each observation counts as 1 for the shape parameter (n)
        self.observations.push_back((distance, timestamp_ms));
        self.observation_count += 1;
        self.sum_distances += distance;
        self.sum_sq_distances += distance * distance;
        self.sum_volume += size; // Keep for metrics, not used in conjugate update

        // Update tick-based variance tracker
        self.distance_variance.update(distance);

        // Expire old observations
        self.expire_old(timestamp_ms);

        // Update posterior
        self.update_posterior();

        self.update_count += 1;

        // Log periodically (every 100 updates)
        if self.update_count.is_multiple_of(100) {
            debug!(
                observations = self.observation_count,
                sum_volume = %format!("{:.2}", self.sum_volume),
                mean_distance_bps = %format!("{:.2}", self.mean_distance * 10000.0),
                kappa_posterior = %format!("{:.0}", self.kappa_posterior_mean),
                kappa_std = %format!("{:.0}", self.kappa_posterior_std),
                ci_95 = %format!("[{:.0}, {:.0}]", self.ci_95_lower, self.ci_95_upper),
                confidence = %format!("{:.2}", self.confidence()),
                cv = %format!("{:.2}", self.cv),
                "Kappa posterior updated (V2 Bayesian)"
            );
        }
    }

    /// Expire old observations outside the rolling window.
    fn expire_old(&mut self, now: u64) {
        let cutoff = now.saturating_sub(self.window_ms);
        while let Some(&(dist, ts)) = self.observations.front() {
            if ts < cutoff {
                // V2 FIX: Decrement observation COUNT
                self.observation_count = self.observation_count.saturating_sub(1);
                self.sum_distances -= dist;
                self.sum_sq_distances -= dist * dist;
                self.observations.pop_front();
            } else {
                break;
            }
        }

        // Ensure running sums don't go negative due to float precision
        self.sum_distances = self.sum_distances.max(0.0);
        self.sum_sq_distances = self.sum_sq_distances.max(0.0);
    }

    /// Update posterior parameters from sufficient statistics.
    fn update_posterior(&mut self) {
        // Calculate mean distance for diagnostics
        if self.observation_count > 0 {
            self.mean_distance = self.sum_distances / self.observation_count as f64;

            // Calculate CV for exponential fit checking
            let mean_sq = self.sum_sq_distances / self.observation_count as f64;
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

        // V2 FIX: Correct conjugate posterior parameters
        // The shape update uses OBSERVATION COUNT (n), not volume sum!
        // For Gamma-Exponential conjugacy:
        //   Prior: κ ~ Gamma(α₀, β₀)
        //   Posterior: κ | data ~ Gamma(α₀ + n, β₀ + Σδᵢ)
        let posterior_alpha = self.prior_alpha + self.observation_count as f64;
        let posterior_beta = self.prior_beta + self.sum_distances;

        // Posterior mean: E[κ | data] = (α₀ + n) / (β₀ + Σδ)
        self.kappa_posterior_mean = posterior_alpha / posterior_beta;

        // Posterior variance: Var[κ | data] = (α₀ + n) / (β₀ + Σδ)²
        self.kappa_posterior_var = posterior_alpha / (posterior_beta * posterior_beta);
        self.kappa_posterior_std = self.kappa_posterior_var.sqrt();

        // Update 95% credible interval
        let (ci_lower, ci_upper) =
            Self::compute_credible_interval_static(posterior_alpha, posterior_beta, 0.95);
        self.ci_95_lower = ci_lower;
        self.ci_95_upper = ci_upper;
    }

    /// Compute credible interval for Gamma(α, β) distribution.
    fn compute_credible_interval_static(alpha: f64, beta: f64, level: f64) -> (f64, f64) {
        // Use normal approximation for large α
        if alpha > 30.0 {
            let mean = alpha / beta;
            let std = (alpha / (beta * beta)).sqrt();
            let z = 1.96; // 95% for normal
            return ((mean - z * std).max(0.0), mean + z * std);
        }

        // For smaller α, use chi-squared relationship
        // Gamma(α, β) = (1/β) × χ²(2α) / 2
        let chi_sq_df = 2.0 * alpha;

        let lower_chi = chi_squared_quantile(chi_sq_df, (1.0 - level) / 2.0);
        let upper_chi = chi_squared_quantile(chi_sq_df, (1.0 + level) / 2.0);

        (lower_chi / (2.0 * beta), upper_chi / (2.0 * beta))
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
    /// V2: Uses observation count (not volume) as the effective sample size.
    /// Confidence = 1 / (1 + CV_posterior) where CV = std/mean.
    pub(crate) fn confidence(&self) -> f64 {
        // Posterior CV: std / mean
        let cv = self.kappa_posterior_std / self.kappa_posterior_mean.max(1e-10);
        // Map CV to confidence: CV=0 → 1.0, CV=1 → 0.5, CV→∞ → 0
        1.0 / (1.0 + cv)
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

    /// Get observation count (V2: the actual count used in posterior).
    #[allow(dead_code)]
    pub(crate) fn observation_count(&self) -> usize {
        self.observation_count
    }

    /// Get effective sample size (observation count + prior strength).
    #[allow(dead_code)]
    pub(crate) fn effective_sample_size(&self) -> f64 {
        self.observation_count as f64 + self.prior_alpha
    }

    /// Get 95% credible interval (lower bound).
    #[allow(dead_code)]
    pub(crate) fn ci_95_lower(&self) -> f64 {
        self.ci_95_lower
    }

    /// Get 95% credible interval (upper bound).
    #[allow(dead_code)]
    pub(crate) fn ci_95_upper(&self) -> f64 {
        self.ci_95_upper
    }

    /// Get posterior variance.
    #[allow(dead_code)]
    pub(crate) fn posterior_variance(&self) -> f64 {
        self.kappa_posterior_var
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

        // V2 FIX: Add observation with UNWEIGHTED distance
        self.observations.push_back((distance, timestamp_ms));
        self.observation_count += 1;
        self.sum_distances += distance;
        self.sum_sq_distances += distance * distance;
        self.sum_volume += fill_size; // Keep for metrics

        // Update tick-based variance tracker
        self.distance_variance.update(distance);

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
            ci_95 = %format!("[{:.0}, {:.0}]", self.ci_95_lower, self.ci_95_upper),
            confidence = %format!("{:.2}", self.confidence()),
            "Own fill recorded for kappa estimation (V2)"
        );
    }

    /// Update the prior parameters to reflect a new prior mean and strength.
    ///
    /// This allows online adaptation of the prior (e.g., adapting toward observed
    /// market kappa for new/unfamiliar assets). The posterior is recomputed from
    /// the new prior combined with existing sufficient statistics.
    ///
    /// # Arguments
    /// * `new_mean` - New prior mean for kappa
    /// * `new_strength` - New prior strength (effective pseudo-observations)
    ///
    /// # Safety
    /// Both `new_mean` and `new_strength` must be > 0. Values are clamped internally.
    pub(crate) fn update_prior(&mut self, new_mean: f64, new_strength: f64) {
        let safe_mean = new_mean.clamp(100.0, 50000.0);
        let safe_strength = new_strength.clamp(1.0, 50.0);

        self.prior_alpha = safe_strength;
        self.prior_beta = safe_strength / safe_mean;

        // Recompute posterior from new prior + existing data
        let posterior_alpha = self.prior_alpha + self.observation_count as f64;
        let posterior_beta = self.prior_beta + self.sum_distances;
        self.kappa_posterior_mean = posterior_alpha / posterior_beta;
        self.kappa_posterior_var = posterior_alpha / (posterior_beta * posterior_beta);
        self.kappa_posterior_std = self.kappa_posterior_var.sqrt();

        // Recompute credible interval
        let (ci_lower, ci_upper) =
            Self::compute_credible_interval_static(posterior_alpha, posterior_beta, 0.95);
        self.ci_95_lower = ci_lower;
        self.ci_95_upper = ci_upper;
    }

    /// Get the current prior mean (alpha / beta).
    pub(crate) fn prior_mean(&self) -> f64 {
        self.prior_alpha / self.prior_beta
    }

    /// Get the current prior strength (alpha).
    #[allow(dead_code)]
    pub(crate) fn prior_strength(&self) -> f64 {
        self.prior_alpha
    }

    // === Checkpoint persistence ===

    /// Extract sufficient statistics for checkpoint persistence.
    ///
    /// The rolling observation window (VecDeque) is NOT persisted — the sufficient
    /// statistics fully determine the posterior.
    pub(crate) fn to_checkpoint(&self) -> crate::market_maker::checkpoint::KappaCheckpoint {
        crate::market_maker::checkpoint::KappaCheckpoint {
            prior_alpha: self.prior_alpha,
            prior_beta: self.prior_beta,
            observation_count: self.observation_count,
            sum_distances: self.sum_distances,
            sum_sq_distances: self.sum_sq_distances,
            kappa_posterior_mean: self.kappa_posterior_mean,
            total_observations: self.update_count,
        }
    }

    /// Restore sufficient statistics from a checkpoint.
    ///
    /// The VecDeque rolling window stays empty — it refills from live fills.
    /// The posterior is recomputed from the restored sufficient statistics.
    pub(crate) fn restore_checkpoint(
        &mut self,
        cp: &crate::market_maker::checkpoint::KappaCheckpoint,
    ) {
        self.prior_alpha = cp.prior_alpha;
        self.prior_beta = cp.prior_beta;
        self.observation_count = cp.observation_count;
        self.sum_distances = cp.sum_distances;
        self.sum_sq_distances = cp.sum_sq_distances;
        self.kappa_posterior_mean = cp.kappa_posterior_mean;
        // Restore cumulative count (backward-compat: 0 from old checkpoints, fall back to rolling)
        self.update_count = if cp.total_observations > 0 {
            cp.total_observations
        } else {
            cp.observation_count
        };
        // Recompute derived fields
        let post_alpha = self.prior_alpha + self.observation_count as f64;
        let post_beta = self.prior_beta + self.sum_distances;
        self.kappa_posterior_var = post_alpha / (post_beta * post_beta);
        self.kappa_posterior_std = self.kappa_posterior_var.sqrt();
        if self.observation_count > 0 {
            self.mean_distance = self.sum_distances / self.observation_count as f64;
        }
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

// ============================================================================
// Statistical Helper Functions
// ============================================================================

/// Chi-squared quantile approximation (Wilson-Hilferty transformation)
fn chi_squared_quantile(df: f64, p: f64) -> f64 {
    if df <= 0.0 || p <= 0.0 || p >= 1.0 {
        return df; // fallback
    }

    // Standard normal quantile
    let z = normal_quantile(p);

    // Wilson-Hilferty approximation for chi-squared quantiles
    let term = 1.0 - 2.0 / (9.0 * df) + z * (2.0 / (9.0 * df)).sqrt();
    df * term.powi(3).max(0.0)
}

/// Standard normal quantile (Beasley-Springer-Moro approximation)
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return -8.0;
    }
    if p >= 1.0 {
        return 8.0;
    }

    let p_adj = if p < 0.5 { p } else { 1.0 - p };
    let t = (-2.0 * p_adj.ln()).sqrt();

    // Rational approximation
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 {
        -z
    } else {
        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v2_conjugate_update() {
        let mut kappa = BayesianKappaEstimator::new(500.0, 10.0, 60_000);

        // Prior mean = 500 (α=10, β=0.02)
        assert!((kappa.posterior_mean() - 500.0).abs() < 1.0);

        // Add observation at distance 0.001 (10 bps)
        kappa.on_trade(1000, 100.001, 1.0, 100.0);

        // V2: α should be 10 + 1 = 11 (COUNT not volume)
        assert_eq!(kappa.observation_count(), 1);

        // Mean should update based on new observation
        // Posterior mean = (α₀ + n) / (β₀ + Σδ)
        // = (10 + 1) / (0.02 + 0.00001) = 11 / 0.02001 ≈ 549.7
        let expected_mean = 11.0 / (0.02 + 0.00001);
        assert!((kappa.posterior_mean() - expected_mean).abs() < 1.0);
    }

    #[test]
    fn test_v2_count_not_volume() {
        let mut kappa = BayesianKappaEstimator::new(100.0, 5.0, 60_000);

        // Add 3 observations with different volumes
        kappa.on_trade(1000, 100.001, 10.0, 100.0); // Large volume
        kappa.on_trade(2000, 100.002, 0.1, 100.0); // Small volume
        kappa.on_trade(3000, 100.003, 5.0, 100.0); // Medium volume

        // V2: observation_count should be 3 (COUNT), not 15.1 (sum of volumes)
        assert_eq!(kappa.observation_count(), 3);

        // Effective sample size = observations + prior_alpha = 3 + 5 = 8
        assert!((kappa.effective_sample_size() - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_v2_window_expiry() {
        let mut kappa = BayesianKappaEstimator::new(500.0, 10.0, 1000); // 1 second window

        kappa.on_trade(0, 100.001, 1.0, 100.0);
        kappa.on_trade(500, 100.001, 1.0, 100.0);
        assert_eq!(kappa.observation_count(), 2);

        // This should expire the first observation
        kappa.on_trade(1500, 100.001, 1.0, 100.0);
        assert_eq!(kappa.observation_count(), 2); // 500 and 1500 remain
    }

    #[test]
    fn test_credible_interval() {
        let mut kappa = BayesianKappaEstimator::new(500.0, 20.0, 60_000);

        // Add some observations
        for i in 0..20 {
            kappa.on_trade(i * 100, 100.0 + (i as f64 * 0.0001), 1.0, 100.0);
        }

        let lower = kappa.ci_95_lower();
        let upper = kappa.ci_95_upper();
        let mean = kappa.posterior_mean();

        assert!(lower < mean);
        assert!(upper > mean);
        assert!(lower > 0.0);
    }

    #[test]
    fn test_normal_quantile() {
        // Check some known values
        assert!((normal_quantile(0.5) - 0.0).abs() < 0.01);
        assert!((normal_quantile(0.975) - 1.96).abs() < 0.1);
        assert!((normal_quantile(0.025) - (-1.96)).abs() < 0.1);
    }

    #[test]
    fn test_book_structure_imbalance() {
        let mut est = BookStructureEstimator::new();

        // More bids than asks
        let bids = vec![(99.0, 10.0), (98.0, 5.0)];
        let asks = vec![(101.0, 5.0), (102.0, 2.0)];
        est.update(&bids, &asks, 100.0);

        // Should show positive imbalance (more bids)
        assert!(est.imbalance() > 0.0);
    }
}
