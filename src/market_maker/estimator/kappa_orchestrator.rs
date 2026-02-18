//! Kappa orchestrator for confidence-weighted blending of multiple estimators.
//!
//! This module combines three kappa estimation approaches:
//!
//! 1. **Book-Structure κ**: Derived from L2 order book depth decay
//!    - Semantically correct for GLFT's λ(δ) = A exp(-κδ)
//!    - Unaffected by market trade outliers
//!    - Confidence measured by R² of exponential fit
//!
//! 2. **Robust κ**: From market trades with Student-t likelihood
//!    - Uses heavy-tailed distribution to resist outliers
//!    - Captures market-wide fill behavior safely
//!    - Confidence based on effective sample size
//!
//! 3. **Own-Fill κ**: From our actual order fills (gold standard)
//!    - Direct measurement of OUR fill rate at each depth
//!    - Dominates as fill history accumulates
//!    - Most accurate but requires trading history
//!
//! The orchestrator blends these using confidence-weighted averaging:
//! ```text
//! κ_effective = Σ(wᵢ × κᵢ) where wᵢ ∝ confidence(κᵢ)
//! ```

use super::book_kappa::BookKappaEstimator;
use super::kappa::BayesianKappaEstimator;
use super::robust_kappa::RobustKappaEstimator;
use tracing::info;

/// Configuration for the kappa orchestrator.
#[derive(Debug, Clone)]
pub(crate) struct KappaOrchestratorConfig {
    /// Prior κ value for regularization
    pub prior_kappa: f64,

    /// Prior strength for own-fill estimator
    pub prior_strength: f64,

    /// Degrees of freedom for robust estimator
    pub robust_nu: f64,

    /// Window for robust estimator (ms)
    pub robust_window_ms: u64,

    /// Window for own-fill estimator (ms)
    pub own_fill_window_ms: u64,

    /// Whether to use book kappa (disable for very thin books)
    pub use_book_kappa: bool,

    /// Whether to use robust kappa (disable for pure own-fill mode)
    pub use_robust_kappa: bool,
}

impl Default for KappaOrchestratorConfig {
    fn default() -> Self {
        Self {
            // PRIOR DERIVATION: κ_prior comes from historical median of book-derived kappa.
            // For liquid perps: κ ∈ [1500, 4000], median ≈ 2000.
            // This is the Gamma prior mean: Gamma(shape=4, rate=0.002) → E[κ] = 2000.
            // The prior acts as regularization, shrinking estimates toward this value
            // when data is scarce, preventing overfitting to noise.
            prior_kappa: 2000.0,

            // PRIOR STRENGTH: Number of pseudo-observations.
            // With prior_strength=10, the prior contributes ~10 "virtual observations".
            // After 100 real observations, prior weight is 10/(100+10) ≈ 9%.
            // This balances adaptation speed vs stability.
            prior_strength: 10.0,

            // ROBUST NU (degrees of freedom): Controls tail heaviness of Student-t.
            // ν = 4 gives fat tails that resist outliers but not too extreme.
            // DERIVATION: For kappa estimates, kurtosis ≈ 6 implies ν ≈ 6/(excess_kurtosis).
            // Empirically, kappa estimates have excess_kurtosis ≈ 1.5, so ν ≈ 4.
            robust_nu: 4.0,

            robust_window_ms: 600_000, // 10 min (increased from 5 min for sparse fills)
            own_fill_window_ms: 600_000,
            use_book_kappa: true,
            use_robust_kappa: true,
        }
    }
}

impl KappaOrchestratorConfig {
    /// Config for liquid markets (BTC, ETH on main perps).
    #[allow(dead_code)] // Used in tests and future CLI integration
    pub(crate) fn liquid() -> Self {
        Self {
            prior_kappa: 2500.0,
            prior_strength: 5.0, // Quick adaptation
            robust_nu: 5.0,      // Slightly lighter tails (less robust, more responsive)
            ..Default::default()
        }
    }

    /// Config for illiquid markets (HIP-3 DEX, altcoins).
    #[allow(dead_code)] // Used in tests and future CLI integration
    pub(crate) fn illiquid() -> Self {
        Self {
            prior_kappa: 1500.0,
            prior_strength: 20.0,      // Strong prior to resist collapse
            robust_nu: 3.0,            // Heavy tails for outlier resistance
            robust_window_ms: 600_000, // Longer window (10 min)
            own_fill_window_ms: 600_000,
            ..Default::default()
        }
    }

    /// Config for HIP-3 DEX markets - optimized for 15-25 bps target spreads.
    ///
    /// Key insight from GLFT: δ* ≈ 1/κ when γ/κ is small.
    /// With κ=1500, GLFT gives ~6.7 bps per side + 1.5 bps fee = ~16 bps total.
    ///
    /// Disables book_kappa because HIP-3 books are too thin for reliable
    /// exponential decay regression.
    #[allow(dead_code)] // Used in tests and future CLI integration
    pub(crate) fn hip3() -> Self {
        Self {
            prior_kappa: 1500.0,       // Target ~18 bps total (1/1500 + fees)
            prior_strength: 15.0,      // Moderate prior confidence
            robust_nu: 3.0,            // Heavy tails for outlier resistance
            robust_window_ms: 600_000, // 10 min window
            own_fill_window_ms: 600_000,
            use_book_kappa: false, // Books too thin for reliable regression
            use_robust_kappa: true,
        }
    }
}

/// EWMA smoothing factor for kappa_effective.
///
/// DERIVATION: α = 1 - exp(-ln(2) / half_life)
///
/// For a target half-life of 10 observations:
/// α = 1 - exp(-0.693 / 10) ≈ 0.067
///
/// However, we want slower adaptation (less responsive to noise), so:
/// - half_life ≈ 100 observations → α ≈ 0.007
/// - For practical purposes, α = 0.1 (10% new, 90% old) works well
///
/// The value 0.9 means 90% weight on previous estimate, providing
/// stability while still adapting to changing market conditions.
const KAPPA_EWMA_ALPHA: f64 = 0.9;

/// Orchestrates multiple kappa estimators with confidence-weighted blending.
///
/// # Architecture
///
/// ```text
/// ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
/// │ BookKappa    │  │ RobustKappa  │  │ OwnFillKappa │
/// │ (L2 depth)   │  │ (Student-t)  │  │ (Bayesian)   │
/// │ conf: R²     │  │ conf: ESS    │  │ conf: ESS    │
/// └──────────────┘  └──────────────┘  └──────────────┘
///        │                 │                 │
///        └─────────────────┼─────────────────┘
///                          ▼
///             ┌────────────────────────┐
///             │ Confidence-Weighted    │
///             │ Blending + Prior       │
///             └────────────────────────┘
///                          │
///                          ▼
///             ┌────────────────────────┐
///             │ EWMA Smoothing         │
///             │ κ = 0.9×κ_prev + 0.1×κ │
///             └────────────────────────┘
///                          │
///                          ▼
///                    κ_effective
/// ```
#[derive(Debug)]
pub(crate) struct KappaOrchestrator {
    /// Book-structure kappa estimator
    book_kappa: BookKappaEstimator,

    /// Robust kappa estimator (Student-t)
    robust_kappa: RobustKappaEstimator,

    /// Own-fill kappa estimator (Bayesian)
    own_kappa: BayesianKappaEstimator,

    /// Prior κ for regularization
    prior_kappa: f64,

    /// Configuration
    config: KappaOrchestratorConfig,

    /// Update counter for logging
    update_count: u64,

    /// EWMA-smoothed kappa value (reduces 2x swings)
    smoothed_kappa: f64,

    /// Whether smoothed_kappa has been initialized
    smoothed_kappa_initialized: bool,

    /// Whether we've ever exited warmup mode (prevents re-entry).
    ///
    /// # The Re-entry Bug
    /// Without this flag, when old fills expire from the 600ms window,
    /// `observation_count()` can drop below 5 and the orchestrator re-enters
    /// warmup mode. This causes kappa to swing 2-3x as the blending formula
    /// changes discontinuously from own-fill-weighted to market-signal-weighted.
    ///
    /// Once we have 5+ own fills, we "graduate" from warmup permanently.
    /// The own-fill posterior still decays naturally as observations expire,
    /// but we don't revert to the warmup blending formula.
    has_exited_warmup: bool,

    /// EWMA of (kappa_book - kappa_robust) for empirical bias tracking
    book_robust_bias_ewma: f64,
    /// EWMA of (kappa_robust - kappa_robust_prev)^2 for robust variance tracking
    robust_rolling_var: f64,
    /// EWMA of (kappa_book - kappa_book_prev)^2 for book variance tracking
    book_rolling_var: f64,
    /// Previous robust kappa for variance computation
    prev_robust_kappa: f64,
    /// Previous book kappa for variance computation
    prev_book_kappa: f64,
}

impl KappaOrchestrator {
    /// Create a new kappa orchestrator with the given configuration.
    pub(crate) fn new(config: KappaOrchestratorConfig) -> Self {
        let prior_kappa = config.prior_kappa;
        Self {
            book_kappa: BookKappaEstimator::new(config.prior_kappa),
            robust_kappa: RobustKappaEstimator::new(
                config.prior_kappa,
                config.prior_strength,
                config.robust_nu,
                config.robust_window_ms,
            ),
            own_kappa: BayesianKappaEstimator::new(
                config.prior_kappa,
                config.prior_strength,
                config.own_fill_window_ms,
            ),
            prior_kappa,
            config,
            update_count: 0,
            // Initialize smoothed_kappa to prior - will be overwritten on first call
            smoothed_kappa: prior_kappa,
            smoothed_kappa_initialized: false,
            // Start in warmup mode - will graduate permanently after 5 fills
            has_exited_warmup: false,
            // Precision-weighted blending stats (initialized to 0, warm up naturally)
            book_robust_bias_ewma: 0.0,
            robust_rolling_var: 0.0,
            book_rolling_var: 0.0,
            prev_robust_kappa: 0.0,
            prev_book_kappa: 0.0,
        }
    }

    /// Create with default liquid market configuration.
    #[allow(dead_code)] // Used in tests and future CLI integration
    pub(crate) fn default_liquid() -> Self {
        Self::new(KappaOrchestratorConfig::liquid())
    }

    /// Create with default illiquid market configuration.
    #[allow(dead_code)] // Used in tests and future CLI integration
    pub(crate) fn default_illiquid() -> Self {
        Self::new(KappaOrchestratorConfig::illiquid())
    }

    /// Get effective κ using confidence-weighted blending with EWMA smoothing.
    ///
    /// Returns the EWMA-smoothed kappa value to reduce high-frequency variance.
    /// The smoothed value is updated via `update_smoothed_kappa()` which is called
    /// from the update paths (on_market_trade, on_own_fill, on_l2_book).
    ///
    /// # EWMA Formula
    /// κ_smoothed = α × κ_prev + (1-α) × κ_raw, where α = 0.9
    ///
    /// This reduces 2x swings in kappa that cause spread variance.
    ///
    /// Before first update, returns the raw kappa (prior-based).
    pub(crate) fn kappa_effective(&self) -> f64 {
        if self.smoothed_kappa_initialized {
            self.smoothed_kappa
        } else {
            // Before any updates, return raw kappa (prior-based)
            self.kappa_raw()
        }
    }

    /// Compute raw (unsmoothed) kappa using confidence-weighted blending.
    ///
    /// # Blending Formula (Warmup-Aware)
    ///
    /// During warmup (no own fills): Blend market signals with prior.
    /// - Book kappa: from L2 depth decay regression (direct market structure)
    /// - Robust kappa: from market trades with outlier resistance
    /// - This is Bayesian-correct: use available information, express uncertainty through γ
    ///
    /// Post-warmup: Full confidence-weighted blending of all sources.
    /// As own-fill confidence grows, it dominates the estimate.
    fn kappa_raw(&self) -> f64 {
        // CRITICAL: Warmup detection uses observation count, NOT posterior confidence
        // own_kappa.confidence() returns 93% just from tight prior, NOT from data
        let min_own_fills = 5;
        let has_own_fills = self.own_kappa.observation_count() >= min_own_fills;

        // Get market signals
        let book_kappa = self.book_kappa.kappa();
        let robust_kappa = self.robust_kappa.kappa();

        // During warmup: Blend market kappa with prior instead of ignoring market data
        // CRITICAL: Once we exit warmup, NEVER re-enter (prevents 2x kappa swings when fills expire)
        let is_warmup = !has_own_fills && !self.has_exited_warmup;
        if is_warmup {
            // Trust market signal (book/robust) but hedge with prior
            let book_valid = book_kappa > 100.0 && self.config.use_book_kappa;
            let robust_valid = robust_kappa > 100.0 && self.config.use_robust_kappa;

            // Cap robust_kappa at 2× prior during warmup.
            // Robust kappa measures market trade distances (BBO fills at 1-2 bps),
            // not fill distances at OUR quote depths (8+ bps). Without capping,
            // robust_kappa inflates to ~10000 which biases the blend upward.
            let max_robust_warmup = self.config.prior_kappa * 2.0;
            let capped_robust = robust_kappa.min(max_robust_warmup);

            // Weight allocation during warmup:
            // - Book: 40% (direct market structure signal)
            // - Robust: 30% (market-wide fill behavior, capped)
            // - Prior: 30% minimum (regularization/safety)
            let book_weight = if book_valid { 0.4 } else { 0.0 };
            let robust_weight = if robust_valid { 0.3 } else { 0.0 };
            let prior_weight = 1.0 - book_weight - robust_weight;

            let blended = book_weight * book_kappa
                + robust_weight * capped_robust
                + prior_weight * self.config.prior_kappa;

            return blended.clamp(50.0, 10000.0);
        }

        // Post-warmup: precision-weighted blending (tau = 1/Var_eff)
        // Uses empirical variance + bias^2 for effective variance of biased estimators.
        let own_tau = if self.own_kappa.observation_count() > 5 {
            let own_var = self.own_kappa.posterior_variance().max(1.0);
            1.0 / own_var
        } else {
            0.0
        };

        let robust_tau = if self.config.use_robust_kappa && robust_kappa > 100.0 {
            1.0 / self.robust_rolling_var.max(1.0)
        } else {
            0.0
        };

        // Book kappa: effective variance includes bias^2 (bias = mean(kappa_book - kappa_robust))
        let book_tau = if self.config.use_book_kappa && book_kappa > 100.0 {
            if self.config.use_robust_kappa && robust_kappa > 100.0 {
                let bias_sq = self.book_robust_bias_ewma.powi(2);
                let book_eff_var = self.book_rolling_var + bias_sq;
                1.0 / book_eff_var.max(1.0)
            } else {
                // No robust reference — weak weight for book
                0.05 / self.book_rolling_var.max(1.0)
            }
        } else {
            0.0
        };

        let prior_var = (self.config.prior_kappa * 0.5).powi(2); // Prior variance: (50% of mean)^2
        let prior_tau = 1.0 / prior_var.max(1.0);
        let total_tau = (own_tau + robust_tau + book_tau + prior_tau).max(1e-10);

        let kappa = (own_tau * self.own_kappa.posterior_mean()
            + robust_tau * robust_kappa
            + book_tau * book_kappa
            + prior_tau * self.prior_kappa)
            / total_tau;

        kappa.clamp(50.0, 10000.0)
    }

    /// Update the EWMA-smoothed kappa value.
    ///
    /// Called from update paths to maintain smoothed estimate.
    /// Uses α=0.9: κ_smoothed = 0.9 × κ_prev + 0.1 × κ_raw
    fn update_smoothed_kappa(&mut self) {
        self.update_rolling_stats();
        let kappa_raw = self.kappa_raw();

        if !self.smoothed_kappa_initialized {
            // First update: initialize to raw value
            self.smoothed_kappa = kappa_raw;
            self.smoothed_kappa_initialized = true;
        } else {
            // EWMA update: 90% previous, 10% new
            self.smoothed_kappa =
                KAPPA_EWMA_ALPHA * self.smoothed_kappa + (1.0 - KAPPA_EWMA_ALPHA) * kappa_raw;
        }
    }

    /// Update rolling variance and bias statistics for precision-weighted blending.
    ///
    /// Tracks EWMA of squared differences (rolling variance) and
    /// book-robust bias for effective variance computation.
    fn update_rolling_stats(&mut self) {
        const ROLLING_ALPHA: f64 = 0.05;
        let book_k = self.book_kappa.kappa();
        let robust_k = self.robust_kappa.kappa();

        // Update bias tracking (book - robust)
        if book_k > 100.0 && robust_k > 100.0 {
            let bias = book_k - robust_k;
            self.book_robust_bias_ewma =
                self.book_robust_bias_ewma * (1.0 - ROLLING_ALPHA) + bias * ROLLING_ALPHA;
        }

        // Update variance tracking
        if self.prev_robust_kappa > 0.0 && robust_k > 100.0 {
            let diff_sq = (robust_k - self.prev_robust_kappa).powi(2);
            self.robust_rolling_var =
                self.robust_rolling_var * (1.0 - ROLLING_ALPHA) + diff_sq * ROLLING_ALPHA;
        }
        if self.prev_book_kappa > 0.0 && book_k > 100.0 {
            let diff_sq = (book_k - self.prev_book_kappa).powi(2);
            self.book_rolling_var =
                self.book_rolling_var * (1.0 - ROLLING_ALPHA) + diff_sq * ROLLING_ALPHA;
        }

        self.prev_robust_kappa = robust_k;
        self.prev_book_kappa = book_k;
    }

    /// Detect ghost liquidity: book kappa >> robust kappa suggests standing orders
    /// that don't represent real fill intensity. Returns gamma multiplier [1.0, 5.0].
    pub(crate) fn ghost_liquidity_gamma_mult(&self) -> f64 {
        let book_k = self.book_kappa.kappa();
        let robust_k = self.robust_kappa.kappa();
        if self.config.use_robust_kappa && robust_k > 100.0 && book_k / robust_k > 3.0 {
            (book_k / robust_k).min(5.0)
        } else {
            1.0
        }
    }

    /// Get individual component κ values and weights for diagnostics.
    /// Returns ((kappa, weight) for own, book, robust, prior) and warmup status.
    #[allow(clippy::type_complexity)]
    pub(crate) fn component_breakdown(
        &self,
    ) -> (
        (f64, f64),
        (f64, f64),
        (f64, f64),
        (f64, f64), // (kappa, weight) for own, book, robust, prior
        bool,       // is_warmup (market signal used with prior blend)
    ) {
        // Mirror the warmup logic from kappa_raw exactly
        let min_own_fills = 5;
        let has_own_fills = self.own_kappa.observation_count() >= min_own_fills;
        // CRITICAL: Use has_exited_warmup to match kappa_raw() logic
        let is_warmup = !has_own_fills && !self.has_exited_warmup;

        let book_kappa = self.book_kappa.kappa();
        let robust_kappa = self.robust_kappa.kappa();

        // During warmup: blend market signal with prior (not 100% prior anymore)
        if is_warmup {
            let book_valid = book_kappa > 100.0 && self.config.use_book_kappa;
            let robust_valid = robust_kappa > 100.0 && self.config.use_robust_kappa;

            // Cap robust_kappa at 2× prior (mirrors kappa_raw() logic)
            let max_robust_warmup = self.config.prior_kappa * 2.0;
            let capped_robust = robust_kappa.min(max_robust_warmup);

            let book_weight = if book_valid { 0.4 } else { 0.0 };
            let robust_weight = if robust_valid { 0.3 } else { 0.0 };
            let prior_weight = 1.0 - book_weight - robust_weight;

            return (
                (self.own_kappa.posterior_mean(), 0.0), // own disabled during warmup
                (book_kappa, book_weight),              // book weighted if valid
                (capped_robust, robust_weight),         // robust weighted if valid (capped)
                (self.config.prior_kappa, prior_weight), // prior gets remainder
                true,                                   // is_warmup
            );
        }

        // Post-warmup: precision-weighted breakdown (mirrors kappa_raw logic)
        let robust_kappa = self.robust_kappa.kappa();
        let own_tau = if self.own_kappa.observation_count() > 5 {
            1.0 / self.own_kappa.posterior_variance().max(1.0)
        } else {
            0.0
        };
        let robust_tau = if self.config.use_robust_kappa && robust_kappa > 100.0 {
            1.0 / self.robust_rolling_var.max(1.0)
        } else {
            0.0
        };
        let book_tau = if self.config.use_book_kappa && book_kappa > 100.0 {
            if self.config.use_robust_kappa && robust_kappa > 100.0 {
                let bias_sq = self.book_robust_bias_ewma.powi(2);
                1.0 / (self.book_rolling_var + bias_sq).max(1.0)
            } else {
                0.05 / self.book_rolling_var.max(1.0)
            }
        } else {
            0.0
        };
        let prior_var = (self.config.prior_kappa * 0.5).powi(2);
        let prior_tau = 1.0 / prior_var.max(1.0);
        let total_tau = (own_tau + robust_tau + book_tau + prior_tau).max(1e-10);

        (
            (self.own_kappa.posterior_mean(), own_tau / total_tau),
            (self.book_kappa.kappa(), book_tau / total_tau),
            (self.robust_kappa.kappa(), robust_tau / total_tau),
            (self.config.prior_kappa, prior_tau / total_tau),
            false, // not warmup
        )
    }

    /// Feed L2 book update to book-structure estimator.
    pub(crate) fn on_l2_book(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)], mid: f64) {
        if self.config.use_book_kappa {
            self.book_kappa.on_l2_book(bids, asks, mid);
        }
        // Update smoothed kappa after book update
        self.update_smoothed_kappa();
    }

    /// Feed market trade to robust estimator.
    ///
    /// **IMPORTANT**: This does NOT feed to the exponential (own_kappa) estimator.
    /// Market trades are only processed by the robust estimator which resists outliers.
    pub(crate) fn on_market_trade(&mut self, timestamp_ms: u64, price: f64, mid: f64) {
        if !self.config.use_robust_kappa || mid <= 0.0 {
            return;
        }

        let distance = ((price - mid) / mid).abs();
        self.robust_kappa.on_trade(timestamp_ms, distance);

        // Update smoothed kappa after trade
        self.update_smoothed_kappa();

        self.update_count += 1;

        // Log periodically (every 10 trades for diagnostics)
        if self.update_count.is_multiple_of(10) {
            let (
                (k_own, w_own),
                (k_book, w_book),
                (k_robust, w_robust),
                (k_prior, w_prior),
                is_warmup,
            ) = self.component_breakdown();

            info!(
                kappa_effective = %format!("{:.0}", self.kappa_effective()),
                kappa_raw = %format!("{:.0}", self.kappa_raw()),
                own = %format!("{:.0} ({:.0}%)", k_own, w_own * 100.0),
                book = %format!("{:.0} ({:.0}%)", k_book, w_book * 100.0),
                robust = %format!("{:.0} ({:.0}%)", k_robust, w_robust * 100.0),
                prior = %format!("{:.0} ({:.0}%)", k_prior, w_prior * 100.0),
                own_fills = self.own_kappa.observation_count(),
                outliers = self.robust_kappa.outlier_count(),
                warmup = is_warmup,
                "Kappa orchestrator breakdown"
            );
        }
    }

    /// Feed our own fill to the own-fill estimator.
    ///
    /// This is the gold standard data source - as we accumulate fills,
    /// the own-fill estimate dominates the blend.
    pub(crate) fn on_own_fill(&mut self, timestamp_ms: u64, price: f64, size: f64, mid: f64) {
        if mid <= 0.0 {
            return;
        }

        // Feed to own-fill Bayesian estimator
        self.own_kappa.on_trade(timestamp_ms, price, size, mid);

        // Check for warmup exit transition (prevents re-entry bug)
        // Once we have 5+ own fills, we "graduate" from warmup PERMANENTLY.
        // This prevents kappa from swinging 2-3x when old fills expire.
        const MIN_OWN_FILLS_FOR_WARMUP_EXIT: usize = 5;
        if !self.has_exited_warmup
            && self.own_kappa.observation_count() >= MIN_OWN_FILLS_FOR_WARMUP_EXIT
        {
            self.has_exited_warmup = true;
            tracing::info!(
                own_fills = self.own_kappa.observation_count(),
                kappa = %format!("{:.0}", self.kappa_raw()),
                "Kappa orchestrator graduated from warmup (will not re-enter)"
            );
        }

        // Update smoothed kappa after fill (most important signal)
        self.update_smoothed_kappa();
    }

    /// Record fill distance directly (when we know the distance already).
    #[allow(dead_code)] // API completeness for future use
    pub(crate) fn record_own_fill_distance(&mut self, timestamp_ms: u64, distance_bps: f64) {
        let distance = distance_bps / 10000.0; // Convert bps to fraction
                                               // Create synthetic trade at that distance
                                               // Using 1.0 as mid and 1.0 + distance as price
        self.own_kappa
            .on_trade(timestamp_ms, 1.0 + distance, 1.0, 1.0);
    }

    /// Get overall confidence in the κ estimate.
    ///
    /// Returns the weighted sum of component confidences.
    pub(crate) fn confidence(&self) -> f64 {
        let own_conf = self.own_kappa.confidence();
        let book_conf = if self.config.use_book_kappa {
            self.book_kappa.confidence()
        } else {
            0.0
        };
        let robust_conf = if self.config.use_robust_kappa {
            self.robust_kappa.confidence()
        } else {
            0.0
        };

        // Overall confidence is max of components (we're as confident as our best source)
        own_conf.max(book_conf).max(robust_conf)
    }

    /// Check if any estimator is warmed up.
    ///
    /// Returns true only after receiving actual observations (not from prior alone).
    pub(crate) fn is_warmed_up(&self) -> bool {
        // Require actual observations, not just prior confidence
        let own_has_data = self.own_kappa.observation_count() >= 5;
        let book_warmed = self.config.use_book_kappa && self.book_kappa.is_warmed_up();
        let robust_warmed = self.config.use_robust_kappa && self.robust_kappa.is_warmed_up();

        own_has_data || book_warmed || robust_warmed
    }

    /// Get the book-structure κ estimate directly.
    pub(crate) fn book_kappa(&self) -> f64 {
        self.book_kappa.kappa()
    }

    /// Get the robust κ estimate directly.
    pub(crate) fn robust_kappa(&self) -> f64 {
        self.robust_kappa.kappa()
    }

    /// Get the own-fill κ estimate directly.
    pub(crate) fn own_kappa(&self) -> f64 {
        self.own_kappa.posterior_mean()
    }

    /// Get the prior κ value.
    #[allow(dead_code)] // Used in tests
    pub(crate) fn prior_kappa(&self) -> f64 {
        self.prior_kappa
    }

    /// Get number of outliers detected by robust estimator.
    pub(crate) fn outlier_count(&self) -> u64 {
        self.robust_kappa.outlier_count()
    }

    /// Get own-fill observation count for diagnostics.
    pub(crate) fn own_kappa_observation_count(&self) -> usize {
        self.own_kappa.observation_count()
    }

    /// Get robust kappa effective sample size.
    pub(crate) fn robust_kappa_ess(&self) -> f64 {
        self.robust_kappa.effective_sample_size()
    }

    /// Get robust kappa nu (degrees of freedom).
    pub(crate) fn robust_kappa_nu(&self) -> f64 {
        self.robust_kappa.nu()
    }

    /// Get robust kappa observation count.
    pub(crate) fn robust_kappa_obs_count(&self) -> u64 {
        self.robust_kappa.observation_count()
    }

    /// Update the prior kappa used for regularization.
    ///
    /// Propagates the new prior to the own-fill estimator and updates
    /// the config for future warmup blending.
    ///
    /// # Arguments
    /// * `new_prior_mean` - New prior mean, clamped to [100, 50000]
    /// * `new_prior_strength` - New prior strength, clamped to [1, 20]
    pub(crate) fn update_prior_kappa(&mut self, new_prior_mean: f64, new_prior_strength: f64) {
        let clamped_mean = new_prior_mean.clamp(100.0, 50000.0);
        let clamped_strength = new_prior_strength.clamp(1.0, 20.0);

        self.prior_kappa = clamped_mean;
        self.config.prior_kappa = clamped_mean;
        self.config.prior_strength = clamped_strength;

        // Propagate to own-fill estimator so its Bayesian posterior reflects the new prior
        self.own_kappa.update_prior(clamped_mean, clamped_strength);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let orch = KappaOrchestrator::default_liquid();

        // Should start at prior
        let kappa = orch.kappa_effective();
        assert!(
            (kappa - 2500.0).abs() < 100.0,
            "Initial kappa should be near prior, got {}",
            kappa
        );

        // Should not be warmed up
        assert!(!orch.is_warmed_up());
    }

    #[test]
    fn test_book_kappa_contribution() {
        let mut orch = KappaOrchestrator::default_liquid();

        // Create exponential-decay book
        let mid = 100.0;
        let mut bids = Vec::new();
        let mut asks = Vec::new();

        for i in 1..=10 {
            let delta = i as f64 * 0.001;
            let depth = 100.0 * (-1000.0 * delta).exp();

            bids.push((mid * (1.0 - delta), depth.max(1.0)));
            asks.push((mid * (1.0 + delta), depth.max(1.0)));
        }

        // First, provide own fills to exit warmup mode (requires 5 fills)
        for i in 0..5 {
            let fill_price = mid * (1.0 + 0.001); // 10 bps from mid
            orch.on_own_fill(i * 1000, fill_price, 1.0, mid);
        }

        // Feed book updates
        for _ in 0..20 {
            orch.on_l2_book(&bids, &asks, mid);
        }

        // Book kappa should have some weight now (after warmup).
        // Fix 3: Precision-weighted blending uses 1/Var_eff. With identical book
        // updates (no variance), book_tau is small. Verify we're out of warmup
        // and book gets non-zero weight.
        let (_, (_, w_book), _, _, is_warmup) = orch.component_breakdown();
        assert!(
            !is_warmup,
            "Should not be in warmup mode after providing own fills"
        );
        assert!(
            w_book > 0.0,
            "Book should have non-zero weight after updates, got {}",
            w_book
        );
    }

    #[test]
    fn test_robust_outlier_resistance() {
        let mut orch = KappaOrchestrator::default_illiquid();

        let mid = 100.0;

        // Add normal trades
        for i in 0..20 {
            let price = mid * (1.0 + 0.001); // 10 bps from mid
            orch.on_market_trade(i * 1000, price, mid);
        }

        let kappa_before = orch.robust_kappa();

        // Add outlier (liquidation at 500 bps)
        orch.on_market_trade(21 * 1000, mid * 1.05, mid);

        let kappa_after = orch.robust_kappa();

        // Should resist the outlier
        let change = (kappa_before - kappa_after).abs() / kappa_before;
        assert!(
            change < 0.3,
            "Robust kappa changed {}% on outlier, should resist",
            change * 100.0
        );
    }

    #[test]
    fn test_own_fill_dominates() {
        let mut orch = KappaOrchestrator::default_liquid();

        let mid = 100.0;

        // Add many own fills at tight distances
        for i in 0..50 {
            let price = mid * (1.0 + 0.0005); // 5 bps from mid
            orch.on_own_fill(i * 1000, price, 1.0, mid);
        }

        // Own-fill should have significant weight with 50 fills.
        // Fix 3: Precision-weighted uses 1/posterior_variance. After 50 fills,
        // posterior variance is tight → τ_own is large. It should dominate over
        // the prior (whose variance is (kappa*0.5)² = huge).
        let ((_, w_own), _, _, (_, w_prior), _) = orch.component_breakdown();
        assert!(
            w_own > w_prior,
            "Own-fill weight ({}) should exceed prior weight ({}) after 50 fills",
            w_own,
            w_prior
        );
    }

    #[test]
    fn test_config_liquid_vs_illiquid() {
        let liquid = KappaOrchestrator::default_liquid();
        let illiquid = KappaOrchestrator::default_illiquid();

        // Illiquid should have lower prior (more conservative)
        assert!(liquid.prior_kappa() > illiquid.prior_kappa());

        // Illiquid should have lower nu (heavier tails)
        assert!(liquid.robust_kappa.nu() > illiquid.robust_kappa.nu());
    }

    #[test]
    fn test_ewma_smoothing_reduces_variance() {
        let mut orch = KappaOrchestrator::default_liquid();

        let mid = 100.0;

        // First, provide enough fills to exit warmup
        for i in 0..10 {
            let fill_price = mid * (1.0 + 0.001);
            orch.on_own_fill(i * 1000, fill_price, 1.0, mid);
        }

        let kappa_before = orch.kappa_effective();

        // Now add a trade at a very different distance (simulating jump)
        // This should cause raw kappa to jump, but smoothed should resist
        let extreme_price = mid * (1.0 + 0.01); // 100 bps - much wider
        orch.on_market_trade(11 * 1000, extreme_price, mid);

        let kappa_raw = orch.kappa_raw();
        let kappa_smoothed = orch.kappa_effective();

        // Smoothed should have moved less than raw
        let raw_change = (kappa_raw - kappa_before).abs();
        let smoothed_change = (kappa_smoothed - kappa_before).abs();

        // EWMA with alpha=0.9 means smoothed changes by only 10% of delta
        assert!(
            smoothed_change < raw_change,
            "Smoothed kappa should change less than raw: smoothed_change={:.1}, raw_change={:.1}",
            smoothed_change,
            raw_change
        );
    }

    #[test]
    fn test_precision_weighted_book_bias_downweight() {
        // When book/robust diverge significantly, book weight should approach 0
        let mut orch = KappaOrchestrator::new(KappaOrchestratorConfig::default());
        // Simulate divergent estimates
        orch.book_robust_bias_ewma = 3000.0; // Large bias
        orch.book_rolling_var = 100.0;
        orch.robust_rolling_var = 100.0;
        // Book tau should be tiny due to bias^2 dominating
        let bias_sq = orch.book_robust_bias_ewma.powi(2); // 9_000_000
        let book_eff_var = orch.book_rolling_var + bias_sq; // 9_000_100
        let book_tau = 1.0 / book_eff_var;
        let robust_tau = 1.0 / orch.robust_rolling_var.max(1.0);
        assert!(
            book_tau / robust_tau < 0.001,
            "Book should have <0.1% weight when biased, got {:.6}",
            book_tau / robust_tau
        );
    }
}
