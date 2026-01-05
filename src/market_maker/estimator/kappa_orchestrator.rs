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

/// Minimum weight for prior (always contributes some regularization)
const PRIOR_MIN_WEIGHT: f64 = 0.05;

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
            prior_kappa: 2000.0,
            prior_strength: 10.0,
            robust_nu: 4.0,
            robust_window_ms: 300_000, // 5 min
            own_fill_window_ms: 300_000,
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
            prior_strength: 20.0, // Strong prior to resist collapse
            robust_nu: 3.0,       // Heavy tails for outlier resistance
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
            prior_kappa: 1500.0,        // Target ~18 bps total (1/1500 + fees)
            prior_strength: 15.0,       // Moderate prior confidence
            robust_nu: 3.0,             // Heavy tails for outlier resistance
            robust_window_ms: 600_000,  // 10 min window
            own_fill_window_ms: 600_000,
            use_book_kappa: false,      // Books too thin for reliable regression
            use_robust_kappa: true,
        }
    }
}

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
}

impl KappaOrchestrator {
    /// Create a new kappa orchestrator with the given configuration.
    pub(crate) fn new(config: KappaOrchestratorConfig) -> Self {
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
            prior_kappa: config.prior_kappa,
            config,
            update_count: 0,
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

    /// Get effective κ using confidence-weighted blending.
    ///
    /// # Blending Formula (Warmup-Aware)
    ///
    /// During warmup (no own fills): Use ONLY prior.
    /// - Book and robust estimators are disabled to prevent unstable estimates
    /// - This ensures κ stays at prior (500) until we have real fill data
    ///
    /// Post-warmup: Blend own-fill κ, book κ, robust κ, and prior.
    ///
    /// As own-fill confidence grows, it dominates the estimate.
    pub(crate) fn kappa_effective(&self) -> f64 {
        // CRITICAL: Warmup detection uses observation count, NOT posterior confidence
        // own_kappa.confidence() returns 93% just from tight prior, NOT from data
        let min_own_fills = 5;
        let has_own_fills = self.own_kappa.observation_count() >= min_own_fills;

        // During warmup: ONLY use prior - disable own, book, and robust
        // This prevents book regression (which can give low κ on sparse books) from
        // dragging down the estimate before we have fill data to anchor it
        if !has_own_fills {
            return self.config.prior_kappa;
        }

        // Post-warmup: full confidence-weighted blending
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

        // Normalize confidences (prior always gets minimum weight)
        let total = own_conf + book_conf + robust_conf + PRIOR_MIN_WEIGHT;

        let w_own = own_conf / total;
        let w_book = book_conf / total;
        let w_robust = robust_conf / total;
        let w_prior = PRIOR_MIN_WEIGHT / total;

        let kappa = w_own * self.own_kappa.posterior_mean()
            + w_book * self.book_kappa.kappa()
            + w_robust * self.robust_kappa.kappa()
            + w_prior * self.prior_kappa;

        kappa.clamp(50.0, 10000.0)
    }

    /// Get individual component κ values and weights for diagnostics.
    /// Returns ((kappa, weight) for own, book, robust, prior) and warmup status.
    pub(crate) fn component_breakdown(
        &self,
    ) -> (
        (f64, f64),
        (f64, f64),
        (f64, f64),
        (f64, f64), // (kappa, weight) for own, book, robust, prior
        bool,       // is_warmup (all estimators disabled, only prior used)
    ) {
        // Mirror the warmup logic from kappa_effective exactly
        let min_own_fills = 5;
        let has_own_fills = self.own_kappa.observation_count() >= min_own_fills;
        let is_warmup = !has_own_fills;

        // During warmup: all weights are 0 except prior (100%)
        if is_warmup {
            return (
                (self.own_kappa.posterior_mean(), 0.0),  // own disabled
                (self.book_kappa.kappa(), 0.0),          // book disabled
                (self.robust_kappa.kappa(), 0.0),        // robust disabled
                (self.config.prior_kappa, 1.0),          // prior = 100%
                true,                                     // is_warmup
            );
        }

        // Post-warmup: full blending
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

        let total = own_conf + book_conf + robust_conf + PRIOR_MIN_WEIGHT;

        (
            (self.own_kappa.posterior_mean(), own_conf / total),
            (self.book_kappa.kappa(), book_conf / total),
            (self.robust_kappa.kappa(), robust_conf / total),
            (self.config.prior_kappa, PRIOR_MIN_WEIGHT / total),
            false,  // not warmup
        )
    }

    /// Feed L2 book update to book-structure estimator.
    pub(crate) fn on_l2_book(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)], mid: f64) {
        if self.config.use_book_kappa {
            self.book_kappa.on_l2_book(bids, asks, mid);
        }
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

        self.update_count += 1;

        // Log periodically (every 10 trades for diagnostics)
        if self.update_count % 10 == 0 {
            let ((k_own, w_own), (k_book, w_book), (k_robust, w_robust), (k_prior, w_prior), is_warmup) =
                self.component_breakdown();

            info!(
                kappa_effective = %format!("{:.0}", self.kappa_effective()),
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
    }

    /// Record fill distance directly (when we know the distance already).
    #[allow(dead_code)] // API completeness for future use
    pub(crate) fn record_own_fill_distance(&mut self, timestamp_ms: u64, distance_bps: f64) {
        let distance = distance_bps / 10000.0; // Convert bps to fraction
        // Create synthetic trade at that distance
        // Using 1.0 as mid and 1.0 + distance as price
        self.own_kappa.on_trade(timestamp_ms, 1.0 + distance, 1.0, 1.0);
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

        // Book kappa should have some weight now (after warmup)
        let (_, (_, w_book), _, _, is_warmup) = orch.component_breakdown();
        assert!(
            !is_warmup,
            "Should not be in warmup mode after providing own fills"
        );
        assert!(
            w_book > 0.1,
            "Book should have significant weight after updates, got {}",
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

        // Own-fill should dominate
        let ((_, w_own), _, _, _, _) = orch.component_breakdown();
        assert!(
            w_own > 0.5,
            "Own-fill should dominate after many fills, got weight {}",
            w_own
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
}
