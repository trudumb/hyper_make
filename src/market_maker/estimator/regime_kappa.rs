//! Regime-Conditioned Kappa Estimation.
//!
//! Extends the kappa orchestrator to maintain separate estimates per volatility regime.
//! This addresses the observation that fill intensity varies significantly across market
//! conditions - κ can be 10x higher in quiet markets vs. cascade conditions.
//!
//! # Architecture
//!
//! ```text
//! Market Data
//!     │
//!     ├── Fills → Regime-Tagged Fill Buffer
//!     │
//!     ├── Regime Detection (HMM)
//!     │           │
//!     │           ▼
//!     │   ┌──────────────────────────────────────┐
//!     │   │ Regime Probabilities                 │
//!     │   │ P(Low), P(Normal), P(High), P(Extreme)│
//!     │   └──────────────────────────────────────┘
//!     │
//!     ▼
//! ┌─────────────────────────────────────────────┐
//! │ RegimeKappaEstimator                        │
//! │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
//! │ │κ_low    │ │κ_normal │ │κ_high   │ │κ_extreme│
//! │ │(tight)  │ │(base)   │ │(wide)   │ │(v.wide) │
//! │ └─────────┘ └─────────┘ └─────────┘ └─────────┘
//! │                    │
//! │                    ▼
//! │        κ_eff = Σ P(regime) × κ_regime
//! └─────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let mut estimator = RegimeKappaEstimator::new(RegimeKappaConfig::default());
//!
//! // Feed regime probabilities from HMM
//! estimator.set_regime_probabilities([0.1, 0.6, 0.2, 0.1]);
//!
//! // Feed fills with regime tag
//! estimator.on_fill(timestamp_ms, price, size, mid, regime);
//!
//! // Get regime-blended kappa
//! let kappa = estimator.kappa_effective();
//! ```

use super::kappa::BayesianKappaEstimator;
use super::VolatilityRegime;
use tracing::{debug, info};

/// Number of regimes (Low, Normal, High, Extreme).
pub const NUM_KAPPA_REGIMES: usize = 4;

/// Configuration for regime-conditioned kappa estimation.
#[derive(Debug, Clone)]
pub struct RegimeKappaConfig {
    /// Prior kappa for Low volatility regime (tighter markets).
    pub prior_kappa_low: f64,
    /// Prior kappa for Normal volatility regime.
    pub prior_kappa_normal: f64,
    /// Prior kappa for High volatility regime.
    pub prior_kappa_high: f64,
    /// Prior kappa for Extreme volatility regime (cascades).
    pub prior_kappa_extreme: f64,

    /// Prior strength (effective sample size) for each regime.
    pub prior_strength: f64,

    /// Observation window in milliseconds.
    pub window_ms: u64,

    /// Minimum observations per regime before using regime-specific estimate.
    pub min_regime_observations: usize,

    /// Whether to use regime-blended kappa or just current regime.
    pub use_blending: bool,
}

impl Default for RegimeKappaConfig {
    fn default() -> Self {
        Self {
            // Priors based on empirical observation:
            // - Low vol: very tight fills (κ ~ 3000, ~3 bps avg distance)
            // - Normal: moderate fills (κ ~ 2000, ~5 bps avg distance)
            // - High vol: wider fills (κ ~ 1000, ~10 bps avg distance)
            // - Extreme: very wide fills (κ ~ 500, ~20 bps avg distance)
            prior_kappa_low: 3000.0,
            prior_kappa_normal: 2000.0,
            prior_kappa_high: 1000.0,
            prior_kappa_extreme: 500.0,

            prior_strength: 10.0,
            window_ms: 600_000, // 10 minutes
            min_regime_observations: 10,
            use_blending: true,
        }
    }
}

impl RegimeKappaConfig {
    /// Config for HIP-3 DEX (illiquid) markets.
    pub fn hip3() -> Self {
        Self {
            // Lower kappa priors for illiquid markets
            prior_kappa_low: 2000.0,
            prior_kappa_normal: 1500.0,
            prior_kappa_high: 800.0,
            prior_kappa_extreme: 400.0,

            prior_strength: 15.0, // Stronger prior (slower adaptation)
            window_ms: 600_000,
            min_regime_observations: 5, // Lower threshold for sparse markets
            use_blending: true,
        }
    }

    /// Config for liquid CEX markets.
    pub fn liquid() -> Self {
        Self {
            prior_kappa_low: 4000.0,
            prior_kappa_normal: 2500.0,
            prior_kappa_high: 1500.0,
            prior_kappa_extreme: 800.0,

            prior_strength: 5.0, // Weaker prior (faster adaptation)
            window_ms: 300_000,  // 5 minutes
            min_regime_observations: 20,
            use_blending: true,
        }
    }

    /// Create config from a base kappa derived from market observables.
    ///
    /// Multipliers: Low=1.5x, Normal=1.0x, High=0.5x, Extreme=0.25x.
    /// This replaces hardcoded BTC priors with asset-adaptive values.
    pub fn from_base_kappa(base: f64) -> Self {
        let base = base.clamp(50.0, 50000.0);
        Self {
            prior_kappa_low: base * 1.5,
            prior_kappa_normal: base,
            prior_kappa_high: base * 0.5,
            prior_kappa_extreme: base * 0.25,
            ..Self::default()
        }
    }

    /// Create regime kappa config from a MarketProfile's implied kappa.
    ///
    /// Replaces hardcoded BTC-calibrated priors {3000, 2000, 1000, 500}
    /// with multipliers of the asset's observed fill intensity.
    /// Uses weaker prior strength (5.0) for profile-derived values to allow
    /// faster adaptation to real fills.
    ///
    /// # Arguments
    /// * `implied_kappa` - Kappa implied by the asset's BBO spread
    /// * `_liquidity_class` - Liquidity class as u8 (0=Illiquid, 1=Moderate, 2=Liquid).
    ///   Reserved for future per-class tuning; currently all classes use the same multipliers.
    pub fn from_profile(implied_kappa: f64, _liquidity_class: u8) -> Self {
        let base = implied_kappa.clamp(50.0, 50000.0);
        Self {
            prior_kappa_low: base * 1.5,
            prior_kappa_normal: base,
            prior_kappa_high: base * 0.5,
            prior_kappa_extreme: base * 0.25,
            // Weaker prior for profile-derived values (faster adaptation)
            prior_strength: 5.0,
            ..Self::default()
        }
    }

    /// Get prior kappa for a regime index.
    fn prior_for_regime(&self, regime: usize) -> f64 {
        match regime {
            0 => self.prior_kappa_low,
            1 => self.prior_kappa_normal,
            2 => self.prior_kappa_high,
            3 => self.prior_kappa_extreme,
            _ => self.prior_kappa_normal,
        }
    }
}

/// Regime-conditioned kappa estimator.
///
/// Maintains separate Bayesian kappa estimates for each volatility regime
/// and blends them using regime probabilities from the HMM.
#[derive(Debug)]
pub struct RegimeKappaEstimator {
    config: RegimeKappaConfig,

    /// Per-regime kappa estimators.
    regime_estimators: [BayesianKappaEstimator; NUM_KAPPA_REGIMES],

    /// Current regime probabilities [P(Low), P(Normal), P(High), P(Extreme)].
    regime_probs: [f64; NUM_KAPPA_REGIMES],

    /// Current dominant regime (for logging).
    current_regime: usize,

    /// Total fills processed.
    total_fills: u64,

    /// Fills per regime (for diagnostics).
    fills_per_regime: [u64; NUM_KAPPA_REGIMES],

    /// Whether the estimator has ever reached warmup threshold.
    /// Once true, stays true even if regime changes reduce observation counts.
    was_ever_warmed: bool,
}

impl RegimeKappaEstimator {
    /// Create a new regime-conditioned kappa estimator.
    pub fn new(config: RegimeKappaConfig) -> Self {
        let regime_estimators = [
            BayesianKappaEstimator::new(
                config.prior_kappa_low,
                config.prior_strength,
                config.window_ms,
            ),
            BayesianKappaEstimator::new(
                config.prior_kappa_normal,
                config.prior_strength,
                config.window_ms,
            ),
            BayesianKappaEstimator::new(
                config.prior_kappa_high,
                config.prior_strength,
                config.window_ms,
            ),
            BayesianKappaEstimator::new(
                config.prior_kappa_extreme,
                config.prior_strength,
                config.window_ms,
            ),
        ];

        Self {
            config,
            regime_estimators,
            regime_probs: [0.0, 1.0, 0.0, 0.0], // Start in Normal regime
            current_regime: 1,
            total_fills: 0,
            fills_per_regime: [0; NUM_KAPPA_REGIMES],
            was_ever_warmed: false,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(RegimeKappaConfig::default())
    }

    /// Set current regime probabilities from HMM.
    ///
    /// # Arguments
    /// * `probs` - [P(Low), P(Normal), P(High), P(Extreme)]
    pub fn set_regime_probabilities(&mut self, probs: [f64; NUM_KAPPA_REGIMES]) {
        // Normalize in case they don't sum to 1
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for (i, &p) in probs.iter().enumerate() {
                self.regime_probs[i] = p / sum;
            }
        }

        // Track dominant regime
        self.current_regime = self
            .regime_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(1);
    }

    /// Set regime from VolatilityRegime enum.
    pub fn set_regime(&mut self, regime: VolatilityRegime) {
        let regime_idx = match regime {
            VolatilityRegime::Low => 0,
            VolatilityRegime::Normal => 1,
            VolatilityRegime::High => 2,
            VolatilityRegime::Extreme => 3,
        };

        // Set hard probability (can also use soft probabilities from HMM)
        let mut probs = [0.0; NUM_KAPPA_REGIMES];
        probs[regime_idx] = 1.0;
        self.set_regime_probabilities(probs);
    }

    /// Process a fill observation.
    ///
    /// The fill is tagged with the current regime and fed to the appropriate estimator.
    pub fn on_fill(&mut self, timestamp_ms: u64, price: f64, size: f64, mid: f64) {
        if mid <= 0.0 || price <= 0.0 {
            return;
        }

        // Tag fill with dominant regime
        let regime = self.current_regime;

        // Feed to regime-specific estimator
        self.regime_estimators[regime].on_trade(timestamp_ms, price, size, mid);

        self.total_fills += 1;
        self.fills_per_regime[regime] += 1;

        // Track first warmup
        if !self.was_ever_warmed && self.is_warmed_up() {
            self.was_ever_warmed = true;
        }

        // Periodic logging
        if self.total_fills.is_multiple_of(50) {
            self.log_status();
        }
    }

    /// Process a fill with explicit regime tag.
    pub fn on_fill_with_regime(
        &mut self,
        timestamp_ms: u64,
        price: f64,
        size: f64,
        mid: f64,
        regime: usize,
    ) {
        if mid <= 0.0 || price <= 0.0 || regime >= NUM_KAPPA_REGIMES {
            return;
        }

        self.regime_estimators[regime].on_trade(timestamp_ms, price, size, mid);
        self.total_fills += 1;
        self.fills_per_regime[regime] += 1;

        // Track first warmup
        if !self.was_ever_warmed && self.is_warmed_up() {
            self.was_ever_warmed = true;
        }
    }

    /// Get effective kappa using regime-probability blending.
    ///
    /// κ_eff = Σ P(regime) × κ_regime
    ///
    /// If a regime has insufficient data, falls back to its prior.
    pub fn kappa_effective(&self) -> f64 {
        if !self.config.use_blending {
            // Just use current regime's kappa
            return self.regime_estimators[self.current_regime].posterior_mean();
        }

        let mut kappa = 0.0;

        for (i, estimator) in self.regime_estimators.iter().enumerate() {
            let regime_kappa = if estimator.observation_count() >= self.config.min_regime_observations
            {
                estimator.posterior_mean()
            } else {
                // Fall back to prior for this regime
                self.config.prior_for_regime(i)
            };

            kappa += self.regime_probs[i] * regime_kappa;
        }

        kappa.clamp(50.0, 10000.0)
    }

    /// Get blended prior kappa using current regime probabilities.
    ///
    /// Weights the per-regime prior kappa by the HMM regime probabilities,
    /// providing the expected prior kappa under the current regime distribution.
    pub fn blended_prior(&self) -> f64 {
        let mut prior = 0.0;
        for i in 0..NUM_KAPPA_REGIMES {
            prior += self.regime_probs[i] * self.config.prior_for_regime(i);
        }
        prior.clamp(50.0, 10000.0)
    }

    /// Get kappa for a specific regime.
    pub fn kappa_for_regime(&self, regime: usize) -> f64 {
        if regime >= NUM_KAPPA_REGIMES {
            return self.config.prior_kappa_normal;
        }

        let estimator = &self.regime_estimators[regime];
        if estimator.observation_count() >= self.config.min_regime_observations {
            estimator.posterior_mean()
        } else {
            self.config.prior_for_regime(regime)
        }
    }

    /// Get current regime index.
    pub fn current_regime(&self) -> usize {
        self.current_regime
    }

    /// Get current regime probabilities.
    pub fn regime_probabilities(&self) -> [f64; NUM_KAPPA_REGIMES] {
        self.regime_probs
    }

    /// Get observation count for a regime.
    pub fn regime_observation_count(&self, regime: usize) -> usize {
        if regime >= NUM_KAPPA_REGIMES {
            return 0;
        }
        self.regime_estimators[regime].observation_count()
    }

    /// Get total fills processed.
    pub fn total_fills(&self) -> u64 {
        self.total_fills
    }

    /// Get fills per regime for diagnostics.
    pub fn fills_per_regime(&self) -> [u64; NUM_KAPPA_REGIMES] {
        self.fills_per_regime
    }

    /// Check if estimator has sufficient data.
    pub fn is_warmed_up(&self) -> bool {
        // Warmed up if dominant regime has enough observations
        self.regime_estimators[self.current_regime].observation_count()
            >= self.config.min_regime_observations
    }

    /// Whether the estimator has ever reached warmup threshold.
    /// Once true, stays true permanently (survives regime changes).
    pub fn was_ever_warmed_up(&self) -> bool {
        self.was_ever_warmed
    }

    /// Get confidence in the estimate (based on observation counts).
    pub fn confidence(&self) -> f64 {
        // Weight by regime probability and observation count
        let mut conf = 0.0;
        for (i, estimator) in self.regime_estimators.iter().enumerate() {
            let regime_conf = estimator.confidence();
            conf += self.regime_probs[i] * regime_conf;
        }
        conf
    }

    /// Get detailed breakdown for logging.
    pub fn breakdown(&self) -> RegimeKappaBreakdown {
        RegimeKappaBreakdown {
            kappa_low: self.kappa_for_regime(0),
            kappa_normal: self.kappa_for_regime(1),
            kappa_high: self.kappa_for_regime(2),
            kappa_extreme: self.kappa_for_regime(3),
            kappa_effective: self.kappa_effective(),
            regime_probs: self.regime_probs,
            current_regime: self.current_regime,
            fills_per_regime: self.fills_per_regime,
            total_fills: self.total_fills,
        }
    }

    /// Log current status.
    fn log_status(&self) {
        let breakdown = self.breakdown();

        debug!(
            kappa_low = %format!("{:.0}", breakdown.kappa_low),
            kappa_normal = %format!("{:.0}", breakdown.kappa_normal),
            kappa_high = %format!("{:.0}", breakdown.kappa_high),
            kappa_extreme = %format!("{:.0}", breakdown.kappa_extreme),
            kappa_eff = %format!("{:.0}", breakdown.kappa_effective),
            regime = breakdown.current_regime,
            fills = breakdown.total_fills,
            "Regime kappa breakdown"
        );

        if breakdown.total_fills.is_multiple_of(100) {
            info!(
                kappa_effective = %format!("{:.0}", breakdown.kappa_effective),
                regime_probs = %format!("[{:.2}, {:.2}, {:.2}, {:.2}]",
                    breakdown.regime_probs[0],
                    breakdown.regime_probs[1],
                    breakdown.regime_probs[2],
                    breakdown.regime_probs[3]
                ),
                fills_by_regime = %format!("[{}, {}, {}, {}]",
                    breakdown.fills_per_regime[0],
                    breakdown.fills_per_regime[1],
                    breakdown.fills_per_regime[2],
                    breakdown.fills_per_regime[3]
                ),
                "Regime-conditioned kappa summary"
            );
        }
    }

    /// Reset all regime estimators.
    pub fn reset(&mut self) {
        for (i, estimator) in self.regime_estimators.iter_mut().enumerate() {
            *estimator = BayesianKappaEstimator::new(
                self.config.prior_for_regime(i),
                self.config.prior_strength,
                self.config.window_ms,
            );
        }
        self.total_fills = 0;
        self.fills_per_regime = [0; NUM_KAPPA_REGIMES];
    }
}

/// Detailed breakdown of regime-conditioned kappa for logging.
#[derive(Debug, Clone)]
pub struct RegimeKappaBreakdown {
    pub kappa_low: f64,
    pub kappa_normal: f64,
    pub kappa_high: f64,
    pub kappa_extreme: f64,
    pub kappa_effective: f64,
    pub regime_probs: [f64; NUM_KAPPA_REGIMES],
    pub current_regime: usize,
    pub fills_per_regime: [u64; NUM_KAPPA_REGIMES],
    pub total_fills: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_kappa_default() {
        let estimator = RegimeKappaEstimator::default_config();

        // Should start in Normal regime
        assert_eq!(estimator.current_regime(), 1);

        // Initial kappa should be prior
        let kappa = estimator.kappa_effective();
        assert!(kappa > 1000.0 && kappa < 3000.0);
    }

    #[test]
    fn test_regime_probabilities() {
        let mut estimator = RegimeKappaEstimator::default_config();

        // Set to High regime
        estimator.set_regime_probabilities([0.0, 0.0, 1.0, 0.0]);
        assert_eq!(estimator.current_regime(), 2);

        // Kappa should be closer to High regime prior
        let kappa = estimator.kappa_effective();
        assert!((kappa - 1000.0).abs() < 100.0);
    }

    #[test]
    fn test_regime_blending() {
        let mut estimator = RegimeKappaEstimator::default_config();

        // 50% Normal, 50% High
        estimator.set_regime_probabilities([0.0, 0.5, 0.5, 0.0]);

        // Kappa should be between Normal and High priors
        let kappa = estimator.kappa_effective();
        assert!(kappa > 1000.0 && kappa < 2000.0);
    }

    #[test]
    fn test_on_fill() {
        let mut estimator = RegimeKappaEstimator::default_config();

        // Set to Normal regime
        estimator.set_regime(VolatilityRegime::Normal);

        // Feed fills
        for i in 0..20 {
            let mid = 50000.0;
            let price = mid * (1.0 + 0.0005); // 5 bps from mid
            estimator.on_fill(i * 1000, price, 1.0, mid);
        }

        // Should have fills in Normal regime
        assert!(estimator.fills_per_regime[1] >= 20);
        assert!(estimator.is_warmed_up());
    }

    #[test]
    fn test_kappa_for_regime() {
        let estimator = RegimeKappaEstimator::default_config();

        // With no fills, should return priors
        let kappa_low = estimator.kappa_for_regime(0);
        let kappa_normal = estimator.kappa_for_regime(1);
        let kappa_high = estimator.kappa_for_regime(2);
        let kappa_extreme = estimator.kappa_for_regime(3);

        assert!(kappa_low > kappa_normal);
        assert!(kappa_normal > kappa_high);
        assert!(kappa_high > kappa_extreme);
    }

    #[test]
    fn test_hip3_config() {
        let estimator = RegimeKappaEstimator::new(RegimeKappaConfig::hip3());

        // HIP3 should have lower priors
        let kappa = estimator.kappa_effective();
        assert!(kappa < 2000.0);
    }

    #[test]
    fn test_liquid_config() {
        let estimator = RegimeKappaEstimator::new(RegimeKappaConfig::liquid());

        // Liquid should have higher priors
        let kappa_normal = estimator.kappa_for_regime(1);
        assert!(kappa_normal > 2000.0);
    }

    #[test]
    fn test_set_regime_enum() {
        let mut estimator = RegimeKappaEstimator::default_config();

        estimator.set_regime(VolatilityRegime::Extreme);
        assert_eq!(estimator.current_regime(), 3);

        estimator.set_regime(VolatilityRegime::Low);
        assert_eq!(estimator.current_regime(), 0);
    }

    #[test]
    fn test_was_ever_warmed_up_starts_false_transitions_to_true() {
        let mut estimator = RegimeKappaEstimator::default_config();
        assert!(!estimator.was_ever_warmed_up());
        assert!(!estimator.is_warmed_up());

        // Set to Normal regime and feed enough fills (min_regime_observations = 10)
        estimator.set_regime(VolatilityRegime::Normal);
        for i in 0..15 {
            let mid = 50000.0;
            let price = mid * (1.0 + 0.0005);
            estimator.on_fill(i * 1000, price, 1.0, mid);
        }

        assert!(estimator.is_warmed_up());
        assert!(estimator.was_ever_warmed_up());
    }

    #[test]
    fn test_was_ever_warmed_up_stays_true_after_regime_change() {
        let mut estimator = RegimeKappaEstimator::default_config();

        // Warm up in Normal regime
        estimator.set_regime(VolatilityRegime::Normal);
        for i in 0..15 {
            let mid = 50000.0;
            let price = mid * (1.0 + 0.0005);
            estimator.on_fill(i * 1000, price, 1.0, mid);
        }
        assert!(estimator.is_warmed_up());
        assert!(estimator.was_ever_warmed_up());

        // Switch to Extreme regime (no fills there yet)
        estimator.set_regime(VolatilityRegime::Extreme);
        // is_warmed_up() may be false now (Extreme regime has 0 observations)
        assert!(!estimator.is_warmed_up());
        // But was_ever_warmed_up() stays true
        assert!(estimator.was_ever_warmed_up());
    }

    // === from_profile tests ===

    #[test]
    fn test_from_profile_hype_kappa() {
        // HYPE-like: implied kappa ~466
        let config = RegimeKappaConfig::from_profile(466.0, 0);
        assert!(
            (config.prior_kappa_low - 699.0).abs() < 0.1,
            "Low should be 1.5x: got {:.1}",
            config.prior_kappa_low
        );
        assert!(
            (config.prior_kappa_normal - 466.0).abs() < 0.1,
            "Normal should be 1.0x: got {:.1}",
            config.prior_kappa_normal
        );
        assert!(
            (config.prior_kappa_high - 233.0).abs() < 0.1,
            "High should be 0.5x: got {:.1}",
            config.prior_kappa_high
        );
        assert!(
            (config.prior_kappa_extreme - 116.5).abs() < 0.1,
            "Extreme should be 0.25x: got {:.1}",
            config.prior_kappa_extreme
        );
    }

    #[test]
    fn test_from_profile_btc_kappa() {
        // BTC-like: implied kappa ~6000
        let config = RegimeKappaConfig::from_profile(6000.0, 2);
        assert!(
            (config.prior_kappa_low - 9000.0).abs() < 0.1,
            "Low should be 9000, got {:.1}",
            config.prior_kappa_low
        );
        assert!(
            (config.prior_kappa_normal - 6000.0).abs() < 0.1,
            "Normal should be 6000, got {:.1}",
            config.prior_kappa_normal
        );
        assert!(
            (config.prior_kappa_high - 3000.0).abs() < 0.1,
            "High should be 3000, got {:.1}",
            config.prior_kappa_high
        );
        assert!(
            (config.prior_kappa_extreme - 1500.0).abs() < 0.1,
            "Extreme should be 1500, got {:.1}",
            config.prior_kappa_extreme
        );
    }

    #[test]
    fn test_from_profile_prior_ordering() {
        for &kappa in &[100.0, 466.0, 2000.0, 6000.0, 20000.0] {
            let config = RegimeKappaConfig::from_profile(kappa, 1);
            assert!(
                config.prior_kappa_low > config.prior_kappa_normal,
                "Low > Normal for kappa={kappa}"
            );
            assert!(
                config.prior_kappa_normal > config.prior_kappa_high,
                "Normal > High for kappa={kappa}"
            );
            assert!(
                config.prior_kappa_high > config.prior_kappa_extreme,
                "High > Extreme for kappa={kappa}"
            );
        }
    }

    #[test]
    fn test_from_profile_clamp_extremes() {
        // Very small kappa -> clamped to 50
        let config = RegimeKappaConfig::from_profile(10.0, 0);
        assert!(
            (config.prior_kappa_normal - 50.0).abs() < 0.1,
            "Should clamp to 50, got {:.1}",
            config.prior_kappa_normal
        );

        // Very large kappa -> clamped to 50000
        let config = RegimeKappaConfig::from_profile(100000.0, 2);
        assert!(
            (config.prior_kappa_normal - 50000.0).abs() < 0.1,
            "Should clamp to 50000, got {:.1}",
            config.prior_kappa_normal
        );
    }
}
