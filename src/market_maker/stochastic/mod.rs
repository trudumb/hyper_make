//! First-Principles Stochastic Control for Market Making.
//!
//! This module implements mathematically-derived decision-making that emerges
//! from Bayesian posteriors and HJB optimization. **No ad-hoc thresholds.**
//!
//! ## Architecture
//!
//! ```text
//! Market Data
//!     ↓
//! ┌─────────────────────────────────────────────────────────────┐
//! │  MarketBeliefs (Bayesian Posterior Aggregator)               │
//! │                                                              │
//! │  ┌──────────────────┐  ┌───────────────────┐                │
//! │  │ NormalInverseGamma │  │ FillIntensityPost │                │
//! │  │ (μ, σ²) drift/vol │  │ (κ) fill rate      │                │
//! │  └──────────────────┘  └───────────────────┘                │
//! │           ↓                      ↓                           │
//! │     E[μ | data]            E[κ | fills]                      │
//! │           ↓                      ↓                           │
//! │  predictive_bias = E[μ]   optimal_spread = f(κ)             │
//! └─────────────────────────────────────────────────────────────┘
//!     ↓
//! ┌─────────────────────────────────────────────────────────────┐
//! │  HJBSolver (Optimal Quote Derivation)                        │
//! │                                                              │
//! │  Half-spread:  δ* = (1/γ) × ln(1 + γ/κ)                     │
//! │  Skew:         skew = γσ²qT/2 + β_t/2                       │
//! │  Sizes:        s^b ≤ Q - q,  s^a ≤ Q + q                    │
//! └─────────────────────────────────────────────────────────────┘
//!     ↓
//! Optimal Quotes (derived, not hardcoded)
//! ```
//!
//! ## Key Insight
//!
//! The predictive bias β_t = E[μ | data] is **NOT** a heuristic - it's the
//! posterior mean of drift from the Normal-Inverse-Gamma conjugate update.
//!
//! When beliefs shift to negative μ (capitulation), β_t < 0 automatically,
//! causing the HJB solution to skew toward sells. **No if-else rules needed.**
//!
//! ## Why This Eliminates Ad-Hoc Rules
//!
//! | Current Heuristic | First-Principles Replacement |
//! |-------------------|------------------------------|
//! | `if changepoint_prob > 0.85 → pull quotes` | β_t = E[μ \| beliefs] < 0 → skew emerges from HJB |
//! | `if signal_weight > 0.1 → quote` | Quote whenever E[utility] > 0, derived from V |
//! | `spread_mult = 1.5` | δ* = (1/γ) × ln(1 + γ/κ) with κ from posterior |
//! | `widen bids by 2bps` | skew = γσ²qT/2 + β_t/2, no magic numbers |
//!
//! ## Usage
//!
//! ```ignore
//! use market_maker::stochastic::{MarketBeliefs, HJBSolver};
//!
//! // Create belief system
//! let mut beliefs = MarketBeliefs::default();
//!
//! // Update beliefs with market data
//! beliefs.observe_price(price_return, dt);
//! beliefs.observe_fill(depth_bps, dt);
//!
//! // Derive optimal quotes from HJB
//! let solver = HJBSolver::default();
//! let quotes = solver.optimal_quotes(&beliefs, position, max_position, target_size);
//!
//! // Use derived values (not hardcoded)
//! let half_spread = quotes.half_spread;
//! let skew = quotes.skew;
//! let predictive_bias = quotes.predictive_bias;
//! ```

pub mod beliefs;
pub mod conjugate;
pub mod continuation;
pub mod hjb_solver;

// Re-export key types
pub use beliefs::{BeliefSummary, MarketBeliefs, Regime};
pub use conjugate::{FillIntensityPosterior, NormalInverseGamma};
pub use continuation::{
    ContinuationFusionConfig, ContinuationPosterior, ContinuationSignalSummary,
};
pub use hjb_solver::{HJBDiagnostics, HJBQuotes, HJBSolver, HJBSolverConfig};

/// Configuration for the stochastic control system.
#[derive(Debug, Clone)]
pub struct StochasticControlConfig {
    /// Enable the first-principles belief system.
    /// When true, uses Bayesian posteriors for all decisions.
    pub enable_belief_system: bool,

    /// Enable HJB-derived quotes.
    /// When true, quotes emerge from HJB optimization, not heuristics.
    pub enable_hjb_quotes: bool,

    /// HJB solver configuration.
    pub hjb_config: HJBSolverConfig,

    /// Prior standard deviation for drift (per-second).
    /// Smaller = more informative prior centered at zero.
    pub drift_prior_sigma: f64,

    /// Prior mean for fill intensity κ.
    pub kappa_prior_mean: f64,

    /// Prior coefficient of variation for κ.
    pub kappa_prior_cv: f64,

    /// Decay factor for beliefs (for non-stationarity).
    /// 0.999 = slow decay, 0.99 = fast decay.
    pub belief_decay_factor: f64,

    /// Apply belief decay every N observations.
    pub belief_decay_interval: u64,

    /// Blend factor for gradual rollout.
    /// 0.0 = use old heuristics, 1.0 = use full first-principles.
    pub blend_factor: f64,
}

impl Default for StochasticControlConfig {
    fn default() -> Self {
        Self {
            enable_belief_system: true,
            enable_hjb_quotes: true,
            hjb_config: HJBSolverConfig::default(),
            drift_prior_sigma: 0.0001, // 1 bps per second std
            kappa_prior_mean: 200.0,   // Conservative for thin DEX
            kappa_prior_cv: 0.5,       // 50% uncertainty
            belief_decay_factor: 0.999,
            belief_decay_interval: 1000,
            blend_factor: 1.0, // Full first-principles
        }
    }
}

impl StochasticControlConfig {
    /// Create config for gradual rollout (blend between old and new).
    pub fn blended(blend: f64) -> Self {
        Self {
            blend_factor: blend.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Create config with conservative priors (for thin markets).
    pub fn thin_market() -> Self {
        Self {
            kappa_prior_mean: 100.0,   // Lower fill rate
            kappa_prior_cv: 0.7,       // Higher uncertainty
            drift_prior_sigma: 0.0002, // Higher drift uncertainty
            hjb_config: HJBSolverConfig {
                gamma: 0.7, // More risk-averse
                min_half_spread_bps: 3.0,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create config with aggressive priors (for liquid markets).
    pub fn liquid_market() -> Self {
        Self {
            kappa_prior_mean: 500.0,    // Higher fill rate
            kappa_prior_cv: 0.3,        // Lower uncertainty
            drift_prior_sigma: 0.00005, // Lower drift uncertainty
            hjb_config: HJBSolverConfig {
                gamma: 0.3, // Less risk-averse
                min_half_spread_bps: 1.5,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// Builder for gradual integration of first-principles system.
///
/// # Deprecation Notice (Phase 7)
///
/// This struct is DEPRECATED in favor of `CentralBeliefState` in the `belief` module.
/// The centralized belief system provides:
/// - Single source of truth for all Bayesian beliefs
/// - Point-in-time snapshots via `BeliefSnapshot`
/// - Unified update path via `BeliefUpdate` enum
///
/// This struct is retained for backward compatibility but is no longer updated
/// with price observations. Use `market_maker.central_beliefs()` instead.
#[derive(Debug, Clone)]
pub struct StochasticControlBuilder {
    config: StochasticControlConfig,
    beliefs: MarketBeliefs,
    solver: HJBSolver,
}

impl Default for StochasticControlBuilder {
    fn default() -> Self {
        Self::new(StochasticControlConfig::default())
    }
}

impl StochasticControlBuilder {
    /// Create a new builder with configuration.
    pub fn new(config: StochasticControlConfig) -> Self {
        let beliefs = MarketBeliefs::with_priors(
            config.drift_prior_sigma,
            config.kappa_prior_mean,
            config.kappa_prior_cv,
        );
        let solver = HJBSolver::new(config.hjb_config.clone());

        Self {
            config,
            beliefs,
            solver,
        }
    }

    /// Get beliefs for observation updates.
    pub fn beliefs(&self) -> &MarketBeliefs {
        &self.beliefs
    }

    /// Get mutable beliefs for observation updates.
    pub fn beliefs_mut(&mut self) -> &mut MarketBeliefs {
        &mut self.beliefs
    }

    /// Get solver for quote computation.
    pub fn solver(&self) -> &HJBSolver {
        &self.solver
    }

    /// Observe price return.
    pub fn observe_price(&mut self, price_return: f64, dt: f64) {
        if self.config.enable_belief_system {
            self.beliefs.observe_price(price_return, dt);
        }
    }

    /// Observe fill.
    pub fn observe_fill(&mut self, depth_bps: f64, dt: f64) {
        if self.config.enable_belief_system {
            self.beliefs.observe_fill(depth_bps, dt);
        }
    }

    /// Update regime probabilities.
    pub fn update_regime(&mut self, probs: [f64; 4]) {
        self.beliefs.update_regime(probs);
    }

    /// Update changepoint probability.
    pub fn update_changepoint(&mut self, prob: f64) {
        self.beliefs.update_changepoint(prob);
    }

    /// Compute optimal quotes.
    pub fn optimal_quotes(&self, position: f64, max_position: f64, target_size: f64) -> HJBQuotes {
        if !self.config.enable_hjb_quotes {
            return HJBQuotes::default();
        }

        self.solver
            .optimal_quotes(&self.beliefs, position, max_position, target_size)
    }

    /// Compute optimal quotes with diagnostics.
    pub fn optimal_quotes_with_diagnostics(
        &self,
        position: f64,
        max_position: f64,
        target_size: f64,
    ) -> (HJBQuotes, HJBDiagnostics) {
        self.solver.optimal_quotes_with_diagnostics(
            &self.beliefs,
            position,
            max_position,
            target_size,
        )
    }

    /// Get predictive bias for external use.
    ///
    /// This can be blended with existing systems during rollout.
    pub fn predictive_bias(&self) -> f64 {
        if !self.config.enable_belief_system {
            return 0.0;
        }

        self.beliefs.predictive_bias() * self.config.blend_factor
    }

    /// Get expected kappa for external use.
    pub fn expected_kappa(&self) -> f64 {
        self.beliefs.expected_kappa
    }

    /// Get expected sigma for external use.
    pub fn expected_sigma(&self) -> f64 {
        self.beliefs.expected_sigma
    }

    /// Is the system warmed up?
    pub fn is_warmed_up(&self) -> bool {
        self.beliefs.is_warmed_up()
    }

    /// Get belief summary for diagnostics.
    pub fn summary(&self) -> BeliefSummary {
        self.beliefs.summary()
    }

    /// Get config reference.
    pub fn config(&self) -> &StochasticControlConfig {
        &self.config
    }

    /// Set blend factor for gradual rollout.
    pub fn set_blend(&mut self, blend: f64) {
        self.config.blend_factor = blend.clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stochastic_control_builder() {
        let mut builder = StochasticControlBuilder::default();

        // Observe some data
        for _ in 0..20 {
            builder.observe_price(-0.001, 1.0);
        }

        // Predictive bias should be negative
        assert!(
            builder.predictive_bias() < 0.0,
            "Should have negative bias after negative returns"
        );
    }

    #[test]
    fn test_config_presets() {
        let thin = StochasticControlConfig::thin_market();
        let liquid = StochasticControlConfig::liquid_market();

        assert!(thin.kappa_prior_mean < liquid.kappa_prior_mean);
        assert!(thin.hjb_config.gamma > liquid.hjb_config.gamma);
    }

    #[test]
    fn test_blend_factor() {
        let mut builder = StochasticControlBuilder::new(StochasticControlConfig::blended(0.5));

        // Observe negative returns
        for _ in 0..30 {
            builder.observe_price(-0.002, 1.0);
        }

        // Predictive bias should be scaled by blend factor
        let full_bias = builder.beliefs.predictive_bias();
        let blended_bias = builder.predictive_bias();

        assert!(
            (blended_bias - full_bias * 0.5).abs() < 1e-10,
            "Blend should scale bias"
        );
    }

    #[test]
    fn test_quotes_computation() {
        let mut builder = StochasticControlBuilder::default();

        // Warm up with observations
        for _ in 0..100 {
            builder.observe_price(0.0005, 1.0);
        }
        for _ in 0..20 {
            builder.observe_fill(5.0, 1.0);
        }

        let quotes = builder.optimal_quotes(0.0, 1.0, 0.1);

        assert!(quotes.half_spread > 0.0);
        assert!(quotes.bid_size > 0.0);
        assert!(quotes.ask_size > 0.0);
    }
}
