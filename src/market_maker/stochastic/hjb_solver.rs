//! HJB-derived optimal quotes from Bayesian beliefs.
//!
//! This module implements the optimal control solution where quotes emerge
//! from solving Hamilton-Jacobi-Bellman under portfolio constraints.
//!
//! ## Key Formulas (Derived, Not Hardcoded)
//!
//! **Half-spread** (from HJB first-order condition):
//! ```text
//! δ* = (1/γ) × ln(1 + γ/κ)
//! ```
//! Where κ comes from Bayesian posterior mean E[κ | fills].
//!
//! **Skew** (inventory + predictive):
//! ```text
//! skew = γ×σ²×q×T/2 + β_t/2
//! where β_t = E[μ | beliefs] from NIG posterior
//! ```
//!
//! ## Why This Works
//!
//! The predictive bias β_t = E[μ | data] is not a heuristic - it's the
//! posterior mean of drift. When beliefs shift to negative μ (capitulation),
//! β_t < 0 automatically, causing sell-biased quotes.
//!
//! No more:
//! - `if changepoint_prob > X`
//! - `if signal_weight > Y`
//! - `spread_mult = Z`
//!
//! Instead: Continuous, smooth, mathematically-derived decisions.

use super::beliefs::MarketBeliefs;

/// HJB-derived optimal quotes.
#[derive(Debug, Clone, Copy)]
pub struct HJBQuotes {
    /// Optimal half-spread (bps).
    /// δ* = (1/γ) × ln(1 + γ/κ) + fee
    pub half_spread: f64,

    /// Optimal skew (bps).
    /// skew = inventory_skew + predictive_skew
    /// Positive = shift quotes down (sell more aggressively)
    /// Negative = shift quotes up (buy more aggressively)
    pub skew: f64,

    /// Optimal bid size (contracts).
    /// Constrained by |q + s^b| ≤ Q
    pub bid_size: f64,

    /// Optimal ask size (contracts).
    /// Constrained by |q - s^a| ≤ Q
    pub ask_size: f64,

    /// Predictive bias component of skew.
    /// β_t = E[μ | data] from beliefs
    pub predictive_bias: f64,

    /// Inventory skew component.
    /// = γ × σ² × q × T / 2
    pub inventory_skew: f64,

    /// Expected fill probability for bids.
    pub bid_fill_prob: f64,

    /// Expected fill probability for asks.
    pub ask_fill_prob: f64,

    /// Confidence in the quote (0-1).
    /// Based on belief confidence.
    pub confidence: f64,
}

impl Default for HJBQuotes {
    fn default() -> Self {
        Self {
            half_spread: 5.0, // 5 bps default
            skew: 0.0,
            bid_size: 0.0,
            ask_size: 0.0,
            predictive_bias: 0.0,
            inventory_skew: 0.0,
            bid_fill_prob: 0.5,
            ask_fill_prob: 0.5,
            confidence: 0.0,
        }
    }
}

/// Configuration for HJB solver.
#[derive(Debug, Clone)]
pub struct HJBSolverConfig {
    /// Risk aversion parameter γ (from utility function).
    /// Higher γ → wider spreads, faster inventory reduction.
    pub gamma: f64,

    /// Trading time horizon T (seconds).
    /// Used in inventory skew: γσ²qT
    pub time_horizon: f64,

    /// Maker fee (bps).
    /// Added to half-spread floor.
    pub maker_fee_bps: f64,

    /// Minimum half-spread (bps).
    /// Floor to ensure profitability.
    pub min_half_spread_bps: f64,

    /// Maximum half-spread (bps).
    /// Cap to ensure competitiveness.
    pub max_half_spread_bps: f64,

    /// Predictive bias scaling factor.
    /// β_t contribution = predictive_bias_scale × E[μ]
    pub predictive_bias_scale: f64,

    /// Minimum order size (contracts).
    pub min_order_size: f64,

    /// Whether to use regime-dependent gamma.
    pub use_regime_gamma: bool,

    /// Gamma multipliers by regime [quiet, normal, bursty, cascade].
    pub regime_gamma_mults: [f64; 4],

    /// Depth sensitivity for fill probability (γ in λ = κ × e^(-γδ))
    pub depth_sensitivity: f64,
}

impl Default for HJBSolverConfig {
    fn default() -> Self {
        Self {
            gamma: 0.5,
            time_horizon: 60.0, // 1 minute
            maker_fee_bps: 1.5, // Hyperliquid maker fee
            min_half_spread_bps: 2.0,
            max_half_spread_bps: 50.0,
            predictive_bias_scale: 0.5, // β_t/2 in the formula
            min_order_size: 0.001,
            use_regime_gamma: true,
            regime_gamma_mults: [0.7, 1.0, 1.5, 3.0], // quiet, normal, bursty, cascade
            depth_sensitivity: 0.5,
        }
    }
}

/// HJB optimal quote solver.
///
/// Derives quotes from Bayesian beliefs and portfolio constraints.
/// No heuristics - all decisions flow from the math.
#[derive(Debug, Clone)]
pub struct HJBSolver {
    config: HJBSolverConfig,
}

impl Default for HJBSolver {
    fn default() -> Self {
        Self::new(HJBSolverConfig::default())
    }
}

impl HJBSolver {
    /// Create a new HJB solver with configuration.
    pub fn new(config: HJBSolverConfig) -> Self {
        Self { config }
    }

    /// Solve for optimal quotes given beliefs and constraints.
    ///
    /// # Arguments
    /// * `beliefs` - Current Bayesian beliefs over market parameters
    /// * `position` - Current position (positive = long)
    /// * `max_position` - Maximum allowed position |q| ≤ Q
    /// * `target_size` - Target order size (before constraint)
    ///
    /// # Returns
    /// Optimal quotes derived from HJB solution.
    pub fn optimal_quotes(
        &self,
        beliefs: &MarketBeliefs,
        position: f64,
        max_position: f64,
        target_size: f64,
    ) -> HJBQuotes {
        // Get posterior means from beliefs
        let kappa = beliefs.expected_kappa.max(1.0); // Floor to avoid blow-up
        let sigma = beliefs.expected_sigma.max(1e-6);
        let mu = beliefs.expected_drift; // E[μ | data] - THE predictive signal

        // Compute effective gamma (regime-dependent if enabled)
        let gamma = self.effective_gamma(beliefs);

        // === Optimal Half-Spread (HJB First-Order Condition) ===
        // δ* = (1/γ) × ln(1 + γ/κ)
        let half_spread_raw = (1.0 / gamma) * (1.0 + gamma / kappa).ln();
        let half_spread_bps = half_spread_raw * 10000.0; // Convert to bps

        // Add fee and apply bounds
        let half_spread = (half_spread_bps + self.config.maker_fee_bps).clamp(
            self.config.min_half_spread_bps,
            self.config.max_half_spread_bps,
        );

        // === Optimal Skew (Inventory + Predictive) ===
        // Inventory skew: γ × σ² × q × T / 2
        let q_normalized = if max_position.abs() > 1e-10 {
            position / max_position
        } else {
            0.0
        };
        let inventory_skew =
            gamma * sigma.powi(2) * q_normalized * self.config.time_horizon / 2.0 * 10000.0;

        // Predictive skew: β_t / 2 where β_t = E[μ | data]
        // mu is per-second drift, convert to bps
        let predictive_bias = mu * self.config.predictive_bias_scale * 10000.0;

        // Total skew (positive = shift quotes down)
        let skew = inventory_skew + predictive_bias;

        // === Size from Portfolio Constraint ===
        // bid_size ≤ Q - q (can buy up to Q - current position)
        // ask_size ≤ Q + q (can sell up to Q + current position)
        let max_bid_size = (max_position - position).max(0.0);
        let max_ask_size = (max_position + position).max(0.0);

        // Apply target size with constraints
        let bid_size = target_size
            .min(max_bid_size)
            .max(self.config.min_order_size);
        let ask_size = target_size
            .min(max_ask_size)
            .max(self.config.min_order_size);

        // === Fill Probabilities ===
        // P(fill) = λ(δ) × dt ≈ κ × exp(-γ_depth × δ)
        let bid_depth = half_spread + skew.max(0.0);
        let ask_depth = half_spread - skew.min(0.0);

        let bid_fill_prob = self.fill_probability(beliefs, bid_depth);
        let ask_fill_prob = self.fill_probability(beliefs, ask_depth);

        // === Confidence ===
        let confidence = beliefs.overall_confidence();

        HJBQuotes {
            half_spread,
            skew,
            bid_size,
            ask_size,
            predictive_bias,
            inventory_skew,
            bid_fill_prob,
            ask_fill_prob,
            confidence,
        }
    }

    /// Compute effective gamma (regime-dependent).
    fn effective_gamma(&self, beliefs: &MarketBeliefs) -> f64 {
        if !self.config.use_regime_gamma {
            return self.config.gamma;
        }

        // Blend gamma across regimes using belief weights
        let blended_mult = beliefs.regime_blend(
            self.config.regime_gamma_mults[0], // quiet
            self.config.regime_gamma_mults[1], // normal
            self.config.regime_gamma_mults[2], // bursty
            self.config.regime_gamma_mults[3], // cascade
        );

        (self.config.gamma * blended_mult).max(0.01)
    }

    /// Compute fill probability at given depth.
    fn fill_probability(&self, beliefs: &MarketBeliefs, depth_bps: f64) -> f64 {
        // λ(δ) = κ × exp(-γ × δ)
        let intensity = beliefs.fill_intensity_at_depth(depth_bps);

        // Convert to probability (assume 1 second window)
        // P(fill in dt) = 1 - exp(-λ × dt)
        (1.0 - (-intensity * 1.0).exp()).clamp(0.0, 1.0)
    }

    /// Compute size distribution across multiple levels.
    ///
    /// Sizes are proportional to expected profit × fill probability.
    ///
    /// # Arguments
    /// * `beliefs` - Current beliefs
    /// * `depths` - Depths for each level (bps from mid)
    /// * `total_size` - Total size to distribute
    /// * `position` - Current position
    /// * `max_position` - Maximum position
    ///
    /// # Returns
    /// Sizes for each level, constrained by portfolio limits.
    pub fn size_distribution(
        &self,
        beliefs: &MarketBeliefs,
        depths: &[f64],
        total_size: f64,
        position: f64,
        max_position: f64,
        is_bid: bool,
    ) -> Vec<f64> {
        if depths.is_empty() {
            return vec![];
        }

        // Compute expected value at each depth
        // EV(δ) = λ(δ) × SC(δ) where SC = spread capture
        let evs: Vec<f64> = depths
            .iter()
            .map(|&depth| {
                let fill_prob = self.fill_probability(beliefs, depth);
                let spread_capture = depth; // SC ≈ depth for simplicity
                fill_prob * spread_capture
            })
            .collect();

        // Normalize to get allocation fractions
        let total_ev: f64 = evs.iter().sum();
        let fractions: Vec<f64> = if total_ev > 1e-10 {
            evs.iter().map(|ev| ev / total_ev).collect()
        } else {
            vec![1.0 / depths.len() as f64; depths.len()] // Uniform if no EV
        };

        // Apply portfolio constraint
        let max_total = if is_bid {
            (max_position - position).max(0.0)
        } else {
            (max_position + position).max(0.0)
        };

        let constrained_total = total_size.min(max_total);

        // Distribute size
        fractions
            .iter()
            .map(|&f| (f * constrained_total).max(self.config.min_order_size))
            .collect()
    }

    /// Get config for external inspection.
    pub fn config(&self) -> &HJBSolverConfig {
        &self.config
    }

    /// Update config.
    pub fn set_config(&mut self, config: HJBSolverConfig) {
        self.config = config;
    }

    /// Update gamma.
    pub fn set_gamma(&mut self, gamma: f64) {
        self.config.gamma = gamma.max(0.01);
    }

    /// Update time horizon.
    pub fn set_time_horizon(&mut self, t: f64) {
        self.config.time_horizon = t.max(1.0);
    }
}

/// Diagnostics for HJB quote derivation.
#[derive(Debug, Clone)]
pub struct HJBDiagnostics {
    /// Input beliefs summary
    pub beliefs_summary: String,

    /// Effective gamma used
    pub effective_gamma: f64,

    /// Raw half-spread before bounds (bps)
    pub raw_half_spread: f64,

    /// Components of skew
    pub skew_breakdown: SkewBreakdown,

    /// Size constraint info
    pub size_constraints: SizeConstraints,
}

/// Breakdown of skew components.
#[derive(Debug, Clone)]
pub struct SkewBreakdown {
    /// Inventory component (γσ²qT/2)
    pub inventory: f64,
    /// Predictive component (β_t/2)
    pub predictive: f64,
    /// Total skew
    pub total: f64,
}

/// Size constraint information.
#[derive(Debug, Clone)]
pub struct SizeConstraints {
    /// Maximum bid size from constraint
    pub max_bid: f64,
    /// Maximum ask size from constraint
    pub max_ask: f64,
    /// Actual bid size after constraint
    pub actual_bid: f64,
    /// Actual ask size after constraint
    pub actual_ask: f64,
}

impl HJBSolver {
    /// Compute quotes with full diagnostics.
    pub fn optimal_quotes_with_diagnostics(
        &self,
        beliefs: &MarketBeliefs,
        position: f64,
        max_position: f64,
        target_size: f64,
    ) -> (HJBQuotes, HJBDiagnostics) {
        let quotes = self.optimal_quotes(beliefs, position, max_position, target_size);
        let gamma = self.effective_gamma(beliefs);
        let kappa = beliefs.expected_kappa.max(1.0);
        let raw_half_spread = (1.0 / gamma) * (1.0 + gamma / kappa).ln() * 10000.0;

        let diagnostics = HJBDiagnostics {
            beliefs_summary: format!(
                "drift={:.6}, sigma={:.6}, kappa={:.1}",
                beliefs.expected_drift, beliefs.expected_sigma, beliefs.expected_kappa
            ),
            effective_gamma: gamma,
            raw_half_spread,
            skew_breakdown: SkewBreakdown {
                inventory: quotes.inventory_skew,
                predictive: quotes.predictive_bias,
                total: quotes.skew,
            },
            size_constraints: SizeConstraints {
                max_bid: (max_position - position).max(0.0),
                max_ask: (max_position + position).max(0.0),
                actual_bid: quotes.bid_size,
                actual_ask: quotes.ask_size,
            },
        };

        (quotes, diagnostics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hjb_solver_default() {
        let solver = HJBSolver::default();
        let beliefs = MarketBeliefs::default();

        let quotes = solver.optimal_quotes(&beliefs, 0.0, 1.0, 0.1);

        assert!(quotes.half_spread > 0.0);
        assert!(quotes.half_spread >= solver.config.min_half_spread_bps);
    }

    #[test]
    fn test_hjb_zero_position_zero_skew() {
        let solver = HJBSolver::default();
        let beliefs = MarketBeliefs::default();

        let quotes = solver.optimal_quotes(&beliefs, 0.0, 1.0, 0.1);

        // With zero position and neutral drift, skew should be ~0
        assert!(
            quotes.skew.abs() < 1.0,
            "Zero position + neutral beliefs should give ~zero skew: {}",
            quotes.skew
        );
    }

    #[test]
    fn test_hjb_long_position_positive_skew() {
        let solver = HJBSolver::default();
        let beliefs = MarketBeliefs::default();

        // Long position should give positive skew (shift quotes down)
        let quotes = solver.optimal_quotes(&beliefs, 0.5, 1.0, 0.1);

        assert!(
            quotes.inventory_skew > 0.0,
            "Long position should give positive inventory skew: {}",
            quotes.inventory_skew
        );
    }

    #[test]
    fn test_hjb_short_position_negative_skew() {
        let solver = HJBSolver::default();
        let beliefs = MarketBeliefs::default();

        // Short position should give negative skew (shift quotes up)
        let quotes = solver.optimal_quotes(&beliefs, -0.5, 1.0, 0.1);

        assert!(
            quotes.inventory_skew < 0.0,
            "Short position should give negative inventory skew: {}",
            quotes.inventory_skew
        );
    }

    #[test]
    fn test_hjb_negative_drift_negative_bias() {
        let solver = HJBSolver::default();
        let mut beliefs = MarketBeliefs::default();

        // Observe negative returns to shift drift posterior
        for _ in 0..30 {
            beliefs.observe_price(-0.002, 1.0);
        }

        let quotes = solver.optimal_quotes(&beliefs, 0.0, 1.0, 0.1);

        // Negative drift should give negative predictive bias
        assert!(
            quotes.predictive_bias < 0.0,
            "Negative drift should give negative predictive bias: {}",
            quotes.predictive_bias
        );
    }

    #[test]
    fn test_hjb_portfolio_constraints() {
        let solver = HJBSolver::default();
        let beliefs = MarketBeliefs::default();

        // Nearly at max position
        let quotes = solver.optimal_quotes(&beliefs, 0.95, 1.0, 0.5);

        // Bid size should be constrained (can only buy 0.05 more)
        assert!(
            quotes.bid_size <= 0.05 + 1e-6,
            "Bid size should be constrained by max position: {}",
            quotes.bid_size
        );

        // Ask size should be larger (can sell up to 1.95)
        assert!(
            quotes.ask_size > quotes.bid_size,
            "Ask size should not be constrained when long"
        );
    }

    #[test]
    fn test_hjb_kappa_affects_spread() {
        let solver = HJBSolver::default();

        // Low kappa → wide spread
        let low_kappa_beliefs = MarketBeliefs::with_priors(0.0001, 50.0, 0.5);
        let low_quotes = solver.optimal_quotes(&low_kappa_beliefs, 0.0, 1.0, 0.1);

        // High kappa → tight spread
        let high_kappa_beliefs = MarketBeliefs::with_priors(0.0001, 500.0, 0.5);
        let high_quotes = solver.optimal_quotes(&high_kappa_beliefs, 0.0, 1.0, 0.1);

        assert!(
            low_quotes.half_spread > high_quotes.half_spread,
            "Lower kappa should give wider spread: {} vs {}",
            low_quotes.half_spread,
            high_quotes.half_spread
        );
    }

    #[test]
    fn test_hjb_size_distribution() {
        let solver = HJBSolver::default();
        let beliefs = MarketBeliefs::default();

        let depths = vec![2.0, 5.0, 10.0, 20.0];
        let sizes = solver.size_distribution(&beliefs, &depths, 1.0, 0.0, 1.0, true);

        // Should have one size per depth
        assert_eq!(sizes.len(), depths.len());

        // Sizes should sum to approximately total (with min size floors)
        let total: f64 = sizes.iter().sum();
        assert!(total <= 1.0 + depths.len() as f64 * solver.config.min_order_size);
    }

    #[test]
    fn test_hjb_preemptive_skew_from_beliefs() {
        let solver = HJBSolver::default();
        let mut beliefs = MarketBeliefs::default();

        // Observe price dropping (no fills yet)
        for _ in 0..20 {
            beliefs.observe_price(-0.002, 1.0);
        }

        // Get quotes with zero position
        let quotes = solver.optimal_quotes(&beliefs, 0.0, 1.0, 0.1);

        // Should have negative predictive bias → skew toward sells
        assert!(
            quotes.predictive_bias < 0.0,
            "Should skew toward sells based on price observations alone"
        );
    }

    #[test]
    fn test_hjb_regime_gamma() {
        let config = HJBSolverConfig {
            use_regime_gamma: true,
            gamma: 0.5,
            regime_gamma_mults: [0.5, 1.0, 2.0, 5.0],
            ..Default::default()
        };

        let solver = HJBSolver::new(config);
        let mut beliefs = MarketBeliefs::default();

        // Set cascade regime
        beliefs.update_regime([0.0, 0.0, 0.0, 1.0]);

        let quotes = solver.optimal_quotes(&beliefs, 0.0, 1.0, 0.1);

        // Cascade should have higher effective gamma → wider spread
        // γ_eff = 0.5 × 5.0 = 2.5
        assert!(
            quotes.half_spread > 5.0,
            "Cascade regime should widen spreads"
        );
    }
}
