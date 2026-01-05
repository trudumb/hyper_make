//! HJB-derived optimal inventory control for market making.
//!
//! Implements the Avellaneda-Stoikov HJB (Hamilton-Jacobi-Bellman) solution
//! for optimal market making with inventory risk.
//!
//! The HJB equation:
//! ```text
//! ∂V/∂t + max_δ { λ(δ)[δ + V(t,x+δ,q-1,S) - V(t,x,q,S)] } - γσ²q² = 0
//! ```
//!
//! With terminal condition: `V(T,x,q,S) = x + q×S - penalty×q²`
//!
//! This module provides:
//! - Optimal inventory skew from the value function gradient
//! - Terminal penalty that forces position reduction before session end
//! - Funding rate integration for perpetuals (carry cost affects optimal inventory)
//! - Theoretically rigorous position management

use std::time::Instant;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for HJB inventory controller.
#[derive(Debug, Clone)]
pub struct HJBConfig {
    /// Session duration for terminal penalty (seconds)
    /// For 24/7 markets, use daily session (86400) or shorter sub-sessions
    pub session_duration_secs: f64,

    /// Terminal inventory penalty ($/unit²)
    /// Higher values force more aggressive position reduction near session end
    /// Typical: 0.0001 - 0.001 (0.01% - 0.1% per unit²)
    pub terminal_penalty: f64,

    /// Base risk aversion (γ)
    /// Used in diffusion skew calculation: γσ²qT
    pub gamma_base: f64,

    /// Funding rate half-life for EWMA (seconds)
    pub funding_ewma_half_life: f64,

    /// Minimum time remaining before terminal urgency kicks in (seconds)
    /// Avoids division by zero and extreme values near session end
    pub min_time_remaining: f64,

    /// Maximum terminal skew multiplier
    /// Caps the terminal penalty contribution to prevent extreme skews
    pub max_terminal_multiplier: f64,

    // === Drift-Adjusted Skew (First Principles Extension) ===
    /// Enable drift-adjusted skew from momentum signals.
    /// When true, incorporates predicted price drift into optimal skew.
    pub use_drift_adjusted_skew: bool,

    /// Sensitivity to momentum-position opposition [0.5, 3.0].
    /// Higher values increase skew urgency when position opposes momentum.
    pub opposition_sensitivity: f64,

    /// Maximum drift urgency multiplier [1.5, 5.0].
    /// Caps the drift contribution to prevent extreme skews.
    pub max_drift_urgency: f64,

    /// Minimum continuation probability to apply drift adjustment [0.3, 0.7].
    /// Below this, momentum is considered noise.
    pub min_continuation_prob: f64,
}

impl Default for HJBConfig {
    fn default() -> Self {
        Self {
            session_duration_secs: 86400.0, // 24 hour session
            terminal_penalty: 0.0005,       // 0.05% per unit²
            gamma_base: 0.3,                // Moderate risk aversion
            funding_ewma_half_life: 3600.0, // 1 hour
            min_time_remaining: 60.0,       // 1 minute minimum
            max_terminal_multiplier: 5.0,   // Cap at 5x normal skew
            // Drift-adjusted skew (enabled by default for first-principles trading)
            use_drift_adjusted_skew: true,
            opposition_sensitivity: 1.5,
            max_drift_urgency: 3.0,
            min_continuation_prob: 0.5,
        }
    }
}

// ============================================================================
// HJB Inventory Controller
// ============================================================================

/// HJB-derived optimal inventory controller.
///
/// Computes optimal inventory skew using the closed-form solution to the
/// Avellaneda-Stoikov HJB equation. Key features:
///
/// 1. **Diffusion Skew**: γσ²qT - standard A-S formula for inventory risk
/// 2. **Terminal Penalty**: Forces position reduction as session end approaches
/// 3. **Funding Integration**: Accounts for perpetual funding costs in carry
/// 4. **Optimal Inventory Target**: Not always zero (funding affects target)
///
/// The controller is stateful, tracking:
/// - Session timing for terminal penalty
/// - Funding rate EWMA for carry cost estimation
#[derive(Debug, Clone)]
pub struct HJBInventoryController {
    config: HJBConfig,

    /// Session start time
    session_start: Instant,

    /// Current volatility estimate (per-second)
    sigma: f64,

    /// Funding rate EWMA (annualized, positive = longs pay shorts)
    funding_rate_ewma: f64,

    /// EWMA alpha for funding rate
    funding_alpha: f64,

    /// Whether controller is initialized
    initialized: bool,
}

impl HJBInventoryController {
    /// Create a new HJB inventory controller.
    pub fn new(config: HJBConfig) -> Self {
        let funding_alpha = (2.0_f64.ln() / config.funding_ewma_half_life).clamp(0.0001, 1.0);

        Self {
            config,
            session_start: Instant::now(),
            sigma: 0.0001, // 1 bp/sec default
            funding_rate_ewma: 0.0,
            funding_alpha,
            initialized: false,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(HJBConfig::default())
    }

    /// Start a new session (resets terminal penalty timing).
    pub fn start_session(&mut self) {
        self.session_start = Instant::now();
        self.initialized = true;
    }

    /// Update volatility estimate.
    pub fn update_sigma(&mut self, sigma: f64) {
        self.sigma = sigma.max(1e-10); // Floor to avoid zero
    }

    /// Update funding rate.
    ///
    /// # Arguments
    /// * `funding_rate` - Current 8-hour funding rate (as decimal, e.g., 0.0001 = 0.01%)
    pub fn update_funding(&mut self, funding_rate: f64) {
        // Convert 8-hour rate to annualized
        let annualized = funding_rate * 3.0 * 365.0; // 3 periods/day × 365 days

        // EWMA update
        if self.initialized {
            self.funding_rate_ewma = self.funding_alpha * annualized
                + (1.0 - self.funding_alpha) * self.funding_rate_ewma;
        } else {
            self.funding_rate_ewma = annualized;
        }
    }

    /// Get time remaining in current session (seconds).
    pub fn time_remaining(&self) -> f64 {
        let elapsed = self.session_start.elapsed().as_secs_f64();
        (self.config.session_duration_secs - elapsed).max(self.config.min_time_remaining)
    }

    /// Get terminal urgency factor (0 = start of session, 1 = end of session).
    pub fn terminal_urgency(&self) -> f64 {
        let elapsed = self.session_start.elapsed().as_secs_f64();
        (elapsed / self.config.session_duration_secs).clamp(0.0, 1.0)
    }

    /// Compute optimal inventory skew from HJB solution.
    ///
    /// The full skew formula:
    /// ```text
    /// skew = γσ²qT + terminal_penalty × q × (1 - t/T) + funding_bias
    /// ```
    ///
    /// Where:
    /// - `γσ²qT`: Diffusion-driven inventory risk (standard A-S)
    /// - `terminal_penalty × q × (1 - t/T)`: Increases as session end approaches
    /// - `funding_bias`: Carry cost shifts optimal inventory target
    ///
    /// Returns skew in fractional units (multiply by price for absolute value).
    pub fn optimal_skew(&self, position: f64, max_position: f64) -> f64 {
        let gamma = self.config.gamma_base;
        let sigma = self.sigma;
        let time_remaining = self.time_remaining();

        // Normalize position to [-1, 1] range
        let q = if max_position > 1e-10 {
            (position / max_position).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // 1. Diffusion skew: γσ²qT
        // This is the standard Avellaneda-Stoikov inventory skew
        let diffusion_skew = gamma * sigma.powi(2) * q * time_remaining;

        // 2. Terminal penalty: increases skew as session end approaches
        // penalty × q × urgency, where urgency = 1 - time_remaining/session_duration
        let urgency = self.terminal_urgency();
        let terminal_skew = self.config.terminal_penalty * q * urgency;

        // Cap terminal contribution
        let terminal_skew_capped = terminal_skew
            .abs()
            .min(diffusion_skew.abs() * self.config.max_terminal_multiplier)
            * terminal_skew.signum();

        // 3. Funding bias: carry cost shifts optimal inventory
        // If funding positive (longs pay), we want to be short → adds positive skew
        // If funding negative (shorts pay), we want to be long → adds negative skew
        let funding_per_second = self.funding_rate_ewma / (365.0 * 24.0 * 3600.0);
        let funding_bias = funding_per_second * time_remaining * q.signum();

        diffusion_skew + terminal_skew_capped + funding_bias
    }

    /// Compute optimal inventory skew in basis points.
    pub fn optimal_skew_bps(&self, position: f64, max_position: f64) -> f64 {
        self.optimal_skew(position, max_position) * 10000.0
    }

    /// Compute drift-adjusted optimal skew from HJB solution with momentum.
    ///
    /// # Theory
    ///
    /// Standard HJB assumes price follows martingale: dS = σdW
    /// With momentum signals, price has drift: dS = μdt + σdW
    ///
    /// The extended HJB optimal skew becomes:
    /// ```text
    /// skew = γσ²qT + terminal_penalty + funding_bias + drift_urgency
    /// ```
    ///
    /// Where drift_urgency = μ × P(continuation) × time_exposure × opposition_factor
    ///
    /// When position **opposes** momentum (short + rising, long + falling):
    /// - Adverse moves are more likely (momentum predicts unfavorable direction)
    /// - We add urgency to accelerate position reduction
    ///
    /// When position **aligns** with momentum:
    /// - Risk is lower, we slightly reduce urgency but stay conservative
    ///
    /// # Arguments
    /// * `position` - Current position (positive = long, negative = short)
    /// * `max_position` - Maximum position for normalization
    /// * `momentum_bps` - Momentum in basis points (positive = rising)
    /// * `p_continuation` - Probability momentum continues [0, 1]
    ///
    /// # Returns
    /// Optimal skew in fractional units (multiply by price for absolute value).
    pub fn optimal_skew_with_drift(
        &self,
        position: f64,
        max_position: f64,
        momentum_bps: f64,
        p_continuation: f64,
    ) -> DriftAdjustedSkew {
        // Start with base HJB skew
        let base_skew = self.optimal_skew(position, max_position);

        if !self.config.use_drift_adjusted_skew {
            return DriftAdjustedSkew {
                total_skew: base_skew,
                base_skew,
                drift_urgency: 0.0,
                variance_multiplier: 1.0,
                is_opposed: false,
                urgency_score: 0.0,
            };
        }

        // Normalize position
        let q = if max_position > 1e-10 {
            (position / max_position).clamp(-1.0, 1.0)
        } else {
            return DriftAdjustedSkew {
                total_skew: base_skew,
                base_skew,
                drift_urgency: 0.0,
                variance_multiplier: 1.0,
                is_opposed: false,
                urgency_score: 0.0,
            };
        };

        // Check if position opposes momentum
        // Short (q < 0) + rising (momentum > 0) = opposed
        // Long (q > 0) + falling (momentum < 0) = opposed
        let is_opposed = q * momentum_bps < 0.0;

        // Only apply drift adjustment if:
        // 1. Position opposes momentum
        // 2. Continuation probability exceeds threshold
        // 3. Momentum is significant (> 10 bps)
        let momentum_significant = momentum_bps.abs() > 10.0;
        let continuation_confident = p_continuation >= self.config.min_continuation_prob;

        if !is_opposed || !momentum_significant || !continuation_confident {
            // No drift adjustment needed
            return DriftAdjustedSkew {
                total_skew: base_skew,
                base_skew,
                drift_urgency: 0.0,
                variance_multiplier: 1.0,
                is_opposed,
                urgency_score: 0.0,
            };
        }

        // === Compute Drift Urgency ===
        // From optimal control with drift: urgency ∝ μ × P(continue) × |q| × T
        let time_remaining = self.time_remaining().min(300.0); // Cap at 5 min exposure

        // Convert momentum_bps to fractional drift rate
        // Assume momentum measured over 500ms, so drift = momentum_bps / 10000 / 0.5
        let drift_rate = (momentum_bps / 10000.0) / 0.5;

        // Urgency formula:
        // drift_urgency = sensitivity × drift_rate × P(continue) × |q| × T
        let raw_urgency = self.config.opposition_sensitivity
            * drift_rate.abs()
            * p_continuation
            * q.abs()
            * time_remaining;

        // Cap urgency
        let max_base_urgency = base_skew.abs() * self.config.max_drift_urgency;
        let drift_urgency_magnitude = raw_urgency.min(max_base_urgency);

        // Sign: urgency should amplify the base skew direction
        // If short (q < 0), base skew is negative (quotes shift up)
        // Urgency should make it MORE negative (more aggressive buying)
        let drift_urgency = drift_urgency_magnitude * q.signum();

        // === Compute Variance Multiplier ===
        // When opposed, increase effective variance for inventory risk
        // σ²_eff = σ² × (1 + κ × |momentum/σ| × P(continue))
        let momentum_vol_ratio = if self.sigma > 1e-10 {
            ((momentum_bps / 10000.0) / self.sigma).abs().min(3.0)
        } else {
            0.0
        };

        let variance_multiplier = 1.0
            + self.config.opposition_sensitivity * momentum_vol_ratio * p_continuation * q.abs();
        let variance_multiplier_capped = variance_multiplier.min(self.config.max_drift_urgency);

        // Urgency score for diagnostics [0, 5]
        let urgency_score = (momentum_bps.abs() / 50.0).min(1.0) // Momentum strength
            + p_continuation // Continuation confidence
            + q.abs() // Position size
            + momentum_vol_ratio.min(1.0) // Vol-adjusted momentum
            + if self.is_terminal_zone() { 1.0 } else { 0.0 }; // Terminal zone boost

        DriftAdjustedSkew {
            total_skew: base_skew + drift_urgency,
            base_skew,
            drift_urgency,
            variance_multiplier: variance_multiplier_capped,
            is_opposed,
            urgency_score: urgency_score.min(5.0),
        }
    }

    /// Compute drift-adjusted skew in basis points.
    pub fn optimal_skew_with_drift_bps(
        &self,
        position: f64,
        max_position: f64,
        momentum_bps: f64,
        p_continuation: f64,
    ) -> DriftAdjustedSkew {
        let mut result =
            self.optimal_skew_with_drift(position, max_position, momentum_bps, p_continuation);
        result.total_skew *= 10000.0;
        result.base_skew *= 10000.0;
        result.drift_urgency *= 10000.0;
        result
    }

    /// Compute the optimal inventory target (not always zero for perpetuals).
    ///
    /// Theory: With non-zero funding rate, the optimal inventory is:
    /// ```text
    /// q* = -funding_rate / (2 × terminal_penalty)
    /// ```
    ///
    /// Positive funding → optimal to be short
    /// Negative funding → optimal to be long
    ///
    /// Returns target as fraction of max_position.
    pub fn optimal_inventory_target(&self) -> f64 {
        if self.config.terminal_penalty.abs() < 1e-10 {
            return 0.0;
        }

        let funding_per_second = self.funding_rate_ewma / (365.0 * 24.0 * 3600.0);

        // Target = -funding / (2 × penalty)
        let target = -funding_per_second / (2.0 * self.config.terminal_penalty);

        // Clamp to reasonable range
        target.clamp(-0.5, 0.5)
    }

    /// Compute how much to adjust gamma based on session timing.
    ///
    /// Near session end, we want to be more aggressive about reducing
    /// inventory, which means higher gamma.
    ///
    /// Returns a multiplier to apply to base gamma.
    pub fn gamma_multiplier(&self) -> f64 {
        let urgency = self.terminal_urgency();

        // Ramp up gamma near session end
        // At t=0: multiplier = 1.0
        // At t=T: multiplier = 1.0 + max_terminal_multiplier
        1.0 + urgency * (self.config.max_terminal_multiplier - 1.0)
    }

    /// Get effective gamma (base × multiplier).
    pub fn effective_gamma(&self) -> f64 {
        self.config.gamma_base * self.gamma_multiplier()
    }

    /// Compute the value function gradient ∂V/∂q at current state.
    ///
    /// This gives the marginal cost of holding one more unit of inventory.
    /// Useful for diagnostics and sizing decisions.
    ///
    /// ∂V/∂q ≈ -S - 2γσ²qT - 2×penalty×q×urgency + funding×T
    pub fn value_gradient(&self, position: f64, max_position: f64, price: f64) -> f64 {
        let gamma = self.config.gamma_base;
        let sigma = self.sigma;
        let time_remaining = self.time_remaining();
        let urgency = self.terminal_urgency();

        let q = if max_position > 1e-10 {
            (position / max_position).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // The gradient captures the marginal cost of inventory
        let inventory_cost = 2.0 * gamma * sigma.powi(2) * q * time_remaining;
        let terminal_cost = 2.0 * self.config.terminal_penalty * q * urgency;
        let funding_benefit = self.funding_rate_ewma / (365.0 * 24.0 * 3600.0) * time_remaining;

        // Negative of costs (value decreases with costs)
        -price - inventory_cost - terminal_cost + funding_benefit * q.signum()
    }

    /// Check if we're in terminal urgency zone (last portion of session).
    ///
    /// Returns true if urgency > 0.8 (last 20% of session).
    pub fn is_terminal_zone(&self) -> bool {
        self.terminal_urgency() > 0.8
    }

    /// Get summary for diagnostics.
    pub fn summary(&self) -> HJBSummary {
        HJBSummary {
            time_remaining_secs: self.time_remaining(),
            terminal_urgency: self.terminal_urgency(),
            is_terminal_zone: self.is_terminal_zone(),
            gamma_multiplier: self.gamma_multiplier(),
            effective_gamma: self.effective_gamma(),
            funding_rate_ewma: self.funding_rate_ewma,
            optimal_inventory_target: self.optimal_inventory_target(),
            sigma: self.sigma,
        }
    }
}

/// Output from drift-adjusted skew calculation.
///
/// Contains both the total skew and its components for analysis.
#[derive(Debug, Clone, Copy, Default)]
pub struct DriftAdjustedSkew {
    /// Total optimal skew (base + drift urgency).
    pub total_skew: f64,

    /// Base HJB skew (γσ²qT + terminal + funding).
    pub base_skew: f64,

    /// Additional urgency from momentum-position opposition.
    /// Same sign as base_skew when opposed (amplifies reduction).
    pub drift_urgency: f64,

    /// Variance multiplier for inventory risk.
    /// > 1.0 when opposed, used for σ²_effective calculation.
    pub variance_multiplier: f64,

    /// Whether position is opposed to momentum.
    pub is_opposed: bool,

    /// Urgency score [0, 5] for diagnostics.
    pub urgency_score: f64,
}

/// Summary of HJB controller state for diagnostics.
#[derive(Debug, Clone)]
pub struct HJBSummary {
    pub time_remaining_secs: f64,
    pub terminal_urgency: f64,
    pub is_terminal_zone: bool,
    pub gamma_multiplier: f64,
    pub effective_gamma: f64,
    pub funding_rate_ewma: f64,
    pub optimal_inventory_target: f64,
    pub sigma: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_controller() -> HJBInventoryController {
        let config = HJBConfig {
            session_duration_secs: 100.0, // Short session for testing
            terminal_penalty: 0.001,
            gamma_base: 0.3,
            ..Default::default()
        };
        let mut ctrl = HJBInventoryController::new(config);
        ctrl.start_session();
        ctrl.update_sigma(0.0001); // 1 bp/sec
        ctrl
    }

    #[test]
    fn test_hjb_basic() {
        let ctrl = make_controller();

        // Should be initialized
        assert!(ctrl.initialized);
        assert!(ctrl.time_remaining() > 0.0);
        assert!(ctrl.terminal_urgency() < 0.5); // Early in session
    }

    #[test]
    fn test_hjb_zero_position_zero_skew() {
        let ctrl = make_controller();

        // With zero position, skew should be ~zero (funding aside)
        let skew = ctrl.optimal_skew(0.0, 1.0);
        assert!(
            skew.abs() < 1e-6,
            "Zero position should give ~zero skew: {}",
            skew
        );
    }

    #[test]
    fn test_hjb_long_position_positive_skew() {
        let ctrl = make_controller();

        // Long position should give positive skew (shift quotes down)
        let skew = ctrl.optimal_skew(0.5, 1.0);
        assert!(
            skew > 0.0,
            "Long position should give positive skew: {}",
            skew
        );
    }

    #[test]
    fn test_hjb_short_position_negative_skew() {
        let ctrl = make_controller();

        // Short position should give negative skew (shift quotes up)
        let skew = ctrl.optimal_skew(-0.5, 1.0);
        assert!(
            skew < 0.0,
            "Short position should give negative skew: {}",
            skew
        );
    }

    #[test]
    fn test_hjb_skew_symmetry() {
        let ctrl = make_controller();

        // Skew should be antisymmetric in position
        let skew_long = ctrl.optimal_skew(0.5, 1.0);
        let skew_short = ctrl.optimal_skew(-0.5, 1.0);

        assert!(
            (skew_long + skew_short).abs() < 1e-8,
            "Skew should be antisymmetric: long={}, short={}",
            skew_long,
            skew_short
        );
    }

    #[test]
    fn test_hjb_gamma_multiplier() {
        let ctrl = make_controller();

        // At start of session, multiplier should be ~1.0
        let mult = ctrl.gamma_multiplier();
        assert!(
            mult >= 1.0 && mult < 1.5,
            "Early multiplier should be near 1.0: {}",
            mult
        );

        // Effective gamma should be base × multiplier
        let eff = ctrl.effective_gamma();
        assert!((eff - ctrl.config.gamma_base * mult).abs() < 1e-6);
    }

    #[test]
    fn test_hjb_optimal_inventory_target_no_funding() {
        let ctrl = make_controller();

        // With zero funding, optimal target is zero
        let target = ctrl.optimal_inventory_target();
        assert!(
            target.abs() < 0.01,
            "With zero funding, target should be ~0: {}",
            target
        );
    }

    #[test]
    fn test_hjb_optimal_inventory_target_with_funding() {
        let mut ctrl = make_controller();

        // Positive funding rate (longs pay) → optimal to be short
        ctrl.update_funding(0.001); // 0.1% 8-hour rate
        let target = ctrl.optimal_inventory_target();

        // With positive funding, target should be negative (short)
        assert!(
            target < 0.0,
            "Positive funding should give negative target: {}",
            target
        );
    }

    #[test]
    fn test_hjb_funding_ewma() {
        // Use a faster-converging controller for testing
        let config = HJBConfig {
            session_duration_secs: 100.0,
            terminal_penalty: 0.001,
            gamma_base: 0.3,
            funding_ewma_half_life: 10.0, // Fast EWMA for testing (10 seconds)
            ..Default::default()
        };
        let mut ctrl = HJBInventoryController::new(config);
        ctrl.start_session();

        // Initial funding rate
        ctrl.update_funding(0.001);
        let rate1 = ctrl.funding_rate_ewma;

        // Update with same rate - EWMA should move toward target
        ctrl.update_funding(0.001);
        let rate2 = ctrl.funding_rate_ewma;

        // Expected annualized rate
        let annualized = 0.001 * 3.0 * 365.0; // = 1.095

        // EWMA should be moving toward the annualized rate
        assert!(
            rate2 > rate1,
            "EWMA should increase toward target: {} -> {}",
            rate1,
            rate2
        );
        assert!(rate2 < annualized, "EWMA should not exceed target");

        // After many updates with fast EWMA, should converge
        for _ in 0..100 {
            ctrl.update_funding(0.001);
        }
        let rate_converged = ctrl.funding_rate_ewma;

        // Should be close to annualized after convergence
        assert!(
            (rate_converged - annualized).abs() / annualized < 0.1,
            "EWMA should converge to annualized rate: {} vs {}",
            rate_converged,
            annualized
        );
    }

    #[test]
    fn test_hjb_value_gradient() {
        let ctrl = make_controller();

        // Value gradient at zero position
        let grad_zero = ctrl.value_gradient(0.0, 1.0, 100.0);

        // Value gradient with long position (should be more negative = higher cost)
        let grad_long = ctrl.value_gradient(0.5, 1.0, 100.0);

        // Holding inventory has cost, so gradient should differ
        // (exact relationship depends on parameters)
        assert!(grad_zero != grad_long, "Gradient should depend on position");
    }

    #[test]
    fn test_hjb_terminal_zone() {
        let ctrl = make_controller();

        // Early in session, not in terminal zone
        assert!(!ctrl.is_terminal_zone());
    }

    #[test]
    fn test_hjb_summary() {
        let ctrl = make_controller();
        let summary = ctrl.summary();

        assert!(summary.time_remaining_secs > 0.0);
        assert!(summary.terminal_urgency >= 0.0 && summary.terminal_urgency <= 1.0);
        assert!(summary.gamma_multiplier >= 1.0);
        assert!(summary.sigma > 0.0);
    }

    #[test]
    fn test_hjb_skew_increases_with_position() {
        let ctrl = make_controller();

        // Larger position → larger skew magnitude
        let skew_small = ctrl.optimal_skew(0.1, 1.0);
        let skew_large = ctrl.optimal_skew(0.5, 1.0);

        assert!(
            skew_large.abs() > skew_small.abs(),
            "Larger position should give larger skew: small={}, large={}",
            skew_small,
            skew_large
        );
    }

    #[test]
    fn test_hjb_skew_increases_with_volatility() {
        let mut ctrl = make_controller();

        ctrl.update_sigma(0.0001);
        let skew_low_vol = ctrl.optimal_skew(0.5, 1.0);

        ctrl.update_sigma(0.001); // 10x higher vol
        let skew_high_vol = ctrl.optimal_skew(0.5, 1.0);

        assert!(
            skew_high_vol.abs() > skew_low_vol.abs(),
            "Higher vol should give larger skew: low={}, high={}",
            skew_low_vol,
            skew_high_vol
        );
    }
}
