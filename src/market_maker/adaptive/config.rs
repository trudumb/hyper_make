//! Configuration for the Adaptive Bayesian Market Maker system.

use serde::{Deserialize, Serialize};

/// Configuration for the Adaptive Bayesian Market Maker.
///
/// This replaces static parameters with learned, adaptive ones while
/// preserving stochastic predictive power for risk management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveBayesianConfig {
    // === Learned Floor (Component 1) ===
    /// Prior mean for adverse selection (fraction, e.g., 0.0003 = 3 bps)
    pub as_prior_mean: f64,

    /// Prior standard deviation for adverse selection
    pub as_prior_std: f64,

    /// Risk multiplier k for floor = fees + E[AS] + k×σ_AS
    /// Higher k = more conservative (wider floor)
    pub floor_risk_k: f64,

    /// Hard minimum floor (tick size as fraction)
    pub floor_absolute_min: f64,

    /// AS measurement horizon in milliseconds (time to measure mid movement)
    pub as_horizon_ms: u64,

    /// EWMA decay for AS estimation (handles non-stationarity)
    /// 0.99 = ~100 observation half-life
    pub as_ewma_decay: f64,

    // === Shrinkage Gamma (Component 2) ===
    /// Base gamma (risk aversion in normal conditions)
    pub gamma_base: f64,

    /// Initial global shrinkage τ (controls total adjustment magnitude)
    /// Smaller τ = more shrinkage toward base
    pub tau_initial: f64,

    /// Learning rate for weight updates
    pub gamma_learning_rate: f64,

    /// Minimum gamma bound
    pub gamma_min: f64,

    /// Maximum gamma bound
    pub gamma_max: f64,

    /// Signals to include in gamma adjustment
    /// Each signal gets its own learned weight
    pub gamma_signals: Vec<GammaSignal>,

    // === Blended Kappa (Component 3) ===
    /// Prior mean for Bayesian kappa (1/avg_fill_distance)
    /// κ=2500 → 4 bps average fill distance (liquid BTC)
    pub kappa_prior_mean: f64,

    /// Prior strength (effective sample size)
    /// Lower = faster adaptation to data
    pub kappa_prior_strength: f64,

    /// Minimum own fills before blending starts
    pub kappa_blend_min_fills: usize,

    /// Blend sigmoid steepness (higher = sharper transition)
    pub kappa_blend_scale: f64,

    /// Warmup conservatism factor (< 1.0 widens spread during warmup)
    pub kappa_warmup_factor: f64,

    // === Fill Rate Controller (Component 4) ===
    /// Target fill rate (fills per second)
    /// 0.02 = 1 fill per 50 seconds
    pub target_fill_rate: f64,

    /// Ceiling multiplier (allow GLFT to exceed fill target by this factor)
    pub fill_ceiling_mult: f64,

    /// Minimum observation time before controller activates (seconds)
    pub fill_min_observation_secs: f64,

    /// EWMA decay for fill rate estimation
    pub fill_rate_decay: f64,

    // === Maker Fee ===
    /// Maker fee rate (fraction of notional)
    pub maker_fee_rate: f64,

    // === Integration ===
    /// Enable adaptive floor (vs static min_spread_floor)
    pub enable_adaptive_floor: bool,

    /// Enable shrinkage gamma (vs multiplicative scaling)
    pub enable_shrinkage_gamma: bool,

    /// Enable fill rate controller
    pub enable_fill_controller: bool,

    /// Enable blended kappa (vs pure Bayesian)
    pub enable_blended_kappa: bool,
}

/// Signals that can influence gamma adjustment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GammaSignal {
    /// Volatility ratio (σ / σ_baseline)
    VolatilityRatio,
    /// Jump ratio (RV / BV)
    JumpRatio,
    /// Inventory utilization (|q| / q_max)
    InventoryUtilization,
    /// Hawkes intensity percentile
    HawkesIntensity,
    /// Spread regime indicator
    SpreadRegime,
    /// Cascade severity
    CascadeSeverity,
}

impl GammaSignal {
    /// Get the name of this signal for logging.
    pub fn name(&self) -> &'static str {
        match self {
            GammaSignal::VolatilityRatio => "vol_ratio",
            GammaSignal::JumpRatio => "jump_ratio",
            GammaSignal::InventoryUtilization => "inventory",
            GammaSignal::HawkesIntensity => "hawkes",
            GammaSignal::SpreadRegime => "spread_regime",
            GammaSignal::CascadeSeverity => "cascade",
        }
    }
}

impl Default for AdaptiveBayesianConfig {
    fn default() -> Self {
        Self {
            // Learned Floor - calibrated from trade history analysis
            // Trade history showed: fee 1.5 bps + AS ~3 bps = ~4.5 bps break-even per side
            // FIXED: Increased safety margin to achieve 8 bps floor per CLAUDE.md recommendation
            // floor = fee + E[AS] + k×σ = 1.5 + 3 + 1.17×3 = 8 bps (half-spread)
            as_prior_mean: 0.0003, // 3 bps prior AS (calibrated from Dec 2025 trades)
            as_prior_std: 0.0003,  // 3 bps uncertainty (tightened from 5 bps)
            floor_risk_k: 1.17,    // 1.17σ safety margin → floor = 1.5 + 3 + 3.5 = 8 bps
            floor_absolute_min: 0.0001, // 1 bp hard floor (tick size)
            as_horizon_ms: 1000,   // 1 second AS measurement
            as_ewma_decay: 0.995,  // ~200 obs half-life

            // Shrinkage Gamma - moderate base, strong shrinkage
            gamma_base: 0.3,
            tau_initial: 0.1,           // Small initial adjustments
            gamma_learning_rate: 0.001, // Slow learning for stability
            gamma_min: 0.05,
            gamma_max: 2.0,
            gamma_signals: vec![
                GammaSignal::VolatilityRatio,
                GammaSignal::JumpRatio,
                GammaSignal::InventoryUtilization,
                GammaSignal::HawkesIntensity,
            ],

            // Blended Kappa - liquid market priors
            kappa_prior_mean: 2500.0,  // 4 bps avg fill distance
            kappa_prior_strength: 5.0, // Quick adaptation
            kappa_blend_min_fills: 10, // Need 10 fills before trusting own data
            kappa_blend_scale: 5.0,    // Moderate transition steepness
            kappa_warmup_factor: 0.8,  // 20% conservative during warmup

            // Fill Rate Controller
            target_fill_rate: 0.02,           // 1 fill per 50 seconds
            fill_ceiling_mult: 1.5,           // Allow 50% wider than fill target
            fill_min_observation_secs: 120.0, // 2 minutes warmup
            fill_rate_decay: 0.995,           // ~200 obs half-life

            // Fees - Hyperliquid maker fee is 1.5 bps (round-trip 3 bps)
            maker_fee_rate: 0.00015, // 1.5 bps maker fee

            // Integration - all enabled by default
            enable_adaptive_floor: true,
            enable_shrinkage_gamma: true,
            enable_fill_controller: true,
            enable_blended_kappa: true,
        }
    }
}

impl AdaptiveBayesianConfig {
    /// Create a config optimized for tight, competitive quoting.
    pub fn competitive() -> Self {
        Self {
            floor_risk_k: 1.0,      // Less conservative floor
            gamma_base: 0.2,        // Lower base gamma = tighter
            tau_initial: 0.05,      // More shrinkage
            target_fill_rate: 0.05, // Higher fill target
            ..Default::default()
        }
    }

    /// Create a config optimized for conservative, safe quoting.
    pub fn conservative() -> Self {
        Self {
            floor_risk_k: 2.0,      // More conservative floor
            gamma_base: 0.5,        // Higher base gamma = wider
            tau_initial: 0.2,       // Less shrinkage (allow adjustments)
            target_fill_rate: 0.01, // Lower fill target
            ..Default::default()
        }
    }

    /// Create a config with specific maker fee rate.
    pub fn with_maker_fee(mut self, fee_rate: f64) -> Self {
        self.maker_fee_rate = fee_rate;
        self
    }

    /// Create a config with specific gamma base.
    pub fn with_gamma_base(mut self, gamma: f64) -> Self {
        self.gamma_base = gamma;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AdaptiveBayesianConfig::default();
        assert!(config.gamma_base > 0.0);
        assert!(config.maker_fee_rate > 0.0);
        assert!(config.enable_adaptive_floor);
    }

    #[test]
    fn test_competitive_config() {
        let config = AdaptiveBayesianConfig::competitive();
        let default = AdaptiveBayesianConfig::default();

        assert!(config.gamma_base < default.gamma_base);
        assert!(config.target_fill_rate > default.target_fill_rate);
    }

    #[test]
    fn test_conservative_config() {
        let config = AdaptiveBayesianConfig::conservative();
        let default = AdaptiveBayesianConfig::default();

        assert!(config.gamma_base > default.gamma_base);
        assert!(config.target_fill_rate < default.target_fill_rate);
    }
}
