//! Dynamic depth generation for ladder quoting.
//!
//! Computes optimal quote depths from market parameters using GLFT theory:
//! - **GLFT optimal spread**: δ* = (1/γ) × ln(1 + γ/κ)
//! - **Asymmetric depths**: Separate bid/ask depths when κ_bid ≠ κ_ask
//! - **Regime-adaptive**: Depths widen with volatility, tighten with liquidity
//!
//! # Key Components
//!
//! - [`DynamicDepthConfig`]: Configuration for depth generation
//! - [`DynamicDepthGenerator`]: Main generator that computes optimal depths
//! - [`DynamicDepths`]: Per-side depth arrays for bid and ask ladders
//!
//! # Mathematical Foundation
//!
//! The GLFT optimal half-spread is derived from stochastic control theory:
//!
//! ```text
//! δ* = (1/γ) × ln(1 + γ/κ)
//! ```
//!
//! Where:
//! - δ* = optimal half-spread (as fraction of price)
//! - γ = risk aversion parameter (higher = wider spreads)
//! - κ = order arrival intensity (higher = tighter spreads)
//!
//! # Usage
//!
//! ```rust,ignore
//! let config = DynamicDepthConfig::default();
//! let generator = DynamicDepthGenerator::new(config);
//!
//! // Compute depths from market params
//! let depths = generator.compute_depths(
//!     gamma,      // effective risk aversion
//!     kappa_bid,  // bid-side fill intensity
//!     kappa_ask,  // ask-side fill intensity
//!     sigma,      // volatility
//! );
//!
//! // Use depths in ladder generation
//! let ladder = Ladder::generate_with_depths(&ladder_config, &params, &depths);
//! ```

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::EPSILON;

/// Dynamic depth configuration (per-side)
///
/// Contains separate bid and ask depth arrays, enabling asymmetric quoting
/// when order flow intensity differs between sides.
#[derive(Debug, Clone, Default)]
pub struct DynamicDepths {
    /// Bid-side depths in basis points (best to worst, sorted ascending)
    pub bid: Vec<f64>,
    /// Ask-side depths in basis points (best to worst, sorted ascending)
    pub ask: Vec<f64>,
}

impl DynamicDepths {
    /// Create symmetric depths (same for both sides)
    pub fn symmetric(depths: Vec<f64>) -> Self {
        Self {
            bid: depths.clone(),
            ask: depths,
        }
    }

    /// Create asymmetric depths
    pub fn asymmetric(bid: Vec<f64>, ask: Vec<f64>) -> Self {
        Self { bid, ask }
    }

    /// Check if depths are valid (non-empty with positive values)
    pub fn is_valid(&self) -> bool {
        !self.bid.is_empty()
            && !self.ask.is_empty()
            && self.bid.iter().all(|&d| d > 0.0)
            && self.ask.iter().all(|&d| d > 0.0)
    }

    /// Get the tightest bid depth (best bid)
    pub fn best_bid_depth(&self) -> Option<f64> {
        self.bid.first().copied()
    }

    /// Get the tightest ask depth (best ask)
    pub fn best_ask_depth(&self) -> Option<f64> {
        self.ask.first().copied()
    }

    /// Get total spread at touch (best bid + best ask depths)
    pub fn spread_at_touch(&self) -> Option<f64> {
        match (self.best_bid_depth(), self.best_ask_depth()) {
            (Some(bid), Some(ask)) => Some(bid + ask),
            _ => None,
        }
    }
}

/// Depth spacing mode for ladder generation
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum DepthSpacing {
    /// Geometric spacing: depths grow exponentially from optimal
    /// Example with ratio 1.2: [δ*, δ*×1.2, δ*×1.44, δ*×1.728, ...]
    #[default]
    Geometric,
    /// Linear spacing: depths grow by fixed step from optimal
    /// Example with step 2bp: [δ*, δ*+2, δ*+4, δ*+6, ...]
    Linear,
}

/// Configuration for dynamic depth generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicDepthConfig {
    /// Number of levels per side
    pub num_levels: usize,

    /// Minimum practical depth in basis points (exchange tick + safety margin)
    /// Depths below this are clamped up to ensure executable quotes
    pub min_depth_bps: f64,

    /// Maximum depth as multiple of optimal spread
    /// Prevents quotes from going too deep when optimal is wide
    pub max_depth_multiple: f64,

    /// Absolute maximum depth in basis points
    /// Hard cap regardless of optimal spread calculation
    pub max_depth_bps: f64,

    /// Depth spacing mode (geometric or linear)
    pub spacing: DepthSpacing,

    /// Geometric ratio for spacing (only used when spacing = Geometric)
    /// Each level is ratio × previous level
    pub geometric_ratio: f64,

    /// Linear step for spacing in bps (only used when spacing = Linear)
    /// Each level is previous + step_bps
    pub linear_step_bps: f64,

    /// Maker fee rate (as fraction, e.g., 0.0001 = 1bp)
    /// Added to optimal spread to ensure profitability
    pub maker_fee_rate: f64,

    /// Minimum spread floor in basis points
    /// Ensures we never quote tighter than this even in very liquid conditions
    pub min_spread_floor_bps: f64,

    /// Enable asymmetric depths based on bid/ask kappa difference
    pub enable_asymmetric: bool,

    /// Maximum spread as multiple of observed market spread
    /// Caps GLFT optimal to prevent quoting excessively wide
    /// Set to 0.0 to disable (use only GLFT optimal)
    pub market_spread_cap_multiple: f64,

    /// Hard maximum spread per side in basis points (absolute cap).
    /// Applied after all other spread calculations as a final safety valve.
    /// Use for illiquid assets where GLFT may produce very wide spreads.
    /// Set to 0.0 to disable (no hard cap).
    /// Example: max_spread_per_side_bps = 15.0 → never quote wider than 15 bps per side
    pub max_spread_per_side_bps: f64,
}

impl Default for DynamicDepthConfig {
    fn default() -> Self {
        Self {
            num_levels: 5,
            min_depth_bps: 1.0,      // At least 1bp from mid
            max_depth_multiple: 5.0, // Up to 5x optimal spread
            max_depth_bps: 200.0,    // Hard cap at 200bp
            spacing: DepthSpacing::Geometric,
            geometric_ratio: 1.2, // Each level 20% further than previous (tighter clustering near touch)
            linear_step_bps: 3.0, // Or 3bp steps for linear
            maker_fee_rate: 0.00015, // 1.5bp maker fee (Hyperliquid actual)
            // FIRST PRINCIPLES: This is a safety floor, not the primary floor.
            // The adaptive floor from AdaptiveBayesianConfig is the effective floor.
            // Set to 1.0 so GLFT optimal determines the actual spread floor.
            // The adaptive floor (when active) takes precedence at ~6 bps.
            min_spread_floor_bps: 1.0, // Safety floor only; GLFT and adaptive floor drive actual spread
            enable_asymmetric: true,
            // FIRST PRINCIPLES: GLFT optimal spread δ* = (1/γ) × ln(1 + γ/κ)
            // is derived from stochastic control theory and should be trusted.
            // Capping below optimal sacrifices edge for competitive appearance.
            //
            // Set to 0.0 to disable capping entirely (recommended for profitability).
            // Trade history showed capping at 5.0 (2.5× per side) caused -$562 loss.
            //
            // If competitive quoting is needed, set to 15.0+ (7.5× per side)
            // to allow most GLFT optimal spreads through.
            market_spread_cap_multiple: 0.0, // DISABLED - trust GLFT theory

            // Hard cap disabled by default - trust GLFT theory.
            // Enable via --max-spread-bps CLI for illiquid assets.
            max_spread_per_side_bps: 0.0,
        }
    }
}

/// Dynamic depth generator using GLFT optimal spread theory
///
/// Computes optimal quote depths from market parameters, adapting to:
/// - Risk aversion (γ): Higher γ → wider spreads
/// - Order flow intensity (κ): Higher κ → tighter spreads
/// - Volatility regime: Can incorporate σ for regime-dependent adjustments
#[derive(Debug, Clone)]
pub struct DynamicDepthGenerator {
    config: DynamicDepthConfig,
}

impl DynamicDepthGenerator {
    /// Create a new depth generator with the given configuration
    pub fn new(config: DynamicDepthConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(DynamicDepthConfig::default())
    }

    /// Get the configuration
    pub fn config(&self) -> &DynamicDepthConfig {
        &self.config
    }

    /// Compute GLFT optimal half-spread
    ///
    /// δ* = (1/γ) × ln(1 + γ/κ) + maker_fee
    ///
    /// This is the theoretically optimal spread from Guéant-Lehalle-Fernandez-Tapia (2012).
    /// The fee term comes from the modified HJB equation where maker receives δ - fee.
    ///
    /// # Arguments
    /// * `gamma` - Risk aversion parameter (typically 0.1-1.0)
    /// * `kappa` - Order arrival intensity (fills per unit exposure)
    ///
    /// # Returns
    /// Optimal half-spread as a fraction of price (e.g., 0.001 = 10bps)
    pub fn glft_optimal_spread(&self, gamma: f64, kappa: f64) -> f64 {
        let ratio = gamma / kappa.max(EPSILON);

        let glft_spread = if ratio > 1e-9 && gamma > 1e-9 {
            (1.0 / gamma) * (1.0 + ratio).ln()
        } else {
            // When γ/κ → 0, use Taylor expansion: ln(1+x) ≈ x
            // δ ≈ (1/γ) × (γ/κ) = 1/κ
            1.0 / kappa.max(1.0)
        };

        // Add maker fee to ensure profitability
        glft_spread + self.config.maker_fee_rate
    }

    /// Convert spread fraction to basis points
    fn to_bps(fraction: f64) -> f64 {
        fraction * 10000.0
    }

    /// Compute dynamic depths from market parameters
    ///
    /// # Arguments
    /// * `gamma` - Effective risk aversion (after all scaling factors)
    /// * `kappa_bid` - Bid-side order flow intensity
    /// * `kappa_ask` - Ask-side order flow intensity
    /// * `_sigma` - Volatility (reserved for future regime-based adjustments)
    ///
    /// # Returns
    /// [`DynamicDepths`] with per-side depth arrays
    pub fn compute_depths(
        &self,
        gamma: f64,
        kappa_bid: f64,
        kappa_ask: f64,
        _sigma: f64,
    ) -> DynamicDepths {
        // Compute optimal spreads for each side
        let optimal_bid = self.glft_optimal_spread(gamma, kappa_bid);
        let optimal_ask = self.glft_optimal_spread(gamma, kappa_ask);

        // Convert to basis points
        let optimal_bid_bps = Self::to_bps(optimal_bid);
        let optimal_ask_bps = Self::to_bps(optimal_ask);

        // Apply floor and ceiling
        let mut optimal_bid_bps = optimal_bid_bps
            .max(self.config.min_spread_floor_bps)
            .max(self.config.min_depth_bps);
        let mut optimal_ask_bps = optimal_ask_bps
            .max(self.config.min_spread_floor_bps)
            .max(self.config.min_depth_bps);

        // Apply hard cap if configured (for competitive spreads on illiquid assets)
        if self.config.max_spread_per_side_bps > 0.0 {
            optimal_bid_bps = optimal_bid_bps.min(self.config.max_spread_per_side_bps);
            optimal_ask_bps = optimal_ask_bps.min(self.config.max_spread_per_side_bps);
        }

        // Generate depth arrays
        let bid_depths = self.generate_depth_array(optimal_bid_bps);
        let ask_depths = if self.config.enable_asymmetric {
            self.generate_depth_array(optimal_ask_bps)
        } else {
            bid_depths.clone()
        };

        let depths = DynamicDepths {
            bid: bid_depths,
            ask: ask_depths,
        };

        debug!(
            gamma = %format!("{:.3}", gamma),
            kappa_bid = %format!("{:.1}", kappa_bid),
            kappa_ask = %format!("{:.1}", kappa_ask),
            optimal_bid_bps = %format!("{:.2}", optimal_bid_bps),
            optimal_ask_bps = %format!("{:.2}", optimal_ask_bps),
            num_bid_levels = depths.bid.len(),
            num_ask_levels = depths.ask.len(),
            best_bid_depth = ?depths.best_bid_depth(),
            best_ask_depth = ?depths.best_ask_depth(),
            "Dynamic depths computed"
        );

        depths
    }

    /// Compute symmetric depths using average of bid/ask kappa
    ///
    /// Useful when you don't have separate bid/ask flow estimates
    pub fn compute_symmetric_depths(&self, gamma: f64, kappa: f64, sigma: f64) -> DynamicDepths {
        self.compute_depths(gamma, kappa, kappa, sigma)
    }

    /// Compute depths with market spread cap.
    ///
    /// Same as `compute_depths` but caps the GLFT optimal spread to a multiple
    /// of the observed market spread. This prevents quoting excessively wide
    /// when GLFT parameters suggest spreads far from market reality.
    ///
    /// # Arguments
    /// * `gamma` - Effective risk aversion
    /// * `kappa_bid` - Bid-side order flow intensity
    /// * `kappa_ask` - Ask-side order flow intensity
    /// * `sigma` - Volatility
    /// * `market_spread_bps` - Current observed market spread in basis points
    pub fn compute_depths_with_market_cap(
        &self,
        gamma: f64,
        kappa_bid: f64,
        kappa_ask: f64,
        sigma: f64,
        market_spread_bps: f64,
    ) -> DynamicDepths {
        // If market spread cap is disabled or market spread is invalid, use standard method
        if self.config.market_spread_cap_multiple <= 0.0 || market_spread_bps <= 0.0 {
            return self.compute_depths(gamma, kappa_bid, kappa_ask, sigma);
        }

        // Compute GLFT optimal spreads
        let glft_bid = self.glft_optimal_spread(gamma, kappa_bid);
        let glft_ask = self.glft_optimal_spread(gamma, kappa_ask);

        // Convert to basis points
        let glft_bid_bps = Self::to_bps(glft_bid);
        let glft_ask_bps = Self::to_bps(glft_ask);

        // Cap at multiple of market spread (per-side, so divide by 2)
        let market_half_spread = market_spread_bps / 2.0;
        let cap_bps = market_half_spread * self.config.market_spread_cap_multiple;

        let capped_bid_bps = glft_bid_bps.min(cap_bps);
        let capped_ask_bps = glft_ask_bps.min(cap_bps);

        // Apply floor
        let mut optimal_bid_bps = capped_bid_bps
            .max(self.config.min_spread_floor_bps)
            .max(self.config.min_depth_bps);
        let mut optimal_ask_bps = capped_ask_bps
            .max(self.config.min_spread_floor_bps)
            .max(self.config.min_depth_bps);

        // Apply hard cap if configured (for competitive spreads on illiquid assets)
        if self.config.max_spread_per_side_bps > 0.0 {
            optimal_bid_bps = optimal_bid_bps.min(self.config.max_spread_per_side_bps);
            optimal_ask_bps = optimal_ask_bps.min(self.config.max_spread_per_side_bps);
        }

        // Generate depth arrays
        let bid_depths = self.generate_depth_array(optimal_bid_bps);
        let ask_depths = if self.config.enable_asymmetric {
            self.generate_depth_array(optimal_ask_bps)
        } else {
            bid_depths.clone()
        };

        let depths = DynamicDepths {
            bid: bid_depths,
            ask: ask_depths,
        };

        debug!(
            gamma = %format!("{:.3}", gamma),
            kappa_bid = %format!("{:.1}", kappa_bid),
            kappa_ask = %format!("{:.1}", kappa_ask),
            glft_bid_bps = %format!("{:.2}", glft_bid_bps),
            glft_ask_bps = %format!("{:.2}", glft_ask_bps),
            market_spread_bps = %format!("{:.2}", market_spread_bps),
            cap_bps = %format!("{:.2}", cap_bps),
            optimal_bid_bps = %format!("{:.2}", optimal_bid_bps),
            optimal_ask_bps = %format!("{:.2}", optimal_ask_bps),
            best_bid_depth = ?depths.best_bid_depth(),
            best_ask_depth = ?depths.best_ask_depth(),
            "Dynamic depths computed with market cap"
        );

        depths
    }

    /// Compute depths with dynamic Bayesian bounds.
    ///
    /// This method replaces hardcoded CLI arguments with principled model-driven bounds:
    /// - `dynamic_kappa_floor`: From Bayesian confidence + credible intervals
    /// - `dynamic_spread_ceiling`: From fill rate controller + market spread p80
    ///
    /// # Arguments
    /// * `gamma` - Effective risk aversion
    /// * `kappa_bid` - Bid-side order flow intensity (before floor)
    /// * `kappa_ask` - Ask-side order flow intensity (before floor)
    /// * `sigma` - Volatility
    /// * `dynamic_kappa_floor` - Optional floor for kappa values (higher = tighter spreads)
    /// * `dynamic_spread_ceiling_bps` - Optional ceiling for spreads in bps (tighter = more competitive)
    ///
    /// # Mathematical Flow
    ///
    /// 1. Apply kappa floor: `kappa_eff = kappa.max(floor)`
    ///    - Higher kappa floor → tighter GLFT spreads
    ///    - Floor derived from Bayesian CI: principled, not arbitrary
    ///
    /// 2. Compute GLFT optimal spread: `δ* = (1/γ) × ln(1 + γ/κ_eff)`
    ///
    /// 3. Apply spread ceiling: `depth = depth.min(ceiling)`
    ///    - Ceiling from fill rate controller or market p80
    ///    - Ensures competitiveness even when GLFT is conservative
    pub fn compute_depths_with_dynamic_bounds(
        &self,
        gamma: f64,
        kappa_bid: f64,
        kappa_ask: f64,
        _sigma: f64,
        dynamic_kappa_floor: Option<f64>,
        dynamic_spread_ceiling_bps: Option<f64>,
    ) -> DynamicDepths {
        // Apply dynamic kappa floor (if provided)
        let effective_kappa_bid = match dynamic_kappa_floor {
            Some(floor) => kappa_bid.max(floor),
            None => kappa_bid,
        };
        let effective_kappa_ask = match dynamic_kappa_floor {
            Some(floor) => kappa_ask.max(floor),
            None => kappa_ask,
        };

        // Compute GLFT optimal spreads with floored kappa
        let optimal_bid = self.glft_optimal_spread(gamma, effective_kappa_bid);
        let optimal_ask = self.glft_optimal_spread(gamma, effective_kappa_ask);

        // Convert to basis points
        let optimal_bid_bps = Self::to_bps(optimal_bid);
        let optimal_ask_bps = Self::to_bps(optimal_ask);

        // Apply floor (minimum spread)
        let mut optimal_bid_bps = optimal_bid_bps
            .max(self.config.min_spread_floor_bps)
            .max(self.config.min_depth_bps);
        let mut optimal_ask_bps = optimal_ask_bps
            .max(self.config.min_spread_floor_bps)
            .max(self.config.min_depth_bps);

        // Apply dynamic spread ceiling (if provided) - this is the model-driven cap
        if let Some(ceiling_bps) = dynamic_spread_ceiling_bps {
            optimal_bid_bps = optimal_bid_bps.min(ceiling_bps);
            optimal_ask_bps = optimal_ask_bps.min(ceiling_bps);
        }

        // Also apply hard cap from config if set (CLI override takes precedence)
        if self.config.max_spread_per_side_bps > 0.0 {
            optimal_bid_bps = optimal_bid_bps.min(self.config.max_spread_per_side_bps);
            optimal_ask_bps = optimal_ask_bps.min(self.config.max_spread_per_side_bps);
        }

        // Generate depth arrays
        let bid_depths = self.generate_depth_array(optimal_bid_bps);
        let ask_depths = if self.config.enable_asymmetric {
            self.generate_depth_array(optimal_ask_bps)
        } else {
            bid_depths.clone()
        };

        let depths = DynamicDepths {
            bid: bid_depths,
            ask: ask_depths,
        };

        debug!(
            gamma = %format!("{:.3}", gamma),
            kappa_bid = %format!("{:.1}", kappa_bid),
            kappa_ask = %format!("{:.1}", kappa_ask),
            kappa_floor = ?dynamic_kappa_floor.map(|f| format!("{f:.1}")),
            effective_kappa_bid = %format!("{:.1}", effective_kappa_bid),
            effective_kappa_ask = %format!("{:.1}", effective_kappa_ask),
            spread_ceiling_bps = ?dynamic_spread_ceiling_bps.map(|c| format!("{c:.2}")),
            optimal_bid_bps = %format!("{:.2}", optimal_bid_bps),
            optimal_ask_bps = %format!("{:.2}", optimal_ask_bps),
            best_bid_depth = ?depths.best_bid_depth(),
            best_ask_depth = ?depths.best_ask_depth(),
            "Dynamic depths computed with Bayesian bounds"
        );

        depths
    }

    /// Generate depth array from optimal spread
    ///
    /// When the optimal spread exceeds max_depth_bps, we create a distributed ladder
    /// starting from min_spread_floor_bps (competitive quotes) up to max_depth_bps.
    /// This ensures we always have a proper ladder with varying depths, not all
    /// levels clamped to the same value.
    fn generate_depth_array(&self, optimal_bps: f64) -> Vec<f64> {
        let n = self.config.num_levels;
        if n == 0 {
            return vec![];
        }

        // Calculate max depth for this ladder
        let max_depth =
            (optimal_bps * self.config.max_depth_multiple).min(self.config.max_depth_bps);

        // If optimal spread exceeds max_depth_bps, create a distributed ladder
        // from min_spread_floor to max_depth instead of clamping all to max.
        // This ensures we have competitive quotes at the touch AND distribution.
        let effective_start = if optimal_bps > max_depth {
            // Start from floor and expand to max
            self.config
                .min_spread_floor_bps
                .max(self.config.min_depth_bps)
        } else {
            optimal_bps
        };

        match self.config.spacing {
            DepthSpacing::Geometric => self.geometric_depths(effective_start, max_depth, n),
            DepthSpacing::Linear => self.linear_depths(effective_start, max_depth, n),
        }
    }

    /// Generate geometrically spaced depths starting from start
    ///
    /// Depths: [start, start×r, start×r², ..., start×r^(n-1)]
    /// Clamped to [min_depth_bps, max] range.
    ///
    /// When start is much smaller than max, this creates a distributed ladder.
    /// Example: start=2bp, max=50bp, ratio=1.2, n=5 → [2, 2.4, 2.88, 3.46, 4.15]
    fn geometric_depths(&self, start: f64, max: f64, n: usize) -> Vec<f64> {
        let ratio = self.config.geometric_ratio;

        (0..n)
            .map(|i| {
                let depth = start * ratio.powi(i as i32);
                depth.clamp(self.config.min_depth_bps, max)
            })
            .collect()
    }

    /// Generate linearly spaced depths starting from start
    ///
    /// Depths: [start, start+step, start+2×step, ..., start+(n-1)×step]
    /// Clamped to [min_depth_bps, max] range.
    fn linear_depths(&self, start: f64, max: f64, n: usize) -> Vec<f64> {
        let step = self.config.linear_step_bps;

        (0..n)
            .map(|i| {
                let depth = start + step * i as f64;
                depth.clamp(self.config.min_depth_bps, max)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glft_optimal_spread() {
        let generator = DynamicDepthGenerator::with_defaults();

        // With high kappa (liquid), spread should be tight
        let spread_liquid = generator.glft_optimal_spread(0.3, 500.0);
        assert!(spread_liquid < 0.01); // Less than 100bps

        // With low kappa (illiquid), spread should be wide
        let spread_illiquid = generator.glft_optimal_spread(0.3, 10.0);
        assert!(spread_illiquid > spread_liquid);

        // GLFT formula: δ = (1/γ) × ln(1 + γ/κ)
        // When γ/κ is small: ln(1+x) ≈ x, so δ ≈ 1/κ (independent of γ)
        // When γ/κ is large: ln(1+x) grows slowly, so δ ≈ (1/γ) × ln(γ/κ)
        //
        // To see gamma effect clearly, we need γ/κ > 1 (γ > κ)
        // γ=5, κ=1: δ = (1/5) × ln(1 + 5/1) = 0.2 × ln(6) = 0.358
        // γ=1, κ=1: δ = (1/1) × ln(1 + 1/1) = 1 × ln(2) = 0.693
        // Lower gamma gives HIGHER spread when γ/κ is small!
        //
        // But with γ > κ:
        // γ=5, κ=1: δ = 0.358
        // γ=2, κ=1: δ = (1/2) × ln(3) = 0.549
        // Higher gamma still gives lower spread because 1/γ dominates
        let spread_high_gamma = generator.glft_optimal_spread(5.0, 1.0);
        let spread_low_gamma = generator.glft_optimal_spread(2.0, 1.0);
        assert!(
            spread_high_gamma < spread_low_gamma,
            "high_gamma spread {} should be < low_gamma spread {} (ln dominates)",
            spread_high_gamma,
            spread_low_gamma
        );
    }

    #[test]
    fn test_glft_formula_edge_cases() {
        let generator = DynamicDepthGenerator::with_defaults();

        // Very high kappa → spread approaches maker fee
        let spread = generator.glft_optimal_spread(0.3, 10000.0);
        let min_spread = generator.config.maker_fee_rate;
        assert!(
            spread < min_spread + 0.001,
            "spread {} should be < {} + 0.001",
            spread,
            min_spread
        );

        // Very low gamma → spread tightens (but still positive due to Taylor expansion)
        let spread = generator.glft_optimal_spread(0.01, 100.0);
        assert!(
            spread < 0.02,
            "spread {} with low gamma should be < 0.02",
            spread
        ); // Less than 200bps

        // Zero kappa protection
        let spread = generator.glft_optimal_spread(0.3, 0.0);
        assert!(spread.is_finite());
        assert!(spread > 0.0);
    }

    #[test]
    fn test_compute_depths_symmetric() {
        let generator = DynamicDepthGenerator::with_defaults();

        let depths = generator.compute_symmetric_depths(0.3, 200.0, 0.0001);

        assert!(depths.is_valid());
        assert_eq!(depths.bid.len(), 5);
        assert_eq!(depths.ask.len(), 5);

        // Symmetric should have equal depths
        assert_eq!(depths.bid, depths.ask);

        // Depths should be sorted ascending
        for i in 1..depths.bid.len() {
            assert!(depths.bid[i] >= depths.bid[i - 1]);
        }
    }

    #[test]
    fn test_compute_depths_asymmetric() {
        let mut config = DynamicDepthConfig::default();
        config.enable_asymmetric = true;
        let generator = DynamicDepthGenerator::new(config);

        // Use kappa values that result in optimal spreads within max_depth_bps
        // Higher kappa = tighter optimal spread
        // With gamma=0.1, kappa=500: optimal ≈ 2bp
        // With gamma=0.1, kappa=300: optimal ≈ 3bp
        let depths = generator.compute_depths(0.1, 300.0, 500.0, 0.0001);

        assert!(depths.is_valid());

        // Higher kappa_ask → tighter ask depths
        assert!(
            depths.best_ask_depth().unwrap() < depths.best_bid_depth().unwrap(),
            "ask_depth {} should be < bid_depth {}",
            depths.best_ask_depth().unwrap(),
            depths.best_bid_depth().unwrap()
        );
    }

    #[test]
    fn test_compute_depths_fallback_when_optimal_exceeds_max() {
        // When GLFT optimal spread exceeds max_depth_bps, we should still get
        // a distributed ladder starting from min_spread_floor_bps
        let config = DynamicDepthConfig {
            num_levels: 5,
            min_depth_bps: 1.0,
            max_depth_bps: 50.0, // Low max to force fallback
            max_depth_multiple: 5.0,
            min_spread_floor_bps: 2.0, // Competitive floor
            geometric_ratio: 1.5,
            ..Default::default()
        };
        let generator = DynamicDepthGenerator::new(config);

        // gamma=0.9, kappa=137: optimal ≈ 73bp > max_depth=50bp
        // This should trigger fallback to start from min_spread_floor_bps
        let depths = generator.compute_depths(0.9, 137.0, 137.0, 0.0001);

        assert!(depths.is_valid());

        // Should start from floor (2bp), not from optimal (73bp) or max (50bp)
        let best_bid = depths.best_bid_depth().unwrap();
        assert!(
            best_bid <= 3.0,
            "best_bid {} should start near floor (2bp) when optimal exceeds max",
            best_bid
        );

        // Depths should be distributed, not all clamped to max
        let second_bid = depths.bid.get(1).copied().unwrap_or(0.0);
        assert!(
            second_bid > best_bid,
            "second_bid {} should be > best_bid {} (proper ladder distribution)",
            second_bid,
            best_bid
        );

        // Last depth should expand toward max
        let last_bid = depths.bid.last().copied().unwrap_or(0.0);
        assert!(
            last_bid > 5.0,
            "last_bid {} should expand beyond floor",
            last_bid
        );
    }

    #[test]
    fn test_geometric_spacing() {
        let mut config = DynamicDepthConfig::default();
        config.num_levels = 4;
        config.spacing = DepthSpacing::Geometric;
        config.geometric_ratio = 2.0;
        config.min_depth_bps = 1.0;
        config.max_depth_bps = 100.0;
        config.max_depth_multiple = 10.0; // Allow up to 10x starting depth
        let generator = DynamicDepthGenerator::new(config);

        let depths = generator.generate_depth_array(5.0);

        assert_eq!(depths.len(), 4);
        // With ratio 2: [5, 10, 20, 40]
        assert!(
            (depths[0] - 5.0).abs() < 0.01,
            "depths[0] = {} should be 5.0",
            depths[0]
        );
        assert!(
            (depths[1] - 10.0).abs() < 0.01,
            "depths[1] = {} should be 10.0",
            depths[1]
        );
        assert!(
            (depths[2] - 20.0).abs() < 0.01,
            "depths[2] = {} should be 20.0",
            depths[2]
        );
        assert!(
            (depths[3] - 40.0).abs() < 0.01,
            "depths[3] = {} should be 40.0",
            depths[3]
        );
    }

    #[test]
    fn test_linear_spacing() {
        let mut config = DynamicDepthConfig::default();
        config.num_levels = 4;
        config.spacing = DepthSpacing::Linear;
        config.linear_step_bps = 5.0;
        config.min_depth_bps = 1.0;
        config.max_depth_bps = 100.0;
        let generator = DynamicDepthGenerator::new(config);

        let depths = generator.generate_depth_array(5.0);

        assert_eq!(depths.len(), 4);
        // With step 5: [5, 10, 15, 20]
        assert!((depths[0] - 5.0).abs() < 0.01);
        assert!((depths[1] - 10.0).abs() < 0.01);
        assert!((depths[2] - 15.0).abs() < 0.01);
        assert!((depths[3] - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_depth_clamping() {
        let mut config = DynamicDepthConfig::default();
        config.num_levels = 3;
        config.min_depth_bps = 2.0;
        config.max_depth_bps = 10.0;
        config.spacing = DepthSpacing::Geometric;
        config.geometric_ratio = 3.0;
        let generator = DynamicDepthGenerator::new(config);

        // Start at 1bp, should be clamped to 2bp min
        let depths = generator.generate_depth_array(1.0);
        assert!(depths[0] >= 2.0);

        // With ratio 3: [2, 6, 10] (third clamped at max)
        assert!(depths[2] <= 10.0);
    }

    #[test]
    fn test_spread_floor() {
        let mut config = DynamicDepthConfig::default();
        config.min_spread_floor_bps = 5.0;
        let generator = DynamicDepthGenerator::new(config);

        // Even with very high kappa, spread should respect floor
        let depths = generator.compute_depths(0.1, 10000.0, 10000.0, 0.0001);

        assert!(depths.best_bid_depth().unwrap() >= 5.0);
        assert!(depths.best_ask_depth().unwrap() >= 5.0);
    }

    #[test]
    fn test_dynamic_depths_methods() {
        let depths = DynamicDepths {
            bid: vec![5.0, 10.0, 20.0],
            ask: vec![4.0, 8.0, 16.0],
        };

        assert!(depths.is_valid());
        assert_eq!(depths.best_bid_depth(), Some(5.0));
        assert_eq!(depths.best_ask_depth(), Some(4.0));
        assert_eq!(depths.spread_at_touch(), Some(9.0)); // 5 + 4
    }

    #[test]
    fn test_empty_depths() {
        let depths = DynamicDepths::default();
        assert!(!depths.is_valid());
        assert_eq!(depths.best_bid_depth(), None);
        assert_eq!(depths.spread_at_touch(), None);
    }

    #[test]
    fn test_symmetric_constructor() {
        let depths = DynamicDepths::symmetric(vec![5.0, 10.0, 15.0]);
        assert_eq!(depths.bid, depths.ask);
        assert!(depths.is_valid());
    }

    #[test]
    fn test_realistic_btc_scenario() {
        // Simulate realistic BTC market making parameters
        let config = DynamicDepthConfig {
            num_levels: 5,
            min_depth_bps: 1.0,
            max_depth_multiple: 5.0,
            max_depth_bps: 50.0,
            spacing: DepthSpacing::Geometric,
            geometric_ratio: 1.5,
            linear_step_bps: 2.0,
            maker_fee_rate: 0.0001, // 1bp
            min_spread_floor_bps: 2.0,
            enable_asymmetric: true,
            market_spread_cap_multiple: 5.0,
            max_spread_per_side_bps: 0.0, // Disabled for this test
        };
        let generator = DynamicDepthGenerator::new(config);

        // Typical liquid BTC: gamma=0.1, kappa=300
        // GLFT spread = (1/0.1) × ln(1 + 0.1/300) ≈ 33bps + fee ≈ 34bps
        let depths = generator.compute_depths(0.1, 300.0, 300.0, 0.0001);

        // Should produce spreads based on GLFT formula
        let best_bid = depths.best_bid_depth().unwrap();
        assert!(best_bid >= 2.0, "best_bid {} should be >= 2.0", best_bid); // At least 2bp (floor)
        assert!(
            best_bid < 50.0,
            "best_bid {} should be < 50.0 (max depth)",
            best_bid
        ); // Should be within max

        // All depths should be increasing
        for i in 1..depths.bid.len() {
            assert!(depths.bid[i] >= depths.bid[i - 1]);
        }
    }
}
