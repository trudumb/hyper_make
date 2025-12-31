//! Ladder quoting engine for multi-level quote generation.
//!
//! Implements multi-level quote ladders with:
//! - **Dynamic depth generation** from GLFT optimal spread theory
//! - Geometric or linear depth spacing
//! - Fill intensity modeling: λ(δ) = σ²/δ²
//! - Spread capture: SC(δ) = δ - AS₀ × exp(-δ/δ_char) - fees
//! - Size allocation proportional to marginal value: λ(δ) × SC(δ)
//! - GLFT inventory skew applied to entire ladder
//! - **Constrained variational optimization** for capital-efficient allocation
//! - **Kelly-Stochastic optimization** using first-passage fill probability
//!
//! # Module Structure
//!
//! - `depth_generator`: Dynamic depth computation from market parameters
//! - `generator`: Ladder generation logic and depth/size calculation
//! - `optimizer`: Constrained optimization with multiple allocation strategies

mod depth_generator;
mod fill_probability;
mod generator;
mod optimizer;

pub use depth_generator::{
    DepthSpacing, DynamicDepthConfig, DynamicDepthGenerator, DynamicDepths,
};
pub use fill_probability::{BayesianFillModel, DepthBucket, FirstPassageFillModel};
pub use optimizer::{
    BindingConstraint, ConstrainedAllocation, ConstrainedLadderOptimizer, KellyStochasticParams,
    LevelOptimizationParams,
};

use serde::{Deserialize, Serialize};

use crate::market_maker::adverse_selection::DepthDecayAS;

/// A single quote level in the ladder
#[derive(Debug, Clone, Copy)]
pub struct LadderLevel {
    /// Price of this level
    pub price: f64,
    /// Size at this level
    pub size: f64,
    /// Distance from mid in basis points
    pub depth_bps: f64,
}

/// Multi-level quote ladder
#[derive(Debug, Clone, Default)]
pub struct Ladder {
    /// Bid levels, sorted best-to-worst (highest price first)
    pub bids: Vec<LadderLevel>,
    /// Ask levels, sorted best-to-worst (lowest price first)
    pub asks: Vec<LadderLevel>,
}

/// Configuration for ladder generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LadderConfig {
    /// Number of levels per side (default: 5)
    pub num_levels: usize,
    /// Minimum depth from mid in basis points (default: 2 bps)
    /// Used as fallback when dynamic_depths is None
    pub min_depth_bps: f64,
    /// Maximum depth from mid in basis points (default: 50 bps)
    /// Used as fallback when dynamic_depths is None
    pub max_depth_bps: f64,
    /// Use geometric spacing (true) or linear spacing (false)
    /// Used as fallback when dynamic_depths is None
    pub geometric_spacing: bool,
    /// Minimum size per level (orders below this are skipped)
    pub min_level_size: f64,
    /// Trading fees in basis points (maker + taker / 2)
    pub fees_bps: f64,
    /// Adverse selection decay characteristic depth in bps
    pub as_decay_bps: f64,
    /// Optional dynamic depths computed from market parameters.
    /// When Some, these depths override min/max_depth_bps and geometric_spacing.
    /// This enables GLFT-optimal depth selection that adapts to γ, κ, and market regime.
    #[serde(skip)]
    pub dynamic_depths: Option<DynamicDepths>,
}

impl Default for LadderConfig {
    fn default() -> Self {
        Self {
            num_levels: 5,
            min_depth_bps: 2.0,
            max_depth_bps: 200.0,
            geometric_spacing: true,
            min_level_size: 0.001,
            // Hyperliquid fees: maker ~1-2bp, taker ~3.5bp
            // Round-trip = maker + taker ≈ 4.5-5.5bp
            // We use 3.5bp as spread capture fee (half of round-trip)
            // to account for adverse fill (we get maker, they get taker)
            fees_bps: 3.5,
            as_decay_bps: 10.0,
            dynamic_depths: None,
        }
    }
}

impl LadderConfig {
    /// Create a config with dynamic depths.
    ///
    /// When dynamic depths are set, they override the static min/max_depth_bps
    /// and geometric_spacing settings.
    pub fn with_dynamic_depths(mut self, depths: DynamicDepths) -> Self {
        self.dynamic_depths = Some(depths);
        self
    }

    /// Check if dynamic depths are enabled
    pub fn has_dynamic_depths(&self) -> bool {
        self.dynamic_depths.is_some()
    }

    /// Get effective number of bid levels (from dynamic or static config)
    pub fn effective_bid_levels(&self) -> usize {
        self.dynamic_depths
            .as_ref()
            .map(|d| d.bid.len())
            .unwrap_or(self.num_levels)
    }

    /// Get effective number of ask levels (from dynamic or static config)
    pub fn effective_ask_levels(&self) -> usize {
        self.dynamic_depths
            .as_ref()
            .map(|d| d.ask.len())
            .unwrap_or(self.num_levels)
    }
}

/// Parameters for ladder generation derived from market state
#[derive(Debug, Clone)]
pub struct LadderParams {
    /// Mid/fair price to quote around
    pub mid_price: f64,
    /// Volatility (per-second)
    pub sigma: f64,
    /// Order flow intensity (κ)
    pub kappa: f64,
    /// Trade arrival intensity (volume ticks per second)
    pub arrival_intensity: f64,
    /// Adverse selection at touch in basis points (legacy, used if depth_decay_as is None)
    pub as_at_touch_bps: f64,
    /// Total size budget per side
    pub total_size: f64,
    /// Inventory ratio q/Q_max in [-1, 1]
    pub inventory_ratio: f64,
    /// Risk aversion parameter (γ)
    pub gamma: f64,
    /// Time horizon T = 1/λ (seconds)
    pub time_horizon: f64,
    /// Price decimals for rounding
    pub decimals: u32,
    /// Size decimals for truncation
    pub sz_decimals: u32,
    /// Minimum order notional value
    pub min_notional: f64,
    /// Optional depth-dependent AS model (calibrated from fills)
    /// If Some, uses first-principles exponential decay: AS(δ) = AS₀ × exp(-δ/δ_char)
    /// If None, falls back to legacy config-based decay
    pub depth_decay_as: Option<DepthDecayAS>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LadderConfig::default();
        assert_eq!(config.num_levels, 5);
        assert!((config.min_depth_bps - 2.0).abs() < 0.01);
        assert!((config.max_depth_bps - 200.0).abs() < 0.01);
        assert!(config.geometric_spacing);
    }

    #[test]
    fn test_ladder_generate_full() {
        let config = LadderConfig {
            num_levels: 3,
            min_depth_bps: 5.0,
            max_depth_bps: 20.0,
            geometric_spacing: false,
            min_level_size: 0.01,
            fees_bps: 0.5,
            as_decay_bps: 10.0,
            dynamic_depths: None,
        };

        let params = LadderParams {
            mid_price: 100.0,
            sigma: 0.001,
            kappa: 100.0,
            arrival_intensity: 1.0,
            as_at_touch_bps: 2.0, // Low AS so all levels are profitable
            total_size: 1.0,
            inventory_ratio: 0.0,
            gamma: 0.3,
            time_horizon: 10.0,
            decimals: 2,
            sz_decimals: 4,
            min_notional: 10.0,
            depth_decay_as: None, // Use legacy AS config
        };

        let ladder = Ladder::generate(&config, &params);

        // Should have quotes on both sides
        assert!(!ladder.is_empty());
        assert!(ladder.num_bids() > 0);
        assert!(ladder.num_asks() > 0);

        // Total sizes should be close to total_size (accounting for min_notional filtering)
        let total_bid = ladder.total_bid_size();
        let total_ask = ladder.total_ask_size();
        assert!(total_bid > 0.0);
        assert!(total_ask > 0.0);

        // Best bid < mid < best ask
        assert!(ladder.best_bid().unwrap() < params.mid_price);
        assert!(ladder.best_ask().unwrap() > params.mid_price);
    }

    #[test]
    fn test_ladder_generate_with_inventory() {
        let config = LadderConfig::default();

        let params_neutral = LadderParams {
            mid_price: 1000.0,
            sigma: 0.001,
            kappa: 100.0,
            arrival_intensity: 1.0,
            as_at_touch_bps: 1.0,
            total_size: 1.0,
            inventory_ratio: 0.0,
            gamma: 0.5,
            time_horizon: 10.0,
            decimals: 2,
            sz_decimals: 4,
            min_notional: 10.0,
            depth_decay_as: None,
        };

        let params_long = LadderParams {
            inventory_ratio: 0.5,
            ..params_neutral.clone()
        };

        let ladder_neutral = Ladder::generate(&config, &params_neutral);
        let ladder_long = Ladder::generate(&config, &params_long);

        // Long inventory should have smaller bid sizes
        assert!(ladder_long.total_bid_size() < ladder_neutral.total_bid_size());
        // Ask sizes should be similar
        assert!((ladder_long.total_ask_size() - ladder_neutral.total_ask_size()).abs() < 0.5);
    }

    #[test]
    fn test_ladder_with_depth_decay_as() {
        use crate::market_maker::adverse_selection::DepthDecayAS;

        let config = LadderConfig {
            num_levels: 5,
            min_depth_bps: 2.0,
            max_depth_bps: 20.0,
            geometric_spacing: true,
            min_level_size: 0.01,
            fees_bps: 0.5,
            as_decay_bps: 10.0, // Ignored when depth_decay_as is Some
            dynamic_depths: None,
        };

        // Calibrated AS model with higher AS at touch
        let depth_decay = DepthDecayAS::new(5.0, 8.0); // AS₀=5bp, δ_char=8bp

        let params = LadderParams {
            mid_price: 100.0,
            sigma: 0.001,
            kappa: 100.0,
            arrival_intensity: 1.0,
            as_at_touch_bps: 2.0, // Ignored when depth_decay_as is Some
            total_size: 1.0,
            inventory_ratio: 0.0,
            gamma: 0.3,
            time_horizon: 10.0,
            decimals: 2,
            sz_decimals: 4,
            min_notional: 10.0,
            depth_decay_as: Some(depth_decay),
        };

        let ladder = Ladder::generate(&config, &params);

        // Should have quotes on both sides
        assert!(!ladder.is_empty());

        // With high AS at touch (5bp), shallow levels should have less size allocated
        // because their spread capture is lower (or negative)
        // Deep levels should get more size because AS decays

        // Verify total size is close to target
        let total_bid = ladder.total_bid_size();
        let total_ask = ladder.total_ask_size();
        assert!(total_bid > 0.0);
        assert!(total_ask > 0.0);
    }
}
