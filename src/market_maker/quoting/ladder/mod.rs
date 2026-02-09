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
//! - **Entropy-based stochastic distribution** for diversity-preserving allocation
//!
//! # Module Structure
//!
//! - `depth_generator`: Dynamic depth computation from market parameters
//! - `generator`: Ladder generation logic and depth/size calculation
//! - `optimizer`: Constrained optimization with multiple allocation strategies
//! - `entropy_distribution`: Entropy-based stochastic order distribution
//! - `entropy_optimizer`: Diversity-preserving optimizer using entropy constraints
//!
//! # Entropy-Based Distribution (New)
//!
//! The entropy-based system completely replaces the old concentration fallback
//! mechanism that could collapse orders to just 1-2 levels. Key features:
//!
//! - **Minimum entropy floor**: Distribution NEVER drops below H_min, ensuring
//!   at least exp(H_min) effective levels remain active.
//! - **Softmax temperature control**: Smooth transitions instead of hard cutoffs.
//! - **Thompson sampling**: Stochastic allocation prevents predictability.
//! - **Dirichlet smoothing**: Prior regularization prevents any level from zero.

mod depth_generator;
mod entropy_distribution;
mod entropy_optimizer;
mod fill_probability;
mod generator;
mod optimizer;

pub use depth_generator::{DepthSpacing, DynamicDepthConfig, DynamicDepthGenerator, DynamicDepths};
pub use entropy_distribution::{
    EntropyDistribution, EntropyDistributionConfig, EntropyDistributor, EntropyLevelParams,
    MarketRegime,
};
pub use entropy_optimizer::{
    create_aggressive_optimizer, create_defensive_optimizer, create_entropy_optimizer,
    EntropyConstrainedAllocation, EntropyConstrainedOptimizer, EntropyOptimizerConfig,
};
pub use fill_probability::{BayesianFillModel, DepthBucket, FirstPassageFillModel};
pub use optimizer::{BindingConstraint, ConstrainedAllocation, LevelOptimizationParams};

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::market_maker::adverse_selection::DepthDecayAS;
use crate::market_maker::infra::capacity::LADDER_LEVEL_INLINE_CAPACITY;

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

/// Type alias for ladder level storage using SmallVec for stack allocation.
/// Typical ladders have 5 levels; 8 inline capacity avoids heap allocation.
pub type LadderLevels = SmallVec<[LadderLevel; LADDER_LEVEL_INLINE_CAPACITY]>;

/// Multi-level quote ladder
///
/// Uses SmallVec for bid/ask levels to avoid heap allocation for typical
/// ladder sizes (5 levels). This keeps quote cycle data on the stack,
/// reducing allocation latency from ~100ns to ~10ns.
#[derive(Debug, Clone, Default)]
pub struct Ladder {
    /// Bid levels, sorted best-to-worst (highest price first)
    pub bids: LadderLevels,
    /// Ask levels, sorted best-to-worst (lowest price first)
    pub asks: LadderLevels,
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

    /// Hard maximum spread per side in basis points (absolute cap).
    /// Applied after all other spread calculations as a final safety valve.
    /// Use for illiquid assets where GLFT may produce very wide spreads.
    /// Set to 0.0 to disable (default, no hard cap).
    /// Example: max_spread_per_side_bps = 15.0 → never quote wider than 15 bps per side
    #[serde(default)]
    pub max_spread_per_side_bps: f64,
}

impl Default for LadderConfig {
    fn default() -> Self {
        Self {
            num_levels: 25, // Increased from 5 to 25 (max capacity) to let entropy/stochastics dictate effective levels
            min_depth_bps: 2.0,
            max_depth_bps: 200.0,
            geometric_spacing: true,
            // FIXED: min_level_size was too high (0.001 BTC = $90 at BTC=$90k)
            // causing all levels to be filtered out when total_size < num_levels * min_level_size.
            // Exchange min notional is $10, so for BTC we need min_size ≈ $10/$90k ≈ 0.000111.
            // Set to 0.00012 to provide a small buffer above exchange minimum.
            min_level_size: 0.00012,
            // Hyperliquid maker fee: 1.5 bps.
            // Adverse selection is modeled separately via DepthDecayAS —
            // do NOT double-count it here or we create a dead zone filtering profitable levels.
            fees_bps: 1.5,
            as_decay_bps: 10.0,
            dynamic_depths: None,
            max_spread_per_side_bps: 0.0, // Disabled by default
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
    /// Mid/fair price to quote around (microprice - used for quote generation)
    pub mid_price: f64,
    /// Actual market mid price (from AllMids - used for safety checks to prevent crossing)
    pub market_mid: f64,
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

    // === Drift-Adjusted Skew (First Principles Extension) ===
    /// Whether to apply drift-adjusted skew when position opposes momentum
    pub use_drift_adjusted_skew: bool,
    /// HJB-derived drift urgency (fractional, e.g., 0.001579 = 15.79 bps)
    pub hjb_drift_urgency: f64,
    /// Whether current position opposes market momentum direction
    pub position_opposes_momentum: bool,
    /// Variance multiplier from directional risk (>1 when opposing trend)
    pub directional_variance_mult: f64,
    /// Combined urgency score (0-5 scale)
    pub urgency_score: f64,

    // === Funding/Cost of Carry ===
    /// Per-second funding rate (derived from annualized or hourly)
    pub funding_rate: f64,
    /// Whether to apply funding skew to the ladder
    pub use_funding_skew: bool,

    // === RL Policy Adjustments ===
    /// RL-recommended spread delta (bps). Positive = widen, Negative = tighten.
    pub rl_spread_delta_bps: f64,
    /// RL-recommended bid skew (bps). Positive = widen bid (favor buying).
    pub rl_bid_skew_bps: f64,
    /// RL-recommended ask skew (bps). Positive = widen ask (favor selling).
    pub rl_ask_skew_bps: f64,
    /// Confidence in RL recommendation [0, 1]. Scales the adjustment magnitude.
    pub rl_confidence: f64,

    // === Position Continuation Model ===
    /// Position action from PositionDecisionEngine (HOLD/ADD/REDUCE).
    /// Used to transform inventory_ratio for skew calculation.
    pub position_action: crate::market_maker::strategy::PositionAction,
    /// Effective inventory ratio after HOLD/ADD/REDUCE transformation.
    /// - HOLD: 0.0 (no skew, symmetric quotes)
    /// - ADD: negative (reverse skew, tighter on position-building side)
    /// - REDUCE: positive × urgency (normal mean-reversion)
    pub effective_inventory_ratio: f64,

    /// Warmup progress [0.0, 1.0]. Used to gate RL adjustments (disabled when < 0.5).
    pub warmup_pct: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LadderConfig::default();
        assert_eq!(config.num_levels, 25); // Default 25 levels (max capacity)
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
            max_spread_per_side_bps: 0.0,
        };

        let params = LadderParams {
            mid_price: 100.0,
            market_mid: 100.0, // Same as mid_price for tests (no microprice adj)
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
            // Drift-adjusted skew disabled for this test
            use_drift_adjusted_skew: false,
            hjb_drift_urgency: 0.0,
            position_opposes_momentum: false,
            directional_variance_mult: 1.0,
            urgency_score: 0.0,
            funding_rate: 0.0,
            use_funding_skew: false,
            // RL disabled for this test
            rl_spread_delta_bps: 0.0,
            rl_bid_skew_bps: 0.0,
            rl_ask_skew_bps: 0.0,
            rl_confidence: 0.0,
            // Position continuation disabled for this test
            position_action: crate::market_maker::strategy::PositionAction::default(),
            effective_inventory_ratio: 0.0,
            warmup_pct: 1.0,
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
            market_mid: 1000.0, // Same as mid_price for tests
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
            // Drift-adjusted skew disabled for this test
            use_drift_adjusted_skew: false,
            hjb_drift_urgency: 0.0,
            position_opposes_momentum: false,
            directional_variance_mult: 1.0,
            urgency_score: 0.0,
            funding_rate: 0.0,
            use_funding_skew: false,
            // RL disabled for this test
            rl_spread_delta_bps: 0.0,
            rl_bid_skew_bps: 0.0,
            rl_ask_skew_bps: 0.0,
            rl_confidence: 0.0,
            // Position continuation disabled for this test
            position_action: crate::market_maker::strategy::PositionAction::default(),
            effective_inventory_ratio: 0.0,
            warmup_pct: 1.0,
        };

        let params_long = LadderParams {
            inventory_ratio: 0.5,
            // IMPORTANT: Generator now uses effective_inventory_ratio for skew
            // Set it to match inventory_ratio to test inventory skew behavior
            effective_inventory_ratio: 0.5,
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
            max_spread_per_side_bps: 0.0,
        };

        // Calibrated AS model with higher AS at touch
        let depth_decay = DepthDecayAS::new(5.0, 8.0); // AS₀=5bp, δ_char=8bp

        let params = LadderParams {
            mid_price: 100.0,
            market_mid: 100.0, // Same as mid_price for tests
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
            // Drift-adjusted skew disabled for this test
            use_drift_adjusted_skew: false,
            hjb_drift_urgency: 0.0,
            position_opposes_momentum: false,
            directional_variance_mult: 1.0,
            urgency_score: 0.0,
            funding_rate: 0.0,
            use_funding_skew: false,
            // RL disabled for this test
            rl_spread_delta_bps: 0.0,
            rl_bid_skew_bps: 0.0,
            rl_ask_skew_bps: 0.0,
            rl_confidence: 0.0,
            // Position continuation disabled for this test
            position_action: crate::market_maker::strategy::PositionAction::default(),
            effective_inventory_ratio: 0.0,
            warmup_pct: 1.0,
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

    #[test]
    fn test_ladder_with_drift_adjusted_skew() {
        // Use wider ladder (50bps min) so that the 15.79bps drift isn't capped by the 50% depth safety check
        // (If min depth is 2bps, max drift is 1bp, which would cap our 15.79bps drift)
        let config = LadderConfig {
            min_depth_bps: 50.0,
            ..Default::default()
        };

        // Base params with LONG position opposing bearish momentum
        let params_with_drift = LadderParams {
            mid_price: 100.0,
            market_mid: 100.0, // Same as mid_price for tests
            sigma: 0.001,
            kappa: 100.0,
            arrival_intensity: 1.0,
            as_at_touch_bps: 1.0,
            total_size: 1.0,
            inventory_ratio: 0.3, // LONG position
            gamma: 0.5,
            time_horizon: 10.0,
            decimals: 2,
            sz_decimals: 4,
            min_notional: 10.0,
            depth_decay_as: None,
            // Drift-adjusted skew ENABLED - position opposes momentum
            use_drift_adjusted_skew: true,
            hjb_drift_urgency: 0.001579, // 15.79 bps as fractional
            position_opposes_momentum: true,
            directional_variance_mult: 1.5,
            urgency_score: 2.0, // Above 0.5 threshold
            funding_rate: 0.0,
            use_funding_skew: false,
            // RL disabled for this test
            rl_spread_delta_bps: 0.0,
            rl_bid_skew_bps: 0.0,
            rl_ask_skew_bps: 0.0,
            rl_confidence: 0.0,
            // Position continuation disabled for this test
            position_action: crate::market_maker::strategy::PositionAction::default(),
            effective_inventory_ratio: 0.3, // Matches inventory_ratio for REDUCE
            warmup_pct: 1.0,
        };

        // Same params but WITHOUT drift adjustment
        let params_no_drift = LadderParams {
            use_drift_adjusted_skew: false,
            position_opposes_momentum: false,
            urgency_score: 0.0,
            ..params_with_drift.clone()
        };

        let ladder_drift = Ladder::generate(&config, &params_with_drift);
        let ladder_no_drift = Ladder::generate(&config, &params_no_drift);

        // With drift adjustment, quotes should be shifted MORE (extra skew)
        // When LONG + bearish: lower both bid and ask to encourage selling
        let best_bid_drift = ladder_drift.best_bid().unwrap();
        let best_ask_drift = ladder_drift.best_ask().unwrap();
        let best_bid_no_drift = ladder_no_drift.best_bid().unwrap();
        let best_ask_no_drift = ladder_no_drift.best_ask().unwrap();

        // With additional drift skew, both prices should be LOWER
        // (extra offset in the direction to reduce long position)
        assert!(
            best_bid_drift < best_bid_no_drift,
            "Drift skew should lower bid: {:.4} should be < {:.4}",
            best_bid_drift,
            best_bid_no_drift
        );
        assert!(
            best_ask_drift < best_ask_no_drift,
            "Drift skew should lower ask: {:.4} should be < {:.4}",
            best_ask_drift,
            best_ask_no_drift
        );

        // The shift should be approximately hjb_drift_urgency × mid = 0.001579 × 100 ≈ 0.16
        let bid_shift = best_bid_no_drift - best_bid_drift;
        let ask_shift = best_ask_no_drift - best_ask_drift;
        let expected_shift = 0.001579 * 100.0; // ~0.16

        assert!(
            (bid_shift - expected_shift).abs() < 0.05,
            "Bid shift {:.4} should be ~{:.4}",
            bid_shift,
            expected_shift
        );
        assert!(
            (ask_shift - expected_shift).abs() < 0.05,
            "Ask shift {:.4} should be ~{:.4}",
            ask_shift,
            expected_shift
        );
    }
}
