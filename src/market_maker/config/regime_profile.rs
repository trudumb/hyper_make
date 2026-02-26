//! Regime-specific configuration profiles for thin DEX vs liquid CEX environments.
//!
//! These profiles provide pre-tuned configurations for different market structures:
//!
//! - **ThinDex**: Optimized for HIP-3 style thin DEX environments
//!   - High changepoint threshold (0.85) for noise tolerance
//!   - Low kappa prior (200) matching thin fill rates
//!   - Wide spread targets (25-40 bps) for profitability
//!   - Aggressive depth gating for microprice
//!   - Preemptive position management
//!
//! - **LiquidCex**: Optimized for liquid centralized exchange environments
//!   - Standard changepoint threshold (0.5)
//!   - High kappa prior (1500) for fast fill rates
//!   - Tight spread targets (5-15 bps)
//!   - Minimal depth gating
//!
//! # Usage
//!
//! ```ignore
//! let profile = RegimeProfile::thin_dex();
//! config.stochastic = profile.stochastic_overrides(&config.stochastic);
//! estimator.set_depth_gate(profile.microprice_min_depth_usd, profile.microprice_full_depth_usd);
//! ```

use crate::market_maker::control::MarketRegime;

/// Regime-specific configuration profile.
///
/// Aggregates all parameters that differ between thin DEX and liquid CEX environments.
#[derive(Debug, Clone)]
pub struct RegimeProfile {
    /// Profile name for logging
    pub name: &'static str,

    /// The market regime this profile is designed for
    pub regime: MarketRegime,

    // ==================== Changepoint Detection ====================
    /// Changepoint probability threshold (higher = more tolerant of noise)
    /// ThinDex: 0.65, LiquidCex: 0.5, Cascade: 0.3
    pub changepoint_threshold: f64,

    /// Number of consecutive high-prob signals required to confirm changepoint
    /// ThinDex: 1, LiquidCex: 1, Cascade: 1
    pub changepoint_confirmation_count: usize,

    /// Prior hazard rate (expected frequency of regime switches)
    pub cp_hazard_rate: f64,

    /// Threshold for soft boosting probability
    pub cp_soft_boost_threshold: f64,

    /// Threshold to force extreme regime
    pub cp_force_extreme_threshold: f64,

    /// Multiplier for trend echo estimation
    pub trend_r_base_multiplier: f64,

    // ==================== Estimator Parameters ====================
    /// Kappa prior for fill intensity estimation (fills per hour)
    /// ThinDex: 200, LiquidCex: 1500
    pub kappa_prior: f64,

    /// Microprice minimum depth threshold (USD)
    /// Below this, microprice falls back to mid
    /// ThinDex: 1000, LiquidCex: 10000
    pub microprice_min_depth_usd: f64,

    /// Microprice full weight depth threshold (USD)
    /// Above this, microprice uses full imbalance weight
    /// ThinDex: 10000, LiquidCex: 100000
    pub microprice_full_depth_usd: f64,

    // ==================== Position Management ====================
    /// Position warning threshold (fraction of max)
    /// Above this, start inventory skew
    /// ThinDex: 0.5, LiquidCex: 0.7
    pub position_warning_threshold: f64,

    /// Position pull threshold (fraction of max)
    /// Above this, pull quotes on the filling side
    /// ThinDex: 0.7, LiquidCex: 0.9
    pub position_pull_threshold: f64,

    /// Inventory skew gamma (risk aversion for skew)
    /// Higher = more aggressive position reduction
    /// ThinDex: 0.5, LiquidCex: 0.3
    pub inventory_skew_gamma: f64,

    // ==================== Spread Targets ====================
    /// Target minimum spread (bps)
    /// ThinDex: 25, LiquidCex: 5
    pub min_spread_bps: f64,

    /// Target comfortable spread (bps)
    /// ThinDex: 40, LiquidCex: 10
    pub target_spread_bps: f64,

    /// Maximum spread before widening concern (bps)
    /// ThinDex: 80, LiquidCex: 25
    pub max_spread_bps: f64,

    // ==================== Quote Lifecycle ====================
    /// Minimum quote lifetime before cancellation (ms)
    /// ThinDex: 120000 (2 min), LiquidCex: 5000 (5 sec)
    pub min_quote_lifetime_ms: u64,

    /// Sync health tolerance (0-1)
    /// Below this, enter degraded quoting mode
    /// ThinDex: 0.7, LiquidCex: 0.9
    pub sync_health_tolerance: f64,

    // ==================== Risk Adjustments ====================
    /// Risk aversion base multiplier
    /// ThinDex: 1.5 (more conservative), LiquidCex: 1.0
    pub risk_aversion_multiplier: f64,

    /// Spread widening factor during degraded sync
    /// ThinDex: 2.0, LiquidCex: 1.5
    pub degraded_spread_factor: f64,

    // ==================== Quoting ====================
    /// Quote both sides when flat without edge
    /// ThinDex: true (spread capture), LiquidCex: false (API conservation)
    pub quote_flat_without_edge: bool,
}

impl RegimeProfile {
    /// Create a profile optimized for thin DEX environments (HIP-3).
    ///
    /// Key characteristics:
    /// - High noise tolerance (changepoint threshold 0.85)
    /// - Low fill rate assumption (kappa prior 200)
    /// - Aggressive depth gating (fall back to mid on thin books)
    /// - Wide spreads for profitability (25-40 bps)
    /// - Long minimum quote lifetime (2 minutes)
    pub fn thin_dex() -> Self {
        Self {
            name: "thin_dex",
            regime: MarketRegime::ThinDex,

            // Changepoint: very tolerant of noise
            changepoint_threshold: 0.65,
            changepoint_confirmation_count: 1,
            cp_hazard_rate: 0.03, // (1/250) / (200/1500)
            cp_soft_boost_threshold: 0.35,
            cp_force_extreme_threshold: 0.65,
            trend_r_base_multiplier: 5.0,

            // Estimator: conservative assumptions
            kappa_prior: 200.0, // Low fill rate
            microprice_min_depth_usd: 1_000.0,
            microprice_full_depth_usd: 10_000.0,

            // Position: preemptive management
            position_warning_threshold: 0.5,
            position_pull_threshold: 0.7,
            inventory_skew_gamma: 0.5,

            // Spreads: wide for profitability
            min_spread_bps: 25.0,
            target_spread_bps: 40.0,
            max_spread_bps: 80.0,

            // Quote lifecycle: long-lived quotes
            min_quote_lifetime_ms: 120_000, // 2 minutes
            sync_health_tolerance: 0.7,

            // Risk: conservative
            risk_aversion_multiplier: 1.5,
            degraded_spread_factor: 2.0,

            // Quoting: spread capture focus
            quote_flat_without_edge: true,
        }
    }

    /// Create a profile optimized for liquid centralized exchange environments.
    ///
    /// Key characteristics:
    /// - Standard changepoint sensitivity (0.5)
    /// - High fill rate assumption (kappa prior 1500)
    /// - Minimal depth gating (deep books are reliable)
    /// - Tight spreads (5-15 bps)
    /// - Short minimum quote lifetime (5 seconds)
    pub fn liquid_cex() -> Self {
        Self {
            name: "liquid_cex",
            regime: MarketRegime::LiquidCex,

            // Changepoint: standard sensitivity
            changepoint_threshold: 0.5,
            changepoint_confirmation_count: 1,
            cp_hazard_rate: 0.004, // 1/250
            cp_soft_boost_threshold: 0.50,
            cp_force_extreme_threshold: 0.90,
            trend_r_base_multiplier: 1.0,

            // Estimator: liquid market assumptions
            kappa_prior: 1500.0, // High fill rate
            microprice_min_depth_usd: 10_000.0,
            microprice_full_depth_usd: 100_000.0,

            // Position: less aggressive
            position_warning_threshold: 0.7,
            position_pull_threshold: 0.9,
            inventory_skew_gamma: 0.3,

            // Spreads: tight for competitiveness
            min_spread_bps: 5.0,
            target_spread_bps: 10.0,
            max_spread_bps: 25.0,

            // Quote lifecycle: short-lived quotes
            min_quote_lifetime_ms: 5_000, // 5 seconds
            sync_health_tolerance: 0.9,

            // Risk: standard
            risk_aversion_multiplier: 1.0,
            degraded_spread_factor: 1.5,

            // Quoting: API conservation
            quote_flat_without_edge: false,
        }
    }

    /// Create a cascade-defensive profile.
    ///
    /// Key characteristics:
    /// - Very sensitive changepoint (0.3)
    /// - Maximum spread widening
    /// - Aggressive position reduction
    /// - Reduce-only mode preferred
    pub fn cascade() -> Self {
        Self {
            name: "cascade",
            regime: MarketRegime::Cascade,

            // Changepoint: very sensitive
            changepoint_threshold: 0.3,
            changepoint_confirmation_count: 1,
            cp_hazard_rate: 0.04, // (1/250) / max(0.1, 50/1500)
            cp_soft_boost_threshold: 0.20,
            cp_force_extreme_threshold: 0.40,
            trend_r_base_multiplier: 5.0, // cascade also trusts itself less

            // Estimator: assume chaos
            kappa_prior: 50.0, // Very low - expect no fills
            microprice_min_depth_usd: 5_000.0,
            microprice_full_depth_usd: 50_000.0,

            // Position: aggressive reduction
            position_warning_threshold: 0.3,
            position_pull_threshold: 0.5,
            inventory_skew_gamma: 1.0, // Maximum skew

            // Spreads: very wide
            min_spread_bps: 50.0,
            target_spread_bps: 100.0,
            max_spread_bps: 200.0,

            // Quote lifecycle: minimal quoting
            min_quote_lifetime_ms: 30_000, // 30 seconds
            sync_health_tolerance: 0.5,

            // Risk: maximum
            risk_aversion_multiplier: 3.0,
            degraded_spread_factor: 3.0,

            // Quoting: reduce-only preferred
            quote_flat_without_edge: false,
        }
    }

    /// Detect appropriate regime from market characteristics.
    ///
    /// Uses simple heuristics based on:
    /// - Average fill rate
    /// - Book depth
    /// - Volatility
    pub fn detect(avg_fills_per_hour: f64, avg_book_depth_usd: f64, volatility_bps: f64) -> Self {
        // Cascade detection: high volatility + thin book
        if volatility_bps > 100.0 && avg_book_depth_usd < 20_000.0 {
            return Self::cascade();
        }

        // Thin DEX detection: low fills + thin book
        if avg_fills_per_hour < 10.0 && avg_book_depth_usd < 50_000.0 {
            return Self::thin_dex();
        }

        // Default to liquid CEX
        Self::liquid_cex()
    }

    /// Check if this is a thin DEX profile.
    pub fn is_thin_dex(&self) -> bool {
        matches!(self.regime, MarketRegime::ThinDex)
    }

    /// Check if this is a cascade profile.
    pub fn is_cascade(&self) -> bool {
        matches!(self.regime, MarketRegime::Cascade)
    }

    /// Get the spread widening multiplier for sync health level.
    pub fn spread_multiplier_for_sync_health(&self, health_score: f64) -> f64 {
        if health_score >= self.sync_health_tolerance {
            1.0
        } else if health_score >= self.sync_health_tolerance * 0.5 {
            self.degraded_spread_factor
        } else {
            self.degraded_spread_factor * 1.5
        }
    }
}

impl Default for RegimeProfile {
    fn default() -> Self {
        // Default to thin DEX for safety
        Self::thin_dex()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thin_dex_profile() {
        let profile = RegimeProfile::thin_dex();

        assert_eq!(profile.name, "thin_dex");
        assert!(matches!(profile.regime, MarketRegime::ThinDex));
        assert_eq!(profile.changepoint_threshold, 0.65);
        assert_eq!(profile.kappa_prior, 200.0);
        assert_eq!(profile.min_spread_bps, 25.0);
        assert!(profile.quote_flat_without_edge);
    }

    #[test]
    fn test_liquid_cex_profile() {
        let profile = RegimeProfile::liquid_cex();

        assert_eq!(profile.name, "liquid_cex");
        assert!(matches!(profile.regime, MarketRegime::LiquidCex));
        assert_eq!(profile.changepoint_threshold, 0.5);
        assert_eq!(profile.kappa_prior, 1500.0);
        assert_eq!(profile.min_spread_bps, 5.0);
        assert!(!profile.quote_flat_without_edge);
    }

    #[test]
    fn test_cascade_profile() {
        let profile = RegimeProfile::cascade();

        assert_eq!(profile.name, "cascade");
        assert!(matches!(profile.regime, MarketRegime::Cascade));
        assert_eq!(profile.changepoint_threshold, 0.3);
        assert_eq!(profile.kappa_prior, 50.0);
        assert_eq!(profile.risk_aversion_multiplier, 3.0);
    }

    #[test]
    fn test_regime_detection() {
        // Thin DEX: low fills, thin book
        let thin = RegimeProfile::detect(5.0, 20_000.0, 50.0);
        assert!(thin.is_thin_dex());

        // Liquid CEX: high fills, deep book
        let liquid = RegimeProfile::detect(100.0, 500_000.0, 30.0);
        assert!(!liquid.is_thin_dex());
        assert!(!liquid.is_cascade());

        // Cascade: high vol, thin book
        let cascade = RegimeProfile::detect(5.0, 10_000.0, 150.0);
        assert!(cascade.is_cascade());
    }

    #[test]
    fn test_spread_multiplier_for_health() {
        let profile = RegimeProfile::thin_dex();

        // Healthy
        assert_eq!(profile.spread_multiplier_for_sync_health(0.9), 1.0);

        // Degraded
        assert_eq!(profile.spread_multiplier_for_sync_health(0.5), 2.0);

        // Critical
        assert_eq!(profile.spread_multiplier_for_sync_health(0.2), 3.0);
    }
}
