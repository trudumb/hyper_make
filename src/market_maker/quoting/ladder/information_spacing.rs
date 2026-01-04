//! Information-theoretic depth spacing for optimal level placement.
//!
//! This module determines WHERE to place orders (depth selection) using
//! information theory. The goal is to place levels at depths that:
//! - Maximize information captured about market state
//! - Minimize redundancy between adjacent levels
//! - Adapt to changing market microstructure
//!
//! # Key Concepts
//!
//! ## Information Density
//! I(δ) = information content at depth δ, measuring how much a fill
//! at that depth tells us about subsequent price movement.
//!
//! ## Optimal Spacing
//! Levels are placed to maximize total information: max Σ I(δ_i)
//! subject to minimum spacing constraint to avoid order overlap.
//!
//! ## Adaptive Placement
//! - Low information depths are skipped
//! - High information depths get multiple levels nearby
//! - Spacing adapts to volatility and order flow

use smallvec::SmallVec;

use crate::market_maker::infra::capacity::DEPTH_INLINE_CAPACITY;
use crate::EPSILON;

/// Type alias for depth vectors.
type DepthVec = SmallVec<[f64; DEPTH_INLINE_CAPACITY]>;

/// Configuration for information-theoretic depth spacing.
#[derive(Debug, Clone)]
pub struct InformationSpacingConfig {
    /// Number of levels to place per side.
    pub num_levels: usize,

    /// Minimum depth from mid in bps.
    pub min_depth_bps: f64,

    /// Maximum depth from mid in bps.
    pub max_depth_bps: f64,

    /// Minimum spacing between adjacent levels in bps.
    /// Prevents order overlap and reduces correlation.
    pub min_spacing_bps: f64,

    /// Information decay rate.
    /// Higher values mean information drops off faster with depth.
    pub info_decay_rate: f64,

    /// Volatility scaling factor.
    /// Depths scale with σ√τ for vol-normalized spacing.
    pub vol_scaling: bool,

    /// Weight for spread capture in placement decision.
    /// Combined objective: w × I(δ) + (1-w) × SC(δ)
    pub profit_weight: f64,
}

impl Default for InformationSpacingConfig {
    fn default() -> Self {
        Self {
            num_levels: 5,
            min_depth_bps: 2.0,
            max_depth_bps: 100.0,
            min_spacing_bps: 1.0,
            info_decay_rate: 0.1, // 10% decay per bps
            vol_scaling: true,
            profit_weight: 0.5, // Balance info and profit
        }
    }
}

/// Information density curve at various depths.
///
/// Represents I(δ) - the information content at each depth level.
/// Calibrated from historical fill data and price movements.
#[derive(Debug, Clone)]
pub struct InformationDensityCurve {
    /// Depth points in bps.
    depths: Vec<f64>,
    /// Information values at each depth (bits or nats).
    information: Vec<f64>,
    /// Interpolation spline coefficients for smooth curve.
    spline_coeffs: Option<Vec<(f64, f64, f64, f64)>>, // (a, b, c, d) for cubic
}

impl InformationDensityCurve {
    /// Create from observed depth-information pairs.
    pub fn from_observations(depths: Vec<f64>, information: Vec<f64>) -> Self {
        let spline = if depths.len() >= 4 {
            Some(Self::compute_cubic_spline(&depths, &information))
        } else {
            None
        };

        Self {
            depths,
            information,
            spline_coeffs: spline,
        }
    }

    /// Create default curve using theoretical model.
    ///
    /// Uses exponential decay: I(δ) = I_0 × exp(-λδ)
    /// where I_0 is information at touch and λ is decay rate.
    pub fn default_exponential(decay_rate: f64) -> Self {
        let depths: Vec<f64> = (0..50).map(|i| 2.0 + i as f64 * 2.0).collect();
        let information: Vec<f64> = depths
            .iter()
            .map(|&d| (-decay_rate * d).exp())
            .collect();

        Self::from_observations(depths, information)
    }

    /// Create curve based on adverse selection model.
    ///
    /// Information peaks at intermediate depths where:
    /// - Too tight: dominated by noise, low information
    /// - Optimal: mix of informed and noise, high information
    /// - Too deep: rarely filled, low information
    pub fn adverse_selection_based(as_touch: f64, as_decay: f64, fill_decay: f64) -> Self {
        let depths: Vec<f64> = (1..100).map(|i| i as f64).collect();

        let information: Vec<f64> = depths
            .iter()
            .map(|&d| {
                // Fill probability decays with depth
                let fill_prob = (-d / fill_decay).exp();
                // Informed probability decays with depth
                let alpha = as_touch * (-d / as_decay).exp();
                // Information = fill_prob × (1 - alpha) × some_base
                // Peaks where fill_prob × (1-alpha) is maximized
                let info = fill_prob * (1.0 - alpha) * (1.0 + d / 20.0).ln();
                info.max(0.0)
            })
            .collect();

        Self::from_observations(depths, information)
    }

    /// Get information at a specific depth (interpolated).
    pub fn information_at(&self, depth_bps: f64) -> f64 {
        if self.depths.is_empty() {
            return 0.0;
        }

        // Use spline interpolation if available
        if let Some(ref coeffs) = self.spline_coeffs {
            return self.spline_interpolate(depth_bps, coeffs);
        }

        // Fall back to linear interpolation
        self.linear_interpolate(depth_bps)
    }

    /// Linear interpolation between observed points.
    fn linear_interpolate(&self, depth: f64) -> f64 {
        if depth <= self.depths[0] {
            return self.information[0];
        }
        if depth >= *self.depths.last().unwrap() {
            return *self.information.last().unwrap();
        }

        for i in 0..self.depths.len() - 1 {
            if depth >= self.depths[i] && depth < self.depths[i + 1] {
                let t = (depth - self.depths[i]) / (self.depths[i + 1] - self.depths[i]);
                return self.information[i] * (1.0 - t) + self.information[i + 1] * t;
            }
        }

        0.0
    }

    /// Cubic spline interpolation for smooth curve.
    fn spline_interpolate(&self, depth: f64, coeffs: &[(f64, f64, f64, f64)]) -> f64 {
        if depth <= self.depths[0] {
            return self.information[0];
        }
        if depth >= *self.depths.last().unwrap() {
            return *self.information.last().unwrap();
        }

        for (i, (a, b, c, d)) in coeffs.iter().enumerate() {
            if depth >= self.depths[i] && depth < self.depths[i + 1] {
                let x = depth - self.depths[i];
                return a + b * x + c * x * x + d * x * x * x;
            }
        }

        0.0
    }

    /// Compute cubic spline coefficients.
    fn compute_cubic_spline(x: &[f64], y: &[f64]) -> Vec<(f64, f64, f64, f64)> {
        let n = x.len();
        if n < 2 {
            return vec![];
        }

        // Simple natural cubic spline
        // For production, consider using a spline library
        let mut coeffs = Vec::with_capacity(n - 1);

        for i in 0..n - 1 {
            let h = x[i + 1] - x[i];
            if h < EPSILON {
                coeffs.push((y[i], 0.0, 0.0, 0.0));
                continue;
            }

            // Simple linear approximation as fallback
            let slope = (y[i + 1] - y[i]) / h;
            coeffs.push((y[i], slope, 0.0, 0.0));
        }

        coeffs
    }

    /// Find depth with maximum information.
    pub fn peak_depth(&self) -> f64 {
        if self.depths.is_empty() {
            return 10.0; // Default
        }

        let max_idx = self
            .information
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.depths[max_idx]
    }

    /// Get total information in a depth range.
    pub fn total_information(&self, min_depth: f64, max_depth: f64) -> f64 {
        // Numerical integration using trapezoidal rule
        let steps = 50;
        let step_size = (max_depth - min_depth) / steps as f64;

        let mut total = 0.0;
        for i in 0..steps {
            let d1 = min_depth + i as f64 * step_size;
            let d2 = d1 + step_size;
            total += 0.5 * (self.information_at(d1) + self.information_at(d2)) * step_size;
        }

        total
    }
}

/// Generator for information-optimal depth levels.
#[derive(Debug, Clone)]
pub struct InformationDepthGenerator {
    config: InformationSpacingConfig,
    info_curve: InformationDensityCurve,
}

impl InformationDepthGenerator {
    /// Create generator with configuration and information curve.
    pub fn new(config: InformationSpacingConfig, info_curve: InformationDensityCurve) -> Self {
        Self { config, info_curve }
    }

    /// Create generator with default exponential information decay.
    pub fn with_default_curve(config: InformationSpacingConfig) -> Self {
        let curve = InformationDensityCurve::default_exponential(config.info_decay_rate);
        Self::new(config, curve)
    }

    /// Create generator with adverse-selection based information curve.
    pub fn with_as_curve(
        config: InformationSpacingConfig,
        as_touch: f64,
        as_decay: f64,
        fill_decay: f64,
    ) -> Self {
        let curve = InformationDensityCurve::adverse_selection_based(as_touch, as_decay, fill_decay);
        Self::new(config, curve)
    }

    /// Generate optimal depths that maximize total information.
    ///
    /// Uses greedy algorithm:
    /// 1. Start with highest-information depth
    /// 2. Add depths that maximize marginal information gain
    /// 3. Respect minimum spacing constraint
    pub fn generate_depths(&self) -> DepthVec {
        let mut depths = DepthVec::new();

        if self.config.num_levels == 0 {
            return depths;
        }

        // Find all candidate depths
        let candidates = self.generate_candidates();

        if candidates.is_empty() {
            // Fall back to geometric spacing
            return self.fallback_geometric();
        }

        // Greedy selection
        let mut remaining: Vec<(f64, f64)> = candidates.clone(); // (depth, info)

        while depths.len() < self.config.num_levels && !remaining.is_empty() {
            // Find candidate with highest information that respects spacing
            let best_idx = remaining
                .iter()
                .enumerate()
                .filter(|(_, (d, _))| self.respects_spacing(*d, &depths))
                .max_by(|(_, (_, i1)), (_, (_, i2))| i1.partial_cmp(i2).unwrap())
                .map(|(i, _)| i);

            match best_idx {
                Some(idx) => {
                    depths.push(remaining[idx].0);
                    remaining.remove(idx);
                }
                None => break, // No valid candidates left
            }
        }

        // Sort depths for ladder construction
        depths.sort_by(|a, b| a.partial_cmp(b).unwrap());
        depths
    }

    /// Generate depths with volatility scaling.
    ///
    /// Depths are scaled by σ√τ to maintain consistent fill probability
    /// across different volatility regimes.
    pub fn generate_depths_scaled(&self, sigma: f64, time_horizon: f64) -> DepthVec {
        if !self.config.vol_scaling {
            return self.generate_depths();
        }

        let expected_move_bps = sigma * time_horizon.sqrt() * 10000.0;
        let scale_factor = (expected_move_bps / 3.0).clamp(0.5, 3.0); // Normalize to ~3bp move

        let base_depths = self.generate_depths();
        base_depths
            .iter()
            .map(|&d| (d * scale_factor).clamp(self.config.min_depth_bps, self.config.max_depth_bps))
            .collect()
    }

    /// Generate candidate depths with their information values.
    fn generate_candidates(&self) -> Vec<(f64, f64)> {
        // Sample depths at fine granularity
        let step = 0.5; // 0.5 bps steps
        let mut candidates = Vec::new();

        let mut d = self.config.min_depth_bps;
        while d <= self.config.max_depth_bps {
            let info = self.info_curve.information_at(d);
            if info > EPSILON {
                candidates.push((d, info));
            }
            d += step;
        }

        candidates
    }

    /// Check if a depth respects minimum spacing from existing depths.
    fn respects_spacing(&self, depth: f64, existing: &[f64]) -> bool {
        for &d in existing {
            if (depth - d).abs() < self.config.min_spacing_bps {
                return false;
            }
        }
        true
    }

    /// Fallback to geometric spacing when information curve is empty.
    fn fallback_geometric(&self) -> DepthVec {
        let n = self.config.num_levels;
        if n == 0 {
            return DepthVec::new();
        }
        if n == 1 {
            let mut depths = DepthVec::new();
            depths.push(self.config.min_depth_bps);
            return depths;
        }

        let ratio = (self.config.max_depth_bps / self.config.min_depth_bps).powf(1.0 / (n - 1) as f64);

        (0..n)
            .map(|i| self.config.min_depth_bps * ratio.powi(i as i32))
            .collect()
    }

    /// Update information curve from new observations.
    pub fn update_curve(&mut self, curve: InformationDensityCurve) {
        self.info_curve = curve;
    }
}

/// Multi-scale depth spacing for different market regimes.
///
/// Uses different spacing strategies based on detected market state:
/// - Trending: Concentrate levels in direction of trend
/// - Mean-reverting: Spread levels symmetrically
/// - High-volatility: Use wider spacing
/// - Low-volatility: Use tighter spacing
#[derive(Debug, Clone)]
pub struct AdaptiveDepthSpacing {
    /// Base configuration.
    config: InformationSpacingConfig,
    /// Current market regime weights.
    regime_weights: RegimeWeights,
    /// Information curve for each regime.
    regime_curves: RegimeCurves,
}

/// Weights for different market regimes.
#[derive(Debug, Clone, Default)]
pub struct RegimeWeights {
    pub trending_up: f64,
    pub trending_down: f64,
    pub mean_reverting: f64,
    pub high_volatility: f64,
    pub low_volatility: f64,
}

impl RegimeWeights {
    /// Create uniform weights.
    pub fn uniform() -> Self {
        Self {
            trending_up: 0.2,
            trending_down: 0.2,
            mean_reverting: 0.2,
            high_volatility: 0.2,
            low_volatility: 0.2,
        }
    }

    /// Normalize weights to sum to 1.
    pub fn normalize(&mut self) {
        let total = self.trending_up
            + self.trending_down
            + self.mean_reverting
            + self.high_volatility
            + self.low_volatility;

        if total > EPSILON {
            self.trending_up /= total;
            self.trending_down /= total;
            self.mean_reverting /= total;
            self.high_volatility /= total;
            self.low_volatility /= total;
        }
    }
}

/// Information curves for different regimes.
#[derive(Debug, Clone)]
pub struct RegimeCurves {
    pub trending: InformationDensityCurve,
    pub mean_reverting: InformationDensityCurve,
    pub high_vol: InformationDensityCurve,
    pub low_vol: InformationDensityCurve,
}

impl Default for RegimeCurves {
    fn default() -> Self {
        Self {
            // Trending: information peaks at intermediate depths
            trending: InformationDensityCurve::adverse_selection_based(0.2, 15.0, 30.0),
            // Mean-reverting: information at tight depths
            mean_reverting: InformationDensityCurve::default_exponential(0.15),
            // High vol: information at deep depths
            high_vol: InformationDensityCurve::adverse_selection_based(0.3, 25.0, 50.0),
            // Low vol: information at tight depths
            low_vol: InformationDensityCurve::default_exponential(0.05),
        }
    }
}

impl AdaptiveDepthSpacing {
    /// Create with configuration.
    pub fn new(config: InformationSpacingConfig) -> Self {
        Self {
            config,
            regime_weights: RegimeWeights::uniform(),
            regime_curves: RegimeCurves::default(),
        }
    }

    /// Update regime weights from market state.
    pub fn update_regime(&mut self, weights: RegimeWeights) {
        self.regime_weights = weights;
        self.regime_weights.normalize();
    }

    /// Generate depths using weighted combination of regime curves.
    pub fn generate_adaptive_depths(&self) -> DepthVec {
        // Compute blended information curve
        let depths: Vec<f64> = (0..100).map(|i| 2.0 + i as f64).collect();

        let blended_info: Vec<f64> = depths
            .iter()
            .map(|&d| {
                let trend_info = (self.regime_weights.trending_up + self.regime_weights.trending_down)
                    * self.regime_curves.trending.information_at(d);
                let mr_info =
                    self.regime_weights.mean_reverting * self.regime_curves.mean_reverting.information_at(d);
                let hv_info =
                    self.regime_weights.high_volatility * self.regime_curves.high_vol.information_at(d);
                let lv_info =
                    self.regime_weights.low_volatility * self.regime_curves.low_vol.information_at(d);

                trend_info + mr_info + hv_info + lv_info
            })
            .collect();

        let blended_curve = InformationDensityCurve::from_observations(depths, blended_info);
        let generator = InformationDepthGenerator::new(self.config.clone(), blended_curve);
        generator.generate_depths()
    }

    /// Get asymmetric depths for bid/ask based on trend direction.
    pub fn generate_asymmetric_depths(&self) -> (DepthVec, DepthVec) {
        let base = self.generate_adaptive_depths();

        // If trending up: bids at tight depths (want to buy), asks at wider depths
        // If trending down: asks at tight depths (want to sell), bids at wider depths
        let trend_bias = self.regime_weights.trending_up - self.regime_weights.trending_down;

        if trend_bias.abs() < 0.1 {
            // No significant trend, use symmetric
            return (base.clone(), base);
        }

        let tight_scale = 0.8;
        let wide_scale = 1.2;

        let (bid_scale, ask_scale) = if trend_bias > 0.0 {
            (tight_scale, wide_scale) // Trending up: tight bids
        } else {
            (wide_scale, tight_scale) // Trending down: tight asks
        };

        let bids: DepthVec = base
            .iter()
            .map(|&d| (d * bid_scale).clamp(self.config.min_depth_bps, self.config.max_depth_bps))
            .collect();

        let asks: DepthVec = base
            .iter()
            .map(|&d| (d * ask_scale).clamp(self.config.min_depth_bps, self.config.max_depth_bps))
            .collect();

        (bids, asks)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_information_curve_interpolation() {
        let depths = vec![2.0, 5.0, 10.0, 20.0];
        let info = vec![1.0, 0.8, 0.5, 0.2];
        let curve = InformationDensityCurve::from_observations(depths, info);

        // Test exact points
        assert!((curve.information_at(2.0) - 1.0).abs() < 0.01);
        assert!((curve.information_at(10.0) - 0.5).abs() < 0.01);

        // Test interpolation
        let mid = curve.information_at(7.5);
        assert!(mid > 0.5 && mid < 0.8);

        // Test extrapolation
        assert!((curve.information_at(1.0) - 1.0).abs() < 0.01); // Below min
        assert!((curve.information_at(30.0) - 0.2).abs() < 0.01); // Above max
    }

    #[test]
    fn test_exponential_curve() {
        let curve = InformationDensityCurve::default_exponential(0.1);

        // Should decay exponentially
        let i_2 = curve.information_at(2.0);
        let i_10 = curve.information_at(10.0);
        let i_20 = curve.information_at(20.0);

        assert!(i_2 > i_10);
        assert!(i_10 > i_20);

        // Ratio should be consistent with decay rate
        let expected_ratio = (-0.1_f64 * 8.0).exp(); // 8bp difference
        let actual_ratio = i_10 / i_2;
        assert!((actual_ratio - expected_ratio).abs() < 0.1);
    }

    #[test]
    fn test_depth_generation() {
        let config = InformationSpacingConfig {
            num_levels: 5,
            min_depth_bps: 2.0,
            max_depth_bps: 50.0,
            min_spacing_bps: 2.0,
            ..Default::default()
        };

        let generator = InformationDepthGenerator::with_default_curve(config);
        let depths = generator.generate_depths();

        assert_eq!(depths.len(), 5);

        // Depths should be sorted
        for i in 1..depths.len() {
            assert!(depths[i] > depths[i - 1]);
        }

        // Depths should respect bounds
        assert!(depths[0] >= 2.0);
        assert!(*depths.last().unwrap() <= 50.0);

        // Depths should respect spacing
        for i in 1..depths.len() {
            assert!(depths[i] - depths[i - 1] >= 2.0 - EPSILON);
        }
    }

    #[test]
    fn test_depth_generation_vol_scaled() {
        let config = InformationSpacingConfig {
            num_levels: 3,
            min_depth_bps: 2.0,
            max_depth_bps: 100.0,
            vol_scaling: true,
            ..Default::default()
        };

        let generator = InformationDepthGenerator::with_default_curve(config);

        // High volatility should give wider depths
        let depths_high_vol = generator.generate_depths_scaled(0.001, 10.0);
        // Low volatility should give tighter depths
        let depths_low_vol = generator.generate_depths_scaled(0.0001, 10.0);

        let avg_high: f64 = depths_high_vol.iter().sum::<f64>() / depths_high_vol.len() as f64;
        let avg_low: f64 = depths_low_vol.iter().sum::<f64>() / depths_low_vol.len() as f64;

        assert!(avg_high > avg_low);
    }

    #[test]
    fn test_adaptive_spacing() {
        let config = InformationSpacingConfig::default();
        let mut adaptive = AdaptiveDepthSpacing::new(config);

        // Trending up regime
        adaptive.update_regime(RegimeWeights {
            trending_up: 0.8,
            trending_down: 0.0,
            mean_reverting: 0.1,
            high_volatility: 0.05,
            low_volatility: 0.05,
        });

        let (bids, asks) = adaptive.generate_asymmetric_depths();

        // Bids should be tighter than asks in uptrend
        let avg_bid: f64 = bids.iter().sum::<f64>() / bids.len() as f64;
        let avg_ask: f64 = asks.iter().sum::<f64>() / asks.len() as f64;

        assert!(avg_bid < avg_ask);
    }

    #[test]
    fn test_as_based_curve() {
        let curve = InformationDensityCurve::adverse_selection_based(0.2, 10.0, 30.0);

        // Should have a peak at intermediate depth
        let peak = curve.peak_depth();
        assert!(peak > 5.0 && peak < 50.0);

        // Information should be low at extremes
        let info_touch = curve.information_at(1.0);
        let info_deep = curve.information_at(100.0);
        let info_peak = curve.information_at(peak);

        assert!(info_peak > info_touch);
        assert!(info_peak > info_deep);
    }

    #[test]
    fn test_total_information() {
        let curve = InformationDensityCurve::default_exponential(0.1);

        let total_narrow = curve.total_information(2.0, 10.0);
        let total_wide = curve.total_information(2.0, 50.0);

        // Wider range should have more total information
        assert!(total_wide > total_narrow);
    }

    #[test]
    fn test_fallback_geometric() {
        let config = InformationSpacingConfig {
            num_levels: 5,
            min_depth_bps: 2.0,
            max_depth_bps: 32.0,
            ..Default::default()
        };

        // Empty info curve triggers fallback
        let empty_curve = InformationDensityCurve::from_observations(vec![], vec![]);
        let generator = InformationDepthGenerator::new(config, empty_curve);
        let depths = generator.generate_depths();

        assert_eq!(depths.len(), 5);
        // Should be geometric: 2, 4, 8, 16, 32
        assert!((depths[0] - 2.0).abs() < 0.1);
        assert!((depths[4] - 32.0).abs() < 0.1);
    }

    #[test]
    fn test_regime_weights_normalize() {
        let mut weights = RegimeWeights {
            trending_up: 2.0,
            trending_down: 1.0,
            mean_reverting: 1.0,
            high_volatility: 1.0,
            low_volatility: 0.0,
        };

        weights.normalize();

        let total = weights.trending_up
            + weights.trending_down
            + weights.mean_reverting
            + weights.high_volatility
            + weights.low_volatility;

        assert!((total - 1.0).abs() < EPSILON);
    }
}
