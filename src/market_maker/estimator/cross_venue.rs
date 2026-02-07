//! Cross-Venue Analyzer - Joint analysis of Binance and Hyperliquid flow signals.
//!
//! This module computes cross-venue features from the joint observation of both
//! exchanges. The key insight is that neither venue is definitively the leader -
//! the signal comes from their **joint relationship**:
//!
//! - **Agreement**: Both venues show same directional pressure → high confidence
//! - **Divergence**: Venues disagree → uncertainty, market dislocation
//! - **Intensity ratio**: Where is price discovery happening?
//! - **Cross-correlation**: Regime-dependent relationship strength
//!
//! # Mathematical Framework
//!
//! ```text
//! θ(t) = [direction, toxicity, regime]  (unobserved true state)
//!
//! Observations:
//!   X_B(t) = h_B(θ(t)) + ε_B   (Binance flow features)
//!   X_H(t) = h_H(θ(t)) + ε_H   (Hyperliquid flow features)
//!
//! Inference:
//!   P(θ(t) | X_B(1:t), X_H(1:t))  →  Bayesian filtering
//! ```
//!
//! # Signal Interpretation
//!
//! | Cross-Venue State | Interpretation | Action |
//! |-------------------|----------------|--------|
//! | Both buying, high agreement | Strong bullish | Lean long, aggressive bids |
//! | Both selling, high agreement | Strong bearish | Lean short, aggressive asks |
//! | Binance buying, HL selling | Dislocation | Widen spreads, reduce size |
//! | High intensity on Binance | Discovery there | Weight Binance more |
//! | Both VPIN > 0.6 | Informed active | Widen significantly |

use super::binance_flow::FlowFeatureVec;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for cross-venue analyzer.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CrossVenueConfig {
    /// Window for rolling correlation (number of samples).
    /// Default: 100
    pub correlation_window: usize,

    /// EMA alpha for intensity ratio tracking.
    /// Default: 0.1
    pub intensity_ema_alpha: f64,

    /// Agreement threshold: |sign(imbal_B) == sign(imbal_H)|
    /// Both must be above this to count as "agreeing" on direction.
    /// Default: 0.1
    pub agreement_threshold: f64,

    /// Divergence threshold: |imbal_B - imbal_H| above this = divergence.
    /// Default: 0.3
    pub divergence_threshold: f64,

    /// Toxicity alert threshold: max(vpin_B, vpin_H) above this = toxic.
    /// Default: 0.6
    pub toxicity_threshold: f64,

    /// Minimum samples before valid analysis.
    /// Default: 20
    pub min_samples: usize,
}

impl Default for CrossVenueConfig {
    fn default() -> Self {
        Self {
            correlation_window: 100,
            intensity_ema_alpha: 0.1,
            agreement_threshold: 0.1,
            divergence_threshold: 0.3,
            toxicity_threshold: 0.6,
            min_samples: 20,
        }
    }
}

/// Cross-venue features computed from joint analysis.
#[derive(Debug, Clone, Copy, Default)]
pub struct CrossVenueFeatures {
    /// Agreement score [-1, 1].
    /// +1 = perfect agreement (both same direction).
    /// -1 = perfect disagreement (opposite directions).
    /// 0 = one or both neutral.
    pub agreement: f64,

    /// Divergence score [0, 1].
    /// High divergence = market dislocation, uncertainty.
    pub divergence: f64,

    /// Intensity ratio [0, 1].
    /// 0 = all activity on HL, 1 = all activity on Binance.
    /// 0.5 = balanced activity.
    pub intensity_ratio: f64,

    /// Rolling correlation of imbalances [-1, 1].
    /// High correlation = venues moving together.
    /// Low/negative = venues decoupled.
    pub imbalance_correlation: f64,

    /// Maximum toxicity across venues [0, 1].
    pub max_toxicity: f64,

    /// Average toxicity across venues [0, 1].
    pub avg_toxicity: f64,

    /// Combined directional signal [-1, 1].
    /// Weighted average of imbalances, higher weight to higher-intensity venue.
    pub combined_direction: f64,

    /// Confidence in cross-venue signal [0, 1].
    /// Based on agreement, data sufficiency, and correlation.
    pub confidence: f64,

    /// Whether toxicity alert is triggered.
    pub toxicity_alert: bool,

    /// Whether divergence alert is triggered.
    pub divergence_alert: bool,

    /// Sample count for this analysis.
    pub sample_count: usize,
}

impl CrossVenueFeatures {
    /// Check if we should be defensive (widen spreads).
    pub fn should_be_defensive(&self) -> bool {
        self.toxicity_alert || self.divergence_alert || self.confidence < 0.3
    }

    /// Get spread multiplier based on cross-venue state.
    /// Returns >= 1.0 (1.0 = no widening).
    pub fn spread_multiplier(&self) -> f64 {
        let mut mult = 1.0;

        // Widen on high divergence
        if self.divergence > 0.3 {
            mult *= 1.0 + (self.divergence - 0.3) * 0.5;
        }

        // Widen on toxicity
        if self.max_toxicity > 0.5 {
            mult *= 1.0 + (self.max_toxicity - 0.5) * 0.8;
        }

        // Widen on low confidence
        if self.confidence < 0.5 {
            mult *= 1.0 + (0.5 - self.confidence) * 0.3;
        }

        mult.clamp(1.0, 2.0)
    }

    /// Get skew recommendation based on cross-venue direction.
    /// Returns (direction, magnitude_bps).
    pub fn skew_recommendation(&self) -> (i8, f64) {
        // Only recommend skew if high confidence and agreement
        if self.confidence < 0.5 || self.agreement.abs() < 0.3 {
            return (0, 0.0);
        }

        let direction = if self.combined_direction > 0.1 {
            1 // Bullish - skew asks wider
        } else if self.combined_direction < -0.1 {
            -1 // Bearish - skew bids wider
        } else {
            0 // Neutral
        };

        // Magnitude scales with direction strength and confidence
        let magnitude = self.combined_direction.abs() * self.confidence * 5.0; // Max 5 bps

        (direction, magnitude.min(5.0))
    }
}

/// Bivariate flow observation - joint snapshot of both venues.
#[derive(Debug, Clone, Default)]
pub struct BivariateFlowObservation {
    /// Binance flow features.
    pub binance: FlowFeatureVec,

    /// Hyperliquid flow features.
    pub hl: FlowFeatureVec,

    /// Cross-venue features computed from joint analysis.
    pub cross: CrossVenueFeatures,

    /// Timestamp of observation (ms).
    pub timestamp_ms: i64,
}

/// Rolling correlation tracker.
#[derive(Debug)]
struct RollingCorrelation {
    x_values: VecDeque<f64>,
    y_values: VecDeque<f64>,
    max_samples: usize,
}

impl RollingCorrelation {
    fn new(max_samples: usize) -> Self {
        Self {
            x_values: VecDeque::with_capacity(max_samples),
            y_values: VecDeque::with_capacity(max_samples),
            max_samples,
        }
    }

    fn add(&mut self, x: f64, y: f64) {
        self.x_values.push_back(x);
        self.y_values.push_back(y);

        while self.x_values.len() > self.max_samples {
            self.x_values.pop_front();
            self.y_values.pop_front();
        }
    }

    fn correlation(&self) -> f64 {
        let n = self.x_values.len();
        if n < 5 {
            return 0.0;
        }

        let n_f64 = n as f64;
        let mean_x: f64 = self.x_values.iter().sum::<f64>() / n_f64;
        let mean_y: f64 = self.y_values.iter().sum::<f64>() / n_f64;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for (x, y) in self.x_values.iter().zip(self.y_values.iter()) {
            let dx = x - mean_x;
            let dy = y - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let denom = (var_x * var_y).sqrt();
        if denom < 1e-12 {
            0.0
        } else {
            (cov / denom).clamp(-1.0, 1.0)
        }
    }

}

/// Cross-venue analyzer - computes joint features from both venues.
#[derive(Debug)]
pub struct CrossVenueAnalyzer {
    config: CrossVenueConfig,

    /// Rolling correlation of imbalances.
    imbalance_correlation: RollingCorrelation,

    /// EWMA of Binance intensity.
    binance_intensity_ema: f64,

    /// EWMA of HL intensity.
    hl_intensity_ema: f64,

    /// Agreement streak (positive = agreeing, negative = disagreeing).
    agreement_streak: i32,

    /// Sample count.
    sample_count: usize,

    /// Last computed features.
    last_features: CrossVenueFeatures,
}

impl CrossVenueAnalyzer {
    /// Create a new cross-venue analyzer.
    pub fn new(config: CrossVenueConfig) -> Self {
        Self {
            imbalance_correlation: RollingCorrelation::new(config.correlation_window),
            binance_intensity_ema: 0.0,
            hl_intensity_ema: 0.0,
            agreement_streak: 0,
            sample_count: 0,
            last_features: CrossVenueFeatures::default(),
            config,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(CrossVenueConfig::default())
    }

    /// Update with new observations from both venues.
    pub fn update(&mut self, binance: &FlowFeatureVec, hl: &FlowFeatureVec) {
        // Update rolling correlation of imbalances (use 5s window)
        self.imbalance_correlation.add(binance.imbalance_5s, hl.imbalance_5s);

        // Update intensity EMAs
        let alpha = self.config.intensity_ema_alpha;
        self.binance_intensity_ema = alpha * binance.intensity + (1.0 - alpha) * self.binance_intensity_ema;
        self.hl_intensity_ema = alpha * hl.intensity + (1.0 - alpha) * self.hl_intensity_ema;

        // Update agreement streak
        let binance_sign = if binance.imbalance_5s > self.config.agreement_threshold {
            1
        } else if binance.imbalance_5s < -self.config.agreement_threshold {
            -1
        } else {
            0
        };

        let hl_sign = if hl.imbalance_5s > self.config.agreement_threshold {
            1
        } else if hl.imbalance_5s < -self.config.agreement_threshold {
            -1
        } else {
            0
        };

        if binance_sign != 0 && hl_sign != 0 && binance_sign == hl_sign {
            // Agreeing
            self.agreement_streak = self.agreement_streak.saturating_add(1).min(100);
        } else if binance_sign != 0 && hl_sign != 0 && binance_sign != hl_sign {
            // Disagreeing
            self.agreement_streak = self.agreement_streak.saturating_sub(1).max(-100);
        } else {
            // Decaying toward zero
            self.agreement_streak = (self.agreement_streak as f64 * 0.95) as i32;
        }

        self.sample_count += 1;

        // Compute features
        self.last_features = self.compute_features(binance, hl);
    }

    /// Compute cross-venue features from current state.
    fn compute_features(&self, binance: &FlowFeatureVec, hl: &FlowFeatureVec) -> CrossVenueFeatures {
        // Agreement: normalized streak
        let agreement = (self.agreement_streak as f64 / 20.0).clamp(-1.0, 1.0);

        // Divergence: |imbal_B - imbal_H|
        let divergence = (binance.imbalance_5s - hl.imbalance_5s).abs();

        // Intensity ratio: binance / (binance + hl)
        let total_intensity = self.binance_intensity_ema + self.hl_intensity_ema;
        let intensity_ratio = if total_intensity > 1e-12 {
            self.binance_intensity_ema / total_intensity
        } else {
            0.5 // Default to balanced
        };

        // Correlation
        let imbalance_correlation = self.imbalance_correlation.correlation();

        // Toxicity: Replace saturating max() with divergence-based signals
        // Key insight: max(vpin_A, vpin_B) saturates at 1.0 on low-volume assets
        // because VPIN hits extremes. Instead, use signals with natural variance:
        //
        // 1. VPIN divergence: |vpin_B - vpin_H| measures information asymmetry
        //    Has variance even in calm markets because it's a *difference*
        // 2. VPIN agreement: min(vpin_B, vpin_H) when both venues show toxicity
        //
        let vpin_divergence = (binance.vpin - hl.vpin).abs();
        let avg_toxicity = (binance.vpin + hl.vpin) / 2.0;

        // Asymmetry signal: when venues disagree, scale by activity balance
        // intensity_balance peaks at 0.5 (both venues equally active)
        let intensity_balance = 4.0 * intensity_ratio * (1.0 - intensity_ratio);
        let asymmetry_signal = vpin_divergence * (0.5 + 0.5 * intensity_balance);

        // Agreement signal: when both venues show elevated toxicity
        // Uses min() so it's only high when BOTH venues see toxicity
        let min_vpin = binance.vpin.min(hl.vpin);
        let agreement_signal = min_vpin * (1.0 - vpin_divergence); // High when both high & similar

        // Combined: asymmetry OR agreement can indicate toxicity
        // This has natural variance unlike max() which saturates
        let max_toxicity = asymmetry_signal.max(agreement_signal).clamp(0.0, 1.0);

        // Combined direction (weighted by intensity)
        let combined_direction = if total_intensity > 1e-12 {
            (self.binance_intensity_ema * binance.imbalance_5s +
             self.hl_intensity_ema * hl.imbalance_5s) / total_intensity
        } else {
            (binance.imbalance_5s + hl.imbalance_5s) / 2.0
        };

        // Confidence calculation (FIXED: use additive approach instead of multiplicative)
        // This prevents a single 0 from zeroing out the entire confidence
        
        // Data sufficiency factor [0, 1]
        let data_conf = (self.sample_count as f64 / self.config.min_samples as f64).min(1.0);
        
        // Agreement factor: higher when venues agree
        let agree_conf = agreement.abs() * 0.5 + 0.5; // Maps [-1,1] agreement to [0.5, 1.0]
        
        // Correlation factor: higher when correlation is positive
        let corr_conf = (imbalance_correlation * 0.5 + 0.5).max(0.1); // Map [-1,1] to [0.1,1]
        
        // Individual venue confidences with floor
        // Use a floor of 0.3 to prevent complete gating when one venue has sparse data
        let binance_conf_adj = binance.confidence.max(0.3);
        let hl_conf_adj = hl.confidence.max(0.3);
        
        // Weighted average of venue confidences (trade count weighted)
        let venue_weight = if binance.trade_count + hl.trade_count > 0 {
            let total_trades = (binance.trade_count + hl.trade_count) as f64;
            let binance_weight = binance.trade_count as f64 / total_trades;
            let hl_weight = hl.trade_count as f64 / total_trades;
            binance_weight * binance_conf_adj + hl_weight * hl_conf_adj
        } else {
            // Fallback: arithmetic mean with floor
            (binance_conf_adj + hl_conf_adj) / 2.0
        };
        
        // Combine factors using geometric mean (less punishing than full multiplication)
        // Factors: data_conf, agree_conf, corr_conf, venue_weight
        let confidence = (data_conf * agree_conf * corr_conf * venue_weight).powf(0.25);
        
        // Ensure minimum confidence when we have some data
        let confidence = if self.sample_count >= 5 {
            confidence.max(0.1)
        } else {
            confidence
        };

        // Alerts
        let toxicity_alert = max_toxicity > self.config.toxicity_threshold;
        let divergence_alert = divergence > self.config.divergence_threshold;

        CrossVenueFeatures {
            agreement,
            divergence,
            intensity_ratio,
            imbalance_correlation,
            max_toxicity,
            avg_toxicity,
            combined_direction,
            confidence,
            toxicity_alert,
            divergence_alert,
            sample_count: self.sample_count,
        }
    }

    /// Get current cross-venue features.
    pub fn features(&self) -> CrossVenueFeatures {
        self.last_features
    }

    /// Get agreement score [-1, 1].
    pub fn agreement(&self) -> f64 {
        self.last_features.agreement
    }

    /// Get intensity ratio [0, 1] (1 = Binance dominant).
    pub fn intensity_ratio(&self) -> f64 {
        self.last_features.intensity_ratio
    }

    /// Get imbalance correlation [-1, 1].
    pub fn correlation(&self) -> f64 {
        self.last_features.imbalance_correlation
    }

    /// Get maximum toxicity across venues [0, 1].
    pub fn max_toxicity(&self) -> f64 {
        self.last_features.max_toxicity
    }

    /// Create bivariate observation from venue features.
    pub fn create_observation(&self, binance: FlowFeatureVec, hl: FlowFeatureVec, timestamp_ms: i64) -> BivariateFlowObservation {
        BivariateFlowObservation {
            binance,
            hl,
            cross: self.last_features,
            timestamp_ms,
        }
    }

    /// Check if analyzer is warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.sample_count >= self.config.min_samples
    }

    /// Get sample count.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Reset the analyzer.
    pub fn reset(&mut self) {
        self.imbalance_correlation = RollingCorrelation::new(self.config.correlation_window);
        self.binance_intensity_ema = 0.0;
        self.hl_intensity_ema = 0.0;
        self.agreement_streak = 0;
        self.sample_count = 0;
        self.last_features = CrossVenueFeatures::default();
    }

    /// Get configuration.
    pub fn config(&self) -> &CrossVenueConfig {
        &self.config
    }
}

impl Default for CrossVenueAnalyzer {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_binance_features(imbalance: f64, vpin: f64, intensity: f64) -> FlowFeatureVec {
        FlowFeatureVec {
            vpin,
            vpin_velocity: 0.0,
            imbalance_1s: imbalance,
            imbalance_5s: imbalance,
            imbalance_30s: imbalance,
            imbalance_5m: imbalance,
            intensity,
            avg_buy_size: 1.0,
            avg_sell_size: 1.0,
            size_ratio: 1.0,
            order_flow_direction: imbalance,
            timestamp_ms: 0,
            trade_count: 100,
            confidence: 0.8,
        }
    }

    #[test]
    fn test_cross_venue_agreement() {
        let mut analyzer = CrossVenueAnalyzer::default();

        // Both venues showing buy pressure
        for _ in 0..30 {
            let binance = make_binance_features(0.5, 0.3, 10.0);
            let hl = make_binance_features(0.6, 0.2, 8.0);
            analyzer.update(&binance, &hl);
        }

        let features = analyzer.features();
        assert!(features.agreement > 0.0, "Expected positive agreement, got {}", features.agreement);
        assert!(features.combined_direction > 0.0, "Expected positive direction");
    }

    #[test]
    fn test_cross_venue_divergence() {
        let mut analyzer = CrossVenueAnalyzer::default();

        // Binance buying, HL selling
        for _ in 0..30 {
            let binance = make_binance_features(0.5, 0.3, 10.0);
            let hl = make_binance_features(-0.6, 0.2, 8.0);
            analyzer.update(&binance, &hl);
        }

        let features = analyzer.features();
        assert!(features.agreement < 0.0, "Expected negative agreement (divergence), got {}", features.agreement);
        assert!(features.divergence > 0.5, "Expected high divergence, got {}", features.divergence);
        assert!(features.divergence_alert, "Expected divergence alert");
    }

    #[test]
    fn test_cross_venue_toxicity() {
        let mut analyzer = CrossVenueAnalyzer::default();

        // Both venues showing high VPIN
        for _ in 0..30 {
            let binance = make_binance_features(0.0, 0.7, 10.0);
            let hl = make_binance_features(0.0, 0.65, 8.0);
            analyzer.update(&binance, &hl);
        }

        let features = analyzer.features();
        assert!(features.max_toxicity > 0.6, "Expected high max toxicity");
        assert!(features.toxicity_alert, "Expected toxicity alert");
        assert!(features.should_be_defensive(), "Should be defensive on toxicity");
    }

    #[test]
    fn test_intensity_ratio() {
        let mut analyzer = CrossVenueAnalyzer::default();

        // High Binance activity, low HL activity
        for _ in 0..30 {
            let binance = make_binance_features(0.0, 0.3, 100.0); // High intensity
            let hl = make_binance_features(0.0, 0.2, 10.0);       // Low intensity
            analyzer.update(&binance, &hl);
        }

        let features = analyzer.features();
        assert!(features.intensity_ratio > 0.7, "Expected Binance dominance, got {}", features.intensity_ratio);
    }

    #[test]
    fn test_correlation_tracking() {
        let mut analyzer = CrossVenueAnalyzer::default();

        // Perfect correlation: both moving together
        for i in 0..50 {
            let imbal = (i as f64 / 50.0) - 0.5; // -0.5 to 0.5
            let binance = make_binance_features(imbal, 0.3, 10.0);
            let hl = make_binance_features(imbal, 0.2, 8.0);
            analyzer.update(&binance, &hl);
        }

        let features = analyzer.features();
        assert!(features.imbalance_correlation > 0.8, "Expected high correlation, got {}", features.imbalance_correlation);
    }

    #[test]
    fn test_spread_multiplier() {
        let mut features = CrossVenueFeatures::default();

        // Normal state
        features.divergence = 0.1;
        features.max_toxicity = 0.3;
        features.confidence = 0.8;
        assert!((features.spread_multiplier() - 1.0).abs() < 0.1, "Expected minimal widening");

        // High divergence
        features.divergence = 0.6;
        assert!(features.spread_multiplier() > 1.1, "Expected widening on divergence");

        // High toxicity
        features.divergence = 0.1;
        features.max_toxicity = 0.8;
        assert!(features.spread_multiplier() > 1.2, "Expected widening on toxicity");
    }

    #[test]
    fn test_skew_recommendation() {
        let mut features = CrossVenueFeatures::default();

        // Low confidence - no recommendation
        features.confidence = 0.3;
        features.agreement = 0.5;
        features.combined_direction = 0.5;
        let (dir, _) = features.skew_recommendation();
        assert_eq!(dir, 0, "No skew on low confidence");

        // High confidence, bullish
        features.confidence = 0.8;
        features.agreement = 0.6;
        features.combined_direction = 0.4;
        let (dir, mag) = features.skew_recommendation();
        assert_eq!(dir, 1, "Expected bullish skew");
        assert!(mag > 0.0, "Expected positive magnitude");

        // High confidence, bearish
        features.combined_direction = -0.4;
        let (dir, mag) = features.skew_recommendation();
        assert_eq!(dir, -1, "Expected bearish skew");
        assert!(mag > 0.0, "Expected positive magnitude");
    }

    #[test]
    fn test_bivariate_observation() {
        let mut analyzer = CrossVenueAnalyzer::default();

        let binance = make_binance_features(0.3, 0.4, 10.0);
        let hl = make_binance_features(0.2, 0.35, 8.0);

        analyzer.update(&binance, &hl);

        let obs = analyzer.create_observation(binance, hl, 1704067200000);

        assert_eq!(obs.timestamp_ms, 1704067200000);
        assert!((obs.binance.imbalance_5s - 0.3).abs() < 0.01);
        assert!((obs.hl.imbalance_5s - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_reset() {
        let mut analyzer = CrossVenueAnalyzer::default();

        // Add data
        for _ in 0..30 {
            let binance = make_binance_features(0.5, 0.3, 10.0);
            let hl = make_binance_features(0.5, 0.3, 10.0);
            analyzer.update(&binance, &hl);
        }

        assert!(analyzer.sample_count() > 0);

        analyzer.reset();

        assert_eq!(analyzer.sample_count(), 0);
    }
}
