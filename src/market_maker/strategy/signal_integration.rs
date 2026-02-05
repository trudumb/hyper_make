//! Signal Integration - Unified interface for all model signals.
//!
//! This module provides a unified interface for integrating:
//! - Lead-lag signal from Binance
//! - Informed flow decomposition
//! - Regime-conditioned kappa
//! - Model gating / IR-based confidence
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    SignalIntegrator                         │
//! │                                                             │
//! │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────┐ │
//! │  │ Lead-Lag  │  │ Informed  │  │ Regime    │  │ Model    │ │
//! │  │ Signal    │  │ Flow      │  │ Kappa     │  │ Gating   │ │
//! │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └────┬─────┘ │
//! │        │              │              │              │       │
//! │        └──────────────┼──────────────┼──────────────┘       │
//! │                       │              │                       │
//! │                       ▼              ▼                       │
//! │              ┌────────────────────────────┐                 │
//! │              │    IntegratedSignals       │                 │
//! │              │  - skew_bps               │                 │
//! │              │  - spread_multiplier      │                 │
//! │              │  - kappa_effective        │                 │
//! │              │  - confidence             │                 │
//! │              └────────────────────────────┘                 │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let mut integrator = SignalIntegrator::new(SignalIntegratorConfig::default());
//!
//! // Update with market data
//! integrator.on_binance_price(binance_mid, timestamp_ms);
//! integrator.on_hl_price(hl_mid, timestamp_ms);
//! integrator.on_trade(&trade_features);
//! integrator.set_regime_probs(regime_probs);
//!
//! // Get integrated signals for quoting
//! let signals = integrator.get_signals(hl_mid);
//! let spread = base_spread * signals.spread_multiplier;
//! let bid_skew = signals.skew_bps * signals.skew_direction as f64;
//! ```

use crate::market_maker::calibration::{InformedFlowAdjustment, ModelGating, ModelGatingConfig};
use crate::market_maker::estimator::{
    BinanceFlowAnalyzer, BinanceFlowConfig, CrossVenueAnalyzer, CrossVenueConfig,
    CrossVenueFeatures, FlowDecomposition, FlowFeatureVec, InformedFlowConfig,
    InformedFlowEstimator, LagAnalyzer, LagAnalyzerConfig, LeadLagStabilityGate,
    RegimeKappaConfig, RegimeKappaEstimator, TradeFeatures, VolatilityRegime,
};
use crate::market_maker::infra::{BinanceTradeUpdate, LeadLagSignal};
use tracing::{debug, info};

/// Configuration for signal integrator.
#[derive(Debug, Clone)]
pub struct SignalIntegratorConfig {
    /// Lead-lag analyzer configuration.
    pub lag_config: LagAnalyzerConfig,

    /// Informed flow estimator configuration.
    pub informed_flow_config: InformedFlowConfig,

    /// Regime kappa configuration.
    pub regime_kappa_config: RegimeKappaConfig,

    /// Model gating configuration.
    pub model_gating_config: ModelGatingConfig,

    /// Informed flow spread adjustment config.
    pub informed_flow_adjustment: InformedFlowAdjustment,

    /// Binance flow analyzer configuration.
    pub binance_flow_config: BinanceFlowConfig,

    /// Cross-venue analyzer configuration.
    pub cross_venue_config: CrossVenueConfig,

    /// Minimum MI for lead-lag signal to be actionable.
    pub min_mi_threshold: f64,

    /// Maximum skew from lead-lag signal (bps).
    pub max_lead_lag_skew_bps: f64,

    /// Whether to use lead-lag signal.
    pub use_lead_lag: bool,

    /// Whether to use informed flow.
    pub use_informed_flow: bool,

    /// Whether to use regime-conditioned kappa.
    pub use_regime_kappa: bool,

    /// Whether to use model gating.
    pub use_model_gating: bool,

    /// Whether to use cross-venue analysis.
    pub use_cross_venue: bool,
}

impl Default for SignalIntegratorConfig {
    fn default() -> Self {
        Self {
            lag_config: LagAnalyzerConfig::default(),
            informed_flow_config: InformedFlowConfig::default(),
            regime_kappa_config: RegimeKappaConfig::default(),
            model_gating_config: ModelGatingConfig::default(),
            informed_flow_adjustment: InformedFlowAdjustment::default(),
            binance_flow_config: BinanceFlowConfig::default(),
            cross_venue_config: CrossVenueConfig::default(),
            min_mi_threshold: 0.05,
            max_lead_lag_skew_bps: 5.0,
            use_lead_lag: true,
            use_informed_flow: true,
            use_regime_kappa: true,
            use_model_gating: true,
            use_cross_venue: true,
        }
    }
}

impl SignalIntegratorConfig {
    /// Config for HIP-3 DEX markets.
    pub fn hip3() -> Self {
        Self {
            regime_kappa_config: RegimeKappaConfig::hip3(),
            // More conservative for illiquid markets
            max_lead_lag_skew_bps: 3.0,
            ..Default::default()
        }
    }

    /// Config for liquid markets.
    pub fn liquid() -> Self {
        Self {
            regime_kappa_config: RegimeKappaConfig::liquid(),
            max_lead_lag_skew_bps: 5.0,
            ..Default::default()
        }
    }

    /// Config with all features disabled (baseline).
    pub fn disabled() -> Self {
        Self {
            use_lead_lag: false,
            use_informed_flow: false,
            use_regime_kappa: false,
            use_model_gating: false,
            use_cross_venue: false,
            ..Default::default()
        }
    }
}

/// Integrated signals for quote generation.
#[derive(Debug, Clone, Copy, Default)]
pub struct IntegratedSignals {
    // === Lead-Lag Signal ===
    /// Suggested skew direction: +1 = bullish (widen asks), -1 = bearish (widen bids), 0 = neutral.
    pub skew_direction: i8,
    /// Suggested skew magnitude in basis points (from lead-lag).
    pub lead_lag_skew_bps: f64,
    /// Whether lead-lag signal is actionable.
    pub lead_lag_actionable: bool,
    /// Current Binance-HL price difference in bps.
    pub binance_hl_diff_bps: f64,

    // === Informed Flow ===
    /// Probability that current flow is informed.
    pub p_informed: f64,
    /// Probability that current flow is noise.
    pub p_noise: f64,
    /// Probability that current flow is forced (liquidation).
    pub p_forced: f64,
    /// Toxicity score [0, 1].
    pub toxicity_score: f64,
    /// Spread multiplier from informed flow (>= 1.0).
    pub informed_flow_spread_mult: f64,

    // === Regime Kappa ===
    /// Effective kappa blended across regimes.
    pub kappa_effective: f64,
    /// Current dominant regime index.
    pub current_regime: usize,

    // === Model Gating ===
    /// Overall model confidence [0, 1].
    pub model_confidence: f64,
    /// Spread multiplier from model gating (>= 1.0).
    pub gating_spread_mult: f64,

    // === Cross-Venue (Bivariate Flow Model) ===
    /// Cross-venue direction belief [-1, 1]. Positive = bullish consensus.
    pub cross_venue_direction: f64,
    /// Cross-venue confidence [0, 1]. High when venues agree.
    pub cross_venue_confidence: f64,
    /// Cross-venue agreement score [-1, 1].
    pub cross_venue_agreement: f64,
    /// Maximum toxicity across venues [0, 1].
    pub cross_venue_max_toxicity: f64,
    /// Average toxicity across venues [0, 1].
    pub cross_venue_avg_toxicity: f64,
    /// Spread multiplier from cross-venue analysis (>= 1.0).
    pub cross_venue_spread_mult: f64,
    /// Skew recommendation from cross-venue [-1, 1].
    pub cross_venue_skew: f64,
    /// Intensity ratio [0, 1] - 0=HL dominant, 1=Binance dominant.
    pub cross_venue_intensity_ratio: f64,
    /// Imbalance correlation [-1, 1] - rolling correlation of directional pressure.
    pub cross_venue_imbalance_correlation: f64,
    /// Divergence score [0, 1] - high when venues show opposite pressure.
    pub cross_venue_divergence: f64,
    /// Whether cross-venue signals are valid.
    pub cross_venue_valid: bool,

    // === Combined ===
    /// Total spread multiplier (product of all adjustments).
    pub total_spread_mult: f64,
    /// Combined skew in bps (positive = bullish).
    pub combined_skew_bps: f64,
}

impl IntegratedSignals {
    /// Check if we should be defensive (widen spreads significantly).
    pub fn should_be_defensive(&self) -> bool {
        self.total_spread_mult > 1.3 || self.toxicity_score > 0.5 || self.model_confidence < 0.5
    }

    /// Check if signals suggest pulling quotes entirely.
    pub fn should_pull_quotes(&self) -> bool {
        self.model_confidence < 0.2 || self.toxicity_score > 0.8
    }
}

/// Signal integrator - combines all model signals.
#[derive(Debug)]
pub struct SignalIntegrator {
    config: SignalIntegratorConfig,

    /// Lead-lag analyzer.
    lag_analyzer: LagAnalyzer,

    /// Lead-lag stability gate (requires consistent positive lag).
    lead_lag_stability: LeadLagStabilityGate,

    /// Informed flow estimator.
    informed_flow: InformedFlowEstimator,

    /// Regime-conditioned kappa estimator.
    regime_kappa: RegimeKappaEstimator,

    /// Model gating system.
    model_gating: ModelGating,

    /// Binance flow analyzer for cross-venue analysis.
    binance_flow: BinanceFlowAnalyzer,

    /// Cross-venue analyzer for joint Binance+HL analysis.
    cross_venue: CrossVenueAnalyzer,

    /// Latest Binance mid price.
    latest_binance_mid: f64,

    /// Latest Hyperliquid mid price.
    latest_hl_mid: f64,

    /// Last computed lead-lag signal.
    last_lead_lag_signal: LeadLagSignal,

    /// Last computed flow decomposition.
    last_flow_decomp: FlowDecomposition,

    /// Last computed cross-venue features.
    last_cross_venue_features: CrossVenueFeatures,

    /// Latest HL flow features (from existing estimators).
    latest_hl_flow: FlowFeatureVec,

    /// Update counter for logging.
    update_count: u64,
}

impl SignalIntegrator {
    /// Create a new signal integrator.
    pub fn new(config: SignalIntegratorConfig) -> Self {
        Self {
            lag_analyzer: LagAnalyzer::new(config.lag_config.clone()),
            lead_lag_stability: LeadLagStabilityGate::default(),
            informed_flow: InformedFlowEstimator::new(config.informed_flow_config.clone()),
            regime_kappa: RegimeKappaEstimator::new(config.regime_kappa_config.clone()),
            model_gating: ModelGating::new(config.model_gating_config.clone()),
            binance_flow: BinanceFlowAnalyzer::new(config.binance_flow_config.clone()),
            cross_venue: CrossVenueAnalyzer::new(config.cross_venue_config.clone()),
            config,
            latest_binance_mid: 0.0,
            latest_hl_mid: 0.0,
            last_lead_lag_signal: LeadLagSignal::default(),
            last_flow_decomp: FlowDecomposition::default(),
            last_cross_venue_features: CrossVenueFeatures::default(),
            latest_hl_flow: FlowFeatureVec::default(),
            update_count: 0,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(SignalIntegratorConfig::default())
    }

    // =========================================================================
    // Data Input Methods
    // =========================================================================

    /// Update with Binance price.
    pub fn on_binance_price(&mut self, mid_price: f64, timestamp_ms: i64) {
        if !self.config.use_lead_lag || mid_price <= 0.0 {
            return;
        }

        self.latest_binance_mid = mid_price;
        self.lag_analyzer.add_signal(timestamp_ms, mid_price);

        // Update lead-lag signal with stability gate
        if let Some((lag_ms, mi)) = self.lag_analyzer.optimal_lag() {
            // Record observation in stability gate
            self.lead_lag_stability.record(lag_ms, mi);

            // Compute signal with stability confidence
            let stability_conf = self.lead_lag_stability.stability_confidence();

            self.last_lead_lag_signal = LeadLagSignal::compute(
                self.latest_binance_mid,
                self.latest_hl_mid,
                lag_ms,
                mi,
                self.config.min_mi_threshold,
                stability_conf,
            );
        }
    }

    /// Update with Hyperliquid price.
    pub fn on_hl_price(&mut self, mid_price: f64, timestamp_ms: i64) {
        if mid_price <= 0.0 {
            return;
        }

        self.latest_hl_mid = mid_price;

        if self.config.use_lead_lag {
            self.lag_analyzer.add_target(timestamp_ms, mid_price);
        }

        self.update_count += 1;
    }

    /// Update with trade observation.
    pub fn on_trade(&mut self, features: &TradeFeatures) {
        if self.config.use_informed_flow {
            self.informed_flow.on_trade(features);
            self.last_flow_decomp = self.informed_flow.decomposition();
        }
    }

    /// Update with fill observation (for kappa estimation).
    pub fn on_fill(&mut self, timestamp_ms: u64, price: f64, size: f64, mid: f64) {
        if self.config.use_regime_kappa {
            self.regime_kappa.on_fill(timestamp_ms, price, size, mid);
        }

        // Update model gating with fill prediction outcome
        if self.config.use_model_gating {
            // Simple heuristic: fill closer than 10 bps is "good" prediction
            let fill_distance_bps = ((price - mid) / mid).abs() * 10000.0;
            let predicted_fill_prob = 0.5; // Placeholder - should come from actual prediction
            let filled = fill_distance_bps < 10.0;
            self.model_gating.update_kappa(predicted_fill_prob, filled);
        }
    }

    /// Set current regime probabilities (from HMM).
    pub fn set_regime_probabilities(&mut self, probs: [f64; 4]) {
        if self.config.use_regime_kappa {
            self.regime_kappa.set_regime_probabilities(probs);
        }
    }

    /// Set current regime (from volatility detector).
    pub fn set_regime(&mut self, regime: VolatilityRegime) {
        if self.config.use_regime_kappa {
            self.regime_kappa.set_regime(regime);
        }
    }

    /// Update adverse selection model tracking.
    pub fn update_as_prediction(&mut self, predicted_as_prob: f64, actual_adverse: bool) {
        if self.config.use_model_gating {
            self.model_gating
                .update_adverse_selection(predicted_as_prob, actual_adverse);
        }
    }

    /// Update informed flow model tracking.
    pub fn update_informed_prediction(&mut self, p_informed: f64, was_informed: bool) {
        if self.config.use_model_gating {
            self.model_gating
                .update_informed_flow(p_informed, was_informed);
        }
    }

    /// Update lead-lag model tracking.
    pub fn update_lead_lag_prediction(
        &mut self,
        predicted_direction_prob: f64,
        correct_direction: bool,
    ) {
        if self.config.use_model_gating {
            self.model_gating
                .update_lead_lag(predicted_direction_prob, correct_direction);
        }
    }

    /// Update with Binance trade for cross-venue flow analysis.
    ///
    /// This feeds the BinanceFlowAnalyzer which computes VPIN, volume imbalance,
    /// and intensity metrics for Binance. Combined with HL flow features, this
    /// enables bivariate cross-venue analysis.
    pub fn on_binance_trade(&mut self, trade: &BinanceTradeUpdate) {
        if !self.config.use_cross_venue {
            return;
        }

        // Update Binance flow analyzer
        self.binance_flow.on_trade(trade);

        // Update cross-venue features if we have both venues
        self.update_cross_venue_features();
    }

    /// Update HL flow features from existing estimators.
    ///
    /// This should be called when the HL flow estimators produce new features.
    /// The features are used for cross-venue comparison with Binance.
    pub fn set_hl_flow_features(&mut self, features: FlowFeatureVec) {
        self.latest_hl_flow = features;

        if self.config.use_cross_venue {
            self.update_cross_venue_features();
        }
    }

    /// Update cross-venue features using both Binance and HL flow.
    fn update_cross_venue_features(&mut self) {
        let binance_flow = self.binance_flow.flow_features();
        let hl_flow = &self.latest_hl_flow;

        // Update the cross-venue analyzer with both flow feature vectors
        self.cross_venue.update(&binance_flow, hl_flow);

        // Cache the latest cross-venue features
        self.last_cross_venue_features = self.cross_venue.features();
    }

    /// Get the current cross-venue features.
    pub fn cross_venue_features(&self) -> &CrossVenueFeatures {
        &self.last_cross_venue_features
    }

    /// Get the Binance flow analyzer for direct access.
    pub fn binance_flow(&self) -> &BinanceFlowAnalyzer {
        &self.binance_flow
    }

    /// Get the cross-venue analyzer for direct access.
    pub fn cross_venue_analyzer(&self) -> &CrossVenueAnalyzer {
        &self.cross_venue
    }

    // =========================================================================
    // Signal Output
    // =========================================================================

    /// Get integrated signals for quote generation.
    pub fn get_signals(&self) -> IntegratedSignals {
        let mut signals = IntegratedSignals::default();

        // === Lead-Lag Signal ===
        if self.config.use_lead_lag && self.last_lead_lag_signal.is_actionable {
            signals.skew_direction = self.last_lead_lag_signal.skew_direction;
            signals.lead_lag_skew_bps = self
                .last_lead_lag_signal
                .skew_magnitude_bps
                .min(self.config.max_lead_lag_skew_bps);
            signals.lead_lag_actionable = true;
            signals.binance_hl_diff_bps = self.last_lead_lag_signal.diff_bps;
        }

        // === Informed Flow ===
        if self.config.use_informed_flow {
            signals.p_informed = self.last_flow_decomp.p_informed;
            signals.p_noise = self.last_flow_decomp.p_noise;
            signals.p_forced = self.last_flow_decomp.p_forced;
            signals.toxicity_score = self.last_flow_decomp.toxicity_score();
            signals.informed_flow_spread_mult = self
                .config
                .informed_flow_adjustment
                .spread_multiplier(signals.p_informed);
        } else {
            signals.informed_flow_spread_mult = 1.0;
        }

        // === Regime Kappa ===
        if self.config.use_regime_kappa {
            signals.kappa_effective = self.regime_kappa.kappa_effective();
            signals.current_regime = self.regime_kappa.current_regime();
        } else {
            signals.kappa_effective = 2000.0; // Default
            signals.current_regime = 1; // Normal
        }

        // === Model Gating ===
        if self.config.use_model_gating {
            let weights = self.model_gating.model_weights();
            signals.model_confidence = weights.average_weight();
            signals.gating_spread_mult = self.model_gating.spread_multiplier();
        } else {
            signals.model_confidence = 1.0;
            signals.gating_spread_mult = 1.0;
        }

        // === Cross-Venue (Bivariate Flow Model) ===
        if self.config.use_cross_venue {
            let cv = &self.last_cross_venue_features;
            signals.cross_venue_direction = cv.combined_direction;
            signals.cross_venue_confidence = cv.confidence;
            signals.cross_venue_agreement = cv.agreement;
            signals.cross_venue_max_toxicity = cv.max_toxicity;
            signals.cross_venue_avg_toxicity = cv.avg_toxicity;
            signals.cross_venue_spread_mult = cv.spread_multiplier();
            // skew_recommendation returns (direction, magnitude_bps)
            let (skew_dir, _skew_mag) = cv.skew_recommendation();
            signals.cross_venue_skew = skew_dir as f64 * cv.combined_direction.abs();
            signals.cross_venue_intensity_ratio = cv.intensity_ratio;
            signals.cross_venue_imbalance_correlation = cv.imbalance_correlation;
            signals.cross_venue_divergence = cv.divergence;
            signals.cross_venue_valid = cv.sample_count >= 20; // min_samples threshold
        } else {
            signals.cross_venue_spread_mult = 1.0;
            signals.cross_venue_valid = false;
        }

        // === Combined Signals ===
        // Include cross-venue spread multiplier in total
        signals.total_spread_mult = signals.informed_flow_spread_mult
            * signals.gating_spread_mult
            * signals.cross_venue_spread_mult;

        // Combine skews: lead-lag is primary, cross-venue provides additional signal
        let base_skew_bps = if signals.lead_lag_actionable {
            signals.lead_lag_skew_bps * signals.skew_direction as f64
        } else {
            0.0
        };

        // Add cross-venue skew contribution (scaled by confidence)
        let cross_venue_skew_bps = if signals.cross_venue_valid {
            // Scale cross-venue skew ([-1, 1]) to bps
            // Use half of max_lead_lag_skew_bps as max cross-venue contribution
            signals.cross_venue_skew
                * signals.cross_venue_confidence
                * (self.config.max_lead_lag_skew_bps / 2.0)
        } else {
            0.0
        };

        signals.combined_skew_bps = base_skew_bps + cross_venue_skew_bps;

        // Log periodically
        if self.update_count % 100 == 0 && self.update_count > 0 {
            self.log_status(&signals);
        }

        signals
    }

    /// Get current lead-lag signal.
    pub fn lead_lag_signal(&self) -> LeadLagSignal {
        self.last_lead_lag_signal
    }

    /// Get current flow decomposition.
    pub fn flow_decomposition(&self) -> FlowDecomposition {
        self.last_flow_decomp
    }

    /// Get regime kappa estimator for direct access.
    pub fn regime_kappa(&self) -> &RegimeKappaEstimator {
        &self.regime_kappa
    }

    /// Get model gating for direct access.
    pub fn model_gating(&self) -> &ModelGating {
        &self.model_gating
    }

    /// Check if any signal component is warmed up.
    pub fn is_warmed_up(&self) -> bool {
        let lag_ready = !self.config.use_lead_lag || self.lag_analyzer.is_ready();
        let flow_ready = !self.config.use_informed_flow || self.informed_flow.is_warmed_up();
        let kappa_ready = !self.config.use_regime_kappa || self.regime_kappa.is_warmed_up();
        let cross_venue_ready =
            !self.config.use_cross_venue || self.binance_flow.is_warmed_up();

        lag_ready || flow_ready || kappa_ready || cross_venue_ready
    }

    /// Get lag analyzer status for diagnostics.
    /// Returns (is_ready, (signal_count, target_count), optimal_lag_ms, mi_bits, last_signal, sample_timestamps)
    #[allow(clippy::type_complexity)]
    pub fn lag_analyzer_status(
        &self,
    ) -> (
        bool,
        (usize, usize),
        Option<i64>,
        Option<f64>,
        &LeadLagSignal,
        ((Option<i64>, Option<i64>), (Option<i64>, Option<i64>)),
    ) {
        let (lag, mi) = self
            .lag_analyzer
            .optimal_lag()
            .map_or((None, None), |(l, m)| (Some(l), Some(m)));
        (
            self.lag_analyzer.is_ready(),
            self.lag_analyzer.observation_counts(),
            lag,
            mi,
            &self.last_lead_lag_signal,
            self.lag_analyzer.sample_timestamps(),
        )
    }

    /// Log current status.
    fn log_status(&self, signals: &IntegratedSignals) {
        if signals.lead_lag_actionable {
            info!(
                binance_hl_diff_bps = %format!("{:.1}", signals.binance_hl_diff_bps),
                skew_direction = signals.skew_direction,
                skew_bps = %format!("{:.1}", signals.lead_lag_skew_bps),
                "Lead-lag signal ACTIVE"
            );
        }

        debug!(
            p_informed = %format!("{:.2}", signals.p_informed),
            p_noise = %format!("{:.2}", signals.p_noise),
            p_forced = %format!("{:.2}", signals.p_forced),
            toxicity = %format!("{:.2}", signals.toxicity_score),
            informed_mult = %format!("{:.2}x", signals.informed_flow_spread_mult),
            "Informed flow status"
        );

        debug!(
            kappa_eff = %format!("{:.0}", signals.kappa_effective),
            regime = signals.current_regime,
            model_conf = %format!("{:.2}", signals.model_confidence),
            gating_mult = %format!("{:.2}x", signals.gating_spread_mult),
            total_mult = %format!("{:.2}x", signals.total_spread_mult),
            "Signal integration summary"
        );
    }

    /// Reset all components.
    pub fn reset(&mut self) {
        self.lag_analyzer.reset();
        self.lead_lag_stability.reset();
        self.informed_flow.reset();
        self.regime_kappa.reset();
        self.model_gating.reset();
        self.binance_flow.reset();
        self.cross_venue.reset();
        self.latest_binance_mid = 0.0;
        self.latest_hl_mid = 0.0;
        self.last_lead_lag_signal = LeadLagSignal::default();
        self.last_flow_decomp = FlowDecomposition::default();
        self.last_cross_venue_features = CrossVenueFeatures::default();
        self.latest_hl_flow = FlowFeatureVec::default();
        self.update_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_integrator_default() {
        let integrator = SignalIntegrator::default_config();
        let signals = integrator.get_signals();

        // Should have default values
        assert_eq!(signals.skew_direction, 0);
        assert!(!signals.lead_lag_actionable);
        // When p_informed = 0.0 (no data), we're below tighten_threshold (0.05),
        // so spread_multiplier returns min_tighten_mult (0.9)
        assert!(signals.informed_flow_spread_mult >= 0.9);
        assert!(signals.informed_flow_spread_mult <= 1.0);
    }

    #[test]
    fn test_signal_integrator_disabled() {
        let config = SignalIntegratorConfig::disabled();
        let integrator = SignalIntegrator::new(config);
        let signals = integrator.get_signals();

        // All multipliers should be 1.0
        assert!((signals.total_spread_mult - 1.0).abs() < 0.01);
        assert!((signals.model_confidence - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_integrated_signals_defensive() {
        let mut signals = IntegratedSignals::default();
        signals.total_spread_mult = 1.5;

        assert!(signals.should_be_defensive());

        signals.total_spread_mult = 1.0;
        signals.toxicity_score = 0.6;
        assert!(signals.should_be_defensive());
    }

    #[test]
    fn test_integrated_signals_pull_quotes() {
        let mut signals = IntegratedSignals::default();
        signals.model_confidence = 0.1;

        assert!(signals.should_pull_quotes());

        signals.model_confidence = 0.5;
        signals.toxicity_score = 0.9;
        assert!(signals.should_pull_quotes());
    }

    #[test]
    fn test_regime_kappa_integration() {
        let mut integrator = SignalIntegrator::default_config();

        // Set to High volatility regime
        integrator.set_regime(VolatilityRegime::High);

        let signals = integrator.get_signals();

        // Kappa should be lower for High regime
        assert!(signals.kappa_effective < 2000.0);
        assert_eq!(signals.current_regime, 2);
    }

    #[test]
    fn test_on_trade_updates_flow() {
        let mut integrator = SignalIntegrator::default_config();

        // Feed trades that look informed
        for i in 0..200 {
            let features = TradeFeatures {
                size: 5.0,
                inter_arrival_ms: 100,
                price_impact_bps: 15.0,
                timestamp_ms: i * 100,
                ..Default::default()
            };
            integrator.on_trade(&features);
        }

        let signals = integrator.get_signals();

        // Should have higher P(informed) after informed-looking trades
        assert!(signals.p_informed > 0.1);
    }
}
