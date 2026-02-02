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
    FlowDecomposition, InformedFlowConfig, InformedFlowEstimator, LagAnalyzer, LagAnalyzerConfig,
    RegimeKappaConfig, RegimeKappaEstimator, TradeFeatures, VolatilityRegime,
};
use crate::market_maker::infra::LeadLagSignal;
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
}

impl Default for SignalIntegratorConfig {
    fn default() -> Self {
        Self {
            lag_config: LagAnalyzerConfig::default(),
            informed_flow_config: InformedFlowConfig::default(),
            regime_kappa_config: RegimeKappaConfig::default(),
            model_gating_config: ModelGatingConfig::default(),
            informed_flow_adjustment: InformedFlowAdjustment::default(),
            min_mi_threshold: 0.05,
            max_lead_lag_skew_bps: 5.0,
            use_lead_lag: true,
            use_informed_flow: true,
            use_regime_kappa: true,
            use_model_gating: true,
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
pub struct SignalIntegrator {
    config: SignalIntegratorConfig,

    /// Lead-lag analyzer.
    lag_analyzer: LagAnalyzer,

    /// Informed flow estimator.
    informed_flow: InformedFlowEstimator,

    /// Regime-conditioned kappa estimator.
    regime_kappa: RegimeKappaEstimator,

    /// Model gating system.
    model_gating: ModelGating,

    /// Latest Binance mid price.
    latest_binance_mid: f64,

    /// Latest Hyperliquid mid price.
    latest_hl_mid: f64,

    /// Last computed lead-lag signal.
    last_lead_lag_signal: LeadLagSignal,

    /// Last computed flow decomposition.
    last_flow_decomp: FlowDecomposition,

    /// Update counter for logging.
    update_count: u64,
}

impl SignalIntegrator {
    /// Create a new signal integrator.
    pub fn new(config: SignalIntegratorConfig) -> Self {
        Self {
            lag_analyzer: LagAnalyzer::new(config.lag_config.clone()),
            informed_flow: InformedFlowEstimator::new(config.informed_flow_config.clone()),
            regime_kappa: RegimeKappaEstimator::new(config.regime_kappa_config.clone()),
            model_gating: ModelGating::new(config.model_gating_config.clone()),
            config,
            latest_binance_mid: 0.0,
            latest_hl_mid: 0.0,
            last_lead_lag_signal: LeadLagSignal::default(),
            last_flow_decomp: FlowDecomposition::default(),
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

        // Update lead-lag signal
        if let Some((lag_ms, mi)) = self.lag_analyzer.optimal_lag() {
            self.last_lead_lag_signal = LeadLagSignal::compute(
                self.latest_binance_mid,
                self.latest_hl_mid,
                lag_ms,
                mi,
                self.config.min_mi_threshold,
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

        // === Combined Signals ===
        signals.total_spread_mult =
            signals.informed_flow_spread_mult * signals.gating_spread_mult;

        // Combine skews (lead-lag is primary, informed flow modulates)
        signals.combined_skew_bps = if signals.lead_lag_actionable {
            signals.lead_lag_skew_bps * signals.skew_direction as f64
        } else {
            0.0
        };

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

        lag_ready || flow_ready || kappa_ready
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
        self.informed_flow.reset();
        self.regime_kappa.reset();
        self.model_gating.reset();
        self.latest_binance_mid = 0.0;
        self.latest_hl_mid = 0.0;
        self.last_lead_lag_signal = LeadLagSignal::default();
        self.last_flow_decomp = FlowDecomposition::default();
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
