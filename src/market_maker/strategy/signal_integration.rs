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
    BinanceFlowAnalyzer, BinanceFlowConfig, BuyPressureTracker, CrossVenueAnalyzer,
    CrossVenueConfig, CrossVenueFeatures, FlowDecomposition, FlowFeatureVec, InformedFlowConfig,
    InformedFlowEstimator, LagAnalyzer, LagAnalyzerConfig, LeadLagStabilityGate,
    RegimeKappaConfig, RegimeKappaEstimator, TimestampRange, TradeFeatures, VolatilityRegime,
};
use crate::market_maker::infra::{BinanceTradeUpdate, LeadLagSignal};
use tracing::{debug, info};

/// Lag analyzer diagnostic status.
pub struct LagAnalyzerStatus<'a> {
    pub is_ready: bool,
    pub observation_counts: (usize, usize),
    pub optimal_lag_ms: Option<i64>,
    pub mi_bits: Option<f64>,
    pub last_signal: &'a LeadLagSignal,
    pub sample_timestamps: (TimestampRange, TimestampRange),
}

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

    /// Whether to apply per-signal model gating (multiply each signal by its IR weight).
    /// When true, signals from low-confidence models are attenuated individually.
    /// When false, only the aggregate gating_spread_mult is applied (legacy behavior).
    pub use_per_signal_gating: bool,

    /// Minimum model weight for soft scaling (only used when hard gate passes).
    /// With two-tier gating: hard gate (should_use_model → zero if false),
    /// then soft scaling by model_weight() if above this floor.
    /// Set to 0.0 to use raw model_weight with no floor after hard gate.
    pub signal_gating_floor: f64,

    /// Whether to blend VPIN toxicity with EM-based informed flow.
    /// VPIN varies naturally (volume-synchronized), fixing the EM concentration problem.
    pub use_vpin_toxicity: bool,

    /// VPIN threshold above which spreads should be widened.
    pub vpin_widen_threshold: f64,

    /// Blend weight for VPIN vs EM toxicity [0=pure EM, 1=pure VPIN].
    /// Default 0.7 (favor VPIN since it has natural variation).
    pub vpin_blend_weight: f64,

    /// Whether to use buy pressure z-score signal.
    pub use_buy_pressure: bool,

    /// Z-score threshold beyond which buy pressure contributes to skew.
    pub buy_pressure_z_threshold: f64,

    /// Buy pressure skew as fraction of max_lead_lag_skew_bps per unit z.
    /// At z=threshold+1, skew = this * max_lead_lag_skew_bps.
    /// Default 0.1 means buy pressure can contribute up to ~30% of max skew.
    pub buy_pressure_skew_fraction: f64,

    /// Maximum fraction of max_lead_lag_skew_bps that buy pressure can contribute.
    /// Default 0.3 → in a 5 bps max skew context, cap at 1.5 bps.
    pub buy_pressure_max_fraction: f64,
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
            use_per_signal_gating: true,
            signal_gating_floor: 0.0,
            use_vpin_toxicity: true,
            vpin_widen_threshold: 0.6,
            vpin_blend_weight: 0.7,
            use_buy_pressure: true,
            buy_pressure_z_threshold: 1.5,
            buy_pressure_skew_fraction: 0.1,
            buy_pressure_max_fraction: 0.3,
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
            use_per_signal_gating: false,
            use_vpin_toxicity: false,
            use_buy_pressure: false,
            ..Default::default()
        }
    }
}

/// Records each signal's individual contribution before blending.
/// Used by the analytics module for per-signal PnL attribution.
#[derive(Debug, Clone, Copy, Default)]
pub struct SignalContributionRecord {
    /// Lead-lag cross-exchange skew in bps
    pub lead_lag_skew_bps: f64,
    /// Whether lead-lag signal was significant (above MI threshold)
    pub lead_lag_active: bool,
    /// IR-based gating weight for lead-lag (0.0-1.0)
    pub lead_lag_gating_weight: f64,

    /// Informed flow spread multiplier (1.0 = no adjustment)
    pub informed_flow_spread_mult: f64,
    /// Whether informed flow decomposition is active
    pub informed_flow_active: bool,
    /// IR-based gating weight for informed flow (0.0-1.0)
    pub informed_flow_gating_weight: f64,

    /// Effective kappa from regime detection
    pub regime_kappa_effective: f64,
    /// Whether regime kappa is active
    pub regime_active: bool,

    /// Cross-venue spread multiplier
    pub cross_venue_spread_mult: f64,
    /// Cross-venue skew in bps
    pub cross_venue_skew_bps: f64,
    /// Whether cross-venue signals are valid
    pub cross_venue_active: bool,

    /// VPIN-based spread widening (multiplier, 1.0 = no adjustment)
    pub vpin_spread_mult: f64,
    /// Whether VPIN is above toxicity threshold
    pub vpin_active: bool,

    /// Buy pressure skew in bps
    pub buy_pressure_skew_bps: f64,
    /// Whether buy pressure is above z-threshold
    pub buy_pressure_active: bool,
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

    // === VPIN Blend ===
    /// Hyperliquid VPIN value [0, 1] (volume-synchronized toxicity).
    pub hl_vpin: f64,
    /// VPIN velocity (rate of change, positive = rising toxicity).
    pub hl_vpin_velocity: f64,

    // === Buy Pressure ===
    /// Buy pressure z-score (deviation of buy_ratio from rolling mean).
    pub buy_pressure_z: f64,

    // === Per-Signal Gating Diagnostics ===
    /// Model gating weight applied to lead-lag signal [0, 1].
    pub lead_lag_gating_weight: f64,
    /// Model gating weight applied to informed flow signal [0, 1].
    pub informed_flow_gating_weight: f64,

    // === Lead-Lag Significance Diagnostics ===
    /// Whether the lead-lag MI passed the significance test against the null distribution.
    pub lead_lag_significant: bool,
    /// 95th percentile of the null MI distribution (for diagnostics).
    pub lead_lag_null_p95: f64,

    // === Combined ===
    /// Total spread multiplier (product of all adjustments).
    pub total_spread_mult: f64,
    /// Combined skew in bps (positive = bullish).
    pub combined_skew_bps: f64,

    // === Attribution ===
    /// Per-signal contribution record for attribution analysis.
    /// Populated by get_signals(), None for manually constructed instances.
    pub signal_contributions: Option<SignalContributionRecord>,
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

    /// Latest VPIN value from HL VPIN estimator.
    latest_hl_vpin: f64,
    /// Latest VPIN velocity.
    latest_hl_vpin_velocity: f64,
    /// Whether VPIN has enough data to be valid.
    vpin_valid: bool,

    /// Buy pressure z-score tracker.
    buy_pressure: BuyPressureTracker,

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
            latest_hl_vpin: 0.0,
            latest_hl_vpin_velocity: 0.0,
            vpin_valid: false,
            buy_pressure: BuyPressureTracker::new(),
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
        } else {
            // MI insignificant or lag unavailable — mark signal stale
            self.last_lead_lag_signal.is_actionable = false;
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

        // Auto-wire Binance VPIN into toxicity blend
        let vpin = self.binance_flow.vpin();
        let velocity = self.binance_flow.vpin_velocity();
        let valid = self.binance_flow.vpin_is_valid();
        self.set_vpin(vpin, velocity, valid);

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

    /// Update with VPIN values from the Hyperliquid VPIN estimator.
    ///
    /// Called after each VPIN bucket completes in the trade handler.
    pub fn set_vpin(&mut self, vpin: f64, velocity: f64, valid: bool) {
        self.latest_hl_vpin = vpin;
        self.latest_hl_vpin_velocity = velocity;
        self.vpin_valid = valid;
    }

    /// Update buy pressure tracker with a trade observation.
    ///
    /// Should be called alongside existing trade handlers to maintain
    /// a rolling z-score of buy/sell ratio.
    pub fn on_trade_for_pressure(&mut self, size: f64, is_buy: bool) {
        if self.config.use_buy_pressure {
            self.buy_pressure.on_trade(size, is_buy);
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
        let mut signals = IntegratedSignals {
            lead_lag_significant: self.lag_analyzer.is_lag_significant(),
            lead_lag_null_p95: self.lag_analyzer.null_mi_p95(),
            ..Default::default()
        };

        // === Lead-Lag Signal ===
        if self.config.use_lead_lag && self.last_lead_lag_signal.is_actionable {
            // Two-tier per-signal gating:
            // Tier 1 (hard gate): should_use_model() → zero if false (weight ≤ 0.3)
            // Tier 2 (soft scale): model_weight() → attenuate proportionally
            let ll_weight = if self.config.use_per_signal_gating && self.config.use_model_gating {
                if !self.model_gating.should_use_model("lead_lag") {
                    signals.lead_lag_gating_weight = 0.0;
                    0.0 // Hard gate: model fails significance → zero
                } else {
                    let w = self.model_gating.model_weight("lead_lag");
                    signals.lead_lag_gating_weight = w;
                    // Soft scale: floor prevents near-zero leakage after hard gate passes
                    if w < self.config.signal_gating_floor {
                        self.config.signal_gating_floor
                    } else {
                        w
                    }
                }
            } else {
                signals.lead_lag_gating_weight = 1.0;
                1.0
            };

            signals.skew_direction = self.last_lead_lag_signal.skew_direction;
            signals.lead_lag_skew_bps = self
                .last_lead_lag_signal
                .skew_magnitude_bps
                .min(self.config.max_lead_lag_skew_bps)
                * ll_weight;
            signals.lead_lag_actionable = ll_weight > 0.0;
            signals.binance_hl_diff_bps = self.last_lead_lag_signal.diff_bps;
        }

        // === Informed Flow ===
        if self.config.use_informed_flow {
            // Two-tier per-signal gating (same pattern as lead-lag)
            let if_weight = if self.config.use_per_signal_gating && self.config.use_model_gating {
                if !self.model_gating.should_use_model("informed_flow") {
                    signals.informed_flow_gating_weight = 0.0;
                    0.0 // Hard gate: model fails significance → zero
                } else {
                    let w = self.model_gating.model_weight("informed_flow");
                    signals.informed_flow_gating_weight = w;
                    if w < self.config.signal_gating_floor {
                        self.config.signal_gating_floor
                    } else {
                        w
                    }
                }
            } else {
                signals.informed_flow_gating_weight = 1.0;
                1.0
            };

            let raw_p_informed = self.last_flow_decomp.p_informed;
            let em_toxicity = self.last_flow_decomp.toxicity_score();

            // Blend VPIN with EM toxicity when VPIN is available and enabled
            let blended_toxicity = if self.config.use_vpin_toxicity && self.vpin_valid {
                let vpin = self.latest_hl_vpin;
                // Guard against saturated VPIN (degenerate bucket composition)
                if vpin >= 0.95 || vpin <= 0.05 {
                    em_toxicity // Fall back to EM-only when VPIN is at extremes
                } else {
                    let w = self.config.vpin_blend_weight;
                    w * vpin + (1.0 - w) * em_toxicity
                }
            } else {
                em_toxicity
            };

            // Store VPIN diagnostics
            signals.hl_vpin = self.latest_hl_vpin;
            signals.hl_vpin_velocity = self.latest_hl_vpin_velocity;

            signals.p_informed = raw_p_informed * if_weight;
            signals.p_noise = self.last_flow_decomp.p_noise;
            signals.p_forced = self.last_flow_decomp.p_forced;
            signals.toxicity_score = blended_toxicity * if_weight;
            signals.informed_flow_spread_mult = if if_weight > 0.0 {
                // Use blended toxicity for spread adjustment instead of raw p_informed
                // This ensures VPIN-driven toxicity also widens spreads
                let effective_p = if self.config.use_vpin_toxicity && self.vpin_valid {
                    blended_toxicity.max(signals.p_informed)
                } else {
                    signals.p_informed
                };
                self.config
                    .informed_flow_adjustment
                    .spread_multiplier(effective_p)
            } else {
                1.0 // No spread adjustment when gated out
            };
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

        // Add cross-venue skew contribution (scaled by confidence and model gating)
        let cross_venue_skew_bps = if signals.cross_venue_valid {
            // Per-signal gating: scale by average model weight
            let cv_gate = if self.config.use_per_signal_gating && self.config.use_model_gating {
                let avg_w = (signals.lead_lag_gating_weight + signals.informed_flow_gating_weight) / 2.0;
                if avg_w < self.config.signal_gating_floor { 0.0 } else { avg_w }
            } else {
                1.0
            };
            // Scale cross-venue skew ([-1, 1]) to bps
            // Use half of max_lead_lag_skew_bps as max cross-venue contribution
            signals.cross_venue_skew
                * signals.cross_venue_confidence
                * (self.config.max_lead_lag_skew_bps / 2.0)
                * cv_gate
        } else {
            0.0
        };

        // === Buy Pressure Z-Score Contribution ===
        // Skew scales proportionally to max_lead_lag_skew_bps so it adapts to asset spread.
        // In a 2 bps spread asset (max_skew=3), 0.1 fraction → 0.3 bps/z, cap 0.9 bps.
        // In a 10 bps spread asset (max_skew=5), 0.1 fraction → 0.5 bps/z, cap 1.5 bps.
        let buy_pressure_skew_bps = if self.config.use_buy_pressure && self.buy_pressure.is_warmed_up() {
            let z = self.buy_pressure.z_score();
            signals.buy_pressure_z = z;
            let threshold = self.config.buy_pressure_z_threshold;
            if z.abs() > threshold {
                let excess = z.abs() - threshold;
                let bps_per_z = self.config.buy_pressure_skew_fraction * self.config.max_lead_lag_skew_bps;
                let cap = self.config.buy_pressure_max_fraction * self.config.max_lead_lag_skew_bps;
                let raw = excess * bps_per_z;
                raw.min(cap) * z.signum()
            } else {
                0.0
            }
        } else {
            0.0
        };

        signals.combined_skew_bps = base_skew_bps + cross_venue_skew_bps + buy_pressure_skew_bps;

        // === Build per-signal contribution record for attribution ===
        let vpin_active = self.config.use_vpin_toxicity && self.vpin_valid
            && self.latest_hl_vpin > self.config.vpin_widen_threshold;
        signals.signal_contributions = Some(SignalContributionRecord {
            lead_lag_skew_bps: signals.lead_lag_skew_bps,
            lead_lag_active: signals.lead_lag_actionable,
            lead_lag_gating_weight: signals.lead_lag_gating_weight,

            informed_flow_spread_mult: signals.informed_flow_spread_mult,
            informed_flow_active: self.config.use_informed_flow && signals.p_informed > 0.0,
            informed_flow_gating_weight: signals.informed_flow_gating_weight,

            regime_kappa_effective: signals.kappa_effective,
            regime_active: self.config.use_regime_kappa,

            cross_venue_spread_mult: signals.cross_venue_spread_mult,
            cross_venue_skew_bps,
            cross_venue_active: signals.cross_venue_valid,

            vpin_spread_mult: if vpin_active {
                // Isolate VPIN contribution: ratio of blended to EM-only spread mult
                // When VPIN is not active this is 1.0 (no adjustment)
                let em_only_mult = self.config.informed_flow_adjustment
                    .spread_multiplier(self.last_flow_decomp.p_informed);
                if em_only_mult > 0.0 {
                    signals.informed_flow_spread_mult / em_only_mult
                } else {
                    1.0
                }
            } else {
                1.0
            },
            vpin_active,

            buy_pressure_skew_bps,
            buy_pressure_active: self.config.use_buy_pressure
                && self.buy_pressure.is_warmed_up()
                && signals.buy_pressure_z.abs() > self.config.buy_pressure_z_threshold,
        });

        // Log periodically
        if self.update_count.is_multiple_of(100) && self.update_count > 0 {
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

    /// Get the model gating spread multiplier based on IR confidence.
    /// Returns 1.0 when models are well-calibrated, up to 2.0 when uncalibrated.
    pub fn model_gating_spread_multiplier(&self) -> f64 {
        if self.config.use_model_gating {
            self.model_gating.spread_multiplier()
        } else {
            1.0
        }
    }

    /// Check if all enabled signal components are warmed up.
    pub fn is_warmed_up(&self) -> bool {
        let lag_ready = !self.config.use_lead_lag || self.lag_analyzer.is_ready();
        let flow_ready = !self.config.use_informed_flow || self.informed_flow.is_warmed_up();
        let kappa_ready = !self.config.use_regime_kappa || self.regime_kappa.is_warmed_up();
        let cross_venue_ready =
            !self.config.use_cross_venue || self.binance_flow.is_warmed_up();

        lag_ready && flow_ready && kappa_ready && cross_venue_ready
    }

    /// Spread multiplier based on signal staleness.
    /// Returns > 1.0 when enabled signals are stale, providing defensive widening.
    pub fn staleness_spread_multiplier(&self) -> f64 {
        let mut stale_count = 0;

        if self.config.use_lead_lag && !self.lag_analyzer.is_ready() {
            stale_count += 1;
        }
        if self.config.use_cross_venue && !self.binance_flow.is_warmed_up() {
            stale_count += 1;
        }
        if self.config.use_informed_flow && !self.informed_flow.is_warmed_up() {
            stale_count += 1;
        }
        if self.config.use_regime_kappa && !self.regime_kappa.is_warmed_up() {
            stale_count += 1;
        }

        match stale_count {
            0 => 1.0,
            1 => 1.5,
            _ => 2.0, // Multiple stale signals = maximum defense
        }
    }

    /// Disable signals that depend on Binance/cross-venue data.
    /// Call this when no Binance feed is available for the asset (e.g. HIP-3 DEX tokens).
    /// Without this, `staleness_spread_multiplier()` permanently returns 2.0x
    /// because lead_lag and cross_venue never receive data to warm up.
    pub fn disable_binance_signals(&mut self) {
        self.config.use_lead_lag = false;
        self.config.use_cross_venue = false;
    }

    /// Get lag analyzer status for diagnostics.
    pub fn lag_analyzer_status(&self) -> LagAnalyzerStatus<'_> {
        let (optimal_lag_ms, mi_bits) = self
            .lag_analyzer
            .optimal_lag()
            .map_or((None, None), |(l, m)| (Some(l), Some(m)));
        LagAnalyzerStatus {
            is_ready: self.lag_analyzer.is_ready(),
            observation_counts: self.lag_analyzer.observation_counts(),
            optimal_lag_ms,
            mi_bits,
            last_signal: &self.last_lead_lag_signal,
            sample_timestamps: self.lag_analyzer.sample_timestamps(),
        }
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
        self.latest_hl_vpin = 0.0;
        self.latest_hl_vpin_velocity = 0.0;
        self.vpin_valid = false;
        self.buy_pressure = BuyPressureTracker::new();
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

    #[test]
    fn test_signal_contributions_recorded() {
        let integrator = SignalIntegrator::default_config();
        let signals = integrator.get_signals();

        // get_signals() should always populate the contribution record
        assert!(signals.signal_contributions.is_some());

        let contrib = signals.signal_contributions.unwrap();

        // With no data fed in, signals should be at their defaults
        assert!(!contrib.lead_lag_active);
        assert_eq!(contrib.lead_lag_skew_bps, 0.0);

        // Informed flow spread mult should be reasonable (0.9-1.0 range for no-data case)
        assert!(contrib.informed_flow_spread_mult >= 0.9);
        assert!(contrib.informed_flow_spread_mult <= 1.0);

        // Regime kappa should be active (default config enables it)
        assert!(contrib.regime_active);
        assert!(contrib.regime_kappa_effective > 0.0);

        // Cross-venue not active (valid) without sufficient data
        assert!(!contrib.cross_venue_active);
        assert!(contrib.cross_venue_spread_mult >= 1.0);

        // VPIN not active without data
        assert!(!contrib.vpin_active);
        assert_eq!(contrib.vpin_spread_mult, 1.0);

        // Buy pressure not active without warmup
        assert!(!contrib.buy_pressure_active);
        assert_eq!(contrib.buy_pressure_skew_bps, 0.0);
    }

    #[test]
    fn test_signal_contributions_none_for_default() {
        // Manually constructed IntegratedSignals should have None contributions
        let signals = IntegratedSignals::default();
        assert!(signals.signal_contributions.is_none());
    }

    #[test]
    fn test_staleness_spread_multiplier() {
        // All signals disabled => no staleness => 1.0
        let disabled = SignalIntegrator::new(SignalIntegratorConfig::disabled());
        assert!((disabled.staleness_spread_multiplier() - 1.0).abs() < f64::EPSILON);

        // Enable one signal that hasn't warmed up => 1.5x
        let mut config_one = SignalIntegratorConfig::disabled();
        config_one.use_lead_lag = true;
        let one_stale = SignalIntegrator::new(config_one);
        assert!((one_stale.staleness_spread_multiplier() - 1.5).abs() < f64::EPSILON);

        // Enable two signals that haven't warmed up => 2.0x
        let mut config_two = SignalIntegratorConfig::disabled();
        config_two.use_lead_lag = true;
        config_two.use_cross_venue = true;
        let two_stale = SignalIntegrator::new(config_two);
        assert!((two_stale.staleness_spread_multiplier() - 2.0).abs() < f64::EPSILON);

        // Enable all four tracked signals => 2.0x (capped)
        let mut config_all = SignalIntegratorConfig::disabled();
        config_all.use_lead_lag = true;
        config_all.use_cross_venue = true;
        config_all.use_informed_flow = true;
        config_all.use_regime_kappa = true;
        let all_stale = SignalIntegrator::new(config_all);
        assert!((all_stale.staleness_spread_multiplier() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_disable_binance_signals_removes_staleness() {
        // Use a config with ONLY Binance-dependent signals enabled.
        // This way disabling them should drop staleness from 2.0x to 1.0x.
        let mut config = SignalIntegratorConfig::disabled();
        config.use_lead_lag = true;
        config.use_cross_venue = true;
        let mut integrator = SignalIntegrator::new(config);
        let before = integrator.staleness_spread_multiplier();
        assert!(
            (before - 2.0).abs() < f64::EPSILON,
            "lead_lag + cross_venue should be stale without data: {before}"
        );

        integrator.disable_binance_signals();
        // After disabling, those two signals no longer count toward staleness.
        let after = integrator.staleness_spread_multiplier();
        assert!(
            (after - 1.0).abs() < f64::EPSILON,
            "staleness should be 1.0 after disabling Binance signals: {after}"
        );
    }

    #[test]
    fn test_staleness_only_counts_hl_native_signals_after_disable() {
        // Start from disabled config, enable only Binance-dependent signals
        let mut config = SignalIntegratorConfig::disabled();
        config.use_lead_lag = true;
        config.use_cross_venue = true;
        let mut integrator = SignalIntegrator::new(config);

        // Both stale => 2.0x
        assert!((integrator.staleness_spread_multiplier() - 2.0).abs() < f64::EPSILON);

        integrator.disable_binance_signals();

        // Now no enabled signals are stale => 1.0x
        assert!((integrator.staleness_spread_multiplier() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_disable_binance_signals_idempotent() {
        let mut integrator = SignalIntegrator::default_config();

        integrator.disable_binance_signals();
        let first = integrator.staleness_spread_multiplier();

        integrator.disable_binance_signals();
        let second = integrator.staleness_spread_multiplier();

        assert!(
            (first - second).abs() < f64::EPSILON,
            "calling disable_binance_signals twice should be idempotent"
        );
        assert!(!integrator.config.use_lead_lag);
        assert!(!integrator.config.use_cross_venue);
    }
}
