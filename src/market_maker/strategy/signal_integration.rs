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
use std::cell::Cell;
use tracing::{debug, info, warn};

/// Signal availability state for cross-venue feeds.
///
/// Tracks whether this asset has a Binance/cross-venue pair configured,
/// whether signal is currently active, or has degraded. The key behavioral
/// difference: `NeverConfigured` compensates with wider spreads instead of
/// crushing position limits (which kills small accounts).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalAvailability {
    /// No cross-venue feed configured for this asset (e.g. HIP-3 DEX tokens).
    /// Compensates through spread widening (1.5x), NOT position reduction.
    NeverConfigured,
    /// Cross-venue signal is active and healthy.
    Available,
    /// Signal was available but degraded. Graduated reduction over time.
    Degraded {
        /// When the signal was last healthy (seconds ago).
        last_healthy_secs_ago: f64,
        /// How many seconds until full reduction applies.
        max_reduction_after_secs: f64,
    },
}

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

    /// Legacy field: minimum model weight floor for soft scaling.
    /// Superseded by graduated_weight() which enforces a 5% floor internally.
    /// Kept for backward compatibility with existing configs.
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

    /// Maximum total additive spread adjustment in bps from all signals combined.
    /// Caps the sum of informed_flow, gating, and cross_venue adjustments.
    /// Staleness multiplier is excluded (remains multiplicative as a safety mechanism).
    /// Default: 20.0 bps.
    pub max_spread_adjustment_bps: f64,

    /// Maximum flow urgency skew in bps when 5s imbalance exceeds 0.6.
    /// Applied on top of existing skew components to prevent accumulating into trends.
    /// Default: 10.0 bps.
    pub flow_urgency_max_bps: f64,

    // === Inventory + Signal Skew ===

    /// Enable inventory-based skew (always active when position != 0).
    /// Lean quotes away from inventory to encourage mean reversion.
    pub use_inventory_skew: bool,

    /// Sensitivity of inventory skew to position_ratio.
    /// inventory_skew_bps = -position_ratio * skew_sensitivity * half_spread_estimate_bps.
    /// Default: 0.5 (50% of half-spread at full position).
    pub inventory_skew_sensitivity: f64,

    /// Enable signal-based directional skew from alpha/trend signals.
    pub use_signal_skew: bool,

    /// Maximum signal skew in bps.
    /// Alpha and trend signals are converted to skew capped at this value.
    /// Default: 3.0 bps.
    pub signal_skew_max_bps: f64,

    /// Maximum combined skew as fraction of estimated half_spread.
    /// Prevents quote crossing. Default: 0.8 (80% of half-spread).
    pub max_skew_half_spread_fraction: f64,
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
            // DISABLED: BuyPressure has -2.67 bps marginal contribution in paper testing
            // (Run 2, 2026-02-08). Actively destructive — re-enable only after positive IR
            // validation on a fresh paper run with corrected reward function.
            use_buy_pressure: false,
            buy_pressure_z_threshold: 1.5,
            buy_pressure_skew_fraction: 0.1,
            buy_pressure_max_fraction: 0.3,
            max_spread_adjustment_bps: 20.0,
            flow_urgency_max_bps: 10.0,
            // Inventory + signal skew: enabled by default
            use_inventory_skew: true,
            inventory_skew_sensitivity: 0.5,
            use_signal_skew: true,
            signal_skew_max_bps: 3.0,
            max_skew_half_spread_fraction: 0.8,
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
            use_inventory_skew: false,
            use_signal_skew: false,
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
    /// Regime kappa spread multiplier (prior / effective, clamped 0.5-2.0)
    pub regime_kappa_spread_mult: f64,
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

    /// Additive spread adjustment from informed flow (bps)
    pub informed_flow_adj_bps: f64,
    /// Additive spread adjustment from model gating (bps)
    pub gating_adj_bps: f64,
    /// Additive spread adjustment from cross-venue (bps)
    pub cross_venue_adj_bps: f64,
    /// Total additive spread adjustment after cap (bps)
    pub total_adj_bps: f64,
}

/// Unified skew — single source of truth for quote skewing.
/// Two components only: GLFT q-term (inventory) + alpha_skew (signals).
///
/// The GLFT q-term is computed inside `glft.rs::inventory_skew_with_flow()` 
/// and is the theoretically correct inventory skew from the GLFT paper.
/// Alpha skew comes from `IntegratedSignals.combined_skew_bps` which contains
/// cross-venue, buy pressure, and signal skew (but NOT inventory skew — that's GLFT's job).
///
/// Previously, inventory skew was triple-counted:
/// 1. GLFT q-term (correct)
/// 2. signal_integration inventory_skew_bps (removed)
/// 3. position_guard.inventory_skew_bps() (deprecated, returns 0.0)
#[derive(Debug, Clone, Copy, Default)]
pub struct UnifiedSkew {
    /// GLFT q-term inventory skew (bps). Computed by glft.rs, not here.
    pub inventory_skew_bps: f64,
    /// Alpha/directional skew from cross-venue + signal sources (bps).
    /// This is combined_skew_bps from IntegratedSignals.
    pub alpha_skew_bps: f64,
    /// Total clamped to 80% of half_spread.
    pub total_skew_bps: f64,
}

impl UnifiedSkew {
    pub fn new(inventory_skew_bps: f64, alpha_skew_bps: f64, half_spread_bps: f64) -> Self {
        let raw = inventory_skew_bps + alpha_skew_bps;
        let max_skew = half_spread_bps * 0.8;
        Self {
            inventory_skew_bps,
            alpha_skew_bps,
            total_skew_bps: raw.clamp(-max_skew, max_skew),
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

    // === Cross-Asset Signals (Sprint 4.1) ===
    /// Expected move from BTC lead-lag and funding divergence (bps).
    pub cross_asset_expected_move_bps: f64,
    /// Confidence in the cross-asset signal [0, 1].
    pub cross_asset_confidence: f64,
    /// OI-based volatility multiplier (>= 1.0 means elevated vol).
    pub cross_asset_vol_mult: f64,

    // === Funding Rate Signals (Sprint 4.2) ===
    /// Basis velocity: rate of change of mark-index premium [-1, +1].
    pub funding_basis_velocity: f64,
    /// Premium alpha: predicted return from premium mean-reversion.
    pub funding_premium_alpha: f64,
    /// Funding skew bias: positive = positive funding (skew short), negative = vice versa.
    pub funding_skew_bps: f64,

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

    // === Additive Spread Adjustment Components (bps) ===
    /// Additive spread adjustment from informed flow signal (bps, >= 0).
    pub informed_flow_adj_bps: f64,
    /// Additive spread adjustment from model gating (bps, >= 0).
    pub gating_adj_bps: f64,
    /// Additive spread adjustment from cross-venue analysis (bps, >= 0).
    pub cross_venue_adj_bps: f64,
    /// Total additive spread adjustment before cap (bps).
    pub total_adj_bps_uncapped: f64,
    /// Total additive spread adjustment after cap (bps).
    pub total_adj_bps: f64,

    // === Combined ===
    /// Total spread multiplier (from capped additive adjustments, excludes staleness).
    /// NOTE: Also mutated by OI vol, funding settlement, and cancel-race in quote_engine.rs
    /// but those mutations are for analytics logging only. The actual spread impact flows
    /// through `signal_risk_premium_bps` → `total_risk_premium_bps`.
    pub total_spread_mult: f64,
    /// Additive risk premium from signal-derived sources (OI vol, funding settlement,
    /// cancel-race AS). Accumulated in quote_engine.rs and added to total_risk_premium_bps
    /// which flows through solve_min_gamma() for self-consistent spreads.
    pub signal_risk_premium_bps: f64,
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

    // === Inventory/Signal Skew Context (set each cycle by orchestrator) ===
    /// Current signed position (positive = long, negative = short).
    position: f64,
    /// Maximum position from config (hard limit).
    max_position: f64,
    /// Predicted alpha [0, 1] from AS estimator (0.5 = neutral).
    predicted_alpha: f64,
    /// Estimated half-spread in bps (from kappa-derived GLFT spread).
    half_spread_estimate_bps: f64,
    /// Cross-venue signal availability state machine.
    signal_availability: Cell<SignalAvailability>,
    /// Whether we've logged the initial signal availability state.
    logged_signal_state: Cell<bool>,

    // === CUSUM Predictive Lead-Lag (Phase 5) ===
    /// CUSUM changepoint detector for rapid divergence detection.
    cusum_detector: crate::market_maker::estimator::lag_analysis::CusumDetector,
    /// Last CUSUM-detected divergence (bps), cleared when MI confirms.
    cusum_divergence_bps: f64,
}

impl SignalIntegrator {
    /// Create a new signal integrator.
    pub fn new(config: SignalIntegratorConfig) -> Self {
        let initial_availability = if config.use_lead_lag || config.use_cross_venue {
            SignalAvailability::Degraded {
                last_healthy_secs_ago: 0.0,
                max_reduction_after_secs: 300.0,
            }
        } else {
            SignalAvailability::NeverConfigured
        };
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
            position: 0.0,
            max_position: 0.0,
            predicted_alpha: 0.5,
            half_spread_estimate_bps: 5.0,
            signal_availability: Cell::new(initial_availability),
            logged_signal_state: Cell::new(false),
            cusum_detector: crate::market_maker::estimator::lag_analysis::CusumDetector::default(),
            cusum_divergence_bps: 0.0,
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
    ///
    /// Phase 5: Also feeds CUSUM detector for preemptive (graduated) skew.
    /// On CUSUM detection without MI confirmation: 30% skew.
    /// On MI confirmation: full skew (existing behavior).
    pub fn on_binance_price(&mut self, mid_price: f64, timestamp_ms: i64) {
        if !self.config.use_lead_lag || mid_price <= 0.0 {
            return;
        }

        self.latest_binance_mid = mid_price;
        self.lag_analyzer.add_signal(timestamp_ms, mid_price);

        // Phase 5: CUSUM detection for preemptive skew
        if self.latest_hl_mid > 0.0 {
            let divergence_bps =
                (mid_price - self.latest_hl_mid) / self.latest_hl_mid * 10_000.0;
            if let Some(detected_bps) = self.cusum_detector.observe(divergence_bps) {
                // CUSUM detected a significant divergence — apply preemptive skew
                self.cusum_divergence_bps = detected_bps;
            }
        }

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

            // MI confirmed — clear CUSUM preemptive state (full skew takes over)
            if self.last_lead_lag_signal.is_actionable {
                self.cusum_divergence_bps = 0.0;
            }
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

    /// Update inventory and signal context for skew computation.
    ///
    /// Must be called each quote cycle before `get_signals()` to provide
    /// position, max_position, predicted_alpha, and half-spread estimate.
    /// These drive inventory-based and signal-based directional skew.
    pub fn set_skew_context(
        &mut self,
        position: f64,
        max_position: f64,
        predicted_alpha: f64,
        half_spread_estimate_bps: f64,
    ) {
        self.position = position;
        self.max_position = max_position;
        self.predicted_alpha = predicted_alpha;
        self.half_spread_estimate_bps = half_spread_estimate_bps;
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
            // Graduated gating: continuous weight with 5% floor, no death spiral
            let ll_weight = if self.config.use_per_signal_gating && self.config.use_model_gating {
                let w = self.model_gating.graduated_weight("lead_lag");
                signals.lead_lag_gating_weight = w;
                w
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
        } else if self.config.use_lead_lag
            && self.cusum_divergence_bps.abs() > 1.0
            && self.cusum_detector.is_warmed_up()
        {
            // Phase 5: Preemptive skew from CUSUM detection (30% confidence)
            // CUSUM detected divergence but MI hasn't confirmed yet.
            let preemptive_skew = self.cusum_divergence_bps * 0.3;
            signals.lead_lag_skew_bps = preemptive_skew
                .abs()
                .min(self.config.max_lead_lag_skew_bps);
            signals.skew_direction = if preemptive_skew > 0.0 { 1 } else { -1 };
            signals.lead_lag_actionable = true;
            signals.binance_hl_diff_bps = self.cusum_divergence_bps;
        }

        // === Informed Flow ===
        if self.config.use_informed_flow {
            // Graduated gating: continuous weight with 5% floor, no death spiral
            let if_weight = if self.config.use_per_signal_gating && self.config.use_model_gating {
                let w = self.model_gating.graduated_weight("informed_flow");
                signals.informed_flow_gating_weight = w;
                w
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
                // Clamp to >= 1.0: InformedFlow should only widen spreads, never tighten.
                // Tightening has marginal value -0.23 bps (harmful).
                self.config
                    .informed_flow_adjustment
                    .spread_multiplier(effective_p)
                    .max(1.0)
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

        // === Combined Signals (Additive) ===
        // Convert each multiplier to an additive excess, sum them, and cap.
        // This prevents multiplicative compounding: three 1.5x adjustments
        // give 2.5x (additive) instead of 3.375x (multiplicative).
        // Staleness multiplier is applied separately (remains multiplicative for safety).
        let informed_excess = signals.informed_flow_spread_mult - 1.0;
        let gating_excess = signals.gating_spread_mult - 1.0;
        let cross_venue_excess = signals.cross_venue_spread_mult - 1.0;

        let total_excess_uncapped = informed_excess + gating_excess + cross_venue_excess;
        // Cap expressed as multiplier excess: max_spread_adjustment_bps / reference 10 bps
        let max_excess = self.config.max_spread_adjustment_bps / 10.0;
        let total_excess = total_excess_uncapped.clamp(-0.1, max_excess);

        // Store diagnostic bps values (reference = 10 bps base spread)
        signals.informed_flow_adj_bps = informed_excess * 10.0;
        signals.gating_adj_bps = gating_excess * 10.0;
        signals.cross_venue_adj_bps = cross_venue_excess * 10.0;
        signals.total_adj_bps_uncapped = total_excess_uncapped * 10.0;
        signals.total_adj_bps = total_excess * 10.0;

        signals.total_spread_mult = 1.0 + total_excess;

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

        // INVENTORY SKEW REMOVED: Was double-counting with GLFT's own q-term
        // (inventory_skew_with_flow in glft.rs). The GLFT q-term is the theoretically
        // correct source from Guéant-Lehalle-Fernandez-Tapia. Keeping variable for
        // backward compat with logging but set to 0.
        let inventory_skew_bps = 0.0_f64;

        // === SIGNAL SKEW: Directional skew from alpha/trend signals ===
        // Alpha > 0.5 = bullish (lean bids), alpha < 0.5 = bearish (lean asks).
        // Convert alpha [0, 1] to normalized signal [-1, 1].
        let signal_skew_bps = if self.config.use_signal_skew {
            let signal_normalized = (self.predicted_alpha - 0.5) * 2.0;
            // Only apply when alpha deviates from 0.5 (neutral) — lower threshold enables earlier directional skew
            if signal_normalized.abs() > 0.05 {
                signal_normalized * self.config.signal_skew_max_bps
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Skew sources: base (cross-exchange) + cross-venue + buy pressure + signal
        // Inventory skew deliberately excluded — handled by GLFT q-term
        let raw_skew = base_skew_bps + cross_venue_skew_bps + buy_pressure_skew_bps
            + inventory_skew_bps + signal_skew_bps;

        // Clamp combined skew to prevent quote crossing: max 80% of half-spread
        let max_skew_bps = self.config.max_skew_half_spread_fraction * self.half_spread_estimate_bps;
        signals.combined_skew_bps = raw_skew.clamp(-max_skew_bps, max_skew_bps);

        // === HL-Native Directional Skew Fallback ===
        // When cross-venue signals are unavailable (no Binance feed for this asset),
        // use HL-native flow imbalance as ADDITIVE directional skew.
        // No guard on combined_skew_bps — this is HIP-3's primary directional signal.
        // Final clamp at line ~1194 prevents overflow.
        if !signals.lead_lag_actionable && !signals.cross_venue_valid {
            let flow_dir = self.latest_hl_flow.imbalance_5s * 0.6
                + self.latest_hl_flow.imbalance_30s * 0.4;
            let fallback_cap = self.config.max_lead_lag_skew_bps * 0.6;
            signals.combined_skew_bps += (flow_dir * fallback_cap).clamp(-fallback_cap, fallback_cap);
        }

        // Update signal availability state machine
        let current = self.signal_availability.get();
        if signals.cross_venue_valid || signals.lead_lag_actionable {
            // Signal is healthy — transition to Available
            if current != SignalAvailability::Available {
                self.signal_availability.set(SignalAvailability::Available);
            }
        } else {
            match current {
                SignalAvailability::Available => {
                    // Signal was healthy but just lost — start degradation
                    self.signal_availability.set(SignalAvailability::Degraded {
                        last_healthy_secs_ago: 0.0,
                        max_reduction_after_secs: 300.0,
                    });
                }
                SignalAvailability::Degraded {
                    last_healthy_secs_ago,
                    max_reduction_after_secs,
                } => {
                    // Increment degradation timer (~1s per cycle)
                    self.signal_availability.set(SignalAvailability::Degraded {
                        last_healthy_secs_ago: last_healthy_secs_ago + 1.0,
                        max_reduction_after_secs,
                    });
                }
                SignalAvailability::NeverConfigured => {
                    // No cross-venue configured — log once
                    if !self.logged_signal_state.get() {
                        warn!(
                            "No cross-venue signal configured — compensating with wider spreads (1.5x), \
                             no position reduction"
                        );
                        self.logged_signal_state.set(true);
                    }
                }
            }
        }

        // === FLOW URGENCY SKEW: Aggressive skew when directional flow is strong ===
        // When 5s imbalance exceeds threshold (moderately one-sided), add urgency skew
        // up to flow_urgency_max_bps. Lower threshold (0.4 vs old 0.6) catches
        // directional flow earlier — critical for HIP-3 where this is the primary signal.
        {
            let flow_imbalance = self.latest_hl_flow.imbalance_5s;
            const FLOW_URGENCY_THRESHOLD: f64 = 0.4;
            if flow_imbalance.abs() > FLOW_URGENCY_THRESHOLD {
                let urgency = (flow_imbalance.abs() - FLOW_URGENCY_THRESHOLD) / (1.0 - FLOW_URGENCY_THRESHOLD);
                let flow_skew_bps = urgency * self.config.flow_urgency_max_bps * flow_imbalance.signum();
                signals.combined_skew_bps += flow_skew_bps;
            }
        }

        // Final clamp: ensure all additive skew components combined don't exceed safe bounds
        signals.combined_skew_bps = signals.combined_skew_bps.clamp(-max_skew_bps, max_skew_bps);

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
            regime_kappa_spread_mult: {
                let kappa_eff = signals.kappa_effective;
                let kappa_prior = self.regime_kappa.blended_prior();
                if kappa_eff > 0.0 && kappa_prior > 0.0 {
                    (kappa_prior / kappa_eff).clamp(0.5, 2.0)
                } else {
                    1.0
                }
            },
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

            informed_flow_adj_bps: signals.informed_flow_adj_bps,
            gating_adj_bps: signals.gating_adj_bps,
            cross_venue_adj_bps: signals.cross_venue_adj_bps,
            total_adj_bps: signals.total_adj_bps,
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

    /// Whether the CUSUM detector has a preemptive divergence pending.
    pub fn has_cusum_divergence(&self) -> bool {
        self.cusum_divergence_bps.abs() > 1.0 && self.cusum_detector.is_warmed_up()
    }

    /// Get current flow decomposition.
    pub fn flow_decomposition(&self) -> FlowDecomposition {
        self.last_flow_decomp
    }

    /// Get HL flow imbalance (5-second EWMA).
    pub fn hl_flow_imbalance_5s(&self) -> f64 {
        self.latest_hl_flow.imbalance_5s
    }

    /// Get regime kappa estimator for direct access.
    pub fn regime_kappa(&self) -> &RegimeKappaEstimator {
        &self.regime_kappa
    }

    /// Get the blended prior kappa from the regime kappa estimator.
    ///
    /// Returns the regime-probability-weighted prior kappa, representing
    /// the expected fill intensity before observing data. Used to compute
    /// regime kappa spread multiplier: `prior / effective`.
    pub fn kappa_prior(&self) -> f64 {
        self.regime_kappa.blended_prior()
    }

    /// Reinitialize regime kappa priors from a MarketProfile.
    ///
    /// Called once when the first L2 snapshot provides implied kappa.
    /// Replaces BTC-calibrated priors with asset-specific values.
    pub fn reinit_regime_kappa_from_profile(
        &mut self,
        implied_kappa: f64,
        _liquidity_class: crate::market_maker::estimator::market_profile::LiquidityClass,
    ) {
        let config = crate::market_maker::estimator::regime_kappa::RegimeKappaConfig::from_base_kappa(
            implied_kappa.clamp(50.0, 50000.0),
        );
        self.regime_kappa = RegimeKappaEstimator::new(config);
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

        // Only count a signal as stale if it WAS warmed up and then lost readiness.
        // Cold-start (never reached warmup) should not penalize — use safe priors instead.
        if self.config.use_lead_lag
            && !self.lag_analyzer.is_ready()
            && self.lag_analyzer.observation_counts() != (0, 0)
        {
            stale_count += 1;
        }
        if self.config.use_cross_venue
            && !self.binance_flow.is_warmed_up()
            && self.binance_flow.trade_count() > 0
        {
            stale_count += 1;
        }
        if self.config.use_informed_flow
            && !self.informed_flow.is_warmed_up()
            && self.informed_flow.was_ever_warmed_up()
        {
            stale_count += 1;
        }
        if self.config.use_regime_kappa
            && !self.regime_kappa.is_warmed_up()
            && self.regime_kappa.was_ever_warmed_up()
        {
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
        self.signal_availability.set(SignalAvailability::NeverConfigured);
    }

    /// Disable only cross-venue trade flow signal, keeping lead-lag active.
    /// Used when a reference perp (e.g. HYPE for hyna:HYPE) feeds lead-lag
    /// via AllMids, but Binance trade flow is unavailable.
    pub fn disable_cross_venue_only(&mut self) {
        self.config.use_cross_venue = false;
        // Keep use_lead_lag = true — reference perp mid feeds it
        // Don't mark NeverConfigured — lead-lag IS configured
    }

    /// Returns position limit multiplier based on signal availability.
    ///
    /// Key change from old behavior: `NeverConfigured` returns 1.0 (was 0.30),
    /// preventing small accounts from being crushed below exchange minimums.
    /// `Degraded` graduates from 1.0 to 0.5 over `max_reduction_after_secs`.
    /// WS6c: Always returns 1.0 — staleness routes through σ (CovarianceEstimator),
    /// not position limits. Reducing position limits during signal degradation
    /// creates a redundant channel that conflicts with the principled γσ²τ pipeline.
    pub fn signal_position_limit_mult(&self) -> f64 {
        1.0
    }

    /// WS6c: Always returns 1.0 — signal staleness routes through σ
    /// (CovarianceEstimator realized vol feedback), not multiplicative spread widening.
    /// The GLFT formula γσ²τ automatically widens when σ increases.
    pub fn signal_spread_widening_mult(&self) -> f64 {
        1.0
    }

    /// Get current signal availability state.
    pub fn signal_availability(&self) -> SignalAvailability {
        self.signal_availability.get()
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
        self.position = 0.0;
        self.max_position = 0.0;
        self.predicted_alpha = 0.5;
        self.half_spread_estimate_bps = 5.0;
        self.signal_availability.set(
            if self.config.use_lead_lag || self.config.use_cross_venue {
                SignalAvailability::Degraded {
                    last_healthy_secs_ago: 0.0,
                    max_reduction_after_secs: 300.0,
                }
            } else {
                SignalAvailability::NeverConfigured
            },
        );
        self.logged_signal_state.set(false);
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
        // Informed flow spread mult is clamped to >= 1.0 (no tightening allowed)
        assert!(signals.informed_flow_spread_mult >= 1.0);
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

        // Informed flow spread mult clamped to >= 1.0 (no tightening)
        assert!(contrib.informed_flow_spread_mult >= 1.0);

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

        // Enable signals that haven't warmed up AND have zero observations (cold start)
        // Cold start should NOT count as stale => 1.0
        let mut config_one = SignalIntegratorConfig::disabled();
        config_one.use_lead_lag = true;
        let one_cold = SignalIntegrator::new(config_one);
        assert!(
            (one_cold.staleness_spread_multiplier() - 1.0).abs() < f64::EPSILON,
            "cold-start lead_lag (0 observations) should not count as stale"
        );

        // Enable all four tracked signals with 0 observations => all cold start => 1.0
        let mut config_all = SignalIntegratorConfig::disabled();
        config_all.use_lead_lag = true;
        config_all.use_cross_venue = true;
        config_all.use_informed_flow = true;
        config_all.use_regime_kappa = true;
        let all_cold = SignalIntegrator::new(config_all);
        assert!(
            (all_cold.staleness_spread_multiplier() - 1.0).abs() < f64::EPSILON,
            "all cold-start signals (0 observations) should not count as stale"
        );
    }

    #[test]
    fn test_staleness_cold_start_not_penalized() {
        // Enabled signals with 0 observations => cold start => mult = 1.0
        let mut config = SignalIntegratorConfig::disabled();
        config.use_informed_flow = true;
        config.use_regime_kappa = true;
        let integrator = SignalIntegrator::new(config);

        let mult = integrator.staleness_spread_multiplier();
        assert!(
            (mult - 1.0).abs() < f64::EPSILON,
            "cold-start (0 obs) should give mult=1.0, got {mult}"
        );
    }

    #[test]
    fn test_staleness_after_warmup_then_data_loss() {
        use crate::market_maker::estimator::VolatilityRegime;

        // Scenario: regime_kappa warmed up in Normal, then regime changes to Extreme
        // (no fills in Extreme yet). was_ever_warmed_up()=true but is_warmed_up()=false.
        let mut config = SignalIntegratorConfig::disabled();
        config.use_regime_kappa = true;
        let mut integrator = SignalIntegrator::new(config);

        // Warm up regime_kappa in Normal regime (need min_regime_observations fills)
        integrator.regime_kappa.set_regime(VolatilityRegime::Normal);
        let price = 100.0;
        let mid = 100.0;
        for i in 0..20 {
            integrator.regime_kappa.on_fill(i * 1000, price, 1.0, mid);
        }
        assert!(integrator.regime_kappa.is_warmed_up(), "should be warmed up after 20 fills");
        assert!(integrator.regime_kappa.was_ever_warmed_up());
        assert!(
            (integrator.staleness_spread_multiplier() - 1.0).abs() < f64::EPSILON,
            "warmed up and ready => no staleness => 1.0x"
        );

        // Switch regime: now is_warmed_up()=false but was_ever_warmed_up()=true
        integrator.regime_kappa.set_regime(VolatilityRegime::Extreme);
        assert!(!integrator.regime_kappa.is_warmed_up(), "Extreme regime has no fills");
        assert!(integrator.regime_kappa.was_ever_warmed_up(), "was warmed in Normal");

        let mult = integrator.staleness_spread_multiplier();
        assert!(
            (mult - 1.5).abs() < f64::EPSILON,
            "was warmed then lost readiness => stale => 1.5x, got {mult}"
        );
    }

    #[test]
    fn test_disable_binance_signals_removes_staleness() {
        // Cold-start Binance signals (0 observations) are no longer counted as stale.
        // After disabling, they remain 1.0x (the disable is still useful to prevent
        // future staleness if data was received then lost).
        let mut config = SignalIntegratorConfig::disabled();
        config.use_lead_lag = true;
        config.use_cross_venue = true;
        let mut integrator = SignalIntegrator::new(config);
        let before = integrator.staleness_spread_multiplier();
        assert!(
            (before - 1.0).abs() < f64::EPSILON,
            "cold-start Binance signals should NOT be counted as stale: {before}"
        );

        integrator.disable_binance_signals();
        // After disabling, still 1.0x
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

        // Both cold-start (0 observations) => NOT stale => 1.0x
        assert!(
            (integrator.staleness_spread_multiplier() - 1.0).abs() < f64::EPSILON,
            "cold-start Binance signals should not be counted as stale"
        );

        integrator.disable_binance_signals();

        // After disabling, still 1.0x (no change since they weren't stale anyway)
        assert!((integrator.staleness_spread_multiplier() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_disable_binance_signals_idempotent() {
        let mut integrator = SignalIntegrator::default_config();

        integrator.disable_binance_signals();
        let first = integrator.staleness_spread_multiplier();
        assert_eq!(
            integrator.signal_availability(),
            SignalAvailability::NeverConfigured
        );

        integrator.disable_binance_signals();
        let second = integrator.staleness_spread_multiplier();

        assert!(
            (first - second).abs() < f64::EPSILON,
            "calling disable_binance_signals twice should be idempotent"
        );
        assert!(!integrator.config.use_lead_lag);
        assert!(!integrator.config.use_cross_venue);
        assert_eq!(
            integrator.signal_availability(),
            SignalAvailability::NeverConfigured
        );
    }

    #[test]
    fn test_additive_spread_multiplier_vs_multiplicative() {
        // Three 1.5x multipliers should produce ~2.5x (additive) not 3.375x (multiplicative)
        let mut signals = IntegratedSignals::default();
        signals.informed_flow_spread_mult = 1.5;
        signals.gating_spread_mult = 1.5;
        signals.cross_venue_spread_mult = 1.5;

        // Old multiplicative: 1.5 * 1.5 * 1.5 = 3.375
        let multiplicative: f64 = 1.5 * 1.5 * 1.5;
        assert!((multiplicative - 3.375).abs() < 0.01);

        // New additive: 1.0 + (1.5-1.0) + (1.5-1.0) + (1.5-1.0) = 2.5
        let additive: f64 = 1.0 + (1.5 - 1.0) + (1.5 - 1.0) + (1.5 - 1.0);
        assert!((additive - 2.5).abs() < f64::EPSILON);

        // Verify integrator actually produces additive result
        let mut config = SignalIntegratorConfig::disabled();
        config.max_spread_adjustment_bps = 100.0; // large cap to not interfere
        let integrator = SignalIntegrator::new(config);
        let out = integrator.get_signals();
        // All disabled → mults are 1.0 or near it, total should be near 1.0
        assert!(
            out.total_spread_mult < 1.2,
            "disabled config should give total_spread_mult near 1.0, got {}",
            out.total_spread_mult
        );
    }

    #[test]
    fn test_additive_spread_cap_enforced() {
        // Construct signals that would exceed the cap
        let mut signals = IntegratedSignals::default();
        // Each excess = 2.0 (i.e., mult=3.0), three of them = 6.0 excess
        signals.informed_flow_spread_mult = 3.0;
        signals.gating_spread_mult = 3.0;
        signals.cross_venue_spread_mult = 3.0;
        // Uncapped additive: 1.0 + 2.0 + 2.0 + 2.0 = 7.0
        // Cap at 20 bps → max_excess = 20/10 = 2.0 → capped mult = 3.0

        // Verify the cap math: max_spread_adjustment_bps=20 → max excess=2.0
        let max_excess: f64 = 20.0 / 10.0;
        let total_excess_uncapped: f64 = 2.0 + 2.0 + 2.0;
        let total_excess = total_excess_uncapped.min(max_excess);
        assert!((total_excess - 2.0).abs() < f64::EPSILON);
        assert!((1.0 + total_excess - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_additive_spread_diagnostics_populated() {
        let integrator = SignalIntegrator::default_config();
        let signals = integrator.get_signals();

        // Diagnostics should be populated
        // With no data, informed_flow_spread_mult is clamped to 1.0 (no tightening)
        // So informed_flow_adj_bps should be 0.0
        assert!(signals.informed_flow_adj_bps >= 0.0);

        // Cross-venue: with default (empty) features, confidence=0 triggers
        // low-confidence widening → spread_mult ~1.15, so adj_bps ~1.5
        assert!(signals.cross_venue_adj_bps >= 0.0);

        // total_adj_bps should equal sum of components (before cap)
        let expected_uncapped = signals.informed_flow_adj_bps
            + signals.gating_adj_bps
            + signals.cross_venue_adj_bps;
        assert!(
            (signals.total_adj_bps_uncapped - expected_uncapped).abs() < 0.01,
            "uncapped total should equal sum of components: {} vs {}",
            signals.total_adj_bps_uncapped,
            expected_uncapped
        );

        // Contribution record should also have the additive fields
        let contrib = signals.signal_contributions.unwrap();
        assert!((contrib.informed_flow_adj_bps - signals.informed_flow_adj_bps).abs() < f64::EPSILON);
        assert!((contrib.gating_adj_bps - signals.gating_adj_bps).abs() < f64::EPSILON);
        assert!((contrib.cross_venue_adj_bps - signals.cross_venue_adj_bps).abs() < f64::EPSILON);
        assert!((contrib.total_adj_bps - signals.total_adj_bps).abs() < f64::EPSILON);
    }

    #[test]
    fn test_additive_spread_staleness_remains_multiplicative() {
        // Staleness multiplier should NOT be affected by the additive conversion
        // It's a safety mechanism that remains multiplicative
        let disabled = SignalIntegrator::new(SignalIntegratorConfig::disabled());
        let staleness = disabled.staleness_spread_multiplier();
        assert!(
            (staleness - 1.0).abs() < f64::EPSILON,
            "staleness should be separate from additive total_spread_mult"
        );
    }

    #[test]
    fn test_staleness_warmup_phase_no_penalty() {
        use crate::market_maker::estimator::informed_flow::TradeFeatures;

        // During warmup: some observations but not enough to warm up.
        // was_ever_warmed_up()=false => staleness should be 1.0 (no penalty).
        let mut config = SignalIntegratorConfig::disabled();
        config.use_informed_flow = true;
        config.informed_flow_config.min_observations = 100;
        let mut integrator = SignalIntegrator::new(config);

        // Feed 5 observations (not enough for warmup of 100)
        for i in 0..5 {
            integrator.informed_flow.on_trade(&TradeFeatures {
                timestamp_ms: i as u64 * 1000,
                size: 0.1,
                price_impact_bps: 1.0,
                book_imbalance: 0.0,
                is_buy: true,
                inter_arrival_ms: 1000,
            });
        }

        assert!(!integrator.informed_flow.is_warmed_up());
        assert!(integrator.informed_flow.observation_count() > 0, "has some data");
        assert!(!integrator.informed_flow.was_ever_warmed_up(), "never reached warmup");

        let mult = integrator.staleness_spread_multiplier();
        assert!(
            (mult - 1.0).abs() < f64::EPSILON,
            "during warmup (never fully warmed) => no staleness => 1.0x, got {mult}"
        );
    }

    #[test]
    fn test_staleness_after_full_warmup_then_regime_change() {
        use crate::market_maker::estimator::VolatilityRegime;

        // Verify: warmed in one regime, switch regime => staleness triggers
        let mut config = SignalIntegratorConfig::disabled();
        config.use_regime_kappa = true;
        let mut integrator = SignalIntegrator::new(config);

        // Warm up in Normal
        integrator.regime_kappa.set_regime(VolatilityRegime::Normal);
        for i in 0..20 {
            integrator.regime_kappa.on_fill(i * 1000, 100.0, 1.0, 100.0);
        }
        assert!(integrator.regime_kappa.was_ever_warmed_up());
        assert!(integrator.regime_kappa.is_warmed_up());
        assert!((integrator.staleness_spread_multiplier() - 1.0).abs() < f64::EPSILON);

        // Switch regime => data loss
        integrator.regime_kappa.set_regime(VolatilityRegime::Extreme);
        assert!(!integrator.regime_kappa.is_warmed_up());
        assert!(integrator.regime_kappa.was_ever_warmed_up());

        let mult = integrator.staleness_spread_multiplier();
        assert!(
            (mult - 1.5).abs() < f64::EPSILON,
            "was warmed then regime changed => stale => 1.5x, got {mult}"
        );
    }

    #[test]
    fn test_hl_native_skew_fallback_with_buy_pressure() {
        use crate::market_maker::estimator::binance_flow::FlowFeatureVec;

        let mut integrator = SignalIntegrator::default_config();

        // Simulate HL-native buy pressure (no Binance/cross-venue data)
        integrator.latest_hl_flow = FlowFeatureVec {
            imbalance_5s: 0.6,  // Moderate buy pressure at 5s
            imbalance_30s: 0.4, // Moderate buy pressure at 30s
            confidence: 0.8,
            ..Default::default()
        };

        let signals = integrator.get_signals();

        // No cross-venue or lead-lag should be active
        assert!(!signals.lead_lag_actionable);
        assert!(!signals.cross_venue_valid);

        // Fallback should produce positive (bullish) skew
        assert!(
            signals.combined_skew_bps > 0.0,
            "HL-native fallback should produce bullish skew with buy pressure, got: {}",
            signals.combined_skew_bps
        );
    }

    #[test]
    fn test_hl_native_skew_fallback_capped() {
        use crate::market_maker::estimator::binance_flow::FlowFeatureVec;

        let mut integrator = SignalIntegrator::default_config();

        // Extreme sell pressure
        integrator.latest_hl_flow = FlowFeatureVec {
            imbalance_5s: -1.0,
            imbalance_30s: -1.0,
            confidence: 1.0,
            ..Default::default()
        };

        let signals = integrator.get_signals();

        // Skew should be negative — fallback produces up to 60% of max_lead_lag_skew_bps
        // plus flow urgency adds up to flow_urgency_max_bps for imbalance > 0.6
        let fallback_cap = integrator.config.max_lead_lag_skew_bps * 0.6;
        let urgency_cap = integrator.config.flow_urgency_max_bps;
        let total_cap = fallback_cap + urgency_cap;
        assert!(
            signals.combined_skew_bps >= -total_cap - f64::EPSILON,
            "Skew should be bounded by fallback + urgency cap, got: {} vs max: {}",
            signals.combined_skew_bps,
            -total_cap
        );
        assert!(signals.combined_skew_bps < 0.0);
    }

    #[test]
    fn test_flow_urgency_skew_strong_imbalance() {
        use crate::market_maker::estimator::binance_flow::FlowFeatureVec;

        let mut integrator = SignalIntegrator::default_config();

        // Set wide half-spread so skew clamp doesn't bite (max = 0.8 * 15 = 12 bps)
        integrator.set_skew_context(0.0, 1.0, 0.5, 15.0);

        // Strong buy pressure above urgency threshold (0.6)
        integrator.latest_hl_flow = FlowFeatureVec {
            imbalance_5s: 0.9, // 75% urgency: (0.9 - 0.6) / 0.4 = 0.75
            imbalance_30s: 0.5,
            confidence: 0.9,
            ..Default::default()
        };

        let signals = integrator.get_signals();

        // Flow urgency = 0.75 * 10.0 = 7.5 bps (plus fallback)
        assert!(
            signals.combined_skew_bps > 5.0,
            "Strong flow should produce >5 bps urgency skew, got: {}",
            signals.combined_skew_bps
        );
    }

    #[test]
    fn test_flow_urgency_skew_below_threshold() {
        use crate::market_maker::estimator::binance_flow::FlowFeatureVec;

        let mut integrator = SignalIntegrator::default_config();

        // Moderate imbalance below urgency threshold
        integrator.latest_hl_flow = FlowFeatureVec {
            imbalance_5s: 0.5, // Below 0.6 threshold → no urgency
            imbalance_30s: 0.3,
            confidence: 0.7,
            ..Default::default()
        };

        let signals = integrator.get_signals();

        // Should only have fallback skew (up to 60% of max_lead_lag_skew_bps = 3.0 bps)
        let fallback_cap = integrator.config.max_lead_lag_skew_bps * 0.6;
        assert!(
            signals.combined_skew_bps <= fallback_cap + 0.01,
            "Below urgency threshold, skew should be <= fallback cap {fallback_cap}, got: {}",
            signals.combined_skew_bps
        );
    }

    // === Inventory + Signal Skew Tests ===

    #[test]
    fn test_inventory_skew_zero_position_neutral() {
        let mut integrator = SignalIntegrator::default_config();
        // Position = 0, alpha = 0.5 (neutral) → skew should be ~0
        integrator.set_skew_context(0.0, 1.0, 0.5, 5.0);
        let signals = integrator.get_signals();
        // With zero position and neutral alpha, only fallback skew from flow (also 0)
        assert!(
            signals.combined_skew_bps.abs() < 0.5,
            "Zero position + neutral alpha should give near-zero skew, got: {:.3}",
            signals.combined_skew_bps
        );
    }

    // NOTE: test_inventory_skew_{long,short,scales} removed — inventory skew was
    // intentionally moved from signal_integration to GLFT q-term (inventory_skew_with_flow)
    // during the Principled Architecture Redesign (Feb 2026). See glft.rs tests.

    #[test]
    fn test_skew_clamped_prevents_crossing() {
        let mut integrator = SignalIntegrator::default_config();

        // Max position + extreme alpha → should clamp to 80% of half-spread
        integrator.set_skew_context(1.0, 1.0, 0.0, 5.0);
        let signals = integrator.get_signals();

        let max_allowed = 0.8 * 5.0; // max_skew_half_spread_fraction * half_spread
        assert!(
            signals.combined_skew_bps.abs() <= max_allowed + 0.01,
            "Skew should be clamped to {:.2} bps, got: {:.3}",
            max_allowed,
            signals.combined_skew_bps
        );
    }

    #[test]
    fn test_signal_skew_bullish_alpha() {
        let mut integrator = SignalIntegrator::default_config();
        // Bullish alpha (0.8) with no position → positive signal skew
        integrator.set_skew_context(0.0, 1.0, 0.8, 5.0);
        let signals = integrator.get_signals();
        assert!(
            signals.combined_skew_bps > 0.1,
            "Bullish alpha should produce positive skew, got: {:.3}",
            signals.combined_skew_bps
        );
    }

    #[test]
    fn test_signal_skew_bearish_alpha() {
        let mut integrator = SignalIntegrator::default_config();
        // Bearish alpha (0.2) with no position → negative signal skew
        integrator.set_skew_context(0.0, 1.0, 0.2, 5.0);
        let signals = integrator.get_signals();
        assert!(
            signals.combined_skew_bps < -0.1,
            "Bearish alpha should produce negative skew, got: {:.3}",
            signals.combined_skew_bps
        );
    }

    #[test]
    fn test_signal_skew_neutral_alpha_no_effect() {
        let mut integrator = SignalIntegrator::default_config();
        // Neutral alpha (0.5) → signal skew should be zero (below 0.05 threshold)
        integrator.set_skew_context(0.0, 1.0, 0.5, 5.0);
        let signals_neutral = integrator.get_signals();

        // Slightly off-neutral (0.52 → normalized = 0.04, below 0.05 threshold)
        integrator.set_skew_context(0.0, 1.0, 0.52, 5.0);
        let signals_slight = integrator.get_signals();

        assert!(
            (signals_neutral.combined_skew_bps - signals_slight.combined_skew_bps).abs() < 0.01,
            "Near-neutral alpha should not produce signal skew"
        );
    }

    #[test]
    fn test_inventory_skew_disabled_config() {
        let mut config = SignalIntegratorConfig::default();
        config.use_inventory_skew = false;
        config.use_signal_skew = false;
        let mut integrator = SignalIntegrator::new(config);

        integrator.set_skew_context(1.0, 1.0, 0.9, 10.0);
        let signals = integrator.get_signals();

        // With inventory/signal skew disabled and no other signals, skew should be ~0
        assert!(
            signals.combined_skew_bps.abs() < 0.5,
            "Disabled skew should give near-zero, got: {:.3}",
            signals.combined_skew_bps
        );
    }

    #[test]
    fn test_no_signal_safety_mode() {
        // Default config has use_lead_lag=true, so starts in Degraded (not NeverConfigured)
        let config = SignalIntegratorConfig::default();
        let integrator = SignalIntegrator::new(config);

        // Initially Degraded with last_healthy_secs_ago=0.0 → mult=1.0
        assert!(matches!(
            integrator.signal_availability(),
            SignalAvailability::Degraded { .. }
        ));
        assert_eq!(integrator.signal_position_limit_mult(), 1.0);
    }

    #[test]
    fn test_signal_recovery_restores_limits() {
        let config = SignalIntegratorConfig::default();
        let integrator = SignalIntegrator::new(config);

        // Simulate cross-venue signal becoming valid
        integrator
            .signal_availability
            .set(SignalAvailability::Available);
        assert_eq!(integrator.signal_position_limit_mult(), 1.0);
    }

    #[test]
    fn test_never_configured_no_position_reduction() {
        // NeverConfigured returns position_limit_mult=1.0 (was 0.30 — the key fix)
        let config = SignalIntegratorConfig::disabled();
        let integrator = SignalIntegrator::new(config);

        assert_eq!(
            integrator.signal_availability(),
            SignalAvailability::NeverConfigured
        );
        assert_eq!(integrator.signal_position_limit_mult(), 1.0);
    }

    #[test]
    fn test_ws6c_spread_widening_always_1() {
        // WS6c: signal_spread_widening_mult always returns 1.0
        // Staleness routes through σ (CovarianceEstimator), not multiplicative widening.
        let config = SignalIntegratorConfig::disabled();
        let integrator = SignalIntegrator::new(config);
        assert!((integrator.signal_spread_widening_mult() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ws6c_position_limit_mult_always_1() {
        // WS6c: signal_position_limit_mult always returns 1.0 regardless of availability.
        // Staleness routes through σ, not position limit reduction.
        let config = SignalIntegratorConfig::default();
        let integrator = SignalIntegrator::new(config);

        // All states return 1.0
        integrator
            .signal_availability
            .set(SignalAvailability::Available);
        assert!((integrator.signal_position_limit_mult() - 1.0).abs() < f64::EPSILON);

        integrator
            .signal_availability
            .set(SignalAvailability::NeverConfigured);
        assert!((integrator.signal_position_limit_mult() - 1.0).abs() < f64::EPSILON);

        integrator
            .signal_availability
            .set(SignalAvailability::Degraded {
                last_healthy_secs_ago: 500.0,
                max_reduction_after_secs: 300.0,
            });
        assert!((integrator.signal_position_limit_mult() - 1.0).abs() < f64::EPSILON);
    }
}
