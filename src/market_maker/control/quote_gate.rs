//! Quote Gate: Determines WHETHER to quote based on directional edge.
//!
//! ## The Problem
//!
//! Traditional market making quotes both sides symmetrically, hoping to capture spread.
//! When flow_imbalance ≈ 0 (no directional edge), we get filled randomly and whipsaw:
//! - Get filled SHORT when momentum is UP → lose money
//! - Have to flip position → more losses
//!
//! ## The Solution
//!
//! The Quote Gate decides WHAT to quote based on:
//! 1. Do we have directional edge? (|flow_imbalance| > threshold)
//! 2. What is our current position?
//!
//! Decision matrix:
//! - Have edge + flat → Quote both sides with directional skew
//! - Have edge + position aligned → Quote both sides
//! - Have edge + position opposed → Only quote to reduce (urgent)
//! - No edge + flat → DON'T QUOTE (wait for signal)
//! - No edge + position → Only quote to reduce
//!
//! ## Key Insight
//!
//! "Being the informed flow" means:
//! 1. Have a directional view BEFORE trading
//! 2. Only quote the side that benefits from that view
//! 3. When view is uncertain, DON'T TRADE (not just "widen spreads")

use tracing::{debug, info, warn};

use super::bayesian_bootstrap::{BayesianBootstrapConfig, BayesianBootstrapTracker, BayesianExitDecision};
use super::calibrated_edge::{BayesianDecision, CalibratedEdgeSignal};
use super::position_pnl_tracker::PositionPnLTracker;
use super::theoretical_edge::TheoreticalEdgeEstimator;

/// Quote decision from the Quote Gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuoteDecision {
    /// Quote both sides normally (have directional edge)
    QuoteBoth,

    /// Quote only bids - want to get filled buying
    /// Used when: short position without edge (reduce), or bearish edge
    QuoteOnlyBids {
        /// Urgency level [0, 1] - higher = more aggressive pricing
        urgency: f64,
    },

    /// Quote only asks - want to get filled selling
    /// Used when: long position without edge (reduce), or bullish edge
    QuoteOnlyAsks {
        /// Urgency level [0, 1] - higher = more aggressive pricing
        urgency: f64,
    },

    /// Don't quote at all - wait for signal
    NoQuote {
        /// Reason for not quoting
        reason: NoQuoteReason,
    },
}

/// Reason for not quoting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoQuoteReason {
    /// System is warming up
    Warmup,
    /// No directional edge and flat position
    NoEdgeFlat,
    /// Cascade detected
    Cascade,
    /// Manual override
    Manual,
}

impl std::fmt::Display for NoQuoteReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NoQuoteReason::Warmup => write!(f, "warmup"),
            NoQuoteReason::NoEdgeFlat => write!(f, "no_edge_flat"),
            NoQuoteReason::Cascade => write!(f, "cascade"),
            NoQuoteReason::Manual => write!(f, "manual"),
        }
    }
}

/// Configuration for the Quote Gate.
#[derive(Debug, Clone)]
pub struct QuoteGateConfig {
    /// Whether the Quote Gate is enabled.
    /// When disabled, always returns QuoteBoth (legacy behavior).
    pub enabled: bool,

    /// Minimum |flow_imbalance| to have directional edge.
    /// Below this threshold, we consider ourselves "edgeless".
    /// Default: 0.15 (market-making appropriate - quote unless very noisy)
    pub min_edge_signal: f64,

    /// Minimum momentum confidence to trust the edge signal.
    /// This is ONLY required when signal is weak (below strong_signal_threshold).
    /// Strong signals override this requirement.
    /// Default: 0.45 (below baseline 0.50, allowing quoting in neutral conditions)
    pub min_edge_confidence: f64,

    /// Signal strength that bypasses confidence requirement.
    /// If |flow_imbalance| >= this, we trust it regardless of confidence.
    /// Default: 0.50 (strong signal = trust it)
    pub strong_signal_threshold: f64,

    /// Minimum position (as fraction of max) to trigger one-sided quoting.
    /// Below this, position is considered "flat".
    /// Default: 0.05 (5% of max position)
    pub position_threshold: f64,

    /// Maximum position (as fraction of max) before ONLY reducing.
    /// Above this, we become very defensive.
    /// Default: 0.7 (70% of max position)
    pub max_position_before_reduce_only: f64,

    /// Enable cascade protection (pull all quotes during cascade).
    /// Default: true
    pub cascade_protection: bool,

    /// Cascade threshold (cascade_size_factor below this = cascade).
    /// Default: 0.3 (70% cascade severity)
    pub cascade_threshold: f64,

    /// Quote both sides when flat, even without strong edge signal.
    /// Market makers profit from spread capture, not direction.
    /// Only disable quoting during genuine danger (cascade, toxic regime).
    /// Default: true (market-making mode)
    pub quote_flat_without_edge: bool,

    /// Use Bayesian IR warmup instead of fixed sample count.
    /// When true, uses P(IR > 1.0 | data) > tiered_threshold.
    /// Default: true
    pub use_bayesian_warmup: bool,

    /// Minimum IR outcomes (not predictions) to trust is_useful().
    /// On illiquid assets, we may have 100+ predictions but <10 outcomes
    /// because price rarely moves enough to record a result.
    /// Default: 25
    pub min_ir_outcomes_for_trust: u64,

    /// Configuration for active probing (Phase 3).
    pub probe_config: ProbeConfig,

    /// Configuration for Bayesian bootstrap tracking.
    pub bootstrap_config: BayesianBootstrapConfig,
}

/// Configuration for active probing to generate learning data.
#[derive(Debug, Clone)]
pub struct ProbeConfig {
    /// Enable probe mode.
    /// Allows quoting with negative expected edge if information value is high.
    pub enabled: bool,

    /// Minimum alpha uncertainty (std dev) to trigger probing.
    /// Only probe if we are uncertain about the alpha.
    /// Default: 0.1
    pub min_uncertainty: f64,

    /// Maximum fill rate (fills/hour) to consider probing.
    /// Only probe if we are not getting enough fills naturally.
    /// Default: 10.0
    pub max_fill_rate: f64,

    /// Information value bonus in basis points.
    /// effective_edge = expected_edge + info_value_bps
    /// Default: 2.0
    pub info_value_bps: f64,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_uncertainty: 0.1,
            max_fill_rate: 10.0,
            info_value_bps: 2.0,
        }
    }
}

impl Default for QuoteGateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_edge_signal: 0.15,
            min_edge_confidence: 0.45,
            strong_signal_threshold: 0.50,
            position_threshold: 0.05,
            max_position_before_reduce_only: 0.7,
            cascade_protection: true,
            cascade_threshold: 0.3,
            quote_flat_without_edge: true,
            use_bayesian_warmup: true,
            min_ir_outcomes_for_trust: 25,
            probe_config: ProbeConfig::default(),
            bootstrap_config: BayesianBootstrapConfig::default(),
        }
    }
}

/// Input state for quote gate decision.
#[derive(Debug, Clone)]
pub struct QuoteGateInput {
    /// Flow imbalance signal [-1, +1].
    /// Positive = buying pressure (bullish), Negative = selling pressure (bearish).
    pub flow_imbalance: f64,

    /// Momentum confidence [0, 1].
    /// How confident are we in the momentum direction?
    pub momentum_confidence: f64,

    /// Momentum direction in bps.
    /// Positive = price rising, Negative = price falling.
    pub momentum_bps: f64,

    /// Current position (signed).
    /// Positive = long, Negative = short.
    pub position: f64,

    /// Maximum allowed position.
    pub max_position: f64,

    /// Is system still warming up?
    pub is_warmup: bool,

    /// Cascade size factor [0, 1].
    /// 1.0 = no cascade, 0.0 = full cascade.
    pub cascade_size_factor: f64,

    // === Fields for theoretical edge calculation ===
    
    /// Book imbalance signal [-1, +1].
    /// Positive = more bid depth (bullish), Negative = more ask depth (bearish).
    /// Used for theoretical edge estimation when IR not calibrated.
    pub book_imbalance: f64,

    /// Current bid-ask spread in basis points.
    /// Used to calculate spread capture edge.
    pub spread_bps: f64,

    /// Current volatility estimate (fractional, e.g., 0.001 = 0.1%).
    /// Used to estimate expected price movements.
    pub sigma: f64,

    /// Expected holding time in seconds.
    /// Used in expected value calculations.
    pub tau_seconds: f64,

    // === Fields for enhanced flow and kappa-driven decisions ===

    /// Current kappa (fill intensity) estimate.
    /// Used for MC simulation and kappa-driven spread adjustment.
    /// Default: 1000.0
    pub kappa_effective: f64,

    /// Enhanced flow signal [-1, +1].
    /// Multi-feature composite flow imbalance for IR calibration.
    /// Default: 0.0 (uses book_imbalance as fallback)
    pub enhanced_flow: f64,

    /// Optional MC simulation EV result (bps).
    /// If Some, can be used to override conservative decisions.
    /// Default: None
    pub mc_ev_bps: Option<f64>,

    // === Hierarchical Edge Belief (L2/L3 Fusion) ===

    /// L2 (learning module) P(positive edge) estimate.
    /// From adaptive ensemble or decision engine.
    /// None if L2 model not available or not trusted.
    pub l2_p_positive_edge: Option<f64>,

    /// L2 model health score [0, 1].
    /// Indicates how reliable the L2 model predictions are.
    /// 0 = unhealthy (don't trust), 1 = fully healthy.
    pub l2_model_health: f64,

    /// L3 (stochastic controller) trust level [0, 1].
    /// How much to trust L3's regime/state beliefs.
    /// 0 = don't trust (use theoretical), 1 = fully trust.
    pub l3_trust: f64,

    /// L3 belief about favorable market conditions.
    /// P(favorable conditions | state) from stochastic controller.
    /// None if L3 not providing belief.
    pub l3_belief: Option<f64>,

    /// Urgency score from L3 controller [0, 5+].
    /// 0 = no urgency, 2+ = high urgency (position needs management).
    /// Used for posterior-justified urgency overrides.
    pub urgency_score: f64,

    /// Adverse selection variance (from RegimeAwareBayesianAdverse).
    /// Used for posterior-justified decisions.
    pub adverse_variance: f64,

    // === Phase 7: Hawkes Excitation Fields ===

    /// Probability of cluster/cascade in next τ seconds [0, 1].
    /// High values indicate elevated risk of self-exciting price cascades.
    /// Default: 0.0
    pub hawkes_p_cluster: f64,

    /// Edge penalty multiplier from Hawkes excitation [0.5, 1.0].
    /// Lower values = more conservative quoting during high excitation.
    /// Default: 1.0
    pub hawkes_excitation_penalty: f64,

    /// Whether Hawkes predictor indicates high excitation state.
    /// When true, quote decisions should be more conservative.
    /// Default: false
    pub hawkes_is_high_excitation: bool,

    /// Spread widening factor from Hawkes [1.0, 3.0].
    /// Multiplier for GLFT optimal spread during high excitation.
    /// Default: 1.0
    pub hawkes_spread_widening: f64,

    /// Current Hawkes branching ratio n = α/β [0, 1).
    /// Values near 1 indicate near-critical clustering regime.
    /// Default: 0.3
    pub hawkes_branching_ratio: f64,

    // === Phase 8: RL and Competitor Model Fields ===

    /// RL policy recommended spread delta (bps).
    /// Positive = widen spread, Negative = tighten spread.
    /// Default: 0.0
    pub rl_spread_delta_bps: f64,

    /// RL policy recommended bid skew (bps).
    /// Positive = widen bid, Negative = tighten bid.
    /// Default: 0.0
    pub rl_bid_skew_bps: f64,

    /// RL policy recommended ask skew (bps).
    /// Positive = widen ask, Negative = tighten ask.
    /// Default: 0.0
    pub rl_ask_skew_bps: f64,

    /// RL policy confidence [0, 1].
    /// How confident is the RL agent in its recommendation?
    /// Default: 0.0
    pub rl_confidence: f64,

    /// Whether RL recommendation is exploration (vs exploitation).
    /// Default: false
    pub rl_is_exploration: bool,

    /// Expected Q-value from RL agent.
    /// Higher = agent expects more reward from current action.
    /// Default: 0.0
    pub rl_expected_q: f64,

    /// Competitor snipe probability [0, 1].
    /// From competitor model inference.
    /// Default: 0.1
    pub competitor_snipe_prob: f64,

    /// Competitor spread widening factor [0.8, 1.5].
    /// How much to widen spread due to competition.
    /// Default: 1.0
    pub competitor_spread_factor: f64,

    /// Estimated number of active competitor MMs.
    /// Default: 3.0
    pub competitor_count: f64,
}

impl Default for QuoteGateInput {
    fn default() -> Self {
        Self {
            flow_imbalance: 0.0,
            momentum_confidence: 0.5,
            momentum_bps: 0.0,
            position: 0.0,
            max_position: 1.0,
            is_warmup: false,
            cascade_size_factor: 1.0,
            book_imbalance: 0.0,
            spread_bps: 10.0,
            sigma: 0.001,
            tau_seconds: 1.0,
            kappa_effective: 1000.0,
            enhanced_flow: 0.0,
            mc_ev_bps: None,
            // Hierarchical Edge Belief defaults
            l2_p_positive_edge: None,
            l2_model_health: 0.0, // Unhealthy until proven otherwise
            l3_trust: 0.0,        // Don't trust L3 until calibrated
            l3_belief: None,
            urgency_score: 0.0,   // No urgency initially
            adverse_variance: 0.01, // Moderate variance
            // Phase 7: Hawkes Excitation defaults
            hawkes_p_cluster: 0.0,            // No cluster risk
            hawkes_excitation_penalty: 1.0,   // No penalty
            hawkes_is_high_excitation: false, // Not excited
            hawkes_spread_widening: 1.0,      // No widening
            hawkes_branching_ratio: 0.3,      // Moderate default
            // Phase 8: RL and Competitor Model defaults
            rl_spread_delta_bps: 0.0,         // No RL adjustment
            rl_bid_skew_bps: 0.0,
            rl_ask_skew_bps: 0.0,
            rl_confidence: 0.0,               // No confidence initially
            rl_is_exploration: false,
            rl_expected_q: 0.0,
            competitor_snipe_prob: 0.1,       // 10% baseline snipe risk
            competitor_spread_factor: 1.0,    // No competition adjustment
            competitor_count: 3.0,            // Assume 3 competitors
        }
    }
}

/// The Quote Gate.
///
/// Determines WHETHER to quote (and which sides) based on directional edge.
#[derive(Debug)]
pub struct QuoteGate {
    config: QuoteGateConfig,
    /// Bayesian bootstrap tracker for adaptive calibration exit.
    bootstrap_tracker: BayesianBootstrapTracker,
}

impl Default for QuoteGate {
    fn default() -> Self {
        Self::new(QuoteGateConfig::default())
    }
}

impl QuoteGate {
    /// Create a new Quote Gate with the given configuration.
    pub fn new(config: QuoteGateConfig) -> Self {
        let bootstrap_tracker = BayesianBootstrapTracker::new(config.bootstrap_config.clone());
        Self { config, bootstrap_tracker }
    }

    /// Get the configuration.
    pub fn config(&self) -> &QuoteGateConfig {
        &self.config
    }

    /// Get mutable reference to bootstrap tracker.
    pub fn bootstrap_tracker_mut(&mut self) -> &mut BayesianBootstrapTracker {
        &mut self.bootstrap_tracker
    }

    /// Get reference to bootstrap tracker.
    pub fn bootstrap_tracker(&self) -> &BayesianBootstrapTracker {
        &self.bootstrap_tracker
    }

    /// Compute hierarchical P(correct) by fusing L1/L2/L3 beliefs.
    ///
    /// Combines three layers of edge estimation:
    /// - L1 (theoretical): P(correct | book imbalance) from microstructure theory
    /// - L2 (tactical): P(positive edge | features) from learning module
    /// - L3 (strategic): P(favorable conditions | state) from stochastic controller
    ///
    /// Weights depend on:
    /// - Calibration state (use more L1 theoretical when uncalibrated)
    /// - L2 model health (use less L2 when health < threshold)
    /// - L3 trust level (use more L3 when trust is high)
    ///
    /// # Arguments
    /// * `theoretical_p` - L1 P(correct) from theoretical edge estimator
    /// * `input` - Contains L2 and L3 inputs
    ///
    /// # Returns
    /// Blended P(correct), typically in [0.5, 0.85]
    pub fn compute_hierarchical_p_correct(
        &self,
        theoretical_p: f64,
        input: &QuoteGateInput,
    ) -> f64 {
        let bootstrap_confidence = self.bootstrap_tracker.should_exit(0).p_calibrated;

        // Weight L1 more heavily when uncalibrated
        // When bootstrap_confidence = 0 (uncalibrated), w1 = 0.8
        // When bootstrap_confidence = 1 (calibrated), w1 = 0.5
        let w1_raw = 0.5 + 0.3 * (1.0 - bootstrap_confidence);

        // Weight L2 by model health and calibration
        // Only include L2 if we have a prediction and model is healthy
        let (w2_raw, l2_contrib) = if let Some(l2_p) = input.l2_p_positive_edge {
            let health_factor = input.l2_model_health.clamp(0.0, 1.0);
            let w2 = 0.3 * health_factor * bootstrap_confidence;
            (w2, l2_p)
        } else {
            (0.0, 0.5) // Neutral contribution
        };

        // Weight L3 by trust level and calibration
        // Only include L3 if we have a belief and trust is sufficient
        let (w3_raw, l3_contrib) = if let Some(l3_b) = input.l3_belief {
            let trust_factor = input.l3_trust.clamp(0.0, 1.0);
            let w3 = 0.2 * trust_factor * bootstrap_confidence;
            (w3, l3_b)
        } else {
            (0.0, 0.5) // Neutral contribution
        };

        // Normalize weights to sum to 1.0
        let total = w1_raw + w2_raw + w3_raw;
        let (w1, w2, w3) = if total > 0.0 {
            (w1_raw / total, w2_raw / total, w3_raw / total)
        } else {
            (1.0, 0.0, 0.0) // Fallback to pure L1
        };

        // Blend the probabilities
        let blended = theoretical_p * w1 + l2_contrib * w2 + l3_contrib * w3;

        debug!(
            theoretical_p = %format!("{:.3}", theoretical_p),
            l2_p = input.l2_p_positive_edge.map_or("N/A".to_string(), |p| format!("{:.3}", p)),
            l3_belief = input.l3_belief.map_or("N/A".to_string(), |p| format!("{:.3}", p)),
            l2_health = %format!("{:.2}", input.l2_model_health),
            l3_trust = %format!("{:.2}", input.l3_trust),
            bootstrap_confidence = %format!("{:.3}", bootstrap_confidence),
            weights = %format!("({:.2}, {:.2}, {:.2})", w1, w2, w3),
            blended = %format!("{:.3}", blended),
            "Hierarchical P(correct) fusion"
        );

        // Safety bounds to prevent extreme values
        blended.clamp(0.5, 0.85)
    }

    /// Check if L3 urgency override is justified by posterior beliefs.
    ///
    /// Override when:
    /// 1. L3 indicates high urgency (> 2.0)
    /// 2. L3 trust is high (> 0.8)
    /// 3. Adverse selection risk is bounded (upper CI < 35%)
    ///
    /// # Returns
    /// `Some(QuoteDecision::QuoteBoth)` if override justified, `None` otherwise.
    pub fn check_l3_urgency_override(&self, input: &QuoteGateInput) -> Option<QuoteDecision> {
        // Check urgency threshold
        if input.urgency_score <= 2.0 {
            return None;
        }

        // Check trust threshold
        if input.l3_trust <= 0.8 {
            return None;
        }

        // Check if adverse selection risk is bounded
        // adverse_variance is std^2, we want std for CI approximation
        let adverse_std = input.adverse_variance.sqrt();
        // Approximate upper CI: mean + 1.96 * std (assuming mean ~0.15 for normal regime)
        let approx_upper_ci = 0.15 + 1.96 * adverse_std;

        if approx_upper_ci >= 0.35 {
            // Adverse selection risk too high, don't override
            debug!(
                urgency = %format!("{:.1}", input.urgency_score),
                l3_trust = %format!("{:.2}", input.l3_trust),
                approx_upper_ci = %format!("{:.3}", approx_upper_ci),
                "L3 urgency override denied: adverse risk too high"
            );
            return None;
        }

        info!(
            urgency = %format!("{:.1}", input.urgency_score),
            l3_trust = %format!("{:.2}", input.l3_trust),
            adverse_std = %format!("{:.3}", adverse_std),
            approx_upper_ci = %format!("{:.3}", approx_upper_ci),
            "L3 urgency override: posterior supports action"
        );

        Some(QuoteDecision::QuoteBoth)
    }

    /// Main decision function: should we quote, and which sides?
    pub fn decide(&self, input: &QuoteGateInput) -> QuoteDecision {
        // If disabled, always quote both sides (legacy behavior)
        if !self.config.enabled {
            return QuoteDecision::QuoteBoth;
        }

        // 1. During warmup, don't quote
        if input.is_warmup {
            debug!("Quote gate: warmup");
            return QuoteDecision::NoQuote {
                reason: NoQuoteReason::Warmup,
            };
        }

        // 2. Cascade protection
        if self.config.cascade_protection
            && input.cascade_size_factor < self.config.cascade_threshold
        {
            info!(
                cascade_factor = %format!("{:.2}", input.cascade_size_factor),
                threshold = %format!("{:.2}", self.config.cascade_threshold),
                "Quote gate: cascade detected, pulling quotes"
            );
            return QuoteDecision::NoQuote {
                reason: NoQuoteReason::Cascade,
            };
        }

        // 2b. Hawkes excitation protection (Phase 7: Bayesian Fusion)
        // When Hawkes signals high cluster probability, be more conservative
        if let Some(hawkes_decision) = self.check_hawkes_protection(input) {
            return hawkes_decision;
        }

        // Compute normalized position
        let position_ratio = if input.max_position > 0.0 {
            input.position / input.max_position
        } else {
            0.0
        };
        let position_abs_ratio = position_ratio.abs();

        // 3. Check if we have directional edge
        // Edge requires EITHER:
        //   a) Signal >= min_edge_signal AND confidence >= min_edge_confidence
        //   b) Signal >= strong_signal_threshold (strong signal bypasses confidence check)
        let signal_strength = input.flow_imbalance.abs();
        let strong_signal = signal_strength >= self.config.strong_signal_threshold;
        let weak_signal_with_confidence = signal_strength >= self.config.min_edge_signal
            && input.momentum_confidence >= self.config.min_edge_confidence;
        let has_edge = strong_signal || weak_signal_with_confidence;

        // Determine if position is significant
        let has_significant_position = position_abs_ratio >= self.config.position_threshold;

        // Determine if position is large (needs reduction priority)
        let needs_reduction = position_abs_ratio >= self.config.max_position_before_reduce_only;

        // Determine edge direction
        // flow_imbalance > 0 = buying pressure = bullish
        // flow_imbalance < 0 = selling pressure = bearish
        let is_bullish = input.flow_imbalance > 0.0;

        // Determine if position aligns with edge
        // Long (position > 0) aligns with bullish (flow > 0)
        // Short (position < 0) aligns with bearish (flow < 0)
        let position_aligns_with_edge = if has_edge {
            (input.position > 0.0 && is_bullish) || (input.position < 0.0 && !is_bullish)
        } else {
            false
        };

        // Decision logic
        let decision = if has_edge {
            if needs_reduction && !position_aligns_with_edge {
                // Have edge but large position opposes it → URGENT reduce only
                let urgency = (position_abs_ratio / 1.0).min(1.0);
                if input.position > 0.0 {
                    // Long opposing bearish edge → sell urgently
                    QuoteDecision::QuoteOnlyAsks { urgency }
                } else {
                    // Short opposing bullish edge → buy urgently
                    QuoteDecision::QuoteOnlyBids { urgency }
                }
            } else {
                // Have edge, position is manageable or aligned → quote both with skew
                // (The skew is handled by the strategy, not the gate)
                QuoteDecision::QuoteBoth
            }
        } else {
            // No edge
            if has_significant_position {
                // No edge but have position → only quote to reduce
                let urgency = (position_abs_ratio * 0.5).min(1.0); // Less urgent than opposed
                if input.position > 0.0 {
                    // Long without edge → only sell to reduce
                    QuoteDecision::QuoteOnlyAsks { urgency }
                } else {
                    // Short without edge → only buy to reduce
                    QuoteDecision::QuoteOnlyBids { urgency }
                }
            } else if self.config.quote_flat_without_edge {
                // MARKET MAKING MODE: Quote both sides even without edge.
                // Market makers profit from spread capture, not direction.
                // Only stop quoting during genuine danger (cascade, toxic regime).
                debug!(
                    flow_imbalance = %format!("{:.3}", input.flow_imbalance),
                    momentum_conf = %format!("{:.2}", input.momentum_confidence),
                    "Quote gate: no edge but quoting both (market-making mode)"
                );
                QuoteDecision::QuoteBoth
            } else {
                // DIRECTIONAL MODE: No edge, flat position → DON'T QUOTE, wait for signal
                QuoteDecision::NoQuote {
                    reason: NoQuoteReason::NoEdgeFlat,
                }
            }
        };

        // Log significant decisions
        match &decision {
            QuoteDecision::NoQuote { reason } => {
                info!(
                    reason = %reason,
                    flow_imbalance = %format!("{:.3}", input.flow_imbalance),
                    momentum_conf = %format!("{:.2}", input.momentum_confidence),
                    position_ratio = %format!("{:.2}", position_ratio),
                    "Quote gate: NO QUOTE"
                );
            }
            QuoteDecision::QuoteOnlyBids { urgency } => {
                info!(
                    urgency = %format!("{:.2}", urgency),
                    flow_imbalance = %format!("{:.3}", input.flow_imbalance),
                    position_ratio = %format!("{:.2}", position_ratio),
                    has_edge = has_edge,
                    "Quote gate: ONLY BIDS (reducing short / bullish edge)"
                );
            }
            QuoteDecision::QuoteOnlyAsks { urgency } => {
                info!(
                    urgency = %format!("{:.2}", urgency),
                    flow_imbalance = %format!("{:.3}", input.flow_imbalance),
                    position_ratio = %format!("{:.2}", position_ratio),
                    has_edge = has_edge,
                    "Quote gate: ONLY ASKS (reducing long / bearish edge)"
                );
            }
            QuoteDecision::QuoteBoth => {
                debug!(
                    flow_imbalance = %format!("{:.3}", input.flow_imbalance),
                    momentum_conf = %format!("{:.2}", input.momentum_confidence),
                    has_edge = has_edge,
                    "Quote gate: BOTH SIDES"
                );
            }
        }

        decision
    }

    // =========================================================================
    // Phase 7: Hawkes-Bayesian Fusion Helpers
    // =========================================================================

    /// Check if Hawkes excitation warrants protective action.
    ///
    /// Returns Some(decision) if Hawkes state requires immediate action,
    /// None if normal decision logic should proceed.
    ///
    /// Protection levels:
    /// - Critical: p_cluster > 0.8 AND branching_ratio > 0.85 → NoQuote
    /// - High: p_cluster > 0.5 OR high_excitation → Quote only reducing side
    /// - Normal: No intervention, let normal logic proceed
    fn check_hawkes_protection(&self, input: &QuoteGateInput) -> Option<QuoteDecision> {
        // If cascade protection is disabled, skip Hawkes protection too
        if !self.config.cascade_protection {
            return None;
        }

        // Critical excitation: near-critical Hawkes regime
        // branching_ratio > 0.85 means process is nearly explosive
        if input.hawkes_p_cluster > 0.8 && input.hawkes_branching_ratio > 0.85 {
            warn!(
                p_cluster = %format!("{:.3}", input.hawkes_p_cluster),
                branching_ratio = %format!("{:.3}", input.hawkes_branching_ratio),
                penalty = %format!("{:.3}", input.hawkes_excitation_penalty),
                "Quote gate: CRITICAL Hawkes excitation, pulling quotes"
            );
            return Some(QuoteDecision::NoQuote {
                reason: NoQuoteReason::Cascade, // Use existing reason for compatibility
            });
        }

        // High excitation with significant position: only reduce
        if input.hawkes_is_high_excitation && input.position.abs() > 0.1 {
            let urgency = input.hawkes_p_cluster.min(1.0);

            info!(
                p_cluster = %format!("{:.3}", input.hawkes_p_cluster),
                is_high = input.hawkes_is_high_excitation,
                position = %format!("{:.3}", input.position),
                spread_widening = %format!("{:.2}", input.hawkes_spread_widening),
                "Quote gate: High Hawkes excitation, reduce-only mode"
            );

            return if input.position > 0.0 {
                // Long position → only sell to reduce
                Some(QuoteDecision::QuoteOnlyAsks { urgency })
            } else {
                // Short position → only buy to reduce
                Some(QuoteDecision::QuoteOnlyBids { urgency })
            };
        }

        // Moderate excitation: log warning but allow normal quoting
        if input.hawkes_p_cluster > 0.3 {
            debug!(
                p_cluster = %format!("{:.3}", input.hawkes_p_cluster),
                excitation_penalty = %format!("{:.3}", input.hawkes_excitation_penalty),
                spread_widening = %format!("{:.2}", input.hawkes_spread_widening),
                "Quote gate: Moderate Hawkes excitation, proceeding with caution"
            );
        }

        // No intervention needed
        None
    }

    /// Apply Hawkes-based edge penalty to a decision.
    ///
    /// Reduces effective edge when Hawkes excitation is high.
    /// This allows spreads to widen naturally via the GLFT formula.
    pub fn hawkes_adjusted_edge(&self, base_edge_bps: f64, input: &QuoteGateInput) -> f64 {
        base_edge_bps * input.hawkes_excitation_penalty
    }

    /// Get recommended spread widening factor from Hawkes state.
    ///
    /// Returns a multiplier >= 1.0 to widen GLFT optimal spread.
    /// Strategy layers should multiply their gamma by this factor.
    pub fn hawkes_spread_widening(&self, input: &QuoteGateInput) -> f64 {
        input.hawkes_spread_widening
    }

    // ========================================================================
    // Phase 8: RL and Competitor Model Integration
    // ========================================================================

    /// Apply RL policy adjustment to spread.
    ///
    /// Returns (bid_adjustment_bps, ask_adjustment_bps) to apply to GLFT spread.
    /// Uses RL recommendation weighted by confidence.
    pub fn rl_spread_adjustment(&self, input: &QuoteGateInput) -> (f64, f64) {
        // Only apply if confidence is meaningful
        if input.rl_confidence < 0.1 {
            return (0.0, 0.0);
        }

        // Scale adjustments by confidence
        let confidence_weight = input.rl_confidence.clamp(0.0, 1.0);

        // During exploration, use smaller adjustments
        let exploration_dampening = if input.rl_is_exploration { 0.5 } else { 1.0 };

        let effective_weight = confidence_weight * exploration_dampening;

        let bid_adj = (input.rl_spread_delta_bps + input.rl_bid_skew_bps) * effective_weight;
        let ask_adj = (input.rl_spread_delta_bps + input.rl_ask_skew_bps) * effective_weight;

        // Clamp to reasonable bounds
        (bid_adj.clamp(-3.0, 5.0), ask_adj.clamp(-3.0, 5.0))
    }

    /// Get total spread widening from all sources (Hawkes + Competitor + RL).
    ///
    /// Returns a multiplier for the GLFT optimal spread.
    pub fn total_spread_widening(&self, input: &QuoteGateInput) -> f64 {
        let hawkes_factor = input.hawkes_spread_widening;
        let competitor_factor = input.competitor_spread_factor;

        // RL adjustments in bps -> convert to multiplicative factor
        // Assume base spread is ~5 bps, so 1 bps delta = 20% change
        let (bid_adj, ask_adj) = self.rl_spread_adjustment(input);
        let avg_rl_adj = (bid_adj + ask_adj) / 2.0;
        let rl_factor = 1.0 + avg_rl_adj / 10.0; // 10 bps base spread assumption

        // Combine multiplicatively, with sanity bounds
        (hawkes_factor * competitor_factor * rl_factor).clamp(0.8, 3.0)
    }

    /// Check if RL recommends aggressive quoting (tighter spreads).
    ///
    /// Used to potentially override conservative decisions during
    /// high-confidence exploitation.
    pub fn rl_recommends_aggressive(&self, input: &QuoteGateInput) -> bool {
        // Must have high confidence, be exploiting, and recommend tightening
        input.rl_confidence > 0.7
            && !input.rl_is_exploration
            && input.rl_spread_delta_bps < -1.0  // Tightening by >1 bps
            && input.rl_expected_q > 0.5         // Positive expected reward
    }

    /// Check if competitor model suggests defensive quoting.
    ///
    /// Returns true if snipe risk is elevated and spreads should be wider.
    pub fn competitor_suggests_defensive(&self, input: &QuoteGateInput) -> bool {
        input.competitor_snipe_prob > 0.3  // >30% snipe risk
            || input.competitor_spread_factor > 1.3  // >30% competition premium
    }

    /// Get combined RL + competitor spread recommendation (bps).
    ///
    /// Returns the recommended spread adjustment in basis points.
    /// Positive = widen, Negative = tighten.
    pub fn combined_spread_recommendation_bps(&self, input: &QuoteGateInput) -> f64 {
        // RL component
        let (bid_adj, ask_adj) = self.rl_spread_adjustment(input);
        let rl_component = (bid_adj + ask_adj) / 2.0;

        // Competitor component (convert factor to bps)
        // (factor - 1) * 10 bps base spread
        let competitor_component = (input.competitor_spread_factor - 1.0) * 10.0;

        // Combine: RL is tactical, competitor is strategic
        let combined = rl_component + competitor_component * 0.7; // 70% weight on competition

        combined.clamp(-5.0, 10.0)
    }

    /// Check if the gate would allow any quoting given current inputs.
    pub fn would_quote(&self, input: &QuoteGateInput) -> bool {
        !matches!(self.decide(input), QuoteDecision::NoQuote { .. })
    }

    /// Get a human-readable summary of the current state.
    pub fn summary(&self, input: &QuoteGateInput) -> String {
        let decision = self.decide(input);
        let edge_status = if input.flow_imbalance.abs() >= self.config.min_edge_signal {
            if input.flow_imbalance > 0.0 {
                "BULLISH"
            } else {
                "BEARISH"
            }
        } else {
            "NO EDGE"
        };

        let position_status = if input.max_position > 0.0 {
            let ratio = input.position / input.max_position;
            if ratio.abs() < self.config.position_threshold {
                "FLAT".to_string()
            } else if ratio > 0.0 {
                format!("LONG {:.0}%", ratio * 100.0)
            } else {
                format!("SHORT {:.0}%", ratio.abs() * 100.0)
            }
        } else {
            "UNKNOWN".to_string()
        };

        format!(
            "QuoteGate[{} | {} | {:?}]",
            edge_status, position_status, decision
        )
    }

    /// Make a calibrated decision using Information Ratio-based edge detection.
    ///
    /// This replaces arbitrary thresholds with principled, data-derived values:
    /// - Edge signal: IR > 1.0 means signal adds value (vs. arbitrary 0.15)
    /// - Position threshold: Derived from P&L data (vs. arbitrary 0.05)
    /// - Reduce-only threshold: Regime-specific, derived from P&L (vs. arbitrary 0.70)
    /// - Cascade detection: Uses changepoint probability (vs. arbitrary 0.30)
    ///
    /// # Arguments
    /// * `input` - Standard quote gate input
    /// * `edge_signal` - Calibrated edge signal tracker
    /// * `pnl_tracker` - Position P&L tracker
    /// * `changepoint_prob` - Bayesian changepoint probability (from BOCD)
    pub fn decide_calibrated(
        &self,
        input: &QuoteGateInput,
        edge_signal: &CalibratedEdgeSignal,
        pnl_tracker: &PositionPnLTracker,
        changepoint_prob: f64,
    ) -> QuoteDecision {
        // If disabled, use legacy behavior
        if !self.config.enabled {
            return QuoteDecision::QuoteBoth;
        }

        // 1. During warmup, don't quote
        if input.is_warmup {
            debug!("Quote gate (calibrated): warmup");
            return QuoteDecision::NoQuote {
                reason: NoQuoteReason::Warmup,
            };
        }

        // 2. Cascade protection using changepoint probability (not arbitrary threshold)
        // High changepoint_prob means regime shift detected
        const CHANGEPOINT_CASCADE_THRESHOLD: f64 = 0.70;
        if self.config.cascade_protection && changepoint_prob > CHANGEPOINT_CASCADE_THRESHOLD {
            warn!(
                changepoint_prob = %format!("{:.3}", changepoint_prob),
                "Quote gate (calibrated): regime change detected, pulling quotes"
            );
            return QuoteDecision::NoQuote {
                reason: NoQuoteReason::Cascade,
            };
        }

        // Also use cascade_size_factor as a fallback (empirical market data)
        if self.config.cascade_protection
            && input.cascade_size_factor < self.config.cascade_threshold
        {
            info!(
                cascade_factor = %format!("{:.2}", input.cascade_size_factor),
                "Quote gate (calibrated): cascade via OI drop"
            );
            return QuoteDecision::NoQuote {
                reason: NoQuoteReason::Cascade,
            };
        }

        // 3. Compute position ratio
        let position_ratio = if input.max_position > 0.0 {
            input.position / input.max_position
        } else {
            0.0
        };
        let position_abs_ratio = position_ratio.abs();

        // 4. Get calibrated thresholds
        // Position threshold: derived from P&L data (where E[PnL] crosses zero)
        let position_threshold = pnl_tracker.derived_position_threshold();

        // Reduce-only threshold: regime-specific (infer regime from changepoint)
        // High changepoint_prob → cascade (regime 2), low → calm (regime 0)
        let regime = if changepoint_prob > 0.5 {
            2 // cascade
        } else if changepoint_prob > 0.2 {
            1 // volatile
        } else {
            0 // calm
        };
        let reduce_only_threshold = pnl_tracker.reduce_only_threshold(regime);

        // 5. Check if we have directional edge using IR
        // Either use Bayesian warmup (P(IR > 1.0) > tiered_threshold) or fixed threshold
        let (has_edge, signal_weight, bayesian_decision): (bool, f64, Option<BayesianDecision>) = if self.config.use_bayesian_warmup {
            // Use Bayesian IR check with L2-adjusted prior
            // l2_confidence derived from momentum_confidence (as proxy)
            let l2_confidence = input.momentum_confidence.clamp(0.5, 1.0);
            let decision = edge_signal.bayesian_check(l2_confidence);
            
            // Compute diagnostics
            let prior_influence = edge_signal.bayesian_prior_influence();
            let tier_threshold = edge_signal.get_current_tier_threshold();
            
            // Log Bayesian decision details
            debug!(
                posterior_prob = %format!("{:.3}", decision.posterior_prob),
                credible_lower = %format!("{:.3}", decision.credible_lower),
                post_mu = %format!("{:.3}", decision.post_mu),
                prior_influence = %format!("{:.2}", prior_influence),
                prior_mu_used = %format!("{:.3}", decision.prior_mu_used),
                samples = decision.samples,
                tier_threshold = %format!("{:.2}", tier_threshold),
                reason = %decision.reason,
                "Bayesian IR decision"
            );
            
            let weight = if decision.is_useful {
                (edge_signal.overall_ir() - 1.0).max(0.0)
            } else {
                0.0
            };
            (decision.is_useful, weight, Some(decision))
        } else {
            // Legacy: fixed sample count threshold
            let has_edge = edge_signal.is_useful();
            let signal_weight = edge_signal.signal_weight();
            
            // During warmup, fall back to effective threshold check
            let has_edge = if !edge_signal.is_warmed_up() {
                // Use cold-start threshold during calibration warmup
                let effective_threshold = edge_signal.effective_edge_threshold();
                input.flow_imbalance.abs() >= effective_threshold
                    && input.momentum_confidence >= self.config.min_edge_confidence
            } else {
                has_edge && signal_weight > 0.0
            };
            (has_edge, signal_weight, None)
        };

        // Log BayesianDecision diagnostics when available
        if let Some(decision) = &bayesian_decision {
            debug!(
                bayesian_is_useful = decision.is_useful,
                bayesian_posterior_prob = %format!("{:.3}", decision.posterior_prob),
                bayesian_credible_lower = %format!("{:.3}", decision.credible_lower),
                bayesian_post_mu = %format!("{:.3}", decision.post_mu),
                bayesian_stopped_out = decision.stopped_out,
                bayesian_reason = %decision.reason,
                "BayesianDecision diagnostics"
            );
        }

        // Determine if position is significant (using derived threshold)
        let has_significant_position = position_abs_ratio >= position_threshold;

        // Determine if position is large (needs reduction priority)
        let needs_reduction = position_abs_ratio >= reduce_only_threshold;

        // Determine edge direction
        let is_bullish = input.flow_imbalance > 0.0;

        // Determine if position aligns with edge
        let position_aligns_with_edge = if has_edge {
            (input.position > 0.0 && is_bullish) || (input.position < 0.0 && !is_bullish)
        } else {
            false
        };

        // Decision logic (same structure as original, but with calibrated thresholds)
        let decision = if has_edge {
            if needs_reduction && !position_aligns_with_edge {
                // Have edge but large position opposes it → URGENT reduce only
                let urgency = (position_abs_ratio / 1.0).min(1.0);
                if input.position > 0.0 {
                    QuoteDecision::QuoteOnlyAsks { urgency }
                } else {
                    QuoteDecision::QuoteOnlyBids { urgency }
                }
            } else {
                // Have edge, position is manageable or aligned → quote both
                QuoteDecision::QuoteBoth
            }
        } else {
            // No edge
            if has_significant_position {
                // No edge but have position → only quote to reduce
                let urgency = (position_abs_ratio * 0.5).min(1.0);
                if input.position > 0.0 {
                    QuoteDecision::QuoteOnlyAsks { urgency }
                } else {
                    QuoteDecision::QuoteOnlyBids { urgency }
                }
            } else if self.config.quote_flat_without_edge {
                // Market-making mode: quote both sides even without edge
                debug!(
                    signal_weight = %format!("{:.3}", signal_weight),
                    ir_warmed_up = edge_signal.is_warmed_up(),
                    "Quote gate (calibrated): no edge but quoting both (market-making mode)"
                );
                QuoteDecision::QuoteBoth
            } else {
                // No edge, flat position → DON'T QUOTE
                QuoteDecision::NoQuote {
                    reason: NoQuoteReason::NoEdgeFlat,
                }
            }
        };

        // Log calibrated decisions
        match &decision {
            QuoteDecision::NoQuote { reason } => {
                info!(
                    reason = %reason,
                    signal_weight = %format!("{:.3}", signal_weight),
                    ir_useful = has_edge,
                    position_ratio = %format!("{:.2}", position_ratio),
                    position_threshold = %format!("{:.2}", position_threshold),
                    changepoint_prob = %format!("{:.3}", changepoint_prob),
                    "Quote gate (calibrated): NO QUOTE"
                );
            }
            QuoteDecision::QuoteOnlyBids { urgency } => {
                info!(
                    urgency = %format!("{:.2}", urgency),
                    signal_weight = %format!("{:.3}", signal_weight),
                    position_ratio = %format!("{:.2}", position_ratio),
                    reduce_only_threshold = %format!("{:.2}", reduce_only_threshold),
                    "Quote gate (calibrated): ONLY BIDS"
                );
            }
            QuoteDecision::QuoteOnlyAsks { urgency } => {
                info!(
                    urgency = %format!("{:.2}", urgency),
                    signal_weight = %format!("{:.3}", signal_weight),
                    position_ratio = %format!("{:.2}", position_ratio),
                    reduce_only_threshold = %format!("{:.2}", reduce_only_threshold),
                    "Quote gate (calibrated): ONLY ASKS"
                );
            }
            QuoteDecision::QuoteBoth => {
                debug!(
                    signal_weight = %format!("{:.3}", signal_weight),
                    has_edge = has_edge,
                    position_threshold = %format!("{:.2}", position_threshold),
                    "Quote gate (calibrated): BOTH SIDES"
                );
            }
        }

        decision
    }

    /// Make a quote decision with theoretical edge fallback and Bayesian bootstrap.
    ///
    /// This extends `decide_calibrated` by falling back to the theoretical edge model
    /// when the IR-based edge signal is not yet calibrated. This solves the bootstrap
    /// problem on illiquid assets where price rarely moves enough for IR calibration.
    ///
    /// ## Bayesian Bootstrap Mode
    ///
    /// Uses a Gamma posterior over the calibration threshold to decide when to exit
    /// bootstrap mode. During bootstrap, we quote based on theoretical edge even when
    /// IR signals no edge, to generate the fills needed for calibration.
    ///
    /// Exit bootstrap when ANY of:
    /// 1. P(θ < current_outcomes | data) > 0.95
    /// 2. Posterior variance < 100
    /// 3. Expected remaining outcomes < 5
    ///
    /// ## Fallback Logic (post-bootstrap)
    ///
    /// When IR is calibrated but detects no edge, we still compute theoretical edge
    /// for comparison and can override if conditions warrant.
    ///
    /// # Arguments
    /// * `input` - Standard quote gate input (extended with book_imbalance, spread, sigma, tau)
    /// * `edge_signal` - Calibrated edge signal tracker
    /// * `pnl_tracker` - Position P&L tracker for thresholds
    /// * `changepoint_prob` - Bayesian changepoint probability
    /// * `theoretical_edge` - Theoretical edge estimator for fallback
    ///
    /// # Returns
    /// `QuoteDecision` using Bayesian bootstrap logic, IR, or theoretical edge
    pub fn decide_with_theoretical_fallback(
        &mut self,
        input: &QuoteGateInput,
        edge_signal: &CalibratedEdgeSignal,
        pnl_tracker: &PositionPnLTracker,
        changepoint_prob: f64,
        theoretical_edge: &mut TheoreticalEdgeEstimator,
    ) -> QuoteDecision {
        // === BAYESIAN BOOTSTRAP MODE ===
        // Check if we should still be in bootstrap phase
        let exit_decision = self.bootstrap_tracker.should_exit(edge_signal.total_outcomes());
        let is_bootstrap_phase = !exit_decision.should_exit;

        // Update bootstrap tracker with IR convergence observation
        self.bootstrap_tracker.observe(edge_signal.overall_ir(), edge_signal.total_outcomes());

        // First, try the IR-based decision
        let ir_decision = self.decide_calibrated(input, edge_signal, pnl_tracker, changepoint_prob);

        // If IR says we should quote, trust it
        match &ir_decision {
            QuoteDecision::QuoteBoth
            | QuoteDecision::QuoteOnlyBids { .. }
            | QuoteDecision::QuoteOnlyAsks { .. } => {
                return ir_decision;
            }
            QuoteDecision::NoQuote { reason } => {
                // Only fall back on NoEdgeFlat when IR isn't calibrated
                if *reason != NoQuoteReason::NoEdgeFlat {
                    return ir_decision;
                }
            }
        }

        // IR returned NoQuote with NoEdgeFlat reason
        // Log outcome ratio for illiquidity diagnostics
        let outcome_ratio = if edge_signal.total_predictions() > 0 {
            edge_signal.total_outcomes() as f64 / edge_signal.total_predictions() as f64
        } else {
            0.0
        };

        // === BAYESIAN BOOTSTRAP OVERRIDE ===
        // During bootstrap phase, force quoting if theoretical edge is positive
        // This breaks the vicious cycle: no quotes → no fills → IR never calibrates
        if is_bootstrap_phase {
            let theoretical_result = theoretical_edge.calculate_edge(
                input.book_imbalance,
                input.spread_bps,
                input.sigma,
                input.tau_seconds,
            );

            let bootstrap_min_edge = self.bootstrap_tracker.bootstrap_min_edge_bps();

            if theoretical_result.should_quote
                || theoretical_result.expected_edge_bps >= bootstrap_min_edge
            {
                info!(
                    mode = "bootstrap",
                    ir_outcomes = edge_signal.total_outcomes(),
                    p_calibrated = %format!("{:.3}", exit_decision.p_calibrated),
                    expected_remaining = %format!("{:.1}", exit_decision.expected_remaining),
                    posterior_mean = %format!("{:.1}", exit_decision.posterior_mean),
                    posterior_std = %format!("{:.1}", exit_decision.posterior_std),
                    theoretical_edge_bps = %format!("{:.2}", theoretical_result.expected_edge_bps),
                    p_correct = %format!("{:.3}", theoretical_result.p_correct),
                    book_imbalance = %format!("{:.3}", input.book_imbalance),
                    "BAYESIAN BOOTSTRAP: Forcing quote (posterior not yet confident)"
                );
                return QuoteDecision::QuoteBoth;
            }

            // Even in bootstrap, allow market-making mode to quote
            if self.config.quote_flat_without_edge && !input.is_warmup {
                debug!(
                    mode = "bootstrap",
                    ir_outcomes = edge_signal.total_outcomes(),
                    p_calibrated = %format!("{:.3}", exit_decision.p_calibrated),
                    "BOOTSTRAP MM mode: quoting to gather calibration data"
                );
                return QuoteDecision::QuoteBoth;
            }
        }

        // Past bootstrap phase or bootstrap conditions not met
        // Only trust IR if it has MEANINGFUL outcome data
        let has_meaningful_ir_data = edge_signal.total_outcomes() >= self.config.min_ir_outcomes_for_trust;

        if edge_signal.is_useful() && has_meaningful_ir_data {
            // IR has meaningful data AND detected no significant edge → trust it
            debug!(
                total_outcomes = edge_signal.total_outcomes(),
                total_predictions = edge_signal.total_predictions(),
                outcome_ratio = %format!("{:.2}", outcome_ratio),
                min_required = self.config.min_ir_outcomes_for_trust,
                p_calibrated = %format!("{:.3}", exit_decision.p_calibrated),
                "IR is_useful=true with meaningful data - respecting decision"
            );
            return ir_decision;
        }

        // === POSTERIOR-GUIDED MC THRESHOLDS ===
        // MC thresholds scale with bootstrap confidence
        // Lower thresholds when we need data (low P(calibrated))
        // Higher thresholds when calibrated (high P(calibrated))
        let confidence_factor = exit_decision.p_calibrated; // [0, 1]

        // Interpolate thresholds based on calibration confidence
        let mc_kappa_thresh = 1000.0 + 1000.0 * confidence_factor; // [1000, 2000]
        let mc_ev_thresh = 0.1 + 0.1 * confidence_factor;          // [0.1, 0.2]

        if let Some(mc_ev) = input.mc_ev_bps {
            if input.kappa_effective > mc_kappa_thresh && mc_ev > mc_ev_thresh {
                info!(
                    kappa_effective = %format!("{:.0}", input.kappa_effective),
                    mc_ev_bps = %format!("{:.2}", mc_ev),
                    p_calibrated = %format!("{:.3}", confidence_factor),
                    kappa_thresh = %format!("{:.0}", mc_kappa_thresh),
                    ev_thresh = %format!("{:.2}", mc_ev_thresh),
                    "MC override: thresholds adjusted by calibration confidence"
                );
                return QuoteDecision::QuoteBoth;
            }
        }

        // === L3 URGENCY OVERRIDE (Phase 6) ===
        // Override when L3 indicates high urgency AND posterior supports action
        if let Some(decision) = self.check_l3_urgency_override(input) {
            return decision;
        }

        // IR is NOT calibrated → use theoretical edge as fallback
        let theoretical_result = theoretical_edge.calculate_edge(
            input.book_imbalance,
            input.spread_bps,
            input.sigma,
            input.tau_seconds,
        );

        info!(
            expected_edge_bps = %format!("{:.2}", theoretical_result.expected_edge_bps),
            p_correct = %format!("{:.2}", theoretical_result.p_correct),
            book_imbalance = %format!("{:.3}", input.book_imbalance),
            spread_bps = %format!("{:.1}", input.spread_bps),
            should_quote = theoretical_result.should_quote,
            direction = theoretical_result.direction,
            ir_samples = edge_signal.total_outcomes(),
            outcome_ratio = %format!("{:.2}", outcome_ratio),
            p_calibrated = %format!("{:.3}", exit_decision.p_calibrated),
            "Theoretical edge fallback (IR not calibrated)"
        );

        // Soft threshold: allow QuoteBoth if edge is marginally negative
        // This adds robustness to parameter uncertainty in alpha/adverse_prior
        let min_edge_bps = theoretical_edge.config().min_edge_bps;
        let is_marginally_negative = !theoretical_result.should_quote
            && theoretical_result.expected_edge_bps > -min_edge_bps / 2.0;

        if !theoretical_result.should_quote {
            // === Phase 3: Probe Mode ===
            // If we are uncertain and need data, treat "info value" as edge.
            // This encourages quoting even when edge is slightly negative/neutral.
            let alpha_uncertainty = theoretical_edge.alpha_uncertainty();
            let fill_rate = theoretical_edge.fill_rate_per_hour();

            // Check if position is "flat enough" to probe
            let position_ratio = if input.max_position > 0.0 {
                input.position / input.max_position
            } else { 0.0 };
            let is_flat_enough = position_ratio.abs() < self.config.position_threshold;

            let should_probe = self.config.probe_config.enabled
                && !input.is_warmup
                && is_flat_enough
                && alpha_uncertainty >= self.config.probe_config.min_uncertainty
                // If fill rate is low OR we have very few samples, we should probe
                && (fill_rate <= self.config.probe_config.max_fill_rate || theoretical_edge.bayesian_fills() < 50);

            if should_probe {
                let effective_edge = theoretical_result.expected_edge_bps + self.config.probe_config.info_value_bps;

                // If effective edge (including info value) is positive, we quote.
                // We use a lower threshold (0.0) instead of min_edge_bps because we are paying for info.
                if effective_edge > 0.0 {
                    debug!(
                        uncertainty = %format!("{:.3}", alpha_uncertainty),
                        fill_rate = %format!("{:.2}", fill_rate),
                        info_value = %format!("{:.2}", self.config.probe_config.info_value_bps),
                        effective_edge = %format!("{:.2}", effective_edge),
                        "Probe mode active - quoting to gather data"
                    );
                    return QuoteDecision::QuoteBoth;
                }
            }

            // If marginally negative AND market-making mode, quote anyway
            // If strongly negative, only quote if market-making mode enabled
            let should_mm_quote = self.config.quote_flat_without_edge && !input.is_warmup;

            if is_marginally_negative && should_mm_quote {
                debug!(
                    edge_bps = %format!("{:.2}", theoretical_result.expected_edge_bps),
                    threshold = %format!("{:.2}", -min_edge_bps / 2.0),
                    "Marginal edge in MM mode - quoting both sides"
                );
                return QuoteDecision::QuoteBoth;
            }

            if should_mm_quote {
                debug!(
                    ir_outcomes = edge_signal.total_outcomes(),
                    outcome_ratio = %format!("{:.2}", outcome_ratio),
                    "Theoretical edge negative but market-making mode enabled - quoting both sides"
                );
                return QuoteDecision::QuoteBoth;
            }

            return QuoteDecision::NoQuote {
                reason: NoQuoteReason::NoEdgeFlat,
            };
        }

        // Theoretical edge is positive → quote in that direction
        // Compute position-based urgency
        let position_ratio = if input.max_position > 0.0 {
            (input.position / input.max_position).abs()
        } else {
            0.0
        };

        // Check if position would benefit from the edge direction
        let position_aligned = (input.position > 0.0 && theoretical_result.direction > 0)
            || (input.position < 0.0 && theoretical_result.direction < 0);

        // If we have a large position opposing the edge, reduce-only mode
        if position_ratio > 0.3 && !position_aligned && input.position.abs() > 0.0 {
            let urgency = (position_ratio * 0.8).min(1.0);
            if input.position > 0.0 {
                return QuoteDecision::QuoteOnlyAsks { urgency };
            } else {
                return QuoteDecision::QuoteOnlyBids { urgency };
            }
        }

        // Standard case: quote based on theoretical edge direction
        match theoretical_result.direction {
            1 => {
                // Bullish edge - we want to buy
                // Quote both sides but will skew toward bids
                QuoteDecision::QuoteBoth
            }
            -1 => {
                // Bearish edge - we want to sell
                // Quote both sides but will skew toward asks
                QuoteDecision::QuoteBoth
            }
            _ => {
                // Neutral (shouldn't happen if should_quote is true)
                QuoteDecision::NoQuote {
                    reason: NoQuoteReason::NoEdgeFlat,
                }
            }
        }
    }

    /// Get the current Bayesian bootstrap exit decision (for diagnostics).
    pub fn bootstrap_exit_decision(&self, current_outcomes: u64) -> BayesianExitDecision {
        self.bootstrap_tracker.should_exit(current_outcomes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_input() -> QuoteGateInput {
        QuoteGateInput {
            max_position: 100.0,  // Override for test consistency
            ..Default::default()
        }
    }

    #[test]
    fn test_no_edge_flat_quotes_both_in_mm_mode() {
        // Market-making mode (default): quote both sides even without edge
        let gate = QuoteGate::default();
        let input = QuoteGateInput {
            flow_imbalance: 0.1, // Below threshold
            momentum_confidence: 0.5,
            position: 0.0, // Flat
            ..default_input()
        };

        let decision = gate.decide(&input);
        // In MM mode, flat position without edge still quotes both sides
        assert!(matches!(decision, QuoteDecision::QuoteBoth));
    }

    #[test]
    fn test_no_edge_flat_no_quote_in_directional_mode() {
        // Directional mode: don't quote when flat without edge
        let gate = QuoteGate::new(QuoteGateConfig {
            quote_flat_without_edge: false,
            ..Default::default()
        });
        let input = QuoteGateInput {
            flow_imbalance: 0.1, // Below threshold
            momentum_confidence: 0.5,
            position: 0.0, // Flat
            ..default_input()
        };

        let decision = gate.decide(&input);
        assert!(matches!(
            decision,
            QuoteDecision::NoQuote {
                reason: NoQuoteReason::NoEdgeFlat
            }
        ));
    }

    #[test]
    fn test_no_edge_long_should_only_quote_asks() {
        let gate = QuoteGate::default();
        let input = QuoteGateInput {
            flow_imbalance: 0.1, // Below threshold
            momentum_confidence: 0.5,
            position: 10.0, // Long (10% of max)
            ..default_input()
        };

        let decision = gate.decide(&input);
        assert!(matches!(decision, QuoteDecision::QuoteOnlyAsks { .. }));
    }

    #[test]
    fn test_no_edge_short_should_only_quote_bids() {
        let gate = QuoteGate::default();
        let input = QuoteGateInput {
            flow_imbalance: 0.1, // Below threshold
            momentum_confidence: 0.5,
            position: -10.0, // Short (10% of max)
            ..default_input()
        };

        let decision = gate.decide(&input);
        assert!(matches!(decision, QuoteDecision::QuoteOnlyBids { .. }));
    }

    #[test]
    fn test_bullish_edge_should_quote_both() {
        let gate = QuoteGate::default();
        let input = QuoteGateInput {
            flow_imbalance: 0.4, // Above threshold (bullish)
            momentum_confidence: 0.7, // Above confidence threshold
            position: 0.0,
            ..default_input()
        };

        let decision = gate.decide(&input);
        assert!(matches!(decision, QuoteDecision::QuoteBoth));
    }

    #[test]
    fn test_bearish_edge_should_quote_both() {
        let gate = QuoteGate::default();
        let input = QuoteGateInput {
            flow_imbalance: -0.4, // Above threshold (bearish)
            momentum_confidence: 0.7,
            position: 0.0,
            ..default_input()
        };

        let decision = gate.decide(&input);
        assert!(matches!(decision, QuoteDecision::QuoteBoth));
    }

    #[test]
    fn test_bullish_edge_but_large_short_should_reduce() {
        let gate = QuoteGate::default();
        let input = QuoteGateInput {
            flow_imbalance: 0.4, // Bullish
            momentum_confidence: 0.7,
            position: -80.0, // Large short (80% of max, opposes bullish)
            ..default_input()
        };

        let decision = gate.decide(&input);
        // Short opposing bullish → urgently buy to reduce
        assert!(matches!(decision, QuoteDecision::QuoteOnlyBids { .. }));
    }

    #[test]
    fn test_warmup_should_not_quote() {
        let gate = QuoteGate::default();
        let input = QuoteGateInput {
            is_warmup: true,
            ..default_input()
        };

        let decision = gate.decide(&input);
        assert!(matches!(
            decision,
            QuoteDecision::NoQuote {
                reason: NoQuoteReason::Warmup
            }
        ));
    }

    #[test]
    fn test_cascade_should_not_quote() {
        let gate = QuoteGate::default();
        let input = QuoteGateInput {
            cascade_size_factor: 0.1, // Severe cascade
            ..default_input()
        };

        let decision = gate.decide(&input);
        assert!(matches!(
            decision,
            QuoteDecision::NoQuote {
                reason: NoQuoteReason::Cascade
            }
        ));
    }

    #[test]
    fn test_disabled_gate_always_quotes_both() {
        let gate = QuoteGate::new(QuoteGateConfig {
            enabled: false,
            ..Default::default()
        });
        let input = QuoteGateInput {
            flow_imbalance: 0.0, // No edge
            position: 0.0,       // Flat
            ..default_input()
        };

        // Even with no edge and flat, disabled gate quotes both
        let decision = gate.decide(&input);
        assert!(matches!(decision, QuoteDecision::QuoteBoth));
    }

    #[test]
    fn test_edge_with_aligned_position_quotes_both() {
        let gate = QuoteGate::default();
        let input = QuoteGateInput {
            flow_imbalance: 0.4, // Bullish edge
            momentum_confidence: 0.7,
            position: 50.0, // Long aligns with bullish
            ..default_input()
        };

        let decision = gate.decide(&input);
        assert!(matches!(decision, QuoteDecision::QuoteBoth));
    }

    #[test]
    fn test_probe_mode_activates_with_info_value() {
        let mut config = QuoteGateConfig::default();
        config.probe_config.enabled = true;
        // High info value to overcome spread cost/fees
        config.probe_config.info_value_bps = 50.0;
        config.probe_config.min_uncertainty = 0.0; // Always trigger
        config.probe_config.max_fill_rate = 100.0; // Always trigger

        // Disable market making mode so we can test probe specifically
        config.quote_flat_without_edge = false;

        let mut gate = QuoteGate::new(config);

        let input = QuoteGateInput {
            flow_imbalance: 0.0, // No edge
            // Large spread ensures theoretical edge is negative without probe info value
            spread_bps: 20.0,
            position: 0.0,
            ..default_input()
        };

        let edge_signal: CalibratedEdgeSignal = CalibratedEdgeSignal::new(Default::default());
        let pnl_tracker: PositionPnLTracker = PositionPnLTracker::new(Default::default());
        let mut theo_edge = TheoreticalEdgeEstimator::new();

        // changepoint_prob low -> IR not calibrated, will fallback to theoretical
        // theoretical edge with 0 imbalance and 20bps spread will be negative
        let decision = gate.decide_with_theoretical_fallback(
            &input,
            &edge_signal,
            &pnl_tracker,
            0.0,
            &mut theo_edge
        );

        // Should quote both due to probe bonus
        assert!(matches!(decision, QuoteDecision::QuoteBoth));
    }

    #[test]
    fn test_bayesian_bootstrap_forces_quotes() {
        // Test that Bayesian bootstrap mode forces quotes during calibration
        let mut gate = QuoteGate::default();

        let input = QuoteGateInput {
            flow_imbalance: 0.2, // Moderate signal
            book_imbalance: 0.2,
            spread_bps: 10.0,
            sigma: 0.001,
            tau_seconds: 1.0,
            position: 0.0,
            ..default_input()
        };

        let edge_signal: CalibratedEdgeSignal = CalibratedEdgeSignal::new(Default::default());
        let pnl_tracker: PositionPnLTracker = PositionPnLTracker::new(Default::default());
        let mut theo_edge = TheoreticalEdgeEstimator::new();

        // With 0 outcomes, we're definitely in bootstrap mode
        let decision = gate.decide_with_theoretical_fallback(
            &input,
            &edge_signal,
            &pnl_tracker,
            0.0,
            &mut theo_edge
        );

        // Should quote both in bootstrap mode
        assert!(matches!(decision, QuoteDecision::QuoteBoth));
    }

    #[test]
    fn test_bootstrap_tracker_updates() {
        let gate = QuoteGate::default();

        // Initially not exited
        assert!(!gate.bootstrap_tracker().is_exited());

        // Get exit decision at various outcome levels
        let decision_early = gate.bootstrap_exit_decision(5);
        assert!(!decision_early.should_exit);

        let decision_mid = gate.bootstrap_exit_decision(30);
        // May or may not exit, depending on convergence
        assert!(decision_mid.p_calibrated >= 0.0);
        assert!(decision_mid.p_calibrated <= 1.0);
    }

    // ==================== Hierarchical Edge Belief Tests ====================

    #[test]
    fn test_hierarchical_p_correct_l1_only() {
        let gate = QuoteGate::default();
        let input = QuoteGateInput {
            l2_p_positive_edge: None,
            l2_model_health: 0.0,
            l3_trust: 0.0,
            l3_belief: None,
            ..default_input()
        };

        // When L2 and L3 not available, should use 100% L1
        let theoretical_p = 0.6;
        let blended = gate.compute_hierarchical_p_correct(theoretical_p, &input);

        // Should be exactly theoretical_p (or very close)
        assert!((blended - theoretical_p).abs() < 0.01);
    }

    #[test]
    fn test_hierarchical_p_correct_with_l2() {
        let gate = QuoteGate::default();
        let input = QuoteGateInput {
            l2_p_positive_edge: Some(0.7),
            l2_model_health: 1.0, // Fully healthy
            l3_trust: 0.0,
            l3_belief: None,
            ..default_input()
        };

        let theoretical_p = 0.55;
        let blended = gate.compute_hierarchical_p_correct(theoretical_p, &input);

        // With healthy L2 providing higher probability, blended should be > theoretical
        // But since bootstrap_confidence = 0, L2 weight is low
        assert!(blended >= 0.5);
        assert!(blended <= 0.85);
    }

    #[test]
    fn test_hierarchical_p_correct_with_l3() {
        let gate = QuoteGate::default();
        let input = QuoteGateInput {
            l2_p_positive_edge: None,
            l2_model_health: 0.0,
            l3_trust: 1.0, // High trust
            l3_belief: Some(0.8), // Very favorable conditions
            ..default_input()
        };

        let theoretical_p = 0.55;
        let blended = gate.compute_hierarchical_p_correct(theoretical_p, &input);

        // With high trust L3, should influence result
        // But since bootstrap_confidence = 0, L3 weight is low
        assert!(blended >= 0.5);
        assert!(blended <= 0.85);
    }

    #[test]
    fn test_hierarchical_p_correct_bounds() {
        let gate = QuoteGate::default();

        // Test that result is always bounded [0.5, 0.85]
        let inputs = vec![
            QuoteGateInput {
                l2_p_positive_edge: Some(0.99),
                l2_model_health: 1.0,
                l3_belief: Some(0.99),
                l3_trust: 1.0,
                ..default_input()
            },
            QuoteGateInput {
                l2_p_positive_edge: Some(0.01),
                l2_model_health: 1.0,
                l3_belief: Some(0.01),
                l3_trust: 1.0,
                ..default_input()
            },
        ];

        for input in inputs {
            let blended = gate.compute_hierarchical_p_correct(0.9, &input);
            assert!(blended >= 0.5, "blended={} should be >= 0.5", blended);
            assert!(blended <= 0.85, "blended={} should be <= 0.85", blended);
        }
    }

    #[test]
    fn test_l3_urgency_override_triggers() {
        let gate = QuoteGate::default();

        // High urgency, high trust, low adverse variance → should override
        let input = QuoteGateInput {
            urgency_score: 3.0,      // High urgency (> 2.0)
            l3_trust: 0.9,           // High trust (> 0.8)
            adverse_variance: 0.001, // Low variance → low AS risk
            ..default_input()
        };

        let result = gate.check_l3_urgency_override(&input);
        assert!(result.is_some());
        assert!(matches!(result.unwrap(), QuoteDecision::QuoteBoth));
    }

    #[test]
    fn test_l3_urgency_override_denied_low_urgency() {
        let gate = QuoteGate::default();

        // Low urgency → should NOT override
        let input = QuoteGateInput {
            urgency_score: 1.0, // Low urgency (< 2.0)
            l3_trust: 0.9,
            adverse_variance: 0.001,
            ..default_input()
        };

        let result = gate.check_l3_urgency_override(&input);
        assert!(result.is_none());
    }

    #[test]
    fn test_l3_urgency_override_denied_low_trust() {
        let gate = QuoteGate::default();

        // Low trust → should NOT override
        let input = QuoteGateInput {
            urgency_score: 3.0,
            l3_trust: 0.5, // Low trust (< 0.8)
            adverse_variance: 0.001,
            ..default_input()
        };

        let result = gate.check_l3_urgency_override(&input);
        assert!(result.is_none());
    }

    #[test]
    fn test_l3_urgency_override_denied_high_adverse() {
        let gate = QuoteGate::default();

        // High adverse variance → high AS risk → should NOT override
        let input = QuoteGateInput {
            urgency_score: 3.0,
            l3_trust: 0.9,
            adverse_variance: 0.1, // High variance → AS upper CI will be > 35%
            ..default_input()
        };

        let result = gate.check_l3_urgency_override(&input);
        assert!(result.is_none());
    }

    // ==================== Phase 7: Hawkes Protection Tests ====================

    #[test]
    fn test_hawkes_protection_normal() {
        let gate = QuoteGate::default();

        // Normal Hawkes state → no intervention
        let input = QuoteGateInput {
            hawkes_p_cluster: 0.1,
            hawkes_excitation_penalty: 0.95,
            hawkes_is_high_excitation: false,
            hawkes_spread_widening: 1.0,
            hawkes_branching_ratio: 0.3,
            ..default_input()
        };

        let result = gate.check_hawkes_protection(&input);
        assert!(result.is_none());
    }

    #[test]
    fn test_hawkes_protection_critical() {
        let gate = QuoteGate::default();

        // Critical Hawkes: p_cluster > 0.8 AND branching > 0.85 → NoQuote
        let input = QuoteGateInput {
            hawkes_p_cluster: 0.9,
            hawkes_excitation_penalty: 0.5,
            hawkes_is_high_excitation: true,
            hawkes_spread_widening: 2.5,
            hawkes_branching_ratio: 0.9, // Near critical
            ..default_input()
        };

        let result = gate.check_hawkes_protection(&input);
        assert!(result.is_some());
        assert!(matches!(result.unwrap(), QuoteDecision::NoQuote { .. }));
    }

    #[test]
    fn test_hawkes_protection_high_excitation_with_position() {
        let gate = QuoteGate::default();

        // High excitation with long position → reduce only (asks)
        let input = QuoteGateInput {
            hawkes_p_cluster: 0.6,
            hawkes_is_high_excitation: true,
            hawkes_branching_ratio: 0.75,
            position: 0.5, // Long position
            ..default_input()
        };

        let result = gate.check_hawkes_protection(&input);
        assert!(result.is_some());
        assert!(matches!(result.unwrap(), QuoteDecision::QuoteOnlyAsks { .. }));
    }

    #[test]
    fn test_hawkes_protection_high_excitation_short_position() {
        let gate = QuoteGate::default();

        // High excitation with short position → reduce only (bids)
        let input = QuoteGateInput {
            hawkes_p_cluster: 0.6,
            hawkes_is_high_excitation: true,
            hawkes_branching_ratio: 0.75,
            position: -0.5, // Short position
            ..default_input()
        };

        let result = gate.check_hawkes_protection(&input);
        assert!(result.is_some());
        assert!(matches!(result.unwrap(), QuoteDecision::QuoteOnlyBids { .. }));
    }

    #[test]
    fn test_hawkes_protection_high_excitation_no_position() {
        let gate = QuoteGate::default();

        // High excitation but no position → no intervention (let normal logic decide)
        let input = QuoteGateInput {
            hawkes_p_cluster: 0.6,
            hawkes_is_high_excitation: true,
            hawkes_branching_ratio: 0.75,
            position: 0.05, // Small position (< 0.1 threshold)
            ..default_input()
        };

        let result = gate.check_hawkes_protection(&input);
        assert!(result.is_none());
    }

    #[test]
    fn test_hawkes_adjusted_edge() {
        let gate = QuoteGate::default();

        let input_normal = QuoteGateInput {
            hawkes_excitation_penalty: 1.0,
            ..default_input()
        };

        let input_excited = QuoteGateInput {
            hawkes_excitation_penalty: 0.6,
            ..default_input()
        };

        let base_edge = 10.0;

        // Normal: full edge
        let adjusted_normal = gate.hawkes_adjusted_edge(base_edge, &input_normal);
        assert!((adjusted_normal - 10.0).abs() < 0.01);

        // Excited: reduced edge
        let adjusted_excited = gate.hawkes_adjusted_edge(base_edge, &input_excited);
        assert!((adjusted_excited - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_hawkes_spread_widening() {
        let gate = QuoteGate::default();

        let input = QuoteGateInput {
            hawkes_spread_widening: 1.8,
            ..default_input()
        };

        let widening = gate.hawkes_spread_widening(&input);
        assert!((widening - 1.8).abs() < 0.01);
    }

    #[test]
    fn test_hawkes_protection_disabled() {
        let mut config = QuoteGateConfig::default();
        config.cascade_protection = false; // Disable cascade protection
        let gate = QuoteGate::new(config);

        // Even with critical Hawkes, should not protect when disabled
        let input = QuoteGateInput {
            hawkes_p_cluster: 0.99,
            hawkes_is_high_excitation: true,
            hawkes_branching_ratio: 0.95,
            position: 1.0,
            ..default_input()
        };

        let result = gate.check_hawkes_protection(&input);
        assert!(result.is_none());
    }

    // ==================== Phase 8: RL and Competitor Tests ====================

    #[test]
    fn test_rl_spread_adjustment_low_confidence() {
        let gate = QuoteGate::default();

        // Low confidence → no adjustment
        let input = QuoteGateInput {
            rl_confidence: 0.05,  // Below threshold
            rl_spread_delta_bps: 2.0,
            rl_bid_skew_bps: 1.0,
            rl_ask_skew_bps: -1.0,
            ..default_input()
        };

        let (bid_adj, ask_adj) = gate.rl_spread_adjustment(&input);
        assert!((bid_adj - 0.0).abs() < 0.01);
        assert!((ask_adj - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_rl_spread_adjustment_high_confidence() {
        let gate = QuoteGate::default();

        // High confidence → apply adjustments
        let input = QuoteGateInput {
            rl_confidence: 0.8,
            rl_spread_delta_bps: 2.0,
            rl_bid_skew_bps: 1.0,
            rl_ask_skew_bps: -1.0,
            rl_is_exploration: false,
            ..default_input()
        };

        let (bid_adj, ask_adj) = gate.rl_spread_adjustment(&input);
        // bid_adj = (2.0 + 1.0) * 0.8 = 2.4
        // ask_adj = (2.0 + -1.0) * 0.8 = 0.8
        assert!((bid_adj - 2.4).abs() < 0.01);
        assert!((ask_adj - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_rl_spread_adjustment_exploration_dampened() {
        let gate = QuoteGate::default();

        // Exploration mode → dampened adjustments
        let input = QuoteGateInput {
            rl_confidence: 0.8,
            rl_spread_delta_bps: 2.0,
            rl_bid_skew_bps: 0.0,
            rl_ask_skew_bps: 0.0,
            rl_is_exploration: true,  // Exploration dampening
            ..default_input()
        };

        let (bid_adj, ask_adj) = gate.rl_spread_adjustment(&input);
        // bid_adj = 2.0 * 0.8 * 0.5 = 0.8
        assert!((bid_adj - 0.8).abs() < 0.01);
        assert!((ask_adj - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_total_spread_widening_combines_factors() {
        let gate = QuoteGate::default();

        let input = QuoteGateInput {
            hawkes_spread_widening: 1.5,
            competitor_spread_factor: 1.2,
            rl_confidence: 0.5,
            rl_spread_delta_bps: 1.0,  // 1 bps widen
            rl_bid_skew_bps: 0.0,
            rl_ask_skew_bps: 0.0,
            rl_is_exploration: false,
            ..default_input()
        };

        let total = gate.total_spread_widening(&input);
        // Should combine Hawkes (1.5), competitor (1.2), and RL component
        assert!(total > 1.5);  // At least Hawkes factor
        assert!(total < 3.0);  // Bounded
    }

    #[test]
    fn test_rl_recommends_aggressive() {
        let gate = QuoteGate::default();

        // High confidence, exploitation, tightening, positive Q
        let input = QuoteGateInput {
            rl_confidence: 0.8,
            rl_is_exploration: false,
            rl_spread_delta_bps: -2.0,  // Tightening
            rl_expected_q: 1.0,         // Positive Q
            ..default_input()
        };

        assert!(gate.rl_recommends_aggressive(&input));
    }

    #[test]
    fn test_rl_recommends_not_aggressive_low_q() {
        let gate = QuoteGate::default();

        // Low expected Q → not aggressive
        let input = QuoteGateInput {
            rl_confidence: 0.8,
            rl_is_exploration: false,
            rl_spread_delta_bps: -2.0,
            rl_expected_q: 0.2,  // Low Q (< 0.5)
            ..default_input()
        };

        assert!(!gate.rl_recommends_aggressive(&input));
    }

    #[test]
    fn test_competitor_suggests_defensive() {
        let gate = QuoteGate::default();

        // High snipe prob → defensive
        let high_snipe = QuoteGateInput {
            competitor_snipe_prob: 0.4,  // > 30%
            competitor_spread_factor: 1.0,
            ..default_input()
        };
        assert!(gate.competitor_suggests_defensive(&high_snipe));

        // High competition factor → defensive
        let high_competition = QuoteGateInput {
            competitor_snipe_prob: 0.1,
            competitor_spread_factor: 1.4,  // > 30%
            ..default_input()
        };
        assert!(gate.competitor_suggests_defensive(&high_competition));

        // Normal conditions → not defensive
        let normal = QuoteGateInput {
            competitor_snipe_prob: 0.1,
            competitor_spread_factor: 1.1,
            ..default_input()
        };
        assert!(!gate.competitor_suggests_defensive(&normal));
    }

    #[test]
    fn test_combined_spread_recommendation() {
        let gate = QuoteGate::default();

        let input = QuoteGateInput {
            rl_confidence: 0.6,
            rl_spread_delta_bps: 1.0,
            rl_bid_skew_bps: 0.5,
            rl_ask_skew_bps: -0.5,
            rl_is_exploration: false,
            competitor_spread_factor: 1.2,  // +2 bps from competition (20% of 10 base)
            ..default_input()
        };

        let recommendation = gate.combined_spread_recommendation_bps(&input);
        // RL component: (1.0 + 0.5 + 1.0 + -0.5) / 2 * 0.6 = 0.6
        // Competitor component: (1.2 - 1.0) * 10 * 0.7 = 1.4
        // Total ≈ 2.0
        assert!(recommendation > 0.0);
        assert!(recommendation < 5.0);
    }
}
