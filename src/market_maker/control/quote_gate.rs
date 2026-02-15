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
use super::changepoint::MarketRegime;
use super::position_pnl_tracker::PositionPnLTracker;
use super::theoretical_edge::TheoreticalEdgeEstimator;
use crate::market_maker::belief::BeliefSnapshot;

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

    /// Quote both sides but widen spreads (changepoint pending confirmation)
    /// Used when: changepoint probability is high but not yet confirmed.
    /// This fixes the "pending widening" bug where we logged but didn't actually widen.
    WidenSpreads {
        /// Spread multiplier [1.0, 2.0+] - how much to widen spreads
        multiplier: f64,
        /// Current changepoint probability for logging/diagnostics
        changepoint_prob: f64,
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
    /// Rate limit quota exhausted (shadow price too high)
    QuotaExhausted,
    /// Pre-fill AS classifier predicts toxic flow - skip quoting
    ToxicFlow,
}

impl std::fmt::Display for NoQuoteReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NoQuoteReason::Warmup => write!(f, "warmup"),
            NoQuoteReason::NoEdgeFlat => write!(f, "no_edge_flat"),
            NoQuoteReason::Cascade => write!(f, "cascade"),
            NoQuoteReason::Manual => write!(f, "manual"),
            NoQuoteReason::QuotaExhausted => write!(f, "quota_exhausted"),
            NoQuoteReason::ToxicFlow => write!(f, "toxic_flow"),
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

    /// Market regime for regime-aware changepoint thresholds.
    /// ThinDex: High threshold (0.85), requires 2 confirmations
    /// LiquidCex: Standard threshold (0.5), requires 1 confirmation
    /// Cascade: Low threshold (0.3), requires 1 confirmation
    /// Default: ThinDex (conservative for DEX environments)
    pub market_regime: MarketRegime,

    /// Pre-fill toxicity threshold to completely skip quoting.
    /// When max(bid_toxicity, ask_toxicity) exceeds this, pull all quotes.
    /// Default: 0.75 (75% toxicity = very likely toxic)
    pub toxicity_gate_threshold: f64,

    /// Enable pre-fill toxicity gating.
    /// When true, toxic flow prediction can stop quoting entirely.
    /// Default: true
    pub enable_toxicity_gate: bool,

    /// Continuous shadow pricing configuration for quota-aware spread adjustment.
    pub quota_shadow: QuotaShadowConfig,
}

/// Configuration for continuous quota shadow pricing.
///
/// Instead of hard tier cutoffs, shadow pricing smoothly adjusts spreads
/// based on rate limit headroom. The shadow spread is:
///   shadow_spread_bps = lambda_shadow_bps / headroom_pct.max(0.01)
///
/// At 100% headroom: 0.5 bps (negligible)
/// At 50% headroom: 1.0 bps (mild)
/// At 10% headroom: 5.0 bps (significant)
/// At 5% headroom: 10.0 bps (aggressive)
/// At 1% headroom: 50.0 bps (prohibitive)
#[derive(Debug, Clone)]
pub struct QuotaShadowConfig {
    /// Base shadow price lambda (bps). Higher = more spread at low headroom.
    pub lambda_shadow_bps: f64,
    /// Headroom threshold below which ladder density is reduced.
    /// At full headroom: all levels. At zero: 1 level.
    /// Scaling: levels = max(1, (max_levels * headroom.sqrt()) as usize)
    pub min_headroom_for_full_ladder: f64,
    /// Maximum shadow spread in bps (cap to prevent blowup at headroom -> 0)
    pub max_shadow_spread_bps: f64,
}

impl Default for QuotaShadowConfig {
    fn default() -> Self {
        Self {
            lambda_shadow_bps: 0.5,
            min_headroom_for_full_ladder: 0.20,
            max_shadow_spread_bps: 50.0,
        }
    }
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
            market_regime: MarketRegime::ThinDex, // Conservative default for DEX
            toxicity_gate_threshold: 0.75,        // 75% toxicity = skip quoting
            enable_toxicity_gate: true,           // Enable toxicity gating by default
            quota_shadow: QuotaShadowConfig::default(),
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

    // === Phase 9: Rate Limit Shadow Price Fields ===

    /// Rate limit headroom as fraction [0, 1].
    /// 1.0 = full budget available, 0.0 = exhausted.
    /// Used for shadow price calculation and quota-aware decisions.
    /// Default: 1.0
    pub rate_limit_headroom_pct: f64,

    /// Volatility regime indicator (0=low, 1=normal, 2=high, 3=extreme).
    /// Higher regimes → faster quota recharge via fills → lower shadow price.
    /// Default: 1 (normal)
    pub vol_regime: u8,

    // === Pre-Fill Adverse Selection (Phase 10) ===

    /// Pre-fill toxicity prediction for bid side [0, 1].
    /// Higher = more likely to be adversely selected if filled.
    /// Default: 0.0
    pub pre_fill_toxicity_bid: f64,

    /// Pre-fill toxicity prediction for ask side [0, 1].
    /// Higher = more likely to be adversely selected if filled.
    /// Default: 0.0
    pub pre_fill_toxicity_ask: f64,

    /// Whether pre-fill signals are stale and should be treated conservatively.
    /// When true, toxicity predictions may be unreliable.
    /// Default: false
    pub pre_fill_signals_stale: bool,

    // === Phase 6: Centralized Belief Snapshot ===
    /// Direct access to centralized belief state for unified decision making.
    /// When Some, provides:
    /// - `beliefs.drift_vol` for drift/volatility posteriors
    /// - `beliefs.kappa` for fill intensity beliefs
    /// - `beliefs.regime` for market regime state
    /// - `beliefs.changepoint` for BOCD state
    /// - `beliefs.continuation` for position continuation beliefs
    /// - `beliefs.calibration` for model quality metrics
    pub beliefs: Option<BeliefSnapshot>,
}

impl QuoteGateInput {
    // === Phase 6: Unified belief accessors ===
    // These methods prefer centralized beliefs when available, falling back to input fields.

    /// Get effective sigma (volatility) from beliefs or input field.
    pub fn effective_sigma(&self) -> f64 {
        if let Some(ref beliefs) = self.beliefs {
            if beliefs.is_warmed_up() && beliefs.drift_vol.expected_sigma > 0.0 {
                return beliefs.drift_vol.expected_sigma;
            }
        }
        self.sigma
    }

    /// Get effective kappa (fill intensity) from beliefs or input field.
    pub fn effective_kappa(&self) -> f64 {
        if let Some(ref beliefs) = self.beliefs {
            if beliefs.is_warmed_up() && beliefs.kappa.kappa_effective > 0.0 {
                return beliefs.kappa.kappa_effective;
            }
        }
        self.kappa_effective
    }

    /// Get overall belief confidence [0, 1].
    pub fn belief_confidence(&self) -> f64 {
        self.beliefs
            .as_ref()
            .map(|b| b.overall_confidence())
            .unwrap_or(0.0)
    }

    /// Get drift confidence from beliefs.
    pub fn drift_confidence(&self) -> f64 {
        self.beliefs
            .as_ref()
            .map(|b| b.drift_vol.confidence)
            .unwrap_or(0.0)
    }

    /// Get kappa confidence from beliefs.
    pub fn kappa_confidence(&self) -> f64 {
        self.beliefs
            .as_ref()
            .map(|b| b.kappa.confidence)
            .unwrap_or(0.0)
    }

    /// Get regime confidence from beliefs.
    pub fn regime_confidence(&self) -> f64 {
        self.beliefs
            .as_ref()
            .map(|b| b.regime.confidence)
            .unwrap_or(0.0)
    }

    /// Get continuation probability from beliefs.
    pub fn continuation_probability(&self) -> f64 {
        self.beliefs
            .as_ref()
            .map(|b| b.continuation.p_fused)
            .unwrap_or(0.5)
    }

    /// Get changepoint probability (5-obs window) from beliefs.
    pub fn changepoint_probability(&self) -> f64 {
        self.beliefs
            .as_ref()
            .map(|b| b.changepoint.prob_5)
            .unwrap_or(0.0)
    }

    /// Check if beliefs indicate a changepoint is pending/confirmed.
    pub fn changepoint_detected(&self) -> bool {
        self.beliefs
            .as_ref()
            .map(|b| b.changepoint.result.is_detected())
            .unwrap_or(false)
    }

    /// Get learning trust factor from changepoint beliefs.
    /// Lower when changepoint is likely (stale model).
    pub fn learning_trust(&self) -> f64 {
        self.beliefs
            .as_ref()
            .map(|b| b.changepoint.learning_trust)
            .unwrap_or(1.0)
    }

    /// Check if belief system is warmed up.
    pub fn beliefs_warmed_up(&self) -> bool {
        self.beliefs
            .as_ref()
            .map(|b| b.is_warmed_up())
            .unwrap_or(false)
    }

    /// Get expected edge from beliefs (bps).
    pub fn expected_edge_bps(&self) -> f64 {
        self.beliefs
            .as_ref()
            .map(|b| b.edge.expected_edge)
            .unwrap_or(0.0)
    }

    /// Get probability of positive edge from beliefs.
    pub fn prob_positive_edge(&self) -> f64 {
        self.beliefs
            .as_ref()
            .map(|b| b.edge.p_positive)
            .unwrap_or(0.5)
    }

    /// Get fill prediction information ratio from beliefs.
    pub fn fill_ir(&self) -> f64 {
        self.beliefs
            .as_ref()
            .map(|b| b.calibration.fill.information_ratio)
            .unwrap_or(0.0)
    }

    /// Get adverse selection information ratio from beliefs.
    pub fn as_ir(&self) -> f64 {
        self.beliefs
            .as_ref()
            .map(|b| b.calibration.adverse_selection.information_ratio)
            .unwrap_or(0.0)
    }

    /// Check if fill model adds value (IR > 1.0 with enough samples).
    pub fn fill_model_calibrated(&self) -> bool {
        self.beliefs
            .as_ref()
            .map(|b| b.calibration.fill.adds_value())
            .unwrap_or(false)
    }

    /// Check if AS model adds value (IR > 1.0 with enough samples).
    pub fn as_model_calibrated(&self) -> bool {
        self.beliefs
            .as_ref()
            .map(|b| b.calibration.adverse_selection.adds_value())
            .unwrap_or(false)
    }

    /// Get current regime from beliefs.
    pub fn current_regime(&self) -> Option<crate::market_maker::belief::Regime> {
        self.beliefs.as_ref().map(|b| b.regime.current)
    }

    /// Get regime probabilities from beliefs.
    pub fn regime_probs(&self) -> Option<[f64; 4]> {
        self.beliefs.as_ref().map(|b| b.regime.probs)
    }
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
            // Phase 9: Rate Limit Shadow Price defaults
            rate_limit_headroom_pct: 1.0,     // Full budget available
            vol_regime: 1,                    // Normal volatility
            // Phase 10: Pre-Fill AS Toxicity defaults
            pre_fill_toxicity_bid: 0.0,       // No toxicity initially
            pre_fill_toxicity_ask: 0.0,       // No toxicity initially
            pre_fill_signals_stale: false,    // Signals fresh initially
            // Phase 6: Centralized Belief Snapshot
            beliefs: None,
        }
    }
}

/// The Quote Gate.
///
/// Determines WHETHER to quote (and which sides) based on directional edge.
/// Now includes bistability escape mechanisms to prevent getting stuck in
/// low-quote attractor states.
///
/// Regime-aware changepoint detection:
/// - ThinDex: threshold=0.85, requires 2 consecutive high-prob signals
/// - LiquidCex: threshold=0.50, requires 1 confirmation
/// - Cascade: threshold=0.30, requires 1 confirmation
#[derive(Debug)]
pub struct QuoteGate {
    config: QuoteGateConfig,
    /// Bayesian bootstrap tracker for adaptive calibration exit.
    bootstrap_tracker: BayesianBootstrapTracker,
    /// Last time a quote was placed (for time-decaying thresholds).
    last_quote_time: Option<std::time::Instant>,
    /// Consecutive cycles without quoting (for ε-probing escalation).
    consecutive_no_quote_cycles: u32,
    /// Consecutive cycles with high changepoint probability.
    /// Used for confirmation-based regime change detection.
    consecutive_high_changepoint: u32,
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
        Self {
            config,
            bootstrap_tracker,
            last_quote_time: None,
            consecutive_no_quote_cycles: 0,
            consecutive_high_changepoint: 0,
        }
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

    /// Get the current market regime.
    pub fn market_regime(&self) -> MarketRegime {
        self.config.market_regime
    }

    /// Set the market regime for regime-aware changepoint thresholds.
    ///
    /// This allows runtime adaptation based on detected market conditions:
    /// - ThinDex: High threshold (0.85), requires 2 confirmations
    /// - LiquidCex: Standard threshold (0.50), requires 1 confirmation
    /// - Cascade: Low threshold (0.30), requires 1 confirmation
    pub fn set_market_regime(&mut self, regime: MarketRegime) {
        if self.config.market_regime != regime {
            info!(
                old_regime = ?self.config.market_regime,
                new_regime = ?regime,
                "Quote gate: market regime changed"
            );
            self.config.market_regime = regime;
            // Reset confirmation counter on regime change
            self.consecutive_high_changepoint = 0;
        }
    }

    /// Get the current changepoint confirmation count.
    pub fn changepoint_confirmation_count(&self) -> u32 {
        self.consecutive_high_changepoint
    }

    // =========================================================================
    // Shadow Price for Request Cost (Death Spiral Prevention)
    // =========================================================================

    /// Compute the shadow price of making an API request in basis points.
    ///
    /// The shadow price represents the marginal value of quota - it explodes
    /// nonlinearly as headroom approaches zero. This is derived from the MDP
    /// formalization where λ ≈ ∂V/∂h (marginal value of quota).
    ///
    /// The shadow price is regime-dependent:
    /// - High volatility → more fills → faster quota recharge → lower shadow price
    /// - Low volatility → fewer fills → slower recharge → higher shadow price
    ///
    /// # Arguments
    /// * `headroom_pct` - Current quota headroom as fraction [0, 1]
    /// * `vol_regime` - Volatility regime (0=low, 1=normal, 2=high, 3=extreme)
    ///
    /// # Returns
    /// Shadow price in basis points. When this exceeds expected edge, don't quote.
    ///
    /// # Boundaries
    /// - When headroom >= 50%: shadow price = 0 (free to quote)
    /// - When headroom <= 5%: shadow price = 100 bps (effectively infinite)
    /// - Between: cubic explosion λ ∝ (1 - h)³
    pub fn compute_request_shadow_price(headroom_pct: f64, vol_regime: u8) -> f64 {
        // Regime adjustment: high-vol → fills → faster recharge → lower shadow
        let regime_mult = match vol_regime {
            0 => 1.2,  // Low volatility: conservative (fills rare)
            1 => 1.0,  // Normal: baseline
            2 => 0.7,  // High: can afford more requests
            3 => 0.5,  // Extreme: fills plentiful
            _ => 1.0,  // Fallback
        };

        // When headroom is high (>50%), shadow price ≈ 0
        if headroom_pct >= 0.50 {
            return 0.0;
        }

        // When headroom is critically low (<=5%), shadow price → ∞
        // Effectively prohibit quoting
        if headroom_pct <= 0.05 {
            return 100.0;
        }

        // Sigmoid-like explosion between 5% and 50%
        // Normalized x ∈ [0, 1] where x=0 at 50% headroom, x=1 at 5% headroom
        let x = (0.50 - headroom_pct) / 0.45;

        // Base price for cubic explosion
        let base_price = 50.0; // Max shadow price in bps (before regime adjustment)

        // Cubic curve: steeper as headroom drops
        // At 30% headroom: x ≈ 0.44, shadow ≈ 4.3 bps
        // At 15% headroom: x ≈ 0.78, shadow ≈ 23.7 bps
        // At 10% headroom: x ≈ 0.89, shadow ≈ 35.2 bps
        let shadow = base_price * x.powi(3) * regime_mult;

        // Log at debug level for transparency
        debug!(
            headroom_pct = %format!("{:.1}%", headroom_pct * 100.0),
            vol_regime = vol_regime,
            regime_mult = %format!("{:.2}", regime_mult),
            shadow_bps = %format!("{:.2}", shadow),
            "Shadow price computed"
        );

        shadow
    }

    /// Check if edge justifies the shadow price of requesting.
    ///
    /// Returns true if the expected edge exceeds the shadow price.
    /// This internalizes quota cost into quoting decisions.
    pub fn edge_justifies_request(&self, expected_edge_bps: f64, input: &QuoteGateInput) -> bool {
        let shadow_price = Self::compute_request_shadow_price(
            input.rate_limit_headroom_pct,
            input.vol_regime,
        );
        expected_edge_bps > shadow_price
    }

    /// Compute continuous shadow spread adjustment in basis points.
    ///
    /// Instead of hard-blocking at low headroom, this returns a spread addition
    /// that smoothly increases as headroom decreases. The GLFT spread absorbs
    /// this cost, naturally reducing quoting frequency at low headroom without
    /// cliff effects.
    ///
    /// Formula: shadow_spread = lambda / headroom.max(0.01), capped at max_bps.
    ///
    /// At 100% headroom: 0.5 bps (negligible)
    /// At 50% headroom:  1.0 bps (mild)
    /// At 20% headroom:  2.5 bps (noticeable)
    /// At 10% headroom:  5.0 bps (significant)
    /// At 5% headroom:   10.0 bps (aggressive)
    pub fn continuous_shadow_spread_bps(&self, headroom_pct: f64) -> f64 {
        let config = &self.config.quota_shadow;
        let effective_headroom = headroom_pct.max(0.01);
        let raw = config.lambda_shadow_bps / effective_headroom;
        raw.min(config.max_shadow_spread_bps)
    }

    /// Compute continuous ladder density based on headroom.
    ///
    /// Smoothly scales ladder levels from 1 to max_levels based on headroom.
    /// Uses sqrt scaling so levels drop gradually, not in cliff steps.
    ///
    /// At 100% headroom: max_levels
    /// At 25% headroom:  max_levels/2
    /// At 4% headroom:   max_levels/5
    /// At 1% headroom:   1 level
    pub fn continuous_ladder_levels(&self, max_levels: usize, headroom_pct: f64) -> usize {
        let min_headroom = self.config.quota_shadow.min_headroom_for_full_ladder;
        if headroom_pct >= min_headroom {
            return max_levels;
        }
        // Scale by sqrt(headroom / min_headroom) for smooth reduction
        let scale = (headroom_pct / min_headroom).sqrt();
        (max_levels as f64 * scale).round().max(1.0) as usize
    }

    // =========================================================================
    // Bistability Escape Mechanisms
    // =========================================================================

    /// Record that a quote was placed (resets no-quote counters).
    pub fn record_quote_placed(&mut self) {
        self.last_quote_time = Some(std::time::Instant::now());
        self.consecutive_no_quote_cycles = 0;
    }

    /// Record a no-quote cycle (for ε-probing escalation).
    pub fn record_no_quote_cycle(&mut self) {
        self.consecutive_no_quote_cycles += 1;
    }

    /// Get time since last quote in seconds.
    pub fn time_since_last_quote(&self) -> f64 {
        self.last_quote_time
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Compute adjusted min_edge threshold that decays over time.
    ///
    /// The longer the system has been in no-quote state, the lower the threshold
    /// becomes. This helps escape bistable traps where:
    /// - Few quotes → few fills → uncertain posteriors → even fewer quotes
    ///
    /// # Arguments
    /// * `base_min_edge` - The base minimum edge threshold (from config or calibration)
    ///
    /// # Returns
    /// Adjusted threshold, typically in [0.2 × base, 1.0 × base]
    ///
    /// # Time Decay
    /// - 0-60s without quote: no decay (use base threshold)
    /// - 60-300s: linear decay to 20% of base
    /// - >300s: floor at 20% of base
    pub fn adjusted_min_edge_threshold(&self, base_min_edge: f64) -> f64 {
        let time_since = self.time_since_last_quote();

        if time_since < 60.0 {
            return base_min_edge;
        }

        // Linear decay from 60s to 300s
        // decay ∈ [0, 1] where 0 = no decay, 1 = max decay
        let decay = ((time_since - 60.0) / 240.0).min(1.0);

        // Threshold drops to 20% of base at max decay
        base_min_edge * (1.0 - 0.8 * decay)
    }

    /// Determine if we should perform an ε-probe to escape bistability.
    ///
    /// ε-probing tunnels through bistable barriers by occasionally forcing
    /// a quote even when edge is marginal. This generates fills which:
    /// 1. Restore rate limit quota (fills contribute to recharge)
    /// 2. Generate calibration data (fills update posteriors)
    /// 3. Test if conditions have changed (edge may have returned)
    ///
    /// # Arguments
    /// * `input` - Contains rate_limit_headroom_pct for budget check
    ///
    /// # Returns
    /// True if we should override NoEdgeFlat and force a quote
    ///
    /// # ε Schedule
    /// - Only probe when headroom > 10% (lowered from 30% to allow escape at low quota)
    /// - ε increases with time stuck in no-quote state
    /// - After 60s: ε = 0.02 (2% chance per cycle)
    /// - After 300s: ε = 0.10 (10% chance per cycle)
    pub fn should_epsilon_probe(&self, input: &QuoteGateInput) -> bool {
        // Only probe when we have quota budget
        // Threshold lowered from 30% to 10% so the system can escape NoQuote
        // even at low quota — wide two-sided quoting handles the quota conservation
        if input.rate_limit_headroom_pct < 0.10 {
            return false;
        }

        let time_since = self.time_since_last_quote();

        // Don't probe if we've been quoting recently
        if time_since < 60.0 {
            return false;
        }

        // ε increases with time stuck in no-quote state
        let base_epsilon = 0.02;
        let max_epsilon = 0.10;
        let time_factor = ((time_since - 60.0) / 240.0).min(1.0);
        let epsilon = base_epsilon + (max_epsilon - base_epsilon) * time_factor;

        let probe = rand::random::<f64>() < epsilon;

        if probe {
            info!(
                time_since_quote_secs = %format!("{:.1}", time_since),
                epsilon = %format!("{:.3}", epsilon),
                headroom_pct = %format!("{:.1}%", input.rate_limit_headroom_pct * 100.0),
                "ε-probe triggered: forcing quote to escape bistable trap"
            );
        }

        probe
    }

    /// Quota-driven spread widening — DISABLED.
    ///
    /// Previously computed a `1/sqrt(headroom)` spread multiplier when quota < 10%,
    /// but this caused a death spiral: at 9% headroom the 3.34x multiplier made quotes
    /// uncompetitive, killed fills, and prevented quota recovery.
    ///
    /// Quota pressure is now handled by:
    /// 1. `quota_shadow_spread_bps` (additive, not multiplicative)
    /// 2. Cycle frequency throttling (fewer updates, not wider spreads)
    pub fn wide_two_sided_decision(&self, _input: &QuoteGateInput) -> Option<QuoteDecision> {
        None
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
            l2_p = input.l2_p_positive_edge.map_or("N/A".to_string(), |p| format!("{p:.3}")),
            l3_belief = input.l3_belief.map_or("N/A".to_string(), |p| format!("{p:.3}")),
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

        // 2a. Pre-fill toxicity gating
        // If either side has toxicity above threshold, pull all quotes to avoid AS
        // When signals are stale, use a lower threshold (be more conservative)
        if self.config.enable_toxicity_gate {
            let effective_threshold = if input.pre_fill_signals_stale {
                // 30% lower threshold when signals are stale (defense-first)
                self.config.toxicity_gate_threshold * 0.7
            } else {
                self.config.toxicity_gate_threshold
            };
            let max_toxicity = input.pre_fill_toxicity_bid.max(input.pre_fill_toxicity_ask);
            if max_toxicity >= effective_threshold {
                info!(
                    bid_toxicity = %format!("{:.3}", input.pre_fill_toxicity_bid),
                    ask_toxicity = %format!("{:.3}", input.pre_fill_toxicity_ask),
                    threshold = %format!("{:.2}", effective_threshold),
                    signals_stale = input.pre_fill_signals_stale,
                    "Quote gate: toxic flow predicted, pulling quotes"
                );
                return QuoteDecision::NoQuote {
                    reason: NoQuoteReason::ToxicFlow,
                };
            }
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

        // Decision logic - SIMPLIFIED via gamma modulation
        //
        // The position_direction_confidence feature in RiskFeatures handles the
        // "position from informed flow" case via gamma modulation (beta_confidence < 0):
        // - High confidence → lower gamma → tighter two-sided quotes
        // - Low confidence → higher gamma → natural inventory skew via GLFT
        //
        // Only go one-sided for EXTREME positions (needs_reduction threshold, e.g. 70%)
        let decision = if has_edge {
            if needs_reduction && !position_aligns_with_edge {
                // Have edge but VERY large position opposes it → URGENT reduce only
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
                // (The skew is handled by the strategy via gamma modulation)
                QuoteDecision::QuoteBoth
            }
        } else {
            // No edge - rely on gamma modulation, not magic thresholds
            //
            // PRINCIPLED APPROACH: Position direction confidence modulates gamma:
            // - High confidence (position from informed fills) → low gamma → quote both tightly
            // - Low confidence (adverse position) → high gamma → wide spreads + natural skew
            //
            // Only go one-sided for EXTREME positions (>70% threshold)
            if needs_reduction {
                // Very large position without edge → safety valve: reduce only
                let urgency = (position_abs_ratio * 0.5).min(1.0);
                if input.position > 0.0 {
                    QuoteDecision::QuoteOnlyAsks { urgency }
                } else {
                    QuoteDecision::QuoteOnlyBids { urgency }
                }
            } else if has_significant_position {
                // Moderate position: Quote both sides, let gamma modulation handle risk.
                debug!(
                    flow_imbalance = %format!("{:.3}", input.flow_imbalance),
                    momentum_conf = %format!("{:.2}", input.momentum_confidence),
                    has_significant_position = has_significant_position,
                    "Quote gate: quoting both (has position, gamma handles risk)"
                );
                QuoteDecision::QuoteBoth
            } else if self.should_use_directional_mode(input) {
                // REGIME-AWARE: Strong trending market detected via L2 confirmation.
                // Switch to directional mode even if quote_flat_without_edge is true.
                info!(
                    flow_imbalance = %format!("{:.3}", input.flow_imbalance),
                    l2_p_positive = input.l2_p_positive_edge.map_or("N/A".to_string(), |p| format!("{p:.3}")),
                    "Quote gate: DIRECTIONAL MODE (strong trend confirmed by L2)"
                );
                QuoteDecision::NoQuote {
                    reason: NoQuoteReason::NoEdgeFlat,
                }
            } else if self.config.quote_flat_without_edge {
                // MARKET MAKING MODE: Quote both sides.
                // Let gamma modulation via position_direction_confidence handle the risk.
                debug!(
                    flow_imbalance = %format!("{:.3}", input.flow_imbalance),
                    momentum_conf = %format!("{:.2}", input.momentum_confidence),
                    "Quote gate: quoting both (market making mode)"
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
            QuoteDecision::WidenSpreads { multiplier, changepoint_prob } => {
                info!(
                    multiplier = %format!("{:.2}", multiplier),
                    changepoint_prob = %format!("{:.3}", changepoint_prob),
                    flow_imbalance = %format!("{:.3}", input.flow_imbalance),
                    position_ratio = %format!("{:.2}", position_ratio),
                    "Quote gate: BOTH SIDES (widened spreads)"
                );
            }
        }

        decision
    }

    // =========================================================================
    // Regime-Aware Mode Detection
    // =========================================================================

    /// Check if we should switch to directional mode even when quote_flat_without_edge is true.
    ///
    /// This triggers when:
    /// 1. Flow imbalance is strong (|flow_imbalance| > 0.5)
    /// 2. L2 confirms the trend (p_positive_edge far from 0.5)
    ///
    /// The insight: when both L1 (flow imbalance) and L2 (learning module) agree
    /// on a strong directional bias, we're likely in a trending market where
    /// market-making will be picked off. Better to wait for signal than get run over.
    fn should_use_directional_mode(&self, input: &QuoteGateInput) -> bool {
        // Threshold for strong flow imbalance
        const STRONG_FLOW_THRESHOLD: f64 = 0.5;
        // Threshold for L2 confirmation (how far p_positive must be from 0.5)
        const L2_CONFIRMATION_THRESHOLD: f64 = 0.15; // |p - 0.5| > 0.15
        // Minimum L2 health to trust its signal
        const MIN_L2_HEALTH: f64 = 0.5;

        // Check if flow imbalance is strong
        let strong_flow = input.flow_imbalance.abs() > STRONG_FLOW_THRESHOLD;
        if !strong_flow {
            return false;
        }

        // Check if L2 is healthy enough to trust
        if input.l2_model_health < MIN_L2_HEALTH {
            return false;
        }

        // Check if L2 confirms the trend direction
        if let Some(l2_p) = input.l2_p_positive_edge {
            let l2_directional_strength = (l2_p - 0.5).abs();
            let l2_confirms = l2_directional_strength > L2_CONFIRMATION_THRESHOLD;

            // L2 direction should match flow direction
            let l2_bullish = l2_p > 0.5;
            let flow_bullish = input.flow_imbalance > 0.0;
            let direction_match = l2_bullish == flow_bullish;

            if l2_confirms && direction_match {
                debug!(
                    flow_imbalance = %format!("{:.3}", input.flow_imbalance),
                    l2_p_positive = %format!("{:.3}", l2_p),
                    l2_directional_strength = %format!("{:.3}", l2_directional_strength),
                    l2_model_health = %format!("{:.2}", input.l2_model_health),
                    "Regime detection: L2 confirms strong trend → directional mode"
                );
                return true;
            }
        }

        false
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
            "QuoteGate[{edge_status} | {position_status} | {decision:?}]"
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
        &mut self,
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

        // 2. Regime-aware cascade protection using changepoint probability
        // Different market regimes require different sensitivity levels:
        // - ThinDex: High threshold (0.85) with 2 confirmations to avoid false positives
        // - LiquidCex: Standard threshold (0.50) with 1 confirmation
        // - Cascade: Very sensitive (0.30) with 1 confirmation
        let (changepoint_threshold, required_confirmations) = match self.config.market_regime {
            MarketRegime::ThinDex => (0.85, 2),
            MarketRegime::LiquidCex => (0.50, 1),
            MarketRegime::Cascade => (0.30, 1),
        };

        if self.config.cascade_protection {
            if changepoint_prob > changepoint_threshold {
                // Increment consecutive high-prob counter
                self.consecutive_high_changepoint += 1;

                // Only trigger if we have enough confirmations
                if self.consecutive_high_changepoint >= required_confirmations {
                    // FIX: Check if position opposes the inferred market direction
                    // If position opposes flow, accelerate exit instead of pulling all quotes
                    // This prevents getting stuck with an opposed position during regime change
                    let position_opposes = (input.position > 0.0 && input.flow_imbalance < -0.3)
                        || (input.position < 0.0 && input.flow_imbalance > 0.3);

                    if position_opposes && input.position.abs() > input.max_position * 0.01 {
                        // Accelerate exit instead of pulling all quotes
                        let urgency = (input.position.abs() / input.max_position).min(1.0);
                        warn!(
                            changepoint_prob = %format!("{:.3}", changepoint_prob),
                            position = %format!("{:.4}", input.position),
                            flow_imbalance = %format!("{:.3}", input.flow_imbalance),
                            urgency = %format!("{:.2}", urgency),
                            regime = ?self.config.market_regime,
                            "Quote gate: regime change but position opposed - accelerating exit"
                        );
                        if input.position > 0.0 {
                            // Long position + sell pressure → quote asks to reduce
                            return QuoteDecision::QuoteOnlyAsks { urgency };
                        } else {
                            // Short position + buy pressure → quote bids to reduce
                            return QuoteDecision::QuoteOnlyBids { urgency };
                        }
                    }

                    // No position or aligned with flow - safe to pull quotes
                    warn!(
                        changepoint_prob = %format!("{:.3}", changepoint_prob),
                        threshold = %format!("{:.2}", changepoint_threshold),
                        confirmations = self.consecutive_high_changepoint,
                        regime = ?self.config.market_regime,
                        "Quote gate (calibrated): regime change CONFIRMED, pulling quotes"
                    );
                    return QuoteDecision::NoQuote {
                        reason: NoQuoteReason::Cascade,
                    };
                } else {
                    // Pending confirmation - return WidenSpreads instead of just logging
                    // This fixes the bug where we logged "widening spreads" but didn't actually widen
                    let confirmations = self.consecutive_high_changepoint;
                    let progress = confirmations as f64 / required_confirmations as f64;
                    // Spread multiplier ramps from 1.0 to 2.0 as confirmations increase
                    let multiplier = 1.0 + progress;

                    info!(
                        changepoint_prob = %format!("{:.3}", changepoint_prob),
                        threshold = %format!("{:.2}", changepoint_threshold),
                        confirmations = confirmations,
                        required = required_confirmations,
                        regime = ?self.config.market_regime,
                        spread_mult = %format!("{:.2}", multiplier),
                        "Quote gate: changepoint pending confirmation, applying spread widening"
                    );
                    return QuoteDecision::WidenSpreads {
                        multiplier,
                        changepoint_prob,
                    };
                }
            } else {
                // Reset counter when probability drops below threshold
                if self.consecutive_high_changepoint > 0 {
                    debug!(
                        changepoint_prob = %format!("{:.3}", changepoint_prob),
                        previous_confirmations = self.consecutive_high_changepoint,
                        "Quote gate: changepoint probability subsided, resetting counter"
                    );
                }
                self.consecutive_high_changepoint = 0;
            }
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

        // Decision logic - SIMPLIFIED via gamma modulation
        //
        // The position_direction_confidence feature in RiskFeatures now handles the
        // "position from informed flow" case via gamma modulation (beta_confidence < 0):
        // - High confidence → lower gamma → tighter two-sided quotes
        // - Low confidence → higher gamma → natural inventory skew via GLFT
        //
        // This removes magic threshold-based one-sided quoting in favor of the
        // principled GLFT formula: δ = (1/γ) × ln(1 + γ/κ) with inventory skew.
        let decision = if has_edge {
            if needs_reduction && !position_aligns_with_edge {
                // Have edge but VERY large position opposes it → URGENT reduce only
                // This is a safety valve for extreme positions only (e.g., >70% of max)
                let urgency = (position_abs_ratio / 1.0).min(1.0);
                if input.position > 0.0 {
                    QuoteDecision::QuoteOnlyAsks { urgency }
                } else {
                    QuoteDecision::QuoteOnlyBids { urgency }
                }
            } else {
                // Have edge, position is manageable or aligned → quote both
                // Let gamma modulation via position_direction_confidence handle urgency
                QuoteDecision::QuoteBoth
            }
        } else {
            // No edge (IR not useful or not calibrated)
            //
            // PRINCIPLED APPROACH: Don't use magic has_significant_position threshold.
            // Instead, rely on gamma modulation:
            // - If position is from informed flow → high confidence → low gamma → quote both
            // - If position is adverse → low confidence → high gamma → wide spreads + skew
            //
            // Only go one-sided for EXTREME positions (needs_reduction threshold, e.g. 70%)
            if needs_reduction {
                // Very large position without edge → safety valve: reduce only
                let urgency = (position_abs_ratio * 0.5).min(1.0);
                if input.position > 0.0 {
                    QuoteDecision::QuoteOnlyAsks { urgency }
                } else {
                    QuoteDecision::QuoteOnlyBids { urgency }
                }
            } else if self.config.quote_flat_without_edge || has_significant_position {
                // Market-making mode OR moderate position: quote both sides
                // Let gamma modulation handle the risk via position_direction_confidence
                // - High confidence positions get tight two-sided quotes
                // - Low confidence positions get wide quotes with inventory skew
                debug!(
                    signal_weight = %format!("{:.3}", signal_weight),
                    ir_warmed_up = edge_signal.is_warmed_up(),
                    has_significant_position = has_significant_position,
                    position_ratio = %format!("{:.2}", position_ratio),
                    "Quote gate (calibrated): quoting both (gamma modulation handles risk)"
                );
                QuoteDecision::QuoteBoth
            } else {
                // No edge, flat position, not market-making mode → DON'T QUOTE
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
            QuoteDecision::WidenSpreads { multiplier, changepoint_prob: cp } => {
                info!(
                    multiplier = %format!("{:.2}", multiplier),
                    changepoint_prob = %format!("{:.3}", cp),
                    signal_weight = %format!("{:.3}", signal_weight),
                    position_ratio = %format!("{:.2}", position_ratio),
                    "Quote gate (calibrated): BOTH SIDES (widened spreads)"
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
        // =======================================================================
        // PHASE 9: RATE LIMIT DEATH SPIRAL PREVENTION
        // Check quota-aware overrides FIRST, before any other logic
        // =======================================================================

        // Compute shadow price for later use
        let shadow_price = Self::compute_request_shadow_price(
            input.rate_limit_headroom_pct,
            input.vol_regime,
        );

        // 1. Wide two-sided quoting: when quota is critically low AND we have position,
        //    quote BOTH sides with widened spreads instead of one-sided forcing.
        //    CRITICAL: This runs BEFORE shadow price veto because:
        //    - Holding position with no orders is strictly worse than wide two-sided
        //    - Wide spreads conserve quota naturally (fewer fills) while maintaining presence
        //    - Any fill contributes to quota recharge, helping escape low-headroom state
        //    - Position skew handled by quote engine, not by killing one side
        //    - Defense-first: maintaining two-sided avoids whipsaw death spiral
        if let Some(wide_decision) = self.wide_two_sided_decision(input) {
            info!(
                headroom_pct = %format!("{:.1}%", input.rate_limit_headroom_pct * 100.0),
                shadow_price_bps = %format!("{:.1}", shadow_price),
                position = %format!("{:.4}", input.position),
                "Wide two-sided quoting overrides quota exhaustion - both sides with widened spreads"
            );
            self.record_quote_placed();
            return wide_decision;
        }

        // 2. Shadow price check: only hard-veto at truly exhausted quota (<1% headroom)
        //    For moderate quota pressure, continuous shadow spread (added to GLFT spread)
        //    naturally reduces quoting frequency without cliff effects.
        if input.rate_limit_headroom_pct < 0.01 {
            warn!(
                headroom_pct = %format!("{:.1}%", input.rate_limit_headroom_pct * 100.0),
                shadow_price_bps = %format!("{:.1}", shadow_price),
                position = %format!("{:.4}", input.position),
                "Rate limit quota truly exhausted (<1% headroom) - hard veto"
            );
            self.record_no_quote_cycle();
            return QuoteDecision::NoQuote {
                reason: NoQuoteReason::QuotaExhausted,
            };
        }

        // === BAYESIAN BOOTSTRAP MODE ===
        // Check if we should still be in bootstrap phase
        let exit_decision = self.bootstrap_tracker.should_exit(edge_signal.total_outcomes());
        let is_bootstrap_phase = !exit_decision.should_exit;

        // Update bootstrap tracker with IR convergence observation
        self.bootstrap_tracker.observe(edge_signal.overall_ir(), edge_signal.total_outcomes());

        // First, try the IR-based decision
        let ir_decision = self.decide_calibrated(input, edge_signal, pnl_tracker, changepoint_prob);

        // If IR says we should quote, trust it and record
        match &ir_decision {
            QuoteDecision::QuoteBoth
            | QuoteDecision::QuoteOnlyBids { .. }
            | QuoteDecision::QuoteOnlyAsks { .. }
            | QuoteDecision::WidenSpreads { .. } => {
                self.record_quote_placed();
                return ir_decision;
            }
            QuoteDecision::NoQuote { reason } => {
                // Only fall back on NoEdgeFlat when IR isn't calibrated
                if *reason != NoQuoteReason::NoEdgeFlat {
                    self.record_no_quote_cycle();
                    return ir_decision;
                }
            }
        }

        // IR returned NoQuote with NoEdgeFlat reason
        // === PHASE 3: BISTABILITY ESCAPE - ε-PROBING ===
        // When stuck in NoEdgeFlat for too long, occasionally force a quote to:
        // 1. Restore rate limit quota (fills contribute to recharge)
        // 2. Generate calibration data (fills update posteriors)
        // 3. Test if conditions have changed (edge may have returned)
        if self.should_epsilon_probe(input) {
            self.record_quote_placed();
            return QuoteDecision::QuoteBoth;
        }
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
                self.record_quote_placed();
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
                self.record_quote_placed();
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
            self.record_no_quote_cycle();
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
                self.record_quote_placed();
                return QuoteDecision::QuoteBoth;
            }
        }

        // === L3 URGENCY OVERRIDE (Phase 6) ===
        // Override when L3 indicates high urgency AND posterior supports action
        if let Some(decision) = self.check_l3_urgency_override(input) {
            self.record_quote_placed();
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
                    self.record_quote_placed();
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
                self.record_quote_placed();
                return QuoteDecision::QuoteBoth;
            }

            if should_mm_quote {
                debug!(
                    ir_outcomes = edge_signal.total_outcomes(),
                    outcome_ratio = %format!("{:.2}", outcome_ratio),
                    "Theoretical edge negative but market-making mode enabled - quoting both sides"
                );
                self.record_quote_placed();
                return QuoteDecision::QuoteBoth;
            }

            self.record_no_quote_cycle();
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
            self.record_quote_placed();
            if input.position > 0.0 {
                return QuoteDecision::QuoteOnlyAsks { urgency };
            } else {
                return QuoteDecision::QuoteOnlyBids { urgency };
            }
        }

        // Standard case: quote based on theoretical edge direction
        let decision = match theoretical_result.direction {
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
        };

        // Track quote/no-quote cycle
        match &decision {
            QuoteDecision::NoQuote { .. } => self.record_no_quote_cycle(),
            _ => self.record_quote_placed(),
        }

        decision
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
    fn test_no_edge_moderate_position_should_quote_both() {
        // With the principled gamma modulation approach, moderate positions (10%)
        // without edge should quote both sides. Gamma modulation via
        // position_direction_confidence handles the risk via wider spreads and skew.
        let gate = QuoteGate::default();
        let input = QuoteGateInput {
            flow_imbalance: 0.1, // Below threshold
            momentum_confidence: 0.5,
            position: 10.0, // Long (10% of max) - moderate, not extreme
            ..default_input()
        };

        let decision = gate.decide(&input);
        // With quote_flat_without_edge=true (default), moderate positions quote both
        assert!(
            matches!(decision, QuoteDecision::QuoteBoth),
            "Moderate position (10%) should quote both with gamma modulation, got {:?}",
            decision
        );
    }

    #[test]
    fn test_no_edge_extreme_position_should_reduce_only() {
        // Only EXTREME positions (>70% of max) should go one-sided as a safety valve
        let gate = QuoteGate::default();
        let input = QuoteGateInput {
            flow_imbalance: 0.1, // Below threshold
            momentum_confidence: 0.5,
            position: 80.0,     // Long (80% of max) - EXTREME
            max_position: 100.0,
            ..default_input()
        };

        let decision = gate.decide(&input);
        // Extreme positions without edge → reduce only (safety valve)
        assert!(
            matches!(decision, QuoteDecision::QuoteOnlyAsks { .. }),
            "Extreme position (80%) without edge should reduce only, got {:?}",
            decision
        );
    }

    #[test]
    fn test_no_edge_extreme_short_should_reduce_only() {
        // Only EXTREME positions (>70% of max) should go one-sided as a safety valve
        let gate = QuoteGate::default();
        let input = QuoteGateInput {
            flow_imbalance: 0.1, // Below threshold
            momentum_confidence: 0.5,
            position: -80.0,    // Short (80% of max) - EXTREME
            max_position: 100.0,
            ..default_input()
        };

        let decision = gate.decide(&input);
        // Extreme positions without edge → reduce only (safety valve)
        assert!(
            matches!(decision, QuoteDecision::QuoteOnlyBids { .. }),
            "Extreme short position (80%) without edge should reduce only, got {:?}",
            decision
        );
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

    #[test]
    fn test_wide_two_sided_disabled_at_low_quota_with_long_position() {
        // wide_two_sided_decision is disabled: quota-driven spread widening caused
        // death spiral (3.34x at 9% headroom -> uncompetitive -> no fills -> no recovery).
        let gate = QuoteGate::default();

        let input = QuoteGateInput {
            rate_limit_headroom_pct: 0.03,
            position: 5.0,
            max_position: 100.0,
            vol_regime: 1,
            ..default_input()
        };

        let decision = gate.wide_two_sided_decision(&input);
        assert!(decision.is_none(), "wide_two_sided_decision should be disabled");
    }

    #[test]
    fn test_wide_two_sided_disabled_at_7pct_headroom() {
        // wide_two_sided_decision is disabled even at 7% headroom (our live death-spiral scenario).
        let gate = QuoteGate::default();

        let input = QuoteGateInput {
            rate_limit_headroom_pct: 0.07,
            position: 5.0,
            max_position: 100.0,
            vol_regime: 1,
            ..default_input()
        };

        let decision = gate.wide_two_sided_decision(&input);
        assert!(decision.is_none(), "wide_two_sided_decision should be disabled");
    }

    #[test]
    fn test_wide_two_sided_not_triggered_above_10pct() {
        // Above 10% headroom, wide two-sided should NOT activate
        let gate = QuoteGate::default();

        let input = QuoteGateInput {
            rate_limit_headroom_pct: 0.15, // 15% - above threshold
            position: 5.0,
            max_position: 100.0,
            vol_regime: 1,
            ..default_input()
        };

        let decision = gate.wide_two_sided_decision(&input);
        assert!(decision.is_none(), "Should not activate above 10% headroom");
    }

    #[test]
    fn test_wide_two_sided_no_position_returns_none() {
        // When quota is low but no position, should return None
        // (shadow price will then handle the NoQuote decision)
        let gate = QuoteGate::default();

        let input = QuoteGateInput {
            rate_limit_headroom_pct: 0.03,
            position: 0.0, // No position
            max_position: 100.0,
            vol_regime: 1,
            ..default_input()
        };

        let decision = gate.wide_two_sided_decision(&input);
        assert!(decision.is_none(), "Should not activate with no position");
    }

    #[test]
    fn test_wide_two_sided_disabled_short_position() {
        // wide_two_sided_decision is disabled even with short position at low headroom.
        let gate = QuoteGate::default();

        let input = QuoteGateInput {
            rate_limit_headroom_pct: 0.05,
            position: -10.0,
            max_position: 100.0,
            vol_regime: 1,
            ..default_input()
        };

        let decision = gate.wide_two_sided_decision(&input);
        assert!(decision.is_none(), "wide_two_sided_decision should be disabled");
    }

    #[test]
    fn test_wide_two_sided_disabled_at_extreme_low_headroom() {
        // wide_two_sided_decision is disabled even at 1% headroom.
        let gate = QuoteGate::default();

        let input = QuoteGateInput {
            rate_limit_headroom_pct: 0.01,
            position: 5.0,
            max_position: 100.0,
            vol_regime: 1,
            ..default_input()
        };

        let decision = gate.wide_two_sided_decision(&input);
        assert!(decision.is_none(), "wide_two_sided_decision should be disabled");
    }

    #[test]
    fn test_epsilon_probe_works_at_low_headroom() {
        // After lowering epsilon probe threshold from 30% to 10%,
        // probes should be possible at 15% headroom
        let mut gate = QuoteGate::default();
        // Set last_quote_time to 300s ago to ensure time threshold is met
        gate.last_quote_time = Some(std::time::Instant::now() - std::time::Duration::from_secs(300));

        let input = QuoteGateInput {
            rate_limit_headroom_pct: 0.15, // 15% - was blocked before, should work now
            ..default_input()
        };

        // Run many probes - with ε up to 0.10, at least one should trigger in 200 tries
        let mut triggered = false;
        for _ in 0..200 {
            if gate.should_epsilon_probe(&input) {
                triggered = true;
                break;
            }
        }
        assert!(triggered, "Epsilon probe should be possible at 15% headroom (threshold lowered to 10%)");
    }

    #[test]
    fn test_epsilon_probe_blocked_below_10pct() {
        // Below 10% headroom, epsilon probes should still be blocked
        let mut gate = QuoteGate::default();
        gate.last_quote_time = Some(std::time::Instant::now() - std::time::Duration::from_secs(300));

        let input = QuoteGateInput {
            rate_limit_headroom_pct: 0.05, // 5% - below new threshold
            ..default_input()
        };

        // Should never trigger
        let mut triggered = false;
        for _ in 0..200 {
            if gate.should_epsilon_probe(&input) {
                triggered = true;
                break;
            }
        }
        assert!(!triggered, "Epsilon probe should be blocked below 10% headroom");
    }

    // ========================================================================
    // Continuous Shadow Pricing Tests
    // ========================================================================

    #[test]
    fn test_continuous_shadow_spread_at_full_headroom() {
        let gate = QuoteGate::default();
        // At 100% headroom: lambda / 1.0 = 0.5 bps (negligible)
        let shadow = gate.continuous_shadow_spread_bps(1.0);
        assert!((shadow - 0.5).abs() < 0.01, "At 100% headroom shadow should be ~0.5 bps, got {}", shadow);
    }

    #[test]
    fn test_continuous_shadow_spread_at_10pct_headroom() {
        let gate = QuoteGate::default();
        // At 10% headroom: 0.5 / 0.10 = 5.0 bps (significant)
        let shadow = gate.continuous_shadow_spread_bps(0.10);
        assert!((shadow - 5.0).abs() < 0.01, "At 10% headroom shadow should be ~5.0 bps, got {}", shadow);
    }

    #[test]
    fn test_continuous_shadow_spread_at_5pct_headroom() {
        let gate = QuoteGate::default();
        // At 5% headroom: 0.5 / 0.05 = 10.0 bps (aggressive)
        let shadow = gate.continuous_shadow_spread_bps(0.05);
        assert!((shadow - 10.0).abs() < 0.01, "At 5% headroom shadow should be ~10.0 bps, got {}", shadow);
    }

    #[test]
    fn test_continuous_shadow_spread_capped_at_max() {
        let gate = QuoteGate::default();
        // At 0.1% headroom: 0.5 / 0.001 = 500 bps → capped at 50.0 bps
        let shadow = gate.continuous_shadow_spread_bps(0.001);
        assert!((shadow - 50.0).abs() < 0.01, "Shadow should be capped at 50 bps, got {}", shadow);
    }

    #[test]
    fn test_continuous_shadow_spread_smooth_increase() {
        // Verify no cliff effects: shadow spread increases monotonically as headroom decreases
        let gate = QuoteGate::default();
        let headrooms = [1.0, 0.5, 0.3, 0.2, 0.1, 0.07, 0.05, 0.03, 0.01];
        let mut prev_shadow = 0.0;
        for &h in &headrooms {
            let shadow = gate.continuous_shadow_spread_bps(h);
            assert!(shadow >= prev_shadow,
                "Shadow spread should increase as headroom decreases: at {:.0}% got {:.2} bps, prev {:.2} bps",
                h * 100.0, shadow, prev_shadow);
            prev_shadow = shadow;
        }
    }

    #[test]
    fn test_continuous_ladder_levels_at_full_headroom() {
        let gate = QuoteGate::default();
        // At 100% headroom: all levels
        let levels = gate.continuous_ladder_levels(10, 1.0);
        assert_eq!(levels, 10, "At full headroom should get all levels");
    }

    #[test]
    fn test_continuous_ladder_levels_at_min_threshold() {
        let gate = QuoteGate::default();
        // At 20% headroom (min_headroom_for_full_ladder): all levels
        let levels = gate.continuous_ladder_levels(10, 0.20);
        assert_eq!(levels, 10, "At min_headroom_for_full_ladder should get all levels");
    }

    #[test]
    fn test_continuous_ladder_levels_at_5pct() {
        let gate = QuoteGate::default();
        // At 5% headroom: sqrt(0.05/0.20) = sqrt(0.25) = 0.5 → 5 levels
        let levels = gate.continuous_ladder_levels(10, 0.05);
        assert_eq!(levels, 5, "At 5% headroom with 10 max should get ~5 levels, got {}", levels);
    }

    #[test]
    fn test_continuous_ladder_levels_at_1pct() {
        let gate = QuoteGate::default();
        // At 1% headroom: sqrt(0.01/0.20) = sqrt(0.05) ≈ 0.224 → round(2.24) = 2 levels
        let levels = gate.continuous_ladder_levels(10, 0.01);
        assert!(levels >= 1 && levels <= 3,
            "At 1% headroom with 10 max should get ~2 levels, got {}", levels);
    }

    #[test]
    fn test_continuous_ladder_levels_floor_at_1() {
        let gate = QuoteGate::default();
        // Even at extremely low headroom, should always get at least 1 level
        let levels = gate.continuous_ladder_levels(10, 0.001);
        assert_eq!(levels, 1, "Should always get at least 1 level, got {}", levels);
    }

    #[test]
    fn test_continuous_ladder_levels_smooth_decrease() {
        // Verify no cliff effects: levels decrease smoothly as headroom drops
        let gate = QuoteGate::default();
        let headrooms = [0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.01];
        let mut prev_levels = 100;
        for &h in &headrooms {
            let levels = gate.continuous_ladder_levels(25, h);
            assert!(levels <= prev_levels,
                "Levels should not increase as headroom drops: at {:.0}% got {}, prev {}",
                h * 100.0, levels, prev_levels);
            prev_levels = levels;
        }
    }
}
