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

use super::calibrated_edge::CalibratedEdgeSignal;
use super::position_pnl_tracker::PositionPnLTracker;

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
}

/// The Quote Gate.
///
/// Determines WHETHER to quote (and which sides) based on directional edge.
#[derive(Debug, Clone)]
pub struct QuoteGate {
    config: QuoteGateConfig,
}

impl Default for QuoteGate {
    fn default() -> Self {
        Self::new(QuoteGateConfig::default())
    }
}

impl QuoteGate {
    /// Create a new Quote Gate with the given configuration.
    pub fn new(config: QuoteGateConfig) -> Self {
        Self { config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &QuoteGateConfig {
        &self.config
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
        // The only principled threshold: IR > 1.0 means signal adds information
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_input() -> QuoteGateInput {
        QuoteGateInput {
            flow_imbalance: 0.0,
            momentum_confidence: 0.5,
            momentum_bps: 0.0,
            position: 0.0,
            max_position: 100.0,
            is_warmup: false,
            cascade_size_factor: 1.0,
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
}
