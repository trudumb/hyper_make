//! Shared types for the closed-loop learning system.
//!
//! Core insight: Every fill is a prediction that gets scored. This module defines
//! the types for tracking predictions, outcomes, and market state.

use std::collections::VecDeque;

use crate::market_maker::fills::FillEvent;

/// Market state at a point in time - used for prediction context.
///
/// This captures the observable state that edge models use to make predictions.
#[derive(Debug, Clone)]
pub struct MarketState {
    /// Current timestamp in milliseconds
    pub timestamp_ms: u64,

    // === Price state ===
    /// Current microprice estimate
    pub microprice: f64,
    /// Market mid price
    pub market_mid: f64,

    // === Volatility state ===
    /// Clean volatility (bipower)
    pub sigma: f64,
    /// Total volatility including jumps
    pub sigma_total: f64,
    /// Effective volatility for quoting
    pub sigma_effective: f64,

    // === Order flow state ===
    /// Order flow intensity from book depth
    pub kappa: f64,
    /// Bid-side kappa
    pub kappa_bid: f64,
    /// Ask-side kappa
    pub kappa_ask: f64,

    // === Imbalance state ===
    /// Book imbalance [-1, 1]
    pub book_imbalance: f64,
    /// Flow imbalance [-1, 1]
    pub flow_imbalance: f64,
    /// Momentum in basis points
    pub momentum_bps: f64,

    // === Regime state ===
    /// Probability of informed flow
    pub p_informed: f64,
    /// Toxicity score [0, 1]
    pub toxicity_score: f64,
    /// Jump ratio (RV/BV)
    pub jump_ratio: f64,
    /// Volatility regime
    pub volatility_regime: VolatilityRegime,

    // === Adverse selection state ===
    /// Predicted adverse selection in bps
    pub predicted_as_bps: f64,
    /// AS from permanent price impact
    pub as_permanent_bps: f64,
    /// AS from temporary impact
    pub as_temporary_bps: f64,

    // === Funding state ===
    /// Current funding rate
    pub funding_rate: f64,
    /// Predicted funding cost
    pub predicted_funding_cost: f64,

    // === Position state ===
    /// Current position
    pub position: f64,
    /// Max position allowed
    pub max_position: f64,

    // === Cross-asset signals (optional) ===
    /// Cross-asset signal if available
    pub cross_signal: Option<CrossSignal>,
}

impl Default for MarketState {
    fn default() -> Self {
        Self {
            timestamp_ms: 0,
            microprice: 0.0,
            market_mid: 0.0,
            sigma: 0.01,
            sigma_total: 0.01,
            sigma_effective: 0.01,
            kappa: 1.0,
            kappa_bid: 1.0,
            kappa_ask: 1.0,
            book_imbalance: 0.0,
            flow_imbalance: 0.0,
            momentum_bps: 0.0,
            p_informed: 0.1,
            toxicity_score: 0.0,
            jump_ratio: 1.0,
            volatility_regime: VolatilityRegime::Normal,
            predicted_as_bps: 0.0,
            as_permanent_bps: 0.0,
            as_temporary_bps: 0.0,
            funding_rate: 0.0,
            predicted_funding_cost: 0.0,
            position: 0.0,
            max_position: 1.0,
            cross_signal: None,
        }
    }
}

impl MarketState {
    /// Add cross-asset signal to the state.
    pub fn with_cross_signal(mut self, signal: CrossSignal) -> Self {
        self.cross_signal = Some(signal);
        self
    }
}

/// Cross-asset signal from lead-lag relationships.
#[derive(Debug, Clone, Default)]
pub struct CrossSignal {
    /// Expected price move from leader asset
    pub expected_move_bps: f64,
    /// Confidence in the signal [0, 1]
    pub confidence: f64,
    /// Leader asset name (e.g., "BTC")
    pub leader: String,
    /// Time since leader move in ms
    pub age_ms: u64,
}

/// Volatility regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VolatilityRegime {
    Low,
    #[default]
    Normal,
    High,
    Extreme,
}

/// A prediction that is pending outcome measurement.
#[derive(Debug, Clone)]
pub struct PendingPrediction {
    /// Timestamp when prediction was made
    pub timestamp_ms: u64,
    /// The fill that triggered this prediction
    pub fill: FillEvent,
    /// Predicted edge in basis points
    pub predicted_edge_bps: f64,
    /// Uncertainty in the prediction
    pub predicted_uncertainty: f64,
    /// Market state at prediction time
    pub state: MarketState,
    /// Depth in bps from touch at fill
    pub depth_bps: f64,
    /// Predicted fill probability for this depth
    pub predicted_fill_prob: f64,
}

/// Trading outcome after prediction horizon elapsed.
#[derive(Debug, Clone)]
pub struct TradingOutcome {
    /// Original prediction
    pub prediction: PendingPrediction,
    /// Realized adverse selection in bps
    pub realized_as_bps: f64,
    /// Realized edge in bps (spread_captured - realized_as - fees)
    pub realized_edge_bps: f64,
    /// Price at measurement horizon
    pub price_at_horizon: f64,
    /// Time elapsed from fill to horizon
    pub horizon_elapsed_ms: u64,
}

/// Model health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Health {
    Good,
    Warning,
    Degraded,
}

impl Default for Health {
    fn default() -> Self {
        Self::Good
    }
}

/// Overall model health assessment.
#[derive(Debug, Clone, Default)]
pub struct ModelHealth {
    /// Volatility model health
    pub volatility: Health,
    /// Adverse selection model health
    pub adverse_selection: Health,
    /// Fill rate model health
    pub fill_rate: Health,
    /// Edge prediction health
    pub edge: Health,
    /// Edge calibration ratio (|mean_bias| / prediction_rmse), 0.0 when insufficient data
    pub edge_calibration_ratio: f64,
    /// Overall health (most conservative)
    pub overall: Health,
}

impl ModelHealth {
    /// Check if any component is degraded.
    pub fn is_degraded(&self) -> bool {
        self.overall == Health::Degraded
    }

    /// Compute overall health from components.
    pub fn update_overall(&mut self) {
        let healths = [
            self.volatility,
            self.adverse_selection,
            self.fill_rate,
            self.edge,
        ];

        // Overall is the worst of all components
        self.overall = if healths.contains(&Health::Degraded) {
            Health::Degraded
        } else if healths.contains(&Health::Warning) {
            Health::Warning
        } else {
            Health::Good
        };
    }

    /// Convert overall health to a score [0.0, 1.0].
    ///
    /// - Good: 1.0 (full trust)
    /// - Warning: 0.5 (partial trust)
    /// - Degraded: 0.0 (no trust)
    pub fn to_score(&self) -> f64 {
        match self.overall {
            Health::Good => 1.0,
            Health::Warning => 0.5,
            Health::Degraded => 0.0,
        }
    }
}

/// Calibration score for model predictions.
#[derive(Debug, Clone, Default)]
pub struct CalibrationScore {
    /// Root mean square error between predicted and realized
    pub error: f64,
    /// Mean bias (positive = overestimating)
    pub bias: f64,
    /// Sharpness (average predicted uncertainty)
    pub sharpness: f64,
    /// Number of observations
    pub n_observations: usize,
}

impl CalibrationScore {
    /// Check if calibration is acceptable.
    pub fn is_calibrated(&self, max_error: f64) -> bool {
        self.error < max_error && self.n_observations >= 20
    }
}

/// Ring buffer for storing recent predictions.
#[derive(Debug, Clone)]
pub struct RingBuffer<T> {
    items: VecDeque<T>,
    capacity: usize,
}

impl<T> RingBuffer<T> {
    /// Create a new ring buffer with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            items: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Push an item, evicting oldest if at capacity.
    pub fn push(&mut self, item: T) {
        if self.items.len() >= self.capacity {
            self.items.pop_front();
        }
        self.items.push_back(item);
    }

    /// Get number of items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Iterate over items.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }

    /// Clear all items.
    pub fn clear(&mut self) {
        self.items.clear();
    }
}

impl<T> Default for RingBuffer<T> {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Edge prediction from a single model.
#[derive(Debug, Clone)]
pub struct WeightedPrediction {
    /// Model name
    pub model: String,
    /// Predicted edge mean
    pub mean: f64,
    /// Predicted edge std
    pub std: f64,
    /// Weight of this model in ensemble
    pub weight: f64,
}

/// Ensemble prediction combining multiple models.
#[derive(Debug, Clone)]
pub struct EnsemblePrediction {
    /// Combined mean (weighted average)
    pub mean: f64,
    /// Combined std (from law of total variance)
    pub std: f64,
    /// Disagreement between models (std of means)
    pub disagreement: f64,
    /// Individual model contributions
    pub model_contributions: Vec<WeightedPrediction>,
}

impl Default for EnsemblePrediction {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std: 1.0,
            disagreement: 0.0,
            model_contributions: Vec::new(),
        }
    }
}

/// Quote decision from the decision engine.
#[derive(Debug, Clone)]
pub enum QuoteDecision {
    /// Don't quote
    NoQuote {
        /// Reason for not quoting
        reason: String,
    },
    /// Quote with reduced size
    ReducedSize {
        /// Size fraction to use (0-1)
        fraction: f64,
        /// Reason for size reduction
        reason: String,
    },
    /// Normal quote
    Quote {
        /// Size fraction from inverse information asymmetry (0-1)
        /// Higher when p ≈ 0.5 (uncertain), lower at extremes (informed flow)
        size_fraction: f64,
        /// Model confidence - HIGH when p ≈ 0.5 (spread capture dominates)
        /// This is inverse of traditional "confidence" - we're confident when uncertain about direction
        confidence: f64,
        /// Expected edge in bps (drift estimate μ)
        expected_edge: f64,
        /// Reservation price shift in price units (from A-S: μ/(γσ²))
        /// Positive = shift quotes up (aggressive asks), Negative = shift down (aggressive bids)
        reservation_shift: f64,
        // NOTE: spread_multiplier has been REMOVED. All uncertainty is now handled
        // through gamma scaling (kappa_ci_width flows through uncertainty_scalar).
        // The GLFT formula naturally widens spreads when gamma increases due to uncertainty.
    },
}

impl QuoteDecision {
    /// Check if this is a quote decision.
    pub fn is_quote(&self) -> bool {
        matches!(self, Self::Quote { .. } | Self::ReducedSize { .. })
    }

    /// Get effective size fraction.
    pub fn size_fraction(&self) -> f64 {
        match self {
            Self::Quote { size_fraction, .. } => *size_fraction,
            Self::ReducedSize { fraction, .. } => *fraction,
            Self::NoQuote { .. } => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(3);
        buf.push(1);
        buf.push(2);
        buf.push(3);
        assert_eq!(buf.len(), 3);

        buf.push(4);
        assert_eq!(buf.len(), 3);
        let items: Vec<_> = buf.iter().copied().collect();
        assert_eq!(items, vec![2, 3, 4]);
    }

    #[test]
    fn test_model_health() {
        let mut health = ModelHealth::default();
        assert_eq!(health.overall, Health::Good);

        health.volatility = Health::Warning;
        health.update_overall();
        assert_eq!(health.overall, Health::Warning);

        health.edge = Health::Degraded;
        health.update_overall();
        assert_eq!(health.overall, Health::Degraded);
    }

    #[test]
    fn test_quote_decision() {
        let quote = QuoteDecision::Quote {
            size_fraction: 0.5,
            confidence: 0.8,
            expected_edge: 2.0,
            reservation_shift: 0.001,
            // NOTE: spread_multiplier removed - uncertainty flows through gamma
        };
        assert!(quote.is_quote());
        assert_eq!(quote.size_fraction(), 0.5);

        let no_quote = QuoteDecision::NoQuote {
            reason: "test".into(),
        };
        assert!(!no_quote.is_quote());
        assert_eq!(no_quote.size_fraction(), 0.0);
    }
}
