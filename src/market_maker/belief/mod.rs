//! Centralized Belief State Module
//!
//! This module provides a **single source of truth** for all Bayesian beliefs
//! in the market making system. It replaces 6 fragmented belief modules:
//!
//! 1. `MarketBeliefs` (stochastic/beliefs.rs) - drift/vol/kappa posteriors
//! 2. `BeliefState` (control/belief.rs) - fill rate, AS, edge
//! 3. `ContinuationPosterior` (stochastic/continuation.rs) - position continuation
//! 4. `ChangepointDetector` (control/changepoint.rs) - BOCD
//! 5. `RegimeHMM` (estimator/regime_hmm.rs) - 4-regime HMM
//! 6. `KappaOrchestrator` (estimator/kappa_orchestrator.rs) - blended kappa
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                     CentralBeliefState                           │
//! │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                │
//! │  │ DriftVol    │ │ Kappa       │ │ Continuation│                │
//! │  │ Beliefs     │ │ Beliefs     │ │ Beliefs     │                │
//! │  └─────────────┘ └─────────────┘ └─────────────┘                │
//! │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                │
//! │  │ Regime      │ │ Changepoint │ │ Edge        │                │
//! │  │ Beliefs     │ │ Beliefs     │ │ Beliefs     │                │
//! │  └─────────────┘ └─────────────┘ └─────────────┘                │
//! │  ┌─────────────────────────────────────────────┐                │
//! │  │           CalibrationState                   │                │
//! │  └─────────────────────────────────────────────┘                │
//! └──────────────────────────────────────────────────────────────────┘
//!                           │
//!                           ▼ snapshot()
//!               ┌───────────────────────┐
//!               │     BeliefSnapshot    │ ← Consumers read this
//!               └───────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! // In orchestrator initialization:
//! let central_beliefs = CentralBeliefState::new(config);
//!
//! // Publishers send updates:
//! central_beliefs.update(BeliefUpdate::PriceReturn {
//!     return_frac: 0.001,
//!     dt_secs: 1.0,
//!     timestamp_ms: now_ms(),
//! });
//!
//! // Consumers read snapshots:
//! let snapshot = central_beliefs.snapshot();
//! let kappa = snapshot.kappa.kappa_effective;
//! let regime = snapshot.regime.current;
//! ```
//!
//! ## Benefits
//!
//! 1. **Single source of truth** - One place to query all beliefs
//! 2. **Integrated calibration** - Know which models actually work
//! 3. **Reduced threshold spaghetti** - 70-80% fewer if-else branches
//! 4. **Easy testing** - Mock BeliefSnapshot instead of 10 modules
//! 5. **Clean interface for new signals** - Just add to BeliefUpdate enum
//!
//! ## Migration Status (Phase 8 Complete)
//!
//! This module is now the **primary** belief system. The following are deprecated:
//!
//! - `stochastic::StochasticControlBuilder` - No longer updated with price observations
//! - `stochastic::beliefs::MarketBeliefs` - Use `BeliefSnapshot.drift_vol` instead
//!
//! The following remain in use for their specialized purposes:
//!
//! - `control::belief::BeliefState` - Model ensemble weights (different purpose)
//! - `estimator::RegimeHMM` - Still provides regime probabilities (forwarded here)
//! - `control::changepoint` - Still provides BOCD (forwarded here)

pub mod bayesian_fair_value;
mod central;
mod messages;
pub mod posterior;
mod publisher;
mod snapshot;

pub use bayesian_fair_value::{
    BayesianFairValue, BayesianFairValueCheckpoint, BayesianFairValueConfig, FairValueBeliefs,
};
pub use central::{CentralBeliefConfig, CentralBeliefState};
pub use messages::{BeliefUpdate, PredictionLog, PredictionType};
pub use posterior::{
    EmergencyAction, EmergencyThresholds, EmergencyTrigger, EventPosteriorConfig,
    EventPosteriorState, PosteriorSnapshot,
};
pub use publisher::{BeliefPublisher, PublishError, PublisherHandle};
pub use snapshot::{
    BeliefSnapshot, BeliefStats, CalibrationMetrics, CalibrationState, ChangepointBeliefs,
    ChangepointResult, ContinuationBeliefs, ContinuationSignals, DriftVolatilityBeliefs,
    EdgeBeliefs, KappaBeliefs, KappaComponents, RegimeBeliefs,
};

/// Regime enumeration for belief state (matches stochastic/beliefs.rs)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Regime {
    /// Low volatility, normal fills
    Quiet,
    /// Normal market conditions
    #[default]
    Normal,
    /// High activity, elevated volatility
    Bursty,
    /// Liquidation cascade, toxic flow
    Cascade,
}

impl Regime {
    /// Get regime index (0-3)
    pub fn index(&self) -> usize {
        match self {
            Regime::Quiet => 0,
            Regime::Normal => 1,
            Regime::Bursty => 2,
            Regime::Cascade => 3,
        }
    }

    /// Create from index
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Regime::Quiet,
            1 => Regime::Normal,
            2 => Regime::Bursty,
            3 => Regime::Cascade,
            _ => Regime::Normal,
        }
    }

    /// Create from probability array (returns most likely)
    pub fn from_probs(probs: &[f64; 4]) -> Self {
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(1);
        Self::from_index(max_idx)
    }
}

impl std::fmt::Display for Regime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Regime::Quiet => write!(f, "quiet"),
            Regime::Normal => write!(f, "normal"),
            Regime::Bursty => write!(f, "bursty"),
            Regime::Cascade => write!(f, "cascade"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_from_index() {
        assert_eq!(Regime::from_index(0), Regime::Quiet);
        assert_eq!(Regime::from_index(1), Regime::Normal);
        assert_eq!(Regime::from_index(2), Regime::Bursty);
        assert_eq!(Regime::from_index(3), Regime::Cascade);
        assert_eq!(Regime::from_index(99), Regime::Normal); // fallback
    }

    #[test]
    fn test_regime_from_probs() {
        assert_eq!(Regime::from_probs(&[0.8, 0.1, 0.05, 0.05]), Regime::Quiet);
        assert_eq!(Regime::from_probs(&[0.1, 0.7, 0.1, 0.1]), Regime::Normal);
        assert_eq!(Regime::from_probs(&[0.1, 0.1, 0.7, 0.1]), Regime::Bursty);
        assert_eq!(Regime::from_probs(&[0.1, 0.1, 0.1, 0.7]), Regime::Cascade);
    }
}
