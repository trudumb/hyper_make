//! Layer 1: Observer - State Estimation.
//!
//! This layer derives hidden states from market data:
//! - σ_t: Volatility regime (Quiet, Normal, Turbulent)
//! - π_t: Queue priority index [0, 1]
//! - α_t: Flow toxicity indicator

mod state_estimator;
mod toxicity;
mod virtual_queue;

pub use state_estimator::{MarketState, StateEstimator, VolatilityState};
pub use toxicity::ToxicityEstimator;
pub use virtual_queue::{OrderPriority, PriorityClass, VirtualQueueTracker};
