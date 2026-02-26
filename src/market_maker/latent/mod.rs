//! Latent State Estimation Module
//!
//! This module provides components for inferring latent market state
//! from observable market data. The latent state space includes:
//!
//! - `σ(t)`: Local volatility with regime switching
//! - `π_informed(t)`: Probability next trade is informed
//! - `π_forced(t)`: Probability next trade is liquidation/rebalance
//! - `regime(t)`: Discrete market state (LOW, NORMAL, HIGH volatility)
//!
//! ## Architecture
//!
//! ```text
//! Observations (Trades, L2Book, AllMids)
//!     ↓
//! Derived Processes (Volatility, Flow, Book Dynamics)
//!     ↓
//! Latent State Inference (JointDynamics)
//!     ↓
//! Edge Quantification (EdgeSurface)
//! ```
//!
//! ## Components
//!
//! - [`JointDynamics`]: Models correlated evolution of latent parameters
//! - [`EdgeSurface`]: Quantifies expected edge by market condition
//!
//! ## Usage
//!
//! These components are used by the `parameter_estimator` binary to
//! analyze market conditions and estimate edge for market making.

// Submodules
pub mod edge_surface;
pub mod joint_dynamics;

// Re-exports
pub use edge_surface::{EdgeObservation, EdgeSurface, EdgeSurfaceConfig, SurfaceStatistics};
pub use joint_dynamics::{
    JointDynamics, JointDynamicsConfig, JointObservation, LatentStateEstimate,
    ParameterCorrelations, StateCovariance,
};

/// Latent market state representation
#[derive(Debug, Clone, Default)]
pub struct LatentState {
    /// Current volatility estimate (bps/sqrt(s))
    pub sigma: f64,

    /// Current regime (0=LOW, 1=NORMAL, 2=HIGH)
    pub regime: u8,

    /// Probability next trade is informed (0.0-1.0)
    pub p_informed: f64,

    /// Probability next trade is forced/liquidation (0.0-1.0)
    pub p_forced: f64,

    /// Flow momentum (-1.0 to 1.0, negative=selling pressure)
    pub flow_momentum: f64,

    /// Book pressure (positive=bid pressure, negative=ask pressure)
    pub book_pressure: f64,

    /// Confidence in state estimate (0.0-1.0)
    pub confidence: f64,
}

impl LatentState {
    /// Create a new latent state with default values
    pub fn new() -> Self {
        Self {
            sigma: 10.0,      // 10 bps/sqrt(s) typical
            regime: 1,        // NORMAL
            p_informed: 0.05, // 5% baseline informed
            p_forced: 0.02,   // 2% baseline forced
            flow_momentum: 0.0,
            book_pressure: 0.0,
            confidence: 0.0,
        }
    }

    /// Check if this is a high-risk state (high vol or high informed flow)
    pub fn is_high_risk(&self) -> bool {
        self.regime == 2 || self.p_informed > 0.20
    }

    /// Check if state is reliable (sufficient confidence)
    pub fn is_reliable(&self) -> bool {
        self.confidence > 0.5
    }
}

/// Edge estimate with uncertainty
#[derive(Debug, Clone, Default)]
pub struct EdgeEstimate {
    /// Expected edge in basis points
    pub edge_bps: f64,

    /// Uncertainty (standard deviation) in basis points
    pub uncertainty_bps: f64,

    /// Confidence that edge > 0 (probability)
    pub confidence: f64,

    /// Optimal spread for this state
    pub optimal_spread_bps: f64,

    /// Expected fill rate at optimal spread
    pub expected_fill_rate: f64,

    /// Expected adverse selection at this state
    pub expected_as_bps: f64,
}

impl EdgeEstimate {
    /// Check if edge is statistically significant (> 2σ)
    pub fn is_significant(&self) -> bool {
        self.edge_bps > 2.0 * self.uncertainty_bps
    }

    /// Check if we should quote given this edge estimate
    pub fn should_quote(&self) -> bool {
        self.is_significant() && self.confidence > 0.8
    }
}

/// Market condition for edge surface lookup
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct MarketCondition {
    /// Volatility bucket (0-4)
    pub vol_bucket: u8,

    /// Regime (0=LOW, 1=NORMAL, 2=HIGH)
    pub regime: u8,

    /// Hour bucket (0=Asia, 1=London, 2=US)
    pub hour_bucket: u8,

    /// Flow bucket (0-4, 0=strong sell, 4=strong buy)
    pub flow_bucket: u8,
}

impl MarketCondition {
    /// Create from raw values with bucketing
    pub fn from_state(sigma: f64, regime: u8, hour_utc: u8, flow: f64) -> Self {
        // Volatility buckets: <5, 5-10, 10-20, 20-40, >40 bps/sqrt(s)
        let vol_bucket = if sigma < 5.0 {
            0
        } else if sigma < 10.0 {
            1
        } else if sigma < 20.0 {
            2
        } else if sigma < 40.0 {
            3
        } else {
            4
        };

        // Hour buckets: Asia (0-8 UTC), London (8-16 UTC), US (16-24 UTC)
        let hour_bucket = if hour_utc < 8 {
            0 // Asia
        } else if hour_utc < 16 {
            1 // London
        } else {
            2 // US
        };

        // Flow buckets: strong sell, weak sell, neutral, weak buy, strong buy
        let flow_bucket = if flow < -0.3 {
            0
        } else if flow < -0.1 {
            1
        } else if flow < 0.1 {
            2
        } else if flow < 0.3 {
            3
        } else {
            4
        };

        Self {
            vol_bucket,
            regime,
            hour_bucket,
            flow_bucket,
        }
    }

    /// Grid index for array lookup (225 cells total)
    pub fn grid_index(&self) -> usize {
        let vol = self.vol_bucket as usize;
        let regime = self.regime as usize;
        let hour = self.hour_bucket as usize;
        let flow = self.flow_bucket as usize;

        // 5 vol × 3 regime × 3 hour × 5 flow = 225
        vol * 45 + regime * 15 + hour * 5 + flow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latent_state_new() {
        let state = LatentState::new();
        assert_eq!(state.regime, 1);
        assert!(!state.is_high_risk());
        assert!(!state.is_reliable());
    }

    #[test]
    fn test_latent_state_high_risk() {
        let mut state = LatentState::new();
        state.regime = 2; // HIGH
        assert!(state.is_high_risk());

        state.regime = 1;
        state.p_informed = 0.25;
        assert!(state.is_high_risk());
    }

    #[test]
    fn test_edge_estimate_significance() {
        let mut est = EdgeEstimate {
            edge_bps: 3.0,
            uncertainty_bps: 1.0,
            confidence: 0.9,
            ..Default::default()
        };
        assert!(est.is_significant()); // 3 > 2*1
        assert!(est.should_quote());

        est.edge_bps = 1.5;
        assert!(!est.is_significant()); // 1.5 < 2*1
    }

    #[test]
    fn test_market_condition_from_state() {
        let cond = MarketCondition::from_state(15.0, 1, 10, 0.0);
        assert_eq!(cond.vol_bucket, 2); // 10-20 bps
        assert_eq!(cond.regime, 1); // NORMAL
        assert_eq!(cond.hour_bucket, 1); // London
        assert_eq!(cond.flow_bucket, 2); // Neutral
    }

    #[test]
    fn test_market_condition_grid_index() {
        // Test corner cases
        let cond1 = MarketCondition {
            vol_bucket: 0,
            regime: 0,
            hour_bucket: 0,
            flow_bucket: 0,
        };
        assert_eq!(cond1.grid_index(), 0);

        let cond2 = MarketCondition {
            vol_bucket: 4,
            regime: 2,
            hour_bucket: 2,
            flow_bucket: 4,
        };
        assert_eq!(cond2.grid_index(), 224); // Last cell
    }
}
