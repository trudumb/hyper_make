//! Stochastic constraint parameter set.

/// Stochastic constraint parameters extracted from MarketParams.
///
/// Contains first-principles constraints for safe tight spread quoting:
/// - Tick size constraint
/// - Latency-based spread floor
/// - Book depth thresholds
/// - Conditional tight quoting logic
#[derive(Debug, Clone, Copy)]
pub struct StochasticConstraintParams {
    /// Asset tick size in basis points.
    /// Spread floor must be >= tick_size_bps.
    pub tick_size_bps: f64,

    /// Latency-aware spread floor: δ_min = σ × √(2×τ_update).
    /// In fractional terms (multiply by 10000 for bps).
    pub latency_spread_floor: f64,

    /// Near-touch book depth (USD) within 5 bps of mid.
    pub near_touch_depth_usd: f64,

    /// Whether tight quoting is currently allowed.
    pub tight_quoting_allowed: bool,

    /// Combined spread widening factor [1.0, 2.0+].
    pub stochastic_spread_multiplier: f64,
}

impl Default for StochasticConstraintParams {
    fn default() -> Self {
        Self {
            tick_size_bps: 10.0,
            latency_spread_floor: 0.0003,
            near_touch_depth_usd: 0.0,
            tight_quoting_allowed: false,
            stochastic_spread_multiplier: 1.0,
        }
    }
}

impl StochasticConstraintParams {
    /// Get effective spread floor as fraction.
    pub fn effective_floor(&self, risk_config_floor: f64) -> f64 {
        let tick_floor = self.tick_size_bps / 10_000.0;
        risk_config_floor
            .max(tick_floor)
            .max(self.latency_spread_floor)
    }

    /// Check if constraints allow tight spreads.
    pub fn can_quote_tight(&self) -> bool {
        self.tight_quoting_allowed && self.stochastic_spread_multiplier < 1.2
    }
}
