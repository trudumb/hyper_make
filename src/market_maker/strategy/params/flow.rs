//! Flow parameter set.

/// Order flow and momentum signals.
#[derive(Debug, Clone, Default)]
pub struct FlowParams {
    /// Order flow imbalance [-1, 1].
    /// Negative = sell pressure, Positive = buy pressure.
    pub flow_imbalance: f64,

    /// L2 book imbalance [-1, 1].
    /// Positive = more bids, Negative = more asks.
    pub book_imbalance: f64,

    /// Signed momentum over 500ms window (in bps).
    /// Negative = market falling, Positive = market rising.
    pub momentum_bps: f64,

    /// Falling knife score [0, 3].
    /// > 0.5 = some downward momentum, > 1.0 = severe (protect bids!).
    pub falling_knife_score: f64,

    /// Rising knife score [0, 3].
    /// > 0.5 = some upward momentum, > 1.0 = severe (protect asks!).
    pub rising_knife_score: f64,

    // Hawkes order flow (Tier 2)
    /// Hawkes buy intensity (λ_buy) - self-exciting arrival rate.
    pub hawkes_buy_intensity: f64,
    /// Hawkes sell intensity (λ_sell) - self-exciting arrival rate.
    pub hawkes_sell_intensity: f64,
    /// Hawkes flow imbalance [-1, 1].
    pub hawkes_imbalance: f64,
    /// Hawkes activity percentile [0, 1].
    pub hawkes_activity_percentile: f64,

    // Momentum protection (Gap 10)
    /// Bid protection factor (>= 1.0 when market falling).
    pub bid_protection_factor: f64,
    /// Ask protection factor (>= 1.0 when market rising).
    pub ask_protection_factor: f64,
    /// Probability momentum continues.
    pub p_momentum_continue: f64,
}

impl FlowParams {
    /// Create with default protection factors.
    pub fn new() -> Self {
        Self {
            bid_protection_factor: 1.0,
            ask_protection_factor: 1.0,
            p_momentum_continue: 0.5,
            hawkes_activity_percentile: 0.5,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_params_new() {
        let params = FlowParams::new();
        assert!((params.bid_protection_factor - 1.0).abs() < f64::EPSILON);
        assert!((params.ask_protection_factor - 1.0).abs() < f64::EPSILON);
    }
}
