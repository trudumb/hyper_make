//! HJB controller parameter set.

/// Stochastic inventory control from HJB solution.
#[derive(Debug, Clone)]
pub struct HJBParams {
    /// Whether to use HJB optimal skew instead of heuristic.
    pub use_hjb_skew: bool,

    /// HJB optimal inventory skew (from Avellaneda-Stoikov HJB solution).
    /// Formula: γσ²qT + terminal_penalty × q × urgency + funding_bias.
    pub hjb_optimal_skew: f64,

    /// HJB gamma multiplier (for logging/diagnostics).
    pub hjb_gamma_multiplier: f64,

    /// HJB inventory target (optimal q* for current session state).
    pub hjb_inventory_target: f64,

    /// Whether HJB controller is in terminal zone (near session end).
    pub hjb_is_terminal_zone: bool,
}

impl Default for HJBParams {
    fn default() -> Self {
        Self {
            use_hjb_skew: false,
            hjb_optimal_skew: 0.0,
            hjb_gamma_multiplier: 1.0,
            hjb_inventory_target: 0.0,
            hjb_is_terminal_zone: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hjb_params_default() {
        let params = HJBParams::default();
        assert!(!params.use_hjb_skew);
        assert!(!params.hjb_is_terminal_zone);
    }
}
