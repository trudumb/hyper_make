//! Cascade parameter set.

/// Liquidation cascade detection and response.
#[derive(Debug, Clone)]
pub struct CascadeParams {
    /// Tail risk multiplier for gamma [1.0, 5.0].
    /// Multiply gamma by this during cascade conditions.
    pub tail_risk_intensity: f64,

    /// Whether extreme cascades necessitate pulling all quotes.
    pub should_pull_quotes: bool,

    /// Gradual size reduction intensity [0, 1].
    /// 1.0 = full size, 0.0 = no quotes (cascade severe).
    pub cascade_intensity: f64,
}

impl Default for CascadeParams {
    fn default() -> Self {
        Self {
            tail_risk_intensity: 0.0,
            should_pull_quotes: false,
            cascade_intensity: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cascade_params_default() {
        let params = CascadeParams::default();
        assert!(!params.should_pull_quotes);
        assert!(params.tail_risk_intensity.abs() < f64::EPSILON);
    }
}
