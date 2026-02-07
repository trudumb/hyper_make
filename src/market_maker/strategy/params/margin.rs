//! Margin constraint parameter set.

/// Margin and leverage constraints.
#[derive(Debug, Clone)]
pub struct MarginConstraintParams {
    /// Available margin for order placement (USD).
    pub margin_available: f64,

    /// Current leverage ratio.
    pub leverage: f64,
}

impl Default for MarginConstraintParams {
    fn default() -> Self {
        Self {
            margin_available: 0.0,
            leverage: 1.0,
        }
    }
}
