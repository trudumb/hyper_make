//! Margin constraint parameter set.

/// Margin and leverage constraints.
#[derive(Debug, Clone)]
pub struct MarginConstraintParams {
    /// Whether to use constrained ladder optimizer.
    pub use_constrained_optimizer: bool,

    /// Available margin for order placement (USD).
    pub margin_available: f64,

    /// Current leverage ratio.
    pub leverage: f64,
}

impl Default for MarginConstraintParams {
    fn default() -> Self {
        Self {
            use_constrained_optimizer: true, // Enable for entropy-based allocation
            margin_available: 0.0,
            leverage: 1.0,
        }
    }
}
