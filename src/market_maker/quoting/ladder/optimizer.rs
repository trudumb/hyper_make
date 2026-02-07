//! Ladder optimization types and constraints.
//!
//! This module provides the core types used by `EntropyConstrainedOptimizer`
//! for ladder size allocation.

/// Parameters for a single ladder level used in optimization.
#[derive(Debug, Clone)]
pub struct LevelOptimizationParams {
    /// Depth in basis points
    pub depth_bps: f64,
    /// Fill intensity λ(δ) at this depth (used by proportional allocation)
    pub fill_intensity: f64,
    /// Spread capture SC(δ) at this depth
    pub spread_capture: f64,
    /// Margin required per unit size at this level
    pub margin_per_unit: f64,
    /// Adverse selection at this depth (bps)
    pub adverse_selection: f64,
}

/// Result of constrained optimization.
#[derive(Debug, Clone)]
pub struct ConstrainedAllocation {
    /// Optimal sizes per level
    pub sizes: Vec<f64>,
    /// Shadow price λ* (marginal value at binding constraint)
    /// Economic interpretation: return per unit of relaxed constraint
    pub shadow_price: f64,
    /// Total margin used
    pub margin_used: f64,
    /// Total position allocated
    pub position_used: f64,
    /// Which constraint is binding
    pub binding_constraint: BindingConstraint,
}

/// Which constraint binds in the optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingConstraint {
    /// No constraint binding (all levels allocated)
    None,
    /// Margin constraint binding
    Margin,
    /// Position constraint binding
    Position,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_optimization_params() {
        let params = LevelOptimizationParams {
            depth_bps: 5.0,
            fill_intensity: 10.0,
            spread_capture: 2.0,
            margin_per_unit: 10.0,
            adverse_selection: 1.0,
        };
        assert_eq!(params.depth_bps, 5.0);
        assert_eq!(params.fill_intensity, 10.0);
    }

    #[test]
    fn test_constrained_allocation() {
        let alloc = ConstrainedAllocation {
            sizes: vec![0.1, 0.2, 0.3],
            shadow_price: 0.5,
            margin_used: 100.0,
            position_used: 0.6,
            binding_constraint: BindingConstraint::Margin,
        };
        assert_eq!(alloc.sizes.len(), 3);
        assert_eq!(alloc.binding_constraint, BindingConstraint::Margin);
    }

    #[test]
    fn test_binding_constraint_equality() {
        assert_eq!(BindingConstraint::None, BindingConstraint::None);
        assert_eq!(BindingConstraint::Margin, BindingConstraint::Margin);
        assert_eq!(BindingConstraint::Position, BindingConstraint::Position);
        assert_ne!(BindingConstraint::None, BindingConstraint::Margin);
    }
}
