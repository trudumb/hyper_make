//! Spread profile configuration for different market types.

/// Spread profile for different market types.
///
/// Controls kappa and gamma settings based on target spreads:
/// - `Default`: 40-50 bps for liquid perps (BTC, ETH)
/// - `Hip3`: 15-25 bps for HIP-3 DEX (tighter spreads, less AS)
/// - `Aggressive`: 10-20 bps for extremely tight quoting (experimental)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SpreadProfile {
    /// Default profile for liquid perpetuals (40-50 bps target).
    /// Uses gamma_base=0.3, kappa_prior=500.
    #[default]
    Default,

    /// HIP-3 DEX profile for tighter spreads (15-25 bps target).
    /// Uses gamma_base=0.15, kappa_prior=1500.
    /// Disables book depth and time-of-day scaling (inappropriate for HIP-3).
    Hip3,

    /// Aggressive profile for extremely tight spreads (10-20 bps target).
    /// Uses gamma_base=0.10, kappa_prior=2000.
    /// EXPERIMENTAL - use with caution.
    Aggressive,
}

impl SpreadProfile {
    /// Parse from string (CLI argument).
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "hip3" | "hip-3" => Self::Hip3,
            "aggressive" | "tight" => Self::Aggressive,
            _ => Self::Default,
        }
    }

    /// Check if this profile is for HIP-3 DEX.
    pub fn is_hip3(&self) -> bool {
        matches!(self, Self::Hip3)
    }

    /// Static safety bound: minimum half-spread in bps for this profile.
    ///
    /// Acts as a constraint that prevents disaster while the adaptive AS
    /// posterior warms up. NOT the optimal policy — just a floor.
    /// Returns 0.0 for Default (no profile floor).
    pub fn profile_min_half_spread_bps(&self) -> f64 {
        match self {
            Self::Default => 0.0,    // No profile floor for liquid perps
            Self::Hip3 => 7.5,       // 15 bps total minimum for HIP-3
            Self::Aggressive => 5.0, // 10 bps total minimum
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_min_half_spread_per_type() {
        assert_eq!(SpreadProfile::Default.profile_min_half_spread_bps(), 0.0);
        assert_eq!(SpreadProfile::Hip3.profile_min_half_spread_bps(), 7.5);
        assert_eq!(SpreadProfile::Aggressive.profile_min_half_spread_bps(), 5.0);
    }

    #[test]
    fn test_from_str_parsing() {
        assert_eq!(SpreadProfile::from_str("hip3"), SpreadProfile::Hip3);
        assert_eq!(SpreadProfile::from_str("hip-3"), SpreadProfile::Hip3);
        assert_eq!(
            SpreadProfile::from_str("aggressive"),
            SpreadProfile::Aggressive
        );
        assert_eq!(SpreadProfile::from_str("tight"), SpreadProfile::Aggressive);
        assert_eq!(SpreadProfile::from_str("anything"), SpreadProfile::Default);
    }
}
