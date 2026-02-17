//! Joint parameter evolution with phase-based convergence.
//!
//! The `CalibrationCoordinator` manages kappa/sigma/regime as a coordinated system.
//! It starts conservative (wide spreads from MarketProfile observables) and
//! monotonically tightens as fill evidence accumulates.
//!
//! # Phases
//!
//! | Phase      | Entry      | Kappa Source               | Uncertainty Premium |
//! |------------|------------|---------------------------|-------------------|
//! | Cold       | T=0        | 100% MarketProfile        | 5 bps             |
//! | Warming    | 5 fills    | 70% profile + 30% fills   | 3 bps             |
//! | Calibrated | 30 fills   | 30% profile + 70% fills   | 1 bps             |
//! | Confident  | 100 fills  | 10% profile + 90% fills   | 0 bps             |
//!
//! # Safety
//!
//! If adverse selection rate exceeds 40% while kappa is increasing, kappa growth
//! is clamped (spreads are too tight, getting adversely selected).

use serde::{Deserialize, Serialize};

use super::market_profile::MarketProfile;

/// Phase of calibration convergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CalibrationPhase {
    /// No fills yet. 100% profile-driven, maximum uncertainty premium.
    #[default]
    Cold,
    /// 5-29 fills. Blending profile with early fill data.
    Warming,
    /// 30-99 fills. Primarily fill-driven with profile validation.
    Calibrated,
    /// 100+ fills. Optimal spreads, fill data dominates.
    Confident,
}

impl CalibrationPhase {
    /// Minimum fills to enter this phase.
    #[allow(dead_code)] // Used in Phase 6 pipeline wiring
    fn min_fills(self) -> usize {
        match self {
            Self::Cold => 0,
            Self::Warming => 5,
            Self::Calibrated => 30,
            Self::Confident => 100,
        }
    }

    /// Profile weight for kappa blending.
    #[allow(dead_code)] // Used in Phase 6 pipeline wiring
    fn profile_weight(self) -> f64 {
        match self {
            Self::Cold => 1.0,
            Self::Warming => 0.7,
            Self::Calibrated => 0.3,
            Self::Confident => 0.1,
        }
    }

    /// Maximum uncertainty premium in bps.
    fn max_uncertainty_premium_bps(self) -> f64 {
        match self {
            Self::Cold => 5.0,
            Self::Warming => 3.0,
            Self::Calibrated => 1.0,
            Self::Confident => 0.0,
        }
    }
}

/// Fill count thresholds for phase transitions.
const WARMING_FILLS: usize = 5;
const CALIBRATED_FILLS: usize = 30;
const CONFIDENT_FILLS: usize = 100;

/// Maximum AS rate before kappa growth is clamped.
const MAX_AS_RATE_FOR_TIGHTENING: f64 = 0.40;

/// EWMA alpha for fill-based kappa.
const FILL_KAPPA_ALPHA: f64 = 0.1;

/// EWMA alpha for AS rate tracking.
const AS_RATE_ALPHA: f64 = 0.05;

/// Joint parameter evolution coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationCoordinator {
    /// Current calibration phase.
    #[serde(default)]
    phase: CalibrationPhase,
    /// Profile-derived kappa (set on initialization).
    #[serde(default = "default_kappa")]
    profile_kappa: f64,
    /// Profile-derived sigma.
    #[serde(default = "default_sigma")]
    profile_sigma: f64,
    /// Fill-derived kappa (EWMA of 10000/fill_distance_bps).
    #[serde(default = "default_kappa")]
    fill_kappa: f64,
    /// Fill-derived sigma (from markout observations).
    #[serde(default = "default_sigma")]
    fill_sigma: f64,
    /// Total fill count.
    #[serde(default)]
    fill_count: usize,
    /// EWMA adverse selection rate [0, 1].
    #[serde(default)]
    as_rate: f64,
    /// Whether coordinator has been seeded from a profile.
    #[serde(default)]
    seeded: bool,
    /// Peak fill kappa (for AS clamping).
    #[serde(default)]
    peak_fill_kappa: f64,
}

fn default_kappa() -> f64 { 1000.0 }
fn default_sigma() -> f64 { 0.00025 }

impl Default for CalibrationCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl CalibrationCoordinator {
    /// Create a new coordinator in Cold phase.
    pub fn new() -> Self {
        Self {
            phase: CalibrationPhase::Cold,
            profile_kappa: 1000.0,
            profile_sigma: 0.00025,
            fill_kappa: 1000.0,
            fill_sigma: 0.00025,
            fill_count: 0,
            as_rate: 0.0,
            seeded: false,
            peak_fill_kappa: 0.0,
        }
    }

    /// Seed initial parameters from a MarketProfile.
    ///
    /// Should be called once when the first L2 snapshot arrives.
    pub fn initialize_from_profile(&mut self, profile: &MarketProfile) {
        if !profile.is_initialized() {
            return;
        }
        self.profile_kappa = profile.conservative_kappa();
        self.profile_sigma = profile.implied_sigma();
        // Initialize fill_kappa to profile value (will diverge as fills arrive)
        self.fill_kappa = self.profile_kappa;
        self.fill_sigma = self.profile_sigma;
        self.seeded = true;
    }

    /// Record a fill outcome.
    ///
    /// # Arguments
    /// * `fill_distance_bps` - Distance from mid to fill price in bps
    /// * `was_adversely_selected` - Whether this fill was adversely selected (5s markout)
    pub fn on_fill(&mut self, fill_distance_bps: f64, was_adversely_selected: bool) {
        self.fill_count += 1;

        // Update fill-based kappa: kappa = 10000 / fill_distance_bps
        if fill_distance_bps > 0.1 {
            let fill_kappa_instant = 10_000.0 / fill_distance_bps;
            self.fill_kappa = ewma(self.fill_kappa, fill_kappa_instant, FILL_KAPPA_ALPHA);
        }

        // Track peak fill kappa for AS clamping
        if self.fill_kappa > self.peak_fill_kappa {
            self.peak_fill_kappa = self.fill_kappa;
        }

        // Update AS rate
        let as_indicator = if was_adversely_selected { 1.0 } else { 0.0 };
        self.as_rate = ewma(self.as_rate, as_indicator, AS_RATE_ALPHA);

        // AS safety: if AS rate too high and kappa is increasing, clamp
        if self.as_rate > MAX_AS_RATE_FOR_TIGHTENING && self.fill_kappa > self.profile_kappa {
            // Clamp fill_kappa to profile (widen spreads back)
            self.fill_kappa = self.fill_kappa.min(self.profile_kappa * 1.2);
        }

        // Check phase transition
        self.update_phase();
    }

    /// Effective kappa: continuously-blended with warmup factor.
    ///
    /// Uses a smooth sigmoid-based profile weight that transitions gradually
    /// instead of discrete phase jumps. This ensures monotonic convergence.
    ///
    /// The warmup factor halves kappa at 0 fills (doubling spreads) and converges
    /// to 1.0 as fills accumulate: `0.5 + 0.5 × (1 - exp(-fills/10))`
    pub fn effective_kappa(&self) -> f64 {
        // Smooth profile weight: 1.0 at 0 fills → ~0.1 at 100 fills
        // Uses exponential decay for monotonic transition (no discrete jumps)
        let fills_f = self.fill_count as f64;
        let profile_weight = 0.1 + 0.9 * (-fills_f / 40.0).exp();
        let blended = profile_weight * self.profile_kappa + (1.0 - profile_weight) * self.fill_kappa;

        // Warmup factor: starts at 0.5, converges to 1.0
        let warmup = 0.5 + 0.5 * (1.0 - (-fills_f / 10.0).exp());

        let result = blended * warmup;
        result.max(10.0) // Hard floor to prevent GLFT blowup
    }

    /// Effective sigma with smooth blending (same curve as kappa).
    pub fn effective_sigma(&self) -> f64 {
        let fills_f = self.fill_count as f64;
        let profile_weight = 0.1 + 0.9 * (-fills_f / 40.0).exp();
        let blended = profile_weight * self.profile_sigma + (1.0 - profile_weight) * self.fill_sigma;
        blended.max(1e-6)
    }

    /// Uncertainty premium in bps that should be added to spreads.
    ///
    /// Decays within each phase as progress toward next phase increases.
    pub fn uncertainty_premium_bps(&self) -> f64 {
        let max = self.phase.max_uncertainty_premium_bps();

        // Progress within current phase [0, 1]
        let progress = match self.phase {
            CalibrationPhase::Cold => {
                (self.fill_count as f64) / (WARMING_FILLS as f64)
            }
            CalibrationPhase::Warming => {
                ((self.fill_count - WARMING_FILLS) as f64)
                    / ((CALIBRATED_FILLS - WARMING_FILLS) as f64)
            }
            CalibrationPhase::Calibrated => {
                ((self.fill_count - CALIBRATED_FILLS) as f64)
                    / ((CONFIDENT_FILLS - CALIBRATED_FILLS) as f64)
            }
            CalibrationPhase::Confident => 1.0,
        };

        let progress = progress.clamp(0.0, 1.0);
        max * (1.0 - progress * progress)
    }

    /// Current calibration phase.
    pub fn phase(&self) -> CalibrationPhase {
        self.phase
    }

    /// Total fill count.
    pub fn fill_count(&self) -> usize {
        self.fill_count
    }

    /// Current adverse selection rate [0, 1].
    pub fn as_rate(&self) -> f64 {
        self.as_rate
    }

    /// Whether the coordinator has been seeded from a profile.
    pub fn is_seeded(&self) -> bool {
        self.seeded
    }

    /// Diagnostic log string.
    pub fn log_status(&self) -> String {
        format!(
            "Calibration: phase={:?}, fills={}, eff_kappa={:.0}, eff_sigma={:.6}, \
             premium={:.1}bps, as_rate={:.1}%, profile_kappa={:.0}, fill_kappa={:.0}",
            self.phase,
            self.fill_count,
            self.effective_kappa(),
            self.effective_sigma(),
            self.uncertainty_premium_bps(),
            self.as_rate * 100.0,
            self.profile_kappa,
            self.fill_kappa,
        )
    }

    /// Update phase based on fill count.
    fn update_phase(&mut self) {
        let new_phase = if self.fill_count >= CONFIDENT_FILLS {
            CalibrationPhase::Confident
        } else if self.fill_count >= CALIBRATED_FILLS {
            CalibrationPhase::Calibrated
        } else if self.fill_count >= WARMING_FILLS {
            CalibrationPhase::Warming
        } else {
            CalibrationPhase::Cold
        };
        self.phase = new_phase;
    }
}

/// Simple EWMA.
#[inline]
fn ewma(prev: f64, new: f64, alpha: f64) -> f64 {
    (1.0 - alpha) * prev + alpha * new
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Local forward GLFT formula for tests (avoids cross-module dependency).
    fn spread_for_kappa_test(kappa: f64, gamma: f64, fee_bps: f64) -> f64 {
        let ratio = gamma / kappa;
        let glft_bps = if ratio > 1e-12 {
            (10_000.0 / gamma) * (1.0 + ratio).ln()
        } else {
            10_000.0 / kappa
        };
        glft_bps + fee_bps
    }

    fn make_profile_with_kappa(target_kappa: f64) -> MarketProfile {
        // Reverse-engineer the BBO spread that yields this kappa
        let spread_bps = spread_for_kappa_test(
            target_kappa / 0.7, // compensate for 0.7 safety factor
            0.15,
            1.5,
        );
        let mid = 25.0;
        let half_spread_frac = spread_bps / 10_000.0;
        let mut profile = MarketProfile::new();
        profile.on_l2_book(
            mid * (1.0 - half_spread_frac),
            mid * (1.0 + half_spread_frac),
            100.0,
            100.0,
        );
        profile
    }

    #[test]
    fn test_phase_transitions() {
        let profile = make_profile_with_kappa(500.0);
        let mut coord = CalibrationCoordinator::new();
        coord.initialize_from_profile(&profile);

        assert_eq!(coord.phase(), CalibrationPhase::Cold);

        // 5 fills → Warming
        for _ in 0..5 {
            coord.on_fill(20.0, false);
        }
        assert_eq!(coord.phase(), CalibrationPhase::Warming);

        // 30 fills → Calibrated
        for _ in 0..25 {
            coord.on_fill(20.0, false);
        }
        assert_eq!(coord.phase(), CalibrationPhase::Calibrated);

        // 100 fills → Confident
        for _ in 0..70 {
            coord.on_fill(20.0, false);
        }
        assert_eq!(coord.phase(), CalibrationPhase::Confident);
    }

    #[test]
    fn test_effective_kappa_warmup() {
        let profile = make_profile_with_kappa(500.0);
        let mut coord = CalibrationCoordinator::new();
        coord.initialize_from_profile(&profile);

        // At 0 fills: warmup factor = 0.5, profile_weight = 1.0
        // effective ≈ 0.5 × profile_kappa
        let ek0 = coord.effective_kappa();
        let expected_half = coord.profile_kappa * 0.5;
        assert!(
            (ek0 - expected_half).abs() / expected_half < 0.1,
            "At 0 fills, effective_kappa should be ~50% of profile: got {ek0:.0}, expected ~{expected_half:.0}"
        );

        // After 20 fills: warmup ≈ 0.5 + 0.5*(1-exp(-2)) ≈ 0.93
        for _ in 0..20 {
            coord.on_fill(20.0, false);
        }
        let ek20 = coord.effective_kappa();
        assert!(
            ek20 > ek0,
            "Kappa should increase with fills: ek0={ek0:.0}, ek20={ek20:.0}"
        );
    }

    #[test]
    fn test_spreads_monotonically_tighten() {
        let profile = make_profile_with_kappa(500.0);
        let mut coord = CalibrationCoordinator::new();
        coord.initialize_from_profile(&profile);

        let gamma = 0.15;
        let fee_bps = 1.5;
        let mut prev_spread = spread_for_kappa_test(
            coord.effective_kappa(), gamma, fee_bps
        ) + coord.uncertainty_premium_bps();

        // Feed 100 fills at 20 bps (reasonable distance)
        for i in 1..=100 {
            coord.on_fill(20.0, false);
            let spread = crate::market_maker::strategy::spread_oracle::spread_for_kappa(
                coord.effective_kappa(), gamma, 0.0, 0.0, fee_bps
            ) + coord.uncertainty_premium_bps();

            assert!(
                spread <= prev_spread + 0.01, // tiny tolerance for float rounding
                "Spread should monotonically tighten: fill {i}, prev={prev_spread:.2}, curr={spread:.2}"
            );
            prev_spread = spread;
        }
    }

    #[test]
    fn test_as_rate_clamps_kappa() {
        let profile = make_profile_with_kappa(300.0);
        let mut coord = CalibrationCoordinator::new();
        coord.initialize_from_profile(&profile);

        // Feed fills that are adversely selected
        for _ in 0..50 {
            coord.on_fill(5.0, true); // Very close fills + AS
        }

        // AS rate should be high
        assert!(
            coord.as_rate() > 0.3,
            "AS rate should be high: {:.2}",
            coord.as_rate()
        );

        // Kappa should be clamped (not growing freely)
        assert!(
            coord.fill_kappa <= coord.profile_kappa * 1.3,
            "fill_kappa should be clamped near profile: fill={:.0}, profile={:.0}",
            coord.fill_kappa,
            coord.profile_kappa
        );
    }

    #[test]
    fn test_uncertainty_premium_decays() {
        let profile = make_profile_with_kappa(500.0);
        let mut coord = CalibrationCoordinator::new();
        coord.initialize_from_profile(&profile);

        let premium_cold = coord.uncertainty_premium_bps();
        assert!(premium_cold > 4.0, "Cold premium should be ~5 bps");

        for _ in 0..5 {
            coord.on_fill(20.0, false);
        }
        let premium_warming = coord.uncertainty_premium_bps();
        assert!(
            premium_warming < premium_cold,
            "Warming premium should be less than Cold"
        );

        for _ in 0..95 {
            coord.on_fill(20.0, false);
        }
        let premium_confident = coord.uncertainty_premium_bps();
        assert!(
            premium_confident < 0.01,
            "Confident premium should be ~0: {premium_confident:.2}"
        );
    }

    #[test]
    fn test_serialization_round_trip() {
        let profile = make_profile_with_kappa(500.0);
        let mut coord = CalibrationCoordinator::new();
        coord.initialize_from_profile(&profile);
        for _ in 0..10 {
            coord.on_fill(20.0, false);
        }

        let json = serde_json::to_string(&coord).unwrap();
        let restored: CalibrationCoordinator = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.phase(), coord.phase());
        assert_eq!(restored.fill_count(), coord.fill_count());
        assert!((restored.effective_kappa() - coord.effective_kappa()).abs() < 0.01);
    }

    #[test]
    fn test_uninitialized_defaults() {
        let coord = CalibrationCoordinator::new();
        assert_eq!(coord.phase(), CalibrationPhase::Cold);
        assert_eq!(coord.fill_count(), 0);
        assert!(!coord.is_seeded());
        // Should return reasonable defaults
        assert!(coord.effective_kappa() > 0.0);
        assert!(coord.effective_sigma() > 0.0);
    }

    #[test]
    fn test_log_status() {
        let coord = CalibrationCoordinator::new();
        let status = coord.log_status();
        assert!(status.contains("Calibration:"));
        assert!(status.contains("phase=Cold"));
    }

    /// E2E: L2 profile → initialize → 100 fills → verify kappa converges, uncertainty drops.
    #[test]
    fn test_bootstrap_e2e_convergence() {
        // Step 1: Create a MarketProfile from realistic L2 data
        let mut profile = MarketProfile::new();
        // Feed several L2 snapshots to stabilize the profile
        for _ in 0..10 {
            profile.on_l2_book(24.95, 25.05, 500.0, 500.0);
        }
        assert!(profile.is_initialized());

        // Step 2: Seed coordinator from profile
        let mut coord = CalibrationCoordinator::new();
        assert!(!coord.is_seeded());
        coord.initialize_from_profile(&profile);
        assert!(coord.is_seeded());

        let initial_kappa = coord.effective_kappa();
        let initial_premium = coord.uncertainty_premium_bps();
        assert!(initial_kappa > 10.0, "Initial kappa should be meaningful");
        assert!(initial_premium > 3.0, "Cold phase should have significant premium");

        // Step 3: Feed 100 fills at ~15 bps distance (mildly favorable fills)
        let mut prev_kappa = initial_kappa;
        for i in 0..100 {
            let distance_bps = 15.0 + (i as f64 * 0.01); // slight variation
            let was_adverse = i % 5 == 0; // 20% AS rate
            coord.on_fill(distance_bps, was_adverse);

            let k = coord.effective_kappa();
            // After the first few fills, kappa should be growing (warmup factor increasing)
            if i > 5 {
                assert!(
                    k >= prev_kappa * 0.95, // allow small float fluctuation
                    "Kappa should generally grow: fill {i}, prev={prev_kappa:.0}, curr={k:.0}"
                );
            }
            prev_kappa = k;
        }

        // Step 4: Verify convergence properties
        let final_kappa = coord.effective_kappa();
        let final_premium = coord.uncertainty_premium_bps();

        assert!(
            final_kappa > initial_kappa * 1.3,
            "Final kappa ({final_kappa:.0}) should be significantly higher than initial ({initial_kappa:.0})"
        );
        assert!(
            final_premium < 0.1,
            "Confident phase premium should be near zero: {final_premium:.2}"
        );
        assert_eq!(coord.phase(), CalibrationPhase::Confident);
        assert_eq!(coord.fill_count(), 100);
    }

    /// Verify coordinator checkpoint round-trip preserves all state.
    #[test]
    fn test_coordinator_checkpoint_round_trip() {
        let mut profile = MarketProfile::new();
        profile.on_l2_book(24.95, 25.05, 500.0, 500.0);

        let mut coord = CalibrationCoordinator::new();
        coord.initialize_from_profile(&profile);

        // Feed 20 fills to get into Warming phase
        for _ in 0..20 {
            coord.on_fill(18.0, false);
        }
        for _ in 0..5 {
            coord.on_fill(12.0, true); // some adverse
        }

        // Serialize (CalibrationCoordinatorCheckpoint == CalibrationCoordinator)
        let json = serde_json::to_string(&coord).unwrap();
        let restored: CalibrationCoordinator = serde_json::from_str(&json).unwrap();

        // Verify all state matches
        assert_eq!(restored.phase(), coord.phase());
        assert_eq!(restored.fill_count(), coord.fill_count());
        assert!(restored.is_seeded());
        assert!((restored.effective_kappa() - coord.effective_kappa()).abs() < 0.01);
        assert!((restored.uncertainty_premium_bps() - coord.uncertainty_premium_bps()).abs() < 0.01);
        assert!((restored.as_rate() - coord.as_rate()).abs() < 0.001);
    }
}
