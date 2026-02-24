//! Pure-math GLFT formula inversions for spread ↔ kappa conversion.
//!
//! The GLFT optimal half-spread formula is:
//!
//! ```text
//! half_spread_bps = (10000 / gamma) * ln(1 + gamma / kappa) + 5000 * gamma * sigma^2 * T + fee_bps
//! ```
//!
//! This module provides the forward function (`spread_for_kappa`) and its
//! analytical inverse (`kappa_for_spread`), both operating in **bps** for
//! half-spread and fee, with raw (fraction) gamma and sigma.

/// Compute the GLFT half-spread in bps for a given kappa.
///
/// # Formula
///
/// ```text
/// half_spread_bps = (10000 / gamma) * ln(1 + gamma / kappa)
///                 + 5000 * gamma * sigma^2 * time_horizon
///                 + fee_bps
/// ```
///
/// The sigma^2 * T term is the Avellaneda-Stoikov volatility compensation.
/// Pass `sigma = 0.0` or `time_horizon = 0.0` for pure kappa-to-spread.
///
/// # Arguments
/// * `kappa` - Order flow intensity (must be > 0)
/// * `gamma` - Risk aversion parameter, raw fraction (must be > 0)
/// * `sigma` - Volatility per sqrt-second, raw fraction
/// * `time_horizon` - Expected holding time in seconds
/// * `fee_bps` - Maker fee in basis points (e.g. 1.5)
///
/// # Returns
/// Half-spread in basis points.
#[inline]
pub fn spread_for_kappa(
    kappa: f64,
    gamma: f64,
    sigma: f64,
    time_horizon: f64,
    fee_bps: f64,
) -> f64 {
    debug_assert!(kappa > 0.0, "kappa must be > 0, got {kappa}");
    debug_assert!(gamma > 0.0, "gamma must be > 0, got {gamma}");

    let safe_kappa = kappa.max(f64::EPSILON);
    let safe_gamma = gamma.max(f64::EPSILON);

    let ratio = safe_gamma / safe_kappa;

    // GLFT term: (10000 / gamma) * ln(1 + gamma/kappa)
    let glft_bps = if ratio > 1e-12 {
        (10_000.0 / safe_gamma) * (1.0 + ratio).ln()
    } else {
        // Taylor: ln(1+x) ~ x => (10000/gamma)*(gamma/kappa) = 10000/kappa
        10_000.0 / safe_kappa
    };

    // Vol compensation: 5000 * gamma * sigma^2 * T
    // (This is 0.5 * gamma * sigma^2 * T in fraction, converted to bps via *10000)
    let vol_comp_bps = 5_000.0 * safe_gamma * sigma.powi(2) * time_horizon;

    glft_bps + vol_comp_bps + fee_bps
}

/// Compute the kappa implied by a target half-spread in bps.
///
/// Analytical inverse of the GLFT spread formula (ignoring the vol compensation
/// term, which is independent of kappa):
///
/// ```text
/// kappa = gamma / (exp(gamma * (target_half_spread_bps - fee_bps - vol_comp_bps) / 10000) - 1)
/// ```
///
/// If `target_half_spread_bps <= fee_bps + vol_comp_bps`, the GLFT term must
/// be zero or negative, which requires kappa → ∞. In this case, returns
/// `f64::INFINITY`.
///
/// # Arguments
/// * `target_half_spread_bps` - Desired half-spread in basis points
/// * `gamma` - Risk aversion parameter, raw fraction (must be > 0)
/// * `fee_bps` - Maker fee in basis points
///
/// # Returns
/// Implied kappa (order flow intensity). Always > 0 when inputs are valid.
#[inline]
pub fn kappa_for_spread(target_half_spread_bps: f64, gamma: f64, fee_bps: f64) -> f64 {
    debug_assert!(gamma > 0.0, "gamma must be > 0, got {gamma}");

    let safe_gamma = gamma.max(f64::EPSILON);

    // The GLFT-only component of the spread (excluding vol comp which doesn't depend on kappa)
    let glft_bps = target_half_spread_bps - fee_bps;

    if glft_bps <= 0.0 {
        // Target spread is at or below fee — requires infinite kappa (infinitely deep book)
        return f64::INFINITY;
    }

    // Inverse: kappa = gamma / (exp(gamma * glft_bps / 10000) - 1)
    let exponent = safe_gamma * glft_bps / 10_000.0;

    // Guard against overflow: exp(x) for x > 700 overflows f64
    if exponent > 700.0 {
        // Very large exponent means very small kappa (extremely wide spread)
        // kappa ~ gamma * exp(-exponent) -> 0
        return safe_gamma * (-exponent).exp();
    }

    let denom = exponent.exp() - 1.0;

    if denom < 1e-15 {
        // When exponent ~ 0, use Taylor: exp(x)-1 ~ x
        // kappa ~ gamma / (gamma * glft_bps / 10000) = 10000 / glft_bps
        return 10_000.0 / glft_bps;
    }

    safe_gamma / denom
}

/// Extended inverse: kappa from spread accounting for vol compensation.
///
/// Subtracts the vol compensation term before inverting:
/// ```text
/// vol_comp_bps = 5000 * gamma * sigma^2 * T
/// glft_bps = target - fee_bps - vol_comp_bps
/// kappa = gamma / (exp(gamma * glft_bps / 10000) - 1)
/// ```
///
/// # Arguments
/// * `target_half_spread_bps` - Desired half-spread in basis points
/// * `gamma` - Risk aversion, raw fraction (must be > 0)
/// * `sigma` - Volatility per sqrt-second, raw fraction
/// * `time_horizon` - Expected holding time in seconds
/// * `fee_bps` - Maker fee in basis points
#[inline]
pub fn kappa_for_spread_full(
    target_half_spread_bps: f64,
    gamma: f64,
    sigma: f64,
    time_horizon: f64,
    fee_bps: f64,
) -> f64 {
    let safe_gamma = gamma.max(f64::EPSILON);
    let vol_comp_bps = 5_000.0 * safe_gamma * sigma.powi(2) * time_horizon;
    let adjusted_target = target_half_spread_bps - vol_comp_bps;
    kappa_for_spread(adjusted_target, gamma, fee_bps)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON_BPS: f64 = 0.01; // 0.01 bps tolerance
    const FEE_BPS: f64 = 1.5; // Hyperliquid maker fee

    // --- Round-trip tests ---

    #[test]
    fn test_round_trip_various_kappas() {
        let gamma = 0.15;
        let fee = 1.5;

        for &kappa in &[100.0, 500.0, 1000.0, 5000.0, 10000.0] {
            let spread = spread_for_kappa(kappa, gamma, 0.0, 0.0, fee);
            let recovered_kappa = kappa_for_spread(spread, gamma, fee);
            let rel_error = (recovered_kappa - kappa).abs() / kappa;
            assert!(
                rel_error < 1e-9,
                "Round-trip failed for kappa={kappa}: spread={spread:.4} bps, \
                 recovered={recovered_kappa:.4}, rel_error={rel_error:.2e}"
            );
        }
    }

    #[test]
    fn test_round_trip_with_vol_comp() {
        let gamma = 0.15;
        let sigma = 0.0003; // 3 bps/sqrt(s)
        let tau = 60.0;
        let fee = 1.5;

        for &kappa in &[200.0, 1000.0, 5000.0] {
            let spread = spread_for_kappa(kappa, gamma, sigma, tau, fee);
            let recovered = kappa_for_spread_full(spread, gamma, sigma, tau, fee);
            let rel_error = (recovered - kappa).abs() / kappa;
            assert!(
                rel_error < 1e-9,
                "Round-trip (vol) failed for kappa={kappa}: spread={spread:.4} bps, \
                 recovered={recovered:.4}, rel_error={rel_error:.2e}"
            );
        }
    }

    // --- HYPE-like tests ---

    #[test]
    fn test_hype_spread_range() {
        // HYPE: gamma=0.1, fee=1.5 bps, kappa~466
        let spread = spread_for_kappa(466.0, 0.1, 0.0, 0.0, FEE_BPS);
        assert!(
            spread > 15.0 && spread < 30.0,
            "HYPE half-spread should be 15-30 bps, got {spread:.2}"
        );
    }

    #[test]
    fn test_hype_kappa_inversion() {
        let gamma = 0.1;
        let target_spread = 22.0; // ~22 bps for HYPE
        let kappa = kappa_for_spread(target_spread, gamma, FEE_BPS);
        assert!(
            kappa > 300.0 && kappa < 700.0,
            "HYPE kappa should be 300-700, got {kappa:.1}"
        );
    }

    // --- BTC-like tests ---

    #[test]
    fn test_btc_spread_range() {
        // BTC: gamma=0.01, kappa~6000
        let spread = spread_for_kappa(6000.0, 0.01, 0.0, 0.0, FEE_BPS);
        assert!(
            spread > 2.0 && spread < 6.0,
            "BTC half-spread should be 2-6 bps, got {spread:.2}"
        );
    }

    #[test]
    fn test_btc_kappa_inversion() {
        let gamma = 0.01;
        let target_spread = 3.5; // ~3.5 bps for BTC
        let kappa = kappa_for_spread(target_spread, gamma, FEE_BPS);
        assert!(
            kappa > 3000.0 && kappa < 10000.0,
            "BTC kappa should be 3000-10000, got {kappa:.1}"
        );
    }

    // --- Edge cases ---

    #[test]
    fn test_small_kappa_wide_spread() {
        // Very small kappa => very wide spread
        let spread = spread_for_kappa(10.0, 0.1, 0.0, 0.0, FEE_BPS);
        assert!(
            spread > 80.0,
            "kappa=10 should give wide spread (>80 bps), got {spread:.2}"
        );
    }

    #[test]
    fn test_large_kappa_tight_spread() {
        // Very large kappa => spread approaches fee
        let spread = spread_for_kappa(50000.0, 0.1, 0.0, 0.0, FEE_BPS);
        assert!(
            spread < 5.0,
            "kappa=50000 should give tight spread (<5 bps), got {spread:.2}"
        );
        assert!(
            spread > FEE_BPS,
            "spread should be above fee, got {spread:.2}"
        );
    }

    #[test]
    fn test_zero_fee_works() {
        let gamma = 0.1;
        let kappa = 1000.0;
        let spread = spread_for_kappa(kappa, gamma, 0.0, 0.0, 0.0);
        assert!(spread > 0.0, "spread with zero fee should be positive");

        let recovered = kappa_for_spread(spread, gamma, 0.0);
        let rel_error = (recovered - kappa).abs() / kappa;
        assert!(rel_error < 1e-9, "round-trip with zero fee failed");
    }

    #[test]
    fn test_target_at_fee_returns_infinity() {
        // If target = fee, GLFT component is 0, requires infinite kappa
        let kappa = kappa_for_spread(FEE_BPS, 0.1, FEE_BPS);
        assert!(
            kappa.is_infinite(),
            "kappa should be infinite when target = fee, got {kappa}"
        );
    }

    #[test]
    fn test_target_below_fee_returns_infinity() {
        let kappa = kappa_for_spread(0.5, 0.1, FEE_BPS);
        assert!(
            kappa.is_infinite(),
            "kappa should be infinite when target < fee, got {kappa}"
        );
    }

    // --- Monotonicity ---

    #[test]
    fn test_spread_decreases_with_kappa() {
        let gamma = 0.1;
        let mut prev_spread = f64::MAX;
        for kappa in [50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0] {
            let spread = spread_for_kappa(kappa, gamma, 0.0, 0.0, FEE_BPS);
            assert!(
                spread < prev_spread,
                "spread should decrease with kappa: prev={prev_spread:.4}, curr={spread:.4} at kappa={kappa}"
            );
            prev_spread = spread;
        }
    }

    #[test]
    fn test_kappa_decreases_with_spread() {
        let gamma = 0.1;
        let mut prev_kappa = f64::MAX;
        for spread_bps in [3.0, 5.0, 10.0, 20.0, 50.0, 100.0] {
            let kappa = kappa_for_spread(spread_bps, gamma, FEE_BPS);
            if !kappa.is_infinite() {
                assert!(
                    kappa < prev_kappa,
                    "kappa should decrease with wider spread: prev={prev_kappa:.2}, curr={kappa:.2} at spread={spread_bps}"
                );
                prev_kappa = kappa;
            }
        }
    }

    // --- Gamma sensitivity ---

    #[test]
    fn test_spread_increases_with_gamma_vol_regime() {
        // With volatility compensation, spread is monotonically increasing in gamma
        // for moderate-to-large gamma, because 5000*gamma*sigma^2*T dominates.
        // The pure GLFT term (10000/gamma)*ln(1+gamma/kappa) is NOT monotone in
        // gamma when sigma=0, so we test in the vol regime where it holds.
        let kappa = 1000.0;
        let sigma = 0.0005;
        let tau = 60.0;
        let mut prev_spread = 0.0;
        for gamma in [0.1, 0.5, 1.0, 5.0, 10.0] {
            let spread = spread_for_kappa(kappa, gamma, sigma, tau, FEE_BPS);
            assert!(
                spread > prev_spread,
                "spread should increase with gamma in vol regime: prev={prev_spread:.4}, curr={spread:.4} at gamma={gamma}"
            );
            prev_spread = spread;
        }
    }

    // --- Consistency with glft.rs conventions ---

    #[test]
    fn test_matches_glft_half_spread_formula() {
        // Verify our bps formula matches the fraction formula in glft.rs
        // glft.rs: half_spread = (1/gamma)*ln(1+gamma/kappa) + 0.5*gamma*sigma^2*T + fee_fraction
        let gamma: f64 = 0.15;
        let kappa: f64 = 2000.0;
        let sigma: f64 = 0.0002;
        let tau: f64 = 60.0;
        let fee_fraction: f64 = 0.00015; // 1.5 bps

        // glft.rs formula (fraction)
        let glft_fraction = (1.0 / gamma) * (1.0 + gamma / kappa).ln()
            + 0.5 * gamma * sigma.powi(2) * tau
            + fee_fraction;
        let glft_bps = glft_fraction * 10_000.0;

        // Our formula (bps)
        let oracle_bps = spread_for_kappa(kappa, gamma, sigma, tau, 1.5);

        assert!(
            (oracle_bps - glft_bps).abs() < EPSILON_BPS,
            "Mismatch with glft.rs: oracle={oracle_bps:.6} bps, glft={glft_bps:.6} bps"
        );
    }

    // --- Vol compensation ---

    #[test]
    fn test_vol_comp_adds_to_spread() {
        let kappa = 1000.0;
        let gamma = 0.15;
        let sigma = 0.001; // 10 bps/sqrt(s)
        let tau = 120.0;

        let spread_no_vol = spread_for_kappa(kappa, gamma, 0.0, 0.0, FEE_BPS);
        let spread_with_vol = spread_for_kappa(kappa, gamma, sigma, tau, FEE_BPS);

        let expected_vol_bps = 5_000.0 * gamma * sigma.powi(2) * tau;
        let actual_diff = spread_with_vol - spread_no_vol;

        assert!(
            (actual_diff - expected_vol_bps).abs() < EPSILON_BPS,
            "Vol comp mismatch: expected={expected_vol_bps:.4}, got={actual_diff:.4}"
        );
    }

    // --- Numerical stability ---

    #[test]
    fn test_very_small_gamma() {
        // gamma -> 0: GLFT term -> 10000/kappa (Taylor)
        let kappa = 1000.0;
        let gamma = 1e-8;
        let spread = spread_for_kappa(kappa, gamma, 0.0, 0.0, 0.0);
        let expected = 10_000.0 / kappa; // 10 bps
        assert!(
            (spread - expected).abs() < 0.1,
            "small gamma: expected ~{expected:.2} bps, got {spread:.2} bps"
        );
    }

    #[test]
    fn test_very_large_gamma() {
        // Large gamma: vol comp dominates
        let kappa = 1000.0;
        let gamma = 50.0;
        let sigma = 0.0002;
        let tau = 60.0;

        let spread = spread_for_kappa(kappa, gamma, sigma, tau, FEE_BPS);
        assert!(
            spread.is_finite(),
            "spread should be finite for large gamma, got {spread}"
        );
        assert!(spread > 0.0, "spread should be positive for large gamma");
    }

    #[test]
    fn test_kappa_for_very_wide_spread() {
        // Very wide target spread -> very small kappa
        let kappa = kappa_for_spread(500.0, 0.1, FEE_BPS);
        assert!(
            kappa > 0.0 && kappa < 25.0,
            "500 bps spread should imply kappa < 25, got {kappa:.4}"
        );
    }
}
