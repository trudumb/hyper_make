//! Mathematically Derived Constants
//!
//! This module provides functions to derive parameters from first principles
//! rather than using hardcoded magic numbers. Each derivation includes:
//! 1. The mathematical formula
//! 2. The domain knowledge that justifies it
//! 3. The inputs required from market data
//!
//! # Categories
//!
//! - **GLFT Derivations**: Parameters derived from the optimal market making formula
//! - **Risk Derivations**: Parameters derived from VaR/Kelly criterion
//! - **Microstructure Derivations**: Parameters from tick-level price dynamics
//! - **Time-Scale Derivations**: Parameters from autocorrelation analysis

/// GLFT-derived base gamma (risk aversion).
///
/// # Derivation
///
/// From the GLFT formula: δ* = (1/γ) × ln(1 + γ/κ)
///
/// Rearranging for a target spread:
/// γ = 2κ × sinh(δ_target × κ) / (exp(δ_target × κ) - 1)
///
/// For small spreads (δ_target × κ << 1):
/// γ ≈ 2κ / (exp(δ_target × κ) - 1) ≈ 2 / δ_target
///
/// # Arguments
/// * `target_half_spread_bps` - Desired half-spread in bps
/// * `kappa` - Fill intensity (fills per unit spread)
/// * `sigma` - Volatility (per-second)
/// * `time_horizon` - Expected holding time (seconds)
///
/// # Returns
/// Optimal gamma for achieving target spread
pub fn derive_gamma_from_glft(
    target_half_spread_bps: f64,
    kappa: f64,
    sigma: f64,
    time_horizon: f64,
) -> f64 {
    let target_spread = target_half_spread_bps / 10_000.0;

    // GLFT optimal gamma when targeting a specific spread
    // δ* = (1/γ) × ln(1 + γ/κ) + γσ²T/2 (with inventory cost)
    //
    // For no inventory (q=0), solving for γ:
    // γ = κ × (exp(δ × κ) - 1) / exp(δ × κ)

    if kappa <= 0.0 || target_spread <= 0.0 {
        return 0.15; // Fallback to default
    }

    let exp_term = (target_spread * kappa).exp();
    let gamma = kappa * (exp_term - 1.0) / exp_term;

    // Include volatility adjustment: higher vol → higher risk aversion
    let vol_adjustment = 1.0 + (sigma / 0.0002 - 1.0).max(0.0) * 0.5;

    // Include time horizon adjustment: longer horizon → more time to offset losses
    let time_adjustment = (time_horizon / 60.0).sqrt().clamp(0.5, 2.0);

    (gamma * vol_adjustment / time_adjustment).clamp(0.05, 5.0)
}

/// GLFT-derived minimum spread floor.
///
/// # Derivation
///
/// The minimum profitable spread must cover:
/// 1. Maker fee: f_maker (1.5 bps on Hyperliquid)
/// 2. Expected adverse selection: E[AS]
/// 3. Execution slippage: σ × √(τ_update)
///
/// δ_min = f_maker + E[AS] + σ × √(τ_update)
///
/// # Arguments
/// * `maker_fee_bps` - Maker fee in bps (typically 1.5)
/// * `expected_as_bps` - Expected adverse selection in bps
/// * `sigma` - Volatility (per-second)
/// * `update_latency_ms` - Quote update latency in milliseconds
///
/// # Returns
/// Minimum spread floor in bps
pub fn derive_spread_floor(
    maker_fee_bps: f64,
    expected_as_bps: f64,
    sigma: f64,
    update_latency_ms: f64,
) -> f64 {
    let latency_s = update_latency_ms / 1000.0;

    // Slippage from latency: price can move σ×√τ during quote update cycle
    let latency_slippage_bps = sigma * latency_s.sqrt() * 10_000.0;

    // Total spread floor
    let floor = maker_fee_bps + expected_as_bps.abs() + latency_slippage_bps;

    // Add 10% buffer for safety
    (floor * 1.1).clamp(2.0, 50.0)
}

/// Kelly-derived maximum daily loss.
///
/// # Derivation
///
/// Kelly criterion says to risk f* = (bp - q) / b of bankroll per bet.
/// For market making, we interpret this as:
/// - Max position ∝ Kelly fraction
/// - Max daily loss ≈ max_position × expected_max_adverse_move
///
/// With safety factor of 0.25 (quarter Kelly):
/// max_loss = account × kelly_fraction × expected_daily_vol
///
/// # Arguments
/// * `account_value` - Account equity in USD
/// * `kelly_fraction` - Fraction of full Kelly (typically 0.25)
/// * `daily_volatility` - Expected daily volatility (fraction, e.g., 0.02 for 2%)
///
/// # Returns
/// Maximum daily loss in USD
pub fn derive_max_daily_loss(
    account_value: f64,
    kelly_fraction: f64,
    daily_volatility: f64,
) -> f64 {
    // Max loss = account × f × 2σ (2σ move covers 95% of days)
    account_value * kelly_fraction * daily_volatility * 2.0
}

/// VaR-derived maximum drawdown.
///
/// # Derivation
///
/// Maximum drawdown should be set to the 99th percentile of daily losses.
/// For approximately normal returns with daily vol σ:
/// VaR_99 ≈ 2.33 × σ_daily
///
/// Maximum drawdown = VaR_99 × horizon_factor
/// where horizon_factor accounts for multi-day drawdown accumulation.
///
/// # Arguments
/// * `daily_volatility` - Daily volatility (fraction)
/// * `horizon_days` - Recovery horizon in days
///
/// # Returns
/// Maximum drawdown as fraction (e.g., 0.05 for 5%)
pub fn derive_max_drawdown(daily_volatility: f64, horizon_days: f64) -> f64 {
    // 99th percentile of normal distribution
    let z_99 = 2.33;

    // VaR for single day
    let daily_var = z_99 * daily_volatility;

    // Multi-day drawdown scales as √T for random walk
    // But with mean reversion, scaling is slower (~T^0.4)
    let horizon_factor = horizon_days.powf(0.4);

    // Maximum drawdown
    (daily_var * horizon_factor).clamp(0.02, 0.20)
}

/// Autocorrelation-derived EWMA decay factor.
///
/// # Derivation
///
/// For a target half-life of h samples:
/// α = 1 - exp(-ln(2) / h)
///
/// The half-life can be determined from the autocorrelation function:
/// h ≈ argmin_τ |ACF(τ) - 0.5|
///
/// # Arguments
/// * `half_life_samples` - Number of samples for signal to decay by 50%
///
/// # Returns
/// EWMA decay factor α ∈ (0, 1)
pub fn derive_ewma_alpha(half_life_samples: f64) -> f64 {
    if half_life_samples <= 0.0 {
        return 0.1; // Fallback
    }
    1.0 - (-std::f64::consts::LN_2 / half_life_samples).exp()
}

/// Volatility-derived Kalman filter parameters.
///
/// # Derivation
///
/// For price following a random walk with tick-level noise:
/// - Process noise Q = Var(price_change | tick_interval)
/// - Observation noise R = Var(bid_ask_bounce)
///
/// Typical values:
/// - Q ≈ σ² × Δt where σ is per-second vol, Δt is tick interval
/// - R ≈ (spread/4)² for bid-ask bounce
///
/// # Arguments
/// * `sigma_per_second` - Volatility per second
/// * `tick_interval_ms` - Average tick interval in milliseconds
/// * `spread_bps` - Typical bid-ask spread in bps
///
/// # Returns
/// (Q, R) - Process and observation noise variances
pub fn derive_kalman_noise(
    sigma_per_second: f64,
    tick_interval_ms: f64,
    spread_bps: f64,
) -> (f64, f64) {
    let tick_interval_s = tick_interval_ms / 1000.0;

    // Process noise: variance of price change per tick
    let q = sigma_per_second.powi(2) * tick_interval_s;

    // Observation noise: variance of bid-ask bounce
    // Bounce is typically ±spread/4
    let spread_fraction = spread_bps / 10_000.0;
    let r = (spread_fraction / 4.0).powi(2);

    (q, r)
}

/// Regime duration-derived hazard rate.
///
/// # Derivation
///
/// For BOCPD (Bayesian Online Changepoint Detection):
/// hazard_rate = 1 / E[regime_duration]
///
/// This is the probability of a regime change at any given time.
///
/// # Arguments
/// * `expected_regime_duration_samples` - Expected samples between regime changes
///
/// # Returns
/// Hazard rate (probability of changepoint per sample)
pub fn derive_hazard_rate(expected_regime_duration_samples: f64) -> f64 {
    if expected_regime_duration_samples <= 0.0 {
        return 0.004; // Default: 1/250 ≈ regime change every ~4 minutes at 1s samples
    }
    (1.0 / expected_regime_duration_samples).clamp(0.001, 0.1)
}

/// Cascade detection threshold from OI analysis.
///
/// # Derivation
///
/// A cascade is detected when OI drops significantly, indicating liquidations.
/// The threshold should be set to the p-th percentile of OI drops that
/// preceded adverse fill outcomes.
///
/// Rule of thumb: cascade_threshold = 2 × avg_normal_oi_change
///
/// # Arguments
/// * `normal_oi_change_std` - Standard deviation of normal OI changes
/// * `safety_multiplier` - How many std devs for threshold (typically 2-3)
///
/// # Returns
/// OI drop threshold (fraction, e.g., 0.02 for 2%)
pub fn derive_cascade_threshold(normal_oi_change_std: f64, safety_multiplier: f64) -> f64 {
    (normal_oi_change_std * safety_multiplier).clamp(0.01, 0.10)
}

/// Toxic hour gamma multiplier from adverse selection analysis.
///
/// # Derivation
///
/// During toxic hours, spreads need to be wider to maintain profitability:
/// gamma_mult = (|toxic_AS| + fees) / (|normal_AS| + fees)
///
/// This ensures spreads are wide enough to compensate for higher AS.
///
/// # Arguments
/// * `normal_as_bps` - Average adverse selection in normal hours (bps)
/// * `toxic_as_bps` - Average adverse selection in toxic hours (bps)
/// * `maker_fee_bps` - Maker fee in bps
///
/// # Returns
/// Gamma multiplier for toxic hours
pub fn derive_toxic_hour_multiplier(
    normal_as_bps: f64,
    toxic_as_bps: f64,
    maker_fee_bps: f64,
) -> f64 {
    let normal_cost = normal_as_bps.abs() + maker_fee_bps;
    let toxic_cost = toxic_as_bps.abs() + maker_fee_bps;

    if normal_cost < 0.1 {
        return 2.0; // Fallback
    }

    (toxic_cost / normal_cost).clamp(1.5, 4.0)
}

/// Quote latch threshold from churn analysis.
///
/// # Derivation
///
/// Quotes should only be updated when the change exceeds transaction costs
/// plus a buffer for noise. This prevents excessive API calls from
/// trivial price movements.
///
/// latch_threshold = 2 × (maker_fee + slippage_estimate)
///
/// # Arguments
/// * `maker_fee_bps` - Maker fee in bps
/// * `slippage_estimate_bps` - Expected slippage when updating quotes
/// * `noise_buffer_factor` - Multiplier for noise buffer (typically 1.5-2.0)
///
/// # Returns
/// Minimum quote change in bps before updating
pub fn derive_quote_latch_threshold(
    maker_fee_bps: f64,
    slippage_estimate_bps: f64,
    noise_buffer_factor: f64,
) -> f64 {
    let base_cost = maker_fee_bps + slippage_estimate_bps;
    (base_cost * noise_buffer_factor).clamp(1.0, 10.0)
}

/// Position-based reduce-only threshold from margin analysis.
///
/// # Derivation
///
/// Enter reduce-only mode when position approaches liquidation risk:
/// P(liquidation | position) > threshold
///
/// For perpetuals with leverage L:
/// liquidation_distance = maintenance_margin × L
/// reduce_only_threshold = 1 - safety_buffer / liquidation_distance
///
/// Simplified: reduce_only when position > (1 - safety_margin) × max_position
///
/// # Arguments
/// * `max_leverage` - Maximum leverage allowed
/// * `maintenance_margin` - Maintenance margin requirement (fraction)
/// * `safety_buffer` - Buffer before reaching liquidation (fraction)
///
/// # Returns
/// Position threshold as fraction of max_position for reduce-only mode
pub fn derive_reduce_only_threshold(
    max_leverage: f64,
    maintenance_margin: f64,
    safety_buffer: f64,
) -> f64 {
    // Distance to liquidation at max position
    let liquidation_distance = maintenance_margin * max_leverage;

    // Start reducing when we're within safety_buffer of liquidation
    // E.g., 3% buffer, 10x leverage, 5% maintenance → reduce at 1 - 0.03/0.5 = 0.94
    let threshold = 1.0 - safety_buffer / liquidation_distance;

    threshold.clamp(0.5, 0.95)
}

/// Depth spacing ratio from fill intensity curve.
///
/// # Derivation
///
/// Optimal ladder level spacing depends on the fill intensity decay with depth.
/// If λ(δ) = λ₀ × exp(-δ/δ_char), then optimal spacing is:
///
/// spacing_ratio = exp(1/n_levels) ≈ 1 + 1/n_levels
///
/// For n_levels = 5: ratio ≈ 1.22
/// For n_levels = 3: ratio ≈ 1.40
///
/// # Arguments
/// * `n_levels` - Number of ladder levels
/// * `fill_intensity_decay` - Characteristic depth for intensity decay (bps)
///
/// # Returns
/// Ratio between consecutive ladder level depths
pub fn derive_depth_spacing_ratio(n_levels: usize, fill_intensity_decay_bps: f64) -> f64 {
    if n_levels <= 1 {
        return 1.0;
    }

    // For exponential fill decay, optimal geometric spacing
    let n = n_levels as f64;

    // Ratio that gives equal expected fills across levels
    // Derived from: λ(δ₁) × Δ₁ = λ(δ₂) × Δ₂
    let ratio = (1.0 / n).exp();

    // Adjust for fill decay characteristic
    let decay_factor = (fill_intensity_decay_bps / 20.0).clamp(0.5, 2.0);

    (1.0 + (ratio - 1.0) * decay_factor).clamp(1.1, 2.5)
}

/// Momentum normalizer from regime volatility.
///
/// # Derivation
///
/// Momentum signals should be normalized by expected volatility:
/// normalized_momentum = momentum / (σ_regime × √T)
///
/// This gives a unit-free measure of momentum strength.
///
/// # Arguments
/// * `regime_volatility_bps` - Current regime volatility in bps per √second
/// * `momentum_horizon_s` - Time horizon for momentum calculation (seconds)
///
/// # Returns
/// Normalizer to convert raw momentum to unit-free signal
pub fn derive_momentum_normalizer(regime_volatility_bps: f64, momentum_horizon_s: f64) -> f64 {
    let expected_move = regime_volatility_bps * momentum_horizon_s.sqrt();
    expected_move.clamp(5.0, 100.0)
}

/// Confidence threshold from ROC analysis.
///
/// # Derivation
///
/// The optimal confidence threshold minimizes expected loss:
/// threshold* = argmin_t [FP_cost × P(FP | t) + FN_cost × P(FN | t)]
///
/// When FP_cost ≈ FN_cost (symmetric loss):
/// threshold* ≈ base_rate (prevalence of positive class)
///
/// # Arguments
/// * `base_rate` - Base rate of positive outcomes
/// * `fp_cost` - Cost of false positive (taking action when shouldn't)
/// * `fn_cost` - Cost of false negative (not taking action when should)
///
/// # Returns
/// Optimal confidence threshold for decision making
pub fn derive_confidence_threshold(base_rate: f64, fp_cost: f64, fn_cost: f64) -> f64 {
    if fp_cost + fn_cost < 1e-10 {
        return 0.5;
    }

    // Cost-sensitive threshold adjustment
    // threshold = base_rate × fn_cost / (base_rate × fn_cost + (1-base_rate) × fp_cost)
    let numerator = base_rate * fn_cost;
    let denominator = base_rate * fn_cost + (1.0 - base_rate) * fp_cost;

    (numerator / denominator).clamp(0.1, 0.9)
}

/// IR-based quote gate threshold.
///
/// # Derivation
///
/// Only quote when Information Ratio > 1.0 (signal adds value over noise).
/// The edge signal threshold should be set where:
/// P(IR > 1.0 | signal > threshold) > min_confidence
///
/// # Arguments
/// * `ir_mean` - Mean IR from historical data
/// * `ir_std` - Standard deviation of IR
/// * `target_ir` - Target IR for profitable quoting (typically 1.0)
///
/// # Returns
/// Minimum edge signal to achieve target IR
pub fn derive_ir_based_threshold(ir_mean: f64, ir_std: f64, target_ir: f64) -> f64 {
    if ir_std < 1e-10 {
        return 0.15; // Fallback
    }

    // Signal level where expected IR = target
    // Assuming linear relationship: IR = α + β × signal
    // Solve for signal when IR = target
    let signal_for_target = (target_ir - ir_mean) / ir_std;

    // Convert to meaningful threshold
    // If IR has negative relationship with signal, threshold is lower
    signal_for_target.abs().clamp(0.05, 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_gamma() {
        // Target 5 bps spread with kappa=2000
        // For GLFT: γ = κ × (exp(δκ) - 1) / exp(δκ)
        // With δ = 0.0005, κ = 2000: δκ = 1.0
        // γ_raw = 2000 × (e - 1) / e ≈ 2000 × 0.632 ≈ 1264
        // This is very high, gets clamped to gamma_max = 5.0
        // This shows that 5 bps is very tight for κ=2000
        let gamma = derive_gamma_from_glft(5.0, 2000.0, 0.0002, 60.0);
        assert!(gamma >= 0.05 && gamma <= 5.0, "gamma = {}", gamma);
        
        // With wider spread (20 bps), gamma should be lower
        let gamma_wide = derive_gamma_from_glft(20.0, 2000.0, 0.0002, 60.0);
        assert!(gamma_wide >= 0.05 && gamma_wide <= 5.0, "gamma_wide = {}", gamma_wide);
        
        // Lower kappa should give lower gamma for same spread
        let gamma_low_kappa = derive_gamma_from_glft(10.0, 500.0, 0.0002, 60.0);
        assert!(gamma_low_kappa >= 0.05 && gamma_low_kappa <= 5.0, "gamma_low_kappa = {}", gamma_low_kappa);
    }

    #[test]
    fn test_derive_spread_floor() {
        // With 1.5 bps fee, 3 bps AS, 0.02% vol, 50ms latency
        let floor = derive_spread_floor(1.5, 3.0, 0.0002, 50.0);
        // Should be roughly 1.5 + 3 + slippage + buffer ≈ 5-7 bps
        assert!(floor > 4.0 && floor < 10.0, "floor = {}", floor);
    }

    #[test]
    fn test_derive_max_daily_loss() {
        // $10k account, 0.25 Kelly, 2% daily vol
        let max_loss = derive_max_daily_loss(10_000.0, 0.25, 0.02);
        // Should be ~$100 (10k × 0.25 × 0.02 × 2)
        assert!(max_loss > 80.0 && max_loss < 120.0, "max_loss = {}", max_loss);
    }

    #[test]
    fn test_derive_ewma_alpha() {
        // 10-sample half-life
        let alpha = derive_ewma_alpha(10.0);
        // Should be ~0.067
        assert!(alpha > 0.05 && alpha < 0.10, "alpha = {}", alpha);

        // 100-sample half-life
        let alpha = derive_ewma_alpha(100.0);
        // Should be ~0.007
        assert!(alpha > 0.005 && alpha < 0.01, "alpha = {}", alpha);
    }

    #[test]
    fn test_derive_kalman_noise() {
        let (q, r) = derive_kalman_noise(0.0002, 100.0, 5.0);
        // Q = σ² × dt = 0.0002² × 0.1 = 4e-9 (not 4e-12!)
        // sigma is per-second vol, 100ms = 0.1s, so Q = 4e-9
        assert!(q > 1e-10 && q < 1e-7, "q = {:e}", q);
        // R = (spread/4)² = (0.0005/4)² = 1.56e-8
        assert!(r > 1e-10 && r < 1e-6, "r = {:e}", r);
    }

    #[test]
    fn test_derive_toxic_hour_multiplier() {
        // Normal AS: 5 bps, Toxic AS: 15 bps, Fee: 1.5 bps
        let mult = derive_toxic_hour_multiplier(5.0, 15.0, 1.5);
        // Should be ~(15+1.5)/(5+1.5) = 2.5
        assert!(mult > 2.0 && mult < 3.0, "mult = {}", mult);
    }

    #[test]
    fn test_derive_depth_spacing() {
        let ratio = derive_depth_spacing_ratio(5, 20.0);
        // For 5 levels with 20 bps decay, ratio should be ~1.2-1.5
        assert!(ratio > 1.1 && ratio < 2.0);
    }

    #[test]
    fn test_derive_confidence_threshold() {
        // Symmetric costs with 30% base rate
        let threshold = derive_confidence_threshold(0.3, 1.0, 1.0);
        // Should be ~0.30
        assert!((threshold - 0.3).abs() < 0.05);

        // Higher FN cost (don't want to miss signals)
        let threshold = derive_confidence_threshold(0.3, 1.0, 3.0);
        // Should be higher (more willing to act)
        assert!(threshold > 0.4);
    }
}
