//! EWMA + deadband parameter smoothing for MarketParams.
//!
//! Reduces parameter noise reaching the strategy by filtering oscillations
//! in gamma, kappa, sigma, lead_lag, and tail_risk. Supports regime-aware
//! reset with configurable cooldown and ramp-back.

use serde::{Deserialize, Serialize};

use crate::market_maker::strategy::market_params::MarketParams;

// === Serde default helpers ===

fn default_false() -> bool {
    false
}
fn default_sigma_alpha() -> f64 {
    0.15
}
fn default_sigma_deadband_pct() -> f64 {
    0.03
}
fn default_kappa_alpha() -> f64 {
    0.20
}
fn default_kappa_deadband_pct() -> f64 {
    0.05
}
fn default_gamma_mult_alpha() -> f64 {
    0.20
}
fn default_gamma_mult_deadband_pct() -> f64 {
    0.03
}
fn default_lead_lag_alpha() -> f64 {
    0.40
}
fn default_lead_lag_deadband_bps() -> f64 {
    1.5
}
fn default_tail_risk_alpha_up() -> f64 {
    0.80
}
fn default_tail_risk_alpha_down() -> f64 {
    0.15
}
fn default_tail_risk_deadband_pct() -> f64 {
    0.05
}
fn default_regime_cooldown_cycles() -> u32 {
    3
}
fn default_regime_ramp_cycles() -> u32 {
    5
}

/// Configuration for parameter smoothing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmootherConfig {
    /// Master enable/disable. Defaults to false for safe rollout.
    #[serde(default = "default_false")]
    pub enabled: bool,

    // --- Per-parameter EWMA alphas and deadbands ---
    /// EWMA alpha for sigma (clean volatility). Lower = smoother.
    #[serde(default = "default_sigma_alpha")]
    pub sigma_alpha: f64,
    /// Relative deadband for sigma (fraction, e.g. 0.03 = 3%).
    #[serde(default = "default_sigma_deadband_pct")]
    pub sigma_deadband_pct: f64,

    /// EWMA alpha for regime_kappa (only smoothed when Some).
    #[serde(default = "default_kappa_alpha")]
    pub kappa_alpha: f64,
    /// Relative deadband for kappa (fraction, e.g. 0.05 = 5%).
    #[serde(default = "default_kappa_deadband_pct")]
    pub kappa_deadband_pct: f64,

    /// EWMA alpha for regime_gamma_multiplier.
    #[serde(default = "default_gamma_mult_alpha")]
    pub gamma_mult_alpha: f64,
    /// Relative deadband for gamma_multiplier (fraction).
    #[serde(default = "default_gamma_mult_deadband_pct")]
    pub gamma_mult_deadband_pct: f64,

    /// EWMA alpha for lead_lag_signal_bps. Higher = faster tracking.
    #[serde(default = "default_lead_lag_alpha")]
    pub lead_lag_alpha: f64,
    /// Absolute deadband for lead_lag_signal_bps (in bps).
    #[serde(default = "default_lead_lag_deadband_bps")]
    pub lead_lag_deadband_bps: f64,

    /// EWMA alpha for tail_risk_multiplier when raw > smoothed (fast escalation).
    #[serde(default = "default_tail_risk_alpha_up")]
    pub tail_risk_alpha_up: f64,
    /// EWMA alpha for tail_risk_multiplier when raw < smoothed (slow relaxation).
    #[serde(default = "default_tail_risk_alpha_down")]
    pub tail_risk_alpha_down: f64,
    /// Relative deadband for tail_risk_multiplier (fraction).
    #[serde(default = "default_tail_risk_deadband_pct")]
    pub tail_risk_deadband_pct: f64,

    /// Regime change cooldown: ignore transitions within this many cycles
    /// of the last accepted regime change.
    #[serde(default = "default_regime_cooldown_cycles")]
    pub regime_cooldown_cycles: u32,

    /// Number of cycles to ramp from fast alpha (1.0) back to normal after regime change.
    #[serde(default = "default_regime_ramp_cycles")]
    pub regime_ramp_cycles: u32,
}

impl Default for SmootherConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sigma_alpha: default_sigma_alpha(),
            sigma_deadband_pct: default_sigma_deadband_pct(),
            kappa_alpha: default_kappa_alpha(),
            kappa_deadband_pct: default_kappa_deadband_pct(),
            gamma_mult_alpha: default_gamma_mult_alpha(),
            gamma_mult_deadband_pct: default_gamma_mult_deadband_pct(),
            lead_lag_alpha: default_lead_lag_alpha(),
            lead_lag_deadband_bps: default_lead_lag_deadband_bps(),
            tail_risk_alpha_up: default_tail_risk_alpha_up(),
            tail_risk_alpha_down: default_tail_risk_alpha_down(),
            tail_risk_deadband_pct: default_tail_risk_deadband_pct(),
            regime_cooldown_cycles: default_regime_cooldown_cycles(),
            regime_ramp_cycles: default_regime_ramp_cycles(),
        }
    }
}

/// Per-parameter EMA state.
#[derive(Debug, Clone)]
struct EmaState {
    /// Current smoothed value.
    value: f64,
    /// Whether the EMA has been initialized (first cycle seeds with raw).
    initialized: bool,
}

impl EmaState {
    fn new() -> Self {
        Self {
            value: 0.0,
            initialized: false,
        }
    }

    /// Reset to uninitialized state (used on regime change).
    fn reset(&mut self) {
        self.initialized = false;
        self.value = 0.0;
    }
}

/// EWMA + deadband smoother for MarketParams.
///
/// For each smoothed parameter:
/// 1. First cycle: initialize EMA to raw value (no smoothing on cold start)
/// 2. Regime change: reset EMA to raw, ramp alpha from 1.0 → normal over `ramp_cycles`
/// 3. Apply EWMA: `smoothed = alpha * raw + (1 - alpha) * prev`
/// 4. Apply deadband: suppress update if change is within threshold
/// 5. Asymmetric alpha for tail_risk (fast up, slow down)
pub struct ParameterSmoother {
    config: SmootherConfig,

    // EMA states
    sigma_ema: EmaState,
    kappa_ema: EmaState,
    gamma_mult_ema: EmaState,
    lead_lag_ema: EmaState,
    tail_risk_ema: EmaState,

    // Last output values (for deadband comparison)
    sigma_output: f64,
    kappa_output: Option<f64>,
    gamma_mult_output: f64,
    lead_lag_output: f64,
    tail_risk_output: f64,

    // Regime tracking
    last_regime: Option<u8>,
    regime_change_cycle: Option<u64>,
    cooldown_until_cycle: u64,
    cycle_count: u64,

    // Metrics
    pub deadband_suppressions: u64,
    pub total_updates: u64,
}

impl ParameterSmoother {
    pub fn new(config: SmootherConfig) -> Self {
        Self {
            config,
            sigma_ema: EmaState::new(),
            kappa_ema: EmaState::new(),
            gamma_mult_ema: EmaState::new(),
            lead_lag_ema: EmaState::new(),
            tail_risk_ema: EmaState::new(),
            sigma_output: 0.0,
            kappa_output: None,
            gamma_mult_output: 1.0,
            lead_lag_output: 0.0,
            tail_risk_output: 1.0,
            last_regime: None,
            regime_change_cycle: None,
            cooldown_until_cycle: 0,
            cycle_count: 0,
            deadband_suppressions: 0,
            total_updates: 0,
        }
    }

    /// Smooth MarketParams in place. Call once per quoting cycle.
    ///
    /// `regime_changed`: true if HMM detected a regime transition this cycle.
    pub fn smooth(&mut self, market_params: &mut MarketParams, regime_changed: bool) {
        if !self.config.enabled {
            self.cycle_count += 1;
            return;
        }

        // Handle regime change with cooldown
        let effective_regime_change = if regime_changed && self.cycle_count >= self.cooldown_until_cycle
        {
            self.regime_change_cycle = Some(self.cycle_count);
            self.cooldown_until_cycle =
                self.cycle_count + u64::from(self.config.regime_cooldown_cycles);
            // Reset all EMA states to pick up new regime parameters immediately
            self.sigma_ema.reset();
            self.kappa_ema.reset();
            self.gamma_mult_ema.reset();
            self.lead_lag_ema.reset();
            self.tail_risk_ema.reset();
            true
        } else {
            false
        };

        let _ = effective_regime_change; // used implicitly via EMA reset

        // Compute ramp factor: how far through the post-regime-change ramp we are.
        // 0.0 = just changed (use alpha=1.0), 1.0 = fully ramped (use normal alpha).
        let ramp_factor = self.ramp_factor();

        // --- Sigma ---
        {
            let alpha = ramp_alpha(self.config.sigma_alpha, ramp_factor);
            let (smoothed, just_init) = apply_ema(&mut self.sigma_ema, market_params.sigma, alpha);
            if just_init
                || !within_deadband(
                    smoothed,
                    self.sigma_output,
                    self.config.sigma_deadband_pct,
                    true,
                )
            {
                self.sigma_output = smoothed;
            } else {
                self.deadband_suppressions += 1;
            }
            market_params.sigma = self.sigma_output;
        }

        // --- Kappa (only when Some) ---
        if let Some(raw_kappa) = market_params.regime_kappa {
            let alpha = ramp_alpha(self.config.kappa_alpha, ramp_factor);
            let (smoothed, just_init) = apply_ema(&mut self.kappa_ema, raw_kappa, alpha);
            if just_init
                || self.kappa_output.is_none()
                || !within_deadband(
                    smoothed,
                    self.kappa_output.unwrap_or(raw_kappa),
                    self.config.kappa_deadband_pct,
                    true,
                )
            {
                self.kappa_output = Some(smoothed);
            } else {
                self.deadband_suppressions += 1;
            }
            market_params.regime_kappa = self.kappa_output;
        } else {
            // When None, reset kappa EMA so it re-initializes when kappa becomes available
            self.kappa_ema.reset();
            self.kappa_output = None;
        }

        // --- Gamma multiplier ---
        {
            let alpha = ramp_alpha(self.config.gamma_mult_alpha, ramp_factor);
            let (smoothed, just_init) = apply_ema(
                &mut self.gamma_mult_ema,
                market_params.regime_gamma_multiplier,
                alpha,
            );
            if just_init
                || !within_deadband(
                    smoothed,
                    self.gamma_mult_output,
                    self.config.gamma_mult_deadband_pct,
                    true,
                )
            {
                self.gamma_mult_output = smoothed;
            } else {
                self.deadband_suppressions += 1;
            }
            market_params.regime_gamma_multiplier = self.gamma_mult_output;
        }

        // --- Lead-lag signal (absolute deadband in bps) ---
        {
            let alpha = ramp_alpha(self.config.lead_lag_alpha, ramp_factor);
            let (smoothed, just_init) = apply_ema(
                &mut self.lead_lag_ema,
                market_params.lead_lag_signal_bps,
                alpha,
            );
            if just_init
                || !within_deadband(
                    smoothed,
                    self.lead_lag_output,
                    self.config.lead_lag_deadband_bps,
                    false,
                )
            {
                self.lead_lag_output = smoothed;
            } else {
                self.deadband_suppressions += 1;
            }
            market_params.lead_lag_signal_bps = self.lead_lag_output;
        }

        // --- Tail risk (asymmetric alpha) ---
        {
            let raw = market_params.tail_risk_multiplier;
            let base_alpha = if raw > self.tail_risk_ema.value || !self.tail_risk_ema.initialized {
                self.config.tail_risk_alpha_up
            } else {
                self.config.tail_risk_alpha_down
            };
            let alpha = ramp_alpha(base_alpha, ramp_factor);
            let (smoothed, just_init) = apply_ema(&mut self.tail_risk_ema, raw, alpha);
            if just_init
                || !within_deadband(
                    smoothed,
                    self.tail_risk_output,
                    self.config.tail_risk_deadband_pct,
                    true,
                )
            {
                self.tail_risk_output = smoothed;
            } else {
                self.deadband_suppressions += 1;
            }
            market_params.tail_risk_multiplier = self.tail_risk_output;
        }

        self.total_updates += 1;
        self.cycle_count += 1;
    }

    /// Clear all EMA states. Call on session restart.
    pub fn reset(&mut self) {
        self.sigma_ema.reset();
        self.kappa_ema.reset();
        self.gamma_mult_ema.reset();
        self.lead_lag_ema.reset();
        self.tail_risk_ema.reset();
        self.sigma_output = 0.0;
        self.kappa_output = None;
        self.gamma_mult_output = 1.0;
        self.lead_lag_output = 0.0;
        self.tail_risk_output = 1.0;
        self.last_regime = None;
        self.regime_change_cycle = None;
        self.cooldown_until_cycle = 0;
        self.cycle_count = 0;
        self.deadband_suppressions = 0;
        self.total_updates = 0;
    }

    /// Compute ramp factor: 0.0 immediately after regime change → 1.0 when fully ramped.
    fn ramp_factor(&self) -> f64 {
        let ramp_cycles = self.config.regime_ramp_cycles;
        if ramp_cycles == 0 {
            return 1.0;
        }
        match self.regime_change_cycle {
            Some(change_cycle) => {
                let elapsed = self.cycle_count.saturating_sub(change_cycle);
                (elapsed as f64 / ramp_cycles as f64).min(1.0)
            }
            None => 1.0, // No regime change yet — fully ramped
        }
    }
}

/// Apply EWMA: `smoothed = alpha * raw + (1 - alpha) * prev`.
/// On first call, seeds with the raw value (no smoothing).
/// Returns `(smoothed_value, just_initialized)` — callers should skip deadband
/// when `just_initialized` is true to avoid false suppressions against stale output.
fn apply_ema(state: &mut EmaState, raw: f64, alpha: f64) -> (f64, bool) {
    if !state.initialized {
        state.value = raw;
        state.initialized = true;
        return (raw, true);
    }
    let smoothed = alpha * raw + (1.0 - alpha) * state.value;
    state.value = smoothed;
    (smoothed, false)
}

/// Check whether a smoothed value is within the deadband of the current output.
///
/// `is_relative`: if true, deadband is a fraction of the current output (e.g. 0.03 = 3%).
///                if false, deadband is an absolute threshold (e.g. 1.5 bps).
fn within_deadband(smoothed: f64, current_output: f64, deadband: f64, is_relative: bool) -> bool {
    if current_output == 0.0 && !is_relative {
        // Absolute deadband: simple absolute difference
        return (smoothed - current_output).abs() < deadband;
    }
    if is_relative {
        if current_output.abs() < 1e-15 {
            // Can't compute relative deadband when output is ~0; always update
            return false;
        }
        let relative_change = (smoothed - current_output).abs() / current_output.abs();
        relative_change < deadband
    } else {
        (smoothed - current_output).abs() < deadband
    }
}

/// Blend alpha towards 1.0 based on ramp factor.
/// `ramp_factor=0.0` → alpha=1.0 (instant tracking after regime change).
/// `ramp_factor=1.0` → alpha=normal.
fn ramp_alpha(normal_alpha: f64, ramp_factor: f64) -> f64 {
    // Linear interpolation: 1.0 → normal_alpha as ramp_factor goes 0→1
    normal_alpha + (1.0 - normal_alpha) * (1.0 - ramp_factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_market_params() -> MarketParams {
        MarketParams::default()
    }

    fn enabled_config() -> SmootherConfig {
        SmootherConfig {
            enabled: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_passthrough_when_disabled() {
        let mut smoother = ParameterSmoother::new(SmootherConfig::default());
        assert!(!smoother.config.enabled);

        let mut params = default_market_params();
        params.sigma = 0.005;
        params.regime_kappa = Some(5000.0);
        params.regime_gamma_multiplier = 1.5;
        params.lead_lag_signal_bps = 3.0;
        params.tail_risk_multiplier = 2.0;

        smoother.smooth(&mut params, false);

        // All values pass through unchanged
        assert_eq!(params.sigma, 0.005);
        assert_eq!(params.regime_kappa, Some(5000.0));
        assert_eq!(params.regime_gamma_multiplier, 1.5);
        assert_eq!(params.lead_lag_signal_bps, 3.0);
        assert_eq!(params.tail_risk_multiplier, 2.0);
    }

    #[test]
    fn test_ema_first_cycle_initialization() {
        let mut smoother = ParameterSmoother::new(enabled_config());
        let mut params = default_market_params();
        params.sigma = 0.005;
        params.regime_gamma_multiplier = 1.5;
        params.lead_lag_signal_bps = 3.0;
        params.tail_risk_multiplier = 2.0;

        smoother.smooth(&mut params, false);

        // First cycle: EMA seeds with raw values
        assert_eq!(params.sigma, 0.005);
        assert_eq!(params.regime_gamma_multiplier, 1.5);
        assert_eq!(params.lead_lag_signal_bps, 3.0);
        assert_eq!(params.tail_risk_multiplier, 2.0);
    }

    #[test]
    fn test_ema_convergence() {
        let mut smoother = ParameterSmoother::new(enabled_config());

        // Run 200 cycles with constant sigma=0.01
        for _ in 0..200 {
            let mut params = default_market_params();
            params.sigma = 0.01;
            params.regime_gamma_multiplier = 1.0;
            params.lead_lag_signal_bps = 0.0;
            params.tail_risk_multiplier = 1.0;
            smoother.smooth(&mut params, false);
        }

        // After many cycles with constant input, EMA should converge
        let mut params = default_market_params();
        params.sigma = 0.01;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, false);

        assert!((params.sigma - 0.01).abs() < 1e-10, "sigma should converge to constant input");
    }

    #[test]
    fn test_deadband_suppression_relative() {
        let mut smoother = ParameterSmoother::new(SmootherConfig {
            enabled: true,
            sigma_alpha: 1.0, // instant tracking (no smoothing)
            sigma_deadband_pct: 0.05, // 5% deadband
            ..Default::default()
        });

        // Cycle 1: initialize
        let mut params = default_market_params();
        params.sigma = 1.0;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, false);
        assert_eq!(params.sigma, 1.0);

        // Cycle 2: small change within 5% deadband (1% change)
        let mut params = default_market_params();
        params.sigma = 1.01;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, false);
        // Should be suppressed — output stays at 1.0
        assert_eq!(params.sigma, 1.0);
        assert!(smoother.deadband_suppressions > 0);

        // Cycle 3: large change beyond 5% deadband
        let mut params = default_market_params();
        params.sigma = 1.10;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, false);
        // Should pass through
        assert!((params.sigma - 1.10).abs() < 1e-10);
    }

    #[test]
    fn test_deadband_suppression_absolute() {
        let mut smoother = ParameterSmoother::new(SmootherConfig {
            enabled: true,
            lead_lag_alpha: 1.0, // instant tracking
            lead_lag_deadband_bps: 2.0, // 2 bps absolute deadband
            ..Default::default()
        });

        // Cycle 1: initialize at 5.0 bps
        let mut params = default_market_params();
        params.sigma = 0.001;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 5.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, false);
        assert_eq!(params.lead_lag_signal_bps, 5.0);

        // Cycle 2: small change (1 bps, within 2 bps deadband)
        let mut params = default_market_params();
        params.sigma = 0.001;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 6.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, false);
        // Suppressed
        assert_eq!(params.lead_lag_signal_bps, 5.0);

        // Cycle 3: large change (4 bps, beyond deadband)
        let mut params = default_market_params();
        params.sigma = 0.001;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 9.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, false);
        assert!((params.lead_lag_signal_bps - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_regime_reset() {
        let mut smoother = ParameterSmoother::new(SmootherConfig {
            enabled: true,
            sigma_alpha: 0.10, // slow tracking
            regime_ramp_cycles: 0, // no ramp (for simpler test)
            ..Default::default()
        });

        // Run several cycles with sigma=1.0 to build up EMA
        for _ in 0..20 {
            let mut params = default_market_params();
            params.sigma = 1.0;
            params.regime_gamma_multiplier = 1.0;
            params.lead_lag_signal_bps = 0.0;
            params.tail_risk_multiplier = 1.0;
            smoother.smooth(&mut params, false);
        }

        // Regime change with sigma jumping to 2.0
        let mut params = default_market_params();
        params.sigma = 2.0;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, true);

        // After regime reset, EMA re-initializes to raw value
        assert_eq!(params.sigma, 2.0);
    }

    #[test]
    fn test_regime_cooldown() {
        let mut smoother = ParameterSmoother::new(SmootherConfig {
            enabled: true,
            sigma_alpha: 0.10,
            regime_cooldown_cycles: 5,
            regime_ramp_cycles: 0,
            ..Default::default()
        });

        // Cycle 0: init with sigma=1.0
        let mut params = default_market_params();
        params.sigma = 1.0;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, false);

        // Cycle 1: first regime change accepted
        let mut params = default_market_params();
        params.sigma = 2.0;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, true);
        assert_eq!(params.sigma, 2.0); // reset to raw

        // Cycle 2: another regime change within cooldown — should be IGNORED
        let _pre_suppressions = smoother.deadband_suppressions;
        let mut params = default_market_params();
        params.sigma = 3.0;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, true);
        // Should NOT reset — should apply normal EMA from previous value of 2.0
        // EMA: alpha=0.10 * 3.0 + 0.90 * 2.0 = 2.1. That's > 3% deadband from 2.0, so it updates.
        assert!(
            (params.sigma - 2.1).abs() < 1e-10,
            "should apply normal EMA, not regime reset. got {}",
            params.sigma
        );
    }

    #[test]
    fn test_regime_ramp() {
        let mut smoother = ParameterSmoother::new(SmootherConfig {
            enabled: true,
            sigma_alpha: 0.10,
            sigma_deadband_pct: 0.0, // disable deadband for this test
            regime_cooldown_cycles: 0,
            regime_ramp_cycles: 4,
            ..Default::default()
        });

        // Cycle 0: init
        let mut params = default_market_params();
        params.sigma = 1.0;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, false);

        // Cycle 1: regime change
        let mut params = default_market_params();
        params.sigma = 2.0;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, true);
        assert_eq!(params.sigma, 2.0); // reset to raw

        // Cycle 2: ramp_factor = (2-1)/4 = 0.25 → alpha = 0.10 + 0.90*(1-0.25) = 0.775
        let mut params = default_market_params();
        params.sigma = 2.5;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, false);
        // alpha=0.775: 0.775*2.5 + 0.225*2.0 = 1.9375 + 0.45 = 2.3875
        assert!(
            (params.sigma - 2.3875).abs() < 1e-10,
            "ramp alpha should be fast early. got {}",
            params.sigma
        );

        // Cycle 3: ramp_factor = (3-1)/4 = 0.50 → alpha = 0.10 + 0.90*0.50 = 0.55
        let mut params = default_market_params();
        params.sigma = 2.5;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, false);
        let expected = 0.55 * 2.5 + 0.45 * 2.3875;
        assert!(
            (params.sigma - expected).abs() < 1e-10,
            "ramp alpha should decrease. got {} expected {}",
            params.sigma,
            expected
        );
    }

    #[test]
    fn test_asymmetric_tail_risk_fast_up() {
        let mut smoother = ParameterSmoother::new(SmootherConfig {
            enabled: true,
            tail_risk_alpha_up: 0.80,
            tail_risk_alpha_down: 0.15,
            tail_risk_deadband_pct: 0.0, // disable deadband
            ..Default::default()
        });

        // Cycle 0: init at 1.0
        let mut params = default_market_params();
        params.sigma = 0.001;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, false);

        // Cycle 1: tail risk spikes to 3.0 — should track fast (alpha_up=0.80)
        let mut params = default_market_params();
        params.sigma = 0.001;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 3.0;
        smoother.smooth(&mut params, false);
        let expected = 0.80 * 3.0 + 0.20 * 1.0; // 2.6
        assert!(
            (params.tail_risk_multiplier - expected).abs() < 1e-10,
            "tail risk should escalate fast. got {} expected {}",
            params.tail_risk_multiplier,
            expected
        );
    }

    #[test]
    fn test_asymmetric_tail_risk_slow_down() {
        let mut smoother = ParameterSmoother::new(SmootherConfig {
            enabled: true,
            tail_risk_alpha_up: 0.80,
            tail_risk_alpha_down: 0.15,
            tail_risk_deadband_pct: 0.0,
            ..Default::default()
        });

        // Init at 3.0
        let mut params = default_market_params();
        params.sigma = 0.001;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 3.0;
        smoother.smooth(&mut params, false);

        // Cycle 1: tail risk drops to 1.0 — should relax slowly (alpha_down=0.15)
        let mut params = default_market_params();
        params.sigma = 0.001;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, false);
        let expected = 0.15 * 1.0 + 0.85 * 3.0; // 2.7
        assert!(
            (params.tail_risk_multiplier - expected).abs() < 1e-10,
            "tail risk should relax slowly. got {} expected {}",
            params.tail_risk_multiplier,
            expected
        );
    }

    #[test]
    fn test_kappa_none_handling() {
        let mut smoother = ParameterSmoother::new(enabled_config());

        // Cycle 1: kappa=None — should pass through
        let mut params = default_market_params();
        params.sigma = 0.001;
        params.regime_kappa = None;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, false);
        assert_eq!(params.regime_kappa, None);

        // Cycle 2: kappa becomes Some(5000) — initializes fresh
        let mut params = default_market_params();
        params.sigma = 0.001;
        params.regime_kappa = Some(5000.0);
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, false);
        assert_eq!(params.regime_kappa, Some(5000.0));

        // Cycle 3: kappa goes back to None — resets EMA
        let mut params = default_market_params();
        params.sigma = 0.001;
        params.regime_kappa = None;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, false);
        assert_eq!(params.regime_kappa, None);
        assert!(!smoother.kappa_ema.initialized, "kappa EMA should be reset when None");
    }

    #[test]
    fn test_deadband_count_metric() {
        let mut smoother = ParameterSmoother::new(SmootherConfig {
            enabled: true,
            sigma_alpha: 1.0,
            sigma_deadband_pct: 0.10, // 10% — generous deadband
            gamma_mult_alpha: 1.0,
            gamma_mult_deadband_pct: 0.10,
            lead_lag_alpha: 1.0,
            lead_lag_deadband_bps: 100.0, // very wide
            tail_risk_alpha_up: 1.0,
            tail_risk_alpha_down: 1.0,
            tail_risk_deadband_pct: 0.10,
            kappa_alpha: 1.0,
            kappa_deadband_pct: 0.10,
            ..Default::default()
        });

        // Cycle 0: init
        let mut params = default_market_params();
        params.sigma = 1.0;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 5.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, false);
        assert_eq!(smoother.deadband_suppressions, 0);

        // Cycle 1: tiny changes within all deadbands
        let mut params = default_market_params();
        params.sigma = 1.01; // 1% < 10% deadband
        params.regime_gamma_multiplier = 1.01;
        params.lead_lag_signal_bps = 5.5; // 0.5 < 100 bps deadband
        params.tail_risk_multiplier = 1.01;
        smoother.smooth(&mut params, false);

        // Should have at least 4 suppressions (one per smoothed param, kappa is None)
        assert!(
            smoother.deadband_suppressions >= 4,
            "expected >= 4 suppressions, got {}",
            smoother.deadband_suppressions
        );
    }

    #[test]
    fn test_default_config() {
        let config = SmootherConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.sigma_alpha, 0.15);
        assert_eq!(config.sigma_deadband_pct, 0.03);
        assert_eq!(config.kappa_alpha, 0.20);
        assert_eq!(config.kappa_deadband_pct, 0.05);
        assert_eq!(config.gamma_mult_alpha, 0.20);
        assert_eq!(config.gamma_mult_deadband_pct, 0.03);
        assert_eq!(config.lead_lag_alpha, 0.40);
        assert_eq!(config.lead_lag_deadband_bps, 1.5);
        assert_eq!(config.tail_risk_alpha_up, 0.80);
        assert_eq!(config.tail_risk_alpha_down, 0.15);
        assert_eq!(config.tail_risk_deadband_pct, 0.05);
        assert_eq!(config.regime_cooldown_cycles, 3);
        assert_eq!(config.regime_ramp_cycles, 5);
    }

    #[test]
    fn test_serde_default() {
        // Deserialize empty JSON → should get all defaults
        let config: SmootherConfig = serde_json::from_str("{}").unwrap();
        assert!(!config.enabled);
        assert_eq!(config.sigma_alpha, 0.15);
        assert_eq!(config.kappa_alpha, 0.20);
        assert_eq!(config.regime_cooldown_cycles, 3);
        assert_eq!(config.regime_ramp_cycles, 5);

        // Deserialize with only enabled=true → rest should be defaults
        let config: SmootherConfig = serde_json::from_str(r#"{"enabled": true}"#).unwrap();
        assert!(config.enabled);
        assert_eq!(config.sigma_alpha, 0.15);
    }

    #[test]
    fn test_reset_clears_all_state() {
        let mut smoother = ParameterSmoother::new(enabled_config());

        // Run a few cycles
        for _ in 0..5 {
            let mut params = default_market_params();
            params.sigma = 0.01;
            params.regime_gamma_multiplier = 1.5;
            params.lead_lag_signal_bps = 3.0;
            params.tail_risk_multiplier = 2.0;
            smoother.smooth(&mut params, false);
        }
        assert!(smoother.total_updates > 0);
        assert!(smoother.sigma_ema.initialized);

        smoother.reset();

        assert_eq!(smoother.total_updates, 0);
        assert_eq!(smoother.deadband_suppressions, 0);
        assert_eq!(smoother.cycle_count, 0);
        assert!(!smoother.sigma_ema.initialized);
        assert!(!smoother.kappa_ema.initialized);
        assert!(!smoother.gamma_mult_ema.initialized);
        assert!(!smoother.lead_lag_ema.initialized);
        assert!(!smoother.tail_risk_ema.initialized);
    }

    #[test]
    fn test_ema_tracking_step_change() {
        // Verify EMA tracks a step change at the right speed
        let mut smoother = ParameterSmoother::new(SmootherConfig {
            enabled: true,
            sigma_alpha: 0.20,
            sigma_deadband_pct: 0.0,
            ..Default::default()
        });

        // Init at 1.0
        let mut params = default_market_params();
        params.sigma = 1.0;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, false);

        // Step to 2.0 — after 1 cycle: 0.20*2 + 0.80*1 = 1.20
        let mut params = default_market_params();
        params.sigma = 2.0;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, false);
        assert!((params.sigma - 1.20).abs() < 1e-10);

        // After 2nd cycle: 0.20*2 + 0.80*1.20 = 1.36
        let mut params = default_market_params();
        params.sigma = 2.0;
        params.regime_gamma_multiplier = 1.0;
        params.lead_lag_signal_bps = 0.0;
        params.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut params, false);
        assert!((params.sigma - 1.36).abs() < 1e-10);
    }

    #[test]
    fn test_multiple_regime_changes_with_cooldown() {
        let mut smoother = ParameterSmoother::new(SmootherConfig {
            enabled: true,
            sigma_alpha: 0.10,
            sigma_deadband_pct: 0.0,
            regime_cooldown_cycles: 3,
            regime_ramp_cycles: 0,
            ..Default::default()
        });

        // Cycle 0: init
        let mut p = default_market_params();
        p.sigma = 1.0;
        p.regime_gamma_multiplier = 1.0;
        p.lead_lag_signal_bps = 0.0;
        p.tail_risk_multiplier = 1.0;
        smoother.smooth(&mut p, false);

        // Cycle 1: regime change accepted
        p.sigma = 2.0;
        smoother.smooth(&mut p, true);
        assert_eq!(p.sigma, 2.0);

        // Cycles 2-3: regime changes within cooldown — ignored
        p.sigma = 3.0;
        smoother.smooth(&mut p, true);
        assert!(p.sigma < 3.0, "should not reset during cooldown");

        p.sigma = 4.0;
        smoother.smooth(&mut p, true);
        assert!(p.sigma < 4.0, "should not reset during cooldown");

        // Cycle 4: past cooldown (cycle_count=4, cooldown_until=1+3=4, 4 >= 4) — accepted
        p.sigma = 5.0;
        smoother.smooth(&mut p, true);
        assert_eq!(p.sigma, 5.0, "should reset after cooldown expires");
    }
}
