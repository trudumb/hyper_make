//! Hidden Markov Model for regime detection.
//!
//! This module implements a proper HMM for market regime detection, replacing
//! the threshold-based classification with a probabilistic model that learns
//! transition probabilities and emission distributions from observed data.
//!
//! # Four Regime States
//!
//! - **Low (0)**: Very quiet market, can tighten spreads
//! - **Normal (1)**: Standard market conditions
//! - **High (2)**: Elevated volatility, widen spreads
//! - **Extreme (3)**: Crisis/cascade conditions, maximum caution
//!
//! # Key Insight
//!
//! Single parameter values are almost always wrong. Use the HMM belief state
//! to blend parameters based on regime probabilities:
//!
//! ```text
//! gamma_effective = P(low) * gamma_low + P(normal) * gamma_normal
//!                 + P(high) * gamma_high + P(extreme) * gamma_extreme
//! ```
//!
//! # Algorithm
//!
//! The HMM uses:
//! - Forward algorithm for filtering (online updates)
//! - Gaussian emissions for volatility and spread observations
//! - Dirichlet priors on transition probabilities for online learning
//! - Sticky diagonal transitions to prevent rapid switching

use std::collections::VecDeque;

use crate::market_maker::estimator::volatility::RegimeBeliefState;

/// Number of regimes in the HMM.
pub const NUM_REGIMES: usize = 4;

/// Regime indices for clarity.
pub mod regime_idx {
    pub const LOW: usize = 0;
    pub const NORMAL: usize = 1;
    pub const HIGH: usize = 2;
    pub const EXTREME: usize = 3;
}

/// Observation vector for the HMM.
///
/// Each observation contains multiple features that help distinguish regimes.
/// Key insight: volatility and spread are LAGGING indicators. We add OI and
/// liquidation pressure as LEADING indicators that can predict cascades.
#[derive(Debug, Clone, Copy)]
pub struct Observation {
    /// Realized volatility (per-second, e.g., 0.0002 for typical BTC)
    pub volatility: f64,
    /// Current spread in basis points (e.g., 5.0 for 5 bps)
    pub spread_bps: f64,
    /// Order flow imbalance [-1, 1] where positive = buy pressure
    pub flow_imbalance: f64,

    // === LEADING INDICATORS ===
    /// Open Interest level relative to recent average (ratio, 1.0 = average)
    /// Low OI can signal reduced liquidity/increased fragility
    pub oi_level: f64,

    /// Open Interest velocity: rate of change in OI
    /// Negative velocity (OI dropping) signals liquidations
    pub oi_velocity: f64,

    /// Liquidation pressure indicator: combines OI drop + extreme funding
    /// High values indicate forced selling pressure
    pub liquidation_pressure: f64,

    /// Hawkes intensity ratio: lambda_total / mu_baseline.
    /// 1.0 = baseline activity, >1 = clustering/burst, <1 = quiet.
    /// Replaces count-based burst detection with principled self-exciting process.
    pub hawkes_intensity_ratio: f64,
}

impl Default for Observation {
    fn default() -> Self {
        Self {
            volatility: 0.00025,
            spread_bps: 5.0,
            flow_imbalance: 0.0,
            oi_level: 1.0,               // Average
            oi_velocity: 0.0,            // No change
            liquidation_pressure: 0.0,   // No pressure
            hawkes_intensity_ratio: 1.0, // Baseline activity
        }
    }
}

impl Observation {
    /// Create a new observation (basic version for backward compatibility).
    pub fn new(volatility: f64, spread_bps: f64, flow_imbalance: f64) -> Self {
        Self {
            volatility,
            spread_bps,
            flow_imbalance: flow_imbalance.clamp(-1.0, 1.0),
            oi_level: 1.0,
            oi_velocity: 0.0,
            liquidation_pressure: 0.0,
            hawkes_intensity_ratio: 1.0,
        }
    }

    /// Create a full observation with all leading indicators.
    pub fn new_full(
        volatility: f64,
        spread_bps: f64,
        flow_imbalance: f64,
        oi_level: f64,
        oi_velocity: f64,
        liquidation_pressure: f64,
    ) -> Self {
        Self {
            volatility,
            spread_bps,
            flow_imbalance: flow_imbalance.clamp(-1.0, 1.0),
            oi_level: oi_level.max(0.0),
            oi_velocity,
            liquidation_pressure: liquidation_pressure.clamp(0.0, 1.0),
            hawkes_intensity_ratio: 1.0,
        }
    }

    /// Full constructor with Hawkes intensity ratio.
    ///
    /// Use when Hawkes estimator is warmed up and providing intensity data.
    /// The intensity ratio replaces count-based burst detection with a principled
    /// self-exciting process observation for the HMM.
    pub fn new_with_hawkes(
        volatility: f64,
        spread_bps: f64,
        flow_imbalance: f64,
        hawkes_intensity_ratio: f64,
    ) -> Self {
        Self {
            volatility,
            spread_bps,
            flow_imbalance: flow_imbalance.clamp(-1.0, 1.0),
            oi_level: 1.0,
            oi_velocity: 0.0,
            liquidation_pressure: 0.0,
            hawkes_intensity_ratio: hawkes_intensity_ratio.clamp(0.0, 50.0),
        }
    }

    /// Check if this observation has leading indicator data
    pub fn has_leading_indicators(&self) -> bool {
        self.oi_level != 1.0
            || self.oi_velocity != 0.0
            || self.liquidation_pressure != 0.0
            || self.hawkes_intensity_ratio != 1.0
    }
}

/// Emission distribution parameters for a single regime.
///
/// Uses Gaussian distributions for volatility and spread, with additional
/// parameters for leading indicators (OI and liquidation pressure).
#[derive(Debug, Clone, Copy)]
pub struct EmissionParams {
    /// Mean volatility for this regime (per-second)
    pub mean_volatility: f64,
    /// Standard deviation of volatility
    pub std_volatility: f64,
    /// Mean spread in basis points
    pub mean_spread: f64,
    /// Standard deviation of spread
    pub std_spread: f64,

    // === Leading indicator parameters ===
    /// Expected OI level for this regime (1.0 = average)
    /// Lower OI suggests cascade regime
    pub mean_oi_level: f64,
    /// Std of OI level
    pub std_oi_level: f64,

    /// Expected OI velocity for this regime
    /// Negative velocity suggests cascade (forced selling)
    pub mean_oi_velocity: f64,
    /// Std of OI velocity
    pub std_oi_velocity: f64,

    /// Weight for liquidation pressure signal (0-1)
    /// Higher weight in cascade regime
    pub liquidation_weight: f64,

    /// Mean Hawkes intensity ratio for this regime.
    /// 1.0 = baseline, higher = more trade clustering.
    pub mean_hawkes_intensity: f64,
    /// Standard deviation of Hawkes intensity ratio.
    pub std_hawkes_intensity: f64,
}

impl EmissionParams {
    /// Create new emission parameters (basic version).
    pub fn new(mean_vol: f64, std_vol: f64, mean_spread: f64, std_spread: f64) -> Self {
        Self {
            mean_volatility: mean_vol.max(1e-9),
            std_volatility: std_vol.max(1e-9),
            mean_spread: mean_spread.max(0.1),
            std_spread: std_spread.max(0.1),
            // Defaults for leading indicators
            mean_oi_level: 1.0,
            std_oi_level: 0.2,
            mean_oi_velocity: 0.0,
            std_oi_velocity: 0.05,
            liquidation_weight: 0.0,
            // Hawkes defaults: baseline intensity, moderate spread
            mean_hawkes_intensity: 1.0,
            std_hawkes_intensity: 0.5,
        }
    }

    /// Apply validation constraints: clamp and enforce minimums on all fields.
    ///
    /// Use with struct literal syntax:
    /// ```ignore
    /// EmissionParams { mean_volatility: 0.001, ... }.validated()
    /// ```
    pub fn validated(mut self) -> Self {
        self.mean_volatility = self.mean_volatility.max(1e-9);
        self.std_volatility = self.std_volatility.max(1e-9);
        self.mean_spread = self.mean_spread.max(0.1);
        self.std_spread = self.std_spread.max(0.1);
        self.mean_oi_level = self.mean_oi_level.max(0.1);
        self.std_oi_level = self.std_oi_level.max(0.01);
        // mean_oi_velocity has no constraint (can be negative)
        self.std_oi_velocity = self.std_oi_velocity.max(0.01);
        self.liquidation_weight = self.liquidation_weight.clamp(0.0, 1.0);
        self.mean_hawkes_intensity = self.mean_hawkes_intensity.max(0.1);
        self.std_hawkes_intensity = self.std_hawkes_intensity.max(0.05);
        self
    }

    /// Compute log-likelihood of observation under this emission distribution.
    fn log_likelihood(&self, obs: &Observation) -> f64 {
        // Log-likelihood of volatility (Gaussian)
        let vol_z = (obs.volatility - self.mean_volatility) / self.std_volatility;
        let ll_vol = -0.5 * vol_z * vol_z - self.std_volatility.ln();

        // Log-likelihood of spread (Gaussian)
        let spread_z = (obs.spread_bps - self.mean_spread) / self.std_spread;
        let ll_spread = -0.5 * spread_z * spread_z - self.std_spread.ln();

        // Log-likelihood of OI level (Gaussian) - only if data provided
        let ll_oi = if obs.oi_level != 1.0 {
            let oi_z = (obs.oi_level - self.mean_oi_level) / self.std_oi_level;
            -0.5 * oi_z * oi_z - self.std_oi_level.ln()
        } else {
            0.0 // No contribution if no data
        };

        // Log-likelihood of OI velocity (Gaussian) - only if data provided
        let ll_oi_vel = if obs.oi_velocity != 0.0 {
            let vel_z = (obs.oi_velocity - self.mean_oi_velocity) / self.std_oi_velocity;
            -0.5 * vel_z * vel_z - self.std_oi_velocity.ln()
        } else {
            0.0
        };

        // Liquidation pressure contribution
        // High pressure boosts cascade regime likelihood
        let ll_liq = if obs.liquidation_pressure > 0.0 {
            // Log-odds style: positive contribution if pressure matches regime expectation
            self.liquidation_weight * obs.liquidation_pressure.ln().max(-10.0)
        } else {
            0.0
        };

        // Log-likelihood of Hawkes intensity ratio (Gaussian) - only if non-default
        let ll_hawkes = if (obs.hawkes_intensity_ratio - 1.0).abs() > 1e-6 {
            let hawkes_z = (obs.hawkes_intensity_ratio - self.mean_hawkes_intensity)
                / self.std_hawkes_intensity;
            -0.5 * hawkes_z * hawkes_z - self.std_hawkes_intensity.ln()
        } else {
            0.0 // No contribution if at default (Hawkes not warmed up)
        };

        // Combine (treating as independent features)
        // Hawkes weighted at 0.5 like OI — informative but not dominant
        ll_vol + ll_spread + ll_oi * 0.5 + ll_oi_vel * 0.5 + ll_liq + ll_hawkes * 0.5
    }

    /// Compute likelihood (not log) of observation.
    pub fn likelihood(&self, obs: &Observation) -> f64 {
        self.log_likelihood(obs).exp().max(1e-300)
    }
}

/// Default emission parameters for each regime.
///
/// These are calibrated for typical crypto market conditions (BTC):
/// - Low: Very quiet, vol ~0.1%, tight spreads ~3 bps
/// - Normal: Standard, vol ~0.25%, moderate spreads ~5 bps
/// - High: Elevated, vol ~1%, wider spreads ~10 bps
/// - Extreme: Crisis/cascade, vol ~5%, very wide spreads ~25+ bps
///
/// Leading indicators for cascade detection:
/// - OI level: Cascade shows OI dropping (< 1.0)
/// - OI velocity: Cascade shows negative velocity (forced liquidations)
/// - Liquidation pressure: High in cascade, low otherwise
fn default_emission_params() -> [EmissionParams; NUM_REGIMES] {
    [
        // Low regime: quiet market, stable OI, below-baseline Hawkes
        EmissionParams {
            mean_volatility: 0.001,
            std_volatility: 0.0005,
            mean_spread: 3.0,
            std_spread: 1.5,
            mean_oi_level: 1.05,
            std_oi_level: 0.1,
            mean_oi_velocity: 0.0,
            std_oi_velocity: 0.02,
            liquidation_weight: 0.0,
            mean_hawkes_intensity: 0.5,
            std_hawkes_intensity: 0.2,
        }
        .validated(),
        // Normal regime: standard conditions, baseline Hawkes
        EmissionParams {
            mean_volatility: 0.0025,
            std_volatility: 0.001,
            mean_spread: 5.0,
            std_spread: 2.0,
            mean_oi_level: 1.0,
            std_oi_level: 0.15,
            mean_oi_velocity: 0.0,
            std_oi_velocity: 0.03,
            liquidation_weight: 0.0,
            mean_hawkes_intensity: 1.0,
            std_hawkes_intensity: 0.3,
        }
        .validated(),
        // High regime: elevated vol, OI starting to drop, clustering trades
        EmissionParams {
            mean_volatility: 0.01,
            std_volatility: 0.005,
            mean_spread: 10.0,
            std_spread: 5.0,
            mean_oi_level: 0.95,
            std_oi_level: 0.2,
            mean_oi_velocity: -0.02,
            std_oi_velocity: 0.05,
            liquidation_weight: 0.2,
            mean_hawkes_intensity: 2.5,
            std_hawkes_intensity: 0.8,
        }
        .validated(),
        // Extreme/Cascade regime: crisis conditions, OI dropping fast, intense clustering
        EmissionParams {
            mean_volatility: 0.05,
            std_volatility: 0.025,
            mean_spread: 25.0,
            std_spread: 15.0,
            mean_oi_level: 0.8,
            std_oi_level: 0.25,
            mean_oi_velocity: -0.1,
            std_oi_velocity: 0.08,
            liquidation_weight: 0.8,
            mean_hawkes_intensity: 5.0,
            std_hawkes_intensity: 2.0,
        }
        .validated(),
    ]
}

/// Default transition matrix with sticky diagonal.
///
/// Diagonal elements are 0.95 (sticky - regimes persist).
/// Off-diagonal elements split remaining 0.05 probability among adjacent regimes,
/// with stronger probability for transitions to neighboring regimes.
fn default_transition_matrix() -> [[f64; NUM_REGIMES]; NUM_REGIMES] {
    // Sticky factor on diagonal
    let diag = 0.95;
    let off_diag = (1.0 - diag) / 3.0; // Equal split among other states

    [
        // From Low: mostly stay, can go to Normal
        [diag, off_diag * 2.0, off_diag * 0.8, off_diag * 0.2],
        // From Normal: can go either direction
        [off_diag, diag, off_diag * 1.5, off_diag * 0.5],
        // From High: can go to Normal or Extreme
        [off_diag * 0.5, off_diag * 1.5, diag, off_diag],
        // From Extreme: mostly stay or go to High
        [off_diag * 0.2, off_diag * 0.5, off_diag * 2.3, diag],
    ]
}

/// Default Dirichlet prior counts for transition learning.
///
/// Uses pseudo-counts that encode our prior belief about transition structure.
/// Higher counts = stronger prior = slower adaptation.
fn default_prior_counts() -> [[f64; NUM_REGIMES]; NUM_REGIMES] {
    // Start with moderate pseudo-counts (equivalent to ~20 effective observations per row)
    let diag_count = 19.0;
    let off_diag_count = 1.0 / 3.0;

    [
        [
            diag_count,
            off_diag_count * 2.0,
            off_diag_count * 0.8,
            off_diag_count * 0.2,
        ],
        [
            off_diag_count,
            diag_count,
            off_diag_count * 1.5,
            off_diag_count * 0.5,
        ],
        [
            off_diag_count * 0.5,
            off_diag_count * 1.5,
            diag_count,
            off_diag_count,
        ],
        [
            off_diag_count * 0.2,
            off_diag_count * 0.5,
            off_diag_count * 2.3,
            diag_count,
        ],
    ]
}

/// Hidden Markov Model for regime detection.
///
/// Implements online filtering using the forward algorithm with
/// Gaussian emission distributions and learnable transition probabilities.
#[derive(Debug, Clone)]
pub struct RegimeHMM {
    /// Transition matrix: P(regime_t | regime_{t-1})
    /// transition_matrix[i][j] = P(regime_t = j | regime_{t-1} = i)
    transition_matrix: [[f64; NUM_REGIMES]; NUM_REGIMES],

    /// Emission parameters for each regime's observation distribution
    emission_params: [EmissionParams; NUM_REGIMES],

    /// Current regime probabilities (filtered belief state)
    belief: [f64; NUM_REGIMES],

    /// Dirichlet prior counts for transition probability learning
    /// prior_counts[i][j] = pseudo-count for transition i -> j
    prior_counts: [[f64; NUM_REGIMES]; NUM_REGIMES],

    /// Observed transition counts for online learning
    transition_counts: [[f64; NUM_REGIMES]; NUM_REGIMES],

    /// Number of observations processed
    observation_count: u64,

    /// Learning rate for emission parameter updates
    emission_learning_rate: f64,

    // === Auto-calibration state ===
    /// Rolling volatility buffer for calibration (decaying window).
    vol_buffer: VecDeque<f64>,
    /// Rolling spread buffer for calibration (decaying window).
    spread_buffer: VecDeque<f64>,
    /// Number of observations needed before initial auto-calibration triggers.
    calibration_buffer_size: usize,
    /// Whether initial auto-calibration has been performed.
    initial_calibration_done: bool,
    /// Maximum window size for rolling recalibration buffers.
    recalibration_window: usize,
    /// Number of new observations between periodic recalibrations.
    recalibration_interval: usize,
    /// Observations accumulated since last recalibration.
    observations_since_recalibration: usize,
    /// Total recalibrations performed (for diagnostics).
    recalibration_count: usize,

    // === Online soft EM state ===
    /// Previous belief state (for soft transition count updates).
    prev_belief: [f64; NUM_REGIMES],

    /// Soft emission sufficient statistics: per-regime running mean of volatility.
    /// Updated via EWMA weighted by regime responsibility (soft assignment).
    emission_vol_sum: [f64; NUM_REGIMES],
    /// Soft emission sufficient statistics: per-regime running mean of spread.
    emission_spread_sum: [f64; NUM_REGIMES],
    /// Soft emission sufficient statistics: per-regime running variance of volatility.
    emission_vol_sq_sum: [f64; NUM_REGIMES],
    /// Soft emission sufficient statistics: per-regime running variance of spread.
    emission_spread_sq_sum: [f64; NUM_REGIMES],
    /// Soft emission sufficient statistics: per-regime running mean of Hawkes intensity.
    emission_hawkes_sum: [f64; NUM_REGIMES],
    /// Soft emission sufficient statistics: per-regime running variance of Hawkes intensity.
    emission_hawkes_sq_sum: [f64; NUM_REGIMES],
    /// Effective sample count per regime (sum of soft assignments).
    emission_effective_n: [f64; NUM_REGIMES],
}

impl Default for RegimeHMM {
    fn default() -> Self {
        Self::new()
    }
}

impl RegimeHMM {
    /// Create a new HMM with sensible defaults.
    ///
    /// Defaults:
    /// - Sticky diagonal (0.95 probability of staying in same regime)
    /// - Regime-appropriate emission distributions calibrated for crypto
    /// - Dirichlet prior for Bayesian transition learning
    pub fn new() -> Self {
        Self {
            transition_matrix: default_transition_matrix(),
            emission_params: default_emission_params(),
            belief: [0.1, 0.6, 0.25, 0.05], // Start with Normal most likely
            prior_counts: default_prior_counts(),
            transition_counts: [[0.0; NUM_REGIMES]; NUM_REGIMES],
            observation_count: 0,
            emission_learning_rate: 0.01,
            vol_buffer: VecDeque::with_capacity(2000),
            spread_buffer: VecDeque::with_capacity(2000),
            calibration_buffer_size: 600,
            initial_calibration_done: false,
            recalibration_window: 2000,
            recalibration_interval: 500,
            observations_since_recalibration: 0,
            recalibration_count: 0,
            prev_belief: [0.1, 0.6, 0.25, 0.05],
            emission_vol_sum: [0.0; NUM_REGIMES],
            emission_spread_sum: [0.0; NUM_REGIMES],
            emission_vol_sq_sum: [0.0; NUM_REGIMES],
            emission_spread_sq_sum: [0.0; NUM_REGIMES],
            emission_hawkes_sum: [0.0; NUM_REGIMES],
            emission_hawkes_sq_sum: [0.0; NUM_REGIMES],
            emission_effective_n: [0.0; NUM_REGIMES],
        }
    }

    /// Create HMM with custom transition matrix.
    pub fn with_transition_matrix(mut self, matrix: [[f64; NUM_REGIMES]; NUM_REGIMES]) -> Self {
        self.transition_matrix = matrix;
        self
    }

    /// Create HMM with custom emission parameters.
    pub fn with_emission_params(mut self, params: [EmissionParams; NUM_REGIMES]) -> Self {
        self.emission_params = params;
        self
    }

    /// Create HMM with custom initial belief.
    pub fn with_initial_belief(mut self, belief: [f64; NUM_REGIMES]) -> Self {
        self.belief = belief;
        self.normalize_belief();
        self
    }

    /// Create HMM with custom emission learning rate.
    pub fn with_emission_learning_rate(mut self, rate: f64) -> Self {
        self.emission_learning_rate = rate.clamp(0.001, 0.1);
        self
    }

    /// Create HMM with emission params scaled relative to a baseline volatility.
    ///
    /// Instead of using hardcoded absolute thresholds (which may be unreachable
    /// for many assets), this scales emission parameters as multiples of the
    /// asset's baseline volatility and spread:
    /// - Low: 0.3x baseline
    /// - Normal: 1.0x baseline
    /// - High: 3.0x baseline
    /// - Extreme: 10.0x baseline
    ///
    /// This makes the HMM immediately usable for any asset without warmup.
    pub fn with_baseline_volatility(mut self, baseline_vol: f64, baseline_spread_bps: f64) -> Self {
        let bv = baseline_vol.max(1e-6);
        let bs = baseline_spread_bps.max(0.5);

        self.emission_params[regime_idx::LOW] =
            EmissionParams::new(bv * 0.3, bv * 0.15, bs * 0.6, bs * 0.3);
        self.emission_params[regime_idx::NORMAL] = EmissionParams::new(bv, bv * 0.4, bs, bs * 0.4);
        self.emission_params[regime_idx::HIGH] =
            EmissionParams::new(bv * 3.0, bv * 1.5, bs * 2.0, bs * 1.0);
        self.emission_params[regime_idx::EXTREME] =
            EmissionParams::new(bv * 10.0, bv * 5.0, bs * 5.0, bs * 3.0);

        // Mark as pre-calibrated so auto-calibration doesn't override
        self.initial_calibration_done = true;
        self.seed_em_sums_from_emissions();

        self
    }

    /// Seed emission distributions from a MarketProfile's observable parameters.
    ///
    /// Convenience wrapper around `with_baseline_volatility` logic that takes
    /// raw f64 parameters (extracted from MarketProfile by the caller).
    /// This avoids a direct dependency on the MarketProfile struct.
    ///
    /// Enforces minimum regime separation: HIGH vol >= 2x NORMAL vol.
    ///
    /// # Arguments
    /// * `normal_vol` - Normal-condition per-second volatility (from `MarketProfile::implied_sigma()`)
    /// * `normal_spread_bps` - Normal-condition BBO spread in bps (from `MarketProfile::bbo_half_spread_bps()`)
    pub fn seed_emissions_from_profile(&mut self, normal_vol: f64, normal_spread_bps: f64) {
        let vol = normal_vol.max(1e-6);
        let spread = normal_spread_bps.max(1.0);

        // Set emissions using the same multiplier pattern as with_baseline_volatility:
        // Low: 0.3x, Normal: 1.0x, High: 3.0x, Extreme: 10.0x
        self.emission_params[regime_idx::LOW] =
            EmissionParams::new(vol * 0.3, vol * 0.15, spread * 0.6, spread * 0.3);
        self.emission_params[regime_idx::NORMAL] =
            EmissionParams::new(vol, vol * 0.4, spread, spread * 0.4);
        self.emission_params[regime_idx::HIGH] =
            EmissionParams::new(vol * 3.0, vol * 1.5, spread * 2.0, spread * 1.0);
        self.emission_params[regime_idx::EXTREME] =
            EmissionParams::new(vol * 10.0, vol * 5.0, spread * 5.0, spread * 3.0);

        // Enforce regime separation: HIGH vol >= 2x NORMAL vol
        let normal_vol_mean = self.emission_params[regime_idx::NORMAL].mean_volatility;
        let high_vol_mean = self.emission_params[regime_idx::HIGH].mean_volatility;
        if high_vol_mean < normal_vol_mean * 2.0 {
            self.emission_params[regime_idx::HIGH].mean_volatility = normal_vol_mean * 2.0;
        }

        self.initial_calibration_done = true;

        // Seed soft EM running sums from emission params so EWMA starts correctly
        self.seed_em_sums_from_emissions();
    }

    /// Seed soft EM running statistics from current emission parameters.
    /// Called after any emission initialization to avoid cold-start EWMA drift.
    fn seed_em_sums_from_emissions(&mut self) {
        for k in 0..NUM_REGIMES {
            self.emission_vol_sum[k] = self.emission_params[k].mean_volatility;
            self.emission_spread_sum[k] = self.emission_params[k].mean_spread;
            self.emission_vol_sq_sum[k] = self.emission_params[k].std_volatility.powi(2);
            self.emission_spread_sq_sum[k] = self.emission_params[k].std_spread.powi(2);
            self.emission_hawkes_sum[k] = self.emission_params[k].mean_hawkes_intensity;
            self.emission_hawkes_sq_sum[k] = self.emission_params[k].std_hawkes_intensity.powi(2);
            // Start effective_n at 0 — prior dominates until enough soft counts
            self.emission_effective_n[k] = 0.0;
        }
    }

    /// Set the calibration buffer size (number of observations before auto-calibration).
    pub fn with_calibration_buffer_size(mut self, size: usize) -> Self {
        self.calibration_buffer_size = size.max(50);
        self
    }

    /// Configure periodic recalibration from rolling window.
    ///
    /// - `window`: max observations kept in rolling buffer (default 2000)
    /// - `interval`: recalibrate every N new observations after initial (default 500)
    pub fn with_recalibration(mut self, window: usize, interval: usize) -> Self {
        self.recalibration_window = window.max(200);
        self.recalibration_interval = interval.max(50);
        self
    }

    /// Get number of recalibrations performed so far.
    pub fn recalibration_count(&self) -> usize {
        self.recalibration_count
    }

    /// Perform one step of the forward algorithm to update belief state.
    ///
    /// This is the core online filtering operation:
    /// 1. Predict: P(z_t | y_{1:t-1}) = sum_j P(z_t | z_{t-1}=j) P(z_{t-1}=j | y_{1:t-1})
    /// 2. Update: P(z_t | y_{1:t}) proportional to P(y_t | z_t) P(z_t | y_{1:t-1})
    ///
    /// Returns the updated belief state.
    pub fn forward_update(&mut self, observation: &Observation) -> [f64; NUM_REGIMES] {
        // Always accumulate into rolling buffers for both initial and periodic recalibration
        if observation.volatility.is_finite() && observation.volatility > 0.0 {
            self.vol_buffer.push_back(observation.volatility);
            while self.vol_buffer.len() > self.recalibration_window {
                self.vol_buffer.pop_front();
            }
        }
        if observation.spread_bps.is_finite() && observation.spread_bps > 0.0 {
            self.spread_buffer.push_back(observation.spread_bps);
            while self.spread_buffer.len() > self.recalibration_window {
                self.spread_buffer.pop_front();
            }
        }

        // Initial auto-calibration: trigger once we have enough observations
        if !self.initial_calibration_done
            && self.vol_buffer.len() >= self.calibration_buffer_size
            && self.spread_buffer.len() >= self.calibration_buffer_size
        {
            let vols: Vec<f64> = self.vol_buffer.iter().copied().collect();
            let spreads: Vec<f64> = self.spread_buffer.iter().copied().collect();
            self.calibrate_from_observations(&vols, &spreads);
            self.initial_calibration_done = true;
            self.observations_since_recalibration = 0;
            self.recalibration_count = 1;

            tracing::info!(
                n_vol = vols.len(),
                n_spread = spreads.len(),
                low_vol = %format!("{:.6}", self.emission_params[regime_idx::LOW].mean_volatility),
                normal_vol = %format!("{:.6}", self.emission_params[regime_idx::NORMAL].mean_volatility),
                high_vol = %format!("{:.6}", self.emission_params[regime_idx::HIGH].mean_volatility),
                extreme_vol = %format!("{:.6}", self.emission_params[regime_idx::EXTREME].mean_volatility),
                "HMM auto-calibrated emission thresholds from observed data"
            );
        }

        // Periodic recalibration: every recalibration_interval observations after initial
        if self.initial_calibration_done {
            self.observations_since_recalibration += 1;
            if self.observations_since_recalibration >= self.recalibration_interval
                && self.vol_buffer.len() >= self.calibration_buffer_size
            {
                let vols: Vec<f64> = self.vol_buffer.iter().copied().collect();
                let spreads: Vec<f64> = self.spread_buffer.iter().copied().collect();
                self.calibrate_from_observations(&vols, &spreads);
                self.observations_since_recalibration = 0;
                self.recalibration_count += 1;

                tracing::debug!(
                    recalibration_count = self.recalibration_count,
                    window_size = vols.len(),
                    low_vol = %format!("{:.6}", self.emission_params[regime_idx::LOW].mean_volatility),
                    normal_vol = %format!("{:.6}", self.emission_params[regime_idx::NORMAL].mean_volatility),
                    high_vol = %format!("{:.6}", self.emission_params[regime_idx::HIGH].mean_volatility),
                    extreme_vol = %format!("{:.6}", self.emission_params[regime_idx::EXTREME].mean_volatility),
                    "HMM periodic recalibration from rolling window"
                );
            }
        }

        // Step 1: Prediction (time update)
        // predicted[j] = sum_i transition_matrix[i][j] * belief[i]
        let mut predicted = [0.0; NUM_REGIMES];
        for (j, pred) in predicted.iter_mut().enumerate().take(NUM_REGIMES) {
            for i in 0..NUM_REGIMES {
                *pred += self.transition_matrix[i][j] * self.belief[i];
            }
        }

        // Step 2: Update (measurement update)
        // new_belief[j] proportional to emission_likelihood(j, obs) * predicted[j]
        for (j, &pred) in predicted.iter().enumerate().take(NUM_REGIMES) {
            let likelihood = self.emission_likelihood(j, observation);
            self.belief[j] = likelihood * pred;
        }

        // Normalize to ensure probabilities sum to 1
        self.normalize_belief();

        // === Online Soft EM ===
        // Update emission sufficient statistics using soft assignments (responsibilities).
        // This is the E-step + partial M-step of online EM: each observation contributes
        // to all regimes proportional to its posterior responsibility γ(t,k).
        if self.initial_calibration_done && self.observation_count > 0 {
            let alpha = self.emission_learning_rate;

            for k in 0..NUM_REGIMES {
                let responsibility = self.belief[k];
                if responsibility < 1e-9 {
                    continue;
                }

                // Accumulate effective sample count per regime
                self.emission_effective_n[k] += responsibility;

                // EWMA update of emission means weighted by responsibility.
                // Effective learning rate = alpha * responsibility ensures rarely-visited
                // regimes update slowly (preventing collapse from noise).
                let eff_alpha = alpha * responsibility;

                // Update running means
                self.emission_vol_sum[k] = (1.0 - eff_alpha) * self.emission_vol_sum[k]
                    + eff_alpha * observation.volatility;
                self.emission_spread_sum[k] = (1.0 - eff_alpha) * self.emission_spread_sum[k]
                    + eff_alpha * observation.spread_bps;

                // Update running squared deviations (for variance estimation)
                let vol_dev = observation.volatility - self.emission_params[k].mean_volatility;
                self.emission_vol_sq_sum[k] =
                    (1.0 - eff_alpha) * self.emission_vol_sq_sum[k] + eff_alpha * vol_dev * vol_dev;

                let spread_dev = observation.spread_bps - self.emission_params[k].mean_spread;
                self.emission_spread_sq_sum[k] = (1.0 - eff_alpha) * self.emission_spread_sq_sum[k]
                    + eff_alpha * spread_dev * spread_dev;

                // Update Hawkes intensity running statistics (only if non-default)
                if (observation.hawkes_intensity_ratio - 1.0).abs() > 1e-6 {
                    self.emission_hawkes_sum[k] = (1.0 - eff_alpha) * self.emission_hawkes_sum[k]
                        + eff_alpha * observation.hawkes_intensity_ratio;

                    let hawkes_dev = observation.hawkes_intensity_ratio
                        - self.emission_params[k].mean_hawkes_intensity;
                    self.emission_hawkes_sq_sum[k] = (1.0 - eff_alpha)
                        * self.emission_hawkes_sq_sum[k]
                        + eff_alpha * hawkes_dev * hawkes_dev;
                }
            }

            // Soft transition counts: P(z_{t-1}=i, z_t=j | y_{1:t})
            // Approximated by prev_belief[i] * transition[i][j] * emission_likelihood(j) / normalizer
            // Simplified: use prev_belief[i] * belief[j] (already normalized)
            for i in 0..NUM_REGIMES {
                for j in 0..NUM_REGIMES {
                    let soft_count = self.prev_belief[i] * self.belief[j];
                    self.transition_counts[i][j] += soft_count;
                }
            }

            // Update transition matrix from Dirichlet posterior mean
            for i in 0..NUM_REGIMES {
                let row_sum: f64 = (0..NUM_REGIMES)
                    .map(|j| self.prior_counts[i][j] + self.transition_counts[i][j])
                    .sum();
                if row_sum > 1e-9 {
                    for j in 0..NUM_REGIMES {
                        self.transition_matrix[i][j] =
                            (self.prior_counts[i][j] + self.transition_counts[i][j]) / row_sum;
                    }
                }
            }

            // Periodic M-step: apply accumulated sufficient statistics to emission params.
            // Only every 50 observations to avoid excessive updates.
            if self.observation_count.is_multiple_of(50) {
                self.apply_soft_em_emissions();
            }
        }

        // Store current belief for next soft transition update
        self.prev_belief = self.belief;

        // Update observation count
        self.observation_count += 1;

        self.belief
    }

    /// Apply accumulated soft EM sufficient statistics to emission parameters.
    ///
    /// Uses Normal-InverseGamma prior structure to prevent emission collapse:
    /// - Minimum effective sample count of 5.0 before updating (prior dominates)
    /// - Variance floor at 30% of mean (prevents degenerate point distributions)
    /// - Regime separation enforced: HIGH ≥ 1.5× NORMAL, EXTREME ≥ 2× HIGH
    fn apply_soft_em_emissions(&mut self) {
        /// Minimum effective observations before soft EM updates emission params.
        const MIN_EFFECTIVE_N: f64 = 5.0;

        for k in 0..NUM_REGIMES {
            if self.emission_effective_n[k] < MIN_EFFECTIVE_N {
                continue; // Prior dominates — don't update
            }

            let params = &mut self.emission_params[k];

            // Blend factor: increases with effective sample count
            // At n=5: blend=0.5, at n=50: blend=0.91, at n=500: blend=0.99
            let blend =
                self.emission_effective_n[k] / (self.emission_effective_n[k] + MIN_EFFECTIVE_N);

            // Update means with blend toward soft EM estimate
            let new_vol_mean =
                blend * self.emission_vol_sum[k] + (1.0 - blend) * params.mean_volatility;
            let new_spread_mean =
                blend * self.emission_spread_sum[k] + (1.0 - blend) * params.mean_spread;

            params.mean_volatility = new_vol_mean.max(1e-9);
            params.mean_spread = new_spread_mean.max(0.1);

            // Update standard deviations with variance floor (Normal-InverseGamma prior).
            // Floor = 30% of mean prevents collapse when regime rarely visited.
            let vol_std_floor = params.mean_volatility * 0.3;
            let spread_std_floor = params.mean_spread * 0.3;

            let new_vol_std =
                blend * self.emission_vol_sq_sum[k].sqrt() + (1.0 - blend) * params.std_volatility;
            let new_spread_std =
                blend * self.emission_spread_sq_sum[k].sqrt() + (1.0 - blend) * params.std_spread;

            params.std_volatility = new_vol_std.max(vol_std_floor).max(1e-9);
            params.std_spread = new_spread_std.max(spread_std_floor).max(0.1);

            // Update Hawkes intensity emission params (only if EM has accumulated data)
            if self.emission_hawkes_sum[k] > 0.0 {
                let new_hawkes_mean = blend * self.emission_hawkes_sum[k]
                    + (1.0 - blend) * params.mean_hawkes_intensity;
                params.mean_hawkes_intensity = new_hawkes_mean.max(0.1);

                let hawkes_std_floor = params.mean_hawkes_intensity * 0.3;
                let new_hawkes_std = blend * self.emission_hawkes_sq_sum[k].sqrt()
                    + (1.0 - blend) * params.std_hawkes_intensity;
                params.std_hawkes_intensity = new_hawkes_std.max(hawkes_std_floor).max(0.05);
            }
        }

        // Enforce minimum regime separation after EM update
        let normal_vol = self.emission_params[regime_idx::NORMAL].mean_volatility;
        if self.emission_params[regime_idx::HIGH].mean_volatility < normal_vol * 1.5 {
            self.emission_params[regime_idx::HIGH].mean_volatility = normal_vol * 1.5;
        }
        let high_vol = self.emission_params[regime_idx::HIGH].mean_volatility;
        if self.emission_params[regime_idx::EXTREME].mean_volatility < high_vol * 2.0 {
            self.emission_params[regime_idx::EXTREME].mean_volatility = high_vol * 2.0;
        }
    }

    /// Compute emission likelihood P(observation | regime).
    pub fn emission_likelihood(&self, regime: usize, obs: &Observation) -> f64 {
        if regime >= NUM_REGIMES {
            return 0.0;
        }
        self.emission_params[regime].likelihood(obs)
    }

    /// Update transition and emission parameters from observed regime history.
    ///
    /// This is a simplified M-step that updates:
    /// 1. Transition probabilities using Dirichlet-multinomial conjugacy
    /// 2. Emission parameters using exponential moving average
    ///
    /// # Arguments
    /// - `regime_history`: Sequence of (most_likely_regime, observation) pairs
    pub fn update_parameters(&mut self, regime_history: &[(usize, Observation)]) {
        if regime_history.len() < 2 {
            return;
        }

        // Update transition counts from consecutive regime pairs
        for window in regime_history.windows(2) {
            let from_regime = window[0].0;
            let to_regime = window[1].0;
            if from_regime < NUM_REGIMES && to_regime < NUM_REGIMES {
                self.transition_counts[from_regime][to_regime] += 1.0;
            }
        }

        // Update transition matrix using Dirichlet posterior mean
        for i in 0..NUM_REGIMES {
            let row_sum: f64 = (0..NUM_REGIMES)
                .map(|j| self.prior_counts[i][j] + self.transition_counts[i][j])
                .sum();

            if row_sum > 1e-9 {
                for j in 0..NUM_REGIMES {
                    self.transition_matrix[i][j] =
                        (self.prior_counts[i][j] + self.transition_counts[i][j]) / row_sum;
                }
            }
        }

        // Update emission parameters using EMA
        let alpha = self.emission_learning_rate;
        for (regime, obs) in regime_history {
            if *regime < NUM_REGIMES {
                let params = &mut self.emission_params[*regime];

                // Update means with EMA
                params.mean_volatility =
                    (1.0 - alpha) * params.mean_volatility + alpha * obs.volatility;
                params.mean_spread = (1.0 - alpha) * params.mean_spread + alpha * obs.spread_bps;

                // Update standard deviations (using squared deviation EMA)
                let vol_dev = (obs.volatility - params.mean_volatility).abs();
                params.std_volatility = ((1.0 - alpha) * params.std_volatility.powi(2)
                    + alpha * vol_dev.powi(2))
                .sqrt()
                .max(1e-9);

                let spread_dev = (obs.spread_bps - params.mean_spread).abs();
                params.std_spread = ((1.0 - alpha) * params.std_spread.powi(2)
                    + alpha * spread_dev.powi(2))
                .sqrt()
                .max(0.1);
            }
        }
    }

    /// Get the most likely regime based on current belief state.
    pub fn most_likely_regime(&self) -> usize {
        self.belief
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(regime_idx::NORMAL)
    }

    /// Get current regime probabilities.
    pub fn regime_probabilities(&self) -> [f64; NUM_REGIMES] {
        self.belief
    }

    /// Convert HMM belief state to the existing RegimeBeliefState type.
    ///
    /// This enables integration with `RegimeParameterBlender` for soft parameter blending:
    /// ```ignore
    /// let belief = hmm.to_belief_state();
    /// let blended = blender.blend_all(&belief);
    /// // Use blended.gamma, blended.kappa in GLFT calculation
    /// ```
    ///
    /// # Future Integration
    ///
    /// This method is the integration point for connecting HMM regime detection
    /// to the quote engine's parameter blending. Full wiring will create a
    /// `RegimeAwareQuoteEngine` that:
    /// 1. Holds `RegimeHMM` and `RegimeParameterBlender`
    /// 2. On each update, calls `hmm.to_belief_state()` → `blender.blend_all()`
    /// 3. Uses blended gamma/kappa in GLFT calculation
    pub fn to_belief_state(&self) -> RegimeBeliefState {
        RegimeBeliefState {
            p_low: self.belief[regime_idx::LOW],
            p_normal: self.belief[regime_idx::NORMAL],
            p_high: self.belief[regime_idx::HIGH],
            p_extreme: self.belief[regime_idx::EXTREME],
        }
    }

    /// Get the current transition matrix.
    pub fn transition_matrix(&self) -> &[[f64; NUM_REGIMES]; NUM_REGIMES] {
        &self.transition_matrix
    }

    /// Get the current emission parameters.
    pub fn emission_params(&self) -> &[EmissionParams; NUM_REGIMES] {
        &self.emission_params
    }

    /// Get the number of observations processed.
    pub fn observation_count(&self) -> u64 {
        self.observation_count
    }

    /// Get effective sample counts per regime from soft EM.
    /// Useful for diagnostics — shows which regimes have enough data.
    pub fn emission_effective_n(&self) -> &[f64; NUM_REGIMES] {
        &self.emission_effective_n
    }

    /// Authority ramp: regime has no power early in session.
    /// At 200 obs (old threshold), authority = 0.067 -- regime only reduces by ~5%.
    /// Full authority at 3000 obs.
    pub fn authority(&self) -> f64 {
        (self.observation_count as f64 / 3000.0).min(1.0)
    }

    /// Damped max position fraction -- prevents regime from destroying quoting capacity early.
    /// Returns 1.0 when authority is 0 (full position), gradually applies regime restriction.
    pub fn damped_max_position_fraction(&self, raw_fraction: f64) -> f64 {
        let auth = self.authority();
        1.0 - auth * (1.0 - raw_fraction)
    }

    /// Get entropy of current belief (measure of uncertainty).
    pub fn belief_entropy(&self) -> f64 {
        -self
            .belief
            .iter()
            .filter(|&&p| p > 1e-9)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }

    /// Check if belief is confident (low entropy).
    ///
    /// Threshold derived from information theory: `ln(K)/2` where K = num_regimes.
    /// For K=4: `ln(4)/2 ≈ 0.693`. Below this, one regime dominates.
    /// Max entropy is `ln(4) ≈ 1.386` (uniform).
    pub fn is_confident(&self) -> bool {
        self.belief_entropy() < (NUM_REGIMES as f64).ln() / 2.0
    }

    /// Reset belief to default (Normal-dominated).
    pub fn reset_belief(&mut self) {
        self.belief = [0.1, 0.6, 0.25, 0.05];
    }

    /// Reset all learned parameters to defaults.
    pub fn reset(&mut self) {
        self.transition_matrix = default_transition_matrix();
        self.emission_params = default_emission_params();
        self.belief = [0.1, 0.6, 0.25, 0.05];
        self.prior_counts = default_prior_counts();
        self.transition_counts = [[0.0; NUM_REGIMES]; NUM_REGIMES];
        self.observation_count = 0;
        self.vol_buffer = VecDeque::with_capacity(self.recalibration_window);
        self.spread_buffer = VecDeque::with_capacity(self.recalibration_window);
        self.initial_calibration_done = false;
        self.observations_since_recalibration = 0;
        self.recalibration_count = 0;
        self.prev_belief = [0.1, 0.6, 0.25, 0.05];
        self.emission_vol_sum = [0.0; NUM_REGIMES];
        self.emission_spread_sum = [0.0; NUM_REGIMES];
        self.emission_vol_sq_sum = [0.0; NUM_REGIMES];
        self.emission_spread_sq_sum = [0.0; NUM_REGIMES];
        self.emission_hawkes_sum = [0.0; NUM_REGIMES];
        self.emission_hawkes_sq_sum = [0.0; NUM_REGIMES];
        self.emission_effective_n = [0.0; NUM_REGIMES];
    }

    /// Check if auto-calibration has been performed.
    pub fn is_calibrated(&self) -> bool {
        self.initial_calibration_done
    }

    /// Calibrate emission parameters from observed data.
    ///
    /// This method uses empirical quantiles to set regime thresholds,
    /// which fixes the problem where HMM never detects certain regimes
    /// due to miscalibrated default thresholds.
    ///
    /// # Algorithm
    ///
    /// 1. Collect volatility and spread observations
    /// 2. Compute percentiles (10th, 50th, 75th, 95th)
    /// 3. Set regime emission means at these percentiles:
    ///    - Low: 10th percentile (very quiet)
    ///    - Normal: 50th percentile (median)
    ///    - High: 75th percentile (elevated)
    ///    - Extreme: 95th percentile (crisis)
    ///
    /// # Arguments
    /// * `volatilities` - Observed volatility values
    /// * `spreads` - Observed spread values (in bps)
    pub fn calibrate_from_observations(&mut self, volatilities: &[f64], spreads: &[f64]) {
        if volatilities.len() < 100 || spreads.len() < 100 {
            return; // Not enough data
        }

        // Sort for percentile computation
        let mut vol_sorted: Vec<f64> = volatilities.to_vec();
        let mut spread_sorted: Vec<f64> = spreads.to_vec();
        vol_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        spread_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute percentiles
        let percentile = |sorted: &[f64], p: f64| -> f64 {
            let idx = ((sorted.len() as f64 - 1.0) * p) as usize;
            sorted[idx.min(sorted.len() - 1)]
        };

        let vol_p10 = percentile(&vol_sorted, 0.10);
        let vol_p50 = percentile(&vol_sorted, 0.50);
        let vol_p75 = percentile(&vol_sorted, 0.75);
        let vol_p95 = percentile(&vol_sorted, 0.95);

        let spread_p10 = percentile(&spread_sorted, 0.10);
        let spread_p50 = percentile(&spread_sorted, 0.50);
        let spread_p75 = percentile(&spread_sorted, 0.75);
        let spread_p95 = percentile(&spread_sorted, 0.95);

        // Compute IQR for std estimation
        let vol_iqr = (vol_p75 - percentile(&vol_sorted, 0.25)).max(vol_p50 * 0.1);
        let spread_iqr = (spread_p75 - percentile(&spread_sorted, 0.25)).max(spread_p50 * 0.1);

        // === Minimum variance floors ===
        // Prevent threshold compression during quiet periods: std must be at least
        // 30% of the mean, so regimes always have meaningful overlap and don't
        // collapse into degenerate point distributions.
        let min_vol_std = vol_p50 * 0.3;
        let min_spread_std = spread_p50 * 0.3;

        // === Minimum regime separation ===
        // Ensure HIGH mean >= 1.5x NORMAL, EXTREME >= 2.0x HIGH
        // Prevents quiet-period calibrations from making normal activity look extreme.
        let high_vol = vol_p75.max(vol_p50 * 1.5);
        let high_spread = spread_p75.max(spread_p50 * 1.5);
        let extreme_vol = vol_p95.max(high_vol * 2.0);
        let extreme_spread = spread_p95.max(high_spread * 2.0);

        // Update emission parameters
        // Low regime: 10th percentile
        self.emission_params[regime_idx::LOW] = EmissionParams::new(
            vol_p10,
            (vol_iqr * 0.3).max(min_vol_std),
            spread_p10,
            (spread_iqr * 0.3).max(min_spread_std),
        );

        // Normal regime: 50th percentile (median)
        self.emission_params[regime_idx::NORMAL] = EmissionParams::new(
            vol_p50,
            (vol_iqr * 0.5).max(min_vol_std),
            spread_p50,
            (spread_iqr * 0.5).max(min_spread_std),
        );

        // High regime: 75th percentile (with minimum separation)
        self.emission_params[regime_idx::HIGH] = EmissionParams::new(
            high_vol,
            (vol_iqr * 0.7).max(min_vol_std),
            high_spread,
            (spread_iqr * 0.7).max(min_spread_std),
        );

        // Extreme regime: 95th percentile (with minimum separation)
        self.emission_params[regime_idx::EXTREME] = EmissionParams::new(
            extreme_vol,
            (vol_iqr * 1.0).max(min_vol_std),
            extreme_spread,
            (spread_iqr * 1.0).max(min_spread_std),
        );

        // Re-seed soft EM sums after recalibration
        self.seed_em_sums_from_emissions();
    }

    /// Get calibration statistics for diagnostics.
    pub fn calibration_stats(&self) -> RegimeCalibrationStats {
        RegimeCalibrationStats {
            observation_count: self.observation_count,
            regime_probs: self.belief,
            emission_means: [
                (
                    self.emission_params[0].mean_volatility,
                    self.emission_params[0].mean_spread,
                ),
                (
                    self.emission_params[1].mean_volatility,
                    self.emission_params[1].mean_spread,
                ),
                (
                    self.emission_params[2].mean_volatility,
                    self.emission_params[2].mean_spread,
                ),
                (
                    self.emission_params[3].mean_volatility,
                    self.emission_params[3].mean_spread,
                ),
            ],
        }
    }

    /// Normalize belief state to sum to 1.
    fn normalize_belief(&mut self) {
        let sum: f64 = self.belief.iter().sum();
        if sum > 1e-9 {
            for p in &mut self.belief {
                *p /= sum;
            }
        } else {
            // Reset to default if all probabilities are zero
            self.belief = [0.1, 0.6, 0.25, 0.05];
        }
    }

    // === Checkpoint persistence ===

    /// Extract learning state for checkpoint persistence.
    ///
    /// VecDeque buffers (vol_buffer, spread_buffer) are NOT persisted —
    /// they're observation windows that refill from live data.
    pub fn to_checkpoint(&self) -> crate::market_maker::checkpoint::RegimeHMMCheckpoint {
        crate::market_maker::checkpoint::RegimeHMMCheckpoint {
            belief: self.belief,
            transition_counts: self.transition_counts,
            observation_count: self.observation_count,
            recalibration_count: self.recalibration_count,
            emission_effective_n: self.emission_effective_n,
            emission_vol_sum: self.emission_vol_sum,
            emission_spread_sum: self.emission_spread_sum,
            emission_hawkes_sum: self.emission_hawkes_sum,
        }
    }

    /// Restore learning state from a checkpoint.
    pub fn restore_checkpoint(
        &mut self,
        cp: &crate::market_maker::checkpoint::RegimeHMMCheckpoint,
    ) {
        self.belief = cp.belief;
        self.transition_counts = cp.transition_counts;
        self.observation_count = cp.observation_count;
        self.recalibration_count = cp.recalibration_count;
        self.emission_effective_n = cp.emission_effective_n;
        self.emission_vol_sum = cp.emission_vol_sum;
        self.emission_spread_sum = cp.emission_spread_sum;
        self.emission_hawkes_sum = cp.emission_hawkes_sum;
        self.prev_belief = self.belief;
    }
}

/// Calibration statistics for RegimeHMM diagnostics.
#[derive(Debug, Clone)]
pub struct RegimeCalibrationStats {
    pub observation_count: u64,
    pub regime_probs: [f64; NUM_REGIMES],
    /// (mean_vol, mean_spread) for each regime
    pub emission_means: [(f64, f64); NUM_REGIMES],
}

impl RegimeCalibrationStats {
    pub fn summary(&self) -> String {
        format!(
            "n={} probs=[{:.2},{:.2},{:.2},{:.2}] thresholds=[{:.4}/{:.1}, {:.4}/{:.1}, {:.4}/{:.1}, {:.4}/{:.1}]",
            self.observation_count,
            self.regime_probs[0],
            self.regime_probs[1],
            self.regime_probs[2],
            self.regime_probs[3],
            self.emission_means[0].0, self.emission_means[0].1,
            self.emission_means[1].0, self.emission_means[1].1,
            self.emission_means[2].0, self.emission_means[2].1,
            self.emission_means[3].0, self.emission_means[3].1,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create observation with just volatility.
    fn obs_vol(vol: f64) -> Observation {
        Observation::new(vol, 5.0, 0.0)
    }

    /// Helper to create observation with volatility and spread.
    fn obs_vol_spread(vol: f64, spread: f64) -> Observation {
        Observation::new(vol, spread, 0.0)
    }

    // =========================================================================
    // Test 1: Forward Update with Normal Observations
    // =========================================================================
    #[test]
    fn test_forward_update_normal_observations() {
        let mut hmm = RegimeHMM::new();
        let initial_normal_prob = hmm.belief[regime_idx::NORMAL];

        // Feed observations consistent with Normal regime
        for _ in 0..20 {
            hmm.forward_update(&obs_vol_spread(0.0025, 5.0));
        }

        // Normal probability should increase or stay high
        assert!(
            hmm.belief[regime_idx::NORMAL] >= initial_normal_prob * 0.8,
            "Normal prob {} should stay high after normal observations (was {})",
            hmm.belief[regime_idx::NORMAL],
            initial_normal_prob
        );

        // Should be most likely regime
        assert_eq!(
            hmm.most_likely_regime(),
            regime_idx::NORMAL,
            "Normal should be most likely after normal observations"
        );
    }

    // =========================================================================
    // Test 2: Regime Transitions When Volatility Spikes
    // =========================================================================
    #[test]
    fn test_regime_transition_volatility_spike() {
        let mut hmm = RegimeHMM::new();

        // Start with normal observations to establish baseline
        for _ in 0..10 {
            hmm.forward_update(&obs_vol_spread(0.0025, 5.0));
        }
        let pre_spike_high = hmm.belief[regime_idx::HIGH];
        let pre_spike_extreme = hmm.belief[regime_idx::EXTREME];

        // Spike volatility to extreme levels
        for _ in 0..10 {
            hmm.forward_update(&obs_vol_spread(0.05, 25.0));
        }

        // High or Extreme probability should increase significantly
        let combined_elevated = hmm.belief[regime_idx::HIGH] + hmm.belief[regime_idx::EXTREME];
        let pre_combined = pre_spike_high + pre_spike_extreme;

        assert!(
            combined_elevated > pre_combined,
            "High+Extreme prob {} should exceed pre-spike {} after vol spike",
            combined_elevated,
            pre_combined
        );
    }

    // =========================================================================
    // Test 3: Low Regime Detection
    // =========================================================================
    #[test]
    fn test_low_regime_detection() {
        let mut hmm = RegimeHMM::new();

        // Feed very low volatility observations
        for _ in 0..30 {
            hmm.forward_update(&obs_vol_spread(0.0005, 2.0));
        }

        // Low regime probability should be elevated
        assert!(
            hmm.belief[regime_idx::LOW] > 0.1,
            "Low prob {} should be elevated after quiet observations",
            hmm.belief[regime_idx::LOW]
        );

        // Should be higher than Extreme
        assert!(
            hmm.belief[regime_idx::LOW] > hmm.belief[regime_idx::EXTREME],
            "Low prob {} should exceed Extreme prob {} for quiet market",
            hmm.belief[regime_idx::LOW],
            hmm.belief[regime_idx::EXTREME]
        );
    }

    // =========================================================================
    // Test 4: Extreme Regime from High Spread
    // =========================================================================
    #[test]
    fn test_extreme_regime_high_spread() {
        let mut hmm = RegimeHMM::new();

        // Feed extreme observations
        for _ in 0..20 {
            hmm.forward_update(&obs_vol_spread(0.04, 30.0));
        }

        // Extreme regime should dominate
        assert!(
            hmm.belief[regime_idx::EXTREME] > hmm.belief[regime_idx::LOW],
            "Extreme {} should exceed Low {} for crisis conditions",
            hmm.belief[regime_idx::EXTREME],
            hmm.belief[regime_idx::LOW]
        );
    }

    // =========================================================================
    // Test 5: Parameter Learning - Transition Matrix
    // =========================================================================
    #[test]
    fn test_parameter_learning_transitions() {
        let mut hmm = RegimeHMM::new();

        // Create regime history with many Normal -> Normal transitions
        let history: Vec<(usize, Observation)> = (0..50)
            .map(|_| (regime_idx::NORMAL, obs_vol_spread(0.0025, 5.0)))
            .collect();

        let pre_update = hmm.transition_matrix[regime_idx::NORMAL][regime_idx::NORMAL];

        hmm.update_parameters(&history);

        let post_update = hmm.transition_matrix[regime_idx::NORMAL][regime_idx::NORMAL];

        // Normal->Normal transition should stay high (already was ~0.95)
        assert!(
            post_update >= pre_update * 0.95,
            "Normal->Normal transition {} should stay high (was {})",
            post_update,
            pre_update
        );
    }

    // =========================================================================
    // Test 6: Parameter Learning - Emission Parameters
    // =========================================================================
    #[test]
    fn test_parameter_learning_emissions() {
        let mut hmm = RegimeHMM::new();

        // Create history with Normal regime but shifted volatility
        let shifted_vol = 0.003; // Slightly higher than default 0.0025
        let history: Vec<(usize, Observation)> = (0..100)
            .map(|_| (regime_idx::NORMAL, obs_vol_spread(shifted_vol, 6.0)))
            .collect();

        let pre_mean = hmm.emission_params[regime_idx::NORMAL].mean_volatility;

        hmm.update_parameters(&history);

        let post_mean = hmm.emission_params[regime_idx::NORMAL].mean_volatility;

        // Mean volatility should move toward observed values
        assert!(
            (post_mean - shifted_vol).abs() < (pre_mean - shifted_vol).abs(),
            "Mean vol {} should move toward observed {} (was {})",
            post_mean,
            shifted_vol,
            pre_mean
        );
    }

    // =========================================================================
    // Test 7: Conversion to RegimeBeliefState
    // =========================================================================
    #[test]
    fn test_to_belief_state() {
        let mut hmm = RegimeHMM::new();

        // Set specific belief for testing
        hmm.belief = [0.1, 0.5, 0.3, 0.1];
        hmm.normalize_belief();

        let belief_state = hmm.to_belief_state();

        assert!(
            (belief_state.p_low - hmm.belief[regime_idx::LOW]).abs() < 1e-9,
            "p_low mismatch"
        );
        assert!(
            (belief_state.p_normal - hmm.belief[regime_idx::NORMAL]).abs() < 1e-9,
            "p_normal mismatch"
        );
        assert!(
            (belief_state.p_high - hmm.belief[regime_idx::HIGH]).abs() < 1e-9,
            "p_high mismatch"
        );
        assert!(
            (belief_state.p_extreme - hmm.belief[regime_idx::EXTREME]).abs() < 1e-9,
            "p_extreme mismatch"
        );

        // Verify sum to 1
        let sum = belief_state.p_low
            + belief_state.p_normal
            + belief_state.p_high
            + belief_state.p_extreme;
        assert!((sum - 1.0).abs() < 1e-9, "Belief state should sum to 1");
    }

    // =========================================================================
    // Test 8: Belief Normalization and Edge Cases
    // =========================================================================
    #[test]
    fn test_belief_normalization() {
        let mut hmm = RegimeHMM::new();

        // Set unnormalized belief
        hmm.belief = [0.2, 0.4, 0.2, 0.2];

        let sum: f64 = hmm.belief.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Initial belief should sum to 1");

        // After forward update, should still sum to 1
        hmm.forward_update(&obs_vol(0.001));
        let sum_after: f64 = hmm.belief.iter().sum();
        assert!(
            (sum_after - 1.0).abs() < 1e-9,
            "Belief should sum to 1 after update, got {}",
            sum_after
        );
    }

    // =========================================================================
    // Test 9: Entropy and Confidence
    // =========================================================================
    #[test]
    fn test_entropy_and_confidence() {
        let mut hmm = RegimeHMM::new();

        // High entropy (uncertain) - uniform-ish belief
        hmm.belief = [0.25, 0.25, 0.25, 0.25];
        let high_entropy = hmm.belief_entropy();
        assert!(
            !hmm.is_confident(),
            "Uniform belief should not be confident"
        );

        // Low entropy (confident) - concentrated belief
        hmm.belief = [0.01, 0.95, 0.03, 0.01];
        let low_entropy = hmm.belief_entropy();
        assert!(
            hmm.is_confident(),
            "Concentrated belief should be confident"
        );

        assert!(
            high_entropy > low_entropy,
            "Uniform entropy {} should exceed concentrated {}",
            high_entropy,
            low_entropy
        );
    }

    // =========================================================================
    // Test 10: Sticky Transitions
    // =========================================================================
    #[test]
    fn test_sticky_transitions() {
        let hmm = RegimeHMM::new();

        // Verify diagonal dominance (sticky behavior)
        for i in 0..NUM_REGIMES {
            let diag = hmm.transition_matrix[i][i];
            assert!(
                diag > 0.9,
                "Diagonal element [{},{}] = {} should be > 0.9 (sticky)",
                i,
                i,
                diag
            );

            // Verify row sums to 1
            let row_sum: f64 = hmm.transition_matrix[i].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-9,
                "Transition row {} sum {} should be 1",
                i,
                row_sum
            );
        }
    }

    // =========================================================================
    // Test 11: Emission Likelihood Correctness
    // =========================================================================
    #[test]
    fn test_emission_likelihood() {
        let hmm = RegimeHMM::new();

        // Normal observation should have highest likelihood under Normal regime
        let normal_obs = obs_vol_spread(0.0025, 5.0);
        let ll_low = hmm.emission_likelihood(regime_idx::LOW, &normal_obs);
        let ll_normal = hmm.emission_likelihood(regime_idx::NORMAL, &normal_obs);
        let ll_high = hmm.emission_likelihood(regime_idx::HIGH, &normal_obs);
        let ll_extreme = hmm.emission_likelihood(regime_idx::EXTREME, &normal_obs);

        assert!(
            ll_normal >= ll_extreme,
            "Normal obs likelihood under Normal {} should exceed Extreme {}",
            ll_normal,
            ll_extreme
        );

        // Extreme observation should have highest likelihood under Extreme regime
        let extreme_obs = obs_vol_spread(0.05, 25.0);
        let ll_normal_ext = hmm.emission_likelihood(regime_idx::NORMAL, &extreme_obs);
        let ll_extreme_ext = hmm.emission_likelihood(regime_idx::EXTREME, &extreme_obs);

        assert!(
            ll_extreme_ext > ll_normal_ext,
            "Extreme obs likelihood under Extreme {} should exceed Normal {}",
            ll_extreme_ext,
            ll_normal_ext
        );

        // All likelihoods should be positive
        assert!(ll_low > 0.0 && ll_normal > 0.0 && ll_high > 0.0 && ll_extreme > 0.0);
    }

    // =========================================================================
    // Test 12: Most Likely Regime
    // =========================================================================
    #[test]
    fn test_most_likely_regime() {
        let mut hmm = RegimeHMM::new();

        // Set each regime as most likely and verify
        for expected in 0..NUM_REGIMES {
            hmm.belief = [0.1, 0.1, 0.1, 0.1];
            hmm.belief[expected] = 0.7;
            hmm.normalize_belief();

            assert_eq!(
                hmm.most_likely_regime(),
                expected,
                "Expected regime {} to be most likely with belief {:?}",
                expected,
                hmm.belief
            );
        }
    }

    // =========================================================================
    // Test 13: Reset Functionality
    // =========================================================================
    #[test]
    fn test_reset() {
        let mut hmm = RegimeHMM::new();

        // Modify state
        hmm.belief = [0.9, 0.05, 0.03, 0.02];
        hmm.transition_counts[0][1] = 100.0;
        hmm.observation_count = 1000;

        hmm.reset();

        // Verify reset
        assert_eq!(hmm.observation_count, 0);
        assert!((hmm.transition_counts[0][1] - 0.0).abs() < 1e-9);
        assert!((hmm.belief[regime_idx::NORMAL] - 0.6).abs() < 0.01);
    }

    // =========================================================================
    // Test 14: Flow Imbalance Clamping
    // =========================================================================
    #[test]
    fn test_flow_imbalance_clamping() {
        // Test that flow imbalance is properly clamped
        let obs1 = Observation::new(0.001, 5.0, 2.0);
        assert!(
            (obs1.flow_imbalance - 1.0).abs() < 1e-9,
            "Should clamp to 1.0"
        );

        let obs2 = Observation::new(0.001, 5.0, -2.0);
        assert!(
            (obs2.flow_imbalance - (-1.0)).abs() < 1e-9,
            "Should clamp to -1.0"
        );

        let obs3 = Observation::new(0.001, 5.0, 0.5);
        assert!(
            (obs3.flow_imbalance - 0.5).abs() < 1e-9,
            "Should preserve valid values"
        );
    }

    // =========================================================================
    // Test 15: Builder Pattern
    // =========================================================================
    #[test]
    fn test_builder_pattern() {
        let custom_belief = [0.25, 0.25, 0.25, 0.25];
        let hmm = RegimeHMM::new()
            .with_initial_belief(custom_belief)
            .with_emission_learning_rate(0.05);

        // Verify custom belief was set (after normalization)
        let sum: f64 = hmm.belief.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        // Each should be ~0.25
        for p in hmm.belief {
            assert!((p - 0.25).abs() < 0.01);
        }

        assert!((hmm.emission_learning_rate - 0.05).abs() < 1e-9);
    }

    // === seed_emissions_from_profile tests ===

    #[test]
    fn test_seed_from_profile_hype_conditions() {
        // HYPE: ~0.001 vol, ~15 bps half-spread
        let mut hmm = RegimeHMM::default();
        hmm.seed_emissions_from_profile(0.001, 15.0);

        let normal = &hmm.emission_params()[regime_idx::NORMAL];
        assert!(
            (normal.mean_volatility - 0.001).abs() < 1e-8,
            "Normal vol should be 0.001, got {}",
            normal.mean_volatility
        );
        assert!(
            (normal.mean_spread - 15.0).abs() < 0.01,
            "Normal spread should be 15 bps, got {}",
            normal.mean_spread
        );
    }

    #[test]
    fn test_seed_from_profile_btc_conditions() {
        // BTC: ~0.00025 vol, ~1.5 bps half-spread
        let mut hmm = RegimeHMM::default();
        hmm.seed_emissions_from_profile(0.00025, 1.5);

        let normal = &hmm.emission_params()[regime_idx::NORMAL];
        assert!(
            (normal.mean_volatility - 0.00025).abs() < 1e-9,
            "Normal vol should be 0.00025, got {}",
            normal.mean_volatility
        );
        assert!(
            (normal.mean_spread - 1.5).abs() < 0.01,
            "Normal spread should be 1.5 bps, got {}",
            normal.mean_spread
        );
    }

    #[test]
    fn test_seed_from_profile_regime_separation() {
        let mut hmm = RegimeHMM::default();
        hmm.seed_emissions_from_profile(0.0005, 10.0);

        let normal_vol = hmm.emission_params()[regime_idx::NORMAL].mean_volatility;
        let high_vol = hmm.emission_params()[regime_idx::HIGH].mean_volatility;
        let extreme_vol = hmm.emission_params()[regime_idx::EXTREME].mean_volatility;

        // HIGH >= 2x NORMAL (enforced by safety check)
        assert!(
            high_vol >= normal_vol * 2.0 - 1e-12,
            "HIGH vol ({}) should be >= 2x NORMAL vol ({})",
            high_vol,
            normal_vol
        );
        // EXTREME > HIGH
        assert!(
            extreme_vol > high_vol,
            "EXTREME vol ({}) should be > HIGH vol ({})",
            extreme_vol,
            high_vol
        );
        // LOW < NORMAL
        let low_vol = hmm.emission_params()[regime_idx::LOW].mean_volatility;
        assert!(
            low_vol < normal_vol,
            "LOW vol ({}) should be < NORMAL vol ({})",
            low_vol,
            normal_vol
        );
    }

    // =========================================================================
    // P5: Online Soft EM Tests
    // =========================================================================

    #[test]
    fn test_soft_em_accumulates_effective_n() {
        // Soft EM should accumulate effective sample counts per regime
        let mut hmm = RegimeHMM::new().with_baseline_volatility(0.001, 5.0);

        // Feed normal observations — Normal regime should accumulate most
        for _ in 0..100 {
            hmm.forward_update(&obs_vol_spread(0.001, 5.0));
        }

        let eff_n = hmm.emission_effective_n();
        let normal_n = eff_n[regime_idx::NORMAL];

        // Normal regime should have highest effective N
        assert!(
            normal_n > eff_n[regime_idx::LOW],
            "Normal effective_n ({:.1}) should exceed Low ({:.1})",
            normal_n,
            eff_n[regime_idx::LOW]
        );
        assert!(
            normal_n > eff_n[regime_idx::EXTREME],
            "Normal effective_n ({:.1}) should exceed Extreme ({:.1})",
            normal_n,
            eff_n[regime_idx::EXTREME]
        );
    }

    #[test]
    fn test_soft_em_prevents_emission_collapse() {
        // Even with only Normal observations, HIGH/EXTREME emission std
        // should not collapse to zero (floor at 30% of mean)
        let mut hmm = RegimeHMM::new().with_baseline_volatility(0.001, 5.0);

        for _ in 0..200 {
            hmm.forward_update(&obs_vol_spread(0.001, 5.0));
        }

        for k in 0..NUM_REGIMES {
            let params = &hmm.emission_params()[k];
            let vol_floor = params.mean_volatility * 0.3;
            assert!(
                params.std_volatility >= vol_floor * 0.99, // 1% tolerance
                "Regime {} vol std ({:.6}) should be >= floor ({:.6})",
                k,
                params.std_volatility,
                vol_floor
            );
        }
    }

    #[test]
    fn test_soft_em_transition_learning() {
        // Soft transition counts should update the transition matrix
        let mut hmm = RegimeHMM::new().with_baseline_volatility(0.001, 5.0);

        let pre_nn = hmm.transition_matrix()[regime_idx::NORMAL][regime_idx::NORMAL];

        // Feed 200 normal observations — should reinforce Normal→Normal
        for _ in 0..200 {
            hmm.forward_update(&obs_vol_spread(0.001, 5.0));
        }

        let post_nn = hmm.transition_matrix()[regime_idx::NORMAL][regime_idx::NORMAL];

        // Normal→Normal should stay high or increase
        assert!(
            post_nn >= pre_nn * 0.95,
            "Normal→Normal ({:.4}) should stay high (was {:.4})",
            post_nn,
            pre_nn
        );
    }

    #[test]
    fn test_soft_em_regime_separation_maintained() {
        // After soft EM updates, regime separation should be maintained
        let mut hmm = RegimeHMM::new().with_baseline_volatility(0.001, 5.0);

        // Feed mixed observations to trigger EM updates
        for _ in 0..100 {
            hmm.forward_update(&obs_vol_spread(0.001, 5.0));
        }
        for _ in 0..50 {
            hmm.forward_update(&obs_vol_spread(0.005, 15.0));
        }

        let normal_vol = hmm.emission_params()[regime_idx::NORMAL].mean_volatility;
        let high_vol = hmm.emission_params()[regime_idx::HIGH].mean_volatility;
        let extreme_vol = hmm.emission_params()[regime_idx::EXTREME].mean_volatility;

        // HIGH >= 1.5x NORMAL (enforced by apply_soft_em_emissions)
        assert!(
            high_vol >= normal_vol * 1.49, // slight tolerance
            "HIGH vol ({:.6}) should be >= 1.5x NORMAL vol ({:.6})",
            high_vol,
            normal_vol
        );
        // EXTREME >= 2x HIGH
        assert!(
            extreme_vol >= high_vol * 1.99, // slight tolerance
            "EXTREME vol ({:.6}) should be >= 2x HIGH vol ({:.6})",
            extreme_vol,
            high_vol
        );
    }

    #[test]
    fn test_entropy_threshold_information_theoretic() {
        // is_confident() threshold should be ln(K)/2 = ln(4)/2 ≈ 0.693
        let mut hmm = RegimeHMM::new();

        // Uniform belief: entropy = ln(4) ≈ 1.386 > 0.693 → not confident
        hmm.belief = [0.25, 0.25, 0.25, 0.25];
        assert!(
            !hmm.is_confident(),
            "Uniform belief should not be confident"
        );

        // Strong belief in one regime: entropy ≈ 0.24 < 0.693 → confident
        hmm.belief = [0.01, 0.95, 0.03, 0.01];
        assert!(hmm.is_confident(), "Strong belief should be confident");

        // Moderate belief: entropy ≈ 0.8 > 0.693 → not confident
        hmm.belief = [0.15, 0.4, 0.3, 0.15];
        let _entropy = hmm.belief_entropy();
        let threshold = (NUM_REGIMES as f64).ln() / 2.0;
        // This should be borderline — just verify the threshold is ln(4)/2
        assert!(
            (threshold - 0.693).abs() < 0.01,
            "Threshold should be ln(4)/2 ≈ 0.693, got {:.3}",
            threshold
        );
    }

    #[test]
    fn test_checkpoint_roundtrip_with_soft_em() {
        let mut hmm = RegimeHMM::new().with_baseline_volatility(0.001, 5.0);

        // Feed observations to accumulate soft EM state
        for _ in 0..100 {
            hmm.forward_update(&obs_vol_spread(0.001, 5.0));
        }

        let cp = hmm.to_checkpoint();

        // Restore into fresh HMM
        let mut hmm2 = RegimeHMM::new();
        hmm2.restore_checkpoint(&cp);

        // Verify soft EM state preserved
        for k in 0..NUM_REGIMES {
            assert!(
                (hmm.emission_effective_n[k] - hmm2.emission_effective_n[k]).abs() < 1e-9,
                "emission_effective_n[{}] mismatch: {} vs {}",
                k,
                hmm.emission_effective_n[k],
                hmm2.emission_effective_n[k]
            );
        }
        assert_eq!(hmm.observation_count, hmm2.observation_count);
    }

    // =========================================================================
    // Hawkes Intensity Integration Tests (Upgrade 2C)
    // =========================================================================

    #[test]
    fn test_hawkes_observation_backward_compat() {
        // Observation::new() should set hawkes_intensity_ratio to 1.0 (baseline)
        let obs = Observation::new(0.001, 5.0, 0.0);
        assert!(
            (obs.hawkes_intensity_ratio - 1.0).abs() < 1e-9,
            "Default Hawkes intensity should be 1.0, got {}",
            obs.hawkes_intensity_ratio
        );

        // new_full() should also default to 1.0
        let obs_full = Observation::new_full(0.001, 5.0, 0.0, 1.0, 0.0, 0.0);
        assert!(
            (obs_full.hawkes_intensity_ratio - 1.0).abs() < 1e-9,
            "new_full() Hawkes intensity should be 1.0, got {}",
            obs_full.hawkes_intensity_ratio
        );

        // Default should be 1.0
        let obs_default = Observation::default();
        assert!(
            (obs_default.hawkes_intensity_ratio - 1.0).abs() < 1e-9,
            "Default Hawkes intensity should be 1.0"
        );
    }

    #[test]
    fn test_new_with_hawkes_clamping() {
        // Should clamp negative to 0.0
        let obs_neg = Observation::new_with_hawkes(0.001, 5.0, 0.0, -1.0);
        assert!(
            (obs_neg.hawkes_intensity_ratio - 0.0).abs() < 1e-9,
            "Negative Hawkes should clamp to 0.0, got {}",
            obs_neg.hawkes_intensity_ratio
        );

        // Should clamp above 50.0
        let obs_high = Observation::new_with_hawkes(0.001, 5.0, 0.0, 100.0);
        assert!(
            (obs_high.hawkes_intensity_ratio - 50.0).abs() < 1e-9,
            "Excessive Hawkes should clamp to 50.0, got {}",
            obs_high.hawkes_intensity_ratio
        );

        // Valid values should pass through
        let obs_valid = Observation::new_with_hawkes(0.001, 5.0, 0.0, 3.5);
        assert!(
            (obs_valid.hawkes_intensity_ratio - 3.5).abs() < 1e-9,
            "Valid Hawkes should pass through, got {}",
            obs_valid.hawkes_intensity_ratio
        );
    }

    #[test]
    fn test_hawkes_shifts_hmm_toward_high_extreme() {
        // Elevated Hawkes intensity (ratio=5.0) should shift HMM toward High/Extreme
        let mut hmm = RegimeHMM::new();

        // Establish normal baseline
        for _ in 0..10 {
            hmm.forward_update(&Observation::new(0.0025, 5.0, 0.0));
        }
        let pre_high = hmm.belief[regime_idx::HIGH];
        let pre_extreme = hmm.belief[regime_idx::EXTREME];

        // Feed observations with elevated Hawkes intensity (trade clustering)
        // Use vol/spread consistent with High regime to avoid emission conflict
        for _ in 0..15 {
            hmm.forward_update(&Observation::new_with_hawkes(0.008, 9.0, 0.3, 5.0));
        }

        let post_combined = hmm.belief[regime_idx::HIGH] + hmm.belief[regime_idx::EXTREME];
        let pre_combined = pre_high + pre_extreme;

        assert!(
            post_combined > pre_combined,
            "High+Extreme prob ({:.4}) should increase with elevated Hawkes intensity (was {:.4})",
            post_combined,
            pre_combined
        );
    }

    #[test]
    fn test_hawkes_baseline_keeps_normal() {
        // Baseline Hawkes intensity (ratio=1.0) should not disturb Normal regime
        let mut hmm = RegimeHMM::new();

        // Feed normal observations with baseline Hawkes
        for _ in 0..30 {
            hmm.forward_update(&Observation::new_with_hawkes(0.0025, 5.0, 0.0, 1.0));
        }

        // Normal should still dominate
        assert!(
            hmm.belief[regime_idx::NORMAL] > 0.3,
            "Normal prob ({:.4}) should remain elevated with baseline Hawkes",
            hmm.belief[regime_idx::NORMAL]
        );

        // Should be higher than Extreme
        assert!(
            hmm.belief[regime_idx::NORMAL] > hmm.belief[regime_idx::EXTREME],
            "Normal ({:.4}) should exceed Extreme ({:.4}) at baseline Hawkes",
            hmm.belief[regime_idx::NORMAL],
            hmm.belief[regime_idx::EXTREME]
        );
    }

    #[test]
    fn test_hawkes_low_intensity_supports_low_regime() {
        // Very low Hawkes intensity should support Low regime detection
        let mut hmm = RegimeHMM::new();

        // Feed quiet observations with low Hawkes intensity
        for _ in 0..30 {
            hmm.forward_update(&Observation::new_with_hawkes(0.0005, 2.0, 0.0, 0.3));
        }

        // Low regime should be elevated
        assert!(
            hmm.belief[regime_idx::LOW] > hmm.belief[regime_idx::EXTREME],
            "Low ({:.4}) should exceed Extreme ({:.4}) with quiet Hawkes",
            hmm.belief[regime_idx::LOW],
            hmm.belief[regime_idx::EXTREME]
        );
    }

    #[test]
    fn test_hawkes_emission_params_per_regime() {
        let hmm = RegimeHMM::new();
        let params = hmm.emission_params();

        // Low: mean=0.5, quiet market
        assert!(
            params[regime_idx::LOW].mean_hawkes_intensity < 1.0,
            "Low regime Hawkes mean ({}) should be < 1.0",
            params[regime_idx::LOW].mean_hawkes_intensity
        );

        // Normal: mean=1.0, baseline
        assert!(
            (params[regime_idx::NORMAL].mean_hawkes_intensity - 1.0).abs() < 0.01,
            "Normal regime Hawkes mean ({}) should be ~1.0",
            params[regime_idx::NORMAL].mean_hawkes_intensity
        );

        // High: mean=2.5, elevated clustering
        assert!(
            params[regime_idx::HIGH].mean_hawkes_intensity > 2.0,
            "High regime Hawkes mean ({}) should be > 2.0",
            params[regime_idx::HIGH].mean_hawkes_intensity
        );

        // Extreme: mean=5.0, intense clustering
        assert!(
            params[regime_idx::EXTREME].mean_hawkes_intensity > 4.0,
            "Extreme regime Hawkes mean ({}) should be > 4.0",
            params[regime_idx::EXTREME].mean_hawkes_intensity
        );

        // Monotonic ordering
        assert!(
            params[regime_idx::LOW].mean_hawkes_intensity
                < params[regime_idx::NORMAL].mean_hawkes_intensity
        );
        assert!(
            params[regime_idx::NORMAL].mean_hawkes_intensity
                < params[regime_idx::HIGH].mean_hawkes_intensity
        );
        assert!(
            params[regime_idx::HIGH].mean_hawkes_intensity
                < params[regime_idx::EXTREME].mean_hawkes_intensity
        );
    }

    #[test]
    fn test_hawkes_checkpoint_roundtrip() {
        let mut hmm = RegimeHMM::new().with_baseline_volatility(0.001, 5.0);

        // Feed observations with Hawkes data to populate EM sums
        for _ in 0..100 {
            hmm.forward_update(&Observation::new_with_hawkes(0.001, 5.0, 0.0, 2.0));
        }

        let cp = hmm.to_checkpoint();
        let mut hmm2 = RegimeHMM::new();
        hmm2.restore_checkpoint(&cp);

        for k in 0..NUM_REGIMES {
            assert!(
                (hmm.emission_hawkes_sum[k] - hmm2.emission_hawkes_sum[k]).abs() < 1e-9,
                "emission_hawkes_sum[{}] mismatch: {} vs {}",
                k,
                hmm.emission_hawkes_sum[k],
                hmm2.emission_hawkes_sum[k]
            );
        }
    }
}
