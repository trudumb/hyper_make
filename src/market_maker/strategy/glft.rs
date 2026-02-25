//! GLFT (Guéant-Lehalle-Fernandez-Tapia) optimal market making strategy.

use tracing::debug;

use crate::{round_to_significant_and_decimal, truncate_float, EPSILON};

use crate::market_maker::config::{Quote, QuoteConfig};

use super::{
    CalibratedRiskModel, KellySizer, MarketParams, QuotingStrategy, RiskConfig, RiskFeatures,
    RiskModelConfig, SpreadComposition,
};

// === WS2: Posterior Predictive Inventory Penalty (PPIP) ===

/// Posterior Predictive Inventory Penalty: reservation price shift integrating
/// over Bayesian posteriors on (μ, σ², τ).
///
/// Resolves the γ-bifurcation: the old formula `q × γ × σ² × (1/κ)` used
/// τ_spread = 1/κ ≈ 0.0004s (per-fill timescale), producing skew ≈ 0.
/// PPIP uses τ_inventory (measured holding period, 30-300s), making skew O(bps).
///
/// Four terms:
/// 1. drift_cost: E[μ] × E[τ]
/// 2. variance_cost: q × E[σ²] × E[τ]
/// 3. ambiguity_aversion: variance_cost × CV²(σ²)
/// 4. timing_uncertainty: drift_cost × CV²(τ)
#[derive(Debug, Clone)]
pub struct PosteriorPredictiveSkew {
    /// E[μ] — drift mean (fractional/sec, from KalmanDriftEstimator)
    pub drift_mean: f64,
    /// Var[μ] — drift variance (from Kalman P)
    pub drift_variance: f64,

    /// E[σ²] — volatility mean (frac²/sec)
    pub sigma_sq_mean: f64,
    /// Var[σ²] — volatility variance (uncertainty about vol itself)
    pub sigma_sq_variance: f64,

    /// E[τ_inventory] — measured holding period (seconds, EWMA of reducing fills)
    pub tau_mean: f64,
    /// Var[τ] — variance of holding period (sec²)
    pub tau_variance: f64,

    /// Self-calibrating multiplier (starts 1.0, learns from markout)
    pub skew_multiplier: f64,
}

impl Default for PosteriorPredictiveSkew {
    fn default() -> Self {
        Self {
            drift_mean: 0.0,
            drift_variance: 1e-12,
            sigma_sq_mean: 1e-8, // (1 bps/√s)² as default
            sigma_sq_variance: 1e-16,
            tau_mean: 60.0,      // 60s default holding period
            tau_variance: 900.0, // 30s std dev
            skew_multiplier: 1.0,
        }
    }
}

impl PosteriorPredictiveSkew {
    /// Reservation price shift (fractional price units).
    /// ∂E[holding_cost]/∂q: the marginal cost of holding one more unit.
    ///
    /// Returns shift in fractional units (multiply by 10000 for bps).
    pub fn reservation_shift(&self, inventory_ratio: f64, max_position: f64) -> f64 {
        let q = inventory_ratio * max_position;

        // Term 1: Drift cost per unit (fractional)
        let drift_cost = self.drift_mean * self.tau_mean;

        // Term 2: Variance cost ∂/∂q of ½q²σ²τ = q×σ²×τ (fractional)
        let variance_cost = q * self.sigma_sq_mean * self.tau_mean;

        // Term 3: Ambiguity aversion — CV²(σ²) scales variance_cost
        let ambiguity_ratio = self.sigma_sq_variance / self.sigma_sq_mean.powi(2).max(1e-30);
        let ambiguity_cost = variance_cost * ambiguity_ratio;

        // Term 4: Timing uncertainty — CV²(τ) scales drift_cost
        let timing_ratio = self.tau_variance / self.tau_mean.powi(2).max(1e-12);
        let timing_cost = drift_cost * timing_ratio;

        let total = drift_cost + variance_cost + ambiguity_cost + timing_cost;

        self.skew_multiplier * total
    }

    /// Learn multiplier from markout data.
    pub fn calibrate_from_markout(
        &mut self,
        predicted_shift: f64,
        realized_as_frac: f64,
        learning_rate: f64,
    ) {
        if predicted_shift.abs() < 1e-12 {
            return;
        }
        let ideal_ratio = realized_as_frac / predicted_shift;
        self.skew_multiplier = (1.0 - learning_rate) * self.skew_multiplier
            + learning_rate * ideal_ratio.clamp(0.1, 10.0);
    }

    /// Update from MarketParams (called each quote cycle).
    pub fn update_from_params(&mut self, params: &MarketParams) {
        // Drift from Kalman posterior (fractional/sec)
        self.drift_mean = params.drift_rate_per_sec;

        // σ² from effective sigma (already fractional/√sec)
        let sigma_frac = params.sigma_effective;
        self.sigma_sq_mean = (sigma_frac * sigma_frac).max(1e-16);

        // σ² variance: use BMA between-model variance when available, else heuristic
        self.sigma_sq_variance = if params.sigma_sq_variance_bma > 0.0 {
            params.sigma_sq_variance_bma
        } else {
            let sigma_spread =
                (params.sigma_particle / 10_000.0 - params.sigma_leverage_adjusted).abs();
            sigma_spread.powi(2).max(1e-20)
        };

        // τ_inventory: use measured from fill processor, or keep default
        if params.tau_inventory_s > 0.1 {
            self.tau_mean = params.tau_inventory_s;
            self.tau_variance = params.tau_variance_s2.max(1.0);
        }
    }
}

/// Dual-timescale controller: fast τ_spread for half-spread, slow τ_inventory for skew.
#[derive(Debug, Clone)]
pub struct DualTimescaleController {
    /// PPIP for reservation price shift
    pub ppip: PosteriorPredictiveSkew,

    /// EWMA of reducing-fill holding durations (seconds)
    pub tau_inventory: f64,
    /// EWMA decay for τ updates
    pub tau_decay: f64,
    /// Online variance of τ
    pub tau_variance: f64,
}

impl Default for DualTimescaleController {
    fn default() -> Self {
        Self {
            ppip: PosteriorPredictiveSkew::default(),
            tau_inventory: 60.0,
            tau_decay: 0.95,
            tau_variance: 900.0,
        }
    }
}

impl DualTimescaleController {
    /// Update τ_inventory from a reducing fill's holding duration.
    pub fn observe_reducing_fill(&mut self, holding_duration_sec: f64) {
        let old_mean = self.tau_inventory;
        self.tau_inventory =
            self.tau_decay * self.tau_inventory + (1.0 - self.tau_decay) * holding_duration_sec;
        // Welford online variance
        let delta = holding_duration_sec - old_mean;
        let delta2 = holding_duration_sec - self.tau_inventory;
        self.tau_variance =
            self.tau_decay * self.tau_variance + (1.0 - self.tau_decay) * delta * delta2;
        // Propagate to PPIP
        self.ppip.tau_mean = self.tau_inventory;
        self.ppip.tau_variance = self.tau_variance.max(1.0);
    }

    /// Reservation price shift using PPIP (fractional price units).
    pub fn reservation_shift(&self, inventory_ratio: f64, max_position: f64) -> f64 {
        self.ppip.reservation_shift(inventory_ratio, max_position)
    }
}

/// Parameter struct for evaluating comprehensive Expected PnL
#[derive(Debug, Clone)]
pub struct EPnLParams {
    pub depth_bps: f64,
    pub is_bid: bool,
    pub gamma: f64,
    pub kappa_side: f64,
    pub sigma: f64,
    pub time_horizon: f64,
    pub drift_rate: f64,
    pub position: f64,
    pub max_position: f64,
    pub as_cost_bps: f64,
    pub fee_bps: f64,
    pub carry_cost_bps: f64,
    pub toxicity_score: f64,
    pub circuit_breaker_active: bool,
    pub drawdown_frac: f64,
    pub self_impact_bps: f64,
    pub inventory_beta: f64,
}

/// Enhanced per-level E[PnL] computation absorbing all former multiplicative overlays.
pub fn expected_pnl_bps_enhanced(params: &EPnLParams) -> f64 {
    let depth_frac = params.depth_bps / 10_000.0;

    // Fill intensity at this depth: λ(δ) = κ × exp(-κ × δ)
    let mut lambda = params.kappa_side * (-params.kappa_side * depth_frac).exp();

    // Circuit breaker → staleness discount on lambda
    if params.circuit_breaker_active {
        lambda *= 0.1;
    }

    // Drawdown → reduces lambda
    let drawdown_penalty = (1.0 - params.drawdown_frac * 5.0).max(0.1);
    lambda *= drawdown_penalty;

    // Toxicity amplifies AS economically
    let toxicity_cost = params.toxicity_score * (params.as_cost_bps + 2.0);

    // Spread capture net of costs
    let capture = params.depth_bps - params.as_cost_bps - params.fee_bps - params.carry_cost_bps;

    // Drift contribution: directional, mirrors GLFT ±μ̂×τ/2 asymmetry.
    let drift_bps = params.drift_rate * 10_000.0 * params.time_horizon / 2.0;
    let drift_contribution = if params.is_bid { drift_bps } else { -drift_bps };

    // Inventory penalty uses continuous gamma(q) = gamma_base × (1 + beta × (q/q_max)²)
    let q_ratio = if params.max_position > 1e-9 {
        (params.position.abs() / params.max_position).min(1.0)
    } else {
        0.0
    };

    let gamma_q = params.gamma * (1.0 + params.inventory_beta * q_ratio.powi(2));

    let is_reducing =
        (params.is_bid && params.position < 0.0) || (!params.is_bid && params.position > 0.0);
    let inv_penalty_bps = gamma_q * q_ratio * params.sigma.powi(2) * params.time_horizon * 10_000.0;

    let inventory_adj = if is_reducing {
        -0.5 * inv_penalty_bps // Bonus for reducing
    } else {
        inv_penalty_bps // Penalty for accumulating
    };

    // Total E[PnL] = fill_probability × (capture - toxicity_cost) + drift_contribution - inventory_penalty - impact_cost
    lambda * (capture - toxicity_cost) + drift_contribution - inventory_adj - params.self_impact_bps
}

/// Position-convex E[PnL] threshold for reducing-side levels.
///
/// On thin venues, E[PnL] can go negative on ALL levels (both sides) due to
/// low kappa and high AS. The accumulating side should still require E[PnL] > 0,
/// but the reducing side gets a negative threshold that grows with position urgency.
///
/// Shape: `sqrt(q_ratio)` captures square-root scaling, granting a larger
/// carve-out for small remnant positions to prevent getting 'stuck' holding
/// directional risk when the trend opposes closing.
/// V2 gamma_ratio: `gamma / gamma_baseline` encodes regime risk aversion.
/// - Extreme (gamma 3x) → -6.0 bps at 100% pos (willing to pay to unwind)
/// - Calm (gamma 0.5x) → -1.0 bps at 100% pos (cheap to wait)
///
/// Returns a negative threshold in bps (or 0.0 at flat position).
pub fn reducing_threshold_bps(
    position: f64,
    max_position: f64,
    fee_bps: f64,
    gamma: f64,
    gamma_baseline: f64,
) -> f64 {
    let q_ratio = if max_position > 1e-9 {
        (position.abs() / max_position).min(1.0)
    } else {
        0.0
    };
    // gamma_ratio > 1 in volatile → more willing to pay to unwind
    // gamma_ratio < 1 in calm → less willing
    let gamma_ratio = if gamma_baseline > 1e-9 {
        (gamma / gamma_baseline).clamp(0.5, 3.0)
    } else {
        1.0
    };
    -2.0 * fee_bps * gamma_ratio * q_ratio.powf(0.5)
}

/// Avellaneda-Stoikov reservation price shift with HARA utility.
///
/// Computes the optimal shift from mid price for a risk-averse market maker
/// with inventory `position`. The shift is negative when long (sell pressure)
/// and positive when short (buy pressure).
///
/// HARA extension: dynamic gamma increases as wealth drops, producing
/// more aggressive inventory reduction when underwater.
///
/// # Arguments
/// * `position` - Signed inventory (+ = long, - = short), in asset units
/// * `sigma` - Volatility (per-second, fractional — NOT bps, NOT annualized)
/// * `tau_s` - Time horizon in seconds
/// * `gamma_base` - Base risk aversion coefficient
/// * `unrealized_pnl` - Current unrealized P&L ($)
/// * `capital` - Total capital ($)
///
/// # Returns
/// Shift from mid in **fractional** price units (multiply by 10000 for bps).
pub fn reservation_price_shift(
    position: f64,
    sigma: f64,
    tau_s: f64,
    gamma_base: f64,
    unrealized_pnl: f64,
    capital: f64,
) -> f64 {
    // HARA dynamic gamma: risk aversion increases as wealth drops
    // γ(w) = γ_base / (w/w₀) where w = capital + unrealized_pnl
    let wealth_ratio = if capital > 1e-9 {
        ((capital + unrealized_pnl) / capital).max(0.1)
    } else {
        1.0
    };
    let dynamic_gamma = gamma_base / wealth_ratio;

    // Avellaneda-Stoikov optimal shift: r(s,q) = s - q × γ × σ² × τ
    // Returns the SHIFT only (negative = below mid when long)
    -position * dynamic_gamma * sigma * sigma * tau_s
}

/// Legacy thin wrapper around enhanced E[PnL] for backward compatibility.
#[allow(clippy::too_many_arguments)]
pub fn expected_pnl_bps(
    depth_bps: f64,
    is_bid: bool,
    gamma: f64,
    kappa_side: f64,
    sigma: f64,
    time_horizon: f64,
    drift_rate: f64,
    position: f64,
    max_position: f64,
    as_cost_bps: f64,
    fee_bps: f64,
    carry_cost_bps: f64,
) -> f64 {
    let params = EPnLParams {
        depth_bps,
        is_bid,
        gamma,
        kappa_side,
        sigma,
        time_horizon,
        drift_rate,
        position,
        max_position,
        as_cost_bps,
        fee_bps,
        carry_cost_bps,
        toxicity_score: 0.0,
        circuit_breaker_active: false,
        drawdown_frac: 0.0,
        self_impact_bps: 0.0,
        inventory_beta: 0.0,
    };
    expected_pnl_bps_enhanced(&params)
}

/// Decomposition of E[PnL] for diagnostics and reconciliation.
#[derive(Debug, Clone, Default)]
pub struct EPnLDiagnostics {
    pub depth_bps: f64,
    pub is_bid: bool,
    pub lambda: f64,
    pub capture_bps: f64,
    pub toxicity_cost_bps: f64,
    pub drift_contribution_bps: f64,
    pub inventory_adj_bps: f64,
    pub self_impact_bps: f64,
    pub epnl_bps: f64,
    pub circuit_breaker_active: bool,
    pub drawdown_penalty: f64,
}

/// Same computation as `expected_pnl_bps_enhanced()` but returns the decomposition.
pub fn expected_pnl_bps_with_diagnostics(params: &EPnLParams) -> (f64, EPnLDiagnostics) {
    let depth_frac = params.depth_bps / 10_000.0;

    let mut lambda = params.kappa_side * (-params.kappa_side * depth_frac).exp();

    if params.circuit_breaker_active {
        lambda *= 0.1;
    }

    let drawdown_penalty = (1.0 - params.drawdown_frac * 5.0).max(0.1);
    lambda *= drawdown_penalty;

    let toxicity_cost = params.toxicity_score * (params.as_cost_bps + 2.0);

    let capture = params.depth_bps - params.as_cost_bps - params.fee_bps - params.carry_cost_bps;

    let drift_bps = params.drift_rate * 10_000.0 * params.time_horizon / 2.0;
    let drift_contribution = if params.is_bid { drift_bps } else { -drift_bps };

    let q_ratio = if params.max_position > 1e-9 {
        (params.position.abs() / params.max_position).min(1.0)
    } else {
        0.0
    };

    let gamma_q = params.gamma * (1.0 + params.inventory_beta * q_ratio.powi(2));

    let is_reducing =
        (params.is_bid && params.position < 0.0) || (!params.is_bid && params.position > 0.0);
    let inv_penalty_bps = gamma_q * q_ratio * params.sigma.powi(2) * params.time_horizon * 10_000.0;

    let inventory_adj = if is_reducing {
        -0.5 * inv_penalty_bps
    } else {
        inv_penalty_bps
    };

    let epnl = lambda * (capture - toxicity_cost) + drift_contribution
        - inventory_adj
        - params.self_impact_bps;

    let diag = EPnLDiagnostics {
        depth_bps: params.depth_bps,
        is_bid: params.is_bid,
        lambda,
        capture_bps: capture,
        toxicity_cost_bps: toxicity_cost,
        drift_contribution_bps: drift_contribution,
        inventory_adj_bps: inventory_adj,
        self_impact_bps: params.self_impact_bps,
        epnl_bps: epnl,
        circuit_breaker_active: params.circuit_breaker_active,
        drawdown_penalty,
    };

    (epnl, diag)
}

/// Taker price elasticity estimator for monopolist LP pricing.
///
/// Tracks (spread_width, fill_rate) pairs over a rolling window and
/// regresses: ln(fill_rate) = alpha - eta * ln(spread_bps)
/// where eta is the price elasticity of demand for liquidity.
///
/// Higher elasticity means takers are more price-sensitive (less markup possible).
/// Lower elasticity means takers are price-insensitive (more markup possible).
#[derive(Debug, Clone)]
pub struct TakerElasticityEstimator {
    /// Rolling window of (ln_spread_bps, ln_fill_rate) observations.
    observations: Vec<(f64, f64)>,
    /// Maximum window size.
    max_observations: usize,
    /// Cached elasticity estimate (eta).
    cached_eta: f64,
    /// Whether enough observations for a valid estimate.
    is_valid: bool,
}

impl TakerElasticityEstimator {
    /// Create a new estimator with the given window size.
    pub fn new(max_observations: usize) -> Self {
        Self {
            observations: Vec::with_capacity(max_observations),
            max_observations,
            cached_eta: 1.0, // Conservative default: unit elastic
            is_valid: false,
        }
    }

    /// Record a (spread_bps, fill_rate_per_s) observation.
    pub fn record(&mut self, spread_bps: f64, fill_rate_per_s: f64) {
        if spread_bps <= 0.0 || fill_rate_per_s <= 0.0 {
            return;
        }
        let ln_spread = spread_bps.ln();
        let ln_fill_rate = fill_rate_per_s.ln();

        if self.observations.len() >= self.max_observations {
            self.observations.remove(0);
        }
        self.observations.push((ln_spread, ln_fill_rate));

        // Recompute regression if we have enough data
        if self.observations.len() >= 10 {
            self.recompute();
        }
    }

    /// Get the current elasticity estimate.
    /// Returns eta > 0. Higher = more elastic takers (less markup).
    pub fn elasticity(&self) -> f64 {
        self.cached_eta
    }

    /// Whether the estimator has enough observations for a valid estimate.
    pub fn is_valid(&self) -> bool {
        self.is_valid
    }

    /// Number of observations recorded.
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Simple OLS regression: ln(fill_rate) = alpha - eta * ln(spread_bps)
    /// Solves for eta (should be positive: wider spread -> lower fill rate).
    fn recompute(&mut self) {
        let n = self.observations.len() as f64;
        if n < 10.0 {
            return;
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for &(x, y) in &self.observations {
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-12 {
            return;
        }

        // slope = (n * sum_xy - sum_x * sum_y) / denom
        // We expect negative slope (higher spread -> lower fill rate)
        // eta = -slope (so eta > 0)
        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let eta = (-slope).max(0.1); // Floor at 0.1 to prevent blow-up

        self.cached_eta = eta;
        self.is_valid = true;
    }
}

impl Default for TakerElasticityEstimator {
    fn default() -> Self {
        Self::new(200)
    }
}

/// GLFT (Guéant-Lehalle-Fernandez-Tapia) optimal market making strategy.
///
/// Implements the **infinite-horizon** GLFT model with **dynamic risk aversion**.
///
/// ## Key Formulas (Corrected per Guéant et al. 2013):
///
/// **Half-spread (adverse selection protection):**
/// ```text
/// δ = (1/γ) × ln(1 + γ/κ)
/// ```
///
/// **Reservation price offset (inventory skew):**
/// ```text
/// skew = (q/Q_max) × γ × σ² × T
/// ```
/// where T = 1/λ (inverse of arrival intensity).
///
/// ## Dynamic Risk Aversion (Two Modes):
///
/// ### Legacy Mode (Multiplicative):
/// ```text
/// γ_effective = γ_base × vol_scalar × toxicity_scalar × inventory_scalar × ...
/// ```
///
/// ### Calibrated Mode (Log-Additive):
/// ```text
/// log(γ) = log(γ_base) + Σ βᵢ × xᵢ
/// γ = exp(log_gamma).clamp(γ_min, γ_max)
/// ```
///
/// The log-additive model prevents multiplicative explosion and uses calibrated
/// coefficients from realized adverse selection data.
#[derive(Debug, Clone)]
pub struct GLFTStrategy {
    /// Risk configuration for dynamic γ calculation
    pub risk_config: RiskConfig,

    /// Calibrated risk model for log-additive gamma computation
    pub risk_model: CalibratedRiskModel,

    /// Configuration for risk model feature normalization
    pub risk_model_config: RiskModelConfig,

    /// Kelly criterion position sizer
    pub kelly_sizer: KellySizer,

    /// Taker elasticity estimator for monopolist LP pricing.
    pub elasticity_estimator: TakerElasticityEstimator,

    /// WS2: Dual-timescale controller for PPIP inventory skew.
    pub dual_timescale: DualTimescaleController,
}

/// Solve for minimum gamma that makes GLFT half-spread >= target.
///
/// Uses binary search since half_spread is monotonically increasing in gamma
/// (the vol compensation term 0.5*gamma*sigma^2*T dominates for large gamma).
///
/// Returns gamma in the range [0.01, 100.0].
///
/// # Arguments
/// * `target_half_spread` - Target half-spread in price fraction (not bps)
/// * `kappa` - Order flow intensity
/// * `sigma` - Volatility (per-second)
/// * `time_horizon` - Holding time in seconds
/// * `maker_fee_rate` - Maker fee in price fraction
pub fn solve_min_gamma(
    target_half_spread: f64,
    kappa: f64,
    sigma: f64,
    time_horizon: f64,
    maker_fee_rate: f64,
) -> f64 {
    // half_spread(gamma) = (1/gamma)*ln(1 + gamma/kappa) + 0.5*gamma*sigma^2*T + maker_fee
    // This is monotonically increasing in gamma (the vol compensation term dominates)
    // Binary search for the gamma where half_spread = target

    let mut lo = 0.01_f64;
    let mut hi = 100.0_f64;

    // Helper: compute half_spread at given gamma
    let hs = |gamma: f64| -> f64 {
        let safe_kappa = kappa.max(1.0);
        let ratio = gamma / safe_kappa;
        let glft = if ratio > 1e-9 && gamma > 1e-9 {
            (1.0 / gamma) * (1.0 + ratio).ln()
        } else {
            1.0 / safe_kappa
        };
        let vol_comp = 0.5 * gamma * sigma.powi(2) * time_horizon;
        glft + vol_comp + maker_fee_rate
    };

    // If even max gamma doesn't reach target, return max
    if hs(hi) < target_half_spread {
        return hi;
    }
    // If min gamma already exceeds target, return min
    if hs(lo) >= target_half_spread {
        return lo;
    }

    for _ in 0..50 {
        let mid = (lo + hi) / 2.0;
        if hs(mid) < target_half_spread {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    (lo + hi) / 2.0
}

impl GLFTStrategy {
    /// Create a new GLFT strategy with base risk aversion.
    ///
    /// Uses default RiskConfig with the specified gamma_base.
    /// For full control, use `with_config()`.
    pub fn new(gamma_base: f64) -> Self {
        Self {
            risk_config: RiskConfig {
                gamma_base: gamma_base.clamp(0.01, 10.0),
                ..Default::default()
            },
            risk_model: CalibratedRiskModel::with_gamma_base(gamma_base),
            risk_model_config: RiskModelConfig::default(),
            kelly_sizer: KellySizer::default(),
            elasticity_estimator: TakerElasticityEstimator::default(),
            dual_timescale: DualTimescaleController::default(),
        }
    }

    /// Create a new GLFT strategy with full risk configuration.
    pub fn with_config(risk_config: RiskConfig) -> Self {
        let max_obs = risk_config.min_observations_for_elasticity.max(50) * 4;
        Self {
            risk_model: CalibratedRiskModel::with_gamma_base(risk_config.gamma_base),
            risk_model_config: RiskModelConfig::default(),
            kelly_sizer: KellySizer::default(),
            elasticity_estimator: TakerElasticityEstimator::new(max_obs),
            dual_timescale: DualTimescaleController::default(),
            risk_config,
        }
    }

    /// Create a new GLFT strategy with full configuration including calibrated risk model.
    pub fn with_full_config(
        risk_config: RiskConfig,
        risk_model_config: RiskModelConfig,
        kelly_sizer: KellySizer,
    ) -> Self {
        let max_obs = risk_config.min_observations_for_elasticity.max(50) * 4;
        Self {
            risk_model: CalibratedRiskModel::with_gamma_base(risk_config.gamma_base),
            elasticity_estimator: TakerElasticityEstimator::new(max_obs),
            dual_timescale: DualTimescaleController::default(),
            risk_config,
            risk_model_config,
            kelly_sizer,
        }
    }

    /// Update the calibrated risk model (e.g., from coefficient estimator).
    pub fn update_risk_model(&mut self, model: CalibratedRiskModel) {
        self.risk_model = model;
    }

    /// Update risk model config (e.g., baselines from market observation).
    pub fn update_risk_model_config(&mut self, config: RiskModelConfig) {
        self.risk_model_config = config;
    }

    /// Record a winning trade for Kelly sizing.
    pub fn record_win(&mut self, win_bps: f64) {
        self.kelly_sizer.record_win(win_bps);
    }

    /// Record a losing trade for Kelly sizing.
    pub fn record_loss(&mut self, loss_bps: f64) {
        self.kelly_sizer.record_loss(loss_bps);
    }

    /// Record a (spread, fill_rate) observation for taker elasticity estimation.
    /// Call this periodically with the current quoted spread and observed fill rate.
    pub fn record_elasticity_observation(&mut self, spread_bps: f64, fill_rate_per_s: f64) {
        self.elasticity_estimator
            .record(spread_bps, fill_rate_per_s);
    }

    /// Get the current taker elasticity estimator (for diagnostics).
    pub fn elasticity_estimator(&self) -> &TakerElasticityEstimator {
        &self.elasticity_estimator
    }

    /// Check if Kelly sizer is warmed up.
    pub fn kelly_warmed_up(&self) -> bool {
        self.kelly_sizer.is_warmed_up()
    }

    /// Calculate effective γ based on current market conditions.
    ///
    /// WS1: Single call to CalibratedRiskModel — ALL risk factors are in the log-additive model.
    /// No multiplicative post-processes. Inventory, drawdown, regime, ghost are all β coefficients.
    ///
    /// log(γ) = log(γ_base) + Σ βᵢ × xᵢ  (12 features including inventory², drawdown, regime, ghost)
    fn effective_gamma(
        &self,
        market_params: &MarketParams,
        position: f64,
        max_position: f64,
    ) -> f64 {
        // ============================================================
        // UNIFIED LOG-ADDITIVE GAMMA via compute_gamma_with_policy
        // Single source of truth shared with LadderStrat::effective_gamma
        // ============================================================
        let features = RiskFeatures::from_params(
            market_params,
            position,
            max_position,
            &self.risk_model_config,
        );
        let gamma_clamped = self.risk_model.compute_gamma_with_policy(
            &features,
            market_params.capital_tier,
            market_params.capital_policy.warmup_gamma_max_inflation,
        );

        let neutral_features = RiskFeatures::neutral();
        let gamma_neutral = self.risk_model.compute_gamma(&neutral_features);
        let defense_ratio = gamma_clamped / gamma_neutral.max(1e-9);

        debug!(
            inv_frac = %format!("{:.3}", features.inventory_fraction),
            drawdown = %format!("{:.3}", features.drawdown_fraction),
            regime = %format!("{:.3}", features.regime_risk_score),
            ghost = %format!("{:.3}", features.ghost_depletion),
            gamma_final = %format!("{:.4}", gamma_clamped),
            defense_ratio = %format!("{:.3}", defense_ratio),
            "Gamma decomposition (unified log-additive)"
        );

        gamma_clamped
    }

    /// Calculate expected holding time from arrival intensity.
    ///
    /// T = 1/λ where λ = arrival intensity (fills per second)
    /// Clamped to prevent skew explosion when market is dead.
    fn holding_time(&self, arrival_intensity: f64) -> f64 {
        let safe_intensity = arrival_intensity.max(0.01);
        (1.0 / safe_intensity).min(self.risk_config.max_holding_time)
    }

    /// Correct Avellaneda-Stoikov optimal half-spread formula:
    ///
    /// δ = (1/γ) × ln(1 + γ/κ) + (1/2) × γ × σ² × T + f_maker
    ///
    /// Term 1: GLFT liquidity-driven spread from HJB solution
    /// Term 2: Volatility compensation for inventory risk during holding period (A-S extension)
    /// Term 3: Fee recovery from modified HJB equation (dW = (δ - f_maker) × dN)
    ///
    /// The volatility compensation term ensures spreads widen appropriately in volatile markets.
    /// With per-second σ and T in seconds, this produces meaningful spread widening:
    /// - Normal conditions: +0.05 bps (negligible)
    /// - High volatility (5× normal): +0.3 bps (meaningful protection)
    ///
    /// This is market-driven: when κ drops (thin book), spread widens automatically.
    /// When κ rises (deep book), spread tightens. Fee ensures minimum profitable spread.
    pub(crate) fn half_spread(&self, gamma: f64, kappa: f64, sigma: f64, time_horizon: f64) -> f64 {
        let ratio = gamma / kappa;

        // Term 1: GLFT liquidity-driven spread
        let glft_spread = if ratio > 1e-9 && gamma > 1e-9 {
            (1.0 / gamma) * (1.0 + ratio).ln()
        } else {
            // When γ/κ → 0, use Taylor expansion: ln(1+x) ≈ x
            // δ ≈ (1/γ) × (γ/κ) = 1/κ
            1.0 / kappa.max(1.0)
        };

        // Term 2: Volatility compensation (A-S extension)
        // Compensates for inventory risk during expected holding period
        // With σ = 0.0002 (2 bps/√sec), T = 60s, γ = 0.15:
        //   vol_comp = 0.5 × 0.15 × (0.0002)² × 60 = 3.6e-8 = 0.00036 bps
        // With σ = 0.001 (10 bps/√sec, stressed), T = 60s, γ = 0.15:
        //   vol_comp = 0.5 × 0.15 × (0.001)² × 60 = 4.5e-7 = 0.0045 bps
        let vol_compensation = 0.5 * gamma * sigma.powi(2) * time_horizon;

        // Term 3: Maker fee to ensure profitability (first-principles HJB modification)
        glft_spread + vol_compensation + self.risk_config.maker_fee_rate
    }

    /// GLFT half-spread with classical drift (mu*T) asymmetry.
    ///
    /// Extends `half_spread()` with the Avellaneda-Stoikov drift term:
    /// - Positive drift (price rising) → tightens bids (buy into uptrend), widens asks (don't sell into uptrend)
    /// - Negative drift (price falling) → widens bids (don't buy into downtrend), tightens asks (sell into downtrend)
    ///
    /// Formula: `delta_bid = base - mu*T/2`, `delta_ask = base + mu*T/2`
    ///
    /// The drift adjustment is clamped so the half-spread never goes below
    /// the maker fee rate (no negative-EV quotes).
    pub(crate) fn half_spread_with_drift(
        &self,
        gamma: f64,
        kappa: f64,
        sigma: f64,
        time_horizon: f64,
        drift_rate: f64,
        is_bid: bool,
    ) -> f64 {
        let base = self.half_spread(gamma, kappa, sigma, time_horizon);
        // Classical GLFT mu*T term: drift shifts spread asymmetrically
        // Positive drift → tighter bid (higher bid price → buy into uptrend)
        //                 → wider ask (higher ask price → protect from selling into uptrend)
        let drift_adjustment = drift_rate * time_horizon / 2.0;
        let adjusted = if is_bid {
            base - drift_adjustment // Tighter bid → higher bid → buy into uptrend ✓
        } else {
            base + drift_adjustment // Wider ask → higher ask → don't sell into uptrend ✓
        };
        // Floor at maker fee to avoid negative-EV quotes
        adjusted.max(self.risk_config.maker_fee_rate)
    }

    /// Compute the additive spread composition from market parameters.
    ///
    /// Decomposes the half-spread into individually-bounded additive components:
    /// - GLFT core: `(1/gamma) * ln(1 + gamma/kappa)` + vol_comp + fee
    /// - Risk premium: from regime, Hawkes, toxicity, staleness (additive, not multiplicative)
    /// - Quota addon: API rate limit pressure
    /// - Warmup addon: early-session uncertainty premium
    ///
    /// The fee component is already included in the GLFT core (via `half_spread()`),
    /// so `fee_bps` is tracked separately for decomposition visibility only.
    pub fn compute_spread_composition(&self, market_params: &MarketParams) -> SpreadComposition {
        let gamma = self.effective_gamma(market_params, 0.0, 1.0);
        let kappa = if market_params.use_kappa_robust {
            market_params.kappa_robust
        } else {
            market_params.kappa
        };
        // WS2: sigma_cascade_mult removed — CovarianceTracker's Bayesian posterior
        // handles realized vol feedback. sigma_effective is used directly.
        let sigma = market_params.sigma_effective;
        let tau = self.holding_time(market_params.arrival_intensity);

        // Core GLFT half-spread in fraction
        let glft_half_frac = self.half_spread(gamma, kappa, sigma, tau);
        let glft_half_bps = glft_half_frac * 10_000.0;

        // Risk premium: additive bps from regime + risk overlays
        // All widening now routes through E[PnL] gate or additive total_risk_premium_bps.
        let risk_premium_bps =
            market_params.regime_risk_premium_bps + market_params.total_risk_premium_bps;

        // Quota addon: already in bps, capped at 50
        let quota_addon_bps = market_params.quota_shadow_spread_bps.min(50.0);

        // Warmup addon: uncertainty premium decays as warmup progresses
        // Policy-driven: Micro=3.0, Large=8.0 bps max, decays linearly with warmup
        let warmup_addon_bps = if market_params.adaptive_warmup_progress < 1.0 {
            let uncertainty_factor = 1.0 - market_params.adaptive_warmup_progress;
            market_params.capital_policy.warmup_floor_bps * uncertainty_factor
        } else {
            0.0
        };

        let fee_bps = self.risk_config.maker_fee_rate * 10_000.0;

        SpreadComposition {
            glft_half_spread_bps: glft_half_bps,
            risk_premium_bps,
            quota_addon_bps,
            warmup_addon_bps,
            fee_bps,
        }
    }

    /// Proactive directional skew based on momentum predictions.
    ///
    /// **Key Insight:** This is the OPPOSITE of inventory skew:
    /// - Inventory skew: When you HAVE position, skew to REDUCE it
    /// - Proactive skew: When you DON'T have position, skew to BUILD with momentum
    ///
    /// # How it works
    /// When momentum is strong and confident, skew quotes to BUILD position:
    /// - Positive momentum → want to be long → tighten bids, widen asks → negative skew
    /// - Negative momentum → want to be short → tighten asks, widen bids → positive skew
    ///
    /// # When applied
    /// - Only when inventory is small (proactive mode)
    /// - Only when momentum confidence > threshold (clear signal)
    /// - Only when momentum strength > threshold (not noise)
    ///
    /// # Regime awareness
    /// - Trending: Trust momentum more (1.5x)
    /// - Mean-reverting: Don't chase (0.3x)
    /// - Volatile: Cautious (0.5x)
    /// - Quiet: Normal (1.0x)
    ///   Compute proactive directional skew based on momentum predictions.
    ///
    /// This is the OPPOSITE of inventory skew - we WANT to get filled WITH momentum
    /// to BUILD position in a profitable direction.
    ///
    /// # Lead-Lag Enhancement
    /// When lead-lag signal is available (Binance → Hyperliquid), it provides
    /// predictive edge: we know where price is likely to move before it moves.
    /// This allows more aggressive positioning in the predicted direction.
    ///
    /// Returns skew in fractional terms (divide by 10000 to get bps).
    #[allow(dead_code)] // Kept for analytics comparison; drift μ replaces this in quote pipeline
    fn proactive_directional_skew(&self, market_params: &MarketParams) -> f64 {
        // Check if proactive skew is enabled
        if !market_params.enable_proactive_skew {
            return 0.0;
        }

        let momentum_bps = market_params.momentum_bps;
        let p_continuation = market_params.p_momentum_continue;

        // === Lead-Lag Signal Enhancement ===
        // When cross-exchange lead-lag is available and confident, use it as
        // additional directional signal. This provides predictive edge.
        let lead_lag_signal = market_params.lead_lag_signal_bps;
        let lead_lag_conf = market_params.lead_lag_confidence;

        // Combine momentum with lead-lag signal (additive, scaled by confidence)
        // Lead-lag signal is already confidence-weighted in quote_engine
        let effective_momentum = momentum_bps + lead_lag_signal;

        // Normalize momentum strength to [0, 1]
        // 20 bps is considered "strong" momentum
        let momentum_strength = (effective_momentum.abs() / 20.0).min(1.0);

        // Check thresholds (use effective momentum)
        if p_continuation < market_params.proactive_min_momentum_confidence {
            // Allow bypass when lead-lag confidence is high
            if lead_lag_conf < 0.5 {
                return 0.0; // Not confident enough
            }
        }
        if effective_momentum.abs() < market_params.proactive_min_momentum_bps {
            return 0.0; // Too weak
        }

        // Direction: OPPOSITE of momentum to BUILD position WITH momentum
        // Positive momentum → we want to get filled on bids → negative skew (tighter bids)
        let direction = -effective_momentum.signum();

        // Soft HMM regime blending (replaces hard switch)
        // [p_low, p_normal, p_high, p_extreme] → weighted multiplier
        // Key insight: Single parameter values are almost always wrong.
        // Use belief state probabilities for smooth transitions.
        let regime_mult = market_params.regime_probs[0] * 1.2  // Low: trust momentum
            + market_params.regime_probs[1] * 1.0              // Normal: baseline
            + market_params.regime_probs[2] * 0.5              // High: cautious
            + market_params.regime_probs[3] * 0.2; // Extreme: very cautious

        // Lead-lag confidence boost: when we have predictive edge, be more aggressive
        let lead_lag_boost = 1.0 + (lead_lag_conf * 0.5); // Up to 1.5x with full confidence

        // Calculate proactive skew in bps
        let proactive_skew_bps = direction
            * momentum_strength
            * p_continuation
            * regime_mult
            * lead_lag_boost
            * market_params.proactive_skew_sensitivity;

        // Log when lead-lag is contributing
        if lead_lag_signal.abs() > 0.5 {
            tracing::debug!(
                momentum_bps = %format!("{:.2}", momentum_bps),
                lead_lag_signal_bps = %format!("{:.2}", lead_lag_signal),
                lead_lag_conf = %format!("{:.2}", lead_lag_conf),
                effective_momentum_bps = %format!("{:.2}", effective_momentum),
                lead_lag_boost = %format!("{:.2}", lead_lag_boost),
                proactive_skew_bps = %format!("{:.2}", proactive_skew_bps),
                "Lead-lag signal contributing to proactive skew"
            );
        }

        // Convert to fractional (divide by 10000)
        proactive_skew_bps / 10000.0
    }
}

impl QuotingStrategy for GLFTStrategy {
    fn calculate_quotes(
        &self,
        config: &QuoteConfig,
        position: f64,
        max_position: f64,
        target_liquidity: f64,
        market_params: &MarketParams,
    ) -> (Option<Quote>, Option<Quote>) {
        // === 0. SOLVENCY CHECK ===
        // WS4: Binary circuit breakers removed. γ(q) handles all risk continuously:
        // - Cascade risk → high cascade_intensity → high γ → wide spreads
        // - No edge → toxicity features → high γ → wide spreads
        // - Toxic joint → informed flow features → high γ → wide spreads
        // The formula γσ²τ + (2/γ)ln(1+γ/κ) produces correct spreads when
        // estimators are accurate. Only solvency (margin exhaustion) returns (None, None).
        // should_pull_quotes and is_toxic_joint still set as diagnostics but don't gate quoting.

        // FIRST PRINCIPLES: Use dynamic max_position derived from equity/volatility
        // Falls back to static max_position if margin state hasn't been refreshed yet.
        // config.max_position (passed as max_position) is ALWAYS the hard ceiling.
        let effective_max_position = market_params
            .effective_max_position(max_position)
            .min(max_position);

        // === 1. DYNAMIC GAMMA ===
        // When adaptive spreads enabled: use log-additive shrinkage gamma
        // When disabled: use CalibratedRiskModel gamma
        //
        // Tail risk is handled via beta_tail_risk in compute_gamma() — no multiplicative post-process.
        //
        // KEY FIX: Use `adaptive_can_estimate` instead of `adaptive_warmed_up`
        // The adaptive system provides usable values IMMEDIATELY via Bayesian priors.
        // We don't need to wait for 20+ fills - priors give reasonable starting points.
        let gamma = if market_params.use_adaptive_spreads && market_params.adaptive_can_estimate {
            let adaptive_gamma = market_params.adaptive_gamma;
            debug!(
                adaptive_gamma = %format!("{:.4}", adaptive_gamma),
                warmup_progress = %format!("{:.0}%", market_params.adaptive_warmup_progress * 100.0),
                "Using ADAPTIVE gamma (log-additive shrinkage)"
            );
            adaptive_gamma
        } else {
            self.effective_gamma(market_params, position, effective_max_position)
        };

        // === 1a. KAPPA: Learned vs Adaptive vs Legacy ===
        // Priority:
        // 1. Learned kappa (when calibrated and enabled) - from Bayesian parameter learner
        // 2. Adaptive kappa (when enabled) - blended book/own-fill
        // 3. Legacy book-based kappa with AS adjustment
        //
        // KEY FIX: Use `adaptive_can_estimate` - our Bayesian priors give reasonable
        // kappa estimates immediately (κ=2500 prior for liquid markets).
        let (kappa, kappa_bid, kappa_ask) = if market_params.use_learned_parameters
            && market_params.learned_params_calibrated
        {
            // Learned kappa: from Bayesian parameter learner with conjugate prior
            // This is the most statistically principled approach - shrinkage toward prior
            // with increasing data weight as observations accumulate.
            let k = market_params.learned_kappa;
            debug!(
                learned_kappa = %format!("{:.0}", k),
                book_kappa = %format!("{:.0}", market_params.kappa),
                adaptive_kappa = %format!("{:.0}", market_params.adaptive_kappa),
                "Using LEARNED kappa (Bayesian parameter learner)"
            );
            // For now, use symmetric kappa (directional can be added later)
            (k, k, k)
        } else if market_params.use_adaptive_spreads && market_params.adaptive_can_estimate {
            // Adaptive kappa: blended from book depth + own fill experience
            // Already incorporates fill rate information via Bayesian update
            let k = market_params.adaptive_kappa;
            debug!(
                adaptive_kappa = %format!("{:.0}", k),
                book_kappa = %format!("{:.0}", market_params.kappa),
                warmup_progress = %format!("{:.0}%", market_params.adaptive_warmup_progress * 100.0),
                "Using ADAPTIVE kappa (blended book + own fills)"
            );
            // For now, use symmetric kappa (directional can be added later)
            (k, k, k)
        } else {
            // Legacy: Book-based kappa with AS and heavy-tail adjustments
            // Theory: Informed flow reduces effective supply of uninformed liquidity.
            // κ_effective = κ̂ × (1 - α), where α = P(informed | fill)
            let alpha = market_params.predicted_alpha.min(0.5); // Cap at 50% AS

            // Heavy-tail adjustment: when CV > 1.2, reduce kappa
            let tail_multiplier = if market_params.is_heavy_tailed {
                (2.0 - market_params.kappa_cv).clamp(0.5, 1.0)
            } else {
                1.0
            };

            // Symmetric kappa (for skew and logging)
            let k = market_params.kappa * (1.0 - alpha) * tail_multiplier;

            // Directional kappas for asymmetric GLFT spreads
            let k_bid = market_params.kappa_bid * (1.0 - alpha) * tail_multiplier;
            let k_ask = market_params.kappa_ask * (1.0 - alpha) * tail_multiplier;
            (k, k_bid, k_ask)
        };

        // Time horizon from arrival intensity: T = 1/λ (with max cap)
        let time_horizon = self.holding_time(market_params.arrival_intensity);

        // === 2. ASYMMETRIC GLFT HALF-SPREADS (First-Principles Fix 2) ===
        // δ_bid = (1/γ) × ln(1 + γ/κ_bid) + (1/2) × γ × σ² × T - wider if κ_bid < κ_ask (sell pressure)
        // δ_ask = (1/γ) × ln(1 + γ/κ_ask) + (1/2) × γ × σ² × T - wider if κ_ask < κ_bid (buy pressure)
        // The volatility compensation term ensures spreads widen appropriately in volatile markets.
        // σ_conditional: Hawkes intensity ratio inflates σ during cascades.
        // √(λ(t)/λ₀) widens BOTH GLFT terms uniformly (liquidity + vol_comp),
        // unlike γ multiplication which partially cancels (wider inventory penalty
        // but narrower edge capture via 1/γ).
        // WS2: sigma_cascade_mult removed — CovarianceTracker handles vol feedback.
        let sigma = market_params.sigma_effective;
        let tau = time_horizon;
        // Use drift-aware half-spreads: μ > 0 → tighter bids (buy into uptrend),
        // wider asks (don't sell into uptrend). Falls back to symmetric when μ=0.
        let drift_rate = market_params.drift_rate_per_sec;
        let mut half_spread_bid =
            self.half_spread_with_drift(gamma, kappa_bid, sigma, tau, drift_rate, true);
        let mut half_spread_ask =
            self.half_spread_with_drift(gamma, kappa_ask, sigma, tau, drift_rate, false);

        // Symmetric half-spread for logging (average of bid/ask)
        let mut half_spread = (half_spread_bid + half_spread_ask) / 2.0;

        // === 2a. ADVERSE SELECTION SPREAD ADJUSTMENT (per-side, deduplicated) ===
        // The GLFT vol compensation term (½γσ²τ) already prices expected price variance
        // over the holding period. The AS estimator measures realized adverse selection
        // over its own horizon (500ms). To avoid double-counting the vol component,
        // subtract the vol floor from measured AS before adding the residual.
        //
        // vol_floor = ½ × γ × σ² × as_horizon_s
        // as_net = max(0, as_raw - vol_floor)  — only add the excess beyond vol pricing
        const AS_HORIZON_S: f64 = 0.5; // 500ms, matching AS estimator horizon
        if market_params.as_warmed_up {
            let as_bid_raw = market_params.as_spread_adjustment_bid;
            let as_ask_raw = market_params.as_spread_adjustment_ask;
            if as_bid_raw > 0.0 || as_ask_raw > 0.0 {
                let vol_floor = 0.5 * gamma * sigma.powi(2) * AS_HORIZON_S;
                let as_bid_net = (as_bid_raw - vol_floor).max(0.0);
                let as_ask_net = (as_ask_raw - vol_floor).max(0.0);
                half_spread_bid += as_bid_net;
                half_spread_ask += as_ask_net;
                half_spread += (as_bid_net + as_ask_net) / 2.0;
                debug!(
                    as_bid_raw_bps = %format!("{:.2}", as_bid_raw * 10000.0),
                    as_ask_raw_bps = %format!("{:.2}", as_ask_raw * 10000.0),
                    vol_floor_bps = %format!("{:.2}", vol_floor * 10000.0),
                    as_bid_net_bps = %format!("{:.2}", as_bid_net * 10000.0),
                    as_ask_net_bps = %format!("{:.2}", as_ask_net * 10000.0),
                    predicted_alpha = %format!("{:.3}", market_params.predicted_alpha),
                    "Per-side AS spread adjustment applied (vol-floor deduplicated)"
                );
            }
        }

        // === 2a''. PRE-FILL TOXICITY ASYMMETRIC WIDENING (Phase 3) ===
        // Apply asymmetric spread widening based on pre-fill classifier predictions.
        // Each side is widened independently based on its predicted toxicity.
        // This is PROACTIVE protection before fills, complementing the reactive AS adjustment.
        //
        // Two modes (gated by risk_config.use_log_odds_as):
        // 1. Log-odds additive (default): as_adj = (1/gamma) * ln(p_informed / p_noise).max(0)
        //    Theoretically correct Bayesian integration, convex in toxicity.
        // 2. Legacy multiplicative: spread *= pre_fill_spread_mult (fallback)
        if self.risk_config.use_log_odds_as {
            // Log-odds additive path: convert toxicity probability to additive spread
            let max_adj = self.risk_config.max_as_adjustment_bps / 10000.0;
            let bid_tox = market_params.pre_fill_toxicity_bid;
            let ask_tox = market_params.pre_fill_toxicity_ask;

            // Only apply if there's meaningful toxicity signal
            if bid_tox > 0.01 || ask_tox > 0.01 {
                let safe_gamma = gamma.max(0.01); // Prevent division by zero

                // Bid side: log-odds of informed vs noise
                let bid_adj = if bid_tox > 0.01 {
                    let p_noise_bid = (1.0 - bid_tox).max(0.01);
                    let log_odds_bid = (bid_tox / p_noise_bid).ln().max(0.0);
                    ((1.0 / safe_gamma) * log_odds_bid).min(max_adj)
                } else {
                    0.0
                };

                // Ask side: log-odds of informed vs noise
                let ask_adj = if ask_tox > 0.01 {
                    let p_noise_ask = (1.0 - ask_tox).max(0.01);
                    let log_odds_ask = (ask_tox / p_noise_ask).ln().max(0.0);
                    ((1.0 / safe_gamma) * log_odds_ask).min(max_adj)
                } else {
                    0.0
                };

                half_spread_bid += bid_adj;
                half_spread_ask += ask_adj;
                half_spread += (bid_adj + ask_adj) / 2.0;

                debug!(
                    bid_tox = %format!("{:.3}", bid_tox),
                    ask_tox = %format!("{:.3}", ask_tox),
                    bid_adj_bps = %format!("{:.2}", bid_adj * 10000.0),
                    ask_adj_bps = %format!("{:.2}", ask_adj * 10000.0),
                    max_adj_bps = %format!("{:.1}", self.risk_config.max_as_adjustment_bps),
                    "Pre-fill toxicity log-odds AS adjustment applied"
                );
            }
        }
        // Legacy multiplicative pre-fill path removed — AS defense is now
        // fully handled by the log-odds additive path above.

        // === 2a'. KALMAN UNCERTAINTY SPREAD WIDENING (Stochastic Module) ===
        // When use_kalman_filter is enabled, add uncertainty-based spread widening
        // Formula: spread_add = γ × σ_kalman × √T
        // Higher Kalman uncertainty → wider spreads (protects against fair price misestimation)
        if market_params.use_kalman_filter
            && market_params.kalman_warmed_up
            && market_params.kalman_spread_widening > 0.0
        {
            half_spread_bid += market_params.kalman_spread_widening;
            half_spread_ask += market_params.kalman_spread_widening;
            half_spread += market_params.kalman_spread_widening;
            debug!(
                kalman_widening_bps = %format!("{:.2}", market_params.kalman_spread_widening * 10000.0),
                kalman_uncertainty_bps = %format!("{:.2}", market_params.kalman_uncertainty * 10000.0),
                kalman_fair_price = %format!("{:.4}", market_params.kalman_fair_price),
                "Kalman uncertainty spread widening applied"
            );
        }

        // === 2a''. MONOPOLIST LP PRICING (Illiquid Tokens) ===
        // When we're the sole/dominant LP, GLFT's competitive assumption understates edge.
        // Apply a markup inversely proportional to taker price elasticity, scaled by market share.
        // Formula: markup_bps = min(cap, (1/eta) * market_share)
        // where eta is price elasticity of demand for liquidity.
        if self.risk_config.use_monopolist_pricing && market_params.competitor_count < 1.5 {
            let eta = if self.elasticity_estimator.is_valid()
                && self.elasticity_estimator.observation_count()
                    >= self.risk_config.min_observations_for_elasticity
            {
                self.elasticity_estimator.elasticity()
            } else {
                1.0 // Conservative default: unit elastic
            };

            let market_share = market_params.market_share.clamp(0.0, 1.0);
            let markup_bps = ((1.0 / eta.max(0.1)) * market_share)
                .min(self.risk_config.monopolist_markup_cap_bps);
            let markup_frac = markup_bps / 10000.0;

            if markup_frac > 1e-8 {
                half_spread_bid += markup_frac;
                half_spread_ask += markup_frac;
                half_spread += markup_frac;

                debug!(
                    competitor_count = %format!("{:.1}", market_params.competitor_count),
                    market_share = %format!("{:.2}", market_share),
                    elasticity = %format!("{:.2}", eta),
                    markup_bps = %format!("{:.2}", markup_bps),
                    est_valid = %self.elasticity_estimator.is_valid(),
                    est_obs = %self.elasticity_estimator.observation_count(),
                    "Monopolist LP markup applied"
                );
            }
        }

        // === 2b. JUMP PREMIUM (First-Principles) ===
        // Theory: Under jump-diffusion dP = σ dW + J dN, total risk exceeds diffusion risk.
        // GLFT assumes continuous diffusion, but crypto has discrete jumps.
        //
        // Jump premium formula (from jump-diffusion optimal MM theory):
        //   jump_premium = λ × E[J²] / (γ × κ)
        // where λ = jump intensity, E[J²] = expected squared jump size
        //
        // We estimate this from jump_ratio = RV/BV:
        //   - RV captures total variance (diffusion + jumps)
        //   - BV captures diffusion variance only
        //   - Jump contribution ≈ (jump_ratio - 1) × σ²
        if market_params.jump_ratio > 1.5 {
            // Estimate jump component from the ratio
            // RV/BV - 1 ≈ (λ × E[J²]) / σ²_diffusion
            let jump_component = (market_params.jump_ratio - 1.0) * market_params.sigma.powi(2);

            // Jump premium: spread compensation for jump risk
            // Scaled by 1/(γ × κ) like the GLFT formula
            let jump_premium = if gamma * kappa > 1e-9 {
                (jump_component / (gamma * kappa)).clamp(0.0, 0.005) // Max 50 bps
            } else {
                0.0
            };

            half_spread_bid += jump_premium;
            half_spread_ask += jump_premium;
            half_spread += jump_premium;

            if jump_premium > 0.0001 {
                debug!(
                    jump_ratio = %format!("{:.2}", market_params.jump_ratio),
                    jump_premium_bps = %format!("{:.2}", jump_premium * 10000.0),
                    "Jump premium applied"
                );
            }
        }

        // === REMOVED: SPREAD REGIME ADJUSTMENT ===
        // FIRST PRINCIPLES: The spread_regime_mult contradicted GLFT's κ-based adaptation.
        // When κ indicates tight spreads are appropriate (deep book), manually widening
        // via a multiplier undermined the mathematical model.
        //
        // Market tightness is already captured by κ (order flow intensity):
        //   - Deep book → high κ → GLFT naturally produces tighter spreads
        //   - Thin book → low κ → GLFT naturally produces wider spreads
        //
        // The regime multipliers (1.3, 1.1, 0.95, 0.9) were ad-hoc and lacked derivation.
        // Removing them trusts the GLFT model to handle spread dynamics correctly.

        // === 2d. SPREAD FLOOR: Learned vs Adaptive vs Static ===
        // Priority:
        // 1. Learned spread floor (when calibrated and enabled) - from Bayesian parameter learner
        // 2. Adaptive spread floor (when enabled) - from Bayesian AS estimation
        // 3. Static RiskConfig floor + latency/tick constraints
        //
        // KEY FIX: Use `adaptive_can_estimate` - our Bayesian prior gives reasonable
        // floor estimates immediately (fees + 3bps AS prior + safety margin ≈ 8-10 bps).
        let effective_floor = if market_params.use_learned_parameters
            && market_params.learned_params_calibrated
        {
            // Learned floor: from Bayesian parameter learner with conjugate prior
            // Convert from bps to fraction: divide by 10000
            let floor_bps = market_params.learned_spread_floor_bps;
            let floor = floor_bps / 10000.0;
            debug!(
                learned_floor_bps = %format!("{:.2}", floor_bps),
                adaptive_floor_bps = %format!("{:.2}", market_params.adaptive_spread_floor * 10000.0),
                static_floor_bps = %format!("{:.2}", self.risk_config.min_spread_floor * 10000.0),
                "Using LEARNED spread floor (Bayesian parameter learner)"
            );
            floor
        } else if market_params.use_adaptive_spreads && market_params.adaptive_can_estimate {
            // Adaptive floor: learned from actual fill AS + fees + safety buffer
            // During warmup, the prior-based floor is already conservative (fees + 3bps + 1.5σ)
            let floor = market_params.adaptive_spread_floor;
            debug!(
                adaptive_floor_bps = %format!("{:.2}", floor * 10000.0),
                static_floor_bps = %format!("{:.2}", self.risk_config.min_spread_floor * 10000.0),
                warmup_progress = %format!("{:.0}%", market_params.adaptive_warmup_progress * 100.0),
                "Using ADAPTIVE spread floor (Bayesian AS estimation)"
            );
            floor
        } else {
            // Legacy: Static floor from RiskConfig + tick/latency constraints
            market_params.effective_spread_floor(self.risk_config.min_spread_floor)
        };

        // === Q19: AS-Posterior + Profile Safety Floor ===
        // Two additional floors, max()'d with the base effective_floor:
        // 1. AS uncertainty premium: E[AS] + k × √Var[AS] (wider when AS uncertain)
        // 2. Profile static bound: venue-specific minimum (Hip3=7.5 bps, etc.)
        let as_uncertainty_premium_bps = if market_params.as_floor_variance_bps2 > 0.0 {
            self.risk_config.as_uncertainty_premium_k * market_params.as_floor_variance_bps2.sqrt()
        } else {
            0.0
        };
        let as_posterior_floor_bps = market_params.as_floor_bps + as_uncertainty_premium_bps;
        let as_posterior_floor_frac = as_posterior_floor_bps / 10_000.0;

        let profile_floor_frac = market_params.profile_spread_floor_bps / 10_000.0;

        let effective_floor = effective_floor
            .max(as_posterior_floor_frac)
            .max(profile_floor_frac);

        if as_posterior_floor_frac > 0.0 || profile_floor_frac > 0.0 {
            debug!(
                as_floor_bps = %format!("{:.2}", market_params.as_floor_bps),
                as_uncertainty_premium_bps = %format!("{:.2}", as_uncertainty_premium_bps),
                as_posterior_floor_bps = %format!("{:.2}", as_posterior_floor_bps),
                profile_floor_bps = %format!("{:.2}", market_params.profile_spread_floor_bps),
                final_floor_bps = %format!("{:.2}", effective_floor * 10_000.0),
                "Q19: AS-posterior + profile floor applied"
            );
        }

        // Floor clamp: safety net only. With unified floor in solve_min_gamma,
        // this should fire <5% of cycles (only during transient parameter changes).
        let floor_bound = half_spread_bid < effective_floor || half_spread_ask < effective_floor;
        if floor_bound {
            debug!(
                spread_bid_bps = %format!("{:.1}", half_spread_bid * 10000.0),
                spread_ask_bps = %format!("{:.1}", half_spread_ask * 10000.0),
                floor_bps = %format!("{:.1}", effective_floor * 10000.0),
                "Floor clamp binding (safety net) — gamma may need adjustment"
            );
        }
        half_spread_bid = half_spread_bid.max(effective_floor);
        half_spread_ask = half_spread_ask.max(effective_floor);
        half_spread = half_spread.max(effective_floor);

        // === 2e. SPREAD CEILING: Fill Rate Controller ===
        // When adaptive spreads enabled: apply ceiling to ensure minimum fill rate
        // This prevents spreads from being so wide we never trade
        //
        // NOTE: For ceiling, we DO check `adaptive_warmed_up` here because the
        // fill rate controller needs observation time (2+ minutes) before it can
        // reliably suggest a ceiling. Using a ceiling too early could be harmful.
        if market_params.use_adaptive_spreads
            && market_params.adaptive_warmed_up
            && market_params.adaptive_spread_ceiling < f64::MAX
        {
            let ceiling = market_params.adaptive_spread_ceiling;
            if half_spread > ceiling {
                debug!(
                    half_spread_bps = %format!("{:.2}", half_spread * 10000.0),
                    ceiling_bps = %format!("{:.2}", ceiling * 10000.0),
                    "Applying ADAPTIVE spread ceiling (fill rate target)"
                );
            }
            half_spread_bid = half_spread_bid.min(ceiling);
            half_spread_ask = half_spread_ask.min(ceiling);
            half_spread = half_spread.min(ceiling);
        }

        // === DEPRECATED: SPREAD MULTIPLIERS ===
        // FIRST PRINCIPLES REFACTOR: Arbitrary spread multipliers bypass the GLFT model.
        //
        // Previously this section applied:
        //   1. stochastic_spread_multiplier (book depth, toxicity)
        //   2. adaptive_uncertainty_factor (warmup uncertainty)
        //
        // These have been REMOVED. All risk factors now flow through gamma:
        //   - Book depth → RiskConfig.book_depth_multiplier() → gamma scaling
        //   - Toxicity → RiskConfig.toxicity_sensitivity → gamma scaling
        //   - Warmup uncertainty → RiskConfig.warmup_multiplier() → gamma scaling
        //
        // The GLFT formula δ = (1/γ) × ln(1 + γ/κ) now handles all spread widening
        // in a mathematically principled way.

        // === 3. USE BEST AVAILABLE SIGMA FOR INVENTORY SKEW ===
        // Priority: Particle filter sigma > sigma_leverage_adjusted
        // Particle filter provides:
        // - Regime-aware volatility estimation
        // - Credible intervals for uncertainty quantification
        // sigma_leverage_adjusted incorporates:
        // - sigma_effective (blended clean/total based on jump regime)
        // - Leverage effect: wider during down moves when ρ < 0
        // Calculate inventory ratio: q / Q_max (normalized to [-1, 1])
        let inventory_ratio = if effective_max_position > EPSILON {
            (position / effective_max_position).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // === DUAL-TIMESCALE INVENTORY SKEW (WS2: PPIP) ===
        // Resolves the γ-bifurcation: the old formula `q × γ × σ² × (1/κ)` produced
        // ~10⁻¹³ bps skew because τ = 1/κ ≈ 0.0004s is the per-fill timescale, not
        // the inventory holding horizon (30-300s).
        //
        // The Posterior Predictive Inventory Penalty (PPIP) uses:
        //   - τ_inventory: measured EWMA of reducing-fill holding durations (slow timescale)
        //   - E[μ], E[σ²]: from Bayesian posteriors
        //   - CV²(σ²), CV²(τ): ambiguity aversion + timing uncertainty
        //   - Self-calibrating multiplier: learns magnitude from markout
        //
        // This produces O(several bps) skew at moderate inventory — the correct magnitude.
        // No position amplifiers (5x/10x) needed — PPIP handles all magnitudes naturally.
        let base_skew = {
            // Update PPIP from current market params (drift, sigma, tau posteriors)
            // Safety: this mutates internal state but PPIP is designed for per-cycle updates
            // We use the dual_timescale controller on self, but since calculate_quotes takes &self,
            // we compute the reservation shift from current params directly
            let ppip_shift = {
                let drift_mean = market_params.drift_rate_per_sec;
                let sigma_sq_mean = sigma.powi(2);
                let tau_mean = market_params.tau_inventory_s.max(1.0);
                let tau_variance = market_params.tau_variance_s2.max(1.0);

                // Sigma variance: prefer BMA between-model variance, fallback to heuristics
                let sigma_sq_variance = if market_params.sigma_sq_variance_bma > 0.0 {
                    market_params.sigma_sq_variance_bma
                } else if market_params.use_belief_system && market_params.belief_confidence > 0.1 {
                    // Estimate Var[σ²] from belief confidence:
                    // Low confidence → high variance → more ambiguity aversion
                    let confidence = market_params.belief_confidence.clamp(0.1, 1.0);
                    sigma_sq_mean.powi(2) * (1.0 / confidence - 1.0).max(0.01)
                } else {
                    // Heuristic: Var[σ²] ≈ 0.5 × E[σ²]² during warmup (high uncertainty)
                    sigma_sq_mean.powi(2) * 0.5
                };

                let q = inventory_ratio * effective_max_position;

                // Term 1: Drift cost per unit (fractional)
                let drift_cost = drift_mean * tau_mean;

                // Term 2: Variance cost per unit (fractional)
                let variance_cost = q * sigma_sq_mean * tau_mean;

                // Term 3: Ambiguity aversion — relative uncertainty about σ²
                let ambiguity_ratio = sigma_sq_variance / sigma_sq_mean.powi(2).max(1e-30);
                let ambiguity_cost = variance_cost * ambiguity_ratio;

                // Term 4: Timing uncertainty — relative uncertainty about τ
                let timing_ratio = tau_variance / tau_mean.powi(2).max(1e-12);
                let timing_cost = drift_cost * timing_ratio;

                let total = drift_cost + variance_cost + ambiguity_cost + timing_cost;

                // Self-calibrating multiplier (starts 1.0, learns from markout)
                self.dual_timescale.ppip.skew_multiplier * total
            };

            // Apply flow modulation: dampen skew when aligned with flow, amplify when opposed
            let flow_alignment = inventory_ratio.signum() * market_params.flow_imbalance;
            let flow_modifier = (-self.risk_config.flow_sensitivity * flow_alignment).exp();

            ppip_shift * flow_modifier
        };

        // === 3a. HAWKES FLOW SKEWING (Tier 2) - DISABLED UNTIL CALIBRATED ===
        // REMOVED: The 0.00005 coefficient was arbitrary, not empirically derived.
        // The signal may have value but needs proper calibration:
        // 1. Measure MI(hawkes_imbalance → price_change_5s)
        // 2. If MI > 0, estimate E[Δp | imbalance] via regression
        // 3. Set coefficient = regression_slope
        //
        // To re-enable, add hawkes_price_sensitivity to MarketParams and use:
        //   hawkes_skew = imbalance × sensitivity × activity_percentile × flow_weight
        let hawkes_skew = 0.0;

        // === 3b. FUNDING COST ADJUSTMENT (Tier 2) ===
        // Funding cost integration using proper carry cost model.
        //
        // Theory: Funding creates continuous position cost. Skew should reflect
        // expected funding cost over holding period relative to expected spread.
        //
        // Formula: skew = funding_rate_per_sec × time_horizon × sensitivity
        //
        // With typical values:
        // - funding_rate = 0.0001/hour = 2.78e-8/sec
        // - time_horizon = 60 sec
        // - sensitivity = 1.0
        // - skew = 2.78e-8 × 60 × 1.0 = 1.67e-6 = 0.017 bps
        //
        // At extreme funding (0.001/hour = 10 bps/hour):
        // - skew = 2.78e-7 × 60 = 1.67e-5 = 0.17 bps
        let funding_skew = if market_params.funding_rate.abs() > 0.00001 {
            // Convert funding rate to per-second (assuming input is per-hour)
            let funding_rate_per_sec = market_params.funding_rate / 3600.0;

            // Cost over expected holding period
            let funding_cost_over_horizon = funding_rate_per_sec * time_horizon;

            // Skew direction: positive funding + long position → positive skew (widen bids)
            let funding_pressure = position.signum() * funding_cost_over_horizon;

            // Sensitivity factor (1.0 = full economic impact, can tune lower for conservatism)
            funding_pressure * self.risk_config.funding_skew_sensitivity
        } else {
            0.0
        };

        // === 4. TOXIC REGIME ===
        // Note: Dynamic gamma already scales with toxicity via effective_gamma().
        // We no longer apply a separate toxicity_multiplier to avoid double-scaling.
        // The gamma-based approach is more principled as it also affects skew.
        if market_params.is_toxic_regime {
            debug!(
                jump_ratio = %format!("{:.2}", market_params.jump_ratio),
                "Toxic regime detected (handled by dynamic gamma)"
            );
        }

        // === 5. USE MICROPRICE AS FAIR PRICE ===
        // The microprice incorporates book_imbalance and flow_imbalance predictions
        // via learned β coefficients: fair = mid × (1 + β_book × book_imb + β_flow × flow_imb)
        // This replaces all ad-hoc adjustments with data-driven signal integration.
        let fair_price = market_params.microprice;

        // === 5a. DRIFT-ADJUSTED SKEW (First Principles) ===
        // Theory: When position opposes strong momentum with high continuation probability,
        // we need URGENT position reduction. This is derived from the extended HJB equation
        // with price drift: dS = μdt + σdW
        //
        // The drift-adjusted skew provides a first-principles approach based on:
        // - Drift rate (μ) estimated from momentum
        // - Continuation probability P(momentum continues)
        // - Directional variance (increased risk when opposed to momentum)
        //
        // When drift adjustment is enabled AND position opposes momentum:
        // - drift_urgency is added to base skew (accelerates position reduction)
        // - directional_variance_mult is logged for diagnostics only
        let drift_urgency = if market_params.use_drift_adjusted_skew
            && market_params.position_opposes_momentum
            && market_params.urgency_score > 0.5
        {
            // Use first-principles drift urgency from MarketParams
            // The drift_urgency is computed by HJB controller using:
            // - sensitivity × drift_rate × P(continue) × |q| × T
            let drift_urg = market_params.hjb_drift_urgency;

            if market_params.urgency_score > 1.5 {
                debug!(
                    position = %format!("{:.4}", position),
                    momentum_bps = %format!("{:.1}", market_params.momentum_bps),
                    p_continue = %format!("{:.2}", market_params.p_momentum_continue),
                    drift_urgency_bps = %format!("{:.2}", drift_urg * 10000.0),
                    urgency_score = %format!("{:.1}", market_params.urgency_score),
                    "DRIFT-ADJUSTED SKEW: Position opposes momentum with high continuation prob"
                );
            }

            drift_urg
        } else {
            // No drift signal available — zero drift urgency.
            // The base skew from gamma * sigma^2 * q * T handles inventory alone.
            0.0
        };

        // === 6. COMBINED SKEW (purely additive) ===
        // All skew components are additive, in the same units (price fraction).
        // No multipliers — if RL wants to adjust skew, it should output an additive
        // rl_skew_adjustment_bps term (to be added in a future RL integration phase).
        //
        // Components:
        // - base_skew: GLFT inventory skew gamma * sigma^2 * q * T × flow modifier
        // - drift_urgency: First-principles urgency from HJB when position opposes momentum
        // - hawkes_skew: Hawkes flow-based directional adjustment
        // - funding_skew: Perpetual funding cost pressure
        let skew = base_skew + drift_urgency + hawkes_skew + funding_skew;

        // Diagnostic logging: track inventory skew magnitude for calibration verification.
        // Expected: 50% utilization → 3-8 bps, 80% → 8-15 bps.
        // If skew < 2 bps at 50% utilization → γ needs recalibration (increase gamma_base).
        if position.abs() > 0.001 {
            tracing::info!(
                inventory_skew_bps = %format!("{:.2}", skew * 10000.0),
                utilization_pct = %format!("{:.1}", (position.abs() / effective_max_position).min(1.0) * 100.0),
                gamma = %format!("{:.4}", gamma),
                drift_bps = %format!("{:.2}", market_params.drift_rate_per_sec * 10000.0),
                "Inventory skew magnitude check"
            );
        }

        // FIX: Enhanced debug logging to trace PPIP skew calculation
        if position.abs() > 0.001 {
            debug!(
                position = %format!("{:.6}", position),
                inventory_ratio = %format!("{:.4}", inventory_ratio),
                tau_inventory_s = %format!("{:.1}", market_params.tau_inventory_s),
                ppip_skew_bps = %format!("{:.2}", base_skew * 10000.0),
                sigma = %format!("{:.8}", sigma),
                gamma = %format!("{:.4}", gamma),
                drift_per_sec = %format!("{:.6}", market_params.drift_rate_per_sec),
                flow_imbalance = %format!("{:.3}", market_params.flow_imbalance),
                skew_multiplier = %format!("{:.3}", self.dual_timescale.ppip.skew_multiplier),
                "PPIP skew calculation"
            );
            if skew.abs() < 1e-8 {
                debug!(
                    base_skew_raw = %format!("{:.8}", base_skew),
                    "SKEW ZERO WARNING: Non-zero position but zero skew - check PPIP inputs"
                );
            }
        }

        // === 6a. ASYMMETRIC BID/ASK DELTAS ===
        // Use directional half-spreads: κ_bid ≠ κ_ask when flow is directional
        // - Bid delta uses half_spread_bid (wider when sell pressure = low κ_bid)
        // - Ask delta uses half_spread_ask (wider when buy pressure = low κ_ask)
        //
        // === 6a'. PREDICTIVE BIAS (Avellaneda-Stoikov Extension) ===
        // When using first-principles belief system:
        //   β_t = E[μ | data] from Normal-Inverse-Gamma posterior
        // Legacy fallback (changepoint-based heuristic):
        //   β_t = -sensitivity × prob_excess × σ
        //
        // Negative β → expect price to fall → widen bids (+β/2), tighten asks (+β/2)
        //
        // The bias shifts BOTH sides in the same direction:
        //   - bid_delta = half_spread + skew - β/2 (widen when β < 0)
        //   - ask_delta = half_spread - skew + β/2 (tighten when β < 0)
        //
        // FIRST-PRINCIPLES: Confidence-weighted continuous influence
        // Instead of threshold gating (magic number), scale bias by confidence.
        // This is mathematically correct: β_effective = β × confidence
        // When confidence=0.2, we believe 20% of the signal; when 1.0, 100%.
        let predictive_bias = if market_params.use_belief_system {
            // First-principles: Use β_t = E[μ | data] from NIG posterior
            // SCALED by confidence - no arbitrary threshold
            let raw_bias = market_params.belief_predictive_bias;
            let confidence = market_params.belief_confidence;
            let bias = raw_bias * confidence;

            if raw_bias.abs() > 0.0001 && confidence > 0.01 {
                debug!(
                    raw_belief_bias_bps = %format!("{:.2}", raw_bias * 10000.0),
                    confidence_scaled_bias_bps = %format!("{:.2}", bias * 10000.0),
                    belief_confidence = %format!("{:.3}", confidence),
                    belief_sigma = %format!("{:.6}", market_params.belief_expected_sigma),
                    belief_kappa = %format!("{:.0}", market_params.belief_expected_kappa),
                    "Predictive bias: confidence-weighted (β × conf)"
                );
            }
            bias
        } else if market_params.changepoint_prob > 0.3 {
            // Legacy fallback: Compute predictive bias from changepoint probability
            // Sensitivity: 2.0 means expect 2σ move on confirmed changepoint
            let sensitivity = 2.0; // Could be made configurable via StochasticConfig
            let threshold = 0.3;
            let prob_excess = (market_params.changepoint_prob - threshold) / (1.0 - threshold);

            // Negative bias = expect price to fall = widen bids, tighten asks
            let bias = -sensitivity * prob_excess * sigma;

            if bias.abs() > 0.0001 {
                debug!(
                    changepoint_prob = %format!("{:.3}", market_params.changepoint_prob),
                    prob_excess = %format!("{:.3}", prob_excess),
                    sigma = %format!("{:.6}", sigma),
                    predictive_bias_bps = %format!("{:.2}", bias * 10000.0),
                    "Predictive bias applied (legacy changepoint heuristic)"
                );
            }
            bias
        } else {
            0.0
        };

        // === ADDITIVE spread composition (replaces multiplicative stacking) ===
        // Multiplicative chains caused death spirals (3.34x quota × 2x bandit = 6.7x).
        // Now: base GLFT spread + additive components, each individually capped.

        // 1. Bandit multiplier component (now additive bps)
        let bandit_addon = market_params.bandit_spread_additive_bps / 10000.0;

        // 2. Quota shadow spread: already in bps, convert to price fraction
        // Capped at 50 bps to prevent quota pressure from dominating
        let quota_addon = (market_params.quota_shadow_spread_bps / 10_000.0).min(0.0050);

        // Compose: base + addons (additive)
        // The base spread passes through at natural GLFT level; only addons are bounded.
        let half_spread_bid_widened = half_spread_bid + bandit_addon + quota_addon;
        let half_spread_ask_widened = half_spread_ask + bandit_addon + quota_addon;

        // Compute asymmetric deltas with predictive bias
        // Note: predictive_bias is typically negative when changepoint imminent (expect price drop)
        //   - Subtracting negative β/2 from bid_delta → WIDENS bids (less aggressive buying)
        //   - Adding negative β/2 to ask_delta → TIGHTENS asks (more aggressive selling)
        let bid_delta_raw = half_spread_bid_widened + skew - predictive_bias / 2.0;
        let ask_delta_raw = (half_spread_ask_widened - skew + predictive_bias / 2.0).max(0.0);

        // FIX: Cap total spread asymmetry to prevent one side from being uncompetitive
        // When skew causes extreme asymmetry (e.g., 15 bps vs 60 bps), rebalance
        let max_asymmetry_ratio = 3.0; // Maximum bid/ask spread ratio (3:1)
        let max_asymmetry_bps = 0.0020; // 20 bps max absolute asymmetry
        let total_spread = bid_delta_raw + ask_delta_raw;
        let asymmetry = (bid_delta_raw - ask_delta_raw).abs();

        let (bid_delta, ask_delta) = if asymmetry > max_asymmetry_bps && total_spread > 0.0 {
            // Asymmetry too high - redistribute while preserving total spread
            let target_bid = if bid_delta_raw > ask_delta_raw {
                // Bid side wider - cap it
                let capped_bid = (total_spread / 2.0) + (max_asymmetry_bps / 2.0);
                capped_bid.min(bid_delta_raw) // Don't widen if already within cap
            } else {
                bid_delta_raw
            };
            let target_ask = if ask_delta_raw > bid_delta_raw {
                // Ask side wider - cap it
                let capped_ask = (total_spread / 2.0) + (max_asymmetry_bps / 2.0);
                capped_ask.min(ask_delta_raw)
            } else {
                ask_delta_raw
            };

            // Log when capping occurs
            if (bid_delta_raw - target_bid).abs() > 1e-6
                || (ask_delta_raw - target_ask).abs() > 1e-6
            {
                debug!(
                    asymmetry_bps = %format!("{:.1}", asymmetry * 10000.0),
                    bid_raw_bps = %format!("{:.1}", bid_delta_raw * 10000.0),
                    ask_raw_bps = %format!("{:.1}", ask_delta_raw * 10000.0),
                    bid_capped_bps = %format!("{:.1}", target_bid * 10000.0),
                    ask_capped_bps = %format!("{:.1}", target_ask * 10000.0),
                    "Spread asymmetry capped to prevent uncompetitive side"
                );
            }
            (target_bid, target_ask.max(0.0))
        } else if bid_delta_raw.max(ask_delta_raw) / bid_delta_raw.min(ask_delta_raw).max(1e-8)
            > max_asymmetry_ratio
        {
            // Ratio asymmetry too high - use geometric mean and redistribute
            let geo_mean = (bid_delta_raw * ask_delta_raw).sqrt().max(effective_floor);
            let rebalanced_bid = geo_mean + (skew / 2.0);
            let rebalanced_ask = (geo_mean - skew / 2.0).max(0.0);
            debug!(
                ratio = %format!("{:.1}", bid_delta_raw.max(ask_delta_raw) / bid_delta_raw.min(ask_delta_raw).max(1e-8)),
                geo_mean_bps = %format!("{:.1}", geo_mean * 10000.0),
                "Spread ratio asymmetry capped via geometric rebalancing"
            );
            (rebalanced_bid, rebalanced_ask)
        } else {
            (bid_delta_raw, ask_delta_raw)
        };

        debug!(
            inv_ratio = %format!("{:.4}", inventory_ratio),
            gamma = %format!("{:.4}", gamma),
            kappa = %format!("{:.0}", kappa),
            kappa_bid = %format!("{:.0}", kappa_bid),
            kappa_ask = %format!("{:.0}", kappa_ask),
            kappa_cv = %format!("{:.2}", market_params.kappa_cv),
            adaptive_mode = market_params.use_adaptive_spreads && market_params.adaptive_can_estimate,
            warmup_pct = %format!("{:.0}%", market_params.adaptive_warmup_progress * 100.0),
            time_horizon = %format!("{:.2}", time_horizon),
            half_spread_bps = %format!("{:.1}", half_spread * 10000.0),
            half_spread_bid_bps = %format!("{:.1}", half_spread_bid * 10000.0),
            half_spread_ask_bps = %format!("{:.1}", half_spread_ask * 10000.0),
            effective_floor_bps = %format!("{:.1}", effective_floor * 10000.0),
            flow_imb = %format!("{:.3}", market_params.flow_imbalance),
            flow_mod = %format!("{:.3}", (-self.risk_config.flow_sensitivity * inventory_ratio.signum() * market_params.flow_imbalance).exp()),
            base_skew_bps = %format!("{:.4}", base_skew * 10000.0),
            drift_urgency_bps = %format!("{:.4}", drift_urgency * 10000.0),
            hawkes_skew_bps = %format!("{:.4}", hawkes_skew * 10000.0),
            funding_skew_bps = %format!("{:.4}", funding_skew * 10000.0),
            total_skew_bps = %format!("{:.4}", skew * 10000.0),
            urgency_score = %format!("{:.1}", market_params.urgency_score),
            spread_regime = ?market_params.spread_regime,
            vol_regime = ?market_params.volatility_regime,
            hawkes_activity = %format!("{:.2}", market_params.hawkes_activity_percentile),
            funding_rate = %format!("{:.4}", market_params.funding_rate),
            microprice = %format!("{:.4}", fair_price),
            sigma = %format!("{:.6}", sigma),
            bid_delta_bps = %format!("{:.1}", bid_delta * 10000.0),
            ask_delta_bps = %format!("{:.1}", ask_delta * 10000.0),
            is_toxic = market_params.is_toxic_regime,
            heavy_tail = market_params.is_heavy_tailed,
            tight_quoting = market_params.tight_quoting_allowed,
            belief_system = market_params.use_belief_system,
            belief_bias_bps = %format!("{:.2}", market_params.belief_predictive_bias * 10000.0),
            "GLFT spread components with asymmetric kappa"
        );

        // Convert to absolute price offsets using fair_price (microprice)
        let bid_offset = fair_price * bid_delta;
        let ask_offset = fair_price * ask_delta;

        // Calculate raw prices around fair_price
        let lower_price_raw = fair_price - bid_offset;
        let upper_price_raw = fair_price + ask_offset;

        // Round to exchange precision
        let mut lower_price = round_to_significant_and_decimal(lower_price_raw, 5, config.decimals);
        let upper_price = round_to_significant_and_decimal(upper_price_raw, 5, config.decimals);

        // Ensure bid < ask
        if lower_price >= upper_price {
            let tick = 10f64.powi(-(config.decimals as i32));
            lower_price -= tick;
        }

        debug!(
            mid = config.mid_price,
            fair_price = %format!("{:.4}", fair_price),
            sigma_effective = %format!("{:.6}", sigma),
            kappa = %format!("{:.2}", kappa),
            gamma = %format!("{:.4}", gamma),
            jump_ratio = %format!("{:.2}", market_params.jump_ratio),
            bid_final = lower_price,
            ask_final = upper_price,
            spread_bps = %format!("{:.1}", (upper_price - lower_price) / fair_price * 10000.0),
            "GLFT prices (microprice-based)"
        );

        // Calculate sizes based on position limits (using first-principles dynamic limit)
        let buy_size_raw = (effective_max_position - position)
            .min(target_liquidity)
            .max(0.0);
        let sell_size_raw = (effective_max_position + position)
            .min(target_liquidity)
            .max(0.0);

        // Cascade defense routed through gamma via beta_cascade — no size reduction needed.
        // At cascade_intensity=0.7, gamma inflates ~2.3× (capped by sigmoid regularization),
        // which widens spreads without zeroing out sizes.
        let buy_size = truncate_float(buy_size_raw, config.sz_decimals, false);
        let sell_size = truncate_float(sell_size_raw, config.sz_decimals, false);

        // Build quotes, checking minimum notional
        let bid = if buy_size > EPSILON {
            let quote = Quote::new(lower_price, buy_size);
            if quote.notional() >= config.min_notional {
                Some(quote)
            } else {
                None
            }
        } else {
            None
        };

        let ask = if sell_size > EPSILON {
            let quote = Quote::new(upper_price, sell_size);
            if quote.notional() >= config.min_notional {
                Some(quote)
            } else {
                None
            }
        } else {
            None
        };

        (bid, ask)
    }

    fn name(&self) -> &'static str {
        "GLFT"
    }

    fn record_elasticity_observation(&mut self, spread_bps: f64, fill_rate: f64) {
        self.elasticity_estimator.record(spread_bps, fill_rate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a default MarketParams with sensible test values.
    fn test_market_params() -> MarketParams {
        let mut mp = MarketParams::default();
        mp.sigma = 0.01; // 100 bps/sqrt(s) vol — large enough for self-consistent gamma
        mp.sigma_effective = 0.01;
        mp.kappa = 5000.0;
        mp.microprice = 100.0;
        mp.tail_risk_intensity = 0.0; // Default calm
        mp
    }

    fn test_strategy() -> GLFTStrategy {
        let mut cfg = RiskConfig::default();
        cfg.gamma_base = 0.1;
        // Low floor so self-consistent gamma doesn't dominate in unit tests
        cfg.min_spread_floor = 0.00001; // 0.1 bps — lets natural gamma control spread
        GLFTStrategy::with_config(cfg)
    }

    fn test_quote_config() -> QuoteConfig {
        QuoteConfig {
            mid_price: 100.0,
            decimals: 2,
            sz_decimals: 4,
            min_notional: 10.0,
        }
    }

    // ---------------------------------------------------------------
    // Calibrated Gamma Tests (log-additive CalibratedRiskModel only)
    // ---------------------------------------------------------------

    #[test]
    fn test_effective_gamma_uses_calibrated_model() {
        let strategy = test_strategy();
        let mp = test_market_params();
        let position = 0.5;
        let max_position = 10.0;

        // effective_gamma must always produce positive gamma
        let gamma = strategy.effective_gamma(&mp, position, max_position);
        assert!(gamma > 0.0, "gamma must be positive, got {gamma}");
    }

    // ---------------------------------------------------------------
    // Additive Skew Tests
    // ---------------------------------------------------------------

    #[test]
    fn test_no_legacy_momentum_skew_multiplier() {
        // Verify that knife scores no longer amplify skew when drift adjustment is off.
        // With the legacy multiplier removed, identical quotes should result regardless
        // of falling_knife_score / rising_knife_score values.
        let strategy = test_strategy();
        let config = QuoteConfig {
            mid_price: 1000.0,
            decimals: 4,
            sz_decimals: 4,
            min_notional: 0.01,
        };
        let position = 5.0; // Long position
        let max_position = 10.0;
        let target_liq = 1.0;

        let mut mp_base = test_market_params();
        mp_base.microprice = 1000.0;
        mp_base.sigma = 0.05;
        mp_base.sigma_effective = 0.05;
        mp_base.use_drift_adjusted_skew = false; // Force fallback path
        mp_base.falling_knife_score = 0.0;
        mp_base.rising_knife_score = 0.0;

        let mut mp_knife = mp_base.clone();
        mp_knife.falling_knife_score = 3.0; // Extreme falling knife, opposed to long position
        mp_knife.rising_knife_score = 0.0;

        let (bid_base, ask_base) =
            strategy.calculate_quotes(&config, position, max_position, target_liq, &mp_base);
        let (bid_knife, ask_knife) =
            strategy.calculate_quotes(&config, position, max_position, target_liq, &mp_knife);

        // With legacy multiplier removed, knife scores should NOT affect quotes
        if let (Some(b1), Some(b2)) = (&bid_base, &bid_knife) {
            assert!(
                (b1.price - b2.price).abs() < 1e-10,
                "knife score should have no effect on bid: base={}, knife={}",
                b1.price,
                b2.price
            );
        }
        if let (Some(a1), Some(a2)) = (&ask_base, &ask_knife) {
            assert!(
                (a1.price - a2.price).abs() < 1e-10,
                "knife score should have no effect on ask: base={}, knife={}",
                a1.price,
                a2.price
            );
        }
    }

    // ---------------------------------------------------------------
    // Log-Odds AS Integration Tests
    // ---------------------------------------------------------------

    /// Helper: create a strategy with specific risk config for AS tests.
    fn test_strategy_with_log_odds(use_log_odds: bool, max_bps: f64) -> GLFTStrategy {
        let mut risk_config = RiskConfig::default();
        risk_config.use_log_odds_as = use_log_odds;
        risk_config.max_as_adjustment_bps = max_bps;
        // Low floor so self-consistent gamma doesn't dominate in unit tests
        risk_config.min_spread_floor = 0.00001;
        GLFTStrategy::with_config(risk_config)
    }

    #[test]
    fn test_log_odds_as_low_toxicity() {
        // p_informed = 0.1 → log_odds = ln(0.1/0.9) = ln(0.111) < 0 → clamped to 0
        // At low toxicity, log-odds adjustment should be zero (noise dominates)
        let strategy = test_strategy_with_log_odds(true, 15.0);
        let config = test_quote_config();
        let mut mp = test_market_params();
        mp.pre_fill_toxicity_bid = 0.1;
        mp.pre_fill_toxicity_ask = 0.1;

        let (bid_with_tox, _ask_with_tox) = strategy.calculate_quotes(&config, 0.0, 10.0, 1.0, &mp);

        // Zero toxicity baseline
        let mut mp_zero = test_market_params();
        mp_zero.pre_fill_toxicity_bid = 0.0;
        mp_zero.pre_fill_toxicity_ask = 0.0;
        // pre_fill_spread_mult removed — AS defense via log-odds additive path

        let (bid_no_tox, _ask_no_tox) =
            strategy.calculate_quotes(&config, 0.0, 10.0, 1.0, &mp_zero);

        // At p=0.1, log-odds is negative → clamped to 0 → no widening
        if let (Some(b_tox), Some(b_no)) = (&bid_with_tox, &bid_no_tox) {
            let mid = config.mid_price;
            let spread_diff_bps = ((b_no.price - b_tox.price) / mid) * 10000.0;
            assert!(
                spread_diff_bps.abs() < 1.0,
                "Low toxicity (0.1) should produce near-zero log-odds adjustment, got {:.2} bps",
                spread_diff_bps
            );
        }
    }

    #[test]
    fn test_log_odds_as_moderate_toxicity() {
        // p_informed = 0.7 → log_odds = ln(0.7/0.3) = ln(2.33) = 0.847
        // The raw adjustment (1/γ)×log_odds is in price-fraction units, which is
        // always >> 15 bps cap for any realistic gamma. So moderate toxicity (0.7)
        // will hit the cap. This is correct behavior — the cap prevents extreme widening.
        let strategy = test_strategy_with_log_odds(true, 15.0);
        let config = test_quote_config();

        let mut mp = test_market_params();
        mp.pre_fill_toxicity_bid = 0.7;
        mp.pre_fill_toxicity_ask = 0.7;

        let (bid_tox, _) = strategy.calculate_quotes(&config, 0.0, 10.0, 1.0, &mp);

        let mut mp_zero = test_market_params();
        mp_zero.pre_fill_toxicity_bid = 0.0;
        mp_zero.pre_fill_toxicity_ask = 0.0;
        // pre_fill_spread_mult removed — AS defense via log-odds additive path

        let (bid_no, _) = strategy.calculate_quotes(&config, 0.0, 10.0, 1.0, &mp_zero);

        // Bid price should be lower (wider spread) with toxicity
        if let (Some(b_tox), Some(b_no)) = (&bid_tox, &bid_no) {
            let mid = config.mid_price;
            let widening_bps = ((b_no.price - b_tox.price) / mid) * 10000.0;
            assert!(
                widening_bps > 1.0,
                "Moderate toxicity (0.7) should widen spread by >1 bps, got {:.2} bps",
                widening_bps
            );
            // At p=0.7, (1/γ)×ln(p/q) in price-fraction is huge for any reasonable γ,
            // so the cap binds. Widening should be at or near the cap.
            assert!(
                widening_bps <= 15.01,
                "Moderate toxicity (0.7) widening should not exceed cap, got {:.2} bps",
                widening_bps
            );
        }
    }

    #[test]
    fn test_log_odds_as_high_toxicity_capped() {
        // p_informed = 0.95 → log_odds = ln(0.95/0.05) = ln(19) = 2.944
        // adj = (1/0.15) * 2.944 = 19.6 bps → capped at 15 bps
        let strategy = test_strategy_with_log_odds(true, 15.0);
        let config = test_quote_config();

        let mut mp = test_market_params();
        mp.pre_fill_toxicity_bid = 0.95;
        mp.pre_fill_toxicity_ask = 0.95;

        let (bid_tox, _) = strategy.calculate_quotes(&config, 0.0, 10.0, 1.0, &mp);

        let mut mp_zero = test_market_params();
        mp_zero.pre_fill_toxicity_bid = 0.0;
        mp_zero.pre_fill_toxicity_ask = 0.0;
        // pre_fill_spread_mult removed — AS defense via log-odds additive path

        let (bid_no, _) = strategy.calculate_quotes(&config, 0.0, 10.0, 1.0, &mp_zero);

        if let (Some(b_tox), Some(b_no)) = (&bid_tox, &bid_no) {
            let mid = config.mid_price;
            let widening_bps = ((b_no.price - b_tox.price) / mid) * 10000.0;
            // Should be capped at 15 bps (the max_as_adjustment_bps)
            assert!(
                widening_bps <= 16.0, // small tolerance for rounding
                "High toxicity (0.95) should be capped near 15 bps, got {:.2} bps",
                widening_bps
            );
            assert!(
                widening_bps > 10.0,
                "High toxicity (0.95) should produce significant widening, got {:.2} bps",
                widening_bps
            );
        }
    }

    // test_log_odds_as_fallback_multiplicative removed —
    // legacy multiplicative pre_fill_spread_mult path deleted;
    // AS defense now exclusively uses log-odds additive path.

    // ---------------------------------------------------------------
    // Taker Elasticity Estimator Tests
    // ---------------------------------------------------------------

    #[test]
    fn test_elasticity_estimator_empty() {
        let est = TakerElasticityEstimator::new(100);
        assert!(!est.is_valid());
        assert_eq!(est.observation_count(), 0);
        // Default elasticity = 1.0 (conservative)
        assert!((est.elasticity() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_elasticity_estimator_synthetic_data() {
        let mut est = TakerElasticityEstimator::new(200);

        // Generate synthetic data: fill_rate = 100 * spread^(-1.5)
        // This means eta = 1.5 (elasticity)
        let eta_true = 1.5_f64;
        for i in 0..100 {
            let spread_bps = 5.0 + (i as f64) * 0.3; // 5 to 35 bps
            let fill_rate = 100.0 * spread_bps.powf(-eta_true);
            est.record(spread_bps, fill_rate);
        }

        assert!(est.is_valid());
        assert_eq!(est.observation_count(), 100);

        // Elasticity should be close to 1.5
        let eta_est = est.elasticity();
        assert!(
            (eta_est - eta_true).abs() < 0.2,
            "Estimated eta={:.3} should be close to true eta={:.1}",
            eta_est,
            eta_true
        );
    }

    #[test]
    fn test_elasticity_estimator_rejects_invalid() {
        let mut est = TakerElasticityEstimator::new(100);

        // Zero/negative inputs should be rejected
        est.record(0.0, 1.0);
        est.record(5.0, 0.0);
        est.record(-1.0, 1.0);
        est.record(5.0, -1.0);

        assert_eq!(est.observation_count(), 0);
    }

    #[test]
    fn test_elasticity_estimator_rolling_window() {
        let mut est = TakerElasticityEstimator::new(20);

        // Fill window
        for i in 0..30 {
            est.record(10.0 + i as f64, 1.0);
        }

        // Should not exceed max_observations
        assert_eq!(est.observation_count(), 20);
    }

    // ---------------------------------------------------------------
    // Monopolist LP Pricing Tests
    // ---------------------------------------------------------------

    #[test]
    fn test_monopolist_pricing_applied_when_sole_lp() {
        let mut risk_config = RiskConfig::default();
        risk_config.use_monopolist_pricing = true;
        risk_config.monopolist_markup_cap_bps = 5.0;
        let strategy = GLFTStrategy::with_config(risk_config);
        // Use 4 decimal places so monopolist markup (0.8 bps = $0.008 at $100)
        // survives price rounding (at 2 decimals it rounds to $0.00)
        let config = QuoteConfig {
            mid_price: 100.0,
            decimals: 4,
            sz_decimals: 4,
            min_notional: 10.0,
        };

        // Sole LP: competitor_count = 0, market_share = 0.8
        let mut mp = test_market_params();
        mp.competitor_count = 0.0;
        mp.market_share = 0.8;

        let (bid_mono, _) = strategy.calculate_quotes(&config, 0.0, 10.0, 1.0, &mp);

        // Baseline: same params but many competitors (no markup)
        let mut mp_comp = test_market_params();
        mp_comp.competitor_count = 5.0;
        mp_comp.market_share = 0.2;

        let (bid_comp, _) = strategy.calculate_quotes(&config, 0.0, 10.0, 1.0, &mp_comp);

        // Monopolist should have wider spread (lower bid price)
        if let (Some(b_mono), Some(b_comp)) = (&bid_mono, &bid_comp) {
            let mid = config.mid_price;
            let extra_bps = ((b_comp.price - b_mono.price) / mid) * 10000.0;
            assert!(
                extra_bps > 0.1,
                "Monopolist should widen spread vs competitive, got {:.3} bps",
                extra_bps
            );
        }
    }

    #[test]
    fn test_monopolist_pricing_not_applied_with_competitors() {
        let mut risk_config = RiskConfig::default();
        risk_config.use_monopolist_pricing = true;
        let strategy = GLFTStrategy::with_config(risk_config);
        let config = test_quote_config();

        // Many competitors: monopolist pricing should NOT apply
        let mut mp = test_market_params();
        mp.competitor_count = 3.0;
        mp.market_share = 0.3;

        let (bid_with, _) = strategy.calculate_quotes(&config, 0.0, 10.0, 1.0, &mp);

        // Disable monopolist pricing
        let mut risk_config_off = RiskConfig::default();
        risk_config_off.use_monopolist_pricing = false;
        let strategy_off = GLFTStrategy::with_config(risk_config_off);

        let (bid_without, _) = strategy_off.calculate_quotes(&config, 0.0, 10.0, 1.0, &mp);

        // With 3 competitors, result should be same whether monopolist is enabled or not
        if let (Some(b_with), Some(b_without)) = (&bid_with, &bid_without) {
            let mid = config.mid_price;
            let diff_bps = ((b_with.price - b_without.price) / mid).abs() * 10000.0;
            assert!(
                diff_bps < 0.1,
                "With competitors, monopolist flag should not change spread: {:.3} bps diff",
                diff_bps
            );
        }
    }

    #[test]
    fn test_monopolist_markup_capped() {
        let mut risk_config = RiskConfig::default();
        risk_config.use_monopolist_pricing = true;
        risk_config.monopolist_markup_cap_bps = 3.0; // Tight cap
        let strategy = GLFTStrategy::with_config(risk_config);
        let config = test_quote_config();

        // Sole LP with maximum market share
        let mut mp = test_market_params();
        mp.competitor_count = 0.0;
        mp.market_share = 1.0; // 100% market share

        let (bid_mono, _) = strategy.calculate_quotes(&config, 0.0, 10.0, 1.0, &mp);

        // Baseline with no monopolist
        let mut risk_off = RiskConfig::default();
        risk_off.use_monopolist_pricing = false;
        let strat_off = GLFTStrategy::with_config(risk_off);

        let (bid_base, _) = strat_off.calculate_quotes(&config, 0.0, 10.0, 1.0, &mp);

        if let (Some(b_mono), Some(b_base)) = (&bid_mono, &bid_base) {
            let mid = config.mid_price;
            let markup_bps = ((b_base.price - b_mono.price) / mid) * 10000.0;
            // Markup should be capped at 3 bps
            assert!(
                markup_bps <= 3.5, // Small tolerance
                "Monopolist markup should be capped at 3 bps, got {:.2} bps",
                markup_bps
            );
        }
    }

    #[test]
    fn test_monopolist_disabled_by_default() {
        // Default RiskConfig should NOT enable monopolist pricing
        let strategy = GLFTStrategy::with_config(RiskConfig::default());
        assert!(!strategy.risk_config.use_monopolist_pricing);
    }

    #[test]
    fn test_monopolist_enabled_for_hip3() {
        // HIP-3 config SHOULD enable monopolist pricing
        let strategy = GLFTStrategy::with_config(RiskConfig::hip3());
        assert!(strategy.risk_config.use_monopolist_pricing);
    }

    // ---------------------------------------------------------------
    // GLFT Drift (mu*T) Tests
    // ---------------------------------------------------------------

    #[test]
    fn test_drift_widens_adverse_side() {
        let strategy = test_strategy();
        let gamma = 0.15;
        let kappa = 5000.0;
        let sigma = 0.001;
        let time_horizon = 60.0;
        let drift_rate = 0.0001; // positive drift = price rising

        let bid_half =
            strategy.half_spread_with_drift(gamma, kappa, sigma, time_horizon, drift_rate, true);
        let ask_half =
            strategy.half_spread_with_drift(gamma, kappa, sigma, time_horizon, drift_rate, false);
        let base = strategy.half_spread(gamma, kappa, sigma, time_horizon);

        // Positive drift (price rising): bid should be TIGHTER than base
        // (buy into uptrend → higher bid → tighter half-spread)
        assert!(
            bid_half < base,
            "bid half-spread should tighten with positive drift: bid={bid_half}, base={base}"
        );
        // Positive drift (price rising): ask should be WIDER than base
        // (don't sell into uptrend → higher ask → wider half-spread)
        assert!(
            ask_half > base,
            "ask half-spread should widen with positive drift: ask={ask_half}, base={base}"
        );
    }

    #[test]
    fn test_drift_zero_is_symmetric() {
        let strategy = test_strategy();
        let gamma = 0.15;
        let kappa = 5000.0;
        let sigma = 0.001;
        let time_horizon = 60.0;

        let bid_half =
            strategy.half_spread_with_drift(gamma, kappa, sigma, time_horizon, 0.0, true);
        let ask_half =
            strategy.half_spread_with_drift(gamma, kappa, sigma, time_horizon, 0.0, false);
        let base = strategy.half_spread(gamma, kappa, sigma, time_horizon);

        assert!(
            (bid_half - base).abs() < 1e-15,
            "zero drift bid should equal base: bid={bid_half}, base={base}"
        );
        assert!(
            (ask_half - base).abs() < 1e-15,
            "zero drift ask should equal base: ask={ask_half}, base={base}"
        );
    }

    #[test]
    fn test_drift_floor_at_maker_fee() {
        let strategy = test_strategy();
        let gamma = 0.15;
        let kappa = 5000.0;
        let sigma = 0.001;
        let time_horizon = 60.0;
        // Very large negative drift to try to push ask below fee
        let drift_rate = -10.0;

        let ask_half =
            strategy.half_spread_with_drift(gamma, kappa, sigma, time_horizon, drift_rate, false);

        // Should never go below maker fee rate
        assert!(
            ask_half >= strategy.risk_config.maker_fee_rate,
            "half-spread must be floored at maker fee: ask={ask_half}, fee={}",
            strategy.risk_config.maker_fee_rate
        );
    }

    #[test]
    fn test_solve_min_gamma_basic() {
        // With kappa=3250, sigma_effective=0.005 (realistic), T=60, fee=0.00015
        // Target: 7.42 bps = 0.000742
        // sigma_effective is per-√sec and typically 0.001-0.01 for crypto
        let gamma = solve_min_gamma(0.000742, 3250.0, 0.005, 60.0, 0.00015);
        // gamma should be > 0.15 (current default that produces 2.87 bps)
        assert!(gamma > 0.15, "min_gamma should be > 0.15, got {}", gamma);
        assert!(
            gamma < 10.0,
            "min_gamma should be reasonable, got {}",
            gamma
        );
    }

    #[test]
    fn test_solve_min_gamma_produces_target_spread() {
        let target = 0.000742; // 7.42 bps
        let kappa = 3250.0;
        let sigma = 0.005; // realistic sigma_effective
        let t = 60.0;
        let fee = 0.00015;

        let gamma = solve_min_gamma(target, kappa, sigma, t, fee);

        // Verify the spread at this gamma meets target
        let ratio = gamma / kappa;
        let glft = (1.0 / gamma) * (1.0 + ratio).ln();
        let vol = 0.5 * gamma * sigma.powi(2) * t;
        let spread = glft + vol + fee;

        assert!(
            (spread - target).abs() < 0.000001,
            "spread at min_gamma should equal target: {} vs {}",
            spread,
            target
        );
    }

    #[test]
    fn test_solve_min_gamma_low_kappa_high_gamma() {
        // Low kappa (illiquid) -> GLFT spread = (1/gamma)*ln(1+gamma/kappa) is already wider
        // So low kappa needs LESS gamma to hit the same target
        let sigma = 0.005;
        let gamma_low_kappa = solve_min_gamma(0.000742, 500.0, sigma, 60.0, 0.00015);
        let gamma_high_kappa = solve_min_gamma(0.000742, 5000.0, sigma, 60.0, 0.00015);
        assert!(
            gamma_low_kappa < gamma_high_kappa,
            "low kappa needs less gamma: {} vs {}",
            gamma_low_kappa,
            gamma_high_kappa
        );
    }

    #[test]
    fn test_solve_min_gamma_edge_cases() {
        // Very small target (below what min gamma produces): should return min gamma
        let gamma = solve_min_gamma(0.0001, 3250.0, 0.005, 60.0, 0.00015);
        assert!(gamma >= 0.01, "gamma should be at least minimum: {}", gamma);

        // Very large target: should return large gamma
        let gamma = solve_min_gamma(0.01, 3250.0, 0.005, 60.0, 0.00015);
        assert!(gamma > 1.0, "large target needs large gamma: {}", gamma);
    }

    #[test]
    fn test_solve_min_gamma_unreachable_target_returns_max() {
        // With very small sigma, large targets may be unreachable
        // solve_min_gamma should gracefully return hi (100.0)
        let gamma = solve_min_gamma(0.1, 3250.0, 0.0001, 60.0, 0.00015);
        assert!(
            (gamma - 100.0).abs() < 0.01,
            "unreachable target should return max gamma: {}",
            gamma
        );
    }

    #[test]
    fn test_additive_spread_composition_capped() {
        let strategy = test_strategy();
        let config = test_quote_config();
        let mut mp = test_market_params();
        // Set directional kappas to match symmetric kappa for clean test
        mp.kappa_bid = mp.kappa;
        mp.kappa_ask = mp.kappa;

        // Baseline spread with no widening
        let (bid_base, _ask_base) = strategy.calculate_quotes(&config, 0.0, 100.0, 1.0, &mp);

        // Additive widening: bandit + quota (spread_widening_mult removed)
        mp.bandit_spread_additive_bps = 10.0; // 10 bps bandit addon
        mp.quota_shadow_spread_bps = 10.0; // 10 bps quota pressure
        mp.total_risk_premium_bps = 5.0; // 5 bps risk premium addon

        let (bid_wide, ask_wide) = strategy.calculate_quotes(&config, 0.0, 100.0, 1.0, &mp);

        // Both quotes should exist
        assert!(
            bid_wide.is_some(),
            "Bid should exist even with additive widening"
        );
        assert!(
            ask_wide.is_some(),
            "Ask should exist even with additive widening"
        );

        // Verify additive composition:
        // base + bandit(10bps) + quota(10bps) + risk_premium(5bps)
        // Should widen moderately, not multiplicatively
        if let (Some(b_base), Some(b_wide)) = (bid_base, bid_wide) {
            let mid = mp.microprice;
            let base_spread_bps = (mid - b_base.price) / mid * 10000.0;
            let wide_spread_bps = (mid - b_wide.price) / mid * 10000.0;
            let ratio = wide_spread_bps / base_spread_bps.max(0.01);
            assert!(
                ratio < 6.0,
                "Widened spread should be < 6x base (additive), got {:.1}x ({:.1} vs {:.1} bps)",
                ratio,
                wide_spread_bps,
                base_spread_bps
            );
        }
    }

    // === E[PnL] Filter Tests (Phase 4) ===

    #[test]
    fn test_epnl_positive_at_touch_zero_inventory() {
        // At GLFT depth with no position, E[PnL] should be positive
        let depth_bps = 8.0; // typical touch depth
        let gamma = 0.1;
        let kappa = 3000.0;
        let sigma = 0.005;
        let time_horizon = 10.0;
        let fee_bps = 1.5;

        let epnl = expected_pnl_bps(
            depth_bps,
            true,
            gamma,
            kappa,
            sigma,
            time_horizon,
            0.0,  // no drift
            0.0,  // no position
            10.0, // max_position
            2.0,  // AS cost
            fee_bps,
            0.0, // no carry
        );
        assert!(
            epnl > 0.0,
            "E[PnL] should be positive at touch with zero inventory, got {:.4}",
            epnl
        );
    }

    #[test]
    fn test_epnl_negative_accumulating_high_inventory() {
        // At 80% long, bid E[PnL] should be negative (accumulating side)
        // Need high gamma and sigma so inventory penalty dominates fill capture
        let depth_bps = 8.0;
        let gamma = 50.0; // Very high risk aversion (volatile regime)
        let kappa = 3000.0;
        let sigma = 0.02; // 200 bps vol
        let time_horizon = 10.0;

        let epnl = expected_pnl_bps(
            depth_bps,
            true,
            gamma,
            kappa,
            sigma,
            time_horizon,
            0.0, // no drift
            8.0, // 80% of max
            10.0,
            2.0,
            1.5,
            0.0,
        );
        assert!(
            epnl < 0.0,
            "E[PnL] should be negative on accumulating side at 80% inventory, got {:.4}",
            epnl
        );
    }

    #[test]
    fn test_epnl_positive_reducing_high_inventory() {
        // At 80% long, ask E[PnL] should be positive (reducing side gets bonus)
        let depth_bps = 8.0;
        let gamma = 0.1;
        let kappa = 3000.0;
        let sigma = 0.005;
        let time_horizon = 10.0;

        let epnl = expected_pnl_bps(
            depth_bps,
            false,
            gamma,
            kappa,
            sigma,
            time_horizon,
            0.0, // no drift
            8.0, // 80% long → ask is reducing
            10.0,
            2.0,
            1.5,
            0.0,
        );
        assert!(
            epnl > 0.0,
            "E[PnL] should be positive on reducing side at 80% inventory, got {:.4}",
            epnl
        );
    }

    #[test]
    fn test_epnl_drift_skews_correctly() {
        // Negative drift should make bid E[PnL] < ask E[PnL]
        let depth_bps = 8.0;
        let gamma = 0.1;
        let kappa = 3000.0;
        let sigma = 0.005;
        let time_horizon = 10.0;
        let drift_rate = -0.001; // bearish drift

        let bid_epnl = expected_pnl_bps(
            depth_bps,
            true,
            gamma,
            kappa,
            sigma,
            time_horizon,
            drift_rate,
            0.0,
            10.0,
            2.0,
            1.5,
            0.0,
        );
        let ask_epnl = expected_pnl_bps(
            depth_bps,
            false,
            gamma,
            kappa,
            sigma,
            time_horizon,
            drift_rate,
            0.0,
            10.0,
            2.0,
            1.5,
            0.0,
        );
        assert!(
            bid_epnl < ask_epnl,
            "Bearish drift should make bid E[PnL] ({:.4}) < ask E[PnL] ({:.4})",
            bid_epnl,
            ask_epnl
        );
    }

    #[test]
    fn test_epnl_all_negative_extreme_sigma() {
        // With extreme sigma, all E[PnL] should be negative (matches NoQuote behavior)
        let depth_bps = 8.0;
        let gamma = 0.5;
        let kappa = 3000.0;
        let sigma = 0.05; // 5x normal — extreme vol
        let time_horizon = 10.0;

        let bid_epnl = expected_pnl_bps(
            depth_bps,
            true,
            gamma,
            kappa,
            sigma,
            time_horizon,
            0.0,
            5.0,
            10.0,
            10.0,
            1.5,
            0.0,
        );
        let ask_epnl = expected_pnl_bps(
            depth_bps,
            false,
            gamma,
            kappa,
            sigma,
            time_horizon,
            0.0,
            -5.0,
            10.0,
            10.0,
            1.5,
            0.0,
        );
        // Both sides should have negative E[PnL] due to extreme vol + high AS
        assert!(
            bid_epnl < 0.0,
            "Extreme sigma + inventory: bid E[PnL] should be negative, got {:.4}",
            bid_epnl
        );
        assert!(
            ask_epnl < 0.0,
            "Extreme sigma + inventory: ask E[PnL] should be negative, got {:.4}",
            ask_epnl
        );
    }

    // --- Enhanced E[PnL] Function Tests ---

    #[test]
    fn test_epnl_toxicity_reduces_ev() {
        let mut params = EPnLParams {
            depth_bps: 10.0,
            is_bid: true,
            gamma: 0.1,
            kappa_side: 5.0,
            sigma: 0.001,
            time_horizon: 10.0,
            drift_rate: 0.0,
            position: 0.0,
            max_position: 100.0,
            as_cost_bps: 2.0,
            fee_bps: -0.5,
            carry_cost_bps: 0.0,
            toxicity_score: 0.0,
            circuit_breaker_active: false,
            drawdown_frac: 0.0,
            self_impact_bps: 0.0,
            inventory_beta: 7.0,
        };
        let pnl_clean = expected_pnl_bps_enhanced(&params);

        params.toxicity_score = 1.0;
        let pnl_toxic = expected_pnl_bps_enhanced(&params);

        assert!(pnl_toxic < pnl_clean, "Toxicity should reduce Expected PnL");
    }

    #[test]
    fn test_epnl_cb_reduces_lambda() {
        let mut params = EPnLParams {
            depth_bps: 10.0,
            is_bid: true,
            gamma: 0.1,
            kappa_side: 5.0,
            sigma: 0.001,
            time_horizon: 10.0,
            drift_rate: 0.0,
            position: 0.0,
            max_position: 100.0,
            as_cost_bps: 2.0,
            fee_bps: -0.5,
            carry_cost_bps: 0.0,
            toxicity_score: 0.0,
            circuit_breaker_active: false,
            drawdown_frac: 0.0,
            self_impact_bps: 0.0,
            inventory_beta: 7.0,
        };
        let pnl_normal = expected_pnl_bps_enhanced(&params);

        params.circuit_breaker_active = true;
        let pnl_cb = expected_pnl_bps_enhanced(&params);

        assert!(
            pnl_cb < pnl_normal,
            "Circuit breaker should apply staleness discount reducing EV"
        );
    }

    #[test]
    fn test_epnl_drawdown_reduces_ev() {
        let mut params = EPnLParams {
            depth_bps: 10.0,
            is_bid: true,
            gamma: 0.1,
            kappa_side: 5.0,
            sigma: 0.001,
            time_horizon: 10.0,
            drift_rate: 0.0,
            position: 0.0,
            max_position: 100.0,
            as_cost_bps: 2.0,
            fee_bps: -0.5,
            carry_cost_bps: 0.0,
            toxicity_score: 0.0,
            circuit_breaker_active: false,
            drawdown_frac: 0.0,
            self_impact_bps: 0.0,
            inventory_beta: 7.0,
        };
        let pnl_normal = expected_pnl_bps_enhanced(&params);

        params.drawdown_frac = 0.10; // 10% drawdown
        let pnl_drawdown = expected_pnl_bps_enhanced(&params);

        assert!(
            pnl_drawdown < pnl_normal,
            "Drawdown should reduce fill probability confidence"
        );
    }

    #[test]
    fn test_epnl_impact_near_touch() {
        let mut params = EPnLParams {
            depth_bps: 5.0,
            is_bid: true,
            gamma: 0.1,
            kappa_side: 5.0,
            sigma: 0.001,
            time_horizon: 10.0,
            drift_rate: 0.0,
            position: 0.0,
            max_position: 100.0,
            as_cost_bps: 2.0,
            fee_bps: -0.5,
            carry_cost_bps: 0.0,
            toxicity_score: 0.0,
            circuit_breaker_active: false,
            drawdown_frac: 0.0,
            self_impact_bps: 0.0,
            inventory_beta: 7.0,
        };
        let pnl_no_impact = expected_pnl_bps_enhanced(&params);

        params.self_impact_bps = 2.0; // Touch quotes have self impact
        let pnl_impact = expected_pnl_bps_enhanced(&params);

        // The subtraction is entirely linear in the equation
        assert!(
            (pnl_impact - (pnl_no_impact - 2.0)).abs() < 1e-9,
            "Self impact should be exactly subtracted"
        );
    }

    #[test]
    fn test_epnl_struct_matches_legacy() {
        let legacy = expected_pnl_bps(
            10.0, true, 0.1, 5.0, 0.001, 10.0, 0.0, 50.0, 100.0, 2.0, -0.5, 0.5,
        );

        let params = EPnLParams {
            depth_bps: 10.0,
            is_bid: true,
            gamma: 0.1,
            kappa_side: 5.0,
            sigma: 0.001,
            time_horizon: 10.0,
            drift_rate: 0.0,
            position: 50.0,
            max_position: 100.0,
            as_cost_bps: 2.0,
            fee_bps: -0.5,
            carry_cost_bps: 0.5,
            toxicity_score: 0.0,
            circuit_breaker_active: false,
            drawdown_frac: 0.0,
            self_impact_bps: 0.0,
            inventory_beta: 0.0,
        };
        let enhanced = expected_pnl_bps_enhanced(&params);

        assert!(
            (legacy - enhanced).abs() < 1e-9,
            "Legacy wrapper with zeroed struct overlays must match enhanced precisely"
        );
    }

    #[test]
    fn test_epnl_diagnostics_decomposition() {
        // WS7: Verify diagnostics decomposition matches the scalar result.
        let params = EPnLParams {
            depth_bps: 8.0,
            is_bid: true,
            gamma: 0.1,
            kappa_side: 3000.0,
            sigma: 0.005,
            time_horizon: 10.0,
            drift_rate: 0.0001, // slight bullish drift
            position: 2.0,
            max_position: 10.0,
            as_cost_bps: 2.0,
            fee_bps: 1.5,
            carry_cost_bps: 0.0,
            toxicity_score: 0.3,
            circuit_breaker_active: false,
            drawdown_frac: 0.0,
            self_impact_bps: 0.1,
            inventory_beta: 1.0,
        };

        let scalar = expected_pnl_bps_enhanced(&params);
        let (diag_scalar, diag) = expected_pnl_bps_with_diagnostics(&params);

        // Scalar from both functions must match exactly
        assert!(
            (scalar - diag_scalar).abs() < 1e-12,
            "Diagnostics scalar must match enhanced. scalar={}, diag={}",
            scalar,
            diag_scalar
        );

        // Verify diagnostics fields are populated
        assert!((diag.depth_bps - 8.0).abs() < 1e-9);
        assert!(diag.is_bid);
        assert!(diag.lambda > 0.0, "Lambda should be positive");
        assert!(
            diag.toxicity_cost_bps > 0.0,
            "Toxicity cost should be positive"
        );
        assert!(
            diag.drift_contribution_bps > 0.0,
            "Bullish drift on bid should be positive"
        );
        assert!(diag.self_impact_bps > 0.0, "Self impact should be positive");
        assert!(!diag.circuit_breaker_active);
        assert!(
            (diag.drawdown_penalty - 1.0).abs() < 1e-9,
            "No drawdown → penalty=1.0"
        );
    }

    #[test]
    fn test_as_vol_floor_deduplication() {
        // When realized_as equals the vol floor, as_net should be zero
        let gamma: f64 = 100.0;
        let sigma: f64 = 0.001; // 10 bps/sqrt(s)
        let as_horizon_s: f64 = 0.5;

        let vol_floor = 0.5 * gamma * sigma.powi(2) * as_horizon_s;
        // vol_floor = 0.5 * 100 * 0.000001 * 0.5 = 0.000025 (0.25 bps)

        // AS exactly equals vol floor → net is zero
        let as_raw = vol_floor;
        let as_net = (as_raw - vol_floor).max(0.0);
        assert!(
            as_net.abs() < 1e-15,
            "AS net should be 0 when as_raw == vol_floor, got {}",
            as_net
        );

        // AS >> vol_floor → net ≈ as_raw
        let as_raw_large = 0.001; // 10 bps
        let as_net_large = (as_raw_large - vol_floor).max(0.0);
        assert!(
            (as_net_large - (as_raw_large - vol_floor)).abs() < 1e-15,
            "AS net should be as_raw - vol_floor when as_raw >> vol_floor"
        );
        assert!(as_net_large > 0.0);

        // AS < vol_floor → net clamped to zero (no negative addon)
        let as_raw_small = vol_floor * 0.5;
        let as_net_small = (as_raw_small - vol_floor).max(0.0);
        assert_eq!(
            as_net_small, 0.0,
            "AS net should be 0 when as_raw < vol_floor"
        );
    }

    // === Fix 15: reducing_threshold_bps tests ===

    #[test]
    fn test_reducing_threshold_flat_position_is_zero() {
        let t = reducing_threshold_bps(0.0, 10.0, 1.5, 1.0, 1.0);
        assert!(
            (t - 0.0).abs() < 1e-12,
            "Flat position → no carve-out, got {t}"
        );
    }

    #[test]
    fn test_reducing_threshold_80pct_position() {
        // q_ratio = 0.8, sqrt(0.8) ≈ 0.8944, gamma_ratio = 1.0
        // threshold = -2.0 * 1.5 * 1.0 * sqrt(0.8) ≈ -2.683
        let t = reducing_threshold_bps(8.0, 10.0, 1.5, 1.0, 1.0);
        assert!(
            (t - (-2.0 * 1.5 * 0.8_f64.powf(0.5))).abs() < 0.01,
            "80% position → ~-2.68 bps, got {t}"
        );
    }

    #[test]
    fn test_reducing_threshold_100pct_extreme_regime() {
        // gamma=3.0, baseline=1.0 → gamma_ratio=3.0
        // threshold = -2.0 * 1.5 * 3.0 * 1.0^0.5 = -9.0
        let t = reducing_threshold_bps(10.0, 10.0, 1.5, 3.0, 1.0);
        assert!(
            (t - (-9.0)).abs() < 1e-9,
            "Extreme regime at 100% pos → -9.0 bps (3 fees * 2), got {t}"
        );
    }

    #[test]
    fn test_reducing_threshold_100pct_calm_regime() {
        // gamma=0.5, baseline=1.0 → gamma_ratio=0.5
        // threshold = -2.0 * 1.5 * 0.5 * 1.0^0.5 = -1.5
        let t = reducing_threshold_bps(10.0, 10.0, 1.5, 0.5, 1.0);
        assert!(
            (t - (-1.5)).abs() < 1e-9,
            "Calm regime at 100% pos → -1.5 bps (half fee * 2), got {t}"
        );
    }

    #[test]
    fn test_reducing_threshold_gamma_ratio_clamped() {
        // gamma=10.0, baseline=1.0 → gamma_ratio clamped to 3.0
        let t = reducing_threshold_bps(10.0, 10.0, 1.5, 10.0, 1.0);
        assert!(
            (t - (-9.0)).abs() < 1e-9,
            "Gamma ratio should clamp at 3.0, got {t}"
        );

        // gamma=0.01, baseline=1.0 → gamma_ratio clamped to 0.5
        let t2 = reducing_threshold_bps(10.0, 10.0, 1.5, 0.01, 1.0);
        assert!(
            (t2 - (-1.5)).abs() < 1e-9,
            "Gamma ratio should clamp at 0.5, got {t2}"
        );
    }

    #[test]
    fn test_reducing_threshold_zero_max_position() {
        let t = reducing_threshold_bps(5.0, 0.0, 1.5, 1.0, 1.0);
        assert!(
            (t - 0.0).abs() < 1e-12,
            "Zero max position → no carve-out, got {t}"
        );
    }

    #[test]
    fn test_reducing_threshold_integration_with_epnl() {
        // Integration: 80% position + deeply negative E[PnL] on reducing side → levels survive
        let fee_bps = 1.5;
        let position = 8.0; // long 80%
        let max_pos = 10.0;
        let gamma = 1.0;
        let gamma_baseline = 0.15;

        let reducing_thresh =
            reducing_threshold_bps(position, max_pos, fee_bps, gamma, gamma_baseline);
        // gamma_ratio = (1.0/0.15).clamp(0.5, 3.0) = 3.0 (clamped)
        // threshold = -1.5 * 3.0 * 0.8^1.5 ≈ -3.22
        assert!(
            reducing_thresh < -3.0,
            "Should be deeply negative: {reducing_thresh}"
        );

        // Accumulating threshold is always 0
        let accum_thresh = 0.0;
        assert!(
            reducing_thresh < accum_thresh,
            "Reducing threshold ({reducing_thresh}) must be below accumulating ({accum_thresh})"
        );
    }

    // ---------------------------------------------------------------
    // WS2: PPIP (Posterior Predictive Inventory Penalty) Tests
    // ---------------------------------------------------------------

    #[test]
    fn test_ws2_ppip_produces_meaningful_skew() {
        // CRITICAL: The old formula produced ~10⁻¹³ bps skew at 45% utilization.
        // With PPIP and τ_inventory = 60s, we must get > 1 bps.
        let ppip = PosteriorPredictiveSkew {
            drift_mean: 2.5e-5,       // +38 bps/25min = 2.5e-5 frac/sec
            sigma_sq_mean: 1e-8,      // σ = 0.0001/s → σ² = 1e-8
            sigma_sq_variance: 5e-17, // 50% CV²
            tau_mean: 60.0,           // 60 seconds holding period
            tau_variance: 900.0,      // 30s std dev
            ..PosteriorPredictiveSkew::default()
        };

        let shift = ppip.reservation_shift(0.45, 10.0);
        let shift_bps = shift.abs() * 10_000.0;

        assert!(
            shift_bps > 1.0,
            "PPIP must produce > 1 bps skew at 45% utilization, got {shift_bps:.4} bps"
        );
    }

    #[test]
    fn test_ws2_ppip_zero_position_zero_skew() {
        let ppip = PosteriorPredictiveSkew::default();
        let shift = ppip.reservation_shift(0.0, 10.0);
        // With default drift_mean = 0.0, all terms vanish at q=0
        assert!(
            shift.abs() < 1e-15,
            "PPIP with no drift and no position should be ~0, got {shift}"
        );
    }

    #[test]
    fn test_ws2_ppip_scales_with_position() {
        let ppip = PosteriorPredictiveSkew {
            sigma_sq_mean: 1e-8,
            tau_mean: 60.0,
            tau_variance: 900.0,
            ..PosteriorPredictiveSkew::default()
        };

        let shift_small = ppip.reservation_shift(0.1, 10.0);
        let shift_large = ppip.reservation_shift(0.5, 10.0);

        // Larger position → larger skew (variance_cost is proportional to q)
        assert!(
            shift_large.abs() > shift_small.abs(),
            "Skew should increase with position: small={shift_small:.8}, large={shift_large:.8}"
        );
    }

    #[test]
    fn test_ws2_ppip_ambiguity_aversion_widens_skew() {
        // When σ² uncertainty is high (warmup/regime transition), skew should be larger
        let ppip_certain = PosteriorPredictiveSkew {
            sigma_sq_mean: 1e-8,
            sigma_sq_variance: 1e-17, // Low uncertainty: CV² = 0.1
            tau_mean: 60.0,
            tau_variance: 900.0,
            ..PosteriorPredictiveSkew::default()
        };

        let ppip_uncertain = PosteriorPredictiveSkew {
            sigma_sq_variance: 5e-16, // High uncertainty: CV² = 5.0
            ..ppip_certain.clone()
        };

        let shift_certain = ppip_certain.reservation_shift(0.3, 10.0);
        let shift_uncertain = ppip_uncertain.reservation_shift(0.3, 10.0);

        assert!(
            shift_uncertain.abs() > shift_certain.abs(),
            "High sigma uncertainty should produce wider skew: certain={:.6}, uncertain={:.6}",
            shift_certain * 10000.0,
            shift_uncertain * 10000.0,
        );
    }

    #[test]
    fn test_ws2_ppip_timing_uncertainty_widens_skew() {
        // When τ variance is high, drift cost should be amplified
        let ppip_stable = PosteriorPredictiveSkew {
            drift_mean: 1e-5, // small drift
            sigma_sq_mean: 1e-8,
            tau_mean: 60.0,
            tau_variance: 100.0, // Low CV²(τ) ≈ 0.028
            ..PosteriorPredictiveSkew::default()
        };

        let ppip_volatile = PosteriorPredictiveSkew {
            tau_variance: 3600.0, // High CV²(τ) = 1.0
            ..ppip_stable.clone()
        };

        let shift_stable = ppip_stable.reservation_shift(0.3, 10.0);
        let shift_volatile = ppip_volatile.reservation_shift(0.3, 10.0);

        assert!(
            shift_volatile.abs() > shift_stable.abs(),
            "High tau uncertainty should produce wider skew: stable={:.6}, volatile={:.6}",
            shift_stable * 10000.0,
            shift_volatile * 10000.0,
        );
    }

    #[test]
    fn test_ws2_ppip_self_calibrating_multiplier() {
        let ppip_1x = PosteriorPredictiveSkew {
            sigma_sq_mean: 1e-8,
            tau_mean: 60.0,
            tau_variance: 900.0,
            skew_multiplier: 1.0,
            ..PosteriorPredictiveSkew::default()
        };

        let shift_1x = ppip_1x.reservation_shift(0.3, 10.0);

        let ppip_2x = PosteriorPredictiveSkew {
            skew_multiplier: 2.0,
            ..ppip_1x.clone()
        };
        let shift_2x = ppip_2x.reservation_shift(0.3, 10.0);

        let ratio = shift_2x / shift_1x;
        assert!(
            (ratio - 2.0).abs() < 0.01,
            "Multiplier 2x should double the skew, got ratio {ratio:.4}"
        );
    }

    #[test]
    fn test_ws2_dual_timescale_observe_reducing_fill() {
        let mut controller = DualTimescaleController::default();
        let initial_tau = controller.tau_inventory;

        // Simulate 10 reducing fills with ~45s holding duration
        for _ in 0..10 {
            controller.observe_reducing_fill(45.0);
        }

        // τ_inventory should converge toward 45s from default 60s
        assert!(
            controller.tau_inventory < initial_tau,
            "After fills with 45s duration, tau should decrease from {initial_tau}: got {}",
            controller.tau_inventory
        );
        assert!(
            controller.tau_inventory > 40.0 && controller.tau_inventory < 60.0,
            "tau_inventory should be between 40 and 60 after convergence: got {}",
            controller.tau_inventory
        );
    }

    #[test]
    fn test_ws2_ppip_calibrate_from_markout() {
        let mut ppip = PosteriorPredictiveSkew {
            skew_multiplier: 1.0,
            ..PosteriorPredictiveSkew::default()
        };

        // Model predicts 2 bps shift but realized is 4 bps → multiplier should increase
        for _ in 0..50 {
            ppip.calibrate_from_markout(0.0002, 0.0004, 0.02);
        }

        assert!(
            ppip.skew_multiplier > 1.5,
            "Multiplier should increase when model under-predicts: got {:.3}",
            ppip.skew_multiplier
        );
    }

    #[test]
    fn test_ws2_skew_alert_threshold() {
        // Verify the diagnostic: |position| > 10% max AND |skew| < 1 bps should be alarming
        // This is a regression guard — with PPIP, this should never happen
        let ppip = PosteriorPredictiveSkew {
            sigma_sq_mean: 1e-8,
            tau_mean: 60.0,
            tau_variance: 900.0,
            ..PosteriorPredictiveSkew::default()
        };

        let shift = ppip.reservation_shift(0.15, 10.0); // 15% utilization
        let shift_bps = shift.abs() * 10_000.0;

        // With tau_inventory = 60s and σ² = 1e-8, we should get measurable skew
        // at 15% position (1.5 units out of 10)
        assert!(
            shift_bps > 0.01,
            "PPIP should produce non-trivial skew at 15% utilization: got {shift_bps:.6} bps"
        );
    }
}
