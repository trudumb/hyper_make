//! GLFT (Guéant-Lehalle-Fernandez-Tapia) optimal market making strategy.

use tracing::debug;

use crate::{round_to_significant_and_decimal, truncate_float, EPSILON};

use crate::market_maker::config::{Quote, QuoteConfig};

use super::{
    CalibratedRiskModel, KellySizer, MarketParams, QuotingStrategy, RiskConfig, RiskFeatures,
    RiskModelConfig, SpreadComposition,
};

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
        self.elasticity_estimator.record(spread_bps, fill_rate_per_s);
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
    /// Supports two modes:
    /// 1. **Legacy (Multiplicative)**: γ = γ_base × vol × tox × inv × ...
    /// 2. **Calibrated (Log-Additive)**: log(γ) = log(γ_base) + Σ βᵢ × xᵢ
    ///
    /// Blending controlled by `risk_model_config.risk_model_blend`:
    /// - blend=0.0: Pure legacy multiplicative
    /// - blend=1.0: Pure log-additive calibrated
    /// - blend=0.5: 50/50 blend
    fn effective_gamma(
        &self,
        market_params: &MarketParams,
        position: f64,
        max_position: f64,
    ) -> f64 {
        let cfg = &self.risk_config;
        let blend = self.risk_model_config.risk_model_blend.clamp(0.0, 1.0);

        // ============================================================
        // MODE 1: LEGACY MULTIPLICATIVE GAMMA
        // ============================================================
        let gamma_legacy = if blend < 1.0 {
            self.compute_legacy_gamma(market_params, position, max_position)
        } else {
            0.0 // Not needed if pure log-additive
        };

        // ============================================================
        // MODE 2: CALIBRATED LOG-ADDITIVE GAMMA
        // ============================================================
        let gamma_calibrated = if blend > 0.0 || self.risk_model_config.use_calibrated_risk_model {
            // Build normalized risk features
            let features = RiskFeatures::from_params(
                market_params,
                position,
                max_position,
                &self.risk_model_config,
            );

            // Compute gamma via log-additive model
            self.risk_model.compute_gamma(&features)
        } else {
            0.0 // Not needed if pure legacy
        };

        // ============================================================
        // BLEND BETWEEN MODELS
        // ============================================================
        let gamma_base = if blend <= 0.0 {
            gamma_legacy
        } else if blend >= 1.0 {
            gamma_calibrated
        } else {
            // Linear blend in gamma space (not log space)
            gamma_legacy * (1.0 - blend) + gamma_calibrated * blend
        };

        // ============================================================
        // POST-PROCESS SCALARS (always applied)
        // ============================================================
        // These are event-driven and can't be normalized into features:
        // 1. calibration_gamma_mult: Fill rate controller during warmup
        // 2. tail_risk_multiplier: Cascade detection (discrete event)
        let gamma_with_calib = gamma_base * market_params.calibration_gamma_mult;
        let gamma_with_tail = gamma_with_calib * market_params.tail_risk_multiplier;

        // RL policy multiplier: allows learned risk aversion scaling.
        // Clamped to [0.1, 10.0] to prevent blow-ups even if RL outputs garbage.
        // Default 1.0 = no-op until RL is explicitly enabled.
        let rl_gamma_mult = market_params.rl_gamma_multiplier.clamp(0.1, 10.0);
        let gamma_final = gamma_with_tail * rl_gamma_mult;

        // Self-consistent gamma: ensure GLFT output >= spread floor.
        //
        // The floor is physical constraints only: fee + tick + latency.
        // Regime risk routes through gamma_multiplier, not floor clamping.
        let time_horizon = self.holding_time(market_params.arrival_intensity);
        let physical_floor_frac = cfg.maker_fee_rate.max(cfg.min_spread_floor);
        let min_gamma = solve_min_gamma(
            physical_floor_frac,
            market_params.kappa.max(1.0),
            market_params.sigma_effective.max(1e-8),
            time_horizon,
            cfg.maker_fee_rate,
        );
        // Apply regime gamma multiplier: routes regime risk through gamma
        let gamma_with_floor = gamma_final.max(min_gamma) * market_params.regime_gamma_multiplier;

        let gamma_clamped = gamma_with_floor.clamp(cfg.gamma_min, cfg.gamma_max);

        // Log comparison for shadow mode validation
        if self.risk_model_config.use_calibrated_risk_model || blend > 0.0 {
            debug!(
                gamma_legacy = %format!("{:.4}", gamma_legacy),
                gamma_calibrated = %format!("{:.4}", gamma_calibrated),
                blend = %format!("{:.2}", blend),
                gamma_base = %format!("{:.4}", gamma_base),
                calib_mult = %format!("{:.3}", market_params.calibration_gamma_mult),
                tail_mult = %format!("{:.3}", market_params.tail_risk_multiplier),
                rl_gamma_mult = %format!("{:.3}", rl_gamma_mult),
                gamma_final = %format!("{:.4}", gamma_clamped),
                "Gamma: legacy vs calibrated comparison"
            );
        } else {
            debug!(
                gamma_base = %format!("{:.3}", cfg.gamma_base),
                gamma_raw = %format!("{:.4}", gamma_legacy),
                calib_mult = %format!("{:.3}", market_params.calibration_gamma_mult),
                tail_mult = %format!("{:.3}", market_params.tail_risk_multiplier),
                rl_gamma_mult = %format!("{:.3}", rl_gamma_mult),
                gamma_clamped = %format!("{:.4}", gamma_clamped),
                "Gamma: legacy mode"
            );
        }

        gamma_clamped
    }

    /// Compute gamma using legacy multiplicative model.
    ///
    /// γ = γ_base × vol × tox × inv × hawkes × time × book × uncertainty
    fn compute_legacy_gamma(
        &self,
        market_params: &MarketParams,
        position: f64,
        max_position: f64,
    ) -> f64 {
        let cfg = &self.risk_config;

        // === VOLATILITY SCALING ===
        let vol_ratio = market_params.sigma_effective / cfg.sigma_baseline.max(1e-9);
        let vol_scalar = if vol_ratio <= 1.0 {
            1.0
        } else {
            let raw = 1.0 + cfg.volatility_weight * (vol_ratio - 1.0);
            raw.min(cfg.max_volatility_multiplier)
        };

        // === TOXICITY SCALING ===
        // Combine backward-looking (jump_ratio) and forward-looking (pre-fill classifier) signals
        let legacy_toxicity = if market_params.jump_ratio <= cfg.toxicity_threshold {
            1.0
        } else {
            1.0 + cfg.toxicity_sensitivity * (market_params.jump_ratio - 1.0)
        };

        // Pre-fill classifier provides forward-looking toxicity prediction [0, 1]
        // Use the symmetric (max of bid/ask) for overall gamma scaling
        let pre_fill_mult = market_params.pre_fill_spread_mult_bid.max(market_params.pre_fill_spread_mult_ask);

        // Blend legacy and pre-fill: take the max (conservative approach)
        // This ensures we widen spreads if EITHER signal indicates toxicity
        let toxicity_scalar = legacy_toxicity.max(pre_fill_mult);

        // === INVENTORY SCALING ===
        let utilization = if max_position > EPSILON {
            (position.abs() / max_position).min(1.0)
        } else {
            0.0
        };
        let inventory_scalar = if utilization <= cfg.inventory_threshold {
            1.0
        } else {
            let excess = utilization - cfg.inventory_threshold;
            1.0 + cfg.inventory_sensitivity * excess.powi(2)
        };

        // === HAWKES ACTIVITY SCALING ===
        let hawkes_baseline = 0.5;
        let hawkes_sensitivity = 2.0;
        let hawkes_scalar = 1.0
            + hawkes_sensitivity
                * (market_params.hawkes_activity_percentile - hawkes_baseline).max(0.0);

        // === TIME-OF-DAY SCALING ===
        let time_scalar = cfg.time_of_day_multiplier();

        // === BOOK DEPTH SCALING ===
        let book_depth_scalar = cfg.book_depth_multiplier(market_params.near_touch_depth_usd);

        // === UNCERTAINTY SCALING ===
        let uncertainty_scalar = if market_params.kappa_ci_width > 0.0 {
            1.0 + (market_params.kappa_ci_width / 10.0).min(0.5)
        } else {
            1.0
        };

        // Combine (note: calibration and tail_risk applied in caller)
        let gamma_effective = cfg.gamma_base
            * vol_scalar
            * toxicity_scalar
            * inventory_scalar
            * hawkes_scalar
            * time_scalar
            * book_depth_scalar
            * uncertainty_scalar;

        gamma_effective.clamp(cfg.gamma_min, cfg.gamma_max)
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
    /// - Positive drift (price rising) widens bids, tightens asks
    /// - Negative drift (price falling) tightens bids, widens asks
    ///
    /// Formula: `delta_bid = base + mu*T/2`, `delta_ask = base - mu*T/2`
    ///
    /// The drift adjustment is clamped so the half-spread never goes below
    /// the maker fee rate (no negative-EV quotes).
    #[allow(dead_code)] // Will be wired in signal_integration.rs
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
        // Positive drift -> widen bids (buying into uptrend risky), tighten asks
        let drift_adjustment = drift_rate * time_horizon / 2.0;
        let adjusted = if is_bid {
            base + drift_adjustment
        } else {
            base - drift_adjustment
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
        let sigma = market_params.sigma_effective;
        let tau = self.holding_time(market_params.arrival_intensity);

        // Core GLFT half-spread in fraction
        let glft_half_frac = self.half_spread(gamma, kappa, sigma, tau);
        let glft_half_bps = glft_half_frac * 10_000.0;

        // Risk premium: convert the widening_mult excess to additive bps
        // spread_widening_mult of 1.5 on 5 bps GLFT = 2.5 bps addon.
        // Capped at 100% of GLFT spread (at most doubles the base).
        let widening_excess = (market_params.spread_widening_mult - 1.0).clamp(0.0, 1.0);
        let risk_premium_bps = glft_half_bps * widening_excess
            + market_params.regime_risk_premium_bps
            + market_params.total_risk_premium_bps;

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
            + market_params.regime_probs[3] * 0.2;             // Extreme: very cautious

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

    /// Flow-adjusted inventory skew with exponential regularization.
    ///
    /// The flow_alignment ∈ [-1, 1] measures how aligned position is with flow:
    ///   +1 = perfectly aligned (long + buy flow, or short + sell flow)
    ///   -1 = perfectly opposed (long + sell flow, or short + buy flow)
    ///    0 = no flow signal
    ///
    /// We use exponential regularization that naturally bounds the modifier:
    ///   modifier = exp(-β × flow_alignment)
    ///
    /// This is mathematically clean:
    ///   - exp(0) = 1.0 (no adjustment when no flow)
    ///   - exp(β) ≈ 1 + β for small β (linear approximation)
    ///   - Always positive (can't flip skew sign)
    ///   - Symmetric in positive/negative alignment
    ///
    /// When aligned (flow pushed us here): dampen counter-skew (don't fight momentum)
    /// When opposed (fighting informed flow): amplify counter-skew (reduce risk faster)
    fn inventory_skew_with_flow(
        &self,
        inventory_ratio: f64,
        sigma: f64,
        gamma: f64,
        time_horizon: f64,
        flow_imbalance: f64,
    ) -> f64 {
        // Base GLFT skew (Avellaneda-Stoikov)
        let base_skew = inventory_ratio * gamma * sigma.powi(2) * time_horizon;

        // Flow alignment: positive when position and flow have same sign
        // inventory_ratio.signum() gives direction of position
        // flow_imbalance ∈ [-1, 1] from MarketParams
        // flow_alignment = inventory_ratio.signum() * flow_imbalance ∈ [-1, 1]
        let flow_alignment = inventory_ratio.signum() * flow_imbalance;

        // Regularized modifier using exponential
        // exp(-β × alignment) because:
        //   aligned (positive) → smaller modifier → dampen skew
        //   opposed (negative) → larger modifier → amplify skew
        let flow_modifier = (-self.risk_config.flow_sensitivity * flow_alignment).exp();

        base_skew * flow_modifier
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
        // === 0. CIRCUIT BREAKERS ===
        // Extreme liquidation cascade detected - pull all quotes immediately
        if market_params.should_pull_quotes {
            debug!(
                tail_risk_mult = %format!("{:.2}", market_params.tail_risk_multiplier),
                cascade_size_factor = %format!("{:.2}", market_params.cascade_size_factor),
                "CIRCUIT BREAKER: Liquidation cascade detected - pulling all quotes"
            );
            return (None, None);
        }

        // Edge surface says don't quote (expected edge <= 0 or low confidence)
        if !market_params.should_quote_edge && market_params.flow_decomp_confidence > 0.5 {
            debug!(
                current_edge_bps = %format!("{:.2}", market_params.current_edge_bps),
                flow_decomp_confidence = %format!("{:.2}", market_params.flow_decomp_confidence),
                "CIRCUIT BREAKER: Edge surface indicates no edge - pulling quotes"
            );
            return (None, None);
        }

        // Joint dynamics detects toxic state (high AS + high informed correlated with volatility)
        if market_params.is_toxic_joint && market_params.flow_decomp_confidence > 0.6 {
            debug!(
                p_informed = %format!("{:.3}", market_params.p_informed),
                sigma_kappa_corr = %format!("{:.2}", market_params.sigma_kappa_correlation),
                "CIRCUIT BREAKER: Joint dynamics detects toxic state - pulling quotes"
            );
            return (None, None);
        }

        // FIRST PRINCIPLES: Use dynamic max_position derived from equity/volatility
        // Falls back to static max_position if margin state hasn't been refreshed yet.
        // config.max_position (passed as max_position) is ALWAYS the hard ceiling.
        let effective_max_position = market_params.effective_max_position(max_position).min(max_position);

        // === 1. DYNAMIC GAMMA with Tail Risk ===
        // When adaptive spreads enabled: use log-additive shrinkage gamma
        // When disabled: use multiplicative RiskConfig gamma
        //
        // KEY FIX: Use `adaptive_can_estimate` instead of `adaptive_warmed_up`
        // The adaptive system provides usable values IMMEDIATELY via Bayesian priors.
        // We don't need to wait for 20+ fills - priors give reasonable starting points.
        let gamma = if market_params.use_adaptive_spreads && market_params.adaptive_can_estimate {
            // Adaptive gamma: log-additive scaling prevents multiplicative explosion
            // Still apply tail risk multiplier for cascade protection
            let adaptive_gamma = market_params.adaptive_gamma;
            let gamma_with_tail = adaptive_gamma * market_params.tail_risk_multiplier;
            debug!(
                adaptive_gamma = %format!("{:.4}", adaptive_gamma),
                tail_mult = %format!("{:.2}", market_params.tail_risk_multiplier),
                gamma_final = %format!("{:.4}", gamma_with_tail),
                warmup_progress = %format!("{:.0}%", market_params.adaptive_warmup_progress * 100.0),
                "Using ADAPTIVE gamma (log-additive shrinkage)"
            );
            gamma_with_tail
        } else {
            // Legacy: multiplicative RiskConfig gamma
            let base_gamma = self.effective_gamma(market_params, position, effective_max_position);
            // Apply liquidity multiplier: thin book → higher gamma → wider spread
            let gamma_with_liq = base_gamma * market_params.liquidity_gamma_mult;
            // Apply tail risk multiplier: during cascade → much higher gamma → wider spread
            gamma_with_liq * market_params.tail_risk_multiplier
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
        let sigma = market_params.sigma_effective;
        let tau = time_horizon;
        let mut half_spread_bid = self.half_spread(gamma, kappa_bid, sigma, tau);
        let mut half_spread_ask = self.half_spread(gamma, kappa_ask, sigma, tau);

        // Symmetric half-spread for logging (average of bid/ask)
        let mut half_spread = (half_spread_bid + half_spread_ask) / 2.0;

        // === 2a. ADVERSE SELECTION SPREAD ADJUSTMENT ===
        // Add measured AS cost to half-spreads (only when warmed up)
        if market_params.as_warmed_up && market_params.as_spread_adjustment > 0.0 {
            half_spread_bid += market_params.as_spread_adjustment;
            half_spread_ask += market_params.as_spread_adjustment;
            half_spread += market_params.as_spread_adjustment;
            debug!(
                as_adj_bps = %format!("{:.2}", market_params.as_spread_adjustment * 10000.0),
                predicted_alpha = %format!("{:.3}", market_params.predicted_alpha),
                "AS spread adjustment applied"
            );
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
        } else {
            // Legacy multiplicative path (fallback)
            let bid_mult = market_params.pre_fill_spread_mult_bid;
            let ask_mult = market_params.pre_fill_spread_mult_ask;

            if bid_mult > 1.01 || ask_mult > 1.01 {
                let bid_add = half_spread_bid * (bid_mult - 1.0);
                let ask_add = half_spread_ask * (ask_mult - 1.0);

                half_spread_bid += bid_add;
                half_spread_ask += ask_add;
                half_spread += (bid_add + ask_add) / 2.0;

                debug!(
                    bid_tox = %format!("{:.3}", market_params.pre_fill_toxicity_bid),
                    ask_tox = %format!("{:.3}", market_params.pre_fill_toxicity_ask),
                    bid_mult = %format!("{:.2}", bid_mult),
                    ask_mult = %format!("{:.2}", ask_mult),
                    bid_add_bps = %format!("{:.2}", bid_add * 10000.0),
                    ask_add_bps = %format!("{:.2}", ask_add * 10000.0),
                    "Pre-fill toxicity multiplicative widening applied (legacy)"
                );
            }
        }

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
        if self.risk_config.use_monopolist_pricing
            && market_params.competitor_count < 1.5
        {
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
        // Floor clamp: safety net only. With unified floor in solve_min_gamma,
        // this should fire <5% of cycles (only during transient parameter changes).
        let floor_bound = half_spread_bid < effective_floor
            || half_spread_ask < effective_floor;
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
        let sigma_for_skew =
            if market_params.sigma_particle > 0.0 && market_params.flow_decomp_confidence > 0.3 {
                // Use particle filter sigma (in bps/sqrt(s), convert to fractional)
                market_params.sigma_particle / 10_000.0
            } else {
                // Fall back to leverage-adjusted sigma
                market_params.sigma_leverage_adjusted
            };

        // FIX: Meaningful sigma floor to prevent zero skew during warmup
        // Without this floor, sigma=0 during warmup causes skew=0 always
        // 0.0001 = 1 bp/sec baseline - provides meaningful skew even with small positions
        let sigma_for_skew = sigma_for_skew.max(0.0001);

        // Calculate inventory ratio: q / Q_max (normalized to [-1, 1])
        let inventory_ratio = if effective_max_position > EPSILON {
            (position / effective_max_position).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // === STOCHASTIC MODULE: HJB vs Heuristic Skew ===
        // When use_hjb_skew is true, use optimal skew from HJB controller
        // When false, use flow-dampened heuristic (existing behavior)
        let base_skew = if market_params.use_hjb_skew {
            // HJB optimal skew from Avellaneda-Stoikov HJB solution:
            // skew = γσ²qT + terminal_penalty × q × urgency + funding_bias
            // This is pre-computed by HJBInventoryController in mod.rs
            if market_params.hjb_is_terminal_zone {
                debug!(
                    hjb_skew = %format!("{:.6}", market_params.hjb_optimal_skew),
                    hjb_inv_target = %format!("{:.4}", market_params.hjb_inventory_target),
                    "HJB TERMINAL ZONE: Aggressive inventory reduction"
                );
            }

            // FIX: When position is small but non-zero, amplify skew signal
            // The HJB formula multiplies by q (normalized position), so small q → tiny skew
            // This amplifier ensures meaningful skew even with small positions
            let hjb_skew = market_params.hjb_optimal_skew;
            let position_amplifier = if inventory_ratio.abs() < 0.1 && inventory_ratio.abs() > 0.01 {
                // 10x amplification for small positions (1-10% of max)
                // This compensates for the q multiplication in the HJB formula
                10.0
            } else if inventory_ratio.abs() <= 0.01 && inventory_ratio.abs() > 0.001 {
                // 5x amplification for very small positions (0.1-1% of max)
                5.0
            } else {
                1.0
            };
            hjb_skew * position_amplifier
        } else {
            // Flow-dampened inventory skew: base_skew × exp(-β × flow_alignment)
            // Uses flow_imbalance to dampen skew when aligned with flow (don't fight momentum)
            // and amplify skew when opposed to flow (reduce risk faster)
            let raw_skew = self.inventory_skew_with_flow(
                inventory_ratio,
                sigma_for_skew,
                gamma,
                time_horizon,
                market_params.flow_imbalance,
            );

            // FIX: Position amplifier for small positions (matches HJB path behavior)
            // The base skew formula multiplies by q (inventory_ratio), so small q → tiny skew
            // This amplifier ensures meaningful skew even with small positions to enable balanced fills
            let position_amplifier = if inventory_ratio.abs() < 0.1 && inventory_ratio.abs() > 0.01 {
                10.0 // 10x amplification for small positions (1-10% of max)
            } else if inventory_ratio.abs() <= 0.01 && inventory_ratio.abs() > 0.001 {
                5.0 // 5x amplification for very small positions (0.1-1% of max)
            } else {
                1.0
            };

            raw_skew * position_amplifier
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
        // The drift-adjusted skew replaces the ad-hoc momentum_skew_multiplier with a
        // first-principles approach based on:
        // - Drift rate (μ) estimated from momentum
        // - Continuation probability P(momentum continues)
        // - Directional variance (increased risk when opposed to momentum)
        //
        // When drift adjustment is enabled AND position opposes momentum:
        // - drift_urgency is added to base skew (accelerates position reduction)
        // - directional_variance_mult scales effective variance (higher risk)
        let (drift_urgency, momentum_skew_multiplier) = if market_params.use_drift_adjusted_skew
            && market_params.position_opposes_momentum
            && market_params.urgency_score > 0.5
        {
            // Use first-principles drift urgency from MarketParams
            // The drift_urgency is computed by HJB controller using:
            // - sensitivity × drift_rate × P(continue) × |q| × T
            let drift_urg = market_params.hjb_drift_urgency;

            // Variance multiplier also affects the effective gamma
            // Higher variance when opposed = more risk aversion = wider spreads
            let var_mult = market_params.directional_variance_mult;

            if market_params.urgency_score > 1.5 {
                debug!(
                    position = %format!("{:.4}", position),
                    momentum_bps = %format!("{:.1}", market_params.momentum_bps),
                    p_continue = %format!("{:.2}", market_params.p_momentum_continue),
                    drift_urgency_bps = %format!("{:.2}", drift_urg * 10000.0),
                    variance_mult = %format!("{:.2}", var_mult),
                    urgency_score = %format!("{:.1}", market_params.urgency_score),
                    "DRIFT-ADJUSTED SKEW: Position opposes momentum with high continuation prob"
                );
            }

            // Convert variance multiplier to skew multiplier
            // Higher variance = need stronger skew for same risk reduction
            (drift_urg, var_mult.sqrt())
        } else {
            // Fallback to legacy momentum_skew_multiplier (ad-hoc approach)
            let pos_direction = position.signum();
            let momentum_direction =
                if market_params.falling_knife_score > market_params.rising_knife_score {
                    -1.0 // Falling
                } else if market_params.rising_knife_score > market_params.falling_knife_score {
                    1.0 // Rising
                } else {
                    0.0 // Neutral
                };

            let momentum_severity = market_params
                .falling_knife_score
                .max(market_params.rising_knife_score);

            let is_opposed = pos_direction * momentum_direction < 0.0;

            let legacy_mult = if is_opposed && momentum_severity > 0.5 {
                let amplification = 1.0 + (momentum_severity / 3.0).min(1.0);
                if momentum_severity > 1.0 {
                    debug!(
                        position = %format!("{:.4}", position),
                        falling_knife = %format!("{:.2}", market_params.falling_knife_score),
                        rising_knife = %format!("{:.2}", market_params.rising_knife_score),
                        skew_amplifier = %format!("{:.2}x", amplification),
                        "Legacy momentum-opposed position: amplifying inventory skew"
                    );
                }
                amplification
            } else {
                1.0
            };

            (0.0, legacy_mult)
        };

        // === 6. COMBINED SKEW WITH TIER 2 ADJUSTMENTS ===
        // Combine all skew components:
        // - base_skew: GLFT inventory skew × flow modifier (exp(-β × alignment))
        // - drift_urgency: First-principles urgency from momentum-position opposition (HJB)
        // - hawkes_skew: Hawkes flow-based directional adjustment
        // - funding_skew: Perpetual funding cost pressure
        // - momentum amplification: variance-derived multiplier when opposed

        // === 6.0: PROACTIVE DIRECTIONAL SKEW (Small Fish Strategy) ===
        // When inventory is LOW, add proactive skew to BUILD position with momentum
        // When inventory is HIGH, inventory skew dominates to REDUCE position
        // The blend is additive: inventory_skew + proactive contribution
        let proactive_skew = self.proactive_directional_skew(market_params);

        // Inventory-reactive skew (existing behavior)
        // RL omega multiplier: allows learned skew scaling.
        // Clamped to [0.1, 10.0] to prevent sign-flipping or blow-ups.
        // Default 1.0 = no-op until RL is explicitly enabled.
        let rl_omega_mult = market_params.rl_omega_multiplier.clamp(0.1, 10.0);
        let inventory_skew = (base_skew + drift_urgency + hawkes_skew + funding_skew)
            * momentum_skew_multiplier
            * rl_omega_mult;

        // Blend: inventory skew always applies, proactive skew fades as inventory grows
        // inventory_weight: 0 at zero inventory → 1 at max position
        let inventory_weight = (position.abs() / effective_max_position).min(1.0);

        // Proactive skew is additive when inventory is low, fades to zero at max inventory
        // This ensures inventory skew always applies (preserves existing behavior)
        // while adding proactive component when we have room to build position
        let skew = inventory_skew + proactive_skew * (1.0 - inventory_weight);

        if proactive_skew.abs() > 1e-8 {
            tracing::info!(
                proactive_skew_bps = %format!("{:.2}", proactive_skew * 10000.0),
                inventory_skew_bps = %format!("{:.2}", inventory_skew * 10000.0),
                inventory_weight = %format!("{:.2}", inventory_weight),
                proactive_contribution_bps = %format!("{:.2}", proactive_skew * (1.0 - inventory_weight) * 10000.0),
                blended_skew_bps = %format!("{:.2}", skew * 10000.0),
                momentum_bps = %format!("{:.2}", market_params.momentum_bps),
                p_continue = %format!("{:.2}", market_params.p_momentum_continue),
                "Proactive skew active (Small Fish)"
            );
        }

        // Calculate flow modifier for logging (same as in inventory_skew_with_flow)
        let flow_alignment = inventory_ratio.signum() * market_params.flow_imbalance;
        let flow_modifier = (-self.risk_config.flow_sensitivity * flow_alignment).exp();

        // FIX: Enhanced debug logging to trace skew calculation
        // Log whenever there's a meaningful position to help diagnose balanced fills
        if position.abs() > 0.001 {
            debug!(
                position = %format!("{:.6}", position),
                inventory_ratio = %format!("{:.4}", inventory_ratio),
                sigma_for_skew = %format!("{:.8}", sigma_for_skew),
                skew_bps = %format!("{:.2}", base_skew * 10000.0),
                hjb_optimal_skew = %format!("{:.8}", market_params.hjb_optimal_skew),
                use_hjb_skew = market_params.use_hjb_skew,
                gamma = %format!("{:.4}", gamma),
                time_horizon = %format!("{:.2}", time_horizon),
                flow_imbalance = %format!("{:.3}", market_params.flow_imbalance),
                "Skew calculation (position amplification applied)"
            );
            if skew.abs() < 1e-8 {
                debug!(
                    base_skew_raw = %format!("{:.8}", base_skew),
                    "SKEW ZERO WARNING: Non-zero position but zero skew - check inputs"
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

        // 1. Bandit multiplier: narrow range, still multiplicative (optimizer feedback)
        let bandit_mult = market_params.bandit_spread_multiplier.clamp(0.8, 1.5);

        // 2. Changepoint/AS widening: convert excess multiplier to additive bps
        // spread_widening_mult of 1.5 on 5 bps half-spread = 2.5 bps addon
        // Capped at 100% of base spread (i.e., at most doubles the base)
        let widening_excess = (market_params.spread_widening_mult - 1.0).clamp(0.0, 1.0);
        let widening_addon_bid = half_spread_bid * widening_excess;
        let widening_addon_ask = half_spread_ask * widening_excess;

        // 3. Quota shadow spread: already in bps, convert to price fraction
        // Capped at 50 bps to prevent quota pressure from dominating
        let quota_addon = (market_params.quota_shadow_spread_bps / 10_000.0).min(0.0050);

        // Compose: base × bandit (multiplicative) + addons (additive)
        // The base spread passes through at natural GLFT level; only addons are bounded.
        let half_spread_bid_widened = half_spread_bid * bandit_mult + widening_addon_bid + quota_addon;
        let half_spread_ask_widened = half_spread_ask * bandit_mult + widening_addon_ask + quota_addon;

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
            if (bid_delta_raw - target_bid).abs() > 1e-6 || (ask_delta_raw - target_ask).abs() > 1e-6 {
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
        } else if bid_delta_raw.max(ask_delta_raw) / bid_delta_raw.min(ask_delta_raw).max(1e-8) > max_asymmetry_ratio {
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
            liq_mult = %format!("{:.2}", market_params.liquidity_gamma_mult),
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
            flow_mod = %format!("{:.3}", flow_modifier),
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
            sigma_lev = %format!("{:.6}", sigma_for_skew),
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
            sigma_effective = %format!("{:.6}", sigma_for_skew),
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

        // === Apply cascade size reduction for graceful degradation ===
        // During moderate cascade (before quote pulling), reduce size gradually
        let buy_size_adjusted = buy_size_raw * market_params.cascade_size_factor;
        let sell_size_adjusted = sell_size_raw * market_params.cascade_size_factor;

        let buy_size = truncate_float(buy_size_adjusted, config.sz_decimals, false);
        let sell_size = truncate_float(sell_size_adjusted, config.sz_decimals, false);

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
        mp.calibration_gamma_mult = 1.0;
        mp.tail_risk_multiplier = 1.0;
        mp.rl_gamma_multiplier = 1.0;
        mp.rl_omega_multiplier = 1.0;
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
    // RL Gamma Multiplier Tests
    // ---------------------------------------------------------------

    #[test]
    fn test_rl_gamma_multiplier_default_noop() {
        let strategy = test_strategy();
        let mp = test_market_params();
        let position = 0.5;
        let max_position = 10.0;

        // Default rl_gamma_multiplier = 1.0 should be identical to not having it
        let gamma = strategy.effective_gamma(&mp, position, max_position);
        assert!(gamma > 0.0, "gamma must be positive");

        // Compare with explicit 1.0
        let mut mp2 = mp.clone();
        mp2.rl_gamma_multiplier = 1.0;
        let gamma2 = strategy.effective_gamma(&mp2, position, max_position);
        assert!(
            (gamma - gamma2).abs() < 1e-12,
            "rl_gamma_multiplier=1.0 should be no-op, got {} vs {}",
            gamma,
            gamma2
        );
    }

    #[test]
    fn test_rl_gamma_multiplier_gt1_increases_gamma() {
        let strategy = test_strategy();
        let mp_base = test_market_params();
        let position = 0.5;
        let max_position = 10.0;

        let gamma_base = strategy.effective_gamma(&mp_base, position, max_position);

        let mut mp_high = mp_base.clone();
        mp_high.rl_gamma_multiplier = 2.0;
        let gamma_high = strategy.effective_gamma(&mp_high, position, max_position);

        assert!(
            gamma_high > gamma_base,
            "rl_gamma_multiplier=2.0 should increase gamma: {} vs {}",
            gamma_high,
            gamma_base
        );
    }

    #[test]
    fn test_rl_gamma_multiplier_lt1_decreases_gamma() {
        let strategy = test_strategy();
        let mp_base = test_market_params();
        let position = 0.5;
        let max_position = 10.0;

        let gamma_base = strategy.effective_gamma(&mp_base, position, max_position);

        let mut mp_low = mp_base.clone();
        mp_low.rl_gamma_multiplier = 0.5;
        let gamma_low = strategy.effective_gamma(&mp_low, position, max_position);

        assert!(
            gamma_low < gamma_base,
            "rl_gamma_multiplier=0.5 should decrease gamma: {} vs {}",
            gamma_low,
            gamma_base
        );
    }

    #[test]
    fn test_rl_gamma_multiplier_clamped_to_safe_range() {
        let strategy = test_strategy();
        let mp_base = test_market_params();
        let position = 0.5;
        let max_position = 10.0;

        // Extreme high should be clamped to 10.0
        let mut mp_extreme_high = mp_base.clone();
        mp_extreme_high.rl_gamma_multiplier = 100.0;
        let gamma_extreme = strategy.effective_gamma(&mp_extreme_high, position, max_position);

        let mut mp_at_cap = mp_base.clone();
        mp_at_cap.rl_gamma_multiplier = 10.0;
        let gamma_at_cap = strategy.effective_gamma(&mp_at_cap, position, max_position);

        assert!(
            (gamma_extreme - gamma_at_cap).abs() < 1e-12,
            "rl_gamma_multiplier=100.0 should be clamped to 10.0: {} vs {}",
            gamma_extreme,
            gamma_at_cap
        );

        // Extreme low should be clamped to 0.1
        let mut mp_extreme_low = mp_base.clone();
        mp_extreme_low.rl_gamma_multiplier = 0.001;
        let gamma_extreme_low =
            strategy.effective_gamma(&mp_extreme_low, position, max_position);

        let mut mp_at_floor = mp_base.clone();
        mp_at_floor.rl_gamma_multiplier = 0.1;
        let gamma_at_floor = strategy.effective_gamma(&mp_at_floor, position, max_position);

        assert!(
            (gamma_extreme_low - gamma_at_floor).abs() < 1e-12,
            "rl_gamma_multiplier=0.001 should be clamped to 0.1: {} vs {}",
            gamma_extreme_low,
            gamma_at_floor
        );
    }

    // ---------------------------------------------------------------
    // RL Omega Multiplier Tests (Position Skew)
    // ---------------------------------------------------------------

    #[test]
    fn test_rl_omega_multiplier_default_noop() {
        let strategy = test_strategy();
        let mp = test_market_params();
        let config = test_quote_config();
        let target_liq = 1.0;

        // With default rl_omega_multiplier = 1.0, quotes should be identical
        let (bid1, ask1) = strategy.calculate_quotes(&config, 1.0, 10.0, target_liq, &mp);

        let mut mp2 = mp.clone();
        mp2.rl_omega_multiplier = 1.0;
        let (bid2, ask2) = strategy.calculate_quotes(&config, 1.0, 10.0, target_liq, &mp2);

        if let (Some(b1), Some(b2)) = (&bid1, &bid2) {
            assert!(
                (b1.price - b2.price).abs() < 1e-10,
                "bid prices differ with omega=1.0: {} vs {}",
                b1.price,
                b2.price
            );
        }
        if let (Some(a1), Some(a2)) = (&ask1, &ask2) {
            assert!(
                (a1.price - a2.price).abs() < 1e-10,
                "ask prices differ with omega=1.0: {} vs {}",
                a1.price,
                a2.price
            );
        }
    }

    #[test]
    fn test_rl_omega_multiplier_scales_skew() {
        // Test at the effective_gamma/skew level: verify the omega multiplier
        // affects the final quotes by using extreme vol and high-precision config.
        let strategy = test_strategy();
        let config = QuoteConfig {
            mid_price: 1000.0,
            decimals: 4,
            sz_decimals: 4,
            min_notional: 0.01,
        };
        let position = 8.0; // Near max for strong skew
        let max_position = 10.0;
        let target_liq = 1.0;

        let mut mp_base = test_market_params();
        mp_base.microprice = 1000.0;
        mp_base.sigma = 0.05; // 500 bps vol for large skew
        mp_base.sigma_effective = 0.05;
        mp_base.sigma_leverage_adjusted = 0.05;
        mp_base.arrival_intensity = 0.1; // Long holding time = bigger skew
        let (bid_base, _) =
            strategy.calculate_quotes(&config, position, max_position, target_liq, &mp_base);

        let mut mp_high = mp_base.clone();
        mp_high.rl_omega_multiplier = 5.0;
        let (bid_high, _) =
            strategy.calculate_quotes(&config, position, max_position, target_liq, &mp_high);

        // Higher omega amplifies inventory skew.
        // With long position, skew pushes bid lower; omega=5 should push it even lower.
        if let (Some(b_base), Some(b_high)) = (&bid_base, &bid_high) {
            assert!(
                (b_base.price - b_high.price).abs() > 0.001,
                "omega=5.0 should meaningfully change bid price vs omega=1.0: base={}, high={}",
                b_base.price,
                b_high.price
            );
        } else {
            // If either bid is None (skew pushed it negative), that also proves omega works
            let base_has_bid = bid_base.is_some();
            let high_has_bid = bid_high.is_some();
            assert!(
                base_has_bid != high_has_bid,
                "omega=5.0 should change bid availability: base_has={}, high_has={}",
                base_has_bid,
                high_has_bid
            );
        }
    }

    #[test]
    fn test_rl_omega_multiplier_clamped_to_safe_range() {
        let strategy = test_strategy();
        let config = test_quote_config();
        let position = 2.0;
        let max_position = 10.0;
        let target_liq = 1.0;

        // Extreme high should be clamped to 10.0
        let mut mp_extreme = test_market_params();
        mp_extreme.rl_omega_multiplier = 100.0;
        let (bid_extreme, _) =
            strategy.calculate_quotes(&config, position, max_position, target_liq, &mp_extreme);

        let mut mp_cap = test_market_params();
        mp_cap.rl_omega_multiplier = 10.0;
        let (bid_cap, _) =
            strategy.calculate_quotes(&config, position, max_position, target_liq, &mp_cap);

        if let (Some(b_ext), Some(b_cap)) = (&bid_extreme, &bid_cap) {
            assert!(
                (b_ext.price - b_cap.price).abs() < 1e-10,
                "omega=100.0 should be clamped to 10.0: {} vs {}",
                b_ext.price,
                b_cap.price
            );
        }

        // Extreme low should be clamped to 0.1
        let mut mp_low = test_market_params();
        mp_low.rl_omega_multiplier = 0.001;
        let (bid_low, _) =
            strategy.calculate_quotes(&config, position, max_position, target_liq, &mp_low);

        let mut mp_floor = test_market_params();
        mp_floor.rl_omega_multiplier = 0.1;
        let (bid_floor, _) =
            strategy.calculate_quotes(&config, position, max_position, target_liq, &mp_floor);

        if let (Some(b_low), Some(b_floor)) = (&bid_low, &bid_floor) {
            assert!(
                (b_low.price - b_floor.price).abs() < 1e-10,
                "omega=0.001 should be clamped to 0.1: {} vs {}",
                b_low.price,
                b_floor.price
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
        mp.pre_fill_spread_mult_bid = 1.0;
        mp.pre_fill_spread_mult_ask = 1.0;

        let (bid_with_tox, _ask_with_tox) =
            strategy.calculate_quotes(&config, 0.0, 10.0, 1.0, &mp);

        // Zero toxicity baseline
        let mut mp_zero = test_market_params();
        mp_zero.pre_fill_toxicity_bid = 0.0;
        mp_zero.pre_fill_toxicity_ask = 0.0;
        mp_zero.pre_fill_spread_mult_bid = 1.0;
        mp_zero.pre_fill_spread_mult_ask = 1.0;

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
        mp.pre_fill_spread_mult_bid = 1.0;
        mp.pre_fill_spread_mult_ask = 1.0;

        let (bid_tox, _) =
            strategy.calculate_quotes(&config, 0.0, 10.0, 1.0, &mp);

        let mut mp_zero = test_market_params();
        mp_zero.pre_fill_toxicity_bid = 0.0;
        mp_zero.pre_fill_toxicity_ask = 0.0;
        mp_zero.pre_fill_spread_mult_bid = 1.0;
        mp_zero.pre_fill_spread_mult_ask = 1.0;

        let (bid_no, _) =
            strategy.calculate_quotes(&config, 0.0, 10.0, 1.0, &mp_zero);

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
        mp.pre_fill_spread_mult_bid = 1.0;
        mp.pre_fill_spread_mult_ask = 1.0;

        let (bid_tox, _) =
            strategy.calculate_quotes(&config, 0.0, 10.0, 1.0, &mp);

        let mut mp_zero = test_market_params();
        mp_zero.pre_fill_toxicity_bid = 0.0;
        mp_zero.pre_fill_toxicity_ask = 0.0;
        mp_zero.pre_fill_spread_mult_bid = 1.0;
        mp_zero.pre_fill_spread_mult_ask = 1.0;

        let (bid_no, _) =
            strategy.calculate_quotes(&config, 0.0, 10.0, 1.0, &mp_zero);

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

    #[test]
    fn test_log_odds_as_fallback_multiplicative() {
        // When use_log_odds_as = false, should use legacy multiplicative path
        let strategy = test_strategy_with_log_odds(false, 15.0);
        let config = test_quote_config();

        let mut mp = test_market_params();
        mp.pre_fill_toxicity_bid = 0.5;
        mp.pre_fill_toxicity_ask = 0.5;
        mp.pre_fill_spread_mult_bid = 2.0; // 2x multiplier
        mp.pre_fill_spread_mult_ask = 2.0;

        let (bid_mult, _) =
            strategy.calculate_quotes(&config, 0.0, 10.0, 1.0, &mp);

        let mut mp_no = test_market_params();
        mp_no.pre_fill_toxicity_bid = 0.0;
        mp_no.pre_fill_toxicity_ask = 0.0;
        mp_no.pre_fill_spread_mult_bid = 1.0;
        mp_no.pre_fill_spread_mult_ask = 1.0;

        let (bid_no, _) =
            strategy.calculate_quotes(&config, 0.0, 10.0, 1.0, &mp_no);

        // Multiplicative path: spread roughly doubles from 2x multiplier
        if let (Some(b_mult), Some(b_no)) = (&bid_mult, &bid_no) {
            let mid = config.mid_price;
            let widening_bps = ((b_no.price - b_mult.price) / mid) * 10000.0;
            assert!(
                widening_bps > 0.5,
                "Legacy multiplicative (2x mult) should widen spread, got {:.2} bps",
                widening_bps
            );
        }
    }

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

        let bid_half = strategy.half_spread_with_drift(
            gamma, kappa, sigma, time_horizon, drift_rate, true,
        );
        let ask_half = strategy.half_spread_with_drift(
            gamma, kappa, sigma, time_horizon, drift_rate, false,
        );
        let base = strategy.half_spread(gamma, kappa, sigma, time_horizon);

        // Positive drift: bid should be wider than base (buying into uptrend is risky)
        assert!(
            bid_half > base,
            "bid half-spread should widen with positive drift: bid={bid_half}, base={base}"
        );
        // Positive drift: ask should be tighter than base (selling into uptrend is favorable)
        assert!(
            ask_half < base,
            "ask half-spread should tighten with positive drift: ask={ask_half}, base={base}"
        );
    }

    #[test]
    fn test_drift_zero_is_symmetric() {
        let strategy = test_strategy();
        let gamma = 0.15;
        let kappa = 5000.0;
        let sigma = 0.001;
        let time_horizon = 60.0;

        let bid_half = strategy.half_spread_with_drift(
            gamma, kappa, sigma, time_horizon, 0.0, true,
        );
        let ask_half = strategy.half_spread_with_drift(
            gamma, kappa, sigma, time_horizon, 0.0, false,
        );
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

        let ask_half = strategy.half_spread_with_drift(
            gamma, kappa, sigma, time_horizon, drift_rate, false,
        );

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
        assert!(gamma < 10.0, "min_gamma should be reasonable, got {}", gamma);
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

        // Extreme widening: should be additive, not multiplicative
        mp.spread_widening_mult = 5.0; // Clamped: excess = min(5-1, 1.0) = 1.0 → at most doubles
        mp.bandit_spread_multiplier = 2.0; // Clamped to 1.5
        mp.quota_shadow_spread_bps = 10.0; // 10 bps quota pressure

        let (bid_wide, ask_wide) = strategy.calculate_quotes(&config, 0.0, 100.0, 1.0, &mp);

        // Both quotes should exist
        assert!(bid_wide.is_some(), "Bid should exist even with extreme widening");
        assert!(ask_wide.is_some(), "Ask should exist even with extreme widening");

        // Verify additive not multiplicative:
        // Old multiplicative: base × 5.0 × 2.0 = 10x
        // New additive: base × 1.5 (bandit) + base × 1.0 (widening) + 10bps (quota)
        //             = 2.5 × base + 10bps ≈ 3.5x base (much less than 10x)
        if let (Some(b_base), Some(b_wide)) = (bid_base, bid_wide) {
            let mid = mp.microprice;
            let base_spread_bps = (mid - b_base.price) / mid * 10000.0;
            let wide_spread_bps = (mid - b_wide.price) / mid * 10000.0;
            let ratio = wide_spread_bps / base_spread_bps.max(0.01);
            assert!(
                ratio < 5.0,
                "Widened spread should be < 5x base (additive), got {:.1}x ({:.1} vs {:.1} bps)",
                ratio, wide_spread_bps, base_spread_bps
            );
        }
    }
}
