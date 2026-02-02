//! GLFT (Guéant-Lehalle-Fernandez-Tapia) optimal market making strategy.

use tracing::debug;

use crate::{round_to_significant_and_decimal, truncate_float, EPSILON};

use crate::market_maker::config::{Quote, QuoteConfig};

use super::{
    CalibratedRiskModel, KellySizer, MarketParams, QuotingStrategy, RiskConfig, RiskFeatures,
    RiskModelConfig,
};

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
        }
    }

    /// Create a new GLFT strategy with full risk configuration.
    pub fn with_config(risk_config: RiskConfig) -> Self {
        Self {
            risk_model: CalibratedRiskModel::with_gamma_base(risk_config.gamma_base),
            risk_model_config: RiskModelConfig::default(),
            kelly_sizer: KellySizer::default(),
            risk_config,
        }
    }

    /// Create a new GLFT strategy with full configuration including calibrated risk model.
    pub fn with_full_config(
        risk_config: RiskConfig,
        risk_model_config: RiskModelConfig,
        kelly_sizer: KellySizer,
    ) -> Self {
        Self {
            risk_model: CalibratedRiskModel::with_gamma_base(risk_config.gamma_base),
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
        let gamma_final = gamma_with_calib * market_params.tail_risk_multiplier;

        let gamma_clamped = gamma_final.clamp(cfg.gamma_min, cfg.gamma_max);

        // Log comparison for shadow mode validation
        if self.risk_model_config.use_calibrated_risk_model || blend > 0.0 {
            debug!(
                gamma_legacy = %format!("{:.4}", gamma_legacy),
                gamma_calibrated = %format!("{:.4}", gamma_calibrated),
                blend = %format!("{:.2}", blend),
                gamma_base = %format!("{:.4}", gamma_base),
                calib_mult = %format!("{:.3}", market_params.calibration_gamma_mult),
                tail_mult = %format!("{:.3}", market_params.tail_risk_multiplier),
                gamma_final = %format!("{:.4}", gamma_clamped),
                "Gamma: legacy vs calibrated comparison"
            );
        } else {
            debug!(
                gamma_base = %format!("{:.3}", cfg.gamma_base),
                gamma_raw = %format!("{:.4}", gamma_legacy),
                calib_mult = %format!("{:.3}", market_params.calibration_gamma_mult),
                tail_mult = %format!("{:.3}", market_params.tail_risk_multiplier),
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
        let toxicity_scalar = if market_params.jump_ratio <= cfg.toxicity_threshold {
            1.0
        } else {
            1.0 + cfg.toxicity_sensitivity * (market_params.jump_ratio - 1.0)
        };

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
    /// Compute proactive directional skew based on momentum predictions.
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
        // Falls back to static max_position if margin state hasn't been refreshed yet
        let effective_max_position = market_params.effective_max_position(max_position);

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

        // === 1a. KAPPA: Adaptive vs Legacy ===
        // When adaptive spreads enabled: use blended book/own-fill kappa
        // When disabled: use book-only kappa with AS adjustment
        //
        // KEY FIX: Use `adaptive_can_estimate` - our Bayesian priors give reasonable
        // kappa estimates immediately (κ=2500 prior for liquid markets).
        let (kappa, kappa_bid, kappa_ask) = if market_params.use_adaptive_spreads
            && market_params.adaptive_can_estimate
        {
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

        // === 2d. SPREAD FLOOR: Adaptive vs Static ===
        // When adaptive spreads enabled: use learned floor from Bayesian AS estimation
        // When disabled: use static RiskConfig floor + latency/tick constraints
        //
        // KEY FIX: Use `adaptive_can_estimate` - our Bayesian prior gives reasonable
        // floor estimates immediately (fees + 3bps AS prior + safety margin ≈ 8-10 bps).
        let effective_floor = if market_params.use_adaptive_spreads
            && market_params.adaptive_can_estimate
        {
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
        let inventory_skew =
            (base_skew + drift_urgency + hawkes_skew + funding_skew) * momentum_skew_multiplier;

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

        // Apply spread widening multiplier from quote gate (pending changepoint confirmation)
        let half_spread_bid_widened = half_spread_bid * market_params.spread_widening_mult;
        let half_spread_ask_widened = half_spread_ask * market_params.spread_widening_mult;

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
}
