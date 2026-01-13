//! GLFT (Guéant-Lehalle-Fernandez-Tapia) optimal market making strategy.

use tracing::debug;

use crate::{round_to_significant_and_decimal, truncate_float, EPSILON};

use crate::market_maker::config::{Quote, QuoteConfig};

use super::{MarketParams, QuotingStrategy, RiskConfig};

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
/// ## Dynamic Risk Aversion:
/// ```text
/// γ_effective = γ_base × vol_scalar × toxicity_scalar × inventory_scalar
/// ```
///
/// This preserves GLFT's structure while adapting to:
/// - Volatility regime (higher σ → more conservative)
/// - Toxicity (high RV/BV → informed flow → widen)
/// - Inventory utilization (near limits → reduce risk)
#[derive(Debug, Clone)]
pub struct GLFTStrategy {
    /// Risk configuration for dynamic γ calculation
    pub risk_config: RiskConfig,
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
        }
    }

    /// Create a new GLFT strategy with full risk configuration.
    pub fn with_config(risk_config: RiskConfig) -> Self {
        Self { risk_config }
    }

    /// Calculate effective γ based on current market conditions.
    ///
    /// γ_effective = γ_base × vol_scalar × toxicity_scalar × inventory_scalar × hawkes_scalar × ...
    ///
    /// Each scalar has mathematical justification - no ad-hoc multipliers.
    fn effective_gamma(
        &self,
        market_params: &MarketParams,
        position: f64,
        max_position: f64,
    ) -> f64 {
        let cfg = &self.risk_config;

        // === VOLATILITY SCALING ===
        // Higher realized vol → more risk per unit inventory
        let vol_ratio = market_params.sigma_effective / cfg.sigma_baseline.max(1e-9);
        let vol_scalar = if vol_ratio <= 1.0 {
            1.0 // Don't reduce γ in low vol (often precedes spikes)
        } else {
            let raw = 1.0 + cfg.volatility_weight * (vol_ratio - 1.0);
            raw.min(cfg.max_volatility_multiplier)
        };

        // === TOXICITY SCALING ===
        // High RV/BV indicates informed flow
        let toxicity_scalar = if market_params.jump_ratio <= cfg.toxicity_threshold {
            1.0
        } else {
            1.0 + cfg.toxicity_sensitivity * (market_params.jump_ratio - 1.0)
        };

        // === INVENTORY SCALING ===
        // Near position limits → less room for error
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

        // === REMOVED: VOLATILITY REGIME SCALING ===
        // FIRST PRINCIPLES: This was redundant with vol_scalar above.
        // Both responded to volatility, causing double-scaling:
        //   - vol_scalar: 1 + weight × (σ/σ_baseline - 1) [principled]
        //   - regime_scalar: hardcoded 0.8/1.0/1.5/2.5 [ad-hoc]
        //
        // Volatility is already properly handled via vol_scalar which uses
        // the continuous relationship between σ and γ scaling.
        // The discrete regime buckets (Low/Normal/High/Extreme) with arbitrary
        // multipliers lacked mathematical derivation.

        // === HAWKES ACTIVITY SCALING (Tier 2) ===
        // FIRST PRINCIPLES: High order flow intensity indicates potential informed trading.
        // Instead of arbitrary thresholds (0.8, 0.9) with hardcoded multipliers (1.2, 1.5),
        // use a continuous relationship derived from arrival intensity ratio:
        //
        //   hawkes_scalar = 1 + sensitivity × max(0, percentile - baseline)
        //
        // This is mathematically justified: when λ_current >> λ_baseline,
        // the probability of informed flow increases, warranting wider spreads.
        // The sensitivity parameter controls how aggressively we respond.
        //
        // With baseline=0.5 (median) and sensitivity=2.0:
        //   - 50th percentile: scalar = 1.0 (no adjustment)
        //   - 80th percentile: scalar = 1.0 + 2.0 × 0.3 = 1.6
        //   - 95th percentile: scalar = 1.0 + 2.0 × 0.45 = 1.9
        let hawkes_baseline = 0.5; // Median as baseline
        let hawkes_sensitivity = 2.0; // How much to scale per unit above baseline
        let hawkes_scalar = 1.0
            + hawkes_sensitivity
                * (market_params.hawkes_activity_percentile - hawkes_baseline).max(0.0);

        // === TIME-OF-DAY SCALING ===
        // Trade history showed -13 to -15 bps edge during 06-08, 14-15 UTC
        let time_scalar = cfg.time_of_day_multiplier();

        // === BOOK DEPTH SCALING (FIRST PRINCIPLES) ===
        // Thin order books → harder to exit → higher risk → higher γ
        // This replaces the arbitrary stochastic_spread_multiplier
        let book_depth_scalar = cfg.book_depth_multiplier(market_params.near_touch_depth_usd);

        // === WARMUP UNCERTAINTY SCALING (FIRST PRINCIPLES) ===
        // Use Bayesian posterior CI width for principled uncertainty scaling.
        // Wider credible interval → more parameter uncertainty → higher gamma.
        // Falls back to progress-based scaling if CI data unavailable.
        let warmup_scalar = if market_params.kappa_ci_width > 0.0 {
            // CI-based: directly reflects posterior uncertainty
            cfg.warmup_multiplier_from_ci(market_params.kappa_ci_width)
        } else {
            // Fallback: progress-based linear decay (legacy)
            cfg.warmup_multiplier(market_params.adaptive_warmup_progress)
        };

        // === CALIBRATION FILL-HUNGRY SCALING (FIRST PRINCIPLES) ===
        // During calibration, reduce gamma to attract fills for parameter estimation.
        // calibration_gamma_mult ∈ [0.3, 1.0]: lower = tighter quotes = more fills
        // Once calibrated, calibration_gamma_mult = 1.0 (no adjustment).
        let calibration_scalar = market_params.calibration_gamma_mult;

        // === COMBINE AND CLAMP ===
        // FIRST PRINCIPLES: Each scalar has mathematical justification:
        //   - vol_scalar: σ/σ_baseline relationship (continuous)
        //   - toxicity_scalar: RV/BV informed flow detection (continuous)
        //   - inventory_scalar: quadratic near limits (optimal control)
        //   - hawkes_scalar: order flow intensity (derived from λ)
        //   - time_scalar: toxic hour edge (measured from data)
        //   - book_depth_scalar: exit slippage (market microstructure)
        //   - warmup_scalar: parameter uncertainty (Bayesian)
        //   - calibration_scalar: fill rate targeting (controller)
        let gamma_effective = cfg.gamma_base
            * vol_scalar
            * toxicity_scalar
            * inventory_scalar
            * hawkes_scalar
            * time_scalar
            * book_depth_scalar
            * warmup_scalar
            * calibration_scalar;
        let gamma_clamped = gamma_effective.clamp(cfg.gamma_min, cfg.gamma_max);

        // Log gamma component breakdown for debugging strategy behavior
        debug!(
            gamma_base = %format!("{:.3}", cfg.gamma_base),
            vol_scalar = %format!("{:.3}", vol_scalar),
            tox_scalar = %format!("{:.3}", toxicity_scalar),
            inv_scalar = %format!("{:.3}", inventory_scalar),
            hawkes_scalar = %format!("{:.3}", hawkes_scalar),
            time_scalar = %format!("{:.3}", time_scalar),
            book_scalar = %format!("{:.3}", book_depth_scalar),
            warmup_scalar = %format!("{:.3}", warmup_scalar),
            calibration_scalar = %format!("{:.3}", calibration_scalar),
            gamma_raw = %format!("{:.4}", gamma_effective),
            gamma_clamped = %format!("{:.4}", gamma_clamped),
            "Gamma component breakdown"
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

    /// Correct GLFT half-spread formula with fee recovery:
    ///
    /// δ* = (1/γ) × ln(1 + γ/κ) + f_maker
    ///
    /// The fee term comes from the modified HJB equation:
    /// dW = (δ - f_maker) × dN, so optimal δ* = δ_GLFT + f_maker
    ///
    /// This is market-driven: when κ drops (thin book), spread widens automatically.
    /// When κ rises (deep book), spread tightens. Fee ensures minimum profitable spread.
    fn half_spread(&self, gamma: f64, kappa: f64) -> f64 {
        let ratio = gamma / kappa;
        let glft_spread = if ratio > 1e-9 && gamma > 1e-9 {
            (1.0 / gamma) * (1.0 + ratio).ln()
        } else {
            // When γ/κ → 0, use Taylor expansion: ln(1+x) ≈ x
            // δ ≈ (1/γ) × (γ/κ) = 1/κ
            1.0 / kappa.max(1.0)
        };

        // Add maker fee to ensure profitability (first-principles HJB modification)
        glft_spread + self.risk_config.maker_fee_rate
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
        // δ_bid = (1/γ) × ln(1 + γ/κ_bid) - wider if κ_bid < κ_ask (sell pressure)
        // δ_ask = (1/γ) × ln(1 + γ/κ_ask) - wider if κ_ask < κ_bid (buy pressure)
        let mut half_spread_bid = self.half_spread(gamma, kappa_bid);
        let mut half_spread_ask = self.half_spread(gamma, kappa_ask);

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
            market_params.hjb_optimal_skew
        } else {
            // Flow-dampened inventory skew: base_skew × exp(-β × flow_alignment)
            // Uses flow_imbalance to dampen skew when aligned with flow (don't fight momentum)
            // and amplify skew when opposed to flow (reduce risk faster)
            self.inventory_skew_with_flow(
                inventory_ratio,
                sigma_for_skew,
                gamma,
                time_horizon,
                market_params.flow_imbalance,
            )
        };

        // === 3a. HAWKES FLOW SKEWING (Tier 2) with Informed Flow Adjustment ===
        // Use Hawkes-derived flow imbalance for additional directional adjustment
        // High activity + imbalance → stronger signal than simple flow imbalance
        //
        // When p_informed is high, reduce reliance on flow signals since they're
        // likely from informed traders (adverse selection risk) rather than noise traders.
        // flow_weight = 1 - p_informed (e.g., 80% noise → 80% weight)
        let flow_weight = if market_params.flow_decomp_confidence > 0.3 {
            (1.0 - market_params.p_informed).clamp(0.3, 1.0) // Min 30% weight
        } else {
            1.0 // Not confident in p_informed, use full flow signal
        };
        let hawkes_skew = if market_params.hawkes_activity_percentile > 0.7 {
            // Significant activity - use Hawkes imbalance for flow prediction
            // hawkes_imbalance > 0 = more buy pressure → skew asks tighter (encourage selling)
            //
            // DERIVATION of 0.00005 coefficient:
            // - Target: ~0.5 bps skew per 0.1 imbalance at 100% activity
            // - imbalance ∈ [-1, 1], so max |skew| = 0.00005 × 1.0 × 1.0 = 0.5 bps
            // - This is calibrated to be < half-spread to avoid skew flipping quotes
            // - Scaled by activity percentile (more confident signal at higher activity)
            // - Weighted by noise probability (discount informed flow signals)
            //
            // TODO: Could be derived from E[Δp | imbalance] regression on historical data
            market_params.hawkes_imbalance
                * 0.00005
                * market_params.hawkes_activity_percentile
                * flow_weight
        } else {
            0.0 // Low activity - don't trust flow signal
        };

        // === 3b. FUNDING COST ADJUSTMENT (Tier 2) ===
        // Incorporate perpetual funding cost into inventory decision
        // Positive funding + long = cost → skew to reduce long
        // Negative funding + short = cost → skew to reduce short
        let funding_skew = if market_params.funding_rate.abs() > 0.0001 {
            // Significant funding rate (> 1bp/hour)
            // predicted_funding_cost is signed: positive = cost to current position
            // If cost is positive and we're long, skew to sell (negative skew adds to bid_delta)
            // If cost is positive and we're short, skew to buy (positive skew reduces bid_delta)
            // Simplified: position * funding_rate gives us direction
            let funding_pressure = position.signum() * market_params.funding_rate;
            //
            // DERIVATION of 0.01 coefficient:
            // - funding_rate is in fraction per hour (e.g., 0.0001 = 1 bp/hour)
            // - We want ~0.1 bps skew per 1 bp/hour funding rate
            // - 0.01 × 0.0001 (1bp rate) = 0.000001 = 0.01 bps skew (as fraction)
            // - Actually: 0.01 × 0.0001 = 1e-6 fractional = 0.01 bps - too small
            // - With funding_rate = 0.001 (10 bp/hour), skew = 0.01 × 0.001 = 1e-5 = 0.1 bps
            // - This is conservative: funding is long-term cost, shouldn't dominate short-term skew
            //
            // TODO: Could derive from: skew = funding_cost × expected_holding_time / spread
            funding_pressure * 0.01
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
        let skew =
            (base_skew + drift_urgency + hawkes_skew + funding_skew) * momentum_skew_multiplier;

        // Calculate flow modifier for logging (same as in inventory_skew_with_flow)
        let flow_alignment = inventory_ratio.signum() * market_params.flow_imbalance;
        let flow_modifier = (-self.risk_config.flow_sensitivity * flow_alignment).exp();

        // === 6a. ASYMMETRIC BID/ASK DELTAS ===
        // Use directional half-spreads: κ_bid ≠ κ_ask when flow is directional
        // - Bid delta uses half_spread_bid (wider when sell pressure = low κ_bid)
        // - Ask delta uses half_spread_ask (wider when buy pressure = low κ_ask)
        let bid_delta = half_spread_bid + skew;
        let ask_delta = (half_spread_ask - skew).max(0.0);

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
