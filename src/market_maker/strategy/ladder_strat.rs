//! GLFT Ladder Strategy - multi-level quoting with depth-dependent sizing.

use tracing::debug;

use crate::EPSILON;

use crate::market_maker::config::{Quote, QuoteConfig};
use crate::market_maker::estimator::VolatilityRegime;
use crate::market_maker::quoting::{
    ConstrainedLadderOptimizer, KellyStochasticParams, Ladder, LadderConfig, LadderParams,
    LevelOptimizationParams,
};

use super::{MarketParams, QuotingStrategy, RiskConfig};

/// GLFT Ladder Strategy - multi-level quoting with depth-dependent sizing.
///
/// Generates K levels per side with:
/// - Geometric or linear depth spacing
/// - Size allocation proportional to λ(δ) × SC(δ) (fill intensity × spread capture)
/// - GLFT inventory skew applied to entire ladder
///
/// This strategy uses the same gamma calculation as GLFTStrategy but
/// distributes liquidity across multiple price levels instead of just
/// quoting at the touch.
#[derive(Debug, Clone)]
pub struct LadderStrategy {
    /// Risk configuration (same as GLFTStrategy)
    pub risk_config: RiskConfig,
    /// Ladder-specific configuration
    pub ladder_config: LadderConfig,
}

impl LadderStrategy {
    /// Create a new ladder strategy with default configs.
    pub fn new(gamma_base: f64) -> Self {
        Self {
            risk_config: RiskConfig {
                gamma_base: gamma_base.clamp(0.01, 10.0),
                ..Default::default()
            },
            ladder_config: LadderConfig::default(),
        }
    }

    /// Create a new ladder strategy with custom configs.
    pub fn with_config(risk_config: RiskConfig, ladder_config: LadderConfig) -> Self {
        Self {
            risk_config,
            ladder_config,
        }
    }

    /// Calculate effective γ based on current market conditions.
    /// (Same logic as GLFTStrategy for consistency)
    fn effective_gamma(
        &self,
        market_params: &MarketParams,
        position: f64,
        max_position: f64,
    ) -> f64 {
        let cfg = &self.risk_config;

        // Volatility scaling
        let vol_ratio = market_params.sigma_effective / cfg.sigma_baseline.max(1e-9);
        let vol_scalar = if vol_ratio <= 1.0 {
            1.0
        } else {
            let raw = 1.0 + cfg.volatility_weight * (vol_ratio - 1.0);
            raw.min(cfg.max_volatility_multiplier)
        };

        // Toxicity scaling
        let toxicity_scalar = if market_params.jump_ratio <= cfg.toxicity_threshold {
            1.0
        } else {
            1.0 + cfg.toxicity_sensitivity * (market_params.jump_ratio - 1.0)
        };

        // Inventory scaling
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

        // Volatility regime scaling
        let regime_scalar = match market_params.volatility_regime {
            VolatilityRegime::Low => 0.8,
            VolatilityRegime::Normal => 1.0,
            VolatilityRegime::High => 1.5,
            VolatilityRegime::Extreme => 2.5,
        };

        // Hawkes activity scaling
        let hawkes_scalar = if market_params.hawkes_activity_percentile > 0.9 {
            1.5
        } else if market_params.hawkes_activity_percentile > 0.8 {
            1.2
        } else {
            1.0
        };

        let gamma_effective = cfg.gamma_base
            * vol_scalar
            * toxicity_scalar
            * inventory_scalar
            * regime_scalar
            * hawkes_scalar;
        gamma_effective.clamp(cfg.gamma_min, cfg.gamma_max)
    }

    /// Calculate holding time from arrival intensity.
    fn holding_time(&self, arrival_intensity: f64) -> f64 {
        let safe_intensity = arrival_intensity.max(0.01);
        (1.0 / safe_intensity).min(self.risk_config.max_holding_time)
    }

    /// Generate full ladder from market params.
    ///
    /// This is the main method that creates a multi-level quote ladder.
    pub fn generate_ladder(
        &self,
        config: &QuoteConfig,
        position: f64,
        max_position: f64,
        target_liquidity: f64,
        market_params: &MarketParams,
    ) -> Ladder {
        // Circuit breaker: pull all quotes during cascade
        if market_params.should_pull_quotes {
            return Ladder::default();
        }

        // Calculate effective gamma (includes all scaling factors)
        let base_gamma = self.effective_gamma(market_params, position, max_position);
        let gamma_with_liq = base_gamma * market_params.liquidity_gamma_mult;
        let gamma = gamma_with_liq * market_params.tail_risk_multiplier;

        // AS-adjusted kappa: same logic as GLFTStrategy
        // κ_effective = κ̂ × (1 - α), where α = P(informed | fill)
        let alpha = market_params.predicted_alpha.min(0.5);
        let kappa = market_params.kappa * (1.0 - alpha);

        let time_horizon = self.holding_time(market_params.arrival_intensity);

        let inventory_ratio = if max_position > EPSILON {
            (position / max_position).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // Convert AS spread adjustment to bps for ladder generation
        let as_at_touch_bps = if market_params.as_warmed_up {
            market_params.as_spread_adjustment * 10000.0
        } else {
            0.0 // Use zero AS until warmed up
        };

        // Apply cascade size reduction
        let adjusted_size = target_liquidity * market_params.cascade_size_factor;

        let params = LadderParams {
            mid_price: market_params.microprice,
            sigma: market_params.sigma,
            kappa, // Use AS-adjusted kappa
            arrival_intensity: market_params.arrival_intensity,
            as_at_touch_bps,
            total_size: adjusted_size,
            inventory_ratio,
            gamma,
            time_horizon,
            decimals: config.decimals,
            sz_decimals: config.sz_decimals,
            min_notional: config.min_notional,
            depth_decay_as: market_params.depth_decay_as.clone(),
        };

        // === STOCHASTIC MODULE: Constrained Ladder Optimization ===
        // Solves: max Σ λ(δᵢ) × SC(δᵢ) × sᵢ  (expected profit)
        // s.t. Σ sᵢ × margin_per_unit ≤ margin_available (margin constraint)
        //      Σ sᵢ ≤ max_position (position constraint)
        if market_params.use_constrained_optimizer {
            // 1. Available margin comes from the exchange (already accounts for position margin)
            // NOTE: margin_available is already net of position margin, don't double-count!
            let leverage = market_params.leverage.max(1.0);
            let available_margin = market_params.margin_available;
            let available_position = (max_position - position.abs()).max(0.0);

            // 2. Generate ladder to get depth levels and prices
            let mut ladder = Ladder::generate(&self.ladder_config, &params);

            // 3. Create constrained optimizer
            let optimizer = ConstrainedLadderOptimizer::new(
                available_margin,
                available_position,
                self.ladder_config.min_level_size,
                config.min_notional,
                market_params.microprice,
                leverage,
            );

            // 4. Build LevelOptimizationParams for bids
            let bid_level_params: Vec<_> = ladder
                .bids
                .iter()
                .map(|level| {
                    let fill_intensity =
                        fill_intensity_at_depth(level.depth_bps, params.sigma, params.kappa);
                    let spread_capture = spread_capture_at_depth(
                        level.depth_bps,
                        &params,
                        self.ladder_config.fees_bps,
                    );
                    let adverse_selection = adverse_selection_at_depth(level.depth_bps, &params);
                    LevelOptimizationParams {
                        depth_bps: level.depth_bps,
                        fill_intensity,
                        spread_capture,
                        margin_per_unit: market_params.microprice / leverage,
                        adverse_selection,
                    }
                })
                .collect();

            // 5. Optimize bid sizes (use Kelly-Stochastic if enabled)
            if !bid_level_params.is_empty() {
                let allocation = if market_params.use_kelly_stochastic {
                    // Apply volatility regime adjustment to Kelly fraction
                    // High vol → more conservative (lower Kelly)
                    // Low vol → more aggressive (higher Kelly)
                    let regime_multiplier = market_params.volatility_regime.kelly_fraction_multiplier();
                    let dynamic_kelly = (market_params.kelly_fraction * regime_multiplier).clamp(0.05, 0.75);

                    let kelly_params = KellyStochasticParams {
                        sigma: market_params.sigma,
                        time_horizon: market_params.kelly_time_horizon, // Diffusion-based tau for correct P(fill)
                        alpha_touch: market_params.kelly_alpha_touch,
                        alpha_decay_bps: market_params.kelly_alpha_decay_bps,
                        kelly_fraction: dynamic_kelly,
                    };
                    optimizer.optimize_kelly_stochastic(&bid_level_params, &kelly_params)
                } else {
                    optimizer.optimize(&bid_level_params)
                };
                for (i, &size) in allocation.sizes.iter().enumerate() {
                    if i < ladder.bids.len() {
                        ladder.bids[i].size = size;
                    }
                }
                let effective_kelly = if market_params.use_kelly_stochastic {
                    let regime_mult = market_params.volatility_regime.kelly_fraction_multiplier();
                    (market_params.kelly_fraction * regime_mult).clamp(0.05, 0.75)
                } else {
                    0.0
                };
                debug!(
                    binding = ?allocation.binding_constraint,
                    margin_used = %format!("{:.2}", allocation.margin_used),
                    position_used = %format!("{:.4}", allocation.position_used),
                    shadow_price = %format!("{:.6}", allocation.shadow_price),
                    kelly_stochastic = market_params.use_kelly_stochastic,
                    kelly_fraction = %format!("{:.3}", effective_kelly),
                    regime = ?market_params.volatility_regime,
                    "Constrained optimizer applied to bids"
                );
            }

            // 6. Build LevelOptimizationParams for asks
            let ask_level_params: Vec<_> = ladder
                .asks
                .iter()
                .map(|level| {
                    let fill_intensity =
                        fill_intensity_at_depth(level.depth_bps, params.sigma, params.kappa);
                    let spread_capture = spread_capture_at_depth(
                        level.depth_bps,
                        &params,
                        self.ladder_config.fees_bps,
                    );
                    let adverse_selection = adverse_selection_at_depth(level.depth_bps, &params);
                    LevelOptimizationParams {
                        depth_bps: level.depth_bps,
                        fill_intensity,
                        spread_capture,
                        margin_per_unit: market_params.microprice / leverage,
                        adverse_selection,
                    }
                })
                .collect();

            // 7. Optimize ask sizes (use Kelly-Stochastic if enabled)
            if !ask_level_params.is_empty() {
                let allocation = if market_params.use_kelly_stochastic {
                    // Apply volatility regime adjustment to Kelly fraction
                    // High vol → more conservative (lower Kelly)
                    // Low vol → more aggressive (higher Kelly)
                    let regime_multiplier = market_params.volatility_regime.kelly_fraction_multiplier();
                    let dynamic_kelly = (market_params.kelly_fraction * regime_multiplier).clamp(0.05, 0.75);

                    let kelly_params = KellyStochasticParams {
                        sigma: market_params.sigma,
                        time_horizon: market_params.kelly_time_horizon, // Diffusion-based tau for correct P(fill)
                        alpha_touch: market_params.kelly_alpha_touch,
                        alpha_decay_bps: market_params.kelly_alpha_decay_bps,
                        kelly_fraction: dynamic_kelly,
                    };
                    optimizer.optimize_kelly_stochastic(&ask_level_params, &kelly_params)
                } else {
                    optimizer.optimize(&ask_level_params)
                };
                for (i, &size) in allocation.sizes.iter().enumerate() {
                    if i < ladder.asks.len() {
                        ladder.asks[i].size = size;
                    }
                }
                let effective_kelly = if market_params.use_kelly_stochastic {
                    let regime_mult = market_params.volatility_regime.kelly_fraction_multiplier();
                    (market_params.kelly_fraction * regime_mult).clamp(0.05, 0.75)
                } else {
                    0.0
                };
                debug!(
                    binding = ?allocation.binding_constraint,
                    margin_used = %format!("{:.2}", allocation.margin_used),
                    position_used = %format!("{:.4}", allocation.position_used),
                    shadow_price = %format!("{:.6}", allocation.shadow_price),
                    kelly_stochastic = market_params.use_kelly_stochastic,
                    kelly_fraction = %format!("{:.3}", effective_kelly),
                    regime = ?market_params.volatility_regime,
                    "Constrained optimizer applied to asks"
                );
            }

            // 8. Filter out zero-size levels
            ladder.bids.retain(|l| l.size > EPSILON);
            ladder.asks.retain(|l| l.size > EPSILON);

            ladder
        } else {
            Ladder::generate(&self.ladder_config, &params)
        }
    }
}

impl QuotingStrategy for LadderStrategy {
    /// For compatibility with existing infrastructure, returns best bid/ask from ladder.
    ///
    /// The full ladder can be obtained via `generate_ladder()` for multi-level quoting.
    fn calculate_quotes(
        &self,
        config: &QuoteConfig,
        position: f64,
        max_position: f64,
        target_liquidity: f64,
        market_params: &MarketParams,
    ) -> (Option<Quote>, Option<Quote>) {
        let ladder = self.generate_ladder(
            config,
            position,
            max_position,
            target_liquidity,
            market_params,
        );

        // Return just the best bid/ask for backward compatibility
        let bid = ladder.bids.first().map(|l| Quote::new(l.price, l.size));
        let ask = ladder.asks.first().map(|l| Quote::new(l.price, l.size));

        (bid, ask)
    }

    fn calculate_ladder(
        &self,
        config: &QuoteConfig,
        position: f64,
        max_position: f64,
        target_liquidity: f64,
        market_params: &MarketParams,
    ) -> Ladder {
        self.generate_ladder(
            config,
            position,
            max_position,
            target_liquidity,
            market_params,
        )
    }

    fn name(&self) -> &'static str {
        "LadderGLFT"
    }
}

// ============================================================================
// Helper Functions for Constrained Optimizer
// ============================================================================

/// Fill intensity at depth: λ(δ) = σ²/δ² × κ
///
/// Models probability of price reaching depth δ based on diffusion.
/// At touch (δ → 0), returns kappa as baseline intensity.
fn fill_intensity_at_depth(depth_bps: f64, sigma: f64, kappa: f64) -> f64 {
    if depth_bps < EPSILON {
        return kappa; // At touch, use kappa as baseline
    }
    let depth_frac = depth_bps / 10000.0;
    // σ² / δ² gives diffusion-driven fill probability
    // Scale by kappa for market activity level
    let diffusion_term = sigma.powi(2) / depth_frac.powi(2);
    (diffusion_term * kappa).min(kappa) // Cap at kappa
}

/// Spread capture at depth: SC(δ) = δ - AS(δ) - fees
///
/// Expected profit from capturing spread at depth δ.
/// Uses calibrated depth-dependent AS model if available.
fn spread_capture_at_depth(depth_bps: f64, params: &LadderParams, fees_bps: f64) -> f64 {
    let as_at_depth = adverse_selection_at_depth(depth_bps, params);
    // Spread capture = depth - adverse selection - fees
    (depth_bps - as_at_depth - fees_bps).max(0.0)
}

/// Adverse selection at depth: AS(δ) = AS₀ × exp(-δ/δ_char)
///
/// Returns the expected adverse selection cost at a given depth.
/// Uses calibrated depth-dependent AS model if available.
fn adverse_selection_at_depth(depth_bps: f64, params: &LadderParams) -> f64 {
    if let Some(ref decay) = params.depth_decay_as {
        // Use calibrated first-principles model: AS(δ) = AS₀ × exp(-δ/δ_char)
        decay.as_at_depth(depth_bps)
    } else {
        // Legacy fallback: exponential decay with 10bp characteristic depth
        params.as_at_touch_bps * (-depth_bps / 10.0).exp()
    }
}
