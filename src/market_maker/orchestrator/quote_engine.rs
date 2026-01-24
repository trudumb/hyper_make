//! Quote generation engine for the market maker.

use chrono::Timelike;
use tracing::{debug, error, info, warn};

use crate::prelude::Result;

use super::super::{
    quoting, MarketMaker, OrderExecutor, ParameterAggregator, ParameterSources, Quote, QuoteConfig,
    QuotingStrategy, Side,
};
use crate::market_maker::control::{QuoteGateDecision, QuoteGateInput};
use crate::market_maker::infra::metrics::dashboard::{
    ChangepointDiagnostics, KappaDiagnostics, QuoteDecisionRecord, SignalSnapshot,
};
use crate::market_maker::risk::{CircuitBreakerAction, RiskCheckResult};

/// Minimum order notional value in USD (Hyperliquid requirement)
pub(super) const MIN_ORDER_NOTIONAL: f64 = 10.0;

impl<S: QuotingStrategy, E: OrderExecutor> MarketMaker<S, E> {
    /// Update quotes based on current market state.
    #[tracing::instrument(name = "quote_cycle", skip_all, fields(asset = %self.config.asset))]
    pub(crate) async fn update_quotes(&mut self) -> Result<()> {
        // Don't place orders until estimator is warmed up
        if !self.estimator.is_warmed_up() {
            // Log warmup status every 10 seconds to help diagnose why orders aren't placed
            let should_log = match self.last_warmup_block_log {
                None => true,
                Some(last) => last.elapsed() >= std::time::Duration::from_secs(10),
            };
            if should_log {
                let (vol_ticks, min_vol, trade_obs, min_trades) = self.estimator.warmup_progress();
                warn!(
                    volume_ticks = vol_ticks,
                    volume_ticks_required = min_vol,
                    trade_observations = trade_obs,
                    trade_observations_required = min_trades,
                    "Warmup incomplete - no orders placed (waiting for market data)"
                );
                self.last_warmup_block_log = Some(std::time::Instant::now());
            }
            return Ok(());
        }

        // === Circuit Breaker Checks ===
        let breaker_action = self.tier1.circuit_breaker.most_severe_action();
        match breaker_action {
            Some(CircuitBreakerAction::PauseTrading) => {
                warn!("Circuit breaker: pausing trading");
                return Ok(()); // Skip quoting entirely
            }
            Some(CircuitBreakerAction::CancelAllQuotes) => {
                warn!("Circuit breaker: cancelling all quotes");
                // Cancel existing quotes on both sides
                let bid_orders = self.orders.get_all_by_side(Side::Buy);
                let ask_orders = self.orders.get_all_by_side(Side::Sell);
                let all_oids: Vec<u64> = bid_orders
                    .iter()
                    .chain(ask_orders.iter())
                    .map(|o| o.oid)
                    .collect();
                if !all_oids.is_empty() {
                    self.executor
                        .cancel_bulk_orders(&self.config.asset, all_oids)
                        .await;
                }
                return Ok(());
            }
            Some(CircuitBreakerAction::WidenSpreads { .. }) => {
                // Will apply multiplier below
            }
            None => {}
        }

        // === Risk Limit Checks ===
        let position_notional = self.position.position().abs() * self.latest_mid;
        let position_check = self.safety.risk_checker.check_position(position_notional);
        if matches!(position_check, RiskCheckResult::HardLimitBreached { .. }) {
            error!("Hard position limit breached - cancelling quotes");
            return Ok(());
        }

        // === Drawdown Check ===
        if self.safety.drawdown_tracker.should_pause() {
            warn!("Emergency drawdown - pausing trading");
            return Ok(());
        }

        // Update calibration controller with current calibration status
        // This uses AS fills measured and kappa confidence to track calibration progress
        // and adjust fill-hungry gamma multiplier accordingly
        let as_fills_measured = self.tier1.adverse_selection.fills_measured() as u64;
        let kappa_confidence = self.estimator.kappa_confidence();
        self.stochastic
            .calibration_controller
            .update_calibration_status(as_fills_measured, kappa_confidence);

        // HIP-3: OI cap pre-flight check (fast path for unlimited)
        // This is on the hot path, so we use pre-computed values from runtime config
        let current_position_notional = self.position.position().abs() * self.latest_mid;
        let remaining_oi_capacity = self
            .config
            .runtime
            .remaining_oi_capacity(current_position_notional);

        if remaining_oi_capacity < MIN_ORDER_NOTIONAL {
            warn!(
                oi_cap_usd = %self.config.runtime.oi_cap_usd,
                current_notional = %format!("{:.2}", current_position_notional),
                "HIP-3 OI cap reached, skipping quotes"
            );
            return Ok(());
        }

        // Phase 6: Rate limit throttling - respect minimum requote interval
        if !self.infra.proactive_rate_tracker.can_requote() {
            debug!("Skipping requote: minimum interval not elapsed");
            return Ok(());
        }

        // Check for rate limit warnings
        if self.infra.proactive_rate_tracker.ip_rate_warning() {
            warn!("IP rate limit warning: approaching 80% of budget");
        }
        if self.infra.proactive_rate_tracker.address_budget_low() {
            warn!("Address rate limit warning: budget below 1000 requests");
        }

        // Mark that we're doing a requote
        self.infra.proactive_rate_tracker.mark_requote();

        // === LEARNING MODULE: Periodic model health logging ===
        if self.learning.is_enabled() && self.learning.should_log_health() {
            let health = self.learning.model_health();
            let pending = self.learning.pending_predictions_count();
            info!(
                overall = ?health.overall,
                volatility = ?health.volatility,
                adverse_selection = ?health.adverse_selection,
                fill_rate = ?health.fill_rate,
                edge = ?health.edge,
                pending_predictions = pending,
                "Learning module health"
            );
            // Log detailed calibration metrics for diagnostics
            self.learning.log_calibration_report();
        }

        // Phase 3: Check recovery state and handle IOC recovery if needed
        if let Some(action) = self.check_and_handle_recovery().await? {
            if action.skip_normal_quoting {
                return Ok(());
            }
        }

        let quote_config = QuoteConfig {
            mid_price: self.latest_mid,
            decimals: self.config.decimals,
            sz_decimals: self.config.sz_decimals,
            min_notional: MIN_ORDER_NOTIONAL,
        };

        // Build market params from econometric estimates via ParameterAggregator
        let exchange_limits = &self.infra.exchange_limits;
        // Get pending exposure from resting orders (prevents position breach from multiple fills)
        let (pending_bid_exposure, pending_ask_exposure) = self.orders.pending_exposure();

        // DEBUG: Log open order details for diagnosing skew issues
        let (bid_count, ask_count) = self.orders.order_counts();
        debug!(
            bid_orders = bid_count,
            ask_orders = ask_count,
            pending_bid_exposure = %format!("{:.4}", pending_bid_exposure),
            pending_ask_exposure = %format!("{:.4}", pending_ask_exposure),
            total_orders = bid_count + ask_count,
            "Open order state"
        );

        // Get dynamic position VALUE limit from kill switch (first-principles derived)
        let dynamic_max_position_value = self.safety.kill_switch.max_position_value();
        // Margin state is valid if we've refreshed margin at least once
        let margin_state = self.infra.margin_sizer.state();
        let dynamic_limit_valid = margin_state.account_value > 0.0;

        // CRITICAL: Pre-compute effective_max_position and update exchange limits BEFORE building sources
        // This ensures sources.exchange_effective_bid/ask_limit use the margin-based capacity
        let margin_quoting_capacity = if margin_state.available_margin > 0.0
            && self.infra.margin_sizer.summary().max_leverage > 0.0
            && self.latest_mid > 0.0
        {
            (margin_state.available_margin * self.infra.margin_sizer.summary().max_leverage
                / self.latest_mid)
                .max(0.0)
        } else {
            0.0
        };

        // Compute effective_max_position using same priority as MarketParams::effective_max_position
        let dynamic_max_position = if dynamic_limit_valid && self.latest_mid > 0.0 {
            dynamic_max_position_value / self.latest_mid
        } else {
            0.0
        };

        let pre_effective_max_position = {
            const EPSILON: f64 = 1e-9;
            if margin_quoting_capacity > EPSILON {
                if dynamic_limit_valid && dynamic_max_position > EPSILON {
                    margin_quoting_capacity.min(dynamic_max_position)
                } else {
                    margin_quoting_capacity
                }
            } else if dynamic_limit_valid && dynamic_max_position > EPSILON {
                dynamic_max_position
            } else {
                self.config.max_position // Fallback to config during warmup
            }
        };

        // Update exchange limits with margin-based capacity BEFORE building sources
        self.infra
            .exchange_limits
            .update_local_max(pre_effective_max_position);

        // Pre-compute drift-adjusted skew from HJB controller + momentum signals
        let momentum_bps = self.estimator.momentum_bps();
        let p_continuation = self.estimator.momentum_continuation_probability();
        let position = self.position.position();

        // Update momentum EWMA signals for smoothing (reduces whipsawing)
        self.stochastic.hjb_controller.update_momentum_signals(
            momentum_bps,
            p_continuation,
            position,
            self.config.max_position,
        );

        // Get multi-timeframe trend signal for enhanced opposition detection
        // Position value for underwater severity calculation
        let position_value = (position.abs() * self.latest_mid).max(1.0);
        let trend_signal = self.estimator.trend_signal(position_value);

        // Use enhanced drift-adjusted skew with multi-timeframe trend detection
        let drift_adjusted_skew = self.stochastic.hjb_controller.optimal_skew_with_trend(
            position,
            self.config.max_position,
            momentum_bps,
            p_continuation,
            &trend_signal,
        );

        // Log momentum diagnostics for drift-adjusted skew debugging
        // Enhanced: Now includes multi-timeframe trend data
        // Log at INFO when: (a) position opposes trend, (b) significant momentum, or (c) high trend confidence
        if drift_adjusted_skew.is_opposed
            || momentum_bps.abs() > 10.0
            || trend_signal.trend_confidence > 0.3
        {
            let ewma_warmed = self.stochastic.hjb_controller.is_drift_warmed_up();
            let smoothed_drift = self.stochastic.hjb_controller.smoothed_drift();
            info!(
                short_bps = %format!("{:.2}", momentum_bps),
                medium_bps = %format!("{:.2}", trend_signal.medium_momentum_bps),
                long_bps = %format!("{:.2}", trend_signal.long_momentum_bps),
                agreement = %format!("{:.2}", trend_signal.timeframe_agreement),
                underwater = %format!("{:.2}", trend_signal.underwater_severity),
                trend_conf = %format!("{:.2}", trend_signal.trend_confidence),
                p_continuation = %format!("{:.3}", p_continuation),
                position = %format!("{:.2}", position),
                is_opposed = drift_adjusted_skew.is_opposed,
                drift_urgency_bps = %format!("{:.2}", drift_adjusted_skew.drift_urgency * 10000.0),
                variance_mult = %format!("{:.3}", drift_adjusted_skew.variance_multiplier),
                urgency_score = %format!("{:.1}", drift_adjusted_skew.urgency_score),
                ewma_warmed = ewma_warmed,
                smoothed_drift = %format!("{:.6}", smoothed_drift),
                "Multi-timeframe trend detection"
            );
        }

        let sources = ParameterSources {
            estimator: &self.estimator,
            adverse_selection: &self.tier1.adverse_selection,
            depth_decay_as: &self.tier1.depth_decay_as,
            liquidation_detector: &self.tier1.liquidation_detector,
            hawkes: &self.tier2.hawkes,
            funding: &self.tier2.funding,
            spread_tracker: &self.tier2.spread_tracker,
            hjb_controller: &self.stochastic.hjb_controller,
            margin_sizer: &self.infra.margin_sizer,
            stochastic_config: &self.stochastic.stochastic_config,
            drift_adjusted_skew,
            adaptive_spreads: &self.stochastic.adaptive_spreads,
            position: self.position.position(),
            max_position: self.config.max_position,
            latest_mid: self.latest_mid,
            risk_aversion: self.config.risk_aversion,
            // Exchange position limits
            exchange_limits_valid: exchange_limits.is_initialized(),
            exchange_effective_bid_limit: exchange_limits.effective_bid_limit(),
            exchange_effective_ask_limit: exchange_limits.effective_ask_limit(),
            exchange_limits_age_ms: exchange_limits.age_ms(),
            // Pending exposure from resting orders
            pending_bid_exposure,
            pending_ask_exposure,
            // Dynamic position limits (first principles)
            dynamic_max_position_value,
            dynamic_limit_valid,
            // Stochastic constraints (first principles)
            tick_size_bps: 10.0, // TODO: Get from asset metadata
            near_touch_depth_usd: self.estimator.near_touch_depth_usd(),
            // Calibration fill rate controller
            calibration_gamma_mult: self.stochastic.calibration_controller.gamma_multiplier(),
            calibration_progress: self
                .stochastic
                .calibration_controller
                .calibration_progress(),
            calibration_complete: self.stochastic.calibration_controller.is_calibrated(),
            // Dynamic bounds (model-driven, replaces hardcoded CLI values)
            // true when no CLI override is active (kappa_floor/max_spread_ceiling_bps = None)
            use_dynamic_kappa_floor: !self.estimator.has_static_kappa_floor(),
            use_dynamic_spread_ceiling: !self.estimator.has_static_max_spread_ceiling(),
        };
        let mut market_params = ParameterAggregator::build(&sources);

        // FIX: Wire p_momentum_continue from actual momentum model
        // (p_continuation was computed earlier from self.estimator.momentum_continuation_probability())
        // This was hardcoded to 0.5 in aggregator.rs, causing Quote Gate to never open
        market_params.p_momentum_continue = p_continuation;

        // === Regime-Aware Parameters ===
        // Get HMM regime probabilities for soft blending
        // [p_low, p_normal, p_high, p_extreme]
        let regime_probs = self.stochastic.regime_hmm.regime_probabilities();
        let (p_low, p_normal, p_high, p_extreme) = (
            regime_probs[0],
            regime_probs[1],
            regime_probs[2],
            regime_probs[3],
        );

        // Log regime state when not in normal regime
        if p_extreme > 0.3 || p_high > 0.5 || p_low > 0.7 {
            debug!(
                p_low = %format!("{:.2}", p_low),
                p_normal = %format!("{:.2}", p_normal),
                p_high = %format!("{:.2}", p_high),
                p_extreme = %format!("{:.2}", p_extreme),
                "HMM regime probabilities"
            );
        }

        // Get spread multiplier from circuit breaker if applicable
        let spread_multiplier = match breaker_action {
            Some(CircuitBreakerAction::WidenSpreads { multiplier }) => multiplier,
            _ => 1.0,
        };

        // Apply circuit breaker spread multiplier to market params if needed
        if spread_multiplier > 1.0 {
            debug!(
                multiplier = %format!("{:.2}", spread_multiplier),
                "Circuit breaker: widening spreads"
            );
            // The spread multiplier will be applied in the strategy
        }

        // Apply position-based size reduction from risk checker and drawdown tracker
        let risk_size_mult = self
            .safety
            .risk_checker
            .suggested_size_multiplier(self.position.position().abs() * self.latest_mid);
        let drawdown_size_mult = self.safety.drawdown_tracker.position_multiplier();
        let size_multiplier = risk_size_mult * drawdown_size_mult;

        if size_multiplier < 1.0 {
            debug!(
                risk_mult = %format!("{:.2}", risk_size_mult),
                drawdown_mult = %format!("{:.2}", drawdown_size_mult),
                total_mult = %format!("{:.2}", size_multiplier),
                "Position-based size reduction active"
            );
        }

        // === MEASURED LATENCY: Set from WS ping measurements ===
        // This is MEASURED data, NOT hardcoded assumptions
        // Use prometheus metrics directly (concrete type) for measured WS latency
        let measured_latency = self.infra.prometheus.ws_ping_latency_ms();
        market_params.measured_latency_ms = if measured_latency > 0.0 {
            measured_latency
        } else {
            50.0 // Conservative default during warmup
        };

        // Compute stochastic constraints (latency floor, tight quoting conditions)
        // NOTE: latency_spread_floor now uses measured_latency_ms instead of hardcoded config
        let current_hour_utc = chrono::Utc::now().hour() as u8;
        market_params.compute_stochastic_constraints(
            &self.stochastic.stochastic_config,
            self.position.position(),
            self.config.max_position,
            current_hour_utc,
        );

        // === MODEL-DERIVED TARGET LIQUIDITY (GLFT First Principles) ===
        // Compute GLFT-derived target_liquidity using ALL measured inputs:
        // - account_value: from margin state
        // - leverage: from exchange metadata
        // - sigma: from bipower variation estimator (MEASURED)
        // - fill_rate: from observed fills (MEASURED)
        // - latency: from WS ping (MEASURED)
        // Note: num_levels is used for per-level cap calculation (defensive bound)
        // Default ladder has 25 levels - this constant is used for sizing only
        const DEFAULT_NUM_LEVELS: usize = 25;
        market_params.compute_derived_target_liquidity(
            self.config.risk_aversion, // User preference (γ)
            DEFAULT_NUM_LEVELS,        // Ladder config (default 25 levels)
            MIN_ORDER_NOTIONAL,        // Exchange minimum ($10)
        );

        // CRITICAL: Update cached effective max_position from first principles
        // This is THE source of truth for all position limit checks
        let new_effective = market_params.effective_max_position(self.config.max_position);
        if (new_effective - self.effective_max_position).abs() > 0.001 {
            debug!(
                old = %format!("{:.6}", self.effective_max_position),
                new = %format!("{:.6}", new_effective),
                dynamic_valid = market_params.dynamic_limit_valid,
                "Effective max position updated from first principles"
            );
        }
        self.effective_max_position = new_effective;

        // =======================================================================
        // PROACTIVE POSITION MANAGEMENT (Phases 1-4)
        // =======================================================================
        // This transforms quoting from passive (react to fills) to proactive
        // (assess → size appropriately → quote deliberately).

        // Phase 1: Time-Based Position Ramp
        // Limits max position based on session time elapsed
        let ramp_fraction = if self.stochastic.stochastic_config.enable_position_ramp {
            self.stochastic.position_ramp.current_fraction()
        } else {
            1.0 // Disabled: full capacity immediately
        };
        let ramped_max_position = new_effective * ramp_fraction;

        // Phase 4: Performance-Gated Capacity
        // Adjusts max position based on realized P&L (earn the right to size up)
        let performance_fraction = if self.stochastic.stochastic_config.enable_performance_gating {
            // Update performance gating with current state
            self.stochastic.performance_gating.update_reference_price(self.latest_mid);
            self.stochastic.performance_gating.update_base_max_position(ramped_max_position);

            // Get realized P&L from tracker
            let realized_pnl = self.tier2.pnl_tracker.summary(self.latest_mid).realized_pnl;
            self.stochastic.performance_gating.capacity_fraction(realized_pnl)
        } else {
            1.0 // Disabled: full capacity
        };
        let performance_adjusted_max = ramped_max_position * performance_fraction;

        // Phase 2: Confidence-Gated Sizing
        // Scale quote size by model confidence
        let confidence_size_mult = if self.stochastic.stochastic_config.enable_confidence_sizing {
            // Build aggregate confidence from component confidences
            let (vol_ticks, min_vol_ticks, trade_obs, min_trades) = self.estimator.warmup_progress();
            let warmup_progress = if min_vol_ticks > 0 && min_trades > 0 {
                let vol_progress = (vol_ticks as f64 / min_vol_ticks as f64).min(1.0);
                let trade_progress = (trade_obs as f64 / min_trades as f64).min(1.0);
                (vol_progress + trade_progress) / 2.0
            } else {
                1.0
            };

            let kappa_confidence = self.estimator.kappa_confidence();
            let vol_confidence = if market_params.sigma > 0.0 { 0.8 } else { 0.3 }; // Vol estimate confidence
            let as_confidence = if self.tier1.adverse_selection.is_warmed_up() { 0.7 } else { 0.3 };
            let momentum_confidence = p_continuation; // Use momentum continuation probability

            let aggregate_confidence = crate::market_maker::learning::AggregateConfidence::new(
                momentum_confidence,
                vol_confidence,
                kappa_confidence,
                as_confidence,
                self.tier1.adverse_selection.fills_measured(),
                warmup_progress < 0.99,
            );
            aggregate_confidence.size_multiplier()
        } else {
            1.0 // Disabled: full size
        };

        // Apply all proactive adjustments
        let proactive_max_position = performance_adjusted_max;
        let proactive_size_mult = confidence_size_mult;

        // Log proactive adjustments when significant
        if ramp_fraction < 0.99 || performance_fraction < 0.99 || confidence_size_mult < 0.99 {
            debug!(
                ramp_fraction = %format!("{:.2}", ramp_fraction),
                ramp_elapsed_min = %format!("{:.1}", self.stochastic.position_ramp.elapsed_secs() / 60.0),
                performance_fraction = %format!("{:.2}", performance_fraction),
                confidence_mult = %format!("{:.2}", confidence_size_mult),
                base_max = %format!("{:.4}", new_effective),
                proactive_max = %format!("{:.4}", proactive_max_position),
                "Proactive position management active"
            );
        }

        // Update effective max position with proactive adjustments
        self.effective_max_position = proactive_max_position;

        // Note: exchange limits were already updated BEFORE building sources (line ~1210)
        // using pre_effective_max_position which should equal new_effective

        // === EFFECTIVE TARGET LIQUIDITY: Use GLFT-Derived as Default ===
        // The model-derived liquidity is now the PRIMARY source of truth.
        // User's config.target_liquidity is used ONLY as a cap (safety override).
        //
        // Priority:
        // 1. GLFT-derived (from γ, σ, account_value, latency) - DEFAULT
        // 2. User config - cap only (can reduce, cannot increase beyond derived)
        // 3. Exchange minimum - floor (must pass min_notional)
        let truncation_buffer = 1.5 * (10.0_f64).powi(-(self.config.sz_decimals as i32));
        let min_viable_liquidity =
            (MIN_ORDER_NOTIONAL / market_params.microprice) + truncation_buffer;

        // Use derived liquidity as default, cap with user config
        let new_effective_liquidity = if market_params.derived_target_liquidity > 0.0 {
            // GLFT-derived is available: use it, capped by user config
            market_params
                .derived_target_liquidity
                .min(self.config.target_liquidity) // User cap
                .max(min_viable_liquidity) // Exchange minimum
                .min(self.effective_max_position) // Position limit
        } else {
            // Fallback: use config (warmup or zero account_value)
            self.config
                .target_liquidity
                .max(min_viable_liquidity)
                .min(self.effective_max_position)
        };

        if (new_effective_liquidity - self.effective_target_liquidity).abs() > 0.001 {
            info!(
                old = %format!("{:.6}", self.effective_target_liquidity),
                new = %format!("{:.6}", new_effective_liquidity),
                config_target = %format!("{:.6}", self.config.target_liquidity),
                derived_target = %format!("{:.6}", market_params.derived_target_liquidity),
                min_viable = %format!("{:.6}", min_viable_liquidity),
                max_position = %format!("{:.6}", self.effective_max_position),
                latency_ms = %format!("{:.1}", market_params.measured_latency_ms),
                "Target liquidity: GLFT-derived as default (all inputs from measured data)"
            );
        }
        self.effective_target_liquidity = new_effective_liquidity;

        // Log adaptive system status if enabled
        if self.stochastic.stochastic_config.use_adaptive_spreads {
            let adaptive = &self.stochastic.adaptive_spreads;
            debug!(
                can_estimate = market_params.adaptive_can_estimate,
                fully_warmed_up = market_params.adaptive_warmed_up,
                warmup_progress = %format!("{:.0}%", market_params.adaptive_warmup_progress * 100.0),
                uncertainty_factor = %format!("{:.3}", market_params.adaptive_uncertainty_factor),
                adaptive_floor_bps = %format!("{:.2}", market_params.adaptive_spread_floor * 10000.0),
                adaptive_kappa = %format!("{:.0}", market_params.adaptive_kappa),
                adaptive_gamma = %format!("{:.3}", market_params.adaptive_gamma),
                adaptive_ceiling_bps = %format!("{:.2}", market_params.adaptive_spread_ceiling * 10000.0),
                fill_rate = %format!("{:.4}", adaptive.fill_rate_controller().observed_fill_rate()),
                "Adaptive Bayesian spreads (using immediately via priors)"
            );

            // Call on_no_fill to nudge toward tighter spreads when quoting without fills
            // This is a soft decay that reduces spread over time when not getting filled
            self.stochastic.adaptive_spreads.on_no_fill_simple();
        }

        debug!(
            mid = self.latest_mid,
            microprice = %format!("{:.4}", market_params.microprice),
            position = self.position.position(),
            static_max_pos = self.config.max_position,
            dynamic_max_pos = %format!("{:.6}", market_params.dynamic_max_position),
            effective_max_pos = %format!("{:.6}", self.effective_max_position),
            dynamic_valid = market_params.dynamic_limit_valid,
            config_target_liq = self.config.target_liquidity,
            effective_target_liq = %format!("{:.6}", self.effective_target_liquidity),
            sigma_clean = %format!("{:.6}", market_params.sigma),
            sigma_effective = %format!("{:.6}", market_params.sigma_effective),
            kappa = %format!("{:.2}", market_params.kappa),
            jump_ratio = %format!("{:.2}", market_params.jump_ratio),
            is_toxic = market_params.is_toxic_regime,
            beta_book = %format!("{:.6}", market_params.beta_book),
            beta_flow = %format!("{:.6}", market_params.beta_flow),
            book_imbalance = %format!("{:.2}", market_params.book_imbalance),
            liq_gamma_mult = %format!("{:.2}", market_params.liquidity_gamma_mult),
            "Quote inputs with microprice"
        );

        // === SIGNAL SNAPSHOT FOR DASHBOARD VISUALIZATION ===
        // Record current signal state for the dashboard's SIGNALS tab
        // This captures all the alpha signals used to make quote decisions
        {
            let now = chrono::Utc::now();

            // Get regime probabilities from stochastic controller if enabled
            let (regime_quiet, regime_trending, regime_volatile, regime_cascade) =
                if self.stochastic.controller.is_enabled() {
                    let belief = self.stochastic.controller.belief();
                    (
                        belief.regime_probs[0],
                        belief.regime_probs[1],
                        belief.regime_probs[2],
                        (1.0 - market_params.cascade_size_factor).max(0.0),
                    )
                } else {
                    // Fallback: use market params to estimate regime
                    let cascade = (1.0 - market_params.cascade_size_factor).max(0.0);
                    let volatile = if market_params.jump_ratio > 2.0 {
                        0.3
                    } else {
                        0.0
                    };
                    let trending = if market_params.sigma > 0.002 {
                        0.2
                    } else {
                        0.0
                    };
                    let quiet = (1.0 - cascade - volatile - trending).max(0.0);
                    (quiet, trending, volatile, cascade)
                };

            // Get changepoint info from stochastic controller
            let (cp_prob_5, cp_detected) = if self.stochastic.controller.is_enabled() {
                let cp = self.stochastic.controller.changepoint_summary();
                (cp.cp_prob_5, cp.detected)
            } else {
                (0.0, false)
            };

            let signal_snapshot = SignalSnapshot {
                timestamp_ms: now.timestamp_millis(),
                time: now.format("%H:%M:%S").to_string(),
                // Kappa (fill intensity)
                kappa: market_params.kappa,
                kappa_confidence: self.estimator.kappa_confidence(),
                kappa_ci_lower: market_params.kappa * 0.7, // Approximate CI
                kappa_ci_upper: market_params.kappa * 1.3,
                // Microprice
                microprice: market_params.microprice,
                beta_book: market_params.beta_book,
                beta_flow: market_params.beta_flow,
                // Momentum
                momentum_bps: self.estimator.momentum_bps(),
                flow_imbalance: market_params.book_imbalance,
                falling_knife: if drift_adjusted_skew.is_opposed {
                    drift_adjusted_skew.urgency_score
                } else {
                    0.0
                },
                // Volatility
                sigma: market_params.sigma,
                jump_ratio: market_params.jump_ratio,
                // Regime
                regime_quiet,
                regime_trending,
                regime_volatile,
                regime_cascade,
                // Changepoint
                cp_prob_5,
                cp_detected,
            };
            self.infra
                .prometheus
                .dashboard()
                .record_signal_snapshot(signal_snapshot);

            // Update kappa diagnostics
            let kappa_diag = KappaDiagnostics {
                posterior_mean: market_params.kappa,
                posterior_std: market_params.kappa * (1.0 - self.estimator.kappa_confidence()),
                confidence: self.estimator.kappa_confidence(),
                ci_95_lower: market_params.kappa * 0.7,
                ci_95_upper: market_params.kappa * 1.3,
                cv: 0.15, // Default coefficient of variation
                is_heavy_tailed: false,
                observation_count: self.tier1.adverse_selection.fills_measured() as usize,
                mean_distance_bps: 0.0, // TODO: get from estimator
            };
            self.infra
                .prometheus
                .dashboard()
                .update_kappa_diagnostics(kappa_diag);

            // Update changepoint diagnostics if controller enabled
            if self.stochastic.controller.is_enabled() {
                let cp = self.stochastic.controller.changepoint_summary();
                let cp_diag = ChangepointDiagnostics {
                    cp_prob_1: cp.cp_prob_1,
                    cp_prob_5: cp.cp_prob_5,
                    cp_prob_10: cp.cp_prob_10,
                    run_length: cp.most_likely_run as u32,
                    entropy: cp.entropy,
                    detected: cp.detected,
                };
                self.infra
                    .prometheus
                    .dashboard()
                    .update_changepoint_diagnostics(cp_diag);
            }
        }

        // Update fill model params for Bayesian fill probability
        // Uses Kelly time horizon (τ) which is computed in ParameterAggregator
        self.strategy
            .update_fill_model_params(market_params.sigma, market_params.kelly_time_horizon);

        // === DECISION ENGINE FILTER ===
        // When enabled, uses the learning module's decision engine to evaluate
        // whether to quote and at what size based on:
        // - P(edge > 0) confidence threshold
        // - Model health status
        // - Current drawdown
        // - Model disagreement
        let decision_size_fraction = if self.learning.use_decision_filter() {
            // Get current drawdown from risk state
            let risk_state = self.build_risk_state();
            let current_drawdown = risk_state.drawdown();

            // Evaluate decision
            let decision = self.learning.evaluate_decision(
                &market_params,
                self.position.position(),
                current_drawdown,
            );

            match decision {
                crate::market_maker::learning::QuoteDecision::NoQuote { reason } => {
                    info!(
                        reason = %reason,
                        drawdown = %format!("{:.2}%", current_drawdown * 100.0),
                        "Decision engine: not quoting"
                    );
                    return Ok(()); // Skip this quote cycle
                }
                crate::market_maker::learning::QuoteDecision::ReducedSize { fraction, reason } => {
                    info!(
                        fraction = %format!("{:.0}%", fraction * 100.0),
                        reason = %reason,
                        "Decision engine: reduced size"
                    );
                    fraction
                }
                crate::market_maker::learning::QuoteDecision::Quote {
                    size_fraction,
                    confidence,
                    expected_edge,
                    reservation_shift,
                    spread_multiplier,
                } => {
                    debug!(
                        size_fraction = %format!("{:.0}%", size_fraction * 100.0),
                        confidence = %format!("{:.1}%", confidence * 100.0),
                        expected_edge = %format!("{:.2}bp", expected_edge),
                        reservation_shift = %format!("{:.6}", reservation_shift),
                        spread_multiplier = %format!("{:.2}x", spread_multiplier),
                        "Decision engine: quoting"
                    );
                    // Set L2 outputs on market_params for downstream strategy
                    market_params.l2_reservation_shift = reservation_shift;
                    market_params.l2_spread_multiplier = spread_multiplier;
                    size_fraction
                }
            }
        } else {
            1.0 // No filter, use full size
        };

        // Apply decision sizing AND proactive confidence to effective target liquidity
        let decision_adjusted_liquidity = self.effective_target_liquidity
            * decision_size_fraction
            * proactive_size_mult;

        // === LAYER 3: STOCHASTIC CONTROLLER ===
        // When enabled, provides POMDP-based sequential decision-making on top of Layer 2
        // Can override Layer 2 decisions based on:
        // - Terminal conditions (session end, funding)
        // - Information value (when to wait vs act)
        // - Regime changes (changepoint detection)
        // - Value function optimization
        if self.stochastic.controller.is_enabled() {
            // Build trading state for the controller
            let risk_state = self.build_risk_state();
            // Time to next funding (convert from Duration to hours)
            let time_to_funding = self
                .tier2
                .funding
                .time_to_next_funding()
                .map(|d| d.as_secs_f64() / 3600.0)
                .unwrap_or(8.0); // Default to 8 hours if unknown
            let trading_state = crate::market_maker::control::TradingState {
                wealth: self.tier2.pnl_tracker.summary(self.latest_mid).total_pnl,
                position: self.position.position(),
                margin_used: self.infra.margin_sizer.state().margin_used,
                session_time: self.session_time_fraction(),
                time_to_funding,
                predicted_funding: self.tier2.funding.ewma_rate() * 10000.0, // Convert to bps
                drawdown: risk_state.drawdown(),
                reduce_only: risk_state.should_reduce_only(),
            };

            // Get Layer 2 output for the controller
            let learning_output = self.learning.output(
                &market_params,
                self.position.position(),
                trading_state.drawdown,
            );

            // Get optimal action from Layer 3
            let action = self
                .stochastic
                .controller
                .act(&learning_output, &trading_state);

            // === L1→L2→L3→DECISION TRACE LOG ===
            // This is the key diagnostic trace showing the full pipeline
            info!(
                target: "layer3::trace",
                // Layer 1 (Estimator)
                l1_sigma = %format!("{:.6}", market_params.sigma),
                l1_kappa = %format!("{:.0}", market_params.kappa),
                l1_micro = %format!("{:.2}", market_params.microprice),
                l1_vol_regime = ?market_params.volatility_regime,
                // Layer 2 (LearningModule)
                l2_edge_mean = %format!("{:.2}", learning_output.edge_prediction.mean),
                l2_edge_std = %format!("{:.2}", learning_output.edge_prediction.std),
                l2_model_health = ?learning_output.model_health.overall,
                l2_p_positive = %format!("{:.3}", learning_output.p_positive_edge),
                l2_decision = %format!("{:?}", learning_output.myopic_decision).chars().take(40).collect::<String>(),
                // Layer 3 (StochasticController)
                l3_trust = %format!("{:.2}", self.stochastic.controller.learning_trust()),
                l3_cp_prob = %format!("{:.3}", self.stochastic.controller.changepoint_summary().cp_prob_5),
                l3_action = %format!("{:?}", action).chars().take(40).collect::<String>(),
                // Final
                position = %format!("{:.4}", self.position.position()),
                "[Trace] L1->L2->L3 pipeline"
            );

            // Handle controller decision
            use crate::market_maker::control::Action;
            match action {
                Action::NoQuote { reason } => {
                    debug!(reason = ?reason, "Layer 3: not quoting");
                    return Ok(());
                }
                Action::WaitToLearn {
                    expected_info_gain,
                    suggested_wait_cycles,
                } => {
                    debug!(
                        info_gain = %format!("{:.4}", expected_info_gain),
                        wait_cycles = suggested_wait_cycles,
                        "Layer 3: waiting to learn"
                    );
                    // Cancel all resting orders when waiting to learn
                    // This prevents stale orders from sitting on the book while we gather information
                    let bid_orders = self.orders.get_all_by_side(Side::Buy);
                    let ask_orders = self.orders.get_all_by_side(Side::Sell);
                    let resting_oids: Vec<u64> = bid_orders
                        .iter()
                        .chain(ask_orders.iter())
                        .map(|o| o.oid)
                        .collect();
                    if !resting_oids.is_empty() {
                        info!(
                            count = resting_oids.len(),
                            "WaitToLearn: cancelling resting orders to prevent stale quotes"
                        );
                        self.executor
                            .cancel_bulk_orders(&self.config.asset, resting_oids)
                            .await;
                    }
                    return Ok(());
                }
                Action::DumpInventory { urgency, .. } => {
                    debug!(urgency = %format!("{:.2}", urgency), "Layer 3: dump inventory mode");
                    // Continue with normal quoting but the strategy will handle urgency
                }
                Action::BuildInventory { .. } => {
                    debug!("Layer 3: build inventory mode");
                    // Continue with normal quoting
                }
                Action::DefensiveQuote {
                    spread_multiplier,
                    size_fraction,
                    reason,
                } => {
                    debug!(
                        spread_mult = %format!("{:.2}", spread_multiplier),
                        size_frac = %format!("{:.2}", size_fraction),
                        reason = ?reason,
                        "Layer 3: defensive quote"
                    );
                    // Apply defensive adjustments through the strategy
                }
                Action::Quote { expected_value, .. } => {
                    debug!(expected_value = %format!("{:.4}", expected_value), "Layer 3: normal quote");
                    // Continue with normal quoting
                }
            }
        }

        // === QUOTE GATE: Decide WHETHER to quote based on directional edge ===
        // This prevents whipsaw losses from random fills when we have no directional edge.
        // The gate uses flow_imbalance, momentum confidence, and position to decide.
        let quote_gate_input = QuoteGateInput {
            flow_imbalance: market_params.flow_imbalance,
            momentum_confidence: market_params.p_momentum_continue,
            momentum_bps: market_params.momentum_bps,
            position: self.position.position(),
            max_position: self.effective_max_position,
            is_warmup: !self.estimator.is_warmed_up(),
            cascade_size_factor: market_params.cascade_size_factor,
        };
        let quote_gate_decision = self.stochastic.quote_gate.decide(&quote_gate_input);

        // Handle quote gate decision
        match &quote_gate_decision {
            QuoteGateDecision::NoQuote { reason } => {
                info!(
                    reason = %reason,
                    flow_imbalance = %format!("{:.3}", market_params.flow_imbalance),
                    position = %format!("{:.4}", self.position.position()),
                    "Quote gate: NO QUOTE - waiting for signal"
                );
                // Cancel any resting orders when not quoting
                let bid_orders = self.orders.get_all_by_side(Side::Buy);
                let ask_orders = self.orders.get_all_by_side(Side::Sell);
                let resting_oids: Vec<u64> = bid_orders
                    .iter()
                    .chain(ask_orders.iter())
                    .map(|o| o.oid)
                    .collect();
                if !resting_oids.is_empty() {
                    info!(
                        count = resting_oids.len(),
                        "Quote gate: cancelling resting orders"
                    );
                    self.executor
                        .cancel_bulk_orders(&self.config.asset, resting_oids)
                        .await;
                }
                return Ok(());
            }
            QuoteGateDecision::QuoteOnlyBids { urgency } => {
                debug!(
                    urgency = %format!("{:.2}", urgency),
                    "Quote gate: ONLY BIDS (reducing short or bullish edge)"
                );
                // Will clear asks after ladder generation
            }
            QuoteGateDecision::QuoteOnlyAsks { urgency } => {
                debug!(
                    urgency = %format!("{:.2}", urgency),
                    "Quote gate: ONLY ASKS (reducing long or bearish edge)"
                );
                // Will clear bids after ladder generation
            }
            QuoteGateDecision::QuoteBoth => {
                // Normal path - quote both sides with skew
            }
        }

        // Try multi-level ladder quoting first
        // HARMONIZED: Use decision-adjusted liquidity (first-principles derived, decision-scaled)
        let ladder = self.strategy.calculate_ladder(
            &quote_config,
            self.position.position(),
            self.effective_max_position, // First-principles limit
            decision_adjusted_liquidity, // Decision-adjusted viable size
            &market_params,
        );

        if !ladder.bids.is_empty() || !ladder.asks.is_empty() {
            // Multi-level ladder mode
            let mut bid_quotes: Vec<Quote> = ladder
                .bids
                .iter()
                .map(|l| Quote::new(l.price, l.size))
                .collect();
            let mut ask_quotes: Vec<Quote> = ladder
                .asks
                .iter()
                .map(|l| Quote::new(l.price, l.size))
                .collect();

            // === QUOTE GATE: Apply one-sided filtering if needed ===
            match &quote_gate_decision {
                QuoteGateDecision::QuoteOnlyBids { urgency: _ } => {
                    // Only quote bids (buying) - clear asks
                    if !ask_quotes.is_empty() {
                        info!(
                            cleared_asks = ask_quotes.len(),
                            "Quote gate: clearing ASK side (only quoting bids)"
                        );
                        ask_quotes.clear();
                    }
                }
                QuoteGateDecision::QuoteOnlyAsks { urgency: _ } => {
                    // Only quote asks (selling) - clear bids
                    if !bid_quotes.is_empty() {
                        info!(
                            cleared_bids = bid_quotes.len(),
                            "Quote gate: clearing BID side (only quoting asks)"
                        );
                        bid_quotes.clear();
                    }
                }
                QuoteGateDecision::QuoteBoth | QuoteGateDecision::NoQuote { .. } => {
                    // QuoteBoth: normal, no filtering
                    // NoQuote: should have returned early, but defensive
                }
            }

            // Reduce-only mode: when over max position, position value, OR margin utilization
            // Phase 3: Use exchange-aware reduce-only that checks exchange limits and signals escalation
            // CAPITAL-EFFICIENT: Use margin utilization as primary trigger (80% threshold)
            let margin_state = self.infra.margin_sizer.state();
            let reduce_only_config = quoting::ReduceOnlyConfig {
                position: self.position.position(),
                max_position: self.effective_max_position,
                mid_price: self.latest_mid,
                max_position_value: self.safety.kill_switch.max_position_value(),
                asset: self.config.asset.to_string(),
                margin_used: margin_state.margin_used,
                account_value: margin_state.account_value,
                // Dynamic reduce-only based on liquidation proximity
                liquidation_price: margin_state.liquidation_price,
                liquidation_buffer_ratio: margin_state.liquidation_buffer_ratio(),
                liquidation_trigger_threshold: quoting::DEFAULT_LIQUIDATION_TRIGGER_THRESHOLD,
            };
            let reduce_only_result = quoting::QuoteFilter::apply_reduce_only_with_exchange_limits(
                &mut bid_quotes,
                &mut ask_quotes,
                &reduce_only_config,
                &self.infra.exchange_limits,
            );

            // If escalation is needed, the recovery manager should be notified
            // (This happens automatically via rate limiter when orders get rejected)
            if reduce_only_result.needs_escalation {
                debug!("Reduce-only mode activated with potential escalation");
            }

            debug!(
                bid_levels = bid_quotes.len(),
                ask_levels = ask_quotes.len(),
                best_bid = ?bid_quotes.first().map(|q| (q.price, q.size)),
                best_ask = ?ask_quotes.first().map(|q| (q.price, q.size)),
                "Calculated ladder quotes"
            );

            // === RECORD QUOTE DECISION FOR DASHBOARD ===
            // Capture why this specific spread was chosen
            if let (Some(best_bid), Some(best_ask)) = (bid_quotes.first(), ask_quotes.first()) {
                let now = chrono::Utc::now();
                let mid = self.latest_mid;
                let bid_spread_bps = ((mid - best_bid.price) / mid) * 10000.0;
                let ask_spread_bps = ((best_ask.price - mid) / mid) * 10000.0;

                // Determine regime string from cascade_size_factor
                let cascade_level = 1.0 - market_params.cascade_size_factor;
                let regime = if cascade_level > 0.5 {
                    "Cascade"
                } else if market_params.jump_ratio > 3.0 {
                    "Volatile"
                } else if market_params.sigma > 0.002 {
                    "Trending"
                } else {
                    "Quiet"
                }
                .to_string();

                // Determine defensive reason if any
                let defensive_reason = if market_params.is_toxic_regime {
                    Some("Toxic regime detected".to_string())
                } else if reduce_only_result.was_filtered {
                    Some("Reduce-only mode".to_string())
                } else if decision_size_fraction < 1.0 {
                    Some(format!(
                        "Decision filter: {:.0}% size",
                        decision_size_fraction * 100.0
                    ))
                } else {
                    None
                };

                let decision = QuoteDecisionRecord {
                    timestamp_ms: now.timestamp_millis(),
                    time: now.format("%H:%M:%S").to_string(),
                    bid_spread_bps,
                    ask_spread_bps,
                    input_kappa: market_params.kappa,
                    input_gamma: self.config.risk_aversion * market_params.hjb_gamma_multiplier,
                    input_sigma: market_params.sigma,
                    input_inventory: self.position.position(),
                    regime,
                    momentum_adjustment: drift_adjusted_skew.drift_urgency * 10000.0,
                    inventory_skew: market_params.hjb_optimal_skew,
                    defensive_reason,
                };
                self.infra
                    .prometheus
                    .dashboard()
                    .record_quote_decision(decision);
            }

            // DIAGNOSTIC: Warn when ladder is completely empty after processing
            // This helps diagnose min_notional and capacity issues at INFO level
            if bid_quotes.is_empty() && ask_quotes.is_empty() {
                let pos = self.position.position();
                let max_pos = self.effective_max_position;
                let mid = self.latest_mid;
                warn!(
                    position = %format!("{:.6}", pos),
                    max_position = %format!("{:.6}", max_pos),
                    mid_price = %format!("{:.2}", mid),
                    bid_capacity_notional = %format!("{:.2}", (max_pos - pos).max(0.0) * mid),
                    ask_capacity_notional = %format!("{:.2}", (max_pos + pos).max(0.0) * mid),
                    min_notional = %format!("{:.2}", MIN_ORDER_NOTIONAL),
                    reduce_only_was_filtered = reduce_only_result.was_filtered,
                    "No orders to place: ladder empty after filtering (check min_notional vs capacity)"
                );
                return Ok(());
            }

            // Reconcile ladder quotes
            if self.config.smart_reconcile {
                // Smart reconciliation with ORDER MODIFY for queue preservation
                self.reconcile_ladder_smart(bid_quotes, ask_quotes).await?;
            } else {
                // Legacy all-or-nothing reconciliation
                self.reconcile_ladder_side(Side::Buy, bid_quotes).await?;
                self.reconcile_ladder_side(Side::Sell, ask_quotes).await?;
            }
        } else {
            // Fallback to single-quote mode for non-ladder strategies
            // HARMONIZED: Use decision-adjusted values (first-principles derived, decision-scaled)
            let (mut bid, mut ask) = self.strategy.calculate_quotes(
                &quote_config,
                self.position.position(),
                self.effective_max_position, // First-principles limit
                decision_adjusted_liquidity, // Decision-adjusted viable size
                &market_params,
            );

            // === QUOTE GATE: Apply one-sided filtering if needed (single-quote mode) ===
            match &quote_gate_decision {
                QuoteGateDecision::QuoteOnlyBids { .. } => {
                    // Only quote bids (buying) - clear ask
                    if ask.is_some() {
                        debug!("Quote gate: clearing ASK quote (single-quote mode)");
                        ask = None;
                    }
                }
                QuoteGateDecision::QuoteOnlyAsks { .. } => {
                    // Only quote asks (selling) - clear bid
                    if bid.is_some() {
                        debug!("Quote gate: clearing BID quote (single-quote mode)");
                        bid = None;
                    }
                }
                QuoteGateDecision::QuoteBoth | QuoteGateDecision::NoQuote { .. } => {
                    // QuoteBoth: normal, no filtering
                    // NoQuote: should have returned early, but defensive
                }
            }

            // Reduce-only mode: when over max position, position value, OR margin utilization
            // CAPITAL-EFFICIENT: Use margin utilization as primary trigger (80% threshold)
            let margin_state = self.infra.margin_sizer.state();
            let reduce_only_config = quoting::ReduceOnlyConfig {
                position: self.position.position(),
                max_position: self.effective_max_position,
                mid_price: self.latest_mid,
                max_position_value: self.safety.kill_switch.max_position_value(),
                asset: self.config.asset.to_string(),
                margin_used: margin_state.margin_used,
                account_value: margin_state.account_value,
                // Dynamic reduce-only based on liquidation proximity
                liquidation_price: margin_state.liquidation_price,
                liquidation_buffer_ratio: margin_state.liquidation_buffer_ratio(),
                liquidation_trigger_threshold: quoting::DEFAULT_LIQUIDATION_TRIGGER_THRESHOLD,
            };
            quoting::QuoteFilter::apply_reduce_only_single(&mut bid, &mut ask, &reduce_only_config);

            debug!(
                bid = ?bid.as_ref().map(|q| (q.price, q.size)),
                ask = ?ask.as_ref().map(|q| (q.price, q.size)),
                "Calculated quotes"
            );

            // Handle bid side
            self.reconcile_side(Side::Buy, bid).await?;

            // Handle ask side
            self.reconcile_side(Side::Sell, ask).await?;
        }

        // === Post-Quote Tracking ===
        // Record quote for fill rate calculation
        self.infra.fill_tracker.record_quote();

        // Update dashboard with current position
        let position = self.position.position();
        let position_value = position * self.latest_mid;
        self.infra
            .dashboard
            .update_position(position, position_value);

        Ok(())
    }
}
