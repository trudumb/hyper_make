//! Quote generation engine for the market maker.

use chrono::Timelike;
use tracing::{debug, error, info, trace, warn};

use crate::prelude::Result;

use super::super::{
    quoting, MarketMaker, Quote, TradingEnvironment, ParameterAggregator, ParameterSources,
    QuoteConfig, QuotingStrategy, Side,
};
use crate::market_maker::belief::{BeliefSnapshot, BeliefUpdate};
use crate::market_maker::control::{QuoteGateDecision, QuoteGateInput};
use crate::market_maker::estimator::EnhancedFlowContext;
use crate::market_maker::infra::metrics::dashboard::{
    ChangepointDiagnostics, KappaDiagnostics, QuoteDecisionRecord, SignalSnapshot,
};
use crate::market_maker::analytics::ToxicityInput;
use crate::market_maker::config::{CapacityBudget, Viability};
use crate::market_maker::risk::{CircuitBreakerAction, RiskCheckResult};
use crate::market_maker::strategy::action_to_inventory_ratio;

/// Minimum order notional value in USD (Hyperliquid requirement)
pub(super) const MIN_ORDER_NOTIONAL: f64 = 10.0;


impl<S: QuotingStrategy, Env: TradingEnvironment> MarketMaker<S, Env> {
    /// Update quotes based on current market state.
    #[tracing::instrument(name = "quote_cycle", skip_all, fields(asset = %self.config.asset))]
    pub(crate) async fn update_quotes(&mut self) -> Result<()> {
        // === Data Quality Gate ===
        // Block quoting when data is stale, absent, or crossed.
        // This fills the gap between logging (immediate) and the 30s kill switch.
        if let Some(gate_reason) = self.infra.data_quality.should_gate_quotes(&self.config.asset) {
            warn!(%gate_reason, "Gating quotes due to data quality issue");
            // Cancel all existing orders when gated — holding stale quotes is dangerous
            let bid_orders = self.orders.get_all_by_side(Side::Buy);
            let ask_orders = self.orders.get_all_by_side(Side::Sell);
            let all_oids: Vec<u64> = bid_orders
                .iter()
                .chain(ask_orders.iter())
                .map(|o| o.oid)
                .collect();
            if !all_oids.is_empty() {
                self.environment
                    .cancel_bulk_orders(&self.config.asset, all_oids)
                    .await;
            }
            return Ok(());
        }

        // Track when first valid market data arrives (for warmup timeout).
        // Using session_start_time includes WS connection time (~40s for HYPE),
        // which defeats the 30s warmup timeout.
        if self.first_data_time.is_none() {
            info!(
                session_age_s = self.session_start_time.elapsed().as_secs(),
                "First market data received — warmup timeout starts now"
            );
            self.first_data_time = Some(std::time::Instant::now());
        }

        // Don't place orders until estimator is warmed up (or timeout reached)
        if !self.estimator.is_warmed_up() {
            // Check warmup timeout: with informative Bayesian priors, we can safely
            // quote before all data thresholds are met. Spread floors and kill switches
            // provide protection. The estimator continues refining as data arrives.
            //
            // Capital-tier-aware timeout: small accounts need shorter warmup to avoid
            // the death spiral (no fills → no progress → wider spreads → no fills).
            let base_warmup = self.estimator.max_warmup_secs();
            let max_warmup = if self.effective_max_position > 0.0 && self.latest_mid > 0.0 {
                // Quick tier estimate from position limits
                let min_order_size = (MIN_ORDER_NOTIONAL / self.latest_mid).max(0.01);
                let approx_levels = (self.effective_max_position / min_order_size)
                    .floor() as usize / 2;
                match approx_levels {
                    0..=2 => base_warmup.min(10), // Micro: 10s max
                    3..=5 => base_warmup.min(15), // Small: 15s max
                    _ => base_warmup,
                }
            } else {
                base_warmup
            };
            let warmup_base = self.first_data_time.unwrap_or(self.session_start_time);
            let elapsed = warmup_base.elapsed();
            if max_warmup > 0 && elapsed >= std::time::Duration::from_secs(max_warmup) {
                let (vol_ticks, min_vol, trade_obs, min_trades) = self.estimator.warmup_progress();
                warn!(
                    elapsed_secs = elapsed.as_secs(),
                    max_warmup_secs = max_warmup,
                    volume_ticks = vol_ticks,
                    volume_ticks_required = min_vol,
                    trade_observations = trade_obs,
                    trade_observations_required = min_trades,
                    "Warmup timeout reached, starting with Bayesian prior parameters \
                     (kappa prior, config sigma). Estimation continues in background."
                );
                self.estimator.force_warmup_complete();
                // Fall through to normal quoting
            } else {
                // Log warmup status every 10 seconds to help diagnose why orders aren't placed
                let should_log = match self.last_warmup_block_log {
                    None => true,
                    Some(last) => last.elapsed() >= std::time::Duration::from_secs(10),
                };
                if should_log {
                    let (vol_ticks, min_vol, trade_obs, min_trades) =
                        self.estimator.warmup_progress();
                    let remaining = if max_warmup > 0 {
                        max_warmup.saturating_sub(elapsed.as_secs())
                    } else {
                        0
                    };
                    warn!(
                        volume_ticks = vol_ticks,
                        volume_ticks_required = min_vol,
                        trade_observations = trade_obs,
                        trade_observations_required = min_trades,
                        timeout_remaining_secs = remaining,
                        "Warmup incomplete - no orders placed (waiting for market data)"
                    );
                    self.last_warmup_block_log = Some(std::time::Instant::now());
                }
                return Ok(());
            }
        }

        // === QUOTA-AWARE CYCLE THROTTLING ===
        // Reduce update frequency when API quota is low instead of widening spreads.
        // This conserves quota while keeping quotes competitive.
        let headroom_for_throttle = self
            .infra
            .cached_rate_limit
            .as_ref()
            .map(|c| c.headroom_pct())
            .unwrap_or(1.0);
        // Adaptive cycle interval from market conditions.
        // Replaces fixed tiers with sigma-based staleness estimate.
        let sigma_for_cycle = self.estimator.sigma();
        let latch_bps_for_cycle = self
            .cached_market_params
            .as_ref()
            .map(|mp| {
                let half_spread = mp.market_spread_bps / 2.0;
                (half_spread * 0.3).clamp(1.0, 10.0)
            })
            .unwrap_or(3.0);
        let min_cycle_interval_ms = super::event_accumulator::compute_next_cycle_time(
            sigma_for_cycle,
            latch_bps_for_cycle,
            headroom_for_throttle,
        ).as_millis() as u64;
        if let Some(last) = self.last_quote_cycle_time {
            let elapsed_ms = last.elapsed().as_millis() as u64;
            if elapsed_ms < min_cycle_interval_ms {
                tracing::debug!(
                    headroom_pct = %format!("{:.1}%", headroom_for_throttle * 100.0),
                    min_interval_ms = min_cycle_interval_ms,
                    elapsed_ms,
                    "Quota throttle: skipping cycle to conserve API quota"
                );
                return Ok(());
            }
        }
        self.last_quote_cycle_time = Some(std::time::Instant::now());

        // === PHASE 3: INVENTORY GOVERNOR — First structural check ===
        // The inventory governor enforces config.max_position as a HARD ceiling.
        // This runs BEFORE any other logic to ensure position safety is structural.
        let position_assessment = self.inventory_governor.assess(self.position.position());
        if position_assessment.zone == crate::market_maker::risk::PositionZone::Kill {
            warn!(
                position = %format!("{:.4}", self.position.position()),
                ratio = %format!("{:.2}", position_assessment.position_ratio),
                max_position = %format!("{:.4}", self.inventory_governor.max_position()),
                "InventoryGovernor: KILL zone — cancelling ALL quotes"
            );
            let bid_orders = self.orders.get_all_by_side(Side::Buy);
            let ask_orders = self.orders.get_all_by_side(Side::Sell);
            let all_oids: Vec<u64> = bid_orders
                .iter()
                .chain(ask_orders.iter())
                .map(|o| o.oid)
                .collect();
            if !all_oids.is_empty() {
                self.environment
                    .cancel_bulk_orders(&self.config.asset, all_oids)
                    .await;
            }
            return Ok(());
        }

        // === PHASE 7: Graduated risk overlay multipliers ===
        // Track additive spread and size multipliers from all risk sources.
        // Only the kill switch can actually cancel all quotes — everything else
        // applies graduated spread widening and size reduction.
        let mut risk_overlay_mult = 1.0_f64;
        let mut risk_size_reduction = 1.0_f64;
        let mut risk_reduce_only = false;

        // Yellow zone: widen increasing side, cap exposure
        if position_assessment.zone == crate::market_maker::risk::PositionZone::Yellow {
            risk_overlay_mult *= position_assessment.increasing_side_spread_mult;
            debug!(
                zone = "Yellow",
                spread_mult = %format!("{:.2}", position_assessment.increasing_side_spread_mult),
                max_new_exposure = %format!("{:.4}", position_assessment.max_new_exposure),
                "InventoryGovernor: Yellow zone — widening increasing side"
            );
        }
        // Red zone: reduce-only
        if position_assessment.zone == crate::market_maker::risk::PositionZone::Red {
            risk_reduce_only = true;
            risk_overlay_mult *= position_assessment.increasing_side_spread_mult;
            info!(
                zone = "Red",
                "InventoryGovernor: Red zone — reduce-only mode"
            );
        }

        // === Circuit Breaker Checks (Phase 7: graduated) ===
        let breaker_action = self.tier1.circuit_breaker.most_severe_action();
        match breaker_action {
            Some(CircuitBreakerAction::PauseTrading) => {
                // Phase 7: PauseTrading → 5x spread + 10% size, continue (don't cancel)
                risk_overlay_mult *= 5.0;
                risk_size_reduction *= 0.1;
                warn!("Circuit breaker: PauseTrading → graduated 5x spread + 10% size");
            }
            Some(CircuitBreakerAction::CancelAllQuotes) => {
                // Phase 7: CancelAllQuotes → 5x spread, continue (don't cancel)
                risk_overlay_mult *= 5.0;
                warn!("Circuit breaker: CancelAllQuotes → graduated 5x spread (continuing)");
            }
            Some(CircuitBreakerAction::WidenSpreads { multiplier }) => {
                risk_overlay_mult *= multiplier;
                info!(
                    multiplier = %format!("{:.2}x", multiplier),
                    "Circuit breaker: widening spreads"
                );
            }
            None => {}
        }

        // === Risk Limit Checks (Phase 7: graduated) ===
        let position_notional = self.position.position().abs() * self.latest_mid;
        let position_check = self.safety.risk_checker.check_position(position_notional);
        if matches!(position_check, RiskCheckResult::HardLimitBreached { .. }) {
            // Hard limit → 3x spread + reduce-only, continue
            risk_overlay_mult *= 3.0;
            risk_reduce_only = true;
            error!("Hard position limit breached → graduated 3x spread + reduce-only");
        }

        // === Drawdown Check (Phase 7: graduated) ===
        if self.safety.drawdown_tracker.should_pause() {
            // Drawdown pause → 3x spread + 10% size, continue
            risk_overlay_mult *= 3.0;
            risk_size_reduction *= 0.1;
            warn!("Emergency drawdown → graduated 3x spread + 10% size");
        }

        // Cap risk_overlay_mult at 3x (single overlay) to prevent infinite widening
        // The global spread cap downstream handles total composition
        risk_overlay_mult = risk_overlay_mult.min(3.0);

        // === CENTRALIZED BELIEF SYSTEM: Update with price observations ===
        // Phase 7: Removed dual-write - centralized beliefs are now the single source of truth.
        // This feeds price returns to the Bayesian posterior over (μ, σ²).
        // The resulting E[μ | data] becomes belief_predictive_bias in MarketParams.

        // Bootstrap: Seed prev_mid on first valid mid price to enable belief observations
        if self.prev_mid_for_beliefs == 0.0 && self.latest_mid > 0.0 {
            self.prev_mid_for_beliefs = self.latest_mid;
            self.last_beliefs_update = Some(std::time::Instant::now());
            info!(
                seeded_mid = %format!("{:.4}", self.latest_mid),
                "Belief system: seeded prev_mid_for_beliefs for bootstrap"
            );
        }

        let can_observe = self.latest_mid > 0.0 && self.prev_mid_for_beliefs > 0.0;

        if can_observe {
            let price_return = (self.latest_mid - self.prev_mid_for_beliefs) / self.prev_mid_for_beliefs;
            let dt = self.last_beliefs_update
                .map(|t| t.elapsed().as_secs_f64())
                .unwrap_or(1.0)  // Default to 1 second on first observation
                .max(0.001);     // Floor to avoid division issues

            // Phase 7: Primary write to centralized belief state (removed old beliefs_builder.observe_price)
            let timestamp_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            self.central_beliefs.update(BeliefUpdate::PriceReturn {
                return_frac: price_return,
                dt_secs: dt,
                timestamp_ms,
            });

            // Log belief updates periodically (every 10 observations during warmup, then every 100)
            // Phase 7: Use centralized belief snapshot for logging
            let beliefs = self.central_beliefs.snapshot();
            let log_interval = if beliefs.drift_vol.n_observations < 100 { 10 } else { 100 };
            if beliefs.drift_vol.n_observations.is_multiple_of(log_interval) || beliefs.drift_vol.expected_drift.abs() > 0.0005 {
                info!(
                    price_return_bps = %format!("{:.2}", price_return * 10000.0),
                    dt_secs = %format!("{:.3}", dt),
                    predictive_bias_bps = %format!("{:.2}", beliefs.drift_vol.expected_drift * 10000.0),
                    expected_sigma = %format!("{:.6}", beliefs.drift_vol.expected_sigma),
                    drift_confidence = %format!("{:.3}", beliefs.drift_vol.confidence),
                    overall_confidence = %format!("{:.3}", beliefs.overall_confidence()),
                    n_price_obs = beliefs.drift_vol.n_observations,
                    prob_bullish = %format!("{:.3}", beliefs.drift_vol.prob_bullish),
                    prob_bearish = %format!("{:.3}", beliefs.drift_vol.prob_bearish),
                    "Beliefs updated with price observation"
                );
            }
        }

        // Update tracking for next iteration
        self.prev_mid_for_beliefs = self.latest_mid;
        self.last_beliefs_update = Some(std::time::Instant::now());

        // Update calibration controller with current calibration status
        // This uses AS fills measured and kappa confidence to track calibration progress
        // and adjust fill-hungry gamma multiplier accordingly
        let as_fills_measured = self.tier1.adverse_selection.fills_measured() as u64;
        let kappa_confidence = self.estimator.kappa_confidence();
        self.stochastic
            .calibration_controller
            .update_calibration_status(as_fills_measured, kappa_confidence);

        // === Phase 5: QUOTE OUTCOME TRACKING ===
        // Update mid price and expire old pending quotes each cycle.
        // This enables unbiased edge estimation from both filled and unfilled quotes.
        {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            self.quote_outcome_tracker.update_mid(self.latest_mid);
            self.quote_outcome_tracker.expire_old_quotes(now_ms);

            // No-fill feedback for SpreadBandit: if pending selection is >2s old, treat as no-fill
            let pending_expired = self.stochastic.spread_bandit.pending()
                .map(|p| now_ms.saturating_sub(p.timestamp_ms) > 2000)
                .unwrap_or(false);
            if pending_expired {
                let no_fill_reward = self.stochastic.baseline_tracker.counterfactual_reward(0.0);
                self.stochastic.spread_bandit.update_from_pending(no_fill_reward);
            }
        }

        // === CENTRALIZED BELIEF SNAPSHOT (Phase 4) ===
        // Take a point-in-time snapshot of all beliefs for use throughout this quote cycle.
        // This replaces scattered reads from beliefs_builder, regime_hmm, and changepoint.
        let belief_snapshot: BeliefSnapshot = self.central_beliefs.snapshot();

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
            info!("Skipping requote: minimum interval not elapsed");
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

        // AdaptiveEnsemble weight logging moved to learning health block below.

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

            // Phase 5: Log quote outcome stats for unbiased edge estimation
            let outcome_stats = self.quote_outcome_tracker.outcome_stats();
            if outcome_stats.n_total > 0 {
                info!(
                    target: "layer2::calibration",
                    n_total = outcome_stats.n_total,
                    n_filled = outcome_stats.n_filled,
                    fill_rate = %format!("{:.2}%", outcome_stats.fill_rate * 100.0),
                    mean_edge_given_fill = %format!("{:.2}", outcome_stats.mean_edge_given_fill),
                    expected_edge = %format!("{:.2}", outcome_stats.expected_edge),
                    pending = self.quote_outcome_tracker.pending_count(),
                    "[Phase5] Quote outcome tracker stats"
                );
            }

            // Log AdaptiveEnsemble weight distribution
            let summary = self.stochastic.ensemble.summary();
            if summary.total_models > 0 {
                tracing::info!(
                    target: "layer2::ensemble",
                    total = summary.total_models,
                    active = summary.active_models,
                    degraded = summary.degraded_models,
                    best = ?summary.best_model_name,
                    best_ir = %format!("{:.3}", summary.best_model_ir),
                    entropy = %format!("{:.3}", summary.weight_entropy),
                    avg_ir = %format!("{:.3}", summary.average_ir),
                    predictions = summary.total_predictions,
                    "AdaptiveEnsemble weights"
                );
            }

            // Feed empirical fill rates to GLFT elasticity estimator
            let rates = self.quote_outcome_tracker.fill_rate().all_rates();
            for (spread_mid, fill_rate, count) in &rates {
                if *count >= 5 {
                    self.strategy.record_elasticity_observation(*spread_mid, *fill_rate);
                }
            }

            // Log empirical optimal spread for theory vs data comparison
            if let Some(empirical_optimal) = self.quote_outcome_tracker.optimal_spread_bps(0.5) {
                info!(
                    empirical_optimal_bps = %format!("{:.2}", empirical_optimal),
                    "QuoteOutcomeTracker empirical optimal spread"
                );
            }
        }

        // Phase 3: Check recovery state and handle IOC recovery if needed
        if let Some(action) = self.check_and_handle_recovery().await? {
            if action.skip_normal_quoting {
                info!("Recovery manager: skip_normal_quoting=true, returning early");
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

        // === PHASE 3: config.max_position IS the hard ceiling (InventoryGovernor principle) ===
        // Margin can only LOWER the effective limit for solvency, never raise above config.
        let dynamic_max_position_value = self.safety.kill_switch.max_position_value();
        let margin_state = self.infra.margin_sizer.state();
        let dynamic_limit_valid = margin_state.account_value > 0.0;

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

        // config.max_position is the primary ceiling. Margin is a solvency floor only.
        let pre_effective_max_position = if margin_quoting_capacity > 1e-9 {
            margin_quoting_capacity.min(self.config.max_position)
        } else {
            self.config.max_position
        };

        // Update exchange limits with config-derived capacity BEFORE building sources
        self.infra
            .exchange_limits
            .update_local_max(pre_effective_max_position);

        // Pre-compute drift-adjusted skew from HJB controller + momentum signals
        let momentum_bps = self.estimator.momentum_bps();
        let p_continuation = self.estimator.momentum_continuation_probability();
        let position = self.position.position();

        // Get multi-timeframe trend signal BEFORE momentum update so we can blend
        let position_value = (position.abs() * self.latest_mid).max(1.0);
        let trend_signal = self.estimator.trend_signal(position_value);

        // Blend multi-timeframe momentum for drift estimator input.
        // During smooth trends, short-term momentum oscillates near zero while
        // the 5-min trend is clearly directional. Feed the stronger signal.
        let effective_momentum = if trend_signal.is_warmed_up {
            if trend_signal.long_momentum_bps.abs() > momentum_bps.abs()
                && trend_signal.long_momentum_bps.abs() > 5.0
            {
                // Long-term trend dominates: blend 70/30
                0.7 * trend_signal.long_momentum_bps + 0.3 * momentum_bps
            } else if trend_signal.medium_momentum_bps.abs() > momentum_bps.abs()
                && trend_signal.medium_momentum_bps.abs() > 3.0
            {
                // Medium-term trend dominates: blend 50/50
                0.5 * trend_signal.medium_momentum_bps + 0.5 * momentum_bps
            } else {
                momentum_bps
            }
        } else {
            momentum_bps
        };

        // Update momentum EWMA signals for smoothing (reduces whipsawing)
        self.stochastic.hjb_controller.update_momentum_signals(
            effective_momentum,
            p_continuation,
            position,
            self.config.max_position,
        );

        // Update HJB controller's funding-cycle horizon and terminal penalty
        {
            // Wire funding settlement time into HJB controller for natural urgency
            if let Some(ttf) = self.tier2.funding.time_to_next_funding() {
                self.stochastic
                    .hjb_controller
                    .update_funding_settlement(ttf.as_secs_f64());
            }

            // Update HJB funding rate from estimator
            let funding_rate_8h = self.tier2.funding.current_rate();
            self.stochastic
                .hjb_controller
                .update_funding(funding_rate_8h);

            // Feed funding rate to cross-asset signals (Sprint 4.1)
            self.cross_asset_signals.update_funding(funding_rate_8h);

            // Calibrate terminal penalty from observed market spread
            let market_spread_bps = self.tier2.spread_tracker.current_spread_bps();
            self.stochastic
                .hjb_controller
                .calibrate_terminal_penalty(market_spread_bps);
        }

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

        // === Phase 3: Pre-Fill AS Classifier Updates ===
        // Update classifier with latest signals before building market params
        {
            // Trade flow update: convert flow_imbalance [-1, 1] to buy/sell volumes
            // flow_imbalance = (buy - sell) / (buy + sell) => solve for volumes
            let flow_imb = self.estimator.flow_imbalance();
            // Assume normalized volumes: total = 1, then buy = (1 + imb) / 2
            let buy_volume = (1.0 + flow_imb) / 2.0;
            let sell_volume = 1.0 - buy_volume;
            self.tier1.pre_fill_classifier.update_trade_flow(buy_volume, sell_volume);

            // Regime update: use HMM confidence and changepoint probability from beliefs
            let hmm_confidence = belief_snapshot.regime.confidence;
            // Use prob_5 as a smoothed changepoint indicator (recent window)
            let changepoint_prob = belief_snapshot.changepoint.prob_5;
            self.tier1.pre_fill_classifier.update_regime(hmm_confidence, changepoint_prob);

            // Funding rate update
            let funding_rate_8h = self.tier2.funding.current_rate();
            self.tier1.pre_fill_classifier.update_funding(funding_rate_8h);

            // Trend signal update: feed 5-min EWMA momentum to classifier
            // Kyle (1985) drift conditioning — fills against strong trends are more toxic
            if trend_signal.is_warmed_up {
                self.tier1.pre_fill_classifier.update_trend(trend_signal.long_momentum_bps);
            }

            // Enhanced microstructure classifier blend: inject 10-feature toxicity
            // into pre_fill_classifier as external signal (uses z-score normalized
            // trade intensity, price impact, run length, volume imbalance, etc.)
            // Average bid/ask toxicity to get symmetric microstructure signal.
            let enhanced_bid = self.tier1.enhanced_classifier.predict_toxicity(true);
            let enhanced_ask = self.tier1.enhanced_classifier.predict_toxicity(false);
            let enhanced_avg = (enhanced_bid + enhanced_ask) / 2.0;
            self.tier1.pre_fill_classifier.set_blended_toxicity(enhanced_avg);
            // 30% weight to enhanced classifier, 70% to internal 6-signal classifier
            self.tier1.pre_fill_classifier.set_blend_weight(0.3);
        }

        // === Compute OFI-based toxicity regime ===
        // ToxicityRegime classifies {Benign, Normal, Toxic} using base toxicity + OFI acceleration.
        // Must be computed before FeatureSnapshot and ExecutionMode.
        let ofi_1s = self.stochastic.trade_flow_tracker.imbalance_at_1s();
        let ofi_5s = self.stochastic.trade_flow_tracker.imbalance_at_5s();
        self.tier1.pre_fill_classifier.update_ofi(ofi_1s);
        let current_toxicity_regime = self.tier1.pre_fill_classifier.toxicity_regime(ofi_1s, ofi_5s);

        let sources = ParameterSources {
            estimator: &self.estimator,
            adverse_selection: &self.tier1.adverse_selection,
            depth_decay_as: &self.tier1.depth_decay_as,
            pre_fill_classifier: &self.tier1.pre_fill_classifier,
            liquidation_detector: &self.tier1.liquidation_detector,
            hawkes: &self.tier2.hawkes,
            funding: &self.tier2.funding,
            spread_tracker: &self.tier2.spread_tracker,
            hjb_controller: &self.stochastic.hjb_controller,
            margin_sizer: &self.infra.margin_sizer,
            stochastic_config: &self.stochastic.stochastic_config,
            drift_adjusted_skew,
            adaptive_spreads: &self.stochastic.adaptive_spreads,
            // First-principles belief system
            beliefs_builder: &self.stochastic.beliefs_builder,
            // === Phase 5: Centralized belief snapshot ===
            beliefs: Some(&belief_snapshot),
            position: self.position.position(),
            max_position: self.config.max_position,
            latest_mid: self.latest_mid,
            risk_aversion: self.config.risk_aversion,
            // Cached BBO from L2 book (same data reconciler validates against)
            cached_best_bid: self.cached_best_bid,
            cached_best_ask: self.cached_best_ask,
            // CalibrationCoordinator for L2-derived warmup kappa
            calibration_coordinator: &self.calibration_coordinator,
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
            // Derive tick_size_bps from Hyperliquid's 5-significant-figure pricing.
            // E.g. HYPE@$31: tick=$0.001, tick_bps=0.32; BTC@$97k: tick=$1, tick_bps=0.10
            tick_size_bps: compute_tick_size_bps(self.latest_mid),
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
            // Bayesian learned parameters
            learned_params: if self.stochastic.stochastic_config.use_learned_parameters {
                let status = self.stochastic.learned_params.calibration_status();
                Some(super::super::LearnedParameterValues {
                    enabled: true,
                    calibrated: status.tier1_ready,
                    alpha_touch: self.stochastic.learned_params.alpha_touch.estimate(),
                    kappa: self.stochastic.learned_params.kappa.estimate(),
                    spread_floor_bps: self.stochastic.learned_params.spread_floor_bps.estimate(),
                    alpha_touch_n: self.stochastic.learned_params.alpha_touch.n_observations,
                    kappa_n: self.stochastic.learned_params.kappa.n_observations,
                })
            } else {
                None
            },
        };
        let mut market_params = ParameterAggregator::build(&sources);

        // === L1: Parameter Smoothing (Churn Reduction) ===
        // EWMA + deadband filtering on kappa/sigma/gamma before they reach the ladder.
        // Suppresses sub-deadband oscillations (e.g. kappa 1169→1176, 0.6%) that would
        // otherwise cause unnecessary requotes. Gated by SmootherConfig.enabled.
        self.parameter_smoother.smooth(&mut market_params, false);

        // === Fix 4: Hawkes σ_conditional — per-cycle HWM decay ===
        // On each cycle: compute new σ_cascade_mult from Hawkes intensity.
        // Can raise above HWM, but cannot lower below HWM during 15s cooldown.
        // After cooldown, HWM decays exponentially toward 1.0.
        {
            let hawkes_ratio = self.tier2.hawkes.intensity_ratio();
            let new_mult = if hawkes_ratio > 1.0 {
                hawkes_ratio.sqrt().min(3.0)
            } else {
                1.0
            };

            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            let since_last_fill_ms = now_ms.saturating_sub(self.sigma_cascade_hwm_set_at);
            let decay_ms: u64 = 15_000; // 15s cooldown before HWM decays

            let decayed_hwm = if since_last_fill_ms > decay_ms {
                let elapsed_after_cooldown = (since_last_fill_ms - decay_ms) as f64;
                let decay_factor = (-elapsed_after_cooldown / 10_000.0).exp();
                1.0 + (self.sigma_cascade_hwm - 1.0) * decay_factor
            } else {
                self.sigma_cascade_hwm // Hold HWM during cooldown
            };

            // Cycle can raise above HWM, but not lower below it during cooldown
            market_params.sigma_cascade_mult = new_mult.max(decayed_hwm);
        }

        // === Fix 2: Wire AS floor from estimator ===
        {
            let as_floor = self.tier1.adverse_selection.as_floor_bps();
            market_params.as_floor_bps = as_floor.max(self.as_floor_hwm);
        }

        // === Fix 3: Wire ghost liquidity gamma mult from kappa orchestrator ===
        market_params.ghost_liquidity_gamma_mult =
            self.estimator.kappa_orchestrator().ghost_liquidity_gamma_mult();

        // === Construct FeatureSnapshot for downstream models ===
        // Compact projection of market state for toxicity classification,
        // queue-value estimation, and execution mode selection.
        let feature_snapshot = crate::market_maker::features::FeatureSnapshot::from_market_state(
            &market_params,
            &belief_snapshot,
            &self.stochastic.trade_flow_tracker,
            self.position.position(),
            self.config.max_position,
        );

        // === Phase 6: WARMUP GRADUATED UNCERTAINTY ===
        // During warmup (few fills), quote tighter to attract fills and limit size.
        // Warmup discount reduces gamma → tighter GLFT spreads (more fills for learning).
        // Size multiplier limits loss from miscalibrated models during learning.
        //
        // Prior confidence blending: if a prior was injected, its confidence provides
        // a floor that replaces the fill-count-only warmup. This means a 139h-old prior
        // with ~0.2 confidence still starts more aggressively than cold-start.
        let warmup_size_mult = {
            let fill_count = self.tier1.adverse_selection.fills_measured();
            let warmup_gamma_discount = crate::market_maker::calibration::gate::warmup_spread_discount(fill_count);
            let warmup_sz = crate::market_maker::calibration::gate::warmup_size_multiplier(fill_count);

            // Blend fill-count warmup with prior confidence (take the better of the two).
            let prior_conf = self.prior_confidence;
            let (effective_gamma, effective_sz) = if prior_conf > 0.0 {
                use crate::market_maker::calibration::gate::{
                    spread_multiplier_from_confidence, size_multiplier_from_confidence,
                };
                // Prior confidence gives a spread multiplier (>= 1.0).
                // Convert to gamma discount: gamma_discount = 1/spread_mult.
                let prior_gamma = 1.0 / spread_multiplier_from_confidence(prior_conf);
                let prior_sz = size_multiplier_from_confidence(prior_conf);
                // Take the LESS conservative of fill-based and prior-based
                (warmup_gamma_discount.max(prior_gamma), warmup_sz.max(prior_sz))
            } else {
                (warmup_gamma_discount, warmup_sz)
            };

            if effective_gamma < 1.0 || effective_sz < 1.0 {
                market_params.hjb_gamma_multiplier *= effective_gamma;
                debug!(
                    fill_count,
                    gamma_discount = %format!("{:.2}", effective_gamma),
                    size_mult = %format!("{:.2}", effective_sz),
                    prior_confidence = %format!("{:.2}", prior_conf),
                    "Warmup graduated uncertainty active (prior-blended)"
                );
            }
            effective_sz
        };

        // === V2 INTEGRATION: BOCPD Kappa Relationship Tracking ===
        // Use BOCPD to detect when feature→κ relationships change.
        // If in a new regime, widen spreads by reducing effective kappa.
        {
            // Build features for BOCPD: [funding_magnitude, book_depth_velocity, oi_velocity, momentum]
            // Create FundingFeatures from available data
            let funding_rate_8h = self.tier2.funding.ewma_rate();
            let timestamp_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0);
            // Use current rate as prev_rate for simplicity (delta will be ~0)
            let prev_funding_rate = self.tier2.funding.current_rate();
            let funding_features = crate::market_maker::estimator::FundingFeatures::new(
                timestamp_ms,
                funding_rate_8h,
                prev_funding_rate,
            );
            let funding_magnitude = funding_features.funding_magnitude_proximity;
            let book_depth_velocity = 0.0; // TODO: wire from book depth changes
            let oi_velocity = 0.0; // TODO: wire from OI changes
            let momentum_normalized = (market_params.momentum_bps / 10.0).clamp(-1.0, 1.0);
            
            let bocpd_features = [
                funding_magnitude,
                book_depth_velocity,
                oi_velocity,
                momentum_normalized,
            ];
            
            // Get BOCPD prediction and state
            let p_new_regime = self.stochastic.bocpd_kappa.p_new_regime();
            let bocpd_warmed = self.stochastic.bocpd_kappa.is_warmed_up();
            
            // If BOCPD detects new regime, reduce kappa confidence
            // This causes wider spreads when relationships are unstable
            if bocpd_warmed && p_new_regime > 0.3 {
                // Reduce kappa by p_new_regime fraction (new regime = use prior = lower kappa)
                let kappa_discount = 1.0 - (p_new_regime * 0.5); // Max 50% reduction
                let old_kappa = market_params.kappa;
                market_params.kappa *= kappa_discount;
                
                debug!(
                    p_new_regime = %format!("{:.3}", p_new_regime),
                    kappa_discount = %format!("{:.2}", kappa_discount),
                    old_kappa = %format!("{:.0}", old_kappa),
                    new_kappa = %format!("{:.0}", market_params.kappa),
                    funding_mag = %format!("{:.2}", funding_magnitude),
                    "BOCPD: Kappa relationship unstable, widening spreads"
                );
            }
            
            // Store features for BOCPD update after fill (in fill handler)
            // This allows us to update BOCPD with realized kappa from actual fills
            self.stochastic.bocpd_kappa_features = Some(bocpd_features);
        }

        // === V2 INTEGRATION: Belief Skewness Spread Adjustment ===
        // Use fat-tail awareness from belief system to adjust spreads asymmetrically.
        // Positive skewness = higher vol spike risk = widen spreads defensively.
        if belief_snapshot.drift_vol.sigma_skewness > 0.5 {
            let skew = belief_snapshot.drift_vol.sigma_skewness;
            let tail_risk = belief_snapshot.drift_vol.tail_risk_score();
            
            // Adjust spread widening based on skewness
            // High positive skewness = risk of upward vol spike
            if tail_risk > 0.3 {
                let widening = 1.0 + (tail_risk * 0.2); // Up to 20% wider
                market_params.spread_widening_mult *= widening;
                
                debug!(
                    sigma_skewness = %format!("{:.2}", skew),
                    tail_risk = %format!("{:.2}", tail_risk),
                    widening = %format!("{:.2}", widening),
                    "Belief skewness: applying defensive spread widening"
                );
            }
        }

        // Log belief system status (continuously active - no threshold gating)
        // FIRST-PRINCIPLES: Beliefs influence quotes proportional to confidence.
        // confidence=0.20 means 20% of the signal is used, not binary on/off.
        // PHASE 4: Now using centralized belief snapshot instead of scattered reads
        if market_params.use_belief_system {
            let confidence = market_params.belief_confidence;
            let raw_bias_bps = market_params.belief_predictive_bias * 10000.0;
            let effective_bias_bps = raw_bias_bps * confidence;

            // Log every 20 observations or when bias is meaningful
            if effective_bias_bps.abs() > 0.1 || belief_snapshot.drift_vol.n_observations.is_multiple_of(20) {
                info!(
                    raw_belief_bias_bps = %format!("{:.2}", raw_bias_bps),
                    effective_bias_bps = %format!("{:.2}", effective_bias_bps),
                    belief_confidence = %format!("{:.3}", confidence),
                    belief_sigma = %format!("{:.6}", market_params.belief_expected_sigma),
                    drift_prob_bullish = %format!("{:.2}", belief_snapshot.drift_vol.prob_bullish),
                    drift_prob_bearish = %format!("{:.2}", belief_snapshot.drift_vol.prob_bearish),
                    n_price_obs = belief_snapshot.drift_vol.n_observations,
                    "Belief system: confidence-weighted (bias × conf)"
                );
            }
        }

        // FIX: Wire p_momentum_continue from actual momentum model
        // (p_continuation was computed earlier from self.estimator.momentum_continuation_probability())
        // This was hardcoded to 0.5 in aggregator.rs, causing Quote Gate to never open
        market_params.p_momentum_continue = p_continuation;

        // === Position Continuation Model (HOLD/ADD/REDUCE) ===
        // Decide whether to HOLD, ADD, or REDUCE position based on Bayesian continuation probability.
        // This transforms inventory_ratio for GLFT skew calculation:
        // - HOLD: inventory_ratio = 0 (no skew, symmetric quotes)
        // - ADD: inventory_ratio < 0 (reverse skew, tighter on position-building side)
        // - REDUCE: inventory_ratio > 0 (normal skew, tighter on position-reducing side)
        {
            let position = self.position.position();
            let max_position = self.effective_max_position;

            // === Enhanced Multi-Signal Fusion for Position Continuation ===
            // Wire BOCD changepoint, momentum, trend, and regime signals to continuation model.
            // This enables the model to:
            // 1. Discount fill history when changepoint detected (regime shift)
            // 2. Boost continuation when momentum and trend agree with position
            // 3. Use regime-appropriate priors (cascade = high continuation)
            {
                // Get BOCD changepoint signals from centralized belief snapshot
                // PHASE 4: Now using centralized belief snapshot instead of scattered reads
                let changepoint_prob = if belief_snapshot.changepoint.is_warmed_up {
                    belief_snapshot.changepoint.prob_5
                } else {
                    0.0 // Ignore changepoint during BOCD warmup
                };
                let changepoint_entropy = belief_snapshot.changepoint.entropy;

                // Get momentum continuation probability
                let momentum_p = self.estimator.momentum_continuation_probability();

                // Get trend signal
                let position_value = (position.abs() * self.latest_mid).max(1.0);
                let trend_signal = self.estimator.trend_signal(position_value);

                // Get HMM regime probabilities [quiet, normal, bursty, cascade]
                // PHASE 4: Now using centralized belief snapshot instead of scattered reads
                let regime_probs = belief_snapshot.regime.probs;

                // Update continuation model with all signals
                self.stochastic.position_decision.update_signals(
                    changepoint_prob,
                    changepoint_entropy,
                    momentum_p,
                    trend_signal.timeframe_agreement,
                    trend_signal.trend_confidence,
                    regime_probs,
                );

                // Log signal fusion diagnostics when position is significant
                if position.abs() > max_position * 0.05 {
                    let summary = self.stochastic.position_decision.signal_summary();
                    debug!(
                        p_fill_raw = %format!("{:.3}", summary.p_fill_raw),
                        p_fill_discounted = %format!("{:.3}", summary.p_fill_discounted),
                        p_momentum = %format!("{:.3}", summary.p_momentum),
                        p_trend = %format!("{:.3}", summary.p_trend),
                        p_regime = %format!("{:.3}", summary.p_regime),
                        p_fused = %format!("{:.3}", summary.p_fused),
                        cp_discount = %format!("{:.3}", summary.changepoint_discount),
                        fused_conf = %format!("{:.3}", summary.fused_confidence),
                        "Position continuation signal fusion"
                    );
                }
            }

            // Get belief drift and confidence for alignment check
            let belief_drift = market_params.belief_predictive_bias;
            let belief_confidence = market_params.belief_confidence;

            // Compute expected edge from current AS model (approximate)
            let edge_bps = market_params.current_edge_bps;

            // Decide position action using the engine
            let position_action = self.stochastic.position_decision.decide(
                position,
                max_position,
                belief_drift,
                belief_confidence,
                edge_bps,
            );

            // Compute raw inventory ratio
            let raw_inventory_ratio = if max_position > 1e-9 {
                position / max_position
            } else {
                0.0
            };

            // Transform inventory_ratio based on position action
            let effective_inventory_ratio = action_to_inventory_ratio(position_action, raw_inventory_ratio);

            // Update market_params with position continuation values
            market_params.position_action = position_action;
            market_params.continuation_p = self.stochastic.position_decision.prob_continuation();
            market_params.continuation_confidence = self.stochastic.position_decision.confidence();
            market_params.effective_inventory_ratio = effective_inventory_ratio;

            // Log position decisions when position is significant (>1% of max)
            if position.abs() > max_position * 0.01 {
                info!(
                    position_action = ?position_action,
                    continuation_p = %format!("{:.3}", market_params.continuation_p),
                    continuation_conf = %format!("{:.3}", market_params.continuation_confidence),
                    raw_inv_ratio = %format!("{:.3}", raw_inventory_ratio),
                    effective_inv_ratio = %format!("{:.3}", effective_inventory_ratio),
                    belief_drift_bps = %format!("{:.2}", belief_drift * 10000.0),
                    belief_conf = %format!("{:.3}", belief_confidence),
                    "Position continuation decision"
                );
            }
        }

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

        // Wire HMM regime probabilities to MarketParams for soft blending in GLFT
        market_params.regime_probs = regime_probs;

        // Phase 7: Forward regime update to centralized belief state (primary consumer)
        self.central_beliefs.update(BeliefUpdate::RegimeUpdate {
            probs: regime_probs,
            features: None, // Could add HMM features later for diagnostics
        });

        // Phase 7: Forward volatility observation for changepoint detection (primary consumer)
        // BOCD uses volatility observations to detect regime shifts
        if market_params.sigma > 0.0 {
            self.central_beliefs.update(BeliefUpdate::ChangepointObs {
                observation: market_params.sigma,
            });
        }

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

        // === Regime State Machine Update ===
        // Update the centralized regime state from HMM probs, BOCPD, and kappa estimator.
        // This runs BEFORE signal integration so all downstream consumers see current regime.
        let signals_for_regime = self.stochastic.signal_integrator.get_signals();
        let bocpd_cp = if belief_snapshot.changepoint.is_warmed_up {
            belief_snapshot.changepoint.prob_5
        } else {
            0.0
        };
        let regime_changed = self.stochastic.regime_state.update(
            &regime_probs,
            bocpd_cp,
            signals_for_regime.kappa_effective,
        );
        if regime_changed {
            info!(
                regime = self.stochastic.regime_state.regime_label(),
                kappa = %format!("{:.0}", self.stochastic.regime_state.params.kappa),
                floor_bps = %format!("{:.1}", self.stochastic.regime_state.params.spread_floor_bps),
                confidence = %format!("{:.2}", self.stochastic.regime_state.confidence),
                "Regime transition"
            );
        }

        // Wire regime kappa into MarketParams so ladder strategy uses regime-adjusted kappa
        market_params.regime_kappa = Some(self.stochastic.regime_state.params.kappa);
        market_params.regime_kappa_current_regime = match self.stochastic.regime_state.regime {
            crate::market_maker::strategy::regime_state::MarketRegime::Calm => 0,
            crate::market_maker::strategy::regime_state::MarketRegime::Normal => 1,
            crate::market_maker::strategy::regime_state::MarketRegime::Volatile => 2,
            crate::market_maker::strategy::regime_state::MarketRegime::Extreme => 3,
        };

        // === PHASE 4: Wire ALL regime params to market_params ===
        {
            let rp = &self.stochastic.regime_state.params;
            market_params.regime_as_expected_bps = rp.as_expected_bps;
            market_params.regime_risk_premium_bps = rp.risk_premium_bps;
            market_params.regime_skew_gain = rp.skew_gain;
            market_params.regime_size_multiplier = rp.size_multiplier;
            market_params.controller_objective = rp.controller_objective;
            market_params.max_position_fraction = rp.max_position_fraction;
            market_params.regime_gamma_multiplier = rp.gamma_multiplier;
        }

        // === COST-BASIS-AWARE QUOTING ===
        // Thread entry price and unrealized PnL to ladder for breakeven-aware quoting.
        {
            let position = self.position.position();
            let entry = self.tier2.pnl_tracker.avg_entry_price();
            let fee_rate = 0.00015; // 1.5 bps maker fee

            if position.abs() > 1e-9 && entry > 0.0 {
                market_params.avg_entry_price = Some(entry);
                // Breakeven = entry + fees for longs, entry - fees for shorts
                market_params.breakeven_price = if position > 0.0 {
                    entry * (1.0 + fee_rate) // Must sell above this to profit
                } else {
                    entry * (1.0 - fee_rate) // Must buy below this to profit
                };
                let unrealized = self.tier2.pnl_tracker.unrealized_pnl(self.latest_mid);
                let notional = position.abs() * entry;
                market_params.unrealized_pnl_bps = if notional > 0.0 {
                    unrealized / notional * 10_000.0
                } else {
                    0.0
                };
            } else {
                market_params.avg_entry_price = None;
                market_params.breakeven_price = 0.0;
                market_params.unrealized_pnl_bps = 0.0;
            }
        }

        // === Skew Context Update ===
        // Wire inventory + signal skew context before get_signals().
        // set_skew_context() drives inventory-based and signal-based directional skew.
        let alpha_estimate = {
            let flow_dir = self.stochastic.trade_flow_tracker.imbalance_at_5s();
            (0.5 + flow_dir * 0.3).clamp(0.1, 0.9)
        };
        let half_spread_est_bps = (market_params.adaptive_spread_floor * 10000.0).max(5.0);
        self.stochastic.signal_integrator.set_skew_context(
            self.position.position(),
            self.config.max_position,
            alpha_estimate,
            half_spread_est_bps,
        );

        // === Lead-Lag Signal Integration ===
        // Wire cross-exchange lead-lag signal for predictive skew.
        // Uses SignalIntegrator which combines:
        // - Binance → Hyperliquid price discovery (when Binance feed enabled)
        // - Informed flow decomposition
        // - Regime-conditioned kappa
        // - Model gating (IR-based confidence)
        let mut signals = self.stochastic.signal_integrator.get_signals();

        // Inject cross-asset signals (Sprint 4.1): BTC lead-lag + funding divergence
        {
            let funding_rate = self.tier2.funding.current_rate();
            let cross_signal = self.cross_asset_signals.aggregate_signal(funding_rate);
            signals.cross_asset_expected_move_bps = cross_signal.expected_move_bps;
            signals.cross_asset_confidence = cross_signal.confidence;
            signals.cross_asset_vol_mult = self.cross_asset_signals.vol_multiplier();

            // Blend cross-asset skew into combined_skew_bps (additive, confidence-gated)
            // Regime-dependent clamp: cross-asset signal is most valuable during cascades
            if cross_signal.confidence > 0.3 {
                let cross_asset_skew_bps = cross_signal.expected_move_bps
                    * cross_signal.confidence
                    * 0.5; // 50% weight — conservative blending
                let cross_asset_clamp = match self.stochastic.regime_state.regime {
                    crate::market_maker::strategy::regime_state::MarketRegime::Calm
                    | crate::market_maker::strategy::regime_state::MarketRegime::Normal => 3.0,
                    crate::market_maker::strategy::regime_state::MarketRegime::Volatile => 5.0,
                    crate::market_maker::strategy::regime_state::MarketRegime::Extreme => 6.0,
                };
                signals.combined_skew_bps += cross_asset_skew_bps.clamp(-cross_asset_clamp, cross_asset_clamp);
            }

            // OI-based vol multiplier flows into additive risk premium
            if signals.cross_asset_vol_mult > 1.2 {
                let oi_excess = signals.cross_asset_vol_mult - 1.0;
                // 3 bps per unit of OI excess vol (e.g., vol_mult=1.5 → +1.5 bps)
                signals.signal_risk_premium_bps += oi_excess * 3.0;
            }
        }

        // Inject funding rate signals (Sprint 4.2)
        {
            let funding = &self.tier2.funding;
            signals.funding_basis_velocity = funding.basis_velocity();
            signals.funding_premium_alpha = funding.premium_alpha();

            // Funding skew: positive funding → market pays longs → skew asks tighter (attract shorts)
            // Negative funding → market pays shorts → skew bids tighter (attract longs)
            // Scale: 1% annualized funding ≈ 0.3 bps skew
            let funding_rate = funding.current_rate();
            let funding_skew = -funding_rate * 3000.0; // rate is 8h fraction, *3000 → ~bps
            signals.funding_skew_bps = funding_skew.clamp(-2.0, 2.0);

            // Blend funding skew into combined_skew_bps when warmed up
            if funding.is_warmed_up() {
                signals.combined_skew_bps += signals.funding_skew_bps * 0.3; // 30% weight
            }

            // Widen near settlement (last 30 min of 8h cycle): flow becomes unpredictable
            // Additive risk premium: up to +1.5 bps at settlement
            if let Some(ttf) = funding.time_to_next_funding() {
                let time_to_funding_s = ttf.as_secs_f64();
                if time_to_funding_s < 1800.0 && time_to_funding_s > 0.0 {
                    let settlement_proximity = 1.0 - (time_to_funding_s / 1800.0);
                    signals.signal_risk_premium_bps += settlement_proximity * 1.5;
                }
            }
        }

        // Cancel-race excess AS → additive risk premium (Sprint 6.3)
        // If fills arriving after cancel requests show higher AS than normal fills,
        // widen spreads by the excess amount to compensate for cancel-race toxicity.
        self.cancel_race_tracker.update_momentum(signals.combined_skew_bps);
        let cancel_race_excess = self.cancel_race_tracker.excess_race_as_bps();
        if cancel_race_excess > 0.5 {
            // Direct additive: excess AS bps → risk premium, capped at 5 bps
            signals.signal_risk_premium_bps += cancel_race_excess.min(5.0);
        }

        // Record signal contributions for analytics attribution
        self.live_analytics.record_quote_cycle(&signals);

        // Position guard inventory skew (GLFT gamma*sigma^2*q*T) — always additive
        let position_guard_skew_bps = self.safety.position_guard.inventory_skew_bps();

        if signals.lead_lag_actionable {
            // Real cross-exchange signal from Binance feed
            market_params.lead_lag_signal_bps = signals.combined_skew_bps + position_guard_skew_bps;
            market_params.lead_lag_confidence = signals.model_confidence;

            if signals.combined_skew_bps.abs() > 1.0 {
                debug!(
                    diff_bps = %format!("{:.1}", signals.binance_hl_diff_bps),
                    skew_direction = signals.skew_direction,
                    skew_bps = %format!("{:.2}", signals.combined_skew_bps),
                    model_confidence = %format!("{:.2}", signals.model_confidence),
                    "Lead-lag signal ACTIVE (Binance feed)"
                );
            }
        } else {
            // Fallback: use legacy momentum-based proxy when Binance feed not available
            let lag_model = &self.stochastic.model_calibration.lag_model;
            if lag_model.is_warmed_up() {
                let analyzer = lag_model.analyzer();
                let mi = analyzer.best_lag_mi();
                let lag_ms = analyzer.best_lag_ms();

                // Confidence based on mutual information (0.02 bits = threshold, 0.1 bits = full)
                let confidence = ((mi - 0.02) / 0.08).clamp(0.0, 1.0);
                market_params.lead_lag_confidence = confidence;

                if lag_ms < 0 && confidence > 0.3 {
                    // Negative lag = signal leads target
                    // Amplify momentum signal by confidence (fallback mode)
                    market_params.lead_lag_signal_bps = market_params.momentum_bps * confidence * 0.5;

                    if market_params.lead_lag_signal_bps.abs() > 1.0 {
                        trace!(
                            lag_ms = lag_ms,
                            mi = %format!("{:.4}", mi),
                            confidence = %format!("{:.2}", confidence),
                            signal_bps = %format!("{:.2}", market_params.lead_lag_signal_bps),
                            "Lead-lag signal active (momentum fallback)"
                        );
                    }
                }
            }
            // Always add position guard skew in fallback path
            market_params.lead_lag_signal_bps += position_guard_skew_bps;
        }

        // Apply informed flow spread multiplier from SignalIntegrator
        // This widens spreads when high P(informed) is detected
        if signals.informed_flow_spread_mult > 1.01 {
            trace!(
                p_informed = %format!("{:.2}", signals.p_informed),
                spread_mult = %format!("{:.2}x", signals.informed_flow_spread_mult),
                "Informed flow spread widening"
            );
        }

        // === Cross-Venue Beliefs Integration (Bivariate Flow Model) ===
        // Use cross-venue signals from joint Binance + Hyperliquid analysis.
        // When venues agree, boost confidence in direction. When they disagree, widen spreads.
        if signals.cross_venue_valid {
            // Apply cross-venue spread multiplier (widens when venues disagree or high toxicity)
            market_params.spread_widening_mult *= signals.cross_venue_spread_mult;

            // Add cross-venue skew to lead-lag signal (directional boost from agreement)
            // Scale by agreement: high agreement = strong signal, disagreement = muted
            let cv_skew_contribution = signals.cross_venue_skew
                * signals.cross_venue_confidence
                * 5.0; // Convert [-1,1] direction to bps (max ±5 bps contribution)

            if signals.cross_venue_agreement > 0.5 {
                // Venues agree - boost directional signal
                market_params.lead_lag_signal_bps += cv_skew_contribution;
            }

            // Update central belief state with cross-venue observations
            let timestamp_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);

            self.central_beliefs.update(BeliefUpdate::CrossVenueUpdate {
                direction: signals.cross_venue_direction,
                confidence: signals.cross_venue_confidence,
                discovery_venue: signals.cross_venue_intensity_ratio, // 0=HL, 1=Binance
                max_toxicity: signals.cross_venue_max_toxicity,
                avg_toxicity: signals.cross_venue_avg_toxicity,
                agreement: signals.cross_venue_agreement,
                divergence: signals.cross_venue_divergence,
                intensity_ratio: signals.cross_venue_intensity_ratio,
                imbalance_correlation: signals.cross_venue_imbalance_correlation,
                toxicity_alert: signals.cross_venue_max_toxicity > 0.7,
                divergence_alert: signals.cross_venue_divergence > 0.4,
                timestamp_ms,
            });

            // Log cross-venue diagnostics when actionable
            if signals.cross_venue_spread_mult > 1.05
                || signals.cross_venue_direction.abs() > 0.3
                || signals.cross_venue_max_toxicity > 0.5
            {
                debug!(
                    direction = %format!("{:.2}", signals.cross_venue_direction),
                    confidence = %format!("{:.2}", signals.cross_venue_confidence),
                    agreement = %format!("{:.2}", signals.cross_venue_agreement),
                    max_toxicity = %format!("{:.2}", signals.cross_venue_max_toxicity),
                    spread_mult = %format!("{:.2}x", signals.cross_venue_spread_mult),
                    skew_contribution_bps = %format!("{:.2}", cv_skew_contribution),
                    "Cross-venue signal ACTIVE"
                );
            }
        }

        // === CONTROLLER RISK OVERLAY (Phase 4: unified pipeline) ===
        // Risk assessment replaces binary Quote/NoQuote with continuous multipliers.
        // Called early so risk_overlay_mult and size can be adjusted.
        let risk_overlay = self.stochastic.controller.risk_assessment();
        let risk_level = risk_overlay.risk_level;
        match risk_level {
            crate::market_maker::control::RiskLevel::KillSwitch => {
                info!(reason = %risk_overlay.reason, "Risk overlay: kill switch — cancelling ALL quotes");
                return Ok(());
            }
            crate::market_maker::control::RiskLevel::Emergency => {
                info!(reason = %risk_overlay.reason, "Risk overlay: emergency — position-reducing only");
                risk_reduce_only = true;
                risk_overlay_mult *= 2.0;
            }
            crate::market_maker::control::RiskLevel::Elevated => {
                debug!(reason = %risk_overlay.reason, "Risk overlay: elevated risk");
                risk_overlay_mult *= risk_overlay.spread_multiplier.max(1.0);
            }
            crate::market_maker::control::RiskLevel::Normal => {}
        }

        // === PHASE 5: Additive risk premium replaces multiplicative spread chain ===
        // Instead of compounding N multipliers (which can blow up to 10x+), we compute
        // an additive risk premium in bps that flows to solve_min_gamma() for self-consistent spreads.
        let mut total_risk_premium = market_params.regime_risk_premium_bps;

        // Hawkes synchronicity addon: 0-3 bps when synchronicity > 0.5
        let sync_coeff = self.stochastic.hawkes_predictor.synchronicity_coefficient();
        let hawkes_addon_bps = if sync_coeff > 0.5 {
            3.0 * (sync_coeff - 0.5) * 2.0 // 0-3 bps
        } else {
            0.0
        };
        total_risk_premium += hawkes_addon_bps;

        // Toxicity addon: 0-5 bps when composite score > 0.5
        let toxicity_input = ToxicityInput {
            vpin: {
                let v = signals.hl_vpin;
                if v > 0.0 && v <= 1.0 { Some(v) } else { None }
            },
            vpin_velocity: {
                let v = signals.hl_vpin_velocity;
                if v.is_finite() { Some(v) } else { None }
            },
            p_informed: signals.p_informed,
            trend_long_bps: if trend_signal.is_warmed_up {
                Some(trend_signal.long_momentum_bps)
            } else {
                None
            },
            trend_agreement: if trend_signal.is_warmed_up {
                Some(trend_signal.timeframe_agreement)
            } else {
                None
            },
            book_imbalance: market_params.book_imbalance,
            price_velocity_1s: self.price_velocity_1s,
        };
        let toxicity = self.tier2.toxicity.evaluate(&toxicity_input);
        let toxicity_addon_bps = if toxicity.composite_score > 0.5 {
            5.0 * (toxicity.composite_score - 0.5) * 2.0 // 0-5 bps
        } else {
            0.0
        };
        total_risk_premium += toxicity_addon_bps;

        // Staleness addon: 0-3 bps from signal staleness
        let staleness_mult = self.stochastic.signal_integrator.staleness_spread_multiplier();
        let staleness_addon_bps = if staleness_mult > 1.0 {
            // Convert multiplicative staleness to additive: (mult - 1) * base_spread_bps
            // Use 3 bps cap as proxy for staleness penalty
            ((staleness_mult - 1.0) * 5.0).min(3.0)
        } else {
            0.0
        };
        total_risk_premium += staleness_addon_bps;

        // Signal-derived risk premium: OI vol, funding settlement, cancel-race AS
        // Accumulated on signals.signal_risk_premium_bps during signal injection above
        total_risk_premium += signals.signal_risk_premium_bps;

        // Hidden liquidity discount: when iceberg detected on both sides, reduce premium by 20%
        // IcebergDetector is not yet wired on StochasticComponents — use TODO for future integration
        // TODO: Wire IcebergDetector to StochasticComponents for hidden liquidity discount
        // if self.stochastic.iceberg_detector.has_bilateral_support() {
        //     total_risk_premium *= 0.8;
        // }

        // Store total risk premium on market_params for solve_min_gamma()
        market_params.total_risk_premium_bps = total_risk_premium;

        // === Compose final spread_widening_mult ===
        // Phase 7: Single risk_overlay_mult (from graduated responses above), capped at 3x
        let spread_multiplier = risk_overlay_mult.min(3.0);
        if spread_multiplier > 1.01 {
            market_params.spread_widening_mult *= spread_multiplier;
        }

        if total_risk_premium > 0.1 || spread_multiplier > 1.01 {
            debug!(
                regime_premium = %format!("{:.1}", market_params.regime_risk_premium_bps),
                hawkes_addon = %format!("{:.1}", hawkes_addon_bps),
                toxicity_addon = %format!("{:.1}", toxicity_addon_bps),
                staleness_addon = %format!("{:.1}", staleness_addon_bps),
                total_risk_premium = %format!("{:.1}", total_risk_premium),
                risk_overlay = %format!("{:.2}x", spread_multiplier),
                toxicity_score = %format!("{:.2}", toxicity.composite_score),
                sync_coeff = %format!("{:.2}", sync_coeff),
                "Additive risk premium + risk overlay"
            );
        }

        // Apply position-based size reduction from risk checker and drawdown tracker
        let risk_size_mult = self
            .safety
            .risk_checker
            .suggested_size_multiplier(self.position.position().abs() * self.latest_mid);
        let drawdown_size_mult = self.safety.drawdown_tracker.position_multiplier();
        let size_multiplier = risk_size_mult * drawdown_size_mult * risk_size_reduction;

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
            self.config.target_liquidity, // Fix 6: config target for geometric blend
        );

        // === PHASE 3+4: effective_max_position from InventoryGovernor + regime tightening ===
        // config.max_position is the hard ceiling. Margin is solvency floor only.
        // Regime tightens further: Extreme → 30% of max (via max_position_fraction).
        //
        // FIX: Use unified_regime().max_position_fraction (clamped [0.3, 1.0]) instead
        // of raw params.max_position_fraction. The raw value can be arbitrarily low
        // from checkpoint deserialization or uncalibrated HMM states.
        //
        // COLD-START BYPASS: With 0 fills, the regime HMM has no fill-based data
        // to calibrate position fractions. Using it would reduce effective_max by
        // 30-70%, shrinking the ladder budget and preventing fills.
        // "Quote First, Learn Second" — full capacity until first fills.
        let margin_effective = market_params.effective_max_position(self.config.max_position);
        let unified = self.stochastic.regime_state.unified_regime();
        let fill_count = self.tier1.adverse_selection.fills_measured();
        let regime_fraction = if fill_count < 10 {
            1.0 // No regime tightening until we have fill data
        } else {
            unified.max_position_fraction // Clamped to [0.3, 1.0] by unified_regime()
        };
        // Kelly-aware position scaling (Sprint 5.3): when Kelly tracker recommends
        // reducing exposure (low win rate or bad odds), scale down position limits.
        // Gated by kelly_warmed_up() to avoid reducing limits during warmup.
        let kelly_position_mult = if self.stochastic.stochastic_config.use_kelly_sizing {
            if let Some(raw_kelly) = self.learning.kelly_recommendation() {
                // Scale: full Kelly of 0.25 → mult 1.0, Kelly of 0.05 → mult 0.2
                // Floor at 0.1 to always keep some market presence
                (raw_kelly / self.stochastic.stochastic_config.kelly_fraction)
                    .clamp(0.1, 1.0)
            } else {
                1.0 // Not warmed up yet
            }
        } else {
            1.0
        };
        let new_effective = margin_effective
            .min(self.config.max_position)
            * regime_fraction
            * kelly_position_mult;
        if (new_effective - self.effective_max_position).abs() > 0.001 {
            debug!(
                old = %format!("{:.6}", self.effective_max_position),
                new = %format!("{:.6}", new_effective),
                margin_derived = %format!("{:.6}", margin_effective),
                config_max = %format!("{:.6}", self.config.max_position),
                regime_fraction = %format!("{:.2}", regime_fraction),
                raw_regime_fraction = %format!("{:.2}", market_params.max_position_fraction),
                fill_count = fill_count,
                "Effective max position updated (config cap × regime fraction)"
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
        // User-specified max_position is an absolute hard cap
        if self.config.max_position_user_specified {
            self.effective_max_position = self.effective_max_position.min(self.config.max_position);
        }

        // No-signal safety mode: reduce position limits when no cross-venue feed available
        let signal_limit_mult = self.stochastic.signal_integrator.signal_position_limit_mult();
        if signal_limit_mult < 1.0 {
            self.effective_max_position *= signal_limit_mult;
            tracing::debug!(
                signal_limit_mult = %format!("{:.2}", signal_limit_mult),
                effective_max = %format!("{:.4}", self.effective_max_position),
                "No-signal safety mode: reduced effective max position"
            );
        }

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
        // Risk overlay size_multiplier scales down when trust is low or changepoint detected.
        // Warmup size_multiplier limits position size while models are learning.
        let risk_size_mult = risk_overlay.size_multiplier;
        let combined_size_mult = risk_size_mult * warmup_size_mult;
        let new_effective_liquidity = if market_params.derived_target_liquidity > 0.0 {
            // Fix 6: GLFT-derived liquidity is already geometrically blended
            // with config target in market_params derivation. No double-cap here.
            // Config is INPUT to geometric blend, not a post-cap.
            (market_params
                .derived_target_liquidity
                * combined_size_mult)  // Scale by controller trust + warmup
                .max(min_viable_liquidity) // Exchange minimum
                .min(self.effective_max_position) // Position limit
        } else {
            // Fallback: use config (warmup or zero account_value)
            (self.config
                .target_liquidity
                * combined_size_mult)  // Scale by controller trust + warmup
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

            // Get changepoint info from centralized belief snapshot
            // PHASE 4: Now using centralized belief snapshot instead of scattered reads
            let cp_prob_5 = belief_snapshot.changepoint.prob_5;
            let cp_detected = belief_snapshot.changepoint.result != crate::market_maker::belief::ChangepointResult::None;

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
                observation_count: self.tier1.adverse_selection.fills_measured(),
                mean_distance_bps: 0.0, // TODO: get from estimator
            };
            self.infra
                .prometheus
                .dashboard()
                .update_kappa_diagnostics(kappa_diag);

            // Update changepoint diagnostics from centralized belief snapshot
            // PHASE 4: Now using centralized belief snapshot instead of scattered reads
            {
                use crate::market_maker::belief::ChangepointResult;
                let cp_diag = ChangepointDiagnostics {
                    cp_prob_1: belief_snapshot.changepoint.prob_1,
                    cp_prob_5: belief_snapshot.changepoint.prob_5,
                    cp_prob_10: belief_snapshot.changepoint.prob_10,
                    run_length: belief_snapshot.changepoint.run_length as u32,
                    entropy: belief_snapshot.changepoint.entropy,
                    detected: belief_snapshot.changepoint.result != ChangepointResult::None,
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
            // Get current drawdown from risk state, but suppress when peak is noise.
            // Drawdown is meaningless when peak PnL is from a single fill's spread capture
            // (e.g. $0.02). Without this guard the decision engine blocks ALL quoting
            // at 5% drawdown even when the "peak" is spread noise.
            let risk_state = self.build_risk_state();
            let min_peak = self.safety.kill_switch.config().min_peak_for_drawdown;
            let current_drawdown = if risk_state.peak_pnl < min_peak {
                0.0
            } else {
                risk_state.drawdown()
            };

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
                    ..  // spread_multiplier removed - uncertainty flows through gamma
                } => {
                    debug!(
                        size_fraction = %format!("{:.0}%", size_fraction * 100.0),
                        confidence = %format!("{:.1}%", confidence * 100.0),
                        expected_edge = %format!("{:.2}bp", expected_edge),
                        reservation_shift = %format!("{:.6}", reservation_shift),
                        "Decision engine: quoting"
                    );
                    // Set L2 outputs on market_params for downstream strategy
                    market_params.l2_reservation_shift = reservation_shift;
                    // NOTE: l2_spread_multiplier removed - uncertainty flows through gamma
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

        // === KELLY CRITERION SIZING ===
        // Kelly is applied once to effective_max_position (line ~1540) which governs
        // how much capital to deploy. Applying it again here to liquidity would compound
        // the reduction (W1 audit fix). Use decision_adjusted_liquidity directly.
        let kelly_adjusted_liquidity = decision_adjusted_liquidity;

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
                rate_limit_headroom: self
                    .infra
                    .cached_rate_limit
                    .as_ref()
                    .map(|c| c.headroom_pct())
                    .unwrap_or(1.0),
                last_realized_edge_bps: self.tier2.edge_tracker.last_realized_edge_bps(),
                market_spread_bps: market_params.market_spread_bps,
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
                // PHASE 4: Using centralized belief snapshot for changepoint values
                l3_trust = %format!("{:.2}", belief_snapshot.changepoint.learning_trust),
                l3_cp_prob = %format!("{:.3}", belief_snapshot.changepoint.prob_5),
                l3_action = %format!("{action:?}").chars().take(40).collect::<String>(),
                // Final
                position = %format!("{:.4}", self.position.position()),
                "[Trace] L1->L2->L3 pipeline"
            );

            // Phase 4: Controller action logged for diagnostics only.
            // Quoting decision flows from ensemble, not controller.
            // Emergency pull is handled earlier via risk_assessment().
            debug!(
                action = %format!("{action:?}").chars().take(60).collect::<String>(),
                risk_size = %format!("{:.2}", risk_overlay.size_multiplier),
                risk_spread = %format!("{:.2}", risk_overlay.spread_multiplier),
                risk_reason = %risk_overlay.reason,
                "Controller action (diagnostic) + risk overlay"
            );
        }

        // === CALIBRATED EDGE SIGNAL: Track whether flow_imbalance predicts price direction ===
        // This builds IR (Information Ratio) data for principled threshold derivation.
        // 1. Record outcomes for mature predictions (from previous cycles)
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.stochastic
            .calibrated_edge
            .record_outcomes(self.latest_mid, now_ms);

        // 2. Make a new prediction for this cycle
        // Infer regime from HMM: [p_low, p_normal, p_high, p_extreme]
        let regime_idx = if p_extreme > 0.3 {
            2 // cascade
        } else if p_high > 0.3 || p_low > 0.5 {
            1 // volatile
        } else {
            0 // calm
        };
        let _pred_id = self.stochastic.calibrated_edge.predict(
            market_params.flow_imbalance,
            self.latest_mid,
            regime_idx,
            now_ms,
        );

        // === QUOTE GATE: Decide WHETHER to quote based on directional edge ===
        // This prevents whipsaw losses from random fills when we have no directional edge.
        // The gate uses flow_imbalance, momentum confidence, and position to decide.

        // === Phase 1: Compute Enhanced Flow ===
        // Enhanced flow uses multiple signals for better IR calibration variance
        // Import BookLevel and TradeData types for conversion
        use crate::market_maker::estimator::{BookLevel, TradeData};

        let enhanced_flow_ctx = EnhancedFlowContext {
            book_imbalance: market_params.book_imbalance,
            // Wire cached L2 book sizes (top 5 levels) for depth imbalance calculation
            bid_levels: self
                .cached_bid_sizes
                .iter()
                .map(|&size| BookLevel { size })
                .collect(),
            ask_levels: self
                .cached_ask_sizes
                .iter()
                .map(|&size| BookLevel { size })
                .collect(),
            // Wire cached trades for momentum calculation
            recent_trades: self
                .cached_trades
                .iter()
                .map(|&(size, is_buy, timestamp_ms)| TradeData {
                    size,
                    is_buy,
                    timestamp_ms,
                })
                .collect(),
            kappa_effective: market_params.kappa,
            kappa_avg: self.stochastic.enhanced_flow.avg_kappa(),
            spread_bps: market_params.market_spread_bps.max(1.0),
            now_ms,
        };
        let enhanced_flow_result = self.stochastic.enhanced_flow.compute(&enhanced_flow_ctx);
        let enhanced_flow = enhanced_flow_result.enhanced_flow;
        // Update kappa EMA for future computations
        self.stochastic.enhanced_flow.update_kappa(market_params.kappa);

        // Log enhanced flow diagnostics when depth or momentum contribute
        // This helps verify the L2/trade wiring is working
        if enhanced_flow_result.depth_component.abs() > 0.01
            || enhanced_flow_result.momentum_component.abs() > 0.01
        {
            debug!(
                enhanced_flow = %format!("{:.3}", enhanced_flow),
                base_flow = %format!("{:.3}", enhanced_flow_result.base_flow),
                depth_component = %format!("{:.3}", enhanced_flow_result.depth_component),
                momentum_component = %format!("{:.3}", enhanced_flow_result.momentum_component),
                kappa_component = %format!("{:.3}", enhanced_flow_result.kappa_component),
                flow_variance = %format!("{:.4}", enhanced_flow_result.flow_variance),
                bid_levels = self.cached_bid_sizes.len(),
                ask_levels = self.cached_ask_sizes.len(),
                recent_trades = self.cached_trades.len(),
                "Enhanced flow with L2/trade data"
            );
        }

        // === Phase 3: Compute MC EV for kappa-driven override ===
        // Only compute when IR not calibrated and kappa is strong
        let mc_ev_bps = if market_params.kappa > 1500.0 && !self.stochastic.calibrated_edge.is_useful() {
            let mc_result = self.stochastic.mc_simulator.simulate_ev(
                market_params.kappa,
                enhanced_flow,
                market_params.market_spread_bps.max(1.0),
                1.0, // 1 second horizon
            );
            Some(mc_result.expected_ev_bps)
        } else {
            None
        };

        // Kappa-driven spread cap removed — circular with GLFT (which already uses kappa).
        // GLFT delta = (1/gamma) * ln(1 + gamma/kappa) is the self-consistent spread.
        self.stochastic.kappa_spread.update_avg_kappa(market_params.kappa);

        // === Phase 8: Contextual Bandit Spread Selection ===
        // Replaces RL MDP with contextual bandit (i.i.d. rewards, correct statistics).
        // Build context from current market features
        let position_frac = self.position.position() / self.effective_max_position.max(1.0);
        let vol_ratio_bandit = market_params.sigma / market_params.sigma_effective.max(0.0001);
        let regime_idx = self.stochastic.regime_hmm.most_likely_regime();
        let bandit_context = crate::market_maker::learning::SpreadContext::from_continuous(
            regime_idx,
            position_frac,
            vol_ratio_bandit,
            market_params.flow_imbalance,
        );

        // Select arm via Thompson Sampling (cold start defaults to mult=1.0 = pure GLFT)
        let bandit_selection = self.stochastic.spread_bandit.select_arm(bandit_context);
        market_params.bandit_spread_multiplier = bandit_selection.multiplier;
        market_params.bandit_is_exploration = bandit_selection.is_exploration;

        // Legacy RL fields — kept at defaults for backward-compatible logging
        market_params.rl_spread_delta_bps = 0.0;
        market_params.rl_bid_skew_bps = 0.0;
        market_params.rl_ask_skew_bps = 0.0;
        market_params.rl_confidence = 0.0;
        market_params.rl_is_exploration = false;
        market_params.rl_expected_q = 0.0;

        // === Phase 8: Competitor Model Inference ===
        // Get competitor summary and populate MarketParams
        let competitor_summary = self.stochastic.competitor_model.summary();
        market_params.competitor_snipe_prob = competitor_summary.snipe_rate;
        market_params.competitor_spread_factor = competitor_summary.competition_spread_factor;
        market_params.competitor_count = competitor_summary.n_competitors;

        // === L2 WIRING: Get learning module output for Quote Gate ===
        // This provides p_positive_edge and model_health for hierarchical fusion.
        // Previously this was only computed inside the controller block, but Quote Gate
        // needs it for better decision making.
        let l2_output = if self.learning.is_enabled() {
            // Calculate drawdown from PnL summary, suppressing when peak is noise.
            let pnl_summary = self.tier2.pnl_tracker.summary(self.latest_mid);
            let min_peak = self.safety.kill_switch.config().min_peak_for_drawdown;
            let current_drawdown = if pnl_summary.peak_pnl < min_peak {
                0.0 // Drawdown meaningless when peak is spread noise
            } else if pnl_summary.peak_pnl > 0.0 {
                (pnl_summary.peak_pnl - pnl_summary.total_pnl) / pnl_summary.peak_pnl
            } else {
                0.0
            };
            Some(self.learning.output(
                &market_params,
                self.position.position(),
                current_drawdown,
            ))
        } else {
            None
        };

        // Extract L2 values for Quote Gate, with recalibration adjustment
        let (l2_p_positive, l2_health_score) = if let Some(ref output) = l2_output {
            // Check if edge model needs recalibration (persistent bias > 1.5 bps)
            let tracker = self.learning.confidence_tracker();
            if let Some(recal_bias) = tracker.edge_bias_tracker().should_recalibrate() {
                // Adjust prediction mean by subtracting the bias and recompute p_positive_edge
                let adjusted_mean = output.edge_prediction.mean - recal_bias;
                let sigma_mu = output.edge_prediction.std.max(0.001);
                let z = adjusted_mean / sigma_mu;
                let adjusted_p = crate::market_maker::control::types::normal_cdf(z);
                debug!(
                    original_p = %format!("{:.3}", output.p_positive_edge),
                    adjusted_p = %format!("{:.3}", adjusted_p),
                    bias_bps = %format!("{:.2}", recal_bias),
                    "Edge bias recalibration: adjusting p_positive_edge"
                );
                (Some(adjusted_p), output.model_health.to_score())
            } else {
                (Some(output.p_positive_edge), output.model_health.to_score())
            }
        } else {
            (None, 0.0)
        };

        let quote_gate_input = QuoteGateInput {
            flow_imbalance: market_params.flow_imbalance,
            momentum_confidence: market_params.p_momentum_continue,
            momentum_bps: market_params.momentum_bps,
            position: self.position.position(),
            max_position: self.effective_max_position,
            is_warmup: !self.estimator.is_warmed_up(),
            cascade_size_factor: market_params.cascade_size_factor,
            // Fields for theoretical edge calculation
            book_imbalance: market_params.book_imbalance,
            spread_bps: market_params.market_spread_bps.max(1.0), // Floor to 1 bps
            sigma: market_params.sigma,
            tau_seconds: 1.0, // Default holding horizon of 1 second
            // Fields for enhanced flow and kappa-driven decisions
            kappa_effective: market_params.kappa,
            enhanced_flow,
            mc_ev_bps,
            // Hierarchical Edge Belief (L2/L3 Fusion)
            // L2 inputs from learning module - NOW WIRED!
            l2_p_positive_edge: l2_p_positive,
            l2_model_health: l2_health_score,
            // L3 inputs from stochastic controller
            // Cap L3 trust based on capital-aware policy to prevent death spiral:
            // with few fills, tautological edge causes L3 to ramp trust → kill quoting.
            // Policy caps trust at e.g. 0.30 for Micro until min_fills_for_trust_ramp fills.
            l3_trust: {
                let raw_trust = market_params.bootstrap_confidence.min(1.0);
                let policy = &market_params.capital_policy;
                let fills = self.tier1.adverse_selection.fills_measured() as u64;
                if fills < policy.min_fills_for_trust_ramp {
                    raw_trust.min(policy.max_l3_trust_uncalibrated)
                } else {
                    raw_trust
                }
            },
            l3_belief: if market_params.should_quote_edge {
                Some(0.6) // Favorable conditions
            } else {
                Some(0.4) // Unfavorable conditions
            },
            urgency_score: market_params.urgency_score,
            // Adverse selection variance from RegimeAwareBayesianAdverse
            adverse_variance: market_params.adverse_uncertainty.powi(2),
            // Phase 7: Hawkes Excitation fields from MarketParams
            hawkes_p_cluster: market_params.hawkes_p_cluster,
            hawkes_excitation_penalty: market_params.hawkes_excitation_penalty,
            hawkes_is_high_excitation: market_params.hawkes_is_high_excitation,
            hawkes_spread_widening: market_params.hawkes_spread_widening,
            hawkes_branching_ratio: market_params.hawkes_branching_ratio,
            // Phase 8: RL Policy Recommendations from MarketParams
            rl_spread_delta_bps: market_params.rl_spread_delta_bps,
            rl_bid_skew_bps: market_params.rl_bid_skew_bps,
            rl_ask_skew_bps: market_params.rl_ask_skew_bps,
            rl_confidence: market_params.rl_confidence,
            rl_is_exploration: market_params.rl_is_exploration,
            rl_expected_q: market_params.rl_expected_q,
            // Phase 8: Competitor Model from MarketParams
            competitor_snipe_prob: market_params.competitor_snipe_prob,
            competitor_spread_factor: market_params.competitor_spread_factor,
            competitor_count: market_params.competitor_count,
            // Phase 9: Rate Limit Shadow Price
            // Get headroom from cached rate limit info
            rate_limit_headroom_pct: self
                .infra
                .cached_rate_limit
                .as_ref()
                .map(|c| c.headroom_pct())
                .unwrap_or(1.0),
            // Derive vol_regime from sigma relative to baseline
            vol_regime: {
                let vol_ratio = market_params.sigma / 0.0001; // Relative to 1bp/sec baseline
                if vol_ratio < 0.5 {
                    0 // Low
                } else if vol_ratio < 2.0 {
                    1 // Normal
                } else if vol_ratio < 5.0 {
                    2 // High
                } else {
                    3 // Extreme
                }
            },
            // Phase 10: Pre-Fill AS Toxicity from MarketParams
            pre_fill_toxicity_bid: market_params.pre_fill_toxicity_bid,
            pre_fill_toxicity_ask: market_params.pre_fill_toxicity_ask,
            // Check if pre-fill signals are stale (using 5 second threshold for critical signals)
            pre_fill_signals_stale: self.tier1.pre_fill_classifier.has_stale_signals(5000),
            // Phase 6: Pass centralized belief snapshot for unified decision making
            beliefs: Some(belief_snapshot.clone()),
        };

        // Log L2 wiring status periodically (every 100 cycles or when p_positive is significant)
        if let Some(l2_p) = l2_p_positive {
            if (l2_p - 0.5).abs() > 0.05 || self.learning.pending_predictions_count().is_multiple_of(100) {
                debug!(
                    l2_p_positive_edge = %format!("{:.3}", l2_p),
                    l2_model_health = %format!("{:.2}", l2_health_score),
                    "L2 learning module wired to quote gate"
                );
            }
        }

        // Use calibrated decision with theoretical fallback if enabled, otherwise use legacy thresholds
        // IMPORTANT: Only use changepoint probability after BOCD warmup to avoid
        // false positives at startup (BOCD always shows cp_prob=1.0 before warmup)
        // PHASE 4: Now using centralized belief snapshot instead of scattered reads
        let changepoint_prob = if belief_snapshot.changepoint.is_warmed_up {
            belief_snapshot.changepoint.prob_5
        } else {
            // Log during warmup so we can verify the fix is working
            if belief_snapshot.changepoint.observation_count.is_multiple_of(10)
                || belief_snapshot.changepoint.observation_count < 5
            {
                debug!(
                    raw_cp_prob = %format!("{:.3}", belief_snapshot.changepoint.prob_5),
                    observations = belief_snapshot.changepoint.observation_count,
                    entropy = %format!("{:.2}", belief_snapshot.changepoint.entropy),
                    "BOCD warmup: ignoring changepoint probability (would cause false quote pulling)"
                );
            }
            0.0 // Ignore changepoint during warmup
        };
        let quote_gate_decision = if self.stochastic.stochastic_config.enable_calibrated_quote_gate
        {
            // Use theoretical edge fallback when IR not calibrated
            self.stochastic.quote_gate.decide_with_theoretical_fallback(
                &quote_gate_input,
                &self.stochastic.calibrated_edge,
                &self.stochastic.position_pnl,
                changepoint_prob,
                &mut self.stochastic.theoretical_edge,
            )
        } else {
            self.stochastic.quote_gate.decide(&quote_gate_input)
        };

        // Handle quote gate decision
        // Phase 7: NoQuote → graduated response (3x spread + reduce-only), not cancel all
        match &quote_gate_decision {
            QuoteGateDecision::NoQuote { reason } => {
                info!(
                    reason = %reason,
                    flow_imbalance = %format!("{:.3}", market_params.flow_imbalance),
                    position = %format!("{:.4}", self.position.position()),
                    "Quote gate: NoQuote → graduated 3x spread + reduce-only"
                );
                // Phase 7: Apply 2x widening + reduce-only instead of cancelling
                // Capped at 2.0 — multiplicative stacking beyond 2x is counterproductive
                market_params.spread_widening_mult = (market_params.spread_widening_mult * 2.0).min(2.0);
                risk_reduce_only = true;
                // Fall through to quote generation with widened spreads
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
            QuoteGateDecision::WidenSpreads { multiplier, changepoint_prob } => {
                // Changepoint pending confirmation - apply spread widening
                // Cap at 2.0x — beyond that is counterproductive (makes quotes uncompetitive)
                let capped_mult = multiplier.min(2.0);
                info!(
                    raw_multiplier = %format!("{:.2}", multiplier),
                    capped_multiplier = %format!("{:.2}", capped_mult),
                    changepoint_prob = %format!("{:.3}", changepoint_prob),
                    "Quote gate: widening spreads (changepoint pending, capped at 2.0x)"
                );
                market_params.spread_widening_mult = capped_mult;
                market_params.changepoint_prob = *changepoint_prob;
            }
        }

        // === QUOTA SHADOW SPREAD: Continuous penalty for low API headroom ===
        // Replaces discrete tier cliffs with smooth shadow pricing.
        // At 50% headroom: ~1 bps (mild). At 5% headroom: ~10 bps (aggressive).
        // This naturally widens spreads as quota depletes, reducing order churn.
        {
            let headroom = market_params.rate_limit_headroom_pct;
            let shadow_bps = self.stochastic.quote_gate.continuous_shadow_spread_bps(headroom);
            market_params.quota_shadow_spread_bps = shadow_bps;
            if shadow_bps > 1.0 {
                tracing::info!(
                    headroom_pct = %format!("{:.1}%", headroom * 100.0),
                    shadow_spread_bps = %format!("{:.2}", shadow_bps),
                    "Quota shadow spread applied (continuous pricing)"
                );
            }
        }

        // === CAPACITY BUDGET: Top-down viability check ===
        // Computed once per cycle, flows through the entire pipeline.
        // Uses exact ceiling math to determine if we can place minimum-notional orders.
        let capacity_budget = CapacityBudget::compute(
            market_params.account_value,
            market_params.microprice,
            MIN_ORDER_NOTIONAL,
            self.config.sz_decimals,
            self.effective_max_position,
            self.position.position(),
            kelly_adjusted_liquidity,
        );

        // THE CRITICAL FIX: Wire capital_tier + capacity_budget into market_params.
        // Without this, market_params.capital_tier defaults to Large, causing
        // select_mode step 6 to return Flat instead of Maker for small accounts.
        market_params.capital_tier = capacity_budget.capital_tier;
        market_params.capital_policy = capacity_budget.policy();
        market_params.capacity_budget = Some(capacity_budget.clone());
        // Mirror policy to MarketMaker for components (reconciler) that don't receive MarketParams
        self.capital_policy = capacity_budget.policy();

        // Wire kill switch headroom for dynamic margin utilization.
        // Headroom = 1 - (current_drawdown / max_drawdown_threshold).
        {
            let ks_config = self.safety.kill_switch.config();
            let risk_state = self.build_risk_state();
            let drawdown = if risk_state.peak_pnl < ks_config.min_peak_for_drawdown {
                0.0
            } else {
                risk_state.drawdown()
            };
            market_params.kill_switch_headroom = if ks_config.max_drawdown > 0.0 {
                (1.0 - drawdown / ks_config.max_drawdown).clamp(0.0, 1.0)
            } else {
                1.0
            };
        }

        // Capital-aware warmup bootstrap: small accounts get a warmup floor
        // to prevent the death spiral (no fills → 10% warmup → inflated gamma → no fills).
        // Bayesian priors give sufficient starting estimates for bootstrap_from_book tiers.
        // Fill-based progress: fills/warmup_fill_target (Micro: 5, Small: 10, Large: 50).
        {
            let policy = &market_params.capital_policy;
            // Bootstrap floor: tiers with bootstrap_from_book get a floor from priors
            let bootstrap_floor: f64 = if policy.bootstrap_from_book { 0.40 } else { 0.0 };
            // Fill-based progress scales by tier-specific target (not hardcoded 50)
            let fills = self.tier1.adverse_selection.fills_measured() as f64;
            let fill_progress = if policy.warmup_fill_target > 0 {
                (fills / policy.warmup_fill_target as f64).min(1.0)
            } else {
                1.0
            };
            let warmup_floor = bootstrap_floor.max(fill_progress);
            if warmup_floor > 0.0 {
                market_params.adaptive_warmup_progress =
                    market_params.adaptive_warmup_progress.max(warmup_floor);
            }
        }

        if !capacity_budget.should_quote() {
            if let Viability::NotViable { ref reason, min_capital_needed_usd } = capacity_budget.viability {
                // Throttle log to avoid spam: log on first occurrence then every ~30s
                // (using seconds-based check since there's no cycle counter on MarketMaker)
                let now_sec = chrono::Utc::now().timestamp();
                if now_sec % 30 == 0 {
                    tracing::warn!(
                        reason = %reason,
                        min_capital_usd = %format!("{:.2}", min_capital_needed_usd),
                        account_value = %format!("{:.2}", market_params.account_value),
                        mark_px = %format!("{:.2}", market_params.microprice),
                        "CAPACITY NOT VIABLE: cannot place minimum orders, idling"
                    );
                }
                return Ok(());
            }
        }

        // === STALE DATA CIRCUIT BREAKER ===
        // When data sources go stale, widen spreads proportionally instead of just logging.
        // This converts staleness from a warning into an automatic defensive response.
        {
            let book_age_ms = self.infra.connection_health.time_since_last_data().as_millis() as u64;
            let exchange_limits_age_ms = self.infra.exchange_limits.age_ms();
            let stale_mult = compute_staleness_spread_penalty(book_age_ms, exchange_limits_age_ms);

            if stale_mult > 1.0 {
                market_params.spread_widening_mult *= stale_mult;
                debug!(
                    book_age_ms = book_age_ms,
                    exchange_limits_age_ms = exchange_limits_age_ms,
                    stale_mult = %format!("{stale_mult:.2}"),
                    "Stale data: widening spreads"
                );
            }

            // Critical staleness: pull quotes entirely (book >30s or limits >5min)
            // In paper mode, exchange limits are synthetically initialized and never update
            // via websocket, so skip the limits staleness check to avoid pulling all quotes.
            let limits_stale = exchange_limits_age_ms > 300_000 && self.environment.is_live();
            if book_age_ms > 30_000 || limits_stale {
                warn!(
                    book_age_ms = book_age_ms,
                    exchange_limits_age_ms = exchange_limits_age_ms,
                    "STALE DATA CRITICAL: pulling all quotes"
                );
                return Ok(());
            }
        }

        // Try multi-level ladder quoting first
        // HARMONIZED: Use decision-adjusted liquidity (first-principles derived, decision-scaled)
        let mut ladder = self.strategy.calculate_ladder(
            &quote_config,
            self.position.position(),
            self.effective_max_position, // First-principles limit
            kelly_adjusted_liquidity, // Decision + Kelly-adjusted viable size
            &market_params,
        );

        // Snap ladder prices to grid to prevent sub-tick oscillations from
        // triggering cancel+replace cycles (Phase 2b: Churn Reduction).
        if self.price_grid_config.enabled && self.latest_mid > 0.0 {
            let tick_price = self.latest_mid * market_params.tick_size_bps / 10_000.0;
            // 1-minute sigma in bps: sigma (per-sec fraction) × 10_000 × √60
            let sigma_bps_1m = market_params.sigma_effective * 10_000.0 * 60.0_f64.sqrt();
            let grid = quoting::PriceGrid::for_current_state(
                self.latest_mid,
                tick_price,
                sigma_bps_1m,
                &self.price_grid_config,
                market_params.rate_limit_headroom_pct,
            );
            for level in ladder.bids.iter_mut() {
                level.price = grid.snap_bid(level.price);
            }
            for level in ladder.asks.iter_mut() {
                level.price = grid.snap_ask(level.price);
            }
        }

        // === CANCEL-ON-TOXICITY (Sprint 2.2) ===
        // When pre-fill toxicity exceeds 0.75 on a side, clear that side of the
        // ladder entirely. The reconciler will cancel resting orders. This prevents
        // getting filled at minimum tick spread during toxic flow bursts.
        // 5s cooldown prevents oscillation from noisy toxicity signals.
        const TOXICITY_CANCEL_THRESHOLD: f64 = 0.75;
        const TOXICITY_CANCEL_COOLDOWN: std::time::Duration = std::time::Duration::from_secs(5);

        let now_tox = std::time::Instant::now();
        if market_params.pre_fill_toxicity_bid > TOXICITY_CANCEL_THRESHOLD {
            let should_cancel = self.last_toxicity_cancel_bid
                .is_none_or(|t| now_tox.duration_since(t) >= TOXICITY_CANCEL_COOLDOWN);
            if should_cancel {
                let cleared = ladder.bids.len();
                ladder.bids.clear();
                self.last_toxicity_cancel_bid = Some(now_tox);
                if cleared > 0 {
                    tracing::warn!(
                        toxicity = %format!("{:.3}", market_params.pre_fill_toxicity_bid),
                        cleared_levels = cleared,
                        "TOXICITY CANCEL: bid side cleared (toxicity > {:.0}%)",
                        TOXICITY_CANCEL_THRESHOLD * 100.0
                    );
                }
            }
        }
        if market_params.pre_fill_toxicity_ask > TOXICITY_CANCEL_THRESHOLD {
            let should_cancel = self.last_toxicity_cancel_ask
                .is_none_or(|t| now_tox.duration_since(t) >= TOXICITY_CANCEL_COOLDOWN);
            if should_cancel {
                let cleared = ladder.asks.len();
                ladder.asks.clear();
                self.last_toxicity_cancel_ask = Some(now_tox);
                if cleared > 0 {
                    tracing::warn!(
                        toxicity = %format!("{:.3}", market_params.pre_fill_toxicity_ask),
                        cleared_levels = cleared,
                        "TOXICITY CANCEL: ask side cleared (toxicity > {:.0}%)",
                        TOXICITY_CANCEL_THRESHOLD * 100.0
                    );
                }
            }
        }

        // Post-condition: if capacity said viable but ladder is empty, something is wrong
        if capacity_budget.should_quote() && ladder.is_empty() {
            tracing::error!(
                viability = ?capacity_budget.viability,
                quantum_min = %format!("{:.6}", capacity_budget.quantum.min_viable_size),
                bid_capacity = %format!("{:.6}", capacity_budget.bid_capacity),
                ask_capacity = %format!("{:.6}", capacity_budget.ask_capacity),
                effective_max = %format!("{:.6}", capacity_budget.effective_max_position),
                target_liquidity = %format!("{:.6}", kelly_adjusted_liquidity),
                "CAPACITY INVARIANT VIOLATED: viable budget but empty ladder"
            );
        }

        // Update kill switch with ladder depth for position runaway detection.
        // Uses the larger side so the threshold reflects maximum one-sided exposure.
        let max_one_side = ladder.total_ask_size().max(ladder.total_bid_size());
        self.safety.kill_switch.set_ladder_depth(max_one_side);

        // Capture target level counts for cycle diagnosis
        let diag_target_bids = ladder.bids.len();
        let diag_target_asks = ladder.asks.len();
        #[allow(unused_assignments)]
        let mut diag_reduce_only = false;

        // === EXECUTION MODE: Compute from position zone + toxicity regime ===
        // Replaces QuoteGate side-masking (Phase 4) and hardcoded toxicity filter (Phase 3).
        let execution_mode = {
            use crate::market_maker::execution::{select_mode, ModeSelectionInput};
            // FIX: Evaluate QueueValue at GLFT-optimal depth (where we'll actually quote),
            // not BBO half-spread. With $100 HYPE, BBO spread ≈ 4 bps → half = 2 bps,
            // but QueueValue(2.0, Normal, 0.0) = 2 - 3 - 0 - 1.5 = -2.5 → false.
            // At GLFT depth (5 bps): QueueValue(5.0, Normal, 0.0) = 5 - 3 - 0 - 1.5 = 0.5 → true.
            let evaluation_depth_bps = capacity_budget.min_viable_depth_bps
                .max(feature_snapshot.spread_bps / 2.0);
            let input = ModeSelectionInput {
                position_zone: position_assessment.zone,
                toxicity_regime: current_toxicity_regime,
                bid_has_value: self.queue_value_heuristic.should_quote(
                    evaluation_depth_bps,
                    current_toxicity_regime,
                    feature_snapshot.queue_rank_bid,
                ),
                ask_has_value: self.queue_value_heuristic.should_quote(
                    evaluation_depth_bps,
                    current_toxicity_regime,
                    feature_snapshot.queue_rank_ask,
                ),
                has_alpha: signals.lead_lag_actionable
                    || self.stochastic.signal_integrator.has_cusum_divergence(),
                position: self.position.position(),
                capital_tier: market_params.capital_tier,
                // Use fill-count warmup (not estimator.is_warmed_up which is TRUE
                // from checkpoint vol_filter_obs). With 0 fills, models aren't
                // calibrated yet — warmup protection must remain active.
                is_warmup: self.tier1.adverse_selection.fills_measured() < 10,
            };
            select_mode(&input)
        };
        debug!(
            mode = %execution_mode,
            toxicity = ?current_toxicity_regime,
            zone = ?position_assessment.zone,
            fills_measured = self.tier1.adverse_selection.fills_measured(),
            "Execution mode computed"
        );

        if !ladder.bids.is_empty() || !ladder.asks.is_empty() {
            // Multi-level ladder mode — wrap in ViableQuoteLadder for exchange-minimum protection.
            // All size reductions through reduce_sizes() guarantee surviving quotes meet
            // min_notional, or convert defense intent to spread widening.
            let quantum = crate::market_maker::config::SizeQuantum::compute(
                MIN_ORDER_NOTIONAL,
                quote_config.mid_price,
                self.config.sz_decimals,
            );
            let exchange_rules = quoting::ExchangeRules::new(
                self.config.decimals,
                self.config.sz_decimals,
                MIN_ORDER_NOTIONAL,
            );
            let mut viable = quoting::ViableQuoteLadder::from_ladder(
                &ladder,
                quantum,
                quote_config.mid_price,
                Some(exchange_rules),
            );

            // === EXECUTION MODE: structural side selection ===
            // Fix 1: Trust GLFT's continuous γ·q·σ²·(T-t) inventory penalty.
            // Only Kill zone (HJB constraint boundary) clears sides.
            // All other zones: GLFT handles asymmetry through reservation price skew.
            // Red/Yellow → elevated γ (regime_gamma_multiplier) widens proportionally.
            // Toxic → routes through σ_conditional (Hawkes) and AS floor (Fix 4).
            {
                use crate::market_maker::execution::ExecutionMode;
                match execution_mode {
                    ExecutionMode::Flat => {
                        viable.bids.clear();
                        viable.asks.clear();
                    }
                    ExecutionMode::Maker { .. } => {
                        // GLFT γ·q skew handles all asymmetry continuously.
                        // No side clearing — both sides always quoted.
                    }
                    ExecutionMode::InventoryReduce { .. } => {
                        // With Fix 1, select_mode() only returns Flat or Maker{both}.
                        // InventoryReduce is kept for backward compat but shouldn't trigger.
                        // If it does, treat as Maker{both} — GLFT handles the skew.
                    }
                }
            }

            // === TOXICITY DEFENSE ===
            // Fix 1: Toxicity no longer clears sides. Instead it routes through:
            // - σ_conditional (Fix 4): Hawkes intensity → √(λ/λ₀) → uniform widening
            // - AS floor (Fix 2): dynamic floor from markout history → minimum spread
            // - Pre-fill AS classifier: per-side depth multipliers in ladder_strat.rs
            // These three principled channels replace the binary Toxic→clear heuristic.

            // === QUEUE VALUE: filter individual levels with negative expected edge ===
            // After mode selection and toxicity filtering, remove levels where quoting
            // has negative expected value (tight levels during Toxic periods).
            // Uses viable.retain() which protects the last level on each side.
            {
                let mid = quote_config.mid_price;
                viable.bids.retain(|q| {
                    let depth_bps = ((mid - q.price) / mid * 10_000.0).max(0.0);
                    self.queue_value_heuristic.should_quote(
                        depth_bps,
                        current_toxicity_regime,
                        feature_snapshot.queue_rank_bid,
                    )
                });
                viable.asks.retain(|q| {
                    let depth_bps = ((q.price - mid) / mid * 10_000.0).max(0.0);
                    self.queue_value_heuristic.should_quote(
                        depth_bps,
                        current_toxicity_regime,
                        feature_snapshot.queue_rank_ask,
                    )
                });
            }

            // === RISK LEVEL EMERGENCY FILTER ===
            // When risk_level is Emergency, preserve position-reducing quotes so
            // the system can unwind inventory.  The increasing side is cancelled;
            // the reducing side is kept but at 50% size.  Near-zero positions
            // keep both sides at 30% size.
            // Size reductions use viable.reduce_sizes() for exchange-minimum protection.
            if risk_level == crate::market_maker::control::RiskLevel::Emergency {
                let pos = self.position.position();
                let max_pos = self.effective_max_position.max(0.01);
                let position_ratio = pos / max_pos;
                let vq = *viable.quantum();

                if position_ratio > 0.01 {
                    // Long: cancel bids (would increase), keep asks at 50% (reduce)
                    if !viable.bids.is_empty() {
                        info!(
                            position = %format!("{:.4}", pos),
                            cleared_bids = viable.bids.len(),
                            "Emergency: clearing bids (long position, keeping asks to reduce)"
                        );
                        viable.bids.clear();
                    }
                    viable.asks.reduce_sizes(0.5, &vq);
                } else if position_ratio < -0.01 {
                    // Short: cancel asks (would increase), keep bids at 50% (reduce)
                    if !viable.asks.is_empty() {
                        info!(
                            position = %format!("{:.4}", pos),
                            cleared_asks = viable.asks.len(),
                            "Emergency: clearing asks (short position, keeping bids to reduce)"
                        );
                        viable.asks.clear();
                    }
                    viable.bids.reduce_sizes(0.5, &vq);
                } else {
                    // Near zero: keep both sides at 30% size
                    viable.bids.reduce_sizes(0.3, &vq);
                    viable.asks.reduce_sizes(0.3, &vq);
                }
            }

            // === FILL CASCADE ===
            // Fix 1: Cascades route through σ_conditional (Fix 4), not side clearing.
            // The FillCascadeTracker still tracks same-side fill runs for diagnostics,
            // but its spread_multiplier and is_suppressed outputs are no longer used
            // to clear or widen individual sides. Instead, fill bursts elevate Hawkes
            // intensity → σ_cascade_mult → uniform spread widening through GLFT.

            // === CONTINUOUS INVENTORY PRESSURE ===
            // Smooth function of inventory fraction and momentum, no hysteresis state.
            // Both conditions must be present (product + sqrt). Pressure [0,1] scales
            // increasing-side sizes by (1 - pressure), preventing death spirals.
            // Uses viable.reduce_sizes() for exchange-minimum protection.
            {
                let pos = self.position.position();
                let inventory_frac = if self.effective_max_position > 1e-10 {
                    pos.abs() / self.effective_max_position
                } else {
                    0.0
                };
                let momentum_abs = trend_signal.long_momentum_bps.abs();
                // Estimate current quoted half-spread for proportional thresholds
                let quoted_half_spread_bps =
                    (market_params.adaptive_spread_floor * 10_000.0).max(3.0);

                // Compute continuous inventory pressure (0.0-1.0), no hysteresis state
                let pressure = compute_continuous_inventory_pressure(
                    drift_adjusted_skew.is_opposed,
                    inventory_frac,
                    momentum_abs,
                    quoted_half_spread_bps,
                );

                if pressure > 0.01 {
                    let size_mult = 1.0 - pressure;
                    let vq = *viable.quantum();
                    let label = if pos > 0.0 { "BID" } else { "ASK" };

                    if pos > 0.0 && !viable.bids.is_empty() {
                        viable.bids.reduce_sizes(size_mult, &vq);
                        info!(
                            side = label,
                            size_mult = %format!("{:.2}", size_mult),
                            pressure = %format!("{:.2}", pressure),
                            inventory_pct = %format!("{:.0}", inventory_frac * 100.0),
                            momentum_bps = %format!("{:.1}", momentum_abs),
                            "Inventory pressure: size_mult={:.2}", size_mult
                        );
                    } else if pos < 0.0 && !viable.asks.is_empty() {
                        viable.asks.reduce_sizes(size_mult, &vq);
                        info!(
                            side = label,
                            size_mult = %format!("{:.2}", size_mult),
                            pressure = %format!("{:.2}", pressure),
                            inventory_pct = %format!("{:.0}", inventory_frac * 100.0),
                            momentum_bps = %format!("{:.1}", momentum_abs),
                            "Inventory pressure: size_mult={:.2}", size_mult
                        );
                    }
                }
            }

            // Reduce-only mode: when over max position, position value, OR margin utilization
            // Phase 3: Use exchange-aware reduce-only that checks exchange limits and signals escalation
            // CAPITAL-EFFICIENT: Use margin utilization as primary trigger (80% threshold)
            // Note: reduce-only only calls clear() on the vectors — safe through quotes_mut().
            let margin_state = self.infra.margin_sizer.state();
            // Get unrealized P&L for underwater position protection
            let unrealized_pnl = self.tier2.pnl_tracker.unrealized_pnl(self.latest_mid);
            // When user explicitly specified max_position (CLI/TOML), enforce as hard ceiling.
            // Otherwise, let dynamic margin-based effective_max_position be the sole limit.
            // Phase 7: risk_reduce_only forces reduce-only by capping max to current position.
            let reduce_only_max = if risk_reduce_only {
                // Force reduce-only: cap to current position so any increase is filtered
                self.position.position().abs()
            } else if self.config.max_position_user_specified {
                self.effective_max_position.min(self.config.max_position)
            } else {
                self.effective_max_position
            };
            let reduce_only_config = quoting::ReduceOnlyConfig {
                position: self.position.position(),
                max_position: reduce_only_max,
                mid_price: self.latest_mid,
                max_position_value: self.safety.kill_switch.max_position_value(),
                asset: self.config.asset.to_string(),
                margin_used: margin_state.margin_used,
                account_value: margin_state.account_value,
                // Dynamic reduce-only based on liquidation proximity
                liquidation_price: margin_state.liquidation_price,
                liquidation_buffer_ratio: margin_state.liquidation_buffer_ratio(),
                liquidation_trigger_threshold: quoting::DEFAULT_LIQUIDATION_TRIGGER_THRESHOLD,
                // Underwater position protection - prevents forcing sales at a loss
                unrealized_pnl,
            };
            let reduce_only_result = quoting::QuoteFilter::apply_reduce_only_with_exchange_limits(
                viable.bids.quotes_mut(),
                viable.asks.quotes_mut(),
                &reduce_only_config,
                &self.infra.exchange_limits,
            );
            diag_reduce_only = reduce_only_result.was_filtered;

            // If escalation is needed, cancel all position-increasing quotes
            // This prevents the position from growing further when reduce-only can't place reducing orders
            if reduce_only_result.needs_escalation {
                warn!(
                    position = self.position.position(),
                    filtered_bids = reduce_only_result.filtered_bids,
                    filtered_asks = reduce_only_result.filtered_asks,
                    "Reduce-only ESCALATION: cancelling all position-increasing quotes"
                );
                // Clear the side that would increase position
                if reduce_only_result.filtered_bids {
                    viable.bids.clear();
                }
                if reduce_only_result.filtered_asks {
                    viable.asks.clear();
                }
            }

            // === CLOSE BIAS: tighten closing side spread in reduce-only mode ===
            if reduce_only_result.was_filtered {
                let close_urgency = quoting::ReduceOnlyResult::compute_urgency(
                    self.position.position().abs(),
                    self.effective_max_position,
                    unrealized_pnl,
                );
                if close_urgency > 0.0 {
                    let position = self.position.position();
                    let mid = self.latest_mid;
                    // Apply close bias to each surviving quote on the closing side
                    if position > 0.0 {
                        // Long: tighten asks (sell side)
                        for quote in viable.asks.iter_mut() {
                            let (_, new_ask) = quoting::apply_close_bias(
                                mid - 1.0, quote.price, mid, position, close_urgency,
                            );
                            quote.price = new_ask;
                        }
                    } else if position < 0.0 {
                        // Short: tighten bids (buy side)
                        for quote in viable.bids.iter_mut() {
                            let (new_bid, _) = quoting::apply_close_bias(
                                quote.price, mid + 1.0, mid, position, close_urgency,
                            );
                            quote.price = new_bid;
                        }
                    }
                    debug!(
                        close_urgency = %format!("{close_urgency:.3}"),
                        position = %format!("{position:.6}"),
                        "Applied close bias to tighten closing side"
                    );
                }
            }

            info!(
                bid_levels = viable.bids.len(),
                ask_levels = viable.asks.len(),
                best_bid = ?viable.bids.first().map(|q| (q.price, q.size)),
                best_ask = ?viable.asks.first().map(|q| (q.price, q.size)),
                defense_bid_mult = %format!("{:.2}", viable.bids.defense_spread_mult()),
                defense_ask_mult = %format!("{:.2}", viable.asks.defense_spread_mult()),
                "Calculated ladder quotes"
            );

            // === Phase 5: REGISTER PENDING QUOTES for outcome tracking ===
            // Register best bid/ask as pending quotes for fill/expiry resolution.
            // Only track best level (most likely to fill) to keep overhead minimal.
            // Done BEFORE finalize() so tracking uses pre-widening prices.
            {
                use crate::market_maker::learning::quote_outcome::{CompactMarketState, PendingQuote};
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                let mid = self.latest_mid;
                let compact_state = CompactMarketState {
                    microprice: market_params.microprice,
                    sigma: market_params.sigma,
                    book_imbalance: market_params.book_imbalance,
                    flow_imbalance: market_params.flow_imbalance,
                    toxicity_score: market_params.toxicity_score,
                    kappa: market_params.kappa,
                };
                if let Some(best_bid) = viable.bids.first() {
                    let half_spread_bps = ((mid - best_bid.price) / mid) * 10000.0;
                    self.quote_outcome_tracker.register_quote(PendingQuote {
                        timestamp_ms: now_ms,
                        half_spread_bps,
                        is_bid: true,
                        state: compact_state.clone(),
                    });
                }
                if let Some(best_ask) = viable.asks.first() {
                    let half_spread_bps = ((best_ask.price - mid) / mid) * 10000.0;
                    self.quote_outcome_tracker.register_quote(PendingQuote {
                        timestamp_ms: now_ms,
                        half_spread_bps,
                        is_bid: false,
                        state: compact_state,
                    });
                }
            }

            // === RECORD QUOTE DECISION FOR DASHBOARD ===
            // Capture why this specific spread was chosen (pre-widening)
            if let (Some(best_bid), Some(best_ask)) = (viable.bids.first(), viable.asks.first()) {
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

            // === FINALIZE: apply defense spread widening, validate, and extract quotes ===
            // All quotes guaranteed to meet min_notional AND exchange precision after finalize.
            let (bid_validated, ask_validated, validation_report) = viable.finalize();

            // [VALIDATE] logging — tracks price re-rounding and rejection stats per cycle
            if validation_report.fixed > 0 || validation_report.rejected > 0 {
                info!(
                    proposed = validation_report.proposed,
                    valid = validation_report.valid,
                    fixed = validation_report.fixed,
                    rejected = validation_report.rejected,
                    "[VALIDATE] {validation_report}"
                );
            } else {
                trace!(
                    proposed = validation_report.proposed,
                    valid = validation_report.valid,
                    "[VALIDATE] {validation_report}"
                );
            }

            // Convert validated quotes back to Quote for reconciler compatibility
            let bid_quotes: Vec<Quote> = bid_validated
                .iter()
                .map(|vq| Quote::new(vq.price(), vq.size()))
                .collect();
            let ask_quotes: Vec<Quote> = ask_validated
                .iter()
                .map(|vq| Quote::new(vq.price(), vq.size()))
                .collect();

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

            // Unified reconciliation: economic scoring + budget-constrained allocation.
            // Replaces batch/smart/legacy routing with single principled pipeline.
            // Every API call is an economic decision: update_value = EV_new - EV_keep - api_cost.
            // Budget adapts to headroom: abundant → aggressive updates, scarce → only highest-value.
            self.reconcile_unified(bid_quotes, ask_quotes).await?;
        } else {
            // Fallback to single-quote mode for non-ladder strategies
            // HARMONIZED: Use decision-adjusted values (first-principles derived, decision-scaled)
            let (mut bid, mut ask) = self.strategy.calculate_quotes(
                &quote_config,
                self.position.position(),
                self.effective_max_position, // First-principles limit
                kelly_adjusted_liquidity, // Decision + Kelly-adjusted viable size
                &market_params,
            );

            // === EXECUTION MODE: structural side selection (single-quote mode) ===
            {
                use crate::market_maker::execution::ExecutionMode;
                match execution_mode {
                    ExecutionMode::Flat => {
                        bid = None;
                        ask = None;
                    }
                    ExecutionMode::Maker { bid: bid_ok, ask: ask_ok } => {
                        if !bid_ok { bid = None; }
                        if !ask_ok { ask = None; }
                    }
                    ExecutionMode::InventoryReduce { urgency } => {
                        let pos = self.position.position();
                        if pos > 1e-9 {
                            bid = None;
                            if let Some(q) = ask.as_mut() {
                                q.size *= 0.5 + 0.5 * urgency;
                            }
                        } else if pos < -1e-9 {
                            ask = None;
                            if let Some(q) = bid.as_mut() {
                                q.size *= 0.5 + 0.5 * urgency;
                            }
                        }
                    }
                }
            }

            // === RISK LEVEL EMERGENCY FILTER (single-quote mode) ===
            // Same logic as ladder mode: cancel increasing side, halve reducing side.
            if risk_level == crate::market_maker::control::RiskLevel::Emergency {
                let pos = self.position.position();
                let max_pos = self.effective_max_position.max(0.01);
                let position_ratio = pos / max_pos;

                if position_ratio > 0.01 {
                    // Long: cancel bid, halve ask size
                    if bid.is_some() {
                        debug!("Emergency: clearing BID quote (long position, single-quote mode)");
                        bid = None;
                    }
                    if let Some(ref mut a) = ask {
                        a.size *= 0.5;
                    }
                } else if position_ratio < -0.01 {
                    // Short: cancel ask, halve bid size
                    if ask.is_some() {
                        debug!("Emergency: clearing ASK quote (short position, single-quote mode)");
                        ask = None;
                    }
                    if let Some(ref mut b) = bid {
                        b.size *= 0.5;
                    }
                } else {
                    // Near zero: keep both at 30% size
                    if let Some(ref mut b) = bid {
                        b.size *= 0.3;
                    }
                    if let Some(ref mut a) = ask {
                        a.size *= 0.3;
                    }
                }
            }

            // Reduce-only mode: when over max position, position value, OR margin utilization
            // CAPITAL-EFFICIENT: Use margin utilization as primary trigger (80% threshold)
            let margin_state = self.infra.margin_sizer.state();
            // Get unrealized P&L for underwater position protection
            let unrealized_pnl = self.tier2.pnl_tracker.unrealized_pnl(self.latest_mid);
            // When user explicitly specified max_position (CLI/TOML), enforce as hard ceiling.
            // Otherwise, let dynamic margin-based effective_max_position be the sole limit.
            let reduce_only_max_single = if risk_reduce_only {
                self.position.position().abs() // Force reduce-only from graduated risk
            } else if self.config.max_position_user_specified {
                self.effective_max_position.min(self.config.max_position)
            } else {
                self.effective_max_position
            };
            let reduce_only_config = quoting::ReduceOnlyConfig {
                position: self.position.position(),
                max_position: reduce_only_max_single,
                mid_price: self.latest_mid,
                max_position_value: self.safety.kill_switch.max_position_value(),
                asset: self.config.asset.to_string(),
                margin_used: margin_state.margin_used,
                account_value: margin_state.account_value,
                // Dynamic reduce-only based on liquidation proximity
                liquidation_price: margin_state.liquidation_price,
                liquidation_buffer_ratio: margin_state.liquidation_buffer_ratio(),
                liquidation_trigger_threshold: quoting::DEFAULT_LIQUIDATION_TRIGGER_THRESHOLD,
                // Underwater position protection - prevents forcing sales at a loss
                unrealized_pnl,
            };
            let reduce_only_result =
                quoting::QuoteFilter::apply_reduce_only_single(&mut bid, &mut ask, &reduce_only_config);
            diag_reduce_only = reduce_only_result.was_filtered;

            // === CLOSE BIAS: tighten closing side spread in reduce-only mode ===
            if reduce_only_result.was_filtered {
                let close_urgency = quoting::ReduceOnlyResult::compute_urgency(
                    self.position.position().abs(),
                    self.effective_max_position,
                    unrealized_pnl,
                );
                if close_urgency > 0.0 {
                    let position = self.position.position();
                    let mid = self.latest_mid;
                    if let (Some(b), Some(a)) = (&mut bid, &mut ask) {
                        let (new_bid, new_ask) = quoting::apply_close_bias(
                            b.price, a.price, mid, position, close_urgency,
                        );
                        b.price = new_bid;
                        a.price = new_ask;
                    } else if position > 0.0 {
                        // Long: only asks survive, tighten ask
                        if let Some(a) = &mut ask {
                            let (_, new_ask) = quoting::apply_close_bias(
                                mid - 1.0, a.price, mid, position, close_urgency,
                            );
                            a.price = new_ask;
                        }
                    } else if position < 0.0 {
                        // Short: only bids survive, tighten bid
                        if let Some(b) = &mut bid {
                            let (new_bid, _) = quoting::apply_close_bias(
                                b.price, mid + 1.0, mid, position, close_urgency,
                            );
                            b.price = new_bid;
                        }
                    }
                }
            }

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

        // === Cycle Diagnosis: one-line summary ===
        {
            use super::diagnosis::{BlockingReason, CycleDiagnosis};

            let is_warmup = !self.estimator.is_warmed_up();
            let warmup_pct = {
                let (current, target) = self.estimator.warmup_progress_simple();
                if target > 0 {
                    (current as f64 / target as f64 * 100.0).min(100.0)
                } else {
                    100.0
                }
            };
            let (surviving_bids, surviving_asks) = self.orders.order_counts();

            // Use feature snapshot spread (computed from L2 book)
            let spread_bps = feature_snapshot.spread_bps;

            let blocking_reason = if self.safety.kill_switch.is_triggered() {
                Some(BlockingReason::KillSwitchActive {
                    reason: "kill_switch_active".to_string(),
                })
            } else if diag_target_bids == 0 && diag_target_asks == 0 {
                Some(BlockingReason::EmptyLadder)
            } else if surviving_bids == 0 && surviving_asks == 0 {
                Some(BlockingReason::FilteredToEmpty {
                    filter: "pipeline",
                    levels_before: diag_target_bids + diag_target_asks,
                })
            } else {
                None
            };

            let diagnosis = CycleDiagnosis {
                is_warmup,
                warmup_pct,
                spread_bps,
                target_bid_levels: diag_target_bids,
                target_ask_levels: diag_target_asks,
                surviving_bid_levels: surviving_bids,
                surviving_ask_levels: surviving_asks,
                execution_mode: format!("{execution_mode}"),
                reduce_only: diag_reduce_only,
                blocking_reason,
            };
            diagnosis.emit();
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

        // Cache market params for signal diagnostics (fill handler uses this)
        self.cached_market_params = Some(market_params);

        Ok(())
    }

    /// Check if all existing orders are within the drift threshold of target quotes.
    /// Previously used by batch reconcile; kept for potential future use.
    #[allow(dead_code)]
    fn check_price_drift_within_threshold(
        &self,
        bid_quotes: &[Quote],
        ask_quotes: &[Quote],
        threshold_bps: f64,
    ) -> bool {
        // If no existing orders, always need to place
        let existing_bids: Vec<_> = self.orders.get_all_by_side(Side::Buy);
        let existing_asks: Vec<_> = self.orders.get_all_by_side(Side::Sell);

        if existing_bids.is_empty() && existing_asks.is_empty() {
            return false; // No orders → need to place
        }

        // Check bid drift
        if existing_bids.len() != bid_quotes.len() {
            return false; // Different count → need reconcile
        }
        for (existing, target) in existing_bids.iter().zip(bid_quotes.iter()) {
            let mid = (existing.price + target.price) / 2.0;
            if mid > 0.0 {
                let drift_bps = ((existing.price - target.price) / mid).abs() * 10_000.0;
                if drift_bps > threshold_bps {
                    return false;
                }
            }
        }

        // Check ask drift
        if existing_asks.len() != ask_quotes.len() {
            return false;
        }
        for (existing, target) in existing_asks.iter().zip(ask_quotes.iter()) {
            let mid = (existing.price + target.price) / 2.0;
            if mid > 0.0 {
                let drift_bps = ((existing.price - target.price) / mid).abs() * 10_000.0;
                if drift_bps > threshold_bps {
                    return false;
                }
            }
        }

        true // All within threshold
    }
}

/// Compute continuous inventory pressure as a smooth function of inventory and momentum.
///
/// Returns 0.0 (no pressure) to 1.0 (full clear). Both inventory risk AND adverse
/// momentum must be present — the product ensures neither alone triggers action.
/// The sqrt smooths the pressure surface so small increases in either input produce
/// proportional (not explosive) responses.
///
/// Replaces the discrete trigger + hysteresis system that caused death spirals:
/// at medium urgency (0.3), only 1 bid survived → only asks filled → position
/// increased → urgency increased → bids fully cleared.
fn compute_continuous_inventory_pressure(
    is_opposed: bool,
    inventory_frac: f64,
    momentum_abs_bps: f64,
    quoted_half_spread_bps: f64,
) -> f64 {
    if !is_opposed {
        return 0.0;
    }

    // Inventory component: smooth ramp starting at 30%, full at 100%
    let inv = ((inventory_frac - 0.3) / 0.7).clamp(0.0, 1.0);

    // Momentum component: normalized by our spread (momentum relative to our exposure)
    let mom = (momentum_abs_bps / quoted_half_spread_bps.max(1.0))
        .clamp(0.0, 3.0)
        / 3.0;

    // Product: both conditions must be present. sqrt smooths the surface.
    (inv * mom).sqrt().clamp(0.0, 1.0)
}

/// Compute tick size in basis points from Hyperliquid's 5-significant-figure pricing.
///
/// Hyperliquid perps use 5 significant figures for all prices.
/// The minimum price increment (tick) depends on the price magnitude:
///   - BTC @ $97,000 → tick = $1.00   → 0.10 bps
///   - ETH @  $3,000 → tick = $0.10   → 0.33 bps
///   - HYPE @    $31 → tick = $0.001  → 0.32 bps
///   - SOL @   $200 → tick = $0.01   → 0.50 bps
///
/// Compute spread widening multiplier for stale data.
///
/// When book or exchange limit data goes stale, spreads widen automatically
/// to compensate for increased uncertainty. This converts staleness warnings
/// into an automatic defensive response.
///
/// Returns 1.0 (no penalty) when data is fresh, up to 3.0 when very stale.
fn compute_staleness_spread_penalty(book_age_ms: u64, exchange_limits_age_ms: u64) -> f64 {
    let mut mult: f64 = 1.0;

    // Book data stale: widens at 5s, more at 15s
    if book_age_ms > 15_000 {
        mult *= 2.0;
    } else if book_age_ms > 5_000 {
        mult *= 1.5;
    }

    // Exchange limits stale: widens at 30s, more at 2min
    if exchange_limits_age_ms > 120_000 {
        mult *= 2.0;
    } else if exchange_limits_age_ms > 30_000 {
        mult *= 1.5;
    }

    mult.min(3.0) // Cap at 3x to avoid extreme widening
}

/// This replaces the hardcoded `tick_size_bps: 10.0` which caused spreads
/// to be 32x wider than market (20 bps vs 0.6 bps BBO).
fn compute_tick_size_bps(mid_price: f64) -> f64 {
    if mid_price <= 0.0 {
        return 1.0; // Safe default before first mid price
    }
    const SIGNIFICANT_FIGURES: i32 = 5;
    let integer_digits = (mid_price.log10().floor() as i32 + 1).max(1);
    let decimal_places = (SIGNIFICANT_FIGURES - integer_digits).max(0);
    let tick_size = 10_f64.powi(-decimal_places);
    // Convert to bps, floor at 0.01 bps to avoid zero
    (tick_size / mid_price * 10_000.0).max(0.01)
}

/// Apply continuous inventory pressure to the increasing-side quotes.
///
/// Smoothly reduces sizes by `(1 - pressure)`. At pressure=0 quotes are untouched,
/// at pressure=1 all sizes go to zero and are removed. This replaces the discrete
/// tier system (halve / keep-one / clear) that caused death spirals.
#[cfg(test)]
fn apply_inventory_pressure(pressure: f64, increasing_quotes: &mut Vec<crate::market_maker::config::Quote>) {
    if pressure < 0.01 {
        return;
    }
    let size_mult = 1.0 - pressure;
    for q in increasing_quotes.iter_mut() {
        q.size *= size_mult;
    }
    increasing_quotes.retain(|q| q.size > 0.0);
}

/// Apply risk-level emergency filter to bid and ask quote vectors.
///
/// Position-aware: cancels the side that would increase exposure and reduces
/// the closing side to 50% size.  Near-zero positions keep both sides at 30%.
///
/// This is the testable standalone equivalent of the inline emergency filter
/// in `generate_and_emit_quotes()`.
#[cfg(test)]
fn apply_risk_emergency_filter(
    bid_quotes: &mut Vec<crate::market_maker::config::Quote>,
    ask_quotes: &mut Vec<crate::market_maker::config::Quote>,
    position: f64,
    max_position: f64,
) {
    let max_pos = max_position.max(0.01);
    let position_ratio = position / max_pos;

    if position_ratio > 0.01 {
        // Long: cancel bids, keep asks at 50%
        bid_quotes.clear();
        for q in ask_quotes.iter_mut() {
            q.size *= 0.5;
        }
    } else if position_ratio < -0.01 {
        // Short: cancel asks, keep bids at 50%
        ask_quotes.clear();
        for q in bid_quotes.iter_mut() {
            q.size *= 0.5;
        }
    } else {
        // Near zero: both sides at 30%
        for q in bid_quotes.iter_mut() {
            q.size *= 0.3;
        }
        for q in ask_quotes.iter_mut() {
            q.size *= 0.3;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::config::Quote;

    // ---------------------------------------------------------------
    // Test 1: Continuous pressure — monotonic in inventory_frac
    // ---------------------------------------------------------------
    #[test]
    fn test_pressure_monotonic_in_inventory() {
        let spread = 5.0;
        let momentum = 10.0;
        let mut prev = 0.0_f64;
        // Pressure should be non-decreasing as inventory increases
        for i in 0..=100 {
            let inv = i as f64 / 100.0;
            let p = compute_continuous_inventory_pressure(true, inv, momentum, spread);
            assert!(
                p >= prev - 1e-10,
                "Pressure should be monotonic: at inv={inv}, p={p} < prev={prev}"
            );
            prev = p;
        }
        // Should be near-zero at low inventory, significant at high
        let p_low = compute_continuous_inventory_pressure(true, 0.1, momentum, spread);
        let p_high = compute_continuous_inventory_pressure(true, 0.9, momentum, spread);
        assert!(p_low < 0.1, "Low inventory should yield low pressure, got {p_low}");
        assert!(p_high > 0.3, "High inventory should yield high pressure, got {p_high}");
    }

    // ---------------------------------------------------------------
    // Test 2: Continuous pressure — monotonic in momentum
    // ---------------------------------------------------------------
    #[test]
    fn test_pressure_monotonic_in_momentum() {
        let spread = 5.0;
        let inv = 0.6;
        let mut prev = 0.0_f64;
        for m in 0..=50 {
            let momentum = m as f64;
            let p = compute_continuous_inventory_pressure(true, inv, momentum, spread);
            assert!(
                p >= prev - 1e-10,
                "Pressure should be monotonic: at mom={momentum}, p={p} < prev={prev}"
            );
            prev = p;
        }
    }

    // ---------------------------------------------------------------
    // Test 3: No pressure when not opposed
    // ---------------------------------------------------------------
    #[test]
    fn test_no_pressure_when_not_opposed() {
        // Even extreme inventory + momentum should produce zero pressure
        let p = compute_continuous_inventory_pressure(false, 0.99, 100.0, 5.0);
        assert!(
            (p - 0.0).abs() < 1e-10,
            "Pressure should be 0.0 when not opposed, got {p}"
        );
    }

    // ---------------------------------------------------------------
    // Test 4: Low risk produces near-zero pressure
    // ---------------------------------------------------------------
    #[test]
    fn test_low_risk_near_zero_pressure() {
        // inventory below 30% threshold: inv component = 0, so product = 0
        let p = compute_continuous_inventory_pressure(true, 0.2, 5.0, 5.0);
        assert!(
            p < 0.01,
            "Below 30% inventory should yield near-zero pressure, got {p}"
        );

        // Zero momentum: mom component = 0, so product = 0
        let p2 = compute_continuous_inventory_pressure(true, 0.8, 0.0, 5.0);
        assert!(
            p2 < 0.01,
            "Zero momentum should yield near-zero pressure, got {p2}"
        );
    }

    // ---------------------------------------------------------------
    // Test 5: High risk produces near-one pressure
    // ---------------------------------------------------------------
    #[test]
    fn test_high_risk_near_one_pressure() {
        // Full inventory (1.0) + momentum = 3x spread → both components maxed
        // inv = (1.0 - 0.3) / 0.7 = 1.0
        // mom = (15.0 / 5.0).clamp(0, 3) / 3.0 = 1.0
        // pressure = sqrt(1.0 * 1.0) = 1.0
        let p = compute_continuous_inventory_pressure(true, 1.0, 15.0, 5.0);
        assert!(
            (p - 1.0).abs() < 1e-10,
            "Full inventory + 3x spread momentum should yield 1.0, got {p}"
        );
    }

    // ---------------------------------------------------------------
    // Test 6: Smooth size reduction via apply_inventory_pressure
    // ---------------------------------------------------------------
    #[test]
    fn test_apply_inventory_pressure_smooth() {
        // Moderate pressure should reduce sizes proportionally
        let mut quotes = vec![
            Quote::new(100.0, 1.0),
            Quote::new(99.0, 2.0),
            Quote::new(98.0, 3.0),
        ];
        apply_inventory_pressure(0.4, &mut quotes);
        assert_eq!(quotes.len(), 3, "All levels should remain at moderate pressure");
        assert!((quotes[0].size - 0.6).abs() < 1e-10, "Size should be * 0.6");
        assert!((quotes[1].size - 1.2).abs() < 1e-10, "Size should be * 0.6");
        assert!((quotes[2].size - 1.8).abs() < 1e-10, "Size should be * 0.6");
    }

    // ---------------------------------------------------------------
    // Test 7: Full pressure clears all quotes
    // ---------------------------------------------------------------
    #[test]
    fn test_apply_inventory_pressure_full_clears() {
        let mut quotes = vec![
            Quote::new(100.0, 1.0),
            Quote::new(99.0, 2.0),
        ];
        apply_inventory_pressure(1.0, &mut quotes);
        assert!(quotes.is_empty(), "Pressure=1.0 should clear all quotes");
    }

    // ---------------------------------------------------------------
    // Test 8: Negligible pressure leaves quotes untouched
    // ---------------------------------------------------------------
    #[test]
    fn test_apply_inventory_pressure_negligible() {
        let mut quotes = vec![Quote::new(100.0, 1.0), Quote::new(99.0, 2.0)];
        apply_inventory_pressure(0.005, &mut quotes);
        assert_eq!(quotes.len(), 2, "Below 0.01 threshold should not modify");
        assert!((quotes[0].size - 1.0).abs() < 1e-10, "Size unchanged");
        assert!((quotes[1].size - 2.0).abs() < 1e-10, "Size unchanged");
    }

    // ---------------------------------------------------------------
    // Test 9: Preserves closing side — only increasing side affected
    // ---------------------------------------------------------------
    #[test]
    fn test_preserves_closing_side() {
        let pressure = compute_continuous_inventory_pressure(true, 0.9, 15.0, 5.0);
        assert!(pressure > 0.5, "Should have significant pressure");

        let mut bids = vec![Quote::new(100.0, 1.0), Quote::new(99.0, 2.0)];
        let asks = vec![Quote::new(101.0, 1.0), Quote::new(102.0, 2.0)];

        // For long position: increasing side = bids
        apply_inventory_pressure(pressure, &mut bids);
        // Closing side (asks) is never passed to apply_inventory_pressure

        // Bids should be reduced (may be empty at very high pressure)
        assert!(
            bids.iter().all(|q| q.size < 1.0),
            "Bid sizes should be reduced"
        );
        assert_eq!(asks.len(), 2, "Asks (closing) should be untouched");
        assert!((asks[0].size - 1.0).abs() < 1e-10);
        assert!((asks[1].size - 2.0).abs() < 1e-10);
    }

    // ---------------------------------------------------------------
    // Test 10: No state — same inputs always produce same output
    // ---------------------------------------------------------------
    #[test]
    fn test_pressure_is_stateless() {
        // Unlike the old hysteresis system, identical inputs always yield identical output
        let p1 = compute_continuous_inventory_pressure(true, 0.6, 10.0, 5.0);
        let p2 = compute_continuous_inventory_pressure(true, 0.6, 10.0, 5.0);
        assert!(
            (p1 - p2).abs() < 1e-15,
            "Stateless function must return identical values: {p1} vs {p2}"
        );
    }

    // ---------------------------------------------------------------
    // Risk-level emergency filter tests
    // ---------------------------------------------------------------

    #[test]
    fn test_emergency_long_position_clears_bids_keeps_asks() {
        let mut bids = vec![Quote::new(99.0, 1.0), Quote::new(98.0, 2.0)];
        let mut asks = vec![Quote::new(101.0, 1.0), Quote::new(102.0, 2.0)];
        // Long position: position=5.0, max=10.0
        apply_risk_emergency_filter(&mut bids, &mut asks, 5.0, 10.0);

        assert!(bids.is_empty(), "Bids (increasing) should be cleared when long");
        assert_eq!(asks.len(), 2, "Asks (reducing) should be preserved");
        assert!((asks[0].size - 0.5).abs() < 1e-10, "Ask size should be halved");
        assert!((asks[1].size - 1.0).abs() < 1e-10, "Ask size should be halved");
    }

    #[test]
    fn test_emergency_short_position_clears_asks_keeps_bids() {
        let mut bids = vec![Quote::new(99.0, 1.0), Quote::new(98.0, 2.0)];
        let mut asks = vec![Quote::new(101.0, 1.0), Quote::new(102.0, 2.0)];
        // Short position: position=-5.0, max=10.0
        apply_risk_emergency_filter(&mut bids, &mut asks, -5.0, 10.0);

        assert!(asks.is_empty(), "Asks (increasing) should be cleared when short");
        assert_eq!(bids.len(), 2, "Bids (reducing) should be preserved");
        assert!((bids[0].size - 0.5).abs() < 1e-10, "Bid size should be halved");
        assert!((bids[1].size - 1.0).abs() < 1e-10, "Bid size should be halved");
    }

    #[test]
    fn test_emergency_zero_position_keeps_both_reduced() {
        let mut bids = vec![Quote::new(99.0, 1.0)];
        let mut asks = vec![Quote::new(101.0, 1.0)];
        // Near-zero position: position=0.001, max=10.0 -> ratio=0.0001 < 0.01
        apply_risk_emergency_filter(&mut bids, &mut asks, 0.001, 10.0);

        assert_eq!(bids.len(), 1, "Bids should be preserved at near-zero");
        assert_eq!(asks.len(), 1, "Asks should be preserved at near-zero");
        assert!((bids[0].size - 0.3).abs() < 1e-10, "Bid size should be 30%");
        assert!((asks[0].size - 0.3).abs() < 1e-10, "Ask size should be 30%");
    }

    #[test]
    fn test_emergency_reducing_quotes_survive() {
        // The core invariant: after emergency pull, reducing quotes must exist
        // so the system can unwind stuck positions.
        let mut bids = vec![Quote::new(99.0, 2.0)];
        let mut asks = vec![Quote::new(101.0, 2.0)];
        // Large long position
        apply_risk_emergency_filter(&mut bids, &mut asks, 12.0, 3.0);

        // position_ratio = 12/3 = 4.0 > 0.01 -> long path
        assert!(bids.is_empty(), "Bids cleared");
        assert!(!asks.is_empty(), "Asks (reducing) MUST survive emergency pull");
        assert!(asks[0].size > 0.0, "Reducing side must have positive size");
    }

    // =========================================================================
    // compute_tick_size_bps tests
    // =========================================================================

    #[test]
    fn test_tick_size_bps_hype() {
        // HYPE @ $31: 5 sig figs → 3 decimal places → tick = $0.001
        // tick_bps = 0.001 / 31 * 10000 ≈ 0.323 bps
        let tick = compute_tick_size_bps(31.0);
        assert!(tick > 0.3 && tick < 0.35, "HYPE tick_bps={tick}, expected ~0.32");
    }

    #[test]
    fn test_tick_size_bps_eth() {
        // ETH @ $3000: 5 sig figs → 1 decimal place → tick = $0.1
        // tick_bps = 0.1 / 3000 * 10000 ≈ 0.333 bps
        let tick = compute_tick_size_bps(3000.0);
        assert!(tick > 0.3 && tick < 0.4, "ETH tick_bps={tick}, expected ~0.33");
    }

    #[test]
    fn test_tick_size_bps_btc() {
        // BTC @ $97000: 5 sig figs → 0 decimal places → tick = $1.0
        // tick_bps = 1.0 / 97000 * 10000 ≈ 0.103 bps
        let tick = compute_tick_size_bps(97000.0);
        assert!(tick > 0.09 && tick < 0.15, "BTC tick_bps={tick}, expected ~0.10");
    }

    #[test]
    fn test_tick_size_bps_sol() {
        // SOL @ $200: 5 sig figs → 2 decimal places → tick = $0.01
        // tick_bps = 0.01 / 200 * 10000 = 0.5 bps
        let tick = compute_tick_size_bps(200.0);
        assert!((tick - 0.5).abs() < 0.05, "SOL tick_bps={tick}, expected 0.5");
    }

    #[test]
    fn test_tick_size_bps_zero_price() {
        // Zero/negative price should return safe default
        assert_eq!(compute_tick_size_bps(0.0), 1.0);
        assert_eq!(compute_tick_size_bps(-5.0), 1.0);
    }

    #[test]
    fn test_tick_size_bps_always_below_old_default() {
        // The old hardcoded value was 10.0 bps.
        // For any reasonable perp price ($0.01 to $1M), tick should be well below 10 bps.
        for price in [0.5, 1.0, 5.0, 31.0, 200.0, 3000.0, 50000.0, 97000.0] {
            let tick = compute_tick_size_bps(price);
            assert!(
                tick < 10.0,
                "tick_bps={tick} at price={price} should be < 10.0 (old hardcoded default)"
            );
        }
    }

    // ---------------------------------------------------------------
    // Stale data circuit breaker tests
    // ---------------------------------------------------------------
    #[test]
    fn test_staleness_penalty_fresh_data() {
        // Fresh data: no penalty
        assert!((compute_staleness_spread_penalty(100, 1000) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_staleness_penalty_stale_book() {
        // Book stale at 5s: 1.5x
        let p = compute_staleness_spread_penalty(6_000, 1_000);
        assert!((p - 1.5).abs() < f64::EPSILON, "Expected 1.5, got {p}");

        // Book very stale at 15s: 2.0x
        let p = compute_staleness_spread_penalty(16_000, 1_000);
        assert!((p - 2.0).abs() < f64::EPSILON, "Expected 2.0, got {p}");
    }

    #[test]
    fn test_staleness_penalty_stale_limits() {
        // Exchange limits stale at 30s: 1.5x
        let p = compute_staleness_spread_penalty(100, 35_000);
        assert!((p - 1.5).abs() < f64::EPSILON, "Expected 1.5, got {p}");

        // Exchange limits very stale at 2min: 2.0x
        let p = compute_staleness_spread_penalty(100, 130_000);
        assert!((p - 2.0).abs() < f64::EPSILON, "Expected 2.0, got {p}");
    }

    #[test]
    fn test_staleness_penalty_capped_at_3x() {
        // Both very stale: 2.0 * 2.0 = 4.0, capped at 3.0
        let p = compute_staleness_spread_penalty(20_000, 200_000);
        assert!((p - 3.0).abs() < f64::EPSILON, "Expected 3.0 cap, got {p}");
    }
}
