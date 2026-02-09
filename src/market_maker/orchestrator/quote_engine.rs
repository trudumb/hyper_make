//! Quote generation engine for the market maker.

use chrono::Timelike;
use tracing::{debug, error, info, trace, warn};

use crate::prelude::Result;

use super::super::{
    quoting, MarketMaker, OrderExecutor, ParameterAggregator, ParameterSources, Quote, QuoteConfig,
    QuotingStrategy, Side,
};
use crate::market_maker::belief::{BeliefSnapshot, BeliefUpdate};
use crate::market_maker::control::{QuoteGateDecision, QuoteGateInput};
use crate::market_maker::estimator::EnhancedFlowContext;
use crate::market_maker::infra::metrics::dashboard::{
    ChangepointDiagnostics, KappaDiagnostics, QuoteDecisionRecord, SignalSnapshot,
};
use crate::market_maker::risk::{CircuitBreakerAction, RiskCheckResult};
use crate::market_maker::strategy::action_to_inventory_ratio;

/// Minimum order notional value in USD (Hyperliquid requirement)
pub(super) const MIN_ORDER_NOTIONAL: f64 = 10.0;

impl<S: QuotingStrategy, E: OrderExecutor> MarketMaker<S, E> {
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
                self.executor
                    .cancel_bulk_orders(&self.config.asset, all_oids)
                    .await;
            }
            return Ok(());
        }

        // Don't place orders until estimator is warmed up (or timeout reached)
        if !self.estimator.is_warmed_up() {
            // Check warmup timeout: with informative Bayesian priors, we can safely
            // quote before all data thresholds are met. Spread floors and kill switches
            // provide protection. The estimator continues refining as data arrives.
            let max_warmup = self.estimator.max_warmup_secs();
            let elapsed = self.session_start_time.elapsed();
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
            Some(CircuitBreakerAction::WidenSpreads { multiplier }) => {
                info!(
                    multiplier = %format!("{:.2}x", multiplier),
                    "Circuit breaker: widening spreads"
                );
                // Multiplier is applied below via spread_multiplier composition
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

        // === KAPPA-BASED CYCLE SKIPPING (API Budget Conservation) ===
        // When kappa is high (thick book, many fills expected), skip some cycles
        // to let fills accumulate rather than churning through API budget.
        // This is a stochastic skip - higher kappa = higher skip probability.
        const HIGH_KAPPA_THRESHOLD: f64 = 5000.0; // fills/second - thick book
        const MIN_KAPPA_CONFIDENCE: f64 = 0.7; // Only skip if confident in kappa estimate

        let kappa = self.estimator.kappa();
        let kappa_confidence = self.estimator.kappa_confidence();

        if kappa > HIGH_KAPPA_THRESHOLD && kappa_confidence > MIN_KAPPA_CONFIDENCE {
            // Skip probability increases with kappa (more fills expected = skip more cycles)
            // At threshold: skip 0%, at 2x threshold: skip ~25%, at 3x: ~33%
            let skip_prob = ((kappa - HIGH_KAPPA_THRESHOLD) / kappa) * 0.5;

            // Use deterministic skip based on time to avoid randomness issues
            // Skip if current second modulo pattern matches
            let now = std::time::Instant::now();
            let should_skip = (now.elapsed().as_millis() as f64 / 1000.0) % 1.0 < skip_prob;

            if should_skip {
                debug!(
                    kappa = %format!("{:.0}", kappa),
                    kappa_confidence = %format!("{:.2}", kappa_confidence),
                    skip_prob = %format!("{:.2}", skip_prob),
                    "Skipping quote cycle: high kappa - letting fills accumulate"
                );
                return Ok(());
            }
        }

        // === RL HOT-RELOAD: Check for updated Q-table every 100 cycles ===
        self.rl_reload_cycle_count += 1;
        if self.rl_reload_cycle_count.is_multiple_of(100) {
            self.check_rl_reload();
        }

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
        }

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

        // === Lead-Lag Signal Integration ===
        // Wire cross-exchange lead-lag signal for predictive skew.
        // Uses SignalIntegrator which combines:
        // - Binance → Hyperliquid price discovery (when Binance feed enabled)
        // - Informed flow decomposition
        // - Regime-conditioned kappa
        // - Model gating (IR-based confidence)
        let signals = self.stochastic.signal_integrator.get_signals();

        // Record signal contributions for analytics attribution
        self.live_analytics.record_quote_cycle(&signals);

        if signals.lead_lag_actionable {
            // Real cross-exchange signal from Binance feed
            market_params.lead_lag_signal_bps = signals.combined_skew_bps;
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

        // Get spread multiplier from circuit breaker if applicable
        let mut spread_multiplier = match breaker_action {
            Some(CircuitBreakerAction::WidenSpreads { multiplier }) => multiplier,
            _ => 1.0,
        };

        // === First-Principles Gap 2: Apply ThresholdKappa regime-based spread multiplier ===
        // During momentum regimes (large price moves), widen spreads to reduce adverse selection.
        let threshold_kappa_mult = self.stochastic.threshold_kappa.regime().spread_multiplier();
        if threshold_kappa_mult > 1.0 && self.stochastic.threshold_kappa.is_warmed_up() {
            spread_multiplier *= threshold_kappa_mult;
            trace!(
                regime = ?self.stochastic.threshold_kappa.regime(),
                kappa_ratio = %format!("{:.2}", self.stochastic.threshold_kappa.kappa_ratio()),
                multiplier = %format!("{:.2}", threshold_kappa_mult),
                "ThresholdKappa: Momentum regime detected, widening spreads"
            );
        }

        // === MODEL GATING: Widen spreads when models are poorly calibrated ===
        let model_gating_mult = self.stochastic.signal_integrator.model_gating_spread_multiplier();
        if model_gating_mult > 1.0 {
            spread_multiplier *= model_gating_mult;
            debug!(
                model_gating_mult = %format!("{:.2}", model_gating_mult),
                "ModelGating: Widening spreads (low model confidence)"
            );
        }

        // === SIGNAL STALENESS DEFENSE: Widen when enabled signals go stale ===
        let staleness_mult = self.stochastic.signal_integrator.staleness_spread_multiplier();
        if staleness_mult > 1.0 {
            spread_multiplier *= staleness_mult;
            warn!(
                staleness_mult = %format!("{:.2}", staleness_mult),
                "Signal staleness: widening spreads (stale signals detected)"
            );
        }

        // === NEGATIVE EDGE ALARM: Widen when realized edge is negative ===
        if let Some(edge_alarm_mult) = self.tier2.edge_tracker.negative_edge_alarm() {
            spread_multiplier *= edge_alarm_mult;
            warn!(
                edge_alarm_mult = %format!("{:.2}", edge_alarm_mult),
                mean_edge_bps = %format!("{:.2}", self.tier2.edge_tracker.mean_realized_edge()),
                fills = self.tier2.edge_tracker.edge_count(),
                "Negative edge alarm: widening spreads defensively"
            );
        }

        // === EDGE KILL: Pause trading if consistently losing money per fill ===
        if self.tier2.edge_tracker.should_pause_trading() {
            warn!(
                mean_edge_bps = %format!("{:.2}", self.tier2.edge_tracker.mean_realized_edge()),
                fills = self.tier2.edge_tracker.edge_count(),
                "EDGE KILL: consistently negative realized edge, pausing quotes"
            );
            return Ok(());
        }

        // Apply composed spread multiplier to market params
        if spread_multiplier > 1.0 {
            market_params.spread_widening_mult *= spread_multiplier;
            debug!(
                multiplier = %format!("{:.2}", spread_multiplier),
                total_widening = %format!("{:.2}", market_params.spread_widening_mult),
                "Spread widening active (circuit breaker + threshold kappa + model gating + staleness + edge)"
            );
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
        // When Kelly tracker is warmed up (50+ fills), scale liquidity by optimal fraction.
        // Clamped to [5%, 30%] — never go below 5% to stay visible on the book.
        const KELLY_MIN_FRACTION: f64 = 0.05;
        const KELLY_MAX_FRACTION: f64 = 0.30;
        let kelly_adjusted_liquidity = if self.stochastic.stochastic_config.use_kelly_sizing {
            if let Some(raw_kelly) = self.learning.kelly_recommendation() {
                let kelly_fraction = raw_kelly.clamp(KELLY_MIN_FRACTION, KELLY_MAX_FRACTION);
                let adjusted = decision_adjusted_liquidity * kelly_fraction;
                info!(
                    raw_kelly = %format!("{:.1}%", raw_kelly * 100.0),
                    clamped_kelly = %format!("{:.1}%", kelly_fraction * 100.0),
                    base_liquidity = %format!("{:.6}", decision_adjusted_liquidity),
                    adjusted_liquidity = %format!("{:.6}", adjusted),
                    "Kelly criterion sizing active"
                );
                adjusted
            } else {
                decision_adjusted_liquidity
            }
        } else {
            decision_adjusted_liquidity
        };

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
                // PHASE 4: Using centralized belief snapshot for changepoint values
                l3_trust = %format!("{:.2}", belief_snapshot.changepoint.learning_trust),
                l3_cp_prob = %format!("{:.3}", belief_snapshot.changepoint.prob_5),
                l3_action = %format!("{action:?}").chars().take(40).collect::<String>(),
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
                Action::Quote { expected_value, .. } => {
                    debug!(expected_value = %format!("{:.4}", expected_value), "Layer 3: normal quote");
                    // Continue with normal quoting
                }
            }
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

        // === Phase 3: Compute kappa-driven spread ===
        // Uses fill intensity to dynamically adjust spread (higher kappa → tighter spread)
        let kappa_spread_result = self.stochastic.kappa_spread.compute_spread(
            market_params.market_spread_bps.max(5.0), // Use market spread as base, floor at 5 bps
            market_params.kappa,
        );
        market_params.kappa_spread_bps = Some(kappa_spread_result.spread_bps);

        // === Phase 8: RL Agent Policy Recommendation ===
        // Build MDP state from current market conditions
        let mdp_state = crate::market_maker::learning::MDPState::from_continuous(
            self.position.position(),
            self.effective_max_position,
            market_params.book_imbalance,
            market_params.sigma / market_params.sigma_effective.max(0.0001), // vol ratio
            self.stochastic.theoretical_edge.bayesian_adverse(),
            market_params.hawkes_branching_ratio,
        );

        // Get RL policy recommendation (exploration during bootstrap, exploitation after)
        let explore = !self.stochastic.calibrated_edge.is_useful();
        let rl_recommendation = crate::market_maker::learning::RLPolicyRecommendation::from_agent(
            &mut self.stochastic.rl_agent,
            mdp_state.to_index(),
            explore,
        );

        // Store state-action pair for credit assignment when fill occurs
        self.stochastic.rl_agent.push_state_action_idx(
            rl_recommendation.state_idx,
            rl_recommendation.action_idx,
        );

        // Populate MarketParams with RL recommendations
        market_params.rl_spread_delta_bps = rl_recommendation.spread_delta_bps;
        market_params.rl_bid_skew_bps = rl_recommendation.bid_skew_bps;
        market_params.rl_ask_skew_bps = rl_recommendation.ask_skew_bps;
        market_params.rl_confidence = rl_recommendation.confidence;
        market_params.rl_is_exploration = rl_recommendation.is_exploration;
        market_params.rl_expected_q = rl_recommendation.expected_q;

        // === RL Action Application (when enabled) ===
        if self.rl_enabled {
            let total_rl_updates = self.stochastic.rl_agent.total_updates();
            let mean_reward = self.stochastic.rl_agent.mean_recent_reward();

            let should_apply = total_rl_updates >= self.rl_min_real_fills as u64
                && (total_rl_updates < self.rl_auto_disable_fills as u64 || mean_reward >= 0.0);

            if should_apply {
                // Apply spread delta with safety floor: cannot go negative
                market_params.rl_spread_delta_bps = rl_recommendation
                    .spread_delta_bps
                    .max(-market_params.market_spread_bps + 2.0);

                // Apply skew directly (already stored above, but re-apply clamped)
                market_params.rl_bid_skew_bps = rl_recommendation.bid_skew_bps;
                market_params.rl_ask_skew_bps = rl_recommendation.ask_skew_bps;

                market_params.rl_action_applied = true;

                debug!(
                    spread_delta = %format!("{:.2}", rl_recommendation.spread_delta_bps),
                    bid_skew = %format!("{:.2}", rl_recommendation.bid_skew_bps),
                    ask_skew = %format!("{:.2}", rl_recommendation.ask_skew_bps),
                    confidence = %format!("{:.3}", rl_recommendation.confidence),
                    total_updates = total_rl_updates,
                    "RL action APPLIED to market params"
                );
            } else if total_rl_updates < self.rl_min_real_fills as u64 {
                debug!(
                    total_updates = total_rl_updates,
                    min_required = self.rl_min_real_fills,
                    "RL observation-only (insufficient fills)"
                );
            } else {
                warn!(
                    mean_reward = %format!("{:.3}", mean_reward),
                    total_updates = total_rl_updates,
                    "RL auto-disabled (negative mean reward)"
                );
            }
        }

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

        // Extract L2 values for Quote Gate, with safety net check
        let (l2_p_positive, l2_health_score) = if let Some(ref output) = l2_output {
            // Check edge bias safety net - if bias is critical, don't trust L2
            let bias_summary = self.learning.confidence_tracker().edge_bias_summary();
            if bias_summary.should_halt {
                warn!(
                    mean_bias_bps = %format!("{:.2}", bias_summary.mean_bias_bps),
                    "L2 edge bias critical - setting l2_model_health to 0"
                );
                (Some(output.p_positive_edge), 0.0) // Keep p_positive but zero health
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
            l3_trust: market_params.bootstrap_confidence.min(1.0),
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
            QuoteGateDecision::WidenSpreads { multiplier, changepoint_prob } => {
                // Changepoint pending confirmation - apply spread widening
                info!(
                    multiplier = %format!("{:.2}", multiplier),
                    changepoint_prob = %format!("{:.3}", changepoint_prob),
                    "Quote gate: widening spreads (changepoint pending)"
                );
                // Update market_params with the widening multiplier and changepoint_prob
                market_params.spread_widening_mult = *multiplier;
                market_params.changepoint_prob = *changepoint_prob;
            }
        }

        // Try multi-level ladder quoting first
        // HARMONIZED: Use decision-adjusted liquidity (first-principles derived, decision-scaled)
        let ladder = self.strategy.calculate_ladder(
            &quote_config,
            self.position.position(),
            self.effective_max_position, // First-principles limit
            kelly_adjusted_liquidity, // Decision + Kelly-adjusted viable size
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
                QuoteGateDecision::QuoteBoth
                | QuoteGateDecision::NoQuote { .. }
                | QuoteGateDecision::WidenSpreads { .. } => {
                    // QuoteBoth: normal, no filtering
                    // NoQuote: should have returned early, but defensive
                    // WidenSpreads: quote both sides (spread widening applied earlier)
                }
            }

            // Reduce-only mode: when over max position, position value, OR margin utilization
            // Phase 3: Use exchange-aware reduce-only that checks exchange limits and signals escalation
            // CAPITAL-EFFICIENT: Use margin utilization as primary trigger (80% threshold)
            let margin_state = self.infra.margin_sizer.state();
            // Get unrealized P&L for underwater position protection
            let unrealized_pnl = self.tier2.pnl_tracker.unrealized_pnl(self.latest_mid);
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
                // Underwater position protection - prevents forcing sales at a loss
                unrealized_pnl,
            };
            let reduce_only_result = quoting::QuoteFilter::apply_reduce_only_with_exchange_limits(
                &mut bid_quotes,
                &mut ask_quotes,
                &reduce_only_config,
                &self.infra.exchange_limits,
            );

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
                    bid_quotes.clear();
                }
                if reduce_only_result.filtered_asks {
                    ask_quotes.clear();
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
                        for quote in ask_quotes.iter_mut() {
                            let (_, new_ask) = quoting::apply_close_bias(
                                mid - 1.0, quote.price, mid, position, close_urgency,
                            );
                            quote.price = new_ask;
                        }
                    } else if position < 0.0 {
                        // Short: tighten bids (buy side)
                        for quote in bid_quotes.iter_mut() {
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
                kelly_adjusted_liquidity, // Decision + Kelly-adjusted viable size
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
                QuoteGateDecision::QuoteBoth
                | QuoteGateDecision::NoQuote { .. }
                | QuoteGateDecision::WidenSpreads { .. } => {
                    // QuoteBoth: normal, no filtering
                    // NoQuote: should have returned early, but defensive
                    // WidenSpreads: quote both sides (spread widening applied earlier)
                }
            }

            // Reduce-only mode: when over max position, position value, OR margin utilization
            // CAPITAL-EFFICIENT: Use margin utilization as primary trigger (80% threshold)
            let margin_state = self.infra.margin_sizer.state();
            // Get unrealized P&L for underwater position protection
            let unrealized_pnl = self.tier2.pnl_tracker.unrealized_pnl(self.latest_mid);
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
                // Underwater position protection - prevents forcing sales at a loss
                unrealized_pnl,
            };
            let reduce_only_result =
                quoting::QuoteFilter::apply_reduce_only_single(&mut bid, &mut ask, &reduce_only_config);

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
}
