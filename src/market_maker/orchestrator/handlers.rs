//! WebSocket message handlers for the market maker.

use std::sync::Arc;

use tracing::{debug, info, trace, warn};

use crate::prelude::Result;
use crate::Message;

use super::super::{
    adverse_selection::TradeObservation as MicroTradeObs,
    belief::BeliefUpdate,
    environment::Observation,
    environment::TradingEnvironment,
    estimator::{HmmObservation, MarketEstimator},
    fills,
    infra::{BinancePriceUpdate, BinanceTradeUpdate, BinanceUpdate},
    messages,
    tracking::ws_order_state::WsFillEvent,
    tracking::ws_order_state::WsOrderUpdateEvent,
    CascadeEvent, MarketMaker, OrderState, QuotingStrategy, Side, TrackedOrder,
};

impl<S: QuotingStrategy, Env: TradingEnvironment> MarketMaker<S, Env> {
    /// Handle a message from subscriptions.
    /// Main message dispatcher - routes to focused handlers.
    pub(crate) async fn handle_message(&mut self, message: Message) -> Result<()> {
        match message {
            Message::AllMids(all_mids) => self.handle_all_mids(all_mids).await,
            Message::Trades(trades) => self.handle_trades(trades),
            Message::UserFills(user_fills) => self.handle_user_fills(user_fills).await,
            Message::L2Book(l2_book) => self.handle_l2_book(l2_book),
            Message::OrderUpdates(order_updates) => self.handle_order_updates(order_updates),
            Message::OpenOrders(open_orders) => self.handle_open_orders(open_orders),
            Message::ActiveAssetData(active_asset_data) => {
                self.handle_active_asset_data(active_asset_data)
            }
            Message::WebData2(web_data2) => self.handle_web_data2(web_data2),
            Message::UserNonFundingLedgerUpdates(update) => self.handle_ledger_update(update),
            Message::UserFundings(fundings) => self.handle_user_fundings(fundings),
            _ => Ok(()),
        }
    }

    /// Handle an observation from the environment's observation stream.
    ///
    /// This is the **T: S × O → S** transition function — the same handlers
    /// are called regardless of whether the observation came from a live or
    /// paper environment.
    ///
    /// Bridge: delegates to the same handler methods as `handle_message()`.
    /// In Phase 4, the event loop will call this instead of `handle_message()`.
    #[allow(dead_code)] // Wired in Phase 3/4 when event loop uses observation stream
    pub(crate) async fn handle_observation(&mut self, obs: Observation) -> Result<()> {
        match obs {
            Observation::AllMids(all_mids) => self.handle_all_mids(all_mids).await,
            Observation::Trades(trades) => self.handle_trades(trades),
            Observation::UserFills(user_fills) => self.handle_user_fills(user_fills).await,
            Observation::L2Book(l2_book) => self.handle_l2_book(l2_book),
            Observation::OrderUpdates(order_updates) => self.handle_order_updates(order_updates),
            Observation::OpenOrders(open_orders) => self.handle_open_orders(open_orders),
            Observation::ActiveAssetData(active_asset_data) => {
                self.handle_active_asset_data(active_asset_data)
            }
            Observation::WebData2(web_data2) => self.handle_web_data2(*web_data2),
            Observation::LedgerUpdate(update) => self.handle_ledger_update(update),
            Observation::CrossVenuePrice {
                price,
                timestamp_ms,
            } => {
                self.handle_binance_price_update(BinancePriceUpdate {
                    timestamp_ms: timestamp_ms as i64,
                    mid_price: price,
                    best_bid: price, // Simplified — full data in Phase 4
                    best_ask: price,
                    spread_bps: 0.0,
                });
                Ok(())
            }
            Observation::CrossVenueTrade {
                price,
                size,
                is_buy,
                timestamp_ms,
            } => {
                self.handle_binance_trade(BinanceTradeUpdate {
                    timestamp_ms: timestamp_ms as i64,
                    price,
                    quantity: size,
                    is_buyer_maker: !is_buy,
                    trade_id: 0,
                });
                Ok(())
            }
        }
    }

    /// Handle AllMids message - updates mid price and triggers quote refresh.
    async fn handle_all_mids(&mut self, all_mids: crate::ws::message_types::AllMids) -> Result<()> {
        // === First-Principles Gap 2: Capture previous mid for return calculation ===
        let prev_mid = self.latest_mid;

        // HARMONIZED: Use effective_max_position for position utilization calculations
        let ctx = messages::MessageContext::new(
            self.config.asset.clone(),
            self.latest_mid,
            self.position.position(),
            self.effective_max_position, // First-principles limit
            self.estimator.is_warmed_up(),
            Arc::from(self.config.collateral.symbol.as_str()),
        );

        let mut state = messages::AllMidsState {
            estimator: &mut self.estimator,
            connection_health: &mut self.infra.connection_health,
            connection_supervisor: &self.infra.connection_supervisor,
            adverse_selection: &mut self.tier1.adverse_selection,
            depth_decay_as: &mut self.tier1.depth_decay_as,
            liquidation_detector: &mut self.tier1.liquidation_detector,
            hjb_controller: &mut self.stochastic.hjb_controller,
            stochastic_config: &self.stochastic.stochastic_config,
            latest_mid: &mut self.latest_mid,
        };

        let result = messages::process_all_mids(&all_mids, &ctx, &mut state)?;

        // Record price for dashboard unconditionally when we have a valid mid
        // This ensures dashboard shows price data even during warmup
        if self.latest_mid > 0.0 {
            self.infra
                .prometheus
                .record_price_for_dashboard(self.latest_mid);
        }

        if result.is_some() {
            // Update learning module with current mid for prediction scoring
            self.learning.update_mid(self.latest_mid);

            // === Signal Diagnostic: Update markouts with latest mid ===
            // This updates pending markouts at 500ms, 2s, 10s horizons.
            let current_time_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            self.safety
                .signal_store
                .update_markouts(self.latest_mid, current_time_ms);

            // Check if export is due (every 5 min or 100 fills)
            if self.safety.signal_store.should_export() {
                if let Some(path) = self.safety.signal_store.export_path() {
                    let path = path.to_string();
                    if let Err(e) = self.safety.signal_store.export_to_json(&path) {
                        tracing::warn!(error = %e, path = %path, "Failed to export fill signals");
                    }
                }
            }

            // === AS Outcome Feedback: Check 5-second markout on pending fills ===
            // Drain fills older than 5 seconds: compute whether mid moved against us
            // and feed outcome to pre-fill classifier + model gating.
            self.check_pending_fill_outcomes(current_time_ms);

            // === First-Principles Gap 2: Update ThresholdKappa with return observation ===
            // Feed return_bps to ThresholdKappa for TAR model regime detection.
            // This determines whether we're in mean-reversion or momentum regime.
            if prev_mid > 0.0 && self.latest_mid > 0.0 {
                let return_bps = (self.latest_mid - prev_mid) / prev_mid * 10_000.0;
                self.stochastic.threshold_kappa.update(return_bps);

                // Track information gain for AdaptiveCycleTimer
                self.cycle_state_changes.mid_move_bps += return_bps.abs();
            }

            // === Price Velocity Tracking (Flash Crash Detection) ===
            // Compute abs(delta_mid / mid) per second for flash crash detection.
            if self.last_mid_for_velocity > 0.0 && self.latest_mid > 0.0 {
                let elapsed = self.last_mid_velocity_time.elapsed();
                let elapsed_s = elapsed.as_secs_f64();
                if elapsed_s > 0.01 {
                    // Avoid division by tiny elapsed times
                    let velocity = ((self.latest_mid - self.last_mid_for_velocity)
                        / self.last_mid_for_velocity)
                        .abs()
                        / elapsed_s;
                    self.price_velocity_1s = velocity;
                    self.last_mid_for_velocity = self.latest_mid;
                    self.last_mid_velocity_time = std::time::Instant::now();
                }
            } else if self.latest_mid > 0.0 {
                // First valid mid — seed the velocity tracker
                self.last_mid_for_velocity = self.latest_mid;
                self.last_mid_velocity_time = std::time::Instant::now();
                self.price_velocity_1s = 0.0;
            }

            // === Cross-Exchange Signal Integration: Feed HL price ===
            // Update SignalIntegrator with Hyperliquid mid price for lead-lag calculation.
            // This pairs with Binance prices from handle_binance_price_update(),
            // OR reference perp prices for HIP-3 tokens (below).
            self.stochastic
                .signal_integrator
                .on_hl_price(self.latest_mid, current_time_ms as i64);

            // === Reference Perp Lead-Lag (Fix C) ===
            // For HIP-3 tokens (e.g. hyna:HYPE), the main perp (HYPE) is the
            // leading signal. Extract its mid from AllMids and feed it to the
            // signal integrator's lead-lag system, replacing the Binance feed.
            if let Some(ref ref_sym) = self.config.reference_symbol {
                if let Some(ref_mid_str) = all_mids.data.mids.get(ref_sym) {
                    if let Ok(ref_mid) = ref_mid_str.parse::<f64>() {
                        if ref_mid > 0.0 {
                            self.prev_reference_perp_mid = self.reference_perp_mid;
                            self.reference_perp_mid = ref_mid;

                            // Compute EMA-smoothed drift rate for GLFT asymmetric spreads
                            if self.prev_reference_perp_mid > 0.0
                                && self.reference_perp_last_update_ms > 0
                            {
                                let dt_secs = ((current_time_ms as i64
                                    - self.reference_perp_last_update_ms)
                                    as f64
                                    / 1000.0)
                                    .max(0.1);
                                let return_per_sec =
                                    (ref_mid / self.prev_reference_perp_mid - 1.0) / dt_secs;
                                // EMA α=0.1: smooth over ~10 updates (~10-30s depending on AllMids frequency)
                                self.reference_perp_drift_ema =
                                    0.9 * self.reference_perp_drift_ema + 0.1 * return_per_sec;
                            }
                            self.reference_perp_last_update_ms = current_time_ms as i64;

                            // Feed reference perp as the leading signal for lead-lag
                            self.stochastic
                                .signal_integrator
                                .on_binance_price(ref_mid, current_time_ms as i64);

                            // Fix 13: Update EchoEstimator on external reference move
                            // If reference mid changed, feed the current trend magnitude
                            if (ref_mid - self.prev_reference_perp_mid).abs() > 0.0 {
                                let position_value =
                                    (self.position.position().abs() * self.latest_mid).max(1.0);
                                let trend_signal = self.estimator.trend_signal(position_value);
                                self.echo_estimator
                                    .update_on_reference_move(trend_signal.long_momentum_bps.abs());
                            }
                        }
                    }
                }
            }

            // === Phase 3: Event-Driven Quote Updates ===
            // Instead of calling update_quotes() on every AllMids, record the event
            // to the accumulator. The event loop will check should_trigger() and
            // call update_quotes() only when meaningful events have accumulated.
            if self.event_accumulator.is_enabled() {
                // Record mid price update - only triggers if move exceeds threshold (default 5 bps)
                let _trigger = self.event_accumulator.on_mid_update(self.latest_mid);
                // Note: actual reconciliation happens in event loop via check_event_accumulator()
                Ok(())
            } else {
                // Fallback: event-driven mode disabled, use original timed polling
                self.update_quotes().await
            }
        } else {
            Ok(())
        }
    }

    /// Check pending fill outcomes for 5-second adverse selection markout.
    ///
    /// For each fill older than 5 seconds, compute whether mid moved against us:
    /// - Buy fill is adverse if mid dropped > 1 bps (we overpaid)
    /// - Sell fill is adverse if mid rose > 1 bps (we undersold)
    ///
    /// Feeds outcomes to pre-fill classifier and model gating for calibration.
    /// Checks adverse selection outcomes from pending fills.
    fn check_pending_fill_outcomes(&mut self, now_ms: u64) {
        const OUTCOME_DELAY_MS: u64 = 5_000; // 5-second markout window
                                             // Volatility-scaled threshold: 2σ × √τ prevents noise misclassification.
                                             // Random walk E[|ΔP|] = σ × √τ. At 2σ, only ~5% of noise moves are misclassified.
                                             // BTC (σ≈2bps/√s): threshold ≈ 8.9 bps. HYPE (σ≈1bps/√s): threshold ≈ 4.5 bps.
        const ADVERSE_NOISE_MULT: f64 = 2.0;
        const MIN_ADVERSE_THRESHOLD_BPS: f64 = 1.0;
        const MARKOUT_SECONDS: f64 = OUTCOME_DELAY_MS as f64 / 1000.0;

        let sigma_bps = self.estimator.sigma() * 10_000.0;
        let adverse_threshold_bps = (ADVERSE_NOISE_MULT * sigma_bps * MARKOUT_SECONDS.sqrt())
            .max(MIN_ADVERSE_THRESHOLD_BPS);

        while let Some(front) = self.infra.pending_fill_outcomes.front() {
            if now_ms.saturating_sub(front.timestamp_ms) < OUTCOME_DELAY_MS {
                break; // Not old enough yet
            }

            let pending = self.infra.pending_fill_outcomes.pop_front().unwrap();
            if self.latest_mid <= 0.0 || pending.mid_at_fill <= 0.0 {
                continue;
            }

            // Compute adverse selection:
            // Buy fill is adverse if mid dropped (we overpaid)
            // Sell fill is adverse if mid rose (we undersold)
            let mid_change_bps =
                ((self.latest_mid - pending.mid_at_fill) / pending.mid_at_fill) * 10_000.0;
            let was_adverse = if pending.is_buy {
                mid_change_bps < -adverse_threshold_bps
            } else {
                mid_change_bps > adverse_threshold_bps
            };
            let magnitude_bps = mid_change_bps.abs();

            // Feed outcome to pre-fill classifier for Bayesian AS learning
            self.tier1.pre_fill_classifier.record_outcome(
                pending.is_buy, // is_bid = is_buy (our bid was filled)
                was_adverse,
                Some(magnitude_bps),
            );

            // Feed outcome to enhanced microstructure classifier
            self.tier1.enhanced_classifier.record_outcome(
                pending.is_buy,
                was_adverse,
                Some(magnitude_bps),
            );

            // === P0: Feed fill outcome to Bayesian parameter learner ===
            // This updates all 27 conjugate prior parameters from markout data.
            {
                use crate::market_maker::calibration::parameter_learner::FillOutcome;

                let direction = if pending.is_buy { 1.0 } else { -1.0 };
                // Markout AS: how much did mid move against us from fill time to markout time.
                // Uses mid_at_fill as reference (not fill_price) to isolate pure mid movement
                // from the depth/spread component which is already captured in fill_distance_bps.
                let mid_at_markout = self.latest_mid;
                let as_realized =
                    (mid_at_markout - pending.mid_at_fill) * direction / pending.mid_at_fill;
                let as_realized_bps = as_realized * 10_000.0;
                let fill_distance_bps = ((pending.fill_price - pending.mid_at_fill).abs()
                    / pending.mid_at_fill)
                    * 10_000.0;
                let fill_pnl = -as_realized; // Negative AS = profit
                                             // Use actual AS estimate (bps) from the estimator, not toxicity probability × 10,000
                let predicted_as_bps_val = self.estimator.total_as_bps();

                let outcome = FillOutcome {
                    was_informed: was_adverse,
                    realized_as_bps: as_realized_bps,
                    fill_distance_bps,
                    realized_pnl: fill_pnl,
                    inter_arrival_s: None, // Not tracked at markout time
                    fills_in_window: None,
                    window_exposure: None,
                    predicted_as_bps: Some(predicted_as_bps_val),
                };
                self.stochastic.learned_params.observe_fill(&outcome);
            }

            // Feed outcome to model gating for AS prediction calibration
            let predicted_as_prob = self.tier1.pre_fill_classifier.cached_toxicity();
            self.stochastic
                .signal_integrator
                .update_as_prediction(predicted_as_prob, was_adverse);

            // === Bootstrap from Book: Feed markout outcome to CalibrationCoordinator ===
            // Uses markout-based AS (not fill-time AS) to avoid the tautology bug.
            // fill_distance_bps = distance from mid to fill price (already computed above).
            {
                let fill_dist_bps = ((pending.fill_price - pending.mid_at_fill).abs()
                    / pending.mid_at_fill)
                    * 10_000.0;
                self.calibration_coordinator
                    .on_fill(fill_dist_bps, was_adverse);
            }

            // === Bayesian sigma correction: feed markout to CovarianceTracker ===
            // realized_bps = absolute mid movement from fill to markout (5s window).
            // predicted_bps = sigma_effective × √τ × 10_000 (what we predicted).
            {
                let realized_bps = mid_change_bps.abs();
                let sigma_eff = self.estimator.sigma_effective();
                let predicted_bps = sigma_eff * MARKOUT_SECONDS.sqrt() * 10_000.0;
                self.covariance_tracker
                    .observe_markout(realized_bps, predicted_bps);
            }

            // Close prediction→correction loop: if classifier overpredicts AS,
            // bias correction reduces effective toxicity in spread_multiplier().
            // predicted_as_bps_val and markout_as_bps computed below in EdgeSnapshot block.
            let markout_direction_for_bias = if pending.is_buy { 1.0 } else { -1.0 };
            let markout_as_bps_for_bias = if pending.mid_at_fill > 0.0 {
                (self.latest_mid - pending.mid_at_fill) * markout_direction_for_bias
                    / pending.mid_at_fill
                    * 10_000.0
            } else {
                0.0
            };
            let predicted_as_bps_for_bias = self.estimator.total_as_bps();
            self.tier1
                .pre_fill_classifier
                .observe_as_outcome(predicted_as_bps_for_bias, markout_as_bps_for_bias);

            // === RESOLVED EDGE SNAPSHOT: markout-based edge measurement ===
            // This is the ground-truth edge measurement using 5-second markout AS.
            // Fill-time "instant AS" was tautological (AS ≈ depth from same mid).
            // Now: spread = |fill - placement_mid|, AS = markout mid movement.
            {
                use crate::market_maker::analytics::edge_metrics::EdgePhase;
                use crate::market_maker::analytics::EdgeSnapshot;
                const MAKER_FEE_BPS: f64 = 1.5;

                let markout_direction = if pending.is_buy { 1.0 } else { -1.0 };
                // Markout AS: mid movement from fill time to 5s later.
                // Uses mid_at_fill (not fill_price) to isolate pure adverse mid drift
                // from the depth/spread component already captured in quoted_spread_bps.
                let markout_as_bps = if pending.mid_at_fill > 0.0 {
                    (self.latest_mid - pending.mid_at_fill) * markout_direction
                        / pending.mid_at_fill
                        * 10_000.0
                } else {
                    0.0
                };

                let predicted_as_bps_val = pending.predicted_as_bps;
                let resolved_snap = EdgeSnapshot {
                    timestamp_ns: pending.timestamp_ms * 1_000_000,
                    predicted_spread_bps: pending.quoted_spread_bps,
                    realized_spread_bps: pending.quoted_spread_bps,
                    predicted_as_bps: predicted_as_bps_val,
                    realized_as_bps: markout_as_bps,
                    fee_bps: MAKER_FEE_BPS,
                    predicted_edge_bps: pending.quoted_spread_bps
                        - predicted_as_bps_val
                        - MAKER_FEE_BPS,
                    realized_edge_bps: pending.quoted_spread_bps - markout_as_bps - MAKER_FEE_BPS,
                    gross_edge_bps: pending.quoted_spread_bps - markout_as_bps,
                    phase: EdgePhase::Resolved,
                    mid_at_placement: pending.mid_at_placement,
                    markout_as_bps: Some(markout_as_bps),
                };

                // === PHASE 4: Close measurement loop — feed markout AS to regime state ===
                self.stochastic
                    .regime_state
                    .params
                    .update_as_from_markout(markout_as_bps);

                // === PHASE 5: Online drift parameter adaptation ===
                // Feed realized drift (mid movement in bps over markout window) to Kalman
                // parameter adaptation. This adjusts θ and process_noise for better calibration.
                let realized_drift_bps_per_sec = if pending.mid_at_fill > 0.0 {
                    let mid_change_bps =
                        (self.latest_mid - pending.mid_at_fill) / pending.mid_at_fill * 10_000.0;
                    // Convert to bps/sec: markout window is 5 seconds
                    mid_change_bps / 5.0
                } else {
                    0.0
                };
                self.drift_estimator
                    .update_parameters(realized_drift_bps_per_sec);

                // === PHASE 8: Adaptive spread learning at markout time ===
                // Moved from fill-time handler where AS was tautological (AS ≈ depth).
                // Now uses markout AS = true mid movement over 5 seconds.
                let markout_as_fraction = markout_as_bps / 10_000.0;
                let depth_from_mid = if pending.mid_at_fill > 0.0 {
                    (pending.fill_price - pending.mid_at_fill).abs() / pending.mid_at_fill
                } else {
                    0.0
                };
                let markout_fill_pnl = -markout_as_fraction;
                self.stochastic.adaptive_spreads.on_fill_simple(
                    markout_as_fraction,
                    depth_from_mid,
                    markout_fill_pnl,
                    self.estimator.kappa(),
                );

                // Feed fill outcome to queue value heuristic for online bias correction.
                // depth_from_mid is already in fractional form, convert to bps.
                let depth_bps_for_qv = depth_from_mid * 10_000.0;
                let predicted_qv = self.queue_value_heuristic.queue_value(
                    depth_bps_for_qv,
                    crate::market_maker::adverse_selection::ToxicityRegime::Normal, // conservative default
                    0.0, // queue rank not tracked at fill time
                );
                self.queue_value_heuristic
                    .observe_outcome(predicted_qv, markout_fill_pnl * 10_000.0);

                // Feed fill outcome to reconcile outcome tracker for action-type learning
                self.reconcile_outcome_tracker
                    .record_fill(pending.oid, resolved_snap.realized_edge_bps);

                // Feed fill to cancel-race AS tracker (Sprint 6.3)
                // Uses markout AS (not fill-time) for true adverse selection measurement
                self.cancel_race_tracker.record_fill(
                    pending.oid,
                    markout_as_bps,
                    pending.timestamp_ms,
                );

                // Feed resolved snapshot to learning systems
                self.tier2.edge_tracker.add_snapshot(resolved_snap.clone());
                let fill_pnl_bps = resolved_snap.realized_edge_bps;
                self.live_analytics
                    .record_fill(fill_pnl_bps, Some(&resolved_snap));

                // Update AdaptiveEnsemble with GLFT prediction performance
                {
                    let prediction_error =
                        resolved_snap.predicted_edge_bps - resolved_snap.realized_edge_bps;
                    // IR proxy: realized edge (positive = model captured value)
                    let ir_proxy = resolved_snap.realized_edge_bps;
                    // Brier proxy: squared prediction error (lower = better calibration)
                    let brier_proxy = prediction_error * prediction_error;
                    self.stochastic.ensemble.update_performance(
                        "GLFT",
                        ir_proxy,
                        brier_proxy,
                        pending.timestamp_ms,
                    );
                }

                // === Kelly Sizing: feed markout-based win/loss ===
                // Positive realized edge = win (we captured spread minus AS minus fees).
                // Negative = loss. Uses markout-based edge, not fill-time edge.
                if fill_pnl_bps > 0.0 {
                    self.strategy.record_kelly_win(fill_pnl_bps);
                } else if fill_pnl_bps < 0.0 {
                    self.strategy.record_kelly_loss(-fill_pnl_bps); // loss as positive magnitude
                }

                tracing::info!(
                    spread_bps = %format!("{:.2}", pending.quoted_spread_bps),
                    markout_as_bps = %format!("{:.2}", markout_as_bps),
                    edge_bps = %format!("{:.2}", resolved_snap.realized_edge_bps),
                    gross_bps = %format!("{:.2}", resolved_snap.gross_edge_bps),
                    mid_at_placement = %format!("{:.4}", pending.mid_at_placement),
                    "[EDGE RESOLVED] markout-based edge measurement"
                );
            }

            if was_adverse {
                tracing::info!(
                    side = if pending.is_buy { "Buy" } else { "Sell" },
                    magnitude_bps = %format!("{:.1}", magnitude_bps),
                    mid_at_fill = %format!("{:.4}", pending.mid_at_fill),
                    mid_now = %format!("{:.4}", self.latest_mid),
                    "[AS OUTCOME] adverse fill detected via 5s markout"
                );
            }
        }
    }

    /// Handle Trades message - updates volatility and flow estimates.
    fn handle_trades(&mut self, trades: crate::ws::message_types::Trades) -> Result<()> {
        // HARMONIZED: Use effective_max_position for position utilization calculations
        let ctx = messages::MessageContext::new(
            self.config.asset.clone(),
            self.latest_mid,
            self.position.position(),
            self.effective_max_position, // First-principles limit
            self.estimator.is_warmed_up(),
            Arc::from(self.config.collateral.symbol.as_str()),
        );

        let mut state = messages::TradesState {
            estimator: &mut self.estimator,
            hawkes: &mut self.tier2.hawkes,
            data_quality: &mut self.infra.data_quality,
            prometheus: &mut self.infra.prometheus,
            connection_supervisor: &self.infra.connection_supervisor,
            last_warmup_log: &mut self.last_warmup_log,
        };

        let _result = messages::process_trades(&trades, &ctx, &mut state)?;

        // Cache trades for EnhancedFlowContext momentum calculation
        // Buffer recent trades (size, is_buy, timestamp_ms) for the quote engine
        const MAX_CACHED_TRADES: usize = 500;
        for trade in &trades.data {
            // Filter to our asset
            if trade.coin != *self.config.asset {
                continue;
            }
            // Parse trade data
            let size: f64 = trade.sz.parse().unwrap_or(0.0);
            if size <= 0.0 {
                continue;
            }
            let is_buy = trade.side == "B";
            let timestamp_ms = trade.time;
            let trade_price: f64 = trade.px.parse().unwrap_or(0.0);

            // Feed trade to Shadow Tuner buffer (lock-free, non-blocking)
            if trade_price > 0.0 {
                self.feed_shadow_buffer(crate::market_maker::simulation::ReplayEvent::Trade {
                    timestamp_ns: timestamp_ms * 1_000_000,
                    price: trade_price,
                    size,
                    is_buy,
                });
            }

            // Add to buffer
            self.cached_trades.push_back((size, is_buy, timestamp_ms));

            // Track information gain for AdaptiveCycleTimer
            self.cycle_state_changes.trades_observed += 1;

            // Keep bounded
            while self.cached_trades.len() > MAX_CACHED_TRADES {
                self.cached_trades.pop_front();
            }

            // Phase 7: Forward market trade to centralized belief state (primary consumer)
            if trade_price > 0.0 {
                self.central_beliefs.update(BeliefUpdate::MarketTrade {
                    price: trade_price,
                    mid: self.latest_mid,
                    timestamp_ms,
                });

                // Decrement queue depth for orders at this trade's price level.
                // A buy trade lifts asks, a sell trade hits bids.
                self.tier1
                    .queue_tracker
                    .on_market_trade(trade_price, size, is_buy);
            }

            // === Phase 1: Microstructure Signals - VPIN Update ===
            // Feed trade to VPIN estimator. When a bucket completes, publish MicrostructureUpdate.
            // VPIN uses tick rule (price vs mid) to classify trades, not explicit is_buy.

            // Phase 1A: Feed trade to size distribution tracker
            self.stochastic.trade_size_dist.on_trade(size);

            // === Enhanced Microstructure Classifier: Trade Update ===
            // Feed trade to z-score normalized feature extractor
            self.tier1.enhanced_classifier.on_trade(MicroTradeObs {
                timestamp_ms,
                price: trade_price,
                size,
                is_buy,
            });

            // Wire trade to buy pressure tracker in SignalIntegrator
            self.stochastic
                .signal_integrator
                .on_trade_for_pressure(size, is_buy);

            // Wire trade to EWMA flow tracker for FlowFeatureVec
            self.stochastic.trade_flow_tracker.on_trade(size, is_buy);

            // === Phase 7: Sweep Detector ===
            // Multi-level fills within a short window indicate aggressive informed trading.
            // For now, use price impact as a proxy for levels crossed:
            // impact > 1 tick ≈ swept through multiple levels.
            let levels_crossed = if trade_price > 0.0 && self.latest_mid > 0.0 {
                let impact_bps =
                    ((trade_price - self.latest_mid).abs() / self.latest_mid) * 10_000.0;
                // Assume ~5 bps per level for typical HIP assets
                (impact_bps / 5.0).ceil().max(1.0) as u32
            } else {
                1
            };
            self.tier1
                .sweep_detector
                .record_fill(is_buy, size, levels_crossed, timestamp_ms);

            // === P0: Wire trade features to SignalIntegrator for informed flow classification ===
            {
                use crate::market_maker::estimator::informed_flow::TradeFeatures;

                // Compute inter-arrival from last cached trade
                let inter_arrival_ms = self
                    .cached_trades
                    .back()
                    .map(|&(_, _, prev_ts)| timestamp_ms.saturating_sub(prev_ts))
                    .unwrap_or(1000); // Default 1s if no previous trade

                // Price impact: |trade_price - mid| / mid * 10000 (bps)
                let price_impact_bps = if self.latest_mid > 0.0 {
                    ((trade_price - self.latest_mid).abs() / self.latest_mid) * 10_000.0
                } else {
                    0.0
                };

                let features = TradeFeatures {
                    size,
                    inter_arrival_ms,
                    price_impact_bps,
                    book_imbalance: self.estimator.book_imbalance(),
                    is_buy,
                    timestamp_ms,
                };
                self.stochastic.signal_integrator.on_trade(&features);

                // === WS4: Wire InformedFlow EM → AS estimator alpha calibration ===
                // After each trade, if the EM decomposition has sufficient confidence,
                // calibrate the AS estimator's P(informed) using the flow model's estimate.
                // This replaces hardcoded alpha weights (0.3, 0.4, 0.3) with empirical values.
                let flow_decomp = self.stochastic.signal_integrator.flow_decomposition();
                if flow_decomp.confidence > 0.25 {
                    self.tier1
                        .adverse_selection
                        .calibrate_alpha_from_flow(flow_decomp.p_informed, flow_decomp.confidence);
                }
            }

            if let Some(_vpin_value) =
                self.stochastic
                    .vpin
                    .on_trade(size, trade_price, self.latest_mid, timestamp_ms)
            {
                // Bucket completed - publish microstructure update to central beliefs
                let vpin = self.stochastic.vpin.vpin();
                let vpin_velocity = self.stochastic.vpin.vpin_velocity();

                // Wire VPIN into SignalIntegrator for toxicity blending
                let vpin_fresh =
                    self.stochastic.vpin.is_valid() && !self.stochastic.vpin.is_stale(timestamp_ms);
                self.stochastic
                    .signal_integrator
                    .set_vpin(vpin, vpin_velocity, vpin_fresh);
                let order_flow_direction = self.stochastic.vpin.order_flow_direction();
                let vpin_buckets = self.stochastic.vpin.bucket_count();

                // Get depth-weighted imbalance from cached book sizes
                // Note: using imbalance (static snapshot) rather than OFI (delta)
                // since we don't track previous book state here
                use crate::market_maker::estimator::BookLevel;
                let bid_levels: Vec<BookLevel> = self
                    .cached_bid_sizes
                    .iter()
                    .map(|&sz| BookLevel { size: sz })
                    .collect();
                let ask_levels: Vec<BookLevel> = self
                    .cached_ask_sizes
                    .iter()
                    .map(|&sz| BookLevel { size: sz })
                    .collect();
                let depth_ofi = self
                    .stochastic
                    .enhanced_flow
                    .depth_weighted_imbalance(&bid_levels, &ask_levels);

                // Get liquidity evaporation (updated by L2 handler, just read current value)
                let liquidity_evaporation =
                    self.stochastic.liquidity_evaporation.evaporation_score();

                // Confidence based on bucket count (more buckets = more confidence)
                let confidence = (vpin_buckets as f64 / 50.0).min(1.0);

                // Phase 1A: Get trade size anomaly metrics
                let trade_size_sigma = self.stochastic.trade_size_dist.median_sigma();
                let toxicity_acceleration =
                    self.stochastic.trade_size_dist.toxicity_acceleration(vpin);

                // Phase 1A: Get COFI metrics (updated by L2 handler)
                let cofi = self.stochastic.cofi.cofi();
                let cofi_velocity = self.stochastic.cofi.cofi_velocity();
                let is_sustained_shift = self.stochastic.cofi.is_sustained_shift();

                self.central_beliefs
                    .update(BeliefUpdate::MicrostructureUpdate {
                        vpin,
                        vpin_velocity,
                        depth_ofi,
                        liquidity_evaporation,
                        order_flow_direction,
                        confidence,
                        vpin_buckets,
                        // Phase 1A fields
                        trade_size_sigma,
                        toxicity_acceleration,
                        cofi,
                        cofi_velocity,
                        is_sustained_shift,
                    });

                trace!(
                    vpin = %format!("{:.3}", vpin),
                    vpin_velocity = %format!("{:.4}", vpin_velocity),
                    depth_ofi = %format!("{:.3}", depth_ofi),
                    liquidity_evaporation = %format!("{:.3}", liquidity_evaporation),
                    vpin_buckets = vpin_buckets,
                    trade_size_sigma = %format!("{:.2}", trade_size_sigma),
                    cofi = %format!("{:.3}", cofi),
                    "Microstructure update: VPIN bucket completed"
                );
            }
        }

        // === Phase 2/3 Component Updates ===
        // Update regime HMM with trade observation for soft regime blending
        let obs = HmmObservation::new(
            self.estimator.sigma(),
            self.tier2.spread_tracker.current_spread_bps(),
            self.tier2.hawkes.flow_imbalance(),
        );
        self.stochastic.regime_hmm.forward_update(&obs);

        // Log regime state periodically (when not in normal regime)
        let regime_probs = self.stochastic.regime_hmm.regime_probabilities();

        // === P0: Forward HMM regime probabilities to SignalIntegrator ===
        // This feeds the RegimeKappaEstimator for regime-blended kappa estimation.
        self.stochastic
            .signal_integrator
            .set_regime_probabilities(regime_probs);

        if regime_probs[3] > 0.2 || regime_probs[2] > 0.4 {
            trace!(
                sigma = %format!("{:.6}", self.estimator.sigma()),
                flow_imbalance = %format!("{:.3}", self.tier2.hawkes.flow_imbalance()),
                p_extreme = %format!("{:.2}", regime_probs[3]),
                p_high = %format!("{:.2}", regime_probs[2]),
                "HMM regime elevated"
            );
        }

        // === Position Continuation Model: Sync regime ===
        // Map HMM regime probabilities to continuation model regime names
        // The model uses regime-specific priors: cascade (high), bursty, normal, quiet (low)
        let continuation_regime = if regime_probs[3] > 0.3 {
            "cascade" // Extreme volatility → high continuation prior (0.8)
        } else if regime_probs[2] > 0.4 {
            "bursty" // High volatility → elevated continuation prior (0.6)
        } else if regime_probs[0] > 0.5 {
            "quiet" // Low volatility → low continuation prior (0.3)
        } else {
            "normal" // Normal volatility → neutral prior (0.5)
        };

        // Check if regime changed and reset if needed
        if self.stochastic.position_decision.current_regime() != continuation_regime {
            debug!(
                old_regime = %self.stochastic.position_decision.current_regime(),
                new_regime = %continuation_regime,
                p_extreme = %format!("{:.2}", regime_probs[3]),
                "Position continuation: regime change detected"
            );
            self.stochastic
                .position_decision
                .reset_for_regime(continuation_regime);
        }

        // === P0: Forward HL flow features to SignalIntegrator for cross-venue comparison ===
        // Construct FlowFeatureVec from HL-side estimators (VPIN, Hawkes, trade data).
        {
            use crate::market_maker::estimator::FlowFeatureVec;

            let hl_flow = FlowFeatureVec {
                vpin: self.stochastic.vpin.vpin(),
                vpin_velocity: self.stochastic.vpin.vpin_velocity(),
                imbalance_1s: self.stochastic.trade_flow_tracker.imbalance_at_1s(),
                imbalance_5s: self.stochastic.trade_flow_tracker.imbalance_at_5s(),
                imbalance_30s: self.stochastic.trade_flow_tracker.imbalance_at_30s(),
                imbalance_5m: self.stochastic.trade_flow_tracker.imbalance_at_5m(),
                intensity: self.tier2.hawkes.intensity_ratio(),
                avg_buy_size: self.stochastic.trade_flow_tracker.avg_buy_size(),
                avg_sell_size: self.stochastic.trade_flow_tracker.avg_sell_size(),
                size_ratio: self.stochastic.trade_flow_tracker.size_ratio(),
                order_flow_direction: self.stochastic.vpin.order_flow_direction(),
                timestamp_ms: self.cached_trades.back().map(|t| t.2 as i64).unwrap_or(0),
                trade_count: self.cached_trades.len() as u64,
                confidence: if self.stochastic.vpin.is_valid() {
                    0.8
                } else {
                    0.1
                },
            };
            self.stochastic
                .signal_integrator
                .set_hl_flow_features(hl_flow);
        }

        // === Feed Hawkes excitation predictor for reactive spread widening ===
        // Forward latest summary from the order flow estimator to the excitation predictor.
        // This drives spread_widening_factor() consumed in the quote engine.
        {
            let hawkes_summary = self.tier2.hawkes.summary();
            self.stochastic
                .hawkes_predictor
                .update_summary(hawkes_summary);
        }

        Ok(())
    }

    /// Handle UserFills message - processes fills through FillProcessor.
    async fn handle_user_fills(
        &mut self,
        user_fills: crate::ws::message_types::UserFills,
    ) -> Result<()> {
        // Skip snapshot fills - these are historical fills from previous sessions.
        // Position is already loaded from exchange at startup, so processing these
        // would cause "untracked order filled" warnings for orders we didn't place.
        if user_fills.data.is_snapshot.unwrap_or(false) {
            debug!(
                fills = user_fills.data.fills.len(),
                "Skipping UserFills snapshot (historical fills from previous sessions)"
            );
            return Ok(());
        }

        // HARMONIZED: Use effective_max_position for position utilization calculations
        let ctx = messages::MessageContext::new(
            self.config.asset.clone(),
            self.latest_mid,
            self.position.position(),
            self.effective_max_position, // First-principles limit
            self.estimator.is_warmed_up(),
            Arc::from(self.config.collateral.symbol.as_str()),
        );

        // Create fill state for the processor
        // HARMONIZED: Use effective_max_position for position threshold warnings
        let mut fill_state = fills::FillState {
            position: &mut self.position,
            orders: &mut self.orders,
            adverse_selection: &mut self.tier1.adverse_selection,
            depth_decay_as: &mut self.tier1.depth_decay_as,
            queue_tracker: &mut self.tier1.queue_tracker,
            pre_fill_classifier: &mut self.tier1.pre_fill_classifier,
            enhanced_classifier: &mut self.tier1.enhanced_classifier,
            estimator: &mut self.estimator,
            pnl_tracker: &mut self.tier2.pnl_tracker,
            prometheus: &mut self.infra.prometheus,
            metrics: &self.infra.metrics,
            latest_mid: self.latest_mid,
            asset: &self.config.asset,
            max_position: self.effective_max_position, // First-principles limit
            calibrate_depth_as: self.stochastic.stochastic_config.calibrate_depth_as,
            learning: &mut self.learning,
            stochastic_controller: &mut self.stochastic.controller,
            position_pnl: &mut self.stochastic.position_pnl,
            fee_bps: self.config.fee_bps,
            theoretical_edge: &mut self.stochastic.theoretical_edge,
            model_calibration: &mut self.stochastic.model_calibration,
            signal_store: &mut self.safety.signal_store,
            market_params: self.cached_market_params.as_ref(),
        };

        let result = messages::process_user_fills(
            &user_fills,
            &ctx,
            &mut self.safety.fill_processor,
            &mut fill_state,
        )?;

        // Record own fill timestamp for liquidation self-detection.
        // If position later jumps without a recent fill, it may be a liquidation.
        if result.fills_processed > 0 {
            self.safety.kill_switch.record_own_fill();

            // WS4: sigma_cascade_hwm REMOVED — CovarianceTracker handles realized vol
            // feedback via Bayesian posterior. Hawkes intensity is still tracked as a
            // feature for CalibratedRiskModel but doesn't directly multiply σ.

            // Also update proactive AS floor HWM (same pattern)
            let intensity_ratio = self.tier2.hawkes.intensity_ratio().max(1.0);
            let proactive_as = self.tier1.adverse_selection.as_floor_bps() * intensity_ratio;
            self.as_floor_hwm = self.as_floor_hwm.max(proactive_as);
        }

        // Process fills through WsOrderStateManager for additional state tracking
        // This provides secondary deduplication and state consistency
        for fill in &user_fills.data.fills {
            if fill.coin != *self.config.asset {
                continue;
            }

            // Record fill side in cascade tracker for same-side run detection.
            let is_buy = fill.side == "B" || fill.side.to_lowercase() == "buy";
            let fill_side = if is_buy { Side::Buy } else { Side::Sell };
            let current_pos = self.position.position();
            let fill_size: f64 = fill.sz.parse().unwrap_or(0.0);
            if let Some(cascade_event) =
                self.fill_cascade_tracker
                    .record_fill(fill_side, current_pos, fill_size)
            {
                // Phase 2C: Graduated cascade response.
                // Moderate cascade (λ_ratio < threshold): gamma boost only via beta_cascade.
                // Extreme cascade (λ_ratio ≥ threshold): preserve side-cancel as circuit breaker.
                let cascade_cancel_threshold = self.config.stochastic.cascade_cancel_threshold;
                let current_ratio = self.fill_cascade_tracker.max_intensity_ratio();

                if current_ratio >= cascade_cancel_threshold {
                    // Extreme cascade — preserve side-cancel as emergency circuit breaker
                    let cancel_side = match cascade_event {
                        CascadeEvent::Widen(s) | CascadeEvent::Suppress(s) => s,
                    };
                    let side_orders = self.orders.get_all_by_side(cancel_side);
                    let oids: Vec<u64> = side_orders.iter().map(|o| o.oid).collect();
                    if !oids.is_empty() {
                        warn!(
                            side = ?cancel_side,
                            count = oids.len(),
                            event = ?cascade_event,
                            intensity_ratio = %format!("{:.2}", current_ratio),
                            threshold = %format!("{:.2}", cascade_cancel_threshold),
                            "EXTREME cascade: side-cancel circuit breaker"
                        );
                        self.environment
                            .cancel_bulk_orders(&self.config.asset, oids)
                            .await;
                    }
                } else {
                    // Moderate cascade — gamma boost only (through beta_cascade in risk model)
                    let cancel_side = match cascade_event {
                        CascadeEvent::Widen(s) | CascadeEvent::Suppress(s) => s,
                    };
                    debug!(
                        side = ?cancel_side,
                        event = ?cascade_event,
                        intensity_ratio = %format!("{:.2}", current_ratio),
                        threshold = %format!("{:.2}", cascade_cancel_threshold),
                        "Moderate cascade: gamma boost only (no side-cancel)"
                    );
                }
                // Boost drift estimator responsiveness on cascade events.
                // 3x process noise → Kalman tracks rapid moves within 2-3 updates.
                self.drift_estimator.boost_responsiveness(3.0);
            }

            // Check for fill burst (fast cascade reaction — 3 fills in 5s).
            // Activates sigma boost (2x) for 30s, doubling spreads during sweeps.
            if self.fill_cascade_tracker.check_burst(fill_side) {
                // Burst detected — boost drift estimator too for fast reaction
                self.drift_estimator.boost_responsiveness(2.0);
            }

            // Count-based burst detection: 4+ same-side fills in 2s → emergency cancel.
            // Parallel to Hawkes-based cascade detection — catches sustained-level bursts
            // where intensity stays elevated but doesn't newly escalate.
            if self.fill_cascade_tracker.check_count_burst(fill_side) {
                let side_orders = self.orders.get_all_by_side(fill_side);
                let oids: Vec<u64> = side_orders.iter().map(|o| o.oid).collect();
                if !oids.is_empty() {
                    warn!(
                        side = ?fill_side,
                        count = oids.len(),
                        burst_count = self.fill_cascade_tracker.burst_count_threshold(),
                        window_secs = self.fill_cascade_tracker.burst_count_window_secs(),
                        "BURST CANCEL: count-based burst — emergency side-cancel"
                    );
                    self.environment
                        .cancel_bulk_orders(&self.config.asset, oids)
                        .await;
                }
            }

            // WS4: Feed cascade events to BayesianHawkes for posterior updating.
            // Magnitude = 1.0 per fill; accumulating fills are cascade-relevant.
            let is_accumulating = (is_buy && current_pos > 0.0) || (!is_buy && current_pos < 0.0);
            if is_accumulating {
                self.bayesian_hawkes
                    .observe_cascade(1.0, std::time::Instant::now());
            }

            // WS2: Track reducing-fill holding durations for τ_inventory EWMA.
            let is_reducing = (is_buy && current_pos < 0.0) || (!is_buy && current_pos > 0.0);
            if is_reducing {
                let now = std::time::Instant::now();
                if let Some(last_time) = self.last_reducing_fill_time {
                    let holding_s = now.duration_since(last_time).as_secs_f64();
                    // Only count reasonable durations (1s - 600s)
                    if (1.0..=600.0).contains(&holding_s) {
                        let tau_decay = 0.95;
                        let old_mean = self.tau_inventory_ewma_s;
                        self.tau_inventory_ewma_s =
                            tau_decay * self.tau_inventory_ewma_s + (1.0 - tau_decay) * holding_s;
                        // Welford online variance
                        let delta = holding_s - old_mean;
                        let delta2 = holding_s - self.tau_inventory_ewma_s;
                        self.tau_inventory_variance_s2 = tau_decay * self.tau_inventory_variance_s2
                            + (1.0 - tau_decay) * delta * delta2;
                    }
                }
                self.last_reducing_fill_time = Some(now);
            }

            let fill_price = fill.px.parse().unwrap_or(0.0);
            let fill_event = WsFillEvent {
                oid: fill.oid,
                tid: fill.tid,
                size: fill_size,
                price: fill_price,
                is_buy,
                coin: fill.coin.clone(),
                cloid: fill.cloid.clone(),
                timestamp: fill.time,
            };

            // Feed fill to Kalman drift estimator: bid fill → bearish, ask fill → bullish
            let sigma = self.estimator.sigma_effective();
            self.drift_estimator
                .update_fill(is_buy, fill_price, self.latest_mid, sigma);

            // Fix 13: Update EchoEstimator with current trend magnitude on own fill
            let position_value = (self.position.position().abs() * self.latest_mid).max(1.0);
            let trend_signal = self.estimator.trend_signal(position_value);
            self.echo_estimator
                .update_on_fill(trend_signal.long_momentum_bps.abs());

            // GM fill update on microprice V̂: adversarial evidence shifts fair value
            // Ask fill (they bought from us) → V̂ shifts upward
            // Bid fill (they sold to us) → V̂ shifts downward
            let is_ask_fill = !is_buy; // Our ask was hit = we sold = they bought
            let fill_now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            // Use dynamic p_informed from InformedFlowEstimator (adaptive, not hardcoded)
            let p_informed = self.estimator.p_informed();
            self.estimator.update_microprice_from_fill(
                fill_price,
                self.latest_mid,
                is_ask_fill,
                p_informed,
                fill_now_ms,
            );

            // Note: ws_state.handle_fill would update position, but position is already
            // updated by the main fill processor. We call it with a no-op tracker.
            // The main benefit is order state tracking and dedup validation.
            if let Some(ws_order) = self.ws_state.get_order(fill.oid) {
                // Just record the fill for state tracking (already processed above)
                if !ws_order.fill_tids.contains(&fill.tid) {
                    if let Some(order) = self.ws_state.get_order_mut(fill.oid) {
                        order.record_fill_with_price(
                            fill_event.tid,
                            fill_event.size,
                            fill_event.price,
                        );
                    }
                }
            }
        }

        // Phase 4: Trigger reconciliation for unmatched fills
        if result.unmatched_fills > 0 {
            for _ in 0..result.unmatched_fills {
                self.infra.reconciler.on_unmatched_fill(0, 0.0); // OID/size unknown at this point
            }
        }

        // Post-processing: cleanup completed orders
        messages::cleanup_orders(&mut self.orders, &mut self.tier1.queue_tracker);

        // Record fill observations to the strategy for Bayesian learning
        // These observations update the fill probability model
        for obs in &result.fill_observations {
            self.strategy
                .record_fill_observation(obs.depth_bps, obs.filled);
        }
        if !result.fill_observations.is_empty() {
            debug!(
                observations = result.fill_observations.len(),
                "Recorded fill observations for Bayesian learning"
            );
        }

        // Update adaptive Bayesian spread calculator with fill data
        // This updates: learned floor (AS), blended kappa (fill rate), shrinkage gamma (PnL)
        // ALWAYS feed fills to adaptive system for learning, even if not using adaptive quotes.
        // This ensures the system can learn from actual market behavior and be ready when enabled.
        {
            for fill in &user_fills.data.fills {
                if fill.coin != *self.config.asset {
                    continue;
                }

                let fill_price: f64 = fill.px.parse().unwrap_or(0.0);
                let is_buy = fill.side == "B" || fill.side.to_lowercase() == "buy";

                // Look up the order's placement mid from TrackedOrder.
                // This is the mid price when the order was originally placed,
                // NOT the mid at fill time. Using fill-time mid causes the
                // tautology bug where AS ≈ depth ≈ 0 (same reference point).
                let mid_at_placement = self
                    .orders
                    .get_order(fill.oid)
                    .and_then(|tracked| {
                        if tracked.mid_at_placement > 0.0 {
                            Some(tracked.mid_at_placement)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(self.latest_mid); // Fallback for untracked orders

                // Quoted spread: distance from placement mid to fill price (in bps)
                let quoted_spread_bps = if mid_at_placement > 0.0 {
                    ((fill_price - mid_at_placement).abs() / mid_at_placement) * 10_000.0
                } else {
                    0.0
                };

                // Compute instant AS using mid at order placement time.
                // Previously used self.latest_mid (fill-time mid), which made AS ≈ depth
                // (tautological since both reference the same mid). Using mid_at_placement
                // measures how much the fill price differs from where we THOUGHT mid was.
                let direction = if is_buy { 1.0 } else { -1.0 };
                let as_realized = (mid_at_placement - fill_price) * direction / fill_price;

                // Compute depth from placement mid (used by bandit reward + competitor model)
                let depth_from_mid = if mid_at_placement > 0.0 {
                    (fill_price - mid_at_placement).abs() / mid_at_placement
                } else {
                    (fill_price - self.latest_mid).abs() / self.latest_mid
                };

                // Phase 8: on_fill_simple moved to markout time in check_pending_fill_outcomes()
                // where we have true markout-based AS instead of tautological fill-time AS.

                // Record fill for calibration controller
                // This updates fill rate tracking for fill-hungry mode
                self.stochastic.calibration_controller.record_fill();

                // === Kappa Learning: Feed own fills to parameter estimator ===
                // This teaches the kappa estimator from our own fill intensity,
                // teaching the kappa estimator from our own fills.
                let fill_size: f64 = fill.sz.parse().unwrap_or(0.0);
                // Phase 0A: Use mid_at_placement for kappa learning.
                // Previously passed fill_price as both placement_price and fill_price,
                // producing distance=0 → kappa inflating from 1500→3000+ after 16 fills.
                // Now: orchestrator path uses mid-at-placement (depth ≈ half-spread).
                self.estimator.on_own_fill(
                    fill.time,        // timestamp_ms
                    mid_at_placement, // placement_price (mid when order was placed)
                    fill_price,       // fill_price (actual execution price)
                    fill_size,
                    is_buy,
                );

                // === AS Outcome Queue: Record fill for 5-second markout ===
                // After 5 seconds we check whether mid moved against us (adverse selection).
                // Drained in handle_all_mids -> check_pending_fill_outcomes().
                self.infra.pending_fill_outcomes.push_back(
                    crate::market_maker::fills::PendingFillOutcome {
                        timestamp_ms: fill.time,
                        oid: fill.oid,
                        fill_price,
                        is_buy,
                        mid_at_fill: self.latest_mid,
                        mid_at_placement,
                        quoted_spread_bps,
                        predicted_as_bps: self.estimator.total_as_bps(),
                    },
                );

                // === ModelGating Updates ===
                // Feed fill data to SignalIntegrator's ModelGating tracker.
                // This updates kappa IR tracking and regime-conditioned kappa estimation.
                self.stochastic.signal_integrator.on_fill(
                    fill.time,
                    fill_price,
                    fill_size,
                    self.latest_mid,
                );

                // Update adverse selection prediction tracking in ModelGating.
                // predicted_as_prob comes from the Bayesian edge estimator;
                // was_adverse is true when realized AS exceeds the 3 bps threshold.
                let predicted_as_prob = self.stochastic.theoretical_edge.bayesian_adverse();
                let was_adverse = as_realized * 10000.0 > 3.0;
                self.stochastic
                    .signal_integrator
                    .update_as_prediction(predicted_as_prob, was_adverse);

                // === EDGE TRACKING: Create Pending edge snapshot (deferred to markout) ===
                // Edge learning is deferred to the 5-second markout path where we get
                // the true markout-based AS. At fill time we only have instant AS which
                // suffers from the tautology bug (AS ≈ depth when using fill-time mid).
                // The Resolved snapshot is created in check_pending_fill_outcomes().
                {
                    use crate::market_maker::analytics::edge_metrics::EdgePhase;
                    use crate::market_maker::analytics::EdgeSnapshot;
                    const MAKER_FEE_BPS: f64 = 1.5;
                    let predicted_as_bps = self.estimator.total_as_bps();
                    let snap = EdgeSnapshot {
                        timestamp_ns: fill.time * 1_000_000, // ms to ns
                        predicted_spread_bps: quoted_spread_bps,
                        realized_spread_bps: quoted_spread_bps,
                        predicted_as_bps,
                        realized_as_bps: 0.0, // Deferred to markout
                        fee_bps: MAKER_FEE_BPS,
                        predicted_edge_bps: quoted_spread_bps - predicted_as_bps - MAKER_FEE_BPS,
                        realized_edge_bps: 0.0, // Deferred to markout
                        gross_edge_bps: 0.0,    // Deferred to markout
                        phase: EdgePhase::Pending,
                        mid_at_placement,
                        markout_as_bps: None,
                    };

                    // Log Pending snapshot for diagnostics only — NOT fed to learning yet
                    trace!(
                        quoted_spread_bps = %format!("{:.2}", quoted_spread_bps),
                        mid_at_placement = %format!("{:.4}", mid_at_placement),
                        predicted_as_bps = %format!("{:.2}", predicted_as_bps),
                        "[EDGE] Pending snapshot (deferred to markout)"
                    );

                    // Phase 5: Resolve pending quote as filled (immediate, uses quoted spread)
                    let _ = self
                        .quote_outcome_tracker
                        .on_fill(is_buy, snap.predicted_edge_bps);
                    let _ = snap; // Explicit drop — real snapshot created at markout
                }

                // === V2 INTEGRATION: BOCPD Kappa Relationship Update ===
                // Update BOCPD with observed features→kappa relationship + stress signals.
                // First-Principles Gap 1: Adaptive hazard rate based on market stress.
                if let Some(features) = self.stochastic.bocpd_kappa_features.take() {
                    let realized_kappa = self.estimator.kappa();

                    // Gather stress signals for adaptive hazard rate
                    let vpin = self.stochastic.vpin.vpin();
                    let hawkes_intensity_ratio = self.tier2.hawkes.intensity_ratio();
                    let size_anomaly_sigma = self.stochastic.trade_size_dist.anomaly_sigma();

                    // Wire Hawkes intensity into Kalman filter for activity-scaled Q
                    self.estimator
                        .set_hawkes_intensity_ratio(hawkes_intensity_ratio);

                    // Use stress-aware BOCPD update (First-Principles Gap 1)
                    let changepoint = self.stochastic.bocpd_kappa.update_with_stress(
                        &features,
                        realized_kappa,
                        vpin,
                        hawkes_intensity_ratio,
                        size_anomaly_sigma,
                    );

                    if changepoint {
                        let hazard = self.stochastic.bocpd_kappa.last_hazard_rate();
                        let stress = self.stochastic.bocpd_kappa.market_stress_index();
                        tracing::debug!(
                            realized_kappa = %format!("{:.0}", realized_kappa),
                            hazard = %format!("{:.4}", hazard),
                            stress = %format!("{:.2}", stress),
                            vpin = %format!("{:.2}", vpin),
                            "BOCPD: Changepoint detected (stress-aware)"
                        );
                    }
                }

                // DUAL-WRITE (Phase 3): Forward fill to centralized belief state
                let timestamp_ms = fill.time;
                // Determine if fill is aligned with current position direction
                let is_aligned = (is_buy && self.position.position() > 0.0)
                    || (!is_buy && self.position.position() < 0.0);
                // Realized edge = -realized_as (opposite signs: AS loss = negative edge)
                let realized_edge_bps = -as_realized * 10000.0;
                self.central_beliefs.update(BeliefUpdate::OwnFill {
                    price: fill_price,
                    size: fill_size,
                    mid: self.latest_mid,
                    is_buy,
                    is_aligned,
                    realized_as_bps: as_realized * 10000.0,
                    realized_edge_bps,
                    timestamp_ms,
                    order_id: Some(fill.oid),
                    quoted_size: fill_size, // Best available; exact quoted size not tracked here
                });

                // === Phase 8: RL Agent Learning Update ===
                // Update Q-values from fill outcome
                {
                    use crate::market_maker::learning::MarketEvent;

                    // === Contextual Bandit Reward Update ===
                    // Realized edge = spread_captured - AS - fees (i.i.d. bandit reward)
                    const MAKER_FEE_BPS: f64 = 1.5;
                    let depth_bps = depth_from_mid * 10_000.0;
                    let as_bps = as_realized * 10_000.0;
                    let realized_edge_bps = depth_bps - as_bps - MAKER_FEE_BPS;
                    let was_adverse = as_bps > 3.0;

                    // Update baseline tracker (EWMA of realized edge)
                    self.stochastic.baseline_tracker.observe(realized_edge_bps);

                    // Baseline-adjusted reward: removes fee drag (~-1.5 bps)
                    let bandit_reward = self
                        .stochastic
                        .baseline_tracker
                        .counterfactual_reward(realized_edge_bps);

                    // Update bandit with 1:1 quote→outcome reward attribution
                    let bandit_updated = self
                        .stochastic
                        .spread_bandit
                        .update_from_pending(bandit_reward);

                    // Update ensemble performance for SpreadBandit (enables GLFT vs Bandit comparison)
                    if bandit_updated {
                        let bandit_brier = bandit_reward * bandit_reward;
                        self.stochastic.ensemble.update_performance(
                            "SpreadBandit",
                            realized_edge_bps,
                            bandit_brier,
                            fill.time,
                        );
                    }

                    debug!(
                        realized_edge_bps = %format!("{:.2}", realized_edge_bps),
                        baseline_bps = %format!("{:.2}", self.stochastic.baseline_tracker.baseline()),
                        bandit_reward = %format!("{:.2}", bandit_reward),
                        bandit_updated = bandit_updated,
                        was_adverse = was_adverse,
                        bandit_obs = self.stochastic.spread_bandit.total_observations(),
                        "Bandit updated from fill"
                    );

                    // === Competitor Model Observation ===
                    // Estimate queue position from depth
                    let queue_position = depth_from_mid * 100.0; // Simplified: bps * 100
                    let time_in_queue_ms = 100; // Placeholder: would need order timestamp tracking

                    self.stochastic
                        .competitor_model
                        .observe(&MarketEvent::OurFill {
                            queue_position,
                            time_in_queue_ms,
                        });
                }
            }
        }

        // Phase 0C: Update position guard after fill processing.
        // Without this, inventory_skew_bps() returns 0.0 forever.
        self.safety
            .position_guard
            .update_position(self.position.position());

        // Phase 3B: Update direction hysteresis for zero-crossing detection.
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.safety
            .direction_hysteresis
            .update_position(self.position.position(), now_ms);

        // Record fill volume for rate limit budget calculation
        if result.total_volume_usd > 0.0 {
            self.infra
                .proactive_rate_tracker
                .record_fill_volume(result.total_volume_usd);

            // Earn execution budget tokens from fills (impulse control)
            if self.infra.impulse_control_enabled {
                self.infra.execution_budget.on_fill(result.total_volume_usd);
                debug!(
                    volume_usd = %format!("{:.2}", result.total_volume_usd),
                    budget_available = %format!("{:.1}", self.infra.execution_budget.available()),
                    "Earned execution budget tokens from fill"
                );
            }

            debug!(
                volume_usd = %format!("{:.2}", result.total_volume_usd),
                "Recorded fill volume for rate limit budget"
            );
        }

        // === Phase 2/3 Component Updates ===
        // TODO: Once fill_tracker, drawdown_tracker, and alerter are added to tiers:
        //
        // Update execution tracking
        // for fill in &user_fills.data.fills {
        //     if fill.coin != *self.config.asset { continue; }
        //     let record = FillRecord::new(
        //         fill.time,
        //         Side::from_is_buy(fill.side == "B"),
        //         fill.px.parse().unwrap_or(0.0),
        //         fill.sz.parse().unwrap_or(0.0),
        //         0, // queue_position - needs integration with queue_tracker
        //         0.0, // latency_ms - needs order placement timestamp
        //     );
        //     self.infra.fill_tracker.record_fill(record);
        // }
        //
        // Update drawdown tracker with new equity
        // let current_equity = self.infra.margin_sizer.account_value();
        // self.safety.drawdown_tracker.update_equity(current_equity);
        //
        // Check for fill rate alerts
        // let fill_metrics = self.infra.fill_tracker.metrics();
        // if let Some(alert) = self.infra.alerter.check_fill_rate(
        //     fill_metrics.fill_rate,
        //     self.stochastic.calibration_controller.target_fill_rate(),
        //     std::time::SystemTime::now()
        //         .duration_since(std::time::UNIX_EPOCH)
        //         .unwrap()
        //         .as_millis() as u64,
        // ) {
        //     self.infra.alerter.add_alert(alert);
        // }
        trace!(
            total_volume_usd = %format!("{:.2}", result.total_volume_usd),
            "Fill processed (Phase 2/3 components not yet wired)"
        );

        // Margin refresh on fills
        const MARGIN_REFRESH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);
        if self.infra.last_margin_refresh.elapsed() > MARGIN_REFRESH_INTERVAL {
            if let Err(e) = self.refresh_margin_state().await {
                warn!(error = %e, "Failed to refresh margin after fill");
            }
            self.infra.last_margin_refresh = std::time::Instant::now();
        }

        // === Push fill events to WebSocket dashboard clients ===
        if let Some(ref ws) = self.infra.dashboard_ws {
            let cum_pnl = self.tier2.pnl_tracker.summary(self.latest_mid).total_pnl;
            for fill in &user_fills.data.fills {
                if fill.coin != *self.config.asset {
                    continue;
                }
                let is_buy = fill.side == "B" || fill.side.to_lowercase() == "buy";
                let time_label = {
                    let ts_secs = fill.time / 1000;
                    let hrs = (ts_secs / 3600) % 24;
                    let mins = (ts_secs / 60) % 60;
                    format!("{:02}:{:02}", hrs, mins)
                };
                let fill_pnl = fill.closed_pnl.parse::<f64>().unwrap_or(0.0);
                let record = super::super::infra::metrics::dashboard::FillRecord {
                    time: time_label,
                    timestamp_ms: fill.time as i64,
                    pnl: fill_pnl,
                    cum_pnl,
                    side: if is_buy {
                        "BID".to_string()
                    } else {
                        "ASK".to_string()
                    },
                    adverse_selection: format!(
                        "{:.1}",
                        self.tier1.adverse_selection.realized_as_bps()
                    ),
                };
                let _ = ws
                    .sender()
                    .send(super::super::infra::dashboard_ws::DashboardPush::Fill { record });

                // Accumulate per-regime PnL
                if fill_pnl != 0.0 {
                    let regime_label = if let Some(mp) = self.cached_market_params.as_ref() {
                        let cascade_sev = if mp.should_pull_quotes {
                            1.0
                        } else {
                            (mp.tail_risk_intensity - 1.0) / 4.0
                        };
                        super::super::infra::metrics::dashboard::classify_regime(
                            cascade_sev,
                            mp.jump_ratio,
                            mp.sigma,
                        )
                    } else {
                        "Quiet".to_string()
                    };
                    self.infra
                        .prometheus
                        .dashboard()
                        .record_regime_pnl(&regime_label, fill_pnl);
                }
            }
        }

        // === Phase 3: Event-Driven Quote Updates ===
        // Instead of calling update_quotes() on every fill, record fill events
        // to the accumulator. Fills are high-priority and typically trigger immediately.
        if result.should_update_quotes {
            if self.event_accumulator.is_enabled() {
                // Record fill events for each processed fill
                for fill in &user_fills.data.fills {
                    if fill.coin != *self.config.asset {
                        continue;
                    }
                    let side = if fill.side == "B" || fill.side.to_lowercase() == "buy" {
                        Side::Buy
                    } else {
                        Side::Sell
                    };
                    let size: f64 = fill.sz.parse().unwrap_or(0.0);
                    // Track if this was a full fill (remaining size = 0)
                    // Check if order is now fully filled by looking at orders tracker
                    // Check if this was a full fill (remaining size = 0)
                    // remaining = size - filled; if order not found, assume full fill
                    let is_full_fill = self
                        .orders
                        .get_order(fill.oid)
                        .is_none_or(|o| o.size - o.filled <= 0.0);
                    let _trigger =
                        self.event_accumulator
                            .on_fill(side, fill.oid, size, is_full_fill);

                    // === Position Continuation Model: Update posterior on fill ===
                    // Track whether fill is aligned with current position direction
                    // Aligned fills increase P(continuation), adverse fills decrease it
                    let fill_side_sign = if side == Side::Buy { 1.0 } else { -1.0 };
                    let position_sign = self.position.position().signum();
                    self.stochastic.position_decision.observe_fill(
                        fill_side_sign,
                        position_sign,
                        size,
                    );
                }
                // Note: actual reconciliation happens in event loop via check_event_accumulator()
                Ok(())
            } else {
                // Fallback: event-driven mode disabled, use original timed polling
                self.update_quotes().await
            }
        } else {
            Ok(())
        }
    }

    /// Handle L2Book message - updates order book metrics.
    fn handle_l2_book(&mut self, l2_book: crate::ws::message_types::L2Book) -> Result<()> {
        // HARMONIZED: Use effective_max_position for position utilization calculations
        let ctx = messages::MessageContext::new(
            self.config.asset.clone(),
            self.latest_mid,
            self.position.position(),
            self.effective_max_position, // First-principles limit
            self.estimator.is_warmed_up(),
            Arc::from(self.config.collateral.symbol.as_str()),
        );

        let mut state = messages::L2BookState {
            estimator: &mut self.estimator,
            queue_tracker: &mut self.tier1.queue_tracker,
            spread_tracker: &mut self.tier2.spread_tracker,
            data_quality: &mut self.infra.data_quality,
            prometheus: &mut self.infra.prometheus,
            connection_supervisor: &self.infra.connection_supervisor,
        };

        let result = messages::process_l2_book(&l2_book, &ctx, &mut state)?;

        // Parse L2 data for dashboard and adaptive spreads
        // Recording is unconditional for dashboard; adaptive spreads only when valid
        if let Some((bids, asks)) = Self::parse_l2_for_adaptive(&l2_book.data.levels) {
            // Cache top 5 bid/ask sizes for EnhancedFlowContext
            // This enables depth imbalance calculation in the quote engine
            self.cached_bid_sizes = bids.iter().take(5).map(|(_, sz)| *sz).collect();
            self.cached_ask_sizes = asks.iter().take(5).map(|(_, sz)| *sz).collect();

            // Cache BBO prices and timestamp for pre-placement crossing validation.
            // This prevents "Post only would have immediately matched" rejections
            // that cause one-sided exposure.
            if let Some(&(best_bid, _)) = bids.first() {
                self.cached_best_bid = best_bid;
            }
            if let Some(&(best_ask, _)) = asks.first() {
                self.cached_best_ask = best_ask;
            }
            self.last_l2_update_time = std::time::Instant::now();

            // Update BBO in data quality monitor for quote gating crossed-book check.
            if let (Some(&(best_bid, _)), Some(&(best_ask, _))) = (bids.first(), asks.first()) {
                self.infra
                    .data_quality
                    .update_bbo(&self.config.asset, best_bid, best_ask);

                // Feed L2 snapshot to Shadow Tuner buffer (lock-free, non-blocking)
                let bid_depth: f64 = bids.iter().take(5).map(|(_, sz)| sz).sum();
                let ask_depth: f64 = asks.iter().take(5).map(|(_, sz)| sz).sum();
                let timestamp_ns = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0);
                self.feed_shadow_buffer(crate::market_maker::simulation::ReplayEvent::L2Update {
                    timestamp_ns,
                    best_bid,
                    best_ask,
                    bid_depth,
                    ask_depth,
                });
            }

            // Record book snapshot for dashboard unconditionally
            // This ensures dashboard shows data even during warmup
            self.infra
                .prometheus
                .record_book_for_dashboard(&bids, &asks);

            // Feed L2 data to adaptive spread calculator only when result is valid
            // This enables book-based kappa estimation to work alongside own-fill kappa
            if result.is_valid && self.latest_mid > 0.0 {
                self.stochastic
                    .adaptive_spreads
                    .on_l2_update(&bids, &asks, self.latest_mid);
            }

            // Update queue position tracker with full L2 depth snapshot.
            // This refines depth_ahead estimates using min(current, book) to
            // preserve queue advancement from observed fills/cancellations.
            self.tier1
                .queue_tracker
                .update_depth_from_book(&bids, &asks);

            // === Bootstrap from Book: Feed L2 to MarketProfile ===
            // MarketProfile tracks BBO spread, depth, and trade rate for kappa estimation.
            // When bootstrap_from_book is enabled AND coordinator not yet seeded,
            // the profile seeds the CalibrationCoordinator with conservative L2-derived kappa.
            if let (Some(&(best_bid, _)), Some(&(best_ask, _))) = (bids.first(), asks.first()) {
                let bid_depth_50bps: f64 = bids.iter().map(|(_, sz)| sz).sum();
                let ask_depth_50bps: f64 = asks.iter().map(|(_, sz)| sz).sum();
                self.market_profile.on_l2_book(
                    best_bid,
                    best_ask,
                    bid_depth_50bps,
                    ask_depth_50bps,
                );

                // Seed coordinator once when profile is ready and bootstrap is enabled
                if self.capital_policy.bootstrap_from_book
                    && !self.calibration_coordinator.is_seeded()
                    && self.market_profile.is_initialized()
                {
                    self.calibration_coordinator
                        .initialize_from_profile(&self.market_profile);
                    info!(
                        profile_kappa = %format!("{:.0}", self.market_profile.conservative_kappa()),
                        profile_sigma = %format!("{:.6}", self.market_profile.implied_sigma()),
                        "CalibrationCoordinator seeded from L2 book profile"
                    );
                }
            }

            // === Phase 7: Book Dynamics + Sweep Detector ===
            // Feed depth to BookDynamicsTracker for thinning, ΔBIM, BPG.
            {
                let bid_depth_shallow: f64 = bids.iter().take(3).map(|(_, sz)| sz).sum();
                let ask_depth_shallow: f64 = asks.iter().take(3).map(|(_, sz)| sz).sum();
                let bid_depth_deep: f64 = bids.iter().map(|(_, sz)| sz).sum();
                let ask_depth_deep: f64 = asks.iter().map(|(_, sz)| sz).sum();
                self.tier1.book_dynamics.update(
                    bid_depth_deep,
                    ask_depth_deep,
                    std::time::Instant::now(),
                );
                self.tier1.book_dynamics.update_imbalances(
                    bid_depth_shallow,
                    ask_depth_shallow,
                    bid_depth_deep,
                    ask_depth_deep,
                );
            }

            // === Self-Impact Estimator: Feed book depth + our resting sizes ===
            // L2 depth includes our own orders, so subtract to get other-participant depth.
            {
                let (our_bid_size, our_ask_size) = self.orders.pending_exposure();
                let total_bid_depth: f64 = bids.iter().map(|(_, sz)| sz).sum();
                let total_ask_depth: f64 = asks.iter().map(|(_, sz)| sz).sum();
                let other_bid_depth = (total_bid_depth - our_bid_size).max(0.0);
                let other_ask_depth = (total_ask_depth - our_ask_size).max(0.0);
                self.self_impact.update(
                    our_bid_size,
                    other_bid_depth,
                    our_ask_size,
                    other_ask_depth,
                );
            }

            // === Phase 3: Pre-Fill AS Classifier - Orderbook Imbalance Update ===
            // Feed bid/ask depth to pre-fill classifier for toxicity prediction
            {
                let bid_depth: f64 = bids.iter().take(5).map(|(_, sz)| sz).sum();
                let ask_depth: f64 = asks.iter().take(5).map(|(_, sz)| sz).sum();
                self.tier1
                    .pre_fill_classifier
                    .update_orderbook(bid_depth, ask_depth);

                // === Enhanced Microstructure Classifier: Book Update ===
                // Feed BBO and depth to z-score normalized feature extractor
                if let (Some(&(best_bid, _)), Some(&(best_ask, _))) = (bids.first(), asks.first()) {
                    let timestamp_ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0);
                    self.tier1.enhanced_classifier.on_book_update(
                        best_bid,
                        best_ask,
                        bid_depth,
                        ask_depth,
                        timestamp_ms,
                    );
                }
            }

            // === Phase 1: Microstructure Signals - Liquidity Evaporation ===
            // Feed near-touch depth to evaporation detector
            // Use top 3 levels on each side as "near touch"
            {
                let near_bid_depth: f64 = bids.iter().take(3).map(|(_, sz)| sz).sum();
                let near_ask_depth: f64 = asks.iter().take(3).map(|(_, sz)| sz).sum();
                let near_touch_depth = near_bid_depth + near_ask_depth;
                let timestamp_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                self.stochastic
                    .liquidity_evaporation
                    .on_book(near_touch_depth, timestamp_ms);

                // Phase 1A: Update COFI (Cumulative OFI with decay)
                // We use top-of-book depth as the OFI proxy
                // Delta = current depth change (we use depth directly since COFI handles decay)
                // For proper COFI, we'd track previous depths, but this approximation works
                // by treating current depth as a "flow" signal
                let bid_delta = near_bid_depth * 0.1; // Scale factor to normalize
                let ask_delta = near_ask_depth * 0.1;
                self.stochastic
                    .cofi
                    .on_book_update(bid_delta, ask_delta, timestamp_ms);
            }

            // === Phase 8: Competitor Model Depth Observation ===
            // Track depth changes for competitor inference
            {
                use crate::market_maker::learning::{MarketEvent, Side as LearningSide};

                // Compute total bid/ask depth
                let bid_depth: f64 = bids.iter().map(|(_, sz)| sz).sum();
                let ask_depth: f64 = asks.iter().map(|(_, sz)| sz).sum();

                // Store previous depths for delta calculation (simplified: use static mut or cache)
                // For now, observe raw depth as proxy for competitor activity
                // Large depths suggest more competitor presence
                if bid_depth > 100.0 {
                    self.stochastic
                        .competitor_model
                        .observe(&MarketEvent::DepthChange {
                            side: LearningSide::Bid,
                            delta: bid_depth,
                        });
                }
                if ask_depth > 100.0 {
                    self.stochastic
                        .competitor_model
                        .observe(&MarketEvent::DepthChange {
                            side: LearningSide::Ask,
                            delta: ask_depth,
                        });
                }
            }

            // === Phase 2/3 Component Updates ===
            // TODO: Once circuit_breaker and dashboard are added to tiers:
            //
            // Calculate current spread for circuit breaker
            // let spread_bps = if !bids.is_empty() && !asks.is_empty() {
            //     let best_bid = bids.first().map(|(p, _)| *p).unwrap_or(0.0);
            //     let best_ask = asks.first().map(|(p, _)| *p).unwrap_or(0.0);
            //     if best_bid > 0.0 {
            //         ((best_ask - best_bid) / best_bid) * 10_000.0
            //     } else {
            //         0.0
            //     }
            // } else {
            //     0.0
            // };
            //
            // Update circuit breaker with spread
            // if let Some(cb_type) = self.tier1.circuit_breaker.check_spread(spread_bps) {
            //     warn!(breaker = %cb_type, spread_bps = %format!("{:.1}", spread_bps),
            //           "Circuit breaker triggered by spread blowout");
            // }
            //
            // Update monitoring dashboard
            // let regime = self.stochastic.regime_hmm.current_regime();
            // let confidence = self.stochastic.regime_hmm.max_belief();
            // self.infra.dashboard.update_market(regime, confidence, spread_bps, self.latest_mid);
            trace!(
                best_bid = ?bids.first().map(|(p, _)| *p),
                best_ask = ?asks.first().map(|(p, _)| *p),
                "L2 book processed (Phase 2/3 components not yet wired)"
            );
        }

        Ok(())
    }

    /// Parse L2 book levels into (bids, asks) tuples for adaptive spread calculator.
    /// Returns None if the levels are invalid or insufficient.
    #[allow(clippy::type_complexity)]
    pub(super) fn parse_l2_for_adaptive(
        levels: &[Vec<crate::types::OrderBookLevel>],
    ) -> Option<(Vec<(f64, f64)>, Vec<(f64, f64)>)> {
        if levels.len() < 2 {
            return None;
        }

        let bids: Vec<(f64, f64)> = levels[0]
            .iter()
            .filter_map(|level| {
                let px: f64 = level.px.parse().ok()?;
                let sz: f64 = level.sz.parse().ok()?;
                Some((px, sz))
            })
            .collect();

        let asks: Vec<(f64, f64)> = levels[1]
            .iter()
            .filter_map(|level| {
                let px: f64 = level.px.parse().ok()?;
                let sz: f64 = level.sz.parse().ok()?;
                Some((px, sz))
            })
            .collect();

        Some((bids, asks))
    }

    /// Handle OrderUpdates message - processes order state changes via WsOrderStateManager.
    fn handle_order_updates(
        &mut self,
        order_updates: crate::ws::message_types::OrderUpdates,
    ) -> Result<()> {
        for update in &order_updates.data {
            // Filter to our asset
            if update.order.coin != *self.config.asset {
                continue;
            }

            // Convert to WsOrderUpdateEvent format
            let event = WsOrderUpdateEvent {
                oid: update.order.oid,
                cloid: update.order.cloid.clone(),
                status: update.status.clone(),
                size: update.order.sz.parse().unwrap_or(0.0),
                orig_size: update.order.orig_sz.parse().unwrap_or(0.0),
                price: update.order.limit_px.parse().unwrap_or(0.0),
                coin: update.order.coin.clone(),
                is_buy: update.order.side == "B" || update.order.side.to_lowercase() == "buy",
                status_timestamp: update.status_timestamp,
            };

            // Process through WsOrderStateManager
            self.ws_state.handle_order_update(&event);

            // CRITICAL: Also sync state to OrderManager to keep both tracking systems in sync.
            // The safety_sync uses OrderManager for state comparison, so it must know about
            // filled/cancelled orders to avoid false stale/orphan detection.
            // Note: Using set_state() is appropriate here since we're receiving status from WS,
            // not initiating state transitions internally.
            #[allow(deprecated)]
            match event.status.as_str() {
                "filled" => {
                    if self.orders.set_state(event.oid, OrderState::Filled) {
                        info!(
                            oid = event.oid,
                            status = %event.status,
                            "Order state update: filled (synced to OrderManager)"
                        );
                    } else {
                        // Order not in OrderManager - could be from previous session
                        debug!(
                            oid = event.oid,
                            status = %event.status,
                            "Order filled but not tracked in OrderManager"
                        );
                    }
                }
                "canceled" => {
                    if self.orders.set_state(event.oid, OrderState::Cancelled) {
                        debug!(
                            oid = event.oid,
                            status = %event.status,
                            "Order state update: canceled (synced to OrderManager)"
                        );
                    } else {
                        debug!(
                            oid = event.oid,
                            status = %event.status,
                            "Order canceled but not tracked in OrderManager"
                        );
                    }
                }
                "open" => {
                    // Order is resting - no state change needed unless partially filled
                    // Size changes are handled by ws_state
                    debug!(
                        oid = event.oid,
                        status = %event.status,
                        "Order state update: open"
                    );
                }
                _ => {
                    debug!(
                        oid = event.oid,
                        status = %event.status,
                        "Order state update (unknown status)"
                    );
                }
            }
        }

        // Periodic cleanup of terminal orders in ws_state
        let removed = self.ws_state.cleanup();
        if !removed.is_empty() {
            debug!(
                count = removed.len(),
                "Cleaned up terminal orders from ws_state"
            );
        }

        Ok(())
    }

    /// Handle OpenOrders message - syncs WS order snapshot to local state.
    ///
    /// This subscription provides a reliable snapshot of all open orders directly from
    /// the WebSocket stream, avoiding the race conditions inherent in REST-based polling.
    /// The snapshot is used for:
    /// 1. Initial state synchronization on startup
    /// 2. Periodic state reconciliation (replaces REST-based safety_sync)
    fn handle_open_orders(
        &mut self,
        open_orders: crate::ws::message_types::OpenOrders,
    ) -> Result<()> {
        // Filter to our asset
        let our_orders: Vec<_> = open_orders
            .data
            .orders
            .iter()
            .filter(|o| o.coin == *self.config.asset)
            .collect();

        let ws_open_oids: std::collections::HashSet<u64> =
            our_orders.iter().map(|o| o.oid).collect();

        // Store the latest WS snapshot for use by safety_sync
        self.last_ws_open_orders_snapshot = Some(ws_open_oids.clone());
        self.last_ws_snapshot_time = Some(std::time::Instant::now());

        // === Full Synchronization: ws_state should mirror the snapshot ===
        // Step 1: Get current ws_state OIDs
        let current_ws_oids = self.ws_state.open_order_ids();

        // Step 2: Identify orders to REMOVE (in ws_state but NOT in snapshot)
        // IMPORTANT: Apply grace period for fresh orders to avoid race condition
        // between WS POST placement and openOrders snapshot delivery
        const GRACE_PERIOD_SECS: u64 = 2;
        let grace_period = std::time::Duration::from_secs(GRACE_PERIOD_SECS);

        let (stale_oids, skipped_fresh): (Vec<u64>, Vec<u64>) = current_ws_oids
            .iter()
            .filter(|oid| !ws_open_oids.contains(oid))
            .partition(|oid| {
                // Only mark as stale if order is older than grace period
                self.ws_state
                    .get_order(**oid)
                    .map(|o| o.placed_at.elapsed() > grace_period)
                    .unwrap_or(true) // If order not found, treat as stale
            });

        // Log fresh orders that were skipped
        if !skipped_fresh.is_empty() {
            debug!(
                count = skipped_fresh.len(),
                grace_period_secs = GRACE_PERIOD_SECS,
                "[OpenOrders] Skipping removal of fresh orders (age < grace period)"
            );
        }

        // Step 3: Identify orders to ADD (in snapshot but NOT in ws_state)
        let new_oids: Vec<_> = our_orders
            .iter()
            .filter(|o| !current_ws_oids.contains(&o.oid))
            .collect();

        // Log sync activity
        if !stale_oids.is_empty() || !new_oids.is_empty() {
            info!(
                snapshot_count = our_orders.len(),
                ws_state_count = current_ws_oids.len(),
                removed = stale_oids.len(),
                added = new_oids.len(),
                "[OpenOrders] Synchronizing ws_state with exchange snapshot"
            );
        } else {
            debug!(
                orders = our_orders.len(),
                "Received OpenOrders snapshot (ws_state already in sync)"
            );
        }

        // Step 4: Remove stale orders from ws_state
        let mut removed_count = 0usize;
        for oid in &stale_oids {
            if let Some(order) = self.ws_state.remove_order(*oid) {
                self.safety.fill_processor.record_cancelled_order(
                    *oid,
                    order.side,
                    order.price,
                    order.size,
                );
                // Record unfilled outcome for reconcile action-type learning
                self.reconcile_outcome_tracker.record_unfilled(*oid);
                removed_count += 1;
                debug!(
                    oid = oid,
                    "[OpenOrders] Removed order from ws_state (not in exchange snapshot)"
                );
            }
            // Also clear from orphan tracker if present
            self.infra.orphan_tracker.clear_orphan(*oid);
        }
        // Log summary at INFO level if any were removed
        if removed_count > 0 {
            info!(
                removed_count = removed_count,
                "[OpenOrders] Cleaned up stale orders from ws_state"
            );
        }

        // Step 5: Add new orders to ws_state
        for order in new_oids {
            let side = if order.side == "B" || order.side.to_lowercase() == "buy" {
                Side::Buy
            } else {
                Side::Sell
            };
            let tracked = TrackedOrder::with_cloid(
                order.oid,
                order.cloid.clone().unwrap_or_default(),
                side,
                order.limit_px.parse().unwrap_or(0.0),
                order.sz.parse().unwrap_or(0.0),
                self.latest_mid,
            );
            self.ws_state.add_order(tracked);
            debug!(
                oid = order.oid,
                "Added order from OpenOrders snapshot to ws_state"
            );
        }

        Ok(())
    }

    /// Handle ActiveAssetData message - real-time exchange limit updates.
    ///
    /// This replaces the periodic REST polling in `refresh_exchange_limits`.
    /// Updates position limits (max long/short) and available margin in real-time.
    fn handle_active_asset_data(
        &mut self,
        active_asset_data: crate::ws::message_types::ActiveAssetData,
    ) -> Result<()> {
        // Filter to our asset (should already match since we subscribed per-coin)
        if active_asset_data.data.coin != *self.config.asset {
            return Ok(());
        }

        // Update exchange limits using WebSocket data
        // Note: WS message lacks mark_px, so we use our tracked latest_mid
        self.infra.exchange_limits.update_from_ws(
            &active_asset_data.data,
            self.latest_mid,
            self.effective_max_position,
        );

        // Update staleness timestamp
        self.last_active_asset_data_time = Some(std::time::Instant::now());

        debug!(
            coin = %active_asset_data.data.coin,
            "Exchange limits updated from WebSocket"
        );

        Ok(())
    }

    /// Handle WebData2 message - real-time margin and position updates.
    ///
    /// This replaces the periodic REST polling in `refresh_margin_state` and `sync_position_from_exchange`.
    fn handle_web_data2(&mut self, web_data2: crate::ws::message_types::WebData2) -> Result<()> {
        // Note: web_data2.data does not contain 'user', so we assume it matches our subscription.

        let state = &web_data2.data.clearinghouse_state;

        // 1. Parse margin summary
        let account_value = state
            .margin_summary
            .account_value
            .parse::<f64>()
            .unwrap_or(0.0);
        let margin_used = state
            .margin_summary
            .total_margin_used
            .parse::<f64>()
            .unwrap_or(0.0);
        let total_notional = state
            .margin_summary
            .total_ntl_pos
            .parse::<f64>()
            .unwrap_or(0.0);

        // 2. Find position and liquidation price for our asset
        let (exchange_position, liquidation_price) = state
            .asset_positions
            .iter()
            .find(|p| p.position.coin == *self.config.asset)
            .map(|p| {
                (
                    p.position.szi.parse::<f64>().unwrap_or(0.0),
                    p.position
                        .liquidation_px
                        .as_ref()
                        .and_then(|px| px.parse::<f64>().ok()),
                )
            })
            .unwrap_or((0.0, None));

        // Update staleness timestamp
        self.last_web_data2_time = Some(std::time::Instant::now());

        // 3. Update local position tracking if relevant
        // IMPORTANT: For HIP-3 DEXs, WebData2's clearinghouseState is the USDC clearinghouse,
        // NOT the DEX's clearinghouse. The asset won't be found, resulting in exchange_position=0.
        // Only sync position from WebData2 for validator perps (non-HIP-3).
        if self.config.dex.is_none() {
            let local_position = self.position.position();
            let drift = (exchange_position - local_position).abs();

            if drift > crate::EPSILON {
                self.position.set_position(exchange_position);
                debug!(
                    local = local_position,
                    exchange = exchange_position,
                    "Position synced from WebData2"
                );
            }
        }

        // 4. Update MarginSizer state
        // IMPORTANT: For HIP-3 DEXs, account_value comes from spot balances (Phase 3).
        // WebData2's clearinghouseState reports the USDC clearinghouse only.
        // Unified margin accounts have collateral split across perps + spot —
        // always sum both to get the true account value.
        if self.config.dex.is_none() {
            // Unified account: perps clearinghouse + spot stablecoin balances
            let spot_collateral: f64 = ["USDC", "USDT", "USDE"]
                .iter()
                .filter_map(|coin| self.spot_balance_cache.get(*coin).copied())
                .sum();
            let effective_account_value = account_value + spot_collateral;

            self.infra.margin_sizer.update_state_with_liquidation(
                effective_account_value,
                margin_used,
                total_notional,
                liquidation_price,
                self.latest_mid,
                self.position.position(),
            );

            // Log occasionally
            debug!(
                account_value = effective_account_value,
                margin_used = margin_used,
                liquidation_price = ?liquidation_price,
                "Margin state updated from WebData2"
            );
        }

        Ok(())
    }

    /// Handle ledger updates (spot balance deltas).
    fn handle_ledger_update(
        &mut self,
        update: crate::ws::message_types::UserNonFundingLedgerUpdates,
    ) -> Result<()> {
        let data = update.data;
        if data.user != self.user_address {
            return Ok(());
        }

        use crate::types::LedgerUpdate;

        for item in data.non_funding_ledger_updates {
            match item.delta {
                LedgerUpdate::Deposit(d) => {
                    let amount = d.usdc.parse::<f64>().unwrap_or(0.0);
                    let entry = self
                        .spot_balance_cache
                        .entry("USDC".to_string())
                        .or_insert(0.0);
                    *entry += amount;
                    debug!(
                        coin = "USDC",
                        amount = amount,
                        total = *entry,
                        "Deposit processed"
                    );
                }
                LedgerUpdate::Withdraw(w) => {
                    let amount = w.usdc.parse::<f64>().unwrap_or(0.0);
                    let entry = self
                        .spot_balance_cache
                        .entry("USDC".to_string())
                        .or_insert(0.0);
                    *entry -= amount;
                    debug!(
                        coin = "USDC",
                        amount = amount,
                        total = *entry,
                        "Withdraw processed"
                    );
                }
                LedgerUpdate::SpotTransfer(t) => {
                    let token = t.token.clone();
                    let amount = t.amount.parse::<f64>().unwrap_or(0.0);

                    if t.destination == self.user_address {
                        let entry = self.spot_balance_cache.entry(token.clone()).or_insert(0.0);
                        *entry += amount;
                    } else {
                        let entry = self.spot_balance_cache.entry(token.clone()).or_insert(0.0);
                        *entry -= amount;
                    }
                    debug!(coin = token, amount = amount, "SpotTransfer processed");
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Handle user funding settlement events.
    ///
    /// Phase 6: Wire UserFundings data feed to FundingRateEstimator.
    /// This was previously dropped by the wildcard `_ => Ok(())` match arm.
    fn handle_user_fundings(
        &mut self,
        fundings: crate::ws::message_types::UserFundings,
    ) -> Result<()> {
        let data = fundings.data;
        if data.user != self.user_address {
            return Ok(());
        }

        for f in &data.fundings {
            let rate = f.funding_rate.parse::<f64>().unwrap_or(0.0);
            if rate.abs() < 1e-12 {
                continue;
            }

            self.tier2.funding.record_funding(rate, f.time);

            tracing::debug!(
                coin = %f.coin,
                rate = %format!("{:.6}", rate),
                usdc = %f.usdc,
                time = f.time,
                "Funding settlement recorded"
            );
        }

        Ok(())
    }

    /// Called periodically (e.g., every 1s) to update component states.
    ///
    /// This method performs periodic maintenance tasks that don't need to happen
    /// on every market data tick but should run regularly.
    ///
    /// # Calibration Logging (Phase 0)
    ///
    /// Updates model calibration metrics and logs Brier scores / IR for monitoring.
    /// This enables visibility into prediction quality without changing quoting logic.
    pub fn periodic_component_update(&mut self) {
        // === Calibration Metric Updates ===
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Update model calibration tracking
        self.stochastic.model_calibration.update_all(now);

        // Log calibration metrics periodically (every ~60 calls ≈ 1 minute at 1s interval)
        static PERIODIC_COUNTER: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let count = PERIODIC_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if count.is_multiple_of(60) {
            let summary = self.stochastic.model_calibration.summary();

            // Log calibration summary
            info!(
                fill_ir = %format!("{:.3}", summary.fill_ir),
                fill_samples = summary.fill_model.n_samples,
                fill_brier = %format!("{:.4}", summary.fill_model.brier_score),
                as_ir = %format!("{:.3}", summary.as_ir),
                as_samples = summary.as_model.n_samples,
                lag_ir = %format!("{:.3}", summary.lag_ir),
                lag_mi_decay = %format!("{:.5}/day", summary.lag_mi_decay_rate),
                any_degraded = summary.any_degraded,
                "Calibration metrics"
            );

            // Warn if any model is degraded
            if summary.any_degraded {
                let degraded = self.stochastic.model_calibration.degraded_models();
                warn!(
                    models = ?degraded,
                    "Model calibration degraded - IR < 1.0 or MI decaying"
                );
            }
        }

        // === Tombstone cleanup: expire cancelled-order tombstones older than TTL ===
        self.safety.fill_processor.cleanup_tombstones();

        // === Equity Curve Snapshot: Portfolio-level Sharpe tracking ===
        {
            let pnl_snap = self.tier2.pnl_tracker.summary(self.latest_mid);
            let total_equity_usd = pnl_snap.total_pnl;
            let now_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            self.live_analytics
                .snapshot_equity(now_ns, total_equity_usd);
        }

        // === Live Analytics: Periodic summary (Sharpe, signal attribution) ===
        let mean_edge = self.tier2.edge_tracker.mean_realized_edge();
        let logged_summary = self.live_analytics.maybe_log_summary(mean_edge);

        // === PnL Summary (only when analytics summary fires, to avoid log spam) ===
        if logged_summary {
            let pnl_summary = self.tier2.pnl_tracker.summary(self.latest_mid);

            // PnL decomposition: spread capture, AS, funding, fees, unrealized
            tracing::info!(
                realized = %format!("${:.2}", pnl_summary.realized_pnl),
                unrealized = %format!("${:.2}", pnl_summary.unrealized_pnl),
                total = %format!("${:.2}", pnl_summary.total_pnl),
                spread_capture = %format!("${:.2}", pnl_summary.spread_capture),
                adverse_selection = %format!("${:.2}", pnl_summary.adverse_selection),
                funding = %format!("${:.2}", pnl_summary.funding),
                fees = %format!("${:.2}", pnl_summary.fees),
                fills = pnl_summary.fill_count,
                "[PnL] session decomposition"
            );

            // Equity Sharpe alongside fill Sharpe (clearly labeled)
            let eq_summary = self.live_analytics.equity_summary();
            tracing::info!(
                fill_sharpe = %format!("{:.2}", self.live_analytics.sharpe_tracker().sharpe_ratio()),
                equity_sharpe = %format!("{:.2}", eq_summary.sharpe_all),
                equity_sharpe_1h = %format!("{:.2}", eq_summary.sharpe_1h),
                max_dd_bps = %format!("{:.1}", eq_summary.max_drawdown_bps),
                equity_snapshots = eq_summary.snapshot_count,
                "[ANALYTICS] fill vs equity Sharpe"
            );
        }

        // Log current state for debugging
        trace!(
            sigma = %format!("{:.6}", self.estimator.sigma()),
            kappa = %format!("{:.3}", self.estimator.kappa()),
            position = %format!("{:.4}", self.position.position()),
            "Periodic component update"
        );
    }

    /// Check event accumulator and trigger quote update if needed.
    ///
    /// # Phase 3: Event-Driven Churn Reduction
    ///
    /// This method checks if accumulated events should trigger a quote update.
    /// It replaces the previous approach where every AllMids message triggered
    /// a full reconciliation cycle.
    ///
    /// Events are only triggered when:
    /// - Mid price moves > threshold (default 5 bps)
    /// - A fill is received (high priority)
    /// - Queue depletion detected (P(fill) < 5%)
    /// - Signal changes significantly
    /// - Fallback timer expires (default 5s)
    ///
    /// This reduces order churn from ~94% to target 20-40% by only
    /// reconciling when meaningful changes have occurred.
    pub(crate) async fn check_event_accumulator(&mut self) -> crate::prelude::Result<()> {
        // Skip if event-driven mode is disabled
        if !self.event_accumulator.is_enabled() {
            return Ok(());
        }

        // Check event-driven triggers first, then fallback timer.
        // The fallback guarantees minimum quote frequency even without accumulated events.
        let trigger = self
            .event_accumulator
            .should_trigger()
            .or_else(|| self.event_accumulator.check_fallback());

        if let Some(trigger) = trigger {
            debug!(
                event = ?trigger.event,
                scope = ?trigger.scope,
                priority = trigger.priority(),
                "Event accumulator: triggering quote update"
            );

            // Call update_quotes to perform the reconciliation
            self.update_quotes().await?;

            // Reset the accumulator after successful reconciliation
            self.event_accumulator.reset();

            // Log stats periodically
            let stats = self.event_accumulator.stats();
            if stats.total_reconciles.is_multiple_of(100) {
                info!(
                    total_events = stats.total_events,
                    total_reconciles = stats.total_reconciles,
                    filter_ratio = %format!("{:.1}%", stats.filter_ratio() * 100.0),
                    reconcile_frequency = %format!("{:.3}", stats.reconcile_frequency()),
                    "Event accumulator stats (churn reduction)"
                );
            }
        }

        Ok(())
    }

    /// Handle Binance price update for lead-lag signal.
    ///
    /// This feeds Binance mid prices to the SignalIntegrator, which computes
    /// optimal skew based on cross-exchange price discovery (Binance leads Hyperliquid).
    ///
    /// Called from the event loop when Binance feed is enabled.
    pub(crate) fn handle_binance_price_update(&mut self, update: BinancePriceUpdate) {
        // Feed Binance price to SignalIntegrator
        self.stochastic
            .signal_integrator
            .on_binance_price(update.mid_price, update.timestamp_ms);

        // Feed BTC returns to CrossAssetSignals (Sprint 4.1)
        if self.last_binance_mid > 0.0 && update.mid_price > 0.0 {
            let return_bps = (update.mid_price / self.last_binance_mid - 1.0) * 10_000.0;
            self.cross_asset_signals
                .update_btc_return(update.timestamp_ms as u64, return_bps);
        }
        self.last_binance_mid = update.mid_price;

        // Log periodically (every 1000 updates ≈ every 10 seconds at 100 updates/sec)
        static BINANCE_UPDATE_COUNTER: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let count = BINANCE_UPDATE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if count.is_multiple_of(1000) {
            let signal = self.stochastic.signal_integrator.lead_lag_signal();
            if signal.is_actionable {
                info!(
                    binance_mid = %format!("{:.2}", update.mid_price),
                    hl_mid = %format!("{:.2}", self.latest_mid),
                    diff_bps = %format!("{:.1}", signal.diff_bps),
                    skew_direction = signal.skew_direction,
                    skew_bps = %format!("{:.1}", signal.skew_magnitude_bps),
                    "Lead-lag signal ACTIVE"
                );
            } else {
                trace!(
                    binance_mid = %format!("{:.2}", update.mid_price),
                    spread_bps = %format!("{:.2}", update.spread_bps),
                    "Binance price update"
                );
            }
        }
    }

    /// Handle Binance trade update for cross-venue flow analysis.
    ///
    /// This feeds Binance trades to the SignalIntegrator for bivariate flow analysis.
    /// The BinanceFlowAnalyzer computes VPIN, volume imbalance, and trade intensity
    /// which are combined with HL flow features for cross-venue agreement/divergence detection.
    ///
    /// Called from the event loop when Binance feed with trades is enabled.
    pub(crate) fn handle_binance_trade(&mut self, trade: BinanceTradeUpdate) {
        // Feed Binance trade to SignalIntegrator (cross-venue flow analysis)
        self.stochastic.signal_integrator.on_binance_trade(&trade);

        // Log periodically (every 500 trades)
        static BINANCE_TRADE_COUNTER: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let count = BINANCE_TRADE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if count.is_multiple_of(500) {
            let cv_features = self.stochastic.signal_integrator.cross_venue_features();
            if cv_features.sample_count >= 20 {
                trace!(
                    agreement = %format!("{:.2}", cv_features.agreement),
                    direction = %format!("{:.2}", cv_features.combined_direction),
                    max_toxicity = %format!("{:.2}", cv_features.max_toxicity),
                    intensity_ratio = %format!("{:.2}", cv_features.intensity_ratio),
                    "Cross-venue features"
                );
            }
        }
    }

    /// Handle a unified Binance update (price or trade).
    ///
    /// This is the main entry point for all Binance feed updates.
    /// Routes to the appropriate handler based on update type.
    pub(crate) fn handle_binance_update(&mut self, update: BinanceUpdate) {
        match update {
            BinanceUpdate::Price(price_update) => {
                self.handle_binance_price_update(price_update);
            }
            BinanceUpdate::Trade(trade_update) => {
                self.handle_binance_trade(trade_update);
            }
        }
    }
}
