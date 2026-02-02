//! WebSocket message handlers for the market maker.

use std::sync::Arc;

use tracing::{debug, info, trace, warn};

use crate::prelude::Result;
use crate::Message;

use super::super::{
    belief::BeliefUpdate,
    estimator::HmmObservation,
    fills, messages, tracking::ws_order_state::WsFillEvent,
    tracking::ws_order_state::WsOrderUpdateEvent, MarketMaker, OrderExecutor, OrderState,
    QuotingStrategy, Side, TrackedOrder,
};

impl<S: QuotingStrategy, E: OrderExecutor> MarketMaker<S, E> {
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
            _ => Ok(()),
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

            // === First-Principles Gap 2: Update ThresholdKappa with return observation ===
            // Feed return_bps to ThresholdKappa for TAR model regime detection.
            // This determines whether we're in mean-reversion or momentum regime.
            if prev_mid > 0.0 && self.latest_mid > 0.0 {
                let return_bps = (self.latest_mid - prev_mid) / prev_mid * 10_000.0;
                self.stochastic.threshold_kappa.update(return_bps);
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

            // Add to buffer
            self.cached_trades.push_back((size, is_buy, timestamp_ms));

            // Keep bounded
            while self.cached_trades.len() > MAX_CACHED_TRADES {
                self.cached_trades.pop_front();
            }

            // Phase 7: Forward market trade to centralized belief state (primary consumer)
            let trade_price: f64 = trade.px.parse().unwrap_or(0.0);
            if trade_price > 0.0 {
                self.central_beliefs.update(BeliefUpdate::MarketTrade {
                    price: trade_price,
                    mid: self.latest_mid,
                    timestamp_ms,
                });
            }

            // === Phase 1: Microstructure Signals - VPIN Update ===
            // Feed trade to VPIN estimator. When a bucket completes, publish MicrostructureUpdate.
            // VPIN uses tick rule (price vs mid) to classify trades, not explicit is_buy.

            // Phase 1A: Feed trade to size distribution tracker
            self.stochastic.trade_size_dist.on_trade(size);

            if let Some(_vpin_value) = self.stochastic.vpin.on_trade(size, trade_price, self.latest_mid, timestamp_ms) {
                // Bucket completed - publish microstructure update to central beliefs
                let vpin = self.stochastic.vpin.vpin();
                let vpin_velocity = self.stochastic.vpin.vpin_velocity();
                let order_flow_direction = self.stochastic.vpin.order_flow_direction();
                let vpin_buckets = self.stochastic.vpin.bucket_count();

                // Get depth-weighted imbalance from cached book sizes
                // Note: using imbalance (static snapshot) rather than OFI (delta)
                // since we don't track previous book state here
                use crate::market_maker::estimator::BookLevel;
                let bid_levels: Vec<BookLevel> = self.cached_bid_sizes.iter()
                    .map(|&sz| BookLevel { size: sz })
                    .collect();
                let ask_levels: Vec<BookLevel> = self.cached_ask_sizes.iter()
                    .map(|&sz| BookLevel { size: sz })
                    .collect();
                let depth_ofi = self.stochastic.enhanced_flow.depth_weighted_imbalance(
                    &bid_levels,
                    &ask_levels,
                );

                // Get liquidity evaporation (updated by L2 handler, just read current value)
                let liquidity_evaporation = self.stochastic.liquidity_evaporation.evaporation_score();

                // Confidence based on bucket count (more buckets = more confidence)
                let confidence = (vpin_buckets as f64 / 50.0).min(1.0);

                // Phase 1A: Get trade size anomaly metrics
                let trade_size_sigma = self.stochastic.trade_size_dist.median_sigma();
                let toxicity_acceleration = self.stochastic.trade_size_dist.toxicity_acceleration(vpin);

                // Phase 1A: Get COFI metrics (updated by L2 handler)
                let cofi = self.stochastic.cofi.cofi();
                let cofi_velocity = self.stochastic.cofi.cofi_velocity();
                let is_sustained_shift = self.stochastic.cofi.is_sustained_shift();

                self.central_beliefs.update(BeliefUpdate::MicrostructureUpdate {
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
            self.stochastic.position_decision.reset_for_regime(continuation_regime);
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
        };

        let result = messages::process_user_fills(
            &user_fills,
            &ctx,
            &mut self.safety.fill_processor,
            &mut fill_state,
        )?;

        // Process fills through WsOrderStateManager for additional state tracking
        // This provides secondary deduplication and state consistency
        for fill in &user_fills.data.fills {
            if fill.coin != *self.config.asset {
                continue;
            }

            let fill_event = WsFillEvent {
                oid: fill.oid,
                tid: fill.tid,
                size: fill.sz.parse().unwrap_or(0.0),
                price: fill.px.parse().unwrap_or(0.0),
                is_buy: fill.side == "B" || fill.side.to_lowercase() == "buy",
                coin: fill.coin.clone(),
                cloid: fill.cloid.clone(),
                timestamp: fill.time,
            };

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

                // Compute realized adverse selection: (mid_after - fill_price) × direction
                // direction = +1 for buy (we bought, if mid moved up we gained),
                // direction = -1 for sell (we sold, if mid moved down we gained)
                // Positive AS means we lost (price moved against us after fill)
                let direction = if is_buy { 1.0 } else { -1.0 };
                let as_realized = (self.latest_mid - fill_price) * direction / fill_price;

                // Compute depth of this fill (distance from mid when placed)
                // For now use adverse selection as proxy until we track order placement mid
                let depth_from_mid = (fill_price - self.latest_mid).abs() / self.latest_mid;

                // PnL for this fill: simplified as -AS (negative AS = profit)
                let fill_pnl = -as_realized;

                // Update all adaptive components via simplified fill handler
                self.stochastic.adaptive_spreads.on_fill_simple(
                    as_realized,
                    depth_from_mid,
                    fill_pnl,
                    self.estimator.kappa(),
                );

                // Record fill for calibration controller
                // This updates fill rate tracking for fill-hungry mode
                self.stochastic.calibration_controller.record_fill();

                // === V2 INTEGRATION: BOCPD Kappa Relationship Update ===
                // Update BOCPD with observed features→kappa relationship + stress signals.
                // First-Principles Gap 1: Adaptive hazard rate based on market stress.
                if let Some(features) = self.stochastic.bocpd_kappa_features.take() {
                    let realized_kappa = self.estimator.kappa();

                    // Gather stress signals for adaptive hazard rate
                    let vpin = self.stochastic.vpin.vpin();
                    let hawkes_intensity_ratio = self.tier2.hawkes.intensity_ratio();
                    let size_anomaly_sigma = self.stochastic.trade_size_dist.anomaly_sigma();

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
                let fill_size: f64 = fill.sz.parse().unwrap_or(0.0);
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
                });

                // === Phase 8: RL Agent Learning Update ===
                // Update Q-values from fill outcome
                {
                    use crate::market_maker::learning::{MDPState, MDPAction, Reward, MarketEvent};

                    // Build current MDP state from fill conditions
                    let book_imbalance = self.estimator.book_imbalance();
                    // Use current sigma vs a baseline (use sigma_clean as proxy for short-term)
                    let vol_ratio = self.estimator.sigma() / self.estimator.sigma_clean().max(0.0001);
                    let adverse_prob = self.stochastic.theoretical_edge.bayesian_adverse();
                    // Use intensity percentile as proxy for excitation level
                    let hawkes_excitation = self.tier2.hawkes.intensity_percentile();

                    let current_state = MDPState::from_continuous(
                        self.position.position(),
                        self.effective_max_position,
                        book_imbalance,
                        vol_ratio,
                        adverse_prob,
                        hawkes_excitation,
                    );

                    // Was this fill adverse? (positive AS means we lost)
                    let was_adverse = as_realized > 0.0;

                    // Compute realized edge in bps
                    let realized_edge_bps = fill_pnl * 10000.0;

                    // Inventory risk: |position| / max_position
                    let inventory_risk = (self.position.position().abs()
                        / self.effective_max_position.max(1.0))
                        .clamp(0.0, 1.0);

                    // Compute reward
                    let reward = Reward::compute(
                        self.stochastic.rl_agent.reward_config(),
                        realized_edge_bps,
                        inventory_risk,
                        vol_ratio,
                        was_adverse,
                    );

                    // Get the actual state-action pair that was used when quoting
                    // This enables proper credit assignment for Q-learning
                    if let Some((state, action)) = self.stochastic.rl_agent.take_last_state_action()
                    {
                        // Update Q-values with the transition using the ACTUAL action
                        self.stochastic.rl_agent.update(
                            state,
                            action.clone(),
                            reward,
                            current_state,
                            false, // not done
                        );

                        debug!(
                            realized_edge_bps = %format!("{:.2}", realized_edge_bps),
                            inventory_risk = %format!("{:.3}", inventory_risk),
                            reward_total = %format!("{:.3}", reward.total),
                            was_adverse = was_adverse,
                            action_spread = %format!("{:.1}", action.spread.delta_bps()),
                            action_skew = %format!("{:.1}", action.skew.bid_skew_bps()),
                            "RL agent updated from fill (actual action)"
                        );
                    } else {
                        // Fallback: no stored action (shouldn't happen normally)
                        let action = MDPAction::default();
                        self.stochastic.rl_agent.update(
                            current_state.clone(),
                            action,
                            reward,
                            current_state,
                            false,
                        );
                        debug!(
                            realized_edge_bps = %format!("{:.2}", realized_edge_bps),
                            inventory_risk = %format!("{:.3}", inventory_risk),
                            reward_total = %format!("{:.3}", reward.total),
                            was_adverse = was_adverse,
                            "RL agent updated from fill (default action - no stored state)"
                        );
                    }

                    // === Competitor Model Observation ===
                    // Estimate queue position from depth
                    let queue_position = depth_from_mid * 100.0; // Simplified: bps * 100
                    let time_in_queue_ms = 100; // Placeholder: would need order timestamp tracking

                    self.stochastic.competitor_model.observe(&MarketEvent::OurFill {
                        queue_position,
                        time_in_queue_ms,
                    });
                }
            }
        }

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
                    let is_full_fill = self.orders.get_order(fill.oid).map_or(true, |o| o.size - o.filled <= 0.0);
                    let _trigger = self.event_accumulator.on_fill(side, fill.oid, size, is_full_fill);

                    // === Position Continuation Model: Update posterior on fill ===
                    // Track whether fill is aligned with current position direction
                    // Aligned fills increase P(continuation), adverse fills decrease it
                    let fill_side_sign = if side == Side::Buy { 1.0 } else { -1.0 };
                    let position_sign = self.position.position().signum();
                    self.stochastic.position_decision.observe_fill(fill_side_sign, position_sign, size);
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
                self.stochastic.liquidity_evaporation.on_book(near_touch_depth, timestamp_ms);

                // Phase 1A: Update COFI (Cumulative OFI with decay)
                // We use top-of-book depth as the OFI proxy
                // Delta = current depth change (we use depth directly since COFI handles decay)
                // For proper COFI, we'd track previous depths, but this approximation works
                // by treating current depth as a "flow" signal
                let bid_delta = near_bid_depth * 0.1; // Scale factor to normalize
                let ask_delta = near_ask_depth * 0.1;
                self.stochastic.cofi.on_book_update(bid_delta, ask_delta, timestamp_ms);
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
                    self.stochastic.competitor_model.observe(&MarketEvent::DepthChange {
                        side: LearningSide::Bid,
                        delta: bid_depth,
                    });
                }
                if ask_depth > 100.0 {
                    self.stochastic.competitor_model.observe(&MarketEvent::DepthChange {
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
            if self.ws_state.remove_order(*oid).is_some() {
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
        // WebData2's clearinghouseState reports the USDC clearinghouse (=0 for HIP-3 users).
        // Only update margin for validator perps (non-HIP-3).
        if self.config.dex.is_none() {
            self.infra.margin_sizer.update_state_with_liquidation(
                account_value,
                margin_used,
                total_notional,
                liquidation_price,
                self.latest_mid,
                self.position.position(),
            );

            // Log occasionally
            debug!(
                account_value = account_value,
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

    /// Called periodically (e.g., every 1s) to update component states.
    ///
    /// This method performs periodic maintenance tasks that don't need to happen
    /// on every market data tick but should run regularly.
    ///
    /// # Phase 2/3 Integration
    ///
    /// TODO: Once ensemble, model_calibration, and alerter are added to tiers:
    /// - Recompute ensemble weights based on recent model performance
    /// - Update model calibration metrics
    /// - Check for degraded models and trigger alerts
    pub fn periodic_component_update(&mut self) {
        // === Phase 2/3 Component Updates ===
        //
        // TODO: Uncomment and wire once components are added to StochasticComponents/InfraComponents:
        //
        // // Recompute ensemble weights based on recent performance
        // self.stochastic.ensemble.compute_weights();
        //
        // // Update model calibration tracking
        // let now = std::time::SystemTime::now()
        //     .duration_since(std::time::UNIX_EPOCH)
        //     .unwrap()
        //     .as_millis() as u64;
        // self.stochastic.model_calibration.update_all(now);
        //
        // // Check for degraded models and create alerts
        // if self.stochastic.model_calibration.is_any_degraded() {
        //     let degraded_models = self.stochastic.model_calibration.degraded_models();
        //     for model_name in degraded_models {
        //         let ir = self.stochastic.model_calibration.get_ir(&model_name).unwrap_or(0.0);
        //         if let Some(alert) = self.infra.alerter.check_calibration(ir, &model_name, now) {
        //             warn!(
        //                 model = %model_name,
        //                 ir = %format!("{:.2}", ir),
        //                 "Model calibration degraded"
        //             );
        //             self.infra.alerter.add_alert(alert);
        //         }
        //     }
        // }

        // Log current state for debugging
        trace!(
            sigma = %format!("{:.6}", self.estimator.sigma()),
            kappa = %format!("{:.3}", self.estimator.kappa()),
            position = %format!("{:.4}", self.position.position()),
            "Periodic component update (Phase 2/3 components not yet wired)"
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

        // Check if we should trigger based on accumulated events
        if let Some(trigger) = self.event_accumulator.should_trigger() {
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
            if stats.total_reconciles % 100 == 0 {
                info!(
                    total_events = stats.total_events,
                    total_reconciles = stats.total_reconciles,
                    filter_ratio = %format!("{:.1}%", stats.filter_ratio() * 100.0),
                    reconcile_frequency = %format!("{:.3}", stats.reconcile_frequency()),
                    "Event accumulator stats (churn reduction)"
                );
            }
        }

        // Also check fallback timer (ensures quotes don't become stale)
        if let Some(fallback_trigger) = self.event_accumulator.check_fallback() {
            debug!(
                event = ?fallback_trigger.event,
                "Event accumulator: fallback timer triggered"
            );

            self.update_quotes().await?;
            self.event_accumulator.reset();
        }

        Ok(())
    }
}
