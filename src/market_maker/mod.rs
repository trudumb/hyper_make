//! Modular market maker implementation.
//!
//! This module provides a market making system with pluggable components:
//! - **Strategy**: Determines quote prices and sizes (symmetric, inventory-aware, etc.)
//! - **OrderManager**: Tracks resting orders
//! - **PositionTracker**: Tracks position and deduplicates fills
//! - **Executor**: Handles order placement and cancellation

mod adverse_selection;
mod config;
mod correlation;
mod data_quality;
mod estimator;
mod executor;
mod funding;
mod hawkes;
mod kill_switch;
mod ladder;
mod liquidation;
mod margin;
mod metrics;
mod order_manager;
mod pnl;
mod position;
mod queue;
mod reconnection;
mod spread;
mod strategy;

pub use adverse_selection::*;
pub use config::*;
pub use correlation::*;
pub use data_quality::*;
pub use estimator::*;
pub use executor::*;
pub use funding::*;
pub use hawkes::*;
pub use kill_switch::*;
pub use ladder::*;
pub use liquidation::*;
pub use margin::*;
pub use metrics::*;
pub use order_manager::*;
pub use pnl::*;
pub use position::*;
pub use queue::*;
pub use reconnection::*;
pub use spread::*;
pub use strategy::*;

use std::collections::HashSet;
use std::time::Duration;

use alloy::primitives::Address;
use tokio::sync::mpsc::unbounded_channel;
use tracing::{debug, error, info, warn};

use crate::prelude::Result;
use crate::{InfoClient, Message, Subscription};

/// Minimum order notional value in USD (Hyperliquid requirement)
const MIN_ORDER_NOTIONAL: f64 = 10.0;

/// Market maker orchestrator.
///
/// Coordinates strategy, order management, position tracking, and execution.
/// Includes live parameter estimation for GLFT strategy (σ from trades, κ from L2 book).
/// Tier 1 modules provide adverse selection measurement, queue tracking, and cascade detection.
pub struct MarketMaker<S: QuotingStrategy, E: OrderExecutor> {
    /// Configuration
    config: MarketMakerConfig,
    /// Quoting strategy
    strategy: S,
    /// Order executor
    executor: E,
    /// Order state manager
    orders: OrderManager,
    /// Position tracker
    position: PositionTracker,
    /// Info client for subscriptions
    info_client: InfoClient,
    /// User address
    user_address: Address,
    /// Latest mid price
    latest_mid: f64,
    /// Metrics recorder
    metrics: MetricsRecorder,
    /// Parameter estimator for σ and κ
    estimator: ParameterEstimator,
    /// Last logged warmup progress (to avoid spam)
    last_warmup_log: usize,

    // === Tier 1: Production Resilience Modules ===
    /// Adverse selection estimator - measures E[Δp|fill] and provides spread adjustment
    adverse_selection: AdverseSelectionEstimator,
    /// Queue position tracker - estimates P(fill) for each order
    queue_tracker: QueuePositionTracker,
    /// Liquidation cascade detector - tail risk management with Hawkes process
    liquidation_detector: LiquidationCascadeDetector,

    // === Production Safety ===
    /// Kill switch - emergency shutdown on drawdown/loss/stale data
    kill_switch: KillSwitch,

    // === Tier 2: Process Models ===
    /// Hawkes order flow estimator - self-exciting trade intensity
    hawkes: HawkesOrderFlowEstimator,
    /// Funding rate estimator - perpetual funding cost modeling
    funding: FundingRateEstimator,
    /// Spread process estimator - spread dynamics and regime detection
    spread_tracker: SpreadProcessEstimator,
    /// P&L tracker - attribution breakdown (fills, funding, unrealized)
    pnl_tracker: PnLTracker,

    // === Production Infrastructure ===
    /// Margin-aware sizer - pre-flight checks for leverage constraints
    margin_sizer: MarginAwareSizer,
    /// Prometheus metrics - production monitoring
    prometheus: PrometheusMetrics,
    /// Connection health monitor - WS staleness tracking
    connection_health: ConnectionHealthMonitor,
    /// Data quality monitor - validates incoming market data
    data_quality: DataQualityMonitor,

    // === Multi-Asset (optional) ===
    /// Correlation estimator - portfolio risk (None for single-asset)
    /// Reserved for future multi-asset support
    #[allow(dead_code)]
    correlation: Option<CorrelationEstimator>,
}

impl<S: QuotingStrategy, E: OrderExecutor> MarketMaker<S, E> {
    /// Create a new market maker.
    ///
    /// # Parameters
    /// - `config`: Market maker configuration (asset, position limits, etc.)
    /// - `strategy`: Quoting strategy (GLFT, Symmetric, etc.)
    /// - `executor`: Order executor (Hyperliquid or mock)
    /// - `info_client`: WebSocket client for market data
    /// - `user_address`: Trader's Ethereum address
    /// - `initial_position`: Starting position (from exchange sync)
    /// - `metrics`: Optional metrics recorder
    /// - `estimator_config`: Configuration for parameter estimation
    /// - `as_config`: Adverse selection estimator configuration
    /// - `queue_config`: Queue position tracker configuration
    /// - `liquidation_config`: Liquidation cascade detector configuration
    /// - `kill_switch_config`: Kill switch configuration for emergency shutdown
    /// - `hawkes_config`: Hawkes order flow estimator configuration
    /// - `funding_config`: Funding rate estimator configuration
    /// - `spread_config`: Spread process estimator configuration
    /// - `pnl_config`: P&L tracker configuration
    /// - `margin_config`: Margin-aware sizer configuration
    /// - `data_quality_config`: Data quality monitor configuration
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: MarketMakerConfig,
        strategy: S,
        executor: E,
        info_client: InfoClient,
        user_address: Address,
        initial_position: f64,
        metrics: MetricsRecorder,
        estimator_config: EstimatorConfig,
        as_config: AdverseSelectionConfig,
        queue_config: QueueConfig,
        liquidation_config: LiquidationConfig,
        kill_switch_config: KillSwitchConfig,
        hawkes_config: HawkesConfig,
        funding_config: FundingConfig,
        spread_config: SpreadConfig,
        pnl_config: PnLConfig,
        margin_config: MarginConfig,
        data_quality_config: data_quality::DataQualityConfig,
    ) -> Self {
        // Capture multi_asset flag before moving config
        let enable_correlation = config.multi_asset;

        Self {
            config,
            strategy,
            executor,
            orders: OrderManager::new(),
            position: PositionTracker::new(initial_position),
            info_client,
            user_address,
            latest_mid: -1.0,
            metrics,
            estimator: ParameterEstimator::new(estimator_config),
            last_warmup_log: 0,
            // Tier 1 modules
            adverse_selection: AdverseSelectionEstimator::new(as_config),
            queue_tracker: QueuePositionTracker::new(queue_config),
            liquidation_detector: LiquidationCascadeDetector::new(liquidation_config),
            // Production safety
            kill_switch: KillSwitch::new(kill_switch_config),
            // Tier 2: Process models
            hawkes: HawkesOrderFlowEstimator::new(hawkes_config),
            funding: FundingRateEstimator::new(funding_config),
            spread_tracker: SpreadProcessEstimator::new(spread_config),
            pnl_tracker: PnLTracker::new(pnl_config),
            // Production infrastructure
            margin_sizer: MarginAwareSizer::new(margin_config),
            prometheus: PrometheusMetrics::new(),
            connection_health: ConnectionHealthMonitor::new(),
            data_quality: DataQualityMonitor::new(data_quality_config),
            // Multi-asset correlation tracking (First Principles Gap 5)
            correlation: if enable_correlation {
                Some(CorrelationEstimator::new(CorrelationConfig::default()))
            } else {
                None
            },
        }
    }

    /// Sync open orders from the exchange.
    /// Call this after creating the market maker to recover state.
    pub async fn sync_open_orders(&mut self) -> Result<()> {
        // === Refresh margin state on startup ===
        if let Err(e) = self.refresh_margin_state().await {
            warn!("Failed to refresh margin state on startup: {e}");
        }

        let open_orders = self.info_client.open_orders(self.user_address).await?;

        for order in open_orders.iter().filter(|o| o.coin == self.config.asset) {
            let sz: f64 = order.sz.parse().unwrap_or(0.0);
            let px: f64 = order.limit_px.parse().unwrap_or(0.0);
            let side = if order.side == "B" {
                Side::Buy
            } else {
                Side::Sell
            };

            info!(
                "[MarketMaker] Found resting {}: oid={} sz={} px={}",
                if side == Side::Buy { "BUY" } else { "SELL" },
                order.oid,
                sz,
                px
            );

            self.orders
                .add_order(TrackedOrder::new(order.oid, side, px, sz));
        }

        Ok(())
    }

    /// Start the market maker event loop.
    pub async fn start(&mut self) -> Result<()> {
        let (sender, mut receiver) = unbounded_channel();

        // Subscribe to UserFills for fill detection
        self.info_client
            .subscribe(
                Subscription::UserFills {
                    user: self.user_address,
                },
                sender.clone(),
            )
            .await?;

        // Subscribe to AllMids for price updates
        self.info_client
            .subscribe(Subscription::AllMids, sender.clone())
            .await?;

        // Subscribe to Trades for volatility and arrival intensity estimation
        self.info_client
            .subscribe(
                Subscription::Trades {
                    coin: self.config.asset.clone(),
                },
                sender.clone(),
            )
            .await?;

        // Subscribe to L2Book for kappa (order book depth decay) estimation
        self.info_client
            .subscribe(
                Subscription::L2Book {
                    coin: self.config.asset.clone(),
                },
                sender.clone(),
            )
            .await?;

        drop(sender); // Explicitly drop the sender since we're done with subscriptions

        info!(
            "Market maker started for {} with {} strategy",
            self.config.asset,
            self.strategy.name()
        );
        info!("Warming up parameter estimator...");

        // Safety sync interval (60 seconds) - fallback to catch any state divergence
        let mut sync_interval = tokio::time::interval(Duration::from_secs(60));
        // Skip the immediate first tick
        sync_interval.tick().await;

        // Create shutdown signal listener ONCE before the loop (not inside select!)
        // This ensures the signal is captured even if pressed during message processing
        let mut shutdown_signal = std::pin::pin!(tokio::signal::ctrl_c());

        loop {
            // === Kill Switch Check (before processing any message) ===
            if self.kill_switch.is_triggered() {
                let reasons = self.kill_switch.trigger_reasons();
                error!(
                    "KILL SWITCH TRIGGERED: {:?}",
                    reasons.iter().map(|r| r.to_string()).collect::<Vec<_>>()
                );
                break;
            }

            tokio::select! {
                message = receiver.recv() => {
                    match message {
                        Some(msg) => {
                            if let Err(e) = self.handle_message(msg).await {
                                error!("Error handling message: {e}");
                            }

                            // Check kill switch after each message
                            self.check_kill_switch();
                        }
                        None => {
                            warn!("Message channel closed, stopping market maker");
                            break;
                        }
                    }
                }
                _ = sync_interval.tick() => {
                    if let Err(e) = self.safety_sync().await {
                        warn!("Safety sync failed: {e}");
                    }

                    // === Update Prometheus metrics ===
                    let pnl_summary = self.pnl_tracker.summary(self.latest_mid);
                    self.prometheus.update_position(
                        self.position.position(),
                        self.config.max_position,
                    );
                    self.prometheus.update_pnl(
                        pnl_summary.total_pnl,      // daily_pnl (total for now)
                        pnl_summary.total_pnl,      // peak_pnl (simplified)
                        pnl_summary.realized_pnl,
                        pnl_summary.unrealized_pnl,
                    );
                    self.prometheus.update_market(
                        self.latest_mid,
                        self.spread_tracker.current_spread_bps(),
                        self.estimator.sigma_clean(),
                        self.estimator.jump_ratio(),
                        self.estimator.kappa(),
                    );
                    self.prometheus.update_risk(
                        self.kill_switch.is_triggered(),
                        self.liquidation_detector.cascade_severity(),
                        self.adverse_selection.realized_as_bps(),
                        self.liquidation_detector.tail_risk_multiplier(),
                    );

                    // === Update connection health metrics ===
                    let connected = self.connection_health.state() == ConnectionState::Healthy;
                    self.prometheus.set_websocket_connected(connected);
                    self.prometheus.set_last_trade_age_ms(
                        self.connection_health.time_since_last_data().as_millis() as u64
                    );
                    // L2 book uses the same connection health tracker
                    self.prometheus.set_last_book_age_ms(
                        self.connection_health.time_since_last_data().as_millis() as u64
                    );

                    // === Update P&L inventory snapshot for carry calculation ===
                    if self.latest_mid > 0.0 {
                        self.pnl_tracker.record_inventory_snapshot(self.latest_mid);
                    }

                    // Log Prometheus output (for scraping or debugging)
                    debug!(
                        prometheus_output = %self.prometheus.to_prometheus_text(&self.config.asset),
                        "Prometheus metrics snapshot"
                    );

                    // Log kill switch status periodically
                    let summary = self.kill_switch.summary();
                    debug!(
                        daily_pnl = %format!("${:.2}", summary.daily_pnl),
                        drawdown = %format!("{:.1}%", summary.drawdown_pct),
                        position_value = %format!("${:.2}", summary.position_value),
                        data_age = %format!("{:.1}s", summary.data_age_secs),
                        cascade_severity = %format!("{:.2}", summary.cascade_severity),
                        "Kill switch status"
                    );
                }
                _ = &mut shutdown_signal => {
                    info!("Shutdown signal received");
                    break;
                }
            }
        }

        // Graceful shutdown
        self.shutdown().await?;

        info!("Market maker stopped.");
        Ok(())
    }

    /// Handle a message from subscriptions.
    async fn handle_message(&mut self, message: Message) -> Result<()> {
        match message {
            Message::AllMids(all_mids) => {
                let mids = all_mids.data.mids;
                if let Some(mid) = mids.get(&self.config.asset) {
                    let mid: f64 = mid.parse().map_err(|_| crate::Error::FloatStringParse)?;
                    self.latest_mid = mid;
                    self.estimator.on_mid_update(mid);

                    // === Connection Health: Record data received ===
                    self.connection_health.record_data_received();

                    // === Tier 1: Update AS estimator (resolves pending fills) ===
                    self.adverse_selection.update(mid);

                    // === Tier 1: Update AS signals from estimator ===
                    self.adverse_selection.update_signals(
                        self.estimator.sigma_total(),
                        self.estimator.sigma_clean(),
                        self.estimator.flow_imbalance(),
                        self.estimator.jump_ratio(),
                    );

                    // === Tier 1: Periodic update of liquidation detector ===
                    self.liquidation_detector.update();

                    self.update_quotes().await?;
                }
            }
            Message::Trades(trades) => {
                // Update volatility estimate from trades (feeds volume clock + flow tracker)
                for trade in &trades.data {
                    if trade.coin != self.config.asset {
                        continue;
                    }
                    let price: f64 = trade
                        .px
                        .parse()
                        .map_err(|_| crate::Error::FloatStringParse)?;
                    let size: f64 = trade
                        .sz
                        .parse()
                        .map_err(|_| crate::Error::FloatStringParse)?;

                    // === Data Quality: Validate trade price and size ===
                    if let Err(anomaly) = self.data_quality.check_trade(
                        &self.config.asset,
                        0, // No sequence number available
                        trade.time,
                        price,
                        size,
                        self.latest_mid,
                    ) {
                        warn!(anomaly = %anomaly, price = %price, size = %size, "Trade quality issue");
                        self.prometheus.record_data_quality_issue();
                        continue; // Skip this trade
                    }

                    // Determine aggressor side from trade.side: "B" = buy aggressor, "S" = sell
                    let is_buy_aggressor = trade.side == "B";
                    // Pass price, size, and aggressor side for volume clock + flow tracking
                    self.estimator
                        .on_trade(trade.time, price, size, Some(is_buy_aggressor));

                    // === Tier 2: Feed Hawkes order flow estimator ===
                    self.hawkes.record_trade(is_buy_aggressor, size);
                }

                // Log warmup progress (throttled)
                if !self.estimator.is_warmed_up() {
                    let (vol_ticks, min_vol, l2_updates, min_l2) = self.estimator.warmup_progress();
                    // Log every 5 volume ticks
                    if vol_ticks >= self.last_warmup_log + 5 {
                        info!(
                            "Warming up: {}/{} volume ticks, {}/{} L2 updates",
                            vol_ticks, min_vol, l2_updates, min_l2
                        );
                        self.last_warmup_log = vol_ticks;
                    }
                    if vol_ticks >= min_vol && l2_updates >= min_l2 {
                        info!(
                            "Warmup complete! σ={:.6}, κ={:.2}, jump_ratio={:.2}",
                            self.estimator.sigma(),
                            self.estimator.kappa(),
                            self.estimator.jump_ratio()
                        );
                    }
                }
            }
            Message::UserFills(user_fills) => {
                if self.latest_mid < 0.0 {
                    return Ok(()); // Haven't seen mid price yet
                }

                for fill in user_fills.data.fills {
                    if fill.coin != self.config.asset {
                        continue;
                    }

                    let amount: f64 = fill
                        .sz
                        .parse()
                        .map_err(|_| crate::Error::FloatStringParse)?;
                    let is_buy = fill.side.eq("B");
                    let tid = fill.tid;
                    let oid = fill.oid;

                    // Step 1: Always update position first (handles tid dedup internally)
                    if self.position.process_fill(tid, amount, is_buy) {
                        // Step 2: Update order tracking with proper state management
                        let (order_found, is_new_fill, is_complete) =
                            self.orders.process_fill(oid, tid, amount);

                        if !order_found {
                            // Order not in tracking - log warning for investigation
                            // Position is still correct (updated above)
                            warn!(
                                "[Fill] Untracked order filled: oid={} tid={} {} {} {} | position updated to {}",
                                oid,
                                tid,
                                if is_buy { "bought" } else { "sold" },
                                amount,
                                self.config.asset,
                                self.position.position()
                            );
                        } else if is_new_fill {
                            // New fill on tracked order - record in all systems
                            let fill_price: f64 = fill.px.parse().unwrap_or(self.latest_mid);

                            // === Tier 1: Record fill for AS measurement ===
                            self.adverse_selection.record_fill(
                                tid,
                                amount,
                                is_buy,
                                self.latest_mid,
                            );

                            // === Tier 2: Record fill in P&L tracker ===
                            self.pnl_tracker.record_fill(
                                tid,
                                fill_price,
                                amount,
                                is_buy,
                                self.latest_mid, // mid_at_fill
                            );

                            // === Production: Update Prometheus metrics ===
                            self.prometheus.record_fill(amount, is_buy);

                            let pnl_summary = self.pnl_tracker.summary(self.latest_mid);
                            info!(
                                "[Fill] {} {} {} | oid={} tid={} | position: {} | AS: {:.2}bps | P&L: ${:.2}",
                                if is_buy { "bought" } else { "sold" },
                                amount,
                                self.config.asset,
                                oid,
                                tid,
                                self.position.position(),
                                self.adverse_selection.realized_as_bps(),
                                pnl_summary.total_pnl
                            );

                            // Record metrics
                            if let Some(ref m) = self.metrics {
                                m.record_fill(amount, is_buy);
                                m.update_position(self.position.position());
                            }

                            // Update queue tracker based on fill completeness
                            if is_complete {
                                // Order fully filled - will be removed via cleanup()
                                // Don't remove from queue tracker here; let cleanup() handle it
                            } else {
                                // Partial fill - update queue position
                                self.queue_tracker.order_partially_filled(oid, amount);
                            }
                        }
                        // else: duplicate fill at order level - already processed

                        // Position milestone warnings
                        let current_position = self.position.position();
                        let utilization =
                            (current_position.abs() / self.config.max_position).min(1.0);
                        let utilization_pct = utilization * 100.0;

                        // Warn at 50%, 75%, 90% thresholds (only when crossing upward)
                        if utilization >= 0.90 {
                            warn!(
                                position = %format!("{:.6}", current_position),
                                max_position = %format!("{:.6}", self.config.max_position),
                                utilization = %format!("{:.1}%", utilization_pct),
                                "Position at 90%+ of max - approaching reduce-only mode"
                            );
                        } else if utilization >= 0.75 {
                            warn!(
                                position = %format!("{:.6}", current_position),
                                max_position = %format!("{:.6}", self.config.max_position),
                                utilization = %format!("{:.1}%", utilization_pct),
                                "Position at 75%+ of max"
                            );
                        } else if utilization >= 0.50 {
                            debug!(
                                position = %format!("{:.6}", current_position),
                                max_position = %format!("{:.6}", self.config.max_position),
                                utilization = %format!("{:.1}%", utilization_pct),
                                "Position at 50%+ of max"
                            );
                        }
                    }
                }

                // Run cleanup cycle - deferred removal of terminal orders
                // This handles the fill window for cancelled orders
                let removed_oids = self.orders.cleanup();
                for oid in removed_oids {
                    // Now safe to remove from QueueTracker
                    self.queue_tracker.order_removed(oid);
                }

                // Update quotes after fills
                self.update_quotes().await?;
            }
            Message::L2Book(l2_book) => {
                // Update kappa estimate from L2 order book
                if l2_book.data.coin == self.config.asset && self.latest_mid > 0.0 {
                    // Parse L2 levels: levels[0] = bids, levels[1] = asks
                    if l2_book.data.levels.len() >= 2 {
                        let bids: Vec<(f64, f64)> = l2_book.data.levels[0]
                            .iter()
                            .filter_map(|level| {
                                let px: f64 = level.px.parse().ok()?;
                                let sz: f64 = level.sz.parse().ok()?;
                                Some((px, sz))
                            })
                            .collect();

                        let asks: Vec<(f64, f64)> = l2_book.data.levels[1]
                            .iter()
                            .filter_map(|level| {
                                let px: f64 = level.px.parse().ok()?;
                                let sz: f64 = level.sz.parse().ok()?;
                                Some((px, sz))
                            })
                            .collect();

                        // === Data Quality: Check for crossed book ===
                        if let (Some((best_bid, _)), Some((best_ask, _))) =
                            (bids.first(), asks.first())
                        {
                            if let Err(anomaly) = self.data_quality.check_l2_book(
                                &self.config.asset,
                                0,
                                *best_bid,
                                *best_ask,
                            ) {
                                warn!(anomaly = %anomaly, "L2 book quality issue");
                                self.prometheus.record_data_quality_issue();
                                if matches!(anomaly, AnomalyType::CrossedBook) {
                                    self.prometheus.record_crossed_book();
                                    // Skip processing crossed book
                                    return Ok(());
                                }
                            }
                        }

                        self.estimator.on_l2_book(&bids, &asks, self.latest_mid);

                        // === Tier 1: Update queue tracker with best bid/ask ===
                        if let (Some((best_bid, _)), Some((best_ask, _))) =
                            (bids.first(), asks.first())
                        {
                            self.queue_tracker.update_from_book(
                                *best_bid,
                                *best_ask,
                                self.estimator.sigma_clean(),
                            );

                            // === Tier 2: Update spread tracker from L2 book ===
                            self.spread_tracker.update(
                                *best_bid,
                                *best_ask,
                                self.estimator.sigma_clean(), // volatility
                            );
                        }
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Update quotes based on current market state.
    async fn update_quotes(&mut self) -> Result<()> {
        // Don't place orders until estimator is warmed up
        if !self.estimator.is_warmed_up() {
            return Ok(());
        }

        let quote_config = QuoteConfig {
            mid_price: self.latest_mid,
            decimals: self.config.decimals,
            sz_decimals: self.config.sz_decimals,
            min_notional: MIN_ORDER_NOTIONAL,
        };

        // Build market params from econometric estimates
        let market_params = MarketParams {
            // Volatility (dual-sigma architecture)
            sigma: self.estimator.sigma_clean(), // √BV (jump-robust) for spread
            sigma_total: self.estimator.sigma_total(), // √RV (includes jumps) for risk
            sigma_effective: self.estimator.sigma_effective(), // Blended for skew
            // Order book
            kappa: self.estimator.kappa(), // Fill-rate kappa from trade distances
            arrival_intensity: self.estimator.arrival_intensity(), // Volume ticks/sec
            // Regime detection
            is_toxic_regime: self.estimator.is_toxic_regime(), // RV/BV > 1.5
            jump_ratio: self.estimator.jump_ratio(),           // Fast RV/BV ratio
            // Directional flow (for diagnostics)
            momentum_bps: self.estimator.momentum_bps(), // Signed momentum
            flow_imbalance: self.estimator.flow_imbalance(), // Buy/sell imbalance
            falling_knife_score: self.estimator.falling_knife_score(), // Downward momentum
            rising_knife_score: self.estimator.rising_knife_score(), // Upward momentum
            // L2 book structure
            book_imbalance: self.estimator.book_imbalance(), // Bid/ask depth asymmetry
            liquidity_gamma_mult: self.estimator.liquidity_gamma_multiplier(), // Thin book scaling
            // Microprice: data-driven fair price incorporating signal predictions
            microprice: self.estimator.microprice(), // mid × (1 + β_book×imb + β_flow×flow)
            beta_book: self.estimator.beta_book(),   // Learned coefficient for book imbalance
            beta_flow: self.estimator.beta_flow(),   // Learned coefficient for flow imbalance
            // === Tier 1: Adverse Selection ===
            as_spread_adjustment: self.adverse_selection.spread_adjustment(),
            predicted_alpha: self.adverse_selection.predicted_alpha(),
            as_warmed_up: self.adverse_selection.is_warmed_up(),
            // === Tier 1: Liquidation Cascade ===
            tail_risk_multiplier: self.liquidation_detector.tail_risk_multiplier(),
            should_pull_quotes: self.liquidation_detector.should_pull_quotes(),
            cascade_size_factor: self.liquidation_detector.size_reduction_factor(),
            // === Tier 2: Hawkes Order Flow ===
            hawkes_buy_intensity: self.hawkes.lambda_buy(),
            hawkes_sell_intensity: self.hawkes.lambda_sell(),
            hawkes_imbalance: self.hawkes.flow_imbalance(),
            hawkes_activity_percentile: self.hawkes.intensity_percentile(),
            // === Tier 2: Funding Rate ===
            funding_rate: self.funding.current_rate(),
            predicted_funding_cost: self.funding.funding_cost(
                self.position.position(),
                self.latest_mid,
                3600.0, // 1 hour holding period in seconds
            ),
            // === Tier 2: Spread Process ===
            fair_spread: self.spread_tracker.fair_spread(),
            spread_percentile: self.spread_tracker.spread_percentile(),
            spread_regime: self.spread_tracker.spread_regime(),
            // === Volatility Regime ===
            volatility_regime: self.estimator.volatility_regime(),
            // === First Principles Extensions (Gaps 1-10) ===
            // Jump-Diffusion (Gap 1)
            lambda_jump: self.estimator.lambda_jump(),
            mu_jump: self.estimator.mu_jump(),
            sigma_jump: self.estimator.sigma_jump(),
            // Stochastic Volatility (Gap 2)
            kappa_vol: self.estimator.kappa_vol(),
            theta_vol: self.estimator.theta_vol_sigma().powi(2), // Store as variance
            xi_vol: self.estimator.xi_vol(),
            rho_price_vol: self.estimator.rho_price_vol(),
            // Queue Model (Gap 3) - use defaults until CalibratedQueueModel integrated
            calibrated_volume_rate: 1.0,
            calibrated_cancel_rate: 0.2,
            // Funding Enhancement (Gap 6)
            premium: self.funding.current_premium(),
            premium_alpha: self.funding.premium_alpha(),
            // Momentum Protection (Gap 10) - use defaults until MomentumModel integrated
            bid_protection_factor: 1.0,
            ask_protection_factor: 1.0,
            p_momentum_continue: 0.5,
        };

        debug!(
            mid = self.latest_mid,
            microprice = %format!("{:.4}", market_params.microprice),
            position = self.position.position(),
            max_pos = self.config.max_position,
            target_liq = self.config.target_liquidity,
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

        // Try multi-level ladder quoting first
        let ladder = self.strategy.calculate_ladder(
            &quote_config,
            self.position.position(),
            self.config.max_position,
            self.config.target_liquidity,
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

            // Reduce-only mode: when over max position, only allow quotes that reduce position
            let position = self.position.position();
            if position.abs() > self.config.max_position {
                if position > 0.0 {
                    // Long position over max: only allow sells (no bids)
                    bid_quotes.clear();
                    warn!(
                        position = %format!("{:.6}", position),
                        max_position = %format!("{:.6}", self.config.max_position),
                        "Over max position (long) - reduce-only mode, cancelling all bids"
                    );
                } else {
                    // Short position over max: only allow buys (no asks)
                    ask_quotes.clear();
                    warn!(
                        position = %format!("{:.6}", position),
                        max_position = %format!("{:.6}", self.config.max_position),
                        "Over max position (short) - reduce-only mode, cancelling all asks"
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

            // Reconcile ladder quotes
            self.reconcile_ladder_side(Side::Buy, bid_quotes).await?;
            self.reconcile_ladder_side(Side::Sell, ask_quotes).await?;
        } else {
            // Fallback to single-quote mode for non-ladder strategies
            let (mut bid, mut ask) = self.strategy.calculate_quotes(
                &quote_config,
                self.position.position(),
                self.config.max_position,
                self.config.target_liquidity,
                &market_params,
            );

            // Reduce-only mode: when over max position, only allow quotes that reduce position
            let position = self.position.position();
            if position.abs() > self.config.max_position {
                if position > 0.0 {
                    // Long position over max: only allow sells (no bids)
                    bid = None;
                    warn!(
                        position = %format!("{:.6}", position),
                        max_position = %format!("{:.6}", self.config.max_position),
                        "Over max position (long) - reduce-only mode, cancelling bids"
                    );
                } else {
                    // Short position over max: only allow buys (no asks)
                    ask = None;
                    warn!(
                        position = %format!("{:.6}", position),
                        max_position = %format!("{:.6}", self.config.max_position),
                        "Over max position (short) - reduce-only mode, cancelling asks"
                    );
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

        Ok(())
    }

    /// Initiate cancel and update order state appropriately.
    /// Does NOT remove order from tracking - that happens via cleanup cycle.
    async fn initiate_and_track_cancel(&mut self, oid: u64) {
        // Mark as CancelPending before sending request
        if !self.orders.initiate_cancel(oid) {
            debug!("Order {} not in cancellable state", oid);
            return;
        }

        let cancel_result = self.cancel_with_retry(&self.config.asset, oid, 3).await;

        match cancel_result {
            CancelResult::Cancelled | CancelResult::AlreadyCancelled => {
                // Cancel confirmed - start fill window (DO NOT REMOVE YET)
                self.orders.on_cancel_confirmed(oid);
                info!("Cancel confirmed for oid={}, waiting for fill window", oid);
            }
            CancelResult::AlreadyFilled => {
                // Order was filled - mark appropriately
                self.orders.on_cancel_already_filled(oid);
                info!("Order {} was already filled when cancelled", oid);
            }
            CancelResult::Failed => {
                // Cancel failed - revert state
                self.orders.on_cancel_failed(oid);
                warn!("Cancel failed for oid={}, reverted to active", oid);
            }
        }
    }

    /// Reconcile orders on one side.
    /// Cancels ALL orders on a side (not just first) with retry logic.
    async fn reconcile_side(&mut self, side: Side, new_quote: Option<Quote>) -> Result<()> {
        // Get ALL active orders on this side (excludes those being cancelled)
        let current_orders: Vec<u64> = self
            .orders
            .get_all_by_side(side)
            .iter()
            .map(|o| o.oid)
            .collect();

        // Check for orders in cancel process (for logging/debugging)
        let pending_cancels: Vec<u64> = self
            .orders
            .get_all_by_side_including_pending(side)
            .iter()
            .filter(|o| {
                matches!(
                    o.state,
                    OrderState::CancelPending | OrderState::CancelConfirmed
                )
            })
            .map(|o| o.oid)
            .collect();

        if !pending_cancels.is_empty() {
            debug!(
                side = %side_str(side),
                pending_cancel_count = pending_cancels.len(),
                "Orders still in cancel window on this side"
            );
        }

        debug!(
            side = %side_str(side),
            current_count = current_orders.len(),
            current_oids = ?current_orders,
            new = ?new_quote.as_ref().map(|q| (q.price, q.size)),
            "Reconciling"
        );

        match (current_orders.is_empty(), new_quote) {
            // Have orders, no quote -> cancel all
            (false, None) => {
                for oid in current_orders {
                    self.initiate_and_track_cancel(oid).await;
                }
            }
            // Have orders, have quote -> check if needs update, cancel all if so
            (false, Some(quote)) => {
                let needs_update = self
                    .orders
                    .needs_update(side, &quote, self.config.max_bps_diff);

                if needs_update {
                    // Cancel all orders on this side
                    for oid in current_orders {
                        self.initiate_and_track_cancel(oid).await;
                    }
                    // Place new order
                    self.place_new_order(side, &quote).await?;
                }
            }
            // No orders, have quote -> place new
            (true, Some(quote)) => {
                self.place_new_order(side, &quote).await?;
            }
            // No orders, no quote -> nothing to do
            (true, None) => {}
        }

        Ok(())
    }

    /// Reconcile multi-level ladder orders on one side.
    ///
    /// Manages multiple orders per side for ladder quoting strategy.
    /// Uses bulk order placement for efficiency (single API call for all levels).
    async fn reconcile_ladder_side(&mut self, side: Side, new_quotes: Vec<Quote>) -> Result<()> {
        // Get all active orders on this side (excludes those being cancelled)
        let current_orders: Vec<u64> = self
            .orders
            .get_all_by_side(side)
            .iter()
            .map(|o| o.oid)
            .collect();

        debug!(
            side = %side_str(side),
            current_count = current_orders.len(),
            new_levels = new_quotes.len(),
            "Reconciling ladder"
        );

        // If no new quotes, cancel all existing orders
        if new_quotes.is_empty() {
            for oid in current_orders {
                self.initiate_and_track_cancel(oid).await;
            }
            return Ok(());
        }

        // Check if ladder needs update
        let needs_update = self.ladder_needs_update(side, &new_quotes);

        if needs_update || current_orders.is_empty() {
            // Cancel all existing orders on this side first
            for oid in &current_orders {
                self.initiate_and_track_cancel(*oid).await;
            }

            // Build order specs for bulk placement
            let is_buy = side == Side::Buy;
            let mut order_specs: Vec<OrderSpec> = Vec::new();

            for quote in &new_quotes {
                if quote.size <= 0.0 {
                    continue;
                }

                // Apply margin check to each level
                let sizing_result = self.margin_sizer.adjust_size(
                    quote.size,
                    quote.price,
                    self.position.position(),
                    is_buy,
                );

                if sizing_result.adjusted_size <= 0.0 {
                    debug!(
                        side = %side_str(side),
                        price = quote.price,
                        requested_size = quote.size,
                        reason = ?sizing_result.constraint_reason,
                        "Ladder level blocked by margin constraints"
                    );
                    continue;
                }

                order_specs.push(OrderSpec {
                    price: quote.price,
                    size: sizing_result.adjusted_size,
                    is_buy,
                });
            }

            if order_specs.is_empty() {
                debug!(
                    side = %side_str(side),
                    "No ladder levels to place (all blocked by margin)"
                );
                return Ok(());
            }

            debug!(
                side = %side_str(side),
                levels = order_specs.len(),
                "Placing ladder levels via bulk order"
            );

            // Place all orders in a single API call
            let results = self
                .executor
                .place_bulk_orders(&self.config.asset, order_specs.clone())
                .await;

            // Track placed orders
            for (i, result) in results.iter().enumerate() {
                if result.oid > 0 {
                    let spec = &order_specs[i];
                    self.orders.add_order(TrackedOrder::new(
                        result.oid,
                        side,
                        spec.price,
                        result.resting_size,
                    ));

                    // Initialize queue tracking for this order
                    // depth_ahead is 0.0 initially; will be updated on L2 book updates
                    self.queue_tracker.order_placed(
                        result.oid,
                        spec.price,
                        result.resting_size,
                        0.0, // depth_ahead estimated; will be refined by L2 updates
                        side == Side::Buy,
                    );
                }
            }
        }

        Ok(())
    }

    /// Check if the ladder needs to be updated.
    ///
    /// Returns true if:
    /// - Number of levels has changed
    /// - Any level's price has moved more than max_bps_diff
    /// - Any level's size has changed by more than 10%
    fn ladder_needs_update(&self, side: Side, new_quotes: &[Quote]) -> bool {
        let current = self.orders.get_all_by_side(side);

        // Different number of levels
        if current.len() != new_quotes.len() {
            return true;
        }

        // Sort current orders by price (bids: descending, asks: ascending)
        let mut sorted_current: Vec<_> = current.iter().collect();
        if side == Side::Buy {
            sorted_current.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap());
        } else {
            sorted_current.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap());
        }

        // Check each level for meaningful price/size changes
        for (order, quote) in sorted_current.iter().zip(new_quotes.iter()) {
            // Price change check
            let price_diff_bps = ((order.price - quote.price) / order.price).abs() * 10000.0;
            if price_diff_bps > self.config.max_bps_diff as f64 {
                return true;
            }

            // Size change check (10% threshold)
            if order.size > 0.0 {
                let size_diff_pct = ((order.size - quote.size) / order.size).abs();
                if size_diff_pct > 0.1 {
                    return true;
                }
            }
        }

        false
    }

    /// Place a new order.
    async fn place_new_order(&mut self, side: Side, quote: &Quote) -> Result<()> {
        let is_buy = side == Side::Buy;

        // === Margin Pre-Flight Check ===
        // Verify order meets margin requirements before placing
        let sizing_result = self.margin_sizer.adjust_size(
            quote.size,
            quote.price,
            self.position.position(),
            is_buy,
        );

        // Skip order if margin check reduces size to zero or margin constrained
        if sizing_result.adjusted_size <= 0.0 {
            debug!(
                side = %side_str(side),
                requested_size = quote.size,
                adjusted_size = sizing_result.adjusted_size,
                reason = ?sizing_result.constraint_reason,
                "Order blocked by margin constraints"
            );
            return Ok(());
        }

        // Use adjusted size from margin check
        let adjusted_size = sizing_result.adjusted_size;
        let result = self
            .executor
            .place_order(&self.config.asset, quote.price, adjusted_size, is_buy)
            .await;

        if result.oid != 0 {
            info!(
                "{} {} {} resting at {} (margin-adjusted from {})",
                side_str(side),
                result.resting_size,
                self.config.asset,
                quote.price,
                quote.size
            );

            self.orders.add_order(TrackedOrder::new(
                result.oid,
                side,
                quote.price,
                result.resting_size,
            ));

            // === Tier 1: Register with queue tracker ===
            // Estimate depth ahead conservatively (2x our target liquidity)
            let depth_ahead = self.config.target_liquidity * 2.0;
            self.queue_tracker.order_placed(
                result.oid,
                quote.price,
                result.resting_size,
                depth_ahead,
                is_buy,
            );
        }

        Ok(())
    }

    /// Cancel an order with retry logic.
    ///
    /// Returns `CancelResult` indicating what happened:
    /// - `Cancelled`: Order was cancelled successfully
    /// - `AlreadyCancelled`: Order was already cancelled
    /// - `AlreadyFilled`: Order was filled before cancel (DO NOT remove from tracking!)
    /// - `Failed`: Cancel failed after all retries
    async fn cancel_with_retry(&self, asset: &str, oid: u64, max_attempts: u32) -> CancelResult {
        for attempt in 0..max_attempts {
            let result = self.executor.cancel_order(asset, oid).await;

            // If order is gone (any reason), stop retrying
            if result.order_is_gone() {
                return result;
            }

            // Only retry on Failed
            if attempt < max_attempts - 1 {
                let delay = Duration::from_millis(100 * 2_u64.pow(attempt));
                debug!(
                    "Cancel failed for oid={}, retrying in {:?} (attempt {}/{})",
                    oid,
                    delay,
                    attempt + 1,
                    max_attempts
                );
                tokio::time::sleep(delay).await;
            }
        }
        CancelResult::Failed
    }

    /// Graceful shutdown - cancel all resting orders and log final state.
    ///
    /// Ensures:
    /// 1. All resting orders are cancelled with retry logic
    /// 2. Final position and P&L state is logged
    /// 3. Kill switch status is reported
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("=== GRACEFUL SHUTDOWN INITIATED ===");

        // Log final state before cancelling
        let final_position = self.position.position();
        let final_mid = self.latest_mid;
        let position_value = final_position.abs() * final_mid;
        let kill_switch_summary = self.kill_switch.summary();

        info!(
            position = %format!("{:.6}", final_position),
            mid_price = %format!("{:.2}", final_mid),
            position_value = %format!("${:.2}", position_value),
            "Final position state"
        );

        // Log kill switch status
        if kill_switch_summary.triggered {
            warn!(
                reasons = ?kill_switch_summary.reasons,
                "Kill switch was triggered during session"
            );
        } else {
            info!(
                daily_pnl = %format!("${:.2}", kill_switch_summary.daily_pnl),
                max_drawdown = %format!("{:.1}%", kill_switch_summary.drawdown_pct),
                "Kill switch was NOT triggered"
            );
        }

        // Log Tier 1 module summaries
        let as_summary = self.adverse_selection.summary();
        info!(
            fills_measured = as_summary.fills_measured,
            realized_as_bps = %format!("{:.4}", as_summary.realized_as_bps),
            spread_adjustment_bps = %format!("{:.4}", as_summary.spread_adjustment_bps),
            "Adverse selection summary"
        );

        let liq_summary = self.liquidation_detector.summary();
        info!(
            total_liquidations = liq_summary.total_liquidations,
            cascade_severity = %format!("{:.2}", liq_summary.cascade_severity),
            "Liquidation cascade summary"
        );

        // Cancel all resting orders with confirmation
        let oids: Vec<u64> = self.orders.order_ids();
        let total_orders = oids.len();

        if total_orders == 0 {
            info!("No resting orders to cancel");
        } else {
            info!("Cancelling {} resting orders...", total_orders);

            let mut cancelled = 0;
            let mut failed = 0;

            for oid in oids {
                let cancel_result = self.cancel_with_retry(&self.config.asset, oid, 3).await;
                if cancel_result.order_is_gone() {
                    // Order is gone (cancelled, already cancelled, or already filled)
                    if cancel_result == CancelResult::AlreadyFilled {
                        info!("Order already filled during shutdown: oid={}", oid);
                    } else {
                        info!("Cancelled order: oid={}", oid);
                    }
                    cancelled += 1;
                } else {
                    warn!("Failed to cancel order after retries: oid={}", oid);
                    failed += 1;
                }
            }

            if failed > 0 {
                warn!(
                    cancelled = cancelled,
                    failed = failed,
                    "Some orders failed to cancel - may need manual cleanup"
                );
            } else {
                info!(cancelled = cancelled, "All orders cancelled successfully");
            }
        }

        info!("=== GRACEFUL SHUTDOWN COMPLETE ===");
        Ok(())
    }

    /// Safety sync - periodically verify local state matches exchange.
    /// This is a fallback mechanism; if working correctly, should find no discrepancies.
    async fn safety_sync(&mut self) -> Result<()> {
        debug!("Running safety sync...");

        // === Run cleanup first to expire any fill windows ===
        let cleaned = self.orders.cleanup();
        for oid in cleaned {
            self.queue_tracker.order_removed(oid);
        }

        // === Check for stuck cancels ===
        let stuck = self.orders.check_stuck_cancels();
        if !stuck.is_empty() {
            warn!("[SafetySync] Stuck cancel orders detected: {:?}", stuck);
        }

        // === Update margin state from user account ===
        if let Err(e) = self.refresh_margin_state().await {
            warn!("Failed to refresh margin state: {e}");
        }

        // Query open orders from exchange
        let exchange_orders = self.info_client.open_orders(self.user_address).await?;
        let exchange_oids: HashSet<u64> = exchange_orders
            .iter()
            .filter(|o| o.coin == self.config.asset)
            .map(|o| o.oid)
            .collect();

        let local_oids: HashSet<u64> = self.orders.order_ids().into_iter().collect();

        // Check for orphaned orders on exchange (exist on exchange but not in local tracking)
        for oid in exchange_oids.difference(&local_oids) {
            warn!(
                "[SafetySync] Orphan order detected on exchange: oid={} - cancelling",
                oid
            );
            let cancel_result = self.executor.cancel_order(&self.config.asset, *oid).await;
            if cancel_result.order_is_gone() {
                info!("[SafetySync] Cancelled orphan order: oid={}", oid);
            }
        }

        // Check for stale orders in tracking (exist locally but not on exchange)
        // IMPORTANT: Don't remove orders that are in cancel/fill window - they may be waiting for late fills
        for oid in local_oids.difference(&exchange_oids) {
            if let Some(order) = self.orders.get_order(*oid) {
                // Don't remove if still in cancel/fill window
                if matches!(
                    order.state,
                    OrderState::CancelPending | OrderState::CancelConfirmed
                ) {
                    debug!("[SafetySync] Order {} in cancel window - not removing", oid);
                    continue;
                }
            }

            warn!(
                "[SafetySync] Stale order in tracking (not on exchange): oid={} - removing",
                oid
            );
            self.orders.remove_order(*oid);
            self.queue_tracker.order_removed(*oid); // Tier 1: Remove from queue tracking
        }

        // Log if everything is in sync (excluding orders in cancel window)
        let active_local_oids: HashSet<u64> = self
            .orders
            .order_ids()
            .into_iter()
            .filter(|oid| {
                self.orders
                    .get_order(*oid)
                    .map(|o| o.is_active())
                    .unwrap_or(false)
            })
            .collect();

        if exchange_oids == active_local_oids {
            debug!(
                "[SafetySync] State in sync: {} active orders (exchange matches local)",
                active_local_oids.len()
            );
        }

        Ok(())
    }

    /// Refresh margin state from exchange.
    ///
    /// Fetches user account state and updates the MarginAwareSizer with current values.
    async fn refresh_margin_state(&mut self) -> Result<()> {
        let user_state = self.info_client.user_state(self.user_address).await?;

        // Parse margin values from the response
        let account_value: f64 = user_state
            .cross_margin_summary
            .account_value
            .parse()
            .unwrap_or(0.0);
        let margin_used: f64 = user_state
            .cross_margin_summary
            .total_margin_used
            .parse()
            .unwrap_or(0.0);
        let total_notional: f64 = user_state
            .cross_margin_summary
            .total_ntl_pos
            .parse()
            .unwrap_or(0.0);

        // Update the margin sizer with fresh values
        self.margin_sizer
            .update_state(account_value, margin_used, total_notional);

        debug!(
            account_value = %format!("{:.2}", account_value),
            margin_used = %format!("{:.2}", margin_used),
            total_notional = %format!("{:.2}", total_notional),
            available = %format!("{:.2}", account_value - margin_used),
            "Margin state refreshed"
        );

        Ok(())
    }

    /// Get current position.
    pub fn position(&self) -> f64 {
        self.position.position()
    }

    /// Get latest mid price.
    pub fn latest_mid(&self) -> f64 {
        self.latest_mid
    }

    /// Get kill switch reference for monitoring.
    pub fn kill_switch(&self) -> &KillSwitch {
        &self.kill_switch
    }

    /// Get prometheus metrics for external monitoring.
    pub fn prometheus(&self) -> &PrometheusMetrics {
        &self.prometheus
    }

    /// Get asset name for metrics labeling.
    pub fn asset(&self) -> &str {
        &self.config.asset
    }

    /// Check kill switch conditions and update state.
    /// Called after each message and periodically.
    fn check_kill_switch(&self) {
        // Calculate data age from connection health
        let data_age = self.connection_health.time_since_last_data();
        let last_data_time = std::time::Instant::now()
            .checked_sub(data_age)
            .unwrap_or_else(std::time::Instant::now);

        // Build current state for kill switch check using actual P&L
        let pnl_summary = self.pnl_tracker.summary(self.latest_mid);
        let state = KillSwitchState {
            daily_pnl: pnl_summary.total_pnl, // Using total as daily for now
            peak_pnl: pnl_summary.total_pnl,  // Simplified (actual peak needs tracking)
            position: self.position.position(),
            mid_price: self.latest_mid,
            last_data_time,
            rate_limit_errors: 0, // TODO: Track rate limit errors
            cascade_severity: self.liquidation_detector.cascade_severity(),
        };

        // Update position in kill switch (for value calculation)
        self.kill_switch
            .update_position(self.position.position(), self.latest_mid);

        // Update cascade severity
        self.kill_switch
            .update_cascade_severity(self.liquidation_detector.cascade_severity());

        // Warn if data is stale
        if self.connection_health.is_data_stale() {
            warn!(
                data_age_secs = %format!("{:.1}", data_age.as_secs_f64()),
                "WebSocket data is stale - connection may be unhealthy"
            );
        }

        // Check all conditions
        if let Some(reason) = self.kill_switch.check(&state) {
            warn!("Kill switch condition detected: {}", reason);
        }
    }
}

fn side_str(side: Side) -> &'static str {
    match side {
        Side::Buy => "BUY",
        Side::Sell => "SELL",
    }
}
