//! Modular market maker implementation.
//!
//! This module provides a market making system with pluggable components:
//! - **Strategy**: Determines quote prices and sizes (symmetric, inventory-aware, etc.)
//! - **OrderManager**: Tracks resting orders
//! - **PositionTracker**: Tracks position and deduplicates fills
//! - **Executor**: Handles order placement and cancellation

mod config;
mod estimator;
mod executor;
mod order_manager;
mod position;
mod strategy;

pub use config::*;
pub use estimator::*;
pub use executor::*;
pub use order_manager::*;
pub use position::*;
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
}

impl<S: QuotingStrategy, E: OrderExecutor> MarketMaker<S, E> {
    /// Create a new market maker.
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
    ) -> Self {
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
        }
    }

    /// Sync open orders from the exchange.
    /// Call this after creating the market maker to recover state.
    pub async fn sync_open_orders(&mut self) -> Result<()> {
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
            tokio::select! {
                message = receiver.recv() => {
                    match message {
                        Some(msg) => {
                            if let Err(e) = self.handle_message(msg).await {
                                error!("Error handling message: {e}");
                            }
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
                    // Determine aggressor side from trade.side: "B" = buy aggressor, "S" = sell
                    let is_buy_aggressor = Some(trade.side == "B");
                    // Pass price, size, and aggressor side for volume clock + flow tracking
                    self.estimator
                        .on_trade(trade.time, price, size, is_buy_aggressor);
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

                    // Process fill (handles dedup internally)
                    if self.position.process_fill(fill.tid, amount, is_buy) {
                        // Use fill.oid for exact order matching (not side-based guessing)
                        if self.orders.update_fill(fill.oid, amount) {
                            info!(
                                "[Fill] {} {} {} | oid={} | position: {}",
                                if is_buy { "bought" } else { "sold" },
                                amount,
                                self.config.asset,
                                fill.oid,
                                self.position.position()
                            );
                        } else {
                            // Order not in tracking - log warning for investigation
                            warn!(
                                "[Fill] Untracked order filled: oid={} {} {} {}",
                                fill.oid,
                                if is_buy { "bought" } else { "sold" },
                                amount,
                                self.config.asset
                            );
                        }

                        // Record metrics
                        if let Some(ref m) = self.metrics {
                            m.record_fill(amount, is_buy);
                            m.update_position(self.position.position());
                        }
                    }
                }

                // Cleanup filled orders
                self.orders.cleanup_filled();

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

                        self.estimator.on_l2_book(&bids, &asks, self.latest_mid);
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
            kappa: self.estimator.kappa(), // Weighted L2 regression
            arrival_intensity: self.estimator.arrival_intensity(), // Volume ticks/sec
            // Regime detection
            is_toxic_regime: self.estimator.is_toxic_regime(), // RV/BV > 1.5
            jump_ratio: self.estimator.jump_ratio(),           // Fast RV/BV ratio
            // Directional flow
            momentum_bps: self.estimator.momentum_bps(), // Signed momentum
            flow_imbalance: self.estimator.flow_imbalance(), // Buy/sell imbalance
            falling_knife_score: self.estimator.falling_knife_score(), // Downward momentum
            rising_knife_score: self.estimator.rising_knife_score(), // Upward momentum
        };

        debug!(
            mid = self.latest_mid,
            position = self.position.position(),
            max_pos = self.config.max_position,
            target_liq = self.config.target_liquidity,
            sigma_clean = %format!("{:.6}", market_params.sigma),
            sigma_effective = %format!("{:.6}", market_params.sigma_effective),
            kappa = %format!("{:.2}", market_params.kappa),
            jump_ratio = %format!("{:.2}", market_params.jump_ratio),
            is_toxic = market_params.is_toxic_regime,
            momentum_bps = %format!("{:.1}", market_params.momentum_bps),
            flow = %format!("{:.2}", market_params.flow_imbalance),
            falling_knife = %format!("{:.2}", market_params.falling_knife_score),
            rising_knife = %format!("{:.2}", market_params.rising_knife_score),
            "Quote inputs with directional flow"
        );

        let (bid, ask) = self.strategy.calculate_quotes(
            &quote_config,
            self.position.position(),
            self.config.max_position,
            self.config.target_liquidity,
            &market_params,
        );

        debug!(
            bid = ?bid.as_ref().map(|q| (q.price, q.size)),
            ask = ?ask.as_ref().map(|q| (q.price, q.size)),
            "Calculated quotes"
        );

        // Handle bid side
        self.reconcile_side(Side::Buy, bid).await?;

        // Handle ask side
        self.reconcile_side(Side::Sell, ask).await?;

        Ok(())
    }

    /// Reconcile orders on one side.
    /// Cancels ALL orders on a side (not just first) with retry logic.
    async fn reconcile_side(&mut self, side: Side, new_quote: Option<Quote>) -> Result<()> {
        // Get ALL resting orders on this side (not just first)
        let current_orders: Vec<u64> = self
            .orders
            .get_all_by_side(side)
            .iter()
            .map(|o| o.oid)
            .collect();

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
                    // Mark as cancelling before attempting cancel
                    self.orders.set_state(oid, OrderState::Cancelling);

                    if self.cancel_with_retry(&self.config.asset, oid, 3).await {
                        info!("Cancelled {} order: oid={}", side_str(side), oid);
                        self.orders.remove_order(oid);
                    } else {
                        // Revert to resting if cancel failed
                        self.orders.set_state(oid, OrderState::Resting);
                        warn!(
                            "Failed to cancel {} order after retries: oid={}",
                            side_str(side),
                            oid
                        );
                    }
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
                        self.orders.set_state(oid, OrderState::Cancelling);

                        if self.cancel_with_retry(&self.config.asset, oid, 3).await {
                            info!("Cancelled {} order for update: oid={}", side_str(side), oid);
                            self.orders.remove_order(oid);
                        } else {
                            self.orders.set_state(oid, OrderState::Resting);
                            warn!(
                                "Failed to cancel {} order after retries: oid={}",
                                side_str(side),
                                oid
                            );
                        }
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

    /// Place a new order.
    async fn place_new_order(&mut self, side: Side, quote: &Quote) -> Result<()> {
        let is_buy = side == Side::Buy;
        let result = self
            .executor
            .place_order(&self.config.asset, quote.price, quote.size, is_buy)
            .await;

        if result.oid != 0 {
            info!(
                "{} {} {} resting at {}",
                side_str(side),
                result.resting_size,
                self.config.asset,
                quote.price
            );

            self.orders.add_order(TrackedOrder::new(
                result.oid,
                side,
                quote.price,
                result.resting_size,
            ));
        }

        Ok(())
    }

    /// Cancel an order with retry logic.
    /// Returns true if cancel succeeded, false if all retries failed.
    async fn cancel_with_retry(&self, asset: &str, oid: u64, max_attempts: u32) -> bool {
        for attempt in 0..max_attempts {
            if self.executor.cancel_order(asset, oid).await {
                return true;
            }
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
        false
    }

    /// Graceful shutdown - cancel all resting orders.
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("Cancelling all resting orders...");

        let oids: Vec<u64> = self.orders.order_ids();
        for oid in oids {
            if self.cancel_with_retry(&self.config.asset, oid, 3).await {
                info!("Cancelled order: oid={}", oid);
            } else {
                warn!("Failed to cancel order after retries: oid={}", oid);
            }
        }

        Ok(())
    }

    /// Safety sync - periodically verify local state matches exchange.
    /// This is a fallback mechanism; if working correctly, should find no discrepancies.
    async fn safety_sync(&mut self) -> Result<()> {
        debug!("Running safety sync...");

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
            if self.executor.cancel_order(&self.config.asset, *oid).await {
                info!("[SafetySync] Cancelled orphan order: oid={}", oid);
            }
        }

        // Check for stale orders in tracking (exist locally but not on exchange)
        for oid in local_oids.difference(&exchange_oids) {
            warn!(
                "[SafetySync] Stale order in tracking (not on exchange): oid={} - removing",
                oid
            );
            self.orders.remove_order(*oid);
        }

        // Log if everything is in sync
        if exchange_oids == local_oids {
            debug!(
                "[SafetySync] State in sync: {} orders on both sides",
                local_oids.len()
            );
        }

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
}

fn side_str(side: Side) -> &'static str {
    match side {
        Side::Buy => "BUY",
        Side::Sell => "SELL",
    }
}
