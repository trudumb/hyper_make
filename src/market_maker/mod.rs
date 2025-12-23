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

use alloy::primitives::Address;
use tracing::{debug, error, info, warn};
use tokio::sync::mpsc::unbounded_channel;

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

            self.orders.add_order(TrackedOrder::new(order.oid, side, px, sz));
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

        // Subscribe to Trades for volatility estimation
        self.info_client
            .subscribe(
                Subscription::Trades {
                    coin: self.config.asset.clone(),
                },
                sender.clone(),
            )
            .await?;

        // Note: L2Book subscription removed - kappa is now estimated from trade depths
        // If L2 data is needed for other purposes in the future, re-add the subscription here
        drop(sender); // Explicitly drop the sender since we're done with subscriptions

        info!(
            "Market maker started for {} with {} strategy",
            self.config.asset,
            self.strategy.name()
        );
        info!("Warming up parameter estimator...");

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
                _ = tokio::signal::ctrl_c() => {
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
                    let mid: f64 = mid
                        .parse()
                        .map_err(|_| crate::Error::FloatStringParse)?;
                    self.latest_mid = mid;
                    self.estimator.on_mid_update(mid);
                    self.update_quotes().await?;
                }
            }
            Message::Trades(trades) => {
                // Update volatility estimate from trades
                for trade in &trades.data {
                    if trade.coin != self.config.asset {
                        continue;
                    }
                    let price: f64 = trade
                        .px
                        .parse()
                        .map_err(|_| crate::Error::FloatStringParse)?;
                    self.estimator.on_trade(trade.time, price);
                }

                // Log warmup progress (throttled)
                if !self.estimator.is_warmed_up() {
                    let (current, min) = self.estimator.warmup_progress();
                    // Log every 10 trades
                    if current >= self.last_warmup_log + 10 || current == min {
                        info!("Warming up: {}/{} trades collected", current, min);
                        self.last_warmup_log = current;
                    }
                    if current >= min {
                        info!(
                            "Warmup complete! σ={:.4}, κ={:.4}",
                            self.estimator.sigma(),
                            self.estimator.kappa()
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
                        let side = if is_buy { Side::Buy } else { Side::Sell };
                        info!(
                            "[Fill] {} {} {} | position: {}",
                            if is_buy { "bought" } else { "sold" },
                            amount,
                            self.config.asset,
                            self.position.position()
                        );

                        // Update order manager
                        if let Some(order) = self.orders.get_by_side(side) {
                            let oid = order.oid;
                            self.orders.update_fill(oid, amount);
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
            half_spread_bps: self.config.half_spread_bps,
            decimals: self.config.decimals,
            sz_decimals: self.config.sz_decimals,
            min_notional: MIN_ORDER_NOTIONAL,
        };

        // Build market params from live estimates
        let market_params = MarketParams {
            sigma: self.estimator.sigma(),
            kappa: self.estimator.kappa(),
            tau: self.estimator.tau(),
        };

        debug!(
            mid = self.latest_mid,
            position = self.position.position(),
            max_pos = self.config.max_position,
            target_liq = self.config.target_liquidity,
            sigma = %format!("{:.6}", market_params.sigma),
            kappa = %format!("{:.4}", market_params.kappa),
            tau = %format!("{:.2e}", market_params.tau),
            "Quote inputs"
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
    async fn reconcile_side(&mut self, side: Side, new_quote: Option<Quote>) -> Result<()> {
        let current_order = self.orders.get_by_side(side);

        debug!(
            side = %side_str(side),
            current = ?current_order.map(|o| (o.oid, o.price, o.remaining())),
            new = ?new_quote.as_ref().map(|q| (q.price, q.size)),
            "Reconciling"
        );

        match (current_order, new_quote) {
            // Have order, no quote -> cancel
            (Some(order), None) => {
                let oid = order.oid;
                if self.executor.cancel_order(&self.config.asset, oid).await {
                    info!("Cancelled {} order: oid={}", side_str(side), oid);
                    self.orders.remove_order(oid);
                }
            }
            // Have order, have quote -> check if needs update
            (Some(order), Some(quote)) => {
                let needs_update = self.orders.needs_update(side, &quote, self.config.max_bps_diff);
                if needs_update {
                    let oid = order.oid;
                    // Cancel first
                    if self.executor.cancel_order(&self.config.asset, oid).await {
                        self.orders.remove_order(oid);
                        // Place new order
                        self.place_new_order(side, &quote).await?;
                    }
                    // If cancel failed, wait for fill event
                }
            }
            // No order, have quote -> place new
            (None, Some(quote)) => {
                self.place_new_order(side, &quote).await?;
            }
            // No order, no quote -> nothing to do
            (None, None) => {}
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

    /// Graceful shutdown - cancel all resting orders.
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("Cancelling all resting orders...");

        let oids: Vec<u64> = self.orders.order_ids();
        for oid in oids {
            if self.executor.cancel_order(&self.config.asset, oid).await {
                info!("Cancelled order: oid={}", oid);
            }
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
