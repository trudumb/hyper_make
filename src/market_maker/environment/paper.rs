//! Paper trading environment — wraps `SimulationExecutor` + `FillSimulator`.
//!
//! This is the **E_paper** in the mathematical structure:
//! `S →[T,π,E_paper]→ S'`
//!
//! The `PaperEnvironment` produces observations from market data WS channels
//! and synthesizes fill events using the FillSimulator.

use alloy::primitives::Address;
use async_trait::async_trait;
use futures_util::Stream;
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::mpsc;
use tracing::{debug, trace};

use crate::prelude::Result;
use crate::ws::message_types::UserFills;
use crate::Message;

use crate::market_maker::simulation::executor::SimulationExecutor;
use crate::market_maker::simulation::fill_sim::{FillSimulator, FillSimulatorConfig, MarketTrade};
use crate::market_maker::{
    CancelResult, ModifyResult, ModifySpec, OrderExecutor, OrderResult, OrderSpec, Side,
};
use crate::TradeInfo;
use crate::UserFillsData;

use super::{Observation, TradingEnvironment};

/// Configuration for `PaperEnvironment`.
#[derive(Debug, Clone)]
pub struct PaperEnvironmentConfig {
    /// Asset being traded (e.g. "BTC", "HYPE").
    pub asset: String,
    /// User address for synthetic UserFills messages.
    pub user_address: Address,
    /// Fill simulator configuration.
    pub fill_sim_config: FillSimulatorConfig,
    /// Optional DEX name for HIP-3 assets (None for validator perps).
    pub dex: Option<String>,
}

/// Paper trading environment.
///
/// Wraps `SimulationExecutor` for order management and `FillSimulator` for
/// probabilistic fill simulation. Market data comes from the same exchange WS
/// feed as the live environment; fills are synthesized from market trades.
pub struct PaperEnvironment {
    /// Shared simulation executor for order state management.
    executor: Arc<SimulationExecutor>,
    /// Configuration for this paper environment.
    config: PaperEnvironmentConfig,
    /// InfoClient for subscribing to market data channels.
    info_client: Option<crate::InfoClient>,
}

impl PaperEnvironment {
    /// Create a new `PaperEnvironment`.
    ///
    /// # Arguments
    /// - `config`: Paper environment configuration.
    /// - `info_client`: WebSocket client for market data subscriptions.
    pub fn new(config: PaperEnvironmentConfig, info_client: crate::InfoClient) -> Self {
        let executor = Arc::new(SimulationExecutor::new(false));
        Self {
            executor,
            config,
            info_client: Some(info_client),
        }
    }

    /// Create a `PaperEnvironment` with an existing executor.
    ///
    /// Useful when the executor is pre-configured or shared with other systems.
    pub fn with_executor(
        executor: Arc<SimulationExecutor>,
        config: PaperEnvironmentConfig,
        info_client: crate::InfoClient,
    ) -> Self {
        Self {
            executor,
            config,
            info_client: Some(info_client),
        }
    }
}

#[async_trait]
impl TradingEnvironment for PaperEnvironment {
    type ObservationStream = PaperObservationStream;

    async fn observation_stream(&mut self) -> Result<Self::ObservationStream> {
        use crate::Subscription;

        let info_client = self.info_client.as_mut().ok_or_else(|| {
            crate::Error::GenericRequest(
                "PaperEnvironment::observation_stream() requires InfoClient".to_string(),
            )
        })?;

        let (sender, receiver) = mpsc::unbounded_channel::<Arc<Message>>();

        // Subscribe to market data channels only (3 channels vs live's 9).
        // Paper doesn't need UserFills, OrderUpdates, OpenOrders, ActiveAssetData,
        // WebData2, or LedgerUpdates — those are synthesized from the simulation.
        info_client
            .subscribe(
                Subscription::AllMids {
                    dex: self.config.dex.clone(),
                },
                sender.clone(),
            )
            .await?;

        info_client
            .subscribe(
                Subscription::L2Book {
                    coin: self.config.asset.clone(),
                    dex: self.config.dex.clone(),
                },
                sender.clone(),
            )
            .await?;

        info_client
            .subscribe(
                Subscription::Trades {
                    coin: self.config.asset.clone(),
                    dex: self.config.dex.clone(),
                },
                sender,
            )
            .await?;

        // Create FillSimulator and move it into the stream.
        // The simulator holds its own Arc<SimulationExecutor> for order access.
        let fill_simulator = FillSimulator::new(
            Arc::clone(&self.executor),
            self.config.fill_sim_config.clone(),
        );

        Ok(PaperObservationStream {
            receiver,
            fill_simulator,
            pending: VecDeque::new(),
            asset: self.config.asset.clone(),
            user_address: self.config.user_address,
        })
    }

    // === Action execution: delegate to SimulationExecutor ===

    async fn place_order(
        &self,
        asset: &str,
        price: f64,
        size: f64,
        is_buy: bool,
        cloid: Option<String>,
        post_only: bool,
    ) -> OrderResult {
        self.executor
            .place_order(asset, price, size, is_buy, cloid, post_only)
            .await
    }

    async fn place_bulk_orders(&self, asset: &str, orders: Vec<OrderSpec>) -> Vec<OrderResult> {
        self.executor.place_bulk_orders(asset, orders).await
    }

    async fn place_ioc_reduce_order(
        &self,
        asset: &str,
        size: f64,
        is_buy: bool,
        slippage_bps: u32,
        mid_price: f64,
    ) -> OrderResult {
        self.executor
            .place_ioc_reduce_order(asset, size, is_buy, slippage_bps, mid_price)
            .await
    }

    async fn cancel_order(&self, asset: &str, oid: u64) -> CancelResult {
        self.executor.cancel_order(asset, oid).await
    }

    async fn cancel_bulk_orders(&self, asset: &str, oids: Vec<u64>) -> Vec<CancelResult> {
        self.executor.cancel_bulk_orders(asset, oids).await
    }

    async fn modify_order(
        &self,
        asset: &str,
        oid: u64,
        new_price: f64,
        new_size: f64,
        is_buy: bool,
        post_only: bool,
    ) -> ModifyResult {
        self.executor
            .modify_order(asset, oid, new_price, new_size, is_buy, post_only)
            .await
    }

    async fn modify_bulk_orders(
        &self,
        asset: &str,
        modifies: Vec<ModifySpec>,
    ) -> Vec<ModifyResult> {
        self.executor.modify_bulk_orders(asset, modifies).await
    }

    async fn sync_state(&mut self) -> Result<()> {
        // No-op for paper: no exchange state to sync.
        Ok(())
    }

    fn is_live(&self) -> bool {
        false
    }

    async fn reconnect(&self) -> Result<()> {
        if let Some(ref client) = self.info_client {
            client.reconnect().await
        } else {
            Ok(())
        }
    }
}

// ============================================================================
// PaperObservationStream
// ============================================================================

/// Stream that yields observations for paper trading.
///
/// Market data (AllMids, Trades, L2Book) comes from real WS channels.
/// Fills are synthesized by [`FillSimulator`] when market trades match resting orders.
///
/// ## Causal Ordering
///
/// On each trade:
/// 1. `FillSimulator::on_trade()` checks for fills against resting orders
/// 2. Trade observation is yielded
/// 3. Any resulting fill observations are yielded from the pending buffer
///
/// This ensures MarketMaker processes the trade (updates kappa, sigma, etc.)
/// before processing fills (updates position, PnL, etc.).
pub struct PaperObservationStream {
    /// Receiver for raw WS messages (market data only).
    receiver: mpsc::UnboundedReceiver<Arc<Message>>,
    /// Fill simulator — checks trades against resting orders.
    fill_simulator: FillSimulator,
    /// Buffer for pending observations (fill events queued after trade processing).
    pending: VecDeque<Observation>,
    /// Asset being traded.
    asset: String,
    /// User address for synthetic UserFills.
    user_address: Address,
}

impl PaperObservationStream {
    /// Process a trade through the fill simulator.
    ///
    /// For each trade in the Trades message, checks if any resting orders
    /// would be filled. Synthesized fills are added to the pending buffer
    /// as `Observation::UserFills` events.
    fn process_trades_for_fills(&mut self, trades: &crate::ws::message_types::Trades) {
        for trade in &trades.data {
            let price: f64 = trade.px.parse().unwrap_or(0.0);
            let size: f64 = trade.sz.parse().unwrap_or(0.0);
            let is_buy = trade.side == "B" || trade.side.to_lowercase() == "buy";

            if price <= 0.0 || size <= 0.0 {
                continue;
            }

            // Update executor's mid price for post-only validation
            self.fill_simulator.executor.update_mid(price);

            let market_trade = MarketTrade {
                timestamp_ns: trade.time * 1_000_000, // ms → ns
                price,
                size,
                side: if is_buy { Side::Buy } else { Side::Sell },
            };

            let fills = self.fill_simulator.on_trade(&market_trade);

            if !fills.is_empty() {
                debug!(
                    fill_count = fills.len(),
                    trade_price = price,
                    trade_size = size,
                    "Paper fill simulator triggered"
                );

                // Convert each SimulatedFill → WsFillEvent → TradeInfo → UserFills observation
                let mut trade_infos = Vec::with_capacity(fills.len());
                for fill in &fills {
                    // Apply fill to the executor's order state and get WsFillEvent
                    if let Some(ws_fill) = self.fill_simulator.executor.apply_fill(
                        fill.oid,
                        fill.fill_size,
                        fill.fill_price,
                    ) {
                        // Synthesize TradeInfo from WsFillEvent
                        let trade_info = TradeInfo {
                            coin: ws_fill.coin.clone(),
                            side: if ws_fill.is_buy {
                                "B".to_string()
                            } else {
                                "A".to_string()
                            },
                            px: ws_fill.price.to_string(),
                            sz: ws_fill.size.to_string(),
                            time: ws_fill.timestamp,
                            hash: format!("sim-{}", ws_fill.tid),
                            start_position: "0".to_string(),
                            dir: if ws_fill.is_buy {
                                "Open Long".to_string()
                            } else {
                                "Open Short".to_string()
                            },
                            closed_pnl: "0".to_string(),
                            oid: ws_fill.oid,
                            cloid: ws_fill.cloid,
                            crossed: false,
                            fee: "0".to_string(),
                            fee_token: "USDC".to_string(),
                            tid: ws_fill.tid,
                            builder_fee: None,
                        };
                        trade_infos.push(trade_info);
                    }
                }

                if !trade_infos.is_empty() {
                    let user_fills = UserFills {
                        data: UserFillsData {
                            is_snapshot: Some(false),
                            user: self.user_address,
                            fills: trade_infos,
                        },
                    };
                    self.pending.push_back(Observation::UserFills(user_fills));
                }
            }
        }
    }

    /// Process L2 book update — refresh fill simulator's queue estimators.
    fn process_book_for_queue(&mut self, book: &crate::ws::message_types::L2Book) {
        // Convert L2BookData levels to (price, size) tuples.
        // L2BookData.levels[0] = bids, levels[1] = asks
        if book.data.levels.len() >= 2 {
            let bids: Vec<(f64, f64)> = book.data.levels[0]
                .iter()
                .filter_map(|level| {
                    let px: f64 = level.px.parse().ok()?;
                    let sz: f64 = level.sz.parse().ok()?;
                    Some((px, sz))
                })
                .collect();

            let asks: Vec<(f64, f64)> = book.data.levels[1]
                .iter()
                .filter_map(|level| {
                    let px: f64 = level.px.parse().ok()?;
                    let sz: f64 = level.sz.parse().ok()?;
                    Some((px, sz))
                })
                .collect();

            self.fill_simulator.update_book(&bids, &asks);
        }
    }
}

impl Stream for PaperObservationStream {
    type Item = Observation;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Drain pending observations first (buffered fills from previous trade).
        if let Some(obs) = self.pending.pop_front() {
            trace!(
                obs = obs.label(),
                "Paper stream: yielding pending observation"
            );
            return Poll::Ready(Some(obs));
        }

        // Poll for next WS message.
        loop {
            match self.receiver.poll_recv(cx) {
                Poll::Ready(Some(msg)) => {
                    // Try to convert Message → Observation.
                    if let Ok(obs) = Observation::try_from((*msg).clone()) {
                        // Side effects: feed fill simulator before yielding.
                        match &obs {
                            Observation::Trades(trades) => {
                                self.process_trades_for_fills(trades);
                            }
                            Observation::L2Book(book) => {
                                self.process_book_for_queue(book);
                            }
                            Observation::AllMids(_) => {
                                // AllMids: update executor's mid for the asset.
                                // Extracted in handle_all_mids by the core, but
                                // we also need the executor to know mid for post-only.
                                if let Observation::AllMids(ref mids) = obs {
                                    if let Some(mid_str) = mids.data.mids.get(&self.asset) {
                                        if let Ok(mid) = mid_str.parse::<f64>() {
                                            self.fill_simulator.executor.update_mid(mid);
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                        return Poll::Ready(Some(obs));
                    }
                    // Non-observation message (NoData, Pong, etc.) — skip.
                    continue;
                }
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}
