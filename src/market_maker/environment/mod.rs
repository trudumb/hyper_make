//! Environment abstraction for paper/live trading unification.
//!
//! # Mathematical Structure
//!
//! ```text
//! S₀ →[T,π,E_paper]ⁿ→ S_paper →[φ]→ P →[ψ]→ S_live →[T,π,E_live]^∞→ execution
//! ```
//!
//! where **T** (state transition) and **π** (policy) are invariant under
//! the substitution E_paper ↔ E_live. This module encodes the substitution:
//!
//! - [`Observation`]: The observation space **O** = O_market ⊕ O_fill ⊕ O_account
//! - [`TradingEnvironment`]: The environment trait **E**: S × A → P(O)

pub mod live;
pub mod paper;
pub mod stream;

use async_trait::async_trait;
use futures_util::Stream;

use crate::market_maker::{CancelResult, ModifyResult, ModifySpec, OrderResult, OrderSpec};
use crate::prelude::Result;
use crate::ws::message_types::{
    ActiveAssetData, AllMids, L2Book, OpenOrders, OrderUpdates, Trades, UserFills,
    UserNonFundingLedgerUpdates, WebData2,
};

// Re-export core executor types for convenience
pub use crate::market_maker::{HyperliquidExecutor, OrderExecutor};

/// Observation: the unified observation space consumed by MarketMaker core.
///
/// Both paper and live environments produce `Observation` values.
/// The core (`MarketMaker`) consumes them through `handle_observation()`.
///
/// Market data observations (AllMids, Trades, L2Book) use existing WS types
/// because both environments receive identical market data from the exchange.
/// Fill observations also use the existing UserFills type — paper environments
/// synthesize fills using the same structure.
#[derive(Debug, Clone)]
pub enum Observation {
    // === Market data (same exchange WS feed for both environments) ===
    /// Mid price update for all assets.
    AllMids(AllMids),
    /// Trade executions on the exchange.
    Trades(Trades),
    /// L2 order book snapshot.
    L2Book(L2Book),

    // === Fills (live: UserFills WS, paper: synthesized from FillSimulator) ===
    /// Fill events — same structure for both live and simulated fills.
    UserFills(UserFills),

    // === Order state (live: WS, paper: synthesized from SimulationExecutor) ===
    /// Individual order state changes.
    OrderUpdates(OrderUpdates),
    /// Full snapshot of open orders.
    OpenOrders(OpenOrders),

    // === Account/limits (live: WS, paper: synthesized) ===
    /// Exchange position/margin limits for the traded asset.
    ActiveAssetData(ActiveAssetData),
    /// Account-wide margin and position data.
    WebData2(Box<WebData2>),
    /// Spot balance ledger changes.
    LedgerUpdate(UserNonFundingLedgerUpdates),

    // === Cross-venue (optional Binance feed, same for both environments) ===
    /// Binance mid price update (for lead-lag skew).
    CrossVenuePrice { price: f64, timestamp_ms: u64 },
    /// Binance trade (for cross-venue flow analysis).
    CrossVenueTrade {
        price: f64,
        size: f64,
        is_buy: bool,
        timestamp_ms: u64,
    },
}

impl Observation {
    /// Returns a short label for logging/metrics.
    pub fn label(&self) -> &'static str {
        match self {
            Self::AllMids(_) => "AllMids",
            Self::Trades(_) => "Trades",
            Self::L2Book(_) => "L2Book",
            Self::UserFills(_) => "UserFills",
            Self::OrderUpdates(_) => "OrderUpdates",
            Self::OpenOrders(_) => "OpenOrders",
            Self::ActiveAssetData(_) => "ActiveAssetData",
            Self::WebData2(_) => "WebData2",
            Self::LedgerUpdate(_) => "LedgerUpdate",
            Self::CrossVenuePrice { .. } => "CrossVenuePrice",
            Self::CrossVenueTrade { .. } => "CrossVenueTrade",
        }
    }
}

/// Convert a WS [`Message`] into an [`Observation`], if applicable.
///
/// Returns `None` for message types that don't map to observations
/// (e.g., `NoData`, `Pong`, `SubscriptionResponse`).
impl TryFrom<crate::Message> for Observation {
    type Error = ();

    fn try_from(msg: crate::Message) -> std::result::Result<Self, ()> {
        match msg {
            crate::Message::AllMids(m) => Ok(Self::AllMids(m)),
            crate::Message::Trades(t) => Ok(Self::Trades(t)),
            crate::Message::L2Book(b) => Ok(Self::L2Book(b)),
            crate::Message::UserFills(f) => Ok(Self::UserFills(f)),
            crate::Message::OrderUpdates(o) => Ok(Self::OrderUpdates(o)),
            crate::Message::OpenOrders(o) => Ok(Self::OpenOrders(o)),
            crate::Message::ActiveAssetData(a) => Ok(Self::ActiveAssetData(a)),
            crate::Message::WebData2(w) => Ok(Self::WebData2(Box::new(w))),
            crate::Message::UserNonFundingLedgerUpdates(l) => Ok(Self::LedgerUpdate(l)),
            // NoData, Pong, SubscriptionResponse, etc. — not observations
            _ => Err(()),
        }
    }
}

/// TradingEnvironment: the boundary trait between MarketMaker core and execution.
///
/// Absorbs [`OrderExecutor`] — one trait captures the full environment boundary.
/// The core calls environment methods for both observation and action.
///
/// # Implementations
///
/// - [`LiveEnvironment`](live::LiveEnvironment): wraps `HyperliquidExecutor` + full WS subscription
/// - `PaperEnvironment` (Phase 4): wraps `SimulationExecutor` + `FillSimulator` + market-data WS
#[async_trait]
pub trait TradingEnvironment: Send + Sync + 'static {
    /// The observation stream type produced by this environment.
    type ObservationStream: Stream<Item = Observation> + Send + Unpin;

    /// Create the observation stream (subscribes to data sources).
    ///
    /// For live: subscribes to all 9 WS channels, translates `Message` → `Observation`.
    /// For paper: subscribes to market data channels, synthesizes fills from FillSimulator.
    async fn observation_stream(&mut self) -> Result<Self::ObservationStream>;

    // === Action execution (absorbed from OrderExecutor) ===

    /// Place a single limit order.
    async fn place_order(
        &self,
        asset: &str,
        price: f64,
        size: f64,
        is_buy: bool,
        cloid: Option<String>,
        post_only: bool,
    ) -> OrderResult;

    /// Place multiple orders in a single API call.
    async fn place_bulk_orders(&self, asset: &str, orders: Vec<OrderSpec>) -> Vec<OrderResult>;

    /// Place an IOC order for position reduction.
    async fn place_ioc_reduce_order(
        &self,
        asset: &str,
        size: f64,
        is_buy: bool,
        slippage_bps: u32,
        mid_price: f64,
    ) -> OrderResult;

    /// Cancel a single order.
    async fn cancel_order(&self, asset: &str, oid: u64) -> CancelResult;

    /// Cancel multiple orders in a single API call.
    async fn cancel_bulk_orders(&self, asset: &str, oids: Vec<u64>) -> Vec<CancelResult>;

    /// Modify an existing order in place.
    async fn modify_order(
        &self,
        asset: &str,
        oid: u64,
        new_price: f64,
        new_size: f64,
        is_buy: bool,
        post_only: bool,
    ) -> ModifyResult;

    /// Modify multiple orders in a single API call.
    async fn modify_bulk_orders(&self, asset: &str, modifies: Vec<ModifySpec>)
        -> Vec<ModifyResult>;

    /// Startup sync (live: sync position + cancel all orders; paper: no-op).
    async fn sync_state(&mut self) -> Result<()>;

    /// Whether this is a live environment (controls safety sync, margin refresh).
    fn is_live(&self) -> bool;

    /// Force reconnect the underlying WebSocket data feed.
    ///
    /// Default: no-op (returns Ok). Environments with their own InfoClient
    /// (e.g., PaperEnvironment) should override to trigger WS reconnection
    /// when the connection supervisor detects stale data.
    async fn reconnect(&self) -> Result<()> {
        Ok(())
    }
}
