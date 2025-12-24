#![deny(unreachable_pub)]

// Core modules
mod consts;
mod eip712;
mod errors;
mod helpers;
mod prelude;
mod req;

// New shared utilities
pub mod serde_utils;
pub mod types;

// Feature modules
mod exchange;
pub mod info;
pub mod market_maker;
mod meta;
mod signature;
pub mod ws;

// Re-exports
pub use consts::{EPSILON, LOCAL_API_URL, MAINNET_API_URL, TESTNET_API_URL};
pub use eip712::Eip712;
pub use errors::{Error, HttpErrorKind, ParseError, SigningError, WsError};
pub use exchange::*;
pub use helpers::{bps_diff, round_to_significant_and_decimal, truncate_float, BaseUrl};
pub use info::info_client::*;
pub use info::response_structs::*;
pub use market_maker::{
    EstimatorConfig, GLFTStrategy, HyperliquidExecutor, InventoryAwareStrategy, MarketMaker,
    MarketMakerConfig, MarketMakerMetricsRecorder, MarketParams, MetricsRecorder, OrderExecutor,
    OrderManager, OrderResult, ParameterEstimator, PositionTracker, Quote, QuoteConfig,
    QuotingStrategy, RiskConfig, Side, SymmetricStrategy, TrackedOrder,
};
pub use meta::{AssetContext, AssetMeta, Meta, MetaAndAssetCtxs, SpotAssetMeta, SpotMeta};
pub use serde_utils::hyperliquid_chain;
pub use types::*;
pub use ws::message_types::{
    ActiveAssetCtx, ActiveAssetData, ActiveSpotAssetCtx, AllMids, Bbo, CandleMessage, L2Book,
    Notification, OrderUpdates, Trades, User, UserFills, UserFundings, UserNonFundingLedgerUpdates,
    WebData2,
};
// Backwards-compatible alias - ws::Candle is now CandleMessage to avoid conflict with types::Candle
pub use ws::message_types::Candle as WsCandle;
pub use ws::{Message, Subscription};
