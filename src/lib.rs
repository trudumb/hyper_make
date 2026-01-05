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
    // Phase 6: Structured Logging
    init_logging,
    targets as log_targets,
    // Tier 1: Adverse Selection
    AdverseSelectionConfig,
    AdverseSelectionEstimator,
    AdverseSelectionSummary,
    // Data Quality Monitoring
    AnomalyType,
    // HIP-3 Runtime Config
    AssetRuntimeConfig,
    // Tier 1: Liquidation Cascade
    CascadeDirection,
    // WebSocket Reconnection
    ConnectionHealthMonitor,
    ConnectionState,
    DataQualityConfig,
    DataQualityMonitor,
    // Dynamic Risk Configuration
    DynamicRiskConfig,
    // Core types
    EstimatorConfig,
    // Tier 2: Funding Rate
    FundingConfig,
    FundingRateEstimator,
    FundingSummary,
    GLFTStrategy,
    // Tier 2: Hawkes Order Flow
    HawkesConfig,
    HawkesOrderFlowEstimator,
    HawkesSummary,
    HealthSummary,
    HyperliquidExecutor,
    // Statistical Impulse Control
    ImpulseControlConfig,
    InventoryAwareStrategy,
    // Production Safety: Kill Switch
    KillReason,
    KillSwitch,
    KillSwitchConfig,
    KillSwitchState,
    KillSwitchSummary,
    // Ladder quoting
    Ladder,
    LadderAction,
    LadderConfig,
    LadderLevel,
    LadderParams,
    LadderStrategy,
    LiquidationCascadeDetector,
    LiquidationConfig,
    LiquidationSummary,
    LogConfig,
    LogFormat,
    // Production: Margin-Aware Sizing
    MarginAwareSizer,
    MarginConfig,
    MarginState,
    MarginSummary as MarginSizerSummary,
    MarketMaker,
    MarketMakerConfig,
    MarketMakerMetricsRecorder,
    MarketParams,
    MetricsRecorder,
    // Prometheus Metrics
    MetricsSummary,
    ModifyResult,
    // Configuration
    MonitoringConfig,
    OrderExecutor,
    OrderManager,
    OrderResult,
    ParameterEstimator,
    // Tier 2: P&L Attribution
    PnLComponents,
    PnLConfig,
    PnLSummary,
    PnLTracker,
    // Phase 4: Position Reconciliation
    PositionReconciler,
    PositionTracker,
    PrometheusMetrics,
    // Tier 1: Queue Position
    QueueConfig,
    QueuePositionTracker,
    QueueSummary,
    Quote,
    QuoteConfig,
    QuotingStrategy,
    // Reconciliation config for smart ladder updates
    ReconcileConfig,
    ReconciliationConfig,
    ReconciliationTrigger,
    ReconnectionConfig,
    // Phase 3: Recovery Manager
    RecoveryConfig,
    RecoveryManager,
    RecoveryState,
    RejectionRateLimitConfig,
    // Phase 5: Rejection Rate Limiting
    RejectionRateLimiter,
    RiskConfig,
    Side,
    SizingResult,
    // Tier 2: Spread Process
    SpreadConfig,
    SpreadProcessEstimator,
    // Spread profiles for different market types
    SpreadProfile,
    SpreadRegime,
    SpreadSummary,
    // Stochastic Module Integration
    StochasticConfig,
    SymmetricStrategy,
    TrackedOrder,
    // Volatility Regime
    VolatilityRegime,
};
pub use meta::{
    AssetContext, AssetMeta, CollateralInfo, Meta, MetaAndAssetCtxs, PerpDex, PerpDexLimits,
    SpotAssetMeta, SpotMeta,
};
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
