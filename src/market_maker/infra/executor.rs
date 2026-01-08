//! Order execution abstraction for the market maker.

use async_trait::async_trait;
use tracing::{debug, error, info, warn};

use crate::{
    ClientCancelRequest, ClientLimit, ClientModifyRequest, ClientOrder, ClientOrderRequest,
    ExchangeClient, ExchangeDataStatus, ExchangeResponseStatus,
};

use crate::market_maker::config::MetricsRecorder;

/// Result of placing an order.
#[derive(Debug, Clone)]
pub struct OrderResult {
    /// Amount that is now resting (may be less than requested if partially filled)
    pub resting_size: f64,
    /// Order ID from the exchange (0 if order failed)
    pub oid: u64,
    /// Whether the order was immediately filled
    pub filled: bool,
    /// Client Order ID (echoed back for fill matching)
    pub cloid: Option<String>,
    /// Error message if order placement failed (Phase 5: for rate limiting)
    pub error: Option<String>,
}

impl OrderResult {
    /// Create a failed order result.
    pub fn failed() -> Self {
        Self {
            resting_size: 0.0,
            oid: 0,
            filled: false,
            cloid: None,
            error: None,
        }
    }

    /// Create a failed order result with error message.
    pub fn failed_with_error(error: String) -> Self {
        Self {
            resting_size: 0.0,
            oid: 0,
            filled: false,
            cloid: None,
            error: Some(error),
        }
    }

    /// Create a failed order result with CLOID preserved.
    pub fn failed_with_cloid(cloid: Option<String>) -> Self {
        Self {
            resting_size: 0.0,
            oid: 0,
            filled: false,
            cloid,
            error: None,
        }
    }

    /// Create a failed order result with CLOID and error.
    pub fn failed_with_cloid_and_error(cloid: Option<String>, error: String) -> Self {
        Self {
            resting_size: 0.0,
            oid: 0,
            filled: false,
            cloid,
            error: Some(error),
        }
    }
}

/// Result of modifying an order.
#[derive(Debug)]
pub struct ModifyResult {
    /// New order ID (may be same as original)
    pub oid: u64,
    /// New resting size
    pub resting_size: f64,
    /// Whether the modify was successful
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

impl ModifyResult {
    /// Create a failed modify result.
    pub fn failed(error: String) -> Self {
        Self {
            oid: 0,
            resting_size: 0.0,
            success: false,
            error: Some(error),
        }
    }

    /// Create a successful modify result.
    pub fn success(oid: u64, resting_size: f64) -> Self {
        Self {
            oid,
            resting_size,
            success: true,
            error: None,
        }
    }
}

/// Result of cancelling an order.
///
/// Distinguishes between different cancel outcomes to allow proper order tracking:
/// - `Cancelled`: Order was successfully cancelled (safe to remove from tracking)
/// - `AlreadyCancelled`: Order was already cancelled (safe to remove from tracking)
/// - `AlreadyFilled`: Order was already filled (DO NOT remove - fill notification will handle it)
/// - `Failed`: Cancel failed for other reasons (keep in tracking, may retry)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelResult {
    /// Order was successfully cancelled
    Cancelled,
    /// Order was already cancelled (no longer exists)
    AlreadyCancelled,
    /// Order was already filled before cancel arrived
    AlreadyFilled,
    /// Cancel failed for other reasons
    Failed,
}

impl CancelResult {
    /// Returns true if the order is gone (cancelled, already cancelled, or already filled)
    pub fn order_is_gone(&self) -> bool {
        matches!(
            self,
            CancelResult::Cancelled | CancelResult::AlreadyCancelled | CancelResult::AlreadyFilled
        )
    }

    /// Returns true if it's safe to remove the order from tracking
    /// (i.e., the order won't generate any more fill notifications)
    pub fn safe_to_remove_tracking(&self) -> bool {
        matches!(
            self,
            CancelResult::Cancelled | CancelResult::AlreadyCancelled
        )
    }
}

/// Order specification for bulk placement.
#[derive(Debug, Clone)]
pub struct OrderSpec {
    /// Order price
    pub price: f64,
    /// Order size
    pub size: f64,
    /// Whether this is a buy order
    pub is_buy: bool,
    /// Client Order ID (UUID) for deterministic fill tracking.
    /// If provided, this CLOID is sent to the exchange and returned in fill notifications.
    pub cloid: Option<String>,
    /// Whether to use ALO (Add Liquidity Only) / Post-Only TIF.
    pub post_only: bool,
}

impl OrderSpec {
    /// Create a new order spec without CLOID (legacy).
    pub fn new(price: f64, size: f64, is_buy: bool, post_only: bool) -> Self {
        Self {
            price,
            size,
            is_buy,
            cloid: None,
            post_only,
        }
    }

    /// Create a new order spec with CLOID for deterministic tracking.
    pub fn with_cloid(
        price: f64,
        size: f64,
        is_buy: bool,
        cloid: String,
        post_only: bool,
    ) -> Self {
        Self {
            price,
            size,
            is_buy,
            cloid: Some(cloid),
            post_only,
        }
    }
}

/// Specification for order modification.
/// Used with bulk modify to preserve queue position when updating multiple orders.
#[derive(Debug, Clone)]
pub struct ModifySpec {
    /// Order ID to modify
    pub oid: u64,
    /// New limit price
    pub new_price: f64,
    /// New order size
    pub new_size: f64,
    /// Whether this is a buy order
    pub is_buy: bool,
    /// Whether to use ALO (Add Liquidity Only) / Post-Only TIF.
    pub post_only: bool,
}

/// Trait for order execution.
/// Abstracts the exchange client to enable testing and mocking.
#[async_trait]
pub trait OrderExecutor: Send + Sync {
    /// Place a limit order.
    ///
    /// # Arguments
    /// - `asset`: Asset symbol
    /// - `price`: Limit price
    /// - `size`: Order size
    /// - `is_buy`: Whether this is a buy order
    /// - `cloid`: Optional client order ID. If provided, uses this CLOID for tracking.
    ///            If None, generates a new UUID. Passing a CLOID allows the caller
    ///            to pre-register pending orders for deterministic matching.
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
    ///
    /// This is much faster than placing orders sequentially when submitting
    /// multiple ladder levels. Returns results for each order in the same order
    /// as the input specs.
    async fn place_bulk_orders(&self, asset: &str, orders: Vec<OrderSpec>) -> Vec<OrderResult>;

    /// Place an IOC (Immediate or Cancel) order for position reduction (Phase 3 Fix).
    ///
    /// This is used when reduce-only mode is stuck and we need to aggressively
    /// reduce position with a market-like order that crosses the spread.
    ///
    /// # Arguments
    /// - `asset`: Asset symbol
    /// - `size`: Size to reduce (always positive, direction determined by `is_buy`)
    /// - `is_buy`: Whether to buy (true) or sell (false)
    /// - `slippage_bps`: Allowed slippage in basis points (e.g., 50 = 0.5%)
    /// - `mid_price`: Current mid price (used to calculate limit price)
    ///
    /// # Returns
    /// OrderResult with the outcome of the IOC order
    async fn place_ioc_reduce_order(
        &self,
        asset: &str,
        size: f64,
        is_buy: bool,
        slippage_bps: u32,
        mid_price: f64,
    ) -> OrderResult;

    /// Cancel an order.
    ///
    /// Returns a `CancelResult` indicating what happened:
    /// - `Cancelled`: Order was cancelled successfully
    /// - `AlreadyCancelled`: Order was already cancelled
    /// - `AlreadyFilled`: Order was filled before cancel arrived (keep tracking!)
    /// - `Failed`: Cancel failed for other reasons
    async fn cancel_order(&self, asset: &str, oid: u64) -> CancelResult;

    /// Cancel multiple orders in a single API call.
    ///
    /// RATE LIMIT OPTIMIZATION: Uses bulk cancel to reduce address-based rate limit consumption.
    /// Per Hyperliquid docs: "A batched request with n orders is treated as one request for
    /// IP based rate limiting, but as n requests for address-based rate limiting."
    ///
    /// However, cancels have a higher limit: min(limit + 100000, limit * 2).
    /// Bulk cancel still saves on IP rate limits (1 request vs n requests).
    ///
    /// # Arguments
    /// - `asset`: Asset symbol
    /// - `oids`: List of order IDs to cancel
    ///
    /// # Returns
    /// Vec of CancelResult for each order in the same order as input
    async fn cancel_bulk_orders(&self, asset: &str, oids: Vec<u64>) -> Vec<CancelResult>;

    /// Modify an existing order.
    ///
    /// Updates the price and/or size of an order in place, preserving queue position
    /// where possible. Falls back to cancel+replace if modify is not supported.
    ///
    /// # Arguments
    /// - `asset`: Asset symbol
    /// - `oid`: Order ID to modify
    /// - `new_price`: New limit price
    /// - `new_size`: New order size
    /// - `is_buy`: Whether this is a buy order
    ///
    /// # Returns
    /// ModifyResult with success status and new order details
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
    ///
    /// RATE LIMIT OPTIMIZATION: Uses bulk modify to reduce API calls.
    /// Preserves queue position where possible, which is critical for spread capturing.
    ///
    /// # Arguments
    /// - `asset`: Asset symbol
    /// - `modifies`: List of modification specifications
    ///
    /// # Returns
    /// Vec of ModifyResult for each order in the same order as input
    async fn modify_bulk_orders(&self, asset: &str, modifies: Vec<ModifySpec>)
        -> Vec<ModifyResult>;
}

/// Hyperliquid exchange executor.
pub struct HyperliquidExecutor {
    client: ExchangeClient,
    metrics: MetricsRecorder,
}

impl HyperliquidExecutor {
    /// Create a new Hyperliquid executor.
    pub fn new(client: ExchangeClient, metrics: MetricsRecorder) -> Self {
        Self { client, metrics }
    }
}

#[async_trait]
impl OrderExecutor for HyperliquidExecutor {
    async fn place_order(
        &self,
        asset: &str,
        price: f64,
        size: f64,
        is_buy: bool,
        caller_cloid: Option<String>,
        post_only: bool,
    ) -> OrderResult {
        let side = if is_buy { "BUY" } else { "SELL" };

        // Use caller-provided CLOID if available, otherwise generate one.
        // This allows the caller to pre-register pending orders with a known CLOID
        // for deterministic matching when the API response arrives.
        let cloid_str = caller_cloid.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let cloid = uuid::Uuid::parse_str(&cloid_str).unwrap_or_else(|_| uuid::Uuid::new_v4());

        let order = self
            .client
            .order(
                ClientOrderRequest {
                    asset: asset.to_string(),
                    is_buy,
                    reduce_only: false,
                    limit_px: price,
                    sz: size,
                    cloid: Some(cloid),
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: if post_only {
                            "Alo".to_string()
                        } else {
                            "Gtc".to_string()
                        },
                    }),
                },
                None,
            )
            .await;

        match order {
            Ok(order) => match order {
                ExchangeResponseStatus::Ok(order) => {
                    if let Some(order) = order.data {
                        if !order.statuses.is_empty() {
                            match order.statuses[0].clone() {
                                ExchangeDataStatus::Filled(order) => {
                                    info!(
                                        "Order filled immediately: oid={} cloid={}",
                                        order.oid, cloid_str
                                    );
                                    if let Some(ref m) = self.metrics {
                                        m.record_order_placed();
                                    }
                                    return OrderResult {
                                        resting_size: size,
                                        oid: order.oid,
                                        filled: true,
                                        cloid: Some(cloid_str),
                                        error: None,
                                    };
                                }
                                ExchangeDataStatus::Resting(order) => {
                                    info!("Order resting: oid={} cloid={}", order.oid, cloid_str);
                                    if let Some(ref m) = self.metrics {
                                        m.record_order_placed();
                                    }
                                    return OrderResult {
                                        resting_size: size,
                                        oid: order.oid,
                                        filled: false,
                                        cloid: Some(cloid_str),
                                        error: None,
                                    };
                                }
                                ExchangeDataStatus::Error(e) => {
                                    error!(
                                        "Order rejected: asset={} side={} sz={} price={} cloid={} error={}",
                                        asset, side, size, price, cloid_str, e
                                    );
                                }
                                _ => {
                                    warn!("Unexpected order status: {:?}", order.statuses[0]);
                                }
                            }
                        } else {
                            error!("Exchange data statuses is empty when placing order: {order:?}")
                        }
                    } else {
                        error!("Exchange response data is empty when placing order: {order:?}")
                    }
                }
                ExchangeResponseStatus::Err(e) => {
                    error!(
                        "Order failed: asset={} side={} sz={} price={} cloid={} error={}",
                        asset, side, size, price, cloid_str, e
                    );
                }
            },
            Err(e) => {
                error!(
                    "Order request failed: asset={} side={} sz={} price={} cloid={} error={}",
                    asset, side, size, price, cloid_str, e
                );
            }
        }

        OrderResult::failed_with_cloid(Some(cloid_str))
    }

    #[tracing::instrument(name = "order_placement", skip_all, fields(count = orders.len(), asset = %asset))]
    async fn place_bulk_orders(&self, asset: &str, orders: Vec<OrderSpec>) -> Vec<OrderResult> {
        if orders.is_empty() {
            return vec![];
        }

        // Generate CLOIDs for orders that don't have them
        // Keep track of the CLOIDs we're using for each order
        let cloids: Vec<Option<String>> = orders
            .iter()
            .map(|spec| {
                spec.cloid
                    .clone()
                    .or_else(|| Some(uuid::Uuid::new_v4().to_string()))
            })
            .collect();

        // Convert to ClientOrderRequest vec with CLOIDs
        let order_requests: Vec<ClientOrderRequest> = orders
            .iter()
            .zip(cloids.iter())
            .map(|(spec, cloid)| ClientOrderRequest {
                asset: asset.to_string(),
                is_buy: spec.is_buy,
                reduce_only: false,
                limit_px: spec.price,
                sz: spec.size,
                cloid: cloid.as_ref().and_then(|c| uuid::Uuid::parse_str(c).ok()),
                order_type: ClientOrder::Limit(ClientLimit {
                    tif: if spec.post_only {
                        "Alo".to_string()
                    } else {
                        "Gtc".to_string()
                    },
                }),
            })
            .collect();

        let num_orders = order_requests.len();
        debug!("Placing {} orders in bulk with CLOIDs", num_orders);

        let result = self.client.bulk_order(order_requests, None).await;

        match result {
            Ok(response) => match response {
                ExchangeResponseStatus::Ok(data) => {
                    if let Some(data) = data.data {
                        let mut results = Vec::with_capacity(num_orders);
                        let mut resting_oids = Vec::new();
                        let mut filled_oids = Vec::new();
                        let mut error_count = 0u32;

                        // Each status corresponds to one order in sequence
                        for (i, status) in data.statuses.iter().enumerate() {
                            let spec = &orders[i];
                            let cloid = cloids[i].clone();
                            let side = if spec.is_buy { "BUY" } else { "SELL" };

                            match status {
                                ExchangeDataStatus::Filled(order) => {
                                    filled_oids.push(order.oid);
                                    if let Some(ref m) = self.metrics {
                                        m.record_order_placed();
                                    }
                                    results.push(OrderResult {
                                        resting_size: spec.size,
                                        oid: order.oid,
                                        filled: true,
                                        cloid,
                                        error: None,
                                    });
                                }
                                ExchangeDataStatus::Resting(order) => {
                                    resting_oids.push(order.oid);
                                    if let Some(ref m) = self.metrics {
                                        m.record_order_placed();
                                    }
                                    results.push(OrderResult {
                                        resting_size: spec.size,
                                        oid: order.oid,
                                        filled: false,
                                        cloid,
                                        error: None,
                                    });
                                }
                                ExchangeDataStatus::Error(e) => {
                                    error!(
                                        "Bulk order {} rejected: asset={} side={} sz={} price={} cloid={:?} error={}",
                                        i, asset, side, spec.size, spec.price, cloid, e
                                    );
                                    error_count += 1;
                                    // Phase 5: Capture error for rate limiter
                                    results.push(OrderResult::failed_with_cloid_and_error(
                                        cloid,
                                        e.clone(),
                                    ));
                                }
                                _ => {
                                    warn!("Unexpected bulk order {} status: {:?}", i, status);
                                    error_count += 1;
                                    results.push(OrderResult::failed_with_cloid(cloid));
                                }
                            }
                        }

                        // Log consolidated summary instead of per-order logs
                        if !resting_oids.is_empty() || !filled_oids.is_empty() {
                            info!(
                                resting = resting_oids.len(),
                                filled = filled_oids.len(),
                                errors = error_count,
                                "Bulk order placed"
                            );
                        }

                        // If we got fewer statuses than orders, fill remaining with failures
                        while results.len() < num_orders {
                            let cloid = cloids.get(results.len()).cloned().flatten();
                            results.push(OrderResult::failed_with_cloid(cloid));
                        }

                        return results;
                    }
                    error!("Bulk order response data is empty");
                }
                ExchangeResponseStatus::Err(e) => {
                    error!("Bulk order failed: {}", e);
                }
            },
            Err(e) => {
                error!("Bulk order request failed: {}", e);
            }
        }

        // All failed - preserve CLOIDs for cleanup
        orders
            .iter()
            .zip(cloids.iter())
            .map(|(_, cloid)| OrderResult::failed_with_cloid(cloid.clone()))
            .collect()
    }

    async fn place_ioc_reduce_order(
        &self,
        asset: &str,
        size: f64,
        is_buy: bool,
        slippage_bps: u32,
        mid_price: f64,
    ) -> OrderResult {
        let side = if is_buy { "BUY" } else { "SELL" };

        // Calculate aggressive limit price with slippage
        // For buys: we're willing to pay up to slippage_bps above mid
        // For sells: we're willing to accept down to slippage_bps below mid
        let slippage_mult = slippage_bps as f64 / 10_000.0;
        let limit_price = if is_buy {
            mid_price * (1.0 + slippage_mult) // Pay more to buy
        } else {
            mid_price * (1.0 - slippage_mult) // Accept less to sell
        };

        let cloid = uuid::Uuid::new_v4();
        let cloid_str = cloid.to_string();

        info!(
            side = %side,
            size = %format!("{:.6}", size),
            limit_price = %format!("{:.6}", limit_price),
            mid_price = %format!("{:.6}", mid_price),
            slippage_bps = slippage_bps,
            cloid = %cloid_str,
            "Placing IOC reduce order (Phase 3 recovery)"
        );

        let order = self
            .client
            .order(
                ClientOrderRequest {
                    asset: asset.to_string(),
                    is_buy,
                    reduce_only: true, // CRITICAL: reduce_only to avoid increasing position
                    limit_px: limit_price,
                    sz: size,
                    cloid: Some(cloid),
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: "Ioc".to_string(), // IOC = Immediate or Cancel
                    }),
                },
                None,
            )
            .await;

        match order {
            Ok(order) => match order {
                ExchangeResponseStatus::Ok(order) => {
                    if let Some(order) = order.data {
                        if !order.statuses.is_empty() {
                            match order.statuses[0].clone() {
                                ExchangeDataStatus::Filled(order) => {
                                    info!(
                                        "IOC reduce order filled: oid={} cloid={}",
                                        order.oid, cloid_str
                                    );
                                    if let Some(ref m) = self.metrics {
                                        m.record_order_placed();
                                    }
                                    return OrderResult {
                                        resting_size: 0.0, // IOC won't rest
                                        oid: order.oid,
                                        filled: true,
                                        cloid: Some(cloid_str),
                                        error: None,
                                    };
                                }
                                ExchangeDataStatus::Resting(order) => {
                                    // Shouldn't happen for IOC, but handle it
                                    warn!(
                                        "IOC order resting (unexpected): oid={} cloid={}",
                                        order.oid, cloid_str
                                    );
                                    if let Some(ref m) = self.metrics {
                                        m.record_order_placed();
                                    }
                                    return OrderResult {
                                        resting_size: size,
                                        oid: order.oid,
                                        filled: false,
                                        cloid: Some(cloid_str),
                                        error: None,
                                    };
                                }
                                ExchangeDataStatus::Error(e) => {
                                    error!(
                                        "IOC reduce order rejected: asset={} side={} sz={} price={} cloid={} error={}",
                                        asset, side, size, limit_price, cloid_str, e
                                    );
                                }
                                _ => {
                                    warn!("Unexpected IOC order status: {:?}", order.statuses[0]);
                                }
                            }
                        } else {
                            error!("IOC order response statuses empty");
                        }
                    } else {
                        error!("IOC order response data is empty");
                    }
                }
                ExchangeResponseStatus::Err(e) => {
                    error!(
                        "IOC reduce order failed: asset={} side={} sz={} price={} cloid={} error={}",
                        asset, side, size, limit_price, cloid_str, e
                    );
                }
            },
            Err(e) => {
                error!(
                    "IOC reduce order request failed: asset={} side={} error={}",
                    asset, side, e
                );
            }
        }

        OrderResult::failed_with_cloid(Some(cloid_str))
    }

    async fn cancel_order(&self, asset: &str, oid: u64) -> CancelResult {
        let cancel = self
            .client
            .cancel(
                ClientCancelRequest {
                    asset: asset.to_string(),
                    oid,
                },
                None,
            )
            .await;

        match cancel {
            Ok(cancel) => match cancel {
                ExchangeResponseStatus::Ok(cancel) => {
                    if let Some(cancel) = cancel.data {
                        if !cancel.statuses.is_empty() {
                            match cancel.statuses[0].clone() {
                                ExchangeDataStatus::Success => {
                                    if let Some(ref m) = self.metrics {
                                        m.record_order_cancelled();
                                    }
                                    return CancelResult::Cancelled;
                                }
                                ExchangeDataStatus::Error(e) => {
                                    // Distinguish between "already cancelled" and "already filled"
                                    // CRITICAL: If filled, DO NOT remove from tracking - fill will arrive via WebSocket
                                    if e.contains("filled") {
                                        info!("Order already filled: oid={} - keeping in tracking for fill", oid);
                                        return CancelResult::AlreadyFilled;
                                    }
                                    if e.contains("already canceled")
                                        || e.contains("does not exist")
                                    {
                                        info!("Order already cancelled: oid={}", oid);
                                        return CancelResult::AlreadyCancelled;
                                    }
                                    error!("Cancel error: {e}");
                                }
                                _ => {
                                    warn!("Unexpected cancel status: {:?}", cancel.statuses[0]);
                                }
                            }
                        } else {
                            error!("Exchange data statuses is empty when cancelling: {cancel:?}");
                        }
                    } else {
                        error!("Exchange response data is empty when cancelling: {cancel:?}");
                    }
                }
                ExchangeResponseStatus::Err(e) => error!("Cancel failed: {e}"),
            },
            Err(e) => error!("Cancel request failed: {e}"),
        }

        CancelResult::Failed
    }

    async fn cancel_bulk_orders(&self, asset: &str, oids: Vec<u64>) -> Vec<CancelResult> {
        if oids.is_empty() {
            return vec![];
        }

        info!("Bulk cancelling {} orders", oids.len());

        // Convert to cancel requests
        let cancel_requests: Vec<ClientCancelRequest> = oids
            .iter()
            .map(|&oid| ClientCancelRequest {
                asset: asset.to_string(),
                oid,
            })
            .collect();

        let result = self.client.bulk_cancel(cancel_requests, None).await;

        match result {
            Ok(response) => match response {
                ExchangeResponseStatus::Ok(data) => {
                    if let Some(data) = data.data {
                        let mut results = Vec::with_capacity(oids.len());

                        for (i, status) in data.statuses.iter().enumerate() {
                            let oid = oids.get(i).copied().unwrap_or(0);

                            match status {
                                ExchangeDataStatus::Success => {
                                    if let Some(ref m) = self.metrics {
                                        m.record_order_cancelled();
                                    }
                                    results.push(CancelResult::Cancelled);
                                }
                                ExchangeDataStatus::Error(e) => {
                                    // Distinguish between "already cancelled" and "already filled"
                                    if e.contains("filled") {
                                        info!(
                                            "Bulk cancel: oid={} already filled - keeping in tracking",
                                            oid
                                        );
                                        results.push(CancelResult::AlreadyFilled);
                                    } else if e.contains("already canceled")
                                        || e.contains("does not exist")
                                    {
                                        info!("Bulk cancel: oid={} already cancelled", oid);
                                        results.push(CancelResult::AlreadyCancelled);
                                    } else {
                                        error!("Bulk cancel error for oid={}: {}", oid, e);
                                        results.push(CancelResult::Failed);
                                    }
                                }
                                _ => {
                                    warn!(
                                        "Unexpected bulk cancel status for oid={}: {:?}",
                                        oid, status
                                    );
                                    results.push(CancelResult::Failed);
                                }
                            }
                        }

                        // Fill remaining with Failed if we got fewer responses
                        while results.len() < oids.len() {
                            results.push(CancelResult::Failed);
                        }

                        return results;
                    }
                    error!("Bulk cancel response data is empty");
                }
                ExchangeResponseStatus::Err(e) => {
                    error!("Bulk cancel failed: {}", e);
                }
            },
            Err(e) => {
                error!("Bulk cancel request failed: {}", e);
            }
        }

        // All failed
        vec![CancelResult::Failed; oids.len()]
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
        let side = if is_buy { "BUY" } else { "SELL" };

        let modify_request = ClientModifyRequest {
            oid,
            order: ClientOrderRequest {
                asset: asset.to_string(),
                is_buy,
                reduce_only: false,
                limit_px: new_price,
                sz: new_size,
                cloid: None,
                order_type: ClientOrder::Limit(ClientLimit {
                    tif: if post_only {
                        "Alo".to_string()
                    } else {
                        "Gtc".to_string()
                    },
                }),
            },
        };

        let result = self.client.modify(modify_request, None).await;

        match result {
            Ok(response) => match response {
                ExchangeResponseStatus::Ok(data) => {
                    if let Some(data) = data.data {
                        if !data.statuses.is_empty() {
                            match data.statuses[0].clone() {
                                ExchangeDataStatus::Filled(order) => {
                                    info!("Modified order filled immediately: oid={}", order.oid);
                                    if let Some(ref m) = self.metrics {
                                        m.record_order_placed();
                                    }
                                    return ModifyResult::success(order.oid, new_size);
                                }
                                ExchangeDataStatus::Resting(order) => {
                                    info!(
                                        "Order modified successfully: oid={} -> new price={} size={}",
                                        order.oid, new_price, new_size
                                    );
                                    if let Some(ref m) = self.metrics {
                                        m.record_order_placed();
                                    }
                                    return ModifyResult::success(order.oid, new_size);
                                }
                                ExchangeDataStatus::Error(e) => {
                                    warn!(
                                        "Modify rejected: asset={} oid={} side={} error={}",
                                        asset, oid, side, e
                                    );
                                    return ModifyResult::failed(e);
                                }
                                _ => {
                                    warn!("Unexpected modify status: {:?}", data.statuses[0]);
                                    return ModifyResult::failed("Unexpected status".to_string());
                                }
                            }
                        } else {
                            error!("Exchange data statuses is empty when modifying: {data:?}");
                            return ModifyResult::failed("Empty status response".to_string());
                        }
                    } else {
                        error!("Exchange response data is empty when modifying");
                        return ModifyResult::failed("Empty response data".to_string());
                    }
                }
                ExchangeResponseStatus::Err(e) => {
                    error!(
                        "Modify failed: asset={} oid={} side={} error={}",
                        asset, oid, side, e
                    );
                    return ModifyResult::failed(e);
                }
            },
            Err(e) => {
                error!(
                    "Modify request failed: asset={} oid={} side={} error={}",
                    asset, oid, side, e
                );
                return ModifyResult::failed(e.to_string());
            }
        }
    }

    async fn modify_bulk_orders(
        &self,
        asset: &str,
        modifies: Vec<ModifySpec>,
    ) -> Vec<ModifyResult> {
        if modifies.is_empty() {
            return vec![];
        }

        let num_modifies = modifies.len();

        // Convert ModifySpec to ClientModifyRequest
        let modify_requests: Vec<ClientModifyRequest> = modifies
            .iter()
            .map(|spec| ClientModifyRequest {
                oid: spec.oid,
                order: ClientOrderRequest {
                    asset: asset.to_string(),
                    is_buy: spec.is_buy,
                    reduce_only: false,
                    limit_px: spec.new_price,
                    sz: spec.new_size,
                    cloid: None,
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: if spec.post_only {
                            "Alo".to_string()
                        } else {
                            "Gtc".to_string()
                        },
                    }),
                },
            })
            .collect();

        let result = self.client.bulk_modify(modify_requests, None).await;

        match result {
            Ok(response) => match response {
                ExchangeResponseStatus::Ok(data) => {
                    if let Some(data) = data.data {
                        let mut results = Vec::with_capacity(num_modifies);
                        let mut success_count = 0u32;
                        let mut error_count = 0u32;

                        for (i, status) in data.statuses.iter().enumerate() {
                            let spec = &modifies[i];

                            match status {
                                ExchangeDataStatus::Filled(order) => {
                                    if let Some(ref m) = self.metrics {
                                        m.record_order_placed();
                                    }
                                    success_count += 1;
                                    results.push(ModifyResult::success(order.oid, spec.new_size));
                                }
                                ExchangeDataStatus::Resting(order) => {
                                    if let Some(ref m) = self.metrics {
                                        m.record_order_placed();
                                    }
                                    success_count += 1;
                                    results.push(ModifyResult::success(order.oid, spec.new_size));
                                }
                                ExchangeDataStatus::Error(e) => {
                                    error_count += 1;
                                    results.push(ModifyResult::failed(e.clone()));
                                }
                                _ => {
                                    error_count += 1;
                                    results.push(ModifyResult::failed(
                                        "Unexpected status".to_string(),
                                    ));
                                }
                            }
                        }

                        // Log consolidated summary
                        if success_count > 0 || error_count > 0 {
                            info!(
                                success = success_count,
                                errors = error_count,
                                "Bulk modify completed"
                            );
                        }

                        // Fill remaining with failed if we got fewer responses
                        while results.len() < num_modifies {
                            results.push(ModifyResult::failed("Missing response".to_string()));
                        }

                        return results;
                    }
                    error!("Bulk modify response data is empty");
                }
                ExchangeResponseStatus::Err(e) => {
                    error!("Bulk modify failed: {}", e);
                }
            },
            Err(e) => {
                error!("Bulk modify request failed: {}", e);
            }
        }

        // All failed
        modifies
            .iter()
            .map(|_| ModifyResult::failed("Bulk modify failed".to_string()))
            .collect()
    }
}
