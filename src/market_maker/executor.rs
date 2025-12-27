//! Order execution abstraction for the market maker.

use async_trait::async_trait;
use tracing::{error, info, warn};

use crate::{
    ClientCancelRequest, ClientLimit, ClientModifyRequest, ClientOrder, ClientOrderRequest,
    ExchangeClient, ExchangeDataStatus, ExchangeResponseStatus,
};

use super::config::MetricsRecorder;

/// Result of placing an order.
#[derive(Debug, Clone)]
pub struct OrderResult {
    /// Amount that is now resting (may be less than requested if partially filled)
    pub resting_size: f64,
    /// Order ID from the exchange (0 if order failed)
    pub oid: u64,
    /// Whether the order was immediately filled
    pub filled: bool,
}

impl OrderResult {
    /// Create a failed order result.
    pub fn failed() -> Self {
        Self {
            resting_size: 0.0,
            oid: 0,
            filled: false,
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

/// Order specification for bulk placement.
#[derive(Debug, Clone)]
pub struct OrderSpec {
    /// Order price
    pub price: f64,
    /// Order size
    pub size: f64,
    /// Whether this is a buy order
    pub is_buy: bool,
}

/// Trait for order execution.
/// Abstracts the exchange client to enable testing and mocking.
#[async_trait]
pub trait OrderExecutor: Send + Sync {
    /// Place a limit order.
    async fn place_order(&self, asset: &str, price: f64, size: f64, is_buy: bool) -> OrderResult;

    /// Place multiple orders in a single API call.
    ///
    /// This is much faster than placing orders sequentially when submitting
    /// multiple ladder levels. Returns results for each order in the same order
    /// as the input specs.
    async fn place_bulk_orders(&self, asset: &str, orders: Vec<OrderSpec>) -> Vec<OrderResult>;

    /// Cancel an order.
    /// Returns `true` if cancel was successful, `false` otherwise.
    async fn cancel_order(&self, asset: &str, oid: u64) -> bool;

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
    ) -> ModifyResult;
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
    async fn place_order(&self, asset: &str, price: f64, size: f64, is_buy: bool) -> OrderResult {
        let side = if is_buy { "BUY" } else { "SELL" };

        let order = self
            .client
            .order(
                ClientOrderRequest {
                    asset: asset.to_string(),
                    is_buy,
                    reduce_only: false,
                    limit_px: price,
                    sz: size,
                    cloid: None,
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: "Gtc".to_string(),
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
                                    info!("Order filled immediately: oid={}", order.oid);
                                    if let Some(ref m) = self.metrics {
                                        m.record_order_placed();
                                    }
                                    return OrderResult {
                                        resting_size: size,
                                        oid: order.oid,
                                        filled: true,
                                    };
                                }
                                ExchangeDataStatus::Resting(order) => {
                                    info!("Order resting: oid={}", order.oid);
                                    if let Some(ref m) = self.metrics {
                                        m.record_order_placed();
                                    }
                                    return OrderResult {
                                        resting_size: size,
                                        oid: order.oid,
                                        filled: false,
                                    };
                                }
                                ExchangeDataStatus::Error(e) => {
                                    error!(
                                        "Order rejected: asset={} side={} sz={} price={} error={}",
                                        asset, side, size, price, e
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
                        "Order failed: asset={} side={} sz={} price={} error={}",
                        asset, side, size, price, e
                    );
                }
            },
            Err(e) => {
                error!(
                    "Order request failed: asset={} side={} sz={} price={} error={}",
                    asset, side, size, price, e
                );
            }
        }

        OrderResult::failed()
    }

    async fn place_bulk_orders(&self, asset: &str, orders: Vec<OrderSpec>) -> Vec<OrderResult> {
        if orders.is_empty() {
            return vec![];
        }

        // Convert to ClientOrderRequest vec
        let order_requests: Vec<ClientOrderRequest> = orders
            .iter()
            .map(|spec| ClientOrderRequest {
                asset: asset.to_string(),
                is_buy: spec.is_buy,
                reduce_only: false,
                limit_px: spec.price,
                sz: spec.size,
                cloid: None,
                order_type: ClientOrder::Limit(ClientLimit {
                    tif: "Gtc".to_string(),
                }),
            })
            .collect();

        let num_orders = order_requests.len();
        info!("Placing {} orders in bulk", num_orders);

        let result = self.client.bulk_order(order_requests, None).await;

        match result {
            Ok(response) => match response {
                ExchangeResponseStatus::Ok(data) => {
                    if let Some(data) = data.data {
                        let mut results = Vec::with_capacity(num_orders);

                        // Each status corresponds to one order in sequence
                        for (i, status) in data.statuses.iter().enumerate() {
                            let spec = &orders[i];
                            let side = if spec.is_buy { "BUY" } else { "SELL" };

                            match status {
                                ExchangeDataStatus::Filled(order) => {
                                    info!(
                                        "Bulk order {} filled immediately: oid={}",
                                        i, order.oid
                                    );
                                    if let Some(ref m) = self.metrics {
                                        m.record_order_placed();
                                    }
                                    results.push(OrderResult {
                                        resting_size: spec.size,
                                        oid: order.oid,
                                        filled: true,
                                    });
                                }
                                ExchangeDataStatus::Resting(order) => {
                                    info!("Bulk order {} resting: oid={}", i, order.oid);
                                    if let Some(ref m) = self.metrics {
                                        m.record_order_placed();
                                    }
                                    results.push(OrderResult {
                                        resting_size: spec.size,
                                        oid: order.oid,
                                        filled: false,
                                    });
                                }
                                ExchangeDataStatus::Error(e) => {
                                    error!(
                                        "Bulk order {} rejected: asset={} side={} sz={} price={} error={}",
                                        i, asset, side, spec.size, spec.price, e
                                    );
                                    results.push(OrderResult::failed());
                                }
                                _ => {
                                    warn!("Unexpected bulk order {} status: {:?}", i, status);
                                    results.push(OrderResult::failed());
                                }
                            }
                        }

                        // If we got fewer statuses than orders, fill remaining with failures
                        while results.len() < num_orders {
                            results.push(OrderResult::failed());
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

        // All failed
        vec![OrderResult::failed(); num_orders]
    }

    async fn cancel_order(&self, asset: &str, oid: u64) -> bool {
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
                                    return true;
                                }
                                ExchangeDataStatus::Error(e) => {
                                    // "already canceled or filled" means the order is gone - that's success
                                    if e.contains("already canceled") || e.contains("filled") {
                                        info!("Order already gone: {e}");
                                        return true;
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

        false
    }

    async fn modify_order(
        &self,
        asset: &str,
        oid: u64,
        new_price: f64,
        new_size: f64,
        is_buy: bool,
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
                    tif: "Gtc".to_string(),
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
                                    info!(
                                        "Modified order filled immediately: oid={}",
                                        order.oid
                                    );
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
}
