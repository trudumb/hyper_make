//! Order execution abstraction for the market maker.

use async_trait::async_trait;
use tracing::{error, info, warn};

use crate::{
    ClientCancelRequest, ClientLimit, ClientOrder, ClientOrderRequest, ExchangeClient,
    ExchangeDataStatus, ExchangeResponseStatus,
};

use super::config::MetricsRecorder;

/// Result of placing an order.
#[derive(Debug)]
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

/// Trait for order execution.
/// Abstracts the exchange client to enable testing and mocking.
#[async_trait]
pub trait OrderExecutor: Send + Sync {
    /// Place a limit order.
    async fn place_order(&self, asset: &str, price: f64, size: f64, is_buy: bool) -> OrderResult;

    /// Cancel an order.
    /// Returns `true` if cancel was successful, `false` otherwise.
    async fn cancel_order(&self, asset: &str, oid: u64) -> bool;
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
}
