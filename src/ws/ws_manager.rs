use std::{
    borrow::BorrowMut,
    collections::HashMap,
    ops::DerefMut,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use alloy::primitives::Address;
use futures_util::{stream::SplitSink, SinkExt, StreamExt};
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use tokio::{
    net::TcpStream,
    spawn,
    sync::{mpsc::UnboundedSender, oneshot, Mutex},
    time,
};
use tokio_tungstenite::{
    connect_async,
    tungstenite::{self, protocol},
    MaybeTlsStream, WebSocketStream,
};

use crate::{
    prelude::*,
    ws::message_types::{
        ActiveAssetData, ActiveSpotAssetCtx, AllMids, Bbo, Candle, L2Book, OpenOrders,
        OrderUpdates, Trades, User, WsPostRequest, WsPostResponse, WsPostResponseData,
        WsPostResponsePayload,
    },
    ActiveAssetCtx, Error, Notification, UserFills, UserFundings, UserNonFundingLedgerUpdates,
    WebData2,
};

#[derive(Debug)]
struct SubscriptionData {
    sending_channel: UnboundedSender<Arc<Message>>,
    subscription_id: u32,
    id: String,
}

/// Configuration for WebSocket connection health and reconnection.
#[derive(Debug, Clone)]
pub struct WsHealthConfig {
    /// Interval for sending application-level ping messages (default: 30s)
    pub ping_interval: Duration,
    /// Timeout for receiving pong response before considering connection dead (default: 90s)
    pub pong_timeout: Duration,
    /// Initial delay before first reconnection attempt (default: 1s)
    pub initial_reconnect_delay: Duration,
    /// Maximum delay between reconnection attempts (default: 60s)
    pub max_reconnect_delay: Duration,
    /// Backoff multiplier for exponential delay (default: 2.0)
    pub backoff_multiplier: f64,
    /// Jitter factor to prevent thundering herd (default: 0.2 = ±20%)
    pub jitter_factor: f64,
    /// Maximum consecutive reconnection failures before giving up (0 = unlimited)
    pub max_consecutive_failures: u32,
}

impl Default for WsHealthConfig {
    fn default() -> Self {
        Self {
            ping_interval: Duration::from_secs(30),
            pong_timeout: Duration::from_secs(90),
            initial_reconnect_delay: Duration::from_secs(1),
            max_reconnect_delay: Duration::from_secs(60),
            backoff_multiplier: 2.0,
            jitter_factor: 0.2,
            max_consecutive_failures: 10,
        }
    }
}

/// Statistics about WebSocket connection health.
#[derive(Debug, Clone)]
pub struct WsHealthStats {
    /// Total number of reconnections since start
    pub reconnection_count: u64,
    /// Number of pong timeouts detected
    pub pong_timeout_count: u64,
    /// Current consecutive failures
    pub consecutive_failures: u64,
    /// Average ping round-trip time in milliseconds
    pub avg_ping_latency_ms: f64,
    /// Time since last successful pong received
    pub time_since_last_pong: Duration,
    /// Whether connection is currently healthy
    pub is_healthy: bool,
}

/// Internal health state tracking.
#[derive(Debug)]
struct HealthState {
    /// Start time for duration calculations
    start_time: Instant,
    /// Last pong received timestamp (nanos since start)
    last_pong_nanos: AtomicU64,
    /// Last ping sent timestamp (nanos since start)
    last_ping_nanos: AtomicU64,
    /// Exponential moving average of ping latency (nanos)
    avg_latency_nanos: AtomicU64,
    /// Total reconnection count
    reconnection_count: AtomicU64,
    /// Pong timeout count
    pong_timeout_count: AtomicU64,
    /// Consecutive failures
    consecutive_failures: AtomicU64,
    /// Is currently reconnecting
    is_reconnecting: AtomicBool,
}

impl HealthState {
    fn new() -> Self {
        let now_nanos = 0u64; // Start at 0
        Self {
            start_time: Instant::now(),
            last_pong_nanos: AtomicU64::new(now_nanos),
            last_ping_nanos: AtomicU64::new(now_nanos),
            avg_latency_nanos: AtomicU64::new(0),
            reconnection_count: AtomicU64::new(0),
            pong_timeout_count: AtomicU64::new(0),
            consecutive_failures: AtomicU64::new(0),
            is_reconnecting: AtomicBool::new(false),
        }
    }

    fn record_ping_sent(&self) {
        let nanos = self.start_time.elapsed().as_nanos() as u64;
        self.last_ping_nanos.store(nanos, Ordering::Relaxed);
    }

    fn record_pong_received(&self) {
        let now_nanos = self.start_time.elapsed().as_nanos() as u64;
        self.last_pong_nanos.store(now_nanos, Ordering::Relaxed);

        // Calculate round-trip time
        let ping_nanos = self.last_ping_nanos.load(Ordering::Relaxed);
        if now_nanos > ping_nanos {
            let rtt_nanos = now_nanos - ping_nanos;
            // EWMA with alpha = 0.2
            let current_avg = self.avg_latency_nanos.load(Ordering::Relaxed);
            let new_avg = if current_avg == 0 {
                rtt_nanos
            } else {
                (current_avg * 8 + rtt_nanos * 2) / 10 // 0.8 * old + 0.2 * new
            };
            self.avg_latency_nanos.store(new_avg, Ordering::Relaxed);
        }

        // Reset consecutive failures on successful pong
        self.consecutive_failures.store(0, Ordering::Relaxed);
    }

    fn time_since_last_pong(&self) -> Duration {
        let last_pong = self.last_pong_nanos.load(Ordering::Relaxed);
        let now_nanos = self.start_time.elapsed().as_nanos() as u64;
        Duration::from_nanos(now_nanos.saturating_sub(last_pong))
    }

    #[allow(dead_code)]
    fn avg_latency_ms(&self) -> f64 {
        self.avg_latency_nanos.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    fn record_reconnection(&self) {
        self.reconnection_count.fetch_add(1, Ordering::Relaxed);
    }

    fn record_pong_timeout(&self) {
        self.pong_timeout_count.fetch_add(1, Ordering::Relaxed);
    }

    fn record_failure(&self) {
        self.consecutive_failures.fetch_add(1, Ordering::Relaxed);
    }

    fn reset_failures(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug)]
pub(crate) struct WsManager {
    stop_flag: Arc<AtomicBool>,
    writer: Arc<Mutex<SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, protocol::Message>>>,
    subscriptions: Arc<Mutex<HashMap<String, Vec<SubscriptionData>>>>,
    subscription_id: u32,
    subscription_identifiers: HashMap<u32, String>,
    /// Health state tracking
    health_state: Arc<HealthState>,
    /// Health configuration
    health_config: WsHealthConfig,
    /// Pending WS post requests awaiting response, keyed by request ID
    pending_posts: Arc<Mutex<HashMap<u64, oneshot::Sender<WsPostResponseData>>>>,
    /// Counter for generating unique post request IDs
    next_post_id: Arc<AtomicU64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
#[serde(rename_all = "camelCase")]
pub enum Subscription {
    /// Subscribe to all mid prices.
    /// For HIP-3 DEXs, use `dex` to specify which DEX.
    #[serde(rename_all = "camelCase")]
    AllMids {
        /// Optional HIP-3 DEX name (e.g., "hyena", "felix").
        #[serde(skip_serializing_if = "Option::is_none")]
        dex: Option<String>,
    },
    Notification {
        user: Address,
    },
    WebData2 {
        user: Address,
    },
    /// Subscribe to candle data.
    #[serde(rename_all = "camelCase")]
    Candle {
        coin: String,
        interval: String,
        /// Optional HIP-3 DEX name.
        #[serde(skip_serializing_if = "Option::is_none")]
        dex: Option<String>,
    },
    /// Subscribe to L2 order book updates.
    #[serde(rename_all = "camelCase")]
    L2Book {
        coin: String,
        /// Optional HIP-3 DEX name.
        #[serde(skip_serializing_if = "Option::is_none")]
        dex: Option<String>,
    },
    /// Subscribe to trade feed.
    #[serde(rename_all = "camelCase")]
    Trades {
        coin: String,
        /// Optional HIP-3 DEX name.
        #[serde(skip_serializing_if = "Option::is_none")]
        dex: Option<String>,
    },
    OrderUpdates {
        user: Address,
    },
    UserEvents {
        user: Address,
    },
    UserFills {
        user: Address,
    },
    UserFundings {
        user: Address,
    },
    UserNonFundingLedgerUpdates {
        user: Address,
    },
    ActiveAssetCtx {
        coin: String,
    },
    ActiveAssetData {
        user: Address,
        coin: String,
    },
    /// Subscribe to best bid/offer updates.
    #[serde(rename_all = "camelCase")]
    Bbo {
        coin: String,
        /// Optional HIP-3 DEX name.
        #[serde(skip_serializing_if = "Option::is_none")]
        dex: Option<String>,
    },
    /// Subscribe to open orders snapshot.
    /// For HIP-3 DEXs, use `dex` to specify which DEX.
    #[serde(rename_all = "camelCase")]
    OpenOrders {
        user: Address,
        /// Optional HIP-3 DEX name (e.g., "hyna", "felix").
        #[serde(skip_serializing_if = "Option::is_none")]
        dex: Option<String>,
    },
}

/// Normalize subscription identifier for HashMap lookup.
///
/// For HIP-3 DEXs, we need to strip the `dex` field because:
/// - Subscriptions are sent to server WITH `dex` (for correct routing)
/// - Incoming messages don't include `dex` in the channel info
/// - We need consistent keys for HashMap lookup
///
/// This ensures that when we subscribe with `dex: Some("hyna")`, the HashMap
/// key matches what `get_identifier()` produces for incoming messages.
fn normalize_identifier(identifier: &str) -> Result<String> {
    let sub: Subscription =
        serde_json::from_str(identifier).map_err(|e| Error::JsonParse(e.to_string()))?;

    let normalized = match sub {
        Subscription::AllMids { dex: _ } => Subscription::AllMids { dex: None },
        Subscription::L2Book { coin, dex: _ } => Subscription::L2Book { coin, dex: None },
        Subscription::Trades { coin, dex: _ } => Subscription::Trades { coin, dex: None },
        Subscription::Candle {
            coin,
            interval,
            dex: _,
        } => Subscription::Candle {
            coin,
            interval,
            dex: None,
        },
        Subscription::Bbo { coin, dex: _ } => Subscription::Bbo { coin, dex: None },
        // Non-DEX subscriptions pass through unchanged
        other => other,
    };

    serde_json::to_string(&normalized).map_err(|e| Error::JsonParse(e.to_string()))
}

#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "channel")]
#[serde(rename_all = "camelCase")]
#[allow(clippy::large_enum_variant)]
pub enum Message {
    NoData,
    HyperliquidError(String),
    AllMids(AllMids),
    Trades(Trades),
    L2Book(L2Book),
    User(User),
    UserFills(UserFills),
    Candle(Candle),
    SubscriptionResponse,
    OrderUpdates(OrderUpdates),
    UserFundings(UserFundings),
    UserNonFundingLedgerUpdates(UserNonFundingLedgerUpdates),
    Notification(Notification),
    WebData2(WebData2),
    ActiveAssetCtx(ActiveAssetCtx),
    ActiveAssetData(ActiveAssetData),
    ActiveSpotAssetCtx(ActiveSpotAssetCtx),
    Bbo(Bbo),
    OpenOrders(OpenOrders),
    Pong,
}

#[derive(Serialize)]
pub(crate) struct SubscriptionSendData<'a> {
    method: &'static str,
    subscription: &'a serde_json::Value,
}

#[derive(Serialize)]
pub(crate) struct Ping {
    method: &'static str,
}

impl WsManager {
    /// Create a new WsManager with default health configuration.
    pub(crate) async fn new(url: String, reconnect: bool) -> Result<WsManager> {
        Self::with_config(url, reconnect, WsHealthConfig::default()).await
    }

    /// Create a new WsManager with custom health configuration.
    pub(crate) async fn with_config(
        url: String,
        reconnect: bool,
        health_config: WsHealthConfig,
    ) -> Result<WsManager> {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let health_state = Arc::new(HealthState::new());

        let (writer, mut reader) = Self::connect(&url).await?.split();
        let writer = Arc::new(Mutex::new(writer));

        let subscriptions_map: HashMap<String, Vec<SubscriptionData>> = HashMap::new();
        let subscriptions = Arc::new(Mutex::new(subscriptions_map));
        let subscriptions_copy = Arc::clone(&subscriptions);

        // Clone config values for use in async tasks
        let initial_delay = health_config.initial_reconnect_delay;
        let max_delay = health_config.max_reconnect_delay;
        let backoff_mult = health_config.backoff_multiplier;
        let jitter = health_config.jitter_factor;
        let max_failures = health_config.max_consecutive_failures;

        // Create pending_posts Arc before reader task
        let pending_posts: Arc<Mutex<HashMap<u64, oneshot::Sender<WsPostResponseData>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let pending_posts_reader = Arc::clone(&pending_posts);

        // Reader task with enhanced reconnection logic
        {
            let writer = writer.clone();
            let stop_flag = Arc::clone(&stop_flag);
            let health_state = Arc::clone(&health_state);
            let reader_fut = async move {
                while !stop_flag.load(Ordering::Relaxed) {
                    if let Some(data) = reader.next().await {
                        // Check for WebSocket protocol pong frame
                        if let Ok(ref msg) = data {
                            if matches!(msg, protocol::Message::Pong(_)) {
                                debug!("Received WebSocket protocol pong");
                                health_state.record_pong_received();
                            }
                        }

                        if let Err(err) = WsManager::parse_and_send_data(
                            data,
                            &subscriptions_copy,
                            Some(&health_state),
                            Some(&pending_posts_reader),
                        )
                        .await
                        {
                            error!("Error processing data received by WsManager reader: {err}");
                        }
                    } else {
                        warn!("WsManager disconnected");
                        if let Err(err) = WsManager::send_to_all_subscriptions(
                            &subscriptions_copy,
                            Message::NoData,
                        )
                        .await
                        {
                            warn!("Error sending disconnection notification err={err}");
                        }
                        if reconnect {
                            // Exponential backoff with jitter
                            health_state.is_reconnecting.store(true, Ordering::Relaxed);
                            let mut attempt = 0u32;

                            loop {
                                let delay = Self::calculate_backoff_delay(
                                    attempt,
                                    initial_delay,
                                    max_delay,
                                    backoff_mult,
                                    jitter,
                                );

                                info!(
                                    "WsManager reconnecting with backoff attempt={} delay_ms={}",
                                    attempt + 1,
                                    delay.as_millis()
                                );

                                tokio::time::sleep(delay).await;

                                match Self::connect(&url).await {
                                    Ok(ws) => {
                                        let (new_writer, new_reader) = ws.split();
                                        reader = new_reader;
                                        let mut writer_guard = writer.lock().await;
                                        *writer_guard = new_writer;

                                        // Clean up orphaned WS POST requests from before reconnection
                                        // These requests will never receive responses on the new connection
                                        {
                                            let mut pending = pending_posts_reader.lock().await;
                                            let orphaned_count = pending.len();
                                            if orphaned_count > 0 {
                                                warn!(
                                                    "Cleaning up {} orphaned WS POST requests after reconnection",
                                                    orphaned_count
                                                );
                                                // Send error to all pending requests so they fail fast
                                                // instead of waiting for timeout
                                                for (id, sender) in pending.drain() {
                                                    let _ = sender.send(WsPostResponseData {
                                                        id,
                                                        response: WsPostResponsePayload::Error {
                                                            payload: "Connection reset during WS reconnection".to_string(),
                                                        },
                                                    });
                                                }
                                            }
                                        }

                                        // Resubscribe to all subscriptions
                                        for (identifier, v) in
                                            subscriptions_copy.lock().await.iter()
                                        {
                                            if identifier.eq("userEvents")
                                                || identifier.eq("orderUpdates")
                                            {
                                                for subscription_data in v {
                                                    if let Err(err) = Self::subscribe(
                                                        writer_guard.deref_mut(),
                                                        &subscription_data.id,
                                                    )
                                                    .await
                                                    {
                                                        error!(
                                                            "Could not resubscribe {identifier}: {err}"
                                                        );
                                                    }
                                                }
                                            } else if let Err(err) = Self::subscribe(
                                                writer_guard.deref_mut(),
                                                identifier,
                                            )
                                            .await
                                            {
                                                error!(
                                                    "Could not resubscribe correctly {identifier}: {err}"
                                                );
                                            }
                                        }

                                        health_state.record_reconnection();
                                        health_state.reset_failures();
                                        health_state
                                            .is_reconnecting
                                            .store(false, Ordering::Relaxed);
                                        // Record initial pong time to reset timeout
                                        health_state.record_pong_received();
                                        info!(
                                            "WsManager reconnect finished successfully reconnection_count={}",
                                            health_state.reconnection_count.load(Ordering::Relaxed)
                                        );
                                        break;
                                    }
                                    Err(err) => {
                                        health_state.record_failure();
                                        let failures = health_state
                                            .consecutive_failures
                                            .load(Ordering::Relaxed);

                                        if max_failures > 0 && failures >= max_failures as u64 {
                                            error!(
                                                "Max reconnection failures exceeded, giving up failures={} max={}",
                                                failures,
                                                max_failures
                                            );
                                            health_state
                                                .is_reconnecting
                                                .store(false, Ordering::Relaxed);
                                            break;
                                        }

                                        error!(
                                            "Could not connect to websocket: {} attempt={} failures={}",
                                            err,
                                            attempt + 1,
                                            failures
                                        );
                                        attempt += 1;
                                    }
                                }
                            }
                        } else {
                            error!("WsManager reconnection disabled. Will not reconnect and exiting reader task.");
                            break;
                        }
                    }
                }
                warn!("ws message reader task stopped");
            };
            spawn(reader_fut);
        }

        // Enhanced ping task with pong timeout detection
        let ping_interval = health_config.ping_interval;
        let pong_timeout = health_config.pong_timeout;
        {
            let stop_flag = Arc::clone(&stop_flag);
            let writer = Arc::clone(&writer);
            let health_state = Arc::clone(&health_state);
            let ping_fut = async move {
                // Initial delay to let connection stabilize
                time::sleep(Duration::from_secs(1)).await;

                while !stop_flag.load(Ordering::Relaxed) {
                    // Skip ping if we're currently reconnecting
                    if health_state.is_reconnecting.load(Ordering::Relaxed) {
                        time::sleep(Duration::from_secs(1)).await;
                        continue;
                    }

                    // Check for pong timeout BEFORE sending new ping
                    let time_since_pong = health_state.time_since_last_pong();
                    if time_since_pong > pong_timeout {
                        warn!(
                            "Pong timeout detected - connection may be dead time_since_pong_secs={:.1} timeout_secs={:.1}",
                            time_since_pong.as_secs_f64(),
                            pong_timeout.as_secs_f64()
                        );
                        health_state.record_pong_timeout();
                        // Close the connection to trigger reconnect in reader task
                        let mut writer = writer.lock().await;
                        if let Err(err) = writer.send(protocol::Message::Close(None)).await {
                            debug!("Error sending close frame: {err}");
                        }
                        // Wait for reconnection
                        time::sleep(Duration::from_secs(2)).await;
                        continue;
                    }

                    // Send application-level JSON ping
                    match serde_json::to_string(&Ping { method: "ping" }) {
                        Ok(payload) => {
                            let mut writer = writer.lock().await;
                            health_state.record_ping_sent();
                            if let Err(err) = writer.send(protocol::Message::Text(payload)).await {
                                error!("Error pinging server: {err}")
                            }

                            // Also send WebSocket protocol-level ping for more reliable keepalive
                            if let Err(err) = writer.send(protocol::Message::Ping(vec![])).await {
                                debug!("Error sending protocol ping: {err}");
                            }
                        }
                        Err(err) => error!("Error serializing ping message: {err}"),
                    }
                    time::sleep(ping_interval).await;
                }
                warn!("ws ping task stopped");
            };
            spawn(ping_fut);
        }

        Ok(WsManager {
            stop_flag,
            writer,
            subscriptions,
            subscription_id: 0,
            subscription_identifiers: HashMap::new(),
            health_state,
            health_config,
            pending_posts,
            next_post_id: Arc::new(AtomicU64::new(1)),
        })
    }

    /// Calculate exponential backoff delay with jitter.
    fn calculate_backoff_delay(
        attempt: u32,
        initial: Duration,
        max: Duration,
        multiplier: f64,
        jitter_factor: f64,
    ) -> Duration {
        // Calculate base delay: initial * multiplier^attempt
        let base_secs = initial.as_secs_f64() * multiplier.powi(attempt as i32);
        let capped_secs = base_secs.min(max.as_secs_f64());

        // Add jitter: delay * (1 ± jitter)
        // Use a simple deterministic "random" based on attempt for reproducibility
        let jitter_mult = if attempt.is_multiple_of(2) {
            1.0 + jitter_factor * 0.5
        } else {
            1.0 - jitter_factor * 0.5
        };
        let jittered_secs = capped_secs * jitter_mult;

        Duration::from_secs_f64(jittered_secs.max(0.1))
    }

    /// Get current health statistics.
    #[allow(dead_code)]
    pub(crate) fn health_stats(&self) -> WsHealthStats {
        let consecutive_failures = self
            .health_state
            .consecutive_failures
            .load(Ordering::Relaxed);
        let is_reconnecting = self.health_state.is_reconnecting.load(Ordering::Relaxed);
        let time_since_pong = self.health_state.time_since_last_pong();

        WsHealthStats {
            reconnection_count: self.health_state.reconnection_count.load(Ordering::Relaxed),
            pong_timeout_count: self.health_state.pong_timeout_count.load(Ordering::Relaxed),
            consecutive_failures,
            avg_ping_latency_ms: self.health_state.avg_latency_ms(),
            time_since_last_pong: time_since_pong,
            is_healthy: !is_reconnecting
                && consecutive_failures == 0
                && time_since_pong < self.health_config.pong_timeout,
        }
    }

    /// Check if connection is currently healthy.
    #[allow(dead_code)]
    pub(crate) fn is_healthy(&self) -> bool {
        !self.health_state.is_reconnecting.load(Ordering::Relaxed)
            && self
                .health_state
                .consecutive_failures
                .load(Ordering::Relaxed)
                == 0
            && self.health_state.time_since_last_pong() < self.health_config.pong_timeout
    }

    /// Send a POST request over WebSocket and await the response.
    ///
    /// This is the unified method for sending actions (orders, cancels) and info
    /// requests over WebSocket instead of REST. The response is delivered via
    /// a oneshot channel.
    ///
    /// # Arguments
    /// * `payload` - The signed action or info request payload as JSON
    /// * `is_action` - true for action (order/cancel), false for info request
    /// * `timeout` - Maximum time to wait for response
    ///
    /// # Returns
    /// The response data if successful, or an error if timeout/send failure.
    pub(crate) async fn post(
        &self,
        payload: serde_json::Value,
        is_action: bool,
        timeout: Duration,
    ) -> Result<WsPostResponseData> {
        // Generate unique request ID
        let request_id = self.next_post_id.fetch_add(1, Ordering::SeqCst);

        // Create the request
        let request = if is_action {
            WsPostRequest::action(request_id, payload)
        } else {
            WsPostRequest::info(request_id, payload)
        };

        // Create channel for response
        let (tx, rx) = oneshot::channel();

        // Register pending request
        {
            let mut pending = self.pending_posts.lock().await;
            pending.insert(request_id, tx);
        }

        // Serialize and send
        let message_text = match serde_json::to_string(&request) {
            Ok(s) => s,
            Err(e) => {
                // Remove pending on serialization error
                let mut pending = self.pending_posts.lock().await;
                pending.remove(&request_id);
                return Err(Error::JsonParse(e.to_string()));
            }
        };

        debug!(
            "Sending WS post request id={}: {}",
            request_id, &message_text
        );

        {
            let mut writer = self.writer.lock().await;
            if let Err(e) = writer.send(protocol::Message::Text(message_text)).await {
                // Remove pending on send error
                let mut pending = self.pending_posts.lock().await;
                pending.remove(&request_id);
                return Err(Error::WsSend(e.to_string()));
            }
        }

        // Wait for response with timeout
        match time::timeout(timeout, rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => {
                // Channel closed (shouldn't happen normally)
                Err(Error::WsSend("Response channel closed".to_string()))
            }
            Err(_) => {
                // Timeout - remove pending request
                let mut pending = self.pending_posts.lock().await;
                pending.remove(&request_id);
                Err(Error::WsSend(format!(
                    "WS post request {} timed out after {:?}",
                    request_id, timeout
                )))
            }
        }
    }

    async fn connect(url: &str) -> Result<WebSocketStream<MaybeTlsStream<TcpStream>>> {
        Ok(connect_async(url)
            .await
            .map_err(|e| Error::Websocket(e.to_string()))?
            .0)
    }

    fn get_identifier(message: &Message) -> Result<String> {
        match message {
            // Note: For incoming messages, we don't know which DEX they belong to.
            // The server routes messages based on the subscription, so we use None here.
            // This matches the default subscription behavior.
            Message::AllMids(_) => serde_json::to_string(&Subscription::AllMids { dex: None })
                .map_err(|e| Error::JsonParse(e.to_string())),
            Message::User(_) => Ok("userEvents".to_string()),
            Message::UserFills(fills) => serde_json::to_string(&Subscription::UserFills {
                user: fills.data.user,
            })
            .map_err(|e| Error::JsonParse(e.to_string())),
            Message::Trades(trades) => {
                if trades.data.is_empty() {
                    Ok(String::default())
                } else {
                    serde_json::to_string(&Subscription::Trades {
                        coin: trades.data[0].coin.clone(),
                        dex: None,
                    })
                    .map_err(|e| Error::JsonParse(e.to_string()))
                }
            }
            Message::L2Book(l2_book) => serde_json::to_string(&Subscription::L2Book {
                coin: l2_book.data.coin.clone(),
                dex: None,
            })
            .map_err(|e| Error::JsonParse(e.to_string())),
            Message::Candle(candle) => serde_json::to_string(&Subscription::Candle {
                coin: candle.data.coin.clone(),
                interval: candle.data.interval.clone(),
                dex: None,
            })
            .map_err(|e| Error::JsonParse(e.to_string())),
            Message::OrderUpdates(_) => Ok("orderUpdates".to_string()),
            Message::UserFundings(fundings) => serde_json::to_string(&Subscription::UserFundings {
                user: fundings.data.user,
            })
            .map_err(|e| Error::JsonParse(e.to_string())),
            Message::UserNonFundingLedgerUpdates(user_non_funding_ledger_updates) => {
                serde_json::to_string(&Subscription::UserNonFundingLedgerUpdates {
                    user: user_non_funding_ledger_updates.data.user,
                })
                .map_err(|e| Error::JsonParse(e.to_string()))
            }
            Message::Notification(_) => Ok("notification".to_string()),
            Message::WebData2(_) => Ok("__WEBDATA2__".to_string()),
            Message::ActiveAssetCtx(active_asset_ctx) => {
                serde_json::to_string(&Subscription::ActiveAssetCtx {
                    coin: active_asset_ctx.data.coin.clone(),
                })
                .map_err(|e| Error::JsonParse(e.to_string()))
            }
            Message::ActiveSpotAssetCtx(active_spot_asset_ctx) => {
                serde_json::to_string(&Subscription::ActiveAssetCtx {
                    coin: active_spot_asset_ctx.data.coin.clone(),
                })
                .map_err(|e| Error::JsonParse(e.to_string()))
            }
            Message::ActiveAssetData(active_asset_data) => {
                serde_json::to_string(&Subscription::ActiveAssetData {
                    user: active_asset_data.data.user,
                    coin: active_asset_data.data.coin.clone(),
                })
                .map_err(|e| Error::JsonParse(e.to_string()))
            }
            Message::Bbo(bbo) => serde_json::to_string(&Subscription::Bbo {
                coin: bbo.data.coin.clone(),
                dex: None,
            })
            .map_err(|e| Error::JsonParse(e.to_string())),
            Message::OpenOrders(open_orders) => {
                // Extract dex from the message if present
                let dex = if open_orders.data.dex.is_empty() {
                    None
                } else {
                    Some(open_orders.data.dex.clone())
                };
                serde_json::to_string(&Subscription::OpenOrders {
                    user: open_orders.data.user,
                    dex,
                })
                .map_err(|e| Error::JsonParse(e.to_string()))
            }
            Message::SubscriptionResponse | Message::Pong => Ok(String::default()),
            Message::NoData => Ok("".to_string()),
            Message::HyperliquidError(err) => Ok(format!("hyperliquid error: {err:?}")),
        }
    }

    #[allow(clippy::type_complexity)]
    async fn parse_and_send_data(
        data: std::result::Result<protocol::Message, tungstenite::Error>,
        subscriptions: &Arc<Mutex<HashMap<String, Vec<SubscriptionData>>>>,
        health_state: Option<&HealthState>,
        pending_posts: Option<&Arc<Mutex<HashMap<u64, oneshot::Sender<WsPostResponseData>>>>>,
    ) -> Result<()> {
        match data {
            Ok(data) => match data.into_text() {
                Ok(data) => {
                    if !data.starts_with('{') {
                        return Ok(());
                    }

                    // Check if this is a WS post response first (channel: "post")
                    if let Some(pending) = pending_posts {
                        if let Ok(post_response) = serde_json::from_str::<WsPostResponse>(&data) {
                            if post_response.channel == "post" {
                                let request_id = post_response.data.id;
                                let mut pending_guard = pending.lock().await;
                                if let Some(sender) = pending_guard.remove(&request_id) {
                                    debug!(
                                        "Routing WS post response to pending request id={}",
                                        request_id
                                    );
                                    // Send response to waiting receiver
                                    let _ = sender.send(post_response.data);
                                } else {
                                    warn!("WS post response for unknown request ID={}", request_id);
                                }
                                return Ok(());
                            }
                        }
                    }

                    let message = serde_json::from_str::<Message>(&data)
                        .map_err(|e| Error::JsonParse(e.to_string()))?;

                    // Record pong received for JSON-level pong messages
                    if matches!(message, Message::Pong) {
                        if let Some(state) = health_state {
                            state.record_pong_received();
                            debug!("Received JSON pong response");
                        }
                    }

                    let identifier = WsManager::get_identifier(&message)?;
                    if identifier.is_empty() {
                        return Ok(());
                    }

                    // Wrap message in Arc once; Arc::clone is cheap (pointer copy)
                    let arc_message = Arc::new(message);

                    let mut subscriptions = subscriptions.lock().await;
                    let mut res = Ok(());

                    // Special handling for WebData2 which lacks user field in response
                    if identifier == "__WEBDATA2__" {
                        for (key, subscription_datas) in subscriptions.iter_mut() {
                            if key.contains("webData2") {
                                for subscription_data in subscription_datas {
                                    if let Err(e) = subscription_data
                                        .sending_channel
                                        .send(Arc::clone(&arc_message))
                                        .map_err(|e| Error::WsSend(e.to_string()))
                                    {
                                        res = Err(e);
                                    }
                                }
                            }
                        }
                    } else if let Some(subscription_datas) = subscriptions.get_mut(&identifier) {
                        for subscription_data in subscription_datas {
                            if let Err(e) = subscription_data
                                .sending_channel
                                .send(Arc::clone(&arc_message))
                                .map_err(|e| Error::WsSend(e.to_string()))
                            {
                                res = Err(e);
                            }
                        }
                    }
                    res
                }
                Err(err) => {
                    let error = Error::ReaderTextConversion(err.to_string());
                    Ok(WsManager::send_to_all_subscriptions(
                        subscriptions,
                        Message::HyperliquidError(error.to_string()),
                    )
                    .await?)
                }
            },
            Err(err) => {
                let error = Error::GenericReader(err.to_string());
                Ok(WsManager::send_to_all_subscriptions(
                    subscriptions,
                    Message::HyperliquidError(error.to_string()),
                )
                .await?)
            }
        }
    }

    async fn send_to_all_subscriptions(
        subscriptions: &Arc<Mutex<HashMap<String, Vec<SubscriptionData>>>>,
        message: Message,
    ) -> Result<()> {
        // Wrap message in Arc once; Arc::clone is cheap (pointer copy)
        let arc_message = Arc::new(message);

        let mut subscriptions = subscriptions.lock().await;
        let mut res = Ok(());
        for subscription_datas in subscriptions.values_mut() {
            for subscription_data in subscription_datas {
                if let Err(e) = subscription_data
                    .sending_channel
                    .send(Arc::clone(&arc_message))
                    .map_err(|e| Error::WsSend(e.to_string()))
                {
                    res = Err(e);
                }
            }
        }
        res
    }

    async fn send_subscription_data(
        method: &'static str,
        writer: &mut SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, protocol::Message>,
        identifier: &str,
    ) -> Result<()> {
        let payload = serde_json::to_string(&SubscriptionSendData {
            method,
            subscription: &serde_json::from_str::<serde_json::Value>(identifier)
                .map_err(|e| Error::JsonParse(e.to_string()))?,
        })
        .map_err(|e| Error::JsonParse(e.to_string()))?;

        writer
            .send(protocol::Message::Text(payload))
            .await
            .map_err(|e| Error::Websocket(e.to_string()))?;
        Ok(())
    }

    async fn subscribe(
        writer: &mut SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, protocol::Message>,
        identifier: &str,
    ) -> Result<()> {
        Self::send_subscription_data("subscribe", writer, identifier).await
    }

    async fn unsubscribe(
        writer: &mut SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, protocol::Message>,
        identifier: &str,
    ) -> Result<()> {
        Self::send_subscription_data("unsubscribe", writer, identifier).await
    }

    pub(crate) async fn add_subscription(
        &mut self,
        identifier: String,
        sending_channel: UnboundedSender<Arc<Message>>,
    ) -> Result<u32> {
        let mut subscriptions = self.subscriptions.lock().await;

        let identifier_entry = if let Subscription::UserEvents { user: _ } =
            serde_json::from_str::<Subscription>(&identifier)
                .map_err(|e| Error::JsonParse(e.to_string()))?
        {
            "userEvents".to_string()
        } else if let Subscription::OrderUpdates { user: _ } =
            serde_json::from_str::<Subscription>(&identifier)
                .map_err(|e| Error::JsonParse(e.to_string()))?
        {
            "orderUpdates".to_string()
        } else {
            // Normalize DEX subscriptions for consistent HashMap lookup keys.
            // This strips the `dex` field so subscriptions with dex: Some("hyna")
            // can be found when incoming messages produce keys with dex: None.
            normalize_identifier(&identifier)?
        };
        let subscriptions = subscriptions
            .entry(identifier_entry.clone())
            .or_insert(Vec::new());

        if !subscriptions.is_empty() && identifier_entry.eq("userEvents") {
            return Err(Error::UserEvents);
        }

        if subscriptions.is_empty() {
            Self::subscribe(self.writer.lock().await.borrow_mut(), identifier.as_str()).await?;
        }

        let subscription_id = self.subscription_id;
        self.subscription_identifiers
            .insert(subscription_id, identifier.clone());
        subscriptions.push(SubscriptionData {
            sending_channel,
            subscription_id,
            id: identifier,
        });

        self.subscription_id += 1;
        Ok(subscription_id)
    }

    pub(crate) async fn remove_subscription(&mut self, subscription_id: u32) -> Result<()> {
        let identifier = self
            .subscription_identifiers
            .get(&subscription_id)
            .ok_or(Error::SubscriptionNotFound)?
            .clone();

        let identifier_entry = if let Subscription::UserEvents { user: _ } =
            serde_json::from_str::<Subscription>(&identifier)
                .map_err(|e| Error::JsonParse(e.to_string()))?
        {
            "userEvents".to_string()
        } else if let Subscription::OrderUpdates { user: _ } =
            serde_json::from_str::<Subscription>(&identifier)
                .map_err(|e| Error::JsonParse(e.to_string()))?
        {
            "orderUpdates".to_string()
        } else {
            // Use same normalization as add_subscription for consistent lookup
            normalize_identifier(&identifier)?
        };

        self.subscription_identifiers.remove(&subscription_id);

        let mut subscriptions = self.subscriptions.lock().await;

        let subscriptions = subscriptions
            .get_mut(&identifier_entry)
            .ok_or(Error::SubscriptionNotFound)?;
        let index = subscriptions
            .iter()
            .position(|subscription_data| subscription_data.subscription_id == subscription_id)
            .ok_or(Error::SubscriptionNotFound)?;
        subscriptions.remove(index);

        if subscriptions.is_empty() {
            Self::unsubscribe(self.writer.lock().await.borrow_mut(), identifier.as_str()).await?;
        }
        Ok(())
    }
}

impl WsManager {
    /// Force a reconnection by closing the current WebSocket connection.
    ///
    /// This sends a Close frame to the server, which will cause the reader task
    /// to enter its reconnection loop with exponential backoff.
    ///
    /// Use this when the ConnectionSupervisor detects stale data and recommends
    /// reconnection, rather than waiting for the ping/pong timeout.
    pub(crate) async fn force_reconnect(&self) {
        info!("Force reconnect requested - sending Close frame");
        let mut writer = self.writer.lock().await;
        if let Err(err) = writer.send(protocol::Message::Close(None)).await {
            warn!("Error sending Close frame during force_reconnect: {err}");
        }
    }
}

impl Drop for WsManager {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
    }
}
