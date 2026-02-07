//! WebSocket Dashboard Server
//!
//! Provides real-time push updates to dashboard clients, replacing HTTP polling.
//!
//! # Architecture
//!
//! ```text
//! QuoteEngine ──► broadcast::Sender ──► DashboardWsState ──► WebSocket clients
//! ```
//!
//! # Message Protocol
//!
//! Server pushes `DashboardPush` messages to all connected clients:
//! - `Snapshot`: Full state on connect and periodically
//! - `Update`: Throttled incremental updates (100ms)
//! - `Fill`: Immediate push on fill events

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::RwLock;
use std::time::{Duration, Instant};
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

use super::metrics::dashboard::{
    DashboardState, FillRecord, LiveQuotes, PnLAttribution, RegimeState,
};

// ============================================================================
// Message Types
// ============================================================================

/// Server-to-client push messages
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DashboardPush {
    /// Full state snapshot (sent on connect + periodically)
    Snapshot { state: Box<DashboardState> },

    /// Incremental update (throttled, only changed fields)
    Update {
        #[serde(skip_serializing_if = "Option::is_none")]
        quotes: Option<LiveQuotes>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pnl: Option<PnLAttribution>,
        #[serde(skip_serializing_if = "Option::is_none")]
        regime: Option<RegimeState>,
        timestamp_ms: i64,
    },

    /// New fill occurred (immediate push)
    Fill { record: FillRecord },

    /// Connection status
    Connected {
        client_id: usize,
        server_time_ms: i64,
    },
}

/// Client-to-server commands (Phase 2 - not implemented yet)
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DashboardCommand {
    /// Request full state snapshot
    RequestSnapshot { id: u64 },

    /// Ping for keepalive
    Ping { id: u64 },
    // Phase 2 commands (not yet implemented):
    // SetGamma { id: u64, value: f64 },
    // SetQuotingState { id: u64, active: bool },
}

// ============================================================================
// Configuration
// ============================================================================

/// WebSocket server configuration
#[derive(Debug, Clone)]
pub struct DashboardWsConfig {
    /// Minimum interval between throttled updates (ms)
    pub update_throttle_ms: u64,
    /// Full snapshot interval (ms)
    pub snapshot_interval_ms: u64,
    /// Maximum connected clients
    pub max_clients: usize,
    /// Ping interval for keepalive (ms)
    pub ping_interval_ms: u64,
    /// Broadcast channel capacity
    pub broadcast_capacity: usize,
}

impl Default for DashboardWsConfig {
    fn default() -> Self {
        Self {
            update_throttle_ms: 100,
            snapshot_interval_ms: 5000,
            max_clients: 10,
            ping_interval_ms: 30000,
            broadcast_capacity: 256,
        }
    }
}

// ============================================================================
// Server State
// ============================================================================

/// Shared state for WebSocket dashboard server
pub struct DashboardWsState {
    /// Broadcast channel for pushing updates to all clients
    tx: broadcast::Sender<DashboardPush>,
    /// Current dashboard state (for snapshots on connect)
    current_state: Arc<RwLock<DashboardState>>,
    /// Active client count
    client_count: AtomicUsize,
    /// Next client ID
    next_client_id: AtomicUsize,
    /// Configuration
    config: DashboardWsConfig,
}

impl DashboardWsState {
    /// Create a new WebSocket dashboard state
    pub fn new(config: DashboardWsConfig) -> Self {
        let (tx, _) = broadcast::channel(config.broadcast_capacity);
        Self {
            tx,
            current_state: Arc::new(RwLock::new(DashboardState::default())),
            client_count: AtomicUsize::new(0),
            next_client_id: AtomicUsize::new(1),
            config,
        }
    }

    /// Get the broadcast sender for pushing updates
    pub fn sender(&self) -> broadcast::Sender<DashboardPush> {
        self.tx.clone()
    }

    /// Update the current state (call from quote engine)
    pub fn update_state(&self, state: DashboardState) {
        let mut current = self.current_state.write().unwrap();
        *current = state;
    }

    /// Get current client count
    pub fn client_count(&self) -> usize {
        self.client_count.load(Ordering::Relaxed)
    }

    /// Push an update to all clients
    pub fn push(&self, msg: DashboardPush) {
        // Ignore send errors (no receivers is fine)
        let _ = self.tx.send(msg);
    }

    /// Push a snapshot to all clients
    pub fn push_snapshot(&self) {
        let state = self.current_state.read().unwrap().clone();
        self.push(DashboardPush::Snapshot { state: Box::new(state) });
    }

    /// Push a fill event immediately
    pub fn push_fill(&self, record: FillRecord) {
        self.push(DashboardPush::Fill { record });
    }

    /// Push an incremental update
    pub fn push_update(
        &self,
        quotes: Option<LiveQuotes>,
        pnl: Option<PnLAttribution>,
        regime: Option<RegimeState>,
    ) {
        self.push(DashboardPush::Update {
            quotes,
            pnl,
            regime,
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
        });
    }
}

// ============================================================================
// WebSocket Handler
// ============================================================================

/// Axum handler for WebSocket upgrade
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<DashboardWsState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_connection(socket, state))
}

/// Handle a WebSocket connection
async fn handle_connection(socket: WebSocket, state: Arc<DashboardWsState>) {
    // Check client limit
    let count = state.client_count.fetch_add(1, Ordering::SeqCst);
    if count >= state.config.max_clients {
        state.client_count.fetch_sub(1, Ordering::SeqCst);
        warn!(
            max = state.config.max_clients,
            "Dashboard WS: max clients reached, rejecting connection"
        );
        return;
    }

    let client_id = state.next_client_id.fetch_add(1, Ordering::SeqCst);
    info!(
        client_id,
        count = count + 1,
        "Dashboard WS: client connected"
    );

    // Subscribe to broadcast channel
    let mut rx = state.tx.subscribe();

    // Split socket
    let (mut sender, mut receiver) = socket.split();

    // Send initial snapshot
    {
        let current = state.current_state.read().unwrap().clone();
        let msg = DashboardPush::Snapshot { state: Box::new(current) };
        if let Ok(json) = serde_json::to_string(&msg) {
            if sender.send(Message::Text(json)).await.is_err() {
                state.client_count.fetch_sub(1, Ordering::SeqCst);
                return;
            }
        }
    }

    // Send connected confirmation
    {
        let msg = DashboardPush::Connected {
            client_id,
            server_time_ms: chrono::Utc::now().timestamp_millis(),
        };
        if let Ok(json) = serde_json::to_string(&msg) {
            let _ = sender.send(Message::Text(json)).await;
        }
    }

    let snapshot_interval = Duration::from_millis(state.config.snapshot_interval_ms);
    let mut last_snapshot = Instant::now();

    // Main loop: forward broadcasts and handle incoming messages
    loop {
        tokio::select! {
            // Forward broadcast messages to this client
            result = rx.recv() => {
                match result {
                    Ok(push) => {
                        if let Ok(json) = serde_json::to_string(&push) {
                            if sender.send(Message::Text(json)).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        debug!(client_id, skipped = n, "Dashboard WS: client lagged, skipped messages");
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        break;
                    }
                }
            }

            // Handle incoming messages from client
            result = receiver.next() => {
                match result {
                    Some(Ok(Message::Text(text))) => {
                        // Parse command (Phase 2: handle commands)
                        if let Ok(cmd) = serde_json::from_str::<DashboardCommand>(&text) {
                            debug!(client_id, ?cmd, "Dashboard WS: received command");
                            match cmd {
                                DashboardCommand::RequestSnapshot { .. } => {
                                    let current = state.current_state.read().unwrap().clone();
                                    let msg = DashboardPush::Snapshot { state: Box::new(current) };
                                    if let Ok(json) = serde_json::to_string(&msg) {
                                        let _ = sender.send(Message::Text(json)).await;
                                    }
                                }
                                DashboardCommand::Ping { id } => {
                                    // Respond with pong (via Update with timestamp)
                                    let msg = DashboardPush::Update {
                                        quotes: None,
                                        pnl: None,
                                        regime: None,
                                        timestamp_ms: chrono::Utc::now().timestamp_millis(),
                                    };
                                    if let Ok(json) = serde_json::to_string(&msg) {
                                        let _ = sender.send(Message::Text(json)).await;
                                    }
                                    debug!(client_id, id, "Dashboard WS: pong sent");
                                }
                            }
                        }
                    }
                    Some(Ok(Message::Ping(data))) => {
                        if sender.send(Message::Pong(data)).await.is_err() {
                            break;
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        break;
                    }
                    _ => {}
                }
            }

            // Periodic snapshot
            _ = tokio::time::sleep(Duration::from_millis(100)) => {
                if last_snapshot.elapsed() >= snapshot_interval {
                    last_snapshot = Instant::now();
                    let current = state.current_state.read().unwrap().clone();
                    let msg = DashboardPush::Snapshot { state: Box::new(current) };
                    if let Ok(json) = serde_json::to_string(&msg) {
                        if sender.send(Message::Text(json)).await.is_err() {
                            break;
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    state.client_count.fetch_sub(1, Ordering::SeqCst);
    info!(
        client_id,
        remaining = state.client_count.load(Ordering::Relaxed),
        "Dashboard WS: client disconnected"
    );
}

// ============================================================================
// Throttled Update Helper
// ============================================================================

/// Helper for throttling dashboard updates
pub struct DashboardThrottle {
    last_push: Instant,
    throttle_duration: Duration,
}

impl DashboardThrottle {
    pub fn new(throttle_ms: u64) -> Self {
        Self {
            last_push: Instant::now() - Duration::from_millis(throttle_ms), // Allow immediate first push
            throttle_duration: Duration::from_millis(throttle_ms),
        }
    }

    /// Check if enough time has passed for another update
    pub fn should_push(&mut self) -> bool {
        if self.last_push.elapsed() >= self.throttle_duration {
            self.last_push = Instant::now();
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_serialization() {
        let push = DashboardPush::Snapshot {
            state: Box::new(DashboardState::default()),
        };
        let json = serde_json::to_string(&push).unwrap();
        assert!(json.contains("\"type\":\"snapshot\""));
    }

    #[test]
    fn test_update_serialization() {
        let push = DashboardPush::Update {
            quotes: None,
            pnl: None,
            regime: None,
            timestamp_ms: 12345,
        };
        let json = serde_json::to_string(&push).unwrap();
        assert!(json.contains("\"type\":\"update\""));
        assert!(json.contains("\"timestamp_ms\":12345"));
        // None fields should be skipped
        assert!(!json.contains("\"quotes\""));
    }

    #[test]
    fn test_command_deserialization() {
        let json = r#"{"type":"ping","id":42}"#;
        let cmd: DashboardCommand = serde_json::from_str(json).unwrap();
        assert!(matches!(cmd, DashboardCommand::Ping { id: 42 }));
    }

    #[test]
    fn test_throttle() {
        let mut throttle = DashboardThrottle::new(100);
        assert!(throttle.should_push()); // First always allowed
        assert!(!throttle.should_push()); // Too soon
        std::thread::sleep(Duration::from_millis(110));
        assert!(throttle.should_push()); // Enough time passed
    }
}
