//! WebSocket-enabled order executor with REST fallback.
//!
//! This module provides configuration for WS POST ordering. The actual WS POST
//! logic is now integrated directly into `ExchangeClient` via `enable_ws_post()`.
//!
//! # Usage
//!
//! ```ignore
//! // Enable WS POST on an existing ExchangeClient
//! let info_client = Arc::new(RwLock::new(info_client));
//! exchange_client.enable_ws_post(Arc::clone(&info_client), Some(Duration::from_secs(3)));
//!
//! // Now all orders placed via HyperliquidExecutor will use WS POST with REST fallback
//! let executor = HyperliquidExecutor::new(exchange_client, metrics);
//! ```
//!
//! # Architecture
//!
//! ```text
//! HyperliquidExecutor
//!     └── ExchangeClient
//!             ├── http_client (REST fallback)
//!             └── ws_info_client (WS POST when enabled)
//!                     └── ws_manager
//!                             └── pending_posts (request/response correlation)
//! ```

use std::time::Duration;

/// Configuration for WebSocket POST ordering.
///
/// These settings can be passed to `ExchangeClient::enable_ws_post()` for
/// configuring WS POST behavior.
#[derive(Debug, Clone)]
pub struct WsPostConfig {
    /// Whether to use WS POST for order operations (default: true)
    pub enabled: bool,
    /// Timeout for WS POST responses (default: 5s)
    pub timeout: Duration,
    /// Whether to fall back to REST on WS failure (default: true, always enabled)
    /// Note: REST fallback is automatic in ExchangeClient.
    pub fallback_to_rest: bool,
}

impl Default for WsPostConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            // FIX: Reduced from 1s to 500ms for faster placement latency
            // This reduces the ~600ms placement latency target to ~200ms
            // REST fallback kicks in quickly if WS fails
            timeout: Duration::from_millis(500),
            fallback_to_rest: true,
        }
    }
}

impl WsPostConfig {
    /// Create a new WS POST config with custom timeout.
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            enabled: true,
            timeout,
            fallback_to_rest: true,
        }
    }

    /// Create a fast WS POST config optimized for latency.
    /// Uses 500ms timeout with aggressive REST fallback.
    pub fn fast() -> Self {
        Self {
            enabled: true,
            timeout: Duration::from_millis(500),
            fallback_to_rest: true,
        }
    }

    /// Create a disabled WS POST config (REST only).
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            timeout: Duration::from_secs(5),
            fallback_to_rest: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ws_post_config_default() {
        let config = WsPostConfig::default();
        assert!(config.enabled);
        // FIX: Updated to match new 500ms timeout for faster placement
        assert_eq!(config.timeout, Duration::from_millis(500));
        assert!(config.fallback_to_rest);
    }

    #[test]
    fn test_ws_post_config_with_timeout() {
        let config = WsPostConfig::with_timeout(Duration::from_secs(3));
        assert!(config.enabled);
        assert_eq!(config.timeout, Duration::from_secs(3));
    }

    #[test]
    fn test_ws_post_config_disabled() {
        let config = WsPostConfig::disabled();
        assert!(!config.enabled);
    }
}
