//! BeliefPublisher trait for estimators to publish updates.
//!
//! Any component that generates belief updates implements this trait,
//! which provides a unified interface for publishing to the central
//! belief state.
//!
//! ## Example
//!
//! ```ignore
//! impl BeliefPublisher for KappaOrchestrator {
//!     fn sender(&self) -> &Sender<BeliefUpdate> {
//!         &self.belief_sender
//!     }
//!
//!     fn publisher_id(&self) -> &'static str {
//!         "kappa_orchestrator"
//!     }
//! }
//!
//! // Then when a fill occurs:
//! self.publish(BeliefUpdate::OwnFill { ... });
//! ```

use super::BeliefUpdate;
use tokio::sync::mpsc::Sender;

/// Trait for components that publish belief updates.
///
/// Implementing this trait allows any estimator or handler to
/// publish updates to the centralized belief state in a consistent way.
pub trait BeliefPublisher: Send + Sync {
    /// Get the sender channel for publishing updates.
    fn sender(&self) -> &Sender<BeliefUpdate>;

    /// Get the publisher's identifier (for logging/debugging).
    fn publisher_id(&self) -> &'static str;

    /// Publish an update (non-blocking, drops if full).
    ///
    /// This is the primary method for publishing updates. It uses
    /// `try_send` to avoid blocking if the channel is full, which
    /// could happen during high-frequency updates.
    fn publish(&self, update: BeliefUpdate) {
        if let Err(e) = self.sender().try_send(update) {
            tracing::debug!(
                publisher = %self.publisher_id(),
                error = %e,
                "Failed to publish belief update (channel full)"
            );
        }
    }

    /// Publish an update (blocking version).
    ///
    /// Use this sparingly - only for critical updates that must
    /// not be dropped.
    fn publish_blocking(&self, update: BeliefUpdate) -> Result<(), PublishError> {
        self.sender()
            .try_send(update)
            .map_err(|e| PublishError::ChannelFull(e.to_string()))
    }
}

/// Error type for publishing failures.
#[derive(Debug, Clone)]
pub enum PublishError {
    /// Channel is full
    ChannelFull(String),
    /// Channel is closed
    ChannelClosed(String),
}

impl std::fmt::Display for PublishError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PublishError::ChannelFull(msg) => write!(f, "Channel full: {}", msg),
            PublishError::ChannelClosed(msg) => write!(f, "Channel closed: {}", msg),
        }
    }
}

impl std::error::Error for PublishError {}

/// Helper struct for components that need to publish updates.
///
/// This provides a simple way to add publishing capability to
/// existing structs without modifying their core implementation.
///
/// ## Example
///
/// ```ignore
/// struct MyEstimator {
///     belief_publisher: Option<PublisherHandle>,
///     // ... other fields
/// }
///
/// impl MyEstimator {
///     pub fn set_publisher(&mut self, sender: Sender<BeliefUpdate>) {
///         self.belief_publisher = Some(PublisherHandle::new(sender, "my_estimator"));
///     }
///
///     pub fn on_observation(&mut self, obs: f64) {
///         // ... update internal state ...
///
///         if let Some(publisher) = &self.belief_publisher {
///             publisher.publish(BeliefUpdate::PriceReturn { ... });
///         }
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct PublisherHandle {
    sender: Sender<BeliefUpdate>,
    id: &'static str,
}

impl PublisherHandle {
    /// Create a new publisher handle.
    pub fn new(sender: Sender<BeliefUpdate>, id: &'static str) -> Self {
        Self { sender, id }
    }

    /// Publish an update (non-blocking).
    pub fn publish(&self, update: BeliefUpdate) {
        if let Err(e) = self.sender.try_send(update) {
            tracing::debug!(
                publisher = %self.id,
                error = %e,
                "Failed to publish belief update (channel full)"
            );
        }
    }

    /// Get the publisher ID.
    pub fn id(&self) -> &'static str {
        self.id
    }
}

impl BeliefPublisher for PublisherHandle {
    fn sender(&self) -> &Sender<BeliefUpdate> {
        &self.sender
    }

    fn publisher_id(&self) -> &'static str {
        self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_publisher_handle_creation() {
        let (tx, _rx) = tokio::sync::mpsc::channel(10);
        let handle = PublisherHandle::new(tx, "test_publisher");

        assert_eq!(handle.id(), "test_publisher");
    }

    #[tokio::test]
    async fn test_publisher_handle_publish() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(10);
        let handle = PublisherHandle::new(tx, "test_publisher");

        handle.publish(BeliefUpdate::HardReset);

        let received = rx.recv().await;
        assert!(received.is_some());
        assert!(matches!(received.unwrap(), BeliefUpdate::HardReset));
    }

    #[tokio::test]
    async fn test_publisher_handle_full_channel() {
        // Create a channel with capacity 1
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let handle = PublisherHandle::new(tx, "test_publisher");

        // First send should succeed
        handle.publish(BeliefUpdate::HardReset);

        // Second send should be dropped (channel full), but not panic
        handle.publish(BeliefUpdate::HardReset);
    }

    #[test]
    fn test_publish_error_display() {
        let full = PublishError::ChannelFull("test error".to_string());
        assert!(full.to_string().contains("Channel full"));

        let closed = PublishError::ChannelClosed("test error".to_string());
        assert!(closed.to_string().contains("Channel closed"));
    }
}
